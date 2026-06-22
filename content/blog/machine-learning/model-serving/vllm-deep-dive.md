---
title: "vLLM deep dive: Architecture, APIs, and production operations"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master vLLM from LLMEngine internals to EngineArgs tuning, speculative decoding, prefix caching, tensor parallelism, and quantization so you can build a production LLM service that saturates the GPU."
tags:
  [
    "model-serving",
    "inference",
    "vllm",
    "llm-serving",
    "pagedattention",
    "speculative-decoding",
    "tensor-parallelism",
    "quantization",
    "gpu-inference",
    "continuous-batching",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/vllm-deep-dive-1.png"
---

It was 2:47 AM and the on-call page read: "LLM gateway p99 latency 42 seconds, SLA is 5." The runbook said "restart the serving process." The engineer did. Latency spiked to 78 seconds as the KV cache warmed up, then gradually crept back down to 38 seconds. The model was Llama-3-70B running on two A100 80GB GPUs in the most naive configuration imaginable: `max_batch_size=1`, no tensor parallelism, `gpu_memory_utilization=0.95` with no understanding of what that number actually controlled.

The fix took four minutes once the right person woke up. Change `tensor_parallel_size=2`, set `gpu_memory_utilization=0.85` to leave room for activations, enable `enable_prefix_caching=True` because 80% of requests started with the same 2,048-token system prompt, and switch from the synchronous `LLM.generate()` to `AsyncLLMEngine` so the event loop could interleave I/O with inference. Throughput went from 180 tokens/s to 1,140 tokens/s. p99 latency dropped to 2.1 seconds. That morning's traffic peak passed without another page.

That's what understanding vLLM actually buys you. It's the dominant LLM serving framework for good reason — it exposes a clean API, supports every major quantization format, and ships with all the key optimizations out of the box. But "out of the box" doesn't mean "correctly configured by default." This post goes all the way in: the internal component architecture that determines why every parameter matters, every consequential `EngineArgs` option and its first-principles effect on the SLO triangle, the quantitative math behind speculative decoding and chunked prefill, and the operational patterns that separate a stable production deployment from a time bomb. By the end, you will be able to size a vLLM deployment for a target QPS, configure quantization to quadruple your KV-cache capacity, explain why speculative decoding hurts throughput at batch=32 even though it helps at batch=1, and instrument your serving stack so the next 2:47 AM page is the last one.

![vLLM component architecture showing LLMEngine, Scheduler, Workers, and BlockSpaceManager](/imgs/blogs/vllm-deep-dive-1.png)

The SLO triangle for LLM serving — **latency ↔ throughput ↔ cost** — is governed almost entirely by how efficiently you use GPU memory. vLLM's core innovation, PagedAttention, is fundamentally a memory allocator. Everything else (the scheduler, the workers, the parallelism strategies) is built to serve that allocator. Once you understand that hierarchy, every configuration knob makes sense.

This post is C3 in the Model Deployment and Serving series. If you need the foundational concepts — KV cache memory pressure, the autoregressive bottleneck, and why LLMs require fundamentally different serving infrastructure than classical models — see [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different). If you want the PagedAttention algorithm and continuous batching mechanics before diving into vLLM's APIs, see [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention).

## 1. The component architecture: how a request actually moves

vLLM's codebase is organized around five major components. Getting their relationships right is the prerequisite for all tuning decisions. A surprising number of production misconfigurations trace back to engineers treating vLLM as a black box and adjusting numbers until things "seemed fine." The architecture makes it clear why most default settings are wrong for high-throughput scenarios.

### LLMEngine: the control center

`LLMEngine` is the synchronous core of vLLM. Every request enters through it, and every token exits through it. When you call `llm.generate()`, you are ultimately calling `LLMEngine.generate()`. The engine owns three critical sub-components: a `Scheduler`, a `BlockSpaceManager`, and a set of `Worker` processes.

The engine runs a tight event loop. On each call to `engine.step()`, it:

1. Accepts new `SequenceGroup` objects from the request queue. A `SequenceGroup` is a bundle representing one request and all its output sequences (since you can request `n=3` completions in parallel).
2. Asks the `Scheduler` which sequence groups to advance this step — which ones to run, which ones to preempt, and which ones to swap from CPU back to GPU.
3. Asks the `BlockSpaceManager` to allocate physical KV-cache blocks for sequences being promoted from waiting to running, free blocks for finished sequences, and perform copy-on-write for sequences that are being forked (e.g., beam search with branching).
4. Assembles `ExecuteModelRequest` objects and sends them to each `Worker` via inter-process communication. Depending on the deployment backend, this is either Ray remote calls or PyTorch's `mp.Process` with shared memory.
5. Collects `SamplerOutput` from each Worker — the newly sampled token IDs for every sequence in the running batch.
6. Updates sequence states: appends the sampled token to each sequence's token list, checks stop conditions (end-of-sequence token, stop strings, `max_tokens` exhaustion), and marks finished sequences.
7. Returns finished sequences as `RequestOutput` objects to their respective callers.

This entire loop constitutes **one engine step**. The scheduler decides the batch composition; the block manager decides the memory layout; the workers execute the arithmetic. The critical insight is that all three decisions happen in lockstep — no asynchronous memory management, no speculative block allocation, just a synchronous round of "what should I run and where should I put its memory."

The step loop frequency determines throughput. If each step takes 20ms (a single decode step for a 70B model at batch=32), you're running 50 steps per second. Each step produces one token per running sequence, so at 32 concurrent sequences you get 50 × 32 = 1,600 output tokens/s. This is why `max_num_seqs` is a throughput multiplier up to the point where the batch becomes compute-bound.

### AsyncLLMEngine: the production wrapper

`AsyncLLMEngine` wraps `LLMEngine` in an asyncio event loop so that a FastAPI or aiohttp server can handle concurrent HTTP connections without blocking on inference steps. The key difference from the synchronous engine is in its execution model.

The `AsyncLLMEngine` runs the engine's step loop in a background asyncio task. When you call `engine.generate(prompt, sampling_params, request_id)`, you get back an async generator. Each `async for` iteration yields the current `RequestOutput` for your request — but "yields" here means "hands control back to the event loop," which lets the server process incoming connections, accept new requests, and send partial streaming responses to other clients.

This design is why you must use `AsyncLLMEngine` in any production serving scenario. With the synchronous `LLMEngine`, a request for 500 output tokens at 20ms/token takes 10 full seconds of blocking Python execution. During those 10 seconds, no other HTTP request can be accepted or processed. The effective concurrency of your server is one. With `AsyncLLMEngine`, those 10 seconds are 500 asyncio yields, each lasting 20ms — during each 20ms yield, the event loop can accept connections, process completions from other concurrently running requests, and send SSE chunks to streaming clients.

The `request_id` parameter in `engine.generate()` is your handle for cancellation. If a client disconnects before the generation finishes — which happens constantly in real production traffic — call `await engine.abort(request_id)` immediately. This frees the KV-cache blocks allocated to that request. Without explicit abortion, abandoned sequences continue consuming KV-cache blocks and compute until they hit `max_tokens`, which can take tens of seconds for long-context requests. On a server with 256 concurrent sequences, a 10% client disconnect rate without cleanup means up to 25 ghost sequences consuming ~10% of your KV cache at all times.

### Scheduler: continuous batching in practice

The `Scheduler` implements vLLM's continuous batching policy, which is the key operational difference from static batching systems. In static batching, the server waits until a batch is full, then runs inference synchronously, and only accepts new requests after the batch finishes. In continuous batching, new requests join the running pool after every step — a request that finishes at step 400 immediately frees space for a waiting request to join at step 401, without waiting for the other sequences in the batch to finish.

At each step, the scheduler inspects three queues:

- `waiting`: requests that have arrived but not yet been assigned any KV-cache blocks.
- `running`: sequences that have been allocated blocks and are actively being decoded.
- `swapped`: sequences whose KV-cache blocks have been evicted to CPU RAM due to memory pressure.

The scheduler's decision procedure is approximately:

1. For each sequence in `running`, extend its KV-cache by one block if necessary (when the current position hits a block boundary).
2. Attempt to swap back sequences from `swapped` that can be re-admitted without causing further preemption.
3. Admit new sequences from `waiting` as long as: (a) the total running sequences are below `max_num_seqs`, (b) the total tokens being processed (running sequences × 1 decode token + newly admitted prefill tokens) are below `max_num_batched_tokens`, and (c) the `BlockSpaceManager` has enough free blocks.
4. If GPU memory pressure forces it, preempt lowest-priority running sequences by evicting their KV-cache blocks to CPU RAM.

Preemption is expensive — it involves copying KV-cache blocks from GPU HBM to system DRAM over PCIe, then copying them back when the sequence is re-admitted. For a sequence with 4K tokens of KV cache at BF16, each block copy is about 4 MB of data over PCIe. A PCIe 4.0 x16 link does ~28 GB/s, so copying 4 MB takes about 143 µs. That's tolerable, but if you're preempting 50 sequences simultaneously, that's 7 ms of PCIe transfer time during which GPU utilization drops.

A well-tuned deployment shows essentially zero preemption events. If your logs show frequent preemption (the metric is `vllm:num_preemptions_total`), the most common causes are: `gpu_memory_utilization` set too high (not enough headroom for KV allocation), `max_num_seqs` too high relative to average sequence length, or a workload where a small number of very long sequences starve short ones.

### BlockSpaceManager: the KV cache allocator

The `BlockSpaceManager` is where PagedAttention lives, and understanding it is essential for reasoning about memory configuration. It divides the GPU's reserved KV-cache memory into fixed-size **blocks** of 16 token positions each. Each block stores the K and V tensors for all attention layers for 16 tokens.

The memory math for one block:

$$\text{block\_size} = 2 \times 16 \times n\_layers \times n\_kv\_heads \times head\_dim \times \text{dtype\_size}$$

For Llama-3-8B (32 layers, 8 KV heads, 128 head dim, BF16):

$$\text{block\_size} = 2 \times 16 \times 32 \times 8 \times 128 \times 2 = 2,097,152 \text{ bytes} \approx 2 \text{ MB}$$

Each block holds 16 tokens worth of KV across all layers. A sequence of 1024 tokens requires $\lceil 1024/16 \rceil = 64$ blocks = 128 MB of KV cache.

For each running sequence, the `BlockSpaceManager` maintains a **block table**: a list of physical block addresses corresponding to logical positions 0–15 → block A, 16–31 → block B, and so on. These physical addresses are non-contiguous in GPU memory — block A might be at address 0x0000, block B at address 0x3000, block C at 0x1000. The PagedAttention CUDA kernel reads this mapping and performs scatter-gather reads to access K and V tensors from their physical locations.

This is the fundamental difference from naive KV-cache allocation, which pre-reserves a maximum-length contiguous region for each sequence. In naive allocation, a sequence that might grow to 4096 tokens needs 4096 tokens of contiguous GPU memory from the start — even if it only produces 100 tokens. With paged allocation, you allocate one 16-token block at a time, wherever free physical blocks exist. Memory waste drops from 40–60% (naive) to under 4% (paged).

![PagedAttention eliminates KV-cache fragmentation by allocating non-contiguous 16-token blocks](/imgs/blogs/vllm-deep-dive-2.png)

For prefix caching (Section 8), the BlockSpaceManager extends this scheme with a **prefix block cache**: a hash map from (hash of token IDs in a block) → physical block address. When two sequences share a prefix, they share physical blocks. The shared blocks are reference-counted and freed only when all sequences referencing them complete. This is the mechanism that makes system-prompt sharing essentially free — the 2,048-token system prompt is hashed into 128 blocks, stored once in GPU memory, and referenced by every concurrent request.

### Worker: one GPU, one model shard

Each `Worker` runs in its own OS process and owns exactly one GPU. It loads its shard of the model weights, maintains its own copy of the KV-cache block allocations (the actual HBM memory), and participates in collective communication operations.

Workers receive `ExecuteModelRequest` objects from the engine. Each request contains the block tables for all sequences in the current batch, the token IDs to process (the new prompt tokens for prefill sequences, or a single token for decode sequences), and metadata for the sampler. The worker runs `model.forward()` with these inputs, producing logits. The sampler converts logits to token IDs according to the `SamplingParams` for each sequence. The worker returns a `SamplerOutput` containing the new token IDs and any requested log probabilities.

The attention computation in each Worker uses vLLM's custom PagedAttention CUDA kernels. These kernels are the hot path — they perform a gather from the non-contiguous physical block addresses specified by the block table, compute attention scores, and write the new K and V tensors into the appropriate block. The kernel is optimized for the block-table scatter-gather pattern, with cache-line-aligned block sizes and prefetching to hide HBM latency.

For tensor-parallel deployments, the Workers on different GPUs coordinate via NCCL collective operations. After each attention layer's output projection and each FFN's second linear, all Workers call `allreduce` to sum their partial results. The `allreduce` is implemented over NVLink for single-node TP or NVLink/InfiniBand for multi-node, whichever is available.

## 2. The LLM class: offline inference

The `LLM` class is the right tool for batch processing of fixed datasets. It wraps `LLMEngine` with a simple synchronous interface that takes a list of prompts and returns a list of `RequestOutput` objects when all prompts have finished generating.

```python
from vllm import LLM, SamplingParams

# Instantiate the engine — this loads model weights and profiles GPU memory
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    dtype="bfloat16",
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=512,
    stop=["</s>", "<|eot_id|>"],
    presence_penalty=0.1,
)

# Batch inference — vLLM schedules all prompts together
prompts = [
    "Explain gradient descent in two sentences.",
    "What is the difference between TCP and UDP?",
    "Write a Python function to compute Fibonacci numbers.",
    "Summarize the PagedAttention algorithm in one paragraph.",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    finish_reason = output.outputs[0].finish_reason
    num_tokens = len(output.outputs[0].token_ids)
    print(f"Prompt: {prompt[:50]!r}...")
    print(f"Generated {num_tokens} tokens, finish: {finish_reason}")
    print(f"Output: {generated_text[:200]!r}")
    print()
```

`llm.generate()` submits all prompts to the scheduler simultaneously. They are batched together on each step according to the continuous batching policy — short prompts that finish early free up KV-cache blocks for longer ones that are still running. The return order matches the input order regardless of completion order.

### SamplingParams in depth

`SamplingParams` is the per-request configuration for token selection. Each parameter controls a specific aspect of the sampling process:

- **`temperature`**: the scale factor applied to logits before softmax. Temperature 0 gives greedy decoding (always select the highest-probability token). Temperature 1.0 uses the raw model probabilities. Temperature > 1.0 makes the distribution more uniform (more random); temperature < 1.0 makes it more peaked (less random). For factual Q&A or summarization, use 0.1–0.3. For creative writing, use 0.7–1.0.

- **`top_p`**: nucleus sampling threshold. After applying temperature, sort tokens by probability (highest first) and keep the smallest prefix whose cumulative probability exceeds `top_p`. Sample from this nucleus. This prevents sampling from very long tails of improbable tokens. `top_p=0.9` is a common default; `top_p=1.0` disables nucleus sampling.

- **`top_k`**: keep only the `k` highest-probability tokens before applying nucleus sampling. `top_k=50` combined with `top_p=0.9` means: first take the top 50 tokens, then apply nucleus sampling within those 50. Setting `top_k=-1` disables it.

- **`max_tokens`**: the hard ceiling on new tokens generated. vLLM sets the default to the model's maximum context length minus the prompt length. For throughput-sensitive workloads, always set this explicitly to prevent a handful of very long sequences from monopolizing the KV cache.

- **`stop`**: a list of strings that trigger immediate generation termination when any of them appears in the output. The stop string is not included in `outputs[0].text` by default. Common stop tokens: `["</s>", "<|eot_id|>", "Human:", "Assistant:"]`.

- **`presence_penalty`**: a scalar added to the logit of any token that has appeared in the output at least once. Positive values discourage repetition. Values in 0.1–0.3 are gentle nudges; above 1.0 becomes aggressive.

- **`frequency_penalty`**: a scalar multiplied by the number of times a token has appeared in the output, then subtracted from its logit. Stronger anti-repetition effect than presence penalty for tokens that repeat many times.

- **`n`**: the number of independent output sequences to generate per prompt. Uses beam-search-style KV sharing for the common prefix, then branches. Memory cost scales with `n`.

- **`best_of`**: generate `best_of` sequences and return only the top `n` by log probability. `best_of=5, n=1` generates 5 completions and returns the highest-likelihood one.

- **`logprobs`**: if set to an integer N, return the top-N log probabilities at each generated token position. Useful for uncertainty estimation, but increases memory usage and serialization cost.

### When to use offline vs online serving

Use the `LLM` class when:
- Processing a fixed evaluation dataset (MMLU, HellaSwag, your internal test set).
- Running batch annotation for fine-tuning data generation.
- Doing throughput benchmarking to measure max tokens/s.
- Prototyping a pipeline before building the serving layer.

Use `AsyncLLMEngine` when:
- Serving live traffic with variable inter-arrival times.
- Implementing streaming responses (SSE/WebSocket).
- Integrating with a web framework (FastAPI, aiohttp).
- Running any workload where the server needs to accept requests while inference is in progress.

Never use `LLM.generate()` behind a web server. Each call blocks the Python process until all prompts complete. At 100ms TPOT and 200 output tokens, that's 20 seconds of blocked execution per call — catastrophic for any concurrent serving scenario.

## 3. AsyncLLMEngine: streaming for production

`AsyncLLMEngine` is the right engine for all production serving. Here is a complete, production-grade streaming implementation that mirrors what the built-in API server uses internally:

```python
import asyncio
import uuid
from typing import AsyncIterator
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

# ── Engine initialization ─────────────────────────────────────────────────
engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    dtype="bfloat16",
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

# ── Streaming generator ───────────────────────────────────────────────────
async def stream_tokens(prompt: str) -> AsyncIterator[str]:
    """Yield token deltas as they are generated."""
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["<|eot_id|>"],
    )
    request_id = str(uuid.uuid4())
    
    previous_text = ""
    try:
        async for request_output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            output = request_output.outputs[0]
            # outputs[0].text is CUMULATIVE — compute the delta
            delta = output.text[len(previous_text):]
            previous_text = output.text
            
            if delta:
                yield delta
            
            if request_output.finished:
                # finish_reason: "stop" | "length" | "abort"
                break
    except asyncio.CancelledError:
        # Client disconnected — abort the generation to free KV cache
        await engine.abort(request_id)
        raise

# ── Integration with FastAPI SSE ──────────────────────────────────────────
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
#
# app = FastAPI()
#
# @app.post("/generate")
# async def generate(request: GenerateRequest):
#     async def event_stream():
#         async for token in stream_tokens(request.prompt):
#             yield f"data: {token}\n\n"
#         yield "data: [DONE]\n\n"
#     return StreamingResponse(event_stream(), media_type="text/event-stream")
```

The streaming loop pattern has several subtleties that trip up engineers new to vLLM:

**The text is cumulative**: `outputs[0].text` grows with each yield — it is always the full generated text so far, not just the new token. You must compute the delta yourself by tracking the previous length. This is a deliberate design choice: it makes it easy for non-streaming clients (those that wait for `request_output.finished`) to get the full text without string concatenation.

**Cancellation semantics**: if you use `asyncio.CancelledError` to handle client disconnects (which FastAPI/Starlette will propagate), you must catch it and call `engine.abort(request_id)`. Without this, the KV-cache blocks remain allocated until the sequence hits `max_tokens`.

**Multiple outputs**: if `SamplingParams.n > 1`, `request_output.outputs` contains multiple `CompletionOutput` objects, one per output sequence. Each has its own `.text` and `.finish_reason`. The streaming pattern above only handles `n=1`.

**Token IDs vs text**: `outputs[0].token_ids` gives you the raw token ID list if you need it (for log-probability analysis, for passing tokens to another model without tokenization overhead, etc.).

## 4. EngineArgs: the complete configuration reference

`EngineArgs` (and its async sibling `AsyncEngineArgs`) is the single source of truth for vLLM configuration. Every CLI flag in `python -m vllm.entrypoints.openai.api_server` maps to a field in `EngineArgs`. Understanding the parameter space systematically is more valuable than memorizing individual flags.

![EngineArgs parameter layers: model identity, memory budget, parallelism, and optimization flags](/imgs/blogs/vllm-deep-dive-3.png)

Here is a production-grade configuration with full annotations:

```python
from vllm.engine.arg_utils import EngineArgs

args = EngineArgs(
    # ── Model identity ──────────────────────────────────────────────────────
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    # HuggingFace repo, local path, or s3://bucket/path
    
    tokenizer="meta-llama/Meta-Llama-3-70B-Instruct",
    # Defaults to model if not set. Override if tokenizer is elsewhere.
    
    dtype="bfloat16",
    # "auto": use the model's preferred dtype (from config.json)
    # "float16": half-precision, good for older GPUs without BF16 support
    # "bfloat16": preferred for modern GPUs (A100/H100); better numerics
    # "float8_e4m3fn": FP8 quantization (H100 only)
    
    quantization="awq",
    # None: no quantization, load as specified by dtype
    # "gptq": GPTQ INT4/INT8 (pre-quantized model required)
    # "awq": AWQ INT4 (pre-quantized model required)
    # "fp8": FP8 weights on H100 (can auto-quantize at load time)
    # "squeezellm": SqueezeLLM non-uniform quantization
    
    max_model_len=32768,
    # Maximum sequence length (prompt + output). Reducing this below the
    # model's native context window directly reduces KV-cache consumption.
    # For Llama-3 8B (128K native), setting max_model_len=8192 frees 93.75%
    # of the KV cache that would otherwise be reserved for context extension.
    
    # ── Memory budget ───────────────────────────────────────────────────────
    gpu_memory_utilization=0.88,
    # Fraction of GPU VRAM allocated for weights + KV cache.
    # The actual KV cache gets: (total_vram × utilization) - weights - activations
    # Set lower (0.80) if you see OOMs during warm-up; set higher (0.92)
    # only on very large VRAM GPUs with well-characterized workloads.
    
    swap_space=8,
    # GB of CPU RAM reserved for preempted sequences.
    # Preemption is a last resort; swap_space is the safety valve.
    # 8 GB accommodates ~64 sequences with 4K tokens of KV cache each.
    
    kv_cache_dtype="auto",
    # "auto": match the model dtype (BF16 → BF16 KV cache)
    # "fp8_e5m2": store KV cache in FP8 to halve KV memory at slight accuracy cost
    # This is independent of the weight dtype.
    
    # ── Parallelism ─────────────────────────────────────────────────────────
    tensor_parallel_size=2,
    # Number of GPUs for tensor parallelism within a single model replica.
    # Must divide evenly into the number of attention heads.
    # All ranks must be on the same physical node with NVLink.
    
    pipeline_parallel_size=1,
    # Pipeline parallelism stages. Still experimental for serving.
    # PP=2 means each node runs half the layers; activations are passed
    # across nodes via point-to-point NCCL. Avoid unless model doesn't fit
    # in any other TP configuration.
    
    # ── Scheduler controls ──────────────────────────────────────────────────
    max_num_seqs=256,
    # Maximum concurrent sequences. Higher = more throughput up to GPU saturation.
    # Reduce if you see frequent preemption or memory pressure.
    
    max_num_batched_tokens=8192,
    # Maximum total tokens per step (prefill + decode).
    # For H100: 32768 is reasonable. For A100 40GB: 8192-16384.
    # Also controls chunk size when enable_chunked_prefill=True.
    
    # ── Optimization flags ──────────────────────────────────────────────────
    enable_prefix_caching=True,
    # Hash-keyed shared KV blocks for repeated prefixes.
    # Near-zero cost when hit rate is low; huge win when high.
    
    enable_chunked_prefill=True,
    # Break long prefills into max_num_batched_tokens chunks.
    # Strongly recommended for mixed-length workloads.
    
    # ── Speculative decoding (commented out as it requires a draft model) ──
    # speculative_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # num_speculative_tokens=5,
    # speculative_draft_tensor_parallel_size=1,
    
    # ── Miscellaneous ────────────────────────────────────────────────────────
    trust_remote_code=False,
    # Set True only for custom models with non-standard architectures.
    
    seed=42,
    # Random seed for sampling reproducibility.
    
    disable_log_requests=False,
    # Set True in very high-QPS environments to reduce log I/O.
)
```

### The gpu_memory_utilization math, precisely

This is the most commonly misconfigured parameter. Many engineers assume `gpu_memory_utilization=0.90` means "use at most 90% of GPU VRAM." That is incorrect.

What vLLM actually does during startup:

1. Loads model weights into GPU memory. For Llama-3-70B in BF16, this is approximately 140 GB.
2. Runs a dummy forward pass with `max_model_len` tokens to measure peak activation memory. For the same model, this is typically 3–6 GB depending on batch size.
3. Computes available KV cache memory:

$$\text{KV cache bytes} = (\text{total VRAM} \times \text{gpu\_memory\_utilization}) - \text{weights} - \text{activations peak}$$

4. Divides the KV cache bytes into blocks of 16 tokens each.

For Llama-3-70B BF16 on 2×A100 80GB (160 GB total), with `gpu_memory_utilization=0.90`:

$$\text{KV cache} = (160 \times 0.90) - 140 - 5 = 144 - 145 = -1 \text{ GB}$$

Negative — this configuration doesn't work. You cannot serve 70B BF16 on 2×A100 80GB at all. You need 4×A100 80GB (320 GB) or quantization.

With AWQ INT4 on 2×A100 80GB, weights drop to ~35 GB:

$$\text{KV cache} = (160 \times 0.90) - 35 - 5 = 104 \text{ GB}$$

That 104 GB buys you an enormous number of KV cache blocks. For one Llama-3-70B attention layer (GQA: 8 KV heads, 128 dim), per-block memory:

$$\text{block} = 2 \times 16 \times 80 \times 8 \times 128 \times 2 = 52,428,800 \text{ bytes} \approx 50 \text{ MB}$$

Total blocks: $104,000 / 50 \approx 2,080$ blocks. Each block holds 16 tokens. Total KV cache capacity: $2,080 \times 16 = 33,280$ tokens of KV. At `max_model_len=8192`, that supports $33,280 / 8192 \approx 4$ simultaneous maximum-length sequences, or proportionally more at shorter contexts (e.g., 33 sequences at 1K context).

This is why `max_model_len` matters even when your average request is short. If you set `max_model_len=128K` for a model with 128K native context, vLLM reserves enough block capacity to theoretically support that. Reducing it to `max_model_len=8192` multiplies your KV cache block count by 16 relative to the 128K case.

## 5. The OpenAI-compatible API server

The built-in OpenAI-compatible server is the fastest path to a deployable vLLM service. It runs FastAPI on Uvicorn, exposes the standard OpenAI endpoints, and internally uses `AsyncLLMEngine` for all request handling.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --port 8000 \
    --api-key your-secret-key-here \
    --served-model-name llama-3-8b \
    --uvicorn-log-level warning
```

The server exposes three primary endpoints:

**`GET /v1/models`**: returns a JSON list of available models. The model ID is `--served-model-name`. Useful for discovery and health verification.

**`POST /v1/completions`**: the legacy prompt-completion API. Takes a `prompt` string (or list of strings), `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, and other parameters. Returns `CompletionResponse` with `choices[0].text`.

**`POST /v1/chat/completions`**: the chat API. Takes a `messages` array (each with `role` and `content`), applies the model's chat template, and runs generation. Supports `stream=true` for SSE-streamed responses. This is the API you want for any instruction-tuned model.

**`GET /health`**: returns HTTP 200 when the engine is initialized and serving. Returns 503 during startup. Use this for Kubernetes readiness probes.

**`GET /metrics`**: Prometheus-formatted metrics. Scraped by your monitoring stack.

Because the API surface is fully OpenAI-compatible, existing client code points to your vLLM server with a single URL change:

```python
from openai import OpenAI

# Point the OpenAI client at your vLLM server
client = OpenAI(
    base_url="http://your-vllm-host:8000/v1",
    api_key="your-secret-key-here",
)

# Non-streaming chat completion
response = client.chat.completions.create(
    model="llama-3-8b",   # matches --served-model-name
    messages=[
        {
            "role": "system",
            "content": "You are an expert ML systems engineer."
        },
        {
            "role": "user",
            "content": "Explain how vLLM's continuous batching scheduler works."
        },
    ],
    max_tokens=800,
    temperature=0.2,
)
print(response.choices[0].message.content)

# Streaming chat completion
stream = client.chat.completions.create(
    model="llama-3-8b",
    messages=[
        {"role": "user", "content": "List the 5 most important vLLM config knobs."},
    ],
    max_tokens=400,
    temperature=0.5,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content is not None:
        print(delta.content, end="", flush=True)
print()
```

### The --api-key flag

The `--api-key` flag enables simple bearer token authentication. Any request must include `Authorization: Bearer <your-key>`. Without it, the request gets a 401. This is not a production-grade authentication system — use it to prevent accidental open access while you add a real API gateway. In production, put Nginx or an API gateway (Kong, Traefik, AWS API Gateway) in front that handles:

- TLS termination
- Rate limiting per client
- Request logging with client identity
- Token counting for billing
- Circuit breaking for downstream failures

## 6. Quantization in vLLM

Quantization is the single highest-leverage knob for LLM serving cost and throughput. Its benefit operates through two orthogonal mechanisms:

1. **Reduced weight memory**: smaller weights → more GPU VRAM left for KV cache → larger concurrent batch sizes → higher throughput.
2. **Reduced compute for some formats**: FP8 uses hardware tensor cores that run at 2× the TFLOPS of BF16 on H100, reducing per-step compute time at large batch sizes.

![Quantization modes: GPTQ INT4, AWQ INT4, FP8, and BF16 baseline compared on weight size, KV headroom, hardware requirements, and accuracy](/imgs/blogs/vllm-deep-dive-5.png)

### Loading a GPTQ model

GPTQ is the most widely available format. Pre-quantized GPTQ models for virtually every major model family are available on HuggingFace under the `TheBloke` and `bartowski` namespaces, among others.

```python
from vllm import LLM

# Load Llama-3-70B in 4-bit GPTQ
llm = LLM(
    model="TheBloke/Llama-2-70B-GPTQ",
    quantization="gptq",
    dtype="float16",           # GPTQ dequantizes to float16 for compute
    gpu_memory_utilization=0.90,
    tensor_parallel_size=2,    # 70B GPTQ ≈35 GB fits on 2×A100 40GB
)
```

GPTQ performs **weight-only quantization**: it stores weights in INT4 (or INT8 for higher accuracy), but dequantizes them on-the-fly to FP16 for each matrix multiplication. Compute happens in FP16. The memory savings are entirely in weight storage: the 70B parameter model in BF16 needs ~140 GB; in GPTQ INT4 it needs ~35 GB. The compute time is identical to BF16 after dequantization because the actual matmuls run at FP16 speed — you're not getting FP8 compute acceleration, just weight compression.

The GPTQ algorithm works by finding a symmetric INT4 quantization grid that minimizes the reconstruction error of the model's outputs on a small calibration dataset. The quantization error is concentrated in the least important weight components (identified via the inverse Hessian), so accuracy degradation is modest — typically 0.3–0.8% on downstream task benchmarks for well-tuned 4-bit GPTQ.

### Loading an AWQ model

AWQ (Activation-aware Weight Quantization) is generally more accurate than GPTQ at 4-bit, especially for instruction-following tasks, because it identifies and preserves "salient" weights — those that multiply inputs with unusually large activation magnitudes. Small errors in salient weights propagate to large output errors; by keeping salient weights at higher precision (or scaling them before quantization), AWQ achieves better accuracy per bit.

```python
from vllm import LLM

# Load Llama-3-8B in 4-bit AWQ
llm = LLM(
    model="casperhansen/llama-3-8b-instruct-awq",
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.90,
)
```

AWQ models are slightly slower to generate than GPTQ models at the same batch size because the saliency-based weight ordering requires a different dequantization kernel. At batch=1, this overhead is negligible; at large batch sizes it can add 5–10% to step latency.

### FP8 on H100

H100 GPUs introduce native FP8 tensor cores using the `e4m3fn` format (4 exponent bits, 3 mantissa bits, no NaN). FP8 provides two distinct benefits over BF16:

1. **Half the memory**: FP8 weights occupy 1 byte each vs 2 bytes for BF16. A 70B-parameter model needs ~70 GB in FP8 vs ~140 GB in BF16.
2. **Double the compute throughput**: H100 FP8 tensor cores deliver up to 1,979 TFLOPS vs 989 TFLOPS for BF16 (with sparsity). Even without sparsity, FP8 is ~2× faster for matrix multiplications.

```python
from vllm import LLM

# FP8 on H100 — native hardware quantization
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    dtype="float8_e4m3fn",
    gpu_memory_utilization=0.90,
    tensor_parallel_size=2,    # 70B FP8 ≈70 GB fits on 2×H100 80GB
)
```

FP8 quantizes both weights and activations (unlike GPTQ/AWQ which quantize weights only). The activations are quantized per-tensor or per-token using dynamic scaling factors. This means FP8 requires hardware that can execute FP8 matmuls natively — currently only H100 and newer. On older GPUs, vLLM will either fall back to BF16 or raise an error.

The accuracy loss from FP8 is typically below 0.1% on MMLU and similar benchmarks, making it the best quantization option when you have H100s.

### The KV cache capacity derivation

Here's the precise math showing why GPTQ/AWQ INT4 more than quadruples effective serving capacity, not just memory savings. Consider 2×A100 80GB (160 GB total) serving Llama-3-70B with `max_model_len=8192`:

**BF16 baseline:**
- Weights: 70B × 2 bytes = 140 GB
- Activations peak: ~5 GB  
- KV cache available: $(160 × 0.90) - 140 - 5 = -1$ GB — doesn't fit

**GPTQ INT4 on same hardware:**
- Weights: 70B × 0.5 bytes = 35 GB
- Activations peak: ~5 GB  
- KV cache available: $(160 × 0.90) - 35 - 5 = 104$ GB

That 104 GB of KV cache translates directly to concurrent capacity. At `max_model_len=8192` and 2 MB per block (calculation from Section 1):

- Blocks: $104,000 / 2 = 52,000$ blocks
- Tokens: $52,000 \times 16 = 832,000$ total token capacity
- At 1K average context: $832,000 / 1000 = 832$ concurrent sequences

vs. BF16 where the model doesn't fit at all. This is not a marginal improvement — it's the difference between the model being serveable and not serveable.

## 7. Speculative decoding: the math and the configuration

Speculative decoding addresses the fundamental inefficiency of autoregressive decoding at small batch sizes. When batch size is 1 or 2, each decode step processes one or two tokens but must still load all model weights from HBM into registers. A 70B BF16 model has 140 GB of weights. At 2 TB/s HBM bandwidth, loading all weights takes 70ms — but the actual matmul arithmetic for a single token at 70B operations takes only 3ms. The GPU spends 95% of its time waiting for weight data. This is the memory-bandwidth bottleneck.

![Speculative decoding step sequence: draft proposes 5 tokens, target verifies all in one pass](/imgs/blogs/vllm-deep-dive-4.png)

Speculative decoding exploits this by using a small *draft model* to propose K candidate tokens, then using the large *target model* to verify them all in a single forward pass. The key insight is that the target model verifying K tokens costs only modestly more than verifying 1 token (the matmul cost scales with batch dimension, which stays 1; the sequence length increases by K, adding a small overhead). The draft model, being small (100M–1.1B parameters), costs roughly K/S of a single target model step, where S is the speed ratio.

### The speedup formula

The theoretical speedup of speculative decoding with K draft tokens and per-token acceptance rate α is:

$$\text{speedup} \approx \frac{1 - \alpha^{K+1}}{(1-\alpha)\left(1 + \frac{K}{S}\right)}$$

Where:
- $\alpha$ is the probability that each draft token matches the target model's sampling distribution.
- $K$ is the number of draft tokens proposed per step.
- $S$ is the target model's steps per second divided by the draft model's steps per second.

The formula has a natural interpretation. The numerator $1 - \alpha^{K+1}$ is the expected number of tokens accepted per step (by the geometric series for accepted tokens). The denominator $(1-\alpha)(1 + K/S)$ is the cost per step normalized to one target model step.

Let's build intuition for the formula with a few concrete evaluations:

At $\alpha=0.8$, $K=5$, $S=8$ (a 7B target with a 110M draft at 10ms vs 1.25ms per step):

$$\text{speedup} = \frac{1 - 0.8^6}{0.2 \times (1 + 5/8)} = \frac{0.738}{0.325} \approx 2.3\times$$

At $\alpha=0.5$ (poor draft model for the task), same K and S:

$$\text{speedup} = \frac{1 - 0.5^6}{0.5 \times (1 + 5/8)} = \frac{0.984}{0.8125} \approx 1.21\times$$

At $\alpha=0.5$ and large batch size where S shrinks to 2 (the large model runs faster because it's now compute-bound with the large batch):

$$\text{speedup} = \frac{0.984}{0.5 \times (1 + 5/2)} = \frac{0.984}{1.75} \approx 0.56\times$$

**Below 1.0 — speculative decoding made things worse.** This is the critical regime to avoid.

The speedup formula tells us precisely when speculative decoding helps and hurts:
- When $S$ is large (small batch, memory-bandwidth-bound): speculative decoding helps significantly.
- When $S$ is small (large batch, compute-bound): speculative decoding hurts because the draft overhead is a large fraction of the total cost.
- The breakeven $S$ is approximately $K/\alpha$ — speculative decoding is beneficial when the draft is at least $K/\alpha$ times cheaper than the target.

### Configuring speculative decoding in vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --port 8000
```

Or in Python via `EngineArgs`:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_speculative_tokens=5,
    speculative_draft_tensor_parallel_size=1,
    gpu_memory_utilization=0.88,
)
```

The draft and target models must share the same vocabulary tokenizer. Most Llama-family models are compatible with TinyLlama (which uses the Llama-2 tokenizer) or with smaller Llama-3 derivatives (which use the Llama-3 tokenizer). Mismatched tokenizers produce garbage results — vLLM does not validate this at startup.

The `--speculative-draft-tensor-parallel-size` flag controls how many GPUs the draft model uses. For a 1.1B draft model, `1` is usually correct — the draft model is so small that it fits on one GPU even if the target model is sharded across many.

#### Worked example: speculative decoding for a coding assistant

You're running a code completion service backed by Llama-3-8B-Instruct on a single A100 40GB GPU. Traffic pattern: 15 concurrent users, each making independent requests at ~3 second intervals. This gives an effective batch size of $15 / 3 = 5$ concurrent sequences per step. Request characteristics: median input 300 tokens (docstring + partial function), median output 150 tokens (function body), high predictability (code has narrow next-token distributions).

Current performance without speculative decoding:
- Step time: ~12ms (one decode step at batch≈5 on A100 40GB with 8B model)
- TPOT: 12ms / 5 × 1 = 12ms per token per sequence (since each sequence gets one token per step)
- p50 total latency: TTFT + 150 × 12ms = 80ms + 1,800ms = 1.88 seconds

Measured acceptance rate for code completion: $\alpha \approx 0.83$ (code is highly predictable; the draft model gets most tokens right).

With $K=5$, $S \approx 12$ (8B model at 12ms/step vs TinyLlama at ~1ms/step):

$$\text{speedup} = \frac{1 - 0.83^6}{0.17 \times (1 + 5/12)} = \frac{1 - 0.327}{0.17 \times 1.417} = \frac{0.673}{0.241} \approx 2.8\times$$

New effective TPOT: $12ms / 2.8 = 4.3ms$.
New p50 total: $80ms + 150 × 4.3ms = 80ms + 645ms = 725ms$ — a 2.6× reduction in total latency.

The draft model (TinyLlama 1.1B) needs ~2.2 GB of additional weight memory on the GPU. At `gpu_memory_utilization=0.88` and ~30 GB for the 8B model weights, there is about 5 GB remaining before hitting the utilization limit, so this fits comfortably.

## 8. Prefix caching: configuration and measurement

Prefix caching (also called KV prefix sharing) is one of the highest-ROI vLLM features for workloads where requests share a common prefix. The idea is simple: if multiple requests start with the same prompt prefix, compute the KV cache for that prefix once and share it across all requests. No redundant computation, no duplicated memory.

### How it works mechanically

When `enable_prefix_caching=True`, the `BlockSpaceManager` maintains a prefix cache: a hash table keyed by the hash of each 16-token block's token IDs. When a new sequence is admitted:

1. vLLM hashes the first 16 tokens of the prompt. If that hash is in the cache, the physical block at that address is reused (reference count incremented) and marked as immutable.
2. The process continues block by block until either (a) the hash doesn't match (the prefix diverges), or (b) the full prompt has been matched.
3. Only the un-cached suffix of the prompt needs to run the prefill computation.

The cache eviction policy is LRU (least-recently-used). When the cache is full and a new block needs to be cached, the least-recently-used unreferenced block is evicted.

### Enabling and measuring prefix caching

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-prefix-caching \
    --port 8000
```

When prefix caching is active, vLLM logs cache hit statistics approximately once per minute:

```
INFO 06-22 14:31:02 metrics.py:344] GPU KV cache usage: 74.2%, hit rate: 82.1%
INFO 06-22 14:32:02 metrics.py:344] GPU KV cache usage: 77.8%, hit rate: 79.6%
```

An 82% cache hit rate means 82% of prompt tokens were served from cache. For a workload where prefill is the dominant cost (long system prompts, many-shot examples), this directly reduces GPU compute by 82% for those tokens — the prefill step for a 2048-token cached prefix takes microseconds instead of hundreds of milliseconds.

You can also scrape the Prometheus metrics endpoint (`/metrics`) for programmatic monitoring:

```bash
curl -s http://localhost:8000/metrics | grep cache_hit_rate
# vllm:gpu_cache_hit_rate 0.821
```

### When prefix caching pays off

The ROI on prefix caching depends on hit rate and prefix length. A rough threshold: if your average cache hit rate exceeds 30% and your average cached prefix length exceeds 512 tokens, prefix caching is beneficial.

High-ROI scenarios:
- **Fixed system prompt**: a 2,048-token agent persona/tool list/safety policy prepended to every request. After the first request, every subsequent request pays zero prefill cost for those tokens.
- **RAG with a recurring document corpus**: if 100 users query against the same 50 documents, each document is computed once. For a typical RAG setup with 2–4 documents per request at 1,000 tokens each, the hit rate can exceed 70%.
- **Multi-turn conversation**: conversation history (prior turns) accumulates in the prefix. vLLM caches each turn as it completes, so turn N only needs to prefill the latest user message, not the full conversation history.
- **Few-shot templates**: a 20-example few-shot prompt that never changes is an ideal prefix cache entry.

Low-ROI scenarios:
- **Unique documents**: a translation service where each request is a different document. No prefix sharing, no cache hits.
- **Very short prompts**: prompts under 64 tokens generate few blocks. The hashing overhead becomes a non-negligible fraction of the total prefill time.
- **Memory pressure**: if the cache is constantly evicted before requests can reuse it, the caching overhead adds latency without benefit. In this case, reduce `max_num_seqs` or `gpu_memory_utilization` to create more stable cache entries.

## 9. Chunked prefill: TTFT vs TPOT trade-off

The prefill bottleneck is one of the less-discussed challenges in LLM serving. In a standard setup, when a long-context request arrives (say, a 16K-token document for summarization), its prefill step monopolizes the scheduler for potentially hundreds of milliseconds. Every other running sequence — a short chatbot request that's one decode step from finishing, a streaming response that's already been sending tokens to the client — sits stalled.

Chunked prefill solves this by splitting a long prefill into chunks no larger than `max_num_batched_tokens` tokens. Each chunk takes one engine step. Between chunks, other sequences can run their decode steps.

![Chunked prefill cuts p99 TTFT from 4 s to 0.8 s by interleaving decode steps with long-context prefill chunks](/imgs/blogs/vllm-deep-dive-7.png)

### The scheduling math

Without chunked prefill, a 16K-token prefill with `max_num_batched_tokens=8192` would require one 16K-token step (exceeding the limit — vLLM would actually need to reduce the batch to fit). With `enable_chunked_prefill=True` and `max_num_batched_tokens=4096`, the 16K prefill is split into four chunks of 4096 tokens each, taking four engine steps.

In those four steps, other sequences each get 4 decode tokens. For a request that was 10 steps from finishing its decode phase, chunked prefill delays its completion by at most 4 additional steps — 40ms at 10ms/step. But for a short request that arrived while the long prefill was happening, it now gets scheduled within the first chunk step instead of waiting for the full 4s prefill.

The TPOT cost is real but small. Chunked prefill adds $\lceil L_{prefill} / C \rceil - 1$ extra scheduling overhead calls, where $L_{prefill}$ is the prefill length and $C$ is the chunk size. Each overhead call adds approximately 0.1–0.5ms of scheduler overhead. For a 16K prefill with C=4096, that's 3 extra overhead calls = 0.3–1.5ms total overhead, or 1–5% of a 30ms decode step time.

### Configuring chunked prefill

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --port 8000
```

The `max-num-batched-tokens` is both the chunk size and the maximum compute budget per step. Setting it too low (e.g., 512) creates many small chunks and adds significant scheduler overhead. Setting it too high (e.g., 32768) reduces the effectiveness of chunking (a 4K prefill won't be chunked at all). For most A100/H100 deployments, 4096–8192 is the right range.

Chunked prefill is recommended to be always on for any mixed-length workload. The only scenario where you'd disable it is a throughput-only benchmark where all requests are identical length and you want maximum per-step efficiency.

## 10. Benchmarking vLLM

The benchmarking script `benchmarks/benchmark_serving.py` in the vLLM repository is the definitive tool for sizing a deployment and validating configuration changes.

```bash
# Clone vLLM for access to benchmark scripts
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Download the ShareGPT dataset (a realistic mix of conversation lengths)
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/\
resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
-O data/ShareGPT_V3_unfiltered_cleaned_split.json

# Start the vLLM server (in a separate terminal or background)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 &

# Run the benchmark
python benchmarks/benchmark_serving.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --backend openai \
    --base-url http://localhost:8000 \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --request-rate 20 \
    --max-concurrency 200
```

Key flags:
- `--request-rate N`: Poisson arrival rate in requests/second. This simulates realistic traffic rather than saturating the server with max concurrency.
- `--max-concurrency N`: cap on simultaneous outstanding requests. Prevents a burst of slow requests from starving faster ones in the benchmark itself.
- `--backend openai`: sends requests via the OpenAI HTTP client, measuring full end-to-end latency including HTTP overhead, JSON serialization, and network time. Use `--backend vllm` for Python-direct benchmarks that exclude HTTP.
- `--dataset-name sharegpt`: uses the ShareGPT conversation dataset for realistic prompt length distributions (heavy tail of long conversations mixed with short exchanges).

Sample output from a tuned Llama-3-8B deployment on A100 40GB:

```
Benchmarking summary:
  Backend:                  openai
  Total time:               51.3 s
  Throughput:               19.49 req/s (< 20 req/s — slight queue buildup)
  Throughput (tokens/s):    4,271 output tokens/s
  Mean TTFT (ms):           187.4
  Median TTFT (ms):         143.2
  P99 TTFT (ms):            612.8
  Mean TPOT (ms):           11.8
  Median TPOT (ms):         10.9
  P99 TPOT (ms):            31.4
  Mean E2E latency (ms):    2,341.7
  P99 E2E latency (ms):     9,482.1
```

Interpreting these numbers:

- **Throughput at 19.49 req/s vs 20 req/s input rate**: the server is at ~97% utilization. Any further load increase will cause queue depth to grow unboundedly (Little's Law: if $\lambda > \mu$, queue depth → ∞).
- **p99 TTFT 613ms**: dominated by queue wait time for requests that arrive when the scheduler is busy with long prefills. Chunked prefill can reduce this significantly.
- **p99 TPOT 31.4ms vs median 11ms**: the p99 outliers are steps where the batch included heavy prefill work that delayed decode steps.

To find the maximum sustainable throughput (the knee of the latency-throughput curve), run the benchmark at 5, 10, 15, 20, 25 req/s and plot p99 TTFT vs throughput. You'll see p99 TTFT rise sharply around the server's saturation point. The optimal operating point is typically 70–80% of saturation, leaving headroom for traffic bursts.

## 11. Tensor parallelism: when and how

Tensor parallelism (TP) splits each weight matrix across multiple GPUs by sharding along the column dimension (for input projections) or row dimension (for output projections). Each GPU computes a shard of the output activations, then a single AllReduce collective sums all shards to reconstruct the full activations.

![Tensor parallelism across 4 GPUs: each GPU holds one column-shard of Q/K/V/O and FFN matrices, AllReduce at the end](/imgs/blogs/vllm-deep-dive-6.png)

For a Transformer attention layer with TP=4:

- Each GPU holds 1/4 of the Q, K, V weight matrices (column-sharded) and 1/4 of the output projection (row-sharded).
- Each GPU computes its 1/4 of the attention output.
- One AllReduce sums the 1/4 outputs to get the full result.

The same pattern applies to FFN layers: the first linear is column-sharded, the second is row-sharded, and one AllReduce after the second linear reconstructs the full FFN output.

### AllReduce cost on NVLink

For each Transformer layer with TP=P, the AllReduce transfers $2 \times N_{elements} \times (P-1)/P$ bytes per GPU (the ring-AllReduce pattern). For H100 with NVLink 4.0 (450 GB/s bidirectional), a typical AllReduce of hidden-size elements:

For Llama-3-8B (hidden dim 4096), batch=1, BF16:
$$\text{elements} = 4096 \times 2 \text{ bytes} = 8,192 \text{ bytes per token per layer}$$

With TP=4:
$$t_{allreduce} = \frac{2 \times (4-1)}{4} \times \frac{8192}{450 \times 10^9} \approx 27 \text{ ns}$$

Negligible. Even for batch=128 (131,072 bytes):

$$t_{allreduce} = \frac{1.5 \times 131,072}{450 \times 10^9} \approx 437 \text{ ns}$$

Still sub-microsecond on NVLink. This is why TP on NVLink-connected GPUs is essentially free from a communication standpoint. On PCIe (63 GB/s inter-GPU over CPU bridge), the same AllReduce takes ~3 µs — still acceptable. Over Ethernet (100 Gbps ≈ 12.5 GB/s), it takes ~15 µs per layer, and for a 32-layer model that accumulates to 480 µs per step — now meaningful relative to a 5ms decode step.

The rule is simple: **use NVLink for TP.** Never cross PCIe or Ethernet for TP.

### TP sizing decision

**TP = 1 (single GPU)**: use when the model fits on one GPU. Best throughput efficiency since there's no AllReduce overhead.

**TP = 2**: use when the model doesn't fit on one GPU but fits on two with NVLink. Also beneficial for latency at batch=1 (two GPUs doing half the compute each = ~1.8× faster per step).

**TP = 4 or 8**: use for very large models (70B+) or when you need sub-10ms TPOT for latency-critical single-user scenarios. At TP=8 with 8×H100, a 70B decode step can be as fast as 3ms.

**Pipeline parallelism (PP)**: don't use for serving unless you have no choice. PP introduces pipeline bubbles at the start and end of each batch, reducing efficiency at small batch sizes. TP is almost always preferable.

## 12. Operational patterns

### Running behind Nginx

vLLM's built-in server uses Uvicorn for HTTP handling, which is sufficient for moderate traffic. For production, put Nginx in front for TLS termination, connection pooling, and request buffering.

```nginx
upstream vllm_backend {
    server 127.0.0.1:8000;
    keepalive 100;
    keepalive_requests 10000;
    keepalive_timeout 60s;
}

server {
    listen 443 ssl http2;
    server_name api.yourcompany.com;

    ssl_certificate /etc/ssl/certs/api.crt;
    ssl_certificate_key /etc/ssl/private/api.key;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Streaming responses require no buffering
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    proxy_connect_timeout 5s;
    proxy_send_timeout 300s;

    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Authorization $http_authorization;
    }

    location /health {
        proxy_pass http://vllm_backend/health;
        proxy_read_timeout 5s;
        access_log off;
    }

    location /metrics {
        # Restrict metrics to internal prometheus scraper
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://vllm_backend/metrics;
    }
}
```

The critical setting for streaming is `proxy_buffering off`. Without it, Nginx accumulates the entire response body before forwarding it to the client. A 500-token streaming response at 10ms/token takes 5 seconds to generate — the client waits 5 seconds with no output, then gets all 500 tokens at once. This defeats the entire purpose of streaming.

### Health check and readiness probes

For Kubernetes deployments, configure liveness and readiness probes on the `/health` endpoint:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 300    # Model loading can take 2-5 minutes
  periodSeconds: 30
  failureThreshold: 3
  timeoutSeconds: 5

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 300
  periodSeconds: 10
  failureThreshold: 1         # Remove from rotation immediately on failure
  timeoutSeconds: 5
```

The `initialDelaySeconds=300` is important — a 70B model on A100s can take 3–4 minutes to load from NFS/S3. Configure your liveness probe to not kill the pod during this legitimate startup window.

### Memory monitoring and alerting

Set up Prometheus scraping of the `/metrics` endpoint and alert on these signals:

- `vllm:gpu_cache_usage_perc > 0.95`: KV cache nearly full — throughput will drop as the scheduler has to preempt sequences.
- `vllm:cpu_cache_usage_perc > 0.1`: significant swapping — step latency will spike.
- `vllm:num_requests_waiting > 50`: queue buildup — request TTFT is growing.
- `vllm:time_to_first_token_seconds{quantile="0.99"} > 2.0`: p99 TTFT SLA breach.

### Graceful shutdown

When deploying rolling updates or shutting down for maintenance, use graceful shutdown rather than hard kills. Send `SIGTERM` to the vLLM process. The server will:

1. Stop accepting new requests (return 503 on new connections).
2. Allow in-flight requests to complete.
3. Exit after all sequences finish or after the `--uvicorn-timeout-graceful-shutdown` seconds (default 30).

For long-context generation workloads (where individual requests can run for minutes), you may need to increase the graceful shutdown timeout, or drain the deployment by shifting traffic to other instances before sending SIGTERM.

## 13. The vLLM deployment decision matrix

Before reaching for a specific configuration, it helps to think about the problem space systematically. Every deployment lives somewhere on the latency-throughput-cost triangle, and vLLM's configuration knobs map cleanly to moves along that triangle.

![vLLM deployment decision matrix: four hardware/config combinations mapped to SLO targets and cost](/imgs/blogs/vllm-deep-dive-8.png)

The four archetypal deployment configurations for a 70B-class model reveal how the knobs interact:

**Single A100 80GB, BF16**: the model barely fits (140 GB weights on 80 GB — it doesn't, you need 2). This configuration is only valid for models up to ~30B in BF16. Throughput is moderate, cost per token is moderate, and you have limited KV cache headroom.

**Single A100 80GB with AWQ INT4**: a 30–70B model now fits in 35–40 GB, leaving 35–40 GB for KV cache. Throughput roughly doubles compared to BF16 on the same hardware because you can sustain much larger batch sizes. Cost per token drops significantly. This is the sweet spot for most cost-sensitive serving scenarios.

**4×A100 80GB with TP=4, BF16**: the 70B model is sharded across 4 GPUs. Each GPU holds 1/4 of the weights (~35 GB) plus its share of the KV cache. Total KV cache capacity is 4× a single GPU's available headroom. TPOT is lower than single-GPU because each matrix multiplication is 4× smaller (4 GPUs each doing 1/4 the work). This configuration maximizes throughput and minimizes per-request latency, at 4× the cost.

**4×A100 80GB with TP=4, BF16, plus speculative decoding**: adds speculative decoding on top of the 4-GPU configuration. At low batch sizes (the regime where this machine can support latency SLAs in the single-digit milliseconds), speculative decoding provides another 2–3× TPOT reduction. This is the configuration for a single-user real-time code completion service or an agentic loop with tight latency budgets.

### Matching configuration to SLO

The right configuration depends on which SLO you're optimizing:

**Throughput-first (batch jobs, background processing)**: AWQ INT4 on the fewest GPUs that fit the model. Maximize batch size. Disable speculative decoding. Use prefix caching if there's prompt sharing.

**Latency-first (interactive chatbot, real-time code completion)**: TP across multiple GPUs to reduce TPOT. Add speculative decoding if batch size is consistently < 4. Use chunked prefill to keep TTFT predictable.

**Cost-first (high-volume API)**: AWQ INT4 + prefix caching + chunked prefill on the cheapest GPU class that meets your p99 latency SLA. Optimize for tokens per dollar, not tokens per second per GPU.

**Mixed workload (both interactive and batch)**: this is the common production case. Run separate deployments for interactive (latency-optimized) and batch (throughput-optimized) traffic, and route at the gateway layer. The interactive tier runs speculative decoding at TP=2 or TP=4; the batch tier runs AWQ INT4 at maximum batch size.

## 14. Case studies and benchmarks

### The vLLM paper (Kwon et al., SOSP 2023)

The original vLLM paper reported throughput comparisons on A100 80GB with OPT-13B and OPT-66B:

| System | OPT-13B throughput | vs HuggingFace |
|---|---|---|
| HuggingFace Text Generation | ~120 tokens/s | 1× baseline |
| FasterTransformer | ~840 tokens/s | 7× |
| Orca (static batching) | ~1,200 tokens/s | 10× |
| vLLM (PagedAttention) | ~2,880 tokens/s | 24× |

The 24× improvement over HuggingFace is largely from continuous batching (HF runs each request serially). The 3.5× improvement over FasterTransformer (which does dynamic batching but with pre-allocated contiguous KV memory) is from PagedAttention's fragmentation elimination, enabling 3× larger effective batch sizes.

The paper also measured memory waste: naive allocation wasted 20.4% of KV cache memory on the ShareGPT workload; PagedAttention wasted 4.0%, a 5× improvement.

### Anyscale throughput benchmarks (Llama-2-7B, A100 80GB, 2023)

Anyscale published public benchmarks comparing vLLM configurations on Llama-2-7B at fixed concurrency=100:

| Configuration | Throughput (tokens/s) | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---|---|---|
| vLLM 0.1 baseline | 1,847 | 38 | 8.4 |
| + prefix caching | 2,341 (+27%) | 22 | 8.5 |
| + AWQ INT4 | 3,109 (+68%) | 31 | 5.3 |
| + AWQ + prefix caching | 3,847 (+108%) | 19 | 5.2 |
| + speculative decoding | 2,194 (+19%) | 38 | 3.9 |

The speculative decoding row is at the same concurrency=100, where batch size is large — hence the modest speedup (less than prefix caching or AWQ). At batch=1, speculative decoding would show 2–3× improvement.

### Production case study: RAG document Q&A with prefix caching

A document Q&A system running Llama-3-8B served 50,000 requests/day from 5,000 unique documents. Each request included 1–3 retrieved document chunks (average 1,400 tokens/chunk) prepended to a user question.

After enabling prefix caching and running for 2 hours (enough for the cache to warm with popular documents):
- Cache hit rate: 76%
- Effective prefill tokens per request (after cache): 420 tokens average (down from 1,750 average)
- TTFT p50: reduced from 380ms to 95ms
- GPU utilization: increased from 68% to 89% (freed compute from cached prefills allocated to more concurrent requests)
- Effective throughput: +2.3×

The improvement was visible only after cache warmup. During the first 30 minutes of deployment, the hit rate was near zero. This "cold start" period is important to account for in deployment planning — prefix caching doesn't help when you first bring up a new instance after a rolling update.

### H100 vs A100 measured performance

Measured on vLLM v0.4.x with Llama-3-8B-Instruct, default settings, ShareGPT workload:

| Metric | A100 40GB SXM | A100 80GB SXM | H100 80GB SXM |
|---|---|---|---|
| HBM bandwidth | 1.55 TB/s | 2.0 TB/s | 3.35 TB/s |
| Max KV cache (8B BF16) | ~28 GB | ~56 GB | ~56 GB |
| TPOT at batch=1 | ~12ms | ~9ms | ~4ms |
| TPOT at batch=32 | ~22ms | ~18ms | ~12ms |
| Max throughput (tok/s) | ~2,100 | ~2,800 | ~5,400 |
| Optimal batch size | ~64 | ~64 | ~128 |

The H100's 1.7× higher HBM bandwidth directly translates to 1.7× lower TPOT at small batch sizes (memory-bandwidth-bound regime). The H100's FP8 support and higher compute throughput additionally help at large batch sizes (compute-bound regime).

## 14. Sizing a deployment: worked examples

#### Worked example: sizing for 100 QPS at a chat assistant

You're building a customer support chatbot backed by Llama-3-8B-Instruct. Traffic profile: 100 QPS average, 300 QPS peak (3× factor, consistent with typical chat traffic patterns). Workload: average input 400 tokens (3–4 turns of chat history + user message), average output 280 tokens. SLA requirements: p99 TTFT < 500ms, p99 TPOT < 25ms, p99 E2E latency < 8 seconds.

**Step 1: estimate peak throughput requirement**

At 300 QPS × 280 output tokens/request:
$$\text{peak output throughput} = 300 × 280 = 84,000 \text{ tokens/s}$$

**Step 2: per-GPU capacity**

A single A100 40GB running Llama-3-8B-Instruct with AWQ INT4 and prefix caching (realistic for a chat assistant where 70% of tokens are conversation history cached from the system prompt and prior turns) achieves approximately 3,500 tokens/s at `batch_size≈50` (based on Anyscale benchmarks, adjusted for instruction-tuned model).

**Step 3: GPU count**

$$\text{GPUs required at peak} = \lceil 84,000 / 3,500 \rceil = 24 \text{ GPUs}$$

With 20% headroom (operating at 80% utilization to avoid latency cliff): $24 / 0.80 = 30$ GPUs.

**Step 4: per-instance configuration**

Each instance handles: $300 \text{ QPS} / 30 \text{ instances} = 10 \text{ req/s}$ per instance.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3-8b-instruct-awq \
    --quantization awq \
    --gpu-memory-utilization 0.88 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 128 \
    --port 8000 \
    --served-model-name llama-3-8b
```

**Step 5: validate with load testing**

```bash
python benchmarks/benchmark_serving.py \
    --model llama-3-8b \
    --backend openai \
    --base-url http://instance-0:8000 \
    --request-rate 10 \
    --num-prompts 2000 \
    --input-len 400 \
    --output-len 280
```

Expected results: p99 TTFT ~350ms (under 500ms SLA), p99 TPOT ~18ms (under 25ms SLA), p99 E2E ~6.4s (under 8s SLA).

**Cost estimate:**
- 30 × A100 40GB at \$1.50/hr each = \$45/hr
- At 100 QPS average and 280 tokens/response: 100 × 280 × 3600 = 100.8M output tokens/hr
- Cost per 1M output tokens: \$45 / 100.8 ≈ \$0.45/M tokens

With AWQ quantization on 80GB A100s (fewer GPUs needed): \$0.28/M tokens. With H100s (higher throughput, fewer instances): \$0.32/M tokens at higher single-unit cost but fewer units.

#### Worked example: sizing speculative decoding for a batch summarization pipeline

You maintain a news summarization pipeline that processes 10,000 articles per day in sequential batches (not concurrent requests — one at a time). Article length: 800–2,500 tokens. Summary length: 150–300 tokens. The pipeline runs on a single A100 40GB.

Current performance (Llama-3-8B BF16, batch=1):
- TTFT per article: ~150ms (800-token prefill)
- TPOT: ~9ms/token (batch=1, memory-bandwidth-bound)
- Mean summary time: 150ms + 225 tokens × 9ms = 150 + 2,025 = 2,175ms per article
- Daily pipeline runtime: 10,000 × 2,175ms = 6.04 hours

Task is news summarization — moderately predictable output (factual summary follows article structure), but less predictable than code. Expected $\alpha \approx 0.70$.

With $K=5$, $S \approx 15$ (8B at 9ms vs 1.1B TinyLlama at ~0.6ms):

$$\text{speedup} = \frac{1 - 0.70^6}{0.30 \times (1 + 5/15)} = \frac{1 - 0.118}{0.30 \times 1.333} = \frac{0.882}{0.400} \approx 2.2\times$$

New TPOT: $9ms / 2.2 = 4.1ms$.
New mean summary time: $150ms + 225 × 4.1ms = 150 + 923 = 1,073ms$.
New daily runtime: $10,000 × 1,073ms = 2.98$ hours.

The pipeline finishes 3 hours earlier each day, freeing the GPU for other tasks during morning hours. The TinyLlama draft model adds ~2.2 GB weight memory (fits easily on A100 40GB with 8B BF16 using `gpu_memory_utilization=0.85`).

Configuration:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num-speculative-tokens 5 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port 8000
```

Note: `max-model-len=4096` is sufficient for this pipeline (800 input + 300 output = 1,100 tokens max), and reducing it frees additional KV cache memory that could support higher concurrency if you ever parallelize the pipeline.

## 15. When to use vLLM, and when not to

**Use vLLM for virtually all transformer LLM serving.** The combination of continuous batching, PagedAttention, quantization support, and the OpenAI-compatible API makes it the correct choice for almost every LLM serving scenario. The only reasons to reach for an alternative (TGI, TorchServe, Triton with a custom LLM backend) are narrow:

- You need features not yet in vLLM: SpecDec Medusa, custom attention backends, or server-side LoRA selection without the S-LoRA pattern.
- You're running on a non-CUDA device (AMD ROCm, Habana Gaudi) where TGI has better support.
- You have tight TorchScript or ONNX export requirements for compliance reasons.

**Use speculative decoding when:**
- Effective batch size is consistently ≤ 4.
- The task has predictable output (code completion, news summarization, template filling).
- You have spare VRAM for a 100M–1.1B draft model.
- Acceptance rate $\alpha > 0.65$ measured on your task's output distribution.

**Do NOT use speculative decoding when:**
- Batch size is consistently > 8. At large batches, the model is already compute-bound; draft token verification adds cost without addressing the bottleneck.
- Temperature is above 0.9 (high-entropy sampling). Acceptance rates drop below 0.5, speculative decoding becomes a net negative.
- Latency is dominated by TTFT rather than TPOT. Speculative decoding only helps TPOT, not TTFT.

**Use tensor parallelism (TP > 1) when:**
- The model doesn't fit on a single GPU — TP is necessary.
- You need lower per-request TPOT (more GPUs = less compute per step at batch=1).
- All TP ranks are on the same physical server with NVLink or high-bandwidth interconnect.

**Do NOT use TP across PCIe or across servers over Ethernet** unless the model fits in no other way. Inter-node AllReduce over 100Gbps Ethernet adds ~15 µs per layer, which accumulates to milliseconds per step.

**Use prefix caching when:**
- More than 30% of requests share a prefix longer than 512 tokens.
- You're using a fixed system prompt, few-shot template, or RAG with a recurring document corpus.

**Use chunked prefill for all mixed-length workloads.** The cost is minimal (< 5% TPOT overhead), and the p99 TTFT improvement for long-context requests is significant (3–8×).

**Use AWQ INT4 over GPTQ INT4** for instruction-following models — better accuracy at identical memory footprint. Use FP8 (H100 only) when you need both memory savings and compute acceleration.

**Set max_model_len to your actual usage context**, not the model's maximum. A Llama-3 model with 128K native context that you're only using at 8K saves 16× KV cache memory.

## 16. Key takeaways

1. **PagedAttention is the core, not the API.** Every vLLM performance advantage traces back to non-contiguous KV block allocation eliminating fragmentation. The scheduler, the workers, and the parallelism strategies are all in service of this memory management layer.

2. **Use AsyncLLMEngine in production, always.** The synchronous `LLM` class blocks the event loop. A production server handling concurrent requests must use `AsyncLLMEngine`.

3. **`gpu_memory_utilization` controls KV cache size, not GPU occupancy.** The formula is $(VRAM × utilization) - weights - activations$. Always calculate actual KV cache headroom explicitly before running in production.

4. **Quantization (GPTQ/AWQ INT4) is about KV cache headroom, not just weight compression.** Reducing weight size from 140 GB to 35 GB gives 3–4× more GPU memory for KV cache, enabling proportionally larger batch sizes and throughput.

5. **Speculative decoding is a batch-size-1 trick.** It addresses the memory-bandwidth bottleneck in decode, which only exists when the GPU isn't compute-saturated. At batch > 8, speculative decoding is neutral-to-harmful.

6. **Prefix caching requires cache warmup.** Hit rates are low for the first 15–30 minutes after a fresh deployment. Factor this into rolling update timing.

7. **Chunked prefill is almost always the right default.** For mixed-length workloads, the p99 TTFT improvement (3–8×) far outweighs the TPOT overhead (< 5%).

8. **TP on NVLink is essentially free.** AllReduce latency on NVLink is sub-microsecond for typical sequence lengths. Don't hesitate to use TP=4 or TP=8 on NVLink-connected servers to reduce TPOT at low batch sizes.

9. **Abort cancelled requests immediately.** `engine.abort(request_id)` frees KV cache blocks. Without it, disconnected clients continue consuming resources.

10. **Monitor `vllm:cpu_cache_usage_perc`.** Any value above 5% means you're preempting and paying PCIe transfer costs. Reduce `max_num_seqs` or `gpu_memory_utilization` to fix it before p99 latency degrades.

## Further reading

- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023* — the foundational paper for vLLM's architecture.
- [vLLM official documentation](https://docs.vllm.ai) — EngineArgs reference, quantization guides, distributed serving configuration.
- [Continuous batching and PagedAttention internals](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — the C2 post in this series; full treatment of the scheduler algorithm and KV block allocation.
- [Why LLM serving is different from classical model serving](/blog/machine-learning/model-serving/why-llm-serving-is-different) — C1 post; memory wall, autoregressive bottleneck, and the motivation for PagedAttention.
- [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) — C6 post; GPTQ vs AWQ vs FP8 accuracy-throughput trade-offs, calibration datasets, quantization artifacts.
- [Speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production) — E5 post; dedicated deep-dive on draft model selection, acceptance rate measurement, and production deployment patterns.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — series capstone; the complete decision tree from model to production.

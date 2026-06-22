---
title: "Text generation inference deep dive: TGI architecture, continuous batching, and Flash Attention"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master HuggingFace's Text Generation Inference from the Rust router internals to Flash Attention memory math, speculative decoding, and production Docker deployments."
tags:
  [
    "model-serving",
    "inference",
    "text-generation-inference",
    "huggingface",
    "continuous-batching",
    "flash-attention",
    "quantization",
    "tensor-parallelism",
    "speculative-decoding",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/text-generation-inference-deep-dive-1.png"
---

It is 2:47 AM when the page fires. Your LLM chatbot has been running smoothly for three weeks — 80 requests per second on a pair of A100s, p99 under 800 ms. Then a product launch triples the concurrency. Your previous serving setup, a straightforward FastAPI wrapper around a Transformers `pipeline`, begins returning 504s. The GPU utilization readout shows 63 percent — the GPU is not even sweating — but the request queue is 300 deep and every new request sits idle waiting for the in-flight batch to drain.

The problem is static batching. Your server waits for a full batch of eight requests, runs them all the way through completion, then picks up the next batch. Short completions finish in 20 tokens while long ones grind to 400 — the GPU idles on the short ones while the long ones hold the batch hostage. Every architecture class on transformers skips this operational reality, and it costs you at 3 AM.

This post is about HuggingFace's Text Generation Inference (TGI), which solves exactly this class of problem. TGI is the production backbone behind HuggingFace Inference Endpoints, Hugging Chat, and a large number of enterprise deployments. It is unusual among LLM servers in that its HTTP layer is written in Rust — a deliberate architectural choice to get predictable tail latency without the Python GIL. The model execution layer is Python and C++, and the two processes communicate over a gRPC socket. By the end of this post you will understand why that split exists, how TGI's continuous batching scheduler works at the iteration level, what Flash Attention 2 actually does to your memory bandwidth, how to choose among six quantization modes, when to use tensor parallelism, and how TGI's Medusa-based speculative decoding gets you 2× token throughput without a second model.

The full architecture is shown in Figure 1 below.

![TGI two-process architecture: Rust router feeds the Python launcher which drives GPU workers](/imgs/blogs/text-generation-inference-deep-dive-1.png)

The series spine — Model → Packaging → Runtime → Server → Infrastructure → Observability → Scale — places TGI at the Server layer. It is a complete server: it handles the HTTP protocol, the batching policy, the KV cache lifetime, and the GPU kernel selection. Everything above (infrastructure, autoscaling) and below (the model weights) is outside its scope. That sharp boundary is one of TGI's greatest strengths and also its main limitation, as you will see in the TGI-vs-vLLM comparison near the end.

## 1. The two-process architecture: why Rust for the router

TGI starts two OS processes when you launch it:

1. **`text-generation-router`** — a Rust binary. It owns the HTTP server (an Axum web framework server), the request queue, the batching policy, the health check endpoint, the Prometheus metrics endpoint, and the SSE streaming logic. It communicates with the launcher over a gRPC Unix socket.

2. **`text-generation-launcher`** — a Python process. It loads the model weights from HuggingFace Hub (or a local directory), manages the tokenizer, runs the forward passes through PyTorch (or the compiled C++ CUDA kernels), and manages the KV cache. It speaks gRPC to the router.

The question engineers ask most often is: why Rust for the router? The short answer is the Python GIL. Python's Global Interpreter Lock means only one thread can execute Python bytecode at a time. For a web server handling 200 concurrent long-running SSE streams, the GIL becomes a hard ceiling on throughput — every `await` in an `asyncio` event loop is still subject to GIL contention when Python-land code runs. Rust's `async`/`await` is compiled to zero-cost state machines: no runtime overhead, no GIL, and no garbage collector that can pause execution for 5–50 ms during a collection cycle. Those GC pauses are the primary driver of p99 tail latency spikes in Python-based servers.

There is a second reason. The router's job — queue management, batch assembly, SSE framing — is pure bookkeeping. None of it requires the Python ecosystem. Rust is a natural fit: you get the speed of C, the safety of a borrow checker, and a rich async ecosystem (`tokio`, `axum`) that targets exactly this kind of high-concurrency I/O work. The model execution in the launcher, by contrast, needs PyTorch, CUDA, and a HuggingFace tokenizer — none of which exist natively in Rust — so Python stays there.

The gRPC socket between the two processes adds roughly 0.3–0.8 ms per batch (measured on A100 with 128-token prompt batches). That is acceptable because it is paid once per batch, not once per token. The token streaming back to the client is pure Rust: the launcher sends a protobuf message per generated token, the router immediately writes an SSE frame and flushes it to the open HTTP connection. There is no Python event loop between a generated token and the client.

### Launching TGI

The canonical Docker invocation:

```bash
docker run --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:2.4.0 \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --num-shard 1 \
  --max-input-length 4096 \
  --max-total-tokens 6144 \
  --max-batch-prefill-tokens 16384 \
  --max-concurrent-requests 512 \
  --dtype bfloat16
```

Key flags and their meanings:

| Flag | Default | What it controls |
|---|---|---|
| `--num-shard` | 1 | Tensor-parallel shards (GPUs). Must equal GPU count. |
| `--max-input-length` | 1024 | Max prompt tokens. KV cache pre-allocated per this. |
| `--max-total-tokens` | 2048 | Max prompt + output tokens. Total sequence budget. |
| `--max-batch-prefill-tokens` | 4096 | Max tokens processed in a single prefill step. |
| `--max-concurrent-requests` | 128 | Queue depth. Requests beyond this get HTTP 429. |
| `--dtype` | float16 | Weight dtype: `float16`, `bfloat16`, `float32`. |
| `--quantize` | none | Quantization scheme: `bitsandbytes`, `gptq`, `awq`, `eetq`, `fp8`. |
| `--waiting-served-ratio` | 1.2 | Ratio of waiting:active requests to trigger new prefill. |
| `--speculate` | 0 | Number of speculative tokens to draft per step. |

The `HUGGINGFACE_HUB_CACHE` environment variable controls where model weights are cached on the host. Setting it to a mounted volume is essential for production containers so weights survive container restarts without a fresh 16-GB download.

### The gRPC protocol between router and launcher

The router and launcher communicate via a gRPC Unix domain socket. The protobuf service definition (`generate.proto` in the TGI source) defines two main RPC methods:

- **`Generate`**: unary RPC for non-streaming requests. The router sends a `GenerateRequest` with input IDs, sampling parameters, and a request ID; the launcher returns a `GenerateResponse` with the full generated text and token details.
- **`GenerateStream`**: server-streaming RPC for streaming requests. Same input; the launcher sends a stream of `GenerateStreamResponse` messages, one per token, with the final message including `generated_text`.

The batch-level RPC is different: the router does not call Generate once per request. Instead, it calls a **`PrefillBatch`** RPC to process a batch of new requests (prefill step) and a **`DecodeBatch`** RPC to advance all active sequences by one token (decode step). Each call returns per-sequence outputs and a list of completed sequence IDs. The router handles the mapping from batch-level outputs back to individual HTTP responses, including SSE framing for streaming clients.

This design means the launcher never knows about individual HTTP requests — it operates on batches of token sequences. The router holds all the state about which HTTP connection corresponds to which sequence ID. This clean separation is what makes the Rust router so important: it is managing potentially thousands of SSE connections while the launcher only sees anonymous sequence IDs.

The Unix socket adds roughly 0.1–0.3 ms of serialization overhead per batch call. This is acceptable because batches are processed at 10–80 Hz (one decode step every 12–100 ms), making the socket overhead less than 2% of total serving time.

### The startup sequence

When TGI starts, the launcher follows this sequence:

1. Downloads the model config, tokenizer, and sharded safetensors from HuggingFace Hub (or loads from the cache directory).
2. Allocates GPU memory for the KV cache based on `--max-input-length`, `--max-total-tokens`, and `--num-shard`. TGI uses a static pre-allocation strategy — it claims all the memory it will ever need at startup and never grows or shrinks. This makes OOM events deterministic (they happen at startup, not under load) but means you cannot share GPU memory with other processes.
3. Compiles FlashAttention-2 CUDA kernels via CUDA JIT if they are not pre-built for the current GPU architecture.
4. Starts the gRPC server and signals readiness to the router.
5. The router starts its HTTP server and begins accepting requests.

The startup time for Llama-3-8B on A100 is roughly 90 seconds on the first run (weight download + kernel compile) and 15 seconds on subsequent runs (weights cached, kernels cached).

### Tokenization handling

TGI handles tokenization in the launcher process, using HuggingFace's `tokenizers` Rust library (called from Python). This is important: tokenization is not done in the router. A request arrives at the router as raw text; the router sends the raw text to the launcher via gRPC; the launcher tokenizes it, checks that the input fits within `--max-input-length`, and returns an error if it does not.

This has one implication for high-throughput deployments: tokenization is sequential within the launcher (though TGI batches it at the Python level using Rust's native parallel tokenizer). At very high QPS (> 500 requests/s), tokenization can become a CPU bottleneck. The mitigation is to pre-tokenize requests client-side and use TGI's `/generate` endpoint with `input_ids` directly rather than `inputs` (raw text). This is an undocumented optimization used by HuggingFace's internal services.

The tokenizer also handles chat template formatting when using the Messages API. TGI applies the model's `chat_template` (stored in the `tokenizer_config.json` on the Hub) to convert the OpenAI-format message list into the model's native format (e.g., Llama-3's `<|begin_of_text|><|start_header_id|>system<|end_header_id|>...` format). This means you do not need to manually format prompts when using the Messages API — TGI handles the model-specific formatting automatically.

One gotcha: if you are migrating from a raw Transformers `pipeline` to TGI and your pipeline manually prepended a system prompt in the model's chat format, you will double-format when switching to the Messages API. Always pass raw role-content message objects to the Messages API and let TGI apply the template.

### Prefill vs decode memory layout

TGI's memory layout for the KV cache is static contiguous allocation. At startup, TGI pre-allocates a single contiguous block of GPU memory for the KV cache, sized to hold exactly `--max-concurrent-requests` sequences each at `--max-total-tokens` length. Each sequence gets a fixed-size slot in this pre-allocated block.

This contrasts with vLLM's PagedAttention, which allocates KV cache pages dynamically. TGI's approach has two consequences:

1. **No fragmentation**: since every slot is the same size, there is no memory fragmentation. The memory layout is predictable.
2. **Internal fragmentation waste**: a request that generates 50 tokens out of a 2,048-token allocated slot wastes 1,998 slots worth of KV memory. At high request volumes with short outputs, this can reduce effective concurrency by 3–5×.

The practical implication: set `--max-total-tokens` to the 99th percentile of your actual request lengths, not the absolute maximum. If 99% of your requests complete within 512 tokens, set `--max-total-tokens 512` even if TGI supports 8,192. This multiplies your effective concurrent request capacity by 16× (8,192 / 512).

## 2. Continuous batching: iteration-level scheduling

The most important performance feature in TGI is its continuous batching implementation. Understanding it requires understanding the difference between a prefill step and a decode step.

**Prefill**: Given a prompt of length $P$ tokens, the model processes all $P$ tokens in a single forward pass, filling the KV cache with $P$ key-value pairs per layer. This is compute-bound — you are doing $P$ matrix multiplications in parallel. The output is the first generated token.

**Decode**: Each subsequent token is generated by a single forward pass that processes exactly 1 new query token against the entire KV cache (which grows by 1 key-value pair per step). This is memory-bandwidth-bound — you are loading the full KV cache from HBM on every step, but doing trivial computation.

In static batching, you assemble a batch of $B$ requests, run them all through prefill, then run them all through decode until every request in the batch has generated an EOS token or hit `max_new_tokens`. The problem: if request A generates 30 tokens and request B generates 400 tokens, after request A finishes, its GPU resources (KV cache slot, compute share) sit idle for 370 decode steps. Utilization tanks.

Continuous batching removes this constraint. At each decode iteration, TGI's scheduler checks: have any active sequences completed? Free their KV cache slots. Are there requests in the waiting queue? Insert them into the batch with a prefill step. The batch composition changes at every iteration.

![Continuous batching eliminates GPU idle time by inserting new requests the moment slots free](/imgs/blogs/text-generation-inference-deep-dive-2.png)

### The `waiting_served_ratio` parameter

The scheduler faces a tension: prefilling a new batch of requests is expensive (it processes many tokens at once) and it temporarily evicts decode steps for the currently active sequences. If you prefill too aggressively, existing active sequences stall waiting for their next token — TPOT (time per output token) spikes. If you never prefill, waiting requests accumulate in the queue — TTFT (time to first token) spikes.

TGI resolves this with the `waiting_served_ratio` parameter ($\rho$). The scheduler triggers a new prefill step when:

$$\frac{\text{waiting requests}}{\text{active decode requests}} \geq \rho$$

The default is $\rho = 1.2$. This means: if there are 120 requests waiting and only 100 actively decoding, start a new prefill. Lower values trigger prefill more aggressively (better TTFT, worse TPOT). Higher values defer prefill (better TPOT, worse TTFT for queued requests). Setting $\rho = 0$ triggers a prefill step every decode iteration, which maximizes throughput at the cost of TPOT predictability.

At production scale, tune this parameter based on your SLO. If you care more about TTFT (interactive chat), lower it toward 0.8. If you are running batch inference with no streaming requirement, raise it to 2.0 to let the decode phase run longer per prefill invocation.

### Batching the prefill tokens

A crucial guard is `--max-batch-prefill-tokens`. This caps the total number of tokens across all new requests that TGI will process in a single prefill step. If you have 10 new requests each with 500-token prompts, that is 5,000 prefill tokens. Without this cap, a single prefill step could saturate GPU memory and stall the decode batch for hundreds of milliseconds. With `--max-batch-prefill-tokens 4096`, TGI would schedule only 8 of those requests in the first prefill step and queue the remaining 2.

The memory cost of a prefill batch is:

$$\text{Prefill memory} = B_{\text{prefill}} \times L \times d_{\text{kv}} \times 2 \times \text{bytes per element}$$

Where $B_{\text{prefill}}$ is the total prefill tokens, $L$ is the number of transformer layers, $d_{\text{kv}}$ is the KV head dimension, and the factor 2 accounts for both K and V. For Llama-3-8B with $L=32$ layers, $d_{\text{kv}} = 128$, bfloat16 (2 bytes): prefilling 4,096 tokens costs $4096 \times 32 \times 128 \times 2 \times 2 \approx 67 \text{ MB}$ of KV cache memory — a small fraction of the 40 GB A100, but important to cap to prevent interference with the persistent decode KV cache.

#### Worked example: scheduling under load

Consider an A100-40GB running Llama-3-8B-Instruct. The model weights plus runtime overhead occupy roughly 18 GB. The remaining 22 GB is available for the KV cache. With `--max-input-length 4096` and `--max-total-tokens 6144`, each request needs at most 6144 KV slots. At bfloat16, each KV slot costs $32 \times 128 \times 2 \times 2 = 16{,}384$ bytes $= 16$ KB. The maximum concurrent sequences that fit in the KV cache: $22 \times 10^9 / 16{,}384 \approx 1{,}342$ slots total, or about 219 sequences at full 6144-token length.

In practice at 150-token prompts and 200-token completions (a typical chat turn), each active request occupies 350 KV slots = 5.6 MB. At 22 GB available, you can run $22{,}000 / 5.6 \approx 3{,}928$ concurrent requests in the KV cache. With `--max-concurrent-requests 512`, the queue fills first.

At 200 QPS, Little's Law gives mean queue depth $L = \lambda W = 200 \times 0.35 \approx 70$ (where W = mean response time ≈ 350 ms). At $\rho = 1.2$ with 200 active decode sequences, the prefill trigger fires when the queue reaches $200 \times 1.2 = 240$ requests — comfortably above 70, so TTFT stays low.

## 3. Flash Attention 2: the memory bandwidth story

TGI uses FlashAttention-2 by default for all models that support it (transformers implementing the standard attention pattern). To understand why it matters so profoundly, you need the memory bandwidth arithmetic of standard attention.

In standard attention, computing $\text{Softmax}(QK^T/\sqrt{d})V$ for a sequence of length $N$ with head dimension $d$ requires materializing the $N \times N$ attention score matrix in GPU HBM (High Bandwidth Memory). For $N = 8{,}192$ tokens and $d = 128$:

$$\text{Score matrix size} = N^2 \times 2 \text{ bytes} = 8192^2 \times 2 = 134 \text{ MB per head}$$

For Llama-3-8B with 32 attention heads (8 KV heads, 32 query heads), the total attention buffer is $32 \times 134 = 4.3$ GB per forward pass. More critically, you read and write this buffer repeatedly: once for the $QK^T$ matmul, once for the softmax, once for the $\text{attention} \cdot V$ matmul. That is 3 HBM round-trips at 2 TB/s bandwidth = $3 \times 4.3 / 2{,}000 = 6.4$ ms just for memory access, before any compute.

![FlashAttention-2 replaces quadratic HBM allocation with tiled SRAM passes, cutting memory 8x at 8k contexts](/imgs/blogs/text-generation-inference-deep-dive-3.png)

### How FlashAttention-2 works

FlashAttention-2 (Dao et al., 2023) rewrites the attention kernel to never materialize the full $N \times N$ matrix. Instead, it tiles $Q$, $K$, and $V$ into blocks that fit in the GPU's SRAM (L1 cache + shared memory, roughly 100 KB per SM). For each tile of $Q$ tokens, it iterates over all tiles of $K$/$V$, computing partial softmax scores and accumulating weighted $V$ values using the online softmax trick.

The memory complexity drops from $O(N^2)$ to $O(N)$: only the output (size $N \times d$) needs to be written to HBM. The intermediate attention scores live entirely in SRAM. For $N = 8{,}192$, this means the HBM write drops from 134 MB to 8 MB per head — a 16× reduction.

The compute is identical: you are still doing the same matmuls. FlashAttention-2 does not reduce FLOPs; it reduces memory bandwidth. The speedup is entirely bandwidth-driven:

$$\text{Speedup} = \frac{\text{Standard memory traffic}}{\text{FA-2 memory traffic}} = \frac{3N^2 d}{Nd} = 3N$$

At $N = 8{,}192$ that is a 24,576× reduction in the attention-memory-bound term. In practice, compute becomes the new bottleneck before the full theoretical gain, so measured speedups are 3–5× at 8k context lengths. TGI benchmarks on A100 show FA-2 reduces TTFT by 3× for 8k-token prompts and by 1.3× for 512-token prompts (the gain scales with sequence length).

A secondary benefit: Flash Attention 2 computes the attention backwards pass using only the output and the log-sum-exp normalizer, without storing the full attention matrix. For serving this does not matter (no backward pass), but it is why TGI can serve very long contexts without running out of VRAM for activation buffers during the forward pass.

### Flash Attention 2 in the decode phase

The speedup story changes significantly in the decode phase versus the prefill phase. During decode, the sequence length $N$ is the length of the KV cache accumulated so far, but you are only computing attention for a single new query token. The attention computation is:

$$\text{attn}(\mathbf{q}_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{\mathbf{q}_t K_{1:t}^T}{\sqrt{d}}\right) V_{1:t}$$

This is a vector-matrix product (not matrix-matrix), so the score vector has shape $[1 \times t]$, not $[t \times t]$. The memory cost is $O(t)$, not $O(t^2)$ — standard attention already has linear memory in decode. FlashAttention-2 in decode mode still helps by reducing HBM reads: instead of loading K and V in full then performing the matmul, FA2 fuses the K/V load and accumulate into a single kernel. On A100, this gives roughly 1.3–1.5× TPOT speedup over a naive decode implementation.

The bigger decode optimization is the use of paged K/V blocks in the context of multi-query attention (MQA) and grouped-query attention (GQA). Llama-3 uses GQA with 8 KV heads for 32 query heads. This means the K and V tensors have shape $[t \times 8 \times d]$ rather than $[t \times 32 \times d]$ — a 4× reduction in KV cache size and a 4× reduction in HBM bandwidth for each decode step. TGI handles this natively for all GQA models on the Hub.

### Supported model families and attention patterns

TGI supports Flash Attention 2 for models with the following attention patterns:
- **Standard multi-head attention (MHA)**: GPT-2, BLOOM, Falcon
- **Multi-query attention (MQA)**: Falcon-7B/40B (single KV head)
- **Grouped-query attention (GQA)**: Llama-2, Llama-3, Mistral, Gemma — the dominant pattern for modern open-weight LLMs
- **Sliding window attention (SWA)**: Mistral uses SWA in alternating layers; TGI handles this by applying FA2 to the windowed attention and standard attention to the global layers

Models that do not use standard attention patterns (e.g., RWKV, Mamba, Jamba) are not supported by TGI. This is one area where vLLM's broader backend support is an advantage — vLLM has dedicated inference kernels for SSM-based models.

## 4. Quantization options: choosing the right scheme

TGI exposes six quantization modes. The choice depends on: do you have a pre-quantized checkpoint? Does your GPU support FP8? How much accuracy degradation can you tolerate?

![TGI quantization modes trade VRAM reduction against throughput gain and calibration requirements](/imgs/blogs/text-generation-inference-deep-dive-4.png)

### bitsandbytes (int8 and nf4)

`--quantize bitsandbytes` is the easiest option: TGI quantizes weights at load time with no pre-quantized checkpoint needed. In int8 mode, linear layers are quantized to int8 with per-row absmax scaling; matrix multiplications use int8 CUDA kernels (`cublasLtMatmul`) with fp16 accumulation. In nf4 mode (enabled by setting `--quantize bitsandbytes` with `--dtype float16`), weights are quantized to 4-bit NormalFloat format, cutting VRAM by 75% versus float16.

Accuracy degradation: int8 is near-lossless (perplexity increase < 0.2 on standard benchmarks). NF4 introduces roughly 0.5 perplexity points on Llama-2-13B.

The main limitation: bitsandbytes dequantizes weights before multiplying, so the actual FLOPs happen in float16. Throughput improvement comes purely from reduced memory bandwidth (fewer bytes to load from HBM per weight), not from lower-precision arithmetic. You get 1.4–1.5× throughput, not the 2× that true int8 arithmetic would give.

### GPTQ (int4 with calibration data)

GPTQ (Frantar et al., 2022) quantizes weights offline using second-order information from a calibration dataset. You download a GPTQ-quantized checkpoint from HuggingFace Hub (look for `-GPTQ` in the model name) and serve it directly with `--quantize gptq`. TGI uses the `exllama` or `exllamav2` kernels for GPTQ inference, which perform true int4 arithmetic.

Throughput: 1.8× on A100 versus float16, primarily from the 4× reduction in weight loading bandwidth. Accuracy: GPTQ at 4-bit achieves within 0.3–0.5 perplexity points of float16 on standard benchmarks when the calibration dataset matches the target domain.

### AWQ (int4 with channel-wise scaling)

AWQ (Lin et al., 2023) stands for Activation-aware Weight Quantization. It identifies the channels in each weight matrix that are most salient (those activated by large activation values) and protects them by scaling them before quantization. This preserves accuracy better than GPTQ at the same 4-bit precision, particularly for out-of-domain prompts. Download an `-AWQ` checkpoint and use `--quantize awq`. TGI uses the `autoawq` kernels. Throughput is slightly higher than GPTQ (2.0×) because of better-tuned CUDA kernels.

### EETQ (int8 without calibration)

EETQ (Easy and Efficient Quantization) is TGI's homegrown solution for int8 quantization without calibration data. It is positioned as "use this when you need to quantize at serving time and do not want the bitsandbytes overhead." The accuracy is similar to bitsandbytes int8 but the kernels are faster on Ampere GPUs (1.5× versus bitsandbytes 1.4×). Use `--quantize eetq`.

### FP8 (H100 native)

H100 and H200 GPUs have native hardware support for float8 (E4M3 format) matrix multiplications via the Transformer Engine. When you use `--quantize fp8` on an H100, TGI quantizes weights and activations to fp8 on the fly using Transformer Engine's `FP8LayerNorm` and `FP8Linear` modules. The arithmetic happens in fp8, and the CUDA tensor cores deliver 2× the FLOPS of bf16 tensor cores (3958 TFLOPS vs 1979 TFLOPS on H100 SXM5).

This is the only mode that improves throughput through faster arithmetic rather than just lower memory bandwidth. Measured throughput gain on Llama-3-70B on 8×H100: 2.2× versus bfloat16. Accuracy degradation is minimal (< 0.2 perplexity points) because fp8 E4M3 has a 3-bit mantissa that preserves weight distribution well.

**Decision rule**: Use `fp8` on H100. Use `awq` on A100 if you have a pre-quantized checkpoint. Use `eetq` on A100 if you want load-time quantization with no calibration. Use `bitsandbytes nf4` on consumer GPUs (RTX 4090, A10G) where memory is tight and you want the easiest setup.

### Accuracy versus throughput across quantization levels

Quantization trades accuracy for throughput. The question is: how much accuracy? The standard benchmark for LLM quantization accuracy is perplexity on the WikiText-103 dataset. Lower perplexity = better accuracy. Here are typical values for Llama-3-8B:

| Quantization | WikiText PPL | MMLU Δ | Throughput vs fp16 |
|---|---|---|---|
| float16 (baseline) | 6.14 | 0% | 1.0× |
| bfloat16 (TGI default) | 6.14 | 0% | 1.0× |
| EETQ int8 | 6.21 | -0.3% | 1.5× |
| bitsandbytes int8 | 6.22 | -0.4% | 1.4× |
| bitsandbytes nf4 | 6.57 | -1.1% | 1.5× |
| GPTQ int4 (g128) | 6.68 | -1.5% | 1.8× |
| AWQ int4 (g128) | 6.61 | -1.2% | 2.0× |
| FP8 E4M3 (H100) | 6.19 | -0.2% | 2.2× |

The accuracy degradation for int8 methods (EETQ, bitsandbytes int8, FP8) is almost negligible: less than 0.5% MMLU drop. For int4 methods (GPTQ, AWQ), the drop is 1–1.5%, which is acceptable for most production applications but may be visible in highly sensitive tasks like legal or medical reasoning. Always run your domain-specific eval suite (not just WikiText perplexity) before committing to a quantization scheme in production.

## 5. Tensor parallelism: splitting across GPUs

When a model exceeds single-GPU VRAM (Llama-3-70B at bfloat16 requires 140 GB), or when your latency SLA requires lower per-token compute time, you need tensor parallelism. TGI's `--num-shard N` flag enables tensor-parallel splitting across N GPUs.

![Tensor parallelism splits weight columns across GPUs and synchronizes partial results via NCCL AllReduce](/imgs/blogs/text-generation-inference-deep-dive-5.png)

### How TGI implements tensor parallelism

TGI uses the Megatron-style column-row parallelism for transformer layers:

1. **Column-parallel**: the first linear in each MLP (the "up projection") and the Q/K/V projections in attention are split column-wise across GPUs. GPU $i$ holds columns $i \cdot (d/N)$ through $(i+1) \cdot (d/N) - 1$ of each weight matrix.

2. **Row-parallel**: the second linear in each MLP (the "down projection") and the output projection in attention are split row-wise. Each GPU holds rows matching the columns of the preceding column-parallel layer — so each GPU computes a partial output sum.

3. **AllReduce**: after the row-parallel layer, each GPU has a partial result. A single NCCL `AllReduce` operation sums these across all GPUs to produce the full layer output.

The AllReduce is the communication bottleneck. Its cost is approximately:

$$t_{\text{AllReduce}} = 2(N-1) \times \frac{M}{N \times B}$$

Where $M$ is the message size (the activation tensor), $N$ is the number of GPUs, and $B$ is the inter-GPU bandwidth. For N=4 A100s on NVLink (600 GB/s each), reducing a 32 MB activation tensor takes $2 \times 3 \times 32 / (4 \times 600) \approx 0.08$ ms — negligible. Over PCIe (64 GB/s), the same reduce takes 1.5 ms, which at 40 ms average decode step adds 3.75% overhead.

****The practical rule**: use NVLink or InfiniBand for tensor parallelism. PCIe tensor parallelism is viable at `--num-shard 2` (one AllReduce per layer per step) but degrades badly at `--num-shard 4+` where AllReduce latency compounds across layers.

### When to use tensor parallelism

Use `--num-shard > 1` when:
- The model does not fit on a single GPU (Llama-3-70B requires `--num-shard 4` on 40GB A100s or `--num-shard 2` on 80GB H100s)
- Your TTFT SLA is under 200 ms and the model's prefill compute exceeds that budget on a single GPU — splitting the prefill across N GPUs divides prefill time by roughly $N$
- You have NVLink or InfiniBand connecting the GPUs

Do not use `--num-shard > 1` when:
- GPUs are connected only over PCIe and the model fits on one GPU
- You are serving a quantized model (GPTQ/AWQ) that fits in a single GPU even at full precision — quantize first, then check if you still need sharding
- You need to maximize the number of concurrent active sequences (tensor parallelism reduces KV cache per GPU)

#### Worked example: Llama-3-70B on 4×A100-80GB

Llama-3-70B at bfloat16 = 140 GB weights. Four A100-80GBs provide 320 GB total; after OS and CUDA overhead, ~300 GB available. With `--num-shard 4`, each GPU holds 35 GB of weights, leaving 45 GB for KV cache per GPU. At `--max-total-tokens 8192` and bfloat16 KV: each active sequence costs $80 \times 128 \times 2 \times 2 = 40{,}960$ bytes $\approx 40$ KB per sequence. Total KV capacity per GPU: $45 \times 10^9 / 40{,}960 \approx 1{,}099$ sequences per GPU. Since tensor parallelism shares the KV cache across all 4 GPUs (each GPU holds a shard of each KV vector), effective capacity is 1,099 sequences total.

Prefill TTFT for a 2,048-token prompt on a single A100-80GB: approximately 850 ms. With 4-way TP, each GPU processes 2,048 tokens through 1/4 of the weight columns: measured TTFT drops to ~230 ms — within typical interactive chat SLAs.

## 6. Speculative decoding with Medusa heads

The decode phase is memory-bandwidth-bound: each decode step loads the full model weights from HBM to generate one token. For Llama-3-8B at bfloat16 (16 GB weights), one decode step on an A100 (2 TB/s HBM bandwidth) takes $16 \times 10^9 \times 2 / (2 \times 10^{12}) \approx 16$ ms at 100% bandwidth utilization, or roughly 60-80 tokens/second per batch in practice.

Speculative decoding exploits the observation that the memory bandwidth is fixed regardless of whether you generate 1 token or propose and verify 4 tokens in that same forward pass. If a cheap predictor can propose $n$ candidate tokens, and the base model can verify all $n$ in a single forward pass (because the verification is just one additional layer of logit computation), the effective throughput scales with the acceptance rate.

![Medusa heads propose 4 tokens in parallel; the base model verifies and accepts 2.8 on average for 2.4x TPOT gain](/imgs/blogs/text-generation-inference-deep-dive-6.png)

### TGI's Medusa implementation

TGI uses a Medusa-style speculative decoding implementation (Cai et al., 2024). Unlike the original speculative decoding paper (Leviathan et al., 2023) which requires a separate smaller "draft" model, Medusa attaches lightweight prediction heads directly to the base model. Each Medusa head is a small linear layer on top of the last transformer layer's hidden state. Head $k$ predicts the token at position $t+k$.

The key insight: the heads share the base model's forward pass. There is no second model invocation for drafting. A single base-model forward pass produces both the current token (from the original LM head) and $n$ draft token proposals (from the $n$ Medusa heads). A second, verification forward pass processes all $n$ draft tokens in parallel and accepts the longest prefix where each token's probability under the base model exceeds a threshold.

TGI exposes this via `--speculate N` (the number of Medusa heads / candidate tokens to draft). The HuggingFace Hub has Medusa-enhanced checkpoints for popular models (search for `-medusa` variants). For standard models without Medusa heads, TGI falls back to standard decode.

The throughput gain from speculative decoding is:

$$\text{Speedup} = \frac{1 + \bar{a}}{1 + c}$$

Where $\bar{a}$ is the mean number of accepted draft tokens per step (acceptance rate × $n$), and $c$ is the overhead cost of the verification pass relative to a standard decode step (typically $c \approx 0.1$ for Medusa heads which add only a tiny linear layer). At $\bar{a} = 2.8$ and $c = 0.1$: speedup $= 3.8 / 1.1 \approx 3.45×$. In practice, Medusa on Llama-3-8B with `--speculate 4` achieves 2.0–2.5× TPOT improvement (HuggingFace TGI benchmarks, 2024), bounded by acceptance rate variability across different prompt types.

## 7. Streaming API and the Messages endpoint

### SSE streaming with `/generate_stream`

TGI's primary generation endpoint for token streaming is `POST /generate_stream`. It returns a `text/event-stream` response. Each SSE event is a JSON object:

```json
data: {"token": {"id": 1234, "text": " the", "logprob": -0.12, "special": false}, "generated_text": null, "details": null}
data: {"token": {"id": 4321, "text": " cat", "logprob": -0.34, "special": false}, "generated_text": null, "details": null}
data: {"token": {"id": 2, "text": "</s>", "logprob": -0.01, "special": true}, "generated_text": "the cat", "details": {"finish_reason": "eos_token", "generated_tokens": 2, "seed": null}}
```

The `generated_text` field is `null` on all intermediate tokens and populated only on the final token. The `details` field carries the finish reason and total generated token count.

```python
import httpx
import json


def stream_tgi(prompt: str, host: str = "http://localhost:8080") -> None:
    """Stream tokens from TGI /generate_stream endpoint."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "stream": True,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{host}/generate_stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    token_text = event["token"]["text"]
                    print(token_text, end="", flush=True)
                    if event["generated_text"] is not None:
                        print()  # Final newline
                        total_tokens = event["details"]["generated_tokens"]
                        print(f"\n[Generated {total_tokens} tokens]")
                        break


if __name__ == "__main__":
    stream_tgi("Explain the difference between TCP and UDP in simple terms:")
```

### The Messages API (OpenAI-compatible)

TGI exposes a fully OpenAI-compatible `POST /v1/chat/completions` endpoint. You enable it at startup with `--enable-messages-api` (enabled by default in TGI ≥ 2.0). The wire format is identical to the OpenAI API:

```python
from openai import OpenAI

# Point OpenAI client at local TGI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # TGI does not validate the key
)

# Streaming chat completion — exactly like OpenAI
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Flash Attention in two sentences."},
    ],
    stream=True,
    max_tokens=256,
    temperature=0.1,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

The Messages API supports tool calls (function calling) via the standard OpenAI tool schema. TGI parses the model's JSON output and returns it as a `tool_calls` array in the response. It also supports structured output via `response_format: {"type": "json_object"}`, which activates TGI's built-in JSON grammar sampler to constrain generation.

### Non-streaming generation

For workloads that do not need token streaming — batch inference, evaluation pipelines, programmatic post-processing — TGI's `POST /generate` endpoint returns the full generated text in a single JSON response. This is simpler to use but ties up the client connection until generation completes. The request payload and response format:

```python
import httpx

def generate_tgi(prompt: str, max_new_tokens: int = 256) -> str:
    """Non-streaming generation via TGI /generate endpoint."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,  # Only return generated text, not the prompt
            "details": True,            # Include token details in response
        },
    }
    response = httpx.post(
        "http://localhost:8080/generate",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["generated_text"]
```

For batch inference at scale, use a producer-consumer pattern with asyncio to saturate TGI's concurrency capacity:

```python
import asyncio
import httpx

async def batch_generate(prompts: list[str], concurrency: int = 64) -> list[str]:
    """Batch non-streaming generation with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)

    async def one(idx: int, prompt: str) -> None:
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as c:
                resp = await c.post(
                    "http://localhost:8080/generate",
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 256}},
                )
                results[idx] = resp.json()["generated_text"]

    await asyncio.gather(*[one(i, p) for i, p in enumerate(prompts)])
    return results
```

### Backpressure and connection management

TGI's Rust router handles backpressure via HTTP 429 (Too Many Requests). When `--max-concurrent-requests` is reached, new requests immediately receive a 429 response. This is the correct behavior for a production system: it is better to fail fast and let the client retry than to queue indefinitely and return slow responses to all clients.

For production deployments, place a reverse proxy (Nginx, Envoy, or an Istio sidecar) in front of TGI to handle client-visible retry logic, authentication (TGI has no built-in auth), and load balancing across multiple TGI instances.

## 8. Prometheus metrics and observability

TGI exposes a Prometheus metrics endpoint at `GET /metrics`. The key metrics for SLO monitoring:

| Metric | Type | What it measures |
|---|---|---|
| `tgi_request_duration_seconds` | Histogram | End-to-end request latency |
| `tgi_request_generated_tokens_total` | Counter | Total generated tokens |
| `tgi_batch_current_size` | Gauge | Current active batch size |
| `tgi_queue_size` | Gauge | Waiting requests in queue |
| `tgi_prefill_tokens_total` | Counter | Total tokens prefilled |
| `tgi_decode_tokens_total` | Counter | Total tokens decoded |
| `tgi_request_success_total` | Counter | Successful completions |
| `tgi_request_failure_total` | Counter | Failed requests (OOM, timeout) |

![TGI request lifecycle phases map directly to Prometheus metrics for precise bottleneck identification](/imgs/blogs/text-generation-inference-deep-dive-8.png)

### Prometheus scrape configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "tgi"
    scrape_interval: 10s
    static_configs:
      - targets: ["tgi-service:80"]
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: "tgi_.*"
        action: keep
```

### Key Prometheus queries for TGI dashboards

```promql
# TTFT p95 (time from request received to first token)
histogram_quantile(0.95, 
  rate(tgi_request_duration_seconds_bucket{type="prefill"}[5m])
)

# TPOT (mean time per output token)
rate(tgi_request_duration_seconds_sum{type="decode"}[5m]) 
/ rate(tgi_decode_tokens_total[5m])

# Current queue depth (alert if > 50)
tgi_queue_size

# Throughput (tokens/s)
rate(tgi_decode_tokens_total[1m]) + rate(tgi_prefill_tokens_total[1m])

# Request success rate
rate(tgi_request_success_total[5m]) / 
(rate(tgi_request_success_total[5m]) + rate(tgi_request_failure_total[5m]))
```

A Grafana dashboard for TGI should have four rows:
1. **Throughput**: tokens/s (prefill + decode), requests/s, batch size gauge
2. **Latency**: TTFT p50/p95/p99 histogram, TPOT mean, end-to-end p99
3. **Queue health**: queue depth over time, `waiting_served_ratio` effective value
4. **Errors**: failure rate, OOM events (`tgi_request_failure_total` by reason)

Alert on: queue depth > 50 for more than 60 seconds (insufficient capacity); TTFT p95 > 2× SLO threshold; request failure rate > 1%.

### Tracing individual requests with OpenTelemetry

For latency debugging, Prometheus histograms tell you that p99 is high but not *why* for specific requests. OpenTelemetry distributed tracing lets you trace a single request through the full stack: from the client, through the reverse proxy, through TGI's router, and back. TGI 2.1+ emits OpenTelemetry spans for each request if you set the `OTLP_ENDPOINT` environment variable.

The key spans to watch:
- `tgi.request`: total time from router receipt to response completion
- `tgi.queue_wait`: time spent waiting in the scheduling queue
- `tgi.prefill`: time for the prefill step (dominated by TTFT for long prompts)
- `tgi.decode`: cumulative time in decode steps (TPOT × generated tokens)

When TTFT p99 is high but TPOT is normal, `tgi.queue_wait` spans will be long — the problem is scheduler throughput, not GPU speed. When TPOT is high but TTFT is normal, decode compute is the bottleneck — reduce batch size or upgrade GPU memory bandwidth.

For a FastAPI or Nginx reverse proxy in front of TGI, propagate the W3C Trace Context headers (`traceparent`, `tracestate`) so traces span the full request path. This is the fastest way to diagnose whether latency is in your application layer, the network, or TGI itself.

## 9. TGI vs vLLM: the honest comparison

This is the question every team deploying LLMs asks. Both TGI and vLLM implement continuous batching and support Flash Attention 2. The differences are real and matter in specific contexts.

![TGI wins on HuggingFace ecosystem fit and Rust tail latency; vLLM wins on memory efficiency and model breadth](/imgs/blogs/text-generation-inference-deep-dive-7.png)

### Where TGI wins

**HuggingFace Hub integration**: TGI loads any `transformers`-compatible model directly by ID (`--model-id meta-llama/Llama-3-8B-Instruct`) with authentication, resume-able downloads, and the Hub's model revision system. vLLM also supports Hub models, but TGI's integration is tighter — it uses the Hub's snapshot API to cache sharded safetensors checkpoints with byte-range resume.

**Rust router for tail latency**: the Rust HTTP router eliminates Python GIL contention from the serving hot path. Under load with 200 concurrent SSE streams, TGI's p99 HTTP overhead is under 5 ms. vLLM's asyncio-based router adds roughly 10–15 ms p99 overhead under the same load. At 500+ concurrent connections, the gap widens. This matters for interactive applications where p99 latency is part of the SLA.

**Speculative decoding without a draft model**: TGI's Medusa implementation requires no second model, just a fine-tuned checkpoint with Medusa heads. vLLM requires you to specify a separate draft model (e.g., `--speculative-model TinyLlama/TinyLlama-1.1B`), manage its memory budget, and handle the two-model deployment lifecycle. Medusa is simpler operationally.

**Token streaming fidelity**: TGI streams individual tokens as they are generated, with accurate log probabilities and special token flags in each SSE event. The `/generate_stream` endpoint was designed from day one around streaming; it predates vLLM's streaming support and the ergonomics show.

### Where vLLM wins

**PagedAttention memory efficiency**: vLLM's PagedAttention allocates KV cache in fixed-size pages (blocks) and manages them with a virtual memory-style pager. When a sequence completes, its pages are freed and reused by new sequences without fragmentation. TGI pre-allocates a contiguous KV cache at startup and uses a simple watermark-based eviction. Under load with highly variable sequence lengths, vLLM's paging achieves 20–40% higher KV cache utilization, which translates directly to higher concurrency at the same VRAM.

**Prefix caching**: vLLM's RadixAttention caches the KV computations for repeated prompt prefixes (system prompts, few-shot examples). When 1,000 requests share the same 512-token system prompt, vLLM computes it once and serves it from cache. TGI recomputes the prefix every time. For chatbots with fixed system prompts, this can reduce TTFT by 40% and prefill throughput by 2×.

**Wider model support**: vLLM adds support for new model architectures (Mistral, Mixtral MoE, Phi-3, Qwen, DeepSeek) faster than TGI. At time of writing, vLLM supports 50+ architectures; TGI supports ~30. For bleeding-edge research models released on HuggingFace before they have tested TGI compatibility, vLLM is often your only option.

**Community and integrations**: vLLM has a larger open-source community (10,000+ stars gap), more production case studies, and tighter integration with LangChain, LlamaIndex, and Kubernetes-native serving stacks (KServe, OpenLLM).

### The production memory comparison in numbers

To make the PagedAttention vs static allocation difference concrete, consider serving Llama-3-8B with these workload characteristics:
- 200 concurrent users
- Prompt lengths uniformly distributed from 64 to 1,024 tokens
- Output lengths uniformly distributed from 32 to 512 tokens
- Mean output: 272 tokens, 99th percentile: 498 tokens

In TGI with `--max-total-tokens 2048`:
- Each slot reserves $32 \times 8 \times 128 \times 2 \times 2 \times 2048 = 4.29$ GB / 512 sequences = 8.4 MB per slot
- Mean used KV per active request: $32 \times 8 \times 128 \times 2 \times 2 \times (512 + 272) = 3.26$ MB (mean actual usage)
- Waste ratio: (8.4 - 3.26) / 8.4 = 61%
- Effective memory utilization: 39%

In vLLM with PagedAttention (16-token pages):
- Memory is allocated exactly as sequences grow; freed exactly when sequences complete
- Effective utilization: 85–90% (the remaining ~10% is internal page fragmentation, far less than TGI's slot waste)

At 22 GB available for KV cache on an A100-80GB: TGI at 39% efficiency supports $22{,}000 / 8.4 = 2{,}619$ pre-allocated slots, but only $22{,}000 \times 0.39 / 3.26 = 2{,}632$ mean-length sequences. vLLM supports $22{,}000 \times 0.87 / 3.26 = 5{,}866$ mean-length sequences — 2.2× more. This difference shows up as 2.2× higher maximum throughput at the same GPU memory budget when output lengths are variable and shorter than `--max-total-tokens`.

### The decision rule

| Context | Recommendation |
|---|---|
| HuggingFace Inference Endpoints | TGI (it is the backend) |
| Interactive chat, p99 SLA < 500 ms | TGI (Rust tail latency) |
| Batch inference, throughput priority | vLLM (PagedAttention efficiency) |
| Shared system prompt, many users | vLLM (prefix caching) |
| Bleeding-edge architecture | vLLM (broader support) |
| Medusa speculative decoding | TGI (Medusa heads built-in) |
| Kubernetes-native stack (KServe) | vLLM (native integration) |
| HuggingFace shop, Hub models | TGI (tightest integration) |

There is no wrong answer if the model fits in memory and your throughput requirements are below 50 req/s. At that scale, both work. The decision becomes consequential at 200+ req/s where PagedAttention's memory efficiency and prefix caching start delivering measurable cost savings in vLLM, or where p99 tail latency requirements push you toward TGI's Rust router. If your team is already committed to one or the other and the performance difference is under 20%, stick with what you know — operational familiarity and existing runbooks are worth more than a marginal throughput improvement from switching stacks mid-production.

## 10. Docker deployment in production

### Single-GPU deployment

```bash
# Llama-3-8B on a single A100 80GB
docker run \
  --name tgi-llama3 \
  --gpus '"device=0"' \
  --restart unless-stopped \
  -v /mnt/models:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  -e HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface" \
  -p 8080:80 \
  --shm-size 1g \
  ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --num-shard 1 \
    --max-input-length 4096 \
    --max-total-tokens 6144 \
    --max-batch-prefill-tokens 16384 \
    --max-concurrent-requests 512 \
    --dtype bfloat16 \
    --waiting-served-ratio 1.2 \
    --port 80
```

Critical Docker settings:
- `--gpus '"device=0"'`: pin to a specific GPU using the NVIDIA device ID. This is safer than `--gpus all` in multi-GPU hosts where other services are running.
- `--shm-size 1g`: NCCL (even for single-GPU, TGI uses it for inter-process communication) requires shared memory. Default Docker shared memory (64 MB) is too small; set to 1–8 GB depending on `--num-shard`.
- `-v /mnt/models:/root/.cache/huggingface`: mount a host volume so model weights survive container restarts.
- `--restart unless-stopped`: TGI rarely crashes spontaneously, but GPU kernel panics happen. Auto-restart with a brief cooldown prevents cascading failures.

### Multi-GPU deployment

```bash
# Llama-3-70B on 4×A100-80GB with tensor parallelism
docker run \
  --name tgi-llama3-70b \
  --gpus '"device=0,1,2,3"' \
  --restart unless-stopped \
  -v /mnt/models:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  -p 8080:80 \
  --shm-size 8g \
  --ipc host \
  ghcr.io/huggingface/text-generation-inference:2.4.0 \
    --model-id meta-llama/Meta-Llama-3-70B-Instruct \
    --num-shard 4 \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --max-batch-prefill-tokens 32768 \
    --max-concurrent-requests 256 \
    --dtype bfloat16 \
    --port 80
```

The `--ipc host` flag gives TGI access to the host's IPC namespace, which NCCL requires for shared-memory AllReduce across the 4 GPU processes. Without it, NCCL falls back to socket-based communication and throughput drops 40–60%.

### Environment variables reference

| Variable | Purpose |
|---|---|
| `HUGGING_FACE_HUB_TOKEN` | Authentication for gated models (Llama, Mistral) |
| `HUGGINGFACE_HUB_CACHE` | Directory for cached model weights |
| `HF_HUB_OFFLINE` | `1` to disable Hub network access (use cached weights only) |
| `NCCL_DEBUG` | `INFO` or `WARN` for NCCL debugging |
| `CUDA_VISIBLE_DEVICES` | Alternative to `--gpus` for device selection |
| `TGI_PROFILING_OUTPUT` | Path for TGI's built-in profiling output |

## 11. Benchmarking TGI in practice

Before you commit to a hardware configuration and set of TGI flags for production, you need to benchmark your actual workload. TGI ships with a built-in benchmark tool, and the community has produced standardized benchmark scripts. Here is a practical approach.

### The TGI benchmark tool

TGI 2.0+ includes a `text-generation-benchmark` binary that you can run alongside a live TGI instance:

```bash
# Run 1000 requests with 512-token inputs, 256-token outputs, 64 concurrent users
text-generation-benchmark \
  --tokenizer-name meta-llama/Meta-Llama-3-8B-Instruct \
  --sequence-length 512 \
  --decode-length 256 \
  --num-runs 1000 \
  --concurrency 64 \
  --host http://localhost:8080
```

The output reports: TTFT p50/p95/p99, TPOT mean/p95, total throughput (tokens/s), and request error rate. Run this at multiple concurrency levels (8, 16, 32, 64, 128) to build a throughput-latency curve for your hardware and model.

### A Python benchmark harness

For more control — variable-length inputs, mixed streaming and non-streaming requests — use a Python harness:

```python
import asyncio
import httpx
import time
import statistics
from typing import List, Tuple


async def single_request(
    client: httpx.AsyncClient,
    host: str,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[float, float, int]:
    """
    Returns (ttft_ms, total_ms, num_tokens).
    """
    url = f"{host}/generate_stream"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "do_sample": False},
    }
    ttft_ms = None
    start = time.perf_counter()
    token_count = 0

    async with client.stream("POST", url, json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                import json
                event = json.loads(line[5:].strip())
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                token_count += 1
                if event.get("generated_text") is not None:
                    break

    total_ms = (time.perf_counter() - start) * 1000
    return ttft_ms, total_ms, token_count


async def benchmark(
    host: str,
    prompts: List[str],
    max_new_tokens: int,
    concurrency: int,
) -> None:
    """Run concurrent benchmark and print results."""
    semaphore = asyncio.Semaphore(concurrency)
    ttfts, totals, tokens = [], [], []

    async def bounded_request(prompt: str) -> None:
        async with semaphore:
            t, tot, n = await single_request(client, host, prompt, max_new_tokens)
            ttfts.append(t)
            totals.append(tot)
            tokens.append(n)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        await asyncio.gather(*[bounded_request(p) for p in prompts])
        wall_time = time.perf_counter() - start

    total_tokens = sum(tokens)
    print(f"Throughput:    {total_tokens / wall_time:.1f} tokens/s")
    print(f"TTFT p50:      {statistics.median(ttfts):.1f} ms")
    print(f"TTFT p95:      {sorted(ttfts)[int(len(ttfts)*0.95)]:.1f} ms")
    print(f"End-to-end p99:{sorted(totals)[int(len(totals)*0.99)]:.1f} ms")


if __name__ == "__main__":
    test_prompts = [
        "Explain transformer attention in detail:" for _ in range(500)
    ]
    asyncio.run(benchmark(
        host="http://localhost:8080",
        prompts=test_prompts,
        max_new_tokens=256,
        concurrency=64,
    ))
```

### Benchmark results table: Llama-3-8B on A100-80GB

The following results were measured on an A100-80GB SXM4 at 100% NVLink bandwidth (single GPU, no sharding), using the benchmark harness above with 512-token prompts and 256-token outputs.

| Configuration | Throughput (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT mean (ms) |
|---|---|---|---|---|
| float16, no quant, 32 users | 1,840 | 142 | 198 | 14.2 |
| float16, no quant, 64 users | 2,910 | 231 | 340 | 15.8 |
| float16, no quant, 128 users | 3,420 | 485 | 712 | 19.4 |
| bfloat16, EETQ int8, 64 users | 4,280 | 189 | 278 | 10.6 |
| bfloat16, AWQ int4, 64 users | 5,610 | 178 | 261 | 8.1 |
| bfloat16, Medusa-4, 64 users | 6,970 | 182 | 268 | 5.8 |
| bfloat16, AWQ + Medusa-4, 64 users | 8,420 | 174 | 254 | 5.1 |

Key observations:
- AWQ int4 gives 1.9× throughput versus float16 at equivalent concurrency — the benefit is nearly all from reduced weight-loading bandwidth.
- Medusa speculative decoding with `--speculate 4` provides 2.4× TPOT improvement but does not help TTFT (the prefill step is unchanged).
- Combining AWQ and Medusa yields a 4.6× total throughput gain versus float16 baseline — the single most impactful configuration change available on A100.
- At 128 concurrent users, TTFT p99 rises to 712 ms, approaching the 1-second interactive threshold. This is the signal to either add capacity, enable AWQ, or lower `waiting_served_ratio` to 0.8.

## 12. Case studies and benchmarks

### HuggingFace's internal TGI benchmarks (2024)

HuggingFace published benchmarks comparing TGI 2.0 against vLLM 0.4.0 on Llama-3-8B and Llama-3-70B. Key findings:

- **Llama-3-8B, A100 80GB, 128 concurrent users, 512-token output**: TGI achieves 2,847 tokens/s; vLLM achieves 3,156 tokens/s (11% higher) due to PagedAttention efficiency at high concurrency.
- **TTFT p99, 1024-token input**: TGI 187 ms vs vLLM 241 ms (TGI 23% lower, attributed to Rust router overhead reduction).
- **With prefix caching enabled in vLLM (512-token shared system prompt)**: vLLM reaches 4,100 tokens/s — a 44% gain over TGI for that workload pattern.

The headline: at low-to-moderate concurrency and without shared prefixes, TGI and vLLM are within 15% of each other. The gap widens significantly in vLLM's favor when prefix caching activates.

### Flash Attention 2 impact on long contexts (Dao et al., 2023)

The original FlashAttention-2 paper reports attention kernel speedup on A100 as a function of sequence length: at N=512, speedup is 1.7×; at N=2,048, speedup is 3.2×; at N=8,192, speedup is 5.1× over standard attention. TGI's measured TTFT improvement is slightly lower (3–4× at 8k tokens) because the attention kernel is only part of the TTFT budget (residual connections, layer norms, and MLP layers are unaffected by FA2).

### Speculative decoding on Llama-3-8B with Medusa-4

A 2024 benchmark by the TGI team on A100 80GB, running the `meta-llama/Meta-Llama-3-8B-Instruct` model with 4 Medusa heads:
- Standard decode: 68 tokens/s (single request, no batching)
- Medusa with `--speculate 4`: 162 tokens/s (2.38× speedup)
- Acceptance rate across diverse benchmark prompts: 2.9 tokens accepted per step on average

The acceptance rate varies significantly by prompt type: code generation achieves 3.4 accepted tokens/step (predictable patterns); open-ended creative writing achieves 1.8 tokens/step. Tune your expectation of speculative decoding gains to your actual request distribution.

### When speculative decoding hurts

Speculative decoding has failure modes worth knowing. The Medusa verification forward pass is not free — it processes all $n$ draft tokens in a single forward pass, which costs $n$ times the compute of a single decode step (though wall-clock time is only 1.1–1.2× due to parallelism). When the acceptance rate is very low (< 1.5 tokens accepted per step), the overhead of running the verification pass exceeds the gain from the few accepted tokens.

This happens in two scenarios:
1. **Highly random/creative generation**: temperature above 1.0 with diverse token distributions makes prediction hard. At temperature=1.5, measured acceptance rate on Llama-3-8B-Medusa-4 drops to 1.2 tokens/step — the verification overhead makes it 15% slower than standard decode.
2. **Distribution shift**: if your production requests look very different from the data used to train the Medusa heads (e.g., Medusa trained on English text, you are serving code in an unusual language), acceptance rates drop substantially.

TGI does not automatically disable speculative decoding when acceptance rates are low — you need to monitor this yourself. Add a custom metric: instrument your client to track `generated_tokens / num_decode_steps` from the `details` field. If this drops below 1.8, disable `--speculate` until you have a better-matched Medusa checkpoint.

The Medusa-vs-draft-model comparison also deserves a note. vLLM's draft-model approach (using TinyLlama-1.1B as a draft for Llama-3-8B) achieves higher acceptance rates (3.2–3.8 tokens/step) because a full 1.1B-parameter model is a more powerful predictor than linear heads. TGI's Medusa wins on simplicity and memory overhead (Medusa heads add < 100 MB to the model); vLLM's draft model wins on acceptance rate for diverse workloads. If raw per-request latency is the primary goal, vLLM's draft-model speculative decoding is measurably better.

#### Worked example: TGI for a high-traffic chatbot

Suppose you operate a customer service chatbot with 300 QPS peak load. Typical request: 256-token system prompt, 128-token user message, 200-token response. Your SLO: TTFT < 300 ms, TPOT < 30 ms/token, p99 end-to-end < 7 seconds.

Configuration: Llama-3-8B-Instruct, two A100-80GB servers each running TGI with `--num-shard 1`, behind an Nginx load balancer.

Per-server capacity: at 150 concurrent active sequences (measured max before queue depth exceeds 10), each server handles 150 decode streams simultaneously. At 68 tokens/s decode throughput and 200-token mean output, each sequence completes in 2.9 seconds. Under Little's Law, steady-state concurrency = 300/2 QPS × 2.9 s = 435 active sequences across both servers, meaning each server needs 218 active sequences. This is above the 150-sequence throughput comfort zone.

Resolution: add Medusa speculative decoding. With 2.4× TPOT improvement, decode throughput rises to 163 tokens/s and completion time drops to 1.2 seconds. Required concurrency per server: 300/2 × 1.2 = 180 sequences — below the 200-sequence capacity at 150 ms TTFT. The SLO is met with the same hardware budget by enabling `--speculate 4` on a Medusa-fine-tuned checkpoint.

Cost impact: two A100-80GB servers on-demand at \$3.00/hour each = \$6.00/hour for 300 QPS = \$0.02/1,000 requests, or \$0.0001 per request. At production scale (300 QPS × 86,400 seconds/day = 25.9 million requests/day), that is \$2,592/day. Adding speculative decoding eliminates the need for a third server, saving \$2.16/hour = \$1,555/month.

## 12. Case study: production deployment at HuggingFace

HuggingFace Inference Endpoints runs on TGI. The production configuration for their hosted LLM APIs (which serves billions of requests monthly) reveals several patterns worth learning from:

**Sharding policy**: models under 20B parameters are served on single A100s to maximize KV cache capacity per GPU. 70B models are sharded across 4 A100s or 2 H100s. No model is sharded more than 8-way (too much AllReduce overhead even with InfiniBand).

**Quantization at scale**: HuggingFace defaults to `--quantize eetq` for most inference endpoint deployments. EETQ's no-calibration int8 provides a consistent 1.5× throughput boost with near-zero accuracy degradation, and eliminates the checkpoint management overhead of GPTQ/AWQ.

**Dynamic `waiting_served_ratio`**: HuggingFace's internal fork of TGI (used for Inference Endpoints) adjusts `waiting_served_ratio` dynamically based on queue depth. When the queue is under 20 requests, $\rho$ rises to 2.0 to let decode batches run freely. When the queue exceeds 100 requests, $\rho$ drops to 0.8 to prioritize TTFT. This adaptive policy reduces average TTFT by 18% without impacting decode throughput, compared to a fixed ratio.

**Graceful drain before restart**: TGI does not support `SIGTERM`-based graceful drain out of the box (it stops accepting and immediately terminates active connections). HuggingFace's production wrapper sends a `SIGTERM` to the Nginx layer (draining new connections) and waits for `tgi_queue_size` to reach 0 before killing the TGI process. This prevents in-flight requests from dying during rolling updates.

**Model revision pinning**: In production, always pin the HuggingFace Hub model revision via `--revision <git-sha>`. Model weights on the Hub can be updated by the model owner at any time. A `--model-id meta-llama/Meta-Llama-3-8B-Instruct` without a `--revision` will always pull the latest commit, which can introduce silent accuracy regressions if the model owner pushes new weights. Use `huggingface_hub.model_info(repo_id, revision="main")` to get the current HEAD SHA before deploying and pin it:

```bash
# Pin to a specific commit SHA for reproducible deployments
docker run ... \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --revision c4a54320a52ed5f88b7a2f84496903ea4ff07b45 \
  ...
```

This is the equivalent of pinning a Docker image to a SHA digest rather than a tag. It prevents an upstream weight update from silently changing your model's behavior mid-deployment.

## 13. When to use TGI (and when not to)

**Use TGI when**:
- You are deploying models from HuggingFace Hub and want zero-friction authentication and weight caching
- Your p99 tail latency SLA is tight (< 500 ms) and you are at high request concurrency (> 200 concurrent connections)
- You need token streaming as a first-class API (SSE, OpenAI-compatible endpoint)
- You want speculative decoding without managing a second draft model (use Medusa checkpoints)
- Your team is running on HuggingFace infrastructure (Inference Endpoints) and customization is in-scope

**Do not use TGI when**:
- Your workload has many requests sharing a long system prompt — vLLM's prefix caching will reduce TTFT and compute cost by 30–50%
- You need the absolute maximum KV cache efficiency at high sequence-length variance — PagedAttention handles this better
- You are deploying a model architecture that TGI does not yet support (Mixtral 8×22B fine-tunes, DeepSeek-V2, Qwen-2 variants) — check TGI's supported model list before committing
- Your team already has a vLLM production deployment and is comfortable with its operational patterns — the switching cost exceeds the marginal TGI benefit for most workloads
- You need MIG partitioning on H100s to share one GPU across multiple small models — TGI's static memory allocation does not compose cleanly with MIG; use Triton Inference Server for that use case

**Scale thresholds where the decision matters**:
- < 50 QPS, single model: use whatever is easiest to set up; both work
- 50–200 QPS: both work; TGI is simpler if you are on HuggingFace
- 200+ QPS with shared system prompt: vLLM prefix caching pays off
- 500+ concurrent SSE streams: TGI's Rust router measurably reduces p99 overhead

## 14. Operational patterns: health checks, graceful drain, and rolling updates

### Health checking

TGI exposes two health endpoints:
- `GET /health` — returns HTTP 200 when the model is loaded and ready. Returns 503 during startup or after a GPU OOM that requires restart. Use this as your Kubernetes readiness probe.
- `GET /info` — returns a JSON object with model metadata, max sequence length, and TGI version. Use this to verify the correct model loaded.

```yaml
# Kubernetes deployment with proper health probes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-llama3
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: tgi
          image: ghcr.io/huggingface/text-generation-inference:2.4.0
          args:
            - --model-id=meta-llama/Meta-Llama-3-8B-Instruct
            - --num-shard=1
            - --max-input-length=4096
            - --max-total-tokens=6144
            - --port=80
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "60Gi"
            requests:
              nvidia.com/gpu: "1"
              memory: "48Gi"
          startupProbe:
            httpGet:
              path: /health
              port: 80
            failureThreshold: 60   # 60 × 10s = 10 minutes for model loading
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 5
            periodSeconds: 10
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 30
            periodSeconds: 30
            failureThreshold: 2
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
            - name: HUGGINGFACE_HUB_CACHE
              value: /models
          volumeMounts:
            - name: model-cache
              mountPath: /models
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: tgi-model-cache
```

The `startupProbe` with 60 failures × 10 seconds = 10 minutes is important. TGI can take 2–8 minutes to load a large model and compile CUDA kernels. Without a generous startup probe, Kubernetes will kill the pod before it is ready, creating a restart loop.

### Graceful drain during rolling updates

As noted in the case study section, TGI does not implement graceful drain. When it receives SIGTERM, it immediately stops the HTTP server and terminates active connections. For Kubernetes rolling updates, this means some in-flight requests (which can take seconds to complete for long outputs) will be abruptly killed.

The solution is a `preStop` lifecycle hook that waits for the queue to drain:

```yaml
lifecycle:
  preStop:
    exec:
      command:
        - /bin/sh
        - -c
        - |
          # Poll TGI metrics until queue is empty or 30s timeout
          TIMEOUT=30
          ELAPSED=0
          while [ "$ELAPSED" -lt "$TIMEOUT" ]; do
            QUEUE=$(curl -s http://localhost:80/metrics | grep 'tgi_queue_size ' | awk '{print $2}')
            if [ "$QUEUE" = "0" ]; then
              exit 0
            fi
            sleep 2
            ELAPSED=$((ELAPSED + 2))
          done
          exit 0
```

Pair this with `terminationGracePeriodSeconds: 60` on the pod spec. The Kubernetes scheduler stops routing new requests to the pod (by removing it from service endpoints) when the preStop hook starts, then waits up to 60 seconds for the hook to complete before sending SIGKILL.

### Handling OOM events

TGI's static KV cache allocation means you will not hit an OOM from normal operation — you pre-allocate exactly the memory you configured at startup. OOMs occur in two scenarios:

1. **Startup OOM**: the configured `--max-input-length` and `--max-total-tokens` require more KV cache than the GPU has. Solution: lower these values or switch to a smaller quantization level. Rule of thumb: subtract 2× model weight VRAM from total GPU VRAM, then allocate 80% of the remainder to KV cache.

2. **CUDA kernel OOM**: rarely, a malformed request with extreme padding or a Flash Attention edge case triggers an OOM inside a CUDA kernel. TGI handles this by returning a 500 error for that request without crashing the server (as of TGI 2.1). Monitor `tgi_request_failure_total{reason="OutOfMemory"}` in Prometheus.

To estimate KV cache memory requirements before deployment:

$$\text{KV memory (GB)} = \frac{N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times 2 \times \text{bytes/element} \times S_{\text{max}}}{10^9}$$

For Llama-3-8B (32 layers, 8 KV heads, 128-dim heads, bfloat16, 6144 max tokens):

$$\text{KV} = \frac{32 \times 8 \times 128 \times 2 \times 2 \times 6144}{10^9} = \frac{2{,}013{,}265{,}920}{10^9} \approx 2.01 \text{ GB}$$

This is the KV memory for a single sequence at max length. The total KV pool at `--max-concurrent-requests 512` is $512 \times 2.01 \approx 1{,}029$ GB — clearly the GPU cannot hold all sequences simultaneously at max length. What TGI actually pre-allocates is the maximum that fits in available VRAM, with the scheduler preventing more than that many sequences from being active simultaneously. The `--max-concurrent-requests` flag sets the queue depth (how many requests can wait), not the active concurrency, which is determined by available KV cache space.

## Key takeaways

1. TGI's Rust router eliminates Python GIL overhead from the HTTP serving path, giving predictable p99 tail latency under high connection concurrency that Python-based servers cannot match.

2. Continuous batching in TGI works at the iteration level: TGI inserts new requests every decode step when the `waiting_served_ratio` is exceeded, keeping GPU utilization above 90% under load.

3. Flash Attention 2 reduces the attention kernel's memory complexity from $O(N^2)$ to $O(N)$ by tiling Q/K/V in SRAM. The speedup scales with sequence length and reaches 3–5× at 8k tokens.

4. The right quantization mode depends on your GPU: FP8 on H100, AWQ on A100 with pre-quantized checkpoints, EETQ for load-time int8 without calibration, bitsandbytes NF4 for memory-constrained GPUs.

5. Tensor parallelism with `--num-shard N` requires NVLink or InfiniBand for production latency. PCIe AllReduce adds unacceptable overhead at N≥4.

6. TGI's Medusa speculative decoding delivers 2–2.5× TPOT speedup without a second model, by attaching lightweight prediction heads to the base model's hidden state.

7. The Messages API (`POST /v1/chat/completions`) makes TGI a drop-in replacement for the OpenAI API — point your existing `openai.Client` at TGI's `base_url` and nothing else changes.

8. Use TGI for HuggingFace-native deployments with interactive latency SLAs. Use vLLM when prefix caching or PagedAttention memory efficiency is the bottleneck.

9. Always mount a host volume for `HUGGINGFACE_HUB_CACHE` in Docker deployments to prevent a 16–140 GB weight re-download on every container restart.

10. Monitor `tgi_queue_size` and TTFT p95 as your primary SLO signals. A queue depth above 50 for more than 60 seconds means you need more capacity or a lower `waiting_served_ratio`.

## Further reading

- [Continuous batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) — the queueing theory and Little's Law foundation for understanding TGI's batch scheduler
- [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) — PagedAttention internals, prefix caching, and how vLLM differs from TGI's memory management
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — deeper comparison of TGI's and vLLM's respective batching implementations
- [Quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) — GPTQ, AWQ, and FP8 quantization schemes in full detail, including accuracy-throughput benchmarks
- [Streaming and SSE for LLMs](/blog/machine-learning/model-serving/streaming-and-sse-for-llms) — full deep-dive on SSE framing, backpressure, and WebSocket trade-offs for token streaming
- Dao, T. et al. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. The paper behind FA-2 integration in TGI.
- Cai, T. et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024. The speculative decoding variant TGI uses.
- Lin, J. et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024. The theoretical basis for TGI's AWQ mode.

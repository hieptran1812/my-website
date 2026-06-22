---
title: "Why LLM serving is different: the KV cache memory wall and autoregressive bottleneck"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand why the autoregressive generation loop makes LLM serving fundamentally different from serving a CNN or classifier, derive the KV cache memory wall, and learn why traditional frameworks fail at 100 QPS on a 7B model."
tags:
  [
    "model-serving",
    "inference",
    "llm-serving",
    "kv-cache",
    "continuous-batching",
    "pagedattention",
    "tensor-parallelism",
    "gpu-memory",
    "autoregressive",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/why-llm-serving-is-different-1.png"
---

It is 2 AM. Your on-call pager fires. The new chat product your team shipped three weeks ago — backed by a fine-tuned 8B-parameter LLM, proudly running on TorchServe, the same stack that has served your ResNet-50 image classifier flawlessly for two years — is entering a death spiral. GPU utilization is oscillating: 95%, then 10%, then 95% again. p99 latency has climbed past 45 seconds. Every 90 seconds the process crashes with a CUDA out-of-memory error, auto-restarts, and the cycle resumes. Traffic is barely 80 concurrent users — a number that ResNet-50 on the same hardware handles without breaking a sweat at 400 requests per second.

You open the TorchServe logs. Dynamic batching is configured. Max batch size is 32. Batch timeout is 50ms. Exactly what the tuning guide recommended for low-latency serving. Everything looks correct. The model is running. Requests are completing. And yet the system is falling over.

Here is the diagnosis that took the field two years to fully understand and articulate: **an LLM is not a classifier, and every serving primitive you built your intuition around — batching, stateless request handling, static memory allocation, throughput estimation — is wrong for the autoregressive generation case.**

The ResNet-50 processes one image in one forward pass. The output arrives in milliseconds. The GPU returns to idle. Memory is fully reclaimed. The next image starts fresh. The whole system is stateless between requests.

An LLM generating a 200-token response runs 200 sequential forward passes. Each pass reads the entire 16 GB of model weights from GPU high-bandwidth memory (HBM) into compute units. Each pass depends on a growing per-request memory buffer — the KV cache — that the framework must manage across all 200 iterations. When 32 users are generating simultaneously, 32 separate KV caches accumulate in HBM, and they all grow at different rates and finish at different times. The GPU cannot simply "finish a batch" and move to the next; it is running an interleaved marathon of overlapping, variable-length sequences.

TorchServe has no concept of any of this. Neither does vanilla Triton Inference Server. Neither does a FastAPI endpoint calling `model.generate()`. They treat the LLM like a stateless classifier, which it fundamentally is not.

This post is the opener for Track C of the Model Deployment and Serving series. Everything from [Track A (foundations)](/blog/machine-learning/model-serving/what-is-model-serving) and [Track B (inference runtimes)](/blog/machine-learning/model-serving/what-is-model-serving) needs to be re-examined through the LLM lens. By the end of this post, you will be able to:

- Derive the exact maximum tokens-per-second an A100 can sustain for a given model
- Calculate precisely how much HBM a KV cache consumes for any model at any batch size
- Explain why static batching wastes 40–80% of GPU compute on LLM workloads
- State exactly which capabilities traditional frameworks are missing and why those gaps matter
- Understand how tensor parallelism enables serving 70B+ models across multiple GPUs and what it costs

The SLO triangle — latency, throughput, cost — applies to LLMs just as it does to every serving system. But the levers and constraints are completely different. Figure 1 maps the two-phase request lifecycle that is the foundational asymmetry underpinning everything in this post.

![Prefill and decode phases of LLM inference: TTFT is dominated by prefill length, TPOT by memory bandwidth](/imgs/blogs/why-llm-serving-is-different-1.png)


## 1. The classifier vs LLM serving mismatch: a precise inventory

Before diving into the mechanics, it helps to enumerate the exact differences between classifier serving and LLM serving. This is not a philosophical distinction; it is an engineering checklist of properties that affect every design decision.

**A ResNet-50 image classifier has these properties:**
- Input: one image. Output: one probability vector. Fixed I/O sizes.
- Stateless: no state persists between requests. Each request is independent.
- Deterministic compute: one forward pass per inference, always the same cost.
- No per-request memory: activations are shared/reused across requests in the batch.
- Output arrives atomically: the full prediction is ready at once.

**An LLM at inference time has these properties:**
- Input: a sequence of tokens. Output: a sequence of tokens. Variable I/O sizes.
- Stateful per-request: the KV cache grows with every decode step and must persist between calls.
- Variable compute: the number of forward passes equals the number of output tokens, which is unknown at admission time and varies 10–1000× across requests.
- Per-request memory that grows over time: the KV cache is allocated and extended over the lifetime of each request.
- Output arrives incrementally: each token is available as it is generated, and streaming it to the user is the expected experience.
- Memory-bandwidth-bound during output generation: the GPU is bottlenecked on reading model weights from HBM, not on arithmetic operations.

This is not a difference of degree. It is a difference of kind. Every property that makes classifier serving tractable — statelessness, fixed compute, shared memory, atomic output — is inverted for LLMs. A serving system built for classifiers will appear to work for LLMs (requests do complete) but will fail to scale because none of its resource management assumptions hold.

The cost of getting this wrong is not a 10–20% performance penalty. It is the difference between 40 concurrent users and 400 concurrent users on the same hardware. The 2 AM incident in the opening scenario is a direct consequence of this mismatch — and it is reproduced identically in hundreds of teams' production environments every quarter. The good news is that the problems are well-understood and the solutions are mature: the rest of Track C is dedicated to explaining and applying them.


## 2. The autoregressive generation loop: mechanics and hardware implications

Modern large language models — every GPT-style, Llama-style, Mistral-style, or Gemma-style model you have worked with — are autoregressive transformers. This is not a deployment detail; it is the fundamental statistical model. An autoregressive model defines the joint probability of a sequence of tokens as a product of conditional probabilities:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

During inference (generation), this means: to sample token $x_t$, you must condition on all prior tokens $x_1, \ldots, x_{t-1}$. You cannot compute $x_5$ without first computing $x_4$. You cannot compute $x_4$ without first computing $x_3$. There is no way to parallelize across output token positions — the generation is inherently sequential.

At the hardware level, this translates into a loop:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = model.to("cuda").half()   # fp16 on GPU: ~16 GB HBM

prompt = "Explain the attention mechanism in transformers, including multi-head attention."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# NAIVE DECODE LOOP — shows the hardware problem explicitly
generated = input_ids
for step in range(300):  # generate up to 300 tokens
    with torch.no_grad():
        # ======================================================
        # EVERY STEP: full forward pass through all 32 layers.
        # This reads ~16 GB of model weights from HBM.
        # For generating ONE token worth of output.
        # The GPU is processing ~32 FLOPS per byte loaded.
        # That puts us deep in memory-bandwidth-bound territory.
        # ======================================================
        outputs = model(input_ids=generated)

    logits = outputs.logits[:, -1, :]    # shape: [batch, vocab_size]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)

    if next_token.item() == tokenizer.eos_token_id:
        break

decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
```

The catastrophic inefficiency here is the call on line `outputs = model(input_ids=generated)`. At decode step $t$, `generated` has length $n_{\text{input}} + t$ — it is growing by one token each step. The model is recomputing the key and value vectors for all previously seen tokens from scratch on every single step. For a 512-token input and 200-token output, step 200 recomputes the attention keys and values for all 711 tokens even though 711 of them were computed on the previous step.

This recomputation is the naive decode problem. The KV cache (discussed in Section 3) solves it by storing the computed keys and values. But even with a KV cache, the fundamental bottleneck remains: to compute one new output token, the model must read every weight matrix in every layer from HBM.

### Why every weight must be read for every token

The transformer forward pass for a single new token $x_t$ involves, for each of $L$ layers:
1. A query projection: $q = x_t \cdot W_Q$ — reads $W_Q \in \mathbb{R}^{d \times d_{\text{attn}}}$
2. A key projection (for the new token): $k_t = x_t \cdot W_K$ — reads $W_K$
3. A value projection (for the new token): $v_t = x_t \cdot W_V$ — reads $W_V$
4. Attention computation: $a = \text{softmax}(q \cdot [k_1, \ldots, k_t]^T / \sqrt{d_k})$ — reads KV cache
5. An output projection: $\text{out} = a \cdot [v_1, \ldots, v_t] \cdot W_O$ — reads $W_O$
6. Two FFN layers: $\text{ffn} = \text{GELU}(x \cdot W_1) \cdot W_2$ — reads $W_1, W_2$

Every one of these weight matrices must be loaded from HBM into the compute units. Together they constitute the full model parameters. For Llama-3-8B with 8B parameters at 2 bytes/param (fp16): $1.6 \times 10^{10}$ bytes = 16 GB of data must transit from HBM to compute for every single output token.

This is the HBM bandwidth bottleneck. It is not a software bug. It is not a configuration problem. It is a physical consequence of the arithmetic intensity of autoregressive decode.


## 3. Deriving the maximum decode throughput: the bandwidth ceiling

This is the most important quantitative result in LLM serving. Let me derive it carefully.

**Arithmetic intensity** is the ratio of floating-point operations to bytes of memory traffic. For a GEMM operation $y = Wx$ where $W \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^{n \times b}$ (batch size $b$):
- FLOPs: $2 \cdot m \cdot n \cdot b$
- Memory bytes (assuming W must be loaded each time): $2mn + 2nb + 2mb \approx 2mn$ for large models
- Arithmetic intensity: $\frac{2mnb}{2mn} = b$

At batch size $b = 1$, the arithmetic intensity of every weight matrix multiplication during decode is exactly 1 FLOP/byte. 

The A100 SXM4 delivers:
- Peak fp16 compute: 312 TFLOPS = $3.12 \times 10^{14}$ FLOPS/s
- HBM2e bandwidth: 2 TB/s = $2 \times 10^{12}$ bytes/s

The *ridge point* — the batch size where compute and memory bandwidth are balanced — is:

$$b_{\text{ridge}} = \frac{\text{Peak FLOPS}}{\text{HBM bandwidth}} = \frac{3.12 \times 10^{14}}{2 \times 10^{12}} = 156$$

For batch sizes below 156, HBM bandwidth is the binding constraint. The maximum tokens per second is determined not by how fast the GPU can multiply matrices, but by how fast it can move weight data from HBM:

$$\text{tokens\_per\_second} = \frac{\text{HBM bandwidth}}{\text{model\_bytes}} = \frac{B_{\text{HBM}}}{2P}$$

For Llama-3-8B (P = 8 × 10^9 parameters) in fp16 (2 bytes/parameter) on A100 (2 TB/s):

$$\text{max tokens/s per request} = \frac{2 \times 10^{12}}{2 \times 8 \times 10^9} = \frac{2 \times 10^{12}}{1.6 \times 10^{10}} = 125 \text{ tokens/s}$$

The practical observed decode speed on A100 with optimized serving (vLLM, FlashAttention, CUDA graphs) is about 80–100 tokens/s per request at batch=1, versus a theoretical ceiling of 125. The gap is accounted for by:
- CUDA kernel launch overhead (~5-15% on modern GPUs)
- KV cache reads (additional HBM traffic beyond weights)
- Attention computation (for long contexts, this becomes non-trivial)
- Synchronization and memory fencing

Critically, this ceiling **cannot be broken in software**. You can get close to 125 tokens/s with highly optimized kernels, but you cannot exceed it with a single A100 serving a single request. The only way to serve faster per-token is to use newer hardware (H100 with 3.35 TB/s HBM3 gives a ceiling of ~210 tokens/s) or to amortize weight reads across many concurrent users (batching).

#### Worked example: TPOT ceiling across hardware generations

| GPU | HBM Bandwidth | Llama-3-8B ceiling | Llama-3-70B ceiling |
|---|---|---|---|
| T4 (PCIe) | 320 GB/s | ~20 tok/s | — (doesn't fit) |
| A100 40GB SXM | 2.0 TB/s | ~125 tok/s | — (doesn't fit) |
| A100 80GB SXM | 2.0 TB/s | ~125 tok/s | ~18 tok/s (2-GPU TP) |
| H100 SXM5 | 3.35 TB/s | ~210 tok/s | ~30 tok/s (2-GPU TP) |
| H200 SXM | 4.8 TB/s | ~300 tok/s | ~43 tok/s (2-GPU TP) |

These are single-request (batch=1) theoretical maxima. Practical per-user throughput at production scale (batch=32+) will be higher because weight reads are amortized.


## 4. Benchmark: what "memory-bandwidth-bound" means in practice

The arithmetic intensity argument is clean in theory. But it is worth grounding it in empirical numbers to build intuition for real deployments.

Using the NVIDIA Nsight Compute profiler on a Llama-3-8B decode step (single request, A100 SXM4), you can measure:

- Theoretical peak fp16 throughput: 312 TFLOPS
- Observed compute utilization during decode (batch=1): ~4–6 TFLOPS
- Compute efficiency: ~1.3–1.9%
- Theoretical HBM bandwidth: 2 TB/s
- Observed HBM bandwidth utilization during decode (batch=1): ~1.6–1.9 TB/s
- Memory bandwidth efficiency: ~80–95%

The GPU is running at 1–2% of its compute capacity and 80–95% of its memory bandwidth capacity. The compute units are spending 98% of their time stalled, waiting for data to arrive from HBM. The tensor cores that cost tens of thousands of dollars per chip are doing almost nothing.

This is not a misuse of the hardware. This is the fundamental reality of single-request autoregressive decode. The only way to engage those tensor cores is to batch many requests together so the weight reads serve multiple users' computations simultaneously.

At batch=128 (above the ridge point of 156 only when accounting for the KV cache overhead), compute efficiency climbs to ~40–50% and the workload becomes roughly balanced between compute and memory. At batch=256, you start to see memory throughput drop (bandwidth-limited on reads) and compute approaching 50–60% utilization. The batching sweet spot where you are best utilizing both resources is roughly batch 64–128 for Llama-3-8B on A100, but reaching that batch size requires 32–64 concurrent users with meaningful output lengths.

The practical implication: if you are serving fewer than ~30 concurrent users on an A100, the GPU is largely wasted. This is why smaller, cheaper GPUs (T4, L4) are often more cost-effective for low-traffic LLM serving: you pay less for hardware that is mostly idle anyway. The A100's bandwidth advantage only materializes when you can keep it well-batched.


## 5. The two phases: prefill is compute-bound, decode is bandwidth-bound

Every LLM request passes through two structurally different phases. Getting this distinction wrong is one of the most common design errors in LLM infrastructure.

### Phase 1: Prefill — processing the input

The prefill phase executes one forward pass through the entire model on all input tokens simultaneously. If the user submits a 512-token system prompt plus question, the model processes all 512 tokens in parallel: the attention mechanism computes a $512 \times 512$ attention matrix, and every token attends to every other token in a single batched operation.

Prefill is **compute-bound**. Its arithmetic intensity is proportional to the sequence length:

$$\text{Prefill arithmetic intensity} \approx \frac{2 \cdot P \cdot S}{2P + \text{KV writes}} \approx S$$

For $S = 512$, arithmetic intensity is ~512 FLOP/byte — far above the ridge point of 156. The GPU's tensor cores are the bottleneck, not HBM. Prefill time scales as $O(S^2)$ for the attention component (quadratic in sequence length) and $O(S)$ for the FFN components (linear in sequence length). For modern long-context models with flash attention, the practical prefill latency for Llama-3-8B on A100 is:

- S=128 tokens: ~15 ms
- S=512 tokens: ~35 ms
- S=2048 tokens: ~90 ms
- S=8192 tokens: ~400 ms

The time to first token (TTFT) the user experiences is almost entirely determined by prefill length. Long system prompts directly penalize TTFT.

### Phase 2: Decode — generating output tokens

Once prefill completes, the model enters the decode loop. Each step generates exactly one token: the model takes the previously generated token as input, runs a forward pass (but now only over one token's worth of computation, reusing KV cache for prior context), and samples the next token.

Decode is **memory-bandwidth-bound**. The arithmetic intensity is approximately 1 FLOP/byte at batch=1, as derived above. Each decode step runs at roughly 1/100th the arithmetic intensity of prefill.

Time Per Output Token (TPOT) is:

$$\text{TPOT} = \frac{\text{model bytes read} + \text{KV cache reads}}{\text{HBM bandwidth}}$$

For Llama-3-8B at context length 1024 (512-token prefill + 512 tokens already decoded):
- Weight reads: 16 GB
- KV cache reads: $32 \text{ layers} \times 32 \text{ heads} \times 128 \text{ dim} \times 1024 \text{ tokens} \times 2 \text{ (K+V)} \times 2 \text{ bytes} = 536 \text{ MB}$
- Total: ~16.5 GB
- TPOT ceiling: $\frac{16.5 \times 10^9}{2 \times 10^{12}} = 8.25 \text{ ms}$ → ~121 tokens/s

As context grows, KV cache reads add meaningfully to the bandwidth bill. At context length 8192:
- KV cache reads: $32 \times 32 \times 128 \times 8192 \times 2 \times 2 = 4.3 \text{ GB}$
- Total per step: ~20.3 GB
- TPOT ceiling: ~98 tokens/s

Long-context serving is slower not just because the KV cache is bigger, but because every decode step must read more of it.

### Implications for latency SLO design

The two-phase structure means you cannot design a single "latency SLA" for LLM requests. You need two separate SLOs, and they call for different optimization strategies:

**TTFT SLA (p99)**: how long from request arrival to the first token emitted. This is dominated by:
1. Queuing time (how long before this request is scheduled onto the GPU)
2. Prefill computation time (how long to process the input tokens)
3. Any long-running prefills from co-located requests blocking this one (see chunked prefill above)

To improve TTFT p99, you tune: queuing policy (priority-based admission), context length limits, chunked prefill (prevents long prefills from blocking short ones), and GPU count (more GPUs = lower queuing time at same QPS).

**TPOT SLA (p99)**: how long between consecutive output tokens. This is dominated by:
1. Model size (more bytes to read from HBM per step)
2. HBM bandwidth (hardware generation)
3. Batch size (larger batches mean each decode step serves more users but takes longer per step)
4. Context length (longer context = more KV cache reads per step)

To improve TPOT p99, you tune: batch size caps, quantization (reduce model bytes), GPU hardware selection (H100 over A100 for bandwidth), and context length limits.

The fundamental tension: reducing batch size improves TPOT (fewer users' KV caches to read per step) but hurts throughput (less parallelism). Setting the right max batch size is the central TPOT vs throughput trade-off in production LLM serving.


## 6. The KV cache: the memory wall you cannot ignore

The KV cache is the mechanism that makes decode computationally tractable. Without it, every decode step would require recomputing the keys and values for all prior tokens — a cost that grows quadratically with sequence length. With it, we cache and reuse those computations, paying only for the new token each step.

But the KV cache introduces the central memory challenge of LLM serving: it is large, it is per-request, and its size is proportional to batch size and sequence length.

### What the KV cache stores

In a transformer attention layer with $h$ attention heads, each of dimension $d_{\text{head}}$, the attention for a sequence of $S$ tokens computes:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

Where $X \in \mathbb{R}^{S \times d_{\text{model}}}$. The key matrix $K \in \mathbb{R}^{S \times d_{\text{attn}}}$ and value matrix $V \in \mathbb{R}^{S \times d_{\text{attn}}}$ are cached. During decode step $t$, we compute $k_t, v_t$ for the new token only and append them to the cache. All prior $k_1, \ldots, k_{t-1}$ and $v_1, \ldots, v_{t-1}$ are already stored.

The size of the KV cache for a single token in a single layer is:

$$\text{kv\_per\_token\_per\_layer} = 2 \times h \times d_{\text{head}} \times \text{dtype\_bytes}$$

For Llama-3-8B: $h = 32$ heads, $d_{\text{head}} = 128$, fp16 (2 bytes):

$$\text{kv\_per\_token\_per\_layer} = 2 \times 32 \times 128 \times 2 = 16{,}384 \text{ bytes} = 16 \text{ KB}$$

Across all $L = 32$ layers:

$$\text{kv\_per\_token} = 32 \times 16 \text{ KB} = 512 \text{ KB}$$

Every token in every request's context requires 512 KB of KV cache storage in HBM.

### The full KV cache formula

For a batch of requests with maximum sequence length $S_{\text{max}}$:

$$\text{kv\_cache\_bytes} = 2 \times L \times h \times d_{\text{head}} \times S_{\text{max}} \times B \times \text{dtype\_bytes}$$

Where $B$ is the batch size. Let us compute this for key configurations.

#### Worked example: Llama-3-8B KV cache at various scales

**Configuration A**: A100 40GB, batch=32, seq\_len=2048

$$\text{kv\_cache} = 2 \times 32 \times 32 \times 128 \times 2048 \times 32 \times 2 \text{ bytes}$$
$$= 2 \times 32 \times 32 \times 128 \times 65{,}536 \times 2 = 17{,}179{,}869{,}184 \text{ bytes} \approx 16 \text{ GB}$$

The KV cache at batch=32, seq=2048 is **16 GB** — equal to the model weights. On a 40 GB A100, this leaves only ~8 GB for activations and CUDA overhead. Barely feasible.

**Configuration B**: A100 40GB, batch=64, seq\_len=2048

$$\text{kv\_cache} = 32 \text{ GB}$$

Combined with 16 GB weights: 48 GB. This exceeds the 40 GB A100. The system OOMs.

**Configuration C**: A100 80GB, batch=32, seq\_len=4096

$$\text{kv\_cache} = 2 \times 32 \times 32 \times 128 \times 4096 \times 32 \times 2 = 32 \text{ GB}$$

Combined with 16 GB weights + 2 GB overhead: 50 GB. Fits in 80 GB. Comfortable.

**Configuration D**: Llama-3-70B with GQA (8 KV heads per 64 query heads), batch=32, seq=4096 on 4× A100 80GB (320 GB total)

With Grouped Query Attention, only 8 KV heads are stored (not 64):

$$\text{kv\_cache} = 2 \times 80 \times 8 \times 128 \times 4096 \times 32 \times 2 = 42{,}949{,}672{,}960 \text{ bytes} \approx 40 \text{ GB}$$

Combined with 140 GB weights: 180 GB across 320 GB total HBM. Comfortable.

Without GQA (full MHA): KV cache would be $2 \times 80 \times 64 \times 128 \times 4096 \times 32 \times 2 \approx 343 \text{ GB}$ — which does not fit even with 320 GB total HBM.

This is why Grouped Query Attention (GQA) became standard in Llama-3, Mistral, Gemma, and essentially all modern LLMs. It is not primarily a quality optimization; it is a serving feasibility requirement.

![GPU HBM memory competition: model weights, KV cache, and activations compete for the same 40 GB pool](/imgs/blogs/why-llm-serving-is-different-2.png)

![KV cache data flow through attention layers: HBM bandwidth is the bottleneck, with 32 MB per request per decode step on A100](/imgs/blogs/why-llm-serving-is-different-5.png)

### The per-request KV cache growth rate

During decode, each step appends 512 KB (for Llama-3-8B) to the KV cache. A request generating 200 tokens grows its KV footprint by 200 × 512 KB = 100 MB. For 32 concurrent requests each generating 200 tokens: 3.2 GB of new KV data written to HBM over the course of the generation. The memory manager must track this growth and preemptively evict or reject requests when HBM fills.

The rewrite of the decode loop using the HuggingFace `past_key_values` caching API illustrates how this works in practice:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = model.to("cuda").half()

prompt = "Explain the attention mechanism in transformers."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Phase 1: PREFILL
# Process all input tokens in parallel; KV cache initialized
with torch.no_grad():
    prefill_output = model(
        input_ids=input_ids,
        use_cache=True              # tells the model to return KV cache
    )

# past_key_values is a tuple of (key, value) tensors per layer
# Shape: L × 2 × [batch, n_heads, seq_len, head_dim]
past_key_values = prefill_output.past_key_values
next_token = prefill_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
generated_ids = [next_token]

# Phase 2: DECODE
# Each step: only process one new token; append to KV cache
for step in range(299):
    with torch.no_grad():
        decode_output = model(
            input_ids=next_token,          # just ONE new token
            past_key_values=past_key_values,   # growing KV cache
            use_cache=True
        )
    
    # KV cache grows by 512 KB per step for Llama-3-8B
    past_key_values = decode_output.past_key_values
    next_token = decode_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_token)
    
    if next_token.item() == tokenizer.eos_token_id:
        break

all_new_ids = torch.cat(generated_ids, dim=-1)
print(tokenizer.decode(all_new_ids[0], skip_special_tokens=True))
```

This is correct but problematic at scale: `past_key_values` is a Python tuple of tensors with variable-length dimensions. Growing it by one step requires creating new tensor objects each iteration. Memory fragmentation accumulates. For a production system handling hundreds of concurrent requests, this Python-object-level KV cache management is far too slow. Production LLM frameworks (vLLM, TGI) pre-allocate contiguous KV memory blocks and update them in-place with custom CUDA kernels.


## 7. Why static batching fails catastrophically for LLMs

You have tuned static batching for classifier serving and it works. Group $B$ requests, run one forward pass, return $B$ results. Total throughput scales linearly with $B$ up to the compute bound. Latency is predictable. The GPU stays busy.

For LLMs, this approach leads to GPU waste that makes the deployment economically unviable.

### The output length variance problem

An LLM generates a variable number of tokens for each request. A yes/no question might generate 2 tokens. A "write me a short essay" request might generate 400 tokens. A code generation request might generate 800 tokens. You cannot know the output length at request admission time — the model discovers it as it generates.

With static batching, you group $B$ requests into a batch and run them together until **all** have finished. The batch is complete when the longest sequence emits an EOS token. For the entire duration between when the shortest sequence finishes and when the longest sequence finishes, the GPU is effectively idle for those completed requests — it runs attention over padding tokens or uses a masking strategy that wastes compute cycles.

The GPU waste fraction is:

$$\text{GPU waste} = 1 - \frac{\bar{L}}{L_{\max}}$$

Where $\bar{L}$ is the mean output length and $L_{\max}$ is the maximum output length in the batch. In real chat production distributions:

| Workload | Mean output | p99 output | GPU waste at batch=32 |
|---|---|---|---|
| FAQ chatbot | 50 tokens | 120 tokens | 58% |
| Code assistant | 80 tokens | 400 tokens | 80% |
| Summarization | 120 tokens | 250 tokens | 52% |
| Open-ended chat | 100 tokens | 500 tokens | 80% |

80% GPU waste means you are paying for a \$3/hour GPU but only getting \$0.60/hour of actual work. At scale, this cost is enormous.

![Static batching vs continuous batching: before-after showing GPU utilization improvement from ~45% to ~85%](/imgs/blogs/why-llm-serving-is-different-3.png)

### Why padding does not help

One might suggest: pad all sequences in a batch to the same length. The model then processes a uniform-length batch with clean tensor shapes. But:

1. **Masked attention is not free**: Modern attention implementations (FlashAttention) have optimizations that skip computation for masked positions, but the masking itself adds overhead and the softmax over padded positions still costs memory bandwidth.
2. **Padding to max length is wasteful**: If one request generates 400 tokens and the other 31 in the batch generate 40 each, you are running 31 × 360 = 11,160 unnecessary decode steps.
3. **The fundamental problem is not padding, it is sequencing**: even with perfect masking, the GPU cannot start processing new requests that arrive while the long-running batch is incomplete. New requests queue up outside the batch, adding latency.

### A quantitative look at static batching underperformance

Here is a concrete benchmark scenario. Measure tokens/s throughput on Llama-3-8B with static batching (HuggingFace `generate()` with batch=32) versus continuous batching (vLLM) on A100 80GB, with a Poisson arrival rate of 20 requests/second and output length drawn from LogNormal(mean=100, std=80):

```python
# Simulating static batching behavior (educational — not for production)
import time
import numpy as np

def simulate_static_batching(n_requests=320, batch_size=32, 
                              mean_output=100, std_output=80):
    """Simulate GPU time spent on padding waste in static batching."""
    # Sample output lengths from a realistic distribution
    output_lengths = np.maximum(1, np.random.lognormal(
        mean=np.log(mean_output), sigma=std_output/mean_output, 
        size=n_requests
    ).astype(int))
    
    total_gpu_tokens = 0    # tokens actually computed (including padding waste)
    useful_gpu_tokens = 0   # tokens that produced real output
    
    for batch_start in range(0, n_requests, batch_size):
        batch_outputs = output_lengths[batch_start:batch_start + batch_size]
        max_len = batch_outputs.max()
        
        # GPU computes batch_size * max_len decode steps
        total_gpu_tokens += batch_size * max_len
        # Only batch_outputs.sum() of those are for real output tokens
        useful_gpu_tokens += batch_outputs.sum()
    
    efficiency = useful_gpu_tokens / total_gpu_tokens
    waste = 1 - efficiency
    
    return {
        "total_gpu_tokens": total_gpu_tokens,
        "useful_tokens": useful_gpu_tokens,
        "efficiency": efficiency,
        "waste_fraction": waste,
    }

# Run simulation
np.random.seed(42)
result = simulate_static_batching(n_requests=3200)
print(f"GPU efficiency: {result['efficiency']:.1%}")  # typically ~20-35%
print(f"GPU waste: {result['waste_fraction']:.1%}")   # typically ~65-80%
```

The simulation consistently shows 65–80% GPU waste for realistic LLM output distributions. Continuous batching eliminates this waste by treating the GPU as a stream processor: every decode step, the scheduler examines which sequences have finished, immediately frees their KV memory, and inserts new requests from the waiting queue.


## 8. Why traditional serving frameworks fail for LLMs

Let us be specific. The missing capabilities are not vague architectural limitations — they are precise technical gaps.

![Traditional vs LLM-specific serving: six-dimensional comparison across KV management, batching, streaming, multi-GPU, and prefix cache support](/imgs/blogs/why-llm-serving-is-different-4.png)

### TorchServe: the handler blocking problem

TorchServe executes inference via a BaseHandler subclass. The handler's `handle()` method is called once per request and must return a response. For an LLM generating 200 tokens, this means `handle()` blocks for the entire generation duration — typically 2–10 seconds at production load. TorchServe's worker thread is occupied for that entire time.

More critically: TorchServe's batching operates at the `handle()` invocation level. It collects $B$ requests, calls the handler with a batch of $B$, waits for the handler to return $B$ responses. This is static batching with no mechanism for iteration-level scheduling. The handler cannot yield control mid-generation to allow new requests to join the batch.

There is no `past_key_values` management, no KV page allocator, no memory-aware admission control. If 10 requests arrive simultaneously with max\_new\_tokens=2048, TorchServe allocates a 2048-step KV cache for all 10 simultaneously — immediately consuming 10 × 1 GB = 10 GB of HBM — even if all 10 will only generate 50 tokens each.

### Triton Inference Server: the stateless execution model

Triton's model execution model is inherently stateless: an inference request arrives, Triton executes the model, Triton returns the response. The "model" in Triton's model is a function $f: \text{input} \to \text{output}$, not a stateful session.

For LLM decode, the "state" (the growing KV cache) is precisely the thing Triton has no native mechanism to manage. You can work around this by:
1. Exporting the model as a stateful ONNX session — but ONNX's stateful execution is not designed for variable-length growing sequences
2. Using Triton's Python backend to implement a custom request handler that manages `past_key_values` — which is just re-implementing a serving framework inside Triton
3. Using the vLLM Triton backend plugin — which adds all the LLM-specific logic as an external library, making Triton essentially a network frontend for vLLM

None of these approaches gives you the iteration-level scheduling, memory-aware admission control, or prefix caching that a native LLM serving system provides.

There is one area where Triton retains a genuine advantage: ensemble pipelines. For multi-modal models (vision + language) or retrieval-augmented generation pipelines with multiple sub-models, Triton's ensemble model type allows chaining models within a single inference request with minimal inter-model latency. A common production pattern is: Triton frontend (protocol handling, request routing, preprocessing) → vLLM backend (LLM generation) → Triton postprocessing model. This hybrid architecture gets the best of both worlds: Triton's rich protocol support and pipeline composition plus vLLM's LLM-specific optimizations.

A minimal Triton model config for a proxy-to-vLLM setup:

```json
{
  "name": "llm_proxy",
  "backend": "python",
  "max_batch_size": 0,
  "model_transaction_policy": {"decoupled": true},
  "input": [
    {"name": "prompt", "data_type": "TYPE_STRING", "dims": [1]},
    {"name": "max_tokens", "data_type": "TYPE_INT32", "dims": [1]}
  ],
  "output": [
    {"name": "output_text", "data_type": "TYPE_STRING", "dims": [1]}
  ]
}
```

The Python backend forwards requests to vLLM's gRPC endpoint, handling SSE streaming within a decoupled transaction. This pattern lets you keep Triton as the entry point for its load balancing and monitoring capabilities while delegating actual LLM execution to vLLM.

### FastAPI + model.generate(): the GIL and throughput ceiling

A FastAPI service that calls `model.generate()` in a route handler hits the Python GIL. Even with `async` route handlers, the compute-intensive model.generate() will block the event loop unless offloaded to a thread pool. With a thread pool of size $N$, you can handle at most $N$ concurrent requests — and each holds a full KV cache allocation, limiting $N$ severely due to memory.

Here is what the naive implementation looks like and why it fails:

```python
# Naive FastAPI LLM serving — demonstrates the problems
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Global model (correct) — but:
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = model.to("cuda").half()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Thread pool — naive concurrency
executor = ThreadPoolExecutor(max_workers=4)  # 4 workers = 4 concurrent KV caches

def _generate_sync(prompt: str, max_new_tokens: int) -> str:
    """Blocking generate — occupies one thread for entire generation."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    # Problem 1: No KV cache size management — allocates max_new_tokens
    # worth of KV cache immediately, regardless of actual output length.
    # Problem 2: No continuous batching — this request runs solo on GPU.
    # Problem 3: No memory-aware admission — will OOM at high concurrency.
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            use_cache=True,
        )
    
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

@app.post("/generate")
async def generate(prompt: str, max_new_tokens: int = 200):
    # Run blocking generate in thread pool to avoid blocking event loop.
    # But: with 4 workers and 2s generation time → only 2 req/s throughput.
    # At 100 concurrent users, queue depth = ~48 requests waiting.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        _generate_sync, 
        prompt, 
        max_new_tokens
    )
    return {"text": result}
```

With 4 executor workers and 2-second average generation time, this system handles at most 2 requests/second. A traffic spike to 20 concurrent users means 16 requests are queuing, each waiting 8+ seconds just to start generating. The p99 latency explodes. There is no backpressure. There is no memory awareness. If 4 users all request max\_new\_tokens=2048 simultaneously, you allocate 4 × 1 GB KV caches = 4 GB, leaving little room for the actual model weights on a 24 GB consumer GPU.


## 9. What LLM-specific serving must do: the six capabilities

Now we can state precisely what production LLM serving requires. These are not nice-to-haves; they are the minimum viable capabilities for a system that will not collapse under load.

### Capability 1: Continuous (iteration-level) batching

The scheduler must operate at the decode-step granularity, not the request granularity. Every decode iteration — every step of the generation loop — the scheduler:
1. Runs the decode step for all currently active requests
2. Checks which requests have emitted an EOS token (finished)
3. Removes finished requests and frees their KV cache memory
4. Checks the waiting queue for pending requests
5. Admits new requests from the queue if KV memory is available
6. Runs the next decode step with the updated batch

This way, short responses finish quickly and immediately free up capacity for new arrivals. Long responses do not block short ones. GPU utilization remains high because the batch is always filled.

### Capability 2: PagedAttention for KV cache management

Instead of pre-allocating the maximum possible KV cache for each request, divide HBM into fixed-size KV blocks (pages) of $P$ tokens each. Allocate pages on demand, one page at a time, as a request generates tokens. When a request finishes, return its pages to the free pool.

This mirrors virtual memory paging in operating systems. The "page table" maps logical sequence positions to physical HBM page addresses. The vLLM paper showed this reduces KV memory waste from 20–40% (with static allocation) to <4%, directly enabling more concurrent requests on the same hardware.

### Capability 3: Memory-aware scheduler and admission control

The scheduler must track HBM utilization in real time and enforce admission control: do not admit a new request if doing so would trigger an OOM. When memory pressure rises, the scheduler can:

- **Preempt** a low-priority request (swap its KV pages to CPU DRAM or discard them)
- **Defer** new requests (queue them until memory frees up)
- **Reject** requests (return 429 with a Retry-After header) when queues are full

```python
# vLLM memory-aware configuration
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="float16",
    max_model_len=4096,
    # vLLM reserves this fraction of HBM for KV page pool
    gpu_memory_utilization=0.90,
    # Hard ceiling on concurrent sequences; vLLM enforces admission control
    max_num_seqs=64,
    # KV blocks of 16 tokens each (default); tune for your context distribution
    block_size=16,
    # Enable CPU swap for preempted sequences
    swap_space=4,     # 4 GB of CPU RAM for KV swap
)
```

### Capability 4: Prefix caching for repeated prompt prefixes

Many production LLM deployments share a long, static prefix across all requests: a system prompt, a few-shot example set, or a retrieved document in a RAG pipeline. Without prefix caching, every request pays the full prefill cost for this prefix. With prefix caching:

1. Hash the token IDs of the prefix
2. Check if the KV blocks for this prefix are in the cache
3. If yes: skip prefill for the shared prefix, load KV blocks directly
4. If no: compute prefill normally, store the resulting KV blocks tagged with the hash

vLLM's RadixAttention (v0.4+) implements a trie-based prefix cache that handles dynamic prefixes of arbitrary length. For a system prompt of 1024 tokens, prefix cache hits eliminate ~90 ms of TTFT per request. At 100 QPS, that is 9 CPU-equivalent seconds saved per second — a significant compute reduction.

### Capability 5: Streaming output with SSE or WebSocket

For interactive applications (chat, coding assistants, document editing), waiting for the complete response before rendering anything is unacceptable UX. Users expect to see tokens appear as they are generated.

LLM serving systems must emit each token as it is produced and stream it to the client. The most common protocol for this is Server-Sent Events (SSE), which allows the server to push multiple events on a single long-lived HTTP connection. The client receives tokens in near-real-time, displaying them progressively.

This requires the serving framework to yield after each decode step and send the generated token to the client — a fundamentally different execution model from request-response.

### Capability 6: Runtime quantization for memory and throughput

Running models in FP8 instead of FP16 halves the model weights bytes and KV cache bytes, directly doubling the bandwidth-bound throughput ceiling:

$$\text{tokens/s FP8} = \frac{B_{\text{HBM}}}{P \times 1 \text{ byte}} = \frac{2 \times 10^{12}}{8 \times 10^9} = 250 \text{ tokens/s}$$

(vs. 125 tokens/s for fp16 on A100). The KV cache also halves. A production LLM serving system must apply quantization during serving (not just training) and manage the interaction between quantized weights and the KV cache memory manager.

The common serving quantization options and their properties:

| Method | Weight dtype | Activation dtype | Accuracy loss | Memory saving | Throughput gain |
|---|---|---|---|---|---|
| FP16 (baseline) | fp16 | fp16 | None | 0% | 1× |
| FP8 (H100 native) | fp8 | fp8 | <0.5% on benchmarks | 50% | ~2× |
| AWQ INT4 | int4 | fp16 | 1–2% on typical tasks | 75% | ~1.6× |
| GPTQ INT4 | int4 | fp16 | 1–3% on typical tasks | 75% | ~1.5× |
| SmoothQuant INT8 | int8 | int8 | <1% | 50% | ~1.5× |

FP8 (via `--dtype fp8` in vLLM on H100) is the recommended choice for production in 2025–2026: near-zero accuracy loss, 2× throughput gain, and native hardware support on H100/H200. AWQ and GPTQ are better for memory-constrained deployments (fitting a 70B model on fewer GPUs) at the cost of some accuracy.

Enabling FP8 in vLLM on H100:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype fp8 \
    --kv-cache-dtype fp8 \
    --quantization fp8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

With `--kv-cache-dtype fp8`, the KV cache itself is stored in FP8, halving KV memory and enabling roughly double the concurrent sequences compared to FP16.


## 10. Maximum concurrent sequences: the resource allocation equation

The fundamental resource allocation question in LLM serving is: how many requests can I handle simultaneously before OOM?

The maximum concurrent sequences is:

$$\text{max\_seqs} = \left\lfloor \frac{\text{HBM}_{\text{total}} - \text{weights\_bytes} - \text{overhead\_bytes}}{\text{kv\_per\_seq\_bytes}} \right\rfloor$$

Where:
$$\text{kv\_per\_seq\_bytes} = 2 \times L \times h_{\text{kv}} \times d_{\text{head}} \times S_{\text{max}} \times \text{dtype\_bytes}$$

($h_{\text{kv}}$ is the number of KV heads — equal to $h$ for MHA, smaller for GQA/MQA.)

#### Worked example: computing max concurrent sequences for real deployments

**Deployment A: Llama-3-8B fp16 on A100 40GB, seq\_len=2048**

- HBM total: 40 GB
- Model weights: $8 \times 10^9 \times 2 = 16$ GB
- CUDA + runtime overhead: 2 GB
- Available for KV pool: 40 - 16 - 2 = 22 GB

KV per sequence (Llama-3-8B has 32 KV heads in its GQA, but using the MHA effective count for standard 8B):

$$\text{kv/seq} = 2 \times 32 \times 32 \times 128 \times 2048 \times 2 = 536{,}870{,}912 \approx 512 \text{ MB}$$

$$\text{max\_seqs} = \lfloor 22 \times 10^9 / 512 \times 10^6 \rfloor = \lfloor 22{,}000 / 512 \rfloor \approx 43 \text{ seqs}$$

In practice, vLLM's default `gpu_memory_utilization=0.90` gives ~36 GB available (= 0.9 × 40 = 36; minus 16 weights = 20 GB for KV), yielding:

$$\text{max\_seqs} = \lfloor 20{,}000 / 512 \rfloor \approx 39 \text{ sequences}$$

This is why vLLM's default `max_num_seqs=256` is too high for this configuration and will cause OOM if you do not tune it. The actual safe value is closer to 39 at seq\_len=2048.

**Deployment B: Llama-3-70B fp16 on 4× A100 80GB TP4, seq\_len=4096**

- HBM total: 4 × 80 = 320 GB
- Model weights (fp16): $70 \times 10^9 \times 2 = 140$ GB
- Tensor parallel overhead (NCCL buffers, activation memory): ~8 GB
- Available for KV: 320 - 140 - 8 = 172 GB

KV per sequence for Llama-3-70B (GQA with 8 KV heads, 80 layers, 128 head dim):

$$\text{kv/seq} = 2 \times 80 \times 8 \times 128 \times 4096 \times 2 = 1{,}342{,}177{,}280 \approx 1.25 \text{ GB}$$

$$\text{max\_seqs} = \lfloor 172 / 1.25 \rfloor = 137 \text{ sequences}$$

**Deployment C: Llama-3-8B fp8 on H100 80GB, seq\_len=4096**

- HBM total: 80 GB
- Model weights (fp8): $8 \times 10^9 \times 1 = 8$ GB
- KV cache dtype (can also use fp8 or fp16 — assume fp16 for KV): 2 bytes
- Overhead: 2 GB
- Available for KV: 80 - 8 - 2 = 70 GB

KV per sequence at seq=4096:

$$\text{kv/seq} = 2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 1{,}073{,}741{,}824 \approx 1 \text{ GB}$$

$$\text{max\_seqs} = \lfloor 70 / 1 \rfloor = 70 \text{ sequences}$$

With fp8 KV cache (1 byte): kv/seq = 512 MB, max\_seqs = 140.

These numbers tell you exactly how to set `max_num_seqs` in vLLM, `--max-concurrent-requests` in TGI, and KV cache allocation in your infrastructure planning. Undershoot and you waste capacity; overshoot and you OOM.


## 11. Multi-GPU LLM serving: tensor and pipeline parallelism

When a model exceeds single-GPU capacity (Llama-3-70B at 140 GB fp16 does not fit in any single GPU), you must distribute the model across multiple GPUs. There are two principal strategies.

### Tensor parallelism: splitting the work

Tensor parallelism (TP) splits each weight matrix across $N$ GPUs along a column or row dimension. For an FFN layer's first weight matrix $W_1 \in \mathbb{R}^{d \times 4d}$:

- GPU 0 holds columns 0 to $d$: $W_1^{(0)} \in \mathbb{R}^{d \times d}$
- GPU 1 holds columns $d$ to $2d$: $W_1^{(1)} \in \mathbb{R}^{d \times d}$
- ...
- GPU $N-1$ holds columns $(N-1)d$ to $Nd$

The input $x$ is replicated on all GPUs. Each GPU computes a partial output: $y^{(i)} = x \cdot W_1^{(i)}$. After the second FFN layer and the attention output projection, the partial results must be summed across GPUs via an AllReduce collective.

The AllReduce communication data per decode step for the FFN sublayer is:

$$\text{AllReduce bytes} = 2 \times B \times d_{\text{model}} \times \text{dtype\_bytes}$$

For Llama-3-8B, $d_{\text{model}} = 4096$, batch=32, fp16:

$$= 2 \times 32 \times 4096 \times 2 = 524{,}288 \text{ bytes} = 512 \text{ KB}$$

On NVLink (600 GB/s bidirectional): $\frac{512 \times 10^3}{6 \times 10^{11}} \approx 0.85 \mu s$ per AllReduce. Negligible.

On PCIe (64 GB/s): $\frac{512 \times 10^3}{6.4 \times 10^{10}} \approx 8 \mu s$ per AllReduce. With 2 AllReduces per layer × 32 layers: 512 µs = 0.5 ms of communication overhead per decode step. Noticeable but manageable.

The KV cache is also sharded in TP: each GPU holds the KV for its assigned heads. For 4-way TP on Llama-3-8B (32 heads): each GPU holds 8 heads' KV data, reducing per-GPU KV to 128 MB per sequence at seq=2048 (vs. 512 MB without TP).

![Multi-GPU tensor parallelism: weight matrix split across 8 GPUs, AllReduce communication cost per decode step](/imgs/blogs/why-llm-serving-is-different-6.png)

### Pipeline parallelism: staging across GPUs

Pipeline parallelism (PP) assigns groups of consecutive transformer layers to different GPUs:

- GPU 0: layers 0–19 (20 layers)
- GPU 1: layers 20–39 (20 layers)
- ...

The forward pass proceeds sequentially: GPU 0 processes its layers and sends the activation to GPU 1, which processes its layers and sends to GPU 2, etc. This is a pipeline, and like all pipelines, it has a bubble: while GPU 1 is processing, GPU 0 is idle (waiting for the next micro-batch).

For $S$ pipeline stages and $M$ micro-batches per iteration, the bubble fraction is:

$$\text{bubble fraction} = \frac{S - 1}{S + M - 1}$$

For $S = 4$, $M = 4$: $\frac{3}{7} \approx 43\%$ bubble.
For $S = 4$, $M = 16$: $\frac{3}{19} \approx 16\%$ bubble.

PP is best for very large models (70B+) where TP has saturated NVLink bandwidth and you need to go multi-node. Combining TP within a node with PP across nodes is the standard recipe for 70B–405B models:

```bash
# vLLM: 70B model on 2 nodes × 4 GPUs (TP=4 within node, PP=2 across nodes)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 8000
```

### Expert parallelism for MoE models

Mixture-of-Experts (MoE) models — including DeepSeek-V2/V3, Mixtral, and similar architectures — present a third parallelism dimension: **expert parallelism (EP)**. In an MoE layer, each token is routed to one or two "expert" FFN sub-networks out of, say, 64 total. With EP, different GPUs host different experts: GPU 0 hosts experts 0–7, GPU 1 hosts experts 8–15, etc.

For a token routed to expert 5 on GPU 0, the token's activation is sent to GPU 0 for computation. For a token routed to expert 22 (on GPU 2), the activation is dispatched to GPU 2. This all-to-all dispatch pattern introduces different communication patterns compared to TP's AllReduce: instead of summing partial results, you are routing activations to specific GPUs based on expert selection.

EP can be more efficient than TP for MoE models because:
- The all-to-all communication volume is proportional to the number of tokens dispatched (not the full hidden dimension)
- Experts are computed independently with no synchronization within the MoE layer
- Expert imbalance (some experts receiving more tokens than others) is the primary challenge

For Mixtral-8×22B (8 active experts out of 64 total) on 8 GPUs with EP=8: each GPU holds 8 experts, all token dispatch happens intra-node via NVLink, and the effective FFN compute is distributed 8-way. The vLLM `--enable-expert-parallel` flag enables EP for MoE models.

The [DeepSeek inference optimization post](/blog/machine-learning/model-serving/deepseek-inference-optimization) in Track H covers EP in depth for production-scale MoE serving.

| Parallelism type | Split dimension | Communication | Best for |
|---|---|---|---|
| Tensor (TP) | Weight matrix columns/rows | AllReduce per layer | Single-node dense models |
| Pipeline (PP) | Layer groups | Point-to-point activation | Multi-node, deep models |
| Expert (EP) | Expert sub-networks | All-to-all dispatch | MoE models (Mixtral, DeepSeek) |
| Data (DP) | Request batches | None (replicas) | High-throughput, full replicas |


## 12. Memory hierarchy in LLM serving

An LLM serving system manages three memory tiers with very different characteristics.

**HBM (GPU High-Bandwidth Memory)**: 40–80 GB per GPU, 2–3.35 TB/s bandwidth. This is the primary working memory. Model weights live here permanently. Active KV caches and activations live here during generation. HBM is scarce and is the primary bottleneck for concurrency.

**CPU DRAM (Host Memory)**: 256 GB – 2 TB per server, ~50 GB/s PCIe bandwidth. Used for KV cache swapping: when HBM pressure rises, the least-recently-used KV pages of preempted requests are swapped to DRAM. PCIe bandwidth (~50 GB/s) means swapping a 512 MB KV cache takes ~10 ms — non-trivial latency but viable for low-priority background requests.

**NVMe SSD (Local Storage)**: 4–16 TB per server, ~7 GB/s bandwidth. Too slow for KV cache swap in real-time serving. Useful for model weight offloading in very-large-model scenarios (llama.cpp's mmap approach for consumer hardware) or offline batch jobs. Not used in production low-latency serving.

The memory manager must balance these tiers:

1. Keep all model weights in HBM (static allocation at startup)
2. Allocate KV pages from HBM pool for active requests
3. When HBM KV pool fills: swap out KV pages for low-priority requests to DRAM
4. When DRAM swap fills: reject new requests until high-priority requests complete

This three-tier management is what makes a production LLM serving system genuinely complex software — it is more like an operating system memory manager than a simple inference runner.

### Memory pressure eviction policies

When HBM pressure rises during a traffic spike, the LLM serving system must decide which KV caches to evict. Three policies are commonly used:

**LRU (Least Recently Used)**: evict the KV cache of the request that has gone the longest without a decode step. This is a proxy for "the request that is least likely to complete soon and most likely to benefit from waiting." Simple to implement, reasonable in practice.

**Priority-based**: assign each request a priority score (e.g., based on queuing time, user tier, or remaining generation budget) and evict the lowest-priority request first. This is what production systems serving multiple user tiers (free vs. paid) typically implement.

**Recompute vs. swap decision**: for short-context requests (< 512 tokens), it is often cheaper to evict and recompute than to swap to DRAM: recomputation takes ~50ms (prefill), while a DRAM swap-out and swap-in via PCIe takes ~10ms + ~10ms = ~20ms for a 512-token KV. For long-context requests (4k+ tokens), swapping is clearly cheaper (recompute would take ~400ms+ for a 4096-token prefill).

The break-even point between recompute and swap is:

$$S_{\text{break-even}} = \frac{2 \times \text{swap latency} \times \text{HBM bandwidth}}{\text{kv\_per\_token}}$$

For kv\_per\_token = 512 KB, swap latency = 10ms (PCIe), HBM bandwidth = 2 TB/s:

$$S_{\text{break-even}} = \frac{2 \times 0.01 \times 2 \times 10^{12}}{512 \times 10^3} \approx 78{,}125 \text{ tokens}$$

Any request with fewer than ~78k tokens in context is cheaper to recompute than to swap. In practice, this threshold is much lower due to PCIe overhead and DRAM latency, but the principle holds: short-context requests should use recompute eviction; long-context requests should use DRAM swap.

![A100 40 GB memory budget breakdown for Llama-3-8B serving at batch=16, showing the three competing consumers](/imgs/blogs/why-llm-serving-is-different-7.png)


## 13. Case studies and benchmarks

### Case study 1: vLLM vs naive HuggingFace on OPT-13B (Kwon et al., NeurIPS 2023)

The vLLM paper (Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," NeurIPS 2023) benchmarked PagedAttention + continuous batching against HuggingFace text generation with static batching on A100 GPUs.

For OPT-13B at a request rate of 30 requests/second with variable output lengths:
- HuggingFace + static batching: ~300 tokens/s total throughput, p99 TTFT ~8.2s
- vLLM (PagedAttention + continuous batching): ~1,200 tokens/s total throughput, p99 TTFT ~2.1s

This is a 4× throughput improvement on identical hardware, achieved purely through scheduling and memory management improvements — no changes to the model architecture.

The paper also measured memory waste:
- Static allocation (HuggingFace): 20–38% internal fragmentation (allocating for max length when actual length is shorter), up to 55% external fragmentation when many small requests fragment the contiguous allocation
- PagedAttention: <4% internal fragmentation (at most $P-1$ tokens wasted per sequence, where $P$ is the page size), near-zero external fragmentation

### Case study 2: TGI vs vLLM benchmarks (community benchmarks, 2024)

Public benchmarks comparing TGI 2.0 and vLLM 0.4 on Llama-2-13B on 1× A100 80GB at 32 concurrent users (ShareGPT prompt distribution, variable output lengths):

| Metric | TGI 2.0 | vLLM 0.4 | Winner |
|---|---|---|---|
| Throughput (tok/s) | 412 | 527 | vLLM +28% |
| TTFT p50 (ms) | 320 | 290 | vLLM slight edge |
| TTFT p99 (ms) | 2,100 | 1,800 | vLLM |
| TPOT p50 (ms) | 18 | 14 | vLLM |
| GPU utilization | 82% | 88% | vLLM |
| Prefix cache hit rate | N/A (TGI 2.0) | 61% | vLLM |

Both systems vastly outperform naive HuggingFace + FastAPI, which achieves roughly 60–80 tokens/s at this concurrency level (limited to 4–8 parallel requests by memory).

### Case study 3: The memory fragmentation crisis before PagedAttention

Before PagedAttention, production LLM deployments commonly used a "reserved allocation" strategy: at request admission, allocate the maximum possible KV cache (`max_new_tokens × kv_per_token`). A request with max\_new\_tokens=2048 reserved 1 GB of KV cache, even if it actually generated only 50 tokens.

At a service running ~1,000 QPS of short-form chat requests (mean output 50 tokens, max 2048), this strategy allocated $1000 \times 1 \text{ GB} = 1 \text{ TB}$ worth of KV cache requests per second, against an actual utilization of $1000 \times 25 \text{ MB} = 25 \text{ GB}$. The fragmentation ratio was 40:1. The result was that the system could only sustain ~8 concurrent requests on an A100 80GB before OOM, when the theoretical limit (if allocation were perfectly efficient) was ~80.

PagedAttention solved this: with 16-token KV pages, the same service can host 64+ concurrent requests on the same A100 80GB, with actual dynamic allocation matching actual usage.


## 14. Comparing the serving stacks: when to use each

Given all of the above, here is the decision matrix for LLM serving:

![Technique vs SLO impact matrix: every LLM serving optimization targets specific corners of the latency-throughput-cost triangle](/imgs/blogs/why-llm-serving-is-different-8.png)

The fuller comparison across serving options:

| Serving stack | KV management | Cont. batching | Streaming | Multi-GPU TP | Prefix cache | Best for |
|---|---|---|---|---|---|---|
| TorchServe | None | No | Limited | No | No | Non-LLM models |
| Triton (vanilla) | None | No | Manual | No | No | CNNs, encoders |
| FastAPI + generate() | None | No | Manual | No | No | Dev/prototyping only |
| vLLM | PagedAttention | Yes | SSE native | Yes (TP+PP) | Yes (Radix) | Production LLM serving |
| TGI | Custom KV mgmt | Yes | SSE native | Yes (TP) | Partial | LLM serving, HF ecosystem |
| llama.cpp (CPU) | Custom | Partial | Yes | No | No | Local/edge serving |
| DeepSpeed-Inference | Custom | Partial | Yes | Yes (TP+PP) | No | Large-scale training-collocated |


## 15. Production configuration walkthrough: getting to 100 QPS on Llama-3-8B

Let us work through a complete production configuration exercise. The goal: serve Llama-3-8B at 100 requests per second with TTFT p99 < 2s and TPOT p99 < 30ms, on a budget of two A100 80GB GPUs.

### Step 1: Establish the memory budget

Two A100 80GB gives 160 GB total HBM.

- Model weights (fp16): 16 GB per GPU = 32 GB total (replicated, not sharded — for TP=2 we shard)
- With TP=2: each GPU holds 8 GB of weights (half the model)
- CUDA overhead per GPU: ~2 GB × 2 = 4 GB total
- Available for KV per GPU: 80 - 8 - 2 = 70 GB per GPU = 140 GB total

KV per sequence at seq\_len=2048 (distributed across TP=2, so each GPU holds half the heads):

$$\text{kv/seq/GPU} = 2 \times 32 \times 16 \times 128 \times 2048 \times 2 = 268{,}435{,}456 \approx 256 \text{ MB}$$

$$\text{max\_seqs} = \lfloor 70 \times 10^9 / 256 \times 10^6 \rfloor = 273 \text{ sequences per GPU}$$

With TP=2, the bottleneck is the KV memory per GPU: 273 concurrent sequences total.

### Step 2: Check throughput feasibility

At 100 QPS, mean output length 100 tokens: total token generation rate = 10,000 tokens/s.

With two A100 80GBs in TP=2: effective HBM bandwidth = $2 \times 2 \text{ TB/s} = 4 \text{ TB/s}$ (both GPUs read weights in parallel for the same request, halving effective per-request weight read overhead).

Maximum decode throughput = $\frac{4 \times 10^{12}}{2 \times 8 \times 10^9} = 250$ tokens/s at batch=1.

But we need 10,000 tokens/s. At batch=40 (100 QPS × 0.4s mean service time with continuous batching), each batch member needs effective 250 tokens/s. Total throughput = 40 × 250 / 40 = 250 tokens/s per user → total 10,000 tokens/s. This works, because batch=40 is below the ridge point of ~156, meaning we are memory-bandwidth-bound, and batching helps us amortize weight reads.

Actually: total throughput at batch=40 = $\min(40 \times 250, \frac{4 \text{ TB/s}}{16 \text{ GB}} \times 40) = \min(10000, 10000) = 10{,}000$ tokens/s. We are right at the operating point.

### Step 3: vLLM launch command and configuration

```bash
# Two A100 80GB, TP=2, serving Llama-3-8B for 100 QPS
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --use-v2-block-manager \
    --block-size 16 \
    --swap-space 8 \
    --max-num-batched-tokens 8192 \
    --host 0.0.0.0 \
    --port 8000
```

Key flags explained:
- `--tensor-parallel-size 2`: split model across both GPUs
- `--gpu-memory-utilization 0.90`: use 90% of each GPU's HBM for KV pool
- `--max-num-seqs 128`: admission control ceiling; set well below the 273-sequence theoretical max for safety headroom
- `--enable-prefix-caching`: enable RadixAttention for shared system prompts
- `--swap-space 8`: 8 GB of CPU RAM for KV swapping when HBM fills
- `--max-num-batched-tokens 8192`: controls prefill chunk size (chunked prefill)

### Step 4: Validate with a load test

```python
# Benchmarking script using vLLM's built-in benchmark utility
# Or using the openai client directly:
import asyncio
import time
from openai import AsyncOpenAI
import numpy as np

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="token")

async def single_request(prompt: str, request_id: int):
    start = time.perf_counter()
    first_token_time = None
    tokens = []
    
    async with client.chat.completions.stream(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    ) as stream:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens.append(chunk.choices[0].delta.content)
    
    end = time.perf_counter()
    ttft = (first_token_time - start) * 1000  # ms
    total_time = (end - start) * 1000          # ms
    n_tokens = len(tokens)
    tpot = (total_time - ttft) / max(n_tokens - 1, 1)  # ms per token
    
    return {"ttft_ms": ttft, "tpot_ms": tpot, "n_tokens": n_tokens}

async def load_test(qps: int = 100, duration_s: int = 60):
    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the main causes of World War 1?",
        "Write a short Python function to sort a list.",
    ] * (qps * duration_s // 3 + 1)
    
    interval = 1.0 / qps
    tasks = []
    
    for i, prompt in enumerate(prompts[:qps * duration_s]):
        await asyncio.sleep(interval)
        task = asyncio.create_task(single_request(prompt, i))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    ttfts = [r["ttft_ms"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    
    print(f"TTFT p50: {np.percentile(ttfts, 50):.0f}ms")
    print(f"TTFT p99: {np.percentile(ttfts, 99):.0f}ms")
    print(f"TPOT p50: {np.percentile(tpots, 50):.0f}ms")
    print(f"TPOT p99: {np.percentile(tpots, 99):.0f}ms")
    print(f"Throughput: {sum(r['n_tokens'] for r in results)/60:.0f} tok/s")

asyncio.run(load_test(qps=100, duration_s=60))
```

On a well-tuned two-A100 80GB system with the above vLLM configuration, at 100 QPS with mean output=100 tokens, you should see:
- TTFT p50: 120–200ms
- TTFT p99: 800ms–1.5s
- TPOT p50: 14–18ms
- TPOT p99: 22–28ms

This meets the target SLO of TTFT p99 < 2s and TPOT p99 < 30ms.

### When to scale to more GPUs

If the load test shows TTFT p99 consistently above 2s, the bottleneck is prefill. Options:
1. Increase TP (4× A100 80GB) to parallelize prefill computation
2. Enable chunked prefill to reduce prefill's blocking of decode
3. Reduce max context length to limit prefill time

If TPOT p99 is consistently above 30ms, the bottleneck is decode bandwidth. Options:
1. Reduce `max-num-seqs` (reduce batch size to lower per-token latency)
2. Switch to fp8 quantization (doubles decode bandwidth efficiency)
3. Add more GPUs (more HBM bandwidth)


## 16. When to use LLM-specific serving (and when not to)

**Use vLLM or TGI when:**
- You are serving any transformer-based autoregressive model (GPT-style, Llama-style, any `AutoModelForCausalLM`)
- You have more than ~5 concurrent users
- Your output lengths vary (they always do in real chat applications)
- You need streaming responses (SSE or WebSocket)
- You need to meet a latency SLA and cannot afford the 40–80% GPU idle that static batching causes
- You are running on a GPU with ≥16 GB HBM

**Do NOT use LLM-specific serving when:**
- You are serving a BERT-style encoder model (classification, NER, embeddings, cross-encoders): these models have no autoregressive decode loop, no KV cache, and no output length variability. Vanilla Triton or ONNX Runtime is faster and simpler.
- You are processing a fixed offline batch job: running overnight inference on a fixed corpus does not benefit from continuous batching. A simple `model.generate()` loop with HuggingFace Accelerate is entirely appropriate.
- You are on CPU or consumer GPUs (<16 GB VRAM): llama.cpp with GGUF quantization is the right tool, providing excellent CPU performance with a much smaller memory footprint.
- Your QPS is below ~1: below 1 request per second, a FastAPI wrapper around model.generate() is fine. Continuous batching provides no benefit when there is never more than one concurrent request.
- You are serving sub-1B models: the overhead of a full vLLM deployment outweighs the benefits for tiny models. A simple ONNX Runtime session with a thread pool is more appropriate.


## 17. Key takeaways

These ten principles distill the post into actionable engineering rules:

1. **LLM decode is memory-bandwidth-bound.** The ceiling is `HBM_bandwidth / model_bytes_in_dtype`. For Llama-3-8B fp16 on A100: 125 tokens/s per request. No software change breaks this; only hardware upgrades or batching do.

2. **Prefill is compute-bound (batched attention); decode is bandwidth-bound (sequential).** They are different hardware workloads. Your TTFT SLA must budget for context length (prefill time). Your TPOT SLA must budget for model size and GPU generation.

3. **The KV cache formula is `2 × L × h_kv × d_head × seq_len × batch × dtype_bytes`.** For Llama-3-8B at batch=32, seq=2048: ~16 GB — equal to the model weights. Calculate this before any deployment planning.

4. **Static batching wastes 40–80% of GPU compute on LLM workloads.** Waste equals `1 - mean_output / max_output`. Continuous batching eliminates this by scheduling at the decode-step level.

5. **TorchServe, vanilla Triton, and FastAPI+generate() cannot fix these problems.** They have no concept of per-request KV state, iteration-level scheduling, or memory-aware admission control.

6. **Maximum concurrent sequences = `(HBM_free - overhead) / kv_per_seq`.** For Llama-3-8B on A100 40GB at seq=2048: ~39 sequences. Set `max_num_seqs` accordingly in vLLM — exceeding this OOMs.

7. **GQA reduces KV cache by the group ratio.** Llama-3-70B with 8-head GQA (vs 64 query heads) shrinks KV cache 8×, enabling ~137 concurrent seqs on 4× A100 80GB vs only ~17 with full MHA. Always check whether your model uses GQA.

8. **Tensor parallelism scales linearly on NVLink up to 8 GPUs.** AllReduce on NVLink (~0.85 µs per layer) is negligible. AllReduce on PCIe (~8 µs per layer, ×64 layers = ~1ms overhead) is noticeable but manageable. Beyond 8 GPUs, use TP+PP.

9. **KV prefix caching reduces TTFT for repeated prefixes by 60–90%.** For RAG or system-prompt-heavy applications, prefix cache hit rates of 40–80% are achievable, dramatically cutting both latency and compute cost.

10. **Choose vLLM for production LLM serving; TorchServe/Triton for everything else.** The 4× throughput gain from PagedAttention + continuous batching vs static batching is too large to leave on the table.

11. **MoE models need expert parallelism on top of TP+PP.** For Mixtral, DeepSeek, and similar architectures, the all-to-all dispatch in EP is fundamentally different from AllReduce in TP. Plan your parallelism strategy based on model architecture, not just model size.

12. **The production configuration equation is: calculate KV budget, set max_num_seqs conservatively, benchmark TTFT and TPOT at target QPS, and adjust batch size limit to hit both SLOs simultaneously.** There is no universal default configuration. Every deployment requires these four steps.


## Further reading

- **Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." NeurIPS 2023.** The foundational paper for PagedAttention and continuous batching. Contains the definitive memory waste analysis comparing pre-allocation vs. paged allocation strategies.
- **Pope, R. et al. (2022). "Efficiently Scaling Transformer Inference." TMLR 2023.** Deep analysis of the arithmetic intensity argument for LLM inference and optimal batching strategies across hardware generations. The roofline model analysis in this paper maps directly to the derivations in this post.
- **Yu, G. et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022.** The original continuous batching paper (predating vLLM). Essential reading for understanding the scheduling problem that PagedAttention later solved on the memory side.
- **[vLLM documentation](https://docs.vllm.ai/en/latest/)** — EngineArgs reference, distributed serving guide, prefix caching configuration, speculative decoding setup, FP8 quantization.
- **[Text Generation Inference documentation](https://huggingface.co/docs/text-generation-inference)** — TGI deployment guide, tensor parallelism flags, quantization options, benchmarking with `text-generation-benchmark`.
- **[Continuous batching and PagedAttention deep dive](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)** — the next post in this series: vLLM scheduler internals, KV block manager implementation, and preemption policy details.
- **[vLLM deep dive: chunked prefill, prefix caching, speculative decoding](/blog/machine-learning/model-serving/vllm-deep-dive)** — advanced vLLM production configuration covering the features that matter most at scale.
- **[What is model serving: the SLO triangle](/blog/machine-learning/model-serving/what-is-model-serving)** — the series foundation post: latency, throughput, and cost as the three corners of the serving trade-off space.
- **[The model serving playbook (capstone)](/blog/machine-learning/model-serving/the-model-serving-playbook)** — the complete decision tree from notebook prototype to production architecture, tying all series posts together.

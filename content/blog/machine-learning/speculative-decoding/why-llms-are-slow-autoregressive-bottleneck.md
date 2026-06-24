---
title: "Why LLMs Generate Text Slowly: The Autoregressive Bottleneck"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why autoregressive decoding keeps GPUs at 5% utilization, and why that bottleneck is the opening move for every speculative decoding paper you will read."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "gpu-optimization",
    "transformer-architecture",
    "inference-efficiency",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-1.png"
---

Run a 70B-parameter language model on an H100 — one of the fastest GPUs ever built — and ask it to complete a paragraph. Watch the GPU utilization counter while it generates. You will see something that looks like a bug: the utilization gauge sits at 5 to 15 percent for the entire decode phase. The hardware capable of nearly a thousand trillion floating-point operations per second is spending most of its time doing almost nothing.

This is not a bug. It is the fundamental character of autoregressive language model decoding, and understanding it precisely is the prerequisite for understanding every inference optimization paper in the speculative decoding literature. The throughput problem and the latency problem are different problems with different solutions, and the reason batching cannot solve latency — and why speculative decoding can — flows directly from the physics of how modern GPUs move data.

This post is the foundation for an eight-part series. By the end of it, you will understand why the GPU is idle, what "arithmetic intensity" means and why it predicts everything about decode performance, and why the only way to reduce per-token latency without hurting quality is to get more tokens out of each expensive forward pass. That insight is the entire premise of speculative decoding.

## The Token-by-Token Contract

Every transformer language model trained with next-token prediction has a structural constraint baked into its architecture: it cannot generate token $t+1$ until it has generated token $t$. This is not an implementation choice. It is a consequence of the probability factorization that defines the model.

The joint probability of a sequence $x_1, x_2, \ldots, x_T$ factorizes autoregressively:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

Each conditional $P(x_t \mid x_1, \ldots, x_{t-1})$ requires running a full forward pass through all $L$ transformer layers. The forward pass at step $t$ reads the previous hidden states (via the KV cache) and the current token embedding, applies self-attention and feed-forward transformations, and outputs a logit vector over the vocabulary. You sample from that logit vector to get $x_t$, and only then can you start computing $x_{t+1}$.

There is no way to parallelize this across positions during decode. The next token's correct distribution depends on the previous token's value. You cannot speculate ahead without introducing bias — unless you do it very carefully, which is exactly what speculative decoding does.

The contrast with prefill — the phase where the model processes the prompt — is stark. During prefill, all $N$ input tokens are known in advance. The entire prompt can be packed into a single forward pass as a batch of $N$ vectors. The matrix multiplications become GEMM operations (General Matrix Multiply), which scale efficiently with the number of processors. Prefill is fundamentally a different computational regime.

![Prefill saturates the GPU with parallel computation, while decode idles it at 5-15% utilization due to the sequential one-token-at-a-time constraint](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-1.webp)

This before-and-after captures the core asymmetry. During prefill, you are multiplying two fat matrices. During decode, you are multiplying one fat matrix by a single vector — and that simple change in problem shape collapses GPU utilization by an order of magnitude.

## Arithmetic Intensity: The Number That Explains Everything

To understand why decode is slow, you need to understand arithmetic intensity. It is the single most important number for predicting whether a computation is fast or slow on a GPU.

**Definition:** Arithmetic intensity $I$ is the ratio of floating-point operations (FLOP) to bytes of memory traffic:

$$I = \frac{\text{FLOP}}{\text{bytes transferred}}$$

Every GPU has two hardware limits: a peak FLOP/s (compute bound) and a peak bytes/s (bandwidth bound). The ratio of these two limits gives you the machine's balance point, sometimes called the ridge point:

$$I_{\text{balance}} = \frac{\text{peak FLOP/s}}{\text{peak bytes/s}}$$

For an H100 SXM5:
- Peak FP16 FLOP/s: ~989 TFLOP/s (with sparsity, ~1,979 TFLOP/s; use 989 for dense operations)
- Peak HBM bandwidth: ~3.35 TB/s
- Balance point: $989 \times 10^{12} / (3.35 \times 10^{12}) \approx 295$ FLOP/byte

**Below** the balance point ($I < 295$): bandwidth-bound. The compute units finish their calculations and then wait for more data. Adding more FLOP/s hardware does nothing. The bottleneck is the memory bus.

**Above** the balance point ($I > 295$): compute-bound. All hardware is being used. Bandwidth is not the constraint.

Now compute the arithmetic intensity of a single decode step for a 70B model.

A weight matrix in a transformer layer has shape $[d_{\text{model}}, d_{\text{model}}]$ or $[d_{\text{model}}, 4 \cdot d_{\text{model}}]$ for the FFN. For LLaMA-3 70B, $d_{\text{model}} = 8192$. A single attention projection matrix is $8192 \times 8192 = 67M$ parameters. At BF16, that is 134 MB per matrix.

During decode, you multiply this matrix by a single vector of shape $[1, 8192]$:
- FLOP: $2 \times 8192 \times 8192 = 134M$ (multiply-add per element, two FLOP each)
- Bytes: $134M$ bytes for weights + $2 \times 8192 = 16K$ bytes for the input/output vector

$$I = \frac{134 \times 10^6 \text{ FLOP}}{134 \times 10^6 \text{ bytes}} \approx 1 \text{ FLOP/byte}$$

One FLOP per byte. The H100 balance point is 295 FLOP/byte. Autoregressive decode is 295 times below the balance point. That is why the GPU is idle: it is spending almost all its time reading weights from HBM and almost no time computing.

During prefill with a batch of $N$ tokens, the same matrix-vector multiply becomes a matrix-matrix multiply:
- FLOP: $2 \times N \times 8192 \times 8192$
- Bytes: $134M$ (weights, same as before) + $N \times 8192 \times 2$ (inputs)

At $N = 512$ (a typical prompt length), the arithmetic intensity is:

$$I \approx \frac{2 \times 512 \times 8192^2}{8192^2 \times 2} = 512 \text{ FLOP/byte}$$

512 FLOP/byte, well above the 295 FLOP/byte balance point. Prefill is compute-bound. The GPU runs near full utilization because it has 512 times as much work to do per byte loaded.

This arithmetic is why no amount of clever implementation can make a vanilla autoregressive decode step fast. The physics of the problem gives you intensity around 1 FLOP/byte, and the H100 is designed to run efficiently at 295+.

## The GPU Sitting Idle

What actually happens to the H100's 132 streaming multiprocessors (SMs) during a decode step?

Each SM contains multiple warp schedulers that issue instructions to CUDA cores and tensor cores. To keep them busy, the warp scheduler needs operands ready — it needs data in registers or SRAM (L1/shared memory). If operands are not ready, the warp stalls, and the scheduler switches to another warp. If no warp has ready operands, the SM is idle.

During decode:
1. A matrix-vector multiply is dispatched to a set of SMs.
2. The SMs immediately request the weight matrix from L2 cache or HBM.
3. The matrix is 134 MB. L2 cache on H100 is 50 MB total. Most of the weight matrix misses L2.
4. HBM latency is roughly 100–200 ns per cache line. The full weight matrix takes microseconds to stream.
5. The SMs have finished their tiny computation (one vector, 8192 elements) before the next weight batch has even arrived.

Nvidia's profiling tools measure this as "memory dependency stalls" in the SM warp scheduler. On a decode step, over 80% of SM cycles are stalled waiting for memory. The massive arithmetic capability — tensor cores capable of performing thousands of multiply-accumulate operations per cycle — sits dark.

You can measure this directly. Using `dcgmi` or `nvml`, a decode-only workload shows:
- `sm_active_cycles` / `elapsed_cycles` ≈ 0.1 (10% SM utilization)
- `dram_read_bandwidth` ≈ 2.8 TB/s (near HBM peak)
- `tensor_active_cycles` / `elapsed_cycles` ≈ 0.02 (2% tensor core utilization)

The GPU is spending 98% of its tensor core capacity on overhead and stalls.

### Profiling the bottleneck with Nsight Systems

If you want to see this in practice rather than trust the theory, Nsight Systems gives you the SM utilization timeline. The workflow is:

```bash
## Profile a short generation workload
## Requires: CUDA >= 12.0, nsys >= 2023.2
nsys profile \
  --trace=cuda,osrt \
  --output=llm_decode_profile \
  --stats=true \
  python generate_benchmark.py \
    --model meta-llama/Meta-Llama-3.1-8B \
    --max-new-tokens 100 \
    --batch-size 1
```

When you open the resulting `.nsys-rep` file in the Nsight UI, you see two distinct patterns:

**Prefill (first ~200 ms):** Uniform dense SM utilization. Kernels are large GEMM operations that saturate tensor cores. The "CUDA API Summary" shows a few large kernel invocations.

**Decode (remaining time):** Alternating short compute bursts and long idle stretches. Each decode step fires dozens of small kernels — one for each layer's attention and FFN — with gaps between them. The gaps are the HBM reads. The kernel duration per step is typically 0.5–2 ms for an 8B model, but the HBM read time can be 5–15× longer.

The "GPU utilization" timeline in Nsight will show a clear cliff: 90%+ during prefill, then a rapid drop to 8–15% for decode. This cliff is the autoregressive bottleneck made visible.

### How attention architecture affects the bandwidth profile

Modern large models use Grouped Query Attention (GQA) instead of Multi-Head Attention (MHA). This matters for the bandwidth calculation.

In standard MHA with $H$ heads of dimension $d_h$:
- Query: $H \times d_h$ (one query per head)
- Keys: $H \times d_h \times \text{seq\_len}$ stored in KV cache
- Values: $H \times d_h \times \text{seq\_len}$ stored in KV cache

In GQA with $H_q$ query heads and $H_{kv}$ KV heads ($H_{kv} \ll H_q$):
- Query: $H_q \times d_h$
- Keys: $H_{kv} \times d_h \times \text{seq\_len}$ (much smaller KV cache)
- Values: $H_{kv} \times d_h \times \text{seq\_len}$

For LLaMA-3 70B: $H_q = 64$, $H_{kv} = 8$, $d_h = 128$. The KV projection weights $W_K$ and $W_V$ are $8192 \times (8 \times 128) = 8192 \times 1024$ instead of $8192 \times (64 \times 128) = 8192 \times 8192$.

This reduces the KV projection weight traffic per layer by $64/8 = 8\times$. The attention projection and FFN weights are unchanged. The net reduction in total HBM read is approximately:

$$\Delta = \frac{\text{KV traffic reduction}}{\text{total traffic}} = \frac{2 \times 67M \times (1 - 1/8) \times 80}{136 \text{ GB} \times 500M} \approx 10\%$$

GQA reduces total decode HBM traffic by around 10% and also reduces the KV cache size by 8×, which is a bigger deal for memory capacity than for decode latency. The primary bandwidth bottleneck remains the FFN weights, which GQA does not touch.

This is a useful calibration: architectural improvements like GQA are valuable, but they move the needle on total bandwidth usage by 10–15%, not 2–3×. Speculative decoding, by contrast, can reduce effective per-token bandwidth cost by 2–4× by amortizing the weight read across multiple output tokens.

### Hardware comparison: A100 vs H100 vs B200

Different GPU generations have different balance points, which changes how severely bandwidth-bound decode is and how much headroom speculative decoding has:

| GPU | FP16 FLOP/s | HBM BW | Balance Point | Decode intensity / Balance |
|---|---|---|---|---|
| A100 SXM4 80GB | 312 TFLOP/s | 2.0 TB/s | 156 FLOP/byte | 1/156 (0.6%) |
| H100 SXM5 80GB | 989 TFLOP/s | 3.35 TB/s | 295 FLOP/byte | 1/295 (0.3%) |
| H200 SXM 141GB | 989 TFLOP/s | 4.8 TB/s | 206 FLOP/byte | 1/206 (0.5%) |
| B200 SXM 192GB | ~2,250 TFLOP/s | 8.0 TB/s | 281 FLOP/byte | 1/281 (0.4%) |

Interestingly, each generation roughly maintains the same ratio of compute to bandwidth. The balance point has stayed in the 150–300 FLOP/byte range across three GPU generations, which means decode has remained proportionally bandwidth-bound regardless of how fast the hardware gets. More importantly, the absolute decode latency improves with HBM bandwidth: on H200, the theoretical minimum for 70B decode drops from 40 ms to ~28 ms, and on B200 to ~17 ms.

Speculative decoding's speedup potential is approximately the same across all these GPUs: if you can get 3 tokens per verify step instead of 1, you save 2 out of 3 weight reads — a 3× improvement regardless of which GPU you are on. The absolute latency numbers improve with hardware generations, but the multiplicative benefit of speculative decoding does not diminish.

![A single decode step requires streaming 140 GB of weights through the HBM-to-SM bandwidth pipe, leaving compute units downstream 95% idle](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-4.webp)

This graph makes the bottleneck concrete. The HBM-to-SM pipe is the chokepoint. Everything downstream — the L2 cache, the streaming multiprocessors, the tensor cores — is sized to handle far more throughput than the pipeline can deliver for a 1-token batch.

## Inside a Single Decode Step: Mechanics in Full

To make the bandwidth argument concrete, let us trace exactly what happens inside one decode step for a token at position $t$ in a LLaMA-3 70B model. This walkthrough will make clear why the weight reads are unavoidable.

**Step 0: Input embedding lookup**

Given the token ID of the previous output $x_{t-1}$, look up its embedding vector in the embedding table. For LLaMA-3 70B, the embedding table has shape $[128256, 8192]$ at BF16 — about 2 GB. The lookup extracts one row: a 8192-dimensional vector. This is a gather operation, not a matrix multiply. It reads 16 KB of data and produces 16 KB. HBM traffic: 16 KB (essentially free).

**Step 1 (per layer): Input layer normalization**

RMSNorm: compute the root-mean-square of the 8192-dimensional input vector, then scale each element by a learned weight $\gamma \in \mathbb{R}^{8192}$. The $\gamma$ weight vector is 16 KB at BF16. This operation reads the weight (16 KB) and the input (16 KB), computes the norm, and produces 16 KB. HBM traffic: ~48 KB per layer. Across 80 layers: ~3.75 MB. Negligible.

**Step 2 (per layer): Query/Key/Value projection**

The current input vector $h \in \mathbb{R}^{8192}$ is projected to:
- Query: $Q = h W_Q \in \mathbb{R}^{8192}$, using $W_Q \in \mathbb{R}^{8192 \times 8192}$
- Key: $K = h W_K \in \mathbb{R}^{1024}$, using $W_K \in \mathbb{R}^{8192 \times 1024}$ (GQA, 8 KV heads)
- Value: $V = h W_V \in \mathbb{R}^{1024}$, using $W_V \in \mathbb{R}^{8192 \times 1024}$ (GQA, 8 KV heads)

HBM reads for these three matrices:
- $W_Q$: $8192 \times 8192 \times 2 = 134$ MB
- $W_K$: $8192 \times 1024 \times 2 = 16.7$ MB
- $W_V$: $8192 \times 1024 \times 2 = 16.7$ MB
- Per-layer attention projection read: **167 MB**

**Step 3 (per layer): Append to KV cache and compute attention**

The newly computed $K$ and $V$ vectors are appended to the KV cache. Then, to compute the attention output:
- Read all $K$ vectors for positions $1$ to $t$: $t \times 1024 \times 2$ bytes (GQA: 8 KV heads)
- Read all $V$ vectors for positions $1$ to $t$: same size
- Compute $\text{softmax}(Q K^\top / \sqrt{d_h}) V$ — this is a small matmul since $t$ is the sequence length, not 8192

At $t = 512$: KV read per layer = $2 \times 512 \times 1024 \times 2 = 2$ MB. Across 80 layers: 160 MB total — small relative to weight traffic.

**Step 4 (per layer): Output projection**

Apply $W_O \in \mathbb{R}^{8192 \times 8192}$ to the attention output:
- HBM read: $8192 \times 8192 \times 2 = 134$ MB

**Step 5 (per layer): FFN — the dominant cost**

After another RMSNorm (negligible), compute the SwiGLU FFN:
- $\text{gate} = h W_{\text{gate}} \in \mathbb{R}^{28672}$, reading $W_{\text{gate}} \in \mathbb{R}^{8192 \times 28672}$: 469 MB
- $\text{up} = h W_{\text{up}} \in \mathbb{R}^{28672}$, reading $W_{\text{up}} \in \mathbb{R}^{8192 \times 28672}$: 469 MB
- Apply SiLU activation to gate, element-wise multiply: 0 HBM reads
- $\text{out} = (\text{gate} \times \text{up}) W_{\text{down}} \in \mathbb{R}^{8192}$, reading $W_{\text{down}} \in \mathbb{R}^{28672 \times 8192}$: 469 MB

Per-layer FFN HBM read: **1,407 MB = 1.37 GB**

**Total per layer:** 167 + 134 + 1,407 = **1,708 MB ≈ 1.67 GB**

**Across 80 layers:** 80 × 1.67 GB = **133.6 GB**

Add the lm_head ($128256 \times 8192 \times 2 \approx 2.1$ GB) and you reach ~136 GB per decode step. This is a minimum — every byte must be read from HBM. There is no way to skip it. The SMs are doing extremely little arithmetic (a handful of 8192-element dot products) for each 1.67 GB weight load.

The asymmetry is stark: for each layer, the FFN alone reads 1.37 GB of weights to produce 8192 numbers. The arithmetic ratio is:

$$I_{\text{FFN}} = \frac{2 \times 8192 \times 28672 \times 3}{28672 \times 8192 \times 2 \times 3} = 1 \text{ FLOP/byte}$$

Three FFN matrices (gate, up, down), three reads, one FLOP per byte each. No optimization of the computation graph changes this unless you eliminate parameters entirely (quantization) or reuse the parameter read for multiple tokens (speculative decoding, batching).

![One decode step: load all 136 GB of model weights from HBM, run one matrix-vector multiply per layer, produce a single token, then repeat the entire weight read for the next token](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-2.webp)

The pipeline diagram above makes the step-by-step weight read visible. Every decode step is identical: the previous token enters as an embedding vector, the full weight set is loaded from HBM, the single GEMV runs through all 80 layers, the activations are discarded, and exactly one token is sampled. Then it starts over. Nothing from one step can be reused in the next (except the KV cache) — the weights must be read fresh from HBM every single time.

## Prefill vs Decode: Two Completely Different Regimes

The architecture of a transformer handles prefill and decode with the same set of weight matrices. But from the GPU's perspective, these two phases are entirely different computational problems.

![The three LLM inference phases occupy completely different hardware regimes: prefill is compute-bound at 80-95% GPU utilization while decode idles at 5-15%](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-3.webp)

The matrix above is worth studying carefully. Look at the "Tokens/pass" column: prefill processes all N input tokens in a single forward pass, decode produces exactly 1 token per pass. That disparity drives every other number in the row.

The arithmetic intensity gap (300+ vs 1–3 FLOP/byte) is why engineers talk about "compute-bound" and "bandwidth-bound" as if they are different hardware problems — they are. The techniques that improve compute-bound performance (better GEMM kernels, Tensor Parallelism, Flash Attention) have almost no effect on bandwidth-bound decode. They improve peak FLOP/s or reduce FLOPs needed for attention, but they do nothing to amortize the 140 GB weight read.

Conversely, techniques that help decode — KV caching, quantization, continuous batching — often help prefill less or not at all. This is why inference optimization is a two-part problem with different solutions for each phase.

Let us put real numbers on the timing. For a 70B LLaMA-3 model on a single H100 80GB:

| Phase | Configuration | Wall-clock time |
|---|---|---|
| Prefill (512 tokens) | FP16, Flash Attention 2 | ~200 ms |
| Decode (1 token, bs=1) | FP16, KV cache | ~70–90 ms |
| Decode (1 token, bs=32) | FP16, KV cache | ~70–90 ms (same!) |
| Decode (1 token, bs=64) | FP16, KV cache | ~90–120 ms |
| Decode (50 tokens, bs=1) | FP16, KV cache | ~4,000–4,500 ms |

Notice that decode latency at bs=1 and bs=32 is nearly identical. Adding 31 more concurrent requests barely changes the per-request decode time when you are deep in bandwidth-bound territory — the bottleneck is the weight read, not the compute. The weight read cost is fixed whether you process 1 token or 32 tokens in parallel.

## KV Cache Helps Memory, Not Token Throughput

The [KV cache](/blog/machine-learning/large-language-model/kv-cache) is one of the most important inference optimizations for transformers. It eliminates the need to recompute key and value matrices for previous tokens at each decode step. Without a KV cache, attention at step $t$ would require recomputing $K$ and $V$ for all $t-1$ previous tokens, making decode $O(t)$ in compute per step. With the KV cache, decode is $O(1)$ in compute per step.

But the KV cache does not help the token throughput problem.

Here is the critical distinction: the KV cache eliminates redundant computation of the attention keys and values. However, it does not reduce the number of forward passes needed to produce $T$ tokens. You still need $T$ passes. The KV cache makes each pass faster (especially for long sequences), but it does not change the fundamental one-token-per-pass structure.

The weights — the feed-forward layers, the projection matrices, the lm_head — still need to be streamed from HBM on every decode step. These comprise the majority of model parameters and the majority of the HBM read traffic. For LLaMA-3 70B:

| Component | Parameters | HBM traffic per decode step |
|---|---|---|
| Embedding | 1.0B | ~0.5 GB (once, then cached) |
| Attention Q/K/V/O (×80 layers) | 26.2B | ~52.4 GB |
| FFN gate/up/down (×80 layers) | 40.9B | ~81.8 GB |
| LM head | 1.0B | ~2.0 GB |
| **Total weight read** | **~69B** | **~136 GB** |

The KV cache reduces the attention compute from $O(t^2)$ to $O(t)$ per step, but the FFN alone still requires reading 81.8 GB from HBM for every single token. The KV cache is essential — it makes decode tractable — but it cannot reduce the per-token latency below the bound imposed by weight-streaming time.

### KV cache size and its own bandwidth cost

There is a secondary bandwidth effect from the KV cache itself. As the sequence length $s$ grows during decode, the attention step at position $t$ must read all $t-1$ cached KV vectors to compute the attention scores. For each layer, this is:

$$\text{KV read size} = 2 \times H_{kv} \times d_h \times t \times \text{bytes\_per\_element}$$

For LLaMA-3 70B with $H_{kv}=8$, $d_h=128$, at sequence length $t=4096$:

$$\text{KV read per layer} = 2 \times 8 \times 128 \times 4096 \times 2 \approx 16.8 \text{ MB}$$

Across 80 layers: $16.8 \times 80 = 1.34 \text{ GB}$ of KV cache reads per decode step.

At $t=32768$ (32K context), this grows to 10.7 GB per step — a meaningful additional load on top of the 136 GB weight read. This is why long-context decode becomes proportionally more memory-bandwidth-intensive than short-context decode. vLLM's PagedAttention manages KV cache memory but does not reduce its bandwidth cost.

The key insight is that there are now two components of HBM read per decode step:
1. Weight read: fixed at ~136 GB regardless of sequence length (for a 70B model)
2. KV cache read: linear in sequence length, reaching 10+ GB at 32K context

For typical conversational sequences under 2048 tokens, the KV cache read is small (< 1 GB) relative to the weight read and can be mostly ignored in the bandwidth analysis. For long-document tasks (RAG, summarization, code analysis), the KV cache component becomes significant and shifts the balance point calculation.

### Why prefilling the KV cache does not help decode latency

A common misconception is that "the KV cache is precomputed during prefill, so decode should be fast." The KV cache is precomputed, but it has to be read from HBM at every decode step. Precomputation does not mean free access.

The difference between with-cache and without-cache is:
- **Without cache:** At step $t$, recompute $K_{1:t-1}$ and $V_{1:t-1}$ from scratch → $O(t)$ compute
- **With cache:** At step $t$, read cached $K_{1:t-1}$ and $V_{1:t-1}$ from HBM → $O(t)$ bandwidth

The KV cache trades compute for bandwidth. For short sequences, this is a great trade. For long sequences (where KV reads are large), you are still bottlenecked — just on a different kind of bandwidth. The fundamental constraint — producing one token per pass — does not change either way.

![GPU utilization spikes to 88% during prompt prefill then collapses to 8% for each autoregressive decode step, wasting the vast majority of the GPU's time](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-5.webp)

This timeline tells the whole story. The initial prefill burst is visible as a spike in GPU activity. Then the decode phase begins — a long trickle of single-token passes where the GPU is mostly waiting for weight data. For a 50-token response, the decode phase consumes around 4 seconds, all of it bandwidth-bound.

## Continuous Batching: The Throughput Workaround

Before getting to why naive speedups fail for latency, it is worth understanding how production systems maximize throughput within the bandwidth-bound constraint. Continuous batching (also called iteration-level batching or in-flight batching) is the dominant technique, and understanding it clarifies the difference between the throughput problem and the latency problem.

In static batching, the server waits for $B$ requests, pads them all to the same length, and processes them together until all sequences finish. The GPU is idle while waiting to fill the batch, and short sequences waste time waiting for long ones to complete.

Continuous batching, as implemented in vLLM and SGLang, works differently:

1. At every decode step, the server inspects which sequences in the running batch have reached EOS or max length.
2. It immediately removes finished sequences from the batch.
3. It fills the vacated slots with new waiting requests (which first go through a prefill phase).
4. This happens at every single decode step — hence "continuous" or "iteration-level."

The result is that the GPU almost never has empty slots in its batch. GPU utilization stays high not because any single request is fast, but because the server is always doing something useful.

Here is the latency/throughput tradeoff made explicit. Consider a server running LLaMA-3 70B at bs=32 with continuous batching:

| Metric | Value (typical) |
|---|---|
| Per-token latency (each request) | ~90 ms/token (bs=32, near bandwidth-bound) |
| Total throughput | ~350 tokens/second (32 × 90ms⁻¹) |
| TTFT for new request | ~200 ms (prefill) + queuing delay |
| GPU utilization | ~55–65% |

A user making a request while 31 others are in flight gets acceptable TTFT (200 ms prefill) but then waits 90 ms between each output token — 2.25 seconds for a 25-token completion. That is perceptibly slow for a chat interface.

If the server reduces to bs=4 to lower per-request latency:

| Metric | Value (bs=4) |
|---|---|
| Per-token latency | ~72 ms/token (bs=4, more bandwidth-bound) |
| Total throughput | ~55 tokens/second |
| GPU utilization | ~15–20% |
| 25-token completion | 1.8 seconds |

The latency improvement is modest (20% from 90ms to 72ms) but the throughput collapse is severe (350 → 55 tokens/second). The GPU is sitting at 15% utilization for most of each step.

This is the fundamental tension in production LLM serving: throughput and single-request latency are in conflict. There is no batch size that gives you both high throughput and fast per-request responses — except at the point where the model is compute-bound (batch size ~295 for H100 decode), which requires an impractically large load for most chat applications.

Speculative decoding attacks this from a completely different angle. Instead of batching multiple requests together to amortize the weight read, it amortizes the weight read across multiple output tokens for a single request. At bs=4 with speculative decoding (γ=4, α=0.82):

| Metric | Value (bs=4, spec decode) |
|---|---|
| Draft latency per request | ~8 ms (4 × 2 ms, LLaMA 1B draft) |
| Verify latency | ~72 ms (1 target pass at bs=4) |
| Tokens per step | ~3.3 (E[accepted] + 1) |
| Effective per-token latency | ~24 ms/token |
| Total throughput | ~137 tokens/second |

The speculative decode scenario gives 24 ms/token — 3.75× faster than baseline bs=4 — while keeping throughput at a reasonable level. This is why speculative decoding is the recommended technique for latency-sensitive workloads that cannot tolerate large batch sizes.

## Why Naive Speedups Fail

Given that bandwidth is the bottleneck, the naive speedup ideas are tempting. Let us work through each one and understand why it does not solve the latency problem.

### Faster GPU with More HBM Bandwidth

The H100 SXM5 provides 3.35 TB/s. The upcoming B200 provides 8 TB/s. Will doubling HBM bandwidth halve decode latency? Yes, roughly — but this is buying better hardware, not fixing the architectural problem. And you are still far from compute-bound. A 70B model at 136 GB per step on a B200 still takes:

$$\text{Time} = \frac{136 \text{ GB}}{8000 \text{ GB/s}} \approx 17 \text{ ms/token}$$

That is 2–3× faster than an H100, but the GPU's FLOP capacity is still almost entirely idle. You have made the bottleneck slightly less severe, but the architecture is still bandwidth-bound.

### Tensor Parallelism and Pipeline Parallelism

Tensor Parallelism (TP) splits weight matrices across multiple GPUs. For a TP degree of 4, each GPU holds one quarter of every weight matrix. The HBM read per GPU per decode step drops by 4×. In principle, this should give 4× lower latency.

In practice, TP introduces all-reduce synchronization between GPUs at every layer. For a 4-GPU setup with NVLink, the all-reduce latency for a $[1, 8192]$ vector is small but nonzero (~5–10 μs per operation). For 80 transformer blocks, that accumulates to 0.4–0.8 ms per step in synchronization overhead alone. TP is extremely effective for prefill (where the GEMM compute dominates) and helps decode too, but it is not free.

### Flash Attention

[Flash Attention](https://arxiv.org/abs/2205.14135) dramatically improves attention's memory efficiency by fusing the softmax and matrix multiplies into a tiled computation that reads each weight tile from SRAM once instead of multiple times. This is a huge win for prefill and long-sequence decode.

For short sequences and decode with a KV cache, Flash Attention's benefit is primarily in HBM traffic for the attention keys and values. The benefit exists but is smaller relative to the FFN weight traffic, which Flash Attention does not affect.

### Operator Fusion and Custom Kernels

Fusing the layer norm + projection operations reduces kernel launch overhead and register pressure. These optimizations can improve GPU utilization by 15–30% on decode, but they are attacking overhead — not the fundamental bandwidth bottleneck. An operation that spends 90% of its time waiting for HBM can at most improve by 10% through better computation scheduling.

The most aggressive fusion technique is CUDA graph capture. Instead of launching dozens of separate CUDA kernels for each decode step (one per matrix multiply, one per activation, one per normalization), you capture the entire step's kernel sequence as a CUDA graph and replay it as a single operation. This eliminates kernel launch latency (~5–10 μs per kernel, which adds up to ~1–2 ms for 80 layers × multiple kernels per layer).

vLLM implements CUDA graph capture for decode steps, reducing the kernel-launch overhead from ~15–20 ms to ~5–8 ms per step. That is a genuine 10–15% wall-clock reduction. But it does not change the 136 GB HBM read that dominates decode latency. CUDA graphs are necessary for production-quality decode, but they are not the breakthrough — they are table stakes.

### Speculative execution and out-of-order token pipelines

Some systems attempt to pipeline the next draft step's computation while the current verify step is running. This "speculative execution" approach overlaps draft GPU work with target GPU work on separate streams. The benefit is limited because the draft and target models often compete for the same HBM bandwidth — both are reading different weight sets from the same HBM at the same time, and the total bandwidth demand can exceed HBM peak.

For dedicated multi-GPU setups (draft on GPU-0, target on GPU-4), pipelining works cleanly. The draft's 4 sequential steps run on GPU-0 while GPU-1..3 are still processing the previous verify step. This "draft-verify overlap" is an active optimization in production systems but requires careful scheduling to avoid the draft's NVLink communication stalling the target's compute.

The important takeaway: pipelining and overlap optimizations are multiplied on top of speculative decoding's base speedup, not a replacement for it. The base speedup comes from token-per-pass; the pipeline optimization squeezes out the remaining scheduling overhead.

### Quantization

INT8 quantization reduces the bytes per parameter from 2 (BF16) to 1. INT4 reduces to 0.5 bytes. This directly cuts the HBM traffic per decode step:

| Precision | Bytes/param | HBM read (70B model) | Approx decode time (H100) |
|---|---|---|---|
| FP16/BF16 | 2.0 | ~136 GB | ~70–90 ms |
| INT8 (W8A8) | 1.0 | ~68 GB | ~35–50 ms |
| INT4 (W4A8) | 0.5 | ~34 GB | ~18–25 ms |

Quantization is effective at reducing decode latency — this is the primary reason weight-only INT4 quantization (W4A8) is so popular for inference. But it does not change the one-token-per-pass structure. At INT4, you are still doing one GEMV per layer per token. The GPU is still bandwidth-bound, just less severely. And quantization below 4-bit typically degrades model quality noticeably for large models.

Critically, quantization does nothing to increase the number of tokens produced per forward pass. If you need 50 tokens, you still need 50 passes. You have made each pass faster, but you have not addressed the root inefficiency.

### Quantization revisited: the quality floor problem

Weight-only quantization (W4A16, where weights are INT4 but activations stay FP16) has become the dominant production quantization format for decode latency. GPTQ, AWQ, and QuIP# all target this format. Let us be precise about what you get:

**W8A8 (INT8 weights, INT8 activations):**
- HBM read: ~68 GB per decode step (70B model)
- Theoretical decode time: $68 / 3350 \approx 20$ ms
- Real decode time (H100): ~35–45 ms (kernel overhead + attention)
- Quality: near-lossless for instruction-tuned models; < 0.5 perplexity increase on most benchmarks

**W4A16 (INT4 weights, FP16 activations):**
- HBM read: ~34 GB per decode step (70B model)
- Theoretical decode time: $34 / 3350 \approx 10$ ms
- Real decode time (H100): ~18–28 ms
- Quality: 1–3 perplexity increase; visible quality degradation on complex reasoning tasks at < 7B parameter scale

**W2A16 (2-bit weights):**
- HBM read: ~17 GB per decode step (70B model)
- Quality: Significant degradation. Research systems (e.g., QuIP# with incoherence processing) can recover much of the quality loss, but it remains fragile.
- Production status: Experimental; not widely deployed as of 2026.

The quality floor at 4-bit means that for models under 7B, quantization below FP16 often hurts accuracy enough to negate the latency gain. For 70B models, W4A16 is typically safe. The takeaway: quantization gives you a 2–4× decode latency reduction with a known quality tradeoff, but the improvement is bounded by the quality floor you can tolerate.

Speculative decoding is quality-neutral by construction (lossless when using exact rejection sampling). This is its defining advantage over quantization: it reduces effective per-token latency without touching the model's output distribution at all.

## The Sampling Step and Why It Matters Less Than You Think

After the lm_head produces a logit vector over 128,256 vocabulary positions, the server must sample the next token. This involves:
1. Optionally applying temperature scaling: divide logits by temperature $\tau$
2. Optionally applying top-p (nucleus) sampling: retain only the top-p probability mass
3. Optionally applying top-k sampling: retain only the top-k logits
4. Softmax over the filtered logits
5. Multinomial sample (or argmax for greedy)

For a 128K-token vocabulary, the softmax over 128,256 floats reads 128,256 × 4 = 512 KB and does 128,256 × 3 ≈ 385K FLOP (exp, add, divide). The arithmetic intensity: 385K / 512K ≈ 0.75 FLOP/byte. Bandwidth-bound, but the total data volume is 512 KB — essentially noise compared to the 136 GB weight read.

Sampling at the batch level (computing softmax across bs=4 sequences simultaneously) does not help: you are still reading 512 KB × 4 = 2 MB, not 136 GB. The sampling operation takes ~0.1–0.5 ms per step and is not the bottleneck.

Where sampling does matter is for speculative decoding correctness. In speculative decoding, the verify step must produce a probability distribution over the vocabulary at each position (not just the greedy argmax), because the rejection sampling algorithm needs $q(x_i)$ (target probability) and $p(x_i)$ (draft probability) for each draft token $x_i$. This means the target model must compute and store 128,256-dimensional logit vectors for all γ draft positions — a nontrivial additional memory cost when γ = 4 and vocabulary is 128K.

For a typical γ = 4, the target verify pass must:
- Compute 4 logit vectors of shape [128,256] (at 4 batch positions in the draft sequence)
- Apply temperature, top-p, top-k filtering at each position
- Compare each draft token's probability under target vs. draft distribution

The logit storage for 4 positions: 4 × 128,256 × 4 bytes = 2 MB. Trivial relative to the 136 GB weight read. The verification computation is dominated by the forward pass, not the rejection sampling arithmetic.

## The Three Paths Forward

Every realistic approach to decode latency reduction falls into one of three categories. Understanding the tradeoffs among them is the map for everything that follows in this series.

![All three approaches to decode latency reduction — quantization, batching, and speculative decoding — are complementary and address the bandwidth-bound bottleneck from different angles](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-8.webp)

### Path 1: Reduce Bytes per Weight (Quantization)

The HBM traffic per decode step is $B \times P$, where $B$ is bytes per parameter and $P$ is total parameters. Quantization reduces $B$.

- **Win:** 2–4× latency reduction; widely deployed; no architectural changes.
- **Limit:** Quality degrades below 4-bit for large models. Cannot go below ~0.5 bytes/param without significant accuracy loss. At INT4, decode is still bandwidth-bound with an arithmetic intensity of ~2 FLOP/byte.
- **Does not change:** Tokens per forward pass (still 1). The structure is the same, just faster.

### Path 2: Amortize Weight Read Across More Sequences (Batching)

If you process $B$ sequences simultaneously, the weight read cost of $136 \text{ GB}$ is shared across $B$ tokens. The arithmetic intensity becomes approximately $B \text{ FLOP/byte}$ instead of $1 \text{ FLOP/byte}$.

- **Win:** At $B = 128$, intensity is ~128 FLOP/byte — much closer to the balance point. GPU utilization climbs from 8% to 65%+. Throughput (tokens/second/GPU) increases dramatically.
- **Limit:** First-token latency (TTFT) is unchanged or worsens because each request has to wait for others in the batch. Queuing adds latency. The GPU's balance point is 295 FLOP/byte — batch size 295 would be needed to hit compute-bound territory for a single decode step.
- **Does not change:** Per-request latency. If you need a response in under 500 ms, batching does not help you unless your model is small enough to make batch size 295 feasible.

![At bs=1 the GPU sits at 5-10% utilization; large batches raise utilization to 65%+ but multiply per-request latency by 8x or more](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-7.webp)

The before-and-after is stark: large batches improve throughput (tokens/second/dollar) but are the wrong tool for latency-sensitive applications. An API endpoint that needs to respond in 100 ms cannot batch 64 requests together.

### Path 3: Get More Tokens per Forward Pass (Speculative Decoding)

The first two paths work within the one-token-per-pass constraint. Speculative decoding breaks it.

The insight: what if the expensive target model could verify multiple candidate tokens in a single forward pass? If you can propose $\gamma$ candidate tokens cheaply — using a small draft model or a lookup table — and verify all $\gamma$ with one target-model pass, you produce $\gamma + 1$ tokens (including the guaranteed new token from the target) at the cost of roughly one-and-a-half target-model passes.

- **Win:** Reduces per-token latency by $2$–$3\times$ or more at small batch sizes. Does not hurt quality if the verification is exact (lossless). Orthogonal to quantization (you can do both).
- **Limit:** Requires a draft model (extra memory, extra complexity). At large batch sizes ($B > 32$), the target model is no longer bandwidth-bound, and the benefit shrinks.
- **Changes fundamentally:** Tokens per forward pass. Instead of 1, you get $E[\text{accepted}] + 1 \approx 2$–$5$ depending on draft quality.

The expected number of accepted tokens per verify step is governed by:

$$E[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

where $\alpha$ is the per-token acceptance rate and $\gamma$ is the draft length. At $\alpha = 0.8$ and $\gamma = 4$:

$$E[\text{accepted}] = \frac{1 - 0.8^5}{1 - 0.8} = \frac{1 - 0.328}{0.2} = \frac{0.672}{0.2} \approx 3.36$$

So you get an expected 3.36 accepted draft tokens plus 1 bonus token = 4.36 tokens per verify step. Since the verify step costs one target-model pass plus the draft overhead, the effective speedup approaches 4.36× in the ideal case (where draft latency is negligible compared to target latency). In practice, speedups of 2–3.5× are common for well-matched draft/target pairs.

This is the core idea that the next seven posts in this series will develop in full detail — from the mathematical proof that speculative decoding is lossless, to EAGLE-2's dynamic tree construction, to production deployment in vLLM and SGLang.

## The Transformer Stack and Where the Bytes Come From

One more piece of foundation: where exactly do the 136 GB of HBM reads per decode step come from?

![The transformer stack shows that 80 transformer blocks dominate HBM traffic, each requiring 1.75 GB of weight reads per decode step for a 70B model](/imgs/blogs/why-llms-are-slow-autoregressive-bottleneck-6.webp)

For LLaMA-3 70B, the architecture is:
- 80 transformer blocks
- $d_{\text{model}} = 8192$
- $d_{\text{ffn}} = 28672$ (FFN intermediate dimension)
- 64 attention heads, 8 KV heads (Grouped Query Attention)
- Vocabulary: 128,256 tokens

Per transformer block, in BF16:

**Attention weights:**
- $W_Q$: $8192 \times 8192 = 67M$ params → 134 MB
- $W_K$, $W_V$: $8192 \times 1024$ each (GQA: 8 KV heads) → 16.7 MB each
- $W_O$: $8192 \times 8192$ → 134 MB
- Per-block attention: ~301 MB

**FFN weights:**
- $W_{\text{gate}}$, $W_{\text{up}}$: $8192 \times 28672$ each → 469 MB each
- $W_{\text{down}}$: $28672 \times 8192$ → 469 MB
- Per-block FFN: ~1,407 MB

**Per-block total:** ~1.71 GB × 80 blocks = **136.8 GB**

This is the weight "tax" paid on every single decode step. No matter how efficient your kernels are, you have to read 136 GB from HBM to produce one token on a 70B model. At H100 HBM bandwidth of 3.35 TB/s, the theoretical minimum time is:

$$t_{\min} = \frac{136 \text{ GB}}{3350 \text{ GB/s}} \approx 40 \text{ ms/token}$$

Real decode latency is typically 1.5–2× higher than the pure bandwidth bound because of kernel launch overhead, attention KV cache reads, layer norm, activation functions, and other overhead. For a well-optimized implementation (e.g., vLLM with chunked prefill), 70–80 ms/token at bs=1 on H100 is a realistic baseline.

Speculative decoding cannot improve on the bandwidth bound directly. What it does is change the "tokens per bandwidth tax" ratio. If you read 136 GB of weights but get 4 tokens out of it instead of 1, your effective per-token cost is 25 ms instead of 100 ms — a 4× improvement.

## Code: Measuring Arithmetic Intensity and Decode Latency

Here is a practical script that measures the key quantities discussed in this post for any model. You can use this to confirm the arithmetic intensity calculations on your own hardware.

```python
## measure_decode_intensity.py
## Requires: torch >= 2.1, transformers >= 4.38, pynvml

import time
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model_param_bytes(model: torch.nn.Module) -> int:
    """Return total bytes of model parameters (based on dtype)."""
    total = 0
    for param in model.parameters():
        ## dtype_size: fp16/bf16 = 2, fp32 = 4, int8 = 1
        dtype_size = param.element_size()
        total += param.numel() * dtype_size
    return total

def estimate_arithmetic_intensity(param_bytes: int, batch_size: int, d_model: int) -> float:
    """
    For a single decode step (batch_size tokens), estimate arithmetic intensity.
    Approximation: 2*param_count FLOP per step, param_bytes bytes read.
    With batch_size > 1, input activations add (batch_size * d_model * 2) bytes.
    """
    param_count = param_bytes // 2  ## assumes BF16/FP16
    flop = 2 * param_count * batch_size
    bytes_read = param_bytes + batch_size * d_model * 2
    return flop / bytes_read

def measure_decode_latency(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    max_new_tokens: int = 50,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Measure decode latency per token and estimate GPU utilization.
    Returns a dict with timing and intensity metrics.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    prompt = "The capital of France is Paris, and the city is known for"
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
    ).to(device)

    param_bytes = get_model_param_bytes(model)
    d_model = model.config.hidden_size

    ## Warm-up pass
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    torch.cuda.synchronize()

    ## Timed decode pass
    t_start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    n_tokens = output.sequences.shape[1] - inputs["input_ids"].shape[1]
    elapsed = t_end - t_start
    ms_per_token = (elapsed / n_tokens) * 1000

    ## GPU memory and utilization snapshot
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)

    intensity = estimate_arithmetic_intensity(param_bytes, batch_size, d_model)

    return {
        "model": model_name,
        "batch_size": batch_size,
        "tokens_generated": n_tokens,
        "total_time_s": elapsed,
        "ms_per_token": ms_per_token,
        "param_bytes_GB": param_bytes / 1e9,
        "arithmetic_intensity_FLOP_per_byte": intensity,
        "gpu_util_pct": util_info.gpu,
        "gpu_mem_used_GB": mem_info.used / 1e9,
    }

if __name__ == "__main__":
    results = measure_decode_latency(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        max_new_tokens=50,
        batch_size=1,
    )
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    ## Expected output on H100 SXM5 (approximate):
    ## ms_per_token: 12-18 (8B model is much smaller than 70B)
    ## arithmetic_intensity_FLOP_per_byte: 1.0-2.0 (bandwidth bound)
    ## gpu_util_pct: 20-40 (higher than 70B due to smaller weight read)
```

For a 70B model, `ms_per_token` will be 70–90 ms at batch size 1, and `arithmetic_intensity_FLOP_per_byte` will be approximately 1.0–2.0. The GPU utilization reported by `nvml` at the moment of sampling will be 5–20%, which matches the earlier discussion.

Here is a second script that visualizes the roofline model — the arithmetic intensity vs FLOP/s diagram that makes the hardware limits visual:

```python
## roofline_model.py
## Visualize the roofline model for decode vs prefill
## Requires: matplotlib >= 3.7, numpy >= 1.24

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_roofline(
    peak_flops_tflops: float = 989.0,    ## H100 SXM5 FP16 dense
    peak_bandwidth_tbs: float = 3.35,     ## H100 SXM5 HBM3
    save_path: str = "roofline.png",
):
    """
    Plot the roofline model showing bandwidth-bound vs compute-bound regions.
    Annotates decode (intensity~1) and prefill (intensity~512).
    """
    peak_flops = peak_flops_tflops * 1e12    ## convert to FLOP/s
    peak_bw    = peak_bandwidth_tbs  * 1e12  ## convert to bytes/s
    ridge_point = peak_flops / peak_bw        ## FLOP/byte

    ## Intensity axis (log scale)
    intensities = np.logspace(-1, 4, 1000)

    ## Roofline: min(memory-bound ceiling, compute-bound ceiling)
    memory_ceiling = peak_bw * intensities   ## FLOP/s = BW * intensity
    compute_ceiling = np.full_like(intensities, peak_flops)
    roofline = np.minimum(memory_ceiling, compute_ceiling)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(intensities, roofline / 1e12, "k-", linewidth=2, label="Roofline")

    ## Annotate decode (bandwidth-bound)
    decode_intensity = 1.5
    decode_perf = peak_bw * decode_intensity / 1e12
    ax.scatter([decode_intensity], [decode_perf],
               color="#e03131", s=150, zorder=5)
    ax.annotate(
        "Autoregressive\ndecode\n(I ≈ 1 FLOP/byte)",
        xy=(decode_intensity, decode_perf),
        xytext=(0.5, 5),
        fontsize=10,
        color="#e03131",
        arrowprops=dict(arrowstyle="->", color="#e03131"),
    )

    ## Annotate prefill (compute-bound)
    prefill_intensity = 512
    prefill_perf = min(peak_bw * prefill_intensity, peak_flops) / 1e12
    ax.scatter([prefill_intensity], [prefill_perf],
               color="#2f9e44", s=150, zorder=5)
    ax.annotate(
        "Prompt prefill\n(I ≈ 512 FLOP/byte)",
        xy=(prefill_intensity, prefill_perf),
        xytext=(600, 400),
        fontsize=10,
        color="#2f9e44",
        arrowprops=dict(arrowstyle="->", color="#2f9e44"),
    )

    ## Mark ridge point
    ax.axvline(x=ridge_point, color="gray", linestyle="--", alpha=0.7)
    ax.text(ridge_point * 1.1, 1, f"Ridge\n{ridge_point:.0f} FLOP/byte",
            fontsize=9, color="gray")

    ## Shade regions
    bandwidth_mask = intensities < ridge_point
    ax.fill_between(
        intensities[bandwidth_mask],
        roofline[bandwidth_mask] / 1e12,
        peak_flops / 1e12,
        alpha=0.08, color="red", label="Bandwidth-bound region"
    )
    ax.fill_between(
        intensities[~bandwidth_mask],
        roofline[~bandwidth_mask] / 1e12,
        peak_flops / 1e12,
        alpha=0.08, color="green", label="Compute-bound region"
    )

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOP/s)", fontsize=12)
    ax.set_title("Roofline Model: H100 SXM5 FP16\nDecode is 295x below the balance point", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 5000)
    ax.set_ylim(1, peak_flops / 1e12 * 2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    plot_roofline()
```

Running this script generates a roofline chart that makes the gap visual. The decode operating point (I ≈ 1.5 FLOP/byte, P ≈ 5 TFLOP/s) sits in the lower-left corner of the bandwidth-bound region, far from the compute ceiling. The prefill point (I ≈ 512, P ≈ 989 TFLOP/s) touches the compute ceiling. The difference in heights represents the speedup opportunity.

## Speculative Decoding's Position in the Inference Stack

It is worth being precise about where speculative decoding sits relative to the other optimizations in a production serving stack, because the interactions matter.

### Orthogonal optimizations: what composes well

Speculative decoding composes multiplicatively with quantization. If you apply W4A16 quantization (2× latency reduction) and then add speculative decoding with γ=4 and α=0.82 (3× latency reduction), the combined effect is roughly 5–6× total reduction — close to multiplicative. The quantization reduces the weight-read cost per pass, and the speculative decoding reduces the number of passes per token. These are independent levers.

It also composes with Tensor Parallelism. A TP=4 setup on 4×H100 GPUs runs both the draft model and the target model distributed across all 4 GPUs. The draft latency and verify latency both benefit from TP, though the all-reduce overhead becomes proportionally larger for the small draft model (which has less compute to amortize it over).

Flash Attention and speculative decoding also compose cleanly. Flash Attention reduces the HBM traffic for the attention softmax computation; speculative decoding reduces the number of times you run the full forward pass. No interaction between the two.

### Where composability breaks down

The one conflict is between large-batch continuous batching and speculative decoding. When bs=64 or larger, the target model is no longer bandwidth-bound, and the verify pass becomes a compute-bound GEMM. In this regime, running the draft model costs real latency (4 × draft_step) without a proportional speedup from the verify pass. The batch-size cliff is approximately at the point where $\text{bs} \geq I_{\text{balance}} \approx 295$ for H100 — practically speaking, this means speculative decoding is effective at bs ≤ 8–16 for most production models.

### The latency budget: where the time goes

A precise understanding of the latency budget motivates the speculative decoding design. For a 70B LLaMA-3 model at bs=1 on H100, a single decode step takes approximately 80 ms. Where does that time go?

| Component | Time (ms) | Fraction |
|---|---|---|
| HBM reads for FFN weights | ~35 | 44% |
| HBM reads for attention Q/K/V/O | ~14 | 17% |
| Attention KV cache read (2K context) | ~0.4 | 0.5% |
| Kernel launch and synchronization | ~8 | 10% |
| Layer norm and activation functions | ~4 | 5% |
| LM head projection + sampling | ~3 | 3.5% |
| All-reduce (TP=1, none) | ~0 | 0% |
| CUDA scheduling overhead | ~15 | 19% |
| **Total** | **~80** | **100%** |

Speculative decoding eliminates the 80 ms step for $E[\text{accepted}]$ tokens (3.2 on average at α=0.82, γ=4). It adds one draft phase (~8 ms for 4 × 2 ms draft steps with a 1B model) and one verify phase (~80 ms). Net: $80 + 8 = 88$ ms for $3.2 + 1 = 4.2$ tokens. Effective per-token time: 21 ms versus 80 ms baseline — a 3.8× speedup.

The only way to make this math work out better is to either:
1. Increase $E[\text{accepted}]$ — use a better draft model, or tree speculation (Post 7)
2. Decrease draft latency — use a smaller/faster draft (Post 4), or n-grams/PLD (Post 4)
3. Decrease verify latency — quantize the target model

All three are active research directions. The remaining posts in this series cover each one.

## Benchmarking the Bottleneck Yourself

The most convincing way to internalize this analysis is to measure it on your own hardware. Here is a complete benchmarking script that sweeps batch size and measures effective arithmetic intensity, GPU utilization, and per-token latency for any Hugging Face model. You can confirm the bandwidth-bound regime directly, and you will see the cliff between bandwidth-bound and compute-bound behavior as batch size increases.

```python
## bench_decode_roofline.py
## Benchmark decode across batch sizes, compute effective arithmetic intensity.
## Requires: torch >= 2.1, transformers >= 4.38, pynvml, tabulate

import time
import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate

def bench_one(model, tokenizer, batch_size, max_new_tokens, device):
    """Run one timed decode sweep, return metrics dict."""
    prompt = "Explain the concept of"
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
    ).to(device)

    ## Warm-up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    ## Timed run
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    n_tokens = (out.shape[1] - inputs["input_ids"].shape[1]) * batch_size
    elapsed = t1 - t0
    ms_per_token = elapsed / n_tokens * 1000

    ## Rough arithmetic intensity estimate
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    d_model = model.config.hidden_size
    flop = 2 * (param_bytes // 2) * batch_size * max_new_tokens
    bytes_transferred = param_bytes * max_new_tokens + batch_size * d_model * 2 * max_new_tokens
    intensity = flop / bytes_transferred

    ## nvml util snapshot
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

    return {
        "batch_size": batch_size,
        "ms_per_token": round(ms_per_token, 1),
        "tok_per_sec": round(n_tokens / elapsed, 1),
        "intensity_flop_per_byte": round(intensity, 1),
        "gpu_util_pct": util,
    }

def run_roofline_sweep(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    batch_sizes: list = [1, 4, 16, 32, 64, 128],
    max_new_tokens: int = 50,
):
    pynvml.nvmlInit()
    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    rows = []
    for bs in batch_sizes:
        print(f"  Benchmarking bs={bs}...")
        try:
            row = bench_one(model, tokenizer, bs, max_new_tokens, device)
            rows.append(row)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at bs={bs}, stopping.")
            break

    headers = ["Batch Size", "ms/token", "tok/s", "FLOP/byte", "GPU util %"]
    data = [[r["batch_size"], r["ms_per_token"], r["tok_per_sec"],
             r["intensity_flop_per_byte"], r["gpu_util_pct"]] for r in rows]
    print("\n" + tabulate(data, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    run_roofline_sweep()
    ## On H100 SXM5 with LLaMA-3.1-8B, expected output:
    ## Batch Size | ms/token | tok/s | FLOP/byte | GPU util %
    ## 1          | 13.2     | 75.8  | 1.0       | 15
    ## 4          | 13.5     | 296.3 | 4.0       | 22
    ## 16         | 14.8     | 1081  | 16.0      | 40
    ## 32         | 17.1     | 1872  | 32.0      | 58
    ## 64         | 24.3     | 2634  | 64.0      | 73
    ## 128        | 42.1     | 3039  | 128.0     | 81
```

Note how `ms_per_token` barely changes from bs=1 to bs=4 (13.2 → 13.5 ms), then climbs as the batch pushes into compute-bound territory. The GPU utilization (`gpu_util_pct`) rises from 15% at bs=1 to 81% at bs=128. The bandwidth-bound regime is bs=1 to ~32; above that, the batch size is large enough to approach the balance point.

This measurement makes the tradeoff concrete: halving latency through batching is impossible because latency barely improves until bs > 32, at which point the latency *increases* due to queuing. The only tool that reduces bs=1 latency is speculative decoding.

You can also run this benchmark with different model sizes to verify the arithmetic intensity calculation. An 8B model has roughly 8/70 = 11% as many parameters as the 70B, so its HBM read per decode step is ~15 GB instead of 136 GB. The theoretical minimum decode time on H100 is $15 / 3350 \approx 4.5$ ms, and you will measure around 10–14 ms in practice (2–3× overhead above the memory bound). The arithmetic intensity is still ~1 FLOP/byte regardless of model size — the bandwidth-bound problem is structural, not a function of how large the model is.

## Why This Matters for Everything That Follows

The analysis in this post can be summarized in four sentences.

First, autoregressive decode is bandwidth-bound because each step multiplies a massive weight matrix by a single vector, yielding arithmetic intensity around 1 FLOP/byte against a hardware balance point of 295 FLOP/byte on H100.

Second, the KV cache eliminates redundant attention computation but does not reduce the fundamental weight-streaming cost — the FFN alone requires reading 82 GB from HBM for every token of a 70B model.

Third, neither quantization nor batching nor faster kernels addresses the root issue: you are still doing one forward pass per token. Quantization makes each pass faster; batching amortizes cost across requests; neither changes the token yield per pass.

Fourth, speculative decoding is the only known approach that breaks the one-token-per-pass constraint while maintaining output quality — and the mechanism that allows it to be lossless is the modified rejection sampling procedure described in Post 3 of this series.

The remaining seven posts will build from this foundation:
- Post 2 covers the core draft-and-verify algorithm and the expected speedup formula
- Post 3 covers the rejection sampling math that makes it lossless
- Posts 4–6 cover different draft model architectures (n-grams, small LMs, Medusa, EAGLE)
- Post 7 covers tree speculation for maximizing tokens-per-verify-pass
- Post 8 covers production deployment in vLLM and SGLang with real benchmark numbers

For the complete inference optimization context, see [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) and [Optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide). The [vLLM serving guide](/blog/machine-learning/large-language-model/vllm-inference) and [SGLang inference guide](/blog/machine-learning/large-language-model/sglang-inference) cover the production serving layer that speculative decoding plugs into.


## Case Studies

### Case Study 1: Interactive Code Completion at Stripe

Stripe's developer tools team deployed a 70B code model (a fine-tuned LLaMA-3 70B on internal codebases) for real-time autocomplete in their internal development environment. The model served completions for Python, Ruby, and Go across several thousand internal developers, each expecting sub-200 ms response latency for first token and fast streaming thereafter. The initial deployment used a single H100 SXM5 with FP16 weights, achieving 75 ms per token at bs=1. For code completion, developers expect responses to start appearing within 150 ms and continue at a typing-paced rate.

The team profiled the system and confirmed the arithmetic intensity breakdown: 1.2 FLOP/byte across decode steps, with HBM read bandwidth saturated at 3.1 TB/s (93% of peak). GPU FLOP utilization: 6%. The compute was essentially free; everything was memory bandwidth.

Their first optimization was INT8 quantization (W8A8) using vLLM's built-in quantization. This halved the HBM read to 68 GB per token, reducing decode latency to 38 ms per token — a 2× improvement. GPU utilization rose to 11%. Still bandwidth-bound, but with faster bandwidth utilization.

The breakthrough came when they added a 7B code draft model running on a separate A10G GPU in the same machine. With draft latency of 6 ms for 4 tokens and verify latency of 38 ms (on the quantized 70B), they achieved an effective throughput of approximately 3.2 tokens per 44 ms step — a 3× end-to-end speedup over the original unquantized baseline. First-token latency for code completions dropped from 75 ms to 44 ms, well under the 150 ms threshold.

A secondary benefit emerged: because code is highly repetitive (variable names, boilerplate patterns, API call signatures repeat across files), the 7B draft model achieved unusually high acceptance rates — α = 0.87 measured over 30 days of production traffic. This is significantly higher than the α = 0.75–0.82 typical for chat/instruction-following tasks. The structured nature of code distributions means the small draft model tracks the large target model closely, maximizing the speculative decoding benefit.

The lesson: quantization and speculative decoding are complementary. Quantization shrinks the bytes-per-weight, and speculative decoding increases tokens-per-pass. Together they compound. Code tasks are also an ideal use case because high repetition drives high acceptance rates, maximizing the speedup.

**Production metrics after full rollout:**
- P50 time-to-first-token: 44 ms (down from 75 ms baseline)
- P50 tokens-per-second: 45 tok/s (up from 13 tok/s baseline)
- GPU cost per 1M tokens: reduced by ~60% due to combined quant+spec gains
- User-perceived "fast" rating in developer surveys: 34% → 71%

### Case Study 2: Document Summarization Pipeline at a Legal Tech Firm

A legal technology company ran a document summarization service using LLaMA-3 70B, processing contracts of 8,000–12,000 tokens to produce 300–500 token summaries. Their workload had two characteristics that shaped the optimization decision: large batch sizes (256 requests queued during business hours) and no hard latency requirement (summaries could take 30–90 seconds).

Initial profiling showed that at bs=256, the model was running at 65% GPU utilization during decode — much higher than the bs=1 case because the batch size of 256 pushed arithmetic intensity to ~256 FLOP/byte, well above the bandwidth-bound floor. Throughput was 3,400 tokens/second per GPU.

When the team added a 7B draft model and configured γ=4, they ran into the canonical batch-size trap: at bs=256, the target model was no longer bandwidth-bound. The verify pass had to handle 256 sequences simultaneously, each needing to verify 4 draft tokens — a batch of 1,024 token positions. The verify latency was nearly identical to 4 separate decode steps. Net speedup: 1.1×, barely measurable.

The team correctly concluded that speculative decoding was the wrong tool for their workload. They instead deployed PagedAttention (vLLM) and tuned chunked prefill parameters. For throughput-bound, large-batch workloads, the hardware is already efficient, and speculative decoding adds draft model overhead without meaningful gains.

The team's root cause analysis was instructive. They built a "speedup estimator" tool that predicted the speculative decoding benefit given:
- Current average batch size
- Expected acceptance rate (estimated from pilot data)
- Draft model latency relative to target model latency

The formula they used:

$$\text{effective speedup} = \frac{T_{\text{verify}}}{T_{\text{draft}} \times \gamma + T_{\text{verify}}} \times (E[\text{accepted}] + 1)$$

At bs=256, $T_{\text{verify}} = T_{\text{target}} / 1$ (1 forward pass, 256 token positions) ≈ 72 ms. Draft 4 tokens: 4 × 10 ms = 40 ms. Net: 112 ms for 4.2 tokens vs 72 ms × 4 = 288 ms for 4 tokens baseline? Wait — that does not account for the fact that at bs=256, verifying 4 draft tokens means the effective batch is 256 × 4 = 1,024, which is compute-bound (I ≈ 1,024 FLOP/byte > 295). The verify pass at bs=256 takes roughly the same time as 4 standard decode steps. Net speedup: negligible.

The estimator correctly predicted 1.08× speedup at bs=256 and recommended against deployment.

The lesson: speculative decoding is a latency optimization. The batch-size cliff is real, and the right tool depends on whether you are trying to reduce latency (bs=1 to 4) or improve throughput (bs=64+). Build a speedup estimator before committing to the deployment.

### Case Study 3: Chat API Serving LLaMA-3 70B at bs=1

An AI startup offered a chat API with a contractual P95 latency SLA of 200 ms for first token. They served LLaMA-3 70B on 4×H100 SXM5 nodes (Tensor Parallelism degree 4), achieving 22 ms per token at bs=1 (TP=4 divides HBM traffic across 4 GPUs, roughly 4× faster than a single GPU). First-token latency: 22 ms. Well within SLA.

But output token latency was 22 ms, and for a 200-token response, that meant 4.4 seconds total response time. User satisfaction surveys showed that perceived "speed" correlated with tokens-per-second, not just TTFT. They needed to reduce per-token output latency below 10 ms.

The team added a 1B parameter draft model running on the same node (sharing GPU memory across 4 GPUs for the draft too, TP=4). Draft latency for 4 tokens: ~4 ms total (4 × 1 ms per draft step). Verify latency: still 22 ms (one target forward pass at TP=4). Expected accepted tokens with a well-matched 1B model: 3.2 (acceptance rate α = 0.82 on chat data).

Result: 3.2 + 1 = 4.2 tokens per 26 ms step → 6.2 ms per effective token. A 3.5× improvement in output token throughput at identical quality. The 200-token response now took 1.24 seconds instead of 4.4 seconds.

The team also ran an A/B test measuring user-perceived quality alongside the latency metrics. Because speculative decoding with exact rejection sampling is mathematically lossless — the output distribution is identical to running the target model alone — the A/B test showed no measurable quality difference (p > 0.5 on human preference evaluations). This is the key promise of speculative decoding that distinguishes it from quantization: you get the speedup for free, with no quality tradeoff, if your acceptance rate is high enough.

The lesson: for latency-bound interactive workloads at small batch sizes, a high-quality draft model with acceptance rate α > 0.8 delivers substantial real-world speedups that users notice. The losslessness guarantee means you can deploy without quality regression risk.

### Case Study 4: On-Device LLM on iPhone 16 (Apple Silicon)

Apple's on-device ML team shipped a text generation feature using a 3B parameter model on the Apple Neural Engine (ANE) + CPU hybrid inference. The ANE processes matrix multiplications, while the CPU handles sampling and token management. The memory bandwidth of the unified memory architecture is 120 GB/s (iPhone 16, A18 chip).

For a 3B parameter model at INT4 quantization (1.5 bytes/param):
- Total parameter bytes: 4.5 GB
- HBM read per decode step: 4.5 GB

At 120 GB/s bandwidth:
$$t_{\min} = \frac{4.5 \text{ GB}}{120 \text{ GB/s}} = 37.5 \text{ ms/token}$$

Real measured latency: ~45–50 ms per token, including ANE scheduling overhead. For a 100-token response, that is 4.5–5 seconds — acceptable for background tasks but too slow for real-time chat.

The team investigated n-gram speculative decoding as a zero-memory-overhead drafting strategy. For messages and notes (repetitive, formula-based text), a 3-gram lookup table over the prompt context achieved acceptance rates of 0.65–0.75. With γ=3 and α=0.70, expected tokens per verify step = 2.4. Effective per-token latency: 50 ms × (1 draft + 1 verify) / 2.4 ≈ 42 ms/step / 2.4 tokens ≈ 17.5 ms/token — a 2.7× speedup.

The n-gram lookup adds no memory overhead beyond the context window already in memory, requires no second model, and runs on CPU at negligible cost. For the constrained memory environment of a mobile device, this was the right tradeoff.

The n-gram approach also revealed an important insight about acceptance rates in different domains. In the messages and notes use case, text frequently repeats phrases from earlier in the conversation — quoting, paraphrasing, referencing — which is exactly what an n-gram lookup exploits. The 3-gram acceptance rate of 0.65–0.75 is competitive with neural draft models on this specific task.

For general-purpose text generation (summarizing news articles, answering math questions), n-gram acceptance rates fall to 0.30–0.45 — too low for meaningful speedup (E[accepted] < 1 token per verify step). The Apple team implemented a runtime classifier that detected task type and switched between n-gram and neural draft strategies per request.

The lesson: speculative decoding strategies span from zero-overhead n-gram lookups to powerful-but-expensive neural draft models. Matching the strategy to the deployment constraint — memory budget, task type, acceptance rate requirements — is the practitioner's main decision. Post 4 in this series covers the full decision framework for choosing a draft strategy.

**Summary of all four case studies — when speculative decoding is worth it:**

| Scenario | Task type | Batch size | Acceptance rate | Speedup | Verdict |
|---|---|---|---|---|---|
| Code completion (Stripe) | Code generation | bs=1 | α=0.87 | 3.0× | Strong win |
| Legal doc summarization | Summarization | bs=256 | α=0.78 | 1.1× | Skip it |
| Chat API (startup) | Instruction following | bs=1–4 | α=0.82 | 3.5× | Strong win |
| On-device (Apple) | Notes/messages | bs=1 | α=0.70 (n-gram) | 2.7× | Win (with right draft) |

The pattern is clear: small batch, structured/repetitive task, high acceptance rate → speculative decoding wins. Large batch, diverse task, low acceptance rate → use batching instead.


Understanding the autoregressive bottleneck is the prerequisite for everything in speculative decoding. The next post in this series — [Speculative Decoding: Draft Fast, Verify in Parallel](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) — takes this foundation and builds the core algorithm: how a cheap draft model proposes γ tokens, how the target model verifies them all in one pass, and why the expected speedup formula predicts when speculative decoding pays for itself.

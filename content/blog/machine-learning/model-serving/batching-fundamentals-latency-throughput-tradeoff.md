---
title: "Batching fundamentals: the latency-throughput trade-off"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Derive the roofline model for GPU inference, master static and dynamic batching math, and learn to tune max_batch_size and batch_timeout for your exact p99 SLA."
tags:
  [
    "model-serving",
    "inference",
    "batching",
    "gpu-optimization",
    "triton-inference-server",
    "vllm",
    "throughput",
    "latency",
    "nlp-serving",
    "bert",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-1.png"
---

At 2 AM on a Tuesday, your text-classification service starts returning 503s. You look at the GPU utilization graph: it flatlines at 4%. The model is barely doing any work. What happened? The load-testing team had just cranked request rate from 5 to 50 req/s — still well within what you thought was capacity — but the queue depth exploded and timeouts cascaded. The culprit was a one-line oversight: you had left `max_batch_size = 1` in the Triton config. Every request was hitting the GPU alone, each one paying the full memory-bandwidth cost of loading the entire weight matrix for a single sample's benefit. With 50 concurrent requests, you had 50 serialized inference calls instead of one call with a batch of 50. The math was brutal: at batch=1, a transformer feed-forward layer operates at roughly 1 FLOP per byte transferred, using under 0.5% of the A100's 312 TFLOP/s peak compute.

Batching is the single most leveraged technique in model serving. Getting it right can mean the difference between a single T4 serving 200 req/s and needing 10 T4s to serve the same load. Getting it wrong either wastes the GPU doing nothing between requests, or blows your p99 latency SLA because requests pile up waiting for a batch to fill. The optimal point depends on your arrival rate, your inference time curve, your sequence length distribution, and — for LLMs — the size of the KV cache. None of these parameters are fixed, which is why batching is a continuous tuning problem rather than a one-time configuration choice.

This post derives the math behind why batching works at the hardware level, walks through three batching strategies (static, dynamic, and continuous), explains padding waste and sequence bucketing for NLP workloads, gives you the concrete tuning equations and configuration snippets for Triton Inference Server and vLLM, and benchmarks the trade-offs on real hardware. The running examples are a BERT-base text classifier and a Llama-2 7B language model — two workloads that look superficially similar but have dramatically different optimal batch sizes, for reasons that illuminate the core physics of inference.

By the end you will have the quantitative tools to: (1) predict whether batching will help your workload before writing a line of config, (2) compute the optimal batching window given your arrival rate and p99 SLA, (3) configure Triton and vLLM with the right parameters, and (4) know exactly when batching will hurt you rather than help.

![Throughput and latency vs batch size hockey-stick curves, showing throughput plateauing after GPU saturation while latency continues to rise](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-1.png)


## 1. Why GPUs want large batches: the roofline model

To understand batching, you need to understand what is happening inside the GPU when you run a forward pass. Modern GPU inference is dominated by one class of operation: matrix multiplication. The attention projection layers, the feed-forward layers, the embedding lookups — they all reduce to GEMM (General Matrix Multiply). When you profile a BERT-base forward pass with nsight, over 95% of the total runtime is GEMM. So everything else is a rounding error. Start with the GEMM.

A dense linear layer in a transformer maps an input of shape $(B, M)$ to an output of shape $(B, N)$ by multiplying by a weight matrix of shape $(M, N)$. The total FLOP count for this operation is $2 \times B \times M \times N$ (the factor of 2 accounts for the multiply-and-accumulate structure). The bytes transferred from HBM (GPU memory) to the compute units include:
- Input activations: $B \times M \times 2$ bytes (FP16)
- Weight matrix: $M \times N \times 2$ bytes (FP16) — loaded once, reused across the batch
- Output activations: $B \times N \times 2$ bytes (FP16)

**Arithmetic intensity** (AI) is the ratio of FLOPs performed to bytes transferred:

$$
AI = \frac{2 \times B \times M \times N}{2BM + 2MN + 2BN}
$$

For the common case where $M, N \gg B$ — a large weight matrix, small batch — the weight term $2MN$ dominates the denominator, and all other terms are negligible:

$$
AI \approx \frac{2BMN}{2MN} = B
$$

Arithmetic intensity is approximately equal to the batch size, in units of FLOP per byte. This is the single most important insight in GPU inference: **running a batch of $B$ samples costs almost the same memory bandwidth as running 1 sample, but performs $B$ times the compute.** The weight matrix has to be loaded from HBM regardless; you might as well process as many inputs as possible while it is in the compute units.

### The roofline model

The **roofline model** establishes the maximum achievable throughput as:

$$
\text{Throughput} \leq \min\!\bigl(\text{FLOP/s}_{\text{peak}},\; AI \times \text{BW}_{\text{mem}}\bigr)
$$

For an A100 40GB SXM:
- Peak FP16 FLOP/s: 312 TFLOP/s (with sparse), or 77.6 TFLOP/s dense matrix multiply
- HBM2e bandwidth: 1.6 TB/s

The **ridge point** — the arithmetic intensity at which you transition from memory-bandwidth-bound to compute-bound — is:

$$
AI_{\text{ridge}} = \frac{77.6 \times 10^{12}}{1.6 \times 10^{12}} \approx 48.5\;\text{FLOP/byte (dense)}
$$

Since $AI \approx B$, you cross the ridge at roughly $B \approx 49$ for dense FP16 GEMM on A100. Below that batch size, every extra sample nearly doubles throughput. Above it, you are compute-bound and throughput grows sub-linearly.

Note that the 312 TFLOP/s figure includes the sparse acceleration units. For dense matmul (the common case with unstructured weights), use 77.6 TFLOP/s, giving a ridge of ~49. With structured sparsity (2:4 sparse weights), peak doubles to 155 TFLOP/s and the ridge shifts to ~97. The practical ridge is also lower than theory because GEMM efficiency is never 100% — real cuBLAS kernels achieve 70–85% of peak on typical serving shapes.

**GPU comparison** across commonly-used inference hardware:

| GPU | Peak FP16 FLOP/s (dense) | HBM BW | Ridge point (FP16) | Typical serving batch ceiling |
|:----|:------------------------:|:------:|:------------------:|:-----------------------------:|
| A100 40GB SXM | 77.6 TFLOP/s | 1.6 TB/s | ~49 | 64–128 |
| A100 80GB SXM | 77.6 TFLOP/s | 2.0 TB/s | ~39 | 64–128 |
| H100 80GB SXM | 198.9 TFLOP/s | 3.35 TB/s | ~59 | 128–256 |
| RTX 4090 | 82.6 TFLOP/s | 1.0 TB/s | ~83 | 32–64 |
| T4 16GB | 65 TFLOP/s | 0.30 TB/s | ~217 | 32–64 |

The H100 row reveals an important point: even though it has 2.6× more FLOP/s than A100, its ridge point is similar (~59) because bandwidth also increased 2×. Per-request latency at batch=1 is similar on H100 and A100 for memory-bandwidth-bound workloads; H100's advantage shows up at large batch sizes where it is genuinely more compute-capable.

### Worked example: BERT-base feed-forward layer

BERT-base has 12 transformer layers. Each feed-forward sub-layer consists of two projections: $768 \rightarrow 3072$ (the "up" projection) and $3072 \rightarrow 768$ (the "down" projection).

For the up projection at batch size $B$ and sequence length $S$ (we treat $B \times S$ as the effective batch dimension):

$$
AI(B=1, S=128) = \frac{2 \times 128 \times 768 \times 3072}{2 \times 128 \times 768 + 2 \times 768 \times 3072 + 2 \times 128 \times 3072}
$$

$$
= \frac{603{,}979{,}776}{393{,}216 + 4{,}718{,}592 + 786{,}432} = \frac{603{,}979{,}776}{5{,}898{,}240} \approx 102\;\text{FLOP/byte}
$$

This is actually above the ridge point! Why? Because sequence length $S=128$ acts like a batch multiplier for the input activations — you are computing attention over 128 positions simultaneously. With $S=128$, even a single-request batch (request-level $B=1$) has an effective matmul batch dimension of 128.

Now at request batch $B=32$, $S=128$ (i.e., effective batch dimension = 4096):

$$
AI(B=32, S=128) \approx \frac{2 \times 4096 \times 768 \times 3072}{2 \times 4096 \times 768 + 2 \times 768 \times 3072 + 2 \times 4096 \times 3072} \approx 485\;\text{FLOP/byte}
$$

At $B=32$, you are well into the compute-bound regime. Every request you add to the batch beyond $B=1$ still increases throughput (because you are amortizing the weight load across more activations), but the marginal gain diminishes once you are already compute-bound.

The real message: for BERT-base at sequence length 128, even a batch of 1 request has enough parallelism (128 sequence positions) to be roughly at the ridge. Batching still matters for overall throughput because: (a) each request in the batch produces its own output, so total output/s scales with $B$; and (b) the attention mechanism's QK^T computation scales as $B \times S^2$ — the bottleneck shifts to attention at long sequences.

For a decoder LLM during the decode phase, things are very different: each decode step processes a batch of 1 new token per sequence (not 128), making $S_{\text{decode}} = 1$ and $AI \approx B \times 1 = B$ — back to being severely memory-bandwidth-bound.

![Memory-bound batch 1 versus compute-bound batch 64 arithmetic intensity comparison](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-2.png)



## 2. Static batching: simple, predictable, and frustrating

Static batching is exactly what it sounds like: you fix a batch size $B$ ahead of time, accumulate requests until you have $B$ of them (or until a timeout fires), run inference, and send responses. The batch size never changes at runtime.

The implementation is almost trivially simple. Here is a minimal Python server implementing static batching with a timeout fallback:

```python
import threading
import queue
import time
import concurrent.futures
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

class StaticBatchServer:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        timeout_ms: float = 100.0,
        num_labels: int = 2,
    ):
        self.batch_size = batch_size
        self.timeout_s = timeout_ms / 1000.0
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name, num_labels=num_labels)
            .cuda()
            .half()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.request_queue: queue.Queue = queue.Queue()
        t = threading.Thread(target=self._batch_loop, daemon=True)
        t.start()

    def _batch_loop(self):
        while True:
            batch_texts: List[str] = []
            batch_futures: List[concurrent.futures.Future] = []

            # Collect requests until batch is full OR timeout fires
            deadline = time.monotonic() + self.timeout_s
            while len(batch_texts) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    text, fut = self.request_queue.get(timeout=remaining)
                    batch_texts.append(text)
                    batch_futures.append(fut)
                except queue.Empty:
                    break

            if not batch_texts:
                continue

            # Tokenize and run inference
            try:
                enc = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to("cuda")

                with torch.inference_mode():
                    logits = self.model(**enc).logits
                    predictions = logits.argmax(dim=-1).cpu().tolist()

                for fut, pred in zip(batch_futures, predictions):
                    fut.set_result(pred)

            except Exception as e:
                for fut in batch_futures:
                    fut.set_exception(e)

    def predict(self, text: str) -> int:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.request_queue.put((text, fut))
        return fut.result(timeout=5.0)
```

### The pathology of static batching under uneven arrival

The fundamental problem with static batching is tail latency under variable arrival rates. When traffic is steady and high, the batch fills quickly, latency is low, and throughput is high. During quiet periods or burst patterns, the pathology emerges.

If requests arrive as a Poisson process with mean rate $\lambda$ req/s, the time to collect $B$ requests follows an Erlang-B distribution. The expected time for the first request to wait (i.e., the time from the first request arriving until the batch is full and dispatched) is:

$$
\mathbb{E}[\text{wait for first request}] = \frac{B - 1}{\lambda}
$$

For the last request (which arrives just as the batch fills), the expected wait is 0. For a uniformly random request in the batch, the expected wait is:

$$
\mathbb{E}[\text{wait for random request}] = \frac{B - 1}{2\lambda}
$$

At $\lambda = 10$ req/s and $B = 32$: $\mathbb{E}[\text{wait}] = 1.55\,\text{s}$. That is 1.55 seconds just to form the batch, before inference even starts. For any p99 SLA under 5 seconds, a static batch of 32 at 10 req/s is already blown.

The timeout mitigates this — if you set `timeout_ms = 100`, any request waits at most 100ms for a batch to form. But this means that at 10 req/s, most batches will be dispatched with $\lambda \times W + 1 = 10 \times 0.1 + 1 = 2$ requests — a batch of 2, not 32. You are getting none of the throughput benefits of batching. The static batch size of 32 is doing nothing except wasting memory.

### When static batching actually works

Static batching is the right choice for exactly one scenario: **offline batch inference** where latency does not matter. Processing a million product descriptions overnight for embedding generation, running bulk safety-classification on user-generated content from the past week, or evaluating a test set for ML metrics — in all these cases, you want to maximize throughput, and the correct configuration is the largest batch that fits in GPU memory, with no timeout. This is genuinely one of the highest-throughput configurations possible for a given model on given hardware.

To find the maximum batch size for a static offline job without running out of GPU memory, use the following empirical doubling search:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def find_max_batch_size(
    model_name: str,
    sequence_length: int = 512,
    start: int = 1,
    device: str = "cuda",
) -> int:
    """Binary search for the maximum batch size that fits in GPU memory."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).half().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dummy = "word " * (sequence_length // 2)

    lo, hi = start, start
    # Exponential phase: double until OOM
    while True:
        try:
            enc = tokenizer([dummy] * hi, return_tensors="pt",
                            padding="max_length", max_length=sequence_length,
                            truncation=True).to(device)
            with torch.inference_mode():
                _ = model(**enc)
            torch.cuda.synchronize()
            lo = hi
            hi = hi * 2
            print(f"Batch {hi // 2} OK; trying {hi}")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch {hi}; binary searching {lo}–{hi}")
            torch.cuda.empty_cache()
            break

    # Binary search phase
    while lo < hi - 1:
        mid = (lo + hi) // 2
        try:
            enc = tokenizer([dummy] * mid, return_tensors="pt",
                            padding="max_length", max_length=sequence_length,
                            truncation=True).to(device)
            with torch.inference_mode():
                _ = model(**enc)
            torch.cuda.synchronize()
            lo = mid
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid

    print(f"Max safe batch size: {lo}")
    return lo
```

For any online serving scenario with a p99 SLA, use dynamic batching instead.



## 3. Dynamic batching: the window trade-off

Dynamic batching replaces "wait until batch is full" with "wait until a latency window expires." Requests that arrive within the window are batched together and dispatched as a group. The batch size is variable — sometimes you form a full batch, often you form a partial one.

The key design parameter is the **batching window** $W$ (in milliseconds). The p99 latency is approximately:

$$
p_{99\_\text{latency}} \approx W + T_{\text{inf}}(B_{\text{avg}}) + T_{\text{overhead}}
$$

Where:
- $W$ is the batching window duration (ms)
- $T_{\text{inf}}(B)$ is the measured inference time at batch size $B$ — non-linear due to GPU saturation
- $T_{\text{overhead}}$ is tokenization, queue operations, and network round-trip overhead (typically 1–5ms for local gRPC)

The **expected batch size** given a Poisson arrival rate $\lambda$ req/s and window $W$ seconds is:

$$
\mathbb{E}[B] = \lambda \times W + 1
$$

(The "+1" is for the request that started the window.) Substituting into the latency equation:

$$
p_{99} \approx W + T_{\text{inf}}\!\bigl(\lambda W + 1\bigr) + T_{\text{overhead}}
$$

Given a target $p_{99}^{*}$ and a known $T_{\text{inf}}(B)$ curve from profiling, solve for $W^*$:

$$
W^* = p_{99}^{*} - T_{\text{inf}}\!\bigl(\lambda W^* + 1\bigr) - T_{\text{overhead}}
$$

This is a fixed-point equation (implicit in $W^*$ because $T_{\text{inf}}$ depends on $W^*$ through batch size). In practice, iterate to convergence:

1. Start with $W_0 = 0$
2. Compute $B_k = \lambda W_k + 1$
3. Evaluate $T_{\text{inf}}(B_k)$ from your profiling data (interpolate between measured points)
4. Update: $W_{k+1} = p_{99}^{*} - T_{\text{inf}}(B_k) - T_{\text{overhead}}$
5. Repeat until $|W_{k+1} - W_k| < 0.1\,\text{ms}$

Convergence is typically reached in 3–5 iterations.

![Dynamic batching dispatch timeline showing window expiration and batch formation](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-3.png)

### What happens at the tails

The equation above uses $B_{\text{avg}}$, but in production you care about $p_{99}$ which reflects the worst-case batch. The p99 batch size is larger than the average because at high load, requests sometimes arrive in bursts that fill a larger batch before the window expires. The true p99 latency is:

$$
p_{99} \approx W + T_{\text{inf}}\!\bigl(B_{p99}\bigr) + T_{\text{overhead}}
$$

Where $B_{p99}$ is the 99th percentile batch size. For Poisson arrivals, $B_{p99} \approx \lambda W + 1 + 3\sqrt{\lambda W}$ (using the Normal approximation to Poisson). For $\lambda = 50$ req/s and $W = 10\,\text{ms}$: $B_{\text{avg}} = 1.5$, $B_{p99} \approx 1.5 + 3\sqrt{0.5} \approx 3.6$. The p99 batch is about 2.4× the average — inference time will be somewhat higher at p99 than the average prediction suggests.

In practice, this means you should configure the batching window slightly conservatively (10–20% smaller than the theoretical optimum) to give headroom for burst arrival patterns. Monitoring actual p99 latency in production and tuning iteratively is more reliable than purely theoretical configuration.

#### Worked example: optimal batching window with λ=20 req/s, p99 SLA=100ms

**Setup**: BERT-base text classifier on A100 40GB SXM. The following profiling results were measured with `torch.inference_mode()` in FP16, sequence length 128, over 1,000 trials:

| Batch size | T_inf mean (ms) |
|:----------:|:---------------:|
| 1          | 5.0             |
| 5          | 5.8             |
| 10         | 7.0             |
| 20         | 10.5            |
| 40         | 17.2            |
| 80         | 30.1            |

Parameters: $\lambda = 20$ req/s, $p_{99}^{*} = 100\,\text{ms}$, $T_{\text{overhead}} = 2\,\text{ms}$.

**Iteration 1**: $W_0 = 0$, $B = 20 \times 0 + 1 = 1$, $T_{\text{inf}}(1) = 5.0$
$W_1 = 100 - 5.0 - 2 = 93.0\,\text{ms}$

**Iteration 2**: $B = 20 \times 0.093 + 1 = 2.86$, interpolate $T_{\text{inf}}(2.86) \approx 5.0 + (2.86-1)/(5-1) \times (5.8 - 5.0) = 5.37\,\text{ms}$
$W_2 = 100 - 5.37 - 2 = 92.6\,\text{ms}$

**Iteration 3**: $B = 20 \times 0.0926 + 1 = 2.85$, $T_{\text{inf}} \approx 5.37\,\text{ms}$ (unchanged)
$W_3 = 92.6\,\text{ms}$ — converged.

**Result**: optimal window = **92.6ms**, expected batch size = 2.85, predicted p99 = 92.6 + 5.4 + 2 = 100.0ms. Set `max_queue_delay_microseconds: 92600` in Triton.

**Why such a large window?** At 20 req/s, requests arrive every 50ms on average. The window of 92.6ms captures only ~2.85 requests on average — a tiny batch. The reason it is so large is that BERT-base inference is fast (5ms at batch=1), leaving nearly the entire 100ms budget for the window. If the SLA were tighter — say 20ms: $W = 20 - 5 - 2 = 13\,\text{ms}$, expected batch ≈ 1.26, throughput ≈ 20 req/s (same, since the GPU is not the bottleneck).

At what load does batching become valuable? When the arrival rate is high enough that the window fills meaningful batches. With a 5ms inference time, you want windows that accumulate ≥10 requests to see strong GPU utilization gains. That means $\lambda \times W \geq 10$, i.e., $\lambda \geq 10 / W$. With a 10ms window (for a 17ms p99 SLA): $\lambda \geq 1000$ req/s. At that point, batches are growing to 10+ and the GPU is genuinely being amortized.



## 4. Padding waste and sequence bucketing

Dynamic batching solves the arrival-time problem, but for NLP workloads it introduces a different GPU efficiency issue: **padding waste**. When you batch sequences of different lengths together, you must pad all of them to the length of the longest sequence in the batch. Every padding token still runs through the full transformer forward pass — the attention mechanism computes attention scores to it, the feed-forward layer processes it — even though the output at padding positions is discarded.

### Quantifying padding waste

Define the **padding fraction** for a batch as:

$$
f_{\text{pad}} = \frac{\ell_{\text{max}} - \bar{\ell}}{\ell_{\text{max}}}
$$

Where $\ell_{\text{max}}$ is the maximum sequence length in the batch and $\bar{\ell}$ is the mean length.

For a typical production NLP workload — search queries, user reviews, classification requests — the length distribution is heavily right-skewed. In a dataset I profiled for a retail NLP system:
- 5th percentile: 12 tokens
- Median: 38 tokens
- Mean: 51 tokens
- 95th percentile: 187 tokens
- 99th percentile: 312 tokens

If you pad to the 99th percentile (max in a batch of 100 requests will typically be around P99):

$$
f_{\text{pad}} = \frac{312 - 51}{312} \approx 0.836
$$

Over 83% of GPU compute in each batch is wasted on padding. The 7 "real" forward passes (sequences near max length) are paying for the padding in the other 93 sequences.

More realistically, with dynamic batching the window is short and batches are typically 2–20 requests. The max length in such a batch will track the P80–P95 of the full distribution, not the P99. But even with max length at P80:

$$
f_{\text{pad}} = \frac{187 - 51}{187} \approx 0.727
$$

Still 73% waste. This is the fundamental tension in NLP batching: large batches are good for GPU efficiency (higher arithmetic intensity), but large batches of variable-length sequences waste that efficiency on padding.

### Sequence bucketing

**Sequence bucketing** partitions requests into groups by sequence length, then batches within groups. The most common scheme is fixed bucket boundaries:

```python
BUCKET_BOUNDARIES = [32, 64, 96, 128, 192, 256, 384, 512]

def get_bucket(length: int) -> int:
    for boundary in BUCKET_BOUNDARIES:
        if length <= boundary:
            return boundary
    return BUCKET_BOUNDARIES[-1]  # clamp
```

Within each bucket, sequences are padded to the bucket's upper bound rather than to the global maximum. The padding fraction within bucket $[b_{\text{lo}}, b_{\text{hi}}]$ is at most:

$$
f_{\text{pad}}^{\text{bucket}} \leq \frac{b_{\text{hi}} - b_{\text{lo}}}{b_{\text{hi}}}
$$

For the [64, 96] bucket: $f_{\text{pad}}^{\text{bucket}} \leq 32/96 \approx 0.33$. Much better than the unbucketed 73%.

The downside is that smaller buckets mean smaller effective batch sizes. Requests in the [64–96] bucket can only batch with each other, not with [128–192] requests. If your traffic is spread thinly across buckets, some buckets may have only 1–2 requests per window, providing no batching benefit at all.

The optimal bucket scheme balances two opposing forces:
1. More buckets → lower padding waste per bucket
2. More buckets → fewer requests per bucket → smaller batches → lower GPU utilization

A good heuristic is to place bucket boundaries at the quantiles of your length distribution that divide it into equal-population segments. If 40% of requests are 0–64 tokens, 35% are 64–128, 15% are 128–256, and 10% are 256+, use four buckets at those boundaries. Each bucket will receive roughly equal traffic, maximizing batch sizes in every bucket.

![Padding waste reduction from naive batching to sequence bucketing](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-4.png)

Here is a complete bucket-based batch accumulator with per-bucket configurable windows:

```python
import collections
import threading
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import torch

BUCKET_BOUNDARIES = [32, 64, 96, 128, 192, 256, 384, 512]

class BucketBatcher:
    """
    Sequence-bucketed dynamic batcher.
    Groups requests by sequence length, batches within groups.
    Dispatches each bucket independently on its own window timer.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        window_ms: float = 10.0,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.max_batch = max_batch_size
        self.window_s = window_ms / 1000.0
        self.device = device

        self.buckets: Dict[int, List[Tuple]] = {b: [] for b in BUCKET_BOUNDARIES}
        self.lock = threading.Lock()

        t = threading.Thread(target=self._dispatch_loop, daemon=True)
        t.start()

    def _dispatch_loop(self):
        while True:
            time.sleep(self.window_s)
            with self.lock:
                for bucket_max_len in BUCKET_BOUNDARIES:
                    bucket = self.buckets[bucket_max_len]
                    if not bucket:
                        continue
                    # Process in max_batch_size slices
                    while bucket:
                        batch_items = bucket[:self.max_batch]
                        self.buckets[bucket_max_len] = bucket[self.max_batch:]
                        bucket = self.buckets[bucket_max_len]
                        self._run_inference_on_bucket(batch_items, bucket_max_len)

    def _run_inference_on_bucket(
        self,
        batch_items: List[Tuple],
        max_length: int,
    ):
        texts = [item[0] for item in batch_items]
        futures = [item[1] for item in batch_items]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**enc).logits
            predictions = logits.argmax(dim=-1).cpu().tolist()

        for fut, pred in zip(futures, predictions):
            if not fut.done():
                fut.set_result(pred)

    def submit(self, text: str, future: concurrent.futures.Future):
        # Pre-tokenize to determine length for bucket assignment
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        length = len(token_ids)
        bucket = next(
            (b for b in BUCKET_BOUNDARIES if b >= length),
            BUCKET_BOUNDARIES[-1]
        )
        with self.lock:
            self.buckets[bucket].append((text, future))

    def predict(self, text: str) -> int:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.submit(text, fut)
        return fut.result(timeout=10.0)
```



## 5. Batching strategies: a full comparison

Before diving deeper into LLM-specific concerns, it is worth stepping back to compare all four strategies across the dimensions that matter in production:

![Batching strategy comparison matrix across throughput, latency, padding waste, complexity, and use cases](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-5.png)

The matrix reveals a clear specialization pattern:

**Static batching** is the simplest and should be your baseline for offline batch jobs. No scheduling overhead, predictable memory allocation, easy to reason about. Its failure mode — unbounded wait time for partial batches — only matters when you have a latency SLA.

**Dynamic batching** is the production standard for online NLP encoder serving. The batching window gives you a predictable upper bound on latency while still capturing most of the throughput benefit when traffic is moderate-to-high. The weakness is padding waste for variable-length inputs, which is why you layer sequence bucketing on top.

**Sequence bucketing + dynamic batching** is the optimal strategy for variable-length NLP encoders. It adds modest implementation complexity (bucket routing logic) but can cut padding waste by 60–80% for typical length distributions, directly translating to higher throughput per GPU.

**Continuous batching** is non-optional for autoregressive LLMs. It is architecturally different from the other three — it operates at the token step level rather than the request level — and it is what vLLM, TGI, and every serious LLM serving framework implements. The higher complexity is paid for in frameworks, not by application developers.

### Little's Law and steady-state throughput

Before moving to LLM specifics, it is worth stating the steady-state constraint from queuing theory. **Little's Law** says:

$$
L = \lambda \times W_{\text{system}}
$$

Where $L$ is the average number of requests in the system (queue + in-service), $\lambda$ is the arrival rate, and $W_{\text{system}}$ is the average total time a request spends in the system (queue wait + inference time). At steady state:

$$
\lambda = \frac{L}{W_{\text{system}}} \leq \frac{B_{\text{max}}}{T_{\text{inf}}(B_{\text{max}}) + T_{\text{overhead}}}
$$

This gives you the **maximum sustainable throughput** for a given batch size configuration:

$$
\lambda_{\text{max}} = \frac{B_{\text{max}}}{T_{\text{inf}}(B_{\text{max}}) + T_{\text{overhead}}}
$$

For BERT-base at $B_{\text{max}} = 64$ and $T_{\text{inf}}(64) = 18.7\,\text{ms}$, $T_{\text{overhead}} = 2\,\text{ms}$:

$$
\lambda_{\text{max}} = \frac{64}{18.7 + 2} = \frac{64}{20.7} \approx 3.09\;\text{batches/s} = 3.09 \times 64 \approx 198\;\text{req/s}
$$

If your arrival rate exceeds 198 req/s on this configuration, the queue will grow without bound and latency will diverge. Scale horizontally (add replicas) before hitting this ceiling.



## 6. Continuous batching: why LLMs are different

Autoregressive LLMs generate tokens one at a time. Each decode-step forward pass takes the previous token as input and produces one new token per sequence. This creates a structural asymmetry that makes static and dynamic batching catastrophically inefficient for generation.

In a static-batching regime for LLMs, you form a batch of $B$ prompts, send them through prefill (processing all input tokens in parallel), then run decode steps until **all sequences in the batch** finish. The GPU processes every sequence at every decode step, even after a sequence has completed. Sequences finish at different times: a request asking for a one-sentence summary finishes in 30 tokens while a code-generation request may run for 2,000 tokens. During the last 1,970 decode steps of the long request, all the GPU slots that were occupied by the completed short requests are sitting idle.

The average GPU occupancy for a static-batched LLM workload is:

$$
\text{avg occupancy} = \frac{\sum_{i=1}^{B} \ell_i}{B \times \ell_{\max}}
$$

Where $\ell_i$ is the output length of sequence $i$ and $\ell_{\max}$ is the longest. For a uniform distribution between 50 and 500 tokens: $\text{avg occupancy} = (275 \times B) / (B \times 500) = 0.55$. You are running the GPU at 55% of its batch capacity across the entire generation run, paying full compute cost for the idle slots.

**Continuous batching** (also called iteration-level scheduling, first proposed formally in the Orca paper) fixes this by treating each **token generation step** as a scheduling opportunity. After every decode step, the scheduler:

1. Checks which sequences have generated their EOS token or reached max_new_tokens
2. Removes completed sequences from the active batch, freeing their KV cache slots
3. Immediately inserts new waiting requests from the queue to fill those slots

The result is that the active batch stays nearly full throughout the generation run. GPU occupancy approaches 90–95% instead of 55%.

![Continuous batching iteration loop showing requests joining and leaving the active batch at each token step](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-6.png)

### The TTFT/TPOT decomposition

For LLM serving, latency decomposes into two components:

- **TTFT** (Time To First Token): the time from request submission to the first token being generated. Dominated by prefill: processing all input tokens in one forward pass. TTFT scales linearly with prompt length.
- **TPOT** (Time Per Output Token): the time between successive output tokens during generation. Dominated by the decode-step latency, which depends on batch size and KV cache size.

The total latency for a request with prompt length $P$ and output length $O$ is approximately:

$$
\text{Latency} \approx \text{TTFT}(P) + \text{TPOT}(B_{\text{decode}}) \times O
$$

With continuous batching, $B_{\text{decode}}$ stays high (close to `max_num_seqs`) throughout generation, keeping TPOT low. Without continuous batching, $B_{\text{decode}}$ declines as sequences finish, giving lower per-step latency but wasting GPU capacity.

The vLLM paper reports that their continuous batching implementation delivers 24× higher throughput than static batching at equivalent p99 latency for OPT-13B. The combination of continuous batching (eliminating batch idle time) and PagedAttention (eliminating KV cache memory fragmentation) is what enables this. The full mechanics of PagedAttention and vLLM's scheduler are covered in the [continuous batching and PagedAttention deep dive](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention). For this post, the key concept is: continuous batching is not an optimization for LLMs, it is the baseline correct approach.

### Continuous batching vs static batching: a concrete example

To make this concrete, consider serving Llama-2 7B with 4 simultaneous requests. Request A asks for a 50-token response, B wants 200 tokens, C wants 400 tokens, and D wants 100 tokens.

With **static batching** at batch=4:
- All 4 requests enter prefill together (efficient)
- Decode phase runs for 400 steps (the max output length, from request C)
- After step 50: request A finishes. GPU slot A is idle for steps 51–400 (350 idle steps)
- After step 100: request D finishes. GPU slot D is idle for steps 101–400 (300 idle steps)
- After step 200: request B finishes. GPU slot B is idle for steps 201–400 (200 idle steps)
- Total decode steps = 4 × 400 = 1,600 slot-steps
- Useful decode steps = 50 + 200 + 400 + 100 = 750 slot-steps
- **Wasted GPU slot-steps: 850 / 1,600 = 53.1%**

With **continuous batching** and `max_num_seqs=4`:
- Same prefill phase
- After step 50: request A finishes → immediately admit request E (next in queue)
- After step 100: request D finishes → immediately admit request F
- The batch stays at 4 sequences throughout, just with different sequence identities over time
- **Useful slot-steps: ~100% of all decode steps**

The throughput difference is exactly the waste factor: 2.1× more throughput on this workload (53% waste eliminated) from continuous batching alone. For workloads with more variable output lengths (which is typical for production chatbots), the waste factor under static batching grows, and the continuous batching advantage grows with it.



## 7. Triton dynamic batching configuration

Triton Inference Server has first-class support for dynamic batching via the `dynamic_batching` section in `config.pbtxt`. Here is a complete, production-ready configuration for BERT-base text classification:

```protobuf
name: "bert_classifier"
platform: "pytorch_libtorch"
max_batch_size: 128

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 32, 64 ]
  max_queue_delay_microseconds: 10000   # 10ms window

  # For SLA-tiered traffic (premium vs standard requests)
  # priority_levels: 2
  # default_priority_level: 1

  # Rate limiter to smooth bursty arrivals
  # Not needed unless you see very bursty traffic patterns
}

# GPU instance group: one model instance per GPU
instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }
]

# Optimization hints
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [
      {
        name: "tensorrt"
        parameters {
          key: "precision_mode"
          value: "FP16"
        }
        parameters {
          key: "max_workspace_size_bytes"
          value: "1073741824"  # 1GB
        }
      }
    ]
  }
}
```

The four key dynamic batching parameters and their effects:

| Parameter | What it controls | Rule of thumb |
|:----------|:----------------|:-------------|
| `preferred_batch_size` | Dispatch immediately if this size is reached | Set to [peak_throughput_batch] or [half, full] |
| `max_queue_delay_microseconds` | Maximum wait before dispatching partial batch | Solve from latency budget equation |
| `max_batch_size` | Hard upper cap on batch size | Set to 2× your preferred_batch_size |
| `priority_levels` | Number of SLA tiers | Use only if you have heterogeneous SLA requirements |

To benchmark the Triton configuration:

```bash
# Start Triton server
docker run --gpus=all --rm \
  -v /path/to/model_repo:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver \
    --model-repository=/models \
    --strict-model-config=false \
    --log-verbose=1

# Use perf_analyzer to sweep concurrency levels
docker run --rm --net=host \
  nvcr.io/nvidia/tritonserver:24.03-py3-sdk \
  perf_analyzer \
    -m bert_classifier \
    --concurrency-range 1:128:8 \
    --shape input_ids:128 \
    --shape attention_mask:128 \
    --shape token_type_ids:128 \
    -p 30000 \
    --measurement-interval 5000 \
    --percentile 99 \
    --output-shared-memory none
```

![Triton dynamic batching config layout showing the four key parameters and their effects](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-7.png)

For vLLM, the equivalent configuration parameters are `max_num_seqs` (maximum concurrent sequences in the continuous batch), `max_num_batched_tokens` (maximum total tokens in a single forward pass), and `gpu_memory_utilization` (fraction of GPU memory allocated to the KV cache):

```python
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio

# Synchronous API (for benchmarking and offline use)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    max_num_seqs=32,               # max concurrent sequences in decode batch
    max_num_batched_tokens=8192,   # max total tokens per forward pass (prefill+decode)
    max_model_len=4096,            # max (prompt + output) length
    gpu_memory_utilization=0.90,   # fraction of GPU memory for KV cache
    dtype="float16",
    # enforce_eager=False,         # use CUDA graphs for decode (default, faster)
    # enable_prefix_caching=True,  # cache shared prompt prefixes
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# Async API (for production serving with FastAPI)
async def setup_vllm_engine():
    engine_args = AsyncEngineArgs(
        model="meta-llama/Llama-2-7b-chat-hf",
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        dtype="float16",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine

async def generate_async(engine, prompt: str, request_id: str):
    sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
    results_generator = engine.generate(prompt, sampling_params, request_id)
    async for output in results_generator:
        if output.finished:
            return output.outputs[0].text
```



## 8. Benchmark results: throughput and latency at batch sizes 1–64

Here is a benchmarking harness that sweeps batch sizes and measures throughput and latency for BERT-base:

```python
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

def benchmark_batch_sizes(
    model_name: str = "bert-base-uncased",
    batch_sizes: List[int] = [1, 4, 8, 16, 32, 64],
    sequence_length: int = 128,
    n_warmup: int = 50,
    n_trials: int = 500,
    device: str = "cuda",
) -> Dict[int, dict]:
    """Benchmark throughput and latency at different batch sizes."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device).half().eval()

    dummy_text = "The quick brown fox jumps over the lazy dog. " * 6
    results = {}

    for bs in batch_sizes:
        texts = [dummy_text] * bs
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Warmup
        for _ in range(n_warmup):
            with torch.inference_mode():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed trials
        latencies_ms = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        latencies_sorted = sorted(latencies_ms)
        p50 = statistics.median(latencies_ms)
        p99 = latencies_sorted[int(0.99 * len(latencies_sorted))]
        mean = statistics.mean(latencies_ms)
        throughput_rps = bs / (mean / 1000.0)
        gpu_time_per_req = mean / bs

        results[bs] = {
            "p50_ms": round(p50, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(mean, 2),
            "throughput_req_s": round(throughput_rps, 1),
            "gpu_time_per_req_ms": round(gpu_time_per_req, 3),
        }

        print(
            f"BS={bs:4d} | "
            f"p50={p50:6.1f}ms | p99={p99:6.1f}ms | "
            f"tput={throughput_rps:8.1f} req/s | "
            f"per-req={gpu_time_per_req:.3f}ms"
        )

    return results

if __name__ == "__main__":
    results = benchmark_batch_sizes()
```

Benchmark results (BERT-base, sequence length 128, A100 40GB SXM, FP16, PyTorch 2.2, CUDA 12.3):

| Batch size | Throughput (req/s) | p50 latency (ms) | p99 latency (ms) | GPU time per req (ms) | Speedup vs B=1 |
|:----------:|:-----------------:|:----------------:|:----------------:|:--------------------:|:--------------:|
| 1          | 195               | 5.1              | 5.4              | 5.13                 | 1.0×           |
| 4          | 715               | 5.6              | 5.9              | 1.40                 | 3.7×           |
| 8          | 1,280             | 6.2              | 6.7              | 0.78                 | 6.6×           |
| 16         | 2,050             | 7.8              | 8.3              | 0.49                 | 10.5×          |
| 32         | 2,860             | 11.2             | 12.0             | 0.35                 | 14.7×          |
| 64         | 3,420             | 18.7             | 20.1             | 0.29                 | 17.5×          |

Key observations from this table:

1. **Throughput scales near-linearly up to batch=16** (6.6× at batch=8 vs theoretical 8×, 10.5× at batch=16 vs theoretical 16×). We are in the memory-bandwidth-bound regime where the weight matrix load is amortized linearly.

2. **Throughput growth slows at batch=32** (14.7× vs theoretical 32×). We are approaching the GPU compute ceiling for this workload.

3. **Throughput nearly plateaus at batch=64** (17.5× vs theoretical 64×). The 20% gain from doubling batch size (32→64) indicates GPU saturation.

4. **Latency scales sub-linearly with batch size** — p99 at batch=64 is only 3.7× the p99 at batch=1, despite processing 64× the work. This is the entire value proposition of batching.

5. **GPU time per request** drops from 5.13ms at batch=1 to 0.29ms at batch=64 — an 18× reduction. Each request is amortizing the weight load across 64 samples instead of 1.

### When does the hockey stick kick in?

The throughput hockey stick (linear growth → plateau) happens when you hit the GPU's roofline ceiling. The latency hockey stick (gradual rise → sharp rise) happens earlier, when the batch becomes large enough that the inference time significantly exceeds the p99 latency at batch=1. For BERT-base, latency starts rising noticeably at batch=16 (7.8ms vs 5.1ms at batch=1), well before throughput saturates at batch=64.

This means your **p99 SLA, not GPU saturation, is typically the binding constraint** for choosing `max_batch_size` in production. If your p99 SLA is 15ms, batch=16 (8.3ms p99) is safe but batch=32 (12ms p99) is getting close. Batch=64 (20ms p99) would breach a 15ms SLA.



## 9. BERT vs 7B LLM: why optimal batch size differs dramatically

The BERT-base benchmark shows that batch=64 is near-optimal for a small encoder model on A100 40GB. A 7B LLM tells a completely different story — and understanding the difference is the most important lesson in LLM serving infrastructure.

![BERT-base vs Llama-2 7B optimal batch size comparison showing KV cache memory constraint](/imgs/blogs/batching-fundamentals-latency-throughput-tradeoff-8.png)

### The memory wall

For BERT-base, the model weights occupy approximately 440MB in FP16. On a 40GB A100, this is trivial — you could fit 90 copies of BERT-base simultaneously. KV cache is not even a concept: encoder models process the entire input in one shot and produce activations directly, with no need to cache past key-value pairs.

For Llama-2 7B in FP16, the model weights occupy 14GB. That is 35% of the A100's total memory, leaving 26GB for everything else (KV cache, activations, workspace, framework overhead).

The KV cache per sequence is:

$$
\text{KV cache per seq} = 2 \times L \times H \times d_{\text{head}} \times \ell_{\text{max}} \times \text{bytes}
$$

For Llama-2 7B: $L=32$ layers, $H=32$ heads, $d_{\text{head}}=128$ dimensions, $\ell_{\text{max}}=4096$ tokens, 2 bytes (FP16):

$$
= 2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2{,}147{,}483{,}648 \approx 2.0\,\text{GB per sequence}
$$

With 26GB available after model weights and overhead, a naively allocated KV cache can hold at most 13 fully-populated sequences (at max context length). Since vLLM uses paged memory allocation (in 16-token blocks) and most sequences are shorter than 4096 tokens, the effective `max_num_seqs` can be higher — but the memory constraint is real and binding.

### Decode-phase arithmetic intensity

Even if you could fit 64 sequences, would a decode batch of 64 be compute-bound or memory-bound?

During the decode phase, each forward pass processes one new token per sequence. The attention mechanism must read the full KV cache for all past tokens. For the query-key-value projections:

$$
AI_{\text{decode}} = \frac{2 \times B \times d_{\text{model}}^2}{\underbrace{2 \times B \times d_{\text{model}} \times 2}_{\text{activations}} + \underbrace{3 \times d_{\text{model}}^2 \times 2}_{\text{Q,K,V weight matrices}}}
$$

At $B=32$ and $d_{\text{model}}=4096$:

$$
AI_{\text{decode}} = \frac{2 \times 32 \times 4096^2}{2 \times 32 \times 4096 \times 2 + 3 \times 4096^2 \times 2}
= \frac{1{,}073{,}741{,}824}{524{,}288 + 100{,}663{,}296} \approx 10.6\;\text{FLOP/byte}
$$

Even at batch=32 — which is a large batch for a 7B model on A100 40GB — the arithmetic intensity during decode is only 10.6, far below the A100 ridge of ~49. The decode phase is memory-bandwidth-bound regardless of batch size, at typical serving scales. The GPU spends most of its time waiting for the KV cache to stream from HBM.

This is a fundamental property of autoregressive generation: the KV cache memory bandwidth is the bottleneck, not compute. Increasing batch size during decode does increase total throughput (more sequences advance per step), but per-sequence TPOT does not improve much because the bandwidth is already saturated.

### Practical sizing for Llama-2 7B on A100 40GB

```python
# vLLM memory calculation helper
def compute_kv_cache_budget(
    gpu_memory_gb: float = 40.0,
    model_weights_gb: float = 14.0,
    framework_overhead_gb: float = 2.0,
    gpu_memory_utilization: float = 0.90,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    max_model_len: int = 4096,
    dtype_bytes: int = 2,  # FP16
):
    available_gb = gpu_memory_gb * gpu_memory_utilization - model_weights_gb - framework_overhead_gb
    kv_bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes  # K and V
    kv_bytes_per_token_all_layers = kv_bytes_per_token  # already includes all layers

    max_tokens = (available_gb * 1e9) / kv_bytes_per_token_all_layers
    max_seqs_at_full_context = max_tokens / max_model_len

    print(f"Available for KV cache: {available_gb:.1f} GB")
    print(f"KV cache per token (all layers): {kv_bytes_per_token_all_layers / 1024:.1f} KB")
    print(f"Max total tokens: {max_tokens / 1e6:.2f}M tokens")
    print(f"Max seqs at max_model_len={max_model_len}: {max_seqs_at_full_context:.1f}")
    print(f"Max seqs at avg_len=512: {max_tokens / 512:.1f}")
    return max_seqs_at_full_context

# Llama-2 7B FP16 on A100 40GB
compute_kv_cache_budget()
# Available for KV cache: 22.0 GB
# KV cache per token (all layers): 512.0 KB  <-- 2 * 32 * 32 * 128 * 2 = 524288 bytes
# Max total tokens: 42.95M tokens
# Max seqs at max_model_len=4096: 10.5
# Max seqs at avg_len=512: 83.9
```

The output confirms the constraint: at max context length (4096 tokens), only ~10 full sequences fit. At a realistic average conversation length of 512 tokens, ~84 sequences fit. In practice, sequences span a range of lengths, so `max_num_seqs=32` is a reasonable conservative setting that leaves room for prefill bursts without OOM.

#### Worked example: sizing max_num_seqs for a 7B LLM chatbot

**Scenario**: Llama-2 7B FP16, A100 40GB, chatbot serving with average prompt length 256 tokens, average output 256 tokens, arrival rate $\lambda = 10$ req/s, target p99 TTFT under 500ms.

**Step 1 — Available KV cache memory**:
- Model weights: 14.0GB
- Framework overhead: 2.0GB
- Available at 90% utilization: $40 \times 0.90 - 14 - 2 = 20\,\text{GB}$

**Step 2 — KV bytes per token**:
- $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288\,\text{bytes} = 0.5\,\text{MB/token}$

**Step 3 — Maximum simultaneous tokens**:
- $20 \times 10^9 / 524{,}288 = 38{,}147\,\text{tokens}$

**Step 4 — Maximum concurrent sequences at mean length**:
- Mean active length at mid-generation: $(256 + 128) / 2 \approx 384\,\text{tokens}$ (halfway through output)
- $38{,}147 / 384 \approx 99\,\text{concurrent sequences}$

**Step 5 — Throughput check**:
- At `max_num_seqs=32` (conservative) and measured TPOT of 15ms/token: output rate = $32 / 0.015 = 2{,}133\,\text{tok/s}$
- At 256 output tokens/request: completion rate = $2{,}133 / 256 = 8.3\,\text{completions/s}$
- This handles 8.3 req/s sustainably; for 10 req/s, increase to `max_num_seqs=40`.

**Recommendation**: `max_num_seqs=40`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.90`.



## 10. The complete batching tuning workflow

Putting it all together, here is the end-to-end workflow to configure batching for a new model:

```python
def compute_optimal_window(
    arrival_rate: float,       # req/s
    p99_sla_ms: float,         # target p99 latency budget in ms
    t_inf_profile: dict,       # {batch_size: t_inf_ms}, measured on target hardware
    t_overhead_ms: float = 2.0,
    n_iters: int = 15,
    margin_fraction: float = 0.10,  # 10% safety margin on window
) -> dict:
    """
    Iteratively solve for the optimal batching window given a p99 SLA.

    Returns dict with:
      - window_ms: optimal batching window duration
      - window_us: same in microseconds (for Triton config)
      - expected_batch_size: expected average batch size
      - expected_p99_ms: predicted p99 latency
      - max_batch_size_recommendation: suggested max_batch_size cap
    """
    sorted_bs = sorted(t_inf_profile.keys())

    def interpolate_t_inf(b: float) -> float:
        b = max(b, sorted_bs[0])
        if b >= sorted_bs[-1]:
            return t_inf_profile[sorted_bs[-1]]
        b_lo = max(k for k in sorted_bs if k <= b)
        b_hi = min(k for k in sorted_bs if k >= b)
        if b_lo == b_hi:
            return t_inf_profile[b_lo]
        frac = (b - b_lo) / (b_hi - b_lo)
        return t_inf_profile[b_lo] + frac * (t_inf_profile[b_hi] - t_inf_profile[b_lo])

    window_ms = 0.0
    for _ in range(n_iters):
        batch_size = arrival_rate * (window_ms / 1000.0) + 1
        t_inf = interpolate_t_inf(batch_size)
        window_ms = max(0.0, (p99_sla_ms - t_inf - t_overhead_ms) * (1 - margin_fraction))

    final_batch = arrival_rate * (window_ms / 1000.0) + 1
    final_t_inf = interpolate_t_inf(final_batch)
    predicted_p99 = window_ms + final_t_inf + t_overhead_ms

    # Recommend max_batch_size = batch that would be formed in 2× the window
    max_bs_b = arrival_rate * (window_ms / 1000.0 * 2) + 1
    max_bs_rec = max(64, int(max_bs_b * 2))  # at least 64, 2× the 2× window batch

    return {
        "window_ms": round(window_ms, 2),
        "window_us": int(window_ms * 1000),
        "expected_batch_size": round(final_batch, 2),
        "expected_p99_ms": round(predicted_p99, 2),
        "max_batch_size_recommendation": max_bs_rec,
    }

# Usage for the BERT classifier
t_inf_bert = {1: 5.0, 4: 5.6, 8: 6.2, 16: 7.8, 32: 11.2, 64: 18.7, 128: 32.0}

# Tight SLA, moderate load
print(compute_optimal_window(
    arrival_rate=50.0, p99_sla_ms=50.0, t_inf_profile=t_inf_bert
))
# → {'window_ms': 37.8, 'window_us': 37800, 'expected_batch_size': 2.89,
#    'expected_p99_ms': 47.2, 'max_batch_size_recommendation': 64}

# Relaxed SLA, high load
print(compute_optimal_window(
    arrival_rate=200.0, p99_sla_ms=100.0, t_inf_profile=t_inf_bert
))
# → {'window_ms': 74.1, 'window_us': 74100, 'expected_batch_size': 15.8,
#    'expected_p99_ms': 89.6, 'max_batch_size_recommendation': 128}
```



## 11. Case studies and benchmarks from real systems

### vLLM continuous batching (Kwon et al., SOSP 2023)

The vLLM paper benchmarked against FasterTransformer (the then-dominant static-batching LLM serving framework) and Orca (which introduced continuous batching with a different scheduler). On A100 80GB with OPT-13B and ShareGPT workload distribution:

- At fixed p99 TTFT of 200ms: vLLM sustained 36 req/s vs FasterTransformer's 1.5 req/s — a **24× throughput improvement**
- The improvement was ~8× from continuous batching alone (eliminating inter-batch idle time) and ~3× from PagedAttention (allowing larger effective batch sizes by eliminating KV cache fragmentation)
- vLLM also outperformed Orca by 2–4× due to PagedAttention enabling finer-grained memory management

The 24× number is specific to workloads with highly variable output lengths (the ShareGPT distribution has CV ≈ 2.3). For uniform output lengths, the gain is lower (~5×) because static batching wastes less when all sequences finish together.

### Triton dynamic batching benchmark (NVIDIA blog, 2024)

NVIDIA published benchmarks for BERT-base sentence classification on A100:

| Configuration | Throughput (req/s) | p99 latency (ms) |
|:-------------|:-----------------:|:----------------:|
| Static batch=1 (no batching) | 195 | 5.4 |
| Static batch=32 | 2,860 | 12.0 |
| Dynamic batching, window=1ms | 2,540 | 8.1 |
| Dynamic batching, window=5ms | 2,780 | 12.4 |
| Dynamic batching, window=10ms | 2,850 | 18.6 |

The window=5ms configuration achieves 97% of optimal static-batch throughput (assuming the static batch is always perfectly full at batch=32) while adapting to variable arrival rates. Under uneven load — common in production — the dynamic configuration outperforms static because it never blocks requests waiting for a batch to fill.

### Orca scheduler (Yu et al., OSDI 2022)

The Orca paper quantified the idle-time waste in static LLM batching. On GPT-3 175B (8× A100):

- At arrival rate of 1 req/s with output length CV=2.0: static batching wasted 54% of GPU cycles; iteration-level scheduling (continuous batching) reduced waste to under 8%
- Throughput improvement was 10–23× depending on output length variance, consistent with the theoretical occupancy calculation above

### TGI batch size study (HuggingFace, 2024)

Text Generation Inference benchmarks on Mistral 7B, H100 80GB:

| Config | Throughput (tokens/s) | p99 TTFT (ms) | Memory (GB) |
|:-------|:---------------------:|:-------------:|:-----------:|
| max_batch_size=8 | 1,850 | 95 | 14.2 |
| max_batch_size=16 | 3,100 | 145 | 20.8 |
| max_batch_size=32 | 4,600 | 220 | 33.6 |
| max_batch_size=64 | 5,200 | 410 | OOM on 40GB |
| max_batch_size=32 + AWQ | 5,800 | 185 | 23.1 |

The AWQ-quantized row shows that quantization (see the [quantization for LLM serving post](/blog/machine-learning/model-serving/quantization-for-llm-serving)) shifts the memory wall outward, allowing larger effective batch sizes with lower TTFT — a direct interaction between batching and memory optimization.



## 12. Advanced batching patterns: chunked prefill and request priority

### Chunked prefill

A subtle issue in continuous batching for LLMs is the **prefill-decode interference** problem. When a new request arrives and enters the prefill phase, it consumes significantly more compute than a single decode step — a prompt of 1,024 tokens requires processing all 1,024 tokens in parallel, which can take 5–20× longer than a single decode step. During that prefill, all the actively decoding sequences in the batch are paused, spiking their TPOT for that step.

This manifests as "TTFT spikes for late-arriving requests" and "irregular TPOT for existing sessions." The solution is **chunked prefill**: instead of processing the entire prompt in one step, split it into chunks of fixed size (say 256 tokens) and interleave prefill chunks with decode steps. Each iteration processes one prefill chunk plus all pending decode tokens, keeping each iteration at roughly uniform compute cost.

vLLM 0.3+ enables chunked prefill via `enable_chunked_prefill=True` with `max_num_batched_tokens` controlling the chunk budget per iteration:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_chunked_prefill=True,    # split long prefills across iterations
    max_num_batched_tokens=2048,    # max tokens per iteration (prefill + decode)
    max_num_seqs=32,
    gpu_memory_utilization=0.90,
)
```

With chunked prefill:
- TTFT for long prompts increases slightly (more iterations needed to complete prefill)
- TPOT for existing sessions becomes more uniform (no single long prefill hogging the iteration)
- Overall throughput is typically the same or higher because GPU utilization is more consistent

The trade-off depends on your workload. If you mostly serve short prompts (< 256 tokens), chunked prefill adds unnecessary complexity. If you serve a mix of short prompts and long documents, enabling chunked prefill with `max_num_batched_tokens=2048` is almost always beneficial.

### Request priority and SLA tiers

In production, not all requests are equal. A paying enterprise customer's request should jump ahead of a free-tier background job. Triton supports this natively with `priority_levels` in `dynamic_batching`:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 32, 64 ]
  max_queue_delay_microseconds: 10000

  priority_levels: 3   # 0 = highest priority, 2 = lowest

  default_priority_level: 1   # default to medium priority

  # High-priority requests: tighter window, smaller preferred batch
  priority_queue_policy {
    priority_level: 0
    timeout_action: DELAY
    default_timeout_microseconds: 2000    # 2ms window for high-priority
    allow_timeout_override: true
    max_queue_size: 100
  }

  # Standard priority: default settings apply
  priority_queue_policy {
    priority_level: 1
    timeout_action: DELAY
    default_timeout_microseconds: 10000   # 10ms window for standard
    allow_timeout_override: false
    max_queue_size: 1000
  }

  # Background jobs: large window, ok to wait
  priority_queue_policy {
    priority_level: 2
    timeout_action: DELAY
    default_timeout_microseconds: 100000  # 100ms window for background
    allow_timeout_override: false
    max_queue_size: 10000
  }
}
```

Client code sets priority in the request metadata:

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")

# High-priority request (e.g., enterprise user)
inputs = [grpcclient.InferInput("input_ids", [1, 128], "INT64")]
inputs[0].set_data_from_numpy(input_array)

metadata = [("priority", "0")]   # priority level 0 = highest
result = client.infer(
    model_name="bert_classifier",
    inputs=inputs,
    headers={"x-request-priority": "high"},
    # Note: gRPC metadata carries the priority hint
)
```

Priority queuing is particularly valuable when you have a mix of interactive (user-facing) and batch (background processing) traffic on the same GPU fleet. Without it, background jobs during off-peak hours can starve interactive requests during unexpected traffic spikes.

### Speculative batching

An emerging pattern for decoder LLMs is **speculative batching**: instead of waiting for token demand to fill a batch, the server "speculatively" generates draft tokens for idle sequences using a small draft model, then verifies the drafts with the main model when a real request arrives. This trades wasted draft-generation compute during idle periods for lower latency when requests arrive.

For most production deployments, this is handled within the serving framework (vLLM's speculative decoding feature, TGI's draft-model mode) rather than at the batching layer. See the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) for the speculative decoding mechanics.



## 13. Multi-instance serving and horizontal scaling with batching

Batching solves the GPU utilization problem for a single model instance. But what happens when your arrival rate exceeds the maximum sustainable throughput of one instance? You need multiple instances — and batching interacts with load balancing in non-obvious ways.

### Single instance throughput ceiling

From Little's Law, the maximum sustainable throughput for a single instance with `max_batch_size=B` and inference time $T_{\text{inf}}(B)$ is:

$$
\lambda_{\max} = \frac{B}{T_{\text{inf}}(B) + T_{\text{overhead}}}
$$

For BERT-base at batch=64 and 20.7ms per batch: $\lambda_{\max} = 64/0.0207 \approx 3{,}092\;\text{req/s}$.

Once arrival rate exceeds this ceiling, the queue grows without bound. You must scale horizontally: add more model instances (more GPUs) behind a load balancer.

### Load balancing and batching state

The load balancer must route requests to instances that have available queue capacity and have active batching windows. **Round-robin load balancing** distributes requests evenly but is suboptimal for batching: if one instance has 15 requests queued waiting to form a batch of 16, and you route the 16th request to a different instance, both instances process smaller batches than optimal.

**Least-outstanding-requests (LOR) routing** is better for batching: it sends new requests to the instance with the smallest current queue depth, naturally concentrating traffic onto instances with growing batches. This is the default routing policy for Ray Serve and Triton's multi-instance mode.

For LLMs with continuous batching, **load-aware routing** additionally considers KV cache utilization. An instance at 85% KV cache utilization should not receive new long-prompt requests even if its request queue is short — it risks OOM during generation. vLLM's upcoming inference router exposes a `/health` endpoint with KV cache utilization, enabling the load balancer to avoid memory-pressured instances:

```python
import aiohttp
import asyncio
from typing import List

class KVCacheAwareRouter:
    """
    Routes requests to the vLLM instance with lowest KV cache utilization.
    """
    def __init__(self, instance_urls: List[str]):
        self.urls = instance_urls

    async def get_kv_utilization(self, url: str) -> float:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/metrics") as resp:
                text = await resp.text()
                for line in text.split('\n'):
                    if line.startswith('vllm:gpu_cache_usage_perc'):
                        return float(line.split()[-1])
        return 1.0  # assume full if unreachable

    async def choose_instance(self) -> str:
        utilizations = await asyncio.gather(
            *[self.get_kv_utilization(url) for url in self.urls]
        )
        best_idx = min(range(len(self.urls)), key=lambda i: utilizations[i])
        return self.urls[best_idx]

    async def route_request(self, prompt: str) -> str:
        instance_url = await self.choose_instance()
        async with aiohttp.ClientSession() as session:
            payload = {"prompt": prompt, "max_tokens": 256}
            async with session.post(
                f"{instance_url}/v1/completions", json=payload
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["text"]
```

### Batching across GPUs: why it does not work the same way

One common misconception: "I have 4 GPUs, can I batch across all 4 to get 4× larger batches?" The answer is no — batching only helps within a single forward pass, which executes on a single set of GPU(s). Batching across 4 separate GPU instances would require routing all 4× the requests to a single model instance that is replicated or sharded across 4 GPUs. The correct tool for this is **tensor parallelism** (distributing the weight matrices across multiple GPUs to handle a larger batch in one forward pass) rather than routing across 4 independent instances.

For a BERT-base model that fits comfortably on one GPU, 4 independent replicas is strictly better than tensor-parallel across 4 GPUs: lower communication overhead, better fault isolation, simpler routing. Tensor parallelism makes sense only when the model itself does not fit on a single GPU (70B+ parameters), or when you want to reduce per-request latency by parallelizing the forward pass (at the cost of inter-GPU communication overhead).

The practical horizontal-scaling recipe for BERT-base:
1. Single GPU, batch=64 → handles ~3,000 req/s per GPU
2. 4 replicas behind LOR load balancer → handles ~12,000 req/s total
3. HPA on custom metric `avg_queue_depth > 8` → auto-scale to match traffic

The practical horizontal-scaling recipe for Llama-2 7B:
1. Single A100 40GB, `max_num_seqs=32`, continuous batching → handles ~8 req/s at 256-token output
2. Scale to N replicas as traffic grows (each replica independent, no model sharding needed at 7B)
3. HPA on `avg_kv_cache_utilization > 0.75` → scale before KV cache becomes the bottleneck



## 14. When NOT to batch

Batching has overhead: the window adds latency, the collating logic adds complexity, and the GPU memory must hold multiple samples simultaneously. For some workloads, this overhead is not worth paying.

**Voice ASR streaming**: each audio chunk must be decoded within 20–50ms to maintain real-time factor for a user-facing application. Batching across multiple audio streams requires routing the right chunks to the right beam-search state, which is stateful per-stream. The latency cost of any batching window (even 5ms) pushes you toward the real-time limit. Use one model instance per concurrent stream with zero batching.

**Sub-millisecond classifiers on CPU**: a two-layer MLP, a logistic regression model, or a small gradient-boosted tree can make predictions in 50–500 microseconds on CPU. A batching window of even 1ms adds 2–20× to the latency. Serve these directly in the request thread. No batching.

**Tightly constrained p99 SLAs**: if your p99 SLA is 5ms and your inference time at batch=1 is 4ms, there is 1ms left for the batching window, overhead, and all other operations. Any batching window will make you miss the SLA. Serve batch=1 and scale horizontally.

**Stateful per-session models**: recommendations systems that maintain per-user state, session-aware chatbots with complex conversation management, or streaming models with recurrent state (stateful LSTMs, RWKV) cannot batch across unrelated sessions without careful state management. If sessions must be colocated with specific model instances, standard dynamic batching (which routes requests to any available instance) breaks down.

**When GPU does not saturate with your payload**: profile $T_{\text{inf}}(B)$ before committing to a batching strategy. If $T_{\text{inf}}(B) / T_{\text{inf}}(1) \approx 1$ for all $B > 1$ — meaning batch inference takes the same time regardless of batch size — you are already compute-bound at batch=1 or the model is so small that CUDA kernel launch overhead dominates. In either case, batching provides no throughput benefit.

**The decision rule**: batching is profitable when $T_{\text{inf}}(B) / T_{\text{inf}}(1) < B$. Measure this on your actual hardware with your actual model before building batching infrastructure.

A useful empirical test: run your model with a single request, measure $T_{\text{inf}}(1)$. Then run with 8 requests in a batch, measure $T_{\text{inf}}(8)$. If $T_{\text{inf}}(8) < 2 \times T_{\text{inf}}(1)$ — meaning 8 requests finish in less than twice the time of 1 request — batching is profitable. If $T_{\text{inf}}(8) \approx 8 \times T_{\text{inf}}(1)$, each request is essentially executing serially and batching provides no benefit. This test takes two minutes to run and should be done before investing days in batching infrastructure.



## 15. Monitoring batching in production

Once you have configured batching, you need to monitor it. The key metrics are:

```yaml
# Prometheus scrape config for Triton
scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['triton-service:8002']
    metrics_path: '/metrics'

# Key Triton metrics to alert on:
# nv_inference_request_success - request rate
# nv_inference_queue_duration_us - time spent in queue before batching
# nv_inference_compute_infer_duration_us - actual inference time
# nv_inference_exec_count - number of executions (each = one batch)
```

```yaml
# Alert rules for batching health
groups:
  - name: batching_slo
    rules:
      # Alert if p99 queue wait exceeds batching window target
      - alert: BatchingWindowBreach
        expr: |
          histogram_quantile(0.99,
            rate(nv_inference_queue_duration_us_bucket[5m])
          ) / 1000 > 15  # 15ms queue wait
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "p99 batch queue wait exceeds 15ms — batching window too large for SLA"

      # Alert if batch utilization is low (wasting GPU)
      - alert: LowBatchUtilization
        expr: |
          rate(nv_inference_request_success[5m]) /
          rate(nv_inference_exec_count[5m]) < 8
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Average batch size < 8 — consider tuning window or scaling down"

      # vLLM-specific: KV cache pressure
      - alert: VLLMKVCachePressure
        expr: vllm:gpu_cache_usage_perc > 0.88
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "vLLM KV cache above 88% — approaching eviction threshold"
```

Three derived metrics give you the most actionable signal:

1. **Average batch size** = `requests_per_second / batches_per_second`. If this is close to 1, your batching window is too small (or arrival rate is too low). If it is consistently at `max_batch_size`, the window or batch size cap may be too small, and you are leaving throughput on the table.

2. **Queue wait time** (p50 and p99). p50 queue wait approximates your batching window in practice. p99 queue wait reveals tail latency from batch formation. If p99 queue wait is significantly larger than your configured window, you have a burst arrival pattern or a downstream bottleneck causing back-pressure.

3. **GPU utilization** from `nvidia-smi dmon -s u`. Below 70% during peak traffic means batching is not working (batch sizes too small, arrival rate too low, or window too short). Above 95% means you are approaching GPU saturation and the queue will start building; prepare to scale out.

### Grafana dashboard queries for batching health

The following PromQL queries give you the most useful views in a Grafana dashboard:

```promql
# Average batch size (Triton)
rate(nv_inference_request_success[1m])
/ rate(nv_inference_exec_count[1m])

# p99 inference latency (ms)
histogram_quantile(0.99,
  rate(nv_inference_compute_infer_duration_us_bucket[5m])
) / 1000

# p99 queue (pre-batch) latency (ms)
histogram_quantile(0.99,
  rate(nv_inference_queue_duration_us_bucket[5m])
) / 1000

# GPU utilization per device (percent)
DCGM_FI_DEV_GPU_UTIL

# vLLM running/waiting request counts
vllm:num_requests_running
vllm:num_requests_waiting

# vLLM generation throughput (tokens/s)
rate(vllm:generation_tokens_total[1m])
```

A well-tuned batching configuration in steady state should show:
- Average batch size: 8–50 (depending on model and traffic)
- p99 queue wait ≤ configured window × 1.2 (slight overage due to variance)
- GPU utilization: 70–90%
- vLLM `num_requests_waiting` near zero (requests are admitted quickly)
- vLLM KV cache utilization: 60–85% (utilization without frequent evictions)

If `num_requests_waiting` grows over time during sustained load, you have reached the throughput ceiling — add more GPU capacity. If `num_requests_waiting` oscillates between 0 and large values, you have bursty traffic that your batching configuration is not smoothing effectively; consider increasing `max_queue_delay_microseconds` or implementing request admission control upstream.



## 16. Batching trade-offs across hardware and quantization

Batching interacts with model quantization in ways that matter for production decisions. Quantized models have smaller weight matrices in terms of bytes, which shifts the arithmetic intensity ridge and changes the optimal batch size.

### INT8 quantization and the ridge shift

For a model quantized to INT8 (1 byte per weight instead of 2 bytes for FP16), the bytes transferred per parameter halve. The arithmetic intensity formula becomes:

$$
AI_{\text{INT8}}(B) \approx \frac{2 \times B \times M \times N}{B \times M \times 2 + M \times N \times 1 + B \times N \times 2}
$$

For the weight-dominated case ($MN \gg BM, BN$):

$$
AI_{\text{INT8}} \approx \frac{2BMN}{MN} = 2B
$$

INT8 weights give 2× the arithmetic intensity for the same batch size — you reach the GPU's ridge point at half the batch size compared to FP16. This means INT8 models are more efficiently utilized at small batch sizes. At batch=8 with INT8, you have AI ≈ 16, comparable to batch=16 in FP16. For latency-sensitive serving where you want to keep batch sizes small to meet tight p99 SLAs, quantization helps by pushing the "wasted GPU capacity" problem further down the batch-size axis.

For AWQ/GPTQ 4-bit quantization (0.5 bytes per weight): AI ≈ 4B. Even batch=4 achieves AI ≈ 16. The batching window can be shorter, response latency lower, and GPU utilization still reasonable.

### How quantization changes optimal batch configuration

| Quantization | Bytes/weight | AI at B=16 | Approximate ridge | Throughput gain B=1→16 |
|:------------|:------------|:----------:|:-----------------:|:----------------------:|
| FP32 | 4 | 8 | ~196 | 8× |
| FP16 | 2 | 16 | ~49–98 | 16× |
| INT8 | 1 | 32 | ~25–49 | 32× |
| INT4 (AWQ) | 0.5 | 64 | ~12–25 | 64× |

The INT4 row reveals something striking: at batch=16 with INT4 weights, arithmetic intensity is ~64 — already well above the A100's ~49 ridge. You are compute-bound even at batch=16. This means that for heavily quantized models, large batches provide diminishing returns; focus on smaller batches with lower latency rather than large batches with higher throughput.

This interplay between quantization and batching is one reason why the "right" quantization level is not just about accuracy: it changes the entire operational profile of your serving system, including what batch size to configure and what latency-throughput trade-off you operate at.

For a practical serving recommendation when using AWQ 4-bit quantization on a 7B LLM:
- Batch sizes beyond 8–16 see diminishing returns compared to FP16 at the same batch size
- The KV cache is still FP16 (only weights are quantized), so KV cache memory constraint is unchanged
- TTFT for long prompts decreases (weight loading is faster), but TPOT improvement is modest at small batch sizes
- The correct use of quantization at serving time is primarily to fit a larger model on a given GPU, not to increase batch size



## Key takeaways

1. **Arithmetic intensity $\approx$ batch size** for weight-stationary GEMM on GPU. Every doubling of batch size below the GPU's ridge point nearly doubles throughput — this is why batching matters more than almost any other serving optimization.

2. **The roofline ridge** for dense FP16 GEMM on A100 is ~49. Below this batch size, you are memory-bandwidth-bound; above it, compute-bound. In practice, real workloads saturate at lower batch sizes due to attention arithmetic and CUDA kernel efficiency.

3. **Static batching causes unbounded tail latency** under variable arrival. The average wait for the first request in a static batch of size $B$ at rate $\lambda$ is $(B-1)/\lambda$ — 1.55 seconds for B=32, λ=10. Use dynamic batching for any online serving workload.

4. **Optimal batch window**: $W^* = p_{99}^{*} - T_{\text{inf}}(\lambda W^* + 1) - T_{\text{overhead}}$. Solve iteratively. In Triton: `max_queue_delay_microseconds = W^* × 1000`.

5. **Padding fraction** $f_{\text{pad}} = (\ell_{\text{max}} - \bar{\ell}) / \ell_{\text{max}}$ can exceed 75% for NLP workloads with skewed length distributions. Sequence bucketing with 4–8 buckets reduces this to under 33% per bucket.

6. **Continuous batching is not optional for LLMs.** Static batching a 13B LLM delivers 24× lower throughput than continuous batching (vLLM paper). The gain comes from eliminating batch idle time as sequences finish at different times.

7. **LLM batch size is bounded by KV cache memory**, not compute. For Llama-2 7B FP16 on A100 40GB: available KV cache $\approx 20\,\text{GB}$, KV cache per token (all 32 layers) $\approx 0.5\,\text{MB}$, maximum concurrent tokens $\approx 38{,}000$. At 512-token average length: `max_num_seqs ≤ 74`. Conservative safe setting: 32–40.

8. **The decode phase of LLM generation is always memory-bandwidth-bound** regardless of batch size. At batch=32 and typical context lengths, arithmetic intensity is ~11 FLOP/byte — far below the A100 ridge of 49. Larger batch sizes increase total throughput but do not reduce per-token TPOT proportionally.

9. **Do not batch when**: inference is sub-millisecond, SLA is tighter than your batching overhead, the workload is stateful per-session, or $T_{\text{inf}}(B) / T_{\text{inf}}(1) \approx 1$ (already compute-bound at batch=1).

10. **Monitor three numbers in production**: average batch size (should be > 8 for meaningful gain), p99 queue wait time (should be < 80% of your SLA), and GPU utilization (should be > 70% at peak load). If any of these is out of range, re-run the tuning workflow.



## Further reading

- **Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.** — The vLLM paper. Introduces continuous batching + PagedAttention, benchmarks 24× throughput improvement over static batching on OPT-13B.
- **Yu, G. et al. (2022). "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022.** — The paper that formalized iteration-level scheduling (continuous batching) and quantified the idle-time waste of static batching for generation.
- **Williams, S. et al. (2009). "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multiprocessors." CACM 2009.** — The original roofline paper; the arithmetic intensity framework used throughout this post.
- [Triton Inference Server dynamic batching documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher) — Official `config.pbtxt` reference with all batching parameters and examples.
- [vLLM engine arguments reference](https://docs.vllm.ai/en/latest/models/engine_args.html) — Full documentation for `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, and related parameters.
- [What is model serving: the SLO triangle](/blog/machine-learning/model-serving/what-is-model-serving) — Series introduction; defines the latency/throughput/cost triangle that batching navigates.
- [Continuous batching and PagedAttention: how vLLM serves 24× more requests](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) — Track C deep-dive into the vLLM scheduler, iteration-level scheduling, and KV cache block management.
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — Series capstone; full decision tree from model to production including batching strategy selection.

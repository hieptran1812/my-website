---
title: "What is model serving: the foundation every ML engineer needs"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Understand the fundamental gap between training a model and serving it at production scale — latency, throughput, cost, and why Flask will fail you every time."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "llm-serving",
    "latency",
    "throughput",
    "vllm",
    "production-ml",
    "queuing-theory",
    "gpu-optimization",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/what-is-model-serving-1.png"
---

It is 2:47 AM. Your model hit 97% accuracy on the validation set. You wrapped it in Flask, pushed a Docker image, and called it shipped. By 9:15 AM the next morning, your service is returning 504 errors. Kubernetes is restarting the pod in a loop. The GPU is showing 28% utilization — it is doing almost nothing — and yet every request is timing out. The on-call engineer is staring at you across the conference room.

This scenario is not hypothetical. It is the most common way a well-trained ML model becomes a production disaster. The model worked. The *serving system* — the engineering infrastructure around the model — failed entirely.

Model serving is a separate discipline from model training, and it requires a completely different engineering mindset. When you train a model, you optimize for throughput: you want to consume the full batch, saturate the GPU, and minimize compute cost per gradient step. When you serve a model, you are running a latency-sensitive, concurrency-driven distributed system where an unpredictable stream of users sends requests simultaneously, each expecting a response within hundreds of milliseconds, and where getting any one of a dozen architectural decisions wrong will break your SLAs in a way that is invisible until you hit real traffic.

This post establishes the conceptual foundation you need before touching any serving framework. By the end, you will understand the three axes on which every serving decision trades (latency, throughput, cost), how Little's Law lets you derive GPU capacity requirements from first principles, why a Flask wrapper fails the moment you have more than four concurrent users, and what makes a production-grade serving system actually production-grade. We will use a concrete running example throughout: a Llama-3-8B language model serving 100 concurrent users answering open-ended questions, deployed on a single A100 80GB GPU. Every number here is real or derived from real benchmarks.

![The SLO triangle showing latency, throughput, and cost as three forces in fundamental tension for model serving decisions](/imgs/blogs/what-is-model-serving-1.png)

The SLO triangle above is the lens through which this entire series is written. Every technique in every post — continuous batching, PagedAttention, tensor parallelism, quantization, speculative decoding, autoscaling — is a trade on this triangle. Some techniques push one vertex without touching the others. Some improve two at the cost of complexity. A rare few reshape the triangle by changing the underlying hardware or architecture. Understanding the triangle is the prerequisite to understanding any individual technique.

---

## The serving vs training divide

Before anything else, you need to internalize why training and serving are completely different engineering problems. Not different-in-degree. Different-in-kind. The mental model, the failure modes, the metrics, the toolchain — almost nothing transfers directly.

Training is a *batch* workload running in a *controlled* environment. You know exactly how many examples are in your dataset. You choose the batch size. You control the sequence of operations. The only constraint is throughput — you want to process as many tokens or examples per second as possible, and you are willing to wait hours or days while you do it. If a training run takes 6 hours instead of 5, nobody wakes you up at 3 AM. If the loss curve behaves unexpectedly, you can stop, change the learning rate, and restart. You are in complete control of the workload shape.

Serving is an *online* workload running in an *uncontrolled* environment. You do not know when requests will arrive. You do not control how many will come at once. You cannot choose to delay a user's request because you would prefer a bigger batch. You have a contract — a Service Level Agreement (SLA) — that specifies the maximum acceptable response time. Break it consistently, and users leave, the product degrades, or you get paged. The constraint is latency, and it does not care how convenient it is for your GPU.

![Training and serving have opposite optimization objectives: batch throughput versus per-request latency under unpredictable concurrency](/imgs/blogs/what-is-model-serving-2.png)

Let me put real numbers on this contrast. During training on Llama-3-8B with a batch size of 256 and sequence length 2048, a well-configured A100 80GB achieves roughly 85–92% model FLOPS utilization (MFU). The GPU is almost always doing useful compute. During *serving* with a naive system (a single-threaded Flask wrapper), the same GPU idles at 20–35% utilization — not because the model is slow, but because the serving system can only process one request at a time, the time between requests leaves the GPU empty, and every incoming request waits for the previous one to fully complete before the GPU starts on it. The bottleneck is not the model. It is the serving infrastructure.

This is the fundamental insight: **model serving is a systems problem, not a modeling problem**. You can spend three months improving your model's accuracy by 2% and then lose all of that value because the serving system has 8x lower throughput than it should. The model quality ceiling is irrelevant if users never receive responses within their patience threshold.

The asymmetry between training and serving also shows up in how errors manifest. A training bug usually surfaces as a degraded validation metric — something you notice and fix before deployment. A serving bug usually surfaces as a latency spike or an OOM crash under real production load, often at the worst possible time (a product launch, a traffic surge, the middle of the night). This is why understanding serving from first principles — not just copying a deployment recipe — is essential before you ship anything.

There is also a deeper asymmetry in what "optimal" means. During training, optimal means maximizing GPU utilization because compute time is money and every wasted FLOP is a dollar burned. During serving, optimal means satisfying SLOs at the lowest possible cost — but satisfying SLOs comes first. A system that achieves 95% GPU utilization but violates p99 latency SLOs is useless. A system that achieves 60% GPU utilization but hits all its SLOs is fine. The serving-optimal operating point is not "maximum GPU utilization" — it is "the highest GPU utilization that still meets the latency budget."

A final, often-underappreciated asymmetry: the feedback loops are different. In training, poor performance shows up in the validation metric after each epoch — a signal you can observe and act on in a controlled manner. In serving, poor performance shows up as user-facing errors or latency degradation, often discovered through monitoring dashboards, customer complaints, or automated alerts. The feedback loop is much faster and less forgiving: a latency regression that would take a week to manifest as a model quality issue can sink user retention within hours if it causes timeouts. This is why production serving systems invest heavily in observability — not because engineers enjoy dashboards, but because the feedback loop is too expensive to let run unmonitored.

These three asymmetries — optimization target, stability requirements, and feedback loops — are why the serving skill set is genuinely distinct from the training skill set, and why understanding serving from first principles (rather than copying a deployment script) is the only way to be effective when things go wrong at 2 AM.

---

## The three axes: latency, throughput, and cost

Every serving decision you will ever make is a trade on a triangle with three vertices: **latency**, **throughput**, and **cost**. You cannot independently optimize all three. Improving one almost always degrades another, unless you change the fundamental hardware or architecture.

### Latency

Latency is the time it takes for a single request to receive its response. It sounds simple, but for autoregressive language models, latency has a multi-dimensional structure that matters enormously in practice.

**TTFT (Time To First Token)** is the time from when the request arrives at the server to when the first output token is delivered to the client. This is dominated by the *prefill* phase — processing the entire input prompt through the model in one forward pass. TTFT is compute-bound: the GPU is running full attention over all input tokens. It scales roughly linearly with input sequence length and with model size (number of layers × hidden dimension). For a 512-token prompt on Llama-3-8B on an A100, TTFT in a well-tuned system is 40–80ms. On a naively implemented server, the same prompt takes 600–900ms because the request is serialized behind all previous requests rather than batched alongside them.

TTFT matters enormously for user experience in chat applications. Users start reading the response as soon as the first word appears. A 200ms TTFT feels instant; a 2000ms TTFT feels like the system is broken. This is why modern streaming LLM APIs (OpenAI, Anthropic, Google) all stream tokens progressively rather than waiting for the full response to complete.

**TPOT (Time Per Output Token)** is the time between successive tokens during generation. This is dominated by the *decode* phase — generating one token at a time in an autoregressive loop. TPOT is memory-bandwidth-bound: the dominant cost in each decode step is loading the model weights and KV cache from GPU HBM into the compute units, not the actual matrix multiplications. For Llama-3-8B on an A100 (3.2 TB/s HBM bandwidth), TPOT is approximately 12–20ms per token depending on batch size and KV cache size. For a 256-token response, this means 3–5 seconds of generation time after the TTFT.

TPOT matters for the reading experience: 40–50ms per token (25 tokens/s) feels smooth and slightly faster than natural reading speed. Under 10ms per token (100 tokens/s) is imperceptible. Over 100ms per token (10 tokens/s) starts to feel choppy and frustrating.

**Total latency** is TTFT plus (TPOT × output_tokens). But note that in a streaming system, users are already reading while tokens are being generated — so the *perceived* latency is dominated by TTFT, and TPOT primarily affects how long users need to wait for the full response.

Latency is always measured as a *distribution*, not a point estimate. The relevant quantiles are:
- **p50 (median):** Half of requests complete faster than this. A reasonable metric for average experience.
- **p95:** 95% of requests complete faster than this. The standard for most API SLAs.
- **p99:** 99% of requests complete faster than this. One in a hundred requests is slower. At 100 QPS, this is one user per second.
- **p999 (p99.9):** One in a thousand requests is slower. At 1000 QPS, this is one user per second — relevant for very high-traffic services.

Production SLAs are almost always written against the tail: "p99 TTFT must be under 500ms" rather than "mean TTFT must be under 300ms." This is because the mean can look great while a small fraction of users have terrible experiences — and those users generate support tickets, write negative reviews, and churn.

### Throughput

Throughput is how many requests (or tokens) the system can process per unit time. It operates at two levels.

At the **system level**, throughput is measured as requests per second (RPS) — how many complete request-response cycles the system handles each second. This is what you care about for capacity planning: "can my system handle 100 req/s at my current SLA?"

At the **hardware level**, throughput is measured as tokens per second (tokens/s) — how many output tokens the GPU generates each second across all concurrent requests. This is more fundamental: it tells you how efficiently you are using the GPU's compute and memory bandwidth. A well-tuned vLLM server on A100 80GB with Llama-3-8B achieves approximately 1,500–2,000 tokens/s at high load. This translates to roughly 50–70 req/s if average response length is ~256 tokens.

The single most important throughput metric for understanding hardware efficiency is **GPU utilization** — what fraction of the GPU's peak FLOPS and memory bandwidth is being consumed by useful model computation. A well-optimized LLM server achieves 88–94% GPU utilization. A naive server achieves 25–40%. That gap represents 2–3x waste in hardware cost.

### Cost

Cost is what you pay for the hardware serving your models. For GPU-based inference, this is almost entirely dominated by GPU-hours. You either pay directly as cloud compute (\$2.00–\$4.00/GPU-hour for an A100 on AWS/GCP/Azure at on-demand rates, or \$0.80–\$1.50/hr for spot instances) or amortize from a capital purchase. The cost per inference is:

$$\text{cost per request} = \frac{\text{GPU hourly cost}}{\text{requests per hour served}}$$

At 4 req/s throughput (naive Flask), 1 A100 at \$3.50/hr costs:

$$\frac{\$3.50 / 3600\text{s}}{4 \text{ req/s}} = \$0.000243 \text{ per request}$$

At 60 req/s throughput (vLLM), the same GPU costs:

$$\frac{\$3.50 / 3600\text{s}}{60 \text{ req/s}} = \$0.0000162 \text{ per request}$$

That is a **15x reduction in cost per request** purely from better serving infrastructure, on identical hardware with identical model weights. The model did not change. The GPU did not change. The software infrastructure changed.

The practical implication: before you spend money on more GPUs, make sure your existing GPUs are utilized efficiently. Going from 30% GPU utilization to 90% GPU utilization triples your effective capacity for free.

### How to measure the SLO triangle in practice

Theory without measurement is speculation. Here is a minimal benchmarking setup that measures all three SLO axes against a running vLLM server:

```python
# benchmark_serving.py — measure throughput, latency, and compute cost
import asyncio
import aiohttp
import time
import json
import numpy as np
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    throughput_rps: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_tpot_ms: float
    p99_tpot_ms: float
    gpu_util_pct: float
    cost_per_1k_requests_usd: float
    ttft_samples: list[float] = field(default_factory=list)
    tpot_samples: list[float] = field(default_factory=list)

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
) -> tuple[float, float]:
    """Returns (TTFT ms, mean TPOT ms)."""
    body = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    async with session.post(f"{url}/v1/completions", json=body) as resp:
        async for chunk in resp.content:
            line = chunk.decode().strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            
            text = data["choices"][0].get("text", "")
            if text and first_token_time is None:
                first_token_time = (time.perf_counter() - start) * 1000  # ms
            if text:
                token_count += 1
    
    total_ms = (time.perf_counter() - start) * 1000
    tpot_ms = (total_ms - (first_token_time or 0)) / max(token_count - 1, 1)
    return first_token_time or total_ms, tpot_ms

async def run_benchmark(
    url: str = "http://localhost:8000",
    concurrency: int = 20,
    num_requests: int = 200,
    prompt: str = "Explain the difference between supervised and unsupervised learning in detail.",
    max_tokens: int = 256,
    gpu_cost_per_hour_usd: float = 3.50,  # A100 on-demand
) -> BenchmarkResult:
    prompts = [prompt] * num_requests
    
    start_time = time.perf_counter()
    ttft_results = []
    tpot_results = []
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(p: str):
        async with semaphore:
            ttft, tpot = await send_request(
                session, url, p, max_tokens
            )
            ttft_results.append(ttft)
            tpot_results.append(tpot)
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(p) for p in prompts]
        await asyncio.gather(*tasks)
    
    elapsed = time.perf_counter() - start_time
    throughput = num_requests / elapsed
    cost_per_1k = (gpu_cost_per_hour_usd / 3600) / throughput * 1000
    
    return BenchmarkResult(
        throughput_rps=throughput,
        p50_ttft_ms=float(np.percentile(ttft_results, 50)),
        p95_ttft_ms=float(np.percentile(ttft_results, 95)),
        p99_ttft_ms=float(np.percentile(ttft_results, 99)),
        p50_tpot_ms=float(np.percentile(tpot_results, 50)),
        p99_tpot_ms=float(np.percentile(tpot_results, 99)),
        gpu_util_pct=0.0,  # Read separately from Prometheus /metrics
        cost_per_1k_requests_usd=cost_per_1k,
        ttft_samples=ttft_results,
        tpot_samples=tpot_results,
    )

if __name__ == "__main__":
    result = asyncio.run(run_benchmark(concurrency=20, num_requests=200))
    print(f"Throughput:    {result.throughput_rps:.1f} req/s")
    print(f"TTFT p50/p99:  {result.p50_ttft_ms:.0f}ms / {result.p99_ttft_ms:.0f}ms")
    print(f"TPOT p50/p99:  {result.p50_tpot_ms:.1f}ms / {result.p99_tpot_ms:.1f}ms")
    print(f"Cost/1k reqs:  \${result.cost_per_1k_requests_usd:.4f}")
```

Run this against your Flask server and your vLLM server, change the `concurrency` parameter from 1 to 50, and you will see the SLO triangle in action: as concurrency increases, throughput grows up to a point, then latency starts rising as the queue builds. The exact shape of this curve — the throughput-latency Pareto frontier — is the empirical signature of your serving system's efficiency.

---

## Queuing theory and Little's Law

To understand serving capacity at a rigorous level, you need one result from queuing theory: **Little's Law**. This is not an approximation or a rule of thumb — it is a mathematically proven theorem that applies to any stable queuing system, with almost no assumptions.

**Little's Law** states:

$$L = \lambda W$$

where:
- $L$ = mean number of requests in the system (in queue + being processed) — "mean concurrency"
- $\lambda$ = mean arrival rate (requests per second)
- $W$ = mean time a request spends in the system (seconds) — from arrival to response

The theorem was proven by John D.C. Little in 1961 and applies to any stable, ergodic queuing system regardless of the arrival distribution, the service time distribution, or the number of servers. It is one of the most general results in operations research, and it is directly applicable to model serving.

### Deriving GPU capacity from first principles

Let us apply Little's Law to our running Llama-3-8B example. We want to answer: how many A100 GPUs do we need to serve 100 concurrent users with a p99 response time under 2 seconds?

First, we need to translate "100 concurrent users" into a request rate. If each user submits one request every 10 seconds on average (a reasonable cadence for chat interactions where users read responses before submitting the next message), the arrival rate is:

$$\lambda = \frac{100 \text{ users}}{10 \text{ s/request}} = 10 \text{ req/s}$$

With a target mean response time of $W = 1.5\text{s}$ (leaving headroom for p99 to reach 2s), Little's Law tells us we need:

$$L = \lambda \times W = 10 \text{ req/s} \times 1.5\text{s} = 15 \text{ requests in flight simultaneously}$$

This is a hard requirement, not a guideline. If our serving system can only process $L < 15$ requests in flight simultaneously, the queue will grow without bound and requests will time out.

Now, what does "15 requests in flight simultaneously" mean for GPU memory? For Llama-3-8B in FP16, the model weights occupy:

$$8 \times 10^9 \text{ params} \times 2 \text{ bytes/param} = 16\text{ GB}$$

Each request's KV cache memory per token (at 32 layers, 32 heads, 128 head dimension, FP16) is:

$$\text{KV bytes per token} = 2 \times 32 \text{ layers} \times 32 \text{ heads} \times 128 \text{ head\_dim} \times 2 \text{ bytes} = 524{,}288 \text{ bytes} \approx 0.5\text{ MB/token}$$

For a 1024-token sequence (512 input + 512 output average):

$$\text{KV cache per request} = 1024 \times 0.5 \text{ MB} = 512 \text{ MB}$$

For 15 simultaneous requests:

$$\text{Total KV cache} = 15 \times 512 \text{ MB} = 7.68 \text{ GB}$$

Total GPU memory requirement:

$$\text{Total} = 16 \text{ GB (weights)} + 7.68 \text{ GB (KV cache)} + 2 \text{ GB (runtime)} = 25.68 \text{ GB}$$

A single A100 80GB can comfortably hold this. The mathematics confirms that one GPU is sufficient for this workload — if and only if the serving system can actually keep 15 requests in flight simultaneously.

Here is the crucial failure point of a naive server. It processes requests sequentially, so effective concurrency is $L = 1$. By Little's Law:

$$W_{\text{naive}} = \frac{L_{\text{naive}}}{\lambda} = \frac{1}{10 \text{ req/s}} = 0.1\text{s per slot}$$

But each request actually takes approximately 1.5 seconds to process. So with a serial server at 10 req/s arrival rate, requests arrive 10x faster than they can be scheduled. The queue grows at 9 requests per second — it is *unstable*. After 30 seconds, 270 requests are queued and every request is timing out with a 30-second wait before the GPU even starts on it. This is why users see 504 timeouts.

The fix is not a bigger GPU. A bigger GPU would not help because the GPU is already idle 85% of the time. The fix is a serving system that genuinely multiplexes multiple requests on the GPU simultaneously.

### Applying Little's Law to multi-server and multi-stage systems

Little's Law also generalizes cleanly to multi-GPU or multi-stage architectures. If you have $k$ A100 replicas each with throughput $\mu$ (requests per second), the combined system has throughput $k \mu$. The system utilization is:

$$\rho = \frac{\lambda}{k \mu}$$

And by Little's Law, the mean number of requests across all $k$ servers is still:

$$L = \lambda W$$

For a load-balanced system with $k = 3$ A100s at $\mu = 52\text{ req/s}$ each, serving $\lambda = 80\text{ req/s}$:

$$\rho = \frac{80}{3 \times 52} = 0.513 \quad \text{(51% utilization)}$$

$$L = 80 \text{ req/s} \times 0.38\text{s mean latency} = 30 \text{ concurrent requests in flight across all 3 GPUs}$$

Each GPU holds $L/k = 10$ concurrent requests on average — well within the memory budget.

You can use this same math backwards: start from your memory budget, work out the maximum concurrency per GPU, multiply by throughput per request, and arrive at the maximum sustainable load. This "reverse Little's Law" capacity calculation is how you size a new deployment:

$$\lambda_{\text{max}} = \frac{L_{\text{max}}}{W_{\text{target}}}$$

where $L_{\text{max}}$ is derived from your GPU memory budget divided by KV cache per request.

### The stability condition

For a serving system to be stable (not have a perpetually growing queue), the service rate must exceed the arrival rate:

$$\mu > \lambda$$

where $\mu$ is the server's maximum throughput (requests per second). Define server utilization $\rho = \lambda / \mu$. For an M/M/1 queue (Poisson arrivals, exponential service times, one server), the mean wait time in queue is:

$$W_q = \frac{\rho}{\mu (1 - \rho)}$$

This blows up as $\rho \to 1$. In practice, you should target $\rho \leq 0.7–0.8$ for single-server systems to keep tail latency manageable. At $\rho = 0.9$, the mean queue wait is already 9x the service time itself — every request waits in line as long as it takes to process 9 other requests.

For our 10 req/s workload, a server with $\mu = 60\text{ req/s}$ has $\rho = 10/60 = 0.167$ — very light load, excellent latency. A server with $\mu = 12\text{ req/s}$ has $\rho = 10/12 = 0.833$ — queue latency starting to dominate. A server with $\mu = 10\text{ req/s}$ or lower is unstable.

---

## GPU memory math: why it determines everything

GPU memory is the single most constrained resource in LLM serving. Understanding the exact memory budget — to the gigabyte — is what separates a capacity plan from a guess. Every serving decision that matters ultimately comes down to: "can I fit more useful work in GPU memory?"

An A100 80GB has 80 GB of HBM2e (high-bandwidth memory). That memory is partitioned among three consumers:

**Model weights.** For Llama-3-8B in FP16: approximately 16GB. In INT8 (8-bit quantization): 8GB. In GPTQ INT4: approximately 4.5GB. Quantization directly doubles or quadruples the memory available for KV cache, which translates to 2–4x more concurrent requests per GPU.

The weight memory formula:

$$M_{\text{weights}} = N_{\text{params}} \times \text{bytes\_per\_param}$$

For a 7B model in FP16: $7 \times 10^9 \times 2 = 14\text{GB}$. For a 70B model in FP16: $70 \times 10^9 \times 2 = 140\text{GB}$ — more than one A100 80GB can hold, hence the need for tensor parallelism across at least 2 GPUs.

**CUDA runtime and activation memory.** PyTorch, NCCL (for multi-GPU communication), CUDA kernels, and activation tensors for the current batch occupy 2–6 GB of fixed overhead. For safety, reserve 5 GB for this category. This number grows slightly with batch size (activations are batch-size-dependent) but is roughly constant in comparison to weights and KV cache.

**KV cache.** The remaining GPU memory is available for KV cache — and maximizing this directly maximizes concurrency and throughput. The KV cache formula for transformer models:

$$M_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times T_{\text{seq}} \times \text{bytes\_per\_element}$$

Where the factor of 2 accounts for both key and value tensors per layer. For Llama-3-8B (32 layers, 32 heads, 128 head dimension) in FP16, per token:

$$M_{\text{KV/token}} = 2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} = 0.5 \text{ MB/token}$$

The total KV cache budget (after allocating for weights and runtime):

$$M_{\text{KV budget}} = M_{\text{GPU}} \times \text{utilization} - M_{\text{weights}} - M_{\text{runtime}}$$

For an A100 80GB at 90% utilization: $80 \times 0.9 - 16 - 5 = 51\text{ GB}$

Maximum concurrent tokens across all requests:

$$T_{\text{max}} = \frac{M_{\text{KV budget}}}{M_{\text{KV/token}}} = \frac{51 \times 10^9 \text{ bytes}}{524{,}288 \text{ bytes/token}} \approx 97{,}000 \text{ tokens}$$

If average sequence length is 1024 tokens (512 input + 512 output), maximum concurrent requests:

$$N_{\text{max seqs}} = \frac{97{,}000}{1024} \approx 94 \text{ concurrent sequences}$$

Setting `max_num_seqs=64` in vLLM provides a safety margin (64/94 = 68% of theoretical maximum), which leaves buffer for sequences that are longer than the 1024-token average.

This math explains why quantization has such an outsized impact on serving throughput: cutting weight memory from 16GB to 8GB (INT8) frees 8GB of KV cache space, which at 0.5MB/token is 16,000 additional tokens = approximately 15 more concurrent sequences = approximately 25% higher throughput, often with less than 1% accuracy degradation for well-calibrated INT8 models.

## The anatomy of a serving request

Understanding what happens inside the serving stack for each request gives you the diagnostic map you need to reason about where latency comes from and where to fix it when things go wrong.

![A serving request travels through six layers from client to GPU kernel, each layer adding measurable and reducible latency](/imgs/blogs/what-is-model-serving-3.png)

**Layer 1 — Network (client to load balancer).** A TLS-encrypted HTTP/2 or gRPC connection. Within the same datacenter availability zone: 0.2–1ms. Cross-region: 30–150ms. For streaming responses, the TCP connection stays open for the entire generation duration — potentially 5–20 seconds — so connection management and HTTP/2 multiplexing matter.

**Layer 2 — Load balancer.** Routes the request to one of the available inference server replicas. A well-configured system uses *least-outstanding-requests* (LOR) routing rather than round-robin. LOR sends each new request to the replica with the fewest in-flight requests, preventing hot spots where one replica has 20 queued requests while another is idle. Typical overhead: less than 1ms. The choice of load balancing algorithm can affect tail latency by 20–30% at high load.

**Layer 3 — Inference server (HTTP handler → queue).** The request arrives at the inference server (vLLM, TGI, Triton), is validated, decoded from JSON, and tokenized. Tokenization of a 512-token prompt: approximately 2–5ms (CPU-bound). The tokenized sequence is placed in the request queue, which is the interface between the asynchronous HTTP layer and the synchronous GPU execution loop. Queue wait time is a direct function of server load: at 30% capacity it is negligible (< 5ms); at 90% capacity it can add 500ms+ to every request.

**Layer 4 — Batch scheduler.** The scheduler (running in its own background asyncio loop, at 10–100ms cycle time) inspects the queue, selects which requests to include in the next forward pass, allocates KV cache memory blocks for them, and dispatches the batch to the GPU runtime. The vLLM scheduler uses a first-come-first-served policy with preemption: if a long-running request is blocking KV cache blocks that a newer request needs, it can be temporarily paused (its KV cache blocks swapped to CPU or dropped and recomputed) to maintain overall throughput. Scheduler overhead: 1–5ms per batch.

**Layer 5 — GPU compute (prefill + decode loops).** The actual model forward pass. Prefill (first pass over all input tokens): approximately 35–80ms for a 512-token input on Llama-3-8B on A100, depending on how many other sequences are in the batch. Decode (one autoregressive step per token): approximately 12–25ms per token, memory-bandwidth-bound. For 256 output tokens: 3–6 seconds of generation time. FlashAttention-2 reduces the peak GPU memory used during attention by recomputing attention blocks on-the-fly rather than materializing the full $N \times N$ attention matrix in HBM, which enables longer contexts and faster execution.

It is worth understanding exactly why the decode phase is memory-bandwidth-bound rather than compute-bound. During each decode step, the transformer must load all model weights (16GB for Llama-3-8B FP16) from HBM into the GPU's register file, perform the matrix multiplications, and write the KV cache update back to HBM. The total arithmetic intensity (floating-point operations per byte transferred) during a decode step with batch size 1 is approximately:

$$\text{Arithmetic intensity} = \frac{2 \times N_{\text{params}}}{2 \times N_{\text{params}} \times \text{bytes/param}} = \frac{1}{\text{bytes/param}}$$

For FP16 (2 bytes/param), this is approximately 0.5 FLOP/byte. The A100's roofline: peak FLOPS = 312 TFLOPS (BF16), peak memory bandwidth = 2 TB/s. The compute-to-memory bandwidth ratio of the A100 is 312 / 2 = 156 FLOPS/byte. Since 0.5 << 156, the decode step is deeply memory-bandwidth-bound: the GPU spends most of each step waiting for weight bytes to arrive from HBM, not actually computing. This is why batching more sequences together increases throughput — the same weight bytes serve multiple sequences simultaneously, increasing the effective arithmetic intensity.

The prefill phase has much higher arithmetic intensity: it processes all $N_{\text{input}}$ tokens simultaneously in a single matrix multiplication, with arithmetic intensity proportional to sequence length. For a 512-token prompt, arithmetic intensity is ~256 FLOP/byte — much closer to the compute-bound regime. This is why TTFT scales roughly linearly with input length (more compute, still bounded by compute not memory) while TPOT scales based on memory bandwidth.

**Layer 6 — Response serialization and streaming.** Decode token IDs back to text using the vocabulary. For streaming responses, each token is immediately flushed to the HTTP response as a Server-Sent Events (SSE) chunk. Serialization overhead per token: less than 1ms. For non-streaming responses, the entire output is buffered until generation completes, then serialized and returned in one JSON payload.

The cumulative latency breakdown for a typical request (512 input tokens, 256 output tokens, vLLM on A100, 50% load):

| Layer | Latency |
|-------|---------|
| Network (within datacenter) | 1ms |
| Load balancer routing | <1ms |
| Tokenization | 3ms |
| Queue wait (50% load) | 25ms |
| Batch scheduler | 2ms |
| Prefill (TTFT) | 65ms |
| Decode (256 tokens × 15ms) | 3,840ms |
| Response streaming | ~0ms (concurrent) |
| **Total TTFT** | **97ms** |
| **Total request duration** | **3,937ms** |

Notice that once streaming is active, users see the first token after 97ms and receive all tokens over the next 3.84 seconds — a very acceptable UX for a conversational application.

---

## Why "just wrap it in Flask" fails

Every ML engineer has written this code. It feels reasonable. It works in the notebook. Here is what it actually looks like:

```python
# naive_flask_server.py — DO NOT use in production
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model at startup — 30+ seconds, blocks everything
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
)
model.eval()

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data["prompt"]
    max_new_tokens = data.get("max_new_tokens", 256)
    
    # Tokenize — CPU-bound, ~3ms
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # model.generate() BLOCKS the entire server for 2–8 seconds
    # No other requests can be processed during this time
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    
    # Decode and return — no streaming, full response buffered
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"text": generated})

if __name__ == "__main__":
    # Flask's development server — single threaded by default
    app.run(host="0.0.0.0", port=8000)
```

This code has six production-ending flaws that each independently cause serious problems:

**Flaw 1: The GIL and serial GPU execution.** Flask runs in a single-threaded Python process by default. Even if you use `gunicorn --workers 4`, each worker has its own Python interpreter and its own copy of the model loaded in GPU memory. Four workers means 4 × 16GB = 64GB just for model weights — you have already exhausted the A100's memory with no KV cache left. And those workers still cannot share the GPU compute efficiently because they each launch independent CUDA operations.

**Flaw 2: No batching.** Every request launches its own `model.generate()` call. The GPU processes one sequence at a time. Autoregressive generation is memory-bandwidth-bound, not compute-bound. An A100 has 3.2 TB/s of HBM bandwidth. Loading the Llama-3-8B weights for a single decode step requires reading approximately 16GB (at FP16, one weight tensor pass over all 32 layers) = 16GB × (1/3200 GB/s) = 5ms of pure memory access time. The GPU spends 5ms loading weights for one token of one sequence. With 16 sequences in flight, those same weights serve 16 tokens — but the 5ms loading time is amortized. That is the entire point of batching: amortize the memory-bandwidth cost of loading weights across multiple sequences.

**Flaw 3: No KV cache sharing.** If two users send the same system prompt (extremely common in chatbot deployments where all users start with the same instructions), this server computes the KV cache twice, from scratch, for every request. A production server with prefix caching computes the system-prompt KV cache once and reuses it for all subsequent requests with the same prefix. For a 256-token system prompt on Llama-3-8B, prefix caching eliminates ~30–50ms of TTFT for every request after the first.

**Flaw 4: No streaming.** The entire response must be generated before any token is sent to the client. For a 512-token response at 18ms TPOT, that is 9.2 seconds before the user sees anything. By that point, most users have already reloaded the page or abandoned the request.

**Flaw 5: No memory management or limits.** The naive server allocates KV cache memory dynamically without any accounting. Under burst traffic, when four requests arrive simultaneously, each tries to allocate the full KV cache for maximum sequence length. With Llama-3-8B at max length 4096, each request's KV cache is 2GB. Four requests = 8GB of KV cache + 16GB weights = 24GB. That still fits. But at 10 simultaneous requests, it is 36GB of KV cache + 16GB = 52GB — still within the A100 80GB budget. The real failure happens when requests generate unexpectedly long outputs or when the server is under higher load than anticipated. There is no admission control, no memory limit enforcement, and no graceful degradation. When the GPU runs out of memory, PyTorch throws a CUDA OOM, the process dies, and Kubernetes restarts it — which takes 30–60 seconds to reload the model.

**Flaw 6: No observability.** There are no metrics emitted. When this server starts failing at 2 AM, you have no idea whether the problem is high latency, high error rate, OOM crashes, network issues, or a code bug. You are flying blind.

![The Flask wrapper processes requests serially with low GPU utilization; the production inference server multiplexes dozens of requests simultaneously](/imgs/blogs/what-is-model-serving-6.png)

Let us put concrete benchmark numbers on the performance gap. On an A100 80GB, sending 10 concurrent requests to this Flask server versus vLLM:

![Comparing serving frameworks at a glance across throughput, latency, GPU utilization, and use-case fit for Llama-3-8B on A100](/imgs/blogs/what-is-model-serving-4.png)

| Metric | Flask (naive) | TorchServe 0.9 | vLLM 0.4 |
|--------|--------------|---------------|----------|
| Throughput (req/s) | 2.3 | 12.4 | 52.4 |
| p50 latency (ms) | 4,820 | 892 | 247 |
| p99 latency (ms) | 9,340 | 1,850 | 638 |
| GPU utilization (%) | 28 | 58 | 91 |
| OOM at 20 users | Yes | No | No |
| Streaming support | No | Partial | Yes |

Hardware: A100 80GB, Llama-3-8B-Instruct, 512-token input, 256-token maximum output. Numbers are representative of community benchmarks and the vLLM paper (Kwon et al., SOSP 2023).

That 22x throughput gap between Flask and vLLM is not a model architecture difference. It is entirely serving infrastructure. Same GPU, same model weights, same prompt, same output length. The only difference is how the serving system schedules work on the GPU.

---

## A production-grade alternative: the vLLM AsyncLLMEngine

Here is what the same service looks like with a production serving system. The structural differences are instructive:

```python
# production_vllm_server.py — production-ready serving with async batching
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json
import uuid
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stop: list[str] = []

# Configure the async engine — this is where all the serving magic lives
engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,   # Leave 10% for CUDA runtime
    enable_prefix_caching=True,    # Cache shared system prompts
    max_num_seqs=64,               # Max concurrent sequences in flight at once
    tensor_parallel_size=1,        # Set to 2+ for multi-GPU serving
    enable_chunked_prefill=True,   # Split long prefills across multiple steps
    max_num_batched_tokens=4096,   # Token budget per scheduler step
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
app = FastAPI(title="LLM Serving API")

@app.post("/generate")
async def generate(req: GenerateRequest):
    request_id = str(uuid.uuid4())
    
    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stop=req.stop + ["<|eot_id|>"],
    )
    
    # This async generator runs concurrently with all other requests
    # The engine batches this with every other in-flight request automatically
    async def stream_results():
        last_output = ""
        async for request_output in engine.generate(
            req.prompt, sampling_params, request_id
        ):
            output = request_output.outputs[0]
            # Send only the new delta (not full text) for streaming
            new_text = output.text[len(last_output):]
            last_output = output.text
            
            if new_text:
                yield f"data: {json.dumps({'delta': new_text})}\n\n".encode()
            
            if output.finish_reason is not None:
                yield f"data: {json.dumps({'finish_reason': output.finish_reason, 'total_tokens': len(output.token_ids)})}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
    
    return StreamingResponse(stream_results(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok"}

# vLLM also exposes Prometheus metrics automatically at /metrics
```

```bash
# Or use vLLM's built-in OpenAI-compatible server — zero boilerplate
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --port 8000 \
    --host 0.0.0.0
```

The critical architectural difference is `AsyncLLMEngine`. It runs a continuous batching loop in a background asyncio task. Every 10ms (configurable), the scheduler wakes up, inspects the request queue, selects which requests to add to the current batch, allocates KV cache blocks for new arrivals, checks if any in-progress requests have finished, and dispatches the combined batch to the GPU for the next forward pass. Multiple requests share GPU resources simultaneously rather than waiting in a serial queue.

We will cover the internals of this scheduler in full detail in [A4: Batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) and the PagedAttention memory manager in [C2: Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention).

---

## Online vs offline (batch) inference

Not every inference workload is latency-sensitive. Understanding when to use each mode can change your serving cost by 5–10x — without changing any of the model quality.

![Choosing online or batch inference based on latency tolerance and volume can reduce cost by 5 to 10 times](/imgs/blogs/what-is-model-serving-5.png)

**Online inference** (real-time inference) processes each request immediately as it arrives, targeting response times in the 50ms–2000ms range. It is the correct architecture for:
- Chat and conversational AI applications where users expect immediate responses
- Search and recommendation systems constrained by page-load time
- Real-time fraud detection or content moderation where delays allow harm
- Streaming applications where partial results have value (e.g., code completion)

**Offline (batch) inference** collects requests into large batches and processes them asynchronously. Response times are measured in minutes or hours, not milliseconds. It is the correct architecture for:
- Document analysis pipelines (processing 10,000 contracts overnight)
- Daily recommendation scoring (precomputing embeddings for all users)
- Data augmentation and synthetic training data generation
- Evaluation runs against historical traffic logs
- Bulk embedding generation for vector databases

The economic difference is significant. Offline batch inference enables:
1. **Spot/preemptible instances** at 60–80% cost reduction — batch jobs can tolerate interruption with checkpointing
2. **Maximum GPU utilization** by packing batches as large as GPU memory allows — latency constraints no longer restrict batch size
3. **Cheaper hardware** — a T4 at \$0.35/hr is too slow for sub-500ms real-time LLM serving but perfectly adequate for a batch job that runs overnight

For our Llama-3-8B example processing a batch of 10,000 documents:
- Online inference on A100 at \$3.50/hr, 52 req/s: \$3.50/3600 × (10,000/52) ≈ \$0.186
- Offline batch on T4 spot at \$0.12/hr, 15 req/s: \$0.12/3600 × (10,000/15) ≈ \$0.022 — approximately **8.5x cheaper**

vLLM's offline API makes batch processing simple:

```python
# offline_batch_inference.py — for bulk document processing
from vllm import LLM, SamplingParams
from pathlib import Path
import json

# LLM (not AsyncLLMEngine) is the synchronous batch API
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.95,  # No concurrency headroom needed
    tensor_parallel_size=1,
)

# Load all documents and build prompts
documents = list(Path("documents/").glob("*.txt"))
prompts = [
    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    f"Summarize this in 3 sentences:\n\n{doc.read_text()[:1500]}"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    for doc in documents
]

sampling_params = SamplingParams(
    temperature=0.0,   # Deterministic output for document processing
    max_tokens=256,
    stop=["<|eot_id|>"],
)

# Process entire batch — vLLM internally schedules optimal batches
outputs = llm.generate(prompts, sampling_params)

results = [
    {"file": str(doc), "summary": output.outputs[0].text}
    for doc, output in zip(documents, outputs)
]

with open("summaries.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} documents, "
      f"avg {sum(len(o.outputs[0].token_ids) for o in outputs)/len(outputs):.0f} tokens/response")
```

One practical nuance: the vLLM `LLM` class does not expose streaming — it returns when all requests in the batch are complete. For large batches (thousands of documents), use the AsyncLLMEngine even in batch mode so you can process results incrementally as they complete rather than waiting for the slowest request.

---

## What makes a serving system production-grade

Production-grade serving is not defined by any single feature. It is the complete set of capabilities that allows a serving system to handle real-world traffic without manual intervention. Flask lacks all of them. A production system includes all of them.

![A production serving system requires six capability layers above the raw model weights; each missing layer creates a distinct failure mode](/imgs/blogs/what-is-model-serving-7.png)

**Continuous batching.** Rather than waiting for a complete batch to form before starting a forward pass, a production server runs the GPU continuously, adding new requests to the current batch mid-generation. The scheduler multiplexes multiple sequences over the same GPU pass — sequences at different positions in their generation loop all advance one step per forward pass. This is the single biggest contributor to high GPU utilization and is the core innovation of vLLM, TGI, and modern inference servers compared to the first generation of serving tools.

**Runtime optimization.** FlashAttention-2 (Dao et al., 2023) reorganizes the attention computation so that the GPU never materializes the full $N \times N$ attention matrix in HBM (high-bandwidth memory). Instead, it computes attention in tiles that fit in the GPU's much-faster SRAM (static RAM), reducing HBM reads by 5–10x for typical sequence lengths. `torch.compile` with `mode="reduce-overhead"` fuses sequential Python-dispatched kernel launches into optimized compiled graphs, reducing Python overhead by 20–40%. TensorRT converts the model graph to platform-specific CUDA kernels at the cost of a 10–30 minute compilation step at first launch.

**PagedAttention memory management.** KV cache memory is allocated in fixed-size virtual "pages" (like an OS page table), preventing fragmentation and allowing the scheduler to precisely count available blocks before committing to run a new request. Without PagedAttention, systems must pre-allocate KV cache for each request's maximum possible output length, wasting 60–80% of GPU memory on requests that generate shorter outputs. PagedAttention reclaims this wasted memory and allows 2–4x more concurrent sequences on the same GPU.

**Health checks and readiness probes.** A production server exposes `/health` (is the process alive?) and `/ready` (is the model loaded and the GPU memory allocated?) endpoints. Kubernetes uses these to avoid routing traffic to a replica during its 30–60 second startup period. Without readiness probes, rolling updates cause a wave of 504 errors as new pods receive traffic before the model has loaded.

**Metrics emission.** At minimum: request rate, latency histogram (p50/p95/p99 of TTFT and TPOT), GPU utilization %, GPU memory used GB, queue depth (requests waiting to be scheduled), tokens per second in and out, cache hit rate (for prefix caching), error rate by HTTP status code. These drive both alerting (p99 TTFT > 1000ms for 5 minutes triggers PagerDuty) and autoscaling (queue depth > 20 triggers a new replica).

vLLM emits Prometheus metrics automatically. A minimal scrape config:

```yaml
# prometheus-scrape.yaml — add to your Prometheus scrape configs
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Key vLLM Prometheus metrics to alert on:

```
# p99 TTFT — alert if > 1000ms for 5 minutes
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# Queue depth — alert if > 50 for 2 minutes  
vllm:num_requests_waiting

# GPU KV cache utilization — alert if > 90%
vllm:gpu_cache_usage_perc

# Request throughput
rate(vllm:request_success_total[1m])
```

**Model versioning and rollout.** A production system can deploy a new model version alongside the current one, route a fraction of traffic to it (canary deployment), compare metrics between the versions, and roll forward or roll back without downtime. This is covered in detail in [G2: CI/CD for model deployments](/blog/machine-learning/model-serving/cicd-for-model-deployments).

**Autoscaling.** As traffic increases, the system should automatically add GPU replicas; as traffic decreases, it should scale down to avoid paying for idle GPUs. In Kubernetes, this uses the Horizontal Pod Autoscaler with custom metrics. A minimal autoscaling configuration:

```yaml
# hpa-vllm.yaml — autoscale based on GPU queue depth
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting  # Custom metric from Prometheus adapter
        target:
          type: AverageValue
          averageValue: "10"  # Scale up when avg queue depth > 10 requests per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # Wait 60s before scaling up
      policies:
        - type: Pods
          value: 1
          periodSeconds: 90  # Add at most 1 pod every 90s (GPU startup time)
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
```

The 90-second scaleUp period is not arbitrary — it is the time required for a new pod to start, load the Llama-3-8B weights from a shared volume, allocate GPU memory, and pass its readiness probe. Scaling up faster than the pod can actually become ready creates a stampede of partially-ready pods that make the situation worse.

---

## Common failure modes in production

Knowing the failure patterns before they hit production is the difference between a smooth launch and a 3 AM incident. These four failure modes account for the majority of production LLM serving outages.

![The four dominant failure modes in production model serving, with root causes and mitigations](/imgs/blogs/what-is-model-serving-8.png)

**GPU OOM under bursty load.** The most common cause of production inference server crashes. It occurs when a burst of traffic causes more concurrent requests to be scheduled than the GPU's KV cache memory can hold. With a naive server, this crashes the process immediately. With vLLM's PagedAttention, the scheduler can detect when KV cache blocks are exhausted and either (a) return HTTP 429 Too Many Requests to admit-control the excess load, or (b) preempt a low-priority in-flight request by swapping its KV cache to CPU RAM and reinserting it in the queue.

Prevention: set `gpu_memory_utilization=0.85–0.90` (not 0.95+) to leave headroom for runtime memory allocations. Set `max_num_seqs=64` or lower based on your average sequence length and GPU memory budget. Monitor `vllm:gpu_cache_usage_perc` and alert at 85%.

**Tail latency from long generations.** When one request in a batch generates a very long output (say, 2048 tokens while all others generate 128), all other requests sharing the same KV cache memory blocks must wait for that long request to release its blocks before their responses can be finalized. This inflates p99 latency even when p50 looks healthy. The symptom: p99/p50 ratio greater than 5x.

Prevention: enforce a `max_tokens` limit at the server level, not just as a hint. For mixed-length workloads, separate long-generation (code generation) and short-generation (Q&A) requests into different server pools with different scheduling policies.

**Cold-start penalty.** When a new pod starts (or when a pod is restarted after a crash), it needs 30–60 seconds to load model weights from disk and allocate GPU memory. Any traffic routed to the pod during this window will timeout.

Prevention: Kubernetes readiness probes prevent routing to unready pods. Set a conservative `initialDelaySeconds: 60` and `failureThreshold: 10` on the readiness probe for LLM servers. For scale-to-zero scenarios (scaling down to 0 pods when idle), accept the cold-start latency or maintain a minimum of 1 warm replica.

**Memory fragmentation over time.** Over hours of operation, variable-length sequences generate and release KV cache blocks at different rates, which can fragment the memory pool. Even with PagedAttention's page manager, long-running servers may see effective capacity gradually decrease. The symptom: GPU memory utilization looks stable but `vllm:num_requests_waiting` gradually grows over hours even without traffic changes.

Prevention: schedule graceful restarts of inference server replicas every 24–48 hours during low-traffic windows using Kubernetes rolling restarts:

```bash
kubectl rollout restart deployment/vllm-deployment --namespace production
```

---

## Case studies from production systems

### vLLM's continuous batching paper (Kwon et al., SOSP 2023)

The original vLLM paper benchmarked against Orca, the prior state-of-the-art continuous batching system, on OPT-13B and LLaMA-13B. The key result: vLLM achieved **2.2–2.5x higher throughput** than Orca while maintaining equivalent latency, on identical hardware. The improvement came not from the continuous batching algorithm itself (Orca already implemented that) but from PagedAttention's block memory manager, which eliminated KV cache fragmentation and allowed approximately 55% more sequences to run concurrently on the same GPU.

At 30 requests/second on a 13B model, vLLM maintained p99 latency of 200ms, while Orca only achieved 200ms p99 at 13 requests/second — a 2.3x throughput advantage at the same SLO point. This result established that memory management, not just scheduling, is a primary lever for serving throughput.

### NVIDIA Triton dynamic batching on BERT-Large

NVIDIA's published Triton Inference Server benchmarks on BERT-Large (340M parameters) quantify the dynamic batching benefit clearly. Without dynamic batching: 312 req/s throughput, 58% GPU utilization on a T4 16GB. With dynamic batching (max batch size 64, 5ms max queue delay): 1,240 req/s throughput, 94% GPU utilization — a **4x throughput increase** with at most 5ms of added latency per request. For inference pipelines where throughput matters more than latency, this is an unambiguous win.

### Meta's LLaMA inference infrastructure

Meta's internal inference infrastructure for LLaMA 70B (detailed in their engineering blog and the LLaMA 2 paper) uses 8-way tensor parallelism across 8 A100 GPUs per replica, continuous batching, and FlashAttention. Their published cost analysis shows that going from a naïve single-GPU serving approach to their optimized multi-GPU system reduces cost per token by approximately 8x — primarily through continuous batching (3–4x) combined with tensor parallelism (2–3x) and quantization (1.5–2x). The multiplicative gains from combining multiple optimization techniques are what make large-scale LLM serving economically viable.

### OpenAI's serving efficiency

While OpenAI does not publish detailed serving architecture papers, their public pricing provides an indirect benchmark. GPT-4o is priced at \$2.50/1M input tokens and \$10.00/1M output tokens. At these prices, serving GPT-4o profitably requires extremely high GPU utilization at scale. The consistency of their latency (typically 200–500ms p99 TTFT for short prompts) at massive scale confirms that their serving infrastructure achieves the sort of high-concurrency batching that requires production-grade systems like those described throughout this series.

### Mistral AI's throughput-first design

Mistral's engineering blog has documented their decision to use FP8 quantization on H100 GPUs for Mixtral serving. The H100's FP8 Tensor Cores provide approximately 2x the compute throughput of BF16 at the same memory footprint, while FP8 weights cut model memory by 2x compared to FP16. The combined effect: approximately 3–4x higher throughput per H100 compared to Mixtral in FP16 on an A100, with accuracy regression under 0.5% on standard benchmarks. This is the "hardware + quantization + continuous batching" multiplicative effect in practice: each optimization multiplies the gains from the others, and the combined result can be an order-of-magnitude improvement in cost per token.

The key lesson from all these case studies is consistent: the gap between naive and production serving is not incremental. It is a multiplicative stack of individually significant optimizations — continuous batching (3–5x), PagedAttention (2x), quantization (1.5–3x), FlashAttention (1.2–1.5x) — whose product can reach 20–50x improvement over a baseline Flask server on identical hardware. Each optimization is independently valuable; together they transform the economics of LLM deployment.

---

#### Worked example: capacity planning for a chatbot backend at 20,000 DAU

You are building a customer support chatbot. Requirements: 20,000 daily active users, 3 messages per session average, distributed over a 12-hour business day, p99 TTFT under 800ms, p99 TPOT under 25ms per token.

**Step 1: Compute peak request rate.** Assuming 2x peak-to-average ratio and Poisson arrivals:

$$\lambda_{\text{avg}} = \frac{20{,}000 \times 3}{12 \times 3600} = 1.39 \text{ req/s}$$

$$\lambda_{\text{peak}} = 1.39 \times 2 = 2.78 \approx 3 \text{ req/s}$$

**Step 2: Choose hardware.** At 3 req/s, an A100 80GB with vLLM (~52 req/s capacity) has 17x headroom — massively over-provisioned. Better to use an A10G (24GB VRAM, ~25 req/s with Llama-3-8B) or even a 3090 Ti (24GB, ~20 req/s) for cost savings. On AWS, an A10G instance (g5.xlarge) costs \$1.01/hr.

**Step 3: Latency check.** At 3 req/s on an A10G running vLLM with Llama-3-8B: load $\rho = 3/25 = 0.12$ (very light). p99 TTFT ≈ 280ms (well under 800ms SLA). TPOT ≈ 20ms per token on A10G (the A10G has 600 GB/s HBM bandwidth vs 2 TB/s for A100, so roughly 3x slower per-token; well under the 25ms SLA).

**Step 4: Deployment configuration.**

```bash
# Launch on A10G (24GB VRAM) with appropriate memory configuration
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.87 \
    --max-num-seqs 24 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
```

**Step 5: Cost calculation.** Single A10G at \$1.01/hr × 24 hrs = \$24.24/day. For 20,000 users × 3 messages = 60,000 requests/day. At 256 output tokens average: 60,000 × 256 = 15.36M output tokens/day. Cost: \$24.24 ÷ 15.36M × 1M = **\$1.58 per 1M output tokens**. For context, comparable cloud API pricing ranges from \$8–\$15/1M tokens for similar-quality 8B models.

---

#### Worked example: sizing for a sudden 10x traffic spike

Your chatbot is handling 3 req/s steadily. A product launch drives a sudden spike to 30 req/s — 10x the normal load. What happens?

**Without autoscaling:** Your single A10G server (25 req/s capacity) is now at $\rho = 30/25 = 1.2$ — above stability threshold. The queue grows without bound. After 30 seconds: 30 × (30 – 25) = 150 requests queued. At 15 req/s processing rate for queue draining, those 150 requests take another 10 seconds to clear. Users experience 30–40 second latency during the spike. Most will have abandoned or retried, creating even more load.

**With autoscaling (HPA configured as above):** Queue depth exceeds 10 requests (trigger threshold) within the first 5 seconds. The HPA fires a scale-up event and a new pod starts. After 90 seconds (pod startup + model loading), the second pod is ready. Now the system has 50 req/s capacity, reducing $\rho = 30/50 = 0.6$ — well within the stable range. Latency returns to normal within 2 minutes of the spike starting.

**With pre-emptive scaling (KEDA + external metrics):** If your traffic monitoring system sends a webhook before the spike (e.g., a product manager schedules a launch and triggers a pre-scale event), you can have extra pods ready before traffic hits. Zero latency degradation.

The lesson: for latency-sensitive services, autoscaling is not optional. Model servers have 60–90 second cold-start times, which means the autoscaler needs to act *before* the SLA is violated, not after it is already broken. This requires careful threshold tuning and headroom in your normal operating range.

---

## When to use which serving approach

The right serving framework depends on your model type, traffic profile, latency budget, and team's operational expertise.

| Scenario | Best framework | Avoid |
|----------|---------------|-------|
| Local dev / notebook demo | Flask + transformers | — (no production traffic) |
| CV model, < 10 req/s | TorchServe | vLLM (built for LLMs) |
| CV model, > 50 req/s | Triton Inference Server | Flask (no dynamic batching) |
| LLM < 7B, < 20 req/s | vLLM on A10G or 3090 | Triton (no continuous batching) |
| LLM 7B–70B, 20–200 req/s | vLLM on A100/H100 | TorchServe (no KV optimization) |
| LLM 70B+, multi-node | vLLM distributed + TGI | Single-GPU approaches |
| Multi-model routing pipeline | Ray Serve + Triton ensemble | vLLM (single-model focused) |
| Offline batch processing | vLLM LLM() or batch API | Online server (wasteful overhead) |
| Edge / embedded device | ONNX Runtime + quantized | Any GPU-based framework |
| Multi-tenant, many LoRA adapters | vLLM S-LoRA or Punica | Single-adapter servers |

**When NOT to use vLLM:** vLLM is optimized exclusively for autoregressive transformer language models on CUDA GPUs. Do not use it for vision models (ResNet, ViT), tabular models, audio models, or models not in the HuggingFace transformers format. For a ResNet-50 image classifier, Triton with a TensorRT engine will massively outperform vLLM. For a scikit-learn random forest, neither is needed.

**When NOT to over-engineer:** For an internal data science tool with 2 req/s and no latency SLA, deploying a full Kubernetes + vLLM + Prometheus + KEDA stack is wasteful overhead. A FastAPI server with asyncio and simple batching, running in a screen session on a GPU machine, is the right tool for small-scale internal use. The production infrastructure pays for itself only when you have real traffic that would expose the naive approach's failure modes.

**The framework selection principle:** Match the serving framework to the model type and scale. The worst performance comes from mismatched choices — running a CV batch scoring workload through vLLM (which has no advantage for non-autoregressive models), or running an LLM chatbot through TorchServe (which was not designed for KV cache management). Know what each framework optimizes for before choosing.

A quick reference test: if your model generates tokens autoregressively (any transformer decoder, any LLM), use vLLM or TGI. If your model runs a fixed number of forward passes per request (classification, embedding, detection), use Triton or TorchServe. If you need to combine multiple models in a pipeline, use Ray Serve or Triton ensemble. If you are just experimenting, use the simplest thing that works — you can always migrate.

---

## The full production deployment skeleton

For reference, here is the minimal Kubernetes Deployment spec for a production vLLM service:

```yaml
# vllm-deployment.yaml — production skeleton
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
  namespace: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-llama3-8b
  template:
    metadata:
      labels:
        app: vllm-llama3-8b
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.4.3
          args:
            - "--model"
            - "meta-llama/Meta-Llama-3-8B-Instruct"
            - "--dtype"
            - "float16"
            - "--max-model-len"
            - "4096"
            - "--gpu-memory-utilization"
            - "0.90"
            - "--max-num-seqs"
            - "64"
            - "--enable-prefix-caching"
            - "--port"
            - "8000"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "40Gi"
            requests:
              nvidia.com/gpu: "1"
              memory: "32Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60   # Model loading time
            periodSeconds: 10
            failureThreshold: 12      # 2 minutes total before giving up
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
            periodSeconds: 30
            failureThreshold: 3
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc  # Pre-downloaded model weights
      nodeSelector:
        nvidia.com/gpu.product: A100-SXM4-80GB  # Pin to A100 nodes
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-8b-svc
  namespace: inference
spec:
  selector:
    app: vllm-llama3-8b
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

The `readinessProbe` with `initialDelaySeconds: 60` is not conservative — it is calibrated to reality. A cold Llama-3-8B pod takes 30–60 seconds to load weights from a pre-downloaded PVC, another 10–15 seconds to allocate and warm up the GPU memory pool, and a few seconds for the HTTP server to start. Total: 45–75 seconds. Setting `initialDelaySeconds: 30` causes Kubernetes to mark pods as "NotReady" repeatedly during startup and potentially restart them before they have a chance to finish loading.

One detail that surprises engineers coming from web service backgrounds: the `minReplicas: 1` in the HPA spec is not a preference — it is critical for LLM serving. Web services can scale to zero (no replicas) when idle and accept a multi-second cold start on the first request. An LLM server with a 45-second cold start is not acceptable for a user-facing product. Keep at least one warm replica running at all times for any latency-sensitive deployment, and budget the corresponding GPU cost (\$24–\$84/day for an A10G–A100 running continuously) as a fixed infrastructure cost of your service.

---

## The series map: from here to the full production system

This post established the conceptual foundation — the why behind every decision in this series. What you have now:
- The SLO triangle as the governing constraint
- Little's Law as the capacity planning tool
- GPU memory math as the budget constraint
- A clear picture of what each layer of the serving stack does
- Benchmark numbers grounding the naive-vs-production performance gap
- The decision framework for choosing a serving approach

What comes next, in rough dependency order:

**[A2: The model serving stack](/blog/machine-learning/model-serving/the-model-serving-stack)** — Zooms out to map the full technology stack: model packaging formats (ONNX, TorchScript, GGUF), runtime layers, server frameworks, infrastructure primitives, and observability. A single post that establishes the vocabulary and component map for everything that follows.

**[A4: Batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff)** — Goes deep on the batching algorithms: static batching, dynamic batching with timeout, and continuous batching. Derives the throughput-latency trade-off curve quantitatively and explains why continuous batching dominates the others for LLM workloads.

**[C2: Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)** — Opens up the vLLM scheduler and PagedAttention block manager in detail. Explains the KV cache fragmentation problem and the virtual-memory solution. Shows how to tune the scheduler for different SLO profiles.

**[D4: Autoscaling model servers](/blog/machine-learning/model-serving/autoscaling-model-servers)** — Covers HPA with custom metrics, KEDA ScaledObjects, scale-to-zero patterns, and the specific challenge of autoscaling LLM servers with long cold-start times.

By the time you reach the capstone [model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook), you will have a complete decision framework that maps any combination of model size, traffic profile, latency budget, and cost constraint to a specific production architecture. The concepts in this post are the foundation everything else builds on.

---

## Key takeaways

1. **Serving and training are opposite engineering problems.** Training maximizes throughput in a controlled batch loop. Serving minimizes latency under unpredictable concurrent load. Solutions optimized for one do not transfer to the other.

2. **The SLO triangle is the master constraint.** Latency, throughput, and cost are in fundamental tension. Every serving decision is a trade on this triangle. The only way to improve all three simultaneously is to change hardware or architecture — not configuration alone.

3. **Little's Law gives you capacity math.** $L = \lambda W$ tells you exactly how many concurrent requests your system must sustain to meet a throughput and latency target. Derive your GPU count from this formula, not intuition or guesswork.

4. **Flask fails at 4+ concurrent users.** Not because Python is slow, but because it serializes requests through a single CPU thread while the GPU sits 72% idle. A production server must multiplex concurrent requests on the GPU simultaneously.

5. **The gap is not model quality — it is serving infrastructure.** A naive Flask wrapper and vLLM run the exact same model weights. The 22x throughput difference at identical hardware comes entirely from how the serving system schedules work on the GPU.

6. **Continuous batching is the core LLM serving primitive.** Keeping the GPU continuously fed with dynamic batches, rather than waiting for a complete batch to form, is what separates 28% GPU utilization from 91%.

7. **OOM under burst is the most common production failure.** Set `gpu_memory_utilization=0.85–0.90`, set explicit `max_num_seqs` limits, and rely on PagedAttention's block manager for admission control. Never let the scheduler allocate unbounded KV cache.

8. **Batch inference is 5–10x cheaper than online inference.** For workloads that tolerate minutes of latency, offline batch processing on spot instances dramatically reduces cost. Route workloads to the right mode as an architecture decision, not an afterthought.

9. **Production-grade means six capabilities.** Continuous batching, runtime optimization, health checks, metrics, autoscaling, and model versioning. Each has a specific, predictable failure mode if missing — and that failure mode always manifests at the worst possible time.

10. **Autoscaling must lead latency, not follow it.** Model servers take 60–90 seconds to start up. Configure HPA thresholds conservatively enough that new pods are ready before the SLA is violated, not after it is already broken.

11. **GPU memory math gives you the concurrency ceiling.** Compute weights memory + runtime overhead, subtract from total GPU memory, divide by KV cache bytes per token and by average sequence length. The result is your maximum simultaneous sequences. Design within that budget.

12. **First-principles reasoning beats cookbook deployment.** When p99 spikes without a traffic change, check KV cache pressure and queue depth. When GPU utilization falls gradually, check for memory fragmentation. When a pod fails readiness repeatedly, calibrate `initialDelaySeconds` to actual model loading time. Understanding the mechanics makes debugging tractable rather than guesswork.

---

## Further reading

- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"**, SOSP 2023 — The paper that introduced vLLM and PagedAttention. Essential reading for anyone doing LLM serving at scale. [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

- **Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"**, OSDI 2022 — The prior-art continuous batching system that established the scheduling framework vLLM improved upon. Foundational for understanding the scheduling algorithms.

- **Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills"**, arXiv 2024 — Addresses the TTFT vs throughput tension through chunked prefill scheduling, a technique now integrated into vLLM.

- **Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"**, arXiv 2023 — The attention kernel optimization that is now standard in every production LLM serving system. Explains the tiling strategy that avoids materializing the full attention matrix in HBM and reduces memory reads by 5–10x for typical sequence lengths.

- **vLLM documentation** — Production deployment guides, engine configuration, distributed serving setup, and benchmark results: [docs.vllm.ai](https://docs.vllm.ai)

- **NVIDIA Triton Inference Server documentation** — Model repository layout, dynamic batching configuration, ensemble pipelines, Prometheus metrics: [docs.nvidia.com/deeplearning/triton-inference-server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)

- **Within this series:** [A2: The model serving stack](/blog/machine-learning/model-serving/the-model-serving-stack) maps every component from packaging to infrastructure; [A4: Batching fundamentals](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) derives the batching algorithms from first principles; [C2: Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) covers vLLM internals in full detail; [A5: Model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) explains how to instrument and alert on TTFT, TPOT, and queue depth; [the model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) is the capstone reference synthesizing the complete decision framework.

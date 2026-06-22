---
title: "Model serving SLAs and metrics: TTFT, TPOT, p99, and everything in between"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master every metric that matters in model serving — how to compute TTFT, TPOT, MFU, and queue depth, how to translate user-experience requirements into SLO targets, and how to wire Prometheus alerting so you find out about a p99 breach before your users do."
tags:
  [
    "model-serving",
    "inference",
    "sla",
    "metrics",
    "llm-serving",
    "prometheus",
    "gpu-utilization",
    "observability",
    "latency",
    "throughput",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/model-serving-slas-and-metrics-1.png"
---

At 2:47 AM a pager fires. P99 latency for the chatbot is at 4.2 seconds. The SLA says 2 seconds. Support tickets are already rolling in. You ssh into the serving cluster, open Grafana, and immediately realize the problem: you have no idea which dashboard panel is telling you what is actually wrong. GPU utilization shows 71 percent — not alarming. Request error rate is 0.3 percent — below threshold. Mean latency is 480 milliseconds — completely fine. But p99 is at 4.2 seconds, which means one in every hundred users is waiting four times longer than promised.

The post-mortem finding is always the same in situations like this: the team instrumented mean latency and GPU percent utilization, declared victory, and shipped. Nobody computed TTFT or TPOT separately. Nobody set up a p99 histogram. Nobody tracked queue depth. Mean latency was 480 milliseconds because the 99th percentile requests were hiding in the average, keeping it artificially low while a small fraction of users experienced a completely broken product.

This post is the definitive reference for every metric that matters in model serving. By the end you will know what TTFT and TPOT measure and why they are distinct, how to derive an SLA budget from user experience requirements, how to compute MFU and why it matters more than raw GPU percent utilization, how to configure Prometheus to scrape vLLM and Triton, how to write Grafana queries for p99 histograms, how to set alerting rules that fire before the user does, and how to size a serving fleet using Little's Law. The running example throughout this post is an LLM chatbot with a stated SLA of TTFT < 500ms, TPOT < 50ms/token, all at p99. Every formula and recommendation traces back to that concrete target.

![Serving metrics taxonomy: every observable maps to latency, throughput, GPU efficiency, cost, or error signals](/imgs/blogs/model-serving-slas-and-metrics-1.png)

## 1. The measurement philosophy behind serving metrics

Before covering any specific metric, the philosophy matters: you are not measuring the machine; you are measuring the user experience, and the machine is an indirect proxy for that. This distinction drives every metric choice.

The user's experience has three axes. **Perceived responsiveness** is how quickly they see the first word appear — this maps to TTFT (Time to First Token). **Streaming smoothness** is whether the text arrives at a reading pace or in frustrating bursts — this maps to TPOT (Time Per Output Token). **Reliability** is whether the request succeeds at all — this maps to error rate and timeout rate. Everything else in the metrics taxonomy — GPU utilization, MFU, bandwidth utilization, queue depth — is either a cause of poor user experience or a capacity planning tool. Understanding which category a metric falls into prevents the common mistake of over-optimizing a cause metric without measuring the effect.

The SLO triangle introduced in [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) captures the three-way tension: latency versus throughput versus cost. Every metric in this post belongs to one of those three sides. TTFT and TPOT live on the latency side. RPS and TPS live on the throughput side. MFU, bandwidth utilization, and cost-per-token live on the cost side. Improving one side typically trades off against another, and the SLA is the contract that fixes how far you are allowed to trade.

## 2. TTFT and TPOT: the two latency dimensions of LLM serving

Classical ML serving — a ResNet doing image classification, a BERT model doing sentiment analysis — has a single latency number: end-to-end request latency. You send a request, you wait, you get a response. But LLMs are autoregressive: they generate one token at a time, and in production they stream those tokens to the client as they are produced. This creates two fundamentally distinct latency dimensions that are mechanically independent and require different optimization approaches.

### TTFT: Time to First Token

TTFT is the time from when the request arrives at the model server to when the first output token is produced and transmitted to the client. It determines the user's perception of responsiveness — how long they stare at a blinking cursor before anything appears. For interactive chat applications, a TTFT above roughly 500–800 milliseconds begins to feel laggy to users, and above 1.5 seconds it starts to feel broken.

Mechanically, TTFT is dominated by **prefill compute**. During prefill, the model processes the entire input prompt in a single forward pass and constructs the KV cache for every input token. For a Llama-3 70B model processing a 2,048-token prompt on 4× A100 GPUs with tensor parallelism, prefill takes approximately 80–120 milliseconds depending on batch occupancy. There is also a small but real queue wait time if the server is already busy processing other requests. The full TTFT breakdown is:

$$\text{TTFT} = T_{\text{queue wait}} + T_{\text{prefill compute}} + T_{\text{network RTT/2}}$$

where the network term is usually a few milliseconds for co-located clients and can grow to 30–80 milliseconds for geographically distributed users. The queue wait component is zero under light load but grows rapidly under saturation — more on this in the queue depth section.

TTFT is a **compute-bound** quantity during prefill. The computation scales roughly as $O(n_{\text{input}}^2)$ in attention due to the quadratic attention kernel (though FlashAttention-2 reduces memory pressure significantly), and linearly in the number of transformer layers and model parameters. Optimization levers for TTFT include chunked prefill (vLLM's `--enable-chunked-prefill`), prefill/decode disaggregation (a separate prefill fleet that runs at higher GPU utilization), and speculative decoding (which doesn't help TTFT but helps TPOT).

### TPOT: Time Per Output Token

TPOT is the average time between successive output tokens once the first token has been produced. It determines the "streaming feel" — whether the text appears at a natural reading pace (roughly 4–6 tokens per second, equivalent to 200–250ms TPOT) or races ahead faster than the user can read (below 80ms TPOT at 12+ tokens/second). Most chatbot SLAs target TPOT in the range of 30–80ms/token, which yields a comfortable 12–33 tokens per second streaming rate.

Mechanically, TPOT is dominated by **memory bandwidth** during decode, not compute. Each decode step generates exactly one new token. The KV cache for all previous tokens must be read from HBM (High Bandwidth Memory) on every step, and the model weights must be read once per decode step to perform the forward pass. For a Llama-3 70B model, the weights alone consume roughly 140 GB in BF16, which must traverse the HBM bus at every decode step. An H100 SXM5 has 3.35 TB/s HBM bandwidth; reading 140 GB of weights takes approximately $140\,\text{GB} / 3350\,\text{GB/s} \approx 42\,\text{ms}$ at full bandwidth utilization. That 42ms is effectively the theoretical floor for TPOT on a single H100 with the Llama-3 70B model, ignoring KV cache reads.

The key insight is that during decode the GPU arithmetic units are largely idle — the bottleneck is getting bytes from memory to the compute units, not performing the computation itself. This is why the naive nvidia-smi GPU utilization metric can show a deceiving 40–50% while TPOT is already near its theoretical minimum.

### The E2E latency formula

End-to-end latency for a request that generates $N$ output tokens is:

$$\text{E2E Latency} = \text{TTFT} + (N - 1) \times \text{TPOT}$$

The $N-1$ factor rather than $N$ arises because TTFT already accounts for the generation of the first token. For a 200-token response at TTFT = 100ms and TPOT = 10ms:

$$\text{E2E} = 100\,\text{ms} + (200 - 1) \times 10\,\text{ms} = 100 + 1990 = 2090\,\text{ms}$$

This formula has a critical design consequence: for long responses (many output tokens), TPOT dominates the E2E latency and TTFT becomes almost irrelevant. For short responses (fewer than 20 output tokens), TTFT can be 80–90% of the total latency. A chatbot with typical responses of 50–300 tokens operates in the middle regime where both matter.

![LLM request latency anatomy: queue wait plus prefill yields TTFT, then decode loop generates subsequent tokens at TPOT each](/imgs/blogs/model-serving-slas-and-metrics-2.png)

## 3. Throughput metrics: RPS, QPS, and TPS

Latency tells you how fast a single request is processed. Throughput tells you how many requests the system can process simultaneously. The two are connected by Little's Law, which is the foundational capacity planning tool for serving systems.

### RPS and QPS

**RPS (Requests Per Second)** and **QPS (Queries Per Second)** are used interchangeably in practice, though some teams reserve QPS for read-only queries and RPS for all requests. For model serving, they measure the rate at which requests reach the server (arrival rate) or the rate at which requests are completed (departure rate). In a stable system these are equal; when the system is saturated, the departure rate is capped and queue depth grows.

For a dedicated LLM serving deployment, a single A100-80GB running Llama-3 8B with vLLM and continuous batching can typically sustain 40–80 RPS at p99 TTFT < 500ms depending on prompt and output length. A single H100 running Llama-3 70B with 4-way tensor parallelism sustains roughly 8–15 RPS at the same latency target. These numbers are highly workload-dependent — the next section explains why.

### TPS: Tokens Per Second

**TPS (Tokens Per Second)** measures the aggregate output token generation rate across all concurrent requests. It is the more natural throughput metric for LLM billing and cost analysis because LLM APIs typically charge per token rather than per request. TPS = RPS × (average output length).

For capacity planning, TPS is often more stable than RPS because it normalizes for output length variation. A mix of short (20-token) and long (500-token) responses will show highly variable RPS, but TPS will be more consistent because the server is always busy generating tokens regardless of which responses they belong to.

The peak TPS of a system is bounded below by hardware: for a memory-bandwidth-bound decode, the maximum TPS equals the number of bytes that can be moved through HBM per second divided by the bytes per decode step. For the Llama-3 70B model in BF16 on a single H100:

$$\text{Max TPS} = \frac{\text{HBM bandwidth}}{\text{bytes per decode step}} = \frac{3.35\,\text{TB/s}}{140\,\text{GB}} \approx 24\,\text{tok/s per request}$$

With continuous batching across 16 concurrent requests, the denominator increases by the KV cache fraction but the numerator stays fixed, yielding higher aggregate TPS.

### Little's Law for capacity planning

**Little's Law** is the foundation of queuing theory and the single most important formula for model serving capacity planning:

$$L = \lambda \times W$$

where $L$ is the average number of requests in the system (including those being processed), $\lambda$ is the arrival rate (RPS), and $W$ is the average time a request spends in the system (E2E latency in seconds). Rearranging:

$$\lambda_{\max} = \frac{L_{\max}}{W}$$

where $L_{\max}$ is the maximum number of requests the system can hold concurrently (bounded by GPU memory for the KV cache) and $W$ is your target E2E latency. If your H100 server can hold at most 32 concurrent requests in the KV cache (determined by memory capacity), and your target E2E latency is 2 seconds, then the maximum sustainable arrival rate is $32 / 2 = 16\,\text{req/s}$. If actual traffic exceeds 16 req/s, queue depth grows and p99 latency explodes. This calculation determines how many server replicas you need before deploying.

#### Worked example: Fleet sizing with Little's Law

You are launching a chatbot. Requirements: TTFT < 500ms, TPOT < 50ms/token, p99. Typical response length: 150 tokens. Traffic peak: 100 concurrent users each sending one request every 3 seconds.

Step 1 — compute arrival rate: $\lambda = 100 / 3 \approx 33\,\text{req/s}$.

Step 2 — compute target E2E latency: $W = 0.5 + (150-1) \times 0.05 = 0.5 + 7.45 = 7.95\,\text{s}$.

Step 3 — compute required $L$ per Little's Law: $L = \lambda \times W = 33 \times 7.95 \approx 263$.

Step 4 — compute KV cache capacity per server. For Llama-3 8B on an A100-80GB with BF16 KV cache, each token's KV cache consumes approximately 1 MB (32 layers × 2 heads × 2 (K+V) × 8192 head_dim × 2 bytes). For a 150-token response that is 150 MB per request, and the A100 has roughly 45 GB remaining after loading the model weights (35 GB), so the KV cache can hold about $45\,000\,\text{MB} / 150\,\text{MB} = 300$ concurrent requests per GPU. Each A100 can hold all 263 required concurrent requests.

Step 5 — verify throughput: can one A100 sustain 33 req/s? Each request processes 150 output tokens at roughly 40ms TPOT = 6 seconds of decode time. With 263 concurrent requests, you need $263 \times 6 = 1578$ GPU-seconds of decode per second — that is 1578 GPUs. Clearly one GPU cannot handle this. You need at least $\lceil 33 \times 7.95 / \text{concurrent capacity}\rceil$ GPUs. The practical calculation: one A100 can generate roughly 1500 tokens/second with continuous batching on Llama-3 8B; you need $33 \times 150 = 4950$ tokens/second peak, so you need at least 4 A100s. Use 6 A100s for a 1.5× headroom buffer.

## 4. GPU efficiency metrics: utilization, bandwidth, and MFU

GPU metrics are where the most confusion lives. Three distinct efficiency measures are commonly reported, and they tell you completely different things.

### GPU utilization percent

**GPU utilization percent** is what `nvidia-smi` reports and what most dashboards show by default. It measures the fraction of time that at least one streaming multiprocessor (SM) is executing a kernel. A value of 70% means the GPU spent 70% of the measurement window doing *some* computation.

The critical subtlety: GPU utilization counts any kernel execution, including memory copy kernels that are doing no useful computation. During memory-bandwidth-bound LLM decode, the GPU is continuously executing HBM read operations — which counts as 100% utilization even though the arithmetic units (tensor cores) are nearly idle. Conversely, a GPU doing sparse matrix operations may show 45% utilization while performing at 90% of its theoretical FLOPs capacity. **GPU utilization percent is an unreliable proxy for both performance and efficiency during LLM decode.**

The metric is still useful for one thing: detecting complete idleness. If GPU utilization drops to 0–5% under sustained traffic, you have a serving software bug (the model is not being dispatched to the GPU) or a severe queue starvation problem.

### Memory bandwidth utilization

**Memory bandwidth utilization** is the fraction of the GPU's HBM bandwidth that is being consumed. For decode-phase LLM serving, this is the real bottleneck metric. You can compute it as:

$$\text{BW util} = \frac{\text{bytes moved per second}}{\text{peak HBM bandwidth}}$$

For an H100 with 3.35 TB/s peak HBM bandwidth, you want decode bandwidth utilization to be above 75% under load. If it is below 50% during decode, there is a memory access inefficiency — likely due to fragmented KV cache access patterns, suboptimal batch sizes, or overhead from Python dispatch. You can measure bytes moved per second using DCGM (Data Center GPU Manager) via the `DCGM_FI_PROF_DRAM_BW_PERC` metric.

### MFU: Model FLOPs Utilization

**MFU (Model FLOPs Utilization)** is the most rigorous efficiency metric for transformer inference. It measures the fraction of the GPU's theoretical peak FLOPs that are actually spent on model computation:

$$\text{MFU} = \frac{\text{actual model FLOPs per second}}{\text{theoretical peak FLOPs per second}}$$

Computing MFU requires knowing the actual FLOPs for your model architecture. For a transformer with $L$ layers, hidden dimension $d$, sequence length $s$, and batch size $b$, the approximate FLOPs per forward pass are:

$$\text{FLOPs} \approx 2 \times b \times s \times L \times (12 \times d^2 + 2 \times d \times s)$$

The $12d^2$ term covers the QKV projection, output projection, and FFN layers. The $2ds$ term covers attention scores. For Llama-3 70B ($L=80$, $d=8192$), processing a batch of 16 requests each with 512 tokens, the FLOPs are approximately:

$$\text{FLOPs} = 2 \times 16 \times 512 \times 80 \times (12 \times 8192^2 + 2 \times 8192 \times 512) \approx 6.5 \times 10^{15}$$

An H100 SXM5 peaks at approximately $3.9 \times 10^{15}$ BF16 FLOPs/second with tensor cores. A well-optimized vLLM deployment running Llama-3 70B achieves roughly 35–45% MFU during prefill-heavy workloads; during pure decode it drops to 8–15% because the workload is memory-bandwidth-bound, not compute-bound. A 40% MFU during mixed prefill/decode is excellent for LLMs and is the rough target for well-tuned production systems.

MFU is the single best metric for comparing serving frameworks against each other on the same hardware. If vLLM achieves 38% MFU and TGI achieves 28% MFU on the same workload and hardware, vLLM is extracting 36% more useful work from the same GPU, which translates directly into lower cost per token.

## 5. Percentile latency: why p99 and p999 matter more than mean

The relationship between mean latency and percentile latency is one of the most important concepts in serving reliability engineering. Mean latency hides the tail.

### The long tail problem

Consider a system where 99% of requests complete in 200ms and 1% take 8000ms. The mean is approximately $0.99 \times 200 + 0.01 \times 8000 = 278\,\text{ms}$. If you are monitoring only the mean, the system looks completely healthy at 278ms against a 500ms target, but one in every hundred users is waiting 8 seconds for a response and is almost certainly abandoning the session or filing a support ticket.

This pattern is extremely common in LLM serving and has two primary causes. The first is **output length variance**: most requests generate short responses (20–50 tokens) but occasionally a request generates a very long response (1000+ tokens). The long-response requests take many times longer than average, creating a heavy right tail. The second is **KV cache pressure under load**: when the server is near capacity, requests that arrive during a peak experience longer queue waits, which can push their latency to multiples of the baseline. This creates a bimodal distribution — most requests are fast, a few are very slow.

### SLA design: mean vs percentile

The standard practice in production serving is to commit SLAs on percentile latency:

- **p50 (median)**: useful for understanding typical user experience
- **p90**: the threshold where "most users" have a good experience
- **p99**: the industry-standard SLA commitment level; 1 in 100 requests exceeds this
- **p999**: the tail; 1 in 1000 requests exceeds this; important for high-volume APIs where absolute counts matter

For a chatbot handling 50 req/s, a p999 latency of 5 seconds means roughly 0.05 requests per second — about 3 users per minute — experience a 5-second response time. At 500 req/s, that is 30 users per minute getting a terrible experience. The appropriate percentile for SLA commitment scales with request volume.

**Mean latency is still useful for capacity planning.** The mean E2E latency is what appears in the $W$ term of Little's Law when computing required fleet capacity. Using the p99 as $W$ would over-provision capacity by roughly 3–10×.

![Mean vs p99 comparison: a mean of 350ms can hide a p99 of 1800ms, delivering 5x worse experience to 1 percent of users](/imgs/blogs/model-serving-slas-and-metrics-3.png)

### Histogram configuration for accurate percentiles

Prometheus histograms are the standard tool for collecting percentile latency in model serving. The critical detail is **bucket configuration**: the default Prometheus histogram buckets (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10 seconds) are calibrated for web API latencies in the tens-of-milliseconds range and are completely wrong for LLM serving. You need fine-grained buckets in the 0–500ms range for TTFT and coarser buckets for E2E latency on long responses.

For TTFT, a good bucket configuration in seconds is: `[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]`. For TPOT in seconds per token: `[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5]`. Using default buckets will produce imprecise percentile estimates because `histogram_quantile` interpolates linearly within buckets.

## 6. Error rates, queue depth, and saturation signals

Latency and throughput metrics tell you how well the system is performing when it is working. Error rates and queue depth tell you when it is about to stop working.

### Error rate taxonomy

**HTTP 5xx rate** is the most obvious error signal. For model serving, 5xx errors have three primary causes: (1) CUDA out-of-memory (OOM), which occurs when the KV cache fills up and the server cannot accommodate a new request; (2) request timeout, which occurs when a request exceeds the configured maximum generation time; and (3) downstream dependency failure (database unavailable, vector store timeout).

It is important to distinguish these in your alerting because the remediation is completely different. OOM errors require immediate memory pressure relief — either increasing server capacity, reducing `max_model_len`, or preempting low-priority requests. Timeout errors may indicate an input length distribution shift (users suddenly sending very long prompts). Dependency failures require checking the dependency's status, not the model server's.

Expose separate counters for each cause rather than aggregating all 5xx errors:

- `vllm_oom_errors_total` (custom counter)
- `vllm_request_timeout_total` (custom counter)
- `vllm_dependency_error_total` (custom counter)

**Queue overflow rate** — the rate at which new requests are rejected with 429 (Too Many Requests) because the server-side queue is full — is a distinct error type that signals infrastructure undersizing rather than a runtime bug.

### Queue depth: the leading indicator

Queue depth is the number of requests waiting to be scheduled for inference, not yet assigned to a GPU. It is the most important **leading indicator** of latency problems because it begins rising 2–5 minutes before p99 latency crosses the SLA threshold.

The relationship is direct: p99 latency = queue wait time + inference time. When queue depth is zero, p99 latency equals inference time (the model's inherent processing time). When queue depth grows, every new request waits for all queued requests to complete first. At queue depth $q$, the expected additional wait for a new request is approximately $q \times W_{\text{inference}}$, where $W_{\text{inference}}$ is the mean single-request inference time.

For a server processing requests with mean inference time 800ms, queue depth of 5 adds $5 \times 800\,\text{ms} = 4\,\text{seconds}$ to the latency of the next arrival. Set a queue depth alert threshold low — queue depth > 3 for more than 60 seconds is a reliable warning signal for most chatbot workloads.

vLLM exposes queue depth via the `vllm:num_requests_waiting` metric. In Triton, the equivalent is `nv_inference_queue_duration_us` (the cumulative time spent waiting) or `nv_inference_pending_request_count`.

![Queue depth growth pattern: traffic spike causes queue to grow, triggering HPA scale-out before p99 crosses SLA](/imgs/blogs/model-serving-slas-and-metrics-6.png)

### GPU memory pressure: cache utilization

**KV cache utilization** (`vllm:gpu_cache_usage_perc`) measures the fraction of the pre-allocated KV cache blocks that are occupied. When this reaches 85–90%, the vLLM scheduler starts preempting low-priority requests (swapping their KV cache to CPU) to make room for new ones. Preemption adds significant latency to preempted requests and shows up as TPOT spikes. Monitor this metric and alert when it exceeds 85%.

## 7. Cost metrics: the third side of the SLO triangle

Cost metrics connect GPU efficiency to business economics. Three primary cost metrics matter for model serving.

### Cost per token

**\$ per 1M input tokens** and **\$ per 1M output tokens** are the standard billing units in LLM APIs (following the OpenAI pricing model). Input token cost is dominated by prefill compute; output token cost is dominated by decode compute and is typically 2–3× higher than input token cost because decode is less parallelizable.

To compute the cost from first principles for a self-hosted deployment:

$$\text{Cost per token} = \frac{\text{GPU hourly cost}}{3600 \times \text{TPS}}$$

For a single H100 SXM5 on AWS at approximately \$8/hour, running Llama-3 70B at 500 tokens/second:

$$\text{Cost per token} = \frac{\$8}{3600 \times 500} = \$0.0000044 \approx \$4.40 \text{ per 1M tokens}$$

This is the raw compute cost; add approximately 30–40% for infrastructure overhead (networking, storage, load balancing) to get the fully-loaded cost. In comparison, managed providers charge \$0.27–\$3.00 per 1M output tokens for Llama-3 70B equivalents, implying very different margin structures depending on fleet utilization.

### GPU-hours per day and cost per request

**GPU-hours per day** is the capacity planning metric. For a fleet sized to handle peak traffic with 1.5× headroom and running 24 hours, actual utilization will be around 40–60% of peak due to traffic diurnal patterns. You want to track actual GPU-hours consumed per day against the budget, and particularly the **GPU-hour cost per successful request**, which normalizes for request volume changes.

**\$ per request** is derived as (GPU hourly cost × mean E2E latency) / (number of GPUs × utilization). It makes the latency/cost trade-off explicit: reducing mean E2E latency by 20% reduces the cost per request by 20% at the same throughput.

## 8. Prometheus metrics and collection setup

With the metrics taxonomy established, here is the implementation: how to actually collect all of these metrics using the standard observability stack.

### vLLM native metrics

vLLM exposes a Prometheus-compatible `/metrics` endpoint when started with the OpenAI-compatible server. The key metrics are:

```python
# vLLM OpenAI server startup — metrics exposed on :8000/metrics by default
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    # enables the built-in Prometheus metrics endpoint
    disable_log_stats=False,
)
```

Key vLLM Prometheus metrics (as of vLLM 0.4+):

| Metric name | Type | What it measures |
|---|---|---|
| `vllm:num_requests_running` | Gauge | Requests currently being processed |
| `vllm:num_requests_waiting` | Gauge | Requests in the queue (key leading indicator) |
| `vllm:num_requests_swapped` | Gauge | Requests with KV cache swapped to CPU |
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache occupancy 0–1 |
| `vllm:cpu_cache_usage_perc` | Gauge | CPU swap cache occupancy |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end latency per request |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT per request |
| `vllm:time_per_output_token_seconds` | Histogram | TPOT per token |
| `vllm:request_prompt_tokens` | Histogram | Input token count distribution |
| `vllm:request_generation_tokens` | Histogram | Output token count distribution |
| `vllm:request_success_total` | Counter | Successful request completions by finish reason |

### Prometheus scrape configuration for vLLM

```yaml
# prometheus.yml — scrape config for vLLM serving endpoint
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "vllm"
    static_configs:
      - targets: ["vllm-service:8000"]
    metrics_path: "/metrics"
    scrape_interval: 5s          # more frequent for latency metrics
    scrape_timeout: 4s
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - target_label: serving_framework
        replacement: vllm

  - job_name: "triton"
    static_configs:
      - targets: ["triton-service:8002"]  # Triton metrics port
    metrics_path: "/metrics"
    scrape_interval: 5s

  - job_name: "dcgm-exporter"
    static_configs:
      - targets: ["dcgm-exporter:9400"]
    scrape_interval: 5s
    # DCGM exposes: DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_MEM_COPY_UTIL,
    # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, DCGM_FI_PROF_DRAM_BW_PERC
```

### Triton Inference Server built-in metrics

Triton exposes Prometheus metrics on port 8002 by default. The most important:

```
# Triton key metrics
nv_inference_request_success          # counter: successful requests per model
nv_inference_request_failure          # counter: failed requests per model
nv_inference_queue_duration_us        # histogram: time spent in queue (microseconds)
nv_inference_compute_input_duration_us # histogram: input preprocessing time
nv_inference_compute_infer_duration_us # histogram: actual inference time
nv_inference_compute_output_duration_us # histogram: output postprocessing time
nv_gpu_utilization                    # gauge: SM active fraction (misleading for decode)
nv_gpu_memory_used_bytes              # gauge: current GPU memory usage
nv_gpu_power_usage                    # gauge: power draw in watts
```

### Grafana dashboard queries for p99 latency

The Grafana query for p99 TTFT using Prometheus's `histogram_quantile` function:

```
# p99 TTFT in milliseconds (5-minute window)
histogram_quantile(0.99,
  sum(rate(vllm:time_to_first_token_seconds_bucket[5m]))
  by (le, instance)
) * 1000

# p99 TPOT in milliseconds
histogram_quantile(0.99,
  sum(rate(vllm:time_per_output_token_seconds_bucket[5m]))
  by (le, instance)
) * 1000

# p99 E2E latency
histogram_quantile(0.99,
  sum(rate(vllm:e2e_request_latency_seconds_bucket[5m]))
  by (le, instance)
) * 1000

# Current RPS
sum(rate(vllm:request_success_total[1m])) by (instance)

# Queue depth (key leading indicator)
vllm:num_requests_waiting

# KV cache pressure
vllm:gpu_cache_usage_perc * 100

# Error rate
sum(rate(vllm:request_success_total{finished_reason="abort"}[5m])) /
sum(rate(vllm:request_success_total[5m]))
```

## 9. Benchmarking TTFT and TPOT from Python

Before setting SLA targets, you need to benchmark your specific model and workload to establish realistic baselines. Here is a complete Python benchmark script for vLLM that measures TTFT, TPOT, and E2E latency:

```python
"""
benchmark_latency.py — measures TTFT, TPOT, and E2E latency for vLLM
Usage: python benchmark_latency.py --model meta-llama/Meta-Llama-3-8B-Instruct
       --num-prompts 200 --max-tokens 150
"""
import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import List
import argparse

import httpx  # pip install httpx


@dataclass
class RequestResult:
    ttft_ms: float          # time to first token in milliseconds
    tpot_ms: float          # average time per output token in milliseconds
    e2e_ms: float           # total request time in milliseconds
    output_tokens: int      # number of output tokens generated
    error: str | None = None


async def measure_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Stream a single request and measure TTFT and TPOT."""
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    t_start = time.perf_counter()
    t_first_token = None
    token_times: List[float] = []

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                t_now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = t_now   # first SSE chunk = first token
                else:
                    token_times.append(t_now)  # subsequent tokens

    except Exception as exc:
        return RequestResult(
            ttft_ms=0, tpot_ms=0, e2e_ms=0, output_tokens=0, error=str(exc)
        )

    t_end = time.perf_counter()

    if t_first_token is None:
        return RequestResult(
            ttft_ms=0, tpot_ms=0, e2e_ms=0, output_tokens=0,
            error="No tokens received"
        )

    ttft_ms = (t_first_token - t_start) * 1000
    e2e_ms = (t_end - t_start) * 1000
    output_tokens = len(token_times) + 1  # +1 for first token

    # TPOT = (E2E - TTFT) / (output_tokens - 1)
    if output_tokens > 1:
        tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
    else:
        tpot_ms = 0.0

    return RequestResult(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
    )


async def run_benchmark(
    base_url: str,
    num_prompts: int,
    max_tokens: int,
    concurrency: int,
) -> None:
    """Run benchmark with controlled concurrency."""
    prompt = (
        "Explain the difference between supervised and unsupervised learning "
        "in machine learning. Include examples and discuss when to use each."
    )  # ~50-token prompt

    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async def bounded_request(client: httpx.AsyncClient) -> None:
        async with semaphore:
            r = await measure_single_request(client, base_url, prompt, max_tokens)
            results.append(r)

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client) for _ in range(num_prompts)]
        await asyncio.gather(*tasks)

    successful = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    if not successful:
        print("All requests failed.")
        return

    ttfts = sorted(r.ttft_ms for r in successful)
    tpots = sorted(r.tpot_ms for r in successful if r.tpot_ms > 0)
    e2es = sorted(r.e2e_ms for r in successful)

    def pct(data: List[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    print(f"\n=== Benchmark results ({len(successful)}/{num_prompts} successful) ===")
    print(f"Concurrency: {concurrency}, max_tokens: {max_tokens}")
    print(f"\nTTFT (ms):  mean={statistics.mean(ttfts):.1f}  "
          f"p50={pct(ttfts,50):.1f}  p90={pct(ttfts,90):.1f}  "
          f"p99={pct(ttfts,99):.1f}")
    print(f"TPOT (ms):  mean={statistics.mean(tpots):.1f}  "
          f"p50={pct(tpots,50):.1f}  p90={pct(tpots,90):.1f}  "
          f"p99={pct(tpots,99):.1f}")
    print(f"E2E (ms):   mean={statistics.mean(e2es):.1f}  "
          f"p50={pct(e2es,50):.1f}  p90={pct(e2es,90):.1f}  "
          f"p99={pct(e2es,99):.1f}")
    print(f"Mean output tokens: {statistics.mean(r.output_tokens for r in successful):.1f}")
    print(f"Errors: {len(errors)}")
    if errors:
        from collections import Counter
        print("Error breakdown:", Counter(r.error[:50] for r in errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        args.base_url, args.num_prompts, args.max_tokens, args.concurrency
    ))
```

Run this with varying concurrency levels (1, 5, 10, 20, 50) to plot the latency-throughput curve and find the knee point where p99 latency starts to grow super-linearly — that is your maximum sustainable QPS.

## 10. Alerting rules: when to page on-call

Well-designed alerts fire early enough to allow a response before users are impacted, but not so early that they create alert fatigue. The goal is a three-tier system: informational warnings that a human reviews during business hours, soft alerts that go to a Slack channel, and hard alerts that page the on-call engineer immediately.

```yaml
# prometheus/alerts/model-serving.yaml
groups:
  - name: model-serving-sla
    interval: 30s
    rules:
      # ─── HARD ALERTS: page on-call immediately ───────────────────────────
      - alert: TTFTSLABreach
        expr: |
          histogram_quantile(0.99,
            sum(rate(vllm:time_to_first_token_seconds_bucket[5m]))
            by (le, instance)
          ) > 0.5
        for: 3m
        labels:
          severity: critical
          team: ml-serving
        annotations:
          summary: "p99 TTFT > 500ms SLO for 3 minutes on {{ $labels.instance }}"
          description: "Current p99 TTFT: {{ $value | humanizeDuration }}"

      - alert: ErrorRateHigh
        expr: |
          (
            sum(rate(vllm:request_success_total{finished_reason="abort"}[2m]))
            /
            sum(rate(vllm:request_success_total[2m]))
          ) > 0.01
        for: 2m
        labels:
          severity: critical
          team: ml-serving
        annotations:
          summary: "Error rate > 1% for 2 minutes"
          description: "Current error rate: {{ $value | humanizePercentage }}"

      # ─── SOFT ALERTS: Slack notification, no page ─────────────────────────
      - alert: QueueDepthElevated
        expr: vllm:num_requests_waiting > 3
        for: 1m
        labels:
          severity: warning
          team: ml-serving
        annotations:
          summary: "vLLM request queue depth > 3 — latency will rise"
          description: "Queue depth: {{ $value }}. Scale out or investigate."

      - alert: GPUUtilizationSaturation
        expr: avg(DCGM_FI_DEV_GPU_UTIL) by (instance) > 95
        for: 10m
        labels:
          severity: warning
          team: ml-serving
        annotations:
          summary: "GPU utilization > 95% sustained — approaching saturation"

      - alert: KVCachePressure
        expr: vllm:gpu_cache_usage_perc > 0.85
        for: 2m
        labels:
          severity: warning
          team: ml-serving
        annotations:
          summary: "KV cache > 85% — preemption likely, TPOT spikes expected"

      - alert: TTFTSoftWarning
        expr: |
          histogram_quantile(0.99,
            sum(rate(vllm:time_to_first_token_seconds_bucket[5m]))
            by (le, instance)
          ) > 0.4
        for: 5m
        labels:
          severity: warning
          team: ml-serving
        annotations:
          summary: "p99 TTFT > 400ms (80% of SLO budget) — investigate early"
```

### The four golden signals applied to model serving

Google's SRE book describes four golden signals — latency, traffic, errors, saturation — as the minimum viable observability set for any system. Applied specifically to model serving:

**Latency** maps to p99 TTFT and p99 TPOT. Use histograms, not gauges. Track both separately.

**Traffic** maps to RPS (arrival rate), TPS (tokens per second), and queue depth. Queue depth is the most actionable traffic signal because it is the precursor to latency degradation.

**Errors** maps to 5xx rate segmented by cause (OOM, timeout, dependency failure) plus the request abort rate from vLLM's finish reason counter.

**Saturation** maps to KV cache utilization (`vllm:gpu_cache_usage_perc`) and GPU memory utilization. When saturation approaches 100%, all four other signals degrade simultaneously. Saturation is the fire; the other signals are the smoke.

![Alert escalation timeline: three-tier system from soft warning to on-call page](/imgs/blogs/model-serving-slas-and-metrics-7.png)

## 11. The SLO design process: from user experience to fleet size

The correct direction for SLO design is top-down: start from what the user needs, translate that into metric targets, derive capacity requirements, and size the fleet. Doing it bottom-up (sizing the fleet first and then declaring whatever it can do as the SLA) produces SLAs that are disconnected from user experience.

### Step 1: Define user experience requirements

The starting point is a user research or product requirement: "the chatbot must feel responsive for interactive conversation." Translating "feel responsive" into numbers requires understanding human perception: TTFT below 300ms feels instant; TTFT 300–500ms feels fast; TTFT 500ms–1.5s feels slightly laggy; above 1.5s users start showing impatience. For TPOT, reading speed is roughly 3–5 tokens per second, so TPOT below 300ms/token is invisible; above 300ms/token the streaming begins to feel choppy.

For our running example: user experience requirement = "chatbot feels fast for interactive Q&A". This translates to: TTFT < 500ms, TPOT < 50ms/token (= 20 tokens/second streaming), both at p99.

### Step 2: Derive E2E latency budget

With TTFT and TPOT targets and an estimate of typical response length, compute the E2E latency budget:

$$\text{E2E}_{\text{p99}} = 500\,\text{ms} + (150 - 1) \times 50\,\text{ms} = 500 + 7450 = 7950\,\text{ms} \approx 8\,\text{s}$$

This 8-second p99 E2E latency does not mean the chatbot is slow — it means that a 150-token response can take up to 8 seconds to complete streaming while still delivering each token smoothly to the user. The first token arrives in under 500ms; subsequent tokens arrive at 20/second. The user experience is good throughout.

### Step 3: Translate to QPS capacity using Little's Law

With the E2E latency budget and peak traffic estimate, apply Little's Law to derive required concurrency:

$$L = \lambda \times W = 33\,\text{req/s} \times 7.95\,\text{s} = 263\,\text{concurrent requests}$$

Now check whether this is achievable given GPU memory. For Llama-3 8B on A100-80GB, the KV cache capacity is approximately 300–400 concurrent requests at 150-token average response (as computed in the worked example in section 3). So KV cache memory is not the bottleneck at 263 concurrent requests.

### Step 4: Check compute throughput

Peak token demand: $33\,\text{req/s} \times 150\,\text{tokens/req} = 4950\,\text{tokens/s}$. A single A100-80GB running vLLM with Llama-3 8B delivers approximately 1200–1800 tokens/second with continuous batching at moderate concurrency. You need at least $\lceil 4950 / 1500 \rceil = 4$ A100s. Add 1.5× headroom: 6 A100s.

### Step 5: Set infrastructure SLOs

With the fleet sized, set the infrastructure targets that, if met, guarantee the user-facing SLO:

- p99 TTFT < 500ms (primary SLO; if this is met, user experience is good)
- p99 TPOT < 50ms/token (primary SLO; streaming smoothness)
- Queue depth < 3 (warning indicator; triggers scaling)
- KV cache utilization < 85% (saturation indicator)
- Error rate < 0.1% (reliability indicator)
- GPU memory utilization < 90% (headroom indicator)

![SLO design flow: from user experience requirement down through metric targets to hardware sizing](/imgs/blogs/model-serving-slas-and-metrics-5.png)

#### Worked example: budget decomposition for a 2-second SLA

Your product manager wants a "2-second response" SLA. You need to translate this into TTFT and TPOT budgets.

Given: E2E p99 < 2000ms; typical response length: 100 tokens.

The formula is: E2E = TTFT + (N-1) × TPOT. You have two unknowns (TTFT and TPOT) and one equation. The constraint is that the user experience should be good both for responsiveness (low TTFT) and streaming smoothness (low TPOT).

**Option A: Responsiveness-first budget.** Allocate 500ms to TTFT, leaving 1500ms for decode: TPOT = 1500 / (100-1) = 15.15ms/token. This is excellent streaming — 66 tokens/second. Users see the first token in 500ms and then text arrives very fast. This is the right budget for a code completion tool where users are waiting for a complete answer.

**Option B: Even split.** If we target TTFT = 1000ms and use the remaining 1000ms for decode: TPOT = 1000/99 = 10.1ms/token. The streaming is even faster, but users wait 1 second to see anything. This is worse for perceived responsiveness.

**Option C: Aggressive TTFT.** TTFT = 200ms (achieved via prefill/decode disaggregation), leaving 1800ms for TPOT: TPOT = 1800/99 = 18.2ms/token. Best perceived responsiveness, but the 200ms TTFT target requires a dedicated prefill fleet which adds operational complexity.

For a conversational chatbot, Option A (500ms TTFT, ~15ms TPOT for 100-token responses) is almost always the right choice. It balances responsiveness with simplicity.

## 12. Metrics reference table

The complete reference for every metric covered in this post:

| Metric | What it measures | How to collect | SLA threshold | Alert rule |
|---|---|---|---|---|
| p99 TTFT | Time to first streaming token | `vllm:time_to_first_token_seconds` histogram | < 500ms | > 500ms for 3 min |
| p99 TPOT | Time between successive tokens | `vllm:time_per_output_token_seconds` histogram | < 50ms | > 100ms for 3 min |
| p99 E2E latency | Full request completion time | `vllm:e2e_request_latency_seconds` histogram | workload-dependent | > 2× SLO for 5 min |
| RPS | Request arrival/completion rate | `rate(vllm:request_success_total[1m])` | capacity-dependent | N/A (use queue depth) |
| TPS | Output tokens per second | RPS × mean output length | capacity-dependent | N/A (use MFU) |
| Queue depth | Requests waiting for GPU | `vllm:num_requests_waiting` | < 3 | > 3 for 1 min |
| KV cache util | GPU memory occupied by KV | `vllm:gpu_cache_usage_perc` | < 85% | > 85% for 2 min |
| GPU SM util | Fraction of SMs active | `DCGM_FI_DEV_GPU_UTIL` | < 95% | > 95% for 10 min |
| HBM BW util | Fraction of HBM bandwidth used | `DCGM_FI_PROF_DRAM_BW_PERC` | > 75% (decode) | < 30% under load |
| MFU | Actual / peak FLOPs | custom: FLOPs / (TPS × model_params × 2) | > 30% (prefill) | < 10% sustained |
| 5xx error rate | Failed request fraction | rate of non-success finish_reason | < 0.1% | > 1% for 2 min |
| OOM rate | Out-of-memory errors | custom counter | 0 | any OOM in 5 min |
| \$ per 1M tokens | Fully-loaded cost per token | (GPU cost/hr) / (TPS × 3600) | budget-dependent | > 2× baseline |

![Collection method matrix: which tool exposes each metric natively](/imgs/blogs/model-serving-slas-and-metrics-4.png)

## 13. Case studies and benchmarks

### vLLM paper benchmarks (Kwon et al., 2023)

The original vLLM paper ("Efficient Memory Management for Large Language Model Serving with PagedAttention", Kwon et al., NeurIPS 2023) benchmarks throughput against FasterTransformer and Orca on ShareGPT workloads. The key finding relevant to metrics: vLLM achieves 2–4× higher throughput than Orca at the same p99 latency target. This is directly attributable to PagedAttention's near-zero KV cache fragmentation, which keeps KV cache utilization above 95% compared to Orca's 20–40%. The implication: KV cache utilization is not just a capacity metric but a throughput multiplier. Under-utilized KV cache means under-utilized GPU.

Specific benchmark numbers from the paper (OPT-13B on A100 80GB): vLLM achieves 17.6 req/s at p99 < 1s versus Orca's 7.6 req/s at the same latency constraint — a 2.3× throughput improvement from better memory management alone. The paper defines "goodput" as throughput at which all p99 latency SLOs are met, which is a useful framing: raw peak throughput is meaningless without a latency constraint.

### TGI benchmarks (Hugging Face, 2023)

Hugging Face's text-generation-inference benchmarking blog demonstrates that continuous batching (which TGI and vLLM both implement) achieves 3–10× higher throughput than static batching on mixed-length workloads. The key insight is that static batching's effective throughput on LLM-typical prompt-length distributions (ShareGPT: median 60 tokens input, high variance) is dominated by the longest request in the batch. Continuous batching eliminates this by scheduling at the token level rather than the request level, keeping the GPU busy across the full decode chain.

For the metrics implications: if you switch from static to continuous batching, your p99 TPOT may increase slightly (because the GPU is now shared across more concurrent requests) while p99 TTFT decreases significantly (less queue wait). Your MFU will increase substantially. This is the latency/throughput trade-off made concrete.

### Kimi K2 SLO management at scale (Moonshot AI, 2025)

Moonshot AI's production serving system for Kimi K2 (a 1T-parameter MoE model) implements what they call SLO-aware preemption: the scheduler actively preempts requests that are about to violate their TPOT SLO due to KV cache pressure, restoring them to a priority queue that is served before new requests. The operational result, as reported in their public technical blog, is that p99 TPOT remains within 20% of the SLO even at 90%+ KV cache utilization — a result that naive FIFO scheduling cannot achieve at those utilization levels. The key takeaway for metrics: tracking p99 TPOT per cohort (requests by arrival time or priority tier) reveals scheduler behavior that aggregate p99 hides.

### Triton production deployment (NVIDIA, 2024 docs)

NVIDIA's production guidance for Triton Inference Server recommends using `nv_inference_queue_duration_us` as the primary autoscaling signal, not GPU utilization. Their recommendation is to trigger horizontal scale-out when the p99 queue duration (not queue depth count) exceeds 100ms. This is more accurate than queue depth because it normalizes for inference time variance: a queue depth of 5 with 20ms inference time is very different from a queue depth of 5 with 800ms inference time. Using duration rather than count provides a direct connection to the latency SLA.

## 14. When to use this framework (and when not to)

### Use this full metrics framework when:

- You are operating an LLM serving system with interactive users and a latency SLA. The TTFT/TPOT decomposition is only meaningful for autoregressive generation; it does not apply to embedding models, classification models, or batch processing.
- You handle more than 10 req/s sustained. Below this threshold, the statistical validity of p99 estimates is poor — you need at least 100 requests per 5-minute window to get a reliable p99. At 1 req/s you would need to use a 2-hour window to estimate p99, which is not operationally useful.
- You are planning fleet capacity for a new deployment. The Little's Law workflow in section 11 is the correct approach and prevents both under-provisioning (SLA misses) and over-provisioning (wasted cost).
- You are comparing serving frameworks (vLLM vs TGI vs Triton). Use MFU as the primary comparison metric; it is hardware-normalized and framework-agnostic.

### Do not apply this framework when:

- **Batch inference workloads.** If your use case is offline batch processing (summarizing a document corpus overnight, running evaluations), TTFT and TPOT are irrelevant. The relevant metrics are total throughput (tokens/hour), GPU utilization, and cost per batch job. Do not confuse online serving metrics with batch processing metrics.
- **Embedding model serving.** Embedding models are a single forward pass with no autoregressive decode. They behave like classical ML models. Use simple p99 latency, not TTFT/TPOT decomposition.
- **Very-small-scale deployments (fewer than 1 req/s).** If you are serving a personal assistant with one user, operational complexity of Prometheus + Grafana + alerting exceeds the value. Use application-level logging and simple mean latency tracking.
- **Models with fixed output length.** Classification models, regression models, and models that always produce the same number of output tokens do not benefit from TPOT tracking because there is no output-length variance.

## 15. MFU measurement in practice

While the concept of MFU is straightforward, measuring it in practice requires a few non-obvious steps. vLLM does not yet expose MFU as a built-in metric (as of version 0.5). You need to compute it from available primitives:

```python
"""
compute_mfu.py — derive MFU from vLLM metrics and model config
Requires: pip install prometheus_client requests
"""
import requests
import re
import time


# ── Model architecture constants (Llama-3 8B) ────────────────────────────────
# These come from the model's config.json
LAYERS = 32
HIDDEN_DIM = 4096
FFN_INTERMEDIATE = 14336     # SwiGLU intermediate dim
NUM_HEADS = 32
HEAD_DIM = 128
VOCAB_SIZE = 128256

# H100 SXM5 theoretical peak BF16 FLOPs/s with tensor cores
GPU_PEAK_FLOPS_BF16 = 3.958e15   # 3.958 PFLOP/s

# FLOPs per forward pass token for Llama-3 8B (approximate)
# 2 × (attention QKV + attention output + FFN gate/up + FFN down + lm_head)
FLOPS_PER_TOKEN = (
    2 * (
        # QKV projection: 3 × d × d
        3 * HIDDEN_DIM * HIDDEN_DIM +
        # Output projection: d × d
        HIDDEN_DIM * HIDDEN_DIM +
        # FFN gate + up: 2 × d × d_ffn
        2 * HIDDEN_DIM * FFN_INTERMEDIATE +
        # FFN down: d_ffn × d
        FFN_INTERMEDIATE * HIDDEN_DIM
    ) * LAYERS +
    # LM head: d × vocab
    2 * HIDDEN_DIM * VOCAB_SIZE
)


def get_vllm_tps(prometheus_url: str = "http://localhost:8000/metrics") -> float:
    """Read current tokens per second from vLLM metrics."""
    response = requests.get(prometheus_url, timeout=5)
    text = response.text

    # vllm:generation_tokens_total is incremented per output token
    # We use a simple 5-second window by calling twice
    def extract_counter(metric_name: str, text: str) -> float:
        pattern = rf'^{re.escape(metric_name)}\s+([\d.e+]+)'
        match = re.search(pattern, text, re.MULTILINE)
        return float(match.group(1)) if match else 0.0

    t1_text = requests.get(prometheus_url, timeout=5).text
    t1 = time.time()
    tokens_1 = extract_counter("vllm:generation_tokens_total", t1_text)

    time.sleep(5)

    t2_text = requests.get(prometheus_url, timeout=5).text
    t2 = time.time()
    tokens_2 = extract_counter("vllm:generation_tokens_total", t2_text)

    tps = (tokens_2 - tokens_1) / (t2 - t1)
    return tps


def compute_mfu(tps: float) -> float:
    """Compute MFU given observed tokens per second."""
    actual_flops_per_second = tps * FLOPS_PER_TOKEN
    mfu = actual_flops_per_second / GPU_PEAK_FLOPS_BF16
    return mfu


if __name__ == "__main__":
    tps = get_vllm_tps()
    mfu = compute_mfu(tps)
    print(f"TPS: {tps:.1f} tokens/second")
    print(f"FLOPs per token: {FLOPS_PER_TOKEN/1e9:.2f} GFLOPs")
    print(f"Actual FLOPs/s: {tps * FLOPS_PER_TOKEN / 1e12:.2f} TFLOPs/s")
    print(f"MFU: {mfu*100:.1f}%")
    print(f"  (target: >30% for mixed prefill/decode)")
```

### Typical MFU ranges by workload type

On an H100 running Llama-3 8B in BF16:

| Workload | Typical MFU | Bottleneck |
|---|---|---|
| Pure prefill, batch 64, 512-token prompts | 45–55% | Compute (attention FLOPs) |
| Mixed prefill/decode, continuous batching | 30–42% | Mixed compute + memory |
| Pure decode, batch 1 | 3–8% | Memory bandwidth (severe) |
| Pure decode, batch 32 | 12–20% | Memory bandwidth |
| Pure decode, batch 128 (large continuous batch) | 25–35% | Approaching compute-bound |
| FlashAttention-2 + batch 64 prefill | 50–60% | Near optimal |

The implication: you should never evaluate a serving system's efficiency with batch size 1. Single-request benchmarks are only useful for measuring TTFT floor; they dramatically understate the system's MFU and throughput capability.

![MFU comparison: naive eager decode at 12% versus FlashAttention-2 with continuous batching at 38%](/imgs/blogs/model-serving-slas-and-metrics-8.png)

## 16. Advanced latency decomposition: building a flame graph for inference

The TTFT/TPOT split is the first decomposition, but production debugging requires finer granularity. When p99 TTFT is above target, you need to know *where inside* the TTFT budget the time is being spent: is it tokenization, batching overhead, KV cache allocation, the actual prefill kernels, or network serialization? Here is how to build a complete latency breakdown using OpenTelemetry tracing alongside Prometheus metrics.

### Instrumenting vLLM with OpenTelemetry

vLLM 0.5+ supports OpenTelemetry tracing via the `--otlp-traces-endpoint` flag. When enabled, it emits spans for each of the major phases of request processing:

```python
# Launch vLLM with OpenTelemetry tracing enabled
# Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    otlp_traces_endpoint="http://jaeger:4317",  # OTLP gRPC endpoint
    collect_detailed_traces="all",  # emit per-request spans
)
```

Or equivalently at the command line:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --otlp-traces-endpoint http://jaeger-collector:4317 \
  --collect-detailed-traces all
```

The span hierarchy that vLLM emits looks like this:

```
vllm.request (root span, duration = E2E latency)
  ├── vllm.add_request (enqueue, ~0.1ms)
  ├── vllm.scheduler.step (select batch to run)
  │     ├── vllm.kv_cache_alloc (allocate KV blocks)
  │     └── vllm.batch_prefill (if request in prefill phase)
  ├── vllm.execute_model (GPU kernel execution)
  │     ├── vllm.attn_prefill (attention prefill kernel)
  │     ├── vllm.mlp (FFN kernels)
  │     └── vllm.sampling (logits + sampling)
  └── vllm.send_token (stream token to client)
```

Once traces are flowing to Jaeger or Tempo, you can compute the latency contribution of each phase:

- `vllm.execute_model` duration on the first step = TTFT minus network/scheduling overhead
- `vllm.execute_model` duration on subsequent steps = TPOT per decode step
- `vllm.scheduler.step` overhead = scheduling latency added per batch cycle
- `vllm.kv_cache_alloc` duration = memory management overhead (should be < 0.5ms; if higher, indicates KV cache pressure)

### Adding custom OpenTelemetry middleware for FastAPI gateways

If you have a FastAPI gateway sitting in front of vLLM (for authentication, routing, caching), you need to trace that layer too:

```python
"""
gateway_tracing.py — OpenTelemetry middleware for an LLM API gateway
"""
from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import time

# Configure the tracer
provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger-collector:4317")
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-gateway")

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)  # auto-instruments all routes


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    with tracer.start_as_current_span("gateway.request") as span:
        body = await request.json()
        prompt_tokens = len(body.get("messages", [{}])[-1].get("content", "").split())
        span.set_attribute("prompt_tokens_estimate", prompt_tokens)
        span.set_attribute("model", body.get("model", "unknown"))

        t_auth_start = time.perf_counter()
        # ... authentication logic ...
        t_auth_end = time.perf_counter()
        span.set_attribute("auth_latency_ms", (t_auth_end - t_auth_start) * 1000)

        with tracer.start_as_current_span("gateway.route_to_backend") as route_span:
            # ... forward to vLLM ...
            pass

        # TTFT annotation from SSE stream
        with tracer.start_as_current_span("gateway.stream_tokens") as stream_span:
            first_token_seen = False
            # ... stream tokens from vLLM response ...
            # Set TTFT span attribute when first token arrives
            stream_span.set_attribute("ttft_ms", 0)  # populated in streaming loop
```

The resulting trace gives you a complete attribution of where latency comes from: authentication adds X ms, routing adds Y ms, the model server is responsible for Z ms of TTFT. This decomposition is invaluable when debugging a p99 TTFT regression that affects only a specific user cohort (perhaps users with long system prompts triggering a slow authentication path).

### The token variance problem: per-request TPOT instability

One subtlety that aggregate metrics hide: TPOT is not constant across a single response. During high-concurrency operation with continuous batching, TPOT varies significantly per decode step based on how many other requests are sharing the batch at that moment. A request that starts decoding when 32 other requests are in-flight will see higher TPOT than the same request decoding alone.

The metric that captures this is **TPOT variance**, not just TPOT mean. A p99 TPOT of 50ms may be composed of decode steps that take 20ms–80ms with high variance, causing the streaming to appear choppy even though the average is within SLO. To detect this, track TPOT as a histogram rather than just a percentile, and add a gauge for `vllm:time_per_output_token_seconds` variance computed via:

```
# Grafana query: TPOT coefficient of variation (std/mean)
(
  sqrt(
    rate(vllm:time_per_output_token_seconds_sum[5m]) / 
    rate(vllm:time_per_output_token_seconds_count[5m]) -
    pow(
      rate(vllm:time_per_output_token_seconds_sum[5m]) / 
      rate(vllm:time_per_output_token_seconds_count[5m]),
      2
    )
  )
) /
(
  rate(vllm:time_per_output_token_seconds_sum[5m]) / 
  rate(vllm:time_per_output_token_seconds_count[5m])
)
```

A coefficient of variation (CV) above 0.5 for TPOT under moderate load indicates unstable batch composition — a signal that the scheduler is mixing very different request types in ways that cause irregular per-step GPU load.

## 17. SLA multi-tenancy: per-tenant and per-model metrics

Production LLM serving systems commonly serve multiple tenants (different customers, internal teams, or product lines) or multiple models from a shared GPU fleet. In this scenario, aggregate system metrics are insufficient — you need per-tenant attribution to understand which workload is violating the SLA.

### Per-tenant metrics in vLLM

vLLM does not natively support per-tenant metric labels, but you can add them via a request middleware that propagates a tenant identifier from the HTTP header to Prometheus labels:

```python
"""
tenant_metrics.py — per-tenant Prometheus metrics middleware for vLLM gateway
"""
from prometheus_client import Histogram, Counter, REGISTRY
from fastapi import Request
import time

# Custom per-tenant histograms with a 'tenant' label
TENANT_TTFT = Histogram(
    "gateway_ttft_seconds",
    "Time to first token per tenant",
    labelnames=["tenant", "model"],
    buckets=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
)

TENANT_TPOT = Histogram(
    "gateway_tpot_seconds",
    "Time per output token per tenant",
    labelnames=["tenant", "model"],
    buckets=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5],
)

TENANT_REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total requests per tenant",
    labelnames=["tenant", "model", "status"],
)


async def record_tenant_metrics(
    request: Request,
    tenant_id: str,
    model_id: str,
    ttft_seconds: float,
    tpot_seconds: float | None,
    status: str = "success",
) -> None:
    """Record per-tenant metrics for a completed streaming request."""
    TENANT_TTFT.labels(tenant=tenant_id, model=model_id).observe(ttft_seconds)
    if tpot_seconds is not None:
        TENANT_TPOT.labels(tenant=tenant_id, model=model_id).observe(tpot_seconds)
    TENANT_REQUEST_COUNT.labels(
        tenant=tenant_id, model=model_id, status=status
    ).inc()
```

With per-tenant labels, you can write Grafana queries that show each tenant's p99 TTFT separately:

```
# Per-tenant p99 TTFT
histogram_quantile(0.99,
  sum(rate(gateway_ttft_seconds_bucket[5m]))
  by (le, tenant, model)
) * 1000
```

This immediately reveals if a single tenant (perhaps one running extremely long prompts) is responsible for a p99 degradation that appears in system-wide metrics but does not affect other tenants.

### SLA tiers and priority queues

When different tenants have different SLA agreements — for example, a premium tier with TTFT < 200ms and a standard tier with TTFT < 800ms — you need to implement priority scheduling in the serving system. vLLM supports request priority via the `priority` parameter in `AsyncLLMEngine.add_request()`:

```python
from vllm import AsyncLLMEngine, SamplingParams

engine = AsyncLLMEngine.from_engine_args(engine_args)

async def serve_request(prompt: str, tenant_tier: str) -> AsyncGenerator:
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    # Map SLA tier to priority (lower number = higher priority)
    priority_map = {"premium": 0, "standard": 5, "batch": 10}
    priority = priority_map.get(tenant_tier, 5)

    async for output in engine.generate(
        prompt,
        sampling_params=sampling_params,
        request_id=f"req-{time.time()}",
        priority=priority,      # available in vLLM >= 0.5
    ):
        yield output
```

With priority queuing, the scheduler will drain high-priority (premium) requests from the queue before processing standard-tier requests, maintaining the premium TTFT SLA even during queue buildup events. The trade-off is that standard-tier requests may experience higher p99 latency during traffic peaks — which is exactly what the tiered SLA contract permits.

#### Worked example: multi-tier SLA capacity planning

You have two tenant tiers sharing a 4× A100-80GB fleet running Llama-3 8B:

- **Premium tier**: 20 req/s, TTFT < 200ms p99, TPOT < 30ms p99
- **Standard tier**: 50 req/s, TTFT < 800ms p99, TPOT < 80ms p99
- Average response: 100 tokens for both tiers

Step 1 — total arrival rate: $\lambda_{\text{total}} = 20 + 50 = 70\,\text{req/s}$.

Step 2 — effective E2E for capacity planning (use standard tier as the binding constraint): $W_{\text{standard}} = 0.8 + 99 \times 0.08 = 0.8 + 7.92 = 8.72\,\text{s}$.

Step 3 — required concurrency: $L = 70 \times 8.72 \approx 610\,\text{concurrent requests}$. A single A100-80GB holds approximately 350–500 concurrent 100-token Llama-3 8B requests. You need at least 2 A100s for KV cache memory.

Step 4 — compute throughput: $70\,\text{req/s} \times 100\,\text{tok/req} = 7000\,\text{tok/s}$. Four A100s deliver approximately $4 \times 1500 = 6000\,\text{tok/s}$ sustainable throughput — you are slightly over capacity. With priority queuing, premium requests get served first, so their p99 TTFT stays below 200ms while standard tier absorbs the queue wait. Add a 5th A100 for safety margin: $5 \times 1500 = 7500\,\text{tok/s}$, now 7% headroom above peak demand.

Step 5 — validate the premium TTFT target. With priority queuing and 20 premium req/s on a 7500 tok/s fleet, premium requests should almost never queue. Mean premium queue wait ≈ $20 / 7500 \approx 0.003\,\text{s}$ = 3ms. Prefill time for a typical 200-token prompt on Llama-3 8B ≈ 80ms. TTFT ≈ 80 + 3 = 83ms, well within the 200ms premium SLA.

## 18. Connecting metrics to optimization decisions

Metrics are only useful if they drive action. Each metric in the taxonomy maps to a specific optimization lever. This section closes the loop between measurement and the techniques covered elsewhere in this series.

### When p99 TTFT is too high

If p99 TTFT exceeds the SLA target, the cause is one of three things: too much queue wait time, too long a prefill computation, or excessive network latency.

**Diagnosis**: look at `vllm:num_requests_waiting` (queue depth). If it is frequently above 0, the queue is the primary cause. If queue depth is consistently 0 but TTFT is still high, the prefill compute itself is the bottleneck.

**Remedy for queue-caused TTFT**: scale out the fleet (more replicas), or implement prefill/decode disaggregation (a dedicated prefill fleet that processes prompts faster by running at higher batch utilization on compute-bound workloads). This is covered in depth in [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation).

**Remedy for compute-caused TTFT**: reduce prompt length (via prompt compression or retrieval augmentation that provides shorter context), enable chunked prefill in vLLM (`--enable-chunked-prefill --max-num-batched-tokens 4096`), or increase GPU count for tensor parallelism to split the prefill computation across more devices.

### When p99 TPOT is too high

High p99 TPOT during decode is almost always caused by one of: insufficient memory bandwidth, excessive concurrency creating memory pressure, or KV cache preemption overhead.

**Diagnosis**: check `vllm:gpu_cache_usage_perc`. If it is above 85%, KV cache preemption is likely causing TPOT spikes. If KV cache utilization is normal but TPOT is high, check the HBM bandwidth utilization metric from DCGM.

**Remedy for KV-cache-caused TPOT**: reduce `max_model_len` to limit the maximum KV cache per request, or reduce `gpu_memory_utilization` to reserve more free memory as headroom and prevent preemption. See [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) for advanced prefix caching and RadixAttention techniques.

**Remedy for bandwidth-caused TPOT**: use quantization (GPTQ, AWQ, or FP8) to reduce model weight size and thus reduce bytes-per-decode-step. This is the most effective single intervention for improving TPOT on memory-bandwidth-limited hardware. A W4A16 quantized Llama-3 70B uses roughly half the memory bandwidth of BF16, approximately doubling TPOT throughput.

### When MFU is low but GPU utilization is high

If MFU is below 15% while GPU utilization reports 80%, the system is memory-bandwidth-bound but the arithmetic units are idle. The GPU is busy doing memory-access work, not useful FLOPs.

**Remedy**: increase batch size. With continuous batching, vLLM will automatically consolidate multiple concurrent requests into larger decode batches. If your concurrency is low (under 8 concurrent requests), artificially increasing request concurrency (accepting more simultaneous requests) will raise the decode batch size and improve MFU and TPS together. The trade-off is that larger batches mean some requests wait longer for the batch to complete, which increases p99 TPOT slightly.

### When error rate spikes

Sudden error rate increases require rapid diagnosis before remediation, because the wrong fix can make things worse. Check the error breakdown:

- **OOM errors**: `vllm:gpu_cache_usage_perc` approaching 1.0 → reduce `max_model_len` or scale out immediately
- **Timeout errors**: p99 E2E latency > configured timeout → check for unusual prompt length distribution (input length histogram `vllm:request_prompt_tokens`)
- **CUDA errors**: check DCGM for GPU health events, ECC errors, Xid codes → the GPU may be failing hardware

A spike in timeout errors at a consistent time of day often signals a traffic pattern shift (users in a specific timezone sending longer prompts at peak hours). The correct remedy is increasing timeout thresholds for known-long workloads, not reducing model capacity.

## 19. Building a Grafana SLA dashboard

A complete Grafana dashboard for the chatbot SLA example should have five panels in priority order.

**Panel 1 — SLA status (stat panel)**: current p99 TTFT versus 500ms threshold and current p99 TPOT versus 50ms threshold. Use traffic light coloring: green below 80% of threshold, yellow 80–100%, red above threshold. This is the first thing on-call engineers look at.

**Panel 2 — Latency time series (graph panel)**: p50, p90, p99 TTFT and TPOT over the past 6 hours. Overlay the SLO threshold lines. Annotate with deployment events from the annotation API. This panel answers "when did the regression start?"

**Panel 3 — Queue depth and RPS (dual-axis graph)**: queue depth on the left axis, RPS on the right. When RPS spikes and queue depth grows simultaneously, it is a traffic event. When queue depth grows without an RPS spike, it is a server-side degradation.

**Panel 4 — GPU and KV cache saturation (gauge panel)**: `DCGM_FI_DEV_GPU_UTIL` and `vllm:gpu_cache_usage_perc` as gauges. Color the KV cache gauge red above 85%.

**Panel 5 — Error rate breakdown (stacked bar)**: OOM errors, timeout errors, and dependency errors as separate series in a stacked bar chart. This makes it immediately visible whether an error spike is caused by one category or all categories simultaneously (all three spiking simultaneously suggests an infrastructure event, not a model-specific bug).

The Kubernetes ServiceMonitor for this dashboard:

```yaml
# k8s/monitoring/vllm-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-metrics
  namespace: ml-serving
  labels:
    app: vllm
spec:
  selector:
    matchLabels:
      app: vllm-chatbot
  endpoints:
    - port: metrics          # port named "metrics" in the Service spec
      path: /metrics
      interval: 5s
      scrapeTimeout: 4s
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_name]
          targetLabel: pod
        - sourceLabels: [__meta_kubernetes_pod_node_name]
          targetLabel: node
  namespaceSelector:
    matchNames:
      - ml-serving
# The corresponding Service must expose the metrics port
apiVersion: v1
kind: Service
metadata:
  name: vllm-chatbot
  namespace: ml-serving
  labels:
    app: vllm-chatbot
spec:
  selector:
    app: vllm-chatbot
  ports:
    - name: http
      port: 8000
      targetPort: 8000
    - name: metrics            # ServiceMonitor references this by name
      port: 8000
      targetPort: 8000
```

With this ServiceMonitor, the Prometheus Operator automatically configures scraping when vLLM pods start, without manual scrape config updates. Combined with the alert rules from section 10, this is a complete observability stack for a production LLM serving deployment.

## 16. Key takeaways

1. **TTFT and TPOT are mechanically independent.** TTFT is compute-bound (prefill); TPOT is memory-bandwidth-bound (decode). They require different optimization techniques and should be monitored and alarmed separately.

2. **p99, not mean, is the SLA metric.** A healthy-looking mean latency can hide a catastrophic tail. Always use `histogram_quantile(0.99, ...)` in Prometheus, and configure fine-grained histogram buckets calibrated to your latency range.

3. **Queue depth is the leading indicator.** Set a queue depth alert (> 3 for 60 seconds) and it will fire 2–5 minutes before your p99 latency alert. This gives you a window to scale out before users notice.

4. **MFU, not GPU utilization %, is the true efficiency metric.** GPU utilization percent is misleading during memory-bandwidth-bound decode. MFU tells you the fraction of peak FLOPs you are capturing and directly predicts cost per token.

5. **Little's Law sizes your fleet.** $L = \lambda \times W$: required concurrency equals arrival rate times target E2E latency. This is the derivation behind every fleet sizing decision; do not skip it.

6. **SLO design is top-down.** Start from user experience requirements, translate to TTFT/TPOT targets, derive QPS capacity via Little's Law, size the fleet. Bottom-up SLO setting produces SLAs disconnected from user experience.

7. **The E2E formula governs budget allocation.** $\text{E2E} = \text{TTFT} + (N-1) \times \text{TPOT}$. For long responses, TPOT dominates. For short responses, TTFT dominates. Allocate your latency budget accordingly.

8. **Wire four distinct alert tiers.** Soft warning at 80% of SLO (no page), queue depth trigger (scale-out), hard alert at SLO breach (Slack page), critical alert at 2× SLO (on-call page). This prevents both alert fatigue and undetected SLA misses.

9. **KV cache utilization above 85% causes TPOT spikes.** When the KV cache is nearly full, the vLLM scheduler preempts requests to CPU swap, which causes unpredictable TPOT spikes. Alert on KV cache utilization, not just on TPOT directly.

10. **Benchmark at production concurrency.** Single-request latency benchmarks are useful for TTFT floor measurement only. Always benchmark at the concurrency level matching your p99 traffic scenario to get meaningful MFU and p99 TPOT numbers.

## Further reading

- [What is model serving: the SLO triangle](/blog/machine-learning/model-serving/what-is-model-serving) — the latency/throughput/cost triangle that every metric in this post maps to; foundational framing for why latency, throughput, and cost pull in opposite directions
- [Batching fundamentals: the latency-throughput trade-off](/blog/machine-learning/model-serving/batching-fundamentals-latency-throughput-tradeoff) — Little's Law applied to batch scheduling and the direct precursor to the queue depth analysis here; explains how batch formation latency itself consumes part of the TTFT budget
- [vLLM deep dive: continuous batching and PagedAttention](/blog/machine-learning/model-serving/vllm-deep-dive) — the internals that explain why vLLM's KV cache metrics behave as described in this post; how PagedAttention keeps KV cache utilization above 95% versus the 20–40% of static allocation
- [Observability for model servers: Prometheus, Grafana, and OpenTelemetry](/blog/machine-learning/model-serving/observability-for-model-servers) — the full observability setup that complements the metrics instrumentation introduced here; covers distributed tracing across multi-model pipelines
- [The model serving playbook](/blog/machine-learning/model-serving/the-model-serving-playbook) — the series capstone; ties together SLO design, fleet sizing, optimization techniques, and incident response into a complete decision framework
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", NeurIPS 2023 — the vLLM paper that defines "goodput" as the throughput metric that respects SLA constraints; the source for the throughput comparison numbers cited in the case studies section
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022 — the continuous batching paper; establishes the theoretical foundation for why TPOT scales with batch size under memory-bandwidth constraints
- Google SRE Book, Chapter 6 "Monitoring Distributed Systems" — the four golden signals framework applied throughout this post; the definitive reference for the latency/traffic/errors/saturation taxonomy
- Prometheus documentation: `histogram_quantile` function and histogram best practices — the reference for accurate percentile computation from histograms, including the interpolation assumptions that make bucket choice matter



*This post is part of the [Model Deployment & Serving: From Notebook to Production](/blog/machine-learning/model-serving/what-is-model-serving) series. Next up: [Model packaging and formats: ONNX, TorchScript, GGUF, and TensorRT engine plans](/blog/machine-learning/model-serving/model-packaging-and-formats) — the layer beneath the server that determines what the runtime actually loads.*

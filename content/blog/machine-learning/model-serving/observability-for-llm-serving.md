---
title: "Observability for LLM Serving: The Metrics, Dashboards, and Alerts That Actually Run the Service"
date: "2026-07-04"
publishDate: "2026-07-04"
description: "A principal engineer's guide to instrumenting a production LLM service — the vLLM signals that matter, why percentiles beat averages, how to read saturation from queue depth and KV-cache, the Prometheus and Grafana that surface it, and the multi-window burn-rate alerts that page before users churn."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "observability",
    "prometheus",
    "grafana",
    "vllm",
    "slo",
    "llm-serving",
    "opentelemetry",
    "monitoring",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/observability-for-llm-serving-1.webp"
---

The page came in at 3:12 a.m.: `p99 TTFT > 2s for 5 minutes`. You SSH into a serving node expecting a crash and find none. Every GPU reads 96% utilization on the ops dashboard, tokens-per-second is the highest it has been all week, and the process has not restarted. By every metric your web-services team taught you to watch — requests per second, error rate, CPU, memory, GPU utilization — the fleet is healthy. And yet the retry counter on the gateway is climbing like a rocket, a third of your users are staring at a spinner, and the customer-success channel is filling with angry screenshots. Nothing is broken. The dashboard is simply measuring the wrong things, and it is measuring them in the wrong way.

This is the defining trap of running large language models in production. The observability stack that works beautifully for a stateless REST API — the RED method (Rate, Errors, Duration), the USE method (Utilization, Saturation, Errors), a p50 latency line and an error-rate gauge — quietly lies to you the moment the workload becomes autoregressive token generation on a memory-bound accelerator. A generic dashboard treats a request as one atomic unit of work with one latency number. An LLM request is not one unit. It is a compute-bound *prefill* over the prompt followed by *N* memory-bound *decode* steps, each producing a single token, each contending for a finite pool of KV-cache memory, each interleaved with dozens of other requests in a continuously re-formed batch. A dashboard that reports one duration per request cannot see any of that structure, which means it cannot see the failure until it has already become a customer-facing outage.

By the end of this post you will be able to instrument an LLM service so the 3 a.m. page fires on the *cause* (KV-cache saturation, a growing wait queue) minutes before the *symptom* (a p99 latency breach) reaches a user — and so that when it does fire, you can open one dashboard and read the whole health of the fleet without guessing. We will cover the core LLM serving signals and why generic RED and USE miss them; the mechanics of why you must plot histogram percentiles, not averages; how to read saturation from queue depth and KV-cache utilization when GPU-utilization is actively misleading; the concrete Prometheus and Grafana that surface `vllm:*` metrics; multi-window burn-rate alerting that respects an error budget instead of spamming your on-call; the three telemetry pillars (metrics, logs, traces) tuned for LLMs; and the cardinality traps that will crater your Prometheus if you label a metric by user or prompt. This sits squarely on the serving SLO triangle — latency versus throughput versus cost — because observability is how you *see* where you are on that triangle in real time. It is the prerequisite for every other trade you will make. The whole telemetry pipeline, from the engine's raw counters up to the human who gets paged, is the stack in the figure below.

![Layered stack showing telemetry flowing from a vLLM engine metrics endpoint up through Prometheus, Grafana, Alertmanager, and finally an on-call engineer, with each layer transforming the signal.](/imgs/blogs/observability-for-llm-serving-1.webp)

If you are new to this series, start with [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the latency/throughput/cost framing, and [model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) for the definitions of TTFT, TPOT, and p99 that this post assumes you already have. Here we go one level deeper: not *what* the numbers mean, but *how to collect, plot, and alert on them so the system runs itself between pages.*

## Why a request is not one unit: where RED and USE break

The RED method says: for every service, measure the **R**ate of requests, the **E**rror rate, and the **D**uration (latency). The USE method says: for every resource, measure **U**tilization, **S**aturation, and **E**rrors. Both are excellent. Both are, for LLM serving, necessary but wildly insufficient — and worse, both have a specific failure mode where they report *green* while the service is on fire.

Start with Duration. A classic web request has one meaningful latency: time from the first byte in to the last byte out. You plot its p50 and p99 and you understand the user experience. An LLM streaming request has *at least two* latencies that a user feels completely differently, plus a third that most dashboards never capture:

- **TTFT (Time To First Token)** — how long the user stares at a blank screen before the first word appears. This is dominated by *prefill*: a single forward pass over all prompt tokens, which is compute-bound (it saturates the tensor cores) and grows with prompt length. It is also inflated by any time the request spent *waiting in the queue* before it was ever scheduled. TTFT is the latency of "did the machine hear me."
- **TPOT (Time Per Output Token)**, also called ITL (Inter-Token Latency) — the gap between successive tokens once generation starts. This is dominated by *decode*: one forward pass per token, memory-bandwidth-bound because each step must stream the entire model weights and the growing KV-cache from HBM. TPOT is the latency of "how fast does it talk." A user reading along at 5 tokens per second wants TPOT under ~200 ms; a coding assistant streaming into an editor wants it far lower.
- **End-to-end latency** — TTFT plus the sum of all the inter-token gaps plus network egress. For a 150-token response this is roughly `TTFT + 149 × TPOT`. It matters for non-streaming API consumers and for the tail of long generations.

A dashboard that plots a single "request duration" for an LLM is averaging prefill and decode and queue-wait into one meaningless number. A 4-second end-to-end latency could be a healthy 400 ms TTFT followed by smooth 24 ms/token decode over a long answer, or it could be a catastrophic 3.5-second TTFT (the request sat in a queue) followed by fine decode. The user experiences those two identically-durationed requests as *completely different products* — one is a fast assistant that writes a long answer, the other is a frozen box that eventually vomits text. You cannot tell them apart without splitting the signal.

Now Utilization. The USE method's "U" for a GPU is, on nearly every dashboard, `DCGM_FI_DEV_GPU_UTIL`. We will spend a whole section on why that number lies, but the short version: it reports the *fraction of the sample window during which at least one CUDA kernel was running*, not how much of the GPU's compute or memory bandwidth you are actually using. During memory-bound decode, a single tiny kernel can keep `GPU_UTIL` pinned at 100% while the tensor cores sit ~15% idle. So the metric your capacity planning is built on is green precisely when you are leaving the most throughput on the table, and it is *also* green when you are so overloaded that requests are being preempted and recomputed. It cannot distinguish "perfectly loaded" from "melting down," which makes it useless as the saturation signal USE demands.

The signals that actually describe an LLM serving system's health are a different, richer set. Each answers a specific operational question, each has a defensible target range, and each fires at a specific threshold. The figure below is the reference catalog — the seven signals that belong on every LLM serving dashboard, what each one tells you, where it should sit, and when it should page.

![A matrix mapping seven core LLM serving signals to what each tells you, its target range, its alert threshold, and which observability pillar it belongs to.](/imgs/blogs/observability-for-llm-serving-2.webp)

Walk the rows. **TTFT** and **TPOT** are your two latency SLIs, both plotted as histogram percentiles (more on why in the next section). **End-to-end latency** is a derived latency that pairs a metric with a trace so you can decompose a slow request. **Generation tokens per second** is your true throughput SLI — not requests per second, because one request can be 10 tokens or 10,000. Then the three that no web dashboard has ever shown you and that matter most: **queue depth** (`num_requests_waiting`) is load versus capacity, the single best leading indicator you have; **KV-cache utilization** (`gpu_cache_usage_perc`) is your memory headroom, the resource that runs out first on an LLM server; and **preemptions** (`num_preemptions_total`) is the counter that tells you the scheduler has started evicting and later recomputing requests because it ran out of KV-cache — the on-GPU equivalent of thrashing swap.

Here is the same catalog as a concrete reference table you can copy into a runbook, with the actual vLLM metric names, Prometheus types, and target ranges for a chat-style service on an H100. Treat the targets as starting points to tune against your own SLA, not universal constants.

| Signal | vLLM metric | Type | What it answers | Target range | Page threshold |
|---|---|---|---|---|---|
| TTFT | `vllm:time_to_first_token_seconds` | histogram | Prefill + queue wait | p99 < 500 ms | p99 > 800 ms for 5m |
| TPOT / ITL | `vllm:time_per_output_token_seconds` | histogram | Decode speed per token | p99 < 50 ms | p99 > 80 ms for 5m |
| E2E latency | `vllm:e2e_request_latency_seconds` | histogram | Full response time | p99 < 4 s | p99 > 6 s for 5m |
| Gen throughput | `vllm:generation_tokens_total` | counter | Useful output rate | > 2000 tok/s | < 800 tok/s sustained |
| Requests running | `vllm:num_requests_running` | gauge | Active batch size | 20–64 | context-dependent |
| Queue depth | `vllm:num_requests_waiting` | gauge | Load vs capacity | 0–3 waiting | > 10 for 2m |
| KV-cache util | `vllm:gpu_cache_usage_perc` | gauge | Memory headroom | 0.60–0.85 | > 0.95 for 1m |
| Prefix-cache hit | `vllm:gpu_prefix_cache_hit_rate` | gauge | Reuse of shared prefixes | > 0.30 | < 0.10 sustained (investigate) |
| Preemptions | `vllm:num_preemptions_total` | counter | Cache thrashing | ~0 / min | > 5 / min |
| Success/errors | `vllm:request_success_total` | counter | Completion outcomes | > 99.5% success | error rate > 1% |

The point of the table is not the exact numbers. The point is that a generic RED/USE dashboard captures at most three of these ten rows (rate, error, one duration) and is *structurally blind* to the four that predict every capacity incident you will ever have: queue depth, KV-cache utilization, preemptions, and the split of latency into TTFT versus TPOT. Instrumenting those four is the entire difference between a dashboard that pages you when it is already too late and one that pages you while you still have five minutes to act.

Two of these signals deserve a closer look because they are the ones most often misread. **Requests running versus requests waiting** (`num_requests_running` and `num_requests_waiting`) are not redundant — their *ratio and trend* is the whole story. A running count of 48 with 0 waiting means you are packed and keeping up. A running count of 48 with 180 waiting means the batch is full and a queue is forming behind it: the same "busy" GPU, a completely different health state. Plot them on the same panel, stacked, so the moment the waiting band rises above a flat running band you can see the queue form. The absolute running count also tells you your effective batch size, which is what actually drives throughput; if it is pinned well below your `--max-num-seqs`, then KV-cache — not the scheduler's sequence limit — is your binding constraint, and no amount of raising `max-num-seqs` will help.

**Prefix-cache hit rate** (`gpu_prefix_cache_hit_rate`) is the one signal on the list that is simultaneously a latency lever and a cost lever, which is why it belongs on the dashboard even though it rarely pages. Every chat request shares a long system prompt; every turn of a conversation shares the entire history so far. When vLLM's prefix cache is warm, the prefill for those shared tokens is skipped — the KV blocks are reused — which cuts TTFT and frees compute for other requests. A hit rate that collapses (say from 0.40 to 0.05) after a deploy is a strong signal that something invalidated the cache: a changed system prompt, a new template that shifted token boundaries, or a routing change that stopped pinning a conversation's turns to the same replica. You feel it as a TTFT regression and a throughput drop with no change in traffic, and the prefix-hit panel is what tells you *why* in one glance instead of an afternoon of bisecting deploys.

## The mechanics: why you plot percentiles, not averages

If you take one habit from this post, make it this: **never alert on, or make a capacity decision from, an average latency.** For an interactive LLM service the average is not merely less informative than the tail — it is actively deceptive, and the reason is arithmetic, not opinion.

An average is a sum divided by a count. It has no memory of shape. A fleet serving 10,000 requests where 9,900 finish TTFT in 150 ms and 100 finish in 1,850 ms has a mean TTFT of `(9900 × 150 + 100 × 1850) / 10000 ≈ 167` ms. That number looks *wonderful*. It is comfortably under a 500 ms SLO. And it is a lie: one request in a hundred — which, at 58 requests per second, is more than one unhappy user every two seconds, thousands per day — is waiting almost four times your SLO. The mean cannot see them because 100 slow requests contribute almost nothing to a sum dominated by 9,900 fast ones. This is the contrast the figure below makes concrete: the same fleet, read through a mean-only dashboard versus a percentile histogram.

![A before-and-after figure contrasting a mean-only dashboard reporting a healthy 190 ms against a histogram view exposing a hidden p99 of 1850 ms that breaches the SLO.](/imgs/blogs/observability-for-llm-serving-4.webp)

Percentiles have memory of shape because they are *order statistics*. The p99 is the value below which 99% of observations fall — by construction it is exactly the number that describes your unhappiest 1%. For an interactive product the tail *is* the experience: a user does not average their sessions, they remember the one time the assistant froze. Latency is also not symmetric. It has a hard floor (physics: you cannot prefill faster than the tensor cores allow) and a soft, heavy right tail (queueing, preemption, GC pauses, a noisy neighbor, a cold prefix cache). Heavy-tailed distributions are precisely where mean and median diverge most, and where the mean is dragged around by rare events you most need to catch.

There is a deeper reason percentiles matter for LLM serving specifically, and it comes from fan-out. Real requests are rarely a single model call. A RAG pipeline embeds the query, retrieves from a vector store, re-ranks, then calls the LLM — and an agent may make ten or twenty sequential model calls to answer once. If a single downstream call has a p99 of 1%, the probability that a request touching *n* such calls hits *at least one* p99-slow call is $P = 1 - (1-p)^n$. For a modest fan-out of ${n = 20}$ calls at $p = 0.01$, that is ${1 - 0.99^{20} \approx 0.18}$ — eighteen percent of user-visible responses inherit a tail-latency event from *some* backend, even though every individual backend is meeting its "99% fast" SLO. This is the tail-at-scale effect that Dean and Barroso named in their 2013 CACM article, and it is why a serving fleet with a great average and an unmonitored tail produces a product that feels randomly, maddeningly slow.

#### Worked example: how the average hid a real regression

A team ships a change that adds a synchronous safety-classifier call before generation. In load tests the mean TTFT moves from 160 ms to 190 ms — a 30 ms bump, waved through as noise. In production, the classifier occasionally cold-starts and takes 1.5 s. It fires on roughly 0.8% of requests. The new distribution: 99.2% of requests at ~165 ms, 0.8% at ~1,650 ms. Recompute the mean: `0.992 × 165 + 0.008 × 1650 ≈ 177` ms. The mean barely moved — from 160 to 177 — and every average-based dashboard and load-test gate reported success. But the p99 went from ~470 ms to ~1,650 ms, blowing through a 500 ms SLO for one in every 125 users. The regression was invisible to the mean and glaring in the p99. The only reason the team caught it before the SLO burned out its monthly error budget was that TTFT was instrumented as a Prometheus *histogram* and the dashboard plotted `histogram_quantile(0.99, ...)`, not `rate(sum) / rate(count)`.

That is the mechanical core of the whole discipline. A histogram is a set of counters, one per latency bucket (`_bucket{le="0.5"}`, `_bucket{le="1.0"}`, …), plus a `_sum` and a `_count`. Prometheus's `histogram_quantile()` estimates any percentile by interpolating within the bucket where the target rank falls. You get the tail for the price of a handful of counters, and — critically — you can aggregate across replicas *before* computing the quantile, which you cannot do with pre-computed per-pod percentiles (averaging percentiles is a statistical crime that produces numbers with no meaning). The cost is bucket resolution: your p99 is only as precise as your bucket boundaries near the tail, so pick buckets that straddle your SLO. vLLM's default TTFT buckets already cluster around the sub-second range where chat SLOs live; if you serve a workload with a very different latency profile, override them.

There is one honest limitation to name. `histogram_quantile` does not return the true p99; it returns a *linear interpolation within the bucket that contains the 99th-percentile rank*. If your p99 falls in a bucket spanning 0.8 s to 2.0 s, the reported value can be off by hundreds of milliseconds depending on where inside that bucket the real observations actually sit. The fix is bucket *placement*, not more buckets everywhere: cluster fine-grained boundaries around your SLO and around the tail you alert on, and leave the body of the distribution coarse. For a 500 ms TTFT SLO, boundaries at 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, and 5.0 seconds give tight resolution exactly where the alert threshold and the tail live, at a cost of only eight buckets per label combination. If you truly need exact tail quantiles rather than bucketed estimates — a latency-critical contract where 480 ms versus 520 ms is a billing difference — compute them from a high-dynamic-range (HDR) histogram or a t-digest inside the application and export the pre-computed quantiles as gauges, accepting that you then lose the ability to re-aggregate them across replicas. For nearly every serving SLO, well-placed Prometheus buckets are the right trade, and the re-aggregation property is worth more than the last few milliseconds of precision.

## Reading saturation: why GPU utilization lies, and what to read instead

Here is the section that would have saved you at 3:12 a.m. The mental habit from web serving is: high CPU means saturated, so high GPU-util means saturated. For LLM decode that inference is simply false, and believing it is how you end up staring at a 96%-utilized fleet that is dropping requests.

`DCGM_FI_DEV_GPU_UTIL` (and the older `nvidia-smi` "GPU-Util" it derives from) is defined as *the percentage of the last sample period during which one or more kernels was executing on the GPU*. Read that definition twice. It is a **duty-cycle** metric — was the GPU doing *anything* — not an **occupancy** or **throughput** metric. A single kernel that uses 2% of the streaming multiprocessors, launched every millisecond, pins `GPU_UTIL` to 100%. LLM decode is exactly this pattern: tiny per-token matrix-vector products that are bottlenecked on streaming weights from HBM, not on compute. The tensor cores are mostly idle; the memory bus is the bottleneck; and `GPU_UTIL` reports 100% the entire time. Meanwhile, the *same* metric reads ~96% both when your batch is a healthy 48 requests and when it is 48 requests with another 180 backed up behind them being preempted and recomputed. It cannot see the queue. It cannot see the KV-cache. It is blind to saturation by construction.

![A before-and-after figure showing a single GPU-utilization reading of 96 percent judged healthy on the left, contrasted with queue depth, KV-cache, and preemption counters on the right revealing a saturated fleet that must shed load.](/imgs/blogs/observability-for-llm-serving-6.webp)

So what *does* tell you an LLM server is saturated? Three signals, read together:

1. **Queue depth (`vllm:num_requests_waiting`).** The scheduler admits requests into the running batch until it runs out of KV-cache blocks; everything else waits. A waiting count that hovers at 0–3 means you have headroom. A waiting count that is *growing* means arrivals now exceed the rate at which the batch retires work — you have crossed capacity, and every newly admitted request will inherit the accumulated queue wait as pure TTFT. Queue depth is the *leading* indicator: it moves the instant you cross saturation, minutes before the latency histogram's tail fills in enough for a p99 alert to fire.
2. **KV-cache utilization (`vllm:gpu_cache_usage_perc`).** This is the fraction of paged KV-cache blocks currently allocated. It is the resource that runs out first on an LLM server, long before you exhaust FLOPs. At 0.60–0.85 you are efficiently packed. At >0.95 the scheduler has almost no room to admit new sequences or to grow existing ones, and it will start preempting.
3. **Preemptions (`vllm:num_preemptions_total`).** When KV-cache is exhausted, vLLM preempts a running sequence — evicting its KV blocks (recompute mode) or swapping them to CPU (swap mode) — and must redo that work later. A nonzero and *climbing* preemption rate is the unambiguous signal that you are over capacity and actively wasting compute. It is the LLM equivalent of a machine that has started thrashing swap: throughput can even *rise* on the dashboard while goodput collapses.

The relationship between these and latency is not vibes; it is queueing theory. **Little's Law** states that for a stable system, the average number of requests in the system equals the arrival rate times the average time each spends in the system: $L = \lambda W$. Rearranged, the time a request spends in the system is $W = L / \lambda$. When you are below capacity, $L$ (running + waiting) is small and roughly constant, so $W$ tracks pure service time and TTFT is stable. The moment $\lambda$ (arrivals) exceeds the batch's retirement rate, $L$ starts growing without bound — the waiting queue fills — and because $\lambda$ is now capped at capacity while $L$ climbs, $W$ climbs with it. Latency does not degrade *linearly* as you approach capacity; it goes to the knee of the M/M/1-style curve and then explodes. That explosion is what your 3 a.m. p99 alert saw. Queue depth saw it coming; the latency histogram only reported it after the fact.

Preemption itself has two modes worth distinguishing, because they cost differently. In *recompute* mode (vLLM's default), a preempted sequence's KV blocks are dropped and the sequence is re-prefilled from scratch when it is later rescheduled — cheap on memory, but it burns compute proportional to the sequence length generated so far, and it lands as a mid-generation stall on an already-in-flight request. In *swap* mode, the KV blocks are copied out to CPU memory and back, trading GPU memory pressure for PCIe bandwidth and adding latency on the swap-in. Either way, a climbing `num_preemptions_total` means the scheduler is doing the same work twice; the rate of that counter is a direct measure of wasted GPU-seconds and the cleanest early signal that you have crossed from "efficiently full" into "over capacity and thrashing."

#### Worked example: reading the knee from queue depth and KV-cache

An H100 node serves a 13B model with vLLM. At steady state it runs a batch of 48, KV-cache sits at 0.78, `num_requests_waiting` is 1–2, preemptions are zero, and p99 TTFT is 470 ms — comfortably inside a 500 ms SLO. A marketing push lands and arrival rate jumps from 40 to 58 requests per second. Watch the signals move in order: within 30 seconds KV-cache climbs from 0.78 to 0.97 as the scheduler crams in more sequences; at 60 seconds preemptions tick from 0 to 14 per minute as it starts evicting to make room; at 90 seconds `num_requests_waiting` jumps from 2 to 180 because new arrivals can no longer be admitted; and only at the two-minute mark does p99 TTFT — the metric your generic dashboard alerts on — cross from 470 ms to 2.1 s, because it took that long for enough slow requests to accumulate in the histogram's tail buckets. If your only alert is on p99 TTFT, you find out two minutes into the incident. If you alert on KV-cache > 0.95 and a growing wait queue, you find out at 30–60 seconds, with time for autoscaling or load-shedding to act before a single user sees 2 seconds. This is why saturation alerts belong on the cause, not the symptom. The full progression is the incident timeline later in this post.

For the deeper treatment of *acting* on these signals — admission control, load-shedding, and goodput-oriented autoscaling — see [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management). Observability is how you *see* the knee; that post is how you *stay off* it.

One more note on GPU metrics, because you should not throw `GPU_UTIL` away entirely — you should *demote* it and add better ones. DCGM exposes profiling metrics that measure real occupancy: `DCGM_FI_PROF_SM_ACTIVE` (fraction of SMs with at least one warp resident), `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (fraction of cycles the tensor pipes are busy), and `DCGM_FI_PROF_DRAM_ACTIVE` (memory-bandwidth duty cycle). For decode-heavy LLM serving you will typically see `GPU_UTIL` near 100%, `SM_ACTIVE` around 0.4–0.6, `TENSOR_ACTIVE` in the teens, and `DRAM_ACTIVE` high — a fingerprint that says *memory-bound, room in the tensor cores, throughput limited by HBM*. That fingerprint is what tells you whether a bigger batch or a different parallelism layout would help. `GPU_UTIL` alone tells you nothing.

## Building the dashboard: from Prometheus scrape to Grafana panel

Enough theory. Here is the concrete pipeline that turns `vllm:*` counters into the reference dashboard in the figure below. There are four moving parts: the engine exposes a `/metrics` endpoint; Prometheus scrapes it on an interval; Grafana queries Prometheus with PromQL and renders panels; Alertmanager evaluates rules and routes pages. We will build each part.

![A grid dashboard of twelve panels in three rows — latency, throughput, and saturation — showing TTFT p99 at 470 ms, KV-cache at 78 percent, and a GPU-utilization panel flagged as careful at 96 percent.](/imgs/blogs/observability-for-llm-serving-5.webp)

### Exposing and scraping the metrics

vLLM serves Prometheus metrics automatically from its OpenAI-compatible server. Launch it and the counters are live at `/metrics`:

```bash
# Start vLLM with the OpenAI-compatible API server; /metrics is exposed by default.
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama3-8b \
  --host 0.0.0.0 --port 8000 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --disable-log-requests           # keep per-request logs off the hot path

# Confirm the metrics endpoint is populated (labels elided for readability):
curl -s http://localhost:8000/metrics | grep -E '^vllm:(num_requests|gpu_cache_usage|num_preemptions|time_to_first_token)'
# vllm:num_requests_running{model_name="llama3-8b"} 48.0
# vllm:num_requests_waiting{model_name="llama3-8b"} 2.0
# vllm:gpu_cache_usage_perc{model_name="llama3-8b"} 0.78
# vllm:num_preemptions_total{model_name="llama3-8b"} 0.0
# vllm:time_to_first_token_seconds_bucket{model_name="llama3-8b",le="0.5"} 41234
```

On Kubernetes, the clean way to tell the Prometheus Operator to scrape this is a `ServiceMonitor` (or `PodMonitor`). It selects your vLLM `Service` by label and scrapes its metrics port. Note the scrape interval: 15 seconds is the sweet spot for LLM serving — fast enough that queue-depth and KV-cache spikes are visible within one or two samples, slow enough that you are not paying for cardinality you do not need.

```yaml
# servicemonitor-vllm.yaml — kube-prometheus-stack picks this up via its selector.
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm
  namespace: inference
  labels:
    release: kube-prometheus-stack   # must match the operator's serviceMonitorSelector
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: vllm
  namespaceSelector:
    matchNames: ["inference"]
  endpoints:
    - port: http                     # the named port on the vLLM Service
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
      # Keep only the series you actually chart/alert on. This is your first and
      # cheapest defense against a cardinality blow-up (see the last section).
      metricRelabelings:
        - sourceLabels: [__name__]
          regex: "vllm:.*|DCGM_FI_DEV_.*|DCGM_FI_PROF_.*"
          action: keep
```

### PromQL: the queries behind every panel

This is where the histogram mechanics pay off. To plot a true, fleet-wide p99 TTFT, you sum the per-bucket *rates* across all pods first, then apply `histogram_quantile`. Summing before quantiling is the only correct way to aggregate a percentile across replicas.

```promql
# p99 TTFT across the whole fleet, over a 5-minute window.
histogram_quantile(
  0.99,
  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))
)

# p99 TPOT (inter-token latency) — the "how fast does it talk" SLI.
histogram_quantile(
  0.99,
  sum by (le) (rate(vllm:time_per_output_token_seconds_bucket[5m]))
)

# Generation throughput in tokens/sec (your real throughput SLI, not req/s).
sum(rate(vllm:generation_tokens_total[1m]))

# Queue depth and running batch — the saturation leading indicators. Instantaneous
# gauges, so use max over the scrape to avoid smoothing away a spike.
max(vllm:num_requests_waiting)
max(vllm:num_requests_running)

# KV-cache utilization as a percentage (the gauge is a 0..1 fraction).
max(vllm:gpu_cache_usage_perc) * 100

# Preemption rate per minute — nonzero and climbing means over capacity.
sum(rate(vllm:num_preemptions_total[5m])) * 60

# Prefix-cache hit rate (reuse of shared system prompts / conversation prefixes).
avg(vllm:gpu_prefix_cache_hit_rate)

# Error ratio: fraction of requests NOT finishing with a normal stop reason.
sum(rate(vllm:request_success_total{finished_reason!="stop"}[5m]))
  / sum(rate(vllm:request_success_total[5m]))
```

Two subtleties bite people here. First, gauges like `num_requests_waiting` are point-in-time; if you scrape every 15 s and a spike lasts 5 s, an `avg_over_time` can smooth it into nonexistence. Use `max` (or `max_over_time`) for saturation gauges so a transient backlog is never hidden. Second, always divide by `le`-grouped bucket *rates*, never by raw `_bucket` counts — counters only carry meaning as rates, and `histogram_quantile` expects per-second bucket increments.

### The Grafana panel

Grafana panels are JSON. You will normally build them in the UI and export, but it is worth seeing the shape so you can template a whole dashboard as code. Here is a single time-series panel for p99 TTFT with the SLO drawn as a threshold — the red line the on-call watches.

```json
{
  "title": "TTFT p99 (fleet)",
  "type": "timeseries",
  "datasource": { "type": "prometheus", "uid": "${DS_PROM}" },
  "fieldConfig": {
    "defaults": {
      "unit": "s",
      "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 10 },
      "thresholds": {
        "mode": "absolute",
        "steps": [
          { "color": "green", "value": null },
          { "color": "orange", "value": 0.5 },
          { "color": "red", "value": 0.8 }
        ]
      }
    }
  },
  "targets": [
    {
      "expr": "histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))",
      "legendFormat": "p99 TTFT",
      "refId": "A"
    },
    {
      "expr": "histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))",
      "legendFormat": "p50 TTFT",
      "refId": "B"
    }
  ]
}
```

Lay the panels out the way the dashboard figure shows: a **latency row** (TTFT p99, TPOT p99, E2E p99, error rate), a **throughput row** (requests/s, prompt tok/s, generation tok/s, running-versus-waiting), and a **saturation row** (KV-cache %, prefix-hit %, preemptions/min, and a *demoted* GPU-util panel labeled "careful" so nobody mistakes it for health). Three rows, twelve panels, one screen, no scrolling. An on-call engineer paged at 3 a.m. should be able to read the entire fleet's state in five seconds: are we slow (row 1), are we busy (row 2), and *why* (row 3). If they have to scroll or cross-reference, the dashboard has failed its one job. The Grafana community and the vLLM project both publish reference dashboards you can import and adapt rather than build from scratch — start from one of those and prune it to these three rows.

### Synthetic probes: measure what the user actually feels

Every metric so far is *whitebox* — emitted by your own engine, which means it stops at the boundary of your own process. It cannot see the load balancer that added 200 ms, the TLS handshake, a region-to-region hop, or the fact that your engine is healthy but the gateway in front of it is returning 503s. For that you need *blackbox* monitoring: a synthetic client that issues real requests from outside the fleet on a fixed cadence and measures the end-to-end experience the way a user does.

For an LLM service the synthetic probe should exercise the *streaming* path, because TTFT is only meaningful over the real transport. A minimal probe issues a chat completion with streaming enabled, records the wall-clock time to the first streamed chunk (true end-to-end TTFT, including every network and gateway hop), and exports it as its own metric so you can compare it against the engine-reported TTFT — the gap between the two is pure infrastructure overhead.

```python
# blackbox_probe.py — run on a 15-30s schedule; export probe metrics to Prometheus.
import time, requests
from prometheus_client import Histogram, Counter

PROBE_TTFT = Histogram("probe_ttft_seconds", "External TTFT via streaming",
                       buckets=(0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0))
PROBE_FAIL = Counter("probe_failures_total", "Synthetic probe failures")

def probe(url, model):
    start = time.perf_counter()
    try:
        with requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": "ping: reply with one word"}],
            "stream": True, "max_tokens": 8,
        }, stream=True, timeout=10) as r:
            r.raise_for_status()
            for chunk in r.iter_lines():
                if chunk:                        # first streamed token has arrived
                    PROBE_TTFT.observe(time.perf_counter() - start)
                    return
    except Exception:                            # noqa: BLE001 — any failure counts
        PROBE_FAIL.inc()
```

Run the probe every 15–30 seconds from at least one location outside the cluster, ideally one per region you serve. Two alerts pay for it immediately: probe *failure* rate (the fleet is unreachable even though the engine process is up — a load-balancer, DNS, or certificate problem your whitebox metrics will never show) and a persistent *gap* between probe TTFT and engine TTFT (infrastructure latency creeping in between the user and your GPU). The probe is also your fastest deploy canary: a synthetic request that starts failing the instant a rollout begins is a quicker rollback trigger than waiting for real user traffic to accumulate in a burn-rate window.

## SLO burn-rate alerting: page before the users churn

A dashboard is something a human looks at. An alert is something that looks at the dashboard *for* you and wakes you when it matters. The hard part of alerting is not writing a rule that fires when latency is bad — that is trivial and it is also how you train your on-call to ignore the pager. The hard part is firing *only* when the badness is severe enough, or sustained enough, to be worth a human's night. The tool for that is **multi-window, multi-burn-rate** alerting on an error budget, straight from the Google SRE workbook, adapted to LLM latency SLIs.

Start with the SLO. Say your objective is "99% of requests have TTFT under 500 ms, measured over 30 days." That 1% is your **error budget**: over 30 days you are *allowed* to serve up to 1% of requests slower than 500 ms. A "bad" request is one whose TTFT exceeds the threshold. Because TTFT is a histogram, the fraction of good requests is directly computable from the bucket at your SLO boundary:

```promql
# SLI: fraction of requests with TTFT <= 500 ms, over the trailing window.
# The le="0.5" bucket counts requests at or under 500 ms; divide by the total.
sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[1h]))
  / sum(rate(vllm:time_to_first_token_seconds_count[1h]))
```

The **burn rate** is how fast you are spending the budget relative to the pace that would exactly exhaust it over the SLO window. A burn rate of 1 means you will use your entire 30-day budget in exactly 30 days — sustainable. A burn rate of 14.4 means you will use it 14.4× faster, exhausting the whole month's budget in about 2 days, or 2% of it in a single hour. The genius of the multi-window approach is pairing a **long window** (which gives statistical confidence that the badness is real, not a blip) with a **short window** (which ensures the badness is *still happening* right now, so you do not page for an event that already recovered). You page only when *both* windows show a high burn rate.

| Severity | Long window | Short window | Burn rate | Budget consumed | Action |
|---|---|---|---|---|---|
| Page (fast burn) | 1h | 5m | 14.4 | 2% in 1h | Wake on-call now |
| Page (medium burn) | 6h | 30m | 6 | 5% in 6h | Wake on-call |
| Ticket (slow burn) | 3d | 6h | 1 | 10% in 3d | File a ticket, business hours |

Encoded as a Prometheus `PrometheusRule`, the fast-burn page looks like this. The recording-rule pattern (compute the burn rate once, alert on it) keeps the alert expression readable and cheap.

```yaml
# prometheusrule-ttft-slo.yaml — multi-window burn-rate alerting for the TTFT SLO.
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-ttft-slo
  namespace: inference
  labels:
    release: kube-prometheus-stack
spec:
  groups:
    - name: ttft-slo.rules
      interval: 30s
      rules:
        # Error ratio = fraction of requests slower than the 500 ms SLO boundary,
        # recorded at both windows so the alert expression stays simple.
        - record: vllm:ttft_error_ratio:5m
          expr: >
            1 - (
              sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[5m]))
              / sum(rate(vllm:time_to_first_token_seconds_count[5m]))
            )
        - record: vllm:ttft_error_ratio:1h
          expr: >
            1 - (
              sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[1h]))
              / sum(rate(vllm:time_to_first_token_seconds_count[1h]))
            )
        # Fast-burn page: 14.4x the 1% budget in BOTH the 1h and 5m windows.
        # 0.01 * 14.4 = 0.144 error ratio.
        - alert: TTFTErrorBudgetFastBurn
          expr: >
            vllm:ttft_error_ratio:1h > (14.4 * 0.01)
            and
            vllm:ttft_error_ratio:5m > (14.4 * 0.01)
          for: 2m
          labels: { severity: page }
          annotations:
            summary: "TTFT SLO fast burn — 2% of monthly budget/hour"
            description: "TTFT >500ms on {{ $value | humanizePercentage }} of requests across 1h and 5m windows."
```

There is a subtlety specific to LLMs worth calling out. Because saturation (KV-cache, queue depth) is a *leading* indicator of latency, you should run *both* kinds of alert: a burn-rate alert on the latency SLI (the symptom, for accountability against your SLA) and a faster, cause-based alert on saturation (for lead time to act). A pragmatic pairing: page on the multi-window TTFT burn rate *and* fire a lower-severity "warning" the moment KV-cache exceeds 0.95 for one minute with a growing wait queue. The warning gives autoscaling and load-shedding a head start; the burn-rate page is the backstop that holds you honest to the customer contract. The figure below traces a real spike through exactly these signals, and shows why the cause-based warning wins you 90 seconds of lead time.

![A timeline of a latency-spike incident seen through five signals, showing KV-cache and preemptions rising 90 seconds before p99 TTFT breaches and the burn-rate alert pages on-call.](/imgs/blogs/observability-for-llm-serving-7.webp)

#### Worked example: sizing the fast-burn threshold

Your service does 58 requests/second, which is `58 × 3600 = 208,800` requests/hour. Your SLO is 99% under 500 ms over 30 days, so your monthly budget is 1% of `58 × 3600 × 24 × 30 ≈ 150M` requests, i.e. about 1.5M "slow-allowed" requests for the month. A burn rate of 14.4 means you would consume 2% of that budget — roughly 30,000 slow requests — in a single hour. At 208,800 requests/hour, 30,000 slow requests is an error ratio of `30000 / 208800 ≈ 0.144`, which is exactly the `14.4 × 0.01` in the rule. This is not a coincidence; it is the definition of burn rate made concrete. The practical upshot: the fast-burn page fires when about one in seven requests is breaching the SLO *and it is still happening in the last five minutes*. That is a real, sustained, customer-visible incident — worth a human's sleep — and not the transient 30-second blip that a naive `p99 > 500ms for 1m` rule would have paged on three times a night, training your on-call to silence the pager. Alert fatigue is itself an availability risk; the multi-window burn rate is how you buy signal without buying noise.

## The three pillars for LLMs: metrics, logs, and traces

Metrics tell you *that* something is wrong and roughly where. They cannot tell you *why* a specific request was slow, because a metric is an aggregate — it has thrown away the identity of individual requests to be cheap and fast. For root cause you need the other two pillars: **structured logs** (one event per request, richly annotated, queryable) and **distributed traces** (the causal, timed breakdown of one request across every hop). The discipline that makes them useful for LLM serving is *correlation*: every metric exemplar, every log line, and every trace span for a given request must carry the same `request_id`, so that when a burn-rate alert fires you can pivot from "p99 is bad" to "here are the exact 40 slow requests, here is what they had in common, here is the span that ate the time." The figure below shows a single request emitting all three pillars as it threads the gateway and engine.

![A branching graph of one chat request flowing through a gateway and vLLM engine to the GPU while fanning out Prometheus metrics, an OpenTelemetry trace, and a structured log line that share a request id.](/imgs/blogs/observability-for-llm-serving-3.webp)

The division of labor maps cleanly onto the three pillars, and the general principle of when to reach for which is covered in [metrics, logs, and traces: when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which). For LLM serving specifically:

- **Metrics** (Prometheus) — cheap, aggregate, always-on. Bounded cardinality (labeled by `model_name`, `route`, maybe `tenant_tier` — never by anything unbounded). This is your dashboard and your alerting substrate. Retain for weeks.
- **Traces** (OpenTelemetry) — per-request, sampled. A trace decomposes one request into spans: gateway routing, queue wait, prefill, decode, any tool/RAG calls. This is where you see that a slow request spent 1.8 s *waiting in the queue* versus 1.8 s *in decode* — a distinction metrics blur. Sample at 1–10% in steady state, but *always* trace requests that breach a latency threshold (tail-based sampling) so your slow requests are never the ones you dropped. For the deep treatment of trace-driven debugging, see [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving).
- **Logs** — one structured (JSON) event per finished request, carrying `request_id`, `model`, prompt and generation token counts, TTFT, TPOT, `finished_reason`, and error class. Unbounded fields (the user id, a prompt hash) live *here*, not in metric labels, because a log store is built for high-cardinality search and Prometheus is not.

Here is a FastAPI middleware that ties the three together for a gateway sitting in front of vLLM. It emits a Prometheus histogram for TTFT, opens an OpenTelemetry span, and writes one structured log line per request — all sharing a `request_id`. This is copy-and-adapt ready; swap the `call_vllm` stub for your streaming client.

```python
# gateway_middleware.py — one request, three correlated pillars.
import time, uuid, logging, json
from fastapi import FastAPI, Request
from prometheus_client import Histogram, Counter, make_asgi_app
from opentelemetry import trace
from opentelemetry.trace import SpanKind

# --- Metrics: bounded labels only. NEVER add user_id / prompt here. ---
TTFT = Histogram(
    "gateway_ttft_seconds", "Time to first token at the gateway",
    labelnames=["model", "route"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0),  # straddle the 500ms SLO
)
REQS = Counter(
    "gateway_requests_total", "Requests by outcome",
    labelnames=["model", "route", "outcome"],
)

tracer = trace.get_tracer("llm-gateway")
log = logging.getLogger("llm-gateway")
app = FastAPI()
app.mount("/metrics", make_asgi_app())  # Prometheus scrape target

@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    model = request.headers.get("x-model", "unknown")
    route = request.url.path
    start = time.perf_counter()
    first_token_at = None

    # One span per request; the trace_id links to the vLLM engine spans downstream.
    with tracer.start_as_current_span(
        "llm.request", kind=SpanKind.SERVER,
        attributes={"llm.request_id": request_id, "llm.model": model},
    ) as span:
        try:
            response = await call_next(request)          # streams tokens back
            first_token_at = getattr(response, "first_token_at", None)
            outcome = "success"
        except Exception as exc:                          # noqa: BLE001
            span.record_exception(exc)
            outcome = "error"
            REQS.labels(model, route, outcome).inc()
            raise
        finally:
            if first_token_at is not None:
                ttft = first_token_at - start
                TTFT.labels(model, route).observe(ttft)
                span.set_attribute("llm.ttft_seconds", round(ttft, 4))
            REQS.labels(model, route, "success").inc()
            # High-cardinality fields (request_id, token counts) go in the LOG,
            # never in a metric label — this is the cardinality firewall.
            log.info(json.dumps({
                "request_id": request_id,
                "trace_id": format(span.get_span_context().trace_id, "032x"),
                "model": model, "route": route,
                "ttft_seconds": None if first_token_at is None else round(first_token_at - start, 4),
                "total_seconds": round(time.perf_counter() - start, 4),
                "outcome": outcome,
            }))
    return response
```

The single most valuable thing this buys you is the pivot. When the burn-rate alert fires, you filter the log store for `ttft_seconds > 0.5` in the alert window, group by `model` and `route`, and instantly see whether the slow requests share a model version, a route, a tenant, or a prompt shape. Then you take a `trace_id` from one of them and open the trace to see *which span* — queue wait, prefill, a slow RAG retrieval — ate the budget. Metrics found the fire; logs found the room; the trace found the match. Without the shared `request_id` you have three disconnected data sources and a long night.

Prometheus **exemplars** are the glue that makes the metrics-to-traces jump one click instead of a manual log search: an exemplar attaches a `trace_id` to a specific histogram observation, so in Grafana you can click a spike on the p99 TTFT panel and jump straight to a trace of a request that was in that bucket at that moment. Turn them on; they cost almost nothing and they collapse your mean-time-to-diagnosis.

It is worth being concrete about the log schema, because a structured log is only as useful as its fields are consistent. Standardize one JSON event per finished request across every service in the request path, so that a single query joins them:

| Field | Example | Why it is here |
|---|---|---|
| `request_id` | `7f3a2c...` | The join key across all three pillars |
| `trace_id` | `4bf92f...` | One-click pivot to the full trace |
| `model` | `llama3-8b` | Group slow requests by model version |
| `route` | `/v1/chat/completions` | Isolate a bad endpoint |
| `tenant_tier` | `premium` | Fairness and per-tenant SLO analysis |
| `prompt_tokens` | `480` | Correlate slowness with prompt length |
| `generation_tokens` | `150` | Correlate cost and TPOT with output length |
| `ttft_ms` | `470` | The per-request latency, un-aggregated |
| `tpot_ms` | `24` | Decode speed for this specific request |
| `queue_ms` | `12` | How much of TTFT was pure queue wait |
| `finished_reason` | `stop` / `length` / `error` | Outcome classification |
| `error_class` | `TimeoutError` | Bounded error taxonomy — never the raw message |

Notice `queue_ms` broken out from `ttft_ms`: that one decomposition, logged per request, answers the question the aggregate metrics cannot — *was this request slow because it waited, or because it computed slowly?* — for any individual request after the fact. The trace carries the same decomposition as timed spans. A well-instrumented LLM request produces a span tree of `llm.request` (root) over `gateway.route`, `queue.wait`, `engine.prefill`, `engine.decode`, and any `tool.call` or `retrieval.query` children, each with attributes for token counts and model. Because span durations are causal and additive, one slow trace tells you exactly where the time went.

#### Worked example: root-causing one slow request from its trace

The burn-rate alert fires. You filter the log store for `ttft_ms > 800` over the alert window and get 40 requests — all `tenant_tier=premium`, all `route=/v1/chat/completions`, with `prompt_tokens` clustered around 6,000. You grab one `trace_id` and open the trace. The root span is 2.3 s. Its children: `gateway.route` 3 ms, `queue.wait` 40 ms, `engine.prefill` 2,180 ms, first-token `engine.decode` 70 ms. The prefill span alone is 2.18 seconds — this was *not* a queue or saturation problem, it was a genuinely enormous prompt being prefilled from scratch. Cross-check the prefix-hit panel: hit rate for that route just dropped to near zero. Root cause: a new premium feature started sending 6,000-token prompts with a per-user preamble placed *before* the shared system prompt, which defeated the prefix cache, so every request paid full prefill on 6,000 tokens instead of reusing cached blocks. No amount of scaling would have fixed it; the fix was to reorder the prompt so the shared preamble came first and could be cached, which cut prefill back to ~200 ms. The metrics found the fire (a TTFT burn), the logs found the room (premium, one route, long prompts), and the trace found the match (prefill, not queue). That is the three-pillar workflow earning back its instrumentation cost in a single incident.

## Cost observability: watching dollars per million tokens

The third axis of the SLO triangle is cost, and it is the one teams instrument last and regret first — usually when a finance review asks why the inference bill tripled and nobody can attribute the increase to a model, a tenant, or a code change. Cost is an operational signal, not just a monthly invoice, and it belongs on the same Prometheus that holds your latency and saturation.

The mechanics are a division. The dominant cost of GPU inference is the GPU-hour, so the cost of producing tokens is the GPU hourly rate divided by the tokens you actually deliver per hour:

$$ \text{cost per token} = \frac{C_{\text{gpu/hr}}}{3600 \cdot X_{\text{tok/s}}} $$

where $C_{\text{gpu/hr}}$ is the fully-loaded hourly cost of the GPU (on-demand price, or your amortized reserved or owned cost) and $X_{\text{tok/s}}$ is *goodput* — tokens per second that were actually delivered to a user within SLO, not raw throughput. That distinction is the whole point: a fleet that maxes GPU-util by admitting more work than it can serve within SLO produces tokens the user has already abandoned, and those tokens cost real money while earning nothing.

#### Worked example: the cost of chasing utilization

An H100 on-demand runs about \$3.50 per hour (list prices vary; use your own number). At a healthy generation rate of 2,100 tokens per second it produces `3600 × 2100 ≈ 7.56M` tokens per hour, so the cost is `3.50 / 7.56` dollars per million — about **\$0.46 per million generated tokens**. Now suppose an operator, watching the misleading GPU-util panel, pushes the batch past capacity to hold utilization at 99%. Throughput on the dashboard climbs to 2,400 tok/s, but preemptions and recomputation mean roughly 20% of those tokens belong to requests that already timed out and got retried — dead tokens nobody will ever read. Effective goodput is `2400 × 0.80 = 1920` tok/s, which delivers `3600 × 1920 ≈ 6.9M` tokens per hour, so the *user-visible* cost rises to about **\$0.51 per million tokens** — a 10% increase — while p99 latency simultaneously got worse. Chasing utilization made the service both slower and more expensive. This is exactly the failure the goodput-versus-throughput distinction warns about, and it is why cost per *delivered* token is the number to watch, not raw tokens per dollar.

You can approximate cost per million tokens directly in Prometheus with a recording rule, given a per-GPU hourly cost you inject as a constant series (or scrape from a cost exporter):

```promql
# gpu_hourly_cost is a series you define — one value per GPU model — via a small
# static exporter or a recording rule. Cost per 1M generated tokens, 10m window:
(
  sum(gpu_hourly_cost) / 3600
)
/
sum(rate(vllm:generation_tokens_total[10m]))
* 1e6
```

Two further cost signals earn their place on a mature dashboard. **Cost per request** — the rate above multiplied by mean generation tokens per request — lets product teams reason about unit economics per feature. **Cost per tenant tier** — the same rate split by the bounded `tenant_tier` label — tells you whether your free tier is subsidized by your premium tier or quietly bankrupting you. Both are cheap because they ride on bounded labels; neither needs per-user cardinality. Driving this number down with spot fleets, reserved capacity, and batch-versus-online routing is its own discipline, but you cannot optimize a number you do not measure — so measure it here, on the same pane of glass as latency and saturation, and make every capacity decision a three-way latency-throughput-cost trade with the data in front of you instead of a hunch.

## Cardinality traps: the labels that quietly kill Prometheus

Every strong recommendation so far has a failure mode, and for observability the failure mode is *self-inflicted denial of service via cardinality*. A Prometheus time series is uniquely identified by its metric name plus the full set of label key-value pairs. Every distinct combination of label values is a *separate series* that Prometheus holds in memory, indexes, and persists. Cardinality is multiplicative: a metric with a `model` label (5 values) and a `route` label (10 values) is 50 series — fine. Add a `user_id` label with a million values and it is 50 million series, and your Prometheus is dead — first the memory balloons, then the ingestion falls behind, then it OOM-kills and you have lost your observability at exactly the moment you need it. The figure below sorts the labels you will be tempted to add into the ones that are safe and the ones that are a loaded gun.

![A matrix classifying seven candidate metric labels by cardinality, series cost, and verdict, showing user id and prompt text as never-label while model, route, status, and tenant are cheap to keep.](/imgs/blogs/observability-for-llm-serving-8.webp)

The rule is simple and non-negotiable: **metric labels must be bounded and low-cardinality.** A label is safe if the set of its possible values is small and grows slowly. `model_name` (a handful), `route` (a few dozen), `status_code` (~15), `tenant_tier` (a few) — all fine, all *keep*. A label is a bomb if its value set is unbounded or per-user: `user_id`, `session_id`, `request_id`, `prompt` text or a hash of it, `conversation_id`, a raw error *message* (as opposed to an error *class*). Those belong in logs and trace attributes, which are built for high-cardinality search, never in a metric label.

| Candidate label | Cardinality | In metrics? | Put it in |
|---|---|---|---|
| `model_name` | ~5–20 | Yes — core label | metric + trace + log |
| `route` / endpoint | ~10–50 | Yes — core label | metric + trace + log |
| `status` / `finished_reason` | ~15 | Yes — core label | metric + trace + log |
| `tenant_tier` | ~3–5 | Yes — core label | metric + trace + log |
| `user_id` | millions | **Never** | log field, trace attribute |
| `session_id` | 100k+/day | **Never** | trace attribute, exemplar |
| `prompt` text / hash | unbounded | **Never** | trace attribute (truncated), log |
| raw `error_message` | unbounded | **Never** | log; expose an `error_class` label instead |

There are a few softer traps worth naming, because they bite teams that already know the hard rule:

- **Bucket explosion.** Every histogram bucket is a series *per label combination*. A histogram with 20 buckets and a `model × route × tenant` label set of 200 combinations is 4,000 series for that one metric. Keep bucket counts modest (10–14 well-chosen boundaries beat 30 evenly-spaced ones) and prune label combinations that no dashboard or alert reads.
- **Churning labels.** A label whose values *rotate* over time — a pod name in a fleet that autoscales constantly, a build SHA on every deploy — creates unbounded series *over time* even if the instantaneous cardinality is small. Prometheus keeps churned series until they age out of retention. Prefer stable labels; let orchestration metadata live in relabel configs you can drop.
- **Over-alerting as a cardinality-of-attention problem.** Cardinality is not only a machine cost. An alert that fires on every 30-second p99 blip floods the human's attention the same way a `user_id` label floods Prometheus's memory. Every page you send that did not need a human erodes trust in the ones that do. The multi-window burn rate from earlier is the discipline that keeps *alert* cardinality bounded, just as the label rules keep *series* cardinality bounded.

The defense-in-depth is: enforce bounded labels in your instrumentation code (as the middleware's comment insists), then add a `metricRelabelings` `keep` rule in the `ServiceMonitor` (as shown earlier) as a backstop that drops any stray high-cardinality series before Prometheus ingests it, and finally set a per-target sample limit so a misbehaving exporter can never take the whole system down. Three layers, because the day someone adds `labelnames=["user_id"]` "just to debug something" is the day your monitoring dies if you have only one.

## Case studies and benchmarks: what production systems actually expose

Theory is cheap; here is what real serving stacks emit, cited so you can verify.

**vLLM's Prometheus metrics.** The vLLM engine exposes the full `vllm:*` family described throughout this post — `time_to_first_token_seconds`, `time_per_output_token_seconds`, `e2e_request_latency_seconds`, `num_requests_running`, `num_requests_waiting`, `gpu_cache_usage_perc`, `num_preemptions_total`, `prompt_tokens_total`, `generation_tokens_total`, and prefix-cache counters — all Prometheus-native and all labeled by `model_name`. The vLLM documentation ships a reference Grafana dashboard (`examples/production_monitoring`) that wires these into panels; it is the fastest way to a working dashboard, and it lays out exactly the latency/throughput/saturation split recommended here. The important design lesson from vLLM's own instrumentation is that it exposes `num_requests_waiting` and `num_preemptions_total` as first-class metrics *precisely because* the maintainers know these are the saturation signals that GPU-utilization cannot express.

**Text Generation Inference (TGI).** Hugging Face's TGI exposes a parallel set under `tgi_*`: `tgi_request_inference_duration`, `tgi_request_queue_duration` (queue time as its own histogram — a gift, because it isolates the queue-wait component of TTFT), `tgi_request_mean_time_per_token_duration` (TPOT), `tgi_batch_current_size`, and `tgi_queue_size`. TGI's decision to break out `queue_duration` separately validates the whole thesis of the saturation section: the framework authors consider queue wait important enough to instrument on its own, not fold into a single latency number. If you serve on TGI, alert on `tgi_queue_size` as your leading indicator exactly as you would on vLLM's `num_requests_waiting`.

**Triton, Ray Serve, and KServe.** The pattern is universal across serving stacks. NVIDIA Triton's metrics endpoint exposes `nv_inference_request_duration_us`, `nv_inference_queue_duration_us` (again, queue time as a first-class metric), and `nv_inference_count` per model and version — the same queue-versus-compute decomposition under different names. Ray Serve exports replica-level queue and latency metrics through its Prometheus integration, and its autoscaler reads the very queue-depth signal this post argues for. KServe, as a control plane, aggregates these and adds request-count and latency metrics at the InferenceService level. Every serious serving stack instruments queue time separately from inference time, because its authors learned the same lesson independently: a single latency number hides the one component — queue wait — that predicts saturation.

**NVIDIA DCGM and the utilization caveat.** The `dcgm-exporter` is the standard way to get GPU metrics into Prometheus on Kubernetes. Its documentation is explicit that `DCGM_FI_DEV_GPU_UTIL` is a coarse duty-cycle metric and that the `DCGM_FI_PROF_*` profiling metrics (`SM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE`) give true occupancy. NVIDIA's own guidance is to use the profiling metrics for performance analysis — the industry has known for years that raw GPU-util is misleading, yet it remains the default panel on most dashboards because it is the one metric that has always been there. Demote it; do not delete it.

**The tail-at-scale result.** The quantitative backbone of the percentiles section is Dean and Barroso, "The Tail at Scale" (Communications of the ACM, 2013). Their measured example — a service where a single leaf's 99th-percentile latency of 10 ms turns into a *root* 99th-percentile of 140 ms once a request fans out to 100 leaves — is the empirical proof that tail latency compounds with fan-out and that averages are useless for interactive systems. Every RAG and agent pipeline is a fan-out; the result applies directly.

**Multi-window burn-rate alerting.** The alerting design is chapter 5 of the Google *Site Reliability Engineering Workbook* ("Alerting on SLOs"), which introduces the exact 14.4×/6×/1× multi-window ladder used above. Google's contribution was to formalize *why* pairing a long and short window gives you both precision (few false pages) and recall (you still catch real burns), and *why* alerting on budget burn rate beats alerting on raw threshold crossings. It is the difference between an on-call rotation that trusts its pager and one that has muted it.

## When to use this (and when not to)

Everything in this post is the *right* amount of observability for a service with a real SLA and real users. It is emphatically *not* the right amount for every situation, and over-instrumenting is a genuine cost — in engineering time, in Prometheus footprint, and in the attention tax of dashboards nobody reads. Here is the honest decision boundary.

**Build the full stack when:** you have an interactive, user-facing LLM service with a latency SLA; you run more than a handful of QPS; you have an on-call rotation; or you are multi-tenant (where one tenant's traffic can starve another and you need per-tenant-tier visibility). At that point the four LLM-specific signals — queue depth, KV-cache, preemptions, TTFT/TPOT split — and multi-window burn-rate alerting pay for themselves the first time they catch a saturation event before it becomes an outage.

**Do not build all of it when:** you are running a batch or offline inference job (no user is waiting, so TTFT and burn-rate alerts are meaningless — measure throughput and cost per token and stop); you are a single-tenant internal tool with three users and no SLA (a `/metrics` scrape and a p99 panel is plenty; a multi-window burn-rate ladder is theater); or you are still pre-product-market-fit and the honest bottleneck is shipping, not five-nines. In those cases, instrument the *cheap* signals (the vLLM `/metrics` are free — scrape them) and skip the tracing pipeline and the alert ladder until you have a user who will be angry when it is slow.

**Specific anti-patterns to avoid regardless of scale:**

- **Alerting on averages or on raw p99-threshold crossings.** You will page on blips, your on-call will mute the pager, and the pager will be muted the night it matters. Use burn rate.
- **Labeling metrics by anything unbounded.** `user_id`, `session_id`, `prompt` — these kill Prometheus. They go in logs and traces. Enforce it in code *and* with a relabel backstop.
- **Trusting GPU-util as your saturation signal.** It is a duty cycle, not an occupancy. Read queue depth and KV-cache for saturation; keep `GPU_UTIL` as a demoted, clearly-labeled panel.
- **Tracing 100% of requests at high QPS.** It is expensive and mostly redundant. Sample low in steady state, but tail-sample every request that breaches the latency SLO so your slow requests are always captured.
- **A dashboard that requires scrolling to read fleet health.** If the on-call cannot answer "are we slow, are we busy, and why" in five seconds, the dashboard has failed. Three rows: latency, throughput, saturation.

The through-line of the whole series applies here too: every one of these choices is a point on the latency-throughput-cost triangle. Observability does not move you on the triangle — it *shows you where you are*, in real time, so that every other technique in this series (batching, disaggregation, autoscaling, quantization) is a decision you make with data instead of a guess you make in the dark.

## Key takeaways

1. **A request is not one unit.** Split every LLM latency into TTFT (prefill + queue) and TPOT (decode). One "duration" number blends compute-bound, memory-bound, and queue-wait latencies that users experience as completely different products.
2. **Percentiles, never averages.** The mean has no memory of shape and is dragged around by rare events. Instrument latency as a Prometheus histogram, sum bucket rates across replicas, and compute `histogram_quantile(0.99, ...)`. Fan-out makes the tail compound: at 20 backend calls of p99=1%, ~18% of user responses inherit a slow event.
3. **GPU utilization lies.** `DCGM_FI_DEV_GPU_UTIL` is a duty cycle, not an occupancy — it reads ~100% during memory-bound decode and cannot see a backed-up queue. Read saturation from `num_requests_waiting`, `gpu_cache_usage_perc`, and `num_preemptions_total`, and add `DCGM_FI_PROF_*` for true occupancy.
4. **Alert on the cause for lead time, on the SLI for accountability.** KV-cache and queue depth move 60–90 seconds before p99 latency does. Fire a warning on saturation for the head start; page on a multi-window burn rate against the latency SLO to stay honest without spamming.
5. **Multi-window burn-rate beats threshold alerts.** Pair a long window (confidence it is real) with a short window (confidence it is still happening). The 14.4×/6×/1× ladder pages on sustained, customer-visible incidents and stays quiet on 30-second blips.
6. **Three pillars, one `request_id`.** Metrics find the fire, logs find the room, the trace finds the match. Correlate all three with a shared request id and turn on Prometheus exemplars so the metrics-to-trace jump is one click.
7. **Cardinality is a loaded gun.** Metric labels must be bounded: `model`, `route`, `status`, `tenant_tier` — never `user_id`, `session_id`, or `prompt`. Enforce it in instrumentation code, backstop it with a `ServiceMonitor` keep-rule, and cap samples per target.
8. **One screen, five seconds.** A working LLM dashboard has three rows — latency, throughput, saturation — and answers "are we slow, are we busy, why" without scrolling.

## Further reading

- **vLLM documentation — production metrics and monitoring** (`docs.vllm.ai`, and `examples/production_monitoring` in the vLLM repo): the authoritative list of `vllm:*` metrics and a reference Grafana dashboard you can import and prune to the three-row layout.
- **Google, *The Site Reliability Engineering Workbook*, Chapter 5, "Alerting on SLOs"** (O'Reilly / sre.google): the multi-window, multi-burn-rate methodology, with the 14.4×/6×/1× ladder and the reasoning behind precision-versus-recall in alert design.
- **Jeffrey Dean and Luiz André Barroso, "The Tail at Scale," Communications of the ACM 56(2), 2013:** the empirical case for percentiles over averages and the arithmetic of how tail latency compounds under fan-out.
- **NVIDIA DCGM Exporter documentation and DCGM feature reference** (`docs.nvidia.com/datacenter/dcgm`): the definition of `DCGM_FI_DEV_GPU_UTIL` as a duty cycle and the `DCGM_FI_PROF_*` profiling metrics that measure true SM, tensor, and memory occupancy.
- **Prometheus documentation — histograms, `histogram_quantile`, exemplars, and cardinality** (`prometheus.io/docs`): the mechanics of bucketed histograms, why you aggregate bucket rates before quantiling, and the operational limits that make label cardinality a hard constraint.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) for the SLO triangle, [model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) for signal definitions, [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management) for acting on saturation, and [tracing and debugging LLM serving](/blog/machine-learning/model-serving/tracing-and-debugging-llm-serving) for the trace-driven root-cause workflow. For the general metrics-versus-logs-versus-traces decision, see [metrics, logs, and traces: when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which).

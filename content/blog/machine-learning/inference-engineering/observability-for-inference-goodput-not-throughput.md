---
title: "Observability for inference goodput, not throughput"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "Instrument every request, token, queue, cache lookup, and preemption so an inference dashboard predicts collapse before users feel it."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "observability",
    "goodput",
    "latency",
    "throughput",
    "batching",
    "kv-cache",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 27
---

The dashboard says 118,000 output tokens per second. The on-call channel says customers are timing out. Both can be true. A server can produce a great deal of text while spending most of its capacity on requests that waited too long, were preempted and recomputed, or never delivered a useful token before their deadline.

![A request branches from queue wait into GPU work and either useful tokens or a timeout](/imgs/blogs/observability-for-inference-goodput-not-throughput-1.webp)

The diagram above is the mental model: every request has a queue clock, a compute clock, and a useful-output clock. The first two explain the third. By the end of this post, you will have a concrete metric contract, a `nanoserve/observability.py` diff, token-level accounting, TTFT and TPOT histograms, cache and preemption counters, and a dashboard that spots the queueing collapse from [admission control and backpressure](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse) before the p99 alert becomes an incident.

This is post #51 in [Inference Engineering](/blog/machine-learning/inference-engineering/what-inference-engineering-is). It sits at the product edge of the same engine we have been building: weights and kernels create capacity, the KV cache and scheduler decide who gets it, decoding turns capacity into tokens, and observability tells us whether those tokens were useful. The capstone will assemble those decisions in [the inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook).

## 1. Throughput is a production metric with a blind spot

Throughput is attractive because it is easy to aggregate. Count output tokens over wall-clock seconds and you have a number that usually moves smoothly. It is also the wrong first question for an interactive endpoint. A token emitted after a request's deadline is not equivalent to a token emitted inside the deadline. A request that generates 500 tokens before cancellation may have consumed substantial GPU time while delivering zero business value.

Goodput is the useful part of service capacity. We will use the following explicit, explanatory definition throughout this post, not as a claim that a particular paper or serving engine mandates one universal formula:

$$
G_{\text{useful}} = \frac{\sum_{i=1}^{N} \mathbf{1}[\text{request } i \text{ meets its SLO}] \cdot \text{delivered output tokens}_i}{\text{measurement window}}
$$

Here, $N$ is the number of completed or expired requests in the window. The indicator is one when the request meets the endpoint's declared objective, such as TTFT below 500 ms and completion below 5 seconds; it is zero otherwise. You can also define request goodput as successful requests per second. Choose one contract and put the contract in the metric name or dashboard description. Otherwise an engineer will compare useful tokens per second with raw output tokens per second and call both “goodput.”

| Metric | Numerator | What it answers | Source |
|---|---:|---|---|
| Output throughput | all emitted output tokens | How much text left the GPU? | derived: token counter divided by window seconds |
| Request goodput | SLO-compliant completed requests | How many users got an acceptable result? | explanatory abstraction: SLO indicator |
| Useful-token goodput | output tokens from SLO-compliant requests | How much acceptable text did we deliver? | explanatory abstraction: SLO indicator |
| Work efficiency | useful output tokens / all generated tokens | How much work survived cancellation and deadlines? | derived: counters from the trace |

The first useful operational change is therefore a split: report raw throughput beside SLO goodput, never instead of it. The ratio is often more informative than either absolute number:

$$
\eta_{\text{SLO}} = \frac{G_{\text{useful}}}{T_{\text{output}}}
$$

If a dashboard shows 120,000 output tokens per second and only 71% of requests meet the objective, the useful-token approximation is $120{,}000 \times 0.71 = 85{,}200$ SLO-qualified tokens per second. That arithmetic is illustrative and derived from the displayed values; it is not a measurement of this repository.

### What should count as a success?

Do not bury the SLO in a Grafana variable. Put it in the request record. A chat endpoint might define success as first token within 800 ms and completion within 8 seconds. A batch extraction endpoint might care about completion time and valid JSON, not TTFT. A code completion endpoint may care about first useful line and cancellation rate. The same engine can serve all three, but one unqualified “goodput” number cannot describe them.

| Workload slice | Primary objective | Useful output | Dangerous aggregate |
|---|---|---|---|
| Chat, short input / long output | TTFT and completion deadline | tokens streamed before deadline | mean output tok/s |
| RAG, long input / short output | TTFT budget | complete answer within deadline | GPU utilization |
| Code completion | fast first token and cancellation-aware completion | accepted prefix | generated tokens after client disconnect |
| Translation | stable end-to-end latency | completed translation | batch throughput alone |

Source for the workload names: the series prompt suite. The objective mapping is an engineering policy that must be agreed with the product owner.

## 2. Define the request clock before you define the histogram

TTFT means time to first token: submit time to the timestamp at which the first output token becomes available to the client. TPOT means time per output token, commonly the average interval between consecutive output tokens after the first. ITL, inter-token latency, is the individual interval. The vLLM team documents the same distinction in [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), dated 2025-09-05, and uses TTFT, ITL, and TPOT in its latency benchmark descriptions.

![A matrix separates TTFT, TPOT, and SLO pass rate across four prompt workloads](/imgs/blogs/observability-for-inference-goodput-not-throughput-2.webp)

The histogram is more useful than the average because the tail tells us whether a small fraction of requests are experiencing a different mechanism. A prompt that waited behind chunked prefill has a queue-heavy TTFT. A decode batch with a long-context request may have a compute-heavy TPOT. These should not be folded into one latency bucket.

### Timestamp boundaries

Use a monotonic clock for durations. Wall-clock time can jump when NTP corrects the system clock. Record wall time separately for correlation with logs. The minimum trace has these boundaries:

```python
# nanoserve/observability.py
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic_ns, time_ns
from typing import Optional


def mono_ms() -> float:
    return monotonic_ns() / 1_000_000


@dataclass
class RequestTrace:
    request_id: str
    received_ms: float = field(default_factory=mono_ms)
    queue_enter_ms: Optional[float] = None
    admitted_ms: Optional[float] = None
    prefill_start_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    finish_ms: Optional[float] = None
    client_disconnect_ms: Optional[float] = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    cache_hit_tokens: int = 0
    preemptions: int = 0
    deadline_ms: Optional[float] = None
    output_token_times_ms: list[float] = field(default_factory=list)
    wall_received_ns: int = field(default_factory=time_ns)

    def mark_queue(self) -> None:
        self.queue_enter_ms = mono_ms()

    def mark_admitted(self) -> None:
        self.admitted_ms = mono_ms()

    def mark_prefill(self) -> None:
        self.prefill_start_ms = mono_ms()

    def mark_token(self, token_id: int) -> None:
        now = mono_ms()
        self.output_token_times_ms.append(now)
        self.generated_tokens += 1
        if self.first_token_ms is None:
            self.first_token_ms = now

    def mark_finish(self) -> None:
        self.finish_ms = mono_ms()

    @property
    def queue_ms(self) -> Optional[float]:
        if self.admitted_ms is None:
            return None
        return self.admitted_ms - (self.queue_enter_ms or self.received_ms)

    @property
    def compute_to_first_ms(self) -> Optional[float]:
        if self.first_token_ms is None or self.admitted_ms is None:
            return None
        return self.first_token_ms - self.admitted_ms

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_ms is None:
            return None
        return self.first_token_ms - self.received_ms

    @property
    def tpot_ms(self) -> Optional[float]:
        times = self.output_token_times_ms
        if len(times) < 2:
            return None
        return (times[-1] - times[0]) / (len(times) - 1)

    def met_slo(self) -> bool:
        if self.first_token_ms is None:
            return False
        if self.deadline_ms is not None and (self.finish_ms or mono_ms()) > self.deadline_ms:
            return False
        return True
```

This is a diff on the toy engine, not an implementation of an existing library. `token_id` is accepted even though the first version only needs the count; retaining it makes a later token audit possible. Do not put token text in the default trace. Token text is sensitive, and its cardinality makes logs expensive. Store IDs only under an explicit debugging policy, and hash or redact prompts at the edge.

### TTFT decomposition

For a request that is not cancelled, the elapsed first-token time can be decomposed as:

$$
\text{TTFT} = T_{\text{queue}} + T_{\text{admission}} + T_{\text{prefill}} + T_{\text{first-decode}} + T_{\text{transport}}
$$

This is an explanatory accounting identity. It is not a new law of model inference; it simply partitions the trace. The terms should add approximately to the observed client-side time. “Approximately” matters because instrumentation itself and network buffering create small gaps. If the difference is large, that is a signal: either a boundary was missed or a host synchronization is hiding in the compute term.

## 3. Count tokens at every boundary

Token accounting is where a good dashboard becomes an engine diagnostic. Count prompt tokens presented to the model, prompt tokens reused from cache, prompt tokens actually prefetched, generated tokens sampled, generated tokens delivered, and generated tokens discarded after cancellation. These are not interchangeable.

![A trace grid follows one request from receipt through cache lookup, prefill, decode, finish, or timeout](/imgs/blogs/observability-for-inference-goodput-not-throughput-6.webp)

The identity we want is:

$$
T_{\text{prefill}} = T_{\text{prompt}} - T_{\text{cache-hit}} + T_{\text{recompute}}
$$

where $T_{\text{recompute}}$ includes work repeated after preemption or an invalidated prefix. A cache hit is not simply “the prefix cache returned a block.” Count hit tokens, not only hit events. A request with one hit block and one million miss tokens is not a high-hit request.

```python
# nanoserve/observability.py, token accounting
from dataclasses import dataclass


@dataclass
class TokenLedger:
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    prefill_tokens: int = 0
    generated_tokens: int = 0
    delivered_tokens: int = 0
    discarded_tokens: int = 0
    recomputed_tokens: int = 0

    def record_prompt(self, prompt_tokens: int, cache_hit_tokens: int) -> None:
        if prompt_tokens < 0 or not 0 <= cache_hit_tokens <= prompt_tokens:
            raise ValueError("cache hits must be within the prompt")
        self.prompt_tokens += prompt_tokens
        self.cache_hit_tokens += cache_hit_tokens
        self.prefill_tokens += prompt_tokens - cache_hit_tokens

    def record_recompute(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("recomputed tokens cannot be negative")
        self.recomputed_tokens += tokens

    def record_sample(self, tokens: int = 1) -> None:
        if tokens < 0:
            raise ValueError("sampled tokens cannot be negative")
        self.generated_tokens += tokens

    def record_delivery(self, tokens: int) -> None:
        if tokens < 0 or tokens > self.generated_tokens - self.delivered_tokens:
            raise ValueError("cannot deliver tokens that were not sampled")
        self.delivered_tokens += tokens

    def cancel(self) -> None:
        self.discarded_tokens = self.generated_tokens - self.delivered_tokens

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def useful_fraction(self) -> float:
        return self.delivered_tokens / self.generated_tokens if self.generated_tokens else 0.0


ledger = TokenLedger()
ledger.record_prompt(prompt_tokens=512, cache_hit_tokens=256)
ledger.record_sample(128)
ledger.record_delivery(96)
ledger.cancel()
print(ledger.prefill_tokens, ledger.cache_hit_rate, ledger.discarded_tokens)
# expected output: 256 0.5 32
```

The expected output is reader-reproducible arithmetic: 512 minus 256 is 256 prefill tokens, 256 divided by 512 is 0.5, and 128 minus 96 is 32 discarded tokens. It is not a benchmark.

### Cache hit rate is a workload property

The cache hit rate belongs on a dashboard with prefix length, tenant, route, and prompt family. A global hit rate can rise because a popular short prefix is reused while long RAG prompts miss. Report both token-weighted and request-weighted rates:

$$
H_{\text{tokens}} = \frac{\sum_i h_i}{\sum_i p_i}, \qquad H_{\text{requests}} = \frac{\sum_i \mathbf{1}[h_i > 0]}{N}
$$

For four requests with prompt lengths 100, 100, 1,000, and 1,000 tokens and hit lengths 100, 0, 0, and 1,000, token-weighted hit rate is $(100 + 1{,}000) / 2{,}200 = 0.5$. Request-weighted hit rate is $2 / 4 = 0.5$ as well. Change the last hit to 100 tokens and the request-weighted rate remains 50%, while token-weighted rate becomes $200 / 2{,}200 \approx 9.1\%$. This is why the dashboard should show both.

## 4. Queue time is the early-warning signal

The GPU can be busy and the service can still be unhealthy. A decode scheduler may keep the device occupied while requests wait for KV blocks, while a long prefill monopolizes a scheduling step, or while preempted sequences repeatedly recompute. Average GPU utilization sees activity, not progress.

![A timeline shows offered load moving from a healthy queue to a knee and then a latency collapse](/imgs/blogs/observability-for-inference-goodput-not-throughput-3.webp)

Little's law gives the first diagnostic relationship:

$$
L = \lambda W
$$

$L$ is average work in the system, $\lambda$ is arrival rate, and $W$ is average time in the system. If 8 requests per second are admitted and the average in-system time is 250 ms, the expected average number of in-flight requests is $8 \times 0.25 = 2$. If the same arrival rate produces 2.4 seconds of average residency, the in-flight average becomes $8 \times 2.4 = 19.2$. That is derived arithmetic, not a capacity claim. In practice, burstiness and changing service time make the instantaneous value noisy, but the direction is still useful.

Queue time should have at least p50, p90, p99, and max. Add queue depth and oldest request age. The oldest age is particularly valuable when the average looks harmless but one request is stuck behind a resource or priority boundary.

```python
# nanoserve/observability.py, bounded histogram export
from collections import Counter


class LatencyHistogram:
    def __init__(self, boundaries_ms: tuple[float, ...]) -> None:
        if sorted(boundaries_ms) != list(boundaries_ms):
            raise ValueError("boundaries must be sorted")
        self.boundaries_ms = boundaries_ms
        self.counts = Counter()
        self.total = 0

    def observe(self, value_ms: float) -> None:
        if value_ms < 0:
            raise ValueError("latency cannot be negative")
        bucket = next((b for b in self.boundaries_ms if value_ms <= b), "+Inf")
        self.counts[bucket] += 1
        self.total += 1

    def prometheus(self, name: str, labels: str = "") -> str:
        lines = []
        cumulative = 0
        for boundary in self.boundaries_ms:
            cumulative += self.counts[boundary]
            lines.append(f'{name}_bucket{{le="{boundary}",{labels}}} {cumulative}')
        cumulative += self.counts["+Inf"]
        lines.append(f'{name}_bucket{{le="+Inf",{labels}}} {cumulative}')
        lines.append(f'{name}_count{{{labels}}} {self.total}')
        return "\\n".join(lines)


ttft = LatencyHistogram((50, 100, 250, 500, 1000, 2000, 5000))
for value in (120, 180, 700, 700, 2400):
    ttft.observe(value)
print(ttft.total, ttft.counts[250], ttft.counts["+Inf"])
# expected output: 5 2 0
```

The buckets are deliberately explicit. Do not use automatically generated floating-point buckets for an SLO dashboard; the boundary at 500 ms needs to be stable across deployments. Use native Prometheus histograms or OpenTelemetry histograms in a production service, and keep the same bucket schema across replicas so aggregation is meaningful.

### The collapse dashboard

The most valuable panel from post #15 is not “GPU utilization.” It is a four-line overlay: queue p99, oldest request age, preemptions per minute, and SLO goodput. Queue p99 rising first says admission or service rate is the problem. Preemptions rising after it says memory pressure is turning waiting into recomputation. Goodput falling while raw tokens stay flat is the signature that the server is doing work users will not count.

## 5. Separate compute time from waiting and transport

TTFT alone tells you that a request was slow. The split tells you which owner should be paged. Queue time belongs to the scheduler and admission gate. Prefill time belongs to prompt length, cache hit, attention backend, and chunk size. Decode time belongs to batch shape, KV bandwidth, sampling, synchronization, and any preemption-induced replay. Transport time belongs to the server runtime, proxy, network, and client.

<figure class="blog-anim">
<svg viewBox="0 0 760 220" role="img" aria-label="A request token moves from queue into compute while a wait bar grows or shrinks" style="width:100%;height:auto;max-width:860px">
<style>
.og-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.og-t{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.og-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.og-dot{fill:var(--accent,#6366f1)}.og-bar{fill:var(--accent,#6366f1);opacity:.2}@keyframes og-flow{0%,20%{transform:translateX(0);opacity:.3}45%,70%{transform:translateX(260px);opacity:1}90%,100%{transform:translateX(520px);opacity:.2}}@keyframes og-wait{0%,20%{transform:scaleX(1.8);transform-origin:left}45%,100%{transform:scaleX(.35);transform-origin:left}}.og-m{animation:og-flow 8s ease-in-out infinite}.og-w{animation:og-wait 8s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.og-m,.og-w{animation:none}.og-m{transform:translateX(260px)}.og-w{transform:scaleX(.8);transform-origin:left}}
</style>
<rect class="og-box" x="40" y="70" width="150" height="80" rx="10"/><rect class="og-box" x="305" y="70" width="150" height="80" rx="10"/><rect class="og-box" x="570" y="70" width="150" height="80" rx="10"/>
<text class="og-t" x="115" y="105">queue</text><text class="og-t" x="380" y="105">GPU work</text><text class="og-t" x="645" y="105">useful tokens</text>
<text class="og-s" x="115" y="130">wait time</text><text class="og-s" x="380" y="130">prefill + decode</text><text class="og-s" x="645" y="130">goodput</text>
<rect class="og-bar og-w" x="40" y="178" width="150" height="12" rx="6"/><circle class="og-dot og-m" cx="40" cy="110" r="10"/>
</svg>
<figcaption>The request marker moves through queue and compute while the wait bar contracts; goodput is the useful result at the end.</figcaption>
</figure>

The animated figure is intentionally motion-based: the marker crosses the queue boundary while the wait budget contracts. A static architecture box would not show why the same request can be healthy at low load and late at high load.

For a real WebP dashboard figure, the corresponding static split is:

$$
T_{\text{residency}} = T_{\text{queue}} + T_{\text{compute}} + T_{\text{transport}} + T_{\text{other}}
$$

Do not subtract two large client timestamps and call the remainder GPU time. Instrument around the scheduler admission, CUDA launch sequence, first sampled token, and stream flush. CUDA work is asynchronous. If the host reads a tensor or synchronizes at the wrong boundary, the synchronization cost moves into whichever span happens to contain it.

```python
# bench/measure_decode.py
import time
import torch


def measure_gpu_step(step, warmup: int = 20, samples: int = 100) -> tuple[float, float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(samples):
        start.record()
        step()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    elapsed.sort()
    return elapsed[len(elapsed) // 2], elapsed[int(0.99 * (len(elapsed) - 1))]


if __name__ == "__main__":
    x = torch.randn((8, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    p50, p99 = measure_gpu_step(lambda: torch.mm(x, w))
    print(f"GPU step p50={p50:.3f} ms p99={p99:.3f} ms")
```

This script is labeled reader-reproducible. On an RTX 4090, the output should be in a hardware-, driver-, and shape-dependent range; the useful expected result is p99 greater than or equal to p50, not a fabricated millisecond target. A reader should report the exact GPU model, driver, PyTorch version, dtype, shape, warmup, and sample count with the result. For service tests, use a load generator and record both the client timestamp and server spans.

| Span | Start | End | Owner | Source |
|---|---|---|---|---|
| Queue | queue enter | scheduler admission | admission / scheduler | derived: trace boundaries |
| Prefill | prefill launch | first decode-ready state | model and attention path | derived: trace boundaries |
| Decode | first decode step | final sampled token | engine and sampler | derived: token timestamps |
| Transport | server flush | client receive | proxy / network | reproduce: distributed trace |

## 6. Cache hits and preemptions explain “mysterious” regressions

The KV cache is a memory of keys and values already computed for earlier tokens. Prefix caching can avoid recomputing shared prompt prefixes, but a hit is useful only if its blocks are actually reused before eviction. Count lookup attempts, hits, hit tokens, miss tokens, evictions, and bytes copied. Add labels for cache tier and model route, not raw prompt text.

The cache hit rate should be plotted against TTFT. A falling hit rate with flat prompt length predicts prefill growth. A stable hit rate with rising TTFT points elsewhere, often queueing. A high hit rate with rising TPOT can mean the cache saved prefill but decode is now too heavily batched or memory-bound.

Preemption counters are the other half. Each preemption should record its reason: KV block pressure, priority request, deadline policy, explicit cancellation, or worker restart. Increment a request counter and a global counter. Record recomputed tokens when the request resumes. “Preempted requests” without “recomputed tokens” cannot tell you the work multiplier.

![A layered scoreboard connects workload, request latency, token counters, cache state, scheduler pressure, and GPU capacity](/imgs/blogs/observability-for-inference-goodput-not-throughput-4.webp)

```python
# nanoserve/observability.py, counters with bounded labels
from collections import defaultdict


class EngineCounters:
    def __init__(self) -> None:
        self.values = defaultdict(int)

    def inc(self, name: str, value: int = 1, **labels: str) -> None:
        if value < 0:
            raise ValueError("counter increments must be non-negative")
        # Only controlled labels may reach this method in production.
        key = (name, tuple(sorted(labels.items())))
        self.values[key] += value

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {"name": name, "labels": dict(labels), "value": value}
            for (name, labels), value in sorted(self.values.items())
        ]


counters = EngineCounters()
counters.inc("kv_lookup_total", route="chat")
counters.inc("kv_hit_tokens_total", value=256, route="chat")
counters.inc("preemptions_total", route="chat", reason="kv_pressure")
counters.inc("recomputed_tokens_total", value=64, route="chat")
print(counters.snapshot())
```

The label rule is more important than the class. Never label metrics by request ID, user ID, prompt hash, or arbitrary exception text. Those labels create unbounded time series. Put those fields in traces or sampled logs. Prometheus metrics need bounded dimensions such as route, model revision, GPU pool, scheduler policy, and preemption reason.

### What preemption does to goodput

Suppose a request needs 1,000 prompt tokens, is preempted once after 400 tokens, and resumes by recomputing those 400 tokens. The engine did $1{,}000 + 400 = 1{,}400$ prefill-token units, but the request's logical prompt still has 1,000 tokens. The recomputation ratio is $400 / 1{,}000 = 40\%$. That is derived from the trace. If 10 requests each incur the same replay, the service consumed 4,000 extra token units that output throughput will not identify.

## 7. Build histograms that survive aggregation

Histograms have two common failure modes: buckets do not contain the SLO boundary, or labels fragment the series so badly that a fleet-wide quantile is meaningless. Keep the bucket set small and intentional. Suggested TTFT buckets in milliseconds are 25, 50, 100, 250, 500, 800, 1,000, 2,000, 5,000, and infinity. Suggested TPOT buckets depend on the product contract; include the value at which streaming becomes visibly unpleasant.

![A before-and-after comparison shows raw token throughput improving while SLO goodput falls, then admission control restoring useful output](/imgs/blogs/observability-for-inference-goodput-not-throughput-5.webp)

The before-and-after figure is a warning against optimizing the wrong numerator. Increasing batch size may increase raw token throughput but extend queue time enough to lower SLO pass rate. The right question is not “did tok/s increase?” but “did useful tokens per second increase at the target SLO?”

```python
# bench/loadgen.py, open-loop request arrival skeleton
import asyncio
import random
import time


async def poisson_arrivals(rate_per_second: float, send_request) -> None:
    if rate_per_second <= 0:
        raise ValueError("rate must be positive")
    while True:
        delay = random.expovariate(rate_per_second)
        await asyncio.sleep(delay)
        asyncio.create_task(send_request(received_at=time.monotonic()))


async def run(rate_per_second: float, duration_s: float, send_request) -> None:
    task = asyncio.create_task(poisson_arrivals(rate_per_second, send_request))
    await asyncio.sleep(duration_s)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

This is a reader-reproducible load pattern, not a performance result. Open-loop load keeps sending according to the offered rate even when the server slows. Closed-loop load waits for a response before issuing the next request and therefore hides queue growth. Use open-loop to find the knee, then use a realistic closed-loop client to understand user behavior. Label the test mode in every result.

### The measurement protocol

Warm the model, tokenizer, CUDA graphs, allocator, and prefix cache separately. Discard warmup samples. Synchronize before timing host-visible boundaries. Use CUDA events around GPU work. Lock clocks where the platform allows it and record power mode, driver, CUDA, PyTorch, model revision, dtype, prompt suite, output cap, and concurrency. Run enough time to include scheduler churn, not only a handful of requests.

| Experiment choice | Recommended value | Why it changes the conclusion | Source |
|---|---|---|---|
| Load mode | open loop for capacity knee | exposes queue growth | derived: queueing test design |
| Warmup | model, cache, allocator, graphs | removes cold-start mixture | reproduce: `bench/loadgen.py` protocol |
| GPU timing | CUDA events plus synchronize boundaries | avoids asynchronous timing error | cited: [PyTorch CUDA Event documentation](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html), accessed 2026-08-03 |
| Reporting | p50, p90, p99, max, goodput | tails and SLO pass are visible | cited: [vLLM latency metrics](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05 |
| Prompt suite | chat, RAG, code, translation | workload mix changes cache and decode | series protocol: derived test design |

## 8. Practical dashboard design

The dashboard should let an on-call engineer answer five questions in under a minute:

1. Are users waiting before admission?
2. Is the GPU taking longer to produce the first token or each later token?
3. Are cache misses increasing the work per request?
4. Is preemption causing recomputation?
5. Is raw throughput diverging from SLO goodput?

![A diagnostic tree branches from TTFT p99 into queue, compute, cache, admission, prefill, decode, and preemption causes](/imgs/blogs/observability-for-inference-goodput-not-throughput-7.webp)

Put the following panels on one page, with the same time range and model-pool filters:

### Row one: user outcome

Show SLO goodput, request success rate, timeout rate, cancellation rate, and raw output tokens per second. Add a ratio panel for useful-token efficiency. The display should make it impossible to mistake a rising raw counter for a healthy service.

### Row two: latency distribution

Show TTFT and TPOT histograms, each split by route and prompt family. Use heatmaps over time for p50 and p99. A line chart of p99 without a distribution hides multimodality; a histogram without time hides the incident start.

### Row three: decomposition

Show queue p50/p99, prefill compute p50/p99, decode TPOT p50/p99, transport, queue depth, and oldest request age. Stack queue and compute only as a visual aid; keep the raw spans available in a trace link.

### Row four: engine pressure

Show KV allocated blocks, free blocks, cache hit tokens, cache miss tokens, eviction rate, preemptions by reason, recomputed tokens, and active sequence count. Add the ratio of recomputed to logical prompt tokens.

### Row five: hardware context

Show HBM allocated and reserved, achieved memory bandwidth if available, GPU utilization, kernel launch or host-step duration, CPU utilization, tokenizer queue, and network flush latency. Hardware panels explain symptoms; they are not the product SLO.

### Alert rules

Alert on symptoms and causes separately. A strong first alert is “SLO goodput below target for five minutes.” A diagnostic alert is “queue p99 above 40% of TTFT SLO and rising for three minutes.” A safety alert is “preemptions per minute above baseline and recomputed-token ratio above 5%.” The exact thresholds are product policy. Do not invent a universal 5% or 40% threshold; derive them from the latency budget and the workload baseline.

```yaml
# deploy/alerts/inference-goodput.yaml
groups:
  - name: inference-goodput
    rules:
      - alert: InferenceGoodputLow
        expr: rate(inference_slo_qualified_tokens_total[5m])
              < 0.85 * rate(inference_output_tokens_total[5m])
        for: 5m
        labels:
          severity: page
        annotations:
          summary: "Useful token efficiency is falling"
          description: "Compare queue p99, cancellations, and preemptions before changing kernels."
      - alert: InferenceQueueTailGrowing
        expr: histogram_quantile(0.99, sum by (le, route) (rate(inference_queue_ms_bucket[5m])))
              > 0.4 * inference_ttft_slo_ms
        for: 3m
        labels:
          severity: ticket
        annotations:
          summary: "Queue time consumes the TTFT budget"
          description: "Check admission, chunked prefill, and oldest request age."
```

The alert expressions are templates. The 0.85 and 0.4 values are explicit policy examples, not measurements or universal defaults. Put the actual SLO in a recording rule or configuration so the alert and the dashboard use the same contract.

## 9. Worked examples: follow the arithmetic

#### Worked example: a healthy-looking token counter

A one-minute window reports 120,000 raw output tokens per second. There are 600 completed requests. TTFT and completion SLOs pass for 71% of them. If we use the explanatory useful-token approximation, then:

$$120{,}000 \times 0.71 = 85{,}200 \text{ SLO-qualified tokens/s}$$

The 34,800-token-per-second difference is not necessarily wasted GPU work: some requests may be valid but outside this particular objective, and some tokens may have been delivered before a later completion deadline. That is why the ledger must retain per-request reason codes rather than applying a single global discount. The arithmetic is derived from the example values.

#### Worked example: cache hits do not guarantee fast TTFT

Take a 512-token prompt with 256 cached tokens. Logical prefill work is $512 - 256 = 256$ tokens. If the request waits 400 ms in the scheduler and prefill plus first decode takes 120 ms, the server-side TTFT decomposition is:

$$400\text{ ms queue} + 120\text{ ms compute} = 520\text{ ms before transport}$$

The cache saved half the prompt tokens, but the request can still miss an 800 ms objective once transport and serialization are included. Cache hit rate is a capacity signal, not a promise of user latency.

#### Worked example: preemption turns memory pressure into extra work

Suppose 32 requests each have 1,024 logical prompt tokens. One preemption per request replays 256 tokens. Logical prompt work is $32 \times 1{,}024 = 32{,}768$ token units. Replay adds $32 \times 256 = 8{,}192$ units. Effective prefill work is $32{,}768 + 8{,}192 = 40{,}960$, a replay overhead of $8{,}192 / 32{,}768 = 25\%$. This is derived. The dashboard should show the overhead directly because raw prompt-token counters otherwise look normal.

## 10. Case studies and real numbers

### The vLLM metric split

The vLLM team defines TTFT, ITL, and TPOT as distinct latency measures in its [vLLM anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), dated 2025-09-05. Its latency benchmark setup includes 32 input tokens and 128 output tokens with batch size 8, while its throughput benchmark uses 1,000 ShareGPT samples at infinite request rate. The important lesson is not a headline number; it is that “latency benchmark” and “throughput benchmark” are different experiments. A dashboard should preserve that distinction by tagging load mode, prompt length, output cap, and concurrency.

### The queue that looked like idle GPU

The series post [Admission control, backpressure and latency collapse](https://example.com) describes the production-shaped failure we care about here: p99 rises while GPU utilization remains high enough to reassure someone who is looking at the wrong panel. In this repository, the verified cross-link is [admission control and backpressure](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse). The observability lesson is to plot queue age and goodput next to utilization. The cited post's mechanism is queue growth plus preemption and recomputation; the dashboard implementation in this post makes those terms independently visible.

### Prefix caching and the missing denominator

The vLLM grounding reference describes prefix caching as hashing complete blocks and finding the longest cache hit. It also notes that a partial prefix recomputes the remainder of a block. That distinction is why a cache-hit event counter is weak. If the block size is 16 tokens, a hit that ends 3 tokens before a block boundary does not save the same work as a hit that covers 16 complete tokens. The vLLM team's [Inside vLLM post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), 2025-09-05, is the cited source for the block-level mechanism; the token-weighted formulas above are derived instrumentation guidance.

### A production engine's scheduler boundary

The vLLM V1 architecture post says the scheduler represents decisions as a request-to-token-count mapping, allowing chunked prefill, prefix caching, and speculative decoding to share a policy. That is a useful contrast for `nanoserve`: our first implementation can keep explicit queue and decode boundaries because they make the trace legible. A mature engine may collapse those distinctions internally, but it still needs observable semantic boundaries: admitted tokens, computed tokens, sampled tokens, and delivered tokens. Cite [vLLM V1: A Major Upgrade](https://vllm.ai/blog/2025-01-27-v1-alpha-release), dated 2025-01-27, when comparing the architecture; do not turn its cited throughput claims into a benchmark of this toy engine.

## 11. Failure modes in the instrumentation itself

Observability can become a second production problem. High-cardinality labels can exhaust the metrics backend. Per-token logs can cost more CPU and network than the sampler. A synchronous export path can turn a small request into a host stall. Sampling every request can hide rare timeouts if the sample decision happens before the failure path.

### Sampling policy

Keep a complete lightweight trace for every request: IDs, counters, timestamps, outcome, and bounded labels. Sample detailed token IDs, prompt hashes, and scheduler snapshots. Always retain traces for SLO misses, cancellations, preemptions, and unusually high queue age. This gives the incident path higher fidelity than the healthy path without logging private content by default.

### Clock and stream errors

Use monotonic host time for spans and CUDA events for device spans. Do not call `torch.cuda.synchronize()` on the hot path merely to make a metric exact. Place synchronization in the measurement harness or use asynchronous event collection. If a metric is eventually consistent, name it that way and document its lag.

### Counter resets and deployment changes

Prometheus counters reset on process restart; rates handle this if the metric remains a counter. Include model revision, engine version, scheduler policy, and GPU pool in labels with bounded cardinality. Deployments that change histogram buckets make long-term quantile comparisons unreliable, so version the metric name if the bucket contract changes.

### SLO gaming

If success means “first token arrived,” an engine can stream one token and then stall. If success means “request completed,” a client cancellation may look like a server failure even when the client changed its mind. Define success from the user contract, record cancellation separately, and show partial delivery. A goodput metric is only as honest as its outcome taxonomy.

## 12. When to reach for this, and when not to

Build this observability layer when you operate a shared GPU service, mix prompt shapes, use prefix caching or preemption, or need to explain why a throughput optimization hurt interactive latency. It is especially valuable before enabling a new scheduler policy, cache tier, quantization mode, or speculative decoder. Instrument first, then optimize.

Do not build a bespoke metrics stack for a one-off offline generation script. Use the model runner's existing counters and a wall-clock benchmark when there is one request, no queue, and no product SLO. Do not add token-level tracing to every production request if a bounded ledger plus failure-triggered detail is sufficient.

Use vLLM rather than `nanoserve` when the goal is a production engine with broad model, kernel, batching, prefix-caching, and distributed support. The point of `nanoserve` is to make the causal chain inspectable. The point of vLLM is to carry the engineering surface you should not recreate merely to obtain a dashboard. Borrow the metric contract even when you borrow the engine.

## Key takeaways

1. Report raw output throughput beside SLO-qualified goodput; neither replaces the other.
2. Record monotonic timestamps at receipt, queue entry, admission, prefill, first token, every output token, finish, and disconnect.
3. Define TTFT, ITL, and TPOT explicitly and histogram them separately.
4. Split TTFT into queue, compute, first decode, and transport so an owner can be found.
5. Count cache-hit tokens, miss tokens, evictions, and recomputed tokens; a hit event is not enough.
6. Count sampled, delivered, and discarded tokens to expose cancellation waste.
7. Treat preemption as extra work and publish recomputed-token ratio.
8. Use open-loop load to find the queueing knee; closed-loop load hides it.
9. Keep metric labels bounded and put sensitive request details in sampled traces.
10. Make the dashboard's primary question “did users get useful output inside the objective?”

## Further reading

- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), vLLM, 2025-09-05.
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release), vLLM, 2025-01-27.
- [PyTorch CUDA Event documentation](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html), accessed 2026-08-03.
- [Admission control, backpressure and latency collapse](/blog/machine-learning/inference-engineering/admission-control-backpressure-and-latency-collapse).
- [Request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption).
- [Continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention).
- [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).

## Appendix: the contract I would put in the engine review

Before merging a scheduler or kernel change, require one trace fixture and one load result. The fixture should contain a short prompt, a prefix hit, three sampled tokens, one cancellation, and a finish path. Assert that prompt tokens equal cache-hit tokens plus prefill tokens, sampled tokens equal delivered plus discarded tokens, and TTFT is not negative. Then assert that a preemption increments both the request counter and recomputed tokens. These are cheap invariants, but they catch the class of bugs where the dashboard looks healthy because one branch forgot to increment a counter.

The load result should include a manifest, not a screenshot: model revision, tokenizer revision, GPU name, driver, CUDA, PyTorch, dtype, engine commit, scheduler settings, cache settings, prompt suite, offered rate, concurrency, warmup duration, test duration, and histogram buckets. Store the raw per-request summary with prompt text removed. A later engineer should be able to reproduce the shape of the result without guessing which “batch size” meant scheduler slots, concurrent HTTP calls, or active decode sequences.

For each change, compare distributions rather than only means. A p50 improvement with a p99 regression is a production trade, not a free win. A raw token-throughput improvement with a lower useful-token efficiency is a regression for an interactive endpoint. A cache-hit improvement with higher preemption rate may be a memory policy failure. The point of this post is to make these comparisons routine: every optimization should say what happened to the request clock, the token ledger, the cache ledger, and the SLO outcome.

That discipline also prevents a common organizational failure. The kernel owner sees achieved bandwidth, the scheduler owner sees batch occupancy, and the product owner sees timeouts. They are each looking at a true projection of the same system. The request trace is the join key that lets those projections meet. Without it, each team can improve its local graph while the end-to-end goodput line gets worse.

Keep a small set of golden traces in version control with synthetic IDs and no customer text. They are fixtures for dashboard queries as much as unit tests are fixtures for code. When a field is renamed, the fixture should fail loudly instead of producing a blank panel. A dashboard that cannot explain its own missing data is not an observability system; it is decoration.

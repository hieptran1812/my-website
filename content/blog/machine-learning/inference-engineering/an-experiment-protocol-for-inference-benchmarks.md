---
title: "An experiment protocol for inference benchmarks: Measure the system you actually built"
date: "2026-08-03"
publishDate: "2026-08-03"
description: "A reusable nanoserve-oriented protocol for load generation, warmup, timing, provenance, and honest inference economics."
tags: ["inference-engineering", "llm-inference", "benchmarking", "latency", "throughput", "batching", "ml-systems", "pytorch", "gpu"]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 35
---

An inference benchmark can be perfectly reproducible and still answer the wrong question. A closed-loop client that waits for every response may report a comfortable p99 while a real open-loop traffic stream is building an invisible queue. A cold first request may include model loading and CUDA graph capture, while the next request does not. A tokenizer change can turn “the same prompt” into a different number of input tokens. A single throughput number can hide that half the requests violated the latency objective.

The diagram above is the mental model: a benchmark is a small measurement system with controlled setup, a deliberate load controller, a fixed workload, synchronized clocks, request-level observations, and an evidence record. This post turns that model into a reusable protocol and a `bench.py` template for `nanoserve`. It is designed to be copied by the experiments that follow this series, not to manufacture a heroic result for this post. The protocol is also the natural next step after [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is), where the engine is framed as weights, kernels, cache, scheduler, decoder, and API.

The six static diagrams map the protocol from lifecycle to scoreboard. The animated figure later makes one distinction physical: open-loop arrivals keep adding work while closed-loop clients wait for completion. That difference is often the entire explanation for a p99 disagreement.

> A benchmark is not a stopwatch around `generate()`. It is a controlled claim about a workload, a machine, a load process, and an observation window.

## 1. The benchmark is a measurement system

![A benchmark lifecycle separates setup, warmup, steady state, summary, and provenance artifacts](/imgs/blogs/an-experiment-protocol-for-inference-benchmarks-1.webp)

The first rule is to name the experiment before naming the number. Write down the model identifier, revision or commit, tokenizer identifier, engine commit, Python and CUDA versions, GPU name, driver, dtype, tensor-parallel configuration, maximum input and output lengths, scheduler limits, arrival process, concurrency, warmup policy, measured duration, random seeds, and cost assumption. If one of these changes, it is a new experiment or at least a new stratum.

This is not paperwork. It is how we distinguish a software regression from an accidental change in the question. The same Llama-3.1-8B-Instruct weights can produce different token counts if the chat template changes. The same GPU can have different timing if clocks are left to opportunistic boost. The same endpoint can have different p99 if one client submits after completion and another submits according to a wall-clock arrival process.

The benchmark has five phases:

1. **Setup.** Resolve and record identities, seed all relevant random generators, select the device, load the model, allocate the server, and optionally lock clocks.
2. **Warmup.** Send requests that exercise the actual path. Include the same prompt shapes, batching, streaming mode, and sampling configuration used in measurement. Discard their observations.
3. **Steady-state measurement.** Generate arrivals, record a monotonic client timestamp for submission, first byte or first token, every token if available, and completion. Keep the workload fixed while changing one experimental factor.
4. **Summary.** Compute percentiles, token rates, SLO compliance, goodput, queue-time decomposition, and estimated cost from request records. Never compute p99 from rounded display values.
5. **Provenance.** Write a machine-readable manifest next to the JSONL request records. A number without that manifest is a hint, not a result.

The word “steady” needs care. A five-minute test is not steady state merely because five minutes elapsed. The server may still be filling a prefix cache, compiling kernels, growing a memory pool, or adapting clocks. A useful operational definition is: after warmup, the experiment has a fixed request mix and no pending setup work, and the run records enough observations that the chosen percentile is not dominated by one request. For p99, 100 requests gives only one sample in the tail; 1,000 requests gives ten. That is still not a statistical confidence interval, but it is a more legible tail.

The public [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers), accessed 2026-08-03, explains why GPU work must be timed with GPU-side events rather than host wall time alone. PyTorch exposes the same idea through [`torch.cuda.Event`](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html), accessed 2026-08-03. We use both ideas: a monotonic host clock for end-to-end request experience and CUDA events for a narrow device interval.

### The experiment contract

Before touching a GPU, create a contract such as this:

| Field | Example | Why it is fixed | Source |
|---|---|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` | Architecture and tokenizer affect tokens and memory | cited: [official model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), accessed 2026-08-03 |
| GPU | RTX 4090 24 GB | Memory and clocks bound the feasible region | cited: [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/), accessed 2026-08-03 |
| Prompt mix | chat/RAG/code/translation | Each stratum creates a different prefill/decode ratio | reproduce: `bench.py --suite suite.json` |
| Arrival mode | open-loop Poisson or closed-loop concurrency | Controls queueing semantics | derived from protocol |
| Seed | `20260803` | Makes generated prompts and sampling auditable | derived: configuration |
| Cost | $0.35 per GPU-hour | Converts capacity to economics; not a universal price | reproduce: replace with invoice rate |

Do not put a claimed benchmark result in this table. It is the test definition. The result belongs in a file with a run ID, a source label, and the exact command.

## 2. Open-loop and closed-loop load answer different questions

![Open-loop arrivals continue on a wall clock while closed-loop clients wait for their responses](/imgs/blogs/an-experiment-protocol-for-inference-benchmarks-2.webp)

An **open-loop** generator schedules arrivals independently of completion. If the target rate is $\lambda$ requests per second, the next arrival is selected from the arrival process even when prior requests are still running. This is the right model for “what happens when traffic offers this rate to the service?” It exposes queue growth and overload.

A **closed-loop** generator owns a fixed number of clients. Each client sends a request, waits for its response, then sends another. If there are $C$ clients and the mean response time is $R$, the offered rate is approximately $C/R$ requests per second. This is the right model for “how much work can a fixed population complete?” It automatically backs off as latency grows, which can make an overloaded server look stable.

The animated figure below shows why the distinction cannot be repaired with a larger sample. Motion carries the meaning: the open-loop dots do not wait, while the closed-loop dot returns to the client before the next request leaves.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="Open-loop arrivals continue into a queue while closed-loop arrivals wait for a response" style="width:100%;height:auto;max-width:760px">
<style>
.h-open-loop{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.h-open-label{font:600 18px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.h-open-dot{fill:var(--accent,#6366f1)}.h-open-server{fill:var(--surface,#f3f4f6);stroke:var(--text-secondary,#6b7280);stroke-width:2}.h-open-queue{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:5 5}@keyframes h-open-flow{0%{transform:translateX(0);opacity:0}12%{opacity:1}88%{opacity:1}100%{transform:translateX(570px);opacity:0}}@keyframes h-open-return{0%,30%{transform:translateX(0);opacity:0}45%{opacity:1}70%{transform:translateX(180px);opacity:1}100%{transform:translateX(180px);opacity:0}}.h-open-move{animation:h-open-flow 5s linear infinite}.h-open-move2{animation:h-open-flow 5s linear infinite;animation-delay:1.65s}.h-open-move3{animation:h-open-flow 5s linear infinite;animation-delay:3.3s}.h-open-back{animation:h-open-return 5s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.h-open-move,.h-open-move2,.h-open-move3,.h-open-back{animation:none;opacity:1}}
</style>
<rect class="h-open-loop" x="20" y="20" width="720" height="115" rx="12"/><text class="h-open-label" x="90" y="52">open loop</text><path class="h-open-queue" d="M140 85 H590"/><rect class="h-open-server" x="610" y="57" width="85" height="55" rx="8"/><circle class="h-open-dot h-open-move" cx="140" cy="85" r="9"/><circle class="h-open-dot h-open-move2" cx="140" cy="85" r="9"/><circle class="h-open-dot h-open-move3" cx="140" cy="85" r="9"/><rect class="h-open-loop" x="20" y="165" width="720" height="115" rx="12"/><text class="h-open-label" x="100" y="197">closed loop</text><rect class="h-open-server" x="610" y="202" width="85" height="55" rx="8"/><circle class="h-open-dot h-open-back" cx="140" cy="230" r="9"/>
</svg>
<figcaption>Open-loop arrivals keep offering work; a closed-loop client waits for completion before its next arrival.</figcaption>
</figure>

The simplest open-loop process is deterministic inter-arrival time, $\Delta t = 1/\lambda$. A Poisson process samples $\Delta t = -\ln(1-u)/\lambda$ for $u$ uniform on $(0,1)$. The expectation is still $1/\lambda$, but the variation tests queueing differently. Label which one you use. A fixed-period generator is easier to reproduce; Poisson traffic is often more realistic for independent users.

Here is the protocol’s key calculation. If a run offers $\lambda = 20$ requests/s and the server completes only $\mu = 15$ requests/s, the backlog grows at approximately $\lambda-\mu = 5$ requests/s while the overload persists. After 60 seconds, the arithmetic predicts about $5 \times 60 = 300$ additional queued requests, ignoring admission limits and cancellations. That is derived, not measured. A closed-loop test with 20 clients may never offer 20 requests/s because each client is waiting.

### A small load controller

```python
import asyncio
import random
import time

async def closed_loop(client, prompt, clients, seconds, seed):
    rng = random.Random(seed)
    stop = time.perf_counter() + seconds
    async def one(worker):
        while time.perf_counter() < stop:
            started = time.perf_counter()
            await client.generate(prompt, max_tokens=64)
            elapsed = time.perf_counter() - started
            await asyncio.sleep(0.0 if elapsed else rng.random() * 0.001)
    await asyncio.gather(*(one(i) for i in range(clients)))

async def open_loop(client, prompts, rate, seconds, seed):
    rng = random.Random(seed)
    stop = time.perf_counter() + seconds
    tasks = []
    while time.perf_counter() < stop:
        tasks.append(asyncio.create_task(client.generate(rng.choice(prompts), max_tokens=64)))
        gap = -math.log1p(-rng.random()) / rate
        await asyncio.sleep(gap)
    await asyncio.gather(*tasks)
```

The snippet is a shape, not the final runner: add `import math`, request timestamps, cancellation, a maximum in-flight limit, and a response adapter. The complete script later has those pieces. A load generator must never silently turn open-loop into “submit only when a worker is free”; that is a closed-loop test wearing an open-loop label.

## 3. Warmup and clocks are part of the experiment

![Warmup removes cold-start phases before a steady measurement window, while clock locking reduces run-to-run variability](/imgs/blogs/an-experiment-protocol-for-inference-benchmarks-3.webp)

The first token after process start is a lifecycle event. It may include weight loading, lazy CUDA initialization, kernel compilation, memory-pool growth, graph capture, tokenizer construction, or a cache miss. That event matters for a user-facing cold-start SLO, but it does not belong in a steady-state decode benchmark.

Use two named experiments instead of arguing about whether to “ignore” the first request:

* **Cold start:** start the process, send the request, record setup-to-first-token and setup-to-completion. Report model loading and server readiness separately if possible.
* **Steady state:** start the process, complete a declared warmup, then measure a new request sequence. The warmup count and shapes are part of the method.

Warmup should cover the shape buckets you will report. If the production scheduler captures CUDA graphs for batch sizes 1, 2, 4, 8, and 16, warm up those buckets. If a RAG suite has 8k-token inputs and chat has 256-token inputs, warming only the short chat row is not neutral. It can leave long-context kernels or memory pools cold.

Clock locking is a comparability control, not a performance optimization. On a supported NVIDIA system, an operator may inspect clocks with `nvidia-smi` and may use the administrator-approved application-clock mechanism. Record the command and whether it succeeded. Do not imply that a reader can lock clocks on every laptop or cloud instance. If clock control is unavailable, record “unlocked” and report repeated-run dispersion.

```bash
# Inspection only; the exact fields are public nvidia-smi query fields.
nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv

# Optional, privileged, platform-dependent. Record success or failure in provenance.
sudo nvidia-smi -lgc 210,210
```

The NVIDIA RTX 4090 product page, accessed 2026-08-03, lists 24 GB of memory and a 450 W graphics power rating. Those are cited hardware facts, not a promise that a benchmark will consume 450 W or that every system exposes the same sustained clock. The expected timing range is reader-reproducible only after the reader names the exact card, driver, power limit, temperature, and software stack.

### Synchronization and timer selection

Host calls enqueue GPU work asynchronously. If code records `time.perf_counter()` immediately before and after a kernel launch, the interval can measure launch overhead rather than execution. A safe narrow timing pattern is:

```python
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(input_ids)
end.record()
end.synchronize()
device_ms = start.elapsed_time(end)
```

For a service benchmark, the host clock is still essential. It captures queueing, tokenizer time, network transfer, server scheduling, and time to first streamed token. Use a monotonic clock, never wall-clock timestamps, for durations. Store wall-clock UTC only as metadata so another person can identify the run.

The correct ordering is: record submission on the client, send the request, record the first response token, record each subsequent token if streaming, record completion, then compute. Do not call a blocking “get all text” API and pretend its duration is TTFT. If the API does not expose first-token boundaries, label TTFT unavailable and do not substitute completion latency.

#### Worked example: what warmup changes

Suppose a reader declares 32 warmup requests, then measures 1,000 requests. The measurement set has 1,000 observations; it does not have 1,032 observations with the first 32 hidden in a spreadsheet. If a cold request takes 2,000 ms and the steady requests take 80 ms, including one cold request in a mean changes the arithmetic by $(2{,}000 + 999 \times 80)/1{,}000 = 81.92$ ms. That is a derived example, not a report from this repository. It demonstrates why a single lifecycle event can move an average while barely changing p99 in a larger run.

## 4. Prompt suites need fixed strata

![A prompt matrix fixes workload strata across input length, output length, purpose, and tokenization pressure](/imgs/blogs/an-experiment-protocol-for-inference-benchmarks-4.webp)

“The prompt” is not a workload. A benchmark suite should contain named strata with stable input and output targets. This series uses four reusable rows:

| Stratum | Input shape | Output shape | What it stresses | Source |
|---|---|---|---|---|
| Chat | short system and user turns | long enough to expose decode | decode, scheduling, streaming | reproduce: checked-in suite |
| RAG | retrieved context plus question | short answer | prefill, prompt length, KV allocation | reproduce: checked-in suite |
| Code | prefix and incomplete function | continuation | tokenizer, repetition, decode | reproduce: checked-in suite |
| Translation | source sentence or paragraph | target translation | tokenizer ratio and balanced output | reproduce: checked-in suite |

The suite must pin the exact text, not just a character count. Character length is a poor proxy for tokens across languages and tokenizers. Use the [BPE tokenizer explainer](/blog/machine-learning/large-language-model/bpe-tokenizer) when designing the suite, then record `input_tokens` after tokenization. For a multi-model comparison, the text stays fixed while token counts are measured separately per tokenizer. Do not call unequal token counts “equal context.” Call it equal text.

A useful suite file contains an ID, a prompt, an optional chat-message structure, target output tokens, a stratum, and a seed. Keep prompts in source control. Avoid random synthetic words unless the generator and seed are recorded; synthetic prompts can overrepresent easy cache patterns or unusual tokenization.

```json
[
  {"id":"chat-01","stratum":"chat","prompt":"Explain why a queue can increase p99 without increasing model FLOPs.","max_tokens":96},
  {"id":"rag-01","stratum":"rag","prompt":"Context: ...\nQuestion: Which constraint is violated?","max_tokens":48},
  {"id":"code-01","stratum":"code","prompt":"def stable_percentile(values, q):\n    \"\"\"Return a deterministic percentile.\"\"\"\n","max_tokens":80},
  {"id":"translation-01","stratum":"translation","prompt":"Translate to English: The queue is part of the latency.","max_tokens":48}
]
```

Sampling must be explicit. For latency measurement, greedy decoding or temperature zero reduces output variance. For a production-like stochastic test, record temperature, top-p, top-k, repetition settings, and a seed per request. A seed does not guarantee bitwise equality across GPU kernels or batch sizes; it guarantees that the requested random stream is controlled as far as the implementation permits. The [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html), accessed 2026-08-03, explicitly warn that deterministic results are not guaranteed across releases or platforms.

The suite should be replayable in a fixed order for microbenchmarks and shuffled with a recorded seed for mixed workload tests. Report both. Fixed order is better for comparing a code change; shuffled order is better for estimating interference among strata.

## 5. Provenance decides what a number means

![Every benchmark number resolves to a derived equation, a named public source, or a reproducible reader run](/imgs/blogs/an-experiment-protocol-for-inference-benchmarks-5.webp)

Use three provenance classes.

**Derived** means the reader can follow arithmetic from values stated in the post. For example, if a request produces 128 output tokens in 2 seconds, its rate is $128/2 = 64$ tok/s. If the GPU is charged $0.35 per hour, the compute-only cost is $0.35 \times 3{,}600 / 64 = $19.6875$ per million generated tokens, assuming the rate is sustained and ignoring host, storage, and idle costs. Round the display to $19.69, but keep the inputs and unrounded intermediate values in JSON.

**Cited** means a public source makes the claim. The [vLLM paper](https://arxiv.org/abs/2309.06180), accessed 2026-08-03, is evidence about the PagedAttention system described by its authors; it is not a run of `nanoserve`. The vLLM team’s [InferenceMAX Blackwell post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), dated 2025-10-09, reports a “up to 4×” throughput comparison under its stated setup. That belongs in a contrast section, not in our result table. The vLLM speculative decoding post, dated 2024-10-17, reports workload-dependent speedups and slowdowns; attribute those values to vLLM and preserve its setup.

**Reproduce** means the reader can run the named command on named hardware. State an expected range only when it is clearly a range and clearly not our measurement. For an 8B bf16 model on an RTX 4090, a reader may see a wide range depending on engine, prompt, batch, clocks, and driver; the responsible statement is “run the script and report your observed range,” not a precise tok/s claim.

The provenance tree is a review tool. Every row in a results table should say `derived`, `cited: source`, or `reproduce: command`. If you cannot fill Source, delete the row or turn it into a hypothesis.

### Cost arithmetic

Cost needs two clocks: billed GPU time and useful output. If one GPU-hour costs $p and the measured steady output rate is $r$ tok/s, then

$$
\text{cost per million output tokens} = \frac{p \times 3{,}600 \times 1{,}000{,}000}{r \times 1{,}000{,}000} = \frac{3{,}600p}{r}.
$$

For $p=0.35$ and $r=70$, the result is $3{,}600 \times 0.35 / 70 = 18$ dollars per million output tokens. If only 80% of output tokens meet the latency SLO, goodput is $0.8r = 56$ tok/s, and the SLO-qualified cost is $3{,}600 \times 0.35 / 56 = 22.5$ dollars per million qualified output tokens. Both are derived examples. They answer different operational questions.

## 6. Report the scoreboard, not one vanity number

![The benchmark scoreboard connects TTFT and TPOT to throughput, tail latency, goodput, and cost](/imgs/blogs/an-experiment-protocol-for-inference-benchmarks-6.webp)

The minimum request record has these timestamps and counts:

* `submitted_ns`: client monotonic time before sending;
* `first_token_ns`: first streamed token or first-byte boundary;
* `completed_ns`: final token boundary;
* `input_tokens`, `output_tokens`, and `status`;
* `stratum`, `request_id`, `seed`, and error text if any.

From them derive **TTFT** as first token minus submission, **completion latency** as completion minus submission, and **TPOT** as the median inter-token delay after the first token. Some teams call the reciprocal of TPOT “decode tok/s”; name the convention. A streaming endpoint can expose a time series of token arrivals, which is stronger evidence than dividing final output count by total latency.

Throughput is $\sum \text{output tokens}/\text{measurement seconds}$. Do not divide by the number of requests unless you explicitly mean requests per second. P50 describes the middle request; p99 is the value below which 99% of observations fall under the chosen percentile convention. Record the percentile method or use a standard library implementation and version.

Goodput is throughput under an SLO filter. For example, with TTFT $le 300$ ms and TPOT $le 40$ ms, count only requests that satisfy both, then divide their output tokens by the measurement window. This is intentionally stricter than “throughput minus errors.” A request that eventually completes but violates an interactive objective is not goodput.

| Metric | Formula | Failure mode it reveals | Source |
|---|---|---|---|
| TTFT | `first_token - submitted` | queue plus prefill delay | derived: request timestamps |
| TPOT | median consecutive token gaps | decode scheduling and kernel cadence | derived: streamed timestamps |
| Output tok/s | `sum(output_tokens) / window_s` | capacity | derived: token counts |
| p99 TTFT | percentile of TTFT list | tail queueing and long prefills | derived: request records |
| Goodput | SLO-qualified output tokens / window | capacity that users can use | derived: SLO thresholds |
| $ / 1M tokens | `3600 * gpu_hour_price / tok_s` | economics at the stated rate | derived: price and rate |

### Queue time versus service time

If the server exposes enqueue and execution timestamps, split latency into queue and service. Without those fields, you can still compare the client’s TTFT against a single-request baseline, but do not call the difference exact queue time. That difference also contains network and client overhead.

Little’s law provides a sanity check: $L = \lambda W$, where $L$ is average in-flight work, $\lambda$ is completed arrival rate in requests/s, and $W$ is average time in seconds. If a closed-loop run has 8 average in-flight requests and 0.25 s average response time, its implied rate is $8 / 0.25 = 32$ requests/s. This is derived and should be close to the observed completion rate only when the run is stable and the same population is counted.

#### Worked example: a p99 that hides in the mean

Consider 100 requests: 99 have TTFT 100 ms and one has TTFT 2,100 ms. The mean is $(99 \times 100 + 2{,}100)/100 = 120$ ms, while a nearest-rank p99 can select the 2,100 ms request depending on the convention. The lesson is not that one percentile definition is morally correct. The lesson is that a mean cannot stand in for a tail. Store the raw records, the count, and the percentile definition.

## 7. `bench.py`: a runnable nanoserve-oriented template

The following script deliberately keeps the engine adapter small. It first tries a local `nanoserve_client` module with an async `generate` method. If that module is unavailable, it uses an OpenAI-compatible HTTP endpoint. The endpoint path is configurable because the protocol should survive API evolution. The HTTP adapter expects a streaming response with one JSON object per line and a `token` field; adapt that parser to the exact server contract rather than silently treating a whole response as one token.

```python
#!/usr/bin/env python3
"""Protocol runner for nanoserve or an OpenAI-compatible streaming endpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:
    torch = None

try:
    import aiohttp
except Exception:
    aiohttp = None


@dataclass
class Observation:
    request_id: str
    stratum: str
    seed: int
    submitted_ns: int
    first_token_ns: int | None
    completed_ns: int
    input_tokens: int | None
    output_tokens: int
    status: str
    error: str | None = None

    def ttft_ms(self) -> float | None:
        if self.first_token_ns is None:
            return None
        return (self.first_token_ns - self.submitted_ns) / 1e6

    def latency_ms(self) -> float:
        return (self.completed_ns - self.submitted_ns) / 1e6


def load_suite(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not rows:
        raise ValueError("suite is empty")
    required = {"id", "stratum", "prompt", "max_tokens"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"suite row missing {sorted(missing)}")
        if int(row["max_tokens"]) <= 0:
            raise ValueError("max_tokens must be positive")
    return rows


class HttpAdapter:
    def __init__(self, url: str, model: str, timeout_s: float):
        if aiohttp is None:
            raise RuntimeError("pip install aiohttp for the HTTP adapter")
        self.url, self.model, self.timeout_s = url, model, timeout_s

    async def generate(self, row: dict[str, Any], seed: int) -> tuple[int, int | None]:
        payload = {"model": self.model, "prompt": row["prompt"],
                   "max_tokens": row["max_tokens"], "stream": True,
                   "temperature": 0.0, "seed": seed}
        first_ns, tokens = None, 0
        started = time.perf_counter_ns()
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=payload) as response:
                response.raise_for_status()
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    item = json.loads(line)
                    if item.get("token") is not None or item.get("text"):
                        tokens += 1
                        if first_ns is None:
                            first_ns = time.perf_counter_ns()
        if first_ns is None:
            first_ns = started
        return tokens, first_ns


class NanoserveAdapter:
    def __init__(self):
        try:
            from nanoserve_client import Client
        except ImportError as exc:
            raise RuntimeError("use --url or install the local nanoserve client") from exc
        self.client = Client()

    async def generate(self, row: dict[str, Any], seed: int) -> tuple[int, int | None]:
        first_ns, tokens = None, 0
        async for token in self.client.generate(prompt=row["prompt"],
                                                 max_tokens=row["max_tokens"],
                                                 temperature=0.0, seed=seed):
            tokens += 1
            if first_ns is None:
                first_ns = time.perf_counter_ns()
        return tokens, first_ns


async def one_request(adapter, row, seed) -> Observation:
    request_id = str(uuid.uuid4())
    submitted = time.perf_counter_ns()
    try:
        output_tokens, first_ns = await adapter.generate(row, seed)
        completed = time.perf_counter_ns()
        return Observation(request_id, row["stratum"], seed, submitted, first_ns,
                           completed, row.get("input_tokens"), output_tokens, "ok")
    except Exception as exc:
        completed = time.perf_counter_ns()
        return Observation(request_id, row["stratum"], seed, submitted, None,
                           completed, row.get("input_tokens"), 0, "error", repr(exc))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(rows: list[Observation], window_s: float, gpu_hour: float,
              ttft_slo_ms: float, tpot_slo_ms: float) -> dict[str, Any]:
    good = [r for r in rows if r.status == "ok"]
    ttft = [r.ttft_ms() for r in good if r.ttft_ms() is not None]
    lat = [r.latency_ms() for r in good]
    output = sum(r.output_tokens for r in good)
    tok_s = output / window_s if window_s > 0 else 0.0
    qualified = [r for r in good if r.ttft_ms() is not None and
                 r.ttft_ms() <= ttft_slo_ms and r.latency_ms() <= tpot_slo_ms]
    good_tok_s = sum(r.output_tokens for r in qualified) / window_s if window_s > 0 else 0.0
    return {"requests": len(rows), "completed": len(good), "errors": len(rows) - len(good),
            "output_tokens": output, "tok_s": tok_s, "goodput_tok_s": good_tok_s,
            "ttft_p50_ms": percentile(ttft, .50), "ttft_p99_ms": percentile(ttft, .99),
            "latency_p50_ms": percentile(lat, .50), "latency_p99_ms": percentile(lat, .99),
            "cost_per_million_output_tokens": (3600 * gpu_hour / tok_s if tok_s else None),
            "cost_per_million_goodput_tokens": (3600 * gpu_hour / good_tok_s if good_tok_s else None)}


async def run(args) -> tuple[list[Observation], float]:
    suite = load_suite(Path(args.suite))
    rng = random.Random(args.seed)
    adapter = HttpAdapter(args.url, args.model, args.timeout) if args.url else NanoserveAdapter()
    for i in range(args.warmup):
        await one_request(adapter, suite[i % len(suite)], args.seed + i)
    observations: list[Observation] = []
    started = time.perf_counter()
    if args.mode == "closed":
        async def worker(worker_id: int):
            while time.perf_counter() - started < args.duration:
                row = rng.choice(suite)
                observations.append(await one_request(adapter, row, args.seed + worker_id))
        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
    else:
        tasks = []
        while time.perf_counter() - started < args.duration:
            row = rng.choice(suite)
            tasks.append(asyncio.create_task(one_request(adapter, row, rng.randrange(2**31))))
            gap = -math.log1p(-max(rng.random(), 1e-12)) / args.rate
            await asyncio.sleep(gap)
        observations = list(await asyncio.gather(*tasks))
    return observations, max(time.perf_counter() - started, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--url")
    parser.add_argument("--model", default="nanoserve")
    parser.add_argument("--mode", choices=["open", "closed"], default="closed")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--gpu-hour", type=float, default=0.35)
    parser.add_argument("--ttft-slo-ms", type=float, default=300.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=40.0)
    parser.add_argument("--out", default="bench-results.jsonl")
    args = parser.parse_args()
    if args.mode == "open" and args.rate <= 0:
        parser.error("--rate must be positive in open mode")
    observations, window = asyncio.run(run(args))
    summary = summarize(observations, window, args.gpu_hour, args.ttft_slo_ms, args.tpot_slo_ms)
    provenance = {"kind": "reproduce", "command": " ".join(os.sys.argv),
                  "seed": args.seed, "window_s": window, "summary": summary}
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in observations:
            handle.write(json.dumps({"kind": "request", **asdict(row)}) + "\n")
        handle.write(json.dumps({"kind": "summary", **provenance}) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

There are two deliberate limitations in this template. It does not infer token counts from characters, and it does not invent TPOT from a non-streaming response. A production `nanoserve` adapter should return token timestamps and a server-side queue timestamp. It should also make cancellation explicit: an open-loop overload test that leaves hundreds of tasks alive after the timer is not a bounded experiment.

Run it with a fixed suite and record the command:

```bash
python bench.py --url http://127.0.0.1:8000/v1/completions \
  --suite suite.json --mode closed --concurrency 8 --warmup 32 \
  --duration 120 --seed 20260803 --out runs/closed-c8.jsonl

python bench.py --url http://127.0.0.1:8000/v1/completions \
  --suite suite.json --mode open --rate 8 --warmup 32 \
  --duration 120 --seed 20260803 --out runs/open-r8.jsonl
```

The output is expected to be reader-reproducible, not a result from this post. On an RTX 4090, an 8B model may fit in one configuration and fail in another because weights, activations, KV cache, allocator reservation, and the server process compete for 24 GB. The reader should report the observed range, driver, dtype, and model revision. If the run errors, that is evidence about the declared configuration, not a missing number to be filled from memory.

## 8. Comparison discipline: change one axis

The most useful sweep is not every possible combination. Start with a baseline and vary one axis: load mode, concurrency or arrival rate, prompt stratum, output cap, dtype, scheduler limit, or engine commit. A matrix is useful after the baseline has passed sanity checks.

| Sweep | Fixed | Changed | Primary outputs | Interpretation |
|---|---|---|---|---|
| load controller | model, suite, duration | open vs closed | p99, queue, goodput | semantics of offered work |
| concurrency | model, suite, mode | 1, 2, 4, 8 clients | tok/s and TTFT | batching versus queueing |
| prompt stratum | model, mode, seed | chat/RAG/code/translation | TTFT, TPOT | prefill/decode mix |
| output cap | model, suite, mode | 32, 64, 128 tokens | TPOT, completion | decode work |
| dtype | model, suite, mode | bf16/fp16/quantized | memory and quality | speed-quality tradeoff |

Do not compare a batch-1 microbenchmark with an eight-client service run and call it a regression. The former may measure a single decode path; the latter includes queueing, batching, network, and mixed shapes. This is why [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) belongs in the series: throughput is a property of policy as well as kernels.

### Sanity checks before a sweep

First, run one prompt twice with greedy decoding and compare token counts and text. If they differ, investigate seeds, model revision, tokenizer, or nondeterministic kernels before measuring latency. Second, run one request and then the declared warmup; check that the server is not returning an error after the benchmark already started. Third, compare the number of submitted, completed, cancelled, and failed requests. An open-loop test that reports only completed requests can hide work still in flight.

Fourth, verify that the suite’s input and output token distributions match the intended labels. A RAG row with 40 input tokens is not a long-context test just because its field is named `rag`. Fifth, watch memory and temperature but do not treat utilization as performance. A GPU can be highly utilized while the queue violates every SLO.

## 9. Public results are context, not borrowed measurements

The [PagedAttention paper](https://arxiv.org/abs/2309.06180) is a useful historical reference for why memory management affects serving throughput. It describes a system and reports its own evaluated comparisons under its own setup. It does not certify the result of a local `nanoserve` implementation.

The vLLM team’s public [Blackwell InferenceMAX post](https://vllm.ai/blog/2025-10-09-blackwell-inferencemax), dated 2025-10-09, frames performance as a Pareto frontier between throughput and per-token latency and reports up to 4× higher throughput at similar latency in its stated scenarios. That is the right shape of comparison for a later capstone, but we should preserve the hardware, workload, and attribution. A single peak tok/s number would discard the frontier.

The public [speculative decoding post](https://vllm.ai/blog/2024-10-17-spec-decode), dated 2024-10-17, reports that gains vary with workload and can become slowdowns at high query rates. This is exactly why post 45 will use the protocol here: acceptance rate, draft overhead, queueing, and SLO-qualified goodput must appear together.

These sources are benchmark targets and contrasts. They are not first-hand runs in this workspace. The honest table is therefore:

| Claim | What we can say here | Source |
|---|---|---|
| A GPU timer should use device events | PyTorch provides `torch.cuda.Event`; synchronize around the measured interval | cited: PyTorch docs, accessed 2026-08-03 |
| Paged attention changes memory management | The vLLM authors describe and evaluate PagedAttention | cited: vLLM paper, accessed 2026-08-03 |
| Blackwell can move a throughput-latency frontier | vLLM reports up to 4× in named scenarios | cited: vLLM post, 2025-10-09 |
| This `nanoserve` protocol reaches X tok/s | Not claimed; run `bench.py` and attach the JSONL | reproduce: this post |

## 10. Case studies: the benchmark failures that look like speedups

### The cold-start “regression”

A developer changes nothing in the model and sees a two-second first request. The wrong response is to average it with a minute of steady-state traffic and call the result “latency.” The correct response is to split cold-start and steady-state experiments. If the product SLO includes startup, keep the cold result. If the question is decode throughput after readiness, exclude it by a declared warmup and report the exclusion.

### The closed-loop capacity illusion

A test with 16 workers reports stable p99 at a load that should saturate the GPU. The server is not necessarily healthy. Each worker waits for its response, so higher latency reduces the next offered arrival rate. Re-run open-loop at the same observed completion rate, then step above it. If p99 grows and in-flight work rises, the closed-loop test was applying backpressure before the server.

### The tokenizer drift

A prompt file is unchanged, but TTFT shifts after a dependency update. The tokenizer or chat template changed, changing input token count and prefill work. Record tokenizer identity, template, token counts, and model revision. The [model-serving view of why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) is useful here: the unit of work is tokens, not characters.

### The missing tail

A dashboard displays p50 and mean, both healthy. Customers see timeouts. The run has 40 requests, so p99 is effectively one observation and the report has no histogram or count. Increase the sample size, preserve raw records, report failures and cancellations, and define whether the percentile includes timeouts. A timeout is not a fast request with missing output.

### The clock drift

A second run is faster after the first run heated the GPU. The change is not necessarily a kernel improvement; it can be a clock-state transition. Record clock policy, temperature, power, and repeated-run order. If clocks cannot be locked, randomize order or alternate configurations and report dispersion. Do not cherry-pick the warmest run.

### The cost number with the wrong denominator

A team divides GPU-hour price by requests/s and publishes $ per million tokens. That is dimensionally wrong. Use output tokens, or state that the number is $ per million requests. If only SLO-qualified tokens count for the product, use goodput. If the GPU is idle between bursts, use billed wall time rather than a saturated steady-state rate for the capacity plan.

## 11. When to reach for this protocol—and when not to

Use this protocol when you are comparing engine commits, scheduler policies, model variants, quantization modes, speculative decoding settings, or hardware. It is especially valuable when a metric has a tail, when requests have heterogeneous context lengths, or when the system is close to an admission limit.

Do not use a full open-loop sweep to answer a one-kernel correctness question. Use a synchronized microbenchmark and a reference output. Do not use a closed-loop number to size a public endpoint. Do not quote a reader-expected range as a result. Do not lock clocks without permission on shared infrastructure. Do not add more decimals than the experiment supports.

For production comparison, use vLLM as a benchmark target or contrast, not as documentation to copy. A mature engine has scheduling, paged cache management, tokenizer paths, streaming, and observability that a small `nanoserve` adapter may not. The question is where the gap comes from. The [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) is a useful external comparison; the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) will later assemble the honest gap table.

## 12. The runbook for a trustworthy sweep

The protocol becomes useful when it is boring enough to run every week. Put the experiment definition in a directory with a stable layout: `suite.json` for prompts, `commands.txt` for launch commands, `environment.json` for versions and hardware, one JSONL file per run, and a small summary generated from those records. Never edit the summary by hand. If a reviewer asks why p99 changed, regenerate it from the same request records.

Before a run, check the following in order. First, verify that the server is serving the intended model and tokenizer. A model alias is not a sufficient identity because a registry tag can move. Record the resolved revision when the loader exposes it. Second, send a health request and one deterministic smoke prompt. Third, inspect free GPU memory and the configured memory limit. Fourth, check whether a previous process still owns the device. Fifth, run warmup and verify that the number of warmup completions equals the declared count. Only then start the measurement clock.

During a run, collect enough telemetry to explain a change without turning the benchmark into a monitoring platform. The useful minimum is GPU name, temperature, power, memory used, SM clock, and process identifier at a low sampling frequency. A one-second telemetry sample is adequate for a minute-scale run; it is not evidence about a 2 ms kernel. Keep telemetry in a separate file keyed by run ID. Avoid sampling so aggressively that the monitoring path changes the service being measured.

After a run, compare three counts: submitted requests, terminal requests, and records written. Open-loop tests often expose a fourth count, requests still in flight at the end of the window. Decide whether those requests are cancelled, drained, or assigned to a drain interval. The measurement window should end at the last scheduled arrival, not necessarily at the last response. If you include drain time in throughput, say so; otherwise the denominator and numerator describe different intervals.

### Repeated runs and ordering

One run is a trace. A baseline needs repetitions. Three repetitions are a practical starting point for a local code change, but they do not prove a universal distribution. Five or more randomized repetitions are more useful when thermal state, cloud placement, or noisy neighbors matter. Report the median run and the range across runs; retain each run’s p50 and p99 rather than pooling records without a reason.

When comparing A and B, alternate them: A, B, B, A is better than A, A, B, B if the machine warms during the test. The exact order is part of the manifest. For long tests, alternate smaller blocks and include a short calibration request before each block. That calibration request is not part of the measured set; it detects a dead server or a changed endpoint.

Randomization has two levels. Randomize prompt order within a mixed workload using a recorded seed. Randomize configuration order across repetitions when thermal or scheduling drift is possible. Do not randomize the model revision or the clock policy. Those are experimental factors, not noise to be averaged away.

### Percentiles, confidence, and small samples

Percentiles describe the observed sample. They do not automatically provide confidence about the underlying service. With 50 requests, p99 is near the maximum and is unstable. With 10,000 requests, p99 has more support, but a single five-second pause can still dominate it. Publish the count beside every percentile and show a histogram or quantile table when the tail is central to the claim.

For a comparison, bootstrap the difference between two run-level medians or use a confidence interval appropriate to the metric. Do not use a normal approximation for a highly skewed latency distribution without checking it. This article does not prescribe one statistical test because the correct choice depends on paired versus independent requests, repeated configurations, and whether the load process is stationary. It does prescribe raw records and explicit sample counts.

The most useful small-sample statement is often qualitative: “The p99 moved from one outlier to another, so this run cannot establish a tail improvement.” That is more honest than writing a p99 with two decimal places.

## 13. Extending `bench.py` without losing the protocol

Future experiments will need features that the compact template intentionally omits. Add them in ways that preserve the request record schema.

**Server timestamps.** Add `enqueued_ns`, `prefill_start_ns`, `decode_start_ns`, and `server_completed_ns` to the response or an observability side channel. Keep client timestamps too. A server-only duration cannot describe network or queueing before admission; a client-only duration cannot identify the scheduler phase. The two clocks together support a useful decomposition.

**Token-level records.** Change the adapter return value from `(count, first_ns)` to a list of `(token_index, received_ns)`. Store the first 32 or all token timestamps depending on file size. TPOT is then the distribution of adjacent gaps, not a single completion average. If a streaming transport coalesces multiple tokens into one network packet, distinguish token production time from token receipt time and label which one you have.

**Cancellation.** Open-loop overload should have a maximum drain timeout. Use `asyncio.wait_for` around each request and record `timeout` rather than treating an exception as a zero-latency completion. On the server, cancellation should stop GPU work where possible. A client that abandons the socket while the engine continues decoding is measuring an expensive form of goodput failure.

**Backpressure.** Add `--max-inflight` to the open-loop generator and record rejected arrivals. This creates a third useful condition: offered traffic, admitted traffic, and completed traffic. A queue that is bounded by admission control can have lower p99 and lower raw throughput than an unbounded queue, while producing higher goodput.

**Batch labels.** If `nanoserve` exposes batch size or scheduler state, record it at each token. A request can start at batch 1 and decode alongside other requests later. The average batch size is not enough to explain an outlier, but a per-token batch trace can show whether a scheduler change actually increased useful work.

**Memory snapshots.** Record allocated, reserved, and free device memory before warmup, after warmup, and after the measured window. Do not equate “free” memory with KV-cache capacity. Weights, activations, CUDA workspaces, allocator fragmentation, and graph pools all compete for the same device. The earlier [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) material is the right place for cache formulas; this protocol records the conditions under which those formulas are tested.

### A provenance sidecar

The JSONL summary is intentionally not the only artifact. A sidecar can look like this:

```json
{
  "run_id": "2026-08-03-open-r8-seed20260803",
  "kind": "reproduce",
  "hardware": {"gpu": "RTX 4090", "memory_gb": 24},
  "software": {"engine_commit": "replace-with-git-sha", "python": "replace", "cuda": "replace"},
  "model": {"id": "meta-llama/Llama-3.1-8B-Instruct", "revision": "replace"},
  "suite_sha256": "replace-with-hash",
  "load": {"mode": "open", "rate_per_s": 8, "warmup": 32, "duration_s": 120},
  "clock_policy": "unlocked; recorded by nvidia-smi",
  "seed": 20260803,
  "cost": {"gpu_hour_usd": 0.35, "source": "operator input"}
}
```

The values marked `replace` are not decoration. A protocol template should make missing provenance visible. A reviewer should be able to reject a row because the suite hash is absent, before debating whether a 4% speedup is real.

## 14. How the rest of the series should use this template

Post 41 can hold the model and GPU fixed while sweeping batch, context, and dtype. Its result table should use the same prompt IDs, the same warmup definition, and the same output metrics. The memory ceiling can then be compared with the KV arithmetic from the earlier cache post without changing the unit of work halfway through.

Post 42 can compare Qwen3, Gemma 3, and Llama 3 at equal hardware. “Equal” must be defined: same GPU, same software timing method, same text suite, and a disclosed tokenizer token count for each model. Equal character count is not equal token count. If one model uses a different chat template, record that difference rather than hiding it.

Post 43 can test a small MoE on consumer hardware. Active parameter count is an architecture fact, not a throughput result. The protocol should report total and active parameters as cited model metadata, then measure memory, TTFT, TPOT, and goodput under a reader-runnable configuration. Expert offload introduces host traffic and a new clock domain; it should be a separate stratum, not an unexplained footnote.

Post 44 can compare quantization. It must carry a quality gate alongside speed. A faster configuration that violates the task metric is not a win for the same workload. Store the evaluation dataset identity and quality result beside the timing result, and keep quantization preparation time separate from steady-state serving time.

Post 45 can measure speculative decoding. Acceptance rate is a mechanism metric; goodput is the service metric. A draft model can raise acceptance while adding enough draft work or queue pressure to lower SLO-qualified throughput. The vLLM public result cited earlier makes this workload dependence concrete, but the later post must reproduce its own pair and workload rather than copy a headline.

This is why protocol posts belong before experiment posts. Once every experiment uses the same names, the series can compare results without comparing accidental methodology. A later capstone can say that two engines differ on p99 under open-loop RAG traffic, and a reader can inspect the exact suite and arrival process instead of guessing.

## A final checklist before publishing a number

* Is the claim about cold start, warm state, or steady state?
* Does the load mode match the question?
* Are input and output token counts recorded rather than inferred from characters?
* Are model, tokenizer, engine, driver, CUDA, GPU, dtype, clocks, and scheduler settings identified?
* Did warmup exercise all reported shape buckets?
* Are host and device timers used for the phases they can actually observe?
* Are failures, timeouts, cancellations, and in-flight requests counted?
* Do p50 and p99 include an explicit sample count and percentile convention?
* Is goodput tied to explicit TTFT and TPOT thresholds?
* Does the cost denominator say output tokens, input-plus-output tokens, requests, or qualified tokens?
* Is each quantitative statement labeled derived, cited, or reproduce?
* Can another engineer run the command from the saved suite and manifest?

Only after those questions have answers should a graph go into a README or a blog table. The number may be less flattering. It will also be useful.

## Key takeaways

1. Define the load process before measuring the service.
2. Open-loop tests preserve offered traffic; closed-loop tests preserve client population.
3. Separate cold start, warmup, and steady state.
4. Use a monotonic host clock for user latency and CUDA events for device intervals.
5. Pin model, tokenizer, prompt text, seed, engine commit, dtype, GPU, clocks, and scheduler settings.
6. Keep chat, RAG, code, and translation as named strata rather than one blended average.
7. Report TTFT, TPOT, tok/s, p99, errors, goodput, and cost with a denominator.
8. Label every number derived, cited, or reproduce-only.
9. Preserve request-level JSONL so a summary can be recomputed.
10. A benchmark that cannot explain its provenance is not a baseline.

## Further reading

* [CUDA GPU timers in the CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers), NVIDIA, accessed 2026-08-03.
* [`torch.cuda.Event`](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html), PyTorch, accessed 2026-08-03.
* [Llama 3.1 8B Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), Meta, accessed 2026-08-03.
* [A Performance Evaluation of LLM Serving Systems with PagedAttention](https://arxiv.org/abs/2309.06180), Kwon et al.
* [Setting up a reproducible benchmark](/blog/machine-learning/performance-engineering/setting-up-a-reproducible-benchmark).
* [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization).

The artifact added by this post is the `bench.py` template above: a deliberately small protocol runner that future experiments can extend with server-side timestamps, token-level streams, GPU telemetry, and adapters for the exact `nanoserve` API. It produces no magic number. It produces the evidence needed to earn one.

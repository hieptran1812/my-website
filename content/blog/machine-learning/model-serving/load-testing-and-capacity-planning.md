---
title: "Load Testing and Capacity Planning for LLM Servers: From Benchmark Numbers to a GPU Fleet"
date: "2026-07-04"
publishDate: "2026-07-04"
description: "How to benchmark an LLM server the right way and turn the numbers into a defensible capacity plan — how many GPUs for X QPS at Y SLO — using open-loop load sweeps, the knee, Little's Law, and honest cost-per-token math."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "load-testing",
    "capacity-planning",
    "benchmarking",
    "vllm",
    "slo",
    "goodput",
    "llm-serving",
    "queuing-theory",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/load-testing-and-capacity-planning-1.webp"
---

The launch email said "we expect 50 requests per second at peak." The infra channel said "we have four H100s, we're fine." Three weeks later, on the first real marketing push, p99 time-to-first-token went from 800 milliseconds to eleven seconds inside of ninety seconds, the client-side timeouts started firing, and the retry storm they triggered pushed the queue past the point of no return. The postmortem found the load test that "proved" the four-GPU plan: a `for` loop that fired the same 128-token prompt at the server as fast as one Python thread could go, reported "1,900 requests per second," and got screenshotted into a slide. That number was not just optimistic. It was measuring something that has almost nothing to do with the production workload.

This post is about the gap between that screenshot and a capacity plan you can bet a launch on. The core skill is narrow and unglamorous: run a load test that reproduces the real workload, sweep the offered load until you find the point where the server stops meeting its SLO, and turn that single number into a GPU count with enough headroom to survive a bad Tuesday. Everything downstream — autoscaling policy, cost forecasts, the go/no-go on a launch — rests on getting that one measurement honest. Get it wrong in the optimistic direction and you page yourself at 2 a.m.; get it wrong in the pessimistic direction and you burn a six-figure annual GPU budget on capacity you never use.

The figure below is the whole pipeline in one picture, and it doubles as the outline of this post. You benchmark a single replica, sweep the offered load to find the sustainable knee, then fan that one number into fleet sizing and headroom before it merges into a plan. Every arrow is a decision that has a wrong answer, and most teams get at least two of them wrong.

![Flow diagram showing capacity planning as a pipeline that benchmarks one replica, finds the knee, then sizes a fleet and adds headroom before merging into a plan](/imgs/blogs/load-testing-and-capacity-planning-1.webp)

The frame for the whole series applies here: **latency, throughput, and cost form a triangle, and every capacity decision is a trade on that triangle.** A load test measures where you sit on it today; a capacity plan is a bet on how much you can move along it before the SLO breaks. If you have not read the series intro on [what model serving actually is](/blog/machine-learning/model-serving/what-is-model-serving) or the deep dive on [serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics), skim them first — this post assumes you already know what TTFT (time-to-first-token, the latency until the first output token arrives) and TPOT (time-per-output-token, the inter-token latency during generation) mean, and why p99 matters more than the mean.

By the end you will be able to: pick a load-testing tool that measures a knee instead of a mirage; write an open-loop load generator that sweeps request rate and records the right percentiles; read a load-sweep table and point at the knee; and do the arithmetic from a single-replica saturation point to a fleet size, complete with the headroom for p99 bursts and an N+1 failover spare. And you will be able to compute a cost per million tokens that survives a finance review.

## Why a naive load test lies about your LLM

Start with the mistake in the launch story, because it is the most common one and the most expensive. The naive test fires **one fixed prompt** at the server, over and over. It is easy to write, it produces a big number, and that number is worthless for capacity planning. The reason is specific to how LLM inference works.

An LLM request is not one unit of work. It is two very different phases stapled together. **Prefill** processes the input prompt — all of it, in parallel, in essentially one big matrix multiply per layer — and it is *compute-bound*: the GPU's tensor cores are the bottleneck. **Decode** generates the output one token at a time, each token requiring a full forward pass that reads the entire model weights and the growing KV cache from memory, and it is *memory-bandwidth-bound*: the GPU spends most of each step waiting on HBM (high-bandwidth memory) reads, not computing. The cost of a request is roughly `prefill(input_len) + output_len × decode_step_cost`. A request with a 4,000-token input and a 50-token output is dominated by prefill FLOPs; a request with a 100-token input and a 2,000-token output is dominated by decode bandwidth. These two requests stress completely different parts of the GPU and saturate the server at completely different rates.

A fixed prompt collapses that entire distribution to a single point. If you benchmark with 128 input tokens and 128 output tokens, you have measured the server's behavior at exactly one spot in a two-dimensional space that your real traffic spreads across widely. Worse, short outputs are the cheapest possible workload — decode is where the seconds pile up, and 128 output tokens barely exercises it. The fixed-prompt test reports a throughput number that is systematically, and often dramatically, higher than what the real length mix can sustain.

The figure contrasts the two approaches directly. On the left, the naive test with its single collapsed length; on the right, a test that replays a realistic length mix sampled from production, which moves the measured sustainable rate down by a large factor.

![Before-and-after comparison of a naive fixed-prompt load test reporting an inflated 42 requests per second versus a realistic length-mix replay reporting a sustainable 24 requests per second](/imgs/blogs/load-testing-and-capacity-planning-2.webp)

How much does the length mix move the number? It depends on your workload, but the effect is large enough that you cannot ignore it. Here is the arithmetic that makes it concrete. Suppose your real traffic has a p50 output of 256 tokens and a p95 output of 1,024 tokens, with a long tail out to the model's context limit. A batch of requests that all generate 128 tokens finishes fast, frees its KV-cache blocks quickly, and lets the scheduler pack in the next batch — high turnover, high apparent throughput. A batch where a few requests generate 1,024 tokens holds KV-cache blocks for eight times as long, and under vLLM's continuous batching (see [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention)) those long-running sequences occupy slots that could otherwise serve new arrivals. The long tail of your output distribution is what actually determines how many concurrent sequences the GPU can hold, and therefore how much load it can absorb.

### The workload is a distribution, not a point

The fix is to make your load test replay the real distribution. There are three levels of fidelity, in increasing order of trustworthiness:

1. **Synthetic with a realistic distribution.** Sample input and output lengths from distributions that match your production percentiles — for example, log-normal input lengths with the right p50 and p95, and a separate output-length distribution. This is the minimum bar. Tools like NVIDIA GenAI-Perf let you specify mean and standard deviation for input and output tokens directly.
2. **Public trace replay.** vLLM's `benchmark_serving.py` ships with a ShareGPT dataset loader that replays real conversation lengths. This is much better than synthetic because the *correlation* between input and output length is preserved — real long prompts tend to get long answers, and that correlation changes the memory pressure.
3. **Your own production trace.** The gold standard. Log the input length, output length, and arrival timestamp of a representative window of real traffic (you do not need the content — just the lengths and times), then replay exactly that. This captures your specific length mix, your specific input/output correlation, and, if you replay the timestamps, your specific burstiness.

The practical rule: **never size a fleet off a fixed-prompt number, and never trust a load test whose length distribution you cannot describe in one sentence.** If someone hands you a throughput figure, the first question is "what was the input and output length distribution?" If the answer is "128 by 128" or "I don't know," the number tells you nothing about your capacity.

### Building a production trace

The single highest-leverage thing you can do for load-test fidelity is capture a real trace, and it is cheaper than most people assume because you do not need the content. For each request, log four fields: the arrival timestamp (millisecond precision), the input length in tokens, the output length in tokens (what the model actually generated, from the response usage field), and optionally the endpoint or model if you serve several. That is it — no prompt text, no completion text, so the trace carries no user PII and clears privacy review easily. A day of production traffic sampled at even 1% gives you a length distribution and an arrival pattern that no synthetic model will match.

Two things to get right when you capture it. First, capture a *representative window*, not a convenient one. If your traffic has a strong diurnal cycle, a trace from 3 a.m. will have both the wrong length mix (night traffic often differs) and the wrong burstiness. Capture across a full peak period, or better, capture the specific busy hour you are sizing for. Second, preserve the input/output length *correlation*. Long prompts tend to elicit long completions, and that correlation drives memory pressure — a benchmark that samples input and output lengths independently will underestimate the worst case where a long prompt also generates a long answer. Store the pairs, and replay them as pairs.

When you replay, you have a choice about arrivals. Replaying the *actual timestamps* reproduces your real burstiness and is the most faithful, but it pins the offered rate to whatever the trace contained — you cannot sweep. For a capacity sweep, keep the length pairs from the trace but *regenerate* the arrival process at each target rate (Poisson by default, or a burstier process if your trace's inter-arrival coefficient of variation is high — more on that below). That way you hold the workload's length distribution fixed while varying the one axis you are sweeping. The custom generator above does exactly this: it cycles through the trace's length pairs while injecting a fresh Poisson schedule at the target rate.

One subtlety worth naming: the output length is not something you fully control from the client, because the *model* decides when to stop. If you send `max_tokens=2048` but the model emits an end-of-sequence token at 200, you measured a 200-token decode, not a 2,048-token one. For faithful replay you either force exact output lengths by disabling the stop token and pinning `max_tokens` (which measures the worst case), or you replay real prompts and let the model generate naturally (which measures the realistic case). Both are defensible; mixing them silently is how benchmarks drift from reality.

## The knee: where queue delay explodes

Once your load test replays a realistic workload, the next question is what to *do* with it. The answer is not "measure the throughput." It is "find the knee." The knee is the single most important concept in capacity planning, and it comes straight out of queuing theory.

Here is the mechanics. Model the server as a queue with a service capacity `μ` (the maximum rate at which it can complete requests, in requests per second) and an arrival rate `λ` (the offered load). As long as `λ < μ`, the server keeps up; requests occasionally wait behind others, but the queue drains. As `λ` approaches `μ`, waiting time grows — and it does not grow linearly. For a simple M/M/1 queue, the expected time a request spends in the system is:

$$W = \frac{1}{\mu - \lambda}$$

Stare at that denominator. When `λ` is far below `μ`, `W` is small and nearly flat. As `λ → μ`, the denominator goes to zero and `W → ∞`. The curve is a hyperbola: latency is flat, flat, flat, then a knee, then a wall. Real LLM servers are not literally M/M/1 — service times are heavy-tailed because of the output-length distribution, and continuous batching makes the service process far more complex — but the *shape* is universal. Every queue with finite capacity has this knee, and the heavier the tail of your service-time distribution, the sharper and earlier the knee arrives.

The figure lays out the three regions across the load axis. Below the knee, latency is flat and goodput tracks the arrival rate one-for-one. At the knee, p99 TTFT hits your SLO and goodput peaks. Past the knee, queue delay explodes and goodput collapses even as you push more load in.

![Grid showing three load regions — below the knee with flat latency, at the knee where p99 meets the SLO, and above the knee where queue delay explodes and goodput collapses](/imgs/blogs/load-testing-and-capacity-planning-3.webp)

Two definitions make the knee precise, and both matter for sizing.

**Throughput** is the rate of completed requests (or tokens). Past the knee, throughput can look *stable* — the server is fully utilized and finishing work as fast as it physically can — even though latency has gone through the roof. This is the trap: a throughput-only view of an overloaded server looks healthy right up until the timeouts start.

**Goodput** is the rate of requests completed *within the SLO*. This is the metric that matters. Below the knee, goodput equals throughput because every request meets its latency target. Past the knee, throughput stays high but goodput craters, because the requests are still completing — just too late to count. The distinction between throughput and goodput is the whole game, and it is why the [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management) post argues you should optimize for goodput, not raw throughput. A server serving 25 requests per second where only 8 of them meet the SLO has a *capacity of 8*, not 25.

So the operational definition of the knee, the one you will actually compute:

> **The knee is the highest offered load at which p99 (of your chosen SLO metric) still meets the target — equivalently, the highest load at which goodput still equals throughput.**

That is the single number a single replica gives you. On our running example — Llama-3.1-8B-Instruct in BF16 on one H100 80GB under vLLM, serving a chat workload with p50 input 512 tokens and p50 output 256 tokens, against an SLO of p99 TTFT ≤ 2.0 seconds — that number is **24 requests per second**. Everything else in this post is arithmetic on top of 24.

### Why heavy tails sharpen the knee

The M/M/1 formula assumes exponential service times, but LLM service times are far from exponential — they are heavy-tailed, because output lengths are heavy-tailed and a request's service time is roughly proportional to its output length. The more general result for a single-server queue with arbitrary service-time distribution is the **Pollaczek–Khinchine formula**, which gives the mean waiting time in queue:

$$W_q = \frac{\rho}{1 - \rho} \cdot \frac{1 + C_s^2}{2} \cdot \frac{1}{\mu}$$

where `ρ = λ/μ` is the utilization and `C_s` is the coefficient of variation of the service time (standard deviation divided by mean). The first factor, `ρ/(1−ρ)`, is the same hyperbola that blows up as utilization approaches one — that is the knee. The second factor, `(1 + C_s²)/2`, is the multiplier that heavy tails add. For exponential service times `C_s = 1` and the factor is 1; for the highly variable service times of an LLM (short chat replies mixed with long document generations), `C_s` can be 2, 3, or more, and the factor becomes 2.5, 5, or larger. In words: **the more variable your output lengths, the longer the queue at any given utilization, and the earlier and sharper the knee arrives.** This is the queuing-theory reason the length distribution dominates, restated as a formula. A workload with a fat output tail hits its knee at a *lower* utilization than a uniform-length workload on the same hardware, which is exactly why you cannot extrapolate a knee measured on one length mix to another.

There is a mechanism-level caveat: continuous batching complicates the single-server picture, because the "server" processes many sequences at once and long sequences degrade the whole batch rather than blocking a single queue slot. But the qualitative conclusion is unchanged and, if anything, stronger — a few long generations in a batch slow every other sequence in that batch, so the tail's influence on the knee is at least as large as the P-K formula suggests. Measure, do not extrapolate.

### Why the tail decides the knee, not the mean

A common failure is to find the knee using the *mean* or *median* latency, which sits comfortably below the SLO long after p99 has blown through it. The tail is where SLO violations live, and the tail is heavy for LLMs precisely because of the output-length distribution and the head-of-line blocking that long sequences cause. When a 2,000-token generation is monopolizing a batch slot, the short requests queued behind it wait — and those waits land in your p99, not your p50. Always find the knee with the percentile your SLO is written against. If your SLO says "p99 TTFT ≤ 2s," the knee is where **p99** TTFT crosses 2s, full stop. Finding it with the mean will overstate capacity by a wide margin, and you will discover the real knee in production.

## Open-loop vs closed-loop: the discipline that finds the truth

You now know what you are looking for: the offered load at which p99 crosses the SLO. To measure it correctly you have to inject load correctly, and this is where the second big mistake lives — the one that inflates capacity numbers even when the length mix is right. It is the difference between an **open-loop** and a **closed-loop** load test.

A **closed-loop** test uses a fixed number of concurrent clients (call it `C`). Each client sends a request, *waits for the full response*, then immediately sends the next one. The offered load is not something you set directly; it emerges from `C` and how fast the server replies. A **open-loop** test injects requests at a fixed *arrival rate* `λ` — a new request goes out every `1/λ` seconds on average, whether or not the previous ones have come back. The arrivals are independent of the server's speed.

This sounds like a minor implementation detail. It is the difference between a capacity number you can trust and one that will get you paged. The figure makes the contrast concrete: the closed-loop test throttles itself the instant the server slows, hides the queue, and overstates capacity; the open-loop test injects a fixed rate and exposes the real knee.

![Before-and-after comparison showing a closed-loop test with fixed concurrency reporting 60 requests per second while an open-loop test with a fixed arrival rate reveals the true 24 requests-per-second capacity](/imgs/blogs/load-testing-and-capacity-planning-4.webp)

Here is why closed-loop lies, and it is worth deriving because the intuition is load-bearing. This is the mechanics block for this section.

### Little's Law, and why closed-loop self-throttles

**Little's Law** is the one piece of queuing theory every serving engineer should have memorized. For any stable system:

$$L = \lambda W$$

where `L` is the average number of requests *in the system* (being served or waiting), `λ` is the throughput (arrival rate = completion rate in steady state), and `W` is the average time a request spends in the system. It holds regardless of the arrival distribution, the service distribution, or the scheduling policy. It is an accounting identity, not a model.

Now apply it to a closed-loop test with `C` clients and zero think time. Every client is always either waiting or being served, so `L = C` exactly. Rearranging Little's Law:

$$\lambda = \frac{L}{W} = \frac{C}{W}$$

Read that carefully. In a closed-loop test, the offered rate `λ` is `C / W`. When the server slows down — when `W` grows because the queue is building — `λ` **automatically drops**. The test *cannot* offer more load than the server can currently absorb, because every client is blocked waiting for a reply before it sends the next request. The queue never grows without bound. The test self-throttles into a comfortable equilibrium and reports the throughput at that equilibrium as if it were sustainable capacity.

This is why a closed-loop test at `C = 64` might report 60 requests per second and look perfectly stable, while the true open-loop capacity is 24. At 64 concurrent clients each experiencing a 1-second average response time, `λ = 64 / 1.06 ≈ 60`. The clients are happy because they are getting served — but only because they are politely waiting their turn. In production, your users do not wait their turn. New requests arrive at whatever rate the outside world generates, independent of how backed up you are. That is an open-loop arrival process, and if that rate exceeds `μ`, the queue grows without bound and every request's latency climbs until something times out.

The consequence is a hard rule: **size capacity from an open-loop test. A closed-loop test measures how the system behaves at a fixed concurrency, which is a useful thing to know, but it structurally cannot find the knee, because it cannot overload the server.** Closed-loop tests answer "if I hold 64 users, what latency do they see?" Open-loop tests answer "at what arrival rate does the SLO break?" Only the second question sizes a fleet.

There is a real-world nuance that rescues closed-loop somewhat: if you add **think time** between a client's requests, and you use *many* clients each with a realistic think-time distribution, a closed-loop test starts to approximate an open-loop arrival process (this is the classic result behind why "N users with T think time" approximates a Poisson source at rate `N/(T + R)`). But getting this right is fiddly, and the failure mode — too few clients, too little think time — silently reverts to self-throttling. When capacity is the question, inject a rate directly and remove all doubt.

### Burstiness: Poisson is the optimistic case

Open-loop injection settles the *rate*, but there is a second parameter that matters almost as much: the *shape* of the arrival process. Poisson arrivals — independent, exponentially-distributed inter-arrival times — are the standard default, and `vllm bench serve --burstiness 1.0` gives you exactly that. But real traffic is often burstier than Poisson. Users cluster: a push notification goes out and a thousand people open the app in the same ten seconds; a batch job wakes up and dumps its backlog; an upstream retry policy synchronizes a wave of retries. Burstier-than-Poisson arrivals mean that even when your *average* rate is below the knee, momentary clusters push you over it, and those clusters land in your p99.

The parameter to reason about is the **coefficient of variation of the inter-arrival times**, `C_a`. Poisson has `C_a = 1`. Perfectly uniform (one request exactly every `1/λ` seconds) has `C_a = 0` and is the gentlest possible load. Bursty traffic has `C_a > 1`, sometimes well above. There is a generalized queuing approximation (the Kingman formula) that makes the effect precise:

$$W_q \approx \frac{\rho}{1 - \rho} \cdot \frac{C_a^2 + C_s^2}{2} \cdot \frac{1}{\mu}$$

Now *both* variability sources — arrival burstiness `C_a` and service-time variability `C_s` — add to the queue multiplier. A workload that is bursty in arrivals *and* heavy-tailed in service (which describes most real LLM chat traffic) gets hit twice. The practical consequences:

- **Measure your real `C_a` from the trace** and set the load generator's burstiness to match. Testing with uniform arrivals (`C_a = 0`) is a common way to accidentally overstate the knee, because it is the easiest possible arrival pattern.
- **Default to Poisson if you have no better information**, but know it is optimistic relative to clustered real traffic. `vllm bench serve --burstiness 0.5` and the custom generator (swap the exponential for a bursty inter-arrival draw) let you test hostile arrivals.
- **The knee you size from should reflect the burstiness you will actually see.** If your traffic is bursty, the sustainable *average* rate is lower than the Poisson knee, because the bursts eat your headroom.

This is also the queuing-theory justification for the burst-headroom replica in the capacity math below: the headroom is what absorbs the `C_a > 1` clusters without pushing steady-state utilization past the knee.

One more distinction that trips people up. **Concurrency is an output, not an input, of an open-loop test.** In open loop you set `λ`; the resulting concurrency `L = λW` is what the server ends up holding, and it is a *result* you measure, not a knob you turn. When you read a benchmark that says "we ran at concurrency 128," ask whether 128 was the injected arrival mechanism (closed-loop, suspect for capacity) or the observed in-flight count at a set rate (open-loop, trustworthy). vLLM's `benchmark_serving.py` supports both a `--request-rate` mode (open-loop) and a max-concurrency mode; use the rate mode for capacity work.

## The tools: what actually measures a knee

You do not have to write the load generator yourself, though you will often want to. The ecosystem has a handful of tools, and they are not interchangeable — the ones that inject a fixed arrival rate and replay a real length mix produce a knee you can size from, while the generic HTTP tools measure your gateway, not your engine. The matrix below is the decision aid.

![Matrix comparing vLLM bench serve, GenAI-Perf, locust or k6, and a custom async generator across loop model, LLM metrics, length mix support, and best use case](/imgs/blogs/load-testing-and-capacity-planning-5.webp)

Here is the same comparison as a table you can put in a design doc, with the specifics that matter:

| Tool | Loop model | LLM-native metrics | Length-mix support | Streaming/TTFT | Best for |
|---|---|---|---|---|---|
| **vLLM `benchmark_serving.py` / `vllm bench serve`** | Open-loop (`--request-rate`) or concurrency | TTFT, TPOT, ITL, throughput, goodput | ShareGPT replay, sonnet, random with set lengths | Yes, native | Tuning and sizing a vLLM/OpenAI-compatible endpoint |
| **NVIDIA GenAI-Perf** | Open-loop (request-rate) and concurrency sweeps | Full LLM set incl. per-token latencies | Synthetic with mean/stddev, plus custom datasets | Yes, native | Triton/NIM endpoints, standardized reports |
| **locust** | Closed-loop (users), open-loop via custom `LoadShape` | Only what you script | You script it | Only if you parse the stream yourself | Whole-gateway load, mixed HTTP + LLM traffic |
| **k6** | Closed-loop (VUs) or open-loop (`constant-arrival-rate` executor) | Only what you script | You script it | Manual | Gateway, auth, rate-limit, and routing load |
| **Custom async generator** | Whatever you build (open-loop recommended) | Whatever you log | Exact production trace | Yes, you own it | Capacity sign-off with your real workload |

A few opinions to save you time:

- **For sizing a vLLM or any OpenAI-compatible endpoint, start with `vllm bench serve` (the modern CLI wrapping `benchmark_serving.py`).** It is open-loop-capable, it reports TTFT/TPOT/goodput natively, and it can replay ShareGPT. It is the fastest path to a defensible number.
- **For Triton Inference Server or NVIDIA NIM, use GenAI-Perf.** It understands the Triton and OpenAI protocols, produces standardized reports, and its synthetic length controls are good enough for a first pass.
- **Reach for locust or k6 when the question is about the whole request path** — the API gateway, auth, rate limiting, the load balancer's [least-outstanding-requests routing](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) — not the engine's token economics. k6's `constant-arrival-rate` executor is a genuine open-loop injector, which makes k6 usable for capacity work if you are willing to script the streaming parse. locust defaults to closed-loop and needs a custom `LoadShape` to inject a rate.
- **Write a custom async generator when you need to replay your exact production trace and record exactly the percentiles your SLO is written against.** This is the tool for the final capacity sign-off, and it is fifty lines of Python. We build it in the next section.

The unifying principle: a load-testing tool for capacity must do three things — inject a fixed arrival rate (open-loop), replay a realistic length mix, and measure the SLO percentile natively (streaming TTFT and per-token TPOT, not just end-to-end latency). Any tool that misses one of the three is measuring something, but not your capacity.

## The methodology: sweeping offered load

Here is the actual procedure. You do not run one load test; you run a *sweep*. You pick a range of arrival rates that brackets the expected knee, run the workload at each rate for long enough to reach steady state, and record the SLO percentiles at each. Then you plot latency and goodput against offered load and read off the knee.

Start with the tool invocations, because you will reach for these constantly. First, launch the server (our running example):

```bash
# Serve Llama-3.1-8B-Instruct on one H100 with vLLM, OpenAI-compatible API.
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --port 8000
```

Now sweep the offered load with `vllm bench serve`. The key flag is `--request-rate`, which puts it in open-loop mode; run it once per rate in your sweep:

```bash
# One point in the sweep: inject 24 req/s, Poisson arrivals, ShareGPT lengths.
for RATE in 8 12 16 20 22 24 26 28 30 32; do
  vllm bench serve \
    --backend openai-chat \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint /v1/chat/completions \
    --base-url http://localhost:8000 \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate "$RATE" \
    --burstiness 1.0 \
    --num-prompts $(( RATE * 120 )) \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result --result-dir ./sweep --result-filename "rate_${RATE}.json"
done
```

Three details that matter. `--request-rate` sets the open-loop arrival rate (requests per second); without it the tool runs closed-loop as fast as it can, which is exactly the mistake we are avoiding. `--burstiness 1.0` gives Poisson (exponential inter-arrival) arrivals, which is the right default for independent user traffic; lower values make arrivals burstier (closer to a real thundering-herd), higher values make them more uniform. `--num-prompts` is scaled with the rate so each point runs for roughly the same wall-clock duration (about two minutes here) — you need enough requests at each rate to get a stable p99, which means at least a few thousand requests per point.

The GenAI-Perf equivalent, for a Triton or NIM endpoint, sweeping request rate with synthetic lengths that match your production percentiles:

```bash
genai-perf profile \
  -m meta-llama/Llama-3.1-8B-Instruct \
  --service-kind openai --endpoint-type chat \
  --url http://localhost:8000 \
  --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 256 \
  --output-tokens-mean 256 --output-tokens-stddev 128 \
  --request-rate 24 \
  --measurement-interval 120000 \
  --profile-export-file rate_24.json
```

### A custom open-loop load generator

For the final sign-off you want your exact production trace and full control over what gets logged. Here is a compact async open-loop generator. It launches requests on a Poisson schedule at a target rate, does not wait for one to finish before starting the next (that is what makes it open-loop), streams the response so it can time the first token separately from the inter-token latency, and records per-request TTFT and TPOT so you can compute any percentile afterward.

```python
import asyncio, time, json, random, argparse
import aiohttp
import numpy as np

async def one_request(session, url, model, prompt, max_tokens, results):
    """Fire one streaming request; record TTFT and per-token latencies."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    ttft = None
    token_times = []
    try:
        async with session.post(url, json=payload) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - start          # first token arrived
                else:
                    token_times.append(now)     # subsequent tokens
    except Exception as e:                       # timeout, reset, 5xx
        results.append({"ok": False, "error": str(e)})
        return
    # TPOT = mean inter-token gap during decode.
    if len(token_times) >= 2:
        gaps = np.diff(token_times)
        tpot = float(gaps.mean())
    else:
        tpot = float("nan")
    results.append({"ok": True, "ttft": ttft, "tpot": tpot,
                    "e2e": time.perf_counter() - start,
                    "out_tokens": len(token_times) + 1})

async def run_rate(url, model, trace, rate, duration, results):
    """Open-loop: inject Poisson arrivals at `rate` req/s for `duration` s."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60)) as session:
        tasks = []
        deadline = time.perf_counter() + duration
        i = 0
        while time.perf_counter() < deadline:
            row = trace[i % len(trace)]          # replay real length mix
            i += 1
            # Launch WITHOUT awaiting — the queue can grow, as in production.
            tasks.append(asyncio.create_task(
                one_request(session, url, model,
                            row["prompt"], row["max_tokens"], results)))
            # Poisson: exponential inter-arrival with mean 1/rate.
            await asyncio.sleep(random.expovariate(rate))
        await asyncio.gather(*tasks)             # let in-flight finish

def summarize(results, ttft_slo, tpot_slo):
    ok = [r for r in results if r.get("ok")]
    n, errors = len(results), len(results) - len(ok)
    ttfts = np.array([r["ttft"] for r in ok])
    tpots = np.array([r["tpot"] for r in ok if r["tpot"] == r["tpot"]])
    # Goodput: fraction meeting BOTH SLO gates, as a rate.
    good = sum(1 for r in ok
               if r["ttft"] <= ttft_slo and r["tpot"] <= tpot_slo)
    return {
        "completed": len(ok), "errors": errors,
        "ttft_p50": float(np.percentile(ttfts, 50)),
        "ttft_p99": float(np.percentile(ttfts, 99)),
        "tpot_p99": float(np.percentile(tpots, 99)),
        "good_frac": good / max(n, 1),
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="+", type=float, required=True)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--trace", default="prod_trace.json")
    args = ap.parse_args()
    trace = json.load(open(args.trace))          # [{"prompt":..,"max_tokens":..}]
    for rate in args.rates:
        results = []
        await run_rate(args.url, args.model, trace, rate, args.duration, results)
        summary = summarize(results, ttft_slo=2.0, tpot_slo=0.05)
        summary["rate"] = rate
        summary["goodput"] = summary["good_frac"] * rate
        print(json.dumps(summary))

if __name__ == "__main__":
    asyncio.run(main())
```

The load-bearing line is `asyncio.create_task(...)` followed by `await asyncio.sleep(random.expovariate(rate))` — it launches the next request on the arrival schedule *without waiting* for the previous one to complete. That is what lets the in-flight count (and the queue) grow when the server falls behind, which is exactly the behavior you must reproduce to find the knee. A closed-loop version would `await one_request(...)` before sleeping, and it would never overload the server. The trace replay (`trace[i % len(trace)]`) is what keeps the length mix honest.

### Warmup, duration, and steady state

Three measurement hygiene rules separate a trustworthy sweep from a noisy one, and skipping them is how you get a knee that does not reproduce.

**Discard the warmup.** The first requests against a freshly-started server hit cold caches — the CUDA graphs are not captured, `torch.compile` may still be tracing, the prefix cache is empty, the GPU clocks have not spun up. Those requests are slow for reasons that have nothing to do with steady-state capacity. Run each rate point for a warmup period (30-60 seconds is usually enough) and throw those results away before you start recording. `vllm bench serve` does some warmup; for the custom generator, tag the first N seconds and exclude them in `summarize`.

**Run each point to steady state.** At a given rate, the queue takes time to reach its equilibrium length — especially near the knee, where the equilibrium is a large number and it fills slowly. If you measure for ten seconds you catch the queue mid-fill and record an optimistically low p99. Run each point long enough that the in-flight count has stabilized and you have collected enough completed requests for a stable p99. A p99 needs at least a few hundred samples to be meaningful and a few thousand to be stable; at 24 req/s over two minutes you get ~2,880 requests, which is comfortable. Below ~10 req/s you may need longer windows to accumulate enough samples.

**Watch for non-stationarity.** If the p99 at a fixed rate keeps climbing throughout the measurement window instead of leveling off, you are *past* the knee at that rate — the queue is not reaching equilibrium, it is growing without bound, and any single p99 number you report is just a snapshot of an unbounded process. That climbing-p99 signal is itself diagnostic: the last rate at which p99 *stabilizes* is at or just below the knee. Plot p99 over time within each point, not just the final aggregate, and you will see the knee announce itself as the rate where the within-window curve stops flattening.

### Multi-metric SLOs and how goodput combines them

Real SLOs are rarely a single number. A chat product typically has at least three gates: p99 TTFT (the response must *start* fast enough to feel responsive), p95 or p99 TPOT (the tokens must *keep coming* fast enough to read comfortably — roughly ≤ 50ms per token to beat reading speed), and an error-rate ceiling (say, < 0.1% of requests may fail or time out). A request counts toward goodput only if it clears *all* the gates. This matters because the different gates bind at different points on the load curve.

TTFT is usually the *first* to break under load, because queuing delay lands entirely on the first token — a request waiting in the admission queue accrues TTFT, not TPOT. TPOT degrades more gently at first (continuous batching keeps per-token latency reasonable until the batch is genuinely saturated) but then falls off a cliff when the GPU can no longer keep the batch fed. The error rate is usually the *last* gate to break, firing only once the queue is deep enough that client timeouts trigger. So on our running example, the knee is set by TTFT p99 (it hits 2.0s at 24 req/s while TPOT p99 is still a healthy 41ms and errors are zero). But on a different workload — say very long outputs where each request occupies the batch for a long time — TPOT can be the binding constraint, and the knee is wherever TPOT p99 crosses its gate first.

The rule: **compute goodput against the conjunction of all your SLO gates, and let the data tell you which gate binds.** The knee-finder script above checks both TTFT and TPOT; add the error-rate gate the same way. Do not assume TTFT always binds — measure it. A capacity plan built on the wrong binding metric sizes for the wrong constraint.

### Reading the sweep

Run that generator (or the `vllm bench serve` loop) across the rate range and you get a table. This is the load-sweep table for our running example — Llama-3.1-8B on one H100, chat workload, SLO p99 TTFT ≤ 2.0s and p95 TPOT ≤ 50ms:

| Offered rate (req/s) | TTFT p99 (s) | TPOT p99 (ms) | Throughput (req/s) | Goodput (req/s) | Errors |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.41 | 22 | 8.0 | 8.0 | 0 |
| 12 | 0.52 | 24 | 12.0 | 12.0 | 0 |
| 16 | 0.63 | 26 | 16.0 | 16.0 | 0 |
| 20 | 0.94 | 30 | 20.0 | 20.0 | 0 |
| 22 | 1.31 | 34 | 22.0 | 22.0 | 0 |
| **24** | **1.90** | **41** | **24.0** | **24.0** | **0** |
| 26 | 3.40 | 55 | 24.9 | 12.1 | 0 |
| 28 | 6.05 | 78 | 25.1 | 3.2 | 0 |
| 30 | 9.20 | 122 | 24.3 | 0.4 | 41 |
| 32 | 14.1 | 190 | 23.1 | 0.0 | 380 |

Read it top to bottom and the knee is unmistakable. Up to 24 req/s, goodput equals throughput equals offered load — every request meets the SLO. At 24, p99 TTFT is 1.90s, just under the 2.0s target: this is the knee. At 26, throughput actually *ticks up* to 24.9 (the server is now saturated and completing work as fast as it can) but goodput *collapses* to 12.1 — half the requests are now missing the SLO. By 30 req/s the server is dropping requests, and by 32 it is in full retry-storm territory with 380 errors. Notice that a throughput-only chart would show a gently rising line that plateaus around 25 and looks completely healthy across this entire range. The goodput column is the one that screams.

#### Worked example: finding the knee from a sweep

Suppose a teammate hands you a different sweep and asks for the number. The SLO is p99 TTFT ≤ 1.5s. The data:

| Rate | TTFT p99 (s) | Goodput (req/s) |
|---:|---:|---:|
| 30 | 0.7 | 30 |
| 40 | 1.1 | 40 |
| 45 | 1.4 | 45 |
| 48 | 1.9 | 31 |
| 50 | 2.8 | 12 |

Walk it: at 45 req/s, p99 TTFT is 1.4s (under 1.5) and goodput still equals the rate — SLO holds. At 48, p99 jumps to 1.9s (over 1.5) and goodput drops to 31 — SLO broken. **The knee is 45 req/s.** Not 48 (throughput there is nominally higher, but a third of requests miss the SLO), and definitely not 50 (the peak throughput point, which is a disaster). The knee is the *last* rate where goodput still tracks the offered load. If you want a little more precision you can run a finer sweep between 45 and 48, but for sizing, 45 is your per-replica number, and you would typically round *down* to stay safely on the flat part of the curve.

The mechanics of the collapse are worth naming once more: past the knee, arrivals exceed service capacity, so `L = λW` forces the in-flight count and the wait time up together. Every extra request you inject makes the queue longer, which makes every request's TTFT longer, which is why the p99 column goes vertical. The server is not broken — it is doing exactly what a saturated queue does. The load ramp below shows the same story as a time series: latency stays flat through the ramp, bends at the knee, then runs away as the queue grows without bound and requests start timing out.

![Timeline of a load ramp showing flat latency during warmup and ramp, a bend at the knee at 24 requests per second, then runaway latency and collapse past the knee](/imgs/blogs/load-testing-and-capacity-planning-6.webp)

### A knee-finder script

You do not want to eyeball the knee every time. This script ingests the JSON lines the generator prints, finds the knee, and plots the curve so the knee is visible in a design review:

```python
import json, sys
import matplotlib.pyplot as plt

def load_sweep(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return sorted(rows, key=lambda r: r["rate"])

def find_knee(rows, ttft_slo=2.0, tpot_slo=0.05, goodput_tol=0.98):
    """Knee = highest rate where p99 meets SLO AND goodput ~= offered rate."""
    knee = None
    for r in rows:
        meets_slo = r["ttft_p99"] <= ttft_slo and r["tpot_p99"] <= tpot_slo
        tracks = r["goodput"] >= goodput_tol * r["rate"]
        if meets_slo and tracks:
            knee = r            # keep the highest passing rate
        else:
            break               # once it breaks, it stays broken
    return knee

def main(path):
    rows = load_sweep(path)
    knee = find_knee(rows)
    rates = [r["rate"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rates, [r["ttft_p99"] for r in rows], "o-", label="TTFT p99 (s)")
    ax1.axhline(2.0, ls="--", color="crimson", label="TTFT SLO 2.0s")
    ax1.set_xlabel("offered load (req/s)"); ax1.set_ylabel("TTFT p99 (s)")
    ax2 = ax1.twinx()
    ax2.plot(rates, [r["goodput"] for r in rows], "s-",
             color="seagreen", label="goodput (req/s)")
    ax2.set_ylabel("goodput (req/s)")
    if knee:
        ax1.axvline(knee["rate"], ls=":", color="black")
        ax1.set_title(f"Knee = {knee['rate']:.0f} req/s "
                      f"(TTFT p99 {knee['ttft_p99']:.2f}s)")
        print(f"KNEE: {knee['rate']:.0f} req/s per replica")
    fig.tight_layout(); fig.savefig("knee.svg")

if __name__ == "__main__":
    main(sys.argv[1])
```

The `find_knee` logic encodes the definition directly: walk rates ascending, keep the highest one where p99 meets the SLO *and* goodput still tracks the offered rate, and stop at the first failure (once a queue starts running away it does not recover at higher load). The `break` matters — without it, a lucky momentary dip at a higher rate could be mistaken for the knee. The plot gives you the picture to put in the doc: two curves, the SLO line, and a vertical marker on the knee.

## The capacity math: from one knee to a fleet

Now the payoff. You have one number — 24 req/s per replica at the SLO — and a target peak load. The job is to turn that into a GPU count with the right headroom. Every layer of the calculation transforms the knee: divide the peak by it to get base replicas, add headroom for bursts and failover, then price the GPUs. The figure stacks the transformation.

![Layered stack showing the transformation from a single replica at the 24 requests-per-second knee, to a fleet sized for 50 QPS, to headroom plus N+1, to a monthly cost of roughly 22,000 US dollars](/imgs/blogs/load-testing-and-capacity-planning-7.webp)

### The base calculation

The naive fleet size is straightforward division:

$$\text{base replicas} = \left\lceil \frac{\text{peak QPS}}{\text{per-replica knee}} \right\rceil$$

For a 50 QPS target and a 24 req/s knee, that is `ceil(50/24) = ceil(2.08) = 3` replicas. If that were the whole story, capacity planning would not need a blog post. It is not, for three reasons, and each one adds capacity.

**Reason 1: peak is not mean.** "50 QPS" is usually a daily average or a business estimate. Real traffic has a diurnal peak, and within the peak hour there is a peak minute, and within that a burst. If your daily mean is 50 but your busy-minute p99 is 1.35× that, you must size for ~68 QPS, not 50, or you will break the SLO every day at the busy minute. Always size for the peak of the arrival process at the granularity of your SLO window, not the average. Pull this factor from your real traffic; do not guess it.

**Reason 2: you cannot run a replica at 100% of its knee.** The knee is the *edge* of the SLO. Running steady-state traffic right at the knee leaves zero room for the micro-bursts and length spikes that happen constantly — one unusually long generation, one momentary arrival cluster, and you are over the edge. Run each replica at no more than ~70-80% of its knee in steady state, which means adding a **burst-headroom** replica so the fleet's peak utilization sits comfortably below the knee.

**Reason 3: failover. N+1.** If any single replica can die — a GPU falls off the bus, a node gets preempted, a rolling deploy takes one out — and you sized to exactly meet peak with N replicas, then losing one puts you *below* capacity precisely when you can least afford it. The standard answer is **N+1**: provision one spare replica beyond what peak requires, so the loss of any single replica still leaves enough to serve peak within SLO. For higher-availability tiers you go N+2 or spread across availability zones.

Putting it together for the running example:

#### Worked example: size a fleet for 50 QPS at p99 TTFT ≤ 2s

- **Per-replica knee:** 24 req/s (measured, single H100).
- **Base for mean:** `ceil(50 / 24) = 3` replicas. This covers the *average* but not the peak.
- **Burst headroom:** the busy-minute peak is ~68 QPS. Three replicas cap out at `3 × 24 = 72` req/s — technically above 68, but that runs them at `68/72 = 94%` of the knee at peak, with no slack for length spikes. Add one replica: four replicas cap at 96 req/s, so the busy-minute peak sits at `68/96 = 71%` of aggregate knee. That is the headroom. **Four serving replicas.**
- **N+1 failover:** add one spare so losing any replica still leaves four serving. **Five replicas total.**
- **GPUs:** each replica is one H100 → **5 × H100 80GB.**

So the honest answer to "how many GPUs for 50 QPS at p99 TTFT ≤ 2s" is **five, not the two or three a fixed-prompt benchmark would have suggested.** The gap between the naive number and the real one is where launches go to die.

A note on when the N+1 spare *is* the burst headroom: when the base fleet is small (say two replicas), a single N+1 spare already adds 50% capacity, which usually covers both failover and modest bursts — you do not need a separate burst replica on top. As the fleet grows, one spare is a smaller fraction of the whole, so you add a dedicated burst replica in addition to the failover spare. The rule scales: small fleets get proportionally more relative headroom from N+1 for free.

Here is the calculation as a reusable function. Feed it your measured knee and traffic shape; it returns the fleet size and flags the assumptions:

```python
import math
from dataclasses import dataclass

@dataclass
class CapacityPlan:
    per_replica_knee: float      # req/s at SLO, MEASURED (open-loop)
    mean_qps: float              # target average load
    peak_to_mean: float = 1.35   # busy-minute peak / daily mean, from traffic
    max_util: float = 0.75       # never run a replica above this fraction of knee
    failover_spares: int = 1     # N+1 => 1; N+2 => 2
    gpus_per_replica: int = 1    # tensor-parallel degree

    def compute(self):
        peak = self.mean_qps * self.peak_to_mean
        usable = self.per_replica_knee * self.max_util
        serving = math.ceil(peak / usable)              # covers peak w/ headroom
        # A small fleet's failover spare doubles as burst headroom.
        if serving <= 2:
            fleet = serving + self.failover_spares
        else:
            fleet = serving + self.failover_spares
        gpus = fleet * self.gpus_per_replica
        return {
            "peak_qps": round(peak, 1),
            "usable_per_replica": round(usable, 1),
            "serving_replicas": serving,
            "fleet_replicas": fleet,
            "gpus": gpus,
            "aggregate_knee": round(fleet * self.per_replica_knee, 1),
            "peak_util_pct": round(100 * peak / (fleet * self.per_replica_knee), 1),
        }

# Running example: 50 QPS, H100 knee 24 req/s.
plan = CapacityPlan(per_replica_knee=24, mean_qps=50)
print(plan.compute())
# {'peak_qps': 67.5, 'usable_per_replica': 18.0, 'serving_replicas': 4,
#  'fleet_replicas': 5, 'gpus': 5, 'aggregate_knee': 120.0, 'peak_util_pct': 56.2}
```

The function makes the assumptions explicit and auditable, which is the entire point — a capacity plan is only as trustworthy as its stated `peak_to_mean` and `max_util`, and those should come from your traffic data and your SLO tolerance, not from a default. Change `peak_to_mean` to reflect a spikier workload, or `max_util` down to 0.6 for a tighter SLO, and the fleet size responds. Set `gpus_per_replica=2` for a tensor-parallel deployment and the GPU count doubles per replica.

### Little's Law as a sanity check on your fleet

Little's Law is not only the reason closed-loop tests lie; it is also a free consistency check on your capacity plan. At the knee, each replica holds some average number of in-flight requests, `L = λ × W`, where `λ` is the knee rate and `W` is the mean end-to-end latency at the knee. For our example: at 24 req/s with a mean end-to-end latency of roughly 8 seconds (1.9s to first token plus ~250 tokens at ~25ms each), each replica holds `L = 24 × 8 = 192` concurrent sequences on average. That number had better be less than the `--max-num-seqs` you configured (256 here) and less than what the KV cache can hold given your `--gpu-memory-utilization`. If Little's Law says you need 192 concurrent sequences but your KV cache only fits 128, the server will be preempting or queuing before it ever reaches the rate your sweep claimed — and the two measurements disagree, which means one of them is wrong. Reconciling them is how you catch a misconfigured `max-num-seqs` or an over-optimistic sweep.

Run the same check on the whole fleet. Five replicas at the knee hold `5 × 192 = 960` concurrent sequences at peak. That is the number your KV cache, your `max-num-seqs`, and your monitoring dashboards should all agree on. When the observed in-flight count in production drifts above what the plan predicted, either your latency is worse than the sweep (the knee moved) or your rate is higher than forecast — either way, Little's Law turns a vague "feels slow" into a specific, checkable discrepancy. Keep `L`, `λ`, and `W` on the same dashboard and the identity `L = λW` becomes a live invariant you can alert on.

### Modeling burst vs steady state

The `peak_to_mean` factor deserves its own paragraph because it is where over- and under-provisioning both hide. There are two regimes to plan for:

- **Steady, predictable diurnal traffic** (most consumer chat products): the peak-to-mean ratio is stable — maybe 1.3× to 2× depending on how concentrated your usage is by time zone. You size the *always-on* fleet for the daily peak and let [autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) shave the trough. The knee is your scale-up trigger: scale out when a replica's offered load approaches, say, 80% of its knee, so a new replica is warm before the current ones hit the edge.
- **Spiky, event-driven traffic** (a product launch, a viral moment, a batch job kicking off): the peak-to-mean can be 5× or 10× and it arrives in seconds. Autoscaling GPUs is *slow* — pulling a large model onto a fresh GPU can take minutes, far longer than the burst. For these you either pre-provision for the spike (expensive, but the only way to hold SLO through a fast burst) or you protect the SLO with [admission control and load shedding](/blog/machine-learning/model-serving/high-concurrency-slo-management) — reject or queue the overflow rather than let it collapse the whole server past the knee.

The knee is central to both. It is the scale-up trigger for the autoscaler and the admission threshold for load shedding. A fleet that knows its per-replica knee can defend its SLO; one that does not is flying blind.

## Sizing across hardware, and the cost per token

The per-replica knee is not a property of the model alone — it is a property of the model *on a specific GPU*. Because decode is memory-bandwidth-bound, the knee scales roughly with HBM bandwidth, which means the same model and the same SLO produce very different knees, and therefore very different fleet sizes and costs, across cards. The matrix below sizes the same 50 QPS target across four hardware options.

![Matrix sizing the same 50 QPS target across H100, A100, L40S, and dual-H100 tensor-parallel configurations, showing per-replica knee, replica count, total GPUs including N+1, and cost per million tokens](/imgs/blogs/load-testing-and-capacity-planning-8.webp)

As an illustrative sizing table (your knees will differ — measure them):

| Hardware | HBM bandwidth | Knee (req/s @ SLO) | Base replicas (50 QPS) | GPUs incl. headroom + N+1 | Cost / 1M output tokens |
|---|---|---:|---:|---:|---:|
| 1×H100 80GB | ~3.35 TB/s | 24 | 3 | 5 | ~\$0.18 |
| 1×A100 80GB | ~2.04 TB/s | 14 | 4 | 6 | ~\$0.26 |
| 1×L40S 48GB | ~0.86 TB/s | 7 | 8 | 10 | ~\$0.31 |
| 2×H100 (TP=2) | ~3.35 TB/s ×2 | 38 | 2 | 6 | ~\$0.20 |

The A100 knee is roughly 60% of the H100's, tracking the ~0.61 bandwidth ratio — decode is bandwidth-bound, so this is expected, not a coincidence. The L40S, with about a quarter of the H100's bandwidth, has a knee around a third of it (a little better than the bandwidth ratio because the workload is not purely decode). Tensor-parallel across two H100s roughly 1.6× the single-card knee, not 2× — TP adds all-reduce communication on every layer, so you pay a synchronization tax (see [tensor, pipeline, and expert parallelism for serving](/blog/machine-learning/model-serving/tensor-pipeline-expert-parallelism-for-serving)). The lesson: **sticker price per GPU is the wrong thing to optimize; cost per million tokens at your SLO is the right thing, and the fastest card is often the cheapest per token even though it costs the most per hour.**

### When the knee is prefill-bound, not decode-bound

The claim that "the knee scales with HBM bandwidth" holds when decode dominates, which is the common case for chat. But some workloads are prefill-heavy — retrieval-augmented generation with 8,000-token contexts and 100-token answers, classification with long inputs and one-token outputs, reranking. For these, prefill FLOPs, not decode bandwidth, set the ceiling, and the knee scales with *compute* (tensor-core throughput) rather than memory bandwidth. This flips the hardware ranking: an H100's FP8 tensor cores give it a larger compute advantage over an A100 than its bandwidth advantage, so a prefill-bound workload sees an even wider H100-vs-A100 knee gap than a decode-bound one. It also changes which optimizations move the knee — chunked prefill and prefill/decode disaggregation matter enormously for prefill-bound traffic and barely at all for decode-bound traffic.

The way to tell which regime you are in is to look at where the GPU spends its time during the sweep, and where the knee binds. If the knee arrives while GPU compute utilization is pegged and memory bandwidth has headroom, you are prefill-bound; if bandwidth is pegged and compute has headroom, you are decode-bound. A quick tell without profiling: compute the ratio of total input tokens to total output tokens across your trace. A ratio well above ~5:1 usually means prefill-bound; well below ~2:1 means decode-bound. This matters for capacity because the two regimes respond to different levers — you do not buy your way out of a prefill bottleneck with more KV-cache headroom, and you do not fix a decode bottleneck with faster tensor cores. Measure the regime, then pick the optimization, then re-sweep. The [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) work exists precisely because these two phases have different bottlenecks and benefit from being scaled independently.

### Cost per million tokens, done honestly

The cost-per-token number is where benchmarks get abused most, because it is extremely sensitive to definitions. Pin them down.

At the knee, one H100 replica generates about `24 req/s × 256 output tokens ≈ 6,000 output tokens/second`. With an H100 at a representative on-demand rate of \$4/hour (the range across clouds is wide — roughly \$3 to \$12/hour depending on provider and commitment), the **marginal cost at the knee** is:

$$\frac{\$4/\text{hr}}{6{,}000 \text{ tok/s} \times 3600 \text{ s/hr}} \times 10^6 = \frac{\$4}{21.6 \times 10^6} \times 10^6 \approx \$0.185 \text{ per 1M output tokens}$$

That \$0.18 is the number in the sizing table, and it is the *best case*: a single replica running exactly at its knee, fully utilized, GPU cost only. It is the right number for comparing hardware, and the wrong number for forecasting your bill, for two reasons:

1. **You do not run at the knee on average.** You provision for peak plus headroom plus N+1, and the fleet spends most of the day well below the knee. The *provisioned* cost per token — total fleet cost divided by tokens actually served — is higher than the marginal cost, by exactly the ratio of provisioned capacity to average utilization. A fleet at 50% average utilization has roughly double the effective cost per token of the marginal number.
2. **GPUs are not the whole bill.** The load balancer, the CPU nodes running the API gateway and tokenization, storage for weights, egress, observability, and the on-call humans all cost money. GPU rental is typically 65-75% of a serving fleet's fully-loaded cost.

#### Worked example: the fully-loaded monthly cost of the 50 QPS fleet

- **Fleet:** 5 × H100 (from the sizing worked example).
- **GPU rental:** `5 × \$4/hr × 730 hr/month = \$14,600/month`.
- **Everything else** (gateway, LB, storage, observability, ~50% of GPU spend as a rule of thumb for a lean fleet): ~\$7,400/month.
- **Fully-loaded total:** ~**\$22,000/month** — the number in the stack figure.
- **Tokens served at the *mean* 50 QPS:** `50 req/s × 256 tok × 2.6M s/month ≈ 33 billion output tokens/month`.
- **Effective (provisioned, fully-loaded) cost:** `\$22,000 / 33,000M tokens × 1M ≈ \$0.67 per 1M output tokens`.

Sit with the gap between \$0.18 marginal and \$0.67 effective. That 3.7× is not waste to be eliminated — it is the cost of the headroom, the failover spare, the sub-knee average utilization, and the non-GPU infrastructure that a real service requires. Reporting the \$0.18 as your "cost per token" to finance and then getting a \$22k bill is how capacity plans lose credibility. Report both, labeled: marginal cost at the knee for hardware comparison, effective provisioned cost for the forecast.

### Re-test after every optimization

The knee is not a constant of nature; it is a measurement of your current stack, and it moves every time you change the stack. Turn on [FP8 quantization](/blog/machine-learning/model-serving/quantization-for-llm-serving) and the knee moves (more KV headroom, sometimes higher throughput, possibly a small accuracy cost). Enable chunked prefill, prefix caching, or speculative decoding and the knee moves. Upgrade vLLM and the knee moves — often up, sometimes down when a regression sneaks in. Change the model and everything moves. **Every optimization invalidates your capacity plan, so re-run the sweep after every one.** The whole procedure — server launch, rate sweep, knee-finder, capacity calculator — should be a script you run in CI against a canary, not a one-time heroics session before a launch. A capacity number with a date on it older than your last deploy is a guess wearing a lab coat.

## Case studies and published benchmarks

Real published numbers ground the methodology. Treat all of these as illustrative of the *shape* of the results, not as values to copy — every one depends on model, hardware, workload, and SLO, and I am summarizing at a level where approximation is unavoidable.

**vLLM's own throughput benchmarks.** The vLLM project publishes `benchmark_serving.py` results, and their headline figure from the original PagedAttention work (Kwon et al., SOSP 2023, *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*) was 2-4× higher throughput than prior systems at the same latency, driven by eliminating KV-cache fragmentation. The methodology that matters here: they swept request rate (open-loop) and plotted normalized latency against throughput, and the curves show exactly the knee shape — flat, then a sharp bend. The 2-4× is a shift of the knee to the right, not a change in its shape. When you read their numbers, note the input/output length distribution (ShareGPT) they used, because that is what makes the numbers meaningful.

**NVIDIA GenAI-Perf and the TTFT/TPOT trade-off.** NVIDIA's GenAI-Perf documentation and the TensorRT-LLM benchmark reports consistently show the same phenomenon from the other direction: as you increase concurrency (or request rate), TTFT rises faster than TPOT, because queuing delay lands on the first token. Their reports are a good template for how to present a sweep — request rate on the x-axis, separate curves for TTFT and TPOT percentiles, with the SLO drawn in. The practical takeaway is that TTFT is usually the binding SLO constraint under load, which is why our running example's knee is set by p99 TTFT.

**TGI (Text Generation Inference) benchmarks.** Hugging Face's TGI publishes benchmarks that emphasize the length-mix point directly: the same hardware serves wildly different request rates depending on the input/output length profile, and their guidance explicitly warns against single-length benchmarks. This is the published version of the argument in the first section of this post — the length distribution is not a detail, it is the dominant variable. See the [TGI deep dive](/blog/machine-learning/model-serving/text-generation-inference-deep-dive) for how its continuous batching and flash attention shift the knee.

**The general shape across systems.** Across vLLM, TGI, TensorRT-LLM, and Triton, the qualitative result is identical and it is the whole reason this methodology works: latency-vs-load is flat until a knee, then vertical; goodput tracks load until the knee, then collapses; and the knee's *position* moves with optimizations while its *shape* does not. Any benchmark that does not show you this curve — that reports a single throughput number without a latency sweep — has not told you where your knee is, and therefore has not told you your capacity.

## When to use this (and when not to)

The full open-loop-sweep, find-the-knee, size-with-headroom methodology is the right default for any LLM service you are going to put real traffic on. But it has a cost — engineering time, a load-test environment, a representative trace — and there are cases where a lighter touch is correct and cases where a heavier one is mandatory.

**Do the full methodology when:** you are sizing a fleet for a launch or a capacity commitment; you are choosing hardware or a serving framework and need cost-per-token to decide; your traffic is high enough that GPU cost is a material line item; or your SLO is contractual. Anytime the answer to "how many GPUs" has a dollar sign or a pager attached, do the sweep.

**A lighter touch is fine when:** you are prototyping and just need to know if a single GPU holds your dev traffic (a quick closed-loop smoke test answers that); the workload is genuinely fixed-length (some batch or embedding jobs are, and then a fixed-prompt test is actually representative); or the fleet is small enough that the cost of over-provisioning by one GPU is less than the cost of the load-testing engineering. Do not spend a week finding the knee to save half a GPU.

**Now the anti-patterns — the things people do that this methodology exists to prevent:**

- **Do not size off a closed-loop test.** It structurally cannot find the knee; it self-throttles at `λ = C/W` and reports a number that will not survive open-loop production traffic. If the benchmark set a concurrency instead of a rate, it did not measure your capacity.
- **Do not size off a fixed-prompt test.** The length distribution dominates LLM cost; a single length measures one point in a space your traffic spreads across, almost always the optimistic corner.
- **Do not extrapolate a knee across length mixes.** A knee measured at 512/256 tokens does not predict the knee at 4,000/1,000 tokens — they are different workloads on the same hardware. If your traffic mix shifts (a new feature sends longer prompts), re-measure.
- **Do not find the knee with the mean or median latency.** SLO violations live in the tail; find the knee with the percentile your SLO is written against, or you will overstate capacity and meet the real knee in production.
- **Do not report the marginal cost per token as the forecast.** The marginal knee cost is for hardware comparison; the effective provisioned cost — which includes headroom, failover, sub-knee utilization, and non-GPU infrastructure — is for the bill. Confusing the two burns trust.
- **Do not treat the plan as permanent.** Every optimization and every deploy moves the knee. Re-sweep in CI.

This methodology composes with the rest of the serving stack. The knee is the trigger for [autoscaling and GPU scheduling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling), the threshold for the admission control in [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management), and the input to the broader [SRE practice of capacity planning and forecasting](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting). Load testing is not a one-off gate before launch; it is the measurement that keeps the whole capacity apparatus honest.

## Key takeaways

- **The length distribution dominates LLM cost.** A fixed-prompt load test measures one point in a two-dimensional space and almost always the optimistic one. Replay a realistic input/output length mix — synthetic-with-distribution at minimum, your own production trace for sign-off.
- **Size from an open-loop test, never a closed-loop one.** A closed-loop test with fixed concurrency self-throttles at `λ = C/W` (Little's Law) and structurally cannot find the knee. Inject a fixed arrival rate.
- **The knee is the highest offered load where p99 still meets the SLO** — equivalently, where goodput still equals throughput. Past it, throughput can look stable while goodput collapses. Goodput is the metric that sizes a fleet.
- **Find the knee with the percentile your SLO is written against.** The tail is where violations live, and it is heavy for LLMs. Using the mean overstates capacity.
- **Fleet = ceil(peak / knee), then add headroom.** Peak is not mean (size for the busy-minute p99), you cannot run a replica at 100% of its knee (add a burst replica), and any replica can die (add N+1). For the 50 QPS example, the honest answer was 5 H100s, not the 2-3 a naive test suggested.
- **The knee scales with HBM bandwidth,** so cost per million tokens — not GPU sticker price — is the right hardware metric. The fastest card is often the cheapest per token.
- **Report marginal and effective cost separately.** Marginal knee cost is for hardware comparison; effective provisioned cost (headroom + failover + sub-knee utilization + non-GPU infra) is 2-4× higher and is the actual bill.
- **Re-sweep after every optimization and every deploy.** The knee is a measurement of your current stack, not a constant. Automate the sweep in CI against a canary.

## Further reading

- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023** — the vLLM paper; its latency-vs-throughput sweeps are the canonical example of the knee curve and how an optimization shifts it.
- **vLLM documentation, `benchmark_serving.py` / `vllm bench serve`** — the reference open-loop benchmark harness with ShareGPT replay and native TTFT/TPOT/goodput reporting.
- **NVIDIA GenAI-Perf documentation** — request-rate and concurrency sweeps for Triton/NIM endpoints, with synthetic length controls and standardized reports.
- **Little, J.D.C., "A Proof for the Queuing Formula L = λW," Operations Research, 1961** — the original proof of the identity behind why closed-loop tests self-throttle and why concurrency is an output, not an input.
- **k6 documentation, the `constant-arrival-rate` executor** — how to run a genuine open-loop test from a general-purpose load tool when the question spans the whole gateway.
- **Within this series:** [What is model serving](/blog/machine-learning/model-serving/what-is-model-serving) (the SLO triangle), [Model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics) (TTFT, TPOT, p99, goodput), [High-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management) (goodput, admission control, load shedding), and [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) (using the knee as the scale-up trigger).
- **Google SRE Book, and the [capacity planning and forecasting](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting) deep dive** — the broader discipline of turning demand forecasts and measured per-unit capacity into provisioned headroom.

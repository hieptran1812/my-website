---
title: "Estimating Concurrent Users and RPS for LLM Serving — Plus a Practical Self-Hosted vs API Comparison"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "capacity-planning",
    "serving",
    "deployment",
    "cost-optimization",
    "self-hosted",
    "api",
    "rps",
    "throughput",
    "infrastructure",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "How many GPUs do I need? How many concurrent users can one H100 handle? Is it cheaper to self-host a 70B model or pay OpenAI? These are the questions that stall every LLM project at the planning stage, and most answers online are hand-wavy. This article gives you the actual formulas, real throughput numbers, worked examples at 100 / 1,000 / 10,000 user scales, and a concrete break-even analysis for self-hosted vs third-party APIs."
---

## The Question This Article Answers

You have a product idea that uses an LLM. Before you can build it, someone — you, your CTO, your CFO — is going to ask variations of three questions:

1. **How many concurrent users can we handle per GPU?**
2. **What does that cost per request?**
3. **Should we self-host or just pay an API?**

The internet is full of vague answers to these. This article gives you concrete ones: the formulas, the real numbers, worked examples at three different scales, and a break-even analysis you can actually use in a planning doc.

If you haven't read the companion articles on [LLM inference optimization](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) and [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems), those are the technical foundation. This article is the **capacity-planning spreadsheet** built on top of them.

## Part 1: Why LLM Capacity Planning Is Different

### "Concurrent Users" Doesn't Mean What It Means for Web Services

For a REST API, a request takes 50ms. "100 concurrent users" means at any given millisecond, maybe 5 requests are actually in flight. The box mostly idles.

LLMs are the opposite. A single chat response takes **5–30 seconds** to fully stream. During those seconds, that user is actively consuming GPU bandwidth. "100 concurrent users" can mean *100 sequences in flight simultaneously*, each reading weights every step.

This changes every piece of the math:

| Property | Web service | LLM service |
| --- | --- | --- |
| Request duration | 10–200 ms | 2–60 s |
| Resource usage during request | near zero (IO wait) | continuous GPU compute and HBM bandwidth |
| Bottleneck | threads, DB | memory bandwidth, KV cache space |
| Scaling signal | CPU % | GPU utilization? No — see below |
| Natural unit | requests/sec | **tokens/sec** |

If you try to plan LLM capacity in "requests per second," you will be wrong. You have to plan in **tokens per second** and convert.

### The Right Mental Model: Tokens/Sec Is the Currency

Think of an LLM-serving GPU as a bucket that fills at a fixed rate — say, 3,000 output tokens per second. Every user takes some tokens out of the bucket to get their answer. Your capacity is:

```
Users per second = (tokens/sec capacity) / (tokens per user request)
```

Everything else — TTFT, latency, concurrent users, cost per user — is a consequence of this simple ratio.

## Part 2: The Three Numbers You Need About Your Workload

Before any math, you need three numbers about *your* workload. Don't use industry averages; measure these from your own traffic or, for a new product, from realistic prototypes.

### Number 1: Prompt Tokens Per Request

The user's input, plus any system prompt, few-shot examples, retrieved documents, and conversation history. This is **input** tokens.

Typical ranges:

| Workload | Prompt tokens (p50) | Prompt tokens (p95) |
| --- | --- | --- |
| Simple chat | 500 | 3,000 |
| Chat with long system prompt + history | 3,000 | 15,000 |
| RAG (retrieval-heavy) | 8,000 | 30,000 |
| Agent with many tool definitions | 10,000 | 40,000 |
| Document analysis / long-context | 30,000 | 150,000 |

If you don't know yours yet, assume a medium scenario (2,000–5,000) for planning.

### Number 2: Output Tokens Per Request

What the model generates. This is **output** tokens.

Typical ranges:

| Workload | Output tokens (p50) | Output tokens (p95) |
| --- | --- | --- |
| Classification / short answer | 50 | 200 |
| Chat replies | 150 | 600 |
| Long-form generation | 800 | 3,000 |
| Code generation | 400 | 2,000 |
| Reasoning / chain-of-thought | 1,500 | 8,000 |

### Number 3: Traffic Rate (RPS or RPM)

How often requests arrive. Not average over a day — **peak** over your busy windows. Real traffic is bursty, and you plan for peaks.

A useful rule: **peak-to-average ratio is 3–8×** for consumer products, 2–3× for B2B. If you know your daily volume, divide by 86,400 seconds and multiply by that ratio to get peak RPS.

If you don't know yet, plan for a spectrum:

- Small product: 1–10 RPS
- Growing product: 10–100 RPS
- Real scale: 100–1,000 RPS
- Large platform: 1,000+ RPS

## Part 3: From Tokens/Sec to Concurrent Users — The Formulas

![Capacity formula chain: peak QPS + avg in/out tokens → concurrent users ≈ QPS · response time → throughput need → GPU count; then check KV cache fits, else scale out or quantize](/imgs/blogs/capacity-01-formula.png)

### Formula 1: Throughput → RPS

If your GPU produces T tokens/sec of output, and each request averages O output tokens:

```
RPS = T / O
```

Example: a well-optimized H100 serving a 70B model in FP8 delivers roughly **3,000 output tokens/sec** in aggregate. If your average request outputs 300 tokens:

```
RPS = 3000 / 300 = 10 RPS per GPU
```

This is the number that actually matters. Forget "GPU utilization." You have 10 usable RPS on this box.

### Formula 2: RPS → Concurrent Users (Little's Law)

Little's Law, the most useful equation in capacity planning:

```
N = λ × W
```

where:

- `N` = average number of concurrent users/requests in the system
- `λ` = arrival rate (requests per second)
- `W` = average time in system (seconds per request)

For an LLM with average output O tokens at rate T tokens/sec shared across the batch:

```
W ≈ TTFT + O × TPOT
```

For an H100 serving 70B FP8:

- TTFT ≈ 0.3 s
- TPOT ≈ 40 ms per output token (under batch load)

So for a 300-token response: `W ≈ 0.3 + 300 × 0.04 = 12.3 s`.

At 10 RPS: `N = 10 × 12.3 = 123 concurrent streaming sessions`.

That's the real answer to "how many concurrent users per H100?" for this specific workload: **about 120**.

### Formula 3: RPS → User Count

"Concurrent sessions" isn't the same as "users who use the product." Most users aren't mid-stream at any given moment — they're reading the previous answer, typing the next question, or doing something else entirely.

```
Total users = RPS × (session length) / (requests per user per session)
```

A common consumer pattern: user sends one request every 30–60 seconds while actively using the product. So:

```
Active users ≈ RPS × 45
```

10 RPS → roughly **450 active users** at any moment; far more total DAU depending on session patterns.

For B2C chat products, the ratio of MAU to peak RPS is typically **500–2000×**. For B2B document tools, often 5,000–20,000× (users are sporadic).

## Part 4: Real Throughput Numbers on Real Hardware

These are rough 2026 numbers for well-optimized inference (vLLM/SGLang/TRT-LLM class, FP8/INT8 quantization where reasonable, continuous batching on). Your mileage will vary by ±30%.

### Aggregate Output Tokens/Sec (Single GPU, Batched)

| Model class | A100 80GB | H100 80GB | H200 141GB | B200 |
| --- | --- | --- | --- | --- |
| 7–8B, FP16/FP8 | 5,000 | 10,000 | 13,000 | 20,000+ |
| 13B, FP8 | 3,000 | 6,000 | 8,000 | 13,000+ |
| 34–40B, FP8 | 1,200 (tight fit) | 4,500 | 6,000 | 10,000+ |
| 70B, FP8 | requires 2 GPUs (TP) | 3,000 | 4,500 | 7,500+ |
| 70B, INT4 | ~1,500 | 3,500 | 5,500 | 9,000+ |
| 400B+ class | multi-node only | multi-GPU (TP=8) | multi-GPU (TP=4) | multi-GPU (TP=2–4) |

### Per-User TPOT Under Load

Aggregate throughput is shared across the batch. What a single user sees is the **per-user TPOT** — how fast their own tokens stream in. This varies with batch size:

| Batch size | Aggregate tokens/s (70B on H100) | Per-user tokens/s |
| --- | --- | --- |
| 1 | 60 | 60 |
| 4 | 200 | 50 |
| 16 | 700 | 43 |
| 32 | 1,200 | 37 |
| 64 | 2,000 | 31 |
| 128 | 3,000 | 23 |

Bigger batches = more throughput *total*, but slower per-user stream. This tradeoff is the single most important knob in LLM serving. If your product needs "feels fast" streaming (>30 tokens/s per user feels fast to humans), you cap batch size. If you just need throughput and don't care about smoothness, you push it higher.

### Typical TTFT

TTFT depends mostly on prompt length:

| Prompt tokens | TTFT on H100 (70B FP8) | TTFT on H100 (8B FP8) |
| --- | --- | --- |
| 500 | 0.15 s | 0.05 s |
| 2,000 | 0.4 s | 0.12 s |
| 10,000 | 1.8 s | 0.5 s |
| 30,000 | 5.5 s | 1.7 s |

Prefix caching on shared prompts can reduce TTFT by 5–10× on the cached portion; [see the serving article](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) for details.

## Part 5: Three Worked Examples

### Example A: 100 Concurrent Users, Chat Product, Medium Model

**Workload:**

- 13B model, FP8
- Prompt 2,000 tokens (p50), output 400 tokens (p50)
- User sends one message every 30 seconds while active

**Math:**

- Per request: `W ≈ 0.4 + 400 × 0.03 = 12.4 s` of streaming
- Concurrent streaming sessions at any moment = (100 active users × message every 30s × 12.4s streaming) / 30s ≈ **40 sessions**

Actually easier: we want ~100 *active* users → RPS ≈ 100/30 = 3.33 RPS.

Capacity of one H100 for 13B model: ~6,000 tokens/s ÷ 400 tokens/request = **15 RPS**.

**One H100 is overkill**. You'd run one H100, or two for redundancy (no single-GPU failure tolerance).

Cost at $3/hr per H100 × 2 = $6/hr ≈ **$4,400/month**.

### Example B: 1,000 Concurrent Active Users, Same Workload

- RPS ≈ 1000 / 30 = 33 RPS
- Per-GPU capacity = 15 RPS
- **Need 3 GPUs of headroom → 4–5 H100s** (for redundancy, failover, load imbalance).

Cost: 5 × $3/hr = $15/hr ≈ **$11,000/month**.

### Example C: 10,000 Concurrent Active Users, Agent Workload with 70B Model

**Workload:**

- 70B model, FP8, served on H100s
- Long system prompts with tools: **10,000 prompt tokens** (p50), but cacheable (80% cache hit on system prompt)
- Output 800 tokens (p50, agents generate more)
- Average user: one request every 60 seconds during active sessions

**Math:**

- RPS = 10,000 / 60 = **167 RPS**
- Effective prompt tokens to process (after prefix cache): 10,000 × (1 - 0.8) + small overhead ≈ **2,500 effective**
- Per-GPU output throughput for 70B on H100: ~3,000 tokens/s
- Per-GPU RPS capacity: 3,000 / 800 = **3.75 RPS**

**Need 45 H100s of compute** for the decode. For resilience, headroom for prefill spikes, and regional distribution: budget ~55–60.

Cost: 55 × $3/hr ≈ **$120,000/month** in raw GPU spend. Add overhead (observability, routing, storage, the engineers) and call it **$180,000–$250,000/month** total.

## Part 6: The Self-Hosted Cost Model in Detail

Take the raw GPU hourly rate and calculate cost per million tokens. This is the number that goes next to API prices for comparison.

```
$/M tokens = (GPU $/hr) / (throughput tokens/s) × 1,000,000 / 3600
```

Worked examples, assuming $3/hr H100:

| Setup | Throughput tok/s | $/M output tokens |
| --- | --- | --- |
| 8B FP8, 1 H100 | 10,000 | $0.083 |
| 13B FP8, 1 H100 | 6,000 | $0.14 |
| 70B FP8, 1 H100 | 3,000 | $0.28 |
| 70B FP8, 2 H100s (TP) | 5,500 | $0.30 |
| 405B FP8, 8 H100s (TP) | 2,500 | $2.67 |

These numbers assume **100% GPU utilization**. Real deployments run at 50–75% average utilization, so multiply by 1.5–2× for a realistic figure.

### Operational Cost Multipliers

The GPU is the biggest cost, but not the only one. Add to it:

| Cost | Typical multiplier on GPU spend |
| --- | --- |
| Storage (model weights, KV offload, logs) | 2–5% |
| Network (egress, inter-GPU) | 3–10% |
| Observability / monitoring stack | 2–5% |
| Engineering (see below) | 30–200%+ |

Engineering cost is the one everyone underestimates. A serious LLM deployment needs 1–3 engineers on it part-time, at least during the first year. That's real money — often more than the GPU bill for small-scale deployments.

### The Realistic Self-Hosted $/M Tokens

Rough-but-defensible multiplied-out numbers:

| Model | Ideal $/M | Realistic $/M (50% util + ops) |
| --- | --- | --- |
| 8B | $0.08 | $0.20 |
| 13B | $0.14 | $0.35 |
| 70B | $0.28 | $0.70 |
| 405B | $2.67 | $6.00 |

## Part 7: API Pricing (April 2026 Ballpark)

Exact prices shift constantly. These are orders of magnitude you can plan around; look up current numbers before committing.

| Model class | Input $/M | Output $/M |
| --- | --- | --- |
| Flagship (GPT-5-class, Opus 4.x) | $5–$15 | $25–$75 |
| Mid-tier (Sonnet-class, GPT-5-mini) | $2–$4 | $10–$18 |
| Small / cheap (Haiku-class, nano) | $0.20–$1 | $1–$5 |
| Open-source hosted by inference providers (Together, Fireworks, DeepInfra) | $0.10–$0.80 | $0.20–$2 |

### The Hidden Structure of API Pricing

Three numbers compose any API bill:

1. **Input tokens × input price.**
2. **Output tokens × output price.**
3. **Any premium features** (vision, long context, tool use, caching discounts).

Most real workloads are 5–20× more input than output tokens. A chat with 5,000 tokens of system prompt and 200 tokens of output, over thousands of turns, is dominated by *input* cost. This means **input price matters more than output price for most agent workloads**, and cached-input pricing (where providers charge 10–25% of normal rates for cache hits) changes the math dramatically.

### Effective API Cost for Common Workloads

Assuming a typical balance of input and output:

| Workload | Tokens per request (in/out) | Mid-tier API cost per request |
| --- | --- | --- |
| Short chat | 500 / 150 | ~$0.003 |
| Chat with long prompt | 5,000 / 300 | ~$0.018 |
| RAG query | 15,000 / 400 | ~$0.05 |
| Long doc analysis | 80,000 / 800 | ~$0.25 |
| Agent with many tool calls | 10,000 / 1,500 (per step × 5 steps) | ~$0.15 per task |

## Part 8: Self-Hosted vs API — The Break-Even Analysis

![Self-host vs API decision: spiky/low volume → API; high and stable → check custom model / data isolation needs; otherwise compare M tokens/day against break-even](/imgs/blogs/capacity-02-selfhost-vs-api.png)

The classic question: **at what volume does self-hosting become cheaper than API?**

### The Simple Break-Even

```
break-even volume = (fixed self-host cost per month) / (API cost per request − self-host cost per request)
```

Example: deploying a 70B model on 2× H100s for $4,400/month all-in. API cost for comparable quality is $0.018/request (Sonnet-class), self-hosted cost is $0.005/request.

```
break-even = $4,400 / ($0.018 − $0.005) = ~340,000 requests/month
            ≈ 11,000/day ≈ 0.13 RPS sustained
```

For a small product, that might take months to reach. For a consumer app, hours.

### The Full Comparison Matrix

| Dimension | Self-hosted | Third-party API |
| --- | --- | --- |
| **Fixed cost** | High (GPUs 24/7) | Zero |
| **Variable cost** | Low (per token) | Higher (per token) |
| **Cost at 0 traffic** | Full GPU bill | Zero |
| **Cost at large scale** | Lower | Higher |
| **Time to first deployment** | Weeks–months | Hours |
| **Latency** | Tunable, potentially lower, geographic control | Fixed, subject to provider's region |
| **Rate limits** | None (your metal) | Yes — real constraint for bursty traffic |
| **Model choice** | Any open-source model, any fine-tune | Only what provider offers |
| **Capability ceiling** | Open-source frontier (smaller gap, but real) | Vendor's frontier models |
| **Compliance / data residency** | Full control | Provider's certifications |
| **Quality stability** | You control upgrades | Vendor may deprecate or change |
| **Engineering burden** | High, ongoing | Low |
| **On-call burden** | You carry it | Vendor carries it |
| **Innovation speed** | Bounded by open-source release cycle | Vendor's latest, immediately |

### The Hybrid That Most Teams Actually Ship

In practice, most serious deployments end up with a mix:

```
                   User request
                        │
                        ▼
                 ┌─────────────┐
                 │  Router     │  ── picks model by task difficulty
                 └──────┬──────┘
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
         Self-host    Mid API    Frontier API
         (8B-70B)   (Haiku-    (Opus, GPT-5)
                    Sonnet)      │
                                 └── for the hardest 5% of queries
```

- **Cheap self-hosted model** for high-volume, easy queries.
- **Mid-tier API** for quality-sensitive middle.
- **Frontier API** sparingly, for the small fraction of requests that need it.

This routing strategy can cut total cost **5–10×** versus calling the frontier model for everything, while preserving quality on hard queries.

## Part 9: When To Self-Host (Honestly)

Self-host when **at least two** of these are true:

- **Volume is large and stable.** Millions of requests a day, predictable growth.
- **Workload is specialized.** A fine-tuned 8–13B model beats a big API model on your task.
- **Data can't leave your environment.** Compliance, contracts, competitive sensitivity.
- **Latency matters intrinsically.** Voice, real-time applications where 200ms API network overhead is unacceptable.
- **You already have GPU infrastructure and ML-ops people.**
- **You're a platform** whose product *is* LLM inference for others.

## Part 10: When To Use an API (Honestly)

Use an API when **any** of these are true:

- **You're early.** Pre-product-market-fit — spend your engineering time on the product, not on GPU ops.
- **Traffic is bursty or unpredictable.** APIs scale for you; self-hosting at 5× peak-to-average is brutal.
- **You need the best model available.** The frontier models usually lead self-hostable ones for at least 6–18 months on hard benchmarks, and your product lives or dies on that margin.
- **The token volume is small.** Up to tens of millions of monthly tokens, API cost is *less than one engineer's salary*. Self-hosting to save that is a false economy.
- **Your team doesn't have GPU expertise.** And you can't hire it quickly.

## Part 11: A Capacity Planning Worksheet

Fill these in for your product. Once you have the numbers, the rest of the math is mechanical.

```
INPUTS — measure from traffic or realistic prototypes
───────────────────────────────────────────────────
Model size / class:              _____ (e.g., "70B FP8")
Avg input tokens (p50):          _____
Avg input tokens (p95):          _____
Avg output tokens (p50):         _____
Avg output tokens (p95):         _____
Peak RPS target:                 _____
% input that is cacheable:       _____ (shared system prompt?)

HARDWARE BASELINE
─────────────────
Per-GPU aggregate throughput:    _____ tokens/s  (from Table in Part 4)
Per-GPU RPS:                     output tokens/s ÷ avg output tokens
Per-GPU cost:                    $_____/hr

CAPACITY
────────
Required GPUs (raw):             peak RPS ÷ per-GPU RPS
Required GPUs (+ 40% headroom):  × 1.4
Monthly GPU cost:                GPUs × $/hr × 730

COST / REQUEST
──────────────
Self-hosted $/request:           monthly cost ÷ monthly requests
API $/request (looked up):       input_tokens × input_price + output_tokens × output_price

COMPARISON
──────────
Break-even volume:               (self-host fixed) ÷ (API $/req − self-host $/req)
Currently above break-even?      yes / no
By how much?                     × _____
```

If your traffic is below break-even by a significant factor, API is the honest answer. If it's far above, self-host is worth the engineering. If it's close, **use an API until it's obviously worth the effort to move**.

## Part 12: Common Planning Mistakes

Real planning failures I've seen, repeatedly:

1. **Planning for average traffic, not peak.** Peak-to-average ratio of 5× means average-sized capacity melts during spikes.
2. **Ignoring input tokens.** A workload with short outputs and long inputs looks cheap if you only count output tokens. It's not.
3. **Assuming linear scaling.** Two GPUs don't give you exactly 2× throughput — overhead, imbalance, and cold paths eat 10–20%.
4. **Forgetting streaming requests stay in memory.** A 30-second response occupies KV cache the whole time. KV cache capacity, not compute, often hits first.
5. **Confusing aggregate throughput with per-user smoothness.** A batch of 128 might process 3000 tok/s total, but each individual user sees 23 tok/s — readable, but not snappy.
6. **Under-budgeting engineering cost for self-hosting.** The GPU bill is the part you see; the on-call pager is the part you don't.
7. **Over-building early.** If you don't have traffic yet, a $20k/month GPU fleet is not an investment — it's a drain. Start on an API.
8. **Not measuring cached vs uncached performance.** A workload with 80% prefix-cache hit rate has 5× the effective capacity of one without. Without measuring, you'll plan for the wrong number.

## Part 13: The Framework in One Diagram

```
                 How much traffic do you have?
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
      <1M tok/mo      1M–1B tok/mo        >1B tok/mo
        │                 │                  │
     Use API          Run the break-even   Consider self-host
     entirely.        calc. API is often      with API fallback
     Stop reading.    still the answer.       for spikes / frontier.

                      Do you need data residency?
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                   Yes                      No
                    │                       │
         Self-host from day 1         Follow the flow above.
         regardless of volume.

                      Is latency under 500ms critical?
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                   Yes                      No
                    │                       │
         Self-host + edge deployment   API is fine.
         with specialized small model.
```

## Closing

The honest one-sentence answer to "how many users per GPU?" is: **one well-optimized H100 serving a 70B model gives you about 3,000 output tokens per second, which translates to roughly 10 RPS and 100–500 active users, depending on your session pattern.** Scale that number up or down by model size and hardware.

The honest one-sentence answer to "self-host or API?" is: **if your monthly token volume × the API price premium is smaller than the fully-loaded cost of one engineer maintaining a GPU fleet, use the API.** For most companies, this is true for a long time.

The math in this article gets you from vague anxiety to a defensible spreadsheet. That's usually enough to unblock the decision.

---

**Related reading**

- [Optimizing LLM Inference: A Complete Guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — the techniques behind the throughput numbers above.
- [Serving LLMs at Scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) — the production systems that make those numbers survive real traffic.
- [KV Cache](/blog/machine-learning/large-language-model/kv-cache) and [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) — the two techniques that move the $/M-token number the most.

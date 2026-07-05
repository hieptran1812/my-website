---
title: "Cost Optimization at LLM Scale: Driving Down Cost per Million Tokens Without Breaking SLOs"
date: "2026-07-05"
publishDate: "2026-07-05"
description: "The economics of running an LLM service — derive the cost-per-million-tokens model, rank the levers that move it, and drive it down 7x with quantization, autoscaling, spot fleets, and cost-aware routing without ever violating your latency SLO."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "cost-optimization",
    "finops",
    "spot-instances",
    "autoscaling",
    "llm-serving",
    "gpu-economics",
    "unit-economics",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/cost-optimization-at-llm-scale-1.webp"
---

The message that ends the honeymoon is never from engineering. It is from finance, and it is one line: "The inference bill for last month was \$412,000, up from \$180,000 in March, and traffic only grew 40%. What happened?" You open the cloud console and the answer is staring back at you — a fleet of forty H100s, most of them coasting at 30% utilization, billed at the full on-demand rate around the clock including the eight overnight hours when your traffic is a trickle. Nobody did anything wrong. Every GPU was provisioned to hold the p99 latency SLO at peak, the model runs in fp16 because that is how it was trained, and the autoscaler was never wired up because the launch deadline came first. The service works. It is also burning money at a rate that will not survive the next budget review.

This is the moment that separates a model that ships from a model that *stays* shipped. A serving system has three currencies — latency, throughput, and cost — and for the first few months of a service's life you spend almost all of your attention on the first one. Cost is the currency you can ignore right up until you cannot, and then it becomes the only thing anyone wants to talk about. The good news is that cost is not a mystery. It is an equation with exactly three inputs, and every optimization in this post is a way of pushing on one of those three inputs. By the end you will be able to write down your own cost per million tokens from first principles, rank the levers that move it by dollars-saved-per-unit-risk, and drive the number down by roughly seven times on the same hardware — without ever letting a cost cut turn into an SLO violation.

The whole model fits in one picture. Cost per million tokens is a single price divided by two multipliers, throughput and utilization, and because both multipliers sit in the denominator, the levers *stack*: doubling throughput and doubling utilization is a fourfold cut before you have renegotiated a single dollar of GPU price. The figure below is that stack, populated with the optimized end-state we will build toward — a blended GPU price of \$2.36 an hour, 5,500 tokens per second of throughput, 70% utilization, and the \$0.17 per million tokens that falls out of them.

![Layered stack figure showing GPU price of 2.36 dollars per hour divided by throughput of 5500 tokens per second times 70 percent utilization to give effective supply of 13.9 million tokens per hour and a final cost of 0.17 dollars per million tokens](/imgs/blogs/cost-optimization-at-llm-scale-1.webp)

If the serving primitives here are new to you — continuous batching, TTFT (time to first token), the KV cache, GPU utilization as distinct from GPU allocation — the series intro [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) builds the foundation this post stands on. Here we put a dollar sign on every one of those primitives and ask a single question of each: does it move cost per million tokens, by how much, and what does it cost you in latency or operational risk to pull it?

## 1. The unit-economics model: deriving cost per million tokens

Start with the only three numbers that matter and build the equation up so you never have to trust a vendor's marketing figure again.

A GPU costs some price $P$ dollars per hour. That is the numerator, and it is the number every FinOps dashboard shows you. It is also the number that matters least, because it says nothing about how much *work* you got for the money. To turn dollars-per-hour into dollars-per-token you need to know how many tokens that hour bought.

If the engine sustains a throughput of $T$ tokens per second and the GPU runs flat out for a full hour, it produces $T \times 3600$ tokens. But no production GPU runs flat out for a full hour. Some fraction of every hour is spent idle between requests, waiting on a half-full batch, or scaled down entirely overnight. Call that fraction of the hour spent doing useful token generation the utilization $U$, a number between 0 and 1. The *effective* supply of tokens the hour actually delivered is:

$$S = T \times 3600 \times U \quad \text{[tokens per GPU-hour]}$$

The cost to produce one token is then simply the price divided by the supply, and scaling to a million tokens — the unit every LLM API quotes — gives the equation this entire post orbits:

$$C_{1M} = \frac{P \times 10^{6}}{T \times 3600 \times U}$$

Three inputs, one output. Read it slowly, because its shape dictates strategy. Price $P$ is *linear* in the numerator: halve the price, halve the cost, no more and no less. Throughput $T$ and utilization $U$ are both in the *denominator*, which means two things. First, they are equally powerful — a doubling of either one halves cost. Second, and this is the part teams miss, they *multiply*: because the total reduction is the product of the individual reductions, three modest 2x wins are not a 6x improvement stacked additively, they are close to an 8x improvement. The equation is a machine for turning several unglamorous, individually-boring optimizations into one dramatic bottom-line number.

#### Worked example: the baseline and the target

Take a real 8-billion-parameter chat model on a single H100 80GB, and walk the two ends of the journey.

The *baseline* is the service finance just complained about. The GPU is billed at on-demand list price, call it \$4.00 an hour. The model runs in fp16 with naive batching, so aggregate output throughput sits at a modest 2,500 tokens per second. Traffic is spiky and the fleet is over-provisioned for peak, so utilization averages a dismal 35%. Plug it in:

$$C_{1M}^{\text{base}} = \frac{4.00 \times 10^{6}}{2500 \times 3600 \times 0.35} = \frac{4{,}000{,}000}{3{,}150{,}000} \approx \$1.27 \text{ per 1M tokens}$$

The *target* is the same model, same hardware, after this post. FP8 quantization plus continuous batching lifts throughput to 5,500 tokens per second (a 2.2x gain). Autoscaling with scale-to-zero overnight lifts utilization to 70% (a 2.0x gain). A capacity mix of committed base plus spot brings the blended price down to \$2.36 an hour (a 1.7x gain). Plug those in:

$$C_{1M}^{\text{opt}} = \frac{2.36 \times 10^{6}}{5500 \times 3600 \times 0.70} = \frac{2{,}360{,}000}{13{,}860{,}000} \approx \$0.17 \text{ per 1M tokens}$$

The bill dropped from \$1.27 to \$0.17 per million tokens — a **7.5x reduction** on the same model and the same GPU. And notice how it decomposes: $2.2 \times 2.0 \times 1.7 = 7.5$. Each lever was individually unremarkable. Multiplied together they are the difference between a service that loses money on every request and one that prints it. This multiplicativity is the single most important idea in LLM cost engineering, and it is why the rest of this post is organized as a ranked list of levers rather than a single silver bullet.

Here is that equation as a calculator you can drop into a notebook and point at your own fleet. It is the tool I reach for before any capacity conversation, because it forces every proposal — "let's buy reserved instances," "let's turn on FP8" — to declare which of the three inputs it moves and by how much.

```python
from dataclasses import dataclass

@dataclass
class ServingScenario:
    name: str
    gpu_price_per_hour: float   # blended $/hr across the fleet
    tokens_per_sec: float       # sustained aggregate output tok/s per GPU
    utilization: float          # fraction of wall-clock doing useful work (0..1)
    num_gpus: int = 1

    def cost_per_1m_tokens(self) -> float:
        effective_supply = self.tokens_per_sec * 3600 * self.utilization
        return (self.gpu_price_per_hour * 1_000_000) / effective_supply

    def monthly_cost(self, hours: float = 730) -> float:
        return self.gpu_price_per_hour * self.num_gpus * hours

    def monthly_token_capacity(self, hours: float = 730) -> float:
        return self.tokens_per_sec * 3600 * self.utilization * self.num_gpus * hours

baseline = ServingScenario("fp16, naive batch, on-demand",
                           gpu_price_per_hour=4.00, tokens_per_sec=2500, utilization=0.35)
optimized = ServingScenario("fp8, continuous batch, spot+committed, autoscaled",
                            gpu_price_per_hour=2.36, tokens_per_sec=5500, utilization=0.70)

for s in (baseline, optimized):
    print(f"{s.name:52s}  ${s.cost_per_1m_tokens():.3f} / 1M tokens")

reduction = baseline.cost_per_1m_tokens() / optimized.cost_per_1m_tokens()
print(f"\nTotal reduction: {reduction:.1f}x")
# fp16, naive batch, on-demand                          $1.270 / 1M tokens
# fp8, continuous batch, spot+committed, autoscaled     $0.170 / 1M tokens
#
# Total reduction: 7.5x
```

Keep this calculator open as we go. Every lever below is, in the end, a claim about how it changes one of `gpu_price_per_hour`, `tokens_per_sec`, or `utilization`, and the honest way to evaluate any of them is to plug the new number in and read the new cost.

## 2. Where the money actually goes

Before ranking the levers, look at where the dollars leak, because the biggest line item on your bill is almost never "useful token generation." It is idle time, and idle time is invisible on a cost dashboard — a GPU billed at 30% utilization and a GPU billed at 95% utilization cost exactly the same per hour. The waste is a *shadow*: you pay for capacity you provisioned and did not use.

The figure below makes that shadow visible. It is a utilization heatmap of a three-GPU fleet across a day. Every green cell is a GPU doing real work near peak. Every amber cell is a GPU sitting at 5–20% utilization in the small hours — still powered, still billed, producing almost nothing. The white cells are the win we are heading toward: GPU 2 scaled to zero overnight, billed for nothing at all.

![Grid heatmap of three GPUs across five times of day showing green high-utilization cells at midday and evening peaks, amber low-utilization idle cells at 02h and 23h that are still billed, and two white scaled-to-zero cells on GPU 2](/imgs/blogs/cost-optimization-at-llm-scale-4.webp)

Four buckets account for essentially all of the waste, and it is worth naming each one because each has a different lever.

**Idle GPUs.** A fleet sized for peak sits mostly idle off-peak. If your peak-to-trough traffic ratio is 3:1 — utterly normal for a consumer-facing service in a single time zone — and you provision statically for peak, then averaged over the day your fleet is roughly one-third utilized. Two-thirds of the bill is buying air. This is the single largest and most recoverable waste in most LLM services, and it is pure utilization: it does not touch throughput or price at all.

**Low batch utilization.** Even while a GPU is "busy," it may be doing far less work than it could. LLM decode is memory-bandwidth-bound: each step reads the entire weight matrix out of HBM regardless of how many sequences are in the batch. A batch of 1 and a batch of 32 read the same weights and take almost the same wall-clock time per step, so a GPU serving one lonely request at a time is wasting 97% of its throughput potential. This is why continuous batching is the highest-leverage throughput lever — it fills the batch.

**Over-provisioned headroom.** Because latency degrades non-linearly as a queue fills, teams hold a safety margin — often 30–50% spare capacity — so a traffic spike does not blow the p99. That headroom is real insurance and you should not cut it to zero, but static headroom is expensive headroom. Autoscaling converts a fixed 40% margin into a dynamic one that only exists when traffic is actually near the ceiling.

**Prefill versus decode asymmetry.** A single request has two cost profiles glued together. *Prefill* — the one big forward pass over the whole prompt — is compute-bound: it saturates the tensor cores, runs in tens of milliseconds, and is cheap *per token* because it processes the entire prompt in parallel. *Decode* — generating output one token at a time — is memory-bandwidth-bound and cheap in FLOPs but expensive in wall-clock, because every single output token requires another full read of the model weights. The consequence is stark: on the same GPU, an output token costs roughly three to five times as much to produce as an input token. This is not an accident of your setup; it is physics, and it is exactly why every major API prices output tokens several times higher than input tokens (GPT-4o, for instance, charges four times more for output than input). If your workload is output-heavy — long generations, agents that write code — your cost is dominated by decode, and decode-side levers (quantization, bigger decode batches, speculative decoding) matter most. If it is input-heavy — long documents, short answers, RAG — prefill and prompt caching dominate.

The asymmetry has a direct blended-cost consequence worth computing, because it changes which lever you reach for. Suppose a request carries 2,000 input tokens and produces 200 output tokens — a typical RAG shape. If an output token costs 4 times an input token, the request's cost is proportional to $2000 \times 1 + 200 \times 4 = 2800$ cost-units, of which the 2,000 input tokens contribute 2,000 units (71%) and the 200 output tokens contribute 800 units (29%). For this input-heavy shape, the dominant cost is prefill over the prompt, so prefix caching that eliminates repeated prefill is the highest-leverage lever, not decode-side quantization. Flip the shape to a code-generation request — 500 input, 2,000 output — and the arithmetic inverts: $500 \times 1 + 2000 \times 4 = 8500$ units, of which output is 8,000 (94%). Now decode dominates utterly and quantization plus speculative decoding are where the money is. The lesson is to profile your *token mix* before you pick a lever, because the same optimization that saves 90% on one workload saves 5% on the other.

A fifth, smaller bucket deserves a mention because it surprises people: **network egress**. Serving tokens is cheap to move — a megabyte of text is a lot of tokens — but if you are shipping images, audio, or embeddings across regions or out to the internet, egress fees can quietly become 5–15% of a multi-modal bill. It is rarely the headline, but it is worth a line on the dashboard so it does not grow unwatched.

The through-line: **utilization is the largest recoverable waste, and it is recoverable precisely because it costs the same whether you use it or not.** Every idle GPU-hour is a dollar you already spent on nothing. That is why, when we rank the levers, the utilization levers punch above their weight — they convert money you are *already spending* into tokens, rather than negotiating for new discounts.

## 3. The levers, ranked

There are exactly four families of lever, because there are exactly three inputs to the cost equation plus one way to cheat the equation entirely by not running it. Ranked by how I actually reach for them in practice:

1. **Raise throughput per GPU** ($T$ up): batching, quantization, chunked prefill, speculative decoding. Highest leverage, lowest risk, do it first.
2. **Raise utilization** ($U$ up): autoscaling, MIG and bin-packing, scale-to-zero. Recovers money you already spend; the risk is cold-start latency.
3. **Buy cheaper capacity** ($P$ down): spot plus on-demand blends, reserved and committed-use discounts, right-sizing the GPU type. The biggest dollar swings, but spot carries interruption risk and commitments carry lock-in.
4. **Do not compute at all** (sidestep the equation): prefix caching, semantic caching, request deduplication, and routing small requests to small models. Infinite ROI when it applies, because a served-from-cache request costs nothing.

The matrix below is the same ranking with the trade-offs attached: how much each lever cuts, what it risks against your SLO, how much effort it takes, and when it applies. It is the decision surface for the rest of the post.

![Matrix of seven cost levers rated across cost cut, SLO risk, effort, and when-to-use columns, showing quantization and batching as low-risk throughput wins, spot capacity as the biggest dollar cut but the only one with interruption danger, and caching as the highest-variance compute saver](/imgs/blogs/cost-optimization-at-llm-scale-2.webp)

Written out as a table you can paste into a planning doc, with representative numbers for our 8B running example:

| Lever | Moves | Typical cut | SLO risk | Effort | Reach for it when |
|---|---|---|---|---|---|
| FP8 / INT4 quantization | $T$ up | 2–3x throughput | Low (FP8), medium (INT4) | Medium | Almost always; FP8 on H100 is nearly free accuracy |
| Continuous batching | $T$ up | 2–4x throughput | Raises TTFT tail | Low (built into vLLM) | Any service above ~10 QPS |
| Chunked prefill | $T$ up, TTFT stable | 10–30% throughput | Low | Low (a flag) | Long prompts colliding with decode |
| Autoscaling + scale-to-zero | $U$ up | Cut idle 20–50% | Cold-start TTFT | Medium | Spiky or diurnal traffic |
| MIG / bin-packing | $U$ up | 2–7x density | Noisy-neighbor jitter | Medium | Many small models on big GPUs |
| Spot + on-demand blend | $P$ down | 40–70% of \$/hr | Interruptions | High (ops) | Fault-tolerant or batch workloads |
| Reserved / committed-use | $P$ down | 30–60% of \$/hr | Lock-in | Low | Stable, predictable base load |
| Prefix + semantic caching | Skip compute | 10–90% of compute | Staleness | Medium | Repeated system prompts or FAQs |

Two rules of thumb govern the order in which you pull these. First, **do the throughput levers before the capacity levers**, because a 2x throughput win makes every subsequent GPU-hour twice as productive, so you buy *fewer* of whatever you buy — the discount compounds on top of a smaller number. Second, **never let a cost lever push you past your SLO**; a cut that saves 30% of your bill and violates your p99 latency contract has not saved you anything, it has moved the cost from your cloud invoice to your churn rate. We return to that guardrail explicitly at the end.

## 4. Lever one: raise throughput per GPU

Throughput is the first lever because it is nearly free of downside and it multiplies everything downstream. Three techniques do most of the work, and each has a dedicated post in this series, so here I focus on the *dollar* each one moves rather than re-deriving the mechanism.

**Quantization** shrinks the model's numeric representation, and because decode is memory-bandwidth-bound, shrinking the weights you must read every step is close to a direct throughput multiplier. Moving an 8B model from fp16 to FP8 roughly halves the bytes read per decode step, and on an H100 with native FP8 tensor cores the accuracy cost is often under 0.1 perplexity points — an almost-free doubling of $T$. INT4 (via AWQ or GPTQ) shrinks weights fourfold and can push throughput 2.5–3x, at a more real but usually acceptable accuracy cost. In the cost equation, a 2.2x throughput gain from FP8 turns our baseline \$1.27 into \$0.58 per million tokens on its own — before touching anything else. The mechanics of which scheme to pick, and the accuracy trade-offs, are the subject of [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving); for costing purposes, treat FP8 as a default-on 2x throughput lever and INT4 as a 2.5–3x lever you validate against your eval set.

**Continuous batching** is the highest-leverage software change in LLM serving and it costs you a config flag. A naive server processes requests one batch at a time and waits for the slowest sequence in the batch to finish before starting the next — leaving the GPU idle whenever sequences finish at different lengths, which is always. Continuous batching (vLLM's scheduler, TGI's, and Triton's all implement a version) injects new requests into the batch at every decode step, keeping the batch full and the GPU saturated. On real traffic this is a 2–4x aggregate throughput gain over static batching. The cost is a modestly higher TTFT tail, because a newly-arrived request may wait a step or two for admission — a trade we manage in [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management).

**Chunked prefill** addresses the prefill-decode collision. Without it, a long prompt's prefill monopolizes the GPU for a big compute burst, stalling every in-flight decode and spiking their inter-token latency. Chunked prefill slices the prefill into fixed-size chunks and interleaves them with ongoing decode, smoothing utilization and lifting sustained throughput 10–30% while *protecting* decode latency. It is a flag in vLLM (`--enable-chunked-prefill`), and it is one of the rare levers that improves throughput and latency at the same time.

**Speculative decoding** is the fourth throughput lever and the one whose economics are most often misjudged, so it earns a worked example. A small, cheap draft model proposes several tokens at once; the large target model verifies all of them in a single forward pass, accepting the longest correct prefix. Because decode is memory-bandwidth-bound, that one target forward pass costs almost the same whether it emits one token or five, so accepting multiple tokens per pass is close to free throughput.

#### Worked example: speculative decoding's cost math

Let the draft model propose $k$ tokens per step and let $\alpha$ be the probability the target accepts each proposed token. The expected number of tokens the target emits per verification pass follows the standard geometric result $(1 - \alpha^{k+1})/(1 - \alpha)$. With a well-matched draft giving an acceptance rate of 0.7 and a proposal length of 4, that is $(1 - 0.7^{5})/(1 - 0.7) = 0.832 / 0.3 \approx 2.8$ tokens per target forward pass, versus exactly 1 without speculation. Fewer target passes per output token means the expensive weight reads happen 2.8 times less often, so decode throughput rises toward 2.8x — and in the cost equation that is a 2.8x multiplier on $T$.

The catch, and the reason speculative decoding is not a default-on lever like FP8, is threefold. The draft model consumes its own memory and a slice of compute on every step, so the realized speedup after draft overhead is usually 1.5–2.5x rather than the theoretical 2.8x. The acceptance rate $\alpha$ is workload-dependent — a draft model well-aligned to your traffic hits 0.7+, a poorly-matched one drops to 0.4 and can make things *slower* once draft overhead is counted. And a low-$\alpha$ regime wastes target compute on rejected tokens, so the lever has a genuine failure mode where it raises cost. The rule: measure $\alpha$ on your real traffic before you bank the saving, and treat speculative decoding as a 1.5–2x lever you validate, not a free one you assume.

The reason throughput per GPU is inseparable from cost is that it interacts with your choice of *which GPU*. The cheapest GPU per hour is very rarely the cheapest GPU per token, because per-token cost is set by throughput-per-dollar, not by sticker price. The matrix below makes that concrete across the common serving GPUs, all costed at peak utilization so the comparison is apples-to-apples on the hardware alone.

![Matrix comparing H100, A100 80GB, A100 40GB, L40S, L4, and a hosted API across price per hour, 8B FP8 throughput, cost per million tokens at peak, and best-fit workload, showing the cheapest per-hour GPU is not the cheapest per token](/imgs/blogs/cost-optimization-at-llm-scale-8.webp)

The lesson in that table is the L4 row. At \$0.60 an hour it is by far the cheapest GPU listed, seven times cheaper per hour than an H100 — and yet at \$0.37 per million tokens it is the *most expensive* per token, because its throughput on an 8B model (around 450 tokens per second) is more than ten times lower than the H100's. Sticker price lied. Meanwhile the L40S, at \$1.10 an hour, edges out the H100 on cost per token for small models because its throughput-per-dollar is excellent in that regime. The right-sizing rule falls straight out of the equation: **pick the GPU that minimizes $P/T$ for your model size, not the one that minimizes $P$.** Here is that comparison as a script you can run against your own measured throughput numbers.

```python
# right_sizing.py — rank GPU types by cost per token, not by sticker price.
# Fill tokens_per_sec from YOUR benchmark of YOUR model on each GPU;
# these are representative 8B FP8 aggregate-output numbers.
gpus = [
    # name,          $/hr,  tok/s (8B FP8),  vram_gb
    ("H100 80GB",    4.00,  5500,            80),
    ("A100 80GB",    2.50,  2800,            80),
    ("A100 40GB",    1.80,  2400,            40),
    ("L40S 48GB",    1.10,  1600,            48),
    ("L4 24GB",      0.60,  450,             24),
]

def cost_per_1m(price_hr, tok_s, utilization=1.0):
    return (price_hr * 1_000_000) / (tok_s * 3600 * utilization)

ranked = sorted(gpus, key=lambda g: cost_per_1m(g[1], g[2]))
print(f"{'GPU':<12} {'$/hr':>6} {'tok/s':>7} {'$/1M (peak)':>12}  {'per-hr rank'}")
for name, price, toks, _ in ranked:
    c = cost_per_1m(price, toks)
    print(f"{name:<12} {price:>6.2f} {toks:>7} {c:>11.3f}")
# L40S 48GB     1.10    1600        0.191   <- cheapest per token
# H100 80GB     4.00    5500        0.202
# A100 40GB     1.80    2400        0.208
# A100 80GB     2.50    2800        0.248
# L4 24GB       0.60     450        0.370   <- cheapest per HOUR, dearest per TOKEN
```

Two caveats keep this honest. First, throughput-per-dollar flips with model size: a 70B model will not fit on an L4 at all and runs miserably on an L40S, so for large models the H100's raw throughput and memory make it the cost winner despite the price. Second, these throughput numbers are *yours to measure* — they depend on your sequence lengths, batch sizes, and engine config, and the whole exercise is worthless if you plug in a vendor's cherry-picked benchmark instead of your own. Measure on your traffic, then rank.

## 5. Lever two: raise utilization

Utilization is the lever that recovers money you have already spent, which makes it psychologically the easiest to justify and operationally the trickiest to pull, because the thing standing between you and 90% utilization is latency: a fully-packed GPU has no slack to absorb a burst, and the burst is when your SLO is tested.

**Autoscaling** is the primary utilization lever. The goal is to make the size of the fleet track the size of the traffic, so that off-peak hours run on a smaller fleet instead of an idle large one. The subtlety unique to LLM serving is *what signal to scale on*. CPU utilization is useless — the GPU is the bottleneck. Even GPU utilization is a laggard, because by the time it reads 100% the queue is already forming. The signal that actually predicts an SLO breach is **queue depth** (or its close cousin, the number of pending requests per replica), and scaling on it via a custom metric is the difference between an autoscaler that reacts before the p99 moves and one that reacts after. The full HPA-and-KEDA mechanics live in [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling); the cost point is simply that converting a static peak-sized fleet into a queue-driven autoscaled one is what took our example from 35% to 70% utilization, a clean 2x on the $U$ lever.

**Scale-to-zero** is autoscaling's endgame for the right workloads. A model that serves a handful of requests an hour — an internal tool, a rarely-used specialist model, a dev endpoint — should not hold a GPU at idle 24/7. Scaling its replica count to zero when the queue is empty, and cold-starting a GPU when a request arrives, converts a fixed monthly GPU cost into a near-zero one. The catch is the cold start: loading a 16GB model onto a freshly-scheduled GPU can take 30 seconds to a few minutes, which is fine for a background job and unacceptable for an interactive endpoint. Scale-to-zero is a spiky-and-tolerant-traffic lever, not a chat lever, and mixing the two is a classic own-goal.

**MIG and bin-packing** attack a different utilization failure: the small model on the big GPU. If you serve a 3B model that needs 8GB on an 80GB H100, you are wasting 90% of the card. NVIDIA's Multi-Instance GPU (MIG) partitions one physical H100 into as many as seven isolated slices, each with its own memory and compute, so seven small models can share one card at near-full aggregate utilization. Bin-packing is the scheduler-level version: pack multiple small model replicas onto one GPU up to its memory limit. Either way the density gain is 2–7x for small-model fleets, and the cost is noisy-neighbor jitter — one slice's burst can nudge another's latency — which is tolerable for most workloads and disqualifying for the tightest SLOs. Again, the details are in the [GPU scheduling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) post; the costing takeaway is that MIG turns a 10%-utilized big GPU into a 70%-utilized one, and that is a 7x cost cut for that model with no change to price or throughput.

The reason you cannot simply autoscale to 100% utilization and call it done is the shape of the latency-versus-load curve, and it is worth naming because it sets the ceiling on the whole utilization lever. Queuing theory tells us that as a server approaches full saturation, its queue wait does not rise linearly — it rises toward infinity, with a sharp knee. For a system at load factor $\rho$ (offered load over capacity), the mean queue wait scales roughly with $\rho / (1 - \rho)$: at $\rho = 0.7$ the factor is 2.3, at $\rho = 0.9$ it is 9, and at $\rho = 0.95$ it is 19. That knee is why the cost-optimal utilization for an SLO-bound service is not 100% but somewhere around 70–80% — past the knee, a tiny increase in utilization buys a huge increase in tail latency, and you breach the SLO long before you run out of GPU. The utilization lever, in other words, has a hard ceiling set by your latency contract, and the whole game is to autoscale *up to* the knee without crossing it. This is the queuing-theory reason the guardrails section insists on keeping 20% headroom: that headroom is not waste, it is the distance between you and the knee.

The reason utilization levers are ranked below throughput levers despite recovering real money is the cold-start-and-jitter risk: every one of them trades a slice of latency headroom for a slice of cost, and the trade is only safe when you have measured how much headroom your SLO can spare.

## 6. Lever three: buy cheaper capacity

Now we push on price, the numerator, where the biggest single dollar swings live — and the most operational risk. There are three sub-levers: commit to capacity for a discount, rent interruptible capacity for a deeper discount, and pick the cheapest capacity type for each slice of your traffic. The figure below is the whole payoff of getting the mix right: the same traffic served two ways.

![Before-and-after figure contrasting an on-demand over-provisioned fleet of twenty H100s at 35 percent utilization costing 1.27 dollars per million tokens against a committed-base-plus-spot autoscaled fleet at 70 percent utilization costing 0.17 dollars per million tokens](/imgs/blogs/cost-optimization-at-llm-scale-3.webp)

**Reserved and committed-use discounts** are the simplest and least risky price lever: promise the cloud provider you will run a GPU for a year or three, and they knock 30–60% off the on-demand rate. The catch is lock-in — you are paying for that GPU whether or not you use it, which is only a win if your utilization on it is high. The rule is to **commit only to your stable base load**, the floor of your traffic that is present 24/7/365, and to source everything above that floor from more flexible capacity. Committing to your peak is how you recreate the over-provisioning problem with a discount attached.

There is a nuance in the *form* of the commitment that is worth getting right, because the clouds sell two shapes of it. A *reserved instance* (or its committed-use-discount equivalent) locks you to a specific instance type in a specific region — the deepest discount, but the least flexible, because if you later need a different GPU the commitment is stranded. A *savings plan* commits you to a dollar-per-hour spend rather than a specific instance, trading a few points of discount for the freedom to move that spend across instance types as your model or GPU choice evolves. For a fast-moving ML platform that might swap an A100 fleet for an H100 fleet mid-commitment, the savings-plan flexibility is usually worth the smaller discount; for a stable, long-lived service on a settled GPU, the reserved instance's deeper cut wins. Compute both against your realistic 12-month roadmap, not against today's fleet — a deep discount on hardware you abandon in six months is not a discount, it is a stranded cost.

A quick sizing check keeps the commitment honest. If your base load is 6 GPUs present every hour of the month, a one-year commitment at \$2.80 costs $6 \times 2.80 \times 730 = \$12{,}264$ a month regardless of usage; running those same 6 GPUs on-demand at \$4.00 would cost \$17,520. The commitment saves \$5,256 a month — but *only* if those 6 GPUs are genuinely busy every hour. Commit to 10 GPUs when your true floor is 6, and the 4 idle committed GPUs burn \$8,176 a month producing nothing, wiping out the discount entirely. The discipline: measure your actual traffic floor over several weeks, commit slightly *under* it, and let flexible capacity carry the uncertain margin.

**Spot and preemptible instances** are the deepest price cut available: the same GPU at 50–90% off, on the condition that the provider can reclaim it with little notice when it needs the capacity back. For a stateless web server this is terrifying; for LLM inference it is surprisingly manageable, because a single request completes in seconds and the major providers give you a warning before eviction. The blended-cost math is what makes it work, and it is worth deriving carefully because the interruption rate changes the number.

#### The mechanics: spot-plus-on-demand blended cost

You cannot run purely on spot — a mass eviction would take your whole service down — so the pattern is a *base* of committed or on-demand capacity that guarantees a floor of availability, plus a *burst* of spot that carries the rest at a deep discount. If a fraction $f_{\text{spot}}$ of your fleet is spot at price $p_{\text{spot}}$ and the remaining $f_{\text{base}} = 1 - f_{\text{spot}}$ is committed base at price $p_{\text{base}}$, the naive blended price is:

$$p_{\text{blend}} = f_{\text{base}} \cdot p_{\text{base}} + f_{\text{spot}} \cdot p_{\text{spot}}$$

For our example: 45% committed base at \$2.80 an hour (a 30% commitment discount off the \$4.00 on-demand rate) and 55% spot at \$2.00 an hour (a 50% spot discount) gives $0.45 \times 2.80 + 0.55 \times 2.00 = 1.26 + 1.10 = \$2.36$ per hour — the blended price in our target scenario.

But that number is optimistic, because it ignores the *interruption tax*. When a spot GPU is reclaimed mid-flight, you lose the work in progress and must re-run it elsewhere, and you may briefly fail over to more expensive on-demand capacity while spot is unavailable. Model that as an overhead $o$ on the spot fraction's effective cost — the fraction of spot work that has to be redone or absorbed by pricier fallback. With a 10% overhead, the spot GPU's effective price rises from \$2.00 to \$2.20, and the honest blended price becomes:

$$p_{\text{blend}}^{\text{real}} = 0.45 \times 2.80 + 0.55 \times (2.00 \times 1.10) = 1.26 + 1.21 = \$2.47 \text{ per hour}$$

Feeding that back through the cost equation moves the optimized number from \$0.17 to \$0.18 per million tokens — a 6% haircut on the headline. That is the correct way to quote a spot saving: not "spot is 50% off" but "spot, after the interruption tax on my workload, blends to \$2.47 an hour and costs me \$0.18 per million tokens." The overhead $o$ is yours to measure from your own interruption logs; for well-behaved inference with fast draining it is typically 5–15%, and it is the number that decides whether spot is worth the operational complexity.

Which capacity type to use for which slice of traffic is itself a decision with a clean structure, driven entirely by the *shape* of your traffic. The tree below is the sourcing decision I walk through for every new service.

![Decision tree rooted at sourcing GPU capacity, branching on traffic shape into steady 24/7 load buying 3-year commitments, bursty diurnal load blending a 1-year committed base with spot for peaks, and spiky rare load using scale-to-zero serverless or an external API](/imgs/blogs/cost-optimization-at-llm-scale-7.webp)

The logic reads top-down. **Steady 24/7 load above 70% utilization** is the easy case: commit hard, three years if you are confident, and capture the 40–60% discount, because you will use every hour you pay for. **Bursty but predictable diurnal load** — the common case for a consumer service — wants a *split*: a one-year committed base sized to cover the valley (the floor of traffic that is always present), plus spot to cover the predictable daytime peak, with autoscaling driving the spot count. **Spiky, rare load** with a low duty cycle — under 10% of hours see real traffic — should buy nothing standing: use scale-to-zero serverless GPUs, or, below a volume threshold we derive in the build-versus-buy section, skip self-hosting entirely and call an external API where you pay strictly per token and nothing when idle.

#### Worked example: sizing the base-plus-spot split

A service has a daytime peak of 20 H100-equivalents and an overnight floor of 6. Static on-demand provisioning for peak means 20 GPUs at \$4.00, billed 24/7: $20 \times 4.00 \times 730 = \$58{,}400$ a month. Now split it. Commit to the 6-GPU floor on a one-year term at \$2.80: $6 \times 2.80 \times 730 = \$12{,}264$ a month, always on. Source the peaking 0–14 extra GPUs from spot at \$2.00, autoscaled, running an average of 10 hours a day at an average of 9 extra GPUs: $9 \times 2.00 \times (10 \times 30) = \$5{,}400$ a month. Total: about \$17,700 a month against \$58,400 — a **3.3x cut on the capacity bill alone**, before any throughput or utilization gain, purely from matching capacity type to traffic shape. Stack this on the throughput and utilization levers and you are back to the 7.5x headline.

## 7. Lever four: do not compute at all

The cheapest token is the one you never generate. This lever sidesteps the cost equation entirely: a request answered from cache, or routed to a model ten times smaller, or deferred to an offline batch, does not consume the expensive online GPU-second at all. The gains here are the highest-variance in the whole post — sometimes zero, sometimes 90% — because they depend entirely on how repetitive and how heterogeneous your traffic is. But when they apply, nothing else comes close.

**Prefix caching** exploits the fact that most LLM requests share a long, identical prefix — a system prompt, a few-shot preamble, a RAG context reused across a session. The prefill over that shared prefix is pure recomputation on every request unless you cache its KV state and reuse it. vLLM's automatic prefix caching and the RadixAttention scheme in SGLang do exactly this, and for a service with a 2,000-token system prompt and short user turns, prefix caching can eliminate the majority of prefill compute — a large decode-side saving that shows up directly as higher effective throughput. The mechanism is the subject of [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention); the cost framing is that it removes prefill work you were paying to redo.

**Semantic caching** goes further and riskier: cache not on exact-match input but on *embedding similarity*, so that "what is your refund policy?" and "how do I get a refund?" hit the same cached answer. For FAQ-heavy and support workloads the hit rate can be startling — some deployments report 30% or more of traffic served from a semantic cache — and each hit is a request that cost an embedding lookup instead of a full generation. The risk is staleness and false hits: two questions that embed close but want different answers. Semantic caching needs a similarity threshold you tune conservatively and a TTL so answers do not rot.

**Request deduplication and small-model routing** round out the family. Deduplication collapses identical in-flight requests — common during a retry storm or a viral moment — into one generation fanned out to many callers. Small-model routing recognizes that not every request needs your largest model: a classification, a short factual answer, or a simple rewrite can go to an 8B model at a fifth of the cost of the 70B, and only the genuinely hard requests pay for the big model. The router that ties all of this together is the highest-ROI piece of cost infrastructure most teams never build.

![Graph of a cost-aware router where an incoming request is classified in under a millisecond and branches to a prefix or semantic cache at near-zero compute, a small 8B model at 5x cheaper, an offline batch queue at 3x cheaper, or the full-cost 70B online model, with all paths merging at a single response](/imgs/blogs/cost-optimization-at-llm-scale-5.webp)

Here is the routing rule at the center of that figure, written as a small, dependency-light classifier you can put in front of any serving stack. The point is not the exact heuristics — you will tune those to your traffic — but the *structure*: estimate the cheapest viable path for each request, and only fall through to the expensive online large model when nothing cheaper will do.

```python
# cost_router.py — route each request to its cheapest viable path.
import time

# Relative cost weights (per request, normalized to the big online model = 1.0).
COST = {"cache": 0.0, "small": 0.20, "batch": 0.33, "big": 1.0}

def route(req, cache, embed_cache):
    """Return (path, est_cost). Cheapest viable path wins."""
    # 1. Exact prefix / cached response? Near-zero cost.
    if cache.get(req["prompt_hash"]) is not None:
        return "cache", COST["cache"]

    # 2. Semantically close to a cached answer, above a conservative threshold?
    hit, score = embed_cache.nearest(req["embedding"])
    if hit is not None and score >= 0.92:      # tuned high to avoid false hits
        return "cache", COST["cache"]

    # 3. Deadline-loose work (no interactive user waiting) -> offline batch.
    if req.get("max_latency_s", 0) >= 3600 or req.get("async", False):
        return "batch", COST["batch"]

    # 4. Simple/short request the small model handles well -> 8B.
    if req["prompt_tokens"] < 512 and req["task"] in {"classify", "rewrite", "extract"}:
        return "small", COST["small"]

    # 5. Fall through: genuinely hard, interactive -> full-cost 70B online.
    return "big", COST["big"]

def expected_blended_cost(traffic_mix):
    """traffic_mix: dict path -> fraction of requests. Returns cost vs all-big."""
    blended = sum(frac * COST[path] for path, frac in traffic_mix.items())
    return blended, f"{1/blended:.1f}x cheaper than all-big" if blended else "free"

# Example: 30% cache, 15% batch, 25% small, 30% big
mix = {"cache": 0.30, "batch": 0.15, "small": 0.25, "big": 0.30}
blended, verdict = expected_blended_cost(mix)
print(f"blended cost factor: {blended:.3f} of all-big  ({verdict})")
# blended cost factor: 0.400 of all-big  (2.5x cheaper than all-big)
```

That example mix — 30% cache, 15% batch, 25% small, 30% big — blends to 0.40 of the all-big cost, a 2.5x cut, and it required no faster GPUs and no cheaper capacity. It is pure avoided compute. The router is also the safest lever in the post from an SLO standpoint, because every cheaper path it chooses either meets the SLO trivially (cache) or is explicitly gated on the request not being latency-sensitive (batch). The one discipline it demands is honest classification: a router that sends a hard request to the small model to save money produces a bad answer, and a bad answer is a cost too, just one that lands on a different dashboard.

## 8. Batch versus online: the same tokens, a third of the price

There is a structural cost gap between serving a token *now*, under a latency SLO, and serving it *later*, on your own schedule. It is large — typically 2–5x — and it comes from the fact that a latency SLO forbids the very things that make inference cheap.

Online serving under an SLO must cap batch size (a bigger batch raises inter-token latency), must hold headroom (a saturated GPU cannot absorb a burst), and must run on always-available capacity (you cannot risk a spot eviction stalling an interactive user for a minute). Offline batch serving has none of those constraints. With no user waiting, you can push the batch size to the memory limit, drive utilization toward 90%, run entirely on the cheapest spot capacity, and schedule the whole job for the overnight trough when GPUs are idle and spot is cheapest. Every one of those is a lever the online path could not pull.

#### Worked example: routing eval jobs to the batch lane

Our online 8B service costs \$0.17 per million tokens. Take the same model and run a nightly evaluation job — ten million documents to summarize, no user waiting, a deadline of "before the morning standup." Serve it on all-spot capacity at \$2.00 an hour, at 90% utilization (no headroom needed), with a batch large enough to push throughput to 7,000 tokens per second (no latency cap):

$$C_{1M}^{\text{batch}} = \frac{2.00 \times 10^{6}}{7000 \times 3600 \times 0.90} = \frac{2{,}000{,}000}{22{,}680{,}000} \approx \$0.088 \text{ per 1M tokens}$$

The identical tokens cost half as much, purely because no SLO was in the way. The rule that follows is a routing rule: **any work without a human waiting on it belongs in the batch lane.** Evals, bulk summarization, embedding backfills, synthetic data generation, offline classification — all of it. The industry has priced this gap explicitly: OpenAI's Batch API and Anthropic's Message Batches both offer a flat 50% discount for jobs you submit asynchronously with a 24-hour completion window, which is precisely the online-versus-batch gap externalized into a price list. If a provider will sell you the batch lane at half price, that is a strong signal the gap is real and you should be capturing it internally too.

The routing logic is the `batch` branch of the `cost_router.py` above: any request that declares `async=True` or a `max_latency_s` beyond an hour drops out of the expensive online lane and into a queue that a separate, spot-heavy, batch-optimized deployment drains on its own schedule. The one operational cost is that you now run two deployments — an online one tuned for latency and a batch one tuned for throughput — but they can share the same model weights and the same container image, differing only in engine flags and node pool.

## 9. Build versus buy: where self-hosting starts to win

Every cost conversation eventually reaches the question that makes it existential: should we be running our own GPUs at all, or just paying an API per token? The answer is a breakeven volume, and it is computable, but the honest version has two very different answers depending on which API you are comparing against.

The structure is this. Self-hosting is a *fixed* cost — you pay for the GPU whether it serves one token or a billion — with a low marginal cost per token once the hardware is bought. An API is a *pure variable* cost — nothing when idle, a fixed price per token when used. Fixed-versus-variable always has a crossover: below some volume the API's pay-nothing-when-idle wins; above it, self-hosting's amortized fixed cost wins and keeps widening its lead.

#### The mechanics: the breakeven volume

Let the self-hosted fixed cost be $F$ dollars per month and the API price be $A$ dollars per million tokens. Self-hosting is cheaper once your monthly volume $V$ (in millions of tokens) satisfies $F < A \cdot V$, i.e.:

$$V^{*} = \frac{F}{A} \quad \text{[million tokens per month]}$$

Take a 70B model self-hosted on two committed H100s at \$2.80 an hour: $F = 2 \times 2.80 \times 730 = \$4{,}088$ per month. That node, at a 70B throughput of about 1,500 tokens per second and full utilization, can produce $1500 \times 3600 \times 730 \approx 3.94$ billion tokens a month, for a floor cost of \$1.04 per million tokens when saturated.

Now compare against two different APIs. Against a **frontier proprietary API** priced at a blended \$5.00 per million tokens (the ballpark for a top-tier hosted model), the breakeven is $V^{*} = 4088 / 5.00 = 818$ million tokens per month — roughly 27 million tokens a day, or a sustained 316 tokens per second. Above that modest volume, self-hosting a capable open 70B beats the frontier API, and at saturation it is nearly five times cheaper (\$1.04 versus \$5.00 per million). This is the calculation behind every "we moved off the expensive API and self-hosted an open model" blog post: if your volume is real and your quality bar is met by an open model, the frontier API's per-token price is very hard to beat with your own margin on top.

Against a **commodity open-model serverless API** priced at \$0.60 per million — the cut-throat rate for a hosted Llama-70B on a competitive inference provider — the breakeven balloons to $V^{*} = 4088 / 0.60 = 6.8$ billion tokens a month, which is *above* a single node's 3.94-billion-token ceiling. In other words, one node cannot even reach breakeven against a cheap serverless provider; you would need multiple nodes running near saturation, and the provider — who runs at a scale and utilization you cannot match, and batches across thousands of customers — will likely still undercut you. The uncomfortable truth is that **self-hosting rarely beats a well-run commodity serverless API for an open model until you are at very high, steady volume with your own optimization stack** — because that provider has already pulled every lever in this post, at a scale that dilutes their fixed costs far more than yours.

```python
# build_vs_buy.py — breakeven volume for self-host vs an API.
def breakeven_million_tokens(gpus, price_per_gpu_hr, api_price_per_1m, hours=730):
    fixed_monthly = gpus * price_per_gpu_hr * hours
    v_star = fixed_monthly / api_price_per_1m          # million tokens / month
    return fixed_monthly, v_star

for label, api in [("frontier API", 5.00), ("commodity serverless", 0.60)]:
    fixed, v = breakeven_million_tokens(gpus=2, price_per_gpu_hr=2.80, api_price_per_1m=api)
    print(f"vs {label:22s}: fixed=${fixed:,.0f}/mo, breakeven={v:,.0f}M tok/mo "
          f"({v/30:,.0f}M/day)")
# vs frontier API         : fixed=$4,088/mo, breakeven=818M tok/mo (27M/day)
# vs commodity serverless : fixed=$4,088/mo, breakeven=6,813M tok/mo (227M/day)
```

There is a second honesty check the breakeven formula quietly omits: the fixed cost $F$ in the numerator is *not* just the GPU rental. Self-hosting carries a tail of costs that never show up on a per-token API invoice — the storage for model weights and checkpoints, the cross-zone networking for a multi-GPU deployment, the load balancer and gateway, the observability stack, and, largest of all, the fraction of an engineer's salary spent building and operating the fleet. A realistic self-host $F$ for the 70B example is not the \$4,088 of raw GPU rental but closer to \$5,500–6,500 a month once storage, networking, and a slice of on-call are counted. That inflated $F$ pushes the breakeven volume up by 30–60%, and for a small team the engineering-time term dominates everything else: the weeks spent building drain-and-failover, autoscaling, and cost attribution are real money that a managed API simply does not charge you. Fold those into $F$ before you draw the line, or you will self-host at a volume where the true cost — salaries included — is higher than the API you left.

The decision rule: **self-host when your volume clears the breakeven against your quality-matched alternative, and only then.** Below breakeven, the API is not just cheaper, it is also less operational risk, no on-call, and no capacity planning — buy it and spend your engineering time elsewhere. The build-versus-buy question is not ideological; it is a line on a chart, and the line moves with your volume, your quality bar, and which API you are actually replacing.

## 10. Spot interruption handling: making the cheapest capacity safe

The deepest price discount — spot — is only bankable if an interruption is an overhead rather than an outage. The whole trick is that you get *warning*: AWS gives a two-minute termination notice, GCP and Azure give around thirty seconds. Two minutes is an eternity for LLM inference, where a request completes in seconds, and that is what makes spot serving safe. The lifecycle below is the drain-and-failover choreography that turns a reclaim into a non-event.

![Timeline of the spot interruption lifecycle from a healthy spot GPU serving at 2 dollars per hour, through the T-minus-120-second termination notice, stopping admission and draining in-flight requests, finishing decode within 30 seconds, failing over to the on-demand base, and finally reclaiming spot when capacity returns](/imgs/blogs/cost-optimization-at-llm-scale-6.webp)

The five moves, in order: the termination notice fires, so you **stop admitting new requests** to that GPU immediately; you **let in-flight requests drain** (finish their decode — seconds, well inside the two-minute window); the load balancer **fails the traffic over to the on-demand base**, which is why the base exists; and when spot capacity returns, the autoscaler **rebalances** back onto it. Done right, not a single token is dropped, and the only cost is the sliver of work that had to move — the interruption tax we already priced into the blended \$2.47.

Implementing this is mostly Kubernetes plumbing plus a node pool that mixes capacity types. Here is a Karpenter NodePool that lets the scheduler place burst pods on spot while keeping a labeled on-demand base, paired with a Deployment whose base replicas are pinned to on-demand and whose burst replicas prefer spot, plus the drain hook that makes the two-minute notice count.

```yaml
# karpenter-nodepool.yaml — mixed spot + on-demand GPU capacity.
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]        # scheduler picks cheapest that fits
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["p5.48xlarge", "p4d.24xlarge"]
      nodeClassRef:
        name: gpu-nodeclass
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized  # pack pods, kill idle nodes
    consolidateAfter: 2m
  limits:
    nvidia.com/gpu: 40
```

```yaml
# deployment.yaml — base replicas on-demand, burst replicas prefer spot.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 20
  template:
    spec:
      terminationGracePeriodSeconds: 120   # match the spot notice window
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 80                     # prefer spot for the bulk of replicas
              preference:
                matchExpressions:
                  - key: karpenter.sh/capacity-type
                    operator: In
                    values: ["spot"]
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          lifecycle:
            preStop:
              exec:
                # Stop admitting, then drain in-flight decode before the node dies.
                command: ["/bin/sh", "-c",
                          "curl -s -XPOST localhost:8000/v1/admission/close; sleep 100"]
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 2                  # fail out of the LB fast on drain
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: llm-serving-pdb
spec:
  minAvailable: 6                            # never drop below the on-demand base
  selector:
    matchLabels: { app: llm-serving }
```

The load-bearing lines are `terminationGracePeriodSeconds: 120` (use the whole notice window to drain), the `preStop` hook (close admission, then let decode finish before the container exits), the fast `readinessProbe` (evacuate the load balancer the instant a drain starts), and the `PodDisruptionBudget` with `minAvailable: 6` (a hard floor equal to your on-demand base, so a spot storm can never take the service below its guaranteed capacity). With this in place, spot interruptions become a line in a log rather than a page at 3 a.m., and the 50% discount is finally yours to keep.

A spot-safety checklist worth internalizing: never run more than your fault-tolerance budget on spot (the base must survive a full spot eviction); diversify across instance types and availability zones so a single-pool capacity crunch does not evict everything at once; and monitor your realized interruption rate, because that measured number is the $o$ in the blended-cost formula and it decides whether the whole exercise is paying off.

## 11. Cost attribution: showback before you can optimize

You cannot optimize a number you cannot see, and "the inference bill was \$412,000" is not a number you can act on. The actionable version is per-tenant, per-feature, per-model: *which* team, *which* product surface, *which* model is generating the cost. Cost attribution — often called showback when you report it and chargeback when you bill it back — is the instrumentation that turns one scary aggregate into a hundred small, ownable ones.

The mechanism is to tag every request with its cost dimensions — tenant, feature, model, request class — emit a token count per request, and multiply by the unit cost you now know how to compute. In a Prometheus-and-Grafana stack this is a counter incremented per request and a recording rule that turns tokens into dollars. Here is the shape of it.

```python
# instrument.py — emit per-request cost dimensions from the serving middleware.
from prometheus_client import Counter

TOKENS = Counter(
    "llm_tokens_total", "Tokens served",
    ["tenant", "feature", "model", "kind"],   # kind = input | output
)

# Cost per token differs for input vs output (decode is ~4x pricier). Derived
# from the unit-economics model, exported as a Prometheus gauge the query reads.
def record(tenant, feature, model, input_tokens, output_tokens):
    TOKENS.labels(tenant, feature, model, "input").inc(input_tokens)
    TOKENS.labels(tenant, feature, model, "output").inc(output_tokens)
```

```promql
# showback.promql — dollars per tenant over the last day, split by model.
# unit_cost_usd_per_1m is a gauge you set from the cost equation, per model+kind.
sum by (tenant, model) (
    rate(llm_tokens_total[1d])
  * on(model, kind) group_left
    unit_cost_usd_per_1m / 1e6
) * 86400

# Top 10 most expensive (tenant, feature) pairs this week — where to aim the next cut.
topk(10,
  sum by (tenant, feature) (
      rate(llm_tokens_total[7d])
    * on(model, kind) group_left
      unit_cost_usd_per_1m / 1e6
  )
)
```

Attribution changes the conversation with finance from "the bill went up" to "the bill went up because the new document-summarization feature is sending 8-million-token contexts to the 70B model at full price, and here is the fix — route it to the batch lane and turn on prefix caching, which cuts that feature's cost 4x." It also creates the incentive that makes cost optimization self-sustaining: when each team sees its own showback number, the expensive patterns get fixed at the source without a central FinOps team chasing them. The single most valuable dashboard in a mature LLM platform is not the latency dashboard — it is the cost-per-feature dashboard, because it tells you where the next dollar of savings is hiding.

Attribution also unlocks the two controls that keep a bill from surprising you again. The first is a **per-tenant budget with an alert**: once every request is tagged and costed, a Prometheus alert rule that fires when a tenant's rolling daily spend crosses a threshold turns a month-end shock into a same-day heads-up, and it is the difference between catching a runaway retry loop in an hour versus in a billing cycle. The second is **chargeback** — actually billing the cost back to the team or customer that incurred it, rather than merely reporting it. Chargeback is heavier to operate than showback because the numbers must be defensible enough to put on an invoice, but it is the strongest possible incentive: a team that pays for its own tokens optimizes its own prompts, caps its own context lengths, and routes its own cheap requests to cheap models without any prompting from the platform team. Start with showback to build trust in the numbers, then graduate the highest-spend surfaces to chargeback once the attribution is proven accurate.

## 12. Guardrails: never cut cost into an SLO violation

Everything in this post is a trade against latency, and every trade has a line past which the savings are illusory because they have been paid for out of the SLO. A cost cut that pushes p99 TTFT past your contract has not reduced cost; it has converted a cloud-bill cost into a churn cost, which is larger and lands on a dashboard you do not control. The guardrails that keep the levers honest:

**Cap batch size below the throughput-maximizing point.** Bigger batches raise throughput and lower cost, right up until they raise inter-token latency past your TPOT SLO. The cost-optimal batch size and the SLO-safe batch size are different numbers, and you want the smaller one. Set `max_num_seqs` in vLLM to the SLO-safe value, not the throughput-max value.

**Keep real headroom, just make it dynamic.** Autoscaling should not drive utilization to 100% — it should drive it to the highest level that still leaves burst absorption, typically 70–80%. The last 20% of utilization is not waste; it is the insurance that holds your p99 when traffic spikes. Cutting it is the most common way a cost optimization becomes an incident.

**Gate spot on fault tolerance, not on savings.** The spot fraction is bounded by how much of your fleet can vanish at once without breaching availability — never by how much money spot saves. If your base cannot carry the service alone during a full spot eviction, your spot fraction is too high regardless of the discount.

**Route on quality, not just cost.** Small-model routing and semantic caching save money by serving a cheaper answer, and a cheaper answer that is *wrong* is the most expensive outcome of all. Both need a quality gate — a confidence threshold, a conservative similarity cutoff, a fallback to the big model when the small one is unsure.

The discipline that ties these together: instrument the SLO *and* the cost on the same dashboard, and treat any cost optimization that moves the SLO as a regression until proven otherwise. The goal is not the lowest possible cost; it is the lowest cost *at which you still meet the contract*. That constraint is what makes this engineering rather than penny-pinching.

## Case studies and benchmarks

Four real reference points, framed honestly with their sources, to calibrate what these levers deliver in the wild.

**DeepSeek's inference economics (Open Source Week, February 2025).** DeepSeek published a rare public breakdown of their V3/R1 serving system, which combines large-scale expert parallelism, prefill-decode disaggregation, and aggressive utilization on H800 GPUs. Their disclosed *theoretical* figures — assuming a \$2 per H800-hour cost — put the daily cost of the serving system around \$87,000 against a theoretical daily revenue at their API prices of roughly \$562,000, a cost-profit margin they described as about 545%. These are the company's own theoretical numbers and assume full utilization and their published token mix rather than realized billing, so treat them as an existence proof rather than a guarantee: they show that MoE routing plus PD-disaggregation plus high utilization can push cost per token low enough to run an API at a large margin. The techniques are the subject of the series' [multi-node LLM serving](/blog/machine-learning/model-serving/multi-node-llm-serving-100b-plus) discussions of expert and pipeline parallelism.

**Character.AI's caching-driven cost reduction (2024).** Character.AI reported that through int8 quantization, multi-query attention to shrink the KV cache, and a system-wide prompt cache achieving a very high hit rate, they drove serving cost down by more than an order of magnitude relative to their late-2022 baseline while serving traffic at very large scale. The reported cache hit rate — north of 90% for their conversational workload — is the standout number, and it is the clearest public evidence that the "do not compute" lever is the highest-ceiling one when your traffic is repetitive. As always with a company blog, the exact multiplier is theirs and depends on their specific workload; the directional lesson is that caching, not faster GPUs, was the dominant lever for a high-repetition chat service.

**The batch-versus-online gap, priced by the providers themselves.** OpenAI's Batch API and Anthropic's Message Batches both offer a flat 50% discount for asynchronous jobs with a 24-hour completion window. This is not a promotion; it is the online-versus-batch structural gap externalized into a price list. When two major providers independently price the batch lane at half the online lane, that is a strong, citable signal that the gap in our worked example (\$0.17 online versus \$0.088 batch, roughly 2x) is real and conservative, and that any deadline-loose work you run online is leaving that discount on the table.

**Spot inference savings and interruption behavior.** The public, verifiable facts about spot are the notice windows — AWS's two-minute EC2 Spot interruption notice, GCP's and Azure's roughly thirty-second preemption notices — and the discount range, commonly 50–90% off on-demand for GPU instances depending on region and instance type. Realized interruption rates vary too much by pool and region to quote a single number honestly, which is exactly why the blended-cost formula treats the interruption overhead $o$ as a value you measure from your own logs rather than a constant. The two-minute window is the load-bearing fact: it is long enough that a drain-and-failover design drops zero tokens, which is what makes the 50%+ discount safely bankable for inference.

## Measurement: the before-to-after on named hardware

The whole post, compressed into one table. Each row adds one lever to our 8B chat model on H100 80GB and shows what it does to cost per million tokens, so you can see the multiplicativity accumulate rather than take the 7.5x on faith.

| Stage | Lever added | Price (\$/hr) | Throughput (tok/s) | Utilization | Cost / 1M tokens | Cumulative cut |
|---|---|---|---|---|---|---|
| Baseline | none (fp16, naive batch, on-demand) | 4.00 | 2,500 | 35% | \$1.27 | 1.0x |
| + Throughput | FP8 + continuous batching | 4.00 | 5,500 | 35% | \$0.58 | 2.2x |
| + Utilization | queue-driven autoscale + scale-to-zero | 4.00 | 5,500 | 70% | \$0.29 | 4.4x |
| + Capacity | committed base + spot blend | 2.36 | 5,500 | 70% | \$0.17 | 7.5x |
| + Don't-compute | cost router (30% cache/batch/small mix) | 2.36 | 5,500 | 70% | \$0.10 (effective) | ~12x |
| Batch lane | deadline-loose work, all-spot, U=90% | 2.00 | 7,000 | 90% | \$0.088 | ~14x (that slice) |

Two honesty notes on the table. The don't-compute row is marked "effective" because it does not change the per-token *production* cost — it changes how many tokens you produce at all, so it is a blended cost over your traffic mix, not a hardware number. And the batch-lane row applies only to the slice of work you can defer; you cannot serve interactive chat there. The headline for the online interactive path is the 7.5x — from \$1.27 to \$0.17 — achieved entirely with FP8, autoscaling, and a spot-plus-committed capacity mix, on the exact same H100 you started with. The remaining gains come from serving fewer expensive tokens (routing and caching) and moving deferrable work off the SLO-constrained path.

## When to use this (and when not to)

Cost optimization is not a virtue in itself, and there are clear cases where each lever is the wrong move.

**Do the throughput levers essentially always.** FP8 on an H100 and continuous batching in vLLM are close to free — they cost you a flag and a small TTFT-tail increase, and they multiply everything downstream. There is almost no service above trivial scale where you should not have these on. The exception is a latency-critical service so tight that even continuous batching's admission delay is unacceptable, which is rare.

**Reach for utilization levers when your traffic is spiky or your GPUs are under-packed.** Autoscaling pays off in proportion to your peak-to-trough ratio; if your traffic is dead flat 24/7, a static committed fleet at high utilization is already optimal and an autoscaler adds complexity for no gain. Scale-to-zero is for tolerant, spiky workloads only — put it in front of an interactive chat endpoint and the cold start will breach your TTFT SLO on the first request after a quiet spell.

**Do not chase spot into an operational hole.** Spot is worth it when your workload is fault-tolerant, your team can build and own the drain-and-failover machinery, and your volume is large enough that the 50% saving is real money. For a small service, a team without the ops bandwidth, or a workload where a rare dropped request is catastrophic, the on-demand or committed simplicity is worth the premium. The interruption tax is not just the $o$ in the formula — it is also the engineering time to build the handling, and for a small bill that time costs more than it saves.

**Do not self-host below breakeven.** If your volume is under the breakeven against your quality-matched API, the API is cheaper *and* lower-risk. Self-hosting a fleet to serve a few hundred million tokens a month against a cheap serverless provider is a way to spend engineering salaries to lose money. Compute the breakeven, and only build when you are decisively past it.

**Never cut a cost that costs you the SLO.** This is the master rule. Every lever here has a setting that saves more money and breaches the latency contract, and every one of those settings is wrong. The target is the lowest cost that still meets the SLO, and a proposal that beats the cost target by missing the latency target has failed, not succeeded.

## Key takeaways

- **Cost per million tokens has three inputs**: price, throughput, utilization. $C_{1M} = P \times 10^6 / (T \times 3600 \times U)$. Every optimization moves one of them; make each proposal declare which.
- **Throughput and utilization multiply.** Three modest 2x wins are close to an 8x cut, not a 6x one. This is why a ranked stack of unglamorous levers beats hunting for one silver bullet.
- **Utilization is the biggest recoverable waste**, because an idle GPU costs the same as a busy one. Autoscaling on queue depth and scale-to-zero for spiky models convert money you already spend into tokens.
- **Do the throughput levers before the capacity levers**, so every discounted GPU-hour you later buy is already twice as productive and you buy fewer of them.
- **The cheapest GPU per hour is rarely the cheapest per token.** Right-size on $P/T$, not on sticker price — an L4 is seven times cheaper per hour than an H100 and nearly twice as expensive per token.
- **Spot is safe for inference because you get warning.** The two-minute notice is enough to drain and fail over with zero dropped tokens; quote spot savings after the measured interruption tax, not before.
- **The cheapest token is the one you never generate.** Prefix caching, semantic caching, dedup, and small-model routing sidestep the cost equation entirely and have the highest ceiling when traffic is repetitive.
- **Anything without a human waiting belongs in the batch lane**, where no SLO forbids the big batches and full-spot capacity that make tokens 2–5x cheaper.
- **Self-host only past the breakeven** against your quality-matched API; below it, the API is cheaper and lower-risk.
- **Attribute cost per tenant and feature**, and never cut a cost into an SLO violation. The goal is the lowest cost at which you still meet the contract.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023) — the vLLM paper; the throughput foundation the whole cost model rests on.
- DeepSeek-AI, "Open Source Week" inference system disclosure (February 2025) — the published theoretical cost-and-margin breakdown of a large-scale MoE serving system.
- NVIDIA, "Multi-Instance GPU (MIG) User Guide" — the density lever for packing small models onto big GPUs.
- AWS, "Amazon EC2 Spot Instances" and "Spot Instance interruption notices" documentation — the two-minute notice and diversification guidance behind safe spot inference.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving), [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving), [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention), [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling), and [high-concurrency SLO management](/blog/machine-learning/model-serving/high-concurrency-slo-management).

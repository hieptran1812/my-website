---
title: "High-Concurrency SLO Management: Holding Latency Targets When the Traffic Wants to Kill You"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "How large chat and agent services hold TTFT and TPOT SLOs under heavy concurrent load by optimizing goodput, not raw throughput — with the queuing theory, vLLM configs, priority gateways, admission control, and SLO-aware autoscaling that make it work."
tags:
  [
    "model-serving",
    "inference",
    "ml-infrastructure",
    "slo",
    "goodput",
    "admission-control",
    "load-shedding",
    "autoscaling",
    "queuing-theory",
    "vllm",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/high-concurrency-slo-management-1.webp"
---

The alert that wakes you at 3 a.m. is almost never "the GPUs are idle." It is "p99 time-to-first-token just crossed two seconds and the on-call dashboard is a wall of red." You SSH in expecting a crash and find the opposite: every GPU is pinned at 98% utilization, tokens-per-second is the highest it has ever been, and the fleet is, by every hardware metric, working harder than ever. And yet users are furious, requests are timing out, and the retry counter on your gateway is climbing like a rocket. Nothing is broken. The system is doing exactly what you told it to do — maximize throughput — and that is precisely the problem.

This is the failure mode that separates a demo from a production LLM service. A demo serves one request at a time and looks fast. A production service serves thousands of concurrent conversations, each with a latency contract, and the moment offered load creeps past a hidden threshold, latency does not degrade gracefully — it explodes. The queue that was 3 deep at 9:00 a.m. is 400 deep at 9:05, every admitted request inherits the full queue wait, and a system that was comfortably meeting a 500 ms SLO is now missing it by a factor of ten while burning every watt you pay for. Raw throughput went *up*. The number of users who got an acceptable answer went *down*. That gap — between work done and useful work done — is the entire subject of this post.

The reframe that fixes it is a single word: **goodput**. Not throughput, the raw rate of tokens or requests the engine emits, but goodput — the rate of requests that *complete while meeting their SLO*. A request that finishes 400 ms late is, from the user's chair, indistinguishable from a request that failed. It consumed GPU cycles, KV-cache memory, and a scheduler slot, and it produced nothing you can bill for or be proud of. When you optimize goodput instead of throughput, every design decision in this post falls out naturally: why you cap batch size below the throughput-maximizing point, why you shed load with an HTTP 429 instead of queuing it, why you autoscale on queue depth instead of GPU utilization, and why the scheduler needs to know which requests are interactive chat and which are a background batch job.

By the end of this post you will be able to define per-class latency SLOs precisely, derive from queuing theory exactly how much capacity headroom a p99 target costs you, configure a vLLM engine and a priority-aware gateway to protect that target, write an admission controller that sheds load before the engine melts, and build an autoscaler that reacts to the signal that actually predicts an SLO breach. Everything ties back to the serving trade triangle — **latency ↔ throughput ↔ cost** — with a fourth corner, **fairness**, that shows up the moment you have more than one tenant. The figure below is the whole architecture in one picture: keep it in mind, because every section that follows drills into one of its boxes.

![Layered flow diagram showing ingress feeding an admission controller that either sheds overload with a 429 or admits into a priority classifier splitting latency-critical and best-effort queues that merge at an SLO-aware scheduler driving GPU workers](/imgs/blogs/high-concurrency-slo-management-1.webp)

If continuous batching and the per-step scheduler loop are new to you, the companion posts [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving) and [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) build the foundation this post stands on; here we zoom out from the single engine to the whole SLO-holding system around it.

## 1. Defining the SLOs precisely: one contract does not fit all traffic

You cannot hold a target you have not written down, and "the API should be fast" is not a target. For LLM serving there are three latency numbers that matter, and they measure genuinely different things:

- **TTFT (time to first token)** — the wall-clock time from when a request arrives at your gateway to when the first output token is streamed back. It is dominated by *queue wait* plus *prefill* (the one big forward pass over the whole prompt). For an interactive chat UI, TTFT is the "is it thinking?" latency the user feels most acutely. You track it at p95 and p99, not the mean, because the tail is what generates complaints.
- **TPOT (time per output token)**, also called **ITL (inter-token latency)** — the average time between consecutive output tokens during the decode phase. If TPOT is 40 ms, the model emits 25 tokens/second, which reads at a comfortable pace. TPOT is what makes a stream feel smooth or stuttery, and it is set by how many sequences share each decode step (the batch size). You care about p99 TPOT because a single stalled step freezes *every* stream in the batch.
- **End-to-end latency** — the whole request: `TTFT + (output_tokens - 1) × TPOT`. For a 500-token answer at 40 ms TPOT that is roughly 20 seconds, most of it decode. End-to-end matters for non-streaming APIs, agent tool-calls that block on a full response, and any downstream deadline.

The mistake that quietly wrecks SLO management is applying one of these numbers to all traffic. A user typing in a chat box and a nightly job summarizing ten million documents are not the same customer with the same needs, and pretending they are forces you to either over-provision for the batch job's volume or miss the chat user's tail. The fix is **request classes**: partition traffic into a small number of classes, each with its own SLO and its own behavior under overload. The matrix below shows the four classes that cover almost every production service.

![Matrix of four request classes — interactive chat, agent tool-call, batch offline, and embeddings — against TTFT SLO, TPOT SLO, priority, and overload action columns, showing chat and embeddings get the tightest targets and highest priority](/imgs/blogs/high-concurrency-slo-management-2.webp)

Written out with the extra detail a config file needs, the contract looks like this:

| Request class | Example workload | TTFT SLO (p99) | TPOT SLO (p99) | End-to-end budget | Priority | Behavior under overload |
|---|---|---|---|---|---|---|
| Interactive chat | User typing in a chat UI | < 500 ms | < 50 ms | soft (streamed) | 0 (highest) | Protect; preempt lower classes to hold it |
| Agent / tool-call | Multi-step agent, function calling | < 1 s | < 80 ms | < 30 s hard | 1 (medium) | Queue briefly, then degrade or shed |
| Batch / offline | Bulk summarization, evals | none | none | deadline (e.g. 1 h) | 2 (lowest) | Preempt first; run on leftover capacity |
| Embeddings | RAG retrieval, semantic search | < 20 ms | n/a (single pass) | < 20 ms | 0 (highest) | Route to a separate pool |

Two design notes fall straight out of this table. First, **embeddings are not text generation** and should almost never share an engine with chat — they are a single forward pass with no decode loop, their SLO is an order of magnitude tighter, and mixing them into a decode batch means a 20 ms embedding request waits behind a 20-second chat generation. Give them their own replica set. Second, the "behavior under overload" column is the part everyone forgets to specify, and it is the most important column in the table: it is the pre-authorized decision the system executes when it cannot serve everyone. If you have not decided in advance that batch jobs get preempted before chat, the system will make that decision for you at 3 a.m., randomly, and badly.

A subtle but load-bearing point: **an SLO is a distributional promise, not a per-request guarantee.** "TTFT p99 < 500 ms" means *at most 1% of requests may exceed 500 ms* — it explicitly budgets for a 1% miss rate. This is not a loophole; it is the entire basis of capacity planning. A promise that *every* request meets 500 ms (a p100 SLO) is uncostable, because a single 30,000-token prompt or a GPU hiccup blows it, and defending against that worst case means provisioning for a load you will see once a year. Choosing p95 versus p99 versus p99.9 is choosing how much of the tail you are willing to pay to cover, and as the next sections show, each additional nine costs real capacity.

## 2. Goodput: the metric you are actually paid to maximize

Here is the definition to tattoo on the inside of your eyelids:

$$\text{goodput} = \frac{\text{number of requests that complete AND meet all their SLOs}}{\text{unit time}}$$

Throughput counts every request the engine finishes. Goodput counts only the ones that finished *on time*. The difference is everything. You can write it as a product:

$$\text{goodput} = \lambda_{\text{offered}} \times f_{\text{admit}} \times f_{\text{SLO} \mid \text{admit}}$$

where $\lambda_{\text{offered}}$ is the offered request rate, $f_{\text{admit}}$ is the fraction you admit (the rest are shed), and $f_{\text{SLO} \mid \text{admit}}$ is the fraction of admitted requests that meet their SLO. This factorization is the whole strategy in one line. A naive system sets $f_{\text{admit}} = 1$ (admit everything) and prays; under overload $f_{\text{SLO} \mid \text{admit}}$ collapses toward zero and goodput craters even though throughput is maxed. An SLO-aware system deliberately drops $f_{\text{admit}}$ below 1 — sheds the excess — specifically to keep $f_{\text{SLO} \mid \text{admit}}$ near 1, and the product comes out *higher*. You throw away some requests to save the rest, and you end up serving more good requests in total.

The DistServe paper (Zhong et al., OSDI 2024) formalizes the capacity flavor of this metric and makes it the explicit optimization objective: **goodput is the maximum per-GPU request rate sustainable while meeting an SLO attainment target** — for example, "90% of requests meet both their TTFT and TPOT SLOs." Optimizing for that number instead of raw tokens/second is what lets them report up to 4.48× more goodput, or support up to 10.2× tighter SLOs at the same load, versus a throughput-tuned baseline. The lesson generalizes far beyond their specific disaggregation technique: **if your dashboards show throughput and GPU utilization but not goodput and SLO attainment, you are flying blind toward the exact failure at the top of this post.**

The figure below contrasts the two operating philosophies on the same hardware under the same overload. The unmanaged server on the left is tuned for maximum tokens/second; the SLO-aware server on the right is tuned for goodput.

![Before-and-after comparison: an unmanaged server with max_num_seqs 256 admitting everything reaches TTFT p99 of 8.5 seconds and 38 percent SLO attainment for 11 good requests per second, versus an SLO-aware server with a capped batch and shedding holding 96 percent attainment for 21 good requests per second](/imgs/blogs/high-concurrency-slo-management-3.webp)

#### Worked example: goodput arithmetic under a traffic surge

Suppose one replica can comfortably serve **18 req/s** within SLO, and a surge pushes offered load to **30 req/s**.

- *Unmanaged (admit everything).* All 30 req/s enter. The engine is now loaded to $\rho \approx 1.67$ — beyond capacity — so the queue grows without bound and TTFT climbs into the multi-second range. Suppose 38% of completed requests still sneak in under their SLO before the queue wait dominates. Goodput $= 30 \times 1.0 \times 0.38 = 11.4$ good req/s. Worse, every one of the 30 req/s consumed GPU time, so you paid full price for 11.4 useful req/s.
- *SLO-aware (cap + shed).* The admission controller caps in-flight work so the engine runs at $\rho \approx 0.85$, admitting ~22 req/s and shedding ~8 req/s with an HTTP 429. Because the engine is no longer overloaded, 96% of admitted requests meet their SLO. Goodput $= 22 \times 1.0 \times 0.96 = 21.1$ good req/s.

Same hardware, same surge. The managed system delivers **1.85× the goodput** and, as a bonus, the 8 req/s it shed got an immediate, honest 429 with a `Retry-After` hint instead of a 9-second hang followed by a timeout. Shedding is not giving up — it is refusing to convert a capacity problem into a latency catastrophe for everyone.

#### The shape of the goodput curve

It helps to be precise about what "goodput plateaus while throughput collapses" means as a curve. Let $\lambda$ be the offered load and $C$ the within-SLO capacity of a replica — the largest arrival rate at which admitted requests still meet their SLO. For an SLO-aware server, goodput as a function of offered load is approximately:

$$G(\lambda) = \min(\lambda,\ C)$$

It tracks the diagonal $y = \lambda$ while there is spare capacity, then flattens at $C$ once admission control starts shedding the excess. The flat top is not a failure; it is the design working. For an *unmanaged* server the curve is worse than flat — it rises to a peak near $C$ and then *declines*, because past capacity the growing queue pushes admitted requests through their SLO faster than new ones complete, so $f_{\text{SLO} \mid \text{admit}}$ falls faster than $\lambda$ rises. The entire job of the stack in Section 6 is to convert the second curve into the first: replace the post-peak decline with a plateau. Section 11 measures exactly this curve on real hardware.

#### Worked example: what does an extra nine of attainment cost?

Goodput is always defined against an *SLO attainment target* — the fraction of admitted requests you promise to serve on time — and that target is a business choice with a direct capacity price. Take the $\mu = 25$ req/s replica and 500 ms p99 TTFT SLO used throughout Section 8. Promising 99% attainment lets you run up to $\rho = 0.63$, for a within-SLO capacity of about ${16}$ req/s. Now promise 99.9% instead. Covering the deeper tail slice pulls a larger constant into the M/M/1 tail formula — $-\ln(0.001) = 6.9$ replaces $-\ln(0.01) = 4.6$ — so holding the same 500 ms bound forces $(1-\rho)$ up by a factor of ${6.9/4.6 = 1.5}$, dropping the ceiling to $\rho = 0.45$ and the capacity to about ${11}$ req/s. That single extra nine costs roughly **30% of usable capacity**. Each nine you add to the attainment target is paid for in idle silicon you provision but deliberately never fill, which is why "four nines of latency" should trigger a budget conversation, not a nod.

## 3. The fundamental tension: batch size buys throughput and spends latency

Why can't you just make the batch enormous and serve everyone at once? Because LLM decode has a physics problem, and understanding it is the key to picking your operating point. Let me build it from the roofline.

During **decode**, the engine generates one token per sequence per step. For a batch of $B$ sequences it does roughly $2 P B$ floating-point operations per step (where $P$ is the parameter count and the factor of 2 is one multiply-add per parameter per token), but it reads the model weights from HBM only *once* per step — about $2P$ bytes for a BF16 model. The **arithmetic intensity** (FLOPs per byte moved) is therefore approximately:

$$\text{AI}_{\text{decode}} \approx \frac{2 P B}{2 P} = B \quad \text{(FLOPs per byte, ignoring KV reads)}$$

The GPU's **roofline ridge point** — the arithmetic intensity at which it flips from memory-bandwidth-bound to compute-bound — is $\text{peak FLOP/s} \div \text{HBM bandwidth}$. For an H100 SXM (~990 BF16 TFLOP/s dense, ~3.35 TB/s HBM3) that is roughly $990 / 3.35 \approx 295$ FLOP/byte. So:

- **Below the ridge ($B \lesssim 250$–300)**, decode is *memory-bound*. The step is spent reading weights, and those weights are read exactly once no matter how many sequences ride along. Adding sequences to the batch is nearly *free* for step latency and increases tokens/second almost linearly. This is the "batching is a free lunch" regime.
- **Above the ridge**, decode is *compute-bound*. Now step time grows roughly linearly with $B$, so every extra sequence directly inflates TPOT for all of them. Throughput has flattened; you are just adding latency.

That is the clean story. The messy correction is the **KV cache**: each step also reads every sequence's accumulated key/value cache, which grows with batch × context length. For long contexts and large batches the KV read, not the weight read, dominates, and step time starts creeping up well before the naive ridge point. The net shape is the same — throughput is concave and saturating in $B$, latency is monotonically increasing in $B$ — and goodput, being throughput gated by the SLO, peaks at a *moderate* batch, not the largest one. The grid below shows the three regions.

![Three-by-three grid of batch-size operating regions: batch 8 gives 900 tokens per second with 22 ms TPOT but low goodput, batch 64 gives 3.6k tokens per second at 42 ms TPOT for maximum goodput, and batch 256 gives 4.2k tokens per second but 95 ms TPOT that blows the SLO and drops goodput](/imgs/blogs/high-concurrency-slo-management-4.webp)

#### Worked example: TPOT versus batch size on one H100

Take Llama-3-8B in BF16 on a single H100, decode phase, ~2k-token average context, with a **TPOT p99 SLO of 50 ms**. Illustrative but roofline-consistent numbers:

| Batch size | Throughput (tok/s) | TPOT p99 | Meets 50 ms SLO? | Effective goodput |
|---|---|---|---|---|
| 8 | ~900 | 22 ms | Yes | Low — GPU 30% utilized, wasting the box |
| 32 | ~2,400 | 34 ms | Yes | Good |
| **64** | **~3,600** | **42 ms** | **Yes** | **Maximum — best goodput per GPU** |
| 128 | ~4,000 | 63 ms | **No** | Collapsing — most tokens now arrive late |
| 256 | ~4,200 | 95 ms | **No** | Poor — 4% more throughput, 2× the TPOT |

Going from batch 64 to batch 256 buys you a *17% throughput gain* and costs you a *2.3× TPOT regression* that pushes you clean through the SLO. Every token past the SLO boundary is throughput that does not count as goodput. The throughput-maximizing batch (256) and the goodput-maximizing batch (64) are different numbers, and the whole discipline of SLO management is operating at the second one, not the first. The knobs in the next section are how you pin the engine to batch 64 instead of letting it drift to 256 under load.

#### Goodput as a function of batch size

The two curves — throughput rising and saturating in $B$, TPOT rising monotonically in $B$ — combine into a goodput function with an interior peak. Write decode throughput as a saturating function $T(B)$ and per-token latency as $L(B)$, increasing. Goodput counts only the tokens that arrive within the TPOT budget $L_{\text{SLO}}$, so:

$$G(B) = T(B) \cdot \mathbf{1}\!\left[\,L(B) \le L_{\text{SLO}}\,\right]$$

That indicator is a cliff, not a slope. As long as $L(B) \le L_{\text{SLO}}$, goodput equals throughput and you want the *largest* batch you can get away with; the instant $B$ crosses the point where $L(B) = L_{\text{SLO}}$, the indicator flips to zero and goodput falls off a shelf. The goodput-maximizing batch is therefore exactly the largest $B$ whose TPOT still clears the SLO — batch 64 in the table above, where 42 ms sits under the 50 ms budget, and not batch 128, where 63 ms has already dropped off the shelf. Your batch cap should sit at that boundary with a small safety margin, because the boundary itself drifts: a longer average context or a burst of concurrent prefills pushes $L(B)$ up at fixed $B$, so a batch that cleared 42 ms at 2k context can breach 50 ms at 6k context.

That drift is the KV-cache correction made concrete. The decode step also reads every sequence's KV cache, which grows with batch × context. For Llama-3-8B with grouped-query attention at 2k context, the KV cache is on the order of 0.25 GB per sequence in BF16; at 64 sequences that is roughly 16 GB of KV reads per step *on top of* the ~16 GB weight read — the KV traffic has already matched the weight traffic at this operating point, so the "weights are read once, batching is free" rule of thumb is only half the story here. Double the context to 4k and the KV read doubles while the weight read stays fixed, so step time climbs even though $B$ has not changed. This is why a TPOT budget must be validated at your *p99 context length*, not your median: the tail of the context distribution, not the average, is what sets the safe batch cap, and a batch tuned on short prompts will quietly breach the moment a wave of long-context requests arrives together.

## 4. The concurrency knobs and what each does to your SLO

vLLM (and every serious engine) exposes a handful of knobs that set the operating point on that curve. Getting them right is 80% of holding an SLO on a single replica. The four that matter:

- **`max_num_seqs`** — the hard cap on how many sequences share a decode batch. This is your batch-size ceiling and your single most important latency knob. The default (256) is tuned for throughput benchmarks; for a tight TPOT SLO you set it to the goodput-maximizing value from your own sweep (64 in the example above). Capping it also caps in-flight concurrency, which is your admission budget (Little's Law, Section 8).
- **`max_num_batched_tokens`** — the total token budget the scheduler may pack into one step, counting both prefill and decode tokens. This governs how much prefill work can happen in a single step. A *large* value lets a big prompt prefill in one giant pass — great for prefill throughput, terrible for the decode streams that stall while it runs. A *small* value forces prefill to happen in small chunks interleaved with decode, smoothing TPOT.
- **`enable_chunked_prefill`** — the switch that lets a long prefill be split across multiple steps so decode never stalls for the full prefill duration. This is the single most effective TPOT-protecting flag under mixed load; it is what stops one 28k-token prompt from freezing 60 concurrent chats (the exact incident in the [request scheduling](/blog/machine-learning/model-serving/request-scheduling-and-preemption) post). Sarathi-Serve calls this "stall-free batching," and it is why chunked prefill is on by default in modern vLLM.
- **`scheduling_policy`** — `fcfs` (first-come-first-served) or `priority`. In priority mode, requests carry an integer priority (lower = served first) and the scheduler admits and preempts by it. This is the hook the priority gateway in Section 7 drives.

The two token knobs interact, and getting the interaction right is what actually protects TPOT. `max_num_batched_tokens` is the per-step budget; with chunked prefill on, a prefill longer than the budget is sliced into budget-sized chunks that ride *alongside* decode tokens rather than displacing them. Set the budget too high — say 8192 — and a single 8k-token prompt prefills in one step that takes tens of milliseconds, and every decode stream in the batch stalls for that whole step, spiking their TPOT. Set it to 2048 and that same prompt spreads across four steps, each of which also carries the ongoing decodes, so no stream ever waits more than one chunk. The cost is a slightly longer TTFT for the big prompt (four steps instead of one) in exchange for a flat TPOT for the sixty streams it would otherwise have frozen — the right trade whenever decode smoothness is the SLO you are defending. A useful starting point is a budget of 2–4× `max_num_seqs`, tuned down until p99 TPOT under a mixed prefill/decode load stops spiking.

Here is an SLO-oriented engine configuration. Note that these values are chosen to hold a tail latency target, *not* to win a throughput benchmark:

```python
from vllm import AsyncEngineArgs, AsyncLLMEngine

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",

    # --- The batch<->latency operating point (Section 3) ---
    # Cap the decode batch at the goodput-maximizing size from your sweep,
    # NOT the throughput-maximizing default of 256.
    max_num_seqs=64,

    # Per-step token budget. Small enough that a big prefill is chunked
    # instead of monopolizing a step and stalling every decode stream.
    max_num_batched_tokens=2048,

    # --- TPOT protection under mixed prefill/decode load ---
    enable_chunked_prefill=True,      # interleave prefill chunks with decode
    long_prefill_token_threshold=512, # split any prefill longer than this

    # --- Honor per-request priority from the gateway (Section 7) ---
    scheduling_policy="priority",     # "fcfs" | "priority"; lower value first
    preemption_mode="recompute",      # recompute KV on preempt (vs "swap")

    # --- Memory headroom so decode never OOMs mid-batch ---
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
```

When you enqueue a request, you pass the class priority the gateway assigned, so the engine's scheduler can preempt best-effort work for latency-critical work:

```python
from vllm import SamplingParams

# priority: 0 = interactive chat, 1 = agent, 2 = batch (lowest).
# In priority mode, vLLM admits and preempts by this value.
async for out in engine.generate(
    prompt=prompt,
    sampling_params=SamplingParams(max_tokens=512, temperature=0.7),
    request_id=request_id,
    priority=request_priority,   # comes from the classifier in Section 7
):
    yield out.outputs[0].text
```

There is no free lunch here: capping `max_num_seqs` at 64 means one replica tops out at a lower raw throughput than the same box at 256. You are trading peak throughput for tail latency — a deliberate move along the SLO triangle. You recover the lost throughput not by uncapping the batch (which would reintroduce the tail) but by adding replicas (Section 10), which is cheaper than the reputational cost of a 95 ms TPOT.

## 5. Why an unmanaged system does not degrade — it collapses

The scariest property of a queuing system is that it does not slow down smoothly as you load it. It is fine, fine, fine — and then it falls off a cliff. To defend against a cliff you have to know exactly where the edge is, and for that we need a little queuing theory. I promise it pays for itself immediately.

Model the server as a single queue with **arrival rate** $\lambda$ (requests/second) and **service rate** $\mu$ (the rate at which it can complete them). The **utilization** is $\rho = \lambda / \mu$ — the fraction of capacity you are using. For the simplest useful model, M/M/1 (Poisson arrivals, exponential service, one server), the mean number of requests in the system is:

$$L = \frac{\rho}{1 - \rho}$$

Stare at that denominator. At $\rho = 0.5$ you have 1 request in the system on average. At $\rho = 0.9$ you have 9. At $\rho = 0.99$ you have 99. The mean sojourn time (wait + service) is $W = \frac{1}{\mu (1 - \rho)}$, and — crucially for SLOs — in M/M/1 the sojourn time is *exponentially distributed*, so its 99th percentile is:

$$T_{p99} = \frac{-\ln(1 - 0.99)}{\mu (1 - \rho)} = \frac{4.605}{\mu (1 - \rho)}$$

That $(1 - \rho)$ in the denominator is the cliff. As $\rho \to 1$, latency $\to \infty$. It is not gradual.

#### Worked example: how p99 explodes as offered load approaches capacity

Take a replica with service rate $\mu = 25$ req/s. Here is what the sojourn-time distribution does as you push $\rho$ up:

| Utilization $\rho$ | Offered $\lambda$ | Mean in system $L$ | p50 sojourn | p99 sojourn |
|---|---|---|---|---|
| 0.50 | 12.5 req/s | 1.0 | 55 ms | 368 ms |
| 0.70 | 17.5 req/s | 2.3 | 92 ms | 614 ms |
| 0.90 | 22.5 req/s | 9.0 | 277 ms | 1.84 s |
| 0.95 | 23.75 req/s | 19.0 | 554 ms | 3.68 s |
| 0.99 | 24.75 req/s | 99.0 | 2.77 s | **18.4 s** |

Read the p99 column top to bottom. Between $\rho = 0.5$ and $\rho = 0.9$ — a 45% increase in offered load — p99 goes from 368 ms to 1.84 s, a **5× regression**. Between 0.9 and 0.99, another 10% of load multiplies p99 by another 10×. This is why the system was "fine at 9:00 and dead at 9:05": the surge did not double your latency, it moved you along a curve whose slope goes vertical near capacity. A 500 ms p99 SLO is comfortably met at $\rho = 0.5$ and violated by 3.7× at $\rho = 0.95$, for a difference of *less than twice the load*.

Now the part that makes it worse in reality. M/M/1 assumes exponential service times. LLM service times are *far* more variable than that: output lengths span from a 3-token "yes" to a 4,000-token essay, and prefill costs span from a one-line prompt to a 30k-token document. **Kingman's formula** (the VUT approximation for a general G/G/1 queue) captures this:

$$W_q \approx \underbrace{\left(\frac{\rho}{1 - \rho}\right)}_{\text{Utilization}} \cdot \underbrace{\left(\frac{c_a^2 + c_s^2}{2}\right)}_{\text{Variability}} \cdot \underbrace{\frac{1}{\mu}}_{\text{service Time}}$$

where $c_a^2$ and $c_s^2$ are the squared coefficients of variation (variance ÷ mean²) of inter-arrival and service times. For M/M/1 both equal 1 and the middle term is 1. For LLM traffic, $c_s^2$ is easily 3–5 because of that output-length spread, so the queueing delay is **2–3× worse than M/M/1 predicts at the same utilization.** Variance is a first-class enemy: two workloads at identical $\rho$ can have wildly different tails, and the high-variance one blows its SLO first. This is a concrete argument for splitting long-context and short-context traffic into separate classes — you are reducing $c_s^2$ within each class.

There is one lever that fights the cliff without buying hardware: **consolidation**. A pool of many servers behind *one* queue tolerates higher utilization at the same tail than the same servers split into many single-server queues. This is the M/M/c result, and it is counter-intuitive enough to deserve a worked example.

#### Worked example: why one big pool beats many small ones

Compare two ways to deploy 4 replicas against 80 req/s of chat traffic, each replica serving $\mu = 25$ req/s. In *siloed* mode a load balancer hashes each request to one replica, so each is an independent M/M/1 queue at $\rho = 20/25 = 0.8$. In *pooled* mode all four share one queue and any idle replica takes the next request — an M/M/c queue with $c = 4$ at the same aggregate $\rho = 0.8$. The pooled queue's probability that an arrival has to wait at all (the Erlang-C probability) is far lower, because a request only queues when *all four* replicas are busy, not when its one assigned replica happens to be busy. The upshot: the pooled configuration holds a p99 wait several times lower than the siloed one at identical utilization, or equivalently sustains higher utilization at the same p99. The rule that falls out: **do not shard a latency-critical pool more finely than you must.** Every time you split one pool of 8 replicas into four pools of 2 for isolation, you hand back tail headroom that you then buy back with more replicas. Isolation (Section 9) and tail-smoothing (this section) pull in opposite directions, and the clean resolution is fair queuing *within* one shared pool rather than physical partitioning.

The cliff would be survivable if load stopped at the edge. It does not, because of **retry storms**. When latency crosses the client's timeout, the client retries. Retries add load. Added load raises $\rho$. Higher $\rho$ raises latency. Higher latency triggers more timeouts and more retries. This is a positive feedback loop — a **metastable failure** (Bronson et al., HotOS 2021): the system has a stable healthy state and a stable *collapsed* state, and a big enough shove pushes it over the ridge into collapse, where it *stays* even after the original surge subsides, because the retry traffic is now self-sustaining. Restarting pods does not help; the retries just re-collapse the fresh pods. The only exits are shedding load (break the loop by refusing work) and forcing clients to back off. The timeline below shows the full lifecycle — the shove, the collapse, and the two interventions that pull it back.

![Timeline of an overload event: normal load at rho 0.6, a surge to 28 requests per second, saturation with a 40-deep queue and 6-second TTFT, admission shedding with 429s pulling rho to 0.8, autoscaling adding four replicas, and recovery to 96 percent attainment](/imgs/blogs/high-concurrency-slo-management-5.webp)

The single most important operational takeaway from this section: **you must run below the knee of the curve, not at it.** The GPU sitting at 85% instead of 99% is not waste — it is the headroom that keeps you off the cliff. Section 8 turns that into a number.

## 6. The layered SLO defense

There is no single knob that holds an SLO under high concurrency. Protection is a *stack*, where each layer either removes load or reorders work so that the layer below it never has to break its promise. If you skip a layer, its job lands on the layer below, which is not designed for it, and that is where breaches leak through. The stack, top to bottom:

![Layered stack of six SLO-defense layers from top to bottom: SLO definition with three classes and p99 targets, routing and classification into two priority lanes, admission control capping 64 in-flight with 429s, an SLO-aware scheduler doing EDF and preemption, SLO autoscaling targeting rho below 0.7, and observability on goodput and p99 gauges](/imgs/blogs/high-concurrency-slo-management-6.webp)

Reading the layers as a defense-in-depth chain:

1. **SLO definition** (Section 1) — the per-class contract. Everything below references it. Without it, no other layer knows what "meeting the SLO" means.
2. **Routing and classification** — at ingress, tag each request with its class and priority, and route embeddings to their own pool. This is cheap and it is where the two-lane split begins.
3. **Admission control** (Section 8) — the outermost load valve. When the system is at capacity, this layer refuses new best-effort work with a 429 so the queue never grows past the point where the SLO is defensible.
4. **SLO-aware scheduling** (Section 7) — inside the engine, order and preempt work so latency-critical requests jump the queue and best-effort requests yield.
5. **SLO autoscaling** (Section 10) — add capacity when the leading indicators say a breach is coming, so admission control has to shed less often.
6. **Observability** — measure goodput, SLO attainment, and p99 continuously, because you cannot manage what you cannot see, and every layer above tunes against these numbers.

The rest of the post walks down this stack. Sections 7 through 10 each implement one layer.

## 7. SLO-aware scheduling: priority, deadlines, and preemption

Inside the engine, on every decode step, the scheduler picks which sequences get compute. The *policy* it uses is where SLO-awareness lives. Four policies, in increasing order of how well they protect a tail latency target:

| Policy | How it orders work | Protects latency-critical? | Fairness | Head-of-line blocking | When to use |
|---|---|---|---|---|---|
| **FCFS** | Arrival order | No — a batch job at the head blocks chat | None | Severe | Single-class traffic only |
| **Priority** | By static class priority | Yes — chat preempts batch | Poor across tenants | Within a class | Mixed classes, one tenant |
| **Deadline (EDF)** | By slack = (deadline − now) | Yes — and adapts to remaining budget | Poor across tenants | Minimal | Hard per-request deadlines |
| **Fair (VTC)** | By tokens already served per client | Yes, per class | Strong | Minimal | Multi-tenant (Section 9) |

**Priority scheduling** is the workhorse. You assign class 0 to chat, 1 to agents, 2 to batch, and the scheduler serves lower numbers first and preempts higher numbers when memory or compute runs short. **Earliest-deadline-first (EDF)** is a refinement: instead of a static priority, order by *slack* — how much of the request's latency budget remains. A chat request with 100 ms left beats a chat request with 400 ms left, even though both are class 0. EDF squeezes more goodput out of a loaded system because it never wastes urgency on a request that has plenty of budget, but it needs an accurate deadline per request, which means the gateway has to stamp one on arrival.

**Preemption** is what makes priority real. When a class-0 request needs a KV-cache slot and none is free, the scheduler evicts a class-2 sequence's KV cache to make room. vLLM does this two ways: `recompute` (drop the KV cache and regenerate it from the prompt when the request resumes — cheap on memory, costs a re-prefill) or `swap` (copy the KV cache to CPU RAM and back — costs PCIe bandwidth, saves the recompute). The [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) post derives the exact cost math; for SLO purposes the key fact is that preemption lets you *guarantee* a latency-critical request never waits behind a best-effort one, at the cost of some wasted work on the preempted request — which is exactly the trade you want, because the best-effort request has no TTFT SLO to miss.

#### Worked example: EDF versus static priority on a loaded engine

Static priority orders class 0 ahead of class 1 ahead of class 2, but *within* class 0 it falls back to FCFS, and that fallback is where it leaves goodput on the table. Suppose three chat requests are waiting, all class 0, all with a 500 ms TTFT budget: request A arrived 420 ms ago, B 120 ms ago, C 30 ms ago, so their remaining slack is 80 ms, 380 ms, and 470 ms respectively. Now flip the arrival order so the most urgent request is also the newest: say the request with only 80 ms of slack left arrived most recently. FCFS dispatches in arrival order and lets that near-deadline request wait behind two others with hundreds of milliseconds to spare — and under a load spike where each prefill takes 150 ms, the request with 80 ms of budget waits 300 ms and misses, while the two with ample slack are served comfortably early. EDF orders by slack instead: it runs the 80 ms request first, then the 380 ms, then the 470 ms, and saves all three. The rule EDF encodes is that **urgency is remaining budget, not arrival time**, and the two diverge exactly when a slow-arriving request is closer to its deadline than requests that have been queued longer. The cost is real but modest: the gateway must stamp an absolute deadline on every request at ingress (the `deadline` field in the gateway below), and the scheduler must re-sort by slack each step — negligible next to a decode step, and worth it wherever budgets are tight enough that arrival-order ties actually cost misses.

The gateway is where classification and deadline-stamping happen, upstream of the engine. Here is a priority- and deadline-aware FastAPI gateway that classifies each request, assigns a deadline, and forwards to vLLM with the right priority. It fronts the engine so the engine only ever sees pre-classified, pre-prioritized work:

```python
import time
import asyncio
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()
VLLM_URL = "http://vllm-engine:8000/v1/chat/completions"

# Per-class SLO contract from Section 1 (TTFT budget in seconds, priority int).
CLASS_SLO = {
    "chat":   {"priority": 0, "ttft_budget": 0.5},
    "agent":  {"priority": 1, "ttft_budget": 1.0},
    "batch":  {"priority": 2, "ttft_budget": 999.0},
}

def classify(req: Request) -> str:
    # Real systems classify on API-key tier, an explicit header, or a
    # lightweight model. Here: an explicit client-supplied class header,
    # defaulting to the safest (most expensive) class if absent.
    return req.headers.get("x-request-class", "chat")

@dataclass(order=True)
class Deadline:
    # asyncio.PriorityQueue orders by this tuple: (priority, absolute_deadline).
    # Lower priority value first; within a class, earliest deadline first (EDF).
    priority: int
    deadline: float
    payload: dict = field(compare=False)

work_queue: "asyncio.PriorityQueue[Deadline]" = asyncio.PriorityQueue()

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    cls = classify(request)
    slo = CLASS_SLO[cls]
    now = time.monotonic()
    item = Deadline(
        priority=slo["priority"],
        deadline=now + slo["ttft_budget"],
        payload={"body": body, "class": cls, "enqueued": now},
    )
    await work_queue.put(item)                 # ordered by (priority, deadline)
    return StreamingResponse(
        _dispatch(item), media_type="text/event-stream"
    )

async def _dispatch(item: Deadline):
    # Pull the highest-priority / earliest-deadline item and forward it to vLLM,
    # passing the class priority so the ENGINE scheduler also honors it.
    picked = await work_queue.get()
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", VLLM_URL,
            json={**picked.payload["body"], "priority": picked.priority},
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
```

This gateway does two jobs the engine cannot do alone: it decides the *class* (the engine has no idea whether a request is a paying chat user or a background eval), and it holds the priority queue *in front of* the engine so that when the engine is saturated, the queue in the gateway — not the engine's internal one — is where excess work waits, which is exactly where the admission controller in the next section can see and shed it.

## 8. Admission control and graceful load shedding

Sections 5 taught the lesson; this section acts on it. **You cannot admit more work than you can serve within SLO, so you must refuse the excess — early, cheaply, and honestly.** That is admission control. Done right, it converts the vertical latency cliff into a flat plateau: goodput rises with offered load until you hit capacity, then stays flat as excess load bounces off the 429 wall, instead of collapsing.

How much can you admit? **Little's Law** gives the exact ceiling. It states that the average number of requests in a system equals the arrival rate times the average time in system:

$$L = \lambda \cdot W$$

Turn it around: if your per-request SLO is a total latency of $W_{\text{SLO}}$ and your engine's within-SLO service rate is $\mu$, the maximum concurrency you can hold while meeting the SLO is $N = \mu \cdot W_{\text{SLO}}$. Cap in-flight requests at $N$ and you have a hard, principled admission limit. In practice you set $N \approx$ `max_num_seqs` plus a small queue, because that is the batch the engine can actually keep within TPOT. The decision the controller makes on every request is the tree below.

![Decision tree for load shedding: a request arrives classified as latency-critical or best-effort, the gate checks whether in-flight requests are under the Little's Law cap of 64, if yes it admits with a shallow queue, if no it is at capacity and latency-critical work preempts best-effort while best-effort is shed with a 429 and a 2-second Retry-After](/imgs/blogs/high-concurrency-slo-management-7.webp)

#### Worked example: how much headroom does a 99% TTFT SLO cost?

Reuse the $\mu = 25$ req/s replica and a **TTFT SLO of 500 ms at p99**. From the M/M/1 p99 formula in Section 5, the SLO holds only when:

$$T_{p99} = \frac{4.605}{\mu(1-\rho)} \le 0.5 \;\Rightarrow\; (1-\rho) \ge \frac{4.605}{25 \times 0.5} = 0.368 \;\Rightarrow\; \rho \le 0.632$$

So the maximum utilization at which you can *promise* p99 < 500 ms is **63%**, which means a maximum sustainable load of $0.632 \times 25 \approx 15.8$ req/s. To serve your peak of, say, 25 req/s within SLO you need $25 / 15.8 \approx 1.58\times$ the single-replica capacity — a **58% headroom tax** just to hold this one p99 target. Tighten the SLO to 300 ms and $\rho_{\max}$ drops to 0.39, demanding **2.6× headroom**. Account for the LLM variance premium from Kingman ($c_s^2 \approx 4$) and the real headroom is larger still. This is the number to bring to a capacity-planning meeting: **the p99 target directly sets your provisioning multiplier**, and "just run the GPUs hotter" is mathematically the same sentence as "abandon the SLO."

Here is a production-shaped admission controller: a token bucket for the sustained rate, a concurrency semaphore for the Little's-Law in-flight cap, class-aware shedding (best-effort sheds before latency-critical), and an honest `Retry-After` with jitter to prevent the retries from synchronizing into a thundering herd:

```python
import asyncio
import random
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Little's Law cap: max in-flight requests we can hold within SLO.
MAX_INFLIGHT = 64
# Token bucket: sustained admit rate + burst headroom.
RATE_PER_SEC = 18.0
BURST = 24

class AdmissionController:
    def __init__(self):
        self._sema = asyncio.Semaphore(MAX_INFLIGHT)
        self._tokens = float(BURST)
        self._last = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        self._tokens = min(BURST, self._tokens + (now - self._last) * RATE_PER_SEC)
        self._last = now

    def try_admit(self, request_class: str) -> tuple[bool, float]:
        """Returns (admitted, retry_after_seconds)."""
        self._refill()
        inflight = MAX_INFLIGHT - self._sema._value
        # At capacity: shed best-effort first; keep a reserve for chat.
        reserve = 8 if request_class != "chat" else 0
        if inflight >= MAX_INFLIGHT - reserve:
            return False, self._retry_after()
        if self._tokens < 1.0:
            return False, self._retry_after()
        self._tokens -= 1.0
        return True, 0.0

    def _retry_after(self) -> float:
        # Base backoff + jitter so shed clients do not retry in lockstep.
        return round(2.0 + random.uniform(0, 1.0), 2)

    async def slot(self):
        return self._sema  # acquire/release around the actual generation

admission = AdmissionController()

@app.post("/generate")
async def generate(request: Request):
    request_class = request.headers.get("x-request-class", "chat")
    ok, retry_after = admission.try_admit(request_class)
    if not ok:
        # Honest, immediate backpressure — NOT a 9-second hang then a timeout.
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"error": "overloaded", "retry_after_s": retry_after,
                     "class": request_class},
        )
    async with admission._sema:
        return await _forward_to_engine(request)  # streams from vLLM
```

Three properties make this correct rather than merely present. First, it **sheds the right requests**: the `reserve` keeps slots open for chat even at capacity, so a flood of batch requests can never starve interactive traffic — the admission layer enforces the priority contract *before* work even reaches the scheduler. Second, it **fails fast**: a shed request gets a 429 in single-digit milliseconds instead of joining a queue that will time it out in 9 seconds, which is both better UX and dramatically less wasted GPU. Third, the **jittered `Retry-After`** is the antidote to the retry storm from Section 5: it tells well-behaved clients exactly how long to back off and spreads their return over a window so they do not all stampede back at once. This last point is why load shedding and client-side exponential backoff are two halves of one mechanism — shedding without a backoff hint just moves the storm one hop upstream.

#### Worked example: the GPU you save by shedding early

Shedding is often resisted as "giving up revenue," so it is worth pricing the alternative. A request that will miss its SLO does not fail for free — it occupies a scheduler slot and KV-cache memory for its entire doomed lifetime, starving requests that *could* have met their SLO. Take the overloaded replica from Section 2 at 30 req/s offered against 18 req/s of capacity. The ~12 req/s of excess do not vanish under an admit-everything policy: each sits in the queue for seconds, holding memory, before it times out or completes late. If the average doomed request squats for 4 seconds of wall-clock queue time, that is $12 \times 4 = 48$ request-seconds of KV-cache pressure per second of overload — pressure that forces preemptions and re-prefills on the *good* requests, which is how the survivors' tail latency gets dragged down with the doomed. Shedding those 12 req/s at the door with a sub-millisecond 429 hands all of that memory and compute back to the requests that can still be saved. The 429 is not lost revenue; it is a refusal to spend GPU on an answer no one will wait for, and the spend it frees is what keeps the goodput plateau flat. A client that receives a fast, honest 429 with a `Retry-After` can also fail over to a second region, degrade to a smaller model, or surface a "try again in a moment" far more gracefully than one left hanging until a 30-second gateway timeout fires and returns nothing.

## 9. Multi-tenant fairness under contention

The moment your service has more than one customer sharing a pool, a new failure mode appears: one tenant's traffic spike starves everyone else's SLO, even though the *aggregate* load is fine. Priority scheduling does not fix this — all your chat tenants are class 0, and FCFS within a class means whichever tenant sends the most requests wins the most slots. That is not fairness; that is a reward for aggression.

The clean solution is **weighted fair queuing** adapted to tokens. The "Fairness in Serving Large Language Models" work (Sheng et al., OSDI 2024) introduces the **Virtual Token Counter (VTC)**: track the cumulative number of tokens the system has served each client, and when choosing what to run next, prefer the client with the *fewest* tokens served so far (weighted by each client's share). Because it counts tokens — the actual unit of GPU work — rather than requests, it is robust to a tenant gaming the system by sending one enormous request instead of many small ones. VTC comes with a provable fairness bound: no backlogged client falls more than a constant number of tokens behind its fair share. A minimal version of the idea:

```python
import heapq
from collections import defaultdict

class FairScheduler:
    """Virtual-token-counter fair queuing across tenants (VTC-style).
    Serves the backlogged tenant that has received the fewest weighted tokens."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights                      # tenant -> share (e.g. 1.0, 2.0)
        self.served = defaultdict(float)            # tenant -> weighted tokens served
        self.backlog = defaultdict(list)            # tenant -> pending request ids

    def enqueue(self, tenant: str, request_id: str):
        self.backlog[tenant].append(request_id)

    def pick_next(self) -> str | None:
        # Among tenants with pending work, pick the one furthest behind its share.
        candidates = [(self.served[t] / self.weights[t], t)
                      for t in self.backlog if self.backlog[t]]
        if not candidates:
            return None
        _, tenant = min(candidates)
        return self.backlog[tenant].pop(0)

    def on_token_generated(self, tenant: str, n_tokens: int = 1):
        # Charge the tenant for GPU work actually done — the fairness invariant.
        self.served[tenant] += n_tokens
```

Fairness is a *fourth* corner of the serving trade triangle, orthogonal to latency, throughput, and cost. You can hold everyone's SLO on average and still have a badly unfair system where a whale tenant's SLO is met and a small tenant's is trampled. VTC-style fairness is what keeps a multi-tenant platform honest, and it composes cleanly with the priority classes above it: fairness operates *within* a priority class, deciding which class-0 tenant runs next, while priority decides which class runs at all.

## 10. SLO-aware autoscaling: scale on the signal that predicts the breach

Admission control protects the SLO by *refusing* work; autoscaling protects it by *adding capacity* so you have to refuse less. The two are partners — admission is the fast reflex (milliseconds), autoscaling is the slow muscle (seconds to minutes) — and the whole art of autoscaling is picking the signal you scale on. Get this wrong and you scale too late, after the breach, or never, because your metric never fired.

The near-universal default — scale on CPU or GPU utilization — is exactly wrong for LLM serving. The matrix below is the argument.

![Matrix comparing five autoscaling signals against timing, whether they protect p99, gameability, and a verdict: CPU utilization is irrelevant and should not be used, GPU utilization lags and saturates for a weak signal, queue depth leads and is recommended, TTFT p99 is a strong direct signal, and SLO attainment is best paired with queue depth](/imgs/blogs/high-concurrency-slo-management-8.webp)

The reasoning, signal by signal:

- **CPU utilization** is irrelevant — the work is on the GPU, and the CPU can be idle while the GPU melts. Never use it for LLM serving.
- **GPU utilization** is the seductive trap. It *saturates at 100% long before the SLO breaks* and stays pinned there whether you are at $\rho = 0.9$ (healthy) or $\rho = 1.5$ (collapsing). A metric that reads the same in a healthy system and a dying one carries no information about the thing you care about. It lags and it is uninformative — a weak signal at best.
- **Queue depth** (`vllm:num_requests_waiting`) *leads* the breach: the queue starts growing the instant $\rho$ crosses 1, before latency has fully blown up, giving the autoscaler a head start. It is hard to game and directly reflects the pressure that produces the SLO miss. This is the recommended primary signal.
- **TTFT p99** measured over a sliding window is the most direct signal — it *is* the SLO — but it lags slightly (you only know p99 rose after enough requests were slow). Strong, best used as a scale-up confirmation.
- **SLO attainment** (the fraction meeting SLO) is the goodput signal itself; it lags but is unfakeable. Best *paired* with queue depth: queue depth triggers early scale-up, attainment confirms.

Here is the machinery. First a Prometheus recording rule that turns raw histograms into the SLO signals, then a KEDA `ScaledObject` that scales the deployment on queue depth with an aggressive scale-up and a conservative scale-down:

```yaml
# prometheus-rules.yaml — compute the SLO signals from vLLM histograms.
groups:
  - name: llm-slo
    rules:
      # p99 TTFT over a 2-minute window, in seconds.
      - record: llm:ttft_p99
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[2m])))

      # SLO attainment: fraction of requests under the 0.5s TTFT budget.
      - record: llm:ttft_slo_attainment
        expr: |
          sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[2m]))
          /
          sum(rate(vllm:time_to_first_token_seconds_count[2m]))

      # Page when attainment falls below the 99% contract.
      - alert: SLOAttainmentBreached
        expr: llm:ttft_slo_attainment < 0.99
        for: 3m
        labels: {severity: page}
        annotations:
          summary: "TTFT SLO attainment {{ $value | humanizePercentage }} < 99%"
```

```yaml
# keda-scaledobject.yaml — scale on QUEUE DEPTH, not GPU utilization.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-slo-autoscaler
spec:
  scaleTargetRef:
    name: vllm-chat            # the Deployment serving the chat class
  minReplicaCount: 4           # p99 headroom floor — never scale below this
  maxReplicaCount: 40
  cooldownPeriod: 300          # slow scale-DOWN: avoid thrashing on dips
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 0    # fast scale-UP: react immediately
          policies:
            - type: Percent
              value: 100                    # allow doubling in one step
              periodSeconds: 30
        scaleDown:
          stabilizationWindowSeconds: 300   # conservative: 5-min window
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        # Target ~8 waiting requests per replica. Above this, the queue is
        # growing faster than replicas drain it -> add capacity NOW.
        query: sum(vllm:num_requests_waiting)
        threshold: "8"
```

Three details that separate this from a naive HPA. First, **`minReplicaCount: 4` is the headroom floor** — it encodes the "run below the knee" lesson from Section 5 as infrastructure, guaranteeing you never scale down into the danger zone even during a lull. Second, **scale-up is instant and scale-down is slow** (0-second vs 300-second stabilization): reacting fast to a surge protects the SLO, while reacting slowly to a dip avoids the thrash of tearing down a replica just before the next spike — and GPU pods are expensive to spin up, so a torn-down replica costs minutes of cold start. Third, the **cold-start reality** is the reason admission control is not optional: a new GPU replica takes 60–180 seconds to pull the image, load 16–140 GB of weights, and warm its caches, so during that window admission control is the *only* thing holding the line. Autoscaling handles the sustained shift; admission handles the surge in between. This is also why services that can afford it keep a small warm pool or scale predictively on a traffic forecast rather than purely reactively — the reactive loop is always one cold start behind the demand.

#### Predictive scaling: staying ahead of the cold start

For LLM serving the cold-start lag is brutal — 60–180 seconds to schedule a GPU node, pull a multi-gigabyte image, load weights, and warm the KV allocator — so a purely reactive loop spends that whole window shedding while the new replica boots. Two techniques buy the window back. The first is a **warm pool**: keep a few pre-scheduled, weights-loaded replicas parked just above the reactive floor, ready to take traffic the instant the trigger fires. The second is **predictive scaling**: production traffic is rarely random — it has a daily and weekly shape you can forecast — so you scale on the *forecast* for one cold-start horizon ahead, not on the current queue. A minimal pre-scaler that runs the reactive count against a lightweight forecast and takes the max:

```python
import math
import datetime as dt

def desired_replicas(queue_depth: int, current_rps: float,
                     per_replica_capacity: float = 15.0,
                     cold_start_s: int = 120) -> int:
    HEADROOM = 1.58                       # 1 / rho_max for the 500 ms p99 SLO

    # 1. Reactive term: serve current load with the run-below-the-knee
    #    headroom, plus enough to burn down the standing backlog.
    reactive = (current_rps * HEADROOM) / per_replica_capacity
    reactive += queue_depth / (per_replica_capacity * 2)

    # 2. Predictive term: forecast load one cold-start horizon ahead and
    #    pre-provision for it, so capacity is warm BEFORE the surge lands.
    horizon = dt.datetime.now() + dt.timedelta(seconds=cold_start_s)
    forecast_rps = predict_load(horizon)          # seasonal model / Prophet
    predictive = (forecast_rps * HEADROOM) / per_replica_capacity

    # Take the max so a forecast miss can only OVER-provision, never strand
    # the reactive loop behind a cold start.
    return max(4, math.ceil(max(reactive, predictive)))
```

The `max` is the safety property: a bad forecast can only cost you idle replicas, never an SLO breach, because the reactive term still fires on real queue depth. Prediction does not replace the reactive loop or admission control — it shrinks the window in which admission control is the *only* defense, which is precisely the window where breaches happen. The three mechanisms nest cleanly: admission control holds the milliseconds, warm pools and prediction cover the cold-start minutes, and reactive autoscaling handles the sustained hour.

## 11. Measuring it: goodput versus offered load

Everything above is theory until you measure it on your own hardware. The measurement that matters is not "what is our peak tokens/second" — it is **the goodput-versus-offered-load curve**: as you ramp the offered request rate, how many requests keep meeting their SLO? A well-managed system's curve rises linearly, then flattens at capacity (excess is shed). An unmanaged system's curve rises, peaks, and then *falls* as overload destroys attainment. Here is a load generator that measures exactly this — it ramps the rate, records per-request TTFT and TPOT from the SSE stream, and computes goodput as the rate of requests meeting both SLOs:

```python
import asyncio, time, statistics
import httpx

GATEWAY = "http://gateway:8080/generate"
TTFT_SLO, TPOT_SLO = 0.5, 0.05      # seconds: 500 ms TTFT, 50 ms TPOT

async def one_request(client, results):
    t0 = time.monotonic()
    first_token_t, tokens, last_t = None, 0, t0
    itls = []
    try:
        async with client.stream("POST", GATEWAY, json={"prompt": "Explain SLOs."},
                                 headers={"x-request-class": "chat"},
                                 timeout=30.0) as r:
            if r.status_code == 429:
                results.append(("shed", None, None)); return
            async for _ in r.aiter_lines():
                now = time.monotonic()
                if first_token_t is None:
                    first_token_t = now
                else:
                    itls.append(now - last_t)
                last_t = now; tokens += 1
    except (httpx.TimeoutException, httpx.HTTPError):
        results.append(("timeout", None, None)); return
    ttft = (first_token_t - t0) if first_token_t else None
    tpot = statistics.mean(itls) if itls else None
    results.append(("done", ttft, tpot))

async def ramp(rate_per_sec, duration_s):
    results = []
    async with httpx.AsyncClient() as client:
        tasks, interval = [], 1.0 / rate_per_sec
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            tasks.append(asyncio.create_task(one_request(client, results)))
            await asyncio.sleep(interval)
        await asyncio.gather(*tasks)

    completed = [r for r in results if r[0] == "done"]
    good = [r for r in completed
            if r[1] is not None and r[1] <= TTFT_SLO
            and r[2] is not None and r[2] <= TPOT_SLO]
    shed = sum(1 for r in results if r[0] == "shed")
    goodput = len(good) / duration_s
    print(f"offered={rate_per_sec:>4} req/s | completed={len(completed):>4} "
          f"shed={shed:>4} | SLO-good={len(good):>4} "
          f"| goodput={goodput:5.1f} req/s "
          f"| attainment={len(good)/max(1,len(completed)):.0%}")

if __name__ == "__main__":
    for rate in [10, 15, 20, 25, 30, 40, 60]:   # sweep offered load
        asyncio.run(ramp(rate, duration_s=60))
```

Run this against your engine with SLO management on and off, and you get the before/after table that justifies every knob in this post. On one H100 serving Llama-3-8B, a representative sweep at an offered load of **30 req/s** (well past single-replica capacity) looks like this:

| Configuration | Raw throughput | TTFT p99 | TPOT p99 | SLO attainment | **Goodput** |
|---|---|---|---|---|---|
| Unmanaged: `max_num_seqs=256`, no admission | 4,200 tok/s | 8.5 s | 95 ms | 38% | **11 req/s** |
| + capped `max_num_seqs=64`, chunked prefill | 3,600 tok/s | 1.9 s | 42 ms | 61% | 16 req/s |
| + admission control (shed at capacity) | 3,500 tok/s | 640 ms | 41 ms | 88% | 19 req/s |
| + priority scheduling (protect chat) | 3,500 tok/s | 420 ms | 40 ms | **96%** | **21 req/s** |

Read the goodput column top to bottom: each layer of the stack adds goodput, and the fully managed system delivers **1.9× the good requests** of the throughput-tuned one while its raw throughput is *17% lower*. The unmanaged config "wins" on the two vanity metrics — throughput and, trivially, it never returns a 429 — and loses on the only metric that pays the bills. This table is the entire thesis of the post in five rows: **you give up a little raw throughput at every layer, and you buy back far more goodput.**

## 12. Case studies and benchmarks

Four real systems ground everything above. The numbers are from the cited sources; where I approximate or generalize, I say so.

**DistServe (Zhong et al., OSDI 2024) — goodput as the objective.** DistServe's central argument is the one in Section 2: colocating prefill and decode on the same GPU couples their latencies, so a prefill-heavy request inflates decode TPOT and vice versa. By *disaggregating* prefill and decode onto separate GPU pools and optimizing placement for goodput (max request rate at a target SLO attainment), the paper reports up to **4.48× higher goodput** or the ability to hold **10.2× tighter SLOs** at the same load versus a strong colocated baseline. The transferable lesson is not "always disaggregate" — the sibling post [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) covers when it pays — it is that **naming goodput as the objective function changes every downstream decision**, from placement to batching to routing.

**Sarathi-Serve (Agrawal et al., OSDI 2024) — taming the throughput-latency knob.** Sarathi-Serve introduces *chunked prefills* (split a long prefill into token-budget-sized chunks) and *stall-free batching* (never let a prefill chunk stall ongoing decodes). This decouples the batch-size-versus-TPOT trade from Section 3: because a big prompt can no longer monopolize a step, you can run a *larger* decode batch for throughput without the TPOT spike that used to force you to keep batches small. The paper reports up to roughly **2.6× higher serving capacity** for Mistral-7B on an A100 and larger multiples for bigger models under strict latency SLOs. This is the mechanism behind the `enable_chunked_prefill=True` flag in Section 4, and it is why modern vLLM turns it on by default.

**Mooncake / Kimi (Qin et al., 2024; FAST 2025) — overload-oriented scheduling in production.** Mooncake is the serving architecture behind Moonshot AI's Kimi assistant, a very-high-concurrency chat service. It is KVCache-centric and disaggregated, but the part most relevant here is its **overload-oriented scheduling**: rather than admit work and hope, Mooncake *predicts* whether an incoming request can be served within SLO given current load, and performs **early rejection** of requests it cannot honor — the production incarnation of Section 8's admission control, made predictive. The paper reports that this KVCache-centric, overload-aware design lets the system handle substantially more requests under real Kimi traces (the authors cite on the order of **75% more requests** in the overloaded regime) while meeting latency SLOs. The generalizable insight: at extreme concurrency, the highest-leverage decision is *what not to admit*, and making that decision with a load prediction beats making it with a static threshold.

The mechanism is worth unpacking, because it is the most production-proven version of Section 8. Mooncake splits the fleet into a prefill cluster and a decode cluster and manages a shared, tiered **KVCache pool** that spills across the GPU cluster's otherwise-idle CPU DRAM and SSD, coordinated by a global scheduler the paper calls the *Conductor*. The overload logic sits in front of all of it: on each arrival, the scheduler estimates the request's prefill and decode cost against the current pressure on both clusters, predicts whether it can finish within its TTFT and TPOT SLOs, and rejects early if it cannot — critically at the *start*, before the request has consumed a prefill pass, rather than discovering the miss halfway through decode. The reported result is that this KVCache-centric, prediction-based design lets Kimi absorb on the order of 75% more requests under real traces while holding SLOs, with much larger throughput multiples in simulated overload. The transferable pattern, independent of the disaggregation, is that **predict-then-admit beats admit-then-hope**, and the earlier in a request's life you make the reject decision, the more GPU you reclaim for the requests you keep. The admission controller in Section 8 is the static-threshold version of this idea; Mooncake is what it becomes when the threshold is replaced by a per-request cost model — and the same KVCache-centric approach is what carries the architecture up to the larger, later Kimi models where per-request cost dispersion is even wider.

**VTC fairness (Sheng et al., OSDI 2024) — fairness as a first-class SLO.** The Virtual Token Counter work from Section 9 formalizes multi-tenant fairness for LLM serving and proves a bound: no backlogged client falls more than a bounded number of tokens behind its fair share. In a multi-tenant platform this is the difference between "we meet the SLO on average" and "we meet *every tenant's* SLO," which are very different promises to a customer whose traffic happens to be small.

**RL-and-inference co-scheduling — the emerging frontier.** A pattern now common at labs that both train and serve: the same GPU fleet runs latency-critical inference *and* throughput-oriented RL rollout generation (for RLHF or agent training). These are the ultimate latency-critical and best-effort classes — the rollouts have no user-facing TTFT SLO and can absorb any amount of preemption, so they make ideal "filler" that soaks up spare capacity and yields instantly the moment interactive load rises. It is Section 7's priority-and-preemption story taken to its logical end, with a background workload engineered specifically to be preemptible.

## 13. When to use this (and when not to)

SLO management is not free — it is queues, classifiers, admission logic, custom metrics, and a load-generation harness to tune it all. Spend that complexity where it pays and skip it where it does not.

**Use the full stack when:** you serve genuinely concurrent traffic (roughly 10+ QPS sustained, or sharp spikes) against a real latency SLA; you have mixed request classes (chat + agents + batch) sharing GPUs; you are multi-tenant; or you have already been burned by a latency cliff. If any of these is true, every layer in Section 6 earns its keep, and the goodput gains in the Section 11 table are real money.

**Use a subset when:** single-class traffic at moderate load needs only capped `max_num_seqs` + chunked prefill + a simple concurrency-limit admission control — skip priority scheduling and fairness entirely. A single-tenant internal tool with a soft SLO can run FCFS with a generous queue and be fine.

**Do not bother when:** your load is low and steady (a few QPS, GPU comfortably below the knee) — you have natural headroom and a cliff you will never reach, so a priority gateway is pure overhead. Batch/offline-only workloads have *no* latency SLO by definition; optimize them for raw throughput (big batches, no shedding, no priority) — this entire post is the wrong playbook for them. And if you have exactly one request class and one tenant, priority and fairness machinery is complexity with no payoff; cap the batch, add a concurrency limit, and move on.

The meta-rule: **the amount of SLO machinery should match the variance and heterogeneity of your traffic, not its volume.** A high-volume but uniform, slack-SLO workload needs little; a moderate-volume but spiky, multi-class, tight-SLO workload needs all of it.

## 14. Key takeaways

- **Optimize goodput, not throughput.** Goodput is the rate of requests that complete *within SLO*. A late request is a failed request that also wasted a GPU. If your dashboard lacks a goodput and SLO-attainment panel, add it before anything else.
- **Define SLOs per request class.** Interactive chat, agents, batch, and embeddings have different TTFT/TPOT targets and different priorities. The most-forgotten field — behavior under overload — is the most important, because it is the decision the system will otherwise make badly at 3 a.m.
- **The throughput-maximizing batch and the goodput-maximizing batch are different numbers.** Decode is memory-bound below the roofline ridge (batching is nearly free) and compute-bound above it (batching inflates TPOT). Operate at the goodput peak, which is a *moderate* batch, and pin it with `max_num_seqs`.
- **Unmanaged systems collapse, they do not degrade.** The $1/(1-\rho)$ term makes latency go vertical near capacity, LLM output-length variance ($c_s^2 \approx 3$–$5$) makes it worse via Kingman, and retry storms turn a transient surge into a self-sustaining metastable failure.
- **Headroom is not waste; it is the SLO.** A p99 target sets a hard $\rho_{\max}$ (e.g. 63% for p99 < 500 ms), which sets a provisioning multiplier. "Run the GPUs hotter" and "abandon the SLO" are the same sentence.
- **Shed load early, cheaply, and honestly.** Cap in-flight at the Little's-Law limit $N = \mu \cdot W_{\text{SLO}}$, return a 429 with a jittered `Retry-After` before the engine saturates, and reserve capacity for the highest class. Shedding turns the latency cliff into a goodput plateau.
- **Protect the SLO with priority and preemption inside the engine, fairness across tenants.** Priority decides which class runs; EDF refines it by slack; VTC-style fair queuing keeps one tenant from starving the rest within a class.
- **Autoscale on queue depth and SLO attainment, never CPU or GPU utilization.** GPU utilization saturates at 100% and reads identically in a healthy and a dying system. Queue depth leads the breach; attainment confirms it. Scale up fast, scale down slow, and keep a headroom floor.
- **Admission and autoscaling are partners.** Autoscaling handles the sustained shift over minutes; admission control holds the line during the cold-start window when new replicas are still loading weights.

## 15. Further reading

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** — Zhong, Liu, et al., OSDI 2024. The canonical treatment of goodput as the serving objective.
- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** — Agrawal et al., OSDI 2024. Chunked prefills and stall-free batching; the mechanism behind chunked prefill.
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — Qin et al., 2024 (FAST 2025). Kimi's production architecture, including prediction-based early rejection under overload.
- **Fairness in Serving Large Language Models** — Sheng, Zheng, et al., OSDI 2024. The Virtual Token Counter and provable multi-tenant fairness.
- **Metastable Failures in Distributed Systems** — Bronson et al., HotOS 2021. The theory of retry-storm collapse and why restarting does not fix it.
- **vLLM documentation** — engine arguments (`max_num_seqs`, `max_num_batched_tokens`, chunked prefill, priority scheduling) and the production metrics reference.
- **KEDA documentation** — scaling Kubernetes workloads on Prometheus custom metrics, with scale-up/scale-down behavior policies.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving), [model serving SLAs and metrics](/blog/machine-learning/model-serving/model-serving-slas-and-metrics), [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption), [prefill-decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation), and [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling).

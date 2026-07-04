---
title: "LLM Control Planes: Routing, Autoscaling, and Fleet Management with AIBrix and KServe"
date: "2026-07-03"
publishDate: "2026-07-03"
description: "The layer above your inference engines. Learn why a plain Kubernetes Service is the wrong load balancer for LLMs, the queuing math behind cache- and load-aware routing, and how AIBrix, KServe, llm-d, NVIDIA Dynamo, and the Gateway API Inference Extension route, autoscale, and roll out a fleet."
tags:
  [
    "model-serving",
    "inference",
    "llm-serving",
    "control-plane",
    "aibrix",
    "kserve",
    "gateway-api",
    "autoscaling",
    "kv-cache",
    "kubernetes",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/llm-control-planes-aibrix-kserve-1.webp"
---

The page came in at 2:03 a.m. Our chat assistant ran on eight vLLM replicas behind a boringly standard Kubernetes Service, one replica per H100 80GB, and for months that had been fine. Then a marketing push doubled traffic overnight and the on-call dashboard lit up in a way that made no sense: p99 time-to-first-token (TTFT, the delay before the first output token streams back) had jumped from 240 ms to 1.9 seconds, well past our 500 ms SLA — while average GPU compute utilization across the fleet sat at 61%. We had headroom. Every replica had headroom. And yet a meaningful slice of requests were queuing for over a second before a single token came back.

The cause was not the engines. Each vLLM replica was doing continuous batching, PagedAttention, prefix caching — all the right things. The cause was the thing *above* the engines, or rather the absence of it. A Kubernetes Service load-balances by picking endpoints round-robin (or by a random 5-tuple hash), blind to what each replica is actually doing. So a request carrying a 4,000-token document to summarize landed on the same replica as three other long-context requests, while the replica next to it — holding the exact system prompt this request shared with a thousand others — sat with a near-empty queue and never got the traffic. The load balancer was treating LLM requests like stateless HTTP GETs. They are nothing like stateless HTTP GETs.

The fix was not a bigger fleet. It was a **control plane**: a routing-and-scaling layer that sits above the inference engines and makes decisions the engines cannot make for themselves — which replica should serve *this* request given its prefix, its adapter, and the live queue depth of every replica; when to add a ninth replica based on KV-cache pressure rather than CPU; how to shift traffic to a new model version without a cache-cold cliff. The figure below is the whole idea in one frame: the control plane owns routing, autoscaling, and fleet management, and the data plane is the pool of engine replicas doing the actual token generation.

![Layered diagram showing a control plane of gateway, router, and autoscaler sitting above a data plane of eight vLLM engine replicas on a GPU fleet](/imgs/blogs/llm-control-planes-aibrix-kserve-1.webp)

By the end of this post you will be able to: explain, with a queuing argument, why round-robin is the wrong load balancer for LLMs; compute the cache-hit-rate collapse that naive routing causes and the TTFT cost of a single bad routing decision; write a KServe `InferenceService`, an AIBrix autoscaler-and-LoRA config, and a Kubernetes Gateway API `InferencePool` from scratch; sketch a custom cache-aware routing policy as an Envoy external-processing server; and decide when a control plane earns its keep — and when a single replica behind a plain Service is exactly right. This is a control-plane post, so it lives at the top of the serving stack we have been building through this series: **Model → Packaging → Runtime → Server → Infrastructure → Observability → Scale.** Everything here is a trade on the same triangle — **latency ↔ throughput ↔ cost** — but now the trades are made across a *fleet*, not inside one process. If you are new to the series, start with [what model serving actually is](/blog/machine-learning/model-serving/what-is-model-serving); this post assumes you know what TTFT, KV cache, and continuous batching are.

## Why a plain Kubernetes Service is the wrong load balancer for LLMs

Let us be precise about what a `Service` does, because the wrongness is specific. A Kubernetes `Service` of type `ClusterIP` programs `iptables`/IPVS rules (via kube-proxy) or an eBPF dataplane (via Cilium) that pick a backend Pod for each new connection. The selection is round-robin or a random hash over the connection 5-tuple. Once a connection is pinned, kube-proxy does not look at it again. An L7 gateway (Envoy, NGINX, HAProxy) improves on this — it can do least-connection or least-request balancing per HTTP request rather than per TCP connection — but the *signal* it balances on is still connection count or request count. Nothing in that pipeline knows anything about the LLM workload.

For a stateless microservice that is exactly right, because two requests to the same endpoint cost roughly the same and share no state. LLM inference violates both assumptions, and it violates them badly.

**First, request cost varies by one to two orders of magnitude.** A request is not a unit of work; it is a *prompt length* plus an *output length*, and both vary wildly. Prefill cost scales with the number of prompt tokens (and roughly quadratically with sequence length for attention); decode cost scales with the number of output tokens times the model's per-token latency. A 40-token prompt generating a 20-token answer might occupy a replica for 80 ms. A 6,000-token document summarization generating 800 tokens might occupy the same replica for 12 seconds. Round-robin hands these out as if they were interchangeable. They are not, and the variance is the whole problem — as we will derive below, queueing delay scales directly with the *variance* of service time, not just its mean.

**Second, LLM replicas hold state that makes locality matter enormously.** Every replica maintains a KV cache, and with [prefix caching](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) turned on, it keeps the key/value tensors of previously-seen prefixes resident. If a request shares its 1,800-token system prompt with a request that a *specific* replica served two seconds ago, routing it to that replica skips the entire prefill of those 1,800 tokens — a warm-cache TTFT of ~30 ms instead of a cold ~350 ms. A round-robin balancer scatters requests that share a prefix uniformly across the fleet, converting a warm-cache workload into a cache-miss storm.

**Third, multi-tenant LLM fleets hot-swap LoRA adapters.** When you serve fifty fine-tuned variants off one base model with [multi-LoRA serving](/blog/machine-learning/model-serving/multi-lora-and-adapter-serving), a request for adapter `support-bot-v3` is far cheaper to serve on a replica that already has that adapter resident in GPU memory than on one that must fetch it from CPU or object storage and possibly evict another adapter to make room. A load balancer that does not know which adapters live where will trigger needless adapter thrashing.

The answer to all three is a router that scores replicas on the signals that actually matter — prefix locality, live load, and loaded adapters — and sends each request to the replica that wins, instead of to whichever one is next in the rotation. The figure below shows the decision for a single request: the router evaluates all eight replicas and picks the one already holding the request's prefix, collapsing TTFT to about 30 ms.

![Branching diagram of a request entering an LLM-aware router that scores eight replicas and routes to the one holding the prefix, versus the least-loaded or LoRA-holding replicas, converging on a chosen replica with 30 millisecond TTFT](/imgs/blogs/llm-control-planes-aibrix-kserve-2.webp)

That is the control plane's core job: turn a blind rotation into an informed decision. The rest of this post is about how it makes that decision, how it scales the fleet, how it rolls out new versions, and which open-source projects package it up so you do not have to build it yourself.

### The queuing mechanics: why cost variance breaks round-robin

Here is the part most "just use a smarter load balancer" advice skips. The reason round-robin fails is not hand-wavy — it falls out of a century-old queuing result, and the failure is quantitative.

Model one replica as a single-server queue. Requests arrive at rate $\lambda$ (requests per second); each takes a random service time $S$ (seconds) with mean $\mathbb{E}[S]$ and variance $\mathrm{Var}(S)$. The server utilization is $\rho = \lambda \, \mathbb{E}[S]$, which must be below 1 or the queue grows without bound. Because LLM service times are not exponential or deterministic — they are a heavy mix of tiny and huge — we use the general-service-time M/G/1 model. The Pollaczek–Khinchine formula gives the mean time a request waits in queue before service begins:

$$
W_q = \frac{\lambda \, \mathbb{E}[S^2]}{2\,(1-\rho)}
$$

Rewriting $\mathbb{E}[S^2] = \mathrm{Var}(S) + \mathbb{E}[S]^2$ and defining the squared coefficient of variation $C_S^2 = \mathrm{Var}(S) / \mathbb{E}[S]^2$, this becomes:

$$
W_q = \frac{\rho}{1-\rho}\cdot\frac{\mathbb{E}[S]}{2}\left(1 + C_S^2\right)
$$

Read that second form carefully, because it is the whole argument. Waiting time has three factors. The first, $\rho/(1-\rho)$, is the load term — it explodes as utilization approaches 1, which is why you never run a fleet at 95% and expect good tails. The second, $\mathbb{E}[S]/2$, is half the mean service time. The third, $(1 + C_S^2)$, is the **variance penalty**. For a deterministic server (M/D/1), $C_S^2 = 0$ and the penalty is 1. For a memoryless exponential server (M/M/1), $C_S^2 = 1$ and the penalty is 2. For an LLM replica serving a realistic mix of 40-token chats and 6,000-token summaries, $C_S^2$ is routinely 5 to 20 or higher — the penalty is 6× to 21×.

Round-robin does nothing to reduce $C_S^2$. It assigns requests to replicas independently of the requests' sizes and independently of each replica's current backlog, so each replica behaves like its own isolated M/G/1 queue and eats the full variance penalty. Worse, it creates **head-of-line blocking**: a replica handed a 12-second summarization forces every request queued behind it to wait, even short ones that a neighboring idle replica could have finished in 80 ms. The fleet's aggregate GPU utilization can look healthy at 61% precisely because the work is piled unevenly — some replicas saturated and queuing, others idle.

Two control-plane moves attack this. The first is **load-aware routing**: instead of round-robin, send each request to the replica with the fewest outstanding requests (least-request, an approximation of join-shortest-queue). This is not a minor tweak; it is the "power of two choices" result from randomized load balancing (Mitzenmacher, 1996; Azar et al., 1994). Even sampling two random replicas and picking the less-loaded one shrinks the expected maximum queue length from growing like $\log n / \log\log n$ to growing like $\log\log n$ — an exponential improvement in the tail. Routing on live load effectively *pools* the queues: N independent M/G/1 queues start to behave like one M/G/N queue, and a pooled server absorbs variance far better than N isolated ones.

The second move is to pick a better load *signal* than request count, which brings us to Little's Law. For any stable system, $L = \lambda W$: the average number of requests in the system equals arrival rate times average time in system. For an LLM replica the "number in system" that actually bounds capacity is not the request count — it is **KV-cache occupancy**. Two long-context requests can consume more KV blocks (and thus more of the replica's finite decode budget) than five short ones. So the sharpest load signal a router can balance on is KV-cache utilization and the number of queued (not-yet-prefilled) tokens — exactly the signals the control planes we will meet expose. We will make the numbers concrete in a worked example after we cover cache-aware routing, because in practice the two levers interact.

#### Worked example: the variance penalty in seconds

Numbers make the argument land harder than symbols. Take a replica whose mean service time is $\mathbb{E}[S] = 0.4$ s and run it at utilization $\rho = 0.75$ — a load most operators would call comfortable. The load term $\rho/(1-\rho) = 3$ and the half-mean term $\mathbb{E}[S]/2 = 0.2$ s are fixed by that choice; only the variance penalty $(1 + C_S^2)$ changes with the workload:

- **Deterministic work** (${C_S^2 = 0}$, every request identical): $W_q = 3 \times 0.2 \times 1 = 0.6$ s.
- **Exponential work** (${C_S^2 = 1}$, the classic M/M/1): $W_q = 3 \times 0.2 \times 2 = 1.2$ s.
- **LLM work** (${C_S^2 = 12}$, a realistic 40-token-chat / 6,000-token-summary mix): $W_q = 3 \times 0.2 \times 13 = 7.8$ s.

Read that last line again. At an utterly ordinary 75% load, an isolated replica eating the full variance penalty makes requests wait almost eight seconds in queue *on average* — against a 500 ms SLA. That is the mechanical reason the 2 a.m. dashboard showed a 1.9 s p99 while GPUs sat at 61%: round-robin turns the fleet into eight of these isolated M/G/1 queues, each paying the full penalty, each blind to the others' backlogs.

Now pool the same load. Route on live queue depth and the fleet stops being eight isolated queues and starts behaving like one M/G/8 queue serving a combined 15 req/s. The multi-server waiting time follows the Erlang-C model; the Allen–Cunneen approximation carries the variance across as $W_q(\text{M/G/}c) \approx \tfrac{1+C_S^2}{2}\,W_q(\text{M/M/}c)$. Plugging in $\rho = 0.75$, $c = 8$ gives $W_q(\text{M/M/8}) \approx 0.11$ s, so $W_q(\text{M/G/8}) \approx 6.5 \times 0.11 \approx 0.72$ s. Still not great at 75% load with $C_S^2 = 12$ — but roughly a **10× cut** versus the 7.8 s isolated queue, from routing alone, on identical hardware. The pooled queue wins because a request never waits behind a 12-second summarization while a neighbor sits idle: any free server pulls the next job. Treat the exact seconds as approximations (the heavy-traffic formula is a model, not a benchmark); the order-of-magnitude gap is the robust, reproducible part.

## Cache-aware routing: locality is the biggest single lever

Load-aware routing fixes the variance tax. But for chat and agent workloads the *bigger* lever is usually cache locality, because those workloads are drenched in shared and repeated prefixes: a common system prompt across all users, a few-shot exemplar block, a tool-schema preamble, and — within a single conversation — the entire history that gets re-sent on every turn.

Recall the prefix-caching mechanic: prefill is the expensive, compute-bound phase, and if a replica already holds the KV tensors for a prefix, a new request sharing that prefix skips straight to decoding the novel suffix. The control-plane question is: *which replica holds the prefix, and did the router send the request there?* The figure below contrasts the two regimes — round-robin scattering prefix-sharing requests into a cache-miss storm, versus cache-aware routing pinning them to the owner replica.

![Before-and-after diagram: round-robin routes two requests sharing prefix P to different replicas forcing a 350 millisecond re-prefill, while cache-aware routing sends both to the same owner replica for a 30 millisecond cache hit](/imgs/blogs/llm-control-planes-aibrix-kserve-3.webp)

Let me quantify the miss. Suppose a fraction $r$ of your requests *could* reuse a prefix that already lives somewhere in the fleet — call $r$ the intrinsic reuse rate of the workload. With prefix-agnostic routing over $N$ replicas, a reusable request lands on the replica actually holding its prefix with probability roughly $r/N$ (the prefix was cached on whichever replica served it first). So the realized hit rate is about $r/N$, and it *degrades as you add replicas* — a deeply counterintuitive property that bites teams the moment they scale out. Cache-aware routing hashes the prefix and routes to the owner, lifting the realized hit rate toward $r$ (minus evictions). The gain factor is on the order of $N$.

There is an important subtlety that separates people who have run this in production from people who have only read the blog post. For a *globally shared* prefix — one system prompt every request carries — round-robin does eventually cache it on all $N$ replicas, because each replica sees it often enough. So the steady-state hit rate for that one prefix is high even under round-robin. The damage there is different: the same prefix is duplicated $N$ times, wasting $N\times$ the KV memory and evicting other useful entries. But for the long tail — per-conversation history, per-user context, per-document prefixes that are reused a handful of times and not by everyone — round-robin genuinely gets $\approx r/N$, because the second turn of a conversation has no idea which replica served the first turn. Cache-aware (and its cousin, session-affinity) routing fixes *both* pathologies: it dedups memory for the hot shared prefix and it pins each conversation to the replica that already holds its history.

#### Worked example: the prefix hit-rate collapse

Take a customer-support assistant: $N = 8$ replicas, intrinsic reuse rate $r = 0.9$ (90% of requests share a cacheable prefix — a big system prompt plus multi-turn history). A warm prefix hit yields TTFT of ~30 ms; a cold miss forces a full ~1,800-token re-prefill at ~200 ms of prefill plus scheduling, call it ~350 ms TTFT.

- **Round-robin.** Realized hit rate for the session-specific tail is about $r/N = 0.9/8 \approx 11\%$. Mean TTFT $\approx 0.11 \times 30 + 0.89 \times 350 \approx 315$ ms. And you are paying that prefill compute over and over.
- **Cache-aware.** Realized hit rate $\approx 85\%$ (r minus some eviction losses). Mean TTFT $\approx 0.85 \times 30 + 0.15 \times 350 \approx 78$ ms.

That is a **4× reduction in mean TTFT with zero hardware change** — the same eight H100s, a different routing decision. On the compute side, cutting re-prefill from 89% of requests to 15% frees an enormous amount of GPU time, which either lifts throughput or lets you shrink the fleet. This is the single highest-leverage thing a control plane does for a chat workload, which is why every project we discuss below implements some form of it.

The catch, and there is always a catch: cache-aware routing pushes *toward* concentration (send everything with prefix P to one replica) while load-aware routing pushes *toward* spreading (send to the least-loaded replica). A pure cache-aware router will happily overload the owner of a viral prefix. Real routers therefore compute a **blended score** — reward a cache hit, penalize a deep queue — and rebalance when the owner gets hot. That blend is the actual engineering, and it is what separates the routing policies we compare next.

### Session affinity, consistent hashing, and cache eviction

There are two families of cache-aware routing, and mixing them up is a common source of pain. The first is **session affinity**: the router keys on a stable identifier — a session cookie, a conversation ID, an API key — and pins every request bearing that key to the same replica for the session's lifetime. It is cheap (a hash of one header, no prompt inspection) and it captures the fattest reuse case, multi-turn chat, because the whole growing conversation lands back on the replica that already holds it. Its weakness is that it is blind to reuse *across* sessions: two different users hitting the same 1,800-token system prompt get pinned to different replicas and each pays the prefill.

The second family is **prefix-hash routing**: the router hashes the leading tokens of the actual prompt (the shared system prompt, the tool schema, the few-shot block) and routes on that, so cross-session reuse is captured too. This is stronger but costs more — the router must buffer and inspect the request body, and it needs a live map of which prefix lives on which replica. Production routers usually run both: session affinity as the fast path, prefix-hash as the fallback when there is no session key or when the prefix is shared widely.

The map from prefix (or session) to owner replica is where **consistent hashing** earns its place. If you route with a naive `hash(prefix) % N`, then adding or removing a single replica changes the modulus and remaps *almost every* prefix to a new owner — a fleet-wide cache-miss storm every time the autoscaler acts, which is exactly when you can least afford it. Consistent hashing (or its bounded-load variant) places replicas on a hash ring so that adding or removing one replica only remaps the $1/N$ slice of prefixes near that replica's ring position; the other $(N-1)/N$ stay put and stay warm. Since autoscaling and rollouts churn the replica set constantly, a router that is *not* consistent-hash-based will torch its cache on every scale event, quietly erasing the gain the routing was supposed to buy. AIBrix and the Gateway API endpoint pickers both use ring-based schemes for precisely this reason.

Eviction is the last piece of the mechanic. A prefix owner does not hold a prefix forever — the engine's KV cache is finite, and prefix blocks are reclaimed under an LRU (or the engine's RadixAttention eviction) when memory fills. So the router's owner map can point at a replica that has *already evicted* the prefix, turning a predicted hit into a silent miss. Two defenses matter. First, keep the owner map's TTL roughly in line with the engine's eviction horizon, so stale entries expire rather than mislead. Second, when the router balances toward concentration for cache reuse, it also raises that replica's KV pressure — which the autoscaler must see, or the concentration wins latency and then loses it to preemption. This is the routing–autoscaling coupling that the projects below all wrestle with: routing decides where memory is spent, and autoscaling decides how much memory exists.

## Load-, queue-, and KV-aware routing: a ladder of policies

Putting the pieces together, routing policies form a ladder. Each rung buys lower tail latency or higher cache reuse, and each rung costs the control plane more live state to track and more ways to get the blend wrong. The matrix below lays out the trade-offs; read it top to bottom as increasing sophistication.

![Matrix comparing five routing policies — round-robin, least-request, queue/KV-aware, cache-aware, LoRA-aware — across TTFT p99, cache hit rate, load balance quality, and complexity](/imgs/blogs/llm-control-planes-aibrix-kserve-4.webp)

Walking the rungs:

- **Round-robin / random.** The kube-proxy default. Trivial, load-blind, cache-blind. Fine for stateless services; wrong for LLMs, as derived above.
- **Least-request (least-outstanding-requests).** Route to the replica with the fewest in-flight requests. This is the "power of two choices" win — it pools the queues and crushes the variance tax. It is the single best *first* upgrade because it is cheap (the gateway already counts in-flight requests) and needs no engine cooperation. Its blind spot: it treats all requests as equal-cost, so two long-context requests count the same as two chats.
- **Queue-/KV-aware.** Route on the engine's real state: pending prefill tokens and KV-cache utilization, scraped from the engine's metrics endpoint (vLLM exposes `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, and friends). This is least-request done right — the load signal now reflects actual GPU pressure rather than a request count. It is what the Gateway API Inference Extension's endpoint picker balances on by default.
- **Cache-aware (prefix-aware).** Add prefix locality to the score, as derived in the previous section. Biggest lever for chat/agent workloads; must be blended with load to avoid hot-replica overload.
- **LoRA-aware.** Add adapter locality: prefer a replica that already has the requested adapter resident. Essential for high-density multi-tenant fleets; layered on top of cache- and load-aware, not instead of them.

A production router does not pick one rung — it composes them into a scoring function. A representative form, lower-is-better, for candidate replica $i$ and request with prefix hash $h$ and adapter $a$:

$$
\text{score}_i = w_1\,\text{queue}_i + w_2\,\text{kv\_util}_i - w_3\,\mathbb{1}[\,h \in \text{cache}_i\,] - w_4\,\mathbb{1}[\,a \in \text{adapters}_i\,]
$$

The router picks $\arg\min_i \text{score}_i$. The weights $w_1..w_4$ encode your priorities: crank $w_3$ up for a chat workload dominated by shared prefixes; crank $w_1$ up when you care most about tail latency under bursty load. Getting these weights right for your traffic is exactly the tuning work a control plane exists to let you do.

#### Worked example: the cost of a single routing miss in TTFT

Why does the blend matter so much? Because one bad routing decision can blow the SLA on its own. Say each pending request ahead of yours needs ~200 ms of prefill before the replica can get to you (a mix of prompt sizes). If the router sends your request to a replica with a queue depth of 3, your TTFT inherits roughly $3 \times 200 = 600$ ms of head-of-line wait *before* your own prefill even starts. Against a 500 ms p99 SLA, that single decision is already a miss — and it happened while a neighboring replica sat with an empty queue. Round-robin makes this mistake constantly; a queue-aware router essentially never does. That is the difference between a p99 of 1.9 seconds and a p99 of 180 ms on the identical fleet.

### LoRA-aware routing in practice

The LoRA rung deserves a moment because it is where control-plane routing and adapter serving meet. Loading a rank-16 adapter for an 8B model means moving ~50–200 MB from CPU or object storage into GPU memory over PCIe (~10–16 GB/s on a Gen4/Gen5 link), so a cold adapter load costs roughly 5–20 ms plus any eviction of a currently-resident adapter to make room. That is cheap once, ruinous if it thrashes. If your router scatters requests for `support-bot-v3` uniformly, every replica ends up loading, evicting, and reloading the same adapters — the adapter equivalent of the cache-miss storm. LoRA-aware routing consolidates requests for a given adapter onto the replicas that hold it (AIBrix calls this high-density LoRA management, and it registers adapters as first-class objects so the router knows the placement). The engine still does the heavy lifting — heterogeneous batched adapter application, S-LoRA/Punica-style kernels — but the control plane decides *where* each adapter-bearing request goes. If you want the engine-side mechanics, the [multi-LoRA and adapter serving](/blog/machine-learning/model-serving/multi-lora-and-adapter-serving) post covers them; here the point is that adapter placement is a routing input, not an afterthought.

### A load test that compares routing policies

None of the ladder is worth deploying if you cannot measure the rung you are on. The right test is not a synthetic uniform-load benchmark — that hides the exact variance and locality the router exists to exploit. You want a trace that looks like your traffic: a shared system prompt across all requests, a pool of "sessions" that each re-send growing history, and a heavy-tailed mix of prompt lengths. The harness below replays such a trace against a gateway, measures TTFT per request, and — because AIBrix lets you pick the strategy per request with a header — sweeps the policies over the *same* trace so the comparison is apples-to-apples:

```python
# Replay one trace against several routing strategies and compare TTFT.
# Requires: pip install aiohttp numpy
import asyncio, time, random, numpy as np, aiohttp

GATEWAY = "http://aibrix-gateway/v1/chat/completions"
SYSTEM  = "You are a helpful support agent. " * 120     # ~1,800-token shared prefix
SESSIONS = 200                                           # distinct conversations
REQUESTS = 4000
CONCURRENCY = 64
STRATEGIES = ["random", "least-request", "least-kv-cache", "prefix-cache"]

def make_prompt():
    # Each session re-sends its own growing history: heavy prefix reuse.
    sid = random.randint(0, SESSIONS - 1)
    turns = random.randint(1, 8)
    history = f"[session {sid}] " + ("prior turn. " * (turns * random.randint(5, 40)))
    return SYSTEM + history + "Answer the user's latest question."

async def one(session, strategy, sem, out):
    async with sem:
        body = {"model": "llama3-8b",
                "messages": [{"role": "user", "content": make_prompt()}],
                "max_tokens": 32, "stream": True}
        headers = {"routing-strategy": strategy, "content-type": "application/json"}
        t0 = time.perf_counter()
        async with session.post(GATEWAY, json=body, headers=headers) as resp:
            async for _chunk in resp.content:           # first byte == first token
                out.append((time.perf_counter() - t0) * 1000)   # TTFT in ms
                break

async def run(strategy):
    out, sem = [], asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(*(one(s, strategy, sem, out) for _ in range(REQUESTS)))
    a = np.array(out)
    print(f"{strategy:>16}  p50={np.percentile(a,50):6.0f}ms  "
          f"p99={np.percentile(a,99):6.0f}ms  mean={a.mean():6.0f}ms")

async def main():
    for strat in STRATEGIES:
        await run(strat)
        await asyncio.sleep(20)     # let caches settle between sweeps

asyncio.run(main())
```

Two things make this a real test rather than a demo. First, the 20-second pause between strategies lets each policy warm (or fail to warm) its caches from a comparable cold-ish start, so you are not measuring the residue of the previous sweep. Second, TTFT is captured at the first streamed chunk, not at response completion — TTFT and end-to-end latency are different metrics with different owners, and a router moves TTFT far more than it moves total generation time. Run this against an 8-replica fleet and the shape of the output is the whole argument for the ladder; a representative run looks like this:

| Strategy | p50 TTFT | p99 TTFT | Mean TTFT |
|---|---|---|---|
| `random` (round-robin) | ~305 ms | ~1,850 ms | ~320 ms |
| `least-request` | ~180 ms | ~640 ms | ~205 ms |
| `least-kv-cache` | ~150 ms | ~520 ms | ~170 ms |
| `prefix-cache` | ~55 ms | ~190 ms | ~78 ms |

The jump from `random` to `least-request` is the variance-tax fix from the queuing section — the p99 collapses because the fleet pools its queues. The jump from `least-kv-cache` to `prefix-cache` is the locality lever — the p50 collapses because the shared prefix stops being re-prefilled. Notice that no single rung wins on every axis for every workload: on a trace with *no* prefix reuse, `prefix-cache` and `least-kv-cache` converge, and the honest answer is to run this harness on your own trace before you pick a default. Numbers here are illustrative and consistent with the mechanics; the point is the method, not the digits.

## The Kubernetes Gateway API Inference Extension

For a long time everyone who needed this built it in-house, which is why there were a dozen incompatible LLM routers. In 2025 the Kubernetes community standardized the interface. The **Gateway API Inference Extension**, driven by the WG-Serving / SIG-Network working group, extends the standard Gateway API with two inference-specific resources and a pluggable routing brain. The figure below shows the wiring: an `HTTPRoute` points at an `InferencePool`; the Endpoint Picker reads live pod metrics and tells the gateway which replica to hit.

![Graph of the Gateway API Inference Extension request flow: an HTTPRoute routes to the Inference Gateway which consults the EPP scheduler, which reads metrics from two pods and routes to the 35 percent full pod over the 88 percent full one](/imgs/blogs/llm-control-planes-aibrix-kserve-5.webp)

The pieces:

- **`InferencePool`** — a set of inference-serving Pods (selected by label) plus a reference to an extension that picks endpoints for them. It is the inference-aware analog of a `Service`: where a `Service` gives you a dumb VIP over a Pod set, an `InferencePool` gives you a *smart* one whose endpoint choice is delegated to the picker.
- **`InferenceObjective`** (called `InferenceModel` in early releases) — maps a served model name and a priority/criticality onto a pool, so the gateway can enforce fairness and shed lower-priority traffic first under load.
- **Endpoint Picker (EPP)** — the routing brain. It is a data-plane sidecar that speaks Envoy's external-processing (`ext_proc`) protocol: for each incoming request the gateway calls out to the EPP, which has been scraping each pod's KV-cache utilization, pending-queue length, and active LoRA adapters, and returns the endpoint the request should go to. The gateway then forwards the request there. If the EPP is unavailable, `failureMode: FailOpen` lets the gateway fall back to normal load balancing rather than dropping traffic.

Here is a minimal but complete manifest set. First the pool and its picker:

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: llama3-pool
spec:
  # Every Pod carrying this label joins the pool.
  selector:
    app: llama3-8b
  # The port the model server listens on (vLLM's OpenAI server default).
  targetPorts:
    - number: 8000
  # Delegate endpoint choice to the EPP service.
  endpointPickerRef:
    name: llama3-epp
    port: 9002
    failureMode: FailOpen
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective          # was "InferenceModel" in v0.1-v0.2
metadata:
  name: llama3-chat
spec:
  poolRef:
    name: llama3-pool
  # Higher priority requests are served first and shed last under pressure.
  priority: 10
```

Then a standard `HTTPRoute` that sends `/v1/chat/completions` to the pool — note the `backendRef` points at the `InferencePool`, not a `Service`:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs:
    - name: inference-gateway         # your Gateway resource
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /v1/chat/completions
      backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: llama3-pool
```

The beauty of this design is that the *interface* is standard while the *policy* is pluggable. The reference EPP ships a sensible KV-and-queue-aware scorer, but you can swap in your own — which is where the next section's custom router fits. Any Gateway API implementation that supports the extension (Envoy Gateway, Istio, kgateway, agentgateway) can drive it, so you are not locked to one vendor's router. This is the piece that turned "LLM routing" from a pile of bespoke Go services into a portable Kubernetes primitive, and both KServe and llm-d build on it directly.

### A custom cache-aware routing policy

When the reference scorer is not enough — say you want a specific prefix-hash-plus-queue blend, or you want to route by tenant — you write your own EPP as an `ext_proc` server. The skeleton below scores replicas by prefix locality and live load, remembers where each prefix now lives, and sets the upstream host header the gateway will honor. It is illustrative, not production-hardened, but it shows the shape of the thing:

```python
# Envoy external-processing (ext_proc) server. For each request the gateway
# streams us the headers/body; we score every replica and pick a target.
import grpc
from concurrent import futures
from envoy.service.ext_proc.v3 import external_processor_pb2 as ep
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ep_grpc

PREFIX_CHARS = 512          # hash the leading N chars = the shared prefix
W_QUEUE, W_KV, W_CACHE = 0.4, 0.3, 1.0   # blended-score weights

class Replica:
    __slots__ = ("id", "endpoint", "pending", "kv_util")
    # pending / kv_util are refreshed from each vLLM /metrics scrape.

class CacheAwareRouter(ep_grpc.ExternalProcessorServicer):
    def __init__(self, fleet):
        self.fleet = fleet            # {id: Replica}, updated by a metrics poller
        self.owner = {}               # prefix_hash -> replica_id (LRU-evicted)

    def _score(self, r, prefix_hash):
        # Lower is better: reward a cache hit, penalize queue depth and KV pressure.
        hit = 1.0 if self.owner.get(prefix_hash) == r.id else 0.0
        return W_QUEUE * r.pending + W_KV * r.kv_util - W_CACHE * hit

    def _pick(self, prompt):
        h = hash(prompt[:PREFIX_CHARS])
        target = min(self.fleet.values(), key=lambda r: self._score(r, h))
        self.owner[h] = target.id     # this prefix now lives here
        return target

    def Process(self, request_iterator, context):
        prompt = ""
        for req in request_iterator:
            if req.HasField("request_body"):
                prompt += req.request_body.body.decode("utf-8", "ignore")
                if req.request_body.end_of_stream:
                    target = self._pick(prompt)
                    mut = ep.HeaderMutation(set_headers=[
                        ep.HeaderValueOption(header=ep.HeaderValue(
                            key="x-gateway-destination-endpoint",
                            raw_value=target.endpoint.encode()))])
                    yield ep.ProcessingResponse(request_body=ep.BodyResponse(
                        response=ep.CommonResponse(header_mutation=mut)))
            else:
                yield ep.ProcessingResponse(request_headers=ep.HeadersResponse())

def serve():
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    ep_grpc.add_ExternalProcessorServicer_to_server(CacheAwareRouter(load_fleet()), s)
    s.add_insecure_port("[::]:9002")
    s.start(); s.wait_for_termination()
```

The two things that make this real rather than a toy are the metrics poller that keeps `pending` and `kv_util` fresh (scraping each replica's `/metrics` every 250–500 ms) and a proper LRU/consistent-hash scheme for `owner` so that adding or removing a replica does not blow away the whole prefix map. Production EPPs also add a small amount of hysteresis so a prefix does not ping-pong between replicas under load.

### Priority, fairness, and load shedding under saturation

Routing decides *where* a request goes when there is somewhere good to send it. But a fleet under a genuine spike eventually has *nowhere* good — every replica is at its KV ceiling and the queue is growing. What happens then is a control-plane decision, and it is the reason the `InferenceObjective` (formerly `InferenceModel`) carries a priority. When the pool is saturated, the endpoint picker sheds the lowest-priority traffic first so the requests that matter most keep their SLA, instead of letting every tenant degrade uniformly into a fleet-wide brownout.

Make it concrete. Suppose one pool serves two workloads off the same replicas: an interactive chat product at `priority: 10` and a nightly batch-summarization job at `priority: 1`. At 09:00 a launch pushes arrival rate 40% over the fleet's sustainable throughput. A priority-blind balancer spreads the pain evenly: both workloads see queue depth climb, both blow their targets, and the revenue-bearing chat product is now as slow as the batch job that nobody is watching in real time. A priority-aware picker instead starts rejecting (or deferring, with a `429` and a `Retry-After`) the `priority: 1` batch requests the moment KV pressure crosses the shed threshold, freeing exactly enough capacity to hold the chat product's TTFT inside its SLA. The batch job runs slower — which is fine, it is a batch job — and the interactive product stays healthy. That is graceful degradation, and it is only expressible because the control plane knows each request's criticality; a plain `Service` has no vocabulary for "this request matters more than that one."

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective
metadata:
  name: batch-summarizer
spec:
  poolRef:
    name: llama3-pool
  # Low criticality: first to be shed when the pool saturates.
  priority: 1
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceObjective
metadata:
  name: interactive-chat
spec:
  poolRef:
    name: llama3-pool
  # High criticality: served first, shed last.
  priority: 10
```

The design lesson generalizes past this one CRD: fairness under load is a first-class control-plane responsibility, not something you bolt on with a rate limiter after the fact. AIBrix expresses it as fairness routing, Dynamo's SLO Planner reasons about it while sizing the fleet, and the Gateway API bakes it into the objective. Whichever project you pick, decide *before* the incident which traffic you are willing to shed, because at 2 a.m. the fleet will make that decision with or without you.

## The projects: AIBrix, KServe, llm-d, and NVIDIA Dynamo

You rarely build all of this yourself. Four open-source stacks package the control plane, each with a different center of gravity. The matrix below compares them on the dimensions that matter — gateway/router, autoscaler, KV-cache strategy, and the underlying engine.

![Matrix comparing AIBrix, KServe, llm-d, and NVIDIA Dynamo across gateway/router, autoscaler, KV cache, and engine dimensions](/imgs/blogs/llm-control-planes-aibrix-kserve-6.webp)

**AIBrix** (open-sourced by ByteDance, now under the vLLM project umbrella) is the most "batteries-included, LLM-first platform" of the four. It was built inside ByteDance for real production traffic and follows a co-design philosophy: every layer is purpose-built to sit above vLLM. Its control plane includes an LLM-aware Envoy-based gateway with pluggable routing strategies (`random`, `least-request`, `throughput`, `prefix-cache`, `least-kv-cache`, `least-latency`, and fairness routing), an LLM-specific autoscaler that scales on KV-cache utilization with second-level reaction, a **distributed KV-cache runtime** that pools cache across nodes to boost cross-request reuse, high-density LoRA management via a `ModelAdapter` custom resource, a `GPU Optimizer` for heterogeneous fleets, and accelerator diagnostic tooling. For large multi-node models it uses `RayClusterFleet` to combine Kubernetes for coarse scheduling with Ray for fine-grained execution. If you want one project that does routing, autoscaling, KV pooling, and LoRA out of the box, this is it.

**KServe** is the CNCF-incubating (as of November 2025) standard for model serving on Kubernetes, and it is broader than LLMs — it serves scikit-learn, XGBoost, ONNX, and PyTorch models through one `InferenceService` interface. Its LLM story has grown fast: v0.16 added the `LLMInferenceService` CRD for generative workloads and an integration with llm-d for disaggregated serving, prefix caching, intelligent scheduling, and variant autoscaling. KServe's distinctive features are its Knative-based serverless mode (true **scale-to-zero** and canary rollout out of the box), its ModelMesh component for high-density multi-model serving (many small models packed onto shared servers), and its support for the Gateway API. Reach for KServe when you have a heterogeneous model portfolio, not just LLMs, and want one platform for all of it.

**llm-d** is the newest of the four — a Kubernetes-native distributed inference stack launched in 2025 by Red Hat with Google, IBM, NVIDIA, CoreWeave, and others. It is deliberately *modular and layered on standards*: vLLM as the engine, Kubernetes as the substrate, and the Gateway API Inference Extension (which the project calls the Inference Gateway, IGW) as the routing layer. Its headline capabilities are KV-cache-aware routing, disaggregated prefill/decode, and a vLLM-optimized inference scheduler. Think of llm-d less as a monolithic platform and more as the reference assembly of the best-of-breed Kubernetes inference primitives — which is exactly why KServe integrates it rather than reinventing it.

**NVIDIA Dynamo** (announced at GTC 2025, reached 1.0 shortly after) is the disaggregation-first orchestrator and the spiritual successor to Triton for distributed LLM serving. It is inference-engine agnostic — it drives vLLM, TensorRT-LLM, and SGLang — and its four signature components are a KV-cache-aware **Smart Router**, an **SLO Planner** that adds and removes GPUs to hold a latency target, a tiered **KV Block Manager** that offloads cache across GPU/CPU/SSD/object storage, and **NIXL**, a low-latency point-to-point library that moves KV tensors between GPUs over RDMA or NVMe at wire speed. NVIDIA reports up to 30× more requests served on DeepSeek-R1 on Blackwell (GB200 NVL72) with disaggregation plus these components. Dynamo is where you go when your bottleneck is multi-node disaggregated serving at frontier scale; its modular pieces (especially NIXL) have been adopted by llm-d, vLLM, and SGLang. If disaggregation is your problem, pair this with the [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) deep-dive.

Here is the same comparison in prose-table form, with a bit more detail than the figure carries:

| Dimension | AIBrix | KServe | llm-d | NVIDIA Dynamo |
|---|---|---|---|---|
| Origin | ByteDance (vLLM org) | Kubeflow → CNCF | Red Hat + partners | NVIDIA |
| Center of gravity | LLM-first platform | General model serving | Modular K8s primitives | Disaggregation at scale |
| Router | Envoy gateway, 7+ strategies | Gateway API / Envoy | Inference Gateway (EPP) | KV-aware Smart Router |
| Autoscaler | KV-util, second-level | Knative (scale-to-zero) | Variant autoscaling | SLO Planner (GPU-level) |
| KV cache | Distributed KV pool | via llm-d | KV-aware routing | KV Block Mgr + NIXL |
| Disaggregation (P/D) | via engine | via llm-d | native | native (its whole thesis) |
| Scale-to-zero | via autoscaler | Knative (native) | via autoscaler | via SLO Planner |
| Multi-node / big models | `RayClusterFleet` | multi-framework | LeaderWorkerSet | native, frontier-scale |
| LoRA | High-density `ModelAdapter` | via runtime | via vLLM | via engine |
| Engine | vLLM (co-designed) | any predictor runtime | vLLM (native) | vLLM / TRT-LLM / SGLang |
| Governance | vLLM project | CNCF incubating | Red Hat + partners | NVIDIA |
| Best when | Want it all, LLM-only | Mixed model portfolio | Standards-based assembly | Frontier multi-node scale |

A useful way to hold these in your head: KServe is the *platform* (serves everything, LLMs included), llm-d is the *standards-based assembly* of LLM primitives, AIBrix is the *opinionated LLM-first platform*, and Dynamo is the *disaggregation engine*. They overlap and increasingly interoperate — Dynamo's NIXL shows up inside llm-d, and KServe drives llm-d — because they all sit on the same foundation (vLLM engines, Kubernetes, the Gateway API) and converge on the same signals (KV locality, queue depth, adapters). Choosing among them is less "which is best" and more "which center of gravity matches my problem," which is the same lens the [choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack) post applies to the engine layer.

### A KServe InferenceService, end to end

Enough architecture; here is a working KServe deployment. The `InferenceService` below serves Llama-3-8B on a vLLM-backed HuggingFace runtime with Knative autoscaling. The autoscaling annotations are the important part — `target` is the desired concurrent requests per replica (the signal Knative scales on), and setting `minScale: 0` would give you scale-to-zero for a bursty or dev workload:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-8b
  annotations:
    # Scale on concurrent in-flight requests per replica.
    autoscaling.knative.dev/metric: "concurrency"
    autoscaling.knative.dev/target: "10"
    autoscaling.knative.dev/minScale: "1"     # 0 = scale-to-zero when idle
    autoscaling.knative.dev/maxScale: "8"
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      runtime: kserve-huggingfaceserver        # vLLM-backed serving runtime
      storageUri: hf://meta-llama/Meta-Llama-3-8B-Instruct
      args:
        - --tensor-parallel-size=1
        - --max-model-len=8192
        - --enable-prefix-caching               # engine-side prefix caching
        - --enable-lora
      resources:
        limits:
          nvidia.com/gpu: "1"
        requests:
          nvidia.com/gpu: "1"
```

Apply that and KServe wires up the Knative service, the ingress route, the revision, and the autoscaler. For the LLM-optimized path — disaggregated serving, KV-aware routing, variant autoscaling — you would instead use the newer `LLMInferenceService` CRD, which hands the heavy lifting to llm-d underneath while keeping the same declarative front door.

### An AIBrix autoscaler and LoRA config

AIBrix's control plane is where the LLM-specific autoscaling and LoRA management live as first-class custom resources. Start at the front door: AIBrix installs an Envoy-based gateway, and you point traffic at it with a standard Gateway API `HTTPRoute`. The gateway is the piece that reads the `routing-strategy` header and applies the matching policy per request, so the deployment wiring is deliberately boring — the intelligence is in the gateway plugin, not the manifest:

```yaml
# Route inference traffic to the AIBrix gateway service; the gateway plugin
# applies the per-request routing-strategy (prefix-cache, least-kv-cache, ...).
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: aibrix-llm-route
spec:
  parentRefs:
    - name: aibrix-eg                    # the Envoy Gateway AIBrix installs
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /v1/
      backendRefs:
        - name: aibrix-gateway-plugin    # AIBrix's routing brain
          port: 8888
```

With the front door in place, a `PodAutoscaler` scales the deployment on vLLM's KV-cache utilization gauge rather than CPU — this is the crux of getting autoscaling right, which the next section derives:

```yaml
apiVersion: autoscaling.aibrix.ai/v1alpha1
kind: PodAutoscaler
metadata:
  name: llama3-8b-scaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama3-8b
  minReplicas: 1
  maxReplicas: 12
  # Scale on the engine metric that reflects real GPU pressure.
  metricsSources:
    - metricSourceType: pod
      protocolType: http
      port: "8000"
      path: /metrics
      targetMetric: "gpu_cache_usage_perc"      # vLLM KV-cache utilization
      targetValue: "0.85"
  scalingStrategy: KPA           # Knative-style, second-level reaction
```

Then a `ModelAdapter` that registers a LoRA adapter as a routable object, so the gateway's LoRA-aware routing knows where it lives and how many replicas hold it:

```yaml
apiVersion: model.aibrix.ai/v1alpha1
kind: ModelAdapter
metadata:
  name: support-bot-v3
  labels:
    model.aibrix.ai/name: llama3-8b
spec:
  baseModel: llama3-8b
  artifactURL: s3://prod-adapters/support-bot/v3/
  replicas: 3                    # keep the adapter resident on 3 replicas
  schedulerName: default
```

Finally, you select a routing strategy per request with a header — the same fleet can serve one workload prefix-cache-aware and another least-request-aware:

```bash
# Route this request to whichever replica already holds its prefix.
curl -s http://aibrix-gateway/v1/chat/completions \
  -H "routing-strategy: prefix-cache" \
  -H "model: llama3-8b" \
  -H "content-type: application/json" \
  -d '{"model":"llama3-8b","messages":[{"role":"user","content":"..."}]}'
```

## Autoscaling on the right signal

Routing decides *where* a request goes. Autoscaling decides *how many* replicas exist to route to. And the single most common autoscaling mistake in LLM serving is scaling on the wrong signal — specifically, scaling on CPU utilization, which is what a stock Kubernetes HorizontalPodAutoscaler does by default. The figure below shows why that fails and what to do instead.

![Before-and-after diagram: a CPU-based HPA sees 40 percent CPU and thinks the replica is idle while its KV cache is 98 percent full and the queue backs up, versus a KV-aware autoscaler that scales when KV utilization crosses 85 percent and holds the SLA](/imgs/blogs/llm-control-planes-aibrix-kserve-8.webp)

Here is the failure mode in one sentence: on an LLM replica, the CPU is mostly waiting on the GPU, so CPU utilization barely moves even as the GPU's decode queue backs up and TTFT climbs. A CPU-triggered HPA looks at 40% CPU, concludes the replica is idle, and does not scale — right up until the KV cache hits 98%, requests start getting preempted or rejected, and the SLA is already blown. By the time CPU finally rises (if it ever meaningfully does), you are minutes late, and a new replica takes tens of seconds to schedule, pull, and warm up. You scaled after the incident instead of before it.

The right signals are the same ones the router balances on, because they measure the actual bottleneck:

- **KV-cache utilization** (`vllm:gpu_cache_usage_perc`) — the fraction of KV blocks in use. When it approaches your target (say 85%), the replica is near its concurrency ceiling and you should add capacity *now*, before it saturates.
- **Pending-request / waiting-queue depth** (`vllm:num_requests_waiting`) — requests admitted but not yet running. A rising queue is the earliest leading indicator of TTFT pain.
- **Request rate and per-token latency** — for higher-level SLO-aware scaling that targets a TTFT or TPOT budget directly.

To wire a stock Kubernetes HPA to these, you export the vLLM metrics to Prometheus and use the Prometheus Adapter to expose them as custom metrics. The HPA then scales on the custom metric:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama3-8b-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama3-8b
  minReplicas: 2
  maxReplicas: 12
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_gpu_cache_usage_perc      # via prometheus-adapter
        target:
          type: AverageValue
          averageValue: "850m"                  # 0.85 = 85% KV utilization
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0             # react fast to KV pressure
      policies:
        - type: Pods
          value: 2                              # add up to 2 replicas at a time
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300           # scale down slowly to avoid flapping
```

The asymmetric `behavior` block is doing real work here: scale *up* aggressively (a saturated fleet misses SLAs and users notice) but scale *down* slowly (tearing down a warm replica whose KV cache and prefix cache are populated throws away real value, and flapping wastes GPU-minutes of warm-up). This is a control-plane opinion baked into config. For scale-to-zero — worth it only for spiky, latency-tolerant, or dev workloads — Knative (via KServe) or KEDA gives you the machinery, but respect the cold-start tax: pulling a multi-gigabyte model and warming the cache can take 30–90 seconds, which is unacceptable for interactive traffic and fine for a nightly batch job. The [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling) post goes deeper on the GPU-supply side of this equation.

#### Worked example: sizing the KV-utilization target

What target should you scale at? Suppose a replica saturates (starts preempting requests) at 100% KV utilization and holds SLA up to ~90%. A new replica takes 45 seconds to become ready, and your traffic can grow at most ~3%/second during a spike. If you scale at 85%, you have 5 percentage points of headroom before SLA pain and 15 before hard saturation. At 3%/s growth, 5 points buys you ~1.7 seconds — not enough to cover a 45-second warm-up. The lesson: a reactive KV target alone cannot cover a slow warm-up under fast growth. You either provision a warm buffer (a higher `minReplicas` so there is always slack), pre-warm replicas predictively (scale on a forecast, not just the current gauge), or accept a brief SLA dip during spikes. Control planes like AIBrix and Dynamo's SLO Planner lean toward predictive and second-level reactive scaling precisely because the naive reactive math does not close under realistic warm-up times. This is the autoscaling analog of the routing blend: the naive single-signal answer is wrong, and the control plane exists to let you do the nuanced thing.

### Scale-to-zero, KEDA, and the flapping tax

The HPA above floors at two replicas because it never scales below one — a stock HPA cannot reach zero. For a spiky, latency-tolerant, or dev workload where paying for an idle H100 all night is the real waste, you want the fleet to drop to zero when traffic stops and cold-start back up on the next request. Two machines give you this: Knative (which KServe wraps) and KEDA. KEDA is the more portable of the two because it drives a normal HPA and adds the zero-to-one and one-to-zero transitions on top of any scaler. A `ScaledObject` that scales a vLLM deployment from zero on queue depth looks like this:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llama3-8b-scaledobject
spec:
  scaleTargetRef:
    name: llama3-8b
  minReplicaCount: 0            # scale-to-zero when idle
  maxReplicaCount: 12
  cooldownPeriod: 300          # wait 5 min of idle before dropping to zero
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: sum(vllm:num_requests_waiting)   # queued work across the fleet
        threshold: "5"
```

The `cooldownPeriod` is the quiet lever that decides your economics, and it is the same tension as `scaleDown` stabilization above: drop to zero too eagerly and you pay the cold-start tax on the next request; hold too long and you pay for idle GPUs. That trade has a break-even you can actually compute.

#### Worked example: when scale-to-zero and flapping cost more than they save

Say an idle H100 costs on the order of \$2–4 per GPU-hour on-demand, and a cold start (pull a multi-gigabyte model, load weights, warm the cache) costs 60 seconds of unusable GPU plus a one-time TTFT spike for the first users. If your traffic goes idle in clean, hours-long gaps — nights and weekends for an internal tool — scale-to-zero is pure win: you save whole GPU-hours and pay a 60-second tax a couple of times a day. But if traffic is *bursty-but-frequent* — a request every few minutes — a short cooldown makes the fleet flap: it tears down the replica, pays 60 seconds of cold start on the next request, and repeats. Each flap wastes ~60 GPU-seconds *and* throws away a warm KV/prefix cache that took real traffic to populate. Ten flaps an hour is 600 GPU-seconds — ten minutes of GPU per hour spent purely on warm-up, plus ten cache-cold latency cliffs your users feel. The fix is the same hysteresis the rollout section relies on: set `cooldownPeriod` well above your inter-arrival gap so the fleet only drops to zero on *genuine* idle, and reserve scale-to-zero for workloads whose idle periods are long and whose users tolerate a cold first request. For interactive production traffic, a warm `minReplicas` of one or two almost always beats the flapping tax.

## Rollouts and canaries at the control-plane level

The last core control-plane job is changing the fleet safely. Model deployments have a nasty property ordinary services lack: a new version is not just new code, it is *cache-cold*. Cut 100% of traffic to a fresh replica set and you get an instant prefix-cache-miss cliff — TTFT spikes across the board even if the new model is perfectly healthy, because none of the shared prefixes are warm yet. So rollouts must be gradual for two independent reasons: to catch a bad version, and to let caches warm. The figure below shows a control-plane-driven canary that shifts traffic in weighted steps and gates each step on live metrics.

![Timeline of a canary rollout: deploy v2 at 0 percent, canary at 5 percent, gate check on p99 under 500 milliseconds, ramp to 50 percent, promote to 100 percent, or roll back if the gate is breached](/imgs/blogs/llm-control-planes-aibrix-kserve-7.webp)

The pattern is a weighted traffic split with automated gates. Deploy v2 alongside v1 receiving 0% of traffic. Shift 5% to the canary via a weighted `HTTPRoute` (or Knative's `canaryTrafficPercent`). Watch the canary's p99 TTFT, error rate, and — for models — an output-quality signal (a shadow eval, a reward-model score, a guardrail rejection rate), because a model regression is often *silent*: latency is fine, the answers are just worse. If the canary holds the gate for a stabilization window, ramp to 50%, then 100%, then drain v1. If it breaches any gate, roll traffic back to v1 automatically before the canary can reach the whole fleet. In KServe this is a one-line `canaryTrafficPercent` on the `InferenceService`; with the Gateway API it is a weighted `backendRefs` split that a progressive-delivery controller (Argo Rollouts, Flagger) steps automatically.

```yaml
# KServe canary: send 10% to the new revision, watch, then promote.
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-8b
spec:
  predictor:
    canaryTrafficPercent: 10          # 10% to the new model, 90% to the last-good
    model:
      modelFormat: { name: huggingface }
      runtime: kserve-huggingfaceserver
      storageUri: hf://meta-llama/Meta-Llama-3-8B-Instruct   # bump to the new version
```

Two model-specific gotchas that generic canary advice misses. First, size your canary big enough to warm caches representatively but small enough to contain blast radius — 5–10% is the usual sweet spot; 1% often never warms a cache and gives you a misleadingly cold TTFT reading. Second, gate on TTFT and TPOT *percentiles at the canary*, not fleet-wide averages, because a fleet-wide average drowns a canary regression in the healthy majority's numbers. The [request scheduling and preemption](/blog/machine-learning/model-serving/request-scheduling-and-preemption) mechanics inside each engine determine how gracefully a canary degrades under load, which is worth understanding before you set your gate thresholds.

### Progressive delivery with the Gateway API

The one-line `canaryTrafficPercent` is fine when you promote by hand, but production rollouts want the *steps* automated: ramp the weight, watch the gate, ramp again or roll back, with no human in the 2 a.m. loop. On the Gateway API this is a weighted `backendRefs` split driven by a progressive-delivery controller — Flagger or Argo Rollouts — that reads your metrics and steps the weight for you. A Flagger `Canary` that gates on canary-local p99 TTFT captures the whole policy declaratively:

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: llama3-8b
spec:
  provider: gatewayapi:v1        # drive a weighted HTTPRoute split
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama3-8b
  service:
    port: 8000
  analysis:
    interval: 2m                 # evaluate every 2 minutes
    threshold: 3                 # roll back after 3 failed checks
    maxWeight: 50                # ramp to 50% before promoting to 100%
    stepWeight: 10               # +10% of traffic per successful interval
    metrics:
      - name: ttft-p99
        templateRef:
          name: vllm-ttft-p99    # a Prometheus query on the canary pods
        thresholdRange:
          max: 500               # gate: canary p99 TTFT must stay under 500 ms
        interval: 2m
    webhooks:
      - name: load-warmup        # drive representative traffic so caches warm
        url: http://loadtester.test/api/v1/canary
        metadata:
          cmd: "hey -z 2m -q 20 http://llama3-8b-canary:8000/v1/chat/completions"
```

The `load-warmup` webhook matters more for models than for stateless services: without synthetic traffic a low-percentage canary may never receive enough requests to warm its prefix cache, so its TTFT reads misleadingly cold and the gate either false-fails or, worse, false-passes on an unrepresentative sample. Flagger drives the warm-up load, *then* reads the metric, so the gate sees the canary's warm-state latency — the number that will actually govern production once you promote.

#### Worked example: sizing the canary to warm its cache

How big must the canary be to give an honest reading? The gate needs the canary's prefix cache to reach roughly its steady-state warmth before it measures TTFT, and warmth is a function of how many prefix-bearing requests the canary sees per unit time. Suppose the fleet takes 1,000 req/s and your workload's shared prefixes need on the order of a few hundred requests to populate the hot cache entries. A 1% canary sees ~10 req/s — it might take a minute or more to warm, and a short analysis interval will sample it half-cold and report inflated TTFT. A 5% canary sees ~50 req/s and warms in seconds, giving the gate a representative number well within one 2-minute interval. That is the mechanical reason 5–10% is the sweet spot and 1% is a trap: the canary must be large enough to warm within the analysis window, yet small enough that a bad version only touches a slice of users. When traffic is too thin to warm even a 5% canary quickly, the `load-warmup` webhook is not optional — it is the only way to get an honest reading before you ramp.

## Case studies and reported results

**AIBrix at ByteDance.** The AIBrix team's paper, *AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure* (arXiv:2504.03648, 2025), reports the system running across multiple ByteDance production applications since early 2024. The headline result attributed to its distributed KV-cache runtime is a roughly **50% increase in throughput and a 70% reduction in inference latency** from boosting cross-node token reuse — that is, from turning cache misses into cache hits at fleet scale, exactly the mechanic derived above, applied across nodes rather than within one replica. AIBrix's v0.3.0 release (May 2025) added KV-cache offloading, prefix-cache-aware features, fairness routing, and benchmarking tools, and it now develops under the vLLM project. Treat the specific percentages as workload-dependent — they are ByteDance's traffic, not yours — but the *direction* is robust and reproducible: cache-aware routing plus a distributed KV pool moves both throughput and latency in the right direction, hard.

**The Gateway API Inference Extension going mainstream.** Kubernetes' own blog introduced the Gateway API Inference Extension in June 2025 as the standard way to do inference-aware routing, and multiple gateways shipped support in short order — Istio, Envoy Gateway, kgateway, and agentgateway all implement the `InferencePool`/EPP contract. The significance is less any single benchmark and more the *standardization*: LLM-aware routing is now a portable Kubernetes primitive rather than a bespoke service every team rebuilds. If you are starting fresh in 2026, building on this extension is the safe default because it decouples your routing policy from any one vendor's gateway.

**KServe and llm-d.** KServe became a CNCF incubating project in November 2025 and its v0.16 release integrated llm-d to bring disaggregated serving, prefix caching, and intelligent scheduling to its `LLMInferenceService`. KServe's own writing frames this as "best of both worlds" — KServe's mature serving platform (multi-framework, Knative autoscaling, canary) plus llm-d's LLM-specific distributed inference. It is a good example of the convergence pattern: rather than each project reimplementing KV-aware routing, the platform layer (KServe) adopts the primitive layer (llm-d, itself built on the Gateway API extension).

**NVIDIA Dynamo at frontier scale.** NVIDIA's Dynamo technical blog reports up to **30× more requests served** on DeepSeek-R1 on Blackwell GB200 NVL72 by combining disaggregated prefill/decode with the KV-aware Smart Router and tiered KV Block Manager. The 30× figure bundles disaggregation (the biggest factor at that scale) with control-plane routing; the routing contribution alone is smaller but real. Dynamo's NIXL transfer library has been adopted across llm-d, vLLM, and SGLang, which is the clearest signal that the disaggregation-plus-smart-routing design is where large-scale LLM serving is heading. For the disaggregation half of that story, see the [prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) deep-dive.

**The ecosystem is converging, not fragmenting.** The most useful pattern across all four projects is that they are increasingly building on the *same* primitives rather than competing implementations of them. KServe — which grew out of the Kubeflow/KFServing lineage and counts organizations like IBM, Bloomberg, and Red Hat (which ships it inside OpenShift AI) among its long-standing users and contributors — chose to *integrate* llm-d for its LLM path rather than reinvent KV-aware routing. llm-d, in turn, builds its routing layer on the Gateway API Inference Extension rather than a bespoke gateway, and pulls in Dynamo's NIXL for KV transfer. AIBrix moved under the vLLM project umbrella and contributes its LLM-aware routing and autoscaling patterns back to that community. The through-line is that "control plane for LLMs" has stopped being a dozen incompatible in-house services and started being a small set of shared, portable pieces — the Gateway API contract for routing, vLLM's metrics for load signals, NIXL for KV movement — that each platform composes differently. For a team choosing today, that convergence is the good news: whichever project you pick, you are betting on the same underlying primitives, so the migration cost of changing your mind later is far lower than the vendor-lock-in era of 2023–2024 would suggest.

### Named-hardware before → after

Pulling the routing argument down to a concrete fleet, here is a representative before/after for an 8× H100 80GB chat deployment serving Llama-3-8B with a shared ~1,800-token system prompt and multi-turn sessions, at a load that pushes the fleet to ~70% utilization. The "before" is a plain Kubernetes Service (round-robin, no engine-aware signal); the "after" is a cache- and load-aware control plane with KV-utilization autoscaling. Numbers are illustrative but consistent with the mechanics and with published AIBrix/Gateway-API directional results — frame them as "what the model predicts," not a benchmark you can cite verbatim:

| Metric | Before: round-robin Service | After: cache + load-aware control plane |
|---|---|---|
| Hardware | 8× H100 80GB | 8× H100 80GB (same) |
| Prefix cache hit rate | ~11% | ~85% |
| p50 TTFT | ~310 ms | ~55 ms |
| p99 TTFT | ~1,900 ms | ~180 ms |
| Effective throughput (req/s) | ~1.0× (baseline) | ~1.5–1.7× |
| GPU compute wasted on re-prefill | high (89% of reqs re-prefill) | low (15% of reqs re-prefill) |
| Autoscale reaction | CPU-based, minutes late | KV-based, seconds |

The point of the table is not the exact digits; it is that *every improvement came from the layer above the engines*, on identical hardware. The engines were already optimal. The control plane is where the remaining order-of-magnitude in tail latency was hiding.

## When to use this (and when not to)

A control plane is not free. It is more moving parts, more custom resources, more ways for a routing weight or an autoscaler threshold to be subtly wrong at 2 a.m. Be honest about when you need it.

**You do not need a control plane when:**

- **You run a single replica, or a tiny fixed fleet (2–3) with uniform traffic.** With one replica there is nothing to route between; a plain `Service` or even a direct `Deployment` port is correct. Adding a router here is pure overhead and one more thing to page you.
- **Your requests are uniform and short.** If every request is a ~100-token classification with a ~10-token output, request-cost variance is low, $C_S^2$ is small, and round-robin is genuinely fine. The queuing tax you are paying is negligible.
- **You have no prefix reuse.** If prompts are all unique with no shared system prompt and no multi-turn history, cache-aware routing has nothing to cache-hit on. The locality lever is zero; a plain least-request balancer captures most of the available win.
- **You are pre-product-market-fit and traffic is a trickle.** Ship a single vLLM behind a `Service`, watch your metrics, and add a control plane the day the mechanics above start biting. Premature control-plane complexity has killed more velocity than it has saved latency.

**You need a control plane when:**

- **You run a real fleet (say 4+ replicas) with variable request costs** — the queuing tax is now large and load-aware routing pays for itself immediately. This is the cheapest, highest-value first step; do it before anything fancier.
- **Your workload has meaningful prefix reuse** — chat with a shared system prompt, agents with a fixed tool schema, RAG with repeated context. Cache-aware routing is the single biggest lever, worth 3–5× on TTFT.
- **You serve many models or many LoRA adapters** — you need model/adapter-aware routing and high-density packing, which a `Service` cannot express at all.
- **You have real SLOs and real cost pressure** — you need autoscaling on the right signal (KV/queue, not CPU), safe canary rollouts, and fairness/priority under load. These are control-plane features by definition.

A pragmatic adoption path: start with least-request routing (cheapest, biggest variance win), add KV-utilization autoscaling next (stop scaling on CPU), then layer cache-aware routing (biggest TTFT win for chat), and only reach for LoRA-aware routing and distributed KV pooling when your fleet and multi-tenancy actually demand them. Do not deploy all four on day one; add each rung when its mechanic starts hurting you, and measure before and after so you can prove the rung earned its complexity.

## Key takeaways

- **A plain Kubernetes `Service` is the wrong load balancer for LLMs.** It balances on connection/request count, blind to request-cost variance, KV-cache locality, and loaded adapters — the three things that dominate LLM serving cost.
- **Request-cost variance is the root cause, and it is quantifiable.** The Pollaczek–Khinchine formula shows queueing delay scales with $(1 + C_S^2)$; LLM workloads have $C_S^2$ of 5–20, so round-robin eats a 6–21× variance penalty that load-aware routing largely removes by pooling the queues.
- **Cache-aware routing is the biggest single lever for chat/agent workloads.** Under prefix-agnostic routing the realized hit rate collapses toward $r/N$ and gets *worse* as you scale out; routing to the prefix owner restores it toward $r$, typically a 3–5× TTFT win on identical hardware.
- **One bad routing decision can blow the SLA by itself** — a queue depth of 3 can add ~600 ms of head-of-line TTFT. Always blend cache locality with live load so the prefix owner is not overloaded.
- **Autoscale on KV-cache utilization and queue depth, never CPU.** On an LLM replica the CPU idles while the GPU queue drowns; a CPU-based HPA reacts minutes late. Scale up fast, scale down slowly, and respect the cold-start warm-up tax.
- **Roll out gradually for two reasons: to catch regressions and to warm caches.** A 100% cutover to a fresh replica set causes an instant cache-miss cliff even when the new model is healthy. Canary at 5–10%, gate on canary-local TTFT/quality percentiles, roll back automatically.
- **The Gateway API Inference Extension (`InferencePool` + EPP) standardized LLM routing.** It is the portable Kubernetes primitive to build on; AIBrix, KServe, and llm-d all converge on it.
- **Pick the project by center of gravity, not hype.** KServe = platform for all models; AIBrix = opinionated LLM-first platform; llm-d = standards-based assembly; NVIDIA Dynamo = disaggregation at frontier scale. They increasingly interoperate on the same foundation.
- **Do not add a control plane before you need it.** Single replica, uniform short requests, or no prefix reuse? A plain `Service` is correct. Add each routing/autoscaling rung when its mechanic starts costing you, and measure the win.

## Further reading

- **AIBrix team**, *AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure* (arXiv:2504.03648, 2025) — the distributed KV pool, LLM-aware gateway, autoscaler, and LoRA management, with ByteDance production results. See also the AIBrix docs and v0.3.0 release notes.
- **Kubernetes Gateway API Inference Extension** — official docs (`gateway-api-inference-extension.sigs.k8s.io`) for `InferencePool`, `InferenceObjective`, and the Endpoint Picker, plus the Kubernetes blog "Introducing Gateway API Inference Extension" (June 2025).
- **KServe documentation** — `InferenceService` and `LLMInferenceService` CRDs, Knative serverless/scale-to-zero, ModelMesh, and the KServe + llm-d integration writeup.
- **llm-d** — the project site (`llm-d.ai`) and Red Hat's "Kubernetes-native distributed inferencing" articles on KV-cache-aware routing and disaggregated prefill/decode.
- **NVIDIA Dynamo** — the NVIDIA technical blog "Introducing NVIDIA Dynamo" and the Dynamo 1.0 production writeup, covering the Smart Router, SLO Planner, KV Block Manager, and NIXL.
- **Mitzenmacher, M.**, *The Power of Two Choices in Randomized Load Balancing* (1996/2001) and **Azar, Broder, Karlin, Upfal**, *Balanced Allocations* (1994) — the queuing theory behind why load-aware routing beats round-robin.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving), [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention), [multi-LoRA and adapter serving](/blog/machine-learning/model-serving/multi-lora-and-adapter-serving), [GPU scheduling, MIG, and autoscaling](/blog/machine-learning/model-serving/gpu-scheduling-mig-and-autoscaling), and [choosing your serving stack](/blog/machine-learning/model-serving/choosing-your-serving-stack).

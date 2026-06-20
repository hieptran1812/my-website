---
title: "Rate Limiting, Quotas, and Abuse Protection: Keeping the API Up When Traffic Turns Hostile"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Design rate limits and quotas that protect capacity, share it fairly, and contain abuse — with the token-bucket math, a 429 plus Retry-After contract, distributed counters, and the defenses a counter alone cannot provide."
tags:
  [
    "api-design",
    "api",
    "rest",
    "rate-limiting",
    "security",
    "http",
    "token-bucket",
    "abuse-protection",
    "quotas",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/rate-limiting-quotas-and-abuse-protection-1.png"
---

At 09:14 on a Tuesday, our payments service started timing out. Not crashing — timing out, which is worse, because every client kept retrying. The database was pinned at 100% CPU. The on-call dashboard showed `POST /payments` running at fourteen times its normal rate, all from a single integration partner. A deploy on *their* side had introduced a retry loop with no backoff: every failed call spawned three more, and every one of those failed too, because the database was now too busy to answer anyone. The partner was not malicious. They had a bug. But our API had no way to say *"slow down"* — no limiter, no `429`, no `Retry-After`. So instead of one partner degrading, *everyone* degraded. The checkout page for a completely unrelated merchant in a different country went down because one partner's retry loop ate the shared database.

That outage is the entire argument for this post in one sentence: **an API with no rate limit has no way to protect itself, and a backend you cannot protect is a backend that fails for everyone the moment any one caller misbehaves.** The fix was not "add more servers." We had servers. The fix was a contract: a limit per caller, a clear `429 Too Many Requests` when they exceed it, and a `Retry-After` header that tells them exactly how long to wait. With that in place, a buggy partner degrades *only themselves*. The blast radius shrinks from "the whole platform" to "the one caller who is hammering us."

This post is about designing that contract well. We will start with *why* you rate-limit at all (it is four distinct reasons, and conflating them leads to bad designs), separate a **rate limit** from a **quota** (they are not the same thing, and a serious API needs both), then go deep on the four canonical algorithms with the actual math — **token bucket**, **leaky bucket**, **fixed window**, and **sliding window** — including the derivation of why a naive fixed window lets *twice* your limit through at a clock boundary. We will cover where to enforce limits, what key to count against (and why counting by IP is weak), the response contract clients depend on (`429` + `Retry-After` + the standard `RateLimit` headers), cost-based limiting, the hard problem of doing this across many servers with a shared counter in Redis, and finally the abuse that no counter can stop on its own — bots, credential stuffing, scraping — and the layered defenses for it.

![a flow diagram of a request arriving at the limiter, getting keyed and checked against a token counter, then branching to either an allowed handler call or a 429 with Retry-After that the client backs off and retries](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-1.png)

This sits squarely in the security and trust track of the [Designing APIs That Last](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) series. The recurring frame of that series is that *an API is a contract and a product, not a function call*, and rate limiting is one of the most visible parts of that contract: it is a promise about how much you will serve, encoded in headers a client can read and act on. Get it right and a careful client never even sees a `429`, because your headers told them to slow down first. Get it wrong and you either fail to protect your backend, or you punish well-behaved callers for the sins of bad ones. Throughout, we will keep returning to our running example — the Payments & Orders API — and put a token-bucket limit on `POST /payments`, hand back a real `429` with `Retry-After`, and layer a monthly quota on top.

## Why rate-limit at all: four reasons that pull in different directions

It is tempting to treat rate limiting as one thing — "stop people from sending too many requests" — but there are actually four distinct motivations, and they want different limits, different windows, and sometimes different *keys*. If you do not name them separately, you will build one limiter that does all four jobs badly.

**1. Protect capacity.** Your backend has a finite throughput. A database can sustain so many writes per second; a downstream payment processor has its own limits you must stay under; a single service instance has so many worker threads. When demand exceeds capacity, latency does not degrade gracefully — it falls off a cliff, because queues build, timeouts fire, retries pile on, and you enter the death spiral I described in the intro. A rate limit is a *load shedder*: it converts "the system is overwhelmed and everyone times out" into "callers above their limit get a clean, fast `429`, and everyone under their limit keeps getting served." A fast `429` costs you almost nothing — it never touches the database. That is the whole point: **you spend a tiny bit of work rejecting the excess so the rest of the system stays healthy.**

**2. Ensure fairness across tenants.** A multi-tenant API serves many customers from shared infrastructure. Without per-tenant limits, the noisiest tenant takes as much capacity as it can grab, and the quiet, well-behaved tenants suffer for it — the "noisy neighbor" problem. A per-tenant rate limit is a fairness mechanism: it guarantees that no single caller can consume more than its share, which means one tenant's traffic spike (or bug) cannot starve another. This is why the *key* you limit on matters so much, and why a global limit is almost useless on its own — a global limit of 10,000 req/s does nothing to stop one tenant from using all 10,000.

**3. Contain abuse.** Some traffic is not a well-meaning caller with a bug; it is hostile. Scrapers want to vacuum your catalog. Credential-stuffing bots want to test millions of leaked username/password pairs against your login endpoint. Someone wants to enumerate your `/users/{id}` endpoint to harvest data. A rate limit raises the cost of all of these: a scraper that can make 5 requests per second instead of 5,000 is 1,000× slower, which often makes the attack uneconomical. Rate limiting alone will not *stop* a determined, distributed attacker (we will get to that), but it is the floor that every other defense builds on.

**4. Control cost.** In a usage-billed world, every request costs *you* money: compute, bandwidth, a per-call fee to a downstream provider, an LLM token bill. An unbounded API is an unbounded liability. A quota — a long-window allowance, like 1,000,000 requests per month — caps that liability per customer and ties it to what they pay. This is less about protecting the system *right now* (that is reason 1) and more about protecting the *business* over a billing period.

Notice these pull in different directions. Capacity protection wants a *short* window (protect the next second). Cost control wants a *long* window (bound the month). Fairness wants a *per-tenant* key. Abuse containment sometimes wants a *per-IP* or *per-endpoint* key, because the abuser may not have an API key at all. One limiter cannot serve all four cleanly, which is exactly why mature APIs run several limits at once — and why the first real distinction to draw is between a rate limit and a quota.

## Rate limit vs quota: short-window throughput vs long-window allowance

These two words get used interchangeably, and that sloppiness causes real bugs. They are different in window, in what they protect, and in how they reset.

A **rate limit** caps **throughput over a short window** — requests per second, per minute. Its job is reason 1 and 2: protect live capacity and keep tenants fair *right now*. It resets quickly and continuously (a token bucket refills every fraction of a second). Breaching it is transient and recoverable: wait a moment and you are under the limit again. The right response is `429 Too Many Requests` with a short `Retry-After`.

A **quota** caps **total allowance over a long window** — requests per day, per month. Its job is reason 4 (and sometimes 3): bound cost and enforce the plan a customer pays for. It resets on a calendar boundary (the first of the month, midnight UTC). Breaching it is not transient: you are *out* for the rest of the period, and waiting two seconds does nothing. The right response is still `429`, but with a `Retry-After` measured in hours or days, or a `402 Payment Required` if the fix is to upgrade the plan.

The two are independent and *both* apply. A request must pass the rate limit **and** the quota. You can be well under your monthly quota and still get rate-limited because you sent a burst this second; you can be slow and steady all month and still get quota-blocked on the 28th because you have used your million calls.

![a comparison matrix showing rate limit and quota across window, what each protects, how each resets, and the breach status code, making clear they are independent and both apply](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-6.png)

| Dimension | Rate limit | Quota |
|---|---|---|
| Window | Seconds to a minute | Day to a month |
| Protects | Live capacity, fairness | Cost, the plan tier |
| Typical value | 100 req/s | 1,000,000 req/month |
| Reset | Continuous (refill) | Calendar boundary |
| Breach is | Transient, retry soon | Sustained, retry next period |
| Status code | `429` + short `Retry-After` | `429`/`402` + long `Retry-After` |
| Caller's fix | Back off and retry | Slow down or upgrade |

For our Payments API, a sensible pair is **100 req/s** (the rate limit, protecting the database and the downstream processor) **and 1,000,000 req/month** (the quota, matching the customer's plan). A merchant doing a flash sale might brush the rate limit during the spike — that is fine, they back off and retry. A merchant that has grown past their plan brushes the quota near month-end — that is a sales conversation, not a retry. Same status code on the wire, very different meaning, which is why your error body and your `Retry-After` value must distinguish them.

#### Worked example: which limit did I hit?

A client sends a burst of 150 requests in one second to `POST /payments`. The rate limit is 100 req/s. The first 100 succeed; requests 101–150 get `429` with `Retry-After: 1`. The client waits one second and the remaining 50 go through. Total time: about two seconds. The client never touched the quota — they used 150 of their 1,000,000 monthly calls. Now imagine the same client on the 30th of the month, having already made 999,990 calls, sends 20 requests in a slow trickle (well under 100/s). The first 10 succeed; calls 11–20 get `429` with `Retry-After: 86400` (or a `402` and a message: *"monthly quota of 1,000,000 reached; resets 2026-07-01"*). No amount of backing off helps — the window is the calendar. The two limits caught two completely different problems, and the only reason the client can tell them apart is the body and the `Retry-After`.

## The algorithms, with the math

Now the core of the post. There are four classic algorithms. We will derive the behavior of each, not just describe it, because the differences are exactly in the corner cases — the burst at the start, the spike at a boundary, the memory cost at scale — and you cannot reason about those without the math.

![a matrix comparing token bucket, leaky bucket, fixed window, and sliding window log across burst tolerance, accuracy, memory cost, and implementation complexity](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-2.png)

### Token bucket: the canonical choice

The token bucket is the default for a reason: it cleanly separates two parameters that you usually want to set independently — the **average rate** and the **maximum burst**.

The model is a bucket that holds tokens. It has two parameters:

- **Capacity** $C$: the maximum number of tokens the bucket can hold. This is your **burst** size.
- **Refill rate** $r$: tokens added per second. This is your **average** allowed rate.

The bucket refills continuously at $r$ tokens per second, but never above $C$. A request costs one token (we will generalize to weighted costs later). The rule is exactly:

$$\text{allow a request} \iff tokens \ge 1$$

When a request is allowed, you subtract one token. When it is denied, the bucket is empty and the request gets a `429`.

The elegant part is how refill is computed. You do not run a background timer adding a token every $1/r$ seconds — that is wasteful and hard to scale. Instead you store only two numbers per key: the current token count and the timestamp of the last update. On each request, you compute how many tokens *should* have been added since the last check:

$$tokens \leftarrow \min\!\big(C,\ tokens + r \cdot (t_{now} - t_{last})\big)$$

then set $t_{last} \leftarrow t_{now}$, and apply the allow rule. This is **lazy refill**: tokens accrue mathematically based on elapsed time, computed only when you look. It is $O(1)$ in both time and memory — two numbers per key, a subtraction and a comparison per request. That efficiency is why token bucket scales to millions of keys.

Two limiting behaviors fall straight out of the math. First, over a long horizon $T$, the maximum number of requests you can serve is bounded by the tokens that arrive: at most $C + r \cdot T$. The $C$ is the initial burst you could have banked; the $r \cdot T$ is the steady refill. As $T$ grows, the average rate converges to exactly $r$ — the burst term becomes negligible. So **the long-run rate is the refill rate $r$, and the burst on top of that is the capacity $C$.** Second, if traffic has been quiet, the bucket fills back up to $C$, so a caller who has been idle can fire a full burst immediately — which is usually what you want (it rewards bursty-but-light usage and is friendly to interactive clients).

![a timeline of a token bucket starting full, a burst draining it to empty, a request getting 429 with Retry-After, the bucket refilling at the rate r, and a retry succeeding once a token returns](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-3.png)

#### Worked example: walking a burst through a token bucket

Set $C = 10$ tokens and $r = 5$ tokens/sec for `POST /payments`. That means: sustain 5 payments per second on average, but tolerate a burst of up to 10 at once.

At $t = 0$ the bucket is full: 10 tokens. A client fires a burst of 8 requests effectively simultaneously. Each costs one token, so all 8 are allowed and the bucket drops to 2 tokens. Two more arrive immediately — allowed, bucket now 0. The 11th request in that instant finds $tokens = 0$, so $tokens \ge 1$ is false: **denied, `429`**.

How long until the next request can succeed? We need one token. Refill is $r = 5$/sec, so one token every $1/5 = 0.2$ seconds. The honest `Retry-After` for *one* token is therefore 0.2 s, which you round up to 1 (since `Retry-After` is integer seconds; we will discuss this rounding). After 1 second, $r \cdot 1 = 5$ tokens have refilled (capped at $C = 10$, but we are well under), so the client now has 5 tokens and the retry succeeds. The long-run picture: this client banked a burst of 10, then settled to 5/sec. Over 10 seconds it could serve at most $C + r \cdot T = 10 + 5 \cdot 10 = 60$ requests, an average of 6/sec — slightly above $r$ because of the initial burst, converging to 5/sec as time goes on. Exactly as the math predicted.

Here is the lazy-refill check as code. Notice there is no timer; everything is derived from elapsed time:

```python
import time

class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity          # C, the burst size
        self.refill_rate = refill_rate    # r, tokens per second
        self.tokens = capacity            # start full
        self.last = time.monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        # Lazy refill: add tokens for the elapsed time, capped at C.
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def retry_after_seconds(self, cost: float = 1.0) -> float:
        # Time until enough tokens accrue for the next request.
        deficit = cost - self.tokens
        return max(0.0, deficit / self.refill_rate)
```

The `cost` parameter is the hook for cost-based limiting — an expensive endpoint can charge more than one token. Hold that thought.

### Leaky bucket: smoothing to a constant rate

The leaky bucket inverts the metaphor. Picture a bucket with a hole in the bottom that drains (leaks) at a constant rate. Incoming requests pour *in* from the top; they queue in the bucket and drain out the bottom at the fixed leak rate. If the bucket overflows (the queue is full), new requests are dropped.

The defining property: **output is perfectly smooth at the leak rate, regardless of how bursty the input is.** A token bucket lets a burst through *immediately* (up to $C$); a leaky bucket *flattens* the burst into a steady stream, queuing the excess and releasing it at the constant rate. This is the right model when the thing you are protecting cannot tolerate bursts at all — for example, a downstream provider that hard-rejects anything over exactly $N$/sec, or a hardware system that needs an even cadence.

The trade-off is latency and the queue. Because the leaky bucket buffers and releases at a fixed rate, a request that arrives during a burst *waits in the queue* before it is served — it does not get a fast `429`, it gets a delay. That added latency is sometimes acceptable (smoothing a write stream) and sometimes terrible (a user waiting on an interactive request). And the queue itself is state you must bound and manage. There is a "leaky bucket as a meter" variant (the Generic Cell Rate Algorithm) that behaves almost exactly like a token bucket and rejects rather than queues; in practice, when people say "leaky bucket" for an API they often mean either the queuing variant or the GCRA meter, so be specific about which.

For most web APIs, **token bucket beats leaky bucket** because clients generally prefer a fast, explicit `429` they can retry over an invisible delay, and because bursty-but-light traffic (a user clicking around) is the common, benign case you *want* to allow. Reach for leaky bucket when smoothing is the explicit goal.

It is worth being precise about the relationship, because the two algorithms are closer than the metaphors suggest. The token bucket and the rejecting (meter) form of the leaky bucket are, mathematically, near-duals: both enforce an average rate $r$ with a burst allowance, and both can be computed in $O(1)$ with two stored numbers. The visible difference is *what happens to the excess*. The token bucket **rejects** the overflow immediately (a fast `429`, the client decides what to do). The queuing leaky bucket **delays** the overflow (the request waits in a buffer and is served later at the constant rate). The meter-style leaky bucket (the Generic Cell Rate Algorithm, GCRA) rejects like a token bucket but expresses its state as a single "theoretical arrival time" timestamp rather than a token count — it is what several high-performance limiters actually implement because one timestamp is even cheaper to store and update atomically than a token-plus-timestamp pair. For an API designer the practical takeaway is simple: if you want callers to retry on their own schedule, choose a *rejecting* algorithm (token bucket or GCRA); if you want to *absorb* a burst and serve it smoothly without ever rejecting (at the cost of added latency and a bounded queue), choose the *queuing* leaky bucket. Most public APIs want the former, because a visible `429` the client controls beats an invisible delay the client cannot see or reason about.

### Fixed window: simple, and the 2× boundary bug

The fixed-window counter is the most naive algorithm and the one most people reach for first. Divide time into fixed windows (say, one minute), keep a counter per key per window, increment on each request, and reject when the counter exceeds the limit $L$. At the window boundary the counter resets to zero.

It is genuinely cheap — one integer per key, an increment, a comparison. And it is genuinely *wrong* at the edges, in a way that is worth deriving carefully because it bites real systems.

The problem: the counter resets on a wall-clock boundary, but a *client's* burst does not respect your boundary. Consider a limit of $L = 100$ requests per 60-second window, with windows aligned to clock minutes.

- A client sends 100 requests in the last second of one window, from `00:00:59.000` to `00:00:59.999`. The counter for that window hits 100. All allowed.
- The clock ticks to `00:01:00`. The window resets; the counter is now 0.
- The client sends another 100 requests in the first second of the next window, `00:01:00.000` to `00:01:00.999`. The counter goes 0 → 100. All allowed.

In the **two-second span** from `00:00:59` to `00:01:01`, the client successfully sent **200 requests** — twice the limit. Generalize it: in any sliding 60-second interval that straddles a window boundary, a client can send up to $2L$ requests, because each of the two windows independently permits $L$. The worst-case instantaneous rate over a window-sized interval is **double** what you intended. If you sized $L$ to protect the database at exactly 100/min, you have just let 200/min hit it across the boundary — and that is precisely the spike that takes the database down.

![a before-and-after contrast showing a fixed window letting 200 requests through across a clock boundary by allowing 100 on each side, versus a sliding window weighting the prior window to hold the true rate and reject the excess](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-4.png)

#### Worked example: the fixed-window double burst on /payments

Suppose `POST /payments` uses a fixed window of 100 requests per minute, and the database starts to suffer above ~120 writes/min. In normal operation the limiter holds you at 100/min — safe. Then a client (maliciously, or just by aligning a cron job to the minute) sends 100 requests at `12:30:59.9` and 100 more at `12:31:00.1`. Both windows permit their 100 because the reset at `12:31:00.000` wiped the counter. The database sees 200 writes in roughly 200 milliseconds — far above the 120/min danger line — and latency spikes for *every* tenant, not just this one. The limiter reported success the whole time. It did its job exactly as written, and its job was wrong. This is the canonical reason not to ship a fixed window for anything where the boundary spike matters.

### Sliding window: fixing the boundary at a cost

There are two ways to fix the boundary bug, trading memory for accuracy.

**Sliding window log.** Store a timestamp for *every* request in the current trailing window (e.g., the last 60 seconds). On each new request, drop timestamps older than the window, count what remains, and allow only if the count is below $L$. This is **exact** — it counts precisely how many requests happened in the true trailing 60 seconds, with no boundary artifact at all. The cost is memory: $O(L)$ timestamps per key (up to $L$ entries), plus the work to prune old ones. At a limit of 100/min that is fine; at a limit of 100,000/min across millions of keys it is ruinous. Use the log when limits are small and exactness matters.

**Sliding window counter.** A clever approximation that keeps $O(1)$ memory. Keep two fixed-window counters: the current window and the previous one. Estimate the rate in the trailing window by weighting the previous window's count by how much of it still overlaps the trailing window:

$$\text{estimate} = c_{cur} + c_{prev} \cdot \frac{\text{overlap of previous window}}{\text{window size}}$$

Concretely: if you are 25% into the current minute, then the trailing 60 seconds covers all of the current window so far plus the last 75% of the previous window. So you count $c_{cur} + 0.75 \cdot c_{prev}$ and compare to $L$. Walk the boundary attack through it: at `12:31:00.1` you are essentially 0% into the new window, $c_{cur} \approx 0$, $c_{prev} = 100$, overlap weight $\approx 1.0$, so the estimate is $0 + 1.0 \cdot 100 = 100$ — already at the limit, so the next request is rejected. The 2× spike is gone, using only two counters per key. The estimate is a smooth approximation (it assumes the previous window's requests were spread evenly, which is not always true), but it is close enough for capacity protection and dramatically cheaper than the log. This is what Cloudflare and many gateways actually use.

Here is the comparison that should drive your choice:

| Algorithm | Allows bursts? | Boundary accuracy | Memory per key | When to use |
|---|---|---|---|---|
| Token bucket | Yes, up to $C$ | Good (smooth average) | $O(1)$, two numbers | The default; want bursts + average |
| Leaky bucket | No (smooths output) | Good (hard ceiling) | $O(1)$, queue depth | Must smooth to a constant rate |
| Fixed window | 2× at boundary | Poor (boundary bug) | $O(1)$, one counter | Almost never for capacity; cheap counting |
| Sliding window log | No | Exact | $O(L)$, all timestamps | Small limits, exactness required |
| Sliding window counter | Slight | Good (approximate) | $O(1)$, two counters | High-scale, want fixed-window cost + accuracy |

The honest default for an API: **token bucket** for the rate limit (you want to allow benign bursts and the $O(1)$ cost scales), **sliding window counter** at the edge/gateway for coarse high-scale limits, and a plain counter for the *quota* (a monthly count genuinely is a fixed window — and the boundary bug does not matter on a monthly scale because nobody cares about a 2× spike across a month boundary; the window is the calendar).

## Where to enforce: edge for coarse, service for fine

A limit is only as good as the place it runs. Put it too far in and the flood has already done damage; put it too far out and it cannot know what a request actually *costs*. The answer is layers, each enforcing the limit it is best positioned to enforce.

![a layered stack showing limits enforced at the CDN and WAF, then a coarse per-IP flood cap at the edge, then a per-key plan-tier limit at the gateway, then a fine cost-weighted limit in the service, protecting the scarce datastore](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-5.png)

**At the edge / CDN / WAF (coarse, volumetric).** This is the cheapest place to reject the most obvious garbage: a flood of requests from one IP, a known-bad bot signature, a volumetric denial-of-service. The edge runs before your gateway and before your services, so a request killed here costs you nothing downstream. But the edge knows almost nothing about your domain — it sees an IP, a path, maybe a header. So its limits are necessarily *coarse*: "no more than N requests per IP per second," "block this user-agent." It cannot know that `POST /payments` is expensive while `GET /health` is free.

**At the gateway (per-key, plan-tier).** Your API gateway authenticates the request, so it knows *who* the caller is — the API key, the tenant, the plan tier. This is the natural home for the per-tenant fairness limit and the plan-based quota: "this API key is on the Pro plan, so 100 req/s and 1,000,000 req/month." The gateway is the single choke point every request passes through, which makes the counter easy to share and the policy easy to express. We cover the gateway's full responsibilities in the dedicated [API gateways post](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern); here, the relevant point is that the gateway is where *most* API rate limiting lives.

**In the service (fine, cost-weighted).** Some limits depend on knowledge only the service has: that this particular `POST /payments` will fan out to three downstream calls, or that this search query has an unbounded result set, or that this user is currently in a fraud-review state and should be throttled harder. The service is the only layer that can charge *different costs for different requests* (cost-based limiting, below). The downside is that a request rejected here has already traveled through the edge, the gateway, auth, and routing — it cost you more to reject. So you do this only for the fine-grained limits the outer layers cannot express.

The rule of thumb: **shed volume early, enforce policy in the middle, charge by cost late.** Coarse and cheap at the edge; per-identity at the gateway; per-cost in the service. Do not try to do cost-based per-endpoint limiting at the CDN (it does not have the information), and do not rely solely on the service for flood protection (the flood has already cost you everything by the time it gets there).

## The keys to limit on: identity beats IP

A limiter counts requests against a **key**. Choosing the wrong key is the most common rate-limiting design mistake, because it quietly defeats the purpose: a fairness limit keyed on the wrong thing is not fair, and an abuse limit keyed on the wrong thing is trivially bypassed.

The candidates, roughly from best to worst for an authenticated API:

- **Per API key / per tenant.** The strongest key for a B2B or platform API. It maps directly to *who is paying and who is responsible*, which is exactly what fairness and quota want. A caller cannot escape it without obtaining a new key (which you control). Use this as the primary key for authenticated traffic.
- **Per user.** For end-user-facing APIs, the authenticated user id. Good for fairness between users and for catching one compromised account hammering an endpoint.
- **Per endpoint, cost-weighted.** A *modifier* on the above: the same key gets different limits (or token costs) per endpoint, because `POST /payments` and `GET /health` are not equally expensive. More on weighting next.
- **Per IP address.** The fallback when you have no identity — *unauthenticated* endpoints like login, signup, password reset. Necessary there, but **weak**, and you must understand why.

IP-only limiting is weak for several concrete reasons, and these are not theoretical:

1. **Many users share one IP.** Behind a corporate NAT, a university, a mobile carrier's CGNAT, or a cloud provider's egress, thousands of distinct, legitimate users present the *same* IP. An IP-based limit either has to be set so high it provides no protection, or it punishes all those innocent users for the volume of the group. Mobile carriers in particular route huge populations through a handful of addresses.
2. **One attacker controls many IPs.** A botnet, a residential-proxy network, or a few dollars of cloud instances gives an attacker thousands of source IPs. A per-IP limit of 10/s means nothing to an attacker with 10,000 IPs — they get 100,000/s in aggregate while staying perfectly "compliant" per IP. This is exactly how credential-stuffing-at-scale and distributed scraping evade naive IP limits.
3. **IPs are not stable.** A legitimate mobile user's IP changes as they move between cells and networks, so a per-IP counter does not reliably track *one user* over time.

The takeaway: **for authenticated traffic, key on identity (API key/tenant/user), not IP.** Use IP only where you genuinely have no identity (the unauthenticated edge), and even there, treat it as one signal among several rather than the whole defense. The cleanest design is a *layered* key strategy: a coarse per-IP flood cap at the edge to stop the dumbest attacks, plus a precise per-tenant limit at the gateway for everything authenticated.

## The response contract: 429, Retry-After, and the RateLimit headers

This is the part of rate limiting that is *API design*, not infrastructure. The whole point of a well-behaved limit is that a careful client can read your signals and **self-throttle before they ever get rejected.** That requires a clear, standard response contract. There are two layers to it: what you return when you reject (the `429`), and what you return on *every* response so clients can see how close they are (the `RateLimit` headers).

### The 429 and Retry-After

When a request exceeds the limit, return **`429 Too Many Requests`**, defined in [RFC 6585](https://www.rfc-editor.org/rfc/rfc6585). This is the correct, dedicated status code — not `503 Service Unavailable` (which means the *server* is broken, not that *you* sent too much), and absolutely not `200 OK` with an error in the body (which lies to every cache, proxy, and client library between you and the caller). The status code is the truth; tell it. (We go deep on choosing honest status codes in the [status codes post](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx).)

The single most important header on a `429` is **`Retry-After`**. It tells the client exactly how long to wait before trying again, in one of two forms: an integer number of seconds (`Retry-After: 2`) or an HTTP date (`Retry-After: Wed, 01 Jul 2026 00:00:00 GMT`). Without it, a client is guessing — and a guessing client either retries too soon (making the overload worse) or too late (degrading the user for no reason). *With* it, a well-written client library waits precisely the right amount and the system converges. `Retry-After` is the difference between a `429` that *helps* the system recover and a `429` that triggers a thundering herd of premature retries.

Here is the full wire transcript for a rate-limited `POST /payments`:

```http
POST /v1/payments HTTP/1.1
Host: api.shopflow.example
Authorization: Bearer <token>
Idempotency-Key: a1f3c9d2-7b6e-4a10-9f2c-3e8d11b0c4aa
Content-Type: application/json

{ "order_id": "ord_8123", "amount": 4999, "currency": "usd" }
```

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/problem+json
Retry-After: 2
RateLimit-Limit: 100
RateLimit-Remaining: 0
RateLimit-Reset: 2

{
  "type": "https://api.shopflow.example/problems/rate-limited",
  "title": "Too Many Requests",
  "status": 429,
  "detail": "Rate limit of 100 requests/second exceeded for this API key. Retry after 2 seconds.",
  "limit_type": "rate",
  "retry_after_seconds": 2
}
```

A few deliberate choices in that response. The body is `application/problem+json` ([RFC 9457](https://www.rfc-editor.org/rfc/rfc9457)), the same machine-readable error envelope the rest of the API uses, so a client's error handling is uniform. The `detail` field is human-readable and *actionable* — it names the limit, the key it applies to, and what to do. The custom `limit_type` field distinguishes "rate" from "quota," so a client can tell whether to retry in seconds or give up until next month. And `Retry-After` agrees with the body, so a client that only reads headers and a client that only reads the body both get the same answer.

#### Worked example: a 429 with Retry-After and correct client backoff

A merchant's checkout service sends 130 payments in a one-second spike against a 100/s limit. Requests 101–130 come back `429` with `Retry-After: 1`. A *naive* client retries all 30 immediately — and gets 30 more `429`s, and retries again, producing the thundering herd that keeps the limiter pinned. A *correct* client honors `Retry-After`, waits one second, and — critically — adds a small random **jitter** so that 30 retries do not all fire at the exact same instant (which would just re-create the spike). Here is the right retry loop:

```python
import random, time
import httpx

def post_payment(client, body, idem_key, max_attempts=5):
    headers = {
        "Authorization": "Bearer <token>",
        "Idempotency-Key": idem_key,   # same key on every retry: safe to repeat
        "Content-Type": "application/json",
    }
    for attempt in range(max_attempts):
        resp = client.post("/v1/payments", json=body, headers=headers)
        if resp.status_code != 429:
            return resp
        # Honor the server's instruction, then add jitter to avoid a herd.
        wait = int(resp.headers.get("Retry-After", "1"))
        wait += random.uniform(0, 0.5)
        time.sleep(wait)
    raise RuntimeError("payment rate-limited after retries; surface to caller")
```

Two details make this safe rather than dangerous. First, it honors `Retry-After` instead of guessing. Second, it sends the **same `Idempotency-Key` on every retry**, which means a retry of a request that *actually succeeded* on the server (but whose response was lost) returns the original result instead of charging the card twice. Rate-limit retries and idempotency are inseparable: any time you tell a client to retry, you are implicitly promising that retrying is *safe*, which it only is if the write is idempotent. We cover that machinery in depth in [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions); the link between the two is exactly the `Retry-After` you send.

### The RateLimit headers (the IETF draft) — let clients self-throttle

A `429` tells a client they have *already* been rejected. Better is to tell them how close they are *before* they hit the wall, on every response, so a well-behaved client throttles itself and never gets a `429` at all. That is what the standard `RateLimit` headers are for.

These headers have a real history worth knowing so you use the right names. The de-facto convention for years was `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`, popularized by GitHub, Twitter, and others (the `X-` prefix marks them as non-standard extensions). The IETF has since worked to standardize them as an HTTP working-group draft, "RateLimit header fields for HTTP," which defines un-prefixed fields. The draft has evolved across versions — earlier drafts used three separate fields (`RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset`); later drafts consolidated into a single structured-field `RateLimit` header plus a `RateLimit-Policy` header describing the quota policy. Because the draft is still evolving, the pragmatic move today is to **emit both**: the widely-deployed three-field form and, where you can, the newer structured form, so clients written against either work.

The three-field semantics (the form most clients actually parse today):

- `RateLimit-Limit`: the ceiling for the current window (e.g., `100`).
- `RateLimit-Remaining`: how many requests remain in the current window (e.g., `42`).
- `RateLimit-Reset`: when the window resets — in the draft, the number of *seconds* until reset (e.g., `2`). Note: GitHub historically used a Unix epoch timestamp here, not a delta, which is a real interoperability wrinkle — read each API's docs.

A successful response carrying these headers looks like:

```http
HTTP/1.1 201 Created
Content-Type: application/json
RateLimit-Limit: 100
RateLimit-Remaining: 73
RateLimit-Reset: 1
Location: /v1/payments/pay_9f2c

{ "id": "pay_9f2c", "status": "succeeded", "amount": 4999, "currency": "usd" }
```

A client that watches `RateLimit-Remaining` drop toward zero can pre-emptively slow down — spreading its requests out so it never trips the limit. This is the cooperative ideal: the server publishes its state, the client respects it, and nobody has to send or handle a `429` in steady operation. The `429` becomes the safety net for clients that *do not* cooperate, not the primary control surface for the ones that do. This is developer experience as a first-class concern, the theme of the [design-for-the-caller post](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal): a limit the client can *see* is a limit the client can *respect*.

## Cost-based limiting: not all requests are equal

Counting every request as "one" is a fiction. A `GET /health` costs almost nothing; a `POST /payments` fans out to a fraud check, a ledger write, and a call to an external processor; a `GET /reports/annual?expand=line_items` might scan millions of rows. A limit of "100 requests/second" that treats these identically either throttles the cheap calls needlessly or fails to protect against the expensive ones.

**Cost-based limiting** (also called weighted rate limiting) assigns each endpoint a *cost* and debits the bucket by that cost. This is exactly the `cost` parameter in the token-bucket code earlier: an `allow(cost=1)` for a cheap read, an `allow(cost=10)` for an expensive write. The bucket's capacity $C$ and refill rate $r$ are now denominated in *cost units* rather than raw requests. A client with a bucket of 100 units/sec might make 100 cheap reads, or 10 expensive payments, or any mix summing to 100 units — and the bucket naturally protects the underlying resource because expensive operations drain it faster.

This maps cleanly onto the resource the limit is *really* protecting. If a payment costs the database 10× what a health check costs, weighting the payment at 10 makes the limit track actual load rather than raw request count. It also gives you a principled way to handle the noisy expensive endpoint: a client cannot dodge protection by sending fewer but heavier requests.

The most rigorous application of this idea is **GraphQL query cost analysis**. In REST, the endpoint is a coarse proxy for cost. In GraphQL, a single query can request arbitrarily nested, arbitrarily large result sets — `GET /graphql` with a query that asks for every order, each with every line item, each with the full product — so a per-request limit is meaningless. Instead, GraphQL APIs compute a *query cost* by walking the query's structure (each field has a cost; lists multiply by an expected or requested page size) and rate-limit against the *summed cost*, rejecting queries whose cost exceeds the budget. GitHub's GraphQL API does exactly this with a published point system. The general principle is the same as our weighted token bucket: **charge the limit by what the request actually costs, not by counting requests.** The mechanics of why GraphQL forces this (and the N+1 trap behind it) are in the [GraphQL post](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap).

A worked weighting for the Payments API:

| Endpoint | Cost (units) | Rationale |
|---|---|---|
| `GET /health` | 0 | Trivial; do not count it |
| `GET /orders/{id}` | 1 | Single indexed read |
| `GET /orders?expand=line_items` | 3 | Join + larger payload |
| `POST /payments` | 10 | Fraud check + ledger + external processor |
| `POST /refunds` | 10 | Same downstream fan-out |
| `GET /reports/monthly` | 25 | Large scan + aggregation |

With a bucket of 100 units/sec, a client can do 100 health checks, or 10 payments, or one monthly report plus 75 cheap reads — and in every case the *load* on the backend is bounded, which is the actual goal.

## Distributed rate limiting: the shared-counter race

Everything so far assumed one limiter with one counter. Real APIs run many instances of every service behind a load balancer. If each instance keeps its *own* token bucket, the effective limit is multiplied by the number of instances: a 100/s limit across 10 instances becomes 1,000/s in aggregate, because each instance independently allows 100/s and the load balancer spreads traffic across them. Worse, the limit becomes *non-deterministic* — it depends on how the load balancer happened to route, on how many instances are currently up, on autoscaling. A client doing exactly 100/s might get rejected or not, depending entirely on routing luck. That is a broken contract.

The fix is a **shared counter** in a fast central store — almost always **Redis**, because it is in-memory, fast, and has the atomic primitives this needs. Every instance consults the same counter, so the global limit is enforced regardless of which instance handles a request.

![a flow showing two application pods both sending a request to one atomic check-and-decrement script against a single Redis counter holding one token, where one pod wins and is allowed while the other loses and gets a 429](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-7.png)

But a shared counter introduces a classic concurrency bug: the **read-then-write race**. Suppose the bucket has exactly one token left, and two instances each handle a request at the same instant. The naive sequence is: instance A reads `tokens = 1`, instance B reads `tokens = 1` (before A has written), both see $tokens \ge 1$, both decide "allow," both write `tokens = 0`. **Two requests were admitted on the strength of one token.** Across many instances and many keys, these races leak through more traffic than your limit allows — exactly when you are under load, which is exactly when you need the limit to hold.

The race exists because "read the count, decide, write the new count" is three steps, and another actor can interleave between them. The fix is to make the check-and-decrement **atomic** — one indivisible operation that no other request can interleave with. Redis gives you two clean ways:

1. **Atomic integer operations.** For a simple fixed-window counter, `INCR` is atomic: it increments and returns the new value in one step, so the check ("is the new value over the limit?") sees a consistent count. You pair it with `EXPIRE` to reset the window. This is the textbook distributed fixed-window limiter.

2. **A Lua script (the robust choice).** For a token bucket, the logic is "read tokens and last-update, compute lazy refill, check, decrement, write back" — multiple steps that must be atomic *together*. Redis runs a Lua script atomically: nothing else executes while the script runs. So you move the entire token-bucket decision into a script and call it as one operation. This is how production limiters (and libraries like the Generic Cell Rate Algorithm implementations) do it.

```python
# A token-bucket check as one atomic Redis Lua script.
# KEYS[1] = bucket key; ARGV = capacity, refill_rate, now, cost
LUA = """
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

local b = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(b[1])
local ts     = tonumber(b[2])
if tokens == nil then tokens = capacity; ts = now end

-- lazy refill, capped at capacity
local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)
return { allowed, tokens }
"""

def allow(redis_client, key, capacity, rate, cost=1):
    now = time.time()
    allowed, tokens = redis_client.eval(LUA, 1, key, capacity, rate, now, cost)
    return bool(allowed), tokens
```

Because the whole read-refill-check-decrement runs inside the script, the two-instances-on-one-token race cannot happen: the second script execution sees the decrement the first one made. The `EXPIRE` keeps idle keys from accumulating forever (memory hygiene at scale).

#### Worked example: the race that leaks exactly when you need the limit

Walk the bug through concretely so the fix is undeniable. The bucket key holds `tokens = 1`. Ten service instances are behind the load balancer, and a coordinated burst lands such that three of them each pick up a request for the same tenant within the same millisecond. With the **naive** read-then-write pattern: instance A does `GET tokens` → `1`; instance B does `GET tokens` → `1` (A has not written yet); instance C does `GET tokens` → `1` (neither has written). All three evaluate $tokens \ge 1$ as true, all three decide *allow*, all three write `tokens = 0`. **Three requests were admitted against a single token** — the limiter leaked 200% over its ceiling, and it leaked precisely under concurrent load, which is the only condition under which the limit matters. Now the **atomic** version: instance A's Lua script runs start-to-finish (`tokens` 1 → 0, returns *allowed*); only then can B's script run (it reads `tokens = 0`, returns *denied*, `429`); then C's (reads `0`, denied, `429`). One allowed, two rejected — exactly one token, exactly one admission. The difference is not a tuning parameter or a bigger Redis; it is *atomicity*, and it is the entire reason distributed rate limiting is harder than the single-process version. Any design where the check and the decrement are separable will leak under load.

Two more realities of distributed limiting worth naming. First, **the counter is now a dependency and a single point of contention** — every request reads Redis, so Redis latency is on the hot path of every request, and a Redis outage is an availability question (which we address with graceful degradation, next). Second, **you can trade exactness for speed** with a *local-then-sync* design: each instance keeps a small local bucket and periodically reconciles with the central counter, accepting a little over-admission in exchange for not hitting Redis on every single request. This is the standard high-throughput compromise; it loosens the limit slightly but removes Redis from the per-request critical path. For most APIs, the atomic-Redis approach is correct and fast enough; reach for local-then-sync only when the central counter becomes the bottleneck.

## Graceful degradation, shadow mode, and allowlists

A limiter is itself a system that can fail or be misconfigured, and the ways you handle *its* failure modes separate a thoughtful design from a fragile one.

**Graceful degradation: fail open or fail closed?** When the limiter's backing store (Redis) is unreachable, you face a choice. **Fail open** means: if you cannot check the limit, allow the request. **Fail closed** means: if you cannot check the limit, reject the request. For a rate limiter protecting general API capacity, **fail open is usually correct** — a brief Redis blip should not take down your entire API by rejecting everything; you would rather risk a short window of unlimited traffic than guarantee a total outage. But for limits that gate something *expensive or dangerous* — a limit that prevents a runaway from spending real money, or one protecting a fragile downstream — **fail closed** may be right, because admitting unlimited expensive operations is worse than rejecting them. The decision is per-limit and should be explicit, with a sane fallback (e.g., fall back to a conservative *local* limit when the shared store is down, rather than fully open). Never let "the limiter broke" silently become "there is no limit."

**Shadow / monitor mode.** Never roll out a new limit straight to enforcement — you will discover too late that you sized it wrong and started `429`-ing legitimate traffic. Instead run the limit in **shadow mode** first: compute the decision and *log* "this request would have been rejected," but still serve it. Watch the metrics for a week. How many legitimate callers *would* have been throttled? Which keys? Is the limit catching abuse or catching your biggest paying customer's normal traffic? Only after the shadow data confirms the limit is sized right do you flip it to enforce. This is the rate-limiting equivalent of a canary: measure the blast radius before you take the shot. It is the single most effective way to avoid the self-inflicted outage where *your own limit* becomes the incident.

**Allowlists (and the opposite).** Some callers should be exempt or get higher limits: your own internal services, a monitoring/health-check system, a strategic partner with a contract for higher throughput, a load test you are running on purpose. Maintain an **allowlist** (sometimes higher limits rather than full exemption — "unlimited internal traffic" is how you DoS yourself). Conversely, a **denylist** or dynamic *penalty box* lets you drop a known-bad key or IP to zero immediately when you detect abuse, rather than waiting for the normal limit to slowly throttle them. The allowlist and the penalty box are the manual overrides on top of the automatic limiter — the levers you reach for during an incident.

## Abuse beyond rate limits: what a counter cannot catch

Here is the uncomfortable truth that closes the loop on the intro: **a rate limit caps volume, but it cannot tell a friend from a foe.** A well-behaved customer doing 100 legitimate requests/second and a scraper doing 100 requests/second look identical to a counter. And a sophisticated attacker *stays under your limit on purpose* — they spread across thousands of IPs and accounts, each one perfectly compliant, while the aggregate attack proceeds. Rate limiting is the necessary floor, not the whole building. Real abuse protection is layered: volume limits, plus identity checks, plus effort checks.

![a tree of abuse defenses showing a root of layered protection branching into a volume layer with the token bucket, an identity layer with a WAF and credential-stuffing locks, and an effort layer with proof-of-work and CAPTCHA challenges](/imgs/blogs/rate-limiting-quotas-and-abuse-protection-8.png)

**Bot detection.** Distinguishing automated clients from humans (or legitimate automation from malicious automation) uses signals a rate limit ignores: client fingerprints (TLS fingerprint, header order, JA3), behavioral patterns (a human does not request 10,000 product pages in perfect alphabetical order at exactly 50 ms intervals), and reputation databases (this IP range is a known data-center proxy network). Managed bot-management services and the bot rules in a WAF encode much of this. The output is a *score*, which you can feed into a tighter limit, a challenge, or a block.

**WAF (Web Application Firewall).** A WAF inspects requests for known-malicious *patterns* — SQL injection probes, path-traversal attempts, scanner signatures, request shapes that match known exploits. It is signature-and-rule based, sitting at the edge alongside the coarse rate limit. It catches a category of abuse (malicious *content*, not just malicious *volume*) that a counter never sees. Input validation and the OWASP API risks that the WAF complements are the subject of the [input-validation post](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10) — the WAF is a net, not a substitute for validating at the service.

**Anomaly detection.** Statistical and ML-based detection of *deviation from normal*: a key that suddenly requests endpoints it never touched before, a 50× spike in `404`s (someone enumerating ids), a login endpoint whose failure rate jumped from 2% to 90% (a credential-stuffing run in progress), traffic from a country this account has never logged in from. Anomaly detection catches *novel* abuse that no static rule anticipated, at the cost of false positives — so its output usually feeds a challenge or an alert, not an automatic block.

**Proof-of-work and CAPTCHA.** When you suspect abuse but are not certain, *raise the cost of proceeding* rather than blocking outright. A CAPTCHA asks for a proof of humanity; a proof-of-work challenge makes the client burn CPU on a small puzzle before the request is served. Both are nearly free for a legitimate user making occasional requests and ruinously expensive for an attacker making millions — they shift the economics. The modern, less user-hostile form is an invisible challenge (a managed "turnstile"-style check) that most legitimate clients pass without friction while bots stall. The principle is the same as cost-based limiting, applied to *suspect* traffic: make abuse pay.

**Credential-stuffing defense specifically.** This deserves its own treatment because it is the most common serious attack on an API's auth surface, and a per-IP rate limit alone does not stop it. Attackers take databases of leaked username/password pairs and test them against your login endpoint, distributed across many IPs to evade per-IP limits. The layered defense: rate-limit the login endpoint **per username** (not just per IP) so that an attacker hammering one account is throttled even across many source IPs; lock or step up an account after N failed attempts; require a CAPTCHA or step-up challenge when the failure rate spikes; deploy a breached-password check so leaked credentials cannot be reused; and — the real fix — push users toward multi-factor auth so a correct password alone is not enough. The rate limit is one layer; the per-username key and the MFA are what actually defeat the attack. This is why the *key* discussion earlier matters so much: keying the login limit on username instead of IP is the difference between a defense that works and one that is trivially bypassed.

The structuring idea for all of this: **rate limiting controls *how much*; abuse protection controls *who* and *what*.** You need both. A counter without identity checks lets a distributed attacker through; identity checks without a counter let a single compromised key flood you. Layer them.

## Case studies: how real APIs do it

Concrete, accurate examples from APIs you can inspect. These are useful precisely because they show the *contract* — the headers and codes — in production.

**GitHub.** GitHub's REST API uses a primary rate limit (historically 5,000 requests/hour for authenticated requests; 60/hour for unauthenticated, which is the kind of low number that pushes you to authenticate) and returns the limit state on every response. Historically it used the `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers, where `X-RateLimit-Reset` is a **Unix epoch timestamp** of when the window resets — a deliberate design that lets clients compute exactly when to resume. GitHub also enforces *secondary* rate limits to catch abusive patterns (too many concurrent requests, too much content creation in a short time) and returns `429` (or sometimes `403`) with `Retry-After` for those. Their **GraphQL** API does not count requests at all — it uses **query cost analysis**, computing a point cost per query against a node budget, which is the cost-based limiting principle made fully explicit. GitHub is the canonical example of "rate limit headers on every response" plus "cost-based limiting for GraphQL."

**Stripe.** Stripe rate-limits the API and returns **`429 Too Many Requests`** when you exceed it; their guidance is the textbook client behavior — retry with **exponential backoff** and respect the limits — and they recommend idempotency keys so those retries are safe (the exact pairing of `Retry-After`-driven retries with idempotency we walked through earlier). Stripe is worth studying as much for its *deprecation and versioning* discipline as its limits, but on rate limiting the lesson is the clean `429` + backoff + idempotency contract that makes a payment retry safe rather than dangerous.

**Twitter/X.** The Twitter/X API has long used per-endpoint, windowed limits and the `x-rate-limit-limit`, `x-rate-limit-remaining`, and `x-rate-limit-reset` headers (note the per-endpoint granularity — different endpoints have different windows, which is cost-based limiting expressed as per-endpoint buckets). The v1.1 → v2 migration also changed the limit structure and tiers, a reminder that *limits are part of the versioned contract* and changing them is a breaking change to plan for.

**Cloudflare / the edge.** Cloudflare and similar edge providers implement rate limiting at the network edge — before traffic reaches your origin — and notably use a **sliding window counter** (the approximate two-window algorithm we derived) precisely because it gives near-exact accuracy at the fixed-window's $O(1)$ cost across enormous request volumes. The edge is also where bot management and WAF rules run. This is the real-world embodiment of the "coarse limits at the edge" layer: shed volumetric abuse cheaply, far from your origin.

**The IETF `RateLimit` header draft.** The HTTP working group's "RateLimit header fields for HTTP" draft is the effort to standardize the long-used `X-RateLimit-*` convention into interoperable, un-prefixed fields. It is important to be accurate here: it is a *draft*, it has evolved across versions (from three separate fields to a consolidated structured `RateLimit` field plus a `RateLimit-Policy` field), and it is not yet a finished RFC. The practical guidance — emit the widely-deployed three-field form, and the structured form where you can — comes directly from the fact that the standard is still settling while clients in the wild already parse the older convention.

The throughline across all of them: a `429`, a `Retry-After` (or a reset time the client can compute), and per-response limit headers so clients self-throttle. That is the interoperable contract, regardless of the algorithm underneath.

## Tuning and observing the limit: how to size it and know it is working

A limit you set once and never look at is a liability. The right value is not something you can guess from first principles — it depends on your backend's real capacity, your traffic shape, and your customers' legitimate burstiness. You arrive at it empirically, and then you watch it. This is where rate limiting meets observability, and it is the part most teams skip until the limit causes an incident.

**Sizing the rate limit from capacity, not from a round number.** Do not pick "100 req/s" because it sounds reasonable. Start from the thing you are protecting. Suppose the constraint is the database, and you have measured (with a load test, not a guess) that write latency stays healthy up to about 800 writes/second across all tenants and degrades sharply past roughly 1,000. You want headroom, so you target a *total* admitted rate of, say, 600 writes/second at the limiter — that is your global ceiling. Now divide fairly: if you have 60 active tenants and want no single tenant to be able to consume more than a fraction of capacity, a per-tenant rate limit of 100 req/s with a burst of 200 leaves room for several tenants to spike at once without any one of them, or even a few of them together, crossing the 800-write danger line. The math is deliberately conservative because the cost of being slightly too low (a few extra `429`s a careful client retries through) is far smaller than the cost of being too high (the cliff). **Size from measured capacity, leave headroom, then divide for fairness.**

**The four signals to emit on every limiter decision.** A limiter should be one of the best-instrumented parts of your stack, because it sits on the hot path and its decisions directly shape the customer experience. Emit, per key and per endpoint:

- **Allowed count** and **rejected (`429`) count.** The ratio is your *rejection rate*. A healthy steady-state rejection rate is low and stable; a sudden jump means either an attack, a client bug, or a limit set too low.
- **Remaining-tokens distribution.** How close are callers running to the ceiling? If most keys spend most of their time near zero remaining, your limit is biting normal traffic and is probably too low. If no key ever drops below 80% remaining, the limit may be doing nothing.
- **Per-key top talkers.** Who is getting rejected the most? This is how you tell "one buggy partner" (one key dominating the `429`s — the intro scenario) from "the limit is wrong for everyone" (rejections spread evenly across keys).
- **The limiter's own latency and error rate.** Because the shared counter (Redis) is on every request's critical path, its p99 latency *is* part of your API's p99. And its error rate drives your fail-open/fail-closed behavior. If you cannot see when the limiter itself is unhealthy, you cannot make the degradation decision.

These tie directly into the RED method (Rate, Errors, Duration) you would track for any endpoint; the limiter just adds "admitted vs rejected" as a first-class dimension. The deeper treatment of API observability — correlation IDs, traces, SLOs — lives in its own post, but the rate-limit-specific addition is this: **a `429` is a planned, healthy outcome, so chart it separately from `5xx` errors.** Conflating "we deliberately rejected an over-limit request" with "the server broke" hides both. A spike in `429`s is a capacity-or-abuse signal; a spike in `5xx` is a bug. They demand different responses.

**The feedback loop.** Sizing is not one-time. Your capacity changes (a bigger database, a faster downstream), your traffic grows, and your tenant mix shifts. Treat the limit as a living parameter: review the rejection-rate dashboards on a cadence, and when you change a limit, **re-run it through shadow mode first** (the monitor-mode discipline from earlier) so you see the new blast radius before you enforce it. The limit that protected you last quarter at 100/s may be strangling a customer who has since 5×'d their legitimate traffic — the dashboards are how you catch that *before* they open a support ticket, and the shadow run is how you raise it safely.

#### Worked example: reading the dashboards to find the real problem

Your `POST /payments` rejection rate, normally under 0.5%, jumps to 12% over ten minutes. Two very different root causes produce that same top-line number, and the per-key breakdown is what tells them apart. **Case A:** the `429`s are 95% concentrated on a single API key — that is one caller (a buggy retry loop or a deliberate flood), and the fix is a conversation with that partner or a temporary penalty-box entry; the limit is doing exactly its job, protecting everyone else. **Case B:** the `429`s are spread roughly evenly across most of your active keys — that is *not* one bad actor; it means aggregate legitimate load has grown past the ceiling you set, and the fix is to raise capacity and the limit (after a shadow run), not to punish anyone. Same 12% on the headline chart; opposite causes; opposite responses. Without the per-key dimension you would guess, and you would guess wrong half the time. This is why "top talkers by `429`" is not a nice-to-have — it is the signal that turns an ambiguous alert into a decision.

## When to reach for this (and when not to)

Rate limiting is not free — it adds a dependency, latency on the hot path, and a whole class of "my own limit caused an incident" failure modes. Apply it with judgment.

**Reach for it when:**

- **Your API is public, or has external/partner callers.** Anything you do not fully control on the client side will eventually send you a bad burst, intentionally or not. A limit is mandatory here.
- **You are multi-tenant.** Per-tenant limits are the only way to stop the noisy neighbor from starving everyone else. This is non-negotiable for shared infrastructure.
- **A request costs real money or hits a fragile downstream.** Payments, LLM calls, anything metered or rate-limited *below* you. The limit protects your bill and your dependencies.
- **You have authentication endpoints.** Login, signup, password reset, and token endpoints are prime credential-stuffing and abuse targets and need their own (per-username, not just per-IP) limits.

**Be cautious or skip it when:**

- **It is a purely internal API between trusted services you fully control,** *and* you have other backpressure (circuit breakers, connection-pool limits, queue bounds). You may not need a per-caller limit if you trust every caller and have load shedding elsewhere — though a coarse safety limit is cheap insurance against a buggy deploy on the caller side (which is exactly what bit us in the intro, and those *were* internal-ish partners).
- **You would key it on IP for an authenticated API.** Do not. Key on identity. IP-only is the trap covered above — it punishes shared-NAT users and waves through distributed attackers.
- **You would return `200` with an error body, or `503` instead of `429`.** Do not lie about the status. `429` is the dedicated, correct code; use it so caches, proxies, and client libraries behave correctly.
- **You would ship a fixed window for a limit where the boundary spike matters.** The 2× boundary bug is real; use token bucket or a sliding window for capacity protection. A fixed window is fine only for coarse counting where a momentary 2× does not hurt (like a monthly quota).
- **You would enforce a brand-new limit straight into production without shadow mode.** You will throttle legitimate traffic and cause your own outage. Measure first.

The meta-rule: a rate limit is a *contract*, so design it like one — pick the algorithm by force (burst tolerance, accuracy, scale), key it on the right identity, return honest status and headers, and roll it out behind a shadow before you enforce.

## Key takeaways

- **A rate limit protects capacity, ensures fairness, contains abuse, and controls cost** — four distinct goals that want different windows and keys. Name them separately or you will build one limiter that does all four badly.
- **Rate limit ≠ quota.** A rate limit caps short-window throughput (100 req/s, transient breach, short `Retry-After`); a quota caps long-window allowance (1M req/month, sustained breach, long `Retry-After` or `402`). A request must pass both.
- **Token bucket is the default.** Capacity $C$ is the burst, refill rate $r$ is the average; allow iff $tokens \ge 1$, with lazy refill at $O(1)$ memory. It cleanly separates the two parameters you usually want to set independently.
- **Fixed window has a 2× boundary bug** — derivable: $L$ requests on each side of a clock reset is $2L$ in one window-span. Use a sliding window (log if small and exact; counter if high-scale and approximate) or token bucket for anything where the spike matters.
- **Key on identity, not IP.** Per API key / tenant / user for authenticated traffic; IP only at the unauthenticated edge, and even then as one signal — shared NAT punishes innocents and botnets dodge per-IP limits.
- **The response contract is `429` + `Retry-After` + the `RateLimit` headers.** Honest status, an actionable `problem+json` body, and per-response limit state so cooperative clients self-throttle and never see a `429` in steady state.
- **Weight by cost.** Charge expensive endpoints more tokens; in GraphQL, compute query cost. The limit should track real load, not raw request count.
- **Distributed limiting needs an atomic shared counter.** A read-then-write race admits two requests on one token; use Redis `INCR` or a Lua script so check-and-decrement is indivisible. Decide fail-open vs fail-closed per limit.
- **Roll out in shadow mode, keep allowlists, and have a penalty box.** Measure who you would throttle before you enforce; exempt trusted callers; be able to drop a bad key to zero instantly.
- **A counter cannot tell friend from foe.** Layer abuse protection on top: bot detection, WAF, anomaly detection, proof-of-work/CAPTCHA, and per-username credential-stuffing defenses with MFA. Rate limiting is the floor, not the whole building.

## Further reading

- **RFC 6585, "Additional HTTP Status Codes"** — defines `429 Too Many Requests` (and `428`, `431`). The normative source for the status code you return.
- **RFC 9110, "HTTP Semantics"** — defines `Retry-After` and the semantics of the status classes. The foundation under the whole response contract.
- **RFC 9457, "Problem Details for HTTP APIs"** — the `application/problem+json` envelope used for the `429` body, so your rate-limit errors match the rest of your error contract.
- **IETF draft, "RateLimit header fields for HTTP"** (httpwg) — the in-progress standardization of the `RateLimit` / `RateLimit-Policy` headers. Read the latest draft for the current field shapes; it is still evolving.
- **Stripe API docs — rate limits** — the canonical `429` + exponential-backoff + idempotency-key guidance for a payments API.
- **GitHub REST and GraphQL API docs — rate limiting** — per-response limit headers, secondary limits, and GraphQL query-cost analysis (cost-based limiting in production).
- **Within this series:** start at the [intro hub on the API as a contract](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); see [status codes that tell the truth](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx) for why `429` beats `200`-with-an-error, the [API gateway post](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern) for where most limits live, [input validation and the OWASP API Top 10](/blog/software-development/api-design/input-validation-output-encoding-and-the-owasp-api-top-10) for the abuse defenses a counter complements, and the [API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2) for the full review checklist. For the distributed-systems view of paradigms at scale, see [API design — REST, gRPC, GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql).

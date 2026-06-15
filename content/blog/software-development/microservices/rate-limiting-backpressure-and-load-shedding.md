---
title: "Rate Limiting, Backpressure, and Load Shedding: Surviving More Load Than You Can Handle"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Three mechanisms engineers constantly confuse — a quota at the front door, a slow-down signal between stages, and triage under duress — explained with token-bucket math, an adaptive concurrency limiter, a priority shedder, and the worked numbers that keep a service serving at 5x its capacity instead of falling over."
tags:
  [
    "microservices",
    "rate-limiting",
    "backpressure",
    "load-shedding",
    "resilience",
    "distributed-systems",
    "software-architecture",
    "backend",
    "concurrency",
    "reliability",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/rate-limiting-backpressure-and-load-shedding-1.webp"
---

The ShopFast order service had survived two years of steady growth without anyone thinking hard about overload, and then the Black Friday flash sale hit it with five times its normal traffic in the space of ninety seconds. The graphs from that morning are still pinned in the team's incident channel as a teaching artifact, because they tell the whole story of this post in three lines. The first line is request rate: a clean vertical wall as the sale banner went live. The second line is latency: flat at 40 milliseconds for about eight seconds, then a curve that bends upward and keeps bending — 200ms, then 2 seconds, then 18 seconds — until it goes off the top of the chart. The third line is the one that mattered: *successful* checkouts per second, which climbed for a few seconds, peaked, and then fell off a cliff to almost nothing while the request rate was still pinned at maximum. The service was busier than it had ever been and accomplishing less than on a quiet Tuesday. Twelve minutes later the order service pods started getting OOM-killed by Kubernetes, one after another, and because the payment service shared a dependency, it went down too.

Here is the thing that surprised the junior engineers on the team and that every senior has learned the hard way: **a service pushed past its capacity does not degrade gracefully on its own.** It does not serve 100% of what it can handle and politely queue the rest. Left undefended, it does something much worse — it accepts every request, tries to work on all of them at once, and in doing so makes *every* request slow, including the ones it could easily have served. Queues grow without bound, memory fills with half-finished work, latency explodes, clients time out and retry (adding *more* load), and the whole thing spirals into what queueing theory calls congestion collapse: a system doing maximal work and producing minimal useful output, right up until it falls over and takes its neighbors with it. The figure below is the contrast at the heart of this post — the same flash sale hitting a service with no defense versus one that caps, signals, and sheds.

![A before and after comparison contrasting an overloaded order service with no defense that grows an unbounded queue and crashes against a defended service that caps requests at the gateway, signals upstream with a bounded queue, and sheds low-priority work to keep latency flat](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-1.webp)

The good news is that the cures are well understood, and there are exactly three of them. The bad news is that they are *three different things*, and engineers confuse them constantly — using "rate limiting" to mean all three, or reaching for backpressure when the problem actually calls for shedding, or hard-coding a concurrency limit that is wrong by Tuesday. So before we touch a line of code, let me draw the distinction sharply, because getting it clear in your head is most of the battle. **Rate limiting** is a *policy*: a quota enforced at the front door, usually per client, that says "you may make at most N requests per second." **Backpressure** is a *signal*: a message that flows *backward* through a pipeline, from a stage that cannot keep up toward the stage feeding it, that says "slow down, I am full." **Load shedding** is *triage*: the deliberate decision, when you are *already* overloaded, to drop or reject some work — ideally the least important work — so that the rest survives. Rate limiting happens before the work starts; backpressure happens between stages of the work; load shedding happens when the work is already too much. They compose, they do not substitute.

By the end of this post you will be able to do four concrete things. You will be able to implement a token-bucket rate limiter and do the arithmetic that sizes it for "100 requests per second with bursts up to 20." You will be able to build distributed rate limiting on Redis and explain exactly why the naive version lets one client exceed your limit sixfold. You will be able to put a bounded queue with backpressure between two stages and an *adaptive* concurrency limit in front of a service so it finds its own safe operating point instead of relying on a number you guessed. And you will be able to write a priority-based load shedder that drops analytics writes to keep checkout alive, and prove with numbers that it holds p99 latency flat at 5x traffic where the undefended service collapses. We will run all of it on ShopFast's flash sale, and finish with how Stripe, Cloudflare, and Netflix actually do this in production, plus a real unbounded-queue post-mortem.

This post sits in the resilience track of the series. The [resilience patterns post on timeouts, retries, circuit breakers, and bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) covered the patterns that protect *you* from a slow *dependency*. This post is the mirror image: the patterns that protect you from *callers* who, deliberately or not, are sending you more than you can handle. The [partial failures and graceful degradation post](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) covered what to return when a piece is missing; here, *you* are the piece deciding what to keep and what to drop. And where the [message-queue backpressure post](/blog/software-development/message-queue/backpressure-and-flow-control) covers flow control end-to-end in asynchronous pipelines, this post is the synchronous, request/response, microservices-edge view of the same family of problems.

## Why an overloaded service falls over (the shape of the cliff)

Before the cures, you have to internalize the disease, because the instinct to "just add more capacity" or "just retry harder" actively makes it worse. Start with the simplest possible model. A service has some maximum sustainable throughput — call it its *capacity*, C, measured in requests per second. As long as the *offered load* (the rate requests arrive) is below C, the service handles everything, latency is stable, and life is good. The interesting question is what happens when offered load crosses C and keeps climbing.

The naive mental picture is that throughput flattens out at C and stays there: you offer 2C, the service does C, and the extra C piles up in a queue. That picture is wrong in a way that kills systems. In reality, *goodput* — the rate of requests completed **successfully and usefully** — does not flatten at C. It rises to a peak somewhere near C and then *falls*, often all the way to near zero, as offered load increases past that peak. This is the goodput cliff, and there are three compounding reasons for it, each of which a senior engineer should be able to name on sight.

The first is **queueing latency**. By Little's Law — which we will use for real arithmetic later — the number of requests in flight inside a system equals the arrival rate times the time each request spends inside. If arrivals exceed the rate the service can drain them, the in-flight count grows, and because each request now waits behind a growing backlog, the time-in-system grows too. Latency is not a fixed property of your service; it is a function of how full your queues are. A request that would take 40ms in isolation takes 40ms *plus the time spent waiting for everything ahead of it*. When the queue has 10,000 entries and you drain 1,000 per second, the newest entry waits 10 seconds before you even start it. Most of those requests have a client-side timeout of 1 or 2 seconds, so by the time you start working on them, **the client has already given up** — you are spending CPU computing a response nobody is listening for. That is pure waste, and it is the engine of the cliff.

The second is **retry amplification**. When the client times out, what does it do? If it is a well-behaved client with the retry logic from the resilience patterns post, it retries — and now you have *two* copies of that request in your system, then the retry times out and there are three. A single original request becomes 2x or 3x load precisely at the moment you have the least headroom. The system's own clients turn a 2x spike into a 5x spike. This is why "just retry harder" is poison under overload: retries are a positive feedback loop that converts overload into collapse.

The third is **resource exhaustion from work-in-progress**. Every request in flight holds resources: a goroutine or thread, a database connection from the pool, some memory for its buffers and partial results, maybe a lock. When 50,000 requests are in flight because you accepted them all, you have 50,000 requests' worth of memory allocated — and that is the OOM kill that ended ShopFast's morning. The unbounded queue is not a buffer that smooths the spike; it is a deferred out-of-memory crash with a countdown timer set by how fast the queue grows. (This is the exact same lie the message-queue post diagnosed for async queues — the bound is the only thing that turns a buffer from a crash-in-waiting into a real shock absorber.)

Put the three together and the lesson is stark: **the worst thing an overloaded service can do is try to serve everything.** Trying to serve everything means serving nothing well, wasting work on timed-out requests, amplifying load through retries, and exhausting memory. The way out is to *do less on purpose* — admit only what you can actually complete, and reject the rest cheaply and immediately. A fast rejection costs almost nothing and lets the client back off; a slow timeout costs a full request's worth of resources and teaches the client to retry. "Shed before you fall over" is the whole philosophy in five words. Everything in this post is machinery for deciding *what* to admit, *how* to signal "no," and *whom* to say it to.

It is worth pausing on *why* this is so counterintuitive, because the instinct it violates is deep. Engineers are trained to treat dropping a request as a failure — a bug, a number that should be zero, something the on-call should be paged about. And in normal operation, that instinct is correct: a service that drops requests when it has spare capacity is broken. But under overload the instinct inverts. When offered load exceeds capacity, *some* requests are not going to be served well no matter what you do — that is just arithmetic, not a choice. The only choice you have is *which* requests fail and *how* they fail. Do you let all of them fail slowly (everyone waits 18 seconds and then times out), or do you make the excess fail fast (a few get an instant `503`, the rest get served at full speed)? The first option is the default if you do nothing, and it is strictly worse on every axis: worse latency for everyone, more wasted work, more retries, and eventually a crash. The second option requires you to *deliberately* drop work, which feels like giving up, but it is the only path that keeps the system useful. Maturity as an engineer is partly the journey from "dropping requests is always bad" to "under overload, *choosing* what to drop is the most important thing I can do."

The shape of the goodput curve also explains why monitoring "is the service up?" is not enough. A service can be 100% "up" — every pod healthy, CPU pinned, accepting connections — and yet delivering near-zero goodput because it is past the cliff, drowning in timed-out work. The signal that matters is not liveness but *useful throughput*, which is exactly why the [SLOs and golden-signals post](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) treats saturation and latency as first-class signals alongside errors. An overloaded service lies to a naive health check; you have to measure the thing the cliff actually destroys.

## The three mechanisms, side by side

Because the confusion between these three is the single most common conceptual error I see in design reviews, let me lay them out in one matrix before going deep on each. Read this table as a reference you can come back to; the rest of the post is the detailed argument for each cell.

![A decision matrix comparing rate limiting, backpressure, and load shedding across what problem each solves, where it is applied, what it does, when to use it, and what it returns when it trips](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-2.webp)

| Dimension | Rate limiting | Backpressure | Load shedding |
|---|---|---|---|
| **What problem** | Abuse, unfairness, quota enforcement | A consumer can't keep up with a producer | You are *already* past capacity |
| **Nature** | A policy / quota (proactive) | A flow-control signal (cooperative) | Triage under duress (reactive) |
| **Where** | Gateway / edge, per client or tenant | Between two stages of a pipeline | Inside the service, at admission |
| **Granularity** | Per API key, user, tenant, IP | Per connection / queue / stream | Per request, by priority |
| **What it does** | Caps the request *rate* | Tells the upstream to slow down | *Drops* or rejects selected work |
| **When it fires** | Continuously, on every request | When the bounded buffer fills | When latency/concurrency crosses a threshold |
| **What it returns** | `429 Too Many Requests` + `Retry-After` | Blocks, pauses, or pulls slower | `503` fast-reject (or silent drop) |
| **Whose load** | A specific client's excess | The immediate upstream's | The lowest-priority traffic, system-wide |

The crisp distinctions to carry around:

**Rate limiting is a quota at the front door.** It does not care whether you are overloaded right now; it enforces a *contract* — "your plan includes 1,000 requests per minute" — and it is fundamentally about *fairness and abuse protection*, not about your current health. One client in a retry loop should not be able to consume the capacity meant for everyone else. Rate limiting is how you make sure of that, and it is best applied as far forward as possible, ideally at the [API gateway](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend), where saying "no" is cheapest.

**Backpressure is a signal between stages.** It is *cooperative*: it only works when the upstream is willing to listen and slow down. A pull-based consumer that asks for the next batch only when it is ready is applying backpressure by construction. A bounded queue that blocks the producer when full is applying backpressure. The signal flows *backward*, and it is most natural in asynchronous, streaming, or pipeline architectures — which is exactly why the deep mechanism treatment lives in the [message-queue backpressure post](/blog/software-development/message-queue/backpressure-and-flow-control). In a synchronous request/response service, backpressure shows up as bounded thread pools and connection limits that make the *caller* wait or fail fast.

**Load shedding is triage under duress.** It is *reactive*: it only happens when you have detected that you are over capacity *right now*, and its job is to protect the most important work by sacrificing the least important. Unlike rate limiting, it is not per-client — a P0 checkout from a heavy user beats a P2 analytics write from a light one. Unlike backpressure, it does not wait for cooperation — it unilaterally drops work because waiting is no longer an option.

They compose into a defense in depth. On ShopFast's flash-sale day, the *correct* architecture has all three: rate limiting at the gateway stops any single customer from hammering the API, backpressure between the order service and a slow inventory consumer keeps a bounded queue from exploding, and load shedding inside the order service drops analytics writes to keep checkout's p99 flat. The figure below shows where each one sits in the request path.

![A service topology graph showing a request crossing a per-client token-bucket rate limit at the gateway, then an adaptive concurrency limit at the order service, then a priority shedder that protects the checkout core path while dropping low-priority analytics writes](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-3.webp)

Notice the geometry: rate limiting is at the *edge*, concurrency limiting is at the *service*, and shedding is *inside* the service right before the protected core work. Each one catches a different kind of excess. The gateway catches per-client abuse. The concurrency limit catches aggregate overload regardless of which clients caused it. The shedder makes the priority decision when even the concurrency limit's admitted load is too much for the most important path. Now let's build each one.

## Rate limiting: the algorithms and their trade-offs

Rate limiting is the most familiar of the three, and it is where most engineers start, so let's get the algorithms right. Every rate limiter answers one question — "should I allow this request, given how many I have allowed recently?" — and the four classic algorithms differ in how they define "recently" and how they handle bursts.

### Fixed window

The simplest: divide time into fixed windows (say, one-minute buckets), keep a counter per client per window, increment on each request, reject when the counter exceeds the limit, and reset the counter when the window rolls over. It is trivial to implement — one integer per client, one comparison — and that simplicity is its entire appeal. But it has a notorious flaw: the **boundary spike**. If your limit is 100 requests per minute, a client can send 100 requests in the last second of one window and 100 in the first second of the next — 200 requests in a two-second span, double the intended rate, because the counter reset at the window boundary. For coarse quotas this is fine; for protecting a service from bursts it is dangerous.

There is a second, subtler problem with fixed windows that bites at scale: **synchronized resets**. If every client's window is keyed to wall-clock minute boundaries (`floor(now / 60)`), then *every* client's counter resets at the same instant — the top of each minute — and well-behaved clients that were waiting out their limit all wake up and fire simultaneously. You have built a thundering herd into the limiter itself: a traffic spike every sixty seconds, on the second, caused by the limiter's own clock alignment. The fix is to key each client's window to an offset derived from its own ID (so resets are spread across the minute) or to abandon fixed windows for sliding ones. This is the kind of second-order effect that does not show up in a load test with one client and absolutely shows up in production with a million — and it is why the algorithm choice is not just about the boundary spike but about how the limiter behaves *in aggregate* across your whole client population.

### Sliding window log

The most accurate: store a timestamp for every request in the window, and on each new request, drop all timestamps older than the window, count what remains, and allow if the count is under the limit. This is exact — no boundary spike, the window genuinely slides — but it costs memory proportional to the request rate: at 1,000 requests per second with a one-minute window, you are storing up to 60,000 timestamps *per client*. Accurate but expensive. A common middle ground is the **sliding window counter**, which keeps counters for the current and previous fixed windows and *interpolates* — if you are 30% into the current minute, you weight the previous window's count by 70% and add the current window's count. This approximates the true sliding window with two integers instead of thousands of timestamps, and it is what most production limiters actually use.

### Token bucket

The workhorse, and the one ShopFast uses at its gateway. Picture a bucket that holds up to `B` tokens and refills at a steady rate of `R` tokens per second, never exceeding `B`. Each request must remove one token to proceed; if the bucket is empty, the request is rejected (or it waits, depending on configuration). The two parameters say something physically meaningful: `R` is the **sustained rate** you allow over the long run, and `B` is the **maximum burst** you tolerate when the client has been quiet and the bucket has filled up. This is exactly the right model for an API, because real clients are bursty — a mobile app syncs several resources at once on cold start, then goes quiet — and the token bucket says "burst up to `B` if you've earned it, but your long-run average can't exceed `R`."

![A vertical stack showing the token bucket mechanism with tokens refilling at the sustained rate, capacity setting the maximum burst, each request taking one token, a token left meaning allow, and an empty bucket meaning a 429 with Retry-After](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-4.webp)

### Leaky bucket

The token bucket's mirror image. Instead of tokens accumulating to allow bursts, requests enter a queue (the bucket) and *leak out* at a fixed rate, like water through a hole in the bottom. If the bucket overflows (the queue is full), new requests are dropped. The defining property is that the *output* rate is perfectly smooth — exactly `R` requests per second leave the bucket no matter how bursty the input — which is ideal when the thing you are protecting genuinely cannot tolerate bursts (a downstream legacy system, a third-party API with a hard rate cap). The cost is that a burst either gets buffered (adding latency) or dropped, and you lose the token bucket's friendly "save up for a burst" behavior.

Here is the comparison that decides which to reach for:

![A matrix comparing token bucket, leaky bucket, and sliding window across whether they allow bursts, their output shape, boundary spike behavior, memory cost, and what each is best suited for](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-5.webp)

#### Worked example: sizing a token bucket for 100 rps with bursts of 20

ShopFast wants each customer's API key limited to a sustained 100 requests per second, but it should tolerate a short burst of up to 20 extra requests when a client's app refreshes several resources at once. The token bucket parameters fall straight out of that sentence: the refill rate `R` is 100 tokens per second (the sustained rate), and the capacity `B` is 20 tokens (the burst). Let me trace the math so the parameters stop being magic.

Start with a full bucket: 20 tokens. A client fires a burst of 20 requests in 50 milliseconds — all 20 are allowed, draining the bucket to 0. In those 50ms the bucket refilled by `100 tokens/s x 0.05s = 5` tokens, so the bucket is actually at 5 after the burst, not 0. The client immediately fires 10 more: 5 are allowed (draining the 5 tokens), and the other 5 are rejected with `429`. Now the client backs off. After 1 second of quiet, the bucket refills by `100 x 1 = 100` tokens, but capped at `B = 20`, so it sits at 20 — fully recharged for the next burst. The steady state is what matters: if the client sends a constant 100 rps, each second adds 100 tokens and each second removes 100, so the bucket stays near full and never rejects. Push to 120 rps sustained, and the bucket drains by 20 net per second; it empties in 1 second and from then on rejects 20 requests every second — the client is throttled to exactly the 100 rps you allowed. The parameters do precisely what the English sentence asked: `R` caps the long-run average, `B` is the size of the burst you forgive.

Now the code. A correct token bucket does not run a background thread ticking tokens in; it computes the current token count *lazily* from the elapsed time since the last request, which is both more accurate and far cheaper. Here is an idiomatic Go implementation:

```go
package ratelimit

import (
	"sync"
	"time"
)

// TokenBucket caps a client to `rate` tokens/sec with a burst of `burst`.
type TokenBucket struct {
	mu         sync.Mutex
	rate       float64   // tokens added per second (sustained rate)
	burst      float64   // max tokens (max burst)
	tokens     float64   // current tokens (fractional, lazily computed)
	lastRefill time.Time
}

func NewTokenBucket(rate, burst float64) *TokenBucket {
	return &TokenBucket{
		rate:       rate,
		burst:      burst,
		tokens:     burst, // start full
		lastRefill: time.Now(),
	}
}

// Allow reports whether one request may proceed, refilling lazily.
func (b *TokenBucket) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(b.lastRefill).Seconds()
	// Add tokens for the time that has passed, capped at burst.
	b.tokens = min(b.burst, b.tokens+elapsed*b.rate)
	b.lastRefill = now

	if b.tokens >= 1 {
		b.tokens -= 1
		return true
	}
	return false // empty bucket -> reject (caller returns 429)
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
```

The lazy-refill trick is the part juniors get wrong. There is no goroutine adding tokens; instead, on every `Allow` call we look at how much wall-clock time has elapsed since we last touched the bucket and credit exactly that many tokens. This is O(1) per request, holds no background resources, and is exactly accurate. The fractional `tokens` field matters too — if you store tokens as an integer and a request arrives 7ms after the last one at 100 tokens/sec, you should credit 0.7 tokens, and integer truncation would lose that and slowly starve the client below the intended rate.

When the bucket says no, you do not just drop the connection — you tell the client *how long to wait*, which is the difference between a polite limiter and a hostile one.

```python
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import math

app = FastAPI()

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_id = request.headers.get("X-API-Key", request.client.host)
    bucket = buckets.get_or_create(client_id, rate=100, burst=20)

    if bucket.allow():
        resp = await call_next(request)
    else:
        # Tell the client exactly how long until one token is available.
        retry_after = math.ceil(1.0 / bucket.rate)  # seconds until next token
        resp = JSONResponse(
            status_code=429,
            content={"error": "rate_limited",
                     "message": "Too many requests. Slow down."},
        )
        resp.headers["Retry-After"] = str(retry_after)

    # Always advertise the limit so well-behaved clients can self-pace.
    resp.headers["X-RateLimit-Limit"] = "100"
    resp.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
    return resp
```

Three details make this production-grade. First, the status code is `429 Too Many Requests`, the HTTP-standard "you are being rate limited" signal — not `503` (which means *the server* is in trouble) and not `403` (which means *forbidden*, a permission problem). The distinction matters because clients and intermediaries treat them differently: a `429` says "this will succeed if you wait," a `503` says "the service is unhealthy," a `403` says "stop trying." Second, the `Retry-After` header tells the client exactly how long to back off, so it does not hammer you in a tight loop — a limiter without `Retry-After` *causes* the retry storm it was meant to prevent. Third, the `X-RateLimit-*` headers advertise the limit and the remaining budget so a well-behaved client can pace itself *before* hitting the wall, which is far better for everyone than discovering the limit by being rejected.

## Distributed rate limiting and why the naive version is broken

The limiter above lives in one process. But ShopFast runs six gateway instances behind a load balancer, and that breaks everything, in a way that is worth understanding precisely because it is the single most common distributed-rate-limiting bug.

If each of the six gateway nodes keeps its own local token bucket configured for 100 rps, then a client whose requests are spread across all six nodes by the load balancer can do 100 rps *on each node* — 600 rps total, six times the limit you thought you set. The local counters do not know about each other. The figure below shows the failure and the fix.

![A before and after comparison showing per-node local counters letting a client hit six gateway nodes in parallel to slip 600 requests per second through a 100 limit, versus an atomic Redis script enforcing one shared counter for the true global cap](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-6.webp)

The fix is to keep the counter in a *shared* store that all nodes consult — almost always Redis, because it is fast (sub-millisecond), has the right primitives (atomic increment, expiry), and most teams already run it. But the naive Redis approach has its own subtle bug. Consider this:

```python
# BROKEN: a read-modify-write race across gateway nodes.
def allow(client_id: str, limit: int, window_s: int) -> bool:
    key = f"ratelimit:{client_id}:{int(time.time()) // window_s}"
    current = int(redis.get(key) or 0)   # node A reads 99
    if current >= limit:                 # node B also reads 99 here
        return False
    redis.incr(key)                      # both increment -> 100, then 101
    redis.expire(key, window_s)
    return True
```

This has a classic **read-modify-write race**. Two gateway nodes both read the counter at 99, both see `99 < 100`, both increment, and now the counter is 101 — you allowed two requests when only one was under the limit. Under load, with dozens of requests racing, the limit leaks badly. The bug is that the *check* and the *increment* are separate operations with a gap between them where another node can interleave.

The correct fix is **atomicity**: the check and the increment must be a single indivisible operation. Redis gives you two ways to get this. The simplest for a fixed window is to `INCR` first (which is atomic and returns the new value) and *then* compare — there is no gap because the increment and the read of the new value are one operation:

```python
def allow(client_id: str, limit: int, window_s: int) -> bool:
    key = f"ratelimit:{client_id}:{int(time.time()) // window_s}"
    # INCR is atomic and returns the post-increment value in one round trip.
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window_s)  # set TTL only on first hit this window
    return count <= limit
```

But for anything more sophisticated than a fixed-window counter — a token bucket, a sliding window — you need multiple reads and writes to be atomic together, and the right tool is a **Lua script**, which Redis executes atomically as a single unit with no other command interleaving. Here is a token bucket as a Redis Lua script, the pattern that production limiters (and Redis's own `redis-cell` module) use:

```lua
-- token_bucket.lua: atomic token-bucket check in Redis.
-- KEYS[1] = bucket key   ARGV = rate, burst, now_ms, requested
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])   -- tokens per second
local burst    = tonumber(ARGV[2])   -- bucket capacity
local now_ms   = tonumber(ARGV[3])
local needed   = tonumber(ARGV[4])

local state    = redis.call('HMGET', key, 'tokens', 'ts')
local tokens   = tonumber(state[1])
local last_ms  = tonumber(state[2])
if tokens == nil then
  tokens  = burst        -- new client: start full
  last_ms = now_ms
end

-- Lazily refill based on elapsed time, capped at burst.
local elapsed = math.max(0, now_ms - last_ms) / 1000.0
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
if tokens >= needed then
  tokens  = tokens - needed
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, math.ceil(burst / rate * 1000) + 1000)
-- return {allowed, tokens_remaining}
return { allowed, math.floor(tokens) }
```

Because the entire script runs atomically inside Redis, there is no read-modify-write gap; all six gateway nodes call the same script against the same key, and the limit is enforced globally and exactly (modulo a millisecond or two of clock skew between nodes setting `now_ms`, which is negligible).

There is one more honest caveat. Centralizing the counter in Redis adds a network round trip to every request (typically 0.3–1ms on the same network) and makes Redis a dependency on your hot path — if Redis is down, your rate limiter is down. The standard mitigation is **fail-open**: if the Redis call times out, *allow* the request rather than rejecting it, because a rate limiter outage should never become a service outage. (For abuse protection where fail-open is dangerous, some teams fail-open but trip a circuit breaker and fall back to a coarse local limiter.) The deeper performance optimization, used by Stripe and Cloudflare, is to *not* hit Redis on every request: each node keeps a small local budget it can spend without coordination and only syncs with the central store periodically — trading a little accuracy for a lot of latency, which we will quantify in the optimization section.

## The other half of rate limiting: the well-behaved client

A rate limiter is only half a system. The other half is the *client*, and a design that ignores how clients react to a `429` is a design that creates the very retry storm it was meant to prevent. This is the part juniors skip and seniors obsess over, because the difference between a polite client and a hostile one is the difference between a limiter that protects you and a limiter that gets bypassed by sheer volume of retries.

Consider what a naive client does when it gets a `429`: it retries immediately. If the limiter is still saturated, it gets another `429`, and retries again — a tight loop that hammers your gateway thousands of times a second while accomplishing nothing, because every attempt is rejected. Multiply that by ten thousand clients all hitting the limit at the same moment (a flash sale, a scheduled cron job firing across a fleet) and you have a **thundering herd**: a synchronized stampede that turns your rate limiter into a CPU-burning rejection machine. The limiter is working — every request is correctly rejected — and yet the gateway falls over from the sheer cost of saying "no" ten million times a second. Saying "no" is cheap per request but not free, and at enough volume even free becomes expensive.

The fix is **exponential backoff with jitter**, honored from the `Retry-After` header you returned. Backoff means each successive retry waits longer than the last (1s, 2s, 4s, 8s) so a client that keeps failing keeps slowing down — the opposite of the tight loop. Jitter means each client adds a *random* amount to its wait so that ten thousand clients that all failed at the same instant do *not* all retry at the same instant; they spread their retries across a window, smoothing the herd into a manageable trickle. Backoff without jitter is a classic mistake: it slows each individual client but keeps them all synchronized, so you get a herd at 1s, then a herd at 2s, then a herd at 4s — quieter than no backoff, but still a stampede. Jitter is what desynchronizes the herd, and it is the single most important line in a client's retry logic.

```python
import random
import time
import httpx

def call_with_backoff(client: httpx.Client, request, max_attempts=5):
    base = 0.5  # seconds
    for attempt in range(max_attempts):
        resp = client.send(request)
        if resp.status_code != 429 and resp.status_code < 500:
            return resp  # success or a non-retryable client error

        # Honor the server's Retry-After if present; else exponential backoff.
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            delay = float(retry_after)
        else:
            delay = base * (2 ** attempt)  # 0.5, 1, 2, 4, 8 ...

        # Full jitter: pick a random wait in [0, delay] to desynchronize the herd.
        delay = random.uniform(0, delay)
        time.sleep(delay)

    raise RuntimeError("exhausted retries against a rate-limited endpoint")
```

The two rules in that code are the whole discipline. **Honor `Retry-After` when the server sends it** — the server knows when a token will be available, so it can give a more precise wait than the client's blind backoff curve, and ignoring it means you retry before there is any chance of success. **Use full jitter** (`random.uniform(0, delay)`, not a fixed `delay`) — AWS's well-known analysis of backoff strategies showed that "full jitter" minimizes both the number of retries and the peak concurrent load on the server, beating both no-jitter backoff and the more conservative "equal jitter" variants. A client team that ships this is a good citizen of the fleet; a client team that retries `429`s in a tight loop is a self-inflicted DDoS, and you will eventually have to block them at the edge to protect everyone else.

This is also why advertising the limit *before* the client hits it (the `X-RateLimit-Remaining` header from earlier) is worth the few bytes: a well-written client watches its remaining budget and *self-paces*, slowing down as it approaches the limit so it never gets a `429` at all. The best rate-limit interaction is the one where the client cooperatively stays under the limit and you never have to reject anything — and that only happens if you tell the client where the limit is. A limiter is a negotiation, not a wall; the headers are how you negotiate.

## Backpressure: the slow-down signal between stages

Rate limiting protects you from *callers*. Backpressure protects a *pipeline* from itself — specifically from the fast-producer-slow-consumer problem, where one stage generates work faster than the next stage can absorb it. The canonical microservices version on ShopFast: the order service writes inventory-reservation events to a queue, and a slow inventory-reconciliation consumer reads them. If the order service produces 2,000 events per second and the consumer drains 800, the gap of 1,200 per second has to go *somewhere*, and where it goes determines whether your system survives.

The wrong answer is "an unbounded buffer." An unbounded queue between producer and consumer does not solve the mismatch; it *hides* it, accumulating the 1,200/second backlog in memory until the process dies. This is the lie the [message-queue backpressure post](/blog/software-development/message-queue/backpressure-and-flow-control) dissects in depth, and it applies identically inside a single service: an unbounded `channel`, an unbounded `Queue`, an unbounded thread-pool work queue — all of them are deferred OOM crashes.

The right answer is a **bounded** buffer plus a decision about what happens when it fills. When the buffer is full, the producer must either *block* (wait until there is room — backpressure by blocking) or *fail fast* (refuse to enqueue — which is load shedding, the next section). Backpressure proper is the blocking case: the act of making the producer wait *is* the slow-down signal. The producer literally cannot run faster than the consumer drains, because it is suspended whenever the buffer is full. The signal has propagated upstream without any explicit message — the bound did it.

#### Worked example: unbounded queue vs bounded queue under a 2.5x mismatch

Suppose the producer offers 2,000 events/second and the consumer drains 800/second — a 2.5x mismatch — for a 10-minute spike. With an *unbounded* queue, the backlog grows by `2000 - 800 = 1200` events per second. After 10 minutes that is `1200 x 600 = 720,000` queued events. If each event holds about 2KB of memory (the event plus its in-flight bookkeeping), that backlog is `720,000 x 2KB ≈ 1.4 GB` of heap — on a service with a 1GB container limit, the process is OOM-killed somewhere around minute 7, *before* the spike even ends, taking the whole service down. Latency for the last events enqueued is `720,000 / 800 = 900 seconds` of queue wait — 15 minutes — so even the events that don't get lost to the crash are useless because every client gave up long ago.

Now bound the queue at 1,000 events with blocking backpressure. The queue fills in `1000 / 1200 ≈ 0.83` seconds and stays full. From then on the producer blocks whenever it tries to enqueue into a full queue, so its *effective* production rate is forced down to the consumer's 800/second — the mismatch is gone. Memory is fixed at `1000 x 2KB = 2 MB`, never grows, and the process never dies. The queue wait is bounded at `1000 / 800 = 1.25` seconds, predictable and within a sane client timeout. The producer is slower, yes — that is the point. Backpressure traded throughput you could never have sustained for survival you can. The unbounded queue offered fake throughput and a guaranteed crash; the bounded queue offers honest, sustainable throughput.

In an asynchronous, internal pipeline, the bound is a bounded channel, and the language gives you backpressure for free if you use it. A Go buffered channel of capacity N *is* a bounded queue: a send onto a full channel blocks until a receive makes room, which means the producer is automatically suspended whenever the consumer falls behind. There is no separate "backpressure mechanism" to build — the channel's bound is the mechanism.

```go
// Bounded channel: capacity caps the queue; a full channel blocks the producer.
events := make(chan InventoryEvent, 1000) // backpressure kicks in at 1000

// Producer (the order service): a send blocks when the channel is full,
// which throttles the producer to the consumer's drain rate. That block
// IS the backpressure signal — no extra plumbing needed.
func produce(events chan<- InventoryEvent, e InventoryEvent, deadline time.Duration) error {
	select {
	case events <- e: // room available: enqueue and continue
		return nil
	case <-time.After(deadline): // full for too long: give up and shed instead
		return ErrShed // -> caller returns 503; block degraded into drop
	}
}

// Consumer (inventory reconciliation): drains at its own sustainable rate.
func consume(events <-chan InventoryEvent) {
	for e := range events {
		reconcile(e) // slow; this is the rate the whole pipeline is paced to
	}
}
```

The `select` with a timeout is the crucial detail that ties backpressure to shedding. A bare `events <- e` would block the producer *forever* if the consumer is permanently stuck — which silently stalls the producer's own request handling. By giving the send a deadline, you say "apply backpressure (wait) for up to this long, but if the queue stays full past the deadline, stop waiting and shed." That single `select` is the exact boundary between the two mechanisms: block while there is hope, drop when there is not.

In a synchronous request/response service, backpressure most often shows up as a **bounded concurrency limit** — a semaphore that caps how many requests are in flight at once. When the limit is reached, new requests either wait briefly or are rejected. This bounds the work-in-progress (and therefore the memory and the queue depth) directly. Here is the pattern in Java, the bulkhead's cousin:

```java
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class ConcurrencyLimiter {
    private final Semaphore permits;
    private final long acquireTimeoutMs;

    public ConcurrencyLimiter(int maxConcurrent, long acquireTimeoutMs) {
        this.permits = new Semaphore(maxConcurrent);
        this.acquireTimeoutMs = acquireTimeoutMs;
    }

    public <T> T execute(java.util.concurrent.Callable<T> work) throws Exception {
        // Wait a SHORT time for a permit; if none, fail fast rather than pile up.
        boolean acquired = permits.tryAcquire(acquireTimeoutMs, TimeUnit.MILLISECONDS);
        if (!acquired) {
            throw new OverloadedException("concurrency limit reached"); // -> 503
        }
        try {
            return work.call();
        } finally {
            permits.release();
        }
    }
}
```

The crucial design choice is the short `acquireTimeoutMs` (say, 50ms). If you let callers wait indefinitely for a permit, you have rebuilt the unbounded queue — the waiters pile up in memory and their latency grows without bound. By waiting only briefly and then failing fast with a `503`, you bound both the concurrency *and* the wait, and the caller's own retry/circuit-breaker logic takes over. The semaphore is backpressure pointed inward: it makes the *handler* refuse new work when it is full, which propagates as a fast `503` back to the caller, which (if it is the gateway) can shed or retry elsewhere.

## Adaptive concurrency limits: stop guessing the number

The concurrency limiter above has one glaring weakness: where did `maxConcurrent` come from? You guessed it, or you load-tested it once and hard-coded it. Both are wrong over time, because the *right* concurrency limit is not a constant — it depends on how slow your downstream dependencies are *right now*, how big your instances are, what else is running, and how the workload mix shifts. A limit of 100 that was perfect last month is too high after a database slowdown (you admit more than you can complete) and too low after a hardware upgrade (you reject traffic you could have served). Hard-coded limits are stale the moment you set them.

The senior move is an **adaptive** concurrency limit — one that *measures* the system and adjusts itself, exactly as TCP congestion control discovers the right window size for a network path it has never seen. This is the idea behind Netflix's open-source `concurrency-limits` library, and the theory under it is Little's Law, which we used informally earlier and now use to size things properly.

Little's Law says: `L = λ × W`, where `L` is the average number of requests in the system, `λ` is the arrival/throughput rate, and `W` is the average time each request spends in the system. Rearranged, the concurrency that *exactly* matches your throughput at a given latency is `L = λ × W`. The insight that makes it adaptive: when a service is healthy, increasing concurrency increases throughput while latency stays flat (you are filling idle capacity). But once you cross the service's true capacity, increasing concurrency *no longer* increases throughput — instead, latency starts climbing (requests are queueing, not being served faster). So you can find the right limit by watching the relationship between concurrency and latency: push the limit up while latency stays flat, and back it off the moment latency starts to rise. That is precisely a TCP Vegas-style "additive increase, multiplicative decrease" controller driven by latency gradient rather than packet loss.

#### Worked example: Little's Law sizing and an adaptive limit finding it

Suppose ShopFast's order service, when healthy, completes a checkout in `W = 50ms` and the order is profitable up to a sustainable `λ = 1,600` requests/second per instance. Little's Law says the ideal in-flight concurrency is `L = λ × W = 1600 × 0.050 = 80` requests. So `80` is the concurrency limit you want — but you should *not* hard-code 80, because if the payment dependency slows and `W` jumps to 100ms, the throughput that 80 concurrent requests can sustain drops to `λ = L / W = 80 / 0.100 = 800` rps, and admitting at the old rate now overloads you. An adaptive limiter notices the latency climb and *lowers* the limit toward the new safe point automatically.

Watch the controller work during the flash sale. Baseline: limit 120, observed p50 latency 50ms, no queueing — healthy. Traffic hits 5x. The limiter, probing upward, briefly tries 130; latency ticks to 60ms — the gradient turned positive, a sign of queueing — so it multiplicatively backs off to `130 × 0.9 ≈ 117`, then keeps shrinking as latency keeps rising, settling around `80` where p50 returns to ~50ms. By holding concurrency at the level where latency is flat, the service runs at its *true* capacity (≈1,600 rps) and rejects the excess with `503` rather than admitting it and melting down. No human guessed 80; the math found it. Here is a compact version of the controller:

```python
import time

class AdaptiveLimiter:
    """Latency-gradient (Vegas-style) adaptive concurrency limit."""
    def __init__(self, init_limit=20, min_limit=4, max_limit=400):
        self.limit = float(init_limit)
        self.min_limit, self.max_limit = min_limit, max_limit
        self.in_flight = 0
        self.rtt_min = float("inf")   # best-case (no-queue) latency seen

    def acquire(self) -> bool:
        if self.in_flight >= int(self.limit):
            return False               # at limit -> shed (503)
        self.in_flight += 1
        return True

    def release(self, rtt_seconds: float, dropped: bool):
        self.in_flight -= 1
        if dropped:                    # a drop/timeout = strong overload signal
            self.limit = max(self.min_limit, self.limit * 0.9)
            return
        self.rtt_min = min(self.rtt_min, rtt_seconds)
        # gradient < 1 means latency is inflating vs best case -> queueing.
        gradient = self.rtt_min / max(rtt_seconds, 1e-6)
        if gradient > 0.9:             # latency near best case -> room to grow
            self.limit = min(self.max_limit, self.limit + 1)   # additive increase
        else:                          # latency inflated -> back off
            self.limit = max(self.min_limit, self.limit * gradient)  # mult. decrease
```

The beauty is that this requires *no* per-environment tuning. The same code finds limit 80 on a small instance behind a slow database and limit 400 on a big instance behind a fast one, and it *re-finds* the right number when conditions change mid-incident. You stop maintaining a magic constant that is wrong somewhere, and you start trusting a controller that measures reality. This is the difference between junior resilience ("I set the pool size to 100") and senior resilience ("the limit finds itself; I tune the controller's aggressiveness, not the limit").

## Load shedding: deliberate triage under duress

Now the third mechanism, and the one that feels most counterintuitive to engineers raised on "never drop a request." When you are *already* over capacity — your concurrency limit is maxed, your queues are full, latency is climbing — you have exactly one good option: **drop work on purpose, and drop the right work.** This is load shedding, and the philosophy is the five words from the start: *shed before you fall over.* A request you reject in under a millisecond costs almost nothing; a request you accept and time out on at 18 seconds costs a full request's worth of CPU, memory, and a database connection, and it teaches the client to retry. Fast rejection is mercy.

The naive version of shedding is to drop *randomly* once you are overloaded — accept 70% of requests, reject 30%, regardless of what they are. That is far better than nothing (it caps the load), but it is leaving enormous value on the table, because not all requests are equal. On ShopFast during the flash sale, a checkout is worth real money and an analytics write is worth almost nothing; dropping them at the same rate is malpractice. The senior version is **priority-based shedding**: classify every request into a priority tier, and when you must shed, shed the lowest tiers first. The revenue path survives; the nice-to-have path is sacrificed.

![A priority-based load shedding graph showing an admission controller reading each request's priority tier, always admitting P0 checkout and payment, admitting P1 browse only when healthy, and shedding P2 analytics and search-index writes first](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-8.webp)

The mechanism has two parts. First, **classification**: every request carries a priority, set by the gateway or the client, based on what it is. A reasonable scheme:

- **P0 (critical):** checkout, payment confirmation, anything that loses money or breaks correctness if dropped. Never shed.
- **P1 (important):** product browse, cart updates, search — degrade the experience if dropped but don't lose money. Shed under heavy pressure.
- **P2 (best-effort):** analytics events, recommendation pre-computation, non-urgent background writes. Shed first.

Second, **admission control**: a component that, given the current health signal (the adaptive concurrency limit's headroom, queue depth, or measured latency), decides which tiers to admit. As pressure rises, it sheds from the bottom up.

```go
type Priority int

const (
	P0Critical  Priority = 0 // checkout, payment — never shed
	P1Important Priority = 1 // browse, cart, search
	P2BestEffort Priority = 2 // analytics, recommendation prewarm
)

type Shedder struct {
	limiter *AdaptiveLimiter
}

// Admit decides whether to serve a request given its priority and current load.
func (s *Shedder) Admit(p Priority) bool {
	headroom := s.limiter.Headroom() // fraction of concurrency budget free, 0..1

	switch {
	case p == P0Critical:
		return true // critical always admitted; protect the core
	case p == P1Important:
		return headroom > 0.15 // shed important when nearly saturated
	default: // P2BestEffort
		return headroom > 0.40 // shed best-effort early, while plenty of room
	}
}

// Handler wires shedding in front of the work.
func (s *Shedder) Handle(req *Request) (*Response, error) {
	if !s.Admit(req.Priority) {
		// Fast 503 with Retry-After — costs ~microseconds, no work wasted.
		return nil, &Shed503{RetryAfter: 2}
	}
	return s.process(req)
}
```

The thresholds encode the policy directly: best-effort traffic starts getting shed when the service drops below 40% free headroom (early, while there is plenty of slack), important traffic only when below 15% (late, when you are nearly out), and critical traffic never. The result is a *graceful* degradation curve instead of a cliff — as load climbs, the service progressively sheds less-important work, and the most-important work sails through untouched until the very end.

#### Worked example: shedding 15% at 90% capacity keeps p99 flat

ShopFast's order service tops out at 1,600 rps per instance, with a healthy p99 of 90ms. Traffic from the flash sale arrives at 1,840 rps — 15% over capacity. The traffic mix is roughly 60% P0 checkout (1,104 rps), 25% P1 browse (460 rps), and 15% P2 analytics (276 rps). Watch the two worlds.

*Without shedding:* the service admits all 1,840 rps against a 1,600 rps capacity. Per Little's Law, the in-flight count and therefore the latency inflate: requests queue, p99 climbs from 90ms to several seconds within a minute, clients time out, retries amplify the 1,840 toward 2,500+, and the service crosses the goodput cliff into collapse. Goodput falls toward zero while CPU is pinned at 100%.

*With priority shedding:* the admission controller sheds the 276 rps of P2 analytics first (a `503` in microseconds each, near-zero cost), bringing admitted load to `1,840 - 276 = 1,564` rps — just under the 1,600 capacity. The service now runs at a sustainable load: no queue growth, p99 stays at ~90ms, all checkout and browse traffic succeeds, and the only casualties are analytics writes that nobody will miss for ten minutes. The single decision to drop 15% of the *least valuable* traffic kept 100% of the *revenue* traffic fast. That is the entire argument for load shedding in one example: you do not lose 15% of value; you lose 15% of *volume* by sacrificing the cheapest 15%, and you keep the system from losing *everything*.

There is a subtle but important refinement that separates good shedders from great ones, used by Google and described in the SRE book: shed based on **CPU cost** and **client retry behavior**, not just request count. A request that the client will retry three times if dropped is more expensive to drop than to serve once (you pay the rejection cost four times). And a request whose work is cheap is worth admitting even under pressure. The most sophisticated shedders (Google's adaptive throttling, Netflix's prioritized load shedding) weight by *expected cost* and *priority* together, but the priority-tier version above captures 90% of the value and is the right place to start.

## Putting it together: the flash-sale timeline

Now watch all three mechanisms compose during ShopFast's actual flash sale, because the interplay is where the real understanding lives. The figure below is the timeline of the defended event — the same spike that crashed the undefended service, handled.

![A timeline of a flash sale showing the sale opening with five times traffic, latency rising as the adaptive limit detects it, the limit shrinking from 120 to 80, the shedder dropping 15 percent of low-priority analytics, checkout p99 holding at 90 milliseconds, and the limit reopening as the spike fades](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-7.webp)

At T+0 the sale opens and traffic jumps to 5x. The **gateway's per-customer token bucket** immediately does its job: it does not see "5x traffic," it sees thousands of individual customers each making a reasonable number of requests, plus a handful of scrapers and one buggy mobile build in a retry loop — and it throttles *those* to their quotas with `429`s, shaving the abusive tail off the spike before it reaches any service. What gets through is genuine demand, but it is still well above capacity.

At T+5s the order service's **adaptive concurrency limiter** notices latency ticking up from 50ms toward 60ms. The latency gradient turned positive — the signal that requests are starting to queue — so over the next few seconds it backs the concurrency limit down from 120 toward 80, the Little's-Law-correct point for the current dependency latency. By T+8s the limit has converged, and the service is admitting exactly as much as it can complete at 90ms p99. Everything beyond the limit gets a fast `503`.

At T+12s, even the limited admission is too much for the critical path, so the **priority shedder** engages: it starts dropping P2 analytics writes (and, briefly, some P1 browse traffic) with cheap `503`s, freeing concurrency budget for P0 checkouts. The combined effect: the slow inventory consumer's bounded queue applies **backpressure** to the event producer (capping memory), the concurrency limit bounds in-flight work, and the shedder protects the revenue path. At T+20s, checkout p99 is holding flat at 90ms while best-effort traffic is being shed — the system is *intentionally* doing less, and as a result it is doing the *important* things well. At T+5m the spike fades, latency drops, the adaptive limit climbs back toward 120, the shedder stops dropping, and the system returns to normal having never crashed, never paged anyone, and never lost a single checkout.

That is defense in depth for overload: rate limiting trimmed the abusive edge at the door, adaptive concurrency found the safe operating point, backpressure bounded the async pipeline, and shedding protected the core. No single mechanism would have been enough — the gateway limit alone wouldn't have stopped legitimate overload, backpressure alone wouldn't have helped the synchronous path, and shedding alone would have engaged later and harder without the concurrency limit smoothing the way.

## Optimization: where to place limits, and the numbers that prove the win

The patterns are clear; now the production engineering. Three optimization questions decide whether your defenses actually help or just add overhead.

**Where do you place the rate limit?** The principle is *reject as early and as cheaply as possible*. A request rejected at the gateway costs one token-bucket check (microseconds) and never touches a service. The same request, if it slips through to be rejected three services deep, has already burned a connection, a thread, and two network hops before it dies — and if it dies as a timeout rather than a fast reject, it has burned far more. ShopFast's numbers: rejecting abusive traffic at the gateway costs ~5μs of CPU per rejection; letting it reach the order service before rejecting costs ~2ms of CPU plus a database connection held for the duration — roughly 400x more expensive. So coarse, per-client abuse limits belong at the [gateway](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend); fine-grained, capacity-aware shedding belongs at the service (only the service knows its own current health). Both, not either.

**How do you make distributed rate limiting fast?** The Redis round trip on every request is the cost. The optimization, used by Stripe and Cloudflare, is the **local-budget** pattern: instead of consulting Redis per request, each node periodically claims a *batch* of budget from the central counter and spends it locally, syncing every, say, 100ms or every N requests. This drops the per-request cost from a ~0.5ms network round trip to a ~50ns local check — a 10,000x latency reduction on the limiter — at the cost of slightly looser enforcement (a node might overshoot by up to its local batch size near a window boundary). For abuse protection where "approximately 1,000 rps" is fine, this is an enormous win. For hard quotas where exactness is contractual, you pay the round trip. The choice is the usual accuracy-vs-latency trade-off, made explicitly.

**How aggressive should the adaptive limiter and shedder be?** This is where measurement matters. The metric that proves load shedding works is **goodput under overload** — successful, useful responses per second when offered load exceeds capacity. Plot goodput against offered load for the undefended and defended services and you see the whole story.

![A before and after comparison showing that at five times capacity a service admitting everything drives goodput to near zero through queueing and retry amplification, while a service that sheds the excess with sub-millisecond rejections holds goodput at its full real capacity](/imgs/blogs/rate-limiting-backpressure-and-load-shedding-9.webp)

#### Worked example: the goodput cliff, quantified

The undefended order service has a capacity of 1,600 rps. Offer it 1x (1,600 rps): goodput is 1,600 rps, p99 90ms — fine. Offer 2x (3,200 rps): queues build, p99 hits ~4s, ~60% of requests time out before being served, retries push effective load to ~4,500 rps, and goodput *falls* to ~600 rps. Offer 5x (8,000 rps): the service is OOM-killed; goodput is 0. The curve is a cliff: goodput peaks near 1x and collapses beyond it.

The defended service with shedding: offer 1x, goodput 1,600 rps. Offer 2x, the shedder rejects 1,600 rps of excess in microseconds each, admits 1,600, and goodput *holds* at 1,600 with p99 still 90ms. Offer 5x, the shedder rejects 6,400 rps (cheaply) and admits 1,600 — goodput *still* 1,600 rps, p99 *still* 90ms. The defended curve is flat: it serves its true capacity no matter how much excess you throw at it, because the excess is rejected before it can cause harm. The measured difference at 5x is the entire point of this post: 0 rps of goodput versus 1,600 rps, a crashed service versus a healthy one. The win is not subtle, and it is exactly measurable.

The general rule the numbers teach: **a system's goodput under overload is a design choice, not a given.** A naive system's goodput collapses past capacity; a system with shedding holds goodput at capacity indefinitely. The difference is a few hundred lines of admission control.

## Stress-testing the design

A senior does not ship a resilience design without breaking it on purpose. Three stress tests for ShopFast's setup.

**Traffic hits 5x capacity — does it fall over or shed?** We just quantified this: with shedding, goodput holds at 1x capacity and p99 stays flat; the excess gets fast `503`s. The one thing to verify is that the *rejection path itself* is cheap — if your `503` response triggers expensive logging, a database write, or a synchronous metric flush per rejection, then under a 5x spike you are doing 6,400 rejections/second of *expensive* work, and you have just moved the overload from the success path to the rejection path. The fix: rejections must be O(microseconds) — sample the logs (log 1 in 1,000 rejections, not all), increment an in-memory counter rather than writing per-rejection, and return a static `503` body. Shedding only works if shedding is *cheap*.

**One tenant floods the API — does it starve the others?** A single customer's integration goes haywire and fires 50,000 rps at ShopFast. With *only* a global concurrency limit and shedding, this is a problem: the flood fills the global admission budget and *legitimate* traffic from other tenants gets shed alongside the abuser. This is the noisy-neighbor failure, and the fix is **per-tenant rate limiting** at the gateway (the token bucket keyed by API key), which caps the abuser at its quota *before* it can consume shared capacity. This is exactly why rate limiting and shedding are *both* needed and not interchangeable: rate limiting enforces fairness *between* clients (the abuser is throttled to its share), while shedding enforces *survival* of the aggregate (even fair traffic gets trimmed if it exceeds total capacity). A robust design also adds a **fair-share concurrency** scheme (bulkhead-style per-tenant pools, from the [resilience patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads)) so no single tenant can monopolize even its admitted share.

**The async queue would grow forever — does backpressure hold?** The inventory consumer falls behind permanently (a downstream database is degraded, not just slow). With an unbounded queue this is the OOM crash. With the bounded queue and blocking backpressure, the producer slows to the consumer's rate — but now the *producer* is blocked, and if the producer is itself serving synchronous requests, *those* back up. Backpressure does not make the problem disappear; it *moves* it upstream to where you can make a policy decision. The correct end state: backpressure propagates from the slow consumer, through the bounded queue, to the producer, which (because it cannot enqueue) returns a fast `503` to *its* caller — at which point shedding takes over. Backpressure and shedding are the same spectrum: backpressure is "make the upstream wait," shedding is "tell the upstream no." When you cannot afford to make anyone wait (synchronous, latency-sensitive), backpressure degrades into shedding, and that is correct. This is the precise boundary the [message-queue backpressure post](/blog/software-development/message-queue/backpressure-and-flow-control) draws between block-or-drop.

## Case studies

**Stripe's rate limiting.** Stripe published a clear engineering account of how it protects its API, and it is a model of layered thinking. Stripe runs *multiple kinds* of limiters, not one: a **request-rate limiter** (a token bucket capping requests per second per user) and a **concurrent-request limiter** (capping how many requests a single user can have *in flight* at once, which catches a small number of very slow requests that a rate limiter alone misses). Crucially, Stripe also distinguishes *load shedders* from *rate limiters*: separate from the per-user quotas, it reserves capacity for *critical* traffic (a payment is more important than a list-charges call) so that even under aggregate overload, the money-moving requests get served while less-critical ones are shed. That is exactly the rate-limiting-plus-priority-shedding composition this post argues for, deployed at one of the highest-stakes APIs in the world. The lesson: one limiter is not enough — you need a rate limiter for fairness *and* a shedder for survival, and they are different tools.

**Cloudflare's distributed rate limiting.** Cloudflare operates rate limiting across hundreds of data centers globally, which is the distributed-counter problem at its hardest: a client's requests can land in any of hundreds of locations, and synchronizing a counter across the planet on every request would add unacceptable latency. Cloudflare's published approach uses the **local-budget / approximate counting** optimization at scale — each location enforces locally and synchronizes counts asynchronously, accepting that the global count is *eventually* accurate rather than *instantly* exact. For DDoS and abuse protection, "approximately right, instantly" beats "exactly right, too slow," and Cloudflare's scale makes the round-trip avoidance non-negotiable. The lesson: at sufficient scale, exact distributed rate limiting is impossible without unacceptable latency, and the right answer is *approximate* limiting with bounded error — the accuracy/latency trade-off made at planetary scale.

**Netflix's adaptive concurrency limits.** Netflix open-sourced its `concurrency-limits` library precisely because hard-coded limits kept being wrong. Its engineers described the core problem: a fixed concurrency limit set during a load test becomes incorrect the moment your dependencies' latency changes, your instance sizes change, or your traffic mix shifts — and a wrong limit either rejects traffic you could serve or admits traffic that melts you down. Their solution is the TCP-congestion-control-inspired, latency-gradient adaptive limiter we built above: measure latency, infer queueing from the gradient, and adjust the limit with additive-increase/multiplicative-decrease, so the limit finds itself per-instance and re-finds itself during incidents. The lesson: the right concurrency limit is not a number you set, it is a controller you run — and treating it as a measured, self-tuning quantity rather than a config value is the senior posture.

**The unbounded-queue meltdown.** A failure pattern documented in many post-mortems (it recurs because the lesson is hard-won): a service puts an *unbounded* in-memory queue or work buffer between two stages "to absorb spikes." For months it works perfectly — the queue absorbs every burst, the dashboards stay green. Then a downstream dependency slows for a sustained period, the queue's drain rate drops below its fill rate, and the backlog grows for tens of minutes, invisibly, until the process is OOM-killed — and because it restarts into the same overload with a now-empty queue and a thundering herd of retries, it crash-loops. The classic example is the **thundering-herd-plus-unbounded-queue** combination that has taken down systems at scale. The fix in every post-mortem is the same: bound the queue, decide block-or-drop at the bound, and never let an in-flight buffer grow without limit. The lesson: an unbounded queue is not resilience, it is a deferred crash with a hidden countdown — and the bound is the whole point.

## When to reach for each (and when not to)

A decisive guide, because over-applying these mechanisms adds complexity you may not need.

**Reach for rate limiting** the moment you have a public or multi-tenant API, *full stop*. Any API exposed to clients you do not control needs per-client rate limiting from day one — it is the cheapest insurance against a single buggy or malicious client taking down everyone else, and it doubles as quota enforcement for tiered plans. Put it at the gateway. Do *not* skip it because "our clients are well-behaved"; the buggy retry loop that floods you is almost always your *own* client, shipped last Tuesday.

**Reach for backpressure** whenever you have a producer and consumer that can run at different speeds — any queue, any stream, any pipeline. The rule is absolute: **never deploy an unbounded queue.** Every buffer gets a bound, and every bound gets a block-or-drop policy. In synchronous services this means bounded thread/connection pools and concurrency limits. Do *not* reach for elaborate reactive-streams machinery if a bounded blocking queue does the job — backpressure is a property of your bounds, not a framework you must adopt.

**Reach for load shedding** when you have a service whose overload would be catastrophic and whose traffic has *distinguishable priorities* — i.e., some requests matter much more than others. If all your traffic is equally important, random shedding (or just a concurrency limit that fast-rejects) is enough and priority tiers add complexity for no gain. The priority machinery earns its keep specifically when you have a revenue path to protect and a best-effort path to sacrifice — which, for most user-facing systems, you do.

**Reach for adaptive concurrency limits** when your dependency latency is variable and your traffic is spiky enough that a hard-coded limit is regularly wrong. For a service with stable, well-understood load behind fast dependencies, a static concurrency limit (sized by Little's Law) is simpler and fine. Adopt the adaptive controller when you are tired of re-tuning the constant, or when incidents keep proving the constant wrong — which, at scale, they will.

**When NOT to:** do not build any of this for an internal service with a handful of trusted callers and predictable load — a sensible timeout and a bounded connection pool is the whole resilience story you need, and a token bucket plus an adaptive limiter plus a priority shedder is over-engineering you will regret maintaining. These mechanisms are insurance against *adversarial or unpredictable* load. If your load is friendly and predictable, the premium is not worth it. As always in this series: the pattern is a cost, and you pay it only when the risk it insures against is real.

## Key takeaways

- **An overloaded service does not degrade gracefully on its own — it collapses.** Goodput peaks near capacity and falls off a cliff beyond it, driven by queueing latency, retry amplification, and resource exhaustion. The only cure is to *do less on purpose*.
- **The three mechanisms are different things.** Rate limiting is a *quota at the front door* (per client, proactive, returns `429`). Backpressure is a *signal between stages* (cooperative, returns "slow down"). Load shedding is *triage under duress* (reactive, drops low-priority work, returns `503`). They compose; they do not substitute.
- **Token bucket is the default API limiter.** Refill rate is the sustained rate, capacity is the burst. Return `429` with `Retry-After` — a limiter without `Retry-After` causes the retry storm it meant to prevent.
- **Distributed rate limiting must be atomic.** Naive per-node counters let one client exceed the limit N-fold; a naive Redis read-modify-write races. Use an atomic `INCR` or a Lua script, and fail *open* so the limiter never becomes the outage.
- **Never deploy an unbounded queue.** It is a deferred OOM crash with a hidden countdown. Bound every buffer and decide block (backpressure) or drop (shed) at the bound.
- **Don't hard-code the concurrency limit — measure it.** The right limit is `L = λ × W` (Little's Law) and it moves when dependency latency moves. An adaptive, latency-gradient controller finds it per-instance and re-finds it during incidents.
- **Shed the lowest-priority work first, and shed cheaply.** Dropping 15% of the *least valuable* traffic keeps 100% of the *revenue* traffic fast. But the rejection path must cost microseconds — sample logs, count in memory — or you just move the overload onto the `503` path.
- **Goodput under overload is a design choice.** A naive system collapses to zero at 5x; a system with shedding holds goodput at its true capacity indefinitely. The difference is a few hundred lines of admission control.
- **Rate limiting enforces fairness *between* clients; shedding enforces *survival* of the aggregate.** You need both, and per-tenant limiting is what stops the noisy neighbor from starving everyone else.

## Further reading

- *Release It!* by Michael Nygard — the canonical treatment of stability patterns, including the bulkhead, the circuit breaker, and the "fail fast" and "shed load" patterns that this post operationalizes.
- *Site Reliability Engineering* (Google), the "Handling Overload" and "Addressing Cascading Failures" chapters — the definitive account of adaptive throttling, criticality-based load shedding, and why retries amplify overload. Free online.
- Netflix's `concurrency-limits` library and its engineering blog post on adaptive concurrency limits — the production implementation of the TCP-Vegas-style limiter built here.
- Stripe's engineering blog, "Scaling your API with rate limiters" — the layered rate-limiter-plus-load-shedder design from a high-stakes payments API.
- Cloudflare's blog on distributed and account-level rate limiting — the approximate-counting / local-budget approach at planetary scale.
- The [message-queue backpressure and flow control post](/blog/software-development/message-queue/backpressure-and-flow-control) — the deep, end-to-end mechanism treatment for asynchronous pipelines, including reactive streams, credit-based flow, and TCP windows.
- The [resilience patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) and the [partial failures and graceful degradation post](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) — the companion patterns for protecting yourself from slow dependencies and for deciding what to return when a piece is missing.
- Once your service sheds correctly, you need to *prove* it is healthy and catch overload early: see the forthcoming siblings on [health checks, readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) and on [SLOs, golden signals, and alerting for microservices](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices). For where the per-client identity that drives rate-limit keys comes from, see [authentication and authorization with OAuth2, JWT, and token propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation).

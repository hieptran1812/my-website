---
title: "Timeouts, Retries, and Backoff Done Right: How Not to DDoS Yourself"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A field manual for the two most-used and most-misused reliability tools — how to bound timeouts to a dependency's p99, propagate deadlines, and retry with backoff, jitter, and a budget so you never amplify the outage you meant to survive."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "timeouts",
    "retries",
    "exponential-backoff",
    "jitter",
    "retry-budget",
    "deadline-propagation",
    "idempotency",
    "resilience",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/timeouts-retries-and-backoff-done-right-1.png"
---

At 02:11 on a Tuesday, the payments service started timing out. Not failing — timing out. The downstream fraud-scoring dependency had slowed from a 40-millisecond p99 to about 900 milliseconds because a cache node had fallen over and every lookup now hit the cold path. That alone would have been a minor brownout. Instead, within ninety seconds, the payments service itself fell over completely, the checkout API behind it stopped serving any requests at all, and the on-call for three separate teams got paged. The fraud dependency recovered its cache in four minutes. The outage lasted thirty-eight.

The autopsy was uncomfortable, because nobody had shipped a bug. Every line of code did exactly what it was written to do. The payments service called the fraud scorer with a generous timeout, and when the call was slow, it retried — three times, immediately, no backoff. So the moment the dependency slowed by a factor of twenty, every in-flight payment request turned into four requests against a dependency that could now barely serve one. The retries did not help a single user. They piled load onto a struggling service at the exact moment it had the least capacity to absorb it, and the pile-up climbed back up the stack: the fraud scorer's slowness held payments threads open, payments' exhausted thread pool held checkout connections open, and checkout — which talks to a dozen things and had nothing to do with fraud — stalled for everyone. A twenty-percent slowdown in one leaf dependency became a full outage of the front door.

That is the whole subject of this post in one story. Timeouts and retries are the two most-reached-for reliability tools in any distributed system, and they are the two most consistently misused. Done right, they are how you survive a flaky dependency without your users ever noticing. Done wrong, they are how you cause the outage they were meant to prevent. The thesis I want to nail into your head is this: **a retry is a small denial-of-service attack that you point at your own struggling dependency — unless it is bounded, backed off, jittered, budgeted, and idempotent.** And a timeout is the only thing standing between a single slow call and your entire thread pool. Get these wrong and no amount of redundancy, autoscaling, or heroics saves you, because the failure is self-amplifying: the more it hurts, the more load you generate.

![Diagram showing how a remote call with no timeout pins a thread and a connection so the hang climbs every layer of the stack until the edge service stalls for all users](/imgs/blogs/timeouts-retries-and-backoff-done-right-1.png)

By the end of this post you will be able to choose a timeout from a dependency's latency distribution instead of from a round number you liked; propagate a single deadline down a call chain so a 2-second user budget is not blown into 10 seconds by five hops each waiting 2 seconds; decide — from a small table — exactly which failures are safe to retry and which must never be; implement exponential backoff with full jitter that does not synchronize your clients into thundering herds; cap retries with a token-bucket budget so a failing dependency can never be retried into oblivion; and recognize the one structural rule that prevents the worst version of all of this, which is to retry at exactly one layer of your stack rather than at all of them. This sits squarely in the *running it* layer of the SRE loop — it is one of the highest-leverage things you can do to spend less of your error budget on self-inflicted outages. For the architectural framing of why systems degrade gracefully, the [SRE mindset intro](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) sets the stage, and the [microservices resilience-patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) covers the design-time mechanics this post operates.

## 1. Why every remote call needs a timeout

Start with the simplest, most violated rule in distributed systems: **every call that leaves your process must have a timeout.** Every network call, every database query, every cache lookup, every gRPC stub, every HTTP client request, every lock acquisition, every DNS resolution. If there is a remote thing on the other side that can be slow, there must be an upper bound on how long you are willing to wait for it.

The reason is not "to fail faster" — that is a side effect. The reason is that **a call with no timeout couples your liveness to the liveness of whatever you called.** When you make a synchronous remote call without a deadline, you are saying: I will wait as long as the network and the remote service make me wait, even if that is forever. And "forever" is not a rhetorical exaggeration. A TCP connection to a host that has silently dropped off the network — a hung VM, a black-holed firewall rule, a kernel that stopped acking — can sit in an established state for a very long time, because TCP keepalives are off by default and, even when on, default to two hours on Linux. So without an application-level timeout, a single dead dependency can hold one of your threads hostage for hours.

Now do the arithmetic that turns one slow call into an outage. Suppose your service runs a thread pool of 200 worker threads, and each incoming request makes one call to a dependency. Under healthy conditions that dependency answers in 40 ms, so a thread is busy for about 40 ms per request and your 200 threads can serve roughly $200 / 0.04 = 5{,}000$ requests per second. Now the dependency hangs — it stops responding entirely — and you have no timeout. Every request that arrives makes a call that never returns. Within a fraction of a second, all 200 threads are parked waiting on dead sockets. Now a brand-new request arrives that does not even touch that dependency. It cannot get a thread. It queues. The queue fills. Your service is down — not slow, down — and the cause is a dependency you call on some code path, holding resources you needed for every other code path.

This is the mechanism in figure 1. The hang does not stay where it started. It pins a thread, and behind that thread a connection from the caller, and behind that a connection from *their* caller, and the resource exhaustion climbs the stack one layer at a time until the service at the top — the one users actually talk to — runs out of the pool it needs to do anything at all. The blast radius of a missing timeout is never the one dependency; it is everything that shares a thread pool or connection pool with the code path that called it. This is exactly the cascading-failure shape that the [system-design treatment of cascading failures](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) describes from the architecture side; here we are looking at the operational trigger.

So the first rule is absolute and admits no exceptions: **no remote call without a timeout.** Audit your codebase for the default. Most HTTP clients ship with *no* timeout out of the box — Go's `http.Client{}` with a zero value, Python's `requests.get(url)` with no `timeout=`, a JDBC connection with no `socketTimeout`, a Jedis client with default settings. Every one of those is a latent outage waiting for the day a dependency hangs instead of erroring. The day it errors, you find out fast. The day it hangs, you find out by going down.

### Connection timeout versus read timeout versus total timeout

"A timeout" is actually three different timeouts, and conflating them causes real bugs. The **connection timeout** bounds how long you wait to establish a TCP connection (and a TLS handshake). The **read timeout** (or per-socket-operation timeout) bounds how long you wait for the next chunk of bytes once connected. The **total timeout** — the one that actually matters for the caller's experience — bounds the whole operation end to end, including connection, all reads, retries, and any redirects.

The classic trap is setting only a read timeout. You set `read_timeout=2s` and feel safe, but a server that sends one byte every 1.9 seconds keeps resetting the read clock and your call runs for minutes. Or you set a per-attempt timeout but allow retries, so the *total* wall-clock time is per-attempt-timeout times attempts, which is far longer than the user is willing to wait. Always reason about, and ideally enforce, the **total deadline** — the only number the caller upstream of you cares about. We will make this concrete with deadline propagation in section 4.

```python
import requests

# WRONG: no timeout at all. One hung server pins this thread forever.
resp = requests.get("https://fraud-scorer/score", json=payload)

# BETTER: (connect_timeout, read_timeout). Bounds the socket ops.
resp = requests.get(
    "https://fraud-scorer/score",
    json=payload,
    timeout=(0.2, 2.0),  # 200ms to connect, 2s per read
)

# BEST: enforce a TOTAL deadline that survives retries and redirects.
# requests has no native total timeout, so compute a per-attempt budget
# from the remaining deadline and never let retries exceed it (section 4).
```

## 2. Choosing the timeout: tie it to the dependency's p99, not a round number

So every call needs a timeout. What value? The wrong answer is "5 seconds because that felt safe." The right answer comes from the dependency's own latency distribution.

The principle: **a timeout should be set just above the latency you are actually willing to wait for, which for most calls is a small multiple of the dependency's healthy p99 — not its average, and not a number you guessed.** The p99 (the 99th-percentile latency: 99% of requests are faster than this) is the right anchor because it captures the realistic tail without chasing the pathological tail. If a dependency answers in 40 ms at p50, 80 ms at p99, then a 250 ms timeout gives healthy requests a comfortable buffer (three times p99) while still cutting off the calls that have clearly gone wrong. Setting the timeout at the average (40 ms) would abandon a large fraction of perfectly normal requests; setting it at 5 seconds means you wait 60 times longer than a healthy call ever needs before you give up — which is most of the way to "no timeout at all" in terms of how much resource a hang can consume.

Two failure modes bracket the right answer, and they trade off against each other directly:

**The timeout that is too long.** You set 5 seconds. A request that was going to take 4.8 seconds because of a transient blip eventually succeeds — good. But during a real incident, where the dependency is hung, every request holds a thread for the full 5 seconds before giving up. That is 5 seconds of pinned resources per request, which is how a dependency brownout becomes your outage. A too-long timeout is functionally close to no timeout when the dependency is sick, because the whole point of the timeout — releasing the resource so you can serve someone else — happens too late to matter.

**The timeout that is too short.** You set 50 ms on a dependency with an 80 ms p99. Now you are *abandoning one to two percent of healthy requests* — calls that would have succeeded in 60 or 70 ms get killed at 50, turned into errors (or, worse, retries), and your error rate jumps even though nothing is actually broken. A too-short timeout manufactures failures out of normal tail latency, and if you retry those manufactured failures you have built a load amplifier that fires under completely healthy conditions.

The honest way to choose, then, is to look at the histogram. In Prometheus, if your client instruments the dependency call as a histogram, you can read the p99 straight off the data:

```promql
# p99 latency of calls to the fraud-scorer over the last 5 minutes
histogram_quantile(
  0.99,
  sum by (le) (rate(fraud_scorer_client_duration_seconds_bucket[5m]))
)
```

Take that p99, look at how stable it is across the day (does it spike at peak?), and set the timeout at roughly two to four times the worst healthy p99. Then verify the choice empirically: alert if the rate of timeout errors under *healthy* conditions is non-trivial, because that means your timeout is too aggressive for the real distribution.

```promql
# Fraction of calls killed by our own timeout (should be near zero
# when the dependency is healthy; a spike means the timeout is too tight)
sum(rate(fraud_scorer_client_errors_total{reason="timeout"}[5m]))
/
sum(rate(fraud_scorer_client_requests_total[5m]))
```

#### Worked example: pricing the timeout for a 99.9% checkout SLO

Suppose checkout has a 99.9% availability SLO over 30 days. That is an error budget of roughly $0.001 \times 30 \times 24 \times 60 \approx 43.2$ minutes of allowed badness per month. Checkout calls the fraud scorer on every request. The scorer's healthy p99 is 80 ms; under load it drifts to 140 ms at the worst minute of the day. If I set the timeout at 120 ms, I will start killing a meaningful slice of healthy peak-hour requests — those legitimately slow 120–140 ms calls — and each killed call burns budget. If I set it at 800 ms (ten times worst-p99), I almost never kill a healthy call, but when the scorer hangs I hold a thread for 800 ms per request, and at 5,000 rps that is 4,000 thread-seconds of work piling up every second against a dependency that has stopped answering — I exhaust the pool in well under a second.

The compromise that respects both the SLO and the blast radius: set the per-attempt timeout at about 350 ms (≈2.5× worst-case p99), which clears virtually all healthy requests, and rely on a *fallback* — score the transaction as "review later" rather than block checkout — when the call does time out, so a scorer brownout costs me latency and a manual review queue, not checkout availability. The timeout protects my threads; the fallback protects the SLO. The numbers here are illustrative, but the method is the point: derive the timeout from the distribution, then design what happens when it fires.

| Candidate timeout | Healthy requests killed | Resource held during a hang | When it fails you |
| --- | --- | --- | --- |
| 50 ms (below p99) | 1–2% (manufactured errors) | Low | Healthy days: false failures, retries |
| 350 ms (≈2.5× p99) | ~0% | Moderate | Balanced; needs a fallback on fire |
| 5 s (≈60× p99) | ~0% | Very high | Incidents: pool exhaustion, your outage |
| None | ~0% | Unbounded | Any hang takes you fully down |

### The timeout fired — but did the work stop?

There is a trap inside the timeout that bites teams long after they have set sensible values, and it is worth a paragraph because it is invisible until it hurts. When your client gives up on a call at the 350 ms deadline, *your* side stops waiting — but the server on the other end usually has no idea you left. It is still holding the request, still running the query, still about to write the row, blissfully unaware that the only client who cared has hung up. This is **work abandonment**: the timeout protected the caller's thread but did nothing to stop the callee's work. Under a brownout this is doubly cruel — the dependency is already overloaded, you time out and retry, and now the dependency is doing the *original* abandoned work plus the retry's work, for a response that only one of them will ever deliver. The timeout, meant to shed load, has secretly multiplied it on the server.

The clean fix is **cancellation that propagates with the deadline**. In gRPC and in Go's `context`, when the client's deadline expires the context is cancelled and that cancellation travels to the server, which can check `ctx.Err()` and abandon its own work — stop the query, roll back the partial transaction, return early. A server that honors cancellation does not keep grinding on dead requests, so a timeout genuinely sheds load on both sides instead of just the client side. The discipline: do not just set a deadline, *check it* in any long-running handler, and make your downstream calls cancellable so the cancellation chains all the way down. A timeout without cancellation is half a timeout — it saves your threads but lets the dependency drown in abandoned work.

```go
// Server handler that HONORS cancellation: it stops working when the
// caller's deadline expires, instead of grinding on a dead request.
func (s *Pricing) Quote(ctx context.Context, req *pb.QuoteRequest) (*pb.QuoteResponse, error) {
    for _, sku := range req.Skus {
        // Bail out the instant the caller has given up.
        if err := ctx.Err(); err != nil {
            // ctx.Err() == context.Canceled or DeadlineExceeded.
            return nil, status.FromContextError(err).Err()
        }
        // This downstream call is also cancellable: it inherits ctx,
        // so when the deadline passes the DB query is aborted too.
        if err := s.priceOneSku(ctx, sku); err != nil {
            return nil, err
        }
    }
    return &pb.QuoteResponse{...}, nil
}
```

## 3. Deadline propagation: the budget is shared, not per-hop

Here is the subtlety that catches even strong engineers. You have done the right thing — every call has a timeout — but you set each one independently, and now the *end-to-end* latency budget is being violated even though no single timeout is.

Picture a request that flows through five hops: the gateway calls the orders service, which calls pricing, which calls inventory, which calls the warehouse. Each hop, sensibly, sets a 2-second timeout on its downstream call. The user, upstream of the gateway, is only willing to wait 2 seconds total. But the timeouts do not compose the way you want: in the worst case, the gateway waits 2 seconds for orders, which waited 2 seconds for pricing, which waited 2 seconds for inventory, and so on. Each hop is individually within its 2-second timeout, yet the user's request can take **up to 10 seconds**, because each hop reset the clock and started a fresh 2-second budget. The user gave up at 2 seconds and retried — and now there are two of these 10-second chains running, doing work for a response nobody is waiting for.

The fix is **deadline propagation**: the deadline is a single, shared, *shrinking* budget that travels with the request, not a fresh timeout granted at each hop. The gateway computes an absolute deadline — "abort this request at wall-clock time T = now + 2s" — and passes the *remaining time* down to orders. Orders, having spent 300 ms, tells pricing "you have 1.7 seconds left." Pricing, having spent 500 ms, tells inventory "you have 1.2 seconds left." When a hop receives a deadline that has already passed (or is too small to do useful work), it fails immediately instead of starting work that can never finish in time. The whole chain aborts at the user's 2-second budget, not at five times it.

![Timeline showing a 2-second user deadline propagated down five hops, with the remaining budget shrinking at each hop from 2000ms to 200ms until the budget is gone and the chain aborts at 2 seconds](/imgs/blogs/timeouts-retries-and-backoff-done-right-4.png)

Figure 4 shows the budget shrinking hop by hop: 2000 ms at the gateway, 1700 left at hop one, 1200 at hop two, 700 at hop three, 200 at hop four — and by the time the warehouse would be called, there is no budget left, so the chain aborts cleanly at the 2-second mark. Contrast that with the broken per-hop version where the chain could run to 10 seconds. This is the difference between figure 6's two columns, which I will get to in the next worked example.

The mechanism in practice is request context. In gRPC, deadlines are first-class: a client sets a deadline on the call, gRPC serializes the remaining time into the `grpc-timeout` header, and every downstream gRPC call made while handling that request inherits the remaining deadline automatically through the context. In Go, this is `context.Context` with `context.WithTimeout`, and the context carries the deadline through the entire call tree. In other ecosystems you propagate a deadline header yourself.

```go
// Go + gRPC: set ONE deadline at the edge; it propagates automatically.
func (s *Gateway) Checkout(ctx context.Context, req *pb.CheckoutRequest) (*pb.CheckoutResponse, error) {
    // The user is willing to wait 2s total for the whole chain.
    ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
    defer cancel()

    // This downstream gRPC call inherits the REMAINING budget, not a
    // fresh 2s. gRPC serializes it into the grpc-timeout header, and the
    // orders service's own downstream calls inherit what's left of it.
    resp, err := s.ordersClient.PlaceOrder(ctx, &pb.PlaceOrderRequest{...})
    if err != nil {
        // If ctx.Err() == context.DeadlineExceeded, the budget is gone.
        // Do NOT retry here — there is no time left to retry into.
        return nil, err
    }
    return &pb.CheckoutResponse{OrderId: resp.OrderId}, nil
}
```

```go
// In the orders service: before calling pricing, check there's enough
// budget left to bother, and pass the SHRUNK deadline down.
func (s *Orders) PlaceOrder(ctx context.Context, req *pb.PlaceOrderRequest) (*pb.PlaceOrderResponse, error) {
    if dl, ok := ctx.Deadline(); ok {
        remaining := time.Until(dl)
        if remaining < 50*time.Millisecond {
            // Not enough time to do useful work; fail fast, don't start.
            return nil, status.Error(codes.DeadlineExceeded, "no budget left for pricing")
        }
    }
    // ctx already carries the remaining deadline; pricing inherits it.
    price, err := s.pricingClient.Quote(ctx, &pb.QuoteRequest{...})
    // ...
}
```

The discipline this enforces is worth naming: **a deadline is a contract about the response the user is still waiting for.** Once the deadline passes, every microsecond of work down the chain is work for a corpse — the response will be discarded. Deadline propagation lets every hop stop doing dead work the instant the budget is gone, which is itself a load-shedding mechanism: under overload, requests that have blown their budget get dropped early instead of consuming capacity all the way down. This interacts directly with ordering and timing bugs across services, which the [debugging post on distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering) gets into when responses arrive after their deadline and get acted on anyway.

#### Worked example: the 2-second deadline blown to 10

A team reported that checkout occasionally took 8–10 seconds under partial degradation, even though every service had a "2-second timeout." The trace told the story: gateway → orders → pricing → inventory → warehouse, five hops, each with an independent `client.timeout = 2s`. When the warehouse slowed to ~1.9 seconds per call (just under its own timeout), every hop above it sat at ~1.9 seconds too, and the times stacked: roughly $5 \times 1.9 = 9.5$ seconds end to end. No single timeout fired, no error was logged, and the user's browser had given up at 2 seconds and retried — doubling the load on the already-slow warehouse.

The fix was not "make the timeouts shorter." Shortening each hop to 400 ms would have killed healthy requests (some hops legitimately need 600 ms) while still allowing $5 \times 0.4 = 2$ seconds in the bad case — fragile and arbitrary. The fix was deadline propagation: set one 2-second deadline at the gateway, propagate the remaining budget via gRPC context, and have each hop fail fast when the remaining budget dropped below the time it needed. After the change, the worst-case end-to-end latency was capped at the 2-second user budget by construction, the spurious browser retries stopped (because requests now failed at 2 seconds with a clear `DeadlineExceeded` instead of hanging), and the warehouse's retry-driven load dropped by about 35%. Latency at p99.9 went from ~9.5 s to ~2.0 s. The arithmetic is the whole lesson: **independent timeouts add; a shared deadline does not.**

![Before and after comparison showing five hops each granted a fresh two-second timeout blowing the budget to ten seconds versus a propagated deadline that holds the whole chain to two seconds](/imgs/blogs/timeouts-retries-and-backoff-done-right-6.png)

Figure 6 puts the two regimes side by side: on the left, five hops each reset to a fresh 2-second timeout, so the worst case is 5× the user's budget; on the right, the remaining deadline rides in the request context, so the chain aborts at or before 2 seconds and the budget is honored. The left column is what almost every system does by default. The right column is what you build on purpose.

## 4. When to retry, and when you must never

Now to retries, where the real damage gets done. A retry is the act of re-issuing a request that failed, in the hope that the failure was transient. That hope is sometimes well-founded and sometimes catastrophically wrong, and the difference is not subtle — it follows two questions you can answer mechanically.

**Question one: was the failure transient?** Retry only failures that have a real chance of succeeding on a second attempt. Those are *transient* failures — conditions that are about the moment, not about the request. A connection reset, a timeout, an HTTP 503 (Service Unavailable), an HTTP 429 (Too Many Requests, which explicitly invites a retry after a delay), a `DNS` hiccup, a leader election in progress: these can plausibly succeed if you wait and try again. By contrast, a *permanent* failure is about the request itself, and retrying it is pure waste plus extra load. An HTTP 400 (Bad Request) will be 400 every single time — the request is malformed, no amount of retrying fixes a bad request. A 401/403 (auth) will not fix itself by retrying. A 404 (Not Found) is not going to find the thing on attempt two. A 409 (Conflict) usually means the operation already happened or contradicts current state; retrying may make it worse.

**Question two: is the operation safe to retry?** Even a transient failure must not be retried blindly if the operation is not *idempotent* — that is, if doing it twice has a different effect than doing it once. This is the prerequisite, and I give it its own section (section 8) because it is the one people skip. For now: a read is naturally idempotent (GET it twice, same answer). A write often is not. "Charge the customer \$50" retried after a timeout might charge them \$100 — because the *first* attempt may have actually succeeded and you simply never got the response. The timeout told you nothing about whether the work happened; it only told you that you stopped waiting. Retrying a non-idempotent write on a timeout is how you double-charge a customer or double-ship an order.

These two questions combine into the decision table in figure 3, which you should treat as a checklist when writing any retry policy.

![Decision matrix mapping HTTP status classes against whether to retry, whether idempotency is required, and the backoff policy, showing that only transient failures on idempotent operations may be retried](/imgs/blogs/timeouts-retries-and-backoff-done-right-3.png)

The table in figure 3 reads like this: a timeout or a 503 — retry, but only if the operation is idempotent (or carries an idempotency key), with exponential backoff and jitter. A connection reset — same. A 429 — retry, but honor the `Retry-After` header the server sent you rather than your own schedule. A 400 or 409 — never retry, fail fast, surface the error to the caller. A non-idempotent write with no idempotency key — never retry; the safe move is to add an idempotency key so it *becomes* safe (section 8), not to retry and hope. Here is the same logic as a policy you can drop into a client:

```python
import random
import time

RETRYABLE_STATUS = {429, 500, 502, 503, 504}  # transient server-side
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 409, 422}  # request is the problem

def should_retry(exc=None, status=None, idempotent=False):
    # A non-idempotent operation is never auto-retried without an idem key.
    if not idempotent:
        return False
    # Transport-level transients: connection reset, read timeout.
    if exc is not None and isinstance(exc, (ConnectionResetError, TimeoutError)):
        return True
    if status in NON_RETRYABLE_STATUS:
        return False
    if status in RETRYABLE_STATUS:
        return True
    return False
```

Notice the structure: idempotency is checked *first* and gates everything. If the operation is not idempotent, no status code makes a retry safe. This ordering is deliberate and is the single most common thing teams get backwards — they decide retryability from the status code and forget to ask whether the operation can tolerate being done twice.

| Failure | Retry? | Reason |
| --- | --- | --- |
| Read timeout, idempotent GET | Yes, with backoff | Transient; safe to repeat |
| 503 on idempotent call | Yes, with backoff | Server overloaded; may recover |
| 429 Too Many Requests | Yes, honor Retry-After | Server told you to slow down |
| 400 Bad Request | Never | Request is malformed; retry is waste |
| 409 Conflict | Never (resolve, don't retry) | State already changed |
| Timeout on a non-idempotent write | Never without idem key | May double-apply the effect |
| Connection refused | Yes, sparingly | Often a deploy/restart; backoff hard |

## 5. The retry storm: a small DDoS you point at yourself

This is the section the whole post is built around, because it is the failure mode that turns retries from a safety net into the cause of the outage. I told the opening story; now let us dissect the mechanism precisely, because once you see the math you can never un-see it.

A **retry storm** (also called a thundering herd of retries, or retry amplification) happens like this: a dependency slows down or starts erroring. Every client that was calling it now sees failures, and every client retries. Those retries are *additional* load on the dependency — load that arrives precisely when the dependency has the least spare capacity, because it is already struggling. The extra load makes the dependency slower, which causes more failures, which causes more retries, which causes more load. The system has entered a self-sustaining overload: it is now generating its own traffic faster than it can serve it, and it will not recover on its own even after the original trigger is gone, because the retries are keeping it pinned down.

The key quantity is the **retry amplification factor**. If every failed request is retried up to $r$ times, then in the limit where the dependency is failing most requests, the load it sees is multiplied by roughly $(1 + r)$ — the original attempt plus $r$ retries. With $r = 3$ immediate retries, a struggling dependency that was already at its limit suddenly faces up to $4\times$ its normal request rate. No service survives a sudden $4\times$ load multiplier applied at the moment it is weakest. And if the retries happen at multiple layers (section 7), the amplification compounds multiplicatively, not additively — which is how you get from $4\times$ to genuinely catastrophic numbers.

![Before and after diagram contrasting an immediate triple retry that multiplies a twenty percent slowdown into four times the load and collapse against backoff plus jitter plus a ten percent budget that holds load near baseline](/imgs/blogs/timeouts-retries-and-backoff-done-right-2.png)

Figure 2 is the contrast in two columns. On the left, the storm: a dependency slows 20%, clients retry 3× with no backoff, load goes to ~4× and the dependency collapses into self-sustaining overload. On the right, the same trigger handled correctly: exponential backoff spreads the retries out in time, full jitter desynchronizes the clients so they do not retry in a wave, and a 10% retry budget caps the total retry load — so the dependency sees roughly 1.1× load instead of 4× and stays up. The trigger is identical in both columns. The only difference is whether the retries were engineered.

#### Worked example: a 20% slowdown becomes a full collapse, then the fix

Let me run the numbers concretely so the amplification is undeniable. A dependency normally serves 10,000 requests per second at the edge of its capacity — say it can do 12,000 rps before it falls over. One node in its fleet of five dies, so it loses 20% of capacity: it can now serve 9,600 rps, just under the offered 10,000. With no retries, this is a mild brownout: about 4% of requests fail or queue, the on-call gets a page, and it is annoying but survivable.

Now add immediate retries, $r = 3$, no backoff, retry on timeout and 503. The ~400 rps of requests that fail get retried, immediately, three times each. But those retries also hit a dependency now serving 9,600 against an offered load that just jumped. The offered load is no longer 10,000 — it is the original 10,000 plus the retries of everything that failed, and as the dependency slows further under that load, *more* requests cross into failure and get their own 3 retries. The feedback loop drives the effective offered load toward $4 \times 10{,}000 = 40{,}000$ rps against a service that can do 9,600. The dependency is now serving roughly a quarter of what is being thrown at it, almost everything fails, almost everything gets retried three times, and the service is in total collapse. A 20% capacity loss — recoverable on its own — became a 100% outage *because of the retries.* The retries did not just fail to help; they were the entire difference between a brownout and an outage.

Now the fix, same scenario. (1) Exponential backoff: first retry after ~1 s, second after ~2 s, third after ~4 s. This spreads the retry load over seconds instead of slamming it in milliseconds, so the instantaneous multiplier is far lower. (2) Full jitter: randomize each backoff so clients do not all retry at $t = 1\text{s}$ together. (3) A retry budget: retries may be at most 10% of total requests, enforced by a token bucket, so even if everything is failing the dependency sees at most $1.10 \times$ baseline from retries, never $4\times$. With these three in place, the same single-node failure stays a brownout: ~9,600 rps of capacity against ~10,000 offered plus at most ~1,000 rps of budgeted, spread-out retries, the dependency limps along at high-but-survivable load, the dead node gets replaced in a few minutes, and the system recovers cleanly. Measured before→after on a real incident pattern like this: a dependency outage that previously cascaded into a 30-plus-minute multi-service outage instead became a 5-minute single-service brownout with zero cascading pages. That is the entire return on getting retries right.

## 6. Exponential backoff and the indispensable jitter

Backoff is the answer to "how long should I wait before retrying?" The naive answer — retry immediately — is what caused the storm above. The standard answer is **exponential backoff**: wait a base interval, and double it on each successive retry. Wait 1 second, then 2, then 4, then 8. The reasoning is that if a quick retry did not work, the problem is probably not momentary, so waiting progressively longer both gives the dependency real time to recover and reduces the rate at which you pile on load. Exponential growth means even a long-lived outage only costs a bounded number of attempts before you back off to minutes-apart probes.

But exponential backoff alone has a vicious failure mode, and it is the part people skip: **without jitter, backoff creates synchronized thundering herds.** Picture the storm again. The dependency hiccups for one second at $t = 0$, and 10,000 clients all see a failure at roughly the same instant. They all back off the same 1 second. So at $t = 1\text{s}$, all 10,000 retry *at the same moment*. They overwhelm the just-recovering dependency, which fails them all again, so they all back off 2 seconds, and at $t = 3\text{s}$ all 10,000 retry together again. Backoff did not spread the load out — it just rescheduled the entire herd to stampede in synchronized waves. The dependency now faces periodic 10,000-rps spikes with quiet gaps between them, which is in some ways *worse* than constant load, because it never gets a stable window to drain its queue.

**Jitter** fixes this by randomizing each client's backoff so the herd desynchronizes. There are two well-known flavors, both from the AWS Architecture Blog's analysis of exponential backoff and jitter:

**Full jitter:** instead of sleeping exactly `base * 2^attempt`, sleep a uniformly random duration between 0 and that cap. So if the computed backoff is 4 seconds, you sleep a random amount in $[0, 4]$ seconds. This maximally spreads the retries — clients that computed the same 4-second backoff now retry uniformly across a 4-second window, smearing what would have been a spike into a flat plateau. Full jitter empirically minimizes both the total work and the peak concurrent load, which is why it is the default recommendation.

**Equal jitter:** sleep half the computed backoff plus a random amount in $[0, \text{half}]$. So for a 4-second backoff you sleep $2 + \text{rand}(0,2)$ seconds. This guarantees a minimum wait (you never retry too eagerly) while still spreading the herd. It is a reasonable choice when you want a floor on the wait, but full jitter is usually better at reducing peak load.

Here is the correct implementation — the artifact you should copy. This is exponential backoff with full jitter, a cap on the per-retry sleep, a cap on the number of attempts, and respect for a total deadline:

```python
import random
import time

def call_with_backoff(do_call, *, max_attempts=4, base=1.0, cap=20.0, deadline=None):
    """
    Exponential backoff with FULL jitter.
    - do_call() performs one attempt; raises on a retryable failure.
    - max_attempts bounds the total tries (the original + retries).
    - base is the first backoff seconds; doubles each attempt.
    - cap bounds any single sleep so backoff can't grow unbounded.
    - deadline (epoch seconds) is the hard total budget; we never sleep past it.
    """
    attempt = 0
    while True:
        try:
            return do_call()
        except RetryableError:
            attempt += 1
            if attempt >= max_attempts:
                raise  # bounded: give up, surface the failure
            # Exponential window, capped: base * 2^(attempt-1), but <= cap.
            window = min(cap, base * (2 ** (attempt - 1)))
            # FULL JITTER: sleep a uniform random amount in [0, window].
            sleep_s = random.uniform(0, window)
            # Respect the TOTAL deadline: don't sleep past it.
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise  # no budget left; stop retrying
                sleep_s = min(sleep_s, remaining)
            time.sleep(sleep_s)
```

The four properties that make this safe are exactly the ones the naive version lacks: it is **bounded** (`max_attempts`), it **backs off** (`base * 2^(attempt-1)`), it is **jittered** (`random.uniform(0, window)`), and it respects the **total deadline** (`deadline`). The only thing it does not yet do is enforce a fleet-wide budget — one client backing off politely does not stop ten thousand clients from collectively overwhelming a dependency — which is the next section.

A note on the math of *why* exponential matters: with a fixed backoff of, say, 1 second, a 10-minute outage means each client makes 600 attempts. With exponential backoff capped at 20 seconds, the same outage costs roughly $1 + 1 + 2 + 4 + 8 + 16 + 20 + 20 + \dots$ — you reach the 20-second cap after about 5 retries and then probe every 20 seconds, so a 10-minute outage costs on the order of 30 attempts, not 600. A 20× reduction in wasted attempts per client, multiplied across the fleet, is the difference between a recoverable dependency and a buried one.

## 7. Retry budgets and retrying at exactly one layer

Backoff and jitter shape *when* an individual client retries. But reliability is a fleet property, and there are two more controls that operate at the fleet and architecture level: the retry budget, and the rule of retrying at exactly one layer.

### The retry budget: cap retries as a fraction of traffic

Even with perfect backoff, a large enough fleet can retry a struggling dependency into the ground — each client is polite, but there are a hundred thousand of them. The defense is a **retry budget**: a hard cap on retries expressed as a *fraction of total requests*. The rule is something like "retries may be at most 10% of the requests we send to this dependency." The moment retries would exceed that fraction, you stop retrying and just return the failure to the caller. This is the mechanism gRPC's built-in retry support uses (`retryThrottling` with a token-ratio), and it is what Envoy and finagle-style clients implement as well.

The clean implementation is a **token bucket**. Each successful request adds a token (or a fraction of one); each retry costs a token. When the bucket is empty, retries are throttled off until success refills it. The beauty of this design is that it is *self-adapting*: when the dependency is healthy, successes constantly refill the bucket and retries flow freely; when the dependency is failing, there are no successes to refill the bucket, it drains, and retries automatically shut off — exactly when retries would be most harmful. You do not need to detect the outage; the budget detects it for you by running dry.

```yaml
# gRPC service config: retry policy + a fleet-wide retry budget.
# retryThrottling caps retries as a RATIO of requests across the channel.
{
  "methodConfig": [
    {
      "name": [{ "service": "fraud.Scorer" }],
      "retryPolicy": {
        "maxAttempts": 4,
        "initialBackoff": "1s",
        "maxBackoff": "20s",
        "backoffMultiplier": 2,
        "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED"]
      }
    }
  ],
  "retryThrottling": {
    "maxTokens": 100,
    "tokenRatio": 0.1
  }
}
```

Read the `retryThrottling` block carefully. `maxTokens: 100` is the bucket size; `tokenRatio: 0.1` means each successful request adds 0.1 tokens back and each retry costs 1 token (a failed RPC costs a token, a successful one returns `tokenRatio`). The steady-state math: the bucket can only sustain retries at a rate where token-cost equals token-refill, which is exactly the `tokenRatio` — so retries settle at about 10% of successful traffic. When success dries up, refill stops, the 100 tokens drain after ~100 retries, and further retries are throttled. The dependency can never be retried into oblivion because the budget physically prevents it. Here is the same idea as a standalone token-bucket you can reason about:

```python
class RetryBudget:
    """Token-bucket retry budget: retries capped at ~ratio of requests."""
    def __init__(self, max_tokens=100, token_ratio=0.1):
        self.max_tokens = max_tokens
        self.token_ratio = token_ratio
        self.tokens = max_tokens

    def on_success(self):
        # Each success refills a fraction of a token.
        self.tokens = min(self.max_tokens, self.tokens + self.token_ratio)

    def can_retry(self):
        # A retry costs one whole token; refuse if the bucket is dry.
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False  # budget exhausted -> fail fast, don't pile on
```

### Retry at exactly one layer, not at every layer

The second architectural control is the one that prevents the worst-case amplification. **Retries compound multiplicatively down a call stack.** If the edge retries 3×, and the edge calls a service that *also* retries 3× on its downstream, and *that* service retries 3× again, then a single user request can become $3 \times 3 \times 3 = 27$ attempts against the bottom dependency — or, across a 4-deep stack all retrying 3×, $3^4 = 81$ attempts. The amplification is not the sum of the retries; it is the product. This is the most dangerous retry mistake because each layer looks reasonable in isolation — "of course we retry our database calls" — but together they build a load multiplier that detonates the instant the bottom dependency slows.

The rule: **retry at exactly one layer.** Pick the layer — usually the edge or the layer closest to the user that can meaningfully retry, or a single chosen middle layer — and have *every other layer fail fast*, surfacing the error upward without retrying. The retrying layer owns the entire retry policy, budget included; the inner layers just pass failures up. This caps the total amplification at the single layer's retry count (3× or 4×, bounded and budgeted) instead of the product of all of them.

![Decision tree contrasting retrying at every layer of a four-deep stack which multiplies to eighty-one attempts against retrying at one chosen layer where the edge owns bounded retries and inner layers fail fast](/imgs/blogs/timeouts-retries-and-backoff-done-right-7.png)

Figure 7 lays out the decision: the dangerous branch — retry at every layer — leads to $3^4 = 81$ attempts and a retry storm; the safe branch — retry at one layer — gives the edge bounded 3× retries while the inner layers do no retrying at all. In practice this means turning *off* the default retry behavior in your inner-service clients and your database/cache drivers when those calls sit beneath a retrying layer, which feels counterintuitive ("why would I not retry my DB call?") until you have watched a four-layer retry stack turn a blip into an 81× hammer. The microservices [resilience-patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) makes the same argument from the service-mesh angle, where a sidecar that retries plus an application that retries silently doubles your amplification.

#### Worked example: 3× per layer becomes 81×, and the page that proved it

A team had retries on at four layers: the CDN/edge retried failed origin fetches 3×, the API gateway retried failed service calls 3×, each service retried failed database calls 3×, and the database client library retried failed connections 3×. Each was added by a different team at a different time, each defensible alone. When the database primary failed over (a ~15-second blip while the replica was promoted), the bottom layer saw not the normal ~2,000 connection attempts per second but, in the worst case, $3^4 = 81$ times the failed-request rate hammering the database during the exact 15 seconds it was least able to accept connections. The failover, which should have been a 15-second partial degradation, turned into a 12-minute outage because the newly-promoted replica was immediately buried under 81× connection load and could not stabilize.

The fix was surgical: keep retries at the edge only (with backoff, jitter, and a 10% budget), and turn off application-level and driver-level retries on every inner layer beneath it, letting them fail fast and propagate the error up to the edge's single retry policy. The next time the database failed over — and databases fail over routinely — the edge retried with backoff, the inner layers passed failures straight up, the newly-promoted replica saw at most ~1.1× normal connection load, and it stabilized in the expected ~15 seconds. The failover became a non-event: a brief spike in edge-level retries, zero cascading pages, the error budget barely touched. Same database behavior; the difference was entirely in how many layers were allowed to retry.

## 8. Idempotency: the prerequisite that makes retries safe at all

I have said "retry only if idempotent" several times; now I owe you the substance, because idempotency is the load-bearing prerequisite for the entire practice. **You can only safely retry an operation that is idempotent or carries an idempotency key.** If you cannot guarantee that doing the operation twice has the same effect as doing it once, then a retry is not a safety mechanism — it is a correctness bug waiting for a timeout.

An operation is **idempotent** if applying it multiple times produces the same result as applying it once. "Set the user's email to x@y.com" is idempotent — do it five times, the email is still x@y.com. "Read the account balance" is idempotent. "Append a charge of \$50 to the ledger" is *not* idempotent — do it twice, the customer is charged \$100. HTTP gives you a hint here: GET, PUT, and DELETE are defined to be idempotent; POST is not. But the HTTP method is only a convention; what matters is your actual semantics.

The deep reason idempotency matters for retries is the **ambiguity of a timeout**. When a write times out, you are in a genuinely uncertain state: the request may have never arrived, or it may have arrived, executed completely, and committed — and the response simply got lost on the way back to you. The timeout cannot distinguish these. So if you retry a non-idempotent write on a timeout, you are gambling: if the first attempt failed, the retry fixes it; if the first attempt secretly succeeded, the retry double-applies it. For a charge, a transfer, an order, or an inventory decrement, that gamble is unacceptable.

The fix is the **idempotency key**: the client generates a unique key for the logical operation (a UUID, often) and sends it with every attempt, including retries. The server records which keys it has already processed; if it sees a key it has handled, it returns the stored result of the first execution instead of executing again. Now a retry is safe *even for a non-idempotent operation*, because the key makes the *effect* idempotent at the server even though the operation is not naturally so.

```python
# Client: generate ONE key for the logical operation; reuse it on every retry.
import uuid

idem_key = str(uuid.uuid4())  # generated ONCE, before the first attempt

def charge_once():
    return http.post(
        "https://payments/charge",
        headers={"Idempotency-Key": idem_key},  # SAME key on every retry
        json={"amount_cents": 5000, "account": "acct_123"},
    )

# Retrying charge_once() is now safe: the server dedups on idem_key.
result = call_with_backoff(charge_once, max_attempts=4, base=1.0, cap=20.0)
```

```python
# Server: dedup on the key. The first execution wins; retries replay its result.
def handle_charge(req):
    key = req.headers["Idempotency-Key"]
    existing = idem_store.get(key)        # atomic check (e.g. INSERT ... ON CONFLICT)
    if existing is not None:
        return existing.stored_response   # replay, do NOT charge again
    response = execute_charge(req)         # the actual, non-idempotent work
    idem_store.put(key, response)          # record under the key
    return response
```

There is one subtlety in the server code worth flagging, because it is where naive implementations still go wrong: the check-and-store must be **atomic**, or two concurrent retries race through the gap between `get` and `put` and both execute the charge. This is a genuine danger because retries are *exactly* the scenario that produces concurrent in-flight copies of the same logical request — the client timed out and fired a retry while the original was still running on the server, so now two requests with the same key are live at once. The fix is to make the key insert the atomic operation: `INSERT ... ON CONFLICT DO NOTHING` returning whether the row was new, or a conditional write keyed on the idempotency key, so that exactly one of the racing requests "wins" the insert and proceeds to execute while the other sees the conflict and waits for or replays the winner's result. If you skip the atomicity and just do read-then-write, you have built an idempotency layer that fails precisely under the concurrent-retry conditions it exists to protect against. This is the same ordering hazard the debugging series dissects in [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering): two operations whose interleaving determines correctness, where the bug only appears under the timing that retries make common.

This is the **at-least-once delivery plus dedup** pattern, and it is the same shape that message queues use to make at-least-once delivery tolerable: deliver possibly-more-than-once, dedup on a key to get effectively-once. The message-queue series covers the storage and edge cases of this pattern thoroughly in [idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the race when two retries arrive concurrently, how long to retain keys, how to make the check-and-store atomic — and the same reasoning applies verbatim to HTTP retries. A practical note on key retention: keep idempotency keys long enough to outlive the longest plausible retry window (a client that backs off to minutes-apart probes, plus clock skew, plus a queued retry that sat in a buffer) — usually 24 hours is a safe floor for payment-class operations — but not forever, or your dedup store grows without bound. The takeaway for retries specifically: **before you turn on retries for any write path, the first question is "is there an idempotency key?" If not, you do not have a retry policy — you have a double-spend generator.**

![Stack diagram showing the five gates a retry must pass in order — bounded, backed off, jittered, budgeted, and idempotent — before it helps rather than amplifies an outage](/imgs/blogs/timeouts-retries-and-backoff-done-right-5.png)

Figure 5 collects the whole discipline into the five gates a retry must pass before it is safe: **bounded** (a hard attempt cap), **backed off** (exponential growth between attempts), **jittered** (randomized so clients desynchronize), **budgeted** (a fleet-wide token-bucket cap as a fraction of traffic), and **idempotent** (or carrying an idempotency key). A retry that skips any one of these gates is the dangerous kind. A retry that passes all five is the kind that quietly saves your night.

## 9. Circuit breakers: the escalation when retries are not enough

Retries, even perfect ones, assume the failure is transient and brief. But what if the dependency is not having a hiccup — what if it is genuinely down, or so degraded it will be sick for minutes? Retrying into a sustained outage, even with backoff and a budget, is wasted work: you spend latency and resources on calls that will fail, and you keep poking a dependency that needs to be left alone to recover. The escalation is the **circuit breaker**.

A circuit breaker wraps the call to a dependency and tracks its recent failure rate. While failures are low it stays **closed** — calls pass through normally. When failures cross a threshold (say, more than 50% of calls fail over a rolling window), the breaker **opens**: it stops sending calls to the dependency entirely and instead fails fast (or serves a fallback) for a cooldown period. After the cooldown, it goes **half-open**: it lets a single trial request through to probe whether the dependency has recovered. If the probe succeeds, the breaker closes and normal traffic resumes; if it fails, the breaker opens again for another cooldown. The breaker is what stops you from hammering a dead dependency, and it gives that dependency the breathing room to come back.

![Graph showing a circuit breaker escalating past retries by opening when failures cross a threshold, serving a fallback while open, then probing with a half-open trial before closing again once recovered](/imgs/blogs/timeouts-retries-and-backoff-done-right-8.png)

Figure 8 shows the escalation path: a call goes through bounded retries; if failures cross the threshold over the window, the breaker trips open and starts failing fast and serving a cached or default fallback; after a cooldown it goes half-open and sends one probe; if the probe succeeds it closes again and recovered traffic resumes. Notice how the breaker and the retry compose — retries handle the brief transient, the breaker handles the sustained outage, and the fallback is what keeps the user experience intact while the breaker is open. A breaker config in a resilience library looks like this:

```yaml
# resilience4j-style circuit breaker for the fraud-scorer dependency.
resilience4j:
  circuitbreaker:
    instances:
      fraudScorer:
        slidingWindowType: COUNT_BASED
        slidingWindowSize: 50            # decide over the last 50 calls
        minimumNumberOfCalls: 20         # don't trip on a tiny sample
        failureRateThreshold: 50         # open when > 50% of calls fail
        slowCallRateThreshold: 80        # treat slow calls as failures too
        slowCallDurationThreshold: 350ms # "slow" == over our timeout budget
        waitDurationInOpenState: 10s     # cooldown before half-open
        permittedNumberOfCallsInHalfOpenState: 3  # probe with 3 trials
        automaticTransitionFromOpenToHalfOpenEnabled: true
```

This is a forward reference: the breaker, alongside bulkheads and load shedding, deserves its own treatment, which the sibling post on circuit breakers, bulkheads, and load shedding (planned slug `circuit-breakers-bulkheads-and-load-shedding`) will cover in full, and which the [designing-for-failure](/blog/software-development/site-reliability-engineering/designing-for-failure) post frames as a design principle. For now, hold the layering in mind: **timeout bounds one call; retry handles a transient blip; the breaker handles a sustained outage; the fallback preserves the user experience through all of it.** Each is the escalation when the previous one is not enough.

## 10. Observing retries: make the storm visible before it kills you

Everything above is prevention. But you also need to *see* a retry storm forming, because the failure mode is fast — from healthy to collapsed in under two minutes in the opening story — and the signal is hidden if you only watch top-line request rates. The measurement discipline is straightforward and pays for itself the first time it catches a storm in its early seconds: **instrument retries as a first-class metric, alert on the retry ratio, and put it on the dependency's dashboard right next to its latency and error rate.**

The single most useful number is the **retry ratio**: retries as a fraction of total attempts to a dependency. In a healthy system this sits near zero — a few tenths of a percent, the occasional transient. When a dependency starts to slow, the retry ratio climbs *before* the error rate does, because retries are the leading indicator: the system is failing-and-retrying internally before those failures become user-visible errors. A retry ratio crossing a few percent is your early warning that the amplification loop is spinning up.

```promql
# Retry ratio: what fraction of our attempts to the dependency are retries.
# Climbs BEFORE user-facing errors do -> a leading indicator of a storm.
sum(rate(fraud_scorer_client_attempts_total{attempt="retry"}[1m]))
/
sum(rate(fraud_scorer_client_attempts_total[1m]))
```

Pair that with the **retry-budget exhaustion** signal — when the token bucket is draining or dry, you are actively throttling retries, which means the dependency is sick enough that the budget is doing its job. That is exactly when you want a page, because the budget has converted "we are about to retry-storm a dependency into the ground" into "we are shedding retries and surfacing failures fast," and a human should know the dependency is in trouble.

```yaml
# Prometheus alert: the retry amplification loop is spinning up.
groups:
  - name: retries
    rules:
      - alert: RetryRatioHigh
        expr: |
          sum(rate(fraud_scorer_client_attempts_total{attempt="retry"}[5m]))
          /
          sum(rate(fraud_scorer_client_attempts_total[5m])) > 0.10
        for: 2m
        labels:
          severity: page
        annotations:
          summary: "Retries are over 10% of attempts to fraud-scorer"
          description: >
            Retry ratio {{ $value | humanizePercentage }} for 2m.
            The amplification loop is forming; check dependency latency
            and confirm the retry budget is throttling. Runbook: trip the
            breaker or disable retries to break the loop if it is storming.
```

The honest way to *measure the win* from this whole post is a before→after on a real dependency-degradation event. Before: a single-node dependency failure cascaded into a 30-plus-minute, multi-service outage with three teams paged, because immediate retries multiplied the load 4× across two retry layers. After applying timeouts-from-p99, deadline propagation, backoff with full jitter, a 10% retry budget, and single-layer retries: the same class of event became a ~5-minute single-service brownout, zero cascading pages, the retry ratio peaked at ~9% (budget holding), and the error budget barely moved. Those are the four numbers that prove the change worked — outage duration, blast radius (services affected), pages, and budget burned — and they are all measurable from your incident records and your Prometheus data without any guesswork. The [SRE post on choosing SLIs that reflect user pain](/blog/software-development/site-reliability-engineering/choosing-slis-that-reflect-user-pain) is the right companion for turning these raw retry metrics into an SLI that actually tracks what users feel.

## 11. War story: the cascading retry storms that taught the industry

The patterns in this post are not theoretical; they are the distilled scar tissue of some of the most studied outages in the field. Three deserve naming.

**The AWS DynamoDB / 2015 metadata storm.** In September 2015, an Amazon DynamoDB disruption in the US-EAST-1 region began as a brief network issue that caused storage-server membership requests to time out. The retries on those requests — combined with a recently increased metadata-table size that made each request slower — created a load spike on the metadata service that exceeded its capacity. The retries kept the metadata service pinned even after the original network blip cleared, turning a transient issue into a multi-hour event that affected dependent services. The fix Amazon described afterward was textbook: reduce the load via better request handling, increase capacity, and — crucially — improve how retries and backoff interact with an already-stressed control plane. It is the retry-storm mechanism from section 5, at the scale of a region. The exact internal numbers are Amazon's; I am summarizing the public post-event description, so treat the magnitudes as directional.

**The exponential-backoff-and-jitter analysis (AWS, 2015).** Around the same era, AWS published a now-canonical analysis showing, with simulation, that exponential backoff *without* jitter barely reduces the competing-client load during contention, while exponential backoff *with* full jitter dramatically reduces both the total work and the peak concurrent calls needed to drain a contended resource. That single piece of analysis is why "full jitter" is the default recommendation everywhere today, and why the implementation in section 6 randomizes the sleep rather than waiting a fixed exponential interval. If you take one citation from this post into your next code review, make it this one.

**The thundering herd of synchronized clients.** A pattern seen across many systems — cron jobs that all fire at the top of the minute, clients that all reconnect after a network partition heals, caches that all expire the same key at the same TTL — is the synchronized herd. The fix is the same in every case: add jitter. Stagger the cron jobs, jitter the reconnect backoff, add a random spread to cache TTLs. The lesson the industry internalized is that *synchronization is the enemy of stability*: any time a large population of clients does the same thing at the same time, you have built a load spike, and jitter is the cheapest, most effective desynchronizer there is.

The through-line across all three is the thesis of this post: retries and synchronized behavior are load multipliers, and the controls — backoff, jitter, budgets, single-layer retry, and the circuit breaker as escalation — exist specifically to keep those multipliers from detonating when a dependency is already weak. For the broader anatomy of how these failures unfold and what the postmortems found, the [system-design post on the anatomy of an outage](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) is a good companion read.

## 12. How to reach for this (and when not to)

Every control here has a cost, and a principled engineer says plainly when a control is not worth it. Here is the decisive guidance.

**Always set a timeout.** There is no exception. Every remote call gets a deadline tied to the dependency's healthy p99 (2–4×), with a fallback defined for when it fires. This is not optional and it is not expensive. If you do nothing else from this post, audit your codebase for clients with no timeout and fix them today.

**Propagate a deadline when you have a call chain of three or more hops, or any user-facing request that fans out.** For a single direct call, a plain timeout is enough. The moment a request flows through multiple synchronous hops, propagate the deadline (gRPC does it for you; otherwise pass a deadline header) so independent timeouts cannot add up. For a two-hop internal batch job with no user waiting, do not over-engineer it.

**Retry only transient failures on idempotent operations, with backoff, jitter, and a budget.** And retry at *one* layer. The most common over-application is retrying everywhere "to be safe" — which builds the multiplicative amplification of section 7. Be deliberate about which layer owns retries and turn the rest off.

**When NOT to retry at all:** Do not retry a non-idempotent write that has no idempotency key — fix the idempotency first. Do not retry a 400/401/403/404/409 — the request is the problem and a retry is pure waste. Do not add retries beneath a layer that already retries. Do not retry a request whose deadline has already passed — there is no time left to retry into. And do not add a circuit breaker to a call that has no fallback and no alternative; a breaker that opens and then just fails fast with no fallback has only converted a slow failure into a fast one, which is sometimes the right call (it protects your threads) but is not free reliability.

**When a circuit breaker is overkill:** for a single low-traffic internal dependency where a slow failure is harmless and a fallback does not exist, a good timeout plus bounded retries may be all you need. Breakers earn their complexity on high-traffic, user-facing paths where a sustained dependency outage would otherwise pin your resources. The mitigate-first instinct from the sibling post [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) applies during the incident: if retries are storming, the fast mitigation is to *turn retries off* or trip the breaker manually, not to debug the dependency live.

## 13. Key takeaways

- **Every remote call needs a timeout.** No timeout means a hung dependency pins your threads and connections, and the hang climbs the stack until you are down. Audit for client libraries with no default timeout.
- **Set the timeout from the dependency's healthy p99**, not a round number — roughly 2–4× p99. Too long behaves like no timeout during an incident; too short manufactures failures out of normal tail latency.
- **Propagate one shared deadline down the call chain.** Independent per-hop timeouts add up — five hops at 2s each can blow a 2s user budget to 10s. A propagated deadline (gRPC context) caps the whole chain at the user's budget.
- **A retry is a small DDoS you point at your own struggling dependency** unless it is bounded, backed off, jittered, budgeted, and idempotent. Immediate 3× retries turn a 20% slowdown into 4× load and a full collapse.
- **Retry only transient failures on idempotent operations.** A timeout or 503 on an idempotent call — retry. A 400 or a non-idempotent write without an idempotency key — never.
- **Exponential backoff needs jitter or it just synchronizes the herd.** Without jitter, all clients back off the same interval and stampede together. Full jitter (random in [0, window]) spreads them into a flat plateau.
- **Cap retries with a token-bucket budget** — retries at most ~10% of traffic — so a failing dependency cannot be retried into oblivion. The budget self-adapts: it dries up exactly when retries would be most harmful.
- **Retry at exactly one layer.** Retries compound multiplicatively: 3× at each of four layers is 81×. Pick one layer to own retries; make the rest fail fast.
- **Idempotency is the prerequisite for safe retries.** If you cannot make the effect idempotent (naturally or via an idempotency key), you do not have a retry policy — you have a double-spend bug.
- **The circuit breaker is the escalation when retries are not enough** — it handles the sustained outage that backoff cannot, and the fallback preserves the user experience while it is open.

## Further reading

- *Site Reliability Engineering* (the Google SRE Book), chapters on **Addressing Cascading Failures** and **Handling Overload** — the canonical treatment of retry amplification, load shedding, and why retries need budgets.
- The *AWS Architecture Blog*, **"Exponential Backoff And Jitter"** — the simulation-backed analysis that established full jitter as the default; the source for section 6.
- The **gRPC documentation** on client retries and `retryThrottling`, and the **gRPC deadlines** guide — the reference implementation for deadline propagation and token-ratio retry budgets.
- The **resilience4j** and **Envoy** docs on circuit breakers, retries, and timeouts — production-grade configuration for the patterns in sections 7 and 9.
- Within this series: the [SRE mindset intro](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later), and the planned siblings on circuit breakers, bulkheads, and load shedding (`circuit-breakers-bulkheads-and-load-shedding`) and designing for failure (`designing-for-failure`).
- Out of series: the microservices [resilience patterns post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads), the message-queue [idempotency and deduplication post](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe), and the debugging post on [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering).

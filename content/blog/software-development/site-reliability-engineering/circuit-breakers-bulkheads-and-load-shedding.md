---
title: "Circuit Breakers, Bulkheads, and Load Shedding: Surviving Overload Without Going Down With the Ship"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Three patterns that decide whether a sick dependency or a traffic spike takes you down with it — the circuit breaker that fails fast, the bulkhead that contains the blast radius, and the load shedding that protects you when you are the overloaded one — with runnable resilience4j, Envoy, and admission-control configs and the thread-pool math behind each."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "circuit-breaker",
    "bulkhead",
    "load-shedding",
    "admission-control",
    "cascading-failure",
    "backpressure",
    "goodput",
    "resilience",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-1.png"
---

It is 14:07 on a Tuesday, and the payments service is fine. The metrics are green, the error budget is barely touched, and the on-call engineer is eating lunch. At 14:08 the fraud-scoring dependency — a third-party API that the payments service calls on every checkout — starts answering slowly. Not failing. Slowly. Each call that used to return in 40 milliseconds now takes 25 seconds before it times out.

Here is what happens next, and it is the same story every time. The payments service has a thread pool of 200 worker threads. Each checkout request grabs a thread, calls the fraud API, and waits. Before, those threads came back in 40 ms and were immediately reused, so 200 threads served thousands of requests per second. Now each thread is parked for 25 seconds waiting on a dependency that is never going to answer. Within about a second and a half, all 200 threads are blocked. The 201st request — including the requests that do not even *touch* the fraud API, like checking an order status — gets no thread, and the whole payments service starts returning errors or hanging. The fraud API is sick, but the payments service is now *dead*, and so is everything downstream that depended on payments. One slow dependency just took down a service that was perfectly healthy, plus a chunk of the platform around it. That is a **cascading failure**, and it is one of the most common ways a distributed system kills itself.

This post is about the three patterns that stop that story cold, and how they compose into a single layered defense. The **circuit breaker** wraps the call to the sick dependency and *fails fast* instead of hanging, so threads never pile up. The **bulkhead** isolates resources so that even if one pool drowns, the rest of the ship stays afloat. And **load shedding** handles the opposite problem — when *you* are the overloaded one and the only sane move is to reject some traffic at the front door rather than serve everyone slowly and serve no one at all. Figure 1 shows the three layers stacked the way a resilient service actually runs them.

![A vertical stack diagram showing load shedding on top, bulkheads in the middle, and a circuit breaker at the bottom guarding a flaky dependency, with the service staying up and high goodput as the outcome](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-1.png)

By the end you will be able to: wrap a flaky dependency in a circuit breaker and tune its thresholds so it trips at the right moment instead of flapping; carve your thread and connection pools into bulkheads so one bad dependency or one noisy tenant cannot starve the rest; and put admission control at your front door that sheds the *right* traffic first so a 3x spike costs you 60% of requests instead of 100% of your service. We will do the thread-pool exhaustion math, write the resilience4j and Envoy configs, and measure the before-and-after the way an SRE has to — in goodput, not in wishful thinking. This is the *running-it* layer of reliability: the architecture decisions live in our sibling [system-design treatment of cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads); here we wire them up, tune them, and watch them work on the pager. It sits squarely in the series' loop — these patterns are how you *respond* to overload in code so you do not have to respond to it at 3am, and they are the operational complement to the intro's [reliability-is-a-feature SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset).

## 1. The shape of the problem: slow failures are worse than fast ones

Before any pattern, internalize one principle, because all three patterns are answers to it: **in a distributed system, a slow failure is far more dangerous than a fast one.**

A fast failure is honest. The dependency returns a 503 in two milliseconds, your code sees the error, and the thread that made the call is immediately free to do something useful. You lose that one request, but you lose nothing else. A *slow* failure lies to you. The dependency does not say "I'm down" — it just takes forever. And while your thread waits, it is holding a finite resource: a worker thread, a database connection, a socket, a slot in a semaphore. Resources you have a limited number of.

This is the saturation problem, and it has a clean piece of arithmetic behind it. By Little's Law, the number of requests *in flight* in a system at steady state is the arrival rate times the average time each request spends in the system:

$$L = \lambda \times W$$

where $L$ is concurrent in-flight requests, $\lambda$ is the arrival rate (requests per second), and $W$ is the average latency in seconds. Your service can hold at most as many in-flight requests as it has threads. So the maximum throughput a thread pool can sustain is:

$$\lambda_{\max} = \frac{\text{pool size}}{W}$$

Plug in the payments service. With 200 threads and a healthy fraud-API latency of 40 ms (0.04 s), the pool can sustain $200 / 0.04 = 5{,}000$ concurrent-bounded requests per second. Now the dependency slows to 25 s. The same pool can now sustain $200 / 25 = 8$ requests per second before every thread is occupied. The arrival rate did not change — still, say, 4,000 requests per second. But the *service rate* collapsed by three orders of magnitude. The pool fills in:

$$t_{\text{exhaust}} \approx \frac{\text{pool size}}{\lambda} = \frac{200}{4000} = 0.05 \text{ s}$$

Fifty milliseconds. That is how long it takes a healthy 200-thread service to fully exhaust its pool once a hot dependency starts hanging. From the outside, the service goes from green to dead in the time it takes to blink, and the dashboards barely have time to register it before the request rate to your *own* upstream callers starts hanging too, and the cascade walks up the call graph.

It is worth being concrete about *which* resource exhausts, because it is not always threads. The most common silent killer is the **database connection pool**. A service typically holds a small pool of database connections — 10, 20, maybe 50 — because each connection is expensive on the database side. When a query gets slow (a missing index, a lock, a degraded replica), each request holds its connection for the duration of the slow query. The connection pool exhausts exactly like the thread pool does, and now *every* request that needs the database blocks waiting to *check out a connection* — including fast queries that would have returned in a millisecond. A 20-connection pool with queries that suddenly take 5 seconds sustains only $20/5 = 4$ queries per second; at any higher arrival rate the pool is empty and the service stalls. The same arithmetic, a smaller and more precious resource. The patterns in this post apply identically: a breaker on the slow query path, a bulkhead that reserves connections per query class, shedding before the queue to the pool grows unbounded. Whenever you reason about exhaustion, ask *which* finite resource — threads, connections, sockets, file handles, semaphore permits, memory — fills first, because that is the one you must protect.

The whole reason circuit breakers, bulkheads, and load shedding exist is to break one of the links in this chain. The breaker stops the slow call from *consuming a thread for 25 seconds* — it converts the slow failure back into a fast one. The bulkhead makes sure that even if the fraud-API calls eat all of *their* threads, the order-status calls have *their own* threads and stay alive. Load shedding makes sure that if the arrival rate $\lambda$ genuinely exceeds your capacity, you drop the excess before it ever grabs a thread. Three different links, three different patterns. Let us take them one at a time.

## 2. The circuit breaker: turn a hanging call into a fast failure

A circuit breaker is a small piece of code that wraps every call to a dependency and watches how those calls go. It is named after the electrical breaker in your house: when too much current flows, the breaker *trips* and opens the circuit so the wiring does not catch fire. The software version trips when too many calls *fail*, so your threads do not pile up on a dead dependency.

It has exactly three states, and understanding the transitions between them is 90% of using one well. Figure 2 is the state machine.

![A state machine graph showing the circuit breaker moving from CLOSED through a failure-rate threshold to OPEN, then through a cooldown to HALF-OPEN, which either closes again on successful trials or re-opens on a failed trial](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-2.png)

**CLOSED** is the normal, healthy state. Calls pass straight through to the dependency, and the breaker quietly counts how many of the recent calls failed (or timed out, or were slow past a threshold). As long as the failure rate stays under your trip threshold, the breaker stays closed and stays out of the way. This is important: a healthy breaker has essentially zero overhead. It is just incrementing counters.

**OPEN** is the tripped state. Once the failure rate over a recent window crosses the threshold — say, more than 50% of the last 20 calls failed — the breaker *opens*. Now, here is the magic: while open, the breaker does not even *try* to call the dependency. It returns an error (or a fallback) immediately, in well under a millisecond, without touching a thread for any meaningful time. This protects *both* sides. It protects you, because no thread sits blocked waiting on a dependency you already know is sick. And it protects the *dependency*, because it is not getting hammered by thousands of retrying callers while it is trying to recover — you are not piling on. A struggling service that is suddenly relieved of 90% of its load has a fighting chance to come back; a struggling service that everyone keeps retrying against just stays down.

**HALF-OPEN** is the probing state. The breaker cannot stay open forever — the dependency might have recovered. So after a *cooldown* period (often 30 seconds), the breaker transitions to half-open and lets a small *trial* of calls through — maybe three. If those trial calls succeed, the breaker concludes the dependency is healthy again and transitions back to CLOSED. If any trial call fails, it snaps back to OPEN and starts the cooldown over. Half-open is how the breaker recovers automatically without a human, and the trial trickle is deliberately tiny so that a still-sick dependency is not slammed the instant it shows the faintest sign of life.

### Why fail-fast beats hanging — the principle made provable

Go back to the math from section 1. Without a breaker, each call to the sick dependency holds a thread for the full timeout, say 25 s (or the timeout you configured — and you *did* configure a timeout, per our sibling post on [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right), because an unbounded call is a guaranteed outage). With the breaker open, each call to the sick dependency holds a thread for *under a millisecond* — long enough to check "is the breaker open? yes → return error." The effective $W$ for the bad path drops from 25 s to ~0.001 s, a 25,000x reduction in how long each doomed call occupies a thread. The pool simply never fills, because the doomed calls clear instantly. That is the entire value proposition: **the breaker converts a slow failure into a fast one, and fast failures do not exhaust pools.**

### A real circuit-breaker config (resilience4j)

Here is a production-shaped resilience4j configuration for the fraud-scoring call. resilience4j is the standard JVM resilience library; the same concepts map directly to Polly (.NET), gobreaker (Go), or a service-mesh outlier-detection config, which we will show next.

```yaml
resilience4j:
  circuitbreaker:
    instances:
      fraudScoring:
        # Count-based window: look at the last 20 calls.
        slidingWindowType: COUNT_BASED
        slidingWindowSize: 20
        # Don't trip on a cold start — need at least 10 calls recorded.
        minimumNumberOfCalls: 10
        # Trip OPEN when > 50% of the window failed.
        failureRateThreshold: 50
        # Also count calls slower than 2s as "failures" — a slow call
        # holds a thread, so treat slowness as failure for tripping.
        slowCallDurationThreshold: 2s
        slowCallRateThreshold: 50
        # Stay OPEN for 30s before probing.
        waitDurationInOpenState: 30s
        # In HALF-OPEN, allow 3 trial calls.
        permittedNumberOfCallsInHalfOpenState: 3
        automaticTransitionFromOpenToHalfOpenEnabled: true
        # Which exceptions count as failures (a 4xx is the caller's fault,
        # not the dependency's — don't trip on those).
        recordExceptions:
          - java.io.IOException
          - java.util.concurrent.TimeoutException
        ignoreExceptions:
          - com.example.payments.BadRequestException
```

Two non-obvious lines do the heavy lifting. First, `slowCallDurationThreshold: 2s` with `slowCallRateThreshold: 50` — this trips the breaker on *slowness*, not just outright errors. Remember, slow is the dangerous failure. A dependency that answers every call successfully but takes 25 seconds will never trip an error-only breaker, yet it will still exhaust your pool. Counting slow calls as failures closes that gap. Second, `ignoreExceptions` — a 400 Bad Request means *you* sent garbage; that is not the dependency being sick, and tripping the breaker on client errors would take the dependency offline because of a bug in *your* request building. Trip on the dependency's faults, not your own.

In code, you wrap the call and supply a fallback:

```java
@CircuitBreaker(name = "fraudScoring", fallbackMethod = "fraudFallback")
public FraudVerdict score(Transaction txn) {
    return fraudClient.score(txn);   // the guarded call
}

// Invoked when the breaker is OPEN or the call fails.
private FraudVerdict fraudFallback(Transaction txn, Throwable t) {
    // Degrade gracefully: approve low-value txns, queue high-value
    // ones for async review. Better than failing checkout entirely.
    if (txn.amountCents() < 5000) {
        return FraudVerdict.approveWithFlag("fraud-degraded");
    }
    return FraudVerdict.queueForManualReview();
}
```

That fallback is where this pattern shakes hands with [graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — the breaker gives you a clean place to decide what "good enough" looks like when the dependency is gone. A breaker without a fallback just turns a slow error into a fast error; a breaker *with* a thought-out fallback turns a slow error into a degraded-but-working experience.

### The same thing in a service mesh (Envoy / Istio)

If your dependency calls go through a sidecar proxy, you get a breaker without touching application code. Envoy calls it *outlier detection* plus *circuit breaking*, and Istio exposes it as a `DestinationRule`:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: fraud-scoring
spec:
  host: fraud-scoring.payments.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100        # bulkhead on connections
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 100
    outlierDetection:
      consecutive5xxErrors: 5      # 5 in a row → eject this host
      interval: 10s                # check every 10s
      baseEjectionTime: 30s        # eject (≈ OPEN) for 30s
      maxEjectionPercent: 50       # never eject more than half the hosts
```

The mesh version operates per-*host* in the upstream cluster: when one fraud-scoring pod returns five 5xx in a row, Envoy *ejects* it from the load-balancing pool for 30 seconds (the mesh's version of OPEN), and gradually returns it (the mesh's HALF-OPEN). `maxEjectionPercent: 50` is the guardrail that stops the mesh from ejecting *every* backend during a correlated outage and leaving you with zero capacity — a subtle but vital setting we will revisit in the war stories. Note that `connectionPool` block is also doing bulkhead duty, which is our next section.

### Tuning: the breaker that trips too eagerly vs too late

A breaker has exactly two ways to be wrong, and they are symmetric.

A breaker that **trips too eagerly** *flaps*. Set `slidingWindowSize` to 5 and `failureRateThreshold` to 40%, and two unlucky timeouts out of five — which happen all the time on a healthy-but-noisy dependency — trip the breaker. Now you are failing fast against a dependency that was *fine*, your fallback path is taking real traffic for no reason, and 30 seconds later the breaker half-opens, three calls succeed, it closes, and two calls later it trips again. The dependency was never sick; your breaker was just twitchy. Flapping breakers are worse than no breaker because they manufacture an outage out of normal noise.

A breaker that **trips too late** does not protect you. Set `slidingWindowSize` to 500 and `minimumNumberOfCalls` to 200, and by the time enough failures have accumulated in that big window to cross the threshold, your pool has been exhausted for ten seconds and the cascade is already underway. The breaker eventually trips, but it trips *after* the damage.

The tuning rule of thumb: size the window so it reflects "recent" without being so small that normal variance trips it. A count window of 20–100 calls with a 50% threshold and a minimum-calls floor around 10–20 is a sane default for a high-traffic path. For a low-traffic path (a few calls per minute), use a *time-based* window instead so a single bad minute does not linger in a count window for an hour. Always pair the breaker with sensible per-call timeouts; the breaker decides *when to stop trying*, the timeout decides *how long each attempt may hang* — they are complementary, not substitutes.

#### Worked example: the cascade stopped by a breaker

Take the payments service concretely. Pool size 200 threads, normal arrival rate 4,000 requests/second, of which 60% (2,400/s) hit the fraud path. Fraud API healthy latency 40 ms; under failure it hangs to its 10 s timeout.

**Without a breaker.** When the fraud API hangs, fraud-path threads are held for 10 s each. By Little's Law the fraud path *demands* $2{,}400 \times 10 = 24{,}000$ concurrent threads but the pool only has 200. The pool exhausts in $200 / 4000 = 50$ ms (all arriving traffic, fraud and non-fraud, competes for the same threads). Every request — including the 1,600/s of *non-fraud* order-status traffic that has nothing to do with the sick dependency — now fails. **Result: ~100% of the payments service is down. Goodput ≈ 0.** A healthy dependency's failure took out the whole service.

**With a breaker (and a per-dependency bulkhead, next section).** The first ~20 fraud calls hang and time out, populating the breaker's window. Within roughly one window the failure rate crosses 50% and the breaker OPENs — call it 2–3 seconds. From that moment, every fraud call returns in <1 ms via the fallback (approve-low-value / queue-high-value). Fraud-path threads are never held. The 1,600/s of non-fraud traffic flows normally because those threads were never starved. **Result: order-status stays 100% up, checkout degrades gracefully (low-value approved, high-value queued) instead of erroring, and the service rides out the dependency outage. Goodput ≈ 60–100% depending on how much the fallback can serve.** When the fraud API recovers, the breaker's half-open probe notices within one cooldown (30 s) and closes. The dependency was down for 12 minutes; the *payments service* served traffic the entire time. Figure 3 contrasts the two timelines.

![A before and after comparison showing the no-breaker path cascading from a dead dependency through blocked threads to a downed service, versus the breaker path where calls fail fast and the service stays up serving non-dependency traffic](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-3.png)

## 3. Bulkheads: contain the blast radius to one compartment

The breaker handles a *sick* dependency. But there is a second failure mode it does not fully solve on its own: what if the dependency is not returning errors fast enough to trip the breaker quickly, and in the few seconds before the breaker trips, the slow calls drain the *shared* thread pool that other, healthy paths also need? Or what if you have ten different dependencies sharing one pool and only one is sick — the breaker for that one dependency will trip, but in the meantime its slow calls were stealing threads from the other nine? You need to *isolate the resources* in the first place.

That is the bulkhead. The name comes from shipbuilding. A ship's hull is divided into watertight compartments by walls called bulkheads, so that if the hull is breached, only *that* compartment floods — the water cannot spread along the whole length of the ship and sink it. (The Titanic's bulkheads famously did not extend high enough, so the flooding spilled over the tops from one compartment to the next. The lesson generalizes: a bulkhead that does not fully contain is not a bulkhead.)

In software, the bulkhead is **resource isolation**: instead of one big shared thread pool (or connection pool, or semaphore) that all dependencies draw from, you give *each* dependency — or each tenant, or each class of traffic — its *own* bounded pool. Now a slow dependency can exhaust *only its own pool*. It physically cannot starve the pool that another path needs, because those threads are in a different compartment. Figure 4 lays out which pattern protects against which failure, and where the bulkhead fits.

![A matrix comparing circuit breaker, bulkhead, and load shedding across the failure each one stops, its mechanism, and its failure mode when mis-tuned](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-4.png)

### A bulkhead config (resilience4j thread-pool isolation)

resilience4j ships a `ThreadPoolBulkhead` (and a lighter semaphore-based `Bulkhead`). Here is per-dependency thread-pool isolation:

```yaml
resilience4j:
  thread-pool-bulkhead:
    instances:
      fraudScoring:
        maxThreadPoolSize: 20      # fraud calls get AT MOST 20 threads
        coreThreadPoolSize: 10
        queueCapacity: 10          # small bounded queue — NOT unbounded
        keepAliveDuration: 1s
      inventoryLookup:
        maxThreadPoolSize: 30      # a separate compartment
        coreThreadPoolSize: 15
        queueCapacity: 10
      loyaltyPoints:
        maxThreadPoolSize: 5       # low-priority, small pool — can't starve others
        coreThreadPoolSize: 2
        queueCapacity: 5
```

The critical property: fraud, inventory, and loyalty draw from *separate* compartments. If the fraud API hangs, the worst case is that 20 fraud threads plus 10 queued fraud requests are stuck. The inventory path still has its 30 threads, and loyalty its 5. The blast radius of a sick fraud API is now exactly 30 threads, not all 200. Figure 6 shows this routing.

![A graph showing incoming requests routed by path into separate bounded pools per dependency, where a slow dependency A saturates only its own pool A while pool B and its healthy dependency keep serving path B fast](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-6.png)

Note `queueCapacity` is small and *bounded*. A bulkhead with an unbounded queue is not a bulkhead — the queue just becomes the new place where requests pile up and time out (we will hammer this point in the load-shedding section). The bounded queue is itself a form of admission control: once the pool and its small queue are full, new fraud requests are *rejected immediately* with a `BulkheadFullException`, which your fallback handles. That is the bulkhead doing fail-fast for you on a per-resource basis.

### Bulkheads for tenants: the noisy-neighbor problem

The other huge use of bulkheads is **per-tenant isolation** in multi-tenant systems, and it ties directly to *blast radius* — one of the core SRE concepts. If your platform serves a thousand customers from one shared pool of database connections, and one customer (one *tenant*) runs a runaway batch job that issues 10,000 slow queries, that one tenant can consume every connection and *every other tenant goes down*. One noisy neighbor took out the building.

The fix is a per-tenant quota — a bulkhead per tenant, or at least per *tier* of tenant:

```yaml
# Example: per-tenant connection quotas at the data-access layer.
tenant_quotas:
  default:
    max_db_connections: 5
    max_inflight_requests: 50
  premium:
    max_db_connections: 20
    max_inflight_requests: 200
  # A single tenant can NEVER exceed its quota, so it cannot
  # starve the shared pool. Blast radius of a noisy tenant = itself.
  enforcement: reject_over_quota   # 429 the over-quota tenant, not everyone
```

The principle: **a bulkhead converts "one bad actor takes down everyone" into "one bad actor takes down only themselves."** That is the most you can ask of resource isolation, and it is a lot. The cost, which we will be honest about, is utilization. If you statically partition 200 threads into ten pools of 20, and only one dependency is busy, you are wasting 180 idle threads while the busy pool throttles. There is a real tension between *isolation* (rigid partitions, safe, lower utilization) and *sharing* (one big pool, efficient, but no blast-radius containment). Most mature systems use a hybrid: a reserved minimum per compartment plus a shared *burst* pool that any compartment may borrow from up to a cap — isolation for the floor, sharing for the headroom.

#### Worked example: bulkhead sizing and the utilization trade-off

The payments service has a 200-thread budget and three dependencies on the hot path. Naively, you might give each pool $200/3 \approx 66$ threads. But size by *demand*, not by fairness. Fraud is called on 100% of checkouts at 40 ms; inventory on 100% at 20 ms; loyalty on 100% at 80 ms but is non-critical. At a target 2,000 checkouts/second:

- Fraud demand: $2000 \times 0.040 = 80$ concurrent → pool 100 (with headroom).
- Inventory demand: $2000 \times 0.020 = 40$ concurrent → pool 50.
- Loyalty demand: $2000 \times 0.080 = 160$ concurrent — but it is non-critical, so cap it at 30 and let it shed under load rather than letting a non-critical path claim 160 threads.

That sums to 180 reserved, leaving 20 for a shared burst pool. Now stress-test it: **what if fraud hangs?** Fraud's 100-thread pool fills and overflows its bounded queue; fraud calls fail fast via the bulkhead; inventory's 50 and loyalty's 30 are untouched; the service serves order-status and inventory-backed reads at full speed. Blast radius = 100 threads + the 10-deep queue, and the breaker trips on top of that to make even those fail fast. **What if loyalty is the slow one?** It can hold at most 30 threads — annoying, invisible to the customer (loyalty points show up a few seconds later via the fallback), and it cannot touch fraud or inventory. The static partition cost us some utilization — when only inventory is busy, the 100 idle fraud threads sit there — but it bought us a bounded, predictable blast radius. For a payments service, that trade is obviously worth it. For a best-effort internal analytics service, it might not be, and you would lean toward one shared pool.

### Two kinds of bulkhead: thread-pool isolation vs the semaphore

There are two ways to build a bulkhead, and the choice matters more than people realize. The **thread-pool bulkhead** (the resilience4j `ThreadPoolBulkhead` above) runs the guarded call on a *separate, dedicated thread pool*. This gives the strongest isolation — the calling thread hands the work to the pool and can walk away, so a slow dependency literally cannot occupy a caller thread — and it gives you a free *timeout* mechanism because the calling thread can stop waiting on the future. The cost is a thread hop per call (context switches, a little latency) and the extra memory of all those pooled threads. Use it for genuinely blocking, slow, or untrusted calls.

The **semaphore bulkhead** (the lighter resilience4j `Bulkhead`) does not use a separate pool at all. It just maintains a counter — a semaphore with N permits — and a call must acquire a permit to proceed, releasing it when done. If all N permits are taken, the call is rejected immediately with `BulkheadFullException`. No thread hop, near-zero overhead, but the call runs *on the caller's thread*, so a semaphore bulkhead caps *concurrency* without giving you the thread-isolation safety net. Use it for fast, non-blocking calls where you want a concurrency limit but cannot afford the thread-hop tax — and where a separate timeout/breaker already bounds how long the call can hang.

```yaml
resilience4j:
  bulkhead:                      # semaphore-based: cheap concurrency cap
    instances:
      inventoryLookup:
        maxConcurrentCalls: 30   # at most 30 in flight, else reject fast
        maxWaitDuration: 0       # do NOT block waiting for a permit — fail now
```

The single most important line is `maxWaitDuration: 0`. If you let callers *block* waiting for a permit, you have reintroduced the exact problem the bulkhead was supposed to solve — now threads pile up *waiting for the bulkhead* instead of waiting on the dependency. A bulkhead that makes you wait is not failing fast. Set the wait to zero (or a tiny bound) so an over-capacity call is rejected immediately and your fallback fires. The rule of thumb: thread-pool bulkhead for slow/blocking/untrusted dependencies where isolation is worth the thread hop; semaphore bulkhead for fast in-process or already-async calls where you only need a concurrency ceiling.

A subtle interaction worth flagging: the bulkhead and the breaker stack on the *same* call, and the order is bulkhead-outside-breaker or breaker-outside-bulkhead depending on what you want to count. resilience4j's recommended decorator order is `Bulkhead → TimeLimiter → RateLimiter → CircuitBreaker → Retry` from outermost to innermost, so the breaker sees the *actual* call outcomes (not bulkhead rejections) and the bulkhead caps concurrency before you even consult the breaker. Get the order wrong and your breaker starts counting bulkhead rejections as dependency failures, which trips it for the wrong reason — your own saturation, not the dependency's sickness.

## 4. Load shedding: when you are the overloaded one

The breaker and the bulkhead both assume the problem is *out there* — a sick dependency. The third pattern handles the case where the problem is *you*: demand has genuinely exceeded your capacity. A marketing email went out, a celebrity tweeted your link, a retry storm from a downstream service hit you, or it is simply Black Friday. The arrival rate $\lambda$ now exceeds your maximum service rate $\lambda_{\max}$, and no amount of breaking or bulkheading helps because nothing is *broken* — there is just more work than you can do.

Here is the hard principle, and it is the one engineers resist most: **when demand exceeds capacity, you MUST reject some load, or the whole system collapses.** Serving everyone slowly is not a kind compromise. It is the worst possible outcome, because it serves *no one*. This is the goodput collapse, and it is worth making precise.

### Goodput vs throughput: the distinction that matters

**Throughput** is how many requests you *accept*. **Goodput** is how many requests you *complete usefully* — that is, complete before the client gives up and within the latency that makes the response valuable. They are not the same number, and under overload they diverge violently.

Picture your service at 3x capacity with an unbounded queue and no shedding. You accept all 3x of the traffic — throughput looks great, 300% of capacity flowing in. But the queue grows without bound, so by Little's Law the wait time $W = L / \lambda$ climbs and climbs. Every client has a timeout — say 5 seconds. Once the queue is deep enough that requests sit for more than 5 seconds, *every* client times out before its response arrives. The request still completes on the server (it does the database work, computes the result) but the client already gave up and the work is thrown away. **Throughput: 300%. Goodput: 0%.** You did three times the work and delivered none of it. Worse, the timed-out clients often *retry*, adding even more load, and you have built a perfect doom loop. (This is exactly the retry-storm dynamic our [timeouts and retries post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) warns about, viewed from the server's side.)

Now shed. At 3x capacity, you reject 60%+ of arriving requests *at the front door*, cheaply, before they grab a thread or a database connection — a fast 429 or 503 that costs you almost nothing. The remaining ~40% (at or just under your capacity) flows through, gets served in normal latency, and the clients get real responses. **Throughput: 40%. Goodput: 40%.** You served less, but everything you served was *useful*. The honest math: shedding turns 0% useful work into 40% useful work. Figure 5 makes the contrast concrete.

![A before and after comparison showing accept-all at 3x load producing an unbounded queue, latency past the client timeout, and 100% timeouts with zero goodput, versus shedding 60% at the door yielding 40% admitted and served fast with high goodput](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-5.png)

The keystone insight is *shed early and cheaply*. The whole point is to reject the request before you have spent any meaningful work on it. A 429 returned by a lightweight admission filter at the edge costs microseconds. A request that gets all the way to your database, does the query, and *then* fails because the client timed out cost you the most expensive resource you have. **Reject at the front door, not at the back.**

### Why a queue with no shedding just moves the failure

Engineers love to reach for a queue when they are overloaded — "we'll just buffer the excess." A bounded queue with shedding is fine. An *unbounded* queue is a trap, because it does not add capacity; it adds *latency*. The work still has to be done at rate $\lambda_{\max}$; the queue just stores the backlog. And stored backlog means wait time, and wait time past the client timeout means wasted work. An unbounded queue converts an *overload* problem into a *latency* problem and then back into a *wasted-work* problem. It moves the failure from "we reject requests" (visible, honest, recoverable) to "everything is mysteriously slow and then everyone times out" (invisible until it is catastrophic). The fix is always a *bounded* queue with a shedding policy on overflow — which is just admission control with a small buffer. This is the same backpressure principle our [system-design post on rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) develops at the architecture layer.

### Prioritized shedding: shed the cheap-to-lose first

Not all requests are equal, and the difference between amateur and professional load shedding is *which* requests you drop. You do not shed randomly. You shed by **priority**: drop the low-value, cheap-to-lose, easy-to-retry-later traffic first, and protect the revenue-critical, user-facing, hard-to-recover traffic. Figure 8 shows the decision tree for an e-commerce service.

![A decision tree for prioritized shedding splitting overload traffic into shed-first low-value paths like batch jobs and recommendations versus protect high-value paths like checkout and first-try requests over retries](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-8.png)

For a typical e-commerce platform under overload, the shed order from first-to-drop to last-to-drop:

1. **Batch / async jobs (P3).** A nightly report or a recommendation precompute can wait an hour. Shed these first; they are retryable and no human is staring at them.
2. **Best-effort enrichments (P2).** Recommendations on the product page, "customers also bought," personalization. The page works without them; serve a generic version. This is the brownout (below).
3. **Read traffic (P1).** Browsing, search. Important, but a user who cannot browse for 30 seconds is annoyed, not lost.
4. **Checkout / payments (P0).** Never shed this if you can possibly avoid it. This is the revenue, and a user who cannot check out leaves and may not come back.

And a parallel axis that is easy to miss: **shed the retry before the first try.** A retry, by definition, is a request the client already failed once — it is duplicate work and often part of a storm. The *first* attempt is the one that has not been served at all. If you must drop one, drop the retry. You can detect retries via a header your clients set (e.g., `X-Retry-Attempt: 2`) and shed by attempt count.

### How do you even know your capacity? Discover it, don't guess it

The admission controller above needs one number it does not actually have: `capacity`. The naive approach is to load-test once, write down "2,000 rps," and hardcode it. That number is wrong the moment your code changes, your instance type changes, a noisy neighbor lands on the same host, or a downstream dependency slows down (which makes each of your requests hold its resources longer, lowering your effective capacity). A hardcoded capacity is a config that silently rots.

The better approach is to *discover* capacity from latency in real time, which is exactly what the adaptive-concurrency controllers do. The principle leans on Little's Law again. A service running below capacity has a stable, low minimum latency — call it $\text{minRTT}$, the time a request takes when nothing is queued. The instant load exceeds capacity, a queue forms, and the *first* symptom is that observed latency climbs above $\text{minRTT}$ — before throughput collapses, before timeouts, while you still have time to act. So the controller continuously measures $\text{minRTT}$ during quiet moments, then computes a concurrency limit from the gradient between current latency and that floor:

$$\text{limit} = \text{limit} \times \sqrt{\frac{\text{minRTT}}{\text{currentRTT}}}$$

When current latency equals the floor, the gradient is 1 and the limit holds steady. When current latency rises above the floor (queuing started), the gradient drops below 1 and the limit shrinks — the controller sheds more — *automatically*, the moment queuing begins, with no human and no hardcoded number. When latency falls back to the floor, the limit grows again to probe for more capacity. This is the same control loop TCP uses for congestion (it is literally borrowed from TCP Vegas), and it is the right default for a public service because it adapts to whatever your real capacity is *right now*, including the slow-dependency case where each request holds resources longer and your effective capacity quietly drops. Netflix's open-source `concurrency-limits` library and Envoy's adaptive-concurrency filter both implement variants of this. The lesson: **prefer a controller that discovers capacity from latency over one that you hand a fixed number** — the fixed number is a future incident waiting for the day your assumptions stop holding.

### Brownout: degrade instead of dropping

Shedding does not have to mean a hard 429. Often the better move under load is a **brownout** — serve a *cheaper* response instead of the full one, or no response at all. Serve a cached (possibly slightly stale) version of the page instead of recomputing it. Skip the expensive personalization and serve the generic page. Return the product without the live inventory count. The user gets *something* useful at a fraction of the cost, and your capacity stretches to cover more of them. Brownout is the gentle end of the shedding spectrum and the direct sibling of [graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — same philosophy, applied to your own overload rather than a dependency's.

### A load-shedding / admission-control snippet (priority-based 429)

Here is a concrete admission filter. It tracks current concurrency against a measured capacity and sheds by priority once you cross a high-water mark. This is a simplified but faithful version of what production admission controllers (and Envoy's adaptive-concurrency filter) do.

```python
import time
from dataclasses import dataclass

# Priorities: lower number = more important. 0 = checkout, 3 = batch.
@dataclass
class AdmissionController:
    capacity: int            # measured max concurrent requests we serve well
    inflight: int = 0
    # Shed thresholds per priority, as a fraction of capacity.
    # Under light load (low inflight) we admit everything.
    # As inflight climbs, we start shedding the low-priority traffic first.
    shed_at = {0: 1.00, 1: 0.90, 2: 0.75, 3: 0.50}

    def admit(self, priority: int, is_retry: bool) -> bool:
        # Retries face a stricter bar than first tries.
        threshold = self.shed_at[priority]
        if is_retry:
            threshold -= 0.15
        limit = int(self.capacity * threshold)
        if self.inflight >= limit:
            return False          # SHED: reject early, cheaply
        self.inflight += 1
        return True

    def release(self):
        self.inflight -= 1


controller = AdmissionController(capacity=2000)

def handle(request):
    priority = classify(request)          # 0..3 by route + tenant tier
    if not controller.admit(priority, request.is_retry):
        # Reject at the door, before any real work.
        return Response(status=429,
                        headers={"Retry-After": "2"},
                        body="overloaded, try again shortly")
    try:
        return do_work(request)           # the expensive part
    finally:
        controller.release()
```

The shape that matters: as `inflight` rises toward `capacity`, the controller progressively closes the door on lower priorities first. At 50% load, batch jobs (P3) start getting 429s. At 75%, recommendations (P2) get shed. Checkout (P0) is only ever refused at 100% — the last thing to go. Retries get a 15-point handicap so they are shed before equivalent first-tries. And every rejection is a cheap 429 with a `Retry-After`, returned *before* `do_work` ever runs. That is shedding early and cheaply, prioritized correctly.

For an HTTP edge, the same idea lives in your gateway. An nginx / Envoy rate-limit plus an adaptive concurrency limit (Envoy's `adaptive_concurrency` filter, or Netflix's `concurrency-limits` library, which uses a TCP-Vegas-style algorithm to *discover* capacity automatically rather than you hardcoding it) gives you load shedding without application changes for the coarse cases, and the application-layer controller above for the priority-aware cases.

```yaml
# Envoy adaptive concurrency: discover capacity from latency, shed the rest.
http_filters:
  - name: envoy.filters.http.adaptive_concurrency
    typed_config:
      "@type": type.googleapis.com/envoy.extensions.filters.http.adaptive_concurrency.v3.AdaptiveConcurrency
      gradient_controller_config:
        sample_aggregate_percentile: { value: 90 }   # watch p90 latency
        concurrency_limit_params:
          max_concurrency_limit: 2000
        min_rtt_calc_params:
          interval: 30s            # recalibrate the "healthy" RTT every 30s
          request_count: 50
      enabled: { default_value: true, runtime_key: "adaptive_concurrency.enabled" }
```

The adaptive controller measures the minimum round-trip time when the service is unloaded, then sheds whenever observed latency rises above that minimum by a gradient — i.e., it sheds the moment *queuing* begins, before the queue ever grows deep. That is the front-door, early-and-cheap shed, automated.

## 5. How the three compose into defense-in-depth

The reason to learn all three together is that they are not alternatives — they are *layers*, each catching a failure the others let through. Go back to Figure 1, and read it as a request's journey through your defenses:

1. **Load shedding at the front door** answers "is there more demand than I can serve?" If yes, reject the excess *now*, cheaply, by priority — before it consumes anything. This protects you when *you* are overloaded.
2. **Bulkheads inside** answer "is one path or tenant trying to consume more than its share?" Each compartment is bounded, so a hot path or a noisy tenant drowns only its own pool. This *contains the blast radius* of whatever made it past the door.
3. **Circuit breakers at the dependency edge** answer "is the dependency I'm about to call actually healthy?" If it is sick, fail fast and use the fallback instead of hanging a thread on it. This protects you from a *sick dependency* and protects the dependency from your pile-on.

Read the failure modes the other direction and you see why you need all three. A breaker alone does not save you from your *own* overload (nothing is broken, so it never trips) — you need shedding. Shedding alone does not save you from a *slow dependency* eating threads on the requests you *did* admit — you need the breaker. And neither, alone, stops one sick dependency or noisy tenant from stealing the threads another path needs *during the few seconds before the breaker trips* — you need the bulkhead. They interlock. The breaker stops you hammering a dead dependency; the bulkhead contains the damage to one pool; the load shedder protects you when you are the overloaded one. Defense-in-depth means every layer can fail and the next one still catches you.

The composition also has a natural ordering in the request path that you should respect: **shed before you bulkhead before you break.** Reject excess at the edge (cheapest), so you never even route it to a compartment. Route the survivors into bounded compartments, so a hot path is contained. Within a compartment, guard each dependency call with a breaker, so a sick dependency fails fast. Each layer does the cheapest possible work to protect the more expensive layers behind it. This composition is also exactly what the microservices version of these patterns assembles at the service-fleet level — see [resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) for the cross-service wiring.

### Rolling these out without causing the outage you fear

There is a meta-risk in all three patterns: a mis-tuned resilience mechanism can *manufacture* the outage it was meant to prevent. A breaker with a too-small window flaps and fails healthy traffic; a bulkhead sized for the average self-throttles at peak; a shedder with a wrong capacity number sheds traffic you could have served. Because the failure mode of the protection is itself an outage, you roll these out the same careful way you roll out any risky change — behind a flag, observed, and canaried, never flipped globally on a Friday.

The rollout playbook I use: ship every new breaker, bulkhead, and shedding threshold *disabled* first, but with the *metrics live*, so you can see in your dashboards what *would* have happened — how often the breaker *would* have tripped, how often a compartment *would* have saturated, how much *would* have been shed — without any of it actually affecting traffic. This "monitor mode" or "dry run" period is where you catch a breaker that would have flapped 40 times a day on normal noise, before it ever fails a real user's request. Once the dry-run metrics look sane (the breaker would have tripped only during real incidents, the shedder would have shed only at genuine overload), enable it on a canary — a small percentage of instances or traffic via a feature flag — and compare the canary's user-facing SLI against the baseline. Only then ramp to 100%.

```yaml
# Feature-flagged, canary-ramped resilience rollout.
resilience_rollout:
  fraudScoring_breaker:
    mode: monitor          # monitor -> canary -> enforce
    canary_percent: 5      # enforce on 5% of traffic when mode=canary
    metrics_only_when: monitor   # always emit would-have-tripped metrics
  admission_control:
    mode: enforce
    shed_dry_run: false    # set true to log "would shed" without shedding
```

The discipline here is identical to canarying any change, and it pays off precisely because the cost of a bad resilience config is so asymmetric: a breaker that should have tripped but did not costs you one incident, but a breaker that flaps in production costs you a slow, confusing degradation that is *hard* to attribute because the system looks like it is "working." Roll out the safety net carefully, or the safety net becomes the hazard.

## 6. Measuring it honestly: the SLIs that prove it works

A pattern you cannot measure is a pattern you cannot trust, and "we added a circuit breaker" is not a result. Here is how an SRE proves these patterns are working, with real Prometheus metrics and the before-after numbers that matter.

For the **circuit breaker**, instrument the state and the fast-fail rate. resilience4j exports `resilience4j_circuitbreaker_state` (a gauge per state) and call outcomes. The SLI you care about is "fraction of dependency calls that failed *fast* rather than hung":

```promql
# How often is the breaker open? (time spent protected)
avg_over_time(
  resilience4j_circuitbreaker_state{name="fraudScoring", state="open"}[5m]
)

# Fast-fail rate: calls short-circuited by an open breaker.
sum(rate(resilience4j_circuitbreaker_not_permitted_calls_total{name="fraudScoring"}[5m]))

# The thing we actually care about: caller p99 latency stays bounded
# even while the dependency is sick.
histogram_quantile(0.99,
  sum by (le) (rate(http_server_request_duration_seconds_bucket{route="/checkout"}[5m]))
)
```

The proof is in that last query: during the fraud-API outage, the checkout p99 should stay near normal (because failed calls return in <1 ms) instead of climbing to the timeout. **Before the breaker: checkout p99 went from 120 ms to 25,000 ms during the outage, and the service fell over. After: checkout p99 rose to ~180 ms — the cost of the fallback path — and the service stayed up.** That is a number you can put in a postmortem and defend.

For **bulkheads**, the SLI is pool saturation per compartment. resilience4j exports `resilience4j_bulkhead_available_concurrent_calls`. Alert when a compartment is pinned at zero available — that compartment is saturated, but the *point* is that the alert is scoped to one compartment, not the whole service:

```promql
# Per-bulkhead saturation. Fires for the SICK compartment only.
resilience4j_bulkhead_available_concurrent_calls{name="fraudScoring"} == 0
# Meanwhile this should be healthy — proving isolation worked:
resilience4j_bulkhead_available_concurrent_calls{name="inventoryLookup"}  # > 0
```

For **load shedding**, the headline SLI is **goodput**, and you must measure it as *useful completions*, not accepted requests. Count requests that completed successfully *and* within the latency budget, over total *offered* load:

```promql
# Goodput: successful responses served within the 1s budget.
sum(rate(http_server_requests_total{status=~"2..", le="1"}[5m]))
/
sum(rate(http_server_offered_total[5m]))    # offered = served + shed

# Shed rate by priority — proves we shed the RIGHT traffic.
sum by (priority) (rate(http_requests_shed_total[5m]))
```

#### Worked example: the before-after table you put in the postmortem

A traffic spike to 3x capacity hit the service on a Friday. Here is the honest measured comparison between the no-defense run (from the incident two weeks earlier) and the run after we added admission control:

| Metric | No shedding (prior incident) | With prioritized shedding | 
|---|---|---|
| Offered load | 6,000 rps (3x capacity) | 6,000 rps (3x capacity) |
| Accepted (throughput) | 6,000 rps | ~2,200 rps |
| Completed within 1s budget (goodput) | ~50 rps (≈1%) | ~2,000 rps (≈33% of offered, ≈91% of accepted) |
| Checkout (P0) success | ~2% | ~98% |
| Recommendations (P2) success | ~2% | shed to ~0% (intentional) |
| p99 latency | 30,000+ ms (timeouts) | 280 ms |
| Service availability during spike | hard down ~22 min | stayed up, degraded |
| User-facing checkout failures | ~98% | ~2% |

Read the goodput row carefully, because it is the whole argument. Accepting everything gave us ~1% goodput — we did 3x the work and completed almost none of it usefully. Shedding 63% of offered load gave us ~33% goodput overall and, crucially, *98% checkout success*, because we protected the revenue path and shed recommendations and batch. We served fewer requests and made far more money. That is load shedding working: **fewer accepted, vastly more completed, and the completions were the ones that mattered.**

### Alert on the right thing: page on user pain, not on a tripped breaker

A trap I have watched teams fall into: they wire an alert to "circuit breaker is OPEN" and page on it. Do not do this. A breaker opening is the system *working as designed* — it is the protection mechanism doing its job, not necessarily a user-facing problem. If the breaker opened on a non-critical dependency and the fallback is carrying the load with zero user impact, paging a human at 3am to look at a breaker that is correctly protecting the service is exactly the alert-that-cries-wolf this series warns against in [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf). Page on the *symptom* — user-facing errors or latency past the SLO — and let the breaker state be a *diagnostic* signal on the dashboard, not a pager trigger.

The right alerting structure is two tiers. A symptom-based, multi-window burn-rate page on the user-facing SLI (checkout success, p99 latency), and a *low-urgency* signal (a ticket or a dashboard annotation, not a page) when a breaker has been open for an extended period, so someone investigates the sick dependency during business hours. Here is the Prometheus alerting rule split:

```yaml
groups:
  - name: resilience
    rules:
      # PAGE: user-facing symptom. The breaker may or may not be open;
      # we page because USERS are seeing errors past our budget.
      - alert: CheckoutErrorBudgetBurningFast
        expr: |
          (
            sum(rate(http_server_requests_total{route="/checkout",status=~"5.."}[5m]))
            /
            sum(rate(http_server_requests_total{route="/checkout"}[5m]))
          ) > (14 * 0.001)        # 14x burn of a 99.9% SLO
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "Checkout error budget burning at >14x — user-facing pain"

      # TICKET (not a page): a breaker has been protecting us for a while.
      # Worth investigating the dependency, but not at 3am if users are fine.
      - alert: CircuitBreakerOpenSustained
        expr: |
          avg_over_time(
            resilience4j_circuitbreaker_state{state="open"}[15m]
          ) > 0.5
        for: 15m
        labels: { severity: ticket }
        annotations:
          summary: "Breaker {{ $labels.name }} open >50% of last 15m — investigate dep"
```

The first rule pages on the *burn rate* of the user-facing error budget (14x burn of a 99.9% SLO, which would exhaust a 30-day budget in ~2 days if sustained — the burn-rate math this series develops in its alerting posts). The second merely files a ticket. That split is the difference between an on-call rotation that sleeps and one that does not.

The other before-after worth tracking is the *budget* angle, because everything in this series ties back to the error budget. A service with a 99.9% availability SLO has an error budget of 43.2 minutes per month. The prior incident burned 22 minutes of that in a single afternoon — half the month's budget in one spike. With shedding, the same spike burned essentially *zero* user-facing budget for checkout (the P0 path stayed up); the budget we "spent" was on intentionally-shed recommendations, which (if you SLO them at all) you SLO far more loosely. The patterns are, in budget terms, a way to spend your scarce error budget on the traffic that does not matter and protect it on the traffic that does.

## 7. The full breaker state-machine, end to end on one incident

It helps to walk the breaker through one real incident on a timeline, because the state transitions are where people get the tuning wrong. Figure 7 lays out the sequence.

![A timeline tracing the circuit breaker across one incident from healthy CLOSED through tripping to OPEN, the cooldown with threads freed, a half-open probe every 30 seconds, and finally closing again after three trial calls pass](/imgs/blogs/circuit-breakers-bulkheads-and-load-shedding-7.png)

- **00:00 — CLOSED, healthy.** Fraud API at 40 ms, 0% failure. Breaker counting, invisible, near-zero overhead.
- **00:02 — failures accumulate.** Fraud API degrades; calls start timing out at 2 s (our slow-call threshold). Within ~20 calls, >50% of the window are slow-or-failed.
- **00:02 — TRIP → OPEN.** The breaker opens. From this instant, fraud calls return in <1 ms via the fallback. The 20 threads in the fraud bulkhead that were stuck on slow calls drain as their timeouts fire, and they are *not* re-occupied, because new calls short-circuit. The pool recovers.
- **00:02–00:14 — OPEN, threads free.** For the next ~12 minutes the breaker stays open. Checkout serves degraded (low-value approved, high-value queued). Order-status and inventory serve normally — different bulkheads, untouched. No cascade.
- **Every 30 s — HALF-OPEN probe.** The breaker periodically lets 3 trial calls through. While the fraud API is still down, those 3 fail and the breaker snaps back to OPEN, restarting the 30 s cooldown. This is the breaker *not* piling on — exactly 3 probe calls per 30 s, not a flood.
- **00:14 — trials pass → CLOSED.** The fraud API recovers. The next half-open probe's 3 calls succeed. The breaker closes. Full fraud scoring resumes. Total dependency outage: 12 minutes. Total payments-service downtime: *zero*.

Now stress-test it, the way the kit demands, because the happy path is not where systems fail:

**What if the dependency is down for 2 hours, not 12 minutes?** No change in behavior — the breaker stays open, probing every 30 s, and your fallback carries checkout the whole time. The only question is whether your fallback can sustain 2 hours (e.g., is your "queue high-value for manual review" backlog going to overflow?). Size the *fallback's* resources for a long outage, not just a blip.

**What if the half-open probe itself takes down the recovering dependency?** This is why the trial count is small (3, not 300) and why you keep per-call timeouts on the probes. If the dependency is fragile, even 3 probes every 30 s is gentle. If you are worried, set the half-open trial count to 1.

**What if two dependencies trip at once?** Each breaker is independent; both open, both fail fast, both bulkheads contain. The risk is your *fallback* — if both fallbacks lean on the same backup resource, you have a hidden coupling. Stress-test your fallbacks for *correlated* failure, not just single failure.

**What if the breaker is open but the fallback is also failing?** Then you are genuinely degraded and should be shedding — fail the request fast with a clear 503, do not hold it. A breaker open with a dead fallback should still fail *fast*; never let it become a slow failure again.

## 8. War stories: when these patterns saved (or sank) real systems

Reliability patterns are abstract until you have seen them decide an outage. Three stories — two famous, one composite-but-realistic — that map directly onto the three patterns.

**Netflix and Hystrix: the breaker as a way of life.** Netflix popularized the application-level circuit breaker with their Hystrix library (now retired in favor of resilience4j, but the ideas live on). Their core insight, drilled into every engineer: in a microservice architecture, *every* remote call is a potential cascade, so every remote call gets wrapped in a breaker with a bulkhead and a fallback by default. Netflix's reported result, repeated in their engineering talks, was that a single failing dependency would degrade *one feature* (say, personalized rows on the home screen fall back to generic rows) rather than taking down the app. The home screen still loaded; you just got non-personalized recommendations. That is the breaker-plus-fallback composition operating as the *default*, not the exception. The lesson is cultural as much as technical: treat fail-fast-with-a-fallback as the standard way to make a remote call, not a special hardening you add after an outage.

**The AWS load-shedding doctrine.** Amazon's published guidance (the Builders' Library article on using load shedding to avoid overload) makes the goodput argument explicit and is worth reading in full. Their hard-won rule: a service should measure its own capacity and *reject* requests it cannot serve within latency, returning a fast error, rather than accept them and time out. They specifically warn against the unbounded queue — the queue that "smooths" load actually destroys goodput under sustained overload because it converts overload into latency and latency into timeouts. Their pattern is to keep queues shallow, shed at the front, and prioritize. This is the load-shedding section of this post, validated at the scale of services handling millions of requests per second. (These are documented engineering principles; the exact internal numbers are not public, so treat any specific figures I give as illustrative of the dynamics, not as Amazon's measured SLAs.)

**The correlated-ejection trap (a realistic composite).** A team adds Envoy outlier detection to eject unhealthy backends — good. But during a *correlated* failure (a bad config push that makes *every* backend start returning 5xx at once), the outlier detector dutifully ejects backend after backend, because they are all unhealthy, until it has ejected *all* of them and the service has *zero* capacity. The "resilience" feature manufactured a total outage from what would otherwise have been a partial one. The fix is the `maxEjectionPercent` guardrail from our Envoy config (cap ejection at 50%) plus *panic mode* — Envoy's behavior where, if too high a fraction of hosts are unhealthy, it stops ejecting and load-balances across *all* of them on the theory that a degraded backend beats no backend. The lesson generalizes to every pattern here: **a resilience mechanism that can take the whole system down is a liability, not an asset.** Always cap the blast radius of the protection itself — the breaker that can trip on everything, the bulkhead partition that starves the system, the shedder that sheds 100%. Guardrail your guardrails.

A fourth, briefer one — the *retry-storm-meets-shedding* interaction. A downstream service times out, retries aggressively, and the retries triple the load on your already-struggling service. If you have no shedding, the retry storm tips you from "slow" to "down," and now *their* retries multiply against a dead service. If you have prioritized shedding that *sheds retries before first-tries*, the storm largely sheds itself at your door — you keep serving first-attempts and reject the duplicate retry traffic, breaking the doom loop. This is why the retry-vs-first-try axis in section 4 is not a nicety; it is the specific lever that defuses the most common overload amplifier. Pair it with the client-side retry budgets from our [timeouts and retries post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) and you have defense from both ends.

## 9. How to reach for these (and when not to)

Every pattern has a cost, and the principal-engineer move is knowing when *not* to add one. Here is the decisive guidance.

**Reach for a circuit breaker when** you make a remote call to a dependency you do not fully control — a third-party API, another team's service, a database under load — and especially when that dependency's failure could hang your threads. If the call is in a hot path, breaker it. **Do not bother with a breaker when** the "call" is a local, in-process, sub-microsecond operation (a breaker around a hashmap lookup is cargo-cult engineering) or when there is no meaningful fallback *and* the call is not on a shared resource — if the request is doomed either way and it is its own isolated thread, a plain timeout is enough. And do *not* add a breaker without also setting the per-call timeout; an un-timed call inside a breaker can still hang for the few calls before the breaker trips.

**Reach for bulkheads when** multiple dependencies or multiple tenants share a resource pool and you need to bound the blast radius — multi-tenant systems, services with one critical and several non-critical dependencies, any place where "one bad path takes down everything" is a real failure mode. **Do not over-partition.** If your service has one dependency and one tenant class, bulkheads buy you nothing but wasted utilization. And do not make pools so small that *normal* load self-throttles — a bulkhead that sheds healthy traffic because you sized it for the average instead of the peak is just a mis-tuned rate limiter. Size by measured peak demand, leave a shared burst pool, and revisit the sizes when traffic patterns change.

**Reach for load shedding when** your service can plausibly be offered more load than it can serve — public-facing services, anything subject to spikes, anything downstream of a retrying client. Every internet-facing service should have *some* admission control, even if it is just an adaptive concurrency limit at the gateway. **Do not set up elaborate prioritized shedding for an internal batch service** that runs on a schedule you control and has no latency-sensitive users — there, a bounded queue and "process when you can" is fine; shedding adds complexity for a problem you do not have. And do not shed *symmetrically* if you have a revenue-critical path — undifferentiated shedding that drops checkout at the same rate as recommendations is leaving money on the table. The whole value of shedding is in the *prioritization*; flat shedding captures maybe half the benefit.

The meta-rule, true of every reliability pattern in this series: **match the mechanism to the actual failure mode, and cap the blast radius of the mechanism itself.** Do not auto-trip, auto-isolate, or auto-shed in a way you do not understand — a self-healing mechanism you do not understand is a way to turn a small incident into a confusing big one, a trap we cover in [self-healing systems and their traps](/blog/software-development/site-reliability-engineering/self-healing-systems-and-their-traps). And remember the layering from [designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure): these three patterns are *runtime* defenses; they buy you time and contain damage, but they do not substitute for an architecture that does not have a single dependency capable of taking everything down. Use them to survive the failures you could not design away.

## 10. Key takeaways

- **A slow failure is worse than a fast one.** A hung call holds a thread; 1,000 hung calls exhaust the pool and your healthy service goes down with its sick dependency. Every pattern here exists to turn slow failures into fast ones or to keep them contained.
- **The circuit breaker converts a hang into a fast fail.** CLOSED counts failures, OPEN fails fast without even trying (protecting you *and* the dependency), HALF-OPEN probes for recovery with a tiny trial. It is the single highest-leverage anti-cascade pattern.
- **Trip on slowness, not just errors,** and ignore the dependency's *client* errors (4xx) — a breaker that does not count slow calls will not save you from the dependency that hangs without erroring.
- **Tune the window so it is recent but not twitchy.** Too small a window flaps (a manufactured outage); too large trips after the damage. A count window of 20–100 calls at a 50% threshold is a sane high-traffic default; use a time window for low-traffic paths.
- **Bulkheads bound the blast radius.** Give each dependency or tenant its own bounded pool so one sick path or one noisy neighbor drowns only its own compartment. Accept the utilization cost; for critical services the predictable blast radius is worth it.
- **Bounded queues only.** An unbounded queue is not capacity; it is latency, and latency past the client timeout is wasted work. A bounded queue with a shedding policy on overflow is the only safe queue under overload.
- **Under overload you MUST shed — serving everyone slowly serves no one.** Goodput (useful completions), not throughput (accepted requests), is the number that matters. At 3x load, shedding 60% at the door turns ~0% goodput into ~40% goodput.
- **Shed early, cheaply, and by priority.** Reject at the front door with a fast 429/503 before you spend any work. Drop batch and recommendations before checkout; drop retries before first-tries. Brownout (cheaper/cached responses) is the gentle end of the same spectrum.
- **The three compose into defense-in-depth, in order: shed → bulkhead → break.** Shedding protects you when you are overloaded; bulkheads contain the blast radius; breakers stop you hammering a dead dependency. Each layer catches what the others let through.
- **Cap the blast radius of the protection itself.** A breaker that can trip on everything, a partition that starves the system, an ejector that ejects all backends — these manufacture outages. Guardrail your guardrails (`maxEjectionPercent`, panic mode, a never-shed P0 floor).

## 11. Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the intro map for this whole series and the define→measure→budget→respond→learn→engineer loop these patterns plug into.
- [Timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) — the per-call companion: a breaker decides *when to stop trying*, a timeout decides *how long each try may hang*, and a retry budget keeps you from amplifying the outage.
- [Graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — what "good enough" looks like when the dependency is gone; the fallback path a breaker hands off to and the brownout a load shedder serves.
- [Designing for failure](/blog/software-development/site-reliability-engineering/designing-for-failure) and [self-healing systems and their traps](/blog/software-development/site-reliability-engineering/self-healing-systems-and-their-traps) — the architecture context and the cautions on automated remediation.
- [Cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) and [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) — the architecture-time treatment of these same ideas in the system-design series.
- [Resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — the service-fleet wiring of these patterns across many services.
- The Google SRE Book, "Addressing Cascading Failures" and "Handling Overload" — the canonical treatment of overload, load shedding, and graceful degradation.
- The Amazon Builders' Library, "Using load shedding to avoid overload" — the goodput argument and the shallow-queue, shed-at-the-front doctrine at hyperscale.
- The resilience4j and Envoy / Istio documentation — for the exact circuit-breaker, bulkhead, and adaptive-concurrency configuration knobs used above.

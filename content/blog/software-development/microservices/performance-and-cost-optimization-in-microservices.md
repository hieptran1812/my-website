---
title: "Performance and Cost Optimization in Microservices: Clawing Back Latency and Dollars"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Microservices are slower and pricier than a monolith by default — here is the senior's playbook for finding the real bottleneck with traces, killing chatty call patterns, right-sizing pods, cutting cross-AZ egress, and measuring the win in p99 and dollars."
tags:
  [
    "microservices",
    "performance",
    "cost-optimization",
    "latency",
    "kubernetes",
    "observability",
    "distributed-systems",
    "software-architecture",
    "backend",
    "scalability",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/performance-and-cost-optimization-in-microservices-1.webp"
---

The ShopFast checkout was slow and nobody could say why. At 10,000 requests per second on Black Friday eve, the p99 latency for "place order" had crept to 600 milliseconds, the support queue was filling with "the spinner just hangs" tickets, and the finance team had separately flagged that the cloud bill for the platform had grown 3x year over year while traffic had only doubled. Two different teams were panicking about two different numbers, and in the all-hands the engineering director asked the question that every senior in the room had been dreading: "We moved to microservices to go *faster*. Why are we slower and more expensive than the monolith we replaced?"

That question deserves an honest answer, because the uncomfortable truth is this: **a microservices architecture is slower and more expensive than an equivalent monolith by default.** Not because anyone did anything wrong, but because of physics and economics. Every call that used to be an in-process function invocation — nanoseconds, no serialization, no failure — is now a network round trip across a wire, through a load balancer, maybe through a sidecar proxy, with the request serialized to JSON or protobuf on the way out and deserialized on the way in. Every service you split out gets its own copy of the overhead: its own observability agent, its own sidecar, its own minimum number of idle replicas burning CPU for nothing. And the moment two chatty services land in different availability zones, every byte between them costs money in cross-zone network egress. The monolith had none of these taxes. You chose to pay them in exchange for independent deployability, team autonomy, and the ability to scale parts of the system independently — and those are real benefits. But "we adopted microservices" is not a license to be slow and wasteful. It is an obligation to *manage* the tax you signed up for.

This post is the senior engineer's playbook for clawing back latency and dollars without giving up the architecture. The opening figure traces a single ShopFast checkout and shows exactly where the 600ms goes: a serial chain of five service calls, an N+1 loop hammering the inventory service fifty times for one order, and — invisible in the latency trace but glaring in the bill — a fleet of pods that requested 2 vCPU each and are sitting at 8 percent utilization. We will fix all three, and by the end you will be able to do something specific: take a real trace and a real cloud bill, find the one change that moves the needle most, ship it, and prove the win in p99 latency and dollars per month rather than in vibes. The discipline that separates a junior from a senior here is not knowing more tricks — it is **optimizing the critical path with data instead of guessing, and treating dollars as a first-class metric alongside latency.**

![A service call graph for the ShopFast checkout showing a serial five-service chain and an N plus one loop into the inventory service carrying most of the p99 latency](/imgs/blogs/performance-and-cost-optimization-in-microservices-1.webp)

## Why microservices cost more in the first place

Before you optimize anything, you have to understand the taxes you are paying, because each tax has a different cure. If you do not know *why* the system is slow, you will optimize the part that feels slow rather than the part that is slow, and those are rarely the same thing.

The first and largest tax is the **network-hop tax on every call.** An in-process method call in a monolith is, for practical purposes, free: a few nanoseconds, no copying, no failure mode worth modeling. The same logical call between two services is a network round trip. Even within a single data center, a round trip is on the order of 0.5 to 1 millisecond of pure network time before anyone does any work — and that is the optimistic number, the one you get when the connection is already warm and pooled. Add the cost of establishing a TCP connection and a TLS handshake if the connection is cold (which can be 2 to 5 milliseconds), add the load balancer in the middle, add a service-mesh sidecar proxy if you run one (each sidecar adds latency on both the inbound and outbound side), and a single "simple" service-to-service call can easily cost 2 to 4 milliseconds of pure overhead before the receiving service executes a line of business logic. This is exactly the fallacy that the [inter-service communication fundamentals](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) post warns about: the network is not free, not reliable, and not instantaneous, and treating a remote call like a local one is the original sin of distributed performance.

The second tax is **serialization.** When you call a function in-process, you pass a pointer. When you call a service, you serialize the request into bytes, ship them, and the receiver deserializes them — and then the whole thing happens again for the response. JSON is convenient and human-readable and *slow*: encoding and decoding a moderately sized object can cost more CPU than the actual work the service does with it. A payload that is 5 kilobytes of JSON might be 1.5 kilobytes as protobuf, and the protobuf encode/decode can be several times faster. This is one of the reasons the [REST vs gRPC vs GraphQL](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis) decision has performance consequences, not just ergonomic ones.

The third tax is **per-service fixed overhead.** Every service you split out is not just business logic — it is a full runtime with a baseline footprint. It needs a minimum number of replicas for availability (you do not run a single replica of anything that matters, so the floor is usually 2 or 3). Each replica runs an observability agent or sidecar that scrapes metrics, ships logs, and exports traces. If you run a service mesh, each pod carries an Envoy sidecar consuming its own CPU and memory. None of this overhead does any customer-facing work; it is the cost of the service *existing*. Split a monolith into 40 services and you have just multiplied this fixed overhead 40 times, and the idle replicas burning CPU at 3am — when traffic is near zero but you still keep the floor of replicas up — are pure waste.

The fourth tax is **cross-AZ and cross-region network egress, which costs actual dollars.** Inside a single availability zone, traffic between your services is typically free. The moment two services talk across availability zones — which Kubernetes will happily arrange by default, because the scheduler places pods wherever there is room without caring about topology — every gigabyte that crosses a zone boundary costs money (on the major clouds, on the order of \$0.01 to \$0.02 per gigabyte each way for cross-zone traffic, and far more for cross-region). At 10,000 requests per second, each carrying a few kilobytes of payload across a zone boundary, this adds up to a five-figure annual line item that nobody decided to spend and nobody is watching.

Add these four taxes together and the picture is clear: a microservices system spends an enormous fraction of its latency and its budget on overhead that has nothing to do with the work the customer asked for. The good news is that this is *also* where the easy wins are. You are not trying to make the business logic faster — it is already fast. You are trying to stop paying overhead you do not need to pay.

#### Worked example: the cost of one extra hop

Suppose the ShopFast checkout flow makes 6 internal service calls, and each call costs 3 milliseconds of pure overhead (network + serialization + sidecar) on top of the work. That is 18 milliseconds of *pure tax* per checkout — before any service computes anything. At 10,000 requests per second, with each call carrying 4 kilobytes in and 4 kilobytes out, the platform moves roughly `10,000 rps × 6 calls × 8 KB = 480 MB per second` of internal traffic, or about 41 terabytes per day. If even a third of that crosses availability-zone boundaries because the scheduler spread the pods around, you are egressing roughly 13.7 TB per day across zones. At \$0.01 per gigabyte each way, that is about \$137 per day, or roughly \$50,000 per year, in cross-AZ egress alone — for a flow whose business logic could run on a single laptop. That number is not the database. It is not the compute. It is the *wire between your own services*, and it is the kind of cost that hides in plain sight until someone finally reads the bill line by line.

## Find the bottleneck before you touch anything

The single most expensive mistake in performance work is optimizing the wrong thing. A junior engineer hears "checkout is slow," opens the order service, sees a database query that *looks* heavy, spends a week adding an index, ships it, and the p99 does not move — because the database query was 4 milliseconds out of a 600-millisecond budget, and the actual problem was somewhere else entirely. The senior move is to refuse to optimize anything until a trace tells you where the time actually goes. **Measure first. Always.**

The tool for this is distributed tracing, covered in depth in [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry). A trace follows a single request across every service it touches and records how long each span took. When you pull up a slow checkout trace and look at the waterfall — the visualization where each span is a horizontal bar positioned by start time and sized by duration — the bottleneck usually announces itself. For ShopFast, the waterfall showed three things at once: the five downstream calls were happening *serially* (each bar starting only after the previous one finished), the inventory service appeared not once but fifty times in a tight cluster (the N+1), and the actual business logic spans were tiny slivers next to the network waits.

The crucial methodological point is *which* trace you look at. The instinct is to grab a random trace, but a random trace is almost always a fast one (most requests are fast, by definition), and a fast trace tells you nothing about why some requests are slow. The senior move is to filter the trace backend for the *slow* requests specifically — "show me traces of checkout where duration exceeded 500ms" — and look at what those particular requests did differently. Often the slow traces share a feature the fast ones lack: they belong to orders with many line items (the N+1 amplifies), they hit a cold cache, they were served by one particular replica, or they happened during a deploy. That shared feature *is* the diagnosis. Aggregate latency dashboards tell you *that* you are slow; the slow-trace sample tells you *why*. A team that only looks at the p99 number on a dashboard, without ever pulling the underlying slow traces, is flying blind — they know the patient has a fever but never take the temperature anywhere specific.

![A latency budget breakdown showing the inventory N plus one and serial waits dominating the checkout p99 while business logic is a tiny sliver](/imgs/blogs/performance-and-cost-optimization-in-microservices-4.webp)

The latency budget breakdown above is the most clarifying artifact you can produce. It takes the p99 number and decomposes it into where the milliseconds actually go. For ShopFast, of the 600ms p99, roughly 220ms was the inventory N+1 (fifty sequential calls at a few milliseconds each), 250ms was the serial fan-out waiting for each downstream in turn, 70ms was raw network and TLS overhead across the hops, 40ms was JSON serialization and deserialization, and only about 20ms was actual business logic. **Read that again: less than 4 percent of the latency was the work.** The other 96 percent was the microservices tax — and that tax is *recoverable*.

### p50 versus p99: optimize the experience real users actually get

Here is a distinction that trips up almost everyone the first time. When you measure latency, the *average* is a lie and the *median* (p50) is only half the story. What matters to the user experience — and what wakes you up at night — is the tail: the p99 (the latency that 1 percent of requests exceed) and the p999. The reason the tail matters so much in microservices specifically is *fan-out amplification*. If a single checkout makes 6 downstream calls, and each downstream has a p99 of 50ms, then the probability that *at least one* of those 6 calls hits its slow tail is much higher than 1 percent — it is roughly `1 − (0.99)^6 ≈ 5.9 percent`. The more services a request touches, the more the request's own p99 is dominated by the *worst* of its dependencies' tails. This is why a system made of services that each look fast in isolation can feel maddeningly slow end to end: the user always waits for the slowest of the many things that had to happen.

The practical consequence is that you should look at p50 and p99 *separately* and treat a large gap between them as a signal. If p50 is 80ms and p99 is 600ms, the system is not uniformly slow — it is *usually* fine and *occasionally* terrible, which points at tail-latency causes: garbage-collection pauses, cold connection pools, a saturated thread pool that queues some requests, a single slow replica, or an N+1 that only fires for orders with many line items. If instead p50 and p99 are both 500ms, the system is uniformly slow, which points at the critical path: a serial chain or a genuinely slow dependency that *every* request hits. The fix is completely different in the two cases, which is exactly why you must look before you leap.

### Traces find the slow service; flame graphs find the slow code

A distributed trace answers "*which service* is slow." It cannot, by itself, answer "*which code inside that service* is slow." For that you need a profile, and the most useful way to read a profile is a flame graph. A flame graph is a visualization of sampled stack traces: the x-axis is the proportion of samples (wider means more time spent), and the y-axis is the call stack depth (each box is a function, sitting on top of its caller). The widest boxes near the top are where the CPU actually went. When the trace says "the pricing service span is 60ms and that is on the critical path," you attach a profiler to the pricing service, take a CPU profile under load, and the flame graph will show you whether those 60ms went into the actual price calculation, into JSON serialization (a shockingly common culprit — it is not unusual to find 30 to 50 percent of a service's CPU in encoding and decoding), into a regex compiled on every request instead of once, or into a logging call that synchronously flushes to disk. The combination is the senior's one-two punch: the trace narrows you to the right *service*, the flame graph narrows you to the right *function*, and only then do you write a single line of optimization. Most runtimes ship a profiler for exactly this — Go's `pprof`, the JVM's async-profiler, Python's `py-spy`, Node's built-in profiler — and the discipline is to profile *under production-like load*, because a profile of an idle service tells you nothing about where it spends time when it is busy.

A related artifact for cost work is the *continuous profiler* (tools like Parca, Pyroscope, or the cloud providers' equivalents) that samples CPU across the whole fleet all the time and lets you ask "which function, across all services, is burning the most CPU-dollars?" That question routinely surfaces surprising answers — a logging library, a deserialization path, a poorly-chosen hash function — that no single trace would have revealed, because the cost was spread thin across millions of requests rather than concentrated in any one slow span.

### Where to optimize: Amdahl and the Universal Scalability Law

Two pieces of theory tell you *where* effort pays off, and both are worth a senior's time.

Amdahl's Law says that the speedup you can get from optimizing one part of a system is capped by the fraction of total time that part consumes. If the inventory N+1 is 220ms of a 600ms budget — about 37 percent — then even making it instantaneous can cut at most 220ms, taking you to 380ms. You cannot get the p99 below the *unoptimized remainder* no matter how brilliant your fix. This is why the latency budget breakdown is so valuable: it tells you the ceiling on each fix before you spend a single hour. Spending a week to shave a span that is 2 percent of the budget is, by Amdahl, a guaranteed disappointment.

The Universal Scalability Law (Neil Gunther's USL) is the deeper, more uncomfortable lesson, and it explains the *super-linear cost* part of ShopFast's problem. The USL says that as you add load (or capacity), throughput does not scale linearly forever. Two effects bend the curve down. The first is **contention** — the serial fraction, the parts that cannot run in parallel because they share a lock, a connection pool, a single-leader database. The second, and the one that genuinely *hurts*, is **coherency** — the cost of keeping things consistent, which grows with the *square* of the number of participants. In a microservices fleet, coherency cost shows up as cross-service coordination: every retry storm, every cache invalidation broadcast, every chatty synchronization between services adds coordination that grows worse than linearly. The USL is why a system can run beautifully at 5,000 rps and fall over at 10,000 rps even though you "only" doubled the load — you did not double the cost, you more than doubled it, because the coordination overhead grew faster than the traffic. When someone asks "why did the bill double at 2x traffic?" the USL is usually the honest answer: the coordination tax is super-linear, and the only durable fixes are reducing contention (more parallelism, fewer shared bottlenecks) and reducing coherency cost (less cross-service chatter, caching to avoid round trips). Both are exactly the levers below.

## The optimization spine: latency

This is the heart of the post. We will walk the levers in roughly the order a senior reaches for them — highest payoff and lowest risk first — and we will measure each one. The decision matrix below is the map; the sections that follow are the territory.

![A decision matrix comparing optimization levers batching parallel fan-out caching async right-sizing and fewer services across what each saves effort and risk](/imgs/blogs/performance-and-cost-optimization-in-microservices-3.webp)

The matrix encodes the senior's instinct: **batching an N+1 and right-sizing idle pods are the highest-payoff, lowest-risk moves**, so do them first. Caching saves both latency and money but introduces staleness you have to reason about. Async offload and consolidating services into fewer, coarser units save real latency and money but cost you in correctness and coupling complexity — they are powerful and they are where you can hurt yourself, so they come last and only with care.

### Lever 1: kill the chatty call patterns

The most common latency killer in microservices is not any single slow service — it is *chattiness*, the habit of making many small calls where you could make one. The worst offender is the **cross-service N+1**, the distributed cousin of the classic ORM N+1. ShopFast's checkout fetched the order, then looped over its line items and called the inventory service once per item to check stock. Fifty line items, fifty network round trips, each a few milliseconds of tax. In a monolith this pattern is a mild performance smell; across the network it is a catastrophe, because every iteration of the loop pays the full network-hop tax.

The fix is a **batch API**: replace N calls of one item each with one call of N items. Here is the naive code and the batched fix in Python.

```python
# NAIVE: one network round trip per line item (the cross-service N+1)
async def check_stock_naive(order):
    results = {}
    for item in order.line_items:           # 50 iterations
        resp = await inventory_client.get(   # 50 network round trips
            f"/stock/{item.sku}"
        )
        results[item.sku] = resp.json()["available"]
    return results

# BATCHED: one network round trip for the whole order
async def check_stock_batched(order):
    skus = [item.sku for item in order.line_items]
    resp = await inventory_client.post(      # 1 network round trip
        "/stock/batch",
        json={"skus": skus},
    )
    return resp.json()["availability"]       # {sku: available, ...}
```

The batch endpoint on the inventory side is a single query (`SELECT sku, available FROM stock WHERE sku = ANY($1)`) instead of fifty, so you save round trips *and* database queries. The only design discipline a batch API demands is a sane cap on batch size — you do not want one request asking for 100,000 SKUs and timing out — so you page large batches into chunks of, say, 500.

#### Worked example: the N+1 fix on a 50-item order

The naive path: 50 sequential calls × (3ms network tax + ~1.5ms inventory query) ≈ 50 × 4.5ms = **225ms**, every one of which is on the critical path. The batched path: 1 call × (3ms network tax + ~5ms for the slightly larger batched query) = **8ms**. That single change removes about 217ms from the checkout's critical path. Because the N+1 fired on every order, it moved both p50 and p99 — but it moved p99 *more*, because the worst-case orders (the ones with the most line items) were the ones hammering the inventory service hardest, and those are precisely the requests that lived in the tail. In ShopFast's case the N+1 fix alone took the p99 from 600ms to roughly 380ms.

The second chattiness fix is to **stop calling serially when the calls are independent.** ShopFast's order service called payment, then inventory, then pricing, then shipping — each waiting for the previous to return — even though none of them depended on each other's results. That is 250ms of waiting where the real cost is the *slowest single call*, not the sum. Firing them in parallel and waiting on all of them collapses the chain.

![A before and after comparison showing serial fan-out at 250ms versus parallel fan-out at 60ms for independent downstream calls](/imgs/blogs/performance-and-cost-optimization-in-microservices-2.webp)

Here is the refactor in Go, which makes the structure especially clear.

```go
// SERIAL: ~250ms, each call waits for the previous
func checkoutSerial(ctx context.Context, o Order) (Result, error) {
    pay, err := paymentClient.Authorize(ctx, o)   // ~70ms
    if err != nil { return Result{}, err }
    inv, err := inventoryClient.Reserve(ctx, o)   // ~60ms
    if err != nil { return Result{}, err }
    pr, err := pricingClient.Quote(ctx, o)        // ~60ms
    if err != nil { return Result{}, err }
    sh, err := shippingClient.Estimate(ctx, o)    // ~60ms
    if err != nil { return Result{}, err }
    return assemble(pay, inv, pr, sh), nil
}

// PARALLEL: ~60ms, all fired at once, wait on the slowest
func checkoutParallel(ctx context.Context, o Order) (Result, error) {
    g, ctx := errgroup.WithContext(ctx)
    var pay PayResp; var inv InvResp; var pr PriceResp; var sh ShipResp
    g.Go(func() (err error) { pay, err = paymentClient.Authorize(ctx, o); return })
    g.Go(func() (err error) { inv, err = inventoryClient.Reserve(ctx, o); return })
    g.Go(func() (err error) { pr, err = pricingClient.Quote(ctx, o); return })
    g.Go(func() (err error) { sh, err = shippingClient.Estimate(ctx, o); return })
    if err := g.Wait(); err != nil {           // first error cancels the rest
        return Result{}, err
    }
    return assemble(pay, inv, pr, sh), nil
}
```

The `errgroup` pattern is idiomatic Go: it runs the calls concurrently, cancels the shared context the moment any one fails, and returns the first error. The latency drops from the *sum* of the calls to the *max* of the calls. This is exactly the kind of fan-out that the [API gateway and backend-for-frontend](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend) layer is built to do well — a BFF's whole job is to fan out to many services in parallel and assemble one response for the client, so the client makes one call instead of six.

#### Worked example: serial to parallel

Serial: payment 70ms + inventory 60ms + pricing 60ms + shipping 60ms = **250ms** on the critical path. Parallel: `max(70, 60, 60, 60)` = **70ms** — call it 60ms once you account for the fact that the truly independent reads (pricing, shipping) are faster and the auth dominates. The catch, and the senior caveat, is in the p99: when you fan out to 4 calls in parallel, your p99 is now governed by the *slowest of the 4 tails*, not by one. As computed earlier, fanning out to more dependencies makes you more exposed to tail latency, not less. Parallelism fixes the *median and the serial sum*; it does **not** fix a fat tail. For the tail you need the tricks in Lever 5.

### Lever 2: caching the hot reads

The fastest network call is the one you never make. Caching — covered in depth in the forward-linked [caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services) post — is the highest-leverage latency *and* cost reducer for read-heavy paths, because a cache hit replaces a multi-millisecond cross-service round trip (and its egress cost) with a sub-millisecond local or near-local lookup. ShopFast's pricing service computed prices that changed at most a few times a day but were requested ten thousand times a second; caching the computed price for 60 seconds turned 10,000 rps of pricing-service calls into a handful per minute.

The workhorse pattern is **cache-aside** (also called lazy loading): on a read, check the cache; on a miss, fetch from the source, populate the cache, and return.

```python
async def get_price(sku: str) -> Decimal:
    cached = await redis.get(f"price:{sku}")
    if cached is not None:
        return Decimal(cached.decode())          # cache hit, sub-ms
    price = await pricing_client.quote(sku)      # cache miss, full round trip
    await redis.set(f"price:{sku}", str(price), ex=60)  # TTL 60s
    return price
```

The trade-off caching always demands is **staleness**: a cached price is by definition possibly out of date for up to its TTL. That is a business decision, not an engineering one — a price stale by 60 seconds is usually fine; a stock count stale by 60 seconds might oversell. The senior discipline is to cache *aggressively where staleness is harmless* (catalog metadata, computed prices, feature flags) and *cautiously or not at all where it is not* (real-time inventory, account balances). Two classic cache hazards to plan for: the **thundering herd** (a popular key expires and a thousand concurrent requests all miss and stampede the source at once — fix with a short lock or request coalescing so only one fetch happens) and **cache stampede on cold start** (a fresh deploy has an empty cache and the source service gets hammered — fix by pre-warming or by serving slightly stale data during warm-up). These are exactly the failure modes the dedicated caching post drills into; the point here is that caching is the single biggest lever on a read-heavy critical path, and it pays you twice — once in latency, once in the cross-service traffic you no longer generate.

#### Worked example: what a 95 percent hit rate is worth

ShopFast's pricing path served 10,000 rps, each request previously making a 60ms round trip to the pricing service. After caching with a 60-second TTL, the hit rate settled at 95 percent. The latency effect on the *average* read drops to `0.95 × 0.5ms (hit) + 0.05 × 60ms (miss) ≈ 3.5ms` — a 17x improvement on the mean. But the more important effects are on the tail and the cost. On the tail: 95 percent of requests now skip the pricing round trip entirely, so the pricing-service span disappears from the p95 of the checkout, and only the unlucky 5 percent still pay it. On the cost: the pricing service's inbound load drops from 10,000 rps to `0.05 × 10,000 = 500 rps`, a 95 percent reduction, which means you can right-size the pricing service down to a fraction of its former replica count — caching paid you a third time, in the compute you no longer need to provision for the service whose load it absorbed. This is why caching sits high on the lever matrix: one change improves mean latency, tail latency, cross-service traffic, *and* the downstream service's compute bill simultaneously. The only thing it costs you is reasoning carefully about staleness, which for a price that changes a few times a day is a cheap thing to reason about.

### Lever 3: connection pooling, keep-alive, and HTTP/2 reuse

Here is a tax almost everyone pays by accident. If your service opens a *new* TCP connection (and a new TLS handshake) for every outbound request, you pay 2 to 5 milliseconds of connection-setup cost on *every single call* — cost that has nothing to do with the work. The fix is connection pooling and keep-alive: maintain a pool of warm, already-handshaked connections and reuse them. With HTTP/2 you go further and multiplex many concurrent requests over a *single* connection, eliminating both the per-request handshake and the head-of-line blocking of HTTP/1.1's one-request-per-connection model.

The trap is that many HTTP client defaults are *terrible* for a service that makes thousands of calls per second. The default Go `http.Client` reuses connections, but its default `MaxIdleConnsPerHost` is only 2 — meaning under concurrency it constantly closes and reopens connections to the same downstream, silently paying the handshake tax. You must tune the transport explicitly.

```go
// A connection pool tuned for a high-throughput downstream, with keep-alive
transport := &http.Transport{
    MaxIdleConns:        200,
    MaxIdleConnsPerHost: 100,            // default is 2 — far too low under load
    IdleConnTimeout:     90 * time.Second,
    ForceAttemptHTTP2:   true,           // multiplex over one connection
    DialContext: (&net.Dialer{
        Timeout:   2 * time.Second,
        KeepAlive: 30 * time.Second,     // TCP keep-alive probes
    }).DialContext,
}
client := &http.Client{Transport: transport, Timeout: 1 * time.Second}
```

The same principle applies everywhere connections are expensive: database connection pools (a service that opens a fresh Postgres connection per request will fall over long before its CPU does), gRPC channels (a single gRPC channel multiplexes many calls and should be created once and reused, not per request), and Redis client pools. The measurable win is twofold: you remove the handshake cost from the critical path (often 2 to 5ms per call, which across a 6-call fan-out is meaningful), and you stop exhausting the downstream's connection table under load, which is a common cause of the cliff where a service is fine at N rps and collapses at 1.5N.

### Lever 4: go async when the caller does not need the result now

A large fraction of "work" a service does on the critical path does not actually need to finish before you respond to the user. When ShopFast places an order, it must authorize payment and reserve inventory *synchronously* — the user needs to know the order succeeded. But it does *not* need to send the confirmation email, update the recommendation model, write to the analytics pipeline, or sync the order to the warehouse management system before responding. Those are fire-and-forget: do them after you have already told the user "order placed."

The fix is to move that work off the critical path by publishing an event and letting downstream consumers handle it asynchronously. This is the choreography pattern from [event-driven microservices](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration), and the reliable way to publish the event without losing it is the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) — write the event to an outbox table in the same database transaction as the order, and a separate process relays it to the message broker.

```python
async def place_order(order):
    async with db.transaction():
        await orders.insert(order)                       # the order itself
        await outbox.insert({                            # event, same txn
            "type": "OrderPlaced",
            "payload": order.to_json(),
        })
    return {"status": "placed", "order_id": order.id}    # respond NOW
    # email, analytics, warehouse sync all happen async,
    # driven by consumers of the OrderPlaced event — OFF the critical path
```

The latency win is whatever those side-effects were costing on the synchronous path — often 100ms or more if the confirmation email service was slow or the analytics write was blocking. The cost, and it is a real one, is **eventual consistency**: the moment you respond "placed" before the email is sent, there is a window where the order exists but the confirmation has not gone out, and you must design for that window (retries on the consumer side, idempotency so a redelivered event does not double-send, monitoring that the outbox is draining). This is the trade-off the [data consistency](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) post unpacks. The senior rule: move work off the critical path whenever the caller genuinely does not need the result *now*, and pay the eventual-consistency tax deliberately rather than discovering it in production.

There is a second, subtler win from going async that matters at scale: it *smooths* load. A synchronous fan-out couples the rate at which you accept requests to the rate at which every downstream can process them — if the analytics service slows down, it backs pressure all the way up to the user-facing checkout. Moving that work behind a queue decouples the two: the checkout keeps accepting orders at 10,000 rps and dropping events on a topic, and the analytics consumer drains the topic at whatever rate it can sustain, with the queue absorbing the difference as a buffer. This is the backpressure story from the [rate limiting and backpressure](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) post, and it is why async is not only a latency optimization but a *resilience* one — a slow non-critical dependency can no longer take down the critical path. The flip side is that you must watch the queue: a consumer that falls permanently behind turns "eventually consistent" into "never consistent," so consumer lag becomes a first-class alert, exactly the kind of golden signal the [SLOs and golden signals](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) post tracks. The litmus test for "should this be async?" is one honest question: *if this step failed silently for thirty seconds, would the user notice or care during their checkout?* If no — email, analytics, warehouse sync, recommendation updates — it belongs off the critical path. If yes — payment authorization, inventory reservation — it stays synchronous and you optimize it in place.

### Lever 5: deadline propagation and tail-latency tricks

Two advanced moves address the tail specifically, and they are where staff-level latency work lives.

**Deadline propagation** means passing a shared deadline down the entire call chain so that no service does work for a request that has already given up. If the gateway sets a 500ms budget for checkout, and 400ms have already elapsed by the time the request reaches the shipping service, the shipping service should be told "you have 100ms left" — and if it cannot finish in 100ms, it should fail fast rather than spend 300ms computing a result that will be thrown away because the client already timed out. Without deadline propagation, a slow upstream causes *wasted work* all the way down the chain: every service keeps grinding on a request nobody is waiting for anymore, which burns CPU and worsens contention for the requests that *are* still live. gRPC propagates deadlines natively; with HTTP you propagate a header and have each service derive its remaining budget.

```go
// Each hop derives its remaining budget from the incoming deadline.
func handler(w http.ResponseWriter, r *http.Request) {
    deadline := time.Now().Add(500 * time.Millisecond) // gateway budget
    if hdr := r.Header.Get("X-Request-Deadline"); hdr != "" {
        if t, err := time.Parse(time.RFC3339Nano, hdr); err == nil {
            deadline = t                                // honor upstream's deadline
        }
    }
    ctx, cancel := context.WithDeadline(r.Context(), deadline)
    defer cancel()
    // pass the same deadline downstream so they fail fast too
    out := callDownstream(ctx, deadline)
    _ = out
}
```

**Hedged requests** are the classic tail-latency trick, from Google's "The Tail at Scale" paper. The idea: if a request to a replica has not returned by the time you have crossed, say, the p95 latency, send a *second* request to a different replica and take whichever returns first. Because a slow response is usually a property of one unlucky replica (a GC pause, a noisy neighbor, a momentary queue), the second request usually lands on a healthy replica and returns fast, dramatically cutting the p99 and p999 at the cost of a small amount of extra load (only the slow fraction of requests get hedged, so the extra load is a few percent). This is closely related to the resilience patterns in [resilience patterns: timeouts, retries, circuit breakers, bulkheads](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) — a hedge is a *speculative retry* fired before the first attempt has failed, and it shares the same hazards (you must make the operation idempotent, or two writes will both land). Use hedging on idempotent reads, where it is nearly free safety; never use it on a non-idempotent write unless you have the [idempotency](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) machinery to deduplicate.

![A trace waterfall as a timeline showing the gateway receiving the inventory N plus one cluster and the serial pricing and shipping waits stacking up to 600ms](/imgs/blogs/performance-and-cost-optimization-in-microservices-5.webp)

The trace waterfall above is the artifact that drives all of this. Reading it left to right tells you the *story* of the request: where it waited, where it fanned out, where the tail lives. The discipline is to keep that waterfall open the entire time you optimize, because it is the only honest scoreboard for whether your change actually moved the span you targeted.

## The optimization spine: cost

Latency is half the job. The other half is dollars, and dollars are where the bigger, more embarrassing wins usually hide — because nobody is watching them in real time the way they watch latency dashboards. Treat dollars as a first-class metric: put cost per request and cost per month on a dashboard next to p99, and review them with the same seriousness.

The most useful single number to compute is **cost per request**, because it converts an abstract monthly bill into a unit economic that the whole team can reason about. Take the total infrastructure cost for a service or a flow over a month, divide by the number of requests it served, and you get a figure like "checkout costs us \$0.0008 per request." That number is powerful for three reasons. First, it makes regressions visible: if a deploy doubles cost per request, that is a bug as surely as a latency regression, and now you can alert on it. Second, it makes the economics of the business legible — if checkout costs \$0.0008 and the average order margin is a few dollars, infrastructure is a rounding error and you should optimize for latency; if you are running a free-tier feature at scale where cost per request actually approaches your revenue per request, cost optimization is existential. Third, it lets you compare services honestly: a service that looks "cheap" in absolute dollars might be wildly expensive per request because it serves little traffic, and a service that looks expensive might be your most efficient. Cost attribution — tagging every resource with the service and team that owns it, so the bill can be sliced by owner — is the unglamorous prerequisite that makes all of this possible, and it is the first thing a FinOps practice sets up. Without attribution you have one giant bill and no idea which service to optimize; with it you have a ranked list of where the dollars go, which is the cost equivalent of a trace.

![A cost stack showing idle over-provisioned compute and cross-AZ egress dominating the monthly bill while useful CPU work is a small slice](/imgs/blogs/performance-and-cost-optimization-in-microservices-7.webp)

The cost stack above is the bill, decomposed. The shocking-but-typical breakdown: idle, over-provisioned compute is the single largest slice, cross-AZ egress is the silent second, observability cardinality is a sneaky third, managed datastores are a visible-but-modest fourth, and the actual useful CPU work — the thing you are paying for — is a thin slice at the bottom. Every slice above "useful work" is a target.

### Lever 6: right-size requests and limits (the number-one waste)

The single biggest source of cloud waste in Kubernetes microservices is over-provisioned pods — pods that *request* far more CPU and memory than they actually use. This happens for a completely human reason: when a team creates a service, they pick a request value by guessing ("2 vCPU and 4GB sounds safe"), they never revisit it, and Kubernetes faithfully reserves that 2 vCPU on a node *whether or not the pod uses it*. A pod requesting 2 vCPU and sitting at 8 percent utilization is reserving 1.84 vCPU of pure waste — and because the scheduler packs nodes based on *requests*, not *usage*, those padded requests force you to run far more nodes than the actual work needs. The [Kubernetes for microservices](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) post covers the mechanics of requests and limits; here is the cost angle.

![A before and after comparison showing over-provisioned pods requesting 2 vCPU at 8 percent usage versus right-sized pods at 0.5 vCPU with horizontal autoscaling packing nodes far tighter](/imgs/blogs/performance-and-cost-optimization-in-microservices-6.webp)

The fix is to look at *actual* utilization (from metrics over a representative window, including peaks) and set the request to match real usage plus a sensible headroom, then let the Horizontal Pod Autoscaler (HPA) add replicas when load rises rather than padding every replica for a peak it rarely sees.

```yaml
# Right-sized: request matches measured p95 usage + headroom; HPA handles peaks.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inventory
spec:
  replicas: 2                       # floor for availability
  template:
    spec:
      containers:
        - name: inventory
          resources:
            requests:
              cpu: "500m"           # was 2000m at 8% usage
              memory: "512Mi"       # was 4Gi, measured peak ~380Mi
            limits:
              cpu: "1000m"          # burst headroom (limit > request)
              memory: "768Mi"       # leave room above peak to avoid OOMKill
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inventory
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inventory
  minReplicas: 2
  maxReplicas: 12                   # scale OUT under load, not pad every pod
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65    # add replicas above 65% CPU
```

Two senior cautions. First, **the gap between request and limit matters.** Set the request to your steady-state usage so the scheduler packs efficiently, and the limit higher to allow bursts — but never set the *memory* limit so close to peak that a brief spike triggers an OOMKill (the kernel killing your container for exceeding its memory limit), because a flapping OOMKill loop is far more expensive in incidents than the memory you saved. Second, **right-sizing without the HPA is dangerous** — if you cut the request to match average usage but do not let the system scale out, the next traffic spike has nowhere to go. Right-sizing and autoscaling are a *pair*.

#### Worked example: right-sizing across 40 services

Suppose ShopFast runs 40 services, each with 3 replicas, each requesting 2 vCPU but averaging 8 percent utilization. That is `40 × 3 × 2 = 240 vCPU` reserved, of which only `240 × 0.08 ≈ 19 vCPU` is actually used. Right-size each to request 0.5 vCPU (still 6x its average, leaving comfortable headroom) and the reservation drops to `40 × 3 × 0.5 = 60 vCPU` — a 75 percent reduction in reserved compute. On a cloud where a vCPU-month runs roughly \$25 to \$30, cutting 180 reserved vCPU saves on the order of `180 × \$27 ≈ \$4,860 per month`, or about \$58,000 per year — from a config change, with zero impact on latency because the pods were using a fraction of what they reserved. This is the single most common five-figure saving available in a microservices estate, and it is almost always sitting there untouched because nobody made it anyone's job.

### Lever 7: autoscale to actual load, including scale-to-zero

The HPA above scales *out* under load and *in* when load drops, which means you stop paying for peak capacity 24/7. The discipline here is to scale on the *right* signal. CPU utilization is the default, but it is the wrong metric for many services: an I/O-bound service that spends its time waiting on a downstream will sit at low CPU even when it is saturated on concurrency, so it never scales out when it should. The better signals are usually the ones that actually correlate with the work — requests per second, queue depth, or in-flight concurrency — which you expose as custom or external metrics and target directly. ShopFast's checkout BFF, for example, scales on requests per second per pod rather than CPU, because at 65 percent CPU it was already queuing requests and bleeding p99.

The deeper version of this lever is **scale-to-zero** for workloads that are spiky or non-production: dev and staging environments that nobody uses overnight, batch jobs that run hourly, and event-driven consumers that only need to exist when there is a queue to drain. With a tool like KEDA (Kubernetes Event-Driven Autoscaling) you can scale a consumer based on queue depth and let it drop to *zero* replicas when the queue is empty — paying nothing for the service when there is no work, and spinning it back up the instant a message lands.

```yaml
# KEDA: scale the order-events consumer on queue depth, down to zero when idle.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-events-consumer
spec:
  scaleTargetRef:
    name: order-events-consumer
  minReplicaCount: 0            # scale to ZERO when the queue is empty
  maxReplicaCount: 20
  cooldownPeriod: 120           # wait 120s of idle before scaling to zero
  triggers:
    - type: kafka
      metadata:
        bootstrapServers: kafka:9092
        consumerGroup: order-events
        topic: order-placed
        lagThreshold: "100"     # add a replica per 100 messages of lag
```

Scaling a dev cluster to zero outside business hours alone can cut its cost by 60 to 70 percent (it runs roughly 50 hours a week out of 168). The caveat is cold-start latency: a service scaled to zero pays a startup cost when the first request arrives, so scale-to-zero is for workloads that can tolerate a few seconds of cold start (dev, async consumers, queue workers) and *not* for a latency-sensitive synchronous path where a user is waiting. The other caveat is the *scale-down* aggressiveness: scale in too fast after a spike and you thrash, paying repeatedly to cold-start pods you just killed. A stabilization window on the HPA (only scale down after sustained low load for, say, five minutes) is the standard guard.

### Lever 8: spot and preemptible instances for stateless work

Cloud providers sell spare capacity at a 60 to 90 percent discount as "spot" or "preemptible" instances, with the catch that they can be reclaimed with little notice. For *stateless* services — which, if you followed the [anatomy of a well-built microservice](/blog/software-development/microservices/anatomy-of-a-well-built-microservice) guidance, is most of them — this is nearly free money: run a fraction of your replicas on spot, let the HPA and the scheduler reschedule any that get reclaimed, and keep a baseline of on-demand replicas so a mass reclamation does not take you down. The pattern is a mixed node pool: a stable on-demand pool sized to handle your floor of traffic, plus a spot pool that absorbs the bulk of variable load at a deep discount. Stateful services (databases, anything with local disk that matters) stay on on-demand. Done right, moving 70 percent of a stateless fleet to spot cuts that fleet's compute cost by roughly half.

The discipline that makes spot safe is treating reclamation as a *normal* event, not an emergency. Spot nodes get a short termination notice (typically around two minutes on the major clouds) before they are reclaimed, and a well-behaved service handles that exactly the way it handles a rolling deploy: it drains in-flight requests, deregisters from the load balancer, and lets the scheduler place a replacement. This is the same graceful-shutdown discipline the [health checks post](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) covers — readiness gates, `preStop` hooks, and respecting `SIGTERM` — and if your services already do it for deploys, they get spot tolerance for free. The one structural guard is a Pod Disruption Budget so the scheduler never reclaims so many replicas at once that you drop below your availability floor; and you spread the workload across multiple spot instance types and zones so a single instance-type shortage cannot reclaim your whole fleet simultaneously. The failure mode to avoid is naively running a stateful or non-idempotent workload on spot — a half-finished payment on a reclaimed node, with no idempotency key, is exactly the kind of correctness bug that turns a cost optimization into an incident.

### Lever 9: cut cross-AZ traffic with zone-aware routing

Recall the \$50,000-a-year cross-AZ egress from the opening worked example. That traffic exists because Kubernetes, by default, load-balances a service's calls across *all* healthy replicas regardless of zone — so an order pod in zone A happily sends a third of its inventory calls to inventory pods in zones B and C, paying egress on each. The fix is **topology-aware routing** (in Kubernetes, `Service.spec.trafficPolicy` / topology-aware hints): prefer routing to a replica in the *same* zone as the caller, and only cross zones when there is no healthy local replica.

![A service call graph showing zone-aware routing keeping inventory calls inside one availability zone with zero egress and crossing zones only on local failover](/imgs/blogs/performance-and-cost-optimization-in-microservices-8.webp)

```yaml
# Prefer same-zone endpoints; cross zones only when no local pod is healthy.
apiVersion: v1
kind: Service
metadata:
  name: inventory
  annotations:
    service.kubernetes.io/topology-mode: "Auto"   # topology-aware routing
spec:
  selector:
    app: inventory
  trafficDistribution: PreferClose                 # keep traffic in-zone
  ports:
    - port: 80
      targetPort: 8080
```

This relates directly to [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing): zone-aware routing is a *load-balancing policy*, and the trade-off is that you must keep enough replicas in each zone to serve that zone's traffic, or you defeat the purpose (a zone with one replica that gets busy will spill to other zones anyway). The win is direct: keep, say, 90 percent of cross-service traffic in-zone and you cut the cross-AZ egress bill by roughly 90 percent. This lever becomes even more important across *regions*, where egress is an order of magnitude pricier and latency is tens of milliseconds — the subject of the forward-linked [multi-region microservices and data locality](/blog/software-development/microservices/multi-region-microservices-and-data-locality) post.

#### Worked example: zone-aware routing on the egress bill

From the opening example, ShopFast egresses about 13.7 TB/day across zones, costing ~\$137/day (~\$50k/year). Suppose topology-aware routing keeps 90 percent of that traffic in-zone. The cross-AZ volume drops to ~1.37 TB/day, the cost to ~\$13.70/day, or about \$5,000/year — a \$45,000 annual saving from a routing policy and a check that each zone has enough replicas. Note the second-order win: in-zone hops are also *faster* (same-zone round trips are sub-millisecond versus 1 to 2ms cross-zone), so this lever shaves a little latency too. It is one of the rare moves that helps both metrics at once.

### Lever 10: cut observability cardinality cost

Observability is essential — the [distributed tracing](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) post argues you cannot run microservices without it — but it has a cost that scales with *cardinality*, the number of unique time series and unique trace/label combinations you emit. The classic blowup: someone adds a metric label like `user_id` or `request_id` or `full_url`, and suddenly a single metric becomes millions of unique time series, each of which the metrics backend must store and index. Cardinality is multiplicative — if a metric has labels for `endpoint` (50 values), `status_code` (15 values), `region` (3 values), and someone adds `customer_id` (2 million values), the time-series count jumps from `50 × 15 × 3 = 2,250` to `2,250 × 2,000,000 = 4.5 billion` series, and most managed metrics backends bill roughly per active time series. Cardinality cost is frequently 10 to 20 percent of the cloud bill in a mature microservices estate, and it is almost always reducible without losing the signal you actually use.

The levers are concrete. Drop high-cardinality labels from metrics entirely — never put unbounded IDs (user, request, order, session) in a metric label; that detail belongs in traces and logs, which you store and sample differently. Sample traces intelligently rather than recording 100 percent: *tail-based* sampling makes the keep/drop decision after the trace completes, so you can keep every error trace and every slow trace (the ones you will actually look at) while dropping the overwhelming majority of fast, successful traces that all look identical. Set log levels and retention by value — keep `INFO` and above for a short window and `ERROR` longer, rather than retaining `DEBUG` from every service for ninety days. The senior framing: observability is a budget you spend to find problems, so spend it where problems hide (errors, tails, the critical path) and stop paying to store millions of identical "everything is fine" data points.

#### Worked example: a single label that doubled the bill

A team at ShopFast added a `session_id` label to the request-latency histogram on the gateway "so we can correlate latency to a session." The gateway sees roughly 5 million unique sessions a day, the histogram has 12 buckets, and it already carried `endpoint` (40) and `status` (10) labels. Before the change the metric was about `40 × 10 × 12 = 4,800` time series. After, it became `4,800 × 5,000,000 = 24 billion` series — and the metrics bill, which is dominated by active series, roughly doubled overnight for *one* well-intentioned label. The fix was to remove the label from the metric (sessions are correlated through traces, not metrics) and the bill snapped back the next day. The lesson generalizes: cardinality cost is rarely a slow creep; it is usually one or two specific labels, and finding them is a five-minute query against the metrics backend's own per-series accounting, not a rewrite.

### Lever 11: sometimes the fix is fewer services

The most counterintuitive cost-and-latency lever is to *delete a boundary*. If two services are so chatty that they make dozens of round trips per request, share the same data, and are always deployed together by the same team, the network between them is paying the microservices tax for *no benefit* — you get all the cost (network hops, serialization, separate deploys, separate observability) and none of the upside (independent scaling, team autonomy). This is the [distributed monolith anti-pattern](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith): services that are physically separate but logically coupled. The cure is to **merge them back** into one service (or one process), turning the chatty network calls into in-process function calls — which removes the network-hop tax, the serialization tax, and the per-service overhead in one move. This is not heresy; it is the same honest reasoning that says microservices are a cost, covered in [what microservices are and when not to use them](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them). The trade-off is that you give up the independent deployability of those two units, so only consolidate where you were not actually using that independence.

How do you know two services are candidates to merge? There are clear signals, and a senior looks for them in the data rather than the org chart. The first signal is that they are *always deployed together* — every change to one requires a coordinated change to the other, which means they do not actually have an independent release cadence (the one benefit that most justifies a service boundary). The second is *chatty synchronous coupling* — the trace shows one calling the other many times per request, on the critical path, with no caching opportunity because the data is genuinely needed fresh. The third is *shared ownership* — the same team owns both, so the team-autonomy argument for splitting them never applied. The fourth is a *shared database* or a thin "data service" that exists only to wrap a table the other service is really the owner of. When several of these are true, the boundary is paying tax for nothing. When you merge, the saving is large and on *both* axes: latency drops (the cross-service round trips become function calls, often shaving tens of milliseconds), and cost drops (one less service's worth of idle replicas, sidecars, and observability footprint). The reverse caution is just as important — do *not* merge services that genuinely deploy and scale independently or are owned by different teams, because then you are trading a real benefit for a modest cost saving, and you will regret it the next time those teams need to ship without coordinating.

## Measuring the win: prove it in p99 and dollars

An optimization you cannot measure did not happen. Every lever above must be validated against numbers from before and after, on the *same* metrics you used to find the problem: p50 and p99 latency, throughput (requests per second the system sustains before latency degrades), error rate, dollars per request and dollars per month, and utilization (the percentage of reserved resources actually used — the honest measure of whether right-sizing worked). For ShopFast, the consolidated scorecard after applying the levers told the whole story.

| Metric | Before | After | Lever |
| --- | --- | --- | --- |
| Checkout p50 | 110ms | 55ms | Parallel fan-out + cache |
| Checkout p99 | 600ms | 180ms | N+1 batch + parallel + cache |
| Inventory calls / checkout | 50 | 1 | Batch API |
| Cross-service round trips | 6 serial | 4 parallel | errgroup fan-out |
| Reserved vCPU (40 svc) | 240 | 60 | Right-sizing + HPA |
| Pod utilization | 8% | ~60% | Right-sizing |
| Cross-AZ egress / year | ~\$50k | ~\$5k | Zone-aware routing |
| Cost / month (platform) | baseline | −40% | Right-size + spot + egress |

The headline ShopFast result: **p99 600ms → 180ms (a 70 percent cut) and total platform cost down 40 percent.** Notice what carried each win. The p99 came mostly from killing the N+1 and the serial chain — pure structural fixes, no new infrastructure. The cost came mostly from right-sizing the over-provisioned fleet and cutting cross-AZ egress — config and policy changes, no code rewrite. None of these required abandoning microservices; they required *managing the tax* the architecture levies, with data.

## Stress-testing the design

A senior does not declare victory after a happy-path measurement. You stress-test the optimized system the way production will.

**"The p99 is 5x the p50 — where's the tail?"** After the structural fixes, ShopFast's p50 was 55ms but p99 was 180ms — still a 3.3x ratio. That gap is not the critical path (which is now fast for everyone); it is *variance*, and the usual suspects are: cold connection pools right after a deploy (fix: pre-warm pools, raise `MaxIdleConnsPerHost`), garbage-collection pauses in one runtime (fix: tune GC, or hedge reads so a paused replica does not stall the request), a single slow replica skewing the tail (fix: outlier ejection, which the [service mesh](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one) and load balancers can do automatically), and the fan-out amplification we computed earlier (fix: hedged requests on the parallel reads). The discipline: a fat tail is a *distribution* problem, not a *mean* problem, and you chase it with traces of the slow requests specifically — filter the trace backend for requests over the p99 threshold and look at what those particular requests did differently.

**"The bill doubled at 2x traffic — why super-linear?"** This is the Universal Scalability Law biting. If cost scaled linearly, 2x traffic would be 2x cost; when it is more, coordination overhead is growing faster than the work. The investigation: look for contention (a shared database hitting connection-pool limits and forcing serialization, a single-leader write path, a lock) and coherency cost (retry storms amplifying load — the cascading-failure pattern from the [resilience post](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads); cache invalidation broadcasts; chatty cross-service sync that grows with the square of participants). The fix is rarely "add more capacity" — that just pays more for the same inefficiency. It is to reduce the coordination: cache to cut round trips, batch to cut chatter, add backpressure and load-shedding (the [rate limiting and backpressure](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) post) so the system degrades gracefully instead of melting down, and break shared bottlenecks (shard the hot database, the techniques in [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding)).

**"A downstream dependency got slow — does the optimized path still hold up?"** Parallel fan-out makes you *more* exposed to a single slow dependency, not less, because you now wait on the slowest of the parallel calls. So the stress test is: degrade the pricing service to a 2-second p99 and watch. Without deadline propagation and a per-call timeout, that slow pricing call now dominates *every* checkout's p99. The defenses are exactly the resilience patterns: a tight per-call timeout (so a slow pricing call fails fast at 200ms instead of hanging for 2s), a circuit breaker (so once pricing is clearly broken you stop calling it and serve a cached or default price), and graceful degradation (the [partial failures post](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation) — show the order without the precise shipping estimate rather than failing the whole checkout). Optimization and resilience are not separate concerns; a fast happy path that has no answer for a slow dependency is not actually fast in production.

## Case studies

Real engineering organizations have published the lessons below; the numbers are framed as orders of magnitude where exact figures are not public.

**Right-sizing at scale — the FinOps norm.** The cloud-cost-management discipline now called FinOps exists largely because over-provisioning is *the* default failure mode of Kubernetes estates. Engineering organizations that run cluster-utilization audits routinely report idle-resource fractions of 50 to 70 percent before right-sizing — pods reserving far more than they use, exactly the ShopFast pattern. Public write-ups from teams adopting tools like the Vertical Pod Autoscaler in recommendation mode, Goldilocks, or commercial right-sizing platforms commonly report cutting compute cost by 30 to 50 percent with no latency regression, purely by aligning requests with measured usage and letting the HPA handle peaks. The lesson: the biggest cost win in a mature microservices estate is almost never a clever rewrite; it is the unglamorous work of measuring real utilization and deleting the padding, and it pays five or six figures.

**Cross-zone traffic at hyperscale — the egress nobody budgeted.** Companies running very large Kubernetes fleets have publicly described cross-availability-zone network traffic as a surprising and material line item — at hyperscale, inter-service traffic spread blindly across zones can run into millions of dollars a year. The fixes the industry converged on are exactly the ones in this post: topology-aware routing to keep service-to-service calls in-zone, co-locating chatty service groups in the same zone, and (at the extreme) running zonal copies of stateless service stacks so a request can be served end to end without crossing a zone boundary. The order-of-magnitude lesson: when traffic is in the terabytes-per-day range, the *placement* of your pods is a cost decision, and the default placement is the expensive one.

**Consolidating chatty services — the distributed-monolith U-turn.** The most cited example of "the fix was fewer services" is Segment's well-documented decision (described publicly in their engineering blog) to *reverse* a microservices decomposition. They had split a data-integration pipeline into a service per destination — over a hundred services — and found that the operational overhead, the shared-library version drift, and the per-service infrastructure cost vastly outweighed the benefit, since the services were not actually deployed or scaled independently in a way that mattered. Consolidating back to a monolith dropped their operational burden and cost dramatically. The lesson is not "monoliths win"; it is the honest one this whole series teaches — microservices are a cost, that cost is only worth paying where you genuinely use the independence, and a senior is willing to *merge a boundary back* when the data says the split is pure tax. Uber's later move to its Domain-Oriented Microservice Architecture (DOMA), which groups fine-grained services into coarser domain-aligned units behind a gateway, is the same instinct applied at a larger scale: fewer, coarser boundaries cut both the chatter and the cognitive and operational cost.

**The tail at scale — hedged requests.** Google's "The Tail at Scale" (Dean and Barroso, 2013) is the canonical source for why p99 dominates user experience in fan-out systems and for the hedged-request technique. The reported result that made hedging famous: sending a backup request after a short delay (around the 95th-percentile latency) cut the 99.9th-percentile latency of a benchmark service roughly in half while adding only a few percent of extra load. The lesson that generalizes to every microservices fleet: in a system where one request touches many services, the *tail* of each dependency, not its median, sets your user-visible latency — so attacking the tail (hedging, outlier ejection, deadline propagation) often beats making the median faster.

**Protobuf over JSON on the hot path — the serialization tax.** Multiple engineering organizations have published the result of switching internal service-to-service communication from JSON-over-HTTP to protobuf-over-gRPC on high-throughput paths, and the consistent finding is a meaningful reduction in both latency and CPU: payloads shrink (binary protobuf is typically a fraction of the equivalent JSON size, cutting both serialization CPU and the bytes on the wire, which also cuts egress), and the encode/decode path is several times cheaper. The order-of-magnitude lesson, consistent with the latency budget breakdown earlier in this post: on a path that moves a lot of data between services, serialization is not a rounding error — it can be a double-digit percentage of CPU and latency, and changing the wire format is one of the higher-leverage changes available when a flame graph points at the encoder. The caveat the same teams report is that protobuf trades human-readability and schema flexibility for that performance, so it earns its place on internal hot paths, not necessarily on public, low-volume, or rapidly-evolving APIs — the exact trade-off the [REST vs gRPC vs GraphQL](/blog/software-development/microservices/rest-vs-grpc-vs-graphql-for-service-apis) post weighs.

## When to optimize — and the premature-optimization warning

Knuth's line that "premature optimization is the root of all evil" is quoted to death and still right. In a microservices context it means: **do not optimize a path until a trace or a bill tells you it is the problem.** Most code in most services does not need to be fast — it needs to be *correct and clear*, because the slow part is almost always somewhere you would not have guessed. Optimizing on intuition is how juniors spend a week indexing a 4ms query while a 220ms N+1 sits untouched in the next service over.

So the senior sequence is: ship the simple, correct version first; instrument it (traces, metrics, cost dashboards) so you can *see* where it spends time and money; and optimize only when there is a real, measured problem and a real budget to claw back. The decision tree below is the routing logic.

![A decision tree that routes a latency problem to read the trace then batch or cache and routes a cost problem to read the bill then right-size or use zone-aware routing](/imgs/blogs/performance-and-cost-optimization-in-microservices-9.webp)

The two branches are the two questions: *Is it slow, or is it expensive?* If slow, read the trace — chatty call patterns route you to batching and parallel fan-out; hot reads route you to caching. If expensive, read the bill — idle pods route you to right-sizing and the HPA; cross-AZ traffic routes you to zone-aware routing. The unifying principle is at the root: **let the data choose the target.** And there is a real *anti*-optimization case too: do not adopt the heavy machinery (hedging, a service mesh for outlier ejection, sophisticated autoscaling) until you have a problem that needs it, because every one of those tools is itself a cost — operational complexity, more moving parts to debug, more ways to fail. The cheapest optimization is the one you do not need to do because you measured first and the path was fine.

## Key takeaways

- **Microservices are slower and pricier than a monolith by default** — you pay a network-hop tax, a serialization tax, a per-service overhead tax, and a cross-AZ egress tax. Adopting the architecture is signing up to *manage* those taxes, not ignore them.
- **Measure before you touch anything.** A trace tells you where the latency goes; the bill tells you where the dollars go. Optimizing without data means optimizing the wrong thing — the slow part is rarely where you would guess.
- **The critical path is usually overhead, not work.** In ShopFast's 600ms p99, business logic was under 4 percent; the rest was an N+1, a serial chain, and network tax — all recoverable.
- **Highest-payoff, lowest-risk latency levers first:** batch the N+1, parallelize independent calls, cache hot reads, pool connections. Async and consolidation come later because they cost you in correctness and coupling.
- **p99, not p50 or the average, is what users feel** — and fan-out amplifies the tail. Look at the p50–p99 gap as a diagnostic: a big gap means variance (chase it with traces of slow requests and tail tricks like hedging); a uniform slowness means the critical path.
- **Right-sizing over-provisioned pods is the number-one cost win** — typically a 30 to 50 percent compute saving with zero latency impact, because the pods were reserving far more than they used. Pair it with the HPA so peaks still scale out.
- **Cross-AZ egress is the silent five-figure line item.** Zone-aware routing keeps traffic in-zone, cutting both the bill and a little latency at once.
- **Treat dollars as a first-class metric.** Put cost per request and cost per month on the dashboard next to p99 and review them with the same seriousness.
- **Sometimes the right optimization is fewer services.** Two chatty, co-deployed, data-sharing services pay the full microservices tax for none of the benefit — merging them back removes network hops, serialization, and idle overhead in one move.
- **Stress-test the optimized path:** a fast happy path with no answer for a slow dependency is not fast in production. Deadlines, timeouts, circuit breakers, and graceful degradation are part of the performance story, not separate from it.
- **The senior point:** optimize the critical path with data — traces and profiles — and never ship an optimization you cannot prove in p99 and dollars.

## Further reading

- Neil Gunther, *Guerrilla Capacity Planning* — the Universal Scalability Law in depth, including how to fit the contention and coherency coefficients to your own load tests; the rigorous version of "why the bill goes super-linear."
- Jeffrey Dean and Luiz André Barroso, "The Tail at Scale" (Communications of the ACM, 2013) — the canonical paper on why p99 dominates fan-out systems and on hedged requests.
- Brendan Gregg, *Systems Performance* — flame graphs, the USE method, and how to profile where CPU and latency actually go within a single service.
- Sam Newman, *Building Microservices* (2nd ed.) — the practitioner's reference for the trade-offs this post operationalizes, especially the chapters on scaling and on when to recombine services.
- The FinOps Foundation framework — the discipline of treating cloud cost as an engineering metric, right-sizing, and showback/chargeback for microservices estates.
- [Inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) — why the network hop is never free, the tax this whole post exists to manage.
- [Distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) — how to produce the trace waterfall that drives every optimization here.
- [Kubernetes for microservices: the essentials](/blog/software-development/microservices/kubernetes-for-microservices-the-essentials) — requests, limits, and the HPA mechanics behind right-sizing.
- [Caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services) and [multi-region microservices and data locality](/blog/software-development/microservices/multi-region-microservices-and-data-locality) — the two forward-linked deep dives on the caching and locality levers introduced here.

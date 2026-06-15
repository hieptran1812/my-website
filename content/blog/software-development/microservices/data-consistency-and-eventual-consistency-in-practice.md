---
title: "Data Consistency and Eventual Consistency in Practice"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Once your data is split across services there is no global ACID anymore, so this is the practical engineering of living with eventual consistency: the session guarantees users actually feel, where to keep a strong boundary, and how to reconcile drift before it corrupts anything."
tags:
  [
    "microservices",
    "eventual-consistency",
    "data-consistency",
    "distributed-systems",
    "software-architecture",
    "backend",
    "idempotency",
    "cap-theorem",
    "reconciliation",
    "conflict-resolution",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-1.webp"
---

A customer at ShopFast clicked "Buy now" on the last copy of a limited-edition sneaker, saw a green "Order placed!" toast, and felt the small dopamine hit of winning the race. Then they refreshed the product page out of habit and the page still said "1 in stock." Confused, they bought it again. Now there were two orders for one pair of shoes, the customer was charged twice, and the warehouse had exactly one box to ship. An engineer who pulled the thread found something worse: the inventory count in the catalog service had been wrong for hours, drifting further from the order service's truth with every sale, because an event had been silently dropped during a deploy three days earlier and nobody had noticed.

None of these were bugs in the ordinary sense. No null pointer, no off-by-one, no typo. They were the *expected* behavior of a system whose data had been split across services and stitched back together with asynchronous events — a system that nobody had explicitly designed to be eventually consistent, but that was eventually consistent anyway, the way a building you forget to heat is eventually cold. The moment ShopFast gave each service its own database and connected them with events, it traded one global ACID transaction for a fleet of independent truths that agree only after a delay. That delay — the gap between "I wrote it" and "everyone sees it" — is not a flaw to be engineered away. It is the price of admission to the architecture, and the entire craft of running microservices is learning to design *for* that gap instead of being ambushed by it.

This post is about that craft. Not the formal theory of consistency models — [the database post on consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) covers the formal hierarchy, and [the CAP theorem and PACELC post](/blog/software-development/database/cap-theorem-and-pacelc) explains *why* the trade is fundamental rather than a mere inconvenience you could code around with enough effort. Read those for the "why it must be so." This post is the practitioner's layer above them: given that your data is eventually consistent across services, *what do you actually build* so that users are not confused, money is not lost, stock is not oversold, and two services that disagree get back in sync before anyone gets hurt? By the end you will be able to look at any cross-service interaction and answer the questions that separate a junior who hopes it works from a senior who knows exactly where the window is and what lives inside it: *where is the strong boundary, what session guarantee does the user need, what happens to a read that races a write, and how does divergence get detected and repaired?*

![A before and after comparison contrasting the imagined single global ACID write where all reads agree instantly against the real database-per-service world where an event propagates over a convergence window before read models agree](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-1.webp)

The figure above is the whole essay in one picture, so let me name both sides plainly because juniors carry the left side around as an unexamined assumption. On the left is the world a single-database application lives in: one commit, and every subsequent read — by anyone, anywhere — sees the result. There is no window. That world is real, it is wonderful, and it is exactly what you give up the day you split the database. On the right is the world you actually operate: the order service commits first to its own database, then an event propagates over the network and through a broker, taking anywhere from fifty milliseconds on a good day to several seconds when a consumer is backed up, and only *after* that propagation do the other services' read models agree. The window between the commit and the agreement is the eventual-consistency window. Everything in this post is a technique for making that window safe to live inside.

## Why microservices force eventual consistency at all

Let me make the "why" concrete before any technique, because the technique only makes sense once you feel the constraint in your bones. In a monolith with one relational database, consistency across your whole domain is *free* in the sense that you do nothing to get it. You wrap the order insert, the inventory decrement, and the payment record in a single `BEGIN ... COMMIT`, and the database's ACID guarantees mean either all three happen or none do, and no other transaction sees a half-finished state. This is not magic; it is the database doing enormous work — write-ahead logging, lock management, isolation levels — on your behalf inside a single process boundary. You inherit linearizability for the whole domain because the whole domain lives in one place that can be coordinated cheaply.

The defining rule of microservices is [database-per-service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices): each service owns its data privately, and no other service may reach into that database directly. This rule is what *gives* you the independence that justifies microservices in the first place — the inventory team can change its schema, switch from Postgres to DynamoDB, or re-index without coordinating with the order team, because nobody else touches inventory's tables. But the same rule is what *takes away* the free global transaction. The moment the order data lives in the order service's Postgres and the inventory count lives in the inventory service's database, there is no `BEGIN ... COMMIT` that can span both. They are different databases, in different processes, possibly on different machines in different availability zones. To make a change atomic across both, you would need a distributed transaction — a two-phase commit (2PC) protocol with a coordinator that holds locks across services until everyone votes to commit.

And here is where theory bites: holding cross-service locks across a network is exactly the thing that destroys the availability and independence you split the monolith to get. If the order service must hold a lock in the inventory database until payment votes, then payment being slow makes order slow, inventory's row is locked the whole time, and a network partition between any two of them leaves locks dangling and the whole flow wedged. This is the practical face of [the CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc): across a network boundary you must choose, when a partition happens, between staying consistent (refuse to proceed, become unavailable) and staying available (proceed, become temporarily inconsistent). PACELC adds the part people forget — *even when there is no partition*, you trade latency for consistency, because coordinating across services to stay strongly consistent costs round-trips on every single request, not just during failures. Two-phase commit is technically possible, and a few systems use it, but for most product surfaces it is a poison: you pay the coordination latency on every request and you inherit the availability of the *least available* participant. So the overwhelmingly common, and usually correct, choice is the other branch: let each service commit locally and stay available, and accept that the other services will catch up *eventually* through asynchronous events. Eventual consistency is not something microservices *opt into* for fun. It is the residue left over once you have correctly refused to spread a transaction across the network.

So the precise statement is this. **Microservices do not make eventual consistency optional; database-per-service plus asynchronous communication makes it the default, and the only thing you choose is whether you design for it deliberately or get surprised by it in production.** The ShopFast double-charge in the intro is what "getting surprised by it" looks like.

### The consistency spectrum you actually operate

It would be a mistake to conclude from all this that everything in a microservices system is eventually consistent and you simply have to live with chaos. The truth is a *spectrum*, and a senior's main job is placing each piece of data on it deliberately. At one end you have **strong consistency within an aggregate**: inside a single service, inside a single database, you still have full ACID, and you should use it ruthlessly for the things that must not be wrong. The inventory service's stock count, the wallet service's balance, the order service's order state machine — each of these is a small island of strong consistency, and the trick is to draw the islands so that every *hard* invariant (the kind that corrupts data or loses money when violated) lives entirely inside one island. At the other end you have **eventual consistency across services**: the catalog's cached "in stock" badge, the search index, the recommendation engine's view of what you bought, the analytics warehouse — all of these are *copies* and *projections* of authoritative data owned elsewhere, and all of them are allowed to lag.

The whole design discipline collapses into one sentence that I want you to tattoo somewhere: **keep the invariant strong inside one aggregate, and let everything across services be eventual.** When you violate this — when you try to enforce a single invariant, like "we never sell more than we have," by coordinating two or three services in real time — you have signed up for distributed-transaction pain and you will lose either correctness or availability. When you respect it — when the "never oversell" invariant lives entirely inside the inventory aggregate and everything else is a lagging projection of that truth — the system is both correct where it matters and available everywhere. The rest of this post is mostly the machinery for respecting that sentence.

## How the write actually propagates: the topology you are reasoning about

Before we can talk about guarantees, you need a crisp mental picture of the path a write takes, because every stale-read bug and every divergence is a story about *where on this path the reader was looking.*

![A branching graph showing a client write committing to the order service owner with an outbox, publishing to an event bus, which fans out to an order read model lagging eighty milliseconds, an inventory projection lagging three hundred milliseconds, and a search index lagging two seconds, all of which a user might read](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-2.webp)

Trace the figure. A write enters the *owner* of the data — the one service that holds the authoritative copy of that fact. The owner commits the change to its own database in a normal local ACID transaction, and in the same transaction it records an outgoing event in an outbox (this is the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing); the sibling post on the [transactional outbox and reliable event publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) covers the microservices framing — the point is that the commit and the "I will tell everyone" are atomic, so you never commit a change you fail to announce or announce a change you failed to commit). A relay then publishes that event to the bus. From the bus it fans out to every downstream that cares, and each downstream applies it to its *own* read model on its *own* schedule.

The crucial insight in this picture is that **the lag is not uniform.** The order service's own read model might be a few milliseconds behind because it is the same service reading its own write. The inventory projection might be a few hundred milliseconds behind because the event has to traverse the broker and a consumer. The search index might be *seconds* behind because reindexing is batched. So at any given instant after a write, the system is not "consistent" or "inconsistent" — it is a gradient, with different observers seeing the new truth at different times. A user who writes an order and then reads the order page is reading a fast, near-write-through path. The same user reading the catalog badge is reading a slow path. The bug in the intro happened because the product page read the *slow* path right after a write went down the fast one. You cannot reason about consistency without first knowing *which read model the reader hit and how far behind it runs.* This is why mature teams put a number on it — a per-read-model "replication lag" or "projection lag" metric — and alert when it exceeds a budget, exactly the way you would alert on database replica lag.

#### Worked example: the lag a user can actually observe

Let me put numbers on it so the window stops being abstract. ShopFast runs at 1,200 orders per second at peak. The order service commits in p50 4ms, p99 18ms, and returns `201 Created` to the client. The outbox relay polls every 100ms and publishes in a batch, so the event hits the bus on average 50ms after commit (0 to 100ms uniformly, so ~50ms mean). The inventory consumer is processing a backlog at peak and runs p50 120ms, p99 600ms behind the bus. Add it up: from the instant the client gets `201`, the inventory projection reflects the decrement after p50 ≈ 50 + 120 = **170ms**, p99 ≈ 100 + 600 = **700ms**. That is the window. If the client's UI re-reads the catalog 8ms after the write — which a single-page app absolutely does — it reads a projection that is, on average, 170ms from being correct, and at p99 is 700ms behind. The user sees their own just-purchased item still showing the old stock count roughly **one time in two** at p50 and essentially *always* if they refresh within 100ms. That is not a rare race. It is the *common* case, and any design that does not account for it is broken for most users on most fast reads.

The honest reframing: eventual consistency is not "usually instant, occasionally stale." For a read that races a write, it is "usually stale for a few hundred milliseconds, and you must design the experience around that few hundred milliseconds." Everything below is how.

There is a subtlety worth pausing on, because it is where juniors underestimate the window and seniors size it correctly. The lag you must design for is not the *average* lag; it is the *tail* lag under load, because the reads that race a write cluster exactly when traffic is highest, which is exactly when consumers are most backed up. At p50 the inventory projection might be 170ms behind, but the moment a consumer falls behind — a slow database, a GC pause, a rebalance after a deploy — the lag does not creep, it *jumps*, because the consumer is now draining a backlog rather than keeping pace. A consumer that processes 5,000 events/second but momentarily receives 8,000/second does not lag by a constant amount; it lags by an *accumulating* amount until the burst passes, and a 10-second burst at +3,000/second leaves 30,000 events queued that take 6 seconds to drain even after the burst ends. So the realistic statement is: the window is sub-second in steady state and seconds-to-tens-of-seconds during the bursts that matter most. Design for the bursts, alert on the lag, and never assume the average. This is why the per-projection lag metric is not a vanity dashboard — it is the number that tells you how stale your "fresh" reads can actually get, and it is the budget your bounded read-your-writes waits are spending against.

## The guarantees that actually matter to users: client-centric consistency

Here is the most liberating idea in this whole subject, and the one that separates an engineer who panics about eventual consistency from one who ships it calmly: **users do not care about global consistency. They care about their own session being coherent.** A user genuinely does not notice or care whether some other user on another continent sees their write at the same microsecond. What they *do* notice — viscerally, as "this app is broken" — is when *their own* actions seem to vanish or contradict themselves. The formal name for the guarantees that capture this is **client-centric** (or session) consistency, and there are four of them that, in practice, are the entire game.

![A stack of guarantee layers showing read-your-writes via a version token on top, then monotonic reads via a sticky session, then monotonic writes via per-key ordering, then consistent prefix via an ordered partition, all sitting over a raw eventual store with no guarantees](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-5.webp)

**Read-your-writes** is the single most important one. It says: after I successfully write something, *I* will see my own write in subsequent reads, even if the rest of the world does not yet. When I post a comment and the page reloads, my comment is there. When I update my profile photo and navigate away and back, it is the new photo. The system is allowed to be eventually consistent for *everyone else*, but it must never tell *me* that the thing I just did did not happen. Violating read-your-writes is the canonical "this app is buggy" experience, and it is exactly the ShopFast double-charge story — the user wrote an order, then read stock that did not reflect it, concluded their order had not gone through, and acted on that false belief.

**Monotonic reads** says: once I have seen a value, I will never see an *older* value on a later read. Time only moves forward within my session. The pathological violation is a user who refreshes a page, sees their order as "Shipped," refreshes again, and sees it as "Processing" — because the second read happened to land on a replica or projection that was *further behind* than the first one. Nothing is more disorienting than data that travels backward in time, and it is depressingly easy to produce when reads are load-balanced across replicas with different lag.

**Monotonic writes** says: my writes are applied in the order I issued them. If I rename a file and then delete it, the system must not apply the delete-then-rename and leave a stale renamed file behind. **Consistent prefix** says: observers see writes in an order consistent with the order they were actually committed — you might see a prefix of the true history, but never a scrambled version where an effect appears before its cause. Consistent prefix is what stops a chat app from showing the reply before the question.

The reason these matter to a practitioner is that **you can provide every one of them without making your system globally strongly consistent**, and that is the whole trick. You are not choosing between "eventual consistency (chaos)" and "strong consistency (expensive)." You are providing *cheap, local, per-session* guarantees on top of an eventually consistent substrate. Here is how each is actually built.

### Providing read-your-writes: version tokens and wait-for-projection

The cleanest mechanism is the **version token** (also called a write token or read fence). When a service performs a write, it returns to the client a token that encodes "this write happened at logical version V." The client stores it and includes it on subsequent reads. A read carrying a token of version V will only be served by a replica or projection that has caught up to *at least* V; if the projection it lands on is behind, it either waits briefly for it to catch up or transparently routes to one that is current. The user gets read-your-writes because their own reads are fenced by their own writes' version, while everyone else's reads are unfenced and stay fast.

```python
# Order service returns a version token on write. The token is just the
# committed log/sequence position of THIS write, signed so a client can't forge it.
@app.post("/orders")
def place_order(req: OrderRequest) -> OrderResponse:
    with db.transaction() as tx:
        order = tx.insert_order(req)
        # outbox row is written in the SAME transaction (see outbox post)
        seq = tx.append_outbox(event="OrderPlaced", payload=order.as_event())
    token = sign_token({"stream": "orders", "seq": seq})  # e.g. HMAC'd opaque string
    return OrderResponse(order_id=order.id, read_token=token, status="PROCESSING")
```

```python
# A read model honors the token: it will not answer with data older than `seq`.
@app.get("/orders/{order_id}")
def get_order(order_id: str, x_read_token: str | None = Header(default=None)):
    required_seq = parse_token(x_read_token).seq if x_read_token else 0
    applied_seq = projection.applied_position()  # how far this projection has consumed
    if applied_seq < required_seq:
        # Option A: wait briefly for the projection to catch up (bounded!)
        if not projection.wait_until(required_seq, timeout_ms=300):
            # Option B fallback: read straight from the owner instead of the projection
            return owner_client.get_order(order_id)
    return projection.get_order(order_id)
```

Two things make this production-grade rather than a toy. First, the wait is *bounded* — you wait up to 300ms for the projection to catch up, and if it does not, you fall back to reading from the owner directly (slower, but correct) rather than returning stale data or blocking forever. Second, the token is *opaque and signed* so a malicious client cannot forge a version to force expensive waits or read another tenant's data. The cost you pay for read-your-writes here is a little extra latency on the reads that carry a token (the ones immediately after a write) and the plumbing to track each projection's applied position. The reads that do *not* carry a token — the vast majority — stay fast and fully eventual.

### Providing monotonic reads: sticky sessions and read-from-write-region

Monotonic reads is usually provided by **stickiness**: pin a user's session to read from the *same* replica or projection instance for the duration of the session, so they never bounce to one that is further behind. At the load-balancer level this is a sticky cookie; at the data-store level it is reading from a designated region or replica. The simplest, sturdiest version of monotonic reads *and* read-your-writes together is **read-from-the-write-region**: for a short window after a user writes, route their reads to the same region (or the leader) that took the write, where the data is freshest. Multi-region systems do this explicitly — after you write in `us-east`, your reads stick to `us-east` for a few seconds, the typical replication lag, before you are allowed to drift back to a nearer replica. The deeper mechanics of how replicas lag and how leaders versus followers behave is [distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless); the practitioner takeaway is that *stickiness converts a system that is monotonic-per-replica into one that is monotonic-per-session*, which is what the user actually needs.

```yaml
# Envoy / gateway: sticky session via a hash on the user id, so a user's reads
# keep landing on the same projection replica -> monotonic reads for that session.
route:
  cluster: order-read-models
  hash_policy:
    - header:
        header_name: x-user-id   # all of one user's reads hash to one replica
load_assignment:
  policy:
    # consistent-hash LB keeps the mapping stable as replicas come and go
    lb_policy: RING_HASH
```

### A note on "tunable" consistency

Many datastores expose this spectrum as a per-operation knob, and it is worth knowing because it is the same idea in a single store rather than across services. In a quorum system with `N` replicas, if your reads touch `R` replicas and your writes touch `W`, then `R + W > N` guarantees a read overlaps the latest write (strong-ish), while `R + W <= N` is faster but may read stale (eventual). Cassandra and DynamoDB let you pick `ONE`, `QUORUM`, or `ALL` per query.

```python
# Cassandra: pick the guarantee per operation. Strong for the balance read,
# eventual (fast, cheap) for the activity feed where staleness is harmless.
session.execute(stmt_read_balance,   consistency_level=ConsistencyLevel.QUORUM)  # R+W>N
session.execute(stmt_read_feed,      consistency_level=ConsistencyLevel.ONE)     # fast, may lag
```

The senior move is *not* to crank everything to `QUORUM`/`ALL` to feel safe — that throws away the availability and latency you came for — but to spend strong consistency only on the few reads where staleness actually corrupts a decision (a balance you are about to debit) and leave the rest eventual.

## Where you genuinely need strong consistency — and how to actually get it

The most damaging mistake I see is engineers reaching for strong consistency *across services* when they should be confining the invariant *within* one service. They feel the danger of overselling or double-spending, correctly conclude "this needs to be strongly consistent," and then incorrectly try to achieve that with a distributed transaction or a synchronous check-then-act spanning two services. The right move is almost always the opposite: **pull the invariant into a single aggregate so the strong consistency you need is local and cheap.**

![A matrix comparing strong consistency within an aggregate against eventual consistency across services on where it holds, coordination cost, availability under partition, and what to use each for](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-3.webp)

The matrix names the trade. Strong-within-an-aggregate costs almost nothing: it is a single local database transaction, the same `BEGIN ... COMMIT` you have used your whole career, and it keeps the service available even under a partition because it never reaches across the network to commit. Eventual-across-services is what you use for everything that is a copy or a projection — and crucially, trying to make *across-services* strong (with 2PC or distributed locks) is the expensive, fragile, availability-destroying option you almost never want. So the design question is never "should this be strong or eventual?" in the abstract. It is "**which single service should own this invariant so that enforcing it is a local transaction?**"

Take "never oversell stock." A junior models it as: order service checks inventory's count, sees stock available, then creates the order, and meanwhile inventory decrements. That is a check-then-act split across two services with a fat race in the middle, and it oversells under concurrency every time. The senior models it as: **the inventory service owns the invariant entirely.** Reserving stock is a single conditional update inside the inventory database, and the order proceeds only if the reservation succeeded.

![A before and after comparison showing an oversell race where two buyers both read an eventual stock count of one and both pass the check, versus a reservation where a conditional update lets exactly one order succeed inside one aggregate](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-7.webp)

```sql
-- Reserve atomically INSIDE the inventory service's own database.
-- The WHERE clause makes the check-and-decrement a single atomic step:
-- no read-then-write race, no two-service coordination, no distributed lock.
UPDATE inventory
SET    available = available - 1,
       reserved  = reserved  + 1,
       version   = version + 1
WHERE  sku = $1
  AND  available >= 1;          -- the invariant lives in this one line
-- rows_affected = 1 -> reservation succeeded; the order may proceed
-- rows_affected = 0 -> sold out; reject the order, no oversell possible
```

This is the entire move, and it is worth dwelling on how *small* it is. There is no distributed transaction, no lock held across services, no quorum. The "never oversell" invariant is enforced by a single row update with a `WHERE` guard, executed by the one service that owns inventory. Two concurrent buyers both issue this `UPDATE`; the database serializes them; the first gets `rows_affected = 1` and the second gets `0` because `available` is now `0`. The loser is rejected cleanly. The order service never sees stock counts at all — it asks inventory to reserve, and inventory says yes or no. Everything *else* about inventory (the "in stock" badge in the catalog, the count on the search results) is allowed to be an eventual projection that lags, because none of those displays enforce the invariant. **The invariant is strong; its shadows are eventual.** That is the pattern, and once it clicks you stop fearing eventual consistency, because you have penned the dangerous part into one cheap, local transaction.

#### Worked example: the oversell race, with numbers

Concretely, what does the naive version cost? Suppose the limited sneaker has 50 units and a flash sale drives 2,000 concurrent "buy" attempts in the first two seconds. In the naive check-then-act-across-services design, each request reads the catalog's cached count (say it reads "50 available"), passes the check, and creates an order; the inventory decrements happen asynchronously and lag. In a 2-second burst with a 170ms projection lag, hundreds of requests read a count that has not yet been decremented by the orders ahead of them. In one real reconstruction I have seen of this exact shape, ~50 units sold became **~340 orders accepted** — a 6.8× oversell — and the company had to cancel and apologize to 290 customers, each of whom had received a "you got it!" email. The cost was not just refunds; it was trust. In the reservation design, the `UPDATE ... WHERE available >= 1` serializes all 2,000 attempts through the inventory row, exactly 50 get `rows_affected = 1`, and the other 1,950 get a clean "sold out" instantly. The latency cost of serializing on one hot row is real — that row becomes a contention point — but it is bounded and local, and it is the *correct* behavior. (If the hot-row contention itself becomes the bottleneck, you shard the stock into N reservation buckets per SKU and reserve from any non-empty bucket; you trade a little "available count is approximate at the very end" for parallelism — a deliberate, scoped relaxation, not an accident.)

### When the invariant truly spans services: the saga

Sometimes the invariant genuinely cannot live in one aggregate — placing an order *must* reserve stock *and* capture payment *and* allocate a shipment, and those are three different services by design. You cannot pull all three into one transaction without recreating the monolith. This is exactly what the [saga pattern](/blog/software-development/microservices/the-saga-pattern-in-practice) exists for: a sequence of local transactions, each in its own service, with **compensating actions** that undo the earlier steps if a later one fails. A saga does not give you atomicity; it gives you *eventual* atomicity — the system passes through inconsistent intermediate states (stock reserved but payment not yet captured) and converges to either "all done" or "all undone." The practitioner's point for *this* post is that a saga is how you implement an invariant that spans services *while staying eventual*: each step is strong-within-its-own-service, and the saga stitches them with compensation rather than a distributed lock. The price is the intermediate window — your UX must show "Processing" during it — and the discipline that every step is idempotent and every compensation is safe to retry, which is the next section.

## The "read-after-write across services" gotcha

I want to isolate the single most common, most confusing failure in this whole space, because once you can *see* it, half your eventual-consistency bugs become obvious. It is the **read-after-write across services** gotcha: you write to service A, then *immediately* read from service B's read model, which has not yet processed A's event.

![A timeline of six events showing a user clicking Buy and the order committing at time zero, a 201 returned at five milliseconds, the UI reading stock from the inventory service at eight milliseconds, the stock still showing the old value because the projection is behind, the OrderPlaced event applied to the projection at two hundred fifty milliseconds, and a re-read agreeing that stock is decremented at two hundred sixty milliseconds](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-4.webp)

Walk the timeline. At T+0 the user clicks Buy and the order commits in the order service. At T+5ms the client receives `201 Created`. The single-page app, being modern and reactive, immediately re-fetches the product to refresh the UI — at T+8ms — and that fetch hits the *inventory* service's projection, which has not yet seen the `OrderPlaced` event. So at T+9ms the UI renders the *old* stock count. The user, who just bought the item, sees it as still available. Only at T+250ms does the event reach the inventory projection, and a re-read at T+260ms finally agrees. The window between T+5ms and T+250ms is the booby trap, and a naive UI walks straight into it on essentially every purchase.

Why is this so easy to get wrong? Because in a monolith, read-after-write is *automatically* consistent — you wrote to the database and you read from the same database, so of course you see your write. Engineers carry that reflex into microservices and apply it across a service boundary where it is simply false. The reflex "I wrote it, so I can read it back" holds *within a service* and *breaks across services.* That sentence is worth memorizing. The fixes are exactly the read-your-writes mechanisms from earlier, applied at the boundary:

- **Carry the version token across the boundary.** When the order write returns a token, the subsequent inventory read carries it, and the inventory projection waits (bounded) until it has applied at least that order's event before answering. This makes the cross-service read-after-write behave like a within-service one, at the cost of a little latency on that specific read.
- **Read the truth you just wrote from the owner, not a projection.** Right after writing the order, if you must show order-related data, read it from the *order* service (the owner, fast path) rather than from a projection in another service. Read the slow projections only for data you did *not* just write.
- **Do not read across the boundary at all for the just-written fact.** The best fix is often to *not* re-fetch. The client already knows the order succeeded (it got `201`); it can optimistically update its own local state ("you reserved this; we're confirming stock") without round-tripping to a projection that cannot possibly be current yet. This is the optimistic-UI pattern in the next section.

The wrong "fix" I see juniors reach for is to make the inventory projection synchronous — have the order write *block* until inventory has applied the event before returning `201`. That re-couples the two services' availability (now inventory being slow makes ordering slow) and reintroduces the very distributed coordination you split them to avoid. Resist it. The right fixes keep the services decoupled and patch the *perception* at the read boundary, not the write path.

## Designing the UX around the lag

If the window is unavoidable, then the user experience must be honest about it, and good UX around eventual consistency is one of the clearest markers of an experienced team. There are three moves, and they compose.

**Name the in-between state.** The cardinal sin is showing a binary "done / not done" when the truth is "in progress." When ShopFast accepts an order before payment has run, the order page must not say "Confirmed" — it says **"Order received, processing payment."** When stock is reserved but the warehouse has not yet allocated a box, it says "Reserved." These intermediate states are not a UX afterthought; they are a *faithful rendering of the actual distributed state*, and they turn "the app is lying to me" into "the app is keeping me informed." The status vocabulary should map one-to-one onto the saga's states: `PROCESSING_PAYMENT`, `PAYMENT_FAILED`, `RESERVED`, `ALLOCATED`, `SHIPPED`. Each transition is driven by an event arriving, and the UI subscribes (via polling or websockets) to watch the state advance.

**Optimistic UI.** Because the client *knows* its own write succeeded (it got the `201` and a token), it can render the new state immediately and locally — show the cart item as "reserved," show the order in the list as "processing" — without waiting for any projection to catch up. The optimistic update is then *reconciled* against reality when the real state arrives: if payment ultimately fails, the optimistic "processing" flips to "payment failed, please update your card," and the reserved stock is released. The key discipline is that an optimistic update is a *prediction*, and you must handle the case where the prediction is wrong — silently, gracefully, and without losing the user's place. Optimistic UI done well feels instant; done without the reconciliation path, it feels like a lie when the prediction fails.

**Idempotent retries everywhere.** Because the network is unreliable, the client *will* retry — the user mashes the button, the mobile connection drops mid-request and the SDK resends, a proxy times out and replays. Every write the user can trigger must carry an **idempotency key** so that a retry of "place this order" produces the *same* order, not a second one. This is the direct fix for the second half of the intro's double-charge: even if the user clicks Buy twice, the same idempotency key means the second click returns the *existing* order rather than creating a new one.

```python
# Idempotent order creation: the client generates a key once and reuses it on
# every retry of THIS logical action. Same key -> same result, never a duplicate.
@app.post("/orders")
def place_order(req: OrderRequest, idempotency_key: str = Header()):
    existing = idem_store.get(idempotency_key)
    if existing is not None:
        return existing                      # retry: return the SAME prior result
    with db.transaction() as tx:
        order = tx.insert_order(req)
        result = OrderResponse(order_id=order.id, status="PROCESSING")
        # store result under the key, in the SAME transaction, so a crash between
        # commit and storing can't lose the mapping (unique constraint on the key)
        idem_store.put(idempotency_key, result, tx=tx)
    return result
```

This is a sketch; the deep treatment of doing it correctly across services — handling concurrent retries, expiry, and the difference between idempotent *requests* and idempotent *effects* — is the forward-linked sibling on [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services). The mechanism-level theory of why at-least-once delivery makes idempotency mandatory is in [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) and [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). The point to internalize here is that **idempotency is not optional in an eventually consistent system; it is the precondition that makes retries safe, and retries are the only way to make an unreliable network reliable.**

It is worth being precise about *where* the idempotency key lives, because a common half-fix protects the synchronous API but leaves the asynchronous event path duplicating. There are two distinct surfaces. The first is the *inbound HTTP write* the user triggers — protected by the idempotency-key header above, so a double-click or an SDK retry returns the existing order. The second is the *event consumer* on the other side of the bus, which receives every event at-least-once and will occasionally see the same `OrderPlaced` twice because the broker redelivered after a consumer crash before its offset was committed. That consumer must *also* be idempotent: applying `OrderPlaced` twice to the inventory projection must not decrement stock twice. The standard implementation is a processed-events table keyed by event id — the consumer checks "have I already applied event `evt-abc123`?" in the same transaction it applies the effect, and skips if so. Get only the HTTP side and you protect against the user double-clicking but not against the broker redelivering; get only the consumer side and you protect against redelivery but not against the user's retry creating two orders. You need both, on both surfaces, and the unifying principle is the same: a unique key plus a "have I seen this key?" check inside the same transaction as the effect. The consumer-side version, sketched:

```python
# Event consumer is idempotent too: at-least-once delivery WILL redeliver, so
# dedup on event id inside the SAME transaction that applies the effect.
def on_order_placed(evt):
    with db.transaction() as tx:
        if tx.exists("processed_events", event_id=evt.id):
            return                              # already applied: skip, ack the message
        tx.insert("processed_events", event_id=evt.id)   # unique constraint here
        tx.execute("UPDATE inventory_projection SET available = available - :n "
                   "WHERE sku = :sku", n=evt.qty, sku=evt.sku)
    # commit makes the effect and the dedup record atomic; a crash before commit
    # safely redelivers and re-applies exactly once
```

## Avoiding corruption: versioning and optimistic concurrency

Eventual consistency across services is about *staleness* — data that is correct but late. There is a sharper danger lurking *within* a service when concurrent writers race: **lost updates**, where two writers each read a value, each modify it, and each write it back, and the second silently clobbers the first. The defense is **optimistic concurrency control** with a version column.

```sql
-- Optimistic concurrency: every row carries a version. A writer reads version V,
-- and writes back ONLY if the row is still at V. If someone else wrote in between,
-- the version moved, rows_affected = 0, and the writer must re-read and retry.
UPDATE cart
SET    items   = $new_items,
       version = version + 1
WHERE  cart_id = $1
  AND  version = $expected_version;   -- the "nobody changed it under me" guard
-- rows_affected = 1 -> my write won, version advanced
-- rows_affected = 0 -> a concurrent write happened; re-read and retry (or merge)
```

This is "optimistic" because it assumes conflicts are rare and detects them rather than preventing them with a lock. The cost is a retry when a conflict *does* happen; the win is no lock held across the read-modify-write, so the common (no-conflict) case is fast and never blocks. The HTTP-native version of this is the `ETag` / `If-Match` header: the server returns an `ETag` (the version) on a `GET`, the client sends `If-Match: <etag>` on the `PUT`, and the server returns `412 Precondition Failed` if the version moved. This is the same optimistic-concurrency idea exposed at the API boundary, and it is how you give a *client* (including another service) the ability to do safe read-modify-write across the network without a distributed lock.

```python
# HTTP optimistic concurrency with ETag/If-Match across a service boundary.
@app.put("/profiles/{user_id}")
def update_profile(user_id: str, body: Profile, if_match: str = Header(alias="If-Match")):
    current = repo.get(user_id)
    if etag_of(current) != if_match:
        # someone changed the profile since the client last read it
        raise HTTPException(status_code=412, detail="Precondition Failed: re-fetch and retry")
    updated = repo.save(user_id, body, expected_version=if_match)
    return Response(updated, headers={"ETag": etag_of(updated)})
```

The senior nuance: optimistic concurrency protects you *within an aggregate* from lost updates. It does **not** solve cross-service consistency — two different services writing two different databases will never see each other's versions. Versioning is the *intra-service* corruption defense; the *inter-service* defenses are conflict resolution and reconciliation, which come next. Keep the two problems separate in your head: *staleness across services* (handled by session guarantees and reconciliation) versus *lost updates within a service* (handled by optimistic concurrency).

## Conflict resolution: LWW vs version vectors vs business merge

When the same logical entity can be written in more than one place — a cart edited on a phone and a laptop, a profile synced across regions, a counter incremented by several services — concurrent writes can *conflict*, and the system must decide which write wins or how to combine them. There are three strategies, and choosing wrong either silently loses data or buries you in complexity.

![A matrix comparing last-writer-wins, version-vector or CRDT, and business-merge conflict resolution strategies across correctness, complexity, whether they lose updates, and when to use each](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-6.webp)

**Last-writer-wins (LWW)** keeps whichever write has the highest timestamp and discards the other. It is trivial to implement — just compare timestamps — and that simplicity is its entire appeal. Its fatal flaw, which the matrix marks in red, is that it **silently loses updates**: if two users both edit the cart concurrently, one edit vanishes with no trace and no error. For data where concurrent edits are genuinely independent and only the latest matters — a "last seen" timestamp, a presence flag, an idempotent "mark as read" — LWW is fine and you should use it because it is free. For anything where a lost update is a lost *intention* (a removed cart item reappearing, a renamed file reverting), LWW is a quiet data-corruption machine. There is a subtler trap: LWW depends on clock comparison, and clocks across machines skew, so "last" can mean "the one whose server happened to have a faster clock," which is not the same as "the one that actually happened later." Treat LWW as a last resort, not a default.

**Version vectors and CRDTs** make conflicts *detectable* or *automatically convergent* without losing data. A version vector tags each write with per-replica counters so the system can tell whether two writes were concurrent (a real conflict) or one strictly followed the other (no conflict). A **CRDT** (Conflict-free Replicated Data Type) goes further: it defines a *merge function* that is commutative, associative, and idempotent, so any two replicas that have seen the same set of operations converge to the same state *regardless of the order they arrived in.* The classic examples fit microservices data perfectly: a **G-Counter** (grow-only counter) for things like view counts, where each replica counts its own increments and the merged value is the sum — so two services incrementing concurrently both count, and neither is lost. An **OR-Set** (observed-remove set) for a shopping cart, where adds and removes both carry tags so a concurrent add-and-remove resolves deterministically instead of one clobbering the other.

```python
# A grow-only CRDT counter (G-Counter): merge is just the per-replica max, so
# concurrent increments from different services BOTH survive a merge. No locks,
# no lost updates, converges no matter what order updates arrive in.
class GCounter:
    def __init__(self): self.counts = {}            # replica_id -> count
    def incr(self, replica_id, n=1):
        self.counts[replica_id] = self.counts.get(replica_id, 0) + n
    def value(self):
        return sum(self.counts.values())            # the merged truth
    def merge(self, other):                          # commutative + idempotent
        for rid, c in other.counts.items():
            self.counts[rid] = max(self.counts.get(rid, 0), c)
```

The cost of CRDTs is real: they carry metadata (the per-replica vectors and tombstones) that grows, the merge logic is non-trivial, and not every data shape has a clean CRDT. Use them where concurrent multi-writer convergence genuinely matters — Amazon's cart, collaborative editing, multi-region counters — and not as a reflex.

**Business merge** is the strongest and most expensive: a domain-specific resolution function that *understands the data*. When two replicas of an inventory count conflict, you do not pick one by timestamp and you do not blindly sum — you re-derive the truth from the authoritative event history, because for stock and money, *only the real sequence of reservations and shipments is correct.* Business merge for money looks like: never resolve a balance conflict automatically at all; instead, keep the ledger of debits and credits as the source of truth (each entry idempotent and ordered) and *recompute* the balance, so there is no "conflict" to resolve because the balance was never the authoritative thing — the immutable, ordered ledger is. This is why banks keep the ledger in one service (the case study below): it sidesteps conflict resolution entirely by making the authoritative data an append-only, ordered log rather than a mutable value two writers can race on.

The decision rule from the matrix: **use the cheapest strategy whose correctness matches the data.** LWW for idempotent flags where a lost update is harmless. CRDT/version-vector for genuinely concurrent multi-writer data like carts and counters where you must not lose updates but can tolerate automatic merge. Business merge — or, better, an ordered authoritative log that removes the conflict — for money and stock where only the real history is correct and a wrong merge corrupts the business.

One more practitioner warning about LWW, because its trap is so seductive. Many datastores default to LWW *silently*. Cassandra resolves every cell conflict by the write timestamp; DynamoDB's default put overwrites; a "just update the row" pattern in any database is LWW by another name. So you can be using last-writer-wins *without ever having chosen it*, and the data loss it causes is invisible — there is no error, no log line, no metric, just an edit that quietly never happened. The way you catch this in review is to ask, for any data written from more than one place: "if two of these writes race, is silently dropping one of them acceptable?" If the answer is no and the storage default is LWW, you have a latent corruption bug that will surface as a baffling "the user swears they changed it" support ticket months later. The fix is to *make the conflict visible* — a version column that forces a 412 and a retry, or a CRDT that merges instead of drops — rather than relying on a timestamp comparison whose loser disappears. The senior habit is to treat LWW as something you *opt into explicitly for data where loss is genuinely fine*, never as a default you backed into.

## Reconciliation and convergence: detecting and repairing divergence

Everything so far reduces the *frequency* of inconsistency. None of it makes inconsistency *impossible*, because in a real system events get dropped, consumers have bugs, a deploy skips a migration, a poison message gets dead-lettered and forgotten, and a network blip swallows a publish. Over weeks, two services *will* drift, quietly, and the only question is whether you *find out* before a customer does. The answer is **reconciliation**: a background process that compares the authoritative source against its projections and repairs the differences. A system without reconciliation is not eventually consistent; it is *eventually wrong, undetectably.*

![A decision tree for placing the strong-consistency boundary, branching from an invariant to enforce into a hard data-corrupting rule that goes into one aggregate or a saga across services, versus a soft stale-UI concern that gets an eventual read model with a UX cue and nightly reconciliation](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-8.webp)

The tree above is the placement decision for *every* piece of data, and notice that the soft branch — the eventual read models — explicitly ends in *reconciliation*. Eventual consistency and reconciliation are a package; if you have one without the other you have a bug factory. There are three layers of reconciliation, increasingly cheap to run continuously.

**Periodic full reconciliation** walks the authoritative data and the projection and compares them row by row, repairing any difference (the owner always wins, because it is by definition authoritative). It is thorough but expensive, so it runs nightly during low traffic.

```python
# Nightly reconciliation: owner is the source of truth; repair drifted projection rows.
def reconcile_inventory(since: datetime):
    drift_count = 0
    for sku, owner_qty in inventory_owner.iter_quantities(updated_since=since):
        proj_qty = catalog_projection.get_quantity(sku)
        if proj_qty != owner_qty:
            drift_count += 1
            log.warning("DRIFT sku=%s owner=%d projection=%d", sku, owner_qty, proj_qty)
            catalog_projection.set_quantity(sku, owner_qty)   # owner wins, repair
            metrics.incr("reconcile.repaired", tags={"entity": "inventory"})
    metrics.gauge("reconcile.drift_rows", drift_count)        # ALERT if this trends up
    return drift_count
```

**Checksum / anti-entropy reconciliation** avoids comparing every row by comparing *hashes* of ranges. Both sides compute a checksum over a key range (or a Merkle tree of ranges); if the checksums match, that whole range is in sync and you skip it; only when they differ do you drill down. This is how systems like Cassandra and DynamoDB run anti-entropy cheaply across huge datasets — you exchange a tree of hashes, not the data itself, and only repair the leaves that differ. The same trick scales reconciliation between two services: hash the inventory state per SKU-prefix on both sides, compare the trees, and only fetch and repair the SKUs under a differing hash.

**Continuous reconciliation via a consistency check stream** runs lightweight checks in near-real-time — for example, a job that, for every order created in the last minute, verifies the corresponding inventory reservation exists, and flags any order with no matching reservation within a grace window. This catches the dropped-event class fast, before it accumulates into the nightly job's haystack.

#### Worked example: a reconciliation job catching real drift

Let me put numbers on what reconciliation actually catches, because juniors imagine it is paranoia for an event that "can't happen." At ShopFast's scale — 1,200 orders/second, ~100M inventory-affecting events per day across all SKUs — the broker's at-least-once delivery and the consumers' occasional bugs produce a small but nonzero drop/double-apply rate. Suppose the *effective* divergence rate after dedup and retries is one part in three million — that sounds vanishingly rare. At 100M events/day that is still ~**33 drifted SKU rows per day**. Without reconciliation, those 33 accumulate: after a month the catalog's stock counts are wrong on ~1,000 SKUs, each off by some small amount, and you find out when customers complain or when an oversell finally bites. With the nightly job, you catch and repair all 33 every night, you emit `reconcile.drift_rows` as a metric, and — this is the real payoff — you *watch the trend.* A healthy system holds steady around 33/day; the morning it jumps to 4,000 is the morning you *know*, before any customer does, that a deploy broke the inventory consumer or a topic's events stopped flowing. Reconciliation is not just a repair tool; it is your **divergence smoke detector**, and the alert on its trend is one of the most valuable signals in an eventually consistent system.

## The optimization angle: where to put the strong boundary to minimize coordination

The performance story of eventual consistency is, perhaps counterintuitively, mostly a *placement* story, not a tuning story. The expensive thing in any consistency design is **coordination** — round-trips, locks, quorum reads, distributed transactions — and coordination cost is determined almost entirely by *where you draw the strong-consistency boundary.* Draw it too wide (try to keep many services mutually consistent in real time) and every request pays coordination latency and the system's availability collapses to the weakest link. Draw it correctly (one tight aggregate per hard invariant, eventual everywhere else) and coordination becomes a single local transaction that costs microseconds, with everything else running fully parallel and available.

Here is the math that should drive the decision. Suppose checkout touches order, inventory, payment, and shipping. If you keep all four strongly consistent with a distributed transaction, the checkout latency is *at least* the sum of the slowest path through all four plus 2PC's prepare-and-commit round-trips — call it p99 of 4 × 60ms + 2 coordination round-trips ≈ **300ms+**, and availability is `0.999^4 ≈ 99.6%`, meaning ~3.5 hours of checkout downtime a year just from the multiplication, *plus* every participant must be up simultaneously. If instead you keep only the inventory *reservation* strong (one local `UPDATE`, ~5ms) and run payment and shipping as eventual saga steps, the *synchronous* path the user waits on is order-create + reservation ≈ **25ms p99**, and availability of that synchronous path is `0.999^2 ≈ 99.8%` — and payment/shipping being briefly down no longer fails the checkout, it just extends the "processing" window. You moved from 300ms to 25ms and from 99.6% to 99.8% on the user-facing path **purely by narrowing the strong boundary from four services to one aggregate.** That is the optimization. There is rarely a faster lever in microservices than *removing* coordination by relocating an invariant into a single owner.

The measurable wins to track when you do this: the **p99 of the synchronous user-facing path** (should drop sharply as you pull coordination out of it), the **per-projection lag** (your eventual-consistency window — keep a budget and alert), the **reconciliation drift rate** (your correctness smoke detector), and the **conflict/retry rate** on optimistic-concurrency writes (if it climbs, your aggregate is too hot and needs sharding). A team that watches these four numbers is *operating* eventual consistency; a team that watches none of them is *hoping.*

## Stress-testing the design

A design is only as good as how it behaves when things go wrong, so let me stress-test the ShopFast design against the three failures that actually happen.

**"The user sees stale data right after writing."** This is the read-after-write-across-services gotcha, and we have already armed against it: the order write returns a version token, the immediate re-read carries it, the inventory projection waits (bounded 300ms) to apply at least that event or the read falls back to the owner. Stress it harder: what if the projection is *minutes* behind because the inventory consumer is wedged? Then the bounded wait times out, the read falls back to the owner (correct, slightly slower), the per-projection lag metric blows past its budget, and an alert fires. The user is never shown stale data for the thing they just wrote; the worst case is a slightly slower read and a paged engineer. Crucially, the failure is *visible and bounded*, not silent.

**"Two services disagree on a balance."** Suppose the wallet service says a customer's balance is \$40 and a cached projection in the rewards service says \$50. If we had stored the balance as a mutable value resolved by LWW, we would have no idea which is right and might "resolve" to the wrong one. Because we kept the *ledger* (ordered, immutable debit/credit entries) as the authoritative source in *one* service and the \$50 is merely a projection, the resolution is unambiguous: recompute the balance from the ledger, get the true number, and repair the projection. The disagreement is resolvable precisely because we refused to make the balance the authoritative thing in the first place. Stress it harder: what if the ledger itself got a duplicate credit from an at-least-once event? Then idempotency keys on ledger entries mean the duplicate is rejected on insert (unique constraint on the event id), so the ledger never double-counts, and the recomputed balance is still correct. The design degrades to "correct but possibly briefly stale in the projection," never to "wrong in the authoritative ledger."

**"An event is lost — divergence."** This is the deadliest because it is silent. A `StockReserved` event is dropped during a deploy; the inventory owner shows 41 units, the catalog projection still shows 42. Nothing errors. No alert fires from the normal monitoring because every individual service is healthy.

![A timeline of six events showing a StockReserved event dropped on day one, the owner at forty-one disagreeing with the projection at forty-two, the drift going unnoticed all day while the UI overcounts, the reconciliation job running at three in the morning comparing checksums, thirty-seven rows flagged as different, and the owner treated as truth so the rows are repaired](/imgs/blogs/data-consistency-and-eventual-consistency-in-practice-9.webp)

This is exactly what reconciliation exists for, and the timeline shows the only thing standing between a dropped event and a customer-facing oversell: the nightly job runs at 03:00, compares checksums between owner and projection, finds 37 SKU rows that differ (this one plus 36 others from the day's normal drift), treats the owner as truth, and repairs them. The divergence existed for up to ~13 hours, during which the catalog *overcounted* stock — which is the *safe* direction of error here, because showing more stock than exists only risks a reservation failure at checkout (handled cleanly by the `UPDATE ... WHERE available >= 1`), not an oversell. Had the projection *undercounted*, you would lose sales but never corrupt data. The reconciliation job converts "eventually wrong forever" into "wrong for at most a day, then repaired, and trended on a dashboard." Stress it harder: what if a *whole topic* stops flowing, not one event? Then the drift count spikes from ~33 to thousands overnight, the `reconcile.drift_rows` alert fires, and you investigate the broken consumer the same morning. The reconciliation metric is the backstop that catches the failure mode your per-service health checks structurally cannot see.

## Case studies

**Amazon's shopping cart and eventual consistency.** The original Dynamo paper (the ancestor of DynamoDB) made the shopping cart the canonical example of *embracing* eventual consistency. Amazon's overriding requirement was that the "Add to Cart" operation must *never* be rejected — an unavailable cart is lost revenue — so they chose an "always writable" design where the cart accepts writes even during partitions and resolves conflicts afterward. The consequence is that concurrent cart edits from two devices can conflict, and Dynamo resolved them with *version vectors* (vector clocks) plus an application-level merge: rather than lose an item via last-writer-wins, the system would *union* the carts so a deleted item might occasionally reappear — a deliberate, well-understood trade where the worst case is a resurrected item the user can re-delete, never a failed write or a lost sale. The lesson is the cleanest possible statement of this whole post: Amazon picked *availability and convergence* for the cart because for *that* data, a rejected write costs more than a rare merge artifact — and they did *not* make the same choice for the order or the payment, which sit behind stronger guarantees. The boundary placement is the design.

**A bank keeping the ledger in one service.** Modern banks built on microservices (Monzo and others have written publicly about ledger design) almost universally keep the *ledger* — the immutable, ordered, double-entry record of every money movement — inside a single service with a strong-consistency boundary, and treat everything else (the balance shown in the app, the spending categories, the notifications) as eventual projections of that ledger. Money is the textbook case where you must *not* spread the invariant: a balance is never resolved by conflict-merging two replicas, because the only correct balance is the one implied by the true ordered sequence of credits and debits. By making the ledger append-only and single-owned, the bank turns "balance consistency" — which would be a nightmare of distributed transactions — into "append a row, idempotently, in one service," and the displayed balance is just a fast projection that reconciliation keeps honest. The lesson: when the data is money, do not look for a clever conflict-resolution strategy; *eliminate the conflict* by making the authoritative truth an ordered log in one place.

**The flash-sale oversell incident.** The oversell pattern from the worked example is a recurring real-world incident across e-commerce, especially during flash sales and limited drops, and the post-mortems rhyme: the system checked an *eventually consistent* cached stock count before accepting orders, a burst of concurrent buyers all read the same pre-decrement value, and the site accepted far more orders than units existed — sometimes by multiples — forcing mass cancellations and reputational damage. The fix in every mature write-up is the same one in this post: move the "never oversell" invariant out of the cached count and into an *atomic reservation* in the single inventory owner (`UPDATE ... WHERE available >= 1`), and treat the catalog's displayed count as a lagging projection that is *allowed* to be optimistic because it no longer enforces anything. The lesson: never enforce a hard invariant against an eventually consistent copy; enforce it against the strongly consistent owner, and let the copies be wrong-but-harmless.

## When to reach for eventual consistency (and when not to)

Be decisive about this, because "everything eventual" and "everything strong" are both wrong.

**Reach for eventual consistency across services** whenever the data is a *copy, projection, cache, search index, feed, count, or notification* — anything that is a *view* of authoritative data owned elsewhere. Reach for it whenever availability matters more than instantaneous agreement, which is most user-facing reads. Reach for it whenever the cost of staleness is "a number is briefly out of date" rather than "money or stock is corrupted." This is the *default* in microservices, and you should be comfortable defending it, not apologizing for it.

**Keep it strong (within one aggregate)** for every *hard invariant* — the rules whose violation corrupts data or loses money: stock count, account balance, the order's own state machine, a seat that cannot be double-booked, a unique-username constraint. The test is the tree from earlier: *does breaking this invariant corrupt data or lose money, or does it just make a UI briefly stale?* If the former, it lives in one strongly consistent aggregate, full stop. If you find a hard invariant that genuinely spans services, your *first* instinct should be to question the service boundaries — often a hard invariant spanning two services is a signal that those two services should be one, because [you should not spread a single invariant across a boundary](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) (this is the seed of the distributed-monolith anti-pattern). Only if the boundary is genuinely correct do you reach for a saga to enforce the invariant eventually with compensation.

**Do not reach for distributed transactions / 2PC across services** as a way to "just make it consistent." It is almost always the wrong tool: it couples availability, adds latency to every request, and rarely survives a partition gracefully. The cases where it is justified are narrow and specialized. When you feel the pull toward 2PC, the lever to reach for instead is *relocating the invariant into one owner* so the consistency you need becomes local.

## Key takeaways

1. **Eventual consistency is not opt-in.** Database-per-service plus async events makes it the default. Your only choice is to design for it deliberately or get ambushed by it in production.
2. **Keep the invariant strong inside one aggregate; let everything across services be eventual.** This single sentence resolves most consistency design questions. The dangerous part lives in one cheap local transaction; its shadows lag harmlessly.
3. **Users want session consistency, not global consistency.** Provide read-your-writes, monotonic reads, monotonic writes, and consistent prefix with version tokens and sticky sessions — cheap, local guarantees over an eventual substrate, no global coordination required.
4. **The read-after-write-across-services gotcha is the most common bug.** "I wrote it, so I can read it" holds within a service and *breaks* across services. Carry a version token across the boundary, read the owner for the just-written fact, or don't re-read at all.
5. **Enforce hard invariants against the strong owner, never against an eventual copy.** Oversells and double-spends come from checking a lagging cache. `UPDATE ... WHERE available >= 1` in the one owner is the whole fix.
6. **Idempotency and optimistic concurrency are non-negotiable.** Idempotency keys make retries safe (and retries are how you survive an unreliable network); version columns / ETags stop lost updates within an aggregate.
7. **Pick the cheapest conflict-resolution strategy that matches the data.** LWW for harmless idempotent flags, CRDTs/version-vectors for genuinely concurrent multi-writer data, an ordered authoritative log for money and stock — and prefer *eliminating* the conflict (append-only ledger) over resolving it.
8. **Reconciliation is mandatory, not paranoia.** Events get dropped; services drift. A nightly checksum/anti-entropy job that repairs from the owner and *trends the drift count* is your divergence smoke detector — the only thing that catches the silent failures per-service health checks structurally cannot.
9. **Optimization = narrowing the strong boundary.** The fastest lever in microservices is removing coordination by relocating an invariant into a single owner. Watch four numbers: synchronous p99, per-projection lag, reconciliation drift rate, optimistic-concurrency retry rate.
10. **Design the UX honestly.** Name the in-between states ("processing," "reserved"), update optimistically and reconcile when reality arrives, and make every user-triggered write idempotent. The window is real; render it truthfully.

## Further reading

- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the formal hierarchy of consistency guarantees this post builds on.
- [The CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — *why* the consistency/availability/latency trade is fundamental, not an implementation gap.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — how replicas lag and the read/write quorum mechanics behind tunable consistency.
- [Database-per-service: the rule that defines microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) — the rule that *creates* eventual consistency and where invariant boundaries should fall.
- [The saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice) — how to enforce an invariant that genuinely spans services, eventually, with compensation.
- [Event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — the coordination style that produces the propagation topology you reason about here.
- [Idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services) — doing idempotency correctly across the network, the precondition for safe retries.
- [Caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services) — the most common eventual-consistency surface, and how to keep caches honest.
- Sam Newman, *Building Microservices* (2nd ed.) — the chapters on data and transactions.
- Chris Richardson, *Microservices Patterns* — the saga and CQRS patterns in depth.
- Martin Kleppmann, *Designing Data-Intensive Applications* — the definitive treatment of consistency, replication, and conflict resolution.
- DeCandia et al., *Dynamo: Amazon's Highly Available Key-value Store* (2007) — the shopping-cart eventual-consistency case study, primary source.

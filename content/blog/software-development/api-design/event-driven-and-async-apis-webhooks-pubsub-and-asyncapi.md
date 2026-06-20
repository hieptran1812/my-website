---
title: "Event-Driven and Async APIs: Webhooks, Pub/Sub, and AsyncAPI"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The push side of API design — when to make the server tell the client instead of the client polling: webhook event design, at-least-once delivery and idempotent consumers, retries with backoff and dead-letters, HMAC signature verification with a worked snippet, pub/sub and event streaming for internal scale, and AsyncAPI plus CloudEvents as the contract for events."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "webhooks",
    "events",
    "asyncapi",
    "pubsub",
    "cloudevents",
    "idempotency",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-1.png"
---

A customer pays. Somewhere in your platform, `payment.succeeded` becomes true. Now seven other things have to happen: the order moves to `paid`, the ledger records the credit, a receipt email goes out, the fraud system updates its model, the analytics warehouse gets a row, the partner who referred the sale gets their commission queued, and a Slack channel in the merchant's workspace lights up. None of those seven systems made the request that caused the payment. None of them are sitting in the HTTP call that the customer's browser opened. So how do they find out?

The lazy answer is: they ask, over and over. Every integration polls `GET /payments/{id}` on a timer, or sweeps `GET /payments?status=succeeded&since=...` every few seconds, hoping to catch the change. I have run platforms where this was the only mechanism, and the numbers are brutal. If a partner polls every five seconds and a payment succeeds on average once an hour, then **3,599 out of every 3,600 polls return nothing** — and the one that matters still arrives up to five seconds late. Multiply that by a few thousand partners and you are paying, in compute and rate-limit budget, for an enormous volume of calls whose only job is to discover that nothing happened. The thing that should have been a single notification became a standing DDoS that you built against yourself.

The push answer flips it. Instead of the client asking "anything new?", the **server tells the client the moment something happens**. The client registers a URL; when `payment.succeeded` fires, your platform does an HTTP `POST` to that URL carrying the event. One call, on the event, near-zero waste, near-zero latency. That is a **webhook**, and it is the most common form of an event-driven API on the public internet. The rest of this post is about getting it right — because a webhook that is naive about delivery, ordering, replay, and signatures is not a feature, it is an incident waiting for a quiet Saturday.

![a two-column comparison contrasting a client polling on a timer with most calls returning no change against a server pushing a single webhook on the event](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-1.png)

By the end you will be able to design a production webhook contract end to end: an event envelope built to be deduplicated and versioned; at-least-once delivery handled with idempotent consumers; retries with exponential backoff and a dead-letter after exhaustion; HMAC signatures with a timestamp to stop replay; a 2xx-fast-or-we-retry consumer contract; replay and backfill endpoints; and a webhook management API. Then we will zoom out to **pub/sub and event streaming** for the internal, many-to-many case, and to **AsyncAPI** and **CloudEvents** as the way to write the contract down — the OpenAPI of the event world. This is the push half of the series spine: the same question of *what does the caller get to assume, and can I change this later without breaking them?*, asked of events instead of requests.

This post is the events chapter of [the API design series](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); it composes directly with [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) (the consumer side of at-least-once), with [long-running operations and async jobs](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks) (the other half of "the work outlives the request"), and with [streaming APIs over SSE and WebSocket](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming) (the open-connection alternative to discrete events).

## 1. Pull versus push: the cost of asking versus the cost of telling

Start with the principle, because the rest of the design falls out of it. There are exactly two ways for a client to learn that server-side state changed: the client **pulls** (polls), or the server **pushes** (calls the client). Every event mechanism — polling loops, webhooks, pub/sub, SSE, WebSocket, long-poll — is a point on that one axis.

A **safe method** is one with no side effects the caller is responsible for; `GET` is the safe method, and polling is just `GET` in a loop. That is why polling feels free to reach for — it uses the read path you already built. But "free" is an illusion at scale. Let me make the cost concrete.

### The polling math

Say a resource changes with mean inter-arrival time $T$ seconds (one payment per hour means $T = 3600$), and a client polls every $p$ seconds. Then:

- **Wasted calls per real event** $\approx T / p$. At $T = 3600$, $p = 5$, that is about **720 polls per actual change** — 719 of which return "no change".
- **Mean detection latency** $\approx p / 2$, with a worst case of $p$. Poll every five seconds and the client is, on average, 2.5 seconds behind reality, and at worst a full 5 seconds.

You cannot win both. Poll faster to cut latency and you multiply wasted calls (and your rate-limit pressure on the client); poll slower to cut calls and you add latency. Push escapes the trade-off entirely: it spends **one call per event** and delivers with the latency of a single network round-trip, because the call *is* the event.

There is a real place for pulling, and I want to be honest about it. Polling is **stateless, trivially resumable, and dead simple to operate**: there is no endpoint to host, no signature to verify, no retry queue, and if the client is down it simply catches up on its next poll. For low-frequency, low-fan-out, or "I just need the final state of one job" cases, a poll with a sensible `Retry-After` is often the right call — that is exactly the pattern in [long-running operations](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks). Push earns its complexity when events are **frequent enough that polling wastes real money**, when **latency matters**, or when **many independent consumers** need the same event.

| Property | Polling (pull) | Webhooks (push) | Streaming (push, persistent) |
| --- | --- | --- | --- |
| Latency to consumer | up to one poll interval | one network round-trip | sub-second, continuous |
| Cost at scale | wasted calls dominate | one call per event | open-connection cost |
| Who hosts an endpoint | nobody (client calls you) | consumer hosts a URL | consumer holds a socket |
| Reliability story | resume on next poll | at-least-once + retries | needs resume/replay token |
| Operational complexity | low | medium (retries, signing, DLQ) | high (backpressure, sticky conns) |
| Best fit | final-state checks, low rate | discrete domain events | high-volume continuous feeds |

We will go deep on webhooks first because they are the workhorse, then bring in streaming and pub/sub where their forces dominate.

### A consequence story: the poll storm that took out a read replica

Let me make the polling cost visceral with a story I have lived. A platform exposed `GET /payments?status=succeeded&since={cursor}` and told partners "poll it for new payments." Partners did exactly that — and because the docs never specified a *minimum* interval, the most enthusiastic ones polled every second. Each poll ran a range query against the payments table. With a few thousand partners polling at one to five seconds each, the read path was carrying on the order of a thousand queries per second whose answer, almost always, was an empty list. None of those queries did any useful work; they existed only to discover that nothing had changed since the last poll.

The failure mode was not the steady state — it was the recovery. One afternoon a brief blip made polls time out for thirty seconds. Every partner's client did the natural thing: it retried immediately, and then kept polling on its normal timer *plus* catching up on the polls it had missed. The instant the database recovered, it was hit by the entire fleet's backlog at once — a synchronized stampede of range queries that pushed the read replica's CPU to the ceiling and made *real* reads (the customer-facing ones) slow. A thirty-second blip became a twenty-minute degradation, entirely manufactured by the polling design. The fix was not a bigger replica. The fix was to **stop making partners poll** — to push a `payment.succeeded` webhook on the event, so the steady-state query load for change-detection dropped to roughly zero and there was no backlog to stampede with. That is the kind of consequence the rest of this post is designed to prevent: a design that is fine at one partner and a self-inflicted outage at a thousand.

We will go deep on webhooks first because they are the workhorse, then bring in streaming and pub/sub where their forces dominate.

## 2. The webhook, precisely: the server POSTs to a URL you registered

A **webhook** is a user-defined HTTP callback. The consumer (the partner integrating with your platform) **registers a URL** with you ahead of time — usually through a dashboard or a management API. When a relevant event occurs, your platform makes an HTTP `POST` to that URL with the event in the body. The consumer's job is to receive it, do its work, and return a `2xx` status to acknowledge it. That is the whole shape; everything else in this post is making that shape survive the real world.

Here is the wire, using our Payments & Orders platform. A payment just succeeded, and the merchant registered `https://merchant.example.com/hooks/payments` as their endpoint:

```http
POST /hooks/payments HTTP/1.1
Host: merchant.example.com
Content-Type: application/json
User-Agent: CommercePlatform-Webhooks/1.2
Webhook-Id: evt_1Nf8a2c0XyZ
Webhook-Timestamp: 1781049600
Webhook-Signature: v1,3a9f0b2c1d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8

{
  "id": "evt_1Nf8a2c0XyZ",
  "type": "payment.succeeded",
  "created": "2026-06-20T14:40:00Z",
  "api_version": "2026-06-20",
  "data": {
    "id": "pay_9KqR2mN",
    "object": "payment",
    "amount": 4999,
    "currency": "usd",
    "status": "succeeded",
    "order_id": "ord_77Bc1Q",
    "customer_id": "cus_3FdE9"
  }
}
```

The consumer replies, fast:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"received": true}
```

Note `amount: 4999` — money on the wire is in the smallest currency unit (cents), so this is a \$49.99 charge. Representing money as an integer minor unit, not a float, is a hill I will die on, but that is the [response-body design](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming) post's fight.

Two things in that request matter more than they look. The `Webhook-Id` is the **stable event id** that lets the consumer deduplicate — we will lean on it hard in §4. The `Webhook-Signature` is the **HMAC** that lets the consumer prove the request really came from you and was not forged or replayed — the security crux, §6. The exact header names vary by platform (Stripe uses `Stripe-Signature`; the [Standard Webhooks](https://www.standardwebhooks.com/) spec proposes `webhook-id`, `webhook-timestamp`, `webhook-signature`), but the *concepts* are universal.

Notice what the webhook is *not*. It is not a request the consumer initiated, so the consumer cannot rely on any of the usual client-side guarantees: there is no session it opened, no correlation to a call it is waiting on, no chance to negotiate the format with an `Accept` header up front. The webhook arrives unbidden, from a server the consumer must authenticate by signature alone, carrying a body the consumer must have agreed to in advance. That inversion of control is the whole reason webhooks need careful design: every guarantee that a normal request gives you for free — identity, format, ordering relative to your own calls, exactly-once-ish semantics within a transaction — you now have to engineer back into the contract explicitly. The header set above is the start of that re-engineering: an id for dedup, a timestamp and signature for trust.

There is also a reachability constraint that surprises people the first time. Because *you* call *them*, the consumer's URL must be publicly reachable from your network — which means a service running only on `localhost`, or behind a corporate firewall, or inside a private VPC, cannot receive a webhook without a tunnel or a public ingress. During development, integrators typically use a tunneling tool (the kind that exposes a local port at a public HTTPS URL) so a webhook from your platform reaches their laptop. This is the mirror image of polling's operational simplicity: polling needs no inbound reachability at all, which is exactly why it stays the right choice for consumers who cannot or will not host a public endpoint.

## 3. Designing the event payload: id, type, created, version, and the thin/fat choice

The event body is a contract, and like every contract in this series, you will be living with it for years. Design the **envelope** — the fields that wrap every event regardless of type — once, deliberately, and never touch them lightly. A good envelope is self-describing: a consumer can route, deduplicate, version, and decode any event using only the envelope, without special-casing each `type`.

![a vertical stack of the event envelope fields showing id as the dedup key, type, created timestamp, api version, the data block, and an optional sequence number](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-2.png)

The five fields every event should carry:

- **`id`** — a globally unique, immutable identifier for *this delivery's event*. This is the consumer's deduplication key. It must be stable: if you retry the same event, the `id` does not change. Prefix it (`evt_...`) so it is recognizable in logs.
- **`type`** — a dotted, namespaced string: `payment.succeeded`, `payment.failed`, `refund.created`, `order.fulfilled`. The namespace (`payment`, `order`) lets consumers subscribe to a family; the verb in past tense (`succeeded`, not `succeed`) signals that this is a **fact that already happened**, not a command. Events are facts; commands are requests. Never name an event imperatively.
- **`created`** — when the event occurred, as an RFC 3339 / ISO 8601 timestamp in UTC. This is event time, not delivery time; a retried event keeps its original `created`.
- **`api_version`** — the schema version of the `data` block. This is what lets you evolve the payload without breaking pinned consumers, exactly mirroring the request-side [versioning strategies](/blog/software-development/api-design/versioning-strategies-uri-header-media-type-and-not-versioning). A consumer that registered when the version was `2026-01-01` should keep receiving the shape it understood, even after you ship a newer one.
- **`data`** — the typed payload for this `type`. Its shape depends on the event, but the envelope around it does not.

### Naming events: facts in the past tense, namespaced by resource

The `type` string is a long-lived part of the contract — consumers subscribe to it, route on it, and write code branching on it — so name it with the same care you would a URI. Three rules earn their keep:

1. **Name the fact, not the command.** `payment.succeeded`, not `process.payment` or `payment.succeed`. An event announces something that *already happened*; the consumer is free to react or ignore it. Naming it imperatively (`charge.customer`) invites consumers to treat the event as an instruction, which couples them to your intent rather than to a fact — and the moment two consumers want to react differently, the command framing breaks.
2. **Namespace by the resource, then the change.** `resource.change` — `payment.succeeded`, `payment.failed`, `refund.created`, `order.fulfilled`, `order.canceled`. The resource prefix lets a consumer subscribe to a whole family (`payment.*`) and lets you add new event types under an existing resource without inventing a new naming scheme. Keep the segment count consistent; do not have `payment.succeeded` and `order.line_item.added` living at different depths unless you have a real reason.
3. **Make each type granular and single-purpose.** Prefer `payment.succeeded` and `payment.failed` over a single `payment.updated` with a status field the consumer must inspect. Granular types let consumers subscribe to exactly what they care about (the fraud team wants `payment.failed`, the fulfillment team wants `payment.succeeded`), and they keep the payload's meaning unambiguous. A catch-all `updated` event forces every consumer to receive everything and filter, which is the polling waste problem reincarnated inside your event types.

The cost of getting names wrong is the cost of any contract break: once partners have written `if event.type == "payment.succeeded"` against your name, you cannot rename it without breaking them. Adding a *new* type is non-breaking (consumers ignore types they do not subscribe to); renaming or removing one is breaking. So treat the set of event types as an append-mostly registry, deprecate types the same way you deprecate endpoints, and document each type's `data` schema in your AsyncAPI document (§11) so the contract is written down rather than discovered.

### Thin events versus fat events

Now the genuinely interesting design fork: how much do you put in `data`?

A **thin event** (also called *id-only* or *notification*) carries just enough to identify what changed — typically the event `id`, `type`, and a resource id — and the consumer then calls back to your API to fetch the current state:

```json
{
  "id": "evt_1Nf8a2c0XyZ",
  "type": "payment.succeeded",
  "created": "2026-06-20T14:40:00Z",
  "api_version": "2026-06-20",
  "data": { "id": "pay_9KqR2mN", "object": "payment" }
}
```

A **fat event** (also called *event-carried state transfer*) carries the full resource state in `data`, so the consumer needs no callback:

```json
{
  "id": "evt_1Nf8a2c0XyZ",
  "type": "payment.succeeded",
  "created": "2026-06-20T14:40:00Z",
  "api_version": "2026-06-20",
  "data": {
    "id": "pay_9KqR2mN", "object": "payment", "amount": 4999,
    "currency": "usd", "status": "succeeded", "order_id": "ord_77Bc1Q",
    "customer_id": "cus_3FdE9", "captured_at": "2026-06-20T14:40:00Z"
  }
}
```

![a two-column comparison of a thin id-only event that forces a fetch but stays fresh against a fat full-payload event that skips the fetch but may be stale](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-7.png)

The trade-off is real and goes both ways.

| Dimension | Thin event (id only) | Fat event (full state) |
| --- | --- | --- |
| Payload size | tiny, constant | grows with the resource |
| Extra round-trip | yes — consumer must `GET` the resource | none — state is in the event |
| Freshness | always current (fetch reads live state) | snapshot at emit time, can be stale on arrival |
| Auth coupling | consumer needs API credentials to fetch | event is self-contained |
| Ordering sensitivity | lower — fetch sees latest | higher — out-of-order fat events overwrite newer state |
| Sensitive data exposure | minimal on the wire | full object crosses the boundary |
| Coupling to your schema | loose (envelope only) | tight (the full object shape) |

The staleness point is the subtle one. Webhook delivery is **at-least-once and unordered** (§5), so a fat event can arrive *after* the resource has already changed again. If a consumer blindly writes the fat event's `data` into its own store, an out-of-order delivery can stomp newer state with older. A thin event sidesteps this — the callback `GET` always reads the latest — which is why high-stakes integrations often prefer thin events plus a fetch even though it costs a round-trip.

My default: **thin for anything mutable and high-stakes** (payments, balances), **fat for immutable facts** (`payment.succeeded` is a point-in-time truth; an audit-log entry; a "receipt generated" event). When you do send fat events, include a **version or sequence** field (§5) so the consumer can reject a stale overwrite. Stripe famously sends fat events but tells you to refetch the object if you need guaranteed-current state, which is the pragmatic middle: ship the snapshot for convenience, but be honest that it is a snapshot.

There is a third option that splits the difference and that I have come to like for high-fan-out events: a **medium event** that carries the immutable, decision-relevant fields fat but leaves the volatile or large fields thin. For `payment.succeeded`, the amount, currency, and final status are immutable facts about *that* payment — they will never change — so carrying them fat is safe and saves the round-trip. But the parent order's status, its line items, and the customer's current address are mutable and possibly large, so leave those out and let the consumer fetch them if it needs them. The rule of thumb: **a field is safe to carry fat exactly when it is immutable for the lifetime of the event's subject.** Anything that can change after the event is emitted is a staleness trap waiting to happen, and belongs behind a fetch.

#### Worked example: a fat event that arrives stale

Here is the staleness bug in concrete form. A customer pays \$49.99, then immediately the merchant issues a partial refund of \$10.00, so two events fire close together: `payment.succeeded` (amount 4999, status `succeeded`) and `refund.created` (the order's `amount_refunded` is now 1000). Both are fat. Network paths differ, and the `refund.created` delivery happens to arrive *first*, then the older `payment.succeeded` arrives a second later on a retry.

A consumer that naively writes each fat event's `data` into its own store ends up with the order showing `amount_refunded: 1000` (from the refund), and then the late `payment.succeeded` event — whose snapshot predates the refund and may carry `amount_refunded: 0` — **overwrites it back to zero**. The order now claims no refund happened, the merchant's dashboard disagrees with reality, and a support ticket is born. Nothing was lost in transit; the events were simply applied in the wrong order because fat events plus at-least-once-unordered delivery is a trap. The fix is §5's sequence check, or going thin so the consumer always fetches the live, post-refund state. This is exactly why I default high-stakes mutable data to thin: the fetch is one extra call, and one extra call is cheap next to a balance that silently un-refunds itself.

## 4. Delivery is at-least-once, so consumers must be idempotent

This is the single most important sentence about webhooks, and the place most naive integrations break: **webhook delivery is at-least-once, never exactly-once.** Your platform will sometimes deliver the same event more than once. Plan for it, or get paged for it.

Why can't you just deliver once? Because of the classic distributed-systems impossibility: the network can fail *after* the consumer processed the event but *before* its `2xx` ack reaches you. From your side, that delivery looks failed, so you retry. From the consumer's side, the work already happened. There is no way to distinguish "consumer didn't get it" from "consumer got it but the ack was lost" — so a reliable sender must retry, and a correct consumer must tolerate the duplicate. This is the same at-least-once reality that broker internals live with; the [delivery-semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) derives why exactly-once-delivery is a fiction and exactly-once-*processing* is what you actually engineer toward.

The engineering answer is **idempotent consumption**, and it has exactly one mechanism: the **stable event `id`**. The consumer keeps a record of processed event ids and skips anything it has seen. Concretely:

```python
def handle_webhook(event: dict) -> None:
    event_id = event["id"]                      # the stable dedup key

    # Atomically claim this event id; if it's already there, we've seen it.
    inserted = processed_events.insert_if_absent(event_id, ttl_days=30)
    if not inserted:
        return                                  # duplicate — ack and stop, no double work

    # First time we've seen this event: do the side effect.
    payment = event["data"]
    ledger.credit(payment["order_id"], payment["amount"])
```

`insert_if_absent` must be **atomic** — an `INSERT ... ON CONFLICT DO NOTHING` against a unique index on `event_id`, or a Redis `SET event_id 1 NX`. The atomicity is what makes it safe under concurrency: if two retries of the same event arrive at the same instant (it happens), exactly one wins the insert and does the work; the other sees the conflict and no-ops. This is the consumer-side mirror of the producer-side [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) you put on `POST /payments`: there the *caller* sends a key so *your* server dedups; here *you* send the event `id` so the *consumer* dedups. Same principle, opposite direction.

A subtle requirement: the **dedup window** must outlive your **retry window**. If you retry for up to 3 days but the consumer only remembers event ids for 1 hour, a 2-day-late retry will be processed twice. Make the consumer's dedup TTL at least as long as your maximum retry horizon, with margin.

A second subtlety that trips people: the dedup record and the side effect must be **committed together, or in a way that the side effect cannot happen without the dedup record existing**. The dangerous shape is "do the side effect, then write the dedup record" — if the process crashes between the two, the next retry sees no dedup record and does the side effect again. The safe shapes are either (a) write the dedup record and perform the side effect in the **same database transaction**, so they commit atomically, or (b) make the side effect *itself* the thing keyed on the event id — for example, the ledger entry's primary key is derived from the event id, so a duplicate event produces a duplicate-key conflict on the ledger insert directly and there is no separate dedup table to fall out of sync. Pattern (b) is the cleanest when your side effect is a database write you control; pattern (a) is the fallback when the side effect spans systems. What you must never do is treat dedup as a best-effort cache that can quietly evict an id you still need.

It is worth being precise about what idempotency buys you and what it does not. Idempotent consumption guarantees **exactly-once *effect*** — the ledger is credited once — even under at-least-once *delivery*. It does **not** make the delivery exactly-once (that is impossible over an unreliable network), and it does **not** by itself fix ordering (a duplicate is dropped, but a stale *different* event is a separate problem handled by the sequence check in §5). The rule to hold onto: delivery is at-least-once and out of order; your job is to make *processing* converge to the same correct state regardless of how many times each event arrives and in what order. Idempotency handles the "how many times," sequence handles the "what order," and together they give you a consumer that is correct under the full chaos of real delivery.

#### Worked example: a retried webhook, consumed idempotently

Let me walk one concrete delivery through failure and recovery.

1. **14:40:00** — `payment.succeeded` (`evt_1Nf8a2c0XyZ`) fires. Your platform `POST`s it to the merchant.
2. The merchant's handler runs: `insert_if_absent("evt_1Nf8a2c0XyZ")` succeeds, it credits the ledger \$49.99, and it begins to return `200`.
3. **The TCP connection drops** before the `200` reaches your platform. From your side: timeout. You mark the delivery failed and schedule a retry.
4. **14:41:00** — retry #1 of the *same* event, same `id`. The merchant's handler runs again: `insert_if_absent("evt_1Nf8a2c0XyZ")` now **conflicts** — the id is already recorded. The handler returns `200` immediately, **does not credit the ledger again**, and your platform marks the delivery done.

The customer was charged once, the ledger was credited once, and the duplicate delivery was a no-op. Without the dedup check, step 4 credits \$49.99 a second time — a reconciliation nightmare that surfaces days later when finance asks why the ledger and the processor disagree by exactly one transaction. The fix is not "make delivery exactly-once" (impossible); it is "make the *effect* idempotent" (a unique index and four lines of code).

## 5. Retries, dead-letters, and the ordering you do not get

A reliable sender does not give up on the first failure. When a delivery does not return a `2xx` within the timeout, your platform **retries with exponential backoff** — waiting progressively longer between attempts so a briefly-down consumer recovers without being hammered, and a permanently-down one is eventually given up on.

![a timeline of webhook delivery showing a failed attempt followed by retries at one second one minute and one hour then a dead-letter and finally a manual replay that acks](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-4.png)

A typical schedule for delivery attempt $n$ uses delay $d_n = \min(b \cdot 2^{\,n-1},\ d_{\max})$ plus a little jitter, with $b$ a base of a few seconds and a cap of an hour or so. The doubling is the key: it spreads a fixed number of attempts across a long window (seconds, then minutes, then hours), so a consumer that is down for a deploy recovers on an early retry while one down for a day still has attempts left near the end. The **jitter** (a random fraction added to each delay) matters more than it looks: if a consumer goes down and comes back up, every event queued during the outage would otherwise retry *at the same instant* — a thundering herd that knocks the consumer right back over. Jitter smears the retries across the window. The mechanics of backoff and jitter are the same ones the [dead-letter-queues and retries post](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff) derives for brokers; webhooks are that pattern applied to HTTP delivery.

After $N$ exhausted attempts (commonly tried over a window of several hours to a few days), the event is **dead-lettered**: moved off the live retry path into durable storage for later inspection and **manual or automated replay**. The dead-letter store is what stops a single permanently-broken consumer from clogging your delivery pipeline forever, and it is what lets a partner who was down for a long maintenance window recover the events they missed. Do not silently drop after the last retry — drop *into a dead-letter*. The difference between "we lost your events" and "we held your events and you can replay them" is the difference between a churned customer and a grateful one.

#### Worked example: a delivery retried to exhaustion, then replayed

The merchant deploys a bad build; their webhook endpoint returns `503` for two hours.

| Attempt | Time | Backoff | Consumer response | Outcome |
| --- | --- | --- | --- | --- |
| 1 | 14:40:00 | — | `503` | failed, schedule retry |
| 2 | 14:40:05 | ~5s | `503` | failed |
| 3 | 14:41:00 | ~1m | `503` | failed |
| 4 | 14:50:00 | ~10m | `503` | failed |
| 5 | 15:40:00 | ~1h | `503` | failed |
| 6 | 16:40:00 | ~1h (cap) | `503` | exhausted → **dead-lettered** |

At 17:00 the merchant fixes the build. They hit your replay endpoint, `POST /v1/webhook_endpoints/{id}/replay` with a time range, and your platform re-delivers the dead-lettered `evt_1Nf8a2c0XyZ`. Their (idempotent, §4) handler processes it once. Net result after a two-hour outage: zero lost events, zero duplicates, one slightly stressed on-call engineer who is very glad the dead-letter existed.

### Ordering is not guaranteed — design events to be order-independent

Here is the constraint that bites teams who came from a single database and assume the world is sequential: **webhook delivery does not guarantee order.** Because deliveries retry independently and travel different network paths, `payment.succeeded` and the later `refund.created` for the same payment can arrive in either order. A retry can deliver an *older* event *after* a newer one. You must design for this; you cannot wish it away over HTTP at fan-out scale. (Brokers can offer per-key ordering with real constraints — see [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — but a fan-out webhook system generally does not.)

Two robust patterns, used together:

1. **Make events order-independent (idempotent on state).** Prefer events that describe a *resulting fact* a consumer can apply in any order. If the consumer's reaction to `payment.succeeded` is to set `order.status = "paid"`, and the reaction to `refund.created` is to set `order.status = "refunded"`, then receiving them out of order still needs a tiebreaker — which is the second pattern.
2. **Carry a sequence or version and reject the stale one.** Put a monotonic `sequence` (or the resource's own `version` / `updated_at`) in the event. The consumer stores the highest sequence it has applied per resource and **ignores any event with a lower-or-equal sequence**:

```python
def apply_order_event(event: dict) -> None:
    data = event["data"]
    order_id = data["order_id"]
    seq = event["sequence"]                      # monotonic per resource

    current = orders.get_version(order_id)
    if seq <= current:
        return                                   # stale or duplicate — drop, don't overwrite
    orders.apply(order_id, data, version=seq)
```

This is "last-writer-wins by sequence, not by arrival time" — the same conditional-write logic an `If-Match`/`ETag` gives you on the request side, moved into the event. With it, a delayed retry of an old event cannot stomp newer state: the consumer simply sees a stale sequence and drops it.

A practical note on where the sequence comes from. The cleanest source is the **resource's own version**, which you are probably already tracking for optimistic concurrency on the request side — the same integer that backs your `ETag` on `GET /orders/{id}`. Emit it on every event about that resource, and the consumer's "highest applied version" check becomes a direct comparison against a number it already understands. If you do not have per-resource versioning, a monotonic per-resource counter or even a high-resolution event timestamp can serve, with the caveat that timestamps from different machines can skew and tie; a real counter is safer. What you should *not* do is rely on a single global sequence across all resources — that forces total ordering you cannot actually deliver at fan-out scale, and it creates a contention point. Order is only meaningful *per subject*: the events about order `ord_77Bc1Q` need to be orderable relative to each other, not relative to some unrelated payment.

A consequence worth stating plainly: **if you emit events without any ordering token, you are asking every consumer to either not care about order or to invent their own tiebreaker.** Some will care, most will not have thought about it, and the ones who naively apply fat events in arrival order will have the silent-data-corruption bug from §3 — and they will blame your webhooks, not their handler, when the order's state goes wrong. Shipping a `sequence` (or `version`) in the envelope from day one is cheap insurance against an entire class of consumer bug you would otherwise spend support cycles diagnosing on their behalf.

## 6. Signature verification: HMAC over the body, plus a timestamp against replay

A webhook endpoint is a publicly reachable URL that performs real side effects — crediting ledgers, fulfilling orders, sending money. If anyone on the internet can `POST` a forged `payment.succeeded` to it, you have built a free-money machine for attackers. So the consumer **must** verify that each request genuinely came from you and was not tampered with or replayed. This is the security crux of the whole design, and the standard answer is an **HMAC signature**.

An **HMAC** (Hash-based Message Authentication Code) is a keyed hash: $\text{HMAC}(k, m) = H\big((k \oplus opad)\ \|\ H((k \oplus ipad)\ \|\ m)\big)$ for a hash $H$ like SHA-256, a shared secret key $k$, and message $m$. The property that matters: **only someone who knows $k$ can produce a valid MAC for a given $m$**, and any change to $m$ produces a completely different MAC. You and the consumer share a secret (a **signing secret**, often shown as `whsec_...`); you sign each event with it, and the consumer recomputes the signature and checks it matches. No private keys, no certificate machinery — just a shared secret and a hash.

![a timeline of the HMAC flow where the sender builds the payload with a timestamp signs it with the secret and the receiver recomputes the MAC does a constant-time compare and rejects stale or wrong signatures](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-5.png)

Two non-obvious requirements make this actually secure:

- **Sign the timestamp together with the body, and the consumer must check the timestamp's freshness.** A bare signature over the body alone is replayable: an attacker who captures one valid signed request can resend it verbatim forever, and it will verify every time because the signature is still correct. Binding a **timestamp** into the signed string and rejecting requests whose timestamp is outside a tolerance window (commonly five minutes) closes the replay window. The signed string is `timestamp + "." + raw_body`, not just `raw_body`.
- **Sign the raw bytes, and compare in constant time.** Verify over the **exact raw request body** before any JSON parse-and-reserialize — re-encoding can reorder keys or change whitespace and break the MAC. And compare the computed and received signatures with a **constant-time** equality (`hmac.compare_digest`), never `==`, so a timing side-channel cannot leak the correct signature byte by byte.

#### Worked example: verifying a signed webhook with a timestamp

Here is the full verification a consumer runs on every request. The signing secret here is an obvious placeholder — in production it comes from your dashboard and is rotated, never hard-coded.

```python
import hashlib
import hmac
import time

SIGNING_SECRET = b"whsec_YOUR_SIGNING_SECRET"   # placeholder — load from a secret store
TOLERANCE_SECONDS = 300                          # 5-minute replay window

def verify_webhook(raw_body: bytes, headers: dict) -> dict:
    timestamp = headers["Webhook-Timestamp"]            # e.g. "1781049600"
    received = headers["Webhook-Signature"]             # e.g. "v1,3a9f0b2c..."

    # 1. Reject anything outside the freshness window — this kills replay.
    age = abs(time.time() - int(timestamp))
    if age > TOLERANCE_SECONDS:
        raise ValueError("timestamp outside tolerance — possible replay")

    # 2. Recompute the MAC over EXACTLY "timestamp.rawbody" using the raw bytes.
    signed_payload = timestamp.encode() + b"." + raw_body
    expected = hmac.new(SIGNING_SECRET, signed_payload, hashlib.sha256).hexdigest()

    # 3. Strip the "v1," scheme prefix and compare in CONSTANT time.
    received_sig = received.split(",", 1)[1]
    if not hmac.compare_digest(expected, received_sig):
        raise ValueError("signature mismatch — forged or tampered")

    # 4. Only now is it safe to parse and act on the body.
    import json
    return json.loads(raw_body)
```

The sender side is the mirror image — same secret, same `timestamp.rawbody` string:

```python
def sign_event(raw_body: bytes, secret: bytes) -> dict:
    ts = str(int(time.time()))
    signed_payload = ts.encode() + b"." + raw_body
    sig = hmac.new(secret, signed_payload, hashlib.sha256).hexdigest()
    return {
        "Webhook-Timestamp": ts,
        "Webhook-Signature": f"v1,{sig}",
    }
```

Two more practices that distinguish a robust implementation. First, the `v1,` **scheme prefix** on the signature is deliberate: it lets you introduce `v2` (a new algorithm) later and send *both* during a migration, so consumers can verify against either. Second, support **two active secrets at once** during rotation: when a partner rotates their signing secret, generate the new one, send signatures under **both** old and new for an overlap window, then retire the old. A consumer that accepts a match against either secret experiences zero downtime during rotation. Rotation that forces a flag-day cutover is rotation nobody does, which is how secrets end up a decade old.

A note on what HMAC does and does not give you. It gives **authenticity** (it came from someone with the secret) and **integrity** (the body was not altered). It does **not** give confidentiality — the body is plaintext on the wire, so use HTTPS, and do not put secrets in event payloads. And it is symmetric: the consumer holds the same secret you do, so they could in principle forge events *to themselves*, which is fine — the threat model is third parties, not the consumer. (If you need the consumer to prove receipt to a third party, you need asymmetric signatures, which is heavier and rarely worth it for webhooks.)

### The attacks the signature stops, in order of nastiness

It helps to name the concrete attacks, because each design rule maps to one. **Forgery** is the obvious one: an attacker who discovers the consumer's webhook URL (which leaks easily — it is in logs, in configuration, sometimes in client-side code) sends a fabricated `payment.succeeded` to trigger fulfillment for an order that was never paid. The HMAC stops this cold: without the secret, the attacker cannot produce a matching signature, and the consumer rejects the request before any side effect runs. This is why **verifying the signature must happen before the body is trusted for anything** — including before logging the body in a way that might itself be exploited.

**Tampering** is forgery's subtler cousin: a man-in-the-middle (or a malicious proxy) intercepts a legitimate event and changes `amount` from 4999 to 99 before forwarding it. Because the HMAC is computed over the exact body bytes, any change to the body invalidates the signature, so the consumer detects the tamper. This is also why you verify over the **raw bytes**: if you parse the JSON, mutate nothing, and re-serialize, a different key order or whitespace produces different bytes and the legitimate signature now fails to verify — you would have created a self-inflicted denial of service against your own valid events. Capture the raw body *before* any framework middleware parses it.

**Replay** is the one teams forget. An attacker captures one valid, correctly-signed `payment.succeeded` — say by reading it from a log or a proxy — and resends the identical bytes, headers and all, a hundred times. The signature is genuine, so a naive verifier accepts every copy; if the consumer is not idempotent, the order is fulfilled a hundred times. Two defenses stack here: the **timestamp window** rejects a replay sent outside the tolerance (so an old captured event cannot be replayed tomorrow), and **idempotent consumption** on the event id (§4) makes even an in-window replay a no-op. Defense in depth: the timestamp narrows the window to minutes, idempotency closes whatever is left. Relying on only one is fragile; relying on both is robust.

There is one more operational rule that prevents a whole category of self-owns: **never disable signature verification "temporarily" to debug.** I have seen an endpoint shipped with verification commented out behind a "we'll turn it on later" flag, and later never came; the endpoint sat open to forgery for months. If you need to debug, use the test-event endpoint (§9) and the dashboard's signed sample, not a verification bypass. A webhook endpoint with verification off is, for an attacker who finds the URL, a button labeled "fulfill any order for free."

## 7. The consumer contract: 2xx fast, or we retry

A webhook is a two-party contract, and the consumer has obligations too. The most important one: **return `2xx` quickly, and do the slow work asynchronously.** Your delivery system has a timeout — commonly a few seconds — and it treats a slow response the same as a failure: it retries. So a consumer that does heavy work *inside* the webhook handler before responding will (a) hit your timeout, (b) get retried, and (c) do that heavy work *again* on the retry. The handler that tries to be thorough becomes the handler that triple-processes every event.

The correct shape is **ack-then-process**: validate the signature, enqueue the event onto the consumer's own internal queue (or write it to a table), and return `200` immediately. The actual business logic runs out-of-band off that queue.

```python
@app.post("/hooks/payments")
def receive(request):
    event = verify_webhook(request.raw_body, request.headers)   # §6 — fast, in-line
    internal_queue.enqueue(event)                               # hand off, do not process here
    return Response(status=200)                                 # ack within the timeout
```

This decouples "I received it durably" (fast, must beat the timeout) from "I finished acting on it" (slow, retried internally on the consumer's terms). It also means the consumer's retries are *theirs* — under their control, with their own DLQ — instead of yours re-delivering because their database was briefly slow. The contract is therefore: **the consumer acks receipt fast; the consumer owns processing.** Document the timeout explicitly so integrators design for it; a timeout discovered in production is a support ticket, a timeout documented up front is a non-event.

Status-code semantics on the consumer side, which you should document:

- **`2xx`** — acknowledged; do not retry.
- **`4xx`** (except `429`) — the request is malformed or the endpoint rejects it permanently; usually *do not* retry a `400`/`401`/`410`, because retrying a permanently-bad request just wastes attempts. (A `410 Gone` is a useful signal from the consumer that the endpoint is retired — stop delivering and disable it.)
- **`429`** / **`5xx`** — transient; retry with backoff.
- **timeout / connection error** — transient; retry with backoff.

There is a sharp design question hiding in the ack-then-process rule: **what does the consumer do if the body fails signature verification?** The instinct is to return a `400` so you stop retrying — but think about who is on the other end. If the signature genuinely came from you and the consumer's verification is misconfigured (wrong secret, parsed-then-reserialized body), a `400` makes you give up and the consumer silently loses real events. If the request is a forgery, you should never have sent it in the first place, so your retry behavior is irrelevant. The pragmatic answer most platforms land on: a signature failure on the consumer side returns `4xx` (the consumer is refusing to trust it), and that is fine — *your* legitimate, correctly-signed events will verify, and a consumer that is dropping them is misconfigured in a way they need to notice and fix, which the dashboard's delivery log surfaces as a wall of `400`s. The thing to avoid is a consumer that returns `200` on a verification failure: that acks an event it never actually processed, and the event is lost with no signal at all.

The deeper principle here is that **the response status is a contract between two systems about what should happen next**, and a webhook delivery is no exception. A `2xx` means "I have it durably, do not send it again." A `5xx` means "something broke on my side, try again later." A `4xx` means "do not bother resending this exact thing, it will not get better." Honoring those semantics on both sides — you respecting them in your retry logic, the consumer setting them truthfully — is what makes the system self-correcting instead of either lossy or stuck in a retry loop. This is the same honest-status-code discipline the [status codes post](/blog/software-development/api-design/status-codes-that-tell-the-truth-2xx-3xx-4xx-5xx) argues for on the request side, now governing delivery instead of a request.

## 8. Replay, backfill, and the event log

If you take one operational lesson from this post, take this: **persist every event in an append-only event log, decoupled from delivery.** The event log is your source of truth; delivery is a projection off it. The moment you separate "the event happened and is recorded" from "the event was delivered to subscriber X," three capabilities you will absolutely need become trivial:

- **Replay** — re-deliver a specific event (or range) that a consumer missed because they were down. Without a log you have nothing to replay *from*.
- **Backfill** — when a consumer registers a *new* endpoint, optionally deliver historical events from a point in time so they start with a complete picture instead of only future events.
- **Audit and debugging** — "what exactly did we send and when did each delivery succeed?" is answerable from the log plus per-delivery attempt records. When a partner swears they never got an event, you can show them the timestamps.

A minimal replay API on our platform:

```http
POST /v1/webhook_endpoints/wep_42/replay HTTP/1.1
Host: api.commerceplatform.example
Authorization: Bearer <token>
Content-Type: application/json

{
  "since": "2026-06-20T14:00:00Z",
  "until": "2026-06-20T17:00:00Z",
  "types": ["payment.succeeded", "refund.created"]
}
```

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "replay_id": "rpl_5xQ",
  "matched_events": 318,
  "status": "queued"
}
```

Replay returns `202 Accepted` — it is itself a long-running operation, the [LRO pattern](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks) in action — and because every replayed event keeps its original `id`, idempotent consumers absorb the replay without double-processing. That is the payoff of getting §4 right: replay and backfill are *safe by construction* the moment consumers dedup on the event `id`. Replay without idempotent consumers is a way to double-charge everyone twice; replay with idempotent consumers is a routine recovery tool.

Alongside the event log, keep **per-delivery attempt records**: for each (event, endpoint) pair, store every attempt with its timestamp, the HTTP status returned, the response time, and a truncated copy of the response body. This is a different table from the event log — the log says "this event happened," the attempt records say "here is what happened each time we tried to deliver it to endpoint X." Together they answer the two questions you get paged about: "did the event happen?" (log) and "did the consumer ever get it, and what did they say?" (attempts). When a partner insists they never received `evt_1Nf8a2c0XyZ`, you can show them six attempts, each with their own endpoint returning `503`, the exact timestamps, and the response bodies their own server sent back. That moves the conversation from "your webhooks are broken" to "your endpoint was down from 14:40 to 16:40, here is the proof, and here is the replay button." The attempt records are also what power the dashboard delivery log and the per-endpoint health metrics (success rate, p95 response time) that let you proactively disable a chronically-failing endpoint before it drags down your delivery pipeline.

A note on backfill scope. Backfill is genuinely useful — a partner who just integrated wants the last 30 days of payments, not just future ones — but it is also a foot-gun if unbounded. A naive "replay everything since the beginning of time" against a busy endpoint can deliver millions of events and overwhelm a consumer that is sized for steady-state traffic. Bound backfill by a time range *and* a rate limit on re-delivery, and make the consumer opt in to the volume. The same idempotency that makes replay safe makes backfill safe too — a consumer that already has some of those historical events simply dedups them — but safe is not the same as gentle, so pace the firehose.

## 9. The webhook management API: register, list, rotate, test

Webhooks are a product surface, so the configuration around them is itself an API — and a good one is the difference between a partner who integrates in an afternoon and one who files three support tickets. The management API lets consumers register endpoints, choose which event types they want, list and inspect their endpoints, rotate the signing secret, and **send a test event** so they can verify their handler before real money flows through it.

Register an endpoint, subscribing to a filtered set of event types:

```http
POST /v1/webhook_endpoints HTTP/1.1
Host: api.commerceplatform.example
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://merchant.example.com/hooks/payments",
  "enabled_events": ["payment.succeeded", "payment.failed", "refund.created"],
  "description": "Production payments handler"
}
```

```http
HTTP/1.1 201 Created
Location: /v1/webhook_endpoints/wep_42
Content-Type: application/json

{
  "id": "wep_42",
  "url": "https://merchant.example.com/hooks/payments",
  "enabled_events": ["payment.succeeded", "payment.failed", "refund.created"],
  "status": "enabled",
  "signing_secret": "whsec_YOUR_SIGNING_SECRET",
  "created": "2026-06-20T14:30:00Z"
}
```

The `signing_secret` is returned **once** at creation (and on rotation), never on subsequent reads — treat it like a password. Rotation is its own endpoint, and it should support an overlap window so the consumer can deploy the new secret before the old one stops being used (§6):

```http
POST /v1/webhook_endpoints/wep_42/rotate_secret HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{"overlap_hours": 24}
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "wep_42",
  "signing_secret": "whsec_NEW_SIGNING_SECRET",
  "old_secret_expires_at": "2026-06-21T14:30:00Z"
}
```

And the single most underrated endpoint — **send a test event**:

```bash
curl -X POST https://api.commerceplatform.example/v1/webhook_endpoints/wep_42/test \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"type": "payment.succeeded"}'
```

A test event lets an integrator confirm their URL is reachable, their signature verification works, and their handler returns `2xx` — *before* a real customer's payment depends on it. Couple this with a per-endpoint **delivery log** in the dashboard (every attempt, its status code, its response time, the raw payload, and a "resend" button) and you have turned the most opaque part of an integration — "why didn't my webhook arrive?" — into something the partner can debug themselves. The endpoints, in one table:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/v1/webhook_endpoints` | `POST` / `GET` | register a new endpoint / list existing ones |
| `/v1/webhook_endpoints/{id}` | `GET` / `PATCH` / `DELETE` | inspect / update event filter or URL / remove |
| `/v1/webhook_endpoints/{id}/rotate_secret` | `POST` | rotate the signing secret with an overlap window |
| `/v1/webhook_endpoints/{id}/test` | `POST` | send a sample event to validate the handler |
| `/v1/webhook_endpoints/{id}/replay` | `POST` | re-deliver historical events in a range |

## 10. Pub/sub and event streaming: the internal-scale alternative

Webhooks are the right tool when the consumer is an **external party** behind a URL you do not control. But inside your own platform — service to service, where you control both ends and the fan-out is large and high-volume — webhooks are usually the *wrong* tool. Hosting an HTTP endpoint per consumer, signing every delivery, and running a per-consumer retry queue is a lot of machinery for two of your own services talking. Internally, you reach for a **broker**: pub/sub or an event log.

![a graph showing one payment dot succeeded event emitted once into an event bus that fans out to a ledger URL an email URL and a fraud URL that times out and dead-letters after N failures](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-3.png)

The three internal messaging models — and the forces that pick between them — are exactly what [queue versus pub/sub versus log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) lays out in depth; I will summarize the API-design-relevant differences rather than re-derive them:

- **Queue** (point-to-point, competing consumers): each message goes to *one* consumer in a group. Use it for work distribution — "process this payment" handled by one of N workers.
- **Pub/sub** (topic, fan-out): each message goes to *every* subscriber. Use it when many independent services each need the same event — `payment.succeeded` to ledger, email, and analytics at once. This is the broker analog of a webhook fan-out, but in-house.
- **Log** (durable, ordered, replayable — Kafka-style): messages are appended to a partitioned, retained log; consumers track their own offset and can rewind. Use it when you need **ordering per key**, **replay from history**, and **many consumers reading at their own pace**. The log is what makes "rebuild a consumer's state from scratch by replaying everything" possible.

The delivery and ordering realities carry straight over from the webhook section, and the broker gives you stronger tools to manage them: brokers can offer **at-least-once** delivery (so, same as webhooks, your consumers must be idempotent — [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once)) and, crucially, **per-partition ordering** if you partition by a key like `payment_id` ([ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees)). A common production architecture is a **hybrid**: services publish internally to a log/topic; a dedicated *webhook-delivery service* subscribes to that topic and is the one thing that fans events out to external partner URLs with signing and retries. Internal scale gets the broker; external partners get webhooks; the event log sits in the middle as the shared source of truth. (Publishing to that log atomically with your database write is the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — the way you avoid the "committed the payment but crashed before publishing the event" gap.)

#### Worked example: why the webhook-delivery service is its own thing

Picture the alternative — each internal service that produces an event also being responsible for delivering it to external partner URLs. The payments service would need to know about every partner endpoint, hold every partner's signing secret, run its own retry queue and dead-letter, and host its own delivery metrics. So would the orders service, and the refunds service, and every future service. You would have re-implemented the entire webhook machinery N times, with N subtly different retry schedules and N places a signing secret can leak. Worse, a slow partner endpoint would now apply backpressure directly onto your payments service's threads, coupling a third party's downtime to your core write path.

Factoring out a single **webhook-delivery service** fixes all of it. Every internal service just publishes a domain event to the log — a fast, local, fire-and-forget write — and is done. The delivery service is the one component that subscribes to the log, knows the endpoint registry, holds the secrets, signs, retries, dead-letters, and exposes delivery health. A partner being down now affects only the delivery service's queue depth, not your payment-capture latency. And because there is exactly one place that does delivery, there is exactly one place to fix a bug, tune a backoff, or add a new security scheme. This is the same separation-of-concerns instinct that puts authentication in a gateway rather than in every handler: concentrate the cross-cutting, error-prone machinery in one well-tested component and let the rest of the system stay simple. The event log in the middle is what makes the factoring possible — it is the seam between "the event happened" (every service's job) and "the event got delivered" (one service's job).

| Force | Webhook (HTTP push) | Pub/sub topic | Event log (Kafka-style) |
| --- | --- | --- | --- |
| Consumer location | external, you don't control | internal services | internal services |
| Fan-out | per-endpoint, you push to each | broker fans out | many readers, own offset |
| Ordering | none (per delivery) | best-effort / none | per-partition, strong |
| Replay | from your event log | usually not | native, rewind offset |
| Per-event security | HMAC signature required | network/mTLS, internal | network/mTLS, internal |
| Operational owner | your delivery service | the broker | the broker |

## 11. AsyncAPI and CloudEvents: writing the event contract down

Here is the gap that bites every event-driven platform eventually. On the request side you have OpenAPI — a machine-readable spec that documents every endpoint, generates clients, mocks the server, and lints for consistency ([the spec-first workflow](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs) is the contract-test cousin of it). On the event side, for years, there was *nothing* — the "contract" was a wiki page and the source code of the publisher. A consumer learned the shape of `payment.succeeded` by getting one and reverse-engineering it. That is not a contract; that is folklore.

**AsyncAPI** fills that gap: it is, deliberately, "OpenAPI for event-driven APIs." It describes **channels** (the topics/queues/webhook routes events flow over), **messages** (the event payloads, with JSON Schema), **operations** (publish/subscribe), and the **servers** (the broker or HTTP endpoint). From an AsyncAPI document you can generate documentation, validate payloads, and scaffold code — the same DX payoff OpenAPI gives REST. Here is a fragment describing our `payment.succeeded` webhook:

```yaml
asyncapi: 3.0.0
info:
  title: Commerce Platform Webhooks
  version: 2026-06-20
channels:
  paymentEvents:
    address: /hooks/payments
    messages:
      paymentSucceeded:
        $ref: "#/components/messages/PaymentSucceeded"
operations:
  receivePaymentSucceeded:
    action: send
    channel:
      $ref: "#/channels/paymentEvents"
components:
  messages:
    PaymentSucceeded:
      name: payment.succeeded
      contentType: application/json
      payload:
        type: object
        required: [id, type, created, api_version, data]
        properties:
          id: { type: string, description: "stable event id, the dedup key" }
          type: { type: string, const: payment.succeeded }
          created: { type: string, format: date-time }
          api_version: { type: string }
          data:
            type: object
            required: [id, amount, currency, status]
            properties:
              id: { type: string }
              amount: { type: integer, description: "minor units, e.g. cents" }
              currency: { type: string }
              status: { type: string, enum: [succeeded] }
```

That document is a real contract: a consumer can validate every incoming event against it, and you can run a **schema-diff in CI** to catch a breaking change to an event before it ships — exactly the discipline the [contract-testing post](/blog/software-development/api-design/contract-testing-consumer-driven-contracts-and-schema-diffs) applies to request APIs, now applied to events. Adding an optional field to `data` is non-breaking (tolerant readers ignore unknown fields); removing a field or changing a type is breaking and the diff flags it. The [compatibility rules](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change) for request payloads apply unchanged to event payloads.

### CloudEvents: a standard envelope

The other piece of standardization is **CloudEvents** (a CNCF spec): a vendor-neutral **envelope** for event data, so events from different producers share a common metadata shape and tools can route them without bespoke parsers. It defines required attributes — `id`, `source`, `type`, `specversion` — plus optional ones like `time`, `subject`, and `datacontenttype`, with the payload in `data`. The same `payment.succeeded` in CloudEvents structured form:

```json
{
  "specversion": "1.0",
  "id": "evt_1Nf8a2c0XyZ",
  "source": "/commerceplatform/payments",
  "type": "com.commerceplatform.payment.succeeded",
  "time": "2026-06-20T14:40:00Z",
  "datacontenttype": "application/json",
  "data": {
    "id": "pay_9KqR2mN", "amount": 4999, "currency": "usd", "status": "succeeded"
  }
}
```

You will recognize every field — it is the envelope from §3, standardized. The win of adopting CloudEvents is **interoperability**: a consumer (or a gateway, or a broker, or a tracing system) that understands the CloudEvents envelope can handle events from *any* CloudEvents-emitting source without custom code, and major brokers and cloud event buses speak it natively. AsyncAPI describes the contract; CloudEvents standardizes the envelope; they compose. Pair them and your events have both a spec (AsyncAPI) and a common shape (CloudEvents) — the event world finally catching up to where REST has been for a decade.

## 12. Case studies: how real platforms do this

**Stripe** is the reference implementation of webhooks as a product, and most of this post mirrors choices they popularized. Stripe sends **fat events** — the `Event` object wraps a full snapshot of the changed object in `data.object` — while explicitly advising you to refetch the object from the API if you need guaranteed-current state, the pragmatic middle of §3. They sign every webhook with **HMAC-SHA256** over a string composed of the **timestamp and the raw body**, sent in a `Stripe-Signature` header, and their docs walk through exactly the timestamp-tolerance and constant-time-compare verification in §6 (their default tolerance is five minutes, where our example's 300 seconds comes from). They retry failed deliveries with **exponential backoff over a multi-day window**, expose a **dashboard event log with per-attempt detail and a resend button**, and version the event payload by the account's **pinned API version** so an old integration keeps receiving the shape it was built against. The whole shape — envelope, signing, retries, replay, versioned payloads — is the playbook this post teaches.

**GitHub** sends webhooks for repository and organization activity, and adds details worth borrowing. They include both an **`X-GitHub-Delivery`** header (a unique GUID per delivery — the dedup key) and an **`X-GitHub-Event`** header (the event type, so a consumer can route without parsing the body first), and they sign with an HMAC in **`X-Hub-Signature-256`**. Their UI exposes a **"Recent Deliveries"** view with the full request, the response, and a **redeliver** button — the self-service debugging surface that turns "my webhook didn't fire" from a support ticket into a thing the integrator solves alone. The lesson: put dedup id and event type in *headers*, not only the body, so a consumer can dedup and route before it even parses.

**CloudEvents and AsyncAPI** represent the industry's move to standardize the event surface. CloudEvents graduated through the CNCF as a common envelope adopted across major cloud event buses and serverless platforms; AsyncAPI has grown into the de-facto spec language for documenting event-driven and message-driven APIs, with tooling for docs, codegen, and validation that mirrors the OpenAPI ecosystem. The [Standard Webhooks](https://www.standardwebhooks.com/) initiative is a more recent effort to standardize the webhook-specific bits — header names (`webhook-id`, `webhook-timestamp`, `webhook-signature`), the signing scheme, and verification — so that consuming webhooks from different vendors stops requiring a bespoke verifier per provider. The direction of travel is clear: events are becoming a first-class, specified, tooled part of the API surface rather than an undocumented side channel.

## 13. When to reach for events (and when not to)

Events are powerful and they are not free — you are taking on delivery, retries, signing, ordering, and a dead-letter to operate. Reach for the push model deliberately.

**Reach for webhooks when:**

- A consumer needs to know about a **discrete domain event** soon after it happens (`payment.succeeded`, `order.shipped`) and polling would waste real money or add unacceptable latency.
- The consumer is an **external party** behind a URL you do not control — webhooks are the lingua franca of third-party integrations.
- You can make consumers **idempotent on the event id** and you are willing to operate retries, signing, and a dead-letter.

**Reach for pub/sub or an event log when:**

- The producer and consumer are **internal services you both control**, and the fan-out is high-volume or many-to-many — let a broker do what it is built for instead of hosting HTTP endpoints between your own services.
- You need **per-key ordering** or **replay from history** — that is a log's job, not a webhook's.

**Do not reach for events when:**

- The client just needs **the final state of one operation** and the operation is infrequent — a `202 Accepted` plus a poll with `Retry-After` is simpler, stateless, and has no endpoint to host or secret to verify ([long-running operations](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks)).
- You need a **continuous, high-frequency stream** to a live UI (a price ticker, a chat) — that is **SSE or WebSocket**, an open connection, not discrete webhooks ([streaming APIs](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming)).
- The interaction is genuinely **request/response** and the caller wants the answer in the same call — do not turn a synchronous read into an event-driven dance for fashion's sake. Most of your API is, and should remain, synchronous request/response.
- You cannot yet make the consumer **idempotent**. Shipping at-least-once webhooks to a consumer that double-processes is shipping a double-charge bug on a timer. Fix idempotency first, then ship the webhook.

![a comparison matrix scoring polling webhooks and streaming across latency cost reliability and complexity](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-6.png)

The honest framing: webhooks trade the *client's* polling cost for *your* delivery complexity. You take on the retry queue, the signing, the dead-letter, the replay tooling, and the management API so that thousands of consumers do not each poll you to death. That trade is overwhelmingly worth it past a certain scale and integration count — and overwhelmingly *not* worth it for a single internal caller who could just make a request. Choose by force, not fashion, which is the whole series' refrain applied to the push side.

## 14. Putting it together: the payment.succeeded webhook, end to end

Let me assemble the full lifecycle of one event on our Payments & Orders platform, so the pieces connect.

![a tree showing the choice of push mechanism branching from who consumes the event into a signed webhook for external partners and pub sub or an event log for internal services](/imgs/blogs/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi-8.png)

1. A customer's \$49.99 payment captures. Your payments service writes the row and, **atomically with that write** (transactional outbox), appends `payment.succeeded` to the internal event log with a stable `id` of `evt_1Nf8a2c0XyZ`.
2. The webhook-delivery service consumes that event from the log. It looks up every endpoint subscribed to `payment.succeeded` — here, the merchant's `wep_42`.
3. It builds the envelope (`id`, `type`, `created`, `api_version`, `data`), sets the merchant's account-pinned `api_version` so they get the shape they integrated against, **signs** `timestamp.rawbody` with the merchant's signing secret, and `POST`s to their URL with `Webhook-Id`, `Webhook-Timestamp`, and `Webhook-Signature` headers.
4. The merchant's handler **verifies the signature and freshness** (rejecting forgeries and replays), **enqueues** the event internally, and returns `200` within the timeout — ack-then-process.
5. Off their internal queue, the merchant **deduplicates on `evt_1Nf8a2c0XyZ`** (idempotent consumption), checks the `sequence` against the order's current version (order-independence), and credits their ledger exactly once.
6. Had the delivery failed, your service would retry on exponential backoff with jitter, and after exhaustion dead-letter the event for replay — which the merchant could trigger from the dashboard, with their idempotent handler absorbing it cleanly.

Every design choice in this post shows up in those six steps: the outbox closes the publish gap, the envelope makes the event self-describing, the signature secures it, at-least-once forces idempotency, the sequence handles ordering, and the dead-letter plus replay make missed events recoverable. Get those six right and your webhook is not a liability — it is a contract your partners can build a business on.

## 15. Key takeaways

- **Push beats poll when events are frequent or latency matters.** Polling wastes roughly $T/p$ calls per real change and adds up to one interval of latency; a webhook fires once, on the event. Keep polling for infrequent final-state checks.
- **Design the envelope once: `id`, `type`, `created`, `api_version`, `data`.** The `id` is the dedup key, the past-tense `type` says "this happened," the `api_version` lets you evolve the payload without breaking pinned consumers.
- **Delivery is at-least-once, never exactly-once.** Make consumers idempotent on the stable event `id` (an atomic insert against a unique index), with a dedup window that outlives your retry window. This is the consumer-side mirror of [idempotency keys](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions).
- **Retry with exponential backoff plus jitter, then dead-letter — never silently drop.** Jitter prevents the recovered-consumer thundering herd; the dead-letter plus a replay endpoint turn an outage into a recoverable event, not lost data.
- **Ordering is not guaranteed.** Design events to be order-independent and carry a `sequence`/`version` so a stale retry cannot stomp newer state.
- **Sign with HMAC over `timestamp.rawbody`, verify freshness, compare in constant time.** The timestamp window kills replay; the raw-bytes-before-parse rule keeps the MAC valid; rotate secrets with an overlap window.
- **The consumer contract is ack-fast, process-async.** Return `2xx` within the timeout and do the work off an internal queue, or your timeout becomes their double-processing.
- **Use webhooks for external partners, brokers for internal scale.** Pub/sub and logs ([queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models)) handle internal many-to-many fan-out, ordering, and replay better than per-endpoint HTTP push.
- **Write the contract down with AsyncAPI and CloudEvents.** Events deserve the same spec-first, schema-diffed, tooled discipline as REST — folklore is not a contract.

## Further reading

- [AsyncAPI specification](https://www.asyncapi.com/docs/reference/specification/latest) — the spec language for event-driven and message-driven APIs (channels, messages, operations).
- [CloudEvents specification (CNCF)](https://cloudevents.io/) — the vendor-neutral envelope for event data; required and optional attributes.
- [Stripe webhooks documentation](https://docs.stripe.com/webhooks) — production webhook design: event objects, signature verification, retries, and the event log, from the reference implementation.
- [Standard Webhooks](https://www.standardwebhooks.com/) — an effort to standardize webhook headers, signing, and verification across providers.
- [RFC 9457: Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457) — the error-envelope standard your webhook endpoints should return on rejection.
- Within this series: [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) · [idempotency keys, safe retries, and exactly-once illusions](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) · [long-running operations: async jobs, polling, and callbacks](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks) · [streaming APIs: SSE, WebSockets, and server streaming](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming) · [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2)
- Broker internals: [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) · [dead-letter queues, retries, exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff) · [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) · [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) · [the transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing)

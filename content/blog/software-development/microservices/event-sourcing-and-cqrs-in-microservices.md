---
title: "Event Sourcing and CQRS in Microservices: When the Log Is the Truth"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Two patterns that look like one, the enormous complexity they buy, and the small number of services that should actually pay for it: storing events as the source of truth, and splitting the write model from many read models."
tags:
  [
    "microservices",
    "event-sourcing",
    "cqrs",
    "distributed-systems",
    "software-architecture",
    "backend",
    "eventual-consistency",
    "domain-driven-design",
    "data-modeling",
    "event-driven-architecture",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/event-sourcing-and-cqrs-in-microservices-1.webp"
---

A senior engineer joined the ShopFast platform team, looked at the order service, and asked a question that should have been trivial: "What was order #A7F2 worth on March 1, and what state was it in?" The answer that came back was an embarrassed silence. The order had been refunded, partially re-charged, had two items returned, and its shipping address corrected twice. The `orders` table held exactly one row, and that row held exactly the *current* truth. Every prior truth had been overwritten by an `UPDATE`. The history existed, sort of, scattered across an application log that rotated after 14 days, a payments ledger owned by another team, and the memory of whoever had handled the support ticket. Reconstructing March 1 was not a query. It was an investigation.

This is the wound that event sourcing is designed to heal, and the team's first instinct was the right one: *what if we never overwrote anything?* What if, instead of storing the order's current state, we stored the ordered sequence of facts that produced it — `OrderPlaced`, `ItemAdded`, `OrderPaid`, `ItemReturned`, `AddressChanged`, `OrderShipped` — and computed the current state by replaying those facts on demand? Then March 1 is not lost. It is just the state you get when you stop replaying at March 1. The history is not a side effect you hope you captured; the history *is the database*.

That instinct is correct, and it is also a trap, because event sourcing is one of the highest-leverage and highest-cost patterns in the entire microservices toolbox. It pairs naturally with a second pattern — CQRS, separating the write model from the read models — and together they can give you full auditability, time-travel queries, and a read layer tuned per query that no single relational schema could match. They can also turn a service that three people understood into a distributed system that nobody understands, with eventual-consistency bugs in the UI, an event schema you are now afraid to change, and a GDPR delete request you cannot satisfy. By the end of this post you will be able to event-source an aggregate correctly, build CQRS read models from its events, reason about the consistency gap between write and read, optimize the whole thing with snapshots and rebuilds, and — most importantly — *say no* to both patterns for the 90% of services that should never use them.

![A two-column comparison contrasting a state-stored CRUD service that overwrites a single row against an event-sourced service that appends facts and derives state by folding them](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-1.webp)

We will use ShopFast — the same e-commerce system from across this series — as the spine. The `Order` aggregate is our worked example throughout: event-sourced on the write side, with two CQRS read models (an "order history" view and a "fulfillment dashboard") on the read side. This post is the *application* layer: how you build these patterns inside a service. The log mechanism itself — how an append-only event log is stored, partitioned, and consumed — is covered in [Event Sourcing and CQRS with an Event Log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log); we cross-link to it rather than re-deriving the storage internals.

## Event sourcing: store the change, derive the state

Start from the thing every junior engineer already knows: CRUD. You have an `orders` table. When an order is paid, you run `UPDATE orders SET status = 'PAID', paid_at = now() WHERE id = ?`. The row now reflects the new truth and the old truth is gone. This is fine. It is fine for the overwhelming majority of data in the world. The price you pay — and most of the time it is a price worth paying — is that the table answers exactly one question: *what is true right now?*

Event sourcing inverts this. You do not store the current state at all. You store the **events** — the immutable, past-tense facts about state changes — and you treat that ordered list as the single source of truth. The current state is not stored; it is **derived** by replaying (the technical word is *folding*) every event from the beginning.

Concretely, for the ShopFast `Order` aggregate, instead of an `orders` row you have a stream of events:

```json
[
  { "seq": 1, "type": "OrderPlaced",    "data": { "orderId": "A7F2", "customerId": 42, "currency": "USD" } },
  { "seq": 2, "type": "ItemAdded",      "data": { "sku": "SKU-19", "qty": 2, "unitPrice": 30.00 } },
  { "seq": 3, "type": "ItemAdded",      "data": { "sku": "SKU-7",  "qty": 1, "unitPrice": 24.00 } },
  { "seq": 4, "type": "OrderPaid",      "data": { "amount": 84.00, "method": "card" } },
  { "seq": 5, "type": "AddressChanged", "data": { "zip": "94107" } },
  { "seq": 6, "type": "OrderShipped",   "data": { "carrier": "UPS", "tracking": "1Z..." } }
]
```

The order's current state — total \$84.00, status SHIPPED, shipping to 94107, two SKUs — exists nowhere as a stored fact. It is what you get when you start from an empty `Order` and apply each event in sequence. That is the whole idea, and almost everything else in this post is a consequence of it.

![A six-event timeline of one Order aggregate from OrderPlaced through ItemAdded, OrderPaid, AddressChanged, and OrderShipped showing that current state is the running total of every appended fact](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-2.webp)

Read that timeline left to right and notice what is *not* there: at no point is the order's total or status written down as a value. Each card is a fact about a change, and the current order is simply the result of stacking all of them. The clean mental model is `state = fold(apply, emptyState, events)` — state is a left-fold over history. Hold onto that equation; it is the one the entire pattern rests on, and we will keep coming back to it when we talk about loading, snapshots, and rebuilding read models.

The vocabulary you need, defined once so the rest reads cleanly:

- **Event**: an immutable fact that *already happened*, named in the past tense (`OrderPaid`, not `PayOrder`). Events are never updated and never deleted. They are appended.
- **Aggregate**: a cluster of objects treated as one consistency unit — in our case the `Order` plus its line items. The aggregate is the boundary inside which a command is validated and within which events are atomically appended. (This term comes from Domain-Driven Design; see [Service Boundaries with Domain-Driven Design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design) for how you find aggregates in the first place.)
- **Event stream**: the ordered sequence of events for one aggregate instance, identified by the aggregate id (one stream per order).
- **Fold / apply**: the function that takes a current state and one event and returns the next state. Folding the whole stream from an empty state gives you the current state.
- **Command**: an *intent* to change state (`PlaceOrder`, `PayOrder`), named in the imperative. A command can be rejected. An event cannot — it already happened.
- **Projection / read model**: a query-shaped view derived from the events. We get to these when we cover CQRS; for now, just know that the events are also the input that builds every queryable view of the data.

One more distinction that trips up newcomers: events are not commands, and the naming convention is load-bearing. `PayOrder` is a *request* — it can fail validation, bounce off a business rule, or lose a concurrency race. `OrderPaid` is a *fact* — by the time it exists, the payment is irrevocably part of history. A command handler's whole job is to turn a fallible command into zero or more irrevocable events (or a rejection). If you ever find yourself wanting to "undo" an event, stop: you do not delete history, you append a *compensating* event (`OrderRefunded`, `ItemReturned`) that the fold interprets as a reversal. The arrow of time only points forward in an event-sourced system, and that constraint is exactly what gives you the audit guarantee.

### The aggregate that applies and folds events

Here is the ShopFast `Order` aggregate in TypeScript. Notice two distinct responsibilities: `apply` mutates in-memory state from an event (it never validates — the event already happened), and the command methods (`pay`, `addItem`) *validate* then *produce* new events. This separation is the heart of doing event sourcing correctly.

```typescript
type OrderEvent =
  | { type: "OrderPlaced"; orderId: string; customerId: number; currency: string }
  | { type: "ItemAdded"; sku: string; qty: number; unitPrice: number }
  | { type: "OrderPaid"; amount: number; method: string }
  | { type: "AddressChanged"; zip: string }
  | { type: "OrderShipped"; carrier: string; tracking: string };

class Order {
  id = "";
  status: "NEW" | "PLACED" | "PAID" | "SHIPPED" = "NEW";
  lines: { sku: string; qty: number; unitPrice: number }[] = [];
  zip = "";
  version = 0; // = number of events folded; this is the optimistic-lock token

  // apply: pure state transition. NEVER validates, NEVER throws on business rules.
  // It just folds one event into state. Used both on load (replay) and after a command.
  apply(e: OrderEvent): void {
    switch (e.type) {
      case "OrderPlaced":   this.id = e.orderId; this.status = "PLACED"; break;
      case "ItemAdded":     this.lines.push({ sku: e.sku, qty: e.qty, unitPrice: e.unitPrice }); break;
      case "OrderPaid":     this.status = "PAID"; break;
      case "AddressChanged":this.zip = e.zip; break;
      case "OrderShipped":  this.status = "SHIPPED"; break;
    }
    this.version++;
  }

  total(): number {
    return this.lines.reduce((s, l) => s + l.qty * l.unitPrice, 0);
  }

  // Rebuild current state from history. This IS the read of an event-sourced aggregate.
  static fromHistory(events: OrderEvent[]): Order {
    const o = new Order();
    for (const e of events) o.apply(e);
    return o;
  }
}
```

The `apply` method is deliberately dumb. It does not check whether paying is allowed; it just records that the order is now paid. All the *intelligence* lives in the command methods, which we look at next. This split matters because `apply` runs on every replay of every historical event, forever. If you ever put a business rule in `apply` — "reject payment if total is zero" — you will fail to replay old events that were valid under old rules, and your aggregate will refuse to load. Validation belongs to commands (the present); `apply` belongs to events (the past, which is unconditionally true).

Two payoffs fall straight out of this `fromHistory` design, and they are exactly the benefits worth paying for. The first is the **temporal query**: because the current state is a fold over *all* events, the state *as of any past moment* is the fold over events *up to that moment*. You answer "what did order #A7F2 look like on March 1" with the same code, stopping the fold at the March-1 boundary — no separate history table, no audit log to reconcile, no investigation:

```typescript
// State as of a point in time = fold the events that occurred on or before it.
static asOf(events: OrderEvent[], at: Date, occurredAt: (e: OrderEvent) => Date): Order {
  const o = new Order();
  for (const e of events) {
    if (occurredAt(e) > at) break; // stop the fold at the historical boundary
    o.apply(e);
  }
  return o; // the order exactly as it was at `at`
}
```

The second payoff is **rebuild/replay for free**: if a bug corrupts your in-memory model or you change how you compute the total, you do not run a data fix. You throw away the derived state and re-fold from the events, which are still pristine. The events are the truth; everything derived from them is regenerable. Those two properties — time-travel and replay — are the reason the audit-heavy domains we discuss later reach for this pattern at all.

### The command handler that appends events

A command handler does four things, in this exact order, every time:

1. **Load** the aggregate by replaying its event stream (we will optimize this with snapshots later).
2. **Validate** the command against the loaded state — this is where business rules live and where a command can be rejected.
3. **Produce** new events (do not mutate state directly).
4. **Append** the new events to the store atomically, with an optimistic concurrency check.

```typescript
class PayOrderHandler {
  constructor(private store: EventStore) {}

  async handle(cmd: { orderId: string; amount: number; method: string }): Promise<void> {
    // 1. LOAD: replay the stream to get current state
    const { events, version } = await this.store.load(cmd.orderId);
    const order = Order.fromHistory(events);

    // 2. VALIDATE: business rules live here, in the present
    if (order.status !== "PLACED") {
      throw new ConflictError(`cannot pay an order in status ${order.status}`);
    }
    if (Math.abs(cmd.amount - order.total()) > 0.001) {
      throw new ValidationError(`amount ${cmd.amount} != order total ${order.total()}`);
    }

    // 3. PRODUCE the new event(s) — past tense, immutable facts
    const newEvents: OrderEvent[] = [
      { type: "OrderPaid", amount: cmd.amount, method: cmd.method },
    ];

    // 4. APPEND atomically with optimistic concurrency on `version`.
    // If another writer appended since we loaded, expectedVersion won't match -> retry.
    await this.store.append(cmd.orderId, newEvents, /* expectedVersion */ version);
  }
}
```

The `expectedVersion` check is the consistency mechanism. The aggregate is the **consistency boundary**: all the rules that must hold together — an order cannot be paid twice, the paid amount must equal the total — are enforced inside a single aggregate against a single stream, with the version acting as an optimistic lock. If two `PayOrder` commands race, both load version 6, both try to append at expected-version 6, and exactly one wins; the loser gets a version conflict and retries (re-loading, re-validating, and now correctly seeing status `PAID` and rejecting the double payment). This is how event sourcing gives you **strong consistency on the write side** even though the read side will be eventually consistent. That distinction — strong writes, eventual reads — is the single most important thing to internalize, and we will keep returning to it.

The retry on a concurrency conflict is worth showing explicitly, because juniors often forget it and end up surfacing a 409 to the user for a conflict the system could have resolved itself:

```typescript
async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      // A concurrency error means someone appended between our load and append.
      // Re-running re-loads the now-newer state and re-validates against it.
      if (err instanceof ConcurrencyError && i < attempts - 1) continue;
      throw err;
    }
  }
  throw new Error("unreachable");
}

// usage: withRetry(() => payHandler.handle(cmd))
```

Notice the retry is *safe to loop* precisely because the command re-validates each time. The second attempt to pay an already-paid order does not "retry the payment" — it re-loads, sees `status: PAID`, and rejects with a clean business error. Optimistic concurrency plus re-validation gives you correctness without a single explicit lock, which is exactly what you want in a service that may be running dozens of replicas.

How big should an aggregate be? This is the most consequential modeling decision in event sourcing, and it is a trade-off, not a formula. The aggregate is the unit of *transactional consistency* — everything inside it is updated atomically, everything outside it is eventually consistent. Make the aggregate too *large* (an entire customer with all their orders as one stream) and you get contention (every order touch serializes on the customer version) plus unbounded stream growth. Make it too *small* (a single line item as its own aggregate) and you lose the ability to enforce cross-item invariants transactionally ("the order total must equal the sum of line items"). The ShopFast `Order` is a good aggregate because the invariants that must hold *together* — total equals sum of lines, cannot pay twice, cannot ship before paying — all live inside one order and rarely contend across orders. When two pieces of state need to change atomically, they belong in the same aggregate; when they only need to agree *eventually*, they belong in different aggregates connected by events. That single rule — *transactional inside, eventual between* — is how you draw aggregate boundaries, and it is the same reasoning used to draw service boundaries in [Service Boundaries with Domain-Driven Design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design).

## The event store: append-only, load-by-snapshot

The event store is the database of an event-sourced service. Its API is tiny on purpose: `append(streamId, events, expectedVersion)` and `load(streamId)`. There is no `UPDATE` and no `DELETE` in the happy path; the table is append-only. Here is a minimal Postgres-backed event store, which is how most teams start (you do not need a specialized event-store database to begin).

```sql
CREATE TABLE order_events (
  stream_id    TEXT     NOT NULL,
  seq          BIGINT   NOT NULL,          -- per-stream version, 1,2,3,...
  event_type   TEXT     NOT NULL,
  data         JSONB    NOT NULL,
  event_version INT     NOT NULL DEFAULT 1, -- schema version for upcasting (see below)
  occurred_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (stream_id, seq)             -- the PK enforces the optimistic lock for free
);
```

The composite primary key `(stream_id, seq)` does the heavy lifting: appending at an already-used `seq` raises a unique-violation, which *is* your optimistic-concurrency failure. No explicit locking required. Append:

```typescript
async append(streamId: string, events: OrderEvent[], expectedVersion: number): Promise<void> {
  const client = await this.pool.connect();
  try {
    await client.query("BEGIN");
    let seq = expectedVersion;
    for (const e of events) {
      seq++;
      // INSERT fails on (stream_id, seq) PK collision if someone else wrote first.
      await client.query(
        `INSERT INTO order_events (stream_id, seq, event_type, data, event_version)
         VALUES ($1, $2, $3, $4, $5)`,
        [streamId, seq, e.type, JSON.stringify(stripType(e)), schemaVersionFor(e)]
      );
    }
    await client.query("COMMIT");
  } catch (err) {
    await client.query("ROLLBACK");
    if (isUniqueViolation(err)) throw new ConcurrencyError(streamId, expectedVersion);
    throw err;
  } finally {
    client.release();
  }
}
```

Loading is the part that scales badly without help. A naive `load` selects every row for a stream ordered by `seq` and folds them all. For a young order with six events that is instant. For a long-lived aggregate — a customer's loyalty account, a warehouse bin, a bank account — that has accumulated thousands of events, you are reading and folding the entire history *on every single command*. That is the first place event sourcing bites you, and the fix is **snapshots**.

![A vertical stack showing an aggregate load that reads the latest snapshot then folds only the events after it instead of the full history](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-6.webp)

A snapshot is a serialized copy of the aggregate's state at a known sequence number. To load, you read the latest snapshot (say at seq 9,800), deserialize it into the aggregate, then fold only the events *after* the snapshot (seq 9,801 onward). You replay a few dozen events instead of ten thousand.

```typescript
async load(streamId: string): Promise<{ order: Order; version: number }> {
  // 1. read latest snapshot if any
  const snap = await this.readLatestSnapshot(streamId); // { state, seq } | null
  const order = snap ? Order.fromSnapshot(snap.state) : new Order();
  const fromSeq = snap ? snap.seq : 0;

  // 2. fold only the tail
  const tail = await this.pool.query(
    `SELECT event_type, data, event_version, seq FROM order_events
     WHERE stream_id = $1 AND seq > $2 ORDER BY seq`,
    [streamId, fromSeq]
  );
  for (const row of tail.rows) {
    order.apply(upcast(deserialize(row))); // upcast handles old schema versions, see below
  }
  return { order, version: order.version };
}

// Write a snapshot every N events. Cheap insurance against unbounded replay cost.
async maybeSnapshot(streamId: string, order: Order): Promise<void> {
  if (order.version % 100 === 0) {
    await this.writeSnapshot(streamId, order.version, order.toSnapshot());
  }
}
```

Snapshots are a pure optimization: they never change the truth (the events are still the source of truth), they only cache a fold result so you do not recompute it. You can delete every snapshot and the system is still correct, just slower. That property — snapshots are a derived cache, never the truth — is what keeps event sourcing honest. Let us put numbers on why they matter.

#### Worked example: replay cost with and without snapshots

Take a long-lived ShopFast aggregate — a customer's loyalty account with **10,000 events** accumulated over three years. Suppose folding one event takes about 5 microseconds of CPU, and reading it from Postgres averages 50 microseconds amortized (batched fetch). Loading the whole stream:

- Read: 10,000 events × 50 µs = **500 ms** just in I/O.
- Fold: 10,000 × 5 µs = **50 ms** of CPU.
- Total per load: **~550 ms** — and this happens on *every command* against that account.

At 200 commands/second against hot loyalty accounts, that is 110 seconds of work per wall-clock second; you would need a small fleet just to load aggregates. Now add a snapshot every 100 events. The latest snapshot is at seq 9,900 or later, so a load reads at most one snapshot (~1 ms to deserialize) plus the ≤100-event tail:

- Snapshot read + deserialize: **~1 ms**.
- Read tail: ≤100 × 50 µs = **5 ms**.
- Fold tail: ≤100 × 5 µs = **0.5 ms**.
- Total per load: **~6.5 ms**.

That is an **~85× speedup** (550 ms → 6.5 ms), and the saving grows linearly as the stream gets longer — a state-stored CRUD row would have stayed at a flat ~1 ms forever. Snapshotting does not eliminate the gap with CRUD; it bounds the replay cost so the gap stops widening. The lesson: choose the snapshot interval N so the tail fold is comfortably under your latency budget. For a write-budget of 20 ms p99 and a 5 µs fold, even N = 1,000 is fine; pick N = 100 if your aggregates are write-hot and you want headroom.

## CQRS: split the write model from the read models

Event sourcing solves "I lost the history." It does *not* by itself solve "I need to query this." Try answering "show me all orders shipping to California, sorted by value, that contain SKU-19" against a pile of event streams. You cannot. There is no table to `SELECT` from; there are only per-aggregate streams you can fold one at a time. Querying across aggregates by anything other than their id is, in raw event sourcing, impossible. This is the famous "you can't just `SELECT`" tax, and the answer to it is CQRS.

**CQRS** — Command Query Responsibility Segregation — is the simple, powerful idea that the model you use to *change* data does not have to be the model you use to *read* it. You split your service into two sides:

- The **command side** (the write model): the event-sourced aggregate we just built. It is optimized for *consistency and validation*. Its only job is to accept commands, enforce invariants, and append events. It is never queried by users.
- The **query side** (one or more read models): denormalized, query-shaped views, each tuned for a specific question. They are *derived from the events* and optimized for *fast reads*. They are never used to make decisions about validity.

![A two-column comparison contrasting one shared model serving both reads and writes against CQRS with a write model feeding multiple denormalized read models](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-3.webp)

Crucially, **CQRS and event sourcing are separable**. You can do CQRS without event sourcing (a write database and a separate read replica/materialized view, kept in sync by CDC — see [Change Data Capture and the Outbox Pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern)). You can do event sourcing without CQRS (an event-sourced aggregate whose only read is load-by-id). But they **pair beautifully**, because event sourcing already produces a clean stream of facts, and those facts are exactly the input a read model needs to keep itself up to date. The events that the write side appends become the feed that builds the read side. That is why this post treats them together while insisting they are two patterns, not one.

### Read models are denormalized and disposable

A read model (also called a *projection* or *materialized view*) is just a regular table — or document, or search index, or cache — shaped for one query, kept current by *consuming the event stream*. For ShopFast we build two:

1. **Order history view** — for the customer-facing "my orders" page. A flat row per order with status, total, and timestamps.
2. **Fulfillment dashboard** — for the warehouse, showing only `PAID` orders not yet shipped, with item counts, grouped by carrier.

These two read models are derived from the *same* event stream but shaped completely differently. The order-history table is keyed by customer; the fulfillment table is keyed by status and carrier and only holds a slice of orders. Neither is "the truth" — they are caches you can throw away and rebuild from the events at any time. That disposability is the superpower: if a product manager asks for a brand-new query next quarter, you add a brand-new read model and replay the events into it. You do not migrate anything. You do not touch the write side.

![A branching dataflow showing one append-only Order event stream fanning out to an order-history projection, a fulfillment dashboard, a revenue rollup, and a search index](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-4.webp)

### The projector that builds a read model from events

A **projector** (or projection handler) is a consumer that reads the event stream in order and applies each event to the read model. It is structurally similar to the aggregate's `apply`, but the target is a query-shaped table, not the aggregate's in-memory state.

```typescript
// Fulfillment-dashboard projector. Consumes Order events in stream order,
// maintains a denormalized table the warehouse UI queries directly.
class FulfillmentProjector {
  constructor(private db: ReadModelDb) {}

  async on(streamId: string, e: OrderEvent, seq: number): Promise<void> {
    switch (e.type) {
      case "OrderPaid":
        // becomes visible to the warehouse the moment it is paid
        await this.db.upsert("fulfillment_queue", {
          order_id: streamId,
          status: "AWAITING_PICK",
          item_count: await this.lineCount(streamId),
          paid_at: new Date(),
        });
        break;

      case "OrderShipped":
        // leaves the queue once shipped — denormalized table only holds open work
        await this.db.delete("fulfillment_queue", { order_id: streamId });
        break;

      // ItemAdded/AddressChanged/OrderPlaced don't affect THIS view -> ignored.
      // A different projector (order history) handles them. One stream, many views.
    }
    // record progress so we can resume and detect lag (see consistency section)
    await this.db.setCheckpoint("fulfillment", seq);
  }
}
```

Two details make this production-grade. First, **each projector ignores events it does not care about** — the fulfillment view ignores `ItemAdded`, while the order-history view consumes it. One stream, many opinions about it. Second, **the projector records a checkpoint** (the last `seq` it processed). The checkpoint is how it resumes after a crash without re-processing, and — critically — how you *measure* whether a projection has fallen behind, which we will use to monitor the consistency gap.

A subtle but vital rule: **projectors must be idempotent**. The event stream delivers at-least-once in most setups, so a projector may see the same event twice (after a crash, a redeploy, a rebalance). `upsert` and "delete if present" are naturally idempotent; a blind `count = count + 1` is not. This is the same idempotency discipline the whole event-driven world depends on — see [Idempotency and Deduplication: Making At-Least-Once Safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) for the patterns (dedup keys, idempotent upserts, the inbox table). The cleanest way to guarantee idempotence is to compare the incoming event's `seq` against the projection's stored checkpoint and skip anything already applied — then even an exactly-once-effect projection falls out of an at-least-once delivery.

### Multiple read models from one stream

The second ShopFast read model — **order history** — consumes the *same* event stream but builds a completely different shape. Where the fulfillment projector only cares about `OrderPaid`/`OrderShipped` and keeps a tiny open-work table, the order-history projector cares about almost every event and maintains a customer-facing row per order:

```typescript
// Order-history projector. Same stream as fulfillment, opposite shape:
// keyed by customer, holds ALL orders, accumulates total as items are added.
class OrderHistoryProjector {
  constructor(private db: ReadModelDb) {}

  async on(streamId: string, e: OrderEvent, seq: number): Promise<void> {
    if (await this.alreadyApplied("order_history", seq)) return; // idempotency by checkpoint

    switch (e.type) {
      case "OrderPlaced":
        await this.db.upsert("order_history", {
          order_id: streamId, customer_id: e.customerId,
          status: "PLACED", total: 0, placed_at: new Date(),
        });
        break;
      case "ItemAdded":
        // denormalize the join: keep a running total so the read needs no join later
        await this.db.increment("order_history", { order_id: streamId },
                                { total: e.qty * e.unitPrice });
        break;
      case "OrderPaid":    await this.set(streamId, { status: "PAID" }); break;
      case "OrderShipped": await this.set(streamId, { status: "SHIPPED" }); break;
      case "AddressChanged": /* history view doesn't surface the zip -> ignore */ break;
    }
    await this.db.setCheckpoint("order_history", seq);
  }
}
```

Now contrast the two query paths they enable. The customer's "my orders" page runs `SELECT order_id, status, total, placed_at FROM order_history WHERE customer_id = ? ORDER BY placed_at DESC` — one index scan, no joins, because the projector pre-computed the total. The warehouse runs `SELECT order_id, item_count FROM fulfillment_queue WHERE status = 'AWAITING_PICK'` — a different table, a different key, a different shape. **Two read models, one event stream, zero coordination.** Adding a third (say, a tax-reporting rollup) means writing a third projector and replaying; it touches neither of the first two and never touches the write side. This is the property that makes CQRS-on-event-sourcing so powerful for evolving systems: the cost of a new query is bounded and isolated.

How does a projector actually *receive* the events? Two delivery styles, and the choice has real consequences:

- **Polling the store.** The projector periodically asks "any events with `seq` greater than my checkpoint?" Simple, no extra infrastructure, works directly against the Postgres event table. The downside is the poll interval *is* your minimum lag floor — a 50 ms poll means at least 0–50 ms of staleness baseline. Fine for back-office views, a bit coarse for customer-facing ones.
- **Pushing / streaming.** The store (or a CDC relay tailing it) pushes new events to subscribers the moment they land, via a log like Kafka or a database change stream. Lower latency (single-digit milliseconds), at the cost of running and operating that streaming layer. This is where the [Change Data Capture and the Outbox Pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) machinery earns its place: CDC turns the append into a push without the projector polling.

A practical default: poll for back-office projections (fulfillment, reporting) where seconds of lag are fine and simplicity wins, push for the handful of customer-facing projections where you want sub-100ms freshness. And whichever you choose, the **checkpoint is sacred** — it is both how the projector resumes exactly where it left off and how you measure lag. Store it transactionally *with* the read-model write (same transaction) so a crash can never leave the checkpoint ahead of the data it claims to have applied; otherwise you silently skip events and the read model is quietly wrong forever, which is far worse than being briefly stale.

### Command path versus query path

Step back and look at the whole shape. A *write* (command) and a *read* (query) take completely different routes through the service and never share storage.

![A branching diagram showing the command path flowing through the aggregate to the event store while the query path reads a separate denormalized read model](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-7.webp)

The command path: client sends `PlaceOrder` → command handler loads the aggregate → validates → appends events to the event store. Strongly consistent, slow-ish, low-volume (writes are rarer than reads).

The query path: client sends `GET /orders/A7F2` → the API reads straight from the denormalized read model. No aggregate load, no fold, no validation. Fast, high-volume, horizontally scalable (read models can be replicated freely because they are disposable caches).

In between sits the projector, asynchronously turning appended events into updated read models. And *that gap* — the time between "the event is appended" and "the read model reflects it" — is eventual consistency, the cost you pay for this separation. We tackle it head-on next.

### Events as the natural feed for cross-service integration

There is a bonus that event sourcing hands you almost by accident, and it is a big reason the pattern fits microservices specifically. The events you append for your own write model are *already* the facts other services want to react to. The payment service does not need a bespoke "tell me when an order is paid" API; it can subscribe to the `OrderPaid` events that the order service is already producing. The same stream that builds your internal read models becomes the integration backbone between services — this is the choreography we cover in [Event-Driven Microservices: Choreography vs Orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration).

But there is a subtlety that catches teams, and it is worth stating plainly: **the events you store for your aggregate are often not the events you want to publish to the outside world.** Internal events are fine-grained and tied to your domain model (`ItemAdded`, `AddressChanged`); external consumers usually want coarser, more stable, intentionally-designed *integration events* (`OrderConfirmed`, `OrderReadyToShip`). Coupling other services directly to your internal event schema recreates the shared-database anti-pattern in event form — now you cannot evolve your internal events without breaking three other teams. The discipline is to keep two vocabularies: internal events for your own fold, and a curated set of published integration events that form a stable contract.

That publishing step also has a reliability trap. Appending the event to your store and publishing it to the message bus are two separate operations; if the process dies between them, you have a stored event nobody heard about. You do not solve this with a distributed transaction across the store and the broker — you solve it with the **transactional outbox**, which we cross-link as forward reading: [The Transactional Outbox and Reliable Event Publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing). In an event-sourced service the outbox often *is* the event store — your appended events are the outbox, and a relay tails them to the bus. That elegant collapse (the audit log and the outbox being the same append) is one of the quiet reasons event sourcing and event-driven microservices fit so well together.

## The eventual-consistency gap (and how the UI survives it)

In a plain CRUD service, the moment your `UPDATE` commits, the very next `SELECT` sees the new value. Read-your-own-writes is free. In CQRS, the write commits to the event store, returns `200 OK`, and *then* — milliseconds later, usually, but sometimes seconds — a projector catches up and updates the read model. For that window, a query against the read model returns *stale* data. The customer clicks "Pay," sees a spinner resolve to success, immediately lands on the order page, and the order still says "Awaiting payment." Nothing is broken. The read model just has not caught up yet.

![A six-event timeline tracing a paid order from event append at T+0 through command success, projector lag, and the read model catching up at T+120ms](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-8.webp)

This is not a bug you fix; it is a property you design around. The mechanics of *why* eventual consistency exists and the full spectrum from linearizable to eventual are covered in [Consistency Models: From Linearizable to Eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual); here we focus on the *practitioner's* job — keeping the UI honest during the gap. There are four field-tested techniques, in rough order of cost:

1. **Return the new state from the command itself.** The command handler already computed the post-command aggregate state to validate it. Return that in the `200` response. The client renders the result of the *command* it just issued, not a read-model query. Cheapest, covers the most common "I just did X and want to see X" case, and almost always enough.

2. **Optimistic UI / client-side echo.** The browser knows it issued a payment; it locally marks the order paid and renders that until the next refresh, regardless of what the read model says. Pairs well with #1.

3. **Read-your-writes via a version token.** The command returns the new aggregate version (e.g. `version: 7`). The client passes `If-Projection-At-Least: 7` on the follow-up query; the read API blocks (briefly) or returns a "still catching up" flag until the projection's checkpoint reaches 7. This gives genuine read-your-writes at the cost of a little query latency and more plumbing. Use it only where staleness is genuinely user-visible and confusing.

4. **Tell the truth in the UI.** Show "Payment processing — your order will update shortly." Sometimes the most honest and cheapest fix is to stop pretending the read is instant.

The mistake juniors make is reaching for #3 (the expensive one) everywhere, or worse, abandoning CQRS the first time a tester sees a stale page. The senior move is to use #1 by default and reserve #3 for the handful of screens where stale reads actually confuse users. Let us quantify the window so you know what you are designing around.

#### Worked example: the projection lag window and what the UI shows

ShopFast's `OrderPaid` flow under normal load:

- **T+0 ms** — command handler appends `OrderPaid` to the event store (transaction commits).
- **T+5 ms** — command returns `200 OK` to the client, carrying the post-command state (`status: PAID`) via technique #1.
- **T+40 ms** — the projector polls/streams the new event off the log (poll interval averages this; with push it is lower).
- **T+90 ms** — projector processes the event and writes the fulfillment-queue row.
- **T+120 ms** — read model fully consistent; the warehouse dashboard now shows the order.

So the **stale window is ~120 ms** for the *fulfillment dashboard* (the warehouse), but **0 ms for the customer**, because the customer's "thank you" page renders the state returned by the command at T+5 ms. The customer never sees a stale read; the only consumer that experiences the 120 ms gap is the warehouse dashboard, where a 120 ms delay before an order appears in the pick queue is completely irrelevant. This is the key insight: **measure the gap per consumer**, not globally. The same 120 ms is invisible to one consumer and harmless to the other. Now stress it: under a deploy or a poison-message backlog the projector can fall *minutes* behind. The fulfillment dashboard would then be minutes stale — still tolerable for a warehouse (orders queue up; nobody is harmed). But if you had naively pointed the *customer's* "did my payment go through?" page at the read model, that same minutes-long lag becomes a flood of "I was charged but my order says unpaid" support tickets. The architecture decision and the UX decision are the same decision.

## Trade-offs: when these patterns earn their complexity

Now the section that matters most, because the dominant failure mode with these patterns is not implementing them wrong — it is implementing them *at all* when you should not have. Let us be explicit about what each pattern gains, what it costs, and when it wins.

![A decision matrix comparing CRUD, event-sourced, and CQRS approaches across auditability, query flexibility, complexity cost, consistency, and deletion difficulty](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-5.webp)

| Property | CRUD (state-stored) | Event-sourced | Event-sourced + CQRS |
|---|---|---|---|
| **Auditability / history** | None — `UPDATE` destroys the past | Complete — every change is a stored fact | Complete |
| **Temporal queries ("state on March 1")** | Impossible without extra tables | Trivial — fold up to that timestamp | Trivial |
| **Query flexibility** | One schema; joins at read time | Poor — only load-by-id | Excellent — a read model per query |
| **Write consistency** | Strong (DB transaction) | Strong (aggregate + optimistic lock) | Strong write side |
| **Read consistency** | Strong (read-your-writes free) | Strong (you fold current state) | **Eventual** — projection lag |
| **Implementation complexity** | Low | High | Highest |
| **Delete / GDPR erasure** | Easy — `DELETE` the row | **Hard** — events are immutable | Hard |
| **Schema evolution** | `ALTER TABLE` | Hard — must upcast old events forever | Hard + rebuild projections |
| **Operational burden** | Low | Medium-high (snapshots, replay) | High (projectors, rebuilds, lag monitoring) |

Read that table as a series of bills. Event sourcing's bill is paid in schema evolution, deletion difficulty, and operational machinery (snapshots, replay tooling). CQRS adds eventual consistency and a projector to operate per read model. CRUD's bill is paid in lost history and rigid querying. The pattern wins when the thing in the "gains" column is *worth more to your business than the bill*, and the brutal truth is that for most services it is not.

When event sourcing **earns it**:

- **Audit is a hard requirement, not a nice-to-have.** Ledgers, payments, trading, healthcare records, anything regulated. If "who changed what, when, and to what" is a legal obligation, the audit log being your *source of truth* (rather than a fragile side-channel) is a profound simplification.
- **Temporal/historical queries are part of the product.** "Show this account as of the statement date," "replay this game's moves," "what did the cart contain when the price changed."
- **The domain is genuinely event-shaped.** Things that are naturally a sequence of facts — an order's lifecycle, a shipment's journey, a workflow — model more naturally as events than as a mutable row.
- **You need to derive new views from old history.** If "we'll want to slice this data in ways we can't predict yet" is real, replayable events let you build views retroactively.

When CQRS **earns it** (independently): when your read and write workloads have genuinely conflicting requirements — extreme read fan-out, many distinct query shapes, read volume orders of magnitude above writes — *and* your users can tolerate eventual reads on the affected screens.

When you should **not** use either — and this is most services:

- **Simple CRUD with no audit requirement.** A `users` table, a `products` catalog, a `settings` store. Use a boring relational table. Storing `UserEmailChanged` events for a profile screen is pure ceremony that buys you a GDPR headache.
- **Reads and writes have the same shape and modest volume.** If one normalized schema serves both fine, CQRS is two systems doing one system's job.
- **The team is small and junior to distributed systems.** These patterns demand fluency in eventual consistency, idempotency, and replay. Adopting them while still learning service basics is how you build a distributed monolith you can't operate.

![A decision tree that routes a service to plain CRUD unless audit or history is required, then to event sourcing, and only to event sourcing plus CQRS when complex reads and eventual consistency are both acceptable](/imgs/blogs/event-sourcing-and-cqrs-in-microservices-9.webp)

The default answer is the top branch of that tree: **plain CRUD, store the state.** You walk down toward event sourcing only when audit or temporal history is a genuine requirement, and toward CQRS only when complex varied reads *and* tolerable eventual consistency are both true. A senior says no to these patterns far more often than yes, and says it without apology. (For the broader trap of over-coupling services and rebuilding a monolith you cannot deploy independently, see [Database per Service: The Rule That Defines Microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices).)

## Optimization and making it production-grade

Assume you have correctly decided event sourcing earns its place — a ShopFast `Order` aggregate with an audit mandate and a fulfillment read model. The naive version works in a demo and falls over in production. Here is where the real bottlenecks are and how to measure each win.

**1. Snapshots (we covered the math; here's the policy).** Snapshot every N events; tune N so the tail-fold stays under your write-latency budget. Measure with the load-latency p99. In the worked example we drove a 10k-event load from 550 ms to 6.5 ms. The win is measurable as p99 aggregate-load latency; alert if it creeps above your budget (it means an aggregate's stream is growing faster than your snapshot interval — usually a sign that aggregate is too coarse and should be split).

**2. Projection throughput and lag.** A projector that processes events one-at-a-time, synchronously, with a round-trip to the read-model DB per event, tops out around a few hundred events/second. Three fixes, each with a number:

```typescript
// Naive: one event, one DB round-trip. ~300 events/sec at 3ms per round-trip.
for (const e of batch) await this.handle(e);

// Optimized: batch the read-model writes in one transaction.
// 500 events, one round-trip -> ~50,000 events/sec; the bottleneck moves to the log read.
await this.db.transaction(async (tx) => {
  for (const e of batch) this.stage(tx, e);  // accumulate writes
  await tx.flush();                          // single batched commit
});
await this.db.setCheckpoint("fulfillment", batch[batch.length - 1].seq);
```

Batching read-model writes typically moves a projector from hundreds to tens of thousands of events/second — an ~100× throughput gain — by amortizing the per-event DB round-trip. Measure with **projection lag**: `latest_event_seq − projection_checkpoint_seq`, exported as a gauge. Alert when lag exceeds a threshold for your slowest-tolerable read model (seconds for a customer-facing view, minutes for a back-office dashboard).

**3. Read-model denormalization is the point — lean into it.** The fulfillment query should be `SELECT * FROM fulfillment_queue WHERE status = 'AWAITING_PICK' ORDER BY paid_at` — a single index scan, no joins, because the projector already did the joining at write time. Compare to the CRUD world where the same query joins `orders`, `order_items`, and `customers` at read time. You have traded a one-time write cost (paid by the projector, off the user's critical path) for a fast, join-free read. A read model can also be a *Redis cache*, an *Elasticsearch index*, or a *materialized aggregate* — whatever the query wants. One write side, many specialized read stores.

**4. Partition by aggregate id for parallelism.** Because each aggregate's stream is independent, projectors can shard by `stream_id` and run in parallel with no cross-stream ordering concerns — as long as events *within* a stream stay ordered. This is exactly the ordering-and-partitioning guarantee discussed in [Message Ordering and Partitioning Guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees): partition by aggregate id, preserve per-partition order, parallelize across partitions.

#### Worked example: storage growth of an event log versus stored state

A common objection: "won't storing every event blow up my storage?" Let us check with ShopFast numbers. Suppose 1,000,000 orders/year, each with on average **8 events** over its lifetime, each event averaging **400 bytes** of JSON plus ~100 bytes of row overhead (≈500 bytes stored).

- **Event-sourced storage/year**: 1,000,000 × 8 × 500 bytes = **4.0 GB/year**.
- **State-stored (CRUD) storage**: 1,000,000 orders × ~1.5 KB/row (denormalized current state) = **1.5 GB**, roughly *flat* (rows are updated in place, not appended).

So after one year the event log is ~2.7× the stored-state size, after three years ~8× (12 GB vs 1.5 GB). That sounds bad until you price it: 12 GB of SSD is on the order of **a few dollars a month**. The "storage explosion" fear is almost always a non-issue at human scale; storage is the cheapest thing in the building. Where it *does* matter is hot-path replay cost (handled by snapshots) and very high-frequency aggregates (an IoT sensor emitting events per second — millions of events per stream — where you genuinely need stream archival/truncation strategies). For a business-event domain like orders, the storage cost of full history is a rounding error against its audit value. Compute the number for *your* event rate before either dismissing the cost or fearing it.

## Stress test: what breaks, and how you survive it

A design is only as good as its behavior under stress. Here are the three failures that *will* happen to an event-sourced CQRS service in production, with the survival plan for each.

### Stress 1: "The event schema must change"

This is the one that haunts teams, because **events are immutable and live forever**. Three years from now you will still be folding `OrderPlaced` events written today, by code you have rewritten five times. You cannot run an `ALTER TABLE` against history. Suppose the product adds multi-currency and `OrderPlaced` must now carry an `currency` field it never had. The old events have no `currency`. What happens when you fold a v1 event with v2 code?

The wrong answer is to *mutate old events* (rewriting history breaks the entire premise and your audit guarantee). The right answer is **upcasting**: a function that reads an old-version event and transforms it, *in memory at load time*, into the shape current code expects. The stored event is never touched; you adapt it on the way in.

```typescript
// Each event carries its schema version (the event_version column).
// Upcasters form a chain: v1 -> v2 -> v3, applied in order until current.
function upcast(raw: { type: string; data: any; event_version: number }): OrderEvent {
  let { data, event_version } = raw;

  if (raw.type === "OrderPlaced") {
    // v1 -> v2: currency was added; default legacy orders to USD (a documented fact:
    // ShopFast was USD-only before this date, so USD is the CORRECT historical value).
    if (event_version < 2) {
      data = { ...data, currency: "USD" };
      event_version = 2;
    }
    // v2 -> v3: customerId widened from int to string customer key
    if (event_version < 3) {
      data = { ...data, customerId: String(data.customerId) };
      event_version = 3;
    }
  }
  return { type: raw.type, ...data } as OrderEvent;
}
```

The rules that keep schema evolution sane:

- **Only ever add optional fields or new event types.** Never remove a field's meaning, never repurpose a type. Old events must remain interpretable.
- **Write an upcaster for every breaking shape change**, chained by version. Loading always upcasts to current; appending always writes current version.
- **Default with a *correct historical fact*, not a guess.** Defaulting legacy `OrderPlaced` to `currency: USD` is right *because ShopFast was genuinely USD-only then*. If you don't know the historical value, that's a sign the field shouldn't be back-filled.
- **Test upcasters against real historical event samples in CI**, because a broken upcaster means aggregates that *cannot load* — a total outage for those streams.

This is real, permanent work, and it is the single biggest hidden cost of event sourcing. Budget for it honestly when deciding whether the pattern earns its place.

### Stress 1b: "A customer invokes their right to be forgotten"

Here is the constraint that catches teams off guard the first time legal walks over: **events are immutable, but GDPR (and CCPA, and similar regimes) gives a person the right to have their personal data erased.** If a customer's name, email, and address are baked into `OrderPlaced` and `AddressChanged` events that you have sworn never to mutate, how do you delete them? "Just don't store personal data in events" is the glib answer, and it is partly right and partly impossible — an order genuinely involves a customer.

There is no painless solution; there are three workable ones, in increasing order of upfront cost and decreasing order of operational pain:

1. **Crypto-shredding (the standard answer).** Encrypt every piece of personal data inside an event with a per-subject key (one key per customer), and store the keys in a *separate*, mutable key vault. The events stay immutable and append-only — you never touch them. To "forget" a customer, you delete *their key*. The encrypted bytes remain in the log but become permanently undecryptable, which most regulators accept as erasure. This preserves the audit trail's structure (you can still see *that* an order happened and its non-personal facts) while rendering the personal payload unrecoverable. It is the de facto standard for event-sourced systems under GDPR.

```typescript
// On append: encrypt PII fields with the subject's key (fetched from the vault).
function encryptPii(event: OrderEvent, subjectKey: Buffer): OrderEvent {
  if (event.type === "AddressChanged") {
    return { ...event, zip: encrypt(event.zip, subjectKey) }; // ciphertext stored in the log
  }
  return event;
}

// On load: decrypt with the key — UNLESS the key has been shredded.
function decryptPii(event: OrderEvent, subjectKey: Buffer | null): OrderEvent {
  if (event.type === "AddressChanged") {
    // key deleted -> field is permanently unreadable -> render as "[redacted]"
    return { ...event, zip: subjectKey ? decrypt(event.zip, subjectKey) : "REDACTED" };
  }
  return event;
}

// "Right to be forgotten" = delete one row from the key vault. The log is never touched.
async function forgetSubject(subjectId: string) {
  await keyVault.delete(subjectId); // all that customer's PII is now undecryptable forever
}
```

2. **Keep PII out of events entirely; reference it by id.** Store only `customerId` in events; keep the actual name/email/address in a *separate, mutable* customer service that supports normal `DELETE`. The event log holds references, not personal data. This is cleaner but requires discipline forever (one careless engineer who logs an email into an event reopens the problem) and means your audit trail alone cannot reconstruct who the customer was — which may itself be a problem for some audits.

3. **Tombstone-and-rewrite (the nuclear option).** Some specialized event stores support rewriting or truncating a stream to physically excise events. This *does* break immutability and must be tightly controlled and itself audited (who erased what, when, under what legal request). Use only when crypto-shredding is insufficient for a regulator.

The honest takeaway: **deletion is genuinely hard in event sourcing, and it is a first-class design input, not an afterthought.** If your domain carries heavy personal data *and* has frequent erasure requests *and* does not have a strong audit mandate, that combination is a real argument *against* event sourcing — the deletion pain may outweigh the audit benefit. Decide this before you append the first event, because retrofitting crypto-shredding onto a log full of plaintext PII is a brutal migration.

### Stress 2: "Rebuild a read model from 100M events"

A product manager wants a new "orders by region" report. The beauty of CQRS: you do not migrate, you *project*. Write a new projector, point it at the start of the event stream, and let it fold all of history into the new read model. But the history is **100 million events**. A naive single-threaded rebuild at the unoptimized ~300 events/second would take **~93 hours** — unacceptable. The production rebuild playbook:

1. **Build into a new table, in the background, while the old read model keeps serving.** Never rebuild in place; you'd serve garbage during the rebuild. Build `orders_by_region_v2`, then atomically swap a view/alias when caught up.
2. **Parallelize by aggregate-id partition.** 100M events across, say, 64 partitions = ~1.6M events each, run 64 projector workers concurrently.
3. **Batch the writes** (the ~100× win from the optimization section). At a batched ~50,000 events/sec per worker, one partition of 1.6M events takes ~32 seconds; 64 in parallel finish in **under a minute of wall-clock** (bounded now by read throughput from the log, not the projector).
4. **Catch up the tail.** Once the bulk replay reaches near-current, switch the new projector to live-tailing the stream, let it close the last few seconds of lag, then swap.

The fact that you *can* rebuild any read model from scratch — cheaply, in the background, without touching the write side — is the headline benefit of pairing CQRS with event sourcing. A new query is a new projector plus a replay, not a data migration with downtime.

### Stress 3: "A projection falls behind"

A projector consumes events more slowly than they are produced (a slow read-model DB, a deploy pause, a poison event it keeps crashing on). Lag grows. Reads get staler. If unhandled, the gap can reach minutes or hours.

- **Detect it.** The checkpoint gauge `latest_seq − checkpoint_seq` is your early warning. Alert per read model at the right threshold. (This is exactly the kind of thing covered in the resilience and observability tracks — a projector falling behind is a form of backpressure; see [Event-Driven Microservices: Choreography vs Orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) for the broader event-flow failure modes.)
- **Isolate poison events.** If one event crashes the projector, it blocks the whole stream behind it. Route repeat-failing events to a dead-letter store after N attempts and *keep going*, then investigate the poison event out-of-band. Never let one bad event freeze a projection forever.
- **Scale projector throughput.** Batch writes, partition by aggregate id, add workers. A read model is disposable, so you can spin up extra rebuild capacity without risk.
- **Degrade the UI honestly.** While the customer-facing projection is behind, fall back to the command-returned state (technique #1) or display a "syncing" indicator rather than confidently showing stale numbers.

The deeper point under all three stress tests: in an event-sourced CQRS service the read side can fail, lag, or be rebuilt without ever risking the write side, because the write side's truth is the immutable log and the read side is downstream of it. That asymmetry is a feature. A corrupted read model is a temporary inconvenience you replay away; a corrupted write model would be a catastrophe. By concentrating all the "this must be exactly right" pressure onto the small, strongly-consistent write side and treating every read model as a disposable, rebuildable cache, you have arranged the system so that the failure modes you will actually hit in production are the cheap ones to recover from. Designing for the failure you can afford, rather than the one you cannot, is the whole game. For the broader practitioner playbook on operating with eventual reads across an entire fleet — not just one service — see the forward-linked [Data Consistency and Eventual Consistency in Practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice).

## Case studies

**Banks and ledgers — event sourcing as the natural model.** Double-entry accounting *is* event sourcing, and predates computers. A bank balance is never stored as the truth; it is the fold of an append-only sequence of immutable transaction entries (credits and debits). You never `UPDATE` a balance — that would be fraud-enabling and would destroy the audit trail regulators require. You append a correcting entry. This is why ledger and core-banking systems gravitate to event sourcing: the domain was event-sourced before the term existed, the audit requirement is legally mandatory, and "what was the balance on the statement date" is a routine temporal query answered by folding entries up to that date. When your domain looks like this — immutable facts, audit as law, balance-as-fold — event sourcing is not over-engineering; it is the *honest* model, and trying to force it into CRUD is the mistake.

**A CQRS read-model rebuild as a feature, not a migration.** Systems built on event-sourced platforms (notably those using EventStoreDB or Axon-style frameworks, and the patterns popularized by Greg Young's CQRS work and Martin Fowler's writing on event sourcing) routinely treat "we need a new view of the data" as a *projector + replay*, not a schema migration. The concrete lesson: a team that needed a new analytics view on years of order history built a fresh projection, replayed the event log into it overnight while the existing read models kept serving traffic, and shipped the new view with zero migration, zero downtime, and zero risk to the write side. The same capability lets them *fix a buggy projection* by deleting the read model and replaying — the events are the truth, so the read model is always reconstructable. That disposability of read models is the benefit teams remember most fondly.

**The over-engineered-ES regret story — and it is the most important one.** Greg Young, who popularized CQRS, has publicly and repeatedly warned against applying event sourcing to whole systems rather than the few aggregates that need it; the common regret is teams that event-sourced *everything*. The pattern: a team, excited by the audit and replay benefits, made event sourcing the default for every service — user profiles, product catalog, settings, the lot. Two years later they were drowning. Every trivial CRUD screen now carried eventual-consistency bugs. Every schema change required upcasters for events nobody would ever audit. New engineers took weeks to become productive because the simplest "change a setting" operation went through commands, events, projectors, and read models. They had paid the full event-sourcing tax across the entire codebase to get audit value on the 5% of aggregates (orders, payments) that actually needed it. The fix was a painful partial retreat: they re-implemented the CRUD-shaped services as plain CRUD and kept event sourcing only for the ledger-like aggregates. The lesson is the one this whole post argues: **event sourcing and CQRS are per-aggregate decisions, never system-wide defaults.** The cost is real and it compounds; pay it only where the benefit is real too.

**Segment-style consolidation pressure (the meta-lesson).** Segment's well-documented monolith-to-microservices-and-partly-back journey is not about event sourcing specifically, but it teaches the same discipline: distributed-systems complexity (separate stores, separate deploys, separate consistency models) is a cost that must be justified per component, and when it isn't, consolidation is the senior move, not a defeat. Reaching for event sourcing and CQRS is the same kind of decision in miniature: add the complexity exactly where it pays, and resist the urge to make it the house style.

A fourth lesson worth naming because it is the one that quietly saves teams: **the projection-bug recovery.** A team running an event-sourced order service shipped a projector with a subtle bug — it double-counted line totals when an item was added then removed in the same session, so the order-history "total" column was wrong for thousands of orders. In a CRUD system this is a data-corruption incident: you write a one-off SQL fix, hope you got the logic right, and you have no way to verify against ground truth because the ground truth was overwritten. In the event-sourced system it was a non-incident: they fixed the projector code, deleted the corrupted `order_history` table, and replayed the events into a fresh one. The events — the actual `ItemAdded` and `ItemRemoved` facts — were never wrong; only the *derived* view was, and derived views are regenerable by definition. The whole recovery was a deploy plus a background replay, with the old (buggy) view still serving until the new one caught up. This is the benefit that does not show up in a feature list but shows up at 2am: **when your source of truth is immutable facts, almost every data bug becomes a replay, not a forensic reconstruction.**

## When to reach for this (and when not to)

Let me be decisive, because vagueness here is how teams get hurt.

**Reach for event sourcing when**, and essentially only when, at least one of these is *genuinely* true: audit/history is a hard (often legal) requirement; temporal "state-as-of" queries are part of the product; or the domain is so naturally a sequence of facts that modeling it as mutable state actively loses information you need. Orders, payments, ledgers, shipments, workflows, anything regulated — these are the homeland of event sourcing.

**Reach for CQRS when** read and write workloads have genuinely conflicting needs — many distinct query shapes, read volume vastly above write volume, or a need to derive new views from existing data — *and* the affected screens can tolerate eventual reads (use the command-return trick to keep the user's own writes instant).

**Do NOT reach for either — and this is the default — when** the service is ordinary CRUD with no audit mandate, when one normalized schema serves reads and writes well at your volume, or when your team is still building fluency in basic service patterns and would be adopting eventual consistency, idempotency, and replay all at once. For a `products` catalog, a `users` table, a `settings` store, a `notifications` log — store the state, run an `UPDATE`, and move on. The boring choice is almost always the correct one. A staff engineer's instinct is *not* "where can I apply this elegant pattern" but "can I justify the complexity bill for this specific aggregate, and if not, store the state." Say no by default; say yes precisely.

A final framing that helps: these patterns are not all-or-nothing for a service or a system. You can event-source *one* aggregate (the `Order`) inside a service while keeping its `customer_preferences` as a plain CRUD table in the same service. You can run CQRS for the order-history read while serving the settings page from the write store directly. Granularity is your friend — apply the expensive pattern at the smallest scope that captures the benefit.

## Key takeaways

1. **State equals fold over events.** Event sourcing stores immutable past-tense facts and derives current state by replaying them. There is no current-state row; the log is the truth. Internalize `state = fold(apply, empty, events)`.
2. **The aggregate is the write-side consistency boundary.** Commands load it, validate against it, and append events under an optimistic-concurrency check on its version. That is how you get *strong* write consistency in an event-sourced world.
3. **CQRS and event sourcing are two patterns, not one.** CQRS splits the consistency-optimized write model from many denormalized read models. They pair beautifully (events feed projections) but you can use either alone.
4. **Read models are disposable, denormalized caches.** A projector consumes events and maintains one per query. A new query is a new projector plus a replay — never a migration. Make projectors idempotent.
5. **The write→read gap is eventual consistency you design around, not a bug.** Return the post-command state to keep the user's own writes instant; measure projection lag per consumer and degrade the UI honestly.
6. **Snapshots bound replay cost.** Snapshot every N events so a load folds a short tail, not the whole history (the 10k-event example: 550 ms → 6.5 ms). Snapshots are a cache, never the truth.
7. **Schema evolution is forever and is the hidden tax.** Events live for years; never mutate them. Add fields and upcast old versions in memory at load time, defaulting with correct historical facts. Test upcasters against real samples.
8. **Storage is cheap; complexity is not.** Full event history costs a few dollars a month at human scale. The real cost is upcasters, eventual consistency, projector operations, and GDPR-erasure difficulty — pay it only where audit/temporal value is real.
9. **Say no by default.** Most services are CRUD and should stay CRUD. Apply event sourcing and CQRS per aggregate, at the smallest scope that captures the benefit, never as a system-wide house style. The over-engineered-ES regret is the most common outcome.

## Further reading

- [Event Sourcing and CQRS with an Event Log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log) — the storage *mechanism*: how an append-only event log is structured, partitioned, and consumed. Read this for the internals we cross-linked instead of re-deriving.
- [Change Data Capture and the Outbox Pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the other way to build CQRS read models when you are *not* event-sourcing the write side: stream a CRUD database's changes into projections.
- [Consistency Models: From Linearizable to Eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the full spectrum behind the write→read gap, and the precise guarantees each model gives you.
- [Event-Driven Microservices: Choreography vs Orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) — how the events you append flow between services, and the failure modes of event-driven integration.
- [Database per Service: The Rule That Defines Microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) — why each service owns its store, the foundation that makes per-service event sourcing possible.
- [Idempotency and Deduplication: Making At-Least-Once Safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the discipline every projector needs, because event delivery is at-least-once.
- [Message Ordering and Partitioning Guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — partition by aggregate id to parallelize projectors while preserving per-stream order.
- Forward reading in this series: [The Transactional Outbox and Reliable Event Publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) for getting your appended events reliably onto the wire, and [Data Consistency and Eventual Consistency in Practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) for the broader practitioner's playbook on living with eventual reads.
- Books: Greg Young's talks and writing on CQRS and event sourcing; Martin Fowler's "Event Sourcing" and "CQRS" articles; Chris Richardson, *Microservices Patterns* (the Event Sourcing and CQRS chapters); Vaughn Vernon, *Implementing Domain-Driven Design* (aggregates and the consistency boundary).

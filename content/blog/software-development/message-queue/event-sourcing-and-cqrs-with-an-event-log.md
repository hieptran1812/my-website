---
title: "Event Sourcing and CQRS on a Commit Log"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn to store the events and derive the state: append every change to an immutable log, fold it into current state, snapshot for speed, and split write and read models with CQRS — with the honest costs laid bare."
tags:
  [
    "message-queue",
    "event-sourcing",
    "cqrs",
    "event-store",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "projections",
    "eventual-consistency",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-1.webp"
---

Open the accounting ledger of any business that has survived a century and you will not find a column labeled "current balance" that someone keeps erasing and rewriting. You will find a sequence of entries — money in, money out, each dated, each immutable, each signed. The balance is not stored. It is *computed*, by adding up every entry from the beginning of time. If an auditor questions the number, you do not show them the number; you show them the entries that produce it. Accountants figured out three hundred years before computers that the safest way to know the truth of a system is to never throw away the events that changed it, and to derive the present from the past on demand.

Software, for most of its history, did the opposite. We stored the current balance in a row and ran `UPDATE accounts SET balance = balance - 30 WHERE id = 42`, and the previous value evaporated the instant the write committed. The history — *why* the balance is what it is, *when* each change happened, *who* caused it — was gone, recoverable only from whatever logs you happened to keep on the side. **Event sourcing** is the discipline of going back to the accountant's model: instead of storing state and mutating it, you store every state-changing *event* as an immutable fact in an append-only log, and you compute current state by folding those events together. The figure below puts the two models side by side — the same account, one row overwritten on the left, a stream of facts appended on the right.

![A comparison of state-oriented CRUD storage that overwrites the current row and loses history against event-sourced storage that appends immutable Deposited and Withdrawn events and folds them into the balance with a full audit trail](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-1.webp)

This post teaches you to think in events. By the end you will be able to model a domain as a stream of facts, fold those facts into current state, add snapshots so that fold stays fast as the stream grows, split your write model from your read models with **CQRS** (Command Query Responsibility Segregation), rebuild any read model from scratch by replay, and evolve your event schema over years without breaking the oldest events on disk. You will also be able to say *no* to event sourcing for the large majority of systems where it is the wrong tool — because the costs are real and most CRUD apps should stay CRUD. This builds directly on the stream-table duality from [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log), which shows how an append-only log and a derived table are two views of the same data, and on [Queue vs Pub/Sub vs Log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models), which establishes why a retained, replayable log is a different beast from a transient queue. It is a close cousin of [the transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing), and it leans heavily on event-schema discipline, which the sibling post [schema management and evolution](/blog/software-development/message-queue/schema-management-evolution-avro-protobuf-registry) covers in depth.

## 1. State-oriented vs event-sourced storage

Start with the default. A **state-oriented** system — the CRUD app every one of us has built a hundred times — stores the *current value* of each thing and mutates it in place. An account is a row: `(id, balance, updated_at)`. A deposit is an `UPDATE` that adds to `balance`. The row always reflects the present, and the present is all you can see. This is wonderfully simple, maps cleanly onto a relational table, and is the right choice for the overwhelming majority of software. It has exactly one structural weakness, and that weakness is the entire reason event sourcing exists: **the act of recording a change destroys the previous state.** The moment the `UPDATE` commits, the question "what was the balance yesterday?" has no answer in the database. The question "how many times has this account been overdrawn?" has no answer. The question "show me, step by step, every change that produced this number" has no answer. You overwrote the only copy.

An **event-sourced** system inverts the relationship between change and state. It never stores the current balance at all — at least not as the source of truth. It stores the *changes* as a sequence of immutable events: `Deposited(\$100)`, `Withdrawn(\$30)`, `Deposited(\$50)`. Each event is a fact about something that happened, written in the past tense, and once written it is never updated and never deleted. The current balance is not a stored field; it is a *function* of the event sequence — start at zero, apply each event in order, and the number that falls out the end is the balance. We call this computation a **fold** (or a left-fold, or a reduce), and it is the beating heart of the whole approach. State is derived; events are the truth.

The difference is not cosmetic. It changes what your system can answer. With events on disk, "what was the balance yesterday?" is "fold the events up to yesterday's timestamp and stop." "How many times overdrawn?" is "scan the stream and count the points where the running fold went negative." "Show me every change" is *literally the stored data* — you do not reconstruct the audit trail, you read it. The audit log is not a feature you bolt on; it is the database. This is the property that makes event sourcing irresistible in regulated domains — banking, trading, healthcare, insurance — where the question "prove how you arrived at this state" is not optional, it is the law.

### The cost of the inversion, stated up front

I want to be honest about the trade before we fall in love with it, because the rest of this post is going to make event sourcing look very attractive and I would be doing you a disservice not to plant the flag now. Storing events instead of state costs you three things immediately. First, **more storage** — you keep every change forever instead of one current value, so a row that would be 200 bytes in CRUD becomes a stream of hundreds of events totaling kilobytes. Second, **more compute on read** — every time you need current state, you fold, and folding a long stream is work (which is why snapshots exist, section 4). Third, and most importantly, **far more conceptual complexity** — your team now reasons about commands, events, aggregates, projections, eventual consistency, and schema versioning, instead of `SELECT` and `UPDATE`. That last cost is the one that sinks unprepared teams. Keep the matrix in section 9 in mind throughout; event sourcing is a power tool, and power tools remove fingers.

> Event sourcing is not "add an audit table." It is a wholesale inversion: events become the source of truth and state becomes a derived, recomputable view. You do not get the benefits by half-measures, and you do not get them for free.

### What makes a good event

Before we go further, a word on what an event *should* contain, because the quality of your events determines the quality of everything downstream — every fold, every projection, every replay reads these same records, possibly for a decade. A good event captures **intent and consequence, not mechanism.** Name it for the business fact that occurred (`MoneyWithdrawn`, `OrderShipped`), not for the database operation that happened to implement it (`AccountRowUpdated`). The first is meaningful to a domain expert and stable across refactors; the second is an implementation leak that will embarrass you the moment you change tables. The event carries the *delta* — what changed and the context needed to understand it — but resists the temptation to embed *derived state* like the resulting balance, because a stored derived value can disagree with the fold and now you have two sources of truth for the same number, which is the exact disease event sourcing was supposed to cure.

Events should also be **self-contained enough to interpret without external lookups.** A `SeatReserved` event that says only "seat reserved" forces every reader to go ask another system *which* seat, *which* customer, *which* screening. A good `SeatReserved` carries the screening id, the seat id, the customer id, and the timestamp — everything a projection or a fold needs to act on it in isolation. This matters acutely during replay: when you reprocess 40 million events through a fresh projection, you do not want each event triggering a synchronous call to some other service that may have changed or vanished since the event was written. The event should mean the same thing in five years, replayed cold, as it did the instant it was appended. That permanence is a design constraint you feel in every field you choose to include or omit.

One more principle that separates novices from experienced practitioners: **model events at the granularity of business decisions, not field changes.** A new modeler is tempted to emit `EmailChanged`, `PhoneChanged`, `AddressChanged` — one event per mutated field, essentially a change-log. An experienced modeler asks "what *decision* did the user make?" and emits `ContactDetailsUpdated` or, better, the actual intent like `CustomerMovedHouse`, which carries the new address *and the meaning*. The coarser, intent-named events are dramatically more useful to projections (a fraud model cares that a customer moved house, not that three fields changed) and far more stable as a forever-API. Resist the change-log temptation; events are a record of *decisions*, and the decision is the unit worth preserving.

## 2. The event store and the aggregate

If events are the truth, where do they live and how are they organized? They live in an **event store** — an append-only, ordered, durable log of events — and they are organized into **streams**, where one stream holds the events for one **aggregate**.

Let me define these three terms precisely, because sloppiness here is where event-sourced designs go wrong. An **event** is an immutable record of something that happened in the domain, named in the past tense: `MoneyDeposited`, `OrderShipped`, `SeatReserved`. It carries the data that describes *what changed* and nothing more — not the resulting state, just the delta and its context. An **aggregate** is a consistency boundary: a cluster of domain objects that must change together atomically and must be validated as a unit. A single bank account is an aggregate. A single shopping cart is an aggregate. The aggregate is the unit of *write consistency* — the thing you load, validate a command against, and append events to, all atomically. A **stream** is the ordered sequence of all events for one aggregate instance, identified by the aggregate's id: stream `account-42` holds every event that ever happened to account 42, in the order they happened.

The pipeline figure below traces what happens when a command arrives — and it is worth memorizing, because this five-step path is the entire write side of an event-sourced system.

![A pipeline showing the event-sourcing write path where a Withdraw command loads the aggregate stream and folds it to current state, validates the balance is sufficient, appends a Withdrawn event, and folds again to the new balance](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-2.webp)

The aggregate is where validation lives, and this is the single most important design rule in event sourcing: **events are only ever produced by an aggregate that has validated a command against its current folded state.** You never write an event directly. A command comes in — "withdraw \$30 from account 42" — and the flow is always: load the aggregate (fold its stream into current state), let the aggregate decide whether the command is legal *given that state*, and only if it is, append the resulting event(s). If the account balance is \$20, the `Withdraw(\$30)` command is rejected by the aggregate and *no event is ever written*. The event store therefore contains only events that were legal at the moment they happened. The log is not a dumping ground for intentions; it is a record of validated facts.

### Choosing aggregate boundaries

Aggregate design is where domain modeling meets performance, and it is genuinely hard. Too large an aggregate — say, "all accounts at the bank is one aggregate" — and you have a single stream that everything contends on, a write bottleneck, and a fold that takes minutes. Too small — "each individual transaction is its own aggregate" — and you cannot enforce invariants that span the natural unit, like "an account may not be overdrawn," because the balance lives across many tiny aggregates that no single command can see atomically. The right size is the **smallest cluster of data that must be transactionally consistent.** For a bank account, that is the account: its balance invariant must hold on every write, so the account is the consistency boundary, and `account-42` is one stream. Transactions belong *inside* that stream as events, not as their own aggregates.

The practical test: ask "what invariant must never be violated, even for an instant?" The data that invariant touches is one aggregate. An account balance must never go negative — so the account, with all its deposits and withdrawals, is one aggregate and one stream. A reservation must never double-book a seat — so the screening (showing) is the aggregate, and all seat reservations for that screening live in one stream so the no-double-book rule can be enforced on append. Get this wrong and you will either fight write contention forever or discover your invariants quietly breaking under concurrency.

There is a deeper, almost philosophical point hiding in aggregate design that is worth stating because it determines whether your system stays sane as it grows: **invariants that span aggregates cannot be enforced synchronously, and you should stop trying.** Suppose you want "the total balance across all of a customer's accounts may never exceed a credit limit." If each account is its own aggregate (which it should be, for write throughput), no single command can atomically check and enforce a rule that spans all of them — the accounts live in separate streams, possibly on separate partitions, possibly on separate machines. The instinct to make "customer" the aggregate so the cross-account rule can be enforced atomically is a trap: it serializes every write to every account a customer owns through one stream, and your throughput dies. The mature answer is to enforce cross-aggregate rules *eventually and reactively*: a projection watches the events, detects the limit breach after the fact, and emits a compensating action (flag the account, reverse the last transaction, notify a human). You accept that the rule can be momentarily violated and is corrected shortly after, rather than guaranteed never-violated at the cost of throughput. This is the same boundary that distributed systems hit everywhere — strong consistency within a unit, eventual consistency across units — and event sourcing makes it explicit rather than hiding it. Choosing what is *inside* the consistency boundary versus what is reconciled *across* boundaries is the single most consequential modeling decision you will make.

```python
# An aggregate folds its own event stream and validates commands.
# State is rebuilt from events; it is never stored directly.

from dataclasses import dataclass, field

@dataclass
class AccountState:
    balance_cents: int = 0
    is_open: bool = False
    version: int = 0          # number of events folded so far

class Account:
    def __init__(self, account_id: str):
        self.id = account_id
        self.state = AccountState()

    # ---- the FOLD: apply one event to advance state ----
    def apply(self, event: dict) -> None:
        t = event["type"]
        if t == "AccountOpened":
            self.state.is_open = True
        elif t == "MoneyDeposited":
            self.state.balance_cents += event["amount_cents"]
        elif t == "MoneyWithdrawn":
            self.state.balance_cents -= event["amount_cents"]
        else:
            raise ValueError(f"unknown event type {t}")
        self.state.version += 1

    def load(self, events: list[dict]) -> None:
        for e in events:              # left-fold over the whole stream
            self.apply(e)

    # ---- a COMMAND: validate against folded state, emit events ----
    def withdraw(self, amount_cents: int) -> list[dict]:
        if not self.state.is_open:
            raise ValueError("account is closed")
        if amount_cents <= 0:
            raise ValueError("amount must be positive")
        if self.state.balance_cents < amount_cents:
            raise ValueError("insufficient funds")     # NO event written
        event = {"type": "MoneyWithdrawn", "amount_cents": amount_cents}
        self.apply(event)             # advance our own state immediately
        return [event]                # caller appends this to the store
```

Notice the symmetry: `apply` is the only place state changes, and it changes state for *both* loading from history and handling a new command. The command method validates, then produces an event, then applies that event to itself so the in-memory aggregate stays current, and hands the event back to the caller to be appended to the store. The aggregate never touches the store; that separation is what makes it trivially unit-testable — feed it a list of events, send it a command, assert on the events it returns. No database in the test at all.

## 3. Folding events into current state

The fold deserves its own section because it is the concept people stumble on, and once it clicks the whole pattern clicks. **Current state is a left-fold of the event stream**: you begin with an empty initial state, you apply each event in order, and each application produces the next state. In functional terms it is `reduce(apply, events, initial_state)`. In imperative terms it is the loop you already saw: `for e in events: state = apply(state, e)`. The output of folding the entire stream is the current state of the aggregate.

#### Worked example: a bank account folding to a balance

Let me run a concrete stream all the way through, because nothing makes the fold clearer than watching the number move. Account `account-42` has this stream, in order:

| Offset | Event | Payload | Running balance after fold |
| --- | --- | --- | --- |
| 0 | `AccountOpened` | owner=alice | \$0.00 |
| 1 | `MoneyDeposited` | \$100.00 | \$100.00 |
| 2 | `MoneyDeposited` | \$50.00 | \$150.00 |
| 3 | `MoneyWithdrawn` | \$30.00 | \$120.00 |
| 4 | `MoneyWithdrawn` | \$20.00 | \$100.00 |
| 5 | `MoneyDeposited` | \$25.50 | \$125.50 |

To answer "what is the current balance of account 42?" you load all six events and fold: start at \$0, apply `AccountOpened` (balance unchanged, account now open), apply the two deposits (\$150), apply the two withdrawals (\$100), apply the last deposit (\$125.50). The balance is **\$125.50**, and it was never stored — it was computed. Now the magic: to answer "what was the balance after offset 3?" you fold only events 0 through 3 and stop. That is \$120.00. To answer "what was the balance at 2pm last Tuesday?" you fold every event whose timestamp is ≤ that instant. **Temporal queries are free**, because every past state is just a partial fold. A CRUD system cannot answer these at all; the past states never existed on disk.

The fold has a property that matters enormously for correctness: it must be **deterministic and order-dependent**. Given the same events in the same order, the fold must always produce the same state — no clocks, no random numbers, no reads of external systems inside `apply`. And order matters: applying `Withdrawn(\$30)` before the deposits that funded the account would be nonsensical. This is exactly why the event store guarantees **per-stream ordering** — within stream `account-42`, events have a strict, gapless order, and the fold relies on it. Across streams, ordering need not be total (account 42 and account 99 are independent), which is precisely what lets the store scale by partitioning streams across machines, the same way Kafka partitions a topic, as covered in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log).

### Optimistic concurrency on the stream

There is a subtle correctness hazard in the load-validate-append cycle: between the moment you fold the stream and the moment you append your new event, *another* command might append to the same stream. If you withdraw \$30 based on a folded balance of \$40, but a concurrent withdrawal of \$30 sneaks in first, you could overdraw. The standard fix is **optimistic concurrency control on the expected stream version.** When you load `account-42` and fold it, you note the version — say, 6 events. When you append, you tell the store "append these events *only if* the stream is still at version 6." If a concurrent write bumped it to 7, your append is rejected, you reload, refold, re-validate, and retry. This is the event-sourcing equivalent of a compare-and-swap, and every serious event store (EventStoreDB, the Kafka-based designs, Marten on Postgres) supports an expected-version parameter on append. Without it, your aggregate invariants are not safe under concurrency.

```python
# Append with optimistic concurrency: the write fails if the stream moved.
def handle_withdraw(store, account_id, amount_cents):
    events = store.read_stream(account_id)          # current history
    acct = Account(account_id)
    acct.load(events)                               # fold to current state
    expected_version = acct.state.version           # e.g. 6

    new_events = acct.withdraw(amount_cents)        # validates, may raise

    # append only if no one else wrote since we read:
    store.append(account_id,
                 expected_version=expected_version, # compare-and-swap
                 events=new_events)                 # raises on version mismatch
```

If `store.append` raises a `WrongExpectedVersion` error, the handler catches it and retries the whole cycle. Under low contention this almost never fires; under high contention on a single hot aggregate it can become the bottleneck — which loops back to aggregate sizing. A correctly sized aggregate has low enough write contention that optimistic retries are rare.

### Where the event store physically lives

People ask, reasonably, "is the event store a special database, or do I build it on something I already run?" Both answers are valid and the choice has real consequences. On one end sits a **purpose-built event store** like EventStoreDB, which natively models streams, gives you optimistic concurrency on expected version as a first-class operation, supports subscriptions and built-in projections, and handles snapshots — it is event sourcing with the batteries included. On the other end, you build the event store on **a database you already trust**: a Postgres table of `(stream_id, version, event_type, payload, recorded_at)` with a unique constraint on `(stream_id, version)` gives you append-only semantics and optimistic concurrency for free (a duplicate version violates the constraint and your append fails, exactly the compare-and-swap you wanted). The Marten library does precisely this on Postgres and is a popular, low-operational-burden choice. In the middle sits **Kafka as the event store**, where each aggregate maps to a key within a topic, the partition gives you per-key ordering, and infinite retention (or compaction) keeps the history — this is the stream-table duality from [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) used in earnest, and it shines when the event volume is high and you want the same log to also drive downstream consumers. The tradeoff: Kafka does not give you per-aggregate optimistic concurrency out of the box (the unit of ordering is the partition, not the key), so enforcing "append only if at version N" requires extra machinery — a common pattern is to keep the authoritative write model in a database that *does* enforce expected-version, and publish the resulting events to Kafka via the outbox. That is exactly why [the transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) is a close sibling of this post: it is the reliable bridge from your write-side commit to the log that the read side consumes. Choose the purpose-built store when streams and concurrency are your central concern, the Postgres approach when you want minimal new operational surface, and the Kafka approach when the log must also be a high-throughput integration backbone.

## 4. Snapshots: folding from a checkpoint

The obvious objection to "fold the whole stream on every load" is performance, and it is a real objection. If `account-42` has six events, folding is instant. If it has fifty thousand events because it is a busy corporate account ten years old, folding from offset 0 on every single request is wasteful — you redo the same arithmetic over the same forty-nine-thousand-nine-hundred events every time. **Snapshots** are the standard optimization, and the idea is exactly what an accountant does when they "close the books" each month: record the balance at a checkpoint, then you only need to fold the entries *since* that checkpoint.

A snapshot is a serialized copy of an aggregate's folded state at a specific stream version, stored alongside (but separate from) the events. To load an aggregate with snapshots, you do not start at offset 0. You load the **latest snapshot**, which gives you the state as of, say, version 50000, and then you fold only the events *after* version 50000 — usually a handful. The before-and-after figure makes the difference stark.

![A comparison of loading an aggregate without a snapshot which folds all fifty thousand events taking roughly four hundred milliseconds against loading from a snapshot at event fifty thousand which folds only twelve newer events in under five milliseconds](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-9.webp)

#### Worked example: snapshotting a hot account at event 1000

Take an account that accumulates roughly 1000 events per quarter. After ten years it has about 40,000 events. Folding 40,000 events — each a dictionary lookup and an integer add — might take 300 to 500 milliseconds in a high-level language, which is a catastrophic per-request cost on a hot path. Now add a snapshot policy: every 1000 events, write a snapshot of the current state. To load the aggregate, the store fetches the most recent snapshot (state as of event 40,000) and then folds the events from 40,001 to the current tail. If the account has had 12 events since the last snapshot, you fold 12 events instead of 40,000 — a three-thousand-fold reduction in fold work, and the load drops from ~400 ms to under 5 ms. The snapshot is pure cache: if you delete every snapshot, nothing breaks; loads just get slow again until snapshots rebuild. **The events remain the only source of truth; the snapshot is a derived optimization you can always recompute.**

```python
# Load with a snapshot: fold only the tail after the checkpoint.
SNAPSHOT_EVERY = 1000

def load_with_snapshot(store, account_id):
    snap = store.latest_snapshot(account_id)        # may be None
    acct = Account(account_id)
    if snap is not None:
        acct.state = deserialize(snap["state"])     # state as of snap["version"]
        from_version = snap["version"]
    else:
        from_version = 0
    tail = store.read_stream(account_id, after_version=from_version)
    acct.load(tail)                                 # fold only the few newer events
    return acct

def maybe_snapshot(store, acct):
    # write a new snapshot when enough events have accumulated
    if acct.state.version % SNAPSHOT_EVERY == 0:
        store.save_snapshot(acct.id, version=acct.state.version,
                            state=serialize(acct.state))
```

Two operational cautions on snapshots, learned the hard way. First, **a snapshot is coupled to the aggregate's state schema, and the fold logic that produced it.** If you change how `apply` computes state — say you fix a bug in how interest accrues — your old snapshots now encode a *wrong* state computed by the old, buggy logic. The safe move is to *version your snapshots* and invalidate (delete and recompute) all snapshots whenever the fold logic changes. Because snapshots are pure cache, throwing them all away is always safe; it just costs a wave of slow loads while they rebuild. Second, **do not snapshot too eagerly.** Snapshots cost storage and write I/O. For aggregates with short streams, snapshots are pure overhead — folding 30 events is faster than deserializing a snapshot blob. Reserve snapshots for the long-lived, high-event-count aggregates where the fold is genuinely expensive, and measure before adding them.

### Choosing the snapshot interval

The interval — how many events between snapshots — is a tunable with a clean cost model, and it is worth reasoning about rather than picking 1000 by reflex. The cost you pay on every load is "fold the events since the last snapshot," so the *worst-case* fold length is the snapshot interval (right before the next snapshot is due). If you snapshot every 1000 events, a load folds at most 1000 events; every 100 events, at most 100. Smaller intervals mean cheaper loads but more snapshot writes and more snapshot storage. The right number balances load frequency against write frequency: an aggregate read a thousand times for every write wants frequent snapshots (cheap loads dominate); an aggregate written far more often than read wants sparse snapshots or none (snapshot writes would dominate). A reasonable default for a read-heavy aggregate is an interval that keeps the worst-case fold under a few milliseconds — measure your per-event fold cost, divide your latency budget by it, and that's your interval. And do not snapshot synchronously on the write path: take snapshots asynchronously in a background process that watches the stream, so the cost of writing a snapshot never sits in a user request's latency. The snapshot is an optimization for *readers*; it should never tax *writers*.

A second subtlety: **snapshots and replay interact, and you usually want replay to ignore snapshots.** When you rebuild a *projection* (a read model), you replay raw events from offset 0 — projections are downstream of events and have nothing to do with aggregate snapshots. But when you load an *aggregate* for a command, you use snapshots to skip the fold. Keep these two paths mentally separate: snapshots accelerate the *write-side aggregate fold*; they are irrelevant to *read-side projection replay*. Conflating them is a common source of confusion for engineers new to the pattern, who reasonably but wrongly assume "snapshot" means a snapshot of the read model. It does not — it is a snapshot of one aggregate's folded write-side state.

## 5. CQRS: separating the write and read models

Here is a tension you have probably already felt reading sections 2 through 4. The write side wants events: small deltas, validated against an aggregate, appended in order. But the read side — the API that serves "GET /account/42/balance" or "show me this customer's last 50 transactions sorted by date" — does *not* want to fold a stream of events on every query. Folding is great for one aggregate's write-time validation; it is terrible for serving a list view that aggregates across thousands of accounts. The data shape the write side produces (a stream of fine-grained events) is the worst possible shape for most reads.

**CQRS — Command Query Responsibility Segregation — resolves this tension by using different models for writing and reading.** The write model is the aggregate: it accepts *commands*, validates them, and produces events. The read model is a *projection*: a separate, query-optimized data structure built by consuming the event stream — a key-value store of current balances, a SQL table of transactions indexed for fast sorting, a search index, whatever the queries need. Crucially, these two sides are *physically separate* and *independently shaped, scaled, and deployed.* The write side optimizes for consistency and validation; the read side optimizes for query latency and throughput. The grid figure shows the two sides connected by the event log that flows between them.

![A grid showing the CQRS write side where a Deposit command flows through an aggregate to append-only events, the event log as the single source of truth in the middle, and the read side where a projection consumer builds a balance read model that serves queries](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-3.webp)

CQRS and event sourcing are *separable* — and this confuses people, so let me be precise. You can do CQRS without event sourcing (two database models kept in sync by some mechanism), and you can do event sourcing without CQRS (fold the stream to serve every read, if your read load is light). But they are *natural partners*, because event sourcing produces exactly the artifact CQRS needs to keep the read side fresh: a stream of events. The write side appends events; the read side subscribes to those events and updates its projections. The event log is the seam between them, and that seam is where the famous cost enters: **the read side is updated by consuming events asynchronously, so it lags the write side by some interval. The two are eventually consistent, not strongly consistent.** We will pay that bill explicitly in section 9.

### What "command" and "query" really mean

The vocabulary is load-bearing. A **command** is an imperative request to change state — `WithdrawMoney`, `PlaceOrder`, `CancelReservation` — phrased in the imperative mood, and it can be *rejected*. A command is not a fact; it is a request that the aggregate may refuse. A **query** is a request to read state — `GetBalance`, `ListTransactions` — that never changes anything and never fails on business grounds. An **event** is the past-tense *result* of an accepted command — `MoneyWithdrawn`, `OrderPlaced`, `ReservationCancelled` — and it is a fact that already happened and cannot be rejected (it's history). Commands flow into the write side; events flow out of it onto the log; queries hit the read side. Get this trio straight — command (request, rejectable), event (fact, immutable), query (read, side-effect-free) — and the architecture organizes itself. The naming convention is not pedantry; it forces you to model intent (command) separately from consequence (event), which is the discipline that makes event-sourced code readable years later.

```java
// Commands, events, queries are distinct types — the type system enforces CQRS.

// COMMAND: a request that can be rejected
record WithdrawMoney(String accountId, long amountCents) { }

// EVENT: an immutable fact, the result of an accepted command
record MoneyWithdrawn(String accountId, long amountCents, Instant at) { }

// QUERY: a side-effect-free read against a projection
record GetBalance(String accountId) { }

// The command handler lives on the WRITE side:
class AccountCommandHandler {
    EventStore store;
    void handle(WithdrawMoney cmd) {
        var events = store.readStream(cmd.accountId());
        var acct = new Account(cmd.accountId());
        acct.load(events);                       // fold to current state
        var newEvents = acct.withdraw(cmd.amountCents());  // validate + emit
        store.append(cmd.accountId(), acct.version(), newEvents);
    }
}

// The query handler lives on the READ side and never touches the event store:
class BalanceQueryHandler {
    BalanceReadModel readModel;                  // a fast key-value view
    long handle(GetBalance q) {
        return readModel.balanceOf(q.accountId());  // O(1) lookup, no fold
    }
}
```

## 6. Projections and multiple read models

A **projection** is the process that consumes the event stream and maintains a read model. It is, structurally, an event consumer with a fold-into-a-table on the side: read the next event, update the relevant rows in the read store, repeat. The read model it maintains is a *materialized view* — precomputed, query-shaped, and disposable. And here is the property that makes CQRS-on-events genuinely powerful rather than merely tidy: **one event stream can feed many independent projections, each producing a read model shaped for a different question, and you can add a new one at any time without touching the write side at all.**

![A graph showing one account event log fanning out to four independent projections that build a balance key-value view, a monthly statement view, a sliding-window fraud model, and an external Elasticsearch search index](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-4.webp)

Consider the same account event stream feeding four different read models. A **balance projection** keeps a simple key-value map of `account_id → current_balance`, updated by adding deposits and subtracting withdrawals — it serves "GET /balance" in O(1). A **statement projection** keeps a table of transactions grouped by month, indexed for the "show me my December statement" query. A **fraud projection** keeps a sliding window of recent withdrawals per account and flags anomalies — it reads the *same events* but builds a completely different structure. A **search projection** pushes account activity into Elasticsearch so support agents can full-text-search across accounts. Four read models, one event stream, four teams able to work independently, because none of them can affect the others or the write side. Adding the fraud model six months after launch required *zero changes* to the write side — you write a new projection, replay the existing event history through it (section 7), and it catches up to live.

### Projections must be idempotent and track their position

Two non-negotiable properties make projections survivable in production, and both follow directly from the delivery semantics of the underlying log. First, **a projection must track its position in the stream** — the offset of the last event it processed — and persist that position *transactionally with the read-model update* whenever possible. On restart, it resumes from the last committed offset. This is the same offset-commit discipline as any log consumer, and getting it wrong is the classic way projections drift. Second, **a projection must be idempotent**, because the log gives you at-least-once delivery: the same event can be delivered twice (after a crash between updating the read model and committing the offset). If processing `MoneyDeposited(\$50)` twice adds \$100, your read model is now wrong. The fix is to make updates idempotent — either store the last-applied event offset *in the read row itself* and skip events at-or-below it, or make the operation naturally idempotent (set absolute state rather than increment). This connects directly to the consumer-side reliability machinery; an event-sourced projection is a log consumer with all the same failure modes.

```python
# A projection: consume events, update a read model, persist position.
# Idempotent via a per-row last_applied_offset check.

def run_balance_projection(log, read_db):
    offset = read_db.get_checkpoint("balance_projection")  # resume point
    for event in log.read_from(offset):
        acct = event["aggregate_id"]
        last = read_db.last_applied_offset(acct)           # idempotency guard
        if event["offset"] <= last:
            continue                                       # already applied; skip
        if event["type"] == "MoneyDeposited":
            read_db.add_balance(acct, event["amount_cents"])
        elif event["type"] == "MoneyWithdrawn":
            read_db.add_balance(acct, -event["amount_cents"])
        # persist read-model change AND the new offset together:
        read_db.set_last_applied_offset(acct, event["offset"])
        read_db.set_checkpoint("balance_projection", event["offset"])
```

The deep benefit hiding here is *decoupling of read shape from write shape over time.* In a CRUD system, the moment you need a new query shape, you are altering tables that the write path also uses, with all the migration risk that implies. In CQRS-on-events, the events are the stable contract, and read models are cheap, disposable, recomputable views downstream of that contract. You can throw away a read model and build a different one tomorrow without ever touching the events or the write side. The events are forever; the views are software you can rewrite.

### Synchronous versus asynchronous projections

There is a design choice on the read side that materially changes your consistency story: do projections update *synchronously* (in the same transaction as the event append) or *asynchronously* (in a separate consumer process)? The synchronous variant — append the event and update the read model in one database transaction — gives you read-your-own-writes immediately, because by the time the command returns, the read model already reflects it. It sounds appealing, and for a single read model in the same database it can be the right call. But it couples the write side's latency and availability to the read side's: if the projection update is slow or the read store is down, the *command* fails, which defeats much of the point of separating them. It also does not scale to many read models — you cannot synchronously update a search index, a fraud window, and a SQL view all inside one transaction without making the write path fragile and slow.

The asynchronous variant — projections are independent consumers that catch up to the log on their own schedule — is the dominant choice for a reason: it preserves the independence of write and read sides, lets each read model scale and fail in isolation, and supports arbitrarily many projections. Its cost is precisely the eventual-consistency window of section 9. A practical middle ground used in production is *synchronous for the one read model the user reads immediately after a command, asynchronous for everything else*: update the balance view in the command's transaction so the user sees their new balance instantly, and let the statement view, fraud model, and search index catch up asynchronously a few milliseconds later. You spend the cost of synchronous coupling only where read-your-own-writes truly matters, and keep the rest decoupled. This kind of per-read-model consistency tuning is one of the underappreciated freedoms CQRS gives you — consistency is not a single global setting, it is a knob you turn per projection.

## 7. Replay: rebuilding a view from scratch

Because read models are derived purely from events, they are **disposable and rebuildable.** This is, to my mind, the single most liberating property of the whole architecture, and it is worth dwelling on because it changes how you operate the system. If a projection has a bug — say the fraud model double-counted withdrawals for a week — you do not write a careful, error-prone data-migration script to patch the corrupted read model in place. You *delete the read model and replay the event stream from offset 0*, running every event back through the *corrected* projection code. The events are pristine (they were never wrong; only the derived view was), so the rebuilt read model is correct by construction. This is **replay**, and it is the operational superpower that justifies a great deal of event sourcing's complexity.

![A timeline showing a projection rebuild that drops the read model and starts at offset zero, replays ten million events in about eight minutes, replays thirty million more with batched writes, catches the live tail with sub-second lag, and switches to live streaming mode](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-5.webp)

#### Worked example: rebuilding a read model by replaying 40M events

Suppose your event store holds 40 million events across all account streams, and you have just deployed a fixed version of the statement projection. You need to rebuild it. The procedure: drop (truncate) the statement read model, reset the projection's checkpoint to offset 0, and let it consume the entire log from the beginning. The economics are governed by how fast you can read events and apply them. Reading from a Kafka-backed log sequentially is cheap — page-cache-friendly sequential I/O, as covered in the Kafka log internals — so the bottleneck is usually the *write* side of the projection: how fast can the read store ingest updates? If your projection applies, say, 50,000 events per second with batched writes to the read database, 40 million events take 40,000,000 / 50,000 = **800 seconds, about 13 minutes.** If you optimize the projection for replay — larger write batches, dropping read-model indexes during the rebuild and recreating them after, parallelizing across independent aggregate streams — you might hit 200,000 events/second and finish in around 3.3 minutes. The figure above sketches a 40M-event rebuild reaching the live tail in roughly half an hour at a more conservative steady rate.

The operational subtlety is **how you cut over without downtime.** A naive rebuild deletes the live read model and serves stale-or-empty data for 13 minutes — unacceptable. The professional move is a **blue-green projection rebuild**: build the *new* read model alongside the old one (in a separate table or index), replay history into the new one while the old one keeps serving live queries, and when the new model has caught the live tail, atomically flip the read path to point at the new model and retire the old. Zero downtime, and if the rebuild was buggy you flip back. This is only possible because the events are an immutable, replayable log — the same property that lets you A/B two read-model designs against the same history, or spin up a temporary projection to answer a one-off analytical question and then throw it away.

```bash
# Blue-green projection rebuild: build v2 alongside live v1, then flip.

# 1. create the new read model table/index (empty)
psql -c "CREATE TABLE statement_view_v2 (LIKE statement_view INCLUDING ALL);"

# 2. point a NEW projection instance at v2, checkpoint = 0, replay from start
PROJECTION_TABLE=statement_view_v2 \
PROJECTION_CHECKPOINT=0 \
  ./run-statement-projection &        # consumes the full log, builds v2

# 3. wait until v2 lag < 1s (caught the live tail), then atomically flip reads
psql -c "BEGIN;
         ALTER TABLE statement_view RENAME TO statement_view_old;
         ALTER TABLE statement_view_v2 RENAME TO statement_view;
         COMMIT;"

# 4. retire the old projection + drop the old table once confident
psql -c "DROP TABLE statement_view_old;"
```

This rebuildability is also your disaster-recovery story for read models: you do not back up read models, you back up the *event log* (the source of truth), and you regenerate every read model from it. The blast radius of a corrupted read model is "a replay," not "a restore-from-backup-and-pray." That is a profoundly better place to be operationally — though it does mean the event log itself becomes the one thing you absolutely cannot lose, so it must be replicated and backed up with the seriousness of any system of record.

#### Worked example: a new read model launched six months post-go-live

Concretely, here is replay's superpower in business terms. Your product has been live for six months, accumulating 12 million events. Product asks for a brand-new feature: "show each customer their spending trend over the last twelve weeks." In a CRUD system, this is a nightmare — the historical data needed to compute past weekly trends was never retained; you only ever kept current state, so you can build the trend *going forward from today* but you cannot show the customer the past twelve weeks they actually lived. In an event-sourced system, the answer is almost insultingly easy: write a `WeeklySpendProjection` that consumes `MoneyWithdrawn` events and buckets them by week, reset its checkpoint to offset 0, and replay all 12 million events. At 80,000 events/second it finishes in 150 seconds, and the moment it catches the tail, *every customer has twelve weeks of accurate history* — including the eleven weeks before the feature existed. The feature ships with a full backfill for free, because the events were always there waiting to be folded into a shape nobody had asked for yet. This is the quiet, compounding payoff of keeping the events: future questions you have not thought of yet are already answerable, as long as the relevant facts were captured as events. It is the closest thing in software to time travel, and it is why teams that genuinely need historical analytics find event sourcing transformative rather than merely tidy.

The constraint, of course, is that **you can only project what the events actually recorded.** If a fact was never captured as an event — if you stored "balance changed" without recording *why* — then no replay can reconstruct the reason, because the information was never there. This is the flip side of "events are forever": the *absence* of a field is also forever. It is why section 1's emphasis on capturing intent and rich context in events is not pedantry but foresight; the richness of your events is the ceiling on every future read model. Capture generously (within reason and within privacy constraints), because the cost of an extra field today is bytes, and the cost of a missing field is a question you can never answer.

## 8. Event versioning and upcasting

Now the cost that surprises teams a year in. Your events are immutable and they live *forever*. That is the whole point — but it means the `MoneyDeposited` event you wrote on day one will still be read, by a fold or a projection, ten years from now, long after your code has changed shape many times. **You cannot migrate events the way you ALTER a table**, because there is no "current schema" to migrate to — the log is a museum of every schema version you ever wrote, all of which must remain readable. Schema evolution in event sourcing is therefore not a one-time migration; it is a permanent compatibility obligation. This is the same discipline that the sibling post [schema management and evolution](/blog/software-development/message-queue/schema-management-evolution-avro-protobuf-registry) treats in full — registries, Avro and Protobuf compatibility rules, and how to roll changes safely — and everything there applies here, with one event-sourcing-specific twist I will focus on.

The twist is that, unlike a Kafka topic you can let age out under retention, **event-sourced events are retained forever**, so you can never "wait for the old schema to scroll off." Every reader must be able to handle every version that has ever existed. The two techniques that make this tractable are **additive-only schema changes** and **upcasting.** Additive-only means: you may *add* optional fields with defaults, but you never remove a field, never rename a field, and never change a field's type or its meaning. An old reader ignores new fields it doesn't know; a new reader fills missing fields with defaults. As long as you stay additive, every version stays mutually compatible and no transformation is needed. The taxonomy figure places versioning where it belongs in the concept map — a property of the event store itself, sitting under the write side.

![A tree taxonomy of event-sourcing concepts showing the write side branching into the event store with versioning and upcasting and the aggregate with snapshots, and the read side branching into projections and materialized read models](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-8.webp)

### Upcasting: transforming old events on read

When additive-only isn't enough — you genuinely need to restructure an event, split one event into two fields, or change a unit — you reach for **upcasting.** An upcaster is a small, pure function that transforms an old event version into the current version *as it is read from the store*, before the fold or projection ever sees it. The events on disk are never changed (they remain immutable, true to history); instead, a chain of upcasters runs at read time to bring each old event up to the shape today's code expects. You register an upcaster from v1→v2, another from v2→v3, and reading a v1 event runs it through both, emerging as v3. The fold logic only ever deals with the current version, blissfully unaware that the bytes on disk were written years ago in an older shape.

```python
# Upcasters transform old event versions to current at READ time.
# Events on disk are NEVER mutated; the chain runs as you read.

UPCASTERS = {}

def upcaster(event_type, from_version):
    def reg(fn):
        UPCASTERS[(event_type, from_version)] = fn
        return fn
    return reg

# v1 of MoneyDeposited stored a float "amount" in DOLLARS.
# v2 stores an integer "amount_cents" to avoid float rounding.
@upcaster("MoneyDeposited", 1)
def deposited_v1_to_v2(e):
    e = dict(e)
    e["amount_cents"] = round(e.pop("amount") * 100)   # dollars -> cents
    e["version"] = 2
    return e

def read_event(raw):
    e = dict(raw)
    v = e.get("version", 1)
    # run the upcaster chain until the event is at the current version:
    while (e["type"], v) in UPCASTERS:
        e = UPCASTERS[(e["type"], v)](e)
        v = e["version"]
    return e        # fold/projection always sees the current shape
```

The discipline this demands is real, and it is the part teams most consistently underestimate. Every event type carries a version number from day one (retrofitting versions onto unversioned events is miserable, so do it from the start). Every breaking change ships with an upcaster, and that upcaster lives forever — you accumulate a growing library of transformations, one for every old shape that still exists somewhere in the log. Tooling like a schema registry with enforced compatibility rules (see the schema-evolution post) turns "remember not to break compatibility" from a tribal-knowledge hope into a build-time gate that rejects an incompatible schema before it ever ships. Treat your event schemas as a *public API that you can never break* and you will stay out of trouble; treat them casually and you will spend a weekend writing upcasters for events you wrote in a hurry two years ago.

### Upcasting at read time versus rewriting the log

There is a tempting alternative to maintaining an ever-growing upcaster library: just *rewrite the old events on disk* into the new shape once, and delete the upcaster. This is sometimes called an "in-place migration" or "event copy-transform," and it is occasionally the right call — but understand what you are giving up. The moment you rewrite events, you have *mutated history*, and one of event sourcing's load-bearing promises is that the log is an immutable record of what actually happened. If a regulator or an auditor needs to see the events exactly as they were originally recorded, a rewritten log can no longer prove it. The honest pattern, when you genuinely must restructure, is **copy-transform to a new stream**: read the old events, transform them, and write them to a *new* version of the stream, keeping the original immutable and archived. You get a clean current log without erasing the historical truth. For most teams, though, read-time upcasting is the better default precisely because it never touches history — the bytes on disk stay exactly as written, and the transformation lives only in code that runs as you read. The cost is a growing upcaster library and a tiny per-read CPU cost to run the chain; the benefit is that your immutability promise stays intact, which in regulated domains is the entire reason you adopted event sourcing in the first place.

A related discipline that pays for itself: **keep the upcasters tested against real archived events.** It is not enough that your v1→v2 upcaster compiles; it must correctly transform the actual v1 events sitting in your log, including the weird ones written during that incident in 2022 when a bug produced malformed payloads. Snapshot a sample of old events into a test fixture and assert that each upcaster chain produces the expected current shape. Because the events are immutable, these fixtures never go stale — the v1 event you captured for the test will be byte-identical to the v1 events in production forever. This is a rare gift in testing: a fixture that is guaranteed to match production because production can never change the historical data the fixture represents.

## 9. The costs: complexity and eventual consistency

I have been honest about costs throughout, but they deserve a section of their own, stated plainly, because event sourcing is *oversold* and the failure mode I see most often is a team adopting it for a domain that did not need it and drowning in accidental complexity. The stack figure shows everything you are now operating — five layers where CRUD had two — and every layer is a thing that can break, lag, or confuse a new hire.

![A stack diagram of an event-sourced system layering the query API on top of projections, snapshots, and aggregates, all derived from the durable append-only event store at the base](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-6.webp)

The first cost is **conceptual complexity**, and it is the largest. In a CRUD app, a junior engineer reads a row and understands the state. In an event-sourced CQRS app, that same engineer must understand: commands and how they differ from events, aggregates and consistency boundaries, the fold, snapshots and their invalidation, projections and their idempotency, replay and blue-green rebuilds, schema versioning and upcasting, and — the killer — *eventual consistency between the write and read sides.* That is a large and unusual mental model, and it does not come for free. Every feature is more code, every bug is harder to reason about (was the event wrong, or the fold, or the projection, or just stale?), and onboarding takes longer. If your domain does not *need* the audit trail, time-travel, and replayable read models, you are paying this tax for nothing.

The second cost is **eventual consistency**, which is not a bug but a defining property, and it bites in a specific, predictable way: **read-your-own-writes is not guaranteed.** A user withdraws \$30 (a command, processed on the write side, event appended), and immediately refreshes their balance (a query, served from the read-side projection). If the projection has not yet consumed the new event — even if only by 50 milliseconds — the user sees their *old* balance and thinks the withdrawal failed. This is the number-one complaint about CQRS systems in production. The mitigations are well-trodden: serve the immediate post-command response from the write side (you just folded the new state, return it directly rather than reading the projection); show a "processing" state in the UI; or, in the rare case you truly need it, have the client wait for the projection to reach the event's offset before reading (a "read-after-write" token). But understand the trade — you chose to decouple read and write, and the price of that decoupling is a consistency window. For the deep theory of what guarantees you are and are not getting, see [consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual). The matrix below is the honest scorecard for the whole pattern versus plain CRUD.

![A matrix comparing event sourcing against CRUD across audit trail where event sourcing is perfect and CRUD is lossy, temporal query where event sourcing replays any point and CRUD cannot, conceptual complexity where event sourcing is high and CRUD is low, and storage footprint where event sourcing grows forever and CRUD is bounded](/imgs/blogs/event-sourcing-and-cqrs-with-an-event-log-7.webp)

The third cost is **storage growth and the deletion problem.** Events accumulate forever, so storage grows without bound — usually fine (events are small and storage is cheap), but it has a sharp edge: **you cannot easily delete data.** If a regulation (GDPR's right-to-erasure) requires you to delete a user's personal data, but that data is woven into immutable events that the fold depends on, you have a genuine conflict between "events are immutable" and "delete this user." The standard escape is **crypto-shredding**: store personal data encrypted with a per-user key held outside the event store, and "delete" the user by destroying their key, rendering the events' payloads permanently unreadable while leaving the event structure (and the fold's ability to run over non-personal fields) intact. It works, but it is yet another mechanism your simple CRUD app never needed. The cost of immutability is that you must engineer deletion as a special, deliberate capability rather than getting it for free from `DELETE FROM`.

The fourth cost, less discussed but very real in practice, is **operational and debugging overhead.** When a balance is wrong in a CRUD app, you look at the row. When a balance is wrong in an event-sourced CQRS app, the wrongness could live in any of several places: the events themselves might encode a bad sequence (a bug in the aggregate that produced them), the fold logic might apply them wrongly, the snapshot might be stale or computed by old buggy code, the projection might have a bug, or the read model might simply be lagging and *not actually wrong at all, just not caught up yet.* Distinguishing "the read model is wrong" from "the read model is merely behind" is a daily operational task that has no analog in CRUD, and it demands good observability: per-projection lag metrics, the ability to fold an aggregate on demand to see its true write-side state, and tooling to diff a read model against a fresh replay. None of this is insurmountable — mature event-sourcing shops build exactly these tools — but it is a body of operational investment that you must budget for and that a CRUD app simply does not require. Add it to the ledger of what the power costs.

There is also a subtler tax worth naming: **the "what events do we have?" discoverability problem.** In a CRUD app, the schema is the database and you can read it with `\d table`. In an event-sourced app, the meaning of the system is distributed across dozens of event types accumulated over years, some of which are no longer produced but still exist in the log and still must be handled. New engineers cannot easily discover "what are all the events, what do they mean, which are deprecated?" without good documentation and tooling — an event catalog, ideally generated from the schema registry. Without it, tribal knowledge becomes load-bearing and the system grows mysterious. This is solvable, but it is solved by *deliberate documentation discipline* that a self-describing relational schema gives you for free. Every benefit of event sourcing has a shadow cost, and honest adoption means budgeting for the shadows, not just admiring the benefits.

> The honest summary: event sourcing trades simplicity for auditability, replayability, and temporal power. If your domain genuinely needs those — finance, trading, healthcare, anything regulated or anything where "how did we get here?" is a first-class question — the trade is excellent. If it doesn't, you have bought a Formula 1 car to drive to the grocery store.

## Case studies and war stories

**The corporate-account fold that took four hundred milliseconds.** A payments team launched event sourcing for accounts and, for the first year, never snapshotted — streams were short, folds were instant, everyone was happy. Then their largest enterprise customer, a marketplace processing tens of thousands of payouts a day, accumulated an account stream in the hundreds of thousands of events. Every balance check on that one account folded the entire stream, and p99 latency on the balance endpoint spiked to nearly half a second *for that customer only*, which made it maddening to diagnose because the median was fine. The fix was textbook — add snapshots every 1000 events — and the load dropped to single-digit milliseconds. The lesson: **fold cost is O(stream length), and stream length is unbounded; you will eventually meet an aggregate big enough to hurt, so design the snapshot policy before you need it, even if you don't enable it on day one.**

**The projection that double-counted after a deploy.** A team ran a balance projection that incremented and decremented a running total, and it was *not* idempotent. A routine deploy killed the projection consumer after it had updated the read model but before it had committed its offset. On restart it reprocessed the last batch of events — re-adding deposits and re-subtracting withdrawals that were already applied. Balances across thousands of accounts drifted by small amounts, and because the drift was small and scattered, nobody noticed for three days until a customer's balance was off by exactly one duplicated deposit. The fix had two parts: make the projection idempotent (store last-applied offset per row, skip already-applied events) and, to repair the damage, *replay the entire stream from offset 0 into a fresh read model* — which produced correct balances by construction. The lesson is the two-sided one this whole post hammers: **at-least-once delivery means projections must be idempotent, and when a derived view is corrupt, replay beats patch.**

**The unversioned events that cost a weekend.** A startup shipped event sourcing fast and skipped event versioning — every event was just a JSON blob with no version field. Eight months later they needed to change `amount` from a float (dollars) to an integer (cents) to kill a rounding bug. Old events on disk had floats; new code expected cents. With no version field, there was no clean way to tell an old event from a new one at read time, so the upcaster had to *guess* based on the value's type, a fragile hack that broke on edge cases (whole-dollar amounts that happened to serialize as integers). They spent a weekend writing and testing a heuristic upcaster that should have been a trivial v1→v2 transform. The lesson is cheap insurance: **version every event from the very first one you write, even when you are sure the schema will never change — it always changes, and retrofitting versions onto a live log is far worse than carrying a `version` field you never needed.**

**Audit as a feature, not a chore.** A trading firm adopted event sourcing specifically because regulators demand a complete, tamper-evident record of every order, modification, and cancellation, with the ability to reconstruct the exact state of the order book at any past microsecond. In a CRUD system, satisfying that demand meant a sprawling, error-prone audit-logging side-channel that was perpetually out of sync with the real state. With event sourcing, *the audit log was the database* — there was no side-channel to drift, because the events were the source of truth, and "reconstruct the book at time T" was just a fold up to T. The compliance cost that had been a permanent tax in their CRUD system became a free, structural property. The lesson: **in domains where audit and reconstruction are first-class requirements, event sourcing is not over-engineering — it is the cheapest correct design, because it makes the regulator's hardest question a trivial query.**

## When to reach for this (and when not to)

Reach for event sourcing when the *history of changes is itself valuable* — not as a nice-to-have audit log, but as a first-class part of the domain. Finance and accounting (the balance is the fold of the ledger, by law and by nature). Trading and order books (regulators demand full reconstruction). Healthcare and insurance (every change to a record must be provable and reversible). Anything where "how did we arrive at this state?" and "what was the state at time T?" are questions the business actually asks. Reach for it, too, when you need many differently-shaped read models over the same data and want to add and rebuild them freely, or when temporal queries and time-travel are core features rather than curiosities. In these domains the complexity tax buys something you genuinely need, and the trade is decisively worth it.

Do **not** reach for event sourcing for a standard CRUD app — a typical web application managing users, settings, content, a catalog, a dashboard — where the current state is all anyone ever needs and an occasional "who changed this?" is satisfied by a simple audit-log table. Do not adopt it because it is intellectually fashionable or because a conference talk made it look elegant; the elegance is real and so is the cost, and most domains do not redeem the cost. Do not apply it uniformly across a whole system: event sourcing is a *per-aggregate* decision. The right architecture is almost always **hybrid** — event-source the handful of aggregates where history is genuinely valuable (the accounts, the orders, the trades), and leave the rest of the system as boring, reliable CRUD. A team that event-sources its core financial aggregates and keeps its user-profile service on a plain relational table has made two correct decisions, not one inconsistent one. And if you are an early-stage team still discovering your domain model, lean toward CRUD first — event sourcing locks in your event schema as a forever-API, and you do not want to commit to a forever-API for a domain you don't yet understand. You can always introduce event sourcing later for the aggregates that prove to need it; reversing out of it is far harder.

## Key takeaways

- **Store the events, derive the state.** Append every state-changing event to an immutable, append-only log, and compute current state as a left-fold of the stream. State is a derived view; events are the source of truth.
- **The aggregate is the consistency boundary.** One stream per aggregate, ordered and gapless. Commands are validated against the folded aggregate state before any event is written — the log holds only validated facts. Use optimistic concurrency on the expected stream version to stay safe under concurrent writes.
- **Snapshots are pure cache.** Fold from the latest snapshot plus the few events after it, not from offset 0, to keep loads fast on long streams. Delete them anytime — the events still produce the correct state. Invalidate all snapshots whenever the fold logic changes.
- **CQRS splits write from read.** The write model accepts commands and emits events; one or more read-model projections consume those events into query-shaped materialized views. Command (rejectable request), event (immutable fact), query (side-effect-free read) are distinct types — keep them distinct.
- **One stream feeds many read models.** Add a new projection at any time, replay history through it, and it catches up to live — with zero changes to the write side. Read shape is decoupled from write shape, forever.
- **Read models are disposable; replay rebuilds them.** When a view is buggy or corrupt, drop it and replay the log from offset 0 through corrected code; the events are pristine, so the rebuild is correct by construction. Use blue-green rebuilds for zero-downtime cutover.
- **Events live forever, so schemas are a forever-API.** Stay additive-only when you can; use versioned events and read-time upcasters when you can't. Version every event from the first one. Lean on a schema registry to gate compatibility.
- **Eventual consistency is the price of CQRS.** Read-your-own-writes is not free — serve immediate responses from the write side, show processing states, and accept a consistency window between write and read.
- **It is a per-aggregate decision, not a religion.** Event-source the aggregates where history is genuinely valuable; keep the rest CRUD. Most systems should be hybrid, and most aggregates should stay CRUD.

## Further reading

- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the stream-table duality and the append-only log internals that make an event store and its derived views two faces of the same data.
- [Queue vs Pub/Sub vs Log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) — why a retained, replayable log is the right backbone for events, distinct from a transient queue.
- [The transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — how to atomically write a change and publish the event that announces it, the reliability foundation under an event store.
- [Schema management and evolution with Avro, Protobuf, and a registry](/blog/software-development/message-queue/schema-management-evolution-avro-protobuf-registry) — the compatibility rules and registry tooling that keep a forever-retained event log readable across years of schema change.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the precise guarantees you are and are not getting from an eventually consistent read side.
- Greg Young, "CQRS Documents" and his foundational talks — the canonical articulation of Command Query Responsibility Segregation and event sourcing.
- Martin Fowler, "Event Sourcing" and "CQRS" — the patterns explained with the tradeoffs front and center.
- The EventStoreDB documentation — a purpose-built event store with streams, optimistic concurrency, and built-in projections, useful even if you implement your own on Kafka or Postgres.

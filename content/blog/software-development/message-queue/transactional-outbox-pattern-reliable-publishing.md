---
title: "The Transactional Outbox Pattern: Atomic Database Writes and Reliable Publishing"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn why you can never atomically write to your database and publish to a broker in two steps, and how the transactional outbox turns one local commit into reliable, ordered, deduplicated event delivery."
tags:
  [
    "message-queue",
    "transactional-outbox",
    "outbox-pattern",
    "idempotency",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "reliability",
    "dual-write",
    "cdc",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/transactional-outbox-pattern-reliable-publishing-1.webp"
---

There is a class of bug that no test catches, no code review flags, and no error log records — and it is sitting in almost every service that writes to a database and then publishes a message. The shape is always the same. A request comes in to place an order. Your handler commits the order row to Postgres. Then it publishes an `OrderPlaced` event to Kafka so the fulfillment service, the email service, and the analytics pipeline all find out. Both lines of code work. They pass every test. And one day in three months, between the commit and the publish, the pod gets rolled by a deploy, or the broker connection times out, or the process hits an OOM — and now the database says the order exists and Kafka has never heard of it. No exception was thrown that anyone caught. No alert fired. The order simply never ships, and the customer emails you a week later asking where their package is.

This is the **dual-write problem**, and it is the single most important reliability failure in event-driven architecture. It is not a coding mistake you can fix by being more careful, because the flaw is structural: you are writing to two independent systems — a database and a message broker — and there is no transaction that spans both. Any crash in the gap between the two writes leaves them disagreeing, and nothing in your system notices. The naive instinct to "just publish first, then commit" does not fix it; it trades a lost event for a *phantom* event, which is worse. The figure below is the whole problem and its cure in one frame: on the left, two arrows that break independently; on the right, one atomic local commit that a separate relay later turns into a durable event.

![A naive dual write commits the database and publishes to the broker as two separate steps that diverge on a crash, while the transactional outbox writes the change and the event in one local transaction and lets a relay publish afterward](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-1.webp)

This post is the messaging-reliability companion to the database-side treatment in [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), which covers the CDC internals — the write-ahead log, Debezium, replicating to derived stores. Here I deliberately point the lens the other way: at the *publishing* side. How the relay actually gets events onto the broker, what ordering you can and cannot promise, why the relay will inevitably publish some rows twice and what that demands of your consumers, how to keep the outbox table from growing without bound, and the mirror-image **inbox pattern** that makes the consumer side just as safe. By the end you will be able to take any service that does a dual write and convert it to a no-loss, no-phantom event pipeline, reason precisely about every crash point in it, and size a polling relay with real throughput math. Two sibling posts go deeper on the pieces this one leans on: [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) for what at-least-once really means at the wire, and [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) for the consumer-side machinery that makes duplicate delivery harmless.

## 1. The dual-write problem

Let me state the problem with the precision it deserves, because engineers routinely believe they have solved it when they have only moved it. A **dual write** is any operation where one logical change must be reflected in two systems that do not share a transaction. The canonical case in this series is "update the database and publish an event," but the shape is identical for "update the database and invalidate a cache," "update the database and POST to a search-indexing service," and "update the database and write to a second database." Two stores, one logical change, no atomic boundary spanning them.

The reason this cannot be fixed by ordinary error handling is that there is no ordering of the two writes that is safe. Walk the cases. Suppose you commit the database first, then publish. If the process dies after the commit and before the publish, the database has the change and the broker does not — a **lost event**. Every consumer is now permanently behind on this fact, and there is no error anywhere, because from the database's point of view the commit succeeded and from the broker's point of view nothing was ever attempted. The intent to publish lived only in the memory of a process that is now gone.

Now suppose you flip it — publish first, then commit — precisely to avoid losing the event. If the publish succeeds and then the commit fails (a constraint violation, a deadlock, a disk-full error, a crash), the broker now holds an event for a change that never happened. This is a **phantom event**: consumers act on a state transition that the source of truth never recorded. The email service emails the customer that their order shipped; the order does not exist. Phantom events are strictly worse than lost events in most systems because a lost event leaves you behind the truth (recoverable by replay) while a phantom event puts you *ahead* of a truth that will never arrive (often unrecoverable without compensating actions). There is no safe order. That is the trap.

| Failure | What the systems end up holding | Why retries don't save you |
| --- | --- | --- |
| Crash after commit, before publish | DB has the change, broker has nothing | The intent to publish died with the process; nobody remembers to retry |
| Publish succeeds, then commit rolls back | Broker has an event for a change that never happened | Consumers act on a phantom; "publish first" makes this worse, not better |
| Publish times out then lands late | Broker has the event twice | A duplicate, not a loss — a different problem solved by idempotency |
| Two requests interleave | Events reach the broker out of DB-commit order | Consumers apply stale-over-fresh; ordering is now wrong |

The fourth row is the one people forget. Even when each individual dual write succeeds, two concurrent requests can commit to the database in one order and publish to the broker in the opposite order, because the publish happens outside the transaction and is subject to its own scheduling, batching, and retry timing. A consumer that applies events in arrival order then applies a stale value over a fresh one. So a complete solution must address all four: no loss, no phantom, controlled duplication, and preserved ordering. Keep those four in mind — they are the scorecard we grade every approach against.

> The dual write is not a bug you fix with better error handling. It is a structural property of writing to two systems without a shared transaction. You either make it one atomic write, or you derive one write from a durable log of the other.

The deeper reason this is unavoidable connects to a result far older than your service: the two-generals problem. You cannot, over an unreliable network, guarantee that two parties agree on whether a message was delivered, because the acknowledgement can be lost as easily as the message. A database commit and a broker publish are two such parties. No amount of careful sequencing makes their agreement atomic — the network in between can always fail at the worst instant. The only escape is to stop trying to coordinate two writes and instead make the thing you publish a *consequence* of a single write that already committed atomically. That is exactly what the outbox does.

## 2. Why two-phase commit across DB and broker is a bad fit

The textbook answer to "make two writes atomic across two systems" is the **distributed transaction**, implemented as two-phase commit (2PC), usually via the XA protocol. A transaction coordinator asks every participant to *prepare* (do everything but the final commit, and promise you can commit if asked), and only when all participants vote yes does it tell them all to *commit*. On paper this gives you atomicity across the database and the broker. In practice it is the wrong tool here, and understanding precisely why is worth the detour, because the failure modes of 2PC are the reason the entire industry adopted the outbox instead.

The first problem is **blocking on coordinator failure**. In 2PC, once a participant has voted "yes" in the prepare phase, it is in an uncertain in-doubt state: it has promised to commit and locked the relevant rows, but it does not yet know the global decision. If the coordinator crashes after collecting the votes but before broadcasting the decision, that participant is stuck — it cannot unilaterally commit (the other participant might have voted no) and it cannot unilaterally abort (the others might have committed). It holds its locks and waits, sometimes forever, for the coordinator to recover and tell it what happened. A database stuck holding locks on hot rows because a broker transaction is in doubt is exactly the production incident you do not want. 2PC is a *blocking* protocol; it converts a coordinator failure into a liveness failure across every participant.

The second problem is **held locks across a network round trip**. Between prepare and commit, every participant holds its locks. That window now includes at least one full network round trip to the coordinator and back, plus the slowest participant's prepare time. Under load, those held locks serialize concurrent transactions on the same rows and throughput collapses. The latency of your fastest store is now hostage to the latency of your slowest participant.

The third problem is **brokers are bad XA participants — or not participants at all**. XA assumes every participant implements the prepare/commit interface with durable in-doubt state. Most modern message brokers do not. Kafka has transactions, but they are Kafka-internal (atomic across Kafka partitions and consumer offsets); Kafka is not an XA resource manager you can enlist alongside Postgres in a single distributed transaction. RabbitMQ historically had a fragile XA story that the community steers away from. So even setting aside the protocol's blocking nature, you frequently cannot get a usable XA participant out of the broker you actually run. The theory does not survive contact with the products.

The fourth problem is **operational and conceptual weight**. XA requires a transaction manager (a JTA/JTS coordinator), durable transaction logs for recovery, careful timeout and heuristic-resolution configuration, and a team that understands in-doubt resolution at 3 a.m. It is a large, sharp dependency to take on for what is, at heart, "publish an event after a row changes." The outbox achieves the same end-to-end guarantee — no loss, no phantom — using only a feature your database already has and you already trust: a local transaction. The taxonomy figure later in this post places 2PC where it belongs: a theoretically valid but practically rejected branch of the solution tree.

| Property | 2PC / XA across DB + broker | Transactional outbox |
| --- | --- | --- |
| Atomicity mechanism | Distributed prepare/commit | Single local DB transaction |
| Behavior on coordinator crash | Participants block, hold locks | No coordinator; nothing to block |
| Lock hold time | Across a network round trip | Local txn duration only |
| Broker support needed | XA resource manager (rare) | None — broker only sees publishes |
| Delivery guarantee | Exactly-once (in theory) | At-least-once + consumer dedup |
| Operational weight | Transaction manager, recovery logs | One table, one relay process |

There is one honest concession to make. 2PC, when it works, gives you something the outbox does not: the event is published *exactly* once, atomically with the commit. The outbox gives you at-least-once publishing and pushes the deduplication onto the consumer. But that trade — accept duplicates, dedup at the edge — is overwhelmingly the right one, because at-least-once-plus-idempotency is robust, debuggable, and cheap, while 2PC is brittle, blocking, and frequently impossible to wire up at all. The rest of this post is about making that trade well.

## 3. The outbox table and the local transaction

The transactional outbox pattern — catalogued by Chris Richardson on [microservices.io](https://microservices.io/patterns/data/transactional-outbox.html) as a core data pattern — is almost insultingly simple once it clicks, and the simplicity is the entire point. Instead of writing the business change and then publishing an event as two separate operations against two systems, you write the business change **and an event row into an `outbox` table in the same local database transaction.** Because both writes live in one transaction, the database's own atomicity covers them: either the order update and the outbox row both commit, or neither does. There is no crash window in which one exists without the other. You have converted a cross-system atomicity problem you cannot solve into a single-system atomicity problem your database already solves.

The stack figure below shows why this works at the level of the commit boundary. The business write and the outbox insert both sit *below* a single `COMMIT`. The database does not know or care that one row is "business state" and the other is "an event to publish" — to the transaction log they are just two writes in one atomic unit. When the transaction commits, both are durable; if anything fails before commit, both roll back together.

![A stack showing a single request transaction wrapping the business write and the outbox insert beneath one atomic commit boundary so both writes succeed together or fail together](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-6.webp)

A workable outbox schema in Postgres looks like this. Every column earns its place.

```sql
CREATE TABLE outbox (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    aggregate_id  TEXT        NOT NULL,   -- e.g. the order id; used as partition key
    event_type    TEXT        NOT NULL,   -- "OrderPlaced", "OrderCancelled"
    payload       JSONB       NOT NULL,   -- the serialized event body
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at  TIMESTAMPTZ,            -- NULL until the relay confirms publish
    message_id    UUID        NOT NULL DEFAULT gen_random_uuid()
);

-- The relay's hot query: find unsent rows in insertion order.
CREATE INDEX outbox_unsent_idx
    ON outbox (id)
    WHERE published_at IS NULL;
```

A few deliberate choices. The `id` is a monotonic identity column — it gives the relay a total order to publish in, which matters enormously for ordering guarantees (section 6). The `message_id` is a stable UUID generated *at insert time inside the transaction*, not at publish time; this is the deduplication key the consumer will use, and it must be assigned once and never change even if the relay publishes the row twice. The partial index `WHERE published_at IS NULL` is the single most important performance decision in the whole design — it keeps the relay's "find unsent rows" query scanning only the tiny set of pending rows, not the entire history, even after the table has accumulated millions of already-sent rows. We will return to why that partial index is load-bearing when we do the throughput math.

The application code that writes through this is unremarkable, which is the goal. Here is the order-placement handler in Python with SQLAlchemy, doing both writes in one transaction.

```python
def place_order(session, customer_id, items):
    # Both writes happen inside ONE transaction. The session.begin()
    # context manager commits at the end, or rolls back on any exception.
    with session.begin():
        order = Order(customer_id=customer_id, status="PLACED")
        session.add(order)
        session.flush()  # assign order.id without committing yet

        event = Outbox(
            aggregate_id=str(order.id),
            event_type="OrderPlaced",
            payload={
                "order_id": order.id,
                "customer_id": customer_id,
                "items": items,
                "total": sum(i["price"] for i in items),
            },
        )
        session.add(event)
        # COMMIT happens here, atomically, for BOTH rows.
    return order.id
```

Notice what is *absent*: there is no broker client in this handler at all. The handler does not know Kafka exists. It cannot fail to publish because it never publishes — it only records that an event *should* be published, in the same breath as the state change that justifies the event. The publishing is somebody else's job, and that somebody is the relay. This separation is the source of the pattern's robustness: the part that must be atomic (state + intent-to-publish) is atomic by virtue of being one transaction, and the part that cannot be made atomic with a remote broker (the actual network publish) is moved entirely outside the transaction where its failures are retriable rather than catastrophic.

The pipeline figure below traces the full lifecycle of one event: begin the transaction, write the business row and the outbox row, commit atomically, then — separately and later — the relay publishes and marks the row sent. The hard boundary is `commit`: everything left of it is "make the event durable," everything right of it is "deliver the durable event," and the two never share fate.

![A pipeline showing begin transaction, write business and outbox rows, atomic commit making the event durable, then the relay publishing to the broker and marking the outbox row sent](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-3.webp)

One subtlety worth flagging early: the outbox row's `payload` should be the *fully serialized event you intend to publish*, not a pointer to other tables. If the relay has to join back to live tables to construct the event at publish time, it sees whatever those tables hold *now*, which may have changed since the event was recorded — you would publish "order total \$50" for an event that, at the time it occurred, was "order total \$40." Capture the event's full content at the moment of the business change and freeze it in the outbox row. The outbox is an append-only log of facts that happened, serialized at the instant they happened.

## 4. The relay: getting events to the broker

The relay (sometimes called the message relay or outbox poller) is the process that reads unsent outbox rows and publishes them to the broker. It is the only component that talks to both the database and the broker, and it is where all the interesting reliability behavior lives. The whole architecture is in the grid figure below: the service writes to the database, the relay reads from the database and publishes to the broker, the broker fans out to consumers, and consumers update their derived stores. The service never touches the broker; the relay never touches business logic. Clean seams.

![A grid showing the service writing to the database with business and outbox rows, the relay reading unsent rows and publishing to a Kafka broker, and the broker delivering to a consumer that deduplicates and applies to a warehouse and search index](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-2.webp)

The relay's core loop, for a polling implementation, is short:

```python
def relay_loop(db, producer, batch_size=500, poll_interval=0.2):
    while True:
        # 1. Read a batch of unsent rows in id order (the total order).
        rows = db.execute("""
            SELECT id, message_id, aggregate_id, event_type, payload
            FROM outbox
            WHERE published_at IS NULL
            ORDER BY id
            LIMIT %s
        """, (batch_size,)).fetchall()

        if not rows:
            time.sleep(poll_interval)
            continue

        # 2. Publish each row. The broker key is the aggregate_id so all
        #    events for one order land on the same partition, preserving order.
        for r in rows:
            producer.send(
                topic="orders",
                key=r.aggregate_id.encode(),
                value=serialize(r),
                headers=[("message_id", str(r.message_id).encode())],
            )
        producer.flush()  # block until the broker acks every send

        # 3. Mark the whole batch sent AFTER the broker has acknowledged.
        ids = [r.id for r in rows]
        db.execute(
            "UPDATE outbox SET published_at = now() WHERE id = ANY(%s)", (ids,)
        )
        db.commit()
```

The ordering of those three steps is everything, and it is the same ordering decision as a consumer's ack-after-process from the [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) post. The relay publishes first, waits for the broker to acknowledge (`flush()` blocks until every send is acked), and *only then* marks the rows sent. This gives at-least-once publishing: if the relay crashes after the broker acks but before the `UPDATE` commits, the rows are still marked unsent, so on restart the relay reads them again and re-publishes. The event is duplicated, never lost. If you reversed the order — mark sent first, then publish — a crash in between would lose the event, exactly the bug we are eliminating. So the relay deliberately chooses duplicates over loss, and pushes deduplication downstream.

Two operational details make this safe under concurrency. First, **only one relay instance may publish a given row at a time**, or you get the same row published by two relays and the duplication rate spikes. For a single-relay setup that is trivial; for a high-availability pair, use a leader lock (a Postgres advisory lock, a row in a `relay_leader` table with a lease, or a Kubernetes lease object) so exactly one relay is active and the standby takes over only when the leader's lease expires. Alternatively, partition the outbox by `aggregate_id` hash and assign disjoint partitions to relay instances so two relays never contend for the same row. Second, the relay's mark-sent `UPDATE` and the broker publish are themselves a dual write — and yes, that means the relay can crash between them. But this dual write is *safe in the at-least-once direction*: its only failure mode is re-publishing an already-published row, which is a duplicate, not a loss. We deliberately concentrate the unavoidable dual write at the one point where its failure mode is benign.

### Scaling the relay with SKIP LOCKED

There is a third option for running multiple relay workers that avoids both a leader lock and static partitioning, and it is worth knowing because it scales smoothly: Postgres's `FOR UPDATE SKIP LOCKED`. Each relay worker selects a batch of unsent rows `FOR UPDATE SKIP LOCKED`, which atomically locks the rows it reads and *skips* any rows another worker has already locked. Two workers running the same query never grab the same rows — the first to lock a row owns it, the second skips past it to the next unlocked rows. You can run ten relay workers against one outbox table and they cleanly divide the unsent rows among themselves with zero coordination beyond the database's own row locks.

```python
def claim_batch(db, batch_size):
    # SKIP LOCKED lets N workers divide the unsent rows with no coordination:
    # each worker locks the rows it reads and skips rows another worker holds.
    return db.execute("""
        SELECT id, message_id, aggregate_id, event_type, payload
        FROM outbox
        WHERE published_at IS NULL
        ORDER BY id
        FOR UPDATE SKIP LOCKED
        LIMIT %s
    """, (batch_size,)).fetchall()
```

The one caveat is ordering. `SKIP LOCKED` distributes rows across workers, so two events for the *same* aggregate can end up on two different workers and be published out of order. If you need per-aggregate ordering (you usually do), you must keep all of one aggregate's events on one worker. The clean way is to claim by aggregate, not by row: select the next batch of *distinct* `aggregate_id`s with unsent rows `FOR UPDATE SKIP LOCKED`, and have each worker publish all unsent rows for the aggregates it claimed, in `id` order. Then one aggregate's events are never split across workers, ordering is preserved within each aggregate, and you still get the horizontal scaling. This is the production-grade relay-scaling pattern: `SKIP LOCKED` on the aggregate, ordered publish within it. It scales to many workers without a leader election and without statically partitioning the table, which is why high-throughput outbox deployments tend to converge on it.

#### Worked example: tracing an order through every crash point

Let me make the no-loss claim concrete by walking an order placement and crashing at every possible point, showing that no point loses an event and the only bad outcome is a recoverable duplicate.

A customer places order #8821. The handler begins a transaction, inserts the order row (`status=PLACED`) and an outbox row (`id=44190`, `message_id=7f3a...`, `event_type=OrderPlaced`), and the relay will later publish it. Now crash at each stage:

- **Crash before COMMIT.** The transaction rolls back. Neither the order row nor the outbox row exists. The customer's request failed and they will retry. Nothing is half-done; no event, no order. Correct.
- **Crash exactly at COMMIT.** Postgres's commit is itself atomic (the WAL flush either happens or it does not). Either both rows are durable or neither is. There is no state where the order exists but the outbox row does not. Correct.
- **Crash after COMMIT, before the relay reads the row.** The order and outbox row are both durable, `published_at IS NULL`. When the relay (or its restarted replacement) runs its query, it finds row 44190 and publishes it. Event delivered, slightly late. Correct.
- **Crash after the relay reads the row, before publish.** The relay had the row in memory but never published. On restart it queries again; the row is still unsent; it publishes. No loss. Correct.
- **Crash after publish acks, before mark-sent.** This is the interesting one. The broker has the event. The outbox row is still `published_at IS NULL`. On restart the relay re-reads row 44190 and **publishes it again** — a duplicate. The consumer sees `message_id=7f3a...` twice and, being idempotent, applies it once. No loss, no phantom, one harmless duplicate. Correct.
- **Crash after mark-sent commits.** The row is `published_at = now()`; the relay will never read it again. The event was published exactly once and recorded as sent. Correct.

There is no crash point that loses the event and no crash point that produces a phantom. The worst case is a single duplicate, and the consumer handles it. That is the entire promise of the outbox, demonstrated exhaustively. Contrast every one of these against the naive dual write, where "crash after commit, before publish" loses the event with no recovery path at all.

## 5. Polling relay vs CDC log-tailing

There are two fundamentally different ways to build the relay, and choosing between them is the main architectural decision in adopting the outbox. The before-after figure below contrasts them: a **polling relay** repeatedly queries the outbox table for unsent rows on a fixed interval; a **CDC relay** tails the database's write-ahead log and emits an event the instant a row is committed, never querying the table at all.

![A before-after comparison where the polling relay runs a SELECT for unsent rows every interval and updates them sent, while the CDC relay tails the write-ahead log for committed inserts and streams them in commit order without scanning the table](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-8.webp)

The **polling relay** is the one in the code above. Its virtues are simplicity and zero new infrastructure: it is a loop, a `SELECT ... WHERE published_at IS NULL ORDER BY id`, a publish, and an `UPDATE`. You can write it in an afternoon, run it as a sidecar or a cron-like worker, and debug it by reading SQL. Its costs are latency and database load. Latency is bounded by the poll interval — an event committed just after a poll waits up to one full interval before the next poll picks it up. Database load comes from running that query on every interval forever, which is why the partial index on unsent rows is mandatory: without it, every poll scans the whole table, and a table that grows to tens of millions of rows turns each poll into a multi-second sequential scan that hammers the database. With the partial index, each poll touches only the small set of currently-unsent rows and the cost stays flat as history grows.

The **CDC relay** flips both tradeoffs. It connects to the database as a logical replication consumer — in Postgres, via a logical replication slot and a tool like [Debezium](https://debezium.io/); in MySQL, by reading the binlog — and receives a stream of committed changes. When an `INSERT` into the outbox table commits, the CDC connector emits it as a record, in commit order, within milliseconds. There is no polling, no `SELECT`, and crucially **no load on the table from scanning** — the connector reads the WAL, which the database is writing anyway. Latency drops to single-digit milliseconds and is no longer tied to a poll interval. The cost is operational: you now run and monitor a CDC connector (Debezium on Kafka Connect, typically), manage a replication slot that can accumulate WAL if the connector falls behind (a slot that stops consuming can fill the disk with retained WAL — a real production hazard), and handle connector restarts, schema changes, and offset management. The CDC internals — how the WAL is structured, how Debezium snapshots and streams, how to operate the connector — are exactly the subject of the [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) post, so I will not re-derive them here. The point for *this* post is the publishing tradeoff, which the matrix below summarizes.

![A matrix comparing polling relay and CDC log-tailing across publish latency, database load, ordering ease, and operational complexity, showing polling is simpler but laggier and CDC is low-latency but operationally heavier](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-4.webp)

My default recommendation: **start with a polling relay**. It is enough for the vast majority of services. A 200ms-to-1s publish latency is invisible to almost every business workflow, the implementation fits in one file you fully understand, and you take on zero new operational surface. Move to CDC when you have a concrete reason — you need sub-50ms event latency, your outbox throughput is high enough that polling load matters even with the partial index, or you are *already* running Debezium for other CDC needs and adding the outbox table to it is nearly free. Reaching for Debezium on day one to publish a few hundred events per second is over-engineering; you take on a replication slot, a Kafka Connect cluster, and connector operations to save a few hundred milliseconds nobody will notice.

#### Worked example: sizing a polling relay's throughput and lag

Now the numbers. Suppose your service produces outbox rows at a steady **5,000 rows/s** at peak, and your relay polls every **200ms** with a batch size of **500**. Does it keep up, and what is the worst-case publish lag?

Per poll the relay can publish at most `batch_size = 500` rows. At one poll every 200ms, that is 5 polls/s, so the ceiling is `500 × 5 = 2,500 rows/s`. That is *below* the 5,000 rows/s arrival rate — the relay falls behind and the backlog grows by 2,500 rows/s indefinitely. This configuration is broken. The fix is to make `batch_size × polls_per_second ≥ arrival_rate` with margin. Two levers: raise the batch size or shorten the interval.

Set `batch_size = 2,000` and keep the 200ms interval: ceiling becomes `2,000 × 5 = 10,000 rows/s`, which is 2× the 5,000 rows/s arrival rate. That gives a healthy 50% headroom for bursts and for catching up after a brief stall. Now the worst-case lag for a single event: a row committed one microsecond after a poll completes waits up to the full 200ms interval before the next poll begins, plus the time to publish a batch and get broker acks (say 30ms for 2,000 messages with `acks=all` batched), so worst-case end-to-end lag is roughly `200ms + 30ms ≈ 230ms`. Median lag is about half the interval plus publish time, `~130ms`. For most workflows, fine.

Suppose 230ms is too slow. Shorten the interval to 50ms with `batch_size = 600`: ceiling is `600 × 20 = 12,000 rows/s` (2.4× headroom) and worst-case lag drops to `~50ms + publish ≈ 75ms`. The cost is 4× more queries against the database (20/s instead of 5/s), but each query, thanks to the partial index, touches only the small unsent set, so the load stays modest. The general design rule falls out of this: **pick the interval from your latency budget, then pick the batch size so `batch × (1/interval) ≥ 2 × peak_arrival_rate`.** And always size for the *peak* arrival rate, not the average, or the relay falls behind exactly when traffic spikes and you need it most — at which point lag climbs, events publish late, and your "real-time" pipeline is suddenly minutes behind during the busiest hour of the day.

One more number worth internalizing: the *backlog drain rate*. If your relay was down for 2 minutes during a deploy while the service kept producing at 5,000 rows/s, it accumulated `5,000 × 120 = 600,000` unsent rows. With the 10,000 rows/s ceiling and 5,000 rows/s continuing to arrive, the relay drains the backlog at `10,000 − 5,000 = 5,000 rows/s`, so it takes `600,000 / 5,000 = 120 seconds` to catch back up. Symmetric to the outage, which is a nice sanity check. If your relay's ceiling were only 6,000 rows/s, the drain rate would be just 1,000 rows/s and the same outage would take 10 minutes to recover from — headroom is not a luxury, it is your recovery-time guarantee.

## 6. Ordering and idempotent publishing

The outbox gives you a powerful ordering primitive almost for free: the monotonic `id` column is a **total order** over all events the service has ever produced, in commit order. A polling relay that publishes `ORDER BY id` and a CDC relay that streams in WAL commit order both respect that order. The question is what order the *consumer* ultimately observes, and the answer depends on how you key messages onto the broker.

The crucial move is to use the `aggregate_id` (the order id, the user id — whatever entity the events are about) as the broker partition key. In Kafka, all messages with the same key land on the same partition, and a single partition preserves order. So **all events for one order are delivered to consumers in the exact order they were committed**, because they share a key, share a partition, and the relay publishes them in `id` order. You do not get a global total order across all orders — events for order A and order B may interleave arbitrarily across partitions — but you almost never want that. What you want is **per-aggregate ordering**: every event about a given order, in order. The outbox plus key-by-aggregate gives you exactly that, and the deep treatment of why partition-level ordering is the only ordering a log can cheaply promise is in [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees).

There is a sharp subtlety that bites people: **the relay must publish a given aggregate's events strictly in order, which means it cannot publish them in parallel without care.** If the relay fires off two sends for the same order concurrently and the second one's broker ack returns first, or a retry on the first one reorders them, the consumer sees them out of order. The defenses are: publish a batch in `id` order and, if using async sends with retries, set the Kafka producer's `max.in.flight.requests.per.connection=1` (or use the idempotent producer with `enable.idempotence=true`, which preserves order even with up to 5 in-flight requests by sequencing them). For a polling relay that publishes one batch and flushes, ordering within the batch is preserved as long as you do not reorder the rows and the producer does not silently reorder retries. This is the same producer-idempotence machinery covered in the delivery-semantics post, now applied to the relay.

Now, **idempotent publishing**. We established that the relay publishes at-least-once: a crash after publish-ack and before mark-sent re-publishes the row. So the same logical event, carrying the same `message_id`, can reach the broker twice. The relay cannot prevent this — preventing it would require atomicity between the broker ack and the database mark-sent, which is the very dual write we said is impossible. So the relay does not try to make publishing exactly-once; it makes publishing *deterministic*. Every re-publish of outbox row 44190 carries the identical `message_id` that was frozen into the row at insert time. The duplicate is byte-for-byte identifiable as a duplicate.

That stable `message_id` is what lets the consumer deduplicate, and deduplication is the consumer's job, not the relay's. This is the load-bearing division of labor in the whole pattern: **the relay guarantees at-least-once delivery with a stable idempotency key; the consumer guarantees idempotent processing using that key.** Together they compose to *effectively-once*. The consumer keeps a set of processed `message_id`s (a table, a Redis set, a bloom filter with a backing store) and on each delivery checks whether it has seen this id; if so, it skips. The full machinery — where to store the dedup set, how to bound it, how to make the dedup check and the business write atomic — is the subject of [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe), and it is mandatory reading to deploy an outbox safely. The outbox without consumer idempotency is only half a solution: you have eliminated lost events and phantom events but introduced duplicate events, and an unprepared consumer will double-charge the card.

The timeline figure below makes the duplicate concrete: commit, relay reads, publish acks, crash before mark-sent, relay restarts, re-publishes. Two events on the broker, one `message_id`, one effect at the consumer.

![A timeline of a crash where the relay commits the outbox row, reads it, publishes and gets a broker ack, then crashes before marking it sent, restarts, re-reads the row, and re-publishes it as a duplicate carrying the same message id](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-5.webp)

Why does the relay accept duplicates instead of trying to be exactly-once? Because the alternative is the phantom event. Look at the graph figure below, which shows the failure you get if you try to be clever and publish *before* you are sure the change is durable — the "publish first" instinct from section 1, generalized. The publish succeeds and fans out to the email service and the search index; then the database transaction rolls back; and now those consumers have acted on a change the source of truth never recorded. The phantom event is unrecoverable in general — you cannot un-send the email — whereas a duplicate is trivially recoverable by a consumer that dedups. The outbox makes the deliberate choice to live on the at-least-once side of the line, where every failure is a duplicate (recoverable) and never a phantom (not recoverable) or a loss (recoverable only by lucky external audit).

![A graph of the phantom-event failure where a handler publishes to the broker first, the database transaction then rolls back, the email service and search index act on the event, but the source of truth holds no such change](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-9.webp)

A common misconception is that Kafka's idempotent producer already solves the duplicate problem, so the consumer does not need to dedup. It does not, and the distinction matters. The idempotent producer (`enable.idempotence=true`) de-duplicates retries *within a single producer session* — if a `send()` is retried because its ack was lost on the network, the broker recognizes the producer's sequence number and stores the record only once. That eliminates duplicates caused by the producer's own internal retries. But the outbox's duplicate is a *different* duplicate: it comes from the relay crashing and *re-reading* an unsent row in a brand-new producer session after restart, then publishing it again. To the broker that is a genuinely new record from a new producer with a new sequence — the idempotent producer has no way to know it is the same logical event, because the broker's dedup window is per-producer-session and the relay restarted. So Kafka's idempotent producer is necessary (it preserves order and kills retry duplicates, as the inventory war story shows) but not sufficient. The relay's stable `message_id` and the consumer's dedup table are what catch the cross-session, post-crash re-publish. Two layers, two different duplicate sources; you need both.

## 7. Outbox cleanup and retention

An outbox table that only ever grows is a slow-motion outage. Every sent row stays forever, the table bloats to hundreds of millions of rows, indexes grow, vacuum gets expensive, and one day a poorly-considered query does a sequential scan and the database falls over. You must reap sent rows. The question is how aggressively and by what mechanism, and there are three sane strategies.

**Strategy 1: delete sent rows on a schedule.** A periodic job deletes rows where `published_at < now() - interval '1 hour'`. The hour of slack is intentional — it gives you a window to inspect recently-published events for debugging and to absorb any relay lag without deleting rows that are not actually published yet. This is the simplest approach and works well for moderate volumes. The catch is that bulk `DELETE`s on Postgres create dead tuples that vacuum must reclaim, and on a hot table that vacuum churn can be significant. Delete in bounded batches (`DELETE ... WHERE id IN (SELECT id ... LIMIT 10000)`) on a tight loop rather than one giant `DELETE`, so you never lock the table for long and you keep dead-tuple accumulation steady rather than spiky.

```sql
-- Reap in bounded batches to avoid long locks and vacuum spikes.
DELETE FROM outbox
WHERE id IN (
    SELECT id FROM outbox
    WHERE published_at IS NOT NULL
      AND published_at < now() - INTERVAL '1 hour'
    ORDER BY id
    LIMIT 10000
);
```

**Strategy 2: partition the table by time and drop whole partitions.** Declare the outbox as a range-partitioned table by `created_at` — one partition per day, say. Cleanup becomes `DROP TABLE outbox_2026_06_10`, which is instantaneous and creates zero dead tuples (it just unlinks the partition's files). This is the best approach at high volume: you never run a `DELETE`, never generate vacuum work from reaping, and dropping a day's partition is metadata-only. The cost is the modest complexity of managing partitions (creating tomorrow's partition ahead of time, dropping old ones), which `pg_partman` or a small cron job handles. If your outbox sustains thousands of rows per second, partition-and-drop is the right answer and `DELETE`-based reaping is not.

**Strategy 3: with CDC, you may not need to mark or delete at all.** If you publish via CDC log-tailing, the relay never queries the table for unsent rows — it tails the WAL. In that model the outbox table exists only so the *insert* shows up in the WAL; the connector reads the insert from the log and you never need a `published_at` column or an `UPDATE` to mark sent. Some setups even use an `INSERT ... DELETE` or an `UNLOGGED` table trick, but the common, clean pattern is: insert the row (it hits the WAL, the CDC connector reads it), then let a time-partition drop reap the table purely to bound its size, since the connector has already consumed every insert it needs. CDC turns retention from a correctness concern into a pure housekeeping concern.

Whatever strategy you choose, **monitor the count of unsent rows** (`SELECT count(*) FROM outbox WHERE published_at IS NULL`) as a first-class metric. A healthy outbox has a near-zero unsent count that briefly spikes and drains. A growing unsent count is your earliest, clearest signal that the relay has stalled or fallen behind — long before consumers notice missing events. Alert on `unsent_count > threshold` and on `oldest_unsent_age > threshold`, and you will catch relay failures in seconds instead of discovering them from a customer complaint a day later. The unsent count is to the outbox what consumer lag is to a Kafka consumer group: the single number that tells you whether the pipeline is healthy.

One more retention trap: do not retain so little that you cannot replay. If a downstream consumer has a bug and you need to re-publish the last six hours of events, a one-hour retention has already deleted them. The fix is not to retain forever in the outbox — that is the broker's job — but to recognize that the outbox is a *staging buffer for publishing*, not the system of record for events. The durable event history lives in the broker's own retained log (Kafka topics with multi-day retention, or a compacted topic). Replay comes from the broker, not the outbox. Keep the outbox small and the broker retentive; do not confuse the two roles.

## 8. The inbox pattern on the consumer side

The outbox solves the *producer* side: it guarantees that a committed business change reliably becomes a published event, at-least-once. But the consumer side has a symmetric dual-write problem, and the symmetric solution is the **inbox pattern** (sometimes called the consumer-side outbox or the idempotent-receiver table). Here is the consumer's version of the dual write: it receives an event from the broker, processes it (updates its own database), and then acknowledges the message back to the broker. Those are two writes to two systems — the consumer's database and the broker's offset/ack state — and they can diverge on a crash exactly like the producer's dual write. If the consumer processes the event (commits its database change) and then crashes before acking, the broker redelivers the event and the consumer processes it twice. If the consumer acks before processing and crashes, the message is lost. Same problem, mirror image.

The inbox pattern closes this the same way the outbox does: by collapsing the two writes into one local transaction. The consumer keeps an `inbox` table (or a `processed_messages` table) keyed by `message_id`. When an event arrives, the consumer, **in a single local transaction**, (1) checks whether `message_id` is already in the inbox and bails out if so, (2) does its business processing (updates its own tables), and (3) inserts the `message_id` into the inbox. All three in one transaction. Because the business update and the inbox insert commit atomically, the consumer can never apply the business change without recording that it processed this message, and can never record processing without applying the change. After this transaction commits, it acks the broker. If it crashes before acking, the broker redelivers; the consumer's next attempt finds the `message_id` already in the inbox and skips the business change, then acks. Exactly-once *effect*, built from at-least-once delivery plus an idempotent receiver.

```python
def handle_event(db, broker_msg):
    message_id = broker_msg.headers["message_id"]
    with db.begin():  # one local transaction
        # 1. Dedup check: have we processed this message_id before?
        already = db.execute(
            "SELECT 1 FROM inbox WHERE message_id = %s", (message_id,)
        ).fetchone()
        if already:
            return  # duplicate; skip processing, the outer ack still fires

        # 2. Business processing, atomic with the dedup record.
        order = deserialize(broker_msg.value)
        db.execute(
            "INSERT INTO orders (id, customer_id, status) "
            "VALUES (%s, %s, 'PLACED') ON CONFLICT (id) DO NOTHING",
            (order["order_id"], order["customer_id"]),
        )

        # 3. Record that this message_id is handled — atomic with step 2.
        db.execute(
            "INSERT INTO inbox (message_id, processed_at) VALUES (%s, now())",
            (message_id,),
        )
    broker_msg.ack()  # ack AFTER the transaction commits
```

The outbox and inbox compose into an end-to-end guarantee that no individual hop provides. The producer's outbox guarantees the event is published at-least-once with a stable `message_id`. The broker guarantees at-least-once delivery to the consumer. The consumer's inbox guarantees each `message_id` produces its effect at most once. Chain them: the business change at the source produces *exactly one effect* at the destination, surviving any single crash anywhere along the path. That is the real prize. You have built effectively-once end-to-end out of three at-least-once links and two idempotency tables, with no distributed transaction anywhere. This is why the outbox-plus-inbox pairing, not either one alone, is the production-grade pattern.

A practical note on bounding the inbox: it has the same growth problem as the outbox, and the same retention strategies apply. But you can often bound it more cleverly using the event's own ordering. If events for an aggregate carry a monotonic sequence number, the consumer can store just the *highest sequence number processed per aggregate* instead of every `message_id`, and treat any event with a sequence ≤ the stored high-water mark as a duplicate. That collapses the dedup state from one row per message to one row per aggregate, a huge saving. When events do not carry a natural sequence, fall back to storing `message_id`s with a time-based retention window sized to the broker's redelivery window — you only need to remember a message id for as long as the broker might redeliver it.

There is a failure mode the inbox does *not* solve, and it is important to call it out so you do not mistake the inbox for a cure-all: the **poison message**. Suppose an event arrives that the consumer cannot process not because of a transient fault but because of a permanent one — a malformed payload, a schema the consumer cannot deserialize, a referenced entity that violates an invariant. The inbox's dedup logic does not help here, because the message was never successfully processed, so it is never recorded as handled, so the broker keeps redelivering it forever. Each redelivery fails the same way, the consumer never advances its offset past the poison message, and the entire partition stalls behind it — a head-of-line block that takes down processing for every well-formed event sitting behind the poison one. The inbox guarantees you do not double-process a *successful* message; it says nothing about a message that can never succeed. The fix is orthogonal and complementary: a dead-letter queue. After N failed processing attempts, the consumer routes the poison message to a dead-letter topic, records it (so it is not silently lost), and advances past it so the partition unblocks. The outbox-inbox pairing handles duplication and loss from crashes; the dead-letter queue handles permanent processing failures. Production consumers need both, and conflating them — expecting the inbox to handle poison messages — leaves you with a stalled partition during an incident.

It is also worth being explicit about how the inbox interacts with the broker's own offset commit. In the Kafka example above, the consumer commits its offset *after* the inbox transaction commits. That ordering is the consumer-side analog of the relay's publish-then-mark-sent: process-and-record first (atomically, in one DB transaction), then acknowledge to the broker. A crash after the inbox commit but before the offset commit redelivers the message, and the inbox check makes the redelivery a no-op. A crash before the inbox commit means the DB transaction rolled back, nothing was recorded, and the redelivery reprocesses cleanly from scratch. There is no ordering of inbox-commit and offset-commit that loses an effect, and the only failure mode is a redelivery the inbox absorbs — exactly the property we engineered the whole pipeline to have at every hop.

## 9. A complete, no-loss order-service walkthrough

Let me assemble everything into one end-to-end order service and trace a single order from API request to fulfillment, naming every component and every guarantee. This is the pattern in full: producer outbox, polling relay, Kafka, consumer inbox.

The order service exposes `POST /orders`. The handler opens a transaction, inserts the order row with `status=PLACED`, inserts an outbox row with `event_type=OrderPlaced` and a freshly generated `message_id`, and commits. That commit is the atomic moment — order and event become durable together (figure 6's commit boundary). The handler returns `201 Created` to the customer. It has not published anything; it has only recorded that an event should be published.

The polling relay, running as a separate process with a leader lock so only one instance is active, wakes every 200ms, selects unsent outbox rows in `id` order, and publishes each to the Kafka topic `orders` keyed by `aggregate_id` (the order id) with the `message_id` in a header. It flushes (waits for broker acks with `acks=all`), then marks the batch sent. If it crashes mid-cycle, restart re-publishes unsent rows — at-least-once with stable `message_id`s (figure 5's timeline).

Kafka stores the event durably in the `orders` topic, partitioned by order id, replicated across brokers per [Kafka replication and ISR](/blog/software-development/message-queue/kafka-replication-isr-acks-durability). All `OrderPlaced` and later `OrderShipped` events for the same order share a partition, so they are delivered in commit order. Kafka retains the topic for seven days, which is your replay buffer.

The fulfillment consumer, a Kafka consumer group, polls the `orders` topic. For each event it runs the inbox transaction: check `message_id` in the inbox, and if new, create the fulfillment record and insert the `message_id`, atomically. Then it commits its Kafka offset. A redelivered duplicate finds its `message_id` in the inbox and is skipped — the fulfillment record is created exactly once even though the event may be delivered twice (figure 1's "after" column, generalized to the consumer).

#### Worked example: where is order #8821 at each failure?

Trace order #8821 with a failure injected at each hop, confirming exactly-once *effect* survives any single crash.

1. **Handler crashes before commit.** No order row, no outbox row. Customer gets a 500 and retries; a fresh attempt creates a fresh order with a fresh `message_id`. No duplicate fulfillment, no orphaned order. Clean.
2. **Handler commits, then crashes before returning 201.** Order and outbox row are durable. The customer's client times out and retries `POST /orders` — but if you keyed the request with an idempotency key (an HTTP-level idempotency token), the retry is recognized and does not create a second order. The relay publishes the first order's event normally. The HTTP idempotency key is the *third* idempotency layer, guarding the API itself; the outbox guards DB-to-broker; the inbox guards broker-to-consumer.
3. **Relay crashes after publishing #8821 but before mark-sent.** On restart it re-publishes #8821 — same `message_id`. Kafka now holds two copies. The fulfillment consumer processes the first, records the `message_id` in its inbox; the second is skipped on the inbox check. Fulfillment happens once. The duplicate cost the consumer one no-op transaction.
4. **Consumer processes #8821, commits its inbox transaction, then crashes before committing the Kafka offset.** Kafka redelivers #8821. The consumer's inbox already holds the `message_id`; it skips processing and commits the offset this time. Fulfillment happens once. The redelivery cost one inbox lookup.
5. **A network partition isolates the relay from Kafka for 90 seconds.** The relay's publishes fail and it does not mark rows sent; the unsent count climbs (your alert fires at 60s). When the partition heals, the relay drains the backlog at its headroom rate. Every event publishes, late but lossless. No order is dropped; the only symptom is temporary latency, visible in your unsent-count metric.

At no point in this trace is an order lost, an order phantomed, or a fulfillment duplicated. Every single-crash scenario resolves to exactly-once effect, because three idempotency boundaries (HTTP key, outbox `message_id`, inbox dedup) catch the duplicates that at-least-once delivery inevitably produces, and the local-transaction atomicity at producer and consumer eliminates loss and phantoms. That is what "reliable publishing" actually means, mechanically, end to end.

## The taxonomy: where the outbox sits among dual-write cures

It helps to see the outbox in context. There are exactly four families of solutions to the dual-write problem, and the tree figure below organizes them. Every one of them either makes a single atomic write or derives the second write from a durable log of the first — there is no third trick.

![A tree taxonomy of dual-write solutions branching into rejected two-phase commit, the transactional outbox with polling and CDC sub-branches, event sourcing, and listen-to-yourself](/imgs/blogs/transactional-outbox-pattern-reliable-publishing-7.webp)

**Two-phase commit / XA** tries to make the two writes one distributed atomic write. We rejected it in section 2: blocking on coordinator failure, held locks across the network, and brokers that are not usable XA participants.

**Transactional outbox** makes the business change and the intent-to-publish one *local* atomic write, then derives the publish from the outbox via a relay — polling or CDC. This is the subject of this post and the right default for most services.

**Event sourcing** sidesteps the dual write entirely by making the event log *the* source of truth. Instead of storing current state and separately publishing events, you store the events themselves as the primary record, and current state is a fold over the event stream. There is only one write — append the event — so there is no second write to diverge. The cost is a fundamental rearchitecture: your storage model is now an append-only event log, your reads come from projections you maintain, and you take on event-schema versioning and snapshotting. Powerful, but a much larger commitment than adding an outbox table to an existing service.

**Listen to yourself** is a lighter variant where the service publishes the event to the broker *first*, then a consumer (often the same service) listens to that event and applies it to its own database. The single write is the publish; the database update is derived from consuming your own event. It avoids the dual write but inverts the trust model — the broker becomes the source of truth and your database becomes a projection, which means reads must tolerate the lag between publish and self-consumption, and a publish that the broker accepts but the self-consumer never processes leaves your own database behind. It works, but it makes your service's own reads eventually consistent with its own writes, which surprises people.

The outbox wins as a default precisely because it is the *least invasive* member of the family that fully solves the problem. It does not require rearchitecting your storage (event sourcing), does not make your own reads eventually consistent (listen-to-yourself), and does not need a distributed transaction coordinator (2PC). You add one table and one process to a service you already have. That low cost-to-adopt, combined with full no-loss-no-phantom correctness, is why it has become the standard answer.

## Case studies and war stories

**The phantom shipment.** A retail team I worked near had a fulfillment service that published `OrderShipped` to a broker and *then* updated the order's status in the database, reasoning that the event was the important part. One afternoon a deploy caused the database update to fail (a migration had briefly locked the table) for a batch of orders while the publishes had already gone out. Customers received "your order has shipped" emails for orders the warehouse had no record of shipping, because the warehouse system read the order status from the database, which still said `PENDING`. The events were phantoms. The fix was textbook outbox: write the `OrderShipped` event into an outbox row in the same transaction as the status update, so the event can only exist if the status change committed. After the change, no event could ever again outrun the truth it described. The lesson the team tattooed onto its design docs: never publish a state-change event outside the transaction that makes the state change.

**The polling relay that ate the database.** A startup adopted the outbox correctly — same-transaction writes, relay publishes then marks sent — but skipped the partial index on unsent rows. For the first few months the outbox table was small and the relay's `SELECT ... WHERE published_at IS NULL` was fast. By month six the table had thirty million sent rows, and without the partial index every poll did a sequential scan over all thirty million to find the handful of unsent ones. The relay polled every 100ms, so the database was running a thirty-million-row scan ten times a second, and database CPU sat at 90%. The fix was one line — `CREATE INDEX ... WHERE published_at IS NULL` — which dropped each poll from scanning thirty million rows to scanning the few hundred actually unsent. The broader lesson: the outbox's correctness is robust, but its *performance* lives or dies on the partial index that keeps the relay's hot query bounded to the unsent set, plus retention that keeps the table from growing without limit.

**Debezium and the disappearing disk.** A platform team ran the outbox via CDC with Debezium on a Postgres logical replication slot. The Debezium connector crashed during a Kafka Connect upgrade and stayed down for six hours over a weekend. Because a logical replication slot holds back WAL until its consumer confirms it has processed up to that point, Postgres retained every WAL segment generated during those six hours — and the database's disk filled, threatening to take down the primary entirely. They caught it at 95% disk usage and restarted the connector, which drained the backlog and released the WAL. The lesson is specific to CDC relays: a replication slot is a contract that the database will *not* discard WAL the connector has not consumed, so a down connector turns into unbounded WAL retention and a disk-full outage. Monitor replication slot lag (`pg_replication_slots`) as a first-class alert when you run a CDC relay; the polling relay does not have this failure mode, which is one more reason to default to polling unless you specifically need CDC.

**The reordered inventory.** A team published inventory adjustments through an outbox but ran the relay with an async Kafka producer at `max.in.flight.requests.per.connection=5` and `enable.idempotence=false`. Most of the time events arrived in order. But under retry — when a transient broker error caused the producer to retry an in-flight send — a later adjustment occasionally overtook an earlier one for the same SKU, and the consumer applied "set stock to 10" before "set stock to 8," leaving the wrong final value. Diagnosing it took a week because it was rare and only under broker stress. The fix was `enable.idempotence=true` on the producer, which makes Kafka sequence and de-duplicate the producer's sends so retries cannot reorder. The lesson: the outbox gives you the *right* order at the source (the `id` column), but you must not let the producer's retry behavior scramble that order in flight. Per-aggregate ordering is only preserved end to end if every link preserves it.

## When to reach for this (and when not to)

Reach for the transactional outbox whenever a service must change its database and publish an event about that change, and you cannot tolerate the event being lost or phantomed. That is the overwhelming majority of event-driven services: order placement, payment processing, user lifecycle events, inventory changes, anything where a downstream system must reliably learn about a state change. If your events drive money, fulfillment, notifications, or any other action with real-world consequences, the outbox is not optional — the dual write *will* lose or phantom events eventually, and you will find out from a customer.

Default to a **polling relay** with a partial index on unsent rows and time-based or partition-drop retention. It is enough for the vast majority of throughput and latency needs, adds no new infrastructure, and is debuggable with SQL. Move to a **CDC relay** only when you have a concrete driver: sub-50ms event latency requirements, outbox throughput high enough that polling load matters, or you are already operating Debezium for other purposes. Always pair the outbox with consumer-side **idempotency** (an inbox table or equivalent), because the outbox is at-least-once by design and an unprepared consumer will double-apply.

Do *not* reach for the outbox when you do not actually have a dual write — if a single service writes to a single database and nothing else needs to know, there is no problem to solve, and adding an outbox is pure overhead. Do not reach for it when **event sourcing** is already your architecture; there the event log is the source of truth and the dual write does not exist. Be cautious about the outbox when your events must be published with strictly lower latency than a relay can provide and you cannot run CDC — though in practice CDC covers the low-latency case, so this is rare. And never reach for **2PC across your database and broker** as the "proper" alternative; it is the option the entire industry tried and abandoned for the reasons in section 2. The outbox is the pragmatic, correct default, and the bar for choosing something else is high.

One sizing reminder that belongs in every adoption decision: provision the relay's throughput ceiling at roughly **2× your peak** outbox arrival rate, not your average. The worked example in section 5 showed why — a relay sized for the average falls behind exactly during the traffic spike, and the lag it accumulates then takes as long to drain as the spike lasted. Headroom is your recovery-time guarantee, not a luxury.

## Key takeaways

- **The dual write is structural, not a coding bug.** Writing to your database and then publishing to a broker are two writes to two systems with no shared transaction; any crash between them loses an event or, if you publish first, creates a phantom event. No ordering of two independent writes is safe.
- **The outbox converts a cross-system atomicity problem into a single-system one.** Write the business change and the event into an `outbox` table in the *same local transaction*; the database's atomicity guarantees both commit or neither does, eliminating loss and phantoms.
- **The relay publishes at-least-once on purpose.** It publishes, waits for the broker ack, then marks the row sent. A crash in between re-publishes a duplicate — never a loss. Duplicates are recoverable; losses and phantoms often are not.
- **A stable `message_id` frozen at insert time is the contract with the consumer.** Every re-publish carries the same id, so the consumer can deduplicate. The relay guarantees delivery; the consumer guarantees idempotent effect; together they compose to effectively-once.
- **Key broker messages by `aggregate_id` for per-aggregate ordering,** and set the producer's idempotence so retries cannot reorder in-flight sends. You get every event about one entity in commit order, which is the ordering you actually want.
- **Default to a polling relay; reach for CDC only with a concrete reason.** Polling needs only a loop, a partial index on unsent rows, and retention. CDC gives lower latency at the cost of a replication slot and connector operations — and a disk-full failure mode if the connector stalls.
- **The partial index and retention are not optional.** Without `WHERE published_at IS NULL` on the index, the relay's hot query degrades to a full scan as history grows; without retention, the table bloats until the database falls over. Partition-and-drop beats `DELETE` at high volume.
- **Mirror the outbox with an inbox on the consumer.** Check-and-record the `message_id` in the same transaction as the business write, then ack. This closes the consumer's symmetric dual write and completes the end-to-end effectively-once guarantee.
- **Monitor the unsent count.** A near-zero unsent count that spikes and drains is healthy; a growing one is your earliest signal that the relay stalled — alert on it like you alert on consumer lag.

## Further reading

- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the database-side companion to this post, covering the CDC internals, the write-ahead log, and Debezium that the CDC relay tails.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the consumer-side machinery that makes the duplicate publishes the relay produces harmless.
- [Delivery semantics: at-most-once, at-least-once, and the exactly-once myth](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — why at-least-once-plus-idempotency is the honest target and what the ack placement decides.
- [Message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — why keying by aggregate id buys you per-aggregate ordering and not a global total order.
- [Kafka replication, ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — what `acks=all` and the in-sync replica set actually guarantee when the relay flushes a publish.
- [Pattern: Transactional outbox](https://microservices.io/patterns/data/transactional-outbox.html) — Chris Richardson's canonical catalog entry for the pattern.
- [Debezium documentation](https://debezium.io/documentation/) — the reference for building a CDC log-tailing relay over Postgres, MySQL, and other databases.

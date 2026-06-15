---
title: "The Transactional Outbox and Reliable Event Publishing"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Why updating your database and publishing an event are two writes that can't be made atomic the naive way, how the outbox pattern folds the event into your local transaction to fix it, and how to build the relay, the dedup, and the cleanup so events are never lost or invented."
tags:
  [
    "microservices",
    "transactional-outbox",
    "event-driven-architecture",
    "change-data-capture",
    "distributed-systems",
    "software-architecture",
    "backend",
    "kafka",
    "idempotency",
    "reliability",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-1.webp"
---

The ShopFast order service had a bug that only showed up about once every ten thousand orders, which is the worst possible frequency: rare enough that nobody could reproduce it on demand, common enough that it generated a steady trickle of furious support tickets. The symptom was always the same. A customer placed an order, got a confirmation page, got charged — and then nothing. No shipping email, no warehouse pick, no inventory decrement. The order simply sat in the database in state `PLACED` forever, an orphan that no downstream service had ever heard of. The inventory service swore it had never received an `OrderPlaced` event for that order. The order service swore it had published one. Both were telling the truth, and that is exactly why it took three engineers two weeks to find it.

The code looked completely reasonable. When a customer checked out, the order service did two things: it inserted the order row into its Postgres database and committed, and then it published an `OrderPlaced` event to Kafka so inventory, shipping, payment-reconciliation, and analytics could all react. Two operations, in the obvious order, exactly as a hundred tutorials show it. The problem is that those two operations are two *separate* writes to two *separate* systems with no transaction spanning them, and once in a while — during a deploy, a node drain, an OOM kill, a network blip to the broker — the process committed the database row and then died, was killed, or timed out before the publish landed. The order existed. The event never did. There was no crash, no error log, no alert. The system had quietly, permanently, lost a fact that the rest of the company depended on.

This is the **dual-write problem**, and it is the single most common way that event-driven microservices silently corrupt themselves. It is so easy to write the buggy version and so hard to see why it is buggy that I would estimate the majority of teams shipping their first event-driven service have this bug right now and do not know it. The fix is a pattern with an unglamorous name and an enormous payoff: the **transactional outbox**. By the end of this post you will be able to look at any "update my database and then tell other services" code path and immediately see whether it loses events, explain to a skeptical teammate exactly why a database transaction and a broker publish cannot be made atomic the naive way, and build the outbox plumbing — the atomic write, the relay that drains it, the dedup on the consumer side, and the retention job that keeps the table from eating your disk — correctly enough to run it in production.

![A before and after comparison contrasting the naive dual-write that loses or invents events against the outbox pattern that commits the event atomically with the state change](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-1.webp)

We are deliberately staying at the practitioner's layer here: the *integration* problem and how to build it right. The mechanics of how a database's write-ahead log works, and the deep theory of change data capture, live in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern); the broker-side reliability story — how a message system gives you at-least-once, and the internals of the outbox as a publishing mechanism — lives in [the transactional outbox pattern for reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing). I will cross-link to those for the internals and concentrate on what a microservices engineer actually has to decide, type, and operate.

## The dual-write problem: why two writes can't be one

Let us be painfully precise about what "dual write" means, because the precision is the whole insight. Your order service has a piece of business state — the order row — that lives in its own database. It also has an obligation to the rest of the system: when an order is placed, other services need to know. The two things you must do are therefore **write your state** (insert the order) and **publish a fact** (emit `OrderPlaced`). The trap is that these are writes to two different storage systems: a relational database and a message broker. There is no transaction that spans both. There is no `BEGIN; insert into orders; publish to kafka; COMMIT;` because the database does not know about Kafka and Kafka does not know about your database. They are separate processes, separate failure domains, separate durability guarantees. You cannot wrap them in one atomic unit, no matter how much you want to.

So you do them one after another, and now you have a sequencing problem with two equally bad orderings. Suppose you **commit the database first, then publish**. Most of the time both succeed and you are fine. But if the process dies in the gap — the commit landed, the publish did not — you have an order that exists in your database that no one else will ever hear about. The event is **lost**. This is ShopFast's bug exactly: a phantom order, perfectly real in your DB, invisible to everyone downstream. No charge gets reconciled, no stock gets reserved, no email gets sent. The order is a ghost.

Now suppose you flip it: **publish first, then commit the database**. If the process dies in *that* gap — the publish landed, the commit did not — you have told the whole company that an order was placed, and consumers have reserved stock and started a fulfillment workflow, but the order does not actually exist in your database. The event is a **phantom**: a fact about state that never came to be. Inventory is now decremented for an order you will never ship, the customer was promised a confirmation for an order you have no record of, and reconciliation will scream. This is arguably worse than the lost event, because the damage is *active* rather than *passive* — downstream side effects fired against a state change that did not happen.

The reason this bug is so pervasive deserves naming directly, because understanding *why* good engineers ship it is how you stop shipping it. The naive dual-write works flawlessly in every demo, every local test, every code review, and the first weeks of production — because the failure requires a crash in a window that is only milliseconds wide, and crashes are rare, and the intersection of "crash" and "this specific 5ms window" is rarer still. So the code passes review (it looks obviously correct), passes tests (no test kills the process mid-handler), and runs clean for a while. The bug does not announce itself; it accrues silently as a slow drip of lost events that only becomes visible when someone correlates "orders stuck in PLACED" with "no downstream record" weeks later. This is the most dangerous category of bug there is: one that is invisible, probabilistic, and proportional to your success — the more traffic you handle and the more often you deploy, the more events you lose. A junior who has internalized "two writes to two systems is the bug" will never ship it; everyone else ships it once and learns the hard way.

There is no third ordering that escapes the dilemma, because the dilemma is structural: any time you must perform two writes to two systems with no shared transaction, a crash in the window between them leaves the two systems disagreeing about what is true. People reach for clever-sounding escapes and they all fail. "Publish, wait for the broker's acknowledgment, *then* commit" still loses the commit if the process dies after the ack. "Commit, then retry the publish until it succeeds" loses everything if the process dies before the first retry attempt and holds no record of the pending publish. "Use a distributed transaction across the DB and the broker" — a two-phase commit, or XA — technically exists, but it is a operational nightmare that couples the availability of your database to the availability of your broker (if either is down, neither write can commit), introduces a blocking coordinator that can leave transactions in-doubt, and is not even supported by most modern brokers. Two-phase commit is the cure that is worse than the disease, and the broader reason it falls down is the same reason distributed transactions fail in general; the deep treatment of why is in [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), which exists precisely because 2PC across services does not scale.

![A six event timeline showing a process crash in the window between committing the order row and publishing the event, which leaves a phantom order no downstream consumer ever hears about](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-2.webp)

The timeline above is the failure in slow motion. At `T+0` the order row commits. At `T+3ms` the service returns `201 Created` to the customer — the order is now real and acknowledged. At `T+5ms` the publish call to Kafka begins. At `T+6ms` a Kubernetes node drain kills the pod (this happens constantly — node upgrades, autoscaler scale-downs, spot-instance reclaims; a pod has no right to expect a graceful death). The event is never sent. Minutes later the inventory service is still blind, stock was never reserved, and the order is stuck. Note the cruelty of the timing: the customer already has their confirmation, so there is no retry path from the client. The fact is lost the instant the pod dies.

#### Worked example: how often does the naive dual-write actually lose an event?

Engineers under-rate this bug because the window feels tiny — "the gap between commit and publish is a few milliseconds, what are the odds?" Let us put real numbers on it. Suppose the publish path (serialize, send to broker, await ack) takes a mean of 5ms, and during that 5ms the process is vulnerable: if it dies in that window after the commit, the event is lost. ShopFast does 200 orders per second at peak, so 200 commits per second each open a 5ms vulnerability window. That is `200 × 0.005 = 1.0` second of cumulative vulnerable time per wall-clock second — meaning at any given instant there is, on average, roughly one order sitting in the danger zone.

Now, how often does the process die? In a healthy cluster a given order-service pod might be killed (deploy, node drain, OOM, crash) a few times a day. Say the *fleet* experiences an ungraceful pod termination every 30 minutes during a busy day — a rolling deploy alone recycles every pod, and ShopFast deploys several times a day. Each termination has, on average, ~1 order in the 5ms window at the moment it dies. So you lose on the order of one event per ungraceful termination, and at one termination per 30 minutes that is ~48 lost events per day, or roughly one lost order in every `200 × 86400 / 48 ≈ 360,000` — close enough to ShopFast's "one in ten thousand" once you account for deploys clustering during traffic and the window being longer than 5ms whenever the broker is slow. The point is not the exact rate. The point is that **the rate is never zero, it scales with your traffic and your deploy frequency, and it is completely invisible** because a lost event produces no error. You do not find this bug by reading logs. You find it by a customer telling you their order vanished.

## The outbox pattern: make the event part of the transaction

Here is the move that dissolves the whole problem, and it is almost embarrassingly simple once you see it. You cannot make a transaction span your database and the broker. But you *can* make a transaction span two tables in your *own* database. So instead of publishing the event to the broker inside your request handler, you **write the event into an `outbox` table in the same local database transaction as the state change**. One transaction, two inserts: the order row and the outbox row. They commit together or they roll back together. There is now no window in which the order exists but the event does not, because the event *is in your database*, sitting in the outbox, committed atomically with the order.

Then, completely separately and asynchronously, a **relay** (sometimes called a message relay, publisher, or dispatcher) reads unpublished rows from the outbox table, publishes them to the broker, and marks them as sent. The relay can crash, retry, run slow, fall behind — none of that loses an event, because the event is durably in the outbox until the relay confirms it reached the broker. The worst the relay can do is publish an event *twice* (if it crashes after publishing but before marking the row sent), and a duplicate is a problem we already know how to solve: idempotent consumers. We have traded an *unsolvable* problem (lost events) for a *solved* one (duplicate events). That is the entire trick.

![A graph showing one local transaction writing the order and outbox rows, a relay draining the outbox to a Kafka topic, and inventory, shipping, and analytics consumers fanning out from the same durable event](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-3.webp)

Look at the dataflow above. The order transaction writes both the order row and the outbox row. The outbox holds the event in `status=pending`. The relay — whether it polls the table or tails the database log, which we will get to — reads pending rows and publishes them to the `orders.events` Kafka topic. From there the event fans out to the inventory service, the shipping service, and analytics, each of which dedups by event id so a redelivery is harmless. The critical structural property: the order transaction's only job is to write to its *own* database. It never touches the broker. The broker is entirely the relay's concern, on the relay's own schedule. The boundary between "my state changed" and "the world finds out" is now an asynchronous, durable, crash-safe handoff through a table.

Why does this actually work where everything else failed? Because **atomicity within a single database is a solved problem**. Your database's transaction machinery — the write-ahead log, the fsync at commit, the ACID guarantees you already rely on for not losing the order row itself — now also covers the event. If the order row survives a crash, so does its outbox row, because they are the same transaction, the same WAL record, the same fsync. You are not inventing a new reliability mechanism; you are *reusing the one your database already gives you* and extending its coverage from "my state" to "my state plus the fact that my state changed." That reuse is the elegance of the pattern. The deep version of why a single-database transaction is atomic and a cross-system one cannot be is in [the CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) and the consistency-models discussion; for our purposes the takeaway is operational: keep the event in the same blast radius as the state, and the database does the hard part for free.

![A stack showing one transaction committing the order row, the outbox row, a single write-ahead log record, and one fsync, so the state change and its event are atomic at the storage layer](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-4.webp)

The stack above makes the atomicity concrete: `BEGIN`, insert the order row (business state), insert the outbox row (`OrderPlaced`), and both inserts ride into one WAL record and one `COMMIT`/fsync. There is no point at which one is durable and the other is not. That single shared fsync is the whole guarantee.

### The outbox table schema

Let us make it real. Here is the outbox table I would actually create. The columns are chosen deliberately, and each one earns its place.

```sql
CREATE TABLE outbox (
    id              BIGSERIAL PRIMARY KEY,          -- monotonic, gives ordering
    aggregate_type  TEXT        NOT NULL,           -- e.g. 'order'
    aggregate_id    TEXT        NOT NULL,           -- e.g. 'A-8842' (used as partition key)
    event_type      TEXT        NOT NULL,           -- e.g. 'OrderPlaced'
    payload         JSONB       NOT NULL,           -- the event body, serialized once
    event_id        UUID        NOT NULL DEFAULT gen_random_uuid(), -- stable dedup key
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at    TIMESTAMPTZ,                    -- NULL = not yet published
    attempts        INT         NOT NULL DEFAULT 0
);

-- The relay's hot path: find unpublished rows in order, fast.
CREATE INDEX idx_outbox_unpublished
    ON outbox (id)
    WHERE published_at IS NULL;
```

The `event_id` is the **stable identity of this event for its entire life** — it is generated once, in the transaction, and it travels with the event through the broker to every consumer, where it becomes the dedup key. This is the single most important column for correctness on the consumer side, and people forget it constantly. The `aggregate_id` is the **ordering and partitioning key**: all events for order `A-8842` carry the same `aggregate_id`, so when you publish to Kafka you use it as the message key and all of that order's events land in the same partition, preserving their order. The partial index on `published_at IS NULL` is the relay's performance lifeline — it keeps the "find pending rows" query cheap even when the table has millions of already-published rows, which it will. We will revisit that index when we talk about cleanup, because it interacts with retention in a way that bites people.

### The atomic write in application code

Here is the order service's checkout handler, doing the atomic write. The whole pattern lives in the fact that both inserts share one transaction:

```python
import json
import uuid

def place_order(conn, customer_id: str, items: list, total_cents: int) -> str:
    order_id = f"A-{next_order_seq(conn)}"
    event_id = str(uuid.uuid4())

    # ONE transaction. Both inserts commit together or not at all.
    with conn.transaction():                      # BEGIN ... COMMIT
        conn.execute(
            """INSERT INTO orders (id, customer_id, total_cents, status)
               VALUES (%s, %s, %s, 'PLACED')""",
            (order_id, customer_id, total_cents),
        )
        conn.execute(
            """INSERT INTO outbox
                 (aggregate_type, aggregate_id, event_type, payload, event_id)
               VALUES ('order', %s, 'OrderPlaced', %s, %s)""",
            (
                order_id,
                json.dumps({
                    "event_id": event_id,         # carried into the payload too
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "items": items,
                    "total_cents": total_cents,
                    "occurred_at": now_iso(),
                }),
                event_id,
            ),
        )
    # COMMIT happened here. The order AND the event are now durable, atomically.
    # We do NOT publish to Kafka here. The relay will. Notice the broker is
    # never even imported in this file.
    return order_id
```

Read what is *not* in that function: there is no Kafka producer, no broker connection, no publish call, no try/except around a network operation that might leave you half-committed. The handler's entire universe is its own database. It returns the instant the local transaction commits, which is fast and predictable, and the customer gets their `201` knowing the event is as durable as the order. The dual-write window is gone because there is no longer a second write to a second system in the request path. This is the discipline that makes the pattern work: **the request handler must never touch the broker.** If you see a `producer.send()` next to a `db.commit()` in a code review, that is the bug, every time.

## Two relay strategies: polling vs log tailing (CDC)

The event is safely in the outbox. Now something has to get it to the broker. There are two families of relay, and choosing between them is the main engineering decision in this whole pattern. They differ in how they discover that new rows have appeared.

![A before and after comparison of a polling publisher that runs a SELECT loop you own against change data capture that tails the write-ahead log for lower latency but more operational parts](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-5.webp)

### Strategy one: the polling publisher

The polling publisher is exactly what it sounds like. A background process — a goroutine in your service, a sidecar, or a separate worker deployment — wakes up on an interval, runs `SELECT * FROM outbox WHERE published_at IS NULL ORDER BY id LIMIT N`, publishes those rows to the broker, marks them published, and goes back to sleep. That is the whole thing. It is simple, it has no dependencies beyond your database and your broker, you fully understand and own every line, and you can debug it with a `SELECT`. For a huge number of services this is the correct choice and I would reach for it first.

Here is a polling relay I would actually run in production — note the locking, the batching, and the careful ordering of operations:

```python
import time

POLL_INTERVAL_S = 1.0
BATCH_SIZE = 500

def run_relay(conn, producer):
    while True:
        published_any = poll_once(conn, producer)
        # Adaptive: if we drained a full batch, there's probably more — don't sleep.
        if not published_any:
            time.sleep(POLL_INTERVAL_S)

def poll_once(conn, producer) -> bool:
    with conn.transaction():
        # FOR UPDATE SKIP LOCKED lets multiple relay replicas run safely:
        # each grabs a disjoint batch, none block on the others.
        rows = conn.execute(
            """SELECT id, aggregate_id, event_type, payload, event_id
                 FROM outbox
                WHERE published_at IS NULL
                ORDER BY id
                LIMIT %s
                FOR UPDATE SKIP LOCKED""",
            (BATCH_SIZE,),
        ).fetchall()

        if not rows:
            return False

        for r in rows:
            # Key by aggregate_id so one order's events stay in one partition (ordered).
            producer.send(
                topic="orders.events",
                key=r["aggregate_id"].encode(),
                headers=[("event_id", r["event_id"].encode()),
                         ("event_type", r["event_type"].encode())],
                value=r["payload"],  # already JSON bytes
            )
        producer.flush()  # block until the broker acks every message in the batch

        # Only mark published AFTER the broker has acked the whole batch.
        ids = [r["id"] for r in rows]
        conn.execute(
            "UPDATE outbox SET published_at = now() WHERE id = ANY(%s)",
            (ids,),
        )
    return True
```

The ordering of operations in `poll_once` is load-bearing. We select pending rows, publish them, **block on `producer.flush()` until the broker has acknowledged every message**, and only *then* mark them published in the same transaction we selected them in. If the relay crashes after `flush()` but before the `UPDATE` commits, those rows are still `published_at IS NULL`, so the next poll re-publishes them — a duplicate, which the consumer dedups. If it crashes before `flush()` acks, same thing. There is no ordering of the crash that loses an event; every crash window resolves to "re-publish," never "drop." That is the at-least-once guarantee falling out of the structure. The `FOR UPDATE SKIP LOCKED` is what lets you run several relay replicas for throughput without them stepping on each other: each replica locks and claims a disjoint batch, and the others skip the locked rows instead of blocking.

The cost of polling is right there in the loop: you are issuing a query every interval whether or not there is work, and your publish latency is bounded below by your poll interval. If you poll every second, an event sits in the outbox for up to a second before anyone hears about it. That is fine for most domains (a confirmation email a second later is invisible to a human) and unacceptable for a few (a fraud signal, a real-time price update). You can tighten the interval, but tighter polling means more queries against your primary database, and now you are trading database load for latency.

#### Worked example: polling interval vs publish latency vs database load

Let us quantify the trade so you can pick an interval with numbers instead of vibes. ShopFast runs the order service across 6 pods, each running the relay loop, polling the primary Postgres. Consider the empty-poll cost first: when there is no work, each poll is a single indexed query against the partial index, which on a warm database costs maybe 0.3ms of database CPU. At a 1-second interval across 6 pods that is `6 polls/sec × 0.3ms = 1.8ms` of database CPU per second — i.e. 0.18% of one core. Negligible. Drop the interval to 100ms and it becomes `60 polls/sec × 0.3ms = 18ms/sec`, 1.8% of a core — still cheap. Drop it to 10ms and it is 18% of a core *just for empty polls*, which on a database that is also serving your order writes is now a real tax you would notice.

Now the latency side. With a 1-second interval, mean added publish latency is ~500ms (uniform over the interval) and worst-case is ~1s plus the publish time. With a 100ms interval, mean ~50ms, worst-case ~100ms. So the question is purely: is the difference between "consumers hear about it in ~500ms" and "in ~50ms" worth roughly 10× the (still small) database poll load? For inventory reservation, shipping, and analytics — no, 500ms is invisible, run the 1-second poll and stop thinking about it. For a real-time fraud-scoring consumer that wants sub-100ms — maybe, but at that point you should ask whether polling is the right tool at all, which is the cue for CDC. The decision rule I use: **start at a 1-second poll; only tighten it if a specific consumer has a measured latency requirement the interval violates; and if you find yourself wanting sub-100ms, switch to CDC rather than hammering the database with a 10ms poll.**

### Strategy two: log tailing with change data capture (Debezium)

The other family eliminates polling entirely by reading the database's own write-ahead log. Every committed write to Postgres — including the insert into your outbox table — produces a record in the WAL (the replication stream Postgres uses to feed replicas). A change-data-capture tool, most famously **Debezium**, connects as a logical-replication consumer, reads the WAL stream, and turns each outbox insert into a message it publishes to Kafka. No polling, no `SELECT` load on your primary, and latency measured in milliseconds because the relay sees the row essentially the moment it commits. The full mechanism — logical decoding, replication slots, how the WAL becomes an event stream — is the subject of [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and I will not re-derive it; here is what it looks like to *configure and operate*.

A Debezium Postgres connector pointed at the outbox table, using Debezium's purpose-built outbox event router so you get clean events on Kafka rather than raw row-change envelopes:

```json
{
  "name": "shopfast-outbox-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "orders-db.internal",
    "database.port": "5432",
    "database.user": "debezium",
    "database.dbname": "orders",
    "plugin.name": "pgoutput",
    "slot.name": "shopfast_outbox_slot",
    "publication.name": "shopfast_outbox_pub",
    "table.include.list": "public.outbox",

    "transforms": "outbox",
    "transforms.outbox.type":
        "io.debezium.transforms.outbox.EventRouter",
    "transforms.outbox.table.field.event.id": "event_id",
    "transforms.outbox.table.field.event.key": "aggregate_id",
    "transforms.outbox.table.field.event.type": "event_type",
    "transforms.outbox.table.field.event.payload": "payload",
    "transforms.outbox.route.by.field": "aggregate_type",
    "transforms.outbox.route.topic.replacement": "${routedByValue}.events",

    "heartbeat.interval.ms": "10000",
    "tombstones.on.delete": "false"
  }
}
```

Walk through what this buys you. The `EventRouter` transform reads each outbox row off the WAL and maps its columns onto a clean Kafka message: `event_id` becomes the message id (your dedup key, carried in a header), `aggregate_id` becomes the partition key (preserving per-order ordering), `payload` becomes the message value, and `aggregate_type` routes it — every `order` event lands on the `order.events` topic. You write rows; Debezium produces clean domain events. The `slot.name` is a Postgres **replication slot**, and this is the operational gotcha that catches everyone: a replication slot makes Postgres retain WAL until the consumer has read past it. If Debezium goes down and stays down, Postgres keeps WAL forever waiting for it, and your primary's disk fills up and the database stops accepting writes. A dead CDC connector can take down your database. That is the price of CDC's low latency and you must monitor `pg_replication_slots` lag and alert on it.

The trade between the two strategies is the heart of this post's decision. Polling is dead simple, fully owned, debuggable with SQL, costs you a little database load and a little latency, and has no exotic failure modes. CDC gives you the lowest possible latency and zero polling load on the primary, but it adds a whole subsystem to run — a Kafka Connect cluster (or equivalent), a connector to configure and version, replication slots to monitor, and a new failure mode (the slot-bloat disk-fill) that is more dangerous than anything polling can do. There is also the `tombstones.on.delete: false` line, which matters because Debezium would otherwise emit a tombstone message every time your cleanup job deletes a published outbox row — you do not want your cleanup generating spurious events.

## The decision matrix

Here is the whole landscape in one table. The naive dual-write is included on purpose, as the option you must never ship, so the contrast is explicit.

![A matrix comparing naive dual-write, outbox with polling, outbox with CDC, and listen-to-yourself across no lost events, publish latency, steady database load, operational complexity, and ordering control](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-6.webp)

| Property | Naive dual-write | Outbox + polling | Outbox + CDC | Listen-to-yourself |
| --- | --- | --- | --- | --- |
| No lost events | No — drops on crash | Yes | Yes | Yes |
| Publish latency | Lowest (in-request) | Bounded by poll interval | Sub-second | Sub-second |
| Steady DB load | None extra | Poll queries | WAL read only | WAL read only |
| Ops complexity | Trivial | Low | Connector + slots | High |
| Ordering control | None | Per aggregate key | Per partition | Per partition |
| When it wins | Never (it's a bug) | Default for most teams | High volume, low latency | Already event-sourced |

The **listen-to-yourself** column needs a word, because it is a real fourth option people use and it confuses juniors. In that variant, the service does not write to its own database synchronously at all; instead it publishes the event first (the outbox or CDC still guarantees the publish is reliable), and then the service *consumes its own event* to update its own read model, the same way every other consumer does. This collapses "update my state" and "tell the world" into a single event-driven path — your state is just another projection of the event stream. It is elegant when you have already gone all-in on event sourcing, where the event *is* the source of truth and the database row is a derived view; the connection to that world is [event sourcing and CQRS in microservices](/blog/software-development/microservices/event-sourcing-and-cqrs-in-microservices). For a service whose database is still the source of truth, listen-to-yourself adds a confusing layer of indirection (your own write is now eventually consistent with itself) and I would not reach for it unless you are already living in event-sourced land. For everyone else, the real choice is **polling vs CDC**, and the honest default is polling until you measure a reason to switch.

My one-line recommendation, said plainly because the kit demands I name costs: **use outbox-plus-polling by default — it solves the dual-write problem completely with code you fully own and a sub-second latency that is invisible for almost every business event. Pay the operational tax of CDC only when you have a measured latency requirement polling cannot meet, or when poll load on the primary becomes material at high volume, and when you do, budget for monitoring replication-slot lag because a dead connector can fill your primary's disk.**

## At-least-once means consumers must be idempotent

Everything above guarantees the event is **never lost**. None of it guarantees the event is **delivered exactly once** — and it can't, because the relay's only safe failure behavior is to re-publish on crash. The outbox gives you **at-least-once delivery**: every event reaches the broker one or more times. The "or more" is not a defect you can engineer away; it is the necessary cost of never losing anything. (The taxonomy of at-most-once, at-least-once, and exactly-once, and why exactly-once delivery is largely a myth while exactly-once *effects* are achievable, is laid out in [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).)

This pushes one non-negotiable obligation onto every consumer: **idempotency**. A consumer that is not idempotent will, the first time the relay redelivers an event, double-reserve stock, double-charge a card, send two emails, or award loyalty points twice. The fix is **deduplication by event id**, and the most robust form of it is the **inbox pattern** — the mirror image of the outbox on the consumer side.

![A graph showing a consumer checking an inbox table for the event id, skipping duplicates, and otherwise applying the effect and inserting the event id in the same transaction so a redelivery is a no-op](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-7.webp)

The inbox is a table where the consumer records the `event_id` of every event it has successfully processed, **inside the same transaction as the effect**. When an event arrives, the consumer first checks: have I seen this id? If yes, it acknowledges the message and does nothing (the effect already happened). If no, it applies the effect *and* inserts the id into the inbox in one transaction. The atomicity is the whole point: the side effect and the record-that-I-did-it commit together, so there is no window where the effect happened but the id was not recorded (which would let a redelivery re-apply it). Here is the inventory consumer doing exactly that:

```python
def on_order_placed(conn, event):
    event_id = event.headers["event_id"]
    order_id = event.payload["order_id"]

    with conn.transaction():
        # Dedup: try to claim this event_id. If it's already there, this is a
        # redelivery — do nothing and let the caller ack the message.
        claimed = conn.execute(
            """INSERT INTO inbox (event_id, consumer, processed_at)
               VALUES (%s, 'inventory', now())
               ON CONFLICT (event_id, consumer) DO NOTHING
               RETURNING event_id""",
            (event_id,),
        ).fetchone()

        if claimed is None:
            return  # duplicate; already processed. Ack and move on.

        # First time we've seen it: apply the effect IN THE SAME TXN as the claim.
        for line in event.payload["items"]:
            conn.execute(
                """UPDATE inventory
                      SET reserved = reserved + %s
                    WHERE sku = %s""",
                (line["qty"], line["sku"]),
            )
    # COMMIT: the reservation and the dedup record are now atomic.
```

```sql
CREATE TABLE inbox (
    event_id    UUID        NOT NULL,
    consumer    TEXT        NOT NULL,         -- which consumer processed it
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, consumer)          -- one row per (event, consumer)
);
```

The composite primary key `(event_id, consumer)` is deliberate: the *same* event is legitimately processed by many consumers (inventory, shipping, analytics), and each must dedup independently. Inventory seeing `event_id=X` for the first time is a new event for inventory even if shipping processed `X` an hour ago. The `ON CONFLICT DO NOTHING ... RETURNING` is the idiom that makes the check-and-claim atomic and lock-free under concurrency — two threads racing on the same `event_id` cannot both win the insert, so exactly one applies the effect. This is the same dedup machinery the message-queue post derives in detail; if you want the broader treatment of making at-least-once safe, including dedup windows and why a natural business key sometimes beats a synthetic event id, read [idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). The cross-service version — coordinating idempotent *effects* across a whole call graph rather than one consumer — is the forward-looking topic of [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services), which builds directly on this inbox.

#### Worked example: a duplicate publish with and without the inbox

Make the failure and the fix concrete. The relay publishes `OrderPlaced` for order `A-8842` (`event_id = e7f3...`) to Kafka, the broker acks, and then — before the relay's `UPDATE outbox SET published_at` commits — the relay pod is OOM-killed. The row is still `published_at IS NULL`. The next relay poll finds it and publishes `A-8842` *again*. The inventory consumer now receives `e7f3...` twice.

Without the inbox: the consumer runs `UPDATE inventory SET reserved = reserved + 3` twice. SKU `WIDGET-1`'s reserved count goes up by 6 instead of 3. Three phantom units are now reserved against an order that needs three. Multiply across a relay restart that re-publishes a batch of 500 rows and you have over-reserved hundreds of SKUs; inventory reports the warehouse is out of stock when it is not, oversell protection blocks real orders, and someone gets paged about a stock discrepancy that traces back to a duplicate publish nobody can see in the logs. With the inbox: the second delivery hits `INSERT INTO inbox ... ON CONFLICT DO NOTHING`, the insert returns nothing, the consumer returns early, and `reserved` is incremented exactly once. The reservation is `+3`, correct, no matter how many times `e7f3...` is delivered. The duplicate is a non-event. **This is why the outbox and the inbox are a matched pair: the outbox makes delivery at-least-once, and the inbox makes processing exactly-once-in-effect.** Shipping one without the other is half a solution.

## Ordering, partitioning, and the listen-to-yourself note

Two events for the same order must usually be applied in the order they happened: `OrderPlaced` before `OrderCancelled`, `PaymentRequested` before `PaymentCaptured`. The outbox gives you the *means* to preserve this, but only if you use it deliberately. Two mechanisms combine.

First, the outbox's `BIGSERIAL id` is monotonic, so when the relay reads `ORDER BY id` it reads events in commit order. Second — and this is the one people miss — Kafka only guarantees ordering *within a partition*, and a topic has many partitions for parallelism. If two events for order `A-8842` land in different partitions, two different consumer instances might process them concurrently in either order. The fix is to **key by `aggregate_id`**, which is exactly what both relays above do: keying by `A-8842` hashes all of that order's events to the same partition, where Kafka's per-partition ordering and a single consumer per partition guarantee in-order processing. The deep version of this — partition count, key skew, how a hot key serializes a whole partition — is [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees), but the practitioner's rule is short: **key your events by the aggregate id, and you get per-aggregate ordering for free; forget to, and you get a heisenbug where an order occasionally cancels before it places.**

There is a subtle interaction with the polling relay's multi-replica mode. `FOR UPDATE SKIP LOCKED` lets two replicas grab disjoint batches, which is great for throughput but means batch B (claimed by replica 2) might publish before batch A (claimed by replica 1) if replica 1 is slow. Since Kafka ordering is per-partition and you key by aggregate, this is usually fine — two events for *the same order* will both be in whichever batch contains the lower id, because a single order's events are written close together and a batch reads a contiguous id range. But if you are paranoid about strict global ordering, run a single relay replica (you lose horizontal scale but gain strict id-order publishing) or accept per-aggregate ordering as sufficient, which it almost always is. CDC sidesteps this entirely: Debezium reads the WAL in commit order by construction, so it publishes in commit order without any locking dance.

The **event-carried state transfer** consideration belongs here too. There are two philosophies for what goes in the event payload: a thin **notification** event that says only "order A-8842 was placed, go look it up," or a fat **state-transfer** event that carries the full order — items, prices, customer — so consumers never have to call back. The outbox makes either work, but the choice has a reliability consequence. A thin event forces every consumer to make a synchronous callback to the order service to get details, which re-introduces the runtime coupling and availability multiplication that events were supposed to remove — if the order service is down, every consumer stalls. A fat state-transfer event lets consumers work entirely from the event, fully decoupled, at the cost of larger messages and the need to version the schema carefully (covered in [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing)). I lean toward state-transfer events for exactly this reason; the relationship between event style and coupling is explored further in [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration). The outbox row's `payload` is where you make this call — make it fat enough that consumers do not have to phone home.

## Optimization: making the relay production-grade

The naive relay publishes one row, marks it, publishes the next. That works and it is correct, but it leaves a lot of throughput on the table, and throughput is what determines how fast you drain a backlog after an outage — which is the metric that actually matters at 3am.

The first optimization is **batching**, which the production relay above already does: select N rows, publish them all, `flush()` once to await all the acks, then one `UPDATE ... WHERE id = ANY(...)` to mark the whole batch. Batching turns N round-trips to the broker and N database updates into one of each. Concretely, a single-row relay limited by a 2ms broker round-trip caps at ~500 events/sec per replica. A batch of 500 with one flush amortizes that round-trip across 500 events and, with the broker pipelining sends, comfortably exceeds 10,000 events/sec per replica. The batch size is a tuning knob: bigger batches amortize better but increase the latency of the last row in a batch and the memory held per poll. I start at 500 and only change it with a measurement.

The second is the **latency win of CDC over polling**, with numbers. Under a 1-second poll, an event's mean time-in-outbox is ~500ms and p99 is near the full second. Under CDC, the event is read off the WAL within milliseconds of commit; p99 time-to-Kafka is typically well under 200ms dominated by Debezium's flush and the network. So CDC cuts publish latency by roughly an order of magnitude (from hundreds of milliseconds to tens). Whether that matters is the per-consumer judgment from the earlier worked example — for most events it does not, which is why polling remains the default despite being "slower."

The third is **partition and consumer sizing for drain rate**, which is really about the next worked example.

#### Worked example: outbox backlog growth when the relay is down for 10 minutes

This is the scenario you will actually be paged for. The relay (or the Kafka Connect cluster, or the broker) goes down at 14:00. Orders keep arriving — the order service is fine, it is still committing to its database and outbox, because the whole point of the outbox is that the request path does not depend on the relay. So the outbox **grows** at the arrival rate while no one drains it.

![A six event timeline showing the outbox growing at the arrival rate while the relay is down for ten minutes, then draining faster than it filled once the relay recovers and outpaces new writes](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-8.webp)

Run the numbers. Arrival rate is 200 events/sec. The relay is down for 10 minutes = 600 seconds. Backlog at recovery: `200 × 600 = 120,000` rows sitting `published_at IS NULL`. The good news: the order service kept working the entire time, no orders were lost, no customer saw an error — the backlog is purely deferred work, which is exactly the resilience the outbox buys. Now the relay recovers and must drain 120,000 rows *while new events keep arriving at 200/sec*. If the relay drains at 2,000 events/sec (batched, as above), the net drain rate is `2,000 − 200 = 1,800` events/sec, so the backlog clears in `120,000 / 1,800 ≈ 67 seconds`. The system is fully caught up about a minute after recovery.

This gives you the operational design rule: **your relay's drain throughput must comfortably exceed your peak arrival rate, with enough headroom to clear a realistic outage backlog in minutes, not hours.** Here the relay drains at 10× the arrival rate, so even a 10-minute outage clears in ~1 minute. If your relay could only do 250/sec against a 200/sec arrival rate, the same outage would clear at a net 50/sec and take `120,000 / 50 = 2,400` seconds = 40 minutes — and that is during which every downstream service is 40 minutes behind reality. The headroom is not optional; it is the difference between an outage being a blip and being a multi-hour consistency lag. Size your relay (batch size, replica count, broker partitions to publish into in parallel) so that drain rate is several multiples of peak arrival, and alert on **outbox lag** (count of `published_at IS NULL` rows, and age of the oldest one) so you find out the relay is down in seconds, not when a customer complains.

## How to deploy and operate the relay

Where the relay *runs* is a real decision, and the two common shapes have different operational personalities. The first is the **in-process relay**: the polling loop runs as a background thread inside the same order-service pods that handle requests. It is the least infrastructure — no extra deployment, no extra image, the relay ships with the service — and for a small or medium service it is perfectly fine. Its weakness is that the relay's resource needs (which spike during backlog drain) compete with the request path inside the same pod, and you cannot scale the relay independently of the API. If a backlog drain pegs CPU, your request latency suffers, exactly when you least want it to.

The second shape is the **standalone relay deployment**: a separate process, separate image, separate Kubernetes Deployment, that does nothing but drain the outbox. This is the shape I prefer for any service whose events matter, because it decouples the publishing concern from the request-serving concern — you scale the relay on backlog and the API on request load, independently, and a misbehaving relay cannot starve your request path. The cost is one more thing to deploy and monitor. Here is the standalone relay as a Kubernetes Deployment, sized for ShopFast's drain-rate math:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orders-outbox-relay
  labels: { app: orders-outbox-relay }
spec:
  replicas: 3                       # SKIP LOCKED lets replicas share the table
  selector:
    matchLabels: { app: orders-outbox-relay }
  template:
    metadata:
      labels: { app: orders-outbox-relay }
    spec:
      containers:
        - name: relay
          image: shopfast/orders-outbox-relay:1.7.2
          env:
            - name: POLL_INTERVAL_MS
              value: "1000"
            - name: BATCH_SIZE
              value: "500"
            - name: KAFKA_ACKS
              value: "all"           # never mark published until durably acked
            - name: DB_DSN
              valueFrom:
                secretKeyRef: { name: orders-db, key: relay-dsn }
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits:   { cpu: "1",    memory: "512Mi" }
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
            periodSeconds: 5
```

Three replicas with `SKIP LOCKED` give you ~3× the single-replica drain rate and survive a single pod loss without the backlog stalling. The relay should expose `/healthz` that reports *unhealthy* if it has not successfully published a batch within some window while pending rows exist — a relay that is silently stuck is worse than one that is crash-looping, because the crash-loop is at least visible. Note `KAFKA_ACKS=all`: the relay must wait for the broker to durably replicate each message before marking the row published, or a broker failover can swallow an "acked" message and re-create the lost-event bug.

The observability you need is short and specific. Emit these metrics and alert on them:

```python
# Prometheus-style gauges/counters the relay should export.
outbox_pending_rows         = Gauge("outbox_pending_rows", "rows with published_at IS NULL")
outbox_oldest_pending_secs  = Gauge("outbox_oldest_pending_secs", "age of oldest unpublished row")
outbox_published_total      = Counter("outbox_published_total", "rows marked published")
outbox_publish_errors_total = Counter("outbox_publish_errors_total", "broker publish failures")
outbox_batch_drain_seconds  = Histogram("outbox_batch_drain_seconds", "time to publish a batch")

# In the poll loop, after each pass:
pending = conn.execute("SELECT count(*), "
                       "COALESCE(EXTRACT(EPOCH FROM now() - min(created_at)), 0) "
                       "FROM outbox WHERE published_at IS NULL").fetchone()
outbox_pending_rows.set(pending[0])
outbox_oldest_pending_secs.set(pending[1])
```

The single most important alert is on **`outbox_oldest_pending_secs`** — the age of the oldest unpublished row. A healthy relay keeps this in the low single-digit seconds (one poll interval plus publish time). If it climbs to 30 seconds, the relay is falling behind or down, and every downstream service is now that far behind reality. Alert at, say, 30 seconds warning and 120 seconds critical. The row *count* is a useful secondary signal (it tells you the backlog size and thus the drain time you are facing), but age is the leading indicator of a consistency-lag incident, and it is the one that maps directly to "how stale is the rest of the company's view of the world right now." This metric is also the cleanest input to an [SLO on event-publishing freshness](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration), should you choose to make one.

## Schema and what goes in the payload

The outbox `payload` column is your wire contract with every consumer, and the discipline around it is the difference between a system you can evolve and one that freezes the day you ship it. Two rules carry most of the weight. First, **serialize the payload once, in the producing transaction, and never recompute it** — the row in the outbox is the immutable historical fact, and if you later change how you build the payload, old rows still carry the old shape, which is correct (they happened in the past). Second, **version the schema explicitly and evolve it only in backward-compatible ways** so that a consumer written against `v1` does not break when the producer starts emitting `v2`.

```python
def build_order_placed_v2(order_id, customer_id, items, total_cents) -> dict:
    return {
        "schema": "OrderPlaced",
        "schema_version": 2,            # bump only for breaking changes
        "event_id": str(uuid.uuid4()),
        "order_id": order_id,
        "customer_id": customer_id,
        "currency": "USD",              # added in v2; v1 consumers ignore unknown fields
        "items": items,
        "total_cents": total_cents,
        "occurred_at": now_iso(),
    }
```

The practical compatibility rules are the same ones any event schema lives by: you may *add* optional fields freely (old consumers ignore what they do not read), you may *never* remove or rename a field a consumer might depend on, and a genuinely breaking change means emitting a new event type or a new version *alongside* the old until every consumer has migrated. This is consumer-driven contract territory, and the discipline of proving you have not broken a consumer before you deploy is exactly [API versioning and consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing). The relevant decision here — fat state-transfer payload versus thin notification — feeds directly into how much schema surface you have to maintain: a fat payload carries more fields and thus more contract to keep stable, but it buys consumers true decoupling because they never have to call back to you. The reason I still lean fat is that the alternative re-introduces the synchronous coupling and availability multiplication that pushed you toward events in the first place, a tension drawn out across the whole of [database-per-service: the rule that defines microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), which is where the "no shared database, talk through events" rule originates and where the dual-write problem becomes unavoidable.

## Cleanup and retention: the table that eats your disk

There is a failure mode that the happy-path tutorials never mention and that *will* page you in month two: the outbox table grows without bound. Every order writes a row, the relay marks it published, but nothing deletes it. At 200 events/sec that is `200 × 86400 ≈ 17 million` rows per day. In a month the outbox has half a billion rows, the partial index on `published_at IS NULL` is fine (it only indexes the tiny unpublished set) but the table itself is enormous, vacuum struggles, backups balloon, and eventually you are paged for disk pressure on your primary. The outbox is a *queue*, not a *log* — published rows are garbage and must be reaped.

A retention job that deletes published rows older than a safety window, run in small batches to avoid long-held locks and replication lag:

```sql
-- Run every few minutes (cron, pg_cron, or an app scheduler).
-- Delete in bounded batches so we never hold a long lock or generate a
-- huge WAL spike that lags replicas / CDC.
DELETE FROM outbox
WHERE id IN (
    SELECT id FROM outbox
    WHERE published_at IS NOT NULL
      AND published_at < now() - INTERVAL '1 hour'   -- safety window
    ORDER BY id
    LIMIT 10000
);
```

```bash
#!/usr/bin/env bash
# Loop the bounded delete until a pass clears nothing, then exit.
# Keeps each transaction small; safe to run on a busy primary.
set -euo pipefail
while :; do
  deleted=$(psql -tAc "WITH d AS (
      DELETE FROM outbox
      WHERE id IN (
        SELECT id FROM outbox
        WHERE published_at IS NOT NULL
          AND published_at < now() - INTERVAL '1 hour'
        ORDER BY id LIMIT 10000)
      RETURNING 1)
    SELECT count(*) FROM d;")
  echo "deleted $deleted rows"
  [ "$deleted" -eq 0 ] && break
  sleep 0.5   # breathe between batches so we don't starve order writes
done
```

The **safety window** (`1 hour` here) matters: do not delete a row the instant it is marked published, because if you are using CDC, Debezium might not have read past it yet, and deleting it could create a tombstone or a gap depending on your config. Keep published rows around long enough that every relay and every replica has certainly moved past them — an hour is generous; even a few minutes works if you understand your slot lag. The **batched delete** matters because a single `DELETE` of 17 million rows takes a table-level lock long enough to stall order writes, generates a giant WAL burst that lags your replicas and CDC consumers (ironically causing the slot-bloat problem you were trying to avoid), and can blow out your transaction log. Always delete the outbox in small bounded batches with a breath between them.

There is a clever alternative worth knowing: **table partitioning by time**. Make `outbox` a range-partitioned table by `created_at` (a partition per hour or per day), and retention becomes `DROP TABLE outbox_2026_06_14` — an instant metadata operation that reclaims the whole partition's disk at once, with no row-by-row delete, no lock, no WAL flood. For high-volume outboxes this is the right answer and it is how you run an outbox at serious scale. The trade-off is the operational overhead of managing partitions (creating tomorrow's partition ahead of time, usually with `pg_partman`), which is why I would start with the batched delete and graduate to partitioning when the delete job itself becomes a bottleneck.

![A stack showing the end to end reliability chain of an atomic write, a durable relay that marks rows sent, a durable replicated broker, an idempotent consumer with inbox dedup, and a retention job that prunes sent rows](/imgs/blogs/the-transactional-outbox-and-reliable-event-publishing-9.webp)

Step back and see the whole chain, because reliability here is exactly that — a chain, only as strong as its weakest link. The atomic write removes lost-or-phantom events. The durable relay (marking rows sent only after the broker acks) removes lost-in-flight events while accepting duplicates. The durable, replicated broker removes events-lost-in-the-broker (a single-replica topic can lose your event after the relay thinks it published — set `acks=all` and `min.insync.replicas` accordingly, the broker-side story in [the anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers)). The idempotent consumer with inbox dedup removes the duplicate problem the relay introduced. And the retention job removes the disk-fill that would otherwise take down the whole apparatus weeks later. Drop any one link and you have a system that works in the demo and corrupts itself in production.

## Stress-testing the design: what breaks under fire

A pattern is only as good as its behavior on the bad days. Let us deliberately break the outbox setup five ways and trace what happens, because this is how you build the confidence to run it.

**The relay crashes.** Already covered, and the answer is the whole reason we chose this pattern: nothing is lost. The order service keeps committing orders and outbox rows because its request path does not touch the relay or the broker. The outbox grows (worked example above). When the relay recovers it drains the backlog at multiples of the arrival rate. The only visible effect is a temporary consistency lag — downstream services are behind reality for as long as the relay was down plus the drain time — and the only thing that turns this from a blip into an incident is an undersized relay that drains too slowly or a missing alert on outbox lag. Action: alert on `count(*) WHERE published_at IS NULL` and on `age(oldest unpublished row)`; size drain rate to several multiples of peak arrival.

**The broker is down.** The relay's `producer.flush()` does not get acks, so it does not mark rows published, so it retries the same batch when the broker recovers. No events are lost; the backlog grows just as in the relay-crash case and drains on recovery. The danger here is a relay that publishes optimistically and marks rows published *before* the broker acks — then a broker outage swallows the in-flight batch and those rows are marked sent but never arrived: lost events, the exact bug we set out to kill, re-introduced by sloppy relay code. The discipline is absolute: **mark a row published only after the broker has acked it durably.** With CDC, Debezium handles this for you (it commits its WAL offset only after Kafka acks), which is one of CDC's quiet advantages — there is no place for you to get the ordering wrong.

**A duplicate publish.** Covered in the duplicate worked example. The relay crashes after the broker acks but before the `UPDATE` commits, so the batch is re-published. The consumer's inbox dedup makes the redelivery a no-op. The failure is benign *if and only if* every consumer is idempotent. The catastrophic version is a non-idempotent consumer (double-charges, double-reserves), which is why "all consumers must dedup" is a hard rule, not a nice-to-have. Action: make the inbox dedup a required part of every consumer's template; treat a consumer without dedup as a bug.

**The outbox grows unbounded.** Covered in retention. The relay is publishing fine, but nobody is deleting published rows, so the table grows by ~17M rows/day, vacuum falls behind, and eventually disk pressure on the primary stalls *all* writes — including new orders, which is now a full outage caused by a hygiene oversight. Action: run the batched retention job (or time-partition the table) from day one, and monitor outbox table size as well as lag. The retention job failing silently is itself a failure mode, so alert if the published-row count stops shrinking.

**Events arrive out of order.** Suppose order `A-8842` is placed and then cancelled within the same second. Two outbox rows: `OrderPlaced` (id 1001) then `OrderCancelled` (id 1009). If both are keyed by `A-8842`, they land in the same Kafka partition in id order and a single consumer applies place-then-cancel correctly. But if you forgot the key — say the relay published with a null key or keyed by a random id — Kafka spreads them across partitions, two consumer instances pick them up concurrently, and `OrderCancelled` can be applied *before* `OrderPlaced`. Now inventory tries to release a reservation that was never made (a no-op or an error), and then applies the reservation from the later-processed `OrderPlaced` — leaving stock reserved for a cancelled order forever. This is the heisenbug from the ordering section made concrete, and it is insidious because it only manifests when two events for the same aggregate happen close enough together to race, which is rare enough to survive testing and common enough to corrupt production. The fix is the discipline already stated: **always key by `aggregate_id`.** The defensive backstop is to make consumers tolerant of out-of-order arrival where you can — version your aggregate state and ignore an event whose version is older than what you have already applied — so that even a keying mistake degrades to a dropped stale update rather than corruption. Action: assert in tests that every published event carries a non-null aggregate key; consider a state-version check in consumers for the events where ordering is safety-critical.

The meta-lesson across all five: the outbox converts a class of *invisible, unrecoverable* failures (lost events) into a class of *visible, recoverable* ones (backlog lag, duplicates, disk growth, out-of-order races you can guard against) — provided you have the alerts to see them and the consumer-side dedup and keying discipline to absorb them. That conversion is the entire value proposition. You are not making failures impossible; you are making them survivable and observable. And that is the deepest reason to prefer this pattern over the clever-sounding alternatives: it does not promise that nothing will ever go wrong, which is a promise no distributed system can keep. It promises that when something goes wrong, you will see it, it will not have silently destroyed data, and it will heal once the failed component comes back — which is the most a distributed system can honestly offer.

## Case studies

Three real-world stories, each teaching one lesson the pattern exists to deliver.

**Debezium and the rise of CDC-based outboxes.** Debezium, the open-source CDC platform built by Red Hat on top of Kafka Connect, is the reason "outbox via change data capture" went from an exotic technique to a standard one. Its maintainers wrote up the outbox pattern explicitly — the `EventRouter` transform shown above is a first-class Debezium feature built for exactly this use case — and the approach is documented at length on the Debezium blog and in Gunnar Morling's talks. The concrete lesson: the outbox table as a CDC source decouples your *internal* schema (the raw outbox row) from your *external* contract (the clean domain event on Kafka), so you can refactor your database without breaking consumers, and you publish events at near-WAL latency without polling. The accompanying caution, repeated in their own operational docs, is the replication-slot disk-fill — Debezium's own guidance is to monitor slot lag religiously, because a stalled connector is one of the few ways a CDC outbox can hurt your primary.

**The lost-event incident from a naive dual-write.** This pattern is famous precisely because so many teams independently rediscover the dual-write bug the hard way, and the public engineering literature is full of post-mortems with the same shape: an order/payment/booking service that committed its database row and then published, lost events during deploys and node failures, and spent weeks tracking down "orders that exist but no downstream system knows about." Chris Richardson's *Microservices Patterns* opens its treatment of the outbox with exactly this failure, and his microservices.io catalog documents the dual-write problem as the canonical motivation. The lesson is uncomfortable and worth internalizing: this is not an exotic edge case you can defer — it is the *default* behavior of the most obvious code you would write, the bug is silent, and the only way to not have it is to know the pattern before you ship. Teams that learn the outbox after the incident always say the same thing: "I wish someone had told me a database commit and a broker publish were two writes."

**Outbox at scale and event-carried state transfer.** Companies running large event-driven platforms — the pattern is widely discussed in the engineering writing from teams at Shopify, Wix, and others operating high-volume order and commerce systems — converge on a recognizable set of practices at scale: time-partition the outbox so retention is a `DROP PARTITION` rather than a delete storm; key events by aggregate id for per-entity ordering; carry enough state in the payload that consumers do not have to call back (event-carried state transfer), trading bigger messages for true decoupling; and run the relay as a separate horizontally-scaled deployment with `SKIP LOCKED` so it scales independently of the request path. The lesson: the outbox is not just a correctness fix for small services — it is the reliable-publishing backbone of large event-driven systems, and the scaling concerns (retention, ordering, payload size, relay throughput) are exactly the ones this post has walked through, because they are the ones that bite when the volume is real.

## When to reach for this (and when not to)

Reach for the outbox **the moment a service must both change its own state and tell other services about it** — which is to say, the moment a service in an event-driven system writes to its database and publishes an event in the same logical operation. That is an enormous fraction of all services in a microservices architecture, so in practice the outbox is close to mandatory infrastructure for any service that emits domain events off the back of state changes. If your service publishes events, and those events matter (someone reserves stock, charges a card, ships a box based on them), you need the outbox or CDC. There is no "we'll add reliability later" — the naive version is losing events from day one, silently.

You do **not** need the outbox when there is no dual write to begin with. A service that only *reads* and publishes derived analytics where an occasional lost event is acceptable does not need it. A service whose "event" is purely a fire-and-forget metric or log line, where loss is fine, does not need it. And a true event-sourced service, where the event log *is* the source of truth and the database is a derived projection, does not need a separate outbox because there is no second write to reconcile — the event store and the state are one and the same, which is the listen-to-yourself world. Likewise, if your two writes happen to be to the *same* system that does support a real shared transaction — for instance, both your state and your "events" live in the same database and your "broker" is also that database (a queue table consumed by an in-database worker) — then you already have atomicity and the outbox is implicit. The pattern is specifically the answer to "two different storage systems, no shared transaction"; remove that condition and you do not need it.

Between polling and CDC, the decision rule one more time, decisively: **default to outbox-plus-polling.** It is simpler, fully owned, debuggable with SQL, and sub-second latency is invisible for almost every business event. Graduate to CDC only when you have a measured latency requirement polling cannot meet (sub-100ms publish), or when poll load on the primary becomes material at very high volume, and when you do, staff the operational reality — a Kafka Connect cluster and replication-slot monitoring — before you turn it on. Do not adopt CDC because it sounds more sophisticated; adopt it because polling stopped being good enough and you can prove it.

## Key takeaways

- **A database commit and a broker publish are two writes to two systems with no shared transaction.** A crash between them loses the event or publishes a phantom. There is no naive ordering that escapes this; it is structural.
- **The outbox dissolves the problem by making the event part of the local transaction.** Write the event to an `outbox` table in the same transaction as the state change; they commit or roll back together with one fsync. You reuse your database's atomicity instead of inventing a new mechanism.
- **The request handler must never touch the broker.** A `producer.send()` next to a `db.commit()` is the dual-write bug. The handler writes only to its own database; a separate relay does the publishing.
- **You get at-least-once, not exactly-once — and that is correct.** The relay's only safe crash behavior is to re-publish, so duplicates are unavoidable. You trade an unsolvable problem (lost events) for a solved one (duplicates).
- **Therefore every consumer must be idempotent.** Dedup by event id, ideally with an inbox table that records the id in the same transaction as the effect. The outbox and the inbox are a matched pair; shipping one without the other is half a solution.
- **Polling is the honest default; CDC is the measured upgrade.** Polling is simple, owned, and sub-second. CDC cuts latency ~10× and removes poll load but adds a connector, replication slots, and a disk-fill failure mode. Switch only when you can prove polling is insufficient.
- **Key events by aggregate id for ordering.** Kafka orders within a partition only; keying by the aggregate keeps one entity's events in one partition and in order. Forgetting this gives you a heisenbug where events apply out of order.
- **Retention is not optional.** The outbox is a queue, not a log. Delete published rows in bounded batches (or time-partition and drop partitions). An unbounded outbox fills your primary's disk and takes the whole service down weeks later.
- **The outbox converts invisible, unrecoverable failures into visible, recoverable ones** — backlog lag, duplicates, disk growth — but only if you have the alerts (outbox lag, oldest-unpublished-row age, table size) and the consumer-side dedup to absorb them.

## Further reading

- Chris Richardson, *Microservices Patterns* (Manning) — the canonical treatment of the transactional outbox, the dual-write problem, and the polling-publisher vs transaction-log-tailing distinction, with the same running-example spine.
- Sam Newman, *Building Microservices, 2nd ed.* (O'Reilly) — on inter-service communication, eventual consistency, and why reliable event publishing is foundational to an event-driven architecture.
- The Debezium documentation and engineering blog, especially the outbox event router and the operational guidance on replication slots and connector monitoring.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the database-internals mechanism: logical decoding, replication slots, and how the WAL becomes an event stream.
- [The transactional outbox pattern for reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — the broker-side view of the outbox as a publishing mechanism.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) and [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the delivery-guarantee theory the outbox depends on.
- Sibling posts: [event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration), [database-per-service: the rule that defines microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), and the forward-looking [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services).

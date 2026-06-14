---
title: "Change Data Capture and the Transactional Outbox Pattern"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Why writing to your database and a message broker in two steps silently corrupts your system, and how the transactional outbox and log-based change data capture make every derived system converge on the truth."
tags:
  [
    "change-data-capture",
    "cdc",
    "outbox-pattern",
    "debezium",
    "kafka",
    "event-driven",
    "dual-write",
    "microservices",
    "distributed-systems",
    "database",
    "postgresql",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/change-data-capture-and-the-outbox-pattern-1.webp"
---

Almost every distributed-systems outage I have been paged for at 3 a.m. reduces to the same root cause, and it is never the one on the runbook. It is not the network partition, the slow query, or the GC pause. It is that two systems that were supposed to agree on a fact no longer do. The orders table says the payment succeeded; the search index says the order is still pending. The database says the user deactivated their account; the cache still serves their profile. The warehouse report says revenue was $4.2M; finance's ledger says $4.18M. Nobody wrote a bug. Every individual write succeeded. And yet the system as a whole is lying, because somewhere a process wrote to one store, intended to write to a second, and died — or timed out, or got rate-limited, or hit a deploy — in the gap between the two.

This is the **dual-write problem**, and it is the quiet tax on every architecture that has more than one place data lives. The moment you add a cache, a search index, a read replica in another service, an analytics warehouse, or a Kafka topic that other teams consume, you have committed to keeping two or more systems in sync. The naive way — write to the database, then write to the other thing — is wrong in a way that no amount of careful coding fixes, because the two writes are not in the same transaction and there is no atomic boundary spanning them. On any crash between them, they diverge, and nothing detects it.

The industry has converged on two interlocking answers: the **transactional outbox** pattern, which makes "publish an event" part of the same database transaction as the business change so the two can never disagree; and **change data capture** (CDC), which reads the database's own replication log — the write-ahead log in Postgres, the binlog in MySQL — and turns every committed change into an event stream that any number of downstream systems can subscribe to. Put them together with a tool like [Debezium](https://debezium.io/) and you get a pipeline where a single committed row in your database reliably, in order, with no lost events, becomes an update to your cache, your search index, your warehouse, and three other microservices — and stays correct through crashes, restarts, and replays. The diagram above is the mental model for this entire article: a dual write is two arrows that can break independently; the outbox collapses them into one atomic commit that a relay later turns into a durable, replayable stream.

## The dual write problem: the bug you cannot code your way out of

![A dual write splits one logical change across two systems with no shared transaction, so any crash leaves them diverged; the outbox makes the change and its event commit atomically](/imgs/blogs/change-data-capture-and-the-outbox-pattern-1.webp)

Read the figure left to right. On the left is the broken pattern almost everyone writes first. A request comes in to mark an order paid. The handler does two things: it commits a row to Postgres, and then it publishes an event to Kafka (or invalidates a Redis key, or POSTs to a search-indexing service — the shape is identical for all of them). The Postgres commit succeeds. Then the process crashes — a deploy rolls the pod, the JVM hits an OOM, the broker connection times out — before the publish lands. The database now says "paid." Kafka has no record of it. Every consumer of that Kafka topic — the email service, the search index, the fraud pipeline — is now permanently behind, and there is no error anywhere to alert on, because from each component's local point of view nothing failed.

The reason this is unfixable by ordinary means is worth stating precisely, because engineers reach for fixes that do not work. The two writes target two different systems — Postgres and Kafka — and **there is no transaction that spans both.** A database transaction gives you atomicity *within* the database. A Kafka transaction gives you atomicity *within* Kafka. Nothing gives you atomicity across the boundary between them, because that would require a distributed transaction — a two-phase commit (2PC) across a SQL database and a message broker — and 2PC across heterogeneous systems is operationally toxic: it needs an XA transaction manager, it holds locks across a network round trip, it blocks indefinitely when the coordinator dies mid-protocol, and most modern brokers (Kafka included) do not even offer a usable XA participant interface. This is the same fundamental constraint we hit with [synchronous versus asynchronous replication](/blog/software-development/database/database-replication-sync-async-logical-physical): you cannot get free atomicity across an asynchronous boundary; you can only choose where the inconsistency window lives and how you detect and repair it.

Let me be concrete about the four ways a dual write can fail, because people often think they have handled "the crash" when they have handled only one of these.

| Failure | What happens | Why retries do not save you |
| --- | --- | --- |
| Crash after DB commit, before publish | DB has the change, broker never gets the event | The in-memory intent to publish died with the process; nothing remembers to retry |
| Publish succeeds, DB transaction rolls back | Broker has an event for a change that never happened | Consumers act on a phantom change; the "fix" of publishing first makes this worse |
| Publish times out, then succeeds late | You retried, so the broker has the event twice | Now you have a duplicate, not a loss — a different bug, addressed later by idempotency |
| Two concurrent requests, interleaved | Events published out of order relative to DB commit order | Consumers apply a stale value over a fresh one; ordering is now wrong |

Notice the trap in row two. A common instinct is "publish to Kafka *first*, then commit to the DB, so we never lose an event." This swaps a lost-event bug for a phantom-event bug: the publish succeeds, then the DB commit fails (constraint violation, deadlock, disk full), and now every consumer has acted on a state change that the source of truth never recorded. There is no ordering of two independent writes that is safe. The only safe constructions are (a) make the two writes one atomic write, or (b) derive the second write from a log of the first. The outbox is (a); CDC is (b); the combination is both.

> The dual write is not a bug you fix with better error handling. It is a structural property of writing to two systems without a shared transaction. You either make it one write, or you derive one from the other.

## The transactional outbox: one commit, then a relay

![Writing the event into an outbox row inside the same transaction makes the business change and its event atomically durable together, and a relay publishes it afterward](/imgs/blogs/change-data-capture-and-the-outbox-pattern-2.webp)

The transactional outbox pattern — catalogued by Chris Richardson on [microservices.io](https://microservices.io/patterns/data/transactional-outbox.html) as one of the core data patterns for microservices — is almost embarrassingly simple once you see it, and that simplicity is the point. Instead of writing the business change and then publishing an event as two separate operations, you write the business change **and an event row into an outbox table in the same database transaction.** Because both writes are in one transaction, the database's own atomicity guarantee covers them: either both the order update and the outbox row commit, or neither does. There is no crash window in which one exists without the other. You have replaced a cross-system atomicity problem you cannot solve with a single-system atomicity problem the database already solves for you.

A separate process — the **message relay** (Richardson's term), sometimes called the publisher or the dispatcher — reads rows from the outbox table and publishes them to the broker. Crucially, this read-and-publish step happens *after* the transaction has committed, so if the relay crashes, the outbox rows are still sitting in the durable table waiting to be published when it restarts. The relay can retry forever; the events are safe in the database. This is what makes the outbox an **at-least-once** delivery mechanism: an event in the outbox will be published one or more times (more if the relay crashes after publishing but before recording that it published), but it will never be lost.

Here is the outbox table. I am using Postgres; the shape is identical in MySQL.

```sql
CREATE TABLE outbox (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type  TEXT        NOT NULL,   -- e.g. 'Order', 'Customer' — becomes the topic
    aggregate_id    TEXT        NOT NULL,   -- e.g. the order id — becomes the partition key
    event_type      TEXT        NOT NULL,   -- e.g. 'OrderPaid', 'OrderCancelled'
    payload         JSONB       NOT NULL,   -- the event body the consumer reads
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at    TIMESTAMPTZ            -- NULL until the relay has published it
);

-- The relay only ever scans unpublished rows; keep that scan cheap.
CREATE INDEX outbox_unpublished_idx
    ON outbox (created_at)
    WHERE published_at IS NULL;
```

And here is the write — the entire trick is that the `UPDATE` and the `INSERT` are one transaction:

```sql
BEGIN;

UPDATE orders
   SET status = 'paid', paid_at = now()
 WHERE id = '7f3c…';

INSERT INTO outbox (aggregate_type, aggregate_id, event_type, payload)
VALUES (
    'Order',
    '7f3c…',
    'OrderPaid',
    jsonb_build_object(
        'order_id', '7f3c…',
        'amount_cents', 4200,
        'currency', 'usd',
        'paid_at', now()
    )
);

COMMIT;
```

In application code — say a Spring or a Python service — the only discipline you must enforce is that the outbox insert uses the *same* transaction/connection as the business write. In SQLAlchemy that means both statements run inside one `session.begin()` block; in Spring it means both happen inside one `@Transactional` method using the same `EntityManager`. If you accidentally open a second connection for the outbox insert, you have reintroduced the dual write — now between two transactions on the same database, which at least both live in one system but no longer share atomicity. This is the single most common way teams subtly break the outbox: the table is there, but the insert is not actually in the business transaction.

### The relay: polling publisher vs transaction-log tailing

There are two ways to build the relay, and they correspond to two of Richardson's named patterns. The first is the **polling publisher**: a loop that periodically queries the outbox for unpublished rows, publishes them, and marks them published.

```sql
-- One relay tick. FOR UPDATE SKIP LOCKED lets N relay workers run concurrently
-- without two of them grabbing the same row.
BEGIN;

SELECT id, aggregate_type, aggregate_id, event_type, payload
  FROM outbox
 WHERE published_at IS NULL
 ORDER BY created_at
 LIMIT 100
   FOR UPDATE SKIP LOCKED;

-- … application publishes each row to Kafka here …

UPDATE outbox
   SET published_at = now()
 WHERE id = ANY(:published_ids);

COMMIT;
```

The polling publisher is trivially correct and needs no special database configuration — just a table and a `SELECT`. Its costs are the obvious ones: polling latency (you publish at most once per poll interval, so a 1-second interval means up to ~1 second of added lag), and constant query load on the database even when nothing is happening. The `FOR UPDATE SKIP LOCKED` is the part people miss: it lets you run multiple relay workers safely, each grabbing a disjoint batch, which you will need once your event volume exceeds what one worker can publish. Mark published rows and let a periodic job delete old ones, or have the relay delete instead of update if you never need the outbox as an audit trail.

The second way is **transaction-log tailing**, and it is where this story merges with change data capture. Instead of polling the table, you point a CDC tool at the database's replication log and let it stream every insert into the outbox table out to Kafka as it commits. No polling, no query load, sub-second latency. This is exactly what Debezium's **Outbox Event Router** does, and we will build it in full later. The tradeoff is operational complexity: you now run a CDC connector, manage a replication slot, and monitor a Kafka Connect cluster. For low-to-moderate volume, the polling publisher is the right default; for high volume or when you already run CDC, log tailing wins.

| Relay approach | Latency | DB load | Ordering | Operational cost | Use when |
| --- | --- | --- | --- | --- | --- |
| Polling publisher | poll interval (e.g. 500 ms–2 s) | constant `SELECT` load | by `created_at`, single worker | low — just a table | low/moderate volume, no CDC infra |
| Log tailing (Debezium) | sub-second | near zero (reads the log) | by commit order in the log | high — connector + slot | high volume, or CDC already deployed |

### Second-order gotcha: outbox bloat and the relay's own dual write

Two non-obvious problems bite teams in production. First, **outbox bloat**: the outbox table is write-heavy and, if you mark-as-published rather than delete, it grows without bound, and the partial index degrades as dead tuples accumulate. The fix is a retention job (`DELETE FROM outbox WHERE published_at < now() - interval '7 days'`) plus aggressive autovacuum tuning on that table, or simply deleting rows once published if you do not need them for audit. With log-tailing CDC you can skip the `published_at` column entirely — Debezium tracks its own log offset, so the outbox row's job is done the instant it is committed and captured, and you can delete it immediately (Debezium even handles the delete event gracefully).

Second, and subtler: **the polling relay has its own at-least-once boundary.** If the relay publishes a row to Kafka and then crashes before the `UPDATE … SET published_at` commits, it will publish that row again on restart. That is fine — at-least-once is exactly the contract — but it means downstream consumers *must* be idempotent. There is no relay design that gives you exactly-once publishing without cooperation from the consumer, which is the central truth we return to in the idempotency section.

## Change data capture: read the log, not the table

![Log-based CDC decodes committed WAL records from a replication slot and streams every row change into Kafka topics, adding no query load to the source](/imgs/blogs/change-data-capture-and-the-outbox-pattern-3.webp)

Change data capture is the general technique of observing every change made to a database and making those changes available as a stream of events. The outbox is one way to *produce* a clean event stream; CDC is the broader mechanism that can capture *any* table's changes — including the outbox table, which is how the two patterns combine. There are three fundamentally different ways to implement CDC, and choosing among them is one of the higher-leverage decisions in a data platform.

The figure shows the one that has won for most use cases: **log-based CDC.** Every durable database already maintains a log of changes for its own crash recovery and replication — the **write-ahead log** (WAL) in Postgres, the binary log (binlog) in MySQL, the redo log in Oracle. We covered why this log exists and what it guarantees in [the write-ahead log article](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability): every committed change is appended to the log before the data pages are updated, in commit order, durably. Log-based CDC piggybacks on that log. A CDC tool connects as if it were a replica, reads the log, decodes each change into a structured event (table, operation, before-image, after-image), and streams it out. The source database does not run a single extra query for this — the reader is just tailing a file the database was already writing.

In Postgres, the mechanism is **logical decoding** over a **replication slot**. Logical decoding, introduced in Postgres 9.4 and made far more usable by the built-in `pgoutput` plugin in Postgres 10, takes the physical WAL records — which are page-level and meaningless outside the server — and decodes them into logical row-level changes. A replication slot is a server-side bookmark that remembers how far a particular consumer has read, and — this is the critical operational fact — **the slot prevents Postgres from recycling WAL segments the consumer has not yet read.** That is what makes it lossless: if your CDC connector is down for an hour, the WAL it needs is retained, and it resumes exactly where it left off. It is also the number-one way to take down a Postgres instance with CDC: if the consumer is down (or stuck) long enough, retained WAL fills the disk and the database stops accepting writes. We will return to this failure mode in the case studies, because it has caused real outages.

```sql
-- Postgres must be configured for logical decoding (postgresql.conf):
--   wal_level = logical
--   max_replication_slots = 10
--   max_wal_senders = 10

-- A publication declares which tables to capture.
CREATE PUBLICATION cdc_pub FOR TABLE orders, customers, outbox;

-- Debezium (or any consumer) creates a logical slot using the pgoutput plugin.
-- This is what Debezium does under the hood; you can inspect it:
SELECT slot_name, plugin, active, restart_lsn,
       pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained_wal
  FROM pg_replication_slots;
--  slot_name   |  plugin  | active | restart_lsn | retained_wal
-- -------------+----------+--------+-------------+--------------
--  debezium    | pgoutput | t      | 3F/A2C100   | 24 MB
```

That `retained_wal` column is the one to alert on. If it grows monotonically, your consumer is not keeping up and your disk is the clock you are racing.

One more Postgres-specific detail with teeth: **REPLICA IDENTITY.** By default, an `UPDATE` or `DELETE` in the WAL only records the primary key of the changed row, not the old values of the other columns. If you want the *before-image* (the previous state of a row, which many CDC consumers need to compute diffs or handle deletes), you must set `ALTER TABLE orders REPLICA IDENTITY FULL`, which makes Postgres log the entire old row. That costs WAL volume, so it is a deliberate per-table choice, not a default.

### Query-based and trigger-based CDC, and why they lose

![Query, trigger, and log-based capture trade off completeness, latency, and database overhead in distinct, predictable ways](/imgs/blogs/change-data-capture-and-the-outbox-pattern-7.webp)

Before log-based CDC was easy, two other approaches were common, and they still show up — sometimes appropriately. The matrix above lays out the tradeoffs; let me walk the two alternatives.

**Query-based CDC** (polling) works by adding an `updated_at` timestamp column to every table and periodically running `SELECT * FROM orders WHERE updated_at > :last_poll`. It is dead simple, needs no special database privileges, and works against any database. It has two fatal weaknesses for many use cases. First, **it cannot capture deletes** — a deleted row simply stops appearing in query results, with no event marking its departure (a workaround is soft deletes with a `deleted_at` flag, which then leaks into all your queries). Second, **it misses intermediate states**: if a row changes three times between two polls, you see only the final state, not the three transitions. For an audit log or an event-sourced system, that is data loss. It also puts repeated scan load on the source and its latency is bounded below by the poll interval.

**Trigger-based CDC** installs database triggers that fire on every insert/update/delete and write a row into an audit/shadow table, which a relay then reads. This captures everything, including deletes and intermediate states, and works on databases without accessible logs. Its cost is **write amplification and latency on the hot path**: every business write now also does a trigger-driven write, synchronously, inside the same transaction, roughly doubling write cost and adding contention. Triggers are also brittle — they couple your capture logic to your schema, must be maintained as the schema evolves, and are notoriously hard to debug. Trigger-based CDC is still the right answer when you cannot get log access (a managed database that does not expose logical replication, for instance), but it is a fallback, not a default.

Log-based CDC dominates on the axes most teams care about: it captures every committed change including deletes and intermediate states, in commit order, with sub-second latency, and with near-zero load on the source because it reads the log the database already writes. Its only real cost is operational setup — the replication slot, the connector, and the discipline to monitor WAL retention.

## The log as the source of truth

![Replacing pairwise integrations with one shared log collapses O(N squared) brittle pipelines into O(N) independent subscribers](/imgs/blogs/change-data-capture-and-the-outbox-pattern-8.webp)

In 2013, Jay Kreps — then at LinkedIn, soon to co-found Confluent — wrote [*The Log: What every software engineer should know about real-time data*](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying), and it reframed how a generation of engineers think about data integration. His central claim is that **a log — an append-only, totally-ordered sequence of records — is the simplest possible storage abstraction and the unifying primitive behind databases, replication, consensus, and stream processing.** A database's replication log, a Kafka topic, a WAL, a Paxos/Raft log: all the same shape. Once you see that, CDC stops looking like a niche integration trick and starts looking like *exposing the log the database already keeps, so the rest of your systems can be derived from it.*

The before/after figure makes the architectural payoff concrete. Without a shared log, every pair of systems that must stay in sync needs its own bespoke pipeline: DB-to-cache here, DB-to-search there, DB-to-warehouse as a nightly batch, and a new one every time someone adds a system. That is O(N²) pipelines for N systems, each independently written, monitored, and broken. Kreps's argument — and the architecture LinkedIn built around Kafka and its predecessor **Databus** — is to insert one log in the middle. The source database's changes flow into the log once (via CDC), and every downstream system becomes a **subscriber** that reads the log at its own pace from its own offset. N systems, N subscribers, one capture pipeline. Adding a new derived system is now "write a consumer," not "build and operate a new ETL pipeline."

LinkedIn's [Databus](https://dl.acm.org/doi/abs/10.1145/2391229.2391247) (described in the SoCC '12 paper *All Aboard the Databus!*) is the canonical early implementation of this idea: a *source-agnostic* CDC platform with three parts — a **relay** that streams committed changes from the source at low-millisecond latency through an in-memory circular buffer, a **bootstrap service** that gives new or far-behind consumers a consistent snapshot and serves laggards from a long-term store so one slow consumer cannot stall the live stream, and a **consumer client** that subscribes. It guaranteed *timeline consistency* (changes in commit order) at thousands of events per second per server. Airbnb's open-source [SpinalTap](https://medium.com/airbnb-engineering/capturing-data-evolution-in-a-service-oriented-architecture-72f7c643ee6f) is the same pattern a generation later, built to let a service-oriented architecture share data without sharing databases: it tails the MySQL binlog into Kafka with **at-least-once delivery, zero permanent data loss, and per-record ordering**, and powers exactly the fan-out we are about to discuss — cache invalidation, Elasticsearch indexing, offline export, and cross-service signaling. Every one of these systems — Databus, SpinalTap, Debezium — is the same shape: tail the log, preserve per-key order, fan out to subscribers.

This is also the bridge between Kleppmann's two big chapters in *Designing Data-Intensive Applications*. Chapter 11 (Stream Processing) makes the case that change data capture is the principled way to keep [derived data systems](/blog/software-development/database/database-partitioning-and-sharding) in sync: treat the database's change log as the authoritative event stream, and rebuild every cache, index, and aggregate as a deterministic function of that stream. Chapter 12 (The Future of Data Systems) pushes it further into the idea of *unbundling the database* — your system of record holds the truth, the log broadcasts every change, and your indexes, caches, and materialized views are all just different downstream consumers maintaining different views of the same log. The log is the source of truth for *change*; the database is the source of truth for *current state*; and CDC is the wire between them.

The deep property that makes this work is **determinism through ordering.** Because the log is totally ordered (at least per key), any consumer that processes it from offset 0 will arrive at exactly the same derived state as any other consumer that processes the same prefix — and a consumer that falls behind, crashes, or is added years later can replay from the beginning and catch up to a consistent state. You cannot do that with a pile of point-to-point pipelines. This is why "rebuild the search index from scratch" is a routine, boring operation in a log-centric architecture (reset the consumer offset to 0 and let it run) and a terrifying multi-day project in a point-to-point one.

## Fan-out: keeping cache, search, warehouse, and views in sync

![A single ordered change stream lets cache, search, warehouse, and materialized views stay in sync without dual writes](/imgs/blogs/change-data-capture-and-the-outbox-pattern-4.webp)

Here is where the log earns its keep. Once your database's changes are an ordered stream, every one of the systems you used to keep in sync with a fragile dual write becomes a consumer of that stream. The figure shows the four canonical derived systems; let me give the concrete recipe for each, because the details differ in ways that matter.

**Cache invalidation.** This is the textbook use case and the one that motivates most teams' first CDC project. The classic cache bug is exactly a dual write: you update the database, then delete the cache key, and a crash (or a race with a concurrent read that repopulates the key) leaves a stale value cached forever. With CDC, a small consumer subscribes to the change stream and, for every change to a cached entity, invalidates (or proactively updates) the corresponding [Redis](/blog/software-development/database/redis-applications-and-optimization) key. The invalidation now derives from the *committed* change, in commit order, and is retried until it succeeds. You have moved cache invalidation from "a thing the write path hopes to do" to "a thing the log guarantees gets done."

```python
# A CDC cache-invalidation consumer (Kafka + Redis). Idempotent by construction:
# DEL is idempotent, and we key off the primary key in the change event.
from confluent_kafka import Consumer
import redis, json

r = redis.Redis(host="redis", decode_responses=True)
c = Consumer({"bootstrap.servers": "kafka:9092",
              "group.id": "cache-invalidator",
              "auto.offset.reset": "earliest",
              "enable.auto.commit": False})
c.subscribe(["dbserver.public.orders"])

while True:
    msg = c.poll(1.0)
    if msg is None or msg.error():
        continue
    change = json.loads(msg.value())          # Debezium envelope
    after, before = change.get("after"), change.get("before")
    op = change["op"]                          # c=create, u=update, d=delete, r=snapshot
    key_id = (after or before)["id"]
    r.delete(f"order:{key_id}")                # invalidate; next read repopulates
    c.commit(msg)                             # commit offset only after the side effect
```

**Search-index sync.** Your Elasticsearch/OpenSearch index is a derived view of your database. A CDC consumer maps each change event to an index operation: create/update become an `index` (upsert) call keyed by the document id; delete becomes a `delete`. Because the stream is ordered per key, you never apply a stale update over a fresh one *for the same document*. The huge operational win is reindexing: when you change your mapping, you spin up a new consumer group from offset 0, build a fresh index, and atomically swap an alias — no custom backfill script, because the log *is* the backfill.

**Warehouse and data-lake ingestion.** Analytics systems want the same change stream, landed as upserts (or as an append-only change history for slowly-changing dimensions). CDC is now the standard ingestion mechanism for Snowflake, BigQuery, Redshift, and lakehouse formats like Iceberg and Delta — a Kafka Connect sink consumes the change topics and writes them, replacing the brittle nightly `SELECT * WHERE updated_at > yesterday` batch that missed deletes and intermediate states. The result is a warehouse that lags production by seconds instead of a day, and that correctly reflects deletes. [Shopify documented exactly this migration](https://shopify.engineering/capturing-every-change-shopify-sharded-monolith): they replaced a query-based system ("Longboat") with log-based CDC using Debezium reading MySQL binlogs across a sharded monolith of 100+ databases, running ~150 connectors across a dozen Kubernetes pods and sustaining 65,000 records/second (peaking at 100,000 on Black Friday/Cyber Monday) at sub-10-second p99 latency from MySQL insert to Kafka — and, critically, the log-based approach finally captured hard deletes that the query-based predecessor structurally could not.

**Materialized views and read models.** This is the CQRS pattern. A consumer joins and aggregates several change streams into a precomputed read model — "current basket total per user," "order count per merchant per day" — and serves reads from it. The view is eventually consistent with the source, kept current by the log, and rebuildable from offset 0. This is also how microservices share data without sharing a database: service B subscribes to service A's change stream and maintains a *local* read-only replica of just the fields it needs, so B can serve reads even when A is down, with no synchronous cross-service call on the read path.

The unifying property across all four: each consumer is **independent**. The cache consumer falling behind does not slow the warehouse consumer; the search consumer crashing does not lose events (it resumes from its committed offset); a brand-new consumer can be added years later and replay history. That independence is the dividend of putting the log in the middle.

## At-least-once and the idempotent consumer

![An idempotent consumer that records processed event ids turns at-least-once redelivery into a single observable effect — effectively-once](/imgs/blogs/change-data-capture-and-the-outbox-pattern-5.webp)

Now we confront the hardest-to-internalize truth in this entire space: **there is no exactly-once delivery in a distributed system, and any system that claims it is either lying or quietly relying on idempotency.** The outbox is at-least-once. CDC is at-least-once. Kafka is at-least-once (its "exactly-once semantics" is a real feature, but it only applies *within* Kafka — to consume-transform-produce loops between Kafka topics — and evaporates the moment your side effect touches an external system like Redis or Elasticsearch). The reason is fundamental: between "perform the side effect" and "record that I performed it" there is always a gap, and a crash in that gap forces a choice — redeliver (risking a duplicate) or skip (risking a loss). At-least-once chooses redelivery. So your events *will* arrive more than once, and your job is to make the duplicate harmless.

The figure walks the exact sequence. At t0 the broker delivers event id=42. At t1 the consumer applies the side effect and records that it processed id=42. At t2 it crashes *before committing its consumer offset*, so the broker still thinks id=42 was never acknowledged. At t3 the broker redelivers id=42 (this is at-least-once doing its job). At t4 the consumer checks its dedup record, sees id=42 already processed, and **skips** — so the side effect happened exactly once even though the event was delivered twice. This is what people mean by **effectively-once** (or "exactly-once *processing*"): not magic exactly-once *delivery*, but at-least-once delivery plus an idempotent consumer.

There are three ways to make a consumer idempotent, in rough order of preference:

**1. Naturally idempotent operations.** Some side effects are idempotent by their nature: `SET key value` in Redis, `DELETE key`, an upsert keyed by primary key, "index document with id X." Applying these twice yields the same result as once. When you can express your consumer's side effect as one of these, you need no dedup table at all — this is why the cache-invalidation consumer above is correct without any bookkeeping. Prefer this whenever possible.

**2. A dedup table keyed by event id.** When the side effect is not naturally idempotent (incrementing a counter, sending an email, charging a card), record each processed event id and check before acting. Critically, the dedup insert and the side effect must be in the same transaction as the business effect when the side effect is a database write:

```sql
-- Idempotent consumer: the dedup insert and the business write are one transaction.
-- If the event was already processed, the INSERT conflicts and we skip.
BEGIN;

INSERT INTO processed_events (event_id, processed_at)
VALUES (:event_id, now())
ON CONFLICT (event_id) DO NOTHING;          -- returns 0 rows if already seen

-- Only apply the effect if we actually inserted (i.e. first time seeing this event).
-- In app code: if rowcount == 0, skip the effect and just commit the offset.
UPDATE account_balances
   SET balance = balance + :amount
 WHERE account_id = :account_id;

COMMIT;
```

This is exactly how [Stripe handles idempotency](https://stripe.com/blog/idempotency): clients send an `Idempotency-Key` header (a high-entropy UUID), Stripe records it, and a retried request with the same key returns the original result instead of charging the card again. And for the *other* direction — Stripe's webhooks to you — Stripe is explicit that delivery is at-least-once: every event has a stable `id` like `evt_1OxYz…` that stays constant across retries, and [Stripe's docs tell you to use that event id as your dedup key](https://docs.stripe.com/webhooks#handle-duplicate-events) because endpoints "might occasionally receive the same event more than once." The whole payments industry runs on at-least-once-plus-idempotency, not on exactly-once.

**3. Ordering and tombstones.** Idempotency handles duplicates; **ordering** handles staleness. If two updates to the same key arrive, you must apply them in the right order or you will overwrite a fresh value with a stale one. The mechanism is **partition-by-key**: Kafka guarantees ordering *within a partition*, and CDC tools set the message key to the row's primary key, so all changes to a given row land in the same partition and are consumed in commit order. This is per-key ordering, not global ordering — and per-key is exactly what you need, because the warehouse does not care whether order A's update is processed before or after an unrelated customer B's update; it only cares that order A's two updates are not reordered relative to each other.

A **tombstone** is the special case of deletes. When a row is deleted, log-compacted Kafka topics use a tombstone — a message with the row's key and a `null` value — to signal "this key is gone." Debezium emits a delete event followed by a tombstone so that downstream compaction can eventually drop all records for that key, and so that consumers like a cache know to evict it. Handling deletes correctly (including the tombstone) is the most commonly forgotten part of a CDC consumer, and it is exactly the part query-based CDC cannot do at all.

It is worth doing the arithmetic on why per-key ordering, rather than global ordering, is the right design — because the instinct to demand a single globally-ordered stream is exactly what kills CDC throughput. Suppose you have 12 partitions and 50,000 changes per second spread across millions of distinct keys. With per-key ordering, all twelve partitions process in parallel — roughly 4,000 changes/second each — and the only constraint is that two changes to the *same* key never reorder, which the partition-by-key guarantee gives you for free. With global ordering, every change must funnel through a single partition consumed by a single thread, capping you at whatever one consumer can do (often a few thousand per second) and throwing away eleven-twelfths of your parallelism. Global ordering is almost never what correctness actually requires: the warehouse genuinely does not care whether order #A's update lands before or after unrelated customer #B's update — it only cares that #A's two updates stay in order. Per-key ordering is the precise amount of ordering correctness needs and the maximum amount throughput can afford, which is why every serious CDC system (Databus, SpinalTap, Debezium) provides exactly that and no more.

> Stop chasing exactly-once delivery. Build at-least-once delivery and an idempotent, order-aware consumer. That combination is effectively-once, it is achievable, and it is what every payments system you trust actually runs.

## Snapshot then stream: the initial load problem

![Watermarks bracket each snapshot chunk so live log changes deterministically overwrite stale snapshot rows without table locks — the DBLog algorithm](/imgs/blogs/change-data-capture-and-the-outbox-pattern-6.webp)

When you turn on CDC for a table that already has a billion rows, the log only contains *future* changes — it does not contain the rows that were written before you started capturing. So every CDC system must solve the **bootstrap** problem: load the existing data (a snapshot), then seamlessly switch to tailing the log, with no gap (a row that changed during the snapshot must not be lost) and no inconsistency (a stale snapshot value must not overwrite a fresher log value). LinkedIn's Databus solved this with a dedicated **bootstrap service**: a separate store that a new or far-behind consumer reads to catch up, before switching to the live relay — decoupling slow consumers from the fast online change stream so one straggler could not stall the pipeline.

The naive approach is to lock the table, snapshot it, note the log position, unlock, and tail from there. Locking a billion-row table for the duration of a snapshot is a non-starter in production. The elegant solution is Netflix's [**DBLog watermark algorithm**](https://netflixtechblog.com/dblog-a-generic-change-data-capture-framework-69351fb9099b), described in the [2020 paper by Andreas Andreakis and Ioannis Papapanagiotou](https://arxiv.org/abs/2010.12597), and now the basis of Debezium's *incremental snapshots*. DBLog runs in production at Netflix across *tens of microservices*, works against both MySQL and Postgres, tracks progress externally (in ZooKeeper) by log sequence number, and its key innovation is doing chunked snapshots **interleaved with live log tailing, with no locks** — the only writes it makes to the source are the lightweight watermark rows.

The figure shows the watermark dance for one chunk. The algorithm processes the table in chunks (say 10,000 rows at a time) and, for each chunk, brackets it with watermarks written to a dedicated signal table:

1. **Write a low watermark** — a unique marker row — to the signal table. This insert appears in the WAL/binlog at a definite position.
2. **SELECT a chunk** of rows from the table (no lock, just a read).
3. **Write a high watermark** to the signal table. This too appears in the log at a definite position.
4. **Process the log between the watermarks.** The CDC stream is also being tailed live. For any primary key that appears *both* in the chunk's snapshot rows *and* in a log event that occurred between the low and high watermarks, the log event is authoritative — so that key is **removed from the chunk's snapshot rows** (the live change wins; the snapshot row was stale).
5. **Emit the deduplicated chunk**, then resume tailing the log normally until the next chunk.

The brilliance is that watermarks turn an ordering problem into a set-difference problem. The window between low and high watermark is precisely the region of uncertainty — any row touched in that window might have a stale snapshot value — and the algorithm resolves it deterministically by letting the log win for exactly those keys, with no locks and no need to pause writes. Because it is chunked, an incremental snapshot can be paused, resumed, and run concurrently with normal streaming; you can even kick off a re-snapshot of one table via a signal without restarting the connector. This is the difference between "CDC bootstrap" being a scary maintenance window and being a routine background operation.

## Outbox + CDC: the Debezium Event Router

![The Debezium outbox event router reads the outbox rows from the log and routes each event to a topic chosen by its aggregatetype column](/imgs/blogs/change-data-capture-and-the-outbox-pattern-9.webp)

Now we combine the two patterns into the production-grade architecture most teams should actually run. The transactional outbox gives you clean, intentional, atomically-committed events. Log-based CDC gives you a zero-load, low-latency, lossless relay. Put them together: your service writes its business change and an outbox row in one transaction, and **Debezium tails the outbox table from the WAL and publishes each row to Kafka.** The relay is no longer a polling loop you maintain; it is Debezium reading the log. This is the [Debezium Outbox Event Router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html), a Single Message Transform (SMT) purpose-built for exactly this pattern.

The Event Router expects the outbox table to have a known shape (the column names are configurable, but these are the defaults): an `id` (UUID, the event id), `aggregatetype` (which becomes the Kafka topic), `aggregateid` (which becomes the Kafka message key, giving you per-aggregate ordering), `type` (the event type), and `payload` (the event body). The SMT reads each inserted outbox row from the change stream and **routes it**: an event with `aggregatetype = 'Order'` goes to the `Order` topic; one with `aggregatetype = 'Customer'` goes to the `Customer` topic; and the `aggregateid` is set as the message key so all events for one order land in the same partition and stay ordered. The figure traces this: service transaction → outbox table → WAL → EventRouter SMT → per-aggregate topic → consumers.

Here is the Debezium connector configuration for the Postgres source with the outbox router:

```json
{
  "name": "outbox-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "${file:/secrets:dbz-password}",
    "database.dbname": "shop",
    "topic.prefix": "shop",
    "plugin.name": "pgoutput",
    "slot.name": "debezium_outbox",
    "publication.name": "dbz_outbox_pub",
    "table.include.list": "public.outbox",
    "snapshot.mode": "initial",

    "transforms": "outbox",
    "transforms.outbox.type": "io.debezium.transforms.outbox.EventRouter",
    "transforms.outbox.route.by.field": "aggregatetype",
    "transforms.outbox.route.topic.replacement": "${routedByValue}.events",
    "transforms.outbox.table.field.event.key": "aggregateid",
    "transforms.outbox.table.field.event.payload": "payload",
    "transforms.outbox.table.fields.additional.placement": "type:header:eventType",

    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": false
  }
}
```

The `snapshot.mode` setting is where the snapshot-then-stream machinery plugs in: `initial` (snapshot the table once when no offset exists, then stream — the default), `never` (skip the snapshot and stream from the current log position, for when you have already bootstrapped another way), `initial_only` (snapshot and stop), and `no_data` (capture the schema but no existing rows, then stream — this is what older Debezium versions called `schema_only`). For large tables, you enable incremental snapshots via a signal table and the DBLog watermark algorithm above, so the initial load does not block streaming — Debezium's incremental snapshot is, by its own documentation, a direct implementation of Andreakis and Papapanagiotou's DBLog paper.

One subtle but important benefit: because Debezium tracks its position via the replication slot's LSN, the outbox row's *only* job is to exist long enough to be captured. You can **delete the outbox row in the same transaction that inserts it** (an insert immediately followed by a delete still produces an insert event in the WAL), so the outbox table stays empty and never bloats — a trick Debezium explicitly supports. The event is captured from the log; the table is just a staging area for the WAL record.

[Confluent](https://www.confluent.io/blog/) — the company Kreps founded around Kafka — documents this same outbox-plus-CDC architecture as the recommended way to bridge a database and Kafka without dual writes, using Kafka Connect to run Debezium and SMTs to shape the events. The lineage is direct: the log idea (Kreps), the log-tailing implementation (Debezium), and the application pattern (outbox) all converge here.

## Schema and event evolution

The events you publish today will be consumed by code you have not written yet, possibly years from now, possibly after you have changed your database schema three times. Event evolution is therefore a first-class concern, and the outbox plus a schema registry is the standard answer.

The key discipline is to **decouple your event schema from your table schema.** This is a major advantage of the outbox over raw table-level CDC: with raw CDC, your Kafka events *are* your table rows, so every column rename or type change ripples out to every consumer. With the outbox, the `payload` is a deliberately-designed event contract that you control independently of how the data happens to be stored. You can refactor `orders` into three tables and keep emitting the same `OrderPaid` event.

For the event payloads themselves, use a schema registry (Confluent Schema Registry, Apicurio) with Avro, Protobuf, or JSON Schema, and enforce **backward and forward compatibility**: new fields are optional with defaults (so old consumers ignore them and new consumers tolerate their absence in old events), and you never remove or repurpose a field. Version the event type in the envelope (`"eventType": "OrderPaid", "version": 2`) when you must make a breaking change, and run old and new consumers in parallel during the migration. The registry rejects incompatible schema changes at publish time, which turns "we broke a downstream team's consumer" from a production incident into a CI failure.

| Evolution scenario | Safe approach | What breaks if you ignore it |
| --- | --- | --- |
| Add a field | Optional with a default | Nothing — old consumers ignore it |
| Remove a field | Deprecate, keep emitting, remove later | Consumers reading the field NPE/KeyError |
| Rename a field | Add new, emit both, migrate consumers, drop old | Every consumer reading the old name breaks |
| Change a field's type | New field + version bump, dual-emit | Deserialization fails for all consumers |
| Change topic routing | New topic, dual-publish, migrate, retire old | Consumers on the old topic silently stop |

## Operating CDC in production: the parts nobody tells you about

The patterns above are the easy 80%. The 20% that determines whether your CDC pipeline is an asset or a liability is operational, and it is almost entirely about three things: monitoring the right signals, understanding precisely how far the "exactly-once" guarantee extends, and handling the source database's lifecycle (failover, schema changes, backfills). A senior rule of thumb here: **the steady state of a CDC pipeline is boring; all the danger lives in the transitions** — startup, failover, schema migration, and consumer lag.

### Monitor end-to-end lag and slot retention, not connector liveness

The naive monitoring story is "alert if the connector is down." That is necessary but wildly insufficient, because the two failure modes that actually hurt you — a slot pinning WAL until the disk fills, and a consumer silently falling behind until events are hours stale — both happen while the connector reports perfectly healthy. The signals that matter form a chain from the source database to the final consumer, and you want a number at each link.

| Signal | Where | Why it matters | Alert when |
| --- | --- | --- | --- |
| Replication slot WAL retention | `pg_replication_slots` | Retained WAL fills the primary's disk | grows monotonically, or > a few GB |
| Slot `active` flag | `pg_replication_slots` | An inactive slot still retains WAL | `false` for more than a few minutes |
| Connector source lag (LSN behind head) | Debezium JMX metrics | Time between commit and capture | > a few seconds sustained |
| Kafka consumer group lag | `kafka-consumer-groups` | Events captured but not yet processed | grows or exceeds an SLO |
| End-to-end latency | synthetic probe row | The number your users actually feel | > your freshness SLA |

The single most valuable one is the **synthetic end-to-end probe**: periodically write a heartbeat row into the source, timestamp it, and measure how long until it appears at the far end of the pipeline (in the cache, the search index, the warehouse). That one number captures every link at once and is the only metric that maps directly to "how stale is the data my users see." Everything else is a decomposition you reach for when the probe goes red.

```sql
-- Source side: a cron writes a heartbeat row every 10 seconds.
INSERT INTO cdc_heartbeat (id, beat_at)
VALUES (1, now())
ON CONFLICT (id) DO UPDATE SET beat_at = now();
-- Consumer side: when this row's change arrives, export a gauge:
--   end_to_end_lag_seconds = now() - beat_at
-- Alert if it exceeds your freshness SLA. This single number
-- subsumes slot lag, connector lag, and consumer lag end to end.
```

This heartbeat does double duty: it also keeps the replication slot advancing during quiet periods. If your captured tables are idle but *other* tables generate WAL, the slot's confirmed-flush LSN can stall and WAL accumulates anyway — which is exactly what Debezium's `heartbeat.interval.ms` setting prevents by periodically emitting a heartbeat that advances the slot. A heartbeat table you control is the belt to that suspenders.

### The "exactly-once" illusion, stated precisely

It is worth being surgical about what each layer guarantees, because vague "exactly-once" marketing causes real double-counting bugs (see case study 10). Here is the precise ladder, from the source to your side effect:

- **Postgres → WAL**: each committed change is in the log exactly once, in commit order. Solid ground.
- **WAL → Debezium → Kafka**: at-least-once. If Debezium publishes to Kafka and crashes before recording its offset (LSN), it re-reads from the last committed LSN and re-publishes. Kafka topics therefore can contain duplicate change events (rare, but possible on connector restart).
- **Kafka → Kafka (consume-transform-produce)**: Kafka's exactly-once semantics (EOS) genuinely applies *here and only here* — a transaction spanning the consumer's offset commit and the producer's writes makes a read-process-write loop atomic *within Kafka*.
- **Kafka → external side effect (Redis, Elasticsearch, a warehouse, an email)**: at-least-once, full stop. EOS does not extend past Kafka's boundary. The moment your consumer touches an external system, a crash between "do the side effect" and "commit the offset" forces redelivery.

The practical conclusion is unambiguous: **assume duplicates everywhere a boundary is crossed, and make the final side effect idempotent.** There is exactly one place EOS buys you something for free — Kafka-to-Kafka — and your business logic almost never lives entirely there. Every claim of "exactly-once delivery" you read should be mentally rewritten as "at-least-once delivery, made effectively-once by idempotency I still have to implement." This is not pessimism; it is the actual contract, and the payments industry — Stripe, Shopify — runs on it deliberately rather than fighting it.

### Source failover, schema changes, and security

Three more operational realities round out the picture. **Failover:** a Postgres logical replication slot exists only on the primary, and historically did not survive a failover to a replica — promoting a standby meant the slot was gone and the connector had to re-snapshot (Postgres 16+ improves this with slot synchronization to standbys, but verify your version and tooling). Plan your CDC failover story explicitly; do not discover it during an incident. **Schema changes:** logical decoding does *not* emit DDL, so an `ALTER TABLE` on the source is invisible to the change stream — Debezium handles column adds reasonably, but renames and type changes need the same backward-compatible discipline as event evolution, applied to the *table* this time. Coordinate DB migrations with the CDC pipeline; a careless `DROP COLUMN` can break every consumer. **Security and multi-tenancy:** the change stream contains every column of every captured row, including PII and secrets, so treat CDC topics as sensitive — use column filtering or masking SMTs to drop or hash sensitive fields *before* they hit Kafka, restrict the replication role to least privilege, and in a multi-tenant system be deliberate about whether tenants share topics (cheaper, but a consumer bug can leak across tenants) or get isolated ones (safer, but more topics to manage). The outbox helps here too: because you hand-craft the `payload`, you simply never put a secret in it, whereas raw table CDC ships whatever happens to be in the column.

## Case studies from production

### 1. The replication slot that filled the disk

A team turned on Debezium against their primary Postgres and went home. Over the weekend, the Kafka Connect cluster hit a bad deploy and the connector stayed down for 40 hours. Because a logical replication slot retains all WAL the consumer has not read, Postgres dutifully kept every WAL segment from the moment the connector died. Monday morning the primary's disk hit 100% and Postgres stopped accepting writes — a full production outage caused not by load but by a *paused* CDC consumer. The wrong first hypothesis was "a runaway query is writing too much WAL." The actual cause was the inactive slot pinning WAL. The fix: alert on `pg_replication_slots.restart_lsn` lag (the `retained_wal` query above), set a `max_slot_wal_keep_size` cap (Postgres 13+) so a dead consumer drops its slot instead of killing the database, and configure Debezium's heartbeat so an idle-but-alive connector still advances the slot. The lesson: a replication slot is a loaded gun pointed at your primary's disk; monitor it like one.

### 2. The cache that served deleted users

An auth service updated the database and then deleted the user's Redis session on logout/deactivation — a classic dual write. Under normal load it was invisible. During a deploy, pods were killed mid-request after the DB commit but before the Redis delete, leaving deactivated users with live sessions — a security finding, not just a staleness bug. The team had assumed "Redis delete almost never fails," which is true and irrelevant: the problem was not Redis failing but the *process* dying between the two writes. The fix was a CDC consumer on the `users` table that evicts the session key on every change, deriving the invalidation from the committed change instead of hoping the write path completes both steps. The lesson: "the second write rarely fails" is not the same as "the second write always happens"; a crash between two writes is a different failure than either write failing.

### 3. The search index that drifted 0.3% per month

An e-commerce search index was maintained by application code that wrote to Elasticsearch right after writing to the database. Over months, the index and the database diverged by a fraction of a percent — a few hundred products out of a hundred thousand were wrong: deleted products still searchable, price changes not reflected. Each individual miss was a transient failure (an ES timeout, a deploy, a thrown exception swallowed by a `try/except`) that nobody noticed because there was no reconciliation. The wrong hypothesis was "Elasticsearch is dropping writes." The real cause was thousands of tiny dual-write gaps accumulating. Switching to CDC made every committed change drive an index operation, retried until it succeeded and ordered per document. The drift went to zero, and reindexing on a mapping change became "reset the consumer to offset 0" instead of a custom multi-hour backfill. The lesson: dual-write divergence is rarely a single dramatic loss; it is slow, silent drift that no single error log captures.

### 4. The outbox insert that was not in the transaction

A team implemented the outbox correctly — table, relay, idempotent consumers — but still lost events occasionally. The bug: the ORM was configured so that the outbox `INSERT` ran on a *different* connection from the business `UPDATE` (a separate repository with its own session). The two writes were each transactional but not *co*-transactional, so a crash between them lost the outbox row exactly as a raw dual write would. It passed every test because tests ran both writes to completion. The fix was to assert, in a single integration test, that rolling back the business transaction also rolls back the outbox insert — if the outbox row survives a business rollback, the connection is wrong. The lesson: the outbox pattern's entire correctness rests on one invariant — same transaction — and that invariant is invisible in code review and easy to break with ORM session management. Test it explicitly.

### 5. The duplicate emails from at-least-once

A notification service consumed an `OrderShipped` topic and sent an email per event. After a Kafka rebalance, customers received two shipping emails. The team's first reaction was "Kafka delivered the message twice — Kafka is broken." Kafka was working exactly as designed: the consumer processed the event, sent the email, and then crashed before committing its offset, so on rebalance the partition was reassigned and the event redelivered. At-least-once delivery means duplicates on redelivery — that is the contract, not a bug. The fix was a dedup table keyed by the event id (`evt_…`), checked before sending, with the dedup insert and the offset commit ordered so a duplicate is skipped. The lesson: at-least-once is not a Kafka defect to be escalated; it is the guarantee, and the consumer owns deduplication. There is no configuration flag that makes the duplicate go away — only an idempotent consumer.

### 6. The reordered balance updates

A wallet service consumed account-change events and applied balance deltas. Occasionally a balance went briefly negative and then corrected — two updates to the same account arrived out of order because the producer was hashing to partitions by a field that was not the account id, so the two updates landed in different partitions with independent ordering. The wrong hypothesis was "a race condition in our balance code." The real cause was lost per-key ordering: ordering is only guaranteed within a partition, and these updates were not co-partitioned. The fix was to set the Kafka message key to the account id (which Debezium does by default from the primary key, but this was a hand-rolled producer that did not), co-partitioning all changes to one account. The lesson: CDC gives you per-key ordering *only if the key is the entity's identity*; choose the partition key as the thing whose updates must not be reordered.

### 7. The snapshot that locked the table

An early CDC setup used a tool that took a `FLUSH TABLES WITH READ LOCK`-style consistent snapshot to bootstrap. On a 400 GB table, that lock was held long enough to stall all writes and trigger application timeouts cascading into a partial outage during what was supposed to be a zero-impact rollout. The team had tested on a small table where the snapshot took seconds. The fix was to migrate to Debezium's incremental snapshot (the DBLog watermark algorithm), which chunks the snapshot, interleaves it with live streaming, and never takes a table lock — so the 400 GB bootstrap ran for hours in the background with zero impact on writes. The lesson: the bootstrap is the dangerous part of CDC, not the steady-state streaming; choose a lock-free, chunked, resumable snapshot mechanism and test it on a production-sized table.

### 8. The schema change that broke six consumers

A producer team renamed a field in their event payload from `amount` to `amount_cents` and deployed. Six downstream consumers — in other teams' services — broke simultaneously, because they all read `amount` and got `null`. There was no schema registry enforcing compatibility, so the breaking change shipped silently and surfaced as six separate incidents over the next hour. The fix was to introduce a schema registry with backward-compatibility enforcement, which would have rejected the rename at publish time, and to adopt the "add new field, emit both, migrate consumers, then remove old" migration discipline. The lesson: an event payload is a public API with more consumers than you can see, and the only scalable way to evolve it safely is a registry that enforces compatibility as a build-time gate, not a runtime surprise.

### 9. The poll-based relay that fell behind at peak

A team ran a polling-publisher outbox relay with a 5-second interval and a `LIMIT 500` batch. At normal volume it kept up. During a flash sale, the outbox accumulated rows faster than one worker publishing 500 every 5 seconds could drain it, and event lag grew to 20 minutes — the warehouse and the email service fell hours behind. The wrong hypothesis was "Kafka is slow." The actual cause was relay throughput: 100 events/second is nowhere near a sale's write rate. The fix was twofold: run multiple relay workers with `FOR UPDATE SKIP LOCKED` so they drain disjoint batches in parallel, and for the long term migrate the relay to Debezium log-tailing, which is not bounded by a poll interval. The lesson: a polling relay has a throughput ceiling set by `batch_size / poll_interval × workers`; size it for peak, not average, or move to log tailing where there is no poll at all.

### 10. The "exactly-once" pipeline that double-counted revenue

A finance pipeline was built on Kafka's exactly-once semantics (EOS) and the team believed double-counting was impossible. Then a reconciliation found revenue overstated by 0.2%. The misunderstanding: Kafka EOS guarantees exactly-once for *consume-transform-produce within Kafka* — reading a topic, transforming, and writing another topic, all in one Kafka transaction. The pipeline's final step wrote to an external warehouse, and that write was *outside* the Kafka transaction, so a retry after a crash double-wrote a batch. EOS never covered the external sink. The fix was to make the warehouse write idempotent (upsert keyed by a deterministic event id, so a re-applied batch is a no-op). The lesson: "exactly-once" in Kafka is a precise, bounded guarantee that stops at the edge of Kafka; the moment your side effect touches an external system, you are back to at-least-once and you need idempotency. Read the fine print on every exactly-once claim.

### 11. The microservice that coupled through a shared database

Two services read and wrote the same Postgres tables to "share data," which meant a schema change by one team broke the other, and neither could deploy independently. They tried to decouple with synchronous REST calls, which created a new failure mode: service B's reads failed whenever service A was down. The durable fix was CDC-driven local read models: service A publishes its changes (via outbox + Debezium), and service B consumes them into a small, local, read-only projection of exactly the fields it needs. Now B serves reads from its own store even when A is down, A can refactor its schema freely as long as it keeps emitting the same events, and the two deploy independently. The lesson: a shared database is the tightest possible coupling between services; CDC lets you share *data* (the change stream) without sharing *state* (the tables), which is what actually enables independent deployment.

### 12. The audit log that missed the deletes

A compliance team built an audit trail with query-based CDC — polling `WHERE updated_at > :last`. It passed the initial audit. A year later, a real investigation needed the history of records that had since been *deleted*, and they were not in the audit log at all, because a deleted row simply stops appearing in query results — there is no `updated_at` to poll for a row that no longer exists. The wrong assumption was "polling captures all changes." It captures all *surviving* rows' latest state, which is a strict subset of all changes. The fix was log-based CDC, which captures the delete operation itself (with the before-image, given `REPLICA IDENTITY FULL`), giving a complete, append-only history including deletions. The lesson: for any use case that needs deletes or intermediate states — audit, event sourcing, accurate analytics — query-based CDC is structurally incapable, and only log-based (or trigger-based) capture will do.

## When to reach for this, and when not to

The outbox and CDC are powerful, but they are infrastructure, and infrastructure has carrying cost. Here is the honest decision boundary.

**Reach for the transactional outbox when:**

- You publish events (to Kafka, SQS, a webhook, anything) as a consequence of a database change, and correctness requires that the event and the change never disagree. This is the default for any event-driven service.
- You are doing microservice integration and need reliable domain events without a shared database or distributed transactions.
- Your event volume is low to moderate and you want the simplest correct relay — a polling publisher and a table get you there with no new infrastructure.

**Reach for log-based CDC (Debezium et al.) when:**

- You need to keep multiple derived systems — cache, search, warehouse, read models — in sync with a database, and you are tired of N bespoke pipelines.
- You need low-latency, lossless, complete capture including deletes and intermediate states (audit, event sourcing, real-time analytics).
- You want to ingest into a warehouse/lake continuously instead of with a nightly batch that misses deletes.
- You are running the outbox at high volume and want the relay to be a log tailer instead of a polling loop — the outbox-plus-Debezium combination.

**Skip or defer all of this when:**

- You have exactly one data store and no downstream systems to keep in sync. There is no dual write, so there is nothing to solve — do not add Kafka to a CRUD app that has no consumers.
- You can express the integration as a single transaction in a single database (e.g. a materialized view Postgres maintains for you). Let the database do it; do not rebuild it on Kafka.
- The data is small and a periodic full reload is genuinely cheaper than the operational cost of CDC. A 10,000-row reference table re-exported every hour does not need a replication slot.
- You cannot staff the operational burden. CDC means owning a Kafka Connect cluster, replication slots, schema evolution, and the disk-fill failure mode from case study 1. If you cannot monitor WAL retention and respond to a stuck connector, query-based CDC or a polling outbox is a safer fit than log-based CDC.

The throughline of this entire article is one idea, stated three ways. A dual write is two operations that can break independently, so do not do it. The outbox makes them one atomic operation, so they cannot. CDC derives every downstream system from the one log the database already keeps, so they all converge on the same truth and any of them can be rebuilt from offset zero. Get those three ideas right and the 3 a.m. page about two systems disagreeing — the one that is never on the runbook — simply stops happening.

## Further reading

- Jay Kreps, [*The Log: What every software engineer should know about real-time data*](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) — the foundational essay on the log as a unifying primitive.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapters 11 (Stream Processing) and 12 (The Future of Data Systems) — change data capture, keeping derived data in sync, and unbundling the database.
- Chris Richardson, [Transactional Outbox](https://microservices.io/patterns/data/transactional-outbox.html), [Polling Publisher](https://microservices.io/patterns/data/polling-publisher.html), and [Transaction Log Tailing](https://microservices.io/patterns/data/transaction-log-tailing.html) on microservices.io.
- [Debezium documentation](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html) — the Outbox Event Router SMT and the Postgres connector.
- Andreas Andreakis & Ioannis Papapanagiotou, [DBLog: A Generic Change-Data-Capture Framework](https://netflixtechblog.com/dblog-a-generic-change-data-capture-framework-69351fb9099b) (Netflix Tech Blog) and the [DBLog paper](https://arxiv.org/abs/2010.12597).
- Shirshanka Das et al., [*All Aboard the Databus!*](https://dl.acm.org/doi/abs/10.1145/2391229.2391247) (LinkedIn, SoCC '12) — the relay/bootstrap/consumer architecture that anticipated Kafka-based CDC.
- Jad Abi-Samra, [Capturing Data Evolution in a Service-Oriented Architecture](https://medium.com/airbnb-engineering/capturing-data-evolution-in-a-service-oriented-architecture-72f7c643ee6f) (Airbnb, SpinalTap) and John Martin & Adam Bellemare, [Capturing Every Change From Shopify's Sharded Monolith](https://shopify.engineering/capturing-every-change-shopify-sharded-monolith) (Debezium at 65K–100K records/sec).
- [Stripe: Designing robust and predictable APIs with idempotency](https://stripe.com/blog/idempotency) — idempotency keys and at-least-once delivery in a payments system.
- [PostgreSQL: Logical Decoding](https://www.postgresql.org/docs/current/logicaldecoding-explanation.html) — replication slots, output plugins, `REPLICA IDENTITY`, and the WAL-retention failure mode.
- Sibling posts on this blog: [database replication](/blog/software-development/database/database-replication-sync-async-logical-physical), [the write-ahead log](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability), [Redis in production](/blog/software-development/database/redis-applications-and-optimization), and [partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).

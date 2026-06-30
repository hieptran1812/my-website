---
title: "Polyglot Persistence: Choosing the Right Store for Each Workload"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "At scale no single database serves every access pattern well — here is how to pick the right specialized store per workload without drowning in operational sprawl or corrupting your data with dual writes."
tags: ["polyglot-persistence", "database-scaling", "system-of-record", "change-data-capture", "postgres", "elasticsearch", "cassandra", "clickhouse", "data-architecture", "cdc"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 32
---

The most expensive architecture mistake I have watched teams make is not picking the wrong database. It is picking the *second* one too early. A team hits a slow full-text query, reads that Elasticsearch is "built for search," stands up a three-node cluster, and ships a dual-write. Six months later the search index is silently out of sync with the source of truth, nobody can explain which records are stale, and the on-call rotation has quietly absorbed a distributed system that two people half-understand. The original slow query, it turns out, was a missing GIN index on a `tsvector` column — fifteen minutes of work in the database they already ran.

That is the tension this post is about. The thesis cuts both ways and you have to hold both halves at once: **at sufficient scale, one database genuinely cannot serve every access pattern well — full-text search, time-series metrics, graph traversal, blobs, high-write key-value, and large-scale analytics are each served 10–100× better by a specialized engine — but every new store you add is a permanent operational cost and, worse, a new consistency boundary where your data can drift out of agreement with itself.** Polyglot persistence is the discipline of choosing those stores deliberately, one justified workload at a time, rather than reflexively reaching for a new engine every time a query is slow.

![One write path into Postgres; search, cache, analytics, and graph stores are all downstream projections of its change stream](/imgs/blogs/polyglot-persistence-choosing-the-right-store-1.webp)

The diagram above is the mental model for the entire article. There is exactly one write path: the application commits to Postgres, the system-of-record. Everything else — the Elasticsearch full-text index, the Redis hot cache, the ClickHouse analytics warehouse, the Neo4j graph projection — is a *derived store*, fed downstream from the same change stream. No service ever writes the "truth" to two places. The truth lives in one engine, and the change stream replays every committed mutation into the others. The rest of this post is a tour of that picture: the taxonomy of stores you might fan out to, how to decide which workload earns one, what each store actually costs, why dual writes are a trap, how change data capture closes the loop, and why "just use Postgres" beats most of this for most teams most of the time.

## Why "one database for everything" breaks down

Start with the assumption, because it is usually correct. A single relational database — Postgres or MySQL — is the right default for the overwhelming majority of applications, and it stays right far longer than most engineers expect. It gives you transactions, joins, secondary indexes, a query planner, a mature backup story, and one operational surface. The failure is not that the relational model is weak. The failure is that a *general-purpose* engine makes general-purpose tradeoffs, and a handful of workloads have access patterns so specific that a purpose-built engine beats the generalist by one or two orders of magnitude on the dimension that matters.

| Assumption | The naive view | The reality at scale |
| --- | --- | --- |
| "The database can do search" | `LIKE '%term%'` is full-text search | A leading-wildcard `LIKE` can't use a B-tree; relevance ranking, faceting, and typo tolerance need an inverted index |
| "Metrics are just rows" | Insert a row per metric sample | A row store at 1M samples/sec dies on write amplification and index bloat; a TSDB compresses 10–50× and downsamples natively |
| "Analytics is a big `GROUP BY`" | Run the aggregate on the primary | Scanning a billion rows row-by-row is bounded by random I/O; a columnar engine reads only the touched columns, often 50–100× faster |
| "We'll cache in the database" | Add more RAM to Postgres | The buffer pool competes with your working set; a dedicated KV cache gives sub-millisecond point reads off the hot path |
| "A graph is just join tables" | Recursive CTE over `edges` | Each hop is another self-join; a 6-hop traversal explodes combinatorially where a graph engine walks pointers |

> A specialized store is not "better than Postgres." It is better than Postgres *on exactly one axis*, and worse on every other axis — including the ones you forgot to measure, like operability and backups.

The job, then, is not to collect databases. It is to identify which of your workloads has crossed the threshold where the specialist's order-of-magnitude win outweighs the very real cost of running a second engine — and to wire that specialist up so it can never disagree with your source of truth. Both halves are load-bearing. Skip the first and you sprawl; skip the second and you corrupt.

## 1. The taxonomy: nine stores and when each one wins

**A senior rule of thumb: name the access pattern before you name the product.** Engineers reach for "MongoDB" or "Cassandra" as if the brand were the decision. The decision is the *shape* of access — point lookup, range scan, full-text, aggregate, traverse — and the data model that shape implies. Group the stores by that shape and the map gets small.

![Nine engine classes split into three families by the shape of access they are built to serve](/imgs/blogs/polyglot-persistence-choosing-the-right-store-2.webp)

The taxonomy above splits nine engine classes into three families. The *point & transactional* family answers "give me this row, or these rows, by key" with strong consistency. The *shaped & searchable* family handles data whose structure or text content is the query. The *analytical & bulk* family is built for scans, aggregates, and large opaque objects. Here is the full map with the one access shape each store is built around.

| Store class | Best workload | Native query shape | Consistency model | Primary scale axis | Example systems |
| --- | --- | --- | --- | --- | --- |
| Relational / OLTP | Transactions with joins; the default | Point + range + join, ACID | Strong, serializable | Vertical, then read replicas / shards | Postgres, MySQL |
| Key-value | Cache, session, high-throughput lookup | Point GET/SET by key | Tunable (often eventual) | Horizontal, near-linear | Redis, DynamoDB |
| Wide-column | Massive time-ordered writes, known query | Partition key + clustering range | Tunable (quorum) | Horizontal, near-linear | Cassandra, ScyllaDB, Bigtable |
| Document | Flexible, document-shaped aggregates | Key + secondary index on fields | Tunable per-operation | Horizontal sharding | MongoDB |
| Search | Full-text, faceting, relevance ranking | Inverted-index match + score | Eventual (near-real-time) | Horizontal sharding | Elasticsearch, OpenSearch |
| Time-series | Metrics, downsampling, retention | Time range + tag filter + rollup | Eventual | Horizontal, time-partitioned | Prometheus, InfluxDB, TimescaleDB |
| Graph | Relationship traversal, pathfinding | Variable-depth traversal | Strong (single-instance) | Hard to shard; vertical-first | Neo4j |
| Columnar / OLAP | Analytics, aggregates over billions | Column scan + aggregate | Eventual / batch | Horizontal, MPP | ClickHouse, Snowflake, BigQuery |
| Object / blob | Large files, media, backups | PUT/GET by key, no query | Strong read-after-write | Effectively infinite | S3, GCS |

A short tour of *when to reach for each*, framed as the threshold at which the specialist's win becomes real:

**Relational / OLTP** is the default and the source of truth. Reach for it first, always. It is the only family on this list that gives you multi-row transactions and ad-hoc joins without heroics. You leave it not because it is bad but because one specific workload has outgrown it.

**Key-value** wins when you need a point lookup by a known key at sub-millisecond latency and very high throughput, and you do not need range queries or joins. Sessions, rate-limit counters, feature flags, and read-through caches are the canonical cases. DynamoDB extends this to durable, horizontally-scaled KV with predictable single-digit-millisecond latency; Redis trades durability for raw speed and rich data structures.

**Wide-column** wins for write-heavy, time-ordered, *known-in-advance* query patterns at a scale where a single primary cannot keep up. The price is that you must design the table around the one query you will run — the partition key and clustering columns *are* the query plan, and there is no ad-hoc `JOIN` to bail you out later. This is why Cassandra and ScyllaDB dominate message stores, event logs, and IoT telemetry.

**Document** wins when your aggregates are genuinely document-shaped — a deeply nested object you almost always read and write whole — and your schema varies enough that a fixed relational schema fights you. Be honest here: Postgres `JSONB` covers most "we need flexible documents" cases without leaving the relational world. You graduate to a dedicated document store when sharding the documents horizontally becomes the dominant requirement.

**Search** wins the moment the query is the *text itself*: full-text relevance, fuzzy matching, faceted navigation, autocomplete. A relational `LIKE '%x%'` cannot use an index for a leading wildcard, and bolting on ranking and faceting in SQL is a losing battle past a few million documents. An inverted index is a fundamentally different data structure built for exactly this.

**Time-series** wins for metrics and telemetry — append-mostly, time-stamped, queried by time range and tag, then downsampled and aged out. A TSDB compresses time-ordered data 10–50× over a row store and answers "p99 over the last 6 hours, rolled up to 1-minute buckets" natively. TimescaleDB is the interesting hybrid: it is a Postgres extension, so you can stay in the relational world up to surprisingly high ingest rates.

**Graph** wins when the *relationships* are the query and the traversals are deep and variable — fraud rings, recommendation paths, dependency graphs, social connections. Each hop in a relational model is another self-join; a graph engine stores adjacency as pointers and walks them in roughly constant time per hop. Below a few hops, a recursive CTE in Postgres is fine; the graph engine earns its keep when depth and fan-out explode.

**Columnar / OLAP** wins for analytics: aggregates and scans over hundreds of millions to billions of rows, where you touch a few columns of very many rows. A row store reads whole rows off disk to sum one column; a columnar engine reads only that column, compresses it heavily, and vectorizes the aggregate. The win is routinely 50–100× on the right query.

**Object / blob** wins for anything large and opaque — images, video, model checkpoints, data-lake files, backups. You do not query the bytes; you store a pointer (and metadata) in your relational database and the object in S3. Putting megabyte blobs *in* a row store bloats the buffer pool and wrecks vacuum/replication.

The discipline is that every one of these is a *deliberate graduation*, justified by a specific workload that has provably outgrown the default — not a default you reach for because the brand is famous.

## 2. How to choose: query shape is the strongest signal

**Rule of thumb: the shape of your query predicts the store far better than the size of your data.** Teams obsess over row counts ("we have 500 million rows, we need to scale out") when the row count is rarely what breaks. What breaks is asking a store to serve an access pattern it was never built for. A billion rows that you only ever fetch by primary key is trivial; ten million rows that you full-text-search with relevance ranking and faceting will fight a relational engine hard.

![Each access shape has one native winner; Postgres is workable almost everywhere but native almost nowhere except transactions](/imgs/blogs/polyglot-persistence-choosing-the-right-store-3.webp)

Read the matrix above by row, not by column. Each *query shape* has exactly one native winner, and the green cells march down a near-diagonal: transactional writes want Postgres, point lookups want Redis or Cassandra, time-ordered ranges want Cassandra or ClickHouse, full-text wants Elasticsearch, billion-row aggregates want ClickHouse. Notice the Postgres column: it is "workable" almost everywhere and "native" almost nowhere except transactions. That is exactly what a general-purpose engine looks like — competent across the board, dominant on its home turf, beaten by a specialist on each specialist's home turf. The decision is never "is Postgres good enough" in the abstract; it is "has *this query shape* crossed the line where workable is no longer good enough."

That makes the choice mechanical enough to encode. Here is a small decision function that takes a workload's attributes and recommends a store class. It is not magic — it is the matrix above plus the cost bias toward staying relational, written down so the reasoning is explicit and reviewable instead of living in someone's gut.

```python
from dataclasses import dataclass
from enum import Enum


class QueryShape(str, Enum):
    POINT = "point_lookup"          # fetch by known key
    RANGE = "range_scan"            # ordered scan within a partition
    FULLTEXT = "full_text"          # relevance-ranked text match
    AGGREGATE = "aggregate"         # GROUP BY over many rows
    TRAVERSE = "graph_traverse"     # variable-depth relationship walk
    BLOB = "large_object"           # opaque bytes, no query


@dataclass
class Workload:
    query_shape: QueryShape
    write_rate_per_sec: int         # sustained writes/sec
    read_rate_per_sec: int          # sustained reads/sec
    needs_transactions: bool        # multi-row atomicity / joins
    consistency: str                # "strong" | "eventual"
    p99_latency_budget_ms: float
    dataset_rows: int


def recommend_store(w: Workload) -> str:
    # 1. Hard requirements win first. Transactions + joins => stay relational.
    if w.needs_transactions and w.query_shape in (QueryShape.POINT, QueryShape.RANGE):
        return "relational/OLTP (Postgres) — system-of-record"

    # 2. Blobs never belong in a row store.
    if w.query_shape is QueryShape.BLOB:
        return "object store (S3) — pointer + metadata in Postgres"

    # 3. Query shape routes the specialists.
    if w.query_shape is QueryShape.FULLTEXT:
        return "search (Elasticsearch) — derived index, fed by CDC"
    if w.query_shape is QueryShape.TRAVERSE:
        return "graph (Neo4j) — derived projection if relational is SoR"
    if w.query_shape is QueryShape.AGGREGATE and w.dataset_rows > 100_000_000:
        return "columnar/OLAP (ClickHouse) — derived warehouse, fed by CDC"

    # 4. High-write, time-ordered, known query => wide-column.
    if w.query_shape is QueryShape.RANGE and w.write_rate_per_sec > 50_000:
        return "wide-column (Cassandra/ScyllaDB)"

    # 5. Sub-ms point reads at high throughput => KV cache (derived).
    if (w.query_shape is QueryShape.POINT
            and w.p99_latency_budget_ms < 2
            and w.read_rate_per_sec > 50_000):
        return "key-value cache (Redis) — derived, invalidated by CDC"

    # 6. Default bias: Postgres does a LOT. Prove you can't before leaving.
    return "relational/OLTP (Postgres) — extensions cover most specialist needs"


if __name__ == "__main__":
    cases = [
        Workload(QueryShape.FULLTEXT, 2_000, 20_000, False, "eventual", 200, 8_000_000),
        Workload(QueryShape.AGGREGATE, 500, 1_000, False, "eventual", 5_000, 4_000_000_000),
        Workload(QueryShape.POINT, 1_000, 200_000, False, "eventual", 1, 50_000_000),
        Workload(QueryShape.RANGE, 120_000, 5_000, False, "eventual", 50, 9_000_000_000),
        Workload(QueryShape.POINT, 3_000, 4_000, True, "strong", 20, 30_000_000),
    ]
    for c in cases:
        print(f"{c.query_shape.value:14} -> {recommend_store(c)}")
```

The output routes the first case to Elasticsearch, the billion-row aggregate to ClickHouse, the hot point read to Redis, the 120k-write time-ordered workload to Cassandra, and the transactional workload back to Postgres. The structure matters more than the exact thresholds: hard requirements (transactions, blobs) decide first, query shape routes the specialists, and the *default branch is always Postgres*. You should be able to read this function in a design review and argue about a threshold, rather than litigating someone's instinct that "Mongo feels right here."

### Second-order: the dimensions the matrix hides

The matrix is the strongest single signal, but four other dimensions decide ties and catch mistakes:

- **Read- vs write-heavy.** A 100:1 read:write ratio invites a derived cache or read-optimized projection; a write-heavy stream invites wide-column or a log. The ratio tells you where to spend.
- **Consistency need.** A derived store is *always* eventually consistent with the source of truth — there is replication lag. If a workload cannot tolerate reading its own write a few milliseconds stale, it cannot be served from a derived store; it must hit the system-of-record.
- **Scale axis.** Vertical (bigger box), read replicas (more readers), or sharding (more writers)? Most "we need to scale" problems are read-scaling problems solved with replicas or a cache, not a new engine.
- **Operability.** This is the dimension engineers systematically under-weight, and it is the subject of the next section.

## 3. The hidden cost of every new store

**Rule of thumb: a new store costs you a permanent on-call surface and a new consistency boundary, neither of which shows up in the benchmark.** The benchmark shows you the 50× speedup on the target query. It does not show you the 2 a.m. page when the Cassandra cluster's compaction falls behind, the week your one Elasticsearch expert is on vacation during a mapping change, or the quarter you spend building backup-and-restore tooling that Postgres gave you for free.

![Each store solved a real problem on the day it was added; the accruing cost is the on-call surface nobody priced in](/imgs/blogs/polyglot-persistence-choosing-the-right-store-4.webp)

The timeline above is the pattern I have watched play out at three different companies, and it is the trap that makes "polyglot" a dirty word in some shops. No single decision on that line is wrong. Year 2's Redis and Elasticsearch each solved a real, measured problem. Year 3's Cassandra absorbed a write-heavy event stream the primary genuinely could not hold. Year 4 added a warehouse and a blob store and the Kafka backbone to feed them. Each was justified *in isolation*. What nobody priced in was the integral: by Year 5 you are running nine stores, you have seven boundaries across which the "same" data is duplicated and can drift, and the operational surface — not query load — has become the thing that limits how fast you can ship.

Here is the cost ledger you should actually fill in before adding a store. Every column is a cost that the speedup benchmark conveniently omits.

| New store | On-call / ops burden | Specialized expertise | Backup / DR you must build | Consistency boundary added |
| --- | --- | --- | --- | --- |
| Redis cache | Eviction tuning, memory pressure, failover | Cluster topology, persistence modes | Usually none (it's derived) | Cache ↔ SoR (staleness, invalidation) |
| Elasticsearch | Heap/GC, shard sizing, mapping changes | Analyzers, relevance tuning, rolling upgrades | Snapshot/restore, reindex strategy | Index ↔ SoR (drift, reindex lag) |
| Cassandra | Compaction, repair, tombstones, gossip | Data modeling per query, tuning quorum | Multi-node restore, anti-entropy | If it's a 2nd SoR: split-brain risk |
| ClickHouse | Merge backpressure, part explosions | MergeTree tuning, dictionary design | Backup of large parts, schema migration | Warehouse ↔ SoR (load lag) |
| Kafka (the glue) | Partition rebalancing, consumer lag, retention | Exactly-once semantics, schema registry | Topic replication, offset recovery | It *is* the boundary mechanism |

Notice the last column. **Most of these stores hold a *copy* of data whose truth lives elsewhere.** The cache is a copy. The search index is a copy. The warehouse is a copy. Every copy is a place where, if the synchronization fails, your system now believes two contradictory things at once — and the failure is silent, because each store answers its own queries happily while disagreeing with its neighbors. This is the deepest cost of polyglot persistence, deeper than the on-call burden, and it is why the *how you sync* question matters as much as the *which store* question.

There is exactly one rule that contains this cost, and it is non-negotiable: **never have two systems-of-record for the same data.** One engine owns the truth for any given piece of data. Everything else is explicitly a derived store, and derived stores are allowed — expected — to be stale, because they can always be rebuilt from the source. The moment two stores both accept authoritative writes for the same entity, you have a distributed consensus problem you did not sign up for, and reconciliation becomes a permanent tax.

## 4. Why dual writes corrupt your data

**Rule of thumb: if your application code writes the same fact to two stores, you have already shipped a data-corruption bug; you just haven't hit it yet.** This is the single most common way polyglot architectures go wrong, and it is worth slowing down for because it looks completely reasonable in a pull request.

The pattern is innocent. You add Elasticsearch for search. Now when a user updates their profile, you write to Postgres *and* you write to Elasticsearch, right there in the request handler, so search stays fresh:

```python
# DANGEROUS: dual write. Looks fine in review. Corrupts data in production.
def update_profile(user_id: int, fields: dict) -> None:
    db.execute("UPDATE users SET ... WHERE id = %s", ...)   # write #1
    es.index(index="users", id=user_id, document=fields)    # write #2
```

![Two independent writes can half-fail and leave the index permanently wrong; one write plus a replayed log cannot](/imgs/blogs/polyglot-persistence-choosing-the-right-store-5.webp)

The before/after above is the whole argument. On the left, two independent writes. The Postgres write commits. Then the Elasticsearch call times out — a GC pause, a network blip, a node restart, a rolling upgrade, a deploy that bounced the connection. The function raises, the user sees an error, but Postgres is *already committed*. There is no transaction spanning the two systems; you cannot roll back the Postgres write. The search index is now permanently wrong for that user, and nothing will ever fix it, because the only code that knew how to update the index already ran and failed. Worse, reorder the writes and you get the opposite corruption: the index is updated, then the DB write fails, and now search shows a profile the database says doesn't exist.

You cannot fix this with a try/except, because the failure window is *between* two non-atomic operations and either side can be the one that survives. You cannot fix it with a retry, because the request that would retry has already returned an error to the user and moved on. You cannot fix it with a distributed transaction across Postgres and Elasticsearch, because they do not share a transaction coordinator and two-phase commit across heterogeneous stores is operationally radioactive. The dual write is broken at the level of *what is possible*, not at the level of how carefully you code it.

The right side of the figure is the fix, and it is structural: the application makes exactly *one* write, to Postgres. It commits or it doesn't — a single atomic operation with no partial state. *After* the commit, change data capture reads the committed write-ahead log and replays it into the index. If the indexer is down, the change waits in the log and is applied when it recovers. If the indexer crashes mid-batch, it re-reads from its last committed offset and re-applies — idempotently, keyed by primary key — with no harm done. The index can always catch up to the truth, because the truth is a durable, ordered, replayable log and the index is just a fold over it.

> Two writes can disagree. One write and a replay cannot. That single sentence is the entire architectural difference between a polyglot system that drifts and one that converges.

## 5. The pattern: one system-of-record, many derived stores

**Rule of thumb: pick one engine as the system-of-record, make every other store a function of its change stream, and you have turned a consistency nightmare into a replication-lag number you can monitor.** This is the pattern that makes polyglot persistence safe at scale, and it is worth building correctly because everything downstream depends on it.

![The write-ahead log becomes an ordered, keyed event stream that every derived store replays idempotently](/imgs/blogs/polyglot-persistence-choosing-the-right-store-6.webp)

The pipeline above is change data capture in concrete terms. A transaction commits in Postgres — which means it is durably in the write-ahead log. A logical replication slot exposes that log as a stream of row-level change events. Debezium reads the slot and publishes each change to a Kafka topic, *keyed by primary key* so that all changes to one row land in one partition and stay ordered. Each derived store runs a consumer that polls the topic and applies changes idempotently — upsert by primary key, guarded by a version or log-sequence-number so a replayed event never moves the row backward. When the consumer is healthy, the derived store trails the source of truth by milliseconds. When it falls behind, you have a single number — consumer lag — that tells you exactly how stale each store is.

Setting up the source side is a few lines of Postgres configuration. Logical decoding has to be enabled and a replication slot created:

```sql
-- postgresql.conf: enable logical decoding (requires restart)
-- wal_level = logical
-- max_replication_slots = 8
-- max_wal_senders = 8

-- Create a publication for the tables you want to stream.
CREATE PUBLICATION cdc_pub FOR TABLE users, orders, products;

-- Debezium creates its own replication slot, but you can see them here:
SELECT slot_name, plugin, active, restart_lsn
FROM pg_replication_slots;
```

Debezium itself is configuration, not code — a connector registered against Kafka Connect that watches the publication and streams every change:

```json
{
  "name": "postgres-cdc-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres.internal",
    "database.dbname": "app",
    "plugin.name": "pgoutput",
    "publication.name": "cdc_pub",
    "slot.name": "debezium_slot",
    "topic.prefix": "app",
    "table.include.list": "public.users,public.orders,public.products",
    "tombstones.on.delete": "true",
    "snapshot.mode": "initial"
  }
}
```

The consumer is where you must be disciplined, because *idempotency is the property that makes replay safe*. The consumer cannot assume each message arrives exactly once — Kafka and Debezium give you at-least-once delivery, so the same change can be re-delivered after a crash or rebalance. The fix is to make applying a change a function of the change alone, keyed by primary key and gated by a monotonic version, so re-applying is a no-op:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://es.internal:9200")


def apply_change(event: dict) -> None:
    """Apply one Debezium change event idempotently to the search index.

    Debezium delivers at-least-once, so this MUST be safe to replay. We key
    by primary key and gate every write on the source log-sequence-number
    (LSN) so a re-delivered or out-of-order event never regresses the index.
    """
    op = event["op"]                      # 'c' create, 'u' update, 'd' delete, 'r' snapshot
    after = event.get("after")
    before = event.get("before")
    lsn = event["source"]["lsn"]          # monotonic per source row

    if op == "d":
        # Idempotent delete: gone-twice is still gone.
        es.delete(index="users", id=before["id"], ignore=[404])
        return

    doc = {**after, "_lsn": lsn}
    # Optimistic concurrency: refuse to apply an event older than what we have.
    # Re-delivery of the same or an older LSN becomes a harmless no-op.
    es.update(
        index="users",
        id=after["id"],
        body={
            "scripted_upsert": True,
            "script": {
                "source": "if (ctx._source._lsn == null || ctx._source._lsn < params.doc._lsn) "
                          "{ ctx._source = params.doc } else { ctx.op = 'noop' }",
                "params": {"doc": doc},
            },
            "upsert": doc,
        },
    )
```

Three properties fall out of this design, and they are exactly the properties dual writes lacked. **Convergence:** because every derived store is a deterministic fold over the same ordered log, replaying the log from any point produces the correct end state — a store that fell behind catches up, and a store that got corrupted can be rebuilt by replaying from a snapshot. **Decoupling:** the source of truth does not know or care how many derived stores exist or whether they are healthy; you can add a new consumer (say, the ClickHouse warehouse) months later and backfill it from the log without touching the write path. **Observability:** "how stale is search?" stops being an unanswerable question and becomes consumer lag in milliseconds, on a dashboard, with an alert.

For teams not ready to run Kafka and Debezium, the [outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) is the lighter-weight cousin: write the change to an `outbox` table *in the same transaction* as the business write, then a poller reads the outbox and publishes. It gives you the same atomicity guarantee — one transaction, no dual write — without standing up a CDC platform, at the cost of a polling loop instead of true log streaming.

### Second-order: ordering, deletes, and backfill

Three things bite teams here. **Ordering** must be per-key, not global — keying the Kafka topic by primary key guarantees that two updates to the *same* row stay ordered, which is all you need; demanding global ordering across all rows would force a single partition and kill throughput. **Deletes** are the most-forgotten case: a naive "index the row" consumer never removes anything, so deleted records linger in search forever; Debezium's tombstone events (`tombstones.on.delete`) exist precisely so the consumer can propagate deletions. **Backfill** is how you bootstrap a new derived store: Debezium's `snapshot.mode: initial` reads the entire table once, then switches to streaming, so a brand-new ClickHouse warehouse gets the full history before it starts tailing live changes.

## 6. Just use Postgres until you provably can't

**Rule of thumb: Postgres does far more than most engineers realize, and "just use Postgres" beats premature polyglot for the overwhelming majority of teams and workloads.** Before you graduate a workload to a specialist, check whether the default already covers it — because each store you *don't* add is a consistency boundary you never have to defend.

![One engine with extensions covers document, search, geo, fuzzy, time-series, queue, KV, and light-analytics workloads](/imgs/blogs/polyglot-persistence-choosing-the-right-store-7.webp)

The capability map above is Postgres absorbing eight workloads that engineers routinely (and prematurely) split out to a specialist. `JSONB` with a `GIN` index gives you a perfectly serviceable document store. `tsvector` plus `GIN` gives you real full-text search with ranking. PostGIS is among the best geospatial engines in existence, full stop. `pg_trgm` does trigram fuzzy matching and typo-tolerant search. TimescaleDB turns Postgres into a competitive time-series database. `SELECT ... FOR UPDATE SKIP LOCKED` makes a robust job queue. Arrays and `hstore` cover simple key-value needs. Materialized views handle light analytics. One query can lean on several at once:

```sql
-- Full-text search + JSONB filter + trigram fuzzy match + geo radius,
-- all in one Postgres query, no second store required.
SELECT id, profile->>'name' AS name,
       ts_rank(search_vec, query) AS rank
FROM users,
     to_tsquery('english', 'engineer & remote') AS query
WHERE search_vec @@ query                                  -- full-text (tsvector + GIN)
  AND profile->>'country' = 'VN'                           -- JSONB field filter
  AND profile->>'name' % 'Hieip Tran'                      -- pg_trgm fuzzy (typo-tolerant)
  AND ST_DWithin(location, ST_MakePoint(106.7, 10.8)::geography, 50000)  -- PostGIS, 50km
ORDER BY rank DESC
LIMIT 20;
```

That single query does relevance-ranked text search, a document-field filter, fuzzy name matching, and a geospatial radius query — four "you need a specialized store" workloads — in the engine you already run, inside one transaction, with one backup, on one on-call rotation. The point is not that Postgres beats Elasticsearch on full-text at scale; it doesn't, past tens of millions of documents with heavy relevance tuning. The point is that the *threshold* where you genuinely need Elasticsearch is far higher than the threshold at which teams reach for it, and everything below that threshold is free consistency you get by staying in one store.

So the honest sequence is: build on Postgres. When a workload gets slow, first check whether an index, an extension, or a read replica fixes it — they usually do. Only when you can *show*, with a profile and a projection, that the working set will not fit in any single machine's RAM, or that write volume exceeds what one primary can ever sustain, or that the access pattern is fundamentally wrong for a relational engine, have you earned the right to add a store. And when you do, wire it up as a derived store fed by CDC, never as a second source of truth. This is the same discipline the [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) applies to scaling within the relational model: exhaust the cheap, single-store levers before reaching for the expensive, multi-store ones.

## Case studies from production

### 1. Stack Overflow: a top-100 site on a handful of servers

Stack Overflow is the canonical proof that you do not need nine databases to run at enormous scale. For years they served one of the busiest sites on the internet from a strikingly small fleet: a primary SQL Server cluster as the system-of-record, Redis for caching and counters, and Elasticsearch for the search index — and the entire database tier ran on a number of physical servers you could count on two hands. The architecture is textbook polyglot *discipline*: SQL Server owns the truth, Redis and Elasticsearch are derived stores, and the team resisted the urge to split every workload onto its own engine. The lesson engineers take away is the wrong one if they hear "they used three stores"; the right one is "they used exactly three, each for a workload that earned it, and they kept the system-of-record singular and boring." Most teams running a tenth of that traffic somehow run twice as many stores. The constraint that kept Stack Overflow lean was not technical — it was the deliberate refusal to add a store before a workload proved it needed one.

### 2. Discord: rightsizing the message store to its dominant workload

Discord's message history is a write-heavy, time-ordered, known-query workload — fetch messages for a channel, in time order — at a scale of trillions of messages. They started on MongoDB, hit the wall where the working set no longer fit in RAM and read latency became unpredictable, and migrated to Cassandra, whose partition-key-plus-clustering model is built for exactly "give me this channel's messages in this time range." Years later they migrated again, from Cassandra to ScyllaDB, to escape JVM garbage-collection tail latencies and compaction pain while keeping the same data model. The throughline is that they matched the *store to the dominant access pattern* of one specific, enormous workload — they did not try to make one engine serve messages and everything else. Each migration was driven by a measured failure of the current store on that workload's particular shape, not by chasing a trend. This is polyglot persistence done as engineering: a specialized, write-optimized, time-ordered store for the one workload that justified it.

### 3. Facebook TAO: a derived read store over a relational source of truth

Facebook's social graph reads — "who are this user's friends, what did they like, who commented" — are point and short-traversal lookups at a scale no relational primary could serve directly. Their answer was TAO, a graph-aware, geographically-distributed read-through store layered over MySQL, with memcache underneath. The crucial architectural choice: MySQL remained the system-of-record. TAO is a *derived* store optimized for the graph read pattern, and writes flow through TAO to MySQL and then invalidate the cached views — there is one source of truth, and the fast read layer is explicitly a projection of it that can be rebuilt. This is the mental-model figure from the top of this post at Facebook scale: a relational system-of-record, a specialized derived store for the dominant read shape, and an invalidation/replication path that keeps the derived store converging on the truth rather than diverging from it.

### 4. Netflix: genuinely polyglot, with discipline

Netflix is the rare shop that is *legitimately* polyglot across many stores — Cassandra for high-write durable data, EVCache (their [multi-region Memcached layer](/blog/software-development/database-scaling/netflix-evcache-multi-region-cache)) for caching, Elasticsearch for search and operational logs, S3 for media and data-lake files — and it works because each store owns a clearly-bounded workload and the data flows are explicit. They did not accrete stores by accident; they run a data platform whose job is precisely to move data between systems-of-record and derived stores reliably, with event pipelines doing the synchronization that a hand-rolled dual write never could. Netflix is the counterexample to "polyglot is always premature": at their scale, with their workload diversity and a dedicated platform team, many specialized stores genuinely beat one generalist. The cost — a team whose full-time job is the plumbing between stores — is real and is exactly the cost most companies cannot afford, which is why most companies should not copy the Netflix store count.

### 5. The search index that quietly drifted

This is a composite of an incident I have seen in some form at more than one company, because the dual-write trap is that common. A team added Elasticsearch for product search and wired it as a dual write: update the product in the database, then update it in the index, in the same request handler. It worked in testing and for months in production. Then during a routine rolling upgrade of the Elasticsearch cluster, a fraction of index writes timed out while the database writes all succeeded. No errors surged loudly enough to page anyone — the failures were a small percentage, and the handler's error was swallowed by a retry that re-ran the database write but not the index. Over weeks, the index drifted: discontinued products still appeared in search, price changes lagged, some new products never got indexed at all. The bug surfaced only when a customer complained that a product in search results 404'd at checkout. The fix was the structural one: rip out the dual write, stand up Debezium against the database, and reindex everything from the change stream — after which "how stale is search" became a consumer-lag metric instead of a mystery. The lesson is that dual-write corruption is *silent and cumulative*; you do not get a clean failure, you get slow rot.

### 6. The premature-polyglot regret wave

Around the mid-2010s, a wave of teams adopted document and wide-column stores — MongoDB and Cassandra most often — for the "web scale" they were sure was coming, and a notable number of them publicly migrated *back* to Postgres a year or two later. The pattern in those postmortems was remarkably consistent: the scale that justified the specialized store never arrived, the team lost relational features (joins, transactions, ad-hoc queries) they turned out to need constantly, and they discovered that operating the new store — backups, upgrades, data modeling around a single query shape — cost far more engineering time than the original "slow query" ever would have. They had paid the full price of polyglot persistence and received almost none of the benefit, because the workload had not actually outgrown the default. The lesson is the cost ledger from earlier: the speedup benchmark is real but partial, and for a workload that has not crossed the specialist's threshold, the operational and consistency costs dominate. "We'll need it at scale" is not a workload; it is a guess.

### 7. Uber: specialized stores, consolidated deliberately

Uber's data needs span transactional trips, geospatial indexing, and large-scale analytics, and they have run a [geo-distributed data architecture](/blog/software-development/database-scaling/uber-geo-distributed-data) with several specialized stores — including Schemaless, a key-value layer built *on top of* MySQL, and dedicated systems for geospatial and analytical workloads. What is instructive is not that they used multiple stores — at their scale they must — but that they built Schemaless on a boring, well-understood engine (MySQL) rather than adopting yet another new database, and consolidated their storage primitives onto a small number of patterns their teams could actually operate. The discipline shows up as a deliberate limit on the *variety* of stores, even while the *number* of clusters is large: many instances of a few well-understood engines beats a few instances of many exotic ones, because operational expertise does not scale across engine types the way capacity scales across nodes.

## When to reach for a specialized store, and when not to

Reach for a specialized store when:

- A specific, measured workload has a query shape — full-text, large-scale aggregate, deep graph traversal, very-high-write time-ordered — that the relational engine serves 10–100× worse, and you have confirmed an index or extension does not close the gap.
- You can name the system-of-record and confirm the new store will be *derived* from it, fed by CDC or an outbox, never accepting authoritative writes for the same data.
- You have an owner for the new store's operations: backups, upgrades, capacity, and the 2 a.m. page. If nobody owns it, you are adding a liability, not a capability.
- The working set provably will not fit in a single machine's RAM, or write volume provably exceeds what one primary can sustain — and you can show the projection, not just assert it.

Skip the specialized store — stay on Postgres or your existing system-of-record — when:

- The slow query has an indexing, extension, or read-replica fix you have not tried. This is the overwhelmingly common case.
- You are adding the store for scale you anticipate but cannot yet demonstrate. "We'll need it eventually" justifies a design that *can* graduate later (clean boundaries, CDC-ready), not the store itself today.
- The workload needs transactions or joins against data the specialized store would hold — those are precisely what you lose, and re-implementing them in the application is a worse version of what the database already does.
- You cannot yet articulate which engine owns the truth and how the new store stays in sync. If the answer is "we'll dual-write," stop: you are about to ship the silent-corruption bug from case study 5.

The whole discipline reduces to two sentences. One database is a great default and stays right far longer than you expect, so prove you have outgrown it before you leave. And when you do reach for a specialist, keep exactly one system-of-record and make every other store a derived, replayable function of its change stream — because two writes can disagree, and one write plus a replay cannot.

## Further reading

- [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) — the deeper dive on why a columnar engine beats a row store 50–100× on analytics, and where the line between transactional and analytical workloads actually falls.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the mechanism that makes the system-of-record-plus-derived-stores pattern in this post safe, including the lighter-weight outbox alternative to running Debezium.
- [Redis applications and optimization](/blog/software-development/database/redis-applications-and-optimization) — how to run the most common derived store well: eviction, persistence, and the access patterns where a KV cache earns its keep.
- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — the companion framework for scaling *within* the relational model before you go polyglot: replication, partitioning, and sharding in order of cost.

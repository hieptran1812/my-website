---
title: "Choosing a Datastore: SQL vs NoSQL vs NewSQL by Access Pattern"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Stop arguing SQL versus NoSQL — learn the senior's framework for picking a datastore by access pattern, consistency need, and scale, and know exactly when to leave Postgres."
tags:
  [
    "system-design",
    "databases",
    "nosql",
    "newsql",
    "data-modeling",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-1.webp"
---

The single worst datastore decision I have watched a team make did not involve choosing the wrong database. It involved choosing a database at all, when they should have chosen a column. The team was building a B2B analytics product, they had read that "Postgres doesn't scale," and so before they had a single paying customer they stood up a Cassandra cluster, a separate Elasticsearch cluster, a Redis tier, and a Kafka pipeline to keep all three in sync. Eighteen months later they had four distributed systems to operate, a five-person team, roughly two thousand rows of data per customer, and a feature velocity that had ground to nothing because every product change now required touching three datastores and reasoning about eventual consistency between them. The entire dataset would have fit, comfortably, in the RAM of a laptop. They had paid the operational cost of planet-scale before they had the data of a spreadsheet.

That story is the cautionary tale that frames this whole post, and the lesson is the one most "SQL vs NoSQL" debates miss entirely: the axis that actually decides a datastore is almost never "SQL or NoSQL." Those are query *languages* and loosely a *consistency posture*, not a design axis. The real axes are three: what is the **dominant access pattern** (how do you read and write the data — by key, by range, by join, by full-text, by graph traversal?), what is the **consistency need** of each operation (does a stale read cost money or merely a slightly stale like count?), and what is the **scale** you actually have, today and in the credible next two years (not the scale on the conference slide). Get those three right and the family of datastore falls out almost mechanically — figure 1 is that decision tree, and the rest of this post is teaching you to run it.

By the end you should be able to do the thing the senior does in a design review: look at a workload, name its dominant access pattern out loud, state which datastore family that pattern is built for and what that family gives up, defend the "it fits in one Postgres" default against premature distribution, and — when you genuinely have outgrown a single node — describe the migration path to a sharded or NewSQL system without hand-waving. We will build a big trade-off matrix, run four concrete worked examples (a shopping cart, a product catalog with search, a metrics pipeline, a social graph), do the capacity-and-cost math for one workload across two stores, and stress-test the single-Postgres default until it breaks so you know the exact failure mode that should make you leave it.

A note on scope, because this series has a deliberate division of labor. The `database/` folder on this blog already deep-dives the *mechanisms*: how a [B-tree index actually works](/blog/software-development/database/b-trees-how-database-indexes-work), how [LSM-trees trade write amplification for write throughput](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), how [partitioning and sharding move bytes across nodes](/blog/software-development/database/database-partitioning-and-sharding), and how [CockroachDB](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive), [Spanner](/blog/software-development/database/spanner-truetime-and-external-consistency), and [Vitess](/blog/software-development/database/vitess-sharding-mysql-at-scale) implement distributed SQL. This post is the *decision layer* on top of those: not how an LSM-tree works, but why a write-heavy time-series workload should pick a store built on one. When a mechanism matters, I will explain it briefly and link the deep-dive rather than re-derive it. For the storage-engine substrate underneath these families, the sibling post on [storage engines: B-trees vs LSM-trees for architects](/blog/software-development/system-design/storage-engines-btrees-vs-lsm-trees-for-architects) is the companion to read first.

## 1. The wrong axis, and the right three

Walk into a design review and say "we're using NoSQL" and you have communicated almost nothing. NoSQL is a marketing term coined for a meetup; it groups together databases whose only shared property is "not a traditional relational database," which lumps Redis (an in-memory key-value store), Cassandra (a write-optimized wide-column store), MongoDB (a document store), Neo4j (a graph database), and Elasticsearch (a search engine) into one bucket. Those five share essentially no design decisions. They have wildly different consistency models, query capabilities, and operational profiles. Telling me you picked "NoSQL" is like telling me you bought "a vehicle" — it could be a bicycle or a cargo ship, and the difference is the entire decision.

![A decision tree that routes a workload from its dominant access pattern down to the datastore family built for that query shape, from key-value to relational to wide-column to NewSQL](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-1.webp)

So throw out the SQL/NoSQL axis as a *first* question. It re-enters later as a consequence, not a cause. Figure 1 shows the axis that actually leads somewhere: the **dominant access pattern**. Start at the root and ask, what is the single most frequent and most latency-sensitive way this data is read and written? Not the rare admin query, not the nightly report — the query that runs ten thousand times a second on the hot path. Everything follows from that. If the dominant access is "fetch one object by its primary key, as fast as possible," you want a key-value store and you do not care that it can't do joins, because you are never going to join. If the dominant access is "answer arbitrary questions that span several entities with transactional guarantees," you want a relational database and you will gladly pay for the join machinery because you are going to use it constantly.

The three axes, stated as the questions a senior actually asks:

**Axis 1 — Access pattern (the shape of your dominant query).** Point lookup by key? Range scan over a sorted dimension (time, score, alphabetical)? Multi-entity join with filters? Full-text or faceted search? Graph traversal across many hops? Each of these is served *well* by a different family and *badly* by the others. Picking the family whose native operation matches your dominant query is the single highest-leverage decision, because it turns your hot query from a full scan into an index hit. This is the optimization thesis of the whole post, and we will return to it.

**Axis 2 — Consistency need (per operation, not per database).** Does a stale or lost write cost money, violate a contract, or corrupt an invariant — or is it merely cosmetic? A bank balance and a "number of likes" counter live at opposite ends of this axis, and they might live in the same product. The senior move, which the [CAP and PACELC trade-off post](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) develops in depth, is to classify the *operation*, not the *product*. Most workloads are a mix: a handful of operations need strict consistency and the vast majority tolerate eventual or read-your-writes consistency. The forthcoming [practical consistency-models guide](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) is the companion that turns this axis into a checklist.

**Axis 3 — Scale (real, near-term, honest).** How much data, how many reads per second, how many writes per second — today, and on a credible 18-to-24-month projection? Be honest here. A modern single Postgres node on commodity cloud hardware comfortably serves tens of thousands of transactions per second and stores multiple terabytes. The number of teams that genuinely exceed that is far smaller than the number who *believe* they will. We will quantify this ceiling precisely in section 8, because "when do I actually leave Postgres" is the most consequential and most-fumbled question in this whole space.

Notice what these three axes do together: they *eliminate* most options before you ever compare brand names. If your dominant access is a point lookup, your consistency need is eventual, and your scale is a firehose of writes, you have already narrowed to key-value and wide-column before anyone says the word "Cassandra." The brand comparison is the last 10% of the decision. Getting the family right is the first 90%, and it is the part teams skip.

## 2. The families, by abstraction layer

Before we run the decision tree on real workloads, you need a working map of the families and what native operation each is built around. Figure 2 — the big trade-off matrix — is the reference you will come back to, and figure 5 stacks the families by abstraction so you can see how they relate. Let me walk each family with the one thing it does better than anyone and the one thing it gives up. I will keep the mechanism explanations short and link the deep-dives.

![A trade-off matrix scoring relational, key-value, wide-column, document, and NewSQL families across strong transactions, flexible schema, write volume, range scans, horizontal scale, and operational cost](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-2.webp)

**Relational (Postgres, MySQL).** Native operation: arbitrary queries over normalized tables with joins, secondary indexes, and ACID transactions. This is the most *general-purpose* family and should be your default, full stop. A relational database lets you ask questions you did not anticipate at schema-design time, enforce invariants with constraints and foreign keys, and wrap multi-row changes in a transaction that either fully commits or fully rolls back. What it gives up: horizontal write scale. A single relational node has one writer (or, with synchronous replicas, one writer plus standbys), and while you can scale *reads* with replicas almost indefinitely, scaling *writes* past one machine means sharding, which the relational model does not do for you natively. That ceiling is high — higher than most teams reach — but it is real, and section 8 maps it.

**Key-value (Redis, DynamoDB).** Native operation: `GET(key)` and `PUT(key, value)`, both O(1) and both blisteringly fast. If your dominant access is "I have an id, give me the object," nothing beats a key-value store: a Redis `GET` is a sub-millisecond in-memory hash lookup, and DynamoDB serves single-digit-millisecond reads at any scale because it partitions by hash of the key and adds nodes transparently. What it gives up: everything that is not a key lookup. No joins, no ad-hoc queries, weak or no secondary indexes (DynamoDB bolts on global secondary indexes at extra cost and with eventual consistency; Redis has none natively), and you must know your access patterns *up front* because the data is modeled around the keys you will look up. This is "schema-on-read" taken to its logical end: the store does not understand your data, only your keys.

**Wide-column (Cassandra, Bigtable, HBase).** Native operation: writes at enormous volume, and reads of contiguous ranges of columns within a partition, sorted by a clustering key. Wide-column stores are built on [LSM-trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), which turn random writes into sequential appends, so they ingest writes far faster than a B-tree-based relational store. They scale horizontally by design (consistent hashing distributes partitions across nodes; the [consistent-hashing deep-dive](/blog/software-development/database/consistent-hashing-and-data-partitioning) covers the mechanism) and stay available during node loss. What they give up: you must design the table around exactly the query you will run, because a Cassandra query that does not hit the partition key requires a full-cluster scan that the database will refuse by default. No joins, no ad-hoc queries, eventual consistency by default (tunable to quorum), and a notorious tendency to footgun teams who model it like a relational database. It is a write firehose with a very specific read shape, not a general database.

**Document (MongoDB, DynamoDB's document mode).** Native operation: store and retrieve a nested JSON-ish document by id, with secondary indexes on fields inside the document. The document model fits data that is naturally hierarchical and read as a unit — a product with its variants and reviews embedded, a user with their preferences — because you fetch the whole document in one read instead of joining five tables. Schema is flexible: different documents in a collection can have different fields, which is genuinely useful when your schema is still moving. What it gives up: transactions across documents historically (modern MongoDB has multi-document transactions but they are not the happy path and carry a cost), and the discipline that a rigid schema enforces. Flexible schema is a loaded gun: it is wonderful when your schema is evolving and a slow-motion disaster when five services each write subtly different shapes into the same collection and nobody can tell what a document is supposed to contain.

**Search (Elasticsearch, OpenSearch).** Native operation: full-text and faceted search over an inverted index. If your dominant access is "find documents matching these words, ranked by relevance, filtered by these facets," a search engine is purpose-built and a relational `LIKE '%term%'` is a full table scan that will never be fast. What it gives up: it is not a system of record. Elasticsearch is near-real-time (writes are visible after a refresh, typically a second), its consistency guarantees are weak, and you do not want it to be the authoritative copy of your data. The senior pattern is always "Postgres is the source of truth; Elasticsearch is a denormalized read-optimized projection of it, kept in sync with [change-data-capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern)."

**Graph (Neo4j, and graph layers like Amazon Neptune).** Native operation: traverse relationships — "friends of friends," "shortest path," "all accounts within three hops of this flagged one." In a relational database, a three-hop traversal is three self-joins and the cost explodes with depth; in a graph database, a traversal follows pointers and the cost scales with the size of the *result*, not the size of the *graph*. What it gives up: it is a specialist. Graph databases are operationally less mature, harder to shard, and overkill unless traversal is genuinely your dominant access. Most "social" features (a feed, a follower count) are *not* graph workloads — they are key-value and range-scan workloads. Reach for a graph database only when the *traversal itself* is the product.

**Time-series (InfluxDB, TimescaleDB).** Native operation: append timestamped points at high volume and query ranges over time with downsampling and rollups. Metrics, IoT sensor data, and financial ticks are append-heavy, time-ordered, and almost never updated — a shape that general databases handle badly and time-series databases optimize for with time-partitioned storage and automatic compression. TimescaleDB is especially pragmatic: it is a Postgres extension, so you keep SQL, joins, and the relational ecosystem while getting time-series storage. What it gives up: it is specialized for the time dimension; it is not where your user accounts or your orders live.

**NewSQL (Spanner, CockroachDB, Vitess, YugabyteDB).** Native operation: relational SQL with ACID transactions, but *horizontally scalable* across many nodes and regions. This family exists precisely to break the relational write ceiling without giving up SQL or strong consistency. Spanner uses [TrueTime](/blog/software-development/database/spanner-truetime-and-external-consistency) and synchronized clocks to give externally consistent transactions across continents; [CockroachDB](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive) gives serializable transactions on commodity hardware without atomic clocks; [Vitess](/blog/software-development/database/vitess-sharding-mysql-at-scale) shards MySQL underneath a layer that keeps it looking like one database. What they give up: latency and operational complexity. A cross-shard or cross-region transaction in a NewSQL system pays consensus round-trips that a single-node Postgres commit never pays — often tens of milliseconds where Postgres pays single-digit milliseconds — and the operational burden is meaningfully higher. NewSQL is the answer when you have *genuinely* outgrown single-node relational and still need SQL and strong consistency. It is not a free upgrade; section 7 is about when the latency tax is worth it.

## 3. The "fits in one Postgres" default (and why teams leave too early)

Here is the opinion that should anchor every datastore decision you make for the next five years: **the default is one Postgres, and the burden of proof is on leaving it.** Not on staying. Most teams have the burden backwards — they treat distributing the data as the grown-up, scalable, default-good choice and staying single-node as the naive thing you do until you "graduate." That is exactly inverted, and it is the single most expensive misconception in this whole domain.

![A before-and-after comparison contrasting a premature four-table DynamoDB design that rebuilds joins in application code against a Postgres-first design that splits out only the hot table when needed](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-3.webp)

Figure 3 is the comparison that should make you uncomfortable if you have ever reached for NoSQL early. On the left is the premature-NoSQL design: a team splits its data across four DynamoDB tables, discovers that the application needs to relate them, and so rebuilds joins in application code — fetching from one table, looping, fetching from another, stitching results together in the service. They have re-implemented, badly and without transactions, the exact thing the relational database does for free and correctly. Every one of those hand-rolled joins is a place where a partial failure leaves inconsistent data, because there is no transaction wrapping them. On the right is Postgres-first: one node, joins and ACID for free, and when *one specific table* genuinely becomes a bottleneck, you split *that table* out — not the whole architecture.

Why do teams leave too early? Three reasons, all of them avoidable.

**Reason one: they believe "Postgres doesn't scale," which is folklore.** A single Postgres instance on a modern cloud box — say 32 vCPUs and 128 GB of RAM — serves tens of thousands of transactions per second and stores several terabytes on fast SSDs. With read replicas it serves hundreds of thousands of reads per second. Companies you have heard of ran on a single primary Postgres or MySQL far longer than you would guess; the well-documented pattern at many scale-ups is that the database was a single primary with replicas until they were processing genuinely enormous volume. The ceiling is high. Most products die of no users long before they hit it.

**Reason two: they optimize for an imagined future instead of the measured present.** Think of premature distribution as the database equivalent of premature optimization — you pay a real, large, today cost (operational complexity, lost joins and transactions, slower development) to hedge against a scale problem you do not have and statistically will not have. That figure-3 left column is the cost you pay *now*; the scale problem it hedges is a *maybe-later*. The senior move is to design so that *if* you hit the ceiling, the migration is tractable (we will cover how), and otherwise spend your complexity budget on the product.

**Reason three: resume-driven development.** Cassandra and Kafka look better on a CV than "we used Postgres well." This is real and you should name it when you see it in a review, kindly but plainly. The most senior thing you can do is choose the boring tool and spend the saved complexity on the customer.

When *should* you leave? Specifically, when one of these is true and you have measured it, not imagined it: your *write* throughput approaches the single-node ceiling and read replicas can't help because the bottleneck is writes; your dataset exceeds what one node can hold even after archiving cold data; your *latency* requirement demands data physically near users in multiple regions (a geo-distribution problem a single primary cannot solve); or one specific access pattern (search, time-series, graph) is so dominant and so badly served by relational that a specialized store is warranted *for that pattern*. Notice that the first three are *scale* triggers and the last is an *access-pattern* trigger — and that even the access-pattern trigger usually means *adding* a specialized store alongside Postgres, not replacing it. Section 8 puts hard numbers on the scale triggers.

## 4. The optimization lens: make the dominant query an index hit, not a scan

Now the thread that runs through every good datastore decision, stated as an optimization principle: **pick the store and model the data so that your dominant query is an index hit or an O(1) lookup, never a scan.** This is the senior's actual mental model for performance, and it reframes datastore choice as a query-cost-minimization problem.

![A layered stack of datastore families ordered by query power, from key-value at the base up through wide-column, document, relational, graph, and search at the top](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-5.webp)

Figure 5 stacks the families by how much query power they layer on top of raw storage, and the optimization insight reads right off it: as you climb the stack you gain query power but the engine has to do more work per query, so the *fastest possible* read of a given object is a key-value `GET`, and every richer query is the engine doing extra work on your behalf. The optimization game is to make sure the engine's extra work lands on an *index*, not a *scan*. A query that hits a B-tree index is O(log n) — for a billion rows, about 30 comparisons. A query that scans the table is O(n) — for a billion rows, a billion row-reads. That is not a 2× difference; it is a thirty-million-× difference, and it is the entire reason index design matters more than database choice for read performance.

This is where datastore choice and data modeling fuse. The right store gives your dominant query a *native* fast path; the wrong store forces it into a scan no matter how you tune. Three concrete instances:

- **Time-ordered reads.** "Give me the last 100 events for this user, newest first." In Postgres this is a B-tree index on `(user_id, created_at DESC)` and the read is a single index range scan — fast. In a wide-column store it is the *native shape*: partition by `user_id`, cluster by `created_at`, and the read is a contiguous disk scan within one partition. In a plain key-value store it is *impossible* without a separate index you maintain yourself. Same query, three very different costs depending on whether the store's native operation matches.

- **Search.** "Find products matching 'wireless noise-cancelling headphones under \$200' ranked by relevance." In Postgres, `WHERE name LIKE '%wireless%'` is a full table scan, and even full-text search with a GIN index struggles with relevance ranking and faceting at scale. In Elasticsearch, this is the native operation: an inverted index lookup that is fast regardless of catalog size. The optimization is not "tune the Postgres query" — it is "this access pattern belongs in a different engine."

- **Counters and rate limits.** "How many times has this key been hit in the last minute?" In Redis this is `INCR` plus a TTL — O(1), in memory, sub-millisecond. In Postgres it is a row update with lock contention on a hot row that caps you at a few hundred writes per second per row until you shard the counter. The optimization here is store choice: a hot-counter access pattern belongs in an in-memory key-value store.

Here is the difference made concrete in Postgres, because the gap between a scan and an index hit is the single most measurable thing in this whole post. Run `EXPLAIN ANALYZE` on the same query with and without the right index and the planner tells you exactly which path it took:

```sql
-- 50 million events. The dominant read: latest events for one user.
CREATE TABLE events (
  id          bigint GENERATED ALWAYS AS IDENTITY,
  user_id     bigint NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  payload     jsonb
);

-- Without a matching index, this is a Seq Scan over 50M rows:
EXPLAIN ANALYZE
SELECT * FROM events
WHERE user_id = 4242
ORDER BY created_at DESC
LIMIT 100;
-- Seq Scan on events  (cost=0..1.1M rows=50000000)
-- actual time=812 ms ... reads every row, sorts, takes the top 100

-- The composite index that makes the dominant query a range scan:
CREATE INDEX idx_events_user_time
  ON events (user_id, created_at DESC);

EXPLAIN ANALYZE
SELECT * FROM events
WHERE user_id = 4242
ORDER BY created_at DESC
LIMIT 100;
-- Index Scan using idx_events_user_time  (rows=100)
-- actual time=0.4 ms ... walks the B-tree to user 4242, reads 100 sorted rows
```

That is an 800 ms scan collapsing to a sub-millisecond index hit on the *same* data in the *same* database — a 2000× win from one `CREATE INDEX`, and the planner proves it in the plan output. The lesson generalizes: before you conclude you need a different database, confirm you are not just missing an index, because most "Postgres is slow" complaints are an unindexed dominant query, not a real ceiling. The store change is warranted only when *even the right index* leaves you with a scan — which is exactly the case for full-text search (no B-tree helps `LIKE '%term%'`) and for write throughput past the single-writer wall.

For the access patterns that *do* warrant a different engine, the win shows up the same way in that engine's native idiom. A hot counter in Redis is the canonical example — the rate-limit access pattern that would lock a Postgres row becomes an atomic in-memory operation:

```python
import time
import redis
r = redis.Redis()

def allow_request(user_id: str, limit: int = 100, window_s: int = 60) -> bool:
    # Sliding-window counter: one INCR + one EXPIRE, both O(1), in memory.
    key = f"rl:{user_id}:{int(time.time()) // window_s}"
    pipe = r.pipeline()
    pipe.incr(key)          # atomic, no row lock, sub-millisecond
    pipe.expire(key, window_s)
    count, _ = pipe.execute()
    return count <= limit   # 50k+ checks/s on one node, p99 < 1ms
```

The same counter in Postgres is `UPDATE counters SET n = n + 1 WHERE key = $1`, which serializes on the row lock and caps at a few hundred writes per second per key until you shard the row across N sub-counters and sum them on read. Two stores, the same logical operation, a hundred-fold throughput difference — because the access pattern (hot atomic increment) matches Redis's native operation and fights Postgres's row-locking concurrency model.

How do you *measure* the win in a review? The metrics that matter are p50 and p99 read latency, write throughput in operations per second, and dollar cost per million operations. The win from matching store to access pattern is usually not 20% — it is one or two orders of magnitude, because you are converting a scan into an index hit. When you propose a datastore in a review, the sentence that wins the argument is "our dominant query is X; in store A it's a scan at ~Y ms p99 and in store B it's an index hit at ~Z ms p99," with Y and Z being real numbers from a benchmark, not adjectives.

## 5. Schema-on-write vs schema-on-read, and what you lose going NoSQL

Two trade-offs deserve their own treatment because they are the ones teams discover painfully, after they have already committed: the schema trade-off and the loss of relational machinery.

**Schema-on-write vs schema-on-read.** A relational database enforces schema *on write*: the database rejects a row that does not match the table's columns and types, so every row in the table is guaranteed to have the same shape. A document or key-value store enforces schema *on read* (if at all): you can write any shape you like, and the *reader* has to cope with whatever shapes exist. The trade is real and cuts both ways. Schema-on-write gives you a guarantee — every row is valid, the database is the enforcer, and you cannot accidentally write garbage. Schema-on-read gives you flexibility — you can evolve the shape without a migration, store heterogeneous data, and move fast while the schema is unstable.

The failure mode of schema-on-read is the one to internalize, because it is silent and accumulates. With no write-time enforcement, the *application* becomes the schema authority, and when multiple services or multiple versions of one service write to the same collection, they write subtly different shapes. Six months later you have documents missing fields that newer code assumes, fields that changed type, and optional fields that are sometimes a string and sometimes an array. The database tells you nothing because it never understood the schema. Every read becomes defensive: check if the field exists, check its type, handle three historical shapes. The flexibility you bought up front you repay, with interest, in every reader forever. The pragmatic middle ground many teams land on is schema-on-write where it matters (the core entities) and a `JSONB` column in Postgres for the genuinely flexible parts — you get relational guarantees on the structured fields and document flexibility for the unstructured ones, in one store.

**What you lose going from relational to a typical NoSQL store.** Three specific capabilities, each of which you will have to rebuild in application code if you need it:

*Secondary indexes.* In Postgres you add an index on any column with one statement and queries on that column become fast. In a key-value store there are no secondary indexes — you can only look up by the primary key — so to query by another attribute you maintain a *second* key-value mapping yourself, and keep it consistent yourself, on every write. DynamoDB offers global secondary indexes, but they cost extra, are eventually consistent, and have their own capacity to provision.

*Transactions.* In Postgres, wrapping three writes in `BEGIN ... COMMIT` gives you atomicity (all or nothing) and isolation (concurrent transactions don't corrupt each other) — see the [isolation-levels deep-dive](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) for what the levels actually guarantee. In many NoSQL stores, multi-key transactions are either absent, limited, or expensive. Without them, a logical operation that touches multiple keys can partially fail and leave inconsistent state, and you end up reaching for the [saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions) to coordinate multi-step changes with compensating actions — which is correct and necessary at scale, but is a lot of machinery to take on before you need it.

*Joins.* In Postgres, relating two entities is a join the query planner optimizes. In a key-value or document store, you either *denormalize* (embed the related data so no join is needed — fast reads, write amplification, and the consistency burden of keeping duplicates in sync) or you join in application code (multiple round trips, no transactional consistency across them). Denormalization is often the *right* answer for a known dominant read pattern — that is the next section — but it is a deliberate trade you make per access pattern, not a free property of the store.

The summary sentence: **relational databases give you generality and correctness machinery for free; NoSQL stores give you a specific fast path and make you rebuild the generality yourself.** That trade is worth it when you have one dominant access pattern and don't need the generality. It is a bad trade when your access patterns are varied or still being discovered — which is most products, most of the time.

## 6. Denormalization: trading write work for read speed

Denormalization deserves its own section because it is the most common and most powerful read optimization, and because it is where "model for your access pattern" becomes concrete. The principle: **store the data in the shape your dominant read wants it, even if that means duplicating it, so the read is a single fetch instead of a join.**

![A before-and-after comparison contrasting a normalized timeline read that requires a five-table join at sixty milliseconds against a denormalized per-user feed row served by a single index hit at two milliseconds](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-7.webp)

Figure 7 is the canonical example: a social feed. The normalized design on the left is correct and clean — users, posts, follows, and likes in separate tables — but rendering one user's timeline means joining across all of them, filtering to the people they follow, sorting by time, and the query planner cannot make that cheap because the join fans out across millions of rows. Measured, that read is tens of milliseconds and it gets worse as the graph grows. The denormalized design on the right precomputes a *feed row per user*: when someone posts, you fan the post out into the precomputed feed of each follower at *write* time, so reading a timeline is a single index hit on `(user_id, created_at)` — about 2 ms, flat regardless of graph size. You moved the work from read time (where it happens on every page load) to write time (where it happens once per post).

The two reads, side by side, make the trade visible. The normalized query asks the planner to join four tables and sort, and its cost grows with the size of the social graph; the denormalized query is a single bounded index scan whose cost is independent of how large the graph gets:

```sql
-- Normalized: one timeline = join across follows, posts, users, likes.
SELECT p.id, p.body, p.created_at, u.handle, count(l.*) AS likes
FROM follows f
JOIN posts  p ON p.author_id = f.followee_id
JOIN users  u ON u.id = p.author_id
LEFT JOIN likes l ON l.post_id = p.id
WHERE f.follower_id = $1
GROUP BY p.id, u.handle
ORDER BY p.created_at DESC
LIMIT 50;
-- fans out across millions of (follower, post) pairs; ~60 ms p99, worse as graph grows

-- Denormalized: write-time fan-out builds a per-user feed row.
-- On each new post, insert one row into feed for every follower:
INSERT INTO feed (user_id, post_id, author_handle, body, created_at)
SELECT f.follower_id, $1, $2, $3, now()
FROM follows f WHERE f.followee_id = $4;   -- N inserts, one per follower

-- The timeline read is now a single bounded index scan:
SELECT post_id, author_handle, body, created_at
FROM feed
WHERE user_id = $1
ORDER BY created_at DESC
LIMIT 50;
-- Index Scan on (user_id, created_at DESC); ~2 ms p99, flat regardless of graph size
```

The trade is explicit and you must say it out loud: denormalization buys read speed with **write amplification** and a **consistency burden**. The write amplification is literal — one post becomes N feed-row writes, one per follower, and for a celebrity with ten million followers that is ten million writes from one action (the famous "fan-out problem" that hybrid feeds solve by *not* fanning out the celebrities and merging their posts at read time). The consistency burden is that you now have the same datum in multiple places and must keep the copies in sync; if a post is edited or deleted, you must propagate that to every copy, and any bug leaves stale duplicates. This is why denormalization belongs *downstream* of a normalized source of truth: keep the authoritative, normalized copy in Postgres, and derive the denormalized read-optimized projections from it via [change-data-capture or an outbox](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), so there is always one place that is correct and the duplicates are explicitly derived.

The senior framing of denormalization is therefore not "normalized good, denormalized bad" or the reverse — it is "**normalize the source of truth, denormalize the read paths, and make the derivation explicit and one-directional.**" That gives you correctness (one authoritative copy) and speed (read-shaped projections) without the worst failure mode (multiple independent writers of the same datum disagreeing forever).

## 7. NewSQL: horizontal scale plus SQL plus strong consistency, at a latency cost

For two decades the painful choice was binary: relational (SQL, transactions, strong consistency) *or* horizontally scalable (NoSQL, no joins, eventual consistency). NewSQL exists to refuse that choice — to give you SQL and ACID transactions *and* horizontal scale across many nodes and regions. The marquee systems are Google Spanner, CockroachDB, Vitess, and YugabyteDB, and the obvious question is: if NewSQL gives you everything, why isn't it the default? Because it does not give you everything for free. It gives you scale and consistency at a **latency cost**, and understanding that cost is how you decide whether it is worth it.

The latency comes from consensus. To keep data consistent across many nodes that can fail independently, NewSQL systems replicate each write to a quorum of nodes and use a consensus protocol — [Raft](/blog/software-development/database/raft-consensus-from-scratch) in CockroachDB and YugabyteDB, [Paxos](/blog/software-development/database/paxos-and-multi-paxos-explained) in Spanner — to agree on the order of writes. A write is not acknowledged until a majority of replicas have durably accepted it. If those replicas are in the same datacenter, that round-trip is a few milliseconds; if they are spread across regions for geo-resilience, it is tens of milliseconds (cross-region round-trips run 30–100+ ms). A single-node Postgres commit pays *none* of this — it writes to the local WAL and acknowledges, often in under a millisecond. So the honest framing is: NewSQL trades single-digit-millisecond single-node commits for tens-of-milliseconds distributed commits, in exchange for surviving node and region failures while keeping SQL and strong consistency. Cross-shard transactions are worse still, because they require a distributed two-phase commit across multiple Raft groups.

So when is the latency tax worth it? When you genuinely need *two of these three at once*: SQL-and-transactions, horizontal write scale beyond one node, and strong consistency. If you need SQL and transactions but fit on one node, stay on Postgres — it is faster and simpler. If you need horizontal scale but can tolerate eventual consistency and don't need joins, a key-value or wide-column store is cheaper and faster. NewSQL earns its latency cost specifically when you need all three: a financial or inventory system that has outgrown a single node, needs strong consistency (you cannot oversell inventory or double-spend), and wants to keep its SQL and transactional model rather than rebuild it on an eventually-consistent store. That is a real and important niche — it is exactly why Spanner was built for Google's ads and financial systems — but it is a *niche*, not a default.

The three NewSQL systems also differ in a way that matters for the decision. [Vitess](/blog/software-development/database/vitess-sharding-mysql-at-scale) is the most conservative: it shards real MySQL underneath, so you keep MySQL's exact behavior and your existing tooling, and you take on shard-awareness in exchange — it is the path for a team already deep in MySQL that has hit the single-node write wall (it is how YouTube scaled MySQL). [CockroachDB](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive) is a from-scratch distributed SQL database that runs on commodity hardware with no special clocks, trading some latency for not needing Google's infrastructure. [Spanner](/blog/software-development/database/spanner-truetime-and-external-consistency) uses GPS-and-atomic-clock-synchronized TrueTime to give externally consistent transactions across continents, which is technically remarkable but ties you to Google Cloud. The decision among them is "how much do I want to keep my existing MySQL (Vitess), run anywhere (CockroachDB), or get the strongest global consistency on GCP (Spanner)."

## 8. Stress test: what breaks when you outgrow single-node Postgres

This is the section that earns the post, because "Postgres scales fine until it doesn't" is useless without knowing *exactly* how it doesn't. Let me walk the failure sequence as a single Postgres node grows, with the order in which things break, so you recognize the real signal that it is time to leave.

![A timeline showing a single Postgres node scaling from ten thousand transactions per second on one box, through read replicas and vertical scaling, to replication lag and the single-writer write wall that forces a migration](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-8.webp)

Figure 8 is the lifecycle, and the key insight is the *order* of what breaks. Reads break first and are easiest to fix; writes break last and are the real ceiling.

**Stage 1 — one box, everything fits (up to ~10k TPS, working set in RAM).** Early on, your working set (the hot data you read frequently) fits in the node's RAM, so reads are served from the buffer cache without touching disk, and a single node handles ten thousand-plus transactions per second easily. Nothing is wrong. Don't touch anything.

**Stage 2 — read load grows, add read replicas (to ~40k+ reads/s).** As read traffic climbs, the primary's CPU saturates on reads. The fix is read replicas: stand up async replicas, route read-only queries to them, and the primary only handles writes and reads that need the freshest data. Read scaling is nearly free this way — add more replicas. But this introduces the first crack: **replication lag**. Replicas are asynchronously behind the primary by some milliseconds-to-seconds, so a read from a replica can return stale data — a user updates their profile and immediately reads it from a replica that hasn't caught up, and sees the old value. You handle this with read-your-writes routing (send a user's reads to the primary for a few seconds after they write) but it is the first place the "single logical database" abstraction leaks. The [replication strategies deep-dive](/blog/software-development/database/database-replication-sync-async-logical-physical) covers the sync-vs-async trade in detail.

**Stage 3 — vertical scaling, approaching the ceiling (to ~80k TPS).** When the primary saturates, the first move is to make it bigger — more vCPUs, more RAM, faster disks. Vertical scaling is wonderfully simple (it is just a bigger box) and gets you surprisingly far, but it has a hard ceiling: the biggest box your cloud offers, and a price curve that goes superlinear (the largest instances cost far more than 2× the mid-size ones). You can buy your way up this curve for a while, but you are now spending real money to defer the inevitable.

**Stage 4 — replication lag becomes chronic.** As write volume on the primary grows, the replicas have more to replay, and the lag that was milliseconds becomes seconds and then minutes under load spikes. Now read-your-writes routing isn't enough, your replicas are unreliable for anything time-sensitive, and you are routing more and more to the already-saturated primary. The read-replica escape valve is closing.

**Stage 5 — the write wall (the real ceiling).** Here is the wall: **a single Postgres primary has one writer.** Every write — every `INSERT`, `UPDATE`, `DELETE` — goes through that one node, serializes through its WAL, and there is no replica you can add to scale *writes*, because replicas are read-only copies. When your write throughput approaches what one (even maximally large) node can sustain, you are out of road. Read replicas don't help (they're reads). A bigger box doesn't help past the largest instance. This is the *only* trigger that genuinely forces a distributed system, and it is a *write* trigger, which is why "we're getting a lot of traffic" is not by itself a reason to leave — a lot of *read* traffic is solved by replicas and caching, and only a lot of *write* traffic hits the wall.

**Stage 6 — migrate (shard or move to NewSQL).** At the write wall you have two real paths: shard Postgres (partition the data across multiple primaries, each owning a slice — application-managed sharding or a tool like Citus), or move to a NewSQL system that shards for you (CockroachDB, Spanner, or Vitess if you're on MySQL). Sharding keeps Postgres but takes on cross-shard query and transaction complexity yourself; NewSQL takes that complexity off your hands at the latency cost from section 7. Either way it is a real migration, and the next section is how to do it without downtime — the forthcoming [partitioning-and-sharding-without-downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) sibling goes deep on the cutover mechanics.

The senior reading of figure 8 is this: every stage before the write wall has a cheap escape valve (replicas, a bigger box, caching), and *only the write wall forces distribution*. So the honest answer to "when do I leave Postgres" is "when your **write** throughput, measured not imagined, approaches single-node capacity and you've already exhausted vertical scaling — and not one stage before." Most teams that leave do so at stage 2 or 3, where a replica or a bigger box would have served them for years.

#### Worked example: when does the cart outgrow Postgres?

Let us put numbers on it for a shopping cart. Suppose an e-commerce site does 1,000 orders per minute at peak, and each order involves roughly 20 cart writes (add item, change quantity, remove, etc.) over its lifetime, plus 5 cart reads per write. That is 20,000 cart writes/min and 100,000 reads/min at peak — about 333 writes/s and 1,667 reads/s. A single Postgres node handles thousands of writes per second and tens of thousands of reads per second comfortably; you are at perhaps 5–10% of a single node's write capacity. Even at 10× this traffic — 10,000 orders/min, 3,333 writes/s — you are within a single well-tuned node's reach, and the reads (16,670/s) are handled by two or three replicas. The conclusion: a shopping cart for a large e-commerce site does *not* outgrow single-node Postgres on write volume; it might move cart *state* to Redis for sub-millisecond reads and TTL-based expiry, but that is an access-pattern optimization (key-value point lookups with expiry), not a scale necessity. If someone proposes Cassandra for a cart "because it needs to scale," this math is your rebuttal.

## 9. Polyglot persistence and its operational cost

Once you accept "match the store to the access pattern," the natural endpoint is polyglot persistence: use *several* stores in one system, each for the access pattern it serves best. This is correct, and most mature systems are polyglot. But it has a cost that teams systematically underestimate, and naming that cost is the senior's job.

![A dataflow diagram of one order service routing each access pattern to a best-fit store, with Postgres for orders, Redis for carts, Elasticsearch for catalog search, and Cassandra for clickstream, glued by change-data-capture](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-4.webp)

Figure 4 is a realistic polyglot e-commerce architecture. Orders live in Postgres (they need ACID and they're relational). Carts and sessions live in Redis (point lookups with TTL). The product catalog is searched through Elasticsearch (full-text and faceting). Clickstream events pour into Cassandra (write firehose, time-ordered). And — this is the part teams forget when they draw this diagram — there is glue: change-data-capture from Postgres into Elasticsearch to keep the search index fresh, and a pipeline feeding the clickstream. Each arrow into a second store is a *synchronization* concern, and synchronization is where polyglot architectures spend their pain.

The operational cost of polyglot persistence, itemized honestly:

- **N systems to operate.** Each store needs monitoring, backups, upgrades, capacity planning, on-call expertise, and security hardening. Four datastores is roughly four times the operational surface of one, and operational expertise does not come free — your team needs people who know how to tune *and recover* each one. A 2 a.m. Cassandra incident requires someone who knows Cassandra.
- **Consistency between stores.** When the same datum lives in Postgres and Elasticsearch, they will diverge — the CDC pipeline lags, a message is dropped, a backfill is incomplete — and you will field "search shows a product that's actually deleted" bugs. Keeping derived stores consistent with the source of truth is an ongoing, never-finished tax. The discipline that makes it tractable is one-directional derivation (one source of truth, everything else derived) plus [idempotent, deduplicated sync](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe).
- **Cognitive load.** Every engineer now has to know which store holds what, how to query each, and how they relate. Onboarding is slower; a change that spans stores is harder to reason about; the "where does this data live and who owns the truth" question gets asked constantly.
- **Transactions across stores are gone.** You cannot wrap a write to Postgres and a write to Redis in one transaction. A logical operation spanning stores can partially fail, and you're back to sagas and compensating actions and the [transactional outbox](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) to make cross-store writes reliable.

So the rule is: **be polyglot deliberately, not reflexively.** Add a second store when an access pattern is *genuinely* and *badly* served by your primary store and the win is large and measured — not because "logs feel like they should go somewhere else." The right sequence is almost always: start with one Postgres serving every access pattern; when one pattern (search, hot key-value, time-series, clickstream firehose) measurably strains it, *add* the one specialized store for that one pattern, keep Postgres as the source of truth, and pay the sync cost knowingly. Each new store should clear a high bar, because each one is a permanent operational tax you pay forever.

## 10. The trade-off matrix, read like a senior

Now bring it together with the decision matrix from figure 2, because reading a trade-off matrix correctly is the skill this whole post is teaching. Here is the same matrix as a table you can take into a review:

| Property | Relational | Key-value | Wide-column | Document | NewSQL |
| --- | --- | --- | --- | --- | --- |
| Strong transactions | **Best** | Weak | Weak | Best (single doc) | **Best** |
| Flexible schema | Rigid | Best | Good | **Best** | Good |
| Write volume | OK | Best | **Best** | Good | Good |
| Range scans | **Best** | Weak | OK (in partition) | Good | **Best** |
| Horizontal scale | Hard | **Best** | **Best** | Good | **Best** |
| Operational cost | **Low** | **Low** | High | Low | High |

The senior does not read this matrix looking for the row with the most "Best" — there is no universally best store, and a column full of "Best" doesn't exist because every store pays somewhere. The senior reads it the other way: **start from the property your workload cannot compromise on, and let it eliminate columns.** If you need strong cross-row transactions, the key-value and wide-column columns are out *immediately*, regardless of their scale virtues — you are choosing among relational, NewSQL, and (for single-document scope) document. If you need horizontal write scale, the relational column gets hard, and you're looking at key-value, wide-column, or NewSQL. Two or three non-negotiable properties usually collapse the matrix to one or two viable columns, and *then* you compare brands within the family.

Watch the operational-cost row especially, because it is the one teams ignore and it is the one that bites for years. Relational, key-value, and document stores are *low* operational cost — they're mature, well-understood, and a small team can run them. Wide-column and NewSQL are *high* operational cost — they're distributed systems with real operational depth, and running them well requires expertise and time you must budget for. A store that wins on scale but costs a senior engineer's worth of operational attention is often a *worse* choice than a slightly-less-scalable store your team can actually operate. The most scalable architecture you cannot run reliably is worse than the less scalable one you can.

The matrix's deepest lesson is the one in its caption: **no family is strong everywhere, so the property your workload needs most does the eliminating.** That is why "what's the best database?" is a malformed question and "what's the dominant access pattern and the non-negotiable property?" is the senior's reframing of it.

## 11. Worked examples: four workloads, four stores, the reasoning

Let me run the framework on four concrete workloads from the brief. For each, I will name the dominant access pattern, the consistency need, and the scale, and let the family fall out — and where two stores are plausible, I will do the capacity-and-cost math that decides between them. Figure 6 maps the conclusions.

![A matrix mapping six real workloads including shopping cart, product catalog search, metrics pipeline, and social graph to the best-fit datastore and the reason it fits](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-6.webp)

**Shopping cart.** Dominant access: point lookup by `user_id` (or session id) — fetch the cart, mutate it, store it back. Consistency: read-your-writes for the owning user (you must see your own additions immediately); a cart is not a money ledger, so eventual consistency across replicas is fine. Scale: high read/write rate but small per-cart data, and carts expire. This is a **key-value** workload with a TTL — Redis or DynamoDB. Redis gives you sub-millisecond reads and native TTL expiry; DynamoDB gives you the same access shape with managed durability and item-level TTL. You do *not* want a relational database here as the hot path (cart reads would hammer it needlessly) and you absolutely do not want Cassandra (the worked example in section 8 showed a cart never hits Cassandra's scale niche). The senior twist: many teams keep the cart in Redis for the live session and write the *order* (the cart at checkout) into Postgres, because once it's an order it needs ACID and durability and relational queries. Cart = key-value; order = relational. Same product, two stores, split by access pattern and consistency need.

**Product catalog with search.** Dominant access: two distinct patterns. Fetch a product by id (point lookup), and search the catalog by text and facets ("noise-cancelling headphones under \$200, 4+ stars, in stock"). Consistency: the catalog is read-heavy and changes slowly; a search index a second or two stale is fine. Scale: catalog size up to millions of SKUs; search QPS can be high. This is a **polyglot** answer: the *source of truth* is **Postgres** (products are relational — categories, variants, prices, inventory, with constraints and transactions), and the *search* is **Elasticsearch**, a denormalized read-optimized projection kept in sync from Postgres via [change-data-capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern). Trying to do search in Postgres with `LIKE` is a full table scan; trying to make Elasticsearch the system of record is asking a near-real-time search engine to be a durable transactional store it was never built to be. Each store does its native job. This is polyglot persistence done *right* — two stores, one source of truth, explicit one-directional sync.

**Metrics pipeline.** Dominant access: append timestamped points at high volume; query ranges over time with aggregation ("p99 latency for service X over the last 6 hours, downsampled to 1-minute buckets"). Consistency: metrics are append-only, never updated, and a dropped point here and there is tolerable. Scale: write firehose — potentially millions of points per second — and time-range reads. This is the **time-series** sweet spot: TimescaleDB or InfluxDB. TimescaleDB is the pragmatic pick if you want to keep SQL and the Postgres ecosystem (it's a Postgres extension with time-partitioned hypertables and automatic compression); InfluxDB if you want a purpose-built time-series engine. A general relational table would choke on the write volume and the time-range scans without heavy partitioning, and a key-value store cannot do range-over-time queries at all. The access pattern — high-volume time-ordered append plus time-range aggregation — points straight at the time-series family.

**Social graph.** Dominant access: this one is a trap, and getting it right is a senior signal. Most "social" features are *not* graph workloads. A follower count is a counter (key-value/relational). A feed is a time-ordered range scan (relational with denormalization, or wide-column). Those belong in Postgres or Redis, not Neo4j. You reach for a **graph** database (Neo4j) only when the *traversal itself* is the product — "friends of friends of friends," "shortest path between two users," "all accounts within three hops of this fraud-flagged one," recommendation by graph proximity. If your dominant access is genuinely a multi-hop traversal, a graph database makes it the size of the *result* instead of an exponential pile of self-joins that gets catastrophically slow past two hops. If your dominant access is "show this user's feed," that is *not* a graph workload and Neo4j is the wrong answer — it is a denormalized range-scan workload (figure 7). The discipline is to ask "is the traversal the product, or just incidental?" before reaching for a graph store.

#### Worked example: capacity and cost for the metrics pipeline, two stores

Take the metrics pipeline and price it across two stores to make the trade concrete. Say the system ingests 500,000 metric points per second sustained, each point about 100 bytes, and retains 30 days of data, queried mostly over recent windows.

Storage math: 500,000 points/s × 100 bytes = 50 MB/s of raw ingest. Over 30 days that is 50 MB/s × 86,400 s/day × 30 days ≈ 130 TB raw. Time-series stores compress aggressively (time-ordered numeric data compresses 10–20×), so on disk this is more like 7–13 TB — meaningful but manageable on a clustered time-series store, and *catastrophic* on a naive relational table where 130 TB uncompressed with B-tree indexes would be unusable for both write throughput and range scans.

Now compare the *write path*. On a relational store, 500,000 writes/s is far past a single node's ceiling (recall the write wall) and even sharded relational would burn enormous hardware on a workload it's not built for, because B-tree inserts do random I/O. On a time-series store built on [LSM-trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), those 500,000 writes/s are sequential appends — the native operation — handled by a modest cluster. The cost difference is large: the relational approach needs many sharded nodes fighting their storage engine, while the time-series cluster needs a handful of nodes doing what they were built for, and the compression alone cuts storage cost roughly 10×. Concretely, you might run the time-series workload on a 5-to-8-node cluster where the equivalent sharded-relational attempt needs several times the nodes and still scans poorly. The senior takeaway: when the access pattern matches the store's native operation, you win on *both* throughput and cost, often by an order of magnitude — and the right way to defend the choice in a review is exactly this kind of capacity math, not "everyone uses InfluxDB for metrics."

## 12. Case studies: a wrong pick, DynamoDB at Amazon, and a migration

Three case studies, each teaching one lesson the framework predicts.

**The team that picked the wrong store (the framing story, finished).** Recall the analytics team from the intro who stood up Cassandra, Elasticsearch, Redis, and Kafka for two thousand rows per customer. The lesson is not "Cassandra is bad" — Cassandra is excellent at what it does. The lesson is that they chose for an *imagined* scale and ignored their *actual* access pattern and data size. Their dominant access was relational analytical queries over a tiny dataset; the right answer was one Postgres, possibly with a read replica, and they could have shipped features three times faster. The framework would have caught it at axis 3 (honest scale): their data fit in RAM, so distribution bought them nothing but cost. The fix, when they eventually did it, was to *collapse* the polyglot stack back to Postgres — a rare and humbling migration in the *opposite* direction of the usual one, and a reminder that you can over-distribute as surely as you can under-distribute.

**DynamoDB at Amazon — the access-pattern-first store.** DynamoDB's origin is the canonical case study for "model the store around the access pattern." Amazon's shopping cart and other high-scale services needed predictable single-digit-millisecond reads and writes at any scale, with availability over strong consistency during partitions — and they did *not* need ad-hoc queries or joins, because the access patterns were known and stable (get cart by id, update cart). The original Dynamo paper (2007) and the later DynamoDB service are built on exactly the trade this post argues: give up joins, ad-hoc queries, and (by default) strong consistency, and in exchange get O(1) key-value access that scales horizontally and stays available. The deep lesson for *your* designs is the discipline DynamoDB forces: you must know your access patterns *before* you model the data, because the store is organized around the keys you'll query.

That discipline shows up as the "single-table design" idiom, where you encode every access pattern into the partition and sort keys up front. Where a relational design would have separate `users` and `orders` tables and join them, a DynamoDB design colocates them under one partition key so the dominant read — "give me a user and their recent orders" — is a single `Query`, not a join that does not exist:

```python
# One table, composite keys chosen so each access pattern is one Query.
# PK = USER#<id>, SK encodes the entity type and sort dimension.
table.put_item(Item={"PK": "USER#42", "SK": "PROFILE",        "name": "Ada"})
table.put_item(Item={"PK": "USER#42", "SK": "ORDER#2026-06-01", "total": 39})
table.put_item(Item={"PK": "USER#42", "SK": "ORDER#2026-06-09", "total": 12})

# Access pattern "user + recent orders" = ONE partition Query, no join:
resp = table.query(
    KeyConditionExpression="PK = :u AND begins_with(SK, :o)",
    ExpressionAttributeValues={":u": "USER#42", ":o": "ORDER#"},
    ScanIndexForward=False,   # newest first
    Limit=20,
)  # single-digit-ms read at any table size, because it hits one partition
```

Notice what this buys and what it costs. It buys O(1)-partition reads that stay fast at any scale, and it costs you the ability to ask a question you did not design a key for — there is no `Query` for "all orders over \$1000 across all users" unless you provisioned a secondary index for exactly that. You trade open-ended queryability for guaranteed-fast known queries. That constraint feels restrictive coming from Postgres, but it is exactly *why* it scales — it refuses to let you write a query that would be a scan. When your access patterns are genuinely known, stable, and key-shaped, that's a feature; when they're still being discovered, it's a straitjacket, which is why DynamoDB is a poor *first* database for an early-stage product and an excellent one for a mature, high-scale, known-access-pattern service.

**A Postgres-to-distributed migration.** The general arc — and several well-documented public migrations follow it — is a company that scaled a single primary relational database for years, hit the write wall (section 8, stage 5), and migrated to a sharded or distributed SQL system. The pattern that makes such a migration survivable is the one in figure 9: you do *not* do a big-bang cutover. You dual-write behind a feature flag (writes go to both old and new store), backfill historical data into the new store via change-data-capture, run *shadow reads* (read from both, compare, alert on divergence) until you trust the new store, then shift read traffic gradually, and only after the new store has served all traffic correctly for a sustained period do you stop writing to the old one and delete it. The crucial property is that every stage is reversible — if shadow reads diverge or the new store misbehaves, you flip the flag back and you're on the old store with no data loss. The lesson: a datastore migration's risk is dominated not by the new store's capabilities but by the *cutover process*, and the safe cutover is incremental, observable (shadow reads), and reversible at every step. The forthcoming [partitioning-and-sharding-without-downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) sibling and the [live-resharding deep-dive](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime) go deep on the mechanics.

![A grid showing a safe Postgres to distributed migration that dual-writes behind a feature flag while change-data-capture backfills history into CockroachDB, with shadow reads comparing results before cutover](/imgs/blogs/choosing-a-datastore-sql-nosql-newsql-9.webp)

Figure 9 is the migration architecture. The application writes to Postgres (the source of truth) with the dual-write flag off; a CDC stream backfills history into CockroachDB; when ready, the dual-write flag turns on so writes hit both; shadow reads compare the two stores' answers; and only when comparison is clean for long enough does CockroachDB become the new primary. Every box is a checkpoint you can stop and roll back at. This is the difference between a migration that ships on a Tuesday and one that becomes a multi-quarter outage.

The dual-write and shadow-read steps are small in code and large in safety. The shape is a flag-gated write to the new store that can never fail the request, plus a sampled read comparison that alerts on divergence without serving the new store's answer to users until you trust it:

```go
func WriteOrder(ctx context.Context, o Order) error {
    // Source of truth: must succeed.
    if err := pg.Insert(ctx, o); err != nil {
        return err
    }
    // New store: dual-write behind a flag, best-effort, never fails the request.
    if flags.Enabled("dualwrite_orders") {
        if err := crdb.Insert(ctx, o); err != nil {
            metrics.Inc("dualwrite.error") // alert, but do not fail the user
        }
    }
    return nil
}

func ReadOrder(ctx context.Context, id string) (Order, error) {
    primary, err := pg.Get(ctx, id) // users always get the trusted answer
    if err != nil {
        return Order{}, err
    }
    if flags.Enabled("shadowread_orders") && sample(0.01) { // 1% comparison
        if shadow, e := crdb.Get(ctx, id); e == nil && !shadow.Equal(primary) {
            metrics.Inc("shadowread.divergence") // gate the cutover on this
        }
    }
    return primary, nil
}
```

The invariant that makes this safe: the user is *always* served by the trusted store, the new store is exercised under real traffic without being on the critical path, and the cutover is gated on a divergence metric staying at zero. When `shadowread.divergence` has been flat through a peak traffic window, you flip reads to the new store; if anything looks wrong, you flip the flag back and you have lost nothing. That reversibility is the entire point.

## 13. When to reach for each (and when not to)

Decisive recommendations, the kind you give in a review:

- **Default to one Postgres (or MySQL).** Start here for essentially every new product. It does joins, transactions, secondary indexes, and JSON, scales reads with replicas and writes into the tens of thousands per second, and a small team can operate it. The burden of proof is on *leaving*, not on staying. Reach elsewhere only when a measured constraint forces it.
- **Add Redis (key-value) when** you have a hot point-lookup access pattern — sessions, carts, rate-limit counters, cache — that wants sub-millisecond reads and TTL expiry. Add it *alongside* Postgres, not instead of it.
- **Add Elasticsearch (search) when** you need full-text or faceted search and `LIKE`/`tsvector` in Postgres is a scan. Keep Postgres as the source of truth and sync via CDC.
- **Add a time-series store (TimescaleDB/InfluxDB) when** you have a high-volume append of timestamped points with time-range queries — metrics, IoT, ticks. TimescaleDB if you want to keep SQL.
- **Reach for wide-column (Cassandra/Bigtable) when** you have a genuine write firehose with a known, partition-key-shaped read pattern and can tolerate eventual consistency. Not for general data, not for a cart, not "because it scales."
- **Reach for a graph database (Neo4j) when** the multi-hop traversal *is* the product — recommendations, fraud rings, shortest paths. Not for feeds or follower counts, which are range-scan and counter workloads.
- **Reach for NewSQL (Spanner/CockroachDB/Vitess) when** you need SQL *and* transactions *and* horizontal write scale *and* strong consistency all at once — typically a financial or inventory system that has truly outgrown one node — and you accept the latency tax of distributed consensus. Vitess if you're already deep in MySQL; CockroachDB to run anywhere; Spanner for the strongest global consistency on GCP.
- **Do not** distribute before the write wall, adopt a new store for resume reasons, make a search engine your system of record, model a wide-column store like a relational one, or go polyglot reflexively. Each new store is a permanent operational tax — make it clear a high bar.

## 14. Key takeaways

- **The axis is access pattern, not SQL-vs-NoSQL.** Name your dominant query's *shape* (point lookup, range scan, join, search, traversal) and the right family falls out almost mechanically. SQL/NoSQL is a consequence, not a cause.
- **Three axes decide it: access pattern, consistency need (per operation, not per database), and honest near-term scale.** Two non-negotiable properties usually collapse the choice to one family.
- **Default to one Postgres; the burden of proof is on leaving.** Most teams leave too early, paying a large today-cost (lost joins/transactions, operational complexity) to hedge a scale problem they don't have and probably won't.
- **Optimize by making the dominant query an index hit or O(1) lookup, never a scan.** The win from matching store to access pattern is one or two orders of magnitude, because you convert a scan into an index hit. Measure it in p99 latency and \$/million-ops.
- **Only the *write* wall forces distribution.** Reads scale with replicas and caching; vertical scaling buys years. The single-writer ceiling is the real trigger, and it's a write trigger — "lots of traffic" alone isn't a reason to leave.
- **Going NoSQL means rebuilding secondary indexes, transactions, and joins yourself.** Relational gives you correctness machinery for free; NoSQL gives you one fast path and makes you build the rest. Worth it for one dominant pattern, bad for varied or undiscovered ones.
- **Denormalize the read paths, normalize the source of truth, and keep derivation one-directional.** Buy read speed with write amplification, but always have one authoritative copy and explicit, idempotent sync to the projections.
- **NewSQL is scale + SQL + strong consistency at a latency cost** — it earns its consensus round-trips only when you need all of those at once. It is a niche, not a default.
- **Polyglot persistence is correct but expensive.** Each store is N× operational surface plus cross-store consistency and lost transactions. Add a store only when one access pattern is genuinely, measurably badly served — and keep one source of truth.
- **Migrations are dominated by cutover risk, not store capability.** Dual-write behind a flag, backfill with CDC, shadow-read to compare, shift gradually, and keep every step reversible.

## 15. Further reading

- [Storage engines: B-trees vs LSM-trees for architects](/blog/software-development/system-design/storage-engines-btrees-vs-lsm-trees-for-architects) — the substrate under these families; read it alongside this post.
- [Consistency models: a practical guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) — turns the consistency axis into a checklist.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — the safe migration mechanics when you hit the write wall.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — classify the operation, not the database.
- [CockroachDB distributed SQL deep-dive](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive), [Spanner and TrueTime](/blog/software-development/database/spanner-truetime-and-external-consistency), and [Vitess: sharding MySQL at scale](/blog/software-development/database/vitess-sharding-mysql-at-scale) — the three NewSQL mechanisms compared in section 7.
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) and [LSM-trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) — the mechanisms behind the write firehose and horizontal scale.
- *Dynamo: Amazon's Highly Available Key-value Store* (DeCandia et al., 2007) and *Spanner: Google's Globally-Distributed Database* (Corbett et al., 2012) — the two papers that defined the NoSQL and NewSQL trade-offs.
- *Designing Data-Intensive Applications* (Kleppmann) — the canonical text on why each family makes the trade it makes.

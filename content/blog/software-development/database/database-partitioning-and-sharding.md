---
title: "Database Partitioning and Sharding"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer's tour of splitting data: single-node declarative partitioning in Postgres, the jump to horizontal sharding across nodes, choosing a shard key, hot shards, scatter-gather, and request routing."
tags:
  [
    "partitioning",
    "sharding",
    "postgres",
    "scalability",
    "distributed-systems",
    "shard-key",
    "vitess",
    "citus",
    "hash-partitioning",
    "database",
    "system-design",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 52
---

Every team that scales a database eventually hits the same wall, and they almost always hit it in the wrong order. They reach for sharding — splitting their data across many machines — months or years before they need it, because sharding is the thing they read about in engineering blogs and the thing that sounds like "scale." Then they discover that they have signed up for distributed transactions, cross-shard joins that fan out to every node, a shard key they can never change, and a re-shard project that will consume a quarter of engineering time. Meanwhile the actual problem — a single table that grew to two billion rows and whose vacuum never finishes, or a query that scans a year of data to answer a question about today — could have been solved with a feature that ships in vanilla Postgres and never leaves one machine.

The confusion is that "partitioning" and "sharding" get used interchangeably, and they are not the same thing. **Partitioning** splits one logical table into many physical pieces that still live inside a single database on a single node — same CPU, same RAM, same write-ahead log, same buffer cache. **Sharding** splits your data across many independent databases on separate hosts, each with its own CPU, its own WAL, its own failure domain. Partitioning is a query-planner and storage optimization. Sharding is a distributed-systems commitment. They solve overlapping symptoms (a table is too big, queries are too slow) but they sit on opposite sides of a chasm, and the entire skill of scaling a relational data tier is knowing which side of the chasm you are on and refusing to cross it until you have to.

This article is the map of that chasm. We start with the cheapest tool — vertical and horizontal partitioning inside one node, including Postgres's native declarative partitioning, partition pruning, partition-wise joins, and time-series retention via `ATTACH`/`DETACH`. We then walk to the edge, look at what single-node partitioning cannot do (the absence of a global unique index is the canonical example), and only then take the leap into sharding: how to choose a shard key (the highest-stakes, least-reversible decision in the whole system), hash versus range sharding, the hot-shard problem, why cross-shard queries and transactions hurt, and the three ways to route a request to the right shard. Throughout, we lean on Martin Kleppmann's *Designing Data-Intensive Applications* (DDIA) Chapter 6 for the vocabulary, and on real production writeups from Instagram, Notion, Figma, Pinterest, Vitess at YouTube and Slack, and Citus, because every one of those teams learned these lessons the expensive way so you do not have to.

The diagram above is the mental model for the first half of the article: there are exactly two axes along which you can cut a table, and almost every "should I partition?" question reduces to which axis you are cutting on.

## The two cuts: vertical and horizontal

![Vertical partitioning slices columns apart by access pattern; horizontal partitioning slices rows apart by a partition key](/imgs/blogs/database-partitioning-and-sharding-1.webp)

**Senior rule of thumb: a table has two dimensions — rows and columns — and you can split along either one. Vertical partitioning splits columns; horizontal partitioning splits rows. Sharding is always horizontal.**

Picture a `users` table with a few hot columns that nearly every request reads (`id`, `email`, `name`), a few cold columns that are large and rarely read (a `bio` text field, an `avatar_blob`), and some write-heavy bookkeeping (`last_login`, a `prefs` JSON document that the settings page rewrites constantly). All of it lives in one row, so every time Postgres reads a row to answer "what is this user's name?", it pays for the cold and write-heavy columns too — they share the same heap page, the same TOAST machinery, the same buffer-cache footprint.

**Vertical partitioning** cuts the columns apart. You move the hot columns into `users_hot`, the cold large columns into `users_cold`, and the write-heavy bookkeeping into `users_audit`, joined back by `user_id`. The win is that the hot table is now small and dense — more rows per page, a higher buffer-cache hit rate, less write amplification on the columns that change all the time. This is the same instinct as normalization, and in practice most "vertical partitioning" is just good schema design plus the occasional deliberate split of a `blob` column out of a hot table. It does not scale you past a single node; it makes the single node spend its RAM and I/O on the data that matters.

**Horizontal partitioning** cuts the rows apart. Every piece — call it a partition — has the *same columns*, but holds a disjoint subset of the rows, chosen by a **partition key**. Split `users` into `p0` (user_id 0–3.3M), `p1` (3.3M–6.6M), `p2` (6.6M–10M) and you have three tables with identical schemas, each a third the size. Range scans on the key touch one partition; old partitions can be dropped wholesale; each partition's index is a third as tall. This is the cut that matters for scale, because it is the cut that generalizes: do it inside one node and you have declarative partitioning; do it across many nodes and you have sharding. The rest of this article is almost entirely about the horizontal cut.

Here is the distinction that trips everyone up, stated as a table because it is worth memorizing:

| Question | Vertical partitioning | Horizontal partitioning |
| --- | --- | --- |
| What gets split? | Columns | Rows |
| Each piece has... | A subset of columns | All columns, a subset of rows |
| Chosen by | Access pattern / column size | A partition key value |
| Reassembled by | `JOIN` on the row's PK | `UNION ALL` / the planner, transparently |
| Scales past one node? | No (it is schema design) | Yes — this is what sharding does |
| Postgres native feature | No (manual table split) | Yes — declarative partitioning |

The reason "sharding is always horizontal" is now obvious: to put data on a different machine you must move whole rows there, and you decide which rows by a key. Vertical splits keep all rows on the same machine and merely rearrange their columns; there is no key to route on. So when an interview or a design doc says "we'll shard the database," it means "we'll horizontally partition the rows across nodes by some key" — and the only interesting question is *which key*.

## 1. Declarative partitioning in Postgres: the cheap win

**Senior rule of thumb: before you shard, ask whether single-node partitioning solves your actual problem. It usually does, because most "my table is too big" pains are really "my queries scan too much" and "my retention deletes are too slow" — both of which partitioning fixes without leaving one machine.**

Postgres has had real declarative partitioning since version 10 (2017), and by 12–16 it is mature: partition pruning at plan time *and* execution time, partition-wise joins and aggregates, concurrent index attach, and `DETACH PARTITION ... CONCURRENTLY`. You declare a parent table as partitioned, then create child partitions, and Postgres routes every insert to the right child and rewrites every query to touch only the relevant children. The parent is "virtual" — it holds no rows of its own; it is a router and a schema.

There are three partitioning methods, and choosing among them is the same choice we will revisit for sharding: range, list, and hash.

### RANGE: the time-series workhorse

`PARTITION BY RANGE` assigns a contiguous interval of the key to each partition. The overwhelmingly common case is time: one partition per day, week, or month. Bounds are **inclusive on the lower end, exclusive on the upper**, so adjacent partitions tile without gaps or overlaps.

```sql
-- Parent: a virtual table, no storage of its own.
CREATE TABLE events (
    event_id    bigint       NOT NULL,
    logdate     date         NOT NULL,
    user_id     bigint       NOT NULL,
    payload     jsonb        NOT NULL
) PARTITION BY RANGE (logdate);

-- One child per month. [lower, upper): Jan owns 01-01..02-01 exclusive.
CREATE TABLE events_2026_01 PARTITION OF events
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE events_2026_02 PARTITION OF events
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE events_2026_03 PARTITION OF events
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

-- A catch-all so a row with an out-of-range date never errors on INSERT.
CREATE TABLE events_default PARTITION OF events DEFAULT;

-- Index defined on the parent is auto-created on every partition.
CREATE INDEX ON events (user_id);
```

The payoff is **partition pruning**: when a query's `WHERE` clause constrains the partition key, the planner discards every partition whose bounds cannot match, before it reads a single page.

![Partition pruning reads only the one monthly partition the predicate can match, skipping the other seven](/imgs/blogs/database-partitioning-and-sharding-2.webp)

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM events
WHERE logdate >= '2026-03-01' AND logdate < '2026-04-01';

-- Append  (cost=... rows=...)
--   ->  Seq Scan on events_2026_03 events_1
--         Filter: (logdate >= '2026-03-01' AND logdate < '2026-04-01')
-- (the other 7 partitions are pruned at plan time and never appear)
```

That `Append` node with exactly one child under it is the thing to look for. Without partitioning, that same query is a single sequential scan (or index scan) over the *entire* `events` table — a year of data to count one month. With pruning, it touches one-twelfth of the heap. Crucially, **pruning uses the partition bounds, not an index** — you get the skip even with no index on `logdate` at all. If you want to confirm pruning is happening, read the `Append` node in your plan; I wrote a whole piece on this in [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer), and partition counts in the `Append` are one of the first things I check.

A subtlety worth internalizing: pruning happens at **plan time** when the predicate is a constant, but Postgres also does **execution-time pruning** when the value is a parameter (a prepared statement, a subquery result, the inner side of a nested-loop join). You will see this surface in plans as `(never executed)` on the pruned children, or in the `Subplans Removed: N` line. Both are wins; the plan-time version is just visible earlier.

### LIST: partition by discrete category

`PARTITION BY LIST` assigns explicit, enumerated values to each partition. It fits low-cardinality categorical keys — region, tenant tier, country — where the set of values is small and known.

```sql
CREATE TABLE orders (
    order_id  bigint NOT NULL,
    region    text   NOT NULL,
    amount    numeric NOT NULL
) PARTITION BY LIST (region);

CREATE TABLE orders_us PARTITION OF orders FOR VALUES IN ('us-east', 'us-west');
CREATE TABLE orders_eu PARTITION OF orders FOR VALUES IN ('eu-central', 'eu-west');
CREATE TABLE orders_apac PARTITION OF orders FOR VALUES IN ('ap-southeast', 'ap-northeast');
CREATE TABLE orders_other PARTITION OF orders DEFAULT;
```

LIST partitioning is great for data residency (keep EU rows in a partition on EU-tablespace storage) and for coarse query isolation. It is a poor choice when one category dwarfs the others — if 80% of orders are `us-*`, the `orders_us` partition is itself a giant table and you have bought almost nothing. That is the same hot-partition trap we will meet again under sharding; LIST partitioning makes it visible early because the skew is right there in the value distribution.

### HASH: even spread when there is no natural range

`PARTITION BY HASH` runs the key through a hash function and assigns the row to a partition by `hash(key) mod modulus`. You declare each partition with a `(modulus, remainder)` pair.

```sql
CREATE TABLE sessions (
    session_id uuid  NOT NULL,
    user_id    bigint NOT NULL,
    data       jsonb
) PARTITION BY HASH (user_id);

CREATE TABLE sessions_p0 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE sessions_p1 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE sessions_p2 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE sessions_p3 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

Hash partitioning gives you **even data distribution** without a natural range key. Its cost is the one we will keep paying for hashing throughout this article: **you lose efficient range scans**. A query for `user_id BETWEEN 1000 AND 2000` cannot prune — adjacent user IDs hash to different partitions, so the planner must scan all four. Hash partitioning prunes only on **equality** (`user_id = 4242` prunes to one partition) and on `IN` lists. This is the core range-vs-hash tradeoff, and it is identical at the partition level and the shard level: range keeps your data sorted (great for scans, hot-spot-prone for time keys), hash spreads it evenly (great for point lookups and even load, useless for ranges).

Here is the decision compressed:

| Method | Prunes on | Best for | Fails at |
| --- | --- | --- | --- |
| RANGE | `<`, `>`, `BETWEEN`, `=` | time-series, ordered keys, retention | hot "current" partition for time keys |
| LIST | `=`, `IN` | regions, tiers, residency | skewed categories (one giant partition) |
| HASH | `=`, `IN` only | even spread, no natural range key | range scans (must touch all partitions) |

### Partition-wise joins and aggregates

The second-order win of partitioning is that the planner can sometimes push *joins and aggregates down into the partitions* instead of reassembling the whole table first. If two tables are partitioned the same way on the join key, Postgres can join partition-to-partition (a **partition-wise join**), which is smaller, more parallelizable, and cache-friendlier.

```sql
SET enable_partitionwise_join = on;       -- off by default; turn it on
SET enable_partitionwise_aggregate = on;  -- off by default; turn it on

-- If events and event_meta are both PARTITION BY RANGE (logdate) with
-- matching bounds, this joins events_2026_03 to event_meta_2026_03 directly,
-- never materializing the full cross-partition product.
SELECT e.user_id, count(*)
FROM events e JOIN event_meta m USING (event_id, logdate)
WHERE e.logdate >= '2026-03-01' AND e.logdate < '2026-04-01'
GROUP BY e.user_id;
```

These are off by default because they cost planning time and only help when the partitioning lines up. But for a partitioned warehouse-style query they can be the difference between a plan that streams per-partition and one that builds a giant hash table over the whole year. (If joins themselves are unfamiliar territory, the mechanics of nested-loop, hash, and merge joins are their own deep topic — partition-wise joins are those same algorithms applied per-partition.)

### Retention: ATTACH and DETACH instead of DELETE

This is the single most underrated reason to partition by time. Deleting a month of rows from a billion-row table is a catastrophe: a giant `DELETE` writes a tombstone for every row, bloats the table, and hands [autovacuum](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning) a mountain of dead tuples to reclaim. With range partitioning, "delete last December" is a metadata operation:

```sql
-- Detach old data without blocking writers (SHARE UPDATE EXCLUSIVE lock only).
ALTER TABLE events DETACH PARTITION events_2025_12 CONCURRENTLY;

-- Now it is a standalone table. Archive it, then drop it instantly.
-- DROP TABLE is O(1) catalog work + unlink; no per-row tombstones, no vacuum.
DROP TABLE events_2025_12;

-- Adding next month is equally cheap. Build the table, constrain it, attach it.
CREATE TABLE events_2026_04
    (LIKE events INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
ALTER TABLE events_2026_04
    ADD CONSTRAINT ck CHECK (logdate >= '2026-04-01' AND logdate < '2026-05-01');
ALTER TABLE events ATTACH PARTITION events_2026_04
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
ALTER TABLE events_2026_04 DROP CONSTRAINT ck;  -- redundant after attach
```

The `CHECK` constraint trick before `ATTACH` is the load-bearing detail: if a matching constraint already proves every row fits the new bounds, Postgres skips the full-table validation scan during attach and the operation is near-instant. Without it, `ATTACH PARTITION` scans the whole incoming table to verify the bounds. This pattern — pre-create next month, attach with a proof constraint, detach-and-drop the oldest — is the entire lifecycle of a time-series table, and it is why every metrics store, audit log, and event pipeline that runs on Postgres is range-partitioned by time. It also pairs naturally with [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations): you can roll a new schema onto next month's partition without touching the historical ones.

### When partitioning hurts

Partitioning is not free, and the failure modes are real:

- **Too many partitions.** The planner handles a few thousand partitions well; tens of thousands and plan time balloons, because pruning still has to consider each one. A daily partition kept for ten years is 3,650 partitions — fine. An hourly partition kept for five years is 43,800 — a planning-time problem. Pick the coarsest granularity that still makes retention drops cheap.
- **Queries that omit the partition key.** If your access pattern does not filter on the partition key, you prune nothing and pay the `Append` overhead of touching every partition for no benefit. A `users` table partitioned by `created_at` but queried by `email` is strictly worse than an unpartitioned table.
- **The cross-partition uniqueness limitation.** This is the big one, and it is the bridge to sharding, so it gets its own section.

## 2. The wall: no global unique index across partitions

**Senior rule of thumb: a partitioned table can only enforce uniqueness on constraints that include the partition key. There is no such thing as a global unique index across partitions. The moment your business requires one, you have hit the wall that sharding lives behind too.**

Postgres requires that any `PRIMARY KEY` or `UNIQUE` constraint on a partitioned table **include all partition key columns**. This is not an arbitrary restriction; it is a physical necessity. Each partition is its own table with its own index. A unique index on `events_2026_03` enforces uniqueness *within March*. Nothing connects it to the index on `events_2026_04`. To guarantee a value is globally unique, Postgres would have to check every partition's index on every insert — which defeats the entire point of partitioning, where the win was touching only one partition.

```sql
-- This FAILS: email does not include the partition key (logdate).
ALTER TABLE events ADD CONSTRAINT uq_email UNIQUE (email);
-- ERROR: unique constraint on partitioned table must include all
--        partitioning columns

-- This WORKS: the constraint includes the partition key.
ALTER TABLE events ADD CONSTRAINT pk_events PRIMARY KEY (event_id, logdate);
```

So your "primary key" on a time-partitioned table is really `(event_id, logdate)` — and `event_id` alone is *not* guaranteed unique across the whole table. If two rows in different months happen to share an `event_id`, Postgres will not stop you. For a synthetic key this is usually fine (you control the generator), but for a natural key like `email` or `username` that must be unique across the entire user base, single-node partitioning simply cannot enforce it. Your choices are: don't partition on a column that isn't part of your uniqueness requirement, push uniqueness into the application or a separate unpartitioned lookup table, or accept that the constraint is advisory.

Hold onto this, because it is *exactly* the constraint you inherit when you shard. A sharded `users` table partitioned across nodes by `user_id` can enforce that `user_id` is unique (each shard owns a disjoint range), but it cannot cheaply enforce that `email` is globally unique, because email lives on whatever shard the user's `user_id` routed to, and a new signup with the same email might route to a different shard. Figma's rule, when they sharded Postgres, was blunt and correct: ["unique indexes are only supported on indexes including the sharding key."](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/) That is the same sentence Postgres prints, scaled up to a cluster. The unique-email problem is then solved the same way at both scales — a separate `emails(email PRIMARY KEY, user_id)` table that *is* keyed by email, checked on signup.

This is the natural pivot point. Single-node partitioning has taken us as far as one machine goes. We have made queries prune, made retention a metadata operation, made joins partition-wise. What we *cannot* do on one node is add more CPU, more RAM, more WAL bandwidth, or more independent failure domains. When one box genuinely cannot hold the write throughput or the storage, we cross the chasm.

## 3. Crossing the chasm: partitioning is one DB, sharding is many

![Partitioning splits a table within one node sharing its cache and WAL; sharding splits across independent nodes](/imgs/blogs/database-partitioning-and-sharding-6.webp)

**Senior rule of thumb: shard only when one node truly cannot hold the write or storage load. Read scaling is not a reason to shard — that is what replicas are for. Storage and write throughput are the only reasons.**

The figure makes the distinction physical. On the left, partitioning: one Postgres node, with partitions `p0..p11` sharing the same CPU cores, the same RAM, the same buffer cache, and crucially the same write-ahead log. All your writes still serialize through one WAL on one disk. Adding partitions does not add write bandwidth; it only organizes the data on the one disk you have. The ceiling is one box.

On the right, sharding: N independent database nodes, each with its own CPU, its own RAM, its own WAL on its own disk, its own crash domain. A write to shard 0 and a write to shard 3 hit different machines and do not contend. This is the only structure that adds *write* and *storage* capacity, because it adds physical machines. But look at what it costs (the danger-red box): cross-shard joins, cross-shard transactions, two-phase commit, and a permanent operational tax — every shard is a server to back up, monitor, fail over, and upgrade.

This is why the rule is "shard last." Consider the ladder of cheaper options, in order:

| Symptom | Try first | Then | Shard only if |
| --- | --- | --- | --- |
| Reads too slow / too many | Indexes, query tuning, caching | Read [replicas](/blog/software-development/database/database-replication-sync-async-logical-physical) | (never — replicas scale reads) |
| Table too big to vacuum / scan | Declarative partitioning | Partition + retention drops | one node can't store it all |
| Write throughput maxed | Faster disk, batch writes, tune WAL | Vertical split of hot tables | one WAL truly can't keep up |
| Need geo-residency | LIST partition by region | Per-region replicas | per-region write volume needs it |

[Read replicas](/blog/software-development/database/database-replication-sync-async-logical-physical) scale reads almost for free and without the distributed-systems tax: a follower is eventually consistent, but it is a full copy you can route `SELECT`s to. Replicas do *not* scale writes — every write still goes through the primary's single WAL and is then replayed on every follower. So if your bottleneck is read QPS, add replicas; if it is write QPS or sheer storage volume that exceeds one box, you have finally earned the right to shard. Figma's own timeline is instructive: they ran a single Postgres on AWS's largest instance through 2020, then spent until late 2022 on caching, replicas, and *vertical* partitioning into 12 separate databases, and only shipped their [first horizontally sharded table in September 2023](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/). Years of cheaper scaling came first.

## 4. The shard key: the highest-stakes decision you will make

**Senior rule of thumb: choosing the shard key is the most consequential and least reversible decision in a sharded system. A good key has high cardinality, spreads load evenly, co-locates data that is queried together, and keeps the common query single-shard. Get it wrong and you will pay forever — re-keying means re-sharding the entire dataset.**

Everything else in sharding is mechanics. The shard key is *strategy*, and it is the one thing you cannot easily undo. Once a row lands in a shard, [Pinterest's design principle](https://medium.com/pinterest-engineering/sharding-pinterest-how-we-scaled-our-mysql-fleet-3f341e96ca6f) — adopted by nearly everyone since — is that **it never moves to another shard**. That immutability is what makes the shard ID encodable into the primary key, what makes routing a pure function, and what makes the choice of key so permanent. Citus, the distributed-Postgres extension, calls it the [distribution column](https://docs.citusdata.com/en/stable/sharding/data_modeling.html) and frames the criteria exactly:

1. **High cardinality.** The key must have many distinct values — far more than your shard count, ideally millions. A boolean, a status enum (`new`/`paid`/`shipped`), or a country code cannot spread data across more than a handful of shards. Citus warns explicitly against low-cardinality columns for this reason. The classic disaster is sharding by `status`: you wanted N shards, you got 3.
2. **Even distribution.** The values must spread roughly uniformly. A key that is technically high-cardinality but skewed in practice (think: 90% of rows belong to your ten biggest customers) concentrates data and load on a few shards. This is the celebrity problem, and it is the single most common way good-on-paper keys fail in production.
3. **Co-location of data accessed together.** Rows that are queried, joined, or transacted together should share a shard key value so they land on the same shard. Citus's whole model is that "rows with the same distribution column value are always on the same machine, even across different tables" — which is what lets joins and transactions stay local. Notion shards by `workspace_id` precisely because [each block belongs to exactly one workspace and users query within a single workspace](https://www.notion.com/blog/sharding-postgres-at-notion), so almost every query is single-shard.
4. **The common query is single-shard.** This is the consequence of the first three, and it is the real test. Write down your top five queries by volume. If four of them include the shard key in the `WHERE` clause, you have a good key. If most of them don't, you will be scatter-gathering across every shard on your hot path, and you have a bad key no matter how even the distribution is.

Here is the good-versus-disastrous table that I wish every team drew before committing:

| Workload | Good shard key | Disastrous shard key | Why |
| --- | --- | --- | --- |
| Multi-tenant SaaS | `tenant_id` / `workspace_id` | `created_at` | Tenants are isolated; nearly every query filters by tenant. A time key makes the newest shard hot and every per-tenant query scatter. |
| Social graph / posts | `user_id` (author) | `post_id` (random UUID) | A user's posts co-locate; their timeline is one shard. Random post IDs scatter a user's posts across every shard. |
| E-commerce orders | `customer_id` | `order_status` | A customer's order history is one shard. Status has 5 values — you get 5 shards, all hot. |
| Time-series metrics | `device_id` (hash) | `timestamp` (range) | Devices spread evenly. Time as the shard key sends *all current writes* to one shard. |
| Chat / messaging | `conversation_id` | `message_id` | A conversation's messages co-locate; loading a thread is one shard. Per-message keys scatter every thread read. |

Notice the pattern: the good keys are almost always the **entity that owns the data and that the application naturally queries by** — the tenant, the user, the customer, the conversation. The disastrous keys are either too-low-cardinality (status), or correct-cardinality-but-scattering (random IDs that don't co-locate related rows), or hot-by-construction (time, which we will examine in detail next).

One more thing the production writeups agree on, and that I have learned to insist on: **fold the shard key into a composite primary key from day one.** Notion's explicit hindsight lesson was to ["introduce a combined primary key instead of a separate partition key"](https://www.notion.com/blog/sharding-postgres-at-notion) — merge the original `id` with `space_id` into one column — to avoid threading the partition identifier through every layer of the application. If the shard key is part of the PK, every entity reference carries its own routing information, and you never have a query that knows the `id` but not the shard. This is the same reason Instagram and Pinterest encode the shard *into the ID itself*, which we will see shortly.

## 5. Hash vs range sharding, and the hot-shard problem

**Senior rule of thumb: hash sharding buys you even distribution at the cost of range scans; range sharding buys you range scans at the cost of hot spots. Neither survives a single viral key without help.**

This is the same range-vs-hash tradeoff from declarative partitioning, now operating across nodes, and DDIA Chapter 6 frames it precisely. **Range sharding** (key-range partitioning, in Kleppmann's terms) assigns contiguous key ranges to shards. Its virtue is efficient range queries — "all events from March" or "users A–D" hit a contiguous set of shards. Its vice, in Kleppmann's words, is that "access patterns can create hot spots — notably when using timestamps as keys, all writes concentrate on the current day's partition while others sit idle." Range-shard by time and you have built a system where N−1 shards are cold museums and one shard takes 100% of the write traffic. That is not scaling; that is a single hot node with N−1 expensive replicas of history.

**Hash sharding** (hash partitioning) runs the key through a hash function and assigns by the hash, which "distributes keys uniformly, reducing skew and hot spots." The critical tradeoff, again from DDIA: "we lose the ability to do efficient range queries. Keys once adjacent become scattered, destroying sort order." Cassandra's compromise is a **compound key**: hash the partition-key portion to choose the shard, but keep the clustering-key portion sorted *within* the shard — so you can range-scan within one device's data even though devices are hash-spread. This is the best of both worlds when your access pattern is "range within an entity," which is enormously common (a user's posts ordered by time, a device's metrics ordered by time).

But here is the part everyone underestimates: **hashing does not save you from a hot key.**

![Hashing evens distribution across shards, yet a single viral key still concentrates writes on one shard](/imgs/blogs/database-partitioning-and-sharding-3.webp)

The left side of the figure is the happy path: hash the shard key, and four shards each take roughly 25% of writes, ~24k qps apiece. Beautiful, even, boring. The right side is the celebrity. One key — a celebrity's account, a viral post's ID, a single tenant who is 10× everyone else — hashes to exactly one shard. Hashing spreads *different* keys evenly; it does nothing for *one* key that is itself a firehose. That key's shard goes to 100% CPU and 85% of total writes while the other three idle at 5% each. Kleppmann is blunt: "most data systems today are not able to automatically compensate for a skewed workload."

The mitigations are all variations on "split the hot key's traffic":

- **Key salting / splitting.** Append a small random suffix to the hot key (`celebrity_id:0` .. `celebrity_id:9`) so its writes spread across 10 logical sub-keys and therefore multiple shards. The cost, as DDIA notes, is that "reads then require additional coordination overhead" — to read the celebrity you must gather all 10 sub-keys. You apply this surgically, only to the handful of keys you have *detected* as hot, not to every key.
- **Caching the hot read.** If the hot key is read-heavy (a celebrity's profile read by millions), a cache layer absorbs the reads and the shard only takes writes. This is often enough; the [hot key in Redis](/blog/software-development/database/redis-applications-and-optimization) is a far cheaper place to absorb a million reads than a Postgres shard.
- **Dedicated shard / isolation.** Pull the whale tenant onto its own shard so its load can't starve the small tenants sharing a shard. Multi-tenant systems do this routinely: 95% of tenants share pooled shards; the ten whales each get a dedicated one.
- **Pick a finer key.** Sometimes the hot key is a symptom of too-coarse a shard key. If sharding by `merchant_id` makes Amazon-the-merchant a hot shard, perhaps the real key is `(merchant_id, region)` or the order line, depending on the access pattern.

And the deeper point: **the source of a hot key is almost always a bad shard key choice, not bad luck.** Random UUIDs are a recurring offender here — not because they're hot (they're the opposite, perfectly uniform), but because they *scatter related data* and force scatter-gather, which is the dual problem. The detailed case for why [random UUIDs hurt database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) — destroying locality, maximizing B-tree page splits — is the index-level version of the same locality argument that governs shard-key choice. A shard key wants the *opposite* properties of a UUIDv4: it wants to co-locate, while a UUIDv4 maximally de-locates.

## 6. Encoding the shard into the ID: the Instagram and Pinterest trick

**Senior rule of thumb: if the shard is a pure function of the ID, you never need a lookup to route — the key is its own routing metadata. Pack the shard ID into the primary key and routing becomes a bit-shift.**

The most elegant idea in production sharding is to make the primary key *contain* the shard. Instagram's [Sharding & IDs at Instagram](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c) is the canonical writeup, and the bit layout is worth memorizing.

![Instagram packs a timestamp, a logical shard ID, and a per-shard sequence into one sortable 64-bit primary key](/imgs/blogs/database-partitioning-and-sharding-5.webp)

A 64-bit ID is carved into three fields:

- **41 bits of timestamp** — milliseconds since a custom epoch (Instagram used 2011-01-01). 41 bits of milliseconds is ~69 years of range. Because the timestamp occupies the high bits, **the ID is sortable by creation time** — `ORDER BY id` is `ORDER BY created_at` for free, and you never need a separate index or column for time-ordering. That alone saves a huge amount of RAM on a billion-row table.
- **13 bits of logical shard ID** — up to 8,192 logical shards. Instagram used a few thousand. The shard is computed once, at insert time, as `user_id % num_logical_shards` (e.g. `31341 % 2000 = 1341`), and baked into the ID.
- **10 bits of per-shard sequence** — 1,024 IDs per shard per millisecond, an auto-incrementing counter that guarantees uniqueness within a single millisecond on a single shard.

```sql
-- Instagram's ID generator lives as a PL/pgSQL function inside each schema,
-- so the shard is implicit in WHERE the function runs.
CREATE OR REPLACE FUNCTION insta5.next_id(OUT result bigint) AS $$
DECLARE
    our_epoch  bigint := 1314220021721;  -- custom epoch (ms)
    seq_id     bigint;
    now_millis bigint;
    shard_id   int    := 5;              -- this schema's logical shard
BEGIN
    SELECT nextval('insta5.id_seq') % 1024 INTO seq_id;
    SELECT floor(extract(epoch FROM clock_timestamp()) * 1000) INTO now_millis;
    result := (now_millis - our_epoch) << 23;  -- 41 bits timestamp, shifted up
    result := result | (shard_id << 10);        -- 13 bits shard, shifted up
    result := result | (seq_id);                -- 10 bits sequence
END;
$$ LANGUAGE plpgsql;
```

Now the magic. Given *any* ID, you recover the shard with a bit-shift and a mask — no lookup table, no directory service, no round-trip:

```python
def shard_of(post_id: int) -> int:
    return (post_id >> 10) & 0x1FFF   # drop 10 seq bits, mask 13 shard bits
```

The application reads a post ID off a URL, shifts and masks, and knows instantly which logical shard holds the row. Logical shards then map to physical databases through a small, stable table — and because there are thousands of logical shards but only dozens of physical machines, you can move *logical* shards between *physical* machines to rebalance, without ever changing an ID. This separation of logical and physical shards is the second great idea, and we will return to it.

[Pinterest's scheme](https://medium.com/pinterest-engineering/sharding-pinterest-how-we-scaled-our-mysql-fleet-3f341e96ca6f) is a sibling: 16 bits of shard ID, 10 bits of type ID (which object kind — pin, board, user), and 36 bits of local auto-increment ID, supporting 65,536 shards and 68 billion objects per shard per type. MySQL hands back the auto-incremented local ID, and the application composes the full 64-bit ID by OR-ing in the shard and type. Same principle: the ID self-describes its location, and a row never leaves its shard.

The lesson generalizes even if you don't use 64-bit packed IDs: **make routing a pure function of data the client already has.** The opposite — needing a database lookup to discover where a row lives — is a directory service, which is sometimes the right call (we'll cover it) but is strictly more machinery than a bit-shift.

## 7. Cross-shard queries: the scatter-gather tax

**Senior rule of thumb: a query that includes the shard key is a single-shard query and is fast. A query that does not is a scatter-gather across every shard, bounded by your slowest shard, and it does not scale. Design so the hot path is always single-shard.**

The whole point of a shard key is that the common query carries it and routes to one shard. But some queries genuinely don't have the shard key — "find the user with this email" when you shard by `user_id`, or "top 100 posts globally by likes" when you shard by `author_id`. These become **scatter-gather** (Kleppmann calls the read pattern over document-partitioned indexes exactly this): send the query to every shard (scatter), collect every partial result, and merge them (gather).

![A non-shard-key query must scatter to every shard and gather all partial results, bounded by the slowest shard](/imgs/blogs/database-partitioning-and-sharding-4.webp)

Walk the figure left to right. An app query filters on `email` with no shard key. The router (a Vitess VTGate, a Citus coordinator, a Figma DBProxy) cannot pick a single shard, so it fans the query out to shard 0, 1, 2, and 3. Each shard runs the query against its *local* index and returns its partial result. The router gathers all four, then merges — sorting, applying the global `LIMIT`, deduplicating — in its own memory. The result goes back to the client.

Three things make scatter-gather painful, and they compound:

1. **Latency is bounded by the slowest shard, not the average.** Shard 3 in the figure is in a GC pause, taking 400ms. The whole query waits for it. With four shards each having a p99 of 50ms, the query's p99 is the p99 of the *max of four samples*, which is dramatically worse than 50ms — this is **tail latency amplification**, and it gets worse as you add shards. A scatter-gather over 100 shards waits on the slowest of 100 every single time.
2. **The router does work the database used to do.** Sorting and `LIMIT`-ing across shards happens in the router's memory, not in any one optimized B-tree. A `ORDER BY likes DESC LIMIT 100` over 50 shards means each shard returns its top 100, the router merges 5,000 rows and re-sorts. Aggregations are worse: `COUNT(DISTINCT user_id)` cannot be summed from per-shard counts (the same user might appear on multiple shards), so either you accept approximation or you ship raw data to the router. `AVG` must be decomposed into `SUM` and `COUNT` per shard and recombined. Every aggregate that isn't trivially decomposable (`SUM`, `MIN`, `MAX` are; `DISTINCT`, `MEDIAN`, `PERCENTILE` are not) becomes a research project.
3. **It does not scale.** Adding shards makes single-shard queries *cheaper* (each shard is smaller) but makes scatter-gather *more expensive* (more shards to wait on, more partials to merge). A workload dominated by scatter-gather gets worse as you grow, which is the exact opposite of why you sharded. Figma's explicit guidance is to ["limit scatter-gather usage to prevent scalability degradation."](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/)

The mitigations are all about *avoiding* the scatter, not optimizing it:

- **Add the shard key to more queries.** Often "find by email" can become "find by email *within a workspace*" if the UI context already knows the workspace. The shard key was there all along; the API just dropped it.
- **A secondary lookup table keyed by the alternate field.** Maintain `emails(email PRIMARY KEY, user_id)` — itself sharded by `email` — so "find by email" is a single-shard lookup that returns the `user_id`, then a single-shard fetch on the user. Two single-shard hops beat one N-shard scatter. This is a global secondary index, hand-built, which is the next section.
- **A separate search/analytics system.** "Top posts globally" and full-text search belong in Elasticsearch or a columnar warehouse fed by CDC, not in your sharded OLTP store. Don't make your transactional shards answer analytical questions.

And the cardinal rule, which deserves its own callout:

> A single-shard query scales with your shard count. A cross-shard query scales *against* it. Every scatter-gather on your hot path is a future incident waiting for the shard count to grow large enough to trip it.

### Cross-shard transactions: avoid them

Scatter-gather is the read problem; cross-shard transactions are the write problem, and they are worse. A transaction that writes to shards 2 and 5 atomically requires **two-phase commit (2PC)**: a coordinator asks both shards to *prepare*, waits for both to vote yes, then tells both to *commit*. If the coordinator dies between prepare and commit, both shards sit holding locks on prepared-but-uncommitted data until the coordinator recovers — an availability hole. 2PC is a blocking protocol, it ties your write latency to the slowest participant, and it turns two independent failure domains into one correlated one.

The industry consensus, learned the hard way, is **don't do cross-shard transactions on the hot path.** Co-locate data that must be transactionally consistent under the same shard key (Citus's co-location, Figma's "colos") so the transaction stays single-shard, where it is a normal ACID transaction with normal locking and the database's own [isolation levels](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent). When you genuinely need cross-shard atomicity — a money transfer between two users on different shards — prefer an application-level pattern (sagas with compensating actions, an outbox with idempotent retries) over distributed 2PC. The whole art of shard-key design in section 4 is, in large part, the art of making sure the things that must be transactional share a shard.

## 8. Local vs global secondary indexes under sharding

**Senior rule of thumb: a local (document-partitioned) secondary index makes writes cheap and reads scatter; a global (term-partitioned) index makes reads cheap and writes scatter. You are choosing which operation pays the cross-shard tax.**

This is DDIA's "partitioning and secondary indexes," and it is the deepest concept in the chapter because it has no free lunch. When you shard a table, what happens to indexes on columns *other than* the shard key?

![Local secondary indexes scatter reads while global indexes scatter and slow down writes](/imgs/blogs/database-partitioning-and-sharding-7.webp)

The matrix lays out the two answers side by side.

**Local (document-partitioned) secondary index.** Each shard maintains its own secondary index over *only its own rows*. If you shard `users` by `user_id` and want a secondary index on `email`, each shard indexes the emails of the users it happens to hold. Writes are cheap and atomic: inserting a user touches exactly one shard, which updates its local index in the same transaction. Reads by the secondary key, however, must **scatter to every shard** — the shard holding `alice@example.com` is determined by Alice's `user_id`, which a query that only knows the email cannot compute. This is scatter-gather again, with the p99 bounded by the slowest shard. DDIA notes this is "widely adopted in Elasticsearch, Cassandra, and MongoDB" precisely because cheap writes matter more than cheap secondary reads for most workloads.

**Global (term-partitioned) secondary index.** A single global index spans all shards but is *itself partitioned* — by the indexed term, on a different key than the primary data. The `email` index is sharded by `email`, independent of how the user rows are sharded by `user_id`. Now a read by email is a **single-shard lookup**: hash the email, hit one index shard, get the `user_id`, done. The cost moves to writes: inserting one user now touches *two* shards — the user's data shard (by `user_id`) and the email index shard (by `email`) — which are almost always different machines. That cross-shard write is either a (slow, blocking) distributed transaction or, far more commonly, **asynchronous**: the index is updated after the fact, so it briefly lags the data. DDIA: "writes are slower and more complicated... updates typically occur asynchronously." DynamoDB's Global Secondary Indexes are exactly this, and they are eventually consistent for exactly this reason.

| Property | Local (document-partitioned) | Global (term-partitioned) |
| --- | --- | --- |
| Index lives | On each shard, over its own rows | Separately, sharded by the indexed term |
| Write cost | 1 shard, atomic with the row | 2 shards (data + index), often async |
| Read by secondary key | Scatter to all shards, gather | 1 index shard, then 1 data shard |
| Consistency | Always consistent with the data | Often eventually consistent (lags writes) |
| Examples | Elasticsearch, Cassandra, MongoDB | DynamoDB GSI, hand-built lookup tables |

In practice, on a sharded Postgres/MySQL stack (Vitess, Citus, Figma), you build global secondary indexes by hand: a separate sharded table keyed by the secondary field, kept in sync via the application or change-data-capture. The "email lookup table" from the previous section *is* a hand-rolled global secondary index. The decision — local versus global — is really "which is hotter, writes or reads-by-this-field?" If you write users constantly but rarely look them up by email, keep it local and eat the occasional scatter. If you look up by email on every login, build the global index and accept the dual write.

## 9. Request routing: how a query finds its shard

![App-aware, proxy, and directory routing all turn a shard key into one target node, coordinated by a shared shard map](/imgs/blogs/database-partitioning-and-sharding-8.webp)

**Senior rule of thumb: every sharded system needs a single source of truth for "which shard is on which node," and a way for queries to consult it. The three patterns — app-aware client, routing proxy, directory service — trade developer simplicity against operational coupling.**

DDIA frames routing as three approaches, and every production system is one of them or a blend. The figure shows all three converging on the right shard, with a coordination service (ZooKeeper/etcd) holding the authoritative shard map that everyone watches.

**A) App-aware client (client-side routing).** The application library knows the sharding function and the shard map, computes the target shard in-process, and connects directly to it. This is what Instagram and Pinterest do: `shard = (id >> 10) & 0x1FFF`, then look up the physical node in a config the app already has. The win is one fewer network hop and no routing tier to operate. The cost is that the routing logic lives in every client in every language; a shard-map change must propagate to every app instance, which is why you back it with a watched config in ZooKeeper/etcd rather than a hardcoded table. DDIA: clients "possess partition-to-node mapping knowledge directly."

```python
# App-aware routing: pure function from key to connection, no round-trip.
import bisect

# Logical shard -> physical DSN, loaded from etcd and refreshed on change.
SHARD_MAP = {17: "postgres://db3.internal/shard_17", ...}

def route(workspace_id: int) -> str:
    logical = workspace_id % NUM_LOGICAL_SHARDS   # e.g. % 480 (Notion uses 480)
    return SHARD_MAP[logical]                      # physical DSN for this shard

# A write is a single connection to a single shard.
dsn = route(req.workspace_id)
with connect(dsn) as conn:
    conn.execute("INSERT INTO schema%03d.block (...) VALUES (...)", (logical,))
```

**B) Routing proxy (routing tier).** A dedicated, shard-aware proxy receives every query, parses it, extracts the shard key, picks the shard, and forwards. This is [Vitess's VTGate](https://vitess.io/docs/21.0/reference/features/sharding/) (born at YouTube, now run by Slack, GitHub, and Square) and [Figma's DBProxy](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/). The proxy speaks the database's wire protocol, so the application thinks it's talking to a single database and needs almost no sharding logic. VTGate uses a **Vindex** (a vindex is the mapping from a column value to a keyspace ID, which then maps to a shard) to route, and supports online resharding by updating that mapping. Figma's DBProxy goes further with a real query engine: a parser builds an AST, a logical planner extracts the shard IDs from the query, and a physical planner maps logical shards to physical databases and rewrites the SQL. The proxy is where scatter-gather, load-shedding, and request hedging live. The cost is a new tier to operate and a hop on every query — but the developer experience of "just talk to MySQL" is why large MySQL shops standardize on Vitess.

**C) Directory service.** A lookup table maps each key (or key range) explicitly to a shard: `workspace 7f3a -> shard 17`. The application (or proxy) consults the directory to route. This is the most flexible — you can move *any individual key* to *any shard* by updating one directory row, which makes rebalancing and isolating a whale tenant trivial — but it adds a lookup on the hot path and makes the directory itself a critical, must-be-fast, must-be-highly-available dependency. You cache it aggressively and back it with the same coordination service.

All three share one non-negotiable requirement, which Kleppmann states plainly: there must be "consensus among all the nodes about which partitions belong to which nodes." That is why ZooKeeper and etcd show up in every serious design — they are the source of truth for the shard map, and routing components subscribe (watch) for changes so a rebalance propagates without a deploy. The choice among A/B/C is mostly organizational: a polyglot shop with many languages leans toward a proxy (write the routing once, in the proxy); a monolith in one language can get away with an app-aware library; a system that must move individual keys often wants a directory.

| Routing | Hop on hot path | Where routing logic lives | Best for | Real example |
| --- | --- | --- | --- | --- |
| App-aware client | None (direct) | Every app, every language | Few languages, ID-encoded shard | Instagram, Pinterest |
| Routing proxy | One (proxy) | One proxy tier | Polyglot shops, "just talk to MySQL" | Vitess (YouTube/Slack), Figma DBProxy |
| Directory service | One (lookup) | A lookup table + cache | Frequent per-key rebalancing | Many multi-tenant SaaS |

## 10. Rebalancing, and why `mod N` is a trap

**Senior rule of thumb: never make the shard count part of the routing function, because then changing the count re-shuffles everything. Use many fixed logical shards mapped onto few physical nodes, so adding a node moves whole shards, not individual keys.**

The naive routing function is `shard = hash(key) % N` where N is the number of *physical* nodes. It is also a trap, and DDIA is emphatic: "when partitions equal `hash mod N`, changing N causes excessive rebalancing — most keys must migrate between nodes." Go from 4 nodes to 5 and `hash(key) % 4` versus `hash(key) % 5` disagree for ~80% of keys. Every one of those rows must physically move while the system is live. That is not a rebalance; that is a full re-shard triggered by adding a single machine.

The fix that nearly everyone converged on is **a fixed, large number of logical shards, mapped onto a smaller, variable number of physical nodes.** You compute `logical_shard = hash(key) % L` where L is fixed forever (and large — hundreds or thousands). The logical-to-physical mapping is a separate, mutable table. Adding a node doesn't change L or any key's logical shard; it just reassigns some logical shards from existing nodes to the new one — and only the rows in *those* logical shards move. DDIA calls this the "fixed number of partitions" strategy (Elasticsearch, Riak, Couchbase use it); the critical decision is choosing L up front, because L caps how many physical nodes you can ever spread to.

This is why the production numbers look the way they do. **Notion chose 480 logical shards** across 32 physical databases (15 logical per physical), and they chose 480 specifically because [it factors cleanly](https://www.notion.com/blog/sharding-postgres-at-notion) — 480 = 2⁵ × 3 × 5, divisible by 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80, 96, 120, 160, 240. That divisibility means they can grow from 32 to 40 to 48 to 60 physical hosts and always split the 480 logical shards evenly, "whereas powers of 2 would require doubling capacity each time." **Instagram's** thousands of logical shards mapped to dozens of physical machines is the same idea; **Pinterest's** 65,536-shard ID space is the extreme version. **Citus** does it too: shards have fixed hash ranges, and "shards with the same hash range are always placed on the same node even after rebalance," so a rebalance moves whole shards and preserves co-location.

The DDIA taxonomy of rebalancing strategies is worth keeping in your head:

- **Fixed number of partitions** — pick L large up front; adding a node steals whole partitions. Simple, predictable, but L is a permanent ceiling.
- **Dynamic partitioning** — partitions split when they grow past a threshold and merge when they shrink (HBase, MongoDB). Overhead scales with data, but an empty database starts with one partition, so all early writes hit one node until it splits — which is why these systems let you pre-split.
- **Proportional to nodes** — fixed partitions *per node*; adding a node splits some random partitions and takes half (Cassandra). Keeps per-node partition count constant as you grow.

There is a beautiful technique lurking under "moving whole logical shards" — **consistent hashing**, which arranges shards and nodes on a ring so that adding or removing a node moves only `1/N` of the keys instead of nearly all of them — and the live mechanics of moving a logical shard between nodes without downtime (snapshot, stream the WAL/binlog tail, cut over, verify) is its own deep subject. Both are distributed-systems topics that deserve their own treatment; here it's enough to know that the *reason* they exist is to make the rebalance in this section cheap, and that "many logical shards, few physical nodes" is the precondition that makes them work at all.

## Case studies from production

### 1. Instagram: the shard lives in the primary key

Instagram needed IDs that were unique across thousands of logical shards, sortable by time (for feeds), and routable without a lookup. Auto-increment couldn't be globally unique across shards; a UUID would have been unsortable and would have wrecked index locality. Their answer was the 64-bit packed ID: 41 bits of millisecond timestamp (custom 2011 epoch), 13 bits of logical shard, 10 bits of per-shard-per-ms sequence, generated by a PL/pgSQL function living inside each shard's schema. The two payoffs that compound: `ORDER BY id` equals `ORDER BY created_at` so feeds need no separate time index, and `(id >> 10) & 0x1FFF` recovers the shard from any ID with zero round-trips. The lesson that outlived the specific scheme: make routing a pure function of data the client already holds, and you eliminate an entire class of lookup infrastructure.

### 2. Notion: shard by the unit of access (workspace)

Notion stored everything as "blocks," and by 2021 the single `block` table was the bottleneck. The decision that made the whole project tractable was the shard key: **workspace_id**, because every block belongs to exactly one workspace and users overwhelmingly query within a single workspace — so almost every query is single-shard. They landed on 480 logical shards across 32 physical Postgres databases (15 logical schemas per physical host, `schema001.block` .. `schema480.block`), with the application routing in a single step from `workspace_id` straight to physical DB and schema, rather than using native Postgres partitioning's two-stage routing. The 480 number was chosen for divisibility, so they could grow physical hosts in even increments. Their hard-won hindsight — "introduce a combined primary key instead of a separate partition key" — is the lesson everyone relearns: fold the shard key into the PK so every reference self-routes.

### 3. Figma: separate logical from physical, ship the safe half first

Figma ran one giant Postgres until 2020, then spent two years on the cheap scaling ladder — caching, replicas, and vertical partitioning into 12 databases — before sharding a single table horizontally in September 2023, growing the stack ~100x over four years. Their key insight was risk decomposition: split the migration into **logical sharding** (rewriting queries to be shard-aware, enforced via Postgres views) and **physical sharding** (the actual data move), so the riskless logical change shipped and stabilized before the risky physical failover. They allowed multiple shard keys — `UserID`, `FileID`, `OrgID` — because "almost every table could be sharded using one of these," and grouped tables sharing a key into **colos** that support local joins and transactions. DBProxy (a real parser + logical planner + physical planner) routes queries, limits scatter-gather, and enforces the rule that became this article's recurring refrain: "unique indexes are only supported on indexes including the sharding key." The physical failover took only ~10 seconds of partial primary availability with zero replica impact.

### 4. Pinterest: a row never leaves its shard

Pinterest finished sharding their MySQL fleet in early 2012, starting from 8 EC2 servers each running one MySQL instance, master-master replicated to a backup. The design principle they committed to — and that the whole industry adopted — is that **once data lands in a shard, it never moves out**. That immutability is what let them encode location into the ID: 16 bits of shard, 10 bits of type (pin/board/user), 36 bits of local auto-increment, supporting 65,536 shards and 68 billion objects per shard per type. MySQL returns the auto-incremented local ID; the app OR-s in shard and type to compose the 64-bit ID. No data movement meant no distributed transactions for the common path, no re-keying, and routing as a pure function — at the cost of needing to choose the shard for new objects carefully, since that choice is forever.

### 5. The hot tenant that took down the pool

A multi-tenant analytics product sharded by `tenant_id` (good key: high cardinality, queries filter by tenant, co-located). Distribution looked even in staging. In production, one enterprise customer onboarded 40× the data and 60× the query volume of the median tenant, and their `tenant_id` hashed to a shard shared with 200 small tenants. That shard's CPU pinned at 100%, and because the 200 small tenants shared it, *their* p99 latency tripled — a noisy-neighbor incident caused by a single whale. The fix was a directory-service routing layer that let them pin the whale to a dedicated shard with one row update, no re-keying. The lesson: even a textbook-good shard key has a skew tail, and you need the *ability to isolate individual keys* (directory routing or per-key overrides) before you need it, not during the incident.

### 6. The scatter-gather that got slower as it scaled

A messaging app sharded by `conversation_id` — correct, because loading a thread is single-shard. But the "search all my messages" feature had no `conversation_id`; it filtered by `user_id` and a text query. With 8 shards it was a tolerable 120ms scatter-gather. As they grew to 64 shards, each shard got smaller and single-shard thread loads got faster — but search got *slower*, because it now waited on the slowest of 64 shards and merged 64 partial result sets. p99 search latency went from 120ms to 900ms even though every individual shard was faster. The fix was to stop answering search from the OLTP shards entirely: a CDC pipeline fed messages into Elasticsearch, indexed by `user_id`, and search became a single-system query. The lesson is the cardinal rule made concrete: scatter-gather scales *against* your shard count, so the more successful you are, the worse it gets.

### 7. The `mod N` re-shard from hell

A team sharded with `shard = hash(user_id) % 8` directly against 8 physical MySQL nodes. It worked beautifully until they needed a 9th node. Switching to `% 9` meant ~89% of all users would route to a different node — effectively a full data migration of the entire fleet, online, with dual-writes and a months-long backfill, all to add one machine. They instead bit the bullet once: re-sharded to 4,096 *logical* shards (`% 4096`, fixed forever) mapped onto physical nodes via a table, so the *next* capacity addition would move only a slice of logical shards. The painful migration they did was the last one of its kind. The lesson, straight from DDIA: never put the physical node count in the routing function; route to many fixed logical shards and map those to nodes.

### 8. Partitioning that solved the "sharding" problem

A fintech company was convinced they needed to shard a 1.8-billion-row `transactions` table whose nightly retention `DELETE` ran for six hours and whose `created_at`-range reports timed out. They scoped a six-month sharding project. A staff engineer instead range-partitioned the table by month on `created_at`: reports gained partition pruning and dropped from 40 seconds to under one; retention became `DETACH PARTITION CONCURRENTLY` + `DROP TABLE`, completing in milliseconds instead of six hours and eliminating the vacuum storm that had been bloating the table. The single node had plenty of headroom on writes and storage; the problem was never "too big for one box," it was "queries scan too much and deletes are too slow" — both pure partitioning wins. The six-month sharding project was cancelled. The lesson: diagnose whether your pain is *storage/write capacity* (shard) or *scan amount / retention cost* (partition) before committing to the distributed-systems tax.

### 9. The global unique constraint that couldn't be enforced

A SaaS sharded `users` by `org_id` for clean tenant isolation. Months later, product required that `email` be globally unique across all orgs (for SSO). But a sharded-by-`org_id` table can only enforce uniqueness on constraints including `org_id` — exactly the Postgres declarative-partitioning limitation, scaled to a cluster. Two users in different orgs could register the same email and no shard would object. The fix was a hand-built global secondary index: an `emails(email PRIMARY KEY, user_id, org_id)` table sharded *by email*, written in the same application transaction as the user (accepting a rare cross-shard dual-write, made idempotent via an outbox). Signup checked the email shard first. The lesson: cross-shard uniqueness is the global-secondary-index problem in disguise — local indexes can't enforce it, so you build a global one by hand and decide whether the dual write is sync or async.

### 10. Vitess: making MySQL look like one database

A company on MySQL hit the write ceiling of their largest primary. Rewriting the application to be shard-aware across dozens of services in three languages was untenable. They adopted Vitess, the sharding layer YouTube built and Slack and GitHub run. VTGate, the routing proxy, speaks the MySQL wire protocol, so applications kept connecting to "MySQL" and barely changed. Sharding was configured via Vindexes (column-value to keyspace-ID mappings), and Vitess's online resharding split overloaded shards by streaming binlogs to new shards and cutting over with seconds of read-only time. The team got horizontal write scaling without a multi-year application rewrite, paying instead with a new proxy tier to operate. The lesson: a routing proxy trades operational complexity for developer simplicity, and for a polyglot shop that trade is almost always worth it — write the sharding logic once, in the proxy, not N times across services.

### 11. Citus: distributed Postgres without leaving SQL

A real-time analytics product on Postgres needed to scale writes and parallelize aggregations across nodes while keeping full SQL. They used Citus, which turns Postgres into a distributed database via a coordinator and worker nodes. They `create_distributed_table('events', 'device_id')` to hash-distribute by `device_id` (high cardinality, even spread), and co-located the related `device_metadata` table on the same `device_id` so joins stayed node-local. Small shared lookup tables became **reference tables**, replicated to every worker so they never required a cross-shard join. Aggregations like `SELECT device_id, avg(temp) ... GROUP BY device_id` ran in parallel across workers and merged at the coordinator. The lesson: when your access pattern co-locates cleanly by one high-cardinality key and your cross-table needs are either co-located or small-enough-to-replicate, a distributed-Postgres extension gives you sharding's write scaling while preserving transactions and SQL within a co-location group.

### 12. The skewed range shard (time as the key)

An IoT platform range-sharded sensor readings by `timestamp` — it made "last hour of data" a clean contiguous read. But every new reading, by definition, has a near-current timestamp, so 100% of writes hammered the single newest shard while every historical shard sat idle as an expensive read-only archive. They had built a single hot write node with dozens of cold replicas of the past. The fix was to re-shard by `hash(device_id)` with `timestamp` as a *clustering* key within each shard (the Cassandra compound-key pattern): writes now spread evenly across all shards by device, while "last hour for device X" stayed an efficient in-shard range scan. The lesson is DDIA's hot-spot warning made literal: range-sharding by a monotonically increasing key (time, auto-increment ID) sends all writes to one shard — hash the high-cardinality entity instead, and keep time as the in-shard sort order.

## When to reach for partitioning / sharding, and when not to

### Reach for declarative partitioning when

- A single table is large enough that **vacuum, retention deletes, or full scans hurt**, but one node still has CPU, RAM, and write headroom. This is the 90% case, and it never leaves one machine.
- You have a **time dimension and a retention policy.** Range-partition by time and retention becomes `DETACH` + `DROP` instead of a multi-hour `DELETE` and a vacuum storm.
- Your **hot queries filter on a natural range or category key** (time, region, tenant) so partition pruning skips most of the data.
- You want **partition-wise joins/aggregates** for a warehouse-style workload where two big tables share a partitioning scheme.

### Reach for sharding when

- **One node genuinely cannot hold the write throughput or the storage volume** — you have maxed the disk, the WAL, the largest instance, and replicas don't help because the bottleneck is *writes*, not reads.
- You have a **shard key that keeps the hot path single-shard**: high cardinality, even distribution, co-locating data accessed together. If you can't name such a key, you are not ready to shard.
- You can **tolerate the loss of global unique constraints, cross-shard joins, and cross-shard transactions** on the hot path — or you have designed (co-location, hand-built global indexes, lookup tables) so you don't need them.
- You have, or will build, **a routing layer and a single source of truth for the shard map**, and you have chosen **many fixed logical shards over a small variable physical count** so rebalancing is cheap.

### Skip sharding when

- **Your bottleneck is reads.** Add [read replicas](/blog/software-development/database/database-replication-sync-async-logical-physical). They scale reads almost for free and carry none of the distributed-systems tax. Sharding scales writes and storage, nothing else.
- **You haven't partitioned yet.** If a single big table is the pain, declarative partitioning probably fixes it on one node. Don't take the distributed leap to solve a single-node problem.
- **You can't name a single-shard hot path.** If most of your top queries lack a common high-cardinality key, you'll scatter-gather across every shard and your system gets *slower* as it grows. Fix the data model first or don't shard.
- **Your data requires global uniqueness or frequent cross-entity transactions** that you cannot co-locate. The cross-shard tax (2PC, async global indexes, sagas) may cost more than the scaling buys.
- **You're sharding for resume-driven reasons.** Sharding is a permanent commitment — the shard key is the least reversible decision in your system. Every one of the production teams above earned it by exhausting cheaper options first, over *years*. So should you.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 6 (Partitioning) — the canonical treatment of key-range vs hash, secondary-index partitioning, rebalancing, and routing. Everything here is downstream of it.
- [Sharding & IDs at Instagram](https://instagram-engineering.com/sharding-ids-at-instagram-1cf5a71e5a5c) — the 64-bit packed-ID design.
- [Herding Elephants: Sharding Postgres at Notion](https://www.notion.com/blog/sharding-postgres-at-notion) and their follow-up on the great re-shard — workspace-key sharding and the 480-logical-shard rationale.
- [How Figma's Databases Team Lived to Tell the Scale](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/) — DBProxy, colos, logical/physical separation, and the unique-index rule.
- [Sharding Pinterest: How we scaled our MySQL fleet](https://medium.com/pinterest-engineering/sharding-pinterest-how-we-scaled-our-mysql-fleet-3f341e96ca6f) — "a row never leaves its shard," ID composition.
- [Vitess sharding docs](https://vitess.io/docs/21.0/reference/features/sharding/) (YouTube/Slack/GitHub) and [Citus distribution-column guidance](https://docs.citusdata.com/en/stable/sharding/data_modeling.html) — proxy and distributed-Postgres approaches.
- [PostgreSQL declarative partitioning docs](https://www.postgresql.org/docs/current/ddl-partitioning.html) — the authoritative reference for RANGE/LIST/HASH, pruning, partition-wise operations, and the constraint limitations.
- Sibling posts on this blog: [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer), [database replication](/blog/software-development/database/database-replication-sync-async-logical-physical), [B-trees and how indexes work](/blog/software-development/database/b-trees-how-database-indexes-work), and [why random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance).

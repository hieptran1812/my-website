---
title: "Cassandra and DynamoDB: A Deep Dive into Leaderless Wide-Column Stores"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "How the Amazon Dynamo paper seeded Cassandra, DynamoDB, and ScyllaDB, and how to model, distribute, and operate leaderless wide-column stores without falling into the traps that bite teams who treat them like SQL."
tags:
  [
    "cassandra",
    "dynamodb",
    "scylladb",
    "wide-column",
    "leaderless",
    "nosql",
    "data-modeling",
    "distributed-systems",
    "lsm-tree",
    "databases",
    "system-design",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-1.webp"
---

Every team that adopts Cassandra or DynamoDB has the same week-three meeting. Someone has shipped a feature, the table is live, and then a product manager asks for a query that the schema cannot serve — "show me all orders placed last Tuesday, sorted by total." In a relational database that is a `WHERE` clause and an `ORDER BY`. In Cassandra it is an `ALLOW FILTERING` warning followed by a full-cluster scan that times out, and in DynamoDB it is a `Scan` that burns the table's entire read budget and still returns nothing useful. The engineer who built the table did nothing wrong. They were thinking in the relational model — design the entities, then write whatever queries you need — and that model is exactly inverted from how these systems work.

The reason for the inversion is not laziness or immaturity on the part of the database. It is a deliberate trade, made by the people who built [Amazon's Dynamo](https://www.allthingsdistributed.com/2007/10/amazons_dynamo.html) in 2007 and propagated through every system descended from it. They wanted a store that stays available and stays fast no matter how large it grows, even when whole datacenters fail, and they discovered that you cannot get that and also get the relational query model. So they gave up joins, gave up flexible ad-hoc queries, gave up — in the original Dynamo — even single-key strong consistency, and in exchange they got a database that scales horizontally to petabytes and trillions of rows on commodity hardware, where adding capacity is adding nodes and where no single failure takes the system down. The whole discipline of "modeling for Cassandra" is the discipline of living inside that trade.

This article is the real-systems synthesis of the distributed track. We have covered the building blocks separately — [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) as the storage engine, [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) as the distribution layer, [quorums, anti-entropy and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) as the consistency mechanism, and [single-leader, multi-leader and leaderless replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) as the replication taxonomy. Here we assemble them into the two systems that dominate production: Apache Cassandra (with its faster cousin ScyllaDB) and Amazon DynamoDB. We will trace the lineage, build the data model from access patterns up, walk the storage and distribution internals, work through tunable consistency and lightweight transactions in real CQL, and spend a long time on the operational realities — tombstones, hot partitions, repair, secondary indexes — because that is where teams actually get hurt. The diagram below is the mental model for the family tree we are about to climb.

## The Dynamo lineage

![The Amazon Dynamo paper seeded Cassandra, Riak, DynamoDB, and ScyllaDB downstream](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-1.webp)

The diagram above is the mental model: a single 2007 paper at the root, and everything else a branch. [Dynamo: Amazon's Highly Available Key-value Store](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/dynamo.pdf), presented at SOSP 2007, was not the first distributed key-value store, but it was the first production system to combine consistent hashing, vector-clock versioning, sloppy quorums with hinted handoff, anti-entropy with Merkle trees, and gossip membership into one coherent always-writable design. It was built to keep Amazon's shopping cart available through any failure, because a cart that rejects a write loses a sale, and a lost sale is real money. The paper's central sentence — that Dynamo "targets the design space of an always-writeable data store" and therefore "pushes the complexity of conflict resolution to the reads" — is the genetic code passed to every descendant.

The branches diverged in important ways. **Apache Cassandra**, born at Facebook in 2008 and open-sourced into Apache, took Dynamo's ring-and-replication distribution model and married it to **Google Bigtable's** wide-column data model — the sorted, sparse, column-family row. That hybrid is why Cassandra is a "wide-column" store and not just a key-value store: a partition can hold millions of clustered rows, not one opaque blob. **Riak** stayed closest to pure Dynamo, exposing vector clocks and sibling resolution directly to the application. **Amazon DynamoDB**, launched in 2012, is the most interesting fork: it kept the Dynamo *name* and the partitioning philosophy but threw away leaderless replication entirely. Each DynamoDB partition is a single-leader replication group running [Multi-Paxos](/blog/software-development/database/paxos-and-multi-paxos-explained), and the system is fully managed, so the operator never sees a node. **ScyllaDB**, from 2015, is a C++ rewrite of Cassandra that keeps the on-the-wire protocol and CQL but replaces the JVM with a shard-per-core architecture — same model, dramatically better hardware utilization.

It is worth being precise about the assumption-versus-reality gap, because almost every production incident on these systems traces to a wrong assumption carried over from relational databases.

| Assumption (from SQL) | The naive view | The reality in Cassandra / DynamoDB |
| --- | --- | --- |
| "I design tables, then query them." | Schema first, queries follow. | Queries first, schema follows. You design one table per access pattern and accept duplication. |
| "Joins let me combine tables." | The database joins for me. | There are no joins. You pre-join by denormalizing at write time. |
| "Secondary indexes make any column queryable." | Add an index, query the column. | Native secondary indexes are weak (Cassandra) or a separate replicated table (DynamoDB GSI); both have sharp limits. |
| "More data just means a bigger disk." | Vertical scaling. | Data is partitioned across nodes by a hash of the partition key; a bad key creates hot nodes regardless of disk size. |
| "A transaction is atomic and isolated." | ACID everywhere. | Multi-row ACID is absent or expensive; single-partition atomicity and rare Paxos-backed conditional writes are what you get. |
| "Deletes free up space." | `DELETE` removes the row. | A delete writes a tombstone; space is reclaimed only after compaction past `gc_grace_seconds`. |

Hold these six rows in your head. The rest of the article is an elaboration of why each one is true and what to do about it.

> Dynamo's descendants are not "SQL without the SQL." They are a different machine with a different cost model, and the relational instincts that served you for a decade are precisely the instincts that will hurt you here.

## 1. The data model: partition key picks the node, clustering key sorts within it

**The senior rule of thumb: the partition key decides *where* your data lives, and the clustering key decides *what order* it lives in. Everything about modeling flows from those two sentences.**

A Cassandra (or ScyllaDB) primary key has two parts, and conflating them is the single most common modeling error. The **partition key** is hashed to a token, and that token determines which node owns the data — it is the unit of distribution. The **clustering key** sorts rows *within* a single partition on disk — it is the unit of ordering. Together they uniquely identify a row, but they do completely different jobs. The figure below makes the structure physical.

![Partition key routes to a node while clustering keys sort rows inside the partition](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-2.webp)

Read it left to right and top to bottom. The partition key — here `(channel_id, bucket)` — is fed through Cassandra's `murmur3` hash to produce a token, and that token routes to a node (plus its replicas). Every row sharing that partition key lands on the same node, stored contiguously on disk, sorted by the clustering key. A query that supplies the full partition key and a clustering-key range is the happy path: one node, one partition, one sequential read, no coordination across the cluster. That is the shape every Cassandra table should be designed to produce.

Here is the canonical Discord-style messages table, which we will return to repeatedly because [Discord published exactly how they model it](https://discord.com/blog/how-discord-stores-trillions-of-messages):

```sql
CREATE TABLE messages (
    channel_id   bigint,
    bucket       int,          -- a static 10-day time window
    message_id   bigint,       -- Snowflake ID: chronologically sortable
    author_id    bigint,
    content      text,
    PRIMARY KEY ((channel_id, bucket), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```

The composite partition key is the double-parenthesized part: `(channel_id, bucket)`. The clustering key is `message_id`, ordered descending so the newest messages come first. The `bucket` is not cosmetic — it is the load-bearing decision of the whole schema, and we will see in section 8 exactly why a partition keyed on `channel_id` alone would eventually destroy a node.

The defining query this table serves is "fetch the most recent 50 messages in a channel":

```sql
SELECT * FROM messages
WHERE channel_id = 1234 AND bucket = 9
ORDER BY message_id DESC
LIMIT 50;
```

Notice what is *required*: you must supply the entire partition key (`channel_id` and `bucket`) for Cassandra to know which node to ask. You cannot query by `author_id` here, because `author_id` is neither a partition key nor a clustering key — there is no index that maps authors to messages in this table. If you need "all messages by author X," that is a *different access pattern* and it needs a *different table*. This is the query-first discipline in its rawest form.

### Wide rows are the point, not an accident

A relational row is narrow and fixed-width: a handful of columns, one row per logical record. A wide-column partition is the opposite — it is designed to hold thousands to millions of clustered rows under one partition key. A single channel-bucket partition might hold every message sent in that channel over ten days. This is why the model is called "wide-column": the partition is a wide, sorted strip of rows, and you read a contiguous slice of it. The cost model rewards this — reading a slice of a wide partition is one disk seek and a sequential scan — but it also sets the trap: a partition that grows without bound becomes a single hot, fat object on one node, and there is no automatic resharding to save you. Bounding partition size is your job, done at modeling time through the partition key.

### Synthesizing Kleppmann: wide-column sits between key-value and relational

Martin Kleppmann's *Designing Data-Intensive Applications* draws the model spectrum carefully in Chapter 3. A pure key-value store maps an opaque key to an opaque value; you can only get and put whole values. A relational store gives you rows, columns, joins, and a query planner that figures out access paths. The wide-column model sits deliberately in between: the partition key behaves like a key-value key (it routes, it is the unit of access), but inside the partition you get a sorted, sparse, queryable structure — rows addressed by clustering key, columns that can be added per-row. You get more structure than key-value and far less than relational, and the constraint is the feature: because access is always anchored on a partition key, the system always knows which node to ask, which is what lets it scale horizontally without a query planner that reasons about the whole dataset.

## 2. Query-first modeling: denormalize on purpose, one table per access pattern

**The senior rule of thumb: list your queries first, then build one table per query, and duplicate the data into each one. If two queries need the same data sorted two different ways, that is two tables, and you write to both.**

In a normalized relational schema you store each fact once and reconstruct relationships at read time with joins. In a wide-column store you do the opposite: you store facts as many times as you have ways of reading them, and you do the "join" at *write* time by writing the denormalized rows into every table that needs them. The figure below contrasts the two disciplines side by side.

![Relational schemas join at read time while wide-column schemas duplicate data per query](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-3.webp)

The left column is the relational instinct: three normalized tables — `users`, `orders`, `order_items` — and a three-way join at query time. Data is stored once, which is space-efficient and flexible (a new query is just a new join), but the read fans out across multiple tables and, in a distributed setting, multiple nodes. The right column is the wide-column discipline: you decide up front that you need "orders by user" and "orders by status," so you build *two tables*, `orders_by_user` and `orders_by_status`, each holding the same order data partitioned and sorted for its own query. Data is duplicated on purpose. Each read is a single-partition scan with zero joins. The price is rigidity — a genuinely new access pattern means modeling a new table and backfilling it — but the payoff is that every read is fast and local at any scale.

Here is the denormalized pair in CQL. Both tables describe the same orders; they differ only in how they partition and sort.

```sql
-- Access pattern 1: "show me a user's orders, newest first"
CREATE TABLE orders_by_user (
    user_id      uuid,
    created_at   timestamp,
    order_id     uuid,
    status       text,
    total_cents  bigint,
    PRIMARY KEY (user_id, created_at, order_id)
) WITH CLUSTERING ORDER BY (created_at DESC);

-- Access pattern 2: "show me all PENDING orders in a region, oldest first"
CREATE TABLE orders_by_status (
    region       text,
    status       text,
    created_at   timestamp,
    order_id     uuid,
    user_id      uuid,
    total_cents  bigint,
    PRIMARY KEY ((region, status), created_at, order_id)
) WITH CLUSTERING ORDER BY (created_at ASC);
```

When an order is created or its status changes, the application writes to **both** tables — typically in a single `BATCH` so they stay consistent for that partition. This is the moment relational engineers flinch: you are storing `total_cents`, `created_at`, and `order_id` twice, and if you forget to update one table the two will drift. That is real, and it is the tax you pay. The discipline that makes it survivable is to funnel all writes through a single data-access layer (a service or repository class) that knows about every table an entity lives in, so "update an order" always fans out to every copy. Discord's [data services in Rust](https://discord.com/blog/how-discord-stores-trillions-of-messages) are exactly this layer.

### The same discipline in DynamoDB: single-table design

DynamoDB's vocabulary is different but the philosophy is identical. DynamoDB calls the partition key the **partition key (PK)** and the clustering key the **sort key (SK)**; together they are the primary key, and all items sharing a partition key form an **item collection** retrievable in one `Query`. [Alex DeBrie's single-table design](https://www.alexdebrie.com/posts/dynamodb-single-table/) — popularizing patterns Rick Houlihan demonstrated in his legendary re:Invent talks — takes denormalization to its logical extreme: put *multiple entity types* in one physical table, with a generic `PK`/`SK` schema, so that a single `Query` against one partition returns a user *and* all of their orders together. The key insight DeBrie states plainly: "the main reason for using a single table in DynamoDB is to retrieve multiple, heterogeneous item types using a single request."

Here is what a single-table item collection looks like. The partition key overloads multiple entity types behind a `USER#<id>` prefix, and the sort key distinguishes them:

```
Table: AppData   (PK = pk, SK = sk)

| pk           | sk                  | type    | attributes                          |
|--------------|---------------------|---------|-------------------------------------|
| USER#42      | PROFILE             | User    | name="Ada", email="ada@x.io"        |
| USER#42      | ORDER#2026-06-01#a1 | Order   | total=4200, status="SHIPPED"        |
| USER#42      | ORDER#2026-06-08#b7 | Order   | total=1500, status="PENDING"        |
| USER#42      | ORDER#2026-06-14#c3 | Order   | total=9900, status="PENDING"        |
```

A single query returns the profile and all orders for user 42, sorted by the `ORDER#<date>#<id>` sort key, in one round trip — no join, one partition:

```python
import boto3
from boto3.dynamodb.conditions import Key

table = boto3.resource("dynamodb").Table("AppData")

# "Get user 42's profile and all orders, newest order first"
resp = table.query(
    KeyConditionExpression=Key("pk").eq("USER#42")
        & Key("sk").begins_with(""),   # all items in the collection
    ScanIndexForward=False,            # descending sort-key order
)
items = resp["Items"]
```

The `begins_with("ORDER#")` variant would return only the orders; `eq("PROFILE")` would return only the profile. One partition, one item collection, every access pattern carved out of the sort-key namespace. As DeBrie warns, the downside is real: "a well-optimized single-table DynamoDB layout looks more like machine code than a simple spreadsheet," it has a steep learning curve, and adding a brand-new access pattern after the fact can require an ETL pass over the whole table. For early-stage applications that value flexibility over peak performance, multi-table designs (one entity per table) are a legitimate choice — you trade some single-request efficiency for legibility and easier analytics.

> Normalization optimizes for write-once, read-flexibly. Wide-column denormalization optimizes for write-many, read-fast. You are not choosing a worse model; you are choosing a different point on the same fundamental trade-off, and you must choose it deliberately.

### Why native secondary indexes are a trap in Cassandra

The relational reflex when a new query appears is "add an index." Cassandra has secondary indexes (`CREATE INDEX`), and they are one of the sharpest footguns in the system, because they look like relational indexes and behave nothing like them. A relational secondary index is a separate sorted structure that lets the planner jump straight to matching rows. A Cassandra secondary index is **local to each node**: every node indexes only the data it stores. So a query on a secondary index that is *not* also constrained by the partition key has no way to know which nodes hold matches — it must **scatter to every node in the cluster**, gather partial results, and merge. That is a fan-out query, and its latency is governed by the slowest node, scaling with cluster size in exactly the wrong direction.

```sql
-- Looks innocent, behaves catastrophically at scale:
CREATE INDEX ON orders_by_user (status);

-- This query scatters to EVERY node, because status is not the partition key:
SELECT * FROM orders_by_user WHERE status = 'PENDING';   -- fan-out, do not do this
```

Native secondary indexes are tolerable only in narrow cases: low-cardinality columns *and* a query that also pins the partition key (so the fan-out collapses to one node), or small clusters where scatter-gather is cheap. The right answer for "query by a non-key column at scale" is almost always a **denormalized lookup table** — model `orders_by_status` as its own table with `status` in the partition key, as we did above, and pay the duplication. Cassandra later added SASI and, more recently, storage-attached indexes (SAI) that improve the ergonomics, but the fundamental locality problem — an index on a non-partition-key column cannot avoid consulting many nodes — does not disappear. The discipline holds: model the query into the partition key, do not bolt an index onto the wrong axis.

### The DynamoDB analog: GSI overloading and the inverted index

DynamoDB's equivalent of "a new query" is "a new GSI," and the canonical single-table trick is the **inverted index**: a GSI whose partition key is the base table's sort key and whose sort key is the base table's partition key, flipping the access direction. Combined with **GSI overloading** — giving the GSI generic attribute names (`GSI1PK`, `GSI1SK`) populated differently per entity type — a single GSI can serve several access patterns at once, which is how single-table designs keep the number of physical indexes small. The pattern looks like this:

```
Base table:   PK = USER#42        SK = ORDER#...#c3
Inverted GSI: GSI1PK = ORDER#c3   GSI1SK = USER#42   -> "who placed order c3?"
Overloaded:   GSI1PK = STATUS#PENDING  GSI1SK = 2026-06-14  -> "pending orders by date"
```

The same write that creates an order populates the base item and its `GSI1PK`/`GSI1SK` attributes; DynamoDB asynchronously projects it into the GSI. You have, in effect, hand-built the secondary access path that Cassandra's local index tries and fails to provide cheaply — but here it is a properly partitioned, separately scaled structure, which is why GSIs work where Cassandra's native indexes do not. The cost is the same denormalization discipline plus the GSI's eventual consistency.

### Collections and counters: small conveniences with sharp edges

Cassandra offers collection types (`set`, `list`, `map`) and a special `counter` type, and both reward restraint. Collections are stored inline in the row and are convenient for small, bounded groupings — a user's handful of roles, a message's reactions:

```sql
CREATE TABLE user_roles (
    user_id  uuid PRIMARY KEY,
    roles    set<text>,
    prefs    map<text, text>
);
UPDATE user_roles SET roles = roles + {'admin'} WHERE user_id = 550e8400-...;
```

The trap is that a collection has a hard element cap (tens of thousands) and, more importantly, updating a non-frozen collection writes tombstones for the elements it replaces — so a frequently-mutated `list` becomes a tombstone generator inside the row. Keep collections small and bounded; if a "collection" can grow without limit, it is really a wide partition and should be a clustered table instead. Counters are their own subsystem: a `counter` column supports atomic increment/decrement, but counters are not idempotent on retry (a failed-then-retried increment can double-count), cannot be mixed with non-counter columns in the same table, and — being effectively a single mutable cell — concentrate contention, which is exactly the single-hot-key problem of section 8. For high-rate counting, shard the counter across many partition-key values and sum on read.

## 3. The storage engine: it's an LSM tree all the way down

**The senior rule of thumb: Cassandra, ScyllaDB, and DynamoDB are all LSM-backed, which means writes are cheap sequential appends and the real costs live in compaction and read merging. If you understand the LSM cost model, you understand their performance.**

Underneath the data model, the on-disk engine is a [log-structured merge tree](/blog/software-development/database/lsm-trees-write-optimized-storage-engines). This is not a coincidence; the wide-column model and the LSM engine fit each other perfectly, because both are organized around sorted keys and sequential I/O. The figure below recaps the write and read path in Cassandra's terms.

![Writes append to commit log and memtable then flush to immutable sorted SSTables](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-4.webp)

A write does two cheap things: it appends to the **commit log** (the write-ahead log, fsync'd for durability) and inserts into the **memtable**, an in-memory sorted structure. The write is acknowledged the instant the commit log sync returns — no seeks, no page reads. When the memtable fills (the default flush threshold is tied to heap budget, often effectively tens of megabytes per table), it is frozen and flushed to disk as an **SSTable** (sorted string table): one sequential write of already-sorted rows. SSTables are immutable. Reads must therefore merge: check the memtable and row cache first, then for each candidate SSTable consult its **bloom filter** (a probabilistic "definitely not here / maybe here" test) before touching disk, then merge the matching rows with newest-timestamp-wins. **Compaction** runs in the background, merging SSTables, discarding overwritten values and expired tombstones, keeping the file count and read amplification bounded.

This cost model explains the systems' personality. Writes are fast and uniform because they are sequential appends — Cassandra is famously a write-optimized store. Reads pay an amplification cost proportional to how many SSTables a key might be in, which is why compaction strategy matters and why bloom filters are load-bearing. And "deletes" are writes (tombstones), which is the root of an entire class of operational pain we will get to.

#### Second-order optimization: choose the compaction strategy to match the workload

The non-obvious lever in this whole layer is **compaction strategy**, set per table, and the wrong choice quietly wrecks either read latency or disk usage. The three to know:

```sql
-- Read-heavy, update-in-place workloads: keep few SSTables per key.
ALTER TABLE orders_by_user
  WITH compaction = {'class': 'LeveledCompactionStrategy'};

-- Write-heavy, append-mostly workloads: cheapest writes, more SSTables per read.
ALTER TABLE events
  WITH compaction = {'class': 'SizeTieredCompactionStrategy'};

-- Time-series / TTL data: drop whole expired windows without rewriting them.
ALTER TABLE messages
  WITH compaction = {'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_unit': 'DAYS',
                     'compaction_window_size': 1};
```

**Leveled compaction (LCS)** organizes SSTables into size-tiered levels so that a key lives in at most one SSTable per level, capping read amplification at the number of levels (around 7) — excellent for read-heavy and update-heavy tables, at the cost of higher write amplification because data is rewritten as it moves down levels. **Size-tiered compaction (STCS)** merges SSTables once enough of similar size accumulate — lowest write amplification, best for write-heavy append-only data, but a key can be spread across many SSTables so reads merge more. **Time-window compaction (TWCS)** is STCS bucketed by time, and it is the correct choice for any time-series or TTL'd table: an entire expired time window becomes one set of SSTables that can be dropped wholesale when its data passes TTL plus `gc_grace_seconds`, so you never scan tombstones for old data. Matching the strategy to the workload is one of the highest-leverage tuning decisions in Cassandra, and getting it wrong is a common root cause of "reads got slow as the table grew."

### Why ScyllaDB exists: same model, no JVM

ScyllaDB is a drop-in-compatible rewrite of Cassandra in C++ that keeps the exact same LSM model, CQL, and on-wire protocol but changes the execution architecture. Cassandra runs on the JVM with a thread-per-request model and a shared heap; under heavy load its garbage collector introduces [latency spikes — the "stop-the-world" pauses Discord cited as a top pain point](https://discord.com/blog/how-discord-stores-trillions-of-messages). ScyllaDB uses a [shard-per-core architecture](https://www.scylladb.com/product/technology/shard-per-core-architecture/) built on the Seastar framework: each CPU core gets its own shard of data with dedicated memtables, SSTables, cache, and network queues, and cores communicate by explicit message passing rather than shared memory and locks. There is no JVM and no garbage collector, so there are no GC pauses. The payoff in Discord's migration was concrete: they went from **177 Cassandra nodes to 72 ScyllaDB nodes**, fetch-message p99 latency dropped from **40–125 ms to around 15 ms**, and insert p99 from **5–70 ms to a steady 5 ms**. Same model, same data, roughly 2.5× the hardware efficiency and an order of magnitude better tail latency.

Kleppmann's Chapter 3 frames the deeper point: LSM trees are the answer the whole industry converged on for write-heavy and large-scale workloads precisely because they turn the one thing hardware does best — sequential writes — into the foundation, and buy back read and space cost with background work. Every store in our family tree made that bet.

## 4. Distribution: the consistent-hashing token ring

**The senior rule of thumb: data placement is a pure function of `hash(partition_key)`. No coordinator, no lookup table — every node can compute where any key lives. That is what lets the cluster scale and survive failures without a single point of control.**

A Cassandra cluster is a ring of token ranges. Each physical node owns many small **virtual nodes** (vnodes), each responsible for a slice of the token space. When you write a row, the coordinator hashes the partition key to a token, finds which vnode owns that token, and that locates the primary replica; the next `RF − 1` nodes clockwise hold the other replicas, where `RF` is the **replication factor**. This is [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) in production, and the figure below shows a key landing on the ring and replicating to three nodes.

![Consistent hashing maps each key to a token and replicates it to three consecutive nodes](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-5.webp)

The key `channel_id = 1234` hashes to token 118, which falls in node C's range, so C is the primary replica; with `RF = 3` the data is also written to the next two nodes clockwise, D and E. No node is a master — this is the leaderless property inherited from Dynamo. A client connects to *any* node, which becomes the **coordinator** for that request and forwards it to the actual replica nodes. The dotted ring arrows show the token ordering; the bold arrow shows the key landing on its primary.

Virtual nodes solve two problems at once. First, **even load distribution**: if each physical node owned one large contiguous range, removing a node would dump its entire range onto a single neighbor; with 256 vnodes per node, that range is scattered across many nodes so the rebalance is smooth and parallel. Second, **heterogeneous hardware**: a beefier machine can own proportionally more vnodes. This is straight from the Dynamo paper, which introduced vnodes precisely to avoid the load-skew and slow-bootstrap problems of one-token-per-node.

Creating a keyspace pins down the replication strategy and factor:

```sql
CREATE KEYSPACE chat
WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'us_east': 3,     -- 3 replicas in the us_east datacenter
    'eu_west': 3      -- 3 replicas in eu_west
};
```

`NetworkTopologyStrategy` places replicas across racks and datacenters intelligently — it will not put all three replicas on machines that share a rack or a failure domain, so a rack power loss cannot take out a full quorum. This multi-datacenter awareness is also what makes `LOCAL_QUORUM` (next section) possible: you can demand a quorum *within the local datacenter* and replicate to remote datacenters asynchronously, which is the standard pattern for global low-latency reads.

### How DynamoDB hides the ring

DynamoDB applies the same hashing principle but hides the ring entirely behind a managed service. Your partition key is hashed to place the item in a **partition**, an internal storage unit with a token range, a size cap of roughly **10 GB**, and a throughput ceiling. As a partition grows or gets hot, DynamoDB **splits** it automatically — the operation you must do by hand in Cassandra (re-model the key, backfill) is automatic in DynamoDB, which is a genuine operational advantage. The catch, as we will see, is that automatic splitting cannot rescue a *single hot key*, because a single partition-key value cannot be split across partitions. The physics of consistent hashing are the same in both systems; only the management burden differs.

## 5. Tunable consistency: W + R > N is the whole game

**The senior rule of thumb: in a leaderless store you choose consistency per query by setting how many replicas must acknowledge a write (W) and a read (R). If W + R > N you get strong consistency for that key; if not, you get eventual consistency and lower latency. There is no global setting — it is a per-operation dial.**

This is the most important and most misunderstood feature of Cassandra. Because there is no leader, "is my read consistent?" is not a property of the database — it is a property of the *consistency level you requested on each operation*, combined with the replication factor `N`. The [quorum math](/blog/software-development/database/quorums-anti-entropy-and-read-repair) is simple and exact: if the number of replicas a write waits for (`W`) plus the number a read waits for (`R`) exceeds `N`, then the read and write quorums must overlap on at least one replica, so a read is guaranteed to see the latest acknowledged write. The figure below lays out the common levels at `RF = 3`.

![Strong consistency requires write replicas plus read replicas exceeding the replication factor](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-6.webp)

Walk the rows. **ONE** waits for a single replica on both read and write: lowest latency, tolerates two nodes down, but `W + R = 2 ≤ 3`, so a read can miss a recent write — eventual consistency. **QUORUM** waits for two of three: `W + R = 2 + 2 = 4 > 3`, so the quorums overlap and you get strong consistency, at the cost of needing two replicas up. **LOCAL_QUORUM** is QUORUM scoped to the local datacenter — strong within the DC, low latency because it avoids cross-region round trips, and the workhorse for multi-region deployments. **ALL** demands every replica: linearizable-ish, but a single node down means the operation fails, so it sacrifices the availability that was the whole point of Dynamo.

In CQL the consistency level is set per session or per statement:

```sql
-- Strong: read-your-writes guaranteed because 2 + 2 > 3
CONSISTENCY QUORUM;
INSERT INTO orders_by_user (user_id, created_at, order_id, status, total_cents)
    VALUES (550e8400-e29b-41d4-a716-446655440000, '2026-06-14 10:00:00',
            now(), 'PENDING', 9900);

SELECT * FROM orders_by_user
    WHERE user_id = 550e8400-e29b-41d4-a716-446655440000;

-- Eventually consistent: lowest latency, may miss the just-written row
CONSISTENCY ONE;
SELECT * FROM orders_by_user
    WHERE user_id = 550e8400-e29b-41d4-a716-446655440000;
```

The classic production pattern is `LOCAL_QUORUM` for both reads and writes in a single region: with `RF = 3`, that is `2 + 2 = 4 > 3`, so you get strong consistency within the datacenter while tolerating one node down, and you avoid the latency of cross-region coordination. Use `ONE` only for data where staleness is acceptable (analytics counters, feeds that can lag a few hundred milliseconds), and reserve `ALL` for almost nothing, because it trades away availability.

### Anti-entropy: what makes eventual consistency actually converge

`W + R ≤ N` writes are durable but may leave replicas temporarily divergent, so the system needs background mechanisms to converge. There are three, all inherited from Dynamo. **Hinted handoff**: if a replica is down during a write, the coordinator stores a "hint" and replays it when the node returns (a sloppy quorum). **Read repair**: when a read at QUORUM finds replicas disagree, the coordinator pushes the newest version to the stale replicas inline. **Anti-entropy repair** (`nodetool repair`): a scheduled process that builds Merkle trees over key ranges on each replica, compares them to find divergent ranges without shipping all the data, and reconciles them. Skipping repair is one of the most common operational sins, and it interacts dangerously with tombstones, as section 7 explains. For the full mechanism, see the [quorums, anti-entropy and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) deep dive.

### Where this sits in CAP and PACELC

The Dynamo lineage is the textbook example of an **AP** system in CAP terms: when a network partition splits the replicas, it chooses **availability** — it keeps accepting reads and writes on both sides and reconciles later — over **consistency**. That is the literal meaning of the paper's "always-writeable" goal. But CAP only describes behavior *during a partition*, and partitions are rare; the more useful lens for everyday operation is **PACELC**, which adds: *else* (when there is no partition), do you favor **latency** or **consistency**? Cassandra at `ONE` is PA/EL — available under partition, low-latency otherwise, both at the expense of consistency. Cassandra at `QUORUM` is PA/EC — still available under partition (a quorum may still be reachable), but consistency-favoring in normal operation. The whole point of tunable consistency is that you choose your PACELC posture *per query*, which no single-leader relational database lets you do. The [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) deep dive works through this taxonomy in full.

The subtlety that trips people up is that `W + R > N` gives you **strong consistency for a single key**, not linearizability across keys and not a global snapshot. Two reads of *different* keys at QUORUM can still observe states that no single point in time ever held, because each key's quorum is independent. If you need a consistent multi-key view, you need a transaction (LWT in Cassandra, `TransactWriteItems` in DynamoDB), and you should expect to pay for it. The mental model from Kleppmann's Chapter 5 is precise here: leaderless quorums give you per-object recency, not cross-object ordering — the latter is a consensus problem, which is the next section.

### DynamoDB's two consistency modes

DynamoDB exposes a simpler binary because each partition has a Paxos leader. A read is either **eventually consistent** (the default, served by any replica, half the cost) or **strongly consistent** (`ConsistentRead=True`, routed to the leader, full cost). There is no per-replica tuning because the leader is authoritative — this is the architectural payoff of DynamoDB choosing single-leader-per-partition over leaderless. The trade is that strong reads must go to the leader's availability zone, so they cannot be served from an arbitrary replica the way an eventually-consistent read can.

## 6. Lightweight transactions: Paxos when you truly need compare-and-set

**The senior rule of thumb: leaderless stores cannot do cheap conditional writes, because there is no leader to serialize them. Cassandra's lightweight transactions run a full Paxos round per operation and cost roughly 4× a normal write. Use them for the rare case that genuinely needs linearizable compare-and-set, and never on a hot path.**

Tunable consistency gives you read-your-writes, but it does *not* give you atomic compare-and-set across concurrent clients. "Create this user only if the username is not already taken" is a race: two clients can both read "absent" and both write. Solving it requires linearizability — a total order on the conflicting operations — and in a leaderless system the only way to get that is a consensus protocol. Cassandra implements **lightweight transactions (LWT)** using Paxos, exposed through `IF` clauses:

```sql
-- Atomic "claim this username if nobody has it"
INSERT INTO users (username, user_id, email)
    VALUES ('ada', 550e8400-e29b-41d4-a716-446655440000, 'ada@x.io')
    IF NOT EXISTS;

-- Conditional update: only flip to SHIPPED if currently PENDING
UPDATE orders_by_user
    SET status = 'SHIPPED'
    WHERE user_id = 550e8400-e29b-41d4-a716-446655440000
      AND created_at = '2026-06-14 10:00:00'
      AND order_id = 6f1b...
    IF status = 'PENDING';
```

The response includes an `[applied]` column telling you whether the condition held. This is genuinely useful — it is the only safe way to do uniqueness or state-machine transitions — but it is expensive, and the figure below shows why.

![Lightweight transactions add three extra Paxos round trips over a normal quorum write](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-9.webp)

A normal QUORUM write is one round trip: the coordinator sends to replicas and acknowledges at quorum. An LWT runs the full Paxos cycle: **prepare** (a ballot is proposed to the replicas, Paxos phase 1), **read** (the current value is read to evaluate the `IF` condition), **propose** (the new value is accepted, Paxos phase 2), and **commit** (the value is applied and Paxos state cleared). That is four round trips where a normal write needs one, which is why LWTs run roughly 4× slower and why their throughput collapses under contention on the same partition — concurrent ballots conflict and retry. The rules of thumb that keep teams out of trouble: never put an LWT on a hot path, never use them for high-contention counters, and if you find yourself reaching for them constantly, your data model is probably wrong and you should redesign so the invariant is enforced by the partition structure instead.

DynamoDB's equivalents are **conditional writes** (`ConditionExpression`, backed by the partition's Paxos leader and far cheaper than Cassandra's LWT because the leader already serializes the partition) and **transactions** (`TransactWriteItems`, up to a bounded number of items across the table with all-or-nothing semantics). DynamoDB transactions consume roughly 2× the capacity of a normal write and are limited in item count, so the same discipline applies: use them for genuine invariants, not as a default.

## 7. Tombstones and gc_grace_seconds: how deletes turn into time bombs

**The senior rule of thumb: in an LSM store a delete is a write, not a removal. The deleted data stays on disk as a tombstone until compaction runs past `gc_grace_seconds`, and a partition full of tombstones can make reads slower than if you had never deleted anything.**

This is the operational reality that surprises every team. Because SSTables are immutable, Cassandra cannot reach into an old SSTable and erase a row. To delete, it writes a special marker — a **tombstone** — that shadows the older value. A read must still scan past every tombstone in the relevant SSTables to know the row is gone, so a partition where you delete a lot can accumulate thousands of tombstones that the read has to skip over. Cassandra will log warnings past a tombstone threshold and abort a query past a hard limit (defaults around 1,000 warn / 100,000 fail) to protect itself, which means a delete-heavy queue table can start *failing reads* even though the live data is tiny.

Tombstones cannot be removed immediately, because of a correctness requirement. Suppose a replica was down when a delete happened. If the tombstone were purged before that replica came back and got repaired, the down replica's old (non-deleted) value could resurrect the row — a "zombie." So Cassandra keeps tombstones for `gc_grace_seconds` (default **864000 seconds = 10 days**) before compaction is allowed to purge them, on the assumption that you will run anti-entropy repair within that window so every replica learns about the delete. This couples two operational facts that bite when ignored: **if you do not run repair within `gc_grace_seconds`, you risk zombie data**, and **if you lower `gc_grace_seconds` to reclaim space faster, you shrink your repair window**. The settings live on the table:

```sql
ALTER TABLE messages WITH gc_grace_seconds = 432000;  -- 5 days; now repair must run < 5 days

-- TTL writes auto-expire and become tombstones at expiry:
INSERT INTO sessions (session_id, data) VALUES ('abc', '...')
    USING TTL 3600;  -- expires in 1 hour, then sits as a tombstone until gc_grace
```

The deepest anti-pattern here is the **queue or inbox modeled as delete-heavy rows in one partition**: you insert items, process and delete them, and the partition fills with tombstones at the front while you read from it, so reads get progressively slower until they fail. Discord hit a flavor of this — their final 0.0001% of migration stalled on "uncompacted tombstone ranges" in Cassandra. The fix for queue-like workloads is to use time-bucketed partitions (so old, fully-tombstoned buckets age out wholesale) or `TWCS` (Time Window Compaction Strategy), which groups SSTables by time window so an entire expired window can be dropped at once without scanning. Modeling deletes is as much a part of wide-column design as modeling reads.

## 8. The hot and unbounded partition: the failure mode that bites everyone

**The senior rule of thumb: a partition key that concentrates traffic or grows without bound puts all of that load on one node, and no amount of cluster capacity will save you, because a single partition cannot be split across nodes. Bound every partition by size and spread every partition by traffic — at modeling time.**

This is where the abstract "pick a good partition key" advice becomes a production incident. The figure below contrasts the unbounded design with the bucketed fix.

![Unbounded partition keys overload one node while time bucketing spreads writes across many](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-8.webp)

The left column is the naive Discord schema, keyed on `channel_id` alone. Two failures compound. First, **unbounded growth**: a busy channel accumulates messages forever, so its single partition grows past a gigabyte, GC pauses lengthen, compaction slows, and p99 spikes toward 200 ms — exactly the symptom Discord reported. Second, **hot traffic**: an `@everyone` ping in a 500,000-member server drives a flood of reads and writes at *one* partition key, so all that load hits the three replicas of that one range, and at QUORUM consistency the latency spike ripples across the whole cluster. The right column is the fix: the composite key `(channel_id, bucket)` where `bucket` is a 10-day time window. Now a single channel's data is spread across many bucket-partitions on many nodes, each partition stays under ~100 MB so compaction is fast and p99 holds near 5 ms, and Discord layered **request coalescing** (collapsing simultaneous reads of the same data into one database query) and consistent-hash routing on top to de-duplicate hot reads.

The hard limits to internalize: keep partitions **under ~100 MB and under ~100,000 rows** as a rule of thumb (hard problems start well before the absolute limits), and make sure no single partition key can grow without bound. The standard tool is **bucketing** — appending a time window or a numeric shard to the partition key — which trades a little query complexity (you may have to query multiple buckets to assemble a full result) for bounded, well-distributed partitions.

### The same hazard in DynamoDB, and how it auto-mitigates

DynamoDB has the identical physics but a different toolset. Originally, throughput was provisioned per-table and divided evenly across partitions, so a hot key on one partition would get throttled even though the table had spare capacity overall — the infamous "throughput dilution" problem. The [2022 DynamoDB paper](https://www.usenix.org/conference/atc22/presentation/elhemali) traces how AWS fixed this in stages: **burst capacity** (let a partition borrow unused capacity short-term), **adaptive capacity** (rebalance provisioned throughput toward hot partitions, eventually isolating a hot item onto its own partition), and finally **global admission control (GAC)**, which decouples admission from per-partition limits so any partition can absorb a spike as long as the table's budget allows. On-demand tables remove the provisioning step entirely. The 2022 paper reports the payoff at scale: during the 66-hour 2021 Prime Day, DynamoDB peaked at **89.2 million requests per second** with single-digit-millisecond latency.

But adaptive capacity has a hard limit that mirrors Cassandra's: it can isolate a hot *partition*, but it cannot split a single hot *key*, because every item with the same partition key value must live in the same partition. If your access pattern is "everyone reads `GLOBAL#counter`," no managed magic saves you — you must shard the key yourself (`GLOBAL#counter#<0-9>`, scatter writes, gather reads), exactly as you would in Cassandra. The lesson generalizes: managed services automate the *mechanical* response to hotspots, but the *modeling* response — choosing keys that do not create hotspots in the first place — is irreducibly yours.

## 9. DynamoDB internals: partitions, leaders, GSI, and Streams

**The senior rule of thumb: DynamoDB is Dynamo's partitioning with single-leader replication bolted on per partition, wrapped in a fully-managed control plane. Understanding the partition-and-leader structure is what lets you reason about its costs, its consistency, and its secondary indexes.**

The figure below shows the request lifecycle through DynamoDB's internals as described in the 2022 paper.

![DynamoDB routes by partition key to a Paxos leader and replicates to global secondary indexes](/imgs/blogs/cassandra-and-dynamodb-leaderless-deep-dive-7.webp)

A `PutItem` or `Query` arrives at a **request router**, which consults **global admission control** to decide whether to admit or throttle the request, then hashes the partition key to route to the owning **partition** (which splits as it grows past ~10 GB). Each partition is a replication group of three replicas spread across availability zones, coordinated by **Multi-Paxos**: one replica is the **leader** and serves all writes and strongly-consistent reads. A clever optimization from the paper: alongside full **storage replicas** (which hold the B-tree plus write-ahead log), DynamoDB keeps **log replicas** that store only recent WAL entries, so when a storage replica fails the group can recover a quorum in seconds rather than minutes by spinning up a log replica first. Writes are acknowledged once a quorum of the replication group durably persists the log record. Asynchronously, the leader streams changes to any **global secondary indexes**, which is why GSIs are eventually consistent.

### RCU and WCU: the capacity unit cost model

DynamoDB meters throughput in capacity units, and getting the arithmetic right is the difference between a cheap table and a surprise bill. One **read capacity unit (RCU)** buys one strongly-consistent read per second of an item up to 4 KB (or two eventually-consistent reads, since they cost half). One **write capacity unit (WCU)** buys one write per second of an item up to 1 KB. Larger items consume proportionally more: a 10 KB item costs 3 RCU for a strong read (10 KB rounds up to 12 KB = three 4 KB blocks) and 10 WCU to write. This rounding is why item size discipline matters — fat items multiply cost on every access. On-demand mode bills per request instead of per provisioned unit, trading a higher per-request price for zero capacity planning.

Work a concrete example. Say a feed table serves 5,000 reads/second of 8 KB items and 1,000 writes/second of 2 KB items. Each 8 KB strong read is 2 RCU (8 KB = two 4 KB blocks), so reads need `5000 × 2 = 10,000 RCU`; if you can accept eventually-consistent reads, halve that to **5,000 RCU**. Each 2 KB write is 2 WCU, so writes need `1000 × 2 = 2,000 WCU`. Now add a GSI that projects the full item and is written on every base write: that GSI needs roughly the same **2,000 WCU** of its own, separately provisioned — forget it and the GSI throttles the base table, as case study 6 shows. The arithmetic also reveals the two biggest levers: shrinking items (split rarely-read attributes into a separate item so hot reads stay under 4 KB) and using eventually-consistent reads wherever staleness is tolerable each halve the read bill. None of this is visible in Cassandra, where you pay for nodes and the cost shows up as CPU and disk pressure rather than a line item — but the underlying physics (item size drives I/O) is identical.

### GSI vs LSI: choose the right index

DynamoDB has two secondary index types, and the distinction is sharp. A comparison:

| Property | Local Secondary Index (LSI) | Global Secondary Index (GSI) |
| --- | --- | --- |
| Partition key | Same as base table | Any attribute (different axis) |
| Sort key | Different from base | Any attribute |
| Consistency | Strong reads available | Eventually consistent only |
| Throughput | Shares base table's | Own RCU/WCU provisioning |
| Created | Only at table creation | Anytime, added later |
| Size limit | Item collection ≤ 10 GB | No item-collection size limit |
| Storage | Co-located with base partition | A separate replicated table |

The practical guidance from [the index docs and DeBrie's patterns](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Constraints.html): reach for an **LSI** only when you need a strongly-consistent alternate sort within the same partition key and you can accept the 10 GB item-collection cap and the create-time-only constraint. Reach for a **GSI** for essentially everything else — repartitioning data along a new axis, which is the central move of single-table design. The crucial gotcha is the GSI's eventual consistency: because the leader replicates to the GSI asynchronously, a read immediately after a write may not see the change, so never use a GSI for a read-your-writes invariant. And because a GSI is physically a separate table, it has its own throughput that you must provision and that, if exhausted, can back-pressure writes to the base table.

### DynamoDB Streams: the change-data-capture hook

Every DynamoDB table can emit a **Stream** — an ordered, 24-hour log of item-level changes (old image, new image, or both) — which is the integration point for everything event-driven: triggering Lambda functions, fanning out to search indexes, replicating to other stores, or building the read side of a CQRS system. Streams are how you escape the "I can't add an access pattern" rigidity: you consume the stream and project the data into a new table or a different store entirely. Global Tables (multi-region active-active replication) are built on Streams internally. If single-table design is how you serve known access patterns, Streams are how you serve the ones you discover later.

## 10. Choosing these over a relational database

**The senior rule of thumb: choose a wide-column store when your access patterns are known and limited, your scale exceeds one machine, and you value availability and predictable latency over flexible querying and multi-row transactions. Choose relational when you need ad-hoc queries, joins, and ACID more than you need horizontal scale.**

The decision is not about which database is "better" — it is about which trade matches your workload. A comparison of the dimensions that actually drive the choice:

| Dimension | Relational (Postgres/MySQL) | Cassandra/ScyllaDB | DynamoDB |
| --- | --- | --- | --- |
| Query flexibility | High — ad-hoc SQL, joins | Low — query-first tables | Low — access-pattern-first |
| Horizontal write scale | Hard (sharding is manual) | Native (add nodes) | Native (managed, auto-split) |
| Consistency default | Strong (single node) | Tunable per query | Eventual, opt-in strong |
| Multi-row transactions | Full ACID | LWT only, expensive | TransactWriteItems, bounded |
| Operational burden | Moderate | High (repair, compaction) | Low (fully managed) |
| Cost model | Instance + storage | Nodes + ops headcount | Per-request / per-capacity |
| Tail latency at scale | Degrades, GC/lock contention | Low and flat (esp. ScyllaDB) | Low and flat (single-digit ms) |
| Best fit | OLTP with rich queries | Write-heavy, huge, multi-DC | Serverless, spiky, managed |

The honest summary: most applications should start on Postgres and stay there far longer than the hype suggests. You move to a wide-column store when you have *outgrown* a single machine's write capacity, when you need multi-datacenter active-active availability, or when your latency SLA demands flat tails that a contended relational instance cannot hold — and when your access patterns are stable enough that the query-first rigidity is acceptable. Between Cassandra and DynamoDB, the deciding factor is usually operational: DynamoDB if you want zero database operations and live on AWS and can model within its limits; Cassandra or ScyllaDB if you need multi-cloud or on-prem, want full control, and have the operational maturity to run repair, manage compaction, and tune the JVM (or sidestep the JVM with ScyllaDB).

And between Cassandra and ScyllaDB specifically, the choice has narrowed to a small set of concrete factors. ScyllaDB is protocol- and CQL-compatible, so the data model and application code carry over unchanged; what you gain is the shard-per-core architecture's higher throughput per node and the elimination of JVM GC pauses, which is why Discord's tail latency improved by an order of magnitude on the same model. What you weigh against that is ecosystem maturity — the Apache Cassandra community, tooling, and managed offerings (including AWS Keyspaces) are broader — and the fact that ScyllaDB's deepest performance benefits assume you can give it dedicated cores and NUMA-aware tuning. The pragmatic default for a new self-managed deployment that is latency-sensitive and runs at meaningful scale is ScyllaDB; the default for a team already deep in the Cassandra ecosystem, or one that values the largest community and tooling surface, is Cassandra. Either way the modeling discipline in this article applies verbatim, because both are the same machine underneath.

## Case studies from production

The patterns above stay theoretical until they bite. Here are eleven named incidents — some public, some composites of the failure modes these systems produce — each with the symptom, the wrong first hypothesis, the actual root cause, the fix, and the lesson.

### 1. Discord: trillions of messages, three databases

Discord [migrated MongoDB → Cassandra → ScyllaDB](https://discord.com/blog/how-discord-stores-trillions-of-messages) over five years. The symptom on Cassandra by 2022 was unpredictable latency, frequent on-call pages, and a cluster grown to 177 nodes. The wrong hypothesis was "we need more Cassandra nodes." The actual root cause was twofold: JVM garbage-collection pauses caused latency spikes, and hot partitions (busy channels, `@everyone` in huge servers) concentrated load that QUORUM reads amplified across the cluster. The fix was the `(channel_id, bucket)` partition scheme to bound partitions, a Rust **data-services** layer doing request coalescing and consistent-hash routing to de-duplicate hot reads, and ultimately a migration to ScyllaDB to eliminate GC entirely. The result: **177 → 72 nodes**, p99 from 40–125 ms to ~15 ms, and a custom Rust migrator moving **3.2 million messages/second** to finish in **9 days**. The lesson: the data model (bucketing) and the application layer (coalescing) did as much work as the database swap; do not expect a faster engine to fix a hot-partition model.

### 2. The @everyone hot partition

A chat platform keyed messages on `channel_id` alone and ran fine until a 500k-member community used `@everyone`. The symptom was cluster-wide latency spikes correlated with announcements. The wrong hypothesis was a network or GC issue, because the spike was global. The actual root cause: every read and write for that one channel hit the same partition's three replicas, and at QUORUM the coordinator waited on those overloaded replicas, so requests queued behind them and the latency bled into unrelated queries sharing those nodes. The fix was time-bucketing the partition key plus request coalescing. The lesson: in a leaderless store, a single hot partition is not a local problem — quorum coordination spreads its pain across the whole cluster.

### 3. The tombstone-choked queue

A team modeled a work queue as a Cassandra table: insert jobs, process, `DELETE`. The symptom was reads on the queue table getting slower over days, then failing with `TombstoneOverwhelmingException`. The wrong hypothesis was "the queue is too big," but the live row count was tiny. The actual root cause: every processed job left a tombstone at the front of the partition, and a read for the next job had to scan past tens of thousands of tombstones before reaching live data; past the 100,000 hard limit, Cassandra aborted the query. The fix was to stop using Cassandra as a queue — moving the queue to a purpose-built broker — and where queue-like tables were unavoidable, switching to time-window compaction with bucketed partitions so fully-tombstoned windows drop wholesale. The lesson: deletes are writes; a delete-heavy hot partition is a tombstone time bomb.

### 4. The skipped repair and the zombie

An operations team disabled `nodetool repair` because it was "expensive and caused load." The symptom, weeks later, was deleted records reappearing. The wrong hypothesis was an application bug re-inserting data. The actual root cause: a replica had been down during several deletes; because repair never ran, that replica never learned of the deletes, and after `gc_grace_seconds` (10 days) the tombstones were compacted away on the other replicas — so when the stale replica next served a read, its old non-deleted value resurrected the row as a zombie. The fix was to re-enable scheduled repair on a cadence shorter than `gc_grace_seconds` and run a full repair to reconcile. The lesson: repair within `gc_grace_seconds` is not optional; it is the contract that makes deletes durable.

### 5. The DynamoDB throughput-dilution surprise

A team provisioned a DynamoDB table at 10,000 WCU and still got throttled at a fraction of that. The symptom was `ProvisionedThroughputExceededException` while the table-level metrics showed plenty of unused capacity. The wrong hypothesis was "AWS is mis-metering us." The actual root cause (pre-adaptive-capacity behavior): provisioned throughput was divided evenly across partitions, and a hot partition key drove most traffic to one partition that got only its even slice — say 500 WCU — so it throttled while the other partitions sat idle. The fix was to spread writes across more partition-key values (sharding the hot key) and, on modern DynamoDB, to lean on adaptive capacity and global admission control, which now isolate hot partitions automatically. The lesson: throughput is partition-local in effect even when you provision it table-wide; model keys to spread load.

### 6. The GSI back-pressure outage

A service added a GSI to a busy DynamoDB table and under-provisioned the GSI's write capacity relative to the base table. The symptom was *base table* writes suddenly throttling. The wrong hypothesis was a base-table capacity problem. The actual root cause: a GSI is physically a separate replicated table, and when its write capacity is exhausted, DynamoDB back-pressures the base table to avoid an unbounded replication backlog — so the under-provisioned index throttled the table it indexed. The fix was to provision the GSI's WCU to match the base table's effective write rate (accounting for projected attributes) or move it to on-demand. The lesson: a GSI is not free; it is a second table with its own throughput that can throttle the first.

### 7. The "strong read" that wasn't, via GSI

A checkout flow wrote an order and immediately read it back through a GSI to confirm, and intermittently the read returned nothing. The symptom was sporadic "order not found" errors right after creation. The wrong hypothesis was a write failure. The actual root cause: GSIs are eventually consistent because the partition leader replicates to them asynchronously; a read against the GSI microseconds after the write raced the replication and missed it. The fix was to read back through the base table's primary key (where strong reads are available) rather than the GSI. The lesson: never use a GSI for read-your-writes; its eventual consistency is structural, not a tuning knob.

### 8. The unbounded partition that ate a node

A metrics system keyed time-series on `metric_name` with the timestamp as clustering key — no bucket. The symptom was one node in the ring running out of disk and timing out reads while its peers were healthy. The wrong hypothesis was a failing disk on that node. The actual root cause: the most popular metric's partition grew without bound — billions of points under one partition key — concentrating on the three replicas owning that token, and reads of recent points had to seek into a multi-gigabyte partition. The fix was to add a time bucket to the partition key (`(metric_name, day)`) and use TWCS so old windows compact and age out efficiently. The lesson: any time-series or log-like table needs a time bucket in the partition key from day one; unbounded partitions are a design defect, not a capacity issue.

### 9. The single-hot-key counter

A leaderboard incremented a global counter at one DynamoDB partition key, and a Cassandra analog used a counter column on one partition. The symptom in both was the counter becoming a throughput wall as traffic grew, despite adaptive capacity (DynamoDB) and spare cluster capacity (Cassandra). The wrong hypothesis was "the managed system will rebalance it." The actual root cause: a single partition-key value cannot be split across partitions or nodes, so all increments serialize on one partition — adaptive capacity can isolate a hot partition but not subdivide a single key. The fix was write-sharding: spread increments across `counter#<0-N>` sub-keys and sum the shards on read. The lesson: the one thing no managed magic can do is split a single key; sharded counters are the canonical workaround.

### 10. The cross-region QUORUM that tripled write latency

A team ran Cassandra across `us_east` and `eu_west` with `RF = 3` in each datacenter and used `CONSISTENCY QUORUM` everywhere for safety. The symptom was write latency hovering around 150 ms — far worse than the single-digit milliseconds they expected. The wrong hypothesis was a slow disk or undersized nodes. The actual root cause: a cluster-wide `QUORUM` with two datacenters and `RF = 3` each means `N = 6`, so quorum is 4 replicas, which *forces* the coordinator to wait on acknowledgements from the *other* datacenter across the transatlantic link on every write — adding a full cross-region round trip. The fix was to switch to `LOCAL_QUORUM`, which requires a quorum only within the writing datacenter (`2` of the local `3`) while replication to the remote datacenter proceeds asynchronously. Latency dropped to single digits, and the data still replicated globally. The lesson: in multi-datacenter Cassandra, `LOCAL_QUORUM` is almost always the right consistency level; a plain `QUORUM` silently buys you the latency of your slowest region on every operation.

### 11. The migration that stalled on the last 0.0001%

During Discord's [ScyllaDB cutover](https://discord.com/blog/how-discord-stores-trillions-of-messages), the custom Rust migrator streamed at 3.2 million messages/second and was nearly done — and then the final fraction would not move. The symptom was a tiny tail of partitions that the migrator could not read out of Cassandra. The wrong hypothesis was a bug in the migrator. The actual root cause: those partitions held large ranges of uncompacted tombstones, and reading them tripped Cassandra's tombstone-overwhelming protection, so the source database itself refused to return the data. The fix was targeted manual compaction on those specific partitions to purge the tombstones, after which they read out cleanly. The lesson is a fitting capstone: even a perfectly executed migration runs into the tombstone reality, because deletes-as-writes is not an edge case in these systems — it is the foundation, and it shapes everything from query latency to migration tooling.

## When to reach for a leaderless wide-column store

Reach for Cassandra, ScyllaDB, or DynamoDB when:

- **Your write throughput has outgrown a single machine** and manual relational sharding is becoming the dominant engineering cost. Horizontal write scale is the core thing these systems give you that a single relational node cannot.
- **You need multi-datacenter or multi-region availability** with local-quorum reads and writes, surviving the loss of a whole region or rack without downtime — the original Dynamo use case.
- **Your latency SLA demands flat tail latency at scale**, where a contended relational instance's GC pauses or lock contention would blow your p99. ScyllaDB and DynamoDB both deliver single-digit-millisecond tails at very high throughput.
- **Your access patterns are known and stable**, so you can model one table per query and accept that adding a genuinely new pattern is a project, not a `WHERE` clause.
- **You want managed operations (DynamoDB specifically)** and live on AWS, valuing zero database administration over portability and willing to model within RCU/WCU and partition limits.

Skip them, and stay relational, when:

- **You need ad-hoc queries and joins** — analytics, reporting, "let me slice this data a new way every week." Wide-column stores punish unanticipated access patterns; a relational database or a columnar [OLAP store](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) is the right tool.
- **You need real multi-row ACID transactions** as a routine pattern. LWTs and `TransactWriteItems` exist but are expensive and bounded; if transactions are central, that is a relational signal.
- **Your data fits comfortably on one machine** (or a primary plus replicas) and will for the foreseeable future. The operational complexity of repair, compaction, and key modeling is pure overhead until you actually need horizontal scale. Premature distribution is a classic self-inflicted wound.
- **Your team lacks the operational maturity to run a distributed store** and you are not using a managed one. Cassandra rewards expertise and punishes neglect — skipped repair, ignored compaction, and bad keys all become incidents.

> The deepest mistake teams make with these systems is bringing relational instincts: modeling entities instead of queries, expecting joins, treating deletes as free, and assuming a bigger cluster fixes a bad key. Every one of those instincts is correct in Postgres and wrong here. Learn the new cost model first; the schema follows from it.

## Further reading

- [Dynamo: Amazon's Highly Available Key-value Store](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/dynamo.pdf) (SOSP 2007) and Werner Vogels' [original announcement](https://www.allthingsdistributed.com/2007/10/amazons_dynamo.html) — the source of the whole lineage.
- [Amazon DynamoDB: A Scalable, Predictably Performant, and Fully Managed NoSQL Database Service](https://www.usenix.org/conference/atc22/presentation/elhemali) (USENIX ATC 2022) — how the managed system actually works, from partitions to global admission control.
- [How Discord Stores Trillions of Messages](https://discord.com/blog/how-discord-stores-trillions-of-messages) and [the ScyllaDB migration writeup](https://www.scylladb.com/tech-talk/how-discord-migrated-trillions-of-messages-from-cassandra-to-scylladb/) — the best public account of modeling and operating these systems at scale.
- [Alex DeBrie on single-table design](https://www.alexdebrie.com/posts/dynamodb-single-table/) and *The DynamoDB Book* — the canonical guide to access-pattern-first modeling.
- [ScyllaDB's shard-per-core architecture](https://www.scylladb.com/product/technology/shard-per-core-architecture/) — why a C++ rewrite beats the JVM for this workload.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapters 3 (storage and wide-column), 5 (leaderless replication), and 6 (partitioning) — the conceptual spine under all of it.
- Sibling deep dives on this blog: [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning), [quorums, anti-entropy and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair), and [single-leader, multi-leader and leaderless replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless).

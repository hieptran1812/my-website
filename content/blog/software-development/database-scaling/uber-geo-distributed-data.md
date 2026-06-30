---
title: "How Uber Built Geo-Distributed Storage on MySQL: Schemaless to Docstore"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A walk through Uber's storage evolution — Postgres monolith to Schemaless on sharded MySQL to Docstore — and the recurring principle of building distribution on top of a boring, reliable single-node engine you fully control."
tags: ["uber", "schemaless", "docstore", "mysql", "sharding", "ringpop", "multi-region", "distributed-databases", "database-scaling", "consistent-hashing"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 37
---

The most expensive mistake a fast-growing company can make with its database is to believe that the database is supposed to solve distribution for it. You reach a scaling wall, you panic, and you go shopping for a system whose marketing promises to "just scale" — a distributed SQL engine, a wide-column store, a managed planet-scale service. Then you spend the next two years discovering that the operational model of that system is alien to your team, that its failure modes are subtle and global, and that when it breaks at 3 a.m. nobody on call understands what the cluster is actually doing.

Uber went the other way, and the contrast is the whole lesson of this post. Faced with a workload that was outgrowing a single Postgres instance, Uber did not buy a distributed database. They took the most boring, well-understood, single-node engine on the planet — MySQL with InnoDB — and built the distribution layer themselves, on top, in application code they controlled end to end. That layer became **Schemaless**, an append-only triplestore that shards across thousands of MySQL instances. Years later it grew into **Docstore**, a general-purpose distributed document database that still has MySQL nodes at the bottom of the stack. Around all of it sits **Ringpop**, a library that does consistent-hash request routing in the application tier.

![Uber's storage evolution timeline from Postgres monolith through Schemaless to Docstore](/imgs/blogs/uber-geo-distributed-data-1.webp)

The diagram above is the mental model for the entire article: every step in Uber's storage history kept a simple, reliable single-node engine at the bottom and added sharding, geo-availability, and eventually a query layer *on top* of it. The smarts moved up the stack into code Uber owned; the storage node stayed dumb, fast, and operationally familiar. This is the opposite of the instinct most teams have, which is to push the smarts *down* into a clever database and hope it works. We are going to tour that history, build small working models of the key mechanisms, and end with a set of named production lessons.

A note on sourcing before we start. Several claims here — especially Uber's reasons for moving off Postgres — come from Uber's own engineering blog posts and were vigorously debated when published. I will mark those as **Uber's stated reasons** rather than universal truths, because the Postgres community pushed back hard on several of them, and many of the specific pain points were tied to the Postgres versions of that era. The architecture of Schemaless, Docstore, and Ringpop is much less contested; those are described in Uber's published design write-ups and open-source code.

## Why "build distribution on top of a simple node" is different

Most engineers' default mental model is a spectrum from "single database" to "distributed database," and scaling means sliding rightward along it. Uber's model is a different axis entirely: keep the *node* simple and move *distribution* into a layer you own. The table makes the contrast concrete.

| Question | The "buy a distributed DB" instinct | What Uber actually did |
| --- | --- | --- |
| Where does sharding live? | Inside the database engine, opaque to you | In an application-tier layer (Schemaless worker / Docstore query engine) you wrote |
| What is a storage node? | A peer in a complex cluster protocol | A plain MySQL master with two replicas — boring and familiar |
| Who owns routing? | The database's internal gossip/placement | Your code: a shard map, later Ringpop's hash ring |
| What does on-call debug? | A distributed consensus you didn't write | A single MySQL box and a thin routing layer |
| How do you add capacity? | Trust the cluster to rebalance | Add storage nodes and remap shards explicitly |
| Failure blast radius | Can be cluster-wide and correlated | Bounded to a shard / storage node |

The right column is not obviously better. It means you have to *build and operate* the distribution layer, which is real, ongoing engineering cost. The argument for it is that the cost is **legible**. When a single MySQL node misbehaves, every backend engineer at the company already knows how to read its slow query log, check replication lag, and reason about its buffer pool. When the routing layer misbehaves, it is code in your monorepo with your tests around it. Nothing in the hot path is a black box.

> The cheapest distributed system to operate is the one whose every component you could explain to a new hire on a whiteboard. Uber chose explainability over cleverness, and then scaled the explainable thing.

That philosophy only works if the single-node engine you pick is genuinely reliable and genuinely well-understood. Which is exactly why the first big decision — the one everyone remembers — was about the engine itself.

## 1. The starting point: a Postgres monolith and why Uber moved off it

**Senior rule of thumb: a database migration is never about the database being "bad"; it is about a specific mismatch between the engine's internals and your specific workload.**

In Uber's early days the primary online store was Postgres. As the company grew, the engineering team published a now-famous post, "Why Uber Engineering Switched from Postgres to MySQL," laying out the internals-level reasons they moved the relational workload onto MySQL/InnoDB. These are **Uber's stated reasons**, and they were specific to Postgres as it existed at the time:

- **Write amplification through the on-disk format.** In Postgres, every row version (tuple) has a physical location, and *every* secondary index entry points at that physical location (the `ctid`). When you update a single column of a row, Postgres writes a new tuple, and then **every** index on that table — even indexes on columns you did not touch — must get a new entry pointing at the new tuple's location. A one-column logical update fans out into many physical index writes.
- **Replication carries the amplification.** Postgres streaming replication ships the write-ahead log, which records the *physical* changes. So all those extra index writes don't just hit the primary's disk; they get serialized into the WAL and shipped over the network to every replica. For Uber, with replicas in geographically distant data centers, that verbose replication stream was a real bandwidth and lag concern.
- **Connection handling.** Each Postgres connection was backed by an OS process, making large connection counts expensive. (This was before the connection-pooling and process-model improvements of later Postgres versions, and tools like PgBouncer were the standard mitigation.)
- **Replica MVCC behavior.** The interaction between long-running queries on a replica and vacuum on the primary could cause issues, which complicated read-replica usage.

MySQL with InnoDB has a different design that sidestepped the first point for Uber's pattern: InnoDB secondary indexes store the **primary key** of the row, not a physical location. So an update only has to touch the indexes on the columns that actually changed; an update to one column leaves unrelated indexes alone. The replication story is also different — MySQL's logical/row-based replication ships compact change records rather than physical page deltas.

It is essential to be honest here: the Postgres community contested several of these points, noting that some were version-specific, that the index-update behavior has nuance (HOT updates can avoid index churn when no indexed column changes and the new tuple fits on the same page), and that the connection-handling critique predated significant improvements. The durable takeaway is not "Postgres is worse than MySQL." It is that **Uber profiled their actual workload, found a concrete internals-level mismatch, and picked the engine whose on-disk and replication model fit that workload** — and that engine happened to be the one their team could operate in their sleep. If you want the deeper relational-engine version of this argument, the differences in multi-version concurrency control are exactly the kind of thing covered in [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb).

The second-order point that gets lost: switching the engine did *not* solve scaling. A single MySQL box has the same fundamental ceiling a single Postgres box does. The engine choice mattered because of what came next — Uber was about to shard that engine across thousands of instances, and they wanted the per-node behavior to be predictable when they did. The real scaling story is the layer they built on top.

## 2. Schemaless: an append-only triplestore on sharded MySQL

**Senior rule of thumb: if you can make your write path append-only, an enormous amount of distributed-systems pain (in-place update conflicts, write contention, complex replication semantics) simply evaporates.**

Schemaless was Uber's first home-grown distributed datastore, built initially for the **trip store** — the record of every Uber trip, which is about as mission-critical and write-heavy as data gets. Its design is, in Uber's own description, "an append-only sparse three-dimensional persistent hash map," conceptually similar to Google's Bigtable. The unit of data is the **cell**, and the cell is the entire trick.

![The Schemaless cell data model showing the row key, column name, ref key triple mapping to an immutable JSON body, versioned in a MySQL entity table](/imgs/blogs/uber-geo-distributed-data-2.webp)

A cell is **immutable**. Once written, it can never be overwritten or deleted. It is addressed by a three-part key:

- **`row_key`** — a UUID-like identifier (for the trip store, the trip's UUID). The row key is what the system hashes to pick a shard, so all cells for one entity live together.
- **`column_name`** — a named slot, like `BASE` or `STATUS`. Think of it as a column family in Bigtable terms: a logical grouping of related data within the row.
- **`ref_key`** — an integer version. Higher ref keys are newer. This is how Schemaless does updates without mutating: to "change" a cell you `INSERT` a new cell with the same `row_key` and `column_name` but a higher `ref_key`.

The cell body is a JSON object, stored compressed — Uber serialized it with MessagePack and compressed it with ZLib before persisting the blob. Reading "the current state" of a trip means selecting the cells for that row key and column name and keeping the one with the highest ref key.

Here is a minimal, runnable model of the cell store in Python so the data model stops being abstract. This is a teaching model, not Uber's code, but the semantics match.

```python
import json
import time
from dataclasses import dataclass

@dataclass(frozen=True)   # frozen = immutable, which is the whole point
class Cell:
    row_key: str          # e.g. "trip:8f3a"  -> picks the shard
    column_name: str      # e.g. "BASE"       -> a named blob slot
    ref_key: int          # version; higher == newer
    body: bytes           # MessagePack+ZLib in prod; JSON bytes here
    created_at: float

class SchemalessLikeStore:
    """An append-only triplestore. Writes only ever INSERT; nothing
    is updated or deleted in place. 'Latest' = max(ref_key)."""

    def __init__(self):
        # In prod this is a row in a MySQL `entity` table on one shard.
        self._cells: list[Cell] = []

    def put(self, row_key: str, column_name: str, obj: dict) -> int:
        # Compute the next ref_key for this (row_key, column_name).
        existing = [
            c.ref_key for c in self._cells
            if c.row_key == row_key and c.column_name == column_name
        ]
        next_ref = (max(existing) + 1) if existing else 1
        self._cells.append(Cell(
            row_key=row_key,
            column_name=column_name,
            ref_key=next_ref,
            body=json.dumps(obj).encode(),     # stand-in for msgpack+zlib
            created_at=time.time(),
        ))
        return next_ref   # idempotent retries can re-assert the same ref

    def get_latest(self, row_key: str, column_name: str) -> dict | None:
        cells = [
            c for c in self._cells
            if c.row_key == row_key and c.column_name == column_name
        ]
        if not cells:
            return None
        newest = max(cells, key=lambda c: c.ref_key)   # ORDER BY ref_key DESC LIMIT 1
        return json.loads(newest.body)

    def history(self, row_key: str, column_name: str) -> list[dict]:
        cells = sorted(
            (c for c in self._cells
             if c.row_key == row_key and c.column_name == column_name),
            key=lambda c: c.ref_key,
        )
        return [json.loads(c.body) for c in cells]

store = SchemalessLikeStore()
store.put("trip:8f3a", "BASE", {"status": "requested"})
store.put("trip:8f3a", "BASE", {"status": "on_trip"})
store.put("trip:8f3a", "BASE", {"status": "completed", "fare": 24.50})

assert store.get_latest("trip:8f3a", "BASE")["status"] == "completed"
assert len(store.history("trip:8f3a", "BASE")) == 3   # full audit trail, free
```

Three properties fall out of this design that are worth dwelling on:

1. **Append-only writes are trivially idempotent and trivially auditable.** A retried write that re-asserts the same `(row_key, column_name, ref_key)` is a no-op — you can hammer the write path on a flaky network without corrupting state. And because nothing is deleted, you get a complete version history of every trip for free, which is gold for debugging disputes and for after-the-fact analytics.
2. **The schema lives in your application, not the database.** The body is an opaque JSON blob to MySQL. Adding a field to a trip is a code change, not an `ALTER TABLE` that locks a giant table. ("Schemaless" is the name for a reason — the storage layer enforces no schema.) The cost, of course, is that all validation and migration logic moves into your services.
3. **MySQL becomes a dumb, fast key-value engine.** Look at what the storage node actually does: `INSERT` rows into a table, `SELECT` by an indexed triple. No joins, no complex transactions across rows, no foreign keys. This is the simplest possible thing to ask of a database, which is exactly why it scales and why it never surprises you in production.

### How it maps onto MySQL

Each cell is one row in a MySQL **entity table**. The columns are roughly: `added_id` (auto-increment integer primary key, defining insertion order), `row_key`, `column_name`, `ref_key`, `body` (the compressed blob), and `created_at`. There is a compound index on `(row_key, column_name, ref_key)` so the "give me the latest version of this cell" query is a fast index range scan. The `added_id` primary key matters more than it looks: it gives a total order of writes on that shard, which the trigger system (below) consumes.

### Sharding across thousands of MySQL instances

**Senior rule of thumb: pick a fixed, large shard count up front and map shards to machines — never hash directly to machines, or you will reshard the whole world every time you add a box.**

This is the move that turns one MySQL into a fleet. Uber divides the keyspace into a **fixed number of shards — typically configured at 4096** — and a cell is assigned to a shard purely by hashing its `row_key`. Shards are then *mapped* to physical **storage nodes**. A storage node is a MySQL cluster: in Uber's design, **one master and two minions** (replicas), deliberately spread across data centers.

![Schemaless topology showing a stateless worker hashing the row key to one of 4096 shards, mapped to a storage node of one MySQL master and two minions, with writes to the master and reads to either minion](/imgs/blogs/uber-geo-distributed-data-3.webp)

The indirection — `row_key → shard → storage node` instead of `row_key → storage node` — is the single most important sharding decision in the system, and it is the same lesson as [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) and Instagram's [ID sharding in Postgres](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres). With 4096 logical shards and, say, 16 storage nodes, each node owns 256 shards. When you add capacity, you do not rehash every key; you move some *shards* (whole ranges) from existing nodes to the new node and update the shard map. The number of keys that change ownership is bounded by the shards you move, not by the total keyspace. This is precisely the consistent-hashing-of-buckets idea, and it is why [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) shows up again at the Ringpop layer later.

A stateless **worker** process sits in front of the storage nodes. It receives HTTP requests from service clients, hashes the row key to find the shard, looks up the storage node for that shard, and routes the request: **writes go to the master; reads can go to any node in the cluster, including the minions.** Because the worker is stateless and the routing is just a hash plus a map lookup, you can run as many workers as you need behind a load balancer.

The worker is intentionally a *thin* layer. Uber's own framing is that "Schemaless itself is a relatively thin layer on top of MySQL for routing requests to the right database." All the heavy lifting — durability, the actual reads and writes, replication within a storage node — is done by plain MySQL. This is the philosophy made literal: the clever part is small and stateless; the reliable part is boring and stateful.

#### Second-order optimization: the fixed shard count is a one-way door

The 4096 figure is not arbitrary, and it is not easily changed later. Your shard count caps how finely you can ever spread load: with 4096 shards you can never have more than 4096 storage nodes, and long before that, hot shards (a few very active row keys landing on the same shard) become your limit. Pick the number high enough that you will never want more, but not so high that the per-shard overhead (connections, metadata, the size of the shard map every worker caches) becomes a tax. Uber's choice of 4096 reflects a bet about a plausible ceiling for a single Schemaless *instance* — and they run many instances. The gotcha for anyone copying the pattern: this is the kind of constant you regret either way if you don't think hard about it on day one.

## 3. Secondary indexes and triggers: the parts MySQL won't do for you

**Senior rule of thumb: when you make the primary store dumb, you have to rebuild — explicitly and asynchronously — every smart feature you gave up, and you have to decide its consistency on purpose.**

A pure triplestore answers exactly one question quickly: "give me cells for this row key." But real services ask other questions — "give me all trips for this driver," "find the trip with this receipt number." Those are secondary-index queries, and the row-key-sharded entity table cannot answer them, because the data you want is scattered across shards by a different key.

Schemaless builds **secondary indexes as separate MySQL tables**. An index is defined over fields inside the cell bodies; when a cell is written, the index entries are materialized into their own tables, which are themselves sharded — but sharded by the *indexed* field, not by the original row key, so that an index lookup hits one shard. The crucial property: these indexes are **eventually consistent** with the base data by default. There is a window after a write during which the base cell exists but the index entry does not yet. Uber exposes index consistency as a deliberate choice — you can have indexes updated synchronously in the same transaction as the cell (consistent, slower) or asynchronously (faster, eventually consistent), and you pick per index based on what the query can tolerate.

The asynchronous path is driven by **triggers**, Schemaless's publish-subscribe mechanism. This is not a MySQL trigger; it is Schemaless's own change-notification framework. Because every cell write gets a monotonically increasing `added_id`, the set of all writes on a shard forms an ordered log. Trigger workers tail that log per shard, and for each new cell they invoke registered handlers — to populate secondary indexes, to publish change events to Kafka for downstream consumers, to denormalize data into other stores, and so on. The pattern is exactly [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern): the immutable, ordered write log *is* the change stream, and everything reactive hangs off it.

Here is a model of a trigger consumer that tails the per-shard log and maintains a secondary index. Note the checkpoint: this is what makes it resumable and at-least-once.

```python
class TriggerConsumer:
    """Tails the ordered write log of one shard (by added_id) and runs
    handlers. Mirrors Schemaless triggers: async, ordered, resumable.
    At-least-once delivery -> handlers MUST be idempotent."""

    def __init__(self, shard_log: list, checkpoint: int = 0):
        self.shard_log = shard_log        # ordered by added_id
        self.checkpoint = checkpoint      # last added_id we processed
        self.handlers = []

    def on_cell(self, handler):
        self.handlers.append(handler)

    def poll(self):
        # Only cells appended since our checkpoint, in order.
        new = [c for c in self.shard_log if c["added_id"] > self.checkpoint]
        for cell in sorted(new, key=lambda c: c["added_id"]):
            for handler in self.handlers:
                handler(cell)             # e.g. upsert into an index table
            self.checkpoint = cell["added_id"]   # advance only after success

# A secondary index: trips by driver_id, kept eventually consistent.
driver_index: dict[str, list[str]] = {}

def index_by_driver(cell):
    body = json.loads(cell["body"])
    driver = body.get("driver_id")
    if driver is None:
        return
    trips = driver_index.setdefault(driver, [])
    if cell["row_key"] not in trips:      # idempotent: re-running is safe
        trips.append(cell["row_key"])

log = [
    {"added_id": 1, "row_key": "trip:8f3a", "body": '{"driver_id": "d42"}'},
    {"added_id": 2, "row_key": "trip:1c0e", "body": '{"driver_id": "d42"}'},
    {"added_id": 3, "row_key": "trip:9b21", "body": '{"driver_id": "d77"}'},
]
consumer = TriggerConsumer(log)
consumer.on_cell(index_by_driver)
consumer.poll()
assert driver_index["d42"] == ["trip:8f3a", "trip:1c0e"]
assert consumer.checkpoint == 3          # resumable across restarts
```

The second-order consequence is one every team underestimates: **you are now running a stream-processing system, with all its operational concerns** — checkpointing, lag monitoring, idempotency of handlers, and "what happens when a handler is broken for an hour and then redeployed." The index being eventually consistent is not a bug; it is the price of decoupling. But you must *design for the staleness window*: a feature that reads its own write through a secondary index will see stale or missing results unless it reads the base cell by row key instead. More than one production incident, here and everywhere, has come from a developer assuming an index was as fresh as the primary write.

## 4. Buffered writes: staying up when a MySQL master dies

**Senior rule of thumb: at scale, "the leader is briefly unavailable" is not an exceptional event — it is a Tuesday. Your write path needs a graceful degradation for it, not just an exception.**

Writes in Schemaless must go to a shard's master. So what happens when that master is down — a failover in progress, a network partition, a host that just died? The naive answer is "the write fails," and at Uber's volume that would mean a steady drizzle of failed trip writes any time any master anywhere hiccups. Schemaless does something cleverer, enabled entirely by the append-only model: **buffered writes**.

![Before-and-after comparison of a Schemaless write: the healthy path commits on the master and replicates, while the failed-master path writes to a buffer table on a random cluster and replays on recovery](/imgs/blogs/uber-geo-distributed-data-4.webp)

When the master for a shard is unreachable, the worker does not give up. It picks a *different*, randomly chosen storage cluster and writes the cell into a **buffer table** there. The write is acknowledged to the client — the data is now durably persisted on at least one host (in practice two, via that cluster's own replication). Later, a background job notices the buffered cells, and once the real master for the shard has recovered, it replays them into their proper home and removes them from the buffer.

This works *only because writes are immutable cells*. A buffered cell has a fixed `(row_key, column_name, ref_key)`; replaying it onto the recovered master is an idempotent insert. There is no "merge conflict," because there is no in-place state to conflict with — you are appending an immutable fact, and appending it twice is the same as appending it once. Contrast this with a mutable system, where a buffered "set balance = X" replayed against a master that has since moved on is a correctness nightmare. The append-only design is what makes the failover *safe*, not just *available*.

```python
import random

class BufferedWriter:
    """Models Schemaless buffered writes. If a shard's master is down,
    persist the immutable cell to a buffer on another cluster and
    replay it idempotently once the master recovers."""

    def __init__(self, clusters: dict):
        self.clusters = clusters            # shard -> {"master_up": bool, "cells": [...]}
        self.buffers = {s: [] for s in clusters}   # per-cluster buffer table

    def write(self, shard: str, cell: dict) -> str:
        c = self.clusters[shard]
        if c["master_up"]:
            self._commit(shard, cell)
            return "committed"
        # Master down: buffer on a *different*, randomly chosen cluster.
        others = [s for s in self.clusters if s != shard]
        buf_shard = random.choice(others)
        cell = {**cell, "home_shard": shard}
        self.buffers[buf_shard].append(cell)   # durable on another node
        return "buffered"

    def _commit(self, shard: str, cell: dict):
        existing = self.clusters[shard]["cells"]
        key = (cell["row_key"], cell["column_name"], cell["ref_key"])
        if not any((x["row_key"], x["column_name"], x["ref_key"]) == key
                   for x in existing):       # idempotent replay
            existing.append(cell)

    def replay(self):
        """Background job: drain buffers back to recovered home shards."""
        for buf_shard, cells in list(self.buffers.items()):
            keep = []
            for cell in cells:
                home = cell["home_shard"]
                if self.clusters[home]["master_up"]:
                    self._commit(home, cell)   # safe: immutable cell
                else:
                    keep.append(cell)
            self.buffers[buf_shard] = keep

clusters = {
    "shardA": {"master_up": False, "cells": []},
    "shardB": {"master_up": True,  "cells": []},
}
w = BufferedWriter(clusters)
status = w.write("shardA", {"row_key": "trip:8f3a", "column_name": "BASE", "ref_key": 1})
assert status == "buffered"                 # master A was down
clusters["shardA"]["master_up"] = True      # A recovers
w.replay()
assert len(clusters["shardA"]["cells"]) == 1   # cell landed in its real home
```

The second-order consequence is a consistency caveat you must accept with eyes open: while a cell is buffered, a reader hitting shard A's storage node *will not see it* — it lives on a different cluster's buffer table, invisible to a normal read of shard A. So buffered writes trade a brief read-after-write inconsistency for write availability. For the trip store that is exactly the right trade: never lose a trip write, tolerate a sub-second window where a just-written cell is not yet visible from its home shard. For a system that needs read-your-writes on every path, you would make a different call.

## 5. Docstore: turning the pattern into a general-purpose database

**Senior rule of thumb: a bespoke datastore built for one workload becomes a liability when ten other teams want to use it; the second-generation move is to generalize the pattern into a platform with a real query layer — without abandoning the simple node at the bottom.**

Schemaless was purpose-built for the trip store and a handful of similar use cases. But the pattern was too good not to generalize, and across the company teams kept needing "a scalable place to put documents." Uber's answer was **Docstore**: a general-purpose, distributed document database that, by Uber's account, stores tens of petabytes and serves tens of millions of requests per second, used by microservices across every business vertical. It evolved directly out of Schemaless — Uber described the transition in "Evolving Schemaless into a Distributed SQL Database" — and it still has MySQL-family nodes at the bottom of the stack. The philosophy held; the layer on top got much more capable.

![Docstore's three-layer architecture: a stateless query engine on top, a control plane assigning shards, and a stateful storage engine of Raft-replicated MySQL or MyRocks partitions at the bottom](/imgs/blogs/uber-geo-distributed-data-5.webp)

Docstore is organized into three layers, and the separation is the design:

- **Stateless query engine (top).** This is the thin-routing-layer idea grown up. It handles query planning, request routing, sharding, schema management, authentication and authorization, request parsing and validation, and node health monitoring. Because it is stateless, you scale it horizontally and independently of the data — exactly like the Schemaless worker, but now with a real query surface (Docstore supports a richer query model, including materialized views and a SQL-like interface in its later evolution) instead of just triple lookups.
- **Stateful storage engine (bottom).** Organized as a set of **partitions**, where each partition is a small cluster of MySQL nodes (later also **MyRocks**, the RocksDB-backed MySQL storage engine, for write-heavy and space-sensitive workloads) on NVMe SSDs. A partition is **one leader and two followers**, replicated with **Raft consensus** rather than the older async-replication-plus-buffered-writes scheme. This layer owns consensus, replication, transactions, concurrency control, and load management. Data is sharded across partitions; a partition serves a set of shards.
- **Control plane.** Assigns shards to partitions and adaptively rebalances that placement in response to failure events and load. This is the piece that, in Schemaless, was the static shard map; in Docstore it is an active control loop.

The most important upgrade from Schemaless is the consistency model. By adopting **Raft** within each partition (the same mechanism we build from scratch in [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch)), Docstore provides **strong consistency within a partition** — Uber describes a strict-serializability guarantee at the partition level. A write is committed once a quorum of the partition's replicas has it in their Raft log, so a leader failover promotes a follower that is guaranteed to have the committed data. This is a genuinely stronger guarantee than original Schemaless's asynchronous replication, and it is what lets Docstore support transactional, read-your-writes workloads that the trip store never needed. The scope of the guarantee — *within a partition* — is the crucial qualifier and the same boundary distributed SQL systems like [CockroachDB](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive) and Spanner wrestle with; cross-partition transactions are a harder, more expensive problem, and most schemas are designed so that the entities that need to be transactionally consistent live in the same partition.

Two more pieces are worth naming because they show the pattern continuing to pay off:

- **MyRocks migration.** Uber moved many Docstore (and Schemaless) datastores from InnoDB to MyRocks. Because the storage engine sits *under* the same boring MySQL interface, this was a swap at the bottom of the stack — the LSM-tree-based engine ([LSM-trees: write-optimized storage](/blog/software-development/database/lsm-trees-write-optimized-storage-engines)) gives better compression and write amplification for their workload — without rewriting the distribution layer above. That is the dividend of keeping the node simple and the interface stable: you can change the engine without changing the system.
- **CacheFront.** Docstore added an integrated read-through cache (Redis-backed) at the query-engine layer, invalidated by Docstore's own change-data-capture stream (called **Flux**). Uber reported it serving on the order of **40M+ reads per second** across Docstore instances at very high hit rates, dramatically cutting the CPU footprint the storage engine has to provision for. It is, once again, a smart layer added *on top* of the simple store — and it rhymes with [Netflix's EVCache](/blog/software-development/database-scaling/netflix-evcache-multi-region-cache) and the broader [cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production).

## 6. Geo-distribution: keeping live trips alive across data centers

**Senior rule of thumb: "multi-region" is not one feature; it is a set of explicit decisions about where each piece of data is homed, how it replicates across regions, and what exactly happens to in-flight requests during a regional failover.**

Uber runs in many cities served from multiple data centers and regions, and a trip in progress is the canonical example of data that must survive a data-center loss. You cannot tell a driver mid-trip "sorry, our east-coast data center is down, your fare is gone." The geo story is therefore not a nice-to-have; it is the requirement that shaped everything above.

![Multi-region architecture where a trip is homed in one region's Raft leader, replicated cross-region to a follower, which is promoted to leader if the home region fails](/imgs/blogs/uber-geo-distributed-data-6.webp)

The model is **operate per-region, fail over across regions**:

- Each piece of data (each shard/partition) has a **home region** where its leader lives and which serves its reads and writes in the normal case. Routing the request to its home region keeps the common path fast — you are not paying a cross-region round trip on every write.
- Replicas exist **across regions**. In the Schemaless era this was async MySQL replication to minions in other data centers; in the Docstore era it is Raft followers, some of which are cross-region. Either way, a copy of the data is continuously kept in a second region.
- On a **region failure**, the cross-region replica is **promoted** to take over serving the affected shards. With Raft, the promoted follower is guaranteed to have all committed writes; with async replication, you accept a small replication-lag window of potential loss at the moment of failover. This is the same async-vs-sync replication trade-off covered in [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) and [sharding strategies compared](/blog/software-development/database-scaling/sharding-strategies-compared), now playing out across regions where the latency and the stakes are both higher.

The reason Uber could build geo-distribution incrementally is, again, that the storage node is simple. Cross-DC MySQL replication is a feature MySQL operators have understood for two decades; placing minions in other data centers was a configuration of a well-trodden mechanism, not a research project. The geo behavior is *composed* from boring, reliable per-node primitives plus a routing layer that knows each shard's home region — rather than delegated to a database that promises "global consistency" and hides the trade-offs you actually need to control. When the design relies on per-region operation with explicit failover, your on-call team can reason about a regional outage as "promote the replicas in region B," which is a comprehensible sentence, instead of "trust the planet-scale cluster."

#### Second-order optimization: pin related data to the same region

The subtle failure here is data whose pieces are homed in different regions but which a single request needs together. If a trip's cells live in US-WEST but the driver's profile is homed in US-EAST, every trip-with-driver read pays a cross-region hop, and a regional outage can leave you with half the data. The discipline — and it is a discipline you enforce in your schema and your sharding, not something the database does for you — is to **co-home data that is read together**: shard the driver's data and the trip's data by keys that map to the same region. This is the geo-scale version of the co-location reasoning behind [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key), and getting it wrong is one of the most expensive latency mistakes a multi-region system can make.

## 7. Ringpop: consistent-hash routing in the application tier

**Senior rule of thumb: storage is not the only thing that needs sharding; stateful application work — coordinating a trip, holding a session, owning a key — needs the same "which node owns this?" answer, and you want one consistent way to compute it.**

So far the routing we have discussed lives inside the storage layer. But Uber's dispatch and trip-coordination services also need application-layer sharding: a given live trip should be coordinated by exactly one service instance, and any other instance that gets a request for that trip should be able to find the owner. **Ringpop** is the library Uber built for this. It is not a datastore; it is a routing and membership library that you embed in your service, and it complements the storage layer by answering "which instance owns this key?" for stateful application logic.

![Ringpop consistent-hash routing: instances arranged on a hash ring kept in sync by SWIM gossip, where any instance hashes the key and forwards the request to the owning node](/imgs/blogs/uber-geo-distributed-data-7.webp)

Ringpop has three parts, and they map cleanly onto problems we have already met:

- **A SWIM gossip membership protocol.** Every few seconds, each node gossips its view of the member list to a few random peers. Joins and failures spread "by infection" — epidemically — across the cluster, so within seconds every node converges on the same membership without any central coordinator. This is the gossip-and-failure-detection machinery covered in [failure detection: gossip and phi-accrual](/blog/software-development/database/failure-detection-gossip-and-phi-accrual).
- **A consistent-hash ring.** Ringpop hashes node identities (with a uniform number of replica/virtual points each) onto a ring using FarmHash, stored in a red-black tree for fast lookups. A key is hashed onto the ring and owned by the next node clockwise. Adding or removing a node only reassigns the arcs adjacent to the change — no global reshuffle — which is the entire point of [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning).
- **Request forwarding ("handle or forward").** When a request arrives at any instance, that instance hashes the key. If it owns the arc, it handles the request locally. If not, it transparently **forwards** the request to the owning instance. Callers do not need to know the topology — they hit any instance and Ringpop routes it like a middleware layer, before the request ever reaches your business logic.

Here is a compact, runnable model of the ring and the handle-or-forward decision. The replica points are what keep the load balanced; the "next clockwise" rule is what makes ownership unambiguous.

```python
import bisect
import hashlib

def _h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)   # FarmHash in prod

class HashRing:
    """Consistent-hash ring with virtual nodes, like Ringpop's. Adding or
    removing a node only moves keys on the adjacent arcs."""

    def __init__(self, replicas: int = 100):
        self.replicas = replicas      # uniform replica points per node -> balance
        self._ring: dict[int, str] = {}
        self._sorted: list[int] = []

    def add(self, node: str):
        for i in range(self.replicas):
            point = _h(f"{node}#{i}")
            self._ring[point] = node
            bisect.insort(self._sorted, point)

    def remove(self, node: str):
        for i in range(self.replicas):
            point = _h(f"{node}#{i}")
            self._ring.pop(point, None)
            self._sorted.remove(point)

    def owner(self, key: str) -> str:
        if not self._sorted:
            raise RuntimeError("empty ring")
        h = _h(key)
        idx = bisect.bisect(self._sorted, h) % len(self._sorted)   # next clockwise
        return self._ring[self._sorted[idx]]

class RingpopNode:
    """A service instance. handle_or_forward is the whole routing model."""

    def __init__(self, name: str, ring: HashRing):
        self.name = name
        self.ring = ring

    def handle_or_forward(self, key: str, request: dict) -> str:
        owner = self.ring.owner(key)
        if owner == self.name:
            return f"{self.name} HANDLED {request['op']} for {key}"
        return f"{self.name} FORWARDED {key} -> {owner}"

ring = HashRing(replicas=100)
for n in ["A", "B", "C", "D"]:
    ring.add(n)

# Any node can receive the request; it routes to the deterministic owner.
got_it = RingpopNode("D", ring)
owner = ring.owner("trip:8f3a")
print(got_it.handle_or_forward("trip:8f3a", {"op": "update_location"}))
# -> "D HANDLED ..." if D owns the arc, else "D FORWARDED trip:8f3a -> C"

# Removing a node only disturbs its arcs; most keys keep their owner.
before = {k: ring.owner(k) for k in (f"trip:{i}" for i in range(1000))}
ring.remove("B")
after = {k: ring.owner(k) for k in (f"trip:{i}" for i in range(1000))}
moved = sum(1 for k in before if before[k] != after[k])
assert moved < 400   # ~1/4 of keys move when 1 of 4 nodes leaves, not all of them
```

Ringpop and the storage layer are two applications of the *same idea* at two layers of the stack: a consistent-hash ring decides which node owns a key, and gossip keeps everyone's view of the ring in sync. The storage layer uses it to decide which storage node holds a shard; Ringpop uses it to decide which application instance coordinates a trip. The second-order consequence to respect is that **application-layer sharding makes your service stateful**, with all that implies: a node leaving the ring moves ownership of its keys, so whatever in-memory state those keys had (a trip's coordination context) must be reconstructable from the durable store — which is exactly what Schemaless/Docstore is for. Ringpop routes; the datastore remembers. Mixing those responsibilities up is how you build a system that loses trips during a deploy.

## Case studies from production

The following vignettes are illustrative reconstructions in the spirit of the public design write-ups — composite scenarios that show how each mechanism behaves (and misbehaves) in practice. They are the situations these designs were built to handle.

### 1. The index that lied

A team builds a feature that, right after writing a new trip, queries a Schemaless secondary index to show the user "your recent trips." In testing it works; in production, intermittently, the just-completed trip is missing for a second or two. The wrong first hypothesis is a caching bug. The actual root cause: the secondary index is *eventually* consistent, populated asynchronously by a trigger consumer that is a beat behind the base write. The fix is not to make the index synchronous (that would slow every write); it is to read the just-written record **by row key from the base store**, which is always immediately consistent, and use the index only for queries that can tolerate staleness. The lesson: in a system where you rebuilt indexing yourself, the index's freshness is a property you chose, and read paths must be written to match the choice.

### 2. The master that died at dinner time

At the Friday evening peak, a host running a shard's MySQL master suffers a hardware fault. In a naive write-to-master-only system, every trip write hashing to that shard would fail until failover completed — minutes of lost writes at the worst possible time. Because Schemaless has buffered writes, the workers detect the unreachable master and divert those cells into buffer tables on other clusters, acknowledging the writes. No trip is lost. When the replacement master comes up, the background replay drains the buffers home. The only visible symptom is that, for the failover window, a freshly written cell briefly cannot be read from its home shard. The lesson: design the *degraded* path before you need it, and lean on immutability to make the degraded path safe.

### 3. The reshard that didn't reshuffle the world

Uber adds a batch of new storage nodes to a Schemaless instance to absorb growth. Because data is sharded into 4096 fixed logical shards mapped to nodes — not hashed directly to nodes — the operation is "move some shards to the new nodes and update the map," not "rehash the entire keyspace." Only the keys in the moved shards change homes; everyone else is undisturbed, and the move can be done shard-by-shard with the source still serving until the copy is verified. The contrast case is a team that hashed keys directly to N machines and discovered that going from 16 to 17 machines remapped almost every key. The lesson is the oldest one in sharding and the reason [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) is even possible: shard into a fixed large number of buckets, then map buckets to machines.

### 4. The cross-region read that doubled p99

A service's latency p99 quietly doubles after a seemingly unrelated change. Investigation shows a new code path reads a trip's cells (homed US-WEST) and, in the same request, the rider's preferences (homed US-EAST). Every such request now pays a cross-region round trip. The data was correct; the *placement* was wrong. The fix: shard the rider preferences by a key that co-homes them with the trip data in the same region, so the request stays local. The lesson: in multi-region systems, latency bugs are usually placement bugs, and the database will not fix your data homing for you — that is a schema-and-sharding decision you own.

### 5. The deploy that lost trip ownership

A dispatch service using Ringpop is deployed with a rolling restart. During the roll, instances leave and rejoin the ring, and ownership of in-flight trips reassigns to surviving nodes. A subtle bug: the trip's coordination state was held only in the departing node's memory, not durably persisted, so when ownership moved, the new owner had nothing to resume from. The symptom is dropped or stalled trips during deploys. The root cause is conflating *routing* (Ringpop's job) with *durability* (the datastore's job). The fix is to ensure every piece of routable state is reconstructable from Schemaless/Docstore, so a new owner rehydrates cleanly. The lesson: application-layer sharding makes you stateful, and stateful means "survive the node that holds the state vanishing."

### 6. The write amplification that crossed an ocean

This is the original Postgres pain, dramatized. A high-update table with several secondary indexes generates, per logical update, a cascade of physical writes; those writes are serialized into the replication stream and shipped to a replica in a distant data center over a link that is not free. As update volume climbs, the replica falls behind, and read-after-write expectations on the replica break. The wrong fix is "add bandwidth." The deeper fix Uber chose was to move to an engine whose secondary-index and replication model did not amplify the same way for their pattern, and ultimately to a store (Schemaless) whose write path was append-only and whose replication shipped compact, immutable cells. The lesson, stated carefully as **Uber's experience**: replication cost is downstream of your write model, so fix the write model.

### 7. The hot shard that 4096 couldn't save

A particular row key — say, a single very high-volume account or a synthetic "system" entity that everything writes to — concentrates a wildly disproportionate share of writes onto its one shard. Adding storage nodes does not help, because all those writes still hash to the same shard, which lives on one master. The shard count is irrelevant when the heat is on a single key. The fix is at the data-model layer: split the hot entity's writes across multiple row keys (a fan-out / sub-keying scheme) so the load spreads across shards. The lesson, and it is the limit of the whole approach: **sharding distributes keys, not the load within a key.** A single hot key is a data-modeling problem no shard count solves.

### 8. The engine swap nobody noticed

Uber migrates a set of datastores from InnoDB to MyRocks for better compression and write characteristics. Because the storage engine sits beneath the same MySQL interface that the Schemaless/Docstore layers speak, the migration is done node-by-node underneath a stable API — the distribution layer above does not know or care which engine is storing the bytes. Application teams see lower storage cost and unchanged semantics. The lesson is the quiet payoff of the entire philosophy: **when you keep the node behind a boring, stable interface, you can replace the node's internals without touching the system built on top of it.** That optionality is worth more than any single engine's peak benchmark.

## When to build distribution on top of a simple node — and when not to

Reach for the Uber pattern (own the sharding and routing; keep boring single-node engines underneath) when:

- Your workload is dominated by **key-based access** — get/put by an entity ID — rather than complex ad-hoc joins across the whole dataset. Schemaless thrives because the trip store asks "give me this trip," not "join trips to drivers to payments with three predicates."
- You can make, or already have, an **append-only or version-keyed write model**, which is what makes idempotent retries, buffered failover, and free audit history fall out for nothing.
- You have, or are willing to build, a team with **deep operational competence in the single-node engine** you choose. The pattern trades external complexity for internal ownership; that is only a win if you can staff the ownership.
- You need **geo-distribution with explicit control** over data homing and failover, and you want your on-call to reason about it as "promote the replicas in region B" rather than trusting an opaque global cluster.
- You are large enough that the **legibility of the system at 3 a.m.** is worth the cost of building and running the distribution layer yourself.

Skip it — and reach for an off-the-shelf distributed database (a managed [Cassandra/DynamoDB-style store](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive), a distributed SQL engine like [CockroachDB](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive), or [Spanner](/blog/software-development/database/spanner-truetime-and-external-consistency)) — when:

- You are **not yet at the scale** where a single well-tuned node (plus read replicas and [vertical scaling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling)) is the bottleneck. Building a sharding layer before you need it is a classic premature-distribution mistake; the [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) exists to talk you out of it.
- Your workload genuinely needs **rich cross-shard transactions and ad-hoc relational queries** as a first-class, everyday thing. Rebuilding a distributed query planner and cross-partition transactions on top of dumb nodes is a multi-year project; if that is your core need, buy the database that already did it.
- You **lack the engineering capacity** to own a storage platform indefinitely. This is infrastructure with a permanent staffing cost, not a one-time build. A two-person team running a managed distributed DB will out-deliver the same team trying to operate a home-grown sharding layer.
- You want **strong cross-entity consistency by default** without designing your schema around partition boundaries. The "strong within a partition, design for it" model is powerful but it is a model you have to internalize across every team.

The throughline of Uber's storage history is not "MySQL good, Postgres bad" and it is not "build everything yourself." It is a stance about *where complexity should live*. Uber pushed distribution **up** into layers they wrote, tested, and could explain — the Schemaless worker, the Docstore query engine, the Ringpop hash ring — and kept the bottom of the stack as the most boring, most reliable, most replaceable thing they could: a single MySQL node doing inserts and indexed selects. Distribution on top, simplicity underneath, ownership of the seam between them. When you next hit a scaling wall, the question worth asking before you go shopping is the one Uber answered: *do I need a smarter database, or do I need to keep my database dumb and get smarter about the layer on top of it?*

## Further reading

- Uber Engineering — "Why Uber Engineering Switched from Postgres to MySQL" (the contested, internals-level migration write-up).
- Uber Engineering — the Schemaless trilogy: "Designing Schemaless," "The Architecture of Schemaless," and "Using Triggers on Schemaless."
- Uber Engineering — "Evolving Schemaless into a Distributed SQL Database" and the Docstore / CacheFront posts.
- `uber/ringpop-go` and the Ringpop architecture docs (SWIM gossip, consistent hashing, handle-or-forward).
- On this blog: [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning), [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key), [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch), and [Vitess: sharding MySQL at scale](/blog/software-development/database/vitess-sharding-mysql-at-scale) for a different take on distributing MySQL.

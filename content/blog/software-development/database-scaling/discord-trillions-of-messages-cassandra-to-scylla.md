---
title: "How Discord Stores Trillions of Messages: Cassandra to ScyllaDB"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A case study of Discord's messages store — the (channel_id, bucket) data model, the supernode and tombstone traps, and why JVM garbage-collection pauses drove a rewrite onto ScyllaDB's shard-per-core engine."
tags: ["database-scaling", "cassandra", "scylladb", "wide-column", "data-modeling", "hot-partition", "tombstones", "garbage-collection", "request-coalescing", "case-study", "distributed-systems", "system-design"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 30
---

There is a specific kind of dread that comes from watching a query you have run ten thousand times suddenly take ten seconds. Not ten milliseconds, not even ten times slower — ten full seconds, the kind of pause where a user has already given up, refreshed, and filed a bug. Discord lived that exact failure in 2017: a single channel, holding exactly one live message, took a Cassandra node ten seconds and a full stop-the-world garbage-collection pause to load. The channel had once held millions of messages. Almost all of them had been deleted. And every one of those deletions had left behind a marker that the database still had to walk past, one at a time, before it could answer "give me the last fifty messages."

That single incident contains the whole story of how Discord stores messages, why the obvious design works for years and then quietly stops working, and why the company eventually rewrote its largest data store from Apache Cassandra to ScyllaDB without changing a single line of its data model. This post is a case study of that system — the messages store that grew from billions of messages in 2017 to **trillions** by 2022, as documented in Discord's two now-famous engineering posts. We will reconstruct the data model from first principles, walk through every trap it hit at scale, and end on the migration engineering that moved trillions of rows in nine days.

![Discord's message data model: partition by (channel_id, bucket), cluster by message_id newest-first](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-1.webp)

The diagram above is the mental model for everything that follows. A message belongs to a channel. Its partition is keyed not by `channel_id` alone but by the pair `(channel_id, bucket)`, where the bucket is a fixed time window. Inside that partition, messages are sorted by `message_id` in descending order, so the newest message is physically first. Hold that picture; almost every problem and every fix in this post is a direct consequence of those two decisions — what you partition by, and how you sort inside the partition.

## Why a chat store is a different problem

> The first job is not to pick a database. It is to write down, in one sentence, the read you must serve in single-digit milliseconds. Everything else is downstream of that sentence.

Before any schema, you have to be honest about the workload, because a chat application has a very particular shape that rules out most of the "obvious" choices. Here is the mismatch between how people assume a message store behaves and how it actually behaves:

| Assumption | Naive view | Reality at Discord scale |
| --- | --- | --- |
| Reads dominate | "It's a feed, mostly reads" | Writes are relentless and unbounded; every keystroke-batch, every reaction, every edit is a write |
| Recent data matters | "Cache the last page" | The read is almost always "the most recent N messages in *this* channel," time-ordered, newest-first |
| Data is roughly uniform | "Channels are similar" | Load is wildly skewed — one channel can out-message a hundred thousand others combined |
| Deletes are rare | "People keep their chats" | Bulk deletes happen, and in an LSM store a delete is a *write*, not a removal |
| The set fits in RAM | "Just add memory" | The corpus is measured in petabytes; the working set spills off any single machine immediately |

Discord's first messages store was a single MongoDB replica set. It worked until it didn't: the working set stopped fitting in memory, latency became unpredictable, and the team faced the classic inflection point covered in [when one database is not enough](/blog/software-development/database-scaling/when-one-database-is-not-enough). They needed a store that was write-optimized, horizontally scalable without manual resharding, and tolerant of node loss. That set of requirements points almost directly at a leaderless, wide-column store built on a log-structured merge tree — the family of systems dissected in [Cassandra and DynamoDB: a deep dive into leaderless wide-column stores](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive).

The reason write-optimization matters so much is the storage engine underneath. Cassandra and ScyllaDB both use [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines): writes are appended to an in-memory `memtable` and a commit log, then flushed to immutable on-disk files called SSTables, and merged later by background compaction. A write never seeks; it appends. That is exactly the property an append-heavy chat workload wants. The cost — and this becomes the villain of our story — is that *reads* may have to consult the memtable plus several SSTables and merge the results, and that deletes are recorded as markers rather than executed in place.

## 1. The data model: partition key and clustering order

**The senior rule of thumb here is blunt: in a wide-column store, you do not model your entities, you model your queries.** There are no joins to bail you out later. You pick a partition key so that the rows a single query needs live together on one node, and you pick a clustering order so that those rows are already sorted the way you will read them. Get those two right and reads are a single sequential scan; get them wrong and no amount of hardware saves you.

Discord's dominant query is "load the most recent messages in this channel." The intuitive first schema is to partition by `channel_id` and sort by `message_id`:

```sql
-- The tempting, wrong-at-scale first cut.
CREATE TABLE messages (
    channel_id  bigint,
    message_id  bigint,
    author_id   bigint,
    content     text,
    PRIMARY KEY (channel_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```

This is correct in its query shape — all of a channel's messages live in one partition, sorted newest-first, so "last 50" is a clean `LIMIT 50` scan from the top. The problem is the word *all*. A busy channel accumulates messages forever, and a Cassandra partition that grows without bound is a time bomb: large partitions slow down reads, balloon memory during compaction, and eventually trip the "partition over 100 MB" warnings that every Cassandra operator learns to fear.

The fix is **bucketing**: fold a coarse time window into the partition key so a partition can only ever hold one window's worth of messages. Discord chose a static ten-day window, sized empirically so that even a busy channel's ten-day partition stays comfortably under 100 MB. The real schema, lightly simplified, is:

```sql
CREATE TABLE messages (
    channel_id  bigint,
    bucket      int,        -- which 10-day window
    message_id  bigint,     -- Snowflake: time-ordered, globally unique
    author_id   bigint,
    content     text,
    PRIMARY KEY ((channel_id, bucket), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```

Read the primary key carefully, because the double parentheses are load-bearing. `((channel_id, bucket), message_id)` declares a **composite partition key** of `(channel_id, bucket)` and a **clustering key** of `message_id`. Two messages in the same channel but different buckets live in *different partitions*, potentially on different nodes. Two messages in the same channel and bucket live in the same partition, sorted by `message_id` descending. That is precisely the structure drawn in the mental-model figure above.

### How the bucket is computed

The bucket is not stored as wall-clock time; it is derived deterministically from a timestamp so the application can always recompute it. Discord's IDs are **Snowflakes** — 64-bit integers whose high bits are the milliseconds elapsed since a fixed "Discord epoch." Because the timestamp is embedded in the ID, a `message_id` is simultaneously a unique key, a creation timestamp, and a sort key that orders messages chronologically for free. The bucket function falls straight out of that:

```python
import time

DISCORD_EPOCH = 1420070400000           # 2015-01-01T00:00:00Z, in ms
BUCKET_SIZE   = 1000 * 60 * 60 * 24 * 10  # 10 days, in ms

def snowflake_timestamp(snowflake: int) -> int:
    """Milliseconds since the Discord epoch, recovered from a Snowflake."""
    return snowflake >> 22                # top 42 bits are the ms timestamp

def make_bucket(snowflake: int | None) -> int:
    if snowflake is None:                 # "now" — used when writing a fresh message
        ms_since_epoch = int(time.time() * 1000) - DISCORD_EPOCH
    else:
        ms_since_epoch = snowflake_timestamp(snowflake)
    return ms_since_epoch // BUCKET_SIZE

def buckets_for_range(start_id: int, end_id: int | None = None) -> range:
    """Every bucket a [start_id, end_id] query must touch, newest-first."""
    end = make_bucket(end_id)
    start = make_bucket(start_id)
    return range(end, start - 1, -1)       # descending: probe recent buckets first
```

Now the read becomes a small, bounded loop. To fetch the last fifty messages in a channel, the application starts at the current bucket and walks backward, partition by partition, until it has collected fifty rows:

```python
def recent_messages(session, channel_id: int, limit: int = 50):
    collected = []
    bucket = make_bucket(None)            # start at the current 10-day window
    # Walk backward across buckets until we have enough or run out of history.
    while len(collected) < limit and bucket >= 0:
        rows = session.execute(
            """SELECT message_id, author_id, content
                 FROM messages
                WHERE channel_id = %s AND bucket = %s
             ORDER BY message_id DESC
                LIMIT %s""",
            (channel_id, bucket, limit - len(collected)),
        )
        collected.extend(rows)
        bucket -= 1                        # older window
    return collected[:limit]
```

For an active channel, almost every read is satisfied by the first one or two buckets — exactly the case the design optimizes for. The walk backward only deepens for nearly-dead channels, where you would rather pay a few extra round trips than carry an unbounded partition for the popular ones. This is the entire genius of bucketing: it converts an unbounded data-structure problem into a bounded one, and it does so without giving up the newest-first read.

### Second-order optimization: avoid writing nulls

A non-obvious gotcha lurks in this schema. In Cassandra, writing a `NULL` into a column is not a no-op — it writes a **tombstone**, a marker that says "this cell is deleted." A naive ORM that always writes every column, filling absent fields with `NULL`, will silently scatter tombstones across your data. Discord found their early writes were creating roughly a dozen unnecessary tombstones per message simply because the writer always set every column. The fix is to write only the columns you actually have, and to treat the column list as part of the query design, not an afterthought. Tombstones, as we are about to see, are not free.

## 2. Why Cassandra was the right call in 2017

**The rule that earns its keep: pick the storage engine whose failure mode you can live with, not the one with the best happy-path benchmark.** In 2017, Cassandra's failure modes — eventual consistency, repair overhead, compaction tuning — were ones Discord could engineer around. Its strengths were exactly the ones a message workload needs.

![Cassandra is leaderless: any node coordinates a write and acks at quorum, so throughput scales with node count](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-2.webp)

The figure traces a single write. A client sends a message to *any* node in the cluster. There is no master, no primary, no special node that all writes must funnel through. The node that receives the request becomes the **coordinator** for that write: it hashes the partition key to find which nodes own the data, forwards the write to the replicas (Discord ran a replication factor of three), and acknowledges the client as soon as a configurable quorum of replicas confirm. With a write quorum of two out of three, a single slow or dead replica never blocks the write.

This leaderless design is why the system scales the way it does. Four properties made it the obvious 2017 choice:

- **No single master.** Every node accepts reads and writes, so there is no write bottleneck to outgrow and no failover dance when the primary dies. The architecture and its tradeoffs are unpacked in depth in the [leaderless wide-column deep dive](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive).
- **Linear horizontal scale.** Adding capacity means adding nodes; the consistent-hash ring redistributes ownership automatically. Discord explicitly did not want to ever hand-reshard, the operational nightmare described in [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime).
- **Tunable consistency.** Consistency is a per-query knob (`ONE`, `QUORUM`, `ALL`), so the team could trade latency for safety read-by-read rather than committing the whole system to one point on the spectrum — the dial explored in [tunable consistency at scale](/blog/software-development/database-scaling/tunable-consistency-at-scale).
- **Write-optimized storage.** The LSM engine turns every write into an append, which is exactly right for an append-heavy chat log.

Discord's first cluster was twelve nodes with a replication factor of three, holding billions of messages. It worked, and for several years it worked well. The lesson is not that Cassandra was a mistake — it was the correct decision for the scale and the team of the time. The lesson is that **every architectural choice has a scale at which its weakest property becomes the binding constraint**, and the rest of this post is the story of Discord hitting three of those constraints in succession.

## 3. The first crack: hot partitions and the supernode

**Sharding solves total load. It does nothing, by itself, for concentrated load.** You can have a perfectly even partition map and still melt one node, because the partition map distributes *keys* evenly, not *traffic*. When traffic piles onto one key, that key's partition — and the node hosting it — takes the hit alone. This is the hot-partition problem, and it is its own discipline, covered end to end in [hot partitions and hot rows](/blog/software-development/database-scaling/hot-partitions-and-hot-rows).

In a chat product, the skew is extreme and unavoidable. A server of three friends sends a few messages a day. A server of several hundred thousand people, during a live event, sends thousands of messages a second into a single channel. That channel is a **supernode**: a partition so much hotter than its peers that it defines the capacity of the node hosting it. Bucketing helps here in one specific way and not at all in another, and it is critical to be precise about which.

![The supernode problem: without buckets a hot channel's partition grows unbounded; bucketing caps each partition's size](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-3.webp)

What bucketing fixes is partition **size**. Without it, a hot channel is one partition that grows forever — tens of millions of rows, well past the 100 MB threshold where reads slow and compaction thrashes. With a ten-day bucket, that same channel becomes many bounded partitions, each one a manageable window, and recent reads touch only the latest one or two. Size is bounded; the time bomb is defused.

What bucketing does *not* fix is partition **concurrency**. During a live event, the hot channel's current bucket is still a single partition on a single set of replica nodes, and thousands of concurrent readers and writers all hammer that one partition at once. Bucketing spread the channel across *time*; it did nothing to spread the *current* window across *space*. That residual hot-partition concurrency is the problem Discord eventually solved not in the schema at all, but in an application-tier service that we will reach in section 7. The takeaway is one every senior engineer internalizes the hard way: **a data-model trick that bounds size is not the same as one that bounds load, and confusing the two leaves you exposed exactly when traffic peaks.** If you are choosing a partition key today, the failure modes to reason about up front are catalogued in [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key).

## 4. The second crack: tombstones and the Puzzles & Dragons stall

**In a log-structured store, a delete is the most expensive write you can issue.** It does not remove data; it writes a tombstone — a marker that shadows the deleted row until a future compaction, no sooner than `gc_grace_seconds` later, actually reclaims the space. Until then, every read that crosses the deleted range must read the tombstones, hold them in memory, and merge them away to prove the rows are gone. Deletes, counterintuitively, make reads slower and heavier, not lighter.

![A channel emptied of millions of messages left millions of tombstones that every read had to scan, triggering a 10-second GC pause](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-4.webp)

The timeline above is the incident that opened this post, the one Discord named after the community it happened in. A large server — a Puzzles & Dragons fan community — had a channel where users had posted millions of messages and then, over time, deleted almost all of them. What remained was a single live message and millions of tombstones, all clustered in the same partition. When someone opened that channel, the read for "the last fifty messages" did the only thing it could: it started scanning from the newest `message_id` and walked down through millions of tombstones looking for fifty live rows it would never find in the recent range. Each tombstone is an object the JVM must allocate, hold, and eventually collect. Materializing millions of them at once exhausted the heap and triggered a **ten-second stop-the-world garbage-collection pause**. For ten seconds, that node answered nothing.

The fixes are a checklist that every Cassandra operator eventually memorizes:

- **Stop writing nulls.** Each accidental null is a tombstone; eliminating them at the writer removed the steady drip Discord had been creating per message.
- **Avoid range deletes where a partition rotation will do.** Bucketing already gives you a natural way to retire old data by simply ceasing to read old buckets, rather than issuing deletes that scatter tombstones.
- **Tune `gc_grace_seconds` deliberately.** It defaults to ten days for a reason — it must exceed your repair interval, or you risk resurrecting deleted data — but it is a real knob with real consequences for how long tombstones linger.
- **Read defensively.** Cap how many tombstones a single query is allowed to scan before it fails fast, so one pathological partition degrades one query instead of stalling the whole node.

The deeper lesson is about *coupling*. The tombstone behavior, the LSM read path, and the JVM's garbage collector were three separate systems, each individually reasonable. The incident happened where they met: a data-model decision (allow bulk deletes in a clustered partition) flowed into a storage-engine behavior (tombstones persist until compaction) which flowed into a runtime behavior (allocating millions of objects triggers a stop-the-world GC). No single layer was wrong. The interaction was catastrophic. Which brings us to the layer that turned out to be the real, structural problem.

## 5. The real killer: JVM garbage collection at scale

**A garbage-collected runtime is a wonderful thing right up until the moment the collector's pauses become your tail latency, and at that point no amount of tuning is a cure — it is a delay.** This is the constraint that ultimately drove the rewrite, and it is worth being precise about why.

![Cassandra's shared JVM heap pauses every core to collect garbage; ScyllaDB pins one shard per core with no GC](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-5.webp)

Cassandra runs on the JVM, which means every Cassandra node has one large shared heap, and that heap is managed by a garbage collector. When the collector runs a major collection, it can pause the entire process — every thread on every core stops — while it walks and compacts the heap. For a node serving thousands of reads per second, a multi-hundred-millisecond pause is a cliff in your latency graph, and a multi-second pause (as in the Puzzles & Dragons incident) is an outage. The pauses are not constant, which is worse: they are *unpredictable* spikes that show up as a fat, jittery p99 and p999, the part of the latency distribution that users actually feel.

Discord spent enormous engineering effort fighting this. They tuned the garbage collector and heap settings exhaustively. They ran a maintenance ritual operators came to call the **gossip dance**: to let a node catch up on compaction without serving slow reads, they would cycle it out of rotation, let it compact, and cycle it back. They scaled the cluster up — and here is the number that tells the whole story — to **roughly 177 nodes** by 2022, holding trillions of messages. But scaling out a JVM-bound system does not remove GC pauses; it just gives you 177 independent sources of them. The cluster was bigger, the operational burden was heavier, and the tail latency was still unpredictable. They had hit the wall described in [vertical scaling and its ceiling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling), except the ceiling was not CPU or RAM — it was the runtime itself.

This is the moment a team has to make a genuinely hard call. The data model was *fine* — it had scaled from billions to trillions of messages without a fundamental redesign. The query API was fine. The thing that was wrong was one layer below the database's logic: the execution substrate. Tuning had reached the point of diminishing returns, where each additional month of GC work bought a smaller improvement. The honest conclusion was that the problem was structural and the cure was a different storage engine — one whose performance model did not include a garbage collector at all.

## 6. ScyllaDB: same model, no JVM, shard-per-core

ScyllaDB is, by design, a drop-in-compatible reimplementation of Cassandra: same CQL query language, same data model, same SSTable-based storage, same wire protocol. What is completely different is underneath. ScyllaDB is written in **C++** with manual memory management — no JVM, no garbage collector, no stop-the-world pauses. And it is built on the **Seastar** framework, which uses a **shard-per-core** architecture.

Shard-per-core is the architectural inverse of Cassandra's shared heap. Instead of one big heap and a thread pool that any core can pick work from — which requires locks, cache-line contention, and a shared collector — Seastar pins one shard to each physical core, and that shard owns its own slice of memory and data with no sharing. Cores communicate by explicit message passing, not shared state. A unit of work runs to completion on one core, touching only that core's memory. There is no lock contention because there is nothing shared to contend over, and there is no garbage collector because memory is managed explicitly per shard. The contrast in the figure above is the whole pitch: where Cassandra stalls every core at once to collect a shared heap, ScyllaDB never stalls at all.

The payoff, in Discord's own reported numbers, was dramatic:

| Dimension | Cassandra | ScyllaDB |
| --- | --- | --- |
| Node count (same data) | ~177 | ~72 |
| Runtime | Java / JVM | C++ / Seastar |
| GC pauses | stop-the-world, unpredictable | none |
| p99 read latency | 40–125 ms | ~15 ms |
| p99 write latency | 5–70 ms | ~5 ms |
| Per-node efficiency | low (GC-bound) | high (core-bound) |

The node count is the headline most people remember — the cluster shrank from roughly 177 nodes to roughly 72 for the *same* data — but the latency rows are the ones that mattered operationally. The wild, GC-driven tail collapsed into a tight, predictable band. That is the difference between a system you constantly babysit and one you can mostly leave alone.

### Second-order optimization: the super-disk

A non-obvious piece of the ScyllaDB deployment is how Discord got fast *and* durable storage. Local NVMe SSDs are blazing fast but ephemeral — lose the machine, lose the disk. Network-attached persistent disks are durable but slower and jitter under load. Discord built a **super-disk**: a RAID array of local NVMe SSDs acting as a fast cache layer in front of a durable, network-attached persistent disk, so that reads and writes hit local flash speed while durability lives on the network volume. Each of the 72 nodes carried on the order of 9 TB. The lesson generalizes: the storage engine's performance model and the underlying disk's performance model have to match, or the engine's advantages evaporate against I/O stalls.

## 7. The data-services layer and request coalescing

Switching engines fixed the GC problem, but it did *not* fix the hot-partition concurrency problem from section 3 — a supernode channel during a live event still concentrates thousands of concurrent reads onto one partition. Discord solved that one architecturally, in a layer that sits between the API and the database. They route all database access through a **data-services layer** — originally written in Go, later rewritten in Rust — and its most important trick is **request coalescing**.

![Concurrent reads of the same key collapse into one database query whose single result is shared to every waiter](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-6.webp)

The idea is simple and the figure makes it concrete. When a popular channel is being viewed by many people at once, thousands of identical "read channel C" requests arrive in the same instant. Without coalescing, that is thousands of identical queries hammering one hot partition. With coalescing, the data service notices that a query for that exact key is already in flight, and instead of issuing a second query, the new request simply *waits on the result of the first one*. One query goes to ScyllaDB; the single result is fanned back out to every waiter. A read storm of thousands collapses into one database hit.

This is doubly effective because the data services route by channel ID using consistent hashing, so every request for a given channel lands on the same data-service instance — which means the coalescing actually catches the duplicates instead of scattering them across a fleet. Here is the core of the mechanism, sketched in Rust the way Discord's service is built:

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

#[derive(Clone)]
struct Messages(Arc<Vec<Message>>);   // cheap to clone: Arc bumps a refcount

/// One in-flight read per key. New callers for a key that is already being
/// fetched subscribe to the existing result instead of issuing a second query.
struct Coalescer {
    inflight: Mutex<HashMap<ChannelId, broadcast::Sender<Messages>>>,
    db: ScyllaClient,
}

impl Coalescer {
    async fn read_channel(self: &Arc<Self>, channel: ChannelId) -> Messages {
        // Fast path: is a fetch for this key already running? If so, wait on it.
        let mut rx = {
            let mut map = self.inflight.lock().await;
            match map.get(&channel) {
                Some(tx) => tx.subscribe(),          // join the existing fetch
                None => {
                    let (tx, rx) = broadcast::channel(1);
                    map.insert(channel, tx);         // we own this fetch
                    // Drop the lock before touching the DB — never hold a lock
                    // across an await that does network I/O.
                    drop(map);
                    return self.fetch_and_publish(channel, rx).await;
                }
            }
        };
        // Someone else is fetching; block until they publish the shared result.
        rx.recv().await.expect("publisher dropped without sending")
    }

    async fn fetch_and_publish(
        self: &Arc<Self>,
        channel: ChannelId,
        mut rx: broadcast::Receiver<Messages>,
    ) -> Messages {
        let result = self.db.query_recent(channel).await;   // the ONE real query
        // Publish to every waiter, then clear the slot so the next storm coalesces too.
        if let Some(tx) = self.inflight.lock().await.remove(&channel) {
            let _ = tx.send(result.clone());
        }
        let _ = rx.try_recv();           // we already have `result`
        result
    }
}
```

The shape is what matters more than the exact API: a map from key to an in-flight notification channel, a fast path that *joins* an existing fetch rather than starting a new one, and careful discipline about never holding the lock across the network call. This single pattern is what let ScyllaDB and the data services together absorb traffic spikes that would have flattened a raw Cassandra cluster — because the hottest partition in the system only ever sees one query at a time per data-service instance, no matter how many users are watching. It is the same defensive instinct as the read-side protections in [cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd), applied at the database tier.

## 8. The migration: trillions of messages in nine days

A drop-in-compatible engine sounds like it should make migration easy, and the data-model compatibility genuinely helped — Discord did not have to rewrite a single query. But moving *trillions* of live messages from a 177-node cluster to a 72-node cluster, with zero downtime and zero data loss, is its own hard engineering problem.

![Discord dual-wrote new data, then a Rust migrator backfilled history at up to 3.2M messages/sec, finishing in nine days](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-7.webp)

The strategy in the figure is the canonical zero-downtime migration shape, the one detailed in [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime), specialized for this scale. First, **dual-write**: new messages are written to both Cassandra and ScyllaDB simultaneously, so from the cutover moment forward the two clusters stay in sync at the head. Second, **backfill**: a separate process copies the entire history from old to new. Third, **verify**: compare token ranges between the clusters to prove the copy is complete and correct. Fourth, **cut over**: flip reads to ScyllaDB and decommission Cassandra.

The backfill is where the story gets good. Discord's first plan used ScyllaDB's off-the-shelf Spark-based migrator, and the estimate came back at roughly **three months**. For a team that wanted this cluster gone, three months of running two copies of the largest data store in the company was unacceptable. So they wrote their own migrator in Rust, tuned specifically for this job — driving large token-range scans in parallel and writing into ScyllaDB at rates the generic tool never approached. It hit speeds of up to **3.2 million messages per second**, and it finished the entire migration in **nine days**.

There are two lessons stacked here. The first is that **a purpose-built tool can beat a generic one by an order of magnitude when the workload is well understood** — Discord knew exactly the shape of their data and could exploit it in ways a general framework could not. The second is subtler: they did this migration *before* the 2022 World Cup, deliberately. When the World Cup final produced enormous message-send spikes — the kind of synchronized global event that creates the worst possible supernode load — the new ScyllaDB cluster with its request-coalescing data services handled it smoothly, where the old Cassandra cluster would have been a frantic on-call night. Migrations are not just about the new system being better; they are about being on the better system *before* the moment you need it.

## 9. Cassandra versus ScyllaDB: the decision in one view

It is worth stepping back to see the whole comparison at once, because the most important thing about this migration is what *didn't* change.

![Same data model and query API, but the C++ engine cut node count and tail latency sharply](/imgs/blogs/discord-trillions-of-messages-cassandra-to-scylla-8.webp)

The data model — `(channel_id, bucket)` partitioning, `message_id DESC` clustering, Snowflake IDs, ten-day buckets — survived the migration completely intact. The query language survived. The bucketing strategy survived. What changed was strictly the execution substrate: Java for C++, a shared GC heap for shard-per-core, 177 nodes for 72, an unpredictable tail for a tight one. That separation is the cleanest possible illustration of a principle worth tattooing on the wall: **your data model and your storage engine are different decisions, and they fail for different reasons.** Discord's data model never failed. Its execution engine did. Recognizing which layer is actually the bottleneck — rather than assuming the model must be wrong because the system is slow — is what made a clean engine swap possible instead of a painful redesign.

## Lessons from production

The point of a case study is the transferable lessons, so here are the ones that generalize beyond Discord, each grounded in a concrete moment from this story.

### 1. The 100 MB partition that bounded the design

The symptom was Cassandra logging warnings about partitions exceeding 100 MB. The wrong first hypothesis is "we need bigger nodes." The actual root cause is that an unbounded clustering key — every message in a channel forever — produces an unbounded partition, and large partitions punish reads and compaction alike. The fix was bucketing: fold a coarse time window into the partition key so each partition is bounded by construction. The lesson is that **in a wide-column store you must design a ceiling for every partition before you ship**, because there is no automatic mechanism to split a partition that grew too large. If you cannot answer "what is the maximum size this partition can reach," you have not finished the schema.

### 2. The single message that took ten seconds

The symptom was a channel with one live message taking ten seconds to load. The wrong hypothesis is "the read is inefficient" — it was a perfectly ordinary `LIMIT 50`. The root cause was millions of tombstones from bulk deletes, all of which the read had to scan and the JVM had to allocate, triggering a stop-the-world GC pause. The fix was upstream: stop generating tombstones (no null writes, no gratuitous range deletes) and read defensively. The lesson is that **a delete is a write, and a bulk delete is a latent denial-of-service against your own reads**. The cost of a delete in an LSM store is paid later, by an unrelated reader, which is exactly why it is so easy to miss in testing.

### 3. The supernode that bucketing didn't save

The symptom was one channel during a live event saturating its replica nodes even though partitions were nicely bounded by buckets. The wrong hypothesis is "bucketing should have spread the load." The root cause is that bucketing spreads a channel across *time*, not the current window across *space* — the live bucket is still one partition on one set of nodes. The fix lived in the application tier, not the schema: request coalescing in the data services. The lesson is that **bounding partition size and bounding partition concurrency are two different problems with two different solutions**, and a size fix gives you a false sense of safety against a load spike.

### 4. The gossip dance that signaled the real problem

The symptom was an elaborate operational ritual — cycling nodes out of rotation to let them compact — that the team performed routinely. The wrong hypothesis is "this is just normal Cassandra operations." The root cause is that compaction and GC pressure on a JVM-bound node made it unable to both serve traffic and keep up with background work, so operators had to manually choose one or the other. The fix, ultimately, was a different engine. The lesson is that **a maintenance ritual that becomes routine is a design smell pointing at a structural problem** — when you find yourself building tooling and muscle memory around a recurring pain, ask whether the pain is fundamental rather than incidental.

### 5. The tuning that stopped paying off

The symptom was diminishing returns on JVM and GC tuning: each round of effort bought a smaller latency improvement than the last. The wrong hypothesis is "we just need to tune harder." The root cause is that the garbage collector is intrinsic to the runtime; you can reduce the frequency and duration of pauses, but you cannot eliminate them while staying on the JVM. The fix was to leave the runtime entirely. The lesson is to **know the difference between a problem you can tune and a problem you can only escape** — tuning has an asymptote, and recognizing when you have hit it saves you from spending another year polishing a constraint you should be replacing.

### 6. The three-month estimate that became nine days

The symptom was a generic migration tool projecting a three-month backfill for trillions of rows. The wrong hypothesis is "that's just how long it takes to move this much data." The root cause is that a general-purpose tool cannot exploit the specific structure of your data the way you can. The fix was a purpose-built Rust migrator that drove 3.2 million messages per second and finished in nine days. The lesson is that **for a genuinely large, well-understood, one-time job, building the tool is often cheaper than enduring the generic one** — the engineering cost of the custom migrator was trivial against three months of operating two copies of your largest cluster.

### 7. The migration timed before the storm

The symptom — or rather the non-symptom — was the 2022 World Cup final producing record message spikes and the new cluster handling them calmly. The insight is that the migration's value was not abstract; it was cashed in on a specific, predictable, high-stakes day. The lesson is that **migrations should be timed against the calendar of known stress events**, not run whenever convenient. Being on the better system the week before your biggest traffic moment is worth far more than being on it the week after.

## When to reach for this architecture, and when not to

This case study is specific, but the decision framework generalizes. The full decision flow lives in [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree); here is the short version distilled from Discord's experience.

**Reach for a bucketed wide-column store (Cassandra or ScyllaDB) when:**

- Your dominant query is "the most recent N items in some entity, time-ordered" — feeds, message logs, event histories, time-series per device.
- Writes are heavy and append-shaped, and you can tolerate eventual consistency on reads.
- You need to scale horizontally without ever hand-resharding, and you can express a partition key that bounds partition size by construction (the bucket trick).
- You have, or can build, an application tier to handle hot-key concurrency — request coalescing, caching, fan-out control.

**Reach for ScyllaDB specifically over Cassandra when:**

- You are GC-bound: your tail latency is dominated by unpredictable pauses, and you have already tuned the JVM to its asymptote.
- Per-node efficiency matters to your cost structure — fewer, busier nodes beat many idle-then-stalling ones.
- You want Cassandra's data model and CQL API but a predictable, low tail latency, and you can run the shard-per-core deployment (including storage that keeps up).

**Skip this architecture — or at least think twice — when:**

- Your access pattern needs ad-hoc queries, joins, secondary-index-heavy lookups, or strong transactional guarantees. A wide-column store punishes everything that is not the query you designed for; reach for relational or a [globally distributed SQL](/blog/software-development/database-scaling/globally-distributed-sql-when-its-worth-it) system instead.
- Your data and traffic are small enough to fit comfortably on a well-tuned single primary with replicas. The operational complexity of a leaderless cluster is not free, and [vertical scaling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling) buys more runway than people expect.
- You cannot commit to disciplined data modeling. The schema *is* the performance contract here; if your queries will shift unpredictably, a more flexible store will hurt you less.
- You have heavy delete-and-overwrite churn on clustered data. Tombstone accumulation, as the Puzzles & Dragons incident shows, turns a delete-heavy workload into a read-latency problem.

The throughline of Discord's story is that none of these were one-time decisions. They picked MongoDB and outgrew it, picked Cassandra and rode it from billions to trillions of messages, and swapped the engine underneath when its runtime — not its design — became the wall. The data model they wrote in 2017, `(channel_id, bucket)` with newest-first clustering, is still the model running today. That longevity is the real lesson: get the partition key and the clustering order right, build the application tier to protect your hot keys, and you can change almost everything else underneath without your users ever noticing.

## Further reading

- Discord, ["How Discord Stores Billions of Messages"](https://discord.com/blog/how-discord-stores-billions-of-messages) (2017) — the original Cassandra data-model post.
- Discord, ["How Discord Stores Trillions of Messages"](https://discord.com/blog/how-discord-stores-trillions-of-messages) (2023) — the ScyllaDB migration post.
- [Cassandra and DynamoDB: a deep dive into leaderless wide-column stores](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive) — the engine family in full.
- [LSM trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) — why writes append and deletes leave tombstones.
- [Hot partitions and hot rows](/blog/software-development/database-scaling/hot-partitions-and-hot-rows) and [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) — the partition-design discipline this whole story turns on.

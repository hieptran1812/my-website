---
title: "Netflix on Cassandra: Running a Multi-Region Data Tier at Streaming Scale"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A case study of how Netflix runs one of the world's largest Apache Cassandra footprints as its primary online data tier — multi-region active-active, LOCAL_QUORUM in-region reads, EVCache in front, Priam underneath, and a data-abstraction platform that hides the database from every app team."
tags: ["database-scaling", "cassandra", "netflix", "multi-region", "tunable-consistency", "data-abstraction", "evcache", "priam", "wide-column", "high-availability", "case-study", "distributed-systems"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 29
---

When you press play on Netflix, a surprising amount of what happens next is a database read. The row of "Continue Watching" titles, the bookmark that remembers you were 23 minutes into an episode, which A/B test bucket your account lands in, the personalized artwork on each tile — almost all of it is per-member state that some service has to fetch in single-digit milliseconds, for every member, in every region, while an availability zone is allowed to catch fire without anyone getting paged. The store that has answered most of those reads for over a decade is Apache Cassandra, and the way Netflix runs it is one of the largest and most instructive deployments of a distributed database anywhere.

This post is a case study of that data tier. Not "what is Cassandra" — we have a [leaderless-database deep dive](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive) for the mechanics of hinted handoff, read repair, and Merkle trees. This is about the engineering decisions a company makes when Cassandra is the *primary* online store for a global streaming service: why a leaderless wide-column store and not a relational primary, how multi-region active-active actually behaves under a real region failure, what Netflix put *in front of* the database (EVCache) and *beside* it (the Priam sidecar), and the platform they built *on top of* it — the Data Gateway and its Key-Value Data Abstraction Layer — so that thousands of application engineers never have to know a Cassandra cluster exists.

![Active-active Cassandra rings in three AWS regions, each taking local writes at LOCAL_QUORUM and replicating asynchronously to the others](/imgs/blogs/netflix-on-cassandra-the-data-tier-1.webp)

The diagram above is the mental model for the whole article. Every AWS region runs a full Cassandra ring. Each ring accepts local writes — there is no single master that all writes must funnel through — and each ring asynchronously copies its mutations to every other region. An application in `eu-west-1` reads and writes its own regional ring at `LOCAL_QUORUM`, never paying a transatlantic round trip on the hot path, and the regions reconcile in the background. Hold that one picture: **write local, read local, replicate everywhere, reconcile eventually.** Nearly every design choice in this post falls out of it.

## Why a streaming data tier is a different problem

> Before you pick a database, write down the single sentence that describes the read you must serve in single-digit milliseconds while a data center is on fire. Everything downstream is a consequence of that sentence.

The instinct of most engineers, handed "store per-member state," is to reach for a relational primary with read replicas. That works beautifully until the workload takes on the shape a global streaming service actually has, at which point the assumptions quietly invert.

| Assumption | The naive view | The reality Netflix builds for |
| --- | --- | --- |
| One primary is fine | A leader takes writes; replicas fan out reads | A single primary is a single region's worth of latency and a single failure domain; writes must be accepted in every region |
| Failover is an event | Promote a replica when the primary dies, occasionally | Instance, zone, and even whole-region loss are *routine* — Netflix runs Chaos Monkey and Chaos Kong to force them on purpose |
| Strong consistency by default | Reads see the latest write, always | Most reads tolerate being a few hundred milliseconds stale; trading that for availability and latency is the entire point |
| Reads dominate | It's mostly a feed | Viewing events, bookmarks, and continue-watching updates are a relentless, unbounded write stream |
| The database is the interface | App teams write SQL against it | App teams should never see the database; they call a stable key-value API and the platform owns the storage |

Each row is a design constraint. The store must take writes in every region (rules out a single leader). It must survive a region evacuation without data loss or a frozen write path (rules out anything that blocks on a remote quorum). It must let most reads be cheap and slightly stale (rules out always-strong consistency). And — the constraint people underestimate — it must be hidden behind a platform, because a company with thousands of engineers cannot have every team learning Cassandra's sharp edges independently. Cassandra answers the first three constraints; the Data Gateway answers the fourth.

## 1. Why Cassandra: leaderless, multi-region, always-on

**The senior rule of thumb: choose an AP store when the cost of being unavailable exceeds the cost of being briefly stale — and for a streaming service the hot read path is exactly that case.**

Cassandra is a leaderless, masterless, wide-column store. "Leaderless" is the load-bearing word. In a leader-based system (a single-primary Postgres, a MongoDB replica set), every write goes to one node, and when that node is gone, writes stop until a new leader is elected. In Cassandra, every replica is symmetric. Any node can coordinate any write. There is no election, no promotion, no failover window — a node disappears and the other replicas keep taking writes as if nothing happened. When your operating reality includes Chaos Kong deliberately evacuating an entire AWS region during business hours, "no failover window" stops being a nice-to-have and becomes the only acceptable behavior.

That symmetry is what makes the topology in the mental-model figure possible. Cassandra's `NetworkTopologyStrategy` lets you declare a replication factor *per region*. Netflix runs a full replica set in each region, so each region can satisfy reads and writes entirely from its local ring. The cross-region copies travel asynchronously. A member in Singapore writes to `ap-southeast-1`, the write succeeds the moment the local replicas acknowledge, and the mutation propagates to `us-east-1` and `eu-west-1` in the background.

The second reason is **tunable consistency**, which we cover in depth in [tunable consistency at scale](/blog/software-development/database-scaling/tunable-consistency-at-scale). Cassandra does not force a global consistency choice on you; you pick a consistency level *per query*. The level that makes the whole architecture work is `LOCAL_QUORUM`: a majority of replicas *in the local region* must respond, and the query never waits on a remote region. That keeps the hot path's latency bounded by intra-region network (a millisecond or two) instead of inter-region (tens to hundreds of milliseconds), while still giving a strong-enough guarantee within the region.

The third reason is linear write scalability. Because there is no leader to serialize through, write throughput scales with node count: double the ring, roughly double the writes it absorbs. For a workload whose writes are unbounded — every play, pause, seek, and rating is a write — that property is not a luxury.

### Second-order consequence: you inherit the eventual-consistency tax

Leaderless availability is not free. The same asynchrony that lets every region take writes means a read in one region can miss a write that just landed in another. Two regions can accept conflicting writes to the same cell, and Cassandra resolves them with last-write-wins by timestamp — which silently discards the loser. The discipline this forces on app teams is real: design data models that are append-mostly or idempotent, avoid read-modify-write across regions, and never assume a write in `us-east-1` is visible in `eu-west-1` a millisecond later. Most of the platform features in section 5 exist precisely to make that tax payable without every team rediscovering it.

## 2. What actually lives on Cassandra

**The senior rule of thumb: Cassandra is the home for high-write, per-member online state — not for everything, and emphatically not for large immutable blobs.**

It is tempting to read "Netflix runs on Cassandra" as "everything is in Cassandra." It is not. Netflix is a textbook [polyglot-persistence](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store) shop, and Cassandra owns a specific slice: the per-member, high-write, low-latency online state.

![Matrix mapping data categories to primary store, front cache, write rate, and read consistency level](/imgs/blogs/netflix-on-cassandra-the-data-tier-2.webp)

The matrix above lays out the division. Viewing history, continue-watching bookmarks, profile and preference data, and A/B test membership all live in Cassandra and are fronted by EVCache. They share a shape: keyed by member, written constantly, read on the hot path, and tolerant of being momentarily stale. Artwork, encoded video, and other large immutable assets do *not* belong in Cassandra — they live in S3 and are served through a CDN, with EVCache holding only the small metadata that points at them. Putting multi-megabyte blobs in a wide-column store is one of the classic anti-patterns the platform is built to prevent; Cassandra partitions are meant to hold many small rows, not a few enormous ones.

Notice the read-level column changes by workload. Continue-watching is written so often, and matters so little if a single update is briefly lost, that it can read at `LOCAL_ONE` — one replica's answer is good enough. Profile data, which a member edits rarely but expects to see reflected, reads at `LOCAL_QUORUM`. That per-row tuning is the practical face of tunable consistency: the same cluster serves both, and the consistency knob is set by the access pattern, not the cluster.

Why Cassandra for this slice rather than a managed key-value store like DynamoDB, which the platform also supports as a backend? The honest answer is partly history — Netflix adopted Cassandra early and built deep operational muscle around it — and partly fit. The data is a natural match for the partition-plus-clustering model, the write rates reward a leaderless engine that scales horizontally, and running it on owned EC2 instances gives fine-grained control over compaction, repair cadence, and cost at a scale where managed per-request pricing would be punishing. The point is not that Cassandra wins every comparison; it is that for *this* slice — per-member, high-write, latency-sensitive, multi-region — it sits at a sweet spot the platform then standardizes so individual teams never re-litigate the choice.

## 3. The data model: one partition per member

**The senior rule of thumb: in Cassandra you design the table around the exact query you must serve, because there are no joins and no ad-hoc indexes to save you later.**

Take viewing history. The query the UI needs is "give me this member's most recent N viewing events, newest first." In a relational world you would store a flat `events` table and add `WHERE customer_id = ? ORDER BY event_time DESC LIMIT 50` with an index. In Cassandra you encode that query *into the primary key*.

![One member's partition holding viewing events clustered by event time descending so the most recent rows are physically first](/imgs/blogs/netflix-on-cassandra-the-data-tier-3.webp)

The figure shows the model. The partition key is `customer_id`, so every event for one member hashes to the same partition and lands on the same set of replicas (replication factor 3, so three nodes own a copy). Inside the partition, rows are *clustered* by `event_time` in descending order — meaning they are stored physically sorted, newest first. "Last 50 watched" is then not a search; it is reading the first 50 rows of a contiguous, pre-sorted slice. Here is the naive schema:

```cql
CREATE KEYSPACE viewing
  WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'us-east-1': 3,
    'eu-west-1': 3,
    'ap-southeast-1': 3
  };

CREATE TABLE viewing.history (
    customer_id  bigint,
    event_time   timestamp,
    title_id     bigint,
    position_ms  bigint,
    device       text,
    PRIMARY KEY ((customer_id), event_time)
) WITH CLUSTERING ORDER BY (event_time DESC);
```

And the read, which is now a single partition slice:

```cql
CONSISTENCY LOCAL_QUORUM;

SELECT title_id, position_ms, event_time
FROM viewing.history
WHERE customer_id = 42
LIMIT 50;
```

No `ORDER BY` is needed at read time — the clustering order already stored the rows newest-first, so `LIMIT 50` stops after the 50 physically-first rows. This is the single most important idea in Cassandra data modeling: **the sort order is a storage decision, not a query-time decision.** You pay for it once on write and read it for free forever.

### Second-order consequence: unbounded partitions will kill you

The naive schema has a latent bug that only appears at scale. A member who has watched Netflix for ten years accumulates a partition with tens of thousands of rows — and a partition is owned by a fixed set of replicas and read as a unit. Hot, ever-growing partitions become hot spots, slow scans, and compaction pain. The fix is to bound the partition with a *bucket* — typically a time window — added to the partition key:

```cql
CREATE TABLE viewing.history (
    customer_id  bigint,
    bucket       int,          -- e.g. 202606 for June 2026
    event_time   timestamp,
    title_id     bigint,
    position_ms  bigint,
    device       text,
    PRIMARY KEY ((customer_id, bucket), event_time)
) WITH CLUSTERING ORDER BY (event_time DESC)
  AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'DAYS',
    'compaction_window_size': 30
  };
```

Now each partition holds at most a month of one member's events, the read for "recent history" hits the current and previous buckets, and `TimeWindowCompactionStrategy` (TWCS) groups SSTables by time window so old data ages out cleanly. The shape of this problem — the [hot partition](/blog/software-development/database-scaling/hot-partitions-and-hot-rows) and how to bucket around it — recurs in every Cassandra deployment; getting the partition key right is most of the job.

## 4. Multi-region replication and tunable consistency

**The senior rule of thumb: keep the quorum local, let replication be asynchronous, and be explicit that read-your-writes only holds inside a region.**

The keyspace definition above already declared `NetworkTopologyStrategy` with a replica count per region. That single line is what makes the data tier multi-region. When the coordinator in `us-east-1` accepts a write, it writes to the three local replicas *and* forwards the mutation toward the replicas in `eu-west-1` and `ap-southeast-1` — but it does not wait on the remote ones. The consistency level decides who it *does* wait on.

![Client write reaching a coordinator that fans out to three local replicas and returns once two acknowledge, with the third replica catching up asynchronously](/imgs/blogs/netflix-on-cassandra-the-data-tier-4.webp)

The figure traces a `LOCAL_QUORUM` write. The client sends the write to a coordinator node in its region. The coordinator dispatches to all three local replicas, but returns success to the client as soon as a *quorum* of them — two of three — acknowledge. The third replica completes a moment later; the other regions complete in the background. The quorum math is worth internalizing. With replication factor `RF = 3`, a quorum is `floor(3 / 2) + 1 = 2`. If reads and writes both use `LOCAL_QUORUM`, then `R + W = 2 + 2 = 4 > RF = 3`, which guarantees the read and write quorums overlap on at least one replica — so within the region, a read is guaranteed to see the latest acknowledged write. That is read-your-writes consistency, *locally*.

Here are the levels Netflix-style deployments actually reach for, and what each one buys:

| Consistency level | Who must respond | Latency | Use it for |
| --- | --- | --- | --- |
| `LOCAL_ONE` | One replica in the local region | Lowest | Very high write rate, loss-tolerant (continue-watching ticks) |
| `LOCAL_QUORUM` | Majority in the local region | Low, bounded by intra-region | The default hot path; read-your-writes within a region |
| `EACH_QUORUM` | A quorum in *every* region (writes) | High | Rare writes that must be durable everywhere before ack |
| `QUORUM` | Majority across all replicas globally | Highest | Almost never on the hot path — defeats the point of going multi-region |

The trap is `QUORUM` (without `LOCAL_`). It sounds safer, and it is a latency disaster: it makes every query wait on a cross-region round trip, coupling your `eu-west-1` latency to the health of `us-east-1`. The whole reason to run a replica set in each region is so that a query never has to leave the region; `LOCAL_QUORUM` is what honors that. The cross-region copies still happen — just asynchronously, off the critical path. The cost is the eventual-consistency tax from section 1: a member who writes in one region and is then routed to another (rare, but it happens during a regional failover) may briefly read stale data. For viewing state, "briefly stale" is invisible. For anything where it is not, you redesign the data model rather than reach for global quorum. This is the central trade in any [multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture): in-region speed and always-on availability, paid for with eventual cross-region convergence.

## 5. The data-abstraction platform: Data Gateway and KV DAL

**The senior rule of thumb: at company scale, the database is an implementation detail that application teams should never touch — put a platform between them and it.**

This is the part of the Netflix story that most teams overlook, and it is arguably the most important. Running Cassandra well is hard. Data modeling is unforgiving, consistency levels are subtle, large values and pagination have sharp edges, and an anti-pattern shipped by one team can take down a shared cluster. Netflix's answer was not "train everyone to be a Cassandra expert." It was to build the **Data Gateway** — a platform that fronts the data tier — and on it, the **Key-Value Data Abstraction Layer** (KV DAL).

![Layered stack: thin client app, KV Data Abstraction Layer as a two-level map, Data Gateway Agent owning chunking and idempotency, and pluggable Cassandra, EVCache, and DynamoDB backends](/imgs/blogs/netflix-on-cassandra-the-data-tier-5.webp)

The stack reads top to bottom. An application service is a *thin client*: it speaks a fixed, simple API over gRPC and knows nothing about the storage underneath. Below it, the KV DAL presents a **two-level map**: a record is identified by an `id` (the hash key, which becomes the Cassandra partition key), and within that record, items are a sorted map of `sortKey` to `value` plus metadata (the clustering keys and columns). That is a deliberately Cassandra-shaped data model — partition plus clustered items — exposed as a clean key-value interface so that the common access patterns map straight through, but the *engine* is hidden. Below that sits the Data Gateway Agent, which owns the operational machinery, and at the bottom are pluggable backends: a namespace can be backed by Cassandra, by EVCache, by DynamoDB, or by a combination.

The client API is intentionally small. The core operations are `PutItems`, `GetItems`, and `DeleteItems`:

```python
from kvdal import KeyValueClient, ConsistencyLevel
from uuid import uuid4

# A namespace, not a cluster. The platform decides what backs it.
kv = KeyValueClient(namespace="viewing_history")

# PutItems: id (hash key) -> one or more sorted items keyed by sort_key.
kv.put_items(
    id=f"customer:{customer_id}",
    items=[{
        "sort_key": event_time_iso,          # becomes the clustering key
        "value": encode(title_id, position_ms),
        "metadata": {"device": device},
    }],
    idempotency_token=uuid4().hex,           # safe, exactly-once retries
    consistency=ConsistencyLevel.LOCAL_QUORUM,
)

# GetItems: page the most recent items back, newest first.
page = kv.get_items(
    id=f"customer:{customer_id}",
    sort_key_order="DESC",
    page_size_bytes=64 * 1024,               # adaptive, byte-based paging
    consistency=ConsistencyLevel.LOCAL_QUORUM,
)
```

Every parameter here is hiding a hard problem the platform solved once, for everyone:

- **Idempotency tokens.** Distributed writes get retried; a naive retry can double-apply. The token lets the platform make a retried `PutItems` a no-op, turning at-least-once delivery into effectively exactly-once semantics — without each team hand-rolling deduplication.
- **Chunking.** Cassandra punishes large values, but application objects are sometimes large. The DAL transparently splits an oversized value across multiple physical rows on write and reassembles it on read, so the team never sees the limit.
- **Client-side compression.** Values are compressed before they hit the wire and the SSTable, cutting both network and storage, transparently.
- **Adaptive, byte-based pagination.** Instead of "give me 100 rows" (which can be 1 KB or 100 MB), the API pages by *bytes*, so a single `GetItems` can never accidentally pull a multi-megabyte response and blow a latency budget.
- **Namespaces over clusters.** A `namespace` is a logical store. You can start with every namespace on one shared Cassandra cluster, and when one namespace outgrows it — more traffic, more data, noisier neighbor — the platform moves it to its own dedicated cluster with no application code change. The team keeps calling `namespace="viewing_history"`; the routing underneath changes.

That last point is the whole payoff. Consistency level, backend engine, cluster placement, even *which database* — all of it is configuration the platform owns, not code the application owns. Netflix has described a single KV-backed service — a real-time distributed graph used internally — running on the order of tens of namespaces over a dozen Cassandra clusters and a couple thousand EC2 instances, serving millions of reads and writes per second, all behind this same key-value surface. The application teams calling it did not have to become distributed-database experts to get that.

### Second-order consequence: the platform is also a guardrail

A subtle benefit of routing every access through the gateway is that anti-patterns become *catchable*. An unbounded query, a value that should have been chunked, a consistency level that does not match the workload — the platform can observe, shape, throttle, or reject these before they reach a shared cluster and degrade it for every other tenant. The database stops being a foot-gun that any team can fire. The cost is real — you now operate a platform tier, with its own deployment, sharding, and on-call — but at Netflix's scale the alternative (every team against raw Cassandra) is far more expensive in outages.

## 6. EVCache in front of Cassandra

**The senior rule of thumb: a quorum read is cheap, but a cache hit is cheaper — and for precomputed, read-heavy data the cache, not the database, should serve the hot path.**

Cassandra at `LOCAL_QUORUM` is fast, but it is not free: it is two replicas doing disk and network work per read. For the hottest, most read-heavy data — personalized rows, "because you watched," home-screen tiles, much of which is precomputed offline — Netflix fronts Cassandra with EVCache, its multi-replica memcached layer. The full design is in [Netflix EVCache: a cache that survives an AZ going dark](/blog/software-development/database-scaling/netflix-evcache-multi-region-cache); here we care about how it changes the read path.

![Read path branching from a local EVCache lookup: a hit returns sub-millisecond, a miss reads Cassandra at LOCAL_QUORUM and backfills the cache](/imgs/blogs/netflix-on-cassandra-the-data-tier-6.webp)

The figure is a lookaside cache. A read first asks EVCache in the local availability zone. On a hit — the common case for precomputed data — it returns in sub-millisecond time and Cassandra is never touched. On a miss, the service reads Cassandra at `LOCAL_QUORUM`, then writes the value back into EVCache with a TTL so the next read is a hit. In code:

```python
def get_continue_watching(customer_id: int) -> list[Title]:
    key = f"cw:{customer_id}"

    cached = evcache.get(key)                 # local-AZ read, sub-ms
    if cached is not None:
        return decode(cached)                 # hit: Cassandra untouched

    rows = kv.get_items(                       # miss: one LOCAL_QUORUM read
        id=f"customer:{customer_id}",
        sort_key_order="DESC",
        limit=20,
        consistency=ConsistencyLevel.LOCAL_QUORUM,
    )
    value = build_continue_watching(rows)
    evcache.set(key, encode(value), ttl_seconds=3600)   # backfill for next time
    return value
```

Two things make this more than a textbook cache. First, EVCache is itself multi-replica and AZ-aware: it writes a copy into every availability zone and reads only from the local one, so losing a zone does not punch a hole in the cache. Second, the failure mode it guards against is not "slightly slower" but "self-inflicted outage." The origin services behind the cache are sized for the *miss* trickle, not the full read load, because the entire point of precomputation was to keep that load off them. If a large slice of the cache went cold at once, the resulting stampede onto Cassandra and the origin could take the service down — which is exactly the [thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd) problem, and why request coalescing and careful TTL jitter matter as much here as the cache itself.

## 7. Operating Cassandra at scale: Priam, repairs, and compaction

**The senior rule of thumb: the database engine is maybe a third of the problem; the other two-thirds are tokens, backups, repairs, and compaction — automate them or drown.**

Running hundreds of Cassandra clusters across multiple regions is an operational undertaking that dwarfs the act of issuing queries. Netflix's answer is automation packaged as a sidecar: **Priam**, a co-process that runs *beside every Cassandra node* and takes the manual toil off the operators. (The name is a fitting in-joke: Priam was the King of Troy and the father of Cassandra in Greek myth.)

![Matrix of Priam responsibilities — token assignment, backups, configuration, monitoring, bootstrap and restore — each with its mechanism and the consequence of its absence](/imgs/blogs/netflix-on-cassandra-the-data-tier-7.webp)

The matrix enumerates what the sidecar carries. **Token assignment**: Priam gives each node a stable, deterministic token so that replacing a dead instance preserves the ring's data ownership, coordinating assignments through a small external registry rather than manual `nodetool` surgery. **Backups**: Priam continuously ships SSTables to S3 — both periodic snapshots and incremental backups — compressing them on the fly, so a cluster can be restored to a point in time from object storage. **Configuration**: it centralizes Cassandra's many config knobs per cluster, preventing the config drift that makes one node behave subtly differently from its peers. **Monitoring**: it exposes health and metrics over JMX so the fleet is observable. **Bootstrap and restore**: when a node dies, Priam can stand up a replacement, restore its data from S3, and rejoin it to the ring with minimal human involvement. Strip any one of these away and you are back to running Cassandra by hand at a scale where that is not survivable.

Beyond Priam, two background processes define day-to-day operability, and both are easy to neglect until they bite:

```bash
# Anti-entropy repair: reconcile divergent replicas. Run it continuously
# and incrementally (a scheduler like Cassandra Reaper handles this), not
# as a big-bang job. -pr repairs only this node's primary ranges so the
# fleet doesn't redundantly repair the same data.
nodetool repair -pr viewing history

# Watch the things that quietly degrade reads:
nodetool tablestats viewing.history     # partition sizes, tombstone ratios
nodetool compactionstats                # is compaction keeping up with writes?
```

**Repair (anti-entropy)** is how replicas that have drifted — because of a missed write, a node that was down, or hinted handoff expiring — get reconciled. Skip it and deleted data can resurrect, because a delete in Cassandra is a *tombstone* (a marker), and if a replica that missed the delete is never repaired before the tombstone is garbage-collected, the old value can come back. Repair has to complete on every range within the `gc_grace_seconds` window. At Netflix's scale repair is not a cron job you forget; it is a managed, throttled, never-ending background flow.

**Compaction** is the other one. Cassandra writes are append-only into immutable SSTables, and compaction merges them to reclaim space and bound read amplification. The strategy you pick has to match the data:

| Compaction strategy | Best for | Watch out for |
| --- | --- | --- |
| `SizeTieredCompactionStrategy` (STCS) | Write-heavy, general purpose | Read amplification; transient 2x space during compaction |
| `LeveledCompactionStrategy` (LCS) | Read-heavy, update-in-place | Higher write amplification and CPU |
| `TimeWindowCompactionStrategy` (TWCS) | Time-series, TTL'd data (viewing history) | Out-of-window writes ruin the grouping |

Viewing history is time-series with natural aging, which is why the bucketed schema in section 3 chose TWCS: events written for a given window land in the same SSTables and age out together, so old data is dropped by expiring whole files instead of rewriting them. Pick the wrong strategy and you either burn CPU you did not need to (LCS on append-only data) or accumulate tombstones and read amplification (STCS on TTL'd time-series). The maintenance burden is the honest price of Cassandra's availability, and the lesson of the Netflix deployment is that this price is paid with *automation* — Priam, Reaper-style repair scheduling, and per-table compaction tuning — not with heroics.

## Case studies from production

These are the failure modes a data tier of this shape actually hits. Some are documented Netflix realities; others are the canonical Cassandra-at-scale traps the architecture is explicitly built to absorb. Each is the kind of incident that turns an architectural choice from theory into muscle memory.

### 1. The region that went dark

Netflix runs Chaos Kong, an exercise that evacuates an entire AWS region on purpose. The first time a team experiences it on a Cassandra-backed service, the striking thing is how little happens to the data tier. Traffic that was hitting `us-east-1` shifts to the surviving regions; those regions were already full replicas taking local writes at `LOCAL_QUORUM`, so they keep serving without a failover window. The only visible effect is a brief consistency wrinkle: members re-routed mid-write might read state that has not yet replicated to the new region. Because the affected data is viewing state, the wrinkle is invisible to members. The lesson is that active-active is not a disaster-recovery plan you dust off; it is the steady-state design, validated continuously by deliberately breaking it. A passive standby region you have never failed to is a region you do not actually have.

### 2. The ten-year partition

A power user's continue-watching partition, under the naive `PRIMARY KEY ((customer_id), event_time)` schema, grows without bound. One day a read that used to take two milliseconds takes two hundred, and `nodetool tablestats` shows a single partition holding tens of thousands of rows across dozens of SSTables. The wrong first hypothesis is "the cluster is overloaded" — but other partitions are fine. The root cause is an unbounded partition: a hot, ever-growing unit owned by a fixed replica set. The fix is the bucketed key from section 3, splitting the partition into bounded time windows. The deeper lesson is that Cassandra data models have a time bomb the load test never finds, because the load test does not run for ten simulated years. You bound partitions on day one, by policy, not after the incident.

### 3. The tombstone avalanche

A table that deletes or expires rows — say, A/B test memberships that churn — starts returning slowly, and the logs show queries scanning thousands of tombstones to return a handful of live rows. Every delete left a marker, and reads have to walk past all of them. The wrong fix is to delete more aggressively, which makes it worse. The right fixes are structural: stop modeling as delete-heavy mutable rows, lower `gc_grace_seconds` cautiously (only if repair reliably completes inside it), and switch TTL'd time-series tables to TWCS so expired data is dropped by aging out whole SSTables rather than accumulating per-row tombstones. Tombstones are the single most common way a healthy Cassandra table quietly becomes a slow one, and they are entirely a data-modeling problem wearing an operations costume.

### 4. The repair that never finished

A cluster grows, and the nightly repair job that used to finish in four hours starts taking thirty and overlapping the next run. Eventually it stops completing within `gc_grace_seconds`, and the on-call starts seeing deleted data reappear on some replicas. The naive response is to disable repair to "reduce load" — which removes the only thing keeping replicas consistent. The real fix is to make repair continuous and incremental: a scheduler (Cassandra Reaper is the standard) that repairs small token ranges around the clock with `-pr`, throttled so it never competes with serving traffic. At fleet scale, repair is a managed background service with its own SLOs, not a job. The lesson many teams learn the hard way: an un-repaired Cassandra cluster is a correctness bug accumulating interest.

### 5. The cold-cache stampede

An EVCache layer in front of viewing data gets flushed — a bad deploy, an eviction storm, a zone blip — and a flood of misses hits Cassandra and the origin services simultaneously. The origin was sized for the miss trickle, not the full load, so it browns out, which slows responses, which fills queues, which cascades. The first instinct, "scale up the origin," is too slow to matter mid-incident. The durable fixes live in the read path: request coalescing so a thousand simultaneous misses for the same key collapse into one Cassandra read, TTL jitter so a whole cohort of keys does not expire in the same second, and treating a cold cache as a first-class failure mode in capacity planning. The cache is not a performance optimization here; it is load-bearing, and a load-bearing cache must fail soft.

### 6. The schema change across hundreds of clusters

A team needs to add a field to a record type that, across the company, is backed by dozens of namespaces on many clusters. In a world of raw Cassandra access, this is dozens of coordinated `ALTER TABLE` operations and a fragile migration across teams. Routed through the Data Gateway and KV DAL, the value is an opaque, versioned, compressed blob — so a new field is an application-level serialization change, and the storage layer never has to be altered in lockstep. The platform turns a fleet-wide schema migration into a client-side concern. The lesson is that the key-value platform pays its largest dividends not on the hot path but on the *change* path: the day you need to evolve a data type across the whole company without a coordinated database migration.

### 7. The hot coordinator

A single key turns hot — a title goes viral, and one partition's reads spike far above its neighbors. With `LOCAL_QUORUM`, two replicas per read multiplied by the spike saturates those nodes. The wrong move is to bump the whole cluster's capacity for one key. The right moves are targeted: front the hot key with EVCache so most reads never reach Cassandra, drop the read level to `LOCAL_ONE` for that loss-tolerant access pattern, and, if the heat is structural rather than transient, reshape the key so the load spreads. The recurring lesson across these incidents is that the consistency level and the cache are *per-access-pattern* dials, not cluster-wide constants — the same data can be served three different ways depending on how hot and how loss-tolerant a given read is.

### 8. The write that vanished across regions

A member flips a profile setting in `eu-west-1`, and a moment later a request that gets routed to `us-east-1` — because a load shift moved their session — shows the old value. Support receives a "my change didn't save" ticket. The write *did* save; it simply had not crossed the Atlantic yet, and read-your-writes only holds inside a region. The tempting fix is to switch that read to global `QUORUM` so it always sees the latest write everywhere — which couples every read's latency to remote-region health and dismantles the entire reason for going multi-region. The durable fixes are cheaper and local: pin a member's session to one region so their reads follow their writes, and model settings so that a momentarily stale read is self-correcting rather than alarming. The lesson is that eventual consistency is not a footnote in a design doc; it is a concrete support ticket you will eventually receive, and the architecture has to decide in advance whether the answer is session affinity, a tolerant data model, or — rarely, and only for the one access pattern that truly needs it — a stronger consistency level. Choosing "stronger everywhere" to avoid the conversation is how teams accidentally rebuild the single-region latency they left Cassandra to escape.

## When to reach for this architecture, and when not to

Reach for a Cassandra-style leaderless data tier when:

- You need **writes accepted in every region** with no failover window — availability is worth more to you than immediate global consistency.
- Your reads tolerate being **briefly stale**, so `LOCAL_QUORUM` (or `LOCAL_ONE`) on a local replica set is good enough.
- The workload is **high-write and partition-friendly** — keyed by an entity (member, device, account) with queries that map to a partition plus a clustering order.
- You can **invest in operability**: token management, S3 backups, continuous repair, and compaction tuning, ideally automated by a sidecar and a platform team.
- You have **enough teams** that hiding the database behind a key-value platform pays for the platform itself.

Skip it, or reach for something else, when:

- You need **multi-key transactions or strong global consistency** as the default — a NewSQL system or a single-region relational primary will serve you with far less pain. (See [globally distributed SQL: when it is worth it](/blog/software-development/database-scaling/globally-distributed-sql-when-its-worth-it).)
- Your data is **relational and join-heavy** with ad-hoc query patterns you cannot pin down in advance — Cassandra's "model the query into the key" discipline becomes a straitjacket.
- You are storing **large immutable blobs** — that is S3-and-a-CDN, not a wide-column store.
- Your scale is **one region and modest write volume** — the operational cost of a Cassandra fleet, repairs, and compaction tuning is pure overhead you will not recoup. Vertical scaling or a managed relational store is the cheaper answer until you genuinely outgrow it.
- You **cannot staff the operations** — an unrepaired, untuned Cassandra cluster degrades silently, and "we'll automate it later" is how the tombstones and partition hotspots find you.

The throughline of the Netflix data tier is not "Cassandra is the best database." It is a sequence of deliberate trades: pick an always-on leaderless store because a region must be allowed to fail; keep the quorum local so latency stays bounded; accept eventual cross-region consistency and design data models that tolerate it; put a cache in front for the hottest reads and a sidecar beside every node for operability; and — the move that scales the *organization*, not just the database — hide all of it behind a key-value platform so the database becomes an implementation detail. Any one of those choices is ordinary. Doing all of them coherently, at streaming scale, for over a decade, is what makes it worth studying.

## Further reading

- [Cassandra and DynamoDB: a leaderless deep dive](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive) — the mechanics of hinted handoff, read repair, and quorum that this post builds on.
- [Tunable consistency at scale](/blog/software-development/database-scaling/tunable-consistency-at-scale) — the full `LOCAL_ONE` / `LOCAL_QUORUM` / `QUORUM` decision space and the `R + W > RF` math.
- [Netflix EVCache: a cache that survives an AZ going dark](/blog/software-development/database-scaling/netflix-evcache-multi-region-cache) — the multi-replica, AZ-aware cache that fronts this data tier.
- [Multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture) — the broader design space of geo-distributed data and the consistency trades involved.
- Netflix Tech Blog: the Key-Value Data Abstraction Layer, the Data Gateway platform, and the Priam project on GitHub — the primary sources for the platform and sidecar described here.

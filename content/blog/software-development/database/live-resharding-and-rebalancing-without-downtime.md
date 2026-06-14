---
title: "Live Resharding: Rebalancing a Distributed Database Without Downtime"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer's runbook for moving live data between shards with the workload still running: rebalancing strategies, the snapshot-plus-tail copy protocol, VDiff verification, atomic routing cutover, and how Vitess, CockroachDB, Cassandra, Redis Cluster, and DynamoDB each do it."
tags:
  [
    "resharding",
    "rebalancing",
    "sharding",
    "distributed-systems",
    "zero-downtime",
    "vitess",
    "redis-cluster",
    "cockroachdb",
    "cassandra",
    "scalability",
    "databases",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/live-resharding-and-rebalancing-without-downtime-1.webp"
---

There is a particular kind of dread that settles over an engineering team when the dashboards say what nobody wants to hear: one shard is running hot, it is at 92% CPU at peak, its disk IOPS are pinned, and the only way out is to move some of its data somewhere else. The dread is not about the move itself. Copying bytes from one machine to another is trivial — `pg_dump | pg_restore`, `mysqldump`, `rsync`, take your pick. The dread is that you cannot stop. The application is still taking writes. Users are still editing documents, placing orders, posting comments, the whole time you are trying to move the ground out from under them. You are performing an engine swap on a car doing seventy on the freeway, and the passengers must not feel a bump.

This is resharding, and it is the single hardest operational task in the life of a distributed data tier. It is harder than the initial decision to shard. It is harder than choosing the shard key, harder than writing the cross-shard query layer, harder than the failover runbook. Those are all design problems you solve once, at leisure, with a whiteboard. Resharding is a *live* problem: the data is enormous (terabytes, not gigabytes), the traffic is real and unforgiving, you are not allowed to stop writes, and consistency must hold at every instant — there can never be a moment where a read returns the wrong answer because a row was in flight between its old home and its new one. Every team that has scaled a real database has a re-shard war story, and almost all of them describe it the same way: months of preparation for a cutover that, if it goes well, nobody outside the team ever notices.

![A single hot shard at 92% CPU caps the whole tier; resharding spreads the same keys onto more nodes](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-1.webp)

The diagram above is the mental model for the entire article. On the left is the problem: a cluster where most shards have headroom but one shard — shard 3 — is the ceiling. It is at 92% CPU and its IOPS are saturated, so the *whole tier* is effectively at capacity even though shards 0 and 1 are loafing at 22% and 31%. Adding capacity is useless until shard 3's load is split. On the right is the goal: the same keys, rebalanced, so that no single node is the bottleneck. Everything between those two pictures — how you move the data, how you keep it correct, how you flip traffic atomically, and how you back out if it goes wrong — is what this article is about. It is also the capstone of a long series on databases and distributed systems, so along the way we will tie back to the [partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) decisions, the [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) ring, the [change data capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) machinery, and the replication and consensus posts that make all of this possible.

We will lean heavily on Martin Kleppmann's *Designing Data-Intensive Applications* (DDIA), Chapter 6, which is the canonical treatment of rebalancing partitions. The vocabulary there — fixed number of partitions, dynamic partitioning, partitioning proportional to nodes, and request routing — is the scaffold for everything below, restated in operational terms and grounded in how real systems actually move data while live.

## Why resharding is genuinely the hardest thing

**Senior rule of thumb: the difficulty of resharding is not the copy. It is that the copy must converge to a moving target, and the cutover must be atomic with respect to that target.**

Let us be precise about why this is hard, because the precision is what tells you which techniques are non-negotiable and which are nice-to-have. There are four constraints, and they fight each other.

First, **the data is huge.** A hot shard is hot because it holds a lot of data and serves a lot of traffic; that is the same reason it needs resharding. We are not moving a configuration file, we are moving hundreds of gigabytes to terabytes. A bulk copy at a respectful, throttled rate — say 20 MB/s so we do not starve the live workload — takes hours per shard. Notion's re-shard, which we will dig into below, spent twelve hours per database just on the initial sync even after aggressive optimization, and three full days before that optimization. During those hours, the source keeps changing.

Second, **the traffic is live and you cannot stop writes.** This is the constraint that turns a backup-and-restore into a distributed-systems problem. If you could take a five-minute maintenance window, freeze all writes, copy, and flip, resharding would be a junior-engineer task. But for a product at scale, even a few seconds of write unavailability is a visible incident, and a five-minute freeze is a press release. So the copy has to happen *concurrently with writes to the source*, which means the moment your bulk copy finishes reading row N, rows 1 through N-1 may already be stale.

Third, **consistency must hold throughout.** It is not enough that the data ends up correct after the dust settles. At every instant of the migration, a read must return a correct answer, and a write must land somewhere it will not be lost. There can be no window where a key's writes go to the old shard while its reads come from the new one (you would read stale data), and no window where a key exists on neither shard or on both with diverging values (you would lose or corrupt data). The cutover — the instant a key's ownership transfers — must be atomic with respect to that key's traffic.

Fourth, **it must be reversible.** Cutovers go wrong. A subtle bug in the new shard's schema, a hot-loop in the query rewriter, an unforeseen interaction with a foreign key — any of these can surface only under real production traffic, minutes after you flip. If your only path forward is forward, you are betting the business on a clean cutover every time. Mature reshard tooling keeps the old copy alive and continuously updated (a *reverse replication stream*) so that rollback is itself a fast, safe routing flip rather than a restore-from-backup scramble.

Here is the table I keep in my head when someone says "let's just copy the data over":

| What you'd assume | What's actually true | Why it bites |
| --- | --- | --- |
| "It's just a copy." | The source mutates during the copy. | A snapshot taken at T0 is stale by T1; you must replay the delta. |
| "We'll take a short window." | Even seconds of write-downtime is a visible outage at scale. | The copy takes hours; the window would be hours. |
| "We'll verify at the end." | "At the end" never comes — the source keeps moving. | Verification must run against a frozen snapshot point, then track the tail. |
| "If it breaks, we'll restore." | Restore-from-backup loses every write since the backup. | You need a live, reversible old copy, not a cold backup. |
| "The hard part is the new shards." | The hard part is the routing flip's atomicity. | A non-atomic flip splits a key's reads and writes across two homes. |

The rest of this article is a guided tour of how the industry solves these four constraints at once. We start with the strategy layer — *how* you decide which data moves where — because the move protocol differs subtly depending on whether you are moving hash slots, splitting key ranges, or handing off a virtual node on a consistent-hashing ring.

## Rebalancing strategies, and why `hash % N` is forbidden

**Senior rule of thumb: never make a key's home a function of the current node count. The number of nodes is the one thing that changes during a reshard, and any scheme keyed on it remaps almost everything.**

Before any bytes move, you need a *rebalancing strategy*: a rule that says, given the current set of nodes and the set of keys, which keys live where, and what happens to that mapping when you add or remove a node. DDIA Chapter 6 enumerates the sane options and, more usefully, the one insane option that everyone reaches for first.

The insane option is **`hash(key) mod N`**, where `N` is the number of nodes. It is the most natural thing in the world: hash the key, take it modulo the node count, that's the node. It distributes keys evenly and is trivial to compute. It is also catastrophic the moment `N` changes, which is *exactly the moment a reshard happens*. Go from 10 nodes to 11 and `hash(key) mod 11` sends almost every key to a different node than `hash(key) mod 10` did — roughly `1 - 1/N` of all keys, so about 90% of your data, must move. A scheme whose entire purpose is to spread keys becomes a scheme that, on any topology change, requires moving nearly everything. We covered the full failure mode in the [consistent hashing post](/blog/software-development/database/consistent-hashing-and-data-partitioning); the one-sentence summary is that `mod N` couples the key-to-node mapping to the node count, and rebalancing is precisely the act of changing the node count. So it is forbidden. Every production system uses one of the three strategies below instead, each of which *decouples* the mapping from `N`.

![Fixed slots move slot-sets, range splitting splits on size or load, ring moves hand off virtual-node ownership](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-6.webp)

The matrix above is the strategy menu. Read it as: each family moves a different *unit*, triggers a move on a different *signal*, and reroutes ownership through a different *mechanism*. Let us take them one at a time.

### Strategy 1: a fixed number of partitions (hash slots)

The idea, in DDIA's terms, is to create *many more partitions than nodes* up front — say 16,384 partitions across 6 nodes — and assign multiple partitions to each node. The number of partitions is fixed forever at cluster creation; only the *assignment of partitions to nodes* changes during a reshard. To add a node, you steal a few whole partitions from each existing node and hand them to the newcomer. Critically, **the key-to-partition mapping never changes** — `partition = hash(key) mod 16384` is stable because 16384 is a constant, not the node count — so a key never has to move *because of its hash*; it only moves if its partition is reassigned, and partitions move as indivisible units.

The canonical implementation is **Redis Cluster's 16384 hash slots**. Every key maps to a slot via `HASH_SLOT = CRC16(key) mod 16384`, and slots are distributed across master nodes. Resharding is the act of moving slots (and the keys they contain) from one node to another; the slot-to-node map is the routing table. Elasticsearch uses the same idea with a fixed number of primary shards chosen at index creation that cannot change afterward without a reindex — the rigidity is the price of the simplicity. The tradeoff DDIA highlights: you must pick the partition count well at the start. Too few and you cannot spread across enough nodes later; too many and each partition carries fixed overhead (metadata, file handles, rebalancing bookkeeping). 16384 is Redis's bet that you will never have more than a few thousand nodes but want fine-grained slot movement.

### Strategy 2: dynamic range partitioning (split on size or load)

The second family does not fix the partition count. Instead it partitions by *contiguous key ranges* — partition A owns `[a, c)`, partition B owns `[c, f)`, and so on — and **splits a range in two when it grows too large or too hot**, then potentially moves one half to another node. This is *dynamic partitioning*: the number of partitions grows with the data, adapting to the actual distribution rather than a guess made at creation time.

HBase does this with region splits. **CockroachDB** does it beautifully: its keyspace is divided into ranges that aim to stay between 32 MB and 64 MB, and a background `splitQueue` splits any range that exceeds the size threshold *or* that load-based splitting identifies as carrying disproportionate traffic. A range that gets hot is split into two ranges with two separate leaseholders, and the allocator then rebalances those ranges across nodes so each node carries roughly equal load. Spanner takes the same range-split approach. The win is adaptivity: ranges form where the data actually is, so an empty key range costs nothing and a dense one gets subdivided automatically. The cost is that range boundaries are data-dependent and shift over time, so the routing layer needs an up-to-date map of which range lives where (CockroachDB keeps this in a meta-range that is itself a range).

### Strategy 3: consistent hashing ring moves (virtual nodes)

The third family is *partitioning proportional to nodes*, realized through a **consistent hashing ring**. Keys and nodes both hash onto a ring; a key belongs to the first node clockwise from it. Adding a node inserts a new point on the ring and steals only the arc between it and its predecessor — so only `1/N` of the keys move, not `(N-1)/N`. To smooth out the lumpiness of a single point per node, each physical node owns many **virtual nodes** (vnodes) scattered around the ring, so its load is the sum of many small arcs rather than one large one.

This is how **Cassandra**, **Amazon Dynamo**, and **Riak** distribute data, and it is covered in depth in the [consistent hashing post](/blog/software-development/database/consistent-hashing-and-data-partitioning). The operational consequence for resharding is lovely: adding a node is the rebalance. You bootstrap a new node, it claims a set of token ranges (vnodes), and it streams exactly those ranges from the nodes that previously owned them. The move primitive is "hand off ownership of these token ranges," propagated via gossip so every node learns the new token map.

The three families converge on the same principle, which is the real lesson of DDIA Chapter 6: **separate the key-to-partition mapping (stable, never keyed on `N`) from the partition-to-node mapping (mutable, this is what a reshard changes).** Fixed slots make the first mapping a constant modulo; ranges make it a range lookup; the ring makes it a clockwise walk. In every case, the thing that changes during a reshard is *which node owns a partition*, never *which partition owns a key*. That separation is what makes a reshard a bounded move instead of a full reshuffle.

## The live-move protocol: snapshot, tail, catch up, verify, cut over

**Senior rule of thumb: a live move is a convergence problem. Copy a consistent snapshot, simultaneously tail the change stream, and drive the lag between them to zero before you even think about flipping traffic.**

Now we get to the heart of it. Regardless of which strategy decides *where* data goes, the *act* of moving a unit of data (a slot, a range, a token range, a whole logical shard) from a live source to a live target follows one protocol. Every mature system — Vitess, CockroachDB, Cassandra, Redis, the bespoke pipelines at Notion and Figma — is a variation on it. Internalize this protocol and the rest is implementation detail.

![Bulk-copy a consistent snapshot while tailing the change stream, drive lag to zero, prove equality, then flip routing](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-2.webp)

The timeline above is the whole dance. There are six phases.

**Phase 1 — Mark the source and start the change stream (T0).** Before you copy a single row, you establish a *consistent starting point* and begin capturing every change from that point forward. Concretely: you note the current replication position — a MySQL GTID or binlog coordinate, a Postgres LSN, a Cassandra commit-log position — and you start a consumer of the change stream from exactly that position. This is the [change data capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) machinery doing its job: the binlog, WAL, or commit log is an ordered record of every mutation, and by anchoring the stream consumer at the same position as the snapshot, you guarantee that no change is missed and none is double-applied. This anchoring is the single most important step; get it wrong and your target silently diverges.

**Phase 2 — Bulk-copy a consistent snapshot (T0 to T1).** Now you copy the data as of position T0. A consistent snapshot — Postgres `pg_export_snapshot`, MySQL's `START TRANSACTION WITH CONSISTENT SNAPSHOT`, or a storage-level snapshot — gives you a frozen, transactionally consistent view at exactly the position where the change stream began. You stream rows from this snapshot to the target. A high-leverage optimization that recurs across every real writeup: **skip building secondary indexes during the copy.** Inserting into an indexed table is dramatically slower than bulk-loading an unindexed one, so you copy the raw rows fast and build indexes afterward. Notion used exactly this trick to cut sync time from three days to twelve hours.

**Phase 3 — Tail the change stream (T1 to T2).** The snapshot is now loaded but stale — it reflects T0, and we are at T1, hours later. Every write that hit the source since T0 is sitting in the change stream you started in phase 1. You replay them onto the target, in order, idempotently. As you apply them you also keep consuming new changes, because the source is still taking writes. This is a race: you are applying changes at some rate while new changes arrive at the live write rate. As long as your apply rate exceeds the write rate (which it will, because applying a pre-computed change is cheaper than executing the original statement), the gap — the *replication lag* — shrinks.

**Phase 4 — Catch up to near-zero lag (T2).** You wait until the lag between source and target is small and stable — typically under a second. At this point the target is a near-perfect, continuously updated replica of the source's slot/range/shard. The lag will never be exactly zero while the source takes writes (there is always a change in flight), but it can be made small enough that the final cutover window is sub-second.

**Phase 5 — Verify equality (T3).** Before you trust the target with real traffic, you *prove* it matches the source. The gold standard is a **VDiff-style checksum**: you compute a checksum (or row-by-row comparison) of the source and target over the migrated range and confirm they are identical. Vitess's VDiff captures consistent snapshots of source and target and compares them; Notion ran "dark reads" — every production read was also issued against the new database and the results compared — and reached near-100% equivalence before flipping. We will spend a full section on verification because it is the difference between a confident cutover and a hopeful one.

**Phase 6 — Atomic cutover (T4), then keep the old copy (T5).** Finally, you flip the routing layer so the slot/range/shard's traffic goes to the target. This is a metadata change, not a data move — the bytes are already there. To make it atomic with respect to in-flight writes, mature systems briefly pause writes to the moving unit (Redis), or take a short read-only/two-phase window (Vitess, Figma's ten-second window), apply the last few buffered changes, then switch. And crucially, you do *not* delete the old copy. You keep it hot and, ideally, start a **reverse replication stream** so the old copy keeps receiving the new copy's writes — which makes rollback a second routing flip rather than a catastrophe.

That is the protocol. Snapshot, tail, catch up, verify, flip, retain. Now let us look at what holds each phase together: the dual-track copy, the routing layer, verification, and rollback.

### The dual track: snapshot and stream, side by side

![The change stream keeps the target in sync; dark reads compare old and new before any traffic flips](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-3.webp)

The graph above shows the runtime topology during the move: the application talks to the routing layer, which keeps the source authoritative while the target shadows it via the change stream, and dark reads compare both sides until equality is proven. The leap that trips people up is that phases 2 and 3 are not sequential — they overlap, and they must. If you copied the snapshot first and *then* started capturing changes, you would lose every write that happened during the copy. The change stream must start at or before the snapshot position and run continuously, so that the snapshot (a point-in-time copy) plus the stream (everything after that point) together reconstruct the current state. Snapshot answers "what was true at T0"; stream answers "what changed since T0"; their sum is "what is true now."

```python
# Sketch of the dual-track copy. Real systems (Vitess VReplication,
# Debezium-based pipelines) do this far more carefully, but the shape is this.

# Phase 1: anchor the stream at a consistent position BEFORE the snapshot.
src = connect(SOURCE_DSN)
gtid = src.execute("SELECT @@GLOBAL.gtid_executed").scalar()   # MySQL anchor
stream = ChangeStream(SOURCE_DSN, start_gtid=gtid)             # begins capturing now
stream.start()                                                 # buffers changes from `gtid`

# Phase 2: bulk-copy the consistent snapshot as of `gtid`.
with src.snapshot(gtid) as snap:                              # frozen, transactional view
    for batch in snap.scan("orders", where="shard_key BETWEEN %s AND %s",
                           args=(LO, HI), batch_size=2000):
        tgt.bulk_insert("orders", batch)                      # no secondary indexes yet
tgt.build_indexes("orders")                                   # build AFTER the bulk load

# Phase 3+4: apply the buffered + ongoing stream until lag is near zero.
for change in stream:                                          # ordered, idempotent
    tgt.apply(change)                                          # INSERT/UPDATE/DELETE replay
    if change.position % 10_000 == 0:
        lag = src.position() - stream.applied_position()
        if lag < timedelta(seconds=1):
            break                                             # caught up; proceed to verify
```

The two properties that make this correct are **ordering** and **idempotency**. Ordering: changes must be applied in the same order the source committed them, or you can apply an old value over a newer one. The change stream (binlog/WAL/CDC) is inherently ordered, which is why it is the right source of truth. Idempotency: a change might be applied twice (a retry, a reconnect), so each apply must be safe to repeat — an `UPSERT` on the primary key, a delete that no-ops if the row is gone. Get ordering and idempotency right and the target converges to the source no matter how the copy and stream interleave.

### Why the change stream, and not "just re-read the source"

A reasonable objection: why bother tailing a change stream — why not periodically re-scan the source and copy whatever changed? Because re-scanning to find changes is `O(data)` every pass, and you would never converge: by the time you finish scanning a terabyte, more has changed. The change stream is `O(changes)`, which is exactly the work you need to do and no more. This is the same reason [replication](/blog/software-development/database/database-replication-sync-async-logical-physical) uses a log rather than periodic full syncs, and it is why CDC and the outbox pattern are foundational to resharding. The binlog/WAL already exists for crash recovery and replication; a reshard is just one more consumer of it.

## The routing layer: where the cutover actually happens

**Senior rule of thumb: the cutover is a single, atomic change to the routing layer's mapping for one unit of data. If your routing layer cannot flip one entry atomically, you cannot reshard safely.**

Everything in the move protocol exists to get the target into a state where flipping traffic is safe. The flip itself happens in the *routing layer* — the component that, given a key, decides which physical node serves it. This is the request-routing problem DDIA frames as: "when a client wants to read or write a key, how does it know which node to connect to?" The answer determines how the cutover works.

![Cutover only changes which entry the router resolves a key to](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-4.webp)

The before/after above makes the central point: **cutover is a routing-table flip, not a data move.** Before the flip, the range `[c000, e000)` resolves to shard 3 (old), which serves all its reads and writes while the new shard sits as a shadow, replicating with sub-second lag. After the flip, the same range resolves to shard 3b (new), which now serves the traffic, while the old shard is kept hot with a reverse stream. The bytes did not move during the flip — they were already on the new shard, courtesy of the snapshot-plus-tail protocol. Only the *mapping* changed.

DDIA describes three architectures for request routing, and each handles cutover differently:

| Routing approach | Where the map lives | How cutover works | Example |
| --- | --- | --- | --- |
| Routing tier / proxy | A proxy that all queries pass through | Update the proxy's map; clients unaware | Vitess VTGate, Figma DBProxy |
| Coordination service | ZooKeeper/etcd holds the map, nodes/clients subscribe | Write new map to the coordination service; watchers update | HBase + ZooKeeper, Kafka |
| Cluster-aware client | Clients learn the map and redirect themselves | Nodes return redirects; clients refresh their cached map | Redis Cluster (MOVED/ASK) |

The **routing tier** is the most common at scale because it decouples clients from topology entirely. The application talks to a proxy (Vitess's VTGate, Figma's DBProxy), and the proxy holds the shard map. To cut over, you update the proxy's map — one entry, one range, one atomic change. Figma built DBProxy precisely so that the application sees a logical schema while the proxy translates logical shard IDs to physical databases; the cutover is a change in that translation, invisible to the application. The proxy also makes the brief read-only window manageable: it can hold or queue writes to the moving range for the sub-second flip, so the application sees a tiny latency blip rather than an error.

The **coordination-service** approach stores the authoritative map in a strongly consistent store (ZooKeeper, etcd) built on a [consensus](/blog/software-development/database/raft-consensus-from-scratch) protocol, and everyone subscribes to changes. The cutover is a single write to that store; the consensus layer guarantees all subscribers see the change in a consistent order. This is heavier but bulletproof: the map flip is itself a linearizable operation.

The **cluster-aware client** pushes routing into the clients, and cutover happens via *redirection*: the old node tells the client "this key now lives over there." Redis Cluster's MOVED and ASK redirects are the textbook example, and they are subtle enough to deserve their own section below.

### The atomicity problem, stated carefully

Why does the flip have to be atomic? Consider what a *non-atomic* flip looks like for a single key during a tiny window where some traffic goes to old and some to new. A write lands on old. A read for the same key, routed to new, misses that write — stale read. Worse, a write lands on new while the change stream is still draining old's recent writes into new; ordering breaks, and you get lost updates. The entire correctness of the scheme rests on this invariant: **for any given key, at any given instant, exactly one shard is authoritative, and both reads and writes for that key go to it.** The flip must move a key from "old is authoritative" to "new is authoritative" with no in-between. That is what "atomic cutover" means, and it is why systems either pause the moving unit's writes for the flip's duration (milliseconds), use a two-phase handoff, or rely on a linearizable map store.

## Keys mid-move: redirection and forwarding

**Senior rule of thumb: during the brief overlap when a unit is half-moved, the source must forward or redirect traffic for keys it no longer fully owns, so no client ever gets a wrong answer.**

There is an awkward in-between state, especially in the slot-moving and range-moving families, where a unit is *being* migrated: some keys are already on the target, some are still on the source. A naive system would either reject these keys or serve them inconsistently. Mature systems instead *forward or redirect* — they keep the client correct by sending it to wherever the key actually is right now.

Redis Cluster's protocol is the clearest illustration, so let us walk it concretely.

![While a slot migrates, the source forwards missing keys with ASK; after handoff the cluster answers MOVED](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-7.webp)

The timeline above shows a single slot (slot 8) migrating from a source node to a destination node, and how a client's request is handled at each phase.

To start a slot migration, the operator sets the slot to `MIGRATING` on the source and `IMPORTING` on the destination:

```bash
# On the destination node B: prepare to receive slot 8.
redis-cli -h B CLUSTER SETSLOT 8 IMPORTING <node-id-of-A>

# On the source node A: prepare to give away slot 8.
redis-cli -h A CLUSTER SETSLOT 8 MIGRATING <node-id-of-B>

# Move the keys, in batches, with the atomic MIGRATE command.
keys=$(redis-cli -h A CLUSTER GETKEYSINSLOT 8 100)
redis-cli -h A MIGRATE B-host B-port "" 0 5000 KEYS $keys

# Repeat GETKEYSINSLOT/MIGRATE until the slot is empty on A.
# Then hand off ownership cluster-wide.
redis-cli -h A CLUSTER SETSLOT 8 NODE <node-id-of-B>
redis-cli -h B CLUSTER SETSLOT 8 NODE <node-id-of-B>
```

While the slot is in this half-moved state, here is the redirection logic that keeps clients correct:

- A node receives a command for a key in slot 8. The slot is in `MIGRATING` state on the source.
- If the key *still exists* on the source, the source serves it normally — it has not been migrated yet.
- If the key *does not exist* on the source (because it was already moved to the destination, or it never existed), the source replies `-ASK 8 <dest>`. This means: *for this one query only*, retry against the destination.
- The client, on receiving `-ASK`, sends an `ASKING` command to the destination, then re-sends the original query. The `ASKING` command sets a one-shot flag that tells the destination "yes, I know this slot is still `IMPORTING`, serve this anyway."
- The client does *not* update its slot map on an `-ASK` — the migration is in progress, ownership has not transferred, so the next query for a different key in slot 8 might still belong on the source.

Contrast this with `-MOVED`, which fires only after the migration *completes* and ownership has been handed off via `SETSLOT ... NODE`:

- A node receives a command for a key in slot 8, which it no longer owns.
- It replies `-MOVED 8 <dest>`. This means: *the slot has permanently moved; update your map and send this and all future queries for slot 8 to the destination.*
- The client updates its cached slot-to-node map and retries against the destination.

The distinction is exactly the difference between temporary and permanent. `-ASK` is "this one key, this one time, over there, but don't change your mind about the slot." `-MOVED` is "the slot is gone, remember the new owner forever." Getting this right is what lets Redis Cluster reshard live: at no instant does a client get a wrong answer, because the source always knows whether a key is here-and-now or moved-and-forwarded.

The legacy key-by-key approach has real operational pain, which is why Redis 8.4 introduced **atomic slot migration** (`CLUSTER MIGRATION IMPORT <start-slot> <end-slot>`). The old approach migrated keys one at a time with `MIGRATE`, generated a storm of `-ASK` redirects (one source measured 241 redirects/second at peak), returned `-TRYAGAIN` for multi-key operations whose keys were split across source and destination, and could leave the cluster in an inconsistent state if a migration failed midway. Atomic slot migration borrows the snapshot-plus-replication-stream model from Redis 8.0 replication: the source forks and ships the slot's data, a parallel connection streams incremental writes, and ownership transfers in a single atomic handoff after the destination has everything. The result was roughly 30x faster (6 to 8 seconds versus 192 to 219 seconds for the same slot set), 98% fewer client redirects (`-MOVED` at the end instead of `-ASK` throughout), and — the big correctness win — no `-TRYAGAIN` errors, because all keys are available on the destination before ownership flips. This is the same architectural insight as the whole article: replace per-key choreography with snapshot-plus-stream, and make the flip atomic.

Vitess achieves the equivalent at the SQL layer through **routing rules**. During a MoveTables or Reshard workflow, VTGate's routing rules can send reads and writes for a table or shard to the source while the target catches up, then flip them — and the flip is applied as a routing-rule change in the topology, which VTGate honors immediately. The mechanism differs (SQL routing rules versus slot redirects) but the principle is identical: forward traffic to wherever the data is authoritative right now, and flip the rule atomically when the target is ready.

## Verification: proving the copy is correct before you trust it

**Senior rule of thumb: a reshard is not "copy then hope." It is "copy, prove byte-for-byte equality against a moving target, then flip." If you cannot prove equality, you do not flip.**

The scariest failure mode in resharding is not a crash — crashes are loud and you retry. The scariest failure is *silent divergence*: the target is missing a few rows, or has a stale value for some key, and you do not find out until a customer reports that their data is wrong, days after the old copy was deleted. Verification is the gate that prevents this, and it is non-negotiable.

![No traffic flips until VDiff is clean; the reverse stream keeps the old copy ready to take traffic back](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-8.webp)

The pipeline above is the gated cutover I run. Each gate must pass before the next traffic flip, and a reverse stream sits at the end to enable fast rollback. The gates are:

**Gate 1 — VDiff checksum.** Compare source and target over the migrated range, row by row or by checksum. Vitess's **VDiff** is the reference implementation: it captures a consistent snapshot of the source and the target and compares them, reporting any rows that differ, are missing, or are extra. Because the source is still moving, VDiff snapshots both sides at a comparable position and accounts for the in-flight tail. The output you want is zero mismatches. A nonzero count means your stream application has a bug (a missed change, a wrong ordering, a non-idempotent apply) — fix it and re-run before going further.

```bash
# Vitess: verify the target is an exact representation of the source
# before switching any traffic. Run against the reshard workflow.
vtctldclient VDiff --target-keyspace commerce --workflow cust2cust \
    create

# Poll until it completes, then read the report.
vtctldclient VDiff --target-keyspace commerce --workflow cust2cust \
    show last
# RowsCompared: 48,213,907   MismatchedRows: 0   ExtraRowsSource: 0
# ExtraRowsTarget: 0   -> clean, safe to SwitchTraffic
```

**Gate 2 — Switch read-only traffic first.** Even after VDiff is clean, you do not flip everything at once. You switch *read-only* traffic first — the `rdonly` and `replica` tablet types in Vitess — to the target. Reads are non-destructive: if the target has a subtle problem, a stale read is bad but recoverable, and you have not yet committed any writes to the new home. Watch error rates and latency. If reads are clean for a while, proceed.

**Gate 3 — Canary the writes.** Some teams canary a small fraction of writes to the new shard before flipping all of them, watching the error rate and the reverse-comparison closely. This is optional and depends on your routing layer's granularity, but when you can do it, it catches write-path bugs (constraint violations, trigger interactions, auto-increment collisions) under tiny blast radius.

**Gate 4 — Switch primary traffic, with reverse replication on.** Now you flip writes. In Vitess this is `SwitchTraffic` for the primary tablet type, and by default (`--enable-reverse-replication`) it *automatically creates a reverse replication stream* from the new primary back to the old shard. From this instant, every write to the new shard is also replayed onto the old shard. The old shard is no longer authoritative, but it is a perfect, live, hot standby of the new state.

**Gate 5 — Bake, then keep rollback ready.** You let the new configuration run for a soak period — 24 to 72 hours is typical — before you delete the old copy. During the bake, the reverse stream keeps the old copy current, so rollback (`ReverseTraffic` in Vitess) is a single routing flip back to the old shard, which has every write the new shard took. Only after the bake, when you are confident, do you run `Complete` to tear down the reverse stream and drop the old shard.

### Dark reads: verification under real traffic

VDiff proves equality at a snapshot. **Dark reads** prove equality under *live* traffic and are the technique Notion leaned on. The idea: while the new database is shadow-replicating, every production read query is *also* issued against the new database, and the two results are compared. Notion described a dark read as "a parallel read query issued to a follower database, whose results we can compare with those of the primary to ensure equality." They sampled queries returning up to five rows, added a one-second pause to let replication catch up, and confirmed near-100% equivalence before trusting the new fleet. The beauty of dark reads is that they exercise the *actual query patterns* your application uses, including the weird ones VDiff's full-table comparison might gloss over — the query with the unusual join, the one that depends on a specific index, the one with the off-by-one in its range predicate.

```python
# Dark-read comparison: serve from the source, compare against the target,
# log divergence. The user gets the source's answer; the target is shadowed.
def read_with_dark_compare(query, args):
    primary_result = source.execute(query, args)        # authoritative, returned to user
    try:
        time.sleep(REPLICATION_SETTLE)                  # ~1s for the tail to apply
        shadow_result = target.execute(query, args)     # the candidate, not returned
        if not rows_equal(primary_result, shadow_result):
            metrics.incr("dark_read.mismatch")
            log.warning("divergence", query=query, args=args,
                        primary=primary_result, shadow=shadow_result)
        else:
            metrics.incr("dark_read.match")
    except Exception as e:
        metrics.incr("dark_read.error")                 # never fail the real request
    return primary_result                               # user always gets the source's answer
```

The two verification techniques are complementary. VDiff is a thorough, point-in-time, full-data proof. Dark reads are a continuous, traffic-shaped, sampled proof. Run VDiff to gate the cutover; run dark reads to build confidence under real load. Neither alone is sufficient at the scale where a reshard matters.

## Avoiding hotspots: pre-splitting and salting

**Senior rule of thumb: a reshard that just splits a hot shard down the middle often recreates the hotspot on one of the halves. Spread the heat, do not merely halve it.**

Resharding is often triggered by a hotspot, and the worst outcome is to do all the work of a live move only to discover the new shard is hot too. Two techniques prevent this: pre-splitting (placing split points where the load actually is) and salting (spreading a single hot key across many shards).

![A salt prefix fans the same logical key onto N pre-created shards](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-5.webp)

The figure above contrasts the two regimes. On top, the naive case: a single hot key (`"trending"` taking 50,000 writes/second) lands entirely on shard 7, which pins at 100% CPU. No amount of resharding the *rest* of the data helps, because the hotspot is a single key, and a single key cannot be split by ordinary partitioning — it hashes to one place. On the bottom, salting: you prepend a small random salt (`0#trending` through `3#trending`) so the one logical key becomes N physical keys, which hash to N different pre-split shards, each taking a quarter of the load. Reads must then fan out to all N salted keys and merge, which is the cost you pay for the write distribution.

This is exactly DynamoDB's **write sharding** recommendation. DynamoDB's adaptive capacity will automatically isolate frequently accessed items — "split for heat," where a sustained-hot partition is split into two, with the split point chosen from recent traffic to spread the heat evenly. But adaptive capacity takes minutes to react and cannot push a single partition key past the hard per-partition limit (1,000 write capacity units per second). For a key that is genuinely hot beyond one partition's ceiling, the fix is application-level write sharding: append a suffix `1..N` to the partition key so `CandidateA` becomes `CandidateA#1` through `CandidateA#N`, multiplying throughput by N. The guidance is to shard only the hot entities — if 100 of your millions of keys are hot, salt those 100 and leave the rest alone, because salting imposes a scatter-gather read cost you do not want to pay everywhere.

**Pre-splitting** is the range-partitioning analogue. When you create a new table or new shards and you know the key distribution in advance (say, keys are UUIDs uniformly distributed, or you have a histogram of historical load), you create the split points *up front* so writes spread immediately rather than all landing on one shard until it splits. HBase operators pre-split regions at table creation for exactly this reason; a table created as a single region funnels every initial write to one region server until the auto-splitter catches up, which can be a painful "first hour" hotspot. CockroachDB exposes manual `SPLIT AT` for the same purpose. The mental model: do not make the system discover the hotspot the hard way and react; tell it where the heat will be and pre-position the shards.

```sql
-- CockroachDB: pre-split a range at known boundaries so a bulk import
-- spreads across nodes from the first write instead of hammering one range.
ALTER TABLE events SPLIT AT VALUES ('2026-01-01'), ('2026-04-01'),
                                   ('2026-07-01'), ('2026-10-01');
-- And scatter the resulting ranges across the cluster up front.
ALTER TABLE events SCATTER;
```

A subtle second-order point: salting trades write distribution for read complexity, and the trade is not always worth it. If your hot key is write-heavy and read-light (a counter, an append-only log), salting is a clear win — writes spread, and the occasional fan-out read is cheap relative to the write volume. If the key is read-heavy, the fan-out cost on every read can dwarf the write savings. Measure the read/write ratio before you salt. The same caution applies to the choice of `N`: too small and you do not relieve the hotspot; too large and every read fans out to too many shards. Size `N` to just clear the per-shard ceiling with headroom, not maximally.

## Throttling: the move is a guest, the workload is the host

**Senior rule of thumb: the backfill is a background tenant. It must yield to live traffic at every layer, or you will turn a capacity problem into an availability incident.**

Here is the failure I have personally watched happen more than once: a team kicks off a reshard backfill at full speed to "get it over with," the bulk copy saturates the source's disk I/O and connection pool, the live workload's p99 latency triples, and now you have an outage *caused by the fix for a future outage*. The backfill is competing with production for the same finite resources — CPU, disk IOPS, network, buffer cache, connection slots — and production must win every time.

![Foreground SLO sits on top; each layer below throttles the backfill so the live workload never starves](/imgs/blogs/live-resharding-and-rebalancing-without-downtime-9.webp)

The stack above is how I think about throttling: the live workload's SLO sits on top as the inviolable constraint (p99 read under 5 ms, say, must hold), and every layer below exists to protect it. There are four throttling layers, from coarse to fine:

**Admission control (coarsest).** A control loop watches the live workload's health — p99 latency, replica lag, error rate — and *pauses the backfill entirely* if any of them degrades past a threshold. This is the safety net: if anything goes wrong, the copy stops and production recovers. CockroachDB's admission control does this generically, prioritizing foreground SQL over background work like rebalancing; a reshard backfill should be subject to the same kind of governor. "Pause if replica lag exceeds 30 seconds or p99 spikes" is a typical rule.

**Rate limiting.** Cap the backfill's throughput explicitly — copy at, say, 20 MB/s, in batches of 2,000 rows, with a deliberate sleep between chunks. The sleep matters: it returns the disk and the connection to the live workload between batches rather than monopolizing them. The exact numbers depend on your headroom, but the shape is always "bounded rate, batched, with gaps."

**Resource isolation.** Run the backfill with low-priority I/O (a separate I/O class, `ionice`-style), through a *separate connection pool* so it cannot exhaust the pool the application uses, and ideally against a read replica rather than the primary so the snapshot scan does not touch the write path at all. Reading the snapshot from a replica is the single biggest isolation win available — Notion ran dark reads against followers for exactly this reason, and copying from a replica keeps the primary's resources entirely for production.

**Background-tenant scheduling (finest).** At the bottom, the bulk copy and change-stream apply run as explicitly background work, scheduled to fill spare capacity and back off under contention.

The general principle generalizes far beyond resharding: any large background operation against a live database — a backfill, a [schema migration](/blog/software-development/database/zero-downtime-schema-migrations), a `VACUUM`, an index build — must be a polite guest. The reshard backfill is just the largest and longest-running such guest you will ever host, which is why it deserves all four layers. A team that runs its backfill flat-out and "monitors closely" is one traffic spike away from an outage; a team that lets the backfill take three days but never perturbs production has done it right. Slow and invisible beats fast and visible.

> The reshard you never hear about is the successful one. If users noticed, you went too fast.

## How the systems we have covered actually do it

The protocol is universal, but each system wears it differently. Here is the same six-phase dance as implemented by the systems we have toured across this series.

### Vitess: Reshard and VReplication

Vitess is the most explicit and operator-facing implementation, which is why it is the best one to learn from. A reshard is a *workflow*:

```bash
# 1. Create the workflow: split shard 0 into shards -80 and 80-.
vtctldclient Reshard --workflow cust2cust --target-keyspace commerce \
    create --source-shards '0' --target-shards '-80,80-'

# 2. Monitor copy progress and replication lag.
vtctldclient Reshard --workflow cust2cust --target-keyspace commerce show

# 3. Verify equality (the gate).
vtctldclient VDiff --target-keyspace commerce --workflow cust2cust create
vtctldclient VDiff --target-keyspace commerce --workflow cust2cust show last

# 4. Switch read-only and replica traffic first, then primary.
vtctldclient Reshard --workflow cust2cust --target-keyspace commerce \
    switchtraffic --tablet-types "rdonly,replica"
vtctldclient Reshard --workflow cust2cust --target-keyspace commerce \
    switchtraffic --tablet-types "primary"

# 5. If something is wrong AFTER the primary switch, roll back atomically.
vtctldclient Reshard --workflow cust2cust --target-keyspace commerce reversetraffic

# 6. When confident (after the bake), finalize and drop the source shards.
vtctldclient Reshard --workflow cust2cust --target-keyspace commerce complete
```

Internally, **VReplication** is the engine: it does the consistent snapshot copy and then tails the source's binlog from the snapshot's GTID, applying changes to the target shards — exactly the snapshot-plus-tail protocol. `SwitchTraffic` flips VTGate's routing rules, and because `--enable-reverse-replication` is on by default, switching the primary automatically sets up reverse VReplication streams from the new shards back to the old, which is what makes `ReverseTraffic` a clean rollback. VDiff is the verification gate. This is the protocol of this article, productized — which is why Vitess is what Slack, YouTube, and many others run. Slack migrated 99% of its MySQL traffic to Vitess, serving 2.3 million queries per second across thousands of sharded hosts, and became one of the largest contributors to Vitess in the process precisely because operating reshards at that scale demanded it.

### CockroachDB: the allocator does it continuously

CockroachDB inverts the model: there is no manual reshard because **resharding never stops.** The keyspace is divided into ranges (32 to 64 MB each), and the `splitQueue` continuously splits ranges that exceed the size threshold or that load-based splitting flags as hot. The **allocator** continuously rebalances ranges and leaseholders across nodes to equalize load. Add a node and the allocator notices the imbalance and streams ranges to it; remove a node and it re-replicates the lost ranges elsewhere. Each range is a Raft group, so moving a replica is adding a Raft learner, catching it up via snapshot-plus-log (the same protocol again, at the Raft level — see the [Raft post](/blog/software-development/database/raft-consensus-from-scratch)), then transferring membership. The "cutover" for a range is a Raft leadership/lease transfer, which is atomic by construction. The lesson: if rebalancing is a first-class, always-on background activity governed by admission control, you never face a big-bang reshard at all. The cost is that the database carries the machinery (and complexity) to do this constantly.

### Cassandra: bootstrap and cleanup

Cassandra's reshard is *adding a node*. Because data is distributed on a [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) ring with vnodes, a new node bootstraps by claiming a set of token ranges and *streaming* exactly those ranges from the nodes that previously owned them. During bootstrap, Cassandra notifies the ring of the new node's pending ranges so that reads and writes for those ranges are routed correctly even mid-bootstrap. You add nodes one at a time (concurrent vnode bootstraps cause token collisions), monitor streaming with `nodetool netstats`, and — the step everyone forgets — run `nodetool cleanup` on the nodes that *gave up* ranges, because until cleanup runs they still hold the data they no longer own, and it counts against their load. The protocol maps cleanly: streaming is the bulk copy, the ring's eventual-consistency model and hinted handoff cover the tail, gossip propagates the new token map (the routing flip), and `cleanup` is the deferred deletion of the old copy.

```bash
# Cassandra: add one node, watch it stream, then reclaim space on donors.
# (New node has auto_bootstrap: true and joins the ring.)
nodetool netstats          # watch streaming progress on the joining node
nodetool status            # confirm the new node is UN (Up/Normal)
# On each node that LOST ranges to the newcomer, reclaim the now-foreign data:
nodetool cleanup
```

### Redis Cluster: slot migration

Covered in detail above: slots move via `CLUSTER SETSLOT MIGRATING/IMPORTING`, keys move via `MIGRATE` (legacy) or atomic slot migration (Redis 8.4), `-ASK` forwards mid-move, `-MOVED` reroutes after handoff, and `redis-cli --cluster reshard` automates the whole choreography. The fixed-slot strategy means the key-to-slot map never changes; only slot ownership moves. Redis is the cleanest example of the *fixed number of partitions* strategy from DDIA, and Redis 8.4's atomic slot migration is the cleanest example of replacing per-key choreography with snapshot-plus-stream-plus-atomic-handoff.

### DynamoDB: invisible auto-splitting

DynamoDB hides resharding entirely. Adaptive capacity isolates hot items and "split for heat" automatically splits a sustained-hot partition into two, choosing the split point from recent traffic to balance the halves — all without operator involvement and without downtime. The price of total invisibility is total inflexibility: you cannot push a single partition key past the hard per-partition limit (1,000 WCU/s), and split-for-heat takes minutes to react, so genuinely hot single keys still require application-level write sharding (salting). DynamoDB is the *managed* end of the spectrum — the move protocol runs entirely inside AWS — and the tradeoff is that when you hit the per-key ceiling, you must reach back into your application and salt, because the platform will not do it for you.

Here is the whole landscape in one table:

| System | Strategy (DDIA) | Move unit | Copy + tail | Verify | Cutover | Rollback |
| --- | --- | --- | --- | --- | --- | --- |
| Vitess | manual reshard | shard / range | VReplication (snapshot + binlog) | VDiff | routing-rule flip | ReverseTraffic |
| CockroachDB | dynamic range | range (32-64 MB) | Raft learner snapshot + log | Raft consistency | lease transfer | re-replicate |
| Cassandra | ring / vnodes | token range | streaming + hints | repair / digest | gossip token map | remove node |
| Redis Cluster | fixed slots (16384) | hash slot | MIGRATE / atomic slot migration | implicit | SETSLOT NODE | re-migrate |
| DynamoDB | dynamic (managed) | partition | internal (invisible) | internal | internal | internal |

## A prescriptive runbook: running a live reshard safely

**Senior rule of thumb: write the runbook before you touch production, dry-run it on a staging copy of real data volumes, and make every step independently reversible.**

Here is the runbook I would hand a team about to do their first live reshard. It is opinionated and conservative on purpose; you can go faster once you have done a few, but the first one should be paranoid.

**Before the day:**

1. **Pick the strategy and the move unit.** Are you splitting a hot range, moving slots, or adding ring nodes? This determines the tooling. If you are on Vitess/Cassandra/CockroachDB/Redis, use the native workflow — do not hand-roll. If you are on vanilla Postgres/MySQL (the Notion/Figma situation), build the snapshot-plus-tail pipeline on logical replication or CDC.
2. **Confirm the routing layer can flip one unit atomically.** If it cannot, fix that first. Everything depends on it. A routing tier (proxy) or a consensus-backed map is what you want; a hard-coded shard map in application config is not.
3. **Provision the target with headroom.** Do not split a hot shard into two halves on equally-sized nodes that will both be hot again in a month. Provision so the post-reshard load is comfortable (Notion landed at 20% CPU after, from 90%+ before — that is the right amount of headroom).
4. **Dry-run on staging with production-scale data.** A reshard that works on 1 GB of test data tells you nothing about a reshard of 1 TB of live data. Restore a recent production snapshot at full scale and run the entire runbook against it, timing every phase. This is where you discover the copy takes 12 hours, not 2, and that you forgot to skip index builds.
5. **Write the rollback runbook and rehearse it.** Rollback is not a paragraph at the bottom of the doc; it is a first-class procedure you have actually executed in the dry run.

**During the move (per unit):**

6. **Anchor the change stream, then snapshot.** Record the GTID/LSN/commit-log position, start the stream consumer at it, then begin the consistent snapshot copy. Skip secondary indexes; build them after the bulk load.
7. **Throttle hard.** Copy from a replica if possible. Rate-limit. Put admission control in front: pause if live p99 or replica lag degrades. Watch the live workload's dashboards more than the copy's progress bar.
8. **Drive lag to near-zero.** Let the stream catch up until lag is stable under a second.
9. **Run VDiff. Do not skip this. Do not flip on a nonzero mismatch.** If VDiff is dirty, you have a stream-apply bug; find it and re-run.
10. **Optionally run dark reads** to build confidence under real query shapes.

**The cutover (per unit):**

11. **Switch read-only/replica traffic first.** Watch errors and latency. Bake reads for a while.
12. **Switch primary traffic, reverse replication ON.** This is the atomic flip. The window is sub-second; the routing layer pauses or two-phases the moving unit's writes for the flip's duration.
13. **Verify the new primary is healthy** — write success rate, latency, the reverse stream is flowing.

**After the cutover:**

14. **Bake for 24 to 72 hours with the old copy hot.** Rollback (`ReverseTraffic` / re-flip) stays a single routing change throughout, because the reverse stream keeps the old copy current.
15. **Only then, finalize.** Tear down the reverse stream and drop the old copy (`Complete` / `nodetool cleanup` / `SETSLOT NODE` finalization). This is the one irreversible step; do it last and only when confident.

The shape of this runbook is the shape of the whole article: decouple the mapping from the node count, copy a snapshot while tailing the stream, prove equality, flip the routing layer atomically, and keep the old copy hot until you are sure. Every safe reshard, across every system, is a variation on those five moves.

## Case studies from production

Theory is clean; production is where the lessons live. Here are eight real (and realistic) reshard stories, each with the symptom, the wrong first guess, the root cause, and the fix.

### 1. Notion's "Great Re-shard": 32 to 96 shards, zero downtime

Notion's shards were running past 90% CPU and saturating provisioned disk IOPS at peak. The wrong first instinct — held by many teams — is to scale vertically onto bigger machines; Notion correctly reasoned they would hit the same wall again and chose horizontal resharding instead, tripling from 32 to 96 databases. The mechanism was textbook snapshot-plus-tail on **Postgres logical replication**: three publications per existing database covering five logical schemas apiece, with subscriptions on the new databases. The standout optimization was *skipping index creation during the initial sync*, which cut sync time from three days to twelve hours — a direct application of "bulk-load unindexed, build indexes after." Verification was **dark reads** against followers (sample queries, one-second replication pause, compare), reaching near-100% equivalence. The cutover script per database: pause traffic at PgBouncer, verify replication caught up, repoint PgBouncer to the new database URLs *and flip the replication streams so the new databases replicate back to the old* (instant reverse stream for rollback), resume traffic. Users saw about one second of a "saving" spinner. After, CPU and IOPS hovered around 20% at peak. Every move in this article appears in that one re-shard. (See [Notion's writeup](https://www.notion.com/blog/the-great-re-shard).)

### 2. Figma's nine-month road to the first sharded table

Figma had grown its Postgres stack roughly 100x and, by the end of 2022, had a dozen vertically partitioned databases with caching and read replicas — but a single logical table was still unsharded and growing. The hard part was not the data move; it was building the *routing layer* and de-risking the cutover. Figma built **DBProxy**, a service between the application and the connection pooler with a query engine that parses SQL into an AST, extracts logical shard IDs, and rewrites queries to physical databases. The key de-risking insight was **separating logical from physical sharding**: they first made data *appear* horizontally sharded at the application layer using Postgres views per shard, rolled out behind feature flags with second-level rollback, *before* any physical database changes. Only then did they physically split, and the first physical failover succeeded with "only ten seconds of partial availability on database primaries and no availability impact on replicas." They chose multiple shard keys (UserID, FileID, OrgID) grouped into colocations rather than forcing one global key, and used hash-based routing for uniformity. The lesson: the riskiest part of a reshard is the cutover, so build the ability to test and roll back the *logical* behavior long before you commit the *physical* move. (See [Figma's writeup](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/).)

### 3. The non-idempotent apply that corrupted the tail

A team built a bespoke CDC-based reshard pipeline. The bulk copy went fine, VDiff... was never run (mistake number one). The stream-apply used raw `INSERT` statements derived from the binlog rather than idempotent `UPSERT`s. On a transient reconnect, the consumer re-delivered a handful of changes, the duplicate `INSERT`s hit primary-key conflicts, the apply loop caught the exception and *skipped* those changes (mistake number two), and the target silently lost three updates. Discovered a week later when a customer's record showed a stale value; the old copy had been deleted. Root cause: non-idempotent apply plus swallowed errors plus no verification gate. Fix: every apply became an `UPSERT` keyed on the primary key, deletes became no-op-if-absent, the consumer tracked applied position durably, and VDiff became a mandatory non-skippable gate. The deeper lesson is that the two correctness properties — ordering and idempotency — are not optional niceties; skipping either one produces silent divergence, which is the worst failure class because you find out from a customer, not a dashboard.

### 4. The backfill that caused the outage it was meant to prevent

A team kicked off a reshard backfill at full throttle on a Friday to "have it done by Monday." The bulk copy ran against the *primary* (not a replica), saturated its disk IOPS, exhausted the shared connection pool, and the live workload's p99 went from 8 ms to 400 ms within minutes. Pages fired; the on-call killed the backfill; latency recovered. The wrong hypothesis was "the database is under-provisioned"; the actual cause was that the backfill was an unthrottled, un-isolated competitor for the same finite I/O and connection resources. Fix: the backfill moved to reading from a dedicated replica, got its own connection pool, was rate-limited to 20 MB/s with sleeps between batches, and sat behind an admission-control governor that paused it whenever live p99 exceeded 15 ms. The reshard then took four days instead of two and a half — and nobody noticed. Slow and invisible beats fast and visible, every time.

### 5. Redis Cluster's `-ASK` storm during a large reshard

An operator resharded a large fraction of a Redis Cluster's slots during peak traffic using the legacy key-by-key `MIGRATE` flow. Because so many slots were mid-migration simultaneously, clients hit `-ASK` redirects constantly — one measurement on a similar workload peaked at 241 redirects per second — and every redirect is an extra round trip, so client-observed latency climbed and a few multi-key operations returned `-TRYAGAIN` because their keys straddled source and destination. Nothing was *incorrect*, but the latency was ugly and the `-TRYAGAIN`s confused application code that did not retry. The wrong fix attempted was "add more nodes"; the actual fix was to reshard *fewer slots at a time* and during lower-traffic windows, and ultimately to upgrade to Redis 8.4's atomic slot migration, which replaced the `-ASK` storm with a single `-MOVED` per slot at the end (98% fewer redirects), eliminated `-TRYAGAIN` (all keys present on the destination before handoff), and ran ~30x faster. The lesson: the redirect-during-move mechanism is correct but not free, and doing too much of it at once is its own performance incident.

### 6. The split that recreated the hotspot on one half

A range-partitioned system had a hot shard, so the team split it down the middle at the midpoint of its key range and moved one half to a new node. The new node was hot within a day. Root cause: the load was not uniform across the range — 80% of the traffic targeted keys in the *upper* quarter of the range, so splitting at the midpoint put almost all the heat on the upper half, which was now alone on its node and just as saturated as before. The wrong hypothesis was "we need to split again"; the actual fix was to choose the split point *by load, not by key-space midpoint* — place the boundary where the cumulative traffic is balanced, not where the keys are balanced. This is exactly what CockroachDB's load-based splitting and DynamoDB's split-for-heat do automatically: the split point is chosen from recent traffic, not from the geometric middle. When you split manually, you must do the same analysis yourself: histogram the load, split where the load is even.

### 7. The single hot key that no reshard could fix

A leaderboard service had one entry — a viral item — taking 50,000 writes/second to a single key. The team resharded repeatedly, and every time the key landed entirely on one shard (it hashes to one place; that is the whole point of hashing), which pinned. They blamed the sharding scheme. The real problem is that *partitioning cannot split a single key* — a key is the atomic unit of placement, so no rebalancing strategy can spread one key across nodes. The fix had nothing to do with resharding: **salt the key** at the application layer. `item_42` became `item_42#0` through `item_42#15`, sixteen physical keys spreading the 50k writes across sixteen shards at ~3,000 each, with reads fanning out to all sixteen and summing. This is DynamoDB's write-sharding guidance and it applies to every system: when a *single key* is the hotspot, resharding is the wrong tool entirely; salt at the application layer instead. They salted only the handful of viral keys, leaving the millions of cold keys un-salted to avoid the fan-out read cost everywhere.

### 8. The pre-split that turned a four-hour hotspot into a non-event

A team was bulk-importing two years of historical events into a freshly created range-partitioned table. The first import attempt created the table as a single range; every one of the first hundreds of millions of writes funneled to the one range server that owned it, which pinned at 100% while the auto-splitter slowly, reactively carved the range into pieces — a four-hour hotspot at the start of every import. The fix was **pre-splitting**: before the import, they ran `ALTER TABLE events SPLIT AT VALUES (...)` at quarterly boundaries and `SCATTER` to spread the resulting empty ranges across the cluster, so the very first writes of the import already landed on different nodes. The import that previously spent four hours bottlenecked on one node now spread across the cluster from the first second. The lesson: when you know the key distribution in advance, do not make the system *discover* the hotspot reactively — tell it where the data will be and pre-position the shards. Reactive splitting is a safety net, not a substitute for pre-splitting a known distribution.

## When to reshard, and when not to

Resharding is expensive, risky, and slow even when done perfectly. Reach for it only when the alternatives are exhausted, and skip it when a cheaper tool solves the actual problem.

**Reshard when:**

- A shard is genuinely capacity-bound — CPU, IOPS, storage, or connection count near its ceiling at peak — and the data, not a single key, is the cause. This is the legitimate trigger (Notion at 90%+ CPU, Figma's unbounded table).
- You have proven the load is spread across many keys, so adding shards will actually distribute it. (If it is one hot key, salt instead — see case 7.)
- Your routing layer can flip a unit atomically and you have a snapshot-plus-tail pipeline with a verification gate and reverse replication. If you lack any of these, build them before resharding, not during.
- You have provisioned target headroom so the post-reshard state is comfortable, not just barely-survivable. Resharding into equally-hot halves buys you a month, not a year.

**Skip resharding (or defer it) when:**

- The hotspot is a single key. No partitioning scheme splits one key; salt it at the application layer instead. Resharding here is pure wasted effort (case 7).
- Single-node [partitioning](/blog/software-development/database/database-partitioning-and-sharding) would solve it. Most "my table is too big" pains are really "my queries scan too much" or "my retention deletes are too slow," both fixed by declarative partitioning on one machine without any of the distributed-systems cost.
- A read replica or a cache absorbs the load. If the shard is read-bound, not write-bound, replicas and caching are far cheaper than a reshard.
- You are on a system that reshards itself (CockroachDB, DynamoDB, Cassandra-via-add-node). Then the answer is "add a node and let the allocator/adaptive-capacity/bootstrap handle it," not "design a migration."
- You have not yet exhausted vertical scaling *and* you are confident you will not need horizontal scale soon. Vertical is a stopgap; if you will outgrow it in months, do the reshard now while there is slack, not later under duress. But if the bigger box buys you a couple of comfortable years, take it.

The meta-lesson, and the thread that runs through this entire series: resharding is the *last* tool, not the first. It is the thing you do when partitioning, replicas, caching, salting, and vertical scaling have all been correctly applied and you are *still* capacity-bound on write-heavy, well-distributed data. When you do reach for it, the move is always the same six phases — anchor and snapshot, tail, catch up, verify, atomic flip, retain — and the discipline that separates a non-event from an incident is throttling the copy, never skipping verification, and keeping the old copy hot until you are sure.

## How it all fits: the series map

This post is the capstone of a long arc on databases and distributed systems. Resharding sits at the top of the stack because it *uses* almost everything below it; here is how the pieces connect, and what to read next if a thread caught your interest.

- **Where data lives.** [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) is the prerequisite — the two cuts (vertical and horizontal), choosing a shard key, hot shards, and request routing. Resharding is what you do *after* you have sharded and the choice no longer fits.
- **How keys map to nodes.** [Consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) explains why `hash % N` is forbidden and how the ring, virtual nodes, and preference lists make ring-move resharding (Cassandra, Dynamo) a bounded `1/N` move instead of a full reshuffle.
- **How the copy stays in sync.** [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) is the machinery behind the "tail the change stream" phase — the binlog/WAL/CDC log that lets the target converge to a moving source.
- **How the bytes get there reliably.** [Database replication: sync, async, logical, physical](/blog/software-development/database/database-replication-sync-async-logical-physical) and [distributed replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) cover the snapshot-plus-log mechanism the copy phase reuses, and the reverse-stream that makes rollback safe.
- **How the routing flip stays consistent.** [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch) underpins the consensus-backed routing maps (etcd/ZooKeeper) and CockroachDB's per-range Raft groups, where the cutover is a linearizable lease transfer.
- **The system that productizes it all.** Vitess — VTGate routing, VReplication copy-and-tail, VDiff verification, SwitchTraffic/ReverseTraffic cutover and rollback — is the reference implementation of this article's entire protocol, and the natural next deep-dive: [Vitess: sharding MySQL at scale](/blog/software-development/database/vitess-sharding-mysql-at-scale).

If you read this series in order, the punchline is this: a distributed database is a set of decisions about *where data lives* and *how to move it when that answer changes*. Sharding makes the first decision; resharding executes the second, live, without anyone noticing. Do it slowly, prove it correct, and keep a way back — and the hardest operational task in the business becomes a non-event.

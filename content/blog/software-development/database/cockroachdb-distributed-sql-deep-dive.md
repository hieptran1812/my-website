---
title: "CockroachDB: How Distributed SQL Achieves Serializable Transactions"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-the-architecture-up tour of CockroachDB — the layered SQL-to-Pebble stack, the range as the unit of replication, MVCC plus Hybrid Logical Clocks instead of TrueTime, the uncertainty-interval read restart, write intents and the transaction record as the atomic commit point, the timestamp cache and read refreshing that buy serializable isolation without locks, parallel commits, leaseholders, the allocator, and how it compares to Spanner, TiDB, and Vitess."
tags:
  [
    "cockroachdb",
    "distributed-sql",
    "serializable",
    "raft",
    "hybrid-logical-clocks",
    "mvcc",
    "distributed-transactions",
    "distributed-systems",
    "newsql",
    "databases",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 60
image: "/imgs/blogs/cockroachdb-distributed-sql-deep-dive-1.webp"
---

Google Spanner proved you could have a relational database that spans continents, survives whole-datacenter failures, and still hands every transaction a position in a single global order — strong, serializable SQL at planetary scale. Then it told you the price: atomic clocks and GPS receivers bolted into every datacenter, feeding a service called TrueTime that bounds clock uncertainty to single-digit milliseconds. For Google that is a rounding error. For everyone else it is a non-starter — you cannot rent a GPS-disciplined oscillator from your cloud provider's instance menu. The obvious question, the one CockroachDB exists to answer, is whether you can get *Spanner-class* distributed SQL — serializable transactions, automatic sharding, survive-anything replication — on the commodity hardware you already have, with nothing more exotic than NTP keeping your clocks loosely in sync.

CockroachDB's answer is yes, with one honest caveat, and the entire design is the elaboration of that answer. It replaces TrueTime with Hybrid Logical Clocks (HLC) and a static *maximum clock offset* — 500 ms by default — and instead of Spanner's commit-wait it uses an *uncertainty-interval restart*: a transaction that reads a value too close to its own timestamp to order confidently simply bumps its timestamp and tries again. It keeps the rest of Spanner's playbook — a monolithic sorted key-value space, sliced into ranges, each range its own Raft group — and layers a serializable SQL engine on top using write intents, a per-transaction record that is the single atomic commit point, a timestamp cache, and a read-refresh check that validates a transaction's reads were not invalidated when its timestamp moved. The caveat is that without TrueTime's precision, CockroachDB gives up *strict* serializability across disjoint keys (Spanner's "external consistency") for plain serializability plus single-key linearizability — a gap Jepsen documented and CockroachDB owns openly.

This article is a tour of that machine from the SQL parser down to the bytes on disk. We start with the layered architecture, because every later mechanism lives at a specific layer and only makes sense once you know which contract it is implementing. The diagram below is the mental model the rest of the article unpacks: a SQL query enters the top, and each layer translates it into a narrower problem for the layer beneath — parse and plan becomes key-value operations, key-value operations become atomic multi-key transactions, transactions become reads and writes against ranges, ranges become Raft proposals, and Raft proposals become durable LSM writes in Pebble. Strip away the SQL veneer and CockroachDB is a transactional, MVCC, Raft-replicated, range-sharded key-value store — and that store is where the interesting distributed-systems engineering lives.

![Five stacked layers from SQL down to Pebble, each handing a narrower contract to the layer below, with only the bottom Pebble layer touching the disk](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-1.webp)

> Distributed SQL is not a database with a network bolted on. It is a distributed consensus system with a SQL parser bolted on. The hard part was never the SQL — it was making a few thousand independent Raft groups behave like one serializable database.

This builds directly on machinery covered elsewhere on this blog. If the words "Raft", "leader", and "quorum" do not yet feel concrete, read [Raft from scratch](/blog/software-development/database/raft-consensus-from-scratch) first — CockroachDB runs one Raft group per range, and everything below assumes you know how a single Raft group commits an entry. For the system CockroachDB is consciously imitating-but-cheaper, see [Spanner and TrueTime](/blog/software-development/database/spanner-truetime-and-external-consistency). For the transaction model CockroachDB's intents descend from, see [Percolator](/blog/software-development/database/percolator-distributed-snapshot-isolation). And for the clock theory underneath all of it — happens-before, logical clocks, why wall clocks lie — see [time, clocks, and ordering](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems).

## Why distributed SQL is different from what most engineers assume

Most engineers' mental model of "a SQL database that scales" is a primary with read replicas, or a sharded fleet of MySQL where the application picks the shard. Both of those are real and useful, and neither is what CockroachDB is. The gap between the comfortable model and the distributed reality is the reason the system has the shape it has.

| Assumption | The comfortable mental model | The distributed-SQL reality |
| --- | --- | --- |
| "Scaling SQL means read replicas." | One writable primary; replicas serve stale reads. | Every node is writable for the ranges it leads; there is no single primary, and reads are fresh. |
| "Sharding is the app's job." | The application hashes a key and picks a shard. | The key-space is one sorted map; the database splits and moves ranges automatically, transparent to SQL. |
| "A committed write is on disk on one machine." | `fsync` returns, you're durable. | A write is committed only when a *majority* of its range's replicas have it in their Raft log. |
| "Clocks are accurate enough to order events." | `now()` is the truth. | Wall clocks disagree by up to the max offset; ordering events by raw wall-clock time is a data-corruption bug. |
| "Serializable means locks everywhere." | Two-phase locking, readers block writers. | Serializable here is *lockless and MVCC*: reads never block on writes; conflicts are resolved by moving timestamps and refreshing reads. |
| "A cross-shard transaction is a 2PC nightmare." | Coordinator, prepare, commit, the blocking problem. | 2PC is real but runs *over Raft groups*, the commit point is one replicated record, and parallel commits cut its latency in half. |
| "If a clock drifts, you get subtle corruption." | Drift silently produces wrong answers. | A node whose clock drifts past the offset *kills itself* rather than serve a possibly-inconsistent read. |

Every row in the right column is a deliberate design decision, and most of them are forced moves once you accept the goal: serializable SQL on commodity hardware that survives node and datacenter loss. The discipline of the rest of this article is to introduce each mechanism alongside the failure or the assumption it overturns — never a mechanism for its own sake.

## 1. The layered architecture: five contracts stacked

> **Senior rule of thumb:** when you debug CockroachDB, the first question is always "which layer is this?" A slow query is a SQL-layer plan problem; a `40001` is a transaction-layer conflict; a range that won't take writes is a replication-layer quorum problem; a write-amplification spike is a Pebble compaction problem. The layers are the coordinate system for everything that goes wrong.

CockroachDB is structured as five layers, each of which interacts only with the layers directly above and below it, treating them as relatively opaque services. From the top:

- **SQL layer** — translates client SQL into key-value operations. It parses, plans, optimizes, and executes; it owns the schema, types, indexes, and the distributed SQL execution engine (DistSQL) that pushes computation to where the data lives. Below this layer, nothing knows what a `JOIN` is.
- **Transactional layer** — allows atomic, isolated changes to multiple KV entries. This is where serializability lives: write intents, the transaction record, the timestamp cache, conflict resolution, and read refreshing. Below this layer, there are no transactions, only individual reads and writes at specific timestamps.
- **Distribution layer** — presents the cluster's many replicated ranges as a single monolithic sorted key-value map, and routes a key to the range that holds it. Below this layer, there is no "single map", only individual ranges.
- **Replication layer** — consistently and synchronously replicates each range across nodes using Raft. A write is durable when a majority of the range's replicas have it. Below this layer, there is no replication, only a local key-value store.
- **Storage layer (Pebble)** — reads and writes KV pairs on local disk, organized as a log-structured merge-tree. This is the only layer that touches the disk.

The power of the decomposition is that each layer's contract is narrow enough to reason about and reuse. The SQL layer never has to think about Raft; it emits KV operations and trusts the layers below to make them atomic and durable. The replication layer never has to think about SQL; it replicates opaque byte ranges. This is the same "replicate the log, not the data" discipline that makes Raft tractable, applied recursively up the stack.

A useful way to internalize the stack is to follow a single statement down it. Take `UPDATE accounts SET balance = balance - 100 WHERE id = 42`:

1. **SQL layer** parses and plans the statement, resolves the table and primary-key encoding, and turns the row update into a KV read of the key for row 42 followed by a KV write of the new value. It opens (or joins) a transaction.
2. **Transactional layer** stamps the transaction with an HLC timestamp, writes a *write intent* for the new value (a provisional value plus a pointer to the transaction record), and records the read in the timestamp cache. It will not make the write visible until the transaction commits.
3. **Distribution layer** looks up which range owns the key for row 42 — say range 7 — by walking the meta ranges, and routes the operation to range 7's leaseholder.
4. **Replication layer** on the leaseholder proposes the write to range 7's Raft group; once a majority of replicas have appended it to their logs, it is committed.
5. **Storage layer** on each replica writes the KV pair into Pebble's write-ahead log and memtable, eventually flushing it to an SSTable on disk.

Every box in the figure above is one of those layers, and every later section of this article zooms into one of them. The rest of the architecture — ranges, HLC, intents, leaseholders, the allocator — are all mechanisms *within* these five contracts.

### Second-order consequence: the SQL layer is the hard part to make fast, not correct

A subtlety that surprises people coming from single-node databases: in CockroachDB the *correctness* of a query is the easy part — the lower layers guarantee it — but the *performance* is dominated by how many ranges (and therefore how many leaseholders, on how many nodes) a query touches. A point lookup on a primary key hits one range and is fast anywhere. A `SELECT ... WHERE non_indexed_column = ?` becomes a full scan fanned across every range of the table, each on a possibly-remote leaseholder, and the latency is the slowest leaseholder's round trip. The DistSQL engine mitigates this by shipping filters and aggregations to the leaseholders so only results travel back, but the first-order rule stands: in distributed SQL, the cost of a query is the geography of the ranges it touches. We will return to this when we discuss leaseholders and locality.

## 2. The range: one sorted map sliced into Raft groups

> **Senior rule of thumb:** stop thinking in tables and start thinking in *one giant sorted key-value map*. Tables, indexes, and rows are all just encodings into that one map. The unit the database actually replicates, moves, splits, and reasons about is not the table — it is the *range*, a contiguous slice of that map.

CockroachDB stores everything — every table, every secondary index, every system metadata entry — in a single, cluster-wide, sorted map of byte keys to byte values. A row in table `accounts` with primary key `42` is encoded into a key like `/Table/53/1/42` mapping to the encoded column values; a secondary index entry is another key in the same map. Sorting is total and lexicographic, which is why range scans (`WHERE id BETWEEN 100 AND 200`) are cheap: the rows are physically adjacent in key order.

That single map is far too big to live on one machine, so it is cut into contiguous chunks called **ranges**. The default target size is 512 MB. Each range owns a half-open key span — `[startKey, endKey)` — and the ranges tile the whole key-space with no gaps and no overlaps: range 1 might own `[/Min, h)`, range 2 `[h, p)`, range 3 `[p, /Max)`, and so on for thousands of ranges in a large cluster. Because the ranges are contiguous and sorted, finding the range for a key is a binary search over range boundaries, not a hash.

Here is the part that makes CockroachDB a distributed-systems system rather than a sharding library: **each range is its own independent Raft group**. A range is replicated — three replicas by default — onto three different nodes, and those three replicas run a Raft consensus group amongst themselves to agree on the ordered log of writes to that range. The figure below is the mental model:

![One sorted key-space sliced into three contiguous ranges, each range replicated three ways onto different nodes as an independent Raft group with its own leader](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-2.webp)

Three consequences fall directly out of this design, and they are the whole reason for it.

First, **there is no single primary**. Leadership of the thousands of Raft groups is spread across all the nodes. Node 1 might lead ranges 1, 14, and 200; node 2 might lead ranges 2, 15, and 201. Every node is therefore writable for *some* part of the key-space at all times — CockroachDB calls this "multi-active availability", in contrast to the active-passive model of a primary with read replicas. Lose a node and you lose leadership of the ranges it led, but those ranges' surviving replicas elect new leaders within a couple of seconds and writes resume; you never lose write availability for the whole database.

Second, **fault tolerance is per-range and quantified**. A range with three replicas survives one node failure (two of three remain, a majority). A range with five replicas survives two. The blast radius of a node failure is bounded: only the ranges that had a replica on the dead node are affected, and each of those still has a majority elsewhere.

Third, **the system can split, merge, and move ranges without the application noticing**, because the unit of all that machinery is the range, and SQL only ever sees the single logical map. A range that grows past 512 MB splits in two; two small adjacent ranges merge; a range whose replicas are unbalanced gets a replica moved. We cover that machinery in the allocator section.

### Finding a key: the meta ranges

The distribution layer has to answer "which range, and which nodes, own this key?" for every operation, and it cannot keep that mapping in one place without recreating the single-point-of-failure it was designed to avoid. The answer is a two-level index *stored in the same key-space*. A small set of `meta` ranges map key spans to range descriptors (which range owns the span, and which nodes hold its replicas). The top-level `meta1` range points to `meta2` ranges, which point to the actual data ranges. A node caches these descriptors aggressively, so the common case is a local lookup; a stale cache (the range moved) produces a retry that refreshes the descriptor. This is the same recursive trick as everything else: the routing table is itself just keys in the sorted map, replicated by Raft like any other range.

```sql
-- Ranges are real, inspectable objects. Show how a table is sliced and where
-- each slice's leaseholder lives. (Syntax is for v23.1+; older versions differ.)
SELECT
  range_id,
  start_pretty,
  end_pretty,
  range_size / 1e6 AS size_mb,
  lease_holder,
  replicas
FROM [SHOW RANGES FROM TABLE accounts WITH DETAILS]
ORDER BY start_key;

-- Example output:
--  range_id | start_pretty   | end_pretty     | size_mb | lease_holder | replicas
-- ----------+----------------+----------------+---------+--------------+-----------
--        42 | /Table/53/1    | /Table/53/1/5k | 487.2   |            1 | {1,2,3}
--        43 | /Table/53/1/5k | /Table/53/1/9k | 511.8   |            3 | {1,3,4}
--        44 | /Table/53/1/9k | /Table/53/2    | 122.4   |            2 | {2,4,5}
```

Notice range 43 is at 511.8 MB — one more burst of inserts and the allocator will split it. Notice too that leaseholders (1, 3, 2) and replica sets differ per range: leadership is genuinely spread. This output is the ground truth for diagnosing hotspots ("why is node 3 hot? because it leaseholds three 500 MB ranges of the same table") and for verifying that geo-partitioning actually placed the right ranges in the right region.

### Second-order consequence: the empty-table and sequential-key hotspot

The range model has a famous failure mode. A brand-new table is *one* range with *one* leaseholder. If your primary key is a monotonically increasing sequence or a timestamp, every insert lands at the high end of the key-space — in the same range, on the same leaseholder, on the same node. You have built a single-node database with extra latency. This is the distributed-SQL version of the b-tree right-edge hotspot, and the fixes are the same shape: hash-shard the index (`CREATE INDEX ... USING HASH`) so inserts spread across many ranges, or use a random UUID primary key so writes are uniformly distributed. The general lesson — sequential keys are an anti-pattern for any range-partitioned store — is the same one explored in [why random UUIDs hurt b-trees](/blog/software-development/database/random-uuids-are-killing-your-database-performance), inverted: here you *want* the spread that hurts a single-node b-tree, because the spread is across machines.

## 3. MVCC and Hybrid Logical Clocks: ordering without atomic clocks

> **Senior rule of thumb:** in a distributed database, "what time is it?" is a trick question. The only times that matter are the ones the system *assigns and agrees on*, never the raw wall clock. CockroachDB's whole consistency story is a discipline for assigning timestamps that respect causality even though no two machines' clocks agree.

Everything CockroachDB stores is versioned. The storage layer keys are not just `key → value` but `key + timestamp → value` — multi-version concurrency control (MVCC). A write of `balance = 400` at timestamp 105 does not overwrite the old `balance = 500` at timestamp 90; it adds a new version. A read at timestamp 100 sees the value as of 100 (so it reads 500, the latest version at or before 100); a read at timestamp 110 sees 400. This is the foundation that lets reads never block writes: a reader just picks the right historical version, and a concurrent writer is adding a *newer* version it won't see. It is also what powers time-travel queries (`SELECT ... AS OF SYSTEM TIME '-10s'`), which simply read at an older timestamp.

The question MVCC forces is: *where do the timestamps come from?* If they came from each node's raw wall clock, two nodes could assign the same timestamp to causally-ordered events, or assign an earlier timestamp to a later event, and the version ordering would be a lie. Spanner solves this with TrueTime: atomic clocks bound the uncertainty to a few milliseconds, and the database waits out that uncertainty. CockroachDB has no atomic clocks, so it uses **Hybrid Logical Clocks**.

An HLC timestamp is a pair: a physical component (the node's wall clock, in nanoseconds) and a logical component (a counter, to break ties and preserve causality when the physical clocks are equal or slightly out of order). The rules are simple and are the same Lamport-clock logic generalized to wall time:

- On a local event, set the physical component to `max(local_wall_clock, current_hlc_physical)` and bump the logical counter if the physical part didn't advance.
- On receiving a message carrying a remote HLC timestamp `T_remote`, set the local HLC to `max(local_wall_clock, current_hlc, T_remote)` — i.e., the clock never goes backwards and always advances past any timestamp it has *observed*.

The key property: HLC timestamps respect *happens-before*. If event A causally precedes event B (A sent a message B received, directly or transitively), then `HLC(A) < HLC(B)`, guaranteed, with no dependence on the clocks being accurate — only on the messages carrying timestamps. HLC stays close to wall-clock time (so timestamps are human-meaningful and `AS OF SYSTEM TIME '-10s'` means roughly ten real seconds ago) while inheriting Lamport clocks' causal-ordering guarantee. This is the construction worked through in detail in [time, clocks, and ordering](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems); here we only need its guarantee.

But HLC alone does *not* solve the problem TrueTime solves. HLC orders events that are *causally related* (connected by messages). It says nothing about two events on different nodes that never communicated. If transaction A commits a write on node 1 at wall-clock 100, and transaction B starts on node 2 whose clock reads 99 (it's 1 ms slow) and reads the same key, B might assign itself timestamp 99 and *not see A's write*, even though A finished first in real time. That is exactly the kind of stale read that breaks consistency, and HLC cannot prevent it because A and B never exchanged a message. The mechanism that closes this gap — the uncertainty interval — is the heart of CockroachDB's clock story and the subject of the next section.

### A concrete HLC walkthrough

| Step | Node 1 wall | Node 1 HLC | Node 2 wall | Node 2 HLC | Event |
| --- | --- | --- | --- | --- | --- |
| 1 | 100.0 | (100.0, 0) | 99.7 | (99.7, 0) | both idle; node 2 is 0.3 behind |
| 2 | 100.1 | (100.1, 0) | 99.8 | (99.8, 0) | node 1 commits write, stamps (100.1, 0) |
| 3 | 100.2 | (100.1, 1) | 99.9 | (99.9, 0) | node 1 logical bump (same ms) |
| 4 | 100.3 | (100.3, 0) | 100.0 | **(100.1, 1)** | node 2 receives node 1's msg with (100.1,0), jumps its HLC forward past it |

After step 4, node 2's HLC has been *dragged forward* to be later than anything it observed from node 1, so any timestamp it now assigns is causally after node 1's write — even though node 2's wall clock is still behind. That forward-drag is the entire trick. It costs nothing (no waiting, no atomic clocks) and it guarantees causal order for communicating transactions. The only thing it can't fix is *non*-communicating transactions on skewed clocks, which is why we need one more mechanism.

## 4. The uncertainty interval: the HLC analogue of commit-wait

> **Senior rule of thumb:** the maximum clock offset is not a tuning knob you set and forget — it is a *safety contract*. Every node promises its clock is within `max_offset` of every other node's; in exchange, the database promises serializability. Break the promise (let a clock drift past the offset) and the node must remove itself, because the alternative is silently lying to clients.

The stale-read problem from the last section — a transaction on a slightly-slow clock failing to see an earlier-in-real-time write — is the one Spanner solves with commit-wait. Spanner, before reporting a transaction committed, *waits out* the TrueTime uncertainty (a few milliseconds) so that any transaction starting afterward is guaranteed a later timestamp. The wait is the price of TrueTime's tight bound; with a 7 ms uncertainty, you pay ~7 ms of latency per commit.

CockroachDB cannot afford a 500 ms commit-wait (its uncertainty bound is the 500 ms max offset, not 7 ms — commit-waiting that on every transaction would be catastrophic). So it inverts the trade: instead of *every writer waiting* to remove uncertainty up front, it makes the *occasional reader restart* when it encounters uncertainty. The mechanism is the **uncertainty interval**.

When a transaction begins, it gets a provisional timestamp `T` from its gateway node's HLC. Its uncertainty interval is `[T, T + max_offset]` — the window of timestamps that, *because of possible clock skew*, might actually be in this transaction's past even though they look like its future. Now the read rule:

- If a read finds a value with timestamp **≤ T**, it's unambiguously in the past — read it.
- If a read finds a value with timestamp **> T + max_offset**, it's unambiguously in the future — ignore it (read the older version).
- If a read finds a value with timestamp **in `(T, T + max_offset]`** — inside the uncertainty interval — the database *cannot tell* whether that value was actually committed before this transaction began (clock skew could mean its "future" timestamp is really past). To stay safe, the transaction **restarts at a higher timestamp**, pushing its read timestamp just above the uncertain value so that, on retry, the value is unambiguously in the past and gets read.

The figure below traces it: a transaction reads at `T`, finds a value stamped `T + 120 ms` (inside the 500 ms window, ambiguous), restarts with its read timestamp bumped to `T + 120 ms`, and now the value is clearly in the past, so it reads it and commits.

![A timeline showing a transaction reading at timestamp T, finding a value inside the uncertainty window, performing an uncertainty restart that bumps the read timestamp above the uncertain value, then committing](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-3.webp)

This is the precise dual of commit-wait. Spanner pays uncertainty *up front, on every write, as latency*. CockroachDB pays it *lazily, only on the reads that actually hit the window, as an occasional restart*. The crucial detail that makes the restart bounded rather than a loop: the uncertainty interval's *upper bound stays fixed at the original `T + max_offset`*. So after the restart bumps the read timestamp to `T + 120 ms`, the remaining window is only `(T + 120, T + 500]` — it has shrunk. A transaction can restart at most a finite number of times before the window closes, and in practice the maximum-timestamp ratchet means you almost never see more than one uncertainty restart for a given key. The cost shows up only when clocks are genuinely skewed and reads happen to land in the window; tighten your clock sync (smaller real offset) or lower `max_offset` (smaller declared window) and the restarts vanish.

### What happens if a clock drifts past the offset

The entire scheme rests on the promise that no clock is more than `max_offset` from any other. If a node's clock actually drifts beyond that, the uncertainty interval is *too small* — a value that should be flagged ambiguous slips past as "unambiguously past or future", and you get a stale read: a serializability violation. CockroachDB refuses to let this happen silently. Every node continuously measures its clock offset against the other nodes via the regular intra-cluster RPCs. If a node detects that its clock has drifted more than ~80% of the max offset (e.g., 400 ms with the default 500 ms) relative to at least half of the other nodes, it **commits suicide**: it crashes itself out of the cluster. A missing node is a problem the rest of the cluster handles routinely (its ranges re-elect leaders, its replicas get re-created elsewhere). A node serving stale reads is a problem nothing can detect downstream. So CockroachDB chooses the loud, recoverable failure over the silent, corrupting one. This is a recurring theme — and it is why running CockroachDB with sloppy NTP is dangerous: the database is correct, but it will start euthanizing nodes if their clocks wander.

```bash
# Clock health is a first-class operational concern. The cluster exposes the
# measured offset; watch it the way you'd watch replication lag elsewhere.
# A node approaching 80% of --max-offset is about to self-terminate.

# Set a tighter offset only if your clock sync genuinely supports it. Smaller
# offset = smaller uncertainty window = fewer read restarts, but less slack
# before a node decides it has drifted and kills itself.
cockroach start \
  --max-offset=250ms \           # default is 500ms; lower needs good NTP/PTP
  --store=/mnt/data \
  --join=node1,node2,node3

-- Inspect the live clock offset from SQL / the metric endpoint:
--   clock-offset.meannanos   (mean measured offset, ns)
--   clock-offset.stddevnanos (stddev)
-- Alert when meannanos approaches 0.8 * max_offset.
```

### The honest caveat: serializable, not strict-serializable

This is the one place CockroachDB is weaker than Spanner, and it is worth stating precisely because the difference is subtle and the marketing of both products elides it. Spanner provides *external consistency* (a.k.a. strict serializability): if transaction A finishes in real time before B begins, A is ordered before B in the serial order, *for all transactions, even on disjoint keys*. CockroachDB provides *serializability* (there exists some serial order consistent with all transactions) plus *single-key linearizability* (operations on a single key respect real-time order), but **not** strict serializability across disjoint keys. The gap, which Jepsen's analysis named explicitly, is *causal reverse*: two transactions touching different keys can appear in the serial order in the opposite of their real-time order, because without TrueTime the database can't always pin down real-time order for non-communicating transactions on skewed clocks. CockroachDB mitigates this for *causally related* operations by passing "causality tokens" (the maximum HLC timestamp a client has observed) between dependent statements, so a client that reads A then writes B forces B after A. But two genuinely independent transactions on disjoint keys can be reordered. For the overwhelming majority of applications this is invisible and irrelevant; if you are building something whose correctness depends on real-time ordering of unrelated transactions, you want TrueTime, and you should read the [Spanner deep-dive](/blog/software-development/database/spanner-truetime-and-external-consistency) to understand what you'd be paying for it. Kleppmann's *Designing Data-Intensive Applications* (Chapter 9) draws exactly this line — serializability is about *a* valid order; linearizability/strict-serializability adds the constraint that the order matches real time — and CockroachDB sits deliberately one notch below the strongest model to escape the atomic-clock requirement.

## 5. Write intents and the transaction record: the atomic commit point

> **Senior rule of thumb:** a distributed transaction commits the instant *one* key flips, not when all its writes land. Find that one key — the transaction record — and you understand the whole commit protocol. Everything else is bookkeeping that points at it.

A transaction in CockroachDB may write keys spread across many ranges on many nodes. The classic distributed-systems hazard is partial commit: some writes land, some don't, and the transaction is neither cleanly committed nor cleanly aborted. CockroachDB's solution is the same conceptual move as Google Percolator's (covered in the [Percolator deep-dive](/blog/software-development/database/percolator-distributed-snapshot-isolation)): designate *one* piece of data as the single source of truth for the transaction's outcome, and make every other write defer to it. That one piece is the **transaction record**.

Here is the protocol. While a transaction runs, every value it writes is written as a **write intent** rather than a committed value. A write intent is "a combination of a replicated value and an exclusive lock" — it is a normal MVCC key-value pair (replicated by Raft into its range like any write), but tagged with metadata marking it provisional and carrying a pointer to the transaction's record. The transaction record itself is a key in the range of the transaction's *first* write, holding a status field:

- **PENDING** — transaction in progress.
- **STAGING** — the parallel-commits state (next section).
- **COMMITTED** — transaction done; its intents are now real values.
- **ABORTED** — transaction failed; its intents are garbage to be cleaned up.

The transaction commits at the instant its record's status is set to COMMITTED — a single Raft write to a single range. Before that flip, all the intents are provisional; after it, they are all (logically) committed, even though they are still physically marked as intents. The figure below is the model: the coordinator scatters intents across ranges A, C, and F and creates a PENDING record; the commit is the single flip of that record to COMMITTED, which atomically decides the fate of every intent; afterward, an asynchronous cleanup pass *resolves* each intent into a plain MVCC value.

![A graph showing a transaction coordinator creating a PENDING record and scattering write intents across three ranges, all converging on a single record flip to COMMITTED, followed by asynchronous intent resolution](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-4.webp)

The elegance is that the commit is atomic *despite* spanning many ranges, because it is physically a single-key, single-range, Raft-replicated write. There is no moment where half the transaction is committed and half isn't; the record's status is the indivisible switch.

### Resolving intents and reading through them

What happens when *another* transaction's read encounters one of these intents before it's been resolved? It cannot just read the provisional value (the writer might still abort) and it cannot just ignore it (the writer might have committed). So it follows the intent's pointer to the transaction record and reads the status:

- Record says COMMITTED → the intent is a real value; read it (and helpfully resolve it to a plain value while you're there).
- Record says ABORTED → the intent is garbage; ignore it and read the previous version.
- Record says PENDING → the writer is still in flight; this is a conflict, and the reader must wait or push (next section).
- Record *doesn't exist* → the writer may have died before creating it. If the intent has no record and no heartbeat within the transaction-liveness threshold, the encountering transaction is allowed to treat it as ABORTED and clean it up.

This "read the value, follow the pointer, check the status" dance is exactly Percolator's primary-lock mechanism, with one crucial difference: where Percolator relies on a single replicated KV store (Bigtable), CockroachDB's record and intents are each individually replicated by their own range's Raft group, so the commit point is itself fault-tolerant. The transaction record is the Percolator primary, Raft-replicated.

```sql
-- A serializable transaction is just BEGIN/COMMIT. The intents and record are
-- invisible to you — until two transactions conflict, at which point one gets
-- pushed or restarted and you see a 40001 retry error (next section).
BEGIN;
  -- These two UPDATEs may touch different ranges. Each writes a write intent.
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;   -- intent on range A
  UPDATE accounts SET balance = balance + 100 WHERE id = 9001;-- intent on range F
  -- The transaction record (created on range A, the first write) is PENDING.
COMMIT;
  -- COMMIT flips the record to COMMITTED in one Raft write -> both intents are
  -- now logically committed atomically. Resolution to plain MVCC values is async.
```

### Second-order consequence: intent buildup and the contention footprint

Intents are cheap to create and cheap to resolve, but they are not free to *leave lying around*. An unresolved intent is an exclusive lock — any other transaction touching that key blocks on it. If a long-running transaction holds intents on hot keys, it serializes every other transaction on those keys behind it. Worse, if a coordinator dies, its intents linger until another transaction trips over them and runs the cleanup. The operational symptoms are contention footprints: rising `SQL transaction abort` counts, growing `intentcount` on hot ranges, and tail-latency spikes on the keys a fat transaction touches. The fix is the same as in any MVCC system — keep write transactions short, don't hold a transaction open across a network call or a user's think-time, and break giant batch updates into chunks. CockroachDB's "contention" diagnostics (`crdb_internal.transaction_contention_events`) exist precisely to find the transaction whose intents everyone else is queued behind.

## 6. Parallel commits: collapsing two consensus rounds into one

> **Senior rule of thumb:** in a geo-distributed database, latency is round trips, and round trips are the speed of light. Every optimization that matters is the same shape: do something in *one* WAN round trip that naively takes *two*. Parallel commits is that optimization for the commit path, and it is the single biggest reason CockroachDB's cross-range writes are usable across regions.

The commit protocol as described so far has a latency problem. Committing requires two things to be durable: (1) all the write intents must be replicated (one round of Raft consensus, on each range), and (2) the transaction record must be flipped to COMMITTED (a second round of Raft consensus, on the record's range). Naively these are *sequential* — you can't commit the record until you know the intents succeeded, because the record's commit is the promise that they did. Two sequential consensus rounds across a WAN is roughly `2 × RTT`, and in a three-region cluster RTT can be 60–80 ms, so commits cost 120–160 ms. That doubling is brutal; before this optimization, simply adding the first secondary index to a table (turning single-range inserts into cross-range transactions) doubled insert latency.

**Parallel commits** removes the sequential dependency with one new idea: a third transaction-record state, **STAGING**, that records not "I am committed" but "I *will* be committed if a specific set of writes all succeed." Instead of waiting for the intents and *then* writing the record, the coordinator writes a STAGING record listing all the in-flight write keys (`InFlightWrites: [A, C, F]`) *in parallel with* replicating the intents themselves. The two rounds of consensus now happen concurrently, not sequentially.

The transaction is defined as **implicitly committed** the moment its record is STAGING *and* every write in its `InFlightWrites` list has achieved consensus. The coordinator can return success to the client as soon as it observes both conditions — which is `1 × RTT`, because the intent replication and the staging-record write overlapped. The figure below contrasts the two:

![A before-after diagram contrasting classic two-phase commit at two times RTT with parallel commits at one times RTT, where the staging record write overlaps the intent writes](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-8.webp)

The asynchronous follow-up still flips the record from STAGING to COMMITTED (an "explicit" commit) so the steady-state representation is clean, but that flip is *off the critical path* — the client already got its answer. The measured effect is exactly the theory: without parallel commits, transaction latency grows at twice the rate of RTT as you spread a cluster across regions; with it, latency grows at the same rate as RTT — a 50% improvement on the commit path for geo-distributed transactions.

### How a reader recovers an implicitly-committed transaction

The subtle correctness question: if the coordinator dies after telling the client "committed" but before flipping STAGING to COMMITTED, how does anyone else know the transaction is committed? The information is fully recoverable from the durable state, because "implicitly committed" is a *checkable* predicate. Any transaction that encounters a STAGING record runs the **transaction status recovery procedure**: it checks, for each key in `InFlightWrites`, whether that write achieved consensus.

- If *all* in-flight writes are present → the transaction satisfied its implicit-commit condition; it *is* committed. Flip the record to COMMITTED and proceed.
- If *any* in-flight write is missing (and can be prevented from ever succeeding, by writing a higher-timestamp value to block it) → the transaction can never satisfy the condition; abort it.

This makes the commit decision a deterministic function of durable state, not of a living coordinator — which is what makes it safe to acknowledge the client before the explicit commit. In practice the recovery procedure almost never runs: coordinators heartbeat their records and asynchronously perform the explicit commit, so recovery only fires when a coordinator actually dies mid-commit. But it is the existence of the recovery procedure that licenses the early acknowledgment. This is the same "the commit decision must be recoverable from data, not from a process" discipline that distinguishes a correct 2PC from the broken kind in [two-phase commit and how it fails](/blog/software-development/database/two-phase-commit-and-how-it-fails).

| | Classic commit | Parallel commits |
| --- | --- | --- |
| Intent replication | round 1 | overlapped |
| Record write | round 2 (after round 1) | overlapped (STAGING) |
| Critical-path latency | ~2 × RTT | ~1 × RTT |
| Commit point | record = COMMITTED | record = STAGING **and** all intents replicated |
| Recovery if coordinator dies | read record status | run status-recovery (check in-flight writes) |
| Steady-state record state | COMMITTED | COMMITTED (after async explicit commit) |

## 7. Serializable isolation: the timestamp cache and read refreshing

> **Senior rule of thumb:** CockroachDB does not prevent conflicts by locking readers out. It lets everyone proceed optimistically, *records who read what*, and at commit time checks whether anyone's reads were invalidated. Serializable here is an *optimistic* protocol — closer to optimistic concurrency control than to two-phase locking — and that is why reads never block writes.

CockroachDB runs every transaction at SERIALIZABLE by default — the strongest ANSI isolation level, permitting *none* of the anomalies (dirty read, non-repeatable read, phantom, write skew) catalogued in [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent). It does this without the read locks that a classic serializable database would use. The two mechanisms that make lockless serializability work are the **timestamp cache** and **read refreshing**.

### The timestamp cache: writers must respect past readers

When a transaction reads a key at timestamp `T`, the leaseholder records that fact in an in-memory **timestamp cache**: "key `k` has been read up to timestamp `T`." The cache is the high-water mark of reads per key span. Now the rule that buys serializability: **a write to `k` must take a timestamp strictly greater than the highest read recorded for `k` in the timestamp cache.** If a writer arrives wanting to write `k` at timestamp 90, but the cache says `k` was read up to 100, the writer cannot write at 90 — doing so would mean a reader at 100 *should have seen* this write but didn't, a non-repeatable-read / lost-update anomaly. Instead, the writer's transaction is **pushed**: its timestamp is bumped forward to just above the cache value (101).

This is the inverse of locking. A read lock would *block* a later writer; the timestamp cache *lets the read happen freely* and instead forces any conflicting writer to move forward in time, where its write doesn't contradict the read. Readers never wait. The cost is borne by writers, and only writers that actually conflict with a recent read.

### Read refreshing: validating a pushed transaction

Pushing a transaction's timestamp forward — whether by the timestamp cache, an uncertainty restart, or a conflict — creates a new hazard. The transaction has already *read* some keys at its *original* timestamp. If it now commits at a *higher* timestamp, are those earlier reads still valid? Maybe someone wrote to a key the transaction read, at a timestamp between the original and the pushed one. If so, the transaction's read is stale at its new timestamp, and committing would violate serializability. So before committing at the higher timestamp, the transaction performs a **read refresh**: it re-checks its read-set (the keys it read) to confirm that *no new writes appeared* between its original timestamp and its pushed timestamp.

- Refresh succeeds (no intervening writes to the read-set) → the transaction's reads are still valid at the new timestamp; commit at the new timestamp. The push was free.
- Refresh fails (someone wrote to a key it read) → the transaction's reads are genuinely stale; it must **restart** from scratch at the higher timestamp, re-reading everything. This restart is surfaced to the client as the famous `40001` serialization-failure error.

The figure below traces the timestamp-cache-and-refresh path: transaction A reads `k` at 100 (recorded in the cache up to 100); transaction B tries to write `k` at 90, gets pushed to 101 (above the cache); B then read-refreshes — were there any new writes to B's read-set since 90? — and, finding none, commits at 101, serializably.

![A timeline showing transaction A reading a key at timestamp 100 and recording it in the timestamp cache, transaction B being pushed from 90 to 101, then a read refresh validating B's read-set before B commits at 101](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-7.webp)

### The retry loop: handling 40001 in application code

Because conflicts are resolved by pushing and restarting, a serializable transaction in CockroachDB *can fail at commit* with a `40001` (serialization failure / `RETRY_SERIALIZABLE`) and the contract is that the client retries it. This is not a bug or a degraded mode — it is the price of optimistic serializability, and well-written CockroachDB applications wrap every transaction in a retry loop. CockroachDB's own client drivers do this automatically (via `crdb.ExecuteTx` in Go, `run_transaction` in Python, etc.), but the loop is simple enough to write by hand and worth understanding:

```python
import psycopg2
import psycopg2.errorcodes
import time

def run_serializable_txn(conn, work, max_retries=10):
    """Execute `work(cursor)` in a SERIALIZABLE transaction, retrying on 40001.

    CockroachDB resolves write/read conflicts by pushing timestamps and, when a
    read refresh fails, aborting with SQLSTATE 40001. The contract is: the client
    re-runs the whole transaction. Backoff avoids a thundering-herd retry storm.
    """
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                work(cur)              # the application's reads + writes
            conn.commit()             # may raise 40001 here, at COMMIT
            return                    # success
        except psycopg2.errors.SerializationFailure as e:  # SQLSTATE == '40001'
            conn.rollback()
            if attempt == max_retries - 1:
                raise                 # give up after N tries
            # Exponential backoff with jitter: 5ms, 10ms, 20ms, ... + noise.
            sleep_s = (2 ** attempt) * 0.005
            time.sleep(sleep_s + (sleep_s * 0.1))
        except Exception:
            conn.rollback()
            raise                     # non-retryable error: propagate

# Usage: a classic read-modify-write that two clients might race on.
def transfer(cur):
    cur.execute("SELECT balance FROM accounts WHERE id = %s", (1,))
    (bal,) = cur.fetchone()
    if bal < 100:
        raise ValueError("insufficient funds")
    cur.execute("UPDATE accounts SET balance = balance - 100 WHERE id = %s", (1,))
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = %s", (2,))

run_serializable_txn(conn, transfer)
```

The pseudocode of the commit path, condensed, is:

```
commit(txn):
    # 1. all writes are already provisional intents, replicated by Raft
    # 2. if txn was ever pushed (ts cache / uncertainty / conflict):
    if txn.write_ts > txn.read_ts:
        if not read_refresh(txn.read_set, from=txn.read_ts, to=txn.write_ts):
            raise RetryError(40001)          # reads invalidated; client retries
        txn.read_ts = txn.write_ts           # refresh ok: advance read ts
    # 3. parallel commits: stage the record + finish intent consensus
    record.status = STAGING
    record.in_flight = txn.intent_keys
    wait_until(all_intents_replicated(txn.intent_keys))   # ~1 RTT, overlapped
    ack_client(committed=True)               # implicitly committed: tell client
    # 4. off critical path: explicit commit + intent resolution
    async:
        record.status = COMMITTED            # one Raft write on record's range
        for k in txn.intent_keys:
            resolve_intent(k)                # turn intent into plain MVCC value
```

### Read Committed: the escape valve for high-contention workloads

Newer CockroachDB versions also offer `READ COMMITTED` isolation, for workloads where serializable's restart rate is too painful (lots of contention on hot keys, or applications ported from Postgres that assume RC semantics). Under RC, write-read conflicts don't block reads and the per-statement (rather than per-transaction) retry scope makes `40001` far rarer, at the cost of allowing write-skew and other anomalies serializable forbids. The right default is still SERIALIZABLE — it is the only level that lets you reason about your transactions as if they ran one at a time — and RC is a deliberate, eyes-open downgrade for specific hot paths, not a blanket setting. The anomalies you re-admit by choosing RC are exactly the ones tabulated in the [isolation-levels post](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent).

| Mechanism | What it protects against | Who pays | Failure mode |
| --- | --- | --- | --- |
| Timestamp cache | a writer overwriting under a finished reader | the writer (pushed forward) | none — push is cheap |
| Read refresh | committing stale reads after a push | usually free; sometimes the reader | `40001` restart if refresh fails |
| Write intent (lock) | two writers racing the same key | the second writer (waits or pushes) | `40001` if priority-aborted |
| Uncertainty restart | reading a value of ambiguous order | the reader (restart, bounded) | extra latency on skewed clocks |

## 8. Leaseholders: serving reads without a Raft round-trip

> **Senior rule of thumb:** consensus is for *agreeing on writes*, not for serving reads. If every read needed a Raft quorum you'd pay a round trip to read a single row, and the database would be unusably slow. The leaseholder is the trick that makes reads local — and getting that trick right (the lease must be safe) is where a lot of subtle distributed-systems bugs hide.

Each range has a **leaseholder**: one of its replicas that holds the *range lease* and is responsible for coordinating all reads and writes to that range. Crucially, the leaseholder can serve reads *locally*, without going through Raft, because it holds a lease guaranteeing it has the most up-to-date view of the range. A read is therefore a single local MVCC lookup at the read timestamp — no quorum, no network round trip to other replicas. Writes still need Raft (a majority must agree to append the write to the log), so writes pay a round trip but reads do not. The figure below shows the asymmetry:

![A graph showing a SQL client routed to the leaseholder, which serves a read locally with no Raft round-trip, while a write must replicate to a majority of followers before committing](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-5.webp)

The leaseholder is, by default, the same replica as the Raft leader (the two are co-located so that the leaseholder, which receives the writes, is also the one proposing them to Raft — avoiding an extra hop). They can briefly diverge during a lease transfer, but the steady state is leaseholder == Raft leader.

Why is a local read *safe*? Because the lease is itself a coordinated, time-bounded grant. A lease is valid for an interval; the leaseholder may serve reads only within that interval; and the protocol guarantees at most one valid leaseholder per range at a time (a new lease can't be granted until the old one is provably expired or transferred). So when the leaseholder serves a read, it knows no other replica is concurrently committing writes it hasn't seen — it is the single serialization point for the range. This is the read-side analogue of Raft's leader-completeness guarantee, and it is exactly the "linearizable reads need their own protocol" problem from the [Raft post](/blog/software-development/database/raft-consensus-from-scratch): a naive "just read from the leader" is *unsafe* because a deposed leader might not know it's deposed, and the lease is the mechanism that makes it safe (a stale leaseholder's lease will have expired).

### Follower reads: trading freshness for locality

The leaseholder model means a read for a range whose leaseholder is in another region pays a cross-region round trip. For read-heavy, latency-sensitive, slightly-stale-tolerant workloads, CockroachDB offers **follower reads**: a query with `AS OF SYSTEM TIME follower_read_timestamp()` (typically ~4.8 seconds in the past) can be served by *any* replica, including a local follower, because reading sufficiently far in the past is safe on any replica (all replicas have agreed on everything up to a closed timestamp that old). This turns a cross-region read into a local one at the cost of bounded staleness — the right tool for "show me the catalog" but not for "show me my current balance."

```sql
-- A fresh read goes to the leaseholder (may be remote -> cross-region latency):
SELECT balance FROM accounts WHERE id = 42;

-- A follower read is served by the nearest replica, ~4.8s stale, no remote hop:
SELECT balance FROM accounts AS OF SYSTEM TIME follower_read_timestamp()
WHERE id = 42;

-- Bound staleness explicitly (read the freshest data no older than 10s, locally
-- if possible) -- good for dashboards that tolerate seconds-old numbers:
SELECT count(*) FROM events
AS OF SYSTEM TIME with_max_staleness('10s');
```

### Second-order consequence: leaseholder locality is your latency budget

Because reads go to the leaseholder, *where the leaseholders live* is the dominant lever on read latency in a multi-region cluster. CockroachDB's geo-partitioning features (`REGIONAL BY ROW` tables, partitioning ranges by a region column) exist to *place the leaseholder for a row in the region that reads it most*. Get this right and a European user reading European rows hits a European leaseholder — local latency. Get it wrong (all leaseholders defaulting to one region) and every read from elsewhere pays a cross-ocean round trip. The diagnostic is the `SHOW RANGES` output from earlier: if `lease_holder_locality` doesn't match where the traffic is, your latency problem is a leaseholder-placement problem, fixable in schema, not in code.

## 9. The allocator: split, merge, rebalance, repair

> **Senior rule of thumb:** the allocator is the database's autonomic nervous system. You don't call it; it runs continuously, comparing the actual replica layout to the desired one (from your zone configs) and nudging reality toward intent — one replica move at a time, slowly enough not to saturate the network. When a cluster "heals itself" after a node dies, this is what's doing it.

The range abstraction gives CockroachDB the *unit* of data movement; the **allocator** is the control loop that decides *which* movements to make. It runs on every node and continuously works to keep the cluster in its desired state along several axes. The figure below contrasts a stressed cluster with the post-allocator steady state:

![A before-after diagram showing an oversized 700 MB range, a range down to two replicas after a node death, and a node hotspot on the left, each resolved by a split, an up-replication, and a rebalance on the right](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-6.webp)

The allocator's main jobs:

- **Split.** When a range exceeds the size target (512 MB default) or gets too hot (high QPS), it splits into two ranges at a chosen key, each becoming its own Raft group. This is how a single growing table fans out across the cluster. Splits are cheap — they're a metadata operation on the sorted map, not a data copy.
- **Merge.** When adjacent ranges are both small (e.g., after a bulk delete or a `DROP`), the allocator merges them back together to avoid the per-range overhead of thousands of tiny Raft groups. Merge and split give the cluster elasticity in both directions.
- **Up-replicate (repair).** When a node dies and one of its ranges drops below its replication factor (e.g., 3 → 2 replicas), the allocator creates a fresh replica on a surviving node, restoring the factor. This is the self-healing: a permanently dead node's ranges are re-created elsewhere within minutes, and the cluster is back to full redundancy without operator action.
- **Rebalance.** When data or load is unevenly distributed (one node holding far more ranges, or leaseholders concentrated on one node), the allocator moves replicas and transfers leases to even things out, respecting the constraints in the zone configuration (e.g., "keep one replica per region", "never put two replicas in the same rack").

All of this is governed by **replication zone configurations** — declarative policy attached to the cluster, a database, a table, an index, or a partition:

```sql
-- Declarative intent: the allocator continuously reconciles reality to this.
ALTER TABLE accounts CONFIGURE ZONE USING
  num_replicas = 5,                       -- survive 2 simultaneous node failures
  constraints = '{"+region=us-east": 2, "+region=us-west": 2, "+region=eu": 1}',
  lease_preferences = '[[+region=us-east]]', -- keep leaseholders near the writers
  range_min_bytes = 134217728,            -- 128 MB: merge below this
  range_max_bytes = 536870912,            -- 512 MB: split above this
  gc.ttlseconds = 14400;                  -- keep 4h of MVCC history (time-travel)

-- The allocator's decisions are observable. This range is mid-rebalance:
SHOW RANGES FROM TABLE accounts WITH DETAILS;
--  range_id | replicas  | replica_localities                  | lease_holder
-- ----------+-----------+-------------------------------------+-------------
--        42 | {1,4,7}   | {us-east,us-west,eu}                 |            1
--        43 | {1,2,4}   | {us-east,us-east,us-west}  <- viol.  |            1   -- 2 in us-east, allocator will move one
```

### Second-order consequence: rebalancing is a background tax you can tune but not avoid

Every replica move is a snapshot of up to 512 MB shipped across the network plus a Raft membership change. The allocator deliberately rate-limits this (`kv.snapshot_rebalance.max_rate`) so that healing and rebalancing don't saturate the links and starve foreground traffic. The tension is real: after losing a node, you want up-replication to finish *fast* (to restore redundancy before a second failure) but *not so fast* it tanks query latency. The defaults are conservative; in an incident where you've lost a node and are racing to restore the replication factor, raising the rebalance rate temporarily is a legitimate (and reversible) operator move. The general lesson is that a self-healing system isn't free-healing — the healing competes with your workload for the same network, and tuning that competition is part of operating the database.

## 10. Pebble: the storage engine where the bytes land

> **Senior rule of thumb:** under every distributed layer there is, eventually, a single-node storage engine writing bytes to a disk, and its physics — write amplification, read amplification, compaction debt — leak upward into your tail latencies. Pebble is an LSM tree, and if you don't understand LSM economics you will be surprised by CockroachDB's disk behavior.

The bottom layer, the only one that touches the disk, is **Pebble** — a from-scratch, Go-native, RocksDB-inspired log-structured merge-tree that CockroachDB built to replace RocksDB (eliminating the CGo boundary and tailoring the engine to CockroachDB's exact access patterns). Pebble is the local key-value store each replica writes into; it stores the MVCC-encoded keys (`key + timestamp → value`) durably and serves point and range reads efficiently.

The LSM mechanics, in brief, because they explain CockroachDB's disk profile (covered in depth in [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines)):

- A write goes first to the **write-ahead log** (for durability on crash) and the in-memory **memtable** (a sorted structure). Returning from the write is fast — it's an append plus a memory insert, no random disk seek.
- When the memtable fills, it is flushed to disk as an immutable **SSTable** in level **L0**.
- **Compaction** continuously merges SSTables downward through levels L0 → L6, each level ~10× the size of the one above, keeping keys sorted and discarding superseded MVCC versions and tombstones. L0 is the only level allowed overlapping key ranges; L1–L6 are non-overlapping for efficient lookups.

This shape makes writes fast (sequential, no in-place updates) at the cost of *read amplification* (a read may have to check several levels) and *write amplification* (each byte is rewritten several times as it compacts down). The pathology to watch is an **inverted LSM**: if compaction falls behind the write rate, L0 accumulates too many overlapping files, reads have to scan all of them, and read latency spikes. CockroachDB exposes the LSM "shape" and a read-amplification metric precisely so you can catch this; an inverted LSM under sustained write load is the storage-layer equivalent of a clogged drain, and it usually means the disk's IOPS can't keep up with compaction.

### MVCC garbage collection: the history has a half-life

MVCC keeps old versions, but not forever — that would grow the store without bound. Each range runs **GC** that removes MVCC versions older than the zone's `gc.ttlseconds` (default 25 hours) *once a newer version exists*. This TTL is also the time-travel horizon: `AS OF SYSTEM TIME` can read back only as far as the GC TTL, because older versions are gone. There's a tension: a long TTL means more time-travel range and safer backups but more storage and more versions to scan; a short TTL saves space but can break long-running queries or backups that try to read history that's been collected. **Protected timestamps** resolve this — a backup or changefeed creates a protection record that pins the history it needs, so you can run a short GC TTL (saving space) without your backups failing because the data they wanted got collected.

```sql
-- Time-travel is just reading at an older MVCC timestamp -- bounded by gc.ttlseconds.
SELECT * FROM accounts AS OF SYSTEM TIME '-30s' WHERE id = 42;  -- 30s ago
SELECT * FROM accounts AS OF SYSTEM TIME '2026-06-14 09:00:00'; -- if within TTL

-- Inspect storage-layer health (read-amplification, LSM shape) per store:
SELECT store_id, metric, value
FROM crdb_internal.kv_store_status_metrics  -- illustrative; see admin UI / metrics
WHERE metric IN ('rocksdb.read-amplification', 'storage.l0-sublevels');
```

## 11. How CockroachDB compares: Spanner, TiDB, Vitess

> **Senior rule of thumb:** there is no "best" distributed SQL database; there is the one whose *clock assumptions and sharding model* match your operational reality. Pick by the constraint you can't change — do you have atomic clocks (Spanner), do you need to keep your MySQL ecosystem (Vitess), do you want snapshot-isolation-by-default and a central timestamp oracle (TiDB), or do you want serializable on commodity hardware with no central oracle (CockroachDB)?

The four systems share the broad NewSQL ambition — horizontal scale with relational semantics — but diverge on the axes that actually determine behavior. The matrix below lays them side by side:

![A matrix comparing CockroachDB, Spanner, TiDB, and Vitess across clock source, isolation default, shard unit, consensus, and hardware requirements](/imgs/blogs/cockroachdb-distributed-sql-deep-dive-9.webp)

**Versus Spanner.** Same architecture lineage — monolithic sorted key-space, ranges/tablets, per-range consensus, MVCC. The single difference that cascades into everything is the clock. Spanner has TrueTime (atomic clocks + GPS, ~7 ms uncertainty) and pays uncertainty as *commit-wait* (every write waits out the window), buying *external consistency* (strict serializability across all keys). CockroachDB has HLC + a 500 ms static offset and pays uncertainty as *occasional read restart*, buying *serializability + single-key linearizability* but not strict serializability across disjoint keys. Spanner uses Paxos per tablet; CockroachDB uses Raft per range. Spanner needs Google's hardware; CockroachDB runs on anything with NTP. The whole [Spanner post](/blog/software-development/database/spanner-truetime-and-external-consistency) is the other half of this comparison.

**Versus TiDB / Percolator.** TiDB (built on TiKV) descends directly from Google Percolator. Its transaction model is the same primary-lock-on-data idea CockroachDB also uses — but TiDB gets its timestamps from a *central oracle* (the Placement Driver's TSO), a single logical source of monotonic timestamps, whereas CockroachDB has *no central oracle* and derives ordering from per-node HLCs plus the offset bound. The oracle simplifies ordering (one source of truth for time) but is a component that must itself be made highly available and whose round trip is on the transaction path. TiDB's default isolation is *snapshot isolation* (with an optional pessimistic mode), where CockroachDB defaults to *serializable* — TiDB trades away write-skew protection by default for fewer aborts. Both shard a key-space into ranges/regions replicated by Raft. The shared transaction lineage is in the [Percolator post](/blog/software-development/database/percolator-distributed-snapshot-isolation).

**Versus Vitess.** Vitess is a different animal: it is a sharding and connection-management layer *on top of vanilla MySQL*. It does not have a distributed transaction protocol with cross-shard serializability, a single sorted key-space, or per-range consensus; it has MySQL instances, a routing tier (VTGate), and a sharding scheme (VSchema) you design and maintain. Cross-shard transactions in Vitess are best-effort (2PC is available but not the happy path) and isolation is whatever each MySQL shard provides locally. Vitess shines when your constraint is "we have a giant MySQL deployment and need to scale it horizontally while keeping MySQL compatibility and operational know-how"; it is the *least* "distributed database" of the four and the *most* "scale your existing database." CockroachDB, by contrast, is a single logical database that happens to be distributed, with automatic sharding and genuine cross-range ACID.

| Dimension | CockroachDB | Spanner | TiDB | Vitess |
| --- | --- | --- | --- | --- |
| Clock source | HLC + 500 ms offset | TrueTime (~7 ms, atomic/GPS) | central TSO oracle (PD) | MySQL local clocks |
| Consistency | serializable + single-key linearizable | external consistency (strict serializable) | snapshot isolation (default) | per-shard MySQL |
| Default isolation | SERIALIZABLE | SERIALIZABLE / external | SI (opt. pessimistic) | MySQL level per shard |
| Sharding | auto ranges (~512 MB) | auto tablets | auto regions (TiKV) | manual VSchema |
| Consensus | Raft per range | Paxos per tablet | Raft per region | MySQL replication |
| Cross-shard ACID | yes, native | yes, native | yes, native (Percolator) | best-effort 2PC |
| Special hardware | none (NTP) | atomic clocks + GPS | none (PD oracle) | none |
| Wire protocol | PostgreSQL | gRPC / SQL | MySQL | MySQL |

## Case studies from production

The mechanisms above are clean on paper. Here is how they bite in practice — symptoms, wrong first hypotheses, root causes, fixes, lessons. Names are evocative; the patterns are real and recurring.

### 1. The sequential-key hotspot that scaling couldn't fix

A team migrated an events table from Postgres, keeping its `BIGSERIAL` auto-increment primary key, and onto a 12-node CockroachDB cluster. Throughput was *worse* than the single Postgres box. The wrong first hypothesis was "we need more nodes" — they added six, and nothing improved. The root cause was the range model: a monotonic key means every insert lands at the high end of the key-space, in the *same* range, on the *same* leaseholder, on *one* node. Eleven nodes sat idle while one was saturated; the extra nodes only added cross-node latency for the few reads that hit them. The fix was to hash-shard the primary-key index (`CREATE INDEX ... USING HASH WITH (bucket_count = 16)`), which spreads inserts across 16 ranges and therefore many leaseholders. Throughput went up roughly linearly with nodes after that. The lesson: in a range-partitioned store, the schema *is* the sharding strategy, and a sequential key defeats the entire architecture before it starts.

### 2. The clock-skew node suicides during a noisy-neighbor incident

A cluster on a busy virtualized host started losing nodes — they were crashing with clock-offset errors and restarting in a loop. The wrong first hypothesis was a CockroachDB bug ("why is the database killing itself?"). The actual cause was the hypervisor: under heavy contention, the VMs' clocks were being starved of CPU and drifting more than 400 ms (80% of the 500 ms default offset) from their peers, and CockroachDB was doing *exactly what it's designed to do* — self-terminating rather than risk a stale read. The fix was twofold: move to instances with better clock discipline (and `chrony` instead of stock NTP) to keep real offset tiny, and, as a stopgap, the team understood that *lowering* `max-offset` would make it worse (less slack) and *raising* it trades safety for stability — so they fixed the clocks instead of the knob. The lesson: the clock is a hard dependency of correctness, the node-suicide is a feature not a bug, and good clock sync is non-negotiable infrastructure, not a nice-to-have.

### 3. The cross-region commit latency that parallel commits halved

A fintech ran a three-region cluster (us-east, us-west, eu) and saw transaction latencies of ~140 ms for simple two-row transfers, double what they expected from the ~70 ms cross-region RTT. The wrong first hypothesis was network tuning. The root cause was that the transaction touched two ranges whose leaseholders were in different regions, and on the old version they were on, the commit took *two* sequential consensus rounds — intents, then the record flip — so `2 × RTT ≈ 140 ms`. Upgrading to a version with parallel commits enabled collapsed the two rounds into one overlapped round, and latency dropped to ~75 ms — almost exactly `1 × RTT`. The lesson: in geo-distributed databases, count the round trips on the critical path; an optimization that removes one WAN round trip is worth more than any amount of CPU tuning, and parallel commits is that optimization for the commit path.

### 4. The long-running transaction that serialized the whole hot path

An analytics job opened a transaction, ran a slow aggregation, then updated a summary row — and held the transaction open for the entire aggregation, several seconds. Meanwhile, every OLTP transaction touching the summary row's range stalled. The wrong first hypothesis was a deadlock. The actual cause was the long transaction's *write intent* on the summary row: an intent is an exclusive lock, and every other writer queued behind it for seconds, producing a tail-latency cliff and a spike in `40001` retries as pushed transactions failed their refreshes. The fix was to restructure the job: compute the aggregation *outside* a transaction (or in a follower read), then do the summary update in a tiny, fast transaction. The lesson, identical to MVCC systems everywhere: hold write transactions for microseconds, not seconds; an open transaction is a held lock, and held locks on hot keys serialize everyone.

### 5. The phantom that serializable caught and read committed would have let through

A team ported an inventory system from a database running at READ COMMITTED. Under CockroachDB's default SERIALIZABLE, they started seeing `40001` retries on a "check stock, then reserve" transaction under concurrency. Their first instinct was "serializable is too strict, let's downgrade to READ COMMITTED to make the errors go away." On inspection, the retries were *catching a real write-skew bug*: two concurrent transactions each read "stock = 1", each decided it was safe to reserve, and at the old isolation level *both* would have reserved the last item, overselling. SERIALIZABLE detected the conflict (the second transaction's read refresh failed) and forced one to retry, after which it correctly saw stock = 0 and refused. Downgrading to READ COMMITTED would have re-introduced the oversell. The fix was to *keep* SERIALIZABLE and add the retry loop. The lesson: a `40001` is often not noise to suppress but a real anomaly being prevented; downgrade isolation only when you've proven the anomaly it permits is actually acceptable for that transaction.

### 6. The leaseholder in the wrong region

A European-heavy application on a us-east-default cluster had p99 read latencies of ~120 ms for European users. The wrong first hypothesis was CDN/edge caching. The root cause surfaced in `SHOW RANGES`: every range's leaseholder was in us-east, so every read from Europe paid a transatlantic round trip to the leaseholder. The fix was geo-partitioning — making the table `REGIONAL BY ROW` with a region column, so each row's range got a leaseholder in the row's home region, and European rows were leaseheld in eu. European read latency dropped to local (~5 ms). The lesson: with leaseholder-served reads, leaseholder *placement* is your read-latency budget, and it's controlled in the schema (partitioning + lease preferences), not in application code.

### 7. The inverted LSM under a bulk import

During a large data import, a node's read latencies climbed steadily even though writes were keeping up. The wrong first hypothesis was a slow disk in general. The root cause was an *inverted LSM*: the import's write rate outran Pebble's compaction, L0 accumulated dozens of overlapping sublevels, and every read had to scan all of them, driving read amplification through the roof. The disk wasn't slow for writes — it just couldn't sustain both the write firehose and the compaction needed to keep the LSM in shape. The fix was to throttle the import to give compaction headroom (and, longer term, faster NVMe with the IOPS to keep up). The lesson: an LSM trades write speed for compaction debt, and if you let the debt accumulate, reads pay the interest; monitor L0 sublevels / read-amplification during heavy writes.

### 8. The under-replicated range after a node loss

A node died and didn't come back. The cluster kept serving traffic (every range still had a majority), but an alert fired: dozens of ranges were *under-replicated* (2 of 3 replicas). The wrong first hypothesis was "the cluster is broken." The reality was the cluster working as designed — but the allocator's up-replication was deliberately rate-limited and would take ~20 minutes to restore full redundancy at the default rebalance rate. The risk during that window: a *second* node failure could lose a majority for the ranges that still had a replica on it. The team made the right call — temporarily raised `kv.snapshot_rebalance.max_rate` to finish up-replication faster, accepting some foreground-latency cost to close the redundancy gap sooner, then reverted it. The lesson: self-healing is rate-limited by design, the heal competes with your workload, and during a redundancy gap it can be correct to spend latency to buy back safety faster.

### 9. The contention from a "harmless" SELECT FOR UPDATE pattern

A team used `SELECT ... FOR UPDATE` to lock rows before updating them, a common Postgres pattern, on a hot counter table. Under load, throughput collapsed and `40001`s spiked. The wrong first hypothesis was that `FOR UPDATE` should *reduce* retries by locking up front. The nuance: `FOR UPDATE` does acquire a lock (a write intent) up front, which *does* reduce read-refresh-style retries — but on a *single hot row*, it also serializes every transaction through that one lock, converting a parallel workload into a serial one with a queue. The fix wasn't to remove the locking; it was to remove the *single hot row* — the counter was re-modeled as a set of sharded sub-counters summed at read time, spreading the contention. The lesson: explicit locking changes *which* concurrency-control cost you pay (fewer refreshes, more queueing) but can't conjure parallelism out of a single contended key; the architectural fix is to eliminate the hot key.

### 10. The backup that failed because GC ate the history

A nightly backup started failing intermittently with "batch timestamp must be after replica GC threshold." The wrong first hypothesis was a backup-tool bug. The root cause was a too-short `gc.ttlseconds` (someone had lowered it to save storage) combined with a backup that ran long enough that, by the time it read some ranges, the historical timestamp it was reading *as of* had already been garbage-collected. The fix was to rely on *protected timestamps*: ensure the backup created a protection record pinning the timestamp it needed (modern backups do this automatically), so GC couldn't collect history the backup still wanted, *and* the short TTL could stay for everything else. The lesson: MVCC history has a half-life (`gc.ttlseconds`), anything reading the past (backups, changefeeds, long `AS OF SYSTEM TIME` queries) races against GC, and protected timestamps are the mechanism that lets a short TTL and long readers coexist.

### 11. The Jepsen-class reasoning that saved a real design

A team building on CockroachDB assumed it provided strict serializability (real-time order across all transactions) and designed a workflow where service X writes record R, then signals service Y out-of-band, and Y reads a *different* record S expecting to see effects ordered after R. Occasionally Y saw a state inconsistent with that ordering. The wrong first hypothesis was a CockroachDB consistency bug. The root cause was the *documented* gap that Jepsen's analysis named: CockroachDB is serializable and single-key linearizable but *not* strict-serializable across disjoint keys, so two transactions on different keys (R and S) with no data dependency can be ordered against real time. The out-of-band signal created a causal dependency the *database couldn't see*. The fix was to make the dependency visible to the database — route the signal through the data (have Y read R, or carry a causality token / the observed HLC timestamp from X to Y) so CockroachDB's causality machinery enforced the order. The lesson: know exactly which consistency model you have; CockroachDB's is serializable + single-key linearizable, and dependencies you keep *outside* the database are dependencies the database is allowed to reorder.

## When to reach for distributed SQL (and when not to)

> Distributed SQL is the right tool when "the database must never be the single point of failure, and it must scale writes, and it must stay relational and strongly consistent" are all true at once. Drop any one of those and a simpler system probably wins.

### Reach for CockroachDB when

- **You need to survive node and datacenter failure with no manual failover.** Multi-active availability and per-range Raft mean a node (or a whole zone) can vanish and writes keep flowing for the rest of the key-space, with automatic re-replication. If "the primary died at 3 a.m. and someone has to promote a replica" is a sentence you never want to say, this is the architecture.
- **You need to scale writes horizontally, not just reads.** Read replicas scale reads; they don't scale writes. CockroachDB scales both, because every node is a leaseholder for some ranges and writes are distributed across the cluster.
- **You want strong consistency and relational semantics, not eventual consistency.** SERIALIZABLE by default, MVCC, real SQL with the PostgreSQL wire protocol. You get to *reason* about your transactions as if they ran one at a time — the property that eventually-consistent stores make you give up.
- **You need geo-distribution with data placement control.** `REGIONAL BY ROW`, leaseholder preferences, and per-partition zone configs let you pin data near the users who read it, satisfy data-residency requirements, and still present one logical database.
- **You want automatic sharding you don't have to operate.** Ranges split, merge, and rebalance themselves. You don't design a sharding scheme or run a resharding migration when a shard gets hot.

### Skip CockroachDB when

- **A single Postgres/MySQL instance comfortably fits your load.** Distributed SQL adds latency (consensus round trips, possible cross-range commits) and operational surface (clocks, ranges, the allocator) that a single node simply doesn't have. If a well-tuned single node with a read replica or two handles your traffic with headroom, *use that* — the distributed-systems tax is real and you'd be paying it for nothing.
- **Your workload is single-region, latency-critical, and contention-heavy on hot keys.** The optimistic, restart-based serializability and the consensus round trip can make a hot-key OLTP workload slower and retry-ier than a single-node database with row locks. Measure before committing; the architecture that scales out can lose to the one that doesn't on a contended single-region hot path.
- **You need strict serializability across disjoint keys (external consistency).** CockroachDB deliberately gives this up to avoid atomic clocks. If your correctness genuinely depends on real-time ordering of *unrelated* transactions, that is Spanner's niche, and you should pay for TrueTime or design the dependency into the data.
- **You're scaling an existing MySQL deployment and want to keep the ecosystem.** If the constraint is "we have enormous MySQL and need horizontal scale with MySQL compatibility and the team's MySQL muscle memory", Vitess fits the constraint that CockroachDB doesn't — it scales *MySQL*, rather than replacing it with a different database.
- **Your data is fundamentally analytical, not transactional.** CockroachDB is an OLTP system. For column-store scan-heavy analytics, a purpose-built OLAP engine will crush it; the right pattern is CockroachDB for the transactional system of record plus CDC into a warehouse, not CockroachDB *as* the warehouse.

## Further reading

- CockroachDB architecture docs — the [overview](https://www.cockroachlabs.com/docs/stable/architecture/overview), the [transaction layer](https://www.cockroachlabs.com/docs/stable/architecture/transaction-layer), the [storage layer](https://www.cockroachlabs.com/docs/stable/architecture/storage-layer), and [life of a distributed transaction](https://www.cockroachlabs.com/docs/stable/architecture/life-of-a-distributed-transaction).
- Cockroach Labs engineering blog: [Living Without Atomic Clocks](https://www.cockroachlabs.com/blog/living-without-atomic-clocks/) (HLC vs TrueTime), [Serializable, lockless, distributed: Isolation in CockroachDB](https://www.cockroachlabs.com/blog/serializable-lockless-distributed-isolation-cockroachdb/), [Parallel Commits](https://www.cockroachlabs.com/blog/parallel-commits/), and [Clock Management in CockroachDB](https://www.cockroachlabs.com/blog/clock-management-cockroachdb/).
- [Jepsen's analysis of CockroachDB beta-20160829](https://jepsen.io/analyses/cockroachdb-beta-20160829) — the independent verification that named the serializable-but-not-strict-serializable gap and the causal-reverse anomaly.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 9 (consistency, consensus, linearizability vs serializability) and Chapter 7 (transactions, snapshot and serializable isolation).
- Sibling posts on this blog: [Raft from scratch](/blog/software-development/database/raft-consensus-from-scratch), [Spanner and TrueTime](/blog/software-development/database/spanner-truetime-and-external-consistency), [Percolator](/blog/software-development/database/percolator-distributed-snapshot-isolation), [time, clocks, and ordering](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems), [two-phase commit and how it fails](/blog/software-development/database/two-phase-commit-and-how-it-fails), [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), and [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines).

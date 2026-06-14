---
title: "Percolator: Distributed Snapshot Isolation on a Key-Value Store"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "How Google Percolator and its open descendant TiKV/TiDB layer cross-shard ACID snapshot-isolation transactions on top of a plain sharded key-value store, using MVCC, a global timestamp oracle, and a primary lock that stores the commit decision in the data itself."
tags:
  [
    "percolator",
    "distributed-transactions",
    "snapshot-isolation",
    "tikv",
    "tidb",
    "mvcc",
    "timestamp-oracle",
    "distributed-systems",
    "two-phase-commit",
    "databases",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/percolator-distributed-snapshot-isolation-1.webp"
---

Every distributed SQL database that promises you "ACID transactions across a hundred nodes" is hiding the same magic trick. Underneath the SQL parser, the query optimizer, and the cost-based planner, there is almost always a dumb, sharded key-value store — a thing that knows how to put and get bytes by key, replicate each shard with Raft, and split a shard in half when it gets too big. That key-value store has no idea what a transaction is. It cannot tell you whether two writes to two different shards happened "together." Ask it to debit Bob's account on shard 7 and credit Joe's account on shard 19 as one atomic unit and it will shrug: those are two unrelated put operations to two unrelated machines, and if the second one fails after the first one lands, you have just invented money.

So the entire problem of building a distributed database reduces, in large part, to this: **how do you get cross-shard, serializable-ish, ACID transactions on top of a storage layer that only understands single-key puts and gets?** This is not a theoretical curiosity. It is the exact gap that Google's Percolator was built to close in 2010, that TiKV and TiDB close today for the open-source world, that CockroachDB and Spanner close with different tools, and that you will hit the moment your data outgrows a single Postgres box and someone says the words "let's shard it."

The answer Percolator gives — and the one this article unpacks end to end — is surprisingly economical. You do not need a heavyweight distributed transaction manager with its own write-ahead log and its own consensus group. You do not need every node to phone home to a central coordinator on every commit. Instead you need three things: a way to give every transaction a globally agreed sense of time (a **timestamp oracle**), a way to keep multiple versions of every value so readers and writers never block each other (**MVCC**), and one clever idea that does almost all the heavy lifting — **you store the commit decision inside the data itself**, in a designated "primary" row, so that the question "did this transaction commit?" always has a single, durable, atomically-flippable answer that any participant can go read for themselves.

The diagram above is the mental model the rest of the article unpacks. A single logical cell — say, `Bob.balance` — is not stored as one value. Percolator explodes it into three parallel sub-columns: a `data` column holding the actual values keyed by version timestamp, a `lock` column that is non-empty only while a transaction is mid-flight, and a `write` column that is the index of *committed* versions, mapping a commit timestamp to the start timestamp where the real value lives. A committed write leaves a trail: the value sits in `data` at the writer's start timestamp, the lock is gone, and a pointer in `write` at the commit timestamp tells every future reader "this is the version you may see." Reads and writes are just disciplined walks across these three columns at the right timestamps. Everything else — atomicity, isolation, crash recovery — falls out of that layout.

![A committed Percolator cell splits into a data sub-column holding values at start_ts, an empty lock sub-column, and a write sub-column whose commit_ts entry points back to the data at start_ts](/imgs/blogs/percolator-distributed-snapshot-isolation-1.webp)

> A distributed transaction is not a special kind of write. It is a *protocol* layered on top of ordinary single-key writes, whose only job is to make a set of independent puts appear to happen at one instant — or not at all. Percolator's insight is that the protocol's state can live in the data instead of in a coordinator.

## Why this is different from what most engineers assume

Most engineers' mental model of a transaction comes from a single-node relational database: `BEGIN`, do some writes, `COMMIT`, and a friendly storage engine with full visibility over its own buffer pool and write-ahead log makes the whole thing atomic and isolated for you. That model quietly assumes a single authority that can see every page, take every lock, and flush one durable commit record. None of those assumptions survive contact with a sharded store where the rows of one transaction live on different machines that have never heard of each other.

| Assumption from single-node databases | The comfortable mental model | The sharded-KV reality Percolator must survive |
| --- | --- | --- |
| "The storage engine sees the whole transaction." | One process holds the buffer pool, the locks, the WAL. | Each key may live on a different shard, replicated by its own Raft group, with no shared lock table and no shared log. |
| "Commit is one atomic fsync." | A single commit record makes everything durable at once. | There is no single place to write "committed." Two writes on two shards commit independently; you must *synthesize* atomicity. |
| "Readers and writers coordinate through locks." | Writers block readers, readers block writers. | At cross-shard scale, lock-based blocking is a latency and deadlock disaster. You want snapshot reads that never block on writers. |
| "A coordinator holds the transaction's fate." | If the coordinator crashes, recovery reads its log. | A blocking coordinator is a single point of failure that pins locks forever when it dies. Percolator has no such coordinator. |
| "Clocks are reliable enough to order events." | `now()` is good enough. | Independent machine clocks drift by hundreds of milliseconds; you cannot order cross-shard events by wall clock without help. |

Every row in the right column is a real failure mode, and Percolator's design is a direct response to each. The lack of a shared engine forces MVCC-in-the-data. The lack of a single commit record forces the primary-lock trick. The latency cost of blocking forces snapshot isolation. The fragility of a coordinator forces lazy, reader-driven cleanup. And the unreliability of clocks forces a centralized timestamp oracle. The rest of this article introduces each mechanism alongside the failure it closes — never a mechanism for its own sake.

A note on lineage before we dive in. Percolator was described by Daniel Peng and Frank Dabek in the OSDI 2010 paper [*Large-scale Incremental Processing Using Distributed Transactions and Notifications*](https://research.google/pubs/large-scale-incremental-processing-using-distributed-transactions-and-notifications/). Google built it not to be a general database but to *incrementally* update the web search index: when one page changes, recompute only what depends on it, instead of re-running a colossal MapReduce over the whole web. That incremental workload needed cross-row ACID transactions over Bigtable, and Percolator was the answer. The paper reports it cut the median document indexing latency by roughly **100x** versus the previous MapReduce-based system — at the cost of about a **30x** overhead per unit of work compared to a traditional DBMS, a trade Google made deliberately to get linear scaling on commodity machines. The open-source world adopted the same protocol almost wholesale: [TiKV](https://tikv.org/deep-dive/distributed-transaction/percolator/), the distributed transactional key-value store under [TiDB](https://docs.pingcap.com/tidb/stable/), implements Percolator faithfully and then optimizes it, and it is the canonical implementation you can actually read the source of. We will lean on TiKV's docs throughout, because they are precise where the paper is terse.

## 1. The substrate: what a sharded KV store gives you, and what it doesn't

> **Senior rule of thumb:** before you design a transaction protocol, write down the *exact* atomic primitive your storage layer offers. Everything you build is a tower on top of that one guarantee, and if you misremember it, your tower falls over under concurrency.

Percolator's substrate was Bigtable; TiKV's is a sharded, Raft-replicated log-structured store backed by RocksDB. The two differ in a hundred ways, but they share the one property that makes Percolator possible: **atomic single-row (single-key) read-modify-write**. Bigtable guarantees that operations on a single row are atomic — a conditional mutation on one row either fully happens or doesn't, even under concurrent access. TiKV provides the same via per-key Raft consensus: a write to one key is a single Raft proposal that commits atomically within that key's region.

That single-key atomicity is the only consensus primitive Percolator needs. It does not run its own Paxos or Raft for transactions. It borrows the storage layer's per-key atomicity and composes many such atomic single-key operations into a multi-key transaction. This is the load-bearing decomposition: the *hard* part (agreeing on a single value despite faults) is already solved by the KV store per key; Percolator's job is only to *orchestrate* a set of those atomic single-key writes so they look atomic together.

What the substrate does **not** give you is multi-key atomicity. There is no `BEGIN`/`COMMIT` spanning two keys. There is no global lock table. There is no shared transaction log. If you put `x=1` on shard A and then try to put `y=2` on shard B and shard B is unreachable, you are stuck with a half-written transaction and no built-in way to undo `x=1`. Bridging that gap is the whole game.

Here is the storage interface Percolator assumes, written as a small Python-ish API so the rest of the article has something concrete to call. The key thing to notice is `cas` (compare-and-set on a single row), which is the atomic primitive everything is built from:

```python
from dataclasses import dataclass
from typing import Optional

# A "column" here is one of: 'data', 'lock', 'write'. Each is physically a
# separate column family in RocksDB (CF_DEFAULT, CF_LOCK, CF_WRITE in TiKV).
# Keys in 'data' and 'write' are (row_key, version_ts); 'lock' is keyed by
# row_key alone because at most one lock can exist per key at a time.

@dataclass
class KV:
    """The substrate: a sharded KV store with atomic SINGLE-KEY operations."""

    def get(self, row: str, col: str, ts: int) -> Optional[tuple[int, bytes]]:
        """Newest entry in (row, col) with version <= ts. Returns (version, value)."""
        ...

    def put(self, row: str, col: str, ts: int, value: bytes) -> None:
        """Write one versioned cell. Atomic for this single key."""
        ...

    def erase(self, row: str, col: str, ts: int) -> None:
        """Delete one versioned cell. Atomic for this single key."""
        ...

    def cas_lock(self, row: str, *, expect_empty_above: int) -> bool:
        """
        ATOMIC, single-row: succeed only if (row,'lock') has no lock AND
        (row,'write') has no committed version newer than expect_empty_above.
        On success, the caller's lock has been placed. This is the only
        consensus-grade primitive Percolator needs from the substrate.
        """
        ...
```

That `cas_lock` is a stand-in for "do a Bigtable conditional row mutation" or "issue one TiKV prewrite RPC, which is itself one Raft write to one region." The crucial property is atomicity at the granularity of one row. Hold that thought; it is the hinge of the entire protocol.

### Second-order optimization: column families, not literal extra columns

In the paper, the three sub-columns are, in effect, `c:data`, `c:lock`, `c:write` living in the same Bigtable. In TiKV they are three separate RocksDB column families — `CF_DEFAULT` for data, `CF_LOCK` for locks, `CF_WRITE` for commit records — which is a meaningful engineering win. Locks are tiny and churn constantly (written on prewrite, deleted on commit); putting them in their own column family keeps the lock working set small and hot in memory, and keeps lock churn from fragmenting the much larger data column family's LSM tree. The `write` column family, the version index, is also small and is the one scanned on every read, so it benefits from its own bloom filters and block cache. Separating physical storage by access pattern is one of those decisions that looks cosmetic and is actually load-bearing for tail latency under write-heavy churn.

## 2. Snapshot isolation and MVCC: the isolation model Percolator actually provides

> **Senior rule of thumb:** know your isolation level before you reason about correctness. Percolator gives you *snapshot isolation*, not serializability — which means it prevents dirty reads, non-repeatable reads, and lost updates, but it does **not** prevent write skew. If you assume serializability you will eventually ship a bug.

Snapshot isolation (SI) is the contract Percolator offers, and it is worth being precise about what that means, because it is the same model Kleppmann describes in *Designing Data-Intensive Applications* Chapter 7 as the workhorse of modern databases. Under SI, every transaction reads from a consistent **snapshot** of the database taken at the instant the transaction started. From the transaction's point of view, the database is frozen at `start_ts`: it sees all writes committed before `start_ts` and none committed after, regardless of how many other transactions commit while it runs. Reads never block on concurrent writes, and writes never block on concurrent reads, because they touch different versions. This is multi-version concurrency control — MVCC — and it is why a long analytical read in Postgres or InnoDB doesn't freeze your OLTP traffic. (For the single-node mechanics of how Postgres and InnoDB implement this with version chains and undo logs, see the companion [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb).)

The price of SI is that it is *not* serializable. The canonical hole is **write skew**: two transactions each read an overlapping set of rows, each makes a decision based on what it read, and each writes a *different* row. Neither sees the other's write (they have disjoint write sets, so there's no write-write conflict), both commit, and the combined result violates an invariant that each transaction individually preserved. The textbook example is two doctors on call: each checks "is at least one other doctor on call?", sees yes, and each takes themselves off call — leaving zero doctors covered. Under serializability one of them would have to see the other's change and abort; under SI both commit. Percolator inherits this limitation directly from SI; it detects write-*write* conflicts but not read-write skew. If your application needs serializability you must either add explicit conflict materialization (write a sentinel row that turns the skew into a write-write conflict) or use a database that offers serializable isolation. (The full taxonomy of which anomalies each level prevents is laid out in [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent).)

Now, the mechanism. MVCC needs two things: every value tagged with the version (timestamp) at which it became visible, and a rule for picking which version a given reader sees. Percolator's three-column layout is exactly this. The `data` column stores `(value)` at key `(row, start_ts)` — the version is the *writer's* start timestamp. The `write` column stores a pointer `(start_ts)` at key `(row, commit_ts)` — this is the visibility index, mapping the commit time to where the value lives. A reader at snapshot `read_ts` finds the newest `write` entry whose `commit_ts <= read_ts`, reads the `start_ts` it points to, and fetches `data` at that `start_ts`. That two-hop indirection — write column to find the version, data column to fetch it — is the heart of SI in Percolator.

Why two columns instead of one? Because the version a writer uses to *store* its value (`start_ts`) is different from the version at which that value becomes *visible* (`commit_ts`), and that gap is essential. A transaction writes its data at `start_ts` during prewrite, before it knows whether it will commit. Only at commit does it learn `commit_ts` and publish the pointer. If readers indexed by `start_ts`, they could see a value that was written but never committed. By indexing visibility through the `write` column keyed on `commit_ts`, an uncommitted value in `data` is simply invisible — there's no `write` pointer to it yet. The data is there; nobody can see it until the pointer appears. That is how MVCC and atomic commit fit together in one layout.

| Column | Key | Value | Written when | Read by |
| --- | --- | --- | --- | --- |
| `data` | `(row, start_ts)` | the actual bytes | prewrite (before commit known) | second hop of a read |
| `lock` | `(row)` | primary pointer + metadata | prewrite | conflict / cleanup checks |
| `write` | `(row, commit_ts)` | `start_ts` it points to (+ type) | commit | first hop of a read |

The `write` value also carries a *type* — `Put`, `Delete`, `Rollback`, or `Lock` — which TiKV uses to encode not just "here's a committed value" but also "this version is a tombstone" (`Delete`) or "this transaction was rolled back, don't retry it" (`Rollback`). We'll come back to the `Rollback` type when we discuss cleanup; it is one of the places TiKV improves on the original paper.

## 3. The timestamp oracle: a global, monotonic sense of time

> **Senior rule of thumb:** snapshot isolation is meaningless without a total order on transactions. Someone has to hand out timestamps that everyone agrees are monotonically increasing, or "the snapshot at `start_ts`" is not well-defined. Percolator centralizes that someone.

Every transaction in Percolator is bracketed by two timestamps: a `start_ts` it acquires at the beginning, defining its read snapshot, and a `commit_ts` it acquires at the end, defining when its writes become visible. For snapshot isolation to be correct, these timestamps must satisfy one global invariant: **if transaction A commits before transaction B starts, then A's `commit_ts` < B's `start_ts`.** Without that, B might start "before" A in timestamp order yet run "after" A in real time, and B would fail to see A's writes — a stale read that violates the snapshot guarantee. The only way to enforce that invariant across machines whose clocks drift is to ask a single authority for every timestamp.

That authority is the **timestamp oracle** (TSO). In Percolator it was a dedicated server; in TiDB it lives inside [PD, the Placement Driver](https://github.com/tikv/pd/wiki/Timestamp-Oracle), which is itself a Raft-replicated cluster. Its entire job is to hand out strictly increasing 64-bit integers. The obvious objection writes itself: a single server consulted on every transaction is a textbook scalability bottleneck and a single point of failure. Percolator's answer is that the TSO does almost no work per request and is engineered to survive failure, and the resulting throughput is genuinely enormous.

The trick is **batching plus persisting only a high watermark**. The oracle does not durably write a record for every timestamp it issues — that would cap throughput at the speed of disk. Instead it allocates a *window* of timestamps in memory, persists only the *upper bound* of that window to stable storage, and then hands out timestamps from the window at memory speed without touching disk. The original paper reports a single oracle machine serving roughly **2 million timestamps per second**. PD structures each timestamp as a physical part (the Unix time in milliseconds) plus an 18-bit logical counter, so it can mint up to `2^18 = 262144` distinct timestamps per millisecond before it has to advance the physical clock — and it persists the physical high watermark to etcd only every 3 seconds by default, serving everything in between from RAM. PD's own benchmarks report it allocating *millions* of TSOs per second this way.

![Batched client requests hit the Raft-elected TSO leader, which serves timestamps from an in-memory window at millions per second while persisting only the high watermark to etcd, so a new leader can jump past all previously issued timestamps after failover](/imgs/blogs/percolator-distributed-snapshot-isolation-6.webp)

The diagram traces the request flow. Clients don't ask for timestamps one at a time; the client library coalesces many concurrent transactions' requests into a single batched RPC to the TSO leader, so a thousand transactions starting in the same millisecond cost one round trip, not a thousand. The leader allocates from its in-memory window and replies. The window's high watermark is the only thing persisted, on a lazy 3-second cadence. And the persistence is what makes failover safe: if the TSO leader crashes, a new leader is elected (by PD's Raft group), and the new leader's first act is to read the last persisted watermark from etcd and start issuing timestamps *strictly above* it. Because the old leader never issued a timestamp above the watermark it had persisted, no timestamp can ever be reused, even across a crash. Monotonicity survives leader changes. This is the same kind of fencing logic you see in any well-built leader-leased system — and it is why a "single point of failure" is, in practice, highly available.

Here is the oracle as runnable pseudocode, which makes the batching-plus-watermark idea concrete:

```python
import threading

class TimestampOracle:
    """Hands out strictly monotonic 64-bit timestamps. Persists only a window
    high-watermark, so it survives crashes without an fsync per timestamp."""

    LOGICAL_BITS = 18                       # up to 262144 ts per physical ms
    PERSIST_INTERVAL_MS = 3000              # persist watermark every 3s
    PERSIST_AHEAD_MS = 3000                 # reserve 3s of physical time ahead

    def __init__(self, stable_store):
        self.store = stable_store           # etcd-like, survives crashes
        self.lock = threading.Lock()
        # On (re)start, jump strictly past everything ever issued.
        last = self.store.load("tso_watermark_ms") or now_ms()
        self.physical_ms = max(now_ms(), last)
        self.logical = 0
        self.persisted_ms = self.physical_ms
        self._persist(self.physical_ms + self.PERSIST_AHEAD_MS)

    def _persist(self, watermark_ms: int) -> None:
        # The ONLY durable write. Reserves a window; we may issue any ts below it.
        self.store.save("tso_watermark_ms", watermark_ms)
        self.persisted_ms = watermark_ms

    def get_timestamps(self, count: int) -> list[int]:
        with self.lock:
            # Advance physical clock if the logical counter is exhausted.
            if self.logical + count >= (1 << self.LOGICAL_BITS):
                self.physical_ms = max(self.physical_ms + 1, now_ms())
                self.logical = 0
            # If we'd cross the persisted watermark, persist a new window first.
            if self.physical_ms >= self.persisted_ms:
                self._persist(self.physical_ms + self.PERSIST_AHEAD_MS)
            base = (self.physical_ms << self.LOGICAL_BITS) | self.logical
            self.logical += count
            return [base + i for i in range(count)]
```

### Second-order optimization: the oracle is a serialization point, but a cheap one

The honest critique of the TSO is that it imposes a global serialization point: every transaction's start and commit must round-trip to one logical service. For a geo-distributed cluster spanning continents, that round trip can dominate latency — a transaction in Tokyo waiting on a TSO in Virginia pays a transpacific RTT just to learn what time it is. This is precisely the cost that CockroachDB and Spanner refuse to pay, and we'll contrast their approaches in section 9. But for a cluster within one region — the common case for OLTP — the TSO round trip is sub-millisecond and the batching makes its throughput effectively unbounded for any realistic transaction rate. The design is a deliberate bet: accept one cheap centralized dependency in exchange for a dead-simple, provably-monotonic global clock, rather than the considerable complexity of clock-uncertainty management. TiKV is even exploring a *calculated commit timestamp* optimization that derives `commit_ts` locally as `max(start_ts, region_max_read_ts) + 1` to avoid the second TSO round trip entirely for commit, which we'll touch on in the optimizations section.

## 4. The transaction protocol: prewrite, then commit the primary

> **Senior rule of thumb:** in any two-phase protocol, find the single instant that flips the outcome from "maybe" to "definitely." In Percolator that instant is one atomic write to one row — the primary. Everything before it is reversible; everything after it is just cleanup.

Now we assemble the pieces into the actual protocol. A Percolator transaction proceeds in clear phases, and the figure below is the timeline you should carry in your head.

![A transaction acquires start_ts from the TSO, buffers Set and Get calls client-side with no I/O, prewrites by locking every key and writing data at start_ts and aborting on conflict, acquires commit_ts, commits the primary atomically as the linearization point, then commits secondaries lazily in the background](/imgs/blogs/percolator-distributed-snapshot-isolation-2.webp)

**Phase 0 — start.** The client gets `start_ts` from the TSO. This fixes the read snapshot. From here on, every read the transaction does is "as of `start_ts`."

**Phase 0.5 — buffer.** As the transaction runs application logic, every `Set(key, value)` is *buffered client-side*. Nothing is written to the store yet. Reads (`Get`) go to the store at `start_ts` (resolving locks as described in section 6), but writes accumulate in a local write buffer. This is important: it means a transaction's writes hit the network only at commit time, in one coordinated burst, and it means the client — which is the only entity that knows the full write set — drives the commit. There is no server-side transaction object.

**Phase 1 — Prewrite.** When the client calls `Commit()`, prewrite begins. The client picks one key from its write set to be the **primary**; all others are **secondaries**. For every key (primary first), it issues a single atomic operation against that key's shard that does three things together: check there is no conflicting lock and no committed `write` newer than `start_ts` (if either exists, abort — this is the write-write conflict check); place a lock in the `lock` column (the primary's lock holds its own key; each secondary's lock holds a pointer back to the primary key); and write the value into the `data` column at `start_ts`. If any key's prewrite fails the conflict check, the whole transaction aborts and its already-placed locks become garbage that will be cleaned up lazily. If all prewrites succeed, the transaction is *prepared*: every key is locked, every value is staged in `data`, but nothing is yet visible to readers because no `write` pointer exists.

**Phase 2 — Commit.** The client gets `commit_ts` from the TSO (this is `> start_ts` and `>` any other transaction that committed in between). Then comes the single most important step in the whole protocol: **commit the primary.** This is one atomic operation on the primary's shard that, together, writes a `write` pointer at `(primary, commit_ts) -> start_ts` and erases the primary's lock. The instant that single-row atomic write succeeds, **the entire transaction is committed.** Not "mostly committed," not "committed pending secondaries" — committed, full stop, durably, irrevocably. The secondaries are still locked and still invisible, but their fate is now sealed: because each secondary's lock points at the primary, and the primary now shows a committed `write`, any future reader of a secondary can resolve it forward.

**Phase 3 — Commit secondaries (asynchronous).** The client then commits each secondary the same way — write its `write` pointer, erase its lock — but this is *best-effort cleanup*, not part of the atomicity guarantee. If the client crashes right after committing the primary and before touching a single secondary, the transaction is still committed and correct; the secondaries will simply be resolved lazily by whoever reads them next. This is the asymmetry that makes Percolator robust: the commit is a single atomic write, and everything after it is idempotent cleanup that any participant can perform.

Here is the protocol as runnable pseudocode. Notice how short the commit path is — the atomicity lives entirely in that one `cas`/atomic write to the primary:

```python
import time

class Transaction:
    def __init__(self, kv: KV, oracle: TimestampOracle):
        self.kv = kv
        self.oracle = oracle
        self.start_ts = oracle.get_timestamps(1)[0]   # snapshot point
        self.writes: dict[str, bytes] = {}            # client-side write buffer

    def set(self, row: str, value: bytes) -> None:
        self.writes[row] = value                      # buffered, no I/O

    def get(self, row: str) -> bytes | None:
        return snapshot_read(self.kv, row, self.start_ts)  # section 6

    def commit(self) -> bool:
        if not self.writes:
            return True                               # read-only, nothing to do
        rows = list(self.writes)
        primary, secondaries = rows[0], rows[1:]

        # ---- Phase 1: Prewrite every key. Primary first. ----
        if not self._prewrite(primary, primary):
            return False                              # conflict -> abort
        for sec in secondaries:
            if not self._prewrite(sec, primary):
                return False                          # conflict -> abort

        # ---- Phase 2: Commit the PRIMARY. THIS is the linearization point. ----
        commit_ts = self.oracle.get_timestamps(1)[0]
        if not self._commit_primary(primary, commit_ts):
            return False                              # primary lock vanished -> abort
        # The transaction is now COMMITTED, durably, regardless of what follows.

        # ---- Phase 3: Commit secondaries lazily (best-effort). ----
        for sec in secondaries:
            self._commit_secondary(sec, commit_ts)    # may be skipped if we crash
        return True

    def _prewrite(self, row: str, primary: str) -> bool:
        # ATOMIC single-row op: fail if any lock exists, or any committed write
        # is newer than our start_ts (write-write conflict). Else lock + stage data.
        if self.kv.has_lock(row) or self.kv.has_write_after(row, self.start_ts):
            return False
        self.kv.put(row, "data", self.start_ts, self.writes[row])
        self.kv.put(row, "lock", self.start_ts,
                    encode_lock(primary=primary, is_primary=(row == primary)))
        return True

    def _commit_primary(self, primary: str, commit_ts: int) -> bool:
        # ATOMIC single-row op: the lock we placed at start_ts must still be ours.
        lock = self.kv.get_lock(primary)
        if lock is None or lock.start_ts != self.start_ts:
            return False                              # someone cleaned us up
        self.kv.put(primary, "write", commit_ts, encode_write(self.start_ts, "Put"))
        self.kv.erase(primary, "lock", self.start_ts)
        return True

    def _commit_secondary(self, row: str, commit_ts: int) -> None:
        # Idempotent; safe to repeat or skip. Resolved lazily if we never run it.
        self.kv.put(row, "write", commit_ts, encode_write(self.start_ts, "Put"))
        self.kv.erase(row, "lock", self.start_ts)
```

### Second-order optimization: prewrite ordering and the conflict window

A subtlety the pseudocode glosses: prewriting the primary *first* matters for crash semantics. If the primary's lock is the source of truth, you want it placed before any secondary, so that a crash after some-but-not-all prewrites always leaves a coherent picture (either the primary lock exists and the transaction is recoverable-as-pending, or it doesn't and nothing committed). TiKV relaxes this for performance — it prewrites keys in parallel batches rather than strictly serially — but the primary is always identified up front so its lock is the anchor. The conflict window is also worth naming: between a transaction's `start_ts` and its prewrite, another transaction can commit a newer version of one of its keys; the prewrite's "no committed write newer than `start_ts`" check is exactly what catches that and forces an abort, preventing a lost update.

## 5. The primary lock is the single source of truth

> **Senior rule of thumb:** distributed atomicity always reduces to "make one thing atomic, then derive everything else from it." Percolator's one atomic thing is the primary row's commit. Internalize that and the protocol stops feeling magical.

We need to dwell on *why* committing the primary is the linearization point, because it is the single cleverest idea in Percolator and the one that lets it dispense with a coordinator entirely. The figure below makes the relationship explicit.

![A client transaction prewrites a primary lock on Bob.balance and secondary locks on Joe.balance and an index row that each point back to the primary; committing the primary at commit_ts is the linearization point that flips the whole transaction to committed, after which secondaries are rolled forward lazily](/imgs/blogs/percolator-distributed-snapshot-isolation-3.webp)

During prewrite, every secondary lock stores a pointer to the primary key. This is the wiring that makes the primary authoritative. Once prewrite finishes, the transaction's commit status is *defined* to be the state of the primary: if the primary's lock has been replaced by a committed `write` record, the transaction committed; if the primary's lock is gone with nothing committed in its place (or a `Rollback` record exists), the transaction aborted; if the primary's lock is still there, the transaction is still pending (or its client crashed mid-flight). There is exactly one place to look, and looking there is a single-key read.

This is what replaces the classic two-phase-commit coordinator. In textbook 2PC, a coordinator sends `PREPARE` to all participants, collects their votes, durably logs the decision, and then sends `COMMIT` or `ABORT`. The coordinator's durable log *is* the source of truth, and that's the problem: if the coordinator crashes after participants have voted yes but before it logs and broadcasts the decision, the participants are *blocked* — they've promised to commit, they can't unilaterally abort, and they can't proceed without hearing from a coordinator that may never come back. This is the infamous blocking problem of 2PC, covered in depth in the companion piece on [two-phase commit and how it fails](/blog/software-development/database/two-phase-commit-and-how-it-fails) and in Kleppmann's Chapter 9 treatment of distributed transactions. The coordinator is a single point of failure that can pin participant locks indefinitely.

![Classic two-phase commit blocks because a coordinator holds the transaction log and outcome, participants block until it replies, and a coordinator crash pins locks forever; Percolator has no transaction manager, makes the primary lock the durable commit record, and lets any reader resolve a crashed transaction lazily](/imgs/blogs/percolator-distributed-snapshot-isolation-4.webp)

Percolator's move, shown in the before/after above, is to put the decision *in the data* rather than in a coordinator's private log. The primary row is both a participant and the de facto coordinator log — committing it is the durable decision. There is no separate transaction manager to crash. And because the decision lives in a normal data row that every participant can read, recovery is decentralized: any transaction or reader that stumbles on a stale secondary lock can go ask the primary what happened and act accordingly, with no coordinator in the loop. The blocking problem is dissolved, not solved — there is no coordinator whose absence could block you, because the authority is a row you can always read.

Let me restate the correctness argument crisply, because it is the thing to actually remember:

1. **Atomicity** comes from the primary commit being a single atomic single-row write. Before it, no `write` pointer exists anywhere, so no reader can see any of the transaction's values — the transaction is invisible. After it, the primary is committed and every secondary is *derivable* as committed (its lock points at a now-committed primary). There is no in-between state visible to a correct reader.
2. **Durability** comes from the substrate: that single primary-commit write is a Bigtable row mutation / a TiKV Raft write, already durable and replicated by the storage layer.
3. **Isolation** comes from MVCC: readers at `read_ts < commit_ts` never see the new versions (no `write` pointer with `commit_ts <= read_ts`), readers at `read_ts >= commit_ts` see all of them.
4. **Consistency** is the application's job, as always; Percolator guarantees the transaction is all-or-nothing and snapshot-isolated, and the app guarantees the writes preserve its invariants.

The whole edifice rests on point 1: one atomic write to one row decides everything. That is the sentence to tattoo on the inside of your eyelids.

## 6. Reads: walking the columns at a snapshot, and resolving locks

> **Senior rule of thumb:** a read in an MVCC system is never just "fetch the value." It is "fetch the value *you are allowed to see*, and deal with the in-flight writers you trip over on the way." The lock-handling is half the algorithm.

A snapshot read at `start_ts` is a disciplined walk across the three columns, and it must handle the case where it encounters a lock — a sign that some other transaction is mid-write on this key. The figure below lays out the steps.

![A snapshot read uses read ts equal to start_ts, checks the lock column for any lock in the range zero to start_ts and waits or resolves via the primary if present to avoid dirty reads, then scans the write column for the newest commit_ts at or below start_ts, follows the pointer to the data column at its start_ts, and returns a value from a consistent snapshot](/imgs/blogs/percolator-distributed-snapshot-isolation-7.webp)

The read procedure, step by step:

1. **Check the lock column.** Look for a lock on this key with a timestamp in `[0, start_ts]`. A lock in that range means a transaction that *might* commit at a `commit_ts <= start_ts` is in flight — and if it does, we'd be obligated to see its write, but it's not committed yet, so we cannot safely read past it. (A lock with `start_ts > our read_ts` is from a transaction that started after our snapshot; it cannot affect what we see, so we ignore it.)
2. **If a conflicting lock exists, resolve or wait.** We do not just block forever — that would let a crashed writer freeze readers. We look at how old the lock is and, if it might be abandoned, we go to the primary and resolve the transaction (roll it forward or back, section 7). If it's a live, recent lock, we back off and retry, giving the writer time to finish.
3. **Find the newest visible version.** Scan the `write` column for the entry with the largest `commit_ts <= start_ts`. That `write` entry points at a `data` version's `start_ts`.
4. **Fetch the data.** Read `data` at the `start_ts` the `write` entry pointed to. Return it.

That two-hop indirection (write column to find the version, data column to fetch it) is the price of separating "when a value was written" from "when it became visible," and it's what makes atomic multi-key commit possible. Here's the read as code:

```python
import time

def snapshot_read(kv: KV, row: str, start_ts: int) -> bytes | None:
    while True:
        # 1. Is there a lock that could matter to our snapshot?
        lock = kv.get_lock(row)
        if lock is not None and lock.start_ts <= start_ts:
            # 2. A writer is (or was) in flight at or before our snapshot.
            if lock_is_expired(lock):
                resolve_lock(kv, row, lock)          # roll fwd/back via primary
                continue                              # re-read after resolution
            time.sleep(backoff())                     # live writer: wait it out
            continue
        # 3. Newest committed version visible at our snapshot.
        w = kv.get(row, "write", start_ts)            # largest commit_ts <= start_ts
        if w is None:
            return None                               # no committed version
        data_start_ts = decode_write(w.value).points_to
        if decode_write(w.value).kind == "Delete":
            return None                               # tombstone
        # 4. Fetch the actual bytes.
        d = kv.get_exact(row, "data", data_start_ts)
        return d.value
```

The blocking-on-a-live-lock behavior is worth a comment: snapshot *reads* not blocking *writes* is the whole point of MVCC, but a read can still block on a *concurrent committing writer* of the same key, because it genuinely cannot know yet whether that writer's commit will land at or before its snapshot. This is a narrow, key-level, short-lived wait — not the coarse read-write blocking of a lock-based system — and it is bounded by the writer's commit latency or, if the writer crashed, by lock resolution. It is the one place readers can be delayed, and it is the price of correctness, not a design flaw.

### Second-order optimization: short values inlined, and timestamp-free point reads

TiKV makes two read-path optimizations worth knowing. First, **short values are inlined into the `write` record** rather than stored separately in `data`. For small values (the common case — most rows are tiny), the value bytes ride along inside the `write` column entry, so a read finds the version *and* the value in one column-family lookup instead of two. The `data` column is only consulted for large values. Second, for a **single-key point read** that isn't part of a multi-key transaction, TiKV can skip acquiring a `start_ts` from the TSO entirely and just read the newest committed version directly, treating the read instant as the snapshot. That removes a TSO round trip from the hottest, simplest query shape — a point lookup by primary key — which in practice is an enormous fraction of OLTP traffic.

## 7. Crash recovery: lazy, reader-driven lock resolution

> **Senior rule of thumb:** in a system with no coordinator, recovery is not a background daemon's job — it is a *side effect of normal traffic*. The transaction that trips over a stale lock is the one that cleans it up. This is what "self-healing" actually means mechanically.

The hardest part of any distributed transaction protocol is what happens when a client dies mid-transaction. In Percolator, a client can crash after prewrite (leaving locks but no commit), after committing the primary (committed, but secondaries still locked), or anywhere in between. There is no coordinator running a recovery loop. So who cleans up? The answer is the next transaction or reader that touches one of the abandoned keys. The figure below traces this lazy resolution.

![After a transaction crashes mid-commit leaving locks behind, a later reader that hits a stale secondary lock follows its pointer to the primary lock and, in one check, rolls the secondary forward at commit_ts if the primary committed or rolls it back and erases the lock if the primary is gone or expired, so the reader proceeds with no coordinator needed](/imgs/blogs/percolator-distributed-snapshot-isolation-5.webp)

When transaction R encounters a lock left by transaction T (either during a read, or during R's own prewrite conflict check), R does not assume T is dead, and it does not assume T is alive. It goes and *asks the primary*. R follows the secondary lock's pointer to T's primary key and inspects it:

- **If the primary shows a committed `write` record** (the primary lock is gone and there's a `write` at some `commit_ts` pointing to T's `start_ts`): T committed. R *rolls the secondary forward* — it writes the secondary's `write` record at T's `commit_ts` (which it reads from the primary's write record) and erases the secondary's stale lock. The secondary is now committed, exactly as if T had finished its asynchronous phase 3. R then proceeds.
- **If the primary's lock is gone with nothing committed** (or there is a `Rollback` record): T aborted. R *rolls the secondary back* — it erases the secondary's stale lock (and, in TiKV, writes a `Rollback` marker). R then proceeds as if T never wrote this key.
- **If the primary's lock is still present and recent:** T may genuinely be in flight. R checks the lock's TTL. If the lock has expired (its owner is presumed dead — Percolator used Chubby ephemeral nodes / TTLs; TiKV uses a lock TTL plus a check of whether the owning transaction is still alive), R *resolves* it: it can clean up the primary itself (rolling T back by removing the primary lock), which then makes all of T's secondaries roll-back-able. If the lock is recent and not expired, R backs off and retries, because T is probably about to finish.

The critical correctness property is that resolution is driven entirely by the *primary's* state, which a single-key read reveals atomically, and that the primary can be cleaned up by *anyone* via the same atomic single-row operation. If two transactions both try to resolve T's primary at the same time, the atomic compare-and-set on the primary row ensures only one succeeds; the other re-reads and sees the now-resolved state. There is no race that produces an inconsistent outcome, because every decision funnels through one atomic write to one row.

Here's lock resolution as code:

```python
def resolve_lock(kv: KV, row: str, lock) -> None:
    """Resolve a lock we tripped over by consulting its primary. Idempotent."""
    primary = lock.primary
    pstatus = primary_status(kv, primary, lock.start_ts)

    if pstatus.kind == "COMMITTED":
        # T committed. Roll this secondary FORWARD at the primary's commit_ts.
        kv.put(row, "write", pstatus.commit_ts,
               encode_write(lock.start_ts, "Put"))
        kv.erase(row, "lock", lock.start_ts)

    elif pstatus.kind == "ABORTED":
        # T aborted (primary gone, or Rollback record). Roll this secondary BACK.
        kv.erase(row, "lock", lock.start_ts)
        kv.put(row, "write", lock.start_ts, encode_write(lock.start_ts, "Rollback"))

    else:  # pstatus.kind == "PENDING"
        if lock_is_expired(lock):
            # Presumed-dead owner. Clean the PRIMARY first (atomic), which decides
            # T's fate, then re-resolve this secondary on the next pass.
            rollback_primary(kv, primary, lock.start_ts)
        # else: live writer, caller backs off and retries.


def primary_status(kv: KV, primary: str, start_ts: int):
    plock = kv.get_lock(primary)
    if plock is not None and plock.start_ts == start_ts:
        return Status("PENDING")                      # primary still locked by T
    # No matching primary lock: look for T's outcome in the write column.
    w = kv.find_write_by_start_ts(primary, start_ts)
    if w is None:
        return Status("ABORTED")                       # nothing committed for T
    if decode_write(w.value).kind == "Rollback":
        return Status("ABORTED")
    return Status("COMMITTED", commit_ts=w.commit_ts)  # found T's commit record
```

### Second-order optimization: the Rollback record, TiKV's fix for a paper-era race

The original Percolator simply *removed* a lock when rolling a transaction back. TiKV found a subtle hazard with that. Say a slow prewrite RPC for transaction T gets delayed in the network. Meanwhile a reader presumes T dead and rolls it back by deleting its (not-yet-existent) lock — a no-op. Then T's delayed prewrite finally arrives and re-creates the lock, resurrecting a transaction that was supposed to be dead — a "zombie lock." TiKV's fix is to **write an explicit `Rollback` record in the `write` column** when aborting, rather than just erasing the lock. A late prewrite then sees the `Rollback` marker and knows to fail rather than resurrect itself. This is a great example of the gap between a paper's clean description and a production implementation's need to handle every interleaving — the kind of subtlety that, as with [Raft's commit-safety rules](/blog/software-development/database/raft-consensus-from-scratch), separates "correct on the whiteboard" from "correct under adversarial scheduling." TiKV also *collapses* old `Rollback` records during compaction so they don't accumulate, since they're only needed for a bounded window.

## 8. How distributed SQL gets ACID on a KV store: the full stack

> **Senior rule of thumb:** the SQL layer's job is to compile rich operations into the KV layer's poor vocabulary — point gets, range scans, and a Percolator transaction. If you understand that compilation, you understand the whole database.

It's worth stepping back to see how a real distributed SQL database — TiDB is the cleanest example — assembles a row-level `UPDATE` into the Percolator dance, because this is where the protocol earns its keep. A SQL statement like `UPDATE accounts SET balance = balance - 100 WHERE id = 7` becomes, underneath:

1. The SQL layer (TiDB) gets `start_ts` from the TSO when the transaction begins.
2. It encodes the table row `accounts/7` into a KV key (TiDB uses a key encoding like `t{table_id}_r{row_id}`), and possibly index entries into more KV keys.
3. It issues a snapshot `Get` at `start_ts` to read the current balance, resolving any locks it hits.
4. It computes the new value and *buffers* the write (the new row, plus any updated index entries) client-side.
5. On `COMMIT`, it runs Percolator: prewrite all the buffered keys (the row plus its index entries — these often live on *different shards*, which is exactly why you need a distributed transaction), pick a primary, commit the primary atomically, then commit secondaries.

The reason this is non-trivial is that a single logical row update fans out to multiple KV keys — the row itself and every secondary index entry — and those keys are scattered across shards by the partitioning scheme. (How rows get distributed to shards, and the tradeoffs of range vs. hash partitioning, is its own deep topic; see [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).) Without a distributed transaction, you could update the row but fail to update its index, leaving the index pointing at a stale value — a silent corruption. Percolator makes the row and all its index entries commit atomically, which is the difference between a database and a pile of inconsistent shards.

Here's the compilation sketch, showing how SQL semantics ride on the Percolator primitives:

```python
def sql_update_balance(db, account_id: int, delta: int) -> None:
    txn = Transaction(db.kv, db.oracle)               # gets start_ts

    row_key = encode_row_key(table="accounts", row_id=account_id)
    raw = txn.get(row_key)                             # snapshot read @ start_ts
    if raw is None:
        raise NotFound(account_id)
    acct = decode_row(raw)

    new_balance = acct.balance + delta
    if new_balance < 0:
        raise InsufficientFunds(account_id)            # app-level consistency
    acct.balance = new_balance

    # The row plus any affected secondary index entries — possibly on other shards.
    txn.set(row_key, encode_row(acct))
    for idx_key in affected_index_keys(acct):
        txn.set(idx_key, encode_index_entry(acct))     # buffered

    if not txn.commit():                               # Percolator 2PC underneath
        raise WriteConflict(account_id)                # caller retries
```

This is the punchline of the whole article: **distributed SQL is mostly a compiler from rich operations to a Percolator transaction over a KV store.** The optimizer, the planner, the type system — all of that is real work, but the thing that makes it *ACID across shards* is the protocol we just walked through. A row, its indexes, and rows on entirely different machines all commit or all don't, because they share one primary lock whose single atomic commit decides everything.

| SQL concept | KV / Percolator realization |
| --- | --- |
| `BEGIN` | get `start_ts` from TSO; open client-side write buffer |
| read a row / index | snapshot `Get` at `start_ts`, resolving locks |
| `UPDATE` / `INSERT` / `DELETE` | buffer encoded row + index keys client-side |
| a row + its N secondary indexes | N+1 KV keys, often across shards, in one transaction |
| `COMMIT` | prewrite all keys, commit primary atomically, commit secondaries |
| `ROLLBACK` / conflict | abort: leave locks for lazy cleanup (or write `Rollback`) |
| isolation level | snapshot isolation (TiDB also offers a serializable-ish "pessimistic" mode) |

## 9. Percolator vs. Spanner vs. CockroachDB: three answers to "what time is it?"

> **Senior rule of thumb:** every distributed transaction system is, at its core, a different answer to the question "how do we agree on the order of events across machines whose clocks disagree?" Percolator centralizes time; Spanner measures clock error and waits it out; CockroachDB tracks causality and retries. Pick the one whose costs match your deployment.

Percolator's centralized TSO is one of three influential strategies, and the contrast is the best way to understand its tradeoffs. The matrix below summarizes; then we'll walk each.

![A matrix comparing Percolator/TiKV, Spanner, and CockroachDB across clock source, isolation, the cost of establishing order, and whether there is a single point: Percolator uses a central TSO with snapshot isolation and one RPC per transaction; Spanner uses TrueTime atomic clocks with external consistency and a 7ms commit-wait; CockroachDB uses HLC with no special hardware, serializable isolation, and read restarts](/imgs/blogs/percolator-distributed-snapshot-isolation-8.webp)

**Percolator / TiKV — centralize time.** A single logical timestamp oracle hands out a total order. The upside is conceptual simplicity and provable monotonicity: there is one source of truth for time, batched and made HA, so you never reason about clock skew at all. The downside is that every transaction's start and commit consults that oracle, which adds a round trip and, for geo-distributed clusters, a potentially expensive one. Isolation is snapshot isolation. The TSO is a single logical point, mitigated by Raft-backed HA and batching to millions of timestamps per second.

**Spanner — measure clock error, then wait it out.** Google's Spanner gives each transaction a timestamp from [TrueTime](https://www.cockroachlabs.com/blog/living-without-atomic-clocks/), an API backed by GPS receivers and atomic clocks in every datacenter that returns not a single time but an *interval* `[earliest, latest]` guaranteed to contain the true time, with the uncertainty `epsilon` (roughly 7ms in practice) bounded by the hardware. To guarantee external consistency (a stronger property than SI — it's linearizability for transactions), Spanner does **commit-wait**: after assigning a commit timestamp, it sleeps until TrueTime says that timestamp is definitely in the past — about `epsilon` of waiting — before releasing locks and acknowledging. This guarantees that any transaction that starts later in real time gets a strictly later timestamp, *without* a central oracle, because the atomic-clock hardware bounds the disagreement. The cost is specialized hardware and a built-in `~7ms` commit latency floor.

**CockroachDB — track causality, retry on uncertainty.** [CockroachDB](https://www.cockroachlabs.com/docs/stable/architecture/transaction-layer) runs on commodity hardware with no atomic clocks. Each node keeps a **hybrid logical clock (HLC)** — a timestamp combining a physical component (near wall-clock) and a logical counter that advances on every message exchange, so causally related events always get increasing timestamps even when physical clocks differ. Instead of waiting out clock uncertainty like Spanner, CockroachDB *reads* and, when a read encounters a value within its clock-uncertainty window (default max offset 500ms with NTP), it **restarts** the transaction at a higher timestamp rather than risk a stale read. As the Cockroach Labs phrasing goes: Spanner always waits after writes; CockroachDB sometimes retries reads. Isolation is serializable. There is no central time source and no special hardware — the cost is occasional transaction restarts and a hard dependence on bounded clock offset (exceed the max offset and a node must remove itself to preserve safety).

| Dimension | Percolator / TiKV | Spanner | CockroachDB |
| --- | --- | --- | --- |
| Clock source | central TSO (one logical service) | TrueTime: GPS + atomic clocks | HLC: physical + logical, no hardware |
| How order is established | ask the oracle | bounded clock error (`epsilon ~7ms`) | causality tracking + max offset (default 500ms) |
| Isolation guarantee | snapshot isolation | external consistency (linearizable txns) | serializable |
| Cost paid per transaction | 1–2 TSO round trips | `~7ms` commit-wait | possible read restarts |
| Single point of failure | TSO (made HA via Raft + batching) | atomic-clock infrastructure | none |
| Hardware requirement | none special | GPS + atomic clocks per datacenter | none special |
| Best fit | single-region OLTP, simple ops | global, latency-tolerant, deep pockets | global on commodity, restart-tolerant |

The deep point underneath all three is the one Kleppmann makes in *Designing Data-Intensive Applications* Chapter 9 and that the companion post on [time, clocks, and ordering in distributed systems](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems) develops: you cannot get a total order of events across machines for free. You either centralize the clock (Percolator), buy hardware to bound its error (Spanner), or track causality and pay for the cases where it's ambiguous (CockroachDB). There is no fourth option that escapes the trade.

## 10. TiKV's optimizations: making basic Percolator fast

> **Senior rule of thumb:** the textbook protocol is the *correctness skeleton*. Production systems then shave round trips relentlessly, because in OLTP the difference between two network round trips and one is the difference between a competitive database and a slow one.

Basic Percolator pays two consensus round trips on the commit path: the prewrite (a durable Raft write to lock and stage every key) and the commit (a durable Raft write to commit the primary). For a single small transaction that's two sequential durable writes plus two TSO round trips. TiKV/TiDB attacks this with two headline optimizations, shown in the before/after below.

![Basic Percolator pays two round trips: prewrite all keys as a consensus write, then wait for commit_ts and commit the primary, so the client waits two RTTs before success; async commit fixes the outcome once all prewrites land and acks the client early while committing in the background, and single-region 1PC commits directly in the prewrite phase](/imgs/blogs/percolator-distributed-snapshot-isolation-9.webp)

**Async commit (TiDB 5.0+).** The insight is that *once all prewrites have succeeded, the transaction's outcome is already determined* — it will commit, because nothing can make a fully-prewritten transaction fail. So the client can return success to the application *immediately after the last prewrite lands*, before doing the commit round trip, and complete the commit in the background. This saves one round trip (and one consensus write's worth of latency) from the critical path. The wrinkle is crash recovery: with basic Percolator, the primary lock's presence-or-absence tells you the outcome, but with async commit a crash can leave a fully-prewritten transaction whose `commit_ts` was never written anywhere. TiKV solves this by recording, in the primary lock, a list of all the keys in the transaction, so a recovering reader can examine all of them and *derive* the commit status (if every key is prewritten, the transaction committed). Because the primary lock must list every key, async commit only applies to transactions touching a bounded number of keys (TiDB uses a threshold — on the order of tens of keys); larger transactions fall back to the standard two-phase path. TiKV can also compute the `commit_ts` locally as `max(start_ts, region_max_read_ts) + 1` in this mode, dodging the second TSO round trip.

**One-phase commit (1PC).** When a transaction's entire write set lands in a *single region* (one shard, one Raft group), there is no cross-shard atomicity to coordinate — the storage layer's own per-key atomicity already covers the whole transaction. TiKV detects this case and commits the transaction *within the prewrite phase itself*: a single Raft write that prewrites and commits at once, with no separate commit round trip and no two-phase machinery at all. For the very common pattern of a transaction that touches one row (or a few rows that happen to colocate), 1PC collapses the whole Percolator dance into one durable write. This is the fast path that makes single-row OLTP on a distributed database competitive with a single-node one.

| Optimization | What it removes | Constraint / cost |
| --- | --- | --- |
| Async commit | the commit round trip from the critical path | primary lock must list all keys; bounded txn size |
| 1PC (single region) | the entire second phase | all keys must be in one region |
| Calculated `commit_ts` | the second TSO round trip | only valid for async-commit transactions |
| Short-value inlining | the second column-family lookup on read | only for small values |
| Timestamp-free point read | the TSO round trip on a point lookup | only for single-key reads |
| Parallel prewrite | serial latency across keys | none material; standard in TiKV |

### Second-order optimization: pessimistic transactions and the SI/serializable gap

One more TiDB wrinkle deserves mention. The Percolator protocol as described is *optimistic*: it buffers writes and only checks for conflicts at commit (prewrite). Under high contention on hot rows, that means lots of transactions do all their work and then abort at commit, wasting effort. TiDB therefore also offers a *pessimistic* mode that acquires locks during the transaction (at the time of each `UPDATE`/`SELECT FOR UPDATE`) rather than only at prewrite, trading some throughput for far fewer late aborts under contention — the same optimistic-vs-pessimistic tradeoff covered in [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production). It is still snapshot-isolation-flavored underneath, layered on the same three-column MVCC store, but the locking discipline changes which transaction "wins" a conflict and when it finds out.

## Case studies from production

These are composites drawn from the public behavior, docs, and known failure modes of Percolator-lineage systems — the kinds of incidents that teach you what the protocol actually guarantees and what it quietly doesn't.

### 1. The transaction that "committed" but the secondaries didn't

A team running TiDB sees a transaction return success to the application, but a moment later a different query reads one of the affected rows and gets the *old* value, before eventually seeing the new one on retry. The wrong first hypothesis is "the commit isn't atomic — we have a consistency bug." The actual root cause is benign and by design: the client crashed (or was slow) after committing the primary but before committing one secondary, so that secondary still carried a stale lock. The reader that hit it resolved it forward by consulting the primary, which took one extra round trip — hence the brief delay and then the correct value. The lesson: in Percolator, "committed" means *the primary is committed*; secondaries are eventually consistent in their physical cleanup but logically committed the instant the primary is. Reads always see the correct value because they resolve locks; they just occasionally pay a resolution cost. Nothing is wrong. Engineers who don't internalize the primary-as-truth model misdiagnose this as corruption.

### 2. The hot primary that became the whole cluster's bottleneck

A workload does many transactions that all touch a tiny "counter" row plus a scattered set of other rows. Because the client picks the first key in its write set as the primary, and the counter happened to sort first, *every* transaction designated the same hot counter row as its primary. All commits, all conflict checks, and all lock resolutions funneled through one region's Raft group, which saturated while the rest of the cluster sat idle. The wrong fix is "add more nodes" — the bottleneck is one region, not capacity. The real fix is primary selection: don't make a contended key the primary if you can avoid it, and shard the counter (sum N sub-counters) so there's no single hot row at all. The lesson: the primary is the linearization point, so a hot primary serializes everything that shares it. Primary placement is a performance decision, not just a correctness one.

### 3. The zombie lock from a delayed prewrite

On an early Percolator-style implementation (pre-`Rollback`-record), a network hiccup delayed a prewrite RPC for several seconds. A reader presumed the owning transaction dead and "rolled it back" by deleting its lock — which didn't exist yet, so the delete was a no-op. Then the delayed prewrite landed, creating a lock for a transaction everyone else had already concluded was aborted. Worse, the original client then committed it. The result was a committed transaction that conflicting readers had treated as aborted — a genuine isolation violation. The fix, now standard in TiKV, is the explicit `Rollback` record: aborting writes a tombstone in the `write` column so a late prewrite sees "this transaction was rolled back" and refuses to proceed. The lesson: in a system where any participant can declare a transaction dead, you must make that declaration *durable and visible*, or a slow message will resurrect the corpse.

### 4. The TSO failover that paused the world for a few seconds

A PD leader (hosting the TSO) was killed during a rolling upgrade. For a few seconds, no node could get a timestamp, so every new transaction stalled — the cluster looked frozen. The wrong hypothesis is "the TSO is a SPOF and we have a single-point outage." The reality is subtler: PD re-elected a leader via Raft within its election timeout, the new leader read the persisted high watermark and resumed issuing timestamps strictly above it, and transactions unblocked — no timestamp was reused, no correctness was lost, just a brief availability blip bounded by the election timeout. The lesson: the TSO is a single *logical* service but not a single *physical* point; its availability is exactly the availability of the PD Raft group, and its failover is fast and safe by construction. Tuning the PD election timeout, and not co-locating PD with noisy neighbors, is how you shrink the blip.

### 5. The write skew that snapshot isolation cheerfully allowed

An application enforced "every project must always have at least one admin" by having each admin-removal transaction read the admin count and refuse if it would drop to zero. Two admins removed themselves simultaneously. Each transaction read a snapshot showing two admins, each concluded "fine, there's another admin," and each removed a *different* admin row. Disjoint write sets, no write-write conflict, both committed — and the project had zero admins. The wrong hypothesis is "the database lost a write." The reality is that this is textbook write skew, and snapshot isolation does not prevent it. The fix is to materialize the conflict: have each transaction also write a sentinel row (e.g., a per-project `admin_invariant` row) so the two transactions collide on a shared key and one aborts, or use TiDB's pessimistic/`SELECT FOR UPDATE` locking on the count. The lesson: know that Percolator gives you SI, not serializability, and that write-skew-prone invariants need explicit conflict materialization.

### 6. The giant transaction that blew the async-commit budget

A batch job updated tens of thousands of rows in one transaction and saw latency far worse than expected, with no async-commit speedup. The wrong hypothesis is "async commit is broken." The reality is that async commit requires the primary lock to enumerate *every* key in the transaction (so recovery can derive the outcome), which is only feasible below a key-count threshold; a 50,000-key transaction blows past it and falls back to standard two-phase commit, paying the full second round trip plus a fat primary lock. The deeper lesson is that very large transactions are an anti-pattern on Percolator-style systems regardless of async commit: they hold many locks for a long time, increasing conflict probability and lock-resolution work for everyone else. The fix is to chunk the batch into many smaller transactions (idempotently, so retries are safe), which keeps each under the async-commit threshold and shrinks the contention footprint.

### 7. The read that blocked behind a slow committer

A latency-sensitive read of a hot row intermittently spiked to tens of milliseconds. Tracing showed it was hitting a lock and backing off, then retrying. The wrong hypothesis is "MVCC reads never block, so this must be a network problem." The reality is the one caveat to "reads don't block writes": a read *can* block on a *concurrently committing* writer of the *same* key, because the read cannot yet know whether that writer's `commit_ts` will land at or before its snapshot, so it must wait for the writer to finish (or resolve the lock if the writer crashed). For a hot row under a write-heavy workload, that wait is real. The fix is workload-level — reduce contention on the single hot key (shard it, cache it, or batch the writes) — not a database tuning knob. The lesson: "snapshot reads don't block writers" is true at the granularity of *different* keys; a read and a commit racing on the *same* key still serialize briefly.

### 8. The clock that didn't matter (and the one that would have)

A team migrating from Percolator/TiDB to a different distributed database worried intensely about NTP configuration, having read horror stories about clock skew causing data loss. On TiDB they found that node clock skew was almost irrelevant to *correctness*, because transaction ordering comes entirely from the TSO, not from node wall clocks — a node could be a full second off and transactions would still order correctly. On the HLC-based system they moved to, by contrast, clock skew exceeding the configured max offset would force a node to remove itself or risk stale reads, so NTP discipline became safety-critical. The lesson is a clean illustration of the section-9 tradeoff in operational terms: Percolator's centralized TSO buys you *freedom from clock discipline as a correctness concern* (it's just a latency concern), whereas HLC-based systems make bounded clock offset a hard safety dependency. Different clock strategy, different operational burden.

### 9. The index that fell out of sync — almost

During a TiDB schema change adding a secondary index, a row update and its new index entry landed on different regions. A bug in an earlier homegrown KV-transaction layer (not TiDB itself) had previously let the row commit while the index write silently failed, producing an index that pointed at stale data — invisible until a query used the index and returned wrong rows. The reason this *can't* happen under proper Percolator is exactly the point of section 8: the row and all its index entries are keys in one transaction sharing one primary, so they commit atomically or not at all. The lesson, learned the hard way on the homegrown layer and confirmed by TiDB's correctness, is that secondary indexes are *the* reason you need cross-shard transactions in a SQL database — a row and its indexes are a multi-key invariant, and only an atomic multi-key commit preserves it. Online schema changes add their own choreography on top (see [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations)), but the base guarantee is Percolator's.

### 10. The retry storm under optimistic contention

A surge of traffic all updated the same small set of rows. With optimistic transactions, each one did its reads and buffered its writes, then collided at prewrite, aborted with a write-conflict, and retried — and the retries collided again, producing a thundering herd that made throughput *collapse* as contention rose. The wrong fix is "increase the retry limit." The real fixes are two: switch those transactions to *pessimistic* mode so they acquire locks early and queue rather than all racing to prewrite-and-abort, and reduce the contention at the data-model level. The lesson is the optimistic-vs-pessimistic tradeoff in the flesh: optimistic concurrency is excellent under low contention (no lock-holding overhead) and pathological under high contention (work done then thrown away). Percolator is optimistic by default; knowing when to flip to pessimistic locking is a production skill, not a footnote.

### 11. The 30x overhead nobody read the fine print on

A team prototyping an incremental processing pipeline benchmarked Percolator-style transactions against a single-node database doing the same logical work and was alarmed to find it roughly 30x more expensive per unit of work. The wrong conclusion is "Percolator is slow and we should abandon it." The right reading is the one the original paper states outright: Google accepted roughly a 30x per-work overhead versus a traditional DBMS *in exchange for* linear scaling across thousands of commodity machines and the ability to incrementally update rather than batch-recompute — which cut end-to-end indexing latency by about 100x. The overhead is the cost of the distribution and the per-transaction metadata (locks, write records, TSO round trips); the win is that you can throw hardware at it forever and that you stop doing giant batch jobs. The lesson: measure the *system-level* metric you actually care about (end-to-end freshness, total cost at scale), not the per-transaction microbenchmark, when deciding whether a distributed transaction protocol is worth its overhead.

### 12. The cross-region transaction that paid for the TSO twice

A globally deployed cluster put its PD/TSO in one region and ran transactions from another, far-away region. Every transaction paid a cross-region round trip for `start_ts` and another for `commit_ts`, so even a trivial single-row write took two inter-continental RTTs. The wrong fix is "make the TSO faster" — it was already fast; the latency was the speed of light. The real options are architectural: keep the TSO close to the write-heavy region, use 1PC and timestamp-free point reads to eliminate TSO round trips where possible, or — if you genuinely need globally distributed strongly-consistent writes with low latency — reconsider whether a centralized-TSO design is the right tool versus a TrueTime or HLC system that avoids the central clock. The lesson closes the loop on section 9: the centralized TSO is close to free within a region and increasingly expensive across regions, and that single fact drives much of the architectural divergence between Percolator-lineage and clock-based systems.

## When to reach for Percolator-style transactions, and when not to

Reach for a Percolator-style protocol (or a database built on one, like TiDB) when:

- **You need cross-shard ACID transactions on a sharded KV store** and you want them without a heavyweight, blocking transaction coordinator. The primary-lock trick gives you atomic multi-key commit with no separate transaction manager to operate or to crash.
- **Snapshot isolation is sufficient** for your correctness needs — most OLTP workloads are fine with SI, especially once you materialize the few write-skew-prone invariants explicitly.
- **Your cluster is primarily single-region**, so the TSO round trip is sub-millisecond and the centralized clock is close to free. This is the sweet spot Percolator was designed for and where it shines.
- **You want decentralized, self-healing recovery**: no coordinator means a crashed client never blocks the cluster; the next transaction to touch its locks cleans them up by consulting the primary.
- **You value implementation simplicity and operational predictability** over squeezing out the last microsecond. The protocol is small enough to fully understand, which matters when you're debugging it at 3 a.m.
- **You're building incremental processing** (Percolator's original purpose): transactions plus an observer/notification mechanism let you recompute only what changed instead of batch-reprocessing everything.

Skip it, or reach for something else, when:

- **You need true serializability** and can't tolerate (or don't want to manually patch) write skew. Reach for a serializable system like CockroachDB, or use TiDB's pessimistic mode and explicit conflict materialization, but go in knowing SI is the default.
- **You're globally distributed with tight latency budgets on writes.** A centralized TSO across regions pays the speed of light on every transaction; Spanner's TrueTime or CockroachDB's HLC avoid the central clock and may serve you better, at the cost of atomic-clock hardware or occasional read restarts.
- **Your workload is dominated by extreme contention on a few hot keys.** Optimistic Percolator degrades into retry storms; you'll need pessimistic locking, a different data model, or a system designed for hot-key contention.
- **You have very large transactions** (tens of thousands of keys). They blow past async-commit limits, hold many locks for a long time, and raise conflict rates for everyone. Chunk them, or use a system designed for bulk operations.
- **A single node (or a single-node database with read replicas) still fits your data and throughput.** Distributed transactions carry real overhead (the paper's ~30x per-work figure is sobering); if you don't need horizontal scale, a well-tuned Postgres or MySQL gives you serializable transactions with none of this machinery. Don't pay for distribution you don't need.

The enduring lesson of Percolator is how *little* mechanism it takes to get distributed ACID right, once you make the correct decomposition. Borrow single-key atomicity from the storage layer. Centralize time in a batched oracle. Keep multiple versions so readers and writers never collide. And — the one genuinely surprising idea — store the commit decision in the data itself, in a primary row whose single atomic flip decides the fate of an arbitrary set of keys scattered across an arbitrary number of machines. No coordinator, no blocking, no transaction manager. Just one atomic write to one row, and a discipline for reading and cleaning up around it. That idea has aged remarkably well: it is, sixteen years later, the engine inside one of the most widely deployed open-source distributed SQL databases in the world.

## Further reading

- Daniel Peng and Frank Dabek, [*Large-scale Incremental Processing Using Distributed Transactions and Notifications*](https://research.google/pubs/large-scale-incremental-processing-using-distributed-transactions-and-notifications/) (OSDI 2010) — the original Percolator paper.
- TiKV deep dive: [Percolator](https://tikv.org/deep-dive/distributed-transaction/percolator/) and [Optimized Percolator](https://tikv.org/deep-dive/distributed-transaction/optimized-percolator/) — the canonical open implementation, prewrite/commit, lock resolution, and the optimizations.
- TiDB development guide: [Transaction on TiKV](https://pingcap.github.io/tidb-dev-guide/understand-tidb/transaction-on-tikv.html) and [Async Commit](https://pingcap.github.io/tidb-dev-guide/understand-tidb/async-commit.html).
- [PD Timestamp Oracle](https://github.com/tikv/pd/wiki/Timestamp-Oracle) — how the TSO is batched, persisted, and made highly available.
- Cockroach Labs, [Living Without Atomic Clocks](https://www.cockroachlabs.com/blog/living-without-atomic-clocks/) — the Spanner/TrueTime vs. CockroachDB/HLC contrast.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 7 (snapshot isolation / MVCC) and Chapter 9 (distributed transactions, total order broadcast, the equivalence of consensus and atomic commit).
- Sibling posts: [two-phase commit and how it fails](/blog/software-development/database/two-phase-commit-and-how-it-fails), [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), [time, clocks, and ordering in distributed systems](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems).

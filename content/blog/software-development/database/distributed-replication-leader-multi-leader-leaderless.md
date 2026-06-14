---
title: "Single-Leader, Multi-Leader, and Leaderless Replication"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer tour of the three replication architectures — single-leader, multi-leader, and leaderless — and the write conflicts, quorums, version vectors, and CRDTs that decide whether your data converges or quietly loses writes."
tags:
  [
    "replication",
    "multi-leader",
    "leaderless",
    "distributed-systems",
    "quorum",
    "conflict-resolution",
    "crdt",
    "dynamo",
    "cassandra",
    "couchdb",
    "version-vectors",
    "databases",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-1.webp"
---

A previous article on this blog made a clean, almost smug promise: if you route every write through one leader, write conflicts become impossible by construction. Two clients cannot both win a race to the same row, because the leader serializes their commits into one order. That is the whole reason [single-leader replication](/blog/software-development/database/database-replication-sync-async-logical-physical) dominates production — it makes the hardest problem in distributed data go away for free.

The catch is in the word *one*. One leader is one node that must accept every write. When it is healthy, you get strong consistency and no conflicts. When it is far from your users, every write pays a cross-ocean round trip. When the network between two regions partitions, the region without the leader cannot write at all. And when the leader dies, you are one failover away from either data loss or split-brain. Single-leader replication trades write *availability* for the absence of write *conflicts*, and for a large class of systems — multi-region writes, offline-capable mobile clients, real-time collaboration, internet-scale key-value stores — that is the wrong trade.

This article is about the other two answers. **Multi-leader** replication lets more than one node accept writes; **leaderless** replication lets *any* replica accept writes. Both buy write availability, and both pay for it with the exact problem single-leader avoided: when two writes to the same key happen on two nodes that have not yet talked to each other, the system has two values and no inherent way to decide which is correct. Everything hard about these architectures — last-write-wins data loss, version vectors, CRDTs, quorums, read repair, anti-entropy, sloppy quorums, hinted handoff — is a different attempt to answer one question: *how do divergent replicas converge without losing what users meant?*

The diagram above is the mental model for the whole piece. The single axis that distinguishes the three models is *where writes are allowed*: one leader, several leaders, or any replica. That single choice cascades into everything downstream — whether conflicts can happen, whether a node's death stops writes, and who has to clean up the mess.

## The mental model: where writes are allowed decides everything

![Three replication models compared across who accepts writes, conflict behavior, write availability, conflict resolution, and typical system](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-1.webp)

Read the figure column by column. In **single-leader**, exactly one node accepts writes; conflicts are impossible; if the leader dies, unacknowledged writes can be lost; no conflict resolution is needed; this is Postgres and MySQL. In **multi-leader**, several nodes accept writes; conflicts are frequent and must be merged; the system survives a node loss; resolution is last-write-wins or version vectors; this is CouchDB, Galera, and Postgres-BDR. In **leaderless**, any replica accepts writes; conflicts are again frequent; the system survives node loss; resolution is quorum reads plus read repair; this is Amazon Dynamo, Cassandra, and Riak.

That table is the entire taxonomy, and it comes almost verbatim from the canonical text on this subject: Martin Kleppmann's *Designing Data-Intensive Applications* (DDIA), Chapter 5. Kleppmann frames the three models as a single spectrum: single-leader sacrifices write availability to avoid conflicts; multi-leader and leaderless sacrifice conflict-freedom to gain write availability. There is no fourth option that gets both, because the underlying constraint is physical: to accept a write on a node that cannot currently talk to the others is to risk that another node accepted a different write at the same instant. The [CAP and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) theorems are the formal statement of that constraint; this article is the operational one.

> Single-leader makes conflicts impossible by making one node a single point of write availability. Multi-leader and leaderless make every node a point of write availability by making conflicts inevitable. Pick which problem you would rather have.

The rest of the article is a tour of that tradeoff. We start by recapping single-leader and naming its hard limits precisely. Then multi-leader: the three use cases that justify multiple writers, the central problem of write conflicts, the four families of conflict resolution (last-write-wins, version vectors, application merge, CRDTs), and the three replication topologies with their distinct failure surfaces. Then leaderless: the Dynamo design, the quorum inequality `w + r > n`, read repair and anti-entropy for convergence, sloppy quorums and hinted handoff for availability, and a careful accounting of what quorums do *not* guarantee. Finally, a decision framework and a map of how real systems land on each model. Where this article stops — the precise mechanics of quorum tuning, anti-entropy with Merkle trees, and read repair — a [dedicated quorums, anti-entropy, and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) post picks up.

## 1. Single-leader, recapped — and where it runs out of road

**Senior rule of thumb: single-leader is the correct default for any workload that has a natural single write region and cannot tolerate silent write loss. You move off it only when one of three specific pressures forces you to.**

The single-leader model is simple enough to state in one sentence: one node (the *leader*, *primary*, or *source*) accepts all writes, turns them into an ordered change log — Postgres's Write-Ahead Log (WAL), MySQL's binlog — and ships that log to *followers* (standbys, replicas) that apply it in order and serve reads. Because the log is ordered and the leader is the only writer, every follower converges to exactly what the leader committed, modulo replication lag. The whole [synchronous-vs-asynchronous, physical-vs-logical story](/blog/software-development/database/database-replication-sync-async-logical-physical) is a detail of how that one log gets shipped.

This is the workhorse of the industry. Postgres streaming replication, MySQL replication, MongoDB replica sets, SQL Server Always On, and the relational layer of virtually every managed cloud database are single-leader. The reason is the one we keep returning to: with one writer, the commit order *is* a total order, so two writes to the same row are serialized, and "which write wins" is never ambiguous. You get linearizable writes and, with the right read routing, strong consistency, with zero conflict-resolution code. For an OLTP system of record — a bank ledger, an orders table, an inventory count — that is not a nice-to-have. It is the requirement.

So why would anyone leave? Three specific pressures, each of which single-leader handles badly:

**Pressure 1 — one write region is one round trip away from half your users.** If the leader lives in `us-east-1` and a user is in Sydney, every write pays ~200 ms of round-trip latency before the leader even begins to commit. Read replicas near the user fix *read* latency, but writes still cross the planet. A checkout flow that does six sequential writes pays over a second in pure network time. There is no way to fix this within single-leader, because by definition the one place writes are allowed is one place.

**Pressure 2 — the leader is a single point of write availability.** Followers can serve reads while the leader is down, but no one can write. During a leader failure, your write path is hard-down until failover completes. And failover is the most dangerous operation in the whole model: promote a follower that was behind, and you lose the writes it never received (an RPO breach); promote a new leader while the old one is still alive and reachable by some clients, and you have **split-brain** — two leaders both accepting writes, which you will reconcile by hand. The GitHub October 2018 incident, where a 43-second network partition triggered an automated failover that left two data centers each believing they were authoritative, is the textbook case, and it cost roughly 24 hours of degraded service to untangle.

**Pressure 3 — the write throughput ceiling is one machine.** Followers scale reads but never writes: every follower must apply the *entire* write stream, so adding followers adds replication load to the leader without adding any write capacity. When one machine's write throughput is your ceiling, single-leader is structurally incapable of going higher. (Sharding — [partitioning](/blog/software-development/database/database-partitioning-and-sharding) the data across many independent single-leader groups — is the usual escape, and it is orthogonal to the leader-count question this article is about.)

| Single-leader strength | The same strength, viewed as a limit |
| --- | --- |
| One writer, so no write conflicts | One writer, so one write region and one latency floor |
| Total commit order, linearizable writes | Total commit order requires the leader be reachable |
| Followers scale reads cheaply | Followers do nothing for write throughput |
| Failover gives high availability | Failover is the most dangerous, data-loss-prone operation |
| Simple operational mental model | Split-brain is one timeout-tuning mistake away |

Multi-leader and leaderless are the two ways out. They do not "fix" single-leader; they make a different trade. Hold onto the GitHub split-brain image, because it is the ghost that haunts the rest of this article: the moment two nodes both accept writes, you are *deliberately* doing the thing that, done accidentally, is a disaster. The difference is that multi-leader and leaderless systems are *designed* to converge afterward, and single-leader systems are not.

## 2. Multi-leader: more than one node accepts writes

**Senior rule of thumb: reach for multi-leader only when you have a concrete reason to accept writes in more than one place — multi-region write locality, offline clients, or real-time collaboration. If you cannot name the reason, you do not want multi-leader, because you are signing up for conflict resolution you do not need.**

Multi-leader replication (DDIA also calls it *master-master* or *active-active*) generalizes single-leader by allowing more than one node to accept writes. Each leader is, to its local clients, an ordinary single-leader: it commits writes locally and ships its change log to followers. The new part is that each leader is *also* a follower of the other leaders — it receives and applies their change logs too. Writes propagate asynchronously between leaders, so for a window of time, two leaders can hold different values for the same key. That window is where every hard problem lives.

Kleppmann gives three use cases that genuinely justify the complexity. They are worth memorizing, because they are also a checklist: if your situation is not one of these, you almost certainly want single-leader instead.

### 2.1 Use case: multi-datacenter, write-local

Put a leader in every region. Frankfurt users write to the Frankfurt leader; Virginia users write to the Virginia leader; the two leaders replicate to each other asynchronously across the Atlantic. Now a write is a *local* commit — single-digit milliseconds — instead of a cross-ocean round trip. Each region keeps accepting writes even if the inter-region link is slow or briefly partitioned, because each region has its own leader. Compare that to single-leader-with-remote-followers: there, a write from the non-leader region always pays the cross-region latency, and a partition cuts the non-leader region off from writing entirely.

The price is that the same record can be modified in two regions before they sync, and you now own conflict resolution. This is the canonical multi-leader deployment, and it is exactly why systems like Postgres-BDR and Galera exist.

### 2.2 Use case: offline-capable clients (local-first)

This one surprises people: a phone is a leader. An offline-first app — a calendar, a notes app, a field-service tool — must let the user create and edit data while the device has no network. The local database accepts writes immediately (it is the local leader), and when connectivity returns, it syncs bidirectionally with a server (another leader). DDIA frames this precisely: each device is a leader, the network between device and server is just a particularly unreliable inter-leader link, and the offline window is just a very long replication lag. The architecture is identical to multi-datacenter; only the timescales differ.

This is the design behind **CouchDB** and its browser-side sibling **PouchDB**. CouchDB was built, in its own words, to *embrace conflict*: each node independently accepts writes, and eventual consistency is reached through explicit conflict resolution. A PouchDB database lives inside the browser; pair it with a CouchDB server and you get automatic two-way sync, where multiple devices edit the same document independently and merge on reconnect.

### 2.3 Use case: real-time collaborative editing

When two people edit the same Google Doc or the same Figma file simultaneously, each client is effectively a leader with a very short replication interval — the unit of change can be a single keystroke, and the "leaders" sync many times a second. This is multi-leader replication operating at the speed of a UI. The conflict problem is the same as multi-datacenter, just at millisecond granularity and with a human watching, which raises the bar on resolution: a merge that silently drops a character is far more visible than one that drops a backend write.

We will return to how Figma and Google Docs actually resolve these in the conflict-resolution section, because their answers — server-authoritative last-write-wins per property, and operational transformation — sit at opposite ends of the design space.

### 2.4 The disadvantage that dominates everything: write conflicts

Here is the blunt assessment, and it is Kleppmann's: multi-leader replication is powerful, but its big downside — write conflicts — is severe enough that, in his words, "it should often be avoided" and "is rarely justified." When you can route all writes for a given record to the same leader, you sidestep conflicts (conflict *avoidance*), but the entire point of multi-leader is to *not* do that during a regional outage or offline window. So conflicts are not an edge case; they are the steady state you signed up for. The next section is the heart of the multi-leader story.

## 3. The central problem: write conflicts and how to converge

**Senior rule of thumb: every multi-leader and leaderless system must answer "what happens when two concurrent writes hit the same key?" with one of exactly four strategies. Know which one yours uses before you ship, because the default is almost always last-write-wins, and last-write-wins silently loses data.**

![Timeline of two leaders accepting concurrent writes to the same cart key, diverging, then detecting and resolving the conflict](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-2.webp)

The timeline above is the anatomy of a write conflict. At `t0`, Alice writes `cart = {book}` to the EU leader and Bob writes `cart = {pen}` to the US leader — same key, two regions, both acknowledged locally. At `t1`, each leader asynchronously ships its change to the other. At `t2`, each leader discovers it now holds two different values for one key, and — crucially — *neither write happened-before the other*. They are **concurrent**. At `t3`, something must decide: drop one (last-write-wins) or merge them (`{book, pen}`).

Before we enumerate the strategies, internalize the definition of "concurrent," because it is the single most misunderstood idea in this space.

### 3.1 Concurrent does not mean "at the same wall-clock time"

DDIA's definition: operation A *happened-before* operation B if B knows about A, depends on A, or builds upon A — for instance, B read the value A wrote. If neither A happened-before B nor B happened-before A, the two are **concurrent**. This is a *causal* definition, and it has nothing to do with physical clocks. Two writes seconds apart on the wall clock are concurrent if neither leader had seen the other's write when it accepted its own. Two writes microseconds apart are *not* concurrent if the second one read the first.

This matters because it tells you exactly when a conflict is real. If B causally followed A, there is no conflict — B is simply the newer value, and it wins. A conflict exists *only* between concurrent writes, where the system has no causal information to order them. Detecting concurrency is therefore the first job, and version vectors (below) are the mechanism that does it without trusting clocks.

Here is the happens-before logic as runnable Python, operating on the version-vector representation we will build up in §3.4:

```python
def happened_before(a: dict, b: dict) -> bool:
    """True if version vector `a` causally precedes `b`.

    a precedes b iff every counter in a is <= the matching counter in b,
    AND at least one is strictly less (b has seen strictly more history).
    """
    le_all = all(a.get(k, 0) <= b.get(k, 0) for k in set(a) | set(b))
    lt_any = any(a.get(k, 0) <  b.get(k, 0) for k in set(a) | set(b))
    return le_all and lt_any

def relation(a: dict, b: dict) -> str:
    if a == b:                 return "equal"
    if happened_before(a, b):  return "a -> b (b wins)"
    if happened_before(b, a):  return "b -> a (a wins)"
    return "CONCURRENT (conflict — must resolve)"

# Alice on EU leader, Bob on US leader, neither had seen the other:
alice = {"EU": 1, "US": 0}
bob   = {"EU": 0, "US": 1}
print(relation(alice, bob))   # CONCURRENT (conflict — must resolve)

# Later: Carol reads Alice's value, then writes — she causally follows Alice:
carol = {"EU": 1, "US": 0, "wrote_by": "carol"}  # built on alice's {EU:1}
print(relation(alice, {"EU": 2, "US": 0}))        # a -> b (b wins)
```

The key insight that the code makes concrete: concurrency is detected from the *vectors*, not from timestamps. If the vectors are comparable (one dominates the other), there is no conflict. If they are incomparable, there is.

### 3.2 The four resolution strategies, ranked by how much data they lose

![Matrix comparing last-write-wins, version vectors, and CRDTs across decision rule, data loss, clock dependence, app burden, and adopters](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-7.webp)

The matrix above is the decision space. Four strategies, in increasing order of how much they respect the user's intent:

**(1) Last-write-wins (LWW).** Attach a timestamp (or a globally unique, ordered ID) to each write; on conflict, keep the one with the highest timestamp and discard the rest. It is trivially simple, requires no coordination, and converges — every replica independently picks the same winner. It is also, in Kleppmann's words, "prone to data loss": the losing write is *gone*, even though the user who made it got an acknowledgment. Worse, it depends on synchronized clocks, and clocks are never perfectly synchronized; a write with a slightly-ahead clock can clobber a strictly-later write whose clock was slightly behind. LWW is the only built-in conflict resolution in Apache Cassandra, which is why "Cassandra silently lost my write" is a recurring war story. Use LWW only when losing a concurrent write is genuinely acceptable — caches, last-seen timestamps, telemetry — never for data you would be sad to lose.

**(2) Replica-priority (a degenerate LWW).** Give each replica a unique ID; the write from the higher-numbered replica wins. Same convergence, same data loss, even less defensible. Mentioned for completeness; rarely the right answer.

**(3) Record the conflict and merge in the application (version vectors).** Instead of throwing a write away, *keep both* and surface them on the next read as **siblings**. The application — which knows the data's semantics — decides how to merge. A shopping cart can union the items; a counter can sum the deltas; a title can present both to the user. This loses no data, but it pushes work onto the application and onto every read path, which must now handle "this key has multiple values." Riak's classic mode and Dynamo both do this. The bookkeeping that makes it correct is the version vector, covered in §3.4.

**(4) Conflict-free replicated data types (CRDTs).** Design the data type itself so that concurrent updates *always* merge to the same result, automatically and associatively, regardless of order. A grow-only set (G-Set) merges by union; a counter merges by per-replica sums; a last-writer-wins register is the LWW special case formalized. CRDTs converge with no application code at conflict time and no clocks, at the cost of modeling your data as one of these special types and carrying their metadata. Riak 2.0 shipped CRDTs (maps, sets, counters, flags) as first-class types; Figma models every property on the canvas as a CRDT-flavored register. The tradeoff: not all data fits a CRDT cleanly, and the metadata (especially for sets with deletions) can grow.

There is a fifth, more exotic family worth naming: **operational transformation (OT)**, the algorithm behind Google Docs. OT does not merge *states*; it transforms *operations* against each other so that concurrent edits, applied in different orders on different clients, still converge. "Insert 'x' at position 5" and "delete the character at position 3" must be adjusted for each other's effect on the index. OT is powerful for text but, as Figma's engineers put it, OT implementations have "a combinatorial explosion of possible states" and are "very complicated and hard to implement correctly" — which is precisely why Figma rejected it.

### 3.3 A worked example: LWW versus merge on the same conflict

Let us run the cart conflict from the timeline through both LWW and a merge, in code, so the data loss is not abstract:

```python
import time

# Two concurrent writes to the same key on two leaders.
write_eu = {"value": {"book"}, "ts": 1_000.000_2, "replica": "EU"}
write_us = {"value": {"pen"},  "ts": 1_000.000_1, "replica": "US"}
# Note: EU's clock reads 0.1 ms later than US's — but did EU's write
# actually happen later? We have no causal evidence either way.

def resolve_lww(a, b):
    winner = a if a["ts"] >= b["ts"] else b
    return winner["value"]

def resolve_merge(a, b):
    # Application knows a cart is a set: union preserves both intents.
    return a["value"] | b["value"]

print("LWW   ->", resolve_lww(write_eu, write_us))    # {'book'} — Bob's pen is GONE
print("merge ->", resolve_merge(write_eu, write_us))  # {'book', 'pen'} — both kept
```

LWW keeps `{book}` and silently discards Bob's `{pen}`, purely because EU's clock happened to read 0.1 ms later — a difference that says nothing about which write the user "meant" last. The merge keeps both, which for a shopping cart is exactly right: a customer who added a pen and a book in two sessions wants both items, not whichever session's clock won. This is the entire argument against LWW for anything that matters, in eight lines.

The catch with merge-by-union is deletion. If Bob *removes* the pen on one replica while Alice adds it on another, a naive union resurrects the pen — the classic "deleted item comes back" bug. The fix is a **tombstone**: deletion is recorded as a marker, not an absence, so the merge can see "pen was deleted at version X" and keep it deleted. CRDT sets bake tombstones in; hand-rolled merges must remember them.

### 3.4 Version vectors: detecting concurrency without a clock

**Senior rule of thumb: a version vector is just one counter per replica, advanced on each local write and merged on each sync. If you understand "each node counts its own writes and remembers the highest count it has seen from everyone else," you understand version vectors.**

A version vector (DDIA's term for the per-key, multi-replica generalization of a vector clock) attaches to each value a small map: replica → highest write-counter this value has seen from that replica. On a local write, a replica increments its own counter. On a read, the client gets the value and its version vector. On the client's next write, it sends that version vector back, telling the server "this write is based on the history I saw." The server compares vectors:

- If the incoming vector *dominates* the stored one (every counter ≥, at least one >), the write causally follows — overwrite.
- If the stored vector dominates the incoming one, the write is stale — reject or ignore.
- If neither dominates (incomparable), the writes are **concurrent** — keep both as siblings.

Here is a runnable version-vector store that merges concurrent writes into siblings:

```python
from copy import deepcopy

class VVStore:
    def __init__(self):
        self.value = []         # list of (vv, payload) siblings
    def _dominates(self, a, b):  # a >= b on every key, > on one
        keys = set(a) | set(b)
        return (all(a.get(k,0) >= b.get(k,0) for k in keys)
                and any(a.get(k,0) >  b.get(k,0) for k in keys))
    def write(self, replica, payload, ctx):
        """ctx = the version vector the client last read (causal context)."""
        new_vv = deepcopy(ctx)
        new_vv[replica] = new_vv.get(replica, 0) + 1
        # Drop every existing sibling that this write dominates (it's newer).
        survivors = [(vv, pl) for (vv, pl) in self.value
                     if not self._dominates(new_vv, vv)]
        survivors.append((new_vv, payload))
        self.value = survivors
    def read(self):
        merged_ctx = {}
        for vv, _ in self.value:
            for k, v in vv.items():
                merged_ctx[k] = max(merged_ctx.get(k, 0), v)
        return [pl for _, pl in self.value], merged_ctx

s = VVStore()
s.write("EU", {"book"}, ctx={})          # Alice, no prior context
s.write("US", {"pen"},  ctx={})          # Bob, also no prior context -> concurrent
print(s.read())   # ([{'book'}, {'pen'}], {'EU': 1, 'US': 1})  -> two siblings

# Now Carol reads both siblings, merges them in the app, and writes back:
vals, ctx = s.read()
merged = set().union(*vals)              # {'book', 'pen'}
s.write("EU", merged, ctx=ctx)           # her write dominates both siblings
print(s.read())   # ([{'book', 'pen'}], {'EU': 2, 'US': 1})  -> single converged value
```

Notice the resolution loop: the system keeps siblings until a *causally-later* write (one that read both and therefore dominates both) collapses them. Carol's merge-and-write is what finally converges the key. Version vectors do not resolve conflicts; they *detect and preserve* them faithfully so the application can resolve them without losing data. Riak's "dotted version vectors" are a refinement that keeps the vectors compact under many concurrent clients.

## 4. Multi-leader topologies: how leaders forward each other's writes

**Senior rule of thumb: in any topology other than all-to-all, a single node failure can stop replication entirely. In all-to-all, replication survives node loss but is vulnerable to causal reordering. There is no topology that is both robust to failure and immune to reordering.**

![Three multi-leader topologies — circular, star, and all-to-all — each annotated with its distinct failure mode](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-3.webp)

With two leaders, the topology is trivial: they replicate to each other. With three or more, you must decide *which leader forwards writes to which*. DDIA names three common shapes, and the figure above lays out the failure surface of each.

**Circular (a ring).** Each leader forwards writes to exactly one neighbor: `L1 → L2 → L3 → L1`. A write made at L1 travels around the ring until it returns to L1, which recognizes its own write and stops it (each write carries the IDs of the nodes it has passed through, to prevent infinite loops). This is MySQL's classic circular replication. The fatal weakness: if L2 dies, the chain breaks — L1's writes can no longer reach L3, and vice versa. One dead node halts replication for the whole ring until you manually reroute.

**Star (a hub).** One designated root node forwards writes to all others; leaves forward only to the hub. This is simpler to reason about but has an even sharper single point of failure: if the hub dies, *every* leaf is cut off from every other leaf at once. Some tree-shaped topologies generalize the star, with the same root-failure hazard.

**All-to-all (full mesh).** Every leader sends its writes directly to every other leader. There is no single point of failure: if any one node dies, the survivors still have direct links to each other and replication continues. This is what Postgres-BDR and most serious multi-leader deployments use. The price is a subtler problem: **causal reordering**. Because different network links have different latencies, a write and a *later* write that depends on it can travel different paths and arrive at a third leader out of order — the dependent write before the write it depends on. The third leader then applies an update to a row that, from its perspective, does not yet exist.

This is not hypothetical. DDIA's example: on leader 1, a row is *inserted*; on leader 3, that row is later *updated*. If the update's link to leader 2 is faster than the insert's link, leader 2 sees the update for a row it has never seen inserted. Wall-clock timestamps cannot fix this, because the reordering is causal, not temporal. The correct fix is the same machinery as conflict detection — **version vectors** — used to enforce causal delivery: a leader holds an update until it has applied everything that update causally depends on. DDIA notes, pointedly, that many multi-leader systems implement conflict detection "surprisingly badly" and do not handle this correctly out of the box.

| Topology | Single point of failure? | Causality hazard? | Real example |
| --- | --- | --- | --- |
| Circular (ring) | Yes — any node breaks the chain | Mitigated by ring order | MySQL circular replication |
| Star / tree | Yes — the hub/root | Mitigated by central ordering | Hub-and-spoke deployments |
| All-to-all (mesh) | No — survivors stay connected | Yes — links race, writes reorder | Postgres-BDR, active-active |

The takeaway is a genuine no-free-lunch: the topologies that are robust to a node dying (all-to-all) are exactly the ones exposed to reordering, and the topologies that have a natural total order (star) are exactly the ones with a single point of failure. You cannot pick a topology that escapes both; you can only pick which problem your operations team would rather handle.

## 5. Multi-leader in the wild: Galera, BDR, CouchDB, Figma

The theory becomes concrete fast when you look at how four real systems make the multi-leader trade differently — specifically, *synchronous versus asynchronous* and *how they resolve conflicts*.

### 5.1 Galera Cluster: synchronous multi-master via certification

Galera (and Percona XtraDB Cluster, built on it) is a *synchronous* multi-master plugin for MySQL/InnoDB. You can write to any node, and — unlike the asynchronous models above — Galera resolves conflicts *at commit time*, before the client gets an acknowledgment, using **certification-based replication**.

The mechanism is optimistic. A transaction runs locally on its node, assuming no conflict, until it reaches `COMMIT`. At that point, before committing, the node collects all the row changes plus the primary keys of the changed rows into a **write-set** and broadcasts it to every other node. Each node — including the originator — runs a deterministic **certification test** against the write-set, using the primary keys, to check whether it conflicts with any other in-flight transaction. If certification fails, the transaction is rolled back; if it passes, it commits everywhere. The policy is **first-committer-wins**: of two conflicting transactions, the one that reaches the certification point first commits, and the other is rejected. The application sees the rejection as a deadlock-style error and retries.

This is a fundamentally different bargain from async multi-leader: Galera does not produce siblings or lose writes silently, because it never lets two conflicting writes both commit. The cost is that commit now requires a cluster-wide round trip (so write latency is bounded by the slowest node in the quorum), and that hot rows under heavy multi-node write contention generate a stream of certification failures and retries — which Percona exposes as explicit "multi-node writing conflict" metrics precisely because they can dominate throughput.

### 5.2 Postgres-BDR: asynchronous logical multi-master

BDR (Bi-Directional Replication), originally from 2ndQuadrant, is *asynchronous* multi-master for Postgres, built on Postgres's logical decoding (the same row-level change capture used for [logical replication and CDC](/blog/software-development/database/change-data-capture-and-the-outbox-pattern)). Any node accepts writes; changes are decoded to logical row operations and shipped to the other nodes asynchronously, typically in an all-to-all topology. Because it is asynchronous, conflicts *can* both commit and must be resolved after the fact — the default is last-write-wins (with the data-loss caveats above), with hooks for custom resolution. BDR is the Postgres answer to multi-region write locality, and it inherits every property of the async multi-leader model: low local write latency, survivable regional partitions, and the standing obligation to handle conflicts.

### 5.3 CouchDB and PouchDB: built to embrace conflict

CouchDB takes the most honest position of any system here: conflicts are not an error to be hidden but a normal state to be *stored*. Every CouchDB node independently accepts writes; documents are versioned with [MVCC](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) revisions; and when replication discovers two divergent revisions of one document, it does *not* throw either away. It deterministically picks a **winning revision** that every node will independently agree on (so the cluster converges), but it *keeps the losing revisions in the document's revision tree*, exposed via the `_conflicts` field.

That design is what makes offline-first work. The application can query `_conflicts`, surface the divergence to the user ("you edited this note on two devices — keep both? merge?"), or run an automatic merge — and until it does, no data is lost. PouchDB gives you the same model inside the browser and syncs it to CouchDB. The lesson CouchDB teaches is the opposite of LWW's: when you cannot avoid conflicts, the safest thing is to *make them first-class data* rather than to resolve them by silently discarding a write.

### 5.4 Figma: server-authoritative last-writer-wins per property

Figma's multiplayer is multi-leader at UI speed, and its design is a masterclass in choosing the *simplest correct* resolution for the data. Figma explicitly rejected both operational transformation ("complicated and hard to implement correctly... combinatorial explosion of possible states") and pure decentralized CRDTs (unnecessary overhead given they run a central server). Instead, the server is the central authority, and conflict resolution is **last-writer-wins at the granularity of a single property on a single object**.

The crucial design move is the *granularity*. Figma's server tracks "the latest value that any client has sent for a given property on a given object." Two clients editing *different* properties of the same rectangle — one moves it, the other recolors it — never conflict, because the conflict unit is (object, property), not the whole object. Only when two clients set the *same* property on the *same* object does LWW kick in, and the result is always one of the values a client actually sent, never a corrupted merge. Tree structure (which object is whose child) is itself stored as a property, with cycle rejection on the server and fractional indexing for ordering. The result is a system that feels like a CRDT to users but is far simpler to build, because the server's single ordering removes the need for vector clocks and tombstone garbage collection. It is the strongest argument in this article that the right conflict-resolution strategy is the *least powerful one that fits your data*.

| System | Sync? | Write target | Conflict resolution | Data loss risk |
| --- | --- | --- | --- | --- |
| Galera / XtraDB | Synchronous | Any node | Certification, first-committer-wins | None (loser is rejected, retries) |
| Postgres-BDR | Asynchronous | Any node | LWW (default) or custom | Yes, under LWW |
| CouchDB / PouchDB | Asynchronous | Any device/server | Deterministic winner + stored `_conflicts` | None (losers retained) |
| Figma | Sync to server | Any client | Per-property LWW, server-ordered | Per-property only (by design) |

## 6. Leaderless: any replica accepts writes

**Senior rule of thumb: leaderless replication abandons the idea of a write order enforced by any node. Correctness comes not from a leader but from the arithmetic of quorums and from background convergence. If you tune the quorum wrong, you get stale reads and silent loss; the math is the whole game.**

Leaderless replication is the Amazon Dynamo design, and it is the most radical of the three: there is no leader at all. The client (or a stateless coordinator acting on its behalf) sends each write *directly to several replicas*, and reads *from several replicas*, in parallel. No node is privileged; no node imposes an order. The 2007 Dynamo paper — "Dynamo: Amazon's Highly Available Key-value Store" — is the founding document, and Cassandra and Riak are its most prominent open-source descendants.

The motivation is extreme availability and horizontal scale. Amazon's shopping cart had to accept writes even during data-center failures and network partitions — "always writable" was a hard product requirement, because a customer who cannot add to their cart is lost revenue. A single leader, with its single point of write availability and its dangerous failover, could not deliver that. Leaderless does, by never having a leader to lose.

The data lands on a set of `n` replicas chosen by **consistent hashing**: the key is hashed to a position on a ring, and the `n` nodes clockwise from that position form the key's **preference list**. (Consistent hashing is the same mechanism this blog covers under [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning); here it decides *which* `n` nodes own each key.) Replication factor `n` is typically 3 in production. Because there is no leader to ship a log, replicas drift apart whenever a write does not reach all of them — and two mechanisms, covered in §8, pull them back together: read repair and anti-entropy.

But first, the arithmetic that makes leaderless reads correct at all.

## 7. Quorums: the inequality w + r > n

![Hand-authored figure showing a write set of 2 and a read set of 2 over 3 replicas, with the overlap replica B guaranteeing a fresh read](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-4.webp)

The figure above is the entire idea of quorums in one picture. With `n = 3` replicas, suppose a write is acknowledged by `w = 2` of them (replicas A and B get the new value `v=5`; C is momentarily missed and still holds `v=4`). Later, a read queries `r = 2` of them (say B and C). Because `w + r = 4 > n = 3`, the write set `{A, B}` and the read set `{B, C}` *must* overlap — there are only 3 replicas, and two sets of size 2 cannot be disjoint. The overlap here is B, which holds the fresh `v=5`. The reader sees both `v=5` (from B) and `v=4` (from C), and uses version metadata to pick the newer one. The fresh value cannot hide.

That is the **quorum condition**, and it is the load-bearing inequality of all leaderless systems:

$$w + r > n$$

where `n` is the number of replicas for the key, `w` is the number of replicas that must acknowledge a write, and `r` is the number that must respond to a read. The argument is pure pigeonhole: if the read set and write set each leave fewer than `n` replicas untouched, they cannot both avoid the same replica, so at least one replica is in both — and that replica has the latest write. (This is the same overlap intuition that underlies the [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) discussion; quorums are how leaderless systems approximate strong reads.)

### 7.1 Tuning w and r for the workload

The inequality leaves you a dial. With `n` odd, the symmetric choice is `w = r = (n+1)/2` — for `n=3`, that is `w=r=2`, the balanced quorum. But you can skew it:

- **Read-heavy, write-cheap:** `w = n`, `r = 1`. Every write hits all replicas (slow, fragile writes), but any single replica can serve a fresh read (fast reads). Used when reads vastly outnumber writes and you want them cheap.
- **Write-heavy, read-cheap:** `w = 1`, `r = n`. A write needs only one ack (fast, available writes), but a read must consult every replica (slow reads). Used for high-ingest workloads.
- **Balanced:** `w = r = 2` at `n = 3` — the Dynamo default, and the most common production setting.

```python
def quorum_ok(w: int, r: int, n: int) -> bool:
    return w + r > n

def tolerated_failures(w: int, r: int, n: int):
    """How many node failures can a write / read still tolerate?"""
    return {"write_tolerates": n - w, "read_tolerates": n - r}

n = 3
for (w, r, name) in [(2, 2, "balanced"), (3, 1, "read-optimized"),
                     (1, 3, "write-optimized"), (1, 1, "fast-but-stale")]:
    print(f"{name:16} w={w} r={r}  strong={quorum_ok(w,r,n)}  {tolerated_failures(w,r,n)}")
# balanced         w=2 r=2  strong=True   {'write_tolerates': 1, 'read_tolerates': 1}
# read-optimized   w=3 r=1  strong=True   {'write_tolerates': 0, 'read_tolerates': 2}
# write-optimized  w=1 r=3  strong=True   {'write_tolerates': 2, 'read_tolerates': 0}
# fast-but-stale   w=1 r=1  strong=False  {'write_tolerates': 2, 'read_tolerates': 2}
```

The last row is the warning: `w=r=1` (Cassandra's `ONE`/`ONE`) gives you the most availability and the lowest latency, and `w + r = 2 ≤ 3`, so it is *not* a quorum — reads can and will return stale data. That is a legitimate choice for some workloads, but it must be a *choice*, not an accident.

Dynamo's actual default, in the paper's notation, is `(N, R, W) = (3, 2, 2)`. Cassandra exposes the same dial per query as **consistency levels**: `QUORUM` means `(n+1)/2` replicas; `LOCAL_QUORUM` means a quorum within the coordinator's local datacenter only (so cross-DC latency does not gate every operation); `ONE`, `TWO`, `ALL` set the count explicitly. The standard production recommendation is `LOCAL_QUORUM` for both reads and writes: it satisfies `w + r > n` *within* a datacenter (strong-ish, low-latency reads), is resilient to a single-node failure, and relies on background mechanisms to keep the other datacenters in sync.

### 7.2 Multi-datacenter quorums

Dynamo-style systems extend quorums across datacenters by making `n` span all of them while requiring a write to wait only for its *local* quorum. A write to a 6-replica key spread across two datacenters (3 each) with `LOCAL_QUORUM` waits for 2 local acks and lets the remaining 4 replicas (including all of the remote DC) catch up asynchronously. You get low local write latency and survive a whole-datacenter outage, at the cost of the remote DC being momentarily behind — which is exactly the right trade for geo-distributed availability.

## 8. Convergence without a leader: read repair and anti-entropy

**Senior rule of thumb: quorums make a *single read* likely to be fresh; they do nothing to keep the replicas themselves in sync over time. Two background processes do that, and a key that is written once and never read can stay diverged for a long time if only one of them is running.**

![Graph showing a stale read detected, split into hot-path read repair and background Merkle-tree anti-entropy, both converging the replicas](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-5.webp)

A quorum read can *detect* that replicas disagree — it reads `r` replicas and sees different versions. But detection is not convergence; the stale replica is still stale after the read returns. Dynamo-style systems use two complementary mechanisms to actually heal, shown in the figure above.

**Read repair (the hot path).** When a read sees stale and fresh versions among the `r` replicas it queried, the coordinator immediately writes the fresh value back to the stale replicas, before or just after returning to the client. Frequently-read keys therefore self-heal almost instantly — every read is also a repair opportunity. The weakness is built into the name: read repair only touches keys that get *read*. A key written and rarely read can stay diverged indefinitely, because nothing ever compares its replicas.

**Anti-entropy (the background sweep).** A continuous background process compares replicas pairwise and copies over whatever is missing — including the cold keys read repair never touches. Doing this naively (ship every key and compare) would saturate the network, so Dynamo uses **Merkle trees**: each replica builds a hash tree whose leaves are hashes of individual keys (or key ranges) and whose internal nodes hash their children. To compare two replicas, you compare root hashes; if they match, the replicas are identical and you transfer nothing. If they differ, you descend only into the subtrees whose hashes differ, transferring only the genuinely divergent ranges. The cost of finding the differences is logarithmic in the data size, not linear — which is what makes background reconciliation across terabytes feasible. Membership and which node owns what is propagated by a **gossip** protocol.

Together they give the system its convergence guarantee: read-hot keys converge on the read path in milliseconds; read-cold keys converge on the anti-entropy path in the background; and the union covers every key. This is the operational meaning of **eventual consistency** — not "consistency happens by magic" but "two named, bounded processes guarantee that, absent new writes, all replicas converge." The deeper mechanics of both — repair strategies, anti-entropy scheduling, Merkle-tree granularity — are the subject of the dedicated [quorums, anti-entropy, and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) post.

## 9. Sloppy quorums and hinted handoff: availability past the quorum

![Graph showing a write to a key whose home node is down, accepted on a stand-in node with a hint, then handed off when the home node recovers](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-6.webp)

A strict quorum has a failure mode: if enough of the `n` *home* replicas (the preference-list nodes) are down, the write cannot reach `w` of them and must be *rejected*. For Amazon's "always writable" cart, rejecting a write was unacceptable. The fix is the **sloppy quorum**.

The figure traces it. A write to key `k` needs `w = 2` acks. Its home node C (one of the preference-list nodes A, B, C) is down. A *strict* quorum would now reject the write. A *sloppy* quorum instead accepts it on the next reachable node `D` — a node that is *not* on the key's preference list — letting the write succeed by landing on `n` reachable nodes rather than the `n` *home* nodes. D stores the value along with a **hint**: a note that this data really belongs to C. When C recovers and rejoins the ring, **hinted handoff** kicks in: D replays the held writes to C and then deletes its temporary copy and the hint.

This is the lever that converts "the right replicas are down, so we fail" into "any `w` replicas are up, so we succeed." Cassandra implements the same pattern: when a replica is unreachable, the coordinator stores a hint and replays it on recovery. It dramatically raises write availability during partitions and node failures.

But it weakens the very guarantee quorums exist to provide, and this is the subtle, dangerous part. With a sloppy quorum, the `w` nodes that acknowledged a write may be *stand-ins*, not the home nodes. A later read of the home nodes — even a strict `r`-node read — can entirely miss the stand-in's value, because the stand-in is not on the preference list the reader queries. So `w + r > n` *no longer guarantees overlap*. DDIA is explicit: sloppy quorums increase availability at the cost of durability and the overlap guarantee, even when the inequality holds on paper. Sloppy quorums are an availability mechanism, not a consistency one, and conflating the two is how teams convince themselves they have stronger guarantees than they do.

## 10. What quorums do NOT guarantee

**Senior rule of thumb: a quorum makes stale reads *unlikely*, not *impossible*. It is not linearizability, and treating it as such is the single most common leaderless-systems mistake. Know the edge cases before you build something that assumes they cannot happen.**

This is the section to tattoo on the inside of your eyelids, because `w + r > n` is so clean that engineers reach for it as if it were a strong-consistency guarantee. It is not. DDIA enumerates the ways a quorum read can still return a stale or wrong value even with the inequality satisfied:

**Sloppy quorums break the overlap.** As §9 explained, when writes land on stand-ins, the read and write sets need not intersect. The pigeonhole argument assumes both sets are drawn from the *same* `n` home nodes; sloppy quorums violate that premise.

**LWW plus clock skew loses writes.** If the system resolves concurrent writes by last-write-wins on timestamps (Cassandra's default), then two genuinely concurrent writes are not "kept as siblings" — one is *dropped*, chosen by clock, and a clock that is even slightly skewed can drop the write you wanted to keep. Quorum or not, the write is gone.

**Concurrent write-during-read is ambiguous.** If a write is in flight — acknowledged by some replicas but not yet all — and a read happens at that moment, the read may see the new value on one replica and the old value on another, with no defined rule for which it returns. The result can legitimately be either, and two reads in quick succession can return different values (a non-monotonic read).

**Partial write failure does not roll back.** If a write succeeds on some replicas but fails to reach `w` of them, those partial writes are *not* undone. The value sits on a subset of replicas, and a subsequent read might return it even though the write "failed." There is no transaction to abort.

**Read-during-concurrent-write may not be repaired.** A read that observes divergence triggers read repair — but if writes are concurrent and resolution is sibling-based, the read may return one sibling while another still exists, and which one you get is not deterministic.

The honest summary: **leaderless quorums give you eventual consistency with a high probability of read freshness, not linearizability.** If you need linearizability — a unique-username check, a "claim this seat" operation, anything where two clients must not both think they won — quorums are the wrong tool. You need either single-leader with synchronous commit, or a leaderless system's *light-weight transactions* (Cassandra's `LWT`, which use a Paxos round and are far slower than ordinary quorum writes). The PACELC framing is exact here: leaderless systems are tuned for availability and low latency, and you pay in consistency, *always*, not just during partitions. For the full treatment of where linearizable, sequential, causal, and eventual sit relative to one another, see [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

| Quorum is often assumed to guarantee | What it actually guarantees |
| --- | --- |
| Every read sees the latest write | A read *probably* sees it; sloppy quorums and concurrency break this |
| Concurrent writes are preserved | Only with version vectors/CRDTs; LWW drops one |
| Reads are monotonic | No — a read during an in-flight write can go backward |
| Writes are atomic across replicas | No — partial writes are not rolled back |
| Linearizability | No — eventual consistency with high freshness probability |

## 11. The decision framework: which model for which workload

![Two-by-two grid mapping single-leader, synchronous multi-leader, offline multi-leader, and leaderless to their ideal workloads and systems](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-9.webp)

The grid above is the playbook, organized by your *dominant constraint*. Find the quadrant your workload sits in, and the model follows.

**Single-leader — when you have one natural write region and cannot tolerate silent loss.** Systems of record: ledgers, orders, inventory, anything where a lost write is a correctness bug, not a UX annoyance. You get strong consistency and zero conflict code. You accept that writes go to one region and that failover is your most dangerous operation. This is the right default; you should be able to articulate *why* you are leaving it. Postgres, MySQL.

**Synchronous multi-leader — when you need multi-region write locality and can pay for cluster-wide commit coordination.** Each region writes locally with single-digit-millisecond latency, and certification (Galera) or consensus prevents conflicting commits from both succeeding, so you get no silent loss. You pay with commit latency bounded by the slowest quorum member and with retry storms on hot, multi-region-contended rows. Galera, Postgres-BDR (sync mode).

**Offline / local-first multi-leader — when clients must work disconnected and merge on reconnect.** Mobile apps, field tools, collaborative documents. Each device is a leader; the offline window is just a long replication lag. Conflicts are inevitable and you resolve them with CRDTs, stored conflicts (CouchDB's `_conflicts`), or per-property LWW (Figma). The defining requirement is that the user keeps working with no network and nothing is lost on merge. CouchDB, PouchDB, CRDT-based stacks, Figma.

**Leaderless — when you need always-on writes and internet-scale horizontal scaling, and you can live with eventual consistency.** Shopping carts, activity feeds, time-series, IoT ingest, session stores. Any replica accepts writes; you tune `w` and `r` per workload; you survive node and even datacenter failures with sloppy quorums. You give up linearizability and accept the stale-read edge cases of §10. Cassandra, Riak, DynamoDB.

The single most useful question to ask, before any of this: **can I route all writes for a given record to one place?** If yes — if your data partitions cleanly so that each record has a natural home — then single-leader-per-partition (sharding) gives you strong consistency *and* horizontal scale, and you avoid the entire conflict-resolution tax. Multi-leader and leaderless earn their complexity only when you *cannot* do that: when writes genuinely originate in multiple places that must each stay available independently. If you can name that reason, choose accordingly. If you cannot, you want single-leader.

## 12. How real systems map onto the three models

![Matrix mapping single-leader SQL, sync and async multi-leader, local-first sync, and leaderless Dynamo-style systems to their write paths and conflict handling](/imgs/blogs/distributed-replication-leader-multi-leader-leaderless-8.webp)

The matrix above is the cheat sheet for "what model is *this* database, really?" — because the marketing rarely says it plainly.

- **Single-leader SQL** — Postgres, MySQL. One primary, WAL/binlog fanout, conflicts impossible because writes are serialized. The thing you reach for unless you have a reason not to.
- **Synchronous multi-leader** — Galera, Percona XtraDB. Write to any node, certify the write-set at commit, first-committer-wins. No silent loss; pays cluster-round-trip commit latency.
- **Asynchronous multi-leader** — Postgres-BDR. Write to any node, logical-decode and ship async, LWW or custom resolution. Low local latency; conflicts can both commit.
- **Local-first sync** — CouchDB, PouchDB. Device and server are both leaders; deterministic winner plus retained `_conflicts`. Built to embrace conflict.
- **Leaderless Dynamo-style** — Cassandra, Riak, DynamoDB. Coordinator writes to `n` replicas, tunable `w`/`r`, resolution by LWW (Cassandra), version vectors (Riak classic), or CRDTs (Riak 2.0). Always-on, eventually consistent.

The names differ, the trades do not. Every one of these systems is a point on the single spectrum the first figure drew: how many nodes accept writes, and therefore who has to clean up after them.

## Case studies from production

### 1. The Cassandra write that vanished at midnight

A team ran Cassandra with `n=3` and, for write latency, `w=ONE`/`r=ONE`. A user updated their profile bio twice in quick succession from two app servers. Because `w + r = 2 ≤ 3`, the two writes landed on disjoint single replicas; LWW resolved them by timestamp; and the two app servers' clocks were ~40 ms apart due to NTP drift. The *earlier* edit, written by the server with the faster clock, won — and the user's *later* edit was silently discarded. The symptom ("my change reverted") looked like a UI bug; the wrong first hypothesis was caching; the root cause was sub-quorum consistency plus clock skew under LWW. The fix was `LOCAL_QUORUM`/`LOCAL_QUORUM` (restoring `w + r > n`) and, for the genuinely concurrent case, moving the bio to an explicit "last edit by user action" model. The lesson: `ONE`/`ONE` is not a quorum, and Cassandra's LWW will lose the write your user actually meant.

### 2. The MySQL ring that stopped after one reboot

A three-node MySQL circular replication setup (`L1 → L2 → L3 → L1`) ran fine for a year. Then L2 was rebooted for a kernel patch during a low-traffic window. Replication did not resume on its own: with L2 down, L1's writes could not flow to L3 and vice versa, and even after L2 came back, the relay positions had to be repointed by hand. For six hours, L1 and L3 silently diverged on rows written during the gap. The wrong hypothesis was "replication will self-heal on reboot"; the root cause was that circular topology has no redundancy — any node is a single point of failure for the chain. The fix was migrating to an all-to-all topology (and, later, to a single-leader-plus-followers design, since the deployment never actually needed multi-region writes). The lesson from §4: only all-to-all survives a node loss, and a ring you "set and forgot" is one reboot from a manual recovery.

### 3. The Galera retry storm on the sequence table

A team moved to Galera for active-active multi-region writes and put their application's monotonic ID generator — a single hot row incremented on every insert — in the cluster. Under load, every node tried to increment the same row, certification rejected all but the first committer, and the application drowned in deadlock-style retries. Throughput on that table *fell* relative to the old single-leader setup. The wrong hypothesis was "Galera is slow"; the root cause was that certification-based replication punishes multi-node contention on the same rows — first-committer-wins means everyone else loses and retries. The fix was to stop generating IDs from a shared row (UUIDs, or per-node ID ranges) so writes no longer contended. The lesson: synchronous multi-master is excellent for *partitionable* write load and pathological for *hot-row* write load.

### 4. The shopping cart that resurrected deleted items

A Dynamo-style store used merge-by-union to resolve concurrent cart writes — the textbook "keep both" strategy. Users complained that items they had *removed* kept reappearing. The cause: on one replica the user removed an item; on another, concurrently, a stale client re-added it; the union merge brought the removed item back, because a plain union has no way to represent "this was deleted." The wrong hypothesis was a UI double-submit; the root cause was union-merge without tombstones (the deletion-resurrection bug from §3.3). The fix was tombstones — recording deletions as explicit markers carried in the merge — which is exactly what a proper CRDT set does internally. The lesson: merge-by-union is only safe if deletions are first-class; otherwise "keep both" silently means "keep deleted things too."

### 5. The all-to-all cluster that updated a row before it existed

A Postgres-BDR active-active deployment across three regions logged sporadic "update on nonexistent row" warnings. The pattern: a row inserted in region A and immediately updated in region B would, on region C, occasionally have the *update* arrive before the *insert*, because the two changes traveled different inter-region links with different latencies. The wrong hypothesis was application bugs ordering operations incorrectly; the root cause was causal reordering in an all-to-all topology (§4), where links race and dependent writes can overtake the writes they depend on. The fix involved enforcing causal delivery (holding updates until their causal dependencies arrived) and, for the worst-affected tables, routing all writes for a given key to one region to sidestep the race. The lesson: all-to-all has no single point of failure but is exposed to reordering, and "the database will order my writes" is an assumption that does not survive multiple leaders.

### 6. The sloppy quorum that "lost" a confirmed write

A team relied on `w + r > n` as a strong-read guarantee and could not understand how a write that returned success was missing from a subsequent quorum read. Investigation showed the write had happened during a partition that took two of the three home replicas offline; a sloppy quorum had accepted it on two *stand-in* nodes outside the preference list. The later read queried the (now-recovered) home replicas, which had not yet received the hinted handoff, and the read-set/write-set did not overlap. The wrong hypothesis was data corruption; the root cause was the §9 fact that sloppy quorums break the overlap guarantee even when the inequality holds. The fix was to disable sloppy quorums for the small set of keys requiring read-after-write (accepting reduced availability for them) and to rely on hinted handoff completing for the rest. The lesson: sloppy quorum is an availability feature, not a consistency one, and the inequality lies when stand-ins are involved.

### 7. The offline app that overwrote a week of edits

A field-service mobile app synced to a CouchDB backend. A technician worked offline for a week, accumulating dozens of local edits; meanwhile, the office edited some of the same records on the server. On reconnect, the app's naive sync logic always preferred the *device's* version, silently overwriting a week of office edits. The wrong hypothesis was a replication bug; the root cause was that the app ignored CouchDB's `_conflicts` field — CouchDB had *correctly* retained both revisions, but the application never looked. The fix was to read `_conflicts`, surface divergent records to the user for resolution, and auto-merge the safely-mergeable ones. The lesson from §5.3: CouchDB does the hard part (it never loses a conflicting revision), but it is the *application's* job to resolve conflicts, and "last sync wins" is just LWW with a friendlier name.

### 8. The Figma-style editor where moving and recoloring fought each other

A team building a collaborative diagram editor implemented whole-object LWW: any change to an object replaced the entire object with the last writer's version. Two users editing the same shape — one dragging it, one changing its color — kept clobbering each other, because each sent the *whole object* and the last one won, discarding the other's change entirely. The wrong hypothesis was a sync race that needed locking; the root cause was conflict granularity that was too coarse. The fix was Figma's approach (§5.4): resolve LWW *per property*, so a move and a recolor on the same object are independent and both survive; only same-property edits actually conflict. The lesson: with LWW, the granularity of the conflict unit is a design decision as important as the resolution rule itself — coarse granularity manufactures conflicts that finer granularity would never have.

### 9. The Cassandra LWT that quietly halved throughput

To prevent two users from claiming the same username, a team switched the username-insert from a normal quorum write to a Cassandra lightweight transaction (`IF NOT EXISTS`). It worked correctly — but signup latency tripled and the cluster's write throughput on that path fell sharply. The wrong hypothesis was a capacity problem needing more nodes; the root cause was that LWTs use a Paxos round (multiple round trips and a separate consistency path) and are far more expensive than ordinary quorum writes. The fix was to keep the LWT only for the genuinely-must-be-unique username check and use ordinary quorum writes everywhere else, rather than "upgrading" the whole write path to LWT for safety. The lesson from §10: quorums are not linearizable, so when you *truly* need linearizability you must pay for it explicitly — and you should pay for it on the smallest possible surface.

### 10. The "balanced" quorum that could not tolerate one node down

A three-node Cassandra cluster ran `QUORUM`/`QUORUM` (`w=r=2`, `n=3`). During a routine single-node maintenance, *some* operations began failing. The team expected `QUORUM` to tolerate a node loss — and reads/writes mostly did — but a few keys whose two surviving replicas were exactly the maintenance node plus one slow node intermittently could not assemble two responses in time, tripping timeouts. The wrong hypothesis was that `QUORUM` cannot survive any failure; the root cause was that `w=r=2` at `n=3` tolerates *exactly one* unavailable replica per key, with zero margin — one slow node on top of the one down node is enough to fail. The fix was raising the replication factor to `n=5` (so `QUORUM`=3 tolerates two failures) for the latency-critical keyspace. The lesson: `tolerated_failures = n - w` (and `n - r`); a balanced quorum at `n=3` has a failure budget of one, and "one down plus one slow" exhausts it.

### 11. The Riak deployment that drowned in siblings

A team adopted Riak in its classic mode, which keeps concurrent writes as siblings (version vectors, §3.4) rather than dropping them via LWW. They never wrote the application code to *resolve* those siblings on read. Over months, a handful of frequently-updated keys — user-preference blobs written from multiple devices — accumulated dozens, then hundreds, of siblings each, because every concurrent write added one and nothing ever collapsed them. Read latency on those keys climbed as the object ballooned, and eventually a single object grew large enough to cause memory pressure on its nodes. The wrong hypothesis was a memory leak in Riak; the root cause was unresolved sibling explosion — version vectors faithfully preserve every concurrent write, and if the application never merges them, they grow without bound. The fix was a read-time merge function (union the preferences, then write the merged value back with the combined causal context, exactly like Carol's collapse in the §3.4 code) plus enabling Riak's CRDT map type for new keys so merges happened automatically. The lesson: "keep both" is only half a strategy; without an application-level resolve step, version vectors trade silent data loss for unbounded sibling growth, and you must choose to actually converge.

### 12. The DynamoDB global table whose two regions disagreed for minutes

A team used a DynamoDB global table (active-active across two AWS regions, a managed leaderless multi-region store) and assumed a write in one region would be immediately visible in the other. A workflow wrote a record in `us-east-1` and, milliseconds later, a downstream consumer in `eu-west-1` read it — and intermittently got "not found," then the old value, before finally seeing the new one. The wrong hypothesis was a bug in the consumer's retry logic; the root cause was that cross-region replication in a leaderless global table is asynchronous and eventually consistent, with last-writer-wins resolving any cross-region conflict — there is no synchronous cross-region quorum, so the remote region is simply behind for a replication window. The fix was to stop treating the two regions as one synchronous store: the workflow either pinned read-after-write traffic to the originating region (read your own writes locally) or tolerated the eventual-consistency window explicitly with bounded retries. The lesson ties the whole article together: a managed, leaderless, multi-region database is still leaderless and still eventually consistent across regions — the cloud branding does not buy you linearizability, and §10's caveats apply to DynamoDB exactly as they apply to Cassandra.

## When to reach for each model — and when not to

### Reach for single-leader when:
- Your data has a natural single write region, or partitions cleanly so each record has one home (single-leader-per-shard).
- A lost write is a correctness bug, not a UX annoyance — ledgers, orders, inventory, anything you would page someone over.
- You want strong consistency and zero conflict-resolution code, and you can tolerate writes going to one region.
- You can invest in a *safe* failover path (fencing, proper quorum-based promotion) to bound the split-brain risk.

### Reach for multi-leader when:
- You genuinely need to accept writes in more than one region with local latency, and you can pay for either synchronous certification (Galera/BDR-sync, no silent loss) or async conflict resolution (BDR-async, LWW/custom).
- Clients must work offline and merge on reconnect — local-first apps — and you will build real conflict handling (CouchDB `_conflicts`, CRDTs, or per-property LWW).
- You are building real-time collaboration, and you will pick the *least powerful* resolution that fits (per-property LWW like Figma before reaching for OT or full CRDTs).

### Reach for leaderless when:
- "Always writable" is a hard requirement that must survive node and datacenter failures — carts, feeds, ingest, sessions.
- You need internet-scale horizontal write scaling and can express correctness through tunable `w`/`r` quorums plus background convergence.
- Eventual consistency with high read-freshness probability is acceptable, and you have audited the §10 edge cases for the few operations that need more.

### Skip multi-leader and leaderless when:
- You cannot name the specific reason you need multiple write nodes. The default answer is single-leader; "for scale" without a partition story is not a reason.
- A single lost write is unacceptable and you would deploy LWW anyway — LWW *will* lose writes, so do not pretend otherwise.
- You need linearizability (uniqueness checks, "claim this resource," any mutual-exclusion). Use single-leader or pay explicitly for LWTs/consensus on the smallest possible surface; do not assume a quorum gives it to you.
- You would treat `w + r > n` as a strong-consistency guarantee without auditing sloppy quorums, clock skew, and concurrent-write edge cases. If you have not read §10, you are not ready to operate a leaderless store.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 5 (Replication) — the canonical treatment of all three models, in print and as the [O'Reilly chapter](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/ch05.html).
- DeCandia et al., ["Dynamo: Amazon's Highly Available Key-value Store"](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/dynamo.pdf) (SOSP 2007) — the founding leaderless design: consistent hashing, `(N,R,W)`, sloppy quorum, hinted handoff, vector clocks, Merkle-tree anti-entropy.
- [Figma's multiplayer technology](https://www.figma.com/blog/how-figmas-multiplayer-technology-works/) — server-authoritative per-property last-writer-wins, and why they rejected OT and pure CRDTs.
- [CouchDB replication and conflict model](https://docs.couchdb.org/en/stable/replication/conflicts.html) and [PouchDB conflicts guide](https://pouchdb.com/guides/conflicts.html) — multi-leader offline sync and `_conflicts`.
- [Galera certification-based replication](https://mariadb.com/docs/galera-cluster/galera-architecture/certification-based-replication) — synchronous multi-master and first-committer-wins.
- [Cassandra consistency levels](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html) — `QUORUM`, `LOCAL_QUORUM`, and tunable consistency in a Dynamo descendant.
- On this blog: [single-leader replication](/blog/software-development/database/database-replication-sync-async-logical-physical), [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual), [CAP and PACELC](/blog/software-development/database/cap-theorem-and-pacelc), and the deeper [quorums, anti-entropy, and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) follow-up.

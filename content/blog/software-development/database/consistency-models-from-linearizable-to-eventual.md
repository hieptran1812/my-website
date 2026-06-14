---
title: "Consistency Models: From Linearizability to Eventual Consistency"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A practical tour of the consistency spectrum — linearizability, sequential, causal, session guarantees, and eventual consistency — built model-by-model with timelines, worked examples, and a per-feature playbook."
tags:
  [
    "consistency-models",
    "linearizability",
    "causal-consistency",
    "eventual-consistency",
    "distributed-systems",
    "crdt",
    "vector-clocks",
    "session-guarantees",
    "databases",
    "replication",
    "system-design",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 52
---

There is a bug report that lands on the desk of every distributed-systems engineer eventually, and it always reads the same way. A user posts a comment, the page refreshes, and the comment is gone. Or they upload a new profile picture, see it for a second, then their next pageview shows the old one. Or two people add items to a shared shopping cart and one item silently disappears. The application code is correct. The database "worked." And yet the system did something the user finds insane, because at no single point in real time did anything go wrong on any one machine — the insanity emerged from the *gaps between* the machines.

Every one of those bugs is a consistency-model bug. The team shipped a system that makes a weaker guarantee than the feature actually needed, usually without anyone deciding to. They reached for a replicated, geo-distributed datastore because it was fast and available, and they got fast and available, and the price was that the system is now allowed to return answers that look like time travel. The whole discipline of consistency models is the precise, formal vocabulary for *what a read is allowed to return* — and therefore for which of those time-travel bugs your architecture has quietly made legal.

This article is a tour of that vocabulary, from the strongest guarantee you can buy to the weakest one worth having. We will build each model on the one above it: **linearizability** (the gold standard — every operation looks atomic and respects real time), **sequential consistency** (drop real time, keep a global order), **causal consistency** (keep only the order that cause-and-effect demands), **session guarantees** (a per-client slice of sanity), and **eventual consistency** (replicas converge if you stop writing — and the conflict-resolution machinery, from last-write-wins to CRDTs, that decides what "converge" even means). The diagram above — the consistency ladder — is the mental model for the entire piece. Everything else is a tour of one rung at a time, what each rung forbids, what it costs, and which features in your product genuinely need it.

![The consistency ladder from strict serializable down to strong eventual consistency, where each step trades a real-time or ordering guarantee for availability](/imgs/blogs/consistency-models-from-linearizable-to-eventual-1.webp)

The single most important fact about this ladder, the one that organizes everything below, comes straight from [Jepsen's consistency hierarchy](https://jepsen.io/consistency/models): the models form an *implication order*. When model X "implies" model Y, every history that is legal under X is also legal under Y — X is *stronger*, Y is *weaker*. For single-object operations the chain is: strict serializable implies linearizable implies sequential implies causal, and causal implies the session guarantees, which imply nothing below them except convergence. Strength is restriction. A stronger model forbids more behaviors, gives the programmer fewer surprises, and — this is the catch that runs through the whole article — costs more in latency and availability, because forbidding surprises means coordinating across machines, and coordination is exactly what a network partition prevents.

> A consistency model is not a feature you turn on. It is a contract about which reads are legal. Every system already has one, whether or not anyone chose it on purpose.

## Why "the database worked" is not an answer

Before we climb the ladder, it is worth being brutally precise about the mismatch most teams carry around in their heads, because it is the source of the bug reports above. People reason about a distributed datastore as if it were a single variable in a single-threaded program. It is not. It is a pile of replicas, each lagging the others by some unknown amount, gossiping updates over a network that can delay, reorder, or drop them. The "value of x" is not a fact; it is a *per-replica, per-moment* opinion.

| Assumption | The naive mental model | The distributed reality |
| --- | --- | --- |
| "I wrote x=1, so x is 1." | The write happened; the value is now 1 everywhere. | The write happened *on the replica you talked to*. Other replicas still hold 0 until replication catches up — the "inconsistency window." |
| "My next read sees my write." | Reads and writes hit the same place. | A load balancer may route your read to a different, lagging replica. Without read-your-writes, your own write is invisible to you. |
| "Two reads in a row are monotone." | Time moves forward; so do values. | Read 1 hits a fresh replica, read 2 hits a stale one; the value appears to move *backward*. Without monotonic reads, this is legal. |
| "If A causes B, everyone sees A before B." | Cause precedes effect, obviously. | Replication delivers A and B over independent channels. A replica can apply B before A arrives. Without causal consistency, a reply can appear before the post it answers. |
| "Concurrent writes are serialized somehow." | The database picks a winner sensibly. | Under last-write-wins, "sensibly" means "whichever wall clock was further ahead" — and the loser is *silently deleted*, lost-update style. |

Notice that none of these are bugs in the database. Each one is a *legal behavior* of some consistency model, and the model is the one your stack defaults to. The job is to know which behaviors each model permits, decide which ones your feature can tolerate, and pay for exactly the strength you need — no more (it is slow and fragile) and no less (it is wrong). This is the same discipline as picking a transaction isolation level, and the two are deeply related, as we will see; the standard reference for both halves is Martin Kleppmann's *Designing Data-Intensive Applications* (DDIA), whose Chapter 5 covers the replication-lag session guarantees and Chapter 9 covers linearizability and causality. We will lean on both chapters heavily, in our own words.

## 1. Linearizability: the gold standard for a single object

> **Senior rule of thumb:** linearizability is the model that lets you stop thinking about replication entirely — the system behaves as if there is exactly one copy of the data and every operation hits it instantaneously. You pay for that illusion with a cross-replica round trip on every operation, and with unavailability the moment the network partitions.

Linearizability is the strongest single-object model, and it is the one most people *mean* when they say "strongly consistent" without specifying. The definition, in [Jepsen's phrasing](https://jepsen.io/consistency/models/linearizable), is that "every operation appears to take place atomically, in some order, consistent with the real-time ordering of those operations." Two clauses do all the work. *Atomically, in some order*: there is a single total order of operations, and each operation takes effect at one instant — a single point somewhere between when the client called it and when it returned. *Consistent with real-time ordering*: if operation A finished before operation B started (in actual wall-clock real time, as observed by any outside party), then A comes before B in that order. The formalization Herlihy and Wing gave in 1990, refined by Viotti and Vukolić, is three constraints: a single total order exists, it respects real-time precedence, and each operation obeys the object's single-threaded semantics (a read returns the value of the most recent write in the order).

### What this lets a read return — the timeline

The cleanest way to internalize linearizability is to fix a single register `x` and ask, operation by operation, *what is this read allowed to return?* The figure traces exactly that.

![A timeline of a single register where a read overlapping the in-flight write may return old or new, but any read that starts after the write returns must see the new value](/imgs/blogs/consistency-models-from-linearizable-to-eventual-2.webp)

Walk it left to right. At t=0 a write sets `x=0` and *returns* — the write is now complete in real time. At t=10 a client issues `write x=1`; this operation is *in flight*. At t=15, while that write is still outstanding, client A reads `x`. What may A return? Either 0 or 1. Linearizability says the write takes effect at *some* point between its call (t=10) and its return (t=20); A's read overlaps that interval, so the linearization point of the write might fall before or after A's read. Both answers are legal — and crucially, whichever one A sees pins down where the write's instant must be, constraining everyone else.

At t=20 the write returns. Now the write is complete in real time. Client B reads at t=25 — *after* the write's return. B **must** return 1. There is no legal history in which B sees 0, because the write completed before B began, so real-time order forces the write's instant before B's read. Client C reads at t=30 and must also return 1. And here is the subtle, powerful part: once *one* client observes the new value, *every later* read must observe it too. If A at t=15 had returned 1, and some other client at t=18 returned 0, that would be illegal — it would require the write's instant to be both before t=15 and after t=18. Linearizability is a *recency* guarantee with teeth: the moment the new value is visible to anyone, it is the new truth for everyone who comes after.

This is why linearizability is the right model for anything that behaves like a lock, a leader election, a unique-ID allocator, a "claim this username" check, or a configuration flag that must flip atomically. ZooKeeper and etcd exist precisely to give you a small amount of linearizable storage for coordination metadata, because the moment one node believes it holds the lock, no other node may believe otherwise.

### The cost: coordination, latency, and the CAP wall

The price is steep and unavoidable. To make every read see the latest completed write, the replicas have to *agree* on the order of operations before acknowledging them, which means consensus — Paxos, Raft, or a quorum protocol — on the hot path. A linearizable write to a geo-replicated system pays at least one wide-area round trip to a majority of replicas. Worse, Jepsen states the availability consequence flatly: linearizability "cannot be totally or sticky available; in the event of a network partition, some or all nodes will be unable to make progress." This is the consistency-side horn of the [CAP theorem and its PACELC extension](/blog/software-development/database/cap-theorem-and-pacelc): when the network partitions a linearizable system, the minority side cannot serve writes (and often cannot serve fresh reads) without risking a violation, so it must return errors or block. You trade availability for the single-copy illusion.

DDIA Chapter 9 makes the latency point vivid even *without* a partition: the more replicas you spread across the planet for durability and locality, the more expensive every linearizable operation becomes, because each one waits for a quorum that now spans oceans. The "cost of linearizability" is not a one-time architectural fee; it is a tax on every single operation, forever.

```python
# A linearizable counter using a Raft-backed store (e.g. etcd) needs a
# read-modify-write under a compare-and-swap, which forces a quorum round trip
# on BOTH the read and the write. No local-only fast path exists.
import etcd3

client = etcd3.client(host="etcd-leader", port=2379)

def increment_linearizable(key: str) -> int:
    while True:
        # Linearizable read: served by the leader after confirming it is
        # still leader via a quorum heartbeat. One WAN round trip.
        value, meta = client.get(key)
        current = int(value or b"0")
        new = current + 1
        # Compare-and-swap: succeeds only if no one else wrote in between.
        # This is the consensus round trip — the real cost center.
        ok = client.transaction(
            compare=[client.transactions.version(key) == meta.version],
            success=[client.transactions.put(key, str(new))],
            failure=[],
        ).succeeded
        if ok:
            return new
        # CAS failed: someone raced us. Retry. Under contention this loops,
        # which is the linearizable counter's other hidden cost.
```

The lesson encoded in that loop: linearizability does not just cost a round trip, it costs a round trip *plus* contention. Every client that wants the latest value is serialized through the same coordination point. For a username-claim check that runs once per signup, that is fine. For a per-request hit counter on your hottest page, it is a self-inflicted bottleneck, and the right answer is a weaker model with a CRDT, which we reach at the bottom of the ladder.

## 2. Sequential consistency: a global order, but not the real-time one

> **Senior rule of thumb:** sequential consistency keeps the one thing that makes reasoning easy — a single agreed-upon order everyone sees — but throws away the expensive thing: agreement with the real-world clock. It is what hardware memory models and many "strong" caches actually provide.

Sequential consistency, introduced by Leslie Lamport in 1979 for multiprocessor memory, sits one rung below linearizability. The definition keeps the *single global order* but drops the *real-time* constraint. Formally: there exists one total order of all operations such that (a) every client sees that same order, and (b) the order is consistent with each individual client's *program order* — the sequence in which that client issued its own operations. What it does *not* require is that the global order match real time. If client A's write finished, in wall-clock terms, before client B's write started, sequential consistency is still allowed to place B's write first in the global order, as long as it does so for everyone and it does not reorder A's own operations relative to each other.

### A worked example: what sequential allows that linearizable forbids

Suppose two clients each write to `x`, on a system whose global order is fixed but lags real time. In real time, A writes `x=1` and it returns at t=10; then B writes `x=2` and it returns at t=20. A third client C reads `x` at t=30 and gets `2`. A fourth client D reads `x` at t=31 and gets... `1`.

Under **linearizability** this is illegal. A's write and B's write are non-overlapping in real time (A returned at 10, B started after), so the order must be `x=1` then `x=2`; C reading 2 at t=30 forces D, reading later, to also see 2. D seeing 1 is time travel.

Under **sequential consistency** it depends. Sequential is allowed to pick the global order `x=2, x=1` — placing B's write first even though B happened later in real time — *as long as every client agrees*. But then C and D must both see the *same* order. If the global order is `x=2, x=1`, then the final value is 1, and *both* C and D should read 1 (or whatever the order's last write dictates), consistently. What sequential forbids is C and D *disagreeing*. The two readers must observe one shared timeline. What it permits is for that shared timeline to ignore the wall clock.

The practical consequence: sequential consistency gives you a system that is internally coherent — no client ever sees the order contradict another client's view — but it can lag reality. A read can return a value that is stale relative to a write that, in real time, already completed, *provided no one has yet observed the newer state*. The moment recency across the real-time boundary matters (a lock, a "did my payment go through" check), sequential is not enough; you need linearizable. When you only need everyone to agree on *an* order — a replicated log that all consumers process identically, a configuration that may apply with a small delay as long as it applies in the same sequence everywhere — sequential is the cheaper, sufficient choice. It still requires agreement on the order, so it still cannot be totally available under partition; per Jepsen, "all models at or stronger than sequential cannot be totally available in asynchronous networks." It saves the real-time round trip, not the ordering one.

### Where you have already met sequential consistency

If you have ever debugged a multithreaded program with a data race, you have met sequential consistency — it is the *strongest* memory model most CPUs offer without explicit fences, and even then most modern hardware (x86-TSO, ARM, POWER) provides something *weaker* than sequential by default, which is why you reach for `std::atomic` with `memory_order_seq_cst` or a memory barrier. The reason hardware drops below sequential is the same reason distributed systems do: enforcing a single global order across cores (or replicas) means stalling on a coherence round trip, and the architects decided most code does not need it. Lamport's 1979 definition was written about exactly this multiprocessor setting, and it transplants to distributed storage unchanged. The intuition to carry: "sequential" is the model where every observer agrees on *the movie*, but the movie may be running a few frames behind the live event, and there is no narrator forcing it to match the wall clock on the cinema wall.

### The composability trap

There is a sharp, often-overlooked difference between linearizability and sequential consistency that decides which one you can actually build a system out of: **linearizability composes; sequential consistency does not.** If every individual object in your system is linearizable, then the system as a whole — across all objects — is linearizable, for free, because real time is a single global reference frame that stitches the per-object orders together. This is *the* reason linearizability is so prized: you can reason about each object independently and the guarantees still hold when you use several of them together. Sequential consistency has no such property. Two individually-sequential objects, composed, can produce a history that is *not* sequentially consistent, because each object chose its own internal order with no shared clock to reconcile them, and there may be no single global order that satisfies both. In practice this means a "sequentially consistent cache" in front of two independent backends is not, jointly, sequentially consistent — a subtle footgun that has burned many a caching layer. When you need the guarantee to survive being built out of parts, you need linearizability, and that composability is part of what you are buying with the round trip.

## 3. Causal consistency: order only what cause-and-effect demands

> **Senior rule of thumb:** causal consistency is the strongest model you can keep while staying available during a partition. It enforces the one ordering humans actually notice — effects after their causes — and lets everything genuinely independent happen in any order.

Here is the rung where the cost curve bends sharply, and it is the most important model in the whole article for practical "always-on" systems. Causal consistency drops the demand for a *single global* order and keeps only the orderings implied by the **happens-before** relation, Lamport's partial order of potential causality. Operation A happens-before operation B if: A and B are by the same client and A came first (program order); or B reads a value that A wrote (read-from); or there is a chain of such relations connecting them (transitivity). Two operations with no happens-before path either way are **concurrent**. Causal consistency requires that if A happens-before B, then *every* replica applies A before B. For concurrent operations, it imposes nothing — different replicas may see them in different orders, and that is fine.

![A happens-before graph where Bob's reply causally depends on Alice's post so every replica must order the post first, while Carol's concurrent edit may be seen before or after the reply](/imgs/blogs/consistency-models-from-linearizable-to-eventual-3.webp)

The figure is the canonical example. Alice posts a message P (a write). Bob reads P, then writes a reply R. Because Bob's R was written *after he read* P, there is a happens-before chain: P → (Bob reads P) → R. Causal consistency demands that **every** observer sees P before R. The illegal history — R visible somewhere before P — is the "answer appears before the question" bug, and causal consistency forbids it across the entire system, even during a partition. Meanwhile Carol makes an edit E that is *concurrent* with R: she never saw R, and Bob never saw E. There is no happens-before path between E and R either way, so the system is free to show them in either order, and different users legitimately may see different orderings. That residual freedom is *exactly* what buys availability: because concurrent operations need no agreement, a replica can accept a write locally and gossip it later, never blocking on a quorum.

### Why this is the partition-survivable ceiling

The COPS paper, ["Don't Settle for Eventual: Scalable Causal Consistency for Wide-Area Storage"](https://www.cs.cmu.edu/~dga/papers/cops-sosp2011.pdf) by Lloyd, Freedman, Kaminsky, and Andersen (SOSP 2011), proved the headline result: under the ALPS requirements — Availability, low Latency, Partition tolerance, and high Scalability — *causal+ consistency is the strongest model you can achieve*. Anything stronger (sequential, linearizable) requires coordination that a partition can sever, forcing you to choose unavailability. Causal does not, because it only ever orders things that are already causally linked, and that linkage travels *with* the data. The "+" in causal+ is convergent conflict handling: concurrent writes to the same key are merged deterministically so all replicas land on the same value (more on that in the eventual-consistency section). Causal consistency is exactly the model behind a comment thread that always shows replies after their parents, a collaborative document where your teammate's edits you've seen never vanish, and a social feed where the like you placed on a post never appears detached from it.

### How it is tracked: dependencies and vector clocks

The mechanism is metadata. Each write carries the set of writes it causally depends on — the writes the client had observed when it issued this one. A replica applying a write first checks that all its dependencies are already present; if not, it buffers the write until they arrive. Tracking dependencies precisely uses **vector clocks** (or version vectors): a vector with one counter per node, where applying a local write increments your own counter and receiving a remote write takes the element-wise max plus your increment. Comparing two vectors tells you the relationship for free: if vector U is element-wise ≤ vector V (and not equal), then U happens-before V; if neither dominates the other, the operations are concurrent. That single comparison is the engine that both causal consistency and conflict detection run on.

```python
# Vector clocks: detect happens-before vs concurrent in O(#nodes).
from dataclasses import dataclass, field

@dataclass
class VectorClock:
    counts: dict = field(default_factory=dict)  # node_id -> counter

    def tick(self, node: str) -> None:
        """Local event: increment our own component."""
        self.counts[node] = self.counts.get(node, 0) + 1

    def merge(self, other: "VectorClock") -> None:
        """Receive a remote update: element-wise max."""
        for n, c in other.counts.items():
            self.counts[n] = max(self.counts.get(n, 0), c)

    def compare(self, other: "VectorClock") -> str:
        """Returns 'before', 'after', 'equal', or 'concurrent'."""
        keys = set(self.counts) | set(other.counts)
        le = all(self.counts.get(k, 0) <= other.counts.get(k, 0) for k in keys)
        ge = all(self.counts.get(k, 0) >= other.counts.get(k, 0) for k in keys)
        if le and ge:
            return "equal"
        if le:
            return "before"       # self causally precedes other
        if ge:
            return "after"
        return "concurrent"       # neither dominates -> conflict to resolve

# Alice posts; Bob reads then replies; Carol edits independently.
alice, bob, carol = VectorClock(), VectorClock(), VectorClock()
alice.tick("alice")                      # P:  {alice:1}
bob.merge(alice); bob.tick("bob")        # R:  {alice:1, bob:1}  -> R is AFTER P
carol.tick("carol")                      # E:  {carol:1}
print(bob.compare(alice))                # 'after'      (R must follow P everywhere)
print(carol.compare(bob))                # 'concurrent' (E vs R: any order legal)
```

The cost of causal consistency is real but bounded: you ship a dependency vector (or a compressed summary of it) with every operation, and replicas buffer out-of-order arrivals briefly. There is no quorum, no blocking on the hot path, no cross-replica round trip on the write. That is why causal sits in the sweet spot of the cost ladder — and why "don't settle for eventual" became a rallying cry: for the price of some metadata, you eliminate the most jarring class of anomalies while keeping the availability that made you choose a distributed store in the first place.

## 4. Session guarantees: a slice of sanity per client

> **Senior rule of thumb:** even when the system as a whole is only eventually consistent, you can give each *individual user* a coherent view of their *own* actions for the price of routing tricks and a little client-side metadata. This is the cheapest consistency that users actually feel, and it fixes 80% of the "the database is haunted" tickets.

Causal consistency is global — it orders dependencies for everyone. The **session guarantees**, introduced by Terry, Demers, Petersen, Spreitzer, Theimer, and Welch in the 1994 Bayou paper ["Session Guarantees for Weakly Consistent Replicated Data"](https://tycon.github.io/terry-session-guarantees.html), are a clever weakening: forget global order, just make the database consistent *with one client's own actions* across a single session, even as that client bounces between inconsistent replicas. There are four, and each forbids exactly one replication-lag anomaly. The figure lays them out against the anomaly each one bans and what it costs to enforce.

![A matrix of the four session guarantees showing what each forbids, the concrete user-facing symptom, and the mechanism that enforces it](/imgs/blogs/consistency-models-from-linearizable-to-eventual-4.webp)

**Read-your-writes** (also called read-my-writes): if a client writes a value, every subsequent read *in the same session* reflects that write or a later one. Terry's formal version: if read R follows write W in a session and R runs on server S at time t, then W is included in the database state DB(S, t). The anomaly it forbids is the most common ticket of all — you post a comment, the page refreshes, and the comment is gone, because the refresh read hit a replica that hadn't received your write yet. Vogels, in ["Eventually Consistent"](https://www.allthingsdistributed.com/2007/12/eventually_consistent.html), defines it identically: "Process A, after it has updated a data item, always accesses the updated value and never will see an older value." Enforcement: route the client's reads to a replica known to have its writes — a sticky session pinned to one replica, or reads served from the leader, or the client passing a version token its reads must meet.

**Monotonic reads**: if a client reads a value, every later read in the session returns that value or a newer one — reads never go *backward in time*. The symptom: you see a reply on a thread, refresh, and the reply is gone, because read 1 hit a fresh replica and read 2 hit a lagging one. Formally, RelevantWrites for the first read are a subset of the database state the second read sees. Vogels: "once a process observes a value, subsequent reads never return earlier versions." Enforcement: pin the client to a single replica for the session, so it never moves to a more-stale one.

**Monotonic writes**: the system applies a client's writes in the order the client issued them, on every replica. The symptom: an edit is applied before the create it depends on, or two edits land out of order so the older one wins. Formal version: a replica performs a write only after it has performed all previous writes from the same session. Vogels: "the system guarantees to serialize the writes by the same process." Enforcement: tag each write with a per-session sequence number; replicas hold a write until the prior one has been applied.

**Writes-follow-reads** (session causality): a write a client makes is ordered *after* any write whose effect that client read earlier in the session. The symptom: a reply appears before the post it answers, because the system didn't carry the read dependency into the subsequent write. This is the per-session shadow of causal consistency — it is "the converse of read-your-writes," ensuring writes you make are ordered after the writes you saw. Enforcement: the client carries the version vector of what it read and stamps its next write to depend on it.

### Session consistency is read-your-writes scoped to a session

Vogels names a fifth, practical variant: **session consistency** is read-your-writes restricted to a single session — when the session ends (and a new one begins, perhaps on a different replica), the guarantee resets. This is what most "stick the user to a region/replica via a cookie" setups deliver. It is cheap: a routing decision plus a little state, no quorum, no global agreement. And it is the highest-leverage consistency a product team can add, because session guarantees are about *what one user perceives about their own actions*, and that perception is exactly where the haunted-database tickets come from. Per Jepsen, read-your-writes is the boundary of availability: "all models at or stronger than read your writes can be at most sticky available" — meaning the client must keep talking to the *same* replica to retain the guarantee; if that replica is unreachable, the client either fails over (and may lose the guarantee) or waits. Below read-your-writes, monotonic reads and monotonic writes can be *totally* available.

### Worked example: which session guarantee fixes which ticket

Say your system is eventually consistent under the hood — writes go to any replica and gossip lazily. A user does the following in one session: (1) writes a profile bio "Hello", (2) reads their bio, (3) writes bio "Hello world", (4) reads their bio, then in a *new* session (5) reads their bio.

- Without read-your-writes, step 2 may return the old empty bio (read hit a stale replica). Ticket: "I saved my bio and it didn't save."
- With read-your-writes but without monotonic writes, steps 1 and 3 might be applied out of order, leaving the final stored bio as "Hello" instead of "Hello world." Ticket: "my edit got reverted."
- With read-your-writes and monotonic reads, steps 2 and 4 are coherent and forward-moving within the session. But step 5, in a *new* session, may again read stale data if you only had *session* consistency — the guarantee did not carry over.

That last point is the practical boundary: session guarantees are a per-session contract, not a global one. They are the right tool when each user mostly reads their own writes — profiles, settings, drafts, a personal feed — and they are cheap. They are *not* a substitute for causal consistency when users interact with *each other's* writes (comment threads, shared docs), because they say nothing about ordering across different clients' sessions. That is precisely the line between session causality and full causal consistency.

## 5. Eventual consistency: converge if the writes ever stop

> **Senior rule of thumb:** eventual consistency is not a guarantee about *when* — it is a guarantee about *if*. If writes stop, replicas converge. While writes continue, all bets about recency are off, and the only thing you control is what happens when two replicas disagree.

At the bottom of the ladder is the weakest model still worth a name. Werner Vogels' definition is the canonical one: "The storage system guarantees that if no new updates are made to the object, eventually all accesses will return the last updated value." The operative phrase is *if no new updates are made* — eventual consistency promises convergence only in the limit, once the firehose of writes pauses long enough for gossip to catch up. It says nothing about the **inconsistency window**: how long, after a write, other replicas keep returning the old value. That window might be milliseconds on a healthy LAN or minutes during a partition. Eventual consistency is what you get from Amazon's Dynamo, from Cassandra and Riak with low quorums, from DNS, from CDNs, from any system that prioritizes availability and low latency above recency.

![A timeline where a partition splits two replicas, each accepts a conflicting cart write, reads diverge during the stale window, then the replicas merge and converge once the partition heals](/imgs/blogs/consistency-models-from-linearizable-to-eventual-5.webp)

The figure shows the defining scenario, drawn from the shopping-cart example the [Dynamo paper](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/dynamo.pdf) (DeCandia et al., SOSP 2007) made famous. A partition splits replicas R1 and R2. Client X, talking to R1, adds item A to the cart; client Y, talking to R2, adds item B. During the partition both writes succeed locally — that is the whole point, the system stays available — and reads on the two sides *diverge*: R1 says the cart is [A], R2 says [B]. This is the stale window made concrete; neither replica is wrong, they simply haven't talked. When the partition heals and gossip resumes, the replicas must reconcile [A] and [B] into one cart. Get the reconciliation right (union to [A, B]) and the customer keeps both items; get it wrong (pick one, drop the other) and an item silently vanishes. Eventual consistency guarantees the replicas *will* agree on something; it does not, by itself, guarantee they agree on the *right* thing. That is the job of conflict resolution.

### Quorums: the dial between strong and eventual

The knob that tunes how stale "eventual" gets is the quorum relationship Vogels states crisply: with N replicas, a read touching R of them and a write touching W of them, "if W+R > N then the write set and the read set always overlap and one can guarantee strong consistency." Overlap means every read is guaranteed to touch at least one replica that saw the latest write, so it can return the freshest value. When W+R ≤ N, the read and write sets can be disjoint, a read may miss the newest write entirely, and you are squarely in eventual-consistency territory — faster and more available, but stale reads are now legal.

![A before-and-after comparison of a strict quorum where read and write sets overlap (W+R greater than N) versus a sloppy quorum where they are disjoint (W+R at most N) and stale reads become possible](/imgs/blogs/consistency-models-from-linearizable-to-eventual-9.webp)

The figure makes the overlap physical. With N=3, W=2, R=2, you have W+R=4 > 3, so any read of 2 replicas and any write of 2 replicas must share at least one — the read is guaranteed to see the write, at the cost of a round trip to a majority on each operation. With N=3, W=1, R=1, you have W+R=2 ≤ 3: a write to one replica and a read from a different one can completely miss each other, so a stale read is legal — but each operation is fast and stays available even when most replicas are down. Dynamo goes further with a **sloppy quorum** and hinted handoff: when the first N healthy nodes don't include the "home" replicas (because of a partition), it writes to whatever N healthy nodes it can reach and hands the data off later. This maximizes availability at the cost of an even wider stale window and the possibility of *concurrent* writes to logically the same key — which is exactly why Dynamo needs vector clocks to detect those conflicts. DDIA Chapter 5's treatment of replication lag and Chapter 9's of quorums both make the same point: quorums are not a binary strong/weak switch but a continuous dial, and even "strict" quorums have edge cases (concurrent writes, sloppy quorums, failed read repair) where staleness leaks through.

This is the heart of the [CAP and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) trade-off rendered as a tunable parameter, and it interacts with how you've set up [synchronous versus asynchronous replication](/blog/software-development/database/database-replication-sync-async-logical-physical): a synchronous (high-W) replica gives you stronger reads and slower, less-available writes; an asynchronous (low-W) replica gives you the opposite.

## 6. Conflict resolution: LWW, version vectors, and CRDTs

Eventual consistency guarantees convergence but not correctness; the conflict-resolution strategy decides whether a concurrent update is *lost*, merely *flagged*, or *automatically and correctly merged*. This is where the most expensive silent-data-loss bugs in distributed systems live, and where the most elegant theory — CRDTs — earns its keep. The figure compares the three strategies head to head.

![A matrix comparing last-write-wins, version vectors, and CRDTs across how each decides a conflict, the result for two concurrent writes, and the outcome for the application](/imgs/blogs/consistency-models-from-linearizable-to-eventual-6.webp)

### Last-write-wins and the lost-update trap

**Last-write-wins (LWW)** is the default in many systems (Cassandra, by design) because it is dead simple: tag each write with a timestamp, and on conflict keep the one with the higher timestamp. The fatal flaw is hiding in plain sight: when two writes are *concurrent* — neither causally precedes the other — LWW still picks a "winner" by comparing wall clocks, and the loser is **silently discarded**. This is the classic lost update. If clients X and Y concurrently update the same record and Y's clock happens to be 3 milliseconds ahead, X's write evaporates with no error, no log entry, no trace. Clock skew makes it worse: "higher timestamp" can mean "the replica with the faster-running clock," not "the write that happened later." DDIA is blunt that LWW achieves convergence "at the cost of durability" — it will throw away committed writes to converge. LWW is acceptable only when losing a concurrent write is genuinely fine (a cache entry, a last-seen-online timestamp, a setting where the most recent intent really should win). It is catastrophic for a shopping cart, a counter, or a collaborative document.

### Version vectors: detect, don't decide

**Version vectors** (the per-object cousin of vector clocks) do not pretend a winner exists. When two writes are concurrent, the version vectors are incomparable — neither dominates — and the system *detects* this and keeps both versions as **siblings**, surfacing the conflict to the application (or a merge function) to resolve. This is Dynamo's approach: the shopping cart returns both [A] and [B] as siblings on the next read, and the cart's merge logic (union the items) reconciles them. The cost is that conflicts become the application's problem — you must write merge logic, and reads can return multiple versions. The benefit is *no silent loss*: every concurrent write is preserved until something deliberately reconciles it. Version vectors turn an invisible data-loss bug into a visible, handleable event, which is almost always the right trade for anything you care about.

### CRDTs and strong eventual consistency

**CRDTs** — Conflict-free Replicated Data Types, formalized by Shapiro, Preguiça, Baquero, and Zawirski in their [2011 INRIA paper](https://inria.hal.science/hal-00932836v1) — go one step further: they make merge *automatic and always correct* by constraining the data type so that concurrent operations *commute* or the states form a **join-semilattice**. If the merge operation is commutative, associative, and idempotent, then it does not matter what order replicas receive updates in, how many times they receive them, or how the network reorders things — every replica that has received the same *set* of updates computes the same value, deterministically. Shapiro et al. call the resulting guarantee **Strong Eventual Consistency (SEC)**: replicas that have delivered the same updates have equivalent state, *with no conflict resolution and no consensus*. SEC is strictly better than plain eventual consistency because it removes the "converge on *something*" weasel — there is provably one value everyone converges to, and it is reached by a deterministic merge rather than a coin flip.

The mechanics come in two flavors. **State-based (CvRDT)** replicas periodically ship their whole state and merge via the semilattice *join* (least upper bound); convergence needs only that the join is a proper lattice operation. **Operation-based (CmRDT)** replicas ship individual operations that are designed to commute, requiring reliable causal-order delivery of ops. A G-Counter (grow-only counter) keeps a per-node count and merges by element-wise max then sums — adds commute trivially. A PN-Counter pairs two G-Counters for increments and decrements. An OR-Set (observed-remove set) tags each add with a unique id so a concurrent add and remove resolve deterministically (add-wins or remove-wins by construction), which is exactly what you want for a shopping cart or a set of collaborators.

```python
# A grow-only counter CRDT (G-Counter). Increments commute; merge is
# element-wise max; the value is the sum. No coordination, ever.
class GCounter:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: dict[str, int] = {}

    def increment(self, by: int = 1) -> None:
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + by

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> None:
        # Join in the semilattice: element-wise max is commutative,
        # associative, and idempotent -> Strong Eventual Consistency.
        for n, c in other.counts.items():
            self.counts[n] = max(self.counts.get(n, 0), c)

# Two replicas increment independently during a partition, then merge.
r1, r2 = GCounter("r1"), GCounter("r2")
r1.increment(3)           # r1 sees 3
r2.increment(5)           # r2 sees 5
r1.merge(r2); r2.merge(r1)
assert r1.value() == r2.value() == 8   # converged, nothing lost, no consensus
# Merge again (idempotent) or in the other order (commutative): still 8.
r1.merge(r2)
assert r1.value() == 8
```

CRDTs are the technology behind Riak's distributed data types, Redis's CRDT-based active-active geo-replication (Redis Enterprise / CRDB), Automerge and Yjs for local-first apps, and the collaborative engines behind tools like Figma's multiplayer canvas. The trade-off is not free: CRDTs carry metadata (tombstones for removes, per-node counters) that can grow over time and need garbage collection, and not every data type has a natural conflict-free design — a CRDT text editor must encode intent (insert-here-relative-to-this-character) rather than absolute positions, which is genuinely hard. But where a CRDT fits, it gives you the holy grail for the eventual-consistency world: local-latency, always-available writes that *never* conflict and *never* lose data, achieved with arithmetic instead of consensus.

## 7. How this maps to transaction isolation

> **Senior rule of thumb:** consistency models govern single objects and real-time order; isolation levels govern multi-object transactions and concurrency anomalies. They are two axes of the same space, and the corners have names.

A persistent source of confusion is that distributed-systems people say "consistency" and database people say "isolation," and the words overlap without matching. Jepsen's hierarchy unifies them, and the cleanest way to see the relationship is a two-by-two grid: one axis is *single object versus multiple objects*, the other is *whether real-time order is required*.

![A two-by-two matrix mapping single-object versus multi-object on one axis and real-time order on the other, placing sequential, linearizable, serializable, and strict serializable in the four corners](/imgs/blogs/consistency-models-from-linearizable-to-eventual-7.webp)

Read the corners. **Linearizability** is the single-object, real-time corner: one object, operations respect wall-clock order. **Serializability** is the multi-object, *no real-time* corner: it is the isolation guarantee that a set of transactions over many objects produces a result equivalent to *some* serial order of those transactions — but, exactly like sequential consistency, that serial order need not match real time. A serializable database is allowed to execute transaction T1 (which committed first in real time) *after* T2 in its equivalent serial order, as long as the result is equivalent to some serial schedule. The famous consequence: under plain serializability, a read-only transaction can return a stale snapshot, because "some serial order" can place it in the past.

The bottom-right corner is **strict serializability** (sometimes "strong-1SR" or external consistency): serializable *plus* linearizable — a serial order that *also* respects real-time order across transactions. This is the strongest practical model, and it is what Google Spanner provides. Spanner's documentation describes [external consistency](https://cloud.google.com/blog/products/databases/strict-serializability-and-external-consistency-in-spanner) as "a property of transaction-processing systems" stronger than linearizability, where if transaction T2 starts after T1 commits in real time, T2's commit timestamp is greater than T1's. Spanner achieves it with TrueTime — GPS-and-atomic-clock-bounded timestamps and a *commit-wait* step where a transaction waits out the clock uncertainty interval before releasing its result, guaranteeing that no client sees the effect until the commit timestamp is safely in the past. That commit-wait is the multi-object, planet-scale analog of the linearizable round trip: you pay a few milliseconds of waiting on every commit to buy real-time order globally.

The mental hook worth keeping: **linearizability is the single-object, real-time analog of serializability**, and **strict serializable = serializable + linearizable**. If you understand the consistency ladder, you already understand isolation levels along the orthogonal axis — and for the messier middle of that axis (read committed, snapshot isolation, repeatable read, and the write-skew and phantom anomalies they do and don't prevent), see the companion piece on [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent). The two articles describe the same lattice from two directions.

## 8. The cost ladder: what each rung charges you

We have climbed the whole ladder; now look at the bill. The defining trade-off of the entire field is that *strength costs coordination*, and *coordination costs latency and availability*. The figure stacks the four core models against the two costs that matter — coordination/latency on the hot path, and availability under partition.

![A grid laying out linearizable, sequential, causal, and eventual against their coordination cost and their availability behavior under a network partition](/imgs/blogs/consistency-models-from-linearizable-to-eventual-8.webp)

Read it bottom to top. **Eventual** charges nothing on the hot path: writes go to the local replica and gossip asynchronously, so every operation is local-latency and the system is *totally available* — any replica can answer at any time, even during a partition. **Causal** adds dependency tracking (the version vectors from section 3): a little metadata per operation and some buffering of out-of-order arrivals, but still no quorum and no hot-path round trip, so writes stay local-latency and the system is *sticky available* in a partition (a client keeps its causal view as long as it stays on replicas that have its dependencies). This is the highest rung you can occupy and still survive a partition — the COPS result made concrete. **Sequential** adds the requirement of a single global order, which needs agreement, so it *cannot be totally available* under partition. **Linearizable** adds real-time order on top, which needs consensus or a strict quorum on every operation — the one-WAN-round-trip tax — and is *unavailable* on the minority side of a partition.

The ladder is the whole strategy: each step up buys you fewer anomalies and charges you coordination, latency, and availability. The engineering skill is not "always pick the strongest" (you will build something slow and fragile) nor "always pick the weakest" (you will ship the haunted-database bugs). It is to **pick the weakest model that still makes your feature correct**, per operation, and to know the boundary precisely.

| Model | Coordination on hot path | Availability under partition | Recency guarantee | Typical use |
| --- | --- | --- | --- | --- |
| Strict serializable | Consensus + commit-wait, multi-object | Unavailable (minority) | Real-time, transactional | Financial ledgers, Spanner |
| Linearizable | Consensus / strict quorum per op | Unavailable (minority) | Real-time, single object | Locks, leader election, unique IDs |
| Sequential | Agreement on global order | Not totally available | Single agreed order, may lag real time | Replicated logs, ordered config |
| Causal | Dependency vectors, no quorum | Sticky available | Effects after causes | Comments, feeds, collaborative state |
| Session guarantees | Routing + client metadata | Sticky (RYW) / total (MR, MW) | Per-client own-action coherence | Profiles, settings, drafts |
| Eventual | None (async gossip) | Totally available | Converge only if writes stop | Caches, counters (CRDT), DNS |

## 9. A per-feature playbook: which model does this need?

The payoff of the whole ladder is being able to answer, for a real feature, the question "which model does this need?" without ceremony. Here is the playbook I actually use, framed as the question to ask and the answer it implies.

- **"Can two users grab the same scarce thing?"** (username, seat, inventory unit, distributed lock, leader role) → **Linearizable.** The instant one client believes it holds the resource, no other may believe otherwise. Pay the consensus round trip; it runs rarely. Anything weaker permits double-booking.
- **"Does the money have to be right and in real-time order across accounts?"** (transfers, balances, double-entry ledgers) → **Strict serializable.** Multi-object atomicity *and* real-time order. This is Spanner's home turf; if you can't run Spanner, a single-region serializable database with synchronous replication is the pragmatic fallback.
- **"Must replies follow their posts and edits stay attached for *everyone*?"** (comment threads, activity feeds, shared documents, chat) → **Causal consistency** (causal+ for convergent merges). Strong enough to kill the "reply before post" anomaly, weak enough to stay available worldwide.
- **"Does each user mostly read their *own* writes?"** (profiles, settings, drafts, notification read-state, personal dashboards) → **Session guarantees** — at minimum read-your-writes and monotonic reads. Cheap, routing-based, fixes the bulk of the haunted-database tickets without any global coordination.
- **"Is it a count, a set, or a flag where losing a concurrent update is unacceptable but real-time recency isn't?"** (like counts, view counters, shopping carts, collaborative sets, presence) → **Eventual consistency with a CRDT.** Local-latency, always-available, provably convergent, no silent loss. The single highest-leverage pattern in the always-available world.
- **"Is it a cache, a hint, or a value where the most recent intent should simply win and a lost concurrent write is genuinely fine?"** (CDN content, last-seen-online, derived caches, feature-flag values that change rarely) → **Eventual with LWW.** Accept the lost-update risk explicitly, because here it costs nothing.

> The mistake is never "we chose the wrong model." It is "we never chose a model" — we accepted the default of whatever store we picked and discovered its consistency contract from the bug tracker.

## Case studies from production

### 1. The vanishing comment (read-your-writes)

A social product ran reads through a fleet of read replicas behind a round-robin load balancer, with asynchronous replication from the primary. A user would post a comment (write to primary), the client would immediately refetch the thread (read to a *random* replica), and roughly one time in five the replica hadn't received the write yet, so the comment was absent. Support called it "the database eating comments." The wrong first hypothesis was a write bug — engineers spent a week auditing the write path, which was flawless. The actual root cause was the *absence of read-your-writes*: the post-write read was allowed to hit a stale replica. The fix was a session guarantee, not a stronger global model: for a short window after a write, pin the user's reads to the primary (or to a replica confirmed to hold their latest write via a version token). Zero change to the replication topology, zero added latency for everyone else, ticket volume to near zero. The lesson: the cheapest correct model — a session guarantee — beat the instinct to "make the database strongly consistent."

### 2. Reads that went backward (monotonic reads)

A media site showed a live comment count that visibly *decreased* on refresh. Two adjacent page loads hit two replicas with different replication lag; the first load saw a fresher count, the second an older one, so the number jumped down. Users assumed comments were being deleted. The wrong hypothesis was a caching bug in the CDN. The real cause was the lack of *monotonic reads*: nothing forced a client's successive reads to be non-decreasing in freshness. The fix was to derive a sticky replica choice from a hash of the user/session id, so each session consistently read from the same replica and therefore never moved to a more-stale one. It traded a sliver of load-balancing evenness for monotonicity, and the count stopped going backward. The lesson: "the value moved backward in time" is *always* a monotonic-reads violation, and the fix is routing, not a rewrite.

### 3. The edit that reverted itself (monotonic writes)

A settings service let users save preferences from a mobile app over a flaky connection. The client retried writes aggressively, and writes were routed to whichever replica was reachable. A user would change a setting twice in quick succession; the two writes occasionally landed on different replicas and replicated in the *wrong order*, so the older value won and the newer edit appeared to revert. The wrong hypothesis was a client retry bug. The root cause was missing *monotonic writes*: nothing enforced that a session's writes applied in issue order. The fix was a per-session monotonically increasing sequence number stamped on each write, with replicas refusing to apply a write until the prior sequence number had been applied. The lesson: when the same client issues a rapid sequence of writes, you must order them per session, or last-writer-by-arrival will scramble them.

### 4. The shopping cart that lost an item (version vectors vs LWW)

An e-commerce platform stored carts in a Dynamo-style eventually-consistent store with LWW conflict resolution. During a brief partition, a customer (on two devices, or via a retried request hitting two replicas) added two different items concurrently. LWW kept the write with the higher timestamp and *silently dropped* the other item. Revenue impact: customers checked out missing items they had clearly added. The wrong hypothesis was a frontend bug dropping add-to-cart clicks. The root cause was using LWW for data where concurrent writes must *merge*, not compete. The fix was the Dynamo design itself: switch the cart to version vectors so concurrent writes are preserved as siblings, and add a deterministic merge function (union the line items, sum quantities). Later they replaced the hand-rolled merge with an OR-Set CRDT and deleted the merge code entirely. The lesson: LWW on a shopping cart is a silent revenue leak; carts are the textbook case for version vectors and CRDTs.

### 5. The distributed lock that wasn't (linearizability)

A team built a "distributed lock" on top of an eventually-consistent key-value store: write your owner-id to a key if it's empty, read it back to confirm you won. Under normal conditions it worked; under a partition, two nodes on opposite sides both saw the key empty (stale reads), both wrote their id, both "confirmed" by reading their own replica, and both believed they held the lock. The result was two workers processing the same job, double-charging customers. The wrong hypothesis was a logic error in the lock acquisition code. The root cause was attempting linearizable semantics (mutual exclusion *is* linearizability) on a non-linearizable store. The fix was to move the lock to a system designed for it — etcd with a lease, which provides linearizable compare-and-swap via Raft — and to add a fencing token so that even a delayed lock holder couldn't act after losing the lock. The lesson: mutual exclusion is linearizability by another name; you cannot build it on a store that permits stale reads, full stop.

### 6. The reply before the question (causal consistency)

A geo-distributed forum replicated posts asynchronously across regions. A user in region A posted a question; a user in region B, reading from a replica that *had* received the question, posted an answer. A third user, reading from a region-C replica, received the *answer* before the *question* had replicated there, and saw "Re: [deleted/missing]" — an answer with no visible parent. The wrong hypothesis was data corruption. The root cause was the absence of causal consistency: the answer's causal dependency on the question wasn't tracked, so region C could deliver them out of order. The fix was to attach a dependency version to each write and have replicas buffer a write until its dependencies had arrived — textbook causal consistency, as in COPS. The system stayed fully available (no quorum added); it just stopped delivering effects before their causes. The lesson: any time one user's write is a *reaction* to another user's write, you need at least causal consistency, and it is cheap enough to always afford.

### 7. The counter that throttled the homepage (eventual + CRDT)

A startup implemented a per-article view counter as a linearizable increment in their primary database — read the count, add one, compare-and-swap. On their most popular article during a traffic spike, every pageview serialized through the same CAS, the retry loop thrashed under contention, and the counter became the homepage's bottleneck; latency spiked and some increments were dropped on exhausted retries. The wrong hypothesis was that the database needed more write capacity. The root cause was using the *strongest* model for data that needed almost none: a view count does not need real-time recency, only eventual convergence and no lost increments. The fix was a G-Counter CRDT — each replica increments locally with zero coordination, and counts merge by element-wise max-then-sum. Throughput became effectively unbounded, no increment was ever lost, and the displayed count converged within the gossip interval. The lesson: reaching for linearizability "to be safe" can be the actual bug; counters are the canonical place to drop to eventual-with-CRDT.

### 8. The TrueTime tax nobody budgeted for (strict serializable)

A team migrated a financial ledger to Spanner for its strict-serializable guarantees and were thrilled with correctness — and then surprised by commit latency. Every write transaction paid a *commit-wait*: Spanner waits out the TrueTime uncertainty interval (a handful of milliseconds) before releasing the commit, which is exactly how it guarantees real-time order globally. Under a tight clock-sync deployment the wait was small; on one poorly-provisioned cluster with loose time sync, the uncertainty interval ballooned and commit latency with it. The wrong hypothesis was a Spanner performance regression. The root cause was that *the strongest model has a non-negotiable real-time cost* — external consistency is bought with commit-wait, and the wait scales with clock uncertainty. The fix was tightening time synchronization (better NTP/TrueTime provisioning) and moving genuinely non-transactional reads to stale-read mode, which skips coordination. The lesson: strict serializability is correct and expensive *by design*; the cost is real-time order, and you pay it on every commit.

### 9. Sticky sessions that broke on failover (the limits of session guarantees)

A service delivered read-your-writes via sticky sessions: a cookie pinned each user to one replica. It worked beautifully until a replica failed and clients failed over to another — which did *not* have their recent writes, so on failover users transiently saw their own writes vanish. The wrong hypothesis was a bug in the failover logic. The root cause was the inherent boundary of session guarantees that Jepsen names precisely: read-your-writes is *at most sticky available*; lose the sticky replica and you may lose the guarantee. The fix was to make the guarantee portable — have the client carry a version token of its latest write and route failover reads only to a replica that meets or exceeds it (waiting briefly if necessary). This upgraded "sticky available" toward "causally available" for that client. The lesson: session guarantees are a routing contract; when routing breaks (failover, replica loss), the guarantee breaks unless the client carries enough metadata to re-establish it elsewhere.

### 10. The cache that served yesterday's price (inconsistency window)

A pricing service cached prices in a CDN and an in-memory layer, both eventually consistent. A price change propagated to most edges in seconds but to a few in *minutes*, and during that inconsistency window a handful of customers saw — and were charged — the old price, triggering a compliance escalation. The wrong hypothesis was a cache-invalidation bug. The root cause was treating "eventually consistent" as "consistent enough" for data with a hard correctness deadline. The fix was a hybrid: keep the eventually-consistent cache for *display*, but make the *checkout* path read the price linearizably from the source of truth at the moment of charge, so the customer is always billed the authoritative price even if the displayed price lagged. The lesson: eventual consistency is fine for the read-mostly display path and wrong for the moment of commitment; split the operation and apply the model each half actually needs.

### 11. Vector clocks that grew without bound (the metadata cost)

A team adopted version vectors to track conflicts in a long-lived per-user document store. It worked, but over months the vectors accumulated an entry per writer node the document had ever touched, including decommissioned nodes, and the metadata grew larger than some small documents. Reads slowed and storage bloated. The wrong hypothesis was a serialization inefficiency. The root cause is intrinsic to causal-metadata schemes: vector clocks and CRDT tombstones carry per-actor state that must be *garbage collected*, and naïve implementations never prune it. The fix was dotted version vectors and periodic compaction of entries for retired nodes, plus tombstone GC on the CRDT sets, bounding the metadata to active writers. The lesson: causal consistency and CRDTs are not free — they trade coordination for metadata, and metadata that isn't garbage-collected becomes its own scaling problem.

### 12. The "strongly consistent" read that wasn't, under sloppy quorum

A team ran a Dynamo-style store with a "strict" quorum (W+R > N) and assumed reads were therefore always fresh. During a partition the store fell back to a *sloppy* quorum with hinted handoff to stay available — writing to substitute nodes that weren't the data's home replicas. A subsequent read of the home replicas (which hadn't yet received the hinted-off writes) returned a *stale* value despite W+R > N on paper. The wrong hypothesis was a quorum-math error. The root cause was that sloppy quorums break the overlap guarantee: the "write set" during the partition wasn't the home set, so the read set didn't actually intersect it. The fix was to disable sloppy quorum for the keys that needed real freshness (accepting reduced availability for them) and to rely on read-repair plus higher R only where staleness was tolerable. The lesson, straight from DDIA Chapter 9: even a strict quorum on paper does not guarantee linearizability — sloppy quorums, concurrent writes, and incomplete read-repair all leak staleness, so "W+R > N" is necessary, not sufficient.

### 13. The collaborative editor that fought itself (CRDT vs OT)

A document-editing product built its real-time collaboration on a hand-rolled "merge the diffs" scheme over an eventually-consistent backend. When two users typed in the same paragraph at the same time, the absolute character offsets in their edits referred to *different* versions of the text, so applying both diffs produced garbled, interleaved characters — the document literally fought itself. The wrong hypothesis was a race in the diff transport. The root cause was that absolute positions are not conflict-free: an insert at offset 12 means something different after a concurrent insert at offset 5 shifted everything right. The fix was to model the document as a sequence CRDT (an RGA / replicated growable array, the family behind Automerge and Yjs), where each character carries a unique, stable identity and inserts are expressed *relative to* a neighbor's identity rather than an absolute offset. With identities instead of offsets, concurrent inserts commute and converge deterministically — the same Strong Eventual Consistency guarantee Shapiro et al. proved, applied to text. The team considered operational transformation (OT, the Google Docs lineage) as an alternative; OT achieves the same end but requires a central server to transform operations against each other, whereas the CRDT works peer-to-peer with no server in the merge path. The lesson: collaborative text is the hardest CRDT to get right because intent must be encoded structurally, but once you do, concurrent editing stops being a conflict problem and becomes pure arithmetic.

### 14. Monotonic reads broke a payment retry (compounding anomalies)

A payments service polled an eventually-consistent status store to decide whether to retry a charge: "if status is not yet SUCCESS, retry." The poll loop hit different replicas across iterations. A charge would succeed and the status would flip to SUCCESS on the replica that processed it, but a later poll, hitting a *laggier* replica, would read the old PENDING status — a reads-go-backward violation — and the retry logic would fire a *second* charge. Customers were double-billed. The wrong hypothesis was an idempotency-key bug in the payment gateway. The actual cause was a *monotonic-reads* violation compounding into a correctness failure: the absence of monotonicity let a later read see an earlier state, and the retry logic trusted that read. The fix had two parts: pin the poll loop to a single replica for monotonic reads, *and* make the charge itself idempotent with a stable key so even a spurious retry could not double-charge — defense in depth, because the weak consistency model could not be fully eliminated under the availability requirement. The lesson: a weak consistency model is not just a UX nuisance; when business logic *makes decisions* off a non-monotonic read, the anomaly compounds into real money lost, and the durable fix is to both tighten the read (routing) and make the action safe to repeat (idempotency).

## When to reach for strong consistency, and when not to

**Reach for linearizable or strict serializable when:**

- The operation is a form of mutual exclusion or uniqueness — locks, leader election, unique-username/seat/inventory allocation. The instant one party commits, all others must observe it.
- Money or any conserved quantity moves and must be correct in real-time order across objects — ledgers, balances, transfers. Strict serializable; Spanner-class or single-region serializable + synchronous replication.
- A downstream decision depends on reading the *latest* committed state at the moment of decision — the checkout charge, the "did my payment clear" check, the fencing of a job.
- The operation runs rarely enough that a consensus round trip is a rounding error in your latency budget.

**Reach for causal or session guarantees when:**

- Users interact with each other's writes and effects must follow causes — comment threads, feeds, chat, shared documents → causal (causal+ for merges).
- Each user mostly reads their own writes and you need to kill the haunted-database tickets cheaply — profiles, settings, drafts → read-your-writes + monotonic reads.
- You need to stay available and low-latency worldwide and can tolerate concurrent-but-independent operations being seen in different orders.

**Reach for eventual consistency (with the right conflict resolution) when:**

- The data is a count, set, or flag where availability and local latency dominate and concurrent updates must merge without loss — view counts, carts, presence, collaborative sets → CRDT.
- The most recent intent should simply win and a lost concurrent write is genuinely acceptable — caches, last-seen, rarely-changing flags → LWW, chosen *on purpose*.

**Skip strong consistency when:**

- You are tempted to make *everything* linearizable "to be safe." This is how you build a system that is slow on the happy path and unavailable on the unhappy one. Pick the weakest model per operation that is still correct.
- The feature is read-mostly display data with no hard correctness deadline — a stale read for a few hundred milliseconds harms nothing. Eventual is cheaper and more available.
- You're reaching for a distributed lock or transaction to coordinate work that could instead be made idempotent or commutative — often the cheapest "consistency" is designing the operation so order and duplication don't matter (the CRDT philosophy applied to your own logic).

The whole art compresses to one sentence. Every read your system serves is *allowed* to return a set of values, and that set is defined by the consistency model you are running — chosen or inherited. Walk the ladder, find the weakest rung on which your feature is still correct, pay exactly that much coordination, and you will have neither the haunted-database bugs of going too weak nor the latency-and-availability tax of going too strong. That is the entire discipline, and it begins with refusing to let the bug tracker be the place you discover which rung you are standing on.

## Further reading

- [Jepsen: Consistency Models](https://jepsen.io/consistency/models) and [Linearizability](https://jepsen.io/consistency/models/linearizable) — Kyle Kingsbury's hierarchy map and the precise implication relationships between every model discussed here.
- [Aphyr, "Strong consistency models"](https://aphyr.com/posts/313-strong-consistency-models) — the readable companion to the Jepsen hierarchy.
- Werner Vogels, ["Eventually Consistent"](https://www.allthingsdistributed.com/2007/12/eventually_consistent.html) (and the [CACM version](https://dl.acm.org/doi/10.1145/1435417.1435432)) — the canonical definitions of eventual, read-your-writes, monotonic, and session consistency, plus the W+R>N quorum rule.
- Terry et al., ["Session Guarantees for Weakly Consistent Replicated Data"](https://tycon.github.io/terry-session-guarantees.html) — the Bayou paper that named the four session guarantees.
- DeCandia et al., ["Dynamo: Amazon's Highly Available Key-value Store"](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/dynamo.pdf) — eventual consistency, vector clocks, sloppy quorums, and the shopping-cart conflict.
- Lloyd et al., ["Don't Settle for Eventual: Scalable Causal Consistency with COPS"](https://www.cs.cmu.edu/~dga/papers/cops-sosp2011.pdf) — causal+ as the strongest model under ALPS constraints.
- Shapiro et al., ["Conflict-free Replicated Data Types"](https://inria.hal.science/hal-00932836v1) — the CRDT and Strong Eventual Consistency foundations.
- Google, ["Strict Serializability and External Consistency in Spanner"](https://cloud.google.com/blog/products/databases/strict-serializability-and-external-consistency-in-spanner) — how TrueTime and commit-wait buy real-time order at global scale.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapters 5 (replication lag, session guarantees) and 9 (linearizability, ordering and causality, the cost of linearizability) — the definitive textbook treatment.
- Companion posts: [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc), [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), and [database replication: sync, async, logical, physical](/blog/software-development/database/database-replication-sync-async-logical-physical).

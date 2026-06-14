---
title: "CAP Is Misunderstood, and PACELC Is What You Actually Use"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Why the famous 'pick 2 of 3' framing of CAP is wrong, what the theorem actually proves, and how PACELC's latency-versus-consistency tradeoff is the one you pay on every request."
tags:
  [
    "cap-theorem",
    "pacelc",
    "distributed-systems",
    "consistency",
    "availability",
    "partition-tolerance",
    "linearizability",
    "databases",
    "spanner",
    "dynamo",
    "system-design",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/cap-theorem-and-pacelc-1.webp"
---

Almost every engineer who has touched a distributed database has, at some point, drawn the CAP triangle on a whiteboard. Three corners — Consistency, Availability, Partition tolerance — and the punchline: "pick two." It is a great piece of folklore. It is also wrong in a way that quietly leads teams to make the wrong architectural decision, classify their database into a category that does not exist, and reason about an operating regime they will almost never be in while ignoring the one they live in every single day.

I have sat in design reviews where someone said "we chose a CP system, so we accept lower availability" — and then the team proceeded to ship a feature whose actual pain point was tail latency under perfectly healthy network conditions, a thing CAP says nothing about. I have seen the reverse: a team that picked an "AP" database because availability sounded good, then spent six months chasing a class of bug — lost writes, resurrected deletes, double-charged customers — that was the inevitable downstream of a consistency choice nobody had consciously made. The folklore got them into the room; the folklore could not get them out.

The fix is not to throw CAP away. CAP is a real theorem with a real proof, and it tells you something true and important. The fix is to state it precisely, understand exactly how narrow its scope is, and then reach for the framework that covers the other 99.9% of your system's life: **PACELC**. The diagram above — the partition moment, one write on one node, one read on another, and a dropped link in between — is the mental model for the whole article. CAP is a statement about *that single instant*: when the link is down, the read on the far node can be fresh or it can be answered, but not both. PACELC keeps that instant and then adds the part CAP omits — what you trade off when the link is *up*. The rest of this piece is a tour of that distinction, why it matters per feature, how it classifies the databases you actually use, and what to pick.

![During a partition a replica must either block the read for consistency or answer stale for availability](/imgs/blogs/cap-theorem-and-pacelc-1.webp)

Read the figure left to right. A client writes `x = 2`; node A has the fresh value. The link to node B drops — that is the partition. A reader hits node B, which still holds the old `x = 1`. Node B now faces an unavoidable fork: it can choose **C** and refuse to answer (block or error) because it cannot prove its value is current, or it can choose **A** and return the stale `1` to stay available. There is no third door. That fork, and *only* that fork, is what CAP is about. Everything people commonly believe CAP says beyond this is either imprecise or false.

## Why CAP is different from what you were taught

The mismatch between the textbook CAP and the real CAP is large enough to be worth tabulating before we go deeper. Most of the damage done by the "pick two" mnemonic comes from a handful of specific misconceptions, each of which has a precise correction.

| The common belief | The naive mental model | The reality |
| --- | --- | --- |
| "You pick 2 of the 3 properties." | A system is permanently labeled CA, CP, or AP, like a blood type. | You cannot drop P in a real distributed system, so "CA" is not an operating point. The only genuine choice is C-vs-A *during* a partition. |
| "Consistency means ACID consistency." | The C in CAP is the C in ACID — invariants, foreign keys, constraints. | CAP's C is **linearizability** (a.k.a. atomic / strong consistency): a register that behaves as if there is one copy and every operation is instantaneous. It has nothing to do with ACID's "C." |
| "Availability means the system is up." | High uptime, five nines, no downtime. | CAP's A is a specific liveness property: *every request to a non-failing node returns a (non-error) response*, with no time bound. A node that answers in three hours is "available" by this definition. |
| "Partition tolerance is a feature you can buy or skip." | Some databases tolerate partitions, others don't. | A partition is a fact of the network, not a setting. P means the system keeps *running* when messages are dropped; you don't get to opt out of the network dropping messages. |
| "CAP describes my database's normal behavior." | CAP governs everyday reads and writes. | CAP only constrains behavior *during a partition*. When the network is healthy — which is nearly always — CAP imposes nothing. The everyday tradeoff is latency-vs-consistency, which is PACELC's "ELC." |

Every row of that table is a place teams lose months. The most expensive one is the second: conflating CAP's C with ACID's C. They are unrelated. We will come back to it, because getting it wrong makes people think a single-node SQL database "gives up consistency" the moment they add a read replica, which is nonsense.

Eric Brewer, who first stated the conjecture in 2000, said this himself in his 2012 retrospective ["CAP Twelve Years Later: How the 'Rules' Have Changed"](https://mwhittaker.github.io/papers/html/brewer2012cap.html): the "2 of 3" formulation "was always misleading because it tended to oversimplify the tensions among properties." The man who invented the triangle spent a paper explaining that the triangle misleads you. That alone should make us suspicious of the whiteboard version.

> CAP is a theorem about one bad moment in your system's life. PACELC is a framework for the rest of it. Confusing the two is how you optimize for an emergency you rarely have while ignoring the tax you pay constantly.

## 1. The theorem, stated precisely

Let us nail down what CAP actually proves, because the precise version is both narrower and more interesting than the folklore. In 2002, Seth Gilbert and Nancy Lynch of MIT turned Brewer's informal conjecture into a real theorem and proved it, in a paper with the unwieldy title ["Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services"](https://www.cs.princeton.edu/courses/archive/spr22/cos418/papers/cap.pdf). Their definitions are the canonical ones, and they are far more specific than "consistency, availability, partition tolerance."

### Consistency = linearizability, not ACID's C

Gilbert and Lynch define consistency as **atomic** (linearizable) data objects. Informally: the system behaves as if there is a single copy of the data, and every read and write takes effect atomically at some instant between when the client issued it and when the client got the response. Formally, there exists a total order on all operations consistent with real-time ordering, and each read returns the value of the most recent write in that order.

The crucial consequence: if a write completes and then a read begins (in real time, after the write returned), the read must see that write or something newer. There is no "I just wrote it but the read came back with the old value" — that is precisely the anomaly linearizability forbids. This is a *register* property, about a single key. It says nothing about multi-key invariants. The C in ACID — "a transaction moves the database from one valid state to another, respecting constraints" — is an entirely different idea that lives in the application's transaction logic. You can have an ACID-compliant single-node database that is trivially linearizable (one copy, no replication), and you can have a multi-key transactional system that is serializable but *not* linearizable (it may serve a stale-but-consistent snapshot). The two C's are orthogonal. For a deeper treatment of where these models sit relative to each other, see [consistency models, from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

In Martin Kleppmann's *Designing Data-Intensive Applications* (Chapter 9), he makes the same point with characteristic bluntness: CAP's consistency is linearizability and nothing more, and he calls CAP "almost completely useless in practice" for system design precisely because people stretch its three letters to mean things the theorem never addressed. His preferred phrasing is that linearizability is a *recency guarantee* — a read sees the latest write — and it is expensive, because it forces coordination.

### Availability = every request to a live node gets a response

Gilbert and Lynch's availability is a liveness property: **every request received by a non-failing node must result in a response.** Not a fast response. Not a correct response. Just a response, eventually. A node that is up and not crashed must not hang forever or refuse to answer.

Two subtleties trip people up. First, "non-failing node" — if a node has crashed, no one expects it to answer; availability is about nodes that are alive but possibly cut off. Second, there is no latency bound in the formal definition, which is both a strength (the proof is clean) and a weakness (real availability is absolutely about latency — a 30-second response is a failed request to your users). PACELC will repair this by promoting latency to a first-class concern.

### Partition tolerance = the system keeps running when messages are lost

A partition is a set of dropped messages between groups of nodes. Gilbert and Lynch model it as the network being allowed to lose arbitrarily many messages between two groups. Partition tolerance means the system continues to operate — to satisfy C and A as best it can — despite this message loss. It is not a feature; it is a stance toward an environmental hazard.

### The proof is a two-node argument anyone can follow

Here is the entire proof, and it fits in a paragraph. Take two nodes, `N1` and `N2`, replicating a single value `v`, initially `v0`. Partition them so no message gets through. A client writes `v1` to `N1`. For availability, `N1` must accept and acknowledge the write — but it cannot tell `N2`, because the link is down. Now a client reads `v` from `N2`. For availability, `N2` must respond. For consistency (linearizability), `N2` must return `v1`, since the write completed before the read began. But `N2` never heard about `v1`; it still has `v0`. So `N2` either returns `v0` (violating consistency) or refuses to answer (violating availability). It cannot do both. Therefore, in the presence of a partition, you cannot have both C and A. That is the theorem.

```
Time →
  partition starts  (N1 ✗—✗ N2 : no messages cross)

  client → N1 : write(v1)
  N1        : ack            // available: must respond
  N1        : (cannot reach N2)

  client → N2 : read(v)
  N2        : ??? 
              → return v0    // AVAILABLE but not CONSISTENT (stale)
              → block/error  // CONSISTENT but not AVAILABLE
```

Notice what the proof does *not* say. It does not say you pick a permanent label. It does not say anything about the case where the partition is absent. It does not mention latency, ACID, replication topology, or your specific database. It is a statement about one instant — the partition window — and one impossibility within it. Everything the folklore piles on top is extrapolation, and most of the extrapolation is wrong.

## 2. Why "pick 2 of 3" is the wrong frame

The "pick 2 of 3" slogan implies three symmetric, interchangeable choices: CA, CP, AP, each equally legitimate. The reality is that one of those three — **CA** — is not a real operating point at all, and the asymmetry is the whole point.

![A distributed system cannot opt out of partitions, so CA collapses into a behavior choice during P](/imgs/blogs/cap-theorem-and-pacelc-2.webp)

The figure above contrasts the myth with the reality. On the left is "pick 2 of 3," which offers CA — consistent and available — as if you could just *choose* never to have a partition. But a partition is not a choice; it is what the network does to you. Fibers get cut. Switches reboot. A misconfigured BGP route blackholes a data center. A garbage-collection pause makes a node *look* partitioned to its peers for eight seconds. If your system spans more than one machine connected by a network, partitions will happen, and the only question is what your system does when one does. "Choosing CA" is choosing to be unprepared for an event you cannot prevent — which is not a strategy, it is a bug waiting for a network blip.

On the right is the honest framing. **P is mandatory.** Given that, the real choice is what you do *during* a partition: prioritize A (stay up, possibly serve stale or accept conflicting writes) or prioritize C (stay correct, possibly refuse service on the cut-off side). That collapses the three labels into two meaningful ones: a system is either **PA** (favors availability during a partition) or **PC** (favors consistency during a partition). CA, as a distinct category, evaporates.

Brewer made exactly this point in the 2012 retrospective: because you cannot forgo partition tolerance in a distributed system, the architect's only real decision is whether to forfeit consistency or availability when a partition occurs. The "third option," CA, only exists for a system that is *not distributed* — a single node, where there is no network to partition. The bottom-right cell in the figure makes this explicit: a single box, no replication, is genuinely CA, because there is no partition to survive. The moment you add a second node and a network between them, CA is off the table.

### The single-node exception, and why it confuses people

The reason CA feels real is that single-node databases are everywhere, and they *are* CA. A standalone PostgreSQL instance is linearizable (one copy of the data) and available (it answers as long as it is up), and "partition tolerance" is meaningless because there is no internal network to partition. People then add a read replica for scaling or a standby for failover, mentally keep the "CA" label, and are surprised when the read replica serves stale data or the failover loses the last few committed transactions. They did not lose a property they had; they entered a regime — distribution — where the old label never applied. The label problem is a category error, and it is the same category error Kleppmann warns about: treating CAP's three letters as a taxonomy of systems rather than a constraint on one moment.

This is also why the framing matters for ordinary architecture decisions like [synchronous versus asynchronous replication](/blog/software-development/database/database-replication-sync-async-logical-physical). Async replication is, in effect, a PA/EL choice: the primary acknowledges your write before the replica has it, so you get low latency and high availability, at the cost of a window where a read on the replica (or a failover to it) can be stale or lose recent writes. Sync replication is the PC/EC choice: the primary waits for the replica to confirm, paying latency on every write to guarantee that a failover never loses an acknowledged write. You are not "choosing CA"; you are choosing where on the latency/consistency curve each write sits. That is PACELC, which is where we go next.

## 3. Partitions are rare but inevitable — so CAP describes a narrow situation

Here is the fact that reframes everything: **partitions are rare.** In a well-run single-region cluster on modern hardware and networking, genuine partitions — not node crashes, which CAP handles separately, but actual network splits where live nodes cannot talk — might happen a handful of times a year, lasting seconds to minutes each. Across a fleet, the cumulative partition time is a tiny fraction of total uptime, often well under 0.01%.

That sounds like good news, and it is — but it has a sharp implication for how much CAP should drive your design. CAP only constrains your system *during a partition*. If partitions are 0.01% of your system's life, then CAP directly governs 0.01% of your system's behavior. The other 99.99% — every read and write on a healthy network — is completely outside CAP's scope. CAP says nothing about it. Yet that 99.99% is where your users live, where your latency budget is spent, where your tail-latency incidents happen, and where the vast majority of your engineering effort goes.

Brewer's 2012 paper draws out three reasons the "2 of 3" frame misleads, and the first is exactly this: "Partitions are rare, and when a system is not partitioned, the system can have both strong consistency and high availability." Read that twice. *When there is no partition, you can have both C and A.* CAP does not force a tradeoff in the common case. The tradeoff it describes is real but rare. The tradeoff you pay constantly is a different one, and CAP is silent about it.

The second reason Brewer gives: consistency and availability "can vary by subsystem or even by operation." You do not make one CAP choice for your whole system; you can make different choices per feature, even per request. We will build a per-feature decision matrix later, because this is the single most practically useful idea in the entire CAP/PACELC literature and it is the one teams most consistently fail to apply.

The third reason: there is a *spectrum* of consistency and a spectrum of availability, not a binary. "Strong consistency" and "eventual consistency" are the endpoints of a range that includes bounded staleness, session consistency, consistent prefix, monotonic reads, and read-your-writes. Likewise availability ranges from "always answers, possibly stale" through "answers if a quorum is reachable" to "answers only from the leader." CAP's binary C-or-A is a cartoon of a continuum. The continuum is where the engineering is.

### A partition is a time bound on communication

There is one more insight from Brewer that connects CAP directly to latency and sets up PACELC. **A partition is a time bound on communication.** Concretely: when node A sends a message to node B and waits for a reply, it cannot wait forever. At some point — say, after a 500 ms timeout — A must decide: is B partitioned, or just slow? A *declares* a partition when a timeout fires. The system enters partition mode and makes its C-or-A choice.

But this means the boundary between "no partition" and "partition" is a timeout knob you set. Make the timeout shorter and you declare partitions more aggressively (more time in partition mode, but faster failure detection). Make it longer and you tolerate more latency before declaring a partition (fewer false partitions, but slower reaction). The partition is not a crisp physical event; it is a decision your system makes when communication takes too long. And "communication takes too long" is *also* what causes latency. Partition handling and latency are the same phenomenon viewed at two timescales. This is the bridge to PACELC: if a partition is a time bound on communication, then the everyday cost of consistency — waiting for that communication to succeed before you answer — is just a partition you chose not to declare. The latency-vs-consistency tradeoff in the healthy case is the *same shape* as the availability-vs-consistency tradeoff during a partition, scaled down to milliseconds.

## 4. Enter PACELC: two questions, asked at different times

In 2010, Daniel Abadi proposed the framework that fills CAP's gap, in a paper whose title says it all: ["Consistency Tradeoffs in Modern Distributed Database System Design: CAP is Only Part of the Story"](https://paperswelove.org/papers/consistency-tradeoffs-in-modern-distributed-databa-4a1ea3bd/). His thesis is that CAP, by focusing only on the partition case, misses the tradeoff that has *actually* driven the design of most real distributed databases: the choice between latency and consistency in the *normal*, non-partitioned case.

PACELC reads as a sentence: **if there is a Partition (P), choose between Availability (A) and Consistency (C); Else (E), choose between Latency (L) and Consistency (C).** The first half, "PAC," is just CAP — the partition case. The second half, "ELC," is the new part: when everything is healthy, you *still* face a tradeoff, because consistency requires coordination, and coordination costs time.

![PACELC asks A-or-C only during a partition and L-or-C on every normal request you serve](/imgs/blogs/cap-theorem-and-pacelc-3.webp)

The decision tree above is the structure. A request arrives. The system asks: are we partitioned? If yes (the rare path), it is the CAP fork — stay available with stale data, or stay consistent by refusing. If no (the everyday path, the one that fires on essentially every request you serve), it is the PACELC-specific fork — answer fast from a nearby replica without coordinating (favor L, accept possible staleness), or coordinate with a quorum to guarantee the latest value (favor C, pay the round-trip).

The reason ELC is the tradeoff "you actually use" — the title of this article — is the same arithmetic from the previous section. The partition branch fires perhaps 0.01% of the time. The else branch fires the other 99.99%. Every read your service does, every write, every healthy moment, you are paying *either* latency *or* consistency. That tax is continuous, measurable, and shows up in your p99 dashboards. It is the tradeoff that should dominate your design, and CAP does not even mention it.

Abadi's argument, in his own framing, is that the latency-consistency tradeoff "has had a more direct influence on several well-known distributed database systems" than CAP's availability-consistency tradeoff. Systems were not designed around the rare partition; they were designed around the everyday round-trip. Dynamo, the system that launched the whole NoSQL availability movement, was built primarily to keep latency low and writes always-accepted — the *Else* part — not because Amazon expected constant partitions. Abadi's point is that we mislabeled an entire generation of databases by describing them in CAP terms when their defining choice was a PACELC "Else" choice.

### The ELC tradeoff is just replication math

Why does consistency cost latency even with a perfect network? Because to guarantee a read sees the latest write, the read must consult enough replicas to be sure it is not missing a more recent value. The standard mechanism is quorums: with a replication factor `RF`, if write quorum `W` and read quorum `R` satisfy `R + W > RF`, then every read quorum overlaps every write quorum, so a read is guaranteed to see the latest acknowledged write. (This is the rule [Cassandra's tunable consistency](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html) is built on; more on that below.)

The cost is the round-trips. A linearizable read with `R = majority` must contact a majority of replicas — possibly across availability zones or regions — and wait for the slowest of them. An eventual read with `R = 1` hits the nearest replica and returns immediately. On a healthy network, the difference is pure latency: the strong read waits for coordination the weak read skips. There is no partition involved. This is why DynamoDB charges you exactly twice as many read-capacity units for a strongly consistent read as for an eventually consistent one, and why strong reads add roughly 5–10 ms of latency — the strong read must route to the leader/majority instead of using the nearest replica. You are paying, in dollars and in milliseconds, for the *Else* branch, on a perfectly healthy network, on every request. That is PACELC made concrete.

```python
# Quorum intuition: R + W > RF guarantees a read sees the latest write.
RF = 3            # three replicas per key

# EC (strong) configuration: read and write majorities overlap.
W, R = 2, 2       # 2 + 2 = 4 > 3  -> linearizable-ish, both ops wait for 2 replicas

# EL (eventual) configuration: fast, but R + W can fail to overlap.
W, R = 1, 1       # 1 + 1 = 2 <= 3 -> a read may miss the latest write -> staleness

def overlaps(R, W, RF):
    return R + W > RF

assert overlaps(2, 2, 3)        # strong: every read sees every prior write
assert not overlaps(1, 1, 3)    # eventual: reads can be stale, but each op waits for 1 node
```

The first config is your EC choice: every operation waits for two replicas, guaranteeing recency at the cost of latency. The second is your EL choice: every operation waits for one replica, returning fast but allowing a read to miss a recent write. Same network, same hardware, no partition — just a knob that trades milliseconds for recency. Cassandra and DynamoDB literally expose this knob per query.

## 5. Classifying real systems by PACELC quadrant

PACELC gives every distributed database a two-part label: a P-side letter (PA or PC, what it does during a partition) and an E-side letter (EL or EC, what it does normally). That yields four quadrants, and the database you use sits in one of them — often configurably.

![Real databases occupy distinct PACELC quadrants that predict their behavior both during and outside partitions](/imgs/blogs/cap-theorem-and-pacelc-4.webp)

The matrix above places the four combinations. Notice the diagonal: most systems are either **PA/EL** (availability-first during partitions, latency-first normally — the Dynamo lineage) or **PC/EC** (consistency-first always — the Spanner lineage). The off-diagonal quadrants are real but rarer. Let us walk each one with named systems and the exact mechanisms that put them there.

### PA/EL — Dynamo, Cassandra, Riak

These are the systems built for "always writable, always fast." During a partition they keep accepting reads and writes on both sides (PA), reconciling conflicts later via vector clocks, last-write-wins, or CRDTs. In normal operation they default to low-latency, low-coordination reads and writes (EL), letting you dial consistency up per query if you need it.

Amazon's [DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadConsistency.html) defaults to eventually consistent reads — the EL choice — and offers strongly consistent reads at double the cost and added latency. Apache Cassandra's whole identity is *tunable consistency*: you specify a consistency level per operation (`ONE`, `QUORUM`, `LOCAL_QUORUM`, `ALL`). `ONE` is the EL extreme — one replica acknowledges, fastest possible — and `ALL` or `QUORUM` moves you toward EC. Riak is built on Dynamo's design directly. Abadi classifies all three as PA/EL: their default posture is availability and latency, with consistency as an opt-in.

The defining property of PA/EL is that *the default favors the fast, available path, and you pay extra (in latency or in cost) to get consistency*. If you deploy Cassandra with default settings and never think about consistency levels, you have chosen PA/EL by omission — and the lost-update bugs that follow are not Cassandra's fault, they are the EL choice you made without noticing.

### PC/EC — Spanner, VoltDB, HBase, CockroachDB

These systems prioritize consistency in both regimes. During a partition, the minority side refuses to serve (PC) rather than risk stale or conflicting data. In normal operation, every read and write coordinates to guarantee linearizability (EC), paying the round-trip on every request.

Google Spanner is the flagship PC/EC system; it guarantees external consistency (linearizability) globally and, during a partition, the side that loses quorum stops serving. VoltDB (H-Store) is an in-memory NewSQL system that is fully serializable and PC/EC by design. HBase, built on a single-master-per-region model, is PC/EC: a region whose RegionServer is partitioned away becomes unavailable rather than serving stale data, and reads always go to the authoritative server. CockroachDB and TiDB and YugabyteDB belong here too — they use Raft/Paxos consensus to provide serializable transactions and refuse writes on a side that cannot reach a majority.

The defining property of PC/EC is that *correctness is never traded away; the price is paid in latency (always) and availability (during partitions)*. You choose PC/EC when a wrong answer is worse than a slow answer or no answer — bank ledgers, inventory counts, anything where money or correctness invariants are at stake.

### PC/EL — PNUTS, some Cosmos DB modes

This off-diagonal quadrant is subtle and underappreciated. A PC/EL system, during a partition, favors consistency (the minority refuses or the system blocks), but in *normal* operation it favors latency over consistency. Yahoo's PNUTS is Abadi's canonical example: it uses a per-record master for writes (so it can be consistent and refuses on partition — PC), but reads can be served from a local, possibly-stale replica for low latency (EL). The logic is "we will be consistent when it is cheap-ish — during a partition, when we have to make a hard choice, we choose C — but for fast reads in the common case, we will serve you a slightly stale local copy."

Azure Cosmos DB exposes [five consistency levels](https://learn.microsoft.com/en-us/azure/cosmos-db/consistency-levels) — Strong, Bounded Staleness, Session, Consistent Prefix, and Eventual — and depending on which you pick, you land in different PACELC quadrants. Session and Consistent Prefix are EL-leaning (fast local reads with weaker global guarantees) while the underlying replication can still make a PC-style choice on partition. The point is that a single product spans quadrants by configuration; the label is per-deployment, not per-product.

### PA/EC — rare, and usually a mistake by accident

The fourth quadrant — favor availability during partitions but consistency normally — is the rarest and most awkward. It says "in the common case I will pay for strong consistency, but the moment a partition hits I will throw consistency away and stay up." That is a strange posture: you spent latency on coordination every day, then abandoned the guarantee exactly when it mattered most. Some systems land here through configuration drift — a database set to strong reads normally but configured to fall back to any-available-replica on partition. MongoDB, depending on read/write concern settings, can be coaxed near here, which is part of why its consistency behavior has historically been so confusing (and why Jepsen found real bugs in it; see below). Abadi's classification puts MongoDB's default behavior in the PA family with EC-leaning configurations available — the exact label depends heavily on `writeConcern` and `readConcern`. The takeaway: PA/EC is mostly a place you end up by accident, not by design, and if you find yourself there, you should ask whether you actually wanted PA/EL (cheaper) or PC/EC (safer).

Here is the same classification as a reference table, with the mechanism that determines each placement:

| System | PACELC | During partition (P) | Normal operation (E) | Mechanism |
| --- | --- | --- | --- | --- |
| DynamoDB | PA/EL | both sides serve; reconcile later | eventual reads default; strong = 2× cost, +5–10 ms | quorum + per-request consistency |
| Cassandra | PA/EL | configurable; default serves | `CL=ONE` default; `QUORUM`/`ALL` opt-in | tunable consistency, `R+W>RF` |
| Riak | PA/EL | both sides serve; vector clocks | low-latency default; quorum opt-in | Dynamo design + CRDTs |
| PNUTS | PC/EL | per-record master refuses | local stale reads for low latency | per-record mastership |
| Cosmos DB | configurable | depends on level | Strong → EC; Session/Eventual → EL | five named consistency levels |
| MongoDB | PA/EC* | depends on `writeConcern` | strong-ish with majority concerns | replica set + read/write concern |
| Spanner | PC/EC | minority refuses to serve | linearizable, every op coordinates | TrueTime + Paxos |
| CockroachDB | PC/EC | minority refuses | serializable, Raft per range | Raft consensus |
| HBase | PC/EC | partitioned region unavailable | reads from authoritative server | single-master-per-region |
| VoltDB | PC/EC | refuses on partition | fully serializable, in-memory | single-threaded partitions |

The asterisk on MongoDB is doing real work: its quadrant genuinely depends on configuration, and the default has shifted across versions toward safer concerns. Never quote a PACELC label for MongoDB without naming the `writeConcern`/`readConcern` you actually deployed.

## 6. The everyday tax: latency versus consistency with no partition

Let us make the ELC tradeoff visceral, because it is the one you pay constantly and the one CAP hid from you. Forget partitions entirely. The network is perfect. You still must choose, on every read, whether to coordinate.

![Even with zero partitions, linearizable reads cost an extra quorum round-trip that eventual reads avoid](/imgs/blogs/cap-theorem-and-pacelc-5.webp)

The before/after above is the entire ELC choice in two columns. On the EL side: answer from the nearest replica, no coordination, single-digit-millisecond latency — but the value may be stale, because the nearest replica might not have the latest write yet. On the EC side: contact a majority of replicas, wait for the round-trip (5–10 ms or much more across regions), and return only after you are certain you have the latest committed value. Same healthy network. The only difference is whether you wait for coordination.

The numbers are not hypothetical. DynamoDB's strongly consistent reads cost double the read-capacity units and add measurable latency because they must route to the primary replica rather than the geographically closest one. Cosmos DB's documentation states that for the same request units, read throughput for Strong and Bounded Staleness is *half* that of the weaker levels, because strong reads use a two-replica minority quorum within a four-replica set. Half the throughput, for the same cost, in the common case, with no partition anywhere — that is the price of recency. Across regions the gap is brutal: a cross-region majority quorum can turn a 1 ms local read into a 100+ ms read, because you wait for a replica on another continent.

### Why this is the tradeoff that actually drives architecture

Tail latency is where applications die. Your p50 might be fine, but your p99 — the slowest 1% of requests, which disproportionately hit your most engaged users and your most complex pages — is where the consistency tax compounds. A strongly consistent read that waits for the slowest of three replicas has a p99 governed by the *tail of the slowest replica*, which is far worse than any single replica's p99. This is the "tail at scale" phenomenon: coordination multiplies tail latency. Choosing EC means accepting that your p99 is hostage to your slowest replica on every request.

This is also where the connection to [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) becomes practical. Isolation is the transaction-level analog of the same coordination cost: serializable isolation requires coordination that read-committed does not, and the price is, again, latency and contention in the healthy case. Both CAP/PACELC and isolation levels are, at bottom, the same question asked at different layers — "how much coordination am I willing to pay for how much correctness?" — and in both, the everyday (non-partition, non-conflict) cost is the one that dominates real systems.

Kleppmann's *DDIA* (Chapters 5 and 6, on replication and partitioning) spends pages on exactly this: the cost of linearizability is that *every* operation must coordinate, which limits throughput and inflates latency even when nothing is wrong, and which is fundamentally at odds with the low-latency, partition-resilient goals of geo-distributed systems. His recommendation is the one this article is building toward: do not make linearizability a global default; reserve it for the specific operations that truly need a recency guarantee, and let everything else run on cheaper, weaker models. That is a per-feature decision, which is the next section.

## 7. Pick consistency per feature, not per database

The single most useful idea in this entire literature — Brewer's "consistency can vary by subsystem or even by operation" — is that you do not make one CAP or PACELC choice for your whole system. You make it per feature, sometimes per request. A real application is a mix of data with wildly different consistency needs, and forcing all of it onto one global setting is how you either over-pay for latency on data that does not need it or under-protect data that does.

![Consistency requirements differ per feature, so one global CAP choice is the wrong granularity](/imgs/blogs/cap-theorem-and-pacelc-6.webp)

The matrix above runs three features of one ordinary e-commerce app through the PACELC lens, and they land in different quadrants. Let us reason through each one concretely, because this reasoning *is* the skill.

### Shopping cart — PA/EL

A shopping cart wants to stay writable no matter what. If a partition hits, the worst outcome of accepting an "add to cart" on both sides is that, on heal, you might have two copies of an item or a slightly stale cart — both trivially reconcilable (merge the carts; CRDTs make this automatic). The cost of *refusing* an add-to-cart is a lost sale and an annoyed customer. So during a partition, choose A: keep accepting items. In normal operation, the cart should be fast — nobody wants to wait for a cross-region quorum to add a sock to a cart — so choose L. **PA/EL.** This is, famously, exactly why Amazon built Dynamo: the shopping cart was the motivating use case, and "the cart must always accept writes" was the non-negotiable requirement. Reconciling a stale cart is easy; losing a sale is not.

### Account balance — PC/EC

A bank balance is the opposite. If a partition hits and you let both sides accept debits, a customer can withdraw the same $1,000 twice — once on each side — and you have manufactured money. The cost of refusing a debit during a partition (telling the customer "service temporarily unavailable") is real but bounded; the cost of double-spending is unbounded and possibly criminal. So during a partition, choose C: refuse the risky debit rather than allow a double-spend. In normal operation, a balance read must reflect the latest committed transactions — reading a stale balance and approving a withdrawal against it is the same double-spend bug — so choose C again. **PC/EC.** You pay latency on every balance operation, gladly, because the alternative is wrong money.

### Username uniqueness — PC/EC

Username (or email) uniqueness is a global invariant: at most one account per username. This is a linearizability problem in disguise. If a partition hits and both sides allow `alice` to register, on heal you have two `alice`s and a corrupted unique constraint. The cost of refusing a signup during a partition is a momentary "try again" message; the cost of duplicate usernames is data corruption that may be impossible to clean up. So during a partition, choose C: block the duplicate signup. In normal operation, the uniqueness check must be linearizable — a check that reads a stale "username available" and then commits creates the duplicate — so choose C. **PC/EC.** Uniqueness constraints are one of the clearest "you cannot cheat consistency here" cases in all of system design.

### The pattern: classify by the cost of being wrong

The decision rule that falls out of these three examples is simple and durable:

| Question | If yes → lean | If no → lean |
| --- | --- | --- |
| Is a stale or conflicting answer cheaply reconcilable? | A / L (availability, latency) | C / C (consistency) |
| Does the feature enforce a global invariant (uniqueness, balance ≥ 0)? | C / C | A / L |
| Is the cost of refusing service higher than the cost of a wrong answer? | A / L | C / C |
| Will users notice a few-ms latency more than occasional staleness? | A / L | depends on feature |

Run every feature through those four questions and you get a per-feature PACELC label. Then pick a database (or a configuration of one tunable database, like Cassandra or Cosmos DB) that lets each feature sit in its own quadrant. The mistake is picking one database, reading its single PACELC label, and forcing the bank balance and the shopping cart to share it. Modern tunable systems exist precisely so you do not have to — but you have to *use* the tunability deliberately, per query, which means you have to know which features need which quadrant. That knowledge is the deliverable of this whole exercise.

This per-feature reasoning also interacts with how you [partition and shard your data](/blog/software-development/database/database-partitioning-and-sharding): the consistency choice and the partitioning choice are coupled, because the scope of a transaction (single-shard vs cross-shard) determines how much coordination — and therefore how much latency or partition-fragility — each operation incurs. A balance that lives entirely within one shard can be PC/EC cheaply; a balance transfer that spans two shards needs a distributed transaction and pays the full coordination cost. Designing your shard boundaries so that your strong-consistency features stay single-shard is one of the highest-leverage moves in distributed data modeling.

## 8. Harvest and yield: degrade gracefully instead of failing hard

The CAP fork — "stay available with stale data, or stay consistent by refusing" — sounds binary, but there is a richer middle that Brewer (with Armando Fox) introduced years earlier and revisited in the 2012 paper: **harvest and yield.** It reframes availability from a binary "up or down" into two continuous quantities you can trade independently.

![Trading harvest for yield lets a partitioned system stay up by returning a labeled partial result](/imgs/blogs/cap-theorem-and-pacelc-7.webp)

**Yield** is the probability of completing a request — what fraction of requests get an answer. **Harvest** is the fraction of the data reflected in that answer — how *complete* the answer is. A traditional system treats these as one: either you get the full answer (harvest = 100%, yield succeeds) or you get nothing (yield = 0). The before/after figure above contrasts that binary world (any replica unreachable → whole query fails → yield drops to zero) with the harvest/yield world (answer from the reachable shards → flag that the result covers, say, 92% of the data → yield stays high while harvest dips).

The classic example is a search engine. If one of one hundred index shards is partitioned away, the binary choice is "return a 500" (yield = 0) or "return results from 99 shards and tell the user the result covers 99% of the index" (yield = 1, harvest = 0.99). The second is obviously better for a search engine: a 99%-complete result is nearly as good as a complete one, and a 500 is useless. By *degrading harvest* you *preserve yield*. The user gets a slightly-less-complete answer instead of an error, and — crucially — you tell them it is partial, so they can decide whether to trust it.

### When harvest/yield works, and when it does not

Harvest/yield is the right move when partial answers are meaningful and the data is decomposable: search results, recommendations, aggregations, dashboards, analytics. Returning "here are 92% of your metrics, the rest are temporarily unavailable" is far better than a blank page. The key discipline is *honesty*: the response must carry a flag indicating it is partial, so downstream consumers do not treat 92%-coverage as ground truth. A partial result presented as complete is worse than an error, because it silently corrupts whatever uses it.

Harvest/yield is the *wrong* move when the data is not decomposable or partial answers are dangerous. A bank balance computed from 92% of the transactions is not "92% correct" — it is *wrong*, and acting on it (approving a withdrawal) is a real error. A uniqueness check that only saw 92% of usernames can let a duplicate through. For these, there is no graceful middle: you need the whole answer or you must refuse. This maps cleanly onto the per-feature quadrants from the previous section — PA/EL features (carts, search, feeds) are exactly the ones where harvest/yield shines, and PC/EC features (balances, uniqueness, inventory) are exactly the ones where it is forbidden.

### The recover step: what happens when the partition heals

Brewer's 2012 paper adds a third phase people forget: **recovery.** A system that chose availability during a partition (served stale, accepted conflicting writes) must, when the partition heals, reconcile the divergent states and compensate for any mistakes made during the partition window. This is not free, and it is not automatic unless you designed for it.

The clean way is to use data types that converge by construction. **CRDTs** (conflict-free replicated data types) are structures that *provably* converge to the same state regardless of the order in which concurrent updates are applied — a grow-only counter, an add-wins set, a last-write-wins register with proper timestamps. If your cart is an add-wins set CRDT, two partitioned sides accepting different items merge automatically on heal with no lost items. The messier way is **compensating transactions**: detect the conflict after the fact and run business logic to fix it (refund a double-charge, cancel a duplicate order, merge two accounts). Compensation is application-specific, error-prone, and the source of a lot of "how did this customer get charged twice" incidents — but sometimes it is the only option, because not all business logic maps onto a CRDT.

The lesson: if you choose PA (availability during partition), you have signed up for the recovery problem, and you must design the reconciliation strategy *up front*. Choosing availability and then discovering at heal time that you have no way to merge the divergent states is the worst of both worlds — you stayed up during the partition and corrupted your data permanently afterward.

## 9. The timeline: one partition, two clients, two outcomes

To cement how the C-vs-A choice plays out in real time, walk through a single partition from start to heal, watching two clients with different consistency choices experience it differently.

![The same partition window shows one client stale data and another an error, then both reconcile](/imgs/blogs/cap-theorem-and-pacelc-8.webp)

The timeline above runs left to right through five moments:

- **t0 — healthy.** The link is up; `x = 1` is consistent across all replicas. Both an AP client and a CP client reading `x` see `1`. No tradeoff is visible yet, because there is no partition — exactly Brewer's point that without a partition you get both C and A.
- **t1 — write.** A client writes `x = 2` to the majority side. The write commits on the replicas that can form a quorum. The minority side has not yet received it.
- **t2 — partition.** The link drops. The minority side is now isolated: it cannot reach the majority and does not have `x = 2`. The system enters partition mode (recall: it *declares* the partition when a timeout fires).
- **t3 — the fork, two outcomes.** An **AP** client reading from the minority side gets the stale `1` and keeps going — it stays available, accepting that the answer is out of date. A **CP** client reading from the minority side gets an *error* — the minority side knows it cannot prove freshness, so it refuses, staying correct. Same partition, same instant, two different client experiences determined entirely by the C-vs-A choice baked into each read path.
- **t4 — heal and recover.** The link comes back. The minority side catches up, learns `x = 2`, and reconciles. If any AP-side writes happened during the partition, this is where CRDT merge or compensation runs. After recovery, everyone sees `x = 2` again and consistency is restored.

The figure makes vivid what the two-node proof states abstractly: during the partition window (t2–t4), there is no way to be both fresh *and* answering on the minority side. The AP read is answering but stale; the CP read is fresh-or-nothing but not answering. Outside the window, the question does not even arise. This is the whole of CAP, animated — and it is a tiny slice of the timeline. The vast majority of the line is t0-like: healthy, both properties available, and the only live tradeoff is the ELC latency tax you cannot see on this diagram because it operates at the millisecond scale.

## 10. The Spanner asterisk: "we beat CAP," carefully read

No CAP discussion is complete without Google Spanner, the system whose engineers published a paper provocatively titled "Spanner, TrueTime, and the CAP Theorem" and which is frequently cited as having "beaten" CAP. It did not beat CAP — CAP is a theorem, and you do not beat theorems — but understanding *exactly* what Spanner did and did not do is one of the most clarifying exercises in distributed systems.

![Spanner stays CP and reaches five nines by engineering partitions down, not by escaping the theorem](/imgs/blogs/cap-theorem-and-pacelc-9.webp)

The grid above lays out Spanner's machinery and its outcomes. A client transaction goes to a Paxos group, which commits on a majority. TrueTime — Google's globally-synchronized clock with bounded uncertainty — lets Spanner assign commit timestamps and do a "commit-wait" that guarantees external consistency (linearizability) across the entire globe. Critically, all of this runs over Google's *private, redundant wide-area network* with multiple independent paths between data centers. And the outcomes: Spanner **stays CP** (during a partition, the side that loses its Paxos majority refuses to serve), it achieves roughly **five nines of availability** (partitions almost never cause user-visible unavailability), and it provides **external consistency** (the strongest practical consistency model, stronger than plain linearizability because it respects real-time order across the whole system).

### What Spanner actually did

Eric Brewer, who wrote the analysis (he is at Google), is precise about it: Spanner is a **CP system**. Technically, when a partition occurs and a Paxos group cannot reach a majority, that group becomes unavailable — it chooses consistency over availability, exactly as CAP demands. Spanner does not violate the theorem. What Spanner does is make the *availability* number look like an AP system's by attacking partitions from a completely different direction: **it makes partitions vanishingly rare.**

Google does not rely on the public internet between Spanner replicas. It runs a private global network with redundant links, so that a single fiber cut or switch failure does not partition a Paxos group — there is another path. The partition rate is engineered down to the point where the CP penalty (unavailability during partitions) almost never manifests, because partitions almost never happen. Brewer's framing is that Spanner is *technically* CP but *effectively* CA — it behaves like a consistent-and-available system in practice, not by escaping the theorem, but by spending enormous engineering effort and capital to ensure the partition branch of the CAP fork almost never fires.

### Why this is the deepest lesson, not a loophole

The Spanner story is the perfect illustration of why "P is mandatory but partitions are rare" is the right mental model. CAP says: *during a partition*, choose C or A. Spanner chose C. CAP says nothing about *how often* you are in a partition. Spanner attacked that — the frequency — with infrastructure. The result is a CP system whose availability rivals an AP system, because the C-vs-A choice is only forced during partitions, and Spanner engineered partitions down to near-zero.

This is not a loophole; it is the theorem working exactly as stated, plus a massive investment in reducing partition frequency. You cannot copy Spanner by reading the CAP paper differently. You copy Spanner by building a private redundant global network and synchronizing atomic clocks across continents — which is to say, you mostly cannot copy Spanner, because almost no one has Google's network. But you *can* take the lesson: if you want consistency without giving up much availability, the lever is not a clever reinterpretation of CAP, it is reducing how often you are partitioned — better networks, redundant links, faster partition detection and recovery. Spanner did not change the rules; it changed the inputs.

TrueTime, for its part, is often misunderstood as the thing that "beat CAP." It did not. TrueTime's job is to enable external consistency by bounding clock uncertainty so Spanner can order transactions globally with a commit-wait. It is what lets Spanner be *consistent* efficiently; it is not what makes Spanner *available*. Availability comes from the redundant network and Paxos, not from the clocks. Conflating the two is the most common Spanner misconception, and now you can correct it at parties.

## 11. The Jepsen lesson: test, do not trust the marketing

Everything above is theory — definitions, proofs, classifications. Here is the reality check: **the PACELC label a vendor advertises is a claim, and claims are routinely wrong.** A database that says "we provide serializable isolation" or "we are linearizable" is making an assertion about its behavior under partitions, clock skew, node failures, and concurrent operations — and the only way to know if the assertion holds is to test it adversarially under exactly those conditions.

That is what [Jepsen](https://jepsen.io/) does. Founded by Kyle Kingsbury in 2013, Jepsen has become the de facto standard for distributed systems correctness testing. It works by running a real cluster, hammering it with concurrent operations, *injecting network partitions and clock skew while the operations run*, and then checking whether the recorded history is consistent with the consistency level the database claims. The results have been humbling for the entire industry.

### What Jepsen has actually found

The findings are not academic. Jepsen has repeatedly found that databases advertising strong guarantees violated them in practice:

- **MongoDB** had design flaws in its v0 replication protocol and implementation bugs in v1 that allowed the loss of majority-committed writes — writes the client was told were durable could vanish. This is the worst kind of bug: a PC/EC claim ("majority writes are safe") that was actually false, meaning users who chose MongoDB for safety did not get it.
- **MariaDB Galera Cluster** claimed an isolation level "between Serializable and Repeatable Read" and Jepsen found it failed to satisfy it — the advertised guarantee was simply not what the system provided.
- Numerous systems claimed to implement primitives (distributed locks, linearizable registers) that, under partition, did not hold — a vendor blog claimed distributed mutex capability the system did not actually have.

The positive flip side: Jepsen testing made the field better. CockroachDB, TiDB, and YugabyteDB worked extensively with Jepsen to *achieve* true serializability — the testing surfaced bugs, the vendors fixed them, and the resulting systems are genuinely stronger. After Jepsen publishes a report, vendors fix bugs. The process works. But it only works because someone tested instead of trusting.

### What this means for your PACELC decision

The practical implication is severe: **do not classify your database by its marketing PACELC label; classify it by its tested behavior.** When you read "PC/EC" on a vendor's website, what you actually know is "the vendor intends this to be PC/EC." Whether it *is* depends on whether the consensus protocol is correctly implemented, whether the clock assumptions hold, whether the partition-recovery logic is right — all of which are exactly the things Jepsen tests and frequently finds broken.

Concretely: before you bet money or correctness on a database's consistency guarantee, find its Jepsen report. If one exists, read it — note which version was tested, which guarantees held, which broke, and whether the bugs were fixed in a version you can deploy. If no Jepsen report exists for a database making strong consistency claims, treat the claim as unverified. And for your own system, *test the actual behavior* under partition: tools like Jepsen (or its underlying approach) let you inject partitions and verify your real configuration does what you think. The number of teams who discovered their "strongly consistent" setup was actually serving stale data — because of a misconfigured read concern, a missing quorum setting, or a driver default — is large, and every one of them could have caught it with a partition test before production did.

> The PACELC quadrant a database occupies is a property of its *implementation under failure*, not its *documentation under sunshine*. Jepsen exists because the gap between the two is wide, common, and expensive.

## Case studies from production

Theory survives contact with production unevenly. Here are concrete incidents — some widely documented, some composites of common patterns — where a CAP/PACELC misunderstanding caused real pain, and what the correct framing would have prevented.

### 1. The "CP system" that lost writes on failover

A team ran a primary/replica SQL setup with *asynchronous* replication and described it internally as "CP — we have strong consistency." The primary acknowledged writes the instant they hit its WAL, before the replica had them. One day the primary's instance died and the system failed over to the replica. The last 1.8 seconds of acknowledged writes — orders, in this case — were gone, because the replica never received them. The team was baffled: "we're CP, how did we lose writes?" The answer is that async replication is *not* CP; it is PA/EL. The primary chose latency (ack before replicating) and availability (don't block on the replica), which means a failover can lose recent writes. They had been describing their PACELC quadrant by aspiration, not by mechanism. The fix was to switch the critical-orders path to synchronous replication (PC/EC for that path) while leaving non-critical writes async — a per-feature choice. The lesson: your PACELC label is determined by your replication settings, not by what you call it. See [synchronous versus asynchronous replication](/blog/software-development/database/database-replication-sync-async-logical-physical) for exactly where the write durability boundary sits.

### 2. The shopping cart that refused to add items

A retailer migrated their cart from a custom service to a strongly consistent NewSQL database because "consistency is good." During a brief inter-region network blip, the database — correctly, per its PC/EC design — refused writes on the minority side rather than risk inconsistency. Customers in one region could not add items to carts for ninety seconds. Conversion dropped, and the incident review blamed the network. The real error was the consistency choice: a shopping cart is the textbook PA/EL feature (Dynamo was built for exactly this), and forcing it into PC/EC traded a non-problem (a slightly stale cart, trivially merged) for a real one (lost sales during partitions). The fix was to move the cart to an AP store with CRDT-based merge, keeping the PC/EC database for checkout and payment. The lesson: matching the feature to the quadrant matters more than picking a "good" database — there is no globally good quadrant, only a right one per feature.

### 3. The double-charged customers from an "available" payment path

A payments team chose an AP datastore for an idempotency-key table ("we need it always available so payments never block"). During a partition, both sides accepted the same idempotency key as new, because neither could see the other's record, and the same payment was processed twice on each side. On heal, last-write-wins silently discarded one of the two conflicting records — but both charges had already gone to the card processor. Customers were double-charged. Idempotency and uniqueness are PC/EC by nature: the entire point of an idempotency key is a global "has this been seen?" invariant, which is a linearizability problem. Choosing AP for it guaranteed the exact failure it was meant to prevent. The fix: move the idempotency table to a linearizable store and accept that, during a partition, some payments would be refused (PC) rather than double-processed. Refusing a payment is recoverable; double-charging is a chargeback and a trust problem.

### 4. The p99 that doubled when "consistency" was turned on

A team flipped their DynamoDB reads from eventually consistent to strongly consistent across the board, reasoning that "stronger is safer." Throughput per dollar halved (strong reads cost 2× the RCUs), and p99 latency on read-heavy endpoints jumped, because strong reads route to the primary replica instead of the nearest one. Nothing was partitioned; this was pure ELC tax, paid on every read, most of which did not need recency at all — they were rendering product descriptions and reviews, data that is fine slightly stale. The fix was to make consistency a per-query decision: strong reads only on the handful of endpoints that needed read-your-writes (the user's own just-edited profile), eventual everywhere else. The lesson: "strongest by default" is an expensive non-decision. The ELC choice should be made per query, against the actual recency requirement of that query's data.

### 5. The split-brain that two "available" nodes caused

A team ran a two-node cluster with automatic failover and no quorum/witness. A partition between the two nodes made each believe the other had died, so each promoted itself to primary. Both accepted writes. This is classic split-brain — two primaries, divergent data, an impossible merge. The root cause is a quorum mistake: with only two nodes, neither side can ever have a *majority* during a partition, so a system that stays available on both sides is necessarily choosing AP and inviting split-brain. CP requires a majority to make progress, which requires an odd number of voting members (three, or two plus a lightweight witness). The fix was to add a third voting member so that during a partition exactly one side has a majority and can safely stay primary while the other steps down. The lesson: PC/EC is not just a setting; it requires the *topology* (odd quorum) that makes a unique majority possible.

### 6. The "linearizable" database that wasn't, per Jepsen

A startup adopted a trendy distributed database advertising linearizable transactions and built their financial ledger on it. Months later, a reconciliation job found balances that did not add up — a textbook lost-update under concurrent transactions. The database's own Jepsen report, published *after* they adopted it, showed that the version they were running violated its linearizability claim under partition due to a consensus bug, later fixed in a newer release. They had taken the marketing label as ground truth. The fix was to upgrade to the Jepsen-blessed version and add their own partition tests to CI. The lesson — the Jepsen lesson — is that a consistency guarantee is a claim until tested, and betting a financial ledger on an untested claim is how you spend a quarter on reconciliation forensics.

### 7. The cross-region write that timed out under no failure at all

A team deployed a PC/EC database across three regions for "global consistency," then wired every write to require a global majority quorum. Writes from the region farthest from the quorum's center took 200+ ms — not because anything was broken, but because a global majority quorum means waiting for a round-trip to another continent on every write. Users in the distant region experienced the app as "slow," and the team chased phantom performance bugs for weeks. There was no partition, no failure — just the EC tax at intercontinental distance. The fix was to use per-region write leadership (so most writes stayed local) and reserve global consensus for the few truly global invariants, accepting weaker (session/bounded-staleness) consistency for region-local data. The lesson: EC's cost scales with the *distance* the quorum spans; global linearizability is the most expensive thing you can buy, and most data does not need it.

### 8. The dashboard that showed a blank page instead of 99% of the data

An internal metrics dashboard queried one hundred shards and rendered nothing if any shard was unreachable. During routine maintenance on two shards, the entire dashboard went blank, and on-call was paged for "metrics down" when 98% of the metrics were perfectly available. This is the harvest/yield failure: a binary up/down posture on data that is perfectly decomposable. The fix was to return results from reachable shards with a banner — "showing 98 of 100 shards; 2 temporarily unavailable" — preserving yield by degrading harvest, with an explicit honesty flag so nobody mistook the partial view for complete. The lesson: for decomposable, partial-answer-tolerant data, harvest/yield turns a hard outage into a soft, labeled degradation — but only if you build the partial-result path on purpose.

### 9. The username race that created duplicate accounts

A signup flow checked username availability with an eventually consistent read, then inserted the new account. Under normal load with multiple app servers, two concurrent signups for the same username both read "available" (neither saw the other's not-yet-replicated insert) and both committed, creating duplicate `alice` accounts that broke every downstream join keyed on username. No partition was involved — this was pure ELC staleness in the common case. The fix was a linearizable uniqueness check: either a unique constraint enforced by the database (which serializes the inserts) or a compare-and-set on a linearizable register. The lesson: uniqueness is a linearizability requirement even with a healthy network, and an eventually consistent "check then insert" is a race condition by construction. This is the ELC side of the per-feature matrix biting in the *Else* case, not the partition case.

### 10. The MongoDB write concern that wasn't durable

A team ran MongoDB with the driver's older default write concern (`w=1`, acknowledge after the primary has it, before replication), believed they had majority durability, and lost a batch of acknowledged writes when a primary failed before replicating. They had assumed MongoDB's PACELC label was fixed and safe, when in fact it is *entirely determined by the write/read concern you configure* — `w=1` is a PA/EL choice (fast, available, can lose writes on failover), while `w=majority` is the PC/EC-leaning choice. The fix was to set `writeConcern: majority` on the durable paths and verify with a failover test. The lesson: for configurable databases, the PACELC quadrant is a per-operation property you set explicitly, and the defaults are frequently the *fast* choice, not the *safe* one. Never assume; check the concern, and test the failover.

### 11. The clock skew that broke "external consistency"

A team built a Spanner-inspired system using ordinary NTP-synchronized clocks (not TrueTime's bounded-uncertainty hardware) and assumed they could assign globally-ordered timestamps. Under normal operation, NTP skew of tens of milliseconds caused transactions to be ordered inconsistently with real time — a later transaction occasionally got an earlier timestamp — violating the external consistency they thought they had. There was no partition; this was a clock-assumption failure in the *Else* case. The fix was to abandon timestamp-ordering on commodity clocks and use a consensus-based ordering (Raft log order) instead, which does not depend on synchronized clocks. The lesson: Spanner's external consistency rests on TrueTime's *bounded* uncertainty, a hardware investment most teams cannot replicate; copying the design without the clock guarantees copies the bug, not the property.

### 12. The retry storm that turned a blip into an outage

A PA/EL service degraded gracefully during a partition — exactly as designed, serving slightly-stale reads. But its clients were configured to retry aggressively on any latency increase, and the partition's elevated latency triggered a retry storm that overwhelmed the reachable replicas, turning a graceful degradation into a full outage. The CAP/PACELC choice was correct; the *client* behavior undid it. The fix was exponential backoff with jitter and circuit breakers, so that elevated latency during a partition did not amplify into self-inflicted overload. The lesson: your consistency/availability choice lives in a system that includes clients, timeouts, and retry policies, and a correct server-side PACELC choice can be defeated by a naive client. Partition handling is end-to-end, not just a database setting.

## When to reach for strong consistency, and when not to

Pulling the threads together, here is the playbook — the actual decision procedure, stripped of theory.

### Reach for strong consistency (PC/EC) when:

- The data enforces a **global invariant** that corruption makes unrecoverable: account balances, inventory counts, uniqueness constraints (usernames, emails, idempotency keys), anything where "two truths" is data corruption rather than a mergeable conflict.
- A **wrong answer is worse than a slow answer or no answer**: financial ledgers, authorization decisions, anything where acting on stale data causes real-world harm (double-spend, double-ship, granting access that was revoked).
- You need **read-your-writes** within a critical workflow: a user editing their own data and immediately reading it back must see their change, or the feature feels broken.
- The data is **not decomposable** into partial answers — you need the whole thing or nothing, so harvest/yield offers no graceful middle.
- You can afford the **latency tax on every operation** and you can build the **odd-quorum topology** (three+ voting members) that makes a unique majority possible during partitions.

### Skip strong consistency — choose PA/EL — when:

- Stale or conflicting answers are **cheaply reconcilable**: shopping carts (merge), recommendation feeds (stale is fine), view counters (eventual is fine), session data (mergeable), drafts and autosaves.
- **Availability beats correctness** for the feature: a cart that won't accept items loses sales; a "like" button that refuses during a partition annoys users for no real safety gain.
- The data is **decomposable** and partial answers are useful: search, analytics dashboards, aggregations — use harvest/yield to preserve yield by degrading harvest, with an honesty flag.
- You are **latency-sensitive in the common case** and the data does not need recency: product descriptions, reviews, static-ish content read far more than written.
- You have a **concrete plan for recovery** — CRDTs for automatic merge, or well-defined compensating transactions — because choosing PA means signing up for the reconciliation problem at heal time.

### And regardless of quadrant:

- **Make the choice per feature, per query** — not once for the whole database. Use tunable systems (Cassandra consistency levels, Cosmos DB levels, DynamoDB per-read consistency, MongoDB read/write concerns) deliberately, matching each feature to its quadrant.
- **Determine your quadrant by mechanism, not by name.** Async replication is PA/EL no matter what you call it; `w=1` is not durable no matter what you assumed.
- **Test under partition.** A consistency guarantee is a claim until you have injected a partition and verified the behavior. Read the Jepsen report; if there isn't one, write the test yourself.
- **Reduce partition frequency** if you want consistency without sacrificing much availability — the Spanner lesson. Better networks, redundant links, and fast detection/recovery shrink the window in which the CAP fork even fires.
- **Remember the everyday tax.** Partitions are rare; the latency-vs-consistency choice is paid on every healthy request. Optimize for the regime you are actually in 99.99% of the time.

CAP gets you into the room: it tells you, correctly, that during a partition you cannot have both consistency and availability, and that you must choose. PACELC keeps you in the room: it reminds you that the partition is rare, that the choice you make every day is latency versus consistency, and that the right decision is per-feature, mechanism-determined, and verified by testing. Use CAP to understand the bad moment. Use PACELC to design for all the other ones.

## Further reading

- Eric Brewer, ["CAP Twelve Years Later: How the 'Rules' Have Changed"](https://mwhittaker.github.io/papers/html/brewer2012cap.html) — the inventor's own correction of the "2 of 3" framing, plus the partition decision/recovery model and harvest/yield.
- Seth Gilbert and Nancy Lynch, ["Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services"](https://www.cs.princeton.edu/courses/archive/spr22/cos418/papers/cap.pdf) — the formal proof; consistency = linearizability, availability = every request to a live node responds.
- Daniel Abadi, ["Consistency Tradeoffs in Modern Distributed Database System Design: CAP is Only Part of the Story"](https://paperswelove.org/papers/consistency-tradeoffs-in-modern-distributed-databa-4a1ea3bd/) — the PACELC paper and its system classification.
- Eric Brewer, ["Spanner, TrueTime and the CAP Theorem"](https://research.google.com/pubs/archive/45855.pdf) — why Spanner is technically CP, effectively CA, and not a loophole.
- [Jepsen](https://jepsen.io/) — adversarial distributed-systems testing; the reality check on vendor consistency claims.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapters 5, 6, and 9 — replication, partitioning, and the cost of linearizability; the definitive practitioner treatment.
- [Cassandra tunable consistency](https://docs.datastax.com/en/cassandra-oss/3.0/cassandra/dml/dmlConfigConsistency.html), [DynamoDB read consistency](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadConsistency.html), and [Azure Cosmos DB consistency levels](https://learn.microsoft.com/en-us/azure/cosmos-db/consistency-levels) — the per-query/per-deployment ELC knobs in practice.
- Sibling posts on this blog: [consistency models, from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual), [database replication: sync, async, logical, physical](/blog/software-development/database/database-replication-sync-async-logical-physical), [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), and [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).

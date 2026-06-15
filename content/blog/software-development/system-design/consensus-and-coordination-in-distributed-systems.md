---
title: "Consensus and Coordination in Distributed Systems: When You Actually Need It"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn to recognize the handful of problems that truly require consensus, keep it off the hot path, and use coordination services and fencing tokens without the footguns that cause split-brain outages."
tags:
  [
    "system-design",
    "consensus",
    "coordination",
    "distributed-locks",
    "leader-election",
    "raft",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/consensus-and-coordination-in-distributed-systems-1.webp"
---

The fastest way to spot an engineer who has been burned by distributed systems is to watch their face when someone says "let's just have the services agree on it." A junior hears that as a small thing — a flag, a shared variable, a quick lookup. A senior hears the word *agree* and immediately starts pricing it: a round trip to a majority of nodes, an odd cluster size, a leader that can disappear, an election that freezes writes for seconds, and a failure mode where two nodes each believe they are in charge and quietly corrupt your data. Consensus is one of the most powerful tools in distributed systems and one of the most expensive, and the single biggest difference between a design that scales and one that falls over is *where* you decided you needed it.

Here is the claim this whole post defends: **most systems need consensus, but almost none of them need it where they think they do.** The common failure is not getting the consensus algorithm wrong — you will use etcd or ZooKeeper and they got the algorithm right years ago. The common failure is reaching for agreement on the *request path*, paying a quorum round trip on every read and write, and discovering at 10x scale that your p99 is pinned to the slowest network hop in your cluster. The senior move is the opposite instinct: assume you do *not* need consensus, prove that you do, and when you do, push it as far away from the hot path as you can — into a control plane that changes rarely, not a data plane that runs hot.

This post is the architect's decision layer, not the mechanism. The `database/` folder on this blog already derives the machinery: how [Raft elects a leader and commits a log from scratch](/blog/software-development/database/raft-consensus-from-scratch), how [Paxos and Multi-Paxos reach agreement](/blog/software-development/database/paxos-and-multi-paxos-explained), and how [quorums, anti-entropy, and read-repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) keep replicas honest. I will not re-derive any of that. My job is the layer on top: when a senior reaches for consensus, when they refuse it, how they keep it cheap, and how they avoid the outages that consensus is supposed to prevent but frequently causes. Figure 1 sets the stage — the price tag on a single consensus write — and everything after is about not paying it more often than you must.

By the end you should be able to do three concrete things: look at a feature and decide in minutes whether it needs consensus, a lease, a single-writer, or nothing; compute the latency cost of a consensus write across availability zones so you can defend or kill the design in a review; and build a distributed lock that does not corrupt your data when the lock holder pauses for garbage collection past its lease — the failure that has caused more quiet data corruption than almost anything else in this space.

## 1. What consensus buys you, and what it charges

Consensus is the problem of getting a set of nodes to **agree on a single value** even though some of those nodes may crash, messages may be delayed or reordered, and the network may partition — and to agree in a way that, once decided, the decision is *durable* and *the same* on every node that has heard about it. That is the entire job. A consensus protocol gives you a replicated, totally-ordered log: every participant applies the same entries in the same order, so any deterministic state machine fed that log ends up in the same state. Leader election, locks, configuration, membership — all of them reduce to "agree on the next entry in a log."

![A consensus write travels from the client to a leader that replicates it to followers and acknowledges only after a majority of nodes have persisted the entry](/imgs/blogs/consensus-and-coordination-in-distributed-systems-1.webp)

Figure 1 shows the mechanics that set the price. A client sends one write to the leader. The leader appends it to its log and ships it to the followers. Crucially, the leader does **not** acknowledge the client until a *majority* of the cluster has persisted the entry — in a three-node cluster, that means the leader plus one follower, two of three. The third node can be slow, garbage-collecting, or fully dead, and the write still succeeds, because two of three is a majority. That is the entire trick of fault tolerance: you wait for a majority, never for everyone, so any single node can fail without blocking progress.

Now read the same figure as a bill. Every consensus write is at minimum **one network round trip from the leader to a majority of followers** before the client hears "done." If your followers are in the same rack, that round trip is sub-millisecond and you barely notice. If they are spread across three availability zones for durability — which is the whole reason you replicated in the first place — that round trip is the inter-AZ latency, typically somewhere around 0.5 to 2 milliseconds within a region, and tens of milliseconds if you spread across regions. Consensus does not let you escape the speed of light; it makes you pay it on every committed write. That is the first thing a senior internalizes: **a consensus write is a majority round trip, charged per write, forever.**

The second charge is structural. To have a majority you need an **odd number of nodes**, and you need a majority to be reachable at all times for the system to make progress. Three nodes tolerate one failure (you keep two, a majority). Five nodes tolerate two failures. Even numbers buy you nothing: four nodes still only tolerate one failure (you need three for a majority, so losing two leaves you stuck) while costing more and adding latency, because you wait for more acks. This is why production consensus clusters are almost always three or five nodes and essentially never four or six. We will return to the odd-number rule because it is one of the most reliably misunderstood facts in the field.

The third charge is the one people forget until an incident: consensus has a **leader, and leaders fail**. When the leader dies, no writes can commit until the cluster *elects a new one*, and election takes time — an election timeout (often 150ms to a few seconds depending on configuration) plus a vote round trip. During that window the cluster is *write-unavailable*. It is still durable and still correct; it is simply not accepting writes. For a control-plane system that changes configuration a few times an hour, a two-second blip is invisible. For your request path, a two-second freeze every time a leader hiccups is an outage. This is the deepest reason to keep consensus off the hot path, and it is the spine of this entire post.

## 2. The problems that genuinely require consensus

Consensus is not a thing you sprinkle on a design for safety. It is the *unique* answer to a specific shape of problem: when multiple nodes must **agree on one truth that they all then act on**, and getting it wrong means two nodes act on different truths and corrupt shared state. There is a finite list of these. Memorize it, because everything *not* on this list is a place you should be actively trying to avoid consensus.

**Leader election.** Exactly one node must be "the leader" — the single writer, the coordinator, the thing that owns a partition. If two nodes both think they are the leader (split-brain), they both accept writes and you get divergence. Electing a unique leader, and ensuring the old one steps down before a new one steps up, is a consensus problem.

**Distributed locks and leases.** "Only one worker may run this job / hold this resource / write to this shard at a time." A correct distributed lock requires agreement on who holds it, durable across the failure of any one node, with a safe way to revoke it from a holder that has died or hung. We will spend a lot of this post here, because distributed locks are where the most expensive mistakes live.

**Configuration and metadata management.** The cluster topology, the shard-to-node assignment, feature flags that must be consistent fleet-wide, the schema version, the current set of valid certificates. This is metadata that is *small*, *read constantly*, and *written rarely*, and where a stale read can be catastrophic (two nodes disagree about which shard they own). This is the canonical, textbook *correct* use of consensus, and the reason etcd and ZooKeeper exist.

**Group membership and failure detection.** Which nodes are currently alive and in the cluster? Membership changes must be agreed upon, because if half the cluster thinks node X is in and half thinks it is out, quorum math breaks and you can get two majorities.

**Atomic commit across partitions.** A transaction that touches data on multiple shards must commit *everywhere or nowhere*. Two-phase commit gives you atomicity but blocks if the coordinator dies; making the *coordinator itself* fault-tolerant — so a coordinator crash does not leave the transaction stuck forever — is a consensus problem. This is exactly what Spanner does, layering [TrueTime and external consistency](/blog/software-development/database/spanner-truetime-and-external-consistency) on top of Paxos groups.

**Total ordering.** When every participant must apply a stream of operations in the *exact same order* — a replicated state machine, an ordered event log that drives correctness — you need agreement on the order itself, which is the consensus log.

![A decision tree that walks coordination needs through single-writer and lease options before arriving at full consensus only when strict shared agreement is unavoidable](/imgs/blogs/consensus-and-coordination-in-distributed-systems-2.webp)

Figure 2 is the decision tree a senior runs in their head, and it deliberately puts consensus at the *bottom* of the funnel. The first question is always: *can one writer own this key?* If a single node can be the authoritative writer for a given piece of state — say, by partitioning ownership so each shard has exactly one owner — you need no agreement at all, because there is nothing to disagree about. If you cannot get to a single writer, the next question is: *do you need strict, up-to-the-instant agreement, or is bounded staleness acceptable?* If you can tolerate a holder believing it is valid for a few more seconds than it strictly is (and you have fencing to make that safe — more on this later), a **lease** answers the need with one consensus operation amortized over thousands of local reads. Only when you need strict shared truth, written by anyone, read consistently by everyone, do you arrive at the bottom node: real consensus — and even then, the answer is to *use a coordination service*, not to write your own.

## 3. The senior instinct: avoid needing consensus

The strongest skill here is not implementing consensus well; it is *designing so you need less of it*. Every consensus operation is a coordination point, and coordination points are where latency, unavailability, and operational complexity concentrate. The architectural move that pays the most is restructuring the problem so the hot path needs *no* agreement.

The cleanest way to need no consensus is the **single-writer** pattern: partition your state so that exactly one node is the authoritative owner of each piece. If user 12345's data lives on shard 7, and shard 7 has exactly one owner at a time, then every write to that user goes to one place, gets serialized there trivially (it is a single process), and never needs to coordinate with anyone. There is no agreement problem because there is only one writer. The catch — and it is the whole game — is that *assigning* shard 7 to exactly one owner, and reassigning it safely when that owner dies, **is** a consensus problem. But notice what just happened: you moved consensus from "every write" to "every ownership change." Ownership changes happen when nodes join, leave, or fail — seconds to minutes apart. Writes happen millions of times a minute. You took consensus off the path that runs ten million times a day and put it on the path that runs ten times a day. That is the entire optimization, stated once.

This is the same instinct behind [partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) and [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning): the point of a partition is to create independent units that *do not need to coordinate with each other* on the hot path. The coordination is pushed to the rare event of assigning and moving partitions. When you read a system design and the hot path involves cross-shard agreement, that is a smell — a senior reviewer's eyes narrow and they ask "why does a single request need two shards to agree?"

The second move is to separate the **control plane** from the **data plane** explicitly. The control plane is where decisions that must be consistent fleet-wide are made — who owns what, which config is current, who is the leader. It runs consensus, it is small, it changes rarely, and it can afford to be slow because nobody is waiting on it in a tight loop. The data plane is where the actual requests flow — millions per second, latency-sensitive — and it must *never* block on consensus. The data plane reads a *cached* view of the control-plane decisions and acts on it locally. When the control plane changes something, it pushes the new view down (via a watch or a poll), and the data plane updates its cache. The request path touches no quorum. We will make this concrete with a figure later, but the principle is simple: *consensus belongs in the control plane; the data plane reads its decisions, it does not participate in them.*

## 4. Leases: how to read without paying for consensus every time

If you only remember one optimization from this post, remember the **lease**. A lease is a *time-bounded grant*: "you are the leader / you hold this lock / this cached value is valid — until time T." It is acquired through one consensus operation, and then for the entire duration of the lease, the holder can act *locally* without consulting anyone, because it has a promise that the grant is valid until T and that nobody else will be granted the same thing before T. The lease converts a per-operation consensus cost into a per-lease consensus cost, and a lease can cover thousands or millions of operations.

![A before-and-after comparison showing a per-request quorum round trip replaced by one lease grant that then serves many reads locally](/imgs/blogs/consensus-and-coordination-in-distributed-systems-3.webp)

Figure 3 is the optimization in its purest form. On the left, the naive design: every single read goes through a quorum, paying a cross-AZ round trip each time, pinning p99 to network latency and capping throughput at the rate the consensus group can commit. On the right, the lease design: a single consensus operation grants the node a lease — say, the right to serve reads of this data for the next 10 seconds — and for those 10 seconds, every read is served from local memory at sub-millisecond latency. If you serve 50,000 reads per second and the lease lasts 10 seconds, you just amortized one consensus operation across 500,000 reads. The quorum tax went from "per read" to "per half-million reads." That is a four-to-five-order-of-magnitude reduction in how often you pay for agreement.

This is exactly how real consensus systems make reads fast. A naive Raft implementation would route every read through the log to guarantee linearizability — but that means every read is a consensus operation, which is brutal. The standard optimization is the **leader lease** (sometimes called a leadership lease or a read lease): the leader holds a time-bounded lease establishing that it is *the* leader, and as long as the lease is valid, the leader can answer reads from its local state without contacting followers, because no other node can have become leader during the lease. etcd, Consul, and production Raft systems all do a version of this. The mechanism deep-dive lives in the [Raft post](/blog/software-development/database/raft-consensus-from-scratch); the architectural takeaway is that *leases are the primary tool for keeping consensus off the read path.*

Leases have one sharp edge, and it is the most important caveat in this entire post: a lease is a promise *about wall-clock time*, and wall-clock time is not something distributed nodes agree on reliably. If the lease holder's clock runs slow, or — far more commonly — the lease holder *pauses* (a stop-the-world garbage collection, a hypervisor descheduling the VM, a long disk stall) for longer than the lease duration, the holder can *wake up still believing it holds a lease that has actually expired*. Meanwhile, the system, seeing the lease expire, has granted it to someone else. Now two nodes both believe they hold the lease. That is the split-brain that fencing tokens exist to make safe, and section 8 is entirely about it. For now, hold this thought: **a lease without a fencing token is a correctness bug waiting for a GC pause.**

## 5. The quorum and latency tax, computed

Let us put real numbers on the cost, because "consensus is expensive" is the kind of vague statement that loses design reviews. The cost of a consensus write is dominated by one thing: the round-trip time from the leader to the *slowest node in the majority it must wait for*. In a three-node cluster, the leader must hear back from one follower (itself plus one = majority of three), so the cost is the round trip to the *faster* of its two followers. In a five-node cluster, the leader needs two followers (itself plus two = three of five), so the cost is the round trip to the *second-fastest* follower. Larger clusters are more fault-tolerant but each write waits for more acks, so latency does not improve and often worsens — another reason five is usually the ceiling.

#### Worked example: the latency cost of a consensus write across three AZs

You run a five-node etcd cluster for cluster metadata, spread across three availability zones in one region for AZ-failure tolerance: two nodes in AZ-a, two in AZ-b, one in AZ-c. Inter-AZ round-trip latency in this region is about 1 millisecond (a representative number; real values run roughly 0.3 to 2 ms within a region). Intra-AZ round trip is about 0.1 ms.

The leader sits in AZ-a. A write must reach a majority — three of five — which is the leader plus two followers. The cheapest two followers are: the other AZ-a node (0.1 ms, intra-AZ) and the nearest node in another AZ (1 ms, inter-AZ). The leader can commit as soon as those two have acked, so the commit latency is governed by the *slower* of those two acks: about **1 ms** for the network round trip, plus the time each follower takes to *fsync* the entry to disk before acking — and that fsync is often the real cost. A durable fsync on a decent NVMe SSD is on the order of 0.5 to 1 ms; on slower storage it can be several milliseconds. So a single committed write here lands around **1.5 to 2 ms** end to end, dominated by the inter-AZ round trip and the disk fsync.

Now scale it. If your design routes every user request through one such consensus write, your per-request floor is ~2 ms *of pure coordination*, before any application logic, and your throughput ceiling is roughly how many serialized commits the leader can push — call it a few thousand to low tens of thousands per second for a single Raft group, because the leader serializes the log. At 50,000 requests per second you have already blown past what one consensus group can commit, and you are now sharding the consensus group itself, which is a whole new project. Compare that to the lease design from section 4, where 50,000 reads per second cost you *one* consensus operation every 10 seconds. The same workload is either "impossible on one consensus group" or "trivial," depending entirely on whether you put consensus on the request path.

Two levers reduce the consensus write cost without changing the algorithm. First, **batching and pipelining**: instead of committing one entry per round trip, the leader batches many client writes into a single log append and replicates them together, and it pipelines (sends the next batch before the previous one is acked). A consensus group that commits one entry per round trip might commit a thousand entries per round trip when batched, raising throughput by orders of magnitude while keeping per-entry latency roughly flat. Every production consensus system does this; it is not optional at scale. Second, **co-locate the leader** with the workload that writes most. If 90% of writes originate in AZ-a, put the leader in AZ-a so those writes pay intra-AZ latency to the leader and only the replication leg crosses AZs. A leader in the wrong AZ doubles your latency by adding a client-to-leader cross-AZ hop on top of the replication hop.

## 6. Why three or five, never even — and what a partition really does

The odd-number rule deserves its own treatment because it is misunderstood constantly, and the misunderstanding produces real outages. The rule is: **a consensus cluster needs an odd number of nodes, and progress requires a strict majority (more than half) to be mutually reachable.**

Here is the reasoning from first principles. Fault tolerance comes from requiring a *majority* to agree, because any two majorities of the same set must *overlap in at least one node* — and that overlapping node prevents two conflicting decisions from both being committed (it would have to vote for both, which it refuses). For "majority" to be well-defined and for the overlap guarantee to hold, you compare against half the cluster. With three nodes, a majority is two, so you tolerate one failure. With five, a majority is three, so you tolerate two failures. With four nodes, a majority is *three* — so four nodes tolerate only *one* failure, exactly like three nodes, but with more machines, more replication traffic, and higher write latency (you wait for more acks). The even count buys nothing and costs more. Five tolerates two failures; six still tolerates only two (majority is four). So you go three, five, seven — and seven is already rare because the latency of waiting for a majority of seven usually is not worth tolerating three failures.

![A network partition splits a five-node cluster so only the three-node majority side can elect a leader and accept writes while the two-node minority refuses](/imgs/blogs/consensus-and-coordination-in-distributed-systems-9.webp)

Figure 9 shows the payoff: what a partition actually does, and why the majority requirement is precisely what prevents split-brain. When a five-node cluster partitions, say into a three-node side and a two-node side, only the three-node side has a majority. It can elect a leader and keep accepting writes. The two-node side *cannot* form a majority — it has only two of five — so it *refuses* to elect a leader and *refuses* to accept writes. It is unavailable, deliberately, and that is the correct behavior. The minority makes no progress precisely so that it cannot diverge from the majority. When the partition heals, the two minority nodes catch up from the majority's log. This is the [CAP trade-off](/blog/software-development/database/cap-theorem-and-pacelc) made concrete: during a partition, the minority side chooses consistency over availability by refusing to serve. The cost is brutal but the alternative — letting both sides accept writes — is split-brain and data corruption.

This is also why you must think about *where* your nodes live relative to your failure domains. If you run three nodes and put two of them in the same AZ, then losing that one AZ loses your majority and the whole cluster goes write-unavailable — you bought three-node fault tolerance but deployed it as one-AZ fault tolerance. The right layout for a three-node cluster is one node per AZ across three AZs, so any single AZ failure leaves a two-node majority. For five nodes across three AZs, the standard layout is 2-2-1, so losing any single AZ leaves at least three nodes (a majority). The rule that follows: **never let a single failure domain hold a majority of your consensus nodes**, or that failure domain *is* your availability.

A special case that bites surprisingly often: the **two-data-center problem**. People want a consensus cluster to survive losing an entire data center, so they put half the nodes in each of two data centers. But with two data centers and an even split, losing *either* data center loses exactly half the cluster — never a majority — so the survivor cannot make progress. You cannot achieve consensus-based survival of one data center with only two data centers; you need a *third* location to hold the tie-breaking node (even a tiny one). This is why etcd and ZooKeeper deployments that must survive a DC failure are always spread across *three* DCs or three AZs, never two. The third location does not need to be big; it needs to exist so a majority can form when one of the two main sites is gone.

## 7. Use a coordination service — almost never roll your own

When you have established that you genuinely need consensus, there is one more decision, and for nearly everyone it has the same answer: **use a battle-tested coordination service — etcd, ZooKeeper, or Consul — and do not implement consensus yourself.** Consensus is the kind of problem that looks tractable in a design doc and is a multi-year correctness nightmare in practice. The algorithms have subtle invariants (term numbers, log matching, commit rules, membership-change safety) that production systems took years to get right, and the failure modes are the worst kind: silent, rare, and corrupting. The companies that wrote their own consensus — Google, Amazon, the database vendors — did it because consensus *is* their product, and they staffed teams of distributed-systems specialists on it for years. You are not in that situation. Your situation is "I need leader election for my workers," and the answer to that is "point them at etcd."

What these services actually give you is a small, consistent, watchable key-value store backed by consensus. You get atomic compare-and-swap, ephemeral keys that vanish when a client's session dies (the foundation of failure detection), and watches that notify you when a key changes (the foundation of pushing control-plane decisions to the data plane). On top of those primitives you build leader election, locks, membership, and config — without ever touching a consensus algorithm yourself. Here is leader election in etcd, which is about as much code as you should ever write for this:

```go
package main

import (
	"context"
	"log"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

func main() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{"10.0.1.10:2379", "10.0.1.11:2379", "10.0.1.12:2379"},
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	// A session ties our leadership to a lease with a TTL. If this process
	// dies or stalls past the TTL, etcd revokes the lease and the election
	// key disappears, so another candidate can win. This is the fencing of
	// liveness: leadership is bounded by a lease we must keep renewing.
	session, err := concurrency.NewSession(cli, concurrency.WithTTL(10))
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	election := concurrency.NewElection(session, "/services/scheduler/leader")

	// Campaign blocks until THIS node becomes leader. Only one node returns
	// from Campaign at a time across the whole fleet — that is the consensus
	// guarantee we are buying, expressed as one blocking call.
	if err := election.Campaign(context.Background(), "node-a"); err != nil {
		log.Fatal(err)
	}
	log.Println("I am the leader; safe to run the singleton work")

	// Do leader-only work here. The session auto-renews the lease in the
	// background; if renewal fails (we are partitioned from the quorum),
	// the session closes and we must STOP doing leader work immediately.
	runScheduler(session.Done())
}

func runScheduler(stepDown <-chan struct{}) {
	for {
		select {
		case <-stepDown:
			log.Println("lost leadership (lease lapsed); stopping leader work")
			return
		default:
			// ... do one unit of singleton work, then loop ...
			time.Sleep(time.Second)
		}
	}
}
```

The subtle but critical line is the `select` on `session.Done()`. The consensus service can *tell you that you have a unique leadership*, but it cannot *stop your process from running leader-only work after you have lost it*. If you get partitioned from the quorum, your session lease lapses, etcd removes your leadership key, and someone else wins the election — but your process is still running. If it keeps doing leader work, you now have two leaders. The contract you must honor is: **the moment your session is done, stop doing leader work, immediately, before the next action.** The service handles agreement; honoring the result is your job, and it is the job most people get wrong.

ZooKeeper offers the same shape with a different vocabulary: ephemeral sequential znodes, where each candidate creates a sequenced ephemeral node and the lowest sequence number wins, watching the node just below it. Consul layers a friendly HTTP/DNS API and service discovery on top of its own Raft implementation, with sessions and health checks driving lock liveness. The choice between them is mostly ecosystem fit — etcd if you live in Kubernetes (it is the Kubernetes datastore), ZooKeeper if you are in the Kafka/Hadoop world (though Kafka has moved to its own Raft-based KRaft to *remove* the ZooKeeper dependency), Consul if you want service discovery and consensus in one box. All three are correct; none of them is something you should reimplement.

### Raft vs Paxos vs Multi-Paxos vs ZAB, at the decision level

You will see four names thrown around — Raft, Paxos, Multi-Paxos, and ZAB — and as an architect you should know what *decision* (if any) they represent, without needing to re-derive them (the [Raft](/blog/software-development/database/raft-consensus-from-scratch) and [Paxos and Multi-Paxos](/blog/software-development/database/paxos-and-multi-paxos-explained) deep-dives do that). The short version: they are all solving the same problem — a replicated, totally-ordered log via majority agreement — and they all converge on the same operational shape (a stable leader replicating to followers and committing on a majority). The differences that matter to *you* are not the proofs; they are understandability, the library you inherit, and a couple of operational quirks.

**Paxos** is the original, the one with the correctness proof everyone cites, and it solves *single-value* consensus — agreeing on one value. It is notoriously hard to understand and even harder to implement correctly, and a bare single-value Paxos is not directly useful, because real systems need a *log* of many values, not one value. **Multi-Paxos** is the practical extension: run Paxos repeatedly to agree on a sequence of values (a log), with the key optimization that once a leader is stable, it skips the first phase of Paxos and just streams entries — which makes it operationally identical to Raft (a stable leader, steady-state single-round-trip commits). The catch with Multi-Paxos is that the original papers left the practical details (leader election, membership changes, log compaction) underspecified, so every implementation filled the gaps differently, and "Multi-Paxos" describes a family of subtly different systems rather than one spec.

**Raft** was designed explicitly to be *understandable* — same guarantees as Multi-Paxos, but with leader election, log replication, and membership changes specified as one coherent, teachable algorithm. That is its entire value proposition as an architect's choice: not that it is faster or more correct than Multi-Paxos (it is neither, materially), but that your team can *read it, reason about it, and operate it* — and that there are now many solid open-source implementations (etcd's, HashiCorp's, TiKV's). When you pick "a Raft library," you are buying understandability and ecosystem, which for everyone outside the database-vendor club is the right thing to optimize. **ZAB** (ZooKeeper Atomic Broadcast) is ZooKeeper's own protocol, predating Raft, in the same leader-based family, tuned for ZooKeeper's primary-backup model where the leader broadcasts state changes; you do not choose ZAB independently — you choose it implicitly when you choose ZooKeeper.

| Protocol | What it is | Decision relevance to you |
|---|---|---|
| **Paxos** | Single-value consensus, the original proof | Rarely a direct choice; the theoretical foundation underneath the rest |
| **Multi-Paxos** | Paxos run as a log with a stable leader | What database vendors built on; powerful but underspecified, many variants |
| **Raft** | Multi-Paxos-equivalent, designed to be understandable | The default choice when you need a consensus library you can reason about and operate |
| **ZAB** | ZooKeeper's broadcast protocol | You get it by choosing ZooKeeper, not as a standalone decision |

The architect's takeaway is liberating: **at the decision level, these are nearly the same choice.** They all give you a leader-based replicated log committed on a majority, with the same latency profile (a majority round trip per committed batch) and the same failover behavior (a write-unavailability window during election). So you do not agonize over Raft-versus-Multi-Paxos on theoretical grounds; you choose the *system* (etcd, ZooKeeper, Consul, or a database that embeds one) for its operability and ecosystem, and you inherit whichever protocol it ships. The only time the protocol choice is yours directly is if you are *building* a consensus-backed system — and section 7's opening sentence already told you what to do about that: almost never do that.

## 8. Distributed locks done right: the fencing token

Distributed locks are where I have seen the most expensive, hardest-to-debug corruption, so I am going to be exact. The naive mental model of a distributed lock is "acquire, do work, release," like a mutex. That model is *wrong in a way that silently corrupts data*, and the reason is the same lease problem from section 4: the lock holder can pause.

Here is the failure, concretely. Worker A acquires a lock with a 30-second lease and starts writing to shared storage. Partway through, A's JVM enters a stop-the-world garbage collection pause — or the VM gets descheduled by the hypervisor, or a disk write stalls — and A is frozen for 40 seconds. From the lock service's point of view, A's lease expired at 30 seconds; the lock is now free, so worker B acquires it and starts writing. At second 40, A wakes up. *A does not know it paused.* As far as A is concerned, no time has passed; it still holds the lock; it resumes its write to shared storage — on top of B's work. Now two writers have both written, neither knew about the other, and your data is corrupt. No exception was thrown. No log says "lock contention." The corruption is silent.

![A before-and-after comparison where a lease-only lock lets a paused holder overwrite the new holder while a fencing token rejects the stale write](/imgs/blogs/consensus-and-coordination-in-distributed-systems-7.webp)

Figure 7 contrasts the broken design with the fix. On the left, lease-only: A pauses past its lease, B acquires and writes version 2, A wakes and writes version 1 over version 2 — corruption. The lock service did nothing wrong; the *lock alone cannot prevent this*, because the lock lives in one service and the *write* happens in a different service (the storage), and the storage has no idea the lock expired. The fix, on the right, is the **fencing token**: every time the lock is granted, the lock service returns a *monotonically increasing number* — a token. A gets token 33. When A's lease expires and B acquires, B gets token 34 (strictly greater). Now the rule: **every write to the protected storage must carry the token, and the storage must reject any write whose token is less than the highest token it has already seen.** When A wakes and tries to write with token 33, the storage has already accepted B's token-34 write, so it *rejects* A's stale token-33 write. A is fenced out. The corruption cannot happen.

![A fencing-token lock pipeline where the lock grant returns a monotonic token that the storage layer validates before accepting any write](/imgs/blogs/consensus-and-coordination-in-distributed-systems-6.webp)

Figure 6 traces the token end to end, because the critical insight is *where the check happens*. The token is issued at lock acquisition, carried with every write, and *validated by the storage layer*, not by the lock service. This is the part people miss: the lock service cannot fence the write, because it is not in the write path. The *storage* must do the fencing, by remembering the highest token it has accepted and rejecting anything older. That requires the storage to participate — which is why fencing is easy when you control the storage (it just tracks a max token per key) and harder when the storage is a dumb blob store that does not support conditional writes. If your storage supports a compare-and-set or conditional-put on a version number, you have fencing for free: use the token as the version. If it does not, you either add a thin layer that does, or you accept that your lock is *advisory* (it reduces contention but does not guarantee single-writer safety) and design so that double-writes are merely wasteful, not corrupting.

Here is the lock-with-fencing pattern in Python against etcd, where the lease *is* the token source:

```python
import etcd3

client = etcd3.client(host="10.0.1.10", port=2379)

def run_with_fencing(resource_key, lease_ttl=10):
    # Acquire a lease; its ID is monotonically increasing within a session
    # and serves as (the basis for) our fencing token. We treat the lease's
    # mod_revision on the lock key as the strictly-increasing token.
    lease = client.lease(lease_ttl)
    lock_key = f"/locks/{resource_key}"

    # Atomic acquire: only succeeds if the key does not already exist.
    success, _ = client.transaction(
        compare=[client.transactions.version(lock_key) == 0],
        success=[client.transactions.put(lock_key, "held", lease=lease)],
        failure=[],
    )
    if not success:
        raise RuntimeError("lock already held")

    # The mod_revision is etcd's global, monotonically increasing revision
    # number at the moment we wrote the key. It is strictly greater for any
    # later acquisition, so it is a valid fencing token.
    meta = client.get(lock_key)[1]
    token = meta.mod_revision

    try:
        do_protected_work(resource_key, token)   # carries the token to storage
    finally:
        lease.revoke()   # releasing early is an optimization, not a correctness need

def do_protected_work(resource_key, token):
    # The STORAGE enforces the fence. We pass the token; the storage rejects
    # any write whose token is below the max it has already accepted. If the
    # conditional update fails, WE PAUSED past our lease and must abort.
    ok = storage_conditional_write(resource_key, token, payload=b"...")
    if not ok:
        raise RuntimeError("fenced out: a newer holder exists; aborting write")
```

The line that matters is `storage_conditional_write` returning false, and the code *aborting* rather than retrying or ignoring it. A false return means "a writer with a higher token has been here; you are stale." The only correct response is to stop. If your code logs and continues, you have rebuilt the broken design with extra steps. The senior rule for distributed locks: **a lock is not a mutex; it is a lease plus a fencing token, and the fence is enforced at the storage, or it is not enforced at all.**

This brings us to the **Redlock controversy**, which every architect should know the shape of. Redlock is an algorithm proposed by Redis's author for distributed locking across multiple independent Redis instances: acquire the lock on a majority of N Redis nodes, and you hold the lock. Martin Kleppmann published a widely-read critique arguing that Redlock does not provide the safety guarantees people assume, for exactly the reasons in this section: it relies on bounded clock drift and bounded process pauses, neither of which you can assume, and — critically — it provides *no fencing token*, so a GC pause can still let two clients both believe they hold the lock and both write. Salvatore Sanfilippo (Redis's author) responded defending Redlock's assumptions for certain workloads. The debate is worth reading in full, but the architect's takeaway is the durable lesson: **if your lock protects correctness (not just efficiency), it needs a fencing token, and an algorithm without one is at best an optimization that reduces contention, never a guarantee of single-writer safety.** Use Redlock — or any lock — for *efficiency* (avoid two workers doing the same expensive job) freely; do not use any unfenced lock for *correctness* (prevent two workers from corrupting shared state) ever.

## 9. Leader election and split-brain prevention in practice

Leader election is the most common reason teams reach for a coordination service, and it has its own set of footguns beyond the basic mechanism. The mechanism — a unique leader chosen by majority vote, stepping down when its lease lapses — you get from etcd or ZooKeeper. The *footguns* are operational and they are where systems actually break.

![A timeline of a leader failover where the cluster is write-unavailable from the leader crash until a new leader is elected and commits its first entry](/imgs/blogs/consensus-and-coordination-in-distributed-systems-4.webp)

Figure 4 lays out the failover timeline, and the shape is the thing to internalize: there is a **write-unavailability window** from the moment the leader crashes until a new leader commits its first entry. Walk it left to right. At T+0 the leader crashes. The followers do not know instantly — they wait for an *election timeout* to elapse without hearing a heartbeat, typically a few seconds (configurable; shorter timeouts detect failure faster but risk spurious elections from a brief network blip). Around T+5s a follower becomes a candidate and requests votes. The majority grants them. By T+5.3s the new leader has committed its first entry and writes resume. For those ~5 seconds, *no writes committed*. The cluster was correct and durable the entire time — it just was not accepting writes. The lesson: **leader-based systems have a built-in unavailability window on every failover, measured in seconds, and your SLO must account for it** — which is yet another reason not to put leader-dependent consensus on the request path of a low-latency service.

The deeper footgun is **split-brain on the leadership boundary**, and it is more subtle than "two nodes both win an election" (which the majority rule prevents). The real split-brain happens at the *step-down* edge. The old leader gets partitioned from the quorum but *does not know it yet*. From its perspective, it is still the leader; it has not received a "you are deposed" message, because the partition is exactly what is blocking that message. Meanwhile the majority side, not hearing the old leader's heartbeats, elects a new one. For a window, *two nodes both believe they are the leader.* The majority guarantees only one of them can *commit through the consensus log* — but if the old leader is doing work *outside* the log (writing to external storage, calling an API, holding a lock on a resource), nothing stopped it. This is the same disease as the unfenced lock, and it has the same cure: **the old leader must fence itself.** The standard technique is the *leader lease with a strict step-down*: a leader is only the leader while it holds an unexpired lease, and it must *stop all leader work* the instant its lease cannot be renewed — proactively, before the lease's clock runs out, accounting for clock skew. The etcd code in section 7 does exactly this with the `session.Done()` channel: when the session lapses, leader work *must* stop. A leader that keeps acting after its lease lapses is the most common cause of split-brain corruption in practice, and it is almost always a bug in the *application's* honoring of the step-down signal, not in the consensus service.

A second operational hazard is the **flapping leader**. If your election timeout is too aggressive relative to your network's normal jitter, a brief latency spike causes followers to time out, start an election, depose the perfectly-healthy leader, and then the deposed leader (now back) triggers another election — and the cluster spends its time electing leaders instead of doing work. The fix is to tune election timeouts to be comfortably larger than your p99 network round trip plus normal GC pauses (often a few seconds, with randomized jitter so candidates do not all time out simultaneously and split the vote). And a third: **the thundering herd on failover.** When the leader dies, every client that was talking to it reconnects to the new leader at once, plus the new leader has to catch up its own state — a load spike exactly when the system is most fragile. Bound it with backoff and jitter on client reconnects, and keep followers warm enough that promotion is cheap.

## 10. Control plane vs data plane: the architecture that scales

Everything above converges on one architectural pattern, and it is the most important diagram in this post. Consensus is wonderful for *deciding things that change rarely and must be consistent*, and ruinous for *the path that runs hot*. So you split your system along exactly that seam.

![A layered stack showing consensus confined to the control plane while the data plane serves requests from a locally cached view with no quorum round trip](/imgs/blogs/consensus-and-coordination-in-distributed-systems-8.webp)

Figure 8 shows the split. The **control plane** at the top runs consensus — etcd, say — and holds the small, slowly-changing truth: the shard map, the current leader of each partition, the active config, the membership list. Writes to it are rare (a node joins, a shard moves, a flag flips) and can afford the quorum round trip. Below it, the control plane *pushes its decisions down* via watches or short-interval polls, so every data-plane node holds a **locally cached view** of the current truth. At the bottom, the **data plane** — the services handling actual user requests — reads from its local cache and acts. The request path touches *no quorum*. A request comes in, the service consults its cached shard map (in-memory, sub-microsecond), routes to the right owner, done. The p99 of the request path is governed by application logic and storage, not by consensus, because consensus is nowhere near it.

This is not a theoretical pattern; it is how the systems you use are built. Kubernetes is the cleanest example: etcd is the consensus-backed control plane holding the entire cluster state, but the *data plane* — the kubelets running your pods, the kube-proxy routing your traffic — reads a cached, watched view of that state and acts locally. Your pod's network packets do not make a quorum round trip to etcd; that would be absurd. The control plane decides *what should run where* (rarely, consistently), and the data plane *runs it* (constantly, locally). When the control plane is unavailable — etcd down — the data plane keeps serving from its last known view; you just cannot make *changes*. That degradation mode is the whole point: a control-plane outage stops the system from *changing*, not from *running*. Design your systems so a consensus outage degrades you to "frozen but serving," never to "down."

The discipline this imposes is a question you ask of every consensus interaction: *is this on the request path?* If yes, you are almost always wrong, and the fix is a lease or a cached control-plane view. If no — if it is a rare decision that the request path reads a cached copy of — then consensus is exactly right, and you should not feel bad about the quorum cost, because you pay it rarely. The cost of consensus is acceptable in inverse proportion to how often you pay it.

## 11. Trade-offs: the coordination decision matrix

Time to make the decision explicit, the way you would in a design review. For any coordination need, there are four candidate mechanisms, in increasing order of cost and decreasing order of how often you should reach for them: **avoid entirely** (restructure so no coordination is needed), **single-writer** (partition ownership), **lease** (time-bounded grant, amortized consensus), and **full consensus** (per-decision agreement). The senior walks them in that order and stops at the first one that meets the need.

![A matrix mapping coordination needs to single-writer, lease, and consensus mechanisms, showing most needs are met before full consensus](/imgs/blogs/consensus-and-coordination-in-distributed-systems-5.webp)

Figure 5 maps the common coordination needs to the three active mechanisms (avoid-entirely is the implicit zeroth column — always try it first). Read it as a guide, not gospel: *leader election* genuinely needs consensus (a single-writer cannot elect itself; a lease helps liveness but the election is the consensus). *Distributed locks* are best served by a *lease plus fencing*, with full consensus underneath only for the lease grant. *Config and metadata* is the textbook consensus use — small, read-mostly, must be consistent. *Group membership* needs consensus to agree on who is in. *Atomic commit* across partitions needs consensus for the fault-tolerant coordinator. And *hot-path ordering* — if you can get it from a single-writer (one partition owner serializes its own writes), do that, because routing every ordered operation through consensus is the expensive trap this whole post warns against. The matrix's real message is the column where most needs land: not the consensus column.

Here is the same decision as a written matrix, with the gain, the cost, and the condition that selects each mechanism — the artifact you would actually put on the whiteboard:

| Mechanism | What you gain | What you pay | When it wins |
|---|---|---|---|
| **Avoid entirely** | Zero coordination cost; nothing to fail | Must restructure the problem (partition, make commutative, accept eventual) | Whenever the data can be partitioned so writers never overlap, or operations commute |
| **Single-writer** | No agreement on the hot path; trivial serialization | Consensus moves to ownership changes; one owner is a throughput ceiling and a failure point | One node can own each key, and ownership changes are rare (joins/leaves/failures) |
| **Lease** | One consensus op amortized over millions of local actions; fast reads | Must handle clock skew and pauses; needs fencing tokens to be safe | Bounded staleness is acceptable and the protected action can be fenced at the storage |
| **Full consensus per op** | Strict, linearizable agreement on every decision | A majority round trip per operation; leader failover unavailability; odd-node fleet | The decision must be strictly agreed every time and is rare (config, membership, commit) |

The discipline that makes this a senior matrix rather than a junior one is the **condition column**. Anyone can list mechanisms. The skill is naming the *condition under which each wins* so the choice is mechanical given the constraints, and naming the *cost* so you never recommend consensus without admitting its price. Notice the asymmetry: the cheap mechanisms (avoid, single-writer) win in the *common* case, and full consensus wins only for the *rare, must-be-strict* case. That asymmetry is the whole philosophy. Cross-reference this with the [consistency models guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) and the [replication strategies post](/blog/software-development/system-design/replication-strategies-and-their-failure-modes), because the consistency you promise and the replication you choose are downstream of where you put coordination.

#### Worked example: does this feature need consensus, a lease, or a single-writer?

You are designing a system where each user has a "current active session device" — at most one device can be the active session for a user at a time, and switching devices must revoke the old one. A junior reaches for a distributed lock per user. Let us walk the matrix.

*Avoid entirely?* Can we restructure so there is no coordination? The constraint is genuinely "at most one active device per user," which is a single-writer constraint on a per-user basis — so we cannot fully avoid it, but we *can* partition it. *Single-writer?* Yes: partition users across shards, and each user's "active device" record has exactly one owning shard. Within that shard, a single process serializes "set active device" operations for a given user trivially — there is no cross-node agreement, because one shard owns the user. Setting device B active and revoking device A is a single local transaction on the owning shard. *Do we need a lease or full consensus?* Only for the *shard ownership* itself — which shard owns which users, and reassigning a shard when its owner dies. That is rare (a failover), so it goes to the control plane via consensus, and the data plane reads the cached shard map.

Result: the feature that looked like "a distributed lock per user" (millions of consensus operations) becomes "a single-writer per shard" (zero hot-path consensus) plus "consensus only on shard ownership changes" (a handful per day). The active-device switch is now a local transaction at sub-millisecond latency, and you pay consensus only when a shard moves. That is the difference between a design that scales to ten million users and one that melts. The matrix did the work: it pushed you down the cheap branches before you ever reached the expensive one.

## 12. Stress-testing the design

A design is only as good as its behavior under failure, so let us break the control-plane/data-plane architecture deliberately and see what holds. This is the part of a design review where a senior earns their title: posing the failures *before production does*.

**What breaks at a network partition?** The control plane (consensus) splits into a majority and a minority. The majority keeps the truth and can accept config changes; the minority refuses (figure 9). The *data plane* on the minority side keeps serving requests from its last cached view — it does not need the control plane to handle requests, only to *change* configuration. So during a partition: the minority data plane serves stale-but-consistent config (it cannot see changes, but it is internally consistent), and the majority can still make changes. The degradation is "minority cannot change config," not "minority is down." That is exactly the property you want. The thing that *would* break is if you had put consensus on the request path: then the minority data plane would be fully down, because every request would need a quorum it cannot reach. The architecture's resilience to partition comes entirely from having moved consensus off the request path.

**What breaks at a leader failover?** Per figure 4, the control plane has a multi-second write-unavailability window — no config changes can commit during the election. But the data plane keeps serving from cache throughout, because it does not write to the control plane on the request path. So a control-plane leader failover is invisible to user requests; it only delays *config changes* by a few seconds. If a config change happened to be in flight, it either committed before the crash (and is durable) or did not (and the client retries after the new leader is up). No corruption, brief change-latency. Compare to the broken design where the request path hits consensus: there, every failover is a multi-second *outage* of user-facing traffic.

**What breaks when a lock holder pauses past its lease?** This is the GC-pause scenario from section 8, and it is the one that *does* break unless you built fencing. Without fencing tokens, a paused holder wakes and corrupts data — silently. With fencing tokens enforced at the storage, the paused holder's stale-token write is rejected and the holder aborts. So the stress test for any lock-based design is: *trace what happens when the holder freezes for longer than the lease.* If your answer is "the lock service revokes the lease so it is fine," you have failed the test — the lock service revoking the lease does *nothing* to stop the holder's in-flight write. The only correct answer is "the storage rejects the stale token." If you cannot give that answer, your lock is advisory, and your design must tolerate double-writes (make them idempotent and commutative) or it is unsafe.

**What breaks at 10x scale?** The single consensus group is the ceiling. One Raft/Paxos group serializes its log through one leader, so it commits a bounded number of entries per second regardless of how many nodes you add (more nodes make it *slower*, not faster, because the leader waits for more acks). If your control plane's *change rate* grows past what one group can commit — many thousands of writes per second — you must *shard the consensus itself* into multiple independent groups, each owning a slice of the keyspace, so the groups commit in parallel. This is what etcd's and ZooKeeper's scaling stories run into, and why systems like CockroachDB and Spanner run *many* small Raft/Paxos groups (one per data range) rather than one big one. The architectural rule: a single consensus group does not scale with node count; it scales by *partitioning into many groups*. If your design routes all coordination through one group, your 10x failure mode is "the one group saturates," and the fix is to shard the coordination by key — which, you will notice, is the single-writer-per-partition pattern applied to the consensus layer itself.

**What breaks at a hot key?** If one key (one shard owner, one lock, one config entry) is read or contended far more than others, that one consensus group or that one single-writer becomes a hotspot while the rest of the cluster is idle. The fix mirrors the hot-key fixes elsewhere in distributed systems: for reads, a lease lets the owner serve them locally without consensus (so read hotness costs nothing extra); for writes, you must either accept the single-writer ceiling on that key or restructure so the hot key is split (which only works if the operations on it can be made independent). A hot *lock* is a design smell — it usually means you are serializing something that should be partitioned. The senior question is always "why is everything funneling through one coordination point?", and the answer is usually "because we did not partition the thing under it."

## 13. Case studies: coordination paid in production

**Kafka and ZooKeeper, then KRaft.** For most of its life, Kafka used ZooKeeper as its consensus-backed control plane: ZooKeeper held the cluster metadata — which broker is the controller, which broker leads each partition, the in-sync replica sets. The *data plane* (producing and consuming millions of messages per second) did not touch ZooKeeper on the hot path; brokers cached the metadata and served reads and writes locally, exactly the control-plane/data-plane split from section 10. This worked, but operating a second distributed system (ZooKeeper) alongside Kafka was a real cost, and ZooKeeper's scaling limits bounded the number of partitions Kafka could manage. Kafka's answer was **KRaft** — replacing ZooKeeper with Kafka's own internal Raft implementation, folding the consensus into Kafka itself. The architectural lesson is twofold: first, even Kafka, at its scale, kept consensus strictly in the control plane and never on the message hot path; second, the operational burden of running a separate coordination service was large enough that the Kafka team eventually built consensus in-house — which only makes sense at their scale and after years of investment, not for your service. The [Kafka-as-a-log deep-dive](/blog/software-development/database/kafka-as-a-distributed-log) covers the storage side; the coordination side is this story.

**The Redlock debate.** As covered in section 8, this is the canonical case study in *what a distributed lock does and does not guarantee*. The lasting lesson is not "Redis bad" or "Redlock wrong" — it is the precise distinction the debate forced into the open: a lock used for *efficiency* (don't do redundant work) has very different requirements from a lock used for *correctness* (don't corrupt shared state), and an algorithm without fencing tokens cannot provide the latter regardless of how many nodes it uses, because no amount of cross-node agreement stops a paused holder's in-flight write. Every architect who reads the Kleppmann critique and the Sanfilippo response comes away with the same upgraded mental checklist: *what does this lock protect, and is it fenced?* That single question prevents a class of silent corruption.

**A fencing-token save.** A pattern I have seen repeatedly rescue a system: a team has a worker pool processing jobs from a queue, with a per-job lock to prevent two workers running the same job. Under normal load it is fine; under memory pressure, workers hit long GC pauses, lock leases expire, and *two workers process the same job* — usually harmless (idempotent jobs) until the day a job has a side effect that is *not* idempotent (charge a card, send a payout, mutate an external ledger). The save is to thread a fencing token from the lock grant through to the side-effecting call, and have the *downstream system* (the payment processor's idempotency key, the ledger's version check) reject the stale token. The lock could not prevent the double-execution; the *downstream conditional check* did. This is the [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) pattern meeting the fencing token: when you cannot guarantee single-execution, you make double-execution safe by fencing the *effect*, not just the *lock*. The fencing token is how the lock layer and the idempotency layer talk to each other.

**A split-brain outage.** The classic post-mortem shape — and one that has played out at many companies — is a two-node-per-data-center deployment that loses one DC, the survivors cannot form a majority (section 6's two-DC problem), and the system goes fully write-unavailable even though half the hardware is healthy and reachable. Worse is the variant where someone, mid-incident, "fixes" the unavailability by *manually forcing* the minority side to accept writes ("just make it the primary"), the partition later heals, and now both sides have accepted divergent writes — instant split-brain corruption that takes days to reconcile by hand. The lesson is brutal and simple: **the minority refusing to serve is the system working correctly, not failing**, and the cure for the two-DC problem is a *third* failure domain holding a tie-breaker node — decided in the architecture phase, not improvised during an incident. If your consensus deployment cannot survive losing its largest failure domain while keeping a majority, you do not have a fault-tolerant control plane; you have a single point of failure with extra steps.

## 14. Optimization, gathered in one place

The optimization theme has run through the whole post; here it is consolidated, because in a review you want these levers at your fingertips with numbers attached.

**Keep consensus off the request path.** The single highest-leverage move. A consensus write is a majority round trip (~1–2 ms intra-region across AZs, plus fsync), charged per write; moving it off the hot path to a cached control-plane read takes the request-path coordination cost to *zero*. Measure the win as the p99 difference between "request path hits a quorum" and "request path reads local cache": often the difference between a ~5 ms p99 and a ~30+ ms p99 under load, and the difference between staying up and going down during a leader failover.

**Use leases to amortize.** Convert per-operation consensus into per-lease consensus. A 10-second lease over a workload of 50,000 ops/sec amortizes one consensus operation across 500,000 operations — a five-order-of-magnitude reduction in consensus frequency. Measure the win as consensus-ops-per-second before and after; you want it as close to "per config change" as possible, not "per request."

**Batch and pipeline the log.** Within the consensus group, batch many client writes into one log append and pipeline replication. This raises a consensus group's throughput from "one entry per round trip" (maybe a few thousand entries/sec) to "thousands of entries per round trip" (hundreds of thousands/sec) while keeping per-entry latency roughly flat. This is configured in every production consensus system; verify it is on. Measure the win as committed-entries-per-second at a fixed p99.

**Co-locate the leader with the write-heavy workload.** A leader in the wrong AZ adds a client-to-leader cross-AZ hop on top of the replication hop, roughly *doubling* the write latency. Pin the leader near where most writes originate; many systems let you weight elections or use leader-placement policies for this. Measure the win as the p50/p99 of a consensus write before and after co-location — often a ~2x reduction.

**Right-size the cluster: three or five, never even.** Five tolerates two failures at higher write latency than three (waits for more acks); three tolerates one at lower latency. Pick three unless you genuinely need to survive two simultaneous failures, and never pick an even number — four costs more than three and tolerates the same single failure. Measure the cost as the commit latency difference (waiting for two acks vs one) against the fault-tolerance gained.

**Shard the consensus when one group saturates.** A single group's commit rate is a hard ceiling that adding nodes makes *worse*. When the change rate exceeds what one group commits, partition the keyspace into independent consensus groups committing in parallel — the approach CockroachDB and Spanner take with per-range groups. Measure the ceiling as the max committed-writes/sec of one group, and scale by group count past it.

## 15. When to reach for consensus (and when not to)

**Reach for consensus when:** you have a piece of truth that *must* be the same on every node and *must* survive the failure of any single node — leadership, membership, the shard map, fleet-wide config, the fault-tolerant transaction coordinator. Reach for it when correctness genuinely requires that all participants agree, and when the thing being agreed upon changes *rarely* relative to how often it is read. And when you reach for it, reach for **etcd, ZooKeeper, or Consul** — not your own implementation — and put it in the **control plane**, with the data plane reading a cached view.

**Reach for a lease instead when:** you need consensus-grade safety but on the read path, or for a lock, and you can tolerate bounded staleness made safe by a fencing token. A lease is the right answer far more often than full per-operation consensus, because it amortizes the cost across many local actions. Almost every "we need a distributed lock" is really "we need a lease plus a fence enforced at the storage."

**Reach for a single-writer instead when:** you can partition ownership so one node authoritatively owns each key. This is the cheapest correct answer and you should try it *first*, before any lease or consensus, because it removes hot-path coordination entirely. The consensus you cannot avoid then lives only in *assigning* ownership, which is rare.

**Do not reach for consensus when:** the operation is on the request path of a latency-sensitive service (use a lease or a cached view), when the data can be partitioned to a single writer (partition it), when operations commute or can be made eventually consistent (use [leaderless replication and quorums](/blog/software-development/database/quorums-anti-entropy-and-read-repair) or CRDTs and reconcile), or when you are tempted to build your own consensus algorithm (you are almost certainly not in the small set of organizations for whom that is the right call). And never use an *unfenced* lock to protect *correctness* — that is the single most expensive mistake in this space, because it fails silently.

The meta-rule, the one I would attach to every design review: **consensus is a control-plane tool, and its cost is acceptable in inverse proportion to how often you pay it.** The whole art is paying it rarely. If your design pays for agreement on every request, you have not designed a distributed system; you have designed a single point of latency wearing a cluster costume.

## Key takeaways

- **A consensus write is a majority round trip, charged per write, forever.** Across AZs that is ~1–2 ms plus an fsync; on the request path at scale it is your throughput ceiling and your failover outage. Price it before you commit to it.
- **Most coordination needs are not consensus needs.** Walk the funnel: avoid entirely → single-writer → lease → full consensus, and stop at the first that meets the need. Most stop well before consensus.
- **Push consensus to the control plane, never the data plane.** The control plane decides rare, consistent truths; the data plane reads a cached view and serves requests with no quorum round trip. This is why Kubernetes and Kafka stay fast and survive control-plane outages as "frozen but serving."
- **Leases amortize consensus.** One grant covers millions of local actions, turning per-request coordination into per-lease coordination — often a five-order-of-magnitude reduction. They are the right answer far more often than per-operation consensus.
- **A distributed lock is a lease plus a fencing token, and the fence is enforced at the storage or it is not enforced at all.** Without a monotonic token checked by the storage, a GC-paused holder silently corrupts data. This is the Redlock lesson: unfenced locks are for efficiency, never correctness.
- **Three or five nodes, never even.** Majority needs odd counts; four tolerates the same one failure as three at higher cost. Never let a single failure domain hold a majority, and remember the two-DC problem needs a third location for the tie-breaker.
- **Leader failover has a built-in write-unavailability window of seconds.** Account for it in your SLO, keep it off the request path, and make sure the old leader *fences itself* the instant its lease lapses — the most common cause of split-brain is an application that keeps doing leader work after losing leadership.
- **A single consensus group does not scale with node count; it scales by sharding into many groups.** When one group saturates, partition the keyspace — the same single-writer-per-partition instinct, applied to the coordination layer.
- **The minority refusing to serve during a partition is the system working correctly.** Never manually force a minority to accept writes during an incident; that is how a recoverable outage becomes permanent split-brain corruption.

## Further reading

- [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch) — the mechanism this post sits on top of: how leaders are elected and logs are committed.
- [Paxos and Multi-Paxos explained](/blog/software-development/database/paxos-and-multi-paxos-explained) — the other consensus family, and why Multi-Paxos and Raft converge on the same leader-based shape.
- [Quorums, anti-entropy, and read-repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) — the leaderless alternative when you can trade strict agreement for availability.
- [Spanner, TrueTime, and external consistency](/blog/software-development/database/spanner-truetime-and-external-consistency) — consensus plus synchronized clocks for fault-tolerant atomic commit at global scale.
- [Consistency models: a practical guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) — the consistency you can promise is downstream of where you put coordination.
- [Replication strategies and their failure modes](/blog/software-development/system-design/replication-strategies-and-their-failure-modes) — how the coordination choices here shape what your replication can guarantee.
- *Designing Data-Intensive Applications* (Kleppmann), chapters 8–9 — the definitive treatment of consensus, linearizability, and the distributed-lock fencing argument.
- Martin Kleppmann, "How to do distributed locking," and Salvatore Sanfilippo's response — the Redlock debate in full; read both and form your own view on the fencing-token requirement.
- The Raft paper ("In Search of an Understandable Consensus Algorithm," Ongaro and Ousterhout) and the official etcd, ZooKeeper, and Consul documentation for the coordination primitives you should use instead of rolling your own.

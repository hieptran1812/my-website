---
title: "Paxos and Multi-Paxos: The Algorithm Everyone Cites and Few Understand"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-intuition-to-internals tour of single-decree Paxos and Multi-Paxos — the two phases, the quorum-intersection safety argument, dueling-proposer livelock, the stable-leader optimization, the Fast/EPaxos/Flexible variants, and a Paxos-vs-Raft playbook."
tags:
  [
    "paxos",
    "multi-paxos",
    "consensus",
    "distributed-systems",
    "raft",
    "chubby",
    "fault-tolerance",
    "total-order-broadcast",
    "quorum",
    "databases",
    "replication",
    "system-design",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/paxos-and-multi-paxos-explained-1.webp"
---

There is a particular kind of résumé line that distributed-systems interviewers have learned to distrust: "implemented Paxos." Not because the candidate is lying, but because Paxos occupies a strange place in our field. It is the most cited consensus algorithm in computing, the bedrock under Google's Chubby, Spanner, and Megastore, the engine inside Cassandra's lightweight transactions, the spiritual ancestor of every leader-based replication scheme shipping today — and it is also the algorithm that more engineers can *name* than can *explain*. Leslie Lamport wrote a nine-page paper in 2001 titled "Paxos Made Simple" whose entire abstract reads, with audible exasperation, "The Paxos algorithm, when presented in plain English, is very simple" ([Lamport, 2001](https://lamport.azurewebsites.net/pubs/paxos-simple.pdf)). The fact that he had to write that sentence — and that the field still needed it — tells you everything about the gap between citing Paxos and understanding it.

I have shipped two production systems built on a Paxos-derived log, debugged a livelock in a third at three in the morning, and reviewed more "we'll just use Paxos for that" design docs than I can count. The single most common failure mode is not a subtle bug in the safety proof. It is that the team never internalized *what Paxos actually guarantees and at what price*, so they reached for it where it was overkill, or — worse — built something Paxos-shaped that quietly violated the one invariant that makes it correct. This article is the explanation I wish those teams had read first. We will start from the consensus problem itself, build single-decree Paxos one forced design decision at a time the way Lamport does, stare hard at the quorum-intersection argument that is the entire safety story, watch basic Paxos livelock, fix it with Multi-Paxos and a stable leader, and then tour the variants — Fast Paxos, EPaxos, Flexible Paxos — and finally lay Paxos and Raft side by side so you know which one to actually reach for.

![The three roles of single-decree Paxos: a client asks proposers to get a value chosen, a majority of acceptors decides, and learners observe the chosen value](/imgs/blogs/paxos-and-multi-paxos-explained-1.webp)

The diagram above is the mental model for the entire piece. Three roles do all the work: **proposers** drive rounds and push values, **acceptors** are the distributed memory that votes a value into existence, and **learners** observe what was chosen. A value becomes *chosen* the instant a majority of acceptors have accepted one proposal — not when a proposer decides, not when a client is told, but at that majority. Everything else in Paxos exists to make that moment irreversible: once a value is chosen, no later round, no recovering crashed node, no competing proposer can ever cause a *different* value to be chosen. Hold onto that one sentence; it is the whole algorithm wearing a disguise.

## Why "we'll just use Paxos" is rarely the right sentence

Before we build anything, it is worth being precise about the mismatch most teams carry around, because it is the source of nearly every Paxos misadventure I have witnessed. People treat consensus as a library call — a black box you drop in to "make the system consistent." It is not. It is a specific, expensive contract with sharp edges, and the edges are exactly where the assumptions diverge from reality.

| Assumption | The naive mental model | The distributed reality |
| --- | --- | --- |
| "Paxos picks the *best* value." | The algorithm evaluates the proposals and chooses the right one. | Paxos picks *a* value — any proposed value. It guarantees agreement, not quality. Which value wins is a race, and after Phase 1 a proposer may be *forced* to abandon its own value entirely. |
| "Once I get an ack, it's chosen." | The proposer knows the outcome. | A proposer can succeed and never find out (its acks get lost). A value can be chosen without *any single node* knowing it yet. "Chosen" is a global predicate, not a local flag. |
| "Paxos guarantees progress." | If I run it, it terminates. | Basic Paxos guarantees *safety* always, but *liveness* never — two proposers can duel forever. Termination requires a leader, which the core algorithm does not provide. This is FLP showing up in practice. |
| "More acceptors = safer." | Adding nodes strengthens consensus. | Adding nodes raises the majority threshold and the latency of every decision. Five acceptors tolerate two failures; seven tolerate three but make every round wait on a larger quorum. |
| "Paxos and a replicated log are the same thing." | Run Paxos, get a log. | Single-decree Paxos agrees on *one* value. Turning that into a replicated *log* (Multi-Paxos) is a substantial additional construction with its own leader, slots, and gaps. Most "Paxos" in the wild means Multi-Paxos. |

> Consensus is not a feature you turn on. It is a tax you pay, per decision, forever — one or two wide-area round trips and the loss of availability the moment the network partitions. Use exactly as much of it as the correctness of your feature genuinely demands.

None of those right-hand columns are bugs. They are the *defined behavior* of a correct Paxos. The discipline of using Paxos well is knowing which of them your system can live with. The discipline of *understanding* Paxos — our job here — is seeing why each one is unavoidable given the problem it solves. So let us state the problem.

## 1. The consensus problem, stated precisely

> **Senior rule of thumb:** every consensus algorithm — Paxos, Raft, Zab, Viewstamped Replication — is solving the same problem and ultimately providing the same service: total order broadcast. If you understand the problem crisply, the algorithms stop looking like incantations and start looking like the small number of ways the problem *can* be solved.

Lamport's framing in "Paxos Made Simple" is deliberately minimal. Assume a collection of processes that can propose values. A consensus algorithm ensures that a single one among the proposed values is chosen. The safety requirements, quoted nearly verbatim from the paper, are three:

- **Only a value that has been proposed may be chosen.** (Validity — no consensus algorithm gets to invent values out of thin air.)
- **Only a single value is chosen.** (Agreement — the property everyone means when they say "consensus.")
- **A process never learns that a value has been chosen unless it actually has been.** (No spurious learning.)

Then there is liveness, which Lamport states loosely on purpose: "the goal is to ensure that some proposed value is eventually chosen and, if a value has been chosen, then a process can eventually learn the value." Notice he refuses to make this precise. That refusal is not laziness; it is the [FLP impossibility result](https://groups.csail.mit.edu/tds/papers/Lynch/jacm85.pdf) (Fischer, Lynch, Paterson, 1985) speaking. FLP proved that in a fully asynchronous system — no bound on message delay, even one faulty process — *no* deterministic algorithm can guarantee both safety and termination. So every real consensus algorithm makes a deliberate choice: keep safety unconditional, and provide liveness only under additional timing assumptions (a partially synchronous network, a stable leader). Paxos is the canonical instance of that choice. It is *always* safe and *eventually* live when the network behaves.

### The failure model matters as much as the goal

The properties above are meaningless without saying *what can go wrong*. Paxos assumes the standard **asynchronous, non-Byzantine, crash-recovery** model, and the assumptions are load-bearing:

- **Agents operate at arbitrary speed, may fail by stopping, and may restart.** Critically — and this is a sentence people skim past — "a solution is impossible unless some information can be remembered by an agent that has failed and restarted." Acceptors *must* persist their state to stable storage before responding. An acceptor that forgets its promises on reboot is not a slow acceptor; it is a *Byzantine* one, and it can break agreement. I have seen a real outage caused by exactly this: an acceptor that fsync'd lazily, crashed, came back having "un-promised," and let a second value get chosen. The cluster's invariant was destroyed by a missing `fdatasync`.
- **Messages can take arbitrarily long, can be duplicated, and can be lost, but they are not corrupted.** No malicious nodes, no forged messages. Paxos is *not* a Byzantine-fault-tolerant protocol; if you need to tolerate lying participants, you want PBFT or a blockchain-style protocol, and the cost is dramatically higher.

Within that model, consensus is exactly as hard as it needs to be and no harder. The three roles in the figure above — proposers, acceptors, learners — are Lamport's decomposition of *who does what*. A single physical process usually plays all three roles at once (every node in a five-node etcd cluster is proposer, acceptor, and learner), but separating them by role is what makes the algorithm tractable. The acceptors are the only role that holds durable state; they are the "memory" of the consensus. The proposers are stateless drivers. The learners are pure observers.

### Agreement, validity, termination — the three properties, named

It helps to give the three consensus properties their textbook names, because the literature uses them constantly and Paxos's design is best understood as "which of these is unconditional and which is best-effort." **Agreement**: no two correct processes decide different values. **Validity** (sometimes "integrity" or "non-triviality"): the decided value was actually proposed by some process — consensus may not fabricate. **Termination**: every correct process eventually decides. Paxos makes agreement and validity *unconditional* — they hold in every execution, with any pattern of crashes, message loss, reordering, and delay, full stop. Termination it makes *conditional* — it holds only once the system is "stable enough" (a single leader can talk to a majority for long enough to finish two phases). That split is the entire shape of the FLP bargain, and it explains a counterintuitive operational fact: a Paxos cluster under a bad partition will *block* (sacrifice termination) rather than *diverge* (sacrifice agreement). It would rather make no progress than make wrong progress. If you ever see a consensus system that keeps serving writes happily on *both* sides of a partition, it is not doing consensus — it has quietly dropped agreement, and you have an eventual-consistency system wearing a consensus costume.

There is a fourth, sneakier property worth naming because beginners conflate it with the others: **uniform agreement** versus plain agreement. Uniform agreement says that even a process that decides and *then crashes* cannot have decided differently from the survivors. In a crash-recovery model like Paxos's, where a crashed acceptor's durable state outlives it, you need uniform agreement — a value chosen by a majority that then partly crashes is still chosen, and a recovering node must not be allowed to contradict it. This is precisely why acceptor state must be durable *before* replying: the reply is a promise that survives the replier's death. Skip the durability and you get plain agreement among the living while violating uniform agreement across crashes — which is exactly the homegrown-Paxos bug in the case studies below.

This is the lens DDIA's Chapter 9 (Kleppmann) uses too, and it is worth stating in our own words because it reframes everything that follows. Kleppmann's central observation is that single-value consensus, total order broadcast, linearizable compare-and-set, and leader election with a lock are all *equivalent* problems — solve one and you can build the others. Paxos solves single-value consensus; Multi-Paxos chains that into total order broadcast (a replicated log); and a replicated log of commands is exactly what you need to build a linearizable, fault-tolerant state machine. That chain — single value → log → state machine — is the architecture of essentially every strongly-consistent distributed database. It is why the same algorithm shows up under Chubby, Spanner, etcd, ZooKeeper, and Cassandra's transactions. They are all, underneath, agreeing on the order of entries in a log. (This connects directly to [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual): a linearizable register is precisely what a consensus-backed log gives you, and the [CAP and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) tradeoff is the price you pay for it during partitions.)

## 2. Building single-decree Paxos from forced moves

The reason "Paxos Made Simple" is worth reading in the original is that Lamport does not present the algorithm and then justify it. He *derives* it: he starts with the dumbest possible scheme, shows exactly how it breaks, patches the break, shows how the patch breaks, and patches that — until the only thing left standing is Paxos. Every line of the protocol is a forced move. Let us walk the same path, because once you see that there was *no other choice* at each step, Paxos stops being mysterious.

### Attempt zero: a single acceptor

The easiest way to choose a value is to have one acceptor. Proposers send it values; it picks the first one it receives; done. This is trivially correct on agreement — there is one decider, so there is one decision. It is also useless: the moment that single acceptor crashes, no further progress is possible, and we have built a system *less* available than a single non-replicated server. The entire point of consensus is to survive failures, so a single point of failure is a non-starter.

### Attempt one: multiple acceptors and majorities

So use many acceptors. A proposer sends its value to a set of acceptors; an acceptor may *accept* the value; a value is chosen when "a large enough set" of acceptors have accepted it. How large is large enough? Here is the first crucial design decision, and it is forced by the agreement requirement. We need that if value $v$ is chosen and value $w$ is chosen, then $v = w$. The cleanest way to guarantee that two "chosen" sets cannot disagree is to require that any two of them *overlap* — share at least one acceptor. Because if every pair of qualifying sets shares an acceptor, and an acceptor accepts at most one value, then two different values cannot both reach a qualifying set.

The smallest, simplest family of sets with the property that any two of them intersect is **majorities**. Any two majorities of a set of $N$ acceptors share at least one member (two subsets each of size $> N/2$ cannot be disjoint, since their sizes would sum to more than $N$). This is the **quorum-intersection property**, and it is the load-bearing pillar of the entire algorithm. Lamport notes there is "an obvious generalization of a majority" — any **quorum system** where every two quorums intersect works — and we will return to that generalization when we get to Flexible Paxos. But for now: *majorities*, because any two of them overlap.

With $N = 2f + 1$ acceptors, a majority is $f + 1$, and the system tolerates up to $f$ crash failures while still being able to assemble a quorum from the survivors. Five acceptors tolerate two failures. Three tolerate one. This is why production consensus clusters are almost always sized to an odd number — going from five to six acceptors raises the majority from three to four (so you must now wait on *more* nodes) while still only tolerating two failures. Six acceptors is strictly worse than five: more latency, same fault tolerance.

### The break: P1, and why one acceptance per acceptor is not enough

Now we hit the first real problem. In the absence of failures we want a value chosen even when a single proposer proposes a single value. That suggests:

> **P1. An acceptor must accept the first proposal that it receives.**

But P1 immediately breaks. Suppose three proposers each propose a different value to a five-acceptor cluster at about the same time, and the values land such that no value reaches a majority — two acceptors took value A, two took value B, one took value C. Every acceptor has accepted *something*, P1 is satisfied, and yet *no value is chosen* and worse, the cluster is now stuck: each acceptor, having accepted its first proposal, can never accept anything else, so no majority can ever form. We have a system that satisfies P1 and can never make a decision.

The fix is forced: P1 plus "a value is chosen only when accepted by a majority" *together imply* that an acceptor must be allowed to accept **more than one** proposal. We need acceptors that can change their minds. To track which proposal is which, we attach a **proposal number** (Lamport's term; "ballot number" in much of the literature) to each proposal, so a proposal is a pair $(n, v)$ — a number and a value. To prevent confusion, **different proposals must have different numbers**: proposal numbers are globally unique and totally ordered.

### Proposal numbers: why monotonic uniqueness is non-negotiable

This is the detail people get wrong when they roll their own. The proposal numbers are not decoration; they are the mechanism that lets acceptors safely change their minds without ever un-choosing a chosen value. Two requirements:

1. **Uniqueness.** No two proposals — across all proposers, across all time — may share a number. The standard trick is to construct the number as a pair `(round, proposer_id)` compared lexicographically, or `round * N + proposer_id`. Each proposer draws round numbers from a disjoint set (proposer 3 in a five-node cluster uses `5k + 3`), so two proposers physically *cannot* mint the same number. If they could, two distinct proposals would be indistinguishable and the safety argument collapses.
2. **Monotonicity per proposer.** Each proposer must only ever issue *increasing* numbers. After a crash and recovery, a proposer must not reuse a number it (or anyone) might have used before. In practice this means persisting the highest number used, or deriving it from a monotonic clock plus the node id. (The [Cassandra LWT](https://docs.datastax.com/en/cassandra-oss/2.1/cassandra/dml/dml_ltwt_transaction_c.html) implementation uses a `timeuuid` ballot — a time-ordered UUID — exactly to get monotonic-and-unique for free.)

Why does totally-ordered uniqueness matter so much? Because the safety argument is an *induction on the proposal number*. Lamport's key invariant is:

> **P2. If a proposal with value $v$ is chosen, then every higher-numbered proposal that is chosen has value $v$.**

Since numbers are totally ordered, P2 directly guarantees that only a single value is chosen: take the lowest-numbered chosen proposal, call its value $v$, and P2 says every chosen proposal above it is also $v$, and nothing below it is chosen by definition. Agreement reduces to maintaining P2. And to maintain an induction *on the numbers*, the numbers must be a well-defined total order with no ties — which is exactly the uniqueness-and-monotonicity requirement. Break the ordering and you break the induction and you break agreement.

## 3. The two phases, derived

Now we strengthen P2 step by step into something an algorithm can actually maintain, and the two-phase structure of Paxos falls out automatically.

![Basic Paxos runs two phases per decision: Prepare/Promise claims a ballot and learns prior acceptances, then Accept/Accepted commits the value once a majority votes](/imgs/blogs/paxos-and-multi-paxos-explained-2.webp)

To be chosen, a proposal must be accepted by at least one acceptor. So we can satisfy P2 by satisfying the stronger:

> **P2a. If a proposal with value $v$ is chosen, then every higher-numbered proposal *accepted by any acceptor* has value $v$.**

But P2a fights P1. Communication is asynchronous, so a value could be chosen while some acceptor $c$ has received nothing. Suppose a new proposer wakes up and issues a higher-numbered proposal with a *different* value to $c$. P1 says $c$ must accept it — violating P2a. The fix strengthens again, pushing the constraint back to the proposer:

> **P2b. If a proposal with value $v$ is chosen, then every higher-numbered proposal *issued by any proposer* has value $v$.**

Since a proposal must be issued before it can be accepted, P2b implies P2a implies P2. Now we have to make P2b an *algorithm*. How can a proposer, about to issue proposal number $n$, know whether some value was already chosen by a lower-numbered proposal — and if so, which? It cannot see the future, and it cannot poll all of history. Lamport's move here is the cleverest in the paper. He converts P2b into a checkable invariant about a *single quorum*:

> **P2c.** For any $v$ and $n$, if a proposal with value $v$ and number $n$ is issued, then there is a set $S$ consisting of a majority of acceptors such that *either* (a) no acceptor in $S$ has accepted any proposal numbered less than $n$, *or* (b) $v$ is the value of the highest-numbered proposal among all proposals numbered less than $n$ accepted by the acceptors in $S$.

Read that twice — it is the whole algorithm. P2c says: before you propose value $v$ with number $n$, find a majority $S$, and make sure $v$ is "safe" with respect to $S$, meaning either nobody in $S$ has accepted anything lower than $n$ (so you are free to pick any value) *or* $v$ matches whatever the highest-numbered acceptance in $S$ already chose (so you are forced to carry that value forward). To maintain P2c, "a proposer that wants to issue a proposal numbered $n$ must learn the highest-numbered proposal with number less than $n$, if any, that has been or will be accepted by each acceptor in some majority." Learning about *past* acceptances is easy — ask. Predicting *future* acceptances is impossible — so the proposer *controls* the future by extracting a **promise**: it asks the acceptors not to accept any more proposals numbered less than $n$. That request is **Phase 1**, the prepare request.

### Phase 1 — Prepare and Promise

Quoting Lamport's final two-phase statement, with our annotations:

> **Phase 1. (a)** A proposer selects a proposal number $n$ and sends a *prepare* request with number $n$ to a majority of acceptors. **(b)** If an acceptor receives a *prepare* request with number $n$ greater than that of any *prepare* request to which it has already responded, then it responds with a promise not to accept any more proposals numbered less than $n$ and with the highest-numbered proposal (if any) that it has accepted.

Two things happen in one round trip. The proposer **claims the ballot** — by promising, acceptors agree to ignore everything below $n$, which both lets this proposer make progress *and* preempts any straggler with a lower number. And the proposer **learns the past** — each promise carries back the acceptor's most recent acceptance, if any. When the proposer collects promises from a majority, it has, by quorum intersection, learned about every value that *could already be chosen*, because any chosen value was accepted by a majority, and any two majorities overlap, so at least one acceptor in this proposer's quorum saw it.

### Phase 2 — Accept and Accepted

> **Phase 2. (a)** If the proposer receives a response to its *prepare* requests (numbered $n$) from a majority of acceptors, then it sends an *accept* request to each of those acceptors for a proposal numbered $n$ with a value $v$, where $v$ is the value of the highest-numbered proposal among the responses, or is any value if the responses reported no proposals. **(b)** If an acceptor receives an *accept* request for a proposal numbered $n$, it accepts the proposal unless it has already responded to a *prepare* request having a number greater than $n$.

This is where the value gets pinned. The proposer does *not* get to use its own value if the promises revealed a prior acceptance — it is *forced* to re-propose the highest-numbered value it saw. That forcing is exactly P2c clause (b), and it is what makes a chosen value sticky. If the promises were all empty (nobody had accepted anything below $n$), the proposer is free to use its own value — P2c clause (a). When a majority of acceptors accept $(n, v)$ in Phase 2b, $v$ is **chosen**. Note the asymmetry one more time: the proposer may *not know* the value is chosen if some Phase 2b acks are lost — but it is chosen regardless, because "chosen" is defined by the acceptors' state, not the proposer's knowledge.

The figure below traces the forcing concretely. Some earlier round (ballot 2) got value `X` accepted on acceptor 1. A fresh proposer then comes along with `prepare(5)`; acceptor 1's Phase 1b promise carries back "accepted (2, X)." The new proposer sees that the highest-numbered acceptance is `X`, so P2c forces it to send `accept(5, X)` — *its own intended value is discarded entirely*. The majority then accepts `X`, and the value is preserved across the ballot change. This is not a special failure-handling case; it is the *normal* Phase 1 behavior whenever any prior acceptance exists, and it is the single mechanism by which "once chosen, always chosen" is enforced.

![Phase 1 recovers an already-accepted value: a new proposer's promises reveal a prior acceptance, forcing it to re-propose the same value rather than overwrite it](/imgs/blogs/paxos-and-multi-paxos-explained-4.webp)

The acceptor's logic is delightfully small. It receives two kinds of request and tracks two pieces of durable state — the highest *prepare* number it has promised (`minProposal`) and the highest proposal it has *accepted* (`acceptedNumber`, `acceptedValue`). Here it is as runnable pseudocode:

```python
# Acceptor state — ALL of it must be on stable storage before any reply.
# Losing any of this across a crash can break agreement.
class Acceptor:
    def __init__(self):
        self.min_proposal = 0        # highest prepare number promised
        self.accepted_number = 0     # number of the last accepted proposal
        self.accepted_value = None   # value of the last accepted proposal

    def on_prepare(self, n):
        # Phase 1b. Promise iff n beats every prepare we've answered.
        if n > self.min_proposal:
            self.min_proposal = n
            self.persist()           # fsync BEFORE replying — non-negotiable
            return Promise(ok=True,
                           accepted_number=self.accepted_number,
                           accepted_value=self.accepted_value)
        return Promise(ok=False, min_proposal=self.min_proposal)  # preempted

    def on_accept(self, n, v):
        # Phase 2b. Accept iff we never promised a higher prepare number.
        if n >= self.min_proposal:
            self.min_proposal = n
            self.accepted_number = n
            self.accepted_value = v
            self.persist()           # fsync BEFORE replying
            return Accepted(ok=True, n=n)
        return Accepted(ok=False, min_proposal=self.min_proposal)  # preempted
```

And the proposer drives the two phases, with the all-important "carry forward the highest accepted value" step:

```python
def propose(acceptors, my_value):
    n = next_unique_increasing_number()      # globally unique, monotonic
    quorum = majority_size(len(acceptors))

    # ---- Phase 1: Prepare / Promise ----
    promises = []
    for a in acceptors:
        r = a.on_prepare(n)
        if r.ok:
            promises.append(r)
        if len(promises) >= quorum:
            break
    if len(promises) < quorum:
        return RETRY                          # could not claim the ballot

    # P2c: if ANY acceptor reported a prior acceptance, we are FORCED to
    # re-propose the highest-numbered one. We do not get to use my_value.
    prior = max((p for p in promises if p.accepted_value is not None),
                key=lambda p: p.accepted_number, default=None)
    value = prior.accepted_value if prior else my_value

    # ---- Phase 2: Accept / Accepted ----
    accepts = 0
    for a in acceptors:
        r = a.on_accept(n, value)
        if r.ok:
            accepts += 1
        if accepts >= quorum:
            return CHOSEN(value)              # a majority accepted (n, value)
    return RETRY                              # got preempted mid-Phase-2
```

Read those two functions next to the two-phase definition and you have, in maybe forty lines, *all of single-decree Paxos*. The subtlety is not in the volume of code; it is in two lines: the `prior` computation that forces value carry-forward, and the two `persist()` calls that make state survive crashes. Get either wrong and you have built a fast, confident, *incorrect* consensus.

### The one optimization Lamport adds

There is a small but standard optimization in the original paper. If an acceptor receives a *prepare* numbered $n$ but has already promised some higher number, "there is then no reason for the acceptor to respond" — it will reject the eventual accept anyway, so it can simply ignore the prepare (or, better, send a *negative acknowledgement* with its current `min_proposal` so the proposer learns it has been preempted and should back off rather than time out). That NACK-and-back-off behavior is "a performance optimization that does not affect correctness," but it is the hook on which liveness hangs, as we will see in the dueling-proposers section.

## 4. Quorum intersection: the safety heart, in one picture

Everything above rests on a single geometric fact, and it deserves its own figure because it is the thing to memorize if you memorize nothing else.

![Quorum intersection: any two majorities of five acceptors overlap in at least one acceptor, and that overlap node carries the chosen value into the next round](/imgs/blogs/paxos-and-multi-paxos-explained-3.webp)

The claim the figure proves: *two majorities of the same acceptor set always share at least one acceptor.* In the picture, round $n$ used quorum {1, 2, 3} and chose value $v$ (acceptors 1, 2, 3 all hold $v$). Some time later, a fresh proposer runs round $n+1$ and happens to contact quorum {3, 4, 5}. Acceptors 4 and 5 are idle — they never saw round $n$ — but **acceptor 3 is in both quorums**. When the round-$n+1$ proposer runs Phase 1, acceptor 3's promise carries back "(round $n$, value $v$)." Now P2c clause (b) kicks in: the highest-numbered acceptance the proposer saw is $v$, so the proposer is *forced* to propose $v$ in Phase 2. The chosen value propagates forward. It is impossible for round $n+1$ to choose anything other than $v$.

This is why "once chosen, always chosen" holds. A value is chosen when a *majority* accepts it. Any *future* proposer must, in Phase 1, gather promises from a *majority*. Those two majorities intersect, so the future proposer is guaranteed to see the chosen value and is forced to re-propose it. There is no execution — no message reordering, no crash, no recovery, no competing proposer — that can route around this, because it is not a property of the *timing*; it is a property of the *sets*. Set theory does the work that no amount of careful message handling could.

It is worth dwelling on what this does *not* require, because it is the part that feels like magic. It does **not** require that the round-$n+1$ proposer know that round $n$ happened. It does **not** require that the chosen value be "committed" or "announced" anywhere. It does **not** require any global coordinator. The safety emerges purely from "majority to choose" + "majority to prepare" + "majorities intersect" + "carry forward the highest." Four facts, and consensus is safe forever. When people say Paxos is "subtle," this is the subtlety: the safety is *non-local* and *emergent*, not enforced by any single component. That is also why it is so easy to break when you reimplement it — you remove what looks like an unnecessary step (say, the value carry-forward, or the durable promise) and the emergent safety silently evaporates with no error message.

### A worked counterexample: what breaks without carry-forward

Make it concrete. Five acceptors. Proposer A runs round 1, gets {1, 2, 3} to accept value `RED`. `RED` is now chosen — a majority holds it. Proposer A's acks are lost; A never learns. Now proposer B runs round 2 and contacts {3, 4, 5}. Suppose B *ignored* the carry-forward rule and just proposed its own value `BLUE`. Acceptor 3 promised round 2 (2 > 1), but B threw away the "(round 1, RED)" that acceptor 3 reported. B sends accept(2, BLUE) to {3, 4, 5}; all three accept; `BLUE` is now chosen by majority {3, 4, 5}. We now have two chosen values, `RED` (majority {1,2,3}) and `BLUE` (majority {3,4,5}). Agreement is destroyed. The *only* thing that prevented this in real Paxos was the single line `value = prior.accepted_value if prior else my_value`. That line is not an optimization or a nicety; it is the algorithm. I have personally caught this exact omission in a code review of a homegrown "Paxos-lite" — the author had decided the carry-forward was "for the failure case" and skipped it on the happy path. It is *never* skippable.

## 5. Learning the chosen value

We have agreed on a value; now someone has to find out. Learners are the third role, and the obvious scheme is for every acceptor, whenever it accepts, to notify every learner — then any learner that hears the same acceptance from a majority knows the value is chosen. That works but costs `acceptors × learners` messages per decision. The cheaper, standard approach is a **distinguished learner** (or a small set of them): acceptors report their acceptances to one designated learner, which detects the majority and then broadcasts "chosen" to the others. This costs `acceptors + learners` messages at the price of one extra hop of latency and a single point of failure for *learning* (not for safety — if the distinguished learner dies, a new one can re-derive the chosen value by re-querying a majority of acceptors).

In practice, in Multi-Paxos systems, the leader is also the distinguished learner: it ran Phase 2, so it knows the moment a majority acked, and it piggybacks "this slot is committed" onto the next message. There is a subtle recurring failure here, though: a learner (or a recovering node) that wants to know "what was chosen" cannot just ask one acceptor — one acceptor's accepted value might be from a *losing* proposal. To learn safely, you must read from a majority and take the highest-numbered acceptance, the same way a proposer does in Phase 1. This is why "read your own writes" in a Paxos system is not free; a fresh, safe read is itself a quorum operation. (Multi-Paxos systems optimize this with leader leases, covered below, so that the leader can serve reads locally without a quorum round trip — but that is an optimization layered on top, with its own clock assumptions.)

## 6. Why basic Paxos is slow, and why it can livelock

Basic single-decree Paxos is *correct*, and for a one-shot decision — "who is the leader?", "is this username taken?" — it is exactly right. But two properties make it unsuitable as-is for a high-throughput replicated log.

### Two round trips per decision

Every decision pays for *both* phases: one round trip to a majority for Prepare/Promise, a second for Accept/Accepted. In a single data center that is maybe two times a few hundred microseconds; across regions it is two times tens of milliseconds. For a system committing thousands of log entries per second, paying two wide-area round trips *per entry* is a throughput catastrophe. The whole point of Multi-Paxos, next section, is to amortize Phase 1 away so that the steady state costs one round trip per decision instead of two.

It pays to count in **message delays** rather than round trips, because that is the currency the variant papers use and it makes the comparisons exact. A *round trip* is two message delays (request out, reply back). Basic single-decree Paxos with a client is: client → proposer (1), proposer → acceptors Prepare (2), acceptors → proposer Promise (3), proposer → acceptors Accept (4), acceptors → proposer Accepted (5), proposer → client (6) — roughly six message delays, or two-plus round trips of consensus traffic. Multi-Paxos in steady state drops the Prepare/Promise pair, landing at client → leader → acceptors → leader → client, about four message delays. Fast Paxos and EPaxos push the happy path down to *two* message delays by letting the client reach acceptors directly, which is why the latency column in the variant table is best read in message-delay units: each variant is buying back one or two delays at the cost of larger quorums or more bookkeeping. The throughput knob and the latency knob are separate — Multi-Paxos pipelines many in-flight Accept rounds from one leader to win throughput, while Fast/EPaxos shave the per-decision latency — and a careful system tunes both.

### Dueling proposers: livelock, which is FLP wearing work clothes

The nastier problem is liveness. Basic Paxos has no notion of "whose turn it is." Any proposer can start a round at any time. When two proposers both want to get a value chosen, they can leapfrog each other's ballot numbers forever.

![Dueling proposers livelock: two proposers leapfrog ballot numbers, each preempting the other's Accept before it can commit, so no value is ever chosen despite no crash](/imgs/blogs/paxos-and-multi-paxos-explained-5.webp)

Trace it. Proposer A runs Phase 1 with number 1 and a majority promises. Before A can finish Phase 2, proposer B runs Phase 1 with number 2; the acceptors, having promised 2, will now *reject* A's accept(1, ·) because 1 < 2. A's Phase 2 fails. A, being a good proposer, picks a higher number — 3 — and runs Phase 1 again, which preempts B's pending accept(2, ·). Now B's Phase 2 fails, so B picks 4 and preempts A. And so on. **No value is ever chosen, even though no node has crashed and the network is delivering every message.** This is livelock, and it is precisely the FLP impossibility result manifesting in a concrete protocol: in an asynchronous system you cannot guarantee termination, and here the failure to terminate takes the shape of two healthy proposers starving each other indefinitely.

The fix is not algorithmic cleverness — it *can't* be, because FLP forbids it. The fix is to break the symmetry: elect a single **distinguished proposer** (a leader) and have everyone route proposals through it. With one proposer, there is no one to duel with, so it can complete both phases. Lamport states this directly: "If the leader can communicate successfully with a majority of acceptors, and if it uses a proposal number greater than any already used, then it will succeed... By keeping a sufficiently large gap... the algorithm should make progress." The leader is a *liveness* device, not a *safety* device — and that distinction is the single most important thing to understand about leader-based consensus. Even with a buggy leader election that occasionally elects two leaders, Paxos remains *safe*: the two "leaders" simply become dueling proposers and might livelock, but they can never choose two different values. You can have zero leaders (no progress) or two leaders (possible livelock) and never violate agreement. Leadership is an optimization for liveness layered on top of an always-safe core.

In production, the standard liveness mechanism is **randomized exponential backoff** on preemption: when a proposer is preempted (gets a NACK), it waits a random interval before retrying, so the dueling proposers desynchronize and one of them gets a clean window to finish. Raft uses the same idea for leader election (randomized election timeouts). It does not *guarantee* termination — nothing can — but it makes livelock vanishingly unlikely in practice, which is exactly the partial-synchrony bargain every real system strikes.

## 7. Multi-Paxos: from one value to a replicated log

A single chosen value is not useful on its own. What we actually want is a **replicated log**: an agreed-upon, totally-ordered sequence of commands that every replica applies to its state machine, yielding identical state everywhere. That is total order broadcast, and Multi-Paxos is how you build it out of single-decree Paxos.

The construction is simple to state: run a *separate* instance of Paxos for each **slot** in the log. Slot 0 agrees on the first command, slot 1 on the second, and so on. Each slot is an independent single-decree consensus. The state machine applies committed slots in order. That is the whole idea — but stated that way it is still two round trips per slot, no better than before. The Multi-Paxos optimization is what makes it fast.

![Multi-Paxos: a stable leader runs Phase 1 once, then commits each command in a single Accept round trip, turning 2 RTT per command into 1 RTT](/imgs/blogs/paxos-and-multi-paxos-explained-6.webp)

### The key trick: a stable leader runs Phase 1 once

Here is the insight. Phase 1 — Prepare/Promise — does two things: it claims a ballot number, and it learns prior acceptances. But the *ballot number* part is not slot-specific. If a leader runs Phase 1 *once* with ballot $n$ and gets a majority to promise, that promise can cover *all future slots at ballot $n$*. The leader has, in one round trip, claimed the right to be the sole proposer at ballot $n$ for the entire log. From then on, for each new client command, the leader only needs to run **Phase 2** — a single accept(n, slot, value) round trip — to get the command chosen. The before/after figure makes the win concrete: basic Paxos pays prepare + accept (2 RTT) per command; steady-state Multi-Paxos pays one Phase 1 up front and then a single accept (1 RTT) per command thereafter.

This is the difference between a textbook curiosity and a system you can ship. One round trip per command, with the leader as both proposer and distinguished learner, is competitive with primary-backup replication while retaining the fault tolerance of full consensus. Phase 1 only runs again when leadership changes — a new leader runs Phase 1 (with a higher ballot) across the log to learn the state of all in-flight slots and claim the ballot, then resumes single-round-trip operation.

### The log of slots, with gaps and no-ops

![Multi-Paxos replicates a log of independent slots, where the leader fills gaps with no-ops so the state machine can apply a contiguous prefix in order](/imgs/blogs/paxos-and-multi-paxos-explained-9.webp)

A real Multi-Paxos log is messier than a clean sequence, and the figure shows why. Slots 0 and 1 are chosen. Slot 2 is a **gap** — maybe the previous leader started it and crashed, or its accept got lost. Slot 3 has an accept in flight. Slot 4 is empty, waiting for the next command. The state machine can only apply a **contiguous prefix** of chosen slots: it applies 0 and 1, then *stalls at the gap in slot 2* even though slot 3 might already be chosen. You cannot apply slot 3's `DEL z` before you know what slot 2 was — the order is the entire point.

So a new leader, on taking over, must **fill gaps**. For every slot below its committed watermark that it cannot confirm was chosen, it runs Paxos for that slot. If Phase 1 reveals a value was already accepted there, the leader re-proposes it (carry-forward again). If Phase 1 reveals *nothing* was accepted, the leader proposes a **no-op** — a do-nothing command — purely to close the gap so the prefix becomes contiguous and the state machine can advance. This gap-filling is a real source of bugs in homegrown implementations: people forget that a gap might *already* hold a chosen value at some replica, skip the Phase 1 read, and overwrite it with a no-op, silently losing a committed command. (This is the carry-forward invariant, per slot, all over again.)

### Leader leases: serving reads without a quorum

There is one more crucial optimization that real systems layer on top. A linearizable *read* in plain Multi-Paxos still costs a round trip — the leader must confirm it is *still* the leader (that no one has elected a newer leader behind its back) before answering, or it might serve stale data. To avoid paying that on every read, systems give the leader a **time-bounded lease**: a promise from a majority that no other leader will be elected for, say, the next 10 seconds. While the leader holds a valid lease, it can serve reads *locally* with no quorum round trip, because it knows no one else could have committed a write. [Google's Spanner](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/Spanner.pdf) uses exactly this — long-lived Paxos leader leases (10 seconds by default) so that the leader can pipeline writes and serve reads without re-running consensus. The catch is that leases depend on bounded clock drift, which is why Spanner went to the trouble of building TrueTime (GPS and atomic clocks in every data center) to bound the uncertainty. Leases are a *clock*-based optimization bolted onto a *message*-based safety core, and they are only as safe as your clock-drift bound.

### The stability problem Google hit

The [Paxos Made Live](https://www.semanticscholar.org/paper/Paxos-made-live:-an-engineering-perspective-Chandra-Griesemer/76aad5b272219c6745c76b7874129797e97e6041) paper from the Chubby team documents a Multi-Paxos pathology worth knowing. With intermittent network outages, the stable-leader optimization can *thrash*. A leader temporarily disconnects; the cluster elects a new leader; the old leader reconnects, sees it has been preempted, bumps its sequence number *higher* than the new leader's, and preempts it right back; then disconnects again; and the cycle repeats. It is dueling proposers, resurrected at the Multi-Paxos layer. The Chubby team's fix was twofold: leader **leases** (so a healthy leader is guaranteed a window during which no one else can take over, breaking the thrash) and **epoch numbers** that demarcate leadership terms (so a request can detect whether leadership changed underneath it). These are precisely the kind of engineering details that "Paxos Made Simple" leaves out and that "Paxos Made Live" exists to document — the paper's whole thesis is that there is a vast gap between the published algorithm and a correct, performant, operable implementation, and that gap is where the real work lives.

## 8. The variants: each one relaxes one classic constraint

Once you understand the core — two phases, quorum intersection, leader for liveness — the variant zoo becomes legible. Each variant identifies one constraint of classic Paxos and relaxes it to win something.

![Paxos variants compared on latency, quorum rule, and leadership: Fast Paxos, EPaxos, and Flexible Paxos each relax a different classic constraint](/imgs/blogs/paxos-and-multi-paxos-explained-8.webp)

### Fast Paxos: cut a message delay by letting clients propose directly

[Fast Paxos](https://link.springer.com/article/10.1007/s00446-006-0005-x) (Lamport, 2006) attacks the *latency* of the steady state. In Multi-Paxos, a command travels client → leader → acceptors → back, which is three message delays end to end. Fast Paxos lets clients send their value *directly to the acceptors*, skipping the leader hop and getting the value learned in *two* message delays on the happy path. The price is paid in the quorum math. Because there is no leader to serialize, two clients might send different values to overlapping acceptor sets simultaneously — a **collision** — and to recover safely the quorums must be larger: Fast Paxos needs $3f + 1$ acceptors to tolerate $f$ faults (versus $2f + 1$ for classic), and the rule becomes "any two fast quorums and one classic quorum must share a node." When a collision happens, the coordinator detects it and falls back to a classic round, which costs the extra message delay back. Fast Paxos is fastest when conflicts are rare and worth its larger clusters only when that's true.

### EPaxos: leaderless, and order only what actually conflicts

[Egalitarian Paxos](https://www.usenix.org/conference/nsdi13/technical-sessions/paper/moraru) (Moraru, Andersen, Kaminsky, 2013) makes a deeper observation: a *total* order over all commands is more than most applications need. If two commands **commute** — `SET x=1` and `SET y=2` touch different keys, so their relative order is irrelevant to the final state — there is no reason to force them into a global sequence. EPaxos is **leaderless**: any replica can commit any command, with no distinguished proposer. For each command, the replica computes its **dependencies** — the set of currently-uncommitted commands that *do not* commute with it — and runs a fast quorum to agree on those dependencies. Commands that commute commit in **two message delays** on the fast path; only *conflicting* commands need the slow, classic-Paxos-style path. The payoff is huge for geo-distributed workloads: there is no single leader to be a bottleneck or a far-away latency tax, load spreads evenly, and a client talks to its *nearest* replica. The cost is complexity — EPaxos replaces "agree on a log" with "agree on a dependency graph and execute it in a consistent order," and the literature is candid that the original protocol is "very complex, ambiguously specified and suffers from nontrivial bugs," prompting [corrected variants](https://software.imdea.org/~gotsman/papers/epaxos-opodis25.pdf). It is the most powerful variant and the hardest to get right.

### Flexible Paxos: the quorum-intersection rule is weaker than we thought

[Flexible Paxos](https://arxiv.org/abs/1608.06696) (Howard, Malkhi, Spiegelman, 2016) is my favorite, because it is a *theorem* that retroactively relaxes the core. Recall the safety argument needs majorities only so that a Phase 1 quorum intersects the Phase 2 quorum that chose a value. Flexible Paxos proves that *the only intersection requirement is between Phase 1 quorums and Phase 2 quorums* — Phase 1 quorums need **not** intersect each other, and Phase 2 quorums need **not** intersect each other. Formally, you need every Phase 1 quorum (call it $Q_1$) to intersect every Phase 2 quorum ($Q_2$): $|Q_1| + |Q_2| > N$. That is *strictly weaker* than requiring both to be majorities.

The practical consequences are immediate and delicious. You can make Phase 2 quorums *small* (fast, frequent commits) at the cost of making Phase 1 quorums *large* (slow, rare leader elections). For a stable-leader Multi-Paxos that elects a leader once an hour and commits thousands of times a second, that is exactly the trade you want: pay more on the rare operation to pay less on the common one. With $N = 4$, you could use $|Q_2| = 2$ and $|Q_1| = 3$, since $2 + 3 > 4$ — committing with only two acceptors instead of three. Or with an even cluster you can shave one off the Phase 2 quorum for free. The figure's bottom row captures it: "only $Q_1$ meets $Q_2$" is the whole relaxation, and it shows that decades of "Paxos needs majorities" was an over-constraint. The intersection is required *only across phases*.

There is a further sharpening: if a proposer learns in Phase 1 that a value was already proposed in some round, its Phase 1 no longer needs to intersect the Phase 2 quorums of *that* round, opening even more flexible quorum assignments. The lesson Flexible Paxos teaches is meta: the "magic" of Paxos was never the majorities; it was the *intersection across phases*, and majorities were just the easiest sufficient way to get it.

### A comparison table

| Variant | Latency (no conflict) | Acceptors for $f$ faults | What it relaxes | Best when |
| --- | --- | --- | --- | --- |
| Basic Paxos | 2 RTT (4 msg delays) | $2f+1$ | nothing (baseline) | one-shot decisions |
| Multi-Paxos | 1 RTT steady (2 msg delays) | $2f+1$ | amortizes Phase 1 | replicated log, stable leader |
| Fast Paxos | 2 msg delays | $3f+1$ | client proposes directly | low conflict, latency-critical |
| EPaxos | 2 msg delays if commuting | $2f+1$ | total order → dependency order | geo-distributed, commutative ops |
| Flexible Paxos | tunable per phase | $|Q_1|+|Q_2|>N$ | majority → cross-phase intersection | rare elections, frequent commits |

## 9. Paxos vs Raft: which should I reach for

This is the question I actually get asked, so let us answer it directly rather than diplomatically.

![Paxos vs Raft compared on leadership, log model, spec clarity, and liveness path, showing Raft as Multi-Paxos with a strong leader and a single canonical spec](/imgs/blogs/paxos-and-multi-paxos-explained-7.webp)

[Raft](https://raft.github.io/raft.pdf) (Ongaro and Ousterhout, 2014) was explicitly designed as an answer to the complaint this entire article is built around: Paxos is hard to understand and harder to implement correctly. The paper's abstract says it plainly — Raft "produces a result equivalent to (multi-)Paxos, and it is as efficient as Paxos, but its structure is different... making Raft more understandable." Their user study found 33 of 43 students answered questions about Raft better than about Paxos after learning both. That understandability is not academic vanity; it is *operational risk reduction*. The single most reliable predictor of a correct consensus deployment is whether the team can hold the algorithm in their heads, and Raft was engineered for exactly that.

The deep truth, which Kleppmann emphasizes in DDIA Chapter 9, is that **Raft, Multi-Paxos, Zab (ZooKeeper), and Viewstamped Replication are all implementing the same thing**: total order broadcast via a leader-based replicated log. They differ in *presentation and emphasis*, not in the problem solved. Raft is best understood as a particular, opinionated, strong-leader formulation of Multi-Paxos that nails down all the details Multi-Paxos leaves as folklore.

The concrete differences, mapping to the figure's four rows:

- **Leadership.** Basic Paxos has *no* leader (any proposer); Multi-Paxos has a *distinguished* leader you bolt on for liveness; Raft has a *strong* elected leader that is structurally central — all log entries flow leader-to-follower, never the reverse. Raft's strong leader simplifies reasoning at a small cost in flexibility (a slow leader can't be bypassed the way EPaxos allows).
- **Log model.** Multi-Paxos slots are *independent* — slot 5 can be chosen while slot 3 is still a gap, and gaps get filled later. Raft enforces a *contiguous* log: a follower's log is always a prefix-consistent extension of the leader's, and the **Log Matching Property** guarantees that if two logs agree on an entry, they agree on everything before it. This is more restrictive but dramatically easier to reason about — there is no gap-filling, no out-of-order slots.
- **Spec clarity.** Basic Paxos is terse and underspecified (gloriously so — that's the point of "Paxos Made Simple," but it leaves implementers stranded). Multi-Paxos has *no single canonical spec* — every implementation reinvents the leader, the slots, the gap-filling, and they all differ subtly, which is a quiet source of incompatibility and bugs. Raft has *one* canonical, complete specification including leader election, log replication, membership changes, and snapshotting. You can implement Raft from the paper; you cannot really implement Multi-Paxos from any single paper.
- **Liveness path.** All of them need a leader for liveness and face the same FLP wall. Raft uses **randomized election timeouts** (each follower waits a random interval before standing for election) to avoid split votes — the same desynchronization trick that fixes dueling proposers, made first-class in the spec.

### The actual recommendation

> **Reach for Raft when** you are building a new replicated log or strongly-consistent store from scratch and you want the highest odds of a *correct, maintainable* implementation. The canonical spec, the abundance of audited libraries (etcd's `raft`, `hashicorp/raft`, `tikv/raft-rs`), and the team-comprehensibility win make it the default for new systems. This is the right answer the overwhelming majority of the time.

> **Reach for Paxos / Multi-Paxos when** you are working inside a system that already uses it (you don't rewrite Spanner's core for fun), when you need the specific flexibility Raft's strong leader forecloses — leaderless operation (EPaxos), tunable per-phase quorums (Flexible Paxos), or out-of-order slot commits for pipelining — or when you genuinely need a *one-shot* decision (leader election, a distributed lock, a one-time configuration flip), where single-decree Paxos is the precise tool and a full Raft log is overkill.

> **Skip both when** a single-leader, asynchronously-replicated database with failover meets your durability and consistency needs — which is more often than consensus enthusiasts admit. Consensus costs a round trip per decision and unavailability under partition; if your feature tolerates eventual consistency or a few seconds of failover downtime, you are paying a steep tax for a guarantee you don't need.

## 10. Where Paxos actually runs

Theory is cheap; let us ground this in shipping systems, because the named deployments are how you build intuition for *when* the cost is worth it.

### 1. Google Chubby — the lock service that runs the company

Chubby is Google's distributed lock service, and it is the original "Paxos Made Live" subject. Internally it runs a Multi-Paxos replicated log across (typically) five replicas, electing a master via Paxos and using master leases so the master can serve reads locally. The genius of Chubby is *not* that it's a great lock service — it's a deliberately coarse-grained one — but that it concentrates all the hard consensus into *one* well-tested system that everything else (GFS, Bigtable, MapReduce) leans on for leader election and metadata. The lesson from the paper that has aged best: the published algorithm is maybe 5% of the work; disk corruption handling, log compaction via snapshots, epoch numbers for leadership changes, the `MultiOp` primitive for atomic test-and-set-and-write, and fault-injection testing are the other 95%. If you take one thing from Chubby, take this: do *not* reimplement Paxos per-service; build (or adopt) one consensus system and route everyone through it.

### 2. Google Spanner — Paxos groups under a global database

Spanner shards data into tablets, and *each tablet is replicated by its own Paxos group* (a set of replicas, one per data center, running Multi-Paxos). A spanserver runs a Paxos state machine on top of each tablet; the group's leader holds a long-lived lease (10 seconds default) and pipelines writes through Phase 2. Cross-shard transactions layer **two-phase commit across Paxos groups** — 2PC for atomicity *between* groups, Paxos for durability and availability *within* each group, so a coordinator crash doesn't block the transaction the way classic 2PC would. This is the canonical "Paxos for replication, 2PC for distribution" architecture, and it's why Spanner can offer externally-consistent (linearizable) transactions across a planet-spanning database — bounded by TrueTime's clock uncertainty rather than by consensus latency on the read path.

### 3. Megastore — synchronous cross-datacenter replication

Megastore, Spanner's predecessor, was (per the [Consensus in the Cloud](https://cse.buffalo.edu/tech-reports/2016-02.pdf) survey) the largest system deployed using Paxos to replicate primary user data across data centers *on every write*. It extended Paxos to synchronously replicate multiple write-ahead logs, each governing its own partition. The cost — a wide-area Paxos round on every write — gave it strong consistency across regions but high write latency, which is precisely the tradeoff Spanner's leader-lease pipelining and TrueTime were built to improve. Megastore is the cautionary worked example of "consensus on the hot path of every write is correct but slow."

### 4. Apache Cassandra — lightweight transactions

Cassandra is an eventually-consistent store, but when you need a true compare-and-set — `INSERT ... IF NOT EXISTS`, `UPDATE ... IF col = val` — it runs **Paxos** to get linearizable consistency for that one operation. The [DataStax docs](https://docs.datastax.com/en/cassandra-oss/2.1/cassandra/dml/dml_ltwt_transaction_c.html) are refreshingly blunt about the cost: an LWT makes **four round trips** between the coordinator and the replicas (prepare/promise, read to check the condition, propose/accept, commit/apply), versus one for a normal write. Cassandra uses a `timeuuid` as the Paxos ballot to get monotonic-unique numbers for free, and exposes `SERIAL`/`LOCAL_SERIAL` consistency for reading possibly-uncommitted Paxos state. The operative guidance — which I'd frame onto a wall — is "reserve lightweight transactions for those situations where they are absolutely necessary; Cassandra's normal eventual consistency can be used for everything else." LWT is consensus *à la carte*: you pay the Paxos tax only on the specific rows that need it.

### 5. ZooKeeper / Zab — Paxos-adjacent, not Paxos

ZooKeeper's replication protocol, Zab (ZooKeeper Atomic Broadcast), is *consensus-equivalent* but deliberately *not* Paxos. Zab is built around a strong leader and a strict total order with a primary-order guarantee that plain Paxos doesn't provide — it ensures that a new leader's state strictly extends the previous one's, which matters for ZooKeeper's "everything is a totally-ordered sequence of state changes" model. It belongs in the same family (total order broadcast via a leader) and is in spirit closer to Raft than to single-decree Paxos. I include it because engineers routinely say "ZooKeeper uses Paxos" — it doesn't, quite, and the distinction (Zab guarantees primary order; basic Paxos doesn't) is exactly the kind of nuance that separates citing from understanding.

### 6. etcd and the Raft generation

The modern default — etcd, Consul, CockroachDB, TiKV, and Kubernetes' entire control plane (which stores all state in etcd) — runs **Raft**, not Paxos. This is the clearest evidence for the recommendation above: when the industry got to build consensus from scratch in the 2010s, it overwhelmingly chose Raft for its implementability. These systems are "Paxos" only in the genealogical sense that Raft is Multi-Paxos with the details nailed down. The practical upshot: if you are deploying consensus today and not maintaining a legacy Paxos system, you are almost certainly deploying Raft, and that is the correct choice.

### 7. The lease that outlived its clock (a composite war story)

A second pattern, distinct from the durability bug, comes from the *liveness* scaffolding rather than the safety core. A Multi-Paxos system used leader leases to serve fast local reads — the leader, holding a lease "valid for the next 8 seconds," answered reads without a quorum round trip, exactly the Spanner-style optimization. The safety argument was: while my lease is valid, no other leader can have committed a write, so my local state is fresh. The bug was that the lease's validity was checked against the *leader's own* clock, and one node's clock ran slow by several seconds after an NTP glitch. The cluster, seeing the leader silent, elected a new leader and granted it a lease. The old leader, with its slow clock, still believed its old lease was valid — and kept serving stale reads from a state that was now behind the new leader's commits. Two leaders, briefly, each confident, each serving reads: a classic split brain, but one that *never violated agreement* on the write path (Paxos saw to that) while still returning wrong answers to clients. The fix was to bound the lease against clock *uncertainty*, not a raw timestamp — grant the lease for `duration - max_clock_drift` and never trust a single node's clock — which is the whole reason Spanner built TrueTime. The lesson: the moment you trade a message round trip for a clock assumption to make reads cheap, your correctness now depends on a clock-drift bound, and an unbounded clock is an unbounded correctness hole. Safety on the write path does not save you from a broken read-path optimization.

### 8. The homegrown-Paxos incident (a composite war story)

I'll close the case studies with the pattern I've seen break more than once, assembled into one cautionary tale. A team needs a small amount of strongly-consistent coordination — say, a singleton job scheduler that must not run twice. They decide ZooKeeper/etcd is "too heavy" and implement a quick Paxos themselves over their existing RPC layer. It works in testing. In production, two jobs run simultaneously one day. The post-mortem finds the cause: their acceptors stored promises in memory and flushed to disk *asynchronously*; a node crashed after promising-and-accepting but before the flush, came back having forgotten its acceptance, and promptly accepted a *different* value at a higher ballot — splitting the quorum and choosing two values. The fix was a one-line `fsync` before every reply, but the lesson is the meta-lesson of this whole article: the parts of Paxos that look like incidental implementation details — durable acceptor state, value carry-forward, gap-filling, unique monotonic ballots — *are the algorithm*. Skip any one of them and you have built something that passes tests and violates agreement. The right move was never to hand-roll it; it was to use the one battle-tested consensus system you already had.

## 11. Second-order consequences worth internalizing

A few non-obvious implications that fall out of everything above, the kind that distinguish people who *understand* Paxos from people who can *recite* it.

- **"Chosen" can outrun "known."** A value can be chosen — irrevocably — before any process knows it. A proposer whose Phase 2 acks were all lost has chosen a value and has no idea. This is why crash recovery in Multi-Paxos must *re-discover* the log state via Phase 1 quorum reads rather than trusting any single node's local view. Designs that assume "if no one acked, nothing was chosen" are subtly broken.
- **Safety is free; liveness is bought with timing assumptions.** Every leader, every lease, every backoff, every election timeout exists *only* for liveness. None of them is load-bearing for agreement. If you find yourself reasoning "this is safe because the leader…", stop — your safety argument should never depend on there being exactly one leader. It should hold even with zero or two.
- **The quorum is the unit of trust, and its size is a knob.** Five acceptors, majority of three, tolerate two failures. The intersection requirement (sharpened by Flexible Paxos to "across phases only") is what you're actually buying. Every "make consensus faster" idea — smaller Phase 2 quorums, fast paths, commutativity — is ultimately a different way to slice the quorum while preserving intersection.
- **Reads are writes in disguise.** A *fresh*, linearizable read in plain Paxos is a quorum operation, as expensive as a write, because you must confirm you're seeing the latest chosen value. Leader leases make reads cheap by substituting a *clock* assumption for a *message* round trip — which is why every fast-read consensus system has a clock-drift bound hiding in its correctness argument.
- **More nodes is not more safety past a point.** Going from three to five acceptors buys you one more tolerated failure and costs you a larger quorum on every operation. Going from five to seven costs latency for marginal availability gain. Real clusters live at three or five for a reason; "we'll add nodes to be safe" usually makes the system slower and *less* available, not more.

The throughline, if there is one: Paxos is not complicated because consensus is complicated. Consensus, stated as Lamport states it, is almost trivial — three safety properties and a failure model. Paxos is *subtle* because its safety is **emergent and non-local**, arising from set intersection rather than from any component you can point at, and because the FLP wall forces a permanent, uncomfortable separation between the safety you always get and the liveness you can only ever approximate. Once you see that — that the two phases exist to maintain P2c, that P2c exists to maintain agreement, that agreement reduces to quorum intersection, and that everything labeled "leader" or "lease" is liveness scaffolding bolted onto an always-safe core — Paxos stops being the algorithm everyone cites and few understand, and becomes the algorithm you can derive on a whiteboard from first principles. Which is, after all, exactly what Lamport claimed in that one-sentence abstract.

## Further reading

- Leslie Lamport, ["Paxos Made Simple"](https://lamport.azurewebsites.net/pubs/paxos-simple.pdf) (2001) — the nine-page derivation this article follows. Read it after this; it will click.
- Leslie Lamport, ["The Part-Time Parliament"](https://lamport.azurewebsites.net/pubs/lamport-paxos.pdf) (1998) — the original, infamous Greek-parliament allegory. Historically essential, pedagogically punishing.
- Chandra, Griesemer, Redstone, ["Paxos Made Live — An Engineering Perspective"](https://www.semanticscholar.org/paper/Paxos-made-live:-an-engineering-perspective-Chandra-Griesemer/76aad5b272219c6745c76b7874129797e97e6041) (2007) — the Chubby team on the 95% the algorithm papers omit.
- Ongaro and Ousterhout, ["In Search of an Understandable Consensus Algorithm" (Raft)](https://raft.github.io/raft.pdf) (2014) — the modern default, designed for human comprehension.
- Howard, Malkhi, Spiegelman, ["Flexible Paxos: Quorum Intersection Revisited"](https://arxiv.org/abs/1608.06696) (2016) — the theorem that majorities were always an over-constraint.
- Moraru, Andersen, Kaminsky, ["There Is More Consensus in Egalitarian Parliaments" (EPaxos)](https://www.usenix.org/conference/nsdi13/technical-sessions/paper/moraru) (2013) — leaderless, commutativity-aware consensus.
- Lamport, ["Fast Paxos"](https://link.springer.com/article/10.1007/s00446-006-0005-x) (2006) — trading cluster size for a message delay.
- Corbett et al., ["Spanner: Google's Globally-Distributed Database"](https://www.cs.cornell.edu/courses/cs5414/2017fa/papers/Spanner.pdf) (2012) — Paxos groups, leader leases, and 2PC across them.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 9 — the best book-length treatment of why consensus, total order broadcast, and linearizable CAS are the same problem.
- Sibling posts on this blog: [Raft from scratch](/blog/software-development/database/raft-consensus-from-scratch), [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual), and [CAP and PACELC](/blog/software-development/database/cap-theorem-and-pacelc).

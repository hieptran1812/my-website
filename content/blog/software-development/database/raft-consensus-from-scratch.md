---
title: "Raft: Implementing Distributed Consensus from Scratch"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch tour of the Raft consensus algorithm — replicated state machines, leader election, log replication, the commit-safety rules, membership changes, snapshots, and linearizable reads — with runnable RPC code, comparison tables, and a guide to operating Raft in production."
tags:
  [
    "raft",
    "consensus",
    "distributed-systems",
    "leader-election",
    "replicated-log",
    "etcd",
    "cockroachdb",
    "fault-tolerance",
    "state-machine-replication",
    "databases",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/raft-consensus-from-scratch-1.webp"
---

Every database that promises you "strong consistency across three datacenters" is running a small, paranoid program in the background whose only job is to make a handful of machines agree on a single ordered list of facts. That program is a consensus algorithm, and for most systems shipped in the last decade it is Raft. When you `kubectl apply` a deployment, a Raft cluster inside etcd decides the canonical order of that change. When CockroachDB commits a row, a per-range Raft group votes on it. When Kafka 4.0 elects a partition leader, a Raft quorum of controllers writes that decision down. The algorithm is small enough to implement in a weekend and subtle enough that the people who first implemented it wrong shipped silent data-loss bugs that took years to surface.

This article builds Raft from the ground up — not as a list of RPCs to memorize, but as a sequence of forced moves. We start from the question consensus actually answers (how do machines agree on one value despite crashes and a network that lies?), watch why that question is provably unsolvable in the worst case, and then watch Raft sidestep the impossibility with a single cheap trick: randomized timeouts. From there every piece falls out in order — a replicated log, a leader to order it, an election protocol to pick the leader, a set of voting and commit rules to keep the whole thing safe when leaders die mid-write, and finally the operational machinery (membership changes, snapshots, linearizable reads) that turns the paper into something you can page at 3 a.m.

The diagram below is the mental model the rest of the article unpacks: consensus does not replicate your *data*, it replicates the *log of commands* that produces your data, and as long as every node applies the same commands in the same order to the same deterministic state machine, every node ends up byte-for-byte identical. Replicate the recipe, not the cake.

![A client command flows to the leader, which replicates the identical ordered command log to three nodes, each of which deterministically applies it to reach the same state x=8](/imgs/blogs/raft-consensus-from-scratch-1.webp)

> Consensus is not about agreeing on a value once. It is about agreeing on the same *infinite sequence* of values forever, while machines crash, restart with amnesia, and the network delays, reorders, duplicates, and drops your messages at the worst possible moment.

## Why this is different from what most engineers assume

Most engineers carry a mental model of replication that is roughly "the leader writes, then copies the bytes to the followers." That model is not wrong so much as it is dangerously incomplete — it omits every case that consensus exists to handle. The gap between the comfortable assumption and the adversarial reality is the entire reason Raft has the shape it does.

| Assumption | The comfortable mental model | The distributed reality Raft must survive |
| --- | --- | --- |
| "The leader is the leader." | One machine is in charge and stays in charge. | The leader can crash, freeze for a GC pause, or get partitioned away while *still believing it is leader*. Two machines can think they are leader at once. |
| "A write that reached a majority is safe." | Once it's on most nodes, it's committed. | An entry on a majority can still be silently overwritten by a future leader — unless it was written in the leader's *current* term. This is the Figure-8 hazard. |
| "Followers just copy the leader's log." | Append-only, monotonic, clean. | Followers can have *extra* entries from a crashed old leader that must be deleted, or be missing entries that must be back-filled. Logs diverge and must be repaired. |
| "Reads are free; only writes need coordination." | Read from any node. | A deposed leader that hasn't noticed it lost the election will happily serve stale reads forever. Linearizable reads need their own protocol. |
| "Adding a node is just config." | Update a list, restart. | Changing membership naively can create two disjoint majorities — two leaders, split brain, lost commits. Membership change is a consensus problem itself. |

Every row in the right column is a real failure that Raft's rules close. The discipline of the rest of this article is to introduce each rule alongside the failure it prevents — never a rule for its own sake.

## 1. What consensus is, and why it is genuinely hard

> **Senior rule of thumb:** if you ever find yourself writing "and then the nodes just agree on which one wins," stop — that sentence is the entire hard part, and getting it wrong is how you lose committed data.

Consensus, formally, is the problem of getting a set of processes to agree on a single value, subject to four properties. *Validity*: the agreed value was proposed by some process (you can't agree on garbage nobody suggested). *Agreement*: no two correct processes decide different values. *Integrity*: each process decides at most once. *Termination*: every correct process eventually decides. The first three are *safety* — nothing bad ever happens. The fourth is *liveness* — something good eventually happens. The whole difficulty of consensus is that the network conspires to make you trade one against the other.

Why is this hard? Because the network is asynchronous and the machines can fail. A message that hasn't arrived is indistinguishable, from the receiver's point of view, from a message that will *never* arrive because the sender crashed. You cannot tell "slow" from "dead." If node A is waiting for node B's vote and the vote doesn't come, A has no way to know whether B is computing, B's reply is stuck in a router, or B's motherboard is on fire. Any rule of the form "wait until B responds" can hang forever; any rule of the form "give up on B after a timeout and proceed without it" can proceed *while B is actually alive and doing the opposite thing* — which is how you get two leaders.

To feel the difficulty concretely, take the most naive "algorithm" a beginner reaches for: *the node with the highest ID is the leader.* It needs no voting, no terms, nothing. Now run the failure. Five nodes, N1 through N5; N5 is leader. A network partition splits the cluster into {N1, N2} and {N3, N4, N5}. On the {N3, N4, N5} side, N5 is still the highest ID and still leader — fine. But on the {N1, N2} side, N1 and N2 cannot reach N5, and by the rule "highest reachable ID leads," N2 declares *itself* leader of its partition. Now there are two leaders. A client writes `x=1` to N2's partition and `x=2` to N5's partition; both "commit" locally; both sides acknowledge their clients. When the partition heals, there is no principled way to reconcile — two acknowledged, conflicting writes exist, and one must be silently discarded. *Agreement* is violated. This is split brain, and it is the single failure every consensus algorithm exists to prevent. The fix is not a smarter leader-selection heuristic; it is the insight that **a leader must be chosen by a majority, and a majority of a partitioned cluster can exist on at most one side.** The {N1, N2} side is a minority of five, so it can *never* assemble a majority, so N2 can never legitimately lead — it is forced to reject the client and wait. The minority sacrifices availability to preserve agreement. That majority-quorum rule is the seed from which all of Raft grows.

### The FLP impossibility result

This intuition was made precise in 1985 by Fischer, Lynch, and Paterson, in a result usually abbreviated FLP. They proved that in a fully asynchronous system (no bound on message delay, no synchronized clocks), with even a *single* process that may crash, there is **no deterministic algorithm that guarantees consensus**. Not "it's slow," not "it's expensive" — it is mathematically impossible to guarantee both safety and termination. The proof constructs an adversarial schedule: for any algorithm, there exists a sequence of message delays that keeps the system in a "bivalent" state — a state where the outcome could still go either way — forever, by always delaying the one message that would tip the decision.

FLP sounds like it kills the entire field. It doesn't, and understanding *why* is the key to understanding every practical consensus system. FLP forbids an algorithm that is *always* safe and *always* live in a *fully asynchronous* model. Real systems escape through a deliberate, narrow crack: they keep safety unconditional and make liveness *conditional on timing*. Raft is never allowed to elect two leaders for the same term or commit conflicting entries — safety holds even if the network behaves like a malicious genie. But Raft only *makes progress* (elects a leader, commits entries) during periods when the network is "well-behaved enough" — when messages mostly arrive within the timeout window. The famous quote from the field is that consensus algorithms "are safe under asynchrony and live under partial synchrony."

### How Raft sidesteps FLP with randomized timeouts

The specific liveness hazard FLP exploits, in a leader-based protocol, is the *split vote*: several nodes notice the leader is gone, all become candidates at the same instant, all vote for themselves, and nobody gets a majority. Repeat forever, and you have FLP's perpetual indecision in the flesh.

Raft breaks this not with cleverness but with *randomness*. Each follower waits a random election timeout — in the canonical implementation, a value drawn uniformly from `[150ms, 300ms]` — before deciding the leader is dead and starting an election. Because the timeouts are randomized and re-randomized every term, the probability that two nodes start at the exact same moment shrinks with every retry. One node almost always times out first, grabs the votes, and becomes leader before anyone else even wakes up. A split vote can still happen; it just can't happen *forever*, because each re-election rolls fresh dice. FLP says you can't *guarantee* termination; randomized timeouts make non-termination a measure-zero event you can ignore in practice. That single design decision — pay for liveness with probability instead of a impossible guarantee — is the philosophical core of Raft.

```python
import random

# The entire anti-FLP trick in three lines. Re-rolled at the start of every
# wait, so two nodes that tie this round almost certainly won't tie the next.
ELECTION_TIMEOUT_MIN_MS = 150
ELECTION_TIMEOUT_MAX_MS = 300

def next_election_timeout_ms() -> int:
    return random.randint(ELECTION_TIMEOUT_MIN_MS, ELECTION_TIMEOUT_MAX_MS)

# Rule of thumb from the Raft paper: pick the timeout range so that
# broadcastTime << electionTimeout << MTBF.
#   broadcastTime  ~ 0.5–20 ms  (one round trip to followers)
#   electionTimeout ~ 150–300 ms
#   MTBF           ~ months
# If electionTimeout is too low, healthy leaders get falsely deposed; too high,
# and failover takes seconds. 10x the round-trip time is the usual starting point.
```

## 2. The replicated state machine: replicate the log, not the data

> **Senior rule of thumb:** the moment you can phrase your service as "a deterministic function applied to an ordered log of commands," consensus becomes a solved problem you can buy off the shelf. If you can't, no consensus algorithm will save you.

The mental-model figure at the top of this article is the *replicated state machine* (RSM) model, and it is the abstraction that makes consensus useful. The idea: instead of replicating the *state* of your service (the rows, the key-value pairs, the current `x=8`), you replicate the *log of commands* that produced it (`x=5`, then `x+=3`). Each node keeps an identical copy of this log and feeds it, entry by entry in the same order, into the same deterministic state machine. Determinism is the load-bearing word: given the same command sequence, the state machine must produce the same state on every node, with no dependence on wall-clock time, random numbers, iteration order of a hash map, or anything else that varies between machines. If determinism holds, identical logs imply identical state — automatically, with no further coordination.

This is a profound decomposition. It splits the problem cleanly in two. The *hard* half — getting every node to agree on the same log, in the same order, despite faults — is the consensus algorithm, and it is identical for every application. The *easy* half — applying a command to your data — is your business logic, and consensus never has to understand it. Raft does not know what `x+=3` means; it only guarantees that every node sees `x+=3` at the same position in the log. This is also why the model is sometimes called *total order broadcast* or *atomic broadcast*: the consensus layer broadcasts each command to all nodes such that everyone delivers the same messages in the same total order. Kleppmann's *Designing Data-Intensive Applications* makes the equivalence explicit in Chapter 9 — consensus, total order broadcast, and linearizable compare-and-set are all reducible to one another. Solve one and you have solved them all; Raft solves the log-ordering one because logs are concrete and easy to reason about.

The RSM model also dictates a subtlety that trips up first-time implementers: the consensus log and the state machine are *separate*. An entry being in the log does not mean it has been applied. An entry is **committed** when consensus guarantees it will never be lost, and only then is it *applied* to the state machine and its effect made visible to clients. The gap between "in the log" and "committed" and "applied" is exactly where Raft's safety rules live.

### Raft's bet: understandability over the Paxos lineage

For two decades before Raft, the textbook answer to consensus was Leslie Lamport's Paxos. Paxos is correct, it is foundational, and it is — by near-universal admission, including from people who teach it — extraordinarily hard to understand and even harder to implement correctly. The single-decree Paxos paper proves how to agree on *one* value; turning that into the *log* of values a real system needs (Multi-Paxos) involves a pile of unstated engineering decisions that every implementer must reinvent, and that the original papers gloss over. The result, as Ongaro and Ousterhout note, is that real-world "Paxos" systems are each subtly different, hard to verify, and built on folklore as much as on the published algorithm.

Raft's central design decision is to optimize for a property the field had treated as secondary: *understandability*. The authors ran an actual user study (students taught Paxos vs Raft, then quizzed) and found Raft meaningfully easier to learn. They achieved this with two main techniques. First, **decomposition**: Raft splits consensus into three nearly independent sub-problems — leader election, log replication, and safety — that can be understood one at a time, where Paxos blends them. Second, **reducing the state space**: Raft enforces stronger invariants (a strong leader that is the *only* source of log entries, logs that flow strictly leader-to-follower, the election restriction that prevents a stale leader from ever winning) to eliminate whole classes of states the implementer would otherwise have to reason about. Paxos permits more concurrency and more symmetric roles; Raft deliberately gives that up for a model you can hold in your head.

| Dimension | (Multi-)Paxos | Raft |
| --- | --- | --- |
| Primary design goal | Theoretical minimality, maximum concurrency | Understandability and implementability |
| Leadership | Optional / symmetric; any node can propose | Mandatory strong leader; all entries originate at the leader |
| Log flow | Entries can be filled in any order; gaps allowed | Strictly contiguous, leader-to-follower only |
| Stale-node leadership | A behind node can lead, then back-fill | Forbidden — election restriction blocks it at vote time |
| Membership change | Underspecified in original papers | First-class: joint consensus / single-server change |
| Where the complexity lives | Reconstructing a coherent log from independent decisions | A few sharp safety rules (Figure-8 commit, up-to-date vote) |
| Typical implementation outcome | Each system's "Paxos" is subtly unique | Reusable libraries (etcd raft) shared across many systems |

The bet paid off: the existence of high-quality, *reused* Raft libraries — etcd's `raft` underpinning Kubernetes, CockroachDB, and (via `raft-rs`) TiKV — is itself evidence. Paxos rarely gets reused across organizations because each implementation is a bespoke reading of an ambiguous spec; Raft does, because the spec is precise enough to implement the same way twice. Raft is not *more powerful* than Multi-Paxos — the paper is explicit that they are equivalent in fault tolerance and performance — it is more *teachable* and more *reproducible*, and for production engineering those are the properties that matter.

## 3. The three server states and the role of terms

> **Senior rule of thumb:** in Raft there is at most one leader per term, and a node that sees a higher term than its own immediately surrenders. Terms are the logical clock that makes "who is in charge?" a question with a single answer.

Raft simplifies the design space by giving every server exactly one of three roles at any instant, and a small set of transitions between them.

![A state diagram: start leads to follower; a follower that times out becomes a candidate; a candidate with a majority becomes leader; any server seeing a higher term on an RPC steps down to follower](/imgs/blogs/raft-consensus-from-scratch-2.webp)

A **follower** is passive. It issues no requests of its own; it only responds to RPCs from leaders and candidates, and it resets its election timer every time it hears from a legitimate leader. A **candidate** is a follower that grew impatient — it timed out without hearing from a leader, so it bumped the term, voted for itself, and is now soliciting votes. A **leader** is a candidate that won a majority of votes; it handles all client requests and pumps out periodic heartbeats to keep the followers' timers from firing. The transitions are exactly the arrows in the figure: a follower whose election timer fires becomes a candidate; a candidate that wins a majority becomes leader; and — the rule that prevents most split-brain disasters — *any* server that observes a higher term than its own on any RPC immediately reverts to follower, adopts the new term, and clears its vote.

That last rule depends on **terms**, Raft's most important conceptual tool. Time is divided into terms, numbered with consecutive integers. Each term begins with an election; if the election produces a leader, that leader serves for the rest of the term; if it produces a split vote, the term ends with no leader and a new term (with a new election) begins. Terms are a *logical clock* — they let nodes detect stale information. Every RPC carries the sender's term. When a node receives an RPC with a term greater than its own, it knows its own information is out of date and updates. When it receives an RPC with a term *less* than its own, it knows the sender is stale and rejects the message. There can be at most one leader in any given term (we'll prove why below), so "the current term's leader" is always a well-defined, single entity. The term number is the answer to "whose claim to leadership is more recent?" — and recency, not seniority or identity, is what wins.

Crucially, three pieces of state must survive a crash and reboot, persisted to stable storage *before* responding to any RPC: `currentTerm` (so a rebooted node doesn't accept a stale leader), `votedFor` (so a rebooted node doesn't vote twice in the same term), and `log[]` (so committed entries are never lost). Everything else — `commitIndex`, `lastApplied`, the leader's `nextIndex[]` and `matchIndex[]` — is volatile and rebuilt after a restart.

```python
from dataclasses import dataclass, field
from enum import Enum

class Role(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    term: int          # term in which the leader created this entry
    command: bytes     # opaque to Raft; meaningful only to the state machine

@dataclass
class RaftServer:
    # --- Persistent state (fsync to disk BEFORE replying to any RPC) ---
    current_term: int = 0
    voted_for: int | None = None         # candidateId voted for in current_term
    log: list[LogEntry] = field(default_factory=list)   # 1-indexed in the paper

    # --- Volatile state on all servers (rebuilt after restart) ---
    commit_index: int = 0                # highest log entry known committed
    last_applied: int = 0                # highest log entry applied to the SM

    # --- Volatile state on leaders only (reinitialized after each election) ---
    next_index: dict[int, int] = field(default_factory=dict)   # per follower
    match_index: dict[int, int] = field(default_factory=dict)  # per follower

    # --- Runtime ---
    role: Role = Role.FOLLOWER
    peers: list[int] = field(default_factory=list)

    def step_down(self, new_term: int) -> None:
        """The universal rule: see a higher term, become a follower."""
        if new_term > self.current_term:
            self.current_term = new_term
            self.voted_for = None
            self.role = Role.FOLLOWER
            self.persist()   # currentTerm + votedFor must hit disk here

    def persist(self) -> None:
        ...  # fsync current_term, voted_for, log[] to stable storage
```

## 4. Leader election and the RequestVote RPC

> **Senior rule of thumb:** a vote is a one-time, per-term resource. Once a node has voted in term T, it will never vote again in term T — that single invariant is what guarantees one leader per term.

When a follower's election timer fires without a heartbeat, it starts an election: it increments `currentTerm`, transitions to candidate, votes for itself, persists, and sends a `RequestVote` RPC to every other server in parallel. The RPC carries four fields: the candidate's `term`, its `candidateId`, and — critically for safety — `lastLogIndex` and `lastLogTerm`, describing how up-to-date the candidate's log is.

A candidate wins the election if it collects votes from a *majority* of the cluster (including its own). Majority — `floor(N/2) + 1` — is the quorum that makes everything work: any two majorities of the same cluster must overlap in at least one node, and that overlapping node is what prevents two candidates from both winning the same term (it can only vote once). This is why Raft clusters are almost always odd-sized: a 3-node cluster tolerates 1 failure (majority of 2), a 5-node cluster tolerates 2 (majority of 3). A 4-node cluster also needs 3 for a majority, so it tolerates only 1 failure — the same as 3 nodes, but with more machines to fail. Odd sizes are strictly better.

The candidate side of the election is a short driver. It fans out `RequestVote` to every peer, tallies grants, and resolves into one of three outcomes — won, lost to a higher term, or timed out into another election:

```python
def run_election(self: RaftServer) -> None:
    """Candidate side. Called when the election timer fires."""
    self.current_term += 1                 # bump to a fresh term
    self.role = Role.CANDIDATE
    self.voted_for = self.id               # vote for ourselves
    self.persist()                         # currentTerm + votedFor durable first
    self.reset_election_timer()            # fresh random timeout for THIS attempt

    votes = 1                              # our own vote
    args = RequestVoteArgs(
        term=self.current_term, candidate_id=self.id,
        last_log_index=len(self.log),
        last_log_term=self.log[-1].term if self.log else 0,
    )
    for peer in self.peers:
        reply = self.send_request_vote(peer, args)     # in parallel in real code
        if reply.term > self.current_term:
            self.step_down(reply.term)     # someone is ahead of us; abdicate
            return
        if self.role != Role.CANDIDATE or reply.term != self.current_term:
            return                         # state changed under us; bail out
        if reply.vote_granted:
            votes += 1
            if votes >= len(self.peers) // 2 + 2:      # +2: peers excludes self
                self.become_leader()       # majority reached — we win this term
                return
    # Fell through without a majority: a split vote. The election timer will fire
    # again with a NEW random timeout, and we retry in a higher term.
```

When a candidate wins, `become_leader` reinitializes `nextIndex[i] = len(log) + 1` and `matchIndex[i] = 0` for every follower, then immediately broadcasts a heartbeat (empty `AppendEntries`) to assert authority and stop other nodes' timers. Most implementations also append a no-op entry here to commit something in the new term promptly, which (per the Figure-8 rule) commits any inherited prior-term entries and unblocks linearizable reads.

The receiver's logic for `RequestVote` is short but every clause earns its place:

```python
@dataclass
class RequestVoteArgs:
    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int

@dataclass
class RequestVoteReply:
    term: int
    vote_granted: bool

def handle_request_vote(self: RaftServer, args: RequestVoteArgs) -> RequestVoteReply:
    # 1. A stale candidate (older term) is rejected outright; we tell it our
    #    term so it can step down.
    if args.term < self.current_term:
        return RequestVoteReply(self.current_term, vote_granted=False)

    # 2. A newer term means our info is stale — adopt it and become a follower.
    #    This also resets voted_for, freeing us to vote in the new term.
    if args.term > self.current_term:
        self.step_down(args.term)

    # 3. Grant the vote only if we haven't already voted this term for someone
    #    else, AND the candidate's log is at least as up-to-date as ours.
    already_voted = self.voted_for not in (None, args.candidate_id)
    if not already_voted and self.candidate_is_up_to_date(args):
        self.voted_for = args.candidate_id
        self.persist()                 # votedFor must be durable before we reply
        self.reset_election_timer()    # we backed a candidate; don't race it
        return RequestVoteReply(self.current_term, vote_granted=True)

    return RequestVoteReply(self.current_term, vote_granted=False)

def candidate_is_up_to_date(self: RaftServer, args: RequestVoteArgs) -> bool:
    """The election restriction (see section 7). Compare last-log term first;
    break ties by length. A candidate whose log is staler than ours is refused."""
    my_last_term = self.log[-1].term if self.log else 0
    my_last_index = len(self.log)
    if args.last_log_term != my_last_term:
        return args.last_log_term > my_last_term
    return args.last_log_index >= my_last_index
```

### Randomized timeouts and the split vote, walked through

The figure below traces a real failover, including a split vote and how randomization resolves it. The key is that *each* election uses a freshly randomized timeout, so even after a tie, the next round almost surely has a clear winner.

![A timeline: term 1 leader L1 heartbeats then crashes; in term 2 two followers time out together and split the vote 2-2; in term 3 one node's 170ms timeout fires first and it wins a clean majority](/imgs/blogs/raft-consensus-from-scratch-3.webp)

Read it left to right. In term 1, leader L1 is healthy and heartbeating; everyone's election timer keeps resetting. At t=160ms, L1 crashes. The followers stop hearing heartbeats and their timers start counting down toward expiry. In the diagram's unlucky term 2, two followers (N2 and N3) happen to time out at nearly the same instant, both become candidates for term 2, both vote for themselves, and the cluster ends up split 2-2 with no majority — a *split vote*. No leader emerges; term 2 dies barren. But here is the escape: when the term-2 election fails, each candidate sets a *new* random timeout before trying again. In term 3, N2's timer (say 170ms) fires meaningfully before N3's, so N2 requests votes first, collects a majority while N3 is still waiting, and becomes the term-3 leader cleanly. The randomization doesn't make split votes impossible — it makes a *run* of them exponentially unlikely, which is exactly the FLP-sidestep from section 1 in action.

Heartbeats are just `AppendEntries` RPCs with no entries. The leader sends them every `heartbeatInterval` (well below the election timeout — typically 50ms against a 150-300ms election window) so that followers' timers keep resetting and nobody starts a spurious election while the leader is alive.

## 5. Log replication and the AppendEntries RPC

> **Senior rule of thumb:** the leader never asks followers what they have; it *tells* them what they should have and lets each follower's consistency check reject anything that doesn't line up. Replication is leader-driven and idempotent.

Once a leader is established, it serves client requests. Each request contains a command; the leader appends it to its own log as a new entry (tagged with the current term and its index), then issues `AppendEntries` RPCs in parallel to all followers to replicate it. When the entry has been safely stored on a majority of servers, the leader marks it *committed*, applies it to its state machine, and returns the result to the client. The followers learn of the commit on the next `AppendEntries` and apply the entry to their own state machines.

![A timeline: a client write is appended at index 6; AppendEntries goes to N2 and N3; N2 persists giving a 2-of-3 majority; commitIndex advances to 6; the entry is applied and the client is acked while N3 catches up later](/imgs/blogs/raft-consensus-from-scratch-4.webp)

The figure above is the happy path: the client write lands at index 6, the leader replicates to N2 and N3, N2's acknowledgment brings the count to 2 of 3 (a majority), so the leader advances `commitIndex` to 6, applies the command, and replies to the client — all *before* N3 has necessarily caught up. N3 will receive the entry on a subsequent `AppendEntries` and apply it then. Commitment is about a *majority*, not unanimity; that's what lets the cluster make progress with a node down.

### The AppendEntries RPC and the consistency check

`AppendEntries` carries the leader's `term`, its `leaderId`, the index and term of the entry *immediately preceding* the new ones (`prevLogIndex`, `prevLogTerm`), the `entries[]` to append, and the leader's `leaderCommit`. The `prevLogIndex`/`prevLogTerm` pair is the heart of the protocol — it is the *consistency check* that enforces the Log Matching Property.

```python
@dataclass
class AppendEntriesArgs:
    term: int
    leader_id: int
    prev_log_index: int        # index of entry immediately before new ones
    prev_log_term: int         # term of that entry
    entries: list[LogEntry]    # empty for heartbeats
    leader_commit: int         # leader's commitIndex

@dataclass
class AppendEntriesReply:
    term: int
    success: bool

def handle_append_entries(self: RaftServer, args: AppendEntriesArgs) -> AppendEntriesReply:
    # 1. Reject a stale leader.
    if args.term < self.current_term:
        return AppendEntriesReply(self.current_term, success=False)

    # 2. A valid leader for this (or a newer) term: adopt term, become follower,
    #    and reset the election timer — we just heard from the leader.
    if args.term > self.current_term:
        self.step_down(args.term)
    self.role = Role.FOLLOWER
    self.reset_election_timer()

    # 3. CONSISTENCY CHECK: our log must contain an entry at prev_log_index
    #    whose term matches prev_log_term. If not, we have a gap or a conflict;
    #    reject so the leader backs up. (prev_log_index == 0 means "from start".)
    if args.prev_log_index > 0:
        if len(self.log) < args.prev_log_index:
            return AppendEntriesReply(self.current_term, success=False)  # gap
        if self.log[args.prev_log_index - 1].term != args.prev_log_term:
            return AppendEntriesReply(self.current_term, success=False)  # conflict

    # 4. Append entries, deleting any conflicting suffix first. An existing entry
    #    that conflicts (same index, different term) and everything after it is
    #    truncated, then the new entries are appended.
    for i, entry in enumerate(args.entries):
        idx = args.prev_log_index + 1 + i        # 1-based log position
        if idx <= len(self.log):
            if self.log[idx - 1].term != entry.term:
                del self.log[idx - 1:]           # truncate conflicting suffix
                self.log.append(entry)
        else:
            self.log.append(entry)
    self.persist()

    # 5. Advance our commit index toward the leader's, but never past what we
    #    actually have. Newly committed entries get applied to the state machine.
    if args.leader_commit > self.commit_index:
        self.commit_index = min(args.leader_commit, len(self.log))
        self.apply_committed_entries()

    return AppendEntriesReply(self.current_term, success=True)
```

### The Log Matching Property

The consistency check buys a powerful invariant called the **Log Matching Property**, stated by Ongaro and Ousterhout as two parts: (1) if two entries in different logs have the same index and term, they store the same command; and (2) if two entries in different logs have the same index and term, then *the logs are identical in all preceding entries*. The first part follows because a leader creates at most one entry per index per term, and entries never move. The second part is *induced* by the consistency check: every successful `AppendEntries` verifies that the follower's entry at `prevLogIndex` matches the leader's; by induction, agreement at index `i` plus a successful append at `i+1` means agreement up through `i+1`. The net effect is that two logs agreeing on the *last* shared (index, term) pair agree on *everything* before it. This is what lets the leader treat the log as a prefix-comparable object and repair followers by finding a single agreement point.

### commitIndex, matchIndex, nextIndex

Three indices coordinate replication, and conflating them is the classic implementation bug:

| Index | Where it lives | Meaning | When it advances |
| --- | --- | --- | --- |
| `commitIndex` | every server | highest entry known to be committed (safe, will never be lost) | leader: when a majority's `matchIndex` reaches it *and* the entry is from the current term; follower: from `leaderCommit` |
| `matchIndex[i]` | leader only | highest entry *known* replicated on follower `i` | on a successful `AppendEntries` to `i` |
| `nextIndex[i]` | leader only | the *next* entry the leader will *try* to send follower `i` | optimistically set to `leaderLastIndex + 1` on election; decremented on rejection |

The leader advances its own `commitIndex` to `N` when `N > commitIndex`, a majority of `matchIndex[i] >= N`, **and** `log[N].term == currentTerm`. That last conjunct is the famous commit-safety rule of section 6 — and forgetting it is a silent data-loss bug.

### The commit-advance and state-machine apply loop

The leader recomputes `commitIndex` every time a follower's `matchIndex` changes. The computation is a median over `matchIndex`: sort the `matchIndex` values (including the leader's own, which equals its last log index), and the value at the majority position is the highest index replicated on a majority. Then — and only then — clamp it to the current-term rule:

```python
def maybe_advance_commit_index(self: RaftServer) -> None:
    """Leader-only. Find the highest index replicated on a majority that is
    ALSO from the current term, and advance commitIndex to it."""
    # matchIndex for ourselves is our last log index.
    match = sorted([len(self.log)] + list(self.match_index.values()))
    # The (N - majority)-th element is the highest index a majority has reached.
    # For 5 nodes, majority=3, so index match[len-3] = match[2] (0-based).
    majority_matched = match[len(match) - (len(match) // 2 + 1)]
    for n in range(majority_matched, self.commit_index, -1):
        # CRITICAL: only commit by count if the entry is from OUR term.
        # This is the Figure-8 rule (section 6). Skipping it loses data.
        if self.log[n - 1].term == self.current_term:
            self.commit_index = n
            self.apply_committed_entries()
            break
```

The state-machine apply loop is the other half — the bridge between the consensus log and the actual service. It is deliberately separate from replication: replication moves entries into the log and advances `commitIndex`; the apply loop drains the gap between `lastApplied` and `commitIndex`, feeding each committed command into the deterministic state machine in strict index order, and (on the leader) routing the result back to the waiting client. This loop runs identically on every node; that identical execution is exactly what the replicated-state-machine model promises.

```python
def apply_committed_entries(self: RaftServer) -> None:
    """Runs on every server. Applies newly committed entries to the state
    machine in order. Deterministic apply is what makes all replicas converge."""
    while self.last_applied < self.commit_index:
        self.last_applied += 1
        entry = self.log[self.last_applied - 1]

        # Exactly-once: if this client+serial was already applied, return the
        # cached response instead of re-executing (see section 10).
        client_id, serial, op = self.decode(entry.command)
        cached = self.dedup_table.get(client_id)
        if cached and cached.serial >= serial:
            result = cached.response          # duplicate retry; do not re-apply
        else:
            result = self.state_machine.apply(op)   # the only place state mutates
            self.dedup_table[client_id] = Applied(serial=serial, response=result)

        # On the leader, wake the client RPC that is blocked on this index.
        if self.role == Role.LEADER and self.last_applied in self.pending_clients:
            self.pending_clients.pop(self.last_applied).set_result(result)
```

Notice the invariant chain: an entry is *appended*, then (once on a majority of the current term) *committed* by advancing `commitIndex`, then *applied* by the loop above, then — and only then — its result is visible to a client. Three distinct stages, three distinct indices (`len(log)`, `commitIndex`, `lastApplied`), and a bug that conflates any two of them is a correctness bug. The single most common first-implementation mistake is applying entries as soon as they are appended, before they are committed — which makes a follower act on an entry that a future leader might delete.

## 6. Safety: log repair and the Figure-8 commit hazard

> **Senior rule of thumb:** "it's on a majority of nodes" is *not* the same as "it's committed." A prior-term entry on a majority can still vanish. Only a current-term entry, once on a majority, drags the entries below it into permanent commitment.

Two safety mechanisms separate a correct Raft from a plausible-looking one that loses data: log repair (how the leader fixes divergent follower logs) and the commit rule (which entries the leader may declare committed). Both exist because leaders crash mid-write, leaving the cluster in messy states.

### Repairing divergent follower logs

When a new leader is elected, follower logs may diverge from it in two ways: a follower may be *missing* entries the leader has, or it may have *extra* uncommitted entries (from a previous leader that crashed before replicating them widely). The leader does not care which; it forces every follower's log to match its own by finding the latest point of agreement and overwriting everything after it.

![A before-after: a follower behind the leader has a conflicting entry at index 4 (term 2 vs the leader's term 3) so AppendEntries at index 5 fails; the leader backs nextIndex from 5 to 3, finds the match at index 3, and overwrites the tail so both logs agree](/imgs/blogs/raft-consensus-from-scratch-5.webp)

The mechanism is the `nextIndex` backtrack. On election, the leader optimistically sets `nextIndex[i] = leaderLastIndex + 1` for every follower and starts sending `AppendEntries` from there. If a follower's consistency check fails (its entry at `prevLogIndex` doesn't match), it replies `success=false`, the leader decrements `nextIndex[i]`, and retries with an earlier `prevLogIndex`. This walks backward one entry at a time until it hits an index where the follower's log agrees with the leader's. From that agreement point forward, the leader's entries overwrite the follower's — truncating the follower's conflicting suffix and back-filling whatever it was missing. The Log Matching Property guarantees that once the consistency check *passes* at some index, all earlier entries already agree, so the leader never has to back up further than necessary.

```python
def replicate_to(self: RaftServer, follower: int) -> None:
    """Leader-side replication loop for one follower. Backs off on rejection."""
    prev_index = self.next_index[follower] - 1
    prev_term = self.log[prev_index - 1].term if prev_index > 0 else 0
    entries = self.log[self.next_index[follower] - 1:]   # tail to send

    reply = self.send_append_entries(follower, AppendEntriesArgs(
        term=self.current_term, leader_id=self.id,
        prev_log_index=prev_index, prev_log_term=prev_term,
        entries=entries, leader_commit=self.commit_index,
    ))

    if reply.term > self.current_term:
        self.step_down(reply.term)              # we are stale; abdicate
        return

    if reply.success:
        # Everything we sent is now on the follower.
        self.match_index[follower] = prev_index + len(entries)
        self.next_index[follower] = self.match_index[follower] + 1
        self.maybe_advance_commit_index()
    else:
        # Consistency check failed: back up and retry. The naive version
        # decrements by 1; production code (etcd) returns a conflict hint so
        # the leader can skip a whole conflicting term in one round trip.
        self.next_index[follower] = max(1, self.next_index[follower] - 1)
```

The naive one-at-a-time decrement can be slow when a follower is far behind, so real implementations optimize it. The Raft paper itself suggests the follower return the term of the conflicting entry and the first index it stores for that term, letting the leader jump back a whole term per round trip. etcd's `go.etcd.io/raft` does exactly this with a `RejectHint`.

### The Figure-8 hazard: why counting replicas is not enough

Here is the subtlest and most famous part of Raft, the scenario the paper draws as Figure 8, and the one that catches every from-scratch implementer who reasons "an entry on a majority is committed."

![A grid walking through the Figure-8 scenario: S1 replicates a prior-term entry to a majority, crashes, comes back as leader, and re-replicates that entry to a majority — yet a different server could still become leader and overwrite it, unless the leader first commits a current-term entry above it](/imgs/blogs/raft-consensus-from-scratch-6.webp)

Walk the panels. **(a)** S1 is leader in term 2 and replicates the entry at index 2 (term 2) to S1 and S2 — a partial replication. **(b)** S1 crashes. S5 wins the term-3 election with votes from S3, S4, and S5 (whose logs lack the index-2 entry but are still "up to date" because, in this construction, the index-2 entry was never on a majority that would have refused S5). **(c)** S5 crashes before replicating anything; S1 restarts, wins term 4, and resumes replicating the *old* index-2 (term 2) entry, now copying it to a majority (S1, S2, S3). At this point the index-2 entry sits on a majority of servers. The tempting conclusion: it's committed. **(d)** is why that conclusion is *fatal*: if S1 declares index 2 committed based purely on the replica count, then crashes, S5 can come back, win term 5 (its log is longer in term 3), and *overwrite* the index-2 entry on everyone — erasing a "committed" entry. Agreement violated, data lost.

The fix is the commit rule: **a leader may only directly commit an entry by replica count if that entry is from the leader's *current* term.** In panel **(e)**, the safe S1 does *not* declare the old index-2 entry committed on count. Instead it appends a fresh index-3 entry in term 4 and replicates *that* to a majority. Once the term-4 entry is committed by count, the Log Matching Property drags index 2 along with it — index 2 is now committed *indirectly*, and the same construction that let S5 overwrite it before is now impossible, because any server that could win a future election must contain the committed term-4 entry, which (by log matching) means it contains index 2 too. The rule in one sentence: *count replicas only for your own term's entries; prior-term entries commit for free the moment a later current-term entry above them commits.* This is why the leader, on election, often appends a no-op entry immediately — to get a current-term entry committed fast and thereby commit any inherited prior-term entries.

## 7. The election restriction: only an up-to-date candidate can win

> **Senior rule of thumb:** Raft never lets a server with a stale log become leader, because a leader must have every committed entry — Raft chooses to enforce this at *vote* time rather than fixing it up afterward.

Raft makes a deliberate simplifying bet: a new leader must already contain *all* committed entries from prior terms, so that the log only ever flows from leader to followers — never the reverse. (The alternative, used by some Paxos variants, is to let any node become leader and then back-fill missing committed entries; Raft rejects this as harder to reason about.) The mechanism that enforces "the leader has everything committed" is the **election restriction**, and it lives inside the `RequestVote` handler we saw in section 4: a voter denies its vote to any candidate whose log is *less up-to-date* than its own.

![A graph: candidate A with lastTerm 3 and candidate B with lastTerm 2 both request votes from voters at lastTerm 3; the voters grant A because term 3 beats term 2, and deny B because its term 2 log is staler](/imgs/blogs/raft-consensus-from-scratch-7.webp)

"Up-to-date" has a precise definition: compare the *term* of the last log entries first; the higher term wins. If the last terms are equal, the *longer* log (higher last index) wins. In the figure, candidate A's last entry is in term 3 while candidate B's is in term 2 — so even though B's log is *longer* (index 9 vs A's index 8), A is more up-to-date, because last-term dominates last-index. Voters whose own last term is 3 grant A and refuse B.

Why does this guarantee the new leader has all committed entries? A committed entry, by definition, is stored on a majority. A candidate needs votes from a majority to win. Any two majorities overlap in at least one server. That overlapping server has the committed entry in its log, and — by the up-to-date comparison — it will only vote for a candidate whose log is *at least as up-to-date* as its own, which means the candidate's log must also reach at least that entry (or beyond). Therefore any candidate that *can* win must already contain every committed entry. The election restriction turns "the leader has all committed entries" from a property you'd have to repair after the fact into one that holds by construction the instant the leader is elected. Combined with the Figure-8 commit rule, this is the whole of Raft's safety argument: leaders are complete, and leaders never count prior-term replicas, so committed entries are durable forever.

## 8. Cluster membership changes

> **Senior rule of thumb:** never switch directly from the old config to the new one. For a brief window both could form independent majorities, and that window is a split-brain waiting to happen.

Real clusters need to add and remove servers — to replace dead hardware, scale capacity, or move a replica to another datacenter. The naive approach, swap the config and restart, is unsafe because the switch cannot happen atomically across all servers. There is necessarily an instant where some servers still use the old configuration and others use the new one. If, in that window, the old config can elect a leader from its majority *and* the new config can independently elect a different leader from *its* majority, you have two leaders for the same term — split brain.

![A timeline: phase 0 is C-old {N1,N2,N3}; in phase 1 the leader appends a joint C-old,new entry and every decision needs a majority of BOTH configs; once the joint entry commits, phase 2 appends and commits C-new {N1,N2,N3,N4,N5}](/imgs/blogs/raft-consensus-from-scratch-8.webp)

The Raft paper's original answer is **joint consensus**, a two-phase transition through a combined configuration `C_old,new` that overlaps both. In phase 1, the leader appends a special `C_old,new` configuration entry to the log and replicates it. While `C_old,new` is in effect, *every* decision — both leader elections and entry commitment — requires *separate* majorities from *both* `C_old` and `C_new`. This is the key: because both old and new must agree, neither can act alone, so no two disjoint leaders can be elected during the transition. Once `C_old,new` is committed, the leader appends a `C_new` entry; once *that* commits, the transition is complete and any server not in `C_new` can shut down. (A server adopts a configuration entry the moment it *appears* in its log, even before commitment — this is the one place Raft acts on uncommitted state, and it's safe precisely because of the overlap requirement.)

Joint consensus is correct but fiddly, so Ongaro's dissertation introduced a simpler alternative used by most production systems: **single-server changes**. By restricting membership changes to *one server added or removed at a time*, the old and new majorities are guaranteed to overlap (adding one server to a 3-node cluster makes it 4 nodes; the old majority of 2 and the new majority of 3 always share a member), so no joint phase is needed — you just append a single config-change entry and commit it normally. etcd, CockroachDB, and TiKV all use single-server changes; you grow a 3-node cluster to 5 by adding one node, waiting for it to commit, then adding the next. Removing the current leader is the one edge case that still needs care — most implementations transfer leadership away first, then remove the old leader as a follower.

## 9. Log compaction and snapshots

> **Senior rule of thumb:** a Raft log that only ever grows is a time bomb — it will eventually exhaust disk and turn every restart into an hours-long replay. Snapshotting is not optional in production; it is the mechanism that bounds both.

The log cannot grow forever. A busy etcd or CockroachDB range commits thousands of entries per second; left unchecked, the log would consume unbounded disk, and a restarting node would have to replay every entry since the beginning of time to rebuild its state machine. The standard remedy is **snapshotting**: periodically, each server writes the *current state of its state machine* to a snapshot, records the index and term of the last entry the snapshot includes (`lastIncludedIndex`, `lastIncludedTerm`), and then discards all log entries up through that index.

![A before-after: an unbounded log of 50000 entries that must be fully replayed on restart, versus a compacted version with a snapshot at index 49000 storing lastIncludedIndex, a short log tail from 49001 to 50000, and InstallSnapshot for lagging followers](/imgs/blogs/raft-consensus-from-scratch-9.webp)

Snapshotting is per-server and largely independent — each node decides when to snapshot based on its own log size (etcd defaults to a snapshot every 10,000 applied entries). The snapshot replaces the committed prefix; the log keeps only the uncommitted/recent tail. The `lastIncludedIndex`/`lastIncludedTerm` metadata is what lets the truncated log still satisfy the `AppendEntries` consistency check — they serve as the `prevLogIndex`/`prevLogTerm` for the first surviving entry.

There is one new RPC: **InstallSnapshot**. Normally a leader brings a lagging follower up to date by sending the missing log entries. But if the leader has *already discarded* the entries a slow follower needs (because it snapshotted past them), there are no entries to send. In that case the leader ships its entire snapshot via `InstallSnapshot` — the follower discards its (now-superseded) log, installs the snapshot as its state, and resumes from `lastIncludedIndex + 1`. This is the failover path for a node that was down long enough to fall off the back of the log. In CockroachDB, where a node rejoining after maintenance might be far behind on thousands of ranges, snapshot transfer is a carefully rate-limited subsystem of its own.

## 10. Client interaction and linearizable reads

> **Senior rule of thumb:** a leader that has been deposed but doesn't know it yet is the single most dangerous component in the system. It will keep answering reads with confidently stale data. Every read protocol exists to neutralize this ghost.

Getting writes committed is only half of a usable system. The other half is the client protocol: how clients find the leader, how writes achieve exactly-once semantics despite retries, and — the genuinely hard part — how reads stay linearizable without a full log round trip on every `GET`.

### Routing and exactly-once writes

Clients send requests to the leader. If a client contacts a follower, the follower rejects the request and (in most implementations) redirects to the leader it currently knows. The trickier problem is *duplicates*: a client sends a write, the leader commits it but crashes before replying, the client times out and retries, and now the command risks being applied *twice*. Raft solves this with client-assigned request IDs. Each client tags every command with a unique serial number; the state machine tracks the latest serial it has applied per client and, on seeing a duplicate, returns the cached response without re-applying. This turns at-least-once delivery into exactly-once *execution*, which is what "I clicked submit twice" safety actually requires.

### Why reads are not free

The naive read — "I'm the leader, here's my value" — is *wrong*, and it's the bug Jepsen found in early etcd. The hazard: a leader can be partitioned away from the majority and a new leader elected on the other side, all without the old leader noticing for up to a full election timeout. During that window the deposed leader still thinks it's in charge and will answer reads with state that is now stale — a linearizability violation. So a correct linearizable read must *confirm the leader is still the leader at read time* and must *see all entries committed before the read began*. There are three standard ways to do this, trading safety margin against latency:

| Read method | How it confirms leadership | Latency cost | Risk | Used by |
| --- | --- | --- | --- | --- |
| **Log read (raft read)** | Commit a no-op read entry through the full log | One full consensus round trip per read | None | rarely used; too slow |
| **ReadIndex** | Record `commitIndex`, exchange a heartbeat round with a majority to confirm leadership, wait until `lastApplied >= readIndex`, then read locally | One heartbeat round trip (no disk write) | None | etcd (`ReadOnlySafe`), CockroachDB |
| **Leader lease** | Leader holds a time-bounded lease; within it, no other leader can exist, so read locally with no round trip | Near-zero | Depends on bounded clock drift | etcd (`ReadOnlyLeaseBased`), CockroachDB leaseholder |

**ReadIndex** is the default safe option. The leader notes the current `commitIndex` as the `readIndex`, sends a heartbeat to a majority to confirm it is *still* leader (if a new leader had been elected, this heartbeat round would reveal a higher term and the old leader would step down), waits until its state machine has applied at least up to `readIndex`, and only then serves the read from local state. No log entry is written, so there's no disk fsync — just one network round trip. etcd's published numbers show ReadIndex latency rising from ~4ms at 3 nodes to ~14ms at 11 nodes, because the heartbeat must reach a larger majority.

The two waits in ReadIndex are both load-bearing, and dropping either breaks linearizability. The *heartbeat* wait proves the leader hasn't been deposed (recency of leadership); the *apply* wait proves the leader's state machine actually reflects everything committed as of the read's start (recency of data). There is also a startup subtlety: a freshly elected leader does not yet know the commit index of entries from *prior* terms until it has committed at least one entry of its *own* term (the same Figure-8 logic). So a correct leader, on election, commits a no-op entry before serving any ReadIndex reads — otherwise a read issued in the first milliseconds of a new term could miss a committed-but-not-yet-confirmed prior-term entry.

```python
def linearizable_read(self: RaftServer, key: str) -> bytes:
    """ReadIndex: serve a linearizable read without writing to the log."""
    if self.role != Role.LEADER:
        raise NotLeader(self.current_leader)        # redirect client

    # 1. Pin the read point at the current commit index.
    read_index = self.commit_index

    # 2. Confirm we are STILL the leader by completing a heartbeat round with a
    #    majority. If a newer leader exists, a reply carries a higher term and we
    #    step down here rather than serving stale data.
    if not self.confirm_leadership_via_heartbeat():
        raise NotLeader(self.current_leader)

    # 3. Wait until our state machine has caught up to the pinned index, so the
    #    read reflects every entry committed before it started.
    self.wait_until(lambda: self.last_applied >= read_index)

    # 4. Now a local read is linearizable.
    return self.state_machine.get(key)
```

**Leader lease** (etcd's `ReadOnlyLeaseBased`, CockroachDB's leaseholder) eliminates even that round trip. The leader, on winning the election and on each successful heartbeat round, grants itself a lease valid for slightly less than the election timeout. Within the lease, Raft's own election rules guarantee no other leader can have been elected, so the leader can serve reads from local state with *no* network round trip — etcd reports lease-based reads at ~0.4ms vs ReadIndex's 4ms at 3 nodes. The catch is the word "time": the lease's safety depends on bounded clock drift between nodes. If a leader's clock runs slow, it might believe its lease is still valid after a new leader has already been elected. Leases trade a clock-drift assumption for a 10x latency win — usually a good trade, but one you must understand before enabling it.

## Where Raft runs in production

> **Senior rule of thumb:** almost nobody implements Raft from scratch for production. They embed a battle-tested library (etcd's `raft`, TiKV's `raft-rs`) and spend their effort on the state machine and the operational envelope around it. Implement Raft from scratch to *learn* it; embed someone else's to *ship* it.

The same algorithm shows up across the infrastructure stack, but each system bends it to its workload. The matrix below summarizes the main axes of variation: how many Raft groups a system runs, how it serves reads, and what its state machine actually is.

![A matrix mapping five production systems (etcd/k8s, CockroachDB, TiKV/TiDB, Kafka KRaft, Consul) across three columns: number of Raft groups, read strategy, and what each replicates](/imgs/blogs/raft-consensus-from-scratch-10.webp)

**etcd** is the canonical embedded Raft, and it is the brain of Kubernetes: every object you create — pods, services, secrets — is a key in etcd's single linearizable key-value store, replicated by one Raft group across (usually) three or five members. etcd's `go.etcd.io/raft` library is deliberately split from the storage: the library is a pure state machine you drive (you hand it messages, it hands you back things to persist and send), which is why it has been reused by CockroachDB, TiKV's early prototypes, and others. etcd defaults to linearizable reads via ReadIndex and was validated by [Jepsen against etcd 3.4.3](https://jepsen.io/analyses/etcd-3.4.3) as strict-serializable for its core key-value operations.

**CockroachDB** runs not one Raft group but *one per range* — the keyspace is split into ~512 MiB ranges, each its own Raft consensus group replicated (typically) three or five ways. A single CockroachDB node participates in thousands of Raft groups simultaneously, which would drown in heartbeat traffic if each group ticked independently. Cockroach's answer is **MultiRaft**: coalesce heartbeats so each pair of nodes exchanges one heartbeat per tick regardless of how many ranges they share, and run a small fixed pool of goroutines instead of one per range. Reads are served by a per-range *leaseholder* (usually but not always the Raft leader) under a leader-lease scheme.

**TiKV / TiDB** uses the same per-shard pattern — TiKV splits data into ~96 MiB *regions*, each a Raft group, in a design explicitly modeled on Google Spanner. TiKV's multi-raft generates so many small log writes that PingCAP built a dedicated log-structured storage engine, *Raft Engine*, just to persist them efficiently. The Rust `raft-rs` library at TiKV's core is a port of etcd's Raft.

**Kafka KRaft** (KIP-500) replaced ZooKeeper with an internal Raft quorum. A small set of *controller* nodes form a Raft group whose replicated log *is* the cluster metadata — topic configs, partition assignments, leader elections. This eliminated the ZooKeeper dependency that became a scaling bottleneck (partition reassignments that took minutes now take seconds), and because the new active controller already holds all committed metadata in memory, failover is near-instant. ZooKeeper support was fully removed in Kafka 4.0. KRaft uses an observer/quorum read model for brokers consuming the metadata log.

**Consul** (HashiCorp) embeds Raft for its service-discovery and KV store, defaulting to a stale-read-allowed mode for low latency with a linearizable option when needed. And the pattern keeps spreading: **RabbitMQ quorum queues** and **Redpanda** both replicate their data via Raft, the former for durable queues, the latter as a ZooKeeper-free Kafka-compatible engine written in C++.

## Operating a Raft cluster

Running Raft in production is less about the algorithm and more about respecting its assumptions. The failures below are the ones that page you.

**Size the cluster odd, and small.** Three nodes tolerate one failure; five tolerate two. Beyond five, every commit waits for a larger majority and write latency climbs while fault tolerance gains shrink — seven nodes tolerate three failures but pay a wider quorum on every single write. Most production etcd and Consul clusters are three or five members. Spreading members across three availability zones (2-2-1 for five nodes) survives a full-AZ outage; spreading across two zones does *not*, because losing the zone with the majority halts the cluster.

**Watch disk fsync latency above all else.** Raft must fsync `currentTerm`, `votedFor`, and new log entries to stable storage *before* acknowledging — so commit latency is bounded below by your disk's fsync latency, not your network. A slow or contended disk (a noisy neighbor on shared cloud storage, a degraded SSD) shows up as elevated commit latency and, in the worst case, as a leader so slow to persist its own heartbeat-adjacent state that followers time out and start needless elections. etcd exposes `etcd_disk_wal_fsync_duration_seconds`; alert when its p99 exceeds ~10ms.

**Tune the election timeout to your network's real round-trip.** The invariant `broadcastTime << electionTimeout << MTBF` must hold. In a single datacenter, the default 150-300ms is fine. Across regions, where round-trips can be 50-100ms, an aggressive timeout causes *flapping* — healthy leaders falsely deposed by a transient latency spike, triggering an election storm that makes latency worse. Raise the election timeout for geo-distributed clusters, accepting slower failover for stability.

**Beware the slow-follower and disk-full traps.** A follower that falls far behind forces the leader into snapshot transfer, which is bandwidth-heavy; if many followers lag at once (after a network blip), the leader can saturate its uplink shipping snapshots. And a node whose disk fills cannot append to its log — it stops acknowledging, and if enough nodes fill up, the cluster loses quorum and halts entirely. etcd's infamous quota-exceeded read-only mode exists precisely to fail safe before the disk fills.

## Case studies from production

### 1. The etcd stale-read bug Jepsen caught

In 2014, Kyle Kingsbury's Jepsen tested etcd 0.4.1 and found it returned *stale reads by default*: a read could be served by any node from its local state without confirming that node was still the leader or that it had the latest committed value. A partitioned-away former leader happily answered with old data — a linearizability violation. The root cause was an optimization that skipped the leadership-confirmation round trip on reads to make them fast. The fix was to add an explicit `quorum=true` flag, and then, in the etcd v3 API, to make linearizable (ReadIndex-confirmed) reads the *default* for everything except watches. The lesson generalizes to every Raft system: a read that doesn't confirm leadership is not linearizable, no matter how recently the node was leader. Speed is not correctness.

### 2. CockroachDB and the thundering herd of heartbeats

When CockroachDB's engineers first ran Raft per-range, a single node hosting tens of thousands of ranges spent a crippling fraction of its CPU and network just sending Raft heartbeats — one per range per tick, across every peer. The naive implementation made heartbeat traffic scale with `ranges × peers`, which is catastrophic at scale. The fix, MultiRaft, *coalesced* heartbeats: each pair of nodes exchanges a single heartbeat message per tick that piggybacks liveness for *all* the ranges they share, decoupling heartbeat cost from range count. The deeper lesson is that the textbook Raft assumption "one cluster, one Raft group" breaks the moment you shard, and the per-message overhead the paper ignores becomes the dominant cost. Sharded-Raft systems live or die on amortizing that overhead.

### 3. The membership-change split brain

A team running a 3-node Raft cluster wanted to migrate to new hardware, so they added three new nodes and removed the three old ones — quickly, with overlapping changes, treating membership as just a config list. For a brief window the cluster had configurations in flight where the old three could form a majority *and* the new three could form a different majority. Two leaders were elected for overlapping terms; writes committed on one side were invisible to the other; when the dust settled, a batch of acknowledged writes had vanished. The root cause was changing membership by more than one server at a time without joint consensus. The fix that every production system now enforces: single-server membership changes, one at a time, each committed before the next begins — so the old and new majorities are guaranteed to overlap and two leaders are impossible.

### 4. The GC pause that deposed a healthy leader

A JVM-based service embedding Raft saw periodic, unexplained leader elections in a perfectly healthy network. The cause was stop-the-world garbage-collection pauses on the leader exceeding the election timeout: during a 400ms GC pause, the leader sent no heartbeats, the followers' 300ms timers fired, and they elected a new leader — even though the old leader was about to resume and was never actually faulty. Worse, the deposed leader, on resuming, briefly believed it was still leader until its next RPC revealed the higher term. The fix combined raising the election timeout above the worst-case GC pause and tuning the collector to cap pause times. The lesson: Raft's failure detector is just a timeout, and *anything* that stalls a node past that timeout — GC, a blocked syscall, CPU starvation, a frozen VM — looks exactly like a crash. Your timeout must exceed your worst-case benign stall.

### 5. The fsync that lied

An operator deployed a Raft cluster on cloud instances with a write-back disk cache that acknowledged fsync *before* data actually hit durable storage — a "lying" fsync, common on misconfigured virtual disks. Everything worked until a power event caused two of three nodes to lose their most recent "persisted" log entries that were still in volatile cache. Because the supposedly-committed entries weren't actually durable on a majority, the cluster recovered to a state that had *un-committed* an acknowledged write. The root cause was a violated assumption: Raft's safety proof requires that fsync actually makes data durable. The fix was disabling the volatile write cache (or using storage with battery-backed cache). The lesson is uncomfortable: Raft is only as safe as your weakest durability guarantee, and the algorithm cannot detect a disk that lies about persistence.

### 6. Kafka's ZooKeeper bottleneck and the KRaft migration

Large Kafka deployments hit a wall as partition counts grew into the hundreds of thousands: the ZooKeeper-based controller had to load the full metadata state from ZooKeeper on every controller failover, and metadata operations like partition reassignment serialized through ZooKeeper writes that took *minutes* at scale. The architectural fix was KRaft (KIP-500): replace ZooKeeper with an internal Raft quorum whose replicated log *is* the metadata, so the active controller already holds all committed metadata in memory and failover is near-instant. The migration was multi-year precisely because metadata consistency is load-bearing — a bug would corrupt every topic in the cluster. By Kafka 4.0, ZooKeeper was gone entirely. The lesson: even a correct external coordination service (ZooKeeper itself runs a consensus protocol, ZAB) becomes a bottleneck when it sits on the metadata hot path; folding consensus *into* the system removes a round trip and a failure domain.

### 7. The 4-node cluster that tolerated nothing extra

A team grew their 3-node etcd cluster to 4 nodes "for more redundancy," reasoning that more nodes meant more fault tolerance. It didn't. A 4-node cluster needs 3 for a majority — so it tolerates exactly *one* failure, the same as a 3-node cluster, but with a larger quorum to satisfy on every write (higher latency) and one more machine that can fail. When two of the four nodes went down in an incident, the cluster lost quorum and halted, where a 5-node cluster would have survived. The root cause was the intuition that "more nodes = more safety," which is false at even counts. The fix and the rule: Raft clusters should always be odd-sized, because an even cluster pays for an extra node without buying additional fault tolerance over the odd size below it.

### 8. The snapshot that never ran

A long-running embedded-Raft service was configured without snapshotting (or with a snapshot threshold so high it never triggered). Over months the log grew to tens of gigabytes. When a node restarted for a routine deploy, it had to replay the *entire* log to rebuild its state machine — a process that took over an hour, during which the cluster ran degraded on the remaining nodes. Worse, the un-truncated log eventually threatened to fill the disk. The root cause was treating snapshotting as optional. The fix was enabling periodic snapshots (every N applied entries) so the log stays bounded and restart replays only the recent tail. The lesson: in any Raft system that runs longer than a demo, snapshotting is mandatory infrastructure, not a nice-to-have — it bounds both disk usage and recovery time.

### 9. The non-deterministic state machine that diverged

A team built a replicated state machine whose `apply` function, for one command type, iterated over a Go map to build a response — and Go randomizes map iteration order. The result was that the *same* committed log entry produced slightly different responses (and, in one case, different internal state) on different replicas. For a while nothing visibly broke, because the divergence was in a field nobody read. Then a snapshot taken on one replica was shipped via `InstallSnapshot` to another, and the receiving node's subsequent log replay produced a checksum mismatch that crash-looped the node. The root cause was a violated *determinism* assumption: the replicated-state-machine model only converges if `apply` is a pure function of `(state, command)`. Map iteration order, wall-clock reads, `rand()` without a seeded-and-logged seed, floating-point summation order, and "current time" are all silent non-determinism sources. The fix was making `apply` deterministic (sort before iterating, thread time in as part of the command). The lesson: Raft guarantees identical *logs*, never identical *state* — determinism of the state machine is your job, and the algorithm cannot detect when you've broken it until snapshots collide.

### 10. The lease read that served stale data under clock skew

A latency-sensitive service enabled leader-lease reads (etcd's `ReadOnlyLeaseBased`) to shave the ReadIndex round trip off every `GET`. It worked beautifully — until a VM live-migration froze the leader's clock relative to its peers. The leader believed its lease was still valid (its slow clock said only 100ms had passed) while, in real time, the lease had expired, a new leader had been elected, and committed a write. The old leader, still inside its *perceived* lease, served a read from local state that omitted the new write — a linearizability violation invisible to monitoring. The root cause was that lease safety depends on a bounded clock-drift assumption that a frozen or skewed clock violates. The fix was reverting that path to ReadIndex (which confirms leadership with a real heartbeat round and makes no clock assumption) for the operations that genuinely needed linearizability, keeping leases only where bounded staleness was acceptable. The lesson: leader leases trade a clock assumption for latency, and that trade is only sound if you actually bound clock drift (NTP discipline, max-clock-offset enforcement like CockroachDB's `--max-offset`) — otherwise the cheapest read mode is also the one that silently lies.

## When to reach for Raft, and when not to

Reach for Raft when:

- You need a **strongly consistent, fault-tolerant store** for critical metadata — leader election, configuration, locks, service discovery, small amounts of data where correctness dwarfs throughput. This is etcd's and Consul's sweet spot.
- You need **linearizable reads and writes** that survive the loss of a minority of nodes with no data loss and automatic failover.
- You are building a **replicated state machine** whose commands are deterministic and whose state fits the snapshot-and-replay model.
- You want a consensus protocol your team can actually *understand, debug, and operate* — Raft's explicit design goal, validated by user studies showing it is meaningfully easier to learn than Paxos.

Skip Raft (or reach for something else) when:

- You need **high write throughput across many independent keys** with no cross-key ordering requirement. A single Raft group serializes everything through one leader; you'll bottleneck. Either shard into many Raft groups (the CockroachDB/TiKV path, with all the MultiRaft complexity that implies) or use a leaderless/eventually-consistent store.
- Your workload is **read-heavy and can tolerate staleness**. Paying for linearizable reads on data that doesn't need it is pure overhead; a [single-leader replication](/blog/software-development/database/database-replication-sync-async-logical-physical) setup with async read replicas, or an [eventually-consistent](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) store, is cheaper and more available.
- You need to **survive Byzantine (malicious or arbitrarily buggy) nodes**. Raft assumes crash-stop faults — nodes fail by stopping, not by lying. A node that sends corrupted or adversarial messages can violate Raft's safety. Byzantine fault tolerance needs a different protocol (PBFT, Tendermint) at substantially higher cost.
- You are **geo-distributing for low-latency local writes**. Raft commits at the speed of a majority round trip, so a globe-spanning Raft group pays cross-ocean latency on every write — the consistency horn of the [CAP and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) tradeoffs. If you need fast local writes everywhere, you want a different consistency model, not a wider Raft group.

## Further reading

- Diego Ongaro and John Ousterhout, ["In Search of an Understandable Consensus Algorithm (Extended Version)"](https://raft.github.io/raft.pdf) — the Raft paper itself; the extended version has the full safety proofs and the Figure 8 walkthrough.
- [raft.github.io](https://raft.github.io/) — the canonical hub, with links to dozens of implementations and the original [USENIX ATC '14 presentation](https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro).
- [The Secret Lives of Data: Raft](https://thesecretlivesofdata.com/raft/) and [raftscope](https://github.com/ongardie/raftscope) — interactive visualizations; watch elections and log repair happen step by step.
- [Jepsen: etcd 3.4.3](https://jepsen.io/analyses/etcd-3.4.3) and the [etcd 3.4.3 results writeup](https://etcd.io/blog/2020/jepsen-343-results/) — what linearizability looks like when adversarially tested.
- [CockroachDB: Scaling Raft](https://www.cockroachlabs.com/blog/scaling-raft/) and [Joint consensus in CockroachDB](https://www.cockroachlabs.com/blog/joint-consensus-raft/) — MultiRaft and membership-change correctness at scale.
- [Building a Large-scale Distributed Storage System Based on Raft](https://tikv.org/blog/building-distributed-storage-system-on-raft/) (TiKV) and [Why ZooKeeper Was Replaced with KRaft](https://www.confluent.io/blog/why-replace-zookeeper-with-kafka-raft-the-log-of-all-logs/) (Confluent/Kafka).
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 9 — consensus, total order broadcast, and their equivalence to linearizable storage; the conceptual frame this article builds on. Paxos and Multi-Paxos, Raft's intellectual predecessor, are covered in [a sibling post](/blog/software-development/database/paxos-and-multi-paxos-explained).

---
title: "Two-Phase Commit and Every Way It Can Fail"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch tour of two-phase commit — the prepare/vote and commit/abort phases, why it gives atomicity across nodes, and the long catalogue of ways it fails: coordinator crashes that strand participants in-doubt and blocked, the uncertainty window, why 3PC doesn't fix blocking, why 2PC needs all votes while consensus needs only a majority, the pain of XA, and where 2PC is still legitimately used."
tags:
  [
    "two-phase-commit",
    "distributed-transactions",
    "atomic-commit",
    "xa",
    "consensus",
    "coordinator-failure",
    "distributed-systems",
    "postgres",
    "databases",
    "2pc",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/two-phase-commit-and-how-it-fails-1.webp"
---

Somewhere in your stack there is a piece of code that wants to do two writes that must both happen or neither happen — debit one account, credit another; reserve inventory, charge a card; insert a row, publish an event. On a single database this is a solved problem you stopped thinking about years ago: `BEGIN`, two statements, `COMMIT`, and the storage engine's write-ahead log makes it atomic. The instant those two writes live on *different machines* — two shards, two databases, a database and a message broker — that comfortable guarantee evaporates. You can send a commit to both, but one might succeed while the other fails, and now you have a half-committed transaction that no single node can repair. The protocol that the industry reached for to close this gap is two-phase commit (2PC), and it has a property that is both its whole reason for existing and its fatal flaw: it can leave your databases frozen, holding locks, waiting forever on a coordinator that is never coming back.

This article builds 2PC from the ground up and then spends most of its length on the part that matters in production — the failure modes. We start from the atomic-commit-across-nodes problem (all commit or all abort, no exceptions), derive the protocol as a sequence of forced moves, and show *why* it delivers atomicity. Then we turn it over and catalogue every way it breaks: a participant crash, a coordinator crash before the votes, a coordinator crash *after* the votes (the one that kills you), network partitions, lost messages, and the uncertainty window where a node knows it voted yes but cannot tell whether the global decision was commit or abort. We look at why three-phase commit (3PC) was invented to fix the blocking and why it doesn't — not really, not on a network like yours. We draw the single most important contrast in this whole space: **2PC needs every participant to vote, so any one failure can stall it; consensus needs only a majority, so it survives a minority of failures.** That one sentence is why your config store runs Raft and your distributed transaction runs a fragile XA dance. We walk through XA and why distributed transaction managers are operationally miserable, why microservices ran away from 2PC toward sagas, and finally where 2PC is genuinely the right tool: committing across shards *inside one system* — Postgres prepared transactions, MySQL XA, and the way Percolator and Spanner make the coordinator itself fault-tolerant by backing it with Paxos.

The diagram below is the mental model the rest of the article unpacks: 2PC is two round trips and one durable decision. Phase 1 (prepare) turns each independent node into a binding, crash-proof *promise* that it can commit; Phase 2 (commit) just flips a switch whose outcome was already decided. The genius is in Phase 1; the danger is in the gap between the two phases.

![A 2PC happy-path timeline: coordinator assigns a global txid, sends PREPARE, each participant fsyncs its log and votes YES, the coordinator fsyncs the COMMIT decision, then sends COMMIT and participants release locks and acknowledge](/imgs/blogs/two-phase-commit-and-how-it-fails-1.webp)

> Atomic commit across machines is not "send commit to everyone." It is the much harder problem of getting a set of nodes to make the *same* irrevocable decision despite the fact that any of them, or the network between them, can fail at the single worst moment.

## Why this is different from what most engineers assume

Most engineers carry a single-node mental model of transactions: a transaction is something the database makes atomic for you, end of story. That model is not wrong — it is just silent about the one assumption it leans on entirely, which is that all the data lives in one place with one write-ahead log and one recovery routine. The moment the data is spread across nodes, every comfortable assumption in the left column below collides with the reality in the right column, and 2PC exists precisely to bridge that gap — at a cost most people underestimate until the first 3 a.m. page.

| Assumption | The single-node mental model | The distributed reality 2PC must survive |
| --- | --- | --- |
| "Commit is one atomic action." | The engine writes a commit record; done. | Commit must be coordinated across N independent logs; some can succeed while others crash mid-write. |
| "A transaction either commits or aborts." | Two clean outcomes, decided locally. | A third state exists: **in-doubt** — a node has voted yes but doesn't yet know the global verdict and cannot decide alone. |
| "Once I vote yes I'm basically done." | Voting is the last hard step. | Voting yes is a *binding promise* that locks rows and survives crashes; you've crossed the point of no return and given up the right to abort. |
| "If the coordinator dies, just abort." | A crash means roll back and retry. | A participant that voted yes **cannot** safely abort — the coordinator may have already told someone else to commit. It must block. |
| "Adding more nodes adds safety." | Replication is more copies = more durable. | 2PC requires **all** participants to vote yes, so each added node strictly *lowers* availability — the opposite of consensus. |
| "Timeouts let us make progress." | If it's slow, give up and move on. | A timeout cannot distinguish a crashed coordinator from a slow one; "give up and decide" is exactly how you violate atomicity. |

Every row in the right column is a real failure mode that the rest of this article dissects. The discipline here is the same one I bring to any distributed protocol: never introduce a rule without naming the failure it prevents, and never trust a happy-path diagram until you have run the crash at each step.

## 1. The atomic commit problem, stated precisely

> **Senior rule of thumb:** the hard part of distributed transactions is not making writes happen — it is making sure that once *one* node has revealed a committed value to someone, *every* node is bound to commit too, even across crashes. A commit, once visible, is irrevocable.

Start with what a single node already does, because 2PC is a generalization of it. When Postgres commits a transaction, it writes the transaction's changes to the write-ahead log (WAL), then appends a *commit record* and flushes it to durable storage. The commit record is the atomic switch: if a crash happens before it is durably written, recovery rolls the transaction back; if it happens after, recovery rolls it forward. There is exactly one bit that decides the outcome, it is written atomically by a single `fsync`, and a single recovery routine reads it. Atomicity on one node reduces to "atomically flip one durable bit."

Now spread the transaction across three nodes. Martin Kleppmann frames the core difficulty sharply in *Designing Data-Intensive Applications* (Chapter 9): you cannot simply send a commit request to all of them, because "it's possible that the commit succeeds on some nodes and fails on other nodes, which is a violation of the atomicity guarantee." A node might run out of disk, hit a constraint violation, crash, or have its commit message lost in the network — and if even one node fails to commit after others have already committed and made their changes visible, you have torn the transaction in half. The committed effects are, by definition, *irrevocable*: other transactions may have already read them and acted on them. You cannot un-commit.

So the atomic commit problem is: get N nodes to agree on a single binary outcome — **commit** or **abort** — such that:

- **Agreement.** All nodes that decide, decide the same outcome. No node commits while another aborts.
- **Validity.** If any node cannot commit (constraint violation, crash, out of space), the outcome must be abort. You cannot commit a transaction a participant rejected.
- **Stability/irrevocability.** Once a node has committed (made its writes visible), the outcome is fixed forever; no later event can flip it to abort.
- **Termination (liveness).** Nodes should eventually decide rather than hang forever — though, as we will see, this is the property 2PC is forced to sacrifice under failure.

The first three are *safety* (nothing bad happens); the fourth is *liveness* (something good eventually happens). Hold onto this split, because the entire story of 2PC and its successors is the story of trading liveness for safety. 2PC keeps safety unconditional — it will *never* let one node commit while another aborts — and pays for it by being willing to block forever when the coordinator dies. That is not a bug in a particular implementation; it is the deliberate trade the protocol makes.

A subtle but load-bearing point, also from Kleppmann's framing: this same problem is what consensus solves. Atomic commit, total order broadcast, and linearizable compare-and-set are all reducible to one another. The deep reason 2PC and Raft look like cousins is that they are solving facets of the same underlying agreement problem. The difference — which we will build to — is in *how many* nodes have to participate in the agreement, and that single difference is why one of them is fault-tolerant and the other is not. For the consensus side of that story in full, see [Raft: implementing distributed consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch).

### Why "just send commit to everyone" fails, concretely

Take the most naive algorithm: the application sends `COMMIT` to all three databases in a loop and hopes. Run the failure. The first two databases receive it, commit, and make `x=8` and `y=3` visible; a downstream read picks up `x=8` and a user sees a confirmation screen. The third database, mid-commit, hits a unique-constraint violation and rejects the write. Now node 3 has *not* applied its part of the transaction, but nodes 1 and 2 have, and a user has already seen the result. There is no principled way to fix this: you cannot abort nodes 1 and 2 (someone already read their committed data), and you cannot force node 3 to commit (it has a genuine constraint conflict). Atomicity is violated and the damage has escaped the system.

The lesson is the seed of 2PC: **no node may commit until every node has promised it can.** You must collect a unanimous, binding "I can commit" from all participants *before* anyone actually commits. That collection step is Phase 1. Once you have it, the actual commit in Phase 2 is guaranteed to succeed on every node, because each has already done the hard, fallible work and durably promised the result.

## 2. The two-phase commit protocol, derived

> **Senior rule of thumb:** "prepared" is not "almost committed" — it is a separate, expensive, durable state in which a participant has surrendered its right to abort and is holding locks open until someone tells it the verdict. Treat every `PREPARE` as a small loan against your availability.

The protocol has two roles. The **coordinator** (sometimes called the transaction manager) drives the decision; the **participants** (or resource managers) each own a slice of the data. There is one logical transaction with a single global transaction id that the coordinator assigns. The protocol runs in two phases, exactly as the opening figure shows.

**Phase 1 — prepare / vote.** The coordinator sends a `PREPARE` (vote-request) message to every participant. On receiving it, each participant does *all* the fallible work required to commit — checks constraints, acquires the necessary locks, writes its changes and an undo+redo record to its own log, and `fsync`s that log to disk — and only then replies `YES` (vote-commit). Crucially, when a participant votes `YES`, it is making an unconditional promise: it is guaranteeing that it *can and will* commit this transaction if later told to, no matter what. Kleppmann puts the bar exactly right — once a participant votes yes, it must be able to commit "under all circumstances. A power failure, crash, or memory issue cannot be an excuse for refusing to commit later." If a participant cannot make that promise (constraint violation, disk full, deadlock), it replies `NO` (vote-abort) instead. A participant that has voted `YES` is now **prepared**: its state is durable, its locks are held, and it has crossed the point of no return.

**Phase 2 — commit / abort.** The coordinator collects the votes. If *all* participants voted `YES`, the coordinator makes the decision **commit**; if *any* voted `NO` (or timed out before replying), it decides **abort**. The coordinator writes this decision to its own transaction log and `fsync`s it — this is the moment the global outcome becomes real and irrevocable. Only then does it send `COMMIT` (or `ABORT`) to every participant. Each participant, on receiving the decision, completes the local transaction (commit makes the prepared changes visible and releases locks; abort rolls back the prepared changes and releases locks) and replies with an acknowledgment. When all acks are in, the coordinator can forget the transaction.

There are two **points of no return** in this protocol, and naming them precisely is the key to understanding every failure mode:

1. **A participant's point of no return is voting `YES`.** After that vote, it may not unilaterally abort, because the coordinator might already be telling another participant to commit. It is *committed to whatever the coordinator decides.*
2. **The coordinator's point of no return is writing its decision record.** Before that record is durable, the coordinator is free to abort. After it, the decision is fixed; the coordinator must drive it to completion. As Kleppmann's framing makes vivid: once the decision is logged, "it must be retried forever until it succeeds." The coordinator will resend `COMMIT` after every restart, indefinitely, until every participant acknowledges.

This is why the protocol gives atomicity. By the time *any* participant commits (makes data visible), two facts are already guaranteed: every participant has voted yes (so every participant *can* commit), and the coordinator has durably decided commit (so the decision can't be lost to a crash). The unanimous prepare in Phase 1 removes the possibility of a participant failing in Phase 2, and the durable decision record removes the possibility of the verdict being forgotten. Agreement and validity both fall out.

### Pseudocode: the coordinator and participant state machines

Here is the protocol as runnable-shaped pseudocode. Note where every `fsync` lands — the durability of those writes is the *entire* correctness argument — and note that there are no timeouts that let a prepared participant decide on its own. That omission is deliberate and is exactly the blocking flaw.

```python
# ---- COORDINATOR ----
def coordinator_commit(txn_id, participants):
    # Phase 1: prepare. Ask everyone to durably promise.
    votes = {}
    for p in participants:
        try:
            votes[p] = p.rpc("PREPARE", txn_id, timeout=PREPARE_TIMEOUT)
        except (Timeout, ConnectionError):
            votes[p] = "NO"          # no reply == abort; safe, no one committed yet

    # Decide. Unanimous YES -> commit; anything else -> abort.
    decision = "COMMIT" if all(v == "YES" for v in votes.values()) else "ABORT"

    # Coordinator's POINT OF NO RETURN: the decision becomes irrevocable here.
    # If we crash before this fsync, recovery aborts. After it, recovery commits.
    log.append({"txn": txn_id, "decision": decision})
    log.fsync()

    # Phase 2: broadcast the decision. Retry FOREVER until every ack arrives;
    # a prepared participant is holding locks until it hears this.
    pending = set(participants)
    while pending:
        for p in list(pending):
            try:
                p.rpc(decision, txn_id, timeout=COMMIT_TIMEOUT)
                pending.discard(p)
            except (Timeout, ConnectionError):
                pass                  # keep retrying; do NOT give up
        if pending:
            sleep(backoff())
    log.append({"txn": txn_id, "state": "DONE"})   # safe to forget
    return decision


# ---- PARTICIPANT ----
def on_prepare(txn_id, work):
    try:
        acquire_locks(work.rows)          # hold X-locks until phase 2
        apply_to_undo_redo_log(txn_id, work)
        log.append({"txn": txn_id, "state": "PREPARED"})
        log.fsync()                       # participant's POINT OF NO RETURN
        return "YES"                      # binding promise: I *will* commit on request
    except (ConstraintViolation, DiskFull, Deadlock):
        rollback(txn_id)                  # still free to abort; haven't promised yet
        return "NO"

def on_commit(txn_id):
    make_visible(txn_id)                  # apply the prepared changes
    log.append({"txn": txn_id, "state": "COMMITTED"}); log.fsync()
    release_locks(txn_id)
    return "ACK"

def on_abort(txn_id):
    rollback(txn_id); release_locks(txn_id)
    return "ACK"

# THE BLOCKING FLAW, made explicit: there is no on_timeout() for a PREPARED
# participant. Once prepared, if the coordinator goes silent, the participant
# has no safe local action. It cannot commit (maybe the decision was ABORT) and
# it cannot abort (maybe the decision was COMMIT and another node already did).
# It can only WAIT — holding its locks — for the coordinator to come back.
```

The comment at the bottom is the whole article in eight lines. Everything from here on is an exploration of what happens when the coordinator does not come back quickly, or when the network lies about who is alive.

## 3. What "prepared" actually costs

> **Senior rule of thumb:** a prepared transaction is the most expensive idle object in your database. It pins locks, pins an old MVCC snapshot, blocks vacuum, and holds an `fsync`'d log record open — all while doing nothing, waiting for a message that might never arrive.

It is tempting to think of the gap between prepare and commit as instantaneous — two RPCs, microseconds apart. On the happy path it is. But the protocol's correctness does not depend on that gap being small, and its *cost* is paid the entire time the gap is open. The figure below makes the cost concrete: three prepared participants, each holding exclusive row locks and a flushed prepare record, with other transactions queueing behind those locks.

![Hand-drawn figure: a coordinator awaiting all votes points to three prepared participants, each showing its held X-locks (rows 17/42, row 88, rows 5/9/11) and an fsync'd PREPARE record; a dashed bar-arrow shows other transactions blocked behind these locks](/imgs/blogs/two-phase-commit-and-how-it-fails-3.webp)

Walk through what a participant is holding while prepared:

- **Exclusive locks on every row it touched.** Any other transaction that wants those rows blocks until Phase 2 releases them. If the prepared transaction touched a hot row, you have just serialized a large slice of your workload behind it. (For why these locks are necessary and how they cascade, see [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production).)
- **A durable, `fsync`'d prepare record.** The participant has paid a disk flush to make the promise survivable. That record cannot be discarded until the verdict arrives.
- **An old MVCC snapshot / transaction id.** The transaction is still "in progress" from the engine's perspective, which means its transaction id stays in the active set. In Postgres specifically, this is the quiet killer: a prepared transaction holds back `VACUUM`, which means dead tuples accumulate and — in the extreme — the cluster approaches transaction-id wraparound. The official Postgres documentation does not mince words: "It is unwise to leave transactions in the prepared state for a long time. This will interfere with the ability of VACUUM to reclaim storage, and in extreme cases could cause the database to shut down to prevent transaction ID wraparound. Keep in mind also that the transaction continues to hold whatever locks it held."

That last clause — *the transaction continues to hold whatever locks it held* — is the bridge to the failure modes. A prepared participant is not free. Every second it stays prepared, it is degrading the availability of everything that touches its locked rows. On the happy path that's a few microseconds. When the coordinator crashes, it's *forever* — and "forever holding locks" is how a single dead coordinator takes out an entire database.

### Atomicity is real; isolation is your problem

One thing 2PC does *not* solve, and that bites teams who assume it does: it coordinates atomic *commit*, not isolation across the distributed transaction. Each participant enforces isolation locally with its own locks and MVCC, but there is no global snapshot unless you build one. Two distributed transactions can interleave at different participants in ways that produce anomalies the single-node isolation level would have prevented. The held locks in the figure above are the local isolation mechanism doing its job; the *global* serializability story requires something more (a global timestamp authority, as in Spanner, or snapshot isolation built on a timestamp oracle, as in Percolator). If you reach for 2PC expecting it to also hand you global serializable isolation for free, you will be surprised. For the single-node anomaly catalogue that 2PC inherits but does not extend, see [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent).

## 4. The failure modes, one crash at a time

> **Senior rule of thumb:** to evaluate any commit protocol, do not read its happy path — inject a crash immediately before and immediately after every message, and ask "can a node now violate atomicity, or is it stuck?" 2PC never violates atomicity. It gets stuck.

This is the heart of the article. We will walk through each failure systematically. The taxonomy figure below summarizes where we are headed: the danger of any failure depends almost entirely on *when* it happens relative to the votes and the decision. There is exactly one cell that is genuinely unrecoverable without the coordinator coming back — and it is the one that gives 2PC its reputation.

![A 4x3 matrix of failure modes (participant crash, coordinator crash, network partition, lost decision message) against timing (before vote, after vote pre-decision, after decision); only the coordinator-crash and partition cells in the after-vote-pre-decision column are red and labeled BLOCKED](/imgs/blogs/two-phase-commit-and-how-it-fails-8.webp)

### 4.1 Participant crashes before voting

The easiest case. A participant that crashes before sending its vote simply doesn't reply. The coordinator's prepare RPC times out, the coordinator treats the missing vote as `NO`, and the global decision is **abort**. Every other participant rolls back and releases its locks. The crashed participant, on restart, finds the transaction was never prepared (no durable prepare record), so it has nothing to do. No atomicity violation, no blocking — the system safely aborts. This is the green column-and-row in the matrix: a crash before the promise is harmless because no promises were broken.

### 4.2 Participant crashes after voting yes

Now it gets interesting. A participant votes `YES`, durably writes its prepare record, and then crashes. The coordinator, having (let's say) collected all yes votes, decides commit, logs it, and sends `COMMIT` to everyone — including the crashed participant, where the message is lost. On restart, the participant runs recovery, finds a durable `PREPARED` record for this transaction, and realizes it is **in-doubt**: it knows it voted yes, but it does not know the global verdict. It cannot decide on its own (committing would be wrong if the decision was abort; aborting would be catastrophic if the decision was commit and another node already committed). So it does the only safe thing: it asks the coordinator. The coordinator, which retries the decision forever, re-sends `COMMIT`, the participant completes, releases locks, and acks. Recoverable — *as long as the coordinator is alive*. The participant's own crash is survivable precisely because it made its promise durable before crashing. The durability of the prepare record is what lets a participant resume an in-doubt transaction after a crash instead of guessing.

### 4.3 The coordinator crashes — the fatal flaw

Here is the case that defines 2PC. The participants have all voted `YES`. They are prepared, holding locks, waiting for the verdict. And the coordinator crashes *before writing its decision record* and sending Phase 2. Now consider what each participant can do.

It cannot commit. Maybe the coordinator was about to decide abort (perhaps a fourth participant it hadn't heard from was going to vote no). It cannot abort. Maybe the coordinator had already decided commit and crashed mid-broadcast after telling *another* participant to commit — if this node aborts now, atomicity is destroyed. From a prepared participant's local point of view, these two worlds are indistinguishable. The coordinator holds the only copy of the decision (or the authority to make it), and the coordinator is gone. As Kleppmann puts it, the participants are "in doubt" or "uncertain," and "the problem with a transaction that is stuck waiting for the coordinator is that it cannot release the locks it's holding, which can cause larger parts of the application to become unavailable." The Paper Trail's classic write-up of 2PC makes the same point even more bluntly: "if the coordinator doesn't recover for a long time, the nodes that received the proposal are going to be blocked waiting for the outcome of a protocol that might never be finished... the protocol is blocked on the coordinator, and can't make any progress."

This is **blocking**. The participants are stranded in-doubt, holding their locks, and they will keep holding them until the coordinator comes back. There is no timeout they can take that is safe. The figure below traces the timeline: prepare, all vote yes (point of no return), coordinator crashes before the decision record, and then the cascade — in-doubt participants, locks held indefinitely, vacuum stalled, and finally other transactions queueing behind the locked rows and timing out cluster-wide. A single dead coordinator has converted a microsecond gap into an open-ended outage.

![A timeline of the fatal flaw: PREPARE sent and rows locked, all vote YES (point of no return), coordinator CRASHES before the decision record, participants go in-doubt and cannot commit or abort, row locks held indefinitely and VACUUM stalls, and other transactions queue and time out cluster-wide](/imgs/blogs/two-phase-commit-and-how-it-fails-2.webp)

The standard mitigation is that the coordinator's decision record is durable, so when the coordinator *does* restart, it reads its log, finds the in-doubt transaction, re-derives or recovers the decision, and resends it to unblock everyone. That is why a well-built coordinator `fsync`s its decision *before* sending Phase 2 — so a crash anywhere in Phase 2 is recoverable on restart. But this only helps if the coordinator restarts in a bounded time. If the coordinator's host is gone (disk failure, fire, a multi-hour cloud-zone outage), the in-doubt transactions stay blocked for the entire duration. The blocking window equals the coordinator's recovery time, and the coordinator is a single point of failure that you do not control on a human timescale.

There is a subtler horror inside this case. Suppose another node tries to help recovery by polling its peers for the decision (a "termination protocol"). If a peer that knew the answer has *also* crashed, the recovery node is stuck with an impossible inference: it "can't distinguish between all nodes having already voted to commit... and all nodes but the failed node having voted to commit and the failed node having voted to abort," as the Paper Trail analysis notes. The information needed to decide safely is sitting on a machine that is currently down. So even peer-to-peer recovery cannot reliably break the block — it just moves the dependency from one dead node to another.

### 4.4 Network partition

A partition is, from each side's local perspective, indistinguishable from crashes on the other side. If the partition cuts the coordinator off from the participants after they have prepared, the participants see a silent coordinator and block exactly as in 4.3 — in-doubt, locks held, until the partition heals. If the partition cuts off some participants before they vote, the coordinator times them out, decides abort, and the reachable participants release. The damage is again concentrated in the after-vote, pre-decision window: a partition there freezes the prepared participants for the duration of the partition. This is why the matrix marks both the coordinator-crash and the partition cells in that column as **BLOCKED**. The protocol's safety holds — no split-brain commit/abort — but its liveness is hostage to the network.

### 4.5 Lost decision message

If the coordinator's `COMMIT`/`ABORT` message is dropped, the participant waits, and the coordinator's forever-retry loop eventually resends it. Because applying a decision is idempotent (committing an already-committed prepared transaction is a no-op; the participant tracks state), the resend is safe. This is the most benign post-decision failure: the decision exists durably, so it is only a matter of redelivery. The cost is latency and held locks until the resend lands, not a correctness problem.

### 4.6 The uncertainty window, named

Pulling these together: there is a specific interval in 2PC during which a participant has voted yes but has not yet received the verdict. Call it the **uncertainty window**. Inside it, the participant has surrendered its autonomy (it cannot abort) but lacks the information to act (it doesn't know whether to commit). Every catastrophic failure mode of 2PC lives in this window. Before it, participants can safely abort on timeout; after it, the decision is durable and recoverable. The window's *width* is normally tiny — one network round trip plus one coordinator `fsync`. The whole tragedy of 2PC is that a coordinator crash *inside* this window stretches it to the coordinator's recovery time, which can be unbounded. Minimizing the window (fast coordinator, fast disks, co-located nodes) reduces the *probability* of a crash landing inside it, but cannot eliminate the window, because the window is intrinsic to the protocol. You cannot prepare and decide in the same instant across a network.

## 5. Three-phase commit, and why it doesn't save you

> **Senior rule of thumb:** any protocol that claims to be non-blocking by adding a timeout is implicitly assuming your network has a bounded delay and your failure detector never lies. Both assumptions are false on real infrastructure, and the protocol's "non-blocking" property dies with them.

The obvious response to the blocking flaw is: give the participants enough information that they can make a safe decision on their own if the coordinator vanishes. That is the idea behind three-phase commit (3PC), introduced by Dale Skeen in the early 1980s. 3PC splits Phase 2 into two, inserting a **pre-commit** round between the vote and the actual commit, as the before-after below shows.

![A before-after comparison: 2PC (prepare/vote, then commit/abort, in-doubt blocks on crash, works on async networks) versus 3PC (can-commit vote, then a pre-commit broadcast, then do-commit, but splits diverge on partition)](/imgs/blogs/two-phase-commit-and-how-it-fails-5.webp)

The three phases are: **can-commit** (the coordinator asks "can you commit?" and participants vote, but *without* yet acquiring all resources irrevocably), **pre-commit** (if all said yes, the coordinator broadcasts "prepare to commit" — now everyone knows the decision *will* be commit), and **do-commit** (the actual commit). The clever bit is the middle phase. By the time a participant is in the pre-commit state, it knows that *every* participant voted yes, so the only possible outcome is commit. Therefore, if the coordinator now vanishes, a participant in pre-commit can safely time out and **commit on its own**, and a participant still in the can-commit state (not yet told to pre-commit) can safely time out and **abort**. The extra round seems to buy exactly the non-blocking property 2PC lacks: a participant always has enough local information to decide.

It seems to. Here is why it doesn't, on your network. 3PC's safety rests on two assumptions that the textbook states quietly and that production never satisfies. First, it assumes a **synchronous network with bounded message delay** and nodes with bounded response times — so that a timeout genuinely means "the other side is dead," not "the other side is slow." Second, it assumes a **fail-stop model with a perfect failure detector** — crashes happen by clean stopping and are accurately detected. Kleppmann's framing is exact: 3PC "assumes a network with bounded delay and nodes with bounded response times," and because "most practical systems have unbounded network delays and process pauses," it "cannot guarantee atomicity." The difficulty reduces to building a perfect failure detector that never falsely declares a coordinator dead — which is impossible on an asynchronous network — "which is why 2PC continues to be used today."

Concretely, watch 3PC break under a partition. The network splits the participants into two groups while a transaction is mid-flight. Some participants on side A have reached pre-commit and, seeing the coordinator gone, time out and **commit**. Some participants on side B are still in can-commit and, also timing out, **abort**. Now you have a genuine split: half the transaction committed, half aborted. 3PC traded the blocking problem for a *worse* problem — under partition it can violate atomicity outright, deciding different outcomes on different sides, which is exactly the split-brain that 2PC's blocking was protecting you against. As the analysis puts it, "if the network splits, different partitions may reach different conclusions." 3PC also adds a full extra message round (more latency, more bandwidth — significant at scale) for a guarantee that holds only in a world that doesn't exist. This is why 3PC is essentially never deployed. The field did not improve on 2PC by adding a phase; it improved on 2PC by adding *replication*, which is the next section.

| Property | 2PC | 3PC |
| --- | --- | --- |
| Round trips on commit | 2 | 3 |
| Blocks on coordinator crash (after votes) | Yes — participants in-doubt | No — under its assumptions |
| Network model it requires | Asynchronous (works on real networks) | Synchronous, bounded delay |
| Failure detector it requires | None (just retries) | Perfect — never falsely "dead" |
| Behavior under partition | Safe but blocks | **Unsafe** — sides can diverge |
| Used in production | Yes (XA, prepared txns) | Essentially never |
| What actually replaced it | — | Consensus (Paxos/Raft) |

## 6. The key insight: 2PC needs all, consensus needs a majority

> **Senior rule of thumb:** if a protocol's progress requires hearing from *every* node, adding nodes makes it *less* available. If progress requires hearing from a *majority*, adding nodes makes it *more* available. That single difference is the line between a coordination protocol and a fault-tolerant one.

This is the most important idea in the whole space, and it is simpler than the machinery around it. Two-phase commit requires a *unanimous* vote: the coordinator can only decide commit if **all** participants vote yes, and it can only finish Phase 2 once **all** participants acknowledge. Any single participant that crashes or partitions away stalls the protocol — at best it forces an abort, at worst (in the uncertainty window) it leaves everyone blocked. The coordinator itself is not replicated, not elected, and has no failover. The result, as one analysis states plainly, is that "2PC is not fault-tolerant because the coordinator is not elected and cannot handle failures of participants, unlike fault-tolerant consensus algorithms."

Consensus algorithms — Paxos, Raft — require only a **majority quorum**. A Raft cluster of five nodes commits a log entry once three of them have it; it tolerates the loss of two. It elects a new leader from the survivors when the old one dies, and a recovery process brings everyone back to a consistent state. The contrast is stark: "consensus algorithms only require votes from a majority of nodes, unlike 2PC where all the participants must say YES." Because a majority of a cluster can exist on at most one side of a partition, consensus can keep making progress on the majority side while a minority is down — and it can never split-brain, because two disjoint majorities cannot exist. The before-after figure draws the comparison directly.

![A before-after comparison: 2PC needs ALL votes (5 of 5 must vote YES, 1 crash means no decision, coordinator crash blocks all, availability falls with N) versus consensus needs a MAJORITY (3 of 5 form a quorum, tolerates 2 node failures, new leader elected and finishes, availability rises with N)](/imgs/blogs/two-phase-commit-and-how-it-fails-4.webp)

Look at the availability arithmetic. Suppose each node is independently up with probability $p$. A 2PC transaction across $n$ participants can only commit if all $n$ are up *and* the single coordinator is up, so its availability is roughly $p^{n+1}$ — a number that *shrinks* as you add participants. Every node you add is another thing that must be alive. Consensus over $n$ nodes (with $n$ odd) makes progress whenever a majority $\lceil (n+1)/2 \rceil$ are up; that probability *grows* toward 1 as $n$ increases, because you can lose more and more of the minority. For $p = 0.99$: three-node 2PC (coordinator plus two participants, so four "all-up" requirements) is up about $0.99^4 \approx 96.1\%$ of the time; five-node Raft tolerating two failures is up about $99.999\%$ of the time. Same hardware reliability, opposite curves. That is why your etcd, your CockroachDB ranges, your Kafka controller quorum all run consensus, and why nobody builds a *fault-tolerant* system by adding more participants to a 2PC.

| Dimension | Two-phase commit | Consensus (Paxos/Raft) |
| --- | --- | --- |
| Votes needed to decide | **All** participants | **Majority** quorum |
| Tolerates node failures | No — any failure stalls it | Yes — up to $\lfloor n/2 \rfloor$ |
| Coordinator/leader failover | None — single point of failure | Automatic leader election |
| Availability as $N$ grows | Falls ($\approx p^{N+1}$) | Rises (toward 1) |
| Blocking on crash | Yes (in-doubt window) | No (new leader continues) |
| What it decides | One transaction's commit/abort | An infinite ordered log of values |
| Right job | Atomic commit across *known* nodes | Replicating state fault-tolerantly |

Two caveats so the contrast isn't misread. First, 2PC and consensus solve slightly different problems: 2PC decides one transaction's outcome across a *specific, known* set of participants (you genuinely need *all* of them, because each owns data the transaction touches — you cannot commit a transfer if the account-holding shard is down, majority or not); consensus agrees on a value among *interchangeable replicas*, any majority of which suffices. You can't simply "replace 2PC with Raft" because the participants in 2PC are not redundant copies — they are different data. Second — and this is the synthesis the best modern systems exploit — you can *combine* them: run 2PC across the participants for atomicity, but make each participant (and the coordinator) internally a *consensus group* so that no single machine failure can stall the protocol. That is precisely what Spanner does, and it is the subject of Section 9.

## 7. XA and the misery of distributed transaction managers

> **Senior rule of thumb:** the moment your design doc says "XA" or "distributed transaction manager spanning two different databases," stop and price in the operational tax: in-doubt branches, heuristic decisions, a recovery log that is itself a single point of failure, and an on-call engineer manually resolving stuck transactions at 3 a.m.

When the participants are *heterogeneous* — a Postgres database, an Oracle database, and a JMS message broker — you need a standard protocol so a single coordinator can drive 2PC across all of them. That standard is **XA** (eXtended Architecture), defined by The Open Group's X/Open Distributed Transaction Processing model. XA is, in Kleppmann's deflating phrase, "just a C API for interfacing with the 2PC coordinator" — a set of functions (`xa_start`, `xa_end`, `xa_prepare`, `xa_commit`, `xa_rollback`, `xa_recover`) that each resource manager implements and that a transaction manager calls to run the prepare/commit dance. The architecture has the shape in the figure below: an application begins a global transaction with a **transaction manager** (the coordinator); the TM enlists each **resource manager** (a participant) and, at commit, drives 2PC across all of them; and the TM keeps a durable transaction log so it can recover in-doubt branches after a crash.

![A graph of XA architecture: an application (xa_start work) calls a Transaction Manager (2PC coordinator) which writes to a TM txn log (single point of failure) and sends prepare to three resource managers (Postgres, Oracle, JMS broker); one RM, when the TM is gone, becomes an in-doubt branch requiring a heuristic decision](/imgs/blogs/two-phase-commit-and-how-it-fails-6.webp)

Every problem from Section 4 reappears here, sharpened by heterogeneity:

**The transaction manager is a single point of failure, and it's often in your app process.** In the common XA deployment, the TM is an in-process library inside your application server (think JTA in a Java app server). Its transaction log lives on that host. Kleppmann notes the consequence directly: if the TM is "not a separate service but rather a library loaded into the same process as the application that is issuing the transaction," then when that application server crashes, "the coordinator's log is on the crashed machine's local disk." That log holds the decisions for in-doubt transactions across *all* the enlisted databases — so until that specific application instance restarts and replays its log, those databases sit with prepared transactions holding locks. You have coupled the availability of multiple independent databases to the uptime of one stateful app process. Kleppmann again: "if the coordinator fails and 10% of transactions are left in doubt... those locks may be held for hours or even days." A single point of failure that lives in your most frequently-deployed, most frequently-crashed tier is a poor place to put your durability anchor.

**Heuristic decisions are atomicity violations with a polite name.** When an in-doubt branch holds critical locks and the TM cannot recover in a reasonable time, operators are given an escape hatch: manually force the branch to commit or roll back. The XA/transaction-processing literature calls this a **heuristic decision** — "a unilateral decision during the completion stage of a distributed transaction to commit or rollback updates," which "can leave distributed data in an indeterminate state." Read that carefully: a heuristic decision means a human (or a timeout policy) guesses the outcome of an in-doubt branch *without knowing the global verdict*. If the guess is wrong — you roll back a branch the TM had decided to commit — you have just torn the transaction in half. Heuristic decisions exist precisely because the blocking is otherwise unbounded, and they "trade" the blocking for a *possible* loss of atomicity that a human takes responsibility for. The very existence of a heuristic-decision feature is an admission that the protocol can wedge badly enough that breaking it by hand is the lesser evil.

**Recovery is manual and database-specific.** Each RM exposes its in-doubt branches differently. In Postgres you query `pg_prepared_xacts` and resolve with `COMMIT PREPARED`/`ROLLBACK PREPARED`; in Oracle you look at `DBA_2PC_PENDING`; in MySQL you run `XA RECOVER`. Operators must correlate a global transaction id across multiple heterogeneous systems and decide each branch's fate consistently — by hand, under time pressure, while locks pile up. There is no single "kubectl rollback" for a wedged XA transaction.

**XA constrains everything it touches.** Because a prepared branch holds locks indefinitely in the bad case, XA pushes you toward locking concurrency control everywhere, interacts badly with connection pools (a connection mid-XA cannot be returned to the pool), and forbids many useful optimizations. It also tends to span technologies with wildly different failure and recovery semantics, so the weakest link sets the reliability of the whole.

This is why XA earned its reputation. It is not that 2PC is *wrong*; it is that XA applies 2PC across *independent, heterogeneous, separately-operated* systems, which maximizes the blast radius of every failure mode and minimizes the chance of fast automatic recovery. For coupling two unrelated databases plus a broker into one atomic action, the cure is frequently worse than the disease.

## 8. Why microservices ran away from 2PC

> **Senior rule of thumb:** distributed transactions across service boundaries don't just risk blocking — they couple the availability of every service in the transaction and add a synchronous prepare/commit round trip to every write. In a microservice architecture, that is the opposite of what you bought microservices for.

There's a reason the microservices movement treats 2PC as an anti-pattern. Three reasons, actually, and they compound:

1. **Coupling.** A 2PC across the Orders service, the Payments service, and the Inventory service makes a write succeed only if *all three* are up and reachable at the same instant — and (per Section 6) the availability of that compound action is the *product* of the individual availabilities, which is strictly worse than any one of them. You spent significant effort decomposing a monolith into independently deployable, independently failing services; wrapping them back into a synchronous all-or-nothing commit re-couples them at the worst possible layer.
2. **Blocking.** Every failure mode from Section 4 now plays out *across team and service boundaries*. An in-doubt transaction holds locks (or worse, business-level reservations) inside another team's service, which you cannot inspect or resolve. The blast radius of a coordinator crash crosses org charts.
3. **Latency and throughput.** 2PC adds a synchronous prepare round trip plus a commit round trip — and two `fsync`s — to *every* cross-service write, and holds locks across both. In a service mesh where each hop is already milliseconds, this is a steep, permanent tax on the hot path, and it serializes contended resources for the duration of the protocol.

So microservices reach for a different model entirely: the **saga**. Instead of one atomic distributed transaction, a saga is a *sequence of local transactions*, each committing independently in its own service, with **compensating transactions** to undo earlier steps if a later one fails. The order service commits the order; the payment service commits the charge; if inventory then fails, you run a compensating refund and a compensating order-cancellation. Sagas give up isolation and atomicity-as-a-single-instant in exchange for not blocking and not coupling availability — you get *eventual* consistency with explicit compensation logic rather than *immediate* consistency with implicit blocking. The trade is real and not free (compensations are application code you must write and test; intermediate states are visible; semantic locks may be needed), but for cross-service workflows it is overwhelmingly the right one. I will not relitigate the saga design here — it has its own write-up: [the saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions). The takeaway for *this* article is the boundary: **2PC for atomic commit across nodes you fully control and that share a fate; sagas for workflows across services that must stay independent.**

## 9. Where 2PC is genuinely the right tool

> **Senior rule of thumb:** 2PC is excellent for committing across *shards of a single system you operate*, where the participants share fate, the coordinator is fast and local — and best of all, where you have made the coordinator's state fault-tolerant by replicating it. The blocking flaw is fatal only when the coordinator can stay dead.

After eight sections of failure, it would be easy to conclude 2PC is never worth it. That's wrong. 2PC is alive and well — inside systems that use it where its assumptions hold and that engineer away its one fatal flaw. The pattern is always the same: the participants are *shards of one system* (not independent databases owned by different teams), the coordinator is fast and co-located, and the coordinator's decision state is made *recoverable* so that no single machine failure leaves anyone permanently blocked.

### 9.1 Postgres prepared transactions

Postgres exposes the participant half of 2PC directly via `PREPARE TRANSACTION`, `COMMIT PREPARED`, and `ROLLBACK PREPARED`. The documentation is explicit that this is *not* for application use: "PREPARE TRANSACTION is not intended for use in applications or interactive sessions. Its purpose is to allow an external transaction manager to perform atomic global transactions across multiple databases or other transactional resources. Unless you're writing a transaction manager, you probably shouldn't be using PREPARE TRANSACTION." Here is the participant side of the protocol as runnable SQL:

```sql
-- Enable the feature first (default is 0, which disables it):
--   max_prepared_transactions = <set equal to max_connections> in postgresql.conf
-- Then, as the external transaction manager (coordinator), on EACH shard:

BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 17;  -- acquires X-locks
-- Phase 1: prepare. State is flushed to disk; the txn DETACHES from this session.
-- The GID is the global transaction id the coordinator uses to correlate shards.
PREPARE TRANSACTION 'txn-9f3a-shard-A';

-- ... coordinator collects YES from every shard, fsyncs its decision ...

-- Phase 2: commit the prepared txn by its GID (can be a DIFFERENT session!).
COMMIT PREPARED 'txn-9f3a-shard-A';
-- or, if the global decision was abort:
-- ROLLBACK PREPARED 'txn-9f3a-shard-A';
```

After `PREPARE TRANSACTION`, the documentation notes that "the transaction is no longer associated with the current session; instead, its state is fully stored on disk, and there is a very high probability that it can be committed successfully, even if a database crash occurs before the commit is requested." That is the durable promise from Section 2, in product form: the prepared transaction survives a crash and can be committed by *any* session, which is exactly what lets a coordinator recover it after either side restarts.

Now the operational gotcha, which is the failure modes of Section 3–4 made concrete. To find orphaned prepared transactions — branches the coordinator never resolved — query the catalog:

```sql
-- Every prepared transaction currently holding state in this cluster:
SELECT gid, prepared, owner, database,
       now() - prepared AS age
FROM   pg_prepared_xacts
ORDER  BY prepared;

-- Find what an orphan is blocking (a hung ALTER TABLE, other writers, etc.):
SELECT a.pid, a.state, a.query, l.locktype, l.mode, l.granted
FROM   pg_locks l
JOIN   pg_stat_activity a ON a.pid = l.pid
WHERE  NOT l.granted;

-- Resolve an orphan manually (this is a heuristic decision — be sure of the verdict):
ROLLBACK PREPARED 'txn-9f3a-shard-A';   -- or COMMIT PREPARED if that was the decision
```

Why you must monitor this: an orphaned prepared transaction holds its locks *and* its old snapshot indefinitely. Cybertec and Highgo both document the specific horrors — a prepared transaction that touched a system catalog can take a lock that makes *all future connection attempts hang*, and restarting the database does not help because the prepared transaction is durable and is restored on startup. More routinely, the orphan holds back `VACUUM`, accumulating dead tuples and pushing toward transaction-id wraparound, which can ultimately force the cluster to shut down to protect itself. The practical defenses: set `max_prepared_transactions` to zero (disable the feature entirely) unless you actually run a transaction manager that closes branches promptly; alert on any row in `pg_prepared_xacts` older than a few minutes; and a common operational trick is to *encode a deadline into the GID itself* (for example `'txn-9f3a 5m'`) so a janitor job can parse the age and roll back expired branches automatically. A prepared transaction is a loan against availability; if you take it, you must guarantee you'll pay it back fast.

### 9.2 MySQL XA

MySQL/InnoDB exposes 2PC through SQL-level XA statements. The sequence on a single participant:

```sql
-- Phase 1 on one resource manager (the xid is the global transaction id):
XA START 'gtrid-9f3a','shardB';
UPDATE accounts SET balance = balance + 100 WHERE id = 88;
XA END   'gtrid-9f3a','shardB';
XA PREPARE 'gtrid-9f3a','shardB';   -- durable promise; holds locks; in-doubt until phase 2

-- After the coordinator's decision, phase 2 (possibly from another connection):
XA COMMIT 'gtrid-9f3a','shardB';     -- or: XA ROLLBACK 'gtrid-9f3a','shardB';

-- Recovery: list in-doubt branches after a crash/disconnect:
XA RECOVER;
```

MySQL XA is notorious for a class of bugs that are themselves an object lesson in 2PC's durability requirements — specifically the interaction between InnoDB's internal redo log and the binary log (which is itself a 2PC participant for replication). The MySQL bug tracker has a long history here: bug **#88534** ("XA may lost prepared transaction and cause different between master and slave") describes the binlog being lost while InnoDB has a prepared XA transaction, so the master ends up with a prepared branch the slave never sees; bug **#98616** ("XA PREPARE/XA COMMIT/XA ROLLBACK lost if mysql crash just after binlog flush") describes a crash window where MySQL loses track of the external xid and cannot recover the branch on startup; bug **#87560** and bug **#109434** describe crash windows where InnoDB and the binlog disagree about whether a branch is prepared or committed, leading to incorrect rollback or a stranded branch. Every one of these is the same root issue: the prepare record and the decision must be `fsync`'d in the correct order across *two* logs, and any crash that lands between those flushes can leave an in-doubt branch that recovery cannot resolve. The lesson generalizes far beyond MySQL: **2PC is only as correct as the durability and ordering of its log writes.** If you do not `fsync` the prepare before voting yes, or the decision before Phase 2, the protocol's safety argument collapses — and you get exactly the lost/duplicated branches these bug reports document.

### 9.3 Percolator: client-coordinated 2PC over shards

Google's Percolator adds cross-row, cross-shard ACID transactions on top of Bigtable, which natively offers only single-row atomicity. It runs 2PC, but with two design moves that sidestep the worst of the blocking. The pipeline below traces a Percolator commit.

![A pipeline of Percolator's commit: client buffers writes, prewrite locks all cells, sets a primary lock as the decider, commits the primary at commit_ts, asynchronously cleans the secondaries by rolling the decision forward, and the snapshot-isolated commit becomes visible](/imgs/blogs/two-phase-commit-and-how-it-fails-9.webp)

The first move: the **client is the coordinator**, and the transaction's state lives in Bigtable itself rather than in a separate coordinator process. In the prewrite phase the client tries to lock all the cells it's writing. The second, cleverer move: among all the locks, one is designated the **primary lock**, and the *state of that single cell is the authoritative decision*. The transaction is committed iff the primary lock has been replaced by a committed write record at a commit timestamp. This collapses the "where is the decision?" question — which in classic 2PC depends on a fragile coordinator log — into "read one Bigtable cell," and because Bigtable replicates that cell, the decision is as durable and available as Bigtable is. The secondary locks are cleaned up *lazily and idempotently*: if a later transaction stumbles on a leftover lock, it can look up the primary, determine the original transaction's true outcome, and either roll it forward (if the primary committed) or clean it up (if not). This "lazy recovery" means a crashed client does not block anyone forever — any other transaction that hits the orphaned locks repairs the state on demand. The coordinator's single-point-of-failure flaw is dissolved by putting the decision in a replicated store and making recovery a peer responsibility rather than a coordinator responsibility. Percolator provides snapshot isolation via MVCC and a central timestamp oracle, not serializability, which is the right trade for its bulk-incremental-processing workload.

### 9.4 Spanner: make the coordinator itself fault-tolerant with Paxos

Spanner is the cleanest demonstration of Section 6's synthesis: it runs *2PC for atomicity across shards, with each shard a Paxos group for fault tolerance*. The figure below shows the structure.

![A graph of Spanner's design: a client picks a coordinator; the coordinator leader is a Paxos group that replicates its decision to coordinator replicas; it drives prepare to participant leaders, each itself a Paxos group replicating to a 3-of-5 quorum; on a leader crash, a new leader is elected and resumes the in-flight 2PC](/imgs/blogs/two-phase-commit-and-how-it-fails-7.webp)

Each participant in the 2PC is not a single machine but an entire Paxos-replicated group (a set of replicas across geographies). One group's leader is chosen as the 2PC coordinator; the others' leaders are participants. The decisive change is *where the 2PC state lives*: "all states of the 2PC for both the coordinator and participant are recorded in their Paxos state machine, so if any crash in the middle of 2PC, the new leader can continue and complete the 2PC." Re-read Section 4.3 with this in mind. The fatal flaw was that the coordinator held the only copy of the decision, so a coordinator crash stranded everyone. In Spanner the coordinator's prepare and decision records are committed *through Paxos to a majority of the coordinator's replicas before they take effect* — so a coordinator-machine crash is no longer a single point of failure. A new leader is elected from the replicas (the consensus failover from Section 6), reads the replicated 2PC state, and *resumes the in-flight transaction* — driving Phase 2 to completion or aborting as the durable state dictates. The participants never enter unbounded blocking, because the entity they're waiting on always has a live, up-to-date successor. Spanner layers TrueTime on top to assign globally meaningful commit timestamps and provide external consistency (serializability across the whole system), but the fault-tolerance story is precisely "2PC over Paxos": keep 2PC's all-must-agree atomicity for the transaction, and borrow consensus's majority-quorum fault tolerance for each role's *state*. CockroachDB and TiKV (the latter directly modeled on Percolator) use the same recipe — a transaction's 2PC coordinated across ranges, each range a Raft group — which is why those systems can lose a node mid-transaction and still finish it. That is the whole arc of this article in one architecture: the blocking flaw is fatal only when the coordinator can stay dead, and replication is what stops it from staying dead.

| System | Participants are | Coordinator is | How it dodges the blocking flaw |
| --- | --- | --- | --- |
| Postgres prepared txns | Postgres databases | External TM (you) | You must monitor `pg_prepared_xacts` and resolve orphans fast |
| MySQL XA | InnoDB + binlog + others | External TM (you) | `XA RECOVER` + careful fsync ordering; still operator-driven |
| Classic XA / JTA | Heterogeneous DBs + brokers | In-process TM library | Doesn't — SPOF on app host; heuristic decisions as escape hatch |
| Percolator / TiKV | Bigtable / RocksDB shards | The client | Decision = primary-lock cell in a replicated store; lazy peer recovery |
| Spanner / CockroachDB | Paxos/Raft groups (shards) | A group leader | 2PC state replicated via consensus; new leader resumes in-flight 2PC |

## Case studies from production

What follows are concrete, named incidents — some from public bug trackers and documentation, some composites of the same failure I've watched recur across teams. Each is an instance of a specific section above made painfully real.

### 1. The prepared transaction that locked out every login

A team enabled `max_prepared_transactions` to integrate a JTA transaction manager, shipped it, and moved on. Weeks later, every new connection to one Postgres cluster began hanging at authentication — not erroring, hanging. The wrong first hypothesis was a connection-pool leak. The actual root cause: an application instance had crashed mid-transaction, leaving an orphaned prepared transaction that happened to hold a lock touching a system catalog used during connection setup. As Cybertec documents, a prepared transaction can hold a lock that makes "all future connection attempts hang," and "restarting the database won't help, because the prepared transaction will be retained" — it is durable by design. The fix was to connect through an already-open session (or single-user mode), query `pg_prepared_xacts`, and `ROLLBACK PREPARED` the orphan. The lesson: enabling prepared transactions without an alert on `pg_prepared_xacts` age is shipping a latent total-outage. The orphan does not announce itself; it silently strangles the cluster, and durability — the feature's whole point — is what makes a restart useless.

### 2. MySQL bug #88534 — the prepared branch the replica never saw

Reported on the MySQL bug tracker, this is the canonical XA-durability ordering bug. When InnoDB's `prepare` ran before the binary-log write, a crash in the window could lose the binlog entry while leaving InnoDB with a prepared XA transaction. On recovery, the master had an in-doubt branch the slave had never heard of, so master and slave diverged. The wrong first hypothesis was replication lag; the actual cause was a two-log 2PC (storage engine plus binlog) whose `fsync` ordering was wrong in a crash window. The proposed fix pinned the order: engine prepare, then binlog, and ensure engine operations complete before binary-log rotation. The lesson is Section 9.2's: 2PC's correctness *is* the durability and ordering of its log writes. Get the order wrong by one step and the safety proof — every prepared branch is recoverable to a single decision — silently breaks, producing exactly the lost and duplicated branches the bug describes.

### 3. The JTA coordinator that died with the app server

A Java service used XA to write atomically to an Oracle database and a JMS queue, with the JTA transaction manager running in-process. During a routine deploy, the app server was killed (SIGKILL, not a graceful shutdown) while several global transactions were between prepare and commit. The coordinator's recovery log was on that pod's ephemeral local disk — which the orchestrator promptly reclaimed. Both Oracle and the JMS broker now had in-doubt branches with no coordinator log to resolve them, holding locks. This is exactly Kleppmann's warning: when the TM is "a library loaded into the same process as the application," a crash leaves "the coordinator's log on the crashed machine's local disk," and the enlisted resources block. Resolution required a DBA to query `DBA_2PC_PENDING`, an operator to inspect the broker's prepared messages, and a *heuristic decision* on each branch — guessing the verdict, because the authoritative log was gone. One guess was wrong, tearing one transaction in half. The lesson: never put the durability anchor of a multi-system transaction on the disk of your most-frequently-killed tier.

### 4. The "we'll just add a timeout" 3PC reinvention

An engineer, frustrated by 2PC blocking, proposed (in effect) reinventing 3PC: "if a prepared participant doesn't hear from the coordinator within N seconds, let it decide on its own." It shipped to a staging environment and passed every test — because the test network never partitioned. The first real partition split the participants; some had heard "going to commit" and committed on timeout, others had not and aborted on timeout, and a single logical transaction landed half-committed. The wrong assumption was that a timeout means "the coordinator is dead" rather than "the network is slow." This is precisely why 3PC fails in practice: it needs a perfect failure detector and a synchronous network, and on a real network "if the network splits, different partitions may reach different conclusions." The fix was to rip out the timeout-decides logic and instead make the coordinator state recoverable (Section 9.4's approach). The lesson: you cannot buy non-blocking with a timeout; you buy it with replication.

### 5. The cross-shard report that froze the OLTP workload

A team built cross-shard transactions on Postgres prepared transactions with a homegrown coordinator. A long-running analytical transaction prepared on all shards and then the coordinator process got stuck in a slow downstream call before sending Phase 2. For the duration, every shard held the prepared transaction's locks, and the OLTP workload that touched the same hot rows queued behind them — latency spiked, then requests timed out cluster-wide. The wrong first hypothesis was a CPU saturation event. The actual cause was Section 3 in the flesh: a prepared transaction is the most expensive idle object in the database, and the uncertainty window had been stretched by a slow coordinator into a multi-minute lock-hold. The fix was twofold: keep transactions short before preparing, and put a hard timeout on the coordinator's Phase-1-to-Phase-2 latency that aborts rather than lingers. The lesson: the cost of "prepared" is paid every second the window is open, so the window must be ruthlessly bounded.

### 6. The XA connection-pool starvation

A service used XA across two databases through a JDBC connection pool. Under load, throughput collapsed and the pool exhausted. The wrong hypothesis was a pool-size misconfiguration; doubling the pool only delayed the collapse. The actual cause: a connection enlisted in an in-flight XA transaction cannot be returned to the pool until the transaction completes, and the prepare/commit round trips (plus held locks) kept connections checked out far longer than ordinary statements. As the in-doubt windows lengthened under contention, connections piled up in the enlisted state and the pool drained. This is the operational drag from Section 7: XA constrains everything it touches, including pooling. The fix was to stop using XA for that path entirely and move to an idempotent, saga-style flow. The lesson: 2PC's held resources are not just rows — they're connections, snapshots, and pool slots, and they all stay pinned for the whole protocol.

### 7. CockroachDB surviving a node loss mid-transaction

A positive case. A CockroachDB cluster lost a node (hardware failure) while a multi-range transaction was mid-commit. Nothing blocked. Because each range is a Raft group and the transaction's 2PC state is replicated through Raft to a quorum, the failed node's ranges elected new leaders from their surviving replicas, and the new leaders resumed the in-flight transaction from the replicated state. The transaction committed; clients saw a brief latency blip, not an outage. This is Section 9.4 working as designed: 2PC for atomicity across ranges, consensus for fault tolerance of each range's state, so a single machine death is a routine failover rather than an unbounded block. The lesson: the blocking flaw is not intrinsic to *using* 2PC — it is intrinsic to a *non-replicated coordinator*. Replicate the coordinator's state and the flaw goes away.

### 8. The orphaned prepared transaction that ate the disk

A monitoring gap let a single orphaned prepared transaction live for two weeks on a Postgres cluster. No locks on hot rows, so nobody noticed any blocking — but the transaction's old snapshot held back `VACUUM` the entire time. Dead tuples accumulated, table and index bloat ballooned, disk filled, and query plans degraded as the heap grew. The wrong hypothesis was an application write-volume spike. The actual cause was the documentation's exact warning: a long-lived prepared transaction "will interfere with the ability of VACUUM to reclaim storage, and in extreme cases could cause the database to shut down to prevent transaction ID wraparound." The fix was `ROLLBACK PREPARED` on the orphan, an aggressive `VACUUM`, and an alert on `pg_prepared_xacts` age thereafter. The lesson: prepared transactions damage you even when they aren't blocking anything — the MVCC and vacuum cost is silent and cumulative.

### 9. MySQL bug #109434 — crash between binlog and engine commit

Another MySQL XA durability window: MySQL crashed after syncing the binlog's `XA COMMIT` event but before InnoDB committed the branch. On recovery, the binlog said "committed" while InnoDB still held the branch as merely prepared — the two participants in MySQL's internal 2PC disagreed about the outcome. Recovery has to reconcile this by trusting the binlog as the source of truth and rolling the engine forward, but the bug was that the reconciliation didn't handle this exact ordering, risking a stranded prepared branch. The lesson reinforces #2: a 2PC built on two logs is only correct if recovery has a single, well-defined rule for which log is authoritative and applies it for every crash window. Ambiguity in "which participant decides" is the same bug as Section 4.3's "who holds the decision," just inside one process.

### 10. The heuristic decision that split a payment

A long-blocked XA branch was holding locks on a payments table, and an operator, under pressure to restore service, issued a heuristic `ROLLBACK` on it — reasonable, since most stuck branches end up aborted. But this particular global transaction had been *decided commit* by the TM before it crashed; another resource had already committed its half. The heuristic rollback tore the transaction: the payment was recorded in one system and not the other, surfacing later as a reconciliation discrepancy. This is the documented hazard of heuristic decisions — "a unilateral decision... can leave distributed data in an indeterminate state." The lesson is sobering: heuristic decisions exist *because* 2PC can block unbearably, but they are guesses about an unknown verdict, and a wrong guess is an atomicity violation a human signed off on. The only real cure is to not be in a position where a human must guess — i.e., make the coordinator recoverable so the verdict is never lost.

## When to reach for 2PC, and when not to

Reach for two-phase commit when:

- **You are committing across shards of a single system you operate**, where the participants share fate and you control the coordinator — distributing one logical database across nodes, not stitching together independent products.
- **The coordinator's decision state is replicated** (via Paxos/Raft, as in Spanner/CockroachDB/TiKV) or **lives in a replicated store** (as in Percolator's primary-lock cell), so a coordinator-machine crash is a failover, not an unbounded block. This is the single most important precondition; without it you are shipping the fatal flaw.
- **The uncertainty window is short and bounded** — fast local network between participants, fast disks for the prepare/decision `fsync`s, short transactions, and a hard timeout that aborts rather than lingers in Phase 1.
- **You genuinely need immediate, all-or-nothing atomicity** across the participants and cannot tolerate the visible intermediate states a saga exposes — and the participants are not interchangeable replicas (so consensus alone can't replace it).
- **You have monitoring and automated recovery for in-doubt/prepared state** — alerts on `pg_prepared_xacts` / `XA RECOVER` age, and a janitor that resolves orphans by GID quickly.

Skip 2PC (or its XA incarnation) when:

- **The participants are independent, heterogeneous, separately-operated systems** — two unrelated databases plus a message broker, owned by different teams. The blast radius of every failure mode is maximized and automatic recovery is least likely. This is where XA earns its bad name; prefer the outbox pattern, idempotent retries, or a saga.
- **You're crossing microservice boundaries.** The coupling, blocking, and per-write latency tax defeat the purpose of the architecture. Use a [saga with compensating transactions](/blog/software-development/database/saga-pattern-distributed-transactions) instead and accept eventual consistency with explicit compensation.
- **The coordinator is non-replicated and lives on a tier that gets killed often** — an in-process TM on an app pod with an ephemeral local-disk log. A coordinator crash there blocks every enlisted resource for the coordinator's (possibly very long) recovery time.
- **You actually need fault tolerance against node loss, and the nodes are redundant copies.** That's consensus's job, not 2PC's. Don't add participants to a 2PC hoping for reliability — you get the opposite ($\approx p^{N+1}$ availability). Run [Raft](/blog/software-development/database/raft-consensus-from-scratch) over replicas instead.
- **You're tempted to "fix" blocking with a timeout that lets prepared participants decide alone.** That is 3PC, it requires a synchronous network and a perfect failure detector you do not have, and under partition it will split your data. Buy non-blocking with replication, not with a timer.

The through-line: two-phase commit is a precise, *safe* answer to atomic commit across nodes — it will never let one node commit while another aborts. Its one flaw is liveness: a coordinator that crashes in the uncertainty window strands prepared participants in-doubt, holding locks, until it returns. Every successful production use of 2PC is a use that has engineered away the "until it returns" — by replicating the coordinator's state so it always returns, fast, in the form of a freshly-elected successor. Use 2PC where you can guarantee that. Everywhere else, the protocol's safety is not worth its blocking, and you should reach for consensus (for replicated state) or sagas (for cross-service workflows) instead.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 9 — "Atomic Commit and Two-Phase Commit," the limits of XA, coordinator failure and in-doubt transactions. The clearest treatment in print.
- [PostgreSQL: PREPARE TRANSACTION documentation](https://www.postgresql.org/docs/current/sql-prepare-transaction.html) and [the pg_prepared_xacts view](https://www.postgresql.org/docs/current/view-pg-prepared-xacts.html) — the official warnings about locks, VACUUM, and wraparound.
- [Cybertec: prepared transactions and their dangers](https://www.cybertec-postgresql.com/en/prepared-transactions/) and [Highgo: understanding prepared transactions and handling the orphans](https://www.highgo.ca/2020/01/28/understanding-prepared-transactions-and-handling-the-orphans/) — the operational gotchas in detail.
- MySQL XA bug reports [#88534](https://bugs.mysql.com/bug.php?id=88534), [#98616](https://bugs.mysql.com/bug.php?id=98616), [#87560](https://bugs.mysql.com/bug.php?id=87560), [#109434](https://bugs.mysql.com/bug.php?id=109434) — durability/ordering hazards in real 2PC.
- [The Paper Trail: Consensus Protocols — Two-Phase Commit](https://www.the-paper-trail.org/post/2008-11-27-consensus-protocols-two-phase-commit/) — the classic analysis of why 2PC blocks on coordinator failure.
- [Three-phase commit protocol (Wikipedia)](https://en.wikipedia.org/wiki/Three-phase_commit_protocol) — the synchronous-network assumptions and partition behavior.
- [Spanner: Google's Globally Distributed Database](https://mwhittaker.github.io/papers/html/corbett2013spanner.html) and [Percolator: Large-scale Incremental Processing](https://research.google/pubs/large-scale-incremental-processing-using-distributed-transactions-and-notifications/) — how to make the coordinator fault-tolerant via Paxos and via a primary-lock cell.
- Sibling posts on this blog: [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch), [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production), and [the saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions).

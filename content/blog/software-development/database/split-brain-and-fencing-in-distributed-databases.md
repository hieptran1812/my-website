---
title: "Split-Brain and Fencing: Preventing Two Leaders from Corrupting Your Data"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer deep dive into split-brain — why naive failover promotes a second leader while the first is only paused, and how fencing tokens, leases, STONITH, and quorum stop two leaders from corrupting shared storage."
tags:
  [
    "split-brain",
    "fencing-tokens",
    "distributed-systems",
    "failover",
    "leases",
    "stonith",
    "high-availability",
    "consensus",
    "quorum",
    "databases",
    "raft",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/split-brain-and-fencing-in-distributed-databases-1.webp"
---

On October 21, 2018, a maintenance crew swapped a piece of failing 100G optical equipment and accidentally disconnected GitHub's US East Coast data center from the network for forty-three seconds. Forty-three seconds. The link came back almost before anyone noticed. The outage that followed lasted **twenty-four hours and eleven minutes**, and it was not caused by the broken cable. It was caused by what the cluster *did* during those forty-three seconds: the automated failover system, seeing the East Coast primary stop answering, promoted a West Coast replica to be the new primary. When the link healed, both data centers had a database that believed it was the authority, both had ingested writes the other had never seen, and there was no safe way to merge them. GitHub spent the next day reconciling two divergent copies of its own data by hand.

That is **split-brain**, and it is the single most expensive failure mode in distributed data. It is not a rare bug in exotic systems; it is the *default* outcome of naive failover, and it has bitten GitHub, it has bitten HBase, it has bitten countless two-node Postgres pairs wired up with a shell script and a cron job. The reason it is so common is that it springs from an assumption that feels obviously true and is in fact false: *if a node stops answering, it has stopped doing work.* In a distributed system that assumption is wrong often enough to ruin you. A node that stops answering might be dead — or it might be garbage-collecting, or swapping, or paused by a hypervisor migration, or simply on the wrong side of a network partition, very much alive, very much still writing, and completely unaware that anyone has declared it dead.

This article is the long version of how you stop two leaders from corrupting your data. The diagram above is the mental model for the whole piece: a partition (or an over-eager failure detector) lets two nodes both believe they are the leader, both accept writes, and the two write streams diverge into data that cannot be reconciled. Everything else — fencing tokens, leases, STONITH, quorum, witnesses, consensus epochs — is a different mechanism for making sure that *at most one of those two writers can actually change the shared state.* The deepest idea, the one Martin Kleppmann hammers in Chapter 8 of *Designing Data-Intensive Applications*, is that **a node cannot be trusted to know whether it is still the leader** — so the fix can never live only in the node. It has to live in the resource the node is trying to write to.

## The mental model: one partition, two leaders, two write streams

![Split-brain: a partition lets the old and new leader both accept writes that the other never sees, producing data with no safe merge](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-1.webp)

Read the figure left to right. Some clients can still reach the **old leader** — it is paused or isolated, but it has not crashed, and the clients sitting on the same side of the partition keep sending it writes, which it keeps accepting. Other clients reach the **new leader**, which a failover process promoted when the old one stopped responding. Both leaders append to their own write log. For the duration of the split — forty seconds, forty minutes, however long it takes a human to notice — each log accumulates rows the other will never see. When the partition heals, you do not have one database that is slightly behind another. You have two databases that have each made authoritative, conflicting decisions about the same rows, and there is no algorithm that can merge them without losing somebody's writes or violating an invariant. That final red box — *divergence, no safe merge* — is the whole problem. The twenty-four hours of manual repair is what divergence costs.

The crucial subtlety, and the thing that trips up every first attempt at high availability, is that the old leader is *not dead.* If it were dead, there would be no split-brain — a dead node writes nothing. Split-brain is specifically the failure where the deposed leader is alive, reachable by *some* clients, and acting on a stale belief that it is still in charge. So the question "how do we do safe failover" is really two questions stacked on top of each other:

1. **How do we detect that the old leader is gone** without falsely accusing a leader that is merely slow?
2. **Given that we will sometimes promote a new leader while the old one is still alive, how do we guarantee the old one can no longer corrupt shared state?**

You cannot fully solve the first. Failure detection over an asynchronous network is fundamentally a guess — you cannot distinguish "dead" from "slow" with certainty, a result that goes all the way back to the FLP impossibility theorem. So a well-built system *assumes the first detector will sometimes be wrong* and pours its engineering into the second question. That is what fencing is. Fencing is the discipline of building your system so that being wrong about who is the leader is survivable, because the resource itself refuses to be written by the loser.

> The senior mental shift is this: stop trying to guarantee that only one node *thinks* it is the leader. That is impossible. Instead, guarantee that only one node can *act* as the leader against shared state. Two nodes may both believe they are primary; only one may successfully write.

The rest of this piece is a tour of the mechanisms that enforce that guarantee, from the cheapest (a monotonically increasing token on every write) to the most physical (cutting the other node's power), plus the voting math (quorum, odd node counts, witnesses) that keeps two partitions from both feeling entitled to lead. Throughout, we lean on Kleppmann's *Designing Data-Intensive Applications*, Chapters 8 ("The Trouble with Distributed Systems") and 9 ("Consistency and Consensus"), because that book contains the canonical articulation of every idea here, and on the real incidents — GitHub, HBase, the Redlock debate — that taught the industry these lessons the expensive way. If you want the upstream context, this article is a direct sequel to [database replication](/blog/software-development/database/database-replication-sync-async-logical-physical) (which sets up the single-leader topology), [failure detection](/blog/software-development/database/failure-detection-gossip-and-phi-accrual) (which is question 1 above), [Raft from scratch](/blog/software-development/database/raft-consensus-from-scratch) (which solves a piece of this and leaves a piece for the storage layer), and [time, clocks, and ordering](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems) (because leases live and die by clock skew).

## Why split-brain is different from a normal failure

**Senior rule of thumb: a normal failure costs you availability; a split-brain costs you correctness, and correctness does not come back when the network does.** This is the distinction that justifies the whole engineering investment. Most failures are *fail-stop*: a disk dies, a kernel panics, a process gets OOM-killed, and the affected node produces no further output. Fail-stop failures are easy to reason about because the failed component does nothing wrong — it just does nothing. You lose some capacity, you fail over, you move on. Split-brain is categorically different: the failed component keeps *doing things*, and those things are wrong because they are based on a stale assumption. A fail-stop node is silent; a split-brained node is a confident liar that does not know it is lying.

| | Normal (fail-stop) failure | Split-brain failure |
| --- | --- | --- |
| What the failed node does | Nothing — produces no output | Keeps accepting and applying writes |
| What it costs | Availability (downtime) | Correctness (divergent / corrupted data) |
| Recovery | Restart or fail over; data intact | Manual reconciliation; data may be unrecoverable |
| Detectability | Obvious (node is silent) | Often silent until logs diverge | 
| Time to damage | None — node is already stopped | Grows with the duration of the split |
| The wrong assumption | (none) | "Unreachable means stopped" |

The last row is the root of everything. A naive failover system encodes the assumption "if I cannot reach the leader, it has stopped" directly into its promotion logic. That assumption is violated by at least five common conditions, none of which is exotic:

- **Network partition.** The leader is fine; the link between it and the failover controller is down. The leader keeps serving clients on its side.
- **Garbage-collection pause.** A stop-the-world GC freezes every thread for hundreds of milliseconds to, in pathological cases, *minutes*. Kleppmann notes stop-the-world pauses "have sometimes been known to last for several minutes." During the pause the node sends no heartbeats and looks dead.
- **Hypervisor / VM pause.** A live migration suspends the whole VM. The guest OS does not even know time passed; it resumes mid-instruction as if nothing happened — except seconds or minutes elapsed in the rest of the world.
- **Disk / IO stall.** An `fsync` on a degraded EBS volume can block for tens of seconds. The process is "running" but wedged on a syscall and emits no heartbeat.
- **Clock jump.** NTP steps the clock, or the leader's notion of when its lease expires is simply wrong, and it believes it still holds leadership after the lease has actually expired.

Every one of these makes a *live* node look dead to an observer, and a live-but-presumed-dead node is exactly the raw material for split-brain. As Kleppmann puts it, "a node in a distributed system must assume that its execution can be paused for a significant amount of time at any point" — and, critically, it will not even notice the pause until it next checks the clock. So we cannot ask the node to police itself ("am I still the leader? let me check"). By the time it checks, the answer it gets may already be stale. The discipline has to live elsewhere.

## Naive failover: the deposed leader is not dead

**Senior rule of thumb: any failover that does not actively prevent the old leader from writing is not a failover — it is a coin flip on data corruption.** Let us walk the canonical corruption timeline concretely, because the shape of it is the same in every incident, from a toy two-node setup to GitHub's geo-distributed MySQL fleet.

![Naive failover timeline: a GC pause makes a live leader look dead, a new leader is promoted, and the old leader wakes and writes on a stale assumption](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-2.webp)

The timeline above is the textbook scenario, almost word-for-word the one Kleppmann uses to motivate fencing:

- **t=0s.** The old leader holds the lease and is serving writes normally. Healthy. Heartbeating.
- **t=8s.** The old leader enters a stop-the-world GC pause. Its threads are frozen. It is not dead — it is *paused* — but from the outside it is indistinguishable from dead. No heartbeats go out. Note that the leader does not *know* it is paused; from its own perspective, no time is passing.
- **t=12s.** The failure detector has missed three heartbeats and crosses its threshold. It marks the old leader **DEAD**. This is a guess, and right now it is a *wrong* guess, but the detector has no way to know that.
- **t=14s.** A new leader is promoted and begins serving writes. The system now has two nodes that both believe they are leader. Only one of them knows it.
- **t=22s.** The GC pause ends. The old leader's threads resume *exactly where they left off* — mid-transaction, mid-request — with no awareness that fourteen seconds passed or that it has been deposed. It finishes the write it was working on and sends it to shared storage, stamped with its stale belief that it is the authority.
- **t=22s and after.** Two writers now hold the shared storage. The old leader's write lands on top of, or interleaved with, the new leader's writes. The data is corrupt. Nobody got an error. Both leaders think they succeeded.

The lethal moment is **t=22s**: the resumed write. The old leader was frozen for fourteen seconds and resumes as if it were still t=8s. It has a half-finished operation in flight — a balance to decrement, a row to update, a file to truncate — and it completes it against the live shared resource. Here is what that looks like in code — a deliberately naive lock-and-write that trusts a TTL lock and nothing else:

```python
import time
import redis

r = redis.Redis()

def naive_locked_write(resource_id: str, new_value: str, ttl_seconds: int = 10):
    """The UNSAFE pattern: acquire a TTL lock, do work, write. No fencing."""
    lock_key = f"lock:{resource_id}"
    # Acquire a lock with a 10s TTL. If we crash, it auto-expires.
    acquired = r.set(lock_key, "me", nx=True, ex=ttl_seconds)
    if not acquired:
        raise RuntimeError("could not acquire lock")

    # --- danger zone ---
    # Do some "work". In production this could be an RPC, a query,
    # a model inference, or just an unlucky GC pause right here.
    do_some_work_that_might_pause_for_15_seconds()
    # By now our 10s TTL may have EXPIRED. Another client may hold the
    # lock. But we have no idea. We were frozen.

    # We write anyway, because we still *believe* we hold the lock.
    storage_write(resource_id, new_value)   # <-- corrupts data
    r.delete(lock_key)                       # <-- may delete SOMEONE ELSE's lock
```

Every line of that function is reasonable on its own. The bug is structural: the TTL lock protects against the leader *crashing* (the lock auto-expires, so a dead holder does not deadlock the resource forever), but it does nothing about the leader *pausing*. A pause longer than the TTL silently invalidates the lock, and the paused process has no way to find out before it writes. The `r.delete(lock_key)` at the end is the cherry on top: if another client acquired the lock during our pause, we just deleted *their* lock, and now a third client can come in too. This is not a hypothetical; HBase had exactly this class of bug, where a paused RegionServer would resume and write to a region that had already been reassigned to a healthy server.

The TTL gives you a false sense of safety because it handles the *easy* failure (crash) and ignores the *hard* one (pause). And you cannot fix it by making the TTL longer — a longer TTL means a dead holder blocks the resource longer, hurting availability — or by making it shorter — a shorter TTL means more frequent false expirations under normal load. There is no TTL value that makes this safe, because the problem is not the TTL. The problem is that the *storage write at the end trusts the writer's word about whether it still holds the lock.* The fix is to stop trusting the writer.

## Fencing tokens: make storage reject the stale writer

**Senior rule of thumb: a lock is only safe if the protected resource itself checks the token — fencing that the writer enforces is not fencing at all.** This is the single most important idea in the entire article, and it is the canonical Kleppmann fix. Here is the mechanism, and then we will build it.

![Fencing token timeline: the lock service issues monotonically increasing tokens, and storage rejects any write whose token is below the highest it has seen](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-3.webp)

The lock service is augmented with one new responsibility: **every time it grants the lock, it returns a token that is strictly greater than every token it has ever returned.** A monotonically increasing number. The first grant returns 33, the next returns 34, the next 35, and so on, forever increasing. Kleppmann's wording: "Each time a lock server grants a lock or a lease, it also generates a fencing token (a number that increases every time a lock is granted)."

Then the contract changes on the *write* side. Every write to the protected resource must carry the token the writer holds. And the storage server — not the client, the *storage server* — remembers the highest token it has ever accepted, and **rejects any write whose token is lower than that.** Trace the timeline:

- **t1.** Client 1 acquires the lease and gets token **33**.
- **t2.** Client 1 enters a long GC pause; meanwhile its lease expires.
- **t3.** Client 2 acquires the lease and gets token **34** (the number always increases).
- **t4.** Client 2 sends its write to storage with token **34**. Storage has seen nothing higher; it accepts, and records 34 as the high-water mark.
- **t5.** Client 1 wakes up, still believing it is the leader, and sends its write to storage with its stale token **33**.
- **t6.** Storage has already processed a write with token 34. It sees an incoming token of 33, which is *lower*, and **rejects the request.** Client 1 is fenced — locked out of the resource — even though it never learned it lost the lease.

In Kleppmann's exact framing: "the storage server remembers that it has already processed a write with a higher token number (34), and so it rejects the request with token 33." The deposed leader does not need to know it was deposed. It does not need to receive a "you lost" message. It does not need a working network path to the lock service. It is fenced *by the resource it is trying to corrupt*, on the basis of a number it is carrying that is now in the past. This is the property that survives partitions, pauses, and clock skew: the storage server's high-water mark is monotonic and local, and no amount of confusion on the client side can lower it.

The load-bearing requirement, the one teams skip and then get burned by, is that **the storage server must take an active role.** Kleppmann is blunt: "this requires the storage server to take an active role in checking tokens, and rejecting any writes on which the token has gone backwards." A fencing token that the client carries but the server ignores is decoration. If your "fenced" write path looks like `if my_token < server_token: refuse()` evaluated *on the client*, you have built nothing, because the whole premise is that the client cannot be trusted to evaluate that correctly — it is the one with the stale belief.

Here is the write path done correctly. The token check lives in the storage layer, atomically, as part of the write itself:

```python
import psycopg2

# The shared resource is a Postgres row. We add a `fence_token` column and
# enforce monotonicity IN THE DATABASE, as part of the same statement that
# performs the write. The client cannot bypass this.

DDL = """
CREATE TABLE IF NOT EXISTS account (
    id           BIGINT PRIMARY KEY,
    balance      BIGINT NOT NULL,
    fence_token  BIGINT NOT NULL DEFAULT 0   -- highest token seen for this row
);
"""

def fenced_write(conn, account_id: int, new_balance: int, my_token: int) -> bool:
    """
    Apply the write ONLY IF my_token >= the highest token this row has seen.
    The comparison and the update happen in one atomic SQL statement, so a
    concurrent higher-token write cannot interleave between check and apply.
    Returns True if the write landed, False if we were fenced.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE account
               SET balance     = %s,
                   fence_token = %s
             WHERE id = %s
               AND fence_token <= %s      -- reject stale writers
            """,
            (new_balance, my_token, account_id, my_token),
        )
        applied = cur.rowcount == 1
    conn.commit()
    return applied        # False  => fenced (a higher token already wrote)
```

The entire safety of the system is in the `AND fence_token <= %s` clause and the fact that it runs inside the database's own concurrency control. The old leader (token 33) calls this after the new leader (token 34) has already bumped the row's `fence_token` to 34; the `WHERE` clause matches zero rows; `rowcount` is 0; the function returns `False`; the corruption never happens. The new leader does not need to coordinate with the old leader at all. It does not need to know the old leader exists. It just writes with a higher number, and monotonicity does the rest.

A few practical notes from having shipped this pattern:

- **Where does the monotonic token come from?** From something that already provides ordered, fault-tolerant counters: a consensus system. ZooKeeper's `zxid` (transaction id) is monotonic and is the classic source. etcd's **revision number** is monotonic cluster-wide and is explicitly recommended as a fencing token — the Jepsen analysis of etcd states plainly that "users can use the revision of their lock key as a globally ordered fencing token." A Postgres sequence works if the sequence itself is HA. What does *not* work is a per-node counter or a random UUID — see the Redlock section below for why.
- **Per-resource vs global tokens.** The high-water mark can be per-row (as above), per-table, or global. Per-resource is more precise — a stale write to account A does not block a legitimate fresh write to account B — but requires the token source to be ordered relative to each resource. A single global monotonic token is simpler and is what most lock services hand you.
- **The token must travel with the write, end to end.** If your write goes through three hops, every hop must forward the token unchanged, and the *final* sink must do the check. A token that gets dropped at the API gateway and not propagated to the storage engine fences nothing.

| Approach | Survives crash? | Survives pause / partition? | Where enforced | Verdict |
| --- | --- | --- | --- | --- |
| Plain mutex (no TTL) | No (deadlocks) | No | Client | Unusable in distributed setting |
| TTL lock (Redis SET NX EX) | Yes (auto-expires) | **No** (pause > TTL corrupts) | Client | False safety; the naive trap |
| TTL lock + fencing token | Yes | **Yes** | **Storage** | The correct pattern |
| Fencing token, checked on client | Yes | No | Client | Worthless — client has the stale belief |

## Why Redlock without fencing is unsafe

**Senior rule of thumb: if your distributed lock cannot hand you a monotonically increasing token, it cannot make a write safe, no matter how many nodes it runs on.** This is the crux of the famous 2016 debate between Martin Kleppmann and Salvatore Sanfilippo (antirez, the creator of Redis) over the Redlock algorithm, and it is worth understanding precisely because it crystallizes what fencing is *for*.

Redlock is a distributed lock built on a cluster of independent Redis nodes: a client tries to acquire the same lock on a majority of, say, five Redis instances within a time bound, and if it succeeds on three of five it considers the lock held. The appeal is that no single Redis node is a point of failure. Kleppmann's critique, in his essay "How to do distributed locking," is not primarily about the multi-node acquisition — it is that **Redlock generates no fencing token.** In his words: "Redlock does not have any facility for generating fencing tokens. The algorithm does not produce any number that is guaranteed to increase every time a client acquires a lock."

Walk the GC-pause scenario through Redlock and you see why that is fatal. Kleppmann's sequence: "Client 1 requests lock on nodes A, B, C, D, E. While the responses to client 1 are in flight, client 1 goes into stop-the-world GC. Locks expire on all Redis nodes. Client 2 acquires lock on nodes A, B, C, D, E. Client 1 finishes GC, and receives the responses from Redis nodes indicating that it successfully acquired the lock (they were held in client 1's kernel network buffers while the process was paused). Clients 1 and 2 now both believe they hold the lock." Five Redis nodes did not help. The majority acquisition succeeded; the pause happened *after* acquisition and *before* the write; and now two clients are about to write. The only thing that could save you here is a token check at the storage layer — and Redlock gives you no token to check.

Could you bolt a token onto Redlock? Kleppmann argues you essentially cannot, not cheaply: "It's not obvious how one would change the Redlock algorithm to start generating fencing tokens. The unique random value it uses does not provide the required monotonicity. Keeping a counter on one Redis node would not be sufficient, because that node may fail. Keeping counters on several nodes would mean they would go out of sync. It's likely that you would need a consensus algorithm just to generate the fencing tokens." That last sentence is the punchline of the whole debate: *to generate a safe fencing token you need consensus, and if you have consensus you did not need Redlock.* That is why the canonical token sources are ZooKeeper and etcd — they *are* consensus systems, so the monotonic counter falls out for free.

Antirez's [rebuttal](https://antirez.com/news/101) is worth steelmanning, because he is not wrong about everything. His argument is that for many real workloads the random lock value is sufficient as a *uniqueness check*, and that the order of token issuance need not match the order of operations as long as access is mutually exclusive. He is correct for a large class of best-effort use cases: deduplicating a cron job, preventing a cache stampede, ensuring roughly-once execution of an idempotent task. If a double-run is merely wasteful rather than corrupting, Redlock is fine, and the fencing-token argument is overkill. The two positions resolve cleanly along the line of *what a double-execution costs*:

| You are using a lock to... | Cost of two holders acting | Need a fencing token? |
| --- | --- | --- |
| Avoid duplicate cron runs | Wasted work | No — Redlock-class lock is fine |
| Prevent cache stampede | Brief extra load | No |
| Serialize idempotent jobs | None (idempotent) | No |
| Mutate shared storage / money | **Corruption** | **Yes — and storage must check it** |
| Promote a database leader | **Split-brain** | **Yes — epoch as token** |

Kleppmann's own summary is the rule to carry: "the fact that Redlock fails to generate fencing tokens should be sufficient reason not to use it in situations where correctness depends on the lock." If correctness depends on the lock, you need a token, and the token needs to be checked by the resource. Everything else is a comfort blanket.

## Leases: time-bounded leadership with a clock-skew margin

**Senior rule of thumb: a lease holder must stop acting before the lease expires by a margin at least as large as the maximum clock skew plus message delay — and it must use a monotonic clock to measure that, never wall time.** Fencing tokens stop a stale write at the resource. Leases attack the same problem from the *time* side: they bound how long a leader is allowed to believe it is the leader, so that the window of dangerous overlap is small and controllable. The two compose — leases shrink the overlap window; fencing tokens make the residual overlap harmless.

A **lease** is a lock with a time limit. The issuer (a consensus service, a lock manager) grants leadership for, say, ten seconds. The holder must *renew* before the lease expires, or its leadership lapses automatically. The beauty is that no failure-detection round trip is needed to revoke a dead leader: if the holder crashes or partitions away, it simply fails to renew, and the lease expires on its own. The danger is that the holder and the issuer have *different clocks*, and "ten seconds" means different things to each of them.

![Lease timeline: the holder must self-expire at TTL minus a margin covering clock skew and round-trip delay, or it acts after the issuer has already regranted the lease](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-4.webp)

Walk the timeline. At **T0** the issuer grants a lease with TTL = 10s, measured on the issuer's clock `C_issuer`. The grant takes half a second to arrive, so the holder starts its own countdown at **T0+0.5s** on its clock `C_holder`. Already the two clocks disagree about when the lease ends: the issuer thinks T0+10s, the holder — if it naively counts ten seconds from when *it* received the grant — thinks T0+10.5s. If the holder also runs slightly slow, or its clock has drifted, the gap widens. The safe holder therefore **self-expires early**, at **T0+7s** in the figure: it stops acting at `TTL − margin`, where the margin covers the worst-case clock skew between the two machines plus the round-trip delay. The issuer's own expiry at **T0+10s** then sits comfortably *after* the holder has already stepped down, so there is never a moment where the issuer considers the lease free while the holder still considers itself the leader.

The bottom-right danger box is what happens with **no margin**: the holder keeps acting right up to 9.9s on its own clock, but because of skew the issuer already expired the lease at 9.5s of holder-time and *regranted it to someone else.* For 0.4 seconds, two nodes both believe they hold the lease. That is a split-brain window, and a 0.4-second window is more than enough to corrupt a row. The margin is what closes it.

Two clock subtleties make or break this in practice, and both connect to [time, clocks, and ordering in distributed systems](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems):

1. **Measure elapsed time with a monotonic clock, never wall-clock time.** Wall-clock time (`CLOCK_REALTIME`, `time.time()`) can jump — NTP steps it, a VM resumes with a different value, an operator sets it by hand. If you measure your lease countdown against wall time and the clock jumps backward by five seconds, you just gave yourself five extra seconds of phantom leadership. Use `CLOCK_MONOTONIC` / `time.monotonic()`, which only ever moves forward and is immune to NTP steps. (Antirez, in his rebuttal, conceded exactly this point — that Redis and Redlock implementations "should switch to the monotonic time API provided by most operating systems.")
2. **The margin must bound the actual worst-case skew, and you cannot assume it is small.** "Clocks on well-run servers are within a few milliseconds" is true on average and false at the tail. A misconfigured NTP, a leap-second mishap, or a VM migration can produce skew of seconds. If your margin is 50ms and your tail skew is 2s, your margin is fiction.

Here is a lease holder that does both correctly:

```python
import time

class Lease:
    """
    A leadership lease the holder self-expires EARLY, using a monotonic clock,
    by a safety margin that covers max clock skew + renewal round-trip.
    """
    def __init__(self, ttl: float, max_skew: float, rtt: float):
        self.ttl = ttl                       # issuer-granted TTL, seconds
        self.margin = max_skew + rtt         # how early WE must stop acting
        self._granted_at = None              # monotonic timestamp of grant
        self.token = None                    # fencing token from the issuer

    def on_granted(self, token: int):
        # monotonic(): immune to NTP steps and wall-clock jumps.
        self._granted_at = time.monotonic()
        self.token = token

    def is_safely_held(self) -> bool:
        if self._granted_at is None:
            return False
        elapsed = time.monotonic() - self._granted_at
        # Stop acting at (ttl - margin), NOT at ttl. The margin is the gap
        # that keeps us clear of the issuer's view of expiry under worst-case
        # skew. If elapsed is past that point, we must treat ourselves as
        # NO LONGER the leader, even though the wall clock says we have time.
        return elapsed < (self.ttl - self.margin)

    def guarded_write(self, do_write):
        if not self.is_safely_held():
            raise RuntimeError("lease unsafe to use: self-expired by margin")
        # Even here, do_write MUST carry self.token so storage can fence us
        # if a renewal race still slipped a higher token past us.
        return do_write(self.token)
```

Note the last comment: the lease check and the fencing token are belt *and* suspenders. The lease keeps the dangerous window tiny; the token makes even that tiny window non-corrupting. A serious system uses both. The lease is an optimization (it lets a healthy leader act without consulting the resource on every write); the token is the correctness guarantee (it catches the rare case where the lease check was fooled).

### Second-order optimization: renew at one-third of the TTL

A subtle operational gotcha: do not renew the lease at the last moment. If your TTL is 10s and you renew at 9.5s, a single dropped renewal packet plus one retry can push you past expiry and cause an unnecessary leadership flap. Renew at roughly **one-third of the TTL** — for a 10s lease, renew every ~3s. This gives you two or three renewal attempts before expiry, so a transient packet loss does not cost you leadership. The cost is more renewal traffic; the benefit is dramatically fewer spurious failovers, and every spurious failover is a roll of the split-brain dice. This is the same logic ZooKeeper and etcd clients use internally for session keepalives.

## STONITH: shoot the other node in the head

**Senior rule of thumb: when you cannot make the resource check a token — because the resource is a raw shared disk, a virtual IP, or a filesystem mount — you fence at the infrastructure layer by physically removing the suspect node's ability to do anything.** Fencing tokens are a software contract: they require the storage layer to be modifiable to check a token. But a great deal of high-availability infrastructure predates that idea or operates below it — shared-storage clusters where two nodes can both mount the same block device, clusters that float a virtual IP between nodes, databases on a SAN. For these, the canonical fencing mechanism is the bluntly named **STONITH**: "Shoot The Other Node In The Head." If you suspect a node has failed, you do not ask it nicely to step down — you *forcibly disable it* (power it off, cut its network, revoke its storage path) before you let the survivor take over.

![STONITH graph: Pacemaker fences the suspect node via IPMI power-cut or SBD self-fence before letting the survivor mount shared storage](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-6.webp)

The figure shows the standard Linux HA stack (Pacemaker + Corosync) doing this. **Corosync** is the messaging and membership layer: it is what notices "I can no longer reach my peer, and I no longer have quorum." **Pacemaker** is the resource manager that decides what to do, and its iron rule is *fence before failover.* It will not promote the survivor or mount the shared storage until it has **confirmed** the suspect node is dead. Confirmation comes through one of two fencing paths:

- **Power fencing (IPMI / PDU).** Pacemaker tells the suspect node's out-of-band management controller (IPMI, iLO, DRAC) or a network power distribution unit (PDU) to *cut the power.* The node is hard-off. It cannot write to anything, because it is not running. This is the most reliable form of fencing because it requires no cooperation from the (possibly wedged) node's OS — you are pulling the plug from outside.
- **Self-fencing via SBD watchdog.** SBD ("Storage-Based Death," sometimes "STONITH Block Device") uses a small shared disk partition and a hardware watchdog timer. Each node periodically writes a heartbeat to its slot on the shared disk and pets its watchdog. If a node loses access to the shared disk, or reads a fencing request addressed to it in its slot, it *fences itself* — the hardware watchdog reboots the machine. This handles the case where you cannot reach the node's power controller (because the network is down) but the node can still see the shared disk: it commits suicide so the survivor can safely proceed.

The SUSE knowledge base on preventing a "fence race" makes the stakes explicit: "the only way to prevent split-brains is with fencing, full stop," and a cluster without a configured fencing device "will most likely lead to data corruption" in a split-brain scenario, requiring manual intervention. STONITH exists precisely because, on shared storage, there is no token to check — two mounts of the same `ext4` filesystem will silently corrupt it — so the only safe move is to guarantee one of the two writers literally cannot run.

A configured STONITH device in Pacemaker looks like this; note that the resource agent talks to the *peer's* management interface, not its own:

```bash
# Configure IPMI power fencing for node "db-east" via its BMC.
# When Pacemaker decides db-east must die, it calls this agent, which
# powers the machine off through its out-of-band management controller.
pcs stonith create fence-db-east fence_ipmilan \
    pcmk_host_list="db-east" \
    ip="10.0.5.21" \           # db-east's BMC address (NOT its data NIC)
    username="fenceuser" \
    password="..." \
    lanplus="1" \
    power_timeout="20"

# Make Pacemaker refuse to run resources without successful fencing.
pcs property set stonith-enabled=true
# When quorum is lost, suicide rather than risk a second writer.
pcs property set no-quorum-policy=suicide
```

The two `property set` lines are the policy that turns this from theater into safety. `stonith-enabled=true` means Pacemaker will *not* fail a resource over until fencing of the old owner has succeeded — if fencing fails (the BMC is unreachable), it would rather block than risk two writers. `no-quorum-policy=suicide` means a partition that finds itself without quorum will stop itself rather than keep serving. Both encode the same priority: *correctness over availability.* A node that cannot prove it is allowed to act, stops acting.

### The fence race, and why two-node STONITH needs a delay

A vicious failure mode in symmetric two-node clusters is the **fence race**: the partition happens, each node decides the *other* one is dead, and both simultaneously fire their fencing device. With power fencing, both nodes power each other off at the same instant — and now the whole cluster is dark. The standard mitigation is `pcmk_delay_base` / `pcmk_delay_max`: give one node (or a random delay) a head start so that one fencing action lands before the other, leaving exactly one survivor. This is a band-aid over the deeper problem that two nodes cannot form a majority, which is the subject of the next section. STONITH is necessary, but it is not sufficient on its own with an even node count — you also need quorum to decide *who* is entitled to do the shooting.

## Quorum: only one partition can hold a majority

**Senior rule of thumb: to guarantee that two partitions can never both elect a leader, require a strict majority of voters to elect one — because two disjoint sets cannot both be a majority of the same whole.** Fencing tokens, leases, and STONITH all answer "how do we stop the loser from writing." Quorum answers the prior question: "how do we make sure only one side even *thinks* it won?" The mechanism is arithmetic, and the arithmetic is airtight.

![Quorum grid: with five voters a partition splits them three versus two, so only the side holding three forms a majority and may elect a leader](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-5.webp)

Suppose you have five voters. A network partition can split them in any way, but it always splits them into two groups whose sizes add up to five: 4+1, 3+2, 5+0. In every case, **at most one group has three or more** — a strict majority of five. Require that a leader can only be elected with the votes of a majority, and the conclusion is forced: at most one partition can elect a leader. The figure shows the 3+2 split. Partition A holds nodes 1, 2, 3 — three of five votes, a majority — so it forms quorum and elects a leader. Partition B holds nodes 4 and 5 — two of five, *not* a majority — so it cannot elect anyone and must step down to read-only (or stop entirely). The minority side knows it is the minority because it cannot reach enough peers to gather a majority of votes, and a correctly built node in the minority *refuses to act as leader.*

This is the formal version of Kleppmann's "knowledge, truth and lies" argument from Chapter 8. A node cannot trust its own judgment of whether it is the leader, because its judgment is based on what it can see, and what it can see is distorted by partitions and pauses. The resolution is to define truth by majority vote: "the truth is defined by the majority." A decision (who is leader, whether a node is dead, whether a write committed) requires a minimum number of votes from several nodes, precisely because **only one set can be a majority at a time.** As Kleppmann puts it, "a node cannot necessarily trust its own judgment of a situation" — so the system relies on a quorum, where decisions require a minimum number of votes, because that is the one thing a partitioned minority cannot manufacture on its own.

The quorum condition for electing a leader (or committing a write) is the familiar:

$$
\text{votes received} > \left\lfloor \frac{N}{2} \right\rfloor
$$

For $N = 5$, you need more than 2, i.e. at least 3. For $N = 3$, at least 2. The reason this prevents two leaders is that two majorities of the same set must overlap — if set A has $> N/2$ members and set B has $> N/2$ members, they share at least one member, and that shared member cannot vote for two different leaders in the same election. Overlap is the whole trick. It is the same overlap that makes quorum reads and writes consistent in [leaderless replication](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive), and it is the foundation under [Raft](/blog/software-development/database/raft-consensus-from-scratch) and Paxos.

Here is a minimal leader-election guard built on majority voting. Note that the node refuses to consider itself leader until it has heard from a strict majority:

```python
class Election:
    def __init__(self, all_nodes: list[str], me: str):
        self.all_nodes = set(all_nodes)
        self.me = me
        self.votes = {me}                      # vote for self
        self.term = 0

    @property
    def quorum_size(self) -> int:
        # Strict majority: more than half of the TOTAL cluster, not of
        # the nodes we can currently reach. This is the key: we count
        # against N, so a minority partition can never reach it.
        return len(self.all_nodes) // 2 + 1

    def receive_vote(self, voter: str):
        self.votes.add(voter)

    def is_leader(self) -> bool:
        # We are leader ONLY if a strict majority of the whole cluster
        # voted for us this term. A 2-of-5 partition can never get here.
        return len(self.votes) >= self.quorum_size
```

The single most important line is `len(self.all_nodes) // 2 + 1`: the quorum is computed against the *total* cluster size $N$, not against the number of nodes currently reachable. A naive implementation that computes "majority of the nodes I can see" defeats the entire mechanism — the 2-node minority would see 2 nodes, decide "2 of 2 is a majority," and elect itself, producing exactly the split-brain quorum was meant to prevent. The arithmetic only protects you if it counts against the fixed denominator.

| Cluster size $N$ | Quorum (votes to lead) | Failures tolerated | Can split-brain? |
| --- | --- | --- | --- |
| 1 | 1 | 0 | N/A (single node) |
| 2 | 2 | 0 | **Yes if "majority of reachable" — must be 2-of-2** |
| 3 | 2 | 1 | No |
| 4 | 3 | 1 | No (but no better than 3, costs more) |
| 5 | 3 | 2 | No |
| 7 | 4 | 3 | No |

Look hard at the rows for 3 versus 4, and 5 versus 6. **Adding an even-numbered node buys you nothing in fault tolerance** — a 4-node cluster tolerates the same single failure as a 3-node cluster but needs a larger quorum, so it is strictly worse. This is why the next section exists: you want an *odd* number of voters, and when your data nodes are naturally even, you add a cheap tiebreaker rather than a second data node.

## Odd node counts and the witness tiebreaker

**Senior rule of thumb: always run an odd number of voters; when your data nodes are even (most commonly two), add a lightweight witness that votes but holds no data, rather than a third full replica.** The quorum math above has a sharp implication: an even number of voters is a trap, because a clean half-and-half split leaves *neither* side with a majority. A two-node cluster is the worst case — a partition gives each node exactly one vote, 1-versus-1, and neither can form a quorum. The cluster deadlocks (if it is safe) or split-brains (if it is naive). Two nodes is simultaneously the most common HA topology people reach for and the most dangerous.

![Witness grid: a bare two-node pair deadlocks on a partition, while adding a lightweight witness makes three voters so one side always wins a majority](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-7.webp)

The left panel of the figure is the two-node trap: on a partition you get 1 versus 1, and there is no majority, so either nobody can lead (availability loss) or both lead (correctness loss). The fix is on the right: introduce a **witness** — a third voter that participates in elections but stores no data. Now you have three voters. On a partition, the data node that can still reach the witness forms a 2-of-3 majority and leads; the data node that cannot reach the witness is in a 1-of-3 minority and steps down. One side always wins cleanly, and it is never both. The witness is the cheapest possible tiebreaker: it does not need the storage, CPU, or bandwidth of a full replica, because it never serves data — it only votes. This pattern has many names across the industry, all the same idea:

- **MongoDB arbiter.** A replica-set member that votes in elections but holds no data, used "to act as a tiebreaker in a replica set with an even number of data-bearing nodes, ensuring that a primary can always be elected." MongoDB's operator will even auto-convert a node to non-voting to keep the voting count odd.
- **Windows Failover Cluster / SQL Server Always On witness.** A *disk witness* or *file-share witness* that "acts as a tiebreaker to ensure there's always a majority vote" when the node count is even. Microsoft's own guidance: with an even number of voting nodes, add a witness.
- **Pacemaker `qdevice` / quorum disk.** A third arbitration vote for two-node Linux clusters, replacing the fragile two-node hacks (`two_node: 1`, fence delays) with a real majority.
- **Cloud zonal arbiter.** A tiny VM in a *third* availability zone whose only job is to break ties between the two data-bearing zones, so a single-zone failure leaves a clear majority in the surviving zone plus the arbiter.

The placement of the witness matters as much as its existence. The classic mistake is to put the witness in the *same* failure domain as one of the data nodes — same rack, same AZ, same power feed. Then a failure of that domain takes out both a data node *and* the witness, leaving the surviving data node alone at 1-of-3, unable to form a majority, and your "highly available" pair is down because of a single-domain failure. The witness must live in an *independent* failure domain: a third rack, a third AZ, a third region for the most paranoid. The point of an odd voter count is to break ties, and it only does that if the tiebreaker fails independently of the things it is breaking ties between.

```yaml
# A two-data-node Postgres HA pair with a third-AZ witness, expressed as
# a Patroni / etcd topology. etcd is the DCS (distributed config store)
# that provides the leader lease AND the fencing/quorum. The witness etcd
# member holds NO Postgres data — it only votes.
etcd_members:
  - { name: db-az-a, zone: us-east-1a }     # data node + etcd voter
  - { name: db-az-b, zone: us-east-1b }     # data node + etcd voter
  - { name: witness, zone: us-east-1c }     # etcd voter ONLY (tiebreaker)

patroni:
  ttl: 30                  # leader lease TTL (seconds)
  loop_wait: 10            # renew roughly every TTL/3
  retry_timeout: 10
  # A primary that cannot renew its lease in etcd within `ttl` demotes
  # itself to read-only. etcd's Raft quorum (2 of 3) decides the leader;
  # the lone-partitioned node, being a minority, can never win.
```

With this topology, a partition that isolates `db-az-a` from the other two leaves `db-az-b` and `witness` as a 2-of-3 etcd majority; they elect `db-az-b` as the new Postgres primary, and `db-az-a`, unable to reach etcd's majority, fails to renew its lease and demotes itself to read-only. Two-of-three majority, lease self-demotion, and (if you also add a `fence_token` column as shown earlier) fencing at the storage layer — that is defense in depth.

### Second-order optimization: an even number of voters is sometimes acceptable with a deterministic tiebreak

There is a narrow exception worth knowing. Some systems run an even number of voters and break ties with a *deterministic, pre-agreed rule* — for example, "on a tie, the partition containing the node with the lowest ID wins." This can be safe *if and only if* the tiebreak is evaluated identically by everyone and the losing side reliably fences itself. It is fragile because it depends on every node agreeing on the rule and on the membership, and a stale membership view can make two sides each believe they hold the "lowest ID." Prefer an odd voter count with a witness; reach for a deterministic even-tiebreak only when you genuinely cannot add a third voter and you can guarantee the losing side STONITHs itself.

## Consensus epochs are fencing tokens: what Raft does and does not solve

**Senior rule of thumb: Raft and Paxos guarantee at most one leader per term, but a partitioned leader from an *older* term must still be fenced at the storage or lease layer — and the term number is exactly the fencing token to use.** This is the section that ties consensus back to everything above, and it dissolves a common misconception: "we use Raft, so we can't have split-brain." Raft prevents one specific split-brain (two leaders in the *same* term) and explicitly does *not* prevent another (a stale leader from a *prior* term writing to shared storage). Understanding which is which is the difference between a system that is actually safe and one that merely feels safe.

![Before-after: consensus alone lets an old-term leader resume writing, but treating the term number as a fencing token lets storage reject it](/imgs/blogs/split-brain-and-fencing-in-distributed-databases-8.webp)

Consensus algorithms organize time into numbered, monotonically increasing periods — Raft calls them **terms**, Paxos calls them **ballot numbers** or **proposal numbers**, ZooKeeper calls them **epochs**. The guarantee, as Kleppmann states it in Chapter 9, is that each of these protocols "guarantees a single leader for each epoch" — a number that "is monotonic and ever-increasing." But — and this is the load-bearing caveat — the protocols do *not* guarantee global leader uniqueness across all time. They guarantee uniqueness *within* a term: "all consensus protocols use a leader internally, but don't guarantee the leader uniqueness; instead a weaker guarantee is to only guarantee the leader uniqueness within each epoch." There can be a leader for term 7 and, later, a different leader for term 8, and — crucially — the term-7 leader might have been partitioned and not yet learned that term 8 exists.

The left side of the figure is the gap. Raft elects exactly one leader for term 7. That term-7 leader gets partitioned and pauses. Meanwhile the majority partition, unable to reach it, runs a new election and elects a term-8 leader (the term number increments because it is monotonic). Now the term-7 leader resumes. *Within the consensus protocol itself*, it will discover it is stale the moment it tries to replicate a log entry — its `AppendEntries` will carry term 7, a follower will respond "my term is 8," and the stale leader steps down. Raft handles that path correctly. But here is the hole: **if the leader writes to an external resource that is not part of the Raft group — a shared disk, an object store, a downstream database, a side-effecting RPC — it can perform that write before its next `AppendEntries` reveals the higher term.** The consensus protocol fenced the leader *inside the log*, but the external write already escaped.

The right side of the figure is the fix, and it is the same fix as the whole article: **make the term number a fencing token.** Every write the leader sends to the external resource carries its term. The resource remembers the highest term it has seen and rejects writes carrying a lower term. The term-7 leader's escaped write arrives at storage stamped "term 7"; storage has already accepted a "term 8" write; it rejects the term-7 write; the old leader is fenced. The epoch *is* the monotonically increasing token. This is not a coincidence — it is why consensus systems make such good token sources. ZooKeeper's `zxid` and etcd's revision are the externalized form of exactly this monotonic epoch counter, which is why "use the etcd revision as your fencing token" works: you are reusing the consensus epoch as the fence.

```python
# A Raft (or etcd-backed) leader writing to an EXTERNAL resource.
# Inside the Raft group, term mismatch is handled by the protocol.
# Outside it, WE must carry the term so the resource can fence us.

def leader_external_write(raft, storage, key: str, value: bytes) -> bool:
    """
    Write to an external store, fenced by the Raft term. Returns True if
    the write landed, False if a higher-term leader already wrote (we are
    a stale, partitioned old-term leader and just got fenced).
    """
    term = raft.current_term            # the consensus epoch == fencing token
    if not raft.is_leader_locally():    # cheap local check (NOT sufficient alone)
        return False
    # The storage layer enforces: accept iff term >= highest term it has seen.
    # This catches the case where we BELIEVE we are leader (stale term 7) but
    # a term-8 leader already exists and has written.
    return storage.compare_and_write(key, value, fence_term=term)
```

The comment on `is_leader_locally()` is the crux: the local leadership check is an optimization to avoid pointless writes, but it is *not* the safety mechanism, because a stale leader's local check passes (it still thinks it is leader for term 7). Safety lives in `storage.compare_and_write` enforcing `term >= highest_seen_term`. Consensus narrows the window and provides the token; the resource enforces the token. Neither alone is enough for external writes.

| Layer | What it guarantees | What it does NOT guarantee |
| --- | --- | --- |
| Raft / Paxos term | One leader per term; stale leader detected on next log replication | A stale leader's *external* (off-log) write is blocked |
| Lease | Bounded leadership window; auto-revoke on crash/partition | Safety if clock skew exceeds the margin |
| Fencing token at storage | A lower-token write is rejected by the resource | Anything if the resource doesn't check the token |
| STONITH | The suspect node physically cannot run | Correct *choice* of who to shoot (needs quorum) |
| Quorum / witness | At most one partition elects a leader | The loser doesn't keep writing (needs fencing/STONITH) |

The table is the whole article in one frame: **no single layer is sufficient.** Each closes a different gap, and a production system layers them — quorum decides who leads, a lease bounds the window, the epoch is the token, the storage checks the token, and STONITH handles the resources that cannot check tokens. Defense in depth is not paranoia here; it is the recognition that every individual mechanism has a documented failure mode that another mechanism covers.

## Putting it together: the make-failover-safe playbook

**Senior rule of thumb: a safe failover is one where you can lose the entire failure-detection decision — promote the wrong node, promote too early, promote during a partition — and still not corrupt data.** That is the design target. You do not aim for a perfect failure detector (impossible); you aim for a failover that is *robust to a wrong detector.* Here is the checklist I run through on any HA design, in priority order.

1. **Use an odd number of voters; add a witness if your data nodes are even.** Two data nodes plus a third-AZ witness is the minimum viable HA topology. Never two bare nodes. Never an even voter count without a deterministic, self-fencing tiebreak.
2. **Elect leaders by strict majority of the total cluster, not of reachable nodes.** Compute quorum against the fixed $N$. A minority partition must be unable to elect itself by construction.
3. **Make leadership a lease, renewed at ~TTL/3, measured on a monotonic clock.** The leader self-demotes if it cannot renew. This auto-revokes a crashed or partitioned leader with no detection round trip, and it bounds the dangerous overlap window.
4. **Fence at the resource with a monotonically increasing token, and make the resource check it.** The token comes from the consensus layer (Raft term, etcd revision, ZooKeeper zxid). Every write carries it; storage rejects lower tokens. This is the guarantee that survives when steps 1–3 are fooled.
5. **For resources that cannot check a token (shared disk, virtual IP, mounts), use STONITH and fence before failover.** Confirm the old owner is dead — power off via IPMI, or self-fence via SBD watchdog — before the survivor takes the resource. Set `stonith-enabled=true` and `no-quorum-policy=suicide`.
6. **Default to correctness over availability on quorum loss.** A node that cannot prove it is allowed to act must stop acting. `no-quorum-policy=suicide`, demote-to-read-only, refuse writes — pick the one that fits, but never "keep serving and hope."
7. **Test the partition, not just the crash.** Crashes are easy and your system probably handles them. Inject partitions, GC pauses (`kill -STOP` / `kill -CONT`), and clock skew, and assert that the minority side fences itself and the resource rejects stale tokens. A system that has never been partition-tested has never had its split-brain defenses tested.

> The deepest lesson, straight from Chapter 8: a node cannot trust its own judgment of whether it is the leader. So never build the safety check into the node. Build it into the thing the node is trying to change. The node may be confused; the resource must not be.

## Case studies from production

### 1. GitHub's 43-second partition, October 21, 2018

Routine maintenance to replace failing 100G optical equipment "resulted in the loss of connectivity between our US East Coast network hub and our primary US East Coast data center" for 43 seconds. GitHub's Orchestrator (running its own Raft consensus for the *orchestration* layer) saw the East Coast primary go silent and did what it was built to do: the West Coast and public-cloud Orchestrator nodes "established a quorum and started failing over clusters to direct writes to the US West Coast data center." When the link healed, applications "immediately began directing write traffic to the new primaries in the West Coast site." The East Coast databases held "a brief period of writes that had not been replicated," while West Coast accumulated writes for roughly 40 minutes. The result: "because the database clusters in both data centers now contained writes that were not present in the other data center, we were unable to fail the primary back over to the US East Coast data center safely." Total degradation: **24 hours and 11 minutes.** The root lesson is the one this whole article turns on — the East Coast primary was never dead; it was unreachable, and the cross-region promotion created two authoritative write streams that diverged. The fix space is exactly steps 1–6: a tighter quorum/witness placement, longer detection thresholds before a cross-region promotion, and fencing so that one side's writes could not have been accepted in parallel.

### 2. HBase, the paused RegionServer, and the reassigned region

HBase historically suffered the textbook fencing bug, which is why Kleppmann cites it directly: "HBase used to have this problem." A RegionServer holds regions; its liveness is a ZooKeeper ephemeral znode tied to its session. If the RegionServer pauses (long GC), its ZooKeeper session times out, the ephemeral znode evaporates, and the HBase Master declares it dead and reassigns its regions to other RegionServers, which read the WAL and recover. The danger: the paused RegionServer wakes up *still believing it owns those regions* and issues writes to the underlying HDFS files — which have since been reassigned. Without fencing, those late writes corrupt regions now owned by another server. The fix HBase adopted is fencing at the storage layer: WAL files are renamed/recovered under the new owner, and stale writers are detected and rejected, so a resurrected RegionServer's writes land on a path it no longer owns and fail. This is the fencing-token pattern applied to HDFS file ownership: the resource (HDFS) refuses the deposed owner.

### 3. The Redlock debate: a lock with no token

In 2016 Martin Kleppmann published "How to do distributed locking," arguing that Redlock — distributed locking over multiple Redis nodes — is unsafe for correctness-critical use because it generates no fencing token. Salvatore Sanfilippo (antirez) [responded](https://antirez.com/news/101) defending Redlock for best-effort use cases. The technical heart, in Kleppmann's words: "Redlock does not have any facility for generating fencing tokens... it's likely that you would need a consensus algorithm just to generate the fencing tokens." The debate never fully "resolved" because both are right about different questions: Redlock is fine for deduplicating idempotent work, and unsafe for serializing writes to shared state. The lasting industry takeaway is the discipline — if correctness depends on the lock, the lock must yield a monotonic token *and* the resource must check it. Antirez did concede one universal point: implementations should measure time with the monotonic clock API, not wall time, which is the same lesson as the lease section.

### 4. The two-node Postgres pair and the cron-job failover

A pattern I have personally debugged more than once: two Postgres servers, primary and standby, with a homegrown failover script triggered by "primary didn't answer my health check." There is no third voter, no fencing. A brief network blip between the script's host and the primary makes the script promote the standby. The primary, perfectly healthy, keeps serving its connection pool — applications on its side of the blip keep writing. Now both are primary. The two diverge within seconds, and because there is no `fence_token` and no STONITH, nothing stops either. Recovery means choosing one to keep and replaying or discarding the other's writes by hand. The fix is the playbook: add a witness (a tiny etcd or a third Postgres acting only as a voter via Patroni), promote only on quorum, lease-based leadership, and a `fence_token` column. The cron script is replaced by Patroni precisely because Patroni does all six steps and the script did zero.

### 5. The fence race that powered off the whole cluster

A symmetric two-node Pacemaker cluster with power fencing but no quorum tiebreak and no fence delay. A switch reboot partitions the two nodes. Each node concludes the other is dead and fires its IPMI fence — *simultaneously.* Both machines power off at the same instant. The cluster is now entirely dark; an outage that started as a momentary partition became a full outage requiring someone to walk to the rack. The lesson: STONITH is necessary but, with an even node count, insufficient — you need either a quorum device (`qdevice`) so only the quorate side fences, or at minimum `pcmk_delay_base` so one node wins the race. This is the concrete cost of an even voter count: the tie is resolved by mutual destruction.

### 6. Clock skew defeats a lease with too small a margin

A leader-election system used a 15-second lease with a hardcoded 200ms safety margin, reasoning that "our clocks are NTP-synced to within a few milliseconds." Then a VM live-migration paused a leader for ~3 seconds and resumed it with a clock that had also jumped. The leader's monotonic countdown was fine for the elapsed-time part, but the *grant-to-renewal* path assumed bounded skew that the migration violated; for a couple of seconds the migrated node believed it still held a lease that the issuer had already regranted. Two leaders, briefly. The single corrupting write in that window was caught only because a fencing token happened to be in place — the migrated node's write carried the old token and storage rejected it. The lesson is twofold: size the margin for *tail* skew (seconds, under VM migration / NTP step), not average skew; and keep the fencing token as the backstop for when the lease margin is wrong.

### 7. etcd short lease TTLs and false leadership churn

The Jepsen analysis of etcd notes that very short lease TTLs (1–3 seconds) "generally failed to provide mutual exclusion after only a few minutes" — not because etcd is broken, but because a TTL shorter than the realistic worst-case pause/renewal latency causes the lease to expire under a healthy-but-slow leader, triggering churn and brief overlap. The mitigation is the renew-at-TTL/3 rule plus a TTL comfortably larger than your tail GC/IO pause, *plus* using the etcd revision as a fencing token so that even a churn-induced overlap is non-corrupting: "users can use the revision of their lock key as a globally ordered fencing token." This case is a reminder that lease *tuning* is a safety parameter, not just a performance knob — too short is as dangerous as too long.

### 8. The minority partition that kept serving reads as if they were fresh

A subtler split-brain: a system correctly prevented the minority partition from *writing* (it had quorum-based leader election) but allowed it to keep *serving reads* from its stale local copy without marking them stale. Users on the minority side saw old data presented as current — not corruption of stored data, but corruption of the user's belief about reality, which for some applications (a bank balance, an inventory count, a feature flag) is just as damaging. The fix is to extend the quorum discipline to reads where freshness matters: a node that has lost quorum must either refuse reads, serve them with a staleness warning, or read through to the leader. This connects to [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — split-brain is not only a write-corruption problem; on the read side it is a linearizability violation, where the minority partition serves values that have been superseded.

### 9. STONITH disabled "temporarily" for maintenance

An operations team disabled `stonith-enabled` to do maintenance without nodes rebooting each other, and forgot to re-enable it. Weeks later a real partition occurred; with fencing disabled, Pacemaker failed resources over *without confirming the old owner was dead*, and the old owner — alive on the other side of the partition — kept its shared filesystem mounted. Two mounts of one filesystem corrupted it. The lesson is operational, not algorithmic: fencing that is "temporarily" off is fencing that is off when you need it, and the safest configuration makes the cluster refuse to operate (`stonith-enabled=true` as an enforced invariant, alerted on if changed) rather than silently run unsafe. The SUSE guidance is unambiguous: a cluster without fencing "will most likely lead to data corruption" on split-brain.

### 10. Cross-region "active-active" that was really split-brain by design

A team built "active-active multi-region" by running an independent writable primary in each of two regions with asynchronous replication between them, and no conflict resolution. This is split-brain as an architecture: both sides always accept writes, and the moment the same key is touched in both regions, they diverge with no merge. It worked in demos because traffic happened to be partitioned by region. It failed in production the first time a user's request hit both regions (a retry across a load balancer) and produced two conflicting writes. Active-active is only safe with a conflict model — CRDTs, last-write-wins with synchronized clocks, or per-key home regions — covered in [multi-leader and leaderless replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless). Two writable primaries without a conflict strategy is not high availability; it is permanent, self-inflicted split-brain.

### 11. The honest-but-wrong node and why Byzantine faults are a different problem

Worth stating to bound the scope: every mechanism here assumes nodes are *honest but unreliable.* Kleppmann's framing in Chapter 8: "any node that does respond is assumed to be telling the truth to the best of its knowledge." A fenced node carries a stale token because it genuinely believes it is the leader — it is wrong, not malicious. Fencing tokens, leases, and quorum all rely on this: a node that *lied* about its token (sent 99 when it held 33) could defeat the storage check. Defending against lying nodes is the **Byzantine** fault model, which most server-side systems deliberately do not adopt because, as the chapter notes, the cost is high and the threat model (your own data-center nodes lying) is usually not the one you face. The takeaway: fencing protects you from confusion, not from compromise. If an attacker controls a node and forges tokens, you have a security problem that fencing does not solve.

## When to reach for each mechanism, and when not to

### Reach for fencing tokens when...

- Writes hit shared, mutable state where two concurrent writers cause corruption (storage, money, inventory, a leader-owned external database).
- You already have a consensus system (etcd, ZooKeeper, Raft) that can hand you a monotonic counter — the token is nearly free.
- You can modify the resource (or its access path) to check the token atomically as part of the write.
- Correctness, not just liveness, depends on single-writer access.

### Reach for STONITH when...

- The resource is a raw shared device, a virtual IP, or a filesystem mount that *cannot* check a software token (two mounts silently corrupt).
- You run a classic HA cluster stack (Pacemaker/Corosync) over shared storage or a SAN.
- You can reach the suspect node's out-of-band power/management (IPMI, PDU) or deploy an SBD watchdog.

### Reach for quorum + an odd voter count / witness when...

- You need to *decide* who leads, not just stop the loser — i.e., always, as the layer beneath fencing and STONITH.
- Your data nodes are naturally even (the ubiquitous two-node pair) — add a third-AZ witness rather than a second full replica.
- You can place the tiebreaker in an independent failure domain.

### Skip or simplify when...

- **The lock is best-effort.** Deduplicating an idempotent cron job or preventing a cache stampede does not need fencing tokens; a TTL lock (even Redlock) is fine, because a double-run is wasteful, not corrupting. Do not pay the complexity tax for a non-corrupting race.
- **You have a single node and accept the downtime.** If your RTO budget tolerates a manual restart and you have no replica, you have no leader-election problem and no split-brain — do not build HA you do not need.
- **The data is naturally conflict-free.** Append-only logs, CRDTs, and last-write-wins-with-synced-clocks structures can tolerate concurrent writers by design; for these, the question shifts from "prevent two leaders" to "resolve concurrent writes," which is the multi-leader/leaderless story.
- **You would be adding an even voter count.** Going from 3 to 4 voters, or 5 to 6, adds cost and a tie risk without adding fault tolerance. Stay odd.

The thread through all of it is the one sentence from Chapter 8 that is worth tattooing on the inside of your eyelids: *a node cannot trust its own judgment of whether it is still the leader.* Two nodes will, sometimes, both believe they are primary — that is not a bug you can eliminate, it is a property of asynchronous networks and pausing processes. What you can eliminate is the *consequence.* Make the resource check a monotonic token; bound leadership with a lease and a real clock-skew margin; decide leadership by majority with an odd voter count and an independently-placed witness; STONITH the resources that cannot check tokens; and treat the consensus epoch as the fence it already is. Do that, and a forty-three-second partition stays a forty-three-second blip instead of becoming a twenty-four-hour reconciliation by hand.

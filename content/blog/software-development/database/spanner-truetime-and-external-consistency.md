---
title: "Spanner: How Google Built a Globally-Distributed, Externally-Consistent Database"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-the-architecture-up tour of Google Spanner — the universe/zones/spanservers layout, TrueTime's uncertainty interval, the commit-wait trick that buys external consistency across continents, read-write and snapshot transactions, the honest CP story, and the HLC descendants that drop atomic clocks."
tags:
  [
    "spanner",
    "truetime",
    "external-consistency",
    "distributed-sql",
    "google",
    "linearizability",
    "paxos",
    "commit-wait",
    "distributed-systems",
    "databases",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/spanner-truetime-and-external-consistency-1.webp"
---

For most of the 2000s, the conventional wisdom in distributed databases was a hard trilemma: you could have SQL with ACID transactions, or you could have horizontal scale across many machines, or you could have a single logical database that spanned the planet — but you could not have all three, and you certainly could not have all three *with strong consistency*. The standard playbook was to shard MySQL by hand, give up cross-shard transactions, and bolt on an eventually-consistent cache to paper over the latency. Google ran its entire AdWords business this way, on a sea of manually-sharded MySQL instances, and it hurt: re-sharding a growing table could take two years of engineering effort, and the lack of cross-shard transactions pushed consistency bugs up into application code where they festered.

Spanner is the system Google built to make that trilemma a false one. It is a SQL database that runs ACID transactions across thousands of machines in datacenters on multiple continents, and it provides the *strongest* consistency guarantee a transaction system can offer — external consistency, which is linearizability lifted to whole transactions. If transaction T1 commits in a New York datacenter before T2 starts in Tokyo, every reader anywhere on Earth sees T1's effects before T2's, with no exceptions and no "eventually." This was widely believed to be impossible at global scale, because global ordering seems to require either a single coordinator (which can't scale) or perfectly synchronized clocks (which don't exist). Spanner's answer, published in the [OSDI 2012 paper by Corbett et al.](https://research.google/pubs/spanner-googles-globally-distributed-database-2/), is one of the most elegant ideas in systems engineering: instead of pretending clocks are perfect, *expose their imperfection as a bounded number, and then wait it out*.

The diagram below is the mental model the rest of this article unpacks. A Spanner deployment is a *universe* carved into *zones* (roughly, datacenters); each zone runs hundreds to thousands of *spanservers*; each spanserver hosts hundreds to thousands of *tablets*; and the replicas of one tablet, scattered across zones on different continents, form a *Paxos group* with an elected leader. Data is not replicated as raw bytes — it is replicated as an agreed-upon, timestamp-ordered log, exactly as in [Raft and Paxos consensus](/blog/software-development/database/raft-consensus-from-scratch). What makes Spanner more than "Paxos with SQL on top" is the timestamps: every committed transaction is stamped with a value from TrueTime, and those stamps are chosen so that their order matches real-world order. That single discipline is what turns a pile of independent Paxos groups into one globally-consistent database.

![A Spanner universe contains zones, each zone runs spanservers, each spanserver hosts tablets, and one tablet's replicas across zones form a Paxos group with a leader and followers](/imgs/blogs/spanner-truetime-and-external-consistency-1.webp)

> External consistency is not a performance feature you turn on. It is a contract: the database promises that the timestamps it assigns to transactions respect the real-time order in which those transactions actually happened, even though no two machines in the system share a clock. Everything in Spanner's design — TrueTime, commit-wait, Paxos leases, two-phase commit — exists to keep that one promise.

## Why this is different from what most engineers assume

Most engineers carry a mental model of "strongly consistent distributed database" that quietly assumes one of two cheats: either there's a single node that orders everything (so it doesn't really scale), or the clocks are synchronized well enough to just compare timestamps (so it's secretly broken under clock skew). Spanner does neither, and the gap between the comfortable assumption and what Spanner actually does is the whole reason TrueTime exists.

| Assumption | The comfortable mental model | What Spanner actually does |
| --- | --- | --- |
| "Just use the wall-clock timestamp." | Each node reads its clock, stamps the transaction, and we compare stamps. | Wall clocks drift and disagree by milliseconds to seconds. Comparing two nodes' raw timestamps can reverse the true order. Spanner never trusts a single clock reading — it reads an *interval*. |
| "Strong consistency means a global lock or a single sequencer." | One coordinator hands out monotonic sequence numbers. | A single sequencer caps throughput and adds a cross-continent round trip to every write. Spanner has *no* global sequencer; ordering emerges from independent timestamps plus commit-wait. |
| "Reads need to coordinate too." | Every read contacts a quorum to be safe. | Spanner serves read-only transactions and snapshot reads *lock-free* from any single replica that is fresh enough, with zero coordination — because the timestamp already pins down what "consistent" means. |
| "If the network partitions, a globally-consistent DB just keeps working." | TrueTime is magic; partitions don't matter. | Spanner is a **CP** system. During a real partition it sacrifices availability on the minority side to preserve consistency. Google's private network makes partitions rare, not impossible. |
| "You can get Spanner's guarantee on commodity hardware with NTP." | Atomic clocks are a Google flex, not a requirement. | Without a *tight* clock bound you cannot do commit-wait. The descendants (CockroachDB, YugabyteDB) keep serializability but trade external consistency for *uncertainty restarts* — a real, different guarantee. |

Every row in the right column is a design decision forced by physics. The rest of this article walks each one in order: the clock problem, the TrueTime API that frames it, the commit-wait trick that solves it, the transaction protocols built on top, the honest CAP story, and the systems that tried to do it without the atomic clocks.

## 1. The clock problem, stated precisely

> **Senior rule of thumb:** the moment your correctness argument contains the phrase "and node A's clock says it's later than node B's," you have a bug, unless you can bound how wrong both clocks are. Unbounded clock comparison across machines is never safe.

To order transactions globally, you need a notion of "before" that every node agrees on. The naive idea is to use wall-clock time: stamp each transaction with the committing node's `gettimeofday()` and order by the stamp. This fails because clocks on different machines disagree. Even with NTP (Network Time Protocol) keeping them roughly aligned, two servers in the same rack can differ by a few milliseconds, and servers across a continent can differ by tens or hundreds of milliseconds during NTP hiccups. Worse, clocks *drift*: a typical crystal oscillator drifts at a rate that, left uncorrected, accumulates to seconds per day.

Here is the concrete failure. Suppose a user in New York transfers money (transaction T1) at true time 12:00:00.000, and the New York node's clock happens to run 5 ms fast, so T1 gets stamped 12:00:00.005. A microsecond later in real time, a reader in London (transaction T2) checks the balance; the London node's clock happens to run 5 ms slow, so it reads 11:59:59.995 and serves a snapshot as of that timestamp — which is *before* T1's stamp. The reader sees the old balance, even though the transfer "already happened" in the real world. The timestamps lied about the order. This is not a hypothetical; it is the default behavior of any system that compares raw wall-clock readings, and it silently violates the very property you were trying to guarantee.

The two classical escapes both have fatal costs at global scale. **Logical clocks** (Lamport timestamps, vector clocks — covered in depth in [time, clocks and ordering in distributed systems](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems)) order events by causality without any physical clock, but they only order events that actually communicated; they cannot tell you that a transaction in New York happened before an unrelated one in Tokyo, which is exactly what external consistency requires. **A single global sequencer** gives a true total order, but funneling every commit through one machine caps your write throughput at one machine's rate and adds a wide-area round trip to every transaction. Spanner needed the total order of a sequencer with the scalability of independent nodes. TrueTime is how it squares that circle.

The precise property Spanner wants is worth naming, because the name tells you exactly how strong it is. **External consistency** is *strict serializability*: the combination of serializability (transactions appear to execute in *some* serial order) and linearizability lifted from single objects to whole transactions (that serial order is consistent with real-time order). [Google's own documentation](https://docs.cloud.google.com/spanner/docs/true-time-external-consistency) is blunt that this is *stronger* than linearizability alone — "linearizability doesn't say anything about the behavior of transactions," only individual operations. So Spanner is not merely "linearizable"; it is the strongest practical guarantee a multi-object transaction system can offer, and it offers it across continents. Holding that bar is what makes the rest of the design as intricate as it is — a weaker guarantee would let you skip commit-wait entirely, which is exactly the bargain the HLC descendants strike in §8.

```python
# The unsafe pattern Spanner exists to prevent. Two nodes, two clocks that
# disagree by a few ms, and a comparison that reverses true causal order.

import time

def commit_naive(node_clock_offset_ms: float) -> float:
    # Each node stamps with its own skewed wall clock. Offsets differ per node.
    return time.time() + node_clock_offset_ms / 1000.0

# True order: T1 (New York) commits, THEN T2 (London) reads — microseconds later.
t1_stamp = commit_naive(node_clock_offset_ms=+5)   # NY clock runs 5 ms fast
t2_stamp = commit_naive(node_clock_offset_ms=-5)   # London clock runs 5 ms slow

# Despite T1 happening first in the real world, the stamps can say otherwise:
print(t2_stamp < t1_stamp)   # -> often True. External consistency violated.
```

## 2. TrueTime: turn clock uncertainty into a number you can wait out

> **Senior rule of thumb:** the honest API for a physical clock is not "what time is it?" but "what is the smallest interval I can promise contains the true time right now?" Once uncertainty is a first-class return value, you can reason about it instead of hoping it's zero.

TrueTime's core move is to stop pretending a clock returns an instant. Instead, the `TT.now()` call returns a `TTinterval` — a pair `[earliest, latest]` of timestamps — with the guarantee that the true absolute time `t_abs` of the call lies somewhere inside that interval. The width of the interval is `2ε` (two epsilons), where `ε` is the *instantaneous error bound*: half the interval width, the most the clock could be wrong in either direction at this instant. The figure below is the whole idea: a bracket on the time line, true time guaranteed inside, width `2ε`.

![TrueTime returns an interval [earliest, latest] on the absolute-time axis; true now lies inside; the interval width is 2-epsilon, backed by GPS and atomic clocks](/imgs/blogs/spanner-truetime-and-external-consistency-2.webp)

The TrueTime API is tiny — three methods:

| Method | Returns | Meaning |
| --- | --- | --- |
| `TT.now()` | `TTinterval: [earliest, latest]` | An interval guaranteed to contain the true current time. |
| `TT.after(t)` | `bool` | True if `t` is *definitely* in the past: `t < TT.now().earliest`. |
| `TT.before(t)` | `bool` | True if `t` is *definitely* in the future: `t > TT.now().latest`. |

`TT.after(t)` is the workhorse. It does not ask "is `t` in the past according to my clock?" — it asks the stronger question "can I *prove* `t` is in the past, given the worst case of how wrong my clock might be?" The answer is yes only when even the *earliest* possible true time has passed `t`. That conservatism is what makes correctness possible.

### How TrueTime keeps epsilon small

The guarantee `tt.earliest ≤ t_abs ≤ tt.latest` is only useful if `ε` is small — if the interval were a full second wide, the database would crawl. Google keeps it tight with redundant, independent clock references. Each datacenter runs a set of *time master* machines. Most masters have GPS receivers (GPS satellites carry atomic clocks and broadcast time directly); a few — the paper calls them *Armageddon masters* — have local atomic clocks, as a backstop for the failure modes that would take out all the GPS receivers at once (antenna failures, radio interference, a leap-second bug, GPS spoofing). On every machine, a *timeslave daemon* polls a handful of masters (some nearby, some far, some Armageddon), runs a variant of Marzullo's algorithm to throw out lying clocks, and synchronizes the local clock to the survivors.

Between polls, the daemon does not pretend its clock is still perfect. It advertises a *growing* uncertainty derived from the worst-case drift rate. The result, in Google's production environment, is that `ε` is a **sawtooth**: it dips to about 1 ms right after a poll, then climbs to about 7 ms by the end of the 30-second poll interval, giving an average around 4 ms. The applied drift rate is 200 microseconds per second, which over a 30-second interval accounts for 0 to 6 ms of the sawtooth; the remaining ~1 ms is the round-trip communication delay to the time masters. The whole point of the GPS-plus-atomic-clock investment is to keep that sawtooth low and to keep its *worst case* bounded, even when a time master goes down.

```python
# A faithful sketch of TrueTime's contract (not Google's real implementation).
# Production TrueTime is C++ talking to GPS/atomic time masters; the SHAPE is this.

import time
from dataclasses import dataclass

DRIFT_RATE = 200e-6          # 200 microseconds per second of drift
POLL_INTERVAL_S = 30         # daemon re-syncs every 30 s
COMM_DELAY_S = 0.001         # ~1 ms round trip to the time masters

@dataclass
class TTinterval:
    earliest: float
    latest: float

class TrueTime:
    def __init__(self):
        self._last_sync_wall = time.time()
        self._last_sync_true = time.time()   # assume perfect at sync instant

    def _epsilon(self) -> float:
        # Uncertainty grows linearly with time since last successful sync:
        # the classic sawtooth, reset to ~comm delay at each poll.
        since_sync = time.time() - self._last_sync_wall
        return COMM_DELAY_S + DRIFT_RATE * since_sync   # ~1 ms .. ~7 ms over 30 s

    def now(self) -> TTinterval:
        t = time.time()
        eps = self._epsilon()
        return TTinterval(earliest=t - eps, latest=t + eps)

    def after(self, t: float) -> bool:
        return t < self.now().earliest      # t is PROVABLY in the past

    def before(self, t: float) -> bool:
        return t > self.now().latest        # t is PROVABLY in the future
```

The genius is not that `ε` is small — NTP gets you single-digit milliseconds on a good day too. The genius is that `ε` is a *number Spanner can read and act on*. When `ε` spikes (a time master is unavailable, a machine is overloaded), Spanner does not become incorrect; it becomes *slower*, because the next idea — commit-wait — waits proportionally to `ε`.

### Why two clock technologies, and how lies get rejected

It is worth dwelling on *why* TrueTime uses two independent clock sources rather than just a fleet of GPS receivers, because the reasoning is a clean example of designing for correlated failure. GPS and atomic clocks fail in *uncorrelated* ways, and that is the entire reason both exist in the design. GPS receivers share failure modes that can take out *all* of them at once: a regional radio-frequency interference event, an antenna or cabling fault that's been copy-pasted across a datacenter, a firmware bug in handling a leap second, GPS spoofing, or an outage of the GPS system itself. If TrueTime relied on GPS alone, one of those events would simultaneously inflate `ε` on every machine in a datacenter — or worse, make every GPS receiver agree on the *same wrong time*, which is the dangerous case because agreement looks like correctness. Atomic clocks (the "Armageddon" masters) fail differently: they drift slowly due to frequency error, and they fail independently of each other and of GPS. So when GPS goes dark, the atomic clocks keep advertising time with a *slowly growing* uncertainty derived from their worst-case drift — `ε` climbs, commit-wait stretches, but the system stays correct. Two technologies whose failure modes don't overlap give you a clock that degrades gracefully instead of lying confidently.

The defense against a clock that lies — advertises a wrong time with too-small an uncertainty — is a variant of **Marzullo's algorithm**, run by every machine's timeslave daemon. Each daemon polls several masters (some GPS, some Armageddon, some nearby, some far) and receives from each an interval `[t - ε_master, t + ε_master]`. Marzullo's algorithm computes the smallest interval consistent with the *largest agreeing subset* of these intervals, treating masters whose intervals don't overlap the consensus as liars and discarding them. A single master whose clock has jumped is outvoted by the honest majority and rejected. On top of that, machines that exhibit clock-frequency excursions larger than the worst-case bound derived from their hardware specs are *evicted* from the fleet entirely — Spanner would rather lose a machine than trust a clock it can't bound. The combination — redundant uncorrelated sources, Marzullo to reject liars, eviction of out-of-spec hardware — is what lets Spanner make the strong promise `tt.earliest ≤ t_abs ≤ tt.latest` and actually mean it.

| Clock-infrastructure failure | What happens to `ε` | What happens to correctness |
| --- | --- | --- |
| One GPS receiver fails | Daemon drops it via Marzullo; `ε` from remaining masters | Unaffected — still bounded |
| All GPS in a datacenter fail | Falls back to Armageddon atomic clocks; `ε` grows on worst-case drift | Unaffected — commit-wait just lengthens |
| A master's clock jumps (lies) | Marzullo outvotes it; discarded | Unaffected — the liar is rejected |
| A local machine's clock drifts out of spec | Machine is evicted from the fleet | Unaffected — no reads/writes served from it |
| Time master overloaded / network slow | `ε` spikes datacenter-wide | Unaffected; latency rises (commit-wait proportional to `ε`) |

## 3. Commit-wait: the trick that buys external consistency

> **Senior rule of thumb:** if you want a timestamp to be a faithful proxy for real-world order, the cheapest way is to *delay declaring success* until you can prove the timestamp is already in the past. Trade a few milliseconds of latency for a guarantee you could never get otherwise.

This is the centerpiece of the entire system, so we will build it slowly. The problem from §1 was that a node could stamp a transaction with a time that is actually in the *future* (its clock ran fast), so a later transaction reading a slow clock could get a smaller stamp and appear to have happened first. Spanner kills this with two rules that together enforce the **external-consistency invariant**: if T1 commits before T2 starts (in real time), then `s1 < s2` (T1's commit timestamp is strictly smaller).

The two rules, in the paper's exact framing:

- **Start rule.** The coordinator leader for a write Ti picks a commit timestamp `si` that is no less than `TT.now().latest`, computed at the moment the commit request arrives at the coordinator. Picking `latest` (not `earliest`, not the midpoint) means `si` is at least as large as the true time of the commit request — the timestamp is "now or slightly ahead."
- **Commit-wait rule.** The coordinator does not release the transaction's locks (does not let anyone see Ti's effects) until `TT.after(si)` is true — that is, until `si < TT.now().earliest`. At that instant, `si` is *provably* in the absolute past.

The figure below traces it. T1 acquires its locks, picks `s = TT.now().latest`, then deliberately *blocks* — holding its locks — until TrueTime can prove `s` has passed. Only then does it release locks and make `s` visible. Any transaction T2 that starts *after* T1 became visible therefore starts after `s` is already in the past, so when T2 picks `s2 = TT.now().latest`, it is forced to a value strictly greater than `s`. The order of the timestamps now matches the order of the events. That is external consistency.

![Commit-wait timeline: T1 picks commit timestamp s, holds locks during a wait until TT.after(s) is true so s is provably past, then any later T2 is forced to a strictly greater timestamp s2 > s1](/imgs/blogs/spanner-truetime-and-external-consistency-3.webp)

The proof is four lines, and it is worth internalizing because every other guarantee in Spanner leans on it. Let `e_commit_1` be the real-world instant T1's effects become visible, and `e_start_2` the instant T2 begins:

1. `s1 < t_abs(e_commit_1)` — by the commit-wait rule, T1 waited until `s1` was provably past before becoming visible.
2. `t_abs(e_commit_1) < t_abs(e_start_2)` — by assumption, T1 commits before T2 starts.
3. `t_abs(e_start_2) ≤ s2` — by the start rule, T2 picks `s2 ≥ TT.now().latest ≥` the true time it started.
4. Chaining 1–3: `s1 < s2`. Done.

Here is commit-wait as the coordinator actually runs it, with the structure of the OSDI paper's §4.2.1:

```python
def commit_rw_transaction(txn, participant_leaders, truetime: TrueTime):
    # Phase 0: locks already held; reads done; writes buffered at the client.

    # --- Two-phase commit prepare phase (participants) ---
    prepare_timestamps = []
    for p in participant_leaders:
        # Each participant picks a prepare ts > any ts it has used (monotonicity),
        # logs a PREPARE record via Paxos, and reports its prepare timestamp.
        s_prepare = p.log_prepare_via_paxos(txn)
        prepare_timestamps.append(s_prepare)

    # --- Coordinator picks the commit timestamp (the START rule) ---
    s = max(
        truetime.now().latest,        # >= true time of the commit request
        max(prepare_timestamps),      # >= every participant's prepare ts
        coordinator.last_assigned_ts + 1e-9,  # monotonic at this leader
    )

    # Make the COMMIT decision durable across the coordinator's Paxos group.
    coordinator.log_commit_via_paxos(txn, s)

    # --- COMMIT-WAIT: the load-bearing line of the whole system ---
    # Do NOT reveal Ti's effects until s is provably in the absolute past.
    # Expected wait is ~2*epsilon; typically overlapped with Paxos messaging.
    while not truetime.after(s):
        sleep_briefly()               # busy-ish wait, microseconds at a time

    # Now s < TT.now().earliest for everyone, everywhere. Safe to release.
    for p in participant_leaders:
        p.apply_and_notify(txn, s)    # participants apply at the SAME ts s
    coordinator.apply_at(txn, s)
    release_locks(txn)
    return s
```

### What commit-wait actually costs

The expected wait is roughly `2ε` — Spanner picked `s` up to `ε` ahead of true time (the `latest` edge), and must wait until true time advances past `s`, which takes about `2ε` in the worst alignment. With `ε ≈ 4 ms` on average, that is a single-digit-millisecond tax per read-write transaction. Crucially, the paper notes this wait is *typically overlapped with the Paxos communication* that's happening anyway to replicate the commit record — so the wall-clock cost is often near zero, not an additive `2ε`. The measured numbers in the paper bear this out: a single-replica write averages around 9 ms; replicated writes across three to five replicas land around 14 ms, dominated by Paxos round trips, not commit-wait.

This also explains why `ε` is treated as a system-health metric at Google. If `ε` spikes to 100 ms because of a flaky time master, every read-write transaction's commit-wait stretches to ~200 ms — the database does not break, it just slows down, and the on-call engineer gets paged about clock health, not data corruption. **Spanner converts a correctness problem into a latency problem, and latency problems you can monitor and fix.** That trade — accept a tunable latency penalty in exchange for an unbreakable correctness guarantee — is the single most important design decision in the system.

### A worked example: two transactions, real numbers

Let's nail down the mechanism with concrete numbers so the abstract proof feels inevitable. Say `ε = 4 ms`, and the true absolute time is 12:00:00.000 when T1's commit request reaches its coordinator. Walk it:

- At true time `.000`, the coordinator calls `TT.now()`. Its local clock might be anywhere within `±4 ms` of true time, so it returns an interval roughly `[.000 - 4ms, .000 + 4ms]`, i.e. `[11:59:59.996, 12:00:00.004]`. By the start rule, the coordinator picks `s = latest = 12:00:00.004`. Note `s` is up to `ε` *ahead* of true time — that's deliberate.
- The coordinator now runs commit-wait: it loops on `TT.after(s)`, which becomes true only when `s < TT.now().earliest`. With `ε = 4 ms`, `earliest = true_time - 4ms`, so `earliest` passes `12:00:00.004` only when true time reaches about `12:00:00.008`. The coordinator therefore holds locks until true time `≈ .008` — a wait of about `8 ms = 2ε` from when it picked `s`.
- At true time `.008`, T1 releases locks and becomes visible. `s = .004` is now provably in the past everywhere: any node, reading any clock within `±4 ms` of the true `.008`, computes `earliest ≥ .004`.
- Now suppose T2 starts at true time `.009` — just after T1 became visible. T2's coordinator calls `TT.now()`, gets `latest ≥ .009 - … wait, latest = true_time + ε = .013`, and picks `s2 ≥ .013 > .004 = s1`. T2's timestamp is strictly greater. External consistency holds, and it held without T1 and T2 ever communicating or sharing a clock.

The whole dance hinges on one asymmetry: the start rule picks `s` at the `latest` (optimistically far-future) edge, while `TT.after` checks against the `earliest` (pessimistically far-past) edge. Picking high and waiting until the low edge clears guarantees a `2ε` gap that absorbs *any* clock disagreement within the bound. Shrink `ε` and the gap shrinks proportionally — which is exactly why the atomic-clock investment pays for itself in commit latency.

```python
# Commit-wait, instrumented to show the actual wait. Run against the TrueTime
# sketch from section 2; prints the milliseconds spent holding locks.

import time

def measure_commit_wait(tt: "TrueTime") -> float:
    now = tt.now()
    s = now.latest                       # START rule: pick the latest edge
    t0 = time.time()
    while not tt.after(s):               # COMMIT-WAIT: block until s is past
        time.sleep(0.0002)               # 200 us granularity
    held_ms = (time.time() - t0) * 1000
    return held_ms                       # ~ 2 * epsilon in the steady state

# With epsilon ~4 ms you observe ~8 ms here; with epsilon spiked to 40 ms you'd
# observe ~80 ms — correctness identical, latency tracking epsilon linearly.
```

## 4. The architecture, layer by layer

> **Senior rule of thumb:** in a planet-scale database, the unit you replicate and the unit you *move* are not the same as the unit you query. Get those three granularities straight before you reason about anything else.

We can now zoom out from clocks to the machinery that uses them. Recall the topology from figure 1: universe → zones → spanservers → tablets → Paxos groups. Let's give each layer its job.

A **universe** is a single Spanner deployment. Google runs only a handful — a test/playground universe, a development/production universe, and a production-only universe. A universe is administered as one unit by a *universe master* (a status console) and a *placement driver* (which moves data around on a timescale of minutes; more on it shortly).

A **zone** is the unit of physical isolation and administration — roughly a datacenter, or a slice of one. A zone has one *zonemaster* (assigns data to spanservers) and between one hundred and several thousand *spanservers* (serve data to clients). Zones can be added to or removed from a running universe, which is how Spanner grows: bring a new datacenter online, register its zone, and the placement driver starts migrating replicas into it.

A **spanserver** is the workhorse process. Each one manages between 100 and 1000 *tablets*. A tablet is Spanner's storage unit; like Bigtable's tablet, it maps `(key:string, timestamp:int64) → string`. That timestamp in the key is what makes Spanner multi-version: every value is stored at the timestamp it was written, so an old snapshot read just looks up the most recent version *at or before* its read timestamp. Tablet state lives in B-tree-like files plus a write-ahead log on **Colossus**, the successor to the Google File System.

### The spanserver software stack

The figure below is the per-spanserver stack, bottom to top, and it pins down which components run on *every* replica versus only on the *leader* — a distinction that matters enormously for performance.

![The spanserver software stack from Colossus storage at the bottom through tablet, Paxos state machine, lock table, up to the transaction manager at the top; the lock table and transaction manager run only on the leader replica](/imgs/blogs/spanner-truetime-and-external-consistency-10.webp)

At the bottom sits **Colossus** (distributed storage). Above it, each tablet. On top of *each* tablet, the spanserver runs a single **Paxos state machine** — this is the replication layer, and it runs on every replica of the tablet. The Paxos groups are long-lived: a leader holds a *timed lease* (10 seconds by default) so it doesn't have to re-run an election on every write. Writes must initiate the Paxos protocol at the leader; reads can be served by any replica that is sufficiently up-to-date (we'll make "sufficiently" precise in §6).

The top two layers run **only on the leader replica**. The **lock table** implements two-phase locking for read-write transactions — it maps key ranges to lock state, and it is what makes concurrent read-write transactions serializable. The **transaction manager** supports distributed (multi-group) transactions: it plays the coordinator or participant role in two-phase commit. For a transaction that touches a single Paxos group, the transaction manager can be bypassed entirely — the lock table and Paxos provide all the transactionality needed, no 2PC required. This single-group fast path is why schema designers work so hard to keep related data in one group.

### The directory: the unit of placement and movement

Here is the granularity that trips people up. The unit Spanner *moves between Paxos groups* is neither the row nor the tablet — it is the **directory** (the paper admits "bucket" would have been a better name). A directory is a set of contiguous keys that share a common prefix. All data in one directory has the same replication configuration (how many replicas, in which zones), and the directory is the atomic unit the placement driver relocates.

![Two Paxos groups each holding several directories of contiguous keys; a hot directory moves as one unit via movedir from an overloaded group to one with spare capacity](/imgs/blogs/spanner-truetime-and-external-consistency-6.webp)

The placement driver continuously watches for directories it should move: to shed load from a hot Paxos group, to colocate data that's accessed together, or — this is the killer feature for global apps — to move a directory *physically closer to the clients that read it*. The operation that does this, `movedir`, is a background task that copies a directory to its new group (typically tens of megabytes at a time) and then atomically flips ownership. Because data moves at directory granularity rather than row-by-row or all-at-once, a giant table can be rebalanced incrementally with minimal disruption — the thing that took two years on sharded MySQL becomes a continuous, automatic background process.

There is a deeper reason the directory exists as a separate concept from the tablet. A tablet in Spanner, unlike Bigtable's, is not required to be a single contiguous key range — it can hold multiple directory partitions that are not adjacent in key space. This decoupling is what lets the placement driver pack directories that are *accessed together* into the same group even if their keys are far apart, and split apart directories that contend even if their keys are adjacent. The unit of *storage* (tablet) and the unit of *placement* (directory) being different is what gives Spanner the freedom to optimize locality independently of the key layout — a freedom a system that conflated the two (move whole tablets only) would not have.

This is also where you, the schema designer, exert control. By interleaving child tables under a parent (so a customer's rows and all their orders sit in the same directory), you ensure the common-case transaction touches one Paxos group, avoids two-phase commit, and avoids cross-region round trips. The directory is the lever; interleaving is how you pull it.

```sql
-- Cloud Spanner DDL. Interleave Orders under Customers so that a customer row
-- and all of that customer's orders live in the SAME directory (and thus the
-- same Paxos group and the same split). The child's PK must be PREFIXED by the
-- parent's PK. ON DELETE CASCADE ties their lifecycles together.

CREATE TABLE Customers (
  CustomerId   INT64 NOT NULL,
  Name         STRING(256),
  Region       STRING(32),
  CreatedAt    TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp = true),
) PRIMARY KEY (CustomerId);

CREATE TABLE Orders (
  CustomerId   INT64 NOT NULL,   -- parent key first: this is the interleave key
  OrderId      INT64 NOT NULL,
  Amount       NUMERIC,
  Status       STRING(16),
  PlacedAt     TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp = true),
) PRIMARY KEY (CustomerId, OrderId),
  INTERLEAVE IN PARENT Customers ON DELETE CASCADE;

-- A transaction that reads a customer and writes one of their orders now hits a
-- single Paxos group: single-group fast path, no 2PC, no cross-group commit.
```

## 5. Read-write transactions: 2PC layered over Paxos

> **Senior rule of thumb:** two-phase commit gets a bad reputation because textbook 2PC has a single point of failure — the coordinator. Spanner's 2PC is safe precisely because *every participant in the 2PC is itself a fault-tolerant Paxos group*, so no single machine dying can lose the decision.

A read-write transaction that spans multiple Paxos groups is the most expensive operation Spanner offers, and understanding its choreography explains most of Spanner's latency profile. The flow combines pessimistic two-phase locking (for isolation), two-phase commit (for atomicity across groups), and Paxos (for durability within each group). The figure shows the shape: the client drives 2PC across group leaders, and each leader makes its local decision durable by replicating a record through its Paxos followers.

![A read-write transaction shown as two-phase commit across two Paxos group leaders, where the coordinator logs a commit record and the participant logs a prepare record, each replicated to its own followers](/imgs/blogs/spanner-truetime-and-external-consistency-4.webp)

Walking it step by step:

1. **Reads and locking.** The client issues its reads to the leader of each relevant group. Each leader acquires *read locks* and returns the latest data. Reads use **wound-wait** to avoid deadlock: when a transaction needs a lock held by a younger transaction, it *wounds* (aborts) the younger one; when it needs a lock held by an older transaction, it *waits*. Because the priority order is by transaction timestamp (older = higher priority), wound-wait guarantees no cycle of waits can form. While the transaction is open, the client sends keepalives so participant leaders don't time it out.

2. **Buffer writes, begin commit.** Writes are buffered at the client, not applied during the transaction — so a transaction's own reads do not see its uncommitted writes, and the writes carry no timestamps yet. When the client has done all reads and buffered all writes, it picks one group to be the *coordinator* and sends a commit message to every participant leader (carrying the coordinator's identity and the buffered writes). Having the client drive 2PC avoids shipping the data across wide-area links twice.

3. **Prepare phase (participants).** Each non-coordinator participant leader acquires its *write locks*, chooses a prepare timestamp larger than any timestamp it has previously used (preserving the monotonicity invariant), logs a `PREPARE` record through its Paxos group, and tells the coordinator its prepare timestamp.

4. **Commit decision (coordinator).** The coordinator acquires its own write locks but skips the prepare step. After hearing from all participants, it chooses the commit timestamp `s` subject to three constraints: `s ≥` every participant's prepare timestamp; `s ≥ TT.now().latest` (the start rule); and `s >` any timestamp this leader has previously assigned (monotonicity). It logs a `COMMIT` record through Paxos.

5. **Commit-wait, then apply.** Before letting any replica apply the commit, the coordinator waits until `TT.after(s)` — commit-wait. Then it sends `s` to the client and every participant. *All* groups apply the transaction at the *same* timestamp `s` and release their locks. One transaction, one timestamp, applied identically everywhere.

The reason this 2PC is not the availability liability it is in classic distributed databases: in textbook 2PC, if the coordinator crashes after participants have prepared, the participants are stuck holding locks indefinitely ("blocked"). In Spanner, the coordinator's state is a `COMMIT` record replicated across its Paxos group — if the coordinator machine dies, a new leader is elected from the followers and *already has the commit record*, so it finishes the protocol. The 2PC coordinator is fault-tolerant because it is a Paxos group, not a single node. This is the architectural insight that makes cross-group transactions practical at Google's scale. (For the consensus mechanics underneath, see [Paxos and Multi-Paxos explained](/blog/software-development/database/paxos-and-multi-paxos-explained).)

```python
# Application-level read-write transaction against Cloud Spanner (Python client).
# The runner handles wound-wait aborts by RETRYING the whole function — your
# transaction body must be idempotent because it can run more than once.

from google.cloud import spanner

client = spanner.Client()
database = client.instance("prod-instance").database("ledger")

def transfer_funds(transaction, src_id: int, dst_id: int, cents: int):
    # All reads + writes inside ONE read-write transaction => atomic, serializable.
    src = list(transaction.execute_sql(
        "SELECT Balance FROM Accounts WHERE AccountId = @id",
        params={"id": src_id}, param_types={"id": spanner.param_types.INT64},
    ))[0][0]
    if src < cents:
        raise ValueError("insufficient funds")   # abort: no commit, no effect

    transaction.execute_update(
        "UPDATE Accounts SET Balance = Balance - @c WHERE AccountId = @s",
        params={"c": cents, "s": src_id},
        param_types={"c": spanner.param_types.INT64, "s": spanner.param_types.INT64},
    )
    transaction.execute_update(
        "UPDATE Accounts SET Balance = Balance + @c WHERE AccountId = @d",
        params={"c": cents, "d": dst_id},
        param_types={"c": spanner.param_types.INT64, "d": spanner.param_types.INT64},
    )

# run_in_transaction() retries on ABORTED (wound-wait) until it commits or raises.
database.run_in_transaction(transfer_funds, src_id=1001, dst_id=2002, cents=5000)
```

### Where the milliseconds go

It helps to account for a multi-group read-write transaction's latency in terms of round trips, because that's what determines whether your schema is well-designed. A cross-group commit costs, roughly: one client-to-leader round trip to acquire locks and read; the prepare phase (each participant logs a `PREPARE` via Paxos — one intra-group quorum round trip, done in parallel across participants); the commit decision (the coordinator logs a `COMMIT` via Paxos — one more quorum round trip); commit-wait (`~2ε`, usually overlapped with that Paxos messaging); and a final notify-and-apply. The dominant terms are the Paxos quorum round trips, and *each* of those costs the network round-trip time to a majority of replicas — single-digit milliseconds within a region, tens to 100+ ms if the quorum spans continents.

| Cost component | Round trips | Within one region | Across continents |
| --- | --- | --- | --- |
| Acquire locks + read | 1 (client → leader) | ~1 ms | ~1 ms (local leader) |
| Prepare (Paxos, parallel) | 1 quorum RT | ~3–5 ms | ~50–100 ms |
| Commit decision (Paxos) | 1 quorum RT | ~3–5 ms | ~50–100 ms |
| Commit-wait (`~2ε`) | overlapped | ~0 (hidden) | ~0 (hidden) |
| Notify + apply | 1 | ~1 ms | ~1 ms |
| **Total RW commit** | — | **~9–14 ms** | **~100–200 ms** |

The single-group fast path collapses this dramatically: no prepare phase, no separate participants, just one Paxos round to commit and a hidden commit-wait. That's the whole reason interleaving matters — it converts an expensive multi-group transaction (two Paxos rounds plus 2PC bookkeeping) into a cheap single-group one (one Paxos round). Spanner's measured numbers line up: ~9 ms for a single-replica write, ~14 ms for a 3–5 replica write, with commit-wait essentially free because it hides inside the Paxos messaging that has to happen anyway.

### Paxos leader leases and the disjointness invariant

One subtlety underpins the monotonicity of timestamps *across* leader changes: the **disjointness invariant**. For each Paxos group, every leader's lease interval must be disjoint from every other leader's — at no real-world instant can two machines both believe they hold the lease. Spanner enforces this with timed leases (10 seconds by default): a leader gets a quorum of lease votes, and a replica implicitly extends its vote on each successful write. The invariant matters because timestamps must increase monotonically *even when leadership changes*. A leader only assigns timestamps within its lease interval, and before abdicating it must wait until `TT.after(s_max)` (where `s_max` is the largest timestamp it ever used) — so by the time the next leader can assign timestamps, all of the previous leader's timestamps are provably in the past. Disjoint leases plus "assign only within your lease" gives monotonic timestamps across the whole life of the group, which is what `t_safe^Paxos` (the next section) relies on to decide a replica is up-to-date. TrueTime shows up here too: the lease boundaries are reasoned about in absolute time, which only has meaning because every node shares TrueTime's bounded view of it.

## 6. Read-only transactions and snapshot reads: consistency without locks

> **Senior rule of thumb:** the cheapest strongly-consistent read is one that takes no locks and contacts no quorum. If you already know *which point in time* you want to read, you can serve that read from a single nearby replica — the timestamp does all the coordination work for you.

Most workloads are read-heavy, and this is where Spanner's MVCC design pays off enormously. A **read-only transaction** is not a degraded read-write transaction — it is a distinct, lock-free, side-effect-free path that still gives a globally-consistent view. It executes in two phases: assign a read timestamp `s_read`, then perform the reads as *snapshot reads* at `s_read` against any sufficiently up-to-date replica. No locks are taken, so read-only transactions never block writers and writers never block them.

![A read-only transaction assigns a read timestamp, routes to a nearby replica, waits until that replica's safe time reaches the timestamp, then serves a lock-free MVCC snapshot](/imgs/blogs/spanner-truetime-and-external-consistency-5.webp)

The simplest correct choice is `s_read = TT.now().latest`, which preserves external consistency by the same argument as writes. But a replica can only serve a read at timestamp `t` once it is *up-to-date* through `t` — that is, once its **safe time** `t_safe` has advanced past `t`. The safe time is the maximum timestamp at which a replica can correctly answer reads, and it's the minimum of two quantities:

- `t_safe^Paxos`: the timestamp of the highest Paxos write applied at this replica. Below this, no new writes will appear, because timestamps increase monotonically.
- `t_safe^TM`: governed by the transaction manager — it is `∞` when there are no prepared-but-not-yet-committed transactions, and otherwise one less than the smallest prepare timestamp of any in-flight 2PC transaction (because the outcome of those is not yet determined).

If you pick `s_read = TT.now().latest` aggressively, a freshly-chosen timestamp might be slightly ahead of a nearby replica's `t_safe`, forcing the read to wait for the replica to catch up. To minimize that, Spanner picks the *oldest* timestamp that still preserves external consistency. For a single-group read with no prepared transactions, it can even use `LastTS()` — the timestamp of the last committed write at that group — which trivially sees the latest write and orders after it, with no waiting at all.

This is the architecture of Spanner's read scalability: you can stand up read-only replicas near every population center, and they serve consistent reads *locally*, at single-digit-millisecond latency, with no cross-region traffic, because the read timestamp plus `t_safe` is all the coordination required. The paper's measurements show read-only transactions and snapshot reads completing in about 1.3 ms regardless of replica count, versus ~14 ms for writes — an order-of-magnitude gap that reflects exactly this lock-free, quorum-free design.

### Strong, stale, and bounded-staleness reads

Cloud Spanner exposes these timestamp choices as *timestamp bounds* you select per read. The trade is always the same: how fresh do you need the data versus how low do you want the latency.

| Read type | Timestamp chosen | Latency | When to use |
| --- | --- | --- | --- |
| **Strong read** | `TT.now().latest` (or `LastTS()`) | May wait for `t_safe` to catch up; one round on leader | "I must see every committed write up to now" — read-after-write, account balance after a transfer. |
| **Exact-staleness read** | A specific past timestamp `t` | Lowest; any replica with `t_safe ≥ t` | Reproducible reads, debugging, reading a known snapshot. |
| **Bounded-staleness read** | The newest timestamp within a staleness bound (e.g. ≤ 15 s old) | Low; often a single local replica, no waiting | Dashboards, analytics, anything that tolerates seconds of lag for big latency wins. |

```sql
-- Cloud Spanner: a STRONG read-only transaction sees all committed writes.
-- (gcloud / client libraries; this is the read-only, lock-free path.)
SET TRANSACTION READ ONLY;   -- strong by default: timestamp = latest
SELECT AccountId, Balance FROM Accounts WHERE Region = 'eu-west';
COMMIT;

-- A BOUNDED-STALENESS read: "give me data no more than 15 seconds old."
-- Served lock-free from the nearest replica, typically with zero wait.
-- (Via the client library timestamp bound; shown here as the conceptual knob.)
--   read_only(max_staleness=timedelta(seconds=15))
SELECT AccountId, Balance FROM Accounts WHERE Region = 'eu-west';
```

### Schema changes at a future timestamp

TrueTime enables one more trick that is impossible without a global clock: **non-blocking atomic schema changes**. A schema change in Spanner could touch millions of Paxos groups — you cannot lock them all. Instead, Spanner assigns the schema-change transaction a timestamp *in the future*, registers it during a prepare phase, and lets everything proceed: any read or write whose timestamp precedes the schema-change timestamp uses the old schema; any whose timestamp is after it blocks behind the change. Because TrueTime gives every node a shared notion of "the change happens at time `t`," the cutover is atomic across the whole database without ever stopping the world. Without a global clock, "the schema changes at `t`" would be a meaningless statement — there'd be no agreed `t`.

## 7. Spanner and CAP: it is CP, not magic

> **Senior rule of thumb:** any vendor who tells you their distributed database is "CA" — consistent *and* available under partitions — is either confused or selling something. CAP is a theorem. Spanner obeys it. What Spanner buys you is that partitions are rare, not that they're survivable while staying consistent.

It is tempting to read "globally consistent and available" and conclude Spanner beat the CAP theorem. It did not, and Google is admirably honest about this. The CAP theorem says that when the network partitions (P), a system must forfeit either consistency (C) or availability (A). [Eric Brewer — who coined CAP — wrote a whole paper](https://research.google/pubs/spanner-truetime-and-the-cap-theorem/) on exactly where Spanner sits, and the answer is unambiguous: Spanner is **CP**. During a real partition, Spanner chooses consistency and forfeits availability on the side that can't reach a quorum.

![Before and after a partition: with a healthy network the majority quorum commits and the system is available and consistent; during a partition the minority side rejects writes, choosing consistency over availability](/imgs/blogs/spanner-truetime-and-external-consistency-7.webp)

The figure shows why. A Paxos group needs a majority of replicas to commit a write and to hold the leader lease. When a partition splits a group, at most one side can hold a majority. The minority side cannot assemble a quorum, so its leader's lease lapses and it *rejects writes* — it would rather be unavailable than risk a second leader committing conflicting data (the split-brain failure that majority quorums exist to prevent, exactly as in [Raft](/blog/software-development/database/raft-consensus-from-scratch)). Consistency is preserved by sacrificing availability on the minority side. That is the definition of CP.

So why does everyone *experience* Spanner as always-available? Because of the second half of Brewer's argument: **you only forfeit availability during an actual partition, and Google has engineered partitions to be vanishingly rare.** Spanner does not run over the public internet; it runs over Google's private, redundant, multiply-pathed backbone, which Google controls end to end. Brewer's paper puts numbers on it. The data:

- Internally, Spanner delivers better than **5 nines** of availability — comparable to Chubby, which Google measured at 99.99958% for 30-second-plus outages.
- Of Spanner incidents, network problems account for **under 8%** of the total. The dominant causes are not partitions; they are bugs, operator error, and multi-component hardware failures — none of which CAP is about.
- And critically: in the period Brewer reports, *there was never an event where a large set of clusters was partitioned from another large set, and a Spanner quorum was never on the minority side of a partition.* Individual datacenters got cut off, bandwidth got under-provisioned, one weird one-directional traffic failure happened — but the catastrophic "majority on the wrong side" scenario simply didn't occur.

Brewer's term for this is "**effectively CA**": technically CP, but with partitions so rare that they are not the thing that limits availability in practice. The honest framing is important for system designers: Spanner's availability comes from Google's network engineering, not from defeating a theorem. If you run a Spanner-like system over a flaky network, you will feel the CP tradeoff acutely, because the partitions will be frequent. The atomic clocks buy you consistency; the private network buys you the *appearance* of availability. They are two separate investments, and a clone that copies only the first gets only the first.

| CAP property | What Spanner does | The honest caveat |
| --- | --- | --- |
| **Consistency** | External consistency (strict serializability), unconditionally. | Never sacrificed, even during a partition. This is the whole point. |
| **Availability** | Better than 5 nines in practice. | Sacrificed on the minority side during a real partition — it's CP. |
| **Partition tolerance** | Survives partitions without losing consistency. | The minority side goes unavailable; the system as a whole keeps the majority serving. |
| The "magic" | None. Just commit-wait + Paxos quorums + a very good network. | Replicate the network investment too, or you get CP without the "effectively CA" comfort. |

## 8. The descendants: Spanner without atomic clocks

> **Senior rule of thumb:** when you can't afford a tight clock bound, you don't get to do commit-wait — and that means you can't cheaply guarantee external consistency for *unrelated* transactions. The open-source Spanner clones are honest about this: they give you serializability, plus a mechanism to handle the clock skew they can't eliminate.

Spanner's atomic clocks are not buyable off the shelf, and most companies can't deploy GPS receivers in every rack. So the obvious question — "can I get Spanner's guarantees on commodity cloud hardware with NTP?" — produced a family of open-source distributed SQL databases, chiefly **CockroachDB** and **YugabyteDB**, that keep most of Spanner's design but replace TrueTime with **Hybrid Logical Clocks (HLC)** plus an assumed maximum clock offset. Understanding precisely what they give up is the most useful thing a senior engineer can take from the Spanner story.

An HLC timestamp has two parts: a **physical** component (kept close to local wall-clock time) and a **logical** counter (a Lamport-style tiebreaker that advances when two events share the same physical millisecond). Nodes exchange HLC timestamps on every RPC and bump their clock to the max they've seen, so HLC tracks causality like a logical clock while staying close to real time like a physical one. This is enough to order *causally related* events correctly. What it cannot do — without TrueTime's tight bound — is cheaply order two *unrelated*, concurrent transactions on different nodes whose physical timestamps fall within the clock skew.

The figure compares the three systems across the dimensions that matter.

![A matrix comparing Spanner, CockroachDB and YugabyteDB across clock source, skew bound, ordering tactic, consistency guarantee, and failure cost, showing Spanner pays commit-wait while the HLC systems pay uncertainty restarts](/imgs/blogs/spanner-truetime-and-external-consistency-8.webp)

The key mechanism is the **uncertainty interval** and the **uncertainty restart** (CockroachDB's term; YugabyteDB calls the user-visible version a *read restart*, error `40001`). When a transaction starts, it picks a provisional timestamp from local wall time and establishes an upper bound by *adding the cluster's maximum clock offset* (CockroachDB's `--max-offset` defaults to 500 ms). Now a read at timestamp `T` is genuinely ambiguous for any value whose timestamp falls in the window `(T, T + max_offset)`: the database cannot tell whether that value was truly written *after* the read started, or whether it just *looks* later because of clock skew. The safe move is to **restart the read at a higher timestamp**, above the value it encountered, so the value is unambiguously in the past. Crucially, the upper bound of the uncertainty window does *not* move on restart, so the window shrinks and the read is guaranteed to terminate.

This is the fundamental difference from Spanner, stated as a slogan: **Spanner pays the uncertainty cost up front and deterministically (commit-wait, ~2ε per write); the HLC systems pay it lazily and reactively (a restart, only when a read actually hits an ambiguous value).** Commit-wait is a fixed small tax on every write; uncertainty restarts are zero cost most of the time but spike under clock skew or write-heavy contention.

And the guarantee is genuinely different. As CockroachDB's own engineers put it: *"While Spanner provides linearizability, CockroachDB only goes as far as to claim serializability."* The uncertainty-interval trick guarantees single-key linearizability and serializable transactions, but it does *not* guarantee external consistency (strict serializability) across the whole key space the way commit-wait does — two causally-unrelated transactions on disjoint keys can be ordered in a way that disagrees with real time, within the skew window. For the vast majority of applications this is invisible and irrelevant; for the rare application that truly needs global external consistency on unrelated data, only the atomic-clock approach delivers it cheaply.

Both clones also add a hard safety valve: if a node detects its clock has drifted beyond the configured maximum offset relative to its peers, it **shuts itself down** rather than risk serving inconsistent data. (CockroachDB self-terminates past `max-offset`; it flags trouble when skew exceeds 80% of the bound against a majority of peers.) This is the HLC equivalent of Spanner evicting a machine whose clock drift exceeds spec — both systems would rather lose a node than lose consistency.

| Dimension | Spanner | CockroachDB / YugabyteDB |
| --- | --- | --- |
| Clock backing | GPS + atomic clocks (TrueTime) | NTP + HLC on commodity hardware |
| Skew handling | Exposed as `ε` (~7 ms worst case), waited out | Assumed bound (default 500 ms), restarted around |
| Per-write cost | Deterministic commit-wait (~2ε) | None — but reads may restart |
| Per-read cost | Lock-free snapshot, may wait for `t_safe` | Lock-free, may *restart* on uncertainty |
| Top-line guarantee | External consistency (linearizable transactions) | Serializability (single-key linearizable) |
| Deployability | Google infrastructure / Cloud Spanner only | Anywhere — any cloud, on-prem, laptop |
| Failure under skew | Slower (ε grows), never wrong | Restart storms; node self-fences past max offset |

Kleppmann's *Designing Data-Intensive Applications* (Chapter 9) frames this beautifully: he uses Spanner as the headline example of *using synchronized clocks for global ordering*, and he is careful to note that the approach is only sound *because* Spanner bounds and exposes the clock uncertainty rather than assuming it away. The HLC databases are what you get when you keep Spanner's architecture but relax that assumption — they remain correct because they account for the unbounded skew (via the uncertainty interval) instead of pretending it's zero. The lesson Kleppmann draws, and the one to take from this whole article: synchronized clocks can give you global ordering, but *only if you treat the synchronization error as a real, bounded quantity in your protocol*. Spanner waits it out; the clones restart around it; nobody who is correct ignores it.

## 9. Global deployment in practice

> **Senior rule of thumb:** in a multi-region deployment, every read-write transaction pays the leader region's round-trip latency, but read-only traffic can stay local. Design your placement so the writes are where your users are, and let reads fan out everywhere.

Putting the pieces together, here is what a planet-spanning Spanner deployment looks like in operation. The figure shows a configuration with a leader region (us-east) and read replicas on other continents.

![A multi-region Spanner deployment where the app in the leader region issues read-write transactions to the local quorum, while apps in other regions serve bounded-stale reads from their local replicas](/imgs/blogs/spanner-truetime-and-external-consistency-9.webp)

Write traffic from the leader region goes to the local Paxos leader, which commits to a quorum (which, in a true multi-region config, includes voters on other continents — so writes pay a cross-region round trip to reach quorum) and then runs commit-wait. Read traffic, however, behaves completely differently depending on the freshness you need. An app in eu-west or asia-1 that issues *bounded-staleness* reads gets served from its *local* replica with no cross-continent hop, because the read timestamp plus `t_safe` is all the coordination required. The asymmetry is the whole game: you concentrate write latency at the leader region and distribute read latency to zero everywhere else.

This drives the central placement decision. A *regional* configuration (all replicas in one region, across its zones) gives you low write latency and survives zone failures but not region failures. A *multi-region* configuration places voting replicas across regions — you survive a whole region going down, and you get globally-low-latency strong reads from the read-only replicas, but every write now pays the inter-region quorum round trip (tens to ~100+ ms across continents). There is no free lunch: external consistency across continents costs a wide-area round trip per write, full stop. What Spanner gives you is that this is the *only* cost — correctness is never on the table — and that reads escape it entirely.

```sql
-- Choosing a multi-region instance config at creation time decides the
-- latency/availability tradeoff. nam3 = US multi-region (leader in us-east,
-- voters in us-east1/us-east4, witness/read replica elsewhere).
--   gcloud spanner instances create prod \
--     --config=nam3 --description="ledger" --nodes=3

-- A strong read issued from a faraway region will route to wherever it can be
-- served consistently; a bounded-stale read stays local. Pick per query:
SET TRANSACTION READ ONLY;            -- strong: pays for global freshness
SELECT SUM(Balance) FROM Accounts;    -- consistent total as of "now"
COMMIT;
```

## Case studies from production and the literature

### 1. AdWords and the two-year re-shard that F1 ended

Before Spanner, Google's advertising backend ran on a hand-sharded MySQL deployment so large that *re-sharding it took roughly two years* of dedicated effort, and the lack of cross-shard transactions pushed consistency logic up into application code. The [F1 system](https://research.google/pubs/f1-a-distributed-sql-database-that-scales/) — built on top of Spanner and presented at VLDB 2013 — replaced it. F1's authors were explicit about the trade they accepted: synchronous cross-datacenter replication via Spanner means *higher commit latency* than local MySQL, on the order of tens of milliseconds. They mitigated it with a hierarchical schema (interleaving, exactly as in §4) so most transactions hit one Paxos group, and with client-side batching. The lesson: Spanner did not make the latency disappear; it made the latency *predictable and the consistency automatic*, and a hierarchical schema is the tool that keeps the common case cheap. The business case was overwhelming — eliminating multi-year re-shards and a whole class of application-level consistency bugs was worth single-digit-millisecond write latency.

### 2. The epsilon spike that throttled commits, not correctness

A recurring operational pattern at Google: a time master in a datacenter becomes unavailable or a network link to it degrades, and `ε` for every machine in that datacenter climbs from ~4 ms toward tens of milliseconds. Commit-wait is proportional to `ε`, so read-write transaction latency rises in lockstep — a 40 ms `ε` means ~80 ms of commit-wait per transaction. The crucial detail is what *didn't* happen: no data was lost, no read returned a stale-but-claimed-fresh value, no external-consistency violation occurred. The system degraded gracefully along the one axis it was designed to degrade on — latency — while the on-call engineer chased a clock-infrastructure problem. This is the payoff of converting a correctness problem into a monitorable latency problem: the symptom is a latency graph, not a corruption incident.

### 3. The hot directory and the placement driver

A common Spanner anti-pattern is a monotonically-increasing primary key (a timestamp, an auto-increment ID), which funnels every new write to the *last* directory — and therefore one Paxos group, one leader, one machine — creating a hotspot. The symptom is a single Paxos group's leader pegged at high CPU while the rest of the cluster idles. The placement driver mitigates by splitting and `movedir`-ing directories off the hot group, but it is reactive and operates on a timescale of minutes, so a sudden write burst still hits the wall. The fix is in the schema: hash or reverse the leading key bits, or use a UUID, so writes spread across directories and thus across Paxos groups from the start. The directory-as-movement-unit design gives the system a lever to rebalance, but it can't out-run a schema that aims all the load at one key range.

### 4. CockroachDB's uncertainty-restart storm under contention

Teams migrating to CockroachDB on standard cloud VMs sometimes hit a wall of `RETRY_SERIALIZABLE` / uncertainty-restart errors under write-heavy, contended workloads — exactly the lazy cost from §8 coming due. With NTP keeping skew around tens of milliseconds and a `--max-offset` of 500 ms, any read that touches a recently-written key inside the uncertainty window restarts. The wrong fix is to crank `--max-offset` up (which widens the window and makes it *worse*); the right fixes are to tighten clock sync (run a good NTP/PTP setup so the *actual* skew is small even if the configured bound is generous), reduce contention via better key design, and let the client retry loop do its job. The contrast with Spanner is instructive: Spanner would have paid this as a few extra milliseconds of commit-wait per write, deterministically; CockroachDB pays it as occasional but spiky restarts, and the spikes correlate with the very contention you least want them during.

### 5. The node that fenced itself on clock drift

Both Spanner and its descendants would rather lose a node than serve inconsistent data, and operators of HLC databases have watched it happen: a VM's clock drifts (a noisy-neighbor hypervisor, a bad NTP peer, a VM migration that stalled the clock), the node detects its skew has blown past the configured maximum offset relative to a majority of peers, and it **self-terminates**. To an operator unaware of the mechanism, this looks like a mysterious crash; in fact it is the consistency guarantee working as designed — the alternative was a node confidently stamping transactions with a clock that's 600 ms off, which would corrupt the global order. Spanner does the analogous thing by *evicting* machines whose measured frequency drift exceeds the worst-case bound derived from component specs. The lesson for anyone running these systems: clock health is a first-class production SLO, and "node committed suicide over clock skew" is a feature, not a bug — but only if you're monitoring clock skew so you can fix it before it fences a node.

### 6. Schema migration at scale without a maintenance window

A team needs to add a column and an index to a multi-terabyte table serving live traffic across regions. On a classic RDBMS this means a maintenance window or a fragile online-DDL tool; on sharded MySQL it means coordinating the change across every shard. Spanner's future-timestamped schema change (from §6) makes it a non-event: the change is registered at a future timestamp, transactions before that timestamp keep using the old schema, transactions after it use the new one, and the cutover is atomic across every Paxos group in the database without ever blocking traffic. The operational lesson is that the *global clock is what makes "atomic across millions of groups" even definable* — without an agreed notion of when the change takes effect, you'd be back to per-shard coordination and a maintenance window.

### 7. The read-after-write surprise on a bounded-staleness read

A subtle application bug: a developer uses bounded-staleness reads everywhere because they're fast and local, then files a bug that "the UI shows the old value right after I save." This is not a Spanner bug — it is the staleness bound doing exactly what it says. A bounded-staleness read with a 15-second bound can legitimately return data from 15 seconds ago, served from a local replica that hasn't yet seen the write. The fix is to use a **strong read** specifically for the read-after-write path (re-displaying the thing the user just edited) while keeping bounded-staleness reads for dashboards and lists. The deeper lesson: Spanner gives you a *menu* of consistency-vs-latency points per read, and choosing the wrong point looks like a database bug but is an application design choice. Strong where the user expects their own write to be visible; stale where they don't.

### 8. The multi-region write-latency reckoning

A team deploys a multi-region Spanner config for region-failure survival, then discovers their write p99 jumped from ~10 ms (regional) to ~80–120 ms (multi-region), because every write now waits for a quorum that spans continents. The reflex is to blame Spanner; the reality is physics — strict serializability across continents *requires* a cross-region round trip per write, and no database can repeal the speed of light. The right responses are architectural: keep write-heavy data in a regional config and only use multi-region for the data that genuinely needs cross-region durability; place the leader region where the write traffic originates; and move read traffic to local read-only replicas (which pay none of this). The reckoning is healthy — it forces teams to separate "data that needs global write consistency" (rare, worth the latency) from "data that's read globally but written regionally" (common, cheap). Spanner makes the cost explicit and per-config, which is exactly what you want when making that call.

### 9. Jepsen-style verification and the value of an honest spec

The distributed-systems testing tool Jepsen has analyzed several Spanner-lineage databases, and the pattern in those reports is telling: the systems that documented their guarantee *precisely* (serializability, not "strong consistency"; the exact behavior of stale reads; the uncertainty-restart semantics) held up under adversarial partition-and-clock-skew testing, while vague marketing claims got falsified. The connection to this whole article: external consistency, serializability, and linearizability are *different, precisely-defined* guarantees (see [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual)), and a database that claims the strong one but implements the weaker one will be caught the moment someone tests it with synchronized clocks and an injected partition. Spanner's lasting contribution is partly cultural: it made the field state its clock assumptions and consistency guarantees in checkable terms, which is why its descendants describe themselves as "serializable with bounded clock skew" rather than hand-waving "consistent."

### 10. The single-group fast path that a foreign key broke

A team models a parent/child relationship with a plain foreign key instead of interleaving, scatters the child rows across the key space, and then can't understand why a transaction that reads a parent and updates one child runs at multi-region commit latency (~80 ms) instead of the single-group latency (~10 ms) they expected. The root cause: parent and child landed in *different* directories, hence different Paxos groups, hence the transaction triggered two-phase commit with a cross-group prepare and commit phase. The first wrong hypothesis was "Spanner is slow"; the actual fix was to redefine the child as `INTERLEAVE IN PARENT` so the child rows sit in the same directory as their parent, collapsing the transaction onto the single-group fast path. The lesson generalizes beyond Spanner to every distributed SQL system: *co-locate the data that's transacted together*, or pay a distributed-commit tax on every transaction that spans the boundary. The schema is a performance contract, and interleaving is how you sign it.

### 11. Stale-read amplification behind a cache

A read-heavy service fronts Spanner with an application cache and configures bounded-staleness reads at a generous 60-second bound to maximize cache locality and local-replica serving. Under normal load this is great — most reads never touch the leader and latency is excellent. The trouble appears during a hot-key invalidation storm: when many clients miss the cache simultaneously and all issue 60-second-stale reads, they can collectively observe a *non-monotonic* view across requests (one request sees a value, a later request to a different replica sees an older one, because the two replicas are at different points within the 60-second window). The symptom looked like a cache bug; the root cause was reading at *different* stale timestamps per request with no client-side timestamp pinning. The fix is to pin a single read timestamp for a logical session (exact-staleness at a chosen `t`) so all reads in that session see one consistent snapshot, reserving fresh bounded-staleness reads for cross-session traffic. The deeper point: Spanner gives you per-read consistency, but *session*-level consistency across many reads is something you compose yourself by reusing a timestamp.

### 12. Migrating off sharded Postgres and discovering the latency floor is real

A team migrates a transactional service from sharded Postgres (with application-level cross-shard coordination) to a distributed SQL database, expecting both better consistency *and* equal-or-better latency. Consistency improves immediately — cross-shard transactions that were previously best-effort become truly atomic. But write p99 rises, because the old system "committed" locally on one shard and resolved cross-shard conflicts lazily (sometimes incorrectly), whereas the new system pays a real quorum round trip and commit-wait (or an uncertainty interval) to make every commit genuinely durable and ordered. The reckoning is that the old latency was *cheating* — it was fast because it wasn't actually providing the guarantee. Once the team accepted that strict serializability has a non-negotiable round-trip floor, the productive work was reducing the *number* of distributed transactions (better key design, more single-group operations) rather than trying to make distributed commit free, which physics forbids. This is the most common and most important lesson in the whole space: you cannot make a correct distributed commit as fast as an incorrect local one, so make fewer distributed commits.

## When to reach for Spanner (or a Spanner-like DB), and when not to

> The question is almost never "is Spanner good" — it is "does my problem actually need external consistency at global scale, and am I willing to pay the latency floor that physics imposes on that guarantee?"

**Reach for Spanner / Cloud Spanner when:**

- You need **SQL + ACID transactions** *and* horizontal scale beyond what a single primary can serve, and you've outgrown read replicas and manual sharding. This is the core sweet spot.
- You genuinely need **strong consistency across regions** — financial ledgers, inventory, anything where a stale read is a correctness bug, not a UX annoyance — and you can tolerate tens of milliseconds of write latency to get it.
- You want to **stop re-sharding by hand.** Automatic, online, directory-granularity rebalancing is worth a lot if your data is growing and re-shards currently consume engineering quarters.
- You're already on **Google Cloud** (Cloud Spanner gives you TrueTime without owning the atomic clocks) or you can run a Spanner-like system (CockroachDB/YugabyteDB) and accept serializability-with-bounded-skew instead of true external consistency.
- Your read traffic dominates and can be served from **local replicas** with bounded staleness — then most of your queries pay near-zero latency and only writes pay the consistency tax.

**Skip Spanner (it's overkill or a mismatch) when:**

- A single Postgres/MySQL primary with read replicas comfortably handles your load. Most applications never outgrow this, and Spanner's per-write latency floor and operational complexity are pure cost if you don't need the scale.
- Your workload is **analytical** (large scans, aggregations, columnar). Spanner is an OLTP system; reach for a warehouse (BigQuery, Snowflake) for analytics — running big scans against your transactional Spanner is the wrong tool.
- You can tolerate **eventual consistency** and want maximum write availability and lowest latency — then an AP system (Cassandra, DynamoDB in its eventually-consistent mode) is a better fit, because you're not paying for a guarantee you don't need.
- You need **sub-millisecond writes** and your data fits comfortably in one region/one machine. Commit-wait plus quorum replication has a latency floor that a single-node or single-region in-memory store will always beat.
- You'd be deploying a Spanner clone over a **flaky network** and expecting "effectively CA." Without Google's network engineering, the CP tradeoff bites — you'll see real unavailability during partitions, and that's the half of Spanner's magic the clones cannot copy.

The through-line of the whole system is one disciplined idea: **expose clock uncertainty as a bounded number, then pay a small, deterministic latency to keep timestamps honest.** Commit-wait is three lines of code, but it is the three lines that turn a continent-spanning pile of independent Paxos groups into a single database that orders the world correctly. Everything else — the universe/zones/spanservers hierarchy, the directory placement, the 2PC-over-Paxos, the lock-free snapshot reads, even the honest CP story — is engineering in service of that one promise. The atomic clocks are famous, but the real invention is the humility of an API that admits it doesn't know exactly what time it is, and a protocol that waits just long enough to be sure.

## Further reading

- [Spanner: Google's Globally-Distributed Database](https://research.google/pubs/spanner-googles-globally-distributed-database-2/) — Corbett et al., OSDI 2012. The primary source; §3 (TrueTime) and §4 (concurrency control, commit-wait) are the heart of it.
- [Spanner, TrueTime and the CAP Theorem](https://research.google/pubs/spanner-truetime-and-the-cap-theorem/) — Eric Brewer. The honest CP-not-CA accounting, with availability numbers.
- [F1: A Distributed SQL Database That Scales](https://research.google/pubs/f1-a-distributed-sql-database-that-scales/) — Shute et al., VLDB 2013. AdWords on Spanner, and the latency/schema tradeoffs of building on it.
- [Spanner: TrueTime and external consistency](https://docs.cloud.google.com/spanner/docs/true-time-external-consistency) — Google Cloud docs. The user-facing definition and the read-type menu.
- [Living without atomic clocks](https://www.cockroachlabs.com/blog/living-without-atomic-clocks/) — CockroachDB. Exactly where the HLC approach diverges from commit-wait, and the uncertainty-restart mechanism.
- *Designing Data-Intensive Applications*, Chapter 9 — Martin Kleppmann. Linearizability, the cost of linearizability, and using synchronized clocks for global ordering, with Spanner as the headline example.
- Sibling posts on this blog: [Raft from scratch](/blog/software-development/database/raft-consensus-from-scratch), [Paxos and Multi-Paxos](/blog/software-development/database/paxos-and-multi-paxos-explained), [time, clocks and ordering](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems), and [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual).

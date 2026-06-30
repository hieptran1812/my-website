---
title: "Multi-Region Database Architecture: Active-Passive vs Active-Active"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Going multi-region is where the speed of light and the CAP theorem stop being abstractions; your active-passive vs active-active choice is a direct trade of latency, consistency, and availability, and this is how to make it per data type."
tags: ["database-scaling", "multi-region", "active-active", "active-passive", "disaster-recovery", "data-residency", "conflict-resolution", "crdt", "failover", "distributed-databases"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 37
---

There is a specific moment in the life of a system when the diagrams stop helping. For years you have drawn the database as a cylinder, the app as a box, an arrow between them, and the arrow has been free — a few hundred microseconds, a rounding error against everything else the request does. Then someone says the word "region." Maybe it is a compliance officer who has read the GDPR and wants EU user data to physically stay in the EU. Maybe it is an SRE who watched `us-east-1` brown out for four hours and never wants the whole company to ride on one AWS region again. Maybe it is a product manager who noticed that users in Singapore see a 400 ms spinner on every page load because every byte they touch round-trips to Virginia. Whatever the trigger, the arrow between the box and the cylinder is no longer free. It now costs 80 milliseconds, or 180, and no amount of indexing, connection pooling, or query tuning will give those milliseconds back. They are the speed of light in fiber, and you do not get to negotiate with physics.

Multi-region is the point where two ideas you have treated as textbook trivia — the [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — become the load-bearing walls of your architecture. CAP says that when the network partitions (and across regions, it *will* partition), you choose consistency or availability, not both. PACELC adds the part everyone forgets: *even when there is no partition*, you trade latency against consistency. Going multi-region forces both halves of that statement onto your roadmap simultaneously. The single biggest decision you will make is the one in this post's title: **active-passive** — one region accepts writes, the others stand by — or **active-active**, where multiple regions accept writes at once. It looks like an infrastructure topology choice. It is actually a consistency-and-latency contract with your users, and once you sign it, it propagates into your data model, your routing layer, your incident runbooks, and your on-call rotation.

![Why go multi-region, and the architectural fork it forces](/imgs/blogs/multi-region-database-architecture-1.webp)

The diagram above is the mental model for the whole article: three distinct pressures — disaster recovery, latency, and data residency — push you across regions, and the moment you arrive, the same speed-of-light constraint that you cannot tune away forces you to pick a side of one fork. Active-passive is the disaster-recovery-first answer: simple, strongly consistent on one primary, but with idle capacity and slow writes for distant users. Active-active is the latency-first answer: every region serves its locals quickly and the system survives a whole-region loss, but you inherit the hardest problem in distributed data, which is two regions writing to the same row at the same time. The rest of this article is a tour of that fork — why you would walk up to it, the physics that defines it, what each branch buys and costs, and how to make the choice *per data type* rather than betting your entire system on one answer.

## Why X is different from a single-region database

Before the topology, the honest table. Most engineers carry a single-region mental model into a multi-region design and get blindsided by the gap between what they assume and what is true.

| You assume (single-region intuition) | The naive multi-region view | The multi-region reality |
| --- | --- | --- |
| Reading your own write back always works | Replication is "fast enough" | A replica in another region trails by tens to hundreds of ms; read-your-writes breaks for the user who just moved regions |
| A write is durable once committed | Committing in one region is enough | If that region is lost before async replication catches up, the *unreplicated tail* is gone — that is your RPO |
| Failover is a button you press | The standby takes over instantly | RTO is minutes if you are good; the standby has stale data and a fenced primary can come back and split-brain |
| Adding regions improves everything | More regions = more availability, full stop | Synchronous cross-region replication makes *every write slower*; you bought availability with latency |
| "The database handles consistency" | Strong consistency is on by default | Across regions, strong consistency means a quorum round-trip per write — 80–180 ms you cannot hide |
| One config serves all data | Treat all tables the same | Sessions, profiles, config, and ledgers have wildly different consistency needs; one global setting is wrong for at least three of them |

Every row of that table is a consequence of one physical fact, so let us start there.

## 1. The physics: cross-region latency is the budget you spend forever

> The first rule of multi-region: you can add machines, but you cannot add light. Every synchronous cross-region hop is a tax you pay on every single write, forever.

Inside a single region — even across availability zones — a network round trip is sub-millisecond to low single-digit milliseconds. That is why your single-region database feels instantaneous: a synchronous replica acknowledgment costs less than the disk fsync that precedes it. Cross-region, the number changes by two to three orders of magnitude. Virginia to Ireland (`us-east-1` to `eu-west-1`) is roughly 75–90 ms round trip on a good day. Virginia to Singapore (`ap-southeast-1`) is 150–230 ms. These are not AWS-specific; they are the speed of light through fiber plus switching and the fact that the cable does not run in a straight line. Light in glass travels about 200,000 km/s, and the great-circle distance from Virginia to Singapore is roughly 15,000 km — so even a perfectly straight, perfectly switched path is ~75 ms one way, ~150 ms round trip. The real number is higher because cables route around continents.

![Cross-region round trips are the budget you spend on every write](/imgs/blogs/multi-region-database-architecture-2.webp)

The table above is the entire economics of multi-region in one picture. The key column is the last one: **what a synchronous write inherits.** If your write has to be acknowledged by a quorum that includes another region — because you want strong consistency and durability across regions — then the write latency floor *is* the cross-region RTT. You can have a 0.5 ms write within a region or an 80 ms write across the Atlantic, and no caching, batching, or hardware upgrade changes the floor. This is the single most important sentence in the post: **synchronous cross-region replication makes the RTT the minimum write latency.** Everything in the active-passive vs active-active debate is a strategy for deciding *who pays that tax, when, and for which data.*

Here is a quick way to feel it. Suppose a checkout flow does six sequential writes (reserve inventory, create order, charge, write receipt, decrement stock, enqueue fulfillment). In a single region at 1 ms each, that is 6 ms of write latency. If those writes are synchronous across the Atlantic at 80 ms each, that is 480 ms — and if they are across the Pacific at 180 ms, it is over a second, before any business logic runs. The fix is never "make the network faster." The fix is architectural: stop making cross-region hops synchronous, or stop making them at all for that data.

```python
# A back-of-envelope latency model. The point is not precision; it is that
# the cross-region term dominates the moment any write must be acknowledged
# by another region synchronously.
INTRA_REGION_MS = 1.0       # quorum ack within a region (across AZs)
CROSS_REGION_MS = {         # round-trip, rough public-cloud numbers
    ("us-east-1", "eu-west-1"):      80,
    ("us-east-1", "ap-southeast-1"): 180,
    ("eu-west-1", "ap-southeast-1"): 165,
}

def write_latency_ms(n_sequential_writes: int, sync_cross_region: bool,
                     pair=("us-east-1", "eu-west-1")) -> float:
    if not sync_cross_region:
        # async replication: the write commits locally; the cross-region
        # hop happens off the request path. RPO risk, not latency cost.
        return n_sequential_writes * INTRA_REGION_MS
    rtt = CROSS_REGION_MS[pair]
    return n_sequential_writes * rtt

print(write_latency_ms(6, sync_cross_region=False))  # 6.0   ms  (async)
print(write_latency_ms(6, sync_cross_region=True))   # 480.0 ms  (sync, Atlantic)
print(write_latency_ms(6, True, ("us-east-1", "ap-southeast-1")))  # 1080.0 ms
```

The model is deliberately crude, but it makes the design pressure obvious. Synchronous cross-region anything is a budget you blow on the first multi-write transaction. The asynchronous path keeps the request fast — but, as the next sections show, it moves the cost from *latency* to *consistency and durability*. There is no free lunch; there is only choosing which bill to pay.

### Second-order consequence: latency compounds with retries and chattiness

The naive number above assumes each write is one round trip. Reality is worse. A chatty ORM that issues a `SELECT` to check existence, then an `INSERT`, then a `SELECT` to read back the generated key, triples the round trips. TCP and TLS handshakes to a freshly opened cross-region connection add their own RTTs before the first byte of query. And a retry on a 180 ms write does not cost you 180 ms — it costs you the timeout you set (often 1–5 seconds) plus the retried write. The lesson that bites teams in their first month of multi-region: **the cross-region RTT is not a one-time cost per request; it is a multiplier on every chatty pattern you got away with in a single region.** Collapse round trips before you go multi-region, or the physics will collapse your p99 for you.

## 2. Active-passive: the disaster-recovery-first design

> Reach for active-passive first. It is the topology that gives you regional disaster recovery with the least new failure modes, and it keeps your data model exactly as it is today.

Active-passive is the conservative answer, and that is a compliment. One region — the primary — accepts all reads and writes. Another region holds a standby replica that continuously replays the primary's write-ahead log but serves no production traffic. If the primary region is lost, you promote the standby and redirect traffic. This is how the overwhelming majority of "multi-region" production systems actually run, because it solves the disaster-recovery problem (the one that gets you fired) without touching the consistency model (the one that is genuinely hard).

![Active-passive: one live region, one warm standby](/imgs/blogs/multi-region-database-architecture-3.webp)

The figure makes the asymmetry concrete. The primary region carries the entire load: all reads, all writes, and the latency penalty for any user who is far from it. The standby region does exactly one thing in steady state — replay the log — and serves nothing. The amber and red on the standby side are the two costs you accept in exchange for simplicity: a far user pays the full cross-region RTT on every write because there is only one place to write, and the standby is idle capacity you pay for but cannot use. The red box, "RPO = unreplicated tail," is the durability cost we will dissect under failover.

The crucial design decision inside active-passive is **synchronous vs asynchronous replication to the standby**, and it is the same CAP/PACELC trade in miniature:

| | Synchronous replication | Asynchronous replication |
| --- | --- | --- |
| Write latency | Primary write waits for standby ack: + full cross-region RTT (80–180 ms) | Primary commits locally: intra-region latency (~1 ms) |
| RPO (data loss on region failure) | Zero — standby has every committed write | Non-zero — you lose the unreplicated tail (the writes in flight) |
| Availability during standby outage | Primary writes *block* if the standby is unreachable (or you fall back to async) | Primary unaffected — standby lag just grows |
| Typical use | Financial systems that cannot lose a committed write | Almost everyone else |

Almost everyone runs **asynchronous** replication to the standby, accepting a small RPO (seconds of potential data loss) in exchange for keeping writes fast and not coupling the primary's availability to the standby's reachability. Synchronous cross-region replication exists, but it converts your write latency into the cross-region RTT — the exact tax we just measured — so it is reserved for data where losing a committed write is unacceptable and the latency hit is tolerable.

Here is what an active-passive write path looks like in application terms — a thin layer that always writes to the current primary and reads locally only if the local region is a replica that is fresh enough:

```python
import time
from dataclasses import dataclass

@dataclass
class Region:
    name: str
    role: str          # "primary" | "standby"
    replication_lag_s: float

class ActivePassiveRouter:
    """All writes go to the primary region. Reads can be served locally
    only when the caller tolerates the standby's replication lag."""
    def __init__(self, regions: list[Region]):
        self.regions = {r.name: r for r in regions}

    def primary(self) -> Region:
        return next(r for r in self.regions.values() if r.role == "primary")

    def route_write(self, local_region: str):
        primary = self.primary()
        if local_region == primary.name:
            return primary, 0.0           # local write, cheap
        # far user: the write must cross to the primary region
        return primary, CROSS_REGION_MS_lookup(local_region, primary.name)

    def route_read(self, local_region: str, max_staleness_s: float):
        local = self.regions[local_region]
        if local.role == "primary":
            return local                  # always fresh
        if local.replication_lag_s <= max_staleness_s:
            return local                  # standby is fresh enough — read local
        return self.primary()             # too stale — pay the RTT for freshness
```

The interesting line is the last one in `route_read`: a standby can serve reads *only* when the application explicitly says how much staleness it tolerates. This is the active-passive version of the read-your-writes problem we covered in depth in [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) — the same lag, now amplified by a cross-region link instead of an intra-region one.

### Second-order consequence: the idle standby is more expensive than it looks

The obvious cost of active-passive is that you pay for a whole region of capacity that serves no traffic. The non-obvious cost is **drift**. A standby that only replays a log is never exercised under production load until the day you fail over to it — which is the worst possible day to discover that its instance type is undersized, its connection limits are wrong, its caches are cold, its DNS TTLs are too long, or its IAM roles were never granted the permissions the app needs. Teams that run active-passive well do two things: they periodically *promote the standby for real* (a planned failover drill, not a tabletop exercise), and they let the standby serve a slice of read-only traffic so it is warm and its capacity is proven. An idle standby is a liability you have not tested; a lightly-loaded standby is an asset you have.

## 3. Active-active: low latency and survival, bought with conflict

> Active-active is not "active-passive with both sides on." It is a fundamentally different consistency contract, and the price of admission is having an answer for what happens when two regions write the same row in the same instant.

In active-active, multiple regions accept writes simultaneously. A user in Frankfurt writes to the EU region; a user in Virginia writes to the US region; both commit locally at single-digit-millisecond latency; and the regions replicate to each other in the background. The payoff is enormous: local write latency for every user regardless of geography, and — because no single region is *the* writer — the system keeps accepting writes even if an entire region disappears. This is the topology behind globally interactive products where a 180 ms write would be unacceptable and a regional outage must be invisible.

![Active-active: low local latency, but writes can collide](/imgs/blogs/multi-region-database-architecture-4.webp)

The figure shows the happy path on the left — two regions each accepting a local write in under 2 ms — and the hard part on the right. Bi-directional replication carries each region's writes to the other, and the moment two regions modify the *same row* before that replication has reconciled them, you have a **conflict**. There is no global clock to say which write "really" happened first, no single authority to ask, and the two regions genuinely disagree about the current value. This is not a bug you can fix; it is the defining property of accepting writes in more than one place. Every active-active system is, at its core, a conflict-resolution strategy with a database attached. The options form a spectrum from "lossy but trivial" to "lossless but constrained":

| Strategy | How it resolves a conflict | What you give up | When it fits |
| --- | --- | --- | --- |
| **Last-writer-wins (LWW)** | Highest timestamp wins; the other write is silently discarded | Correctness — you *lose* a real write, and clock skew decides the winner | Caches, presence, "last seen" — data where loss is acceptable |
| **Home-region per record** | Route every write for a record to its owning region; no concurrent writers exist | Local write latency for records owned elsewhere | The default for most business data (see §5) |
| **CRDTs** | Data types that mathematically merge (counters, sets, sequences) with no lost updates | Expressiveness — only certain shapes are CRDTs; rich invariants don't fit | Counters, collaborative docs, shopping carts, presence sets |
| **Multi-leader + app merge** | Surface both versions; application logic (or a human) merges | Complexity — you write and test merge code per data type | Documents, configs, anything with a meaningful "merge two edits" |

The two endpoints are worth internalizing. Last-writer-wins is seductive because it is one line of code and it always "resolves" — but it resolves by *throwing away data*, and the write it keeps is chosen by whichever node's clock was further ahead, which is not a property your product manager will enjoy explaining. At the other end, [CRDTs](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — conflict-free replicated data types — are the genuinely beautiful answer: a grow-only counter, an observed-remove set, or a sequence CRDT merges concurrent updates *deterministically and without loss*, because the data type's merge operation is commutative, associative, and idempotent. The catch is that not everything is a CRDT. A bank balance with an overdraft rule is not a CRDT; a uniqueness constraint is not a CRDT. Where the merge is mechanical, CRDTs are magic; where it requires a business decision, you are back to multi-leader with explicit merge logic.

Here is a worked CRDT example — a grow-only counter (G-Counter) that two regions increment concurrently and then merge with zero lost increments:

```python
class GCounter:
    """A grow-only counter CRDT. Each region keeps its own per-region tally;
    the merged value is the sum of per-region maxima, so concurrent
    increments in different regions never overwrite each other."""
    def __init__(self, region: str):
        self.region = region
        self.counts: dict[str, int] = {}

    def increment(self, n: int = 1) -> None:
        self.counts[self.region] = self.counts.get(self.region, 0) + n

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> None:
        # take the per-region maximum — commutative, associative, idempotent
        for region, c in other.counts.items():
            self.counts[region] = max(self.counts.get(region, 0), c)

# us-east and eu-west both increment "likes" on the same post concurrently
east = GCounter("us-east"); east.increment(3)   # +3 in the US
west = GCounter("eu-west"); west.increment(5)    # +5 in the EU
# replication carries each region's state to the other; both merge
east.merge(west); west.merge(east)
assert east.value() == west.value() == 8         # no lost updates, converges
```

Contrast that with last-writer-wins on the same scenario: if the post had a single `likes = N` integer and both regions wrote `likes = old + delta`, LWW would keep one write and discard the other — the counter would land on 3 or 5 instead of 8. The CRDT converges to the correct total because it was *designed* so that concurrent increments commute. That is the whole game: in active-active, you either choose data types that merge cleanly, route around concurrency entirely (home-region), or accept lost writes (LWW). There is no fourth door.

### Second-order consequence: active-active does not give you a stronger consistency model — it gives you a weaker one, faster

The trap engineers fall into is believing active-active is "better" because it is more available and lower latency. It is, on those axes. But you have traded *down* on consistency: a single-primary system is linearizable for that data by construction, while an active-active system is, for any data with real conflicts, eventually consistent at best. The art is not pretending otherwise. It is deciding which data can live with eventual consistency (most of it) and carving out the slice that genuinely cannot (§5) so it never goes active-active at all.

## 4. Failover: the part everyone underestimates

> Failover is not a feature you turn on; it is a distributed-systems problem you solve, and the failure mode you must design against is not the dead region — it is the dead region coming back to life.

In active-passive, the whole architecture is a bet that you can promote the standby quickly and correctly when the primary region fails. Two numbers govern the bet. **RTO** (recovery time objective) is how long the failover takes — how many minutes of downtime your users see. **RPO** (recovery point objective) is how much data you lose — the unreplicated tail that the asynchronous link had not yet carried to the standby when the primary died. With async replication, RPO is non-zero by definition: every write that committed on the primary in the last few seconds but had not yet been replayed on the standby is *gone* the instant the primary region is unreachable. You cannot reduce RPO to zero without synchronous replication, and synchronous replication costs you the cross-region RTT on every write. That is the trade, stated one final time.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="Active-passive failover: the primary region fails, traffic reroutes through DNS to the standby region, which is promoted to primary" style="width:100%;height:auto;max-width:820px">
<title>Region failover: primary dies, DNS reroutes clients to the standby, which is promoted</title>
<style>
.mr-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mr-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mr-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.mr-prim{fill:var(--accent,#6366f1);opacity:.16;stroke:var(--accent,#6366f1);stroke-width:2}
.mr-dead{fill:#ef4444;opacity:.14;stroke:#ef4444;stroke-width:2}
.mr-clients{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mr-pkt{fill:var(--accent,#6366f1)}
.mr-dns{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes mr-aDie{0%,30%{opacity:1}45%,100%{opacity:0}}
@keyframes mr-aDead{0%,30%{opacity:0}45%,100%{opacity:1}}
@keyframes mr-bWarm{0%,45%{opacity:0}60%,100%{opacity:1}}
@keyframes mr-bWait{0%,45%{opacity:1}60%,100%{opacity:0}}
@keyframes mr-flowA{0%{transform:translateX(0);opacity:0}8%{opacity:1}22%{transform:translateX(180px);opacity:1}30%{opacity:0}100%{opacity:0}}
@keyframes mr-flowB{0%,55%{transform:translate(0,0);opacity:0}62%{opacity:1}85%{transform:translate(180px,170px);opacity:1}92%,100%{opacity:0}}
@keyframes mr-reroute{0%,48%{opacity:0}58%,90%{opacity:1}100%{opacity:0}}
.mr-aLive{animation:mr-aDie 9s ease-in-out infinite}
.mr-aGone{animation:mr-aDead 9s ease-in-out infinite}
.mr-bPromote{animation:mr-bWarm 9s ease-in-out infinite}
.mr-bStandby{animation:mr-bWait 9s ease-in-out infinite}
.mr-pktA{animation:mr-flowA 9s linear infinite}
.mr-pktB{animation:mr-flowB 9s linear infinite}
.mr-rerouteLbl{animation:mr-reroute 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.mr-aLive,.mr-aGone,.mr-bPromote,.mr-bStandby,.mr-pktA,.mr-pktB,.mr-rerouteLbl{animation:none}.mr-aLive,.mr-pktA{opacity:0}.mr-aGone,.mr-bPromote,.mr-rerouteLbl{opacity:1}.mr-bStandby,.mr-pktB{opacity:0}}
</style>
<text class="mr-clients" x="120" y="40">clients</text>
<circle cx="60" cy="60" r="7" fill="var(--text-secondary,#6b7280)"/>
<circle cx="120" cy="60" r="7" fill="var(--text-secondary,#6b7280)"/>
<circle cx="180" cy="60" r="7" fill="var(--text-secondary,#6b7280)"/>
<rect class="mr-box" x="40" y="95" width="160" height="44" rx="8"/>
<text class="mr-lbl" x="120" y="118">DNS / GSLB</text>
<text class="mr-sub" x="120" y="134">health-checked routing</text>
<rect class="mr-box" x="430" y="60" width="250" height="100" rx="10"/>
<rect class="mr-prim mr-aLive" x="430" y="60" width="250" height="100" rx="10"/>
<rect class="mr-dead mr-aGone" x="430" y="60" width="250" height="100" rx="10"/>
<text class="mr-lbl" x="555" y="100">us-east (primary)</text>
<text class="mr-sub mr-aLive" x="555" y="124">read + write, healthy</text>
<text class="mr-sub mr-aGone" x="555" y="124">region down: failed</text>
<rect class="mr-box" x="430" y="220" width="250" height="100" rx="10"/>
<rect class="mr-prim mr-bPromote" x="430" y="220" width="250" height="100" rx="10"/>
<text class="mr-lbl" x="555" y="258">eu-west</text>
<text class="mr-sub mr-bStandby" x="555" y="282">warm standby, replaying log</text>
<text class="mr-sub mr-bPromote" x="555" y="282">promoted: new primary</text>
<text class="mr-dns mr-rerouteLbl" x="300" y="150">reroute (TTL)</text>
<circle class="mr-pkt mr-pktA" cx="210" cy="110" r="8"/>
<circle class="mr-pkt mr-pktB" cx="210" cy="110" r="8"/>
</svg>
<figcaption>Steady state, traffic flows to the us-east primary. When the region fails, health checks flip DNS and clients reroute to eu-west, which is promoted from warm standby to the new primary.</figcaption>
</figure>

Watch the loop above: in steady state, traffic flows from clients through DNS to the us-east primary. When the region fails, the health check flips DNS, clients reroute to eu-west, and the standby is promoted to the new primary. It looks clean in animation. In production, three things make it hard.

**Promotion is a sequence, not a switch.** To promote a standby you must: confirm the old primary is truly gone (not just slow); stop accepting writes to the old primary; bring the standby to the latest log position it has; flip its role from replica to primary; update the application's view of where the primary is; and reroute traffic. Each step can fail or stall. The dangerous one is the first: *confirming the primary is gone.*

**Split-brain is the catastrophe.** If you promote the standby because the primary *looks* dead — but the primary is actually alive and just partitioned away from your health checks — you now have **two primaries**, both accepting writes, diverging. When the partition heals, you have two conflicting histories and no clean way to merge them. This is the single worst outcome in active-passive, far worse than downtime, because downtime is recoverable and a forked history may not be. The defense is **fencing**: before the new primary accepts a single write, the old primary must be provably prevented from accepting any. We cover the mechanisms — fencing tokens, leases, STONITH — in depth in [split-brain and fencing in distributed databases](/blog/software-development/database/split-brain-and-fencing-in-distributed-databases). The one-sentence version: a monotonic fencing token, incremented on every promotion and checked on every write, lets a returning old primary discover it is stale and refuse to write, instead of corrupting the data.

```python
class FencedPrimary:
    """A returning old primary must present a fencing token >= the current
    epoch to accept writes. After a promotion bumps the epoch, the old
    primary's token is stale, so its writes are rejected — no split-brain."""
    def __init__(self):
        self.current_epoch = 1        # bumped on every promotion, durably

    def promote_new_primary(self) -> int:
        self.current_epoch += 1       # fence: old primary's token is now stale
        return self.current_epoch

    def accept_write(self, token: int, write) -> bool:
        if token < self.current_epoch:
            # this node was primary in an older epoch and has been fenced
            raise StaleLeaderError(
                f"token {token} < epoch {self.current_epoch}; refusing write")
        return self._commit(write)

    def _commit(self, write) -> bool:
        return True

# old primary held token=1; a promotion bumps the shared epoch to 2
coordinator = FencedPrimary()
new_epoch = coordinator.promote_new_primary()      # 2
# the old primary comes back and tries to write with its stale token
try:
    coordinator.accept_write(token=1, write="late update")
except StaleLeaderError as e:
    print(e)  # token 1 < epoch 2; refusing write  — split-brain prevented
```

**DNS rerouting is slower than you think.** Flipping traffic from the dead region to the new primary usually rides on DNS (or a global load balancer with health checks). DNS has a TTL, and clients, resolvers, and intermediate caches honor it imperfectly. A 60-second TTL means some clients keep hitting the dead region for a minute or more after promotion. Set TTLs low *before* you need them (the change itself takes a TTL to propagate), and prefer a routing layer with active health checks and fast convergence over raw DNS where you can. The animation's "reroute (TTL)" label is doing a lot of quiet work — that TTL is often the dominant term in your RTO.

### Second-order consequence: the failover you never test is the failover that fails

The grim pattern across post-incident reviews is that the failover machinery — promotion scripts, fencing, DNS automation — was written once, during the project, and never exercised against a real regional loss. Then the real loss happens at 3 a.m., the promotion script has bit-rotted against a since-changed schema or IAM policy, the fencing token store is itself in the dead region, and the team is improvising surgery on a patient under anesthesia. The only defense is **game days**: scheduled, intentional regional failovers in production (or a production-faithful environment), run often enough that the runbook is muscle memory and the automation is known-good. If you have never failed over on purpose, you do not have a failover capability — you have a hope.

## 5. Choosing consistency per data type, not per system

> Stop asking "should *the system* be active-active?" Ask "should *this data* be active-active?" — because the answer is different for sessions, profiles, config, and ledgers, and forcing one answer on all of them is how you get a system that is simultaneously too slow and too inconsistent.

This is the section that separates a senior multi-region design from a junior one. The junior instinct is to pick one consistency model and apply it to the whole database. The senior move is to recognize that a real application contains data with radically different tolerances for staleness and conflict, and to **route each class of data to the consistency model it actually needs.**

![Pick consistency per data type, not per system](/imgs/blogs/multi-region-database-architecture-8.webp)

The matrix above is the decision in one frame. Walk the four rows:

**Session and cart data** is region-local and tolerates eventual consistency happily. A user's session lives where the user is; if they cross an ocean mid-session (rare) and see a slightly stale cart for a moment, the world does not end. Last-writer-wins is fine here — the cost of a lost cart update is a re-click, not a financial discrepancy. Keep this data region-local, replicate it lazily if at all, and never pay a cross-region cost for it.

**User profile data** wants **read-your-writes** within a region but not global strong consistency. The pattern that fits is **home-region per record**: each user has an owning region (usually where they signed up or where they mostly are), all writes to their profile route there, and reads are served from the local replica. Because there is only ever one writer per record, there are *no conflicts* — you have routed around the hard problem entirely. This is the workhorse pattern for active-active business data, and it is the subject of the next section's code.

**Global config** — feature flags, pricing rules, the org-wide settings every region reads constantly and almost never writes — is **read-mostly and replicated everywhere**. The right model is a single writer (config changes are rare and can afford a cross-region write) fanning out to read-only replicas in every region. Reads are local and fast; the rare write pays the cross-region cost, which is fine because writes are rare. This is the one place where "replicate to all regions" is unambiguously correct.

**Financial ledgers** demand **single-region strong consistency or a distributed-SQL system that provides it.** A ledger has invariants — balances cannot go negative, debits must equal credits, an idempotency key must be enforced exactly once — that *cannot* survive eventual consistency or last-writer-wins. You have two honest options: keep the ledger in a single region and accept that ledger writes pay the cross-region RTT for far users (usually acceptable, because correctness dominates latency for money), or adopt a [globally-distributed SQL database](/blog/software-development/database-scaling/globally-distributed-sql-when-its-worth-it) that gives you serializable transactions across regions at the cost of the cross-region RTT baked into commit latency. What you must *not* do is run the ledger active-active with LWW conflict resolution. That is how money disappears.

| Data class | Placement | Consistency target | Conflict policy | Why |
| --- | --- | --- | --- | --- |
| Session / cart | Region-local | Eventual | LWW acceptable | Loss is a re-click; latency matters most |
| User profile | Home-region per record | Read-your-writes | Route to owner (no conflict) | One writer per record dissolves the conflict |
| Global config | Replicated to all | Read-mostly | Single writer | Reads everywhere, writes rare |
| Financial ledger | Single region or distributed-SQL | Strong / serializable | Reject or 2PC, never LWW | Invariants cannot survive eventual consistency |

The meta-lesson: **a multi-region "database" is usually several data stores with different consistency models wearing a trench coat.** The most robust large systems do not run one global active-active cluster for everything; they run region-local stores for session data, home-region routing for business records, replicated read-only config, and a single strongly-consistent system for money — each chosen on its own merits.

## 6. The region-routing layer: write home, read local

> The single most useful piece of infrastructure in an active-active system is a routing layer that knows each record's home region. It turns the hardest problem (concurrent writes) into a non-problem (one writer) for the data that needs it.

Home-region routing is worth its own section because it is the pattern that makes active-active *survivable* for business data. The idea is simple: every record has an owning region, recorded in a routing table (or derivable from the record's key — e.g., user IDs minted in their signup region carry the region in a prefix). Writes for a record are always forwarded to its home region; reads are served from the local replica. Because writes for any given record happen in exactly one place, there are no concurrent writers and therefore no conflicts — you get active-active's local-read latency and regional survivability without active-active's conflict tax.

![Home-region routing: write home, read local](/imgs/blogs/multi-region-database-architecture-7.webp)

The flow in the figure: a request arrives in the EU region, the router looks up the record's home region, and branches. A **read** is served from the local EU replica — fast, possibly slightly stale, which is fine for reads. A **write** is forwarded to the record's home region (here, the US) — slower for this request, but it preserves the invariant that there is only one writer per record. The whole design is a trade: writes to records owned elsewhere pay a cross-region hop, but in exchange you never resolve a conflict, never lose a write, and never reason about CRDTs for this data.

Here is a region-routing layer that implements exactly this — write to the home region, read locally — with the home region resolved from the record key:

```python
from dataclasses import dataclass

@dataclass
class RouteResult:
    target_region: str
    is_local: bool
    est_latency_ms: float

class RegionRouter:
    """Routes writes to a record's home region and reads to the local
    replica. Home region is encoded in the key prefix (e.g. 'us:42'),
    so resolution needs no extra lookup on the hot path."""
    def __init__(self, local_region: str, rtt_ms: dict[tuple[str, str], float]):
        self.local_region = local_region
        self.rtt_ms = rtt_ms

    def home_region(self, key: str) -> str:
        # key format "<region>:<id>"; the writer region is baked into the key
        return key.split(":", 1)[0]

    def _rtt(self, to: str) -> float:
        if to == self.local_region:
            return 1.0
        pair = (self.local_region, to)
        return self.rtt_ms.get(pair) or self.rtt_ms[(to, self.local_region)]

    def route_write(self, key: str) -> RouteResult:
        home = self.home_region(key)
        # writes ALWAYS go home — this is what eliminates conflicts
        return RouteResult(home, home == self.local_region, self._rtt(home))

    def route_read(self, key: str, require_fresh: bool = False) -> RouteResult:
        if require_fresh:
            # read-your-writes for a record you just wrote: go to its home
            home = self.home_region(key)
            return RouteResult(home, home == self.local_region, self._rtt(home))
        # default: read the local replica, accept bounded staleness
        return RouteResult(self.local_region, True, 1.0)

rtt = {("eu-west-1", "us-east-1"): 80.0}
router = RegionRouter(local_region="eu-west-1", rtt_ms=rtt)

# a EU-local user's record: both read and write are local and fast
print(router.route_write("eu-west-1:1001"))  # RouteResult(eu-west-1, True, 1.0)
print(router.route_read("eu-west-1:1001"))   # RouteResult(eu-west-1, True, 1.0)

# a US-owned record being edited from the EU: the write pays the cross hop,
# but there is still exactly one writer, so no conflict ever occurs
print(router.route_write("us-east-1:42"))    # RouteResult(us-east-1, False, 80.0)
print(router.route_read("us-east-1:42"))     # RouteResult(eu-west-1, True, 1.0)  (stale-ok)
```

The two lines to internalize are the last two pairs. A record owned in the local region is fast for both reads and writes. A record owned elsewhere reads locally (fast, stale-tolerant) but writes home (slow, conflict-free). The `require_fresh` flag on reads is the read-your-writes escape hatch: right after you write a record, if you must read your own write back consistently, you route the read to the home region too, paying the hop for correctness exactly when you need it. This is the same token/freshness machinery from [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas), lifted to the cross-region scale.

### Second-order consequence: rebalancing home regions is its own migration

Home-region routing is clean until a user moves — relocates countries, or your traffic analysis shows their records would be better owned elsewhere. Changing a record's home region is not a config flip; it is a small data migration: you must stop writes to the old home, drain in-flight replication, move ownership, fence the old home from accepting writes for that record, and update the routing table — all without losing a write or creating a window where two regions both think they own the record. Systems that do this well treat home-region reassignment as a first-class, throttled, resumable operation, not an afterthought. It is the cross-region cousin of the resharding problem.

## Case studies from production

The patterns above are abstract until they meet a real outage or a real product constraint. Here are named systems and incidents — with rough numbers, because the public details are rough — that show the active-passive vs active-active trade playing out under load.

### 1. The us-east-1 dependency that took down half the internet

`us-east-1` is AWS's oldest and largest region, and for years it has also been the region where global control-plane services and countless companies' single-region databases live. When `us-east-1` has a bad day — and it has had several multi-hour incidents — the blast radius is enormous, because so many systems treated it as "the" region rather than "a" region. The lesson is not that AWS is unreliable; it is that **a single-region architecture inherits the availability of its single region**, and `us-east-1`'s availability, while high, is not 100%. The teams that rode out those incidents gracefully were the ones running active-passive with a standby in another region and a *tested* failover. The teams that did not had elegant single-region architectures and a multi-hour outage. The first multi-region investment most companies should make is the boring one: a warm standby in a second region and a quarterly failover drill.

### 2. GitHub's 24-hour split-brain after a network partition

In October 2018, GitHub suffered a 24-hour degradation that is the canonical real-world split-brain story. A brief network partition between their East and West Coast sites caused an automated failover to promote a database in the West while the East-coast primary had also taken writes. When connectivity returned, the two sides had diverging write histories that could not be cleanly merged, and engineers spent a full day reconciling data by hand to avoid losing customer writes. The root lesson is exactly §4's warning: **the dangerous failure is not the dead primary, it is the primary that looks dead but is not.** Their remediation centered on making failover require stronger confirmation and human checkpoints rather than firing automatically on a transient partition — a direct application of fencing and the principle that promoting too eagerly is worse than promoting too slowly.

### 3. Figma's home-region routing for collaborative documents

Figma's product is real-time collaborative design — multiple users editing the same document, which is precisely the active-active conflict problem at its hardest. Their architecture leans heavily on the home-region pattern and on conflict-handling data structures: a document (or a shard of documents) has an owning location that serializes writes, so concurrent edits to the *same* object are ordered in one place rather than colliding across regions. This is the §6 pattern applied at the document grain — turn the conflict problem into a single-writer problem by routing all writes for an object to one place — combined with CRDT-like merge for the genuinely concurrent fine-grained edits. The takeaway: even a product whose entire value proposition is concurrent editing does not run a naive global active-active store; it routes ownership to dissolve most conflicts and uses mergeable structures only where concurrency is irreducible.

### 4. Notion's per-workspace region pinning for data residency

Notion's move to offer EU data residency is a clean example of the compliance driver from the mental-model figure. Rather than running one global active-active cluster, they pin a workspace's data to a region — a European customer's workspace lives in EU infrastructure and its data does not leave. This is geo-sharding by tenant: the shard key is the workspace, the placement is driven by residency requirements, and the consistency model stays simple because each workspace has a single home. It shows that "multi-region" is often not about latency or availability at all — it is about *where the bytes physically sit*, and the cleanest way to satisfy residency is region-pinned data with a single home per tenant, not active-active replication that would by definition copy data across the boundary you are trying to respect.

### 5. DynamoDB Global Tables and the last-writer-wins reality

AWS's DynamoDB Global Tables offer multi-region active-active with a documented conflict-resolution policy: last-writer-wins, decided by timestamp. This is not a flaw — it is an honest, explicit contract, and for the data DynamoDB Global Tables is typically used for (user state, session-like data, presence, caches) LWW is the right trade. The cautionary tale is the teams that adopt Global Tables for its operational simplicity and *then* put data with real invariants into it — counters that should sum, balances that should never lose an update — and are surprised when concurrent writes from two regions silently drop one. The system did exactly what it documented. The lesson is §3 and §5 fused: **read your active-active store's conflict policy before you choose what data to put in it, and never put invariant-bearing data behind LWW.**

### 6. Spanner and CockroachDB: paying the RTT for global strong consistency

Google's Spanner and CockroachDB take the opposite bet from DynamoDB Global Tables: they provide *externally consistent* (Spanner) or *serializable* (CockroachDB) transactions across regions, which means a globally consistent write must coordinate a quorum that may span regions — and that coordination costs the cross-region RTT. Spanner famously uses TrueTime (GPS and atomic clocks giving bounded clock uncertainty) to order transactions globally; the price is that a write waits out the clock uncertainty interval plus the quorum round trip. These systems are the right answer for the financial-ledger row of §5's matrix: when you genuinely need strong consistency across regions, you pay the RTT in commit latency, and a distributed-SQL system makes that trade explicit and correct rather than leaving you to bolt strong consistency onto an eventually-consistent store. The mistake is reaching for them by default — paying global-consistency latency for session data that would have been happy region-local.

### 7. The Netflix multi-region active-active for availability

Netflix is the textbook active-active-for-availability case: they run their critical services active-active across multiple AWS regions specifically so that losing an entire region is a non-event for members. The enabling design choice is that the data behind those services is largely conflict-tolerant — viewing state, recommendations, presence — the session-and-profile end of §5's matrix, not the ledger end. They pair this with aggressive failure testing (the Chaos Monkey lineage) that regularly *evacuates* a region in production to prove the active-active failover actually works. It is the §4 "game day" principle taken to its logical conclusion: the only way to trust a regional failover is to do it on purpose, often. Netflix can run active-active broadly precisely because they chose data models that tolerate it and tested the failover until it was boring.

### 8. The cross-region chatty-ORM latency cliff

A pattern, not a single company, because nearly every team hits it: a service that performed fine single-region falls off a latency cliff the moment its database moves to another region, because an ORM was issuing six to ten round trips per request — lazy-loaded relations, existence checks, read-after-write confirmations — each of which was a microsecond locally and is now 80–180 ms. The p99 does not degrade gracefully; it falls off a cliff, because the round trips are sequential and they multiply by the RTT. The fix is never "add a faster network." It is the §1 second-order lesson: collapse the round trips — eager-load, batch, use a single transactional statement, cache aggressively in-region — *before* the data crosses a region boundary. Teams that profile their round-trip count before going multi-region avoid this; teams that discover it in production spend a sprint rewriting their data access layer under duress.

### 9. The compliance-driven single-home-per-record design

A fintech operating in multiple jurisdictions needed each customer's financial records to stay in that customer's jurisdiction — a residency requirement with legal teeth, not a latency optimization. They could not run active-active (replication would copy regulated data across the boundary) and they could not run a single global region (it would violate residency for everyone outside it). The design that worked is the §5 + §6 combination: each customer's records have a home region fixed by jurisdiction, all writes route there, reads serve from in-region replicas (which never cross the boundary), and the ledger stays single-region-strong within each home. It is multi-region in topology but single-home in data ownership — and it shows that the active-active/active-passive axis and the residency axis are independent: you can be multi-region for compliance while being single-writer-per-record for correctness.

## When to reach for each — and when not to

The whole post collapses into a decision. Here is the comparison one last time, then the rules.

![Active-passive vs active-active, dimension by dimension](/imgs/blogs/multi-region-database-architecture-6.webp)

The matrix is the trade in one frame: active-passive is green where it counts for correctness and operability (strong consistency, no conflicts, lower complexity) and red on far-user write latency; active-active is green on latency and immediate regional survival and red on conflict handling and complexity. Neither is "better." They are answers to different questions.

**Reach for active-passive when:**

- Your primary driver is **disaster recovery** — surviving a whole-region loss — and not local latency for distant users.
- Your data has **invariants** (ledgers, inventory, uniqueness) that cannot tolerate concurrent-writer conflicts.
- You want multi-region resilience **without changing your data model or consistency assumptions** — the standby is your current database, in another region.
- Your write volume from far regions is modest, so the cross-region write penalty for those users is acceptable.
- You can commit to **regular, real failover drills** — active-passive's correctness depends entirely on the failover actually working.

**Reach for active-active when:**

- **Local write latency** for users in every region is a hard product requirement — a 180 ms write is unacceptable.
- The system must keep **accepting writes through a full regional outage**, with no failover gap visible to users.
- Your data is **largely conflict-tolerant** (sessions, presence, viewing state, counters that can be CRDTs) — or you can apply **home-region routing** to dissolve conflicts for the data that is not.
- You have the **engineering maturity** to own conflict resolution, test it adversarially, and run regular region-evacuation game days.

**Skip multi-region writes (stay single-region, or active-passive at most) when:**

- You have not yet **collapsed your round-trip count** — going multi-region with a chatty ORM is a latency cliff waiting to happen (case study 8).
- Your scale does not justify it — a single well-run region with a standby covers the disaster-recovery need for the vast majority of systems, and multi-region writes add failure modes you must staff to operate.
- Your driver is **data residency**, which is usually better solved by **region-pinned, single-home-per-tenant** data than by active-active replication that copies data across the boundary you are trying to respect (case studies 4 and 9).
- You are tempted to run the **whole system** active-active — almost always the right design is per-data-type (§5): region-local sessions, home-region business data, replicated read-only config, single-region-strong money.

The thesis, restated: multi-region is where the speed of light and the CAP theorem stop being textbook theory and start being line items in your latency budget and your incident reviews. Active-passive and active-active are not a religious choice — they are a contract trading latency, consistency, and availability, and the most robust systems refuse to sign one contract for all their data. They route each class of data to the model it needs, pay the cross-region RTT only where correctness demands it, and — above all — actually test the failover before the night they need it.

## Further reading

- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the consistency/availability and latency/consistency trades that multi-region makes concrete.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the replication topologies and CRDTs underlying active-active.
- [Read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) — the read-your-writes and staleness machinery, at single-region scale.
- [Split-brain and fencing in distributed databases](/blog/software-development/database/split-brain-and-fencing-in-distributed-databases) — the fencing mechanisms that keep failover from forking your history.
- [Globally distributed SQL: when it's worth it](/blog/software-development/database-scaling/globally-distributed-sql-when-its-worth-it) — when paying the cross-region RTT for serializable global transactions is the right call.

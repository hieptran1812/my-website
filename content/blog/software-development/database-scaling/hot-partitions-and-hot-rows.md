---
title: "Hot Partitions and Hot Rows: Breaking Skew at Scale"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A perfectly sharded system still melts when load piles onto one partition or one row — detecting and breaking that skew is its own engineering discipline, and this is the playbook."
tags: ["database-scaling", "hot-partition", "sharding", "skew", "sharded-counter", "fan-out", "distributed-systems", "caching", "write-amplification", "system-design"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 33
---

The first time I watched a sharded cluster fall over, the dashboard told a story that made no sense for a full minute. The cluster-wide CPU average was 23%. The cluster-wide write throughput was nowhere near the rated ceiling. Every capacity number we had been trained to watch was green. And yet writes were timing out, the on-call channel was on fire, and a feature launch was visibly degrading in production. It took us far too long to pull up the *per-shard* graph, where the truth was obvious and ugly: one shard was pinned at 100% CPU pushing nine thousand writes a second, and the other seven were idling under a fifteenth of that. The average was a lie. The cluster was not overloaded. *One partition* was on fire and the rest were asleep.

That is the entire subject of this post. Sharding solves the problem of *total* load exceeding one machine. It does nothing — by itself — for the problem of load *concentrating* on one partition, one row, or one key. Those two problems look similar on a slide and are completely different in production, and the second one is sneakier because every aggregate metric you instinctively reach for averages it away. The thesis is blunt: **a perfectly even shard map does not guarantee even load, and detecting and breaking skew is a discipline separate from sharding itself.**

![A perfectly even 8-shard map still melts when one partition absorbs nearly all the load](/imgs/blogs/hot-partitions-and-hot-rows-1.webp)

The diagram above is the mental model for everything that follows. Eight shards, a textbook-even key distribution, and the system is still down — because shard 3 is absorbing 8,900 writes a second while shards 0, 1, 2, and 4 through 7 sit at one to two hundred. No amount of *more shards* fixes this. If you double the shard count, shard 3's load lands on whatever shard the hot key now hashes to, and you have the same picture with sixteen boxes instead of eight. The fix is never "add capacity"; the fix is to stop the load from concentrating in the first place. This post is the playbook for doing that.

If you have not read [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key), read it first — a bad shard key is the single most common *cause* of the skew we are about to debug, and the two posts are halves of the same lesson. This one assumes you have already sharded and are now staring at a lopsided load graph wondering why.

## Why skew is different from overload

Most engineers carry one mental model for "the database is slow": there is too much load, so add capacity. That model is correct for overload and actively misleading for skew. Here is the assumption-versus-reality table I wish I'd had taped to my monitor that first night.

| Common assumption | The reality with a hotspot |
| --- | --- |
| "CPU average is 23%, so we have headroom." | The average is the enemy. One shard at 100% and seven at 10% averages to ~21% — and the one at 100% is dropping requests right now. |
| "We sharded, so load is spread across N machines." | Sharding spreads the *key space* evenly. It says nothing about whether *traffic* is even. One popular key routes all its traffic to one shard. |
| "Adding shards will fix it." | More shards re-distributes the *cool* keys you didn't have a problem with. The hot key still lands wholesale on one shard. |
| "It's a capacity problem, page the infra team." | It's a data-modeling problem. The fix lives in how you key, write, and read — not in the instance type. |
| "Auto-scaling will absorb the spike." | You cannot auto-scale a single row. A row lives on one primary; horizontal scaling cannot split it. |

The reason skew is mechanically different is that the unit of overload is no longer "the cluster." It is the smallest indivisible thing your traffic concentrates on — and that thing cannot be split by adding machines. A row lives on exactly one primary. A logical key hashes to exactly one shard. A partition is owned by exactly one replica set. When load piles onto one of those indivisible units, you have run out of the one resource sharding was supposed to give you: the ability to spread work across machines. The work refuses to spread.

> Sharding is a bet that your load is as evenly distributed as your keys. Skew is what happens when you lose that bet on a single key.

## The anatomy: hot key, hot row, hot partition

**Before you can fix a hotspot you have to name which of three nested scales it lives at**, because the remedy is different for each. The words get used interchangeably in incident channels, and that imprecision costs hours — someone proposes a fix for a hot partition when the real problem is a single hot row, and the fix does nothing.

![A hotspot lives at one of three nested scales — key, row, partition — and each demands a different remedy](/imgs/blogs/hot-partitions-and-hot-rows-2.webp)

The tree above is the vocabulary. Read it top to bottom as a zoom-out:

- **A hot key** is a *logical* name that receives a disproportionate share of operations: `post:42:likes`, `user:celebrity:timeline`, `inventory:ps5`. The key is an application-level identifier. It is hot when one of your millions of possible keys takes a meaningful fraction of all traffic. This is the level at which you usually have the most leverage, because you control how the key maps to storage.
- **A hot row** is a single *physical* record that everyone writes to or reads from — a counter row, a status flag, a leaderboard head. A hot row is a hot key that has reached the storage engine, and now you are fighting row-level mechanics: lock contention, write-ahead-log serialization, MVCC version churn, and last-writer-wins update storms. You cannot split one row across machines.
- **A hot partition** is the largest scale: a whole shard or partition whose CPU, IO, and replicas are saturated because the keys it owns are collectively hot — or because *one* key it owns is hot enough to take the whole machine down with it. This is the level the infrastructure sees first, because it shows up as a melted box.

The nesting matters: a single hot key produces a hot row, and a hot row (or a cluster of warm ones) produces a hot partition. So when you see a hot partition, the diagnostic question is always "is this one hot key, or genuinely many warm keys that happen to co-locate?" The answer decides whether you reach for write-sharding (one key) or for re-keying and rebalancing (many keys). Diagnose the scale before you reach for a tool.

## Where skew comes from

**Skew is not random; it is born from a small number of recognizable data-modeling and access patterns, and each one has a signature you can learn to spot on sight.** If you can name the cause, you have usually named the fix.

![Each cause concentrates load on a different unit, on a different hot path, with a different classic trigger](/imgs/blogs/hot-partitions-and-hot-rows-3.webp)

The matrix above sorts the five usual suspects by what they concentrate on and which path — read or write — they melt. Walk through them, because the rest of the post is organized around breaking each one:

1. **The celebrity user.** One account has a hundred million followers; one product has a hundred thousand concurrent watchers. Every read of that entity routes to the one shard that owns it. This is overwhelmingly a *read* hotspot, and it is the canonical "fan-out" problem we will spend a whole section on.
2. **The sequential key.** You sharded by an auto-increment `id` or by `created_at`. Every brand-new write has the highest key, which routes to the *newest* shard. You have built a system that, at the write path, can only ever use one machine at a time — the rest hold cold history. Pure *write* hotspot.
3. **The single counter row.** A global "total views," "likes on this post," or "items sold" counter is one row, and every event does a read-modify-write against it. Thousands of concurrent increments serialize on one row lock and one WAL. The most concentrated possible *write* hotspot: one row.
4. **The time-based partition.** You partition events by day. "Today" takes essentially 100% of both reads and writes; every partition for a past day is frozen, cold storage. The hot partition *moves* every midnight, which makes it especially insidious — it is never the same shard twice.
5. **The flash-sale item.** A single SKU goes on sale at noon. For ten minutes, one `product_id` absorbs the read traffic of your entire catalog plus a thundering herd of writes against its inventory counter. A short, violent *both*-paths hotspot.

Notice that these are not exotic. Every one of them is something a reasonable engineer designs on purpose — sequential IDs are convenient, a single counter is obvious, partitioning by day is clean. Skew is the predictable second-order cost of choices that look correct in isolation. The discipline is recognizing the cost *before* the launch that triggers it.

## Detecting skew before it pages you

**The hardest part of a hotspot is not fixing it — it is seeing it, because every dashboard you built for overload averages it into invisibility.** You need metrics whose *unit* is the shard or the key, never the cluster.

![Combining a per-key access histogram with per-shard meters localizes a hotspot that averages cannot reveal](/imgs/blogs/hot-partitions-and-hot-rows-4.webp)

The graph above is the detection pipeline I run in every sharded system, and the key idea is that it has *two parallel arms*. Cluster averages tell you nothing, so you measure two things the average hides: load *per shard*, and operations *per key*. Each arm catches a different failure.

**The per-shard arm** is the cheaper of the two and should always be on. Emit, per shard, the QPS, the p99 latency, and the CPU. Then compute one scalar that compresses the whole cluster into a number you can alarm on: the *skew factor*, the busiest shard divided by the mean shard. A skew factor of 1.0 is perfect balance; 1.5 is worth a look; 4 or more is a hotspot in progress.

```python
def shard_skew(per_shard_qps: list[float]) -> float:
    """Skew factor = hottest shard / mean shard. 1.0 is perfect balance.
    Alarm when this crosses ~2.0 and keeps climbing."""
    if not per_shard_qps:
        return 0.0
    mean = sum(per_shard_qps) / len(per_shard_qps)
    return max(per_shard_qps) / mean if mean else 0.0

# Real numbers from the incident in the intro:
qps = [170, 210, 160, 8900, 190, 150, 220, 180]
print(f"skew factor = {shard_skew(qps):.1f}")   # skew factor = 8.9
```

A skew factor of 8.9 on an 8-shard cluster means one shard is doing nearly nine times its fair share. That single number, alarmed, would have paged us before the average ever moved.

**The per-key arm** answers the next question: *which* key. You cannot keep an exact counter for every key on a firehose — that is unbounded memory — so you do one of two things. Either sample the access log at a low rate and scale the counts back up, or run a streaming heavy-hitters sketch (count-min plus a top-K heap) that uses fixed memory regardless of cardinality. Sampling is the ten-minute version; the sketch is the production version.

```python
import random
from collections import Counter

def top_keys(log_lines, k: int = 20, sample_rate: float = 0.05):
    """Find the k hottest keys from an access log without holding every key.
    We sample at 5% to bound memory on a firehose, then scale counts back up.
    Each log line is assumed to start with the accessed key:  '<key> <op> <ms>'."""
    counts: Counter[str] = Counter()
    for line in log_lines:
        if random.random() > sample_rate:
            continue
        key = line.split(maxsplit=1)[0]
        counts[key] += 1
    # Scale sampled counts back to an estimate of true volume.
    return [(key, round(c / sample_rate)) for key, c in counts.most_common(k)]

# A healthy system: the top key is a small fraction of traffic.
# A hotspot: the top key is 30-90% of traffic. That ratio is the alarm.
```

The diagnostic ratio is *top key over total*. In a healthy system the hottest key might be 0.1% of traffic. When the hottest key is 30%, 50%, 90% of all operations, you have found your hot key by name, and naming it is most of the fix — now you know whether you are looking at a celebrity (`user:celebrity`), a counter (`post:42:likes`), or a sequential write (every key is a fresh `id`, so *no single* key dominates but the newest shard does, which is the tell that you have a *sequential*-key problem rather than a *single*-key one).

> If you only build one thing from this post, build the per-shard skew-factor alarm. It is ten lines and it turns "the cluster is mysteriously down" into "shard 3 is hot" before the pager goes off.

The two arms together localize the problem to a *(shard, key)* pair, and that pair tells you which of the following sections to open.

## Breaking write hotspots

Write hotspots are the harder family, because a write must eventually durably land somewhere, and you cannot cache your way out of a write the way you can a read. The unifying trick across every technique below is the same: **stop forcing many writers to contend on one indivisible thing — spread the writes across N things, and pay a gather cost on read instead.** You are trading a cheap read and an impossible write for a slightly more expensive read and a possible one.

### The sharded counter

The single hot counter row is the purest write hotspot, so it gets the cleanest fix, and it is worth internalizing because the pattern generalizes. The problem: `post:42:likes` is one row, and ten thousand people liking the post at once means ten thousand `UPDATE ... SET likes = likes + 1` statements serializing on one row's lock and one WAL append. Throughput is capped at how fast one row can be updated in series — a few thousand per second on a good day, and every writer past that waits.

The fix is to make the counter *not one row*. Split it into N physical sub-counters. Each increment hits a *random* sub-counter, so writes scatter across N rows that can be updated independently; a read sums all N. The motion below is the whole idea:

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="A sharded counter: each increment lands on a different counter shard in turn, and a read sums all four shards" style="width:100%;height:auto;max-width:820px">
<style>
.hc-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.hc-sum{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2}
.hc-conn{stroke:var(--border,#d1d5db);stroke-width:1.5}
.hc-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.hc-val{font:13px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.hc-op{font:14px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.hc-hi{fill:var(--accent,#6366f1);opacity:.18}
.hc-plus{font:700 18px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
@keyframes hc-sweep{0%{transform:translateX(0)}100%{transform:translateX(620px)}}
.hc-mv{animation:hc-sweep 8s steps(4,end) infinite}
@media (prefers-reduced-motion:reduce){.hc-mv{animation:none}}
</style>
<text class="hc-op" x="360" y="34">write: INCR counter:likes:rand(0..3)</text>
<rect class="hc-box" x="60"  y="90" width="130" height="92" rx="8"/>
<rect class="hc-box" x="215" y="90" width="130" height="92" rx="8"/>
<rect class="hc-box" x="370" y="90" width="130" height="92" rx="8"/>
<rect class="hc-box" x="525" y="90" width="130" height="92" rx="8"/>
<text class="hc-lbl" x="125" y="128">shard 0</text>
<text class="hc-lbl" x="280" y="128">shard 1</text>
<text class="hc-lbl" x="435" y="128">shard 2</text>
<text class="hc-lbl" x="590" y="128">shard 3</text>
<text class="hc-val" x="125" y="158">c0 = 2,201</text>
<text class="hc-val" x="280" y="158">c1 = 2,198</text>
<text class="hc-val" x="435" y="158">c2 = 2,205</text>
<text class="hc-val" x="590" y="158">c3 = 2,196</text>
<g class="hc-mv">
<rect class="hc-hi" x="60" y="90" width="130" height="92" rx="8"/>
<text class="hc-plus" x="125" y="80">+1</text>
</g>
<line class="hc-conn" x1="125" y1="182" x2="125" y2="232"/>
<line class="hc-conn" x1="280" y1="182" x2="280" y2="232"/>
<line class="hc-conn" x1="435" y1="182" x2="435" y2="232"/>
<line class="hc-conn" x1="590" y1="182" x2="590" y2="232"/>
<rect class="hc-sum" x="60" y="232" width="595" height="56" rx="8"/>
<text class="hc-lbl" x="357" y="266">read = c0 + c1 + c2 + c3 = 8,800</text>
</svg>
<figcaption>A sharded counter: each increment hits a random shard so no single row is hot, and a read sums all four shards.</figcaption>
</figure>

In Redis the implementation is a dozen lines, and the only knob is N — the number of shards, which sets your maximum write concurrency before any single key is contended again:

```python
import random
import redis

r = redis.Redis()
N_SHARDS = 16   # max concurrent uncontended writers; larger N => more read fan-out

def incr_likes(post_id: int, by: int = 1) -> None:
    # Scatter: pick a random sub-counter so concurrent writers rarely collide.
    shard = random.randrange(N_SHARDS)
    r.incrby(f"post:{post_id}:likes:{shard}", by)

def read_likes(post_id: int) -> int:
    # Gather: one round trip, sum the N shard values.
    keys = [f"post:{post_id}:likes:{i}" for i in range(N_SHARDS)]
    return sum(int(v or 0) for v in r.mget(keys))
```

The arithmetic of the tradeoff is clean. With N = 16, the maximum sustained write rate to a single logical counter goes up by roughly 16×, because no two writers contend unless they collide on the same sub-counter (birthday-paradox collisions are rare at low concurrency and self-limiting at high). The cost is that a read now does an N-key gather instead of a single GET. For a like counter that is written far more than it is read, or read by a cache that refreshes every few seconds, this is an obvious trade. The same pattern works in any store: Cassandra has native distributed counter columns that do exactly this under the hood; DynamoDB users hand-roll it with a random suffix on the sort key; Postgres users make N rows and `SUM` them.

The one place sharded counters bite you is *exact*, *low-latency* reads. If you need the precise value on every read and N is large, the gather latency and the read amplification can dominate. The escape hatch is to periodically roll the N sub-counters up into a single materialized total in the background, so reads hit one pre-summed row and only the background job pays the gather. That converts an expensive synchronous gather into a cheap asynchronous one — at the cost of the rolled-up total being a few seconds stale.

### Write-sharding an arbitrary hot key

The counter is a special case of a general technique. **Any** hot write key — not just a counter — can be split by suffixing it into N sub-keys, scattering writes across them, and gathering on read. The shape is identical; only the merge step changes.

![Splitting one logical key into N suffixed sub-keys converts a single write hotspot into N cool keys](/imgs/blogs/hot-partitions-and-hot-rows-6.webp)

The before/after above is the move in the abstract. On the left, one key, every write serialized, row-lock contention. On the right, `post:42:likes:{0..15}`, writes scattered, and a read that gathers and merges the sixteen sub-keys. The merge is whatever your data type needs: `SUM` for a counter, list-concatenation for an append-only log, a set-union for a membership set, a max for a high-water mark.

```python
import random

WRITE_SHARDS = 16

def hot_append(key: str, value) -> None:
    # For an append-heavy hot key (an activity feed, an event log), append to a
    # random sub-key. Append is contention-free; there is no read-modify-write.
    suffix = random.randrange(WRITE_SHARDS)
    db.rpush(f"{key}#{suffix}", value)

def hot_read(key: str, limit: int = 50):
    # Gather all sub-keys and merge. For a time-ordered feed, this is an
    # N-way merge of N already-sorted lists -- cheap with a heap.
    import heapq
    lists = [db.lrange(f"{key}#{i}", 0, limit) for i in range(WRITE_SHARDS)]
    return list(heapq.merge(*lists, key=lambda v: v.ts, reverse=True))[:limit]
```

The general principle worth carrying away: **prefer append-then-aggregate over update-in-place for anything hot.** An update-in-place (`SET x = x + 1`) is a read-modify-write that *must* serialize to be correct. An append (`INSERT` a new row, `RPUSH` a new element) has no read step, so it never contends — two appends to the same structure can both succeed without coordination. You move the aggregation to read time, where you have more freedom: you can cache it, batch it, or roll it up offline. This single reframing — turn mutations into appends, turn reads into reductions — is the deepest idea in this whole post, and it is why event-sourced and log-structured systems are so resistant to write hotspots: they never update in place.

### Batching and coalescing

The third write-hotspot lever is to not do the writes at all — or rather, to do fewer, fatter ones. If a hot row receives a thousand increments a second, you do not need a thousand database round trips. **Coalesce** them in the application: accumulate the deltas in memory for a short window — say 200 ms — and flush one `+N` instead of N `+1`s. A thousand single increments become five batched ones per second per app instance.

```python
import threading
from collections import defaultdict

class CoalescingCounter:
    """Accumulate increments in memory; flush a single +N every `interval`.
    Turns thousands of round trips into a handful of fat ones per window."""
    def __init__(self, flush_fn, interval: float = 0.2):
        self._pending: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._flush_fn, self._interval = flush_fn, interval
        threading.Timer(interval, self._flush).start()

    def incr(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._pending[key] += by   # local, lock-cheap, no DB round trip

    def _flush(self) -> None:
        with self._lock:
            batch, self._pending = dict(self._pending), defaultdict(int)
        for key, delta in batch.items():
            self._flush_fn(key, delta)  # one UPDATE ... += delta per key per window
        threading.Timer(self._interval, self._flush).start()
```

Coalescing trades durability and freshness for throughput: an increment buffered in memory is lost if the process crashes before the flush, and the persisted count lags reality by up to one window. For a view counter or a like count, losing a few hundred milliseconds of increments on a crash is completely acceptable, and the throughput win is enormous — you have decoupled the write rate the database sees from the event rate the world generates. For anything where every event must be durable (payments, inventory decrements that can oversell), coalescing is the wrong tool and you stay with the sharded counter or a proper transactional decrement.

## Breaking read hotspots

Read hotspots are the kinder family, because reads are idempotent and cacheable — you have options a write never gives you. **The senior rule is to put as many layers as possible between the hot read and the one machine that owns the data, so that almost no read ever reaches it.**

![Caching plus replica fan-out keeps a celebrity row's reads off the primary even at thousands per second](/imgs/blogs/hot-partitions-and-hot-rows-7.webp)

The graph above is the read-hotspot defense in depth. Nine thousand reads a second for a celebrity's row arrive, and the goal is that essentially none of them touch the primary that owns the row:

1. **Cache the hot key.** A hot key is, by definition, the most cacheable thing in your system — high request rate, one value, and the same value returned every time. An edge cache or a Redis tier in front of the database absorbs 99%+ of reads of a hot key at single-digit-millisecond latency. This is the highest-leverage move and it should be the first one. Read [the caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale) for how to layer those caches.
2. **Replicate it and fan reads across replicas.** The 1% of reads that miss the cache should not all hit the primary. Route them across read replicas, each of which holds a full copy of the row. With three replicas you have tripled the read capacity for that row without touching the primary, which is now free to do nothing but take writes.
3. **Protect the primary.** The combined effect is that the primary — the one machine that *cannot* be replaced for this row, since it owns the authoritative copy — sees a trickle of reads instead of a flood.

There is one trap that turns this defense into a weapon against you: **the thundering herd.** When the hot key's cache entry expires, every one of those 9,000 requests misses simultaneously, and all 9,000 stampede the database in the same instant to recompute the value. The cache that was protecting you becomes a synchronized cannon pointed at your primary. The fixes — request coalescing so only one recompute runs, probabilistic early expiration so the entry refreshes *before* it expires, and serving stale-while-revalidate — are exactly the subject of [cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd), and you must have them in place before you rely on caching a hot key. A hot key without herd protection is a hot key with a detonation timer.

## Sequential-key hotspots

The sequential-key hotspot deserves its own section because it is the one that is *built in at design time* and is silent until the table is large enough to shard. **Any monotonically increasing shard key — an auto-increment `id`, a `created_at` timestamp, a ULID, a Snowflake ID — sends every new write to whichever shard owns the current high end of the key range, so your write throughput is permanently capped at one shard's capacity no matter how many shards you own.**

![Prefixing or reversing a monotonic key spreads new writes across all shards at the cost of range scans](/imgs/blogs/hot-partitions-and-hot-rows-8.webp)

The before/after above shows the four standard remedies, all variations on "break the monotonicity":

- **Hash prefix (salting).** Prepend a small hash of the key — `hash(id) % 16` as a one-byte prefix — so consecutive IDs scatter across 16 prefix buckets and therefore across shards. This is the standard HBase and Bigtable advice for time-series row keys, and OpenTSDB ships it as a built-in "salt" precisely because monotonic metric-timestamp keys would otherwise hammer one region server.
- **Key reversal.** Reverse the digits of the sequential ID so `10001, 10002, 10003` become `10001, 20001, 30001`, which sort into different ranges. This is the classic phone-number / sequential-ID trick that early Twitter and various telco systems used to spread inserts.
- **Random or time-low-bits UUIDs, used carefully.** A random v4 UUID spreads writes perfectly but destroys index locality and bloats your B-tree — which is its own performance disaster, covered in depth in [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance). The nuance is real: you want enough randomness in the *high* bits to spread across shards, but enough time-ordering in the *low* bits to keep each shard's local index append-friendly. This is exactly the design tension behind time-ordered identifiers, and getting it wrong in either direction costs you.
- **Composite key with a real attribute.** Shard by `(tenant_id, created_at)` rather than `created_at` alone, so the high-cardinality tenant dimension does the spreading and the timestamp only orders within a tenant.

Every one of these buys an even write spread and **pays for it in range scans.** A monotonic key gives you cheap "fetch the last hour of rows" because they are physically adjacent; the moment you hash-prefix or reverse it, those rows are scattered across all shards and a range scan becomes a fan-out to every shard. That is the irreducible tradeoff: you cannot have both perfect write distribution *and* cheap sequential range reads on the same key, because they are physically contradictory requirements. Pick the one your workload actually needs, and if it needs both, maintain two differently-keyed copies.

## The celebrity and the fan-out problem

The hardest read hotspot in the industry is the social-timeline fan-out, and it is worth a full treatment because it is where the "you cannot cache your way out of everything" lesson lands hardest. The setup: users follow each other, and each user has a home timeline that merges the recent posts of everyone they follow. The question of *when* you do that merge — at write time or at read time — is the entire ballgame.

![No single fan-out strategy survives both ordinary users and celebrities, so large systems run a hybrid](/imgs/blogs/hot-partitions-and-hot-rows-9.webp)

The matrix above lays out the three strategies and why the obvious two each have a fatal case:

- **Fan-out on write (push).** When you post, you immediately write a copy of your post into the home-timeline cache of every one of your followers. Reads are then trivially cheap: a user reads their own pre-computed timeline in O(1). This is fantastic for the average user with a few hundred followers. It is catastrophic for a celebrity: when an account with 100 million followers posts, you must perform 100 million writes, instantly, and those writes pile onto every shard at once. One celebrity post becomes a cluster-wide write storm. This is the famous "Justin Bieber problem" that nearly broke early Twitter — a handful of accounts whose every post triggered a fan-out so large it threatened the whole system.
- **Fan-out on read (pull).** When you post, you write once, to your own outbox. Reading a timeline then means pulling the recent posts from everyone you follow and merging them at read time — O(following). This is perfect for celebrities (one cheap write, no fan-out) and painful for active readers who follow thousands of accounts, because every timeline load becomes a large scatter-gather.
- **The hybrid: special-case the whale.** This is what every large social system actually runs. Push for normal accounts (cheap reads, the common case), but *pull* for the handful of celebrity accounts whose fan-out would be ruinous. A reader's timeline is then their pushed timeline merged with a small live pull of the few celebrities they follow. You detect "whale" status with the same top-K key counting from the detection section — an account whose follower count crosses a threshold gets flagged and switched from push to pull.

The lesson generalizes far beyond timelines: **when one entity is orders of magnitude more popular than the median, the right architecture is almost always to special-case it.** A uniform strategy that is correct for the median is wrong for the outlier, and the outlier is exactly the thing that takes you down. Detecting the outlier and routing it through a different path is not a hack; it is the design.

## Time-series: "today is always hot"

The time-based partition is the hotspot that *moves*, and it needs its own handling because the usual "find the hot shard and fix it" loop never converges — by the time you have identified today's hot partition, it is tomorrow and the heat has moved to a new one. **For any time-partitioned workload, you must spread within each time bucket and pre-create future buckets, because the write head is always on the newest partition.**

Two techniques, usually combined:

1. **Partition by `(time_bucket, hash)` instead of `time_bucket` alone.** Within each day's data, hash the key into M sub-partitions so the day's writes spread across M shards rather than piling onto one. You keep the time-ordering across days for retention and range queries, but you break the within-day concentration. This is the standard Cassandra wide-row remedy: a partition key of `(sensor_id, day)` rather than just `day`, so no single partition becomes both unboundedly large and unboundedly hot.
2. **Pre-split future partitions.** Many systems lazily create a partition when the first write for a new time bucket arrives — which means at midnight, every writer simultaneously triggers the creation of, and then stampedes, the brand-new "today" partition that lives on one shard. Pre-creating tomorrow's partitions *before* midnight, already spread across shards, removes both the creation stampede and the single-partition concentration. HBase calls this pre-splitting regions; the principle is the same everywhere.

The deeper point is that a time dimension is a *guaranteed* monotonic key — it is the sequential-key problem with a clock attached — so everything from the sequential-key section applies, with the added wrinkle that the hotspot relocates on a schedule. Design for the relocation, not for catching it.

## The hotspot field guide

Pulling the whole post into one table you can pin to the incident-runbook wiki — read it as "I see *this* signal, so it is *this* hotspot, so I reach for *this* fix, and I accept *this* cost":

| Hotspot | Detection signal | Mitigation | Tradeoff you accept |
| --- | --- | --- | --- |
| Single counter row | One row, thousands of `+1`/s, lock waits spike | Sharded counter (N sub-rows, sum on read) | Read amplification: N-key gather per read |
| Arbitrary hot write key | Top-K shows one key is 30–90% of writes | Write-shard with random suffix; append, don't update | Read must merge N sub-keys |
| High write rate, low durability need | Hot row, increments dominate | Coalesce in app, flush `+N` per window | Up to one window of writes lost on crash; stale count |
| Celebrity read row | One key is most of *reads*; one shard's CPU high | Cache the key + fan reads across replicas | Cache staleness; thundering-herd risk on expiry |
| Sequential / monotonic key | Newest shard hot, others cold; no single key dominates | Hash-prefix, reverse, or composite the key | Range scans become cross-shard fan-outs |
| Celebrity fan-out (timeline) | One author's posts trigger cluster-wide writes | Hybrid push/pull: pull for whales, push for the rest | Read merges a small live pull with the pushed feed |
| Time partition ("today") | Hot shard relocates at every bucket boundary | Partition by `(bucket, hash)`; pre-split future buckets | More partitions to manage; cross-shard time-range reads |
| Flash-sale item | One `product_id` spikes both paths for minutes | Cache reads + sharded/transactional decrement for stock | Complexity for a transient event; oversell risk if sloppy |

## Case studies from production

### 1. The 23% average that was a 100% shard

This is the incident from the intro, and it is worth dissecting because the *symptom* is the lesson. A sharded MySQL fleet, eight shards, sharded by `user_id`. A marketing push drove a single promotional account — followed and written to by the whole campaign audience — to 8,900 writes a second, all landing on the one shard that owned that `user_id`. The dashboards we watched (cluster CPU, cluster QPS, cluster connection count) all looked healthy because they averaged the one melted shard against seven idle ones. The wrong first hypothesis was a slow query, then a network blip, then a bad deploy — we burned forty minutes before someone pulled the per-shard graph. The root cause was that a single hot `user_id` had no business being a single row taking all the campaign's writes. The fix was a sharded write key for that account's activity. The lesson, now a permanent alarm: **monitor the skew factor, not the average.**

### 2. Twitter and the Bieber problem

Early Twitter ran fan-out on write: every tweet was copied into each follower's timeline cache at post time. This was elegant until a small number of accounts — the most-followed celebrities — reached tens of millions of followers. A single tweet from one of them triggered tens of millions of cache writes in a burst, and a few of those accounts posting near-simultaneously could saturate the fan-out infrastructure cluster-wide. The wrong instinct was to add fan-out capacity, which only raised the ceiling on the next, bigger celebrity. The actual fix was the hybrid: keep push for the long tail of normal accounts, switch the heaviest accounts to pull, and merge at read time. The lesson is the one this whole post circles: **the median strategy is wrong for the outlier, and the outlier is what melts you, so special-case it deliberately.**

### 3. DynamoDB's hot partition throttling

For years, DynamoDB allocated provisioned throughput evenly across a table's physical partitions. A table provisioned for 10,000 writes a second across ten partitions gave each partition 1,000 — and if your access concentrated on one partition key, that key was throttled at 1,000 even though the table as a whole had 9,000 to spare. Teams hit this constantly with a single popular item or a low-cardinality partition key, saw `ProvisionedThroughputExceededException` on a table that looked under-provisioned, and concluded DynamoDB was broken. The root cause was a hot partition key, full stop. AWS eventually shipped *adaptive capacity* to lend unused throughput to hot partitions automatically, which softens but does not eliminate the problem — a single item still has a hard ceiling. The lesson: **choose a high-cardinality partition key with even access, and never assume table-level capacity is per-key capacity.**

### 4. OpenTSDB and the salted row key

OpenTSDB stores time-series metrics in HBase, where rows are sorted lexicographically by key and contiguous key ranges live on the same region server. A naive metric row key leads with a timestamp, so *every* incoming data point for the current moment writes to the same region — one region server pinned while the rest idle, the canonical sequential-key hotspot at the storage layer. OpenTSDB's fix, shipped as a configuration option, is to prepend a *salt*: a small hash bucket prefix that scatters consecutive timestamps across N region servers. Reads for a time range now fan out across the N salt buckets and merge, which is slightly more expensive, but writes spread evenly. The lesson: **a leading timestamp in a sorted-key store is a write hotspot by construction; salt it.**

### 5. The like counter that serialized

A social product stored each post's like count as a single integer column updated in place. A post went viral and accumulated likes at several thousand per second; the `UPDATE posts SET likes = likes + 1 WHERE id = ?` statements serialized on the row lock, the write queue backed up, and unrelated writes to the same shard stalled behind the hot row. The wrong first fix was a bigger instance, which did nothing because the bottleneck was one row's lock, not the machine. The real fix was a sharded counter — sixteen sub-counters incremented at random and summed by a background job into a cached display value refreshed every two seconds. Write throughput went up more than an order of magnitude, the display lagged reality by at most two seconds, and nobody noticed or cared. The lesson: **a counter is the most concentrated write hotspot there is, and update-in-place is the wrong primitive for it.**

### 6. The flash sale that oversold

An e-commerce platform ran a midnight flash sale on one SKU. The inventory decrement was a transactional `UPDATE stock SET qty = qty - 1 WHERE sku = ? AND qty > 0`, correct but serialized on one row — and at the sale's peak, tens of thousands of buyers contended on that single row, latency exploded, and the checkout path timed out. A panicked attempt to relax the transaction to gain throughput caused the opposite failure: oversell, selling more units than existed. The durable fix combined two ideas from this post — a read cache so the *availability* check did not hit the hot row, and a small pool of pre-allocated inventory "tickets" sharded across rows so the *decrement* spread across N rows that each held a slice of the stock. The lesson: **a flash sale is a both-paths hotspot on one key, and reads and writes need separate treatments — cache the read, shard the decrement, and never trade correctness for throughput on inventory.**

### 7. The Kafka partition that starved one consumer

A streaming pipeline keyed Kafka messages by `customer_id` to preserve per-customer ordering. One enterprise customer generated a hundred times the volume of any other, and since a key always hashes to the same partition, all of that customer's traffic landed on one partition — consumed by exactly one consumer in the group, which fell hours behind while its siblings sat idle. Consumer lag on one partition, the rest at zero: the hot-partition signature, transplanted to a message queue. Adding consumers did nothing, because a partition is consumed by at most one consumer in a group. The fix was to change the key for that customer's high-volume event types to `(customer_id, sub_stream)` so the load spread across partitions, accepting weaker ordering guarantees for those events. The lesson: **partition-by-key concentrates a heavy key onto one partition and one consumer; the unit of parallelism is the partition, and one key cannot exceed it.**

### 8. The Cassandra wide partition that wouldn't compact

A team modeled an event log in Cassandra with a partition key of `event_type`. A handful of event types dwarfed the rest, so a few partitions grew to tens of gigabytes and many millions of rows — and because a Cassandra partition is a unit of storage, replication, and repair that lives entirely on its replica set, those giant partitions made reads time out, compactions take hours, and the owning nodes run hot while others idled. The wrong hypothesis was "we need more nodes"; adding nodes did not move the giant partitions, which were pinned by their key. The fix was to add a bucketing component to the partition key — `(event_type, day)` or `(event_type, hash_bucket)` — to cap any single partition's size and spread the hot event types across the ring. The lesson: **an unbounded or heavily-skewed partition key produces partitions that are both too big and too hot, and the remedy is always to add a dimension that bounds and spreads them.**

## When to break skew, and when to live with it

Breaking skew adds real complexity — sharded counters mean gather-on-read, hybrid fan-out means two code paths, salted keys mean cross-shard range scans. That complexity is justified often, but not always.

**Reach for these techniques when:**

- Your per-shard skew factor is climbing past ~2 and the trend is up, not noise.
- A single key, row, or partition is a double-digit percentage of total traffic on either path.
- You shard by a monotonic key (`id`, timestamp, ULID) and the write path is your bottleneck.
- You have a long-tail popularity distribution — social, e-commerce, media — where some entities are orders of magnitude more popular than the median.
- You partition by time and the "today" partition is your hot spot.

**Skip them — or defer them — when:**

- Your skew factor is near 1 and stable. Do not write-shard a counter that is not contended; you are adding read amplification to solve a problem you do not have.
- The hot key is read-heavy and cacheable, and a cache plus herd protection already keeps it off the primary. Caching is simpler than write-sharding; exhaust it first.
- The total volume is small enough that one machine handles even the concentrated load comfortably. Skew only matters when the indivisible unit's load exceeds one machine's capacity — below that, it is a non-issue.
- The complexity of the fix would exceed the cost of the hotspot. A counter that is hot for ten minutes once a quarter during a known event may be cheaper to over-provision for than to re-architect.

The meta-lesson, and the reason this is a discipline and not a checklist: **sharding spreads keys, but only you can spread load, because only you know which keys are popular and why.** The database will balance the key space for you and then sit there, perfectly balanced and on fire, while one key takes the cluster down. Watching the per-shard graph, naming the hotspot's scale, and reaching for the matching tool — that is the whole job.

## Further reading

- [Choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) — the upstream decision that causes most of the skew here.
- [The caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale) — the first line of defense for read hotspots.
- [Cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd) — required before you cache a hot key.
- [Random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) — the index-locality side of the sequential-key tradeoff.

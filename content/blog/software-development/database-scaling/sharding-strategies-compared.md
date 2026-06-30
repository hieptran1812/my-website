---
title: "Sharding Strategies Compared: Range, Hash, Directory, Geo"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A principal-engineer tour of the four partitioning strategies — range, hash, consistent-hash, directory, and geo — with the sweet spot, the characteristic failure, and the real systems that bet on each."
tags:
  [
    "sharding",
    "partitioning",
    "consistent-hashing",
    "range-sharding",
    "hash-sharding",
    "directory-sharding",
    "geo-sharding",
    "distributed-databases",
    "shard-key",
    "resharding",
    "system-design",
    "scalability",
  ]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 32
---

The worst sharding decision I have watched a team make was not picking the "wrong" strategy. It was picking a reasonable strategy for the wrong reason — "hash is the safe default" — and discovering eighteen months later that the safe default had quietly made every range scan a fan-out across forty nodes, and that the one product feature the company was now built around was a range scan. The fix was a reshard. The reshard took a quarter, a dedicated team, and a multi-week dual-write window. Nobody got promoted for it.

That is the thing about partitioning strategy: it is one of a handful of database decisions that is genuinely hard to reverse. You can add an index in an afternoon. You can swap a cache eviction policy with a config flag. But the function that maps a key to a shard is baked into your write path, your read path, your client routing, your backup topology, and the muscle memory of every on-call engineer. Change it and you are physically moving terabytes while traffic is live.

So this post is not "here is consistent hashing, it is the best, use it." It is the comparison I wish someone had drawn for me before that reshard: four strategies — range, hash (with consistent hashing as its grown-up form), directory, and geo — each with a sweet spot it genuinely owns, and each with a characteristic failure mode that, if it matches your workload, will force you into exactly the reshard above. The whole argument hangs on one observation, which the diagram below makes concrete.

![The shard router as a function: four strategies, one job, mapping a key to a physical shard](/imgs/blogs/sharding-strategies-compared-1.webp)

The diagram above is the mental model: every sharding strategy is the *same function* — `route(key) -> shard_id` — and the only thing that differs is what is inside the box. Range looks at where the key falls in a sorted space. Hash scrambles the key and takes it modulo the node count. Directory keeps an explicit table. Geo reads a region attribute. Same input, same output, four wildly different operational personalities. Once you internalize that, the comparison stops being a memorization exercise ("Cassandra uses a ring, HBase uses ranges") and becomes a question you can actually reason about: *for my access pattern, which routing function fails gracefully and which fails catastrophically?*

If you have not read the companion pieces, they set up the vocabulary I will lean on: [why one database stops being enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) frames the moment you are forced to shard at all, and [the fundamentals of partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) walks the basic mechanics. The deep dive on [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) is the prerequisite for the ring section here, and the forward-looking question of which *attribute* to route on lives in [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key).

## Why partitioning is different from every other scaling lever

Before we tour the strategies, it is worth being precise about *why* this decision deserves a 6,000-word post when "add a read replica" gets a paragraph. The table below is the assumption-versus-reality framing that motivates everything after it.

| Common assumption | Naive view | The reality at scale |
| --- | --- | --- |
| "I can change the shard function later" | It is just a function; refactor it | The function is encoded in petabytes of physical placement; changing it is a data migration, not a code change |
| "Even distribution is the only goal" | Spread bytes uniformly and you win | Uniform bytes with skewed *access* still produces a hot shard; distribution must follow the access pattern, not the data |
| "Hash is always safe" | Hashing randomizes, so no hotspots | Hashing destroys ordering, so every range query becomes a scatter-gather over all N shards |
| "Range is just sorted hash" | Both map keys to shards | Range preserves locality (great for scans, fatal for monotonic keys); they fail in opposite directions |
| "Adding a node is cheap" | Spin up a box, rebalance | Under naive modulo, adding one node remaps ~(N-1)/N of *all* keys; the rebalance is the outage |

> A shard key is the only schema decision you make once and pay for forever. Pick it for the query you run a million times a second, not the one you run in a quarterly report.

The rest of the post is organized as a section per strategy. Each one follows the same shape on purpose, so you can compare them column-for-column: how routing works, the sweet spot, the characteristic failure, a runnable router, and the real systems that bet on it. We close with the scorecard, a worked discussion of hybrids, and a "reach for X / skip X" decision guide.

## 1. Range sharding: contiguous key ranges, one shard per slice

**The senior rule of thumb: reach for range sharding when your dominant query is "give me everything between A and B," and run from it the moment your shard key only ever increases.**

Range sharding assigns each shard a contiguous, non-overlapping interval of the key space. Shard S1 owns keys `[0, 25k)`, S2 owns `[25k, 50k)`, and so on. The router is a sorted lookup: given a key, binary-search the range boundaries to find the owning shard.

![Range sharding: ordered ranges make scans local but a monotonic insert key floods the tail shard](/imgs/blogs/sharding-strategies-compared-2.webp)

The figure shows both faces of range sharding at once. On the happy path (the green box), a range scan for `[26k .. 49k]` touches exactly one shard — S2 — because the data is *ordered* and contiguous. The query planner can do a single sequential seek and stream the result. This is the property hash sharding throws away and never gets back. If your workload is "list the last 100 events for this account, in time order" or "give me all orders in this date range," range sharding is doing exactly the I/O you want and nothing more.

The failure mode is the red box. When the shard key is *monotonically increasing* — an auto-increment id, a `created_at` timestamp, a Snowflake-style sequence — every new row's key is larger than every existing key, so every new row lands in the last shard. S4 in the diagram is the tail, and it absorbs 100% of writes while S1–S3 sit idle. This is the single most common range-sharding incident: a perfectly balanced cluster by *bytes* with a write hotspot that pins one node's CPU at 100% and leaves the rest cold.

Here is a minimal but real range router. Note that the only state it needs is the sorted boundary list, which is small and easy to replicate:

```python
import bisect
from dataclasses import dataclass

@dataclass
class RangeRouter:
    # Sorted lower bounds. boundaries[i] is the inclusive start of shard i.
    # e.g. [0, 25_000, 50_000, 75_000] -> 4 shards
    boundaries: list[int]
    shards: list[str]

    def route(self, key: int) -> str:
        # bisect_right finds the insertion point; subtract 1 to get the
        # shard whose lower bound is <= key. O(log N) in the shard count.
        idx = bisect.bisect_right(self.boundaries, key) - 1
        if idx < 0:
            raise ValueError(f"key {key} below first boundary")
        return self.shards[idx]

    def route_range(self, lo: int, hi: int) -> list[str]:
        # The win: a scan touches only the shards covering [lo, hi].
        start = bisect.bisect_right(self.boundaries, lo) - 1
        end = bisect.bisect_right(self.boundaries, hi) - 1
        return self.shards[start : end + 1]

router = RangeRouter(
    boundaries=[0, 25_000, 50_000, 75_000],
    shards=["S1", "S2", "S3", "S4"],
)
assert router.route(40_000) == "S2"
assert router.route_range(26_000, 49_000) == ["S2"]   # one shard, one seek
assert router.route_range(10_000, 80_000) == ["S1", "S2", "S3", "S4"]
```

### Second-order optimization: splitting is cheap, but the tail never stops moving

The compensating strength of range sharding is that *splitting is online and cheap*. Because shards are ordered, you can split a hot shard's range in two — `[75k, +inf)` becomes `[75k, 90k)` and `[90k, +inf)` — and hand the upper half to a new node without touching any other shard. No global reshuffle. This is exactly what HBase region splits, Bigtable tablet splits, and CockroachDB range splits do automatically when a range exceeds a size or load threshold.

The non-obvious gotcha: with a monotonic key, splitting does not *fix* the hotspot, it *chases* it. You split the tail, traffic immediately concentrates on the new tail, and you split again. You are re-splitting forever. The actual fix is to break the monotonicity — salt the key with a hash prefix, or shard on a high-cardinality attribute instead of the timestamp — which is precisely the question [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) exists to answer.

**Real systems that bet on range:** HBase (regions), Google Bigtable and Cloud Spanner (tablets / splits), MongoDB in ranged-sharding mode, CockroachDB and YugabyteDB (their default is range-based on the primary key), and TiKV. The common thread: these are systems whose users frequently scan ordered data, so they accept the monotonic-key footgun in exchange for cheap splits and local scans.

## 2. Hash sharding: even distribution, dead range scans

**The senior rule of thumb: reach for hash sharding when access is point-lookup-by-key and you need bullet-proof even distribution; never reach for it if you scan ranges, and never use plain modulo if the cluster will ever resize.**

Hash sharding runs the key through a hash function and maps the result to a shard, classically with modulo: `shard = hash(key) % N`. Because a good hash function spreads its inputs uniformly, every shard gets roughly `1/N` of the keys regardless of how skewed the *original* key distribution was. Sequential ids, clustered timestamps, lopsided customer sizes — the hash launders all of it into uniform placement. That is the entire appeal, and it is a real one.

![Hash sharding spreads keys evenly, but bumping N under modulo remaps roughly 80% of keys at once](/imgs/blogs/sharding-strategies-compared-3.webp)

The "before" half of the figure shows the payoff: with `N = 4`, the sixteen keys split cleanly into four buckets of four, no hotspots, no skew. Key `k12` lands on S0 because `12 % 4 == 0`. This is genuinely better than range sharding for any workload dominated by `WHERE id = ?` point lookups — there is no tail shard, no hotspot, no manual rebalancing of byte counts.

The "after" half is the trap that gives the whole strategy its bad reputation. Add one node — `N` goes from 4 to 5 — and the modulo arithmetic changes for almost every key. `k12` was on S0 (`12 % 4 == 0`); now `12 % 5 == 2`, so it must physically move to S2. Run that across all keys and roughly `(N-1)/N` of them — about 80% in this 4-to-5 case — change shards *simultaneously*. The rebalance is not a background nicety; it is a near-total data shuffle that saturates the network, and during it your routing is ambiguous. This is the catastrophe the consistent-hashing section exists to prevent.

A plain modulo hash router is almost insultingly simple, which is part of why people reach for it before they understand the resize behavior:

```python
import hashlib

class ModuloHashRouter:
    def __init__(self, shards: list[str]):
        self.shards = shards

    def route(self, key: str) -> str:
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
        return self.shards[h % len(self.shards)]

    def route_range(self, lo, hi):
        # There is no such thing. Ordering is destroyed by the hash, so a
        # range query must broadcast to EVERY shard and merge the results.
        return list(self.shards)   # scatter-gather over all N

r4 = ModuloHashRouter(["S0", "S1", "S2", "S3"])
r5 = ModuloHashRouter(["S0", "S1", "S2", "S3", "S4"])

# Count how many keys move when we go from 4 shards to 5.
keys = [f"user:{i}" for i in range(100_000)]
moved = sum(1 for k in keys if r4.route(k) != r5.route(k))
print(f"{moved / len(keys):.0%} of keys changed shard")   # ~80%
```

Run that snippet and it really does print roughly 80%. The `route_range` method is the other half of the story: it returns *every* shard, because hashing destroys ordering. A `WHERE created_at BETWEEN ...` query has no locality to exploit, so it fans out to all N nodes, every node does a partial scan, and a coordinator merges the results. At four shards that is annoying; at four hundred it is a query that touches four hundred nodes to return ten rows.

### Second-order optimization: hash the right field, and watch the per-key skew

Even with a perfect hash, you can still build a hotspot — not from key *distribution* but from key *access*. If you hash on `user_id` and one user (a celebrity, a bot, a load-test account) generates 5% of all traffic, that user's shard is hot no matter how uniform the hash is. The hash balances *keys*, not *requests per key*. Detecting this requires per-key traffic sampling, and the fix is usually to either further sub-shard the whale (split its data across a secondary dimension) or to route it through a directory (section 4) so you can pin it to a dedicated node.

**Real systems that bet on hash:** Amazon DynamoDB (partition key fed through an internal hash), Apache Cassandra and ScyllaDB (the partition key is hashed onto a token ring), and most application-level sharding built on plain Postgres/MySQL where the team wrote `user_id % N` in the data-access layer. Critically, DynamoDB and Cassandra do *not* use naive modulo — they use the consistent-hashing variant we turn to next, which is what makes their cluster resizes survivable.

## 3. Consistent hashing: the ring that makes hash resizes survivable

**The senior rule of thumb: consistent hashing is not a different strategy from hashing — it is hashing with the resize footgun removed; reach for it whenever you want hash's even distribution but the cluster will grow, shrink, or lose nodes (which is always, at scale).**

The fundamentals are covered in depth in [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning), so I will keep the mechanics tight and focus on the comparison. Instead of `hash(key) % N`, both keys *and* nodes are hashed onto the same circular keyspace (the "ring," conventionally `0` to `2^32`). A key is owned by the first node you encounter walking clockwise from the key's position. The crucial change: adding or removing a node only affects the arc between that node and its neighbor — roughly `1/N` of keys move, not `(N-1)/N`.

![Consistent hashing: virtual nodes spread each shard across many small arcs, so a join steals only one neighbor's sliver](/imgs/blogs/sharding-strategies-compared-4.webp)

The figure unrolls the ring into a clockwise strip so the arcs are easy to read. Each physical shard (A, B, C) owns *several* small arcs — its virtual nodes, `A.v1`, `A.v2`, and so on — scattered around the ring. Key `k42` hashes to position 2150 and is owned by `C.v1`, the first vnode clockwise. When node D joins at position 3500, it carves out only the sliver of arc just before it — the portion that previously belonged to `A.v2`. Every other arc is untouched. That is the entire point: the dashed "steal" arrow moves one neighbor's keys, not the whole cluster's.

Why virtual nodes? Without them, three physical nodes hashed onto a ring almost never land equidistant, so one node ends up owning a huge arc and another a tiny one — load skew baked into the geometry. Giving each physical node ~100–256 virtual positions averages out the arc sizes (the law of large numbers does the balancing), and it makes a departing node's load spread across *many* survivors instead of dumping entirely on its single clockwise neighbor.

Here is a runnable consistent-hash ring with virtual nodes. It is the same shape as the production implementations in Cassandra and Dynamo, just smaller:

```python
import hashlib
from bisect import bisect_right, insort

class ConsistentHashRing:
    def __init__(self, vnodes: int = 150):
        self.vnodes = vnodes          # virtual nodes per physical shard
        self._ring: dict[int, str] = {}   # position -> physical shard id
        self._sorted: list[int] = []      # sorted ring positions

    def _hash(self, s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

    def add_node(self, shard: str) -> None:
        for v in range(self.vnodes):
            pos = self._hash(f"{shard}#{v}")
            self._ring[pos] = shard
            insort(self._sorted, pos)

    def remove_node(self, shard: str) -> None:
        for v in range(self.vnodes):
            pos = self._hash(f"{shard}#{v}")
            self._sorted.remove(pos)
            del self._ring[pos]

    def route(self, key: str) -> str:
        pos = self._hash(key)
        # First vnode clockwise from the key (wrap to index 0 at the top).
        i = bisect_right(self._sorted, pos) % len(self._sorted)
        return self._ring[self._sorted[i]]

ring = ConsistentHashRing(vnodes=150)
for s in ["A", "B", "C"]:
    ring.add_node(s)

keys = [f"user:{i}" for i in range(100_000)]
before = {k: ring.route(k) for k in keys}
ring.add_node("D")                     # one node joins
after = {k: ring.route(k) for k in keys}
moved = sum(1 for k in keys if before[k] != after[k])
print(f"{moved / len(keys):.1%} of keys moved")   # ~25% (about 1/N), not 80%
```

The contrast between this and the modulo snippet is the whole reason consistent hashing exists, and it is the kind of thing that only really lands in motion. The figure below runs both strategies through "the same round" — one node is added — and lets you watch the difference.

<figure class="blog-anim">
<svg viewBox="0 0 760 420" role="img" aria-label="Adding one node: under hash-modulo almost every key changes shard color; under consistent hashing only one key moves" style="width:100%;height:auto;max-width:820px">
<style>
.sh-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.sh-sub{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.sh-k{font:600 12px ui-sans-serif,system-ui;fill:#1f2937;text-anchor:middle}
.sh-cell{stroke:var(--border,#d1d5db);stroke-width:1.5;rx:6}
.sh-s0{fill:#a5d8ff}
.sh-s1{fill:#b2f2bb}
.sh-s2{fill:#ffec99}
.sh-s3{fill:#d0bfff}
.sh-new{fill:#ffc9c9}
.sh-tag{font:600 13px ui-sans-serif,system-ui;text-anchor:middle;fill:var(--text-secondary,#6b7280)}
@keyframes sh-modulo{0%,30%{fill:#a5d8ff}55%,95%{fill:#ffc9c9}100%{fill:#a5d8ff}}
@keyframes sh-modulo-b{0%,30%{fill:#b2f2bb}55%,95%{fill:#a5d8ff}100%{fill:#b2f2bb}}
@keyframes sh-modulo-c{0%,30%{fill:#ffec99}55%,95%{fill:#b2f2bb}100%{fill:#ffec99}}
@keyframes sh-modulo-d{0%,30%{fill:#d0bfff}55%,95%{fill:#ffec99}100%{fill:#d0bfff}}
@keyframes sh-stay{0%,30%{fill:#a5d8ff}55%,95%{fill:#ffc9c9}100%{fill:#a5d8ff}}
@keyframes sh-fadein{0%,30%{opacity:0}55%,95%{opacity:1}100%{opacity:0}}
.sh-anim-a{animation:sh-modulo 9s ease-in-out infinite}
.sh-anim-b{animation:sh-modulo-b 9s ease-in-out infinite}
.sh-anim-c{animation:sh-modulo-c 9s ease-in-out infinite}
.sh-anim-d{animation:sh-modulo-d 9s ease-in-out infinite}
.sh-anim-stay{animation:sh-stay 9s ease-in-out infinite}
.sh-anim-fade{animation:sh-fadein 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.sh-anim-a,.sh-anim-b,.sh-anim-c,.sh-anim-d,.sh-anim-stay{animation:none}.sh-anim-fade{animation:none;opacity:1}}
</style>
<text class="sh-lbl" x="20" y="34">Add one node, watch what moves</text>
<text class="sh-sub" x="20" y="56">color = which shard each of 12 keys lands on; the band fades in on the "add node" beat</text>
<text class="sh-lbl" x="20" y="118">hash(key) % N</text>
<text class="sh-sub" x="20" y="138">N: 4 -&gt; 5</text>
<rect class="sh-cell sh-s0 sh-anim-a" x="200" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s1 sh-anim-b" x="252" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s2 sh-anim-c" x="304" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s3 sh-anim-d" x="356" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s0 sh-anim-a" x="408" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s1 sh-anim-b" x="460" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s2 sh-anim-c" x="512" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s3 sh-anim-d" x="564" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s0 sh-anim-a" x="616" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s1 sh-anim-b" x="668" y="96" width="44" height="44"/>
<rect class="sh-cell sh-s2 sh-anim-c" x="200" y="148" width="44" height="44"/>
<rect class="sh-cell sh-s3 sh-anim-d" x="252" y="148" width="44" height="44"/>
<text class="sh-tag sh-anim-fade" x="450" y="180">~80% of keys changed shard</text>
<text class="sh-lbl" x="20" y="288">consistent hash</text>
<text class="sh-sub" x="20" y="308">add node D</text>
<rect class="sh-cell sh-s0" x="200" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s1" x="252" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s2" x="304" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s3" x="356" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s0" x="408" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s1" x="460" y="266" width="44" height="44"/>
<rect class="sh-cell sh-anim-stay" x="512" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s3" x="564" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s0" x="616" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s1" x="668" y="266" width="44" height="44"/>
<rect class="sh-cell sh-s2" x="200" y="318" width="44" height="44"/>
<rect class="sh-cell sh-s3" x="252" y="318" width="44" height="44"/>
<text class="sh-tag sh-anim-fade" x="450" y="392">only 1 key moves to D (red)</text>
</svg>
<figcaption>The same round: bumping N under hash-modulo recolors almost every key (each must move shards), while consistent hashing hands exactly one key's arc to the new node and leaves the rest untouched.</figcaption>
</figure>

### Second-order optimization: rendezvous hashing and bounded-load variants

Consistent hashing is not the end of the line. *Rendezvous hashing* (highest-random-weight) achieves the same minimal-movement property without a sorted ring — for each key you compute `hash(key, node)` for every node and pick the max — which is simpler to reason about and naturally supports weighted nodes, at the cost of `O(N)` per lookup instead of `O(log N)`. And vanilla consistent hashing can still produce transient hot arcs; *consistent hashing with bounded loads* (the algorithm behind Google's Maglev and used in some CDN tiers) caps how much any node can own and overflows to the next, trading a little movement for a hard load ceiling. If you are building the routing layer yourself, read past the basic ring before committing.

**Real systems that bet on consistent hashing:** Amazon Dynamo and DynamoDB, Apache Cassandra and ScyllaDB, Riak, Akamai's original CDN, and memcached client libraries (ketama). Anything that must add and remove nodes routinely without a global reshuffle ends up here.

## 4. Directory sharding: an explicit map, total flexibility, one dependency

**The senior rule of thumb: reach for a directory when you need to relocate individual keys or tenants on demand — split a whale onto its own node, pin a noisy customer, migrate one account between regions — and you can afford to make a fast, highly-available lookup part of every request.**

The three strategies so far are *computed* — the shard falls out of an arithmetic function of the key, with no per-key state. Directory sharding throws that away and keeps an explicit table: `map[key] -> shard`. To route, you look up the key in the map. The map can say anything, which is the source of both its power and its cost.

![Directory sharding: a lookup map can relocate any single tenant, but the lookup itself becomes a cached, HA dependency](/imgs/blogs/sharding-strategies-compared-5.webp)

The figure traces a request for tenant `acme`. It first hits a *local cache* of the directory (the green box) — in steady state this is a ~99% hit, so routing costs a hashmap lookup, not a network round trip. On a miss, the request falls through (dashed arrow) to the *directory service* itself, which holds the authoritative `map[tenant] = shard` and must be replicated and highly available, because if it is down, *nothing can be routed*. The payoff is on the right: `acme` is a whale, and when S9 can no longer hold it, you spin up a dedicated `S-acme` node and **rewrite exactly one map row**. No data-space arithmetic, no neighbor arcs — you move precisely the tenant you chose, and only that tenant.

This is the pattern behind most large multi-tenant SaaS. The logical-to-physical indirection lets you treat placement as a runtime decision: small tenants pack many-to-a-shard, mid-size tenants get a shared shard, and the handful of giant tenants each get dedicated capacity. The same indirection is what lets you do an online migration — write to both old and new shard, flip the map row when caught up, drain the old. The map *is* the cutover switch. (This is the machinery that [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) is built on.)

A directory router with a cache, written the way you would actually deploy it:

```python
import time

class DirectoryRouter:
    def __init__(self, directory_service, ttl_seconds: float = 30.0):
        self._svc = directory_service        # authoritative, replicated, HA
        self._cache: dict[str, tuple[str, float]] = {}
        self._ttl = ttl_seconds

    def route(self, key: str) -> str:
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and now - hit[1] < self._ttl:
            return hit[0]                    # ~99% of requests end here
        shard = self._svc.lookup(key)        # network round trip on miss
        self._cache[key] = (shard, now)
        return shard

    def relocate(self, key: str, new_shard: str) -> None:
        # The superpower: move ONE key. Rewrite one row, invalidate one
        # cache entry. No other key is affected.
        self._svc.put(key, new_shard)
        self._cache.pop(key, None)
```

### Second-order optimization: the directory is a single point of failure until you treat it like one

The directory's flexibility is real, but the lookup is now on the critical path of every request, so its availability *is* your availability. Three things make this survivable, and skipping any one of them is how directory deployments fail. First, **cache aggressively** with a short TTL and serve stale on lookup failure — a brief window of slightly-wrong routing beats a hard outage. Second, **replicate the directory** itself (it is small — millions of rows of `key -> shard` fit in memory) and front it with the same HA you would give any tier-0 service. Third, **bound the cardinality**: a directory keyed on `user_id` for a billion users is a billion-row map that no longer fits in a local cache — directories work best keyed on a coarser entity (tenant, account, shard-group) so the map stays small. When teams say "directory sharding doesn't scale," they almost always mean they keyed it too finely.

**Real systems that bet on directory/lookup routing:** Notion's block-to-shard mapping, Figma's logical-shard layer (logical shards mapped to physical Postgres instances so they can move shards without re-sharding application logic), Vitess's vindex / keyspace routing, Slack and many large SaaS tenant-routers, and YouTube's original sharding layer. The common thread: a multi-tenant workload where individual tenants vary enormously in size and must be relocatable.

## 5. Geo / entity sharding: route by region, pay at the border

**The senior rule of thumb: reach for geo sharding when data residency or in-region latency is a hard requirement; budget up front for the fact that any query crossing a region boundary will be slow, expensive, and operationally awkward.**

Geo sharding partitions on a location or entity attribute: European users live on the EU shard, US users on the US shard, APAC on the APAC shard. The router reads `region(key)` and sends the request to that region's database. It is really a special case of directory or attribute-based sharding where the partitioning attribute happens to be geography — but it deserves its own section because its failure modes are about physics (the speed of light across an ocean) and law (data-residency regulation), not just distribution.

![Geo sharding: local reads stay fast and in-region, but a global aggregate scatters across oceans and hits uneven regional load](/imgs/blogs/sharding-strategies-compared-6.webp)

The figure shows the two-sided bargain. A *local query* (the green box) — "show this US user their own data" — stays entirely within US-EAST and returns in ~2 ms, while also satisfying the residency rule that EU users' rows physically remain in the EU shard. That is the win, and for a lot of products it is a regulatory requirement (GDPR data localization) rather than a performance nicety, so it is non-negotiable.

The cost is on the right. A *global aggregate* — `COUNT(*)` across all regions, or any cross-region join — must scatter to all three shards (the dashed arrows), wait for the slowest one across a +120 ms wide-area round trip, and merge. It also pays cloud egress charges per gigabyte that crosses a region boundary. And note the load numbers: US-EAST carries 55% while APAC carries 15% — geo partitioning's distribution follows your *user geography*, which is almost never uniform, so the busiest region's shard runs hot while others idle. You cannot rebalance by moving keys (that would violate residency); you can only scale the hot region vertically or sub-shard within it.

```python
class GeoRouter:
    def __init__(self, region_shards: dict[str, str], home_lookup):
        self.region_shards = region_shards       # "EU" -> "eu-west-pg"
        self._home = home_lookup                 # key -> home region

    def route(self, key: str) -> str:
        region = self._home(key)                 # entity's residency region
        return self.region_shards[region]

    def route_global(self, query) -> list[str]:
        # Any cross-region query is a scatter-gather: every region's shard,
        # WAN latency to the slowest, plus egress cost per GB moved.
        return list(self.region_shards.values())

geo = GeoRouter(
    region_shards={"US": "us-east-pg", "EU": "eu-west-pg", "APAC": "ap-se-pg"},
    home_lookup=lambda k: {"u_91": "US", "u_42": "EU"}.get(k, "US"),
)
assert geo.route("u_42") == "eu-west-pg"           # stays in the EU, ~2 ms
assert len(geo.route_global("count")) == 3         # touches every region
```

### Second-order optimization: the entity that does not have one region

Geo sharding assumes every entity *has* a home region. The pain begins with entities that span regions: a multinational company account with users on three continents, a shared document edited from the US and the EU, a transaction between a US buyer and an EU seller. Now you must pick a home (and accept cross-region latency for the others), duplicate the entity (and reconcile writes), or split it (and lose the single-shard read). There is no clean answer — this is the same "which entity owns the relationship" problem that makes [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) hard, with international borders added. Plan for cross-region entities explicitly; do not let them emerge as a production surprise.

**Real systems that bet on geo/entity sharding:** Salesforce (per-instance / per-region pods), most large SaaS subject to GDPR/data-localization, Uber and DoorDash (city/region as a natural partition for geographically-local workloads), and any global product that runs region-pinned database tiers behind a latency-based router.

## The scorecard: no row wins every column

Lay the five strategies against the dimensions that actually decide the choice and the shape of the problem becomes obvious — there is no dominant row.

![A strategy scorecard: each partitioning strategy is strong on some dimensions and characteristically weak on others](/imgs/blogs/sharding-strategies-compared-7.webp)

The matrix is color-coded by how each strategy fares on each dimension: green is a strength, amber is a tunable tradeoff, red is the characteristic weakness. Read it as a tour of *failure modes*, because that is what picks the strategy: range's red cell is the monotonic-key hotspot; hash's red cell is the modulo reshard; consistent hash trades a little flexibility for surviving resizes; directory pays for full flexibility with the lookup dependency; geo's red cells are cross-region skew and the impossibility of moving keys across borders.

The same data, with the example systems and the one-line "pick it when" that the figure leaves to prose:

| Strategy | Range scans | Distribution | Rebalance cost | Hotspot risk | Per-key flexibility | Example systems | Pick it when |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Range** | Excellent (local) | Skews to data | Cheap online split | High (monotonic keys) | None | HBase, Bigtable, Spanner, CockroachDB, MongoDB ranged | Dominant query is an ordered range scan |
| **Hash (modulo)** | None (broadcast) | Excellent | Catastrophic (~80% move) | Low | None | App-level `id % N` | Point lookups, fixed cluster size only |
| **Consistent hash** | None (broadcast) | Good (vnodes) | Minimal (~1/N moves) | Low | Limited | DynamoDB, Cassandra, Scylla, Riak | Point lookups + cluster will resize |
| **Directory** | Depends on shard | Tunable (you choose) | Move one row | Low (split whales) | Full | Notion, Figma, Vitess, Slack | Tenants vary wildly; need relocation |
| **Geo / entity** | In-region only | Follows geography (skewed) | Add a region | Region skew | By region | Salesforce, Uber, GDPR-bound SaaS | Residency or in-region latency is mandatory |

## Combinations: the strategies are composable

In practice the best designs rarely use one strategy in its pure form. The two most common and most useful hybrids:

**Hash-within-range (compound key).** Use a hash prefix as the high-order part of the key and a meaningful sort field as the low-order part: `(hash(user_id) % 64, created_at)`. The hash prefix spreads writes across 64 buckets — killing the monotonic-`created_at` hotspot — while *within* each bucket the data is still ordered by time, so a single-user time-range scan stays local to one bucket. Cassandra's partition key plus clustering key is exactly this: the partition key is hashed onto the ring (even distribution), and rows within a partition are stored sorted by the clustering columns (local ordered scans). You get hash's distribution and range's scans, at the cost of needing the partition key on every query.

**Directory-over-hash (two-level routing).** Use a directory to map a *logical* shard to a *physical* node, and consistent hashing (or simple modulo) to map keys to logical shards. Keys never reshard — `key -> logical_shard` is a stable hash with a fixed, large logical-shard count (say 1024). Only the cheap `logical_shard -> physical_node` directory changes when you add hardware, and moving a logical shard between nodes is a bounded, schedulable operation. This is precisely Vitess's model and the logical-shard pattern Figma adopted: pick a large fixed number of logical shards once, then let the directory layer pack and unpack them onto physical instances as the fleet grows. It is the cleanest way to get hash's distribution *and* directory's relocatability without the all-keys-reshard of either one alone.

```python
class TwoLevelRouter:
    """Directory-over-hash: keys -> logical shard (stable), logical -> physical (movable)."""
    def __init__(self, logical_count: int, placement: dict[int, str]):
        self.logical_count = logical_count        # fixed forever, e.g. 1024
        self.placement = placement                # logical_shard -> physical_node

    def route(self, key: str) -> str:
        import hashlib
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
        logical = h % self.logical_count          # NEVER changes when hardware changes
        return self.placement[logical]            # only THIS table is rewritten on a move

    def move_logical_shard(self, logical: int, to_node: str) -> None:
        self.placement[logical] = to_node         # bounded, schedulable migration
```

## Case studies from production

Strategy choices are easiest to learn from where they broke. These are paraphrased from public postmortems, conference talks, and engineering blogs; exact figures are approximate, but the failure shapes are real.

### 1. The auto-increment timestamp that built a one-node cluster

A team sharded their event store by range on a `created_at` primary key, reasoning that "we mostly query recent events, and range scans are fast." For a month it worked. Then write volume grew and on-call noticed one node pinned at 100% CPU while the other seven idled. The monotonic timestamp meant every new event landed in the tail shard; the cluster had eight nodes of capacity and one node of throughput. The wrong first hypothesis was "the tail node's hardware is faulty," and an hour was lost swapping it. The root cause was the range strategy meeting a monotonic key — textbook. The fix was a hash-prefixed compound key (`(bucket, created_at)` with 32 buckets), which spread writes across all nodes while keeping per-bucket time scans local. Lesson: range plus a monotonic key is a write hotspot by construction, not by accident.

### 2. The "quick" capacity add that became a six-hour reshuffle

A mid-size service used application-level `user_id % N` hashing across MySQL shards. Black Friday was coming, so they added two nodes to go from 8 to 10 the week before. The deploy flipped `N` from 8 to 10, and *immediately* roughly 80% of all keys resolved to a different shard. Reads started missing on the new nodes (which had no data yet), the migration job to physically move rows saturated the inter-shard network, and the cluster was effectively degraded for six hours during the move. The wrong first hypothesis was "the new nodes are misconfigured." The root cause was naive modulo's resize behavior. The eventual fix was migrating to consistent hashing so future capacity adds moved only ~`1/N` of keys. Lesson: if you hash, never use plain modulo on a cluster that will resize; this is the single most expensive line of code in the post.

### 3. The whale that no shard could hold

A B2B SaaS hashed tenants onto shards. Most tenants were small, but one enterprise customer signed and grew to 40% of total data volume — a single tenant that no longer fit on its hashed shard and starved every co-located tenant of I/O. Hashing offered no escape hatch: you cannot move *one* key under a hash function without moving the function. They retrofitted a directory layer on top of the hash so they could pin the whale to a dedicated node by rewriting a single map row, leaving everyone else on the computed hash. Lesson: pure computed sharding (hash or range) has no per-key relocation; if your tenant size distribution is heavy-tailed, you will eventually need a directory.

### 4. The cross-region JOIN that the dashboard team did not know existed

A global product geo-sharded users by region for GDPR compliance — correct and necessary. Months later, the analytics team shipped an executive dashboard with a query joining users to orders *across all regions*. In staging (single region) it was fast. In production it scattered to three continents, waited on a +150 ms trans-Pacific leg, and racked up a surprising cloud egress bill. The wrong first hypothesis was "the dashboard query is unoptimized — add an index." No index helps a query whose cost is the speed of light. The fix was a separate cross-region read replica / analytics store that aggregated regional data asynchronously, keeping the residency-bound operational shards purely local. Lesson: geo sharding makes *every* cross-region query expensive; route analytics to a dedicated aggregation tier, not the operational shards.

### 5. The directory that took down everything when it blinked

A team adopted directory sharding for its flexibility and ran the directory as a single, un-replicated service "to ship faster." It worked beautifully — until a routine deploy of the directory service caused a 90-second restart, during which *no request in the entire system could be routed*, because the lookup was on every request's critical path and the local caches had a short TTL that expired mid-restart. The wrong first hypothesis was "the application servers crashed." They had not; they were healthy but blind. The fix was threefold: replicate the directory (it was small enough to be trivially HA), lengthen the client cache TTL with serve-stale-on-error, and add a circuit breaker that fell back to last-known-good routing. Lesson: the directory's flexibility is paid for with availability; treat the lookup as a tier-0 dependency or it becomes your tier-0 outage.

### 6. The logical-shard count nobody could change

A company built two-level routing (directory-over-hash) but fixed the logical-shard count at 16, thinking "we'll never need more than 16 shards." Growth proved otherwise: 16 logical shards meant a hard ceiling of 16 physical nodes' worth of placement granularity, and the largest logical shard could not be subdivided without re-hashing every key in it — the exact reshard the architecture was supposed to avoid. The wrong first hypothesis was "we need a bigger machine for shard 7." The root cause was choosing the logical-shard count too small to ever need re-hashing. The fix (painful, mid-flight) was migrating to 1024 logical shards. Lesson: in directory-over-hash, set the logical-shard count generously high from day one — it is cheap when small and impossible to grow without a reshard.

### 7. The hash that balanced bytes but not requests

A social platform hashed content by `author_id`, achieving textbook-uniform byte distribution. Then a celebrity account's post went viral and that one author's shard took 30% of all read traffic while holding 0.001% of the data — a perfectly balanced cluster by storage with one molten node by load. The wrong first hypothesis was "the hash function is biased." It was not; the hash balances *keys*, not *requests per key*. The fix was a hybrid: detect hot keys via traffic sampling, then promote them into a directory that pinned hot authors' read paths to a dedicated read-replica fleet. Lesson: even distribution of keys does not imply even distribution of load; you must measure per-key access, not just per-key storage.

## When to reach for each strategy — and when not to

**Reach for range when:**

- Your dominant, high-frequency query is an ordered range scan (time series, sorted feeds, "last N by timestamp").
- You want cheap, online shard splits without a global reshuffle.
- Your shard key is high-cardinality and *not* monotonic — or you are willing to hash-prefix it to break monotonicity.

**Reach for hash (consistent hashing, not modulo) when:**

- Access is dominated by point lookups by key (`WHERE id = ?`), not range scans.
- You need even distribution regardless of how skewed the raw key space is.
- The cluster will grow and shrink — which, at scale, it always will — so you need the `1/N` resize property.

**Reach for a directory when:**

- Tenant or key sizes are heavy-tailed and you must relocate individuals (split whales, pin noisy neighbors).
- You need online migration with a clean cutover switch (write-both, flip-the-row, drain).
- You can key the directory coarsely (tenant, account, logical shard) so the map stays small enough to cache, and you can make the lookup highly available.

**Reach for geo when:**

- Data residency is a legal requirement, or in-region latency is a hard product requirement.
- Workloads are naturally geographically local (most reads stay within one region).

**Skip these and reconsider when:**

- You are choosing **hash** but your real workload is range scans — you are signing up for a fan-out on your most common query, and eventually a reshard. (This was the opening war story.)
- You are choosing **range** with a monotonic shard key — you have built a single-node write bottleneck with extra steps; hash-prefix the key or pick a different one.
- You are choosing **plain modulo hashing** for a cluster that will ever resize — use consistent hashing instead; the cost of retrofitting it later is a multi-hour reshuffle.
- You are choosing a **directory** but keying it on a billion-row dimension, or running it un-replicated — you have either built an uncacheable map or a single point of failure.
- You are choosing **geo** for a workload with frequent cross-region queries — every one of those queries pays WAN latency and egress; if cross-region access is common, geo is fighting your access pattern.
- You are agonizing over the strategy before you have measured your actual read/write ratio and query mix — the access pattern picks the strategy, so go measure it first. The strategy is downstream of the [shard key](/blog/software-development/database-scaling/choosing-a-shard-key), and both are downstream of knowing what your system actually does a million times a second.

The single most useful habit: write down your top three queries by volume and your expected node-count trajectory *before* you pick. Range, hash, directory, and geo are not better or worse than each other — they fail in different directions, and the only mistake is choosing one whose failure direction points straight at your workload.

## Further reading

- [Consistent hashing and how distributed databases partition data](/blog/software-development/database/consistent-hashing-and-data-partitioning) — the ring, virtual nodes, preference lists, and rendezvous hashing in depth.
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — the foundational mechanics this post builds on.
- [Choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) — which *attribute* to route on, the decision upstream of the strategy.
- [Resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) — the dual-write, backfill, and cutover machinery for when you picked wrong and must migrate live.

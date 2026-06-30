---
title: "Scaling Memcache at Facebook: Leases, Pools, and Cross-Region Caching"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A close reading of the canonical NSDI 2013 paper on how Facebook turned a plain look-aside memcached tier into a petabyte-scale, multi-region caching layer — leases, pools, regional caches, remote markers, and a commit-log invalidation pipeline that bounds staleness instead of pretending to eliminate it."
tags: ["database-scaling", "memcached", "caching", "look-aside-cache", "leases", "thundering-herd", "cache-invalidation", "multi-region", "eventual-consistency", "distributed-systems", "facebook", "case-study"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 32
---

Almost every large read-heavy system arrives at the same place: a database that cannot keep up with reads, a cache bolted in front of it, and a slow dawning realization that the cache is now the hardest part of the system. The cache was supposed to be the easy win — put a `memcached` in front of MySQL, serve hot keys from RAM, go home. Then the traffic grows, a popular key expires at the wrong moment and a thousand web servers hammer the database in the same millisecond, a stale value sticks around long after the row changed, and a write in one datacenter takes seconds to show up in another. Suddenly the "easy" cache layer is where your outages live.

The 2013 NSDI paper [*Scaling Memcache at Facebook*](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala) (Nishtala et al.) is the canonical field report on what it actually takes to run a look-aside cache at the scale of a billion-user social network — billions of requests per second, trillions of cached items. What makes the paper worth reading more than a decade later is not the raw numbers; it is that the problems they hit are the *same problems any large read-heavy system hits*, just earlier and louder. Thundering herds, stale sets, cross-region staleness, one churny key set evicting another — these are not Facebook-specific. They are the universal failure modes of look-aside caching, and the paper is essentially a catalog of the mechanisms that bound each one.

![Look-aside read and write flow: get on read, delete on write](/imgs/blogs/scaling-memcache-at-facebook-1.webp)

The diagram above is the mental model the rest of this article lives inside. Memcached is a *demand-filled, look-aside* cache: the web server checks the cache first, and on a miss it reads MySQL and *sets* the value back into the cache itself. On a write, the web server updates MySQL and then *deletes* the key — it does not update the cache in place. That single asymmetry (fill on read, delete on write) is the seed from which every other mechanism in the paper grows. We will spend the rest of this piece walking outward from it, in the same three concentric scopes the authors use: within a single cluster, within a region of many clusters, and across many geographic regions.

## Why caching at this scale is different

The instinct most engineers bring to caching is "it's just a hashmap with a network hop." That instinct is correct for a single cache server and a handful of clients, and dangerously wrong once you have thousands of each. Here is the gap between the comfortable mental model and the reality the paper describes.

| Assumption | Naive view | Reality at scale |
|---|---|---|
| A cache miss just costs one DB read | Each miss is independent and cheap | A *hot* miss is correlated: thousands of clients miss the same key in the same instant and stampede the DB at once |
| Writing to the DB then updating the cache keeps them in sync | Update-on-write is the obvious pattern | Concurrent updates race; the *delete*-on-write pattern plus leases is what actually avoids permanently stale values |
| LRU eviction treats all keys fairly | One shared pool is simplest | A high-churn key set silently evicts a low-churn, expensive-to-recompute key set sharing the same LRU |
| Replicating the cache everywhere maximizes hit rate | More replicas = more hits | Over-replicating low-traffic items wastes RAM; a *shared* regional pool serves them more efficiently |
| A write is visible everywhere once the DB commits | The DB is the source of truth, so reads are correct | Cross-region MySQL replication lags; a read in a replica region can see a value the master already changed |
| Cache and DB consistency can be made exact | Just invalidate carefully | Exact consistency at this scale is unaffordable; the realistic goal is to *bound* staleness, not eliminate it |

> The single most important sentence in the paper, paraphrased: they were *willing to settle for eventual consistency*. Every mechanism below is an engineering trade that buys a bound on staleness or load, not a guarantee of perfection. If you read the paper looking for a way to make a cache strongly consistent, you will be confused. Read it as a catalog of how to *bound the damage* and it snaps into focus.

The rest of this article is a tour through that catalog, scope by scope.

## 1. The look-aside contract: fill on read, delete on write

**Senior rule of thumb: the cache is an optimization, never the source of truth, and the write path proves it by deleting rather than updating.**

Start with the contract, because everything else is built on it. In a look-aside (also called cache-aside) design, the *application* — not the cache — owns the logic. The cache server is dumb: it stores bytes under keys, evicts under memory pressure, and answers `get`/`set`/`delete`. All the cleverness lives in the client library on the web server.

The read path is the obvious one:

```python
def get_user(user_id):
    key = f"user:{user_id}"
    value = mc.get(key)            # 1. look-aside: ask the cache first
    if value is not None:
        return value               #    hit: done, no DB touch
    row = db.query(                # 2. miss: fall back to the source of truth
        "SELECT * FROM users WHERE id = %s", user_id
    )
    mc.set(key, serialize(row))    # 3. demand-fill: populate the cache for next time
    return row
```

The write path is where the design decision lives. The tempting move is "update the DB, then update the cache" — keep them in lockstep. The paper does the opposite:

```python
def update_user(user_id, changes):
    db.execute(                    # 1. write the source of truth FIRST
        "UPDATE users SET ... WHERE id = %s", user_id
    )
    key = f"user:{user_id}"
    mc.delete(key)                 # 2. delete (do NOT set) the cached value
```

Why delete instead of update? Because *update-on-write races and delete-on-write does not*. Imagine two writers, A and B, both changing the same user. If both write the DB and then both write the cache, the cache can end up holding A's value even though B's write committed to the DB last — the cache `set`s can be reordered relative to the DB commits by arbitrary network and scheduling delays. Now the cache is *permanently* wrong: it disagrees with the DB and nothing will ever fix it until the key expires. Delete-on-write sidesteps this: after a write, the key is simply absent, so the very next reader takes the miss path and reloads the current truth from the DB. The worst case of a delete is an extra cache miss; the worst case of a stale `set` is silent, lasting corruption.

There is a subtlety the paper is careful about: order. You write the DB *first*, then delete the cache. If you deleted first and then wrote the DB, a concurrent reader could slip in between the two steps, read the *old* DB value, and re-populate the cache with stale data right before your DB write lands — and now you are back to permanent staleness. Write-then-delete leaves a small window where the cache is briefly absent (harmless) rather than a window where it can be repopulated stale (harmful).

This look-aside contract has a second-order consequence that drives the rest of the paper: **deletes are the load-bearing operation for correctness.** If a delete is lost, you have a stale key. So the entire cross-region machinery later in this article exists to make sure deletes are delivered reliably and in the right order. Hold that thought.

## 2. Reducing latency inside a cluster: parallelism and incast

**Senior rule of thumb: at scale, latency is dominated by fan-out, and fan-out is dominated by the slowest packet — so control the concurrency, do not just crank it.**

A single page load at Facebook does not fetch one key; it fetches *hundreds* — friends, their names, their photos, the comments on a story, the like counts. The client library models a page's data dependencies as a directed acyclic graph and issues memcached requests in waves: everything with no unmet dependency goes out in parallel, then the next wave, and so on. Batching independent `get`s into one round trip and parallelizing across cache servers is what keeps the per-page latency budget intact when the data graph is wide.

The authors use UDP for `get` requests. A `get` is idempotent and the payload is small, so the connection setup and head-of-line blocking of TCP are pure overhead; on a miss or a dropped packet the client simply treats it as a cache miss and falls back to the DB. Writes (`set`, `delete`) go over TCP through a proxy (`mcrouter`) because they must be reliable. This split — lossy-but-fast reads, reliable writes — is a recurring trick in high-throughput caching.

The interesting failure mode here is **incast congestion**. When one client fans a single logical request out to many cache servers at once, all of their responses come back at nearly the same time and converge on the same client NIC and top-of-rack switch buffer. The buffer overflows, packets drop, and you get a latency cliff precisely *because* you parallelized aggressively. The fix is counterintuitive: throttle yourself. The client maintains a **sliding window** that caps the number of outstanding requests at `n` (configurable); request `n+1` waits in a queue until an in-flight one completes.

```python
# Sliding-window flow control to avoid incast congestion.
# Too small a window underutilizes the network; too large floods
# the switch buffers and triggers packet loss and a latency cliff.
class WindowedClient:
    def __init__(self, window=32):
        self.window = window
        self.in_flight = 0
        self.pending = collections.deque()

    def issue(self, request):
        if self.in_flight < self.window:
            self.in_flight += 1
            self._send(request)        # room in the window: send now
        else:
            self.pending.append(request)  # over the limit: queue it

    def on_response(self, response):
        self.in_flight -= 1
        if self.pending:               # backfill the freed slot
            self.in_flight += 1
            self._send(self.pending.popleft())
```

The window size is a tuning knob, not a constant: too small and you leave the network idle, paying serial latency for work that could overlap; too large and you reproduce the incast collapse you were trying to avoid. The lesson generalizes well beyond Facebook — any system that fans out reads (scatter-gather search, distributed joins, GraphQL resolvers hitting many backends) eventually needs a concurrency cap, and discovering this the hard way during a traffic spike is a rite of passage.

## 3. Leases: one mechanism, two problems

**Senior rule of thumb: a lease is a tiny token that turns "everyone races to fix the cache" into "exactly one client is allowed to, and only with permission."**

Leases are the cleverest idea in the paper, and the reason it is on every distributed-systems reading list. A lease is a 64-bit token that memcached hands to a client *on a miss*. It solves two seemingly unrelated problems with one mechanism: **stale sets** and **thundering herds**.

![Lease get/set sequence: a miss returns a token, only the holder may set, and concurrent readers wait](/imgs/blogs/scaling-memcache-at-facebook-2.webp)

Walk the sequence above. Client A does a `get` and misses. Instead of just returning "miss," memcached returns a miss *plus* a lease token `L` bound to that key. A is now the designated filler. A goes to the DB, computes the value, and calls `set(k, v, lease=L)`. Memcached checks the token: if `L` is still the most recent lease it issued for `k`, the `set` succeeds. If the key has been *deleted* in the meantime (say, because a write happened while A was off querying the DB), memcached has invalidated `L`, and A's `set` is **rejected**. A's value was computed from a now-stale snapshot, so rejecting it is exactly right — the next reader will get a fresh lease and reload current truth.

That is the **stale-set guard**. Here is the precise race it kills:

1. Client A misses `k`, gets lease `L`, reads value `v0` from the DB.
2. A write changes the row and *deletes* `k` from the cache. This invalidates `L`.
3. A (still holding the now-void `L`) finally calls `set(k, v0, lease=L)`.
4. Memcached sees `L` is no longer valid and **drops the set.** The stale `v0` never lands.

Without leases, step 3 would happily install the stale `v0` and the cache would disagree with the DB until expiry. With leases, a slow filler can never overwrite fresher data with the value it computed from an older snapshot.

Now the second problem. When a hot key expires or is deleted, *many* clients miss it at once. Without coordination, all of them go to the DB simultaneously — the thundering herd. Leases fix this too, because **memcached only issues a lease for a given key at a limited rate.** The first client to miss gets a lease and becomes the filler; clients that miss in the same window do *not* get a lease. Instead, they get a "hot miss" response telling them to wait briefly and retry the `get`. By the time they retry, the lease holder has usually populated the cache, so the retry is a hit. The herd of N database reads collapses to one.

![Thundering herd: without leases every reader stampedes MySQL; with leases one client fills](/imgs/blogs/scaling-memcache-at-facebook-3.webp)

The before/after above is the whole argument in one picture. On the left, a hot key expires and N concurrent misses turn into N identical DB reads — the database is doing redundant work N times over and may fall over. On the right, leases funnel all N misses through a single fill: one DB read, the other N−1 clients wait and retry into a hit. Because lease issuance is *rate-limited per key*, even a sustained burst of misses yields at most one fill per interval, which puts a hard ceiling on how much load any single key can inflict on the database.

Here is the lease protocol as pseudo-code on the memcached side, so the rejection logic is concrete:

```python
class LeasedCache:
    def __init__(self, lease_interval_s):
        self.store = {}                  # key -> value
        self.valid_lease = {}            # key -> currently valid lease token
        self.last_lease_at = {}          # key -> last time we issued a lease
        self.lease_interval = lease_interval_s

    def get(self, key, now):
        if key in self.store:
            return ("HIT", self.store[key])
        # Miss. Issue a lease only if we have not issued one for this
        # key too recently — this is the thundering-herd rate limit.
        if now - self.last_lease_at.get(key, 0) >= self.lease_interval:
            token = self._mint_token()
            self.valid_lease[key] = token
            self.last_lease_at[key] = now
            return ("MISS_WITH_LEASE", token)   # caller becomes the filler
        else:
            return ("HOT_MISS", None)           # caller must wait + retry

    def set(self, key, value, token):
        # The stale-set guard: only honor a set whose token is still the
        # currently valid lease. A delete (below) clears valid_lease[key],
        # so any in-flight set from before the delete is rejected.
        if self.valid_lease.get(key) == token:
            self.store[key] = value
            del self.valid_lease[key]
            return "STORED"
        return "REJECTED"                       # stale snapshot, drop it

    def delete(self, key):
        self.store.pop(key, None)
        self.valid_lease.pop(key, None)         # invalidate any outstanding lease
```

There is one more wrinkle the paper notes, and it is a beautiful trade. For some keys, returning a *slightly stale* value to the waiting clients is better than making them wait. So a variant lets the lease holder fill while concurrent readers receive the last-known (stale-marked) value immediately rather than blocking. You give up a little freshness to eliminate the wait entirely — a per-key choice between "always fresh, sometimes slow" and "always fast, occasionally slightly stale." That this is a *tunable* rather than a fixed policy is the paper's consistency philosophy in miniature.

## 4. Pools: stop cheap keys from evicting expensive ones

**Senior rule of thumb: one shared LRU is a tragedy of the commons — the highest-volume key set wins the cache and starves everyone else.**

Not all cached data is alike. Some keys are accessed constantly and recomputed cheaply; others are accessed rarely but cost a heavyweight query or a fan-out to several services to regenerate. If they all share one memcached instance with one LRU eviction policy, the high-volume, high-churn keys constantly push the rarely-touched-but-expensive keys out of memory. The expensive keys then miss constantly, and the database eats the cost — even though those keys would have been cheap to keep cached if only they had not been competing with a firehose of churn.

![Memcache pools: a shared LRU lets churny keys evict expensive ones; pools isolate working sets](/imgs/blogs/scaling-memcache-at-facebook-4.webp)

The fix is to partition memcached into **pools** — logically separate caches with independent LRUs, possibly on different sets of servers — and route keys to a pool based on their access pattern. The paper describes a small "wildcard" pool for high-churn keys that gets sized modestly (because evicting them is cheap), and separate pools for low-churn, valuable keys that are sized to hold their working set comfortably. Once they no longer share an LRU, the firehose of churn can no longer evict the expensive keys; each pool's eviction pressure is decoupled from the others'.

Routing is a client-side concern in look-aside, so it is just a function of the key:

```python
# Pool routing: pick the memcache pool from the key's access pattern.
# High-churn keys live in a small "wildcard" pool sized for cheap eviction;
# expensive, low-churn keys get a protected pool sized for their working set.
POOL_RULES = [
    ("session:",   "wildcard"),   # huge volume, cheap to rebuild
    ("feed:rank:", "wildcard"),   # recomputed constantly anyway
    ("profile:",   "default"),    # moderate churn
    ("expensive:", "lowchurn"),   # heavyweight aggregation, protect it
]

def pool_for(key):
    for prefix, pool in POOL_RULES:
        if key.startswith(prefix):
            return pool
    return "default"

def get(key):
    pool = pool_for(key)
    return mc_client_for(pool).get(key)
```

The second-order point is about *capacity planning*: once pools are separated, each pool can be provisioned for its own hit-rate target. You can measure the working-set size of the low-churn pool and give it exactly enough RAM to hold it, while keeping the wildcard pool deliberately small. Trying to do this with one shared pool is impossible — you cannot give "the expensive keys" more memory when they are sharing an LRU with everything else. This connects directly to the broader practice of [capacity planning for databases](/blog/software-development/database-scaling/capacity-planning-for-databases): the cache tier needs its own working-set sizing, per pool, not a single global guess.

## 5. Within a region: replication, regional pools, and cold clusters

**Senior rule of thumb: a region is many clusters, and the art is deciding what to replicate per-cluster versus share region-wide.**

A single frontend cluster has a fixed amount of cache. As traffic grows, you add clusters. Now you have a choice for each piece of data: replicate it into every cluster's cache (fast local hits, but N copies of the same bytes) or keep one shared copy for the whole region (one copy, but a cross-cluster hop on access). The paper splits the difference along the popularity axis.

![Cluster, region, cross-region: replicate within a cluster, share a regional pool, fan invalidations across regions](/imgs/blogs/scaling-memcache-at-facebook-5.webp)

The topology above shows all three scopes at once; for now focus on the master region on the left. **Popular items are replicated** across the per-cluster memcached tiers — every cluster keeps its own copy, so a hit is always local and cheap, and the replication cost is justified because these keys are requested everywhere. **Less-popular items go in a regional pool**: a single shared memcached pool that all the frontend clusters in the region consult. Replicating a low-traffic item into every cluster would waste RAM on copies that rarely get hit; sharing one regional copy serves them far more efficiently. The decision rule is essentially "is this item popular enough that a per-cluster copy pays for itself?" — popular, replicate; long-tail, share.

```python
# Region-level read: try the local (per-cluster) cache first, then the
# shared regional pool, then the DB. Popular keys are replicated locally;
# long-tail keys live only in the regional pool to avoid wasteful copies.
def regional_get(key):
    v = local_cluster_cache.get(key)        # replicated, hot, local
    if v is not None:
        return v
    if is_regional_pool_key(key):           # long-tail: shared copy
        v = regional_pool.get(key)
        if v is not None:
            return v
    row = db.query_for(key)                 # source of truth
    value = serialize(row)
    if is_regional_pool_key(key):
        regional_pool.set(key, value)       # long-tail keys fill the shared pool
    else:
        local_cluster_cache.set(key, value) # popular keys fill the local replica
    return value
```

### Cold cluster warmup

The harder within-region problem is bringing a *new* cluster online. A freshly started cluster has an empty cache, so every request is a miss, and if those misses all hit the database the new cluster becomes a denial-of-service attack against your own MySQL. The paper's answer is **cold cluster warmup**: while a cluster is cold, its clients fill misses by reading through a *warm* cluster's cache rather than the database. Only if the warm cluster also misses does the request fall through to MySQL.

![Cold cluster warmup: a cold cluster reads through a warm peer with a hold-off to avoid racing deletes](/imgs/blogs/scaling-memcache-at-facebook-6.webp)

The figure shows the flow and the trap. A miss in the cold cluster is satisfied by reading the warm cluster (1) and copying the value into the cold cluster's local cache (2); the database is only touched on the rare double-miss (3). This is a huge load reduction — the warm cluster's high hit rate shields the database during the warmup window.

But there is a race, shown in red on the right. A write deletes the key in *both* clusters. If the cold cluster, mid-warmup, reads a value from the warm peer and `set`s it *just after* the delete swept through, it re-installs a stale value into the cold cache — and because the delete already happened, nothing will clean it up. The fix is a brief **hold-off**: when a delete arrives during warmup, the cold cluster remembers it for a short window and refuses to honor a warmup `set` for that key during that window, so a value read from the warm peer cannot win a race against a concurrent delete. Once the hold-off window passes, normal delete-on-write resumes. The window is short enough that staleness is bounded to a couple of seconds and long enough to cover the delete-versus-set race. It is, again, a *bound* on staleness rather than an elimination of it — and it is exactly the kind of subtle ordering bug that makes cache warming dangerous in any system, not just Facebook's.

## 6. Across regions: one master, many replicas, and the invalidation race

**Senior rule of thumb: cross-region consistency is a delivery problem — get the invalidations to the replica regions reliably and in order, and accept that they will arrive slightly late.**

Now zoom all the way out. Facebook runs multiple geographic regions. To keep writes sane, there is a **single master region** for MySQL: all writes go to the master region's MySQL, which replicates asynchronously to read replicas in the other (replica) regions. Replica regions serve reads locally — both from their own memcached tiers and their own MySQL replicas — but forward writes to the master.

This is where the look-aside contract's reliance on deletes becomes a distributed-systems problem. A write in the master region changes a row and must delete the corresponding cache key *everywhere* — in the master region's clusters and in every replica region's clusters. And it must do so without racing the MySQL replication stream. Consider the race the paper specifically calls out:

1. A write commits to MySQL in the master region.
2. An invalidation (delete) for the affected key is broadcast and reaches a replica region *quickly*.
3. But the MySQL row change has *not yet* replicated to that region's MySQL replica.
4. A read in the replica region misses the (just-deleted) cache, falls through to the local MySQL replica, reads the **old** row, and re-caches the stale value.

The delete arrived, did its job, and then the cache was immediately re-poisoned by a read that hit a not-yet-updated database replica. You deleted too *early* relative to the data.

### mcsqueal: derive invalidations from the commit log

The structural fix is to stop sending invalidations from the *web server that did the write* and instead derive them from the **MySQL commit log** itself. A daemon the paper calls **mcsqueal** tails MySQL's commit log, extracts the keys affected by each committed transaction, and broadcasts the corresponding cache deletes.

![mcsqueal: tail the MySQL commit log and turn committed deletes into batched cache invalidations](/imgs/blogs/scaling-memcache-at-facebook-7.webp)

Why is reading from the log so much better than firing deletes from the web server? The figure makes the chain explicit, and there are three distinct wins:

- **Correctness via ordering.** The commit log is the authoritative, ordered record of what actually committed. Invalidations derived from it replay in commit order and only for transactions that *durably committed* — a web server that fired a delete optimistically before its transaction committed (and then rolled back) would have invalidated a key for nothing, or worse, in the wrong order.
- **Reliability via decoupling.** If a web server crashes after committing but before sending its delete, the invalidation is lost and you get a stale key. Tailing the log decouples invalidation from the request's fate: the delete is generated from the durable log regardless of whether the original web server is still alive.
- **Efficiency via batching.** mcsqueal batches many deletes together and routes them through `mcrouter`, which fans them out to the relevant memcached servers. Batching invalidations is dramatically cheaper than one network round trip per delete at this volume.

The same commit-log stream is what feeds the replica regions: the master region's mcsqueal broadcasts invalidations both within its own region and out to the replica regions, so deletes are sourced from one ordered, durable place. This is exactly the [change-data-capture and outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) applied to cache invalidation — the binlog *is* the change-data-capture stream, and mcsqueal is the consumer that translates row changes into cache deletes. If that pattern looks familiar from event-driven architectures, it should; Facebook was doing CDC for cache coherence before "CDC" was a buzzword.

### Remote markers: bounding staleness right after your own write

The commit-log pipeline handles the steady state, but it does not by itself fix the read-your-own-writes problem in a replica region *immediately* after a write. You wrote in the master region; the MySQL change has not replicated to your local replica yet; your next read in the replica region would see stale data.

The paper's mechanism for this is the **remote marker**. When a web server in a replica region performs a write that needs low staleness, it sets a special marker for the affected key indicating "the authoritative value for this key currently lives in the master region," and deletes the local cached value. A subsequent read that finds the marker knows the local MySQL replica may be behind, so it routes the read to the *master* region instead of trusting the local replica. Once replication catches up and the marker is cleared, reads go local again.

```python
# Remote marker: after a cross-region write, route reads to the master
# region until local MySQL replication has caught up. Bounds the window
# in which a replica region would otherwise serve its own stale data.
def replica_region_read(key):
    if remote_marker_set(key):           # we recently wrote; replica may lag
        return read_from_master_region(key)
    v = local_cache.get(key)
    if v is not None:
        return v
    row = local_mysql_replica.query(key) # safe: no pending local write
    local_cache.set(key, serialize(row))
    return row

def replica_region_write(key, changes):
    set_remote_marker(key)               # "authoritative value is in master"
    forward_write_to_master(key, changes)
    local_cache.delete(key)
    # marker is cleared once the master's commit-log invalidation +
    # MySQL replication for this key have propagated back to us
```

The marker is a small bet: pay a slower cross-region read for the brief window after a write, in exchange for not serving the writer their own stale data. It is a targeted, *per-key, time-bounded* escalation to strong-ish consistency layered on top of an otherwise eventually-consistent system — precisely the surgical approach you want, rather than making every read in the region pay the cross-region cost.

## 7. The consistency philosophy: bound staleness, do not pretend to eliminate it

It is worth pausing to name the thread running through every mechanism above, because it is the real lesson of the paper. Facebook did **not** try to make the cache strongly consistent with the database. They explicitly accepted **best-effort eventual consistency** and then engineered each mechanism to *bound* a specific failure:

| Mechanism | What it bounds | The trade accepted |
|---|---|---|
| Delete-on-write | Permanent staleness from racing `set`s | One extra cache miss after each write |
| Leases (stale-set guard) | A slow filler overwriting fresh data | A rejected `set`, then a clean refill |
| Leases (rate limit) | DB load from a thundering herd | A short wait + retry for non-holders |
| Pools | Expensive keys evicted by churn | Operational complexity of routing + sizing |
| Regional pool | Wasted RAM over-replicating long-tail | A cross-cluster hop for unpopular keys |
| Cold cluster warmup | DB storm when a cluster starts cold | A few seconds of bounded staleness via hold-off |
| Commit-log invalidation | Lost or reordered deletes | Invalidations arrive slightly after the commit |
| Remote markers | Replica region serving the writer's own stale data | A slower cross-region read for a brief window |

Read down the middle column: every entry is a *bound on a specific harm*, never a claim of perfection. This is the mature engineering stance for any large cache. The question is never "how do I make the cache always correct" — that is unaffordable — but "for each way the cache can be wrong, how badly and for how long, and is that bound acceptable for this data?" A like count being a second stale is fine; a password hash being stale is not, so it should not be cached this way at all. The art is matching the staleness bound to the data's tolerance, key by key.

This is the same instinct behind well-run [cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production): you choose a pattern not because it is "consistent" but because its specific staleness and load characteristics fit the data it protects.

## Case studies from production

The paper's mechanisms map onto failure modes that show up everywhere. Here are eight, drawn from the patterns the paper describes and the way the same problems recur in systems that are not Facebook. Each is the symptom, the wrong first hypothesis, the actual root cause, and the fix.

### 1. The expiring hot key that took down the database

The symptom: every few minutes, MySQL CPU spiked to 100% for ten seconds, then recovered, in a perfectly periodic sawtooth. The first hypothesis was a runaway cron job or a periodic analytics query. The actual root cause: a single extremely popular key — a global config blob read on every request — had a TTL, and when it expired, every web server missed it in the same instant and issued the identical `SELECT` to the database simultaneously. The thundering herd was the sawtooth. The fix was leases: the first miss after expiry gets the lease and refills; everyone else waits a few milliseconds and retries into a hit. The DB went from N identical reads per expiry to exactly one. The lesson the paper teaches is that *misses are correlated* — the dangerous miss is not the random one, it is the synchronized herd on a hot key.

### 2. The stale value that would not die

The symptom: a user changed their display name, saw the change, then half an hour later it reverted to the old name in some views and stayed that way. Cache TTL was an hour. The first hypothesis was replication lag. The actual root cause: an update-on-write race. Two requests updated related rows; both wrote MySQL, both `set` the cache, and the network reordered the `set`s so the cache ended up holding the older value while MySQL held the newer one. Nothing would fix it until the hour-long TTL expired. The fix was switching from update-on-write to delete-on-write plus leases. After the change, the worst case of a write was a single extra miss-and-reload, and the stale-set guard meant a slow `set` carrying an old snapshot was rejected outright. The paper's insistence on *delete, never update* is precisely this bug, prevented structurally.

### 3. The cache that kept evicting the wrong thing

The symptom: an expensive aggregation (a personalized recommendation list, costing a multi-second query to regenerate) had a cache hit rate that mysteriously hovered near zero despite plenty of total cache RAM. The first hypothesis was that the keys were not being set at all — a bug in the write path. The actual root cause: those expensive keys shared a memcached pool with a high-volume, high-churn key set (per-request session scratch data), and the churn constantly evicted the expensive keys under shared LRU before they could be reused. The fix was pools: route the session churn to a small wildcard pool and give the expensive aggregations their own pool sized to hold their working set. Hit rate jumped from near zero to the 90s. The lesson: *one shared LRU is a tragedy of the commons*, and the highest-volume tenant always wins.

### 4. The new cache cluster that DDoSed the database

The symptom: every time the team added a frontend cluster to handle growth, the database briefly fell over the moment the cluster took live traffic. The first hypothesis was that the new cluster was misconfigured and sending too many queries. The actual root cause: the cluster started with a *cold* cache, so 100% of its requests were misses, and all of them hit MySQL at once until the cache filled — a self-inflicted denial of service that scaled with how much traffic the new cluster took. The fix was cold cluster warmup: route the cold cluster's misses through a warm peer's cache instead of the DB during the warmup window, so MySQL only sees the rare double-miss. The database barely noticed the next cluster addition. The subtle follow-on bug — a warmup `set` racing a delete and re-installing stale data — was handled by the hold-off window.

### 5. The invalidation that arrived before the data

The symptom: in a replica region, users occasionally saw a value flip to *new*, then back to *old*, then to *new* again — a flicker right after a write. The first hypothesis was a client-side caching bug in the browser. The actual root cause: the cross-region invalidation (delete) reached the replica region faster than the MySQL replication stream carried the actual row change. The delete cleared the cache, the next read fell through to the not-yet-updated local MySQL replica, and re-cached the *old* value — which then got invalidated again when replication finally caught up. The fix was sourcing invalidations from the commit log (mcsqueal) so they are ordered relative to the data, plus remote markers to route post-write reads to the master region until the local replica caught up. The flicker disappeared. This is the canonical "you invalidated too early relative to your own replication" bug.

### 6. The lost delete after a web server crash

The symptom: rare, hard-to-reproduce stale keys that correlated loosely with deploys and host failures. The first hypothesis was bad TTLs. The actual root cause: the original invalidation path fired the cache delete *from the web server* after committing the DB write. When a web server crashed or was killed mid-deploy in the window between commit and delete, the delete was simply lost, leaving a stale key until expiry. The fix was deriving invalidations from the durable commit log instead of from the (mortal) web server: even if the web server vanishes, the committed transaction is in the log, and the log-tailing daemon generates the delete regardless. The lesson generalizes to any system that fires side effects from a request handler — *durably record the intent, then process it asynchronously*, which is exactly the outbox pattern.

### 7. The incast latency cliff under fan-out

The symptom: p99 latency for a particular wide page was fine at moderate load but fell off a cliff above a traffic threshold, with packet retransmits spiking on the client's NIC. The first hypothesis was an undersized cache tier. The actual root cause: the page fanned a single request out to dozens of cache servers in parallel, and at high load all their responses converged on the client's switch buffer at once, overflowing it and dropping packets — incast congestion. Adding cache servers made it *worse* by widening the fan-out. The fix was a sliding-window cap on outstanding requests per client: deliberately limiting concurrency so responses arrive in waves the buffer can absorb. The counterintuitive lesson the paper teaches: past a point, *more* parallelism increases latency, and the fix is to throttle yourself.

### 8. The like count that was wrong, and why that was fine

The symptom: during a flash event, the displayed like count on a viral post was visibly inconsistent across page loads — 10,402 on one refresh, 10,398 on the next, then 10,411. The first hypothesis was a serious cache-coherence bug requiring an emergency fix. The actual root cause: nothing was broken; this was the *designed* behavior. Like counts are a high-churn, low-value-of-freshness datum, cached with best-effort eventual consistency, and under a write storm the cache replicas simply converged at slightly different rates. The "fix" was to recognize that this was acceptable and not to over-engineer it. The lesson — and arguably the deepest one in the paper — is that *consistency is a property you spend money on*, and you spend it where the data demands it. A like count being a few off for a second costs nothing; paying for strong consistency on it would cost a great deal. Match the bound to the data.

## When to reach for this architecture, and when not to

The Facebook memcache architecture is a specific answer to a specific shape of problem. Reaching for it when you do not have that shape is how you end up with a cache layer more complex than the system it was meant to simplify.

**Reach for these patterns when:**

- You are **read-dominated** by a large margin — orders of magnitude more reads than writes — so the cache hit rate is the thing that decides whether your database survives.
- Your hot keys are **shared across many clients**, so misses are correlated and thundering herds are a real risk (leases earn their keep here).
- You have **distinct classes of cached data** with very different churn rates and recompute costs (pools earn their keep here).
- You run **multiple regions** with a clear master-for-writes topology and can tolerate eventual consistency for cached reads (the commit-log invalidation pipeline and remote markers earn their keep here).
- You can articulate, *per data type*, how stale is too stale — and most of your data tolerates seconds.

**Skip or simplify when:**

- Your working set fits in one cache server and your client count is small — you do not need pools, regional caches, or a commit-log pipeline; a plain look-aside cache with sensible TTLs is enough.
- Your data is **write-heavy** or read-your-own-writes-critical for *every* read — caching buys you little and the invalidation machinery costs a lot; consider whether you even need the cache.
- You require **strong consistency** on the cached values themselves — then a look-aside cache is the wrong tool; look at the data store's own consistency features or a read-through cache with synchronous invalidation, and accept the latency.
- You have **one region** — most of the hardest mechanisms here (master/replica regions, remote markers, cross-region invalidation) simply do not apply, and dragging them in is premature.
- Your team cannot yet operate the **simple** version reliably — leases, pools, and CDC invalidation are second-order optimizations layered on a working look-aside cache, not a starting point.

The throughline is the paper's own discipline: start from the dumb, correct look-aside contract (fill on read, delete on write, DB is truth), and add exactly one mechanism per failure mode you actually observe, each one buying a *bounded* improvement at a *named* cost. Leases when herds appear, pools when eviction misbehaves, regional caches when replication wastes RAM, commit-log invalidation when deletes go missing, remote markers when a region serves its own stale writes. Do not adopt the whole architecture on day one; adopt the contract, then let your incidents tell you which mechanism to add next.

## Further reading

- The primary source: [*Scaling Memcache at Facebook*](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala), Nishtala et al., NSDI 2013 — the paper this article reads closely.
- [Cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd) — the herd problem and its mitigations in depth, beyond the lease-specific treatment here.
- [Netflix EVCache: multi-region caching](/blog/software-development/database-scaling/netflix-evcache-multi-region-cache) — a different production answer to the same cross-region caching problem, worth comparing against the master/replica model here.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the general pattern that mcsqueal is a specific instance of.
- [Cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production) — choosing look-aside vs read-through vs write-through by the staleness and load characteristics each one buys.

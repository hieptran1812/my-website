---
title: "The Caching Hierarchy at Scale: What Belongs Where"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Caching is the highest-leverage read-scaling move, but the cache is really six layers with different latency, staleness, and cost profiles, and putting data in the wrong layer is its own bug."
tags: ["database-scaling", "caching", "redis", "cdn", "cache-hit-ratio", "in-process-cache", "distributed-cache", "near-cache", "ttl", "performance"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 35
---

The fastest read your system will ever serve is the one it never makes. That is the entire thesis of caching, and it is why caching sits at rung three of the [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — above tuning and scaling up, below replication and sharding. It is the highest-leverage read-scaling move you have, because every read that a cache answers is a read your database never sees, and the database is the expensive, hard-to-scale, one-way-door resource you are trying to protect.

But here is the mistake I watch teams make over and over: they treat "the cache" as a single thing. They stand up a Redis cluster, sprinkle `get`/`set` around the hot paths, and declare the read path "cached." Then they are confused when the Redis box is at 90% CPU, when a deploy melts the database for ninety seconds, when two app servers disagree about a user's account balance, or when their cloud bill grows a Redis line item that rivals the database it was supposed to relieve.

"The cache" is not one thing. It is a **hierarchy** of layers, each with a different latency budget, a different staleness profile, a different blast radius, and a different cost curve. Putting data in the wrong layer is not a missing optimization — it is an active bug, the same way putting a hot lock on the request path is a bug. A user's private session token cached at the CDN is a security incident. A 50-megabyte computed report duplicated into the in-process cache of two hundred app servers is an out-of-memory crash waiting for a traffic spike. The skill is not "add a cache." The skill is knowing *which layer each piece of data belongs in*, and why.

![The caching hierarchy: six layers, each with its own latency budget from nanoseconds to milliseconds](/imgs/blogs/the-caching-hierarchy-at-scale-1.webp)

The diagram above is the mental model for this entire post. Read a request from the top down: a browser cache hit costs tens of nanoseconds and zero network hops; a CDN edge hit costs single-digit milliseconds but serves a user a thousand kilometers away; an in-process LRU hit is back down in the hundreds of nanoseconds because it is just a hashmap lookup in local RAM; a distributed-cache hit costs one network round-trip, a few hundred microseconds; and only when *every* layer misses does the read fall through to the database, where a buffer-pool hit is a few milliseconds and a cold disk read is tens to hundreds. Each dashed arrow is a miss falling to the next tier down. The whole game of caching at scale is keeping reads as high up that stack as you can — and never, ever letting the wrong data live in the wrong tier.

## Why "just add a cache" is the wrong instinct

The intuition behind reaching for one cache is that caching is a single technique: keep a copy of the answer so you don't recompute it. That is true at the level of a sentence and false at the level of a system. The reason is that the six layers differ on axes that matter enormously in production, and a choice that is correct at one layer is catastrophic at another.

| What the team assumes | The naive view | The reality |
| --- | --- | --- |
| "Caching means Redis." | One distributed cache covers the read path. | Redis is *one* tier. Static assets belong at the CDN; per-request results belong in-process; only shared, expensive, cross-request data belongs in Redis. |
| "A cache is just a perf optimization." | Putting data in a cache can't hurt. | Wrong-layer caching is a correctness bug: private data at a shared CDN leaks; mutable data in a long-TTL local cache serves stale answers forever. |
| "Higher hit ratio is nice-to-have." | 90% and 99% are roughly the same. | DB load tracks the *miss* rate. Going 90% → 99% cuts surviving load 10×; the marginal miss is what melts the database. |
| "Local caching is always faster, so always do it." | In-process beats the network, full stop. | In-process duplicates memory N times and the copies drift; for shared mutable data that inconsistency is a bug, not a tradeoff. |
| "Cache is cheaper than scaling the DB." | Add RAM to Redis instead of upgrading Postgres. | Sometimes true, sometimes the opposite — the cost model depends on hit ratio, object size, and how much DB scale-up you'd avoid. |

Every row in that table is a real failure mode, and every one comes from collapsing six layers into one. Let me take the layers in turn — what each is for, what its budget is, and the gotcha that bites teams — then the hit-ratio economics that decide how hard each layer has to work, then local-versus-distributed, then the cost model, then the incidents.

> A senior rule of thumb: a cache layer is defined by three numbers and one verb — how fast it answers (latency), how shared it is (one copy or N), how stale it can get (TTL or invalidation), and what it does on a miss. If you can't state those four for a piece of data, you haven't decided where it belongs; you've just hidden the decision.

## 1. The read path: a fall-through through the layers

**Senior rule: design the read path as an ordered fall-through, where each layer answers in its own latency budget or passes the miss down — and make sure the cold path that touches every tier is rare, because it is your slowest read.**

Before we go layer by layer, look at the shape of a single read. A well-built cached read is not "check the cache, else hit the DB." It is a *chain*: check the fastest layer, and on a miss, fall to the next, populating the faster layers on the way back up so the next read for that key is faster.

![A read falls through the in-process cache, then Redis, then the database, populating faster layers on the way back](/imgs/blogs/the-caching-hierarchy-at-scale-2.webp)

The figure traces the three outcomes. A *local hit* in the in-process LRU returns in ~300 nanoseconds and never touches the network. A *Redis hit* costs the local miss plus one ~0.5 ms round-trip, and on the way back it fills the local cache so the next read is local. A *cold read* misses everything, costs a ~5 ms database read, and fills both Redis and the local cache. The cold path is the expensive one, and the entire point of the upper layers is to make it rare.

Here is that fall-through as runnable Python — an in-process LRU in front of Redis in front of a database. This is the read-through (cache-aside) pattern layered twice.

```python
import time
import json
from functools import lru_cache  # we'll roll our own TTL-aware LRU instead
from collections import OrderedDict
from typing import Callable, Optional
import redis  # redis-py 5.x

class TTLCache:
    """A tiny thread-unsafe in-process LRU with per-entry TTL.
    In real code use Caffeine (JVM) or cachetools.TTLCache (Python)."""
    def __init__(self, maxsize: int = 10_000, ttl_seconds: float = 5.0):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._store: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() > expires_at:      # lazily expire on read
            del self._store[key]
            return None
        self._store.move_to_end(key)            # mark as recently used
        return value

    def set(self, key: str, value: str) -> None:
        self._store[key] = (time.monotonic() + self.ttl, value)
        self._store.move_to_end(key)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)     # evict least-recently-used

# --- the layered lookup -----------------------------------------------------
local = TTLCache(maxsize=50_000, ttl_seconds=5.0)
rds = redis.Redis(host="cache.internal", port=6379, socket_timeout=0.05)

def layered_get(key: str, load_from_db: Callable[[str], str],
                redis_ttl: int = 300) -> str:
    # Tier 1: in-process LRU (~300 ns, no network)
    hit = local.get(key)
    if hit is not None:
        return hit

    # Tier 2: distributed cache (~0.5 ms, one hop). Fail open on Redis errors.
    try:
        raw = rds.get(key)
    except redis.exceptions.RedisError:
        raw = None                              # degrade to DB, never 500 the user
    if raw is not None:
        value = raw.decode()
        local.set(key, value)                   # backfill the fast tier
        return value

    # Tier 3: the database (~5 ms hit, source of truth). Cold path.
    value = load_from_db(key)
    try:
        rds.set(key, value, ex=redis_ttl)       # populate shared tier
    except redis.exceptions.RedisError:
        pass
    local.set(key, value)                       # and the local tier
    return value
```

Three details in that code are load-bearing and routinely missed. First, **the Redis call fails open** — a Redis timeout degrades to a database read, it does not propagate a 500 to the user. A cache that takes down your app when it is unavailable has turned an optimization into a single point of failure (more on that in the incidents). Second, **the in-process tier has a much shorter TTL (5 s) than Redis (300 s)** — because the local tier cannot be invalidated from the outside, its TTL is your only staleness bound, so you keep it short. Third, **a cold read populates both upper tiers on the way back**, so the expensive fall-through happens once per key per TTL window, not once per request.

### Second-order optimization: the thundering herd on a cold key

The fall-through above has a subtle failure mode: when a hot key expires and a thousand concurrent requests all miss simultaneously, all thousand fall through to the database at once. That is a *cache stampede*, and it can take down a database that was perfectly healthy a millisecond earlier. The fix is **request coalescing** (also called single-flight): the first miss takes a lock, recomputes, and populates the cache; the other 999 wait for that one result instead of dogpiling the DB.

```python
import threading

_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

def coalesced_get(key, load_from_db, redis_ttl=300):
    hit = local.get(key) or _redis_get(key)
    if hit is not None:
        return hit
    with _inflight_lock:
        ev = _inflight.get(key)
        if ev is None:                 # we are the leader; recompute
            ev = threading.Event()
            _inflight[key] = ev
            leader = True
        else:
            leader = False             # someone else is recomputing
    if not leader:
        ev.wait(timeout=2.0)           # wait for the leader, don't hit the DB
        return local.get(key) or _redis_get(key)
    try:
        value = layered_get(key, load_from_db, redis_ttl)
        return value
    finally:
        with _inflight_lock:
            del _inflight[key]
        ev.set()                       # release the waiters
```

In a distributed system the leader-election has to happen across nodes (a short-lived Redis lock with `SET key value NX EX 5`), but the principle is identical: collapse the herd into one recompute. We will revisit stampedes in the cost section, because they are the reason a marginal miss is so expensive.

## 2. Hit-ratio economics: why the marginal miss is what kills you

**Senior rule: stop thinking about hit ratio and start thinking about miss ratio, because database load is proportional to misses, and each extra "nine" of hit ratio strips an order of magnitude off the load your database actually sees.**

Here is the single most counterintuitive fact in caching, and the one that separates engineers who have run a cache at scale from those who haven't: **the difference between a 90% and a 99% hit ratio is not 9% — it is 10×.** Because the database load is proportional to the *miss* rate, and the miss rate went from 10% to 1%.

![Each extra nine of hit ratio strips an order of magnitude off the surviving database load, on a log scale](/imgs/blogs/the-caching-hierarchy-at-scale-3.webp)

The chart makes the arithmetic visceral. Take 100,000 reads. At a 90% hit ratio, 10,000 of them miss and hit the database. At 99%, only 1,000 miss. At 99.9%, 100. At 99.99%, 10. Each added nine divides the surviving database load by ten. The y-axis is log-scaled because that is the only way to show four orders of magnitude on one chart, and the descending steps *are* the economics: the database does not feel your hit ratio, it feels your miss count.

This is why a cache that "mostly works" can still melt your database. Imagine you are running comfortably at a 99.5% hit ratio, your database serving 500 misses per 100k reads without breaking a sweat. Then a deploy flushes the in-process caches across the fleet, or a popular key expires during peak, and your hit ratio dips to 98% for thirty seconds. You did not lose 1.5% of your cache — you *quadrupled* the database load (from 0.5% miss to 2% miss), and if the database had less than 4× headroom, it is now in an incident. The marginal miss, not the average hit, is what you must engineer against. This ties directly to [capacity planning](/blog/software-development/database-scaling/capacity-planning-for-databases): the cache hit ratio is one of the four budgets that decide when you hit the wall, and it is the one that moves fastest and least predictably.

You can put a number on it. The **effective latency** of a cached read is the hit-weighted average of the fast and slow paths:

$$L_{\text{eff}} = h \cdot L_{\text{hit}} + (1 - h) \cdot L_{\text{miss}}$$

where $h$ is the hit ratio, $L_{\text{hit}}$ is the latency of a cache hit, and $L_{\text{miss}}$ is the latency of the full fall-through to the database. Symbolically that looks innocent. Plug in numbers and the tyranny of the slow path appears:

```python
def effective_latency_ms(hit_ratio: float, hit_ms: float, miss_ms: float) -> float:
    return hit_ratio * hit_ms + (1 - hit_ratio) * miss_ms

HIT_MS = 0.3      # local + redis blended hit
MISS_MS = 8.0     # cold fall-through to the DB

for h in (0.90, 0.95, 0.99, 0.999):
    L = effective_latency_ms(h, HIT_MS, MISS_MS)
    miss_share = (1 - h) * MISS_MS / L * 100
    print(f"hit={h:6.3f}  L_eff={L:5.2f} ms  "
          f"misses are {miss_share:4.1f}% of the latency budget")
```

```console
hit= 0.900  L_eff= 1.07 ms  misses are 74.8% of the latency budget
hit= 0.950  L_eff= 0.68 ms  misses are 58.4% of the latency budget
hit= 0.990  L_eff= 0.38 ms  misses are 21.1% of the latency budget
hit= 0.999  L_eff= 0.31 ms  misses are  2.6% of the latency budget
```

Read the right-hand column. At a 90% hit ratio, the 10% of requests that miss consume *three-quarters* of your average latency budget. A handful of slow reads dominates the mean — and the tail (p99) is even worse, because the p99 *is* the miss path. This is why chasing the last few nines of hit ratio is worth real engineering effort: each nine simultaneously cuts database load 10× and shrinks the slow path's share of your latency.

| Hit ratio | Misses per 100k | DB load vs 99.9% | What it takes to get here |
| --- | --- | --- | --- |
| 90% | 10,000 | 100× | A naive cache-aside with short TTLs |
| 99% | 1,000 | 10× | Tuned TTLs, decent key design, warm caches |
| 99.9% | 100 | 1× (baseline) | Near-cache tier, stampede protection, pre-warming |
| 99.99% | 10 | 0.1× | Multi-tier, proactive refresh, careful invalidation |

### Second-order optimization: a cold cache is a latent outage

The numbers above assume a *warm* cache. The moment of maximum danger is a *cold* one — right after a deploy, a Redis failover, or a cache flush, when your hit ratio is briefly near zero and *every* read falls through to a database sized for 99.9% hits. A database that handles 100 misses per 100k comfortably will not survive 100,000 misses per 100k for even a few seconds. This is why mature systems **pre-warm** caches before taking traffic, **stagger** deploys so the whole fleet's local caches don't go cold at once, and rate-limit the fall-through so a cold start degrades latency rather than detonating the database. A cache you cannot warm safely is a database outage scheduled for your next deploy.

## 3. What belongs where: matching data shape to layer

**Senior rule: choose the layer by the data's shape — how shared it is, how fast it changes, how big it is, and who is allowed to see it — not by which cache you happened to stand up first.**

Now the central question of the post. Given a piece of data, which layer is its home? The answer falls out of four properties: *sharing* (is this the same for all users, or per-user?), *mutability* (static, slowly-changing, or per-request?), *size*, and *privacy*. Map those onto the layers and the right home becomes obvious — and the wrong homes become visibly wrong.

![A matrix matching each data shape to the cache layer whose latency, sharing, and staleness profile fits, with the right home in green](/imgs/blogs/the-caching-hierarchy-at-scale-4.webp)

The green cell in each row of the matrix is the right home; the red and amber cells are the wrong ones, labeled with *why* they are wrong. Walk the rows:

- **Static assets** (JS bundles, images, CSS, fonts) belong at the **CDN/edge** with a long TTL and content-hashed URLs. They are identical for every user, they change only on deploy, and serving them from the edge cuts both latency and origin bandwidth. Caching them in your in-process LRU wastes application RAM on bytes the CDN serves for free; pushing them through Redis adds a pointless hop.
- **Expensive computed read models** (a denormalized product page, an aggregated dashboard, a rendered feed) belong in the **distributed cache**. They are expensive to compute, shared across users (or across a tenant), and you want every app node to see the same cached copy and to share the cost of computing it once. Putting them only in-process means N app nodes each recompute and each store their own copy — N× the compute, N× the memory.
- **Per-request memoization** (a value you look up three times while handling one request) belongs in an **in-process** cache scoped to the request. It is needed for microseconds, by one node, for one request. A network hop to Redis to memoize within a single request is slower than just recomputing — the hop costs more than the work.
- **Small, hot reference data** (feature flags, a currency table, a list of country codes, a small config) belongs **replicated in-process** on every node. It is tiny, read on nearly every request, and changes rarely, so the inconsistency window of a short local TTL is acceptable and the nanosecond local read is worth duplicating a few kilobytes everywhere.
- **Sessions and tokens** belong in the **distributed cache** with the TTL set to the session expiry. They are per-user and must be consistent across nodes (a user load-balanced to a different app server must see the same session), they must survive an app restart, and — critically — they must **never** be cached at a shared CDN, where one user could be served another's token. That last one is not a performance bug; it is a security incident.

| Data shape | Right home | TTL strategy | Key design |
| --- | --- | --- | --- |
| Static asset | CDN / edge | Long (1y) + content hash in URL | `app.4f9a2c.js`; URL *is* the version |
| Computed read model | Distributed cache | Medium (minutes) + explicit invalidation on write | `product:{id}:v{schema}` |
| Per-request memo | In-process (request-scoped) | Lifetime of the request | local variable / request context |
| Hot reference data | In-process (replicated) | Short (seconds) so all nodes converge | `flags:all`, refreshed on a timer |
| Session / token | Distributed cache | TTL = session expiry | `sess:{uuid}`, never at the edge |

The TTL column deserves its own rule. **A TTL is a staleness budget, not a memory-management knob.** You do not set a TTL to "free up space" — the LRU eviction does that. You set a TTL to bound how wrong a cached answer is allowed to be. A currency-conversion table cached for 60 seconds means "we accept being up to 60 seconds stale on exchange rates." If 60 seconds of staleness is a problem, the TTL is wrong; if it is fine, a 5-second TTL is just needless misses. Set the TTL from the business tolerance for staleness, and use explicit invalidation when even that is too loose.

### Second-order optimization: the key is the contract

The most under-appreciated part of caching is **key design**, because the key encodes everything about correctness. Three rules I enforce in review: First, **version the key with the schema**, e.g. `product:{id}:v3`. When you change the shape of the cached object, bump the version so old and new code never read each other's incompatible blobs — a deploy then "invalidates" the entire cache for free by simply reading new keys, with no flush. Second, **never cache under a key you cannot reconstruct for invalidation**. If you write `user:{id}:dashboard` but your write path can only compute `user:{id}`, you can never invalidate the dashboard on a write — design the key so the writer can name it. Third, **include in the key every input that changes the output** — locale, tenant, feature-flag cohort. The classic bug is caching a localized page under `page:home` and serving the French version to everyone for a TTL because the first request happened to be French.

## 4. Local vs distributed: the two-tier trap

**Senior rule: an in-process cache is the fastest tier you have and the most dangerous for shared mutable data, because each node holds its own copy and those copies drift; reach for it for small, hot, tolerably-stale data, and reach for the distributed cache when consistency across nodes matters.**

The two tiers most teams actually operate are the in-process (local) cache and the distributed cache, and the choice between them is a genuine tradeoff, not a clear win for either.

![In-process cache versus distributed cache, comparing speed, memory, consistency, and durability across nodes](/imgs/blogs/the-caching-hierarchy-at-scale-5.webp)

The before/after lays the two side by side. The in-process cache wins on raw speed — a hashmap lookup in local heap, ~300 ns, no serialization, no network — but it pays for that speed three ways: each of your N app nodes keeps its **own copy**, so a 1 GB working set becomes N gigabytes of total RAM; the copies **drift**, because a write on node A's local cache is invisible to node B's; and every cache goes **cold on deploy**, so your fleet's aggregate hit ratio craters every time you ship. The distributed cache inverts the trade: one shared copy that every node sees consistently, RAM stored once instead of N times, surviving app restarts — at the cost of a ~0.5 ms network hop on every access and a new operational dependency that can become a single point of failure.

The decision is mostly about that drift. For **immutable or tolerably-stale** data — a config that changes weekly, a feature flag with a 5-second convergence window, reference tables — the drift is harmless and the speed is free, so cache it locally. For **shared mutable** data where two nodes disagreeing is a *bug* — an account balance, an inventory count, a permission set — the local cache's inconsistency is unacceptable and you want the distributed cache's single source of truth.

| Property | In-process (local) | Distributed (Redis/Memcached) |
| --- | --- | --- |
| Hit latency | ~100 ns – 1 µs | ~0.2 – 1 ms (one hop) |
| Memory cost | N × (one copy per node) | 1 × (shared) |
| Consistency across nodes | None — copies drift | Strong — one shared copy |
| Survives app restart | No (cold on every deploy) | Yes |
| Invalidation | TTL only (can't push from outside) | Explicit `DEL` reaches all readers |
| Failure blast radius | Local, fails closed silently | Shared; a SPOF if you fail closed |
| Best for | Small, hot, tolerably-stale data | Shared, expensive, mutable data |

So why not have both? You can — and the pattern of putting a small local cache *in front of* the distributed cache is called a **near cache**. It is the best of both for read-heavy workloads: the local tier absorbs the hottest keys at nanosecond cost, and the distributed tier backs it with a consistent shared copy for everything else. But the near cache reintroduces the exact problem the distributed cache solved — and that is the trap.

![The two-tier near-cache: a write updates Redis and the database but the node-local near caches keep serving stale data until their TTL expires](/imgs/blogs/the-caching-hierarchy-at-scale-6.webp)

Here is the invalidation problem in one picture. App nodes A and B each keep a near cache holding `key=42 → v1`. A writer updates the database and Redis to `key=42 → v2`. Redis is now fresh; the database is fresh; the write *propagated correctly to the distributed tier*. But the write **never reached the near caches** — they are node-local heap, invisible to the writer — so A and B keep serving the stale `v1` until their local TTL expires. With a 5-second near-cache TTL, you have a 5-second window of bounded staleness on every node. That bounded staleness is the *price* of the near-cache hop you saved. The art of near caching is keeping that TTL short enough that the staleness is acceptable, or wiring up an invalidation broadcast (a Redis pub/sub channel that tells every node "drop key 42") to collapse the window — at the cost of yet more complexity and a new way for invalidations to be lost.

### Second-order optimization: pub/sub invalidation is at-most-once

If you do reach for broadcast invalidation, understand its delivery guarantee. Redis pub/sub is **fire-and-forget**: if a node is briefly disconnected when the invalidation fires, it misses the message and keeps serving stale data *past* the TTL window you were counting on. This is exactly the model Redis's own client-side caching (RESP3 "tracking") uses, and the mitigation is always the same: pub/sub is an *optimization* that shortens the staleness window, but the TTL is the *guarantee* that bounds it. Never let the broadcast be the only thing standing between you and unbounded staleness — keep the TTL as the floor, and treat the invalidation as a best-effort accelerant. For the deeper mechanics of running Redis as the distributed tier — pipelining, eviction policies, persistence tradeoffs, and the keyspace-notification machinery — see [Redis in production](/blog/software-development/database/redis-applications-and-optimization).

## 5. The cost model: when a cache is cheaper than more database

**Senior rule: a cache earns its place when the RAM to hit a target hit ratio costs less than the database scale-up it lets you avoid — and that comparison depends on your hit ratio, your object sizes, and how lumpy the next DB tier is.**

Caching is not free, and "add a cache" is not automatically cheaper than "scale the database." The honest comparison is: *the cost of the cache tier* versus *the cost of the database capacity the cache lets you not buy*. Sometimes the cache wins by a mile; sometimes the database scale-up is the cheaper, simpler move.

The cache cost has two parts: the **RAM** to hold your working set at the hit ratio you want, and the **operational** cost of running another stateful system (a Redis cluster you must monitor, fail over, and patch). The database savings is the scale-up tier you avoid — and database scale-up is *lumpy*: you don't buy 1.3× a database, you jump from one instance class to the next, often doubling cost for the next tier up. That lumpiness is what makes caches economical: a cache lets you buy capacity in fine-grained increments (more RAM) instead of coarse, expensive jumps (a 2× bigger primary).

Here is a back-of-envelope model you can adapt:

```python
def cache_vs_db_scaleup(reads_per_sec, hit_ratio,
                        cache_gb, cache_cost_per_gb_month,
                        db_cost_current, db_cost_next_tier):
    """Compare a cache tier against the next DB scale-up step."""
    misses_per_sec = reads_per_sec * (1 - hit_ratio)
    served_by_cache = reads_per_sec * hit_ratio

    cache_monthly = cache_gb * cache_cost_per_gb_month
    db_scaleup_delta = db_cost_next_tier - db_cost_current

    print(f"cache absorbs {served_by_cache:,.0f} reads/s, "
          f"DB sees {misses_per_sec:,.0f} reads/s")
    print(f"cache tier:        ${cache_monthly:,.0f}/mo")
    print(f"DB scale-up delta: ${db_scaleup_delta:,.0f}/mo")
    if cache_monthly < db_scaleup_delta:
        print(f"-> cache is cheaper by "
              f"${db_scaleup_delta - cache_monthly:,.0f}/mo")
    else:
        print(f"-> DB scale-up is cheaper by "
              f"${cache_monthly - db_scaleup_delta:,.0f}/mo")

# 50k reads/s, 99% hit, a 64 GB Redis tier vs doubling the DB instance
cache_vs_db_scaleup(
    reads_per_sec=50_000, hit_ratio=0.99,
    cache_gb=64, cache_cost_per_gb_month=12,
    db_cost_current=4_000, db_cost_next_tier=8_000,
)
```

```console
cache absorbs 49,500 reads/s, DB sees 500 reads/s
cache tier:        $768/mo
DB scale-up delta: $4,000/mo
-> cache is cheaper by $3,232/mo
```

At a 99% hit ratio, a $768/month Redis tier absorbs 49,500 reads per second so the database only sees 500 — and that is *vastly* cheaper than the $4,000/month it would cost to double the database to handle the full 50,000. This is the common case, and it is why caching is rung three: it buys an enormous amount of read capacity per dollar.

But the model also shows when caching *loses*. Flip the inputs: if your hit ratio is only 70% (because your access pattern has no hot keys — a long-tail catalog where every product is read about equally), the cache absorbs far less, the database still sees 15,000 reads per second, and you may *still* need the scale-up. Now you are paying for *both* the cache and the bigger database. **A cache with a low hit ratio is the worst of both worlds**: you pay for the cache, you pay for the database, and you have added a dependency. The cost model is only favorable when the access pattern is skewed enough to give you a high hit ratio at a reasonable cache size — which, for most real workloads, it is, because real access patterns are heavily Zipfian. But verify it; don't assume it.

| Scenario | Hit ratio | Cache verdict |
| --- | --- | --- |
| Hot-key workload (social feed, popular products) | 99%+ | Cache wins big; far cheaper than DB scale-up |
| Moderately skewed (typical web app) | 95–99% | Cache wins; the default right answer |
| Flat long-tail (huge catalog, even access) | 70–90% | Cache may lose; you pay for both — measure first |
| Write-heavy with low re-read | <50% | Skip the cache; it adds cost and staleness for nothing |

### Second-order optimization: object size silently inflates the cost

The model above counts gigabytes, and gigabytes are where caching budgets quietly blow up. A cache holding a million 200-byte session blobs is 200 MB — trivial. A cache holding a million 200-KB rendered pages is 200 GB — a different cost class entirely, and one where the network cost of *transferring* a 200-KB object on every hit may eat the latency win. Large objects also evict more aggressively (fewer fit in RAM, so your hit ratio drops), and they make the cold-fill path expensive. The rule: **cache small, hot, expensive-to-compute objects; do not cache large objects just because they are slow to fetch** — for those, scale the thing that produces them, or cache a compact derived form (the IDs, not the hydrated objects). Caching the wrong-sized object is how a cache that looked cheap in the design doc becomes the largest line item on the bill.

## 6. The cache cliff: visualizing the marginal miss

We have asserted twice now that the marginal miss is what kills you. It is worth *seeing*, because the intuition that "a small drop in hit ratio is a small problem" is so durable that prose alone rarely dislodges it.

<figure class="blog-anim">
<svg viewBox="0 0 640 280" role="img" aria-label="As cache hit ratio slips from 99.9 percent to 99 percent, surviving database load jumps tenfold, from a calm bar to a bar that hits the ceiling" style="width:100%;height:auto;max-width:760px">
<title>The cache cliff: a small drop in hit ratio multiplies the surviving database load</title>
<style>
.a7-frame{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.a7-ceil{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:6 6}
.a7-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a7-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a7-cap{font:700 16px ui-sans-serif,system-ui;text-anchor:middle}
.a7-calm{fill:var(--accent,#2563eb)}
.a7-hot{fill:#dc2626}
@keyframes a7-grow{0%,30%{height:30px;y:200}55%,90%{height:170px;y:60}100%{height:30px;y:200}}
@keyframes a7-fadeCalm{0%,30%{opacity:1}50%,95%{opacity:0}100%{opacity:1}}
@keyframes a7-fadeHot{0%,30%{opacity:0}50%,95%{opacity:1}100%{opacity:0}}
.a7-bar{animation:a7-grow 9s ease-in-out infinite}
.a7-onCalm{animation:a7-fadeCalm 9s ease-in-out infinite}
.a7-onHot{animation:a7-fadeHot 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a7-bar{animation:none;height:170px;y:60}.a7-onCalm{animation:none;opacity:0}.a7-onHot{animation:none;opacity:1}}
</style>
<text class="a7-lbl" x="320" y="28">Surviving DB load vs cache hit ratio</text>
<line class="a7-ceil" x1="120" y1="60" x2="520" y2="60"/>
<text class="a7-sub" x="320" y="52">DB throughput ceiling</text>
<rect class="a7-frame" x="120" y="60" width="400" height="170" rx="4"/>
<rect class="a7-bar a7-calm a7-onCalm" x="280" y="200" width="80" height="30" rx="4"/>
<rect class="a7-bar a7-hot a7-onHot" x="280" y="200" width="80" height="30" rx="4"/>
<text class="a7-cap a7-onCalm" x="320" y="262" fill="var(--accent,#2563eb)">99.9% hit: 100 misses/100k, DB calm</text>
<text class="a7-cap a7-onHot" x="320" y="262" fill="#dc2626">99% hit: 1,000 misses/100k, DB at the wall</text>
</svg>
<figcaption>One lost nine of hit ratio multiplies the misses by ten; the surviving DB load leaps from a calm bar to one slamming the throughput ceiling. The marginal miss, not the average hit, is what melts the database under load.</figcaption>
</figure>

Watch the bar. At 99.9% the surviving load is a small bar, comfortably below the database's throughput ceiling. Lose one nine — drop to 99% — and the bar grows tenfold and slams into the ceiling. Nothing about the *cache* changed dramatically; the hit ratio moved by less than a percentage point. But the *database* experienced a 10× load spike, because it lives downstream of the miss rate, and the miss rate is the small number that got ten times bigger. This is the cache cliff, and it is why an alert on hit ratio is one of the most important signals you can have: a hit ratio sliding from 99.9% toward 99% is not a yellow warning, it is a database about to hit the wall.

## Case studies from production

Caching incidents are some of the most instructive in our field, because they almost always come from the *interaction* between layers, not from a single layer being wrong. Here are eight, with rough numbers — the shapes are real even where I've rounded the figures.

### 1. The Facebook memcache thundering herd

Facebook's memcache tier is the canonical large-scale caching system, and its most famous failure mode is the stampede. A single extremely hot key — say, the data for a celebrity's profile — expires, and in the milliseconds before it is repopulated, every front-end that wants it misses simultaneously and slams the database. At Facebook's read volume, that is hundreds of thousands of simultaneous misses on one key. Their published mitigation (in the "Scaling Memcache at Facebook" paper) is *leases*: the first miss gets a lease token to recompute, and concurrent misses for the same key are told to wait briefly and retry the cache rather than dogpile the database. The lesson is the one from section 1: the marginal miss on a hot key is not one database read, it is a herd, and a cache without stampede protection is a database amplifier pointed at your own primary.

### 2. The Redis-as-SPOF outage

A team I worked with cached aggressively in Redis and, in the read path, treated a Redis error as a fatal error — a failed `get` threw, and the request 500'd. This was fine for two years. Then a routine Redis failover took the cluster unavailable for about forty seconds, and during those forty seconds *every cached read failed*, which meant *every request that touched the cache 500'd* — a total outage, caused by the unavailability of a component whose entire job was to be an optional optimization. The fix was three lines: catch the Redis error, fall through to the database, log a metric. The database briefly saw the full uncached load (it had just enough headroom to survive the cold spike), but the site stayed up. The lesson: **a cache must fail open**. The moment your cache can take down your app, it has stopped being a cache and become a fragile dependency in your critical path.

### 3. The private data at the shared CDN

An e-commerce site put their CDN in front of the entire application, including authenticated pages, and configured caching by URL. The "account" page at `/account` was cacheable. The CDN cached the *first* user's rendered account page — name, address, order history — and served it to *every subsequent user* who hit `/account` until the TTL expired. This is the wrong-layer-as-security-incident case from section 3: per-user private data was placed in a layer (the shared CDN) whose entire premise is "one copy for everyone." The fix was to mark authenticated routes `Cache-Control: private, no-store` and only ever cache static assets and genuinely public pages at the edge. The lesson: the CDN does not know your data is private; the `Vary` and `Cache-Control` headers are the only thing standing between "fast" and "leaking other users' data."

### 4. The in-process cache OOM after a feature launch

A service cached rendered fragments in an in-process Caffeine cache with a size limit set in *entries* (100,000) rather than *bytes*. For two years the fragments averaged a few kilobytes, so the cache used a few hundred megabytes — fine. Then a product launch introduced a new fragment type that embedded a large JSON blob, averaging 400 KB. The cache happily held 100,000 of those: 40 GB, on an 8 GB heap. Every node OOM-killed within minutes of the launch, across the fleet, simultaneously. The lesson from section 5: **size your in-process cache by bytes, not entries**, because the entry count is constant but the object size is not, and an entry-bounded cache is an unbounded cache the day someone caches a bigger object.

### 5. The near-cache staleness on a balance display

A fintech app put a near cache in front of Redis for account data with a 30-second local TTL, which seemed harmless. Then a user made a transfer, saw their balance update on the page (because that request happened to route to node A, whose near cache they had just invalidated), refreshed, got routed to node B — whose near cache still held the *old* balance for up to 30 seconds — and saw their money apparently vanish and reappear. No data was ever wrong in the database or Redis; the near caches on different nodes simply disagreed for the TTL window, exactly the section-4 invalidation problem. The fix was to drop the near cache for balance data (it is shared mutable data that demands consistency, so it belongs in the distributed tier only) while keeping it for genuinely static reference data. The lesson: near caching is for tolerably-stale data; account balances are not tolerably stale.

### 6. The cache key that forgot the locale

A media site cached its homepage under the key `home`, served from Redis with a 5-minute TTL. The homepage was localized by `Accept-Language`. Whichever request populated the cache first — often a bot from a random region — set the language for *everyone* for the next five minutes. German users got a Japanese homepage; English users got German. The bug is the section-3 key-design rule: the cache key omitted an input (locale) that changes the output. The fix was `home:{locale}`. The lesson: **the cache key must include every input that varies the output** — locale, tenant, currency, feature-flag cohort, A/B bucket — or you will serve one user's variant to another.

### 7. The deploy that cold-started the whole fleet

A team ran a healthy 99.7% hit ratio on in-process caches and was comfortable with their database headroom. Their deploy process rolled the entire fleet in a tight window — all nodes restarted within about ninety seconds — and every node's in-process cache went cold at once. For roughly a minute the fleet-wide hit ratio was near zero, the database saw something like 50× its normal miss load, and it fell over, turning a routine deploy into a fifteen-minute outage. This is the section-2 cold-cache cliff combined with the section-4 "cold on every deploy" property of local caches. The fix was twofold: stagger the deploy so only a fraction of the fleet cold-starts at any moment, and put a *distributed* cache behind the in-process one so a cold local cache falls through to a warm Redis rather than all the way to the database. The lesson: a local-only cache turns every full-fleet deploy into a database load test you didn't schedule.

### 8. The 70% hit ratio that cost double

A catalog company with millions of roughly-equally-accessed SKUs added a large Redis tier expecting it to relieve their database. But the access pattern was nearly flat — no hot keys — so the hit ratio settled at about 70%. The database still saw 30% of the read load (plenty to require the scale-up they had hoped to avoid), and now they were paying for a large Redis cluster *on top of* the bigger database. This is the section-5 "worst of both worlds" case: a cache only pays off when the access pattern is skewed enough to deliver a high hit ratio at a sane cache size, and a flat long-tail does not. The fix was to stop trying to cache the full catalog and instead cache only the genuinely hot subset (the top few thousand SKUs that drove most traffic) in a much smaller tier, and to scale the database for the long tail. The lesson: **measure your hit ratio against your real access pattern before sizing the cache** — a cache is an answer to skew, not to volume.

## When to reach for a cache layer, and when not to

Caching is rung three of the ladder for a reason: it is cheap, high-leverage, and it protects the expensive resource. But like every rung, it has conditions where it is the right move and anti-patterns where it is a liability.

**Reach for a given cache layer when:**

- **The read is hot and re-read** — the same data is requested far more often than it changes. A high re-read ratio is what makes a cache pay; without it, you are paying for staleness and a dependency with no hit-ratio reward.
- **The data's shape matches the layer** — static and public → CDN; shared, expensive, mutable → distributed cache; small, hot, tolerably-stale → in-process. Match the four properties (sharing, mutability, size, privacy) to the layer before you write a single `set`.
- **The access pattern is skewed** — a Zipfian distribution where a small fraction of keys carries most of the traffic gives you a high hit ratio at a small cache size, which is when the cost model is favorable.
- **You can name the key for invalidation** — your write path can compute the exact key it must invalidate, so cached data does not outlive its correctness.
- **The cache can fail open** — your read path degrades to the source of truth on a cache error, so the cache is an optimization and not a single point of failure.

**Skip the cache (or that layer) when:**

- **The hit ratio will be low** — a flat long-tail access pattern means you pay for the cache *and* the database, the worst of both worlds. Measure first.
- **The data is write-heavy with little re-read** — if data changes almost as often as it is read, caching adds a staleness window and an invalidation burden for almost no hit-ratio benefit.
- **The data is per-user and you are tempted to put it at a shared layer** — never cache private data at the CDN or any shared tier without a per-user key; the failure mode is a security incident, not a slow page.
- **The objects are large and the network transfer eats the win** — caching a 200-KB object whose hit costs a 200-KB transfer may be slower than recomputing a compact form; cache the small derived data, not the big blob.
- **You haven't measured the database is actually the bottleneck** — caching a database that isn't your constraint is effort spent on the wrong rung. Climb the ladder in order: measure, tune, scale up, *then* cache.

The discipline the whole post comes down to is this: caching is not one decision, it is a decision *per piece of data*, made by asking how shared it is, how fast it changes, how big it is, and who may see it — and then placing it in the one layer whose latency, staleness, and cost profile fits. Get that right and the cache is the cheapest order of magnitude of read capacity you will ever buy. Get it wrong and you have built a faster way to serve incorrect, stale, or other people's data. The next post, [cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production), takes the layer placement as given and digs into the *write* path — cache-aside versus write-through versus write-behind, and the invalidation strategies that keep these layers honest under a steady stream of mutations.

## Further reading

- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — where caching sits in the ordered ladder of scaling moves, and why you climb to it before replication and sharding.
- [Capacity planning for databases](/blog/software-development/database-scaling/capacity-planning-for-databases) — the cache hit ratio as one of the four budgets that decide your next scaling deadline.
- [Redis in production](/blog/software-development/database/redis-applications-and-optimization) — operating the distributed tier: eviction policies, persistence, pipelining, and client-side caching.
- [Cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production) — the write path and invalidation strategies that complement this post's layer placement.
- *Scaling Memcache at Facebook* (Nishtala et al., NSDI 2013) — the canonical paper on leases, stampede control, and running a cache tier at planetary scale.

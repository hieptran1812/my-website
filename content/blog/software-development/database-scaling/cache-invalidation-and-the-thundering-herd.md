---
title: "Cache Invalidation and the Thundering Herd: Keeping a Hot Key From Killing Your Database"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Cache invalidation is the famous hard problem, but the cache stampede is the one that takes down production — here is how to master both, with runnable single-flight, XFetch, jitter, and stale-while-revalidate code."
tags: ["caching", "cache-invalidation", "thundering-herd", "cache-stampede", "redis", "single-flight", "database-scaling", "system-design", "reliability", "performance"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 40
---

There are two famous quotes about caching, and almost everyone gets the wrong one stuck in their head. The first — Phil Karlton's "there are only two hard things in computer science: cache invalidation and naming things" — is the one people quote at parties. The second is the one that pages you at 3 a.m.: a single popular key expires, ten thousand in-flight requests all miss at the same instant, every one of them falls through to the database, and a system that was happily serving a 99% cache hit rate becomes a system serving a 0% hit rate to a database that was provisioned for the 1%. The database does not degrade. It falls over. Connections pile up, CPU pins at 100%, latency climbs from 15 ms to two seconds, clients time out and retry, the retries double the load, and within about ten seconds you have a full outage that nobody's dashboard predicted because every individual component looked healthy right up until it didn't.

That second failure mode is the **thundering herd**, also called a **cache stampede** or **dogpile**. Invalidation is the problem everyone studies; the stampede is the problem that actually shows up in postmortems. This post treats them as the two halves of one discipline. The invalidation half is about *correctness* — making sure a cache doesn't serve stale data forever. The stampede half is about *survival* — making sure the moment a cache entry goes away, you don't take the backing store down with it. You need both, and the techniques interact: a careless invalidation strategy *manufactures* stampedes, and a good stampede defense lets you use aggressive, simple invalidation that would otherwise be too dangerous.

![The cached read path: client request to cache lookup, splitting into a hit served from RAM or a miss that falls through to the database](/imgs/blogs/cache-invalidation-and-the-thundering-herd-1.webp)

The diagram above is the mental model for everything that follows: a cache lookup either hits — served from RAM in a fraction of a millisecond — or misses, and on a miss the request falls through to a database read that is fifty times slower and holds a connection while it runs. The entire art of caching is keeping the hit path wide and the miss path *narrow* — narrow enough that even a burst of simultaneous misses doesn't saturate the thing behind it. This article is a tour of that one picture: first the four ways to decide *when* a cache entry stops being valid, then the anatomy of what happens when too many entries stop being valid at once, and finally the six mitigations — with runnable Go and Python — that turn a cliff into a gentle slope.

## Why caching is harder than "store the result"

The naive mental model of a cache is a dictionary you check before doing real work: look in the map, and if the value isn't there, compute it and put it there. That model is correct and it is also a trap, because it silently assumes three things that are false in production: that misses are rare and uncorrelated, that the backing store can absorb the miss traffic, and that stale data is harmless. Every serious caching incident is one of those assumptions breaking.

| Assumption (naive view) | Reality in production |
| --- | --- |
| Misses are rare and independent | Misses *cluster* — a hot key's expiry triggers thousands of simultaneous misses in the same millisecond |
| The database can handle the miss load | It's provisioned for the *miss rate at steady state* (1%), not for 100% of read traffic at once |
| Stale data is fine, freshness is fine | Both have a cost; the right TTL is a business decision per key class, not a global constant |
| Invalidation means "delete the key" | A `DEL` on a hot key is exactly how you *cause* a stampede — you just synchronized every reader |
| Cache and DB are loosely coupled | Under stampede they are tightly coupled: the cache's failure mode *is* a DB overload |

The last row is the one that surprises people. We add a cache to *decouple* read load from the database, and most of the time it does. But the cache and the database are joined at the miss path, and the miss path has almost no capacity headroom by design — you sized the database for 1% of reads, so a momentary jump to even 10% of reads is a 10× spike. A cache is a leverage machine, and leverage cuts both ways: a 99% hit rate means the database sees 1% of load, but it also means that *losing* the cache for one hot key multiplies that key's DB load by 100×. The job is to make sure the leverage never reverses violently.

> A cache doesn't reduce your database's peak load. It reduces your database's *average* load while concentrating the variance into rare, correlated spikes. Your job is to spread those spikes back out.

## 1. Four ways to invalidate

Invalidation is the question "when does a cached value stop being allowed to serve?" There are exactly four mechanisms, and real systems mix them per key class. Picking wrong doesn't usually corrupt data — it usually either serves stale data for too long (a correctness bug that shows up as a support ticket) or hammers the write path (a performance bug that shows up as latency). The figure below is the decision space.

![Matrix of four invalidation strategies — TTL expiry, explicit on write, event-driven CDC, versioned keys — scored on staleness window, write cost, and best fit](/imgs/blogs/cache-invalidation-and-the-thundering-herd-2.webp)

### TTL expiry: the default, and the stampede generator

**Set a time-to-live and let the value evaporate.** This is the workhorse: `SET key value EX 300`. It's zero write-path cost (you set the TTL once, at write time, and never think about it again) and it bounds staleness to the TTL — after at most 300 seconds, the next reader misses and refreshes. It is correct for any read where "a few seconds stale" is acceptable, which is most reads: product listings, feed contents, prices that update every minute, leaderboard snapshots.

```python
import redis
r = redis.Redis()

def get_product(product_id: int) -> dict:
    key = f"product:{product_id}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)
    # miss: read through to the database
    row = db.query("SELECT * FROM products WHERE id = %s", product_id)
    r.set(key, json.dumps(row), ex=300)  # TTL = 5 minutes
    return row
```

TTL's fatal property is the one we'll spend half this post defending against: **a fixed TTL synchronizes expiry.** If a thousand keys were all written at the same time (say, by a cache-warming job, or because a deploy flushed the cache and the first wave of traffic refilled it in a tight window), then a fixed 300-second TTL means all thousand expire in the same 300-second-later instant. TTL is the simplest strategy *and* the one that builds the stampede; the rest of this post is largely about making TTL safe.

### Explicit invalidation on write: fresh, but coupled

**Delete or overwrite the cache key in the same code path that writes the database.** When a user updates their profile, you `UPDATE users SET ...` and then `DEL user:42` (or `SET user:42 <new value>`). Staleness drops to near zero — the next read after a write sees fresh data immediately — but you've coupled every write path to know which cache keys it invalidates. That coupling is a maintenance tax: every new feature that writes to `users` must remember to invalidate the right keys, and the *derived* keys (a `user:42:friends:count` that depends on a different table) are easy to forget. This is the source of the classic "I updated my name but the old one still shows up in one place" bug.

```python
def update_user_email(user_id: int, email: str) -> None:
    db.execute("UPDATE users SET email = %s WHERE id = %s", email, user_id)
    # invalidate everything derived from this user, in the same transaction-ish unit
    pipe = r.pipeline()
    pipe.delete(f"user:{user_id}")
    pipe.delete(f"user:{user_id}:profile")
    pipe.delete(f"user:{user_id}:settings")
    pipe.execute()
```

Two non-obvious traps. First, **`DEL`-on-write is itself a stampede generator** on hot keys: the moment you delete `user:celebrity`, every reader of that key misses simultaneously. For a genuinely hot key, prefer `SET` (overwrite with the new value) over `DEL` (force a refill) so there's no miss window at all. Second, the **delete-then-write vs write-then-delete ordering** matters under concurrency: if you delete the cache *before* committing the DB write, a concurrent reader can miss, read the *old* committed row, and repopulate the cache with stale data that now lives until the next write. The safe order is write the DB first, *then* invalidate — and even then there's a narrow race (a reader who started before your write but populates after your delete), which is why Facebook's memcache uses **leases** to close it; we'll come back to that.

### Event-driven invalidation (CDC / pub-sub): decoupled at the cost of a pipeline

**Tail the database's change stream and invalidate from there.** Instead of the write path knowing about cache keys, a change-data-capture (CDC) consumer reads the database's replication log (MySQL binlog, Postgres logical replication, DynamoDB Streams) and publishes invalidations to a bus that cache nodes subscribe to. Debezium → Kafka → a fleet of invalidation workers is the canonical shape. This decouples writers from cache topology — a writer just writes, and the invalidation happens downstream — which is the only sane option when *many* services write the same tables, or when one logical write fans out to dozens of cache keys across services.

The cost is real: you now operate a streaming pipeline, accept a few milliseconds of replication lag as your staleness floor, and must handle the pipeline being down (do you fail open and serve stale, or fail closed and stop caching?). It is the heaviest strategy and the right one at large scale precisely because the alternative — every writer everywhere knowing your cache keys — doesn't survive contact with a hundred-engineer org.

### Versioned keys: never invalidate, just stop referencing

**Encode a version in the key and write a new key on change; never invalidate the old one.** Instead of `config:pricing` with a `DEL` on update, you write `config:pricing:v7` and atomically flip a pointer (`config:pricing:current → v7`). Readers resolve the pointer, then read the versioned key. The old `config:pricing:v6` is now simply unreferenced and TTLs out on its own. There is *no invalidation event*, so there is no stampede from invalidation, and reads of the new version are warm the instant you flip the pointer (you can pre-populate `v7` before flipping). This is the cleanest strategy for hot, rarely-changing, globally-shared values — feature flags, pricing tables, ML model configs — where a stampede on the single shared key would be catastrophic.

```python
def publish_pricing(new_table: dict) -> None:
    version = int(time.time())
    key = f"config:pricing:v{version}"
    r.set(key, json.dumps(new_table), ex=86400)   # pre-populate, warm
    r.set("config:pricing:current", str(version))  # atomic pointer flip

def read_pricing() -> dict:
    version = r.get("config:pricing:current")
    return json.loads(r.get(f"config:pricing:v{version.decode()}"))
```

The trade is two reads per access (pointer, then value — often cached locally for a second) and the discipline of cleaning up old versions. For the right key class it's worth it, and it's how a lot of "config that must never stampede" gets shipped.

The four strategies are not mutually exclusive. A mature system uses TTL as the default safety net on everything (so nothing is cached *forever* even if an invalidation is missed), explicit invalidation on the write paths it controls, CDC for cross-service tables, and versioned keys for the handful of globally-hot configs. Now: the failure mode all four must survive.

## 2. Anatomy of a stampede

Here is the part that takes systems down. Walk through it once carefully, because the speed is the surprising part — a healthy system tips into a full outage in *seconds*, not minutes, and nothing in your standard dashboards warns you first.

![Anatomy of a cache stampede: a hot key expires, triggering a synchronized miss that drains the connection pool, pins DB CPU, climbs latency, times out clients, and storms retries into a full outage](/imgs/blogs/cache-invalidation-and-the-thundering-herd-3.webp)

A single **hot key expires** — say the cached homepage, or a celebrity's profile, or a trending product. The instant its TTL passes, the *next* request to read it misses. But this is a hot key, so it isn't the next request — it's the next *thousand* requests, all of which arrived in the few milliseconds it takes to do one database read, and all of which independently observe a miss and independently decide to fall through to the database. That's the **synchronized miss**: 10,000 requests per second that were being served from RAM are now, all at once, issuing database queries.

The database has a finite connection pool — call it 200 slots. The first 200 misses grab all 200 connections; the 201st through 10,000th queue up waiting for a connection. Now the database is doing 200 identical reads of the same row in parallel (every one of those misses wants the *same* key), CPU spikes toward 100% under the redundant load, and each query that used to take 15 ms now takes longer because the box is saturated. **Latency climbs** — p99 goes from 15 ms to 500 ms to 2 seconds. Clients have a 1-second timeout, so they start **timing out** and, because they're well-behaved clients with retry logic, they **retry** — which doubles the offered load. The retried requests also miss (the cache still isn't filled, because the original queries haven't returned yet), so they pile onto the same queue. Load feeds on itself. Within seconds the database is **unavailable**, and now *every* cached key behind it is uncacheable because there's nothing to read through to. One key's expiry has become a total outage.

The cruelty is in the math. At steady state the database sees 1% of 10,000 req/s = 100 req/s, and it's sized with comfortable headroom for that. The stampede presents it with up to 10,000 req/s — a 100× spike — of *redundant* work, since every request is reading the identical row that a single query could have fetched. The database burns its entire capacity computing the same answer thousands of times. That redundancy is the key insight behind the best mitigation: if 10,000 requests all want the same freshly-expired key, they need *one* database read between them, not 10,000.

> A stampede is not a capacity problem you can buy your way out of. Doubling the database doubles the steady-state headroom but the spike is 100×, not 2×. You cannot provision for a 100× redundant burst; you have to make the burst not happen.

### The animation: synchronized vs smoothed

The whole game is visible in one picture if you let it move. Watch the same TTL window play out two ways:

<figure class="blog-anim">
<svg viewBox="0 0 720 340" role="img" aria-label="Database load over time: with synchronized TTL expiry, load spikes to a tall single bar at the expiry instant; with jitter and request coalescing, the same demand becomes a low flat band of bars" style="width:100%;height:auto;max-width:860px">
<title>Synchronized expiry spikes the database; jitter and coalescing flatten the same load</title>
<style>
.h1-axis{stroke:var(--border,#d1d5db);stroke-width:2}
.h1-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.h1-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.h1-cap{fill:var(--accent,#e8590c)}
.h1-bar{fill:var(--accent,#e8590c)}
.h1-flat{fill:#2f9e44}
.h1-base{fill:var(--border,#d1d5db);opacity:.5}
@keyframes h1-spike{0%,8%{transform:scaleY(.06)}22%,42%{transform:scaleY(1)}58%,100%{transform:scaleY(.06)}}
@keyframes h1-spikeFade{0%,8%{opacity:.15}22%,42%{opacity:1}58%,100%{opacity:.15}}
@keyframes h1-smooth{0%,42%{opacity:.12}58%,92%{opacity:1}100%{opacity:.12}}
@keyframes h1-grow{0%,42%{transform:scaleY(.06)}58%,92%{transform:scaleY(1)}100%{transform:scaleY(.06)}}
.h1-spikeGrp rect{transform-box:fill-box;transform-origin:bottom;animation:h1-spike 9s ease-in-out infinite}
.h1-spikeGrp{animation:h1-spikeFade 9s ease-in-out infinite}
.h1-smoothGrp{animation:h1-smooth 9s ease-in-out infinite}
.h1-smoothGrp rect{transform-box:fill-box;transform-origin:bottom;animation:h1-grow 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.h1-spikeGrp,.h1-spikeGrp rect,.h1-smoothGrp,.h1-smoothGrp rect{animation:none;opacity:1;transform:none}}
</style>
<line class="h1-axis" x1="60" y1="250" x2="660" y2="250"/>
<line class="h1-axis" x1="60" y1="60" x2="60" y2="250"/>
<text class="h1-sub" x="34" y="70" transform="rotate(-90 34 70)">DB load</text>
<text class="h1-sub" x="360" y="278">time, one TTL window</text>
<rect class="h1-base" x="90"  y="244" width="34" height="6"/>
<rect class="h1-base" x="134" y="244" width="34" height="6"/>
<rect class="h1-base" x="178" y="244" width="34" height="6"/>
<rect class="h1-base" x="266" y="244" width="34" height="6"/>
<rect class="h1-base" x="310" y="244" width="34" height="6"/>
<rect class="h1-base" x="354" y="244" width="34" height="6"/>
<g class="h1-spikeGrp">
<rect class="h1-bar" x="218" y="84" width="34" height="166"/>
</g>
<text class="h1-cap h1-lbl" x="235" y="305">synchronized expiry: one tall spike, DB tips over</text>
<g class="h1-smoothGrp">
<rect class="h1-flat" x="430" y="214" width="26" height="36"/>
<rect class="h1-flat" x="466" y="208" width="26" height="42"/>
<rect class="h1-flat" x="502" y="216" width="26" height="34"/>
<rect class="h1-flat" x="538" y="210" width="26" height="40"/>
<rect class="h1-flat" x="574" y="214" width="26" height="36"/>
<rect class="h1-flat" x="610" y="212" width="26" height="38"/>
<text class="h1-sub" x="533" y="305" fill="#2f9e44" style="font-weight:600">jitter + coalescing: same demand, flat band</text>
</g>
</svg>
<figcaption>The same TTL window, two strategies running the same round: synchronized expiry collapses all the misses into one tall spike that tips the database over, while TTL jitter plus request coalescing spreads the identical demand into a low, flat band the database absorbs easily.</figcaption>
</figure>

Left, the synchronized world: load is flat (everything's a hit) right up until the expiry instant, when it spikes vertically to a value the database can't sustain. Right, the smoothed world: the *same total amount of refresh work* is spread into a low, flat band — a few queries per second instead of thousands in one millisecond. Every mitigation below is a different way of turning the left picture into the right one. None of them reduces the *total* refresh work; they all reduce its *peak*.

## 3. Mitigation one: TTL jitter

**Add a small random offset to every TTL so a cohort of keys never expires in the same instant.** This is the cheapest mitigation, the first one you should reach for, and the one most people forget until after their first stampede. The cause of synchronized expiry is that keys written together with the same fixed TTL expire together; the fix is to make the TTL slightly different for each key.

![Before-and-after of TTL jitter: a fixed 300-second TTL makes 10k keys expire in one synchronized cliff, while a jittered TTL spreads expiry across a 60-second window for a smooth refill](/imgs/blogs/cache-invalidation-and-the-thundering-herd-5.webp)

Instead of every key getting `EX 300`, each key gets `EX 300 + random(0, 60)`. Now a cohort of 10,000 keys that were all written at `t=0` will expire spread across the window `[300, 360]` — roughly 170 keys per second instead of 10,000 in one instant. The database sees a gentle, sustained refill it's easily sized for instead of a cliff.

```python
import random

def cache_with_jitter(key: str, value, base_ttl: int = 300, jitter: float = 0.2) -> None:
    # jitter = 0.2 means "spread expiry across +/- 20% of the base TTL"
    spread = int(base_ttl * jitter)
    ttl = base_ttl + random.randint(-spread, spread)
    r.set(key, value, ex=ttl)
```

Two refinements worth knowing. First, jitter only helps when the *cause* of synchronization is a shared write time — it does nothing for a single key that's hot enough to stampede on its own (that one key still expires at one instant; jitter just moves the instant). For single-hot-key stampedes you need coalescing or early expiration, below. Jitter solves the *cohort* problem (mass expiry of many keys), which is the more common one and the one that bites after a deploy or a warm-up job. Second, **±20% jitter on a 300-second TTL means up to 60 seconds of extra staleness on some keys** — usually fine, but a deliberate trade you should make consciously, not a free lunch. If you need tight freshness, use a smaller jitter percentage and lean harder on coalescing.

A subtle but important variant: jitter the *base* TTL, not just the expiry, when you re-set a key on refresh. If every refresh resets to exactly `300`, then keys that all refresh together (because they all expired together once) stay synchronized forever. Jittering on every `SET` continuously de-correlates the population.

## 4. Mitigation two: request coalescing (single-flight)

**When N requests miss the same key concurrently, let exactly one of them do the database read and have the other N−1 wait for its result.** This is the single most important stampede mitigation because it attacks the redundancy directly: the stampede is thousands of requests doing the *same* read, and coalescing collapses them to one. It's variously called single-flight, request coalescing, or dogpile prevention.

![Single-flight coalescing: three concurrent requests missing the same key converge on a keyed gate that elects one leader to run the single database query while followers block on a shared future, then all three are served from one result](/imgs/blogs/cache-invalidation-and-the-thundering-herd-6.webp)

The mechanism: a per-key lock (or, more elegantly, a per-key shared future/promise). The first request to miss acquires the key's slot and becomes the *leader* — it does the one database read and fills the cache. Every other request that misses the same key while the leader is in flight finds the slot occupied and *waits* on the leader's result instead of issuing its own query. When the leader finishes, all the waiters wake up with the same value. One database read serves the entire herd.

Go's standard ecosystem ships this as `golang.org/x/sync/singleflight`, and it's about as clean as it gets:

```go
package cache

import (
	"context"
	"encoding/json"

	"github.com/redis/go-redis/v9"
	"golang.org/x/sync/singleflight"
)

type Cache struct {
	rdb   *redis.Client
	group singleflight.Group // dedupes concurrent calls by key
	db    *DB
}

func (c *Cache) GetProduct(ctx context.Context, id string) (*Product, error) {
	key := "product:" + id

	// Fast path: a normal cache hit, no coalescing needed.
	if b, err := c.rdb.Get(ctx, key).Bytes(); err == nil {
		var p Product
		return &p, json.Unmarshal(b, &p)
	}

	// Miss. singleflight.Do guarantees that for a given key, only ONE
	// invocation of the fn runs at a time; concurrent callers for the same
	// key block and receive the leader's result. `shared` is true for the
	// followers, which is handy for metrics.
	v, err, shared := c.group.Do(key, func() (interface{}, error) {
		p, err := c.db.LoadProduct(ctx, id) // the single DB read
		if err != nil {
			return nil, err
		}
		b, _ := json.Marshal(p)
		// jittered TTL so the refilled key doesn't re-synchronize
		c.rdb.Set(ctx, key, b, ttlWithJitter(300))
		return p, nil
	})
	_ = shared
	if err != nil {
		return nil, err
	}
	return v.(*Product), nil
}
```

The Python equivalent uses an `asyncio` lock per key (or `threading.Lock` for sync code). The trick is a registry of locks keyed by cache key, so concurrent misses on the *same* key serialize while misses on *different* keys proceed in parallel:

```python
import asyncio
import json

class SingleFlightCache:
    def __init__(self, rdb, db):
        self.rdb = rdb
        self.db = db
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._registry_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def get_product(self, product_id: int) -> dict:
        key = f"product:{product_id}"
        # Fast path: hit.
        cached = await self.rdb.get(key)
        if cached is not None:
            return json.loads(cached)

        # Miss: only one coroutine per key does the DB read.
        lock = await self._lock_for(key)
        async with lock:
            # Double-check: the leader may have already filled it while we
            # waited for the lock. This is the whole point — followers see
            # the fresh value and never touch the DB.
            cached = await self.rdb.get(key)
            if cached is not None:
                return json.loads(cached)
            row = await self.db.load_product(product_id)  # the single DB read
            await self.rdb.set(key, json.dumps(row), ex=ttl_with_jitter(300))
            return row
```

The **double-check after acquiring the lock** is load-bearing and the most common bug: without it, every waiter wakes up, doesn't re-check the cache, and issues its own redundant query anyway — defeating the entire purpose. Always re-read the cache inside the critical section.

Two scaling caveats. First, an in-process lock (Go's `singleflight`, Python's per-key `Lock`) coalesces only within *one* process. If you run 50 application servers, you get *50* leader queries for one hot key, not one — a 50× reduction, not 10,000×, which is usually plenty but not perfect. For true cross-process coalescing you need a **distributed lock** (a Redis `SET key token NX EX 5` lease that the leader holds while it fills), which is exactly what Facebook's memcache **lease** mechanism implements at scale — we'll come back to that in the dedicated [memcache deep-dive](/blog/software-development/database-scaling/scaling-memcache-at-facebook). Second, coalescing **serializes latency**: every follower waits as long as the leader's full database read takes. That's fine (they'd have waited anyway), but make sure the leader has a timeout so a slow leader doesn't hang the whole herd indefinitely.

## 5. Mitigation three: probabilistic early expiration (XFetch)

**Refresh a hot key *before* it expires, with a probability that rises as expiry approaches, so one lucky request rebuilds the cache while everyone else is still being served the live value.** This is the elegant one — it has essentially no thundering edge at all, because there's never a hard miss on a hot key. It comes from the 2015 paper "Optimal Probabilistic Cache Stampede Prevention" by Vattani, Chierichetti, and Lowenstein, and the algorithm is usually called **XFetch**.

![Timeline of XFetch probabilistic early expiration: the recompute probability is negligible mid-life, rises as the key nears its TTL, a request draws an early recompute before expiry, and the cache is refilled so the TTL passes with no hard miss](/imgs/blogs/cache-invalidation-and-the-thundering-herd-7.webp)

The intuition: instead of waiting for the TTL to hit zero and then having everyone miss at once, each read makes a small gamble. Early in the key's life the gamble almost always says "serve the cached value." As the key ages toward expiry, the gamble increasingly says "you — yes, you, this one request — go recompute it now, in the background, while still serving the current value." Because the probability rises smoothly, *some* request recomputes the key shortly before it would have expired, the cache is refreshed, and the hard expiry never happens. The herd never forms because there's never a synchronized miss to form around.

The decision rule, when you store the value alongside the time `delta` it took to compute it and the absolute `expiry`, is:

$$ \text{now} - \delta \cdot \beta \cdot \ln(\text{rand}()) \;\geq\; \text{expiry} $$

where $\delta$ is the recompute cost (how long the DB read took), $\beta \geq 1$ is a tuning knob (higher = recompute earlier and more eagerly), and $\text{rand}()$ is uniform in $(0, 1]$ so $\ln(\text{rand}())$ is a negative number whose magnitude is occasionally large. The term $-\delta \cdot \beta \cdot \ln(\text{rand}())$ is a small positive "look-ahead" that grows with the recompute cost — expensive keys get refreshed earlier, because they're the ones whose miss would hurt most.

```python
import math
import random
import time

def xfetch_get(key: str, ttl: int = 300, beta: float = 1.0):
    raw = r.get(key)
    if raw is not None:
        value, delta, expiry = unpack(raw)  # stored: value + compute-cost + abs expiry
        # Should THIS request recompute early? Probability rises as now -> expiry.
        if time.time() - delta * beta * math.log(random.random()) < expiry:
            return value  # common case: serve the cached value, no recompute
        # else: fall through and recompute, but we still have a valid value to
        # serve if we wanted to (XFetch is typically paired with a background
        # refresh so the requesting client isn't the one that waits).
    start = time.time()
    value = db.load(key)
    delta = time.time() - start          # how long the recompute took
    expiry = time.time() + ttl
    r.set(key, pack(value, delta, expiry), ex=ttl + 60)  # physical TTL > logical
    return value
```

The physical Redis TTL is set a bit *longer* than the logical `expiry` so the value is still retrievable during the brief recompute — XFetch only works if you keep serving the old value while the chosen request refreshes. In practice you pair XFetch with a background refresh: the request that "wins" the gamble kicks off an async recompute and *still* returns the current cached value, so no user request ever eats the recompute latency. That combination — probabilistic *who* refreshes, background *how* — gives you a hot key that effectively never misses and never stampedes, at the cost of slightly more recomputes than strictly necessary (you're refreshing a little early). Tune $\beta$ down toward 1 to minimize wasted recomputes, up toward 2–3 to be more conservative about ever hard-missing.

## 6. Mitigation four: stale-while-revalidate

**On a miss where you still hold a stale copy, serve the stale copy immediately and refresh in the background.** This is the HTTP-cache pattern (`Cache-Control: stale-while-revalidate`) applied to your application cache, and it's the pragmatic cousin of XFetch: simpler to reason about, slightly more stale, and it keeps client-facing latency flat through every refresh because no client request ever blocks on the database.

![Stale-while-revalidate request flow: a GET of a stale key serves the stale value in 0.3 ms with no wait, spawns one async refresh job that reads the database off the hot path, sets the fresh value, and the next reader gets fresh data still fast](/imgs/blogs/cache-invalidation-and-the-thundering-herd-8.webp)

The model uses two thresholds per key: a **soft TTL** after which the value is considered stale-but-servable, and a **hard TTL** (longer) after which it must not be served at all. A read that finds the value within the soft TTL serves it normally. A read in the stale window (between soft and hard) serves the stale value *and* triggers exactly one background refresh — using the same single-flight gate from mitigation two so only one refresh fires no matter how many readers hit the stale window. A read past the hard TTL is a real miss and falls through (this is the last-resort path you want to almost never hit).

```python
import asyncio
import json
import time

async def swr_get(self, key: str, soft_ttl=60, hard_ttl=600) -> dict:
    raw = await self.rdb.get(key)
    if raw is not None:
        entry = json.loads(raw)
        age = time.time() - entry["fetched_at"]
        if age < soft_ttl:
            return entry["value"]               # fresh: serve directly
        if age < hard_ttl:
            # stale-but-servable: serve NOW, refresh in the background.
            # The single-flight gate ensures only ONE refresh fires.
            asyncio.create_task(self._refresh(key, soft_ttl, hard_ttl))
            return entry["value"]               # client never waits on the DB
    # past hard TTL (or never cached): the only blocking path. Coalesce it too.
    return await self._refresh(key, soft_ttl, hard_ttl, blocking=True)

async def _refresh(self, key, soft_ttl, hard_ttl, blocking=False):
    lock = await self._lock_for(key)
    if not blocking and lock.locked():
        return                                  # someone is already refreshing
    async with lock:
        row = await self.db.load(key)
        entry = {"value": row, "fetched_at": time.time()}
        await self.rdb.set(key, json.dumps(entry), ex=hard_ttl)
        return row
```

The win is that **client latency stays flat through a refresh**: the refresh happens off the request path, so even the request that triggers it returns the stale value in 0.3 ms rather than waiting 15 ms (or 2 seconds under load) for the database. The cost is staleness up to the soft-to-hard window, and the operational subtlety that a *failed* background refresh should not promote a key past its hard TTL — if the DB is down, you want to keep serving stale (within reason) rather than fall into the hard-miss path and create the very stampede you were avoiding. Serving slightly stale data during a database outage is almost always better than serving errors.

## 7. Mitigation five: leases and locks

We've used in-process single-flight (mitigation two); the distributed generalization is a **lease**. When a key misses, a reader attempts to acquire a short-lived lease on it (`SET key:lease <token> NX EX 5`). Exactly one reader wins the `NX` (set-if-not-exists) and becomes the cluster-wide leader, doing the one database read across *all* application servers. Losers either wait-and-retry the cache for a few hundred milliseconds (hoping the leader fills it) or serve stale if they have it.

```python
async def get_with_lease(self, key: str, ttl=300) -> dict:
    cached = await self.rdb.get(key)
    if cached is not None:
        return json.loads(cached)
    token = secrets.token_hex(8)
    got_lease = await self.rdb.set(f"lease:{key}", token, nx=True, ex=5)
    if got_lease:
        try:
            row = await self.db.load(key)                       # the one cluster-wide read
            await self.rdb.set(key, json.dumps(row), ex=ttl_with_jitter(ttl))
            return row
        finally:
            # release only if we still hold it (token check avoids releasing
            # a lease that already expired and was re-acquired by someone else)
            await release_if_owner(self.rdb, f"lease:{key}", token)
    # didn't get the lease: poll briefly for the leader's fill
    for _ in range(10):
        await asyncio.sleep(0.05)
        cached = await self.rdb.get(key)
        if cached is not None:
            return json.loads(cached)
    return await self.db.load(key)   # leader stalled; last-resort direct read

```

This is precisely the shape of Facebook's memcache lease mechanism, which additionally returns the lease *token* to the client so the cache can reject a `SET` from a client whose lease has been invalidated (closing the stale-set race from §1). The full design — including how leases interact with invalidation and the "stale value with a lease" optimization — is worth its own treatment; see [scaling memcache at Facebook](/blog/software-development/database-scaling/scaling-memcache-at-facebook). The takeaway here is that a lease is just single-flight that works across processes, paid for with an extra Redis round-trip and the operational care that distributed locks always demand (token-checked release, a TTL on the lease so a crashed leader can't wedge the key forever).

## 8. Mitigation six: negative caching

**Cache the *absence* of a value so that repeated requests for a missing key don't repeatedly hit the database.** A surprising fraction of stampedes are stampedes on keys that don't exist: a deleted user whose ID is still in a thousand bookmarks, a product that was discontinued, a probing scanner requesting random IDs. Each such request misses the cache (correctly — there's nothing to cache), falls through to the database, gets an empty result, and *doesn't cache anything* (because the naive code only caches non-empty results). So every one of those requests hits the database forever. A bot hammering `/api/user/999999999` with a different nonexistent ID each time can keep your database busy with pointless lookups indefinitely.

The fix is to cache the miss itself, with a short TTL:

```python
NEGATIVE = object()  # sentinel distinct from "not in cache" and from a real value

async def get_user(self, user_id: int):
    key = f"user:{user_id}"
    cached = await self.rdb.get(key)
    if cached == b"__MISSING__":
        return None                    # cached negative: don't touch the DB
    if cached is not None:
        return json.loads(cached)
    row = await self.db.load_user(user_id)
    if row is None:
        # cache the ABSENCE, with a SHORT ttl so a later insert isn't masked
        await self.rdb.set(key, b"__MISSING__", ex=30)
        return None
    await self.rdb.set(key, json.dumps(row), ex=ttl_with_jitter(300))
    return row
```

Use a **short negative TTL** (tens of seconds, not minutes) so that if the key later starts existing — the user signs up, the product is restocked — you don't serve "not found" for five minutes. Negative caching is cheap and underused; it's also a defense against a specific abuse vector (random-ID scanning) that bypasses your entire cache by construction. Pair it with a Bloom filter for the extreme case where the keyspace of possible-but-nonexistent IDs is huge: check the Bloom filter before the cache, and if the filter says "definitely not present," return `None` without even a cache lookup.

### The mitigations on one page

| Mitigation | What it solves | Cost / complexity | Reach for it when |
| --- | --- | --- | --- |
| **TTL jitter** | Cohort mass-expiry (many keys at once) | Trivial — one line, slight extra staleness | Always, on every TTL, as a baseline |
| **Single-flight** (in-process) | Redundant concurrent misses on one key | Low — a per-key lock + double-check | Any hot key; the default per-process defense |
| **Lease / distributed lock** | Same, but across all app servers | Medium — extra Redis round-trip, token release | Cross-process coalescing of a globally-hot key |
| **XFetch (early expiry)** | Hard miss on a single hot key ever happening | Medium — store compute-cost + tuning $\beta$ | Expensive-to-compute, very hot keys |
| **Stale-while-revalidate** | Client latency spiking during refresh | Medium — soft/hard TTLs + background task | Read-heavy keys where slight staleness is fine |
| **Negative caching** | Repeated misses on nonexistent keys | Low — a sentinel + short TTL | Sparse keyspaces, scanner traffic, deleted entities |

These compose. A production read path for a hot, expensive, read-mostly key typically layers *all* of them: jittered TTLs as the baseline, single-flight (or a lease at scale) to collapse concurrent misses, XFetch or stale-while-revalidate so the hot key never hard-misses, and negative caching so nonexistent lookups don't leak through. The cost is a few dozen lines of well-tested cache-access code that every read goes through — which is exactly why you build it once, centrally, and never let individual features hand-roll `get-or-load`.

## 9. The cold-cache problem

Every mitigation above assumes the cache is *warm* — that there's something to coalesce around, something to serve stale, something whose expiry you can preempt. The nastiest stampede is the one where the cache is *empty*: right after a deploy that flushed it, a Redis failover that lost the dataset, a node restart, or a scale-up that adds a cold cache node to the pool. A cold cache has a 0% hit rate by definition, so *every* read misses, and the entire read traffic of your service hits the database at once. It's a stampede on every key simultaneously.

![Before-and-after of the cold-cache problem: a node cold-flipped to 100% traffic with a 0% hit rate sends 100x load to the database, versus a warm-peer read-through plus a gradual traffic ramp that keeps DB load bounded as the hit rate climbs](/imgs/blogs/cache-invalidation-and-the-thundering-herd-9.webp)

The deploy-flush version is the most self-inflicted: a deploy that clears the cache "to be safe" (or a config that changes the cache key format, orphaning every old key) means the first wave of post-deploy traffic finds an empty cache and falls through *en masse*. If the database isn't sized for 100% of read traffic — and it isn't, that's the whole point of the cache — the deploy itself takes the database down. Teams discover this the first time they deploy during peak traffic.

There are three defenses, used together:

1. **Don't flush on deploy.** Make cache keys version-tolerant (or versioned, per §1) so a deploy doesn't orphan the warm dataset. A flush should be a deliberate, rare, rate-limited operation, never a side effect of shipping code.
2. **Ramp traffic gradually.** When a new (cold) cache node or app instance joins the pool, send it 1% of traffic and ramp to 100% over several minutes. Its cache warms as it serves, so by the time it's at full traffic the hit rate has climbed and the database never sees the full cold-miss load. This is just a load-balancer weight schedule, and it's the single highest-leverage cold-cache defense.
3. **Read-through from a warm peer.** On a miss, a cold node can ask a *warm* sibling cache node before falling through to the database — turning "every node independently cold-misses to the DB" into "cold nodes warm from hot nodes, DB sees one fill." This is mirror-pool warming; it's how you bring a failed-over Redis replica back without a stampede.

The deepest version of this discipline overlaps with capacity planning: you should *know* what fraction of your read traffic the database can actually absorb, because that number is the ceiling on how cold your cache is allowed to get before you have an outage. If the answer is "10% of reads," then your traffic ramp must keep the cold-miss rate under 10% at every step — which sets the ramp duration. That's a [capacity-planning](/blog/software-development/database-scaling/capacity-planning-for-databases) calculation, not a guess, and it's the difference between a deploy that's invisible and a deploy that's an incident.

> The cache is not a performance optimization you can remove "temporarily." Once your database is sized assuming the cache absorbs 99% of reads, the cache is load-bearing infrastructure, and an empty cache is an outage waiting for traffic. Treat cache warmth as a hard dependency of staying up.

## Case studies from production

### 1. The celebrity profile that took down the feed

A large social app cached user profiles with a flat 5-minute TTL and no coalescing. A celebrity posted, traffic to their profile spiked to roughly 50,000 reads per second, and five minutes after the cache was first populated, the key expired. Every one of those 50,000 requests missed in the same instant, all fell through to the user-database shard that held the celebrity's row, and that single shard — sized for a few hundred reads per second — saturated immediately. The shard's connection pool drained, queries queued, the shard's other users (who happened to share it) saw their reads time out too, and the timeout-retry loop doubled the load within two seconds. The first hypothesis was a database problem (the shard was clearly unhealthy), and engineers spent twenty minutes looking at the database before someone noticed the perfectly periodic, exactly-five-minutes-apart spikes in the miss-rate graph. The root cause was the fixed TTL with no single-flight. The fix was two lines: jittered TTL plus `singleflight`. After deploy, the same celebrity traffic produced one database read every five minutes per shard instead of 50,000 in one instant. The lesson: the failure was invisible at the component level because every component was healthy *between* spikes; only the *correlation* of the misses was pathological.

### 2. The deploy that flushed the cache at peak

An e-commerce team had a deploy script that ran `FLUSHALL` on their Redis cluster "to avoid serving stale data after a schema change." This was harmless at 3 a.m. when they normally deployed. Then a hotfix needed to ship at noon, on a sale day, at peak traffic. The flush emptied a cache that was absorbing about 95% of read load, the next second of traffic found a 0% hit rate, and 100% of reads — roughly 20× what the database was provisioned for — hit the database at once. The database fell over in under five seconds; the site was down for eleven minutes while the cache slowly refilled under a database that was itself struggling. The wrong first hypothesis was "the new code has a bad query," and they rolled back the application — which didn't help, because the rollback *also* ran the flush. The actual fix was to delete the `FLUSHALL` from the deploy entirely (cache keys were already version-tolerant; the flush was cargo-cult caution) and add a traffic ramp for cold nodes. The lesson: a cache flush is a deliberate, dangerous operation that must be rate-limited or eliminated, never a reflexive deploy step.

### 3. The thundering herd on a key that didn't exist

A payments service exposed `GET /accounts/{id}/balance`, cached by account ID. A misconfigured upstream batch job started requesting balances for account IDs that had been *closed and deleted*. Each request missed the cache (correctly — the account was gone), fell through to the database, got an empty result, cached nothing, and so every one of the millions of batch requests hit the database directly. The database's read replica saturated on lookups that all returned zero rows. The on-call engineer's first hypothesis was a query-plan regression (the balance query had gotten slow), but the query was fine — there were just millions of them, all for nonexistent accounts. The fix was negative caching: cache a `__MISSING__` sentinel for nonexistent account IDs with a 60-second TTL, which collapsed the millions of redundant lookups into one per account per minute. The lesson: a cache that only caches *successes* has an unbounded miss path for failures, and failures can be a stampede too.

### 4. The single-flight that only worked on one box

A team correctly added Go `singleflight` to their hottest endpoint after a stampede, tested it under load on a single instance (where it perfectly collapsed concurrent misses to one query), and shipped it. The next stampede was smaller but still took the database to 80% CPU. The puzzle: single-flight was *working* — each instance was doing exactly one query per hot-key expiry. The catch was that they ran 120 instances, so a hot-key expiry produced 120 simultaneous "leader" queries, one per box, not one globally. In-process coalescing reduced the herd by 120× (from ~12,000 redundant queries to 120), which was enough to survive most expiries but not the hottest key at peak. The fix was to add a Redis lease on top of the in-process single-flight: in-process coalescing collapses each box's herd to one, and the lease collapses the 120 box-leaders to one cluster-wide leader. The lesson: in-process single-flight is necessary but not sufficient at fleet scale; know your instance count, multiply, and decide whether 120 leader queries is acceptable before you find out in production.

### 5. The XFetch tuning that recomputed too eagerly

A search backend adopted XFetch to keep an expensive aggregation (a 400 ms database query) from ever hard-missing. It worked — the key never stampeded — but the database's steady-state load went *up* noticeably after the rollout. The cause was $\beta$ set to 3.0 (chosen "to be safe"), which made the early-expiry probability rise so aggressively that the key was being recomputed roughly every 90 seconds despite a 300-second TTL — three times more often than necessary. For a key that costs 400 ms to compute, that's a lot of wasted database work multiplied across thousands of keys. Tuning $\beta$ down to 1.0 cut the recompute rate by about 2.5× while still preventing every hard miss (the probabilistic guarantee holds for any $\beta \geq 1$; higher $\beta$ just trades extra recomputes for a wider safety margin). The lesson: XFetch's $\beta$ is a real tuning knob with a real cost — it doesn't prevent stampedes "more" above 1.0, it just refreshes earlier, which is wasted work unless you've measured that you need the margin.

### 6. The stale-while-revalidate that hid a dead database

A content site used stale-while-revalidate with a 1-minute soft TTL and a 1-hour hard TTL. The database had a multi-hour outage during a maintenance window gone wrong. For the first hour, the site stayed *up* — every read served stale content and the background refreshes silently failed, which was exactly the intended graceful degradation, and nobody noticed the database was down because the user experience was fine. The problem came at the hard-TTL boundary: an hour in, keys began crossing the hard TTL faster than the (still-dead) database could refresh them, every one of those became a real blocking miss, and the site started erroring in a slow-spreading wave as different keys aged out. The wrong hypothesis was "the cache is failing," but the cache had been heroically covering for a database that had been dead for an hour. The fix had two parts: alert on background-refresh *failure rate* (which had been silently at 100% for an hour), and extend the hard TTL dynamically when the database is known-unhealthy (serve stale longer rather than fall into the blocking path during a known outage). The lesson: stale-while-revalidate is so good at hiding backend failures that it can hide them from *you* — instrument the refresh path, not just the serve path.

### 7. The jitter that wasn't enough

A gaming company added ±10% jitter to a 60-second TTL on session data after a stampede, reasoning that spreading expiry across a 12-second window would smooth the load. It helped, but stampedes still happened at lower amplitude. The issue was that the *hottest* session keys were hot enough to stampede *individually* — a single popular streamer's session key, expiring at one instant, still produced thousands of simultaneous misses, and jitter on a single key just moves the one instant it expires; it does nothing to spread misses *within* that key. Jitter solved the cohort problem (many keys expiring together) but not the single-hot-key problem. The fix was to add single-flight on top of the jitter: jitter spreads the *cohort*, single-flight collapses the *per-key* herd, and you need both. The lesson: jitter and coalescing solve different halves of the stampede — jitter de-correlates *across* keys, coalescing de-duplicates *within* a key — and a system with hot individual keys needs both.

### 8. The pre-warm that became the stampede

A team, having learned about cold caches, added a cache-warming job that ran after every deploy: iterate over the top 100,000 keys and populate them. It eliminated the cold-cache stampede on deploy. But it *created* a new one 300 seconds later: because the warming job populated all 100,000 keys within a tight 30-second window with a fixed 300-second TTL, all 100,000 expired in a 30-second window five minutes later — a textbook synchronized cohort, manufactured by the very job meant to prevent stampedes. The dashboards showed a clean deploy followed by a mysterious database spike exactly five minutes later, every single deploy. The fix was to jitter the TTLs *in the warming job* (the warming code was setting `EX 300` without jitter even though the main read path had jitter). The lesson: warming jobs are cohort generators by construction — they write a huge population in a tight window — so they need jitter *more* than the normal write path, not less.

## When to reach for these, and when not to

### Reach for stampede defenses when:

- A single key is hot enough that all of its readers missing at once would overwhelm the backing store — celebrity profiles, trending items, the homepage, a shared config. This is the single-flight / lease / XFetch case.
- You cache a large *cohort* of keys written in a tight window (a warming job, a post-deploy refill, a batch import). This is the jitter case, and it's the most commonly forgotten one.
- The cached computation is expensive (a heavy aggregation, a cross-service fan-out, an ML inference) so even a *few* redundant recomputes hurt. XFetch and coalescing pay for themselves fastest here.
- Your database is provisioned assuming the cache absorbs most reads — which is essentially always, because that's the point of the cache. The moment the database can't survive 100% of read traffic, cache warmth is a hard dependency and you need cold-cache defenses.
- You serve traffic that probes nonexistent keys (public APIs, scanners, deleted entities still referenced elsewhere). Negative caching is the cheap, specific defense.

### Skip them (or keep it simple) when:

- The keyspace is uniformly cold — every key is read roughly once and never again. There's no hot key to stampede on and nothing to coalesce; a plain TTL cache (or no cache) is correct, and adding single-flight is pure overhead.
- The backing store can comfortably serve 100% of read traffic. If your "cache" is in front of a key-value store that's as fast as the cache, the stampede has no teeth — though you should question whether you need the cache at all.
- Staleness is unacceptable and the data changes constantly. Stale-while-revalidate and long TTLs are wrong here; you want explicit or event-driven invalidation with tight TTLs, and you accept the higher miss rate as the price of freshness.
- You're early and small. A single-process app with one database and modest traffic should ship a plain TTL cache with jitter and a single-flight wrapper and stop there — that handles the overwhelming majority of stampedes for two lines of jitter and a dozen of coalescing. XFetch, leases, and CDC invalidation are scale problems; don't pre-build them.

The throughline of every case study above is the same: a cache makes a system fast by concentrating its load variance into rare, correlated spikes, and an engineering team's job is to spread those spikes back out before they land all at once. Invalidation decides *when* a value stops serving; the stampede defenses decide *what happens to the database the instant it does*. Get the first one wrong and you serve stale data; get the second one wrong and you don't have a database to serve from at all. Both are the same discipline, and now you have the whole toolbox.

## Further reading

- [Cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production) — cache-aside vs read-through vs write-through, and where each one's invalidation lives.
- [Scaling memcache at Facebook](/blog/software-development/database-scaling/scaling-memcache-at-facebook) — the lease mechanism, the stale-set race, and stampede prevention at the scale that named the problem.
- [Redis applications and optimization](/blog/software-development/database/redis-applications-and-optimization) — the data structures and commands (`SET NX EX`, pipelines, Lua) you build these mitigations out of.
- [Capacity planning for databases](/blog/software-development/database-scaling/capacity-planning-for-databases) — how to compute the fraction of read traffic your database can actually absorb, which is the ceiling on how cold your cache may get.
- Vattani, Chierichetti, Lowenstein, "Optimal Probabilistic Cache Stampede Prevention" (VLDB 2015) — the XFetch paper.

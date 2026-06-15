---
title: "Caching Strategies Across Services: The Easy Win and the Hard Problem Hiding Inside It"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Caching is the highest-leverage performance lever in microservices, and cross-service invalidation is one of the genuinely hard problems — this post walks the cache patterns, the layers, an event-driven invalidation flow with a TTL backstop, a cache-stampede fix, and the worked numbers that take a product page from 60% to 95% hit rate."
tags:
  [
    "microservices",
    "caching",
    "redis",
    "performance",
    "cache-invalidation",
    "distributed-systems",
    "software-architecture",
    "backend",
    "reliability",
    "cdn",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/caching-strategies-across-services-1.webp"
---

The ShopFast catalog team had a number they were proud of and a number they were quietly terrified of. The proud number was the product-page load time: 45 milliseconds at the median, fast enough that nobody complained. The terrifying number was on a dashboard nobody looked at unless something was on fire — the read QPS hitting the catalog database, which on a normal afternoon sat around 8,000 and which, during the one big promotion they had run that quarter, had spiked to 31,000 and very nearly toppled the primary. The product page was fast because, almost by accident, an engineer two years earlier had wrapped the catalog read in a thirty-second cache. Nobody had designed a caching strategy. There was a `redis.get` here, a `@lru_cache` there, a CDN in front of the images, and a gateway that cached "some stuff" with rules nobody could fully reconstruct. It worked, mostly, until the afternoon a single hot product — a limited-edition sneaker drop — expired from the cache at the exact moment 5,000 people were refreshing the page, and 5,000 simultaneous cache misses landed on the database in the same fifty-millisecond window and brought the whole catalog down for ninety seconds.

That incident is the whole shape of this post. Caching is the single highest-leverage performance lever you have in a microservices system: a well-placed cache can cut your tail latency by an order of magnitude, cut the load on the service that owns the data, cut the load on that service's database, and cut your cloud bill all at once, with maybe forty lines of code. It is the first thing a senior reaches for when a service is slow or a database is hot, and it is usually the right thing. That is the easy part, and you can learn the easy part in an afternoon. The hard part — the part that gets people paged, the part that turns a clever optimization into a 3am incident — is *invalidation*: knowing when the cached copy is wrong and getting rid of it before it lies to a user. And in a microservices system that hard problem gets dramatically harder, because the data you want to cache is very often *owned by another service*. When the order service caches a product's price that the catalog service owns, the order service has no idea when the catalog team changes that price. That is not a coding problem you can solve with a cleverer `set`; it is a distributed-systems problem about how one service learns that another service's data changed.

The figure below is the read path we will spend the post building and defending: a product read that can be satisfied at the browser, at the CDN edge, at the API gateway, in a shared Redis cache, or — only as a last resort — by the catalog service and its database. Every layer that answers a read is a layer the slow, expensive database never sees.

![A branching read path for a product page showing a browser local cache and CDN edge in front of an API gateway response cache, a shared Redis product cache, the catalog service that owns the data, and the catalog database as the last resort](/imgs/blogs/caching-strategies-across-services-1.webp)

By the end of this post you will be able to do five concrete things. You will be able to pick the right cache *pattern* — cache-aside, read-through, write-through, write-behind, or refresh-ahead — for a given read-write shape, and name what each one costs. You will be able to place caches at the right *layers* of a microservices system and say what each layer is good for. You will be able to design cross-service invalidation that does not lie: event-driven eviction with a bounded TTL backstop, the combination I will argue you should reach for by default. You will be able to diagnose and fix a cache stampede — the thundering herd that took ShopFast down — with single-flight and jittered TTLs, and prove the fix with numbers. And you will be able to say, with a straight face in a design review, *which data you must never cache*, because the staleness would corrupt a decision. We will run all of it on ShopFast, finish with how Facebook, Netflix, and a real stale-cache post-mortem handled the same problems, and land on the one rule that separates juniors from seniors here: caching is easy; invalidation across a service boundary is the hard part, so prefer events plus a bounded TTL and never cache data whose staleness corrupts a decision.

This post closes the scale-and-optimization track. It builds directly on the [performance and cost optimization post](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices), which made the general case that the cheapest request is the one you never make — caching is how you not-make it. It leans on the [data consistency and eventual consistency post](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice), because a cache is just a deliberately stale replica and every consistency lesson there applies. And it forward-links to [multi-region microservices and data locality](/blog/software-development/microservices/multi-region-microservices-and-data-locality), where caching stops being an optimization and becomes a correctness-and-cost problem spread across continents.

## Why we cache: the arithmetic of not asking

Start with the why, because if you do not understand exactly what a cache buys you, you will cache the wrong things and feel clever doing it. A cache buys you three things, and they are worth naming separately because different workloads value them differently.

The first is **latency**. A read served from a local in-process cache is answered in nanoseconds to microseconds. A read served from a shared Redis instance over the network is answered in roughly a millisecond. A read that has to go to the owning service, which then queries its database, which then maybe joins a couple of tables and serializes the result, is answered in tens of milliseconds at the median and possibly hundreds at the tail. The cache is not 10% faster than the database; it is one to four orders of magnitude faster. That gap is why caching is the highest-leverage latency lever you have.

The second is **load on the owning service and its database**. This one matters more in microservices than people expect. When the order service caches product data, every cache hit is a request the *catalog* service never receives, and therefore a query its database never runs. A cache in front of a service is not just faster for the caller — it is a shield for the callee. ShopFast's catalog database survives Black Friday not because it can handle 31,000 reads per second, but because a 95% hit rate means it only ever *sees* about 1,500.

The third is **cost**. This follows directly from load. Database read capacity is expensive — you pay for the instance, you pay for the read replicas you added to spread the load, you pay for the cross-AZ network when those replicas live elsewhere. Cache capacity is cheap by comparison: RAM is cheap, and a Redis instance can serve six figures of operations per second on hardware that costs a fraction of the database it protects. Every read you serve from cache is a read you do not pay the database to serve. The performance-and-cost post made this the central thesis; here is the concrete mechanism behind it.

#### Worked example: what a 60% to 95% hit rate is actually worth

ShopFast's product-detail endpoint serves 10,000 reads per second at peak. The catalog database can comfortably serve about 6,000 reads per second before p99 latency starts to climb; past that it degrades, and past 12,000 it falls over. Start with the cache they had — a careless 60% hit rate, because TTLs were short and the key space was fragmented.

At 60% hit rate, 40% of 10,000 reads — that is 4,000 reads per second — miss the cache and hit the database. The database is below its 6,000 comfort line, but not by much; there is no headroom for a traffic spike, and p99 sits at 80ms because the database is working hard. Now tune the cache to a 95% hit rate (we will cover *how* later — longer-but-jittered TTLs, a warmer key space, two-tier caching). Now only 5% of reads — 500 per second — reach the database. The database load drops by a factor of eight, from 4,000 to 500 QPS. It is now loafing, p99 on a miss falls to 9ms because there is no queueing, and the database has 11x headroom for the next promotion.

Look at the end-to-end latency, because this is the part that surprises people. With a 95% hit and 5% miss, where a hit costs 1ms (Redis) and a miss costs 30ms (database round trip plus the cache write), the *average* read latency is `0.95 × 1ms + 0.05 × 30ms = 0.95 + 1.5 = 2.45ms`. With the 60% hit rate it was `0.60 × 1ms + 0.40 × 30ms = 0.6 + 12 = 12.6ms`. The jump from 60% to 95% — a 35-point improvement in hit rate — cut average latency by 5x *and* cut database load by 8x. That is the leverage. Note the non-linearity: hit rate matters most as it approaches 100%, because the misses are what cost you, and going from 95% to 99% (five-fold fewer misses) is often worth more engineering than going from 60% to 70%.

That non-linearity is the single most important intuition in this entire post. **The metric that matters is not hit rate, it is *miss* rate, because misses are what you pay for.** Halving your miss rate from 4% to 2% halves your database load even though hit rate barely moved from 96% to 98%. Keep that frame as we go.

The contrast is worth seeing as a picture, because it is the before-and-after that justifies every line of caching code you will ever write. On the left is the world with no cache: every single read travels all the way to the database, the database sits pinned near saturation, and the tail latency creeps upward with nowhere to hide. On the right is the same workload with a cache-aside layer absorbing the overwhelming majority of reads: the database load collapses to a fraction, the tail latency goes flat, and the system gains headroom it did not have before. The shape of that picture — a slow, hot, fragile path becoming a fast, cool, resilient one — is the entire reason caching is the first lever a senior reaches for.

![A before and after comparison contrasting a system with no cache where every read hits a saturated database at high tail latency against a cache-aside system where most reads are served from Redis, database load drops to a fraction, and tail latency goes flat](/imgs/blogs/caching-strategies-across-services-4.webp)

One more framing before we get into patterns, because juniors often miss it: a cache is most valuable precisely where it is most dangerous to get wrong. The hotter a key is, the more a cache hit saves you — and the more a *stale* cache entry on that hot key spreads a wrong value to many users at once. A cold, rarely-read key barely benefits from caching and barely hurts when stale; a blazing-hot key delivers enormous savings and, if stale, an enormous blast radius. So the engineering effort you invest in a cache should scale with how hot the data is, not be spread evenly. The hottest keys deserve the most careful invalidation, the tightest staleness bounds, and the stampede defenses we will build; the long tail of cold keys can get a simple TTL and nobody will ever notice. This is why "just cache everything with the same TTL" is the mark of someone who has not yet been paged — the uniform treatment over-invests in the cold keys and under-protects the hot ones.

## The cache patterns: five ways to wire it up

There are exactly five patterns for how reads and writes flow through a cache, and they are not interchangeable. Knowing which one you are using — and most importantly *who is responsible for keeping the cache correct* — is the difference between a cache that helps and a cache that lies. The matrix below is the whole decision in one frame; I will walk each row.

![A decision matrix comparing cache-aside, read-through, write-through, write-behind, and refresh-ahead across read latency, write latency, consistency, and complexity, showing that no pattern wins on every axis](/imgs/blogs/caching-strategies-across-services-3.webp)

**Cache-aside (lazy loading)** is the default, and it is the one you should reach for unless you have a specific reason not to. The application code owns the cache. On a read, the code checks the cache first; on a hit it returns the cached value, and on a miss it reads from the database, writes the value into the cache, and returns it. The cache is populated *lazily* — only keys that someone actually asked for ever get cached, which is exactly what you want, because cache memory is finite and you do not want to warm keys nobody reads. On a write, the code writes the database and then *invalidates* (deletes) the cache key, so the next reader will lazily repopulate it with fresh data. Cache-aside is simple to reason about, it is resilient (if the cache is down, reads just fall through to the database — slower, but correct), and it gives you full control. Its cost is the cold-miss latency (the first reader of any key pays the full database round trip) and a subtle race we will return to: between the delete-on-write and the next read, a concurrent reader can repopulate the cache with stale data.

**Read-through** is cache-aside with the cache-population logic moved *into* the cache layer instead of the application. The application asks the cache; on a miss, the cache library (or a sidecar) fetches from the database, stores the value, and returns it, all transparently. The read profile is identical to cache-aside; the difference is purely where the code lives. Read-through is nice for keeping application code clean and for centralizing the cache-loading logic so every caller behaves the same way, but you pay for it in coupling — your cache layer now needs to know how to load from your data store, which is fine for a library like an ORM second-level cache and awkward for a shared Redis that many services use differently.

**Write-through** changes the *write* path: on a write, the application writes to the cache *and* the database synchronously, in the same operation, before returning. The benefit is that the cache is always warm and always consistent with the database — there is no stale window, because the cache is updated at the moment of write. The cost is write latency: every write now pays for two writes (cache plus database) on the critical path. And it caches data on write that nobody may ever read, wasting cache memory unless you combine it with a read pattern. Write-through shines when you read far more than you write *and* you cannot tolerate even a brief stale window after a write.

**Write-behind (write-back)** is write-through's dangerous cousin. On a write, the application writes only to the cache and returns immediately; a background process flushes the cache to the database asynchronously, often batching many writes together. Write latency is fantastic — you only wait for the in-memory write. The catch is brutal: if the cache node dies before the flush, those writes are *gone*. You have traded durability for write speed. Write-behind is the right call for high-volume writes where some loss is acceptable (view counters, metrics, last-seen timestamps) and absolutely the wrong call for anything you must not lose (orders, payments, inventory decrements).

**Refresh-ahead** is a read optimization layered on top: the cache proactively refreshes hot keys *before* they expire, predicting that a frequently-read key will be read again. The reader never sees a cold miss, because the value is always warm. This is the cleanest defense against the cache stampede we will dissect later, because it removes the synchronized-expiry event entirely. The cost is wasted refreshes (you refresh keys that then *don't* get read) and complexity (you need to track access patterns to decide what to refresh). It is excellent for a small set of very hot keys and wasteful applied indiscriminately.

Here is cache-aside in Python — the pattern you will write 90% of the time. Read the comments; the subtleties are all in the write path.

```python
import json
import redis

cache = redis.Redis(host="redis", port=6379)
TTL_SECONDS = 300  # 5-minute backstop, see invalidation section

def get_product(product_id: str) -> dict:
    key = f"product:{product_id}"
    # 1. Try the cache first.
    cached = cache.get(key)
    if cached is not None:
        return json.loads(cached)          # HIT — the common case, ~1ms
    # 2. Miss: load from the source of truth (the catalog DB).
    product = db.query_product(product_id)  # ~30ms
    if product is None:
        # negative caching: remember the miss briefly so a bad id
        # cannot hammer the DB on every request (see "negative caching")
        cache.setex(key, 30, json.dumps({"__missing__": True}))
        return None
    # 3. Populate the cache for the next reader, with a bounded TTL.
    cache.setex(key, TTL_SECONDS, json.dumps(product))
    return product

def update_product(product_id: str, changes: dict) -> None:
    # Write the source of truth FIRST. The DB is authoritative.
    db.update_product(product_id, changes)
    # Then INVALIDATE, do not UPDATE, the cache. Deleting is safer than
    # writing the new value here, because under concurrency a stale
    # in-flight read could otherwise overwrite your fresh write.
    cache.delete(f"product:{product_id}")
```

Two things in that snippet are load-bearing and worth saying out loud. First, **invalidate on write, do not update on write** — delete the key rather than setting the new value. Setting the new value seems more efficient (no cold miss for the next reader) but it opens a write-after-write race: a slow read that started before your write can complete after it and write its stale value over your fresh one. Deleting is idempotent and race-tolerant; the next read simply repopulates. Second, **write the database before touching the cache**. The database is the source of truth. If you invalidate the cache first and the database write then fails, you have evicted a correct value to repopulate it with the same correct value — harmless. If you wrote the cache first and the database write failed, your cache now disagrees with the truth — a lie.

## Where to cache: the layers, and what each is for

A microservices system does not have *a* cache; it has a stack of them, and the art is knowing what to put where. The figure below is the stack for a product read, from the browser at the top to the database at the bottom. Each tier is cheaper, closer to the user, and able to absorb load so that fewer reads reach the tier below — which is precisely the goal, because the tier at the bottom is the slowest and most expensive one to hit.

![A vertical stack of cache tiers showing the browser cache, CDN edge, gateway cache, in-process L1 cache, shared Redis L2 cache, and the database at the bottom, each absorbing reads so fewer reach the next tier down](/imgs/blogs/caching-strategies-across-services-2.webp)

**The browser / client cache** is the layer everyone forgets and the cheapest one of all, because the data never leaves the user's machine. You control it with HTTP response headers: `Cache-Control`, `ETag`, and friends. For a product image, a `Cache-Control: max-age=31536000, immutable` says "this never changes, keep it for a year," and the browser stops asking. For a product page, a shorter max-age plus an `ETag` lets the browser revalidate cheaply (a 304 Not Modified is a tiny response). This is free latency and free load reduction, and it costs you a header.

**The CDN / edge cache** sits in points of presence around the world and is the right home for anything static or semi-static that many users request: images, CSS, JavaScript bundles, and — increasingly — cacheable API responses. ShopFast serves every product image from a CDN with a year-long TTL and a content-hashed URL, so the CDN gets a near-100% hit rate and the origin almost never sees an image request. The edge is also where you get geographic latency wins for free: a user in Singapore hits a Singapore PoP, not your us-east-1 origin. The CDN layer connects directly to the [multi-region post](/blog/software-development/microservices/multi-region-microservices-and-data-locality) — pushing cached data to the edge is the first and cheapest form of data locality.

**The API gateway cache** is the first server-side layer you control end-to-end. The gateway sits in front of all your services (see [the API gateway and backend-for-frontend post](/blog/software-development/microservices/the-api-gateway-and-backend-for-frontend)) and is a natural place to cache whole *responses*: the catalog endpoint's JSON for a popular product, cached at the gateway, means a hit there never even reaches the catalog service. The gateway cache is shared across all users, so its hit rate is high for popular content, and it is governed by the same `Cache-Control` headers your services emit. The trade-off: it caches at the response granularity, so personalized responses (anything that varies by user) either cannot be cached here or need careful `Vary` handling.

**The in-process / local cache (L1)** lives inside a single service instance's memory — a `caffeine` cache in Java, an `lru_cache` in Python, a `sync.Map` or a library like `ristretto` in Go. It is the fastest cache there is, because there is no network hop: hits are nanoseconds. Its limits define its use: it is *per-instance*, so ten replicas have ten independent caches (ten times the database misses on cold start), it is small (bounded by the instance's memory), and it is hard to invalidate across instances (when data changes, how do you tell all ten replicas to evict?). The local cache is perfect for small, hot, slowly-changing reference data — a feature-flag map, a currency table, a set of category names — and dangerous for large or frequently-changing data.

**The shared distributed cache (L2)** — Redis or Memcached — is the workhorse of microservices caching. It is a separate tier that all instances of all services can read and write, so it has one shared view: a write by one instance is visible to all, and an invalidation by one is seen by all. It is bigger than any local cache (you size it independently), it survives instance restarts (a redeployed service finds a warm cache), and it is the layer where cross-service caching actually happens — the order service and the catalog service can both read the same `product:123` key. Its cost is the network hop (about 1ms versus nanoseconds for local) and that it is a shared dependency you now have to operate, scale, and protect. Most real systems combine L1 and L2 in a two-tier cache, which we will build later.

Here is the gateway/CDN layer expressed where it actually lives — in the response headers your service emits. This is the most under-used caching tool in microservices, because engineers think of caching as Redis code and forget that an HTTP header can make a whole CDN and every browser cache for you.

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/products/{product_id}")
def get_product_endpoint(product_id: str, response: Response):
    product = get_product(product_id)  # cache-aside, from earlier
    # Public: any shared cache (CDN, gateway) may store this.
    # max-age: browsers cache for 60s.
    # s-maxage: shared caches (CDN/gateway) cache for 300s — longer,
    #   because a shared cache amortizes one origin fetch over many users.
    # stale-while-revalidate: serve stale up to 30s while refreshing in
    #   the background, so a user NEVER waits on a refresh (refresh-ahead
    #   at the HTTP layer).
    response.headers["Cache-Control"] = (
        "public, max-age=60, s-maxage=300, stale-while-revalidate=30"
    )
    response.headers["ETag"] = product["version_etag"]
    return product
```

The `stale-while-revalidate` directive is quietly one of the best tools in the box: it lets a CDN or gateway serve a slightly-stale response instantly while it refreshes the entry in the background, so a user is never blocked on a cache refresh. It is refresh-ahead, but you get it for free from any compliant CDN by emitting one header. For anything that does not require strict freshness — and most of a product page does not — this is close to optimal.

## The cross-service caching dilemma: who tells you the data changed?

Now the hard part, and the reason this post exists. Everything above is the easy 80%. Here is the 20% that is genuinely hard, and it is hard specifically *because* this is a microservices system.

In a monolith, caching is a contained problem. The code that reads the data, the code that writes the data, and the cache all live in the same process. When you update a product, the same module that wrote it can invalidate the cache in the same breath. You always know when the data changed, because *you* changed it.

In microservices, the [database-per-service rule](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) means data is owned by exactly one service, and only that service writes it. So picture ShopFast's order service, which needs product details — name, price, image URL — to render an order summary. The product data is owned by the catalog service. The order service does not own it, cannot write it, and is told (correctly) never to reach into the catalog's database directly. So the order service *borrows* the data: it calls the catalog service, gets the product, and — because calling catalog on every order render is slow and hammers catalog — it caches the result in Redis.

Now ask the question that breaks juniors: **the catalog team changes a product's price. How does the order service's cache find out?**

It does not. That is the entire problem. The order service cached a price that catalog owns, and catalog has no idea the order service even has a copy. The catalog team changes the price in their database, emits no signal the order service hears, and the order service serves the old price from its cache for as long as the cache entry lives. The cache is now lying, and nothing told it to stop. "Just invalidate it" is not an answer, because *invalidate it from where?* — the code that changed the data lives in a different service, a different repository, a different deploy, possibly a different team. Cross-service cache invalidation is not a caching problem; it is a distributed-systems problem about propagating a change across a service boundary. There are exactly three serious answers, plus the one you should usually combine. The matrix lays them out.

![A decision matrix comparing TTL expiry, event-driven eviction, versioned keys, and the combination of TTL plus event across staleness, complexity, cross-service fit, and failure mode](/imgs/blogs/caching-strategies-across-services-6.webp)

**TTL (time-to-live)** is the simplest answer and the one you should always have as a floor. Every cache entry gets an expiry; when it expires, the next read repopulates from the source. The order service caches the product for 300 seconds; after 300 seconds it re-fetches from catalog and picks up any change. TTL is trivial to implement (one parameter), it requires zero coordination between services (catalog does not even need to know the order service exists), and it self-heals (a wrong entry is wrong for at most the TTL, then fixes itself). Its cost is the **stale window**: for up to the full TTL, the order service can serve data that changed. If the price changed one second after caching, the order service shows the old price for the next 299 seconds. For a product name that rarely changes, a 300-second stale window is nothing. For a price that drives a purchase decision, it might be unacceptable. TTL trades freshness for simplicity, and the knob is the TTL length: shorter means fresher but more misses (more load on catalog), longer means fewer misses but staler.

**Event-driven invalidation** is the strong answer and the one that actually solves the "how does A know B changed" question. The owning service publishes a change event; borrowing services subscribe and evict the affected key. When catalog changes a product, it emits a `ProductUpdated` event onto the event bus; the order service consumes that event and deletes `product:123` from its cache. The next read repopulates with fresh data. Staleness is now near-zero — bounded by the event-propagation lag (typically tens to a few hundred milliseconds), not by the TTL. This is the [event-driven microservices pattern](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) applied to cache coherence, and it is frequently powered by [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) so the event is emitted reliably as part of the same transaction that changed the data. Its cost is real complexity: you need a reliable event bus, the owning service must reliably emit events (a dropped event means a permanently stale entry — see the failure-mode column), and the borrowing service needs a consumer that maps events to cache keys.

**Versioned / keyed caching** sidesteps invalidation entirely with a clever trick: bake a version into the cache key. Instead of caching under `product:123`, cache under `product:123:v7`, where `v7` is the current version of that product. When the product changes, its version bumps to `v8`, and now every read uses the new key `product:123:v8` — which is a guaranteed miss, so it repopulates fresh, and the old `product:123:v7` entry is simply never read again (it expires on its own TTL eventually). You never *evict*; you just stop *referencing* the stale key. This is beautiful for content-addressed or immutable-ish data (it is exactly how CDNs handle versioned asset URLs), but it requires the reader to know the current version, which usually means the version travels with some other piece of data the reader already has, or you are back to needing a signal. It also accumulates dead keys until they expire, so it costs cache memory.

Here is the event-driven invalidation consumer — the heart of the strong answer. The order service runs this consumer; it turns catalog's `ProductUpdated` events into cache evictions.

```go
package cache

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/redis/go-redis/v9"
)

type ProductUpdated struct {
	ProductID string `json:"product_id"`
	Version   int    `json:"version"`
	EventID   string `json:"event_id"` // for dedup; see idempotency post
}

// Consumes ProductUpdated events emitted by the catalog service and
// evicts the borrowed copy from the order service's Redis cache.
func (c *Consumer) handleProductUpdated(ctx context.Context, raw []byte) error {
	var ev ProductUpdated
	if err := json.Unmarshal(raw, &ev); err != nil {
		return fmt.Errorf("bad event: %w", err) // park in DLQ, don't ack
	}

	key := fmt.Sprintf("product:%s", ev.ProductID)
	// DELETE, not overwrite. We do not have the new value here, and even
	// if we did, deleting is race-safe. The next reader repopulates from
	// the catalog service's API with fresh data.
	if err := c.rdb.Del(ctx, key).Err(); err != nil {
		// Eviction failed. Do NOT ack — let the event be redelivered.
		// The TTL backstop below means even a permanently failed evict
		// self-heals within the TTL. Defense in depth.
		return fmt.Errorf("evict %s: %w", key, err)
	}
	return nil // ack: the cache no longer holds a stale copy
}
```

The single most important design decision in cross-service caching is the last row of that matrix: **combine event-driven invalidation with a bounded TTL backstop.** Use events for freshness — the common case, where staleness drops to milliseconds. Use the TTL as insurance — so that *if an event is ever lost* (the bus dropped it, the consumer was down, a bug ate it), the stale entry still self-heals within the TTL instead of lying forever. Events alone have a catastrophic failure mode (a missed event means permanent staleness); TTL alone is always at least a little stale. Together they cover each other: events keep it fresh in the common case, TTL caps the damage in the failure case. This is the recommendation I will defend to a senior: events plus a bounded TTL is the cross-service caching default. The figure shows it.

![A branching invalidation flow where the catalog service writes its database and emits a ProductUpdated event onto a bus, which fans out to the order service cache, the gateway cache, and the search index, each evicting, with a TTL backstop covering any lost event](/imgs/blogs/caching-strategies-across-services-7.webp)

#### Worked example: TTL stale window versus event-driven lag

ShopFast's order service borrows product prices from catalog. A merchandiser changes the price of the hot sneaker at 14:00:00.000. Question: how long does the order service show the old price under each strategy?

With a **300-second TTL and no events**, it depends on when the entry was last populated. If the order service had cached the price at 13:58:00, the entry expires at 14:03:00, so the old price is served from 14:00:00 to 14:03:00 — a **180-second stale window** for this particular entry, and in the worst case (cached at 13:59:59) it is the full 300 seconds. Every customer who renders that order in those three minutes sees the wrong price. For a product page that is mildly embarrassing; for a checkout total it is a support ticket and possibly a chargeback dispute.

With **event-driven invalidation**, catalog commits the price change at 14:00:00.000 and emits `ProductUpdated`. Measure the lag: the outbox poller picks up the event in, say, 50ms; the broker delivers it in 5ms; the consumer processes and deletes the key in 10ms. Total lag ≈ **65ms**. From 14:00:00.000 to 14:00:00.065, the order service can serve the old price (a 65ms window where an in-flight read might still hit the old entry), and after that the key is gone and the next read is fresh. That is the difference between a three-*minute* stale window and a 65-*millisecond* one — a factor of roughly 2,700x — for the cost of running an event consumer.

With the **combination**, you get the 65ms freshness of events in the normal case, and the 300-second TTL only matters on the rare day the event is lost — in which case the worst case degrades gracefully back to "stale for at most 300 seconds" instead of "stale forever." You almost never pay the TTL cost, but it is there as a floor under your correctness. That is why the combination, not either alone, is the senior default.

## The cache stampede: when a hot key expires under load

Now we get to the failure that actually took ShopFast down, because it is the most common cache-related outage and the one juniors do not see coming. It goes by several names — cache stampede, thundering herd, dog-piling — and they all describe the same event: a hot key expires, and the flood of concurrent readers that were all being served from that one key now *all miss at once* and *all* go to the database in the same instant.

Here is the mechanism precisely. The sneaker product is hot: 5,000 reads per second, all served from one Redis key, the database completely shielded. The key has a 300-second TTL. At T+0 it expires. In the next few milliseconds, every one of those 5,000 readers checks the cache, finds nothing, and — following the cache-aside pattern faithfully — each one independently queries the database to repopulate the key. The database, which was serving zero queries for this product because the cache was shielding it, suddenly gets 5,000 identical queries in a 50ms window. It saturates, p99 latency goes to seconds, the queries pile up, and because the database is a shared dependency, *other* queries slow down too, and the whole catalog browns out. The cache was doing its job perfectly right up until the instant it wasn't, and then it failed all at once. The timeline tells the story.

![A six-event timeline of a cache stampede showing a hot product cached with a 300-second TTL, serving 5000 requests per second, then the key expiring, 5000 simultaneous misses all querying the database, the database saturating with p99 spiking to 3 seconds, and the single-flight fix collapsing it to one query](/imgs/blogs/caching-strategies-across-services-5.webp)

The fix has two complementary parts, and a good system uses both.

The first is **single-flight (request coalescing / locking)**: when a hot key misses, let exactly *one* reader rebuild it while every other reader *waits* for that rebuild instead of independently querying the database. The one reader takes a short-lived lock (or uses a single-flight primitive), does the single database query, writes the cache, and releases; the 4,999 others see the lock, wait a few milliseconds, then find the freshly-populated key and hit. 5,000 misses become *one* database query. Go's standard library ships exactly this primitive, `golang.org/x/sync/singleflight`, and it is the cleanest expression of the idea.

```go
package product

import (
	"context"
	"encoding/json"
	"time"

	"github.com/redis/go-redis/v9"
	"golang.org/x/sync/singleflight"
)

type Service struct {
	rdb *redis.Client
	g   singleflight.Group // coalesces concurrent rebuilds of the same key
}

func (s *Service) GetProduct(ctx context.Context, id string) (*Product, error) {
	key := "product:" + id

	// Fast path: cache hit.
	if raw, err := s.rdb.Get(ctx, key).Bytes(); err == nil {
		var p Product
		if json.Unmarshal(raw, &p) == nil {
			return &p, nil
		}
	}

	// Miss. singleflight.Do ensures that of N concurrent goroutines that
	// all miss "product:123" at once, exactly ONE runs the rebuild func;
	// the other N-1 BLOCK and receive that one's result. 5000 misses
	// collapse to 1 DB query.
	v, err, _ := s.g.Do(key, func() (interface{}, error) {
		p, err := s.loadFromCatalog(ctx, id) // the single DB/API call
		if err != nil {
			return nil, err
		}
		raw, _ := json.Marshal(p)
		// jittered TTL so this hot key does not all-expire-at-once again
		s.rdb.Set(ctx, key, raw, jitteredTTL(5*time.Minute))
		return p, nil
	})
	if err != nil {
		return nil, err
	}
	return v.(*Product), nil
}
```

`singleflight` only coalesces within one process. In a fleet of N service instances you can still get N concurrent rebuilds (one per instance), which is a 5,000-to-N improvement — usually plenty — but if even N is too many, you take a *distributed* lock in Redis (`SET key value NX PX 5000`) so exactly one instance across the whole fleet rebuilds. Most systems do not need the distributed lock; per-process single-flight plus the next fix is enough.

The second fix is **jittered TTL**: never give a batch of keys the *same* expiry, because then they all expire together and you get a synchronized stampede (and its bigger cousin, the cache avalanche, where thousands of keys warmed by one deploy all expire in the same second). Instead of `TTL = 300`, use `TTL = 300 ± random(0, 60)`. Now the hot key's expiry is spread across a window, no two readers' keys expire at the same instant, and the load smears out instead of spiking.

```python
import random

def jittered_ttl(base_seconds: int, jitter_pct: float = 0.2) -> int:
    # base 300s, jitter 20% -> a TTL uniformly in [240, 360].
    # Spreads expiries so a batch of keys warmed together does not
    # all expire in the same second (avalanche) and so one hot key's
    # expiry is not a single synchronized cliff (stampede).
    jitter = int(base_seconds * jitter_pct)
    return base_seconds + random.randint(-jitter, jitter)
```

The third, which we already met, is **refresh-ahead / `stale-while-revalidate`**: refresh the hot key *before* it expires so there is never a miss event at all. For your hottest keys this is the strongest defense, because it removes the expiry cliff entirely rather than managing the herd at the cliff. The mature pattern is layered: refresh-ahead for the few hottest keys, single-flight to coalesce any miss that does slip through, and jittered TTLs everywhere so misses never synchronize. This connects to the [rate-limiting and backpressure post](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) — single-flight is, in effect, a concurrency limit of one on the rebuild of a given key, which is backpressure applied to your own database.

![A before and after comparison contrasting a naive cache miss where 5000 concurrent misses each run a database query and saturate it against a single-flight design where one leader rebuilds the key, the rest wait then hit, and exactly one database query runs](/imgs/blogs/caching-strategies-across-services-8.webp)

#### Worked example: single-flight cuts 5,000 queries to 1

Take ShopFast's incident exactly. The sneaker key serves 5,000 reads/second; on expiry, the readers in the next 50ms window all miss. Without protection, with the cache-aside pattern as originally written, that is `5000 reads/s × 0.050s ≈ 250` misses in the 50ms it takes the first reader to repopulate — but in practice the rebuild itself is slow because the database is now overloaded, so the window stretches: if the database takes 300ms under load to answer, then `5000 × 0.3 = 1,500` readers miss before the first one finishes writing the cache, and all 1,500 query the database. The database, sized for ~6,000 QPS of *normal mixed* traffic, gets 1,500 identical heavy queries on top of everything else, tips past its limit, and p99 climbs to 3 seconds — which lengthens the rebuild further, which lets *more* readers miss. That is the positive feedback loop that turns a 50ms blip into a 90-second outage.

With per-process single-flight across, say, 20 service instances: each instance coalesces its own concurrent misses to one rebuild, so the database sees at most **20** queries (one per instance) instead of 1,500 — a 75x reduction, and well within the database's headroom, so there is no overload, no feedback loop, and the rebuild finishes in the normal 30ms. With a *distributed* lock on top, the database sees exactly **1** query for the whole fleet. Either way the stampede is dead. Add jittered TTLs and the hot key's neighbors do not all expire alongside it, so the next expiry is a trickle, not a cliff. ShopFast shipped per-process single-flight plus 20%-jittered TTLs the week after the incident; the database read spikes on expiry went from "near-outage" to "invisible on the graph."

## Stale data, consistency, and what you must never cache

Step back from mechanics to the principle, because this is where seniority shows. **A cache is a deliberately stale replica.** Every cache trades freshness for speed; the only question is how much staleness you can tolerate, and that question is not a caching question — it is a *business correctness* question. The [data consistency post](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) makes the general argument that microservices live in an eventually-consistent world; a cache is one of the most common sources of that eventual consistency, and the discipline is the same: decide per piece of data what staleness is safe.

Some data tolerates staleness happily. A product *name* changes maybe once a year; a 300-second stale window is irrelevant. A product *description*, *image*, *category* — all fine to cache for minutes. A "customers also bought" recommendation can be hours stale and nobody notices or cares. This is the bulk of a typical read workload, and caching it aggressively is pure upside.

Some data tolerates *bounded* staleness. A product *price* can usually be a few seconds stale on a browse page (you re-validate at checkout). Inventory *availability* can show "in stock" when it just sold out, *as long as* you re-check authoritatively at the moment of purchase. The pattern here is to cache the optimistic display value but make the *decision* against the source of truth.

And some data you must **never** cache, because its staleness corrupts a decision. The canonical example is anything you must be *exact* about at the moment of action:

- **An account balance you are about to debit.** If you cache a balance and debit against the cached value, two concurrent debits both see the stale "sufficient" balance and you allow an overdraft. Read the balance from the source of truth, under the right consistency guarantee, every time you act on it.
- **Inventory at the point of decrement.** Browse can show cached stock; the *decrement* must hit the authoritative count, or you oversell.
- **Authorization decisions on revocation-sensitive operations.** If you cache "user X may access resource Y" and X's access is revoked, a cached "yes" is a security hole. (See the [auth and token-propagation post](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation) — short-lived tokens are the standard way to bound this staleness.)
- **The result of a saga step you are about to commit on.** Caching intermediate state that a [saga](/blog/software-development/database/saga-pattern-distributed-transactions) is mid-flight on invites acting on a value that has since been compensated.

The senior rule is one sentence: **never cache data whose staleness corrupts a decision.** It is fine to cache a balance to *show* it ("your balance is approximately…"); it is never fine to cache a balance to *debit against it*. The display tolerates staleness; the decision does not. Get this distinction wrong and the cache stops being a performance optimization and becomes a correctness bug with great latency numbers. The decision tree below is how I teach this.

![A decision tree for whether to cache a piece of data, branching first on whether stale data corrupts a decision into never-cache, then on whether the data is owned by another service into event-driven eviction with a TTL versus a local TTL cache-aside](/imgs/blogs/caching-strategies-across-services-9.webp)

There is also a quieter correctness trap specific to cache-aside: the **stale-set race**. Reader R1 misses, queries the database, and gets value V1. Before R1 writes V1 to the cache, a writer updates the database to V2 and invalidates the (still-empty) cache. Now R1 finally writes V1 — the *old* value — into the cache, and the cache holds V1 while the database holds V2, with nothing to fix it until the TTL. This is rare but real under high concurrency. The mitigations: keep TTLs bounded (the backstop fixes it within the TTL), use the version-in-key trick for data where it matters (R1 would write `product:123:v1`, but readers now ask for `product:123:v2`, so R1's stale write is never read), or use a check-and-set on the cache write. For most data the bounded TTL is enough; know the race exists so you recognize it when a key is mysteriously stale.

## Hot keys, tiering, and consistent hashing

A cache stampede is one failure mode of a hot key; there is a second, structural one. A distributed cache like Redis spreads keys across nodes by hashing the key. Normally that spreads load evenly. But if one key is *vastly* hotter than the rest — the sneaker drop again — then *all* the traffic for that key lands on the *single* node that owns it, and that one node melts while the others idle. This is the hot-key problem, and it is not solved by adding cache nodes, because the hot key still lives on exactly one of them. How keys map to nodes is governed by [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) — worth reading for the mechanism — but the practitioner's point is that *consistent hashing distributes keys, not load*, and a single hot key is a load that no key-distribution scheme can split.

There are two practical fixes. The first is **local-cache tiering**: put a small in-process L1 cache in front of the shared Redis L2. The hottest keys get served from L1 inside each service instance — nanoseconds, zero network, and crucially *zero* load on the hot Redis node, because the L1 hit never reaches Redis. The hot key is now served from N local caches (one per instance) instead of one Redis node, and the load is spread across the entire fleet's memory. The second fix, for read-heavy hot keys, is **key replication**: store the hot key under several suffixed keys (`product:123#0`, `product:123#1`, …) that hash to different nodes, and have readers pick one at random, spreading the load across nodes at the cost of N-fold invalidation. Tiering is the common answer; key replication is the heavier hammer for the rare extreme case.

Here is the two-tier cache — L1 local plus L2 Redis — which is the production shape for hot-key-prone data. It is also a nice illustration of the whole post: cache-aside at both tiers, bounded TTLs, and a short L1 TTL specifically so an invalidation (which evicts L2) self-corrects in L1 quickly.

```go
package cache

import (
	"context"
	"encoding/json"
	"time"

	"github.com/hashicorp/golang-lru/v2/expirable"
	"github.com/redis/go-redis/v9"
)

type TwoTier struct {
	l1  *expirable.LRU[string, []byte] // in-process, nanoseconds
	rdb *redis.Client                  // shared Redis, ~1ms
}

func NewTwoTier(rdb *redis.Client) *TwoTier {
	// L1 is small and SHORT-LIVED: 10s TTL, 10k entries. The short TTL
	// bounds how long L1 can disagree with an L2 invalidation, since an
	// event evicts L2 but cannot cheaply reach every instance's L1.
	l1 := expirable.NewLRU[string, []byte](10_000, nil, 10*time.Second)
	return &TwoTier{l1: l1, rdb: rdb}
}

func (t *TwoTier) Get(ctx context.Context, key string) ([]byte, bool) {
	if v, ok := t.l1.Get(key); ok {
		return v, true // L1 hit — never touches Redis, shields the hot node
	}
	v, err := t.rdb.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false // L2 miss — caller does cache-aside rebuild
	}
	t.l1.Add(key, v) // promote into L1 for the next reader on this instance
	return v, true
}

func (t *TwoTier) Set(ctx context.Context, key string, val []byte, ttl time.Duration) {
	t.l1.Add(key, val)
	t.rdb.Set(ctx, key, val, ttl)
}
```

The deliberate design choice there is the **short L1 TTL**. An event-driven invalidation can cheaply evict the shared L2 (one `DEL` to Redis), but evicting every instance's L1 requires broadcasting the invalidation to every instance — possible (a Redis pub/sub channel that all instances subscribe to and evict on) but more machinery. The pragmatic compromise most teams choose: keep L1 TTLs short (5–30 seconds) so L1 staleness is tightly bounded *without* a broadcast, and reserve the broadcast-evict for the small set of data where even 10 seconds of L1 staleness is unacceptable. This is the freshness-versus-complexity trade-off in miniature, decided per data type.

## Measuring and tuning: hit rate is the number that matters

You cannot improve what you do not measure, and the number to measure is the cache hit rate — or, per the earlier intuition, the *miss* rate, because misses are what cost you. Instrument every cache with hit and miss counters, broken down by key prefix so you can see *which* data is missing, and watch the ratio. A good product cache should run 90%+; a great one 95–99%. If yours is at 60%, that is not a tuning problem, it is a design problem, and there is usually one of a few culprits.

**Fragmented keys** are the most common cause of a low hit rate. If you cache under `product:123:user:456:locale:en-US:currency:USD`, you have multiplied the key space by every dimension, and each key is read rarely, so each one is usually cold. Cache at the *coarsest granularity that is still correct*: cache the product once, and apply the user/locale/currency transformations *after* the cache read, not in the key. ShopFast's 60% hit rate was largely this — they were caching fully-rendered, per-user product responses, so every key was nearly unique. Splitting it into a shared `product:123` cache plus a cheap per-request render took the hit rate to 95% overnight, which is exactly the worked-example jump from earlier.

**TTLs too short** is the second culprit. A 30-second TTL on data that changes hourly throws away 119 perfectly good cache hits for every real change. Lengthen the TTL toward how often the data *actually* changes, lean on event-driven invalidation for freshness so you can afford a long TTL backstop, and jitter it so the lengthening does not synchronize expiries.

**Cold caches after deploy** quietly tank your hit rate: every deploy restarts instances with empty L1 caches, so you get a miss spike on every release. The fixes are to keep the durable state in the shared L2 (which survives instance restarts) rather than relying on L1, and for critical hot keys, to *warm* the cache on startup by pre-fetching the known-hot keys before taking traffic.

**Cache sizing and eviction policy** is the tuning lever people skip and then blame the cache for. A cache holds a bounded amount of memory; when it fills, it evicts entries by a policy — usually LRU (least recently used) or one of its approximations. If your cache is too small for the working set, it thrashes: it evicts entries that are about to be read again, your hit rate craters, and you are paying for a cache that mostly misses. The fix is to size the cache to hold the *working set* — the set of keys actually read within a window — which you find empirically by growing the cache until the hit rate plateaus. ShopFast found that bumping their Redis from 8GB to 16GB took the hit rate from 91% to 96%, because the extra room held the long tail of moderately-popular products that had been getting evicted; past 16GB the curve flattened, so that was the right size. Watch the eviction-rate metric alongside the hit rate: a high eviction rate with a mediocre hit rate is the unmistakable signature of an undersized cache.

Two more tools belong in the kit. **Negative caching** — caching the *absence* of a value, as in the `__missing__` sentinel in the first code snippet — stops a bad or non-existent ID from hammering the database on every request, which is both a performance fix and a defense against an attacker probing random IDs. Cache the miss briefly (shorter than a hit, because the value might appear), and you turn a flood of "not found" database queries into one. And **knowing what *not* to cache**: do not cache data that is read once (no second reader to benefit), data that changes on nearly every read (the cache is stale before it is read again), or — the senior point again — data whose staleness corrupts a decision. A cache on the wrong data is pure cost: memory, a stale-bug surface, and operational weight, for no hit-rate benefit.

**Cache warming** closes the loop on the cold-deploy problem. For the handful of keys you know will be hot the instant traffic arrives — the homepage products, the items in an active promotion — pre-fetch them into the cache on service startup, before the instance reports ready and starts taking requests (this dovetails with the readiness-probe discipline from the [health checks post](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing): do not go ready until the cache is warm enough to serve without stampeding the database). ShopFast warms the top 500 products on boot, which takes about two seconds and turns the post-deploy miss spike — historically a small but visible bump on the database graph after every release — into nothing at all. Warming is cheap insurance for a known, bounded set of hot keys; do not try to warm the whole catalog, because most of it is cold and you would just be pre-loading keys nobody reads.

#### Worked example: the database-load and cost win, in dollars

Put numbers on the whole thing. ShopFast's catalog read workload at peak is 10,000 QPS. Before the caching cleanup, at a 60% hit rate, the database saw 4,000 QPS and needed a large primary plus three read replicas to handle it with headroom — call it four `db.r6g.2xlarge`-class instances at roughly \$1,100/month each, about \$4,400/month, and even then p99 sat at 80ms with no spike headroom. After the cleanup to a 95% hit rate, the database sees 500 QPS — comfortably one primary plus one replica for failover, about \$2,200/month — plus a Redis cluster to serve the 9,500 QPS of hits, say three cache nodes at \$200/month, \$600/month. The new total is \$2,800/month against the old \$4,400/month: roughly \$1,600/month saved, p99 down from 80ms to under 10ms, *and* the system now has an order of magnitude more read headroom for the next promotion. The cache did not just make things faster; it made them cheaper and more resilient at the same time. That triple win — latency, load, cost — is why caching is the first lever a senior pulls, and the [performance-and-cost post](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices) frames the broader version of this calculation.

## Stress-testing the design

A design you have not tried to break is a hope, not a design. Pose the three failures that actually happen and reason them through.

**"A hot key expires under load — stampede?"** This is the ShopFast incident. Walk it: the key expires, concurrent readers miss. *Without* defense, all of them query the database, it saturates, feedback loop, outage. *With* the design we built — per-process single-flight collapses each instance's misses to one rebuild (database sees ~20 queries, not 1,500), jittered TTLs ensure the hot key's neighbors did not all expire alongside it, and refresh-ahead on the very hottest keys means they were refreshed *before* expiring and never missed at all. The stampede is contained at three independent layers, so even if one fails (say refresh-ahead missed a newly-hot key), single-flight still catches it. Defense in depth is the point: no single mechanism is trusted to be perfect.

**"Service B's data changed — how stale is A's cache?"** This is the cross-service question, and the answer is now bounded and stated, not hand-waved. *In the normal case*, catalog emits `ProductUpdated`, the order service's consumer evicts within ~65ms, and A's cache is fresh within a tenth of a second. *If the event is lost* — bus hiccup, consumer down, bug — the TTL backstop caps the staleness at the TTL (say 300 seconds), after which the entry self-heals on the next read. So the honest answer is "fresh within ~65ms normally, stale for at most 300 seconds in the worst case, never stale forever." That is a sentence you can put in a design doc and defend, which is the whole goal. The combination — and *only* the combination — gives you a bounded worst case.

**"The cache node holding the hot key dies."** Redis nodes die; plan for it. If the hot key lived only on the dead node, every read for it now misses — which, if the key was hot, is itself a stampede risk, so single-flight and the database's headroom (which the 95% hit rate bought you) are what save you here too. With Redis Cluster and replicas, a replica is promoted and the data survives the failover (a brief blip, not a loss); without replicas, the key is simply gone and rebuilt lazily. The deeper lesson: the cache is an *optimization*, and a correct system must survive its complete loss — slower, but correct. This is exactly why cache-aside writes the database first and falls through to the database on a cache miss: pull the entire cache and the system still serves correct data, just at database speed. If your system *breaks* when the cache is empty (because, say, the database genuinely cannot serve the un-cached load), then you do not have a cache, you have a load-bearing dependency masquerading as one — a far more dangerous thing, and a sign you have under-provisioned the source of truth.

## Case studies

**Facebook's memcache, leases, and the thundering herd.** Facebook runs one of the largest memcached deployments on earth, and their 2013 paper *Scaling Memcache at Facebook* is the canonical engineering account of caching at scale. The most-cited idea from it is the **lease**, their solution to exactly the stale-set race and stampede we covered. When a client misses a key, memcached hands it a *lease* — a token — and tells concurrent missers to wait briefly rather than all stampeding the database; only the lease-holder rebuilds. The lease also stamps the value so that a stale in-flight read cannot overwrite a fresher write (the stale-set race). It is single-flight and check-and-set, built into the cache server itself, at a scale where even one stampede per hot key would be catastrophic. The paper also documents their handling of cross-region cache invalidation and the trade-offs of caching at the edge — required reading, and it is in the further-reading list. The lesson: the stampede and stale-set problems are so fundamental that the largest cache operator built the fixes into the infrastructure.

**A CDN/edge-cache win.** The single highest-leverage caching decision most product teams make is putting static and semi-static assets behind a CDN with long, content-hashed TTLs. The pattern is universal across large sites: image, CSS, and JS assets are served under URLs that include a content hash, with `Cache-Control: max-age=31536000, immutable`, so the CDN and every browser cache them effectively forever and the origin serves them once. When the asset changes, the *URL* changes (new hash), so there is no invalidation problem at all — it is the versioned-key trick applied to the edge. A site that moves its assets to this scheme typically drops origin bandwidth and request load dramatically and improves global latency for free, because the bytes are served from a PoP near the user. It is the cheapest, lowest-risk caching win available, and it should be the first thing any team does.

**A stale-cache bug story.** The most instructive cache failures are not outages but *correctness* incidents, and the archetypal one goes like this: a team caches a piece of authorization or pricing data with a generous TTL and no invalidation, reasoning "it rarely changes." Then it changes at exactly the wrong moment — a price is corrected, a permission is revoked, a feature flag is flipped — and for the length of the TTL the system serves the *old* value, confidently and fast. Users see the wrong price; a revoked user retains access; a disabled feature stays on. The bug is invisible in testing (the cache is cold in tests) and invisible in monitoring (latency is great, error rate is zero — the system is *lying*, not erroring). The fix is always the same realization: that data needed event-driven invalidation, or should not have been cached at all because its staleness corrupted a decision. The lesson generalizes to the senior rule of this whole post — a cache that returns stale data fast is sometimes worse than a slow correct one, because nothing alerts you that it is wrong.

**Netflix and tiered/near caching (EVCache).** Netflix operates a massive distributed caching tier (publicly described as EVCache, built on memcached) to shield its services and databases from an enormous read load, and a recurring theme in their engineering writing is *tiering* and resilience: local in-process caches in front of the shared tier to cut latency and shield hot data, and a design where the cache layer can lose nodes or even whole zones and degrade gracefully rather than fail. The concrete lesson for the rest of us is the two-tier pattern we built — L1 local plus L2 shared — and the discipline that the cache is an optimization the system survives losing, not a dependency it cannot live without.

## When to reach for caching (and when it is a footgun)

Caching is one of the most reliably positive tools in the box, but it is not free, and a senior names the cost before reaching for it.

**Reach for caching when** reads dominate writes (the higher the read:write ratio, the more a cache pays off), the same data is read repeatedly (there is reuse to exploit), the data tolerates *some* staleness (almost all display data does), and the source of the read is slow or expensive (a database under load, a service across the network, a third-party API you pay per call). A product catalog, a user profile, reference data, rendered fragments, expensive computed results, third-party API responses — all textbook caching wins. The CDN-for-static-assets case is so lopsidedly positive it is closer to mandatory than optional.

**Caching is a footgun when** the data must be exact at the moment of action (balances, inventory decrements, authorization on revocation-sensitive operations) — here a cache is a correctness bug, not an optimization. It is a footgun when data changes on nearly every read (you pay the cache's cost and get no reuse). It is a footgun when the key space is so fragmented that nothing is read twice (a near-zero hit rate is pure overhead). And it is a footgun when you cache cross-service data with *no* invalidation story — just a TTL you picked by feel — and then act on the result; that is how the stale-cache bug above happens. The honest senior position: cache aggressively for display and read-heavy reference data; cache cautiously and with explicit invalidation for anything that drives a decision; and never cache the value you are about to make an exact, irreversible decision against.

A note on premature caching, in the spirit of the performance post: do not cache to fix a slowness you have not measured. Caching adds a stale-data surface and operational weight; if a query is fast and the database is idle, a cache there is complexity with no payoff. Measure first, find the hot path, cache *that*. The cache is a precision tool aimed at a measured bottleneck, not a blanket you throw over the whole system.

## Key takeaways

1. **Caching buys three things at once — latency, load reduction on the owning service and its database, and cost.** It is the highest-leverage performance lever in a microservices system, and usually the first one a senior pulls.
2. **The metric that matters is the *miss* rate, not the hit rate**, because misses are what you pay for. Going from 95% to 99% (five-fold fewer misses) is often worth more than going from 60% to 70%.
3. **Cache-aside is the default**; reach for write-through when a brief stale window is unacceptable, write-behind only for loss-tolerant high-volume writes, and refresh-ahead for a small set of very hot keys.
4. **Invalidate (delete) on write, do not update on write, and write the database before the cache** — both rules defend against concurrency races that make the cache lie.
5. **Cross-service invalidation is the genuinely hard part**, because data is owned by another service. The senior default is **event-driven eviction plus a bounded TTL backstop**: events for sub-100ms freshness, TTL so a lost event self-heals instead of lying forever.
6. **The cache stampede is the most common cache outage.** Defend in depth: single-flight to coalesce concurrent misses, jittered TTLs so expiries never synchronize, refresh-ahead so the hottest keys never miss.
7. **A cache is a deliberately stale replica; staleness is a business-correctness decision, not a technical one.** Never cache data whose staleness corrupts a decision — show a balance from cache, never debit against one.
8. **Hot keys overwhelm one cache node regardless of consistent hashing**; tier a small short-TTL local L1 in front of the shared L2 to spread the hot load across the fleet.
9. **The system must survive the cache being completely empty or gone** — slower but correct. If it cannot, you have a load-bearing dependency, not a cache, and an under-provisioned source of truth.
10. **Caching is easy; invalidation across a service boundary is the hard part.** That one sentence is the whole post, and the line between a junior who adds a `redis.get` and a senior who designs a coherent, bounded-staleness, stampede-proof caching strategy.

## Further reading

- Rajesh Nishtala et al., *Scaling Memcache at Facebook* (NSDI 2013) — the canonical paper on caching at extreme scale, leases, the thundering-herd and stale-set fixes, and cross-region invalidation.
- Sam Newman, *Building Microservices* (2nd ed.) — the chapters on resilience and inter-service communication frame why borrowed-data caching is a coupling and consistency decision, not just a speed one.
- Chris Richardson, *Microservices Patterns* — the data-management patterns that determine what you can and cannot safely cache across a service boundary.
- The HTTP caching specification (RFC 9111) and the `Cache-Control` / `stale-while-revalidate` directives — the most under-used caching tool in microservices lives in your response headers.
- [Performance and cost optimization in microservices](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices) — the broader frame; the cheapest request is the one you never make.
- [Data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) — a cache is one of the most common sources of eventual consistency; the discipline there is the discipline here.
- [Event-driven microservices: choreography vs orchestration](/blog/software-development/microservices/event-driven-microservices-choreography-vs-orchestration) and [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — how the owning service reliably emits the change events that drive cross-service invalidation.
- [Consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) — how keys map to cache nodes, and why it distributes keys but not the load of a single hot key.
- [Rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding) — single-flight is backpressure applied to your own database; the same family of overload defenses.
- [Multi-region microservices and data locality](/blog/software-development/microservices/multi-region-microservices-and-data-locality) — where caching stops being an optimization and becomes a correctness-and-cost problem spread across continents.

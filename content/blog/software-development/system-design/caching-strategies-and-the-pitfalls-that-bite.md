---
title: "Caching Strategies and the Pitfalls That Bite: From Cache-Aside to Multi-Tier"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn to choose a caching strategy from the workload, do the hit-rate math that defends it, and survive the stampedes, avalanches, and invalidation races that turn a cache from your best optimization into a 3 a.m. outage."
tags:
  [
    "system-design",
    "caching",
    "redis",
    "cdn",
    "performance",
    "cache-invalidation",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-1.webp"
---

There is a particular kind of outage that only happens to systems that are *working well*. Your service is fast. Your dashboards are green. Your database, which would fall over instantly under raw traffic, sits at fifteen percent CPU because a cache is absorbing ninety-five percent of the reads. Everyone is happy. Then, at 3 a.m., a cache node reboots, or a popular key expires, or a deploy warms a million keys with the same time-to-live, and in the space of a single second the cache stops answering and the full, un-throttled weight of production traffic lands on a database that has not seen that load in months. It does not gracefully degrade. It falls over, and every retry makes it worse, and the thing that was making you fast just made you down.

Caching is the highest-leverage optimization in systems engineering and simultaneously the richest source of subtle, intermittent, hard-to-reproduce bugs. The leverage is obvious: a cache hit is one to three orders of magnitude faster than the datastore behind it, and a good cache means you serve ten times the traffic on the same database. The danger is less obvious, because a cache adds a *second copy of your data* with its own consistency, its own failure modes, and its own ways of lying to you. Phil Karlton's old joke — that there are only two hard things in computer science, cache invalidation and naming things — is funny precisely because every senior engineer has the scar tissue to know the first half is not a joke at all.

This post is the architect's decision layer on caching. The mechanism deep-dives elsewhere on this blog cover how the storage underneath behaves — how [B-trees power the database indexes](/blog/software-development/database/b-trees-how-database-indexes-work) your cache is protecting, and how a database buffer pool is itself a cache. My job here is the layer on top: which caching *strategy* a senior reaches for given a workload, the hit-rate math that lets you defend or kill a design in a review, and the named pitfalls — stampede, penetration, avalanche, the invalidation race — that bite hard enough to take down a system, along with the specific fixes for each. Figure 1 is where we start: the multi-tier latency ladder that explains *why* caching has such absurd leverage in the first place.

By the end you should be able to do four concrete things: pick a caching strategy from the read/write shape of a workload rather than from habit; compute how a change in hit rate changes database load, and use that number to decide whether the next percent of hit rate is worth the engineering; design the single-flight fix for a cache stampede with real numbers attached; and write a write-path that does not leave stale data in the cache after a race between a reader and a writer. We will stress-test every design the way an incident does — a cache node dies cold, a key goes viral, a million keys expire at once — because the difference between a junior's cache and a senior's cache is entirely in what happens when things go wrong.

## 1. Why caching has absurd leverage: the latency ladder

A cache is worth understanding through one number: the ratio between how long it takes to answer from the cache versus from the source of truth. That ratio is rarely two or three. It is routinely ten, a hundred, a thousand. Reading a value from a local in-process hash map is a few hundred nanoseconds. Reading it from a Redis instance one network hop away is a few hundred microseconds. Reading it from a database that has to parse SQL, plan a query, walk a B-tree index, and possibly fetch a page from disk is single-digit-to-tens of milliseconds. Serving it from a CDN edge node thirty kilometers from the user instead of from your origin across an ocean can be the difference between ten milliseconds and three hundred. Each rung you climb down the ladder is a roughly tenfold latency cut, and because it is *down*, it also removes load from every rung above it.

![A six-layer stack showing browser cache, CDN edge, reverse proxy, in-process cache, Redis, and the database with their typical latencies](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-1.webp)

Figure 1 shows the ladder a request can be answered at. Read it top to bottom as a latency budget. The browser cache answers in zero milliseconds because the bytes never leave the device. The CDN or edge answers in roughly ten milliseconds because it is geographically near the user, even though it is a network call. The reverse proxy (an nginx or Varnish in front of your service) answers in about a millisecond because it is in your datacenter. The in-process cache — a hash map or an LRU in your service's own heap — answers in a fraction of a millisecond with no network at all. The distributed cache, Redis or Memcached, answers in around half a millisecond over a local network hop. And the database, even on a buffer-pool hit, is several milliseconds, and on a disk read, tens.

The architectural consequence is that *where* you cache matters as much as *how*. A value cached in the browser saves the entire round trip and every layer behind it. A value cached at the CDN saves your origin entirely for that request. A value cached in-process saves the network call to Redis. The senior instinct is to cache as close to the consumer as the data's freshness requirements allow — push static assets to the CDN, push user-specific-but-stable data to the edge or reverse proxy, push hot shared data to in-process and Redis tiers. Every layer you can answer at is a layer of load you remove from everything beneath it. This is why a single design can stack four or five cache tiers: they are not redundant, they each protect a different cost.

But every tier you add is a second copy of the truth, and that is the entire tension of this post. The ladder gives you latency for free; it charges you in consistency. A value that lives in six places can be stale in six places, and the freshness of the slowest-to-invalidate copy is the freshness your users actually see. We will spend most of this post on that bill. First, though, the strategies that govern how a single tier interacts with its source of truth.

## 2. The five strategies, and what each one trades

When people say "caching strategy" they usually mean one specific decision: when a value is read or written, what is the relationship between the cache and the database underneath? There are five canonical patterns, and each one makes a different trade between read latency, write latency, consistency, and the risk of losing data. Figure 2 lays them out as a matrix, and the rest of this section walks each row, because picking the wrong one is the difference between a clean design and one that loses writes in a power failure.

![A matrix comparing cache-aside, read-through, write-through, write-behind, and refresh-ahead across read latency, write latency, consistency, and durability risk](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-2.webp)

**Cache-aside (lazy loading)** is the default, the one you will use most. The application code owns the cache: on a read, it checks the cache first, and on a miss, it reads the database, writes the value back into the cache, and returns it. On a write, it updates the database and *invalidates* (deletes) the cache entry. The cache is a passive store the app talks to alongside the database. Its virtue is that only requested data ever gets cached — you never warm a key nobody reads — and the cache failing is survivable, because the app just reads the database directly. Its cost is the cold miss: the first read of any key always pays the full database latency, and there is a stale window between a database write and the next read repopulating the cache. This is the strategy in Figure 3, and we will return to it constantly.

**Read-through** is cache-aside with the read logic moved *into* the cache layer. The application asks the cache for a key; if the cache does not have it, the cache itself loads it from the database and stores it, transparently. To the app it looks like the data is always in the cache. The trade versus cache-aside is purely about *where the logic lives*: read-through centralizes it (often in a caching library or a sidecar), which keeps application code clean and consistent but couples the cache to the database schema. The latency and consistency profile is identical to cache-aside — same cold-miss penalty, same stale window. You reach for read-through when you want the loading logic in one place rather than scattered across every call site.

**Write-through** changes the write path: every write goes to the cache *and* the database synchronously, in the same operation, before the write is acknowledged. The cache is therefore never stale relative to the last completed write, and reads are always warm because the data was written into the cache. The price is write latency: every write now pays for two stores instead of one, and the write is not done until both succeed. You reach for write-through when reads vastly outnumber writes and you cannot tolerate the cold-miss-on-recently-written problem — a user who updates their profile and immediately reloads should not see the old value. The durability risk is zero, because the database write is part of the synchronous path.

**Write-behind (write-back)** is write-through's dangerous, fast cousin. The write goes to the cache and is acknowledged *immediately*, and the cache flushes to the database asynchronously, often in batches. Writes are extremely fast because the client never waits for the database. The catch is enormous: if the cache node dies before it flushes, those writes are *gone* — they were acknowledged to the client but never persisted. This is acceptable for data you can afford to lose (a view counter, a last-seen timestamp, an analytics event) and catastrophic for data you cannot (a payment, an order). Write-behind also lets the cache become the bottleneck for write batching and coalescing, which can be a feature: a thousand increments to a counter can collapse into one database write. Reach for it only when you have explicitly decided the data is loss-tolerant.

**Refresh-ahead** attacks the cold-miss problem from the other side. Instead of waiting for a key to expire and then reloading it on the next request (which pays the miss penalty), the cache proactively refreshes hot keys *before* they expire — say, when a key with a sixty-second TTL is accessed and has fifteen seconds left, it kicks off an asynchronous reload. Done well, a frequently-read key is *never* served stale and *never* causes a cold miss, because it is always refreshed in the background just before it would have expired. The cost is wasted refreshes on keys that would not have been read again, and complexity. Refresh-ahead pairs beautifully with the stampede fix we will design later, because it is essentially a controlled, predictive version of the same idea.

The matrix is the senior's cheat sheet, but the deeper point is that these are not five disjoint choices. Real systems combine them: cache-aside for most reads, write-through for the handful of read-your-own-writes paths that demand freshness, write-behind for the loss-tolerant counters, refresh-ahead for the dozen hottest keys. The strategy is per-data-class, not per-system.

## 3. Cache-aside in detail: the read path you will write a hundred times

Because cache-aside is the workhorse, it is worth tracing precisely, including the lines people get wrong. Figure 3 is the read path: the application checks the cache, and on a hit returns immediately; on a miss it reads the database, writes the value back into the cache with a TTL, and returns. The numbers on the figure assume a ninety-five percent hit rate, which means the database sees only the five percent of reads that miss — the entire point of the cache.

![A cache-aside read pipeline where the app checks Redis, falls through to Postgres on a miss, repopulates the cache with a TTL, and returns the value](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-3.webp)

Here is the canonical implementation, with the details that matter called out:

```python
import json, redis, time

r = redis.Redis(host="cache", socket_timeout=0.05)  # fail fast, not hang

def get_user(user_id: int) -> dict:
    key = f"user:{user_id}"
    try:
        cached = r.get(key)
        if cached is not None:
            return json.loads(cached)          # HIT: never touch the DB
    except redis.RedisError:
        pass                                   # cache down: degrade to DB, don't fail

    row = db.query_one("SELECT * FROM users WHERE id = %s", user_id)  # MISS
    value = row_to_dict(row) if row else None

    try:
        # TTL with jitter so a batch of keys does not expire together (see avalanche)
        ttl = 60 + random.randint(-10, 10)
        r.set(key, json.dumps(value), ex=ttl)  # repopulate, bounded staleness via TTL
    except redis.RedisError:
        pass                                   # cache write failing is non-fatal

    return value
```

Three lines carry the senior judgment. First, the `socket_timeout` and the `try/except` around the cache calls: a cache is a *performance optimization*, not a source of truth, so a cache that is slow or down must degrade to the database, never take down the request. A surprising number of outages are caused by a cache failure turning into a hard dependency because someone forgot the fallback. Second, the TTL with jitter: giving every key the exact same TTL is how you manufacture an avalanche, so we spread expiry over a window from the very first write. Third, caching the *null* result (`value` can be `None`): if you do not cache the absence of a key, every lookup for a non-existent user re-queries the database forever — that is cache penetration, and we will harden it properly in section 9.

The thing to internalize about cache-aside is the *ordering* on writes. On a write you have two choices: update the cache, or delete it. The senior default is **delete, not update** (invalidate, do not write). Deleting is idempotent and safe: the next read will repopulate from the database. Updating the cache from the write path opens a race — if two writers update the same key concurrently, or a writer's cache update interleaves with a reader's repopulate, you can leave a stale value in the cache that never self-corrects until the TTL expires. We will dissect that exact race in section 8. For now, the rule: **on a write, invalidate the cache entry and let the next read rebuild it.**

## 4. The optimization lens: the hit-rate math that defends a design

Caching is an optimization, and optimizations have to be measured, not asserted. The single most important number is the **hit rate**, and the single most important insight about it is counterintuitive: what loads your database is not the hit rate, it is the *miss* rate, and the miss rate behaves nonlinearly. Going from a ninety percent to a ninety-five percent hit rate does not cut database load by five percent. It cuts it in half.

![A matrix mapping hit rate to miss rate, database QPS at one hundred thousand requests per second, and relative database load](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-9.webp)

Figure 9 makes this concrete. Suppose your service handles 100,000 reads per second. At a ninety percent hit rate, ten percent miss, so the database sees 10,000 QPS. At ninety-five percent, five percent miss: 5,000 QPS — *half*. At ninety-nine percent, one percent miss: 1,000 QPS — a fifth of the ninety-five-percent number. At ninety-nine point nine percent: 100 QPS. The math is trivial — database QPS equals total QPS times the miss rate — but the consequence is profound: **the last percent of hit rate is worth more than the first ninety.** Going from ninety to ninety-five percent saved you 5,000 QPS; going from ninety-nine to ninety-nine point nine percent saved you only 900 QPS in absolute terms, but it took your database from 1,000 QPS to 100, a 10x reduction in the load it actually has to survive, and that is often the difference between one database replica and ten.

This reframes a lot of decisions. It tells you that chasing hit rate has sharply diminishing absolute returns but sharply *increasing* leverage on your database's survival, because the database is sized for the peak miss traffic. It tells you that a cache with a ninety percent hit rate is not "pretty good" — it is letting ten times more traffic through to your database than a ninety-nine percent cache. And it tells you exactly how to value an engineering effort: if raising the hit rate from ninety-five to ninety-nine percent lets you drop from a five-node database cluster to a one-node one, the dollar value of that four-percent improvement is four database nodes plus the operational burden of running them.

#### Worked example: how a hit-rate change reshapes database load

Take a product catalog service at peak: 200,000 reads per second, backed by a Postgres primary plus read replicas. Each database read costs about 4 milliseconds of database CPU time, and a single replica can sustain roughly 4,000 such reads per second before its p99 latency starts climbing past your 50 ms budget.

At a **95% hit rate**, the miss rate is 5%, so the database sees 200,000 times 0.05 equals **10,000 QPS**. At 4,000 QPS per replica, you need 10,000 divided by 4,000, rounded up — **three replicas** — and you are running them hot, near the latency cliff.

Now invest in the hit rate: better key design, longer TTLs on stable catalog data, and refresh-ahead on the top thousand SKUs. You reach a **99% hit rate**. The miss rate drops to 1%, so the database now sees 200,000 times 0.01 equals **2,000 QPS** — comfortably served by a *single* replica with headroom to spare. The four-point hit-rate improvement took you from three hot replicas to one relaxed one. At, say, \$1,500 per replica per month including operational overhead, that is \$3,000 per month and two fewer nodes to page about — bought with cache tuning, not hardware.

Push once more, to **99.5%**: 1,000 QPS, a quarter of one replica. The absolute saving shrinks (from 2,000 to 1,000 QPS is only 1,000 QPS, versus the 8,000 QPS the previous step saved), which is exactly the diminishing-return shape — but it is also the headroom that lets you survive a cache node loss without the database falling over, which is the *real* reason to keep climbing. The hit rate is not just a latency knob; it is your blast-radius budget for the day the cache misbehaves.

The measurement discipline that goes with this: instrument the hit rate *per key class*, not as a single global number. A global ninety-five percent can hide a ten-percent hit rate on one expensive query class that is quietly carrying half your database load. Track hit rate, miss rate, and the p50/p99 latency of both the hit path and the miss path separately, because the miss path's tail latency is what your users feel when the cache is not helping them.

## 5. What to cache: Pareto, working sets, and sizing

Not all data deserves a cache slot, and a cache that tries to hold everything holds nothing well. The governing reality is the **Pareto distribution**: in almost every real workload, a small fraction of the keys account for the overwhelming majority of the accesses. The top 20% of your products, users, or pages get 80% of the traffic; the top 1% often gets half. This is not a nuisance — it is the entire reason caching works. You do not need to cache your hundred-million-row table; you need to cache the few hundred thousand rows that are actually hot right now, and a cache a tiny fraction of the data's size captures almost all of the benefit.

This is what makes cache *sizing* a tractable problem instead of an impossible one. You do not size for the whole dataset; you size for the **working set** — the set of keys accessed within a time window comparable to your TTL. If your hot set is 500,000 user objects averaging 2 KB each, your working set is about 1 GB, and a 2–4 GB cache holds it with room for the long tail and for overhead. The question is never "how big is my data," it is "how big is the set of things accessed often enough to be worth keeping warm."

#### Worked example: sizing a cache for a working set

A social feed service caches rendered feed pages. Analytics show 5 million daily active users, but within any 60-second window — the TTL on a feed page — only about 400,000 users are active. Each cached feed page is roughly 8 KB after compression.

The working set is therefore 400,000 active users times 8 KB equals **3.2 GB** of hot data. Cache overhead (Redis object headers, the hash table, fragmentation) realistically adds 25–30%, so budget about **4.2 GB** of usable memory for the working set itself. You want headroom above the working set so eviction does not thrash the hot keys when traffic spikes, so size the instance at roughly 1.5x the working set: call it **6–8 GB**.

Now sanity-check the hit rate this buys. If 400,000 distinct pages fit comfortably and the access pattern is Pareto — the top 20% of those users (80,000 pages, ~640 MB) generate 80% of the reads — then even a much smaller cache would catch most of the traffic, but the full 6–8 GB cache catches essentially all of the 60-second working set, putting you in the **99%+ hit-rate** regime from the previous example. If instead you under-provisioned at, say, 1 GB, you would hold only the hottest ~125,000 pages, the working set would not fit, and the keys evicted just before they are re-read would each cause a miss — pushing the hit rate down into the low 90s and, per section 4's math, multiplying your database load several-fold. **Under-sizing a cache below its working set does not degrade gracefully; it falls off a cliff** as the eviction rate crosses the re-access rate.

The practical sizing loop: estimate the working set from access logs, provision ~1.5x it, then *measure the eviction rate*. A healthy cache evicts mostly cold keys (the long tail) and rarely evicts a key that is about to be read again. If your eviction rate is high and your hit rate is dropping together, your working set does not fit and you either add memory or shrink what you cache. If your eviction rate is near zero, you may be over-provisioned and can reclaim memory. For the partitioning of a cache that has outgrown a single node, the same [consistent-hashing techniques that partition a datastore](/blog/software-development/database/consistent-hashing-and-data-partitioning) apply directly — a cache cluster is a partitioned key-value store, and consistent hashing is what keeps a node addition from invalidating every key at once.

## 6. Eviction policies: LRU, LFU, ARC, and TTL

Because a cache is finite and the data is not, the cache must *evict* — decide which key to drop to make room for a new one. The eviction policy is the cache's guess about which keys will be read again, and getting it wrong wastes the cache on keys nobody wants.

**LRU (Least Recently Used)** evicts the key that has gone the longest without an access. It is the default in Redis and most caches because it is cheap to approximate and it matches the common pattern that recently-used things tend to be used again. Its weakness is the *scan*: one large sequential pass over many cold keys (a batch job iterating the whole table) can flush the entire hot set out of an LRU cache, because every cold key it touches becomes "recently used" and evicts a genuinely hot key. If you have batch scans hitting the same cache as your live traffic, LRU will betray you.

**LFU (Least Frequently Used)** evicts the key accessed *least often*, tracking a frequency counter per key. It is more resistant to scans — a key touched once by a batch job has a low frequency and gets evicted before your hot keys — but it has the opposite weakness: a key that was hot yesterday and is cold today lingers because its accumulated frequency is high. Redis offers an LFU mode with a decay factor precisely to age out stale-but-historically-popular keys. Reach for LFU when your access pattern is stable over time and you are being hurt by scans.

**ARC (Adaptive Replacement Cache)** tries to get the best of both by maintaining two lists — one for recently-used, one for frequently-used — and *adaptively* shifting the balance based on which list is producing more hits. It self-tunes between recency and frequency. ARC is excellent but patent-encumbered historically and more complex, so you see it more in storage systems (ZFS) than in application caches. Know it exists and what problem it solves; you will rarely configure it by hand.

**TTL (time-to-live)** is an orthogonal policy: regardless of access pattern, a key expires after a fixed duration. TTL is not really about memory pressure — it is about *bounding staleness*. A key with a 60-second TTL is guaranteed to be at most 60 seconds stale relative to the database, even if you never invalidate it explicitly. The senior pattern is to use TTL and an eviction policy *together*: the eviction policy (LRU/LFU) manages memory pressure, and the TTL bounds how stale any cached value can be. The trap with TTL, which we hit repeatedly in the pitfalls, is uniform TTLs causing synchronized expiry — which is why the code in section 3 jitters every TTL.

The choice in one line: **LRU for general workloads, LFU when scans hurt you and access is stable, TTL always (with jitter) to bound staleness, ARC when you have the luxury and the pattern shifts.** And the deeper point: eviction is a *prediction*, and a bad prediction shows up as a low hit rate on keys you thought were cached, so always validate the policy against the actual hit rate per key class rather than trusting that the default is right for your access pattern.

## 7. Pitfall one — the cache stampede (thundering herd)

Now the pitfalls, the part that bites. A **cache stampede**, also called a thundering herd or dogpile, happens when a popular key expires or is evicted and many concurrent requests all miss at the same instant. With cache-aside, every one of those requests sees the miss, every one of them queries the database, and every one of them tries to repopulate the cache. A key being read 1,000 times per second that expires generates, in the worst case, 1,000 simultaneous identical database queries in the moment after expiry — for a value that only needed to be computed *once*.

![A before-and-after comparison showing a cache stampede of one thousand database queries versus a single-flight gate that issues one query](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-4.webp)

Figure 4 contrasts the failure and the fix. On the left, the hot key expires and a thousand requests in the same millisecond all miss, all query the database, and the database — which was sized for the *cached* load of fifty queries per second on this key — gets a thousand at once and melts. The cruelty is that the database was perfectly healthy a moment before; the cache was hiding exactly how much load this key represented, and the expiry instantly un-hid it. Worse, while the database struggles, the requests are slow, so even *more* requests pile up behind them, and if your service retries on timeout, the retries pile on too. This is a self-amplifying spiral that can take a database from healthy to unresponsive in under a second.

The primary fix is **single-flight** (request coalescing): ensure that for any given key, only *one* request at a time is allowed to recompute it, and all the other concurrent requests wait for that one and share its result. Here is the pattern in Go, where the `singleflight` package makes it idiomatic:

```go
import "golang.org/x/sync/singleflight"

var group singleflight.Group

func GetUser(ctx context.Context, id string) (*User, error) {
    if u, ok := cache.Get(id); ok {
        return u, nil                       // fast path: cache hit
    }
    // Only ONE goroutine per key executes the loader; the rest block and
    // receive the SAME result. 1000 concurrent misses -> 1 DB query.
    v, err, _ := group.Do(id, func() (interface{}, error) {
        u, err := db.LoadUser(ctx, id)      // the single in-flight DB read
        if err != nil {
            return nil, err
        }
        cache.SetWithJitter(id, u, 60*time.Second)
        return u, nil
    })
    if err != nil {
        return nil, err
    }
    return v.(*User), nil
}
```

The `group.Do(id, ...)` call is the whole trick: if a thousand goroutines call it with the same `id` while the loader is running, exactly one executes the loader and the other 999 block on it and receive its result. One database query, shared a thousand ways. Across multiple service instances, you extend the same idea with a distributed lock in the cache itself — `SET key:lock value NX EX 5` to claim the right to rebuild, with everyone else briefly waiting or serving slightly stale data while the lock holder rebuilds.

A second, more elegant fix is **probabilistic early expiration** (sometimes called XFetch). Instead of every request treating an unexpired key as fresh and the expired key as a hard miss, each request, as the key approaches expiry, *probabilistically* decides to refresh it early — with a probability that rises as the expiry nears. The math (from the XFetch paper) ties the probability to the time the value took to compute and a tunable factor, so expensive-to-compute keys get refreshed earlier and more eagerly. The effect is that the refresh of a hot key is spread across a window *before* it expires, by a single request that volunteers, while everyone else keeps using the still-valid cached value. No key ever experiences a hard synchronized miss, because it gets refreshed before it can.

#### Worked example: sizing the single-flight fix for a stampede

A pricing endpoint computes a complex quote that takes the database 200 ms to produce. The result is cached with a 30-second TTL. At peak the endpoint serves 2,000 requests per second for the *same* hot quote.

**Without coalescing:** every 30 seconds the key expires, and in the ~200 ms window while the first miss is recomputing, all the requests that arrive miss too. At 2,000 RPS, a 200 ms window means 2,000 times 0.2 equals **400 concurrent database queries** fire for one value, each taking 200 ms of database work. That is 400 times 200 ms equals 80 seconds of database CPU burned in a 200 ms wall-clock window — a 400x load spike on this key, every 30 seconds, like clockwork. The database p99 spikes on every cycle, and the pricing endpoint's own p99 spikes with it.

**With single-flight:** when the key expires, the *first* request acquires the in-flight slot and runs the one 200 ms query; the other ~399 requests that arrive during that window block on it and all receive the same result. Database queries for this key in the spike window: **1**, not 400. Database CPU: 200 ms, not 80 seconds. The 399 waiting requests pay an extra ~200 ms of latency once every 30 seconds (a tiny, bounded tail), and the database never sees the spike at all. Layering probabilistic early expiry on top removes even that: the key gets refreshed a few seconds *before* expiry by one volunteer request, so no request ever blocks on a hard miss — the worst-case tail latency goes from "200 ms every 30 s" to "essentially never."

The senior takeaway: a stampede is not a database-sizing problem, it is a *concurrency-on-miss* problem. You do not fix it by buying a bigger database to survive the 400x spike; you fix it by making the spike not happen, with coalescing and early refresh. Sizing for the spike would mean sizing for 400x your steady load, which is absurd.

## 8. Pitfall two — the invalidation race and stale reads

Cache invalidation earned its place in Karlton's joke because of a race that is invisible in single-threaded reasoning and brutal in production. The setup is innocent: a reader misses, reads the database, and is about to repopulate the cache; concurrently, a writer updates the database and invalidates the cache. If the operations interleave in the wrong order, the cache ends up holding the *old* value indefinitely — until the TTL saves you, which might be minutes.

![A graph showing a slow reader's stale repopulate landing after a writer's database update and cache delete, leaving the cache holding the old value](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-7.webp)

Figure 7 traces the exact interleaving. The reader misses and reads the database, getting the old value (say, `1`). Before the reader writes that `1` into the cache, a writer runs: it updates the database to the new value (`2`) and deletes the cache key. *Then* the reader, still holding its stale `1`, completes its repopulate and writes `1` into the cache. Now the database says `2` and the cache says `1`, and nothing will fix it until the TTL expires. Every read in between gets the wrong answer. This is not a rare race; on a hot key with concurrent reads and writes it happens routinely, and it is maddening to debug because the data is *correct* in the database and *wrong* in the cache with no obvious cause.

There are several defenses, in increasing order of robustness. The cheapest is the one you already have: a **short TTL** bounds the damage — the stale value self-heals when the TTL expires, so a 60-second TTL means at most 60 seconds of staleness even if the race fires. For many systems that is genuinely good enough, and you should not over-engineer past it.

When you need tighter consistency, the next tool is **delete-after-write with a delay** (sometimes "delayed double delete"): the writer invalidates the cache, writes the database, and then schedules a *second* invalidation a short time later (say, after the maximum expected read duration). The second delete catches any stale repopulate that landed in between. It is a hack, but a widely-used one, and it closes most of the window.

The robust fix is **versioning** or compare-and-set on the cache write. Tag each value with a version or timestamp, and on repopulate only write if your version is newer than what is already cached — so the reader's stale `1` (version 5) will not overwrite a fresher invalidation or a newer write (version 6). Redis supports this through Lua scripts or `WATCH`/`MULTI` transactions that check-and-set atomically. The cleanest version of all is to stop having the application write the cache from the read path at all and instead drive cache invalidation from the database's change stream — [change data capture and the outbox pattern](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) let the cache be invalidated by the *committed* sequence of database changes, which is totally ordered and race-free by construction. The cache becomes a downstream consumer of the database's write log, and the ordering problem disappears because the log already ordered the writes.

The decision among these is a consistency call, and it connects directly to [the consistency models a system can promise](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects). A cache makes a system *less* consistent by adding a second copy with its own update timing; the question is how much inconsistency your product can tolerate. For a product catalog, sixty seconds of staleness on a price is invisible. For a user's account balance, even one stale read is a bug. Match the invalidation rigor to the data's actual freshness requirement, and do not pay for linearizability on data nobody would notice was a minute old.

## 9. Pitfall three — cache penetration (the missing-key hammer)

The third pitfall attacks from a direction caches are not built to defend: keys that *do not exist*. Cache-aside caches values it reads from the database. But if a key has no value in the database, the read returns nothing, there is nothing to cache, and so *every* request for that non-existent key misses the cache and hits the database. An attacker — or a buggy client, or a scraper — that requests a stream of random non-existent IDs can therefore bypass your cache entirely and pound the database directly, because the cache never holds anything for keys that do not exist. This is **cache penetration**, and it turns your carefully-protected database back into a directly-exposed one.

![A before-and-after comparison showing repeated database queries for a non-existent key versus a bloom filter rejecting it in microseconds](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-8.webp)

Figure 8 shows the problem and the two fixes. The first and simplest is **negative caching**: cache the *absence* of a key. When a database read returns nothing, store a tombstone — an empty or sentinel value — in the cache with a short TTL, so the next request for that missing key gets the cached "not found" and never reaches the database. This is the `value can be None` line from the code in section 3, and it is the first defense you should always have. The short TTL matters because a key that genuinely gets created later should not be permanently shadowed by a stale "not found"; a 30-second negative TTL bounds that.

Negative caching has a weakness, though: if the attacker requests *random* non-existent keys (each one different), each unique miss still hits the database once before its tombstone is cached, and a high-cardinality stream of random keys fills your cache with tombstones while still leaking one query per unique key to the database. For that, you reach for a **bloom filter**: a compact probabilistic set of *all the keys that exist*. Before querying the database for a key, you check the bloom filter; if it says "definitely not present," you reject the request in about a microsecond without touching cache or database. A bloom filter never has false negatives (if a key exists, the filter always says so), and its small false-positive rate (it occasionally says a non-existent key "might exist") only costs you a wasted database lookup that returns nothing — exactly what you would have done anyway. A bloom filter for, say, 100 million existing keys at a 1% false-positive rate fits in about 120 MB of memory, which is trivial, and it absorbs an unbounded stream of random non-existent keys without any of them reaching the database.

The combined defense is both: a bloom filter to reject keys that provably do not exist, and negative caching to handle the rare false positives and the legitimately-missing keys cheaply. The senior framing: penetration is the failure to cache *misses*, and the fix is to make a miss for a non-existent key as cheap as a hit. A cache that only protects you on the happy path of existing keys is a cache an adversary can route around.

## 10. Pitfall four — the cache avalanche (mass simultaneous expiry)

The fourth pitfall is the one from the opening paragraph, and it is the most violent. A **cache avalanche** is a stampede at the scale of the whole cache: a large set of keys expires (or a whole cache node dies) at the *same moment*, so the cache's hit rate collapses from ninety-nine percent to near zero in an instant, and the full traffic that the cache was absorbing lands on the database all at once. Where a stampede is one hot key hammering the database, an avalanche is the entire keyspace hammering it together.

![A timeline showing fifty thousand keys warmed by a deploy with identical TTLs expiring simultaneously and driving database load up twentyfold](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-5.webp)

Figure 5 traces the most common cause: synchronized TTLs. A deploy or a cache warm-up loads fifty thousand keys into the cache, all in the same minute, all with the same 3,600-second TTL. For an hour everything is fine — the cache serves a ninety-nine percent hit rate. Then, exactly 3,600 seconds after the warm-up, all fifty thousand keys expire *within the same second*, because they were created within the same second with the same TTL. The hit rate craters, every request misses, and the database that was seeing a few hundred QPS suddenly sees twenty times that. If the database cannot absorb the spike — and it usually cannot, because it was sized for the cached load — it saturates, latency spikes, requests time out, retries pile on, and you have a full outage manufactured entirely by the cache helping too uniformly.

The fixes attack each cause. For **synchronized TTLs**, the fix is **TTL jitter**: never give a batch of keys the exact same TTL; add a random spread, so instead of 3,600 seconds every key gets 3,600 plus-or-minus 600 seconds. Now the fifty thousand expirations are spread across a twenty-minute window instead of one second, the database sees a gentle, continuous trickle of misses instead of one cliff, and the hit rate degrades by a fraction of a percent rather than to zero. This is the single most important line in the section-3 code, and it costs one call to a random number generator.

For a **cache node dying** (which expires everything on that node at once), the fixes are structural: spread the cache across multiple nodes with [consistent hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) so losing one node loses only its fraction of the keys, not all of them; replicate the cache so a node's keys survive its failure; and — the most important one — protect the database with the same single-flight coalescing from section 7, so that even during a mass miss, the database sees one query per key rather than one per request. A coalesced cache during an avalanche still hits the database hard, but with a number of queries bounded by the number of *distinct keys*, not the number of *requests*, which can be a hundredfold difference.

For the **cold start** — a fresh cache node with nothing in it, or a whole cache cluster restarting — the avalanche is total, because the hit rate starts at zero. The defenses are to **warm the cache before taking traffic** (replay the hot keys into a new node before adding it to rotation), to **rate-limit the miss path** so a cold cache cannot send unbounded traffic to the database, and to bring a restarted cache back into service *gradually* rather than instantly routing full traffic to an empty cache. A cold cache node added to a hot cluster without warming is a self-inflicted avalanche, and it is a depressingly common cause of outages during routine maintenance.

## 11. Pitfall five — hot keys (when one key is too popular)

The fifth pitfall is subtle because it is a *distribution* problem, not a *whole-system* problem. A **hot key** is a single key so much more popular than the rest that the one cache node or shard holding it becomes a bottleneck while the rest of the cluster sits idle. A celebrity's profile, a viral post, a global configuration value, the homepage feed during a flash sale — one key can attract a hundred thousand requests per second, all routed by consistent hashing to the *same* Redis node, which simply cannot serve that volume even though the cluster as a whole has plenty of capacity. Your cache is "at thirty percent CPU" on average and one node is at a hundred percent, dropping requests.

The defenses are about *spreading the load for that one key*. The most common is **client-side / in-process caching of the hot keys**: replicate the hottest keys into every application instance's local heap, so the hundred thousand requests per second are served from a hundred different application servers' local memory and never reach the single Redis node at all. This is the highest-leverage fix, because it turns one hot node into N warm heaps, but it costs consistency — a value cached in a hundred heaps takes the TTL to propagate an update everywhere, so it is only safe for data that tolerates a few seconds of staleness across instances.

A second fix is **key replication / fan-out**: store the hot value under several keys (`config:0` through `config:9`) hashed to different nodes, and have clients read a random one of the N copies, spreading the read load across N nodes. It costs you N-fold the writes to keep the copies in sync, which is fine for a read-heavy hot key. A third is simply detecting hot keys and giving them a dedicated, beefier path. The thing a senior does *first*, though, is **measure per-key request rate** — most caching incidents blamed on "Redis being slow" are actually one hot key saturating one node, invisible in the cluster's average metrics. You cannot fix a hot key you cannot see, and the average CPU across the cluster is precisely the metric that hides it.

## 12. Choosing the strategy: a decision tree

We have the strategies and the pitfalls; now the decision. Figure 6 routes a workload to a strategy based on a few questions about its read/write shape and its tolerance for staleness — the questions a senior asks in a design review before reaching for a pattern out of habit.

![A decision tree routing a workload to cache-aside, refresh-ahead, write-through, or write-behind based on read-write ratio and staleness tolerance](/imgs/blogs/caching-strategies-and-the-pitfalls-that-bite-6.webp)

The first fork is the read/write ratio, because it determines whether the read path or the write path is the one worth optimizing. If **reads vastly outnumber writes** (a product catalog, a content site, a user-profile service), you optimize the read path, and the next question is staleness tolerance: if you can tolerate a bounded stale window, **cache-aside** with a sensible TTL is the simplest correct answer and where you should start; if you have a handful of very hot keys that must never cause a cold miss, layer **refresh-ahead** on top of them. If instead **writes are heavy** relative to reads, you optimize the write path, and the next question is durability: if every write must survive (orders, payments, anything financial), **write-through** keeps the cache consistent without risking data loss; only if the data is explicitly loss-tolerant (counters, metrics, last-seen timestamps) do you reach for **write-behind** and its async-flush speed, accepting that a cache crash loses the un-flushed writes.

The deeper lesson of the tree is that **most systems are read-heavy and most read-heavy systems should start with cache-aside plus jittered TTLs**, then add the more exotic strategies only for the specific data classes that demand them. A senior does not pick write-behind because it is fast; they pick it because they have decided, explicitly and in writing, that this specific data can be lost on a cache crash. The strategy is a consequence of the data's requirements, not a preference. And the tree is per-data-class: a single service routinely uses cache-aside for most reads, write-through for the read-your-own-writes paths, and write-behind for the analytics counters, all at once.

## 13. Trade-offs: the senior's decision matrix

Everything above collapses into one decision table, the artifact you bring to a design review. Each strategy buys something and pays for it somewhere, and naming both is the whole job.

| Strategy | What you gain | What you pay | When it wins |
|---|---|---|---|
| **Cache-aside (lazy)** | Only-read data is cached; cache failure is survivable; dead simple | Cold-miss latency on first read; stale window until TTL; you own invalidation | The default for read-heavy data with tolerable staleness |
| **Read-through** | Loading logic in one place; clean app code | Couples cache to schema; same cold-miss/stale profile as cache-aside | When load logic is scattered and you want it centralized |
| **Write-through** | Cache never stale vs last write; reads always warm; zero durability risk | Every write pays two synchronous stores; higher write latency | Read-heavy with read-your-own-writes; freshness on writes matters |
| **Write-behind** | Very fast writes; coalesces bursts into batched DB writes | Data loss if cache dies before flush; weak consistency | Explicitly loss-tolerant data: counters, metrics, last-seen |
| **Refresh-ahead** | Hot keys never go cold or stale; no synchronized miss | Wasted refreshes on keys not re-read; more complexity | A small set of very hot, expensive-to-compute keys |

And the pitfalls collapse into a parallel table of failure-and-fix, which is the other half of the senior's mental checklist:

| Pitfall | What triggers it | The fix |
|---|---|---|
| **Stampede / dogpile** | A hot key expires; many concurrent misses each query the DB | Single-flight coalescing + probabilistic early expiry |
| **Invalidation race** | A reader's stale repopulate lands after a writer's invalidate | Short TTL, versioned writes, or CDC-driven invalidation |
| **Penetration** | Requests for non-existent keys bypass the cache to the DB | Negative caching + a bloom filter of existing keys |
| **Avalanche** | Mass simultaneous expiry or a node death; hit rate to zero | TTL jitter, consistent-hashed replication, coalescing, cold-start warming |
| **Hot key** | One key so popular it saturates a single node | In-process replication of hot keys; key fan-out across nodes |

The matrix discipline is the point. You never recommend write-behind without saying "and we accept losing un-flushed writes on a crash." You never add a cache without saying "and we accept up to TTL seconds of staleness." Naming the cost is not pessimism; it is the difference between a design that survives the review and the incident, and one that surprises you both.

## 14. Stress-testing the design: what breaks at 10x and at a node loss

A design is only as good as its behavior under the conditions that actually take systems down. Walk the three failure injections that an incident will run for you whether you planned for them or not.

**A cache node dies (cold-start stampede).** Suppose a six-node Redis cluster loses one node. With consistent hashing, one-sixth of the keys are now uncached, and every request for those keys misses. If the database was sized for a one-percent miss rate and the miss rate on a sixth of the keyspace jumps to a hundred percent, the database's load on those keys spikes roughly a hundredfold while a replacement node warms. *Does the design survive?* It survives if (a) consistent hashing limited the loss to one-sixth rather than everything, (b) single-flight coalescing means each distinct uncached key generates one database query rather than one per request, and (c) the replacement node is warmed gradually rather than thrown into rotation empty. It does *not* survive if a uniform-hash scheme would have remapped every key on node loss, or if there is no coalescing, in which case the database sees the full un-coalesced spike and saturates. The senior pre-mortem is exactly this walk: what fraction of keys go cold, how many database queries does that generate after coalescing, and can the database absorb that number?

**A viral hot key (10x on one key).** A single post goes viral and one key jumps from a thousand to a hundred thousand requests per second, all routed to one Redis node. *Does the design survive?* It survives if the hottest keys are replicated into in-process caches, so the hundred thousand requests per second are served from a hundred application heaps and the single Redis node sees almost none of them. It does *not* survive if every read goes to the one node, which will saturate at a fraction of a hundred thousand QPS regardless of how idle the rest of the cluster is. The stress test reveals whether your design has *any* answer for load concentrating on one key, or whether it silently assumes uniform key popularity — an assumption that is false in every system with a Pareto distribution, which is every system.

**A mass TTL expiry (avalanche).** A deploy warms a hundred thousand keys at once. *Does the design survive their simultaneous expiry?* It survives if TTLs are jittered, spreading the expiry across a window so the database sees a trickle, and if coalescing bounds the per-key query count. It does *not* survive if every key got the identical TTL, in which case the hundred thousand keys expire together and the database sees a hundredfold spike in one second. This is the cheapest failure to prevent (one random-number call per TTL) and one of the most common to ship, because in development and in the first hour after a deploy everything looks perfect — the cliff is exactly one TTL away and invisible until then.

The meta-lesson across all three: a cache's failure mode is *load concentration in time or in space*. A stampede concentrates load on one key in time; an avalanche concentrates the whole keyspace's load in time; a hot key concentrates load on one node in space; a node death concentrates a fraction of the load on the database in time. Every fix in this post is, at root, a way to *spread* that concentration — coalesce it (one query not a thousand), jitter it (a window not an instant), replicate it (N heaps not one node). A senior reading a cache design is looking for exactly one thing: where does load concentrate, and what spreads it?

## 15. Case studies: three real ways caches bite

Patterns are easier to trust when you have seen them fail in production. Three stories, each teaching one of the pitfalls above.

**Facebook's memcache leases (the stampede and the invalidation race, solved together).** Facebook's published work on scaling memcache describes one of the most-cited cache architectures in the field, serving billions of requests per second from a memcache tier in front of MySQL. Two of their innovations map directly onto this post. To solve the stampede, they introduced **leases**: when a client misses, memcache hands it a *lease token* — a short-lived right to be the one to recompute and set that key. Concurrent clients that miss the same key do not all get a lease; they are told to wait briefly and retry, by which point the lease holder has populated the value. That is single-flight implemented at the cache server. The same lease mechanism also addresses the invalidation race: a stale set carrying an outdated lease token is rejected, so a slow reader cannot overwrite a fresher value. The lesson: at extreme scale, coalescing and race-prevention are not optional add-ons, they are built into the cache protocol itself, because at billions of requests per second a hot key without coalescing is not a slow query, it is an instant outage.

**A stampede outage (the hot key that hid its own load).** A common and well-documented class of incident — seen in public post-mortems across many companies — runs like this: a single expensive computed value (a homepage, a leaderboard, a heavily-joined dashboard) is cached with a short TTL. The value is read constantly, so its TTL expires frequently, and because there is no coalescing, each expiry triggers a burst of concurrent identical recomputations against the database. For a long time it is invisible, because the database has just enough headroom to absorb each burst. Then traffic grows, or the query gets slightly more expensive, and one day a burst exceeds the database's capacity; the recomputations slow down, which *widens* the window in which new requests miss, which produces *more* concurrent recomputations — the self-amplifying spiral from section 7. The database saturates and the site goes down. The post-mortem invariably finds that the cache was masking exactly how expensive and how frequently-recomputed that one value was, and the fix is always single-flight plus early-refresh. The lesson: a cached value's *true* cost is hidden until the cache stops absorbing it, so you must coalesce *before* you are big enough to need it, because the day you need it is the day you are already down.

**A CDN cache-key bug (when the cache returns the wrong user's data).** The most frightening cache bugs are not slowness or staleness — they are *cache poisoning* and *cache-key collisions*, where the cache returns one user's data to another user. The canonical version: a CDN or reverse proxy caches a response keyed on the URL path but *not* on a header (an `Authorization` token, an `Accept-Language`, a cookie) that actually varies the response. The first user's personalized or authenticated response gets cached under the shared key, and the next user with the same path but a different identity gets served the first user's cached response. Real public incidents have leaked authenticated content this way. The mechanism is always a **cache key that does not capture everything the response depends on** — the response varies on something the key ignores. The fix is to make the cache key include every input the response varies on (the `Vary` header exists precisely for this), and to *never* cache responses that depend on per-user authentication unless the key includes the user identity. The lesson: a cache key is a *contract* that two requests with the same key get the same answer, and if that contract is false, the cache does not just slow down or go stale — it serves the wrong data to the wrong person, which is a security incident, not a performance one. Audit your cache keys as carefully as you audit your auth.

## 16. Optimization, consolidated: the levers with numbers

The optimization theme has run through every section; here it is consolidated, because in a review you want these levers at your fingertips with numbers attached.

**Chase hit rate where it changes the database's blast radius, not vanity.** Database QPS equals total QPS times miss rate, so the last percent of hit rate has the most leverage on database load — going 95% to 99% is a 5x database load cut. Measure the win as the database QPS before and after, and translate it into replica count and dollars: a 5x QPS reduction is often the difference between five database nodes and one.

**Coalesce every hot-key miss with single-flight.** This turns a stampede from "one DB query per concurrent request" (potentially hundreds) into "one DB query per key." Measure the win as concurrent-queries-per-key during an expiry event; you want it to be one. The cost is a bounded extra latency (the recompute time) on the requests that wait, once per TTL.

**Jitter every TTL.** One random-number call per write converts a synchronized avalanche (the whole keyspace expiring in one second) into a smooth trickle over a window. Measure the win as the database QPS *variance* around expiry events; a jittered cache has flat database load, an un-jittered one has periodic cliffs. This is the cheapest high-leverage line in caching.

**Cache as close to the consumer as freshness allows.** Each tier you can answer at removes load from every tier behind it; a CDN hit removes your origin entirely, an in-process hit removes the Redis hop. Measure the win as the request distribution across tiers — what fraction is served at the edge, at the proxy, in-process, at Redis, at the database — and push the boundary toward the consumer for every data class whose freshness tolerates it.

**Right-size to the working set, not the dataset, and watch the eviction rate.** The working set is the keys accessed within a TTL window, usually a tiny fraction of the data; provision ~1.5x it. Measure the eviction rate against the re-access rate — a healthy cache evicts cold keys, a sick one evicts keys it is about to re-read. Under-sizing below the working set falls off a cliff, not a slope, so size with margin.

**Negative-cache and bloom-filter the misses.** A cache that only protects existing keys is bypassable by non-existent ones. Measure the win as the database QPS for keys that return *nothing* — it should be near zero, not "one per unique missing key requested." A bloom filter for 100M keys at 1% false positives is ~120 MB, trivially worth it.

## 17. When to reach for a cache (and when not to)

**Reach for a cache when:** reads dominate writes, the same data is read repeatedly within a short window (a real working set exists), the source of truth is expensive to query relative to the cache, and the data can tolerate *some* staleness — even a few seconds is usually enough to unlock enormous leverage. This describes the vast majority of read-heavy services, which is why caching is the first optimization a senior reaches for. Start with cache-aside plus jittered TTLs plus negative caching, and add the exotic strategies and pitfall-fixes for the specific data classes that need them.

**Reach for a multi-tier cache when:** a single tier cannot capture all the leverage — push static and cacheable responses to the CDN, per-user-stable data to the reverse proxy or edge, hot shared data to in-process and Redis tiers. Each tier protects a different cost, and a well-layered cache answers most requests before they reach your origin at all. Reach for in-process caching specifically when you have hot keys whose load would saturate a single distributed-cache node and whose freshness tolerates per-instance TTL propagation.

**Do not reach for a cache when:** the data is written far more than it is read (you will spend more on invalidation than you save on reads, and the cache will mostly hold stale or cold entries), the data demands strict freshness on every read (an account balance, a lock state, an inventory count you cannot oversell), or there is no working set — if every read is for a different key that is never read again, a cache holds nothing useful and only adds a hop and a consistency problem. And never cache authenticated or per-user responses in a shared tier without putting the user identity in the cache key — that is the cache-key-collision class of security bug, and it is not worth the latency saved.

The meta-rule for the review: **a cache is a second copy of your data that trades consistency for latency, so the question is never "should we cache" but "how much staleness can this specific data tolerate, and what does that staleness cost us."** If the answer is "none," you do not want a cache on that path; you want a faster source of truth. If the answer is "seconds are fine," a cache is the highest-leverage move you can make. Most data is the second kind, which is why caches are everywhere — but the senior is the one who checks *which kind* each data class is before adding the second copy.

## Key takeaways

- **What loads your database is the miss rate, not the hit rate, and it is nonlinear.** Database QPS equals total QPS times miss rate, so 95% to 99% hit rate cuts database load fivefold and the last percent is worth more than the first ninety. Price your hit rate in replica count, not in percentages.
- **Pick the strategy from the workload, never from habit.** Cache-aside for read-heavy with tolerable staleness; write-through when writes must stay durable and fresh; write-behind *only* for explicitly loss-tolerant data; refresh-ahead for the few hottest expensive keys. The strategy is a consequence of the data's requirements.
- **On a write, invalidate the cache; do not update it.** Deleting is idempotent and race-resistant; updating from the write path opens the reader/writer interleaving that leaves stale data in the cache until the TTL saves you.
- **Single-flight every hot-key miss.** Coalescing turns a stampede from hundreds of identical queries into one. Pair it with probabilistic early expiry so a hot key is refreshed before it can ever cause a synchronized hard miss.
- **Jitter every TTL.** One random-number call per write converts a synchronized avalanche into a smooth trickle. It is the cheapest high-leverage line in all of caching, and shipping uniform TTLs is a self-inflicted outage exactly one TTL away.
- **Cache misses, not just hits.** Negative-cache the absence of keys and bloom-filter the set of keys that exist, or an adversary requesting non-existent keys routes straight around your cache to the database.
- **A hot key concentrates load in space; an avalanche concentrates it in time.** Every cache fix is a way to *spread* concentration — coalesce it, jitter it, replicate it. When you read a cache design, find where load concentrates and find what spreads it.
- **A cache key is a contract that same-key requests get the same answer.** If the key omits something the response varies on (a user, a header, a language), the cache serves the wrong data to the wrong person — a security incident, not a slow query. Audit cache keys like you audit auth.
- **Size to the working set, not the dataset, and watch the eviction rate.** The hot set is a tiny Pareto fraction of the data; provision ~1.5x it. Under-sizing below the working set falls off a cliff as eviction outpaces re-access, not down a gentle slope.
- **A cache is a second copy that trades consistency for latency.** The only question is how much staleness each data class tolerates and what it costs. If the answer is "none," you want a faster source of truth, not a cache.

## Further reading

- [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) — the index your cache is protecting, and why a buffer-pool miss is the cost a cache hit avoids.
- [Consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) — how to partition a cache cluster so losing one node loses one shard's keys, not all of them.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the consistency a cache costs you, and how change-data-capture drives race-free invalidation from the database's write log.
- [Consistency models: a practical guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) — the architect-layer companion: matching invalidation rigor to the freshness your product actually requires.
- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the working-set sizing and QPS math this post leans on, generalized to any capacity question.
- [Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) — how the reverse-proxy and edge tiers in the latency ladder route and cache, the layer in front of your service.
- Nishtala et al., "Scaling Memcache at Facebook" (NSDI 2013) — the definitive production cache architecture, including leases for stampede and race control.
- Vattani, Chierichetti, and Lowenstein, "Optimal Probabilistic Cache Stampede Prevention" (VLDB 2015) — the XFetch probabilistic-early-expiry math behind the stampede fix.
- The official Redis documentation on eviction policies, TTLs, and Lua scripting, and the Varnish and CDN `Vary`/cache-key documentation for the edge tiers.

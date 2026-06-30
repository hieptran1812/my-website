---
title: "Cache Patterns in Production: Cache-Aside, Write-Through, Write-Behind"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "The canonical caching patterns differ in who writes the cache and when — each with a distinct consistency and failure profile — so choose deliberately and get the invalidation order right."
tags: ["caching", "cache-aside", "write-through", "write-behind", "redis", "consistency", "cache-invalidation", "database-scaling", "distributed-systems", "race-conditions"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 36
---

Almost every caching outage I have been paged for traces back to the same root cause, and it is never "the cache was slow." It is that someone reached for a caching pattern without knowing which one they had chosen, wired up the write path on autopilot, and got the invalidation order backwards. The cache served a stale value, a user saw a price that no longer existed or a balance that had already been spent, and by the time the TTL expired and papered over the bug, the damage was done and nobody could reproduce it.

The frustrating part is that the patterns themselves are old, well understood, and genuinely simple. There are essentially four of them — cache-aside, read-through, write-through, and write-behind — plus a refinement (refresh-ahead) that sits on top. The trouble is that engineers treat them as interchangeable performance tricks ("we put Redis in front of the database") when they are actually four different *consistency contracts*, each with its own failure mode. The single most important thing to internalize is this: **the patterns differ in who writes the cache and when, not in how reads are served.** Get that one axis right and the rest follows.

![Where each pattern puts the write: app to four patterns to four outcomes](/imgs/blogs/cache-patterns-in-production-1.webp)

The diagram above is the mental model for the whole post: the application's write path branches into three pattern choices, and each branch lands the data in a different place at a different time. Cache-aside writes the database and then *deletes* the cache key, so the next read repopulates it. Write-through writes both the cache and the database synchronously, so they are always in lockstep. Write-behind writes only the cache, acknowledges immediately, and flushes to the database later in a background batch. Read-through (not shown as a separate branch because it shares the read shape of cache-aside) just moves the populate-on-miss logic from your application into the cache library. The rest of this article is a tour of those branches: how each one works, exactly where it betrays you, and how to choose between them with your eyes open.

## Why "we put Redis in front of the database" is not a plan

Before the patterns, the assumption-versus-reality table that every caching design review should start with. The phrase "add a cache" hides at least five separate decisions, and the defaults you fall into when you do not make them explicitly are almost always wrong for your workload.

| The common assumption | The naive view | The reality |
| --- | --- | --- |
| "A cache just makes reads faster." | Reads hit Redis, miss to the DB, done. | A cache introduces a *second copy* of the truth that can disagree with the first. You now own a consistency problem you did not have before. |
| "Caching is read-path work." | Only the read code changes. | The write path is where every correctness bug lives. The order in which you touch the cache and DB on a write decides whether you are safe or split-brained. |
| "TTL keeps it consistent." | Set TTL 60s and staleness is bounded to 60s. | TTL bounds *how long* a stale value can persist; it does nothing to prevent a stale value from being written in the first place. It is a safety net, not a correctness mechanism. |
| "Update the cache on write so it is always warm." | `SET k = new_value` whenever you write the DB. | Updating the cache in place is the single most race-prone choice you can make under concurrency. Invalidation (delete) is almost always safer than update. |
| "All the patterns are roughly equivalent." | Pick whichever the framework defaults to. | They have different write latencies, different consistency guarantees, and *categorically different* failure modes. The right one is dictated by the workload, not by taste. |

> A cache is not a faster database. It is a second, eventually-disagreeing copy of your data that you have volunteered to keep in sync by hand. Every pattern in this post is a different hand-syncing discipline.

Hold that framing. Each section below is one discipline, and the consistency-hazards section in the middle — the heart of the post — is what happens when the discipline slips.

## 1. Cache-aside (lazy loading): the default, and why

**Senior rule of thumb: cache-aside is the right default for almost every read-heavy workload, because it only populates what is actually read and it keeps the cache out of your write path's critical section.**

Cache-aside — also called lazy loading — puts the application in charge of both halves. On a read, the app asks the cache first; on a miss, it reads the database, writes the result back into the cache, and returns it. On a write, the app updates the database and then *invalidates* (deletes) the cache key so the next read reloads fresh.

![Cache-aside read path: hit returns, miss loads from DB and back-fills; write invalidates](/imgs/blogs/cache-patterns-in-production-2.webp)

The figure walks the read path: a request for key `k` does a `GET` against the cache; a hit returns in roughly 0.3 ms; a miss falls through to a `SELECT`, populates the cache with a TTL, and returns. The write path is the small red box: `UPDATE` the database, then `DEL` the key. Note what the write path does *not* do — it does not put the new value into the cache. It removes it. That asymmetry is the whole point of the pattern, and we will come back to why deletion beats update.

Here is the canonical implementation in Python with `redis-py`. Read it carefully; the comments mark the load-bearing decisions.

```python
import json
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
TTL_SECONDS = 300  ## bound staleness even when invalidation is missed

def get_user(db, user_id: int) -> dict:
    key = f"user:{user_id}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)          ## cache hit, ~0.3 ms

    ## miss: load from the source of truth
    row = db.query_one("SELECT * FROM users WHERE id = %s", (user_id,))
    if row is None:
        ## cache the negative result briefly to absorb hot misses,
        ## but with a much shorter TTL so a real insert shows up soon
        r.set(key, json.dumps(None), ex=30)
        return None

    r.set(key, json.dumps(row), ex=TTL_SECONDS)  ## back-fill on miss
    return row

def update_user(db, user_id: int, **fields) -> None:
    key = f"user:{user_id}"
    ## 1. write the source of truth FIRST
    db.execute("UPDATE users SET email = %s WHERE id = %s",
               (fields["email"], user_id))
    ## 2. THEN invalidate the cache — never the other way around
    r.delete(key)
```

Three things in that snippet are not negotiable and are the difference between a correct cache-aside and a broken one.

First, **the write order is database-then-delete.** Section 5 is dedicated to why; for now, accept it as a rule.

Second, **the write deletes, it does not set.** A natural instinct is to write `r.set(key, json.dumps(new_row))` on update so the cache stays warm. Resist it. Setting the cache on write opens a wide concurrency window (two writers can interleave their `SET`s out of order and the loser's value sticks) and it couples your write latency to a cache round-trip. Deleting is idempotent, order-insensitive between concurrent deletes, and forces the next reader to reload from the authoritative source.

Third, **the negative-result cache.** Caching `None` for a short window stops a stream of requests for a non-existent key from hammering the database — a cheap defense against one flavor of cache penetration. Keep its TTL short so a later `INSERT` is not masked.

### Second-order optimization: the miss-storm and the TTL jitter trick

The non-obvious gotcha with cache-aside is what happens when a *hot* key expires. The instant its TTL elapses, every concurrent request for that key misses simultaneously, and they all stampede the database with the same `SELECT`. This is the thundering-herd problem, and a naive cache-aside makes it worse, not better, because the database now takes a synchronized spike instead of a steady trickle.

Two cheap mitigations belong in every cache-aside deployment. The first is **TTL jitter**: never set a fixed TTL on a class of keys, because they will all expire in lockstep if they were populated together (say, after a deploy that warmed the cache). Add a random spread:

```python
import random

def ttl_with_jitter(base: int = 300, spread: float = 0.2) -> int:
    ## 300s +/- 20% so a population wave does not expire as one wave
    delta = int(base * spread)
    return base + random.randint(-delta, delta)
```

The second is request coalescing on miss — a single-flight lock so only one request rebuilds the key while the others wait. That deserves its own treatment, and it is the subject of [cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd), the companion to this post. For cache-aside specifically, the lesson is that the pattern is *correct* by default but not *stampede-safe* by default; you bolt that on.

## 2. Read-through: move the read logic into the cache layer

**Senior rule of thumb: read-through is cache-aside with the populate-on-miss logic relocated from your application into the cache library — reach for it when you want one uniform loading path instead of populate code scattered across every call site.**

Read-through and cache-aside are identical on the read path from the database's point of view: a miss loads from the DB and populates the cache. The difference is *who owns that logic*. In cache-aside, your application code does the `SELECT`-and-back-fill. In read-through, the cache library does it — you configure the cache with a loader function, and `cache.get(key)` transparently loads from the database on a miss before returning.

```python
## Read-through with a loader the cache library owns.
## (Sketch of the contract — concrete in Caffeine/Guava on the JVM,
##  or a thin wrapper around redis-py.)
class ReadThroughCache:
    def __init__(self, redis_client, loader, ttl=300):
        self.r = redis_client
        self.loader = loader          ## fn(key) -> value, hits the DB
        self.ttl = ttl

    def get(self, key):
        cached = self.r.get(key)
        if cached is not None:
            return json.loads(cached)
        value = self.loader(key)      ## the cache, not the caller, loads
        self.r.set(key, json.dumps(value), ex=self.ttl)
        return value

## the application never sees the DB on the read path
user_cache = ReadThroughCache(r, loader=lambda k: db.load_user(k))
user = user_cache.get("user:42")
```

Why bother, if it is functionally cache-aside? Because **uniformity buys correctness at scale.** When forty call sites each implement their own populate-on-miss, forty of them can get the TTL wrong, forget the negative-result cache, or — most commonly — drift out of sync when the schema changes. Centralizing the loader means there is exactly one place that knows how to turn a key into a value, and improvements (jitter, single-flight, metrics) apply everywhere at once. Many mature cache libraries (Caffeine and Guava on the JVM, Ehcache, the cache abstractions in ORMs) are read-through by design for precisely this reason.

The trade-off is rigidity. Read-through assumes a uniform "key maps to one object loaded one way" model. The moment your read needs to join across tables, fan out to several services, or vary by request context, the loader abstraction starts leaking and you are better off back in explicit cache-aside where the application controls the query. Read-through also shares cache-aside's failure modes wholesale — it is eventually consistent, TTL-bounded, and exposed to the same cold-start stampede when a popular key is missing. It does not fix consistency; it relocates the read code.

## 3. Write-through: pay on write, never serve stale

**Senior rule of thumb: write-through buys you read-after-write consistency for cached keys by making every write pay for a cache update synchronously — choose it only when stale reads are unacceptable and writes are rare relative to reads.**

Write-through inverts the lazy philosophy. Instead of populating the cache on a read miss, it populates the cache on *every write*, synchronously, as part of the same operation that updates the database. The write does not acknowledge to the client until both the database and the cache hold the new value.

![Write-through: write blocks through DB commit and cache set, then acks; cost is every write pays both](/imgs/blogs/cache-patterns-in-production-5.webp)

The pipeline shows the cost and the benefit side by side. A write request commits the database (roughly 5 ms with an `fsync`), then sets the cache (roughly 0.3 ms), then acknowledges. The payoff is the two green boxes: any later read of that key is a guaranteed hit, served warm from the cache, never stale relative to the last completed write. The price is the amber box: *every* write pays for both a database commit and a cache write, even for keys that are never read again.

```python
def update_user_write_through(db, user_id: int, **fields) -> None:
    key = f"user:{user_id}"
    new_row = {**fields, "id": user_id}

    ## 1. commit the source of truth
    db.execute(
        "UPDATE users SET email = %s WHERE id = %s",
        (fields["email"], user_id),
    )
    ## 2. update the cache as part of the same logical write.
    ##    Reads after this point are guaranteed fresh for this key.
    r.set(key, json.dumps(new_row), ex=TTL_SECONDS)
```

Notice this looks almost identical to the cache-aside *write* — except the second step is `set`, not `delete`. That single change is the entire pattern, and it carries a subtle hazard we will dissect in section 6: setting the cache on write reintroduces a concurrency window between two writers that delete-based invalidation does not have. Write-through earns its consistency claim only when writes to a single key are serialized (a per-key lock, an actor, a single-writer partition) or when the rare interleaving is acceptable.

### Second-order optimization: write-through wastes work on cold keys

The non-obvious cost of write-through is the writes you do for keys nobody reads. Imagine an audit-log table where you write a row per request but read a given row maybe once a year. Write-through caches every one of those rows on write, paying a cache `SET` and consuming cache memory for data with a near-zero hit rate. The cache fills with cold entries, evicting genuinely hot ones, and your hit rate *drops*.

The fix is to restrict write-through to key classes with a high read-to-write ratio — user profiles, product catalog entries, configuration — and leave write-once-read-rarely data on plain cache-aside (or uncached). A common production hybrid is **write-through for the hot working set, cache-aside for the long tail**, decided per key class. The mistake is to apply one pattern uniformly across all data because it was easier to wire up.

## 4. Write-behind (write-back): fast writes, deferred durability

**Senior rule of thumb: write-behind gives you the lowest possible write latency by acknowledging from the cache and flushing to the database asynchronously — reach for it for high-volume, loss-tolerant aggregates like counters and metrics, and never for data you cannot afford to lose.**

Write-behind (also called write-back) is the aggressive cousin of write-through. The write updates the cache and *acknowledges immediately* — the client is unblocked in microseconds. A background process then flushes the accumulated changes to the database later, typically batched and coalesced.

![Write-behind tree: write forks into instant cache ack and a deferred flush with a volatile window](/imgs/blogs/cache-patterns-in-production-6.webp)

The tree shows the fork. One write produces two consequences: on the left, the cache applies the change and acks, and the client is unblocked immediately — that is the latency win. On the right, the change is enqueued in a dirty-key set, sits in a *volatile window* where a crash means the change is gone, and is eventually picked up by a flusher that coalesces many updates into one batched `UPSERT` that finally makes the database durable. The red box is the entire reason write-behind is dangerous: between the ack and the flush, the only copy of that write lives in the cache's memory.

The pattern shines for one specific shape of data: **high-frequency aggregates where individual writes are fungible.** A view counter is the textbook case. If a page gets 10,000 views in a second, write-through would issue 10,000 `UPDATE counter = counter + 1` statements and melt the database. Write-behind applies all 10,000 increments to a single Redis counter in memory and flushes one `UPSERT view_count = 47213` per second. The coalescing is the magic — N writes collapse into one.

```python
import time
import threading

class WriteBehindCounter:
    """Counters live in Redis; a background flusher coalesces them to the DB.

    Trade-off: a crash between flushes loses the un-flushed increments.
    Only acceptable because a view count is loss-tolerant by nature.
    """
    def __init__(self, redis_client, db, flush_interval=1.0, max_batch=500):
        self.r = redis_client
        self.db = db
        self.flush_interval = flush_interval
        self.max_batch = max_batch
        self._stop = threading.Event()
        threading.Thread(target=self._flush_loop, daemon=True).start()

    def incr(self, key: str, by: int = 1) -> int:
        ## the write: apply in cache, mark dirty, ack immediately
        new_value = self.r.incrby(f"count:{key}", by)
        self.r.sadd("dirty:counts", key)     ## remember to flush this key
        return new_value                     ## client is unblocked here

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.flush_interval):
            self._flush_once()

    def _flush_once(self) -> None:
        ## atomically claim the current dirty set so new writes accumulate
        ## into a fresh set instead of being dropped mid-flush
        dirty = self.r.spop("dirty:counts", self.max_batch)
        if not dirty:
            return
        rows = []
        for key in dirty:
            value = self.r.get(f"count:{key}")
            if value is not None:
                rows.append((key, int(value)))
        ## one batched write for the whole window — this is the coalescing win
        self.db.executemany(
            "INSERT INTO counters (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            rows,
        )
```

The detail that separates a working write-behind from a data-shredding one is **how you claim the dirty set.** Using `SPOP` to atomically remove a batch means writes that arrive *during* the flush land in the set after the pop and will be caught on the next cycle — they are not silently dropped. A naive implementation that reads the dirty set, flushes, then clears it will lose every write that arrived between the read and the clear. That window is small but, at counter volumes, it is hit constantly.

### Second-order optimization: bounding the loss window

Write-behind's durability risk is not binary — you can dial it. The flush interval *is* your maximum data-loss window: a 1-second interval means a crash loses at most 1 second of un-flushed writes. Shorten it and you lose less but flush more often (less coalescing, more DB load); lengthen it and you coalesce harder but risk more. For a view counter, losing a second of increments is invisible. For anything where the loss is visible, write-behind is the wrong pattern and no flush interval is short enough to make it right. The honest framing: **write-behind trades a bounded amount of durability for unbounded write throughput, and you must be able to name the amount you are trading away.**

## 5. The consistency hazards: where caches actually break

This is the section the whole post exists for. Everything above is mechanics; this is where the mechanics betray you under concurrency. Three hazards account for the overwhelming majority of cache-correctness bugs I have seen in production, and all three come down to *ordering* — the order in which a writer and a reader, or two writers, touch the database and the cache.

### The cache-aside stale-read race

The most common and most insidious cache bug is a race between a reader populating the cache and a writer invalidating it. Picture a slow reader and a fast writer interleaving like this. The reader misses the cache and reads the *old* row from the database. Before it can write that old row back into the cache, the writer updates the database to a new value and deletes the cache key. *Then* the slow reader — holding the now-stale value it read a moment ago — writes it back into the cache. The cache now holds a stale value that the writer's delete was supposed to prevent, and it will keep serving that stale value until the TTL expires.

<figure class="blog-anim">
<svg viewBox="0 0 720 360" role="img" aria-label="Cache-aside stale-read race: a reader misses and reads the old row from the database, the writer updates the database and deletes the key, then the slow reader writes the old value back into the cache, leaving the cache stale" style="width:100%;height:auto;max-width:820px">
<title>Cache-aside stale-read race unfolding step by step</title>
<style>
.cr-lane{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cr-box{fill:var(--background,#fff);stroke:var(--border,#d1d5db);stroke-width:1.5}
.cr-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.cr-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.cr-tag{font:700 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:start}
.cr-step{font:700 12px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.cr-num{fill:var(--accent,#6366f1)}
.cr-cacheval{font:600 14px ui-monospace,monospace;text-anchor:middle}
.cr-old{fill:#b91c1c}
@keyframes cr-reveal{0%,8%{opacity:0;transform:translateY(6px)}14%,100%{opacity:1;transform:translateY(0)}}
@keyframes cr-s2{0%,22%{opacity:0;transform:translateY(6px)}28%,100%{opacity:1;transform:translateY(0)}}
@keyframes cr-s3{0%,40%{opacity:0;transform:translateY(6px)}46%,100%{opacity:1;transform:translateY(0)}}
@keyframes cr-s4{0%,58%{opacity:0;transform:translateY(6px)}64%,100%{opacity:1;transform:translateY(0)}}
@keyframes cr-s5{0%,76%{opacity:0;transform:translateY(6px)}82%,100%{opacity:1;transform:translateY(0)}}
@keyframes cr-cachefill{0%,76%{fill:var(--border,#d1d5db)}82%,100%{fill:#fca5a5}}
@keyframes cr-staletxt{0%,76%{opacity:0}82%,100%{opacity:1}}
.cr-a1{animation:cr-reveal 12s ease-out infinite}
.cr-a2{animation:cr-s2 12s ease-out infinite}
.cr-a3{animation:cr-s3 12s ease-out infinite}
.cr-a4{animation:cr-s4 12s ease-out infinite}
.cr-a5{animation:cr-s5 12s ease-out infinite}
.cr-cache{animation:cr-cachefill 12s ease-out infinite}
.cr-stale{animation:cr-staletxt 12s ease-out infinite}
@media (prefers-reduced-motion:reduce){.cr-a1,.cr-a2,.cr-a3,.cr-a4,.cr-a5,.cr-stale{animation:none;opacity:1;transform:none}.cr-cache{animation:none;fill:#fca5a5}}
</style>
<rect class="cr-lane" x="20" y="40" width="680" height="92" rx="10"/>
<rect class="cr-lane" x="20" y="248" width="680" height="92" rx="10"/>
<text class="cr-tag" x="32" y="34">READER (slow)</text>
<text class="cr-tag" x="32" y="356">WRITER</text>
<rect class="cr-box" x="300" y="160" width="120" height="56" rx="8"/>
<text class="cr-lbl" x="360" y="184">cache key k</text>
<rect class="cr-num cr-cache" x="312" y="192" width="96" height="16" rx="4"/>
<text class="cr-cacheval cr-old cr-stale" x="360" y="205">= v_old (stale)</text>
<g class="cr-a1">
<rect class="cr-num" x="44" y="58" width="150" height="56" rx="8"/>
<text class="cr-step" x="119" y="82">1. GET k -> MISS</text>
<text class="cr-step" x="119" y="100">read old row</text>
</g>
<g class="cr-a3">
<rect class="cr-num" x="270" y="58" width="180" height="56" rx="8"/>
<text class="cr-step" x="360" y="82">3. holds v_old</text>
<text class="cr-step" x="360" y="100">about to SET k</text>
</g>
<g class="cr-a5">
<rect class="cr-num" x="520" y="58" width="160" height="56" rx="8"/>
<text class="cr-step" x="600" y="82">5. SET k=v_old</text>
<text class="cr-step" x="600" y="100">cache now STALE</text>
</g>
<g class="cr-a2">
<rect class="cr-num" x="150" y="266" width="170" height="56" rx="8"/>
<text class="cr-step" x="235" y="290">2. UPDATE DB</text>
<text class="cr-step" x="235" y="308">k = v_new</text>
</g>
<g class="cr-a4">
<rect class="cr-num" x="400" y="266" width="160" height="56" rx="8"/>
<text class="cr-step" x="480" y="290">4. DEL k</text>
<text class="cr-step" x="480" y="308">cache cleared</text>
</g>
<figcaption>The reader's MISS read (step 1) races the writer's update+delete (steps 2 and 4); the reader's late SET (step 5) resurrects the stale value the delete just removed.</figcaption>
</figure>

Watch the sequence: the reader's `MISS` read happens *first* (step 1), then the writer's `UPDATE` and `DEL` complete (steps 2 and 4), and finally the reader's late `SET` (step 5) writes the old value back over the hole the delete made. The delete was correct. The reader simply finished its work after the delete and undid it. This is not a hypothetical — it is constantly possible whenever a read miss and a write to the same key overlap, and it gets *more* likely as your read traffic grows, which is exactly when you can least afford it.

Here is the race made concrete and then fixed, so you can see the structure:

```python
import threading

## --- THE RACE (do not ship this) -------------------------------------
## Reader and writer with no coordination. Under the wrong interleaving
## the reader's back-fill writes a stale value over the writer's delete.
def buggy_reader(db, key):
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)
    row = db.query_one("SELECT * FROM users WHERE id = %s", (key,))  ## may read OLD
    ## ... arbitrary delay here: GC pause, slow network, scheduler ...
    r.set(key, json.dumps(row), ex=300)   ## may overwrite a fresh DELETE
    return row

def buggy_writer(db, key, new_email):
    db.execute("UPDATE users SET email=%s WHERE id=%s", (new_email, key))
    r.delete(key)                          ## correct order, still racy vs reader
```

The fix is not to abandon cache-aside — it is still the right pattern. The fix is the **delayed double-delete**: the writer deletes the key, waits a short interval longer than a typical read, and deletes it again. The second delete evicts any stale value that an in-flight reader repopulated during the window.

![Delayed double-delete timeline: update, first delete, slow reader sets stale, second delete after a delay](/imgs/blogs/cache-patterns-in-production-7.webp)

```python
import threading

def writer_double_delete(db, key, new_email, delay=0.5):
    ## 1. write the source of truth
    db.execute("UPDATE users SET email=%s WHERE id=%s", (new_email, key))
    ## 2. first delete — handles the common case
    r.delete(key)
    ## 3. schedule a second delete AFTER the window in which a slow
    ##    reader (who read the old row before step 1) might back-fill.
    ##    The delay must exceed a typical read's DB-read-to-cache-set gap.
    threading.Timer(delay, lambda: r.delete(key)).start()
```

The timeline figure traces it: the writer updates the database at t=0, deletes the key at t=1ms, a slow reader sets the stale value at t=2ms, and the writer's *second* delete at t=500ms evicts it, so the next read at t=501ms reloads fresh. The delay is a tuning parameter — long enough to outlast the slowest realistic reader's window, short enough that the brief re-staleness does not matter. It is not a perfect guarantee (a pathologically slow reader can still beat it), which is why the TTL underneath remains your final backstop. But it closes the race for the overwhelming majority of real interleavings, and it is cheap.

### Why "write DB, then invalidate" and not "update cache, then DB"

The second hazard is the write ordering, and it is the one I see reversed most often because the wrong order *feels* more natural. The instinct is: "I am updating the user's email, so let me put the new email in the cache, then save it to the database." That is cache-first, and it is fragile.

![Invalidation order: cache-first strands a stale value when the DB write fails; DB-first reloads fresh](/imgs/blogs/cache-patterns-in-production-4.webp)

The before/after figure lays out both orders. On the left, the fragile cache-first path: you `SET` the cache to the new value, then `UPDATE` the database — but the database write fails or times out (a constraint violation, a deadlock, a connection drop). Now the cache says "new" and the database says "old." They have split-brained, and the cache will confidently serve the new value that *was never committed* until its TTL expires. Every read in that window returns a fabricated truth.

On the right, the correct DB-first path: `UPDATE` the database and let the commit succeed, *then* `DEL` the cache key. If the database write fails, you never touched the cache, so there is nothing inconsistent — the old value is still cached and still matches the old, uncommitted database state. The next read after a successful write misses and reloads the freshly committed row. The asymmetry is the point: **the database is the source of truth, so you mutate it first and only invalidate the cache after you know the truth has actually changed.** A failed database write should leave the cache exactly as it was, and only DB-first gives you that.

There is a deeper reason to prefer *delete* over *set* even in the DB-first order, which is the third hazard.

### Why updating the cache in place is fragile under concurrency

Consider two writers updating the same key, both doing DB-first-then-*set* (write-through style). Writer A commits `email=alice@new` and writer B commits `email=alice@newer` — B's write wins in the database. But the cache `SET`s can land in *either* order, because they are separate network round-trips with no ordering guarantee. If A's `SET` lands after B's, the cache ends up holding `alice@new` while the database holds `alice@newer`. The cache is stale and will stay stale until TTL, even though both writes "succeeded" and both followed DB-first ordering.

Now do the same with *delete*. Writer A commits, then `DEL k`. Writer B commits, then `DEL k`. Two deletes in either order produce the same result: the key is gone. The next reader reloads the *current* database value — whichever write actually won — and there is no way to end up with a stale value, because there is no value to be stale. Deletion is idempotent and order-insensitive; in-place update is neither.

```python
## Two concurrent writers, both DB-first.
## With SET (write-through), the cache can end up disagreeing with the DB.
## With DELETE (cache-aside invalidation), it cannot.

## Writer A                       ## Writer B
db.commit(email="alice@new")      db.commit(email="alice@newer")  ## B wins in DB
r.set(k, "alice@new")             r.set(k, "alice@newer")
## if A's SET lands last -> cache="alice@new", DB="alice@newer"  (STALE)

## vs.

r.delete(k)                       r.delete(k)
## either order -> key absent -> next read reloads "alice@newer" (FRESH)
```

This is the mechanical reason cache-aside's invalidation-by-delete is more robust than write-through's update-in-place: it removes a class of races entirely rather than narrowing them. It is also why, when you *do* choose write-through for its read-after-write guarantee, you should serialize writes per key (a lock, an actor, a single-writer shard) so the interleaving above cannot happen. The comparison is not "delete is always better than set" — it is "set requires you to handle write ordering, and delete does not."

### TTL as a safety net versus TTL as a correctness mechanism

The final piece of the consistency story is what TTL is actually for. TTL — time-to-live, the automatic expiry on a cached key — is genuinely valuable: it bounds how long any stale value can persist, so even if every invalidation in your system silently failed, the cache would be no more than one TTL out of date. That is a safety net, and you want it under every cached key for exactly that reason.

The mistake is using TTL *as* your consistency mechanism — "we do not invalidate, we just set a 30-second TTL and accept 30 seconds of staleness." This conflates two different things. TTL bounds the *duration* of staleness; it does nothing to bound the *occurrence* of it. With TTL-only and no invalidation, every single write produces up to a full TTL of guaranteed staleness for that key, because the only thing that ever refreshes the cache is expiry. A 30-second TTL on a frequently updated key means readers see 30-second-old data routinely, not occasionally. For a stock price, a balance, an inventory count, that is a correctness bug wearing a performance hat.

> TTL answers "how stale can it get?" It never answers "is it stale right now?" Invalidation answers the second question. You need both: invalidation for correctness on the common path, TTL as the backstop for when invalidation fails.

The right mental model: invalidate on every write so the common case is fresh, and set a TTL so the *uncommon* case — a missed invalidation, a crashed worker, a network partition that dropped a `DEL` — is self-healing within a bounded time. TTL is your seatbelt, not your steering wheel.

## 6. Refresh-ahead: hide the miss latency for hot keys

**Senior rule of thumb: refresh-ahead proactively reloads a hot key before it expires, so the request that would have eaten a cache miss gets a warm hit instead — use it for a small set of predictably hot keys, never for the whole keyspace.**

Refresh-ahead is a refinement that sits on top of cache-aside or read-through. The idea: for keys that are read constantly, do not wait for the TTL to expire and force a reader to eat the miss-and-reload latency. Instead, when a key is accessed and it is "close" to expiry (say, within the last 20% of its TTL), kick off an *asynchronous* refresh in the background while still serving the current cached value. By the time the old value would have expired, a fresh one has already replaced it, and no reader ever sees a miss for that key.

```python
import time, threading

def get_with_refresh_ahead(db, key, ttl=300, refresh_threshold=0.2):
    cached = r.get(key)
    ttl_remaining = r.ttl(key)                 ## seconds left, -2 if absent

    if cached is not None:
        ## if we are inside the last 20% of the TTL, refresh in the
        ## background WITHOUT blocking this reader
        if 0 < ttl_remaining < ttl * refresh_threshold:
            threading.Thread(
                target=lambda: r.set(key, json.dumps(db.load(key)), ex=ttl),
                daemon=True,
            ).start()
        return json.loads(cached)              ## serve the still-valid value now

    ## genuine miss (cold or evicted): load synchronously
    value = db.load(key)
    r.set(key, json.dumps(value), ex=ttl)
    return value
```

Refresh-ahead is the only pattern here that improves *tail latency on hits* rather than just hit rate — it converts what would have been periodic miss spikes on hot keys into a smooth stream of background reloads. It also doubles as a thundering-herd mitigation for the hottest keys, because the refresh happens once, ahead of time, instead of being triggered by a stampede at expiry.

The cost is that it only pays off for genuinely hot, predictable keys. Applied indiscriminately, it generates background reloads for keys that will never be read again before they expire, wasting database capacity refreshing cold data — the same cold-key waste write-through suffers. Refresh-ahead is a scalpel for your top-N hottest keys, identified from access metrics, not a blanket policy.

## 7. Putting it together: choosing a pattern

The four patterns trade consistency against write latency, and each carries a distinct dominant failure mode. The decision is a function of your workload's read/write ratio, your tolerance for staleness, and your tolerance for write-side cost or data loss.

![Comparison matrix: each pattern across consistency, write latency, read latency, failure mode, and best use](/imgs/blogs/cache-patterns-in-production-8.webp)

The matrix is the cheat sheet to keep. Read it column by column. **Consistency**: cache-aside and read-through are eventually consistent, bounded by TTL; write-through is strongly consistent for cached keys (assuming serialized writes); write-behind is weakly consistent until the flush completes. **Write latency**: cache-aside adds nothing to the write (just a delete); read-through has no write path; write-through is highest because every write pays cache plus DB; write-behind is lowest because the write acks from the cache alone. **Read latency** is low-on-hit for everyone — that is the whole reason a cache exists. **Failure mode** is where they diverge most sharply: cache-aside's signature failure is the stale-read race; read-through's is the cold-start stampede; write-through wastes writes on cold keys; write-behind loses data on crash. **Best for** summarizes the workload fit: cache-aside for general read-heavy traffic, read-through for uniform object loads, write-through for read-after-write needs, write-behind for counters and metrics.

A compact decision procedure, in priority order:

1. **Is this data loss-tolerant and write-heavy** (counters, metrics, telemetry)? Choose **write-behind**, with a flush interval set to your acceptable loss window.
2. **Do you need read-after-write consistency for these keys, and are reads much more frequent than writes** (profiles, catalog, config)? Choose **write-through** for the hot subset, and serialize writes per key.
3. **Do many call sites load the same kind of object the same way?** Choose **read-through**, so the loader lives in one place.
4. **Otherwise** — the default for almost everything — choose **cache-aside**: DB-first invalidation, TTL with jitter as the backstop, single-flight on miss, and a delayed double-delete on hot keys.

When in doubt, the answer is cache-aside. It is the default for a reason: it is the simplest contract, it keeps the cache out of your write critical path, and its failure modes are well-understood and well-mitigated. The other three are specializations you reach for when cache-aside's specific weaknesses — write-side staleness, scattered load logic, write-side throughput — are the thing hurting you.

## Case studies from production

### 1. The double-billed orders (cache-first ordering)

A payments service cached order totals in Redis to render the checkout summary fast. The write path computed the new total, `SET` it in the cache, then wrote the order to the database — cache-first. Under normal load it was invisible. During a flash sale, the database hit a deadlock-and-retry storm; a fraction of order-total writes failed *after* the cache had already been set to the new total. Customers saw a confirmed total in the UI that had no committed order behind it; a retry created a second order at the cached total, and a sliver of customers were charged twice. The symptom looked like a billing bug; the wrong first hypothesis was a race in the payment gateway. The actual root cause was the cache-first ordering: a failed DB write left a fabricated value in the cache. The fix was a one-line reorder to DB-first — commit the order, *then* invalidate the cached total — plus a check that no cache write ever precedes its database commit. The lesson: cache-first is not a performance optimization, it is a latent data-integrity bug that only fires when the database write fails, which is precisely during the incidents you cannot afford it.

### 2. The Facebook leases paper (the canonical race, solved at scale)

Facebook's memcache infrastructure, described in their NSDI paper, hit the cache-aside stale-read race at a scale where "rare" interleavings happen thousands of times a second. Their solution was *leases*: on a miss, memcached hands the requesting client a small lease token and only accepts a `set` for that key from a client holding a current lease. When a `delete` happens, it invalidates outstanding leases, so a slow reader holding a stale value finds its lease void and its back-fill `set` is rejected. This is the principled, server-side version of the delayed double-delete — instead of racing on timing, the cache server arbitrates who is allowed to populate a key. The lesson for the rest of us: the stale-read race is real enough that the largest cache deployment in the world built a protocol specifically to close it, and if you are operating at a scale where you cannot tolerate the double-delete's residual window, leases (or their equivalent in your cache layer) are the next step up.

### 3. The metrics pipeline that lost an hour (write-behind tuning)

An analytics team built a write-behind aggregation layer: per-event counters incremented in Redis, flushed to a warehouse every few minutes. It worked beautifully until a Redis node was failed over by the cloud provider during a routine maintenance window. Because the flush interval had crept up to five minutes during a "reduce DB load" tuning pass, and because the team had disabled Redis persistence to maximize throughput, the failover lost every increment accumulated since the last flush — close to five minutes of event counts during a peak hour, gone. The dashboards showed a visible notch. The wrong first hypothesis was a data-ingestion outage upstream. The actual root cause was that the flush interval — which *is* the data-loss window — had been lengthened without anyone recognizing it as a durability decision. The fix was twofold: cap the flush interval at the team's stated tolerance (under a minute) and enable Redis AOF persistence so a failover replays recent writes instead of losing them. The lesson: in write-behind, the flush interval is not a performance knob, it is your durability SLA written in disguise.

### 4. The inventory oversell (TTL as correctness)

An e-commerce site cached product inventory counts with a 60-second TTL and no invalidation on write — the team had reasoned "60 seconds of staleness is fine for inventory." For most products it was. For a limited-drop item with 50 units that sold out in seconds, it was a disaster: the cache happily served "in stock" for up to 60 seconds after the last unit was gone, the storefront kept accepting orders, and the company oversold by several hundred units it had to refund and apologize for. The wrong first hypothesis was a bug in the decrement logic. The actual root cause was treating TTL as the consistency mechanism: with no invalidation, every reader saw up-to-60-second-old inventory by design, and for a fast-selling item that staleness was catastrophic. The fix was to add DB-first invalidation on every inventory write (so the common case is fresh) and keep the TTL only as a backstop, plus a hard server-side stock check at order placement that never trusts the cache. The lesson: TTL bounds duration, not occurrence — for any value where being stale *right now* causes harm, you must invalidate, and TTL is only the seatbelt.

### 5. The synchronized expiry stampede (no jitter)

A content site warmed its cache after every deploy by pre-loading the top several thousand articles, all with an identical 1-hour TTL. Every hour, on the hour, all of those keys expired in the same instant, and the next wave of requests for them all missed simultaneously and stampeded the database with thousands of identical `SELECT`s. The database CPU spiked to saturation for ten to twenty seconds every hour, p99 latency tripled, and the on-call engineer learned to dread the top of the hour. The wrong first hypothesis was an hourly cron job hammering the database. The actual root cause was synchronized TTL expiry — a population wave that expired as one wave because every key got the same TTL. The fix was TTL jitter (a random spread of plus-or-minus twenty percent) so expiries smeared across a window instead of firing in lockstep, plus single-flight coalescing so even simultaneous misses for one key produced one database read. The lesson: a fixed TTL on a class of keys populated together is a scheduled stampede; jitter is one line of code and it dissolves the spike.

### 6. The write-through cache that throttled writes (cold-key waste)

A logging service adopted write-through "for consistency," caching every log-line metadata row on write. Reads of those rows were vanishingly rare — analysts queried them maybe once a week — but writes were enormous, millions per minute. The write-through cache dutifully `SET` a cache entry for every one of those millions of writes, doing a Redis round-trip on the hot write path that nobody benefited from on the read side. Worse, the flood of cold entries evicted the genuinely hot configuration keys the same Redis cluster was caching, so the *useful* cache's hit rate collapsed. Write throughput dropped because every write now waited on a pointless cache `SET`. The wrong first hypothesis was Redis being undersized. The actual root cause was applying write-through uniformly to a write-once-read-rarely workload, where the cache write is pure overhead. The fix was to drop the log-metadata rows out of the cache entirely (cache-aside, populate only the rare reads) and reserve write-through for the high-read-ratio config keys. The lesson: write-through earns its keep only when reads dominate writes; on write-heavy cold data it is a tax with no benefit.

### 7. The Twitter timeline fan-out (write-behind by another name)

Twitter's home-timeline architecture is, at heart, an enormous write-behind cache. When a user tweets, the write does not synchronously compute every follower's timeline; instead the tweet is written and a background fan-out process pushes it into each follower's cached timeline list in Redis, asynchronously. The "write" (posting) acks fast; the expensive work (materializing millions of timelines) happens behind the ack, coalesced and batched. The durability story is interesting: the *tweet itself* is durably stored synchronously (it cannot be lost), but the *cached timelines* are derived, write-behind, and reconstructible — if a timeline cache is lost, it can be rebuilt from the durable tweets. The lesson, and the pattern worth stealing: write-behind is safe for *derived, reconstructible* state even at massive scale, precisely because losing it is recoverable. The durability risk of write-behind is acceptable exactly when the data can be regenerated from a durable source — counters from an event log, timelines from tweets, aggregates from raw rows. Identify whether your write-behind data is reconstructible; if it is, the crash risk shrinks from "data loss" to "cache rebuild."

### 8. The cross-region cache that served the wrong continent (invalidation reach)

A global service ran a cache cluster per region and invalidated on write — but only in the *local* region. A write in us-east invalidated the us-east cache; the eu-west and ap-south caches kept serving the old value until their TTLs expired, because the `DEL` never crossed the ocean. Users in Europe saw profile changes made in the US lag by the full TTL. The wrong first hypothesis was replication lag in the database. The actual root cause was that invalidation was not propagated across regions — the team had built per-region caches without a cross-region invalidation channel. The fix was to publish invalidations onto a global message bus (so a `DEL` in any region fanned out to all of them), which is a natural fit for a change-data-capture stream — see [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) for the durable way to emit those invalidation events without losing them on a crash. The lesson: in a multi-region deployment, "invalidate the cache" means "invalidate *every* cache," and that requires an explicit propagation mechanism, not an implicit assumption that a local delete is global.

## When to reach for each pattern — and when not to

**Reach for cache-aside when:**

- You have a read-heavy workload with a long tail of keys and want to cache only what is actually read.
- You want the cache out of your write critical path (writes only delete, never block on a cache write).
- You can tolerate eventual consistency bounded by a TTL, with invalidation handling the common case.
- You want the simplest, best-understood pattern with the most off-the-shelf mitigations (jitter, single-flight, double-delete).

**Reach for read-through when:**

- Many call sites load the same kind of object the same way and you want one loader, not forty.
- You want library-level uniformity for TTLs, metrics, and stampede protection applied everywhere at once.
- Your access pattern is genuinely "key maps to one object loaded one way."

**Reach for write-through when:**

- Stale reads of a key are unacceptable and you need read-after-write consistency for that key.
- Reads dominate writes for that key class (high read-to-write ratio), so the per-write cache cost pays off.
- You can serialize writes per key (lock, actor, single-writer shard) so concurrent `SET`s cannot reorder.

**Reach for write-behind when:**

- Writes are high-volume and individually fungible or coalescable (counters, metrics, aggregates).
- The data is loss-tolerant for at least one flush interval, or reconstructible from a durable source.
- Write latency is the binding constraint and synchronous DB writes cannot keep up.

**Skip caching's fancier patterns — and stay on plain cache-aside, or no cache at all — when:**

- The data is written far more often than read (write-through and write-behind both waste effort; often cache nothing).
- A stale value causes real harm and you cannot guarantee invalidation reach (better to read the source of truth than to serve a confident lie).
- The keyspace has no hot set (refresh-ahead and even caching itself buy little when every key is read once).
- You cannot articulate which pattern you are using and what its failure mode is — that uncertainty is itself the signal to stop and choose deliberately.

The throughline of every case study above is the same: the outage was never the cache being slow. It was a pattern chosen by default, a write path wired up on instinct, an ordering left backwards, or a TTL mistaken for a guarantee. The patterns are simple. Choosing one on purpose — and getting the invalidation order right — is the entire job.

## Further reading

- [Redis applications and optimization](/blog/software-development/database/redis-applications-and-optimization) — the concrete Redis-side mechanics (data structures, persistence modes, eviction policies) that these patterns sit on top of.
- [The caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale) — where these patterns fit among the layers of caching (CDN, app-local, distributed) in a large system.
- [Cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd) — the deep dive on single-flight, request coalescing, and stampede protection referenced throughout this post.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the durable way to emit cache-invalidation events (and cross-region propagation) without losing them on a crash.

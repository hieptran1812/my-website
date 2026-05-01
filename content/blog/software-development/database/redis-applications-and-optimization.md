---
title: "Redis in Production: A Deep-Dive on Applications, Pitfalls, and the Optimization Playbook"
date: "2026-04-30"
publishDate: "2026-04-30"
description: "What Redis actually is under the hood, the patterns that ship to production, the failure modes that wake you at 3 AM, and the optimization moves that keep p99 flat."
tags:
  [
    "redis",
    "caching",
    "backend",
    "distributed-systems",
    "performance",
    "database",
    "latency",
    "site-reliability",
    "system-design",
    "in-memory",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

Most articles about Redis open with "Redis is an in-memory key-value store." That sentence is technically true and almost completely unhelpful, because it suggests Redis is interchangeable with Memcached or with a `dict` you happen to have running over TCP. It is not. Redis is an opinionated, single-threaded state machine that ships a curated catalog of data structures, attaches replication and persistence to them as side concerns, and invites you to compose distributed systems primitives — locks, queues, leaderboards, rate limiters, idempotency keys — out of those structures. Once you start treating it like that, every operational surprise it serves you starts to make sense.

This article is the long version of the conversation I have with engineers when they ask "should we put X in Redis?" or, more often, "Redis is slow, what do we do?" It is written for people who already have Redis running in production, have already been burned at least once, and want a model coherent enough to predict the next burn before it happens.

![Redis mental model: a single-threaded event loop over in-memory data](/imgs/blogs/redis-applications-and-optimization-1.png)

The diagram above is the mental model: every client command, no matter how cheap or expensive, gets serialized into one event loop running on one core. Persistence and replication are async taps off the side. Once that picture is internalized, almost every production failure I have seen falls out of it. The article walks the layers from the inside out — data structures, application patterns, persistence, replication, memory, latency — then closes with eight detailed incidents and a frank "when to reach for Redis, when not to" decision matrix.

## 1. The Mental Model: A Single-Threaded State Machine

Redis is, mechanically, a `while (true)` loop. That loop reads RESP-encoded frames off connected client sockets, dispatches each command to a C function that mutates an in-memory data structure, and writes the reply back. There is no per-request thread, no goroutine pool, no `async/await` you should worry about. There is one core executing one command at a time.

That choice is the source of every Redis virtue and every Redis vice.

The virtues come for free: no locks, no contention, no memory ordering bugs, no two-phase commits. A `INCR counter` is genuinely atomic because there is literally one thread that can touch `counter` at a time. A Lua script is atomic because Redis simply does not run anything else while a script is executing. The cost of synchronization is moved out of the data store and into the design of the commands themselves; the result is that p50 latency for trivial commands is around one microsecond of CPU time. RAM access is roughly 100 ns; even a non-trivial command rarely does more than a few hundred memory accesses.

The vices are the same property turned the other way. If your command takes 200 ms, every other client in the world waits 200 ms behind it. There is no fairness, no preemption. A 10-million-element `SMEMBERS` does not "yield"; it runs to completion, occupies the loop, fills the kernel send buffer, and stalls every other request behind it. That is why you will eventually find an internal Slack message that says "Redis is slow" when in fact Redis is doing exactly one thing very fast and 50,000 clients are queued behind it.

Three properties that fall out of this design and are worth holding in working memory at all times:

1. **Big-O for individual commands matters more than you think.** Memcached can hide a slow command behind hashed worker threads. Redis cannot. `O(N)` means "blocks the world for N work units."
2. **Persistence is not durability.** RDB snapshots run in a forked child. AOF buffers writes through the page cache. The default `appendfsync everysec` setting can lose up to about a second of writes. The defaults exist because most users would rather have throughput than the ironclad durability they think they have.
3. **Replication is asynchronous.** A `SET` that returned `OK` to your client may not yet be on any replica. A primary failure within the replication-lag window simply loses those writes. There are no transactions across replicas. Pretending otherwise is the most common architectural error I see.

Multi-threading was added in Redis 6, but only for I/O — `io-threads` parses and serializes RESP on multiple threads to overlap network work with command execution. The command execution itself is still single-threaded. Redis 7 added Functions, ACLs, and a more efficient AOF rewrite. Redis 8 made tighter use of jemalloc and tightened up some encodings. Valkey (the BSD-licensed fork after the 2024 license change) and KeyDB and Dragonfly all chip away at the single-thread constraint in different ways. None of those changes the fundamental shape: one logical thread of execution per shard.

The practical implication for design: **plan around the budget**. Pick a target p99 — say 5 ms — and treat it as a budget. Network round trip eats 0.5 to 2 ms. The kernel and TCP eat another 0.1 to 0.5 ms. That leaves something like 2 to 4 ms inside Redis itself. If you have ten clients doing a 1 ms command each, that is fine. If you have one client doing a 50 ms command, every other client just blew their budget. The math is unforgiving and it explains 90% of the production debugging stories in this article.

## 2. Data Structures and What They Are Actually For

Redis ships nine first-class data structures: strings, lists, hashes, sets, sorted sets, streams, bitmaps, HyperLogLogs, and geospatial indexes. Choosing the right one is the difference between a 2 µs command and a 200 ms freeze. Choosing the wrong one is also the most common cause of memory bloat.

A point that tutorials almost always miss: **the user-visible type is not the storage encoding.** A hash with twelve small fields is stored as a `listpack` (a flat byte array, cache-friendly, no overhead per field). The same hash with one thousand fields gets transparently re-encoded into a `hashtable` (open-addressed, one C struct per field, much higher per-element overhead). The transition happens automatically at thresholds you can configure (`hash-max-listpack-entries`, `hash-max-listpack-value`). Same for sorted sets (listpack ↔ skiplist+hashtable), sets (intset ↔ listpack ↔ hashtable), and lists (listpack ↔ quicklist).

This matters operationally. A small hash uses about 80 bytes total; the same hash after promotion uses 80 bytes per field plus pointer overhead. Crossing the threshold by accident is a 10x memory regression for no functional change. I have seen a deploy that bumped a config value and quietly tripled the memory footprint of every user session because the encoding flipped.

| Type | Small encoding | Large encoding | Sweet spot | Trap |
| --- | --- | --- | --- | --- |
| String | `embstr` (≤44 B), `raw` | — | counters, JSON blobs <1 KB, idempotency keys | storing a 50 MB blob — `GET` blocks the loop |
| List | listpack | quicklist | bounded queues, recent-N feeds | unbounded growth — `LRANGE 0 -1` is a foot-gun |
| Hash | listpack | hashtable | per-entity fields where you fetch a few at a time | huge hashes (>1M fields) — `HGETALL` blocks |
| Set | intset / listpack | hashtable | tag membership, dedup, "online users" | `SMEMBERS` on big sets — large reply stalls write() |
| Sorted set | listpack | skiplist + hashtable | leaderboards, time-windowed counters, sliding-window rate limiters | `ZADD` of millions of elements at once |
| Stream | radix tree of listpacks | — | append-only logs, queues with consumer groups | unbounded streams — `MAXLEN` is your friend |
| Bitmap | string | — | feature flags per user, daily-active sets | sparse user IDs — bitmap holes waste RAM |
| HyperLogLog | string (12 KB sparse → dense) | — | unique-count estimation at billions scale | exact counts — HLL has ~0.81% error |
| Geo | sorted set | — | nearest-N queries, geofencing | very dense regions — sort key collisions |

The rules of thumb that come up over and over:

**Strings are not just strings.** A string is a `byte[]` plus a length. It is what you reach for when the value is opaque to Redis. But Redis also has commands that interpret the bytes as numbers (`INCR`, `INCRBYFLOAT`), as bitfields (`SETBIT`, `BITCOUNT`, `BITPOS`), or as ranges (`GETRANGE`). A 10-million-bit bitmap costs 1.25 MB and answers "is user 9,283,991 active?" in O(1). A `SET` with a JSON blob is the most boring useful primitive in the world.

**Hashes are the right choice when you sometimes want one field.** If you store a user as a JSON string, every read or write of any field re-serializes the whole object. With a hash, `HGET user:42 email` reads one field. The trade-off is that hashes are not type-aware (`HINCRBY` notwithstanding), and the overhead per field grows once the encoding flips to hashtable. Use hashes for small-to-medium structured records (5 to 100 fields).

**Sorted sets are the most under-appreciated structure in the language.** Skiplists support O(log N) insertion, removal, and rank lookup; the hashtable side gives you O(1) score lookup. That combination is exactly what you need for leaderboards (`ZADD leaderboard <score> <user>`, `ZREVRANGE leaderboard 0 9`), for time-bucketed sliding windows (use the timestamp as the score, evict old entries with `ZREMRANGEBYSCORE`), for priority queues, for time-series. Most "I need a queue with priorities" problems map cleanly onto a sorted set with the priority as the score.

**Streams replace pub/sub when you need durability.** Redis pub/sub is fire-and-forget: if a subscriber is down, it loses the message. Streams are an append-only log with consumer groups, acknowledgements, and pending-entry tracking. They are not Kafka — they live on one shard and are not designed for petabytes — but for "queue with retries that fits in RAM," streams are the right primitive. The 80% case that does not need Kafka is exactly this case.

**HyperLogLog is for the cardinality problem only.** "How many unique users hit /search today?" answered exactly costs you a `SET` per day (potentially millions of members). Answered approximately with HLL costs you 12 KB per day, regardless of how many uniques you observe. You give up the exact count and the ability to enumerate; you get a constant-memory estimator with provable error bounds. Most product analytics works fine at 0.81% error.

**Bitmaps are the right answer to "did user X do Y on day Z?" only when user IDs are dense.** A bitmap with 1 billion bits is 125 MB. If your IDs are sparse (UUIDs, hashes), the bitmap is mostly zeros and you are paying for empty space. In that case use a Roaring bitmap module (`redis-roaring`) or fall back to a set.

A short, runnable encoding-audit script that I keep in my dotfiles:

```python
import redis, json
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def audit(pattern, sample=200):
    # SCAN, never KEYS, on a busy box
    keys = []
    for k in r.scan_iter(pattern, count=500):
        keys.append(k)
        if len(keys) >= sample:
            break
    rows = []
    for k in keys:
        t = r.type(k)
        enc = r.object("encoding", k)
        size = r.memory_usage(k) or 0
        rows.append((k, t, enc, size))
    rows.sort(key=lambda r: -r[3])
    for k, t, enc, size in rows[:25]:
        print(f"{size:>10} B  {t:<10} {enc:<12} {k}")

audit("user:*")
audit("session:*")
```

The first run on a strange Redis is always educational. You learn that the team that wrote `session:*` is storing a serialized 8 KB Java object, and that the team that wrote `user:*` has 12 keys above 1 MB which are users with very large social graphs stored as sets, and that you have a hash that should have stayed a listpack but did not because somebody set an absurd value on a single field.

## 3. The Application Patterns We Actually Ship

Once the data structures click, the application patterns become composition. Almost every production use of Redis I have seen falls into one of the patterns below. Each comes with a consistency model and a failure mode; ignore either at your peril.

### 3.1 Cache-Aside (the default everyone reaches for)

Cache-aside is the pattern where the application code, not Redis, owns the cache logic:

```python
def get_user(user_id: int) -> dict:
    key = f"user:{user_id}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)
    row = db.fetch_one("SELECT * FROM users WHERE id = %s", user_id)
    if row is None:
        # Negative caching: avoid hammering the DB on a 404 storm
        r.set(key, json.dumps(None), ex=60)
        return None
    # Add jitter to TTL so a deploy does not align expiries
    ttl = 600 + random.randint(0, 60)
    r.set(key, json.dumps(row), ex=ttl)
    return row
```

Three details that turn a cache from correct into correct *and* survivable:

- **Jittered TTLs.** If everything you wrote at the same time also expires at the same time, you get a thundering herd when they expire. A small uniform jitter (±10%) breaks the alignment and turns a 500 ms latency cliff into a smooth ramp. This is the optimization that has paid for itself the most times in my career.
- **Negative caching.** When the DB returns nothing, *cache that fact* with a short TTL. Otherwise a "user does not exist" lookup hammers Postgres at full QPS. Be careful with the TTL: too short and you do not protect the DB; too long and a user creation in the DB is invisible for minutes.
- **Single-flight for cold-start.** If a hot key expires and 5,000 requests arrive simultaneously, all 5,000 will miss and all 5,000 will hit the DB. Use a per-key lock (`SET key "computing" NX EX 10`) and have the loser waiters either poll briefly or read a stale value. `request coalescing` is the term you want.

### 3.2 Write-Through and Write-Behind

Write-through writes to the cache *and* the database synchronously, returning success only when both succeed. It guarantees the cache is fresh, at the cost of doubling the write tail latency. Write-behind writes only to the cache and uses a background process to flush to the database, getting fast writes at the risk of losing recent writes if the cache crashes.

In Redis, write-through is straightforward. Write-behind is dangerous and rarely what you want unless you have a domain (counter aggregation, click streams) where losing a few seconds of writes is acceptable. The pattern looks like:

```python
def increment_view(post_id: int):
    # cache absorbs the burst; flusher sweeps to DB
    r.hincrby("post_views_buffer", post_id, 1)

# Background process every 5 seconds
def flush_views():
    pipe = r.pipeline()
    pipe.hgetall("post_views_buffer")
    pipe.delete("post_views_buffer")
    counts, _ = pipe.execute()
    if counts:
        db.batch_increment("posts", "view_count", counts)
```

Note the read-then-delete is not atomic across multiple keys; if a write arrives between `HGETALL` and `DELETE`, you lose it. The fix is `RENAME post_views_buffer post_views_flushing` (atomic) followed by `HGETALL` + `DEL` on the renamed key.

### 3.3 Distributed Locks (and why Redlock is contested)

The single-instance lock pattern is simple and, for a vast majority of uses, sufficient:

```python
import uuid, time

def acquire_lock(name: str, ttl_ms: int = 5000) -> str | None:
    token = uuid.uuid4().hex
    if r.set(f"lock:{name}", token, nx=True, px=ttl_ms):
        return token
    return None

RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

def release_lock(name: str, token: str) -> bool:
    return bool(r.eval(RELEASE_LUA, 1, f"lock:{name}", token))
```

The Lua script is mandatory: a naive `if r.get() == token: r.delete()` has a race where the lock expires between the GET and the DEL and you accidentally release someone else's lock.

`SET NX PX` with a unique token gives you mutual exclusion *as long as the Redis instance is up and you trust its clock*. For workflows where two workers running the same job is a correctness violation, you also need to reason about what happens during failover. Redis replication is asynchronous: a `SET NX` that returns OK on the primary may not be on the replica when the primary fails. If the replica is promoted, a second client can acquire the same lock. You now have two holders.

Redlock is Salvatore Sanfilippo's proposed solution: acquire the lock on a majority of N independent Redis primaries within a bounded time. Martin Kleppmann's well-known critique points out that Redlock relies on bounded clock drift and bounded process pauses (GC, preemption, container freezes), and a long enough pause can lead to a holder thinking it still has the lock while another client has acquired it on the majority. Antirez's response is that Redlock is fine for "efficiency" use cases (avoid duplicate work) and that "correctness" use cases need fencing tokens regardless of which lock service you use.

The pragmatic answer: if losing the lock would corrupt data, use a fencing token. The token is a monotonically increasing number issued by the lock service; every operation guarded by the lock writes the token to the resource, and the resource refuses any operation with a token less than the most recent it has seen. With fencing tokens, single-instance Redis locks are fine; without them, no lock service is.

### 3.4 Rate Limiting

The two patterns worth knowing are token bucket and sliding window.

**Token bucket** (fixed-rate refills, burst-friendly) in Lua:

```lua
-- KEYS[1]: bucket key
-- ARGV[1]: capacity
-- ARGV[2]: refill rate (tokens/sec)
-- ARGV[3]: now (ms)
-- ARGV[4]: cost
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1]) or capacity
local ts     = tonumber(data[2]) or now

local delta = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + delta * rate)

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 60000)
return allowed
```

**Sliding window** (exact rolling-window count, more memory) using a sorted set:

```python
def allow_request(user_id: str, limit: int, window_sec: int) -> bool:
    key = f"rl:{user_id}"
    now = int(time.time() * 1000)
    cutoff = now - window_sec * 1000
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff)
    pipe.zadd(key, {f"{now}:{uuid.uuid4().hex}": now})
    pipe.zcard(key)
    pipe.expire(key, window_sec + 1)
    _, _, count, _ = pipe.execute()
    return count <= limit
```

The token bucket is constant memory per key and friendly to bursts. The sliding window is exact but uses memory proportional to the number of requests in the window. At very high QPS per key (think 1000 RPS sustained per user) the sliding window blows memory; at that rate a token bucket or a fixed-window counter is what you want.

### 3.5 Idempotency Keys

```python
def charge_card(idempotency_key: str, amount_cents: int):
    key = f"idem:{idempotency_key}"
    if not r.set(key, "in_progress", nx=True, ex=86400):
        # Replay: somebody else already started or finished this
        existing = r.get(key)
        if existing == "in_progress":
            raise ConflictError("retry later")
        return json.loads(existing)
    try:
        result = stripe.charge(amount_cents)
        r.set(key, json.dumps(result), ex=86400)
        return result
    except Exception:
        r.delete(key)  # let the client retry
        raise
```

Two non-obvious points. First, the success-cache TTL must be longer than the longest plausible client retry window — 24 hours is conservative, days are not unreasonable for payments. Second, `r.delete(key)` on failure is a deliberate choice: failing the request means it never happened, so the client should be able to retry. If you keep the in-progress marker, you've turned a transient failure into a permanent "duplicate request" rejection.

### 3.6 Session Store

The 95% case is `SET session:<id> <serialized blob> EX <session ttl>`. The 5% case where this gets interesting:

- **Sliding sessions:** every request bumps the TTL with `EXPIRE`. Cheap and correct.
- **Concurrent revocation:** to log a user out everywhere, you need to either invalidate all their sessions (track them in a set under `user:<id>:sessions`) or store a `user_revoked_at` timestamp and reject sessions older than it. The set approach is precise but requires writes on every login. The timestamp approach is simpler but checks against a per-user value on every request.
- **Hash vs string:** if you read individual fields more than the whole session (think "just need the user_id"), use a hash. If you always deserialize the whole thing, a string is faster.

### 3.7 Pub/Sub vs Streams (and queue patterns)

The first instinct of someone coming from RabbitMQ or Kafka is to reach for `PUBLISH`/`SUBSCRIBE`. This is almost always wrong. Redis pub/sub has *no durability, no replay, no consumer groups*. If a subscriber is disconnected for 100 ms, it misses everything published in those 100 ms. There is no way to get those messages back.

Streams are the durable replacement:

```python
# Producer
r.xadd("orders", {"order_id": "abc", "amount": "1999"}, maxlen=100000, approximate=True)

# Consumer with consumer group
r.xgroup_create("orders", "billing", id="0", mkstream=True)
while True:
    msgs = r.xreadgroup("billing", consumer="billing-1",
                        streams={"orders": ">"}, count=64, block=5000)
    for stream_name, entries in msgs or []:
        for entry_id, fields in entries:
            try:
                process(fields)
                r.xack("orders", "billing", entry_id)
            except Exception:
                # Stays in pending, will be reclaimed via XAUTOCLAIM later
                pass
```

`XAUTOCLAIM` (Redis 6.2+) lets you reclaim entries from dead consumers based on idle time, which is the missing piece for at-least-once delivery. `MAXLEN ~ 100000` (the `~` means approximate) caps the stream; without it, an unattended stream grows forever and eventually OOMs the instance.

Pub/sub still has its uses: cache-invalidation broadcasts, "low-importance" notifications, the keyspace-notification firehose. Anywhere the assumption "if you missed it, you missed it" is acceptable.

### 3.8 Leaderboards and time-bucketed analytics

The leaderboard is the canonical sorted-set application and worth walking through because it shows up in disguise everywhere. The naive shape is `ZADD leaderboard <score> <user>`, then `ZREVRANGE leaderboard 0 9 WITHSCORES` to read the top ten. Insertion is O(log N), top-K reads are O(K + log N). At a million entries this is sub-millisecond.

The trap is *time-windowed* leaderboards. "Top players in the last 24 hours" cannot be answered from a single sorted set if the score is the user's total. The pattern is to bucket: `ZADD leaderboard:2026-04-30:14 <score> <user>` for hour buckets, then on read merge the relevant 24 buckets via `ZUNIONSTORE`. With expiries on the buckets you get a free sliding window. The cost is N writes per event (one per bucket the user belongs to); for low-cardinality buckets this is fine.

The same shape solves "top searched products this week," "most-active threads in the last hour," "trending tags." Anything where the answer is "the top K of a windowed aggregation," sorted sets are the right tool.

### 3.9 Caching with negative results and bloom filters

When the database returns nothing, *cache that fact*. Otherwise a 404-storm hammers the DB. But TTLs alone are not enough at high cardinality: if the attacker generates random IDs, you cache one negative result per random ID and your cache fills with garbage.

The defensive structure is a Bloom filter (RedisBloom module, or a self-rolled bitmap). The filter answers "could this ID exist?" probabilistically. False positives go to the cache + DB; true negatives short-circuit at O(1). For a million-element filter at 1% false-positive rate, the memory cost is about 1.2 MB. This is the cheapest defense against ID-enumeration storms I know of.

```python
def maybe_exists(user_id: int) -> bool:
    # Bloom filter populated nightly from the DB
    return r.execute_command("BF.EXISTS", "users:bloom", str(user_id))

def get_user(user_id: int) -> dict | None:
    if not maybe_exists(user_id):
        return None
    # ...cache-aside path
```

### 3.10 Pattern selection table

| Pattern | Consistency | Failure mode | When to use |
| --- | --- | --- | --- |
| Cache-aside | eventual; stale window = TTL | DB outage if cache dies cold | default for read-heavy data |
| Write-through | strong cache↔DB | doubled write latency | when reads must always be fresh |
| Write-behind | DB lags cache; lossy | cache crash loses recent writes | counters, analytics, click streams |
| SET NX EX lock | strong on one node | split-brain on failover | efficiency, not correctness |
| Redlock | majority quorum | clock drift / GC pauses | use fencing tokens regardless |
| Token bucket | exact, constant memory | clock skew if not centralized | API rate limiting |
| Sliding window zset | exact | memory grows with QPS | low-QPS per key, exact required |
| Idempotency key | exact | TTL too short → false replays | payment-shaped workflows |
| Streams + groups | at-least-once | XAUTOCLAIM gap | durable queues that fit in RAM |
| Pub/Sub | fire-and-forget | subscriber disconnect = loss | invalidations, notifications |

## 4. Persistence: RDB, AOF, and the Durability Lie

![RDB vs AOF write paths and the hybrid default](/imgs/blogs/redis-applications-and-optimization-2.png)

Most teams configure Redis once, with whatever the install playbook said, and never look at it again. The configuration they end up with is approximately "everysec AOF + occasional RDB," and the durability they imagine they have is approximately "we never lose more than a second." Both of those approximations are wrong in interesting ways.

**RDB** is a point-in-time binary snapshot. When `save 900 1` triggers (or you run `BGSAVE`), Redis calls `fork()`. The child inherits the parent's memory through copy-on-write page tables and serializes the entire keyspace to a `.rdb` file. The parent keeps serving traffic. This is beautiful when it works: restart from a 50 GB RDB takes seconds (it is mostly `mmap` + memcpy), the file is small, you can copy it to S3.

The price is the fork. On Linux, `fork()` on a process with a 50 GB RSS does not copy the 50 GB of pages — but it *does* copy the page tables, which are roughly 0.2% of RSS, or 100 MB. The kernel walks every page table entry, marks them read-only, and the parent stalls until that completes. On commodity hardware, a 50 GB instance can stall for 50 to 200 ms on fork. Tail latency just took a hit and there is nothing the application can do about it. If parent pages are then written during the snapshot, copy-on-write kicks in and memory usage spikes; under high write churn the spike can be 50% to 100% of the keyspace size. I have watched a Redis instance OOM during BGSAVE because the host had insufficient RAM headroom for the CoW spike.

**AOF** is an append-only log of every write command. On `SET foo bar` the server encodes the command back into RESP and writes it to a buffer; the buffer is flushed to the OS via `write()`, and on a configurable schedule the OS page cache is flushed to disk via `fsync()`.

The fsync policy is the lever:

| Policy | Loss window | Latency cost | Use for |
| --- | --- | --- | --- |
| `appendfsync no` | until OS decides | nearly zero | caches, ephemeral data |
| `appendfsync everysec` (default) | up to ~1 second | small, but stalls if disk is slow | most production |
| `appendfsync always` | zero (theoretically) | dominated by disk fsync, often 10x | financial, durability-critical |

Three subtle truths buried in this table. First, "everysec" is not actually "every second" under disk pressure: if the previous fsync has not finished, the *write thread itself* will block until it completes, turning your nice async durability into a synchronous tail-latency spike. The Redis log will say `Asynchronous AOF fsync is taking too long (disk is busy?). Writing the AOF buffer without waiting for fsync to complete, this may slow down Redis.` This is the entry point to a production incident waiting to happen.

Second, `always` does *not* mean every write is flushed to physical media. Most disks (especially virtualized cloud disks, EBS, etc.) implement their own write caches. `fsync()` returns when the kernel has handed the data to the disk, not necessarily when the disk has committed it to media. To get true durability you also need `innodb_flush_method`-style options at the kernel/disk level. For most non-financial use cases this is fine; just know what you are and are not promising.

Third, AOF rewrite uses `fork()` too. `BGREWRITEAOF` forks a child that walks the keyspace and emits the minimal set of commands that reproduce it, replacing the old AOF. Same fork stall, same CoW spike. If you have RDB *and* AOF both configured naively, you can fork twice in quick succession.

The Redis 7 default is the **hybrid format**: `aof-use-rdb-preamble yes`. Rewrite emits an RDB blob as the prefix and then the recent AOF tail. On restart you get the RDB-fast load plus the AOF-durable tail. This is the configuration almost everyone should use today.

A pragmatic configuration for a typical "this is important but not a bank" workload:

```
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-use-rdb-preamble yes
save 3600 1 300 100 60 10000
no-appendfsync-on-rewrite no
```

The last line is interesting. `no-appendfsync-on-rewrite no` means "keep doing fsync even during rewrite." If you set it to `yes`, you skip fsync during the rewrite to dodge the disk contention, *and accept that you might lose the rewrite window's worth of data on a crash during rewrite*. On slow disks, flipping this to `yes` is sometimes the difference between Redis usable and Redis not, but the loss window is now `rewrite duration`, not `1 second`.

A short test to see how bad your fork actually is:

```bash
# As you, with redis-cli:
redis-cli config set save ""
redis-cli config set appendonly no
# preload
redis-cli debug populate 50000000
# now measure
redis-cli --intrinsic-latency 30 &
redis-cli bgsave
# watch the latency report; you'll see the fork stall as a spike
```

If the fork stall is unacceptable for your workload, options in order of preference: (1) shrink the instance (shard sooner), (2) disable RDB and rely on replicas + AOF, (3) move snapshots to a replica, (4) use a managed service that does these things for you.

## 5. Replication, Failover, and Sentinel vs Cluster

![Redis Cluster: 16384 slots and the MOVED/ASK redirect protocol](/imgs/blogs/redis-applications-and-optimization-3.png)

Redis replication has one defining property: **it is asynchronous**. The primary acknowledges a write to the client as soon as it has applied it locally. The replication stream goes out at its own pace. There is a `WAIT` command that blocks until N replicas have ack'd, and there is `min-replicas-to-write` which refuses writes if not enough replicas are connected, but neither of these turns Redis into a synchronously-replicated system. They give you back-pressure, not durability.

The replication protocol is conceptually simple. Primary maintains a circular **replication backlog buffer** with a monotonic offset. Each replica has a `replid` and the latest offset it acknowledged. On reconnect, a replica sends `PSYNC <replid> <offset>`. If the primary still has those bytes in the backlog, it streams the delta (a *partial resync*); if not, it sends a full RDB plus the tail (a *full resync*). The default backlog size is 1 MB, which is laughably small for any non-toy workload — a network blip on a busy primary will trigger a full resync, which means a fork, an RDB transfer, and a fresh load on the replica. Bump `repl-backlog-size` to 256 MB or more on busy primaries. This is one of the cheapest, highest-leverage knobs in the entire config file.

**Sentinel** sits at the level of "primary + replicas" without sharding. A small group of Sentinel processes (3 or 5) gossip about the health of the primary; on perceived failure they vote to promote a replica, then reconfigure the others to follow it, and signal clients via pub/sub. Clients learn the new primary by asking Sentinel. Failover takes 10–30 seconds in practice. Sentinel is the right answer when your dataset fits on one node and you want HA with minimal complexity.

**Cluster** shards data across N primaries (each potentially with its own replicas). The keyspace is partitioned into 16384 *slots*; each key's slot is `CRC16(key) mod 16384`. Slots are owned by primaries; clients keep a local slot→node map.

When the client guesses wrong, the server replies:

- `-MOVED 4523 192.168.0.12:6379`: the slot lives on that node now and forever (until rebalance). Update your slot map permanently.
- `-ASK 4523 192.168.0.12:6379`: the slot is in the middle of being migrated. For *this one command*, retry on the new node with an `ASKING` prefix; do not update the map.

Cluster gossip uses a separate cluster bus port (typically `<port>+10000`) running a binary protocol. Each node ping-pongs with a random subset of peers every second; failure detection is based on majority opinion. Failover requires a majority of *primaries* (not replicas) to agree, which is why you need at least 3 primaries for a working cluster.

Two operational landmines specific to Cluster:

**Multi-key operations are constrained to a single slot.** `MSET foo 1 bar 2` works only if `foo` and `bar` hash to the same slot. They almost certainly do not. Redis solves this with **hashtags**: only the substring inside the first `{...}` is hashed. So `{user42}:profile` and `{user42}:cart` are guaranteed to live on the same shard, and `MGET {user42}:profile {user42}:cart` works. The cost is that you have manually pinned both keys to the same shard, so a hot user can hot-spot one shard. There is no free lunch.

**Slot migration during traffic is the single most disruptive operation.** When you add a node and run `redis-cli --cluster reshard`, slots migrate one by one. During migration of slot S, reads and writes to S can return ASK redirects and clients double their round trips. Bad clients that do not understand ASK simply error out. Aim to do migrations during low-traffic windows; if your client library or proxy does not support cluster-aware retries, fix that first.

Sentinel vs Cluster picking guide:

| Need | Use |
| --- | --- |
| Single-node fits, simple HA | Sentinel |
| Dataset > 25 GB *or* writes > 50k QPS | Cluster |
| Strong cross-key transactions | Neither — use Postgres |
| Multi-region active-active | Neither — use a managed CRDT system or Active-Active Redis Enterprise |

A note on managed services: ElastiCache, MemoryStore, Upstash, Redis Cloud all sit somewhere on the Sentinel/Cluster axis with various opinions about how much you see. Read your provider's docs on `min-replicas-to-write`, the exact failover behavior, and whether they expose the cluster bus or hide it behind a proxy. The default behaviors differ in ways that matter.

## 6. Memory: The Resource That Quietly Kills You

The thing that surprises engineers new to Redis is how much memory they "spend" on metadata. A bare Redis with no data uses 4-5 MB. Each key has overhead for the dictEntry, the SDS header, the embedded encoding tag — typically 40 to 100 bytes per key on top of whatever you stored. For a workload of "1 billion 64-byte values", you do not have a 64 GB problem; you have a 100+ GB problem.

`maxmemory` is the headline knob. When the resident keyspace approaches it, Redis enforces an **eviction policy**. The choices are:

| Policy | Behavior | Use when |
| --- | --- | --- |
| `noeviction` | refuse writes that would exceed `maxmemory` | source of truth, *not* a cache |
| `allkeys-lru` | evict approximate-LRU across all keys | general cache |
| `allkeys-lfu` | evict approximate-LFU (frequency-aware) | skewed access (long-tail hot keys) |
| `volatile-lru` | LRU but only over keys with TTL | mixed cache + persistent state |
| `volatile-lfu` | LFU on keys with TTL | same |
| `allkeys-random` / `volatile-random` | random pick | almost never |
| `volatile-ttl` | evict shortest TTL first | TTL is a priority signal |

"Approximate" is doing real work in those names. Redis does not maintain a true LRU list — that would cost a doubly-linked-list pointer per key. Instead it samples `maxmemory-samples` keys (default 5) and evicts the worst from the sample. Bumping samples to 10 makes eviction more accurate at modest CPU cost; samples=100 starts to slow down eviction noticeably.

LFU is worth thinking about for caches with skewed access. A hot key accessed 10,000 times an hour and a cold key accessed once an hour both have a "recent access" with LRU; they look the same. LFU correctly biases retention toward the hot one. The classic mistake is using `allkeys-lru` for a workload with one very hot key and millions of cold long-tail keys: a momentary scan over the long tail evicts the hot key, app latency cliffs.

**Fragmentation** is the second memory boogeyman. After a long run with mixed allocation sizes, jemalloc's arenas develop holes: `used_memory` (what Redis thinks it allocated) is lower than `used_memory_rss` (what the OS reports). The ratio `mem_fragmentation_ratio = rss / used` is reported in `INFO memory`. A ratio of 1.05 to 1.5 is normal; above 1.5 is wasteful; above 2 is alarming. **Active defrag** (`activedefrag yes`) makes Redis copy live allocations to compact memory in the background. Turn it on for any long-running production instance.

A practical big-key audit:

```python
big_keys = []
for k in r.scan_iter(count=1000):
    size = r.memory_usage(k, samples=0) or 0   # samples=0 = exact, slow
    if size > 1_000_000:
        big_keys.append((size, r.type(k), k))
big_keys.sort(reverse=True)
for size, t, k in big_keys[:20]:
    print(f"{size:>12} B  {t:<10} {k}")
```

`samples=0` means "walk every element" and is expensive on large hashes/sets — do this on a replica or during low traffic. The output regularly surprises people. Big keys are usually one of: a hash that should have had a per-shard breakdown, a list someone forgot to trim, or a set built around a very popular tag.

Two more memory levers worth knowing:

- **`hash-max-listpack-entries`** (default 128) and **`hash-max-listpack-value`** (default 64). These thresholds determine when small hashes flip to hashtable encoding. The defaults are fine for most workloads; bumping them up to ~512/128 saves substantial memory if your hashes cluster around the boundary, at a small CPU cost on the larger end (listpack scan is O(N)).
- **`maxmemory-clients`**: caps the buffer Redis will hold for slow clients. A client that subscribes to a busy keyspace notification stream and falls behind will accumulate output buffers that count against `maxmemory`. Without this cap, that client can OOM your instance. Set it to a fraction of `maxmemory`; 10% is conservative.

## 7. Latency: Where the Milliseconds Actually Go

![End-to-end Redis latency stages and stall modes](/imgs/blogs/redis-applications-and-optimization-4.png)

I have been on the receiving end of "Redis is slow" pages perhaps fifty times in my career. Almost every one resolves to one of the stages in the diagram above. The skill is knowing how to localize, fast.

**Slowlog first, always.** Redis maintains an in-memory log of commands that exceeded `slowlog-log-slower-than` microseconds (default 10000 µs = 10 ms). `SLOWLOG GET 50` shows the last 50, with timestamps, durations, the full command, and the client identity. Lower the threshold to 1000 µs on a quiet instance to widen the net during debugging. The first real lookup almost always reveals the culprit: a `KEYS *` from a forgotten admin script, an `HGETALL` on a runaway hash, an `SMEMBERS` on a 1M-element set.

**`LATENCY DOCTOR`** is the Redis-internal advisor. It correlates events from the `LATENCY` framework — fork stalls, AOF stalls, expired-keys-cycle stalls, fast-command outliers — and prints a human-readable narrative. Run it after a known incident and it will often hand you the answer. `LATENCY HISTORY event-loop` and `LATENCY LATEST` are the raw underlying commands.

**`redis-cli --intrinsic-latency 10`** measures the OS-level latency of the box, independent of Redis. It busy-loops for 10 seconds reading the clock and reports the worst gap. On a healthy modern Linux box this is sub-100 µs. Above 1 ms suggests host contention (CPU steal, THP defrag, hyperthreading neighbor noise).

**`redis-cli --latency`** does the inverse: from the client side, send PINGs and report worst-case round trip. The difference between intrinsic and external latency is what the network and Redis itself are doing.

A short list of commands to ban or use carefully:

- `KEYS pattern`: O(N), blocks the loop. Use `SCAN` with a cursor.
- `SMEMBERS bigset`, `HGETALL bighash`, `LRANGE list 0 -1`: O(N) read + large reply. Use `SSCAN`/`HSCAN`/`LRANGE 0 99` with paging.
- `DEL bigkey`: O(N) free. Use `UNLINK bigkey` (Redis 4+) which moves the free to a background thread.
- `FLUSHDB`/`FLUSHALL`: O(N). Use `FLUSHDB ASYNC` / `FLUSHALL ASYNC`.
- `EVAL` of a long Lua script: O(script duration). Break into smaller pieces or use `redis.breakpoint()` patterns to yield (Redis 7 Functions).
- `MIGRATE`: blocks both source and destination per key; measure before automating.

**Pipelining** is the single biggest latency win for client code. Without it, every command is a round trip; with it, dozens or hundreds of commands fly out before the first reply lands. A simple benchmark:

```python
import time, redis
r = redis.Redis()

n = 10000
t0 = time.time()
for i in range(n):
    r.set(f"k{i}", i)
print(f"sequential: {time.time()-t0:.3f}s")

t0 = time.time()
pipe = r.pipeline(transaction=False)
for i in range(n):
    pipe.set(f"k{i}", i)
pipe.execute()
print(f"pipelined : {time.time()-t0:.3f}s")
```

On a localhost benchmark you might see 0.6 s vs 0.04 s — a 15x speedup. Across an AZ-local network it is closer to 50-100x. The reason is simple: latency dominates. With pipelining, you pay one network RTT to push N commands instead of N RTTs.

`MULTI`/`EXEC` is *not* pipelining — it is a transaction. It does pipeline the commands implicitly, but it also takes a logical lock on the connection. Use `pipeline(transaction=False)` if you do not need atomicity; it is faster in tail-latency terms because Redis does not have to defer the queue.

**Client-side caching** (RESP3, Redis 6+) is a recent addition that deserves more attention than it gets. The client tells Redis "I am caching keys A, B, C" and Redis pushes invalidations when those keys change. The hit returns to the client without a round trip at all. For read-mostly hot keys, this is a 10-100x latency win. The downside is that your client now has cache-coherence concerns; libraries like `redis-py` (with `client_tracking=True`) and `lettuce` (in JVM) handle the bookkeeping.

A concrete walked-through latency budget for a realistic API endpoint that does five Redis operations:

| Stage | Best | Typical | Bad |
| --- | --- | --- | --- |
| Client encode + connection acquire | 50 µs | 200 µs | 5 ms (pool exhausted) |
| TCP, AZ-local | 200 µs | 500 µs | 5 ms (kernel queueing) |
| Redis queue + execute, 5 commands | 50 µs | 200 µs | 200 ms (blocked behind big-key DEL) |
| TCP back | 200 µs | 500 µs | 5 ms |
| Client decode | 50 µs | 200 µs | 1 ms (GIL wait) |
| **Total** | **0.55 ms** | **1.6 ms** | **216 ms** |

The "bad" column is the one to memorize. p99 is dominated by the worst stage, and the worst stage is almost always Redis itself (some other client's command), not the network.

### 7.1 Observability beyond `INFO`

`INFO` is the thirty-thousand-foot view. The fields worth alerting on, with rough thresholds:

| Field | Alert if | Why |
| --- | --- | --- |
| `used_memory_rss / used_memory` | > 1.5 sustained | fragmentation; turn on activedefrag |
| `instantaneous_ops_per_sec` | sudden 5x change | traffic shape changed; correlate with deploy |
| `connected_clients` | climbing without bound | client leak |
| `evicted_keys` rate | non-zero when not expected | memory pressure |
| `expired_keys` rate | sudden 0 | active-expiry stalled |
| `aof_pending_bio_fsync` | sustained > 0 | disk cannot keep up with appendfsync |
| `master_repl_offset - slave_repl_offset` | growing | replica falling behind |
| `latest_fork_usec` | > 200000 (200 ms) | fork stalls becoming user-visible |
| `mem_clients_normal` | climbing | output buffers piling up |

Beyond `INFO`, the underused `LATENCY` framework is your friend during incidents. `LATENCY HISTORY fork` shows fork stalls over time; `LATENCY HISTORY aof-fsync-always` shows fsync stalls; `LATENCY RESET` clears the in-memory log. Wire these into your Prometheus/Datadog scrape and you stop being surprised.

The `MONITOR` command is the nuclear option: it streams every command executed by Redis to your terminal in real time. It is also wildly expensive — Redis serializes the full command plus client info for every single op. Never run `MONITOR` against a busy production primary. Run it against a replica, or use a sampling proxy like `redis-cli --bigkeys --i 0.01`.

For long-term capacity planning, sample `INFO memory` and `INFO commandstats` hourly. The latter is gold: it tells you the total time spent and the call count for each command. If one team's `EVAL` is 40% of total CPU, you know where to optimize. If `CLUSTER COUNTKEYSINSLOT` shows up as expensive, somebody is running a slot-distribution audit too often.

### 7.2 Tracing and per-command attribution

The hardest debugging case is "p99 went up by 4 ms two days ago and we cannot tell why." Single-instance metrics show the elevated latency but not which client or command caused it. Three useful techniques:

1. **Client-side spans**. Have every service that touches Redis emit a span per command with the command name as a tag. In OpenTelemetry, `redis-py`'s instrumentation does this for you. Aggregating by command name across services tells you who is doing what.
2. **`CLIENT LIST` snapshots**. Every minute, run `CLIENT LIST` and bucket by `name=` (clients that bothered to set a name). The `cmd=` field tells you what each client is doing right now. This is poor man's profiling but it works.
3. **`commandstats` deltas**. `INFO commandstats` is cumulative since boot. Capture two samples a minute apart; subtract; the top-10 by `usec` is the dominant work in that minute. Combined with deploys log, you can pinpoint when a command profile changed.

The thing to internalize: in a single-threaded server, *time is the only currency*. Two sources of work compete: yours and somebody else's. The diagnostic question is always "where did the time go" and the answer is always one of "in queue waiting" or "executing."

## 8. The Optimization Playbook

This section is a condensed reference. Each item is something I have actually used to shave real latency or memory on a production instance.

**Pipelining and connection management.**

- Always use a connection pool. Pool size = expected concurrent in-flight requests. For Python with `redis-py`, the default is 50 per pool; size it explicitly.
- Pipeline anywhere you have ≥3 sequential operations against Redis. The cost of doing it (a few extra lines) is dominated by the savings.
- For very high QPS clients, prefer fewer big pipelines over many tiny ones. The CPU cost of `EXEC` on the server dominates at the limits.
- If you are using `redis-py` in async, check that you are using `redis.asyncio.Redis`, not `redis.Redis` wrapped in a thread pool. The latter is a common source of "Redis is slow" reports that turn out to be GIL contention.

**Hot-key mitigation.**

- Identify with `redis-cli --hotkeys` (requires `maxmemory-policy` to be LFU-based).
- Mitigations, in order: (1) cache one level closer (in-process LRU, even with a small TTL); (2) shard the hot key by appending a suffix (`counter:42:0`, `counter:42:1`, ..., random-write, `MGET`-read); (3) use client-side caching with invalidations; (4) move it out of Redis entirely if it is genuinely hot enough.

**Big-key mitigation.**

- Audit with `redis-cli --bigkeys` periodically.
- If you cannot avoid creating one, use `UNLINK` for deletion and `SCAN`-family cursors for iteration. Never `DEL` or `KEYS` on production.
- Streams: always set `MAXLEN ~ N` on every `XADD`. Without it, an unattended stream is a memory leak.
- Lists: bound them. `LPUSH` followed by `LTRIM 0 999` keeps the most recent 1000 entries.

**Memory.**

- Turn on `activedefrag yes` for long-running instances.
- Tune `hash-max-listpack-entries`/`hash-max-listpack-value` if your workload clusters around the threshold.
- Set `maxmemory-policy` deliberately (default is `noeviction`, which surprises people). If Redis is a cache, pick `allkeys-lfu` unless you have a specific reason not to.
- Set `maxmemory-clients` to a fraction of `maxmemory` to bound slow-client buffers.

**Persistence.**

- `aof-use-rdb-preamble yes`.
- `repl-backlog-size 256mb` (or higher) on busy primaries.
- Disable RDB on primaries; let replicas snapshot. Move the fork stall off the critical path.
- If you must run AOF on the primary, watch for "Asynchronous AOF fsync is taking too long" messages — that is your disk telling you it cannot keep up.

**Cluster.**

- 3 primaries minimum. Each with at least 1 replica. Cross-AZ replicas if your failure model includes AZ failure.
- Set `cluster-require-full-coverage no` in degraded scenarios (otherwise a single dead shard takes down all writes).
- Use hashtags carefully. The temptation to "just put everything in `{tenant}:`" creates per-tenant hot shards.

**Kernel and host.**

- `echo never > /sys/kernel/mm/transparent_hugepage/enabled`. THP causes random latency spikes on Redis specifically because of how it allocates many small objects.
- `vm.overcommit_memory = 1`. Otherwise `fork()` can fail under memory pressure exactly when you most need it.
- `net.core.somaxconn = 65535` and `net.ipv4.tcp_max_syn_backlog = 65535`. The default 128 is a connection-accept bottleneck under burst.
- Keep the kernel and `tcp-keepalive` settings sane (60s is usually fine).
- Pin Redis to a specific CPU set with taskset/cgroups if you are sharing the host with noisy neighbors. The single thread is a single core's worth of work, but it really wants that core.

**Client-side caching (RESP3).**

```python
import redis
pool = redis.ConnectionPool(host="...", protocol=3)
r = redis.Redis(connection_pool=pool, client_tracking=True)
# subsequent r.get("hot_key") returns from in-process cache after first miss
```

The numbers I have seen in production: hot-key reads at 5-10 µs (in-process) instead of 200-500 µs (round trip). For workloads that read the same 100 keys at high QPS, this is the single biggest perf win available short of architectural changes.

## 9. Eight Production Incidents

These are composites of incidents I have either run or watched closely. Names and numbers are anonymized but the mechanics are real.

### 9.1 The Midnight Stampede

A high-traffic e-commerce site cached product cards in Redis with `EX 600`. A scheduled job at 02:00 invalidated all product caches when the daily catalog imported. Twenty seconds later, the database CPU went to 100% and stayed there for eleven minutes; p99 latency on the API spiked from 80 ms to 4.2 s.

The post-mortem identified two compounding bugs. First, every product cache was written at roughly the same minute during the previous afternoon's deploy, so they all expired in the same 60-second window. Second, the cache-fill code had no single-flight; the first 30,000 requests for the same product all missed and all hit the database. The DB's connection pool, sized for normal load, couldn't keep up; new requests piled up in the connection queue, and the pile-up itself created retries upstream.

The fix was three lines. Add `+ random(0, 60)` jitter to every cache `SET`. Wrap cache fills in a `SET NX EX 30` lock; losers wait briefly and re-read. Add a circuit breaker around the DB so retries do not amplify the stampede. The next deploy, the catalog import barely moved the database CPU.

The lesson: a cache without jitter and without single-flight is an amplification mechanism, not a protection. The day everything works fine is the day you do not notice it. The day a deploy or an event aligns expiries is the day the lack of protection fires.

### 9.2 The 12 GB Hash

A user-events service stored every user's recent events as a Redis hash, keyed by `user:<id>:events`, with the hope that a TTL would naturally bound it. Several months in, the team noticed sporadic 4-second response times on what should have been microsecond operations. `SLOWLOG GET 20` revealed `DEL user:91827:events` taking 4.1 seconds.

Investigation showed that one user, a power-user who scripted against the API, had accumulated 14 million event entries in their hash. The TTL never fired because every new event reset the TTL. The hash had long since flipped from listpack to hashtable encoding and was using 12 GB of memory by itself.

The first fix was operational: `UNLINK user:91827:events` instead of `DEL`, which moves the free to a background thread and avoids the loop stall. The second was structural: cap every per-user hash at 1000 entries with a bounded `XADD MAXLEN`-style invariant (we migrated to streams). The third was preventive: add a daily cron that enumerates the top 100 keys by `MEMORY USAGE` and pages on anomalies.

The lesson: every per-user data structure needs an explicit upper bound. "TTL will save us" does not save you when the access pattern is "always touching it."

### 9.3 Redlock Under Network Partition

A workflow system used Redlock across five Redis instances to ensure exactly-once execution of a billing job. During an AWS availability-zone partial outage, the network between two of the five Redis instances became flaky. Two instances of the workflow worker each acquired the lock — one against three of the Redis nodes, the other against three others, with one node in both quorums but unreachable from one client.

The result was duplicate billing for several thousand customers over a 40-minute window before the system recovered.

The root cause was that Redlock's correctness argument assumes bounded clock drift and bounded process pauses. In an AZ partition with packet loss, a worker that "had the lock" for some wall-clock time could be paused (by GC, by the network) past its lock expiration without knowing it. A second worker, racing with a fresh acquisition attempt, could legitimately acquire the lock. Both think they hold it; both proceed.

The fix was fencing tokens. The lock acquisition returns a monotonically increasing integer (we used a Redis `INCR` on a separate counter). The billing system writes the token on every charge attempt; the database refuses any operation with a token less than the most recent it has seen. With fencing, the duplicate worker's writes get rejected by the database, the lock service is back to being an "efficiency" optimization, and the system is correct regardless of pause behavior.

The lesson: lock services optimize for happy-path. Correctness in the unhappy path requires the resource itself to enforce ordering, via fencing tokens or via optimistic concurrency.

### 9.4 AOF Disk Full

A small staging cluster ran on cheap EBS volumes with `appendfsync everysec`. During an unrelated traffic spike, the AOF rewrite kicked in. The rewrite child wrote at the maximum rate the volume could support; the parent's normal AOF appends queued behind the rewrite's I/O.

The Redis log filled with `Asynchronous AOF fsync is taking too long`. Write commands started taking 200-800 ms. Investigation showed the EBS volume's burst budget had been exhausted and it was throttled to 100 IOPS. AOF essentially blocked on every fsync for the duration of the rewrite (about 12 minutes), during which the application's p99 was destroyed.

The fix was provisioning: move the AOF target to a higher-IOPS volume, and disable `no-appendfsync-on-rewrite` (i.e. allow Redis to skip fsync during rewrite to dodge contention, accepting the rewrite-window loss risk on staging only). The deeper fix was a runbook: monitor `aof_pending_bio_fsync` in `INFO`. When that climbs, you are seconds away from fsync stalls.

The lesson: cloud disks lie about their throughput capabilities. Test your fsync behavior under sustained load, not in steady-state. Your fsync policy is only as good as the worst minute the disk gives you.

### 9.5 Cluster Slot Migration During Black Friday

A team scaled their cluster from 6 to 12 primaries on the morning of Black Friday under high traffic. The slot migration, which they had practiced in staging, ran fine for the first thousand slots and then slowed. By midmorning, only 3 of 8192 migrating slots were moving per minute, and the entire site's p99 had climbed to 1.2 s.

The cause was twofold. First, the application's Redis client library was older and did not implement `ASKING` correctly; mid-migration commands were doubling round trips because of MOVED chains. Second, the migration tool was running synchronously, slot by slot, which under high write traffic took longer per slot than the steady-state rate.

The mitigation was to pause the migration, upgrade the client library on a fraction of the fleet to confirm the fix, then proceed with a parallel migration script during a quieter window late that night. The day's revenue was saved but the team learned to never rebalance under peak traffic.

The lesson: migrations are operationally expensive. They are not for peak hours. Your client library's cluster-awareness is part of your migration plan; if it does not handle ASK/MOVED gracefully, fix that *before* you migrate.

### 9.6 The Pub/Sub Message Loss

A team built a real-time leaderboard update system using `PUBLISH score_update <user>:<score>`. It worked perfectly in development. In production, customers complained that scores updated five minutes later than expected. Investigation showed that subscribers occasionally missed updates entirely — worse, in a way that was non-deterministic.

The cause: pub/sub is fire-and-forget. When a subscriber's TCP socket fills (say because the consumer's process is briefly slow), Redis disconnects the slow subscriber. Default `client-output-buffer-limit pubsub` was `32mb 8mb 60`, meaning if 8 MB of buffered messages stayed in Redis for 60 seconds, the slow subscriber was disconnected. Disconnected subscribers reconnect — and miss every message published during the gap.

The fix was switching to streams with consumer groups. `XADD leaderboard_events MAXLEN ~ 100000 ...` plus `XREADGROUP` with `XAUTOCLAIM` for stuck consumers gave us at-least-once delivery. The downside was cost: streams use more memory than pub/sub. The upside: customers stopped complaining.

The lesson: pub/sub is a notification firehose, not a queue. If your consumers care about every message, you do not want pub/sub.

### 9.7 LRU vs LFU Mix-Up

A configuration-loading service used Redis as a hot cache for a few hundred frequently-accessed config keys, plus a much larger set of rarely-accessed audit-log lookups. The team configured `maxmemory-policy allkeys-lru` and called it done.

For weeks, every Tuesday between 02:00 and 04:00, the application would briefly serve stale defaults to a few percent of users. The pattern matched a nightly compliance scan that walked the entire audit log for daily reporting. The scan touched millions of "cold" audit-log keys, marking them as recently-used in LRU. The hot config keys, accessed thousands of times per second normally but not within the scan window, fell out of the LRU's "recent" partition and got evicted.

Switching to `allkeys-lfu` with `lfu-log-factor 10` solved the problem within an hour. LFU correctly weights frequency over recency; the once-a-day scan touches each audit key once, while the config keys are touched thousands of times. The hot config keys stayed hot.

The lesson: LRU is the right policy when access is roughly uniform; LFU is the right policy when access is heavily skewed. The default `noeviction` is right for none of these. Pick deliberately.

### 9.8 The Lua Script That Ate the Event Loop

A team wrote a Lua script to atomically update inventory across a product hierarchy: parent SKU, variant SKUs, location-bound stock counts. The script worked correctly. In a stress test, it was fast. In production, every hour or so, every Redis client would see 200-400 ms p99 latency for two seconds, then return to normal.

The cause was a single product with 7000 variants. The Lua script iterated over all variants of a parent SKU. For most products with 2-20 variants the script ran in under 1 ms. For the one outlier product, the script ran for 200 ms. While it ran, every other client — the 50,000 QPS of normal traffic — waited.

The fixes happened in stages. Immediate: cap the script with a guard `if #variants > 100 then return redis.error_reply('too many variants') end`, surface the failure to the application, and have it fall back to a non-atomic path for that product. Medium-term: split the inventory model so a single Lua script never operated on more than 50 entities. Long-term: move very large hierarchical inventory operations out of Redis into a system designed for them.

The lesson: any unbounded loop in a Lua script is a tail-latency bomb. Bound the work, or do not put it in a script.

### 9.9 The Hot Tenant That Took Down a Shard

A multi-tenant SaaS used hashtags so that every key for a given customer lived on one shard: `{tenant_alpha}:users:1`, `{tenant_alpha}:sessions:abc`, and so on. The intent was good — atomic multi-key operations per tenant, no cross-shard transactions. The unintended consequence was that one large customer (a public retailer running a flash sale) generated 70% of the cluster's QPS and routed all of it to a single shard.

The shard's CPU saturated. Latency for *every* tenant on that shard climbed to seconds. The other five shards in the cluster were 8% utilized.

The mitigation in the moment: spin up a second cluster, migrate the hot tenant's keys to it via a dual-write window, and route the tenant's traffic explicitly. The strategic fix: drop the hashtag for the high-cardinality structures (sessions, events) where atomicity within a tenant was not actually required, and keep hashtags only for the few entities that needed cross-key atomicity. The cluster's load became roughly even within a week.

The lesson: hashtags are a knife. They give you cross-key atomicity at the cost of pinning all those keys to one shard. If a tenant grows large, that pin becomes a hot spot. Prefer fine-grained sharding by default; reach for hashtags only at the specific entities that require atomicity.

### 9.10 Active Defrag Saved 30% of RAM

A trading-data cache had been running for nine months without restart. `used_memory` was 42 GB; `used_memory_rss` was 71 GB. The instance was on a 96 GB host and the team was preparing to scale up because they thought they were close to memory limits.

We turned on `activedefrag yes` (with `active-defrag-ignore-bytes 100mb`, `active-defrag-threshold-lower 10`, `active-defrag-threshold-upper 100`, `active-defrag-cycle-min 5`, `active-defrag-cycle-max 75`). Over 6 hours, RSS dropped from 71 GB to 49 GB. CPU usage rose by about 4% during the defrag (well within the configured cycle limits). The team did not need the scale-up.

The lesson: long-lived Redis instances with churn (TTL expiry, eviction, mixed allocation sizes) accumulate fragmentation. Active defrag is nearly free and pays for itself within hours of being turned on. Make it a default in your provisioning playbook.

## 10. When to Reach for Redis — and When Not To

The most useful framing I have for this question: Redis is the right answer when you need *one of these specific properties* and the dataset fits in RAM. It is the wrong answer when you need *durability or relational queries* and you only think you need Redis because somebody on the team already knows it.

**Reach for Redis when:**

- You need sub-millisecond reads/writes on a small-to-medium dataset.
- You need atomic operations on a data structure (counter, set, list, sorted set) that would be a transaction-heavy mess in a relational database.
- You need a fast, distributed primitive — locks, queues, leaderboards, rate limiters, idempotency keys.
- You need ephemeral data with TTLs — sessions, ephemeral state, recent-N caches.
- You need an exact pub/sub firehose for fire-and-forget notifications, or a durable but in-memory queue (streams).

**Do not reach for Redis when:**

- You need durability that survives the worst-second of write loss. Use Postgres, MySQL, or a system whose primary-replica is synchronous. Even the best Redis configuration loses something under fault scenarios.
- You need cross-key transactions that span shards. Redis Cluster does not support them. Use a relational database.
- You need queries on values, not just keys. Redis indexes on the key only. RediSearch helps but at that point ask if you should be using OpenSearch or Postgres + GIN.
- Your dataset is bigger than you can pay to keep in RAM. The crossover is usually around 200-500 GB depending on cloud-RAM prices; beyond that, an SSD-backed system wins on cost-per-byte.
- You need multi-master writes across regions with conflict resolution. CRDT-aware managed Redis (Redis Enterprise) exists but it is a different beast and you should know exactly what consistency model you are buying.

**Comparison with neighbours:**

| System | Sweet spot vs Redis |
| --- | --- |
| Memcached | Pure cache, no data structures, multi-threaded. Wins when you only need GET/SET and want one fewer thing in your head. |
| KeyDB / Valkey / Dragonfly | API-compatible Redis with multi-threading. Worth evaluating if Redis CPU is your bottleneck. Dragonfly in particular has very different memory characteristics. |
| DynamoDB / Aurora | Durability you can stake the business on, at the cost of latency 5-50x higher. |
| Postgres | The right home for relational data and `LISTEN/NOTIFY` for low-volume pub/sub. Use Postgres for the data; use Redis for the cache. |
| Kafka | The right home for "very high throughput durable log." Streams is the in-memory cousin; Kafka is the on-disk parent. |
| etcd / Consul | Strongly-consistent KV with watch. The right answer when you need consensus. Redis is much faster but offers no consensus guarantees. |

The habit I recommend: when somebody on the team says "let's put it in Redis," ask three questions. (1) What is the eviction story if memory grows to N times what we expect? (2) What is the durability story under failover? (3) Is there a Postgres table that could be the source of truth, with Redis as the cache?

If the answers are "we have one," "we know what we lose," and "yes," you are probably right. If any answer is hand-wavy, you are about to write your own incident write-up. The patterns and pitfalls in the rest of this article are just the long version of those three questions.

The good news: once you are in the loop on the three questions and you have configured your instance with the playbook above, Redis genuinely *is* the bargain it advertises. A well-tuned Redis at 50,000 QPS with sub-millisecond p99 is one of the most cost-effective pieces of infrastructure you can run. The cost is upfront thinking; the reward is years of boring operations.

For deeper reading on adjacent topics on this site, the [database connection pooling](/blog/software-development/database/database-connection-pooling) post covers the client-side budget that gates how much Redis throughput you can actually use; [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) is a useful reference on a related "the obvious choice has hidden costs" pattern; the [design patterns guide](/blog/software-development/system-design/design-patterns-guide) lays out the cache-aside / write-through patterns more abstractly; and [KV cache for LLMs](/blog/machine-learning/large-language-model/kv-cache) is the same shape of problem one layer up the stack — a cache that, if you mismanage it, takes the whole system down.

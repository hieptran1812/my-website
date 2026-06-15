---
title: "Design a URL Shortener: The Staff-Level Version of the Classic Question"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Take the canonical interview warm-up to staff depth: distributed key generation, a cache-everything read path, async click analytics, 301 versus 302, hot-key survival, and the trade-off matrix that defends every decision."
tags:
  [
    "system-design",
    "url-shortener",
    "caching",
    "architecture",
    "distributed-systems",
    "scalability",
    "key-generation",
    "rate-limiting",
    "case-study",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/design-a-url-shortener-1.webp"
---

"Design a URL shortener" is the interview question everyone has seen, and that is exactly why it is dangerous. Because it looks easy, candidates rush past it: a table of `short_code → long_url`, a redirect endpoint, done in five minutes. And that answer is not wrong — a URL shortener really is one of the simplest useful systems you can build. But the simplicity is a trap. The interesting decisions are all hiding inside the parts the rushed answer skips: how do you generate billions of unique short codes without two machines ever colliding; how do you serve a hundred thousand redirects a second without the database melting; what happens when one shortened link goes viral and a single key gets fifty thousand requests a second; and how do you count every click without the counting itself slowing down the redirect that users actually feel.

A junior answers the question. A senior answers the *follow-up questions* — and a staff engineer anticipates them before they're asked. The gap between "it works on a laptop" and "it works at billions of redirects with a five-nines redirect SLO and an analytics pipeline that doesn't lie" is precisely the gap this post is about. We are going to design a URL shortener the way you would actually design one in a real architecture review: frame the requirements, do the back-of-envelope math, sketch the API, draw the architecture, then spend most of our time in the genuinely hard parts where senior judgment shows.

The architecture we land on is in Figure 1. It has two paths that look almost unrelated and that is the whole point. The write path — someone shortens a URL — is rare, durable, and can afford to be careful. The read path — someone clicks a short link — is constant, latency-critical, and must be cache-fronted to the point where the database barely participates. Get that split right and everything else follows.

![A two-path architecture diagram showing a cache-fronted read path through CDN and Redis to Postgres, and a write path that appends click events to a queue feeding an analytics store](/imgs/blogs/design-a-url-shortener-1.webp)

By the end you will be able to defend every box in that diagram with a number, name the failure mode each component is there to prevent, and articulate — out loud, in a review — why you chose a counter over a hash, a 302 over a 301, and a queue over a synchronous write. That ability to *defend* the design under pressure is what the question is really testing. The mapping table is just the cover story.

## What we are actually building, and what we are not

Before any boxes, scope. A senior's first move on an ambiguous prompt is to turn the vague ask into a concrete list of what the system must and must not do — the discipline covered in [how seniors approach ambiguous system design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems) and [turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos). Scope is where you win or lose a design review, because an unscoped design is impossible to evaluate.

**Functional requirements** — what the system does:

- **Shorten**: given a long URL, return a short URL on our domain (e.g. `https://sho.rt/aZ3x9Q`).
- **Redirect**: given a short URL, redirect the browser to the original long URL, fast.
- **Custom aliases**: let a user request a specific code (`sho.rt/my-launch`) instead of a generated one.
- **Expiration**: let a link have an optional time-to-live, after which it stops resolving.
- **Analytics**: count clicks per link, ideally with some breakdown (referrer, country, time).

**Non-functional requirements** — the properties that make it production-grade:

- **Low-latency redirect**: the redirect is the user-facing hot path. Target p99 under 50 ms server-side, and effectively instant from a CDN edge.
- **High availability for reads**: a redirect failing is a broken link on someone else's site. Reads must be far more available than writes. We'll target five nines (≈ 99.999%) on the redirect path and accept three or four nines on the create path.
- **Uniqueness**: no two long URLs ever map to a code that's already taken. This is a correctness invariant, not a nicety — a collision corrupts someone's link.
- **Scalability**: handle billions of stored links and tens of thousands of redirects per second, with headroom to grow.
- **Unpredictable codes (security)**: sequential, guessable codes let anyone enumerate every link in the system. For a public shortener that is a real privacy leak.

**Explicitly out of scope** (saying this out loud is a senior move — it bounds the conversation): we are not building link-in-bio pages, not doing malware scanning of destination URLs beyond a basic blocklist, not offering deep marketing-campaign attribution, and not supporting editing a link's destination after creation (that turns an immutable mapping into a mutable one and changes the caching story entirely; we'll note where that bites).

**The one metric that matters.** If you make me pick a single number to optimize, it is **redirect p99 latency at the cache-hit path**, because that is the number every end user feels on every click, billions of times a day. Everything else — create latency, analytics freshness, storage cost — is secondary to making the redirect feel instant. Naming the one metric early disciplines every later trade-off: when two options are otherwise tied, the one that protects redirect latency wins.

## Back-of-the-envelope: the numbers that shape the design

You cannot design this system without first knowing its scale, and the scale will surprise you in a specific way: **it is overwhelmingly a read system, and the write side is tiny enough to fit one database for years.** That single fact, derived from arithmetic, dictates the entire architecture. Let me walk the math the way I'd do it on a whiteboard. For the full estimation workflow — peak factors, the latency ladder, storage-per-row reasoning — see [back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design); here I'll apply it.

#### Worked example: sizing a public URL shortener

Assume a mid-sized public shortener: **100 million new short URLs created per month.** That is the write volume. Let's convert it.

**Write QPS.** A month is about 2.6 million seconds (30 × 86,400 ≈ 2.59M). So:

```
100,000,000 writes / 2,600,000 s ≈ 38 writes/sec average
```

Round to **~40 writes/sec average**. Apply a peak factor of 3–5× for daily and weekly cycles and call peak **~150–200 writes/sec**. That is nothing. A single Postgres instance handles thousands of inserts per second without blinking. **The write side is not the problem.** Internalize that — most candidates over-engineer the write path and under-engineer reads.

**Read QPS.** Shorteners run a read-to-write ratio between 10:1 and 1000:1 depending on the audience; the classic working assumption is **100:1**. At 100:1:

```
40 writes/sec × 100 = 4,000 reads/sec average
```

Apply the same 3× peak factor → **~12,000 reads/sec at peak**, and budget for spikes well above that when a single link goes viral (we'll stress-test that case). So the read path must comfortably do tens of thousands of redirects per second. That is the dominant load and the thing worth engineering. Figure 6 collects these numbers.

![A capacity matrix listing write QPS, peak read QPS, five-year row count, mapping storage size, and hot-cache RAM with notes showing the system stays read-dominated and fits a single node](/imgs/blogs/design-a-url-shortener-6.webp)

**Storage.** Each row is roughly: short code (7 bytes), long URL (assume an average of ~100 bytes, capped at ~2 KB), creation timestamp (8 bytes), creator/owner id (8 bytes), expiry (8 bytes), plus index overhead. Round generously to **~500 bytes per row** including indexes. Over five years:

```
100M/month × 12 × 5 = 6 billion rows
6,000,000,000 × 500 bytes ≈ 3 TB
```

**Three terabytes for the core mapping over five years.** That fits on a single large database node with room to spare (modern instances take many terabytes of SSD). This is the headline result: **you do not need to shard the mapping table for years.** Anyone who opens with "we'll shard across a Cassandra cluster" has skipped the arithmetic that says one box suffices. We will, of course, discuss when sharding *does* become necessary — but designing for it on day one is premature distribution, one of the most expensive mistakes in the field. (When you do shard, [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) is the playbook.)

**Cache size.** The redirect read path is what we cache. By a Pareto-style access pattern, the small fraction of links that are "active" gets nearly all the traffic. Suppose the hot working set is the **10 million most-clicked links** at any time. Each cache entry is roughly the code plus the long URL plus overhead — call it ~500 bytes. Then:

```
10,000,000 × 500 bytes ≈ 5 GB
```

**Five gigabytes of cache holds the entire hot set.** That fits in RAM on a single Redis node, with replicas for availability. The implication is enormous: **almost every redirect can be served from memory**, and the database only ever sees cold-miss traffic. This is the single most important sentence in the whole design.

**Bandwidth.** Redirects are tiny — a 302 response is a few hundred bytes of headers. At 12,000 reads/sec × ~500 bytes ≈ **6 MB/sec**, trivial. Bandwidth is never the constraint for a shortener; it's a metadata service, not a media service. (Contrast a [video streaming platform](/blog/software-development/system-design/design-a-video-streaming-platform), where bandwidth is the whole bill.)

**Key space.** This one is a correctness check, not a capacity check, and it's where the depth begins. With a 7-character base62 code (alphabet of 62 symbols: `0-9`, `a-z`, `A-Z`):

```
62^7 ≈ 3.5 trillion codes
```

At 100M new codes/month, you'd burn `62^7` in `3.5e12 / 1.2e9 ≈ ~2,900 months ≈ 240 years`. Seven characters is plenty for the foreseeable future. Six characters gives `62^6 ≈ 56 billion` — burned in ~47 years at our rate, also fine, and shorter is better for users. We'll standardize on **7 characters with room to grow to 8**, and revisit the exhaustion math under the stress test. The key-space arithmetic is not academic: it's the difference between a strategy that works for decades and one that starts colliding in a year.

The whole estimation story collapses to one line you should be able to say in a review: *"~40 writes/sec, ~12k reads/sec peak, 3 TB over five years that fits one box, a 5 GB hot set that fits in RAM, and a 3.5-trillion key space that lasts centuries — so this is a read-heavy, cache-everything design with no sharding for years."* That sentence is the design.

## The API: small surface, sharp contracts

The API is deliberately tiny — three endpoints — but the contracts carry weight. The status codes and idempotency semantics are where seniors and juniors diverge.

```http
POST /api/v1/urls
Content-Type: application/json
Authorization: Bearer <token>

{
  "long_url": "https://example.com/some/very/long/path?with=params",
  "custom_alias": "my-launch",        // optional
  "expires_at": "2027-01-01T00:00:00Z" // optional
}

201 Created
{
  "short_url": "https://sho.rt/aZ3x9Q",
  "code": "aZ3x9Q",
  "long_url": "https://example.com/some/very/long/path?with=params",
  "expires_at": null
}
```

```http
GET /aZ3x9Q

302 Found
Location: https://example.com/some/very/long/path?with=params
Cache-Control: private, max-age=0
```

```http
GET /api/v1/urls/aZ3x9Q/stats
Authorization: Bearer <token>

200 OK
{
  "code": "aZ3x9Q",
  "clicks_total": 184203,
  "clicks_last_24h": 5120,
  "top_referrers": [{"host": "twitter.com", "clicks": 90211}]
}
```

A few contract decisions worth defending:

- **The redirect lives at the root path**, `GET /aZ3x9Q`, not under `/api/`. Short URLs need to be short; a `/api/v1/redirect/` prefix would defeat the purpose. This forces a routing rule: anything matching the code pattern at the root is a redirect; everything else is a real route. Reserve a small set of paths (`/api`, `/login`, `/health`, `/favicon.ico`) so a user can't claim them as custom aliases.
- **Create should be idempotent on `long_url` per user**, optionally. If the same user shortens the same URL twice, returning the existing code (rather than minting a new one) saves key space and surprises no one. Make this a flag; some products deliberately mint a fresh code each time for per-campaign tracking. The mechanics of safe retries are in [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design).
- **Custom alias conflicts return `409 Conflict`**, not a silently-different code. The caller asked for a specific name; if it's taken, tell them.
- **Auth on create, none on redirect.** Creating consumes key space and is a vector for abuse, so it's authenticated and rate-limited. Redirecting is public by nature — that's the whole point of a short link.

For the broader REST-versus-gRPC-versus-GraphQL reasoning behind shaping these contracts, see [API design: REST, gRPC, GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql). For a shortener, plain REST over HTTP is exactly right: the redirect *is* an HTTP primitive.

## High-level architecture: two paths, one truth

Look again at Figure 1. The architecture has a **write path** and a **read path**, and they share exactly one component: the source-of-truth database holding the `code → long_url` mapping. Everything else is asymmetric, because the loads are asymmetric.

**Write path** (rare, durable, careful): client → API service → generate a unique code → insert `(code, long_url, owner, expiry)` into the primary database → return the short URL. Because writes are only ~40/sec, this path can afford a synchronous, transactional, single-primary design. No queue, no eventual consistency, no cleverness. Durability and uniqueness are the only goals.

**Read path** (constant, latency-critical, cache-everything): client → CDN edge → application cache (Redis) → database on miss. A redirect that hits the CDN never touches our infrastructure at all. A redirect that misses the CDN but hits Redis is served in ~1 ms. Only a cold miss — the long tail of rarely-clicked links — reaches the database. Figure 3 traces this path layer by layer.

![A read-path pipeline showing a client GET hitting the CDN edge, falling through to Redis on a miss, then Postgres on a deeper miss, with hit rates and per-layer latencies annotated](/imgs/blogs/design-a-url-shortener-3.webp)

**The analytics path** (high-volume, can-be-late): every click also needs to be *counted*, and counting is where the real write load lives. If 12,000 redirects/sec each triggered a synchronous database write to increment a counter, the analytics writes would dwarf the create writes by 300× and would block the redirect itself. So clicks do not write to the database inline. Instead the edge or app emits a lightweight click event to a queue (Kafka), and a separate consumer aggregates clicks asynchronously. The redirect returns the moment it has the destination; the click is counted later. This decoupling — covered next in its own deep dive — is the difference between a redirect that's fast and one that isn't.

Three components, three completely different design pressures. A senior names the pressure on each: **write path = correctness**, **read path = latency**, **analytics path = throughput-without-blocking**. Conflating them — for example, writing analytics synchronously on the redirect — is the most common way this design goes wrong.

It's worth pausing on *why* this asymmetry is the central design idea and not just a tidy diagram. In most systems, reads and writes share infrastructure because their loads are comparable; you size one tier and it handles both. A shortener violates that assumption by two-to-three orders of magnitude — 40 writes against 12,000 reads against (counting analytics as writes) another 12,000 click-events — so a single uniform tier would be wrong for every workload at once. You'd over-provision durability for reads that never mutate state, and under-provision throughput for clicks that don't need durability at all. The senior instinct is to notice when a workload's read/write ratio is extreme and *split the path*, sizing each independently. That instinct generalizes far beyond shorteners: any read-dominated system (product catalogs, DNS, configuration distribution, feature flags) wants the same shape, and any write-dominated system (logging, metrics ingestion, the news feed in the next post) wants its mirror image.

## Data model

The core is one table. Resist the urge to over-normalize a system this read-heavy.

```sql
CREATE TABLE urls (
    code         VARCHAR(8)   PRIMARY KEY,      -- the short code, base62
    long_url     TEXT         NOT NULL,
    owner_id     BIGINT       NOT NULL,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ,                   -- NULL = never
    is_custom    BOOLEAN      NOT NULL DEFAULT false
);

-- Lookups by owner for a "my links" dashboard.
CREATE INDEX idx_urls_owner ON urls (owner_id, created_at DESC);

-- Lazy-expiry sweep finds expired rows.
CREATE INDEX idx_urls_expiry ON urls (expires_at) WHERE expires_at IS NOT NULL;
```

The primary key is the `code` itself, because the redirect — the hot path — looks up by code. That gives the most frequent query a direct primary-key hit with no secondary index. The `long_url` is `TEXT` (cap it at the app layer to ~2 KB; browsers choke past that anyway). `owner_id` supports the dashboard and rate-limiting-by-owner. The partial index on `expires_at` keeps the expiry sweep cheap by indexing only rows that can expire.

Analytics get their **own** store, deliberately separate:

```sql
-- Rolled-up counters, updated by the async consumer. Not on the redirect path.
CREATE TABLE click_stats (
    code           VARCHAR(8)  NOT NULL,
    bucket_hour    TIMESTAMPTZ NOT NULL,   -- hourly rollup
    clicks         BIGINT      NOT NULL DEFAULT 0,
    PRIMARY KEY (code, bucket_hour)
);
```

Why split analytics from the mapping? Because their access patterns and durability needs are opposite. The mapping is read-mostly, must be strongly consistent (a wrong destination is a broken promise), and is tiny. The clicks table is write-heavy, append-mostly, tolerant of being seconds-to-minutes stale, and grows without bound. Forcing them into one store would make the wrong trade-off for both. This is a textbook application of choosing a datastore by access pattern — the reasoning in [choosing a datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql). For the mapping, a single-node SQL database is perfect: strong consistency, transactions for custom-alias conflicts, trivial size. For raw click events you'd reach for a column store or time-series database (ClickHouse, Druid, or even a Kafka-to-warehouse pipeline), because the query pattern is analytical aggregation over time, not point lookups.

A few more data-model decisions that look small and aren't:

- **Why `code` is a `VARCHAR(8)` and not the integer.** You might be tempted to store the integer counter value and encode to base62 only at the API boundary, saving a few bytes. Resist it. Storing the rendered `code` as the primary key means the redirect's lookup — the hottest query in the system — is a direct equality match on the exact string the URL carries, with no decode step and no functional index. Bytes are not the constraint here (the table is 3 TB regardless); query simplicity on the hot path is. Optimize the data model for the query that runs a billion times a day, not the one that runs forty times a second.
- **Immutability is a feature, lean on it.** Because a mapping never changes after creation (we deliberately scoped out destination editing), the row is effectively write-once. That single property is what licenses long cache TTLs, edge-cacheable 302s, and coordination-free cross-region read replicas. The moment you allow editing a link's destination, you inherit a cache-invalidation problem across every layer — CDN, Redis, browser — and the clean read path gets muddy. If a product manager asks for editable links, the correct senior response is not "sure" but "that changes the caching architecture; here's what it costs," then offer the cheaper alternative of *creating a new link* instead of mutating an existing one.
- **Don't store click counts in the `urls` table.** Putting a `clicks` column on the mapping row seems convenient, but it turns the read-mostly mapping into a write-hot row (every redirect would update it) and couples the two paths you worked to separate. Counts live in the rollup store, full stop. This is the same hot-row trap the async-analytics deep dive solves, surfacing in the schema.

Now the deep dives — the three parts where this question separates the staff engineers from everyone else.

## Deep dive 1: distributed key generation

This is the heart of the question and where juniors go shallowest. "Generate a unique short code" sounds trivial until you ask: *across how many machines, with what coordination, and what happens when two of them generate at the same instant?* There are four serious strategies, each buying a different property at a different cost. Figure 2 lays them against the four properties that matter — uniqueness, length, coordination, and predictability.

![A matrix comparing counter-plus-base62, hash-plus-collision-check, a key-generation service, and Snowflake-style IDs across uniqueness, length, coordination cost, and predictability](/imgs/blogs/design-a-url-shortener-2.webp)

### Strategy A: counter + base62

Keep a single global monotonic counter. Each new URL takes the next integer, and you encode that integer in base62 to get a short, opaque-looking code. The encoding is a simple repeated divmod, shown in Figure 5.

![A pipeline showing a monotonic counter value being repeatedly divided by 62, mapped to a 62-symbol alphabet, and emitted as a seven-character short code](/imgs/blogs/design-a-url-shortener-5.webp)

```python
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 62 symbols
BASE = len(ALPHABET)

def encode(n: int) -> str:
    if n == 0:
        return ALPHABET[0]
    chars = []
    while n > 0:
        n, rem = divmod(n, BASE)
        chars.append(ALPHABET[rem])
    return "".join(reversed(chars))

def decode(code: str) -> int:
    n = 0
    for ch in code:
        n = n * BASE + ALPHABET.index(ch)
    return n
```

**What you gain:** uniqueness is *guaranteed* with zero collision checks — every integer maps to exactly one code, and the counter never repeats. Codes are minimal length (counter value 3.5 trillion is still 7 chars). Simple to reason about.

**What you pay — and this is the whole problem:** a single global counter is a coordination point. If every create RPC has to ask one central sequence for the next number, that sequence is a write bottleneck and a single point of failure. At 40 writes/sec it's a non-issue, but the *shape* is dangerous, and there's a second, worse cost: **the codes are predictable.** `encode(1000000)` and `encode(1000001)` are adjacent. Anyone can decode a code back to an integer, increment it, and re-encode to find the next link in the system. For a public shortener that leaks the total link count and enables enumeration of everyone's links. That is often a dealbreaker.

You can break the central-counter bottleneck without giving up the counter idea: hand each app server a **range** of integers (a "ticket block"). Server A gets [1, 100000], server B gets [100001, 200000], and so on. Each server allocates from its block locally with no coordination, and only hits the central allocator once per block (every ~2,500 seconds at 40/sec). This is the **key-generation service** pattern (Strategy C) and it neatly removes the coordination cost. But it does *not* fix predictability — the codes are still sequential within and across blocks.

### Strategy B: hash + collision check

Hash the long URL (plus a salt or timestamp to disambiguate identical URLs), take the first 7 base62 characters of the digest, and use that as the code.

```python
import hashlib

def hash_code(long_url: str, salt: str) -> str:
    digest = hashlib.sha256((long_url + salt).encode()).digest()
    # take 42 bits -> 7 base62 chars
    n = int.from_bytes(digest[:6], "big") % (BASE ** 7)
    return encode(n).rjust(7, ALPHABET[0])
```

**What you gain:** no central counter at all — any machine can generate a code from the URL alone, zero coordination. Codes look random (not predictable in the sequential sense).

**What you pay:** **collisions are now possible**, so every insert must check the database for an existing row with that code, and retry with a new salt on conflict. This is the birthday problem in action. With a 42-bit space (`62^7 ≈ 3.5e12 ≈ 2^41.7`) and billions of stored codes, collision probability becomes non-trivial as you fill the space. The collision check turns every write into a read-then-write, and under contention you can get retry storms. At low write rates it's fine; as the table fills, the retry rate creeps up. You're trading the counter's coordination for the hash's collision-retry — neither is free.

### Strategy C: offline key-generation service

Decouple *generating* keys from *using* them. A standalone service pre-generates unique codes in the background and stores them in two pools: "available" and "used." When an app server needs a code, it pops one from the available pool (a single fast operation), marks it used, and moves on. The generator refills the available pool in batches during quiet periods.

**What you gain:** code generation is off the critical path entirely — the app server does an O(1) pop, never a hash-retry loop, never a central-sequence round trip per request. Uniqueness is guaranteed because the generator produces each code exactly once. Codes can be generated in a *shuffled* order, killing predictability. This is the strategy large-scale shorteners gravitate toward because it makes the create path fast *and* the codes unguessable *and* the uniqueness invariant trivially held.

**What you pay:** operational complexity. You now run and monitor a separate service with two pools, you must handle the generator falling behind (available pool draining), and you size the pools so the app never blocks waiting for a key. There's also a subtle correctness wrinkle: when a server pops a key and then crashes before using it, that key is "used" but maps to nothing — a small permanent leak. You mitigate with a short reservation TTL so abandoned keys return to the pool. None of this is hard, but it's *more* than a counter, and that's the cost.

### Strategy D: Snowflake-style distributed IDs

Generate a 64-bit ID locally on each node by composing **timestamp + machine ID + per-node sequence**, the way Twitter's Snowflake does. Each node produces globally unique, roughly time-ordered IDs with zero coordination, and you base62-encode the result.

**What you gain:** no central coordination, no collision checks, no key-gen service to operate — each node is self-sufficient. IDs are k-sorted (roughly time-ordered), which is nice for some downstream uses.

**What you pay:** **length.** A 64-bit number is up to 11 base62 characters, not 7 — your short URLs get noticeably longer, which for a *shortener* is the product's whole reason for existing. Snowflake IDs are also only *semi*-unpredictable: the timestamp and machine bits are structured, so a determined observer can infer creation time and roughly how many IDs a node has minted. Snowflake is a brilliant fit for database primary keys; it's a mediocre fit for user-facing short codes precisely because it optimizes coordination-freedom over brevity, and brevity is the point here.

### Choosing

The decision is a two-question tree, in Figure 7: do you accept a central sequence, and must the codes be unguessable? If a central sequence is fine and prediction doesn't matter (an internal corporate shortener, say), the **counter + base62** is the simplest correct answer and I'd ship it. If you need unguessable codes *and* short length *and* you operate at real scale, the **offline key-generation service** wins despite its operational cost. The hash + collision-check is the right call when you specifically want generation with zero shared state and can tolerate retries. Snowflake is for when coordination-freedom dominates and you'll accept longer codes.

![A decision tree branching on whether a central sequence is acceptable and whether the click count must be hidden, leading to counter, key-gen service, Snowflake, or hash strategies](/imgs/blogs/design-a-url-shortener-7.webp)

My default recommendation for a public, scaled shortener: **offline key-generation service producing shuffled 7-char base62 codes.** It's the only option that holds uniqueness, length, and unpredictability simultaneously, and the operational cost is real but bounded. I'd defend that in a review by naming exactly what it pays — an extra service to run — and why that price buys all three properties no single-mechanism approach gives you at once.

There's an implementation detail in the key-gen service worth getting right, because it's where the design usually leaks. The service produces codes into an "available" pool and app servers pop from it; the question is *how the pop is made atomic* across many app servers pulling concurrently. The cheap-and-correct answer is to give each app server a **lease of a block** of pre-generated codes rather than popping one code per request. A server leases, say, 1,000 codes at once (a single atomic operation against the pool), serves creates from that local block with zero further coordination, and leases a fresh block when it's nearly drained. This collapses the coordination from once-per-create to once-per-thousand-creates, and it bounds the blast radius of a crashed server to at most 1,000 leaked codes — negligible against a 3.5-trillion space. The leased-block pattern is the same trick the counter-range allocator used in Strategy A; the difference is that the key-gen service can shuffle codes *within and across* blocks before handing them out, so leasing a block doesn't hand out a sequential run. This is how you get coordination-free *and* unpredictable simultaneously, and it's the kind of detail that separates "I've read about this" from "I've operated this."

One more failure mode the senior anticipates: **the generator falling behind.** If creates spike and the available pool drains faster than the background generator refills it, app servers start blocking on key acquisition — a self-inflicted outage on the create path. The defenses are a high-water-mark alarm on the available pool (page before it's empty, not after), a generation rate provisioned for several times peak create volume (cheap, since generating a code is just incrementing and shuffling), and a fallback path where, if the pool is genuinely exhausted, an app server can synthesize a code locally via the hash-with-collision-check strategy until the pool recovers. That last point is the senior move: don't pick one key-gen strategy religiously — pick a primary and keep a degraded fallback, so a generator outage degrades to "slightly slower creates with occasional collision retries" instead of "creates down."

#### Worked example: does 7 characters actually last?

Let me make the "lasts centuries" claim concrete and stress it. At 100M new codes/month and `62^7 ≈ 3.5 trillion` codes:

```
3.5e12 / 1e8 per month = 35,000 months ≈ 2,900 years (using long URLs minted)
```

Wait — that's the *available* space divided by *monthly* consumption, so it's 2,900 *months* if I divide wrong; let me be careful. 3.5e12 codes ÷ 100e6 codes/month = 35,000 months = **~2,900 years.** Even at 10× our assumed growth (1 billion codes/month), it's ~290 years. Seven characters is not a constraint in any realistic timeline.

But here's the staff-level subtlety the math hides: **with the hash + collision-check strategy, exhaustion is not the failure mode — collision rate is.** By the birthday bound, once you've stored `N` codes in a space of size `M = 3.5e12`, the chance a *new* random code collides is roughly `N / M`. At 6 billion stored codes (our five-year figure), `N/M ≈ 6e9 / 3.5e12 ≈ 0.0017`, so ~0.17% of inserts collide and retry once. Tolerable. But push to 10× the links (60 billion) and it's ~1.7% — now nearly one in fifty writes retries, and a small fraction retry multiple times. The counter and key-gen-service strategies have *zero* collision rate by construction, so they don't pay this tax at all. That difference — flat zero versus a rate that climbs as the table fills — is exactly the kind of second-order consequence a senior surfaces before it becomes an incident. The fix when you do run short is to widen to 8 characters (`62^8 ≈ 218 trillion`), which you can do gracefully because old 7-char codes still decode fine.

## Deep dive 2: the read path is everything

The system lives or dies on the redirect, and the redirect is a read. Our whole job on this path is to make sure the database almost never sees it. Three layers stand between the user and the database, and each is a cache that short-circuits the rest.

**Layer 0 — the CDN edge (302 caching).** The cleverest optimization, and the most overlooked. A redirect response is just an HTTP 302 with a `Location` header — it's cacheable like any other HTTP response. If we let the CDN cache the redirect for the *popular* links, a viral link's redirect is served entirely at the edge, geographically close to the user, never touching our origin. The catch: caching a redirect means you can't change or expire that link instantly (the edge holds the stale 302 until its TTL lapses). So edge-caching is right for links you know are immutable and high-traffic, with a modest TTL (say 60 seconds) so an expiry or takedown propagates within a minute. This is the single biggest lever for surviving a viral link, and we'll return to it under the stress test.

**Layer 1 — the application cache (Redis).** The workhorse. The redirect service does a Redis `GET code` first; on a hit (the 95% case, per our 5 GB hot-set math), it returns the destination in ~1 ms without ever touching Postgres. The cache is a simple `code → long_url` map. We use a **cache-aside** (lazy-load) pattern: on a miss, read the database, populate the cache, return. For the failure modes this pattern hides — stampedes, stale entries, the thundering herd on a cold key — see [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite); the two that bite a shortener hardest are the cold-start stampede (mitigated with a per-key lock or request coalescing) and negative caching (cache the *absence* of a code too, so a flood of requests for a non-existent link doesn't hammer the database).

```python
def resolve(code: str) -> str | None:
    # 1. Try the cache (the 95% path).
    url = redis.get(f"u:{code}")
    if url is not None:
        return None if url == NEG else url   # negative cache hit

    # 2. Cold miss: read the source of truth.
    url = db.fetch_long_url(code)             # ~8 ms
    if url is None:
        redis.setex(f"u:{code}", 60, NEG)     # negative-cache the miss, short TTL
        return None

    # 3. Populate cache for next time. Long TTL; mappings are immutable.
    redis.setex(f"u:{code}", 86400, url)
    return url
```

**Layer 2 — the database (cold misses only).** Postgres sees only the long tail: links clicked rarely enough to have fallen out of cache. Because the mapping is immutable, cache entries can have long TTLs (a day), and the database read is a single primary-key lookup (~8 ms). We can also put read replicas behind the redirect service so even cold misses scale horizontally without touching the write primary.

The cold-miss path deserves one specific safeguard: **request coalescing on the miss.** When a previously-cold link suddenly gets popular (a link that was clicked once a week is tweeted), the first wave of requests all miss the cache simultaneously and all stampede the database for the same key — the thundering herd. The fix is to let only *one* request per key go to the database on a miss while the others wait briefly for it to populate the cache (a per-key in-flight lock, often called single-flight). Without it, a cold viral link can briefly send thousands of identical reads to Postgres in the seconds before the cache fills — a self-inflicted load spike at the worst possible moment. With it, the database sees exactly one read per cold key regardless of how many clients want it. This is cheap to implement and one of the highest-leverage cache pitfalls to get right; the full taxonomy is in [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite).

There's also a consistency nuance even in this read-mostly system. Reads served from a read replica can be slightly stale relative to the primary — for an *immutable* mapping that's harmless (the row never changes after creation, so a stale read is still correct), which is exactly why immutability buys us coordination-free replica reads. The one window where it bites is the read-your-writes case: a user creates a link and immediately tests it, but their redirect lands on a replica that hasn't yet received the new row. Two clean fixes: route the create response's own confirmation through the primary, and accept that a brand-new link might 404 for a sub-second replication-lag window (or pin the creating user's next read to the primary briefly). The deeper treatment of these guarantees — and why an immutable dataset makes them mostly moot — is in [consistency models: a practical guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects).

Figure 9 puts these layers into a **latency budget** — the discipline of allocating the 50 ms p99 target across layers so you know which one is allowed to be slow.

![A stack diagram of the redirect latency budget allocating milliseconds across TLS and DNS, CDN edge lookup, Redis on miss, Postgres on cold miss, and the 302 response](/imgs/blogs/design-a-url-shortener-9.webp)

The budget makes the design legible: TLS/DNS is amortized to ~0 on warm connections; the CDN edge lookup is ~2 ms; a Redis GET on edge-miss is ~1 ms; a Postgres read on cold-miss is ~8 ms; building the 302 is sub-millisecond. The worst case — edge miss *and* cache miss — is still well under 50 ms, and it's rare. The common case is single-digit milliseconds, and the *most* common case (edge hit) is whatever the user's distance to the nearest CDN POP is, often under 20 ms total including network.

### 301 versus 302: the decision that quietly governs analytics

Here is a choice that looks like trivia and is actually load-bearing. When you redirect, you can return **301 Moved Permanently** or **302 Found** (temporary).

- **301** tells the browser "this is permanent — cache it and don't ask me again." Browsers and intermediaries cache 301s aggressively, sometimes indefinitely. That's *great* for redirect latency (the browser skips us entirely on the second click) and *terrible* for analytics (we never see the second click, so our counts are wrong and undercount badly). It also makes the link effectively un-revocable from the browser's perspective: if you later expire or repoint the link, a browser holding a cached 301 keeps going to the old destination.

- **302** tells the browser "temporary — ask me every time." Every click comes back through us, so we count every click and we can change or expire the link at will. The cost is that we serve every redirect (no browser-side caching), which is exactly the load our cache-everything design is built to absorb.

**A senior chooses 302 for a shortener, almost always.** Analytics are usually a core feature (people shorten links *to* track them), and the ability to expire or take down a link is a hard requirement (abuse, malware, legal). The latency we "lose" by not letting browsers cache is the latency we already engineered away with the CDN and Redis. You'd choose 301 only for a shortener where you truly never need analytics or revocation and want maximum browser-side speed — a rare product. This is the cleanest example in the whole design of how a one-line decision (`301` vs `302`) ripples into two other subsystems (analytics correctness and link revocation). Articulating that ripple is the senior move; the trade-offs reasoning generalizes in [articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond).

## Deep dive 3: click analytics without slowing the redirect

We established that analytics is the real write load: 12,000 clicks/sec at peak, each of which we want to count, versus 40 creates/sec. If counting a click means a synchronous database write on the redirect path, two bad things happen: the redirect waits on that write (adding tens of milliseconds to the user-facing latency we worked so hard to minimize), and 12,000 writes/sec of counter increments hammer a single hot row per popular link, serializing on that row's lock. Figure 4 contrasts the two approaches.

![A before-and-after diagram contrasting a synchronous analytics write that blocks the redirect against an asynchronous design that emits a click event to a queue and returns immediately](/imgs/blogs/design-a-url-shortener-4.webp)

**The fix is to make click counting asynchronous.** On redirect, the service does the bare minimum to send the user on their way — resolve the code (from cache) and return the 302 — and *emits a click event to a queue* as a fire-and-forget side effect. The user's redirect is not blocked by the count; it returns in single-digit milliseconds. A separate fleet of consumers reads the click stream and aggregates counts at its own pace.

```python
def redirect(code: str):
    url = resolve(code)                       # cache-first, ~1 ms
    if url is None:
        return Response(status=404)

    # Fire-and-forget: do NOT block the redirect on this.
    click_producer.send("clicks", {
        "code": code,
        "ts": now_ms(),
        "referer": request.headers.get("Referer"),
        "country": geo_from_ip(request.remote_addr),
    })

    return Response(status=302, headers={"Location": url})
```

This decoupling buys several things at once:

- **The redirect stays fast.** The queue `send` is a local, non-blocking enqueue (or a batched async flush). The user feels nothing.
- **Writes get batched.** The consumer aggregates many click events into per-hour rollups before writing, turning 12,000 individual increments/sec into a handful of batched upserts. The single-hot-row problem disappears because we update a rollup, not a live counter, and we can pre-aggregate in the consumer's memory.
- **Spikes get absorbed.** A viral link produces a burst of click events; the queue is a shock absorber that lets the consumer drain at a steady rate instead of the database taking the spike head-on. This is backpressure done right — see [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure).

The cost we accept: **analytics are eventually consistent.** The click count you see lags real time by seconds to a minute. For a stats dashboard that's completely fine — nobody needs to-the-millisecond click counts. We've traded freshness (cheap to give up here) for redirect latency and database stability (precious). That's the right trade, and naming *what* we traded is the discipline.

There's a delivery-semantics subtlety worth flagging: the queue gives *at-least-once* delivery, so a click event can be processed twice (consumer retries after a crash). That would *over*count clicks. For approximate analytics, slight overcounting is acceptable and you ignore it. If you need exact counts, you deduplicate on a click id in the consumer — the [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) playbook — at the cost of a dedup store. For a shortener I'd ship at-least-once and tell stakeholders the counts are accurate to within a fraction of a percent. The pattern of decoupling work via a log is exactly [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects).

#### Worked example: the write-amplification a queue prevents

Make the "queue saves the database" claim quantitative. Suppose a popular link does 50,000 clicks/sec during a spike, and take the naive synchronous design: each click is an `UPDATE urls SET clicks = clicks + 1 WHERE code = ?`. That's 50,000 writes/sec all targeting *one row*. Row-level locking serializes them: each update must acquire the row lock, so the effective throughput is capped at roughly `1 / lock-hold-time`. If a lock-update-commit cycle takes ~1 ms, that single row tops out near ~1,000 updates/sec — meaning 49,000 of the 50,000 clicks/sec *queue up behind the lock*, the connection pool fills with waiters, and the database's latency for *every other query* (including redirect cold-misses) climbs. One viral link takes the whole database down. This is the hot-row failure, and it's why synchronous counting is a latent outage.

Now the async design. 50,000 click events/sec flow into Kafka — trivial for Kafka, which handles millions of messages/sec per broker. A consumer reads them and aggregates *in memory* into per-(code, hour) buckets, flushing a batched upsert every, say, 5 seconds. In those 5 seconds it accumulated 250,000 clicks for that link into a *single* counter, then writes *one* row with `clicks = clicks + 250000`. The database sees one write every 5 seconds for that link — `0.2 writes/sec` — instead of 50,000. That's a **250,000× reduction in database write load** on the hot link, achieved purely by moving the aggregation off the hot path and batching it. The number is the argument: no amount of database tuning closes a 250,000× gap; you close it by changing where the work happens.

## Custom aliases and collision handling

Custom aliases (`sho.rt/my-launch`) are a small feature with a sharp correctness edge. A custom alias is a user-chosen code, which means it can collide with an existing code (generated or custom) and must be checked atomically. The naive "check if it exists, then insert" is a race: two requests for the same alias both check, both see it free, both insert. You need the database to enforce uniqueness, not the application.

```sql
-- Atomic claim: succeeds only if the alias is free. The PK does the work.
INSERT INTO urls (code, long_url, owner_id, is_custom)
VALUES ('my-launch', 'https://example.com/...', 42, true)
ON CONFLICT (code) DO NOTHING
RETURNING code;
-- 0 rows returned => alias taken => return 409 to the user.
```

The `ON CONFLICT DO NOTHING` makes the claim atomic at the database, so concurrent requests can't both win. Two more rules a senior adds:

- **Custom aliases and generated codes share one namespace.** If a user claims `my-launch`, the key generator must never later produce `my-launch`. With the offline key-gen service this is automatic (custom claims and generated pops both go through the same uniqueness gate). With a raw counter, you must ensure custom aliases live outside the counter's encoding range or you risk a future collision — a real gotcha.
- **Reserve system paths.** Block aliases that collide with routes (`api`, `login`, `health`) or that look like profanity/abuse, via a blocklist checked at claim time.

## Expiry and TTL

Optional link expiry sounds simple — store `expires_at`, refuse to redirect past it — but the *enforcement* has two layers. At read time, the resolver checks `expires_at` and returns 404 (or a friendly "link expired" page) if past. But expired rows also need eventual *deletion* to reclaim storage, and you don't want a giant `DELETE WHERE expires_at < now()` scanning the whole table. Two clean approaches:

- **Lazy expiry** (read-time): the resolver treats an expired link as a miss and, opportunistically, schedules the row for deletion. Simple, no background job, but expired rows linger until someone tries to use them.
- **Background sweep** (batch): a periodic job deletes small batches of expired rows using the partial index `idx_urls_expiry`, paginating to avoid long locks. Combine with lazy expiry so the user-facing behavior is instant and the cleanup is amortized.

Crucially, **cache TTL must be shorter than or equal to link TTL near expiry**, or a cached entry will keep redirecting an expired link until the cache entry lapses. The clean rule: when caching a link with an `expires_at`, set the cache TTL to `min(default_ttl, time_until_expiry)`. And if you're edge-caching 302s, the edge TTL bounds how fast an expiry (or a takedown) propagates — another reason to keep edge TTLs modest.

## Rate limiting and abuse

The create endpoint is the abuse surface. An unthrottled shortener gets weaponized: spammers mint millions of links to mask phishing destinations, attackers enumerate to harvest links, and bots burn key space. Defenses, layered:

- **Rate-limit creates per user and per IP.** A token bucket at, say, 100 creates/minute/user is generous for humans and ruinous for a script. The full design — token bucket vs sliding window, where to enforce, what to do at the limit — is in [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure). Enforce it at the API gateway so abusive load never reaches the create service.
- **Destination blocklist.** Check the long URL against a phishing/malware blocklist (e.g. a reputation feed) at create time; refuse known-bad destinations. This is also why you want 302 (revocable) over 301 — a destination that turns malicious *after* creation can be taken down.
- **Unguessable codes.** Sequential codes invite enumeration; the key-gen choices above (shuffled or hashed codes) defang it. This is the security argument for *not* using a raw counter on a public shortener.
- **Redirect-side protection.** The redirect path is public and cache-fronted, so it's naturally resilient, but you still cap per-IP redirect rates to blunt scraping and reflect-amplification attempts.

The rate limiter itself sits at the gateway with a shared counter store (Redis), so the limit is enforced consistently across all create-service instances — a distributed rate limiter, not a per-instance one, or attackers just spread across your fleet.

A subtlety on the redirect side that catches people: the redirect endpoint is the one part of the system you *cannot* aggressively rate-limit by default, because legitimate viral traffic looks exactly like an attack — fifty thousand requests a second for one code is either a celebrity tweet or a DDoS, and you can't tell from request rate alone. This is precisely why the read path's defense is *caching and edge-offload* rather than rate limiting: you absorb the load instead of rejecting it, because rejecting a viral link's redirects means breaking a legitimately popular link. Rate limiting protects the *create* path (where each request consumes a scarce resource and humans don't legitimately create thousands of links a minute); caching protects the *read* path (where requests are cheap, idempotent, and legitimately spiky). Knowing which tool defends which path — reject-the-excess versus absorb-the-excess — is the senior distinction, and getting it backwards (rate-limiting redirects, or trying to cache your way out of create abuse) is a classic mistake.

## Trade-offs and alternatives rejected

A design review is won by naming what each choice *costs*, not just what it gains. Here are the load-bearing decisions with their rejected alternatives.

| Decision | What we chose | What we rejected | Why, and what it costs |
| --- | --- | --- | --- |
| Key generation | Offline key-gen service (shuffled 7-char base62) | Raw counter, hash+check, Snowflake | Holds uniqueness + short length + unpredictability at once; costs an extra service to operate |
| Read path | CDN + Redis cache-aside, DB on miss | Read straight from DB | 95% of reads served from RAM/edge at ~1 ms; costs cache invalidation discipline |
| Redirect status | 302 (temporary) | 301 (permanent) | Keeps analytics correct and links revocable; costs browser-side caching we replace with our own |
| Analytics | Async via queue, batched rollups | Sync DB increment on redirect | Redirect stays < 5 ms, hot-row contention gone; costs eventual-consistency (seconds-late counts) |
| Storage (mapping) | Single SQL node + replicas | Shard across NoSQL from day one | 3 TB fits one box for years; costs nothing now, revisit when it grows |
| Custom alias claim | DB unique constraint, `ON CONFLICT` | App-level check-then-insert | Atomic, race-free; costs a 409 path the client must handle |

The single most important trade-off — the one I'd put on the whiteboard first — is **302 over 301**, because it's the choice that looks like a detail and silently determines whether your two flagship features (analytics and revocation) work at all. The second is **async analytics**, because synchronous click-counting is the most common way this design accidentally couples the slow write path to the fast read path. A senior reviewer probes exactly those two; have the answers ready.

One alternative deserves a fuller rejection: **"just use a key-value store like DynamoDB for everything."** It's tempting — KV is a natural fit for `code → url` point lookups, and it scales horizontally for free. And honestly, it's a defensible choice; many real shorteners run on exactly this. But for *our* scale it's solving a sharding problem we don't have (3 TB on one SQL node) while making custom-alias atomicity (a conditional write) and the owner-dashboard query (a secondary-index scan) slightly more awkward than they are in SQL. The senior framing: KV isn't *wrong*, it's *premature* — pick the simplest store that holds the invariants today (single-node SQL), and migrate to KV *if and when* the mapping outgrows one node, using the [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) playbook. Choosing the boring store you can defend with "it fits" beats choosing the scalable store you justify with "it'll scale."

## Optimization: where the bottleneck actually is, and how to measure the win

Re-derive the bottleneck from the numbers and the optimization writes itself. The system is 100:1 read-heavy, so the read path is where every microsecond and every dollar concentrates. Three levers, each with a measurable win:

**Lever 1 — cache hit rate.** This is the master dial. At a 95% Redis hit rate, 5% of 12,000 reads/sec = 600 reads/sec hit Postgres — trivial. Push the hit rate to 99% and Postgres sees 120 reads/sec; let it drop to 90% and Postgres sees 1,200 reads/sec — a 10× swing in database load from a 9-point swing in hit rate. **Measure it as cache hit ratio and database read QPS**, and treat a falling hit rate as a leading indicator of trouble (cache too small, churn too high, or a working-set shift). The win from raising hit rate from 95% to 99% is concrete: 5× less database load and a fatter p99 margin.

**Lever 2 — CDN edge offload.** Edge-caching 302s for popular links removes them from your origin entirely. If the top 1% of links carry 90% of traffic (a typical power law for a shortener), edge-caching just those serves ~90% of all redirects at the edge, dropping origin read QPS by an order of magnitude. **Measure it as origin-offload ratio** (fraction of redirects served by the CDN without an origin hit). Going from 0% to 90% offload is the difference between provisioning origin for 12,000 QPS and provisioning for ~1,200.

**Lever 3 — the 302 itself.** The redirect response is tiny and computable from cache alone, so the per-request work is minimal — no template rendering, no database, no serialization beyond a header. **Measure it as p50/p99 redirect latency, server-side.** Our target: p50 ~1–2 ms (cache hit), p99 < 50 ms (cold miss), and effectively network-bound from the edge. If p99 climbs, the cause is almost always cache misses reaching the database, which loops back to Lever 1.

The meta-point: **the read path is the product, so optimize it relentlessly and measure it religiously.** A 50 ms create latency that nobody clicks more than once is irrelevant; a 50 ms redirect latency felt a billion times a day is the entire user experience.

A word on *cost*, because optimization without a dollar figure is just vibes — and cost-as-a-constraint is its own discipline (see [cost as a design constraint: FinOps](/blog/software-development/system-design/cost-as-a-design-constraint-finops)). The three levers above are also the three cost levers, and they compound. Every redirect served from the CDN edge is a request your origin never pays compute or bandwidth for; at 90% edge offload, your origin fleet shrinks by an order of magnitude, and CDN egress for a few-hundred-byte 302 is nearly free. Every Redis hit is a database read you don't pay for in IOPS or replica capacity; at 99% hit rate your Postgres needs to handle hundreds of reads per second, not tens of thousands, which is the difference between a modest instance and an over-provisioned cluster. The optimization story and the cost story are the same story told twice: **push work outward (edge), keep it in memory (cache), and the bill follows the architecture.** The single most expensive mistake you could make — and the one the naive design makes — is letting the database serve the read path directly, because then you're paying for database capacity sized to peak redirect QPS, which is exactly the capacity the cache was supposed to make unnecessary.

## Stress test: what breaks at 10×, at a hot key, at a region outage

A design isn't done until you've tried to break it. Three scenarios that actually happen.

### The viral link (hot key)

One link gets tweeted by a celebrity. It goes from 10 clicks/sec to **50,000 clicks/sec** in minutes — all for a single code. Every one of those requests wants the *same* cache key. A naive Redis cluster shards by key, so all 50,000 QPS land on the *one shard* holding that code, saturating it while the rest of the cluster sits idle. The redirect p99 for that link (and anything else on that shard) climbs. This is the classic hot-key problem, and Figure 8 shows the fix.

![A before-and-after diagram showing a single viral code saturating one Redis shard versus the same code served from the CDN edge and replicated across nodes so origin sees almost no load](/imgs/blogs/design-a-url-shortener-8.webp)

The defense is layered and exactly what the read-path design was built for:

- **Edge-cache the hot 302.** A viral link is, by definition, immutable-and-popular — the perfect edge-cache candidate. Once the CDN caches that 302 (TTL 60s), the celebrity's followers are served at the edge near them, and your origin sees a trickle. This alone usually ends the incident.
- **Replicate the hot key.** Within Redis, replicate a detected hot key to multiple nodes (or front it with a small local in-process cache on each app server) so the load fans out instead of hammering one shard.
- **The queue absorbs the click spike.** The async analytics design means 50,000 clicks/sec of *events* flow into Kafka and drain at a steady rate; the database never takes the spike. If we'd counted synchronously, this is where the system would have fallen over.

Result: origin sees < 100 QPS for a link doing 50,000, because the edge and replicas absorbed it. The hot key is a non-event *because* we cache-fronted and decoupled. A synchronous, single-cache-node design would have melted.

### 10× growth

Traffic 10×'s to 120,000 reads/sec peak, 400 creates/sec, 10 TB of mapping over five years. What breaks?

- **Reads:** the cache scales horizontally (add Redis nodes, the hot set is still only ~50 GB at 10×) and the CDN scales for free. Reads are fine — this is why the cache-everything design matters.
- **Mapping storage:** 10 TB starts to strain a single node's comfortable headroom. *This* is the trigger to shard the mapping — and now, having deferred it, you do it deliberately with [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime), sharding by code (hash or range) since the hot path is a point lookup on `code`. The key insight: you reach for sharding when *storage* forces it, not when *read throughput* does, because the cache already solved throughput.
- **Analytics:** 120,000 click events/sec is real Kafka load but well within its capability; you add partitions and consumers. The rollup store (ClickHouse-class) is built for this write rate.

The system 10×'s gracefully on reads, and the one thing that forces structural change (mapping storage) was deliberately left at its simplest until the numbers demanded more. That deferral is evolutionary architecture in action — see [evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change).

### Region outage

The region holding your primary database and Redis goes dark. Reads and writes both stop unless you planned for it. The asymmetry of our availability requirement guides the response: **the redirect path must survive a region loss; the create path can degrade.**

- **Redirects** are served by a multi-region CDN and read-only Redis/replica fleets in each region. Because the mapping is immutable and replicated, a standby region can serve redirects from its own replica with no coordination — this is the easy case, since read-only replication across regions is well-trodden.
- **Creates** need the primary. If the primary's region is down, you fail over to a standby primary (a few minutes of create unavailability) — acceptable, because creates are 40/sec and a few minutes of "try again shortly" on link creation is a far smaller harm than redirects failing.
- **Analytics** buffer in the regional Kafka and reconcile when the failed region returns; a gap in click counts is the most tolerable failure of all.

The senior framing is the availability *asymmetry*: you spend your reliability budget on the read path (multi-region, replicated, cache-fronted) and accept lower availability on the write path, because that's where the user pain is least. Trying to make creates as available as redirects would cost far more for far less benefit. Multi-region patterns and their failure modes are in [multi-region and geo-distribution](/blog/software-development/system-design/multi-region-and-geo-distribution).

## How real systems actually do it

The classic question has decades of real implementations to learn from. A few, with the concrete lesson each teaches (numbers are order-of-magnitude where I'm not certain of exact figures — the lesson, not the digit, is the point).

**Bitly** is the canonical large-scale shortener and the clearest illustration that a shortener is mostly a read system. Public engineering talks over the years describe exactly the shape we derived: short codes as the cache key, an aggressive caching tier in front of the datastore, and a separate analytics pipeline for click data because click volume dwarfs link-creation volume. The lesson: **the link store and the analytics store are different systems with different access patterns**, and conflating them is the mistake. Bitly's scale also forced unguessable codes — sequential codes would have leaked their entire link graph.

**TinyURL** (one of the oldest) shows the other end: a shortener can run for a very long time on a comparatively simple stack because the core data is so small and so read-heavy. The lesson: **don't over-build.** A single well-cached database carried enormous redirect volume for years. The shorteners that fell over did so on analytics and abuse, not on the core mapping.

**Twitter's `t.co`** wraps every link posted on the platform, which means it operates at the platform's full firehose — billions of links, every tweet a potential create, every impression a potential redirect. `t.co` exists primarily for *measurement and safety* (click analytics and malware interception), which is precisely why a platform shortener uses 302-style behavior and revocable links: Twitter must be able to take down a malicious destination after the fact. The lesson: **the redirect's job is often as much about control and measurement as about brevity** — the 302/analytics/revocation coupling we derived isn't academic, it's why platform shorteners exist.

**Snowflake (Twitter's ID generator)** isn't a shortener but is the canonical answer to "generate unique IDs across many machines without coordination," and it's worth knowing precisely *because* it's the strategy you'd reach for if you wanted coordination-free codes — and then reject for a shortener because 64-bit IDs encode to ~11 characters, too long for the product. The lesson: **the right distributed-ID technique depends on whether brevity or coordination-freedom dominates**, and for a *shortener*, brevity usually wins, which is why offline key-gen services beat Snowflake here.

**URL shorteners as a CDN feature** (Cloudflare, Fastly, and others offer edge-side redirect handling) close the loop on the read-path optimization: when the redirect logic runs *at the edge*, the origin can be cold for popular links. The lesson: **push the redirect as close to the user as the link's mutability allows** — the more immutable and popular the link, the further out you can cache it.

The thread through all of them: the mapping is the easy part, every real system gets crushed (or saved) on **caching, analytics throughput, and abuse**, and the shorteners that scaled treated the read path and the analytics path as the real engineering. That's the staff-level reading of the "easy" question.

## When to reach for this design (and when not to)

**Reach for the full design above when** you're building a *public, scaled* shortener: high redirect volume, analytics as a real feature, abuse a real threat, and unguessable codes a requirement. Then you want the offline key-gen service, the CDN + Redis cache-everything read path, async queue-backed analytics, 302 redirects, and distributed rate limiting. That's the version the question is really asking for, and it's the version this post defends.

**Don't reach for all of it when** the context is smaller:

- **Internal corporate shortener** (links for a wiki, low traffic, trusted users): a raw counter + base62 and a single cached database is plenty. Skip the key-gen service, skip the CDN, maybe skip the queue and count clicks synchronously — at low volume it doesn't matter. Predictable codes are fine internally. Building the full design here is over-engineering.
- **A side project or MVP:** one Postgres table, a Redis cache, 302 redirects, and a simple per-IP rate limit. Add the queue and key-gen service only when the create rate or click rate actually demands them. The whole point of the estimation up front is to *not* build the parts you don't need yet.
- **When 301 is genuinely right:** a shortener that never needs analytics or revocation and wants maximum browser-side speed (rare, but it exists) — use 301 and let browsers cache the redirect.

The decision rule a senior states out loud: **match the design to the scale you can prove with arithmetic, not the scale you guess at.** The full design is correct at billions of redirects; it's gold-plating at thousands. Knowing which one you're building — and being able to say why — is the whole skill.

The next system in this series, [design a news feed and timeline](/blog/software-development/system-design/design-a-news-feed-and-timeline), inverts the difficulty: there the *write* path (fan-out on publish) is the hard part and reads are comparatively easy — the mirror image of the shortener's read-heavy shape. The contrast is instructive: the same toolkit (caching, queues, estimation, sharding) gets pointed at the opposite end of the read/write asymmetry.

## Key takeaways

- **Do the arithmetic first.** ~40 writes/sec, ~12k reads/sec peak, 3 TB over five years on one box, a 5 GB hot set in RAM, a 3.5-trillion key space lasting centuries. Those numbers *are* the design: read-heavy, cache-everything, no sharding for years.
- **The read path is the product.** A redirect is a read, felt billions of times a day; optimize it relentlessly with CDN edge caching and a Redis cache-aside tier so the database sees only cold misses. Measure it as cache hit ratio and redirect p99.
- **Key generation is the real depth.** Counter+base62 is simple but predictable; hash+check trades coordination for collisions; Snowflake trades length for coordination-freedom; the offline key-gen service is the only option that holds uniqueness, short length, and unpredictability at once — at the cost of an extra service.
- **Choose 302 over 301**, almost always. It's the one-line decision that silently governs whether analytics and link revocation work. The browser-caching you give up, you replace with your own cache and CDN.
- **Count clicks asynchronously.** Analytics is the real write load (300× creates); emit click events to a queue and aggregate later, so the redirect never blocks on a count and the database never takes the spike. Accept eventual-consistency on counts.
- **Enforce uniqueness at the database**, not the app — custom aliases claimed with `ON CONFLICT DO NOTHING` are race-free; check-then-insert is not.
- **Defer sharding until storage forces it**, not throughput — the cache already solved throughput. Reach for [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) when the mapping outgrows one node, around 10×.
- **Spend reliability budget asymmetrically:** make the redirect path survive a region loss (multi-region, replicated, cache-fronted); let the create path degrade. The user pain lives on reads.
- **Match the design to provable scale, not assumed scale.** The full architecture is correct at billions; it's gold-plating at thousands. Knowing which you're building is the senior skill the "easy" question is really testing.

## Further reading

- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the QPS, storage, and cache math we applied here, derived in full.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — cache-aside, stampedes, negative caching, and the hot-key problem in depth.
- [Rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) — token buckets, distributed limiters, and the queue-as-shock-absorber pattern.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — the playbook for when the mapping finally outgrows one node.
- [Choosing a datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — why the mapping wants SQL and the click stream wants a column store.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — the reasoning frame behind every decision matrix above.
- [Design a news feed and timeline](/blog/software-development/system-design/design-a-news-feed-and-timeline) — the next case study, where the write path is the hard part.
- The Snowflake ID announcement and Bitly/Twitter engineering talks — primary sources for the distributed-ID and shortener-at-scale patterns referenced in the case studies.

---
title: "Cross-Shard Queries and Distributed Joins: The Tax of Sharding"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Once you shard, any query that omits the shard key fans out to every shard, and joins across shards become the hard problem — here is the cost model, the five techniques to keep joins local, and the router patterns that make scatter-gather survivable."
tags: ["database-scaling", "sharding", "distributed-joins", "scatter-gather", "cross-shard-queries", "vitess", "tail-latency", "two-phase-commit", "saga", "data-modeling"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 35
---

Sharding is the move you make when one machine can no longer hold your data or serve your writes. You pick a shard key, split the rows across N database servers, and your write throughput and storage capacity scale roughly linearly. It feels like a clean win. Then, a few weeks later, somebody runs `SELECT * FROM users WHERE email = ?` against the sharded `users` table — a query that worked instantly when there was one database — and it takes 400 ms, opens a connection to all 64 shards, and pins three of them at 100% CPU. Welcome to the tax of sharding.

This is the part of sharding nobody warns you about loudly enough. Splitting the data is the easy half. The hard half is that the moment your rows live on different machines, two whole classes of query that you took for granted become expensive: queries that don't carry the shard key, and joins between tables that don't live together. The first turns one logical query into N physical queries the router must fan out and merge — a *scatter-gather*. The second is genuinely hard, because the rows you want to join are sitting on different computers connected by a network that is a hundred thousand times slower than the memory bus a single-node join used to run on.

![Targeted query versus scatter-gather: with the shard key the router touches one shard; without it, the query fans out to all N shards and merges](/imgs/blogs/cross-shard-queries-and-distributed-joins-1.webp)

The diagram above is the mental model, and it's the spine of this entire post. A query that carries the shard key in its predicate — `WHERE user_id = 42` when you shard by `user_id` — is *targeted*: the router computes which shard owns key 42, sends the query to exactly that one shard, gets one result, and replies. One connection, one round trip, single-shard latency. A query that omits the shard key — `WHERE email = ?` — gives the router no way to narrow down which shard holds the matching row. So it must broadcast the query to every shard, wait for all of them, and merge the results. One logical query, N physical queries. Everything in this post is about that fork in the road: how to recognize which side a query falls on, what the scatter-gather side actually costs, and the engineering moves that keep you on the targeted side as much as possible.

We'll build the cost model for scatter-gather, walk through why cross-shard joins are the thing sharding makes genuinely hard, lay out the five techniques for handling them, cover cross-shard aggregations and the painful topic of cross-shard transactions, and finish with the query-router pattern that ties it together and a set of production war stories. If you haven't sharded yet, read [Database Partitioning and Sharding](/blog/software-development/database/database-partitioning-and-sharding) and [Choosing a Shard Key](/blog/software-development/database-scaling/choosing-a-shard-key) first — the shard key you pick is the single biggest lever over how much of this tax you pay.

## 1. Targeted vs scatter-gather: the one distinction that matters

**Senior rule of thumb: a query's cost on a sharded cluster is determined almost entirely by one bit — does its predicate pin down the shard key or not.**

Let's make the two paths concrete. Suppose you shard `users` across 64 shards by `hash(user_id) % 64`. The router maintains a mapping from the hash range to a physical shard.

A targeted query:

```sql
-- Router can compute: hash(42) % 64 = shard 17. Send to shard 17 only.
SELECT name, email FROM users WHERE user_id = 42;
```

A scatter-gather query:

```sql
-- email is not the shard key. Router has no idea which shard(s) hold it.
-- Must broadcast to all 64 shards, collect 64 result sets, merge.
SELECT user_id, name FROM users WHERE email = 'ada@example.com';
```

The targeted query touches one shard. The scatter-gather query touches 64. That 64× difference in fan-out is the headline, but it is not the worst part. The worst part is what fan-out does to tail latency.

### Tail-latency amplification: you wait for the slowest shard

A scatter-gather query does not finish when the *average* shard finishes. It finishes when the *slowest* shard finishes, because the router cannot return a complete, correctly-merged result until every shard has reported in. This is tail-latency amplification, and it is the single most under-appreciated cost of fan-out.

<figure class="blog-anim">
<svg viewBox="0 0 760 320" role="img" aria-label="A scatter-gather query fans out to four shards; three reply quickly but the merge cannot complete until the slowest shard returns, so total latency tracks the slowest shard" style="width:100%;height:auto;max-width:820px">
<title>Scatter-gather tail-latency amplification: the query waits for the slowest shard</title>
<style>
.csq-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.csq-router{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2.5}
.csq-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.csq-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.csq-track{stroke:var(--border,#d1d5db);stroke-width:2;fill:none}
.csq-dot{fill:var(--accent,#6366f1)}
.csq-slow{fill:#e8590c}
@keyframes csq-f1{0%{cx:170px;opacity:0}6%{opacity:1}30%{cx:330px}34%,100%{cx:330px;opacity:.25}}
@keyframes csq-f2{0%{cx:170px;opacity:0}6%{opacity:1}40%{cx:330px}44%,100%{cx:330px;opacity:.25}}
@keyframes csq-f3{0%{cx:170px;opacity:0}6%{opacity:1}50%{cx:330px}54%,100%{cx:330px;opacity:.25}}
@keyframes csq-slow{0%{cx:170px;opacity:0}6%{opacity:1}92%{cx:330px}96%,100%{cx:330px;opacity:1}}
@keyframes csq-ret{0%,55%{opacity:0;cx:430px}60%{opacity:0;cx:430px}96%{opacity:0;cx:430px}100%{opacity:1;cx:600px}}
.csq-p1{animation:csq-f1 7s ease-in-out infinite}
.csq-p2{animation:csq-f2 7s ease-in-out infinite}
.csq-p3{animation:csq-f3 7s ease-in-out infinite}
.csq-ps{animation:csq-slow 7s ease-in-out infinite}
.csq-result{animation:csq-ret 7s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.csq-p1,.csq-p2,.csq-p3,.csq-ps,.csq-result{animation:none;opacity:1}}
</style>
<rect class="csq-router" x="40" y="120" width="130" height="80" rx="10"/>
<text class="csq-lbl" x="105" y="155">router</text>
<text class="csq-sub" x="105" y="175">fan out</text>
<line class="csq-track" x1="170" y1="55" x2="330" y2="55"/>
<line class="csq-track" x1="170" y1="120" x2="330" y2="120"/>
<line class="csq-track" x1="170" y1="200" x2="330" y2="200"/>
<line class="csq-track" x1="170" y1="265" x2="330" y2="265"/>
<rect class="csq-box" x="330" y="30" width="140" height="50" rx="8"/>
<text class="csq-lbl" x="400" y="52">shard 1</text>
<text class="csq-sub" x="400" y="70">3 ms</text>
<rect class="csq-box" x="330" y="95" width="140" height="50" rx="8"/>
<text class="csq-lbl" x="400" y="117">shard 2</text>
<text class="csq-sub" x="400" y="135">5 ms</text>
<rect class="csq-box" x="330" y="175" width="140" height="50" rx="8"/>
<text class="csq-lbl" x="400" y="197">shard 3</text>
<text class="csq-sub" x="400" y="215">7 ms</text>
<rect class="csq-box" x="330" y="240" width="140" height="50" rx="8" stroke="#e8590c"/>
<text class="csq-lbl" x="400" y="262">shard 4</text>
<text class="csq-sub" x="400" y="280" fill="#e8590c">90 ms (slow)</text>
<rect class="csq-router" x="560" y="120" width="160" height="80" rx="10"/>
<text class="csq-lbl" x="640" y="150">merge</text>
<text class="csq-sub" x="640" y="170">waits for</text>
<text class="csq-sub" x="640" y="187">slowest shard</text>
<circle class="csq-dot csq-p1" cx="170" cy="55" r="8"/>
<circle class="csq-dot csq-p2" cx="170" cy="120" r="8"/>
<circle class="csq-dot csq-p3" cx="170" cy="200" r="8"/>
<circle class="csq-slow csq-ps" cx="170" cy="265" r="8"/>
<circle class="csq-dot csq-result" cx="430" cy="160" r="9"/>
</svg>
<figcaption>Three shards reply in 3 to 7 ms but the merge cannot complete until shard 4 returns at 90 ms, so the query's latency tracks the slowest shard, not the average.</figcaption>
</figure>

Here's the math that makes this brutal. Suppose each shard independently has a p99 latency of 10 ms — that is, 1% of requests to a single shard take longer than 10 ms. If a query fans out to one shard, the probability it hits that slow tail is 1%. If it fans out to 64 shards and must wait for all of them, the probability that *at least one* shard is slow is $1 - 0.99^{64} \approx 47\%$. Nearly half of your fanned-out queries now hit the slow tail of *some* shard. Fan out to 100 shards and it's $1 - 0.99^{100} \approx 63\%$. The more you shard, the worse this gets — your cluster's effective p99 for scatter queries converges toward the p99.99 of an individual shard.

This is why a query that benchmarks beautifully on a single database can quietly become a latency disaster the day you shard. Nothing about the query changed. The fan-out did.

### The two other costs: wasted throughput and connection fan-out

Tail latency is the headline, but two more costs ride along.

**Wasted throughput.** A scatter-gather query makes every shard do work for a single logical request. If you have 64 shards and run a scatter query that each shard answers in 5 ms of CPU, you've burned 320 ms of aggregate database CPU to answer one request. Run 1,000 such queries per second and you've consumed 320 seconds of CPU per second — you need 320 cores just for the scatter queries, regardless of how little data each returns. Targeted queries scale *out* (more shards = more capacity); scatter queries scale *down* (more shards = each query costs more aggregate work).

**Connection fan-out.** Every scatter query opens (or borrows from a pool) one connection per shard. With 64 shards and 500 concurrent scatter queries, that's 32,000 simultaneous backend connections. Databases have hard connection limits — Postgres defaults to 100, and even tuned it struggles past a few thousand without a pooler like PgBouncer. The router becomes a connection amplifier, and connection exhaustion on the shards is a classic scatter-gather failure mode.

| | Targeted query | Scatter-gather query |
|---|---|---|
| Shards touched | 1 | all N |
| Latency tracks | that shard's p50/p99 | the **slowest** shard's tail |
| p99 with 64 shards @ 10ms shard-p99 | ~10 ms | dominated by ~p99.99 of a shard |
| Aggregate CPU per query | 1× | N× |
| Backend connections | 1 | N |
| Scales out with more shards? | yes | no — gets worse |

The takeaway is blunt: **on a sharded cluster, every query you can keep targeted is a query that scales; every scatter query is a tax you pay forever.** The rest of this post is the toolkit for paying as little of that tax as possible.

## 2. Why cross-shard joins are the genuinely hard problem

Scatter-gather on a single table is annoying but mechanical — the router broadcasts, collects, concatenates. Joins are a different beast, and they are the thing sharding makes *genuinely* hard.

**Senior rule of thumb: a join is cheap when both sides of it live on the same machine, and expensive — sometimes pathologically so — the instant they don't.**

On a single database, a join between `users` and `orders` is a local operation. The query planner picks a hash join or a nested-loop join, both tables' pages are in the same buffer pool, and the whole thing runs at memory-bus speed. Now shard `users` by `user_id` and `orders` by `order_id`. User 42 lives on shard 17. Her orders are scattered across all 64 shards according to their order IDs. To compute `users JOIN orders ON users.user_id = orders.user_id` for user 42, the database would have to reach across the network to gather order rows from up to 64 different machines and stitch them together. There is no single planner that owns both sides; there is no shared buffer pool; there is a network in the middle.

The network is the whole problem. A local hash join probes memory in tens of nanoseconds. A cross-shard join probes another machine over TCP in hundreds of microseconds to milliseconds — four to five orders of magnitude slower. A naive distributed join that ships rows around to find matches can produce enormous intermediate result sets: if you join two 64-shard tables with no shard-key alignment, the worst case is shipping every row of one table to every shard of the other, an $O(N^2)$ shuffle that will saturate your network and time out.

This is why mature sharded systems — Vitess, Citus, MongoDB's sharded clusters, every hand-rolled sharding layer at scale — treat cross-shard joins as something to *design away*, not something to execute well. You cannot make a cross-shard join fast in the general case. You can only arrange your data so the join doesn't need to cross shards, or replace it with something cheaper. There are five ways to do that, and a senior engineer reaches for them in roughly this order of preference.

![Five ways to handle a cross-shard join, compared on whether the join stays local, storage cost, and consistency](/imgs/blogs/cross-shard-queries-and-distributed-joins-3.webp)

The matrix above is the decision surface. Co-location is the only technique that keeps the join local for free, with no extra storage and strong consistency — so it's the default you reach for first. Everything below it trades something: storage duplication, eventual consistency, an extra lookup hop, or application complexity. Let's walk through each.

## 3. Technique 1: co-location — shard related tables by the same key

**Senior rule of thumb: if two tables are routinely joined, shard them by the same key so the join never leaves the box.**

This is the single most important technique, and it's the reason shard-key choice is so consequential. If you shard both `users` and `orders` by `user_id`, then user 42 *and all of her orders* live on the same shard. The join `users JOIN orders ON users.user_id = orders.user_id WHERE user_id = 42` becomes a single-shard, in-process join — exactly as fast as it was before you sharded.

![Co-location keeps the join on one machine: shard users and orders by the same user_id and the join stays local; shard them differently and every join is a network hop](/imgs/blogs/cross-shard-queries-and-distributed-joins-4.webp)

The diagram makes the difference physical. On the left, both tables are sharded by `user_id`, so for any given user both her row and her orders sit inside Shard 3; the join runs in-process with zero network hops. On the right, `users` is sharded by `user_id` and `orders` by `order_id`, so user 42's row is on Shard 3 while her orders are on Shard 7 (and others) — the join now crosses the network and has to be stitched together by the application or a distributed planner.

Vitess formalizes this with **vindexes** and **table groups** (called "keyspaces" and sequences in its model); Citus calls it **co-located distribution** and enforces it with `create_distributed_table('orders', 'user_id', colocate_with => 'users')`. The mechanism differs but the principle is identical: pick a shared key for the tables in a join cluster, and the router can route the whole join to one shard.

Here's the Citus version of getting co-location right:

```sql
-- Both tables distributed by the same column => co-located.
SELECT create_distributed_table('users',  'user_id');
SELECT create_distributed_table('orders', 'user_id', colocate_with => 'users');

-- This join is now pushed down to a single worker per user_id. No shuffle.
SELECT u.name, o.total
FROM users u
JOIN orders o ON o.user_id = u.user_id
WHERE u.user_id = 42;
```

The cost of co-location is that it constrains your shard-key choice: you can only co-locate tables that share a natural join key. A schema usually has one dominant entity that most queries hang off of — the user, the tenant, the account, the merchant. Sharding everything by that entity's ID co-locates the queries that matter. The classic failure is sharding `users` by `user_id` and `orders` by `order_id` because "orders are queried by order ID too" — now the user-to-orders join is cross-shard forever. The fix is almost always: shard `orders` by `user_id` and add a secondary lookup for the rarer order-ID access path (technique 4).

> Co-location is free. It costs nothing at read time, nothing in storage, and gives you strong consistency. The only thing it costs is discipline at schema-design time. Spend that discipline.

### Second-order optimization: the transactional unit defines the shard key

The deepest reason to co-locate isn't joins — it's transactions, which we'll get to in section 7. If a business operation needs to atomically touch a user's row and her orders, co-locating them means that transaction stays single-shard and gets ACID for free. Choose your shard key so the unit of *transactional consistency* lives on one shard, and joins-stay-local falls out as a happy side effect.

## 4. Technique 2: denormalization — embed the fields you'd join on

**Senior rule of thumb: if a join exists only to fetch a couple of columns, copy those columns into the table that needs them and delete the join.**

A lot of joins aren't really "joins" in the relational sense — they're lookups to grab a display field. `SELECT o.*, u.name FROM orders o JOIN users u ON o.user_id = u.user_id` exists only to put the user's name next to the order. If you denormalize and store `user_name` directly on the `orders` row, the join disappears entirely. The order row is self-contained; reading it is a targeted query on the orders shard.

```sql
-- Before: cross-shard join to get the display name.
SELECT o.order_id, o.total, u.name
FROM orders o JOIN users u ON o.user_id = u.user_id
WHERE o.order_id = 9001;  -- order on shard A, user on shard B

-- After: denormalize user_name onto the order. No join, single shard.
SELECT order_id, total, user_name FROM orders WHERE order_id = 9001;
```

The cost is duplication and the consistency problem it creates: when a user changes their name, every denormalized copy is now stale until you propagate the update. This is fine for fields that are immutable or rarely change (the user's name at time of order, a product's name and SKU on a line item, the currency code) and dangerous for fields that change often and must be current (account balance, current address for a pending shipment).

The discipline: **denormalize point-in-time facts, not live state.** An order should record "the customer's name *was* Ada Lovelace when they placed this order" — that's a fact that never changes and is exactly what you want on a receipt. It should *not* denormalize "the customer's current loyalty tier," because that changes and you'll serve stale data. NoSQL stores like DynamoDB lean on this so heavily that denormalization is the default modeling style, not an optimization.

### Second-order optimization: keep a propagation path even for "immutable" fields

Even point-in-time denormalized data sometimes needs correction — a user's name was entered wrong, GDPR requires you to scrub it. Build a backfill path (a job that re-reads the source of truth and updates copies) before you need it, because reconstructing which rows hold a stale copy after the fact is miserable. A change-data-capture stream off the source table feeding an updater is the clean version.

## 5. Technique 3: application-side join — fetch from each shard and join in the app

**Senior rule of thumb: when you can't co-locate or denormalize, do the join in your application code, in two cheap targeted steps instead of one expensive distributed one.**

Sometimes you genuinely need to join two tables that are sharded differently, and you can't redesign the schema. The pragmatic move is to not ask the database to do the join at all. Instead: run one targeted query to get the driving rows, extract the foreign keys, then run targeted (or batched) queries against the other table's shards to fetch the matching rows, and stitch them together in application memory.

The key is to make both steps *targeted*. Get user 42's orders (targeted on the orders shard if co-located, or a scatter if not), collect the `product_id`s, then fetch those products by ID — and if `products` is sharded by `product_id`, those fetches are targeted too. You've replaced one cross-shard join with two sets of point lookups, which the database is extremely good at.

When the second step fans across shards, run the per-shard fetches *concurrently* so you pay the latency of the slowest shard once, not the sum. Here's a concurrent scatter-gather with merge in Python `asyncio`:

```python
import asyncio
from collections import defaultdict

async def fetch_from_shard(shard, ids):
    """Targeted multi-get against one shard. Returns {id: row}."""
    rows = await shard.query(
        "SELECT product_id, name, price FROM products WHERE product_id = ANY($1)",
        list(ids),
    )
    return {r["product_id"]: r for r in rows}

async def app_side_join(orders, router, *, concurrency=16):
    # 1. Group the foreign keys we need by which shard owns them.
    by_shard = defaultdict(set)
    for o in orders:
        by_shard[router.shard_for(o["product_id"])].add(o["product_id"])

    # 2. Fan out to the relevant shards *concurrently*, bounded so we don't
    #    open thousands of connections at once. We pay the slowest shard once.
    sem = asyncio.Semaphore(concurrency)
    async def guarded(shard, ids):
        async with sem:
            return await fetch_from_shard(shard, ids)
    results = await asyncio.gather(*(
        guarded(router.shard(s), ids) for s, ids in by_shard.items()
    ))

    # 3. Merge the partial maps and stitch the join in application memory.
    products = {}
    for partial in results:
        products.update(partial)
    return [
        {**o, "product": products.get(o["product_id"])}
        for o in orders
    ]
```

Three things make this version production-grade. It **batches** foreign keys per shard so each shard gets one `WHERE product_id = ANY(...)` instead of one query per ID (avoiding the N+1 trap). It **bounds concurrency** with a semaphore so a join over thousands of keys doesn't open thousands of connections. And it **gathers concurrently** so total latency is the slowest single shard's latency, not the sum across shards.

The cost is application complexity and the loss of the database's join optimizer — you're hand-writing the join strategy. For simple lookups this is fine and often *faster* than a distributed planner, because you know the access pattern and the planner doesn't. For complex multi-way joins with selective filters, you're reimplementing a query engine badly, and that's the signal to reconsider the schema (or move that workload to an analytical store; see section 9's notes on offloading).

### Second-order optimization: cache the dimension side

If the second step repeatedly fetches the same small set of rows — product catalog, currency table, feature flags — put a cache in front of it. An application-side join against a cached dimension is just a local hash lookup. This shades into technique 5 (reference-data replication) and into the broader caching story in [Cache Patterns in Production](/blog/software-development/database-scaling/cache-patterns-in-production).

## 6. Techniques 4 and 5: global indexes and reference-data replication

Two more tools round out the toolkit, for two specific shapes of problem.

### Technique 4: a secondary / global index

The problem: you shard `users` by `user_id`, but you also need to look users up by `email`, which would otherwise be a scatter. The fix is a **global secondary index** — a separate, smaller table that maps the alternate key to the shard key, itself sharded by the alternate key.

```sql
-- Index table, sharded by email (the lookup key).
CREATE TABLE users_by_email (
    email    TEXT PRIMARY KEY,   -- sharded on this
    user_id  BIGINT NOT NULL     -- the real shard key
);
```

Now `WHERE email = ?` becomes two *targeted* queries instead of one scatter: first hit `users_by_email` (sharded by email) to get the `user_id`, then hit `users` (sharded by `user_id`) to get the row. Two single-shard round trips beat a 64-shard fan-out every time. This is exactly how DynamoDB Global Secondary Indexes and Vitess "lookup vindexes" work under the hood — the index is itself a sharded table on the alternate key.

The cost is a second write on every insert/update of the indexed field (you must keep the index table in sync) and eventual consistency between the base table and the index unless you wrap both writes in a transaction (which, if they're on different shards, is itself a cross-shard transaction — see section 7). Most systems accept eventual consistency here: a brief window where a newly created user isn't yet findable by email is usually tolerable; a lost write is not, so the index write must be durable and retried.

### Technique 5: reference-data replication

The problem: you have a small, read-mostly dimension table — `currencies`, `countries`, `product_categories`, a feature-flag table — that lots of sharded queries want to join against. It's too small to shard meaningfully and joining to it from every shard would be cross-shard. The fix is to **replicate the whole table to every shard**.

![Reference-data replication: a small read-mostly currencies table copied to every shard turns every orders-to-currencies join into a single-shard join](/imgs/blogs/cross-shard-queries-and-distributed-joins-6.webp)

The diagram shows the shape: a master `currencies` table (a couple hundred rows, rarely written) is copied into every shard alongside the sharded `orders` table. Now `orders JOIN currencies` is a local join on *every* shard, because both tables are present everywhere. Vitess calls these **reference tables** and replicates them automatically; Citus has `create_reference_table('currencies')`, which materializes a full copy on every worker node.

```sql
-- Citus: full copy on every worker. Joins to it are always local.
SELECT create_reference_table('currencies');

-- Now this pushes down to each shard with a local join, no shuffle:
SELECT o.order_id, o.amount * c.usd_rate AS amount_usd
FROM orders o JOIN currencies c ON c.code = o.currency_code
WHERE o.user_id = 42;
```

The cost is storage (N copies of the table — fine for 200 rows, absurd for 200 million) and replication lag (an update to the master must propagate to every copy before all shards agree). The rule of thumb: replicate tables that are **small and read-mostly**. A currency table updated once a quarter is perfect. A "user sessions" table is the opposite of a reference table and must never be one.

| Technique | Keeps join local? | Storage cost | Consistency | When to use |
|---|---|---|---|---|
| **Co-location** | Yes | None | Strong | Tables always joined on a shared key; the default first choice |
| **Denormalization** | Yes | Duplicated columns | Eventual (point-in-time) | Join only fetches a few immutable fields |
| **Application-side join** | No (done in app) | None | Read-time | Can't redesign schema; simple lookups; need control of strategy |
| **Global index** | Lookup hop (2 targeted) | Index table | Eventual | Frequent lookups by a non-shard-key column |
| **Reference replication** | Yes | Copy × N shards | Eventual | Small, read-mostly dimension joined everywhere |

## 7. Cross-shard aggregations: count, sum, and top-N

Aggregations across shards are scatter-gather with a merge step that's smarter than concatenation. The pattern is **partial aggregation**: push as much of the aggregation as possible into each shard, then combine the partial results in the router.

**Senior rule of thumb: make each shard do its own local aggregation and ship a summary, so the router moves bytes, not whole tables.**

For `SUM` and `COUNT`, this is straightforward and exact: each shard computes its local sum or count, the router adds them up. `SUM(local sums) = global sum`. Same for `MIN`/`MAX`. `AVG` needs a little care — you can't average the averages; each shard must return `(sum, count)` and the router computes `Σsum / Σcount`. These are *decomposable* aggregations, and a good router rewrites them automatically.

The interesting case is **top-N** (`ORDER BY x LIMIT k`). You might think you need every row to find the global top 10, but you don't. If you want the global top 10 by sales, the global top 10 must be contained in the *union of each shard's local top 10*. So each shard sorts locally, returns only its top 10, and the router does a k-way merge of N short sorted lists and re-truncates to 10.

![Distributed top-N by partial aggregation: each shard returns its own top-10, the router merges N sorted lists and re-truncates, never pulling full rows](/imgs/blogs/cross-shard-queries-and-distributed-joins-5.webp)

The pipeline above is the whole algorithm. The router broadcasts `ORDER BY sales LIMIT 10` to every shard; each shard returns only its local top 10 (its *partial* result), so the wire carries `10 × N` rows instead of the entire table; the router merges those N pre-sorted lists with a heap of size N and re-truncates to the global top 10; then returns 10 rows to the client. The data shipped is bounded by `k × N`, not by table size — that's the win.

```python
import heapq

def merge_top_n(shard_results, n, key):
    """shard_results: list of per-shard lists, each already sorted desc by key.
    Returns the global top-n. Ships only k*N rows total, never full tables."""
    # heapq.merge is a lazy k-way merge of sorted iterables.
    # Each shard list must be sorted by -key so the merge yields descending order.
    merged = heapq.merge(
        *(sorted(rows, key=lambda r: key(r), reverse=True) for rows in shard_results),
        key=lambda r: key(r),
        reverse=True,
    )
    out = []
    for row in merged:
        out.append(row)
        if len(out) == n:
            break
    return out

# Each shard already returned its local top-10 by sales:
shard_results = [
    [{"merchant": "A", "sales": 980}, {"merchant": "B", "sales": 910}],
    [{"merchant": "C", "sales": 999}, {"merchant": "D", "sales": 870}],
    [{"merchant": "E", "sales": 950}, {"merchant": "F", "sales": 860}],
]
print(merge_top_n(shard_results, n=3, key=lambda r: r["sales"]))
# -> [{'merchant': 'C', 'sales': 999}, {'merchant': 'A', 'sales': 980},
#     {'merchant': 'E', 'sales': 950}]
```

There's a subtlety: top-N with a `GROUP BY` is *not* fully decomposable this way if a group spans shards. If you want the top 10 merchants by total sales and a merchant's sales are split across shards, each shard's local top 10 might miss a merchant who is rank 11 on every shard but rank 1 globally. The correct (and expensive) answer requires each shard to return *all* groups' partial sums, then the router aggregates and ranks. The cheap-but-approximate answer is to over-fetch (each shard returns its top 100, not top 10) and accept a small error probability. Know which one your query needs — this is a common source of subtly wrong dashboards.

### Approximate cardinality with HyperLogLog

`COUNT(DISTINCT user_id)` across shards is genuinely hard exactly: a user could appear on multiple shards, so you can't sum per-shard distinct counts (you'd double-count), and shipping every distinct ID to deduplicate defeats the purpose. The standard answer is **HyperLogLog (HLL)**: each shard computes a small (a few KB) HLL sketch of its distinct values, the router merges the sketches (HLL sketches are mergeable by taking the element-wise max of their registers), and reads off the estimated cardinality — typically within ~2% error for a 1–2 KB sketch.

```sql
-- Postgres with the hll extension, pushed to each shard:
SELECT hll_add_agg(hll_hash_bigint(user_id)) AS sketch
FROM events WHERE day = '2026-06-30';     -- each shard returns a small sketch

-- Router merges the per-shard sketches and reads the estimate:
SELECT hll_cardinality(hll_union_agg(sketch)) FROM shard_sketches;
```

This turns an impossible exact distributed `COUNT(DISTINCT)` into a few KB per shard and a constant-time merge. Use it for analytics-grade "how many unique users" numbers where 2% error is invisible; don't use it for billing.

## 8. Cross-shard transactions: the hardest tax of all, and how to avoid it

A query that *reads* across shards is expensive. A transaction that *writes* across shards is dangerous, because you now need atomicity across machines — either all the writes commit or none do — over an unreliable network.

**Senior rule of thumb: the cheapest distributed transaction is the one you didn't have to do, because you co-located its rows on a single shard.**

![Handling a transaction that spans shards: co-locate to keep it single-shard, or fall back to blocking 2PC or compensating sagas](/imgs/blogs/cross-shard-queries-and-distributed-joins-8.webp)

The decision tree above is the order you should think in. The first question is never "2PC or saga?" — it's "does this write genuinely need to span shards?" If you can co-locate the rows the transaction touches onto one shard, the transaction stays single-shard and you get full ACID from the local database engine, for free. A user's account balance and her transaction ledger, sharded together by `user_id`, make a transfer-within-an-account atomic with a plain `BEGIN; ... COMMIT;`. This is the **"shard so transactions stay local"** principle, and it should drive your shard-key choice more than join patterns do.

When a transaction genuinely spans shards — moving money between two users on different shards, say — you have two real options, both with sharp edges.

**Two-phase commit (2PC).** A coordinator asks every participant "can you commit?" (prepare phase), and only if all say yes does it tell them "commit" (commit phase). It gives you atomicity, but it is a blocking protocol: if the coordinator crashes after participants have prepared but before it sends the commit decision, those participants are stuck holding locks, unable to commit or abort, until the coordinator recovers. This is the famous 2PC blocking problem. It also serializes the slowest participant into every transaction and holds locks across two network round trips, which crushes throughput under contention. 2PC is correct and occasionally necessary, but it is not how you build a high-throughput system. The full failure analysis is in [Two-Phase Commit and How It Fails](/blog/software-development/database/two-phase-commit-and-how-it-fails).

**Sagas.** Instead of one atomic transaction, model the operation as a sequence of *local* transactions, each on a single shard, with a compensating action for each step that undoes it if a later step fails. Debit user A (local commit on shard 17); credit user B (local commit on shard 31); if the credit fails, run the compensation "refund user A." Sagas never hold cross-shard locks and scale far better than 2PC, but they give up isolation: there are intermediate states where the money has left A but not yet arrived at B, visible to other readers. You must design for those intermediate states (pending balances, idempotent steps, retries). The pattern and its pitfalls are covered in [The Saga Pattern for Distributed Transactions](/blog/software-development/database/saga-pattern-distributed-transactions).

| Approach | Atomic? | Isolation? | Locks held | Throughput | When to use |
|---|---|---|---|---|---|
| **Co-locate (avoid)** | Yes (local ACID) | Yes | Single shard, brief | High | Always prefer; design the shard key for this |
| **2PC** | Yes | Yes (with locking) | Across shards, 2 round trips | Low | Rare, correctness-critical cross-shard writes |
| **Saga** | Eventually (compensations) | No | Per-step, local only | High | Long-running or high-volume cross-shard workflows |

The honest summary: you should be doing cross-shard transactions almost never. If your design requires them on a hot path, that's usually a sign the shard key is wrong — the transactional unit got split across shards. Revisit the key before you reach for 2PC.

## 9. The query-router / aggregator pattern

Everything above — routing targeted queries, fanning out scatters, merging partials, decomposing aggregations — has to live *somewhere*. That somewhere is the **query router** (also called the aggregator, the coordinator, or the proxy layer).

![The query-router pattern: a stateless router parses each query, decides targeted vs scatter, fans out to shards, and merges before replying](/imgs/blogs/cross-shard-queries-and-distributed-joins-7.webp)

The architecture in the diagram is the canonical shape. The application speaks plain SQL to a stateless router (in Vitess this is `vtgate`). The router parses the query, consults its routing metadata (which column is the shard key, which shard owns which key range), and decides: targeted → send to one shard; scatter → fan out to all. For scatters it collects the partial results and runs the merge/sort/aggregate logic, then returns a single result set. The application never knows whether its query hit one shard or sixty — the router hides the sharding entirely.

This is why **Vitess** is the reference implementation of sharded MySQL at scale: `vtgate` is exactly this router, and it does query rewriting (decomposing `AVG` into `SUM`/`COUNT`, pushing `LIMIT` into shards, planning which joins can be pushed down vs which need a scatter), connection pooling to the shards, and routing-metadata management. The router is stateless, so you run many of them behind a load balancer; the routing intelligence is the value. We'll dig into how Vitess does this in production in the [Slack and YouTube on Vitess](/blog/software-development/database-scaling/slack-and-youtube-on-vitess) case study.

You don't always need a full Vitess deployment. Many teams start with a thin custom router — a library or sidecar that knows the shard map and routes targeted queries, while refusing or specially handling scatters. The danger of a custom router is that it's easy to make targeted queries work and then quietly let scatter queries through unbounded; the discipline is to make the router *explicit* about which queries scatter, ideally logging or rejecting unbounded scatters in production.

### Tail-latency mitigations for the scatters you can't avoid

Some scatters are unavoidable (an admin search, a rare analytics query). For those, the router has three levers to fight the tail-latency amplification from section 1.

**Hedged requests.** If a shard hasn't responded by the time it's slower than, say, p95, send a duplicate request to a replica of that shard and take whichever returns first. This trades a little extra load for a dramatically tighter tail — Google's "The Tail at Scale" paper showed hedging can cut p99 fan-out latency by more than half for a few percent more requests.

```python
async def hedged_query(primary, replica, sql, params, *, hedge_after=0.05):
    """Send to primary; if it hasn't answered in hedge_after seconds, also
    ask a replica and take the first to finish."""
    primary_task = asyncio.create_task(primary.query(sql, params))
    done, _ = await asyncio.wait({primary_task}, timeout=hedge_after)
    if primary_task in done:
        return primary_task.result()
    # Primary is slow; race it against a replica.
    replica_task = asyncio.create_task(replica.query(sql, params))
    done, pending = await asyncio.wait(
        {primary_task, replica_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()
    return next(iter(done)).result()
```

**Query a subset.** For approximate results (trending items, sampled analytics) you don't need every shard. Query a random subset — say 8 of 64 shards — and scale the result. You trade exactness for a fixed fan-out and a bounded tail.

**Hard limits.** Always put a `LIMIT` and a timeout on scatter queries, and cap how many rows each shard may return. An unbounded scatter (`SELECT * FROM orders` with no shard key and no limit) is a cluster-killer — it makes every shard stream its entire table to the router, exhausting memory and connections. A defensive router rejects scatters that lack a limit.

## 10. Case studies from production

### 1. The email login that took down the cluster

A B2C app sharded `users` by `user_id` and shipped happily for a year. Then they added "log in with email." The login path ran `SELECT * FROM users WHERE email = ?` — a scatter across all 48 shards on *every login attempt*. At low traffic nobody noticed. During a marketing push, login QPS spiked, the 48× connection amplification exhausted the shards' connection pools, and *every* query (including targeted ones) started failing because there were no connections left. The fix was technique 4: a `users_by_email` lookup table sharded by email, turning login into two targeted queries. The lesson: a single innocuous-looking scatter on a hot path can take down the whole cluster through connection exhaustion, not just slow itself down.

### 2. The dashboard that was subtly wrong for months

An analytics dashboard showed "top 20 products by revenue." It was implemented as a scatter that asked each of 32 shards for its local top 20, then merged. Products were sharded by `product_id`, so each product lived entirely on one shard — and the answer was correct. Then they re-sharded `orders` by `user_id` for co-location with users, which split each product's line items across shards. The top-20 query kept returning each shard's local top 20 and merging, but now a product that was rank 21 on every shard yet rank 1 globally never appeared. The dashboard was wrong for months before a finance analyst noticed the numbers didn't reconcile. The lesson from section 7: top-N with a `GROUP BY` whose groups span shards is not decomposable by local top-N; you must aggregate all groups or over-fetch deliberately.

### 3. Friendster, MySpace, and the join that didn't scale

The early social networks are the cautionary tale that motivated the whole modern sharding playbook. Friendster's collapse in the mid-2000s is widely attributed to social-graph queries — "friends of friends" — that were inherently cross-shard joins over a graph that didn't partition cleanly. Every profile view triggered joins that fanned out across the data tier, and as the user base grew the fan-out grew with it until pages took tens of seconds to load and users left. The lesson is the one this whole post is built on: some access patterns (especially graph traversals) resist co-location, and if your core read path is a cross-shard join, no amount of hardware saves you. Facebook's later answer — TAO, a purpose-built graph cache in front of sharded MySQL — was an admission that the graph join had to be served from a denormalized, co-located structure, not computed across shards on read.

### 4. Instagram's deliberately boring shard key

Instagram, scaling on Postgres, sharded by user ID and embedded the shard ID directly into their generated primary keys (the famous custom ID scheme: timestamp + shard ID + sequence). Because the shard was derivable from any object's ID, and because they co-located a user's photos and metadata on the user's shard, the overwhelming majority of their queries were targeted single-shard lookups. They consciously avoided features that would require cross-shard joins on the hot path. The lesson is the positive version of Friendster: pick a shard key that matches your dominant access pattern, co-locate aggressively, and most of your traffic never pays the scatter tax at all.

### 5. The 2PC that wedged on a coordinator crash

A payments system used 2PC to move balances between accounts on different shards. It worked under test and light load. In production, the coordinator process was OOM-killed mid-transaction — after participants had prepared (and taken locks) but before the commit decision was sent. Those rows on two shards were now locked, held by an in-doubt transaction, and *no other transaction touching those accounts could proceed*. Because the coordinator's recovery log was on the same box that got killed, recovery took 40 minutes, during which a slice of accounts was frozen. The fix was to redesign so the transactional unit (an account and its ledger) was co-located on one shard, making intra-account transfers single-shard ACID, and to move cross-account transfers to a saga with explicit pending states. The lesson: 2PC's blocking failure mode is not theoretical, and a coordinator crash at the wrong instant freezes data.

### 6. The N+1 application-side join

A team replaced a cross-shard join with an application-side join — the right move — but implemented step two as a loop: for each of the 500 orders in a page, fetch its product with a separate query. That's 500 sequential round trips, and the page took 6 seconds. The join was "local" in the sense that no single query was a scatter, but the latency was worse than the distributed join had been. The fix was to batch the product IDs and fetch them with `WHERE product_id = ANY(...)` grouped by shard, concurrently — exactly the pattern in section 5. The lesson: application-side joins are only fast if you batch and parallelize; a naive per-row loop reinvents the N+1 problem and is often slower than what you replaced.

### 7. The reference table that wasn't read-mostly

A team marked their `feature_flags` table as a Citus reference table (replicated to all 40 workers) because lots of queries joined to it. That was correct — until a new "per-user experiment assignment" feature started writing to it thousands of times a second. Every write now had to propagate to all 40 copies, replication lag ballooned, and the writes serialized into a bottleneck that slowed the whole cluster. The fix was to split the table: keep the genuinely static flags as a reference table, and move the high-write experiment assignments into a normal sharded table co-located with users. The lesson: reference-data replication is for small *and* read-mostly tables; the moment a "reference" table takes heavy writes, the N-way replication cost turns it into a cluster-wide bottleneck.

### 8. Citus and the unaligned distribution column

A team using Citus distributed `orders` by `order_id` and `line_items` by `line_item_id`, then joined them constantly. Citus dutifully executed these as *repartition joins* — shuffling data across workers at query time to align the join keys — and every join query moved gigabytes across the worker network and took seconds. The fix was a one-line change in intent and a re-distribution in practice: `create_distributed_table('line_items', 'order_id', colocate_with => 'orders')`, after which the joins pushed down to single workers and ran in milliseconds. The lesson: co-location is not automatic; if you distribute two frequently-joined tables on different columns, the system *will* fall back to shuffling, and the only fix is to align the distribution column.

## When to reach for these techniques (and when not to)

**Reach for co-location first, always:**

- Two tables are joined on a hot path and share a natural key (user, tenant, account). Shard both by that key.
- A transaction must atomically touch multiple tables. Co-locate them so it stays single-shard.
- You're choosing a shard key. Choose it so the dominant join and the transactional unit both stay local; let everything else pay the scatter tax.

**Reach for denormalization when** a join exists only to fetch a few immutable, point-in-time fields (a name on a receipt, a SKU on a line item) — copy them and delete the join. Build the backfill path before you need it.

**Reach for an application-side join when** you can't redesign the schema and the join is a simple lookup — but only if you batch keys per shard and fan out concurrently. A per-row loop is a regression.

**Reach for a global index when** you frequently look up rows by a column that isn't the shard key, and a scatter on that path is too expensive. Accept the second write and the eventual consistency.

**Reach for reference replication when** the dimension table is small *and* read-mostly. Re-check that "read-mostly" assumption whenever a feature starts writing to it.

**Skip the fancy stuff — and reconsider sharding itself — when:**

- You haven't actually outgrown one machine. A read replica, a bigger box, or caching solves most "the database is slow" problems without any of this tax. See [When One Database Is Not Enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) and the [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree).
- Your workload is genuinely analytical (big scans, many-way joins, ad-hoc aggregations). That belongs in a columnar/analytical store fed by CDC, not in a sharded OLTP cluster you keep trying to scatter-gather over.
- You're reaching for 2PC on a hot path. That's a signal the shard key split your transactional unit — fix the key instead.
- The cross-shard join is rare and ad-hoc (an internal admin tool). A bounded, limited, timeout-guarded scatter is fine; don't build a global index for a query that runs twice a day.

The thread through all of it: sharding doesn't make queries slow — *crossing shards* makes queries slow. Co-locate what's queried and transacted together, denormalize the point-in-time facts, replicate the small read-mostly dimensions, and the overwhelming majority of your traffic stays targeted and scales out cleanly. The scatter-gather tax is real, but it's a tax you mostly opt into through schema design — and mostly avoid through the same.

## Further reading

- [Database Partitioning and Sharding](/blog/software-development/database/database-partitioning-and-sharding) — how to split the data in the first place.
- [Choosing a Shard Key](/blog/software-development/database-scaling/choosing-a-shard-key) — the single decision that determines how much of this tax you pay.
- [Two-Phase Commit and How It Fails](/blog/software-development/database/two-phase-commit-and-how-it-fails) — the blocking failure mode in detail.
- [The Saga Pattern for Distributed Transactions](/blog/software-development/database/saga-pattern-distributed-transactions) — the high-throughput alternative to 2PC.
- [Slack and YouTube on Vitess](/blog/software-development/database-scaling/slack-and-youtube-on-vitess) — the query-router pattern at production scale.
- Dean & Barroso, *The Tail at Scale* (CACM, 2013) — the canonical analysis of fan-out tail latency and hedged requests.

---
title: "Reading EXPLAIN ANALYZE Like a Staff Engineer"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles, war-story-driven guide to reading Postgres and MySQL query plans — the cost model, estimated-vs-actual rows, BUFFERS, every node type, cardinality estimation, and the exact workflow to find and fix the bottleneck node."
tags:
  [
    "postgres",
    "mysql",
    "query-optimization",
    "explain-analyze",
    "query-planner",
    "indexing",
    "database",
    "performance",
    "sql",
    "cardinality-estimation",
    "buffers",
    "extended-statistics",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-1.webp"
---

There is a specific kind of silence that falls over a war room when the database is on fire. The API is timing out, the dashboards are red, and someone has already restarted the app servers twice with no effect. Eventually a senior engineer pulls up a terminal, runs one query against the primary, and stares at a wall of indented text for about forty seconds. Then they say something like: "The planner thinks this filter returns ten rows. It returns four hundred thousand. It's doing a nested loop. Add this index and run `ANALYZE`." Ten minutes later the graphs are green.

That wall of indented text is an `EXPLAIN ANALYZE` plan, and the difference between the engineer who reads it in forty seconds and the one who stares at it helplessly is not raw intelligence. It is that the first person knows what each line means, knows which number to look at first, and knows the three or four root causes that explain ninety percent of bad plans. This post is an attempt to transfer that skill in full — not "here are the EXPLAIN flags," which you can get from the docs, but the actual mental model a staff engineer carries: how the plan tree is shaped, where the cost numbers come from, why estimated-versus-actual rows is the single most important signal on the screen, and the exact sequence of moves to go from "this query is slow" to "here is the one-line fix."

![Anatomy of a Postgres plan tree: leaf scans feed parents, costs accumulate upward, and the root is the last node to finish](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-1.webp)

The diagram above is the mental model for the entire article. A query plan is a *tree* of operators. The leaves are scans — they pull rows out of tables and indexes. Each leaf feeds its parent, which might be a join, a sort, an aggregate, or a limit. The parent feeds its parent, all the way up to the root, which is the operator that produces the final result. The single most counterintuitive fact about reading a plan is this: **the root is printed at the top, but it executes last.** Execution flows from the leaves up. So you read a plan tree from the *inside out* — find the deepest, most indented nodes, understand what they produce, and follow the rows up the tree. Every cost, every row estimate, every timing number on a parent node is the *accumulation* of everything beneath it. Internalize that one shape and the rest of this article is a tour through the kinds of nodes that can appear in it and how to tell which one is killing you.

> A query plan is not a description of what the database *did*. It is a hypothesis the planner made about what would be cheapest, plus — if you used `ANALYZE` — a measurement of what actually happened. The gap between the hypothesis and the measurement is where every interesting bug lives.

## Why reading a plan is different from what you think

Before we go node by node, let's name the mismatch directly, because almost every wrong intuition about query plans comes from treating the optimizer as either omniscient or malicious. It is neither. It is a cost-minimizing search over a space of equivalent plans, fed by statistics that are frequently stale or wrong, and it makes locally rational decisions that occasionally compound into globally terrible ones.

| You assume | The naive mental model | The reality |
| --- | --- | --- |
| The plan tells you what the DB did | A log of executed steps | Without `ANALYZE` it's only an estimate; the actual plan can differ enormously from the estimated one |
| The optimizer knows your data | It "looks at the table" | It looks at a *sample* taken by the last `ANALYZE`; between runs it is reasoning about a snapshot that may be hours or weeks old |
| Cost is measured in milliseconds | `cost=142000` means 142 seconds | Cost is in abstract units anchored to "one sequential page read = 1.0"; it is *ordinal*, only meaningful for comparing plans of the same query |
| A Seq Scan is always bad | Index = fast, scan = slow | For >20–30% of a table a Seq Scan is *correctly* chosen; forcing an index there is slower |
| The slow node is the one with the biggest cost | Sort by cost, fix the top | The slow node is the one with the biggest `actual time × loops`; cost is the planner's *guess*, not the truth |
| Adding an index always helps | More indexes, more speed | An index the planner won't use is pure write-amplification overhead; the fix is often statistics, not indexes |
| `EXPLAIN ANALYZE` is safe to run | It just shows the plan | It *runs the query* — including the `INSERT`/`UPDATE`/`DELETE`. Wrap those in a transaction you `ROLLBACK` |

The most important row in that table is the third one, about cost units, and it trips up nearly everyone. When you see `cost=0.42..142000.55` in a Postgres plan, the `142000` is **not** a time, not a row count, not a byte count. It is the planner's internal currency, defined so that reading one 8 KB page sequentially from disk costs exactly `1.0`. Everything else is priced relative to that. The whole point of the number is *comparison*: the planner generates several candidate plans for your query, computes this abstract cost for each, and picks the smallest. The absolute value is meaningless across different queries. A cost of `142000` is not "bad" in isolation — it is bad only if another plan for the *same* query scores lower.

This is also why the second most common mistake is sorting nodes by cost to find the bottleneck. Cost is the planner's *prediction*. With `ANALYZE`, you get the *measurement* — `actual time`, `rows`, and `loops` — and those are what you trust. A node can have a modest estimated cost and still be the thing eating ninety percent of your wall-clock time, precisely because the planner mis-estimated it. That gap is the whole game, which is why it gets its own section next.

> The first rule of reading plans: the cost numbers are the planner's opinion; the `actual` numbers are the facts. When they disagree, the disagreement *is* the bug.

## 1. The cost model: startup cost, total cost, and the units

**Senior rule of thumb: every cost in a Postgres plan is denominated in "sequential page reads," and the planner's entire job is to add up those units across the tree and pick the cheapest total.**

Let's make the currency concrete. Postgres ships with a handful of planner cost constants, and their default values are worth memorizing because they explain *why* the planner makes the choices it does:

```sql
-- The planner cost constants (Postgres defaults), all relative to seq_page_cost = 1.0
SHOW seq_page_cost;        -- 1.0    : read one 8 KB page sequentially
SHOW random_page_cost;     -- 4.0    : read one 8 KB page at a random offset
SHOW cpu_tuple_cost;       -- 0.01   : process one row
SHOW cpu_index_tuple_cost; -- 0.005  : process one index entry
SHOW cpu_operator_cost;    -- 0.0025 : evaluate one operator/function call
SHOW parallel_setup_cost;  -- 1000.0 : spin up parallel workers
SHOW parallel_tuple_cost;  -- 0.1    : pass one row from a worker to the leader
```

The single most consequential ratio here is `random_page_cost / seq_page_cost = 4.0`. The planner believes a random page read is four times more expensive than a sequential one. On a 2005-era spinning disk with a 10 ms seek, that ratio was, if anything, conservative — random reads were closer to 50–100× sequential. On modern NVMe SSDs, where random and sequential reads are nearly equal, `4.0` is *too pessimistic*, which is why a near-universal piece of production tuning is to lower `random_page_cost` to `1.1` on SSD-backed instances. That single GUC change makes the planner more willing to choose index scans (which are random-access) over sequential scans, and it is responsible for a startling number of "the planner suddenly started using my index" stories. We'll return to it.

Every node in the plan reports its cost as a pair: `cost=<startup>..<total>`. These two numbers answer two different questions, and conflating them is a classic junior mistake.

- **Startup cost** is the work done before the node can emit its *first* row. For a `Seq Scan` it's ~0 — the first row is available immediately. For a `Sort`, it's *high* — a sort must consume its entire input before it can emit a single row, so the startup cost includes the cost of reading and sorting everything. For a `Hash Join`, the startup cost includes building the hash table from the inner relation.
- **Total cost** is the work to emit *all* rows.

Why does the distinction matter? Because of `LIMIT`. Consider these two plans for `SELECT * FROM events ORDER BY created_at LIMIT 10`:

```Plan A:  Limit  (cost=0.43..1.21 rows=10)
           ->  Index Scan using events_created_idx  (cost=0.43..81000.00 rows=1000000)

Plan B:  Limit  (cost=54000.00..54000.03 rows=10)
           ->  Sort  (cost=54000.00..56500.00 rows=1000000)
                 Sort Key: created_at
                 ->  Seq Scan on events  (cost=0.00..18000.00 rows=1000000)
```

Plan A's index scan has a *huge* total cost (81000) but a *tiny* startup cost (0.43), because the index is already in sorted order — it can hand the `LIMIT` its first ten rows almost instantly and stop. Plan B has to read the whole table and sort it (startup cost 54000) before the `LIMIT` sees anything. The planner correctly picks Plan A because, under a `LIMIT`, only the *startup-cost-adjusted* fraction of the total matters. This is exactly why an index on your `ORDER BY` column makes paginated, "show me the latest 10" queries fast: it converts a sort-then-limit into a stop-early index walk. It is also why the same query *without* the index, or with a `LIMIT` large enough to make the sort cheaper than the random-access walk, flips to Plan B. The planner is reasoning about startup cost the whole time; once you see it, the `LIMIT` behavior stops being mysterious.

A subtle and genuinely important corollary: the planner *optimizes for the LIMIT*. If it estimates the matching rows are densely packed near the start of the index, it assumes the early-stop will happen quickly. When that estimate is wrong — say the rows you want are all at the *end* of the index — the index scan that looked cheap walks nearly the entire index before finding ten matches, and you get a query that the planner priced at 1.21 but that actually takes seconds. This is one of the most common "good plan, terrible runtime" traps, and it is invisible unless you run `ANALYZE` and look at the actual numbers.

### Second-order: cost is ordinal, so don't tune to it

The cost number is a *ranking device*, not a stopwatch. Two consequences follow that experienced engineers internalize. First, never set an alerting threshold on plan cost — it has no fixed relationship to time, and the same cost can be 1 ms or 10 s depending on cache state and data distribution. Second, when you compare two plans, compare their costs *only if they are plans for the same query on the same statistics*. A frequent error is running `EXPLAIN` on staging (small tables, different stats) and concluding the production plan will be the same. It often isn't. As the GitLab database team puts it in their internal [understanding explain plans](https://github.com/diffblue/gitlab/blob/master/doc/development/database/understanding_explain_plans.md) guide, the bare `EXPLAIN` "produces an *estimated* execution plan based on the available statistics. This means the actual plan can differ quite a bit" — which is why their review process runs `EXPLAIN (ANALYZE, BUFFERS)` against a production-sized clone, not a toy schema.

## 2. Estimated vs actual rows: the #1 signal on the screen

**Senior rule of thumb: the first thing you look at in any `ANALYZE` plan is not the time and not the cost — it is the ratio of estimated rows to actual rows at every node. A divergence of more than ~10× is where you start digging.**

This is the heartbeat of plan reading. When you run `EXPLAIN (ANALYZE)`, every node prints two row counts: the planner's estimate, and the truth.

```Hash Join  (cost=8.44..142000.55 rows=41 width=64)
           (actual time=0.512..1840.221 rows=312840 loops=1)
```

Read that carefully. The planner estimated this join would produce **41** rows. It actually produced **312,840**. That is a four-orders-of-magnitude underestimate, and it is a flashing red light, because *every decision the planner made above this node was based on the belief that only 41 rows would flow upward.* It chose the join algorithm, the join order, whether to sort, whether to materialize — all on the assumption of 41 rows. When 312,840 show up instead, those choices are catastrophically wrong, and the error compounds as it propagates up the tree.

![Estimated vs actual rows: a 1000x leaf underestimate tricks the planner into a nested loop that runs millions of times](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-2.webp)

The before/after above shows the mechanism that makes this the most dangerous failure mode in all of query planning. On the left, a leaf filter is estimated at 10 rows but actually returns a million. The planner, believing the input is tiny, picks a **Nested Loop** join — which is the *correct* choice for tiny inputs, because a nested loop has near-zero startup cost and is unbeatable when you only loop a handful of times. But with a million rows on the outer side, that nested loop now executes its inner side a million times. A query the planner priced at 120 ms runs for 90 seconds. On the right, with an accurate estimate, the planner picks a **Hash Join** — one pass to build a hash table, one pass to probe it — and the query is stable at 1.4 seconds regardless of how the rows are distributed.

This is not a hypothetical. The Postgres community has documented this exact pathology for two decades; there is a [2009 mailing-list thread titled "Bad plan for nested loop + limit"](https://www.postgresql.org/message-id/603c8f070902271454h5a78a063kcf2b78a10fd956b1@mail.gmail.com) that reads exactly like an incident you'll have next quarter. As Cybertec's write-up on [detecting wrong planner estimates](https://www.cybertec-postgresql.com/en/detecting-wrong-planner-estimates/) puts it: if an estimate is very low, "it can easily happen that a nested loop on top of this is heavily underestimated… if you end up with a nested loop on a million rows if there should be just 500 rows around, it can easily turn out to be a disaster." The asymmetry is brutal: *underestimates* are far more dangerous than overestimates, because the nested loop they trigger has unbounded downside, whereas an overestimate merely makes the planner overly cautious.

So the diagnostic discipline is mechanical. Walk the tree from the leaves up. At each node, divide actual rows by estimated rows. The first node where that ratio blows past ~10× — *that* is your root cause, or very close to it. Everything above it is collateral damage. You don't fix the Hash Join; you fix the leaf filter whose estimate was wrong, and the join fixes itself.

```sql
-- The query whose plan we keep referencing. Find orders over $500 for active
-- enterprise users, newest first, top 20. (Schema is illustrative.)
EXPLAIN (ANALYZE, BUFFERS)
SELECT o.id, o.total, u.email
FROM orders o
JOIN users u ON u.id = o.user_id
WHERE u.plan = 'enterprise'
  AND u.status = 'active'
  AND o.total > 500
ORDER BY o.total DESC
LIMIT 20;
```

If `u.plan = 'enterprise' AND u.status = 'active'` is estimated at 1 row but matches 50,000, the join that consumes those users will be mis-planned, and the fix lives in the `users` scan — stale statistics, or a correlation between `plan` and `status` the planner can't see (more on both later). The plan is *pointing* at the problem; you just have to read the finger.

### A worked numerical example of how the error compounds

Suppose `users` has 1,000,000 rows. The planner has per-column statistics saying `plan = 'enterprise'` matches 2% (20,000 rows) and `status = 'active'` matches 60% (600,000 rows). Lacking any correlation information, it assumes independence and multiplies the selectivities: `0.02 × 0.60 = 0.012`, estimating `12,000` rows. Suppose the truth is that *enterprise users are almost all active*, so the real count is `19,800`. That's only a 1.65× error — survivable.

Now flip it. Suppose the filter is `plan = 'enterprise' AND status = 'churned'`, where churned is 1% overall (`0.02 × 0.01 = 0.0002`, estimate `200` rows), but enterprise customers essentially never churn, so the real count is `3`. The planner over-estimated by 67×, picks a hash join where a nested loop would've been instant, and wastes memory — annoying but not fatal. The truly dangerous direction is when correlated columns make the *real* count far *exceed* the independent-multiplication estimate, because then the nested-loop trap springs. The lesson staff engineers carry: **the planner's independence assumption is the source of most large misestimates, and the sign of the error tells you which way the pain comes.**

## 3. loops × per-loop rows: the multiplier hiding in plain sight

**Senior rule of thumb: a node's reported `rows` is per-loop, not total. Always multiply by `loops` before you believe it.**

This is the single most misread field in the entire plan format, and it has burned every engineer at least once. Consider:

```->  Index Scan using orders_user_idx on orders o
      (cost=0.43..3.85 rows=4 width=24)
      (actual time=0.011..0.018 rows=4 loops=50000)
```

A junior reading this sees `rows=4`, `actual time=0.018 ms` and thinks "trivial, 4 rows, 18 microseconds, who cares." Wrong. The `loops=50000` means this index scan was executed **fifty thousand times** — once for each row from the outer side of a nested loop above it. The *total* rows produced is `4 × 50000 = 200,000`, and the *total* time is `0.018 ms × 50000 = 900 ms`. That innocuous-looking node is nearly a full second of work. The per-loop numbers are deceptively small; the multiplier is where the cost hides.

This is the exact texture of the nested-loop blowup from the previous section, now visible in the numbers. When you see a large `loops` count on the inner side of a `Nested Loop`, you've found the engine of the slowness. The fix is almost always to make the planner *not* choose a nested loop here — which means fixing the outer side's row estimate so the planner realizes it's about to loop 50,000 times, not 5.

There's a related subtlety worth its own note: timings in Postgres plans are **cumulative up the tree but per-loop within a node**. A parent node's `actual time` includes all of its children's time. So to find the time spent *in* a node, you subtract its children's total time from its own. Tools like [explain.dalibo.com](https://explain.dalibo.com/) (the open-source PEV2 visualizer) do this subtraction for you and color the node that has the most *exclusive* time — which is exactly the node you want to fix. Doing it by hand on a deep tree is error-prone, which is why the visualizers exist.

MySQL handles loops the same way conceptually but prints it differently. In MySQL 8's `EXPLAIN ANALYZE` (which always uses the `TREE` format — `FORMAT=JSON` and `FORMAT=TRADITIONAL` are *rejected* with `EXPLAIN ANALYZE`, per the [MySQL 8.0 reference](https://dev.mysql.com/doc/refman/8.0/en/explain.html)), you'll see `(actual time=0.012..0.034 rows=4 loops=50000)` in nearly the same shape. The cross-engine instinct is identical: never trust a per-loop row count until you've multiplied by loops.

## 4. The scan nodes: Seq, Index, Index Only, and Bitmap

**Senior rule of thumb: the planner's access-path choice is driven by *selectivity* — what fraction of the table a predicate keeps — not by table size. A Seq Scan on a 10-billion-row table can be the right call; an Index Scan that returns 40% of the rows is the wrong one.**

This is where most "why isn't it using my index?!" confusion lives, so let's lay out the four scan types and exactly when the planner reaches for each.

![Access-path decision matrix: selectivity, not table size, drives the scan choice and the crossover sits near a few percent of rows](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-3.webp)

The matrix above is the access-path decision in one picture. Read it as "fraction of rows kept" (rows) against "query shape" (columns). The green region — index scans — dominates the top, where you're keeping a tiny slice. The amber region — sequential scans — dominates the bottom, where you're keeping most of the table. The interesting transition is the blue band in the middle: **bitmap scans**, which exist precisely to bridge the gap between "few enough rows to want an index" and "too many rows for a plain index scan to be efficient."

### Seq Scan

A sequential scan reads every page of the table in physical order. It is the planner's baseline — the plan it falls back to when no index helps, and the plan it *correctly prefers* when a predicate matches a large fraction of rows. The intuition: if you're going to touch 40% of the table anyway, reading it sequentially (cheap, `seq_page_cost = 1.0` per page) beats doing 40% of the table's worth of *random* index lookups (`random_page_cost = 4.0` each) plus the index traversal overhead. The crossover is workload-dependent but typically lands somewhere between 5% and 20% of the table.

```Seq Scan on orders  (cost=0.00..98000.00 rows=5746969 width=64)
                    (actual time=0.008..612.443 rows=5746940 loops=1)
  Filter: (total > 500)
  Rows Removed by Filter: 1203060
```

Two things to read here. First, the estimate (5,746,969) matches actual (5,746,940) almost exactly — this scan's estimate is *healthy*, no statistics problem. Second, `Rows Removed by Filter: 1203060` tells you the scan read ~6.9M rows and threw away ~1.2M after applying `total > 500`. That removed-by-filter count is a key efficiency signal: if it's huge relative to the rows kept, you're reading a lot of pages just to discard them, and a partial or expression index on the filter column could let the database skip them entirely. The GitLab guide flags exactly this — in one of their examples a filter discarded **2,487,813 rows** until they built a partial index whose `WHERE` clause matched the predicate, after which those rows never had to be visited.

### Index Scan

An index scan walks the B-tree (see [how database indexes really work](/blog/software-development/database/b-trees-how-database-indexes-work)) to find matching entries, then fetches the corresponding rows from the heap. It's two-phase per match: traverse index → fetch heap page. Because each heap fetch is a *random* read (`random_page_cost = 4.0`), index scans are only cheap when you fetch few rows. The planner's selectivity estimate decides whether the random-fetch tax is worth paying.

```Index Scan using users_pkey on users u  (cost=0.43..8.45 rows=1 width=48)
                                        (actual time=0.014..0.015 rows=1 loops=1)
  Index Cond: (id = 91823)
```

Note `Index Cond` versus `Filter`. An `Index Cond` is a predicate the index itself can satisfy — it narrows the index traversal. A `Filter` is applied *after* fetching rows, on columns the index doesn't cover. If you see a predicate land in `Filter` when you expected it in `Index Cond`, your index doesn't cover that column, and you're fetching-then-discarding. Moving a `Filter` predicate into `Index Cond` by extending or reordering the index is one of the highest-leverage tuning moves there is.

### Index Only Scan and the Heap Fetches trap

An **index only scan** is the dream: the index contains *all* the columns the query needs, so Postgres never touches the heap at all. No random heap fetches, dramatically fewer buffers. This is what a *covering index* buys you.

```Index Only Scan using orders_user_total_idx on orders
      (cost=0.43..142.00 rows=80 width=12)
      (actual time=0.021..0.402 rows=78 loops=1)
  Index Cond: (user_id = 91823)
  Heap Fetches: 0
```

The line to obsess over is **`Heap Fetches: 0`**. An index only scan is only as good as its heap-fetch count. Here's the trap: Postgres can't serve a row from the index alone unless it *knows the row is visible* to your transaction, and visibility lives in the heap. Postgres uses the **visibility map** — a bitmap marking pages where all rows are visible to everyone — to skip the heap when possible. But on a table with heavy writes, recently modified pages aren't marked all-visible yet, so the "index only" scan has to fall back to the heap for those rows. You'll see `Heap Fetches: 41280` and wonder why your covering index isn't fast. The answer is almost always: the table needs a `VACUUM` to refresh the visibility map. This is one of the most common "I built the perfect index and it's still slow" mysteries, and the fix is `VACUUM`, not more indexing.

### Bitmap Index Scan → Bitmap Heap Scan

When a predicate matches *too many* rows for a plain index scan (all those random heap fetches add up) but *too few* for a sequential scan to be worth it, the planner reaches for a two-phase bitmap scan. This is the cleverest of the scan strategies and the one most worth understanding deeply.

![Bitmap Index Scan into Bitmap Heap Scan: the bitmap turns scattered random index hits into one ordered, sequential heap sweep](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-4.webp)

The pipeline above shows the trick. Phase one — the **Bitmap Index Scan** — walks the index and, instead of immediately fetching each matching row, records the *tuple IDs* (which page, which offset) into an in-memory bitmap. If your query has multiple indexable conditions, Postgres can build a bitmap from *each* index and combine them with `BitmapAnd` / `BitmapOr` before touching the heap at all. Phase two — the **Bitmap Heap Scan** — sorts those tuple IDs by physical page and reads the heap *once, in page order*. This converts what would have been thousands of scattered random reads into a single forward, mostly-sequential sweep. It is the planner deliberately trading a little CPU (building and sorting the bitmap) for far less random I/O.

```Bitmap Heap Scan on orders  (cost=512.00..40210.00 rows=51200 width=64)
                            (actual time=8.21..142.55 rows=50831 loops=1)
  Recheck Cond: ((status = 'paid') AND (created_at > now() - interval '7 days'))
  Heap Blocks: exact=9214
  ->  BitmapAnd  (cost=512.00..512.00 rows=51200 width=0)
        ->  Bitmap Index Scan on orders_status_idx  (cost=0.00..240.00 rows=300000)
        ->  Bitmap Index Scan on orders_created_idx  (cost=0.00..260.00 rows=160000)
```

Three things to read. First, `BitmapAnd` of two indexes — Postgres combined `orders_status_idx` and `orders_created_idx` to intersect their matches, which is why neither index alone needs to be a composite covering both columns. Second, `Heap Blocks: exact=9214` — the bitmap fit in `work_mem`, so it tracks exact tuple positions. If `work_mem` were too small, you'd see `Heap Blocks: lossy=N`, meaning the bitmap degraded to tracking whole *pages* rather than exact rows, forcing a **`Recheck Cond`** to re-test every row on those pages. Lossy bitmaps are a quiet `work_mem` smell. Third, the `Recheck Cond` line is always present for bitmap scans (the heap scan re-applies the condition), but it only does *real work* when the bitmap went lossy.

## 5. The join nodes: Nested Loop vs Hash Join vs Merge Join

**Senior rule of thumb: the join algorithm the planner picks is a direct function of the estimated input sizes and whether the inputs are already sorted. Get the size estimate wrong and you get the join algorithm wrong — that's the nested-loop blowup again, now named.**

There are exactly three join strategies in Postgres (and conceptually in MySQL), and each is optimal in a different regime.

![Join algorithm by relation size and ordering: nested loop wins on tiny inputs, hash on big unordered joins, merge when both sides are sorted](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-5.webp)

The matrix above is the join decision. Read down the rows by input size and ordering; the rightmost column is the planner's correct pick.

### Nested Loop

For each row of the outer relation, probe the inner relation. Cost is roughly `outer_rows × inner_lookup_cost`. With a tiny outer side and an indexed inner side, this is *unbeatable* — near-zero startup, no memory, no hashing. The planner loves it when it believes the outer side is small. The danger, as we've now seen three times, is when the outer side is *secretly large*: `outer_rows × inner_cost` explodes quadratically. A nested loop is the right answer for "look up the 3 orders of this 1 user" and the catastrophically wrong answer for "join every order to every user."

```Nested Loop  (cost=0.86..520.00 rows=4 width=72)
             (actual time=0.03..0.31 rows=4 loops=1)
  ->  Index Scan using users_pkey on users u  (rows=1 loops=1)
  ->  Index Scan using orders_user_idx on orders o  (rows=4 loops=1)
```

This one is healthy — `loops=1` on the inner side, 4 rows total. The instant you see `loops` in the thousands on a nested loop's inner node, you've found a blowup.

### Hash Join

Build a hash table from the (smaller) inner relation, then stream the outer relation through it, probing the hash for matches. Cost is roughly `O(outer + inner)` — linear, not quadratic. This is the workhorse for joining two large unordered relations. The price is memory: the hash table must fit in `work_mem`, or it spills to disk in *batches*, which you'll see as `Buckets: ... Batches: 8 Memory Usage: ...` — more than one batch means it spilled.

```Hash Join  (cost=8.44..142000.55 rows=312000 width=64)
           (actual time=0.51..1840.22 rows=312840 loops=1)
  Hash Cond: (o.user_id = u.id)
  ->  Seq Scan on orders o  (rows=5746940 loops=1)
  ->  Hash  (rows=312000 width=48)  (actual rows=312840 loops=1)
        Buckets: 524288  Batches: 1  Memory Usage: 28672kB
        ->  Index Scan using users_plan_idx on users u  (rows=312840 loops=1)
```

`Batches: 1` is what you want — the whole hash table fit in `work_mem` (28 MB here). `Batches: 8` would mean the join partitioned both inputs into 8 chunks and processed them separately, reading the outer relation's spilled partitions back from disk. That's a `work_mem` pressure signal, and the fix is either a bigger `work_mem` (per-session, for this query) or a better estimate so the planner sizes the hash table correctly.

### Merge Join

If *both* inputs are already sorted on the join key — because they're being read in index order, or were just sorted upstream — Postgres can zip them together in a single linear pass, like merging two sorted lists. Cost is `O(outer + inner)` with tiny memory. The catch: it's only cheap if the sort comes for free (from an index). If the planner has to *add* a `Sort` to make a merge join possible, the sort's cost usually makes a hash join win instead. Merge joins shine on large, pre-sorted joins and on `FULL OUTER JOIN` (the only algorithm that handles it without contortions).

The comparison, distilled:

| Algorithm | Best when | Cost shape | Memory | Failure mode |
| --- | --- | --- | --- | --- |
| Nested Loop | inner side small & indexed; outer side small | `O(outer × inner)` | none | quadratic blowup when outer is secretly large |
| Hash Join | both sides large, unordered | `O(outer + inner)` | hash table in `work_mem` | spills to disk in batches if hash > `work_mem` |
| Merge Join | both sides already sorted on join key | `O(outer + inner)` | small (streaming) | a forced upstream `Sort` makes it lose to hash |

The unifying insight: **all three are correct in their regime, and the planner's only real failure is choosing the wrong regime because it mis-sized the inputs.** You almost never want to override the join algorithm directly. You want to fix the row estimate that made the planner think it was in a different regime. When you genuinely can't fix the estimate, Postgres gives you blunt instruments — `SET enable_nestloop = off` will forbid nested loops for the session — but reach for those only as a diagnostic ("does forcing a hash join fix it? then the estimate is the real problem") rather than a permanent fix.

## 6. Sort, Aggregate, and the work_mem cliff

**Senior rule of thumb: every `Sort`, `Hash`, and `HashAggregate` node has a memory budget called `work_mem`, and the difference between fitting in it and spilling past it is the difference between a fast node and a node that's an order of magnitude slower.**

### Sort: quicksort in memory vs external merge on disk

A `Sort` node either fits its input in `work_mem` and does a fast in-memory quicksort, or overflows and does an **external merge sort** that spills sorted runs to temporary files on disk and merges them in multiple passes.

![Sort node: in-memory quicksort vs external merge on disk, where exceeding work_mem makes the node an order of magnitude slower](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-9.webp)

The before/after above is one of the most actionable plan-reading skills, because Postgres tells you *exactly* which path it took. Look for the `Sort Method` line:

```-- Spilled to disk: slow
Sort  (cost=158000..160500 rows=1000000 width=64)
      (actual time=1620..1820 rows=1000000 loops=1)
  Sort Key: o.total DESC
  Sort Method: external merge  Disk: 38912kB

-- Fit in memory after raising work_mem: fast
Sort  (cost=140000..142500 rows=1000000 width=64)
      (actual time=82..95 rows=1000000 loops=1)
  Sort Key: o.total DESC
  Sort Method: quicksort  Memory: 41984kB
```

`Sort Method: external merge Disk: 38912kB` is the planner telling you it spilled 38 MB to temporary files. The fix is often a one-liner: `SET work_mem = '64MB';` for this session, which lets the sort run as `Sort Method: quicksort Memory: 41984kB` instead — in this example, dropping the node from 1820 ms to 95 ms. But `work_mem` is per-sort-node, per-connection, so setting it globally to 64 MB on a server with 500 connections each running 3 sorts can request `64MB × 500 × 3 = 96 GB` and OOM the box. The discipline: raise `work_mem` *locally* for the specific heavy query (`SET LOCAL work_mem` inside a transaction), not globally. There's also `Sort Method: top-N heapsort`, which Postgres uses for `ORDER BY ... LIMIT N` — it maintains only the top N rows in a small heap, so it's cheap even on huge inputs. Seeing `top-N heapsort` instead of `external merge` on a `LIMIT` query is a sign the planner is doing the smart thing.

### Aggregate vs HashAggregate vs GroupAggregate

For `GROUP BY`, Postgres has two strategies. **GroupAggregate** requires sorted input (it groups adjacent equal keys in one pass, like merge join) — cheap if the sort is free, otherwise it needs a `Sort` underneath. **HashAggregate** builds a hash table keyed by the group columns, no sorting required — fast, but the hash table lives in `work_mem`. Before Postgres 13, a `HashAggregate` that exceeded `work_mem` would *ignore the limit and use more memory anyway* (a notorious OOM risk); from 13 onward it spills to disk like everything else, which you'll see as `Disk Usage` on the node. If you see a `GroupAggregate` with an expensive `Sort` beneath it, and the group count is modest, nudging the planner toward `HashAggregate` (better stats, or `enable_sort`/`work_mem` tuning) can be a large win.

```HashAggregate  (cost=92000..92500 rows=5000 width=16)
               (actual time=520..524 rows=4980 loops=1)
  Group Key: o.user_id
  Batches: 1  Memory Usage: 4145kB
  ->  Seq Scan on orders o  (rows=5746940 loops=1)
```

`Batches: 1` again means it fit. The instinct is identical to hash joins: more than one batch means spill.

## 7. BUFFERS: the read path and the only stable signal

**Senior rule of thumb: always run `EXPLAIN (ANALYZE, BUFFERS)`. Timing is volatile across machines and cache states; buffer counts are stable. Optimize the buffers and the time follows.**

This is the move that separates people who read plans casually from people who do it professionally. The `BUFFERS` option adds, to every node, a count of how many 8 KB pages it touched and where they came from. As Nikolay Samokhvalov of Postgres.ai argues in his widely-cited piece, [EXPLAIN (ANALYZE) needs BUFFERS](https://postgres.ai/blog/20220106-explain-analyze-needs-buffers-to-improve-the-postgres-query-optimization-process), the principle is: **"Timing is volatile. Data volumes are stable."** The same query might run in 1 ms on a warm production cache and 1 second on a cold staging box — the *timing* is meaningless to compare. But the *buffer count* — `shared read=9214` — is identical on both, because it's a property of the data and the plan, not the hardware. Buffers let you optimize a query on a clone and trust the result on production.

![What BUFFERS counts on the read path: shared hit is free RAM; shared read crosses into the OS cache or disk and is what actually costs you](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-6.webp)

The layered stack above is what the buffer counters mean. When a node needs a page:

- **`shared hit`** — the page was already in Postgres's `shared_buffers` (its in-process page cache). Essentially free; pure RAM access.
- **`shared read`** — a cache *miss*: Postgres had to fetch the page from the OS, which may serve it from the OS page cache (fast-ish) or go to physical disk (slow). This is the number that actually costs you, and the one to drive down.
- **`shared dirtied`** / **`shared written`** — pages your query modified or had to flush; relevant for write-heavy plans and for understanding why a `SELECT` sometimes does I/O (it can flush dirty pages or set hint bits).
- **`temp read`** / **`temp written`** — temporary files written and read back for sorts and hashes that *spilled past `work_mem`*. Any `temp written` is a direct, quantified signal that a node overflowed — it's the buffer-level view of the `work_mem` cliff from the last section.

```Bitmap Heap Scan on orders  (actual time=8.21..142.55 rows=50831 loops=1)
  Buffers: shared hit=312 read=8902
  ->  Bitmap Index Scan on orders_status_idx
        Buffers: shared hit=4 read=86
```

Here the heap scan did `shared read=8902` — it read 8,902 pages (about 70 MB) from outside the cache. That's the cost. The math GitLab uses internally: each buffer is 8 KB, so you can convert directly — their guide cites a query using "202,622 buffers, which equals 1.58 GB of memory," and flags any single step needing "more than 512 MB" of buffers as worth investigating. Samokhvalov's canonical demo reduces a query from ~1003 buffers to 29 by clustering an index — a 100× drop in buffers that tracks a 100× drop in time. **Buffers are the stable currency; time is just buffers plus luck.**

A practical reading habit: scan the plan for the node with the largest `shared read`. That node is doing the most cold I/O, and it's usually the one to fix — either by making it touch fewer pages (a better/covering index) or by making more of its reads `shared hit` (clustering, or simply a warmer cache for a frequently-run query). The Shopify-scale advice that circulates among multi-tenant app teams is to "run `EXPLAIN (ANALYZE, BUFFERS)` on every query touching tables over 100k rows" — and to treat a `Seq Scan` with large `shared read` on a big table as a missing-index alarm, not a fact of life.

## 8. Gather, parallel workers, Memoize, and JIT

**Senior rule of thumb: the newer plan nodes — parallel workers, Memoize, JIT — are usually wins, but each has a regime where it backfires, and the plan tells you which one you're in.**

### Gather and parallel workers

When a scan or join is large enough, the planner can split the work across **parallel workers**. You'll see a `Gather` node at the top of the parallel region, with `Workers Planned` and `Workers Launched` underneath:

```Gather  (cost=1000..52000 rows=5000000 width=64)
        (actual time=2.1..880 rows=5000000 loops=1)
  Workers Planned: 4
  Workers Launched: 4
  ->  Parallel Seq Scan on orders o  (rows=1250000 loops=5)
```

Read two things. First, `Workers Launched` can be *less* than `Workers Planned` if the server is at its `max_parallel_workers` limit — under that contention, your "parallel" query silently runs with fewer workers and is slower than the plan predicted. Second, the child node's `rows` is *per worker* and `loops=5` reflects 4 workers + 1 leader; you multiply, as always, to get the total. Parallelism has a fixed startup cost (`parallel_setup_cost = 1000`), so the planner only uses it when the table is big enough to amortize spinning up workers. On small tables, forcing parallelism is pure overhead.

### Memoize

Introduced in Postgres 14, a **Memoize** node sits on the inner side of a nested loop and *caches* the inner result for each distinct outer key. If the outer side repeats keys — say you're joining orders to users and many orders share a user — Memoize turns repeated inner lookups into cache hits:

```Nested Loop  (actual time=0.05..120 rows=500000 loops=1)
  ->  Seq Scan on orders o  (rows=500000 loops=1)
  ->  Memoize  (actual time=0.0002..0.0003 rows=1 loops=500000)
        Cache Key: o.user_id
        Hits: 495000  Misses: 5000  Evictions: 0  Memory Usage: 800kB
        ->  Index Scan using users_pkey  (rows=1 loops=5000)
```

`Hits: 495000 Misses: 5000` — Memoize turned 500,000 potential index lookups into 5,000 real ones plus 495,000 cache hits. That's a nested loop made cheap by caching. But notice the failure mode hiding in the same fields: if the outer keys are all *distinct* (`Hits: 0`), Memoize is pure overhead, and if the cache thrashes (high `Evictions`), it's worse than useless. The plan hands you the hit/miss/eviction counts precisely so you can tell which case you're in.

### JIT

For analytical queries with high cost estimates, Postgres can JIT-compile expression evaluation with LLVM. When it helps, it shaves CPU off tight per-row loops over millions of rows. When it *hurts* — and this is a real production gotcha — the JIT *compilation* itself takes longer than it saves, which happens on queries the planner *over-estimated* into the JIT cost threshold but that actually return few rows.

```JIT:
  Functions: 14
  Options: Inlining true, Optimization true, Expressions true, Deforming true
  Timing: Generation 2.1ms, Inlining 8.4ms, Optimization 41.2ms, Emission 22.1ms, Total 73.8ms
```

If you see `JIT … Total 73.8ms` on a query whose actual execution is 40 ms, the JIT *cost more than the query*. This is a classic symptom of a row over-estimate pushing the cost over `jit_above_cost` (default 100000). The fix is either to fix the estimate or, pragmatically, raise `jit_above_cost` / set `jit = off` for the offending workload. JIT-induced slowdowns are sneaky because the query "got slower for no reason" after an upgrade enabled JIT by default — always check the `JIT` block's timing against the total.

## 9. Where estimates come from, and why they go wrong

**Senior rule of thumb: the planner's row estimates come from a statistical sample taken by `ANALYZE`. Three things break that sample — staleness, correlated columns, and unsampleable predicates — and each has a specific fix.**

Everything in sections 2–8 ultimately traces back to one number: the row estimate. To fix bad estimates you have to know where they come from. After `ANALYZE` (run automatically by autovacuum, or manually), Postgres stores per-column statistics in `pg_statistic`, surfaced readably in `pg_stats`:

```sql
SELECT attname, n_distinct, most_common_vals, most_common_freqs, correlation
FROM pg_stats
WHERE tablename = 'orders' AND attname IN ('status', 'user_id', 'created_at');
```

The fields that drive estimation:

- **`n_distinct`** — the number of distinct values. The planner uses this to estimate equality selectivity: for a value not in the most-common list, it assumes `1 / n_distinct` of rows match. A wrong `n_distinct` (common on columns where `ANALYZE`'s sample under-counts distinct values) throws every equality estimate off.
- **`most_common_vals` / `most_common_freqs`** (the MCV list) — the most frequent values and their frequencies. For `WHERE status = 'paid'`, if `'paid'` is in the MCV list, the planner uses its exact recorded frequency; if not, it uses the residual. Skewed columns *need* a large MCV list to be estimated well.
- **`correlation`** — how well the column's physical order on disk matches its sorted order. Near 1.0 means an index scan reads the heap nearly sequentially (cheap); near 0 means random reads (expensive). This is why a freshly `CLUSTER`-ed table makes range scans dramatically faster — it drives correlation toward 1.0.
- **`null_frac`** — fraction of NULLs, used to estimate `IS NULL` / `IS NOT NULL`.

### Failure 1: stale statistics

If you bulk-load ten million rows and immediately query, `ANALYZE` hasn't run yet, so the planner thinks the table is still tiny (or empty) and picks plans for a tiny table — usually a nested loop that then blows up. The fix is literally `ANALYZE orders;`. Autovacuum runs `ANALYZE` automatically once enough rows change (governed by `autovacuum_analyze_scale_factor`, default 10%), but after a big bulk operation you should run it manually rather than wait. The GitLab guide's discipline applies: "you cannot optimize something you do not understand" — and the first thing to confirm when an estimate looks wrong is whether the stats are even current (`last_analyze` in `pg_stat_user_tables`).

### Failure 2: not enough buckets for a skewed column

The default statistics target is 100 — Postgres keeps up to 100 MCV entries and 100 histogram buckets per column. On a column with thousands of meaningfully-frequent values, 100 isn't enough resolution, and rare-ish values get lumped into a coarse residual estimate. The fix is to raise the target on that column and re-analyze:

```sql
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 500;
ANALYZE orders;
```

Higher targets make `ANALYZE` slower and plans slightly slower to compute, so raise it surgically on the columns that need it, not globally.

### Failure 3: correlated columns and CREATE STATISTICS

This is the deep one, and it's the source of the worst nested-loop blowups. By default, Postgres assumes columns are **statistically independent** and multiplies their selectivities. When columns are actually *correlated*, that multiplication produces a wildly wrong estimate — almost always an *under*-estimate, which is the dangerous direction.

![Extended statistics fix a correlated underestimate: CREATE STATISTICS teaches the planner that city implies country instead of multiplying independent selectivities](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-8.webp)

The before/after above is the canonical example. A query filters `WHERE city = 'Paris' AND country = 'France'`. Per-column stats say `city = 'Paris'` matches 2% and `country = 'France'` matches 5%. Assuming independence, the planner multiplies: `0.02 × 0.05 = 0.001`, estimating 50 rows out of 50,000. But city *functionally determines* country — every Paris row is a France row — so the real selectivity is just 2%: 1,000 rows. The planner is off by 20×, and downstream it picks a nested loop sized for 50 rows that then loops 1,000 times. The fix is **extended statistics**, which explicitly teach the planner about the relationship:

```sql
-- Tell the planner that (city, country) move together.
CREATE STATISTICS orders_city_country (dependencies, ndistinct)
  ON city, country FROM orders;
ANALYZE orders;
```

The `dependencies` kind captures *functional dependencies* (city → country); the `ndistinct` kind captures the *combined* distinct count of the column group (vital for `GROUP BY` over multiple correlated columns, where independence multiplication wildly over-counts groups). There's a third kind, `mcv`, which stores multivariate most-common-value lists for the column combination — the most powerful and most expensive. After creating the statistics and re-analyzing, the planner estimates the combination correctly, and the nested-loop trap disappears. This single feature, underused because most people don't know it exists, fixes a huge fraction of "the planner is being stupid" cases. As the Postgres docs and Crunchy Data's [hacking the statistics tables](https://www.crunchydata.com/blog/hacking-the-postgres-statistics-tables-for-faster-queries) walkthrough both stress, correlated columns are the number-one cause of large multi-predicate misestimates, and `CREATE STATISTICS` is the supported, surgical fix.

### Failure 4: predicates the planner can't sample

Some predicates are inherently hard to estimate: `WHERE data ->> 'type' = 'click'` on a JSONB column, `WHERE lower(email) = ...`, `WHERE created_at > now() - interval '7 days'` (the planner can't know what `now()` will be at plan time for a prepared statement). For expression predicates, the fix is often an **expression index** *plus* the statistics that come with it — Postgres gathers stats on the indexed expression. For the JSONB case specifically, Heap's [working around a case where the Postgres planner is "not very smart"](https://www.heap.io/blog/when-the-postgres-planner-is-not-very-smart) is the definitive war story; we'll cover it in full in the case studies.

## 10. MySQL EXPLAIN ANALYZE: the same instincts, a different dialect

**Senior rule of thumb: MySQL's plan format looks different, but the reading discipline is identical — find the node where estimated and actual rows diverge, and watch for the loop multiplier.**

If you work across both engines, you need to translate. MySQL has had a usable `EXPLAIN ANALYZE` since 8.0.18, and it's genuinely good — it prints a tree with both estimated and actual numbers, the same as Postgres. The key facts to keep straight:

| Concept | Postgres | MySQL 8.0+ |
| --- | --- | --- |
| Plan with estimates only | `EXPLAIN` (text/json) | `EXPLAIN`, `EXPLAIN FORMAT=JSON`, `EXPLAIN FORMAT=TREE` |
| Plan with actual measurements | `EXPLAIN (ANALYZE)` | `EXPLAIN ANALYZE` (always `TREE` format) |
| Per-node cost | `cost=startup..total` (page-read units) | `cost=N` (different unit, same ordinal idea) |
| Estimated vs actual rows | `rows=` (est) and `actual … rows=` | `rows=` (est) and `actual … rows=` |
| Loop multiplier | `loops=N` | `loops=N` |
| Buffer accounting | `BUFFERS` (shared hit/read) | no direct equivalent; use `FORMAT=JSON` cost + `performance_schema` |

A critical gotcha from the [MySQL 8.0 reference](https://dev.mysql.com/doc/refman/8.0/en/explain.html): `EXPLAIN ANALYZE` **only** supports `TREE` format. If you write `EXPLAIN ANALYZE FORMAT=JSON ...` you get an error. So in MySQL you use `EXPLAIN FORMAT=JSON` for the *rich estimated* plan (it includes `query_cost`, `filtered` percentages, and access-type details) and `EXPLAIN ANALYZE` for the *measured tree*. A representative MySQL `EXPLAIN ANALYZE` line:

```-> Nested loop inner join  (cost=4521 rows=1250)
   (actual time=0.042..38.114 rows=49210 loops=1)
    -> Index lookup on o using user_idx (user_id=u.id)
       (cost=2.5 rows=4) (actual time=0.004..0.012 rows=18 loops=2734)
```

Read it exactly as you'd read Postgres: the join estimated 1,250 rows, produced 49,210 — a 39× underestimate, the same red flag. The inner `Index lookup` has `loops=2734`, so its real cost is `~18 rows × 2734 loops`. The `EXPLAIN FORMAT=JSON` companion gives you the `filtered` field — the percentage of rows the optimizer thinks a condition will keep — which is MySQL's selectivity estimate and the analog of Postgres's `rows`. A `filtered: 1.00` (1%) that's wildly off from reality is your stale-statistics signal; you fix it with `ANALYZE TABLE orders;` (MySQL 8.0 also supports histogram statistics via `ANALYZE TABLE ... UPDATE HISTOGRAM ON col;`, its answer to skewed-column estimation). The instinct transfers cleanly: divergence first, loops second, access type third.

## 11. The debugging workflow, end to end

**Senior rule of thumb: don't read a plan top to bottom hoping the problem jumps out. Run a fixed sequence: rank by aggregate time, capture the full plan, find the one bottleneck node, classify the root cause, fix that one thing, and verify with buffers — not milliseconds.**

This is the workflow that turns plan-reading from an art into a procedure.

![A staff engineer's EXPLAIN debugging workflow: from pg_stat_statements down to one bottleneck node, branching on estimate vs access-path root cause](/imgs/blogs/reading-explain-analyze-like-a-staff-engineer-7.webp)

The graph above is the procedure. Walk it left to right.

**Step 1 — Find the query worth fixing (`pg_stat_statements`).** You don't optimize the query someone complained about; you optimize the query consuming the most total time. The [`pg_stat_statements`](https://www.postgresql.org/docs/current/auto-explain.html) extension aggregates every query's stats. Rank by *total* time, not mean — a 5 ms query run 200 times a second (exactly the Stack Overflow comment-fetching pattern Nick Craver describes) costs more than a 2-second query run once an hour.

```sql
SELECT
  substring(query, 1, 60) AS query,
  calls,
  round(total_exec_time::numeric, 0) AS total_ms,
  round(mean_exec_time::numeric, 2) AS mean_ms,
  round((100 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 1) AS pct
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

**Step 2 — Capture the full plan.** Always with the full flag set:

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS, FORMAT TEXT)
SELECT ...;
```

`ANALYZE` runs it and measures; `BUFFERS` adds the I/O accounting; `VERBOSE` shows output columns and schema-qualified names; `SETTINGS` (Postgres 12+) prints any non-default planner GUCs in effect — invaluable for catching "oh, someone set `enable_hashjoin = off` in this session." For write statements, wrap in `BEGIN; ... ROLLBACK;` so `ANALYZE` doesn't actually mutate data.

**Step 3 — Find the one bottleneck node.** Not the biggest cost — the biggest *exclusive actual time* (`actual time × loops`, minus children). This is where a visualizer earns its keep: paste the plan into [explain.dalibo.com](https://explain.dalibo.com/) (PEV2) or pganalyze's plan viewer and it highlights the node with the most self-time and the worst estimate. By hand, scan for the deepest node with a large `loops` or a large `shared read`.

**Step 4 — Classify the root cause.** This is the branch in the diagram, and there are essentially two families:

- **Estimate error** (actual rows ≫ estimated rows at this node): the planner was lied to. Fix the *statistics*, not the plan. Sub-branch: single-column skew → `SET STATISTICS` higher + `ANALYZE`; multi-column correlation → `CREATE STATISTICS`; stale → `ANALYZE`.
- **Access-path error** (estimates are fine, but the node does too much I/O — a `Seq Scan` with huge `shared read`, or an index scan with high `Heap Fetches`): the planner picked a reasonable plan for the stats it had, but a better access path exists. Fix with an *index* (partial, covering, expression) — or, if no index helps, *rewrite* the query (a `LATERAL` join, a CTE materialization barrier, decomposing an `OR` into a `UNION`).

**Step 5 — Fix one thing and verify with buffers.** Apply the single highest-leverage change, re-run `EXPLAIN (ANALYZE, BUFFERS)`, and compare the *buffer counts*, not the wall-clock time — because on a warm cache the second run is always faster regardless of whether you helped. If `shared read` dropped, you genuinely improved it. If only time dropped, you just warmed the cache.

### The production-monitoring layer: auto_explain

`EXPLAIN ANALYZE` is for queries you can reproduce. For the slow query that only happens at 3 a.m. under a specific data distribution, you need [`auto_explain`](https://www.postgresql.org/docs/current/auto-explain.html), which automatically logs the plan of any statement exceeding a duration threshold:

```ini
## postgresql.conf — the flight recorder for slow plans
shared_preload_libraries = 'auto_explain'
auto_explain.log_min_duration = '500ms'   # log plans of queries over 500ms
auto_explain.log_analyze = on             # include actual times (has overhead!)
auto_explain.log_buffers = on
auto_explain.log_nested_statements = on
auto_explain.log_format = 'json'
```

The mental model the EDB and Crunchy Data tuning guides use: **`pg_stat_statements` is the radar (what's expensive in aggregate), `auto_explain` is the flight recorder (the exact plan of the exact slow execution).** But heed the warning in the Postgres docs: `auto_explain.log_analyze = on` makes Postgres do per-node instrumentation on *every* statement, not just the slow ones, which "can have an extremely negative impact on performance." Turn it on temporarily during an investigation, or sample it (`auto_explain.sample_rate = 0.01`), not permanently at full rate.

## Case studies from production

The mechanics above are abstract until you've watched them play out on real systems. Here are nine documented incidents — each one is a different way the plan-reading discipline pays off, drawn from public engineering writeups and the Postgres community's collective scar tissue.

### 1. Heap and the JSONB index-only scan the planner refused to use

Heap runs analytics over a distributed Postgres cluster handling billions of events per day. Their "Effort Analysis" feature had to scan over a billion rows to compute median engagement between funnel steps, and at launch the p90 latency was about **20 seconds** — unshippable. The plan showed the query *not* using an index-only scan despite a suitable index existing; it was doing index scans that fell back to the heap. The root cause, documented in [Heap's "when the Postgres planner is not very smart"](https://www.heap.io/blog/when-the-postgres-planner-is-not-very-smart), is a genuine planner limitation: the query filtered on `data ->> 'type'` (a JSONB extraction), and even though both the expression and its result were indexed, Postgres believed it still needed the underlying `data` column from the heap to satisfy the query — "PostgreSQL's planner is currently not very smart about such cases." Including the whole `data` column (most of the event's metadata) in the index was impractical. The fix was a **partial index whose predicate exactly matched the query's filter**: `CREATE INDEX ea_index ON funnel_events WHERE data ->> 'type' IN ('click', 'change', 'touch')`. Because the index predicate matched the query predicate, Postgres realized it didn't need the heap for filtering and enabled the index-only scan. The result: a **2× p90 improvement**, dropping from ~20 s to 10–14 s, with p70 and p50 improving too. The lesson: when the planner won't use an index-only scan, the question is always "what does it think it still needs from the heap?" — and a partial index matching the predicate is a powerful, underused answer.

### 2. Heap's batch-insert fix that saved millions

A separate Heap story is a beautiful reminder that the bottleneck node isn't always where you expect. After rolling out per-customer partial indexes (tens of thousands of them across a sharded schema), CPU spiked and ingestion latency blew up — "it would take hours for a new event to show up in the Heap dashboard." The team's first hypothesis was that *evaluating* the index predicates was expensive. The plans and flame graphs in [How Basic Performance Analysis Saved Us Millions](https://heap.io/blog/engineering/basic-performance-analysis-saved-us-millions) told a different story: ~**55% of CPU was in `ExecOpenIndices`** — Postgres was repeatedly *fetching and parsing index metadata*, not evaluating predicates. Because their connection pool assigned queries round-robin across processes, a given table's index metadata was almost never cached in the process that needed it. The fix wasn't an index change at all: they **batched ~50 inserts per command** to the same table, so Postgres parsed the index metadata once per batch instead of once per row. The payoff was a **10× CPU reduction**, a 10× ingestion-throughput improvement, and an estimated savings of "millions of dollars over the next year." Lesson: read the *whole* profile, not just the query plan — sometimes the expensive thing is the planning/setup overhead the per-row plan doesn't even show.

### 3. The nested-loop blowup from a single stale estimate

The most common production incident there is. A team bulk-loads a few million rows into a table and immediately runs a join against it. Autovacuum hasn't analyzed the table yet, so the planner thinks it's nearly empty. With a tiny estimated outer side, it picks a **nested loop** — correct for an empty table, catastrophic for a few million rows. The query that the planner priced at ~120 ms runs for minutes; the inner index scan shows `loops` in the millions. This is the exact shape the 2009 [Bad plan for nested loop + limit](https://www.postgresql.org/message-id/603c8f070902271454h5a78a063kcf2b78a10fd956b1@mail.gmail.com) mailing-list thread documents, and it still happens weekly somewhere. The diagnosis is mechanical: walk the tree, find the leaf where `actual rows ≫ estimated rows`, confirm `last_analyze` is `NULL` or ancient in `pg_stat_user_tables`. The fix is `ANALYZE the_table;`. The *prevention* is to run `ANALYZE` as the last step of any bulk-load job rather than waiting for autovacuum. Lesson: the first question on any nested-loop-from-hell is "are the statistics even current?"

### 4. The correlated-columns underestimate that extended statistics fixed

A SaaS app filtered on `WHERE country = 'US' AND region = 'CA'`. Per-column stats said `country = 'US'` was 70% and `region = 'CA'` was 4%; the planner multiplied to `0.028` and estimated a few thousand rows. But `region = 'CA'` *implies* `country = 'US'` — the real selectivity was just the 4%, an order of magnitude more rows than estimated. The plan picked a nested loop and a sort sized for the underestimate, both of which then spilled. The fix, straight from the Postgres playbook and Crunchy Data's [statistics-hacking guide](https://www.crunchydata.com/blog/hacking-the-postgres-statistics-tables-for-faster-queries), was `CREATE STATISTICS geo (dependencies) ON country, region FROM users; ANALYZE users;`. The functional-dependency statistic told the planner that `region` determines `country`, so it stopped multiplying, the estimate snapped to reality, and the plan flipped to a hash join with an in-memory sort. Lesson: any time two filtered columns have a real-world relationship (city/country, region/country, brand/category, status/type), suspect a correlation underestimate and reach for `CREATE STATISTICS` before you reach for query hints.

### 5. The covering index that wasn't, because of Heap Fetches

An engineer builds the perfect covering index for a hot query, confirms it's an `Index Only Scan` in the plan, and ships it — and it's still slow under load. The plan, re-read with `BUFFERS`, shows `Heap Fetches: 41280` and large `shared read` on a node labeled "Index Only Scan." The catch: the table is write-heavy, so its visibility map is perpetually out of date, and "index only" scans keep falling back to the heap to check row visibility. The fix isn't more indexing — it's making `VACUUM` run often enough (tuning `autovacuum_vacuum_scale_factor` down on that table) to keep the visibility map fresh, which drives `Heap Fetches` toward 0. Lesson: an index-only scan is only as fast as its `Heap Fetches: 0`; on write-heavy tables, vacuum tuning is part of read performance.

### 6. Stack Overflow and the cardinality-estimator upgrade

When Stack Overflow tested SQL Server 2014, Nick Craver [reported](https://nickcraver.com/blog/2013/11/18/running-stack-overflow-sql-2014-ctp-2/) that the new release's improved **cardinality estimator** made some of their hottest queries dramatically cheaper — the comment-fetching query that "runs 200 times per second" became about **30% cheaper**. This is the same lesson from the other direction: most of the win was not from rewriting queries but from the engine getting *better at estimating rows*. It's also a reminder that engine upgrades can change plans — usually for the better, occasionally for the worse — and that the comment-fetch-200×/sec query is exactly the kind of high-frequency, modest-per-call statement that `pg_stat_statements` (ranked by *total* time) surfaces and that a naive "fix the slowest single query" approach misses entirely. Lesson: aggregate frequency × per-call cost is what matters; cardinality estimation is the lever the database vendors themselves pull hardest.

### 7. The work_mem cliff on a reporting query

A nightly report did a large `GROUP BY` and `ORDER BY` over tens of millions of rows. The plan showed `Sort Method: external merge Disk: 1.2GB` and a `HashAggregate` with `Batches: 16` — both had spilled to disk because the default `work_mem` (4 MB) was orders of magnitude too small for this workload. The query took 40 minutes. The fix was *not* to raise `work_mem` globally (which would risk OOM across hundreds of OLTP connections) but to set it locally for just the report: `SET LOCAL work_mem = '1GB';` inside the report's transaction. The sort became `quicksort` in memory, the aggregate dropped to `Batches: 1`, and the report finished in 4 minutes — a 10× win from one session-scoped setting. Lesson: `work_mem` is per-node and per-connection; tune it surgically per heavy query, never globally to a value that's safe only for one connection.

### 8. The JIT regression after a major-version upgrade

A team upgraded to Postgres 12, and a set of dashboard queries got *slower* despite no schema or data change. The plans, read with attention to the `JIT` block, showed `JIT: Functions: 30 … Timing: … Total 180ms` on queries whose actual row processing was ~60 ms — JIT compilation was costing three times the query's runtime. The trigger: these queries had row *over-estimates* that pushed their cost over the default `jit_above_cost = 100000` threshold, so the planner JIT-compiled them, but they actually returned few rows, so the compilation never paid off. The fix was twofold: fix the over-estimates where possible, and set `jit_above_cost` higher (or `jit = off`) for the dashboard workload. Lesson: after any major upgrade, if queries get mysteriously slower, check the `JIT` timing block against total runtime — JIT that costs more than it saves is a real and common regression.

### 9. The LIMIT query that walked the whole index

A "show the newest 20 unprocessed jobs" query — `WHERE processed = false ORDER BY created_at LIMIT 20` — was instant for months, then suddenly took 8 seconds. The plan looked *great*: a cheap `Index Scan` on `created_at` feeding a `Limit`, startup cost 0.43. The trap was the one from the cost-model section: the planner assumed matching rows (`processed = false`) were densely distributed near the front of the `created_at` index, so the early-stop would happen fast. But the backlog had been cleared — almost all recent jobs were `processed = true` — so the index scan had to walk *nearly the entire `created_at` index* before finding 20 unprocessed rows. The `actual time` was 8 s on a node the planner priced at 1.21. The fix was a **partial index** matching the predicate: `CREATE INDEX ON jobs (created_at) WHERE processed = false;`, so the index contained *only* unprocessed jobs and the `LIMIT 20` truly stopped after 20 entries. Lesson: a `LIMIT` plan with a tiny startup cost is a *bet* that the matching rows are near the start of the scan; when that bet is wrong, only `ANALYZE`'s actual numbers reveal it, and a partial index aligned to the predicate is the durable fix.

## The plan-reading playbook

Distilled into a checklist you can run in the war room. This is the sequence, in order, every time.

**Before you read anything:**

1. Run `EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS)`. Never bare `EXPLAIN` when you can run the query. Never without `BUFFERS`.
2. For `INSERT`/`UPDATE`/`DELETE`, wrap in `BEGIN; ... ROLLBACK;`.
3. Confirm statistics are current: `last_analyze` in `pg_stat_user_tables` for every table involved.

**Reading the tree (inside out):**

4. Walk from the leaves up. At each node compute `actual rows / estimated rows`. The first node past ~10× divergence is your prime suspect.
5. Multiply every `rows` by its `loops`. Per-loop numbers lie; the multiplier is where time hides.
6. Find the node with the largest *exclusive* `actual time × loops` (use [explain.dalibo.com](https://explain.dalibo.com/) / PEV2 to do the subtraction). That's the bottleneck.
7. On that node, check `shared read` (cold I/O), `Heap Fetches` (failed index-only), `Rows Removed by Filter` (wasted scanning), `Sort Method` / `Batches` (work_mem spills), and `loops` (nested-loop blowup).

**Classifying and fixing (one change at a time):**

8. **Estimate wrong?** → fix statistics. Stale → `ANALYZE`. Skewed column → `ALTER COLUMN ... SET STATISTICS 500; ANALYZE`. Correlated columns → `CREATE STATISTICS ... (dependencies, ndistinct, mcv); ANALYZE`.
9. **Access path wrong but estimate fine?** → fix the index. Filter discarding most rows → partial index matching the predicate. Heap fetches on a covering index → `VACUUM` / autovacuum tuning. Predicate stuck in `Filter` not `Index Cond` → extend/reorder the index. Expression predicate → expression index.
10. **Neither?** → rewrite. Nested-loop-from-hell you can't re-estimate → `LATERAL` or restructure. `OR` defeating indexes → `UNION`. CTE optimization fence (pre-12) → inline it. Last resort, *diagnostically*: `SET enable_nestloop = off` to confirm the estimate is the real problem.
11. On SSDs, set `random_page_cost = 1.1` so the planner stops over-pricing index scans.

**Verifying (trust buffers, not the clock):**

12. Re-run `EXPLAIN (ANALYZE, BUFFERS)` and compare **`shared read`**, not milliseconds. A faster second run on a warm cache proves nothing; fewer buffers proves you helped.
13. For the slow query you can't reproduce, enable `auto_explain` (temporarily, with `log_analyze` sampled) and catch it in the act.

> The whole skill compresses to one sentence: read the tree from the inside out, find the first node where the planner's estimate diverges from reality, and ask whether that lie is about *how many rows* (fix the statistics) or *how to get them* (fix the index). Everything else is detail.

## When to reach for deep plan analysis — and when not to

**Reach for `EXPLAIN ANALYZE` deep-dives when:**

- A query in `pg_stat_statements` accounts for a meaningful slice of total database time — fixing it has real leverage.
- Latency is *unstable* (sometimes 10 ms, sometimes 10 s) — that volatility almost always means a plan that flips between a good and a bad estimate, and the plan is the only place you'll see it.
- You just shipped a schema or query change and want to confirm the plan didn't regress — diff the `EXPLAIN` before and after, in CI if you can ([metis](https://www.metisdata.io/) and similar tools gate plans in CI exactly for this).
- A query is fast in staging and slow in production — that's a statistics or data-distribution difference, and only a production-clone plan reveals it.

**Skip the deep-dive and do something simpler when:**

- The query is already fast enough and runs rarely — micro-optimizing a 5 ms query that runs once a minute is a waste; spend the time on the one eating 30% of total time.
- The fix is obviously a missing index on an unindexed foreign key or `WHERE` column — add it, confirm with one `EXPLAIN`, move on. Not every slow query needs a forensic plan reading.
- The real problem is application-level: N+1 queries (fix the ORM, not the plan), missing caching (see [Redis in production](/blog/software-development/database/redis-applications-and-optimization)), or a query that shouldn't run at all.
- The bottleneck is write throughput, not reads — then you're in a different regime entirely, and the relevant knowledge is storage-engine internals like [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) and why [random UUIDs hurt insert performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance), not query plans.

The deeper truth underneath all of it — the one Kleppmann makes in *Designing Data-Intensive Applications*'s chapter on storage and retrieval — is that a query optimizer is a *declarative-to-imperative compiler*. You declare *what* you want in SQL; the planner compiles it into a *how*, choosing access methods and join algorithms from cost estimates. Reading `EXPLAIN ANALYZE` is reading that compiler's output and its profiling data side by side. Once you see it that way, the practice stops being database-specific trivia and becomes what it actually is: debugging a compiler whose decisions are only as good as the statistics you feed it. Keep the statistics honest, read the tree from the inside out, trust the buffers over the clock, and you will diagnose in forty seconds what used to take an afternoon.

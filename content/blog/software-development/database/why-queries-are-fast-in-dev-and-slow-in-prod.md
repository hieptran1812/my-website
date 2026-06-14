---
title: "Why Your Query Is Fast in Dev and Slow in Production"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The eleven mechanical reasons identical SQL runs in milliseconds on your laptop and seconds in production — plus a systematic method to reproduce prod plan-shaping conditions in dev."
tags:
  [
    "query-optimization",
    "performance",
    "postgres",
    "mysql",
    "parameter-sniffing",
    "n-plus-1",
    "statistics",
    "query-planner",
    "orm",
    "database",
    "explain",
    "indexing",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-1.webp"
---

There is a particular flavor of humiliation reserved for the engineer who ships a "fast" query. You wrote it on your laptop. You ran `EXPLAIN ANALYZE`. It came back in 3 milliseconds. You pasted the plan into the pull request, a reviewer nodded, and it merged. Then, sometime between 2 a.m. and the next on-call rotation, the same query — byte-for-byte identical, not one character changed — started taking 14 seconds in production, the connection pool saturated, request latency fanned out across every endpoint that shares the database, and your phone lit up.

The reflex is to blame the query. It is almost never the query. The SQL text is a constant; what changed is everything *around* the query. The database is not executing your `SELECT` in a vacuum. It is executing it against a specific volume of data, with a specific set of statistics describing that data, under a specific cache state, under a specific concurrency load, configured with a specific set of cost constants, possibly through an ORM that turned your one query into five thousand, possibly against a replica that is forty seconds behind the primary. Every one of those inputs is different on your laptop than it is in production, and any one of them can make the optimizer choose a plan that is correct for dev and catastrophic for prod.

This article is a tour of the eleven mechanisms that cause the dev/prod performance gap, each with the *why* (the planner internals that make it happen), a *reproduction* (runnable SQL you can paste into a `psql` session), and a *fix*. The figure below is the mental model for the whole thing: the same SQL fans out into a set of environmental inputs, most of them feed the planner, the planner picks a plan, and the runtime cost of that plan — combined with cache and concurrency effects the planner never modeled — produces the prod tail latency you got paged for.

![The same SQL text fans out into data, statistics, config, cache, and concurrency, all of which diverge between dev and prod and converge on a slower plan and runtime](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-1.webp)

The diagram above is the mental model: read it left to right. The box on the left is the only thing that is *identical* between the two environments — the query text. Everything to its right is an environmental variable that you forgot to hold constant. Three of them (volume, statistics, config) feed the **planner**, which is a cost-based optimizer that picks an execution strategy *before* it touches a single row. Two of them (cache, concurrency) never reach the planner at all — they only show up at runtime, which is why a plan that is theoretically optimal can still be slow. The rest of this article walks each arrow in that diagram and shows you how to neutralize it.

> The query optimizer is not choosing the fastest plan. It is choosing the plan with the lowest *estimated* cost, and every word of "estimated cost" is doing load-bearing work. Estimates come from statistics. Cost comes from configuration constants. Get either wrong and the optimizer will confidently choose a disaster.

A note on framing before we start. Most of the concrete examples here are PostgreSQL, because Postgres exposes its planner internals more honestly than any other mainstream database and because its `EXPLAIN (ANALYZE, BUFFERS)` output is the best teaching tool in the industry. But every mechanism here has a direct analogue in MySQL/InnoDB, SQL Server, and Oracle — they are all cost-based optimizers fed by statistics, and they all have a plan cache, a buffer pool, and cost constants. Where the analogue is interesting (parameter sniffing is *most* famous in SQL Server; the optimizer trace differs in MySQL) I will point it out. If you want to read execution plans fluently before going deep here, the companion piece [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) is the prerequisite; the [join algorithms deep dive](/blog/software-development/database/join-algorithms-nested-loop-hash-merge) explains nested-loop vs hash vs merge, which several of these failure modes flip between.

## Why "it worked in dev" is the wrong mental model

The single most damaging assumption a backend engineer can hold is that the database is deterministic with respect to the query text. It is not. The same SQL string maps to different *plans* and different *runtimes* depending on a dozen hidden inputs. Before we go mechanism by mechanism, it is worth laying out the gap between the naive mental model and the reality, because almost every production incident in this space is a specific instance of one of these mismatches.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "Same query, same plan." | The SQL text determines the execution plan. | The plan is chosen from statistics + cost constants + parameter values. All three differ between dev and prod, so the plan can differ. |
| "It's fast in dev, so it's fast." | 3 ms on my laptop generalizes. | Dev has 5k rows in cache; prod has 50M rows, a cold buffer pool, and a cost crossover that flipped the plan. |
| "The optimizer knows my data." | Postgres counts the rows. | The optimizer reads a *sample* taken at the last `ANALYZE`. After a bulk load it may believe the table is empty. |
| "Averages tell me the latency." | Mean query time is 8 ms, we're fine. | The mean is a lie under skew. The p99 is 14 s because one parameter value or one cold run dominates the tail. |
| "Indexes make queries fast." | Add an index, get speed. | An index helps only a *selective* predicate, only if the query is sargable, only if the planner estimates it's worth the random I/O. |
| "Concurrency is the DBA's problem." | My query is isolated. | Under load your query waits on locks, fights autovacuum, reads bloated heap pages, and starves for a pooled connection. |
| "Dev and prod run the same Postgres." | Same major version, same behavior. | `random_page_cost`, `work_mem`, `effective_cache_size`, and `shared_buffers` differ by 10–100×, and each reshapes the plan. |

That fourth row — about averages — is worth dwelling on because it is the conceptual error underneath all the others. In *Designing Data-Intensive Applications*, Kleppmann spends the first chapter making one point that engineers nod along to and then immediately forget: the *average* response time tells you almost nothing useful about a system under load, because the distribution of latencies is not symmetric. A handful of slow requests — the ones that hit a cold cache, or sniffed a bad parameter, or scanned a skewed value — dominate the tail, and the tail is what your users and your SLOs actually experience. The mean query time on your laptop is a single sample from a distribution that has *no tail at all*, because dev has no concurrency, no cold starts, and no skew. You are measuring the wrong statistic of the wrong distribution. When you measure p50 in dev and get paged on p99 in prod, you have not measured a faster query; you have measured a different thing entirely.

The second conceptual error lives in chapter 3 of the same book, where Kleppmann explains that an analytic or transactional query is planned by a cost model that consults *statistics* — histograms, distinct-value counts, correlation estimates — to guess how many rows each operation will produce. The optimizer is a probabilistic reasoner, not an oracle. It does not run your query to see what happens; it predicts what will happen from a compressed summary of the data, and then commits to a plan based on that prediction. If the summary is stale, or the data violates the optimizer's assumptions (chiefly: that values are uniformly distributed and that columns are independent), the prediction is wrong, and a wrong prediction at plan time is locked in for the entire execution. The optimizer cannot change its mind halfway through a nested loop that it expected to run 10 times and is now running 40 million times.

Hold those two ideas — *the mean is a lie* and *the plan is a bet placed on stale statistics* — in your head. Nine of the eleven mechanisms below are a specific way one of those two things bites you.

## 1. Data volume: the cost crossover that flips the plan

**The senior rule of thumb: a plan is only correct for a range of table sizes; cross the boundary and the optimal strategy changes, but a cached or stats-blind planner won't notice.**

This is the most fundamental reason dev and prod diverge, and it is pure cost-based optimization working exactly as designed. The optimizer assigns a cost to every candidate plan and picks the cheapest. The cost of a sequential scan is roughly proportional to the number of pages in the table. The cost of an index scan is roughly proportional to the number of *matching* rows times the cost of a random page fetch, plus the index descent. These two cost curves cross. Below the crossover, the sequential scan is cheaper (reading the whole small table beats the per-row overhead of index lookups). Above it, the index scan is cheaper. The crossover point is a function of table size, selectivity, and your cost constants.

In dev, your table has 5,000 rows. It fits in one or two megabytes, the whole thing is in cache, and a sequential scan of it costs almost nothing. The planner correctly chooses a sequential scan, and your query runs in 4 milliseconds. In prod, the same table has 40 million rows. Now the sequential scan reads 40 million rows off disk, and an index scan that touches the few hundred matching rows would be thousands of times cheaper — but if the statistics are stale (mechanism #2) or the planner mis-estimates selectivity (mechanism #4), it may still choose the seq scan, and your query runs in 38 seconds.

The same flip happens to join algorithms. With a handful of outer rows, a **nested-loop join** is optimal: for each of the 5 outer rows, probe the inner table's index. With 40 million outer rows, that nested loop runs 40 million times and you want a **hash join** instead — build a hash table on the smaller side once, then stream the larger side through it. The before/after below shows both flips together.

![At 5,000 rows the seq scan and nested loop are correct and run in 4 ms; at 40 million rows the same plan reads 40M rows and runs the loop 40M times, taking 38 s when index and hash join were correct](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-2.webp)

Let us reproduce it. Here is a self-contained Postgres session that builds a small table, shows the seq-scan plan, then grows it and shows the flip.

```sql
-- Dev-sized table: 5,000 rows.
CREATE TABLE events (
    id          bigserial PRIMARY KEY,
    user_id     bigint NOT NULL,
    event_type  text   NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

INSERT INTO events (user_id, event_type)
SELECT (random() * 1000)::bigint, 'click'
FROM generate_series(1, 5000);

CREATE INDEX idx_events_user ON events (user_id);
ANALYZE events;

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM events WHERE user_id = 42;
```

On 5,000 rows the planner may well choose a sequential scan, because reading 5,000 rows is cheaper than the index descent plus random heap fetches:

```
Seq Scan on events  (cost=0.00..98.50 rows=5 width=33)
                    (actual time=0.04..0.41 rows=4 loops=1)
  Filter: (user_id = 42)
  Rows Removed by Filter: 4996
  Buffers: shared hit=37
Planning Time: 0.09 ms
Execution Time: 0.43 ms
```

Now make it prod-sized — 40 million rows — and re-run the *identical* query:

```sql
INSERT INTO events (user_id, event_type)
SELECT (random() * 1000000)::bigint, 'click'
FROM generate_series(1, 40000000);

ANALYZE events;  -- critical; see mechanism #2

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM events WHERE user_id = 42;
```

With fresh statistics the planner now sees that `user_id = 42` matches roughly 40 of 40 million rows — a selectivity of one in a million — and flips to the index scan:

```
Index Scan using idx_events_user on events
        (cost=0.56..168.30 rows=41 width=33)
        (actual time=0.05..0.18 rows=39 loops=1)
  Index Cond: (user_id = 42)
  Buffers: shared hit=4 read=40
Planning Time: 0.12 ms
Execution Time: 0.21 ms
```

That is the system working correctly: it flipped the plan because the cost crossover moved. The danger is the case where it *fails* to flip — where the statistics are stale and the planner still believes the table has 5,000 rows. Then it keeps the sequential scan, reads all 40 million, and you get the 38-second disaster. That is mechanism #2, and it is the single most common root cause of "fast in dev, slow in prod."

### Second-order: the threshold is invisible until you cross it

The cruel property of a cost crossover is that there is no warning. The query is fast at 100k rows, fast at 500k rows, fast at 1M rows, and then at 1.4M rows — the exact point where the planner's estimated cost of the index scan finally exceeds the seq scan, or vice versa — it flips and is suddenly 200× slower, with no code change and no obvious trigger. Teams experience this as "the database randomly got slow on Tuesday." It was not random; a table crossed a threshold. The defense is to test your queries against *prod-sized* data, not dev-sized data, which is the entire point of the reproduction methodology at the end of this article.

## 2. Stale and missing statistics: the planner believes a table from last week

**The senior rule of thumb: after any bulk load, restore, or large delete, run `ANALYZE` before you trust a single query — the planner is reasoning about the data as it was at the last sample.**

The Postgres planner never counts rows at plan time. That would be far too slow. Instead it reads a precomputed summary stored in `pg_statistic` (surfaced through the `pg_stats` view): the estimated number of live rows (`reltuples` in `pg_class`), the number of distinct values per column (`n_distinct`), a list of the most common values and their frequencies (the MCV list), and a histogram of the rest. These statistics are refreshed by `ANALYZE`, which runs automatically via the autovacuum daemon — but only after enough rows have changed, and *never* synchronously with a bulk load.

This produces the classic dev/prod gap and, worse, a classic staging/prod gap. In dev you create the schema, insert a few thousand rows, and either you or some migration ran `ANALYZE`, so the stats are correct for the small table. In prod you restore a 200 GB dump, or run a nightly ETL that bulk-loads 30 million rows, and then your application starts querying *immediately*, before autovacuum has gotten around to analyzing. The planner reads `reltuples = 0` (or some tiny stale value) and plans every query as if the table were empty: it picks nested loops where it should pick hash joins, it picks sequential scans because "the table is tiny," and a query that should take 200 milliseconds takes 10 minutes.

This is not hypothetical. The Stormatics writeup ["Don't Skip ANALYZE"](https://stormatics.tech/blogs/dont-skip-analyze-a-real-world-postgresql-story) documents a production query across six tables that *never completed* after 10 minutes with CPU pegged above 60%, purely because the tables had never been analyzed after a load. Running `ANALYZE` on the involved tables let the planner recompute distributions and pick a sane plan; the query then finished in **under 20 seconds with CPU under 10%**. No SQL was changed. The only change was giving the optimizer accurate statistics. The CYBERTEC and boringSQL teams report the same pattern: a simple lookup jumps from milliseconds to seconds "right after a large batch load, purely because the planner was still trusting outdated row-count estimates."

You can see the lie directly. After a bulk load *without* `ANALYZE`, ask the planner what it thinks:

```sql
-- Right after a 40M-row bulk load, BEFORE analyze:
EXPLAIN
SELECT * FROM events WHERE user_id = 42;
```

```
Seq Scan on events  (cost=0.00..0.00 rows=1 width=33)
  Filter: (user_id = 42)
```

`rows=1` and `cost=0.00..0.00` is the planner telling you it believes the table is empty. It will cheerfully nest-loop this "1-row" table against a 10-million-row table, expecting one probe, and then do ten million. The fix is one line:

```sql
ANALYZE events;
```

After which the row estimate jumps to the real value and the plan flips to something sane. The operational lesson is to make `ANALYZE` part of your load pipeline, not a thing you hope autovacuum eventually does:

```sql
-- ETL pattern: load, then analyze, in the same transaction-adjacent step.
COPY events FROM '/data/events.csv' WITH (FORMAT csv);
ANALYZE events;          -- do this NOW, do not wait for autovacuum
-- For a fast-changing column you query selectively, raise its detail:
ALTER TABLE events ALTER COLUMN user_id SET STATISTICS 1000;
ANALYZE events;
```

### Second-order: autovacuum's analyze threshold scales with table size

The autovacuum analyze threshold is, by default, `autovacuum_analyze_threshold + autovacuum_analyze_scale_factor * reltuples`, where the scale factor defaults to 0.1. That means on a 100-million-row table, autovacuum will not trigger an analyze until **10 million rows** have changed. A daily ETL that touches 3 million rows will *never* trigger an autoanalyze on its own, and your stats will drift further from reality every day until a plan flips. On large, append-heavy tables you should lower `autovacuum_analyze_scale_factor` per-table (e.g. to 0.01 or 0.02) or schedule explicit `ANALYZE` runs. MySQL has the analogous knob in `innodb_stats_auto_recalc` and persistent statistics; the failure mode is identical.

## 3. Parameter sniffing: one cached plan for wildly different inputs

**The senior rule of thumb: a prepared statement is planned once and reused; if your data is skewed across parameter values, the first value that compiles the plan can doom every later value.**

Every database that caches execution plans has this problem, and it is the canonical reason a query is fast for one input and slow for another *with no code change at all*. SQL Server has the most famous version of it (Brent Ozar's ["elephant and the mouse"](https://www.brentozar.com/archive/2013/06/the-elephant-and-the-mouse-or-parameter-sniffing-in-sql-server/) post is required reading), but Postgres has it too, via generic vs custom plans for prepared statements.

Here is the mechanism in Postgres. When you `PREPARE` a statement (which every ORM and driver does for you under the hood — JDBC, the Postgres extended query protocol, pgx, psycopg with server-side prepares), Postgres has a choice: build a **custom plan** that bakes in the specific parameter values and re-plans on every execution, or build a **generic plan** that ignores the values and is reused. Re-planning costs CPU; reuse saves it. Postgres splits the difference with a heuristic documented in the [PREPARE docs](https://www.postgresql.org/docs/current/sql-prepare.html): the first five executions use custom plans, their average estimated cost is recorded, and from the sixth execution onward, if the generic plan's estimated cost is not "much higher" than the custom average, Postgres switches to the generic plan permanently.

The trap is data skew across the parameter. Imagine a multi-tenant table where `tenant_id = 7` is a tiny tenant with 12 rows and `tenant_id = 1` is a whale with 9 million rows. If the first executions sniff the small tenant, the custom plans are index scans, the generic plan is *also* an index scan (because the generic plan estimates the average selectivity, which is dominated by the many small tenants), and then a query for the whale tenant runs an index scan that does 9 million random heap lookups when a sequential scan would have been far cheaper.

![Parameter sniffing: the first execution sniffs a selective tenant with 12 rows and caches an index-scan plan; reusing it for a tenant with 9M rows does 9M random lookups and takes 14 s when a seq scan was correct](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-3.webp)

You can watch the custom-to-generic transition happen. Set `plan_cache_mode` to observe and force it:

```sql
PREPARE tenant_q (bigint) AS
  SELECT * FROM orders WHERE tenant_id = $1;

-- Executions 1..5: custom plans, sniffed per value.
EXPLAIN (ANALYZE) EXECUTE tenant_q(7);   -- index scan, 12 rows, fast
EXPLAIN (ANALYZE) EXECUTE tenant_q(7);
EXPLAIN (ANALYZE) EXECUTE tenant_q(7);
EXPLAIN (ANALYZE) EXECUTE tenant_q(7);
EXPLAIN (ANALYZE) EXECUTE tenant_q(7);

-- Execution 6+: may switch to a GENERIC plan, parameter shown as $1.
EXPLAIN (ANALYZE) EXECUTE tenant_q(1);   -- 9M rows via the cached plan: disaster
```

When the plan is generic, the `EXPLAIN` output shows the condition as `tenant_id = $1` instead of a baked-in literal — that is the tell. The fix in Postgres is the [`plan_cache_mode`](https://vladmihalcea.com/postgresql-plan-cache-mode/) GUC, which you can set per-session or per-role:

```sql
-- Force a fresh custom plan for every execution of prepared statements.
-- Best when the ideal plan depends strongly on the parameter (skewed tenants).
SET plan_cache_mode = force_custom_plan;

-- Or the opposite, when re-planning cost dominates and plans don't vary:
SET plan_cache_mode = force_generic_plan;
```

`force_custom_plan` pays a small re-planning cost on every execution in exchange for never reusing a pathological plan — almost always the right trade for skewed multi-tenant workloads. In SQL Server the equivalent toolbox is `OPTION (RECOMPILE)` (re-plan every time), `OPTION (OPTIMIZE FOR (@p = <typical value>))` (plan for a representative value), and `OPTIMIZE FOR UNKNOWN` (plan for the average distribution); the [Brent Ozar parameter-sniffing field guide](https://www.brentozar.com/blitzcache/parameter-sniffing/) and [SQLPerformance's RECOMPILE analysis](https://sqlperformance.com/2013/08/t-sql-queries/parameter-sniffing-embedding-and-the-recompile-options) lay out the trade-offs. The unifying lesson: a cached plan is a bet that future inputs resemble the input that compiled it. Under skew, that bet loses.

### Second-order: the first execution after a deploy is the dangerous one

Plan caches are flushed on restart, on `DISCARD ALL`, and (in SQL Server) on a recompile event or a statistics update. That means the *first* execution after a deploy, a failover, or a stats refresh is the one that sniffs and caches the plan for everyone else. If your deploy happens to coincide with an admin running a one-off query for an unusual tenant, that unusual tenant's plan can become the cached plan for the whole fleet until the next flush. This is why parameter-sniffing incidents are notoriously hard to reproduce: they depend on *which input arrived first*, which is a race condition you cannot see in the code.

## 4. Data skew: the planner assumes uniformity, your data is lumpy

**The senior rule of thumb: the optimizer's default model is that values are uniformly distributed and columns are independent; real data is neither, and the gap is where row estimates go wrong by orders of magnitude.**

Mechanisms #1–#3 are about *how much* data and *which* parameter. This one is about the *shape* of the data, and it is the deepest of the four because it is baked into the cost model's core assumptions. When the planner has no most-common-values list for a column — or when your value isn't in the MCV list — it estimates the number of matching rows as `total_rows / n_distinct`, i.e. it assumes every distinct value is equally common. That is the uniformity assumption, and real data violates it constantly.

Consider a `status` column on a 10-million-row table where 88% of rows are `'pending'`, 11% are `'active'`, and `'failed'` is a rare 0.09%. If the planner doesn't have an MCV entry for `'failed'`, it estimates `10M / 4 distinct = 2.5M` rows for `WHERE status = 'failed'` — a 250× over-estimate, since only 9,000 rows are actually `'failed'`. Believing 2.5 million rows will match, it chooses a sequential scan, when an index scan touching 9,000 rows would have been hundreds of times faster.

![Real data is skewed — pending is 88% of 10M rows while failed is 0.09% — but the planner without an MCV list assumes uniform 2.5M rows per value, over-estimating the failed filter 250x and picking a seq scan](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-6.webp)

The figure above is the entire problem in one picture. On the left is reality: a wildly skewed histogram with one towering bar. On the right is what the planner "sees" when it lacks fine-grained statistics: four equal bars, a flat uniform line. The planner plans against the right-hand picture and pays for the left-hand picture. Reproduce it:

```sql
CREATE TABLE jobs (id bigserial PRIMARY KEY, status text NOT NULL);
INSERT INTO jobs (status)
SELECT CASE
         WHEN random() < 0.88 THEN 'pending'
         WHEN random() < 0.99 THEN 'active'
         ELSE 'failed'
       END
FROM generate_series(1, 10000000);
CREATE INDEX idx_jobs_status ON jobs (status);
ANALYZE jobs;

-- The skewed value: how many does the planner THINK match?
EXPLAIN SELECT * FROM jobs WHERE status = 'failed';
```

With a default statistics target Postgres usually *does* capture `'failed'` in its MCV list for a column with only three values, and gets this right — which is the point: **the MCV list is the cure for skew.** The failure mode shows up when there are too many distinct values to fit in the MCV list (the default is 100 entries) and your hot-or-cold value falls into the histogram bucket where uniformity is assumed. The fix is to widen the statistics target so the MCV list captures more values:

```sql
-- Capture up to 1000 most-common values instead of the default 100.
ALTER TABLE jobs ALTER COLUMN status SET STATISTICS 1000;
ANALYZE jobs;
```

Skew has a second, nastier form: **correlated columns**. The planner assumes columns are independent, so it estimates the selectivity of `WHERE country = 'JP' AND language = 'ja'` by *multiplying* the individual selectivities. But country and language are highly correlated — almost everyone in Japan speaks Japanese — so the true selectivity is close to the `country = 'JP'` selectivity alone, not the product. The planner under-estimates the matching rows by the correlation factor, plans for far fewer rows than exist, and chooses a nested loop that explodes. Postgres's answer is **extended statistics**:

```sql
-- Tell the planner that country and language are correlated.
CREATE STATISTICS stx_geo (dependencies, ndistinct)
  ON country, language FROM users;
ANALYZE users;
```

This single DDL statement can turn a 40-second query back into a 40-millisecond one, because it corrects the row estimate that drives every downstream join decision.

### Second-order: skew interacts with parameter sniffing and LIMIT

Skew is rarely a lone actor. It amplifies parameter sniffing (the skewed parameter values are exactly the ones whose cached plan is wrong) and it is the root cause of the `LIMIT` plan flip in mechanism #10 (the planner assumes the few matching rows are uniformly spread through the sort order, so it expects to find them early). When you see a 100×+ gap between estimated and actual rows in an `EXPLAIN ANALYZE`, skew is the first hypothesis.

## 5. Cold cache vs warm cache: you benchmarked steady-state, prod paged on first-run

**The senior rule of thumb: your dev `EXPLAIN ANALYZE` is the warm number; the production alert fires on the cold first run after a deploy, failover, or buffer eviction — measure both.**

The query planner's cost model includes a parameter for how expensive it is to read a page, but the *runtime* depends entirely on whether that page is already in memory. Postgres has two layers of cache: its own `shared_buffers` (a fixed pool, typically 25% of RAM) and the operating system's page cache (the rest of RAM). A page that is in either is a fast memory read; a page that is in neither is a slow disk read, and on spinning rust or a congested SAN, the gap between a buffer hit and a disk read is three or four orders of magnitude.

In dev, you run the query, it pulls the pages into cache, and then you run it *again* to get a "clean" number — and you record that warm number in the PR. In prod, the query you care about is the one that runs immediately after a deploy that restarted Postgres (empty `shared_buffers`), or after a failover to a replica that has never served this workload (empty buffers), or after a burst of unrelated queries evicted your hot pages. That first cold run reads thousands of blocks off disk and takes seconds. By the time you SSH in to investigate, the cache is warm again and the query is fast, so you "can't reproduce it." This is the timeline:

![A timeline from a deploy with empty buffers through a cold first run reading 8,400 disk blocks at 1,900 ms, a partially warm second run at 240 ms, a warm steady-state at 12 ms, and then eviction returning to cold](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-4.webp)

The `BUFFERS` option to `EXPLAIN ANALYZE` is the tool that makes cache state visible. It reports `shared hit` (found in `shared_buffers`), `shared read` (had to fetch — possibly from OS cache, possibly disk), and `shared dirtied`/`written`. A plan with `Buffers: shared hit=4 read=40` did 40 reads; the same plan warm shows `Buffers: shared hit=44`.

```sql
-- Drop into a worst-case cold measurement.
-- (Requires a restart or, for OS cache, dropping caches at the OS level.)
EXPLAIN (ANALYZE, BUFFERS)
SELECT sum(amount) FROM transactions WHERE account_id = 12345;
```

```
-- COLD (right after restart):
  Buffers: shared read=8400              <- 8,400 disk reads
  Execution Time: 1893.4 ms

-- WARM (fifth run):
  Buffers: shared hit=8400               <- 8,400 buffer hits, zero disk
  Execution Time: 11.7 ms
```

Same plan. Same query. 160× difference, entirely from cache state. The defense is twofold. First, *measure the cold path on purpose*: in a benchmark, restart Postgres (or use `pg_prewarm` to control exactly which relations are warm) so you know your worst case. Second, tune `shared_buffers` and `effective_cache_size` to match prod so the planner's cost estimates and your warm/cold behavior reflect production reality (mechanism #8).

### Second-order: the planner's cost model bakes in an assumed cache hit rate

`effective_cache_size` is not a memory allocation — it does not reserve anything. It is a *hint to the planner* about how much memory is available for caching across `shared_buffers` plus the OS page cache. A larger `effective_cache_size` makes the planner believe that repeated index lookups will hit cache, lowering the estimated cost of index scans and making the planner more willing to choose them. Set it to 4 GB in dev and 96 GB in prod and the *same query* will be costed differently and may get a different plan — which means cold/warm divergence and config divergence (mechanism #8) are entangled. This is the textbook chapter-3 point: the optimizer reasons about a cost model whose constants encode assumptions about the hardware, and if those constants are wrong, the reasoning is wrong before any data is touched.

## 6. Concurrency: the failure modes that only exist under load

**The senior rule of thumb: dev runs your query alone; prod runs it against a hundred concurrent writers, an autovacuum that can't keep up, and a connection pool that's one slow query away from saturation.**

This is the category of failure that is *structurally impossible* to reproduce on a laptop, because a laptop has no concurrency. Every other mechanism in this article can be reproduced single-threaded. These cannot, and they are responsible for the most baffling incidents — the ones where the query is fast when you test it, fast in the staging environment, and slow only when real traffic is hitting the database.

**Lock waits.** Your `SELECT ... FOR UPDATE` or your `UPDATE` blocks on a row that another transaction holds. The query's CPU time is microseconds, but its wall-clock time is the duration of the conflicting transaction. In `EXPLAIN ANALYZE` the execution looks instant; the latency is entirely lock-wait time that `EXPLAIN` doesn't show. The companion piece on [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) covers this in depth; the relevant point here is that lock contention is a property of the *workload*, not the query, so it is invisible in isolation.

**MVCC bloat.** Postgres (and to a different degree InnoDB) keeps old row versions for concurrent readers — the heart of multi-version concurrency control. A heavily updated table accumulates dead tuples that a sequential scan must read past and that bloat the table's physical size. A table that holds 1 million live rows might occupy the disk footprint of 5 million rows if autovacuum has fallen behind, so every scan does 5× the I/O. Dev never sees this because dev never updates the same rows millions of times under load. The mechanics of why this happens — and why Postgres and InnoDB bloat differently — are in the [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb). Check it with:

```sql
SELECT relname,
       n_live_tup,
       n_dead_tup,
       round(n_dead_tup::numeric / nullif(n_live_tup, 0), 2) AS dead_ratio,
       last_autovacuum
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 10;
```

A `dead_ratio` above ~0.2 on a hot table means autovacuum is losing the race and your scans are reading bloat.

**Autovacuum lag.** Autovacuum is the background process that reclaims dead tuples and refreshes statistics. Under heavy write load it can fall behind, which simultaneously causes bloat (above) *and* stale statistics (mechanism #2). Worse, when it finally runs on a huge table it consumes I/O bandwidth that competes with your queries, so the very act of cleaning up makes everything slower for a window. None of this exists in dev.

**Connection-pool saturation.** This is the one that turns a single slow query into a site-wide outage. Your app has a connection pool of, say, 20 connections. Normally each query takes 5 ms, so 20 connections serve thousands of requests per second. Then one query starts taking 14 seconds (because of any mechanism above). Those slow queries hold their connections for 14 seconds each. New requests queue waiting for a free connection. The pool saturates. Now *every* endpoint that touches the database is slow, including the ones whose queries are individually fast, because they can't get a connection. The slow query is the cause, but the symptom is "the whole site is down," which sends the on-call engineer chasing the wrong thing. This is the canonical tail-latency amplification Kleppmann describes in chapter 1: a slowdown in one component, under concurrency, becomes head-of-line blocking that poisons the whole request distribution.

### Second-order: the observer's paradox of concurrency bugs

The reason these bugs feel like ghosts is that the act of investigating them changes the system. You log in, the load has already passed, autovacuum has caught up, the locks have released, the cache is warm, and the query is fast — so you conclude it was a fluke. The only defense is *continuous* instrumentation that records what the system looked like *during* the incident: `pg_stat_statements` for per-query aggregates, `pg_stat_activity` snapshots for what was waiting on what, and `log_lock_waits = on` so lock waits over `deadlock_timeout` get logged. You cannot reproduce a concurrency bug after the fact; you can only have recorded enough to diagnose it.

## 7. ORM N+1: the query you wrote is not the query that ran

**The senior rule of thumb: lazy loading turns one logical query into one query per parent row; the cost is invisible in dev because N is small and the database is local.**

Every mechanism so far has been about a single query. This one is about the gap between the *one query you think you wrote* and the *thousands of queries the ORM actually issued*. It is the most common performance bug in application code, and it scales precisely with the dev/prod data gap.

The N+1 problem: you fetch a list of parents (1 query), then iterate over them and access a lazily-loaded association on each one, firing one query per parent (N queries). In dev, your test fixture has 11 posts, so you fire 12 queries, each taking 0.2 ms against a local database, total 2.4 ms — invisible. In prod, the page renders 5,000 posts, so you fire 5,001 queries, each taking 1.5 ms across a network round trip to a different host, total 7.5 *seconds*. The query text was never slow. There were just five thousand of them, and round-trip latency dominated.

![Lazy loading fires SELECT posts then one SELECT author per post for 5,001 round trips; eager loading fires SELECT posts then one SELECT authors WHERE id IN for a constant 2 round trips regardless of N](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-5.webp)

Here is the bug in Rails ActiveRecord, the framework where the [Bullet gem](https://github.com/flyerhzm/bullet) was built specifically to catch it:

```ruby
# N+1: one query for posts, then one query PER post for its author.
posts = Post.where(published: true)   # 1 query: SELECT * FROM posts ...
posts.each do |post|
  puts post.author.name              # N queries: SELECT * FROM authors WHERE id = ?
end
# Dev (11 posts): 12 queries, ~3 ms. Prod (5000 posts): 5001 queries, ~7.5 s.
```

The fix is eager loading — tell the ORM to fetch the associations up front in a single additional query using an `IN` list:

```ruby
# Eager load: 2 queries total, regardless of how many posts.
posts = Post.where(published: true).includes(:author)
# 1: SELECT * FROM posts WHERE published = true
# 2: SELECT * FROM authors WHERE id IN (1, 2, 3, ... )   <- one query for ALL authors
posts.each do |post|
  puts post.author.name              # no query: already loaded
end
```

Every ORM has this. In Django it is `select_related` (for `JOIN`-based eager loading of to-one relations) and `prefetch_related` (for `IN`-based eager loading of to-many relations):

```python
# Django N+1:
for post in Post.objects.filter(published=True):
    print(post.author.name)          # one query per post

# Django fix — JOIN the author in:
for post in Post.objects.filter(published=True).select_related("author"):
    print(post.author.name)          # zero extra queries

# For a to-many relation, batch with prefetch_related (issues one IN query):
for post in Post.objects.filter(published=True).prefetch_related("comments"):
    print(len(post.comments.all()))  # one query for ALL comments
```

In SQLAlchemy it is `selectinload` / `joinedload`; in Hibernate it is `JOIN FETCH` or an `@BatchSize`. The pattern is universal, and so is the fix: replace per-row lazy loads with a single batched fetch.

### Second-order: detect N+1 in tests, not in prod

The reason N+1 survives code review is that it is invisible at dev scale. The defense is to make it visible mechanically. Bullet raises a notification (or, configured strictly, an exception) the moment it detects a lazy load that should have been eager. Wire it into your *test suite* so an N+1 fails CI:

```ruby
# config/environments/test.rb
Bullet.enable = true
Bullet.raise  = true   # turn detected N+1 into a test failure

# In a request spec, Bullet now fails the test if the controller N+1s.
```

The deeper lesson is that an ORM is a leaky abstraction over a network protocol. It makes `post.author` look like a free attribute access, but it is a synchronous round trip to another machine. The number of round trips, not the cleverness of any single query, is what kills you, and the only way to count round trips is to log the actual SQL the ORM emits (`ActiveRecord::Base.logger`, Django's `django.db.backends` logger, SQLAlchemy's `echo=True`).

## 8. Config divergence: same SQL, different cost constants, different plan

**The senior rule of thumb: the planner's decisions are functions of cost GUCs, and a laptop's defaults differ from a tuned prod box by 10–100× on every constant that matters.**

The planner does not have absolute knowledge of your hardware. It has a handful of tunable constants — `random_page_cost`, `seq_page_cost`, `effective_cache_size`, `work_mem`, `cpu_tuple_cost` — and it costs every plan against those constants. If the constants differ between dev and prod, the *same query against the same data* gets a different cost and can get a different plan. This is config divergence, and it is the most overlooked cause of the dev/prod gap because it requires zero data difference to manifest.

![A matrix comparing dev defaults to tuned prod SSD values across five GUCs, with the plan consequence of each: dev under-uses index scans, prod prefers index plus cache, sorts spill in dev, cold reads differ, parallel plans only in prod](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-8.webp)

The single most consequential constant is **`random_page_cost`**. It defaults to **4.0**, meaning "a random page read costs 4× a sequential page read." That ratio was reasonable for spinning disks in 2005, where a seek genuinely cost several milliseconds. On an SSD, random reads are nearly as cheap as sequential reads, so the realistic ratio is closer to **1.1** — and [every SSD tuning guide](https://blog.frehi.be/2025/07/28/tuning-postgresql-performance-for-ssd/) recommends setting it there. The effect on plans is direct: a lower `random_page_cost` makes index scans (which do random reads) look cheaper relative to sequential scans, so the planner uses indexes more. If dev is on the default 4.0 and prod is tuned to 1.1, the same query can pick a seq scan in dev and an index scan in prod, or vice versa, with no data change. There is a [famous Hacker News thread](https://news.ycombinator.com/item?id=15761026) about a single `random_page_cost` change improving a slow query by 50×, and [pganalyze's writeup](https://pganalyze.com/blog/5mins-postgres-tuning-random-page-cost) walks through exactly how the constant interacts with index correlation to flip plans.

The second constant that bites is **`work_mem`**, the per-operation memory budget for sorts and hash tables. It defaults to **4 MB**. If a sort or hash join needs more than `work_mem`, Postgres *spills to disk* — an external merge sort or a multi-batch hash join — which is dramatically slower than the in-memory version. In dev with a small result set, the sort fits in 4 MB and runs in memory. In prod, the result set is large, the sort exceeds 4 MB, it spills, and the query crawls. Worse, this can flip the *plan*: if the planner believes a hash join won't fit in `work_mem` it may choose a merge join or nested loop instead. You can see a spill in `EXPLAIN ANALYZE`:

```
Sort  (cost=... rows=2000000 ...) (actual ...)
  Sort Method: external merge  Disk: 247856kB    <- spilled 242 MB to disk
```

`Sort Method: external merge ... Disk:` is the smell. The fix is to raise `work_mem` — but carefully, because it is *per operation per connection*, so a complex query with three sorts across 100 connections can allocate `3 × 100 × work_mem`. Set it conservatively as a global and raise it per-session for known-heavy queries:

```ini
# postgresql.conf — match dev to prod for plan-shaping constants.
random_page_cost = 1.1            # SSD; was 4.0 default
effective_cache_size = 96GB       # ~75% of RAM on a 128GB box; was tiny in dev
work_mem = 64MB                   # per-op; raise per-session for big sorts
shared_buffers = 32GB             # ~25% of RAM; affects warm/cold behavior
max_parallel_workers_per_gather = 4   # 0 in dev means no parallel plans at all
```

```sql
-- Per-session bump for one heavy reporting query, leaving the global low:
SET work_mem = '512MB';
SELECT ... ORDER BY ... ;  -- now sorts in memory
RESET work_mem;
```

That `max_parallel_workers_per_gather` row matters too: if it is 0 in dev (or the table is below `min_parallel_table_scan_size`), dev *never* generates a parallel plan, so the plan shape you tested is structurally different from the parallel plan prod runs. You cannot validate a parallel query's behavior on a config that disables parallelism.

### Second-order: the goal is plan parity, not identical hardware

You do not need a dev box with 128 GB of RAM to catch config-driven plan flips. You need dev to *cost queries the way prod does*, which means setting the **planner cost constants** (`random_page_cost`, `effective_cache_size`, `cpu_*_cost`, `work_mem`, the parallelism knobs) to the prod values, even on small hardware. The cost constants change which plan is *chosen*; the hardware changes how fast that plan *runs*. To catch a plan flip you only need the former, which is free. This is the cheapest, highest-leverage step in the entire reproduction methodology.

## 9. The index that exists but is never used

**The senior rule of thumb: an index helps only if the predicate is sargable, the types match, the column is leading, and the predicate is selective — break any one and the planner reverts to a seq scan.**

You added the index. You confirmed it exists. The query is still doing a sequential scan in prod. This is maddening, and it almost always comes down to one of a small set of reasons the predicate is not "sargable" (Search-ARGument-able — usable as an index search condition). The most insidious thing about these is that they are *silent*: the index exists, the query returns correct results, nothing errors. It is just slow, and the `EXPLAIN` shows a `Seq Scan` where you expected an `Index Scan`.

![A matrix of nine reasons an index goes unused: function on column, implicit cast, leading-column rule, low selectivity, OR across columns, and NULL semantics — each with what the query does, why the index is skipped, and the fix](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-7.webp)

**Function on the column.** An index on `email` is an index on the *value* of `email`, not on `lower(email)`. The moment your predicate wraps the column in a function, the index is useless:

```sql
-- Index on email, but the predicate is on lower(email): seq scan.
CREATE INDEX idx_users_email ON users (email);
SELECT * FROM users WHERE lower(email) = 'a@b.com';   -- Seq Scan

-- Fix: an expression index that matches the predicate.
CREATE INDEX idx_users_email_lower ON users (lower(email));
SELECT * FROM users WHERE lower(email) = 'a@b.com';   -- Index Scan
```

**Implicit cast / type mismatch.** If the column is `bigint` and you compare it to a string literal, Postgres casts one side, and a cast wrapping the column disables the index exactly like a function does. This is a classic ORM-introduced bug: the driver binds an ID as text.

```sql
-- id is bigint; comparing to text forces a cast and disables the index.
SELECT * FROM orders WHERE id = '42';     -- may seq scan depending on cast direction
SELECT * FROM orders WHERE id = 42;       -- index scan
```

**Leading-column rule.** A composite index on `(a, b)` can serve predicates on `a`, or on `a AND b`, but *not* on `b` alone — the index is sorted by `a` first, so a predicate on only `b` has nothing to seek on. This is the most common "I have an index, why isn't it used" cause.

```sql
CREATE INDEX idx_ab ON t (a, b);
SELECT * FROM t WHERE a = 1 AND b = 2;    -- uses the index
SELECT * FROM t WHERE b = 2;              -- CANNOT use it; seq scan
-- Fix: an index leading with b, e.g. (b, a) or a standalone (b).
```

**Low selectivity.** If a predicate matches most of the table — `WHERE status = 'active'` where 90% of rows are active — the planner correctly decides that an index scan (random heap fetches for 90% of rows) is *slower* than just sequentially scanning everything. This is the planner being right; the fix is a **partial index** on the rare values you actually filter for:

```sql
-- A partial index that only covers the selective case.
CREATE INDEX idx_jobs_failed ON jobs (created_at) WHERE status = 'failed';
SELECT * FROM jobs WHERE status = 'failed' ORDER BY created_at;  -- tiny, fast
```

**`OR` across columns** prevents a single index from covering both arms; rewrite as a `UNION` of two index scans or rely on a bitmap `OR`. **Inequalities and `<>`** are not sargable; rewrite as `IN`/`BETWEEN` or use a partial index. **`NULL` semantics** trip people up because `col <> $1` excludes `NULL` rows, and indexes handle `NULL` specially.

### Second-order: dev hides this because seq scans are fast on tiny tables

The reason this is a dev/prod gap and not just a bug is that in dev, the unused index doesn't matter: a seq scan of 5,000 rows is 4 ms, so the query is fast whether or not the index is used. You never notice the index is dead. In prod, the seq scan is 40 million rows and 38 seconds. The bug was present the whole time; it was only *expensive* at scale. Always confirm the index is actually used by reading the `EXPLAIN` plan — `Index Scan` or `Bitmap Index Scan`, not `Seq Scan` — rather than assuming that "the index exists" means "the index is used."

## 10. LIMIT + ORDER BY: the optimization that becomes a trap

**The senior rule of thumb: `ORDER BY ... LIMIT n` makes the planner gamble that it will find n matching rows early in the sort order; when the filter and the sort are uncorrelated, it scans almost everything first.**

This deserves its own section because it is subtle, extremely common (every paginated, "latest N" query has this shape), and the failure is a genuine plan flip driven by the interaction of `LIMIT` with the planner's uniformity assumption. It is also one of the most dramatic — the [pganalyze writeup](https://pganalyze.com/blog/5mins-postgres-planner-order-by-limit) documents the *identical query* going from **0.035 ms to 155 ms** purely from adding rows, and Postgres mailing-list reports describe queries flipping from sub-millisecond to **multi-second or worse**.

The mechanism: when you write `WHERE filter_col = X ORDER BY sort_col LIMIT 50`, the planner has two strategies. (A) Use an index on `filter_col`, fetch all matching rows, sort them, take 50. (B) Walk an index on `sort_col` in order, checking each row against the filter, and stop as soon as it has 50 matches. Strategy B is brilliant *if* matching rows are common in the sort order — it can stop early and never sort. But the planner estimates how early it will stop using the **uniformity assumption**: it believes the matching rows are evenly spread through the sort order, so if 1% of rows match, it expects to find 50 matches after scanning ~5,000 rows. If the filter and sort columns are *uncorrelated or negatively correlated*, the matching rows are clustered at the far end of the sort order, and strategy B scans millions of rows before finding 50 — or scans the entire index and finds none until the end.

Reproduce the flip exactly as pganalyze does:

```sql
CREATE TABLE orders_test (
    order_id      int PRIMARY KEY,
    shipping_date date
);
CREATE INDEX ON orders_test (shipping_date, order_id);

-- Small: the composite index is used, query is instant.
INSERT INTO orders_test SELECT g, '2022-05-01'::date + (g % 30)
FROM generate_series(1, 1000) g;
ANALYZE orders_test;

EXPLAIN (ANALYZE)
SELECT * FROM orders_test
WHERE shipping_date = '2022-05-01'
ORDER BY order_id LIMIT 50;
-- Index Scan, ~0.035 ms
```

Now flood the table with 100,000 rows clustered on that one date and re-run the identical query. The planner now thinks: "I want 50 rows ordered by `order_id`, and lots of rows match the date, so let me walk the primary key (which is `order_id`) and filter." But the matching rows are *not* early in `order_id` order, so it scans far more than expected:

```
Limit  (actual time=... rows=50)
  ->  Index Scan using orders_test_pkey on orders_test
        Filter: (shipping_date = '2022-05-01')
        Rows Removed by Filter: 2000000       <- scanned 2M rows to find 50
  Execution Time: 155 ms                       <- was 0.035 ms
```

`Rows Removed by Filter: 2000000` is the smoking gun: the plan walked two million rows of the wrong index to satisfy a `LIMIT 50`. The fixes, in order of preference:

```sql
-- 1. A composite index aligning filter and sort so one index serves both.
CREATE INDEX idx_orders_date_id ON orders_test (shipping_date, order_id);

-- 2. Raise the statistics target so the planner sees the poor correlation.
ALTER TABLE orders_test ALTER COLUMN shipping_date SET STATISTICS 1666;
ANALYZE orders_test;

-- 3. The "+0 trick": adding a no-op expression stops the planner from
--    believing the sort index satisfies the ORDER BY, forcing it to use
--    the filter index and sort the (few) matching rows.
SELECT * FROM orders_test
WHERE shipping_date = '2022-05-01'
ORDER BY order_id + 0 LIMIT 50;
```

The composite index is the real fix; the `+0` trick is a tactical override for when you can't change DDL. The statistics-target bump helps the planner notice the correlation problem so it stops gambling on the sort index.

### Second-order: keyset pagination sidesteps the whole class

The reason `LIMIT/OFFSET` pagination is so prone to this is that deep offsets force the planner to produce and discard `OFFSET` rows, and the sort/filter interaction compounds at depth. **Keyset (cursor) pagination** — `WHERE (sort_col, id) > (last_sort, last_id) ORDER BY sort_col, id LIMIT n` — uses a composite index seek that is `O(n)` regardless of how deep you page, and never gambles on finding matches early. For any large, paginated dataset, keyset pagination is the structural fix; the `LIMIT + ORDER BY` plan flip is mostly a problem of `OFFSET`-based pagination over uncorrelated columns.

## 11. Reading from a lagging replica

**The senior rule of thumb: a read replica is eventually consistent; "the row I just wrote isn't there" and "this query is slower than on the primary" are both symptoms of the replica, not the query.**

The last mechanism is not about the planner at all — it is about *which database you are actually talking to*. Production read-heavy systems route reads to one or more replicas to offload the primary. In dev, you have a single database, so there is no replica and no lag. In prod, the replica can be seconds or minutes behind, and that lag produces two distinct "slow/wrong in prod only" symptoms.

First, **stale reads**: you write a row to the primary, immediately read it back from a replica, and it's not there yet, so your application re-queries, retries, or falls into an error path that does extra work. The query was fast; the *retry loop* triggered by replication lag is slow. Second, **the replica's plan and runtime can differ from the primary's**, because the replica may have different statistics timing, a different cache state (it serves a different query mix), and — on a hot-standby Postgres replica — query cancellations from `max_standby_streaming_delay` when a long read conflicts with replay of the WAL. A report query that runs fine on the primary can be *cancelled* on the replica with `ERROR: canceling statement due to conflict with recovery`, which the app retries, multiplying load.

```sql
-- On the replica, check how far behind it is:
SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
-- replication_lag = 00:00:43.5   <- 43 seconds behind the primary
```

The defenses are architectural. For read-your-writes consistency, route reads that must see a recent write to the primary, or use `synchronous_commit` with a synchronous standby for the writes that need it, or track the write LSN and wait for the replica to catch up before reading. For replica plan/runtime divergence, run `ANALYZE` cadence and `pg_prewarm` on replicas too, and tune `max_standby_streaming_delay` / `hot_standby_feedback` to balance replica freshness against query cancellation. The meta-lesson: in prod, "the database" is a *fleet*, and which member you hit changes the answer and the latency. Dev's single node hides all of it.

## A systematic method to reproduce production in development

Every mechanism above has the same root cause: dev holds the query constant but lets the *environment* vary. So the cure is a methodology that makes dev's environment match prod's along the five axes that shape plans and runtimes — statistics, cost constants, data volume/skew, cache state, and observability. You do not need prod hardware; you need prod *plan-shaping inputs*. The pipeline below is the checklist.

![A six-step pipeline to reproduce prod: dump prod stats, match GUCs, load prod-size anonymized data, ANALYZE to prod target, EXPLAIN ANALYZE BUFFERS with generic and custom plans, then confirm in prod with auto_explain and pg_stat_statements](/imgs/blogs/why-queries-are-fast-in-dev-and-slow-in-prod-9.webp)

### Step 1 — Clone the statistics, not just the schema

The cheapest way to get prod's *plans* without prod's *data* is to copy prod's statistics into a dev database that has the same schema. The planner reads `pg_statistic`; if you give dev the same `pg_statistic` rows, dev will produce prod's plans even on an empty table. You can dump statistics with `pg_dump --section=...` workflows or, more directly, copy `pg_statistic`/`pg_statistic_ext_data` and set `reltuples`/`relpages` on `pg_class` to prod's values. This is a power move for plan debugging: the table can be empty and the planner will still choose the plan it would choose in prod, because the planner *only ever looks at the statistics*.

### Step 2 — Match the cost GUCs

Set the planner cost constants in dev to prod's values. This is the single highest-leverage, lowest-cost step, because it catches every config-driven plan flip (mechanism #8) for free:

```sql
-- Run these in dev to cost queries the way prod does.
SET random_page_cost = 1.1;
SET effective_cache_size = '96GB';
SET work_mem = '64MB';
SET max_parallel_workers_per_gather = 4;
-- Then EXPLAIN your query: the chosen plan now matches prod's plan.
```

### Step 3 — Load prod-size, prod-shaped, anonymized data

For runtime (not just plan) reproduction, you need data with prod's *volume and skew*, anonymized for safety. The key word is *shaped*: a uniformly-generated 40M-row table will not reproduce a skew bug. Either anonymize a prod snapshot (mask PII, keep distributions) or generate synthetic data that matches the real histograms. Tools in this space: `pg_anonymizer`/`anon` extension, or a snapshot-and-mask pipeline. Whatever you use, preserve the distribution of the columns you filter and join on.

### Step 4 — ANALYZE to the prod statistics target

After loading, run `ANALYZE` with the *same* per-column statistics targets prod uses, so the MCV lists and histograms have prod's resolution (mechanisms #2, #4). If prod sets `STATISTICS 1000` on a skewed column, dev must too, or dev's planner sees a coarser distribution and chooses a different plan.

### Step 5 — EXPLAIN under prod conditions, both plan modes

Now run `EXPLAIN (ANALYZE, BUFFERS)` and read it like the [reading EXPLAIN ANALYZE](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) guide teaches: compare estimated vs actual rows (a >10× gap means a statistics/skew problem), check `Buffers` for the cold/warm story, and check `Sort Method` for spills. Crucially, test **both** the custom plan and the generic plan (mechanism #3), because your app uses prepared statements:

```sql
-- Test the generic plan your app will actually get after 5 executions:
SET plan_cache_mode = force_generic_plan;
PREPARE q (bigint) AS SELECT ... WHERE tenant_id = $1;
EXPLAIN (ANALYZE) EXECUTE q(1);     -- the whale tenant under the generic plan
SET plan_cache_mode = force_custom_plan;
EXPLAIN (ANALYZE) EXECUTE q(1);     -- compare
```

### Step 6 — Confirm in prod with continuous observability

Some mechanisms (concurrency, cache eviction, replica lag) cannot be reproduced offline at all. For those, you instrument prod so the data exists when the incident happens. Turn on `auto_explain` to log the plan of any query over a latency threshold, and use `pg_stat_statements` to find the queries whose *total* time dominates (the ones called a million times, not the one slow one):

```ini
# postgresql.conf
shared_preload_libraries = 'pg_stat_statements,auto_explain'
auto_explain.log_min_duration = '500ms'   # log the plan of anything over 500ms
auto_explain.log_analyze = on             # include actual rows/timing
auto_explain.log_buffers = on             # include cache state
pg_stat_statements.track = all
```

```sql
-- Find the queries that actually cost you the most wall-clock time in prod.
SELECT query,
       calls,
       round(total_exec_time::numeric, 1)  AS total_ms,
       round(mean_exec_time::numeric, 2)   AS mean_ms,
       round((100 * total_exec_time / sum(total_exec_time) OVER ())::numeric, 1) AS pct
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

That `pct` column is the Kleppmann chapter-1 lesson operationalized: rank by *total* contribution to latency, because the query that is 8 ms but runs ten million times costs you more than the one that is 14 s but runs twice — and the one whose *mean is low but tail is high* is the one hiding a parameter-sniffing or cold-cache problem you'll only see in the `auto_explain` logs.

## Case studies from production

The mechanisms above are abstract until they bite. Here are twelve incidents — composites drawn from the patterns in the sources cited throughout — each mapped to its mechanism, with the symptom, the wrong first hypothesis, the actual root cause, the fix, and the lesson.

### 1. The query that died right after the nightly ETL

**Symptom:** A dashboard query that ran in 180 ms all day timed out at 30 s every morning between 2:00 and 2:40 a.m., then recovered on its own. **Wrong hypothesis:** "The ETL is holding locks." **Root cause:** The 2 a.m. ETL bulk-loaded 30 million rows and the dashboard started querying immediately; autovacuum's analyze threshold (10% of a 100M-row table = 10M rows) wasn't met, so the planner believed the new partition had near-zero rows and nested-looped it against a 50M-row table. By 2:40 a.m. an explicit nightly `ANALYZE` finally ran and fixed it. **Fix:** Added `ANALYZE events_today;` to the end of the ETL job and lowered `autovacuum_analyze_scale_factor` to 0.02 on the append-heavy tables. **Lesson:** Stale statistics after a bulk load (mechanism #2) is the single most common "slow in prod" cause, and the fix is one line in the load pipeline.

### 2. The tenant whose dashboard took down everyone's

**Symptom:** Every few weeks, the entire app went slow for ten minutes, then recovered. APM showed *all* endpoints slow, not one. **Wrong hypothesis:** "The database server is overloaded." **Root cause:** A prepared statement filtered by `tenant_id`. Most tenants were small, so the cached generic plan was an index scan. When the largest tenant (9M rows) loaded their dashboard right after a deploy flushed the plan cache, the index scan did 9 million random heap fetches, took 14 s, held its pooled connection the whole time, and connection-pool saturation (mechanism #6) made *every* endpoint slow. **Fix:** `SET plan_cache_mode = force_custom_plan` on the role used by the dashboard, plus a separate connection pool for heavy reports. **Lesson:** Parameter sniffing (mechanism #3) under skew, amplified by pool saturation, turns one slow query into a site outage.

### 3. The index that worked on the laptop and vanished in prod

**Symptom:** A user-lookup query was 2 ms in dev, 9 s in prod. Both had the index. **Wrong hypothesis:** "Prod's index is corrupt; reindex it." **Root cause:** The app's ORM bound the email as `lower(?)` to do a case-insensitive lookup, but the index was on `email`, not `lower(email)`. In dev the table had 3,000 rows so the seq scan was instant and nobody noticed the index was dead; in prod 12 million rows made the seq scan a 9 s disaster (mechanism #9). **Fix:** `CREATE INDEX ... ON users (lower(email));`. **Lesson:** An unused index is invisible at dev scale. Always confirm `Index Scan` in the plan, never assume the index is used because it exists.

### 4. Fast in staging, dead in prod, identical config

**Symptom:** A report ran in 400 ms in staging and 22 s in prod, same Postgres version, same query, same `postgresql.conf`. **Wrong hypothesis:** "Prod hardware is throttled." **Root cause:** Staging had a 2-week-old restore with fresh `ANALYZE` from the restore process; prod hadn't been analyzed since a schema migration three weeks earlier had rewritten a table and reset its statistics. The plans were *different*: staging used a hash join, prod a nested loop, because prod's row estimates were stale (mechanism #2). **Fix:** `ANALYZE` on the affected tables and a scheduled weekly `ANALYZE` on slow-changing large tables. **Lesson:** "Same config" is not "same statistics." Statistics are state, and state drifts.

### 5. The 50× win from one config line

**Symptom:** A geospatial range query took 4 s; the team had spent two days rewriting it. **Wrong hypothesis:** "The query needs a better algorithm." **Root cause:** The box had moved to NVMe SSDs but `random_page_cost` was still the default 4.0, so the planner thought the index scan's random reads were 4× costlier than they were and chose a seq scan (mechanism #8). **Fix:** `random_page_cost = 1.1`. The query dropped to 80 ms — a 50× improvement from one line, matching the well-known [Hacker News report](https://news.ycombinator.com/item?id=15761026). **Lesson:** Default cost constants encode 2005 spinning-disk hardware. On SSD, `random_page_cost = 1.1` is table stakes, and a wrong constant makes the planner reason correctly about the wrong machine.

### 6. The pagination query that was fine until page 1,000

**Symptom:** A "latest orders" list was instant on page 1 and 4 s on page 1,000. **Wrong hypothesis:** "Deep pagination is just inherently slow." **Root cause:** `ORDER BY order_id LIMIT 20 OFFSET 20000` combined with a date filter. The planner walked the `order_id` index expecting to find 20 matches early, but the matching dates were clustered at the far end, so it scanned ~2M rows (mechanism #10), and the `OFFSET` discarded the first 20,000 of them. **Fix:** Keyset pagination — `WHERE (order_date, id) < (?, ?) ORDER BY order_date DESC, id DESC LIMIT 20` — backed by a composite index. **Lesson:** `LIMIT + ORDER BY` over uncorrelated columns is a plan-flip trap; keyset pagination sidesteps the whole class.

### 7. The API endpoint that got slower as the company grew

**Symptom:** An endpoint that listed projects with their owners crept from 50 ms to 6 s over a year, with no code changes. **Wrong hypothesis:** "We need to add caching." **Root cause:** Classic N+1 (mechanism #7): the serializer lazily loaded `project.owner` for each project. At 10 projects it was 11 queries; at 1,200 projects it was 1,201 queries, and per-query round-trip latency to a now-separate database host dominated. **Fix:** `Project.includes(:owner)` (Rails) — two queries total — and Bullet wired into CI to fail any new N+1. **Lesson:** N+1 cost scales with both N and round-trip latency; it is invisible in dev (small N, local DB) and lethal in prod (large N, networked DB).

### 8. The replica that returned yesterday's data slowly

**Symptom:** After a write, users intermittently saw stale data and a spinner that lasted seconds. **Wrong hypothesis:** "The cache is stale; bust the cache." **Root cause:** Reads were routed to a replica running 40 s behind during peak write load (mechanism #11). The app's read-after-write didn't find the row, fell into a retry-with-backoff loop, and the cumulative retries were the "slow" part. **Fix:** Route read-your-writes queries to the primary; track the write LSN and wait for the replica to catch up for the rest. **Lesson:** In prod, "the database" is a fleet; replica lag manifests as both wrong answers and slow retry loops that dev's single node never shows.

### 9. The sort that spilled to disk only at scale

**Symptom:** A nightly aggregation took 90 s in prod, 1.5 s in dev. **Wrong hypothesis:** "Prod's disk is slow." **Root cause:** The aggregation sorted 2 million rows. In dev the result fit in the 4 MB default `work_mem` and sorted in memory; in prod it needed 240 MB, spilled to an external merge sort on disk (mechanism #8), and the `EXPLAIN` showed `Sort Method: external merge Disk: 247856kB`. **Fix:** `SET work_mem = '512MB'` for that one session, leaving the global conservative. The sort went back to memory and the query to 8 s. **Lesson:** `work_mem` is per-operation; a sort that fits in dev spills in prod, and the spill is in the plan if you read `Sort Method`.

### 10. The correlated columns that fooled the join planner

**Symptom:** A filter on `country` and `language` together produced a nested loop that ran for minutes; either filter alone was fast. **Wrong hypothesis:** "Add a composite index." (It didn't help.) **Root cause:** The planner multiplied the two selectivities assuming independence, estimated a tiny number of matching rows, and chose a nested loop — but country and language are correlated, so the true match count was 100× higher and the loop exploded (mechanism #4). **Fix:** `CREATE STATISTICS (dependencies) ON country, language FROM users; ANALYZE;`. The corrected estimate flipped the plan to a hash join and the query to sub-second. **Lesson:** The independence assumption is a silent row-estimate killer; extended statistics is the targeted cure.

### 11. The cold query that only the first user of the day hit

**Symptom:** The first request each morning to a particular report took 6 s; every subsequent one took 90 ms. **Wrong hypothesis:** "The app's JIT/warmup is slow." **Root cause:** Overnight, unrelated batch jobs evicted the report's pages from `shared_buffers` and the OS page cache; the first morning request did 8,000 cold disk reads (mechanism #5), visible as `Buffers: shared read=8000` in `auto_explain`. **Fix:** A `pg_prewarm` of the report's hot relations at the end of the batch window, and `auto_explain.log_buffers = on` to confirm. **Lesson:** Your benchmark is the warm number; prod's worst case is the cold first run, and `BUFFERS` is the only way to see it after the fact.

### 12. The bloated table that scanned 5× its live size

**Symptom:** A queue table's "fetch next job" query degraded from 3 ms to 800 ms over a week, then reset after a maintenance window. **Wrong hypothesis:** "The queue is too big; archive old jobs." (It was already small.) **Root cause:** The queue had 50,000 live rows but, under heavy update churn, 250,000 dead tuples that autovacuum couldn't keep up with (mechanism #6). Every scan read past 5× the live data. `pg_stat_user_tables` showed `n_dead_tup` five times `n_live_tup`. **Fix:** Aggressive per-table autovacuum settings (`autovacuum_vacuum_scale_factor = 0.01`) and a `VACUUM` in the maintenance window. **Lesson:** MVCC bloat under concurrency is a runtime cost the planner doesn't model and dev never produces; monitor `n_dead_tup` on hot, churny tables.

## When to suspect which mechanism

Pattern-match the symptom to the mechanism so you start the investigation in the right place instead of rewriting the query.

| Symptom | Most likely mechanism | First thing to check |
| --- | --- | --- |
| Slow right after a bulk load / restore / migration | #2 stale statistics | `EXPLAIN` row estimate vs actual; run `ANALYZE` |
| Fast for most inputs, slow for one | #3 parameter sniffing or #4 skew | Is the plan generic (`$1`)? Is the value rare/common? |
| Slow only under production load | #6 concurrency | `pg_stat_activity` waits, `n_dead_tup`, pool saturation |
| Slow only on the first run after deploy | #5 cold cache | `Buffers: shared read=...` in `auto_explain` |
| Index exists but `Seq Scan` in plan | #9 unused index | Function/cast on column? Leading-column? Selectivity? |
| Fast without `LIMIT`, slow with `ORDER BY ... LIMIT` | #10 LIMIT plan flip | `Rows Removed by Filter` huge? Composite index? |
| Same query slower than another env, same data | #8 config divergence | Compare `random_page_cost`, `work_mem`, `effective_cache_size` |
| Got slow gradually as the table grew | #1 volume crossover | Test against prod-sized data; check the plan flipped |
| Stale data and slow retries in prod | #11 replica lag | `pg_last_xact_replay_timestamp()` lag |
| One endpoint slow, thousands of identical queries | #7 ORM N+1 | Count the SQL the ORM emits; eager-load |

## Reproduce-prod-in-dev checklist

Pin this to the wall. Before you trust *any* "it's fast" claim, walk it:

- **Statistics parity.** Did you `ANALYZE` after loading? Do dev's per-column statistics targets match prod's? For pure plan debugging, did you clone `pg_statistic` + `reltuples`/`relpages` from prod?
- **Cost-constant parity.** Are `random_page_cost`, `effective_cache_size`, `work_mem`, and the parallelism GUCs set to prod's values in dev? (Free, highest-leverage step.)
- **Data parity.** Does dev have prod's *volume* and prod's *skew/distribution* on the columns you filter and join, not just the same schema with uniform random data?
- **Plan-mode parity.** Did you test the *generic* prepared plan (`force_generic_plan`), not just the ad-hoc custom plan, since the app uses prepared statements?
- **Cache parity.** Did you measure the *cold* path (restart or `pg_prewarm`-controlled), not just the warm steady-state? Did you read `Buffers`?
- **Sargability.** Does the plan actually show `Index Scan`/`Bitmap Index Scan`, not `Seq Scan`? No function, cast, or non-leading column killing the index?
- **LIMIT safety.** For every `ORDER BY ... LIMIT`, is there a composite index aligning the filter and sort, or should this be keyset pagination?
- **Round-trip count.** Did you count the SQL statements the ORM actually emits for the real-world payload size, with a tool (Bullet, query log) and not by eye?
- **Concurrency awareness.** Have you considered lock waits, `n_dead_tup` bloat, and connection-pool behavior that only exist under load — and is prod instrumented (`auto_explain`, `pg_stat_statements`, `log_lock_waits`) to capture them when they fire?
- **Topology awareness.** Are reads going to a replica that can lag, and do read-your-writes paths target the primary?

The unifying idea is simple enough to put on one line: **the query is a constant; the environment is the variable, so reproduce the environment, not the query.** Every time a query is fast in dev and slow in prod, one of the inputs in figure 1 diverged, the planner placed a bet it shouldn't have or the runtime hit a cost the planner never modeled, and the fix is to make dev tell the truth about prod *before* the code ships — not to debug it at 2 a.m. with a warm cache that hides the bug you're chasing.

## Further reading

- *Designing Data-Intensive Applications*, Martin Kleppmann — Chapter 1 (response-time percentiles and why averages mislead) and Chapter 3 (the cost-based optimizer and the statistics it consumes). The conceptual backbone of this entire article.
- [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) — the prerequisite skill: estimated vs actual rows, `Buffers`, `Sort Method`, loops.
- [join algorithms: nested loop, hash, and merge](/blog/software-development/database/join-algorithms-nested-loop-hash-merge) — the join strategies that the volume and skew mechanisms flip between.
- [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) — why concurrency produces bloat and how the two engines differ.
- [pganalyze: ORDER BY + LIMIT planner quirks](https://pganalyze.com/blog/5mins-postgres-planner-order-by-limit) and [random_page_cost tuning](https://pganalyze.com/blog/5mins-postgres-tuning-random-page-cost) — the two most actionable Postgres planner writeups.
- [Brent Ozar: parameter sniffing](https://www.brentozar.com/blitzcache/parameter-sniffing/) and [the elephant and the mouse](https://www.brentozar.com/archive/2013/06/the-elephant-and-the-mouse-or-parameter-sniffing-in-sql-server/) — the canonical treatment of sniffing in SQL Server.
- [PostgreSQL PREPARE docs](https://www.postgresql.org/docs/current/sql-prepare.html) and [plan_cache_mode explained](https://vladmihalcea.com/postgresql-plan-cache-mode/) — generic vs custom plans, the five-execution heuristic.
- [Stormatics: don't skip ANALYZE](https://stormatics.tech/blogs/dont-skip-analyze-a-real-world-postgresql-story) — a real 10-minutes-to-20-seconds statistics fix.
- [Bullet gem](https://github.com/flyerhzm/bullet) — detect N+1 and unused eager loading in Rails, wired into your test suite.

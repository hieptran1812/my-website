---
title: "The Database Scaling Decision Tree: What to Do, In What Order"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "A principal engineer's ordered ladder for scaling a database — measure, tune, scale up, cache, replicate, partition, and only then shard — and why most teams should stop long before the one-way door."
tags: ["database-scaling", "sharding", "read-replicas", "caching", "vertical-scaling", "query-optimization", "system-design", "distributed-systems", "polyglot-persistence", "slo"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 35
---

Every database scaling disaster I have been called into starts the same way. Someone looks at a dashboard, sees the primary at 80% CPU, and says the word "shard." Three months and two engineers later, the team has a half-migrated sharded cluster, a fan-out query that times out, an analytics pipeline that broke when the foreign keys disappeared, and a primary that is *still* at 80% CPU because nobody ever fixed the one query doing a sequential scan on a 40-million-row table.

Scaling a database is not a single decision. It is a **ladder of moves, ordered from cheapest-and-most-reversible to most-expensive-and-irreversible**. The skill is not knowing how to shard. The skill is knowing *which rung you are on* and *what the next rung is* — and having the discipline to climb one rung at a time instead of jumping off the top.

![The database scaling ladder, rung 0 through 8, colored by reversibility and cost](/imgs/blogs/the-database-scaling-decision-tree-1.webp)

The diagram above is the mental model for this entire series. The green rungs at the bottom — measure, tune, scale up — are reversible and nearly free; you can do them on a Tuesday afternoon and undo them on Wednesday. The amber rungs in the middle — cache, replicas, functional partitioning — buy a lot of capacity but add operational complexity and subtle correctness bugs. The red rungs at the top — sharding and globally distributed SQL — are **one-way doors**. Once you walk through, the cost of walking back is so high that, in practice, you never do.

The thesis of this post, and the spine of the series, is simple: **climb the ladder in order, and stop the moment your SLOs are green again.** Most systems run out of load somewhere around rung 4. The teams that get into trouble are the ones who skipped rungs 1 through 5 and leapt straight to rung 6.

## Why "just shard it" is the wrong instinct

The intuition behind reaching for sharding is understandable. Sharding is the move with the highest ceiling — it offers near-unlimited horizontal scale, the thing every infinite-growth narrative wants. So it feels like the "real" answer, and everything below it feels like a stopgap.

That intuition inverts the actual economics. Here is the mismatch between how teams think about scaling and how scaling actually behaves:

| What the team assumes | The naive view | The reality |
| --- | --- | --- |
| "We have a lot of data, so we need to shard." | Data volume drives the scaling strategy. | **Saturated resource** drives it. A 5 TB table on one box is fine; a 50 GB table getting hammered by an unindexed query is not. |
| "Sharding will fix our performance." | More machines = more throughput. | Sharding fixes *write* throughput and dataset size. It does nothing for a slow query, and it makes cross-shard reads *slower*. |
| "We'll shard now to avoid rework later." | Premature sharding is forward-thinking. | Premature sharding is the most expensive form of rework. You pay the full cost up front for capacity you don't use. |
| "Sharding is reversible if it doesn't work out." | It's just another deployment. | The shard key is effectively permanent. Un-sharding a live system is a migration project measured in quarters. |
| "Big companies shard, so we should too." | Copy what scale leaders do. | Big companies shard *one or two* hot tables after exhausting everything else, and staff entire teams to operate them. |

The single most common, most expensive mistake in this entire domain is **jumping to sharding before exhausting tune → scale-up → cache → replicas.** Every rung you skip is capacity you left on the table — and capacity left on the table at the cheap end of the ladder is capacity you are now buying at the expensive end, with interest.

> Sharding is not the summit you climb toward. It is the emergency exit you take when every cheaper door is locked.

The economics become obvious when you plot each rung by what it costs against what it returns. <a id="cost-vs-capability"></a>

![Cost vs capability gained per rung: the first three rungs return the most capability per dollar, the one-way door sits at the expensive end](/imgs/blogs/the-database-scaling-decision-tree-4.webp)

The first three rungs sit in the cheap, high-return corner — tuning gives roughly a 10× improvement for essentially zero dollars, and a bigger box buys years for the price of a config change. As you move right and down, each rung costs more and returns less *marginal* capability per dollar, while irreversibility climbs in lockstep. The one-way door sits in the far corner: maximum cost, maximum irreversibility, and a capability ceiling that most systems never come close to needing. Climbing the ladder in order is just walking this curve from its cheapest, most-reversible corner outward, and stopping the moment you're fast enough.

Let me make the central point visceral with motion. Watch the ladder fill as load grows:

<figure class="blog-anim">
<svg viewBox="0 0 720 420" role="img" aria-label="As load grows from left to right, the scaling ladder fills one rung at a time: tune, scale up, cache, replicas, partition, and only at the far right does sharding light up" style="width:100%;height:auto;max-width:820px">
<title>The scaling ladder fills rung by rung as load grows</title>
<style>
.dl-rail{stroke:var(--border,#d1d5db);stroke-width:3}
.dl-rung{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.dl-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.dl-num{font:700 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dl-axis{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.dl-bar{fill:var(--accent,#6366f1);opacity:.16}
.dl-lit{fill:var(--accent,#6366f1);opacity:0}
.dl-door{fill:#e8590c;opacity:0}
@keyframes dl-grow{0%{width:0}90%,100%{width:660px}}
@keyframes dl-on1{0%,6%{opacity:0}12%,100%{opacity:.85}}
@keyframes dl-on2{0%,18%{opacity:0}24%,100%{opacity:.85}}
@keyframes dl-on3{0%,32%{opacity:0}38%,100%{opacity:.85}}
@keyframes dl-on4{0%,46%{opacity:0}52%,100%{opacity:.85}}
@keyframes dl-on5{0%,60%{opacity:0}66%,100%{opacity:.85}}
@keyframes dl-on6{0%,82%{opacity:0}90%,100%{opacity:.9}}
.dl-fill{animation:dl-grow 12s ease-in-out infinite alternate}
.dl-r1{animation:dl-on1 12s ease-in-out infinite alternate}
.dl-r2{animation:dl-on2 12s ease-in-out infinite alternate}
.dl-r3{animation:dl-on3 12s ease-in-out infinite alternate}
.dl-r4{animation:dl-on4 12s ease-in-out infinite alternate}
.dl-r5{animation:dl-on5 12s ease-in-out infinite alternate}
.dl-r6{animation:dl-on6 12s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.dl-fill{animation:none;width:660px}.dl-r1,.dl-r2,.dl-r3,.dl-r4,.dl-r5{animation:none;opacity:.85}.dl-r6{animation:none;opacity:.9}}
</style>
<rect class="dl-bar dl-fill" x="30" y="24" width="0" height="26" rx="6"/>
<text class="dl-axis" x="30" y="18">load grows</text>
<text class="dl-axis" x="600" y="18">&#8594; sharding territory</text>
<line class="dl-rail" x1="58" y1="372" x2="58" y2="78"/>
<rect class="dl-rung" x="74" y="332" width="300" height="38" rx="7"/>
<rect class="dl-lit dl-r1" x="74" y="332" width="300" height="38" rx="7"/>
<text class="dl-num" x="58" y="356">1</text>
<text class="dl-lbl" x="92" y="356">tune queries + indexes (10x / $0)</text>
<rect class="dl-rung" x="74" y="284" width="300" height="38" rx="7"/>
<rect class="dl-lit dl-r2" x="74" y="284" width="300" height="38" rx="7"/>
<text class="dl-num" x="58" y="308">2</text>
<text class="dl-lbl" x="92" y="308">vertical scale up (buys years)</text>
<rect class="dl-rung" x="74" y="236" width="300" height="38" rx="7"/>
<rect class="dl-lit dl-r3" x="74" y="236" width="300" height="38" rx="7"/>
<text class="dl-num" x="58" y="260">3</text>
<text class="dl-lbl" x="92" y="260">cache reads (Redis / CDN)</text>
<rect class="dl-rung" x="74" y="188" width="300" height="38" rx="7"/>
<rect class="dl-lit dl-r4" x="74" y="188" width="300" height="38" rx="7"/>
<text class="dl-num" x="58" y="212">4</text>
<text class="dl-lbl" x="92" y="212">read replicas + split</text>
<rect class="dl-rung" x="74" y="140" width="300" height="38" rx="7"/>
<rect class="dl-lit dl-r5" x="74" y="140" width="300" height="38" rx="7"/>
<text class="dl-num" x="58" y="164">5</text>
<text class="dl-lbl" x="92" y="164">functional partition (by domain)</text>
<rect class="dl-rung" x="74" y="92" width="300" height="38" rx="7"/>
<rect class="dl-door dl-r6" x="74" y="92" width="300" height="38" rx="7"/>
<text class="dl-num" x="58" y="116">6</text>
<text class="dl-lbl" x="92" y="116">shard (one-way door)</text>
<text class="dl-axis" x="408" y="356">cheap, reversible</text>
<text class="dl-axis" x="408" y="116" fill="#e8590c">expensive, forever</text>
</svg>
<figcaption>As load grows left to right, the ladder fills one rung at a time. Most systems run out of load before the red top rung; sharding lights up only at the far right.</figcaption>
</figure>

The animation encodes the discipline: each rung lights up only when the load bar reaches it. You do not get to light up rung 6 while rung 1 is still dark. If your load bar stops at rung 3 — and for the overwhelming majority of systems it does — then rungs 4 through 8 are work you never have to do.

## How to know which rung you are on

Before any of the moves below, you need to know two things: **which resource is saturated**, and **whether you are read-bound or write-bound.** These two facts route you to the correct rung. Everything else — dataset size, table count, how "big" the system feels — is a distraction.

![The scaling decision flowchart: measure, tune, scale up, then branch on read-bound vs write-bound](/imgs/blogs/the-database-scaling-decision-tree-3.webp)

The flowchart above is the decision tree in its compact form. Notice the branch after vertical scale-up: **read pressure and write pressure leave the ladder at completely different rungs.** Read pressure is cheap to relieve — caching and replicas are reversible amber rungs. Write pressure is the expensive branch, because relieving it eventually forces functional partitioning and then the one-way door of sharding. This asymmetry is the single most useful thing to internalize: *if you are read-bound, you almost certainly never need to shard.*

Here is the heuristic as runnable code. It reads the same metrics you should already be collecting (rung 0) and tells you the next rung to consider:

```python
from dataclasses import dataclass

@dataclass
class DbMetrics:
    cpu_util: float            # 0..1, primary CPU utilization at p95
    p99_read_ms: float         # p99 latency of read queries
    p99_write_ms: float        # p99 latency of write queries
    read_qps: float
    write_qps: float
    slowest_query_share: float # fraction of total DB time spent in the top query
    buffer_cache_hit: float    # 0..1; fraction of reads served from RAM
    replica_lag_ms: float      # 0 if no replicas yet
    data_size_gb: float
    ram_gb: float              # RAM on the current box

def recommend_next_rung(m: DbMetrics) -> str:
    # Rung 1: a single query dominating DB time is almost always an index/plan bug.
    if m.slowest_query_share > 0.30:
        return "RUNG 1 — tune: one query owns >30% of DB time. EXPLAIN it first."

    # Rung 1 (working set spills to disk): low cache hit on a dataset that fits RAM.
    if m.buffer_cache_hit < 0.95 and m.data_size_gb < m.ram_gb:
        return "RUNG 1 — tune: working set fits RAM but cache hit <95%. Missing index?"

    # Rung 2: broadly saturated CPU with no single bad query and a fixable cliff.
    if m.cpu_util > 0.70 and m.data_size_gb < m.ram_gb * 4:
        return "RUNG 2 — scale up: CPU-bound, dataset still fits a bigger box."

    # Read-bound branch (cheap, reversible): rungs 3 and 4.
    read_heavy = m.read_qps > 5 * m.write_qps
    if read_heavy and m.p99_read_ms > 50:
        if m.replica_lag_ms == 0:
            return "RUNG 3 then 4 — cache hot reads, then add read replicas."
        return "RUNG 4 — add more read replicas; you are read-bound."

    # Write-bound branch (expensive, the one-way-door track): rungs 5 and 6.
    write_heavy = m.write_qps > m.read_qps
    if write_heavy and m.p99_write_ms > 50 and m.cpu_util > 0.70:
        return ("RUNG 5 — functional partition by domain; shard (RUNG 6) ONLY "
                "if a single domain's writes still saturate one box.")

    return "GREEN — SLOs are met. Do nothing. Stop climbing."

# Example: a read-heavy app with one runaway query.
m = DbMetrics(cpu_util=0.82, p99_read_ms=140, p99_write_ms=12,
              read_qps=9000, write_qps=400, slowest_query_share=0.41,
              buffer_cache_hit=0.88, replica_lag_ms=0,
              data_size_gb=180, ram_gb=256)
print(recommend_next_rung(m))
# -> RUNG 1 — tune: one query owns >30% of DB time. EXPLAIN it first.
```

This is deliberately blunt — it is a triage tool, not an autopilot. But it encodes the ordering. Notice that no input named `data_size_gb` *alone* ever routes you to sharding; sharding only appears as a conditional escalation off the write-bound branch, after functional partitioning has already failed. That ordering is the whole point.

## 0. Measure and set SLOs — you cannot scale what you cannot see

**The senior rule: instrument before you intervene. A scaling decision made without p99 latency, QPS split, and the top-query breakdown is a guess.**

Rung 0 is not optional and it is not "infrastructure work for later." It is the rung that tells you which other rung to climb. Skip it and every decision downstream is folklore.

You need, at minimum, four signals:

```sql
-- 1. The query breakdown: who is actually burning DB time?
-- (PostgreSQL, requires the pg_stat_statements extension)
SELECT
  substring(query, 1, 60)            AS query,
  calls,
  round(total_exec_time::numeric, 0) AS total_ms,
  round(mean_exec_time::numeric, 2)  AS mean_ms,
  round(100 * total_exec_time / sum(total_exec_time) OVER (), 1) AS pct_of_total
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

If the top row of that query owns 30–50% of total execution time — and it very often does — you have found your work, and it is on rung 1, not rung 6. I have walked into "we need to shard" emergencies where a single missing index on a `WHERE tenant_id = ?` clause was responsible for 60% of database load. The fix was one `CREATE INDEX CONCURRENTLY` and a coffee.

Set **SLOs**, not vibes. "The database feels slow" is unactionable. "p99 read latency must stay under 50 ms at 10k QPS, replication lag under 1 second" is a target you can measure against, alert on, and *stop at*. The stopping condition is as important as the target: without it, teams keep climbing the ladder past the point where their users are already happy, burning quarters of engineering time scaling a system that was already fast enough.

### Second-order optimization: measure the *shape* of load, not just its size

The single most useful derived metric is the **read/write ratio**, because it tells you which branch of the decision tree you are on before you ever feel pressure. A 50:1 read:write ratio means caching and replicas will carry you almost indefinitely. A 2:1 ratio with high write QPS means you are on the expensive branch and should be thinking about domain boundaries *now*, while it is still cheap to draw them. Most OLTP web apps are 10:1 or more read-heavy — which is exactly why most of them never need to shard.

## 1. Tune queries, indexes, and schema — often 10× for near-zero cost

**The senior rule: the cheapest capacity you will ever buy is the load you stop generating. Fix the query before you buy the hardware.**

Rung 1 routinely delivers an order-of-magnitude improvement for the cost of an afternoon, and it is fully reversible — drop the index, revert the migration, and you are back where you started. There is no other rung with this risk-adjusted return.

The big four, in order of how often they are the culprit:

1. **Missing or wrong indexes.** A query filtering on `tenant_id` and `created_at` against a table with no composite index is doing a sequential scan that gets linearly slower as the table grows. `EXPLAIN (ANALYZE, BUFFERS)` shows it instantly.
2. **N+1 query patterns.** The application loads 100 orders, then issues 100 separate queries for each order's line items. The database sees 101 round trips where one join would do. ORMs make this nearly invisible — and nearly universal.
3. **Over-fetching.** `SELECT *` pulling 40 columns and three TOAST'd JSON blobs when the endpoint renders three fields. The wire and the buffer cache both pay for the 37 columns nobody read.
4. **Bad schema for the access pattern.** A wide table that everything reads from, when the hot path only ever touches five columns. Vertical splitting of *columns* (not domains) keeps the hot row narrow.

Here is the N+1 fix, because it is the one I see most and the one ORMs hide best:

```python
# BEFORE — N+1: 1 query for orders, then 1 query per order. 101 round trips.
orders = session.query(Order).filter(Order.user_id == uid).all()
for o in orders:
    o.line_items   # lazy load fires a SELECT per order -> 100 extra queries

# AFTER — 1 query, eager join. One round trip.
from sqlalchemy.orm import selectinload
orders = (session.query(Order)
          .filter(Order.user_id == uid)
          .options(selectinload(Order.line_items))   # batched IN (...) load
          .all())
# 100 orders: 101 queries -> 2 queries. p99 on this endpoint dropped 18x
# in one production app, with zero infrastructure change.
```

And the index that ends most "we need to shard" conversations before they start:

```sql
-- The query: WHERE tenant_id = $1 AND status = 'open' ORDER BY created_at DESC
-- A composite index on (tenant_id, status, created_at DESC) turns a
-- 40M-row sequential scan into a tight index range scan + no sort.
CREATE INDEX CONCURRENTLY idx_orders_tenant_open
  ON orders (tenant_id, status, created_at DESC)
  WHERE status = 'open';   -- partial index: only the hot rows are indexed
```

`CONCURRENTLY` matters: it builds the index without taking a long lock, so you can do this on a live primary during business hours.

### Second-order optimization: tune the *plan*, not just the index

Adding an index is the obvious move; the non-obvious one is making sure the planner *uses* it. Stale statistics (`ANALYZE` not run after a bulk load), a parameter-sniffing surprise, or a query written so the index isn't sargable (`WHERE date_trunc('day', created_at) = ...` instead of a range predicate) will all leave a perfectly good index ignored. The reason rung 1 sometimes "doesn't work" is almost never that the database can't go faster — it is that the query is written in a way the optimizer can't help.

## 2. Vertical scale-up — buys years, reversible, with a cost cliff at the top

**The senior rule: a bigger box is the highest-leverage move you can make without changing a single line of code. Buy it before you build anything.**

Vertical scaling — moving to a machine with more CPU, more RAM, faster NVMe — is boring, which is exactly why it is underrated. It requires no application changes, no new failure modes, no consistency reasoning. You resize the instance, fail over, and you are done. And it is reversible: if the bigger box was a mistake, you resize back down.

The leverage is enormous and people forget the actual numbers. A modern single database server can have **hundreds of CPU cores and multiple terabytes of RAM.** When your entire working set fits in RAM, the disk effectively disappears from the latency budget. Companies have run *enormous* workloads on a single vertically-scaled primary for years — Stack Overflow famously served its entire Q&A traffic for a long time from a small number of very large SQL Server boxes, not a sharded fleet.

```bash
# AWS RDS: a one-line vertical scale-up. No app change, brief failover.
aws rds modify-db-instance \
  --db-instance-identifier prod-primary \
  --db-instance-class db.r6g.16xlarge \   # 64 vCPU, 512 GiB RAM
  --apply-immediately

# The Multi-AZ standby is promoted, the old primary is resized, ~60-120s of
# failover. You just bought a 4x jump in CPU and RAM for a config change.
```

The cliff: vertical scaling has a hard ceiling, and the price curve near the top is brutal. The jump from a large instance to the largest instance can triple the hourly cost for a 1.5× capacity gain. That is the "cost cliff at the top" the ladder figure marks in green-with-a-warning: the rung is reversible and code-free, but the last 20% of headroom is where the dollars stop being linear. When you are paying cliff prices, that is the market telling you to climb to rung 3.

### Second-order optimization: separate the resources that are actually scarce

"Scale up" is not one knob. If you are IOPS-bound, more vCPUs do nothing — you need faster storage or more provisioned IOPS. If you are connection-bound (thousands of app processes each holding a connection), the fix is a **connection pooler** like PgBouncer in front of the primary, not a bigger box. Diagnosing *which* resource is scarce (rung 0's job) keeps you from buying 64 cores to solve a storage problem.

## 3. Cache reads — offload read pressure, at the cost of invalidation

**The senior rule: caching trades a hard problem (database load) for a famously hard problem (invalidation). Take the trade only when reads dominate and staleness is tolerable.**

Once tuning and a bigger box are exhausted and you are read-bound, the next rung offloads reads from the database entirely. A cache — an application-level memoization, a [Redis](/blog/software-development/database/redis-applications-and-optimization) layer, or a CDN for cacheable HTTP responses — absorbs the reads that would otherwise hit the primary. For a read-heavy workload this can remove the *majority* of database load.

The classic pattern is cache-aside:

```python
import json, redis
r = redis.Redis()

def get_product(product_id: int):
    key = f"product:{product_id}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)            # cache hit: DB untouched

    row = db.fetch_one(                       # cache miss: hit the DB
        "SELECT id, name, price_cents FROM products WHERE id = %s",
        (product_id,),
    )
    # TTL bounds staleness AND bounds the blast radius of a bad invalidation.
    r.set(key, json.dumps(row), ex=300)       # expire in 5 minutes
    return row

def update_price(product_id: int, price_cents: int):
    db.execute("UPDATE products SET price_cents=%s WHERE id=%s",
               (price_cents, product_id))
    r.delete(f"product:{product_id}")         # invalidate on write
```

The win is real and large. The cost is that you now own a second copy of the truth, and keeping two copies consistent is the source of an entire genre of bugs: stale prices, deleted records that reappear, the thundering-herd stampede when a hot key expires and a thousand requests miss simultaneously. Caching is *mostly* reversible — you can turn it off and fall back to the database — but the application code that reads through the cache is harder to unwind than an index.

### Second-order optimization: a short TTL is worth more than a clever invalidation scheme

The most robust invalidation strategy I know is a boring one: **a short TTL plus best-effort delete-on-write.** Perfect event-driven invalidation is seductive and brittle — one missed event and you serve a stale value forever. A 60-second TTL caps the damage of *any* invalidation miss at 60 seconds, which is acceptable for the vast majority of read-heavy data. Reach for sophisticated invalidation only on the narrow set of keys where 60 seconds of staleness is genuinely unacceptable.

## 4. Read replicas and read/write split — scale reads N-way, inherit replication lag

**The senior rule: replicas are the right answer to read pressure that caching can't absorb — but every replica read is, by construction, a possibly-stale read.**

When cacheable reads are cached and you are *still* read-bound — typically because the reads are too varied or too fresh to cache — you add read replicas. The primary takes all writes; the replicas take reads. A query router (in the app, in a proxy like ProxySQL, or via the framework's multi-database support) sends each query to the right place.

![Read/write split topology: app routes writes to the primary, which replicates to read replicas serving the read pool, with bounded staleness](/imgs/blogs/the-database-scaling-decision-tree-5.webp)

The topology above is the whole pattern. Writes flow to the primary; the primary streams its changes to the replicas; the read pool load-balances reads across them. You can add replicas almost linearly to scale read throughput — three replicas roughly triple read capacity — and it is reversible: drop a replica and the primary is unaffected.

The catch is the red box on the right: **replication lag.** A replica is always at least slightly behind the primary. Usually that lag is milliseconds; under write bursts or long-running queries on the replica, it can spike to seconds. This produces the most common and most confusing bug class in this rung: **read-your-own-writes violations.** A user updates their profile (write → primary), the page reloads (read → replica), the replica hasn't caught up, and the user sees their *old* profile and concludes the save failed.

```python
# A read/write router that pins recently-writing sessions to the primary,
# so users read their own writes despite replica lag.
import time, random

REPLICAS = ["replica-1", "replica-2", "replica-3"]
WRITE_STICKINESS_SEC = 2.0   # > worst-case p99 replica lag

def route(session, sql: str) -> str:
    is_write = sql.lstrip()[:6].upper() in ("INSERT", "UPDATE", "DELETE")
    if is_write:
        session["last_write_ts"] = time.monotonic()
        return "primary"

    # Read: if this session wrote recently, the replica may be behind ->
    # send the read to the primary to guarantee read-your-own-writes.
    last = session.get("last_write_ts", 0.0)
    if time.monotonic() - last < WRITE_STICKINESS_SEC:
        return "primary"

    return random.choice(REPLICAS)   # safe to serve from a replica
```

This "stick to the primary for N seconds after a write" pattern resolves the overwhelming majority of stale-read complaints for a few percent of reads landing on the primary. The deep version of this — bounded staleness, monotonic reads, and the formal consistency models behind them — is exactly what [CAP, PACELC, and the consistency spectrum](/blog/software-development/database/cap-theorem-and-pacelc) are about; replicas are where those abstractions stop being theory and start generating support tickets.

### Second-order optimization: replicas do not help writes, and can hurt them

It is worth saying plainly because teams forget it under pressure: **read replicas do nothing for write throughput.** Every write still goes to the single primary. Worse, each replica adds replication load to that primary (it has to ship its WAL to every replica). So if your problem is write QPS, replicas are not your rung — you are on the write-bound branch, headed for functional partitioning. Adding replicas to a write-bound primary makes it *slower*.

## 5. Functional partitioning — scale writes by domain, break cross-domain joins

**The senior rule: split the database along the seams your domains already have. The first time you can't run a join, you have found a real boundary — or drawn the wrong one.**

When you are genuinely write-bound — one primary can't keep up with write QPS even after tuning and scaling up — the first move is *not* sharding. It is **functional partitioning**: splitting the single database into several databases, each owning a domain. The `users` tables go to a users database, `orders` to an orders database, `inventory` to an inventory database. Each gets its own primary, so writes now scale by domain — three domains, three independent write capacities.

![Functional partitioning: one database with cross-domain joins becomes per-domain databases where a join becomes a network call](/imgs/blogs/the-database-scaling-decision-tree-6.webp)

The before/after above shows both the win and the price. On the left, everything is in one schema: a `JOIN users × orders` is a single transactional query, but every write hits one primary. On the right, writes scale per domain — but the join is gone. A query that needs user data and order data together must now make **two calls and stitch the results in the application**, and there is no longer a database transaction spanning both. This is the same boundary that defines a [microservice-per-database architecture](/blog/software-development/microservices/database-per-service); functional partitioning is, in effect, where the data tier learns the lesson the service tier already learned.

```python
# After functional partitioning, a "user with their recent orders" view
# is an application-level join across two databases.
def user_dashboard(user_id: int):
    user = users_db.fetch_one(
        "SELECT id, name, email FROM users WHERE id = %s", (user_id,))
    # No SQL JOIN possible across databases -> fetch and stitch in the app.
    orders = orders_db.fetch_all(
        "SELECT id, total_cents, created_at FROM orders "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 10", (user_id,))
    return {"user": user, "recent_orders": orders}   # joined in memory
```

Functional partitioning is **hard to reverse** — once the application is written against three databases and three connection pools, merging them back is real work — but it is *less* irreversible than sharding, because the boundaries follow domains that already exist in your code. You are formalizing seams, not inventing them.

### Second-order optimization: the boundary you can't draw is the one to worry about

The diagnostic value of functional partitioning is that it *forces* you to confront cross-domain coupling. If splitting `orders` from `inventory` means you can no longer enforce "don't sell stock you don't have" in a single transaction, that coupling was real and you now have to handle it with a saga, an outbox, or eventual reconciliation. Discovering this at rung 5 is far cheaper than discovering it at rung 6, when the same coupling spans shards of the *same* table and there is no clean domain seam to cut along.

## 6. Horizontal sharding — near-unlimited scale, and the one-way door

**The senior rule: choose the shard key as if you can never change it, because you effectively can't. Every query without that key fans out to every shard.**

Sharding splits a *single logical table* across many physical databases by a **shard key** — `hash(user_id) % N`, a range of IDs, a geographic region. It is the only rung that scales writes to a single table beyond what one machine can do, and its ceiling is effectively unlimited. It is also the rung where careers and quarters go to die.

![Sharding: a shard router sends key-bearing queries to one shard, while key-less queries scatter-gather across all shards](/imgs/blogs/the-database-scaling-decision-tree-7.webp)

The figure shows the two faces of sharding. A query that *includes the shard key* — "get user 5's profile" — routes to exactly one shard and is fast. A query that *does not* include the shard key — "find all users who signed up yesterday" — has no way to know which shard holds the answer, so it must **scatter-gather**: fan out to every shard, wait for the slowest, and merge. As you add shards to gain capacity, every scatter-gather query gets *slower*, because there are more shards to wait on.

```python
# A minimal shard router. The shard key decision here is permanent:
# resharding (changing N or the key) means rewriting every row's location.
import hashlib

SHARDS = ["shard-0", "shard-1", "shard-2", "shard-3"]

def shard_for(user_id: int) -> str:
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    return SHARDS[h % len(SHARDS)]

def get_user(user_id: int):
    # Has the shard key -> single-shard query, fast.
    return connect(shard_for(user_id)).fetch_one(
        "SELECT * FROM users WHERE id = %s", (user_id,))

def find_recent_signups(since):
    # NO shard key -> scatter-gather across ALL shards, merge in app.
    rows = []
    for s in SHARDS:                       # fan out to every shard
        rows += connect(s).fetch_all(
            "SELECT id, email FROM users WHERE created_at > %s", (since,))
    return sorted(rows, key=lambda r: r["id"])   # gather + merge
```

Why is it a one-way door? Three reasons, all permanent:

1. **The shard key is forever.** Change it and you must physically relocate every row to a new shard. There is no `ALTER` for "actually, shard by `tenant_id` not `user_id`."
2. **Cross-shard transactions die.** A single ACID transaction can't span shards without distributed-transaction machinery that most teams never build correctly. Foreign keys across shards don't exist.
3. **Resharding is a migration, not a config change.** Going from 4 shards to 8 means re-splitting and re-homing data live, with dual-writes and backfills, for weeks.

The reason this whole series exists is to keep you from arriving at this rung by accident. When you *do* need it, [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) is the deep treatment of shard-key selection, routing, and resharding — the mechanics you must get right because you only get one shot.

### Second-order optimization: pick the key that matches your *highest-volume* query, not your biggest table

The novice instinct is to shard by whatever table is largest. The correct instinct is to shard by the key carried by your most frequent query, so the hot path is single-shard. If 95% of your queries are "operations for one user," shard by `user_id` and 95% of traffic stays single-shard; the 5% analytics queries can scatter-gather or, better, run against a separate analytics store (rung 7). Sharding by the wrong key turns your hot path into a scatter-gather and is the most expensive mistake on the most expensive rung.

## 7. Polyglot persistence — the right store per workload

**The senior rule: stop forcing one engine to be good at everything. Keep a relational system of record, and route each workload to the store built for it.**

By the time you're considering sharding, you've probably noticed that a single relational engine is being asked to do five different jobs — transactions, full-text search, time-series metrics, session storage, and analytics — and is mediocre at four of them. Polyglot persistence is the move that routes each workload to a store designed for it, which *removes* load from your relational primary and often makes the question of sharding it moot.

![Polyglot persistence matrix: which store fits transactions, search, time-series, KV, and analytics workloads](/imgs/blogs/the-database-scaling-decision-tree-8.webp)

The matrix above is the routing table. No single column is "best" down the whole table — that is the entire point. Postgres is the right system of record for transactions and is "ok" at search and analytics; Elasticsearch owns full-text search; ClickHouse owns time-series and OLAP; Redis owns session and KV. Pulling full-text search off the relational primary and onto a search engine can drop primary load enough that the sharding pressure evaporates.

```python
# Polyglot routing: each access pattern goes to its best-fit store.
# Postgres remains the system of record; the others are derived/specialized.
def handle(request):
    match request.kind:
        case "checkout":                      # ACID transaction
            return postgres.transaction(do_checkout, request)
        case "product_search":                # inverted index, relevance
            return elasticsearch.search(request.query)
        case "dashboard_metrics":             # columnar, time-bucketed
            return clickhouse.query(request.range)
        case "session_lookup":                # microsecond KV
            return redis.get(request.session_id)
```

Polyglot is **additive, not a one-way door** — each store is a layer you can add or remove — but every store you add is another system to operate, back up, monitor, and reason about during incidents. The cost is operational surface area, paid by your on-call rotation.

### Second-order optimization: derive, don't fork, the system of record

The trap in polyglot persistence is letting two stores both claim to be the source of truth — Postgres *and* Elasticsearch both "owning" product data, drifting apart over time. The discipline is to keep exactly one **system of record** (almost always the relational store) and treat every specialized store as a **derived, rebuildable projection** fed by change-data-capture or an indexing pipeline. If a derived store is corrupted or lost, you rebuild it from the system of record. If you can't rebuild it, it has quietly become a second source of truth, and you've inherited a distributed-consistency problem you didn't sign up for.

## 8. Globally distributed SQL — multi-region ACID, and the latency tax

**The senior rule: a globally consistent write is a multi-continent round trip. Reach for distributed SQL only when you need both global scale and strong consistency — and are willing to pay milliseconds per write to get them.**

The top rung is the systems that promise it all: horizontal scale, strong consistency, *and* multi-region operation. Spanner, CockroachDB, YugabyteDB, TiDB — they shard automatically, replicate across regions, and present a single SQL interface with ACID semantics. From the application's perspective, sharding largely disappears as a concern.

The price is physics. **A strongly-consistent write that must achieve quorum across regions pays the speed-of-light round-trip latency between those regions.** A consensus write spanning, say, North America and Europe can't commit faster than the time light takes to make that trip and back — tens of milliseconds, floor, no matter how good the engineering. This is the latency tax, and it is not negotiable.

```sql
-- CockroachDB: pin a table's data to a region to dodge the tax for
-- locality-bound data. The schema speaks geography directly.
ALTER TABLE users SET LOCALITY REGIONAL BY ROW;

-- Rows tagged with a home region commit at local-quorum latency.
-- A row read/written within its home region avoids cross-region consensus;
-- only truly global tables pay the full multi-region round trip.
INSERT INTO users (id, region, email)
VALUES (501, 'us-east1', 'a@example.com');   -- commits at us-east1 latency
```

Distributed SQL is the **hardest rung to walk back** — your entire data model now assumes a system that a single-node Postgres can't replicate — which is why it sits at the very top. For a genuinely global, write-heavy, must-be-consistent workload, it is the right and sometimes only answer. For the system that is read-heavy in one region, it is a five-figure-per-month solution to a problem a bigger box and a cache would have solved.

### Second-order optimization: most "global" apps are regional with a global veneer

Before paying the latency tax, ask whether your data is actually global or merely *accessed* globally. A user's data usually has a home region; orders belong to a region; the genuinely global, must-be-strongly-consistent dataset is often small (a username uniqueness table, a billing ledger). The cost-effective architecture is frequently regional primaries with a *small* globally-consistent core, not a uniformly global cluster paying cross-region latency on every write.

## Case studies from production

### 1. The "we need to shard" that was a missing index

A B2B SaaS company opened an emergency: the primary was pinned at 90% CPU, dashboards were timing out, and an architect had already drafted a sharding plan. The symptom screamed scale. The wrong first hypothesis — shared by the whole room — was that the dataset had simply outgrown one machine. The actual root cause, found in fifteen minutes with `pg_stat_statements`, was a single tenant-scoped report query doing a sequential scan on a 60-million-row events table because the index was on `(created_at)` but the query filtered on `(tenant_id, created_at)`. One `CREATE INDEX CONCURRENTLY` on the composite key dropped that query from 8 seconds to 40 milliseconds, and primary CPU fell to 30%. The lesson: the loudest scaling emergencies are disproportionately rung-1 problems wearing a rung-6 costume. Always run the query breakdown before you draw an architecture diagram.

### 2. Stack Overflow's vertically-scaled defiance

For years, Stack Overflow served one of the busiest sites on the internet from a strikingly small number of very large SQL Server boxes rather than a sharded fleet. They leaned hard into rungs 1, 2, and 3 — obsessive query tuning, enormous amounts of RAM so the working set lived in memory, and aggressive caching — and consequently never needed the expensive rungs. The wrong assumption their architecture quietly refutes is "high traffic requires horizontal sharding." The root reality is that a read-heavy workload, tuned and cached and given a big enough box, scales astonishingly far vertically. The lesson: vertical scaling plus caching has a far higher ceiling than the "scale-out or die" narrative admits, and the operational simplicity of a single primary is worth defending.

### 3. The replica lag that looked like data loss

A consumer app started getting reports that profile edits "weren't saving." Support couldn't reproduce it; the data was always correct in the database. The wrong first hypothesis was a write bug — a dropped UPDATE, a transaction rollback. The actual root cause was rung 4: the team had recently added read replicas and routed all reads to them, including the read immediately following a profile write. Under load, replica lag spiked to two or three seconds, so a user who saved and instantly reloaded read the *pre-write* state from a lagging replica and concluded the save had failed. The fix was the write-stickiness router shown earlier: pin reads to the primary for a few seconds after a write. The lesson: replication lag doesn't corrupt data, it corrupts *perception of time*, and read-your-own-writes is the bug it manifests as.

### 4. Notion's move to sharded Postgres — done right, at the right time

Notion ran on a single Postgres instance for years, climbing the cheaper rungs, before the block table — the heart of their data model — grew to the point where a single primary genuinely could not keep up with write volume and the table itself was unwieldy. Only then did they shard, and notably they sharded *by workspace*, matching the shard key to the access pattern (almost every query is scoped to one workspace), so the hot path stayed single-shard. The wrong way to read this story is "Notion sharded, so sharding is normal." The right way is "Notion exhausted the cheaper rungs first, sharded exactly one thing, chose the key to match their dominant query, and staffed it." The lesson: when you do reach rung 6, reach it deliberately, for a specific saturated resource, with a key chosen for your hottest query — not as a default.

### 5. The cache stampede that took down the database it was protecting

An e-commerce site cached its hot product pages in Redis with a one-hour TTL. During a flash sale, the single hottest product's cache entry expired at the worst possible moment. With the key gone, every concurrent request missed simultaneously and slammed the database with thousands of identical queries in the same second — a thundering herd — and the primary, which the cache existed to protect, fell over. The wrong hypothesis was that the database had simply been overwhelmed by sale traffic. The actual root cause was synchronized cache expiry plus no stampede protection. The fix was a per-key mutex (only one request recomputes a missed key; the rest wait briefly for the result) plus jittered TTLs so hot keys don't all expire on the same second. The lesson: caching (rung 3) adds its own failure modes, and the nastiest of them target the very database the cache was meant to shield.

### 6. The premature shard that became a multi-year tax

A startup, anticipating hypergrowth that hadn't arrived yet, sharded their main database by `user_id` at launch "to avoid rework later." Growth came, but slower and differently than projected: their most important queries turned out to be analytics across all users, every one of which became a scatter-gather across all shards. They had paid the full operational cost of sharding — the routing layer, the loss of cross-shard joins and transactions, the resharding fear — to get *worse* performance on their actual hot path. Un-sharding back to a single (vertically scaled) primary took the better part of a year. The wrong assumption was "sharding early is cheaper than sharding late." The actual reality is that sharding before you know your true query patterns means choosing the shard key blind, and the shard key is forever. The lesson: rung 6 is a one-way door precisely because the decision it commits you to — the key — can't be known until you've watched real production traffic, which is exactly when you no longer need to rush.

### 7. Figma's functional partitioning before sharding

Figma scaled their Postgres setup for a long time, and a key early move was **functional partitioning** — pulling distinct domains and high-volume tables onto their own databases — before they undertook the much larger project of horizontally sharding their highest-volume tables. By splitting along domain seams first (rung 5), they relieved a great deal of write pressure with a far less irreversible move, and crucially learned where their cross-domain coupling actually lived before committing to shard keys. The wrong framing is "they eventually sharded, so partitioning was just a delay." The right framing is "partitioning bought years of runway *and* surfaced the boundaries that made the later sharding tractable." The lesson: rung 5 is not merely a cheaper rung 6 — it is the rung that teaches you what you need to know to do rung 6 correctly, if you ever get there.

### 8. The analytics query that polyglot persistence rescued

A logistics platform's Postgres primary was buckling, and the team assumed they were out of vertical headroom and headed for sharding. Rung-0 measurement told a different story: 70% of database load came from a handful of heavy analytical aggregations — dashboards scanning months of shipment events — running against the same OLTP primary that served the transactional hot path. The fix was rung 7, not rung 6: stream the events into a columnar analytics store (ClickHouse), point the dashboards there, and leave the OLTP primary to do transactions. Primary load fell by more than half, the dashboards got *faster* (columnar storage is built for exactly those scans), and the sharding project was cancelled. The lesson: a relational primary doing both OLTP and OLAP is often "out of headroom" only because it's doing two jobs; routing the analytical job to a store built for it (rung 7) can reclaim more capacity than any amount of scaling the wrong engine.

## When to climb to the next rung — and when to stop

### Climb to the next rung when

- **Your SLOs are red and the current rung is genuinely exhausted.** Not "feels slow" — measured p99 over target, with the current rung's optimizations actually applied (the index exists, the cache is warm, the box is the right size).
- **You have identified the specific saturated resource.** CPU, IOPS, connections, write QPS, or dataset size — named, measured, and matched to the rung that relieves it.
- **You know whether you're read-bound or write-bound.** This single fact tells you whether the cheap branch (cache, replicas) or the expensive branch (partition, shard) is the right direction.
- **The cheaper rung's cost has gone non-linear.** When vertical scaling hits the price cliff, or caching can't absorb the read variety, that's the market signal to climb.

### Stop climbing when

- **Your SLOs are green.** The single most under-used move in database scaling is *doing nothing more*. If reads are fast and writes are fast and lag is bounded, you are done. Capacity you don't need is complexity you pay for forever.
- **You're tempted to skip a rung.** Skipping straight to sharding because it has the highest ceiling is the canonical mistake. The ceiling is irrelevant if a cheaper rung clears your actual load.
- **You're about to walk through a one-way door for a reversible problem.** If the pressure is read pressure, sharding is the wrong tool no matter how saturated the primary feels — you're on the wrong branch.
- **The work is being driven by a resume, a conference talk, or a hypothetical 100× growth that hasn't arrived.** Build for the next 10×, which the cheap rungs almost always cover; do not architect for a 100× that may never come and whose shape you can't yet know.

The rest of this series is a rung-by-rung tour of this ladder: the caching patterns and pitfalls of rung 3, the replication and consistency machinery of rung 4, the shard-key selection and resharding mechanics of rung 6, the specialized-store deep dives of rung 7, and the big-tech case studies that show these moves under real load. The next post — [when one database is not enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) — picks up exactly where rung 2 ends, at the moment a single primary stops being enough and you have to decide which branch of the tree to take.

The whole series hangs off this one map. Keep it in your head, climb in order, and stop when you're green. Most of you will never reach the red rungs — and that is not a failure to scale. That is scaling done right.

## Further reading

- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — the deep treatment of rung 6: shard-key selection, routing, and resharding.
- [Redis applications and optimization](/blog/software-development/database/redis-applications-and-optimization) — patterns and failure modes for the caching rung.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the consistency models that replicas and distributed SQL make you confront.
- [When one database is not enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) — the next post in this series, on crossing from rung 2 to the read/write branch.

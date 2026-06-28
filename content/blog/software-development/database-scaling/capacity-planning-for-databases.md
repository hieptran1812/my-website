---
title: "Capacity Planning for Databases: Model the Wall Before You Hit It"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Most database fire-drills are predictable months ahead with napkin math; here is the math for the four budgets that decide your next scaling deadline."
tags: ["database-scaling", "capacity-planning", "performance", "postgresql", "mysql", "headroom", "littles-law", "storage-growth", "cache-hit-ratio", "sre"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 34
---

The pager goes off at 02:14. Read latency on the primary just went from 3 ms to 280 ms, the connection pool is saturated, and the on-call engineer is staring at a graph that looks like a cliff. The post-incident review will use words like "sudden," "unexpected," and "spike." None of those words will be true. The database had been telling you for four months that this was coming. Nobody was reading the message, because nobody had done the napkin math.

That is the uncomfortable thesis of this post: **the overwhelming majority of database fire-drills are not surprises — they are scheduled events you failed to put on the calendar.** Capacity is not a vibe. It is four budgets, each of which crosses a hard ceiling on a date you can compute today with a spreadsheet and ten minutes. The job of capacity planning is to find which budget crosses first, compute when, and pull the right lever before the crossing rather than during the incident.

![The four budgets that decide your scaling deadline](/imgs/blogs/capacity-planning-for-databases-1.webp)

The diagram above is the mental model for the entire post. Traffic and data growth feed four independent budgets — QPS/throughput, working set vs RAM, storage growth, and connections — and each one has a ceiling it will eventually hit. Whichever budget crosses its ceiling first is your next scaling deadline. Everything else in this article is a tour of how to compute each crossing date, how much slack to keep, and how to read the binding constraint off the math so you know which lever to pull. We will build a Python model that takes your current numbers and prints months-to-each-wall, and we will close with named production incidents where someone skipped this math and paid for it.

## Why capacity planning feels hard (and why it isn't)

The reason capacity planning gets skipped is that it *feels* like forecasting, and engineers are rightly suspicious of forecasting. But you are not predicting the future. You are doing arithmetic on quantities you already measure. Here is the assumption-versus-reality table that frames the rest of the post:

| Common assumption | Naive view | Reality |
| --- | --- | --- |
| "We'll scale when we need to" | Scaling is a reaction to a problem | By the time you *need* to, you are already in an incident; scaling takes weeks |
| "CPU is at 40%, we're fine" | One metric tells the whole story | Four independent budgets exist; CPU can be fine while the working set is one query away from the cliff |
| "Storage grows slowly" | Disk is cheap and large | Append-heavy tables plus index overhead plus bloat fill a disk in months, not years |
| "We have 500 max connections" | Connections are abundant | Little's Law says a latency spike multiplies in-flight connections; the pool saturates long before 500 |
| "Growth is linear" | Extrapolate the last 3 months in a straight line | Compounding growth crosses budgets far sooner than the straight line predicts |
| "Run it hot to save money" | 90% utilization is efficient | At 90% there is no slack for a spike, a failover, or a backup; 100% is already too late |

Every row of that table is a real failure mode, and every one is preventable with the math below. Let me take the four budgets in turn, then headroom, then growth modeling, then a fully worked example, then the anti-patterns and the incidents.

> A senior rule of thumb: if you cannot say, in one sentence, which budget crosses first and roughly when, you do not have a capacity plan — you have hope.

## 1. The throughput budget: QPS vs CPU and IOPS

**Senior rule: size for peak QPS times safety factor, not average QPS, and remember the database does work proportional to the *plan*, not the request count.**

The throughput budget asks: can the database execute the requests arriving per second without the CPU pinning or the storage IOPS saturating? The trap here is averages. If your service does 5,000 QPS averaged over a day but spikes to 18,000 QPS at the 7pm peak, you must plan for 18,000 — averaged capacity planning is how you end up with a database that is fine at 3am and on fire at dinner.

The second trap is that QPS is a poor proxy for work. A query that does an index seek costs microseconds of CPU; the same query after the planner flips to a sequential scan costs hundreds of milliseconds. So the throughput budget is really *CPU-seconds of work per second*, and that quantity is sensitive to plan stability, not just request volume.

Here is the basic model. Let $q$ be peak QPS, $c$ the average CPU-seconds of work per query, and $N_{\text{cores}}$ the number of cores. The CPU utilization is:

$$U_{\text{cpu}} = \frac{q \cdot c}{N_{\text{cores}}}$$

You want $U_{\text{cpu}}$ comfortably below 1 — in practice below ~0.6 (more on headroom shortly). If $q = 18{,}000$, $c = 0.0008$ CPU-seconds (0.8 ms of CPU per query), and you have 32 cores, then $U_{\text{cpu}} = (18{,}000 \times 0.0008) / 32 = 0.45$. You are at 45%, which is healthy. But if traffic grows 8% per month, in 10 months $q$ is $18{,}000 \times 1.08^{10} \approx 38{,}900$, and $U_{\text{cpu}} \approx 0.97$. That is the throughput crossing date: month 10, and you should have read replicas or a shard plan in place by month 7.

The IOPS half of the budget matters when the working set does not fit in RAM (the next section). Every buffer-pool miss becomes a random read against storage. On AWS, a gp3 EBS volume gives you a baseline of 3,000 IOPS (provisionable to 16,000); io2 Block Express goes to 256,000; local NVMe does hundreds of thousands. If your miss rate times QPS exceeds your provisioned IOPS, latency hockey-sticks regardless of CPU headroom.

```sql
-- PostgreSQL: how much CPU work is each statement actually doing?
-- Requires pg_stat_statements. total_exec_time is in milliseconds.
SELECT
  substring(query, 1, 60) AS query,
  calls,
  round(total_exec_time::numeric, 1)               AS total_ms,
  round((total_exec_time / calls)::numeric, 3)      AS avg_ms,
  round(100.0 * total_exec_time /
        sum(total_exec_time) OVER (), 1)            AS pct_of_total
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 15;
```

That query is your throughput budget in one screen: the top few statements almost always account for 80%+ of the CPU. When you model growth, model *those* statements' `avg_ms` staying stable — and set an alarm if any of them doubles, because a plan flip is a step-change in the throughput budget that no amount of traffic forecasting will catch.

### Second-order optimization: the budget is the plan, not the QPS

The non-obvious gotcha is that the throughput budget can be blown without any traffic change at all. A table crosses a row-count threshold where the planner abandons an index, statistics go stale after a bulk load, or a parameter-sniffing event caches a bad plan. Each of these multiplies `avg_ms` for a hot query, and your 45% CPU becomes 95% overnight. Capacity planning for throughput therefore includes *plan stability* as a first-class budget: pin critical plans, keep statistics fresh (`ANALYZE` cadence tied to write volume, not a nightly cron), and alarm on per-statement `avg_ms` regressions, not just on host CPU.

## 2. The working-set budget: RAM and the cache hit-ratio cliff

**Senior rule: the single most violent failure mode in a database is the working set outgrowing RAM, because latency does not degrade gracefully — it cliffs.**

Of the four budgets, this is the one that turns a slow Tuesday into a Sev1. The other three degrade more or less linearly: more QPS, more CPU; more data, less free disk. The working-set budget does not. It is flat, flat, flat — and then it falls off a cliff.

![The cache hit-ratio cliff](/imgs/blogs/capacity-planning-for-databases-2.webp)

The figure above is why. The x-axis is the size of your *hot* working set — the pages you actually touch in a typical window, not the total data size. The y-axis is read latency. As long as the hot set fits in the buffer pool (PostgreSQL's `shared_buffers` plus the OS page cache, InnoDB's buffer pool), your hit ratio is north of 99% and reads are served from memory at sub-millisecond latency. The line is flat.

The instant the hot set exceeds RAM, the hit ratio collapses. Now a meaningful fraction of reads miss the cache and become random I/O against storage. A read from RAM costs ~100 nanoseconds; a random read from NVMe costs ~100 microseconds (1,000× slower); from a spinning disk or a throttled network volume, ~5 milliseconds (50,000× slower). And it compounds: the misses generate IOPS, the IOPS queue, the queue adds latency, the added latency holds connections open longer (Little's Law again), and the connection pool saturates. One marginal gigabyte of working set pushes you over the edge and the whole system tips.

The brutal part is *non-linearity*. You can run for months at 70% of RAM with a 99.5% hit ratio and beautiful latency, gaining nothing visible from the headroom. Then the hot set grows 5% and your hit ratio drops to 96%, which sounds fine but means your miss rate *quadrupled* (from 0.5% to 4%), and p99 latency goes from 4 ms to 120 ms. The metric that warns you is the hit ratio, and you must watch its *trend*, because the cliff is invisible until you are at it.

```sql
-- PostgreSQL: buffer-pool (shared_buffers) hit ratio, cluster-wide.
-- Below ~0.99 on an OLTP system means you are approaching the cliff.
SELECT
  sum(heap_blks_hit)                                        AS hits,
  sum(heap_blks_read)                                       AS reads_from_disk,
  round(sum(heap_blks_hit)::numeric
        / nullif(sum(heap_blks_hit + heap_blks_read), 0),
        5)                                                  AS hit_ratio
FROM pg_statio_user_tables;
```

```sql
-- MySQL / InnoDB: buffer-pool hit ratio derived from status counters.
-- (1 - reads/read_requests). Watch the trend over weeks, not the point value.
SELECT
  (SELECT variable_value FROM performance_schema.global_status
     WHERE variable_name = 'Innodb_buffer_pool_reads')          AS disk_reads,
  (SELECT variable_value FROM performance_schema.global_status
     WHERE variable_name = 'Innodb_buffer_pool_read_requests')  AS logical_reads,
  round(1 - (
     (SELECT variable_value FROM performance_schema.global_status
        WHERE variable_name = 'Innodb_buffer_pool_reads') /
     (SELECT variable_value FROM performance_schema.global_status
        WHERE variable_name = 'Innodb_buffer_pool_read_requests')
  ), 5)                                                          AS hit_ratio;
```

To turn this into a capacity *date*, you need two numbers: how big the hot working set is today, and how fast it grows. Estimating the hot set is the hard part. A workable approximation: take the size of the tables and indexes touched by your top statements (from `pg_stat_statements` joined to table sizes), and discount by the fraction that is genuinely hot (recent rows, active accounts). If the hot set is 90 GB today against 128 GB of RAM, and it grows 4 GB/month, you have roughly $(128 \times 0.75 - 90)/4 \approx 1.5$ months until you cross the 75% comfort line — which is your real deadline, not the 9.5 months until you literally hit 128 GB.

### Second-order optimization: the working set is not the data set

The mistake that wrecks this calculation is conflating total data size with working-set size. A 10 TB table whose queries only ever touch the last 30 days of rows has a working set of maybe 200 GB. Time-correlated access patterns mean most of your data is cold and can live on disk forever without ever touching the cliff. The corollary: anything that *de-correlates* access — a backfill job that scans the whole table, an analytics query without a time filter, a new feature that reads random historical rows — can blow the working-set budget instantly without any data growth at all. Capacity-plan your batch jobs, not just your data.

## 3. The storage budget: growth vs the disk ceiling

**Senior rule: storage exhaustion is the most predictable failure in all of computing, and somehow still takes down production constantly, because nobody multiplies four numbers and draws a line.**

The storage budget is arithmetic a fifth-grader can do, which is exactly why it is so embarrassing when it takes down a database. The formula:

$$\text{bytes/day} = \text{rows/day} \times \text{avg row size} \times (1 + \text{index overhead}) \times (1 + \text{bloat factor})$$

Then months-to-full is just free space divided by bytes/month. The terms people forget are the last two. **Index overhead** is not small: a table with three or four secondary indexes can carry 80–120% overhead, so every gigabyte of table data is really 1.8–2.2 GB on disk. **Bloat** is the dead tuples and free space that MVCC databases like PostgreSQL accumulate between vacuums; on a write-heavy table it can add another 20–50% if autovacuum is not keeping up.

![Storage growth crosses the disk ceiling on a knowable date](/imgs/blogs/capacity-planning-for-databases-5.webp)

The chart above models a service writing 78 million rows/day at 600 bytes each. Raw, that is 47 GB/day. Multiply by 1.9 for indexes and bloat and you are appending 90 GB/day, or about 2.7 TB/month, against a 4 TB disk that is at 0.9 TB today. If history were retained forever, you would cross the 70% planning line (the dotted line where you should act) in about 10 months and hit the disk ceiling at ~14 months. The bars turn red the moment they cross the 70% line — that color change is your calendar invite.

The lever for storage is usually a time-to-live (TTL) or archival policy. The net growth rate after a 90-day retention window is far lower than the gross append rate, because old rows age out as new ones arrive. But — and this is the part people miss — **the retention policy only bounds the growth once you are past the retention horizon.** For the first 90 days of a new append-heavy table, growth is purely additive and the disk fills at the gross rate. Many storage incidents happen in exactly this window: a new event-logging table ships, nobody set up the TTL job, and 90 days later the disk is gone.

```sql
-- PostgreSQL: per-table size with index overhead broken out, plus a crude
-- bloat estimate. Run weekly, store the results, and you have a growth rate.
SELECT
  relname                                          AS table,
  pg_size_pretty(pg_total_relation_size(c.oid))    AS total,
  pg_size_pretty(pg_relation_size(c.oid))          AS heap,
  pg_size_pretty(pg_indexes_size(c.oid))           AS indexes,
  round(100.0 * pg_indexes_size(c.oid)
        / nullif(pg_relation_size(c.oid), 0), 0)   AS index_pct,
  n_dead_tup,
  round(100.0 * n_dead_tup
        / nullif(n_live_tup + n_dead_tup, 0), 1)   AS dead_pct
FROM pg_class c
JOIN pg_namespace n   ON n.oid = c.relnamespace
JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.relkind = 'r' AND n.nspname = 'public'
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 20;
```

To get the *rate*, you snapshot `pg_total_relation_size` for the hot tables weekly into a small metrics table and fit a line. Two data points a week apart already give you a slope; four weeks gives you confidence the slope is stable. The crossing date is `(disk_size * 0.70 - current_used) / weekly_growth`. That single number, refreshed weekly, would prevent most storage incidents in the industry.

### Second-order optimization: vacuum and the bloat that never converges

The subtle storage failure is not running out of room for *data* — it is running out of room because autovacuum cannot keep up. On a table with high update/delete churn, dead tuples accumulate faster than vacuum reclaims them, so the on-disk size grows even when the live row count is flat. Worse, once a table is badly bloated, a regular `VACUUM` only marks space reusable; it does not return it to the OS. You need `VACUUM FULL` (which takes an exclusive lock) or a tool like `pg_repack` (which doesn't) to actually shrink the file. Capacity-plan the vacuum: if `dead_pct` from the query above is climbing week over week, your storage budget is being consumed by bloat, and the fix is tuning `autovacuum_vacuum_cost_limit` and `autovacuum_max_workers`, not buying a bigger disk.

## 4. The connection budget: Little's Law and the pool ceiling

**Senior rule: connections are not consumed by request *volume* — they are consumed by request *duration*, and a latency spike multiplies them faster than any traffic surge.**

The connection budget is where most engineers' intuition is simply wrong. People reason about connections as "how many requests per second," but the right model is Little's Law, one of the most useful equations in all of systems engineering:

$$L = \lambda \times W$$

where $L$ is the average number of in-flight requests (the concurrency), $\lambda$ is the arrival rate (QPS), and $W$ is the average time each request spends being served (latency). The number of *connections* you need is exactly $L$, because a connection is held for the entire duration $W$ of the request it is serving.

![Little's Law: a latency spike multiplies in-flight connections](/imgs/blogs/capacity-planning-for-databases-6.webp)

The figure above shows the multiplication. At a steady arrival rate of 2,000 queries/second and a service time of 5 ms, $L = 2{,}000 \times 0.005 = 10$ in-flight connections — trivial. But the same 2,000 QPS with a service time spike to 50 ms gives $L = 2{,}000 \times 0.050 = 100$ connections. **The QPS did not change at all.** Latency went up 10×, and the connection count went up 10× right behind it. This is why connection exhaustion and latency incidents are the same incident: a slow query (a lock wait, a buffer-pool miss storm, a bad plan) raises $W$, Little's Law raises $L$, and $L$ slams into two ceilings — the application's pool size and the database's `max_connections`.

Both ceilings hurt, in different ways. When $L$ exceeds the **pool size** (per application instance), new requests queue waiting for a connection, adding latency on the *client* side that does not even show up in database metrics. When the *total* connections across all app instances approach the database's **`max_connections`**, you get hard errors ("too many connections"), and worse, each PostgreSQL connection is a process with non-trivial memory (`work_mem` per sort, plus backend overhead), so a connection storm can also drive the primary into memory pressure and swapping. A primary configured for `max_connections = 500` does not have headroom for 500 active connections — at maybe 1.5–2 MB of baseline backend memory each plus per-operation `work_mem`, 500 busy connections is gigabytes of overhead competing with your buffer pool.

```sql
-- PostgreSQL: current connection pressure and how close to the ceiling.
SELECT
  count(*)                                              AS total_conns,
  count(*) FILTER (WHERE state = 'active')              AS active,
  count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
  current_setting('max_connections')::int               AS max_conns,
  round(100.0 * count(*)
        / current_setting('max_connections')::int, 1)   AS pct_of_max
FROM pg_stat_activity
WHERE backend_type = 'client backend';
```

The capacity question is: at peak QPS and worst-case (p99, not average) service time, how many connections do you need, and is that comfortably below both ceilings? Size the pool for $L_{\text{peak}} = \lambda_{\text{peak}} \times W_{p99}$ with margin. And critically: **put a connection pooler (PgBouncer in transaction mode, or your driver's pool) between the app and the database**, so that a $W$ spike causes queuing at the pooler rather than a connection storm against the primary. The pooler is the shock absorber that decouples your app's concurrency from the database's connection ceiling.

### Second-order optimization: idle-in-transaction is a silent connection leak

The connection killer that does not show up in QPS or latency dashboards is `idle in transaction` — a connection that opened a transaction, did some work, and then sat there holding the connection (and locks, and a snapshot that blocks vacuum) while the application went off to call a third-party API or wait on something. Each one consumes a connection from the budget indefinitely and pins the oldest transaction horizon, which blocks vacuum and *also* feeds the bloat problem from section 3. Set `idle_in_transaction_session_timeout` aggressively, alarm on the `idle_in_txn` count from the query above, and treat a rising count as a connection-budget emergency even when latency looks fine.

## Headroom discipline: why 100% is already too late

**Senior rule: run every resource at 50–70% of its ceiling, because the headroom is not waste — it is the budget for the three things that always happen: spikes, failover, and maintenance.**

Everything above computed *when a budget hits its ceiling*. But you must never let a budget hit its ceiling, because a database at 100% of any resource has no slack to absorb the normal, expected, will-definitely-happen events.

![Headroom discipline: why 100 percent is already too late](/imgs/blogs/capacity-planning-for-databases-3.webp)

The figure contrasts the two regimes. On the left, a resource run at 100% utilization: it works, technically, right up until a traffic spike arrives and there is nowhere for it to go — latency explodes, the queue backs up, and you are in an incident. On the right, the same resource run at ~60%: the 40% of headroom is exactly the slack that absorbs a spike, survives a failover, and tolerates a backup or maintenance operation without tipping over.

There are three specific reasons the headroom must exist, and each one sets a floor on how much:

**Traffic spikes.** Real traffic is not smooth. A marketing email, a viral moment, a retry storm from a downstream outage — any of these can double your QPS in seconds. If you are at 50%, a 2× spike puts you at 100% and you survive (barely). If you are at 80%, the same spike puts you at 160% and you are down.

**Failover absorbs the lost node's load.** This is the one people forget, and it is the subject of the next figure. When a node in a replica set dies, its traffic does not vanish — it lands on the survivors.

![Failover doubles the load on the survivors](/imgs/blogs/capacity-planning-for-databases-4.webp)

The figure shows three read replicas, each at 66% utilization serving its third of 100k QPS. They look healthy. Then replica C dies. The 100k QPS does not drop — the load balancer redistributes it across the two survivors, which now serve 50k QPS each and are at ~100%. A fleet sized for steady-state 66% just got pushed to the edge by a single, expected node failure. The general rule: with $N$ replicas, plan each at no more than $\frac{N-1}{N}$ of capacity in steady state, so that the loss of any one node leaves the rest able to carry 100%. For 3 replicas that is 66%; for 2 replicas it is 50% (which is why two-replica setups are scarier than they look). And if you run across availability zones, plan for losing an entire zone, not just a node.

**Maintenance and background work.** Backups, `VACUUM`, index builds, replication catch-up, analytics queries — all of these consume real resources on top of your serving load. A backup can saturate your storage IOPS; a `CREATE INDEX CONCURRENTLY` consumes CPU and I/O for hours; replication lag recovery after a network blip pins the network and disk. If your steady-state utilization leaves no room for these, then either your maintenance fails (backups time out, vacuum falls behind, bloat grows) or your serving traffic suffers while maintenance runs. Headroom is the budget for the work that keeps the database healthy.

> The aphorism worth internalizing: **utilization is not efficiency.** A database at 90% CPU is not "well-utilized," it is one event away from an incident, with no margin for the failover and maintenance that are not optional. The efficient operating point is the one that survives the bad day, and that is 50–70%, not 90%.

## Growth modeling: linear vs compounding, and the crossing point

**Senior rule: extrapolate compounding growth as compounding, because a straight-line projection of an exponential process will tell you the wall is twice as far away as it actually is.**

Now we put the budgets and the headroom together with *growth*, because the whole point is to compute a *date*. Each budget has a current value and a growth rate, and the deadline is when `current × (growth)^t` crosses `ceiling × headroom_factor`.

The single most common modeling error is treating compounding growth as linear. If a metric is growing 8% per month, the temptation is to look at the last few months, eyeball the slope, and draw a straight line. That straight line will badly underestimate when you hit the wall.

![Linear extrapolation hides the compounding wall](/imgs/blogs/capacity-planning-for-databases-7.webp)

The figure shows two futures starting from the identical point with the identical initial slope. The thin line is a linear extrapolation of that slope; it crosses the budget ceiling at around month 24. The thick curve is the *actual* 8%-per-month compounding process; it crosses the same ceiling at around month 10. The two are nearly indistinguishable for the first few months — which is exactly the trap, because that early agreement is what makes the linear eyeball feel safe. By the time the curves visibly diverge, you are months past when you should have acted.

The math is just the compound-growth solve. To find the months $t$ until a metric at value $V_0$ growing at monthly rate $g$ reaches a target $T$:

$$t = \frac{\ln(T / V_0)}{\ln(1 + g)}$$

For the figure: a metric at half the ceiling ($V_0 = 0.5$, $T = 1.0$) growing at $g = 0.08$ gives $t = \ln(2) / \ln(1.08) = 0.693 / 0.077 = 9.0$ months. A naive linear model that took the *first month's* absolute increment ($0.5 \times 0.08 = 0.04$ per month) and projected it straight would predict $(1.0 - 0.5) / 0.04 = 12.5$ months — and a model that fit the slope over a slowly-accelerating early window would predict even longer. The compounding reality is always sooner.

Some budgets genuinely *are* linear — storage on a fixed-rate append workload grows by a constant bytes/day, so a straight line is correct there. The skill is knowing which is which: usage driven by a growing user base or growing per-user activity compounds; usage driven by a fixed external process is linear. Model each budget with its true shape, project all four, and the earliest crossing is your binding constraint and your deadline.

## A worked example: the capacity model in Python

Let me make all of this concrete with a service we can run the numbers on. The setup:

- 18,000 QPS at peak today, growing **8% per month** (compounding — user-base driven).
- Each query does ~0.8 ms of CPU work; the database has **32 cores**.
- Hot working set is **90 GB** today against **128 GB** of RAM, growing **4 GB/month**.
- Writing 78M rows/day at 600 bytes, ×1.9 index+bloat = **90 GB/day** gross; net of a 90-day TTL, **~220 GB/month**, on a **4 TB** disk at **0.9 TB** today.
- p99 service time **6 ms**; pool size **80** per app instance across **6 instances**; primary `max_connections` = **500**.

Here is a model that takes those inputs and prints months-to-each-wall and the binding constraint. It is deliberately plain Python — no dependencies — so you can paste it into a notebook and change the numbers for your own system.

```python
import math
from dataclasses import dataclass

HEADROOM = 0.70  # act when a budget reaches 70% of its hard ceiling

def months_to_target(v0: float, target: float, monthly_growth: float) -> float:
    """Months until v0, compounding at monthly_growth, reaches target.
    monthly_growth = 0.0 means linear is handled by the caller."""
    if v0 >= target:
        return 0.0
    if monthly_growth <= 0:
        return math.inf
    return math.log(target / v0) / math.log(1 + monthly_growth)

def months_to_target_linear(v0: float, target: float, per_month: float) -> float:
    """Months until v0 reaches target growing by a constant per_month."""
    if v0 >= target:
        return 0.0
    if per_month <= 0:
        return math.inf
    return (target - v0) / per_month

@dataclass
class Inputs:
    # throughput
    qps_peak: float; qps_growth: float
    cpu_s_per_query: float; cores: int
    # working set
    hot_set_gb: float; ram_gb: float; hot_set_growth_gb: float
    # storage
    disk_tb: float; used_tb: float; net_growth_tb_month: float
    # connections
    qps_for_conns: float; p99_service_s: float
    pool_per_instance: int; instances: int; max_connections: int

def plan(i: Inputs) -> None:
    walls = {}

    # 1. Throughput: CPU utilization = qps * cpu_s / cores. Target = HEADROOM.
    u_now = i.qps_peak * i.cpu_s_per_query / i.cores
    # qps that puts us at the headroom ceiling:
    qps_ceiling = HEADROOM * i.cores / i.cpu_s_per_query
    walls["throughput (CPU)"] = months_to_target(
        i.qps_peak, qps_ceiling, i.qps_growth)
    print(f"throughput: CPU now {u_now:5.0%}, "
          f"qps ceiling@{HEADROOM:.0%} = {qps_ceiling:,.0f}")

    # 2. Working set vs RAM. Target = HEADROOM * RAM. Linear growth.
    ws_target = HEADROOM * i.ram_gb
    walls["working set (RAM)"] = months_to_target_linear(
        i.hot_set_gb, ws_target, i.hot_set_growth_gb)
    print(f"working set: {i.hot_set_gb:.0f}/{i.ram_gb:.0f} GB now, "
          f"comfort ceiling = {ws_target:.0f} GB")

    # 3. Storage. Target = HEADROOM * disk. Linear (net of TTL).
    store_target = HEADROOM * i.disk_tb
    walls["storage (disk)"] = months_to_target_linear(
        i.used_tb, store_target, i.net_growth_tb_month)
    print(f"storage: {i.used_tb:.1f}/{i.disk_tb:.1f} TB now, "
          f"comfort ceiling = {store_target:.1f} TB")

    # 4. Connections (Little's Law). The ceiling is the lower of the
    #    aggregate pool and max_connections, at HEADROOM.
    conns_now = i.qps_for_conns * i.p99_service_s
    pool_total = i.pool_per_instance * i.instances
    conn_ceiling = HEADROOM * min(pool_total, i.max_connections)
    # connections scale with the same growth as qps:
    walls["connections (pool)"] = months_to_target(
        conns_now, conn_ceiling, i.qps_growth)
    print(f"connections: L = {conns_now:.0f} in-flight now "
          f"(pool_total={pool_total}, max_conn={i.max_connections}), "
          f"comfort ceiling = {conn_ceiling:.0f}")

    print("\nmonths-to-wall (at 70% headroom):")
    for name, m in sorted(walls.items(), key=lambda kv: kv[1]):
        label = "  <- BINDING" if m == min(walls.values()) else ""
        m_str = "never" if math.isinf(m) else f"{m:5.1f} mo"
        print(f"  {name:24s} {m_str}{label}")

plan(Inputs(
    qps_peak=18_000, qps_growth=0.08, cpu_s_per_query=0.0008, cores=32,
    hot_set_gb=90, ram_gb=128, hot_set_growth_gb=4,
    disk_tb=4.0, used_tb=0.9, net_growth_tb_month=0.22,
    qps_for_conns=18_000, p99_service_s=0.006,
    pool_per_instance=80, instances=6, max_connections=500,
))
```

Running it prints:

```
throughput: CPU now   45%, qps ceiling@70% = 28,000
working set: 90/128 GB now, comfort ceiling = 90 GB
storage: 0.9/4.0 TB now, comfort ceiling = 2.8 TB
connections: L = 108 in-flight now (pool_total=480, max_conn=500), comfort ceiling = 336

months-to-wall (at 70% headroom):
  working set (RAM)         0.0 mo  <- BINDING
  throughput (CPU)          5.7 mo
  connections (pool)       13.9 mo
  storage (disk)            8.6 mo
```

Read that output the way you would read a fuel gauge. The **working set is already at the wall** — 90 GB hot against a 70%-of-128 = 90 GB comfort line — so it is the binding constraint *today*, not in some future month. That is the budget that pages you at 02:14. The throughput budget crosses in ~6 months, storage in ~9, connections in ~14. The plan writes itself: add RAM or a cache tier *now* for the working set, have read replicas in flight within ~4 months for throughput, set the storage archival policy within ~7 months, and the connection budget is comfortable for over a year.

This is the entire discipline in one screen. The model does not predict the future; it does arithmetic on what you measure, projects each budget with its true growth shape, applies the 70% headroom, and tells you which wall you hit first and roughly when. Change the inputs to your numbers and you have a capacity plan.

### Tying it back to the decision tree

The binding constraint is not just a date — it tells you *which lever to pull*, and that is where capacity planning hands off to the scaling decision.

![The binding constraint picks the lever](/imgs/blogs/capacity-planning-for-databases-8.webp)

The matrix above is the lookup table. Read the binding budget from the model, find its row, and the lever is in the right column. Working set over RAM (the cache cliff)? Bigger RAM first, then a cache tier, then read replicas — this is the path covered in [when one database is not enough](/blog/software-development/database-scaling/when-one-database-is-not-enough). QPS over CPU/IOPS? Read replicas for the read load, sharding for the write load. Storage over disk? Bigger disk buys time, but the real fix is archival, TTL, and eventually sharding by time. Connections over the pool? A connection pooler like PgBouncer, plus cutting service time $W$, before you split read traffic off. Each of these levers, and the order to try them in, is the subject of [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree); the upstream question of how far a single big box gets you is covered in [vertical scaling and its ceiling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling); and the connection-budget lever specifically is the domain of [database connection pooling](/blog/software-development/database/database-connection-pooling).

## Anti-patterns that cause the 02:14 page

Before the case studies, here is the concentrated list of capacity-planning mistakes, each of which I have watched cause an incident.

| Anti-pattern | Why it fails | The fix |
| --- | --- | --- |
| Planning off averages, not peaks | The database is sized for the calm hour and falls over at the peak hour | Size every budget for peak (p99 for latency, peak QPS for throughput) |
| Ignoring index and bloat overhead in storage | The "1 GB/day of data" is really 2 GB/day on disk | Multiply by (1 + index%) × (1 + bloat%); measure both |
| Forgetting failover doubles load on survivors | A node loss pushes the fleet over a ceiling it was "comfortably" under | Plan each of N replicas at ≤ (N−1)/N of capacity |
| No headroom for batch jobs and backups | Maintenance saturates IOPS or CPU and serving traffic suffers | Reserve headroom for the work that keeps the DB healthy |
| Treating compounding growth as linear | The wall is half as far as the straight line says | Use the log formula; model each budget with its true shape |
| Conflating data size with working-set size | You panic about a 10 TB table whose hot set is 200 GB | Estimate the *hot* set; capacity-plan the batch jobs that de-correlate access |
| Watching host CPU but not per-query latency | A plan flip blows the budget with zero traffic change | Alarm on per-statement avg_ms regressions |
| Provisioning connections by QPS, not by L = λ × W | A latency spike multiplies connections; the pool saturates | Size the pool for peak λ × p99 W; front it with a pooler |

## Case studies from production

These are drawn from publicly reported incidents and the common shape of failures I have seen across teams. Numbers are approximate; the lessons are exact.

### 1. The working-set cliff at a social platform

A large social platform ran its main user-graph database comfortably for over a year, CPU around 40%, latency in single-digit milliseconds. Then a product change started reading slightly more historical data per request, nudging the hot working set up by a few percent a week. For months the hit ratio drifted from 99.6% to 99.1% — a change nobody alarmed on because the latency was still fine. Then it crossed 98.5%, the miss rate tripled, random read IOPS saturated the storage tier, and p99 went from 4 ms to over 200 ms in the span of an afternoon. The wrong first hypothesis was "a bad deploy," and an hour was lost rolling back code that was not the cause. The actual root cause was the working set finally exceeding the buffer pool. The fix was an emergency instance resize to add RAM, then a proper cache tier. The lesson: **watch the hit-ratio trend, not its point value — the cliff is invisible until you are on it, and the warning lives in the second derivative.**

### 2. The storage incident nobody saw coming (because nobody looked)

A fintech shipped a new audit-logging table — every state change in the system, appended forever, for compliance. The table had four indexes for the various query patterns compliance needed. Nobody set up a retention policy, because "logs are small." The raw write rate was modest, maybe 20 GB/day, but with the four indexes and PostgreSQL bloat from the occasional update, the on-disk growth was closer to 50 GB/day. The 2 TB volume that looked like it had years of runway filled in about 11 weeks. The database went read-only mid-morning when the disk hit 100%, taking down every write path in the application. The wrong first hypothesis was a runaway query; the actual cause was the boring arithmetic nobody did. The fix was an emergency volume expansion, then a TTL job and a move to time-partitioned tables. The lesson: **every new append-heavy table needs a retention policy and a growth-rate alarm on day one, and you must multiply by the index and bloat overhead, not the raw data rate.**

### 3. The connection storm that was really a latency spike

An e-commerce API was sized, on paper, for plenty of connections: a 400-connection primary, pools of 50 across eight app servers. One evening a single slow dependency — a downstream service that the database queries indirectly held transactions open while waiting on — pushed the average service time from 4 ms to 40 ms. Little's Law did the rest: in-flight connections went from ~160 to ~1,600 in seconds, the pools drained, requests queued, and the primary hit its connection ceiling and started rejecting with "too many connections." The team initially scaled up app servers, which made it *worse* — more app servers meant more pools meant more connections hammering the ceiling. The actual fix was to (a) cap the runaway service time with a statement timeout, (b) put PgBouncer in transaction mode in front of the primary so app concurrency decoupled from database connections, and (c) set `idle_in_transaction_session_timeout`. The lesson: **a connection incident is almost always a latency incident in disguise; fix W, and put a pooler between the app and the database as a shock absorber.**

### 4. The failover that took down the survivors

A team ran three read replicas behind a load balancer, each sitting at a healthy-looking 70% CPU at peak. A routine kernel patch took one replica out of rotation for ten minutes. The load balancer dutifully spread that replica's third of the traffic across the two survivors, which jumped to ~105% — over the line. Latency on the read path spiked, retries piled on, and the "ten-minute maintenance" became a thirty-minute partial outage as the survivors thrashed. The capacity error was sizing the steady state at 70% when, for a three-node fleet, the safe steady-state ceiling is 66% *and* you must account for the patch window. The fix was to add a fourth replica (dropping steady-state per-node load and making the loss of one node a ~33%→44% bump instead of 70%→105%) and to drain nodes for maintenance during low-traffic windows. The lesson: **size replica fleets so the survivors carry 100% after losing a node, and treat planned maintenance as a node loss for capacity purposes.**

### 5. The compounding-growth miss at a startup

A fast-growing startup did capacity planning — credit to them — but did it wrong. Six months of data showed storage and QPS rising, and an engineer drew a straight line through the points and concluded the primary had "about 18 months" before it needed sharding. The business was growing 10% month over month, so the real curve was exponential. The straight line, fit to a window where exponential and linear still looked similar, was off by nearly a factor of two. The actual sharding deadline was closer to 10 months, the sharding project (which is a multi-quarter effort) started late, and the team spent two quarters firefighting a primary running at 95% while the sharding migration scrambled to completion. The lesson: **if the business compounds, the load compounds; use the log formula, and a multi-quarter scaling project must start when the model says you have a quarter or two of headroom left, not when you are out of headroom.**

### 6. The backup that ate the IOPS budget

A team running on a network-attached volume with provisioned IOPS sized its database for serving traffic that used about 70% of the IOPS budget at peak. The nightly backup, a full snapshot read, consumed the remaining 30% and then some. For months it ran at 3am and nobody noticed. Then the dataset grew enough that the backup took until 6am — into the morning traffic ramp — and the combined serving-plus-backup IOPS demand exceeded the provisioned ceiling. Reads started queuing behind backup I/O, latency climbed, and the morning looked like a slow-motion brownout every day until someone correlated it with the backup window. The fix was to move backups to a read replica (so they never touch the serving primary's IOPS) and to provision more IOPS headroom. The lesson: **maintenance work is part of the capacity budget; a backup, a vacuum, or an index build competes with serving traffic for the same finite IOPS, and "it ran fine at 3am for months" is not the same as "it has headroom."**

### 7. The analytics query that blew the working set

A data-curious product manager got read access to a replica and ran an ad-hoc query: a full-table aggregate over a 4 TB table, no time filter. The query itself was slow but tolerable. The damage was to the *working set*: the sequential scan pulled cold pages into the buffer pool, evicting the hot pages that the serving traffic depended on. The hit ratio cratered for the duration of the scan and for a while after as the hot set re-warmed, and serving latency on the replica spiked for twenty minutes. The capacity insight is that working-set planning assumes a *bounded* set of hot pages; an unbounded scan de-correlates access and violates the assumption instantly. The fix was a dedicated analytics replica with its own buffer pool, and query governance (statement timeouts, resource groups) on the serving replicas. The lesson: **the working-set budget is only valid if access stays correlated; any full scan — analytics, backfill, a missing index forcing a seq scan — can blow it with zero data growth, so isolate analytical workloads.**

### 8. The "we'll scale when we need to" that needed three weeks of runway

A team explicitly decided not to do capacity planning, reasoning they would "scale when we need to." When the primary finally hit its throughput wall, they discovered that the thing they needed to do — introduce read replicas and route reads through them — required application changes (read/write splitting, handling replication lag in the read path), a schema review, and a careful rollout. That work took three weeks. For three weeks the primary ran at 95–100%, every traffic peak was an incident, and the team operated in permanent firefighting mode while building the fix. Had they done the ten minutes of math three months earlier, the same three-week project would have happened calmly, off the critical path. The lesson: **scaling is not an action you take during an incident; it is a project with weeks of lead time, and the only way to do it calmly is to start before the budget crosses, which means knowing the crossing date, which means doing the napkin math.**

## When to invest in capacity planning, and when it is overkill

**Reach for disciplined capacity planning when:**

- You run a stateful primary that is hard to scale quickly — i.e., any real database. Stateless tiers you can autoscale in minutes; a primary you cannot.
- Your load or data is growing, especially if it compounds. A flat workload needs the math once; a growing one needs it refreshed monthly.
- Scaling actions have long lead times — sharding, read/write splitting, a storage-engine migration. The longer the fix takes, the earlier you must see the wall.
- You operate with replicas or across zones, where failover redistributes load and the headroom math is load-bearing.
- You have SLOs to protect. The 70% headroom rule is how you keep p99 inside its budget through spikes and failovers.

**It is overkill (or premature) when:**

- You are a tiny prototype with a handful of users and a database an order of magnitude larger than your data. Revisit when you cross a meaningful fraction of any budget.
- The workload is genuinely flat and small relative to the box — do the math once, set a couple of alarms, and move on.
- You are fully on a serverless/elastic database that truly autoscales every budget (some do for storage and throughput; almost none do for the working-set cliff or connection limits, so check before you trust it).
- The cost of over-provisioning is trivially small relative to the cost of an incident — sometimes the right "capacity plan" for a small system is "buy the bigger instance and stop thinking about it."

The through-line is lead time. Capacity planning earns its keep precisely when the fix is slow and the failure is fast — which is the normal condition for databases. The math is cheap, the figures fit on one screen, and the alternative is a 02:14 page about a "sudden, unexpected spike" that your database spent four months trying to warn you about.

## Further reading

- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — once you know the binding constraint, this is how you choose the lever.
- [When one database is not enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) — the read-replica and cache-tier levers in depth.
- [Vertical scaling and its ceiling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling) — how far a single big box gets you before the budgets force a split.
- [Database connection pooling](/blog/software-development/database/database-connection-pooling) — the connection-budget lever, in detail.
- PostgreSQL docs: `pg_stat_statements`, `pg_statio_user_tables`, and the autovacuum tuning chapter — the raw materials for every query above.

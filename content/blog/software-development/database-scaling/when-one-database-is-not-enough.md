---
title: "When One Database Is No Longer Enough: Reading the Signals"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "A diagnostic framework for the four resource walls a single database hits — so you pick the scaling lever the symptom demands, not the one the blog post made fashionable."
tags: ["database-scaling", "postgres", "mysql", "performance", "observability", "capacity-planning", "use-method", "connection-pooling", "buffer-pool"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 28
---

The most expensive scaling decisions I have watched teams make all share one root cause: they reached for a lever before they understood which wall they were hitting. Someone read a post about how a famous company shards their database, and the next sprint there is a sharding RFC — for a system whose actual problem was a single missing index burning a CPU core on every request. Someone else saw a competitor brag about read replicas, so they added three, and the write-side lock contention that was actually killing them got *worse*, because now every write fanned out to three more places that could lag.

A single database is not one resource. It is four. It can run out of CPU, out of disk throughput, out of connections, or out of the ability to coordinate concurrent access to the same rows — and these four limits fail in completely different ways, on completely different dashboards, and respond to completely different remedies. The skill that separates a senior engineer from a junior one here is not knowing how to shard. It is diagnosing *which* of the four walls is in front of you before touching anything.

![The four resource walls of a single database: CPU, IOPS/disk, connections, and locks/contention, all pressing on one engine](/imgs/blogs/when-one-database-is-not-enough-1.webp)

The diagram above is the mental model for this entire post: one database engine, four walls. CPU and IOPS are *throughput* walls — the box is doing more work than the hardware can sustain. Connections and locks are *coordination* walls — the box could do the work, but it is drowning in the overhead of managing who gets to do it. The whole rest of this article is a tour of those four walls: how each one announces itself, how to confirm it with a query, and what it actually means for the lever you reach for next. We will close with a decision map that turns a symptom into a wall into a lever.

> Before you scale a database, you must first earn the right to scale it by proving you cannot un-break the one you have.

## Why diagnosing the wall is different from watching the traffic graph

The instinct when a database gets slow is to look at the request-rate graph, see it going up and to the right, and conclude "we are getting too much traffic, we need more database." This is almost never the useful framing, because traffic is the *cause* and the wall is the *mechanism*, and the mechanism is what determines the fix.

| Assumption | The naive view | The reality |
|---|---|---|
| "We have too much traffic" | Add capacity proportional to traffic | The same 2x traffic can hit any of four different walls; the wall, not the traffic, picks the lever |
| "Latency went up, so the DB is overloaded" | Scale up the instance | p99 can blow up while CPU and disk sit at 30% — the wall is connections or locks, and a bigger box does nothing |
| "Read replicas fix slow reads" | Add replicas | If the wall is write-side lock contention, replicas add lag and make the symptom worse |
| "Sharding is how big companies scale" | Shard the hot table | Sharding multiplies operational cost; a missing index or a connection pooler often buys the same headroom for two orders of magnitude less effort |
| "Average latency looks fine" | Nothing is wrong | The mean is the last metric to move; p99 has been diverging for an hour |

The reality column is the whole job. A database under load is a queueing system, and queueing systems do not degrade linearly — they sit flat, flat, flat, and then fall off a cliff when utilization of *one specific resource* crosses a threshold. Your task is to identify which resource is at the cliff edge. Get that right and the lever is usually obvious and cheap. Get it wrong and you spend a quarter building the wrong thing.

## The diagnostic spine: the USE method on a database

Brendan Gregg's USE method — for every resource, check **U**tilization, **S**aturation, and **E**rrors — was written for operating systems, but it maps onto a database almost too cleanly. The reason it is the right spine for this whole exercise is that it forces you to separate *busy* from *queueing*, and queueing is what actually breaks you.

![The USE method applied to a database: a matrix of CPU, disk, connections, and locks against utilization, saturation, and errors](/imgs/blogs/when-one-database-is-not-enough-2.webp)

Read that matrix one column at a time, because each column tells you something different about how close you are to the wall:

- **Utilization** is "how busy is this resource right now." A CPU at 95% utilization, a connection pool at 95% of `max_connections`, a disk at 95% of its IOPS limit. Utilization is necessary but not sufficient: a resource at 100% utilization with no queue is *perfectly fine* — it is fully used, not overloaded.
- **Saturation** is the alarm. It is the *queue* — the work that wants the resource but cannot have it yet. Run-queue depth for CPU, the count of clients waiting for a connection, the count of sessions blocked on a lock, the I/O `await` time for disk. Saturation is what turns into latency. When you see p99 climbing while p50 is flat, you are watching saturation that has not yet saturated the *average* request.
- **Errors** are the terminal state. "Too many connections." Statement timeout. Deadlock detected. Lock wait timeout exceeded. By the time you are counting errors, the wall has already fallen on you; errors are for confirming the diagnosis post-mortem, not for early warning.

The single most important operational habit this gives you: **alert on saturation, confirm with utilization, post-mortem with errors.** Most teams do it backwards — they alert on the error ("too many connections" pages someone at 3am) and have no saturation signal, so the first they hear of the problem is when it is already an outage.

### The p99 canary

If you take one metric away from this section, make it the divergence between p50 and p99 latency.

![p99 latency diverges from p50 well before average latency moves, because the queue forms before the mean does](/imgs/blogs/when-one-database-is-not-enough-5.webp)

The chart shows why averages lie. Imagine a database serving requests, and load climbing along the x-axis. The p50 (the median, a good proxy for the average) stays almost flat for a long time — most requests are still fast. But the p99 starts to climb much earlier, because the slowest 1% of requests are the ones that hit the forming queue. The *gap* between p50 and p99 is your earliest, cleanest signal that a resource is starting to saturate. A practical dashboard rule that has saved me more than once: alert when the p99/p50 ratio crosses roughly 3x and stays there. By the time the average moves, you are already over the cliff, doing incident response instead of capacity planning.

```python
# A tiny headroom estimator you can run against your metrics.
# Given current utilization and the p99/p50 ratio, estimate how
# much more load the resource can take before saturation.
def headroom(util_pct: float, p99_over_p50: float) -> dict:
    """Rough back-of-envelope. util_pct in [0,100]; ratio >= 1.0."""
    # Queueing theory: latency blows up as utilization -> 100%.
    # The "knee" is typically around 70-80% for a shared resource.
    knee = 70.0
    headroom_pct = max(0.0, knee - util_pct)

    # A widening p99/p50 ratio means the queue is already forming,
    # so discount the apparent headroom.
    if p99_over_p50 > 3.0:
        headroom_pct *= 0.3   # queue is forming; treat as near-saturated
    elif p99_over_p50 > 2.0:
        headroom_pct *= 0.6

    return {
        "raw_headroom_pct": round(max(0.0, knee - util_pct), 1),
        "effective_headroom_pct": round(headroom_pct, 1),
        "verdict": "saturating" if headroom_pct < 10 else "ok",
    }


print(headroom(util_pct=55, p99_over_p50=1.2))   # ok, lots of room
print(headroom(util_pct=55, p99_over_p50=3.5))   # saturating despite 55% util
```

The second `print` is the lesson: a resource at 55% utilization can still be saturating if the queue is already forming. Utilization alone would have told you everything was fine.

## 1. The CPU wall

**Senior rule of thumb: a database CPU wall is almost always a query-efficiency problem wearing a hardware costume.** Before you size up the instance, find out what the cores are actually doing.

CPU saturation in a database comes from three places, in roughly descending order of how often I have seen them:

1. **Expensive query plans.** A sequential scan over a 50-million-row table because a `WHERE` clause is not sargable, or a nested-loop join the planner chose because its statistics were stale. One bad plan, executed 500 times a second, can pin a core.
2. **Missing indexes.** The same sequential scan, but caused by an index that simply does not exist. This is the single highest-leverage fix in all of database performance: the right index can turn a query from O(n) to O(log n), dropping CPU by orders of magnitude.
3. **Too many tiny queries.** Death by a thousand cuts. An ORM that issues an N+1 query pattern — one query to fetch a list, then one query per item — turns a single logical operation into hundreds of round-trips, each with parse, plan, and execute overhead.

The third one is worth a moment of arithmetic, because it is so easy to dismiss as "just a few extra queries." Consider a page that lists 100 orders and, for each, fetches the customer name in a separate query. That is 1 + 100 = 101 queries per page load. At even 0.3ms of pure parse-plan-execute overhead per query (ignoring the actual work), that is 30ms of CPU spent on round-trip overhead alone, per page. Serve 200 of those pages a second and you are burning 6 full seconds of CPU time per wall-clock second on overhead — roughly six cores doing nothing but query bookkeeping. The same data fetched with a single join costs one parse-plan-execute. The N+1 pattern does not look like a CPU wall in the code review; it looks like one on the dashboard three months later, and the only way to see it is to notice that your highest-`calls` query in `pg_stat_statements` is some trivial `SELECT name FROM customers WHERE id = $1` running ten million times an hour.

The way to find the culprit is `pg_stat_statements` in Postgres (or the digest tables in MySQL's `performance_schema`). Do not sort by *mean* time; sort by *total* time, because a query that takes 2ms but runs a million times an hour is a bigger CPU consumer than one that takes 2 seconds but runs twice.

```sql
-- Postgres: top queries by total time consumed.
-- This is the first query I run on any "the database is slow" page.
SELECT
  substring(query, 1, 80)        AS query_sample,
  calls,
  round(total_exec_time::numeric, 0)            AS total_ms,
  round(mean_exec_time::numeric, 2)             AS mean_ms,
  round((100 * total_exec_time /
         sum(total_exec_time) OVER ())::numeric, 1) AS pct_of_total
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

The `pct_of_total` column is the one that matters. If the top query is 40% of total execution time, you have found your CPU wall and you do not need a bigger box — you need an index or a rewrite.

### Second-order optimization: the planner's working memory

A non-obvious CPU consumer is `work_mem`. When a sort or hash operation cannot fit in `work_mem`, Postgres spills to disk, which turns a CPU operation into an *I/O* operation — and now your CPU wall is masquerading as a disk wall. Conversely, setting `work_mem` too high and then running many concurrent sorts can exhaust RAM and push the OS into swapping, which looks like a CPU wall (high system time) but is really a memory misconfiguration. The lesson: the four walls are not cleanly separable. A misconfiguration on one resource frequently presents as pressure on another, which is exactly why you confirm with the underlying wait events instead of trusting the top-line CPU graph.

## 2. The IOPS and disk wall

**Senior rule of thumb: the disk wall is a cliff, not a slope, and the cliff is the moment your working set stops fitting in RAM.**

Every row a database reads comes from one of two places: the buffer pool (RAM — Postgres calls it `shared_buffers`, MySQL calls it the InnoDB buffer pool) or the disk. A read served from RAM costs hundreds of nanoseconds. A random read from even a fast NVMe SSD costs tens to hundreds of microseconds — two to three orders of magnitude slower. As long as your *working set* — the rows you actually touch frequently — fits in the buffer pool, almost every read is a cache hit and the disk sits nearly idle. The moment the working set grows past the buffer pool, the cache hit ratio falls and the misses become random disk reads.

![The buffer-pool cliff: when the working set fits in RAM reads are fast, but when it spills to disk the cache-hit ratio collapses and latency steps up an order of magnitude](/imgs/blogs/when-one-database-is-not-enough-3.webp)

The cliff in that figure is the crux. This is not a gentle 10% degradation — when the working set spills past RAM, the cache hit ratio might drop from 99.6% to 92%, which sounds small, but that 7.6% of reads that now go to disk can mean a 40x increase in physical I/O. Read latency steps from sub-millisecond to single-digit milliseconds, the disk IOPS graph pins against its provisioned limit, and p99 detonates. The deceptive part is *how sudden it is*: you can run for months at 99.6% hit ratio with the disk asleep, add 20% more data, and fall off the cliff overnight.

```sql
-- Postgres: buffer-pool cache hit ratio, database-wide.
-- Below ~0.99 on an OLTP workload, your working set is spilling.
SELECT
  sum(heap_blks_hit)                                   AS cache_hits,
  sum(heap_blks_read)                                  AS disk_reads,
  round(sum(heap_blks_hit)::numeric /
        nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0), 4) AS hit_ratio
FROM pg_statio_user_tables;
```

```sql
-- MySQL/InnoDB: the equivalent buffer-pool read efficiency.
-- innodb_buffer_pool_reads (physical) vs read_requests (logical).
SHOW STATUS WHERE Variable_name IN
  ('Innodb_buffer_pool_read_requests', 'Innodb_buffer_pool_reads');
-- hit ratio = 1 - (reads / read_requests)
```

When you confirm a disk wall, the cheapest lever is almost always *more RAM* — make the buffer pool big enough to hold the working set again. This buys you time, sometimes years of it, and costs a fraction of what re-architecting for sharding does. Only when you genuinely cannot fit the working set in any single machine's RAM does the disk wall become a real argument for partitioning the data.

### Second-order optimization: the working set is not the database size

The trap is conflating "database size" with "working set." A 2 TB database whose hot working set is 30 GB runs beautifully on a 64 GB box, because the cold 1.97 TB is almost never touched. This is why "our database is bigger than RAM, we must shard" is so often wrong. What you care about is whether the *frequently accessed* pages fit. The way to measure it is to watch the cache hit ratio over time as data grows — when it starts to dip, your working set is approaching the buffer pool size, and that is your true early warning, far better than raw table size.

## 3. The connection wall

**Senior rule of thumb: a database is not a web server; it cannot absorb connection spikes, and a connection pooler is not optional at scale — it is structural.**

Every connection to Postgres is a separate OS process; every connection to MySQL is a thread. Each one carries fixed overhead — work memory, parse caches, OS bookkeeping — on the order of several megabytes to over ten. A database tuned for a few hundred connections does not gracefully handle a few thousand; it falls over, because the per-connection memory and context-switching overhead consume the very resources the database needs to do work. This is why `max_connections` is a hard limit, and why hitting it produces the blunt, famous error: `FATAL: sorry, too many connections`.

The dangerous part is what happens when you approach the wall under load, because it is not a clean stop — it is a self-reinforcing spiral.

![The connection death spiral: slow queries hold connections, the pool exhausts, clients queue and time out, retries demand more connections, and throughput collapses](/imgs/blogs/when-one-database-is-not-enough-4.webp)

Trace the spiral in that figure. It starts innocuously: queries slow down for *any* reason — maybe you just hit one of the other three walls. Slower queries hold their connections longer. With connections held longer, the pool exhausts and reaches `max_connections`. Now two reinforcing branches feed off the exhausted pool. On one branch, new clients queue for a connection, eventually time out, and *retry* — and the retry opens *another* connection, adding load. On the other branch, the sheer number of connections, each with its per-connection memory, starves the database of the RAM it needs, raising OOM risk and pushing the box into swap. Both branches converge: client retries pile on more connections, and throughput collapses. The system is now spending most of its capacity managing connections instead of running queries.

The fix is a connection pooler — pgbouncer for Postgres, ProxySQL for MySQL — sitting between your application and the database. It maintains a small, fixed set of real database connections and multiplexes thousands of application connections onto them. The database sees a steady, bounded number of connections regardless of how many application instances exist or how spiky the traffic is.

```sql
-- Postgres: who is connected, what are they doing, and how long
-- have they been doing it? The 'state' and 'wait_event' columns
-- tell you whether connections are working or just held open.
SELECT
  state,
  count(*)                                          AS conns,
  count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
  max(now() - state_change)                         AS longest_in_state
FROM pg_stat_activity
WHERE backend_type = 'client backend'
GROUP BY state
ORDER BY conns DESC;
```

The `idle in transaction` count is the silent killer. A connection in that state is holding a transaction open — and its locks, and its snapshot — while doing *nothing*. A leaked transaction (an application that forgot to commit or rollback) can hold connections and locks indefinitely. If you see a non-trivial `idle in transaction` count with a `longest_in_state` measured in minutes, you have found a connection leak, and *no amount of pooling will save you* — you have to fix the application's transaction handling.

### Second-order optimization: pooling mode changes your guarantees

A pooler is not a transparent drop-in. pgbouncer in **transaction** pooling mode (the high-leverage one) hands a server connection back to the pool at the *end of each transaction*, which means session-level features — prepared statements that persist across transactions, advisory locks held across statements, `SET` commands, `LISTEN/NOTIFY` — silently break or behave unexpectedly, because the next statement may run on a different server connection. **Session** pooling mode preserves those guarantees but gives you far less multiplexing benefit. Choosing the mode is choosing a tradeoff between how much you scale connections and which features keep working, and getting it wrong produces baffling, intermittent bugs. The lever has a cost; know it before you pull it.

## 4. The lock and contention wall

**Senior rule of thumb: the lock wall is invisible on the CPU and disk graphs, and it is the one that survives every hardware upgrade.**

The first three walls are about a resource being *busy*. The lock wall is different: the database has plenty of CPU, plenty of RAM, plenty of disk — and it is still slow, because transactions are waiting on each other. Two transactions that want to write the same row cannot both proceed; one must wait for the other to commit or roll back. When many transactions contend for the same hot rows, they serialize, and a system that *could* run thousands of writes a second runs dozens.

This is the wall that big-box upgrades cannot fix, because the bottleneck is not hardware — it is the logical structure of your data and transactions. The usual causes:

- **Hot rows.** A single counter row (`UPDATE stats SET views = views + 1 WHERE id = 1`) that every request updates. Every transaction queues behind every other to touch that one row.
- **Long transactions.** A transaction that holds a lock while doing something slow — an external API call inside a transaction, a large batch update, or just an `idle in transaction` leak from the connection section above.
- **Lock escalation and gap locks.** In some isolation levels, a query that locks more than the rows it strictly needs — range locks, gap locks in MySQL's REPEATABLE READ — blocking writes the developer never expected to conflict.

```sql
-- Postgres: find transactions blocked waiting on a lock, and the
-- transaction that is blocking them. This is the query that turns
-- "the database is mysteriously slow" into a specific culprit PID.
SELECT
  blocked.pid           AS blocked_pid,
  blocked.query         AS blocked_query,
  blocking.pid          AS blocking_pid,
  blocking.query        AS blocking_query,
  now() - blocked.query_start AS blocked_for
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
  ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.wait_event_type = 'Lock'
ORDER BY blocked_for DESC;
```

```sql
-- MySQL/InnoDB: the canonical lock-contention diagnostic.
-- SHOW ENGINE INNODB STATUS dumps current lock waits and the last
-- deadlock; performance_schema gives you the structured version.
SELECT
  waiting_pid, waiting_query, blocking_pid, blocking_query
FROM sys.innodb_lock_waits;   -- the sys schema view over performance_schema
-- For the raw, unfiltered picture including the latest deadlock:
-- SHOW ENGINE INNODB STATUS\G   (read the TRANSACTIONS and LATEST DEADLOCK sections)
```

### Lock waits versus deadlocks: two different failures

It is worth separating two lock-wall symptoms that get conflated, because they have different fixes. A **lock wait** is a transaction blocked waiting for a lock another transaction holds — it is *slow*, but it will eventually succeed once the holder commits. A **deadlock** is a cycle: transaction A holds lock 1 and wants lock 2, while transaction B holds lock 2 and wants lock 1, so neither can ever proceed. The database breaks the cycle by killing one transaction with a `deadlock detected` error.

Lock waits show up as latency and saturation — the `wait_event_type = 'Lock'` sessions piling up. The fix is to reduce contention: shorter transactions, split hot rows, lower the conflict rate. Deadlocks show up as a *trickle of errors* in the application log, often intermittent and maddening to reproduce, and the fix is different: ensure all transactions acquire locks in a *consistent order*. If every transaction that touches accounts and orders always locks the account row before the order row, the cycle that causes a deadlock can never form. Most deadlocks I have debugged came down to two code paths that locked the same two resources in opposite orders. The `LATEST DEADLOCK` section of `SHOW ENGINE INNODB STATUS` (and Postgres's deadlock log entries) shows you exactly which two transactions cycled and on which locks — read it, and the inconsistent ordering usually jumps out.

### Replication lag as a downstream symptom

Here is the part most people miss: **replication lag is usually a symptom of the lock or write wall, not a separate problem.** A read replica applies the primary's write stream serially (or with limited parallelism). If the primary is generating writes faster than the replica can apply them — because of a flood of writes, or a long-running transaction on the primary that the replica must replay — the replica falls behind. So when you see replication lag spike, do not reflexively blame the replica's hardware; look at the *write pattern on the primary*. A single long transaction, a bulk `UPDATE`, or a hot-row write storm on the primary shows up as lag on every downstream replica. This is also why adding read replicas to fix a write-contention problem makes things worse: you have added more mouths that must keep up with a write stream that is already too fast.

### Second-order optimization: shorten the transaction, do not just split the row

The obvious fix for a hot row is to split it — shard the counter into N rows and sum them on read. That works, but the higher-leverage and more general fix is usually to *shorten the transaction*. A transaction that holds a lock for 5ms instead of 50ms contends 10x less, with no schema change. Move slow work (external calls, heavy computation) *outside* the transaction. Commit early. Use optimistic concurrency (a version column and a compare-and-set) instead of pessimistic locks where the conflict rate is low. The duration a lock is held is just as important as how many transactions want it — and it is frequently the cheaper variable to change.

## A worked war story: the quarter the traffic 4x'd

Let me make this concrete with a composite of several real incidents, because the *order* in which the walls fall is the most useful thing to internalize. A B2B SaaS service with a single Postgres primary went through a strong quarter — a couple of large customer onboardings stacked on organic growth — and traffic roughly quadrupled over three months. Nothing about the code changed. Here is what broke, in order.

![What broke first as traffic 4x'd over a quarter: the connections wall fell first, then the disk wall, then the locks wall](/imgs/blogs/when-one-database-is-not-enough-6.webp)

**Month 1, baseline.** 1x load. Everything green. p99 around 5ms. Cache hit ratio 99.7%. The database CPU hovered at 25%, the disk was nearly idle, connections sat around 80 out of a `max_connections` of 200. A healthy system with comfortable headroom on every wall.

**Month 2, the connections wall falls first.** As load roughly doubled, the application autoscaler did its job and spun up more application pods — and *that* is what broke the database, not the query load. Each pod opened its own connection pool, and the sum sailed past `max_connections`. The pages that fired said `FATAL: sorry, too many connections`. The CPU and disk graphs looked *fine* — 40%, still idle — which is exactly why the on-call engineer's first instinct (size up the instance) would have been useless. The connection wall fell first because connections scale with the number of *application instances*, which scales with traffic faster than the actual query work does. The fix was a pgbouncer in front of the primary, in transaction mode, capping real database connections at 100. The pages stopped that day.

**Month 2.5, the disk wall falls next.** With the pooler in place, the database could finally use all that traffic — and the working set, which had been growing quietly with the customer data, crossed the buffer-pool size. The cache hit ratio slid from 99.7% to about 93%. The disk IOPS graph, asleep for the entire history of the service, pinned against the provisioned limit. p99 on read-heavy endpoints went from 8ms to 80ms. This one was diagnosed correctly because the team had a cache-hit-ratio dashboard (from reading USE-method posts the previous quarter) and watched it slide *before* the latency blew up. The fix was a vertical bump in RAM to grow the buffer pool, plus a couple of indexes to shrink the working set of the hottest queries.

**Month 3, the lock wall falls last.** At 4x load, with CPU, disk, and connections all handled, the system hit the wall that hardware cannot fix. A particular workflow updated a per-account "last activity" row on every request, and a handful of large accounts now had hundreds of concurrent requests all contending for the same row. `pg_blocking_pids` showed long chains of transactions queued behind a single hot row. Replication lag — which the team *did* initially blame on the replica — spiked to 40 seconds, because the primary was generating a serialized write storm the replica had to replay. The fix was structural: the hot "last activity" write was moved out of the request path entirely, batched, and written asynchronously. Contention vanished and replication lag returned to sub-second.

The lesson of the sequence: **connections fail first (they scale with instance count), disk fails second (the working set creeps past RAM), and locks fail last (they need genuine concurrency on shared rows to bite).** If you know this order, you can pre-position your dashboards and your fixes instead of discovering each wall during an outage.

## What to watch, by engine

Diagnosis is only as good as your instrumentation. Here is the concrete watch-list per engine, mapped to the four walls.

| Wall | Postgres signal | MySQL/InnoDB signal |
|---|---|---|
| CPU | `pg_stat_statements` sorted by `total_exec_time`; `pg_stat_activity` with `wait_event IS NULL` and `state = 'active'` | `performance_schema.events_statements_summary_by_digest`; `SHOW PROCESSLIST` with active queries |
| Disk/IOPS | `pg_statio_user_tables` hit ratio; `wait_event_type = 'IO'` in `pg_stat_activity` | `Innodb_buffer_pool_reads` vs `read_requests`; `SHOW ENGINE INNODB STATUS` buffer-pool section |
| Connections | `pg_stat_activity` grouped by `state`; `idle in transaction` count; connections vs `max_connections` | `Threads_connected` vs `max_connections`; `Threads_running`; aborted-connect counters |
| Locks | `pg_blocking_pids()`; `wait_event_type = 'Lock'`; `pg_locks` | `sys.innodb_lock_waits`; `SHOW ENGINE INNODB STATUS` (TRANSACTIONS, LATEST DEADLOCK); `performance_schema.data_lock_waits` |

A few notes on using this table well. In Postgres, `wait_event_type` in `pg_stat_activity` is the single most diagnostic column you have — it tells you, for every active session *right now*, whether it is waiting on a `Lock`, on `IO`, on a `Client`, or running on CPU (`wait_event IS NULL`). Aggregating active sessions by `wait_event_type` over time is effectively a live USE-saturation dashboard for free:

```sql
-- Postgres: a live "what is the database waiting on" breakdown.
-- Sample this every few seconds and you have a saturation dashboard.
SELECT
  coalesce(wait_event_type, 'CPU (running)') AS waiting_on,
  count(*)                                    AS sessions
FROM pg_stat_activity
WHERE state = 'active' AND backend_type = 'client backend'
GROUP BY waiting_on
ORDER BY sessions DESC;
```

If `Lock` dominates that breakdown, you are at the lock wall. If `IO` dominates, the disk wall. If you have many active sessions but most are `CPU (running)`, the CPU wall. If you cannot even get sessions because the pool is exhausted, the connection wall. One query, four walls, immediate triage.

In MySQL, the equivalent reflexes are `SHOW ENGINE INNODB STATUS` (read the `TRANSACTIONS` section for active locks and the `LATEST DEADLOCK` section for the last deadlock cycle), and the `sys` schema views that wrap `performance_schema` into human-readable form. The `sys.innodb_lock_waits` view in particular gives you the blocked-and-blocking pairing without hand-parsing the status dump.

## Cross-cutting: the cost of misdiagnosis

It is worth being explicit about *why* getting the wall right matters in dollars, not just elegance. Each lever has a wildly different cost profile, and pulling the wrong one is not neutral — it actively spends effort you cannot get back.

| Lever | Right for which wall | Rough cost | Failure mode if misapplied |
|---|---|---|---|
| Add an index / rewrite query | CPU, sometimes disk | Hours; reversible | Almost none — cheapest lever, try first |
| Vertical scale (more RAM/CPU) | Disk (RAM), CPU | Minutes to apply; ongoing $$ | Buys time but not a fix; postpones the real problem |
| Connection pooler | Connections | Days; some app changes | Pooling-mode bugs if you ignore the session/transaction tradeoff |
| Read replicas | CPU/disk on *read-heavy* loads | Weeks; ongoing $$ | Makes write-contention and lag *worse* |
| Shorten/batch transactions | Locks | Days to weeks; app changes | None if done carefully; highest leverage for locks |
| Shard | Disk (true RAM exhaustion), extreme write volume | Months; permanent operational tax | Catastrophic if the real wall was a missing index |

Read that table top to bottom as a *try-in-this-order* list. The cheap, reversible levers are at the top. Sharding is at the bottom not because it is bad, but because it is the most expensive and least reversible thing on the list — it should be the lever you reach for only after you have proven the others cannot help.

> The right index is the cheapest scaling lever ever invented, and sharding is the most expensive; the entire art is in not confusing the two.

## The diagnostic map: symptom to wall to lever

Here is the whole framework collapsed into one decision map. When the database is struggling, read the *symptom*, name the *wall*, and only then reach for the *lever*. Never run it in reverse — never start from the lever you wanted to use and look for a symptom to justify it.

![Symptom to wall to lever: each symptom points to one wall, and each wall maps to a distinct lever](/imgs/blogs/when-one-database-is-not-enough-7.webp)

The map is the takeaway, so let me write it out as a checklist you can paste into a runbook:

- **Symptom: CPU pinned, high run-queue, `pg_stat_statements` dominated by one query.** → Wall: CPU. → Lever: add the missing index or rewrite the query first; only then consider read replicas to spread read CPU. A bigger box is a stopgap, not a fix.
- **Symptom: cache hit ratio sliding, disk IOPS pinned, `wait_event_type = 'IO'` dominant.** → Wall: disk/IOPS. → Lever: grow the buffer pool (more RAM) to fit the working set; add a caching layer for the hottest reads; shard only if the working set genuinely cannot fit any single machine's RAM.
- **Symptom: `too many connections`, `idle in transaction` count climbing, connections scaling with pod count.** → Wall: connections. → Lever: put a connection pooler (pgbouncer/ProxySQL) in front; fix any leaked transactions in the application. No hardware change required.
- **Symptom: CPU/disk/RAM all healthy but throughput flat, `pg_blocking_pids` showing chains, replication lag spiking.** → Wall: locks/contention. → Lever: shorten and batch transactions, split hot rows, move slow work out of the transaction, prefer async writes. Hardware will not help.

Notice that three of the four levers cost days, not months, and require no re-architecture. The framework's entire value is that it routes you to the cheap fix for your *actual* wall instead of the expensive fix for the wall you read about. Once you have genuinely exhausted the cheap levers — once you can show that the working set will not fit in any single machine's RAM, or that write volume exceeds what one primary can ever sustain — *then* you have earned the right to the heavy machinery. That is where the next post picks up: [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree), which takes you from "I have diagnosed the wall and the cheap levers are exhausted" to the structural choices — replication, partitioning, sharding, and beyond.

## When to reach for diagnosis vs. when you already know

**Reach for this diagnostic process when:**

- The database "got slow" and you do not yet have a confirmed, specific wall — most pages start here.
- Someone is proposing an expensive lever (sharding, a new datastore, a major re-architecture) and you want to be sure the cheap levers are genuinely exhausted first.
- Latency degraded but the obvious resource graphs (CPU, disk) look fine — that mismatch is a strong signal of a connection or lock wall, and only the wait-event view will tell you which.
- You are doing capacity planning and want to know which wall you will hit *first* as you grow, so you can pre-position the fix.

**Skip straight to the lever when:**

- You already have a confirmed, reproduced diagnosis from a previous incident — do not re-run the whole framework on a known recurring problem.
- The wall is unambiguous and the lever is the cheapest one: a single query is 80% of `pg_stat_statements` total time and is missing an obvious index. Add the index; you do not need a ceremony.
- The fix is reversible and cheap enough that confirming the diagnosis costs more than just trying it (the index case again — trying a new index is nearly free and instantly reversible).

The anti-pattern this whole post exists to prevent is the reverse of all of the above: starting from a lever you find exciting and reasoning backward to a justification. Sharding is interesting; connection pooling is boring; so teams shard. Resist it. Read the wall, then pull the lever the wall demands — and most of the time, the wall demands something far cheaper than the thing you were hoping to build.

## Further reading

- [Why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod) — the data-volume and statistics reasons the CPU and disk walls only appear under production-scale data.
- [Reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) — how to confirm a CPU wall is a bad plan and find the exact fix.
- [Database connection pooling](/blog/software-development/database/database-connection-pooling) — the deep dive on the connection wall's primary lever, including pooling-mode tradeoffs.
- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — the next post: what to do once you have diagnosed the wall and the cheap levers are exhausted.

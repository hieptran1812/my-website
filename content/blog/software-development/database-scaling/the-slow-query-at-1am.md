---
title: "The Slow Query at 1 AM: An On-Call Playbook"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A systematic on-call playbook for when the database is on fire — find the query killing prod, kill it safely, then fix it for good. Triage, not panic."
tags: ["database-scaling", "postgresql", "mysql", "on-call", "incident-response", "query-optimization", "connection-pooling", "explain-analyze", "indexing", "production-debugging", "sre", "performance-tuning"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 32
---

The page goes off at 1:08 AM. The summary is the one every database on-call dreads: `p99 latency > 5s`, `error rate climbing`, `DB CPU 99%`. You open the dashboard half-awake, and the picture is unambiguous and useless at the same time: every graph is red, the connection count is climbing toward its ceiling, and the application is starting to throw `503`s because it cannot get a database connection. Somewhere in there, one query — probably one query — is eating the whole machine, and you have a few minutes before the climbing connection count turns a degraded service into a hard outage.

This post is the playbook I wish every team handed its on-call engineers before their first rotation. The thesis is simple and it is the difference between a six-minute blip and a two-hour incident: **a database fire is worked as triage, not as panic.** You do not start by reasoning about root cause. You start by reading vitals, finding the offender, and stabilizing the patient — and only then do you diagnose and fix. The mistake that turns a blip into an outage is reversing that order: opening the codebase to "figure out what changed" while the connection count quietly hits its limit and takes down every other query on the box.

![The on-call triage loop: triage, identify, mitigate, diagnose, fix, prevent](/imgs/blogs/the-slow-query-at-1am-1.webp)

The diagram above is the mental model for the entire post: six ordered phases, left to right. **Triage** (read the vitals, confirm it is the database). **Identify** (find the single offending query). **Mitigate** (cancel or terminate it safely, untangle any locks). Only now do you **Diagnose** (read the plan, find the real root cause), then **Fix** (the index, the rewrite, the stats), and finally **Prevent** (timeouts, alerts, load tests) before you log off. Everything below is a tour of those six boxes, with the exact SQL you run in each, the traps that wait in each, and a stack of war stories where each phase went right or wrong. If you have not read [why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod), keep it open in a tab — half of what we diagnose here is that post's failure modes showing up at 1 AM.

> The first rule of a database fire: the database is almost never broken. A *query* is. Your job is to find which one, stop it without making things worse, and make sure it can never do this again.

## Why a database fire is different from a normal bug

A normal bug has a stable reproduction. You can attach a debugger, add logging, ship a fix on Monday. A database fire is none of those things, and treating it like a normal bug is the single most common way on-call engineers extend an outage. The defining property of a database incident is that **the system is actively getting worse while you investigate**, because the failure mode is a feedback loop, not a static fault.

| Assumption (from normal debugging) | Naive 1 AM reaction | Reality of a database fire |
| --- | --- | --- |
| The system is stable while I investigate | "Let me read the code to see what changed" | Connections pile up every second you wait; the box degrades non-linearly |
| One slow request affects one user | "It's just one slow endpoint" | One slow query holds a connection; enough of them exhaust the pool and take down *every* endpoint |
| CPU at 99% means we need a bigger box | "Page the infra team to scale up" | A bigger box runs the same bad plan slightly faster; you will scale into the same wall in a week |
| The error is in the application | "Roll back the last deploy" | Often right — but rollback without identifying the query means you cannot tell if it worked |
| I should find root cause first | "Why is this query slow?" | Stabilize first. A killed query you do not yet understand beats a healthy theory about a dead service |

The asymmetry is what makes this hard. The *cost of acting* (cancelling a query that turns out to be fine) is usually small and reversible — the user retries. The *cost of not acting* (letting the pool exhaust while you theorize) is a full outage that takes the blast radius from one feature to the entire product. So the playbook is biased toward fast, safe, reversible mitigation, and it defers the satisfying intellectual work of root-cause analysis until after the patient is stable. That single inversion — mitigate before you fully understand — is what separates a calm rotation from a war room.

## The first 90 seconds: read the vitals

**Senior rule of thumb: read four dials before you type a single query, and if they all moved together, you are looking at one root cause, not four problems.** The instinct under stress is to chase whichever graph is reddest. Resist it. The fastest path to the offender is recognizing the *shape* of the incident, and that shape lives in the correlation between a small set of metrics.

![How a 1 AM outage unfolds: deploy at 01:04, latency climbs, page fires, connections max, pool exhausted, then recovery](/imgs/blogs/the-slow-query-at-1am-2.webp)

The timeline above is the canonical escalation, and the dashboard told the whole story eight minutes before the page even fired. Here are the four dials, in the order I read them:

1. **Application latency (p99).** This is what woke you. A jump from 80 ms to multiple seconds is the symptom; it tells you *that* something is wrong, never *what*. Note whether it climbed gradually (a query getting slower as data grows or cache cools) or stepped sharply at a specific minute (a deploy, a cron job, a plan flip).
2. **Error rate.** Distinguish *timeout* errors (requests giving up waiting for the DB) from *connection* errors (`could not obtain a connection`, `too many connections`). Connection errors are the tell that the pool is exhausting — that is the fast-moving emergency.
3. **Database CPU and I/O.** CPU pinned at 99% with low I/O wait usually means a query doing CPU work on rows already in memory — a big sort, a hash join, a sequential scan over a cached table. High I/O wait with moderate CPU means you are reading from disk — a scan over a table too big for the buffer cache. The two point at different fixes.
4. **Active connections.** This is the dial that decides how much time you have. If it is climbing toward `max_connections` (Postgres) or `max_connections` / pool size (your app), you are minutes from a hard outage and you mitigate *now*. If it is flat, you have room to diagnose more carefully.

The correlation is the diagnosis. If latency, errors, CPU, and connections all stepped up at 01:06, you are not hunting four independent gremlins — you are hunting the one thing that happened at 01:06. Check your deploy log and your cron schedule for that timestamp first; a startling fraction of 1 AM fires are a batch job or a deploy that shipped a new query path. And before you blame the database at all, rule out the boring impostors: a runaway application loop hammering the DB, a noisy neighbor on shared hardware, a backup or `VACUUM` saturating I/O, or — the classic — a thundering herd after a [cache expiry](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd) that dumped the read load straight onto the primary.

## Identify the offender

You have confirmed it is the database and you know roughly when it started. Now find the specific query. Every engine answers two questions with two different tools: *what is running right now* (for the live offender holding the box hostage) and *what has cost the most over time* (for the chronic regression that finally crossed a threshold tonight).

![The diagnostic toolmap: Postgres and MySQL each expose a live-activity view and a cumulative-cost view](/imgs/blogs/the-slow-query-at-1am-3.webp)

### PostgreSQL: pg_stat_activity for the live offender

`pg_stat_activity` is your live process list. Filter to `state = 'active'` (queries actually executing, not idle sessions), exclude the system background workers, and sort by how long each has been running. The longest-running active query is your prime suspect.

```sql
-- PostgreSQL: who is running right now, longest-running first
SELECT
  pid,
  now() - query_start            AS runtime,
  state,
  wait_event_type,
  wait_event,
  left(query, 80)                AS query
FROM pg_stat_activity
WHERE state = 'active'
  AND backend_type = 'client backend'
  AND now() - query_start > interval '5 seconds'
ORDER BY runtime DESC;
```

Read three columns hard. `runtime` ranks the suspects. `wait_event_type` tells you *why* a query is slow without reading its plan: `IO` means it is reading from disk (a scan over a cold table); `Lock` means it is blocked behind another transaction (jump straight to the blocking section below); `CPU` (a null wait event on an active query) means it is grinding through rows in memory. And `query` — even truncated — usually names the table and the shape (`SELECT ... FROM reporting_events`) that points you at the offending code path.

### PostgreSQL: pg_stat_statements for the chronic cost

If nothing is dramatically long-running but the box is still pinned, the culprit is volume: a query that is individually fine but is being run thousands of times a second, or one that got slightly slower per call and crossed a tipping point. `pg_stat_statements` aggregates normalized queries by total time, which is exactly the lens you want — *total* time, not mean time, because a 5 ms query run 50,000 times a second hurts more than a 2-second query run twice.

```sql
-- PostgreSQL: which normalized query has burned the most total execution time
SELECT
  queryid,
  calls,
  round(total_exec_time)                                          AS total_ms,
  round(mean_exec_time, 2)                                        AS mean_ms,
  round(100 * total_exec_time / sum(total_exec_time) OVER (), 1)  AS pct_of_total,
  left(query, 70)                                                 AS query
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 15;
```

The `pct_of_total` column is the gold. If one normalized statement is 60% of all execution time on the box, you have found tonight's offender even if no single execution looks alarming. (If `pg_stat_statements` is not installed, install it *now* and add it to `shared_preload_libraries` — it is the single highest-value extension for surviving on-call, and tonight is the night you wished you had it.)

### MySQL: SHOW PROCESSLIST and performance_schema

MySQL's equivalents are `SHOW PROCESSLIST` (or, better, the `information_schema.processlist` view you can filter with SQL) for live queries, and `performance_schema` for cumulative cost.

```sql
-- MySQL: live offenders (filterable; command='Query' excludes idle/sleeping sessions)
SELECT id, user, host, db, time, state, LEFT(info, 80) AS query
FROM information_schema.processlist
WHERE command = 'Query'
  AND time > 5
ORDER BY time DESC;

-- MySQL: cumulative cost by normalized statement digest
SELECT
  DIGEST_TEXT,
  COUNT_STAR                       AS calls,
  ROUND(SUM_TIMER_WAIT / 1e12, 1)  AS total_s,
  ROUND(AVG_TIMER_WAIT / 1e9, 1)   AS avg_ms
FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 15;
```

`time` in the processlist is seconds the query has been in its current state — your live ranking. The `events_statements_summary_by_digest` table is MySQL's `pg_stat_statements`: normalized statements ranked by total wait time. Same two questions, same two tools, different syntax.

## The cascade: how one slow query takes down everything

Before we kill anything, it is worth understanding *why* a single slow query does not stay a single slow problem — because that mechanism is what makes the connection dial the one you watch. A web request that hits a slow query does not politely wait off to the side. It holds its database connection for the entire duration of the query. While it waits, the next request comes in and grabs *another* connection. And the next. The connections do not free up because the thing they are all waiting on — the database — is saturated by the slow query.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="Cascade: one slow query holds its connection, the pool fills slot by slot to its limit, and the app starts returning 503s until the query is killed and connections drain" style="width:100%;height:auto;max-width:840px">
<style>
.sq-q{fill:var(--accent,#6366f1)}
.sq-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.sq-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.sq-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.sq-lblc{font:600 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.sq-cap{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.sq-out{fill:#e5484d}
.sq-fill{fill:#e5484d;transform-box:fill-box;transform-origin:left center}
@keyframes sq-redden{0%,8%{fill:var(--surface,#f3f4f6)}28%,100%{fill:#e5484d}}
@keyframes sq-grow{0%{transform:scaleX(0.02)}12%{transform:scaleX(0.02)}100%{transform:scaleX(1)}}
@keyframes sq-pop{0%,55%{opacity:0}80%,100%{opacity:1}}
@keyframes sq-pulse{0%,100%{opacity:.55}50%{opacity:1}}
.sq-a{animation:sq-redden 9s ease-in-out infinite alternate}
.sq-poolfill{animation:sq-grow 9s ease-in-out infinite alternate}
.sq-outage{animation:sq-pop 9s ease-in-out infinite alternate}
.sq-pulsing{animation:sq-pulse 2.2s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.sq-a,.sq-poolfill,.sq-outage,.sq-pulsing{animation:none}.sq-a{fill:#e5484d}.sq-poolfill{transform:scaleX(1)}.sq-outage{opacity:1}}
</style>
<text class="sq-lbl" x="20" y="36">one slow query</text>
<rect class="sq-q sq-pulsing" x="20" y="118" width="170" height="64" rx="10"/>
<text class="sq-lblc" x="105" y="146">SELECT ... (8.4s)</text>
<text class="sq-lblc" x="105" y="168">holds its connection</text>
<line x1="196" y1="150" x2="232" y2="150" stroke="var(--text-secondary,#6b7280)" stroke-width="2"/>
<polygon points="232,144 244,150 232,156" fill="var(--text-secondary,#6b7280)"/>
<text class="sq-lbl" x="256" y="36">connection pool fills</text>
<rect class="sq-slot sq-a" x="256" y="92"  width="44" height="44" rx="6" style="animation-delay:0s"/>
<rect class="sq-slot sq-a" x="312" y="92"  width="44" height="44" rx="6" style="animation-delay:.5s"/>
<rect class="sq-slot sq-a" x="368" y="92"  width="44" height="44" rx="6" style="animation-delay:1s"/>
<rect class="sq-slot sq-a" x="424" y="92"  width="44" height="44" rx="6" style="animation-delay:1.5s"/>
<rect class="sq-slot sq-a" x="480" y="92"  width="44" height="44" rx="6" style="animation-delay:2s"/>
<rect class="sq-slot sq-a" x="536" y="92"  width="44" height="44" rx="6" style="animation-delay:2.5s"/>
<rect class="sq-slot sq-a" x="256" y="148" width="44" height="44" rx="6" style="animation-delay:.25s"/>
<rect class="sq-slot sq-a" x="312" y="148" width="44" height="44" rx="6" style="animation-delay:.75s"/>
<rect class="sq-slot sq-a" x="368" y="148" width="44" height="44" rx="6" style="animation-delay:1.25s"/>
<rect class="sq-slot sq-a" x="424" y="148" width="44" height="44" rx="6" style="animation-delay:1.75s"/>
<rect class="sq-slot sq-a" x="480" y="148" width="44" height="44" rx="6" style="animation-delay:2.25s"/>
<rect class="sq-slot sq-a" x="536" y="148" width="44" height="44" rx="6" style="animation-delay:2.75s"/>
<rect class="sq-track" x="256" y="212" width="324" height="22" rx="6"/>
<rect class="sq-fill sq-poolfill" x="256" y="212" width="324" height="22" rx="6"/>
<text class="sq-cap" x="256" y="256">pool: 100 / 100 connections checked out</text>
<line x1="586" y1="150" x2="608" y2="150" stroke="var(--text-secondary,#6b7280)" stroke-width="2"/>
<polygon points="608,144 620,150 608,156" fill="var(--text-secondary,#6b7280)"/>
<g class="sq-outage">
<rect class="sq-out" x="624" y="118" width="120" height="64" rx="10"/>
<text class="sq-lblc" x="684" y="146">app: 503</text>
<text class="sq-lblc" x="684" y="168">OUTAGE</text>
</g>
</svg>
<figcaption>One untimed query holds its connection; behind it the pool fills slot by slot to its limit, and the app returns 503s — until the query is killed and the connections drain back out.</figcaption>
</figure>

Watch the loop above. The slow query holds its connection. Behind it, requests stack up, each grabbing a slot until the pool hits its ceiling. At that point the application cannot get a connection at all, so even *fast* queries — your login endpoint, your health check — start failing with `503`s. The database itself may still be mostly idle on those fast queries; the outage is no longer about the slow query, it is about connection starvation. This is why a single missing index can take down a service that has nothing to do with the table that index was missing on. And it is why the kill — releasing that connection — is so often the entire fix: drain one stuck slot and the whole pool recovers in seconds. The mechanics of this pile-up, and why your pool size and timeouts are the real controls, are the subject of [database connection pooling](/blog/software-development/database/database-connection-pooling); tonight, the only thing you need to internalize is that **the connection count is a countdown timer.**

## Mitigate: cancel, terminate, and untangle locks

You have a `pid` (Postgres) or an `id` (MySQL). Now stop it — carefully, because there are two ways to stop a query and choosing wrong can make things worse.

**Senior rule of thumb: cancel before you terminate. Cancelling rolls back one statement and keeps the session; terminating kills the whole connection and any transaction it was holding open.** Reach for the gentle tool first.

```sql
-- PostgreSQL
-- Gentle: cancel the running statement, keep the session alive. Try this first.
SELECT pg_cancel_backend(9002);

-- Forceful: terminate the entire backend (closes the connection, aborts its txn).
-- Use when pg_cancel_backend did nothing, or for an idle-in-transaction lock holder.
SELECT pg_terminate_backend(8821);
```

```sql
-- MySQL
KILL QUERY 9002;        -- cancel the running statement, keep the connection
KILL CONNECTION 8821;   -- terminate the whole connection
```

`pg_cancel_backend` sends the equivalent of `Ctrl-C`: the query receives a cancellation request at its next interrupt check and rolls back. It is safe and usually instant for a query that is actively executing. But it can be ignored by a backend that is stuck deep inside a syscall (some I/O waits, some extension code), and it does nothing for a session that is `idle in transaction` — that session is not running a statement to cancel; it is just *sitting* on locks. For those, `pg_terminate_backend` is the tool: it closes the connection, which forces the transaction to abort and releases every lock it held.

### Find the blocking tree before you kill blindly

If your suspect's `wait_event_type` is `Lock`, the query you see is the *victim*, not the culprit. Killing it just promotes the next waiter to the front of the same queue. You need the blocker — and Postgres hands it to you with `pg_blocking_pids()`.

![The blocking tree: one idle-in-transaction session holds a lock, stalling a chain of waiters and draining the pool](/imgs/blogs/the-slow-query-at-1am-5.webp)

```sql
-- PostgreSQL: the blocking tree — who is waiting on whom, oldest blocker first
SELECT
  waiting.pid                     AS waiting_pid,
  left(waiting.query, 50)         AS waiting_query,
  blocking.pid                    AS blocking_pid,
  blocking.state                  AS blocking_state,
  now() - blocking.xact_start     AS blocker_txn_age,
  left(blocking.query, 50)        AS blocking_query
FROM pg_stat_activity AS waiting
JOIN LATERAL unnest(pg_blocking_pids(waiting.pid)) AS blocker(pid) ON true
JOIN pg_stat_activity AS blocking ON blocking.pid = blocker.pid
WHERE cardinality(pg_blocking_pids(waiting.pid)) > 0
ORDER BY blocker_txn_age DESC;
```

The figure shows the shape this query reveals: PID 8821 is `idle in transaction`, has been for 320 seconds, and is holding a `RowExclusiveLock` that three other sessions are queued behind. Those three waiters are each holding a connection, which is why the pool is at 96/100 and climbing. The fix is not to kill the visible waiters — it is to terminate PID 8821, the root of the tree. An `idle in transaction` blocker is almost always an application bug: code that opened a transaction (`BEGIN`), did some work, and then went off to call an external API or hit a slow code path without committing. Killing it releases the lock; *fixing* it means finding the missing `COMMIT`. We go deep on this failure mode in [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production).

### A break-glass kill script

When the box is on fire you do not want to be hand-copying PIDs at 1 AM. Keep a guarded script in the runbook — conservative by default (dry-run unless told otherwise, a duration floor so you cannot nuke fast queries, and an explicit guard so it never cancels itself).

```bash
#!/usr/bin/env bash
# kill-slow.sh — cancel active queries older than N seconds matching a pattern.
# Defaults are conservative on purpose: this is a break-glass tool, not a cron job.
set -euo pipefail

THRESHOLD="${1:-30}"          # seconds; never cancel anything younger than this
PATTERN="${2:-%}"             # optional ILIKE filter, e.g. '%reporting%'
ACTION="${3:-cancel}"         # cancel | terminate
DSN="${DATABASE_URL:?set DATABASE_URL}"

fn="pg_cancel_backend"
[ "$ACTION" = "terminate" ] && fn="pg_terminate_backend"

echo "About to ${ACTION} active queries > ${THRESHOLD}s matching '${PATTERN}':"
psql "$DSN" -v ON_ERROR_STOP=1 <<SQL
SELECT pid, now() - query_start AS runtime, left(query, 60) AS query
FROM pg_stat_activity
WHERE state = 'active' AND backend_type = 'client backend'
  AND now() - query_start > interval '${THRESHOLD} seconds'
  AND query ILIKE '${PATTERN}'
  AND pid <> pg_backend_pid();
SQL

read -r -p "Type 'yes' to ${ACTION} the rows above: " confirm
[ "$confirm" = "yes" ] || { echo "aborted"; exit 1; }

psql "$DSN" -v ON_ERROR_STOP=1 <<SQL
SELECT ${fn}(pid)
FROM pg_stat_activity
WHERE state = 'active' AND backend_type = 'client backend'
  AND now() - query_start > interval '${THRESHOLD} seconds'
  AND query ILIKE '${PATTERN}'
  AND pid <> pg_backend_pid();    -- never kill the session running this script
SQL
```

The `pid <> pg_backend_pid()` guard is not optional — without it, a broad pattern can match and cancel the very session doing the cancelling, and you spend a confused minute wondering why nothing happened. Default to `cancel`; only pass `terminate` when you have confirmed a lock holder that ignores cancellation.

## Why it was fine in dev

The patient is stable. The connection count is draining, latency is recovering, the page has resolved. Now — and only now — the interesting question: how did a query that passed code review, ran fine in CI, and looked instant in staging bring down production at 1 AM? The answer is almost never "the code is wrong." The answer is that **the query plan that is correct on a thousand rows is catastrophic on fifty million**, and every axis that protected you in dev silently flips in prod.

![Why it was fine in dev: row count, plan, cache, index relevance, and latency all flip between dev and prod](/imgs/blogs/the-slow-query-at-1am-6.webp)

The matrix above is the full list of traps, and each one deserves a sentence because each is a different 1 AM:

- **Data volume.** Your dev database has a thousand orders. Prod has fifty-two million. A sequential scan over a thousand rows is sub-millisecond; the planner does not even bother with the index, and you never notice it is missing. The same scan over fifty-two million rows is 8 seconds.
- **Plan flips.** The planner chooses a plan from statistics about your data. After a bulk import, a partition rotation, or simply enough drift, those statistics go stale, and the planner's row estimate can be off by six orders of magnitude. It picks a nested loop expecting one row and gets two million — a plan that was optimal yesterday is a disaster today. This is the most insidious failure mode because *nothing in your code changed*.
- **Parameter sniffing.** A prepared statement or a cached plan is chosen for the first parameter value it sees. If that value is unrepresentative (a customer with one order), the plan it caches can be terrible for the common case (a customer with millions). MySQL and SQL Server are especially prone; Postgres has its own custom-vs-generic-plan version of this.
- **Cold cache.** In dev, your entire dataset fits in the buffer cache, so every read is a memory hit. In prod, the table is far larger than RAM; a scan has to pull cold pages from disk, and the same query is I/O-bound instead of CPU-bound. The first run after a failover or restart — cold cache — is reliably the slowest.
- **An index that did not matter at 1k rows.** The missing index is invisible in dev precisely *because* the table is small. It becomes load-bearing only at scale, which means the bug ships every time.

This is the same terrain as [why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod), and the practical takeaway for on-call is this: do not trust "it works in staging." Trust the plan, on production-scale data. Which is exactly what we read next.

## Reading the plan under fire

`EXPLAIN` shows the plan the planner *intends* to use; `EXPLAIN (ANALYZE, BUFFERS)` actually runs the query and shows what *happened* — the real row counts, real timings, real buffer reads. Under fire you want `ANALYZE` (on a `SELECT`; be careful running it on writes), and you read it for four specific tells. We cover the full art in [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer); here is the 1 AM subset.

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM orders
WHERE customer_id = $1
ORDER BY created_at DESC
LIMIT 50;
```

Tonight's offender produced this plan:

```
Limit  (cost=10432.55..10432.67 rows=50) (actual time=8401.2..8423.0 rows=50 loops=1)
  ->  Sort  (cost=10432.55..10444.10 rows=4620) (actual time=8401.1..8422.0 rows=2300000 loops=1)
        Sort Method: external merge  Disk: 184320kB
        ->  Seq Scan on orders  (cost=0.00..9201.00 rows=1) (actual time=2.1..7980.4 rows=2300000 loops=1)
              Filter: (customer_id = $1)
              Rows Removed by Filter: 49700000
Planning Time: 0.31 ms
Execution Time: 8423.0 ms
```

Four tells, every one of them screaming:

1. **The sequential scan.** `Seq Scan on orders` reading the whole 52-million-row table to find one customer's orders. With a usable index this would be an `Index Scan` touching a few thousand rows.
2. **The bad row estimate.** `rows=1` in the estimate, `rows=2300000` in the actual. The planner thought this customer had one order; they have 2.3 million. That six-order-of-magnitude miss is *why* it chose the seq scan and the on-disk sort — it planned for one row. A bad estimate is almost always stale statistics: run `ANALYZE` on the table.
3. **The disk sort.** `Sort Method: external merge  Disk: 184320kB` — the sort spilled 180 MB to disk because `work_mem` could not hold 2.3 million rows. The `ORDER BY ... LIMIT 50` should have ridden an index in `created_at` order and never sorted at all.
4. **`Rows Removed by Filter: 49700000`.** The query read 49.7 million rows just to throw them away. Any time "rows removed by filter" dwarfs "rows returned," you are missing an index on the filter column.

In MySQL the equivalents are `EXPLAIN ANALYZE` (8.0.18+) or `EXPLAIN FORMAT=JSON`; you read for the same shapes — `type: ALL` (a full table scan), a `rows` estimate wildly off from reality, and `Using filesort` / `Using temporary`. The grammar differs; the diagnosis is identical.

## The permanent fix

The plan named the disease, so the cures are obvious — and ranked by how durable they are.

![EXPLAIN ANALYZE before and after the index: an 8.4-second seq scan collapses to a 12-millisecond index scan](/imgs/blogs/the-slow-query-at-1am-7.webp)

**1. Add the index.** The single most common permanent fix. The query filters on `customer_id` and orders by `created_at`, so a composite index on both — with `created_at` descending to match the `ORDER BY` — turns the seq-scan-plus-disk-sort into a single index range scan. Build it `CONCURRENTLY` so you do not take an exclusive lock on a hot table (the locking version would *cause* the very outage you are fixing):

```sql
-- Build without an ACCESS EXCLUSIVE lock; takes longer but never blocks writes.
CREATE INDEX CONCURRENTLY idx_orders_customer_created
  ON orders (customer_id, created_at DESC);

-- Refresh planner statistics immediately so it stops guessing.
ANALYZE orders;
```

The figure shows the payoff: the plan node that ran for 8,423 ms collapses to 12 ms because the planner can now *seek* to the right rows instead of *scanning* every row and sorting the survivors. The row estimate also snaps to reality (`est 2.3M / actual 2.3M`), which fixes the cascade of bad downstream choices.

**2. Fix the statistics.** If the plan was bad because the estimate was wrong — not because an index was missing — the fix is `ANALYZE` (or tuning autovacuum/autoanalyze so it keeps up). For columns with skewed distributions, raise the statistics target so the planner samples more:

```sql
ALTER TABLE orders ALTER COLUMN customer_id SET STATISTICS 1000;
ANALYZE orders;
```

If you just bulk-loaded data, run `ANALYZE` as the *last step of the load*, not whenever autovacuum gets around to it. A surprising share of 1 AM plan flips are "we imported 10 million rows at midnight and the stats did not catch up before the morning batch hit." Autovacuum tuning is its own rabbit hole — see [Postgres vacuum, bloat, and autovacuum tuning](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning).

**3. Rewrite the query.** Sometimes no index helps because the query asks for too much: an unbounded result set, a `SELECT *` that drags huge columns through a sort, an `OFFSET 100000` deep-pagination scan, a function on the filter column (`WHERE lower(email) = ...`) that defeats the index. The fix is to make the query ask for less — add a `LIMIT`, replace `OFFSET` with keyset pagination, store a computed column, or push the predicate into a form the index can serve.

**4. Add a guardrail so it can never run forever again.** This is the fix people skip, and it is the one that makes the *next* incident a non-event. `statement_timeout` caps how long any query may run; `idle_in_transaction_session_timeout` kills the lock-holding sessions from the blocking section; `lock_timeout` caps how long a query waits on a lock before giving up.

```sql
-- A reporting role should never run longer than 30 seconds, full stop.
ALTER ROLE reporting SET statement_timeout = '30s';
-- Kill app sessions that sit idle inside a transaction (they hold locks!) after 60s.
ALTER ROLE app       SET idle_in_transaction_session_timeout = '60s';
-- Do not let a query block on a lock for more than 3 seconds.
ALTER ROLE app       SET lock_timeout = '3s';
```

MySQL 8.0 has `max_execution_time` (milliseconds, `SELECT`-only) as a global, a session setting, or a per-query hint:

```sql
SET GLOBAL max_execution_time = 15000;   -- 15s ceiling for SELECTs
-- or, surgically, on the one query you do not trust:
SELECT /*+ MAX_EXECUTION_TIME(2000) */ * FROM orders WHERE customer_id = 42;
```

A `statement_timeout` does not fix a slow query — it converts a *silent outage* into a *loud, bounded error*. That trade is almost always worth it: a query that errors after 15 seconds frees its connection and lets the rest of the service breathe, instead of holding a slot until the pool dies.

## The triage cheat sheet

Tape this to the runbook. It maps the symptom you see on the dashboard to the likely cause, the action you take *right now*, and the fix you ship *this week*.

| Symptom | Likely cause | Triage action (now) | Permanent fix (this week) |
| --- | --- | --- | --- |
| One query running for minutes, CPU pinned | Seq scan / missing index at scale | `pg_cancel_backend(pid)` | `CREATE INDEX CONCURRENTLY`; verify with `EXPLAIN` |
| Many waiters, `wait_event_type = Lock` | `idle in transaction` lock holder | Find blocker via `pg_blocking_pids`, `pg_terminate_backend` the root | Fix the missing `COMMIT`; set `idle_in_transaction_session_timeout` |
| Connections climbing to max, mixed errors | Pool exhaustion from a slow query | Kill the slow query; the pool drains | `statement_timeout`; right-size the pool; add pool/queue timeout |
| Query suddenly slow, no deploy, no code change | Plan flip from stale statistics | `ANALYZE <table>` | Tune autovacuum; raise statistics target; `ANALYZE` after bulk loads |
| Same query fast for some args, slow for others | Parameter sniffing / cached plan | Kill the bad session; force a re-plan | Plan stability tooling; `SET plan_cache_mode`; rewrite to discourage caching |
| High I/O wait, moderate CPU | Cold cache after restart/failover | Let cache warm; throttle the offender | Pre-warm critical tables (`pg_prewarm`); size the buffer cache |
| CPU pinned but *no* slow query visible | Not the database — app loop, leap-second bug, noisy neighbor | Confirm with `pg_stat_activity`; check host metrics | Fix the real cause; do not scale the DB blindly |
| Steady degradation as data grows | A query that was always O(n), n just got big | `pg_stat_statements` to find the top consumer | Index, paginate, or archive cold data |

## Prevention: making 1 AM boring

**Senior rule of thumb: every prevention measure here exists to convert a silent, unbounded failure into a loud, bounded one *before* it pages a human.** None of these is exotic; the reason teams get paged at 1 AM is almost always that one of them was never turned on.

- **`statement_timeout` everywhere.** A global ceiling, tightened per role. Analytics and reporting roles get aggressive timeouts; the OLTP path gets a ceiling that is generous for it but still finite. This single setting prevents the most common outage in this entire post: a query holding a connection forever.
- **The slow-query log.** `log_min_duration_statement = '1s'` in Postgres (or the MySQL slow query log with `long_query_time = 1`) records every query slower than your threshold, with its parameters. This is the difference between "a query was slow last night, no idea which" and a log line you can paste into `EXPLAIN`.
- **Top-query alerting.** Alert on `pg_stat_statements` deltas: if any normalized query's share of total execution time crosses a threshold, or its mean time regresses by 3x week-over-week, page *before* it becomes an outage. Catching the chronic regression at 4 PM on a Tuesday is infinitely cheaper than catching it at 1 AM.
- **Load testing at production scale.** The reason the query was fine in dev is that dev had a thousand rows. The fix is a staging environment with a production-shaped dataset — not production data (privacy), but production *volume and distribution*. Run the new query path against fifty million rows in CI and watch the plan. Most plan-flip outages are catchable here.
- **The "no unbounded query" rule.** Enforced in code review and, ideally, in a linter: every query that can return more than one row has a `LIMIT`; every list endpoint paginates; no `SELECT *` on wide tables in a hot path; no user-controlled `OFFSET` without a ceiling. An unbounded query is a loaded gun that fires the day your data crosses a size you did not test.

The cultural point underneath all of this: **on-call quality is a function of how much you invested before the page.** A team with `statement_timeout`, a slow-query log, `pg_stat_statements` alerting, and load tests treats a 1 AM database event as a five-minute kill-and-go-back-to-sleep. A team without them treats it as a war room. The playbook is the same either way; the difference is entirely in the guardrails.

## War stories from the on-call rotation

### 1. The reporting query with no LIMIT

The symptom: every weekday at exactly 1:00 AM, p99 spiked and the connection count climbed for ten minutes, then recovered on its own. Because it self-resolved, three on-calls in a row acknowledged the page and went back to sleep. The wrong first hypothesis was "the nightly backup is saturating I/O." `pg_stat_statements` told the truth in thirty seconds: a single analytics query — a dashboard refresh that did `SELECT * FROM events WHERE created_at > now() - interval '90 days'` with no `LIMIT` — was pulling tens of millions of rows into the application to count them. It self-resolved because the dashboard cron finished. The fix was a one-line rewrite (`SELECT count(*)` pushed into the database) plus a `statement_timeout` of 60 seconds on the reporting role. The lesson: a self-resolving page is still a page, and "it recovers on its own" is a reason to investigate, not to ignore.

### 2. The plan flip after the bulk load

The symptom: an endpoint that had been fast for a year went to 8 seconds, with no deploy and no code change in the window. The on-call spent forty minutes reading application logs and Git history looking for a change that did not exist. The actual root cause was a data-warehouse sync that had bulk-inserted 12 million rows into a table at midnight; the planner's statistics still reflected the pre-load distribution, so its row estimates were off by a factor of a thousand and it flipped from an index scan to a sequential scan. A single `ANALYZE orders` restored the good plan in under a second. The fix was to make `ANALYZE` the last step of the bulk-load job and to alert on planner estimate error. The lesson: when a query goes slow with no code change, suspect statistics before you suspect a ghost in the codebase.

### 3. The idle-in-transaction lock holder

The symptom: a chain of `UPDATE` statements all stuck in `wait_event_type = Lock`, connections climbing. The on-call's first instinct was to kill the longest-waiting `UPDATE` — which did nothing, because that query was a victim, not the culprit. `pg_blocking_pids()` revealed a session that was `idle in transaction` for over five minutes, holding a row lock. The application had opened a transaction, called a third-party payment API inside it, and the API was timing out — so the transaction sat open holding locks for the full 30-second HTTP timeout, repeatedly. Sentry and other teams running Postgres have publicly documented this exact pattern; it is one of the most common self-inflicted production stalls. The immediate fix was `pg_terminate_backend` on the holder; the real fix was moving the external API call *outside* the transaction and setting `idle_in_transaction_session_timeout`. The lesson: under a `Lock` wait, the query you can see is never the one to kill.

### 4. The N+1 that became a fan-out

The symptom: latency degraded smoothly over three weeks until it finally paged, with no single slow query in `pg_stat_activity` — just thousands of tiny ones. `pg_stat_statements` showed one normalized query at 70% of total execution time, called 40,000 times a second. It was a classic ORM N+1: a page rendering a list of 50 orders, each of which lazily loaded its customer in a separate query, multiplied across a traffic increase. No individual query was slow; the *aggregate* was crushing the box and saturating the connection pool. The triage action was to cache the hot lookup; the permanent fix was an eager-load (`JOIN`) that turned 51 queries into 1. The lesson: "no slow query" does not mean "no query problem" — sort by total time, not by max time, and the death-by-a-thousand-cuts query surfaces immediately.

### 5. The 99% CPU that was not a query at all

The symptom: database CPU pinned at 100%, latency awful — but `pg_stat_activity` showed almost nothing running. This is the one that humbles you. The 2012 leap-second bug is the canonical public example: a kernel/`hrtimer` interaction caused `futex`-heavy processes to spin, and sites including Reddit, Mozilla, and others saw CPU pin across fleets with no application cause — the database process was burning CPU in the kernel, not in any query. The on-call lesson generalizes far beyond that specific bug: before you treat a CPU spike as a query problem, confirm there is actually a query consuming it. If `pg_stat_activity` is quiet while CPU is pinned, the cause is below the database — a kernel issue, a noisy neighbor on shared hardware, a backup process, swapping under memory pressure. Scaling the database, killing queries, or rolling back deploys will all fail to fix a problem that is not in the database. The lesson: the first question in any "database is slow" incident is "is it actually the database?"

### 6. The overload spiral and the danger of acting under pressure

The symptom: a spike in write load (a cleanup job removing a large batch of rows) pushed replication lag up, which triggered alerts, which prompted manual intervention. GitLab's widely-read 2017 database incident post-mortem describes how a load-and-lag situation, worked under pressure in the middle of the night, escalated when a tired engineer ran a destructive command against what turned out to be the wrong node. The database root cause was relatively mundane — load and lag — but the *outage* came from the human response to it. The lesson that the entire industry took from that write-up: under pressure, the most dangerous tool is the one that is hard to undo. Cancel a query (reversible) before you terminate a session; terminate a session before you fail over; fail over before you ever run a manual `DELETE` or `DROP`. And read the hostname twice before you run anything irreversible.

### 7. The pool that was sized for the happy path

The symptom: a slow third-party dependency made a handful of requests take 10 seconds each. Individually harmless — but the connection pool was sized at exactly the steady-state concurrency, with no headroom and no acquisition timeout. The slow requests held their connections; new requests blocked indefinitely waiting to acquire one (no timeout meant they waited forever instead of failing fast), and within ninety seconds every endpoint in the service was hanging on connection acquisition. The database itself was nearly idle. The triage action was to bounce the application to reset the pool; the permanent fix was a connection-acquisition timeout (fail fast in 2 seconds rather than block forever), a circuit breaker on the slow dependency, and a `statement_timeout` so a stuck query could not hold a connection indefinitely. The lesson, again: the connection pool is where a database slowdown becomes an application outage, and an acquisition timeout is the cheapest insurance you can buy. The mechanics are in [database connection pooling](/blog/software-development/database/database-connection-pooling).

## When to reach for the kill switch — and when not to

The playbook biases toward fast, reversible action, but "kill the query" is not the answer to every database page. Knowing when *not* to reach for `pg_terminate_backend` is as important as knowing the syntax.

**Reach for the kill switch when:**

- The connection count is climbing toward its ceiling — the countdown timer is running and a killed query is recoverable while an exhausted pool is not.
- One query is clearly the offender in `pg_stat_activity` (long runtime, CPU or I/O bound) and cancelling it will release a connection the rest of the service desperately needs.
- A session is `idle in transaction` holding locks that a chain of waiters is queued behind — terminate the root of the blocking tree.
- A reporting or batch query is running interactively and starving the OLTP path; cancel it and reschedule it for off-peak.

**Do not reach for the kill switch when:**

- `pg_stat_activity` is quiet but CPU is pinned — the problem is below the database (kernel, neighbor, backup, swap), and killing queries fixes nothing. Diagnose the host first.
- The "slow" query is a critical write mid-transaction whose rollback is more expensive or risky than letting it finish — a long-running migration or a large `UPDATE` may be safer to let complete than to abort and retry.
- You have not yet found the blocker and you are about to kill a *victim* of a lock wait — that just shuffles the queue. Find the root first.
- The real fix is a rollback of a bad deploy. If a deploy at 01:04 introduced the query, rolling it back is faster and more complete than playing whack-a-mole with individual queries — though you still identify the query so you can confirm the rollback worked.

The deepest version of this discipline is the one from the GitLab story: the most dangerous action under pressure is the irreversible one. Order your tools by how hard they are to undo — read a vital, cancel a statement, terminate a session, fail over, restore from backup — and always reach for the leftmost tool that solves the problem. A database fire is won by the engineer who stays in triage mode: stabilize the patient, *then* diagnose, *then* cure, and leave a guardrail behind so the same fire cannot start twice. If tonight's incident also revealed that a single primary simply cannot carry the load anymore, that is a different and longer conversation — start it with [when one database is not enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) and [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree), in daylight, after you have slept.

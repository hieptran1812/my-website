---
title: "PostgreSQL VACUUM, Bloat, and Autovacuum Tuning"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer's deep dive into why Postgres needs VACUUM, how MVCC manufactures bloat, how the autovacuum daemon decides what to clean, and how to tune it for high-write tables before transaction-ID wraparound takes you down."
tags:
  [
    "vacuum",
    "autovacuum",
    "postgres",
    "bloat",
    "mvcc",
    "txid-wraparound",
    "database-operations",
    "performance",
    "tuning",
    "pg-repack",
    "freezing",
    "monitoring",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-1.webp"
---

There is a particular kind of 3 a.m. page that every team running PostgreSQL at scale eventually receives. The application is up. CPU is fine. Replication lag is zero. But writes are failing with a message that looks like a joke: `ERROR: database is not accepting commands to avoid wraparound data loss in database "production"`. The database has not crashed. It has not corrupted anything. It has *deliberately stopped accepting writes* to protect itself from a 32-bit counter that is about to overflow — and the only way out is a maintenance operation that, on a multi-terabyte table, can take days.

This is not a hypothetical. It happened to [Sentry](https://blog.sentry.io/transaction-id-wraparound-in-postgres) on a US working day in 2015. It happened to [Mailchimp's Mandrill](https://mailchimp.com/what-we-learned-from-the-recent-mandrill-outage/) in February 2019, where one shard of a sharded Postgres fleet took roughly 41 hours to recover and degraded transactional email delivery for two days. It is the same force that pushed [Notion](https://www.notion.com/blog/sharding-postgres-at-notion) to shard its monolith into 480 logical shards, and the same reliability ceiling that [Figma](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/) hit when individual tables grew to several terabytes. Every one of these companies knew Postgres well. They were not careless. They were defeated by a maintenance subsystem that is invisible until the day it isn't: **VACUUM**.

The maddening thing is that VACUUM is not exotic. It runs constantly, in the background, on every Postgres instance ever deployed, doing dull and essential work — and it does that work *because of a design decision Postgres made about how to store row versions*. To understand why your database fills with dead weight, why disk usage never goes down even after you `DELETE` half a table, why an idle connection someone left open in a `psql` session can silently break cleanup across the whole cluster, and why the defaults are dangerously timid on a write-heavy table, you have to understand one thing first: in Postgres, an `UPDATE` is not an update. The diagram below is the mental model for the entire article.

## Why VACUUM exists at all

![Every UPDATE and DELETE leaves a dead tuple in the heap; VACUUM marks that space reusable](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-1.webp)

The diagram above is the mental model: read it left to right. On the left, the heap is filling with corpses; on the right, VACUUM has marked that space reusable. Everything else in this article is a consequence of those two columns.

Here is the mechanism. PostgreSQL uses **multi-version concurrency control** (MVCC) so that readers never block writers and writers never block readers. The price of that property is that the database can never overwrite a row in place, because some other transaction with an older snapshot might still need to see the old version. So when you run `UPDATE accounts SET balance = balance - 100 WHERE id = 42`, Postgres does *not* find row 42 and change its bytes. It writes an entirely new physical row — a new **tuple** — with the new balance, stamps it with the current transaction's ID as its `xmin` (the transaction that created it), and stamps the *old* tuple's `xmax` (the transaction that deleted it) with the same ID. The old tuple is now a **dead tuple**: still physically present in the heap, occupying its 8 KB page slot, but invisible to any transaction whose snapshot started after the update committed.

A `DELETE` is even simpler from this lens: it does not remove anything. It sets the target tuple's `xmax` and leaves the bytes exactly where they were. The row is gone *logically* — no query will return it — but it is still *physically* there, still on disk, still counting against the table's size. If you have ever run `DELETE FROM events WHERE created_at < now() - interval '90 days'`, watched it delete 200 million rows, and then watched the table's on-disk size not budge by a single byte, this is why. You did not reclaim space. You manufactured 200 million dead tuples.

This is the heart of [how MVCC works in Postgres versus InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb): Postgres keeps old versions *inline in the heap* and cleans them up later, while InnoDB pushes old versions into a separate undo log and applies them on read. Postgres's choice makes writes cheap and readers cheap, but it defers an unbounded amount of garbage-collection work — and that deferred work is VACUUM. Without it, the heap grows forever. Every update is a small leak; at scale, the small leaks become a flood.

> An `UPDATE` in Postgres is a disguised `INSERT` plus a tombstone. You are not changing a row — you are creating a new one and abandoning the old. VACUUM is the cleanup crew that comes through later and reclaims the abandoned ones.

The analogy I keep coming back to is a library that never reshelves. Every time a patron "edits" a book, the librarian photocopies it, hands the patron the copy, and tosses the original onto the floor next to the shelf. Lookups are fast — the catalog points at the newest copy. Concurrent readers are happy — they can read whatever copy their snapshot pointed at. But the floor fills with discarded books, and eventually you cannot walk through the stacks. VACUUM is the night-shift worker who picks up the discarded books and frees the floor space for new ones. If that worker falls behind, the library doesn't stop working — it just gets slower, fatter, and more expensive, until one specific failure mode (wraparound) slams the doors entirely.

### The relationship to log-structured storage

If you have read about [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), this should feel familiar, because it is the same problem wearing different clothes. In Martin Kleppmann's *Designing Data-Intensive Applications* (Chapter 3), the recurring theme of log-structured storage is that you make writes cheap by *appending* and never updating in place, and you pay for that later with a background **compaction/merging** process that reclaims the space held by overwritten and deleted keys. An LSM engine's SSTables accumulate stale key versions exactly the way a Postgres heap accumulates dead tuples; compaction is the LSM analogue of VACUUM. Both are bets that you can buy back space and read performance with background work, and both fail the same way: if the background process cannot keep up with the foreground write rate, the structure bloats without bound and reads degrade.

The crucial difference — and the reason VACUUM has a failure mode LSM compaction does not — is that Postgres's MVCC scheme is tied to a finite, 32-bit transaction-ID counter (Chapter 7's MVCC mechanics in practice). Compaction falling behind costs you disk and latency. VACUUM falling behind costs you disk and latency *and*, eventually, your ability to write at all. We will get to wraparound; first, let us be precise about what a single VACUUM actually does.

## What plain VACUUM actually does

**A senior rule of thumb: plain `VACUUM` does not lock your table and does not shrink it on disk. It makes dead space reusable, not returned.** Internalize this early, because the most common misconception about VACUUM is that it reclaims disk space back to the operating system. It almost never does. It reclaims space back to the *table*, so future inserts and updates can reuse the slots without growing the file.

![A single VACUUM reclaims dead space, updates two maps, and advances the freeze horizon](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-2.webp)

The figure above shows the four jobs a single VACUUM pass performs. Let us walk through each, because each one is load-bearing and each one is a knob you can monitor.

**1. Mark dead tuples as reusable.** VACUUM scans the heap, finds tuples whose `xmax` is older than the oldest snapshot any transaction could still need (the *xmin horizon*, which we will obsess over later), and marks their line pointers as `LP_DEAD` / reusable. The space is now available for the next `INSERT` or non-HOT `UPDATE` that lands on that page. The file does not shrink — the high-water mark stays where it was — but the dead space is no longer wasted.

**2. Update the free space map (FSM).** Each table has a companion `_fsm` fork that tracks, per page, roughly how much free space is available. After reclaiming dead tuples, VACUUM updates the FSM so the tuple inserter knows which pages have room. A stale or missing FSM is a subtle bloat source: the inserter, not knowing about reclaimed space, appends new pages instead of reusing old ones.

**3. Update the visibility map (VM).** The `_vm` fork has two bits per page: *all-visible* (every tuple on the page is visible to all transactions) and *all-frozen*. The all-visible bit is what makes **index-only scans** possible — if the planner knows a page is all-visible, it can answer a query from the index alone without visiting the heap to check tuple visibility. A table that is never vacuumed has an empty visibility map, which means index-only scans silently degrade into regular index scans with heap fetches. This is a real, measurable performance regression that masquerades as "the query just got slower."

**4. Freeze old tuples and advance `relfrozenxid`.** This is the anti-wraparound job. VACUUM rewrites the `xmin` of very old, definitely-visible-to-everyone tuples to a special **frozen** marker, then advances the table's `relfrozenxid` — the oldest unfrozen transaction ID in the table. Advancing `relfrozenxid` is what keeps the table's transaction-ID *age* from climbing toward the 2-billion wraparound horizon. We will spend an entire section on this because it is the failure mode that takes companies down.

Here is what a verbose VACUUM looks like in practice. Run this on a table you suspect is bloated:

```sql
VACUUM (VERBOSE, ANALYZE) events;
```

```
INFO:  vacuuming "public.events"
INFO:  finished vacuuming "public.events": index scans: 1
pages: 0 removed, 1248732 remain, 1248732 scanned (100.00% of total)
tuples: 18443201 removed, 92117668 remain, 41023 are dead but not yet removable
removable cutoff: 84720113, which was 412 XIDs old when operation ended
new relfrozenxid: 71029944, which is 9 XIDs old
index scan needed: 412095 pages from table (33.00% of total) had 18443201 dead item identifiers removed
avg read rate: 142.118 MB/s, avg write rate: 71.402 MB/s
buffer usage: 2418332 hits, 1248901 misses, 627330 dirtied
WAL usage: 1875204 records, 627330 full page images, 4821934112 bytes
system usage: CPU: user: 18.42 s, system: 9.11 s, elapsed: 71.03 s
```

Read this output like a doctor reads a chart. `18443201 removed` is the dead tuples reclaimed — good, VACUUM did work. `41023 are dead but not yet removable` is the line that should make you sit up: those are dead tuples VACUUM *found* but *could not clean*, because some transaction's snapshot still might need them. A handful is normal. A growing number is a five-alarm fire that means something is pinning your xmin horizon (we will diagnose that). `new relfrozenxid: 71029944, which is 9 XIDs old` tells you freezing happened and the table's age was reset close to current. `avg read rate / write rate` and `WAL usage` tell you how much I/O this VACUUM cost — and on a throttled autovacuum, those rates being low is a symptom of cost-based throttling, which we will tune.

### Second-order optimization: ANALYZE is not optional

`VACUUM (VERBOSE, ANALYZE)` runs `ANALYZE` after the vacuum, refreshing the planner statistics in `pg_statistic`. People treat ANALYZE as a separate, lower-priority chore, but the two are coupled in a way that bites: a table that gets heavily churned — millions of updates — has both dead tuples *and* stale statistics, and stale statistics produce bad query plans (wrong row estimates, wrong join order, sequential scans where an index scan was warranted). The autovacuum daemon actually runs *two* independent triggers, one for vacuuming and one for analyzing, with their own thresholds. A common production mistake is to tune the vacuum trigger aggressively and forget the analyze trigger, leaving a hot table with fresh space but rotten statistics. Tune both.

## VACUUM versus VACUUM FULL versus pg_repack

**A senior rule of thumb: you almost never want `VACUUM FULL` in production. If you need to reclaim space back to the OS on a live table, reach for `pg_repack`.** The naming is one of Postgres's cruelest traps, because `VACUUM FULL` sounds like "VACUUM, but more thorough." It is a completely different operation with completely different lock semantics, and running it casually on a busy table is how you turn a bloat problem into an outage.

![Only VACUUM FULL and pg_repack shrink the file on disk, but only pg_repack avoids an exclusive lock](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-3.webp)

The matrix above lays out the three options. Let us be precise about each.

**Plain `VACUUM`** takes a `SHARE UPDATE EXCLUSIVE` lock, which is weak: it does not block `SELECT`, `INSERT`, `UPDATE`, or `DELETE`. It only conflicts with DDL and with other vacuums on the same table. It does not rewrite the table; it reclaims dead space in place. It does *not* return space to the OS (with one narrow exception: if the dead tuples are all at the physical end of the table, VACUUM can truncate the trailing empty pages — but a single live tuple on the last page prevents this). This is the one you want running constantly.

**`VACUUM FULL`** rewrites the entire table into a brand-new file, packing live tuples densely and discarding all dead space, then swaps the file in and drops the old one. This *does* return space to the OS and produces a perfectly compact table with rebuilt indexes. The catch is the lock: `VACUUM FULL` takes an `ACCESS EXCLUSIVE` lock for the *entire duration*, blocking every read and every write to the table. On a 500 GB table on fast NVMe, that rewrite might take 30–60 minutes; for that whole window, every query touching the table hangs. It also needs free disk space equal to the table size (you have two copies during the rewrite). Reserve it for tables that are offline anyway, or for a tiny table during a real maintenance window.

**`pg_repack`** is the production answer. It is an extension that achieves the same compaction as `VACUUM FULL` — a full table rewrite that returns space to the OS — but *online*. It builds a shadow copy of the table, installs triggers to capture concurrent changes into a log, applies that log to catch up, and then takes a brief `ACCESS EXCLUSIVE` lock only at the very end to swap the tables. Your application keeps reading and writing the whole time; the only blocking moment is the millisecond-scale final swap. The cost is operational complexity (it is a separate binary and extension), extra disk during the rebuild, and the fact that a `pg_repack` run that gets interrupted can leave behind orphaned objects you must clean up. But for a high-churn table that has bloated to 40% dead space, `pg_repack` is how you reclaim the space without an outage.

```bash
# Install the extension (once), then repack one bloated table online.
# Requires the pg_repack extension in shared_preload_libraries and CREATE EXTENSION.
pg_repack \
  --host=db-primary.internal \
  --dbname=production \
  --table=public.events \
  --jobs=4 \
  --no-kill-backend \
  --echo
# --jobs parallelizes index rebuilds; --no-kill-backend makes pg_repack wait
# for conflicting locks rather than killing the queries holding them.
```

Here is the decision table I give every team:

| Situation | Use | Why |
| --- | --- | --- |
| Routine dead-tuple cleanup on a live table | plain `VACUUM` / autovacuum | weak lock, no rewrite, keeps space reusable |
| Reclaim OS disk after a one-time mass delete, table can go offline | `VACUUM FULL` | simplest, no extension, but `ACCESS EXCLUSIVE` |
| Reclaim OS disk on a busy table, zero-downtime required | `pg_repack` | online rewrite, brief lock only at swap |
| Bloated indexes specifically (not the heap) | `REINDEX CONCURRENTLY` | rebuilds index online without repacking heap |
| Table is partitioned and one partition is dead | `DROP`/`DETACH` the partition | instant, no rewrite at all |

### Second-order optimization: partitioning sidesteps the whole question

The last row of that table is the most important and the least used. If your bloat comes from a time-series pattern — events, logs, metrics, audit trails where old rows are deleted en masse — the right answer is often not to vacuum or repack the deletes at all, but to **partition by time and drop whole partitions**. Dropping a partition is a metadata operation: instant, no rewrite, no dead tuples, no vacuum needed. This is the move that connects to [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — the same discipline that lets you change schema without locking also lets you expire data without bloating. A `DELETE` of 200 million rows manufactures 200 million dead tuples and a vacuum bill; `DROP TABLE events_2024_01` manufactures nothing.

## The autovacuum daemon: who decides what to clean

You do not run VACUUM by hand in steady state. The **autovacuum daemon** does it for you, and understanding its scheduling is the difference between a database that quietly stays healthy and one that quietly rots. The daemon has three moving parts: a **launcher** that wakes on a schedule, a set of **workers** that do the actual vacuuming, and a **trigger formula** that decides which tables are eligible.

![The autovacuum launcher polls thresholds every naptime and dispatches eligible tables to a capped throttled worker pool](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-7.webp)

The graph above traces the flow. The **launcher** process wakes every `autovacuum_naptime` (default 1 minute) and, for each database, checks every table against the trigger formula. Tables that exceed their threshold become eligible. The launcher dispatches eligible tables to **workers** — but there are at most `autovacuum_max_workers` of them (default **3**), so on an instance with hundreds of churning tables, work queues up. And every worker is **cost-throttled**: it accumulates a "cost" as it reads and dirties pages, and when it exceeds `autovacuum_vacuum_cost_limit`, it *sleeps* for `autovacuum_vacuum_cost_delay`. That throttle is the single most important and most misunderstood autovacuum behavior, so we will give it its own section.

### The trigger formula

Autovacuum fires a vacuum on a table when the number of dead tuples crosses a threshold computed as:

$$
\text{vacuum threshold} = \texttt{autovacuum\_vacuum\_threshold} + \texttt{autovacuum\_vacuum\_scale\_factor} \times \text{reltuples}
$$

where `reltuples` is the estimated live row count of the table. With the defaults — `autovacuum_vacuum_threshold = 50` and `autovacuum_vacuum_scale_factor = 0.2` — a table is vacuumed once it accumulates `50 + 0.2 × reltuples` dead tuples. For a 1,000-row table, that is 250 dead tuples; reasonable. For a **1-billion-row** table, that is **200,000,050 dead tuples** before autovacuum lifts a finger. Two hundred million dead rows. That is the defining flaw of the default: the scale factor is *proportional*, so the bigger the table, the more bloat you tolerate before acting, and the more painful the eventual vacuum.

There is an exactly analogous formula for ANALYZE:

$$
\text{analyze threshold} = \texttt{autovacuum\_analyze\_threshold} + \texttt{autovacuum\_analyze\_scale\_factor} \times \text{reltuples}
$$

with defaults `50` and `0.1` (10%). Same proportional trap: a billion-row table's statistics go stale by 100 million row-changes before a reanalyze. You can query exactly how close each table is to its trigger:

```sql
SELECT
  relname,
  n_live_tup,
  n_dead_tup,
  (n_dead_tup::float / nullif(n_live_tup, 0)) AS dead_ratio,
  current_setting('autovacuum_vacuum_threshold')::int
    + current_setting('autovacuum_vacuum_scale_factor')::float * n_live_tup
    AS vacuum_threshold,
  last_autovacuum,
  autovacuum_count
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 20;
```

This is the first query you should run on any unfamiliar Postgres instance. `n_dead_tup` greater than `vacuum_threshold` means autovacuum *should* be working on this table right now; if `last_autovacuum` is hours old while `n_dead_tup` keeps climbing, autovacuum is losing the race — either it is throttled too hard, starved of workers, or blocked entirely.

### Why the defaults are too timid for big tables

![Default scale_factor 0.2 lets a billion-row table accumulate 200 million dead tuples before vacuum fires](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-4.webp)

The before/after above is the argument in one picture. On the left, the default `scale_factor = 0.2` on a billion-row table waits for 200 million dead tuples, then triggers one enormous, slow, throttled vacuum that must scan the whole table. On the right, a per-table `scale_factor = 0.01` with a raised cost limit triggers at ~10 million dead tuples and runs frequent, small, cheap vacuums that never let bloat get out of hand.

The defaults are not wrong — they are *conservative for a small, low-traffic database on modest hardware in 2005*. PostgreSQL ships defaults that will not overwhelm a laptop. On a modern server with hundreds of gigabytes of RAM and NVMe storage running a write-heavy workload, they are dramatically too timid, and the symptom is always the same: bloat accumulates on your hottest tables faster than autovacuum is allowed to clean it. The [Snowflake engineering writeup on tuning Postgres vacuum](https://www.snowflake.com/en/blog/engineering/tuning-postgres-vacuum/) and the [Citus "13 tips" debugging guide](https://www.citusdata.com/blog/2022/07/28/debugging-postgres-autovacuum-problems-13-tips/) both make the same point: for large, frequently-updated tables, drop `autovacuum_vacuum_scale_factor` to `0.02` or even `0.002`, so the trigger becomes "vacuum after 2% (or 0.2%) of the table is dead" instead of 20%.

## Cost-based throttling: the brake nobody knew about

**A senior rule of thumb: the default `autovacuum_vacuum_cost_limit` is a speed limit set for a 2005 spinning disk, and on modern hardware it is the single biggest reason autovacuum "can't keep up."** This is the knob that surprises people most, because it is invisible until you measure it.

Autovacuum does not run flat-out. It runs on a budget. As a vacuum worker processes pages, it accumulates cost points: `vacuum_cost_page_hit = 1` (page already in shared buffers), `vacuum_cost_page_miss = 2` (page read from OS cache or disk), `vacuum_cost_page_dirty = 20` (page modified, must be written). When the accumulated cost exceeds `autovacuum_vacuum_cost_limit` (default **200**), the worker sleeps for `autovacuum_vacuum_cost_delay` (default **2 ms** in modern versions; it was 20 ms in older ones).

Do the arithmetic. With `cost_limit = 200` and `cost_delay = 2 ms`, a worker that is mostly dirtying pages (cost 20 each) can dirty about 10 pages, then sleeps 2 ms. Ten 8 KB pages is 80 KB; sleeping 2 ms every 80 KB caps the dirty-write rate at roughly **40 MB/s** — and that budget is *shared across all autovacuum workers combined* (controlled by the autovacuum cost-balancing logic). On hardware that can do gigabytes per second, you have throttled your cleanup crew to a crawl. This is precisely the trap Sentry fell into: their post-incident config zeroed out the delay entirely (`autovacuum_vacuum_cost_delay = 0`) and raised worker count and memory, because the throttle was strangling vacuum exactly when they needed it most.

```sql
-- Inspect the current throttle settings.
SELECT name, setting, unit
FROM pg_settings
WHERE name IN (
  'autovacuum_vacuum_cost_limit',
  'autovacuum_vacuum_cost_delay',
  'autovacuum_max_workers',
  'autovacuum_naptime',
  'vacuum_cost_page_dirty',
  'maintenance_work_mem',
  'autovacuum_work_mem'
);
```

The tuning move is to raise `autovacuum_vacuum_cost_limit` to something like `2000`–`10000` (10×–50× the default) and/or lower `autovacuum_vacuum_cost_delay`, which proportionally raises the I/O budget. The Citus guide recommends raising the cost limit "to a high value (like 10000)"; the Sentry incident config set the delay to `0` outright. The right value depends on your hardware and how much I/O headroom you have during peak, but the default is almost certainly too low if you are reading this article because of a bloat problem.

One subtlety that catches people: raising `autovacuum_max_workers` *alone* does not speed anything up, because the cost budget is balanced *across* workers. Three workers sharing a 200-point budget each get ~67 points; six workers sharing it each get ~33. More workers means more tables vacuumed *concurrently*, but the *total* I/O rate is fixed by the cost limit. If you want autovacuum to go faster, raise the cost limit (or the per-worker cost limit) — not just the worker count. Raise both if you have many hot tables *and* want each one to go fast.

### Second-order optimization: maintenance_work_mem governs index passes

`maintenance_work_mem` (or `autovacuum_work_mem` if set, which overrides it for autovacuum specifically) determines how many dead tuple identifiers a vacuum can hold in memory before it must stop and do a full pass over *every index* on the table. If the buffer fills, VACUUM flushes it by scanning all indexes, empties it, and resumes the heap scan — meaning a single logical vacuum makes *multiple* expensive index scans. On a table with several large indexes, a too-small `maintenance_work_mem` turns one vacuum into five, each re-reading every index. Sentry's recovery config set `maintenance_work_mem = '10GB'` for exactly this reason. A common sane value on a dedicated database server is `1GB`–`2GB` for `maintenance_work_mem`, set higher for the manual vacuums you run during incident recovery. (PostgreSQL 17 changed the dead-tuple storage to a more memory-efficient radix tree, which softens this, but the principle holds: starve the index pass and vacuum gets dramatically slower.)

## Freezing and transaction-ID wraparound

We now arrive at the reason this article opens with a horror story. Everything above is about *space*: dead tuples, bloat, disk. This section is about *time*, and specifically about a 32-bit counter that, if it overflows without VACUUM having done its job, will halt your database.

**A senior rule of thumb: bloat makes your database slow; wraparound makes it stop. Monitor `age(relfrozenxid)` like you monitor disk space, because hitting the wraparound horizon is a self-inflicted outage that takes hours to days to fix.**

![Rising XID age moves the database from healthy through aggressive vacuum into a write-blocking halt at 2 billion](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-5.webp)

The matrix above shows the escalation zones. Here is the mechanism behind them.

Every transaction in Postgres gets a 32-bit **transaction ID** (XID), assigned sequentially. Each tuple records the XID that created it (`xmin`) and the XID that deleted it (`xmax`). Visibility — whether a transaction can see a tuple — is determined by comparing XIDs: "was this tuple created by a transaction that committed before my snapshot?" The problem is that 32 bits gives only ~4.2 billion XIDs, and a busy database burns through XIDs fast (every write transaction consumes one). When the counter overflows, it wraps around to the beginning — and now a tuple created at XID 4,000,000,000 would appear to be *in the future* relative to a fresh transaction at XID 100, making committed data suddenly invisible. That is catastrophic, silent data loss.

Postgres solves this with **modular arithmetic and freezing**. XID comparison is done modulo 2³², treating the space as a circle: for any given XID, roughly 2 billion XIDs are "in the past" and 2 billion are "in the future." To keep old tuples permanently visible no matter how far the counter advances, VACUUM **freezes** them — replacing their `xmin` with a special always-visible marker (in effect, `FrozenTransactionId`) so their visibility no longer depends on XID comparison at all. A frozen tuple is visible forever, regardless of where the XID counter is.

The **age** of a table is how many XIDs have been consumed since its oldest unfrozen tuple was written: `age(relfrozenxid)`. As long as VACUUM keeps freezing old tuples and advancing `relfrozenxid`, the age stays bounded and you are safe. If VACUUM falls behind, the age climbs. The escalation:

- **Age < `autovacuum_freeze_max_age` (default 200 million):** normal operation. Regular lazy vacuums handle freezing opportunistically.
- **Age ≥ `autovacuum_freeze_max_age`:** Postgres triggers an **anti-wraparound autovacuum** on the table — *even if the table is not bloated and even if autovacuum is otherwise disabled*. This vacuum cannot be skipped and will not be canceled by a conflicting lock the way a normal autovacuum is (more on that distinction below). This is Postgres protecting itself.
- **Age approaching ~2 billion (the safety horizon):** Postgres logs increasingly urgent warnings: `WARNING: database "production" must be vacuumed within N transactions`.
- **Age hits the hard limit (~2.1 billion):** Postgres refuses all new write transactions. The error from the intro: `ERROR: database is not accepting commands to avoid wraparound data loss`. You are now in **single-user mode** territory.

### Why companies hit this anyway

If anti-wraparound autovacuum is automatic, how did Sentry, Mandrill, and Notion all hit the wall? Three reasons, all of which appear in their postmortems.

First, **`autovacuum_freeze_max_age` set too high.** Sentry's post-incident analysis noted their previous setting was too aggressive in the wrong direction. If you raise `autovacuum_freeze_max_age` toward its maximum (it can go to 2 billion) to *reduce* anti-wraparound vacuum frequency, you also reduce your safety margin: there is less runway between "freeze vacuum starts" and "hard halt." On a database burning XIDs fast, that runway can evaporate before the freeze vacuum — which must scan the whole table — completes.

Second, **the freeze vacuum cannot keep up.** Mandrill's hottest shard (shard4, which their hashing favored) saw a write spike that consumed XIDs faster than the throttled autovacuum could freeze them. By November 2018 they had observed XIDs climbing to roughly half the limit at peak; they did not act, and in February the spike pushed it over. The anti-wraparound vacuum, once it started on a multi-terabyte table, would have taken an estimated *40 days* at the throttled rate. The throttle (the previous section's cost limit) is the silent culprit: a freeze vacuum that is allowed only 40 MB/s on a 2 TB table simply cannot finish in time.

Third, **the table is simply too big.** Notion and Figma both hit the ceiling where individual tables grew to terabytes with billions of rows, and at that size *any* vacuum — freeze or otherwise — becomes a reliability event in itself: it runs for hours, holds resources, and risks not completing before the next threshold. Their answer was structural: shard the data so no single table is large enough for vacuum to become an existential risk. Notion split into 480 logical shards across 32 physical databases, capping each table at 500 GB; Figma vertically partitioned and then horizontally sharded for the same reason.

### Monitoring and recovering from wraparound

Watch the age relentlessly. This query tells you how close every database is to the cliff:

```sql
SELECT
  datname,
  age(datfrozenxid) AS xid_age,
  2_100_000_000 - age(datfrozenxid) AS xids_until_halt,
  round(100.0 * age(datfrozenxid) / 2_100_000_000, 1) AS pct_to_halt
FROM pg_database
ORDER BY xid_age DESC;
```

And per-table, to find the specific relation dragging the age up:

```sql
SELECT
  c.relname,
  age(c.relfrozenxid) AS xid_age,
  pg_size_pretty(pg_total_relation_size(c.oid)) AS size,
  c.relfrozenxid
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'm', 't')      -- tables, matviews, TOAST
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY age(c.relfrozenxid) DESC
LIMIT 20;
```

If you find a table whose age is climbing toward `autovacuum_freeze_max_age` and autovacuum is not catching up, run an explicit, *unthrottled* freeze vacuum on it before it becomes an emergency:

```sql
-- Force a freeze and disable the cost throttle for this one session,
-- so the vacuum runs at full hardware speed.
SET vacuum_cost_delay = 0;
SET maintenance_work_mem = '8GB';
VACUUM (FREEZE, VERBOSE) the_old_table;
```

And if you have already hit the hard halt — the database refuses writes — the recovery is the procedure Sentry and Mandrill both ran: bring the database up in **single-user mode** (`postgres --single`), where the wraparound protection is relaxed, and run `VACUUM` (or, in true emergencies, the brutal but effective move both companies used: **truncate the offending table** if it holds non-critical, regenerable data, which instantly frees its XIDs and lets the vacuum finish in minutes instead of days). Truncation is a last resort with data loss; both teams accepted minor loss in non-critical features (event rollups for Sentry, search/URL tables for Mandrill) to restore writes immediately.

You can watch a long-running freeze vacuum make progress — crucial during an incident, because Mandrill specifically called out "no clear way to track progress" as part of what made their outage so stressful:

```sql
SELECT
  p.pid,
  p.relid::regclass AS table,
  p.phase,
  p.heap_blks_total,
  p.heap_blks_scanned,
  round(100.0 * p.heap_blks_scanned / nullif(p.heap_blks_total, 0), 1) AS pct_scanned,
  p.num_dead_tuples
FROM pg_stat_progress_vacuum p;
```

That view — `pg_stat_progress_vacuum`, added in Postgres 9.6 — is the single most reassuring thing to have during a wraparound incident. It tells you the phase (`scanning heap`, `vacuuming indexes`, `vacuuming heap`, `cleaning up indexes`) and the percentage scanned, so you can estimate completion instead of staring at a frozen prompt.

### The second wraparound horizon: MultiXact IDs

Here is the trap that catches even teams who monitor XID age diligently: there is a *second*, independent 32-bit counter that can wrap around, and it has its own, separate freeze machinery. When multiple transactions hold a shared lock on the same row at once — the classic case is several concurrent transactions doing `SELECT ... FOR SHARE` or holding foreign-key locks on the same parent row — Postgres cannot store all their XIDs in the single `xmax` slot. Instead it allocates a **MultiXact ID** (a "multixact"), which is a pointer into a separate structure listing all the locking transactions, and stores *that* in `xmax`. MultiXact IDs are also 32-bit, also consumed sequentially, and also subject to wraparound, governed by their own parameters: `autovacuum_multixact_freeze_max_age` (default 400 million) and `vacuum_multixact_freeze_min_age`.

The reason this catches people is that MultiXact consumption is completely decoupled from transaction-ID consumption. A workload heavy on foreign keys and shared row locks — think an e-commerce order system where many concurrent transactions touch the same hot inventory rows, or a job queue where workers lock shared rows — can burn through MultiXacts far faster than plain XIDs, so `age(relminmxid)` climbs while `age(relfrozenxid)` looks healthy. You can hit MultiXact wraparound protection (the same write-halting behavior, with a message about multixact wraparound) while your XID-age dashboard shows green. Monitor both:

```sql
-- Both wraparound horizons side by side; alert when either climbs.
SELECT
  c.relname,
  age(c.relfrozenxid)                       AS xid_age,
  mxid_age(c.relminmxid)                     AS multixact_age,
  pg_size_pretty(pg_total_relation_size(c.oid)) AS size
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY greatest(age(c.relfrozenxid), mxid_age(c.relminmxid)) DESC
LIMIT 20;
```

The same freeze vacuum that advances `relfrozenxid` also advances `relminmxid`, so the *fix* is identical — let VACUUM run, unthrottled if necessary — but the *trigger* is independent, which means a table can be forced into an anti-wraparound vacuum by its MultiXact age even when its plain XID age is nowhere near the threshold. If you operate a foreign-key-heavy or shared-lock-heavy schema and have only ever watched `age(relfrozenxid)`, add `mxid_age(relminmxid)` to your dashboard today. It is the wraparound horizon nobody talks about until it halts their database.

### Why freezing got cheaper over Postgres versions

It is worth knowing how much the freeze story has improved, because it changes how aggressive you need to be. Before Postgres 9.6, every anti-wraparound vacuum had to scan the *entire* table, every time, even pages that had not changed in years — which is exactly why freeze vacuums on multi-terabyte tables took days. Postgres 9.6 introduced the **all-frozen bit** in the visibility map, letting VACUUM skip pages already known to be entirely frozen; a freeze vacuum now scans only the pages that have been modified since the last freeze, turning a multi-day full scan into something proportional to churn. Postgres 14 made autovacuum more eager to advance `relfrozenxid` opportunistically during normal vacuums, and Postgres 16–17 continued tightening freezing heuristics and the dead-tuple storage. The practical upshot: on a modern Postgres with a well-maintained visibility map, freezing is incremental and cheap, *as long as VACUUM is allowed to keep the all-frozen bits current*. The companies that suffered multi-day freeze vacuums were on older versions or had let the visibility map go stale — another reason the routine, never-throttled-into-uselessness autovacuum is your best wraparound insurance.

## Index bloat and rebuilding indexes online

Heaps are not the only thing that bloats. **Indexes bloat too**, and index bloat is sneakier because it does not show up in the heap dead-tuple counts. Every non-HOT update that changes an indexed column inserts a new index entry pointing at the new tuple; the old index entry is not immediately removed. B-tree indexes also suffer from page splits that leave pages half-full. Over months, an index can grow to several times the size it would be if freshly built, slowing every query that uses it and wasting buffer cache on dead entries.

Measure index bloat directly with `pgstattuple`:

```sql
CREATE EXTENSION IF NOT EXISTS pgstattuple;

-- Heap bloat: free_percent and dead_tuple_percent tell you how much is wasted.
SELECT * FROM pgstattuple('public.events');

-- Index bloat: avg_leaf_density below ~70% means the index is bloated.
SELECT * FROM pgstatindex('public.events_created_at_idx');
```

`pgstattuple` scans the whole object so it is expensive on huge tables (there is `pgstattuple_approx` for a sampled estimate), but it is the ground truth: `free_percent` on the heap and `avg_leaf_density` on the index tell you exactly how much space is recoverable. When an index's leaf density drops well below 70%, rebuild it. **The production-safe way is `REINDEX CONCURRENTLY`** (Postgres 12+), which builds a new index alongside the old one and swaps them without blocking writes:

```sql
-- Rebuild a single bloated index online (Postgres 12+).
REINDEX INDEX CONCURRENTLY public.events_created_at_idx;

-- Or every index on a table, online.
REINDEX TABLE CONCURRENTLY public.events;
```

Before Postgres 12, the only options were a blocking `REINDEX` or the classic dance of `CREATE INDEX CONCURRENTLY` a new index, then `DROP INDEX CONCURRENTLY` the old one — which `REINDEX CONCURRENTLY` now automates. `pg_repack` also rebuilds indexes as part of its repack, so if you are repacking the heap anyway you get fresh indexes for free.

| Symptom | Tool | Lock |
| --- | --- | --- |
| Heap has high `free_percent`, table is live | `pg_repack` | brief swap only |
| Heap has high `free_percent`, table can be offline | `VACUUM FULL` | `ACCESS EXCLUSIVE` |
| Index `avg_leaf_density` below 70%, table is live | `REINDEX ... CONCURRENTLY` | none on writes |
| Index bloat + you are repacking heap anyway | `pg_repack` | brief swap only |
| Dead tuples but no OS-space concern | plain `VACUUM` | `SHARE UPDATE EXCLUSIVE` |

## HOT updates and fillfactor: slowing bloat at the source

Everything so far is about *cleaning up* bloat. The best bloat is the bloat you never create, and Postgres has a beautiful mechanism for that: **heap-only tuple (HOT) updates.**

![Lowering fillfactor to 80 lets non-indexed UPDATEs become HOT tuples that need no new index entries](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-6.webp)

The before/after above shows the mechanism. Normally, an `UPDATE` writes the new tuple version somewhere in the table and must add a new entry to *every index* on the table, pointing at the new tuple's location — even indexes on columns you did not change. That index churn is a major bloat source. But Postgres has an optimization: if the updated row does **not change any indexed column**, *and* there is free space **on the same page** as the old tuple, Postgres writes the new version on that same page and links it to the old one with an internal pointer (a HOT chain). Because the new version lives at the same page and is reachable from the existing index entries via that chain, **no index entries are created at all.** The update is a "heap-only tuple" — invisible to indexes. When the old version dies, page-level pruning can reclaim it cheaply without a full vacuum.

The two conditions are the levers. The first — "does not change an indexed column" — argues for not over-indexing columns you frequently update. The second — "free space on the same page" — is controlled by **`fillfactor`**, the percentage of each page Postgres fills on insert. The default is `100` (pack pages full), which leaves *no room* for a same-page update, defeating HOT. Lowering fillfactor to `80` or `90` reserves 10–20% of each page as headroom for future updates to stay HOT:

```sql
-- Reserve 20% of each page for in-place HOT updates on a high-churn table.
ALTER TABLE sessions SET (fillfactor = 80);
-- fillfactor only affects pages written after the change; rewrite to apply now.
VACUUM FULL sessions;   -- or pg_repack for an online rewrite
```

You can verify HOT updates are actually happening with `pg_stat_user_tables`:

```sql
SELECT
  relname,
  n_tup_upd,
  n_tup_hot_upd,
  round(100.0 * n_tup_hot_upd / nullif(n_tup_upd, 0), 1) AS hot_pct
FROM pg_stat_user_tables
WHERE n_tup_upd > 0
ORDER BY n_tup_upd DESC
LIMIT 20;
```

A high `hot_pct` (say, above 80%) on an update-heavy table means HOT is working and you are generating far less index bloat. A low `hot_pct` on a table you *thought* should be HOT-friendly means either you are updating an indexed column or your fillfactor is too tight. The [Cybertec writeup on HOT updates](https://www.cybertec-postgresql.com/en/hot-updates-in-postgresql-for-better-performance/) and [Crunchy Data's fillfactor post](https://www.crunchydata.com/blog/postgres-performance-boost-hot-updates-and-fill-factor) both show benchmarks where lowering fillfactor on an update-heavy table dramatically cuts both heap and index bloat — but Cybertec's pragmatic caveat is worth repeating: if you do not already know you have a HOT problem, often the simpler fix is just to tune autovacuum to run fast on that table, and that is good enough. Fillfactor is a scalpel; reach for it when monitoring shows low `hot_pct` on a hot table.

### Second-order optimization: the trade-off fillfactor makes

Lowering fillfactor is not free. Reserving 20% of every page means your table is ~25% larger on disk at the same row count, which means more pages to read for a sequential scan and more buffer cache consumed. You are trading a permanent ~20% space overhead for a reduction in update-time bloat and index churn. That trade is a clear win on a small-to-medium table with a very high update rate on non-indexed columns (a `sessions` table where you bump `last_seen_at` constantly is the canonical example). It is a poor trade on an append-mostly table that is rarely updated, where the reserved space is pure waste. Match the knob to the workload.

## The failure modes: why VACUUM "isn't working"

This is the section I wish someone had handed me years ago, because "autovacuum is running but the table keeps bloating" is one of the most confusing things to debug. There are three distinct failure modes, and they require completely different fixes.

### Failure mode 1: autovacuum can't keep up

The simplest case: autovacuum *is* running on the table, but the write rate exceeds the cleanup rate. You see `last_autovacuum` updating, but `n_dead_tup` keeps climbing anyway. This is the cost-throttle problem from earlier: the worker is sleeping more than it is working. The fix is to raise the I/O budget (raise `autovacuum_vacuum_cost_limit`, lower `autovacuum_vacuum_cost_delay`), lower the per-table `scale_factor` so vacuums run more often on less garbage, and raise `maintenance_work_mem` so each vacuum makes fewer index passes. This is a tuning problem, and the playbook below solves it.

### Failure mode 2: the xmin horizon is pinned

This is the insidious one, and it is the one that makes engineers think Postgres is broken. The symptom is the `VACUUM VERBOSE` line we flagged earlier: **`N rows are dead but not yet removable`**, with N growing over time, even though VACUUM is running constantly and is *not* throttled.

![An idle-in-transaction backend holds the xmin horizon back so VACUUM keeps recent dead tuples forever](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-8.webp)

The timeline above shows the mechanism. VACUUM can only remove a dead tuple if the tuple is invisible to *every* transaction that could possibly still read it. The boundary is the **xmin horizon**: the oldest XID that any active transaction could still need. A dead tuple newer than the horizon cannot be removed, because some old transaction's snapshot might still see it. Now suppose someone — a developer in a `psql` session, an ORM with autocommit off, a stuck batch job — opens a transaction (`BEGIN`) and *leaves it open*, idle, for two hours. That transaction's snapshot pins the xmin horizon back to when it started. For those two hours, **VACUUM cannot remove a single dead tuple created since that transaction began, anywhere in the database.** Millions of `UPDATE`s pile up dead tuples that VACUUM dutifully visits and dutifully cannot clean. The table bloats while VACUUM runs perfectly. The moment you kill the offending backend, the horizon advances and the next VACUUM reclaims everything at once.

The [pganalyze "5mins" episode on dead tuples not yet removable](https://pganalyze.com/blog/5mins-postgres-autovacuum-dead-tuples-not-yet-removable-postgres-xmin-horizon) walks through this exact diagnosis. There are **four things** that pin the xmin horizon, and you must check all four:

```sql
-- 1. Long-running / idle-in-transaction backends.
SELECT pid, state, age(backend_xmin) AS xmin_age,
       now() - xact_start AS txn_duration, query
FROM pg_stat_activity
WHERE backend_xmin IS NOT NULL
ORDER BY age(backend_xmin) DESC
LIMIT 10;

-- 2. Replication slots holding back the horizon (inactive replicas, stuck CDC).
SELECT slot_name, active, age(xmin) AS xmin_age,
       age(catalog_xmin) AS catalog_xmin_age
FROM pg_replication_slots
ORDER BY age(xmin) DESC NULLS LAST;

-- 3. Standby with hot_standby_feedback = on running long queries.
--    (Check pg_stat_replication on the primary; the standby's backend_xmin
--     propagates back and pins the primary's horizon.)
SELECT application_name, backend_xmin, age(backend_xmin) AS xmin_age
FROM pg_stat_replication
ORDER BY age(backend_xmin) DESC NULLS LAST;

-- 4. Orphaned prepared (two-phase commit) transactions.
SELECT gid, prepared, age(transaction::text::xid) AS xmin_age
FROM pg_prepared_xacts
ORDER BY prepared;
```

The four causes — long-running/idle transactions, abandoned or lagging replication slots, standby queries with `hot_standby_feedback`, and orphaned prepared transactions — are the canonical "VACUUM won't remove dead rows" list that Cybertec, pganalyze, and the Citus guide all converge on. The fixes:

- **Idle transactions:** set `idle_in_transaction_session_timeout` (e.g. `'5min'`) so Postgres terminates backends that hold a transaction open without working, and `statement_timeout` to cap runaway queries. These two settings alone prevent the most common version of this incident.
- **Replication slots:** drop unused slots with `pg_drop_replication_slot()`; set `max_slot_wal_keep_size` so a dead slot cannot pin the horizon (or fill the disk) indefinitely.
- **`hot_standby_feedback`:** decide whether you would rather have replication conflicts on the standby or bloat on the primary. With it on, long analytics queries on the replica pin the primary's horizon.
- **Prepared transactions:** if you do not use distributed transactions, `max_prepared_transactions = 0`; if you do, monitor `pg_prepared_xacts` and `ROLLBACK PREPARED` orphans.

### Failure mode 3: autovacuum is repeatedly canceled

The third mode: autovacuum starts on a table, then gets *canceled* before it finishes, over and over, so the table never gets fully cleaned. The cause is lock conflict. A normal (non-anti-wraparound) autovacuum holds a `SHARE UPDATE EXCLUSIVE` lock, and when another process needs a conflicting lock — `ALTER TABLE`, `CREATE INDEX` (non-concurrent), `TRUNCATE`, or even an automatic lock from certain operations — autovacuum *politely yields*: it cancels itself so it does not block your DDL. On a table with frequent DDL (or an ORM/migration tool that touches it constantly, or a monitoring tool that runs `ANALYZE` or grabs locks), autovacuum can be canceled on nearly every attempt and make no progress. You will see log lines like `canceling autovacuum task` or `ERROR: canceling statement due to conflict`.

The Citus guide names this precisely: "if a particular table has DDL running on it almost all the time, a vacuum might not be able to acquire the needed locks." The key insight is that **a manual `VACUUM` does *not* yield this way** — it holds its lock and waits — so the workaround during an incident is to run VACUUM by hand. The longer-term fix is to stop the lock churn: batch your DDL, stop running unnecessary `ANALYZE`s, and audit any tool grabbing locks on the hot table. Note that anti-wraparound autovacuum does *not* yield — once a table crosses `autovacuum_freeze_max_age`, its freeze vacuum will hold its lock and block your DDL instead, which is its own surprise (a migration that mysteriously hangs because a freeze vacuum is in progress).

## Monitoring: the dashboard that catches all three

You cannot tune what you cannot see, and the three failure modes above are invisible without the right signals. The monitoring discipline is to watch *five independent things* together, because no single metric tells the whole story.

![Healthy vacuum needs dead-tuple ratio, last-vacuum age, XID age, bloat estimate, and xmin lag all watched together](/imgs/blogs/postgres-vacuum-bloat-and-autovacuum-tuning-9.webp)

The grid above is the dashboard. The five signals:

1. **Dead-tuple ratio** (`n_dead_tup / n_live_tup` from `pg_stat_user_tables`): the primary bloat signal. Alert when it exceeds, say, 20% on a tuned table.
2. **Last-vacuum age** (`now() - last_autovacuum`): is autovacuum even touching this table? A hot table whose `last_autovacuum` is hours old is a red flag for failure mode 1 or 3.
3. **XID age** (`age(relfrozenxid)` vs. 2 billion): the wraparound countdown. Alert at 50% (~1 billion) so you have weeks of runway, page at 75%.
4. **Bloat estimate** (`pgstattuple.free_percent` / `avg_leaf_density`): ground-truth wasted space, for deciding when to `pg_repack` or `REINDEX`.
5. **xmin horizon lag** (`age(backend_xmin)` from `pg_stat_activity`, plus slots and standbys): the early warning for failure mode 2. Alert when any backend's `backend_xmin` age exceeds, say, 100 million.

Plus `pg_stat_progress_vacuum` for live visibility during incidents. Here is a single consolidated health-check query I keep in my back pocket:

```sql
SELECT
  st.relname,
  st.n_live_tup,
  st.n_dead_tup,
  round(100.0 * st.n_dead_tup / nullif(st.n_live_tup, 0), 1) AS dead_pct,
  age(c.relfrozenxid)                                        AS xid_age,
  st.last_autovacuum,
  st.autovacuum_count,
  st.n_tup_hot_upd,
  round(100.0 * st.n_tup_hot_upd / nullif(st.n_tup_upd, 0), 1) AS hot_pct,
  pg_size_pretty(pg_total_relation_size(c.oid))             AS total_size
FROM pg_stat_user_tables st
JOIN pg_class c ON c.oid = st.relid
WHERE st.n_live_tup > 100000
ORDER BY st.n_dead_tup DESC
LIMIT 25;
```

One query, all the signals: bloat (`dead_pct`), wraparound (`xid_age`), autovacuum activity (`last_autovacuum`, `autovacuum_count`), and HOT efficiency (`hot_pct`). Wire the underlying metrics into Prometheus via `postgres_exporter` and alert on the thresholds above. The companies that survived their vacuum incidents all added exactly this kind of monitoring *after* the fact — Mandrill explicitly noted "we had new alerting in place for XID wraparound" within hours of recovery. Do it before.

## The autovacuum tuning playbook for high-write tables

Here is the concrete, ordered playbook for a high-churn table. Do not apply these globally; apply per-table, because the whole point is that your hot tables need different settings than your cold ones.

**Step 1 — Identify the hot tables.** Run the consolidated query above. The tables with the highest `n_dead_tup` and `n_tup_upd` are your targets. Do not tune everything; tune the 5–10 tables that dominate write traffic.

**Step 2 — Lower the per-table scale factor.** Make autovacuum trigger on a small fraction of dead tuples rather than 20%:

```sql
ALTER TABLE events SET (
  autovacuum_vacuum_scale_factor   = 0.01,   -- vacuum at 1% dead, not 20%
  autovacuum_vacuum_threshold      = 1000,   -- floor for small phases
  autovacuum_analyze_scale_factor  = 0.01,   -- keep stats fresh too
  autovacuum_analyze_threshold     = 1000
);
```

For very large tables (hundreds of millions of rows and up), go further: `autovacuum_vacuum_scale_factor = 0.002`. The Citus guide explicitly recommends `0.02` or `0.002` for billion-row tables.

**Step 3 — Raise the per-table cost limit so the throttle does not strangle it.** Per-table `autovacuum_vacuum_cost_limit` and `autovacuum_vacuum_cost_delay` override the global throttle for that table's autovacuums:

```sql
ALTER TABLE events SET (
  autovacuum_vacuum_cost_limit = 2000,   -- 10x the default I/O budget
  autovacuum_vacuum_cost_delay = 2       -- ms; or 0 to remove the throttle
);
```

**Step 4 — Raise the global memory and worker budget if you have many hot tables.** These are cluster-wide (set in `postgresql.conf` and reload):

```ini
# postgresql.conf — for a dedicated server with plenty of RAM
autovacuum_max_workers        = 6        # more concurrency for many hot tables
autovacuum_naptime            = 15s      # check thresholds more often
maintenance_work_mem          = 2GB      # fewer index passes per vacuum
autovacuum_work_mem           = 1GB      # autovacuum-specific override
vacuum_cost_limit             = 2000     # global default if not per-table
```

**Step 5 — Reduce bloat creation with fillfactor where HOT applies.** If the table is updated heavily on *non-indexed* columns and `hot_pct` is low, reserve page headroom:

```sql
ALTER TABLE sessions SET (fillfactor = 85);
-- then pg_repack to apply to existing pages
```

**Step 6 — Protect the xmin horizon globally.** Prevent failure mode 2 cluster-wide:

```ini
# postgresql.conf
idle_in_transaction_session_timeout = 5min
statement_timeout                   = 0      # set per-role/per-query, not globally
```

**Step 7 — Keep the wraparound margin honest.** Do not raise `autovacuum_freeze_max_age` to its maximum to "reduce vacuum noise"; a moderate value (the default 200 million, or up to ~1 billion on a well-monitored, fast-vacuuming system) keeps your runway. Monitor `age(relfrozenxid)` and react before the anti-wraparound vacuum becomes an emergency.

**Step 8 — Verify it worked.** After a day, re-run the consolidated query. `dead_pct` should be low and stable on the tuned tables, `last_autovacuum` should be recent (minutes, not hours), and `autovacuum_count` should be climbing steadily — many small vacuums, not one giant one.

> Tuning autovacuum is not about making it run less to "save resources." It is about making it run *more often on less garbage*, at a high enough I/O budget to finish, so it never falls behind. A database where autovacuum is constantly busy doing small jobs is a healthy database. A database where autovacuum is idle is either truly idle or quietly drowning.

## Case studies from production

### 1. Sentry and the table that wouldn't vacuum (2015)

Sentry's database stopped accepting writes for most of a US working day when transaction-ID wraparound protection kicked in. Their write-heavy workload had outpaced a throttled autovacuum, and the age of one relation — a table mapping events to rollups — climbed to the halt threshold. The wrong first instinct in this situation is to restart the database immediately; Sentry resisted that, fearing an interrupted vacuum would only extend downtime, and let autovacuum run with their existing config. When the rollup table still would not vacuum after hours, they made the pragmatic call to **truncate it**, accepting minor data loss in a non-critical feature to restore writes at once. The lesson is in their post-incident config: `autovacuum_vacuum_cost_delay = 0` (remove the throttle), `autovacuum_max_workers = 6`, `maintenance_work_mem = '10GB'`, `autovacuum_freeze_max_age = 500000000` — every change aimed at letting freeze vacuums run *fast* and *early*. They also upgraded hardware (256 GB RAM, 24 cores), because at their write rate the freeze vacuum needed real I/O to keep up. The deeper lesson: a throttle tuned for safety in steady state becomes a liability in the exact emergency it should help with.

### 2. Mandrill's hot shard and the 41-hour outage (2019)

Mailchimp's Mandrill ran a sharded Postgres K/V store. Their hashing favored one shard (shard4), making it hotter than the rest, and a write spike on a Sunday night consumed XIDs faster than the throttled autovacuum could freeze them — pushing that shard into wraparound protection. The outage began at 05:35 UTC on February 4 and was not resolved until 22:09 UTC on February 5, roughly **41 hours**, during which they delivered only 80% of queued transactional email. The brutal detail: an initial vacuum at the throttled rate would have taken an estimated *40 days* on the multi-terabyte shard. They had even seen the warning signs — by November 2018, XIDs were climbing to ~50% of the limit at peak — but did not prioritize action. Recovery came, as it did for Sentry, from truncating large non-critical tables (Search and Url, each on the order of TB) to free XIDs and let the vacuum finish in about an hour. Within hours of recovery they had "new alerting in place for XID wraparound." Two lessons: uneven sharding concentrates risk on one shard, and the cost throttle that protects steady-state performance can make emergency recovery effectively impossible on a large table.

### 3. Notion shards the elephant (2021)

Notion's Postgres monolith hit an inflection point when VACUUM began to *consistently stall* — it could no longer reclaim disk space from dead tuples fast enough — and, more alarmingly, the prospect of TXID wraparound emerged as what they called an "existential threat." Years of growth had accumulated billions of blocks, files, and spaces, generating millions of dead tuples faster than a single database's vacuum could process. Rather than chase ever-more-aggressive vacuum tuning on an unboundedly growing table, Notion went structural: they sharded into **480 logical shards across 32 physical databases**, with a cap of 500 GB per table and 10 TB per physical database. By bounding the size of any single table, they bounded the cost of any single vacuum, taking wraparound off the table as a risk entirely. The lesson is that vacuum tuning has a ceiling: past a certain table size, *no* configuration makes vacuum reliable, and the only fix is to keep tables small — through sharding or partitioning.

### 4. Figma's vacuum reliability ceiling

Figma's database stack grew nearly 100x from 2020, and certain tables reached several terabytes and billions of rows. At that scale they "began to see reliability impact during Postgres vacuums" — the vacuums essential for keeping the database from "running out of transaction IDs and breaking down" became reliability events in their own right, running for hours and risking non-completion. Figma's response was a phased one: **vertical partitioning** first (splitting groups of related tables onto separate databases), which they describe as a high-impact, relatively easy lever that bought significant runway, followed by horizontal sharding via a custom DBProxy query router. The vacuum angle is the quiet driver behind the headline sharding story: a big reason "this table is too large" matters is that vacuum — and specifically freeze vacuum — does not scale gracefully past terabytes. Keeping each shard small keeps vacuum boring, which is exactly what you want vacuum to be.

### 5. The idle psql session that bloated everything

A team I worked with had a `users` table bloating steadily despite a perfectly-tuned, never-throttled autovacuum that was visibly running every few minutes. `pg_stat_user_tables` showed `last_autovacuum` updating constantly, yet `n_dead_tup` climbed all day. The `VACUUM VERBOSE` output held the answer: `4.2 million rows are dead but not yet removable`. The culprit was a single `psql` session a developer had opened the previous afternoon, run one `BEGIN; SELECT ...`, and walked away from — an `idle in transaction` backend pinning the xmin horizon for 18 hours. Every dead tuple created in those 18 hours, across the *entire database*, was unremovable. Killing the backend (`SELECT pg_terminate_backend(pid)`) advanced the horizon, and the next autovacuum reclaimed millions of tuples in one pass. The permanent fix was a one-line config change: `idle_in_transaction_session_timeout = 5min`. This is the single most common version of failure mode 2, and the cheapest to prevent.

### 6. The replication slot nobody cleaned up

A different team migrated off a logical-replication-based CDC pipeline but never dropped the replication slot it had created. The slot sat there, `active = false`, with an ancient `xmin` — pinning the primary's horizon (and, separately, retaining WAL until the disk nearly filled). Autovacuum ran cleanly and still could not remove recent dead tuples; the `pg_replication_slots` query showed a slot with an `xmin_age` of over 800 million. The fix was a single `SELECT pg_drop_replication_slot('old_cdc_slot')`, after which the horizon jumped forward and vacuum caught up overnight. The lesson: every replication slot — for replicas, for CDC, for logical decoding — is a potential xmin anchor and a potential disk-filler. Audit `pg_replication_slots` regularly and set `max_slot_wal_keep_size` so a dead slot can be cut loose automatically.

### 7. The migration that hung on a freeze vacuum

A routine `ALTER TABLE ... ADD COLUMN` migration on a large, old table hung indefinitely in staging, then in production, blocking the deploy. The table was not under heavy load, so lock contention from user traffic was ruled out. The actual cause: an **anti-wraparound autovacuum** was in progress on that table (its age had crossed `autovacuum_freeze_max_age`), and unlike a normal autovacuum, a freeze vacuum does *not* yield its lock to incoming DDL — so the `ALTER TABLE` queued behind it. On a multi-hundred-GB table, the freeze vacuum took 90 minutes; the migration sat blocked the whole time. The lesson runs counter to intuition: normal autovacuum politely cancels itself for your DDL, which trains you to expect autovacuum never to block you — but the *anti-wraparound* variety is the exception, and it appears precisely on your oldest, most-likely-to-be-migrated tables. Check `pg_stat_progress_vacuum` before assuming a hung migration is a lock from user traffic.

### 8. The mass DELETE that didn't free a byte

A team ran a retention cleanup — `DELETE FROM audit_log WHERE created_at < now() - interval '1 year'` — that removed 340 million rows and then watched, baffled, as the table's on-disk size stayed at 1.2 TB. They had deleted a third of the table and reclaimed nothing. This is the canonical MVCC misunderstanding from the opening: `DELETE` only sets `xmax`; the dead tuples remain until vacuum, and even then plain VACUUM only makes the space *reusable*, not *returned to the OS*. The space would slowly be reused by new inserts, but the file would never shrink. The right fix had two parts. Short-term: `pg_repack public.audit_log` to rewrite the table online and return ~400 GB to the filesystem. Long-term: convert `audit_log` to a time-partitioned table so that next year's retention is `DROP TABLE audit_log_2025` — instant, zero dead tuples, zero vacuum. The lesson connects directly to [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations): the data-lifecycle design that avoids bloat is the same partitioning discipline that avoids migration pain.

### 9. The over-indexed table that defeated HOT

A `sessions` table updated its `last_activity_at` column on nearly every request — millions of updates an hour — and bloated relentlessly despite aggressive autovacuum tuning. The `hot_pct` from `pg_stat_user_tables` told the story: it was near *zero*. The table had an index on `last_activity_at` (added long ago for an analytics query nobody ran anymore), which meant every update to that column *was an indexed-column update*, disqualifying it from HOT and forcing a new index entry on every single write. Dropping the unused index flipped `hot_pct` to over 95% overnight, and the bloat that aggressive vacuuming had been frantically cleaning up simply stopped being created. Then, lowering fillfactor to 85 to guarantee same-page headroom kept it there. The lesson: HOT is one of Postgres's best anti-bloat features, and a single unnecessary index on a frequently-updated column silently disables it. Audit indexes on hot columns, and check `n_tup_hot_upd` to confirm HOT is actually firing.

### 10. The "we turned autovacuum off" disaster

A team running a latency-sensitive workload noticed autovacuum occasionally caused I/O spikes during peak hours and made the fateful decision to *disable* autovacuum globally (`autovacuum = off`), planning to "run VACUUM manually during off-hours." The manual vacuum cron job worked for a while. Then it was disabled during an incident and forgotten. Three weeks later, the database hit wraparound protection and refused writes — the exact outcome autovacuum exists to prevent. The recovery was an emergency single-user-mode vacuum. The lesson, echoing the [Citus "autovacuum is not the enemy"](https://www.citusdata.com/blog/2016/11/04/autovacuum-not-the-enemy/) post: turning autovacuum off does not solve I/O spikes, it defers them into a catastrophe. If autovacuum's I/O is the problem, *tune the throttle and scale factor* — make it run more often with smaller, smoother jobs — never disable it. Even with `autovacuum = off`, Postgres will still force anti-wraparound vacuums, but by then you have already lost control of the timing.

### 11. GitLab's autovacuum I/O balancing act

GitLab's infrastructure team documented a real tension that every large Postgres operator eventually faces: an analysis found autovacuum consuming a disproportionate share of read I/O (around 75%) while doing comparatively little write I/O, prompting a debate about whether to throttle or reschedule it away from the busiest hours. This is the legitimate version of the temptation that destroyed the team in case 10 — but GitLab's framing was right: the question is *when* and *how hard* autovacuum runs, never *whether*. The nuanced answer for a large fleet is per-table tuning (aggressive on the hottest tables, relaxed on cold ones), a cost limit set high enough to finish but balanced against peak-hour headroom, and monitoring to confirm the trade is working. The lesson is that "autovacuum uses too much I/O" is a tuning conversation, and the correct output of that conversation is a *configuration*, not an *off switch*.

### 12. The append-only table with the wrong fillfactor

A team, having read that lowering fillfactor reduces bloat, applied `fillfactor = 70` to *every* table in a misguided global sweep — including a large, append-only `events` table that was essentially never updated. The result was a 40%+ increase in that table's on-disk size and a measurable slowdown in its sequential scans (more pages to read for the same rows), with zero bloat benefit because there were no updates to make HOT. They had paid the full fillfactor tax for none of the benefit. The fix was to set the append-only tables back to `fillfactor = 100` and reserve the low fillfactor for genuinely update-heavy tables. The lesson: fillfactor is a per-workload scalpel, not a global setting. Reserve page headroom only where updates can actually use it, and measure `n_tup_hot_upd` to confirm the headroom is being spent on HOT updates rather than wasted.

## When to reach for each tool, and when not to

### Reach for aggressive per-table autovacuum tuning when

- A table takes heavy `UPDATE`/`DELETE` traffic and `n_dead_tup` climbs faster than the default 20% trigger tolerates — lower its `scale_factor` to 0.01–0.002.
- `VACUUM VERBOSE` shows low read/write rates and the table bloats despite autovacuum running — raise the per-table `cost_limit` or drop `cost_delay`; the throttle is the bottleneck.
- A vacuum makes multiple index passes (visible as repeated index scans) — raise `maintenance_work_mem` / `autovacuum_work_mem`.
- `age(relfrozenxid)` is creeping toward `autovacuum_freeze_max_age` on a big table — run a manual, unthrottled `VACUUM (FREEZE)` before it becomes an anti-wraparound emergency.
- You operate many hot tables and workers queue up — raise `autovacuum_max_workers` *and* the cost limit together.

### Reach for pg_repack / REINDEX CONCURRENTLY when

- `pgstattuple` shows high `free_percent` (heap) or low `avg_leaf_density` (index) on a table that must stay online — `pg_repack` for the heap, `REINDEX CONCURRENTLY` for indexes.
- A one-time mass delete left a permanently oversized file you need to return to the OS, with no maintenance window — `pg_repack`.

### Reach for fillfactor and de-indexing when

- A table is updated heavily on non-indexed columns but `hot_pct` is low — lower `fillfactor` to 80–90 and drop unnecessary indexes on the churned columns to enable HOT.

### Skip these and do something else when

- **Your bloat is time-series deletes.** Do not tune vacuum to chew through mass deletes — partition by time and `DROP` old partitions. Dropping a partition creates zero dead tuples and needs zero vacuum.
- **Your table has grown past a few terabytes.** No vacuum configuration makes vacuum reliable at that size. Shard or partition so each table stays small (Notion's 500 GB cap, Figma's vertical-then-horizontal split). Tuning buys time; structure buys reliability.
- **You are tempted to disable autovacuum.** Never. If its I/O is the problem, smooth it with cost-based throttling and per-table tuning. Disabling it defers a manageable cost into a wraparound outage.
- **You reach for `VACUUM FULL` on a live, busy table.** Don't — it takes `ACCESS EXCLUSIVE` for the whole rewrite. Use `pg_repack` for online repacking; reserve `VACUUM FULL` for tables that are already offline or trivially small.
- **`N dead but not yet removable` is growing.** No amount of autovacuum tuning will help, because the problem is not the cleaner — it is the xmin horizon. Find and kill the long transaction, drop the dead replication slot, or roll back the orphaned prepared transaction first.

VACUUM is the quiet machinery that makes Postgres's MVCC promise — readers never block writers — affordable. It is dull by design, and when it is healthy you will never think about it. The entire discipline of running Postgres at scale is keeping it dull: small tables, frequent small vacuums, a high enough I/O budget to finish, an unpinned xmin horizon, and an eye on `age(relfrozenxid)` so the 32-bit counter never sneaks up on you. Get those right and vacuum stays invisible. Get them wrong and, one day at 3 a.m., it will become the only thing you can see.

## Further reading

- [MVCC deep dive: Postgres vs. InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) — why Postgres keeps old versions in the heap (and thus needs VACUUM) while InnoDB uses an undo log.
- [LSM trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) — compaction as the log-structured analogue of VACUUM, and the same "background work buys back space" bet.
- [Zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — partitioning and online table changes that sidestep bloat and the anti-wraparound-vacuum-blocks-DDL trap.
- [Sentry: Transaction ID wraparound in Postgres](https://blog.sentry.io/transaction-id-wraparound-in-postgres) — the canonical incident writeup, with the exact recovery config.
- [Mailchimp: What we learned from the Mandrill outage](https://mailchimp.com/what-we-learned-from-the-recent-mandrill-outage/) — the 41-hour sharded-Postgres wraparound outage.
- [Notion: Herding elephants — lessons from sharding Postgres](https://www.notion.com/blog/sharding-postgres-at-notion) and [Figma: How our databases team lived to tell the scale](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/) — why vacuum reliability forced both to shard.
- [Citus: Debugging Postgres autovacuum problems — 13 tips](https://www.citusdata.com/blog/2022/07/28/debugging-postgres-autovacuum-problems-13-tips/) and [pganalyze: dead tuples not yet removable](https://pganalyze.com/blog/5mins-postgres-autovacuum-dead-tuples-not-yet-removable-postgres-xmin-horizon) — the failure-mode field guide.
- [Cybertec: HOT updates for better performance](https://www.cybertec-postgresql.com/en/hot-updates-in-postgresql-for-better-performance/) and [Crunchy Data: HOT updates and fill factor](https://www.crunchydata.com/blog/postgres-performance-boost-hot-updates-and-fill-factor) — the fillfactor/HOT tuning evidence.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapters 3 (storage and retrieval; compaction) and 7 (transactions; MVCC) — the conceptual grounding for why deferred garbage collection is fundamental to both LSM engines and Postgres.

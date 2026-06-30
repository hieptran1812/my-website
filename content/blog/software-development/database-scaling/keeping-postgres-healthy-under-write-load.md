---
title: "Keeping Postgres Healthy Under Write Load: Vacuum, Bloat, and Wraparound"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Under heavy writes, Postgres's MVCC quietly fills your tables with dead tuples; vacuum, bloat control, and freezing are the line between a healthy database and one that grinds to a halt or shuts down at transaction-ID wraparound."
tags: ["postgresql", "vacuum", "autovacuum", "mvcc", "database-bloat", "transaction-id-wraparound", "hot-updates", "write-heavy", "database-scaling", "reliability"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 30
---

There is a particular kind of database incident that never shows up as a single bad query. Write throughput looks fine. CPU is unremarkable. Then, over weeks, three things creep: disk usage climbs faster than the data you can account for, the same queries get slower for no reason the planner can explain, and one morning a `WARNING: database "prod" must be vacuumed within 38000000 transactions` appears in the logs that nobody is reading. If you ignore it long enough, Postgres stops accepting writes entirely to protect itself, and you are now running a multi-hour recovery on a database that, by every dashboard you had, looked healthy.

That whole failure chain comes from one design decision: Postgres uses **multi-version concurrency control (MVCC)**, and MVCC means writes never overwrite in place. Every `UPDATE` and every `DELETE` leaves a corpse — a dead row version that still occupies space until something comes along and reclaims it. That "something" is vacuum. Under light load you never think about it because autovacuum keeps up silently. Under heavy write load, vacuum and bloat management stop being background hygiene and become the single thing standing between you and a database that slowly grinds to a halt, or hits transaction-ID wraparound and refuses to run.

![The three fates of a write-heavy Postgres table: every UPDATE makes dead tuples, and whether autovacuum and freezing keep pace decides health, bloat, or a wraparound shutdown.](/imgs/blogs/keeping-postgres-healthy-under-write-load-1.webp)

The diagram above is the mental model for the whole post. A write-heavy table manufactures dead tuples as a byproduct of normal operation. From there, exactly one variable decides your fate: whether the cleanup machinery keeps pace. If autovacuum keeps up, the table holds a steady size and scans stay fast. If autovacuum falls behind, you get table and index bloat — slower everything. And if *freezing* specifically falls behind, you walk into transaction-ID wraparound, where Postgres stops accepting writes to avoid corrupting visibility. The rest of this article is a tour of those three branches: what produces dead tuples, how to measure the damage, how vacuum and autovacuum reclaim them, how to design writes so there are fewer corpses in the first place, and how to find the long transaction that quietly sabotages the whole thing.

## Why write load is a different game than read load

Most "make Postgres fast" advice is about reads: add an index, fix the query plan, add a replica, add a cache. None of that addresses the write side, because the write side has a completely different failure model. Reads degrade *gracefully* — a missing index makes a query slow but correct. Write-health problems degrade *cliff-style* — everything is fine until it abruptly is not.

| Common assumption | The naive view | The reality under write load |
| --- | --- | --- |
| "An `UPDATE` modifies a row." | It overwrites the old value in place. | It writes a brand-new tuple and marks the old one dead; the old version lingers on disk. |
| "`DELETE` frees space." | Rows are gone, table shrinks. | Deleted rows become dead tuples that still occupy pages until vacuum reclaims them; the table does not shrink. |
| "Autovacuum handles it." | It is automatic, so I can ignore it. | Defaults target a 1 GB toy database; on a hot 500 GB table they let 20% of rows go dead before acting. |
| "Bigger disk solves bloat." | Just add storage. | Bloat pollutes the buffer cache and inflates every scan and index; you pay in latency, not just bytes. |
| "Transaction IDs are basically infinite." | 32 bits is plenty. | About 4.2 billion total, and Postgres can only see half of them at once; a busy database burns through that in weeks. |

> The senior rule of thumb: on a write-heavy Postgres database, you do not "have" autovacuum — you *operate* it. Treating it as a default you never touch is the most reliable way to schedule an outage you cannot see coming.

Everything below is the operational detail behind that rule. If you want the deep mechanics of how vacuum is implemented internally, this post leans on and frequently links to the companion deep-dive on [Postgres vacuum, bloat, and autovacuum tuning](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning); here the lens is specifically *surviving sustained write load in production*.

## 1. MVCC, in one page: every write leaves a body

To reason about dead tuples you need just enough MVCC to be dangerous. The full treatment, including how Postgres and InnoDB differ, is in the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb); here is the compressed version.

Postgres stores every row version — a *tuple* — directly in the table's heap pages. Each tuple carries two hidden system columns: `xmin`, the transaction ID (xid) that created it, and `xmax`, the xid that deleted or superseded it. A fresh `INSERT` writes a tuple with `xmin` set to the inserting transaction and `xmax` empty (effectively infinity). An `UPDATE` does *not* edit the existing tuple: it writes a new tuple with a new `xmin`, and sets the old tuple's `xmax` to the updating transaction. A `DELETE` just stamps `xmax`. The old version is still physically on the page — it is now a **dead tuple**, waiting to be reclaimed.

![What each reader sees: a row's old versions stay on the page until no active snapshot could still need them, which is exactly what vacuum waits for.](/imgs/blogs/keeping-postgres-healthy-under-write-load-2.webp)

Why keep the corpse around at all? Because other transactions may still need to see it. Every statement (or transaction, under `REPEATABLE READ`) runs against a **snapshot**: a notion of which transactions count as committed-and-visible. In the figure, reader A took its snapshot at xid 120, so it sees version `v1` (created at 100, deleted at 150) as live — for reader A, the row still has its old value. Reader B started at xid 210 and sees `v3`, the live version. The two readers see different versions of the same row at the same instant, with no locking. That is the entire point of MVCC: readers never block writers and writers never block readers.

The cost is that `v1` and `v2` cannot be removed until *no* running snapshot could still need them — concretely, until the oldest active transaction's `xmin` has advanced past the point where those versions matter. Hold that idea. The "oldest active snapshot" is the single most important number for write health, and we will return to it as the **xmin horizon** in the section on long transactions. For now, the takeaway: under heavy `UPDATE`/`DELETE` load, dead tuples are produced continuously, and they are *only* removable once the readers move on.

## 2. Bloat: the tax you pay to read dead rows

**Bloat is the space occupied by dead tuples (and empty slots) that has not been reclaimed for reuse.** It is not just wasted disk. It is a tax on every operation that touches the table.

![Bloat is dead tuples you keep paying to read: unreclaimed versions inflate every table scan, index, and buffer-cache page until vacuum frees the slots for reuse.](/imgs/blogs/keeping-postgres-healthy-under-write-load-3.webp)

Consider what happens when 85% of a table's tuples are dead, as on the left of the figure. The table has grown to several times its live size on disk. A sequential scan must read all those dead pages anyway — the dead tuples are interleaved with live ones, so there is no way to skip them. Worse, when Postgres pulls pages into the shared buffer cache, it caches dead rows alongside live ones, so your precious cache memory is now full of garbage and your effective cache hit ratio collapses. Index scans suffer too: indexes accumulate their own dead entries, so an index that should be 2 GB is 12 GB, and walking it touches more pages. Backups grow. WAL volume grows. Replication lag grows. Bloat is a latency problem wearing a disk-usage costume.

The first job is to *measure* it. The cheap, always-available signal is the statistics collector:

```sql
-- Dead-tuple ratio per table, worst offenders first.
SELECT
    schemaname || '.' || relname            AS table,
    n_live_tup,
    n_dead_tup,
    round(100 * n_dead_tup
          / nullif(n_live_tup + n_dead_tup, 0), 1) AS dead_pct,
    last_autovacuum,
    autovacuum_count
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC
LIMIT 25;
```

`n_dead_tup` is an estimate maintained by the stats collector, not an exact count, but it is the number autovacuum itself uses to decide when to act, so it is exactly the right thing to alert on. When you need a precise, authoritative measurement of a specific table — say, to decide whether a `REINDEX` or rewrite is worth the disruption — reach for the `pgstattuple` extension:

```sql
CREATE EXTENSION IF NOT EXISTS pgstattuple;

-- Exact dead-tuple and free-space percentages (scans the whole relation).
SELECT * FROM pgstattuple('public.events');
--  tuple_count | dead_tuple_count | dead_tuple_percent | free_percent | ...

-- Cheaper sampled estimate for very large tables:
SELECT * FROM pgstattuple_approx('public.events');

-- Index bloat specifically:
SELECT * FROM pgstatindex('public.events_pkey');
```

`pgstattuple` reads every page, so it is heavy on a large table — run it off-peak or use the approximate variant. The dashboard query above is what you watch continuously; `pgstattuple` is what you run when the dashboard says something is wrong and you want the truth. A table sitting at 20–30% dead steady-state is usually fine — that is just the working set of versions between vacuums. A table climbing past 50% and not coming back down means vacuum is losing the race, which is the next section.

## 3. Vacuum and autovacuum: the only thing that reclaims

**Vacuum is the only mechanism that turns dead tuples back into reusable space.** Nothing else does it — not `DELETE`, not a restart, not time. If vacuum does not run, dead tuples accumulate forever.

A plain `VACUUM` (not `FULL`) does four jobs in one pass over the table. It scans for dead tuples whose `xmax` is older than the xmin horizon and marks their line pointers reusable. It records the freed space in the table's **free space map** so future inserts and non-HOT updates can refill it. It updates the **visibility map**, flagging pages where every tuple is visible to everyone (which lets future vacuums and index-only scans skip them). And it **freezes** old tuples to hold off wraparound — the subject of the next section. Crucially, plain vacuum does *not* return space to the operating system; it makes the space reusable *inside* the table. That is the right behavior under steady write load: you want the freed slots refilled by new versions, not handed back to the OS only to be re-extended minutes later.

The animation below is the heartbeat of a healthy write-heavy table: dead tuples accumulate as writes arrive, then an autovacuum pass sweeps through and marks the space reusable, returning the table to its working size before the cycle repeats.

<figure class="blog-anim">
<svg viewBox="0 0 640 230" role="img" aria-label="A table's pages fill with dead tuples as writes arrive, then an autovacuum sweep reclaims the space and the pages are reusable again" style="width:100%;height:auto;max-width:760px">
<title>Dead tuples accumulate under write load, then a vacuum sweep reclaims them</title>
<style>
.v4-frame{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.v4-slot{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.v4-dead{fill:#ef4444;transform-box:fill-box;transform-origin:left center;opacity:1}
.v4-reclaim{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center;opacity:0}
.v4-broom{fill:var(--accent,#6366f1);opacity:0}
.v4-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.v4-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.v4-phaseA{font:700 16px ui-sans-serif,system-ui;fill:#ef4444;text-anchor:middle;opacity:1}
.v4-phaseB{font:700 16px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle;opacity:0}
@keyframes v4-dead{0%{transform:scaleX(0);opacity:1}33%{transform:scaleX(1);opacity:1}85%{transform:scaleX(1);opacity:1}92%{opacity:0}100%{transform:scaleX(0);opacity:0}}
@keyframes v4-reclaim{0%{transform:scaleX(0);opacity:0}45%{transform:scaleX(0);opacity:0}48%{opacity:1}80%{transform:scaleX(1);opacity:1}88%{opacity:1}95%{opacity:0}100%{transform:scaleX(0);opacity:0}}
@keyframes v4-broom{0%{transform:translateX(0);opacity:0}45%{transform:translateX(0);opacity:0}48%{opacity:1}80%{transform:translateX(508px);opacity:1}86%{opacity:0}100%{transform:translateX(0);opacity:0}}
@keyframes v4-phaseA{0%{opacity:1}38%{opacity:1}45%{opacity:0}90%{opacity:0}96%{opacity:1}100%{opacity:1}}
@keyframes v4-phaseB{0%{opacity:0}45%{opacity:0}50%{opacity:1}85%{opacity:1}92%{opacity:0}100%{opacity:0}}
.v4-dead{animation:v4-dead 9s ease-in-out infinite}
.v4-reclaim{animation:v4-reclaim 9s ease-in-out infinite}
.v4-broom{animation:v4-broom 9s ease-in-out infinite}
.v4-phaseA{animation:v4-phaseA 9s ease-in-out infinite}
.v4-phaseB{animation:v4-phaseB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.v4-dead,.v4-reclaim,.v4-broom,.v4-phaseA,.v4-phaseB{animation:none}}
</style>
<text class="v4-lbl" x="320" y="30">one table's heap pages</text>
<rect class="v4-slot" x="40"  y="90" width="64" height="70"/>
<rect class="v4-slot" x="106" y="90" width="64" height="70"/>
<rect class="v4-slot" x="172" y="90" width="64" height="70"/>
<rect class="v4-slot" x="238" y="90" width="64" height="70"/>
<rect class="v4-slot" x="304" y="90" width="64" height="70"/>
<rect class="v4-slot" x="370" y="90" width="64" height="70"/>
<rect class="v4-slot" x="436" y="90" width="64" height="70"/>
<rect class="v4-slot" x="502" y="90" width="64" height="70"/>
<rect class="v4-dead" x="40" y="90" width="526" height="70"/>
<rect class="v4-reclaim" x="40" y="90" width="526" height="70"/>
<rect class="v4-broom" x="38" y="84" width="10" height="82"/>
<rect class="v4-frame" x="40" y="90" width="526" height="70" rx="4"/>
<text class="v4-phaseA" x="320" y="200">UPDATE / DELETE leave dead tuples</text>
<text class="v4-phaseB" x="320" y="200">autovacuum sweep marks slots reusable</text>
<text class="v4-sub" x="320" y="222">the cycle repeats every autovacuum run</text>
</svg>
<figcaption>Under write load dead tuples fill the pages (red); an autovacuum pass sweeps left to right and marks the space reusable, returning the table to its working size.</figcaption>
</figure>

You almost never run vacuum by hand. The **autovacuum daemon** launches worker processes that do it for you. The question that decides everything is *when* it triggers. For dead-tuple cleanup, autovacuum acts on a table when:

```
n_dead_tup  >  autovacuum_vacuum_threshold
             + autovacuum_vacuum_scale_factor * reltuples
```

with defaults `threshold = 50` and `scale_factor = 0.2`. Read that carefully. The scale factor is **20% of the table's live row count**. On a 1,000-row table, autovacuum fires after ~250 dead rows — fine. On a 100-million-row events table, it waits for **20 million dead tuples** before lifting a finger. By the time it triggers, you already have catastrophic bloat, the vacuum has a mountain to chew through, and it runs long enough to interfere with everything else. The default is calibrated for a small database. On a large hot table it is actively harmful.

The fix is per-table tuning. Override the scale factor down to a flat-ish trigger and let autovacuum run small and often:

```sql
-- A large, update-heavy table: vacuum after ~5k dead rows + 2% of the table,
-- and give this table's worker a much larger I/O budget so it finishes.
ALTER TABLE events SET (
    autovacuum_vacuum_scale_factor   = 0.02,   -- 2% instead of 20%
    autovacuum_vacuum_threshold      = 5000,
    autovacuum_vacuum_cost_limit     = 4000,   -- default is 200
    autovacuum_vacuum_cost_delay     = 2       -- ms; lower = more aggressive
);

-- PG13+: also vacuum insert-mostly tables to set the visibility map
-- (helps index-only scans and pre-empts a giant freeze later).
ALTER TABLE events SET (autovacuum_vacuum_insert_scale_factor = 0.05);
```

Two more global levers matter under write load. First, the **cost-based throttle**: autovacuum deliberately pauses (`autovacuum_vacuum_cost_delay`) after doing a budget of work (`autovacuum_vacuum_cost_limit`) so it does not saturate your disks. On modern NVMe, the conservative default budget means autovacuum literally cannot keep up with a fast-writing table — it spends most of its time asleep. Raising `autovacuum_vacuum_cost_limit` to a few thousand (globally or per-table) is the most common high-impact change. Second, **worker count**: `autovacuum_max_workers` defaults to 3. If you have dozens of busy tables, three workers serialize and the queue backs up; bump it to 5–10 on a write-heavy host. The companion [autovacuum tuning post](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning) has the full parameter-by-parameter walkthrough.

When you suspect a vacuum is in flight but losing the race, watch it live rather than guessing. `pg_stat_progress_vacuum` reports exactly what each worker is doing and how far along it is:

```sql
-- What is autovacuum doing right now, and how far through is it?
SELECT p.pid,
       a.query,                       -- shows '(to prevent wraparound)' when anti-wraparound
       p.phase,
       pg_size_pretty(p.heap_blks_total * 8192)    AS table_size,
       round(100.0 * p.heap_blks_scanned
             / nullif(p.heap_blks_total, 0), 1)     AS pct_scanned
FROM pg_stat_progress_vacuum p
JOIN pg_stat_activity a USING (pid);
```

If you see a vacuum that has been crawling at 4% for an hour, the cost throttle is almost certainly the bottleneck — and you can raise `autovacuum_vacuum_cost_limit` and reload without killing the running vacuum, which will speed it up mid-flight.

The failure mode to internalize: autovacuum **falls behind** when the trigger is set so high it acts late, the cost limit is so low it acts slowly, or a worker is busy elsewhere. On a hot table, all three conspire. You will not see an error — you will see `n_dead_tup` climbing on the dashboard and `last_autovacuum` getting stale. That is your early warning, weeks before anything breaks.

## 4. The wraparound emergency

This is the one that takes the whole database down, so it gets its own section. Postgres transaction IDs are **32-bit** — about 4.29 billion values. Visibility is decided by comparing xids, but with only 32 bits the comparison is *modular*: at any moment, roughly 2.1 billion xids are treated as "in the past" (and thus potentially visible) and 2.1 billion as "in the future." As the current xid advances, that window slides. The danger is that a very old tuple whose `xmin` falls more than ~2.1 billion transactions behind would suddenly be reinterpreted as being *in the future* — invisible — and your committed data would vanish.

![Transaction-ID age and the wraparound wall: rising age forces autovacuum at 200M, then warnings, then a hard write stop near two billion.](/imgs/blogs/keeping-postgres-healthy-under-write-load-5.webp)

Postgres prevents this with **freezing**. When vacuum encounters a tuple older than `vacuum_freeze_min_age` (default 50M transactions), it marks the tuple *frozen* — flagged as unconditionally visible regardless of any xid comparison. A frozen tuple is permanently safe; its age no longer matters. The number that determines your wraparound risk is therefore not the current xid but the **age of the oldest unfrozen tuple** in each table, tracked as `age(relfrozenxid)`. The figure shows the escalation as that age rises:

- **Below ~50M:** normal freezing during routine vacuums keeps age low. No action needed.
- **At `autovacuum_freeze_max_age` (200M default):** autovacuum forces an *anti-wraparound* vacuum on the table — even if autovacuum is disabled, even if there are zero dead tuples. This is the safety net. It is also why disabling autovacuum globally is a footgun: you can suppress dead-tuple cleanup, but you cannot suppress the anti-wraparound vacuum, and people who "turn off autovacuum for performance" are shocked when it kicks in anyway — or worse, they block it and march toward the wall.
- **Near 2.1 billion, minus a margin:** Postgres logs escalating `WARNING: database "X" must be vacuumed within N transactions` lines. These start tens of millions of transactions before the cliff.
- **At roughly 2.1 billion, with about a million transactions of headroom left:** Postgres refuses to assign new transaction IDs. The database stops accepting writes. This is not a slowdown; it is a full stop to prevent data loss.

There is one more threshold worth knowing because it controls vacuum *cost*, not just safety. Routine vacuums skip pages the visibility map marks as all-frozen, which makes them cheap — but skipping pages means `relfrozenxid` advances slowly. When a table's age crosses `vacuum_freeze_table_age` (default 150M), the next vacuum becomes **aggressive**: it scans every page, including the all-frozen ones, to fully advance the table's frozen xid. An aggressive vacuum on a huge cold table is expensive and can surprise you with a sudden I/O spike. The defense is to not let tables sit untouched until they trip this threshold — `autovacuum_vacuum_insert_scale_factor` (PG13+) ensures even insert-only tables get vacuumed periodically, spreading the freezing work out instead of saving it all for one giant aggressive pass.

Here is the query that tells you exactly where you stand. Run it on a schedule and alert on it:

```sql
-- Per-database wraparound headroom (higher age = closer to the wall).
SELECT datname,
       age(datfrozenxid)                            AS xid_age,
       2^31 - age(datfrozenxid)                     AS xids_remaining
FROM pg_database
ORDER BY xid_age DESC;

-- Per-table: which relations are dragging the cluster's age up?
SELECT c.oid::regclass                              AS table,
       age(c.relfrozenxid)                          AS xid_age,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS size
FROM pg_class c
WHERE c.relkind IN ('r', 'm', 't')          -- tables, matviews, TOAST
ORDER BY age(c.relfrozenxid) DESC
LIMIT 25;
```

If `xid_age` on any table is creeping toward 200M, your freezing is not keeping up and you should investigate *now*, while it is a tuning problem and not an outage. The usual culprits are the same ones that cause bloat — a throttled or starved autovacuum, or (most often) a long-running transaction holding the horizon back so freezing cannot advance.

**Recovery, if you hit the wall.** First, try to triage without stopping the world. Even when writes are refused, a superuser can often still run vacuum. Find the oldest tables from the query above and freeze them explicitly, oldest first:

```sql
VACUUM (FREEZE, VERBOSE) events;     -- repeat for the worst offenders
```

If the database is fully locked into single-user mode (the classic disaster), you stop the server and run vacuum from the single-user backend, which bypasses the write block:

```bash
postgres --single -D /var/lib/postgresql/data prod
# then at the backend> prompt:
backend> VACUUM (FREEZE);
```

The recovery is mechanical, but it is slow — vacuuming hundreds of gigabytes while production is down is a multi-hour, career-defining afternoon. The entire discipline of this section exists so you never run that command on a live incident.

## 5. HOT updates: keep the churn off your indexes

Everything so far has been about cleaning up dead tuples. The complementary strategy is to *create fewer of them* — and specifically, to keep update churn from bloating your indexes. The mechanism is **HOT**: heap-only tuples.

![HOT updates keep version churn off the indexes: an update that changes no indexed column and fits the same page avoids new index entries entirely.](/imgs/blogs/keeping-postgres-healthy-under-write-load-6.webp)

Normally an `UPDATE` is doubly expensive. It writes a new heap tuple (often on a new page), *and* because indexes point at physical tuple locations, it must insert a new entry into **every** index on the table, even indexes on columns that did not change. Those index entries are themselves versioned, so heavy updates bloat indexes as badly as they bloat the heap — sometimes worse, because index bloat is harder to reclaim.

HOT eliminates the index half of that cost. When an `UPDATE` satisfies two conditions — it changes **no indexed column**, and the new tuple **fits on the same page** as the old one — Postgres performs a heap-only update. The new version is written to the same page and chained from the old tuple via a redirect line pointer. Because no indexed value changed, the indexes still point at the original line pointer, which now redirects to the current version. No index entries are created. Better still, the dead intermediate versions in a HOT chain can be cleaned by *page-level pruning* on ordinary access, without waiting for a full vacuum.

You design for HOT with two decisions:

```sql
-- 1. Leave free space on each page so updated rows fit in place.
--    fillfactor 85 reserves 15% of every page for new versions.
ALTER TABLE sessions SET (fillfactor = 85);
-- (existing rows take effect as pages are rewritten; or VACUUM FULL / pg_repack once)

-- 2. Do NOT index columns that change on every write.
--    An index on last_seen_at forces every heartbeat UPDATE to be non-HOT.
DROP INDEX IF EXISTS sessions_last_seen_at_idx;
```

The first reserves room on the page (the default `fillfactor` of 100 leaves none, so an updated row often cannot fit and must move). The second is the one teams get wrong constantly: adding an index on a frequently-updated column — `updated_at`, `last_seen_at`, a status flag, a counter — silently disables HOT for *every* update to that table, because now an indexed column changes every time. Measure your HOT ratio and watch it like a vital sign:

```sql
-- HOT update ratio per table; want this high (90%+) on update-heavy tables.
SELECT schemaname || '.' || relname AS table,
       n_tup_upd,
       n_tup_hot_upd,
       round(100.0 * n_tup_hot_upd / nullif(n_tup_upd, 0), 1) AS hot_pct
FROM pg_stat_user_tables
WHERE n_tup_upd > 100000
ORDER BY n_tup_upd DESC;
```

A table doing millions of updates with a HOT ratio near zero is a table whose indexes are quietly exploding. The fix is almost always one of the two levers above: raise free space, or stop indexing the volatile column.

## 6. Long transactions: vacuum's quiet enemy

We can now close the loop on the idea from the MVCC section. Vacuum can only reclaim a dead tuple once *no* running transaction could still need to see it. The boundary is the **xmin horizon**: the oldest `xmin` among all active transactions, replication slots, and prepared transactions. Vacuum removes dead tuples whose `xmax` is older than the horizon; anything newer must be kept.

![A long transaction freezes the xmin horizon: while one old transaction stays open, vacuum frees nothing deleted after it began, across the whole database.](/imgs/blogs/keeping-postgres-healthy-under-write-load-7.webp)

Now picture a single transaction that opened three hours ago and is just sitting there — the dreaded `idle in transaction` state, usually an application that ran `BEGIN`, did one query, and then went off to call an external API without committing. Its snapshot pins the horizon at the xid it started with. As shown above, vacuum still runs on schedule, but it can only reclaim tuples to the left of that frozen horizon — anything deleted *after* the long transaction began is off-limits, because that ancient transaction might, in principle, still query it. Dead tuples pile up not just on the table that transaction touched but **across the entire database**, because the horizon is global. You will see autovacuum running constantly and accomplishing nothing; `n_dead_tup` climbs everywhere at once. This is the signature of a held-back horizon, and it is the most common cause of mysterious cluster-wide bloat.

Critically, a held horizon also blocks *freezing*, so a long-enough transaction is a direct path to the wraparound wall from the previous section. One forgotten `BEGIN` can, over days, take the whole cluster down.

Find the offenders:

```sql
-- Transactions holding the horizon back, oldest snapshot first.
SELECT pid,
       state,
       now() - xact_start            AS txn_age,
       now() - state_change          AS in_state_for,
       age(backend_xmin)             AS xmin_age,
       left(query, 60)               AS query
FROM pg_stat_activity
WHERE backend_xmin IS NOT NULL
ORDER BY age(backend_xmin) DESC
LIMIT 20;

-- Replication slots are silent horizon holders: a dead CDC consumer
-- pins xmin/catalog_xmin and blocks vacuum even with no live sessions.
SELECT slot_name, active, age(xmin) AS xmin_age, age(catalog_xmin) AS cat_age
FROM pg_replication_slots
ORDER BY age(xmin) DESC NULLS LAST;

-- Prepared (two-phase) transactions left dangling do the same.
SELECT gid, prepared, age(transaction) AS xid_age FROM pg_prepared_xacts;
```

A session in `idle in transaction` with a large `xmin_age` is your saboteur. The defenses are all about *not letting transactions stay open*:

```sql
-- Cluster-wide: kill idle-in-transaction sessions after 5 minutes.
ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';
-- Cap any single statement (catches runaway analytics queries).
ALTER SYSTEM SET statement_timeout = '120s';
SELECT pg_reload_conf();

-- Manually evict a specific offender once you have its pid:
SELECT pg_terminate_backend(12345);
```

For replication slots, alert on slot lag and set `max_slot_wal_keep_size` so a dead consumer cannot hold WAL (and the horizon) indefinitely; drop slots you no longer use. The rule is simple: **a transaction should be open for milliseconds, not minutes.** Anything that holds one open longer is a write-health liability.

## 7. The practical playbook

Pulling it together: write health is four signals, each with one place to look and one lever to pull. This is the dashboard to build and the runbook to keep next to it.

![The write-health dashboard: each failure mode maps to one signal, one query, one threshold, and one corrective lever.](/imgs/blogs/keeping-postgres-healthy-under-write-load-8.webp)

The matrix above is the whole operational model on one screen. Beyond watching those four signals, three design choices do more for write health than any amount of tuning:

**Rebuild bloated indexes online, never with a lock.** Index bloat does not need `VACUUM FULL`. Since PG12, you rebuild without blocking writes:

```sql
REINDEX INDEX CONCURRENTLY events_created_at_idx;
-- or rebuild every index on a table:
REINDEX TABLE CONCURRENTLY events;
```

`REINDEX ... CONCURRENTLY` builds a fresh, compact index alongside the old one and swaps it in, holding only a brief lock at the end. For tables that need a full rewrite to shed heap bloat, use `pg_repack` rather than `VACUUM FULL` — `VACUUM FULL` takes an `ACCESS EXCLUSIVE` lock and stops all traffic to the table for the duration, which on a large hot table is an outage.

**Partition-and-drop instead of `DELETE`.** This is the single biggest win for any append-then-expire workload (events, logs, metrics, sessions). A bulk `DELETE FROM events WHERE created_at < now() - interval '30 days'` is one of the worst things you can do to a write-heavy database: it converts tens of millions of live rows into dead tuples that autovacuum then has to chase for hours, bloating the table and its indexes along the way. Declarative partitioning by time turns deletion into a metadata operation:

```sql
CREATE TABLE events (
    id bigint, created_at timestamptz NOT NULL, payload jsonb
) PARTITION BY RANGE (created_at);

CREATE TABLE events_2026_06 PARTITION OF events
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

-- Expiring a month is instant and produces ZERO dead tuples:
DROP TABLE events_2026_06;            -- or DETACH then archive
```

Dropping a partition is effectively free, creates no dead tuples, and never touches vacuum. If you take one structural lesson from this post, it is this: at scale, **do not delete rows — drop partitions.** This pairs naturally with the cleanup-and-retention thinking in [why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod), where the production data volume is exactly what dev never reproduces.

**Set `fillfactor` and curate indexes on hot tables** so updates go HOT, as covered in section 5. Cheap to do, compounding payoff.

A final note on what *not* to monitor by: do not watch table size on disk as your bloat signal. A healthy write-heavy table is *supposed* to hold a steady amount of dead-tuple slack between vacuums — that is the system working. Watch the *ratio* and the *trend* (`dead_pct` and whether `last_autovacuum` is keeping up), not the absolute bytes. When the slow-query pager goes off, this dashboard is often the real culprit hiding behind a plan that "used to be fast"; the companion incident walkthrough [the slow query at 1am](/blog/software-development/database-scaling/the-slow-query-at-1am) traces exactly that path.

## 8. A brief InnoDB contrast: purge, undo, and the history list

MySQL's InnoDB engine is also an MVCC system, and it has the same fundamental problem — old row versions must be kept until no read view needs them — but it solves the storage differently, which changes the operational shape.

| Dimension | Postgres | InnoDB (MySQL) |
| --- | --- | --- |
| Where old versions live | Inline in the table heap as dead tuples | In the **undo log** (rollback segments), separate from the table |
| What reclaims them | `VACUUM` / autovacuum | The background **purge** thread |
| The "how far behind" metric | `n_dead_tup`, `age(relfrozenxid)` | **History list length** (pending undo records) |
| Effect of a long transaction | Holds the xmin horizon; heap and indexes bloat | Holds the read view; undo log and history list grow |
| Catastrophic ceiling | Transaction-ID **wraparound** stops writes | No equivalent wraparound shutdown |
| Where bloat shows up | Table files and indexes | Undo tablespace (table data stays compact) |

Because InnoDB keeps current rows compact and pushes prior versions into undo, the table data does not bloat the way a Postgres heap does — but the undo log can balloon, and secondary indexes still accumulate versions that the purge thread must clean. The metric to watch is **history list length**: it is the InnoDB analog of `n_dead_tup` plus vacuum lag rolled into one number, and a growing history list means purge is losing the race. The trigger is identical to the Postgres story: a long-running transaction holds a read view open, purge cannot advance past it, and the undo log grows without bound. The lesson transfers cleanly across both engines — the long-running transaction is the universal enemy of any MVCC system, and bounding transaction lifetime is the highest-leverage discipline you have. What does *not* transfer is wraparound: InnoDB's transaction ID handling has no equivalent forced-freeze emergency, so the specific "database refuses writes to avoid data loss" failure is a Postgres-only hazard you must actively guard against.

## Case studies from production

### 1. Mandrill and the wraparound shutdown

In early 2019, Mandrill — Mailchimp's transactional-email service — suffered a multi-hour outage when its primary Postgres database hit transaction-ID wraparound and stopped accepting writes. The symptom was sudden and total: a service that processes enormous write volume went from healthy to refusing transactions, and recovery meant running `VACUUM` to advance `relfrozenxid` across very large tables while the service was down. The root pattern was the one this post warns about — on a database with that write rate, the freezing work simply was not keeping pace with xid consumption, and the age climbed past the wall before anyone was watching the right number. The lesson the whole industry took from it: `age(datfrozenxid)` is a first-class alert, not a curiosity, and the failure is a cliff, not a slope. There is no graceful degradation to lean on.

### 2. Sentry and owning autovacuum at ingest scale

Sentry ingests a firehose of error events into Postgres, and its engineering team has been candid about how much operational attention vacuum demands at that scale. The recurring theme in their writing is that stock autovacuum settings — the 20% scale factor, the timid cost limit, three workers — are built for a small database and fall apart on tables taking sustained heavy writes. The response was aggressive per-table tuning (small, frequent vacuums via low scale factors), generous cost limits so vacuum can actually finish, and partitioning the highest-churn tables so expiry never generates dead tuples. The generalizable point: at ingest scale you do not get to treat autovacuum as a default. You profile it, tune it per table, and watch it the way you watch query latency.

### 3. The idle transaction that bloated everything

A team I worked with watched disk usage climb on a Postgres cluster while `n_dead_tup` rose simultaneously across tables that their busiest service never wrote to. Autovacuum was running constantly and reclaiming almost nothing. The cause was a single analytics worker that opened a transaction, ran one `SELECT`, then blocked for hours waiting on a slow downstream HTTP call before committing — sitting in `idle in transaction` the whole time. Its snapshot pinned the xmin horizon, so vacuum could not reclaim anything newer than that snapshot anywhere in the database. The fix took thirty seconds: `pg_terminate_backend` on the offending pid, then `idle_in_transaction_session_timeout = '5min'` cluster-wide so it could never recur. Bloat drained on its own within an hour. The lesson: cluster-wide bloat with a busy-but-useless autovacuum almost always means a held horizon, not a vacuum-tuning problem.

### 4. The dead replication slot

A different incident looked like a disk-space emergency: WAL was accumulating and the data volume was filling. The team's first instinct was to tune checkpoints. The actual cause was a logical-replication slot feeding a change-data-capture pipeline that had silently died days earlier. An inactive slot holds back both WAL *and* the xmin horizon (`catalog_xmin`), so not only was WAL piling up, vacuum was also blocked from reclaiming dead tuples. Querying `pg_replication_slots` showed one slot with an enormous `xmin_age` and `active = false`. Dropping the dead slot released both pressures at once. The follow-up was to set `max_slot_wal_keep_size` so a dead consumer can never again hold the database hostage, and to alert on slot inactivity. Replication slots are silent horizon holders precisely because they show up in none of the session-level views.

### 5. DELETE-as-cleanup versus the partition drop

A logging service ran a nightly job: `DELETE FROM logs WHERE created_at < now() - interval '14 days'`, removing tens of millions of rows each run. Over months, the `logs` table and its indexes bloated relentlessly — autovacuum was permanently behind, chasing the dead tuples each `DELETE` produced faster than it could reclaim them, and query latency on recent data degraded because scans waded through dead pages. The rewrite converted `logs` to daily range partitions; the nightly job became `DROP TABLE logs_<oldest_day>`. Dead-tuple production from expiry dropped to literally zero, autovacuum caught up within a day, and steady-state table size stabilized. Same data, same retention policy, opposite write-health outcome — the difference was entirely deletion strategy.

### 6. The index that killed HOT

After adding a feature that surfaced "last active" timestamps, a team indexed `users.last_active_at` to make a dashboard query fast. The dashboard got faster; everything else got slower. `last_active_at` was updated on essentially every authenticated request, and indexing it meant every one of those updates now changed an indexed column — so the HOT ratio on the `users` table collapsed from ~96% to near zero. Each update now wrote a new index entry into multiple indexes, write amplification spiked, the indexes bloated, and overall write latency rose. The fix was to drop the index on the volatile column and answer the dashboard query a different way (a periodically-refreshed summary). HOT ratio recovered immediately. The lesson is counterintuitive and worth tattooing somewhere: an index added for read performance can quietly devastate write performance by disabling HOT.

## When to reach for each lever, and when not to

Reach for **per-table autovacuum tuning** when:

- A table is in the top handful by write volume and its `dead_pct` trends upward or `last_autovacuum` goes stale.
- The table is large enough that the 20% default scale factor means millions of dead tuples before action.
- You see autovacuum start, run for a long time, and visibly interfere with foreground traffic — lower the scale factor so it runs small and often instead of rarely and huge.

Reach for **HOT design (fillfactor + index curation)** when:

- A table's HOT ratio is low despite heavy updates, or its indexes bloat faster than the heap.
- You are about to add an index — first ask whether the column changes on every write; if so, find another way.

Reach for the **wraparound playbook (freeze monitoring, manual `VACUUM FREEZE`)** when:

- Any table's `age(relfrozenxid)` is climbing toward `autovacuum_freeze_max_age`. This is never optional and never "later."

Reach for **partition-and-drop** when:

- Your workload is append-then-expire, or any bulk `DELETE` removes a meaningful fraction of a table.

**Skip / avoid** these traps:

- Do **not** disable autovacuum globally to "save resources." You cannot disable anti-wraparound vacuum anyway, and you will hit the wall blind.
- Do **not** run `VACUUM FULL` on a live hot table to fix bloat — its `ACCESS EXCLUSIVE` lock is an outage. Use `REINDEX CONCURRENTLY` and `pg_repack`.
- Do **not** alert on raw table size as a bloat signal; a steady dead-tuple working set is healthy. Alert on ratio, trend, autovacuum staleness, and xid age.
- Do **not** treat a long-running transaction as harmless because it is read-only. A read-only `idle in transaction` session pins the horizon exactly as hard as a writer and bloats the whole cluster.

The through-line: on a write-heavy Postgres database, the dead tuple is not an edge case — it is the central fact of how the system works. Vacuum, freezing, HOT, and short transactions are the four disciplines that keep that fact from becoming an outage. Operate them deliberately and the database holds a steady size and stays fast indefinitely. Ignore them and you are not avoiding the work; you are deferring it to a single, much worse afternoon.

## Further reading

- [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) — the full row-versioning and visibility model both engines build on.
- [Postgres vacuum, bloat, and autovacuum tuning](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning) — the parameter-by-parameter mechanics behind this post's operational lens.
- [Why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod) — the data-volume and retention realities that make write health matter.
- [The slow query at 1am](/blog/software-development/database-scaling/the-slow-query-at-1am) — an incident walkthrough where bloat and a held horizon hide behind a plan that "used to be fast."
- Postgres documentation: *Routine Vacuuming* and *Preventing Transaction ID Wraparound Failures* — the authoritative reference for every threshold named here.

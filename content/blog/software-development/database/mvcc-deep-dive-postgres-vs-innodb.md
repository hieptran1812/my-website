---
title: "MVCC Deep Dive: How Postgres and InnoDB Implement Concurrent Reads"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles tour of multi-version concurrency control — how Postgres and InnoDB let readers and writers run without blocking each other, and the bloat, vacuum, purge, and wraparound costs that choice imposes."
tags:
  [
    "mvcc",
    "postgres",
    "innodb",
    "mysql",
    "transactions",
    "vacuum",
    "snapshot-isolation",
    "undo-log",
    "database-internals",
    "concurrency",
    "txid-wraparound",
    "performance",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-1.webp"
---

There is a question almost every backend engineer answers wrong the first time: *what happens when one transaction reads a row while another is in the middle of updating it?* The intuitive answer — the reader waits for the writer to finish — is how a naive lock-based system behaves, and it is a performance catastrophe. It means a single slow `UPDATE` on a hot row stalls every dashboard query, every analytics scan, every health check that touches that row. Under load, your read latency becomes a function of your *write* latency, which is exactly backwards from what you want.

Every serious database you are likely to run — PostgreSQL, MySQL's InnoDB, Oracle, SQL Server's RCSI mode, CockroachDB, YugabyteDB — refuses to make readers wait. The trick they all use has a name that sounds more intimidating than the idea: **multi-version concurrency control**, or MVCC. The idea is almost embarrassingly simple. *Never overwrite a row in place. When you change a row, keep the old version around and write a new one alongside it.* Then a reader can be handed a consistent, frozen-in-time view of the database — its **snapshot** — assembled entirely from old versions, without ever blocking a writer and without ever being blocked by one. Readers don't block writers, writers don't block readers. That is the whole promise, and MVCC is the mechanism that delivers it.

The catch — and there is always a catch in storage engineering — is that "keep the old version around" is a debt. Every old version is bytes on disk that some background process must eventually reclaim, and the *where* and *how* of that reclamation is the single biggest architectural difference between Postgres and InnoDB. Postgres keeps old versions **in the table itself** and cleans up with `VACUUM`. InnoDB keeps them in a separate **undo log** and cleans up with `PURGE`. Those two choices ripple out into bloat, replication lag, the famous transaction-ID wraparound shutdowns, and the operational playbook you need to keep either engine healthy. The diagram above is the mental model for the entire article — a single row, four versions, one reader seeing exactly the version that was current when its snapshot was taken. Everything else is a detail of that picture.

## The mental model: many versions, one snapshot per reader

![A row's version chain across three updates, each old version stamped dead while a new tuple is born](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-1.webp)

Read the figure left to right. A bank-account row starts at `balance=100`, created by transaction 100. Transaction 205 updates it to 90; transaction 312 to 70; transaction 480 adds 50. In a non-MVCC engine there would be exactly one copy of this row, mutated four times in place, and any reader catching it mid-update would either block or see garbage. Under MVCC there are **four physical versions** of the row, and each carries two pieces of bookkeeping that decide who can see it: the transaction id that *created* it (call it `xmin`) and the transaction id that *deleted or superseded* it (call it `xmax`). A version is alive to you if it was created by a transaction you can see and has not been superseded by a transaction you can see.

The reader in blue began its work at transaction id 300. When it looks at this row, it walks the versions and asks of each: *was this version created before my snapshot, and not yet superseded as of my snapshot?* Version v2 (`balance=90`) was created by txn 205 (before 300, visible) and superseded by txn 312 (after 300, not visible to this reader). So **v2 is the version this reader sees** — 90 — even though the physically current value is 70+50. The reader gets a clean, consistent answer with zero locking and zero waiting, and the writer that produced v4 never knew the reader existed.

It is worth pausing on how counterintuitive this is the first time you meet it. There is no single "the value" of this row at any instant — there are as many simultaneously-true values as there are distinct open snapshots, and each is correct for its reader. A reporting query that started an hour ago legitimately sees `balance=90` while a dashboard refreshed a second ago sees `balance=120`, both reading the same physical row at the same wall-clock moment, neither blocking the other, neither wrong. The database is not a single mutable cell; it's a stack of immutable photographs, and your snapshot picks which photograph you're looking at. Once that clicks, every downstream behavior — why old versions must be retained, why a long transaction is dangerous, why cleanup is a whole subsystem — stops being a collection of arbitrary rules and becomes one inevitable consequence.

This is snapshot isolation, the workhorse isolation level behind Postgres `REPEATABLE READ` and InnoDB's default `REPEATABLE READ`. Joukowsky-grade textbooks like Martin Kleppmann's *Designing Data-Intensive Applications* (Chapter 7) describe it as the property that **each transaction reads from a consistent snapshot of the database** — it sees all the data committed at the moment the snapshot was taken, and none of the writes committed afterward, even if those writes finish while the transaction is still running. MVCC is simply the implementation: keep enough old versions around that any in-flight transaction can be served the data as it looked at snapshot time.

> The first law of MVCC: a write is never a mutation, it is an append plus a tombstone. The old bytes don't change; they just acquire a death certificate.

The rest of this article is a tour through the machinery that makes this work. We will look at exactly how a snapshot is represented, what bookkeeping lives in each row version, how the visibility check runs, where the old versions are stored in each engine, how they are reclaimed, and the four or five ways this whole system blows up in production — every one of which has a famous postmortem attached.

## Why MVCC instead of read locks

**Senior rule of thumb: if your concurrency-control story requires readers to take locks, you have already lost the latency war on any read-heavy workload.**

Before MVCC, the textbook approach to isolation was two-phase locking (2PL): a reader takes a shared lock on every row it reads, a writer takes an exclusive lock, and the lock manager makes shared and exclusive locks mutually exclusive. This gives you correctness, but it couples read latency to write latency in the worst possible way. Let's make the mismatch concrete.

| You assume | The naive (lock-based) model | The MVCC reality |
| --- | --- | --- |
| A read of a being-updated row blocks | Reader waits on the writer's X-lock | Reader is served the last committed version with zero wait |
| A long analytics scan is harmless to writers | Scan holds shared locks, writers queue behind it | Scan reads a snapshot; writers proceed and pile up *versions* instead of *lock waits* |
| Concurrency cost is "lock contention" | CPU spent in the lock manager, deadlocks | CPU/disk spent storing and later reclaiming old versions |
| Old data is gone the instant it's overwritten | One copy, mutated in place | Many copies coexist; the "current" one is just the newest visible |
| Cleanup is free | No cleanup; in-place update | Cleanup is a first-class subsystem (VACUUM / PURGE) that can fall behind |
| A read-only transaction can't break anything | True under 2PL | Mostly true, but a *long* read-only txn pins old versions and starves cleanup |

The fourth and sixth rows are where the real surprises live. Under 2PL the cost of concurrency is *time* — transactions waiting on each other. Under MVCC the cost of concurrency is *space and background work* — versions accumulating until a cleaner removes them. You have not eliminated the cost; you have *moved* it from the latency-critical foreground (a reader waiting) to a background process that you now have to babysit. Almost every MVCC horror story in this article is the same root cause wearing a different costume: **the cleaner fell behind, and the versions piled up.**

It is worth being precise about what MVCC does *not* solve. MVCC gives you snapshot isolation cheaply, but snapshot isolation is not serializable — it permits the **write-skew** anomaly, where two transactions each read a consistent snapshot, each make a decision based on it, and together violate an invariant neither would violate alone. That is a topic for [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent). Here we focus on the mechanism, not the guarantee — on how the versions are stored, stamped, and reaped.

## 1. A snapshot is a set of "in-flight" transaction ids

**Rule of thumb: a snapshot is not a copy of the data; it is a tiny description of which transactions count as "already committed" from this reader's point of view.**

When a transaction needs a consistent read, it does not photocopy the database. It captures a *snapshot*, which in Postgres is essentially three numbers and a list:

- `xmin` of the snapshot: the lowest transaction id still considered in-progress. Every transaction with id below this is definitely complete (committed or aborted) as far as this snapshot is concerned.
- `xmax` of the snapshot: one past the highest assigned transaction id at snapshot time. Every transaction with id at or above this started *after* our snapshot and is therefore invisible.
- `xip_list`: the list of transaction ids that were *in progress* at snapshot time, i.e. between `xmin` and `xmax` but not yet committed. These are invisible too, because they hadn't committed when we looked.

A transaction `T` is visible to this snapshot if and only if `T < snapshot.xmax` **and** `T` is not in `xip_list` **and** `T` committed. You can see the live snapshot of your own session in Postgres:

```sql
-- Postgres 13+: the modern 64-bit-safe accessors.
SELECT pg_current_xact_id();          -- this transaction's id (assigns one)
SELECT pg_current_snapshot();         -- xmin:xmax:xip_list, e.g. 1001:1010:1003,1007

-- On Postgres 12 and earlier, the txid_* spelling:
SELECT txid_current();                -- assigns + returns the current txid
SELECT txid_current_snapshot();       -- 1001:1010:1003,1007

-- Read it field by field:
SELECT pg_snapshot_xmin(pg_current_snapshot())  AS snap_xmin,
       pg_snapshot_xmax(pg_current_snapshot())  AS snap_xmax,
       pg_snapshot_xip(pg_current_snapshot())   AS in_progress_xids;
```

The string `1001:1010:1003,1007` reads as: "all transactions below 1001 are settled; 1010 and above haven't started; and among 1001–1009, transactions 1003 and 1007 were still running when I looked, so I can't see them." That is the *entire* state a reader needs to decide visibility for every row in the database. It is a handful of bytes, captured once, and reused for every tuple the transaction inspects.

This is why snapshot capture is cheap and why a snapshot is stable: the numbers don't change for the life of the snapshot (in `REPEATABLE READ`; in `READ COMMITTED` a fresh snapshot is taken at the start of each statement, which is why `READ COMMITTED` can see other sessions' commits between statements). The expensive part is never the snapshot — it's keeping around all the row versions that any open snapshot might still need.

InnoDB calls the same concept a **read view**. A read view captures the set of transaction ids active at the moment it is created, plus `up_limit_id` (below which everything is visible, analogous to snapshot `xmin`) and `low_limit_id` (at or above which nothing is visible, analogous to snapshot `xmax`). The naming differs; the math is identical. A transaction's changes are visible to a read view if its id is below `up_limit_id`, or if it equals the read view's own creator, and invisible if it's at/above `low_limit_id` or in the active set. Under InnoDB `REPEATABLE READ`, the read view is created once on the first consistent read and reused for the whole transaction — which is exactly why a long-lived InnoDB transaction pins old versions, a fact that will cost us dearly in the case studies.

| Concept | Postgres term | InnoDB term |
| --- | --- | --- |
| Reader's consistent view | snapshot | read view |
| Lowest still-relevant txn id | snapshot `xmin` | `up_limit_id` |
| One past highest assigned | snapshot `xmax` | `low_limit_id` |
| Transactions in flight at capture | `xip_list` | active trx list |
| When captured (REPEATABLE READ) | first statement, reused | first read, reused |
| When captured (READ COMMITTED) | each statement | each statement |

## 2. The tuple header: where each version keeps its papers

**Rule of thumb: in Postgres, every row version carries a 23-byte header that is pure MVCC bookkeeping — the data you actually stored doesn't even begin until byte 24.**

A snapshot tells the reader which transactions are visible. But to apply that, each *version* of a row must record which transaction created it and which (if any) deleted it. In Postgres this lives in the **heap tuple header**, `HeapTupleHeaderData`, prepended to every tuple in every heap page.

![Inside an 8 KB heap page and the 23-byte tuple header that drives every visibility check](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-2.webp)

The figure shows the page on the left and the header fields on the right. An 8 KB heap page (the `BLCKSZ` default) starts with a 24-byte `PageHeaderData`, then an array of 4-byte `ItemId` line pointers that grow downward, then free space, then the tuples themselves which grow upward from the end of the page. Each tuple begins with its header. The fields that matter for MVCC are:

- **`t_xmin` (4 bytes)** — the transaction id that *inserted* this version. This is the version's birth certificate.
- **`t_xmax` (4 bytes)** — the transaction id that *deleted or updated-away* this version, or 0 if the version is still live. This is the death certificate. An `UPDATE` sets the old version's `t_xmax` to the updating transaction's id and inserts a brand-new tuple with a fresh `t_xmin`.
- **`t_field3`**, which is either `t_cid` (the command id within a transaction, so a statement doesn't see its own later changes) or `t_xvac` (used by old-style VACUUM FULL).
- **`t_ctid` (6 bytes)** — `(block number, offset)`. For a live tuple it points to itself; for a tuple that was superseded by an in-page update it points *forward* to the newer version, forming a chain.
- **`t_infomask2` (2 bytes)** — the number of attributes plus flag bits, notably `HEAP_HOT_UPDATED` and `HEAP_ONLY_TUPLE` (the HOT machinery we cover later).
- **`t_infomask` (2 bytes)** — status flags, including the all-important **hint bits** `HEAP_XMIN_COMMITTED`, `HEAP_XMIN_INVALID`, `HEAP_XMAX_COMMITTED`, `HEAP_XMAX_INVALID`.

The hint bits deserve a moment because they explain a confusing Postgres behavior. When a transaction commits, Postgres does *not* immediately walk every tuple it touched and mark them committed — that would be absurdly expensive. Instead it records the commit in the **commit log** (`pg_xact`, formerly `pg_clog`), a compact bitmap of transaction outcomes. The first reader to visit a tuple after the commit checks the commit log, learns the verdict, and *caches it* by setting the hint bit in `t_infomask`. This is why a `SELECT` immediately after a bulk load can be surprisingly slow and can even dirty pages it only "read": it is the first reader, paying to set hint bits for everyone after it. It also means the same query run twice can have different I/O profiles — a classic "why is my second run faster than my first" mystery that has nothing to do with the OS page cache.

You can see all of this directly with the `pageinspect` extension, which is the single best teaching tool in the Postgres tree:

```sql
CREATE EXTENSION IF NOT EXISTS pageinspect;

CREATE TABLE account (id int PRIMARY KEY, balance int);
INSERT INTO account VALUES (1, 100);
UPDATE account SET balance = 90  WHERE id = 1;
UPDATE account SET balance = 70  WHERE id = 1;

-- The visible truth: xmin, xmax, the ctid chain, and infomask flags per tuple.
SELECT lp,                                   -- line pointer slot
       t_xmin, t_xmax,                       -- birth and death txids
       t_ctid,                               -- forward pointer to newer version
       (t_infomask  & 256)  <> 0 AS xmin_committed,  -- HEAP_XMIN_COMMITTED
       (t_infomask  & 1024) <> 0 AS xmax_committed,  -- HEAP_XMAX_COMMITTED
       (t_infomask2 & 16384)<> 0 AS hot_updated,     -- HEAP_HOT_UPDATED
       (t_infomask2 & 32768)<> 0 AS heap_only        -- HEAP_ONLY_TUPLE
FROM   heap_page_items(get_raw_page('account', 0));
```

Run that and you'll see three rows in the page even though `SELECT * FROM account` returns one. Two are dead versions with their `t_xmax` set; one is live with `t_xmax = 0`. The `t_xmin`/`t_xmax` values are real transaction ids you can correlate with `txid_current()`. This is MVCC made visible: the dead versions are sitting in the page, taking up space, waiting for `VACUUM`.

InnoDB stores the equivalent bookkeeping not in a per-version header in the table but as **hidden system columns** on each clustered-index row, plus pointers into the undo log:

- **`DB_TRX_ID` (6 bytes)** — the transaction id that last inserted or updated this row. This is the analog of `t_xmin` for the *current* version.
- **`DB_ROLL_PTR` (7 bytes)** — the roll pointer: a pointer into the undo log to the record that lets InnoDB reconstruct the *previous* version of this row. Following it repeatedly walks backward through history.
- **`DB_ROW_ID` (6 bytes)** — an internal row id, used as the clustered key only when the table has no user-defined primary key.

So Postgres keeps full old versions inline and chains them by `t_ctid`; InnoDB keeps only the current version inline and chains *backward* into the undo log via `DB_ROLL_PTR`. That single structural difference is the headline of this entire article, and we'll draw it out fully in section 5.

## 3. The visibility check, step by step

**Rule of thumb: every tuple a query touches runs the visibility gauntlet — and on a bloated table, "every tuple" includes a mountain of dead ones the query will ultimately discard.**

![Postgres tuple visibility decision flow: gate on xmin, then gate on xmax](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-3.webp)

The figure is the decision flow Postgres runs (in `HeapTupleSatisfiesMVCC`) for each candidate tuple against the reader's snapshot. There are two gates.

**Gate 1 — is the inserter visible?** Look at `t_xmin`. If the inserting transaction is still in progress relative to our snapshot (it's `>= snapshot.xmax`, or it's in `xip_list`), or it aborted, then this version was never committed-and-visible to us, and the tuple is **invisible** — skip it. The hint bit `HEAP_XMIN_COMMITTED` short-circuits the common case: if it's set, the inserter definitely committed and we only need to check whether it committed before our snapshot. If `t_xmin` equals our *own* transaction id, we check `t_cid` so a statement doesn't see rows its own later commands inserted.

**Gate 2 — is the deleter visible?** If we got past gate 1, look at `t_xmax`. If it's 0, the version was never deleted — it's **visible**. If it's set, we ask the same question of the deleting transaction: did it commit before our snapshot? If yes, then as far as we're concerned this version is already gone — **invisible**. If no (the deleter is still running, or aborted, or started after our snapshot), the deletion hasn't "happened" for us yet, so the version is still **visible**.

Put plainly: *a version is visible to you if its creator committed in your past and its destroyer did not.* That's the whole rule, and it's why the reader in figure 1 sees v2 — v2's creator (205) is in its past, and v2's destroyer (312) is in its future.

The crucial operational consequence is in the figure's title: this check runs *per tuple*, and it runs on dead tuples too. A sequential scan over a table with 1 live row and 5 million dead versions still reads and visibility-checks all 5,000,001 tuples; it just discards the dead ones. This is precisely how a small-by-row-count table becomes catastrophically slow: the *physical* size, not the logical row count, drives scan cost. We will see this exact failure in the case studies. (Index-only scans dodge some of this via the visibility map — section 8.)

InnoDB's version of gate-passing is the read-view check we met in section 1, but with a twist: because the *current* row in the clustered index may be a version too new for our read view, InnoDB doesn't "skip" it — it **reconstructs** the right version by following `DB_ROLL_PTR` into the undo log until it finds one whose `DB_TRX_ID` the read view admits. That is a fundamentally different cost model, and it's section 5's whole point.

Here's a small experiment that makes the two-gate logic tangible without `pageinspect`, using two psql sessions:

```sql
-- Session A
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM account WHERE id = 1;   -- snapshot taken here, sees 70

-- Session B (a different connection, runs to completion)
UPDATE account SET balance = 9999 WHERE id = 1;   -- commits

-- Session A again, same transaction
SELECT balance FROM account WHERE id = 1;   -- STILL sees 70, not 9999
-- Because B's txid >= A's snapshot.xmax: gate 2 keeps the old version visible.
COMMIT;
SELECT balance FROM account WHERE id = 1;   -- now sees 9999
```

Session A sees 70 across both reads even though B committed 9999 in between, because B's transaction id is at or above A's snapshot `xmax` — gate 2 rules B's deletion of the old version invisible to A, so A keeps reading the old version. This *is* `REPEATABLE READ`, implemented entirely through tuple headers and a snapshot, with no read lock anywhere.

## 4. Postgres stores old versions in the heap — which is why it bloats

**Rule of thumb: in Postgres, an UPDATE is a DELETE plus an INSERT in the same table, so a high-churn table physically grows with every write until VACUUM catches up — even if its row count never changes.**

![Heap bloat from updates: dead tuples pile into the page until VACUUM reclaims the slots](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-4.webp)

This is the defining characteristic — and the defining liability — of Postgres MVCC. There is no separate version store. When you `UPDATE` a row, Postgres marks the old tuple dead (sets its `t_xmax`) and writes the new tuple into the heap, preferably on the same page but on a different page if there's no room. When you `DELETE` a row, Postgres just sets `t_xmax`; the bytes stay put. The dead tuples remain physically present, occupying space and slowing every scan, until **autovacuum** comes along and marks their line-pointer slots reusable.

The figure makes the before/after concrete. A hot row updated six times with no vacuum leaves one live tuple and five dead ones; the heap has grown sixfold; a sequential scan reads six versions to find one live row; and the index still has entries pointing at every dead `ctid`. After autovacuum sweeps, the five dead slots are freed for reuse (note: freed for *reuse*, not necessarily returned to the OS — that requires `VACUUM FULL` or `pg_repack`), the scan reads one version per row, and the dead index entries are gone.

You can watch bloat accumulate and then disappear:

```sql
CREATE EXTENSION IF NOT EXISTS pgstattuple;

CREATE TABLE counter (id int PRIMARY KEY, n bigint);
INSERT INTO counter VALUES (1, 0);

-- Hammer one row 100k times. In an MVCC-naive mental model this stays tiny.
DO $$ BEGIN
  FOR i IN 1..100000 LOOP UPDATE counter SET n = n + 1 WHERE id = 1; END LOOP;
END $$;

-- One logical row; thousands of dead tuples and a fat table.
SELECT n_live_tup, n_dead_tup FROM pg_stat_user_tables WHERE relname = 'counter';
SELECT pg_size_pretty(pg_relation_size('counter')) AS heap_size;
SELECT dead_tuple_count, dead_tuple_percent FROM pgstattuple('counter');

VACUUM counter;            -- mark dead slots reusable
SELECT pg_size_pretty(pg_relation_size('counter')) AS after_vacuum;  -- same file size
VACUUM FULL counter;       -- rewrite the table, return space to the OS
SELECT pg_size_pretty(pg_relation_size('counter')) AS after_vacuum_full;  -- now small
```

The output is the lesson: one live row, tens of thousands of dead ones, a heap that ballooned, and a `VACUUM` that frees space *for reuse* without shrinking the file. Only `VACUUM FULL` (which takes an `ACCESS EXCLUSIVE` lock and rewrites the whole table — never run it casually in production) actually returns disk to the OS. This is why "my table is 40 GB but `count(*)` says 12,000 rows" is one of the most common Postgres support tickets in existence.

### Second-order optimization: the fillfactor lever

Because in-page updates are vastly cheaper than cross-page ones (and enable HOT, section 9), Postgres lets you reserve free space in each page for future updates via `fillfactor`. The default is 100 (pack pages full), which is wrong for any update-heavy table:

```sql
-- Leave 20% of every page free so updates can land HOT on the same page.
ALTER TABLE counter SET (fillfactor = 80);
VACUUM FULL counter;   -- rewrite so the new fillfactor takes effect
```

Lowering `fillfactor` trades some disk and some scan efficiency for a dramatically higher HOT-update rate, which is often the single highest-leverage knob on a write-hot Postgres table. It is also a knob that does not exist in InnoDB, because InnoDB doesn't store old versions in the page in the first place.

## 5. InnoDB stores old versions in the undo log — which is why it purges

**Rule of thumb: in InnoDB, the table always holds the newest version of each row, and a reader who needs an older one pays at read time to rebuild it from the undo log.**

![InnoDB read-view reconstruction: walk DB_ROLL_PTR backward through undo records until the read view admits a version](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-5.webp)

InnoDB makes the opposite trade. The clustered index — which *is* the table, since InnoDB tables are index-organized — always stores the most recent version of each row, with its `DB_TRX_ID` and `DB_ROLL_PTR`. The old versions live in **undo log records** inside **rollback segments** in the system tablespace (or in dedicated undo tablespaces). When an `UPDATE` runs, InnoDB writes an undo record describing how to *reverse* the change, points the row's `DB_ROLL_PTR` at it, and updates the row in place in the clustered index.

The figure shows what a reader does when the current row is too new for it. The clustered-index row has `DB_TRX_ID=480`, but our read view has `up_limit_id=300` and rejects any `DB_TRX_ID >= 300`. So InnoDB follows `DB_ROLL_PTR` to undo record u3 (which restores the version as of txn 312 — still too new), then to u2 (txn 205 — accepted, `205 <= 300`), and returns that reconstructed version. Each hop applies an undo record to roll the row back one step. The read view *admits* version 205 and the reader gets it.

This has profound consequences that mirror Postgres's in reverse:

- **The table doesn't bloat from updates.** The clustered index holds one version per row; old versions live in undo, which is a separate, sequentially-written structure. A row updated a million times is still one row in the table.
- **Reads can get expensive.** A reader running against a row that has been updated many times since its read view was created may have to apply many undo records to reconstruct its version. A long-running InnoDB transaction reading a hot table can do a *lot* of reconstruction work.
- **Cleanup is `PURGE`, not `VACUUM`.** A background purge thread removes undo records once no read view could possibly need them, and physically removes delete-marked rows from the index. If purge falls behind, undo grows without bound.

The single most important diagnostic in InnoDB MVCC is the **history list length** — the number of undo records waiting to be purged. You read it from the engine status:

```sql
-- The headline number: how far purge has fallen behind.
SHOW ENGINE INNODB STATUS\G
-- Look in the TRANSACTIONS section for: "History list length 6000000"

-- Modern, scriptable equivalent:
SELECT count FROM information_schema.innodb_metrics
WHERE name = 'trx_rseg_history_len';

-- Find the long-running transaction that is holding purge back:
SELECT trx_id, trx_state,
       TIMESTAMPDIFF(SECOND, trx_started, NOW()) AS age_seconds,
       trx_rows_modified, trx_mysql_thread_id
FROM   information_schema.innodb_trx
ORDER BY trx_started ASC
LIMIT 5;
```

A history list length in the thousands is normal; in the millions it is an emergency, and the cause is almost always a single old transaction — exactly the Percona case study below, where the number sat at 6 million and refused to drop until 940 hung transactions were killed.

### The arithmetic of reconstruction cost

It's worth being concrete about *why* a long InnoDB read transaction degrades reads, because the cost is not constant — it grows with time. Suppose a hot row is updated once per second, and a reporting transaction opens a read view and runs for 30 minutes. Every read of that row from the reporting transaction must reconstruct the version as of the read view, which means walking *back* through every undo record written since the read view was created. After 30 minutes that's ~1,800 undo records to traverse for a single logical row read — each hop a pointer chase and an undo-record application. Multiply across thousands of rows the report touches and the work is enormous, and it gets worse the longer the transaction runs. This is the InnoDB mirror image of Postgres's "sequential scan reads dead tuples" tax: in Postgres the dead versions clog the *table*, so scans pay; in InnoDB the old versions clog the *undo log*, so the *long reader* pays to reconstruct. Either way, the bill is denominated in old versions that couldn't be reclaimed because some snapshot still needed them.

The practical upshot: an InnoDB transaction's read cost is not just a function of how much data it reads, but of *how much concurrent write churn has happened on that data since the transaction started.* A report that would run in seconds against a static table can run for minutes against the same table under heavy concurrent writes, purely because of reconstruction depth. The fix is, again, to shorten the read transaction or move it to a quiescent replica — never to "optimize the query," because the query isn't the problem; the read-view age is.

### Second-order optimization: undo tablespace sizing and purge threads

Two InnoDB knobs directly govern this subsystem. `innodb_purge_threads` (default 4) controls purge parallelism; raising it helps a purge-bound workload catch up. `innodb_max_purge_lag` and `innodb_max_purge_lag_delay` let you *throttle incoming DML* when the history list grows past a threshold — a deliberate self-defense valve that slows writers so purge can keep up, trading write latency for bounded undo growth. And since MySQL 8.0, undo lives in dedicated truncatable undo tablespaces, so a one-time history spike doesn't permanently bloat the system tablespace the way it did in 5.6. The defaults assume a workload where purge comfortably keeps up; the moment you see `trx_rseg_history_len` trending upward over hours rather than oscillating around a flat baseline, one of these valves — or a hunt for the long transaction holding the horizon — is overdue.

## 6. The architectural fork: heap-append vs undo-log, side by side

![Where old versions live: Postgres keeps them in the heap and cleans with VACUUM; InnoDB keeps them in undo and cleans with PURGE](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-6.webp)

Step back and the whole comparison fits on one slide. Both engines are MVCC; both give you snapshot isolation; both let readers and writers run without blocking. They differ only in *where the old versions live* and *therefore how they're cleaned up* — and that one difference determines every operational property that matters.

| Dimension | Postgres (in-heap) | InnoDB (undo log) |
| --- | --- | --- |
| Where old versions live | In the table's heap pages | In undo log / rollback segments |
| What the table holds | All versions, newest among them | Only the newest version |
| Cost of an UPDATE | Mark old dead + write new tuple (may move pages) | Update in place + write undo record |
| Cost of reading an old version | Cheap — it's right there, skip the dead | Expensive — reconstruct by applying undo records |
| Cleanup process | `VACUUM` / autovacuum | `PURGE` thread |
| Cleanup failure mode | Table + index bloat, txid wraparound | Undo growth, history list length explosion |
| Returns space to OS automatically | No (needs VACUUM FULL / pg_repack) | Undo tablespaces truncatable (8.0+) |
| Hot-row update churn | Bloats heap + indexes (unless HOT) | Bloats undo, not the table |
| Long read-only txn impact | Pins `xmin`, blocks all VACUUM cleanup | Pins read view, blocks purge |
| Secondary index versioning | Each index entry is per-version (bloats) | Secondary indexes don't store trx id; uses a per-page max trx id + undo |
| Sequential-scan tax of dead rows | High — scans read dead tuples | Low — table holds one version per row |

Read the table as a set of trades, not a winner. Postgres pays at *write and cleanup* time (bloat, vacuum) and gets *cheap reads of old versions*. InnoDB pays at *read time* (reconstruction) and gets a *compact table that doesn't bloat from updates*. Postgres's design is gloriously simple — there's only one place data lives — but that simplicity is exactly what makes vacuum so load-bearing. InnoDB's design keeps the table small but adds the undo subsystem and read-time reconstruction as moving parts. Neither is free.

> Choosing an MVCC engine is choosing *which* background process you are willing to babysit: Postgres's VACUUM or InnoDB's PURGE. Both will eventually fall behind under abuse, and both will take your database down when they do. The abuse, almost always, is a transaction that stayed open too long.

There is one more asymmetry worth calling out because it surprises people: **secondary indexes**. In Postgres, every index entry points at a heap `ctid`, and because a non-HOT update relocates the tuple, it must insert a new entry into *every* index — so indexes bloat alongside the heap (figure in section 9). In InnoDB, secondary index records don't carry a transaction id; instead InnoDB tracks a per-page maximum transaction id and, when an index read might be stale, consults the clustered index and undo to check visibility. The upshot is that InnoDB secondary indexes are smaller and don't get a new entry per update of an unindexed column — a real advantage for update-heavy workloads with many indexes, and one of the reasons MyRocks and InnoDB hold up under write churn that would bloat a naively-indexed Postgres table. This connects to how [B-trees power database indexes](/blog/software-development/database/b-trees-how-database-indexes-work) and contrasts with the append-only philosophy of [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), where old versions are reclaimed by compaction rather than vacuum or purge.

## 7. The long-running transaction: MVCC's universal failure mode

**Rule of thumb: the most dangerous object in an MVCC database is not a big query — it is a small transaction that someone forgot to close.**

![A long transaction holds the xmin horizon back and balloons bloat until it finally commits](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-7.webp)

Here is the failure mode that underlies *most* MVCC production incidents, and it follows inexorably from everything above. The cleaner — VACUUM or PURGE — may only remove a version if **no open snapshot could still need it**. The oldest open snapshot in the system defines a horizon: any version that some still-running transaction's snapshot might want to read cannot be reclaimed. In Postgres this horizon is the **oldest `xmin`** across all backends and replicas; in InnoDB it's the oldest active read view.

The timeline in the figure tells the story. At 09:00 a transaction opens (perhaps a report, perhaps a buggy connection-pool leak, perhaps an `idle in transaction` session) and takes a snapshot at xid 1000. It then does nothing — but it stays open. Over the next two hours, 10 million updates run on the database. Autovacuum dutifully wakes up at 09:30, scans tables full of dead tuples, and removes **zero** of them, because the open transaction's snapshot at xid 1000 might still want to read the pre-update versions. Dead tuples climb to 9.6 million; bloat balloons by 40 GB; query latency degrades as scans wade through dead rows. The moment the long transaction commits at 13:00, the `xmin` horizon jumps forward, and the *next* vacuum finally reclaims everything at once.

The same dynamic in InnoDB shows up as the history list length climbing while one ancient transaction sits in `information_schema.innodb_trx` with an age of hundreds of thousands of seconds. Purge can't advance past the oldest read view, so undo grows without bound.

The diagnostic queries are essential muscle memory. In Postgres:

```sql
-- The single most useful MVCC-health query: who is holding xmin back?
SELECT pid, state, age(backend_xmin) AS xmin_age,
       now() - xact_start AS txn_age,
       now() - state_change AS idle_for,
       left(query, 60) AS query
FROM   pg_stat_activity
WHERE  backend_xmin IS NOT NULL
ORDER  BY age(backend_xmin) DESC
LIMIT  10;

-- Replication slots and prepared transactions also pin xmin globally:
SELECT slot_name, age(xmin) AS xmin_age, active FROM pg_replication_slots;
SELECT gid, prepared, age(transaction) AS xid_age FROM pg_prepared_xacts;
```

The defenses are equally essential, and every mature Postgres deployment sets them:

```sql
-- Kill any session left idle inside an open transaction after 5 minutes.
ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';

-- Cap how long any single statement may run (catches runaway reports).
ALTER SYSTEM SET statement_timeout = '120s';

-- (Postgres 17+) cap how long a transaction may stay open at all.
ALTER SYSTEM SET transaction_timeout = '30min';
SELECT pg_reload_conf();
```

`idle_in_transaction_session_timeout` is the one that saves you, and it is *off by default*. A single application bug — a transaction opened before a slow external API call, a developer's psql session left on `BEGIN;` over lunch, a connection pool that doesn't reset transaction state — can pin the xmin horizon for hours and silently bloat your entire database. Set the timeout. The same `BEGIN; ... call external API ... COMMIT;` antipattern is responsible for the GitLab and Percona case studies below.

### Second-order optimization: replicas and hot_standby_feedback

A subtle version of this trap lives on read replicas. By default, a long query on a Postgres physical replica does *not* hold back the primary's vacuum — but if it runs long enough to need versions the primary already vacuumed away, the replica's recovery hits a conflict and the query is canceled (`ERROR: canceling statement due to conflict with recovery`). Turn on `hot_standby_feedback = on` and the replica reports its oldest snapshot back to the primary, which then *won't* vacuum versions the replica still needs — protecting the query but **re-coupling the primary's bloat to the replica's longest transaction.** You have moved the problem, not solved it. The right answer is usually short transactions everywhere plus `max_standby_streaming_delay` tuned to your tolerance, not blindly enabling feedback and discovering your primary bloating because of a 6-hour analytics query on a replica.

## 8. Transaction-ID wraparound: the 32-bit clock that has taken down giants

**Rule of thumb: Postgres transaction ids are a 32-bit counter that wraps, and if VACUUM doesn't "freeze" old rows before the counter laps them, the database shuts itself down to avoid corruption. This is not theoretical — it has caused multi-day outages at Sentry, Mailchimp, and nearly Notion.**

![The 32-bit XID clock: age zones escalating from normal vacuum to a forced shutdown](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-8.webp)

Everything so far treated transaction ids as ever-increasing integers. They aren't — they're 32-bit, so there are only about 4 billion of them, and a write-heavy system can burn through 4 billion transactions in weeks. As Sentry's postmortem put it, a busy database can "reach four billion transactions within a matter of weeks." When the counter hits the top, it wraps back to the bottom.

This is a deep problem for visibility. Recall the rule: a tuple is visible if its `t_xmin` is "in the past." Postgres decides past-vs-future by **modulo-2³¹ comparison** — for any transaction id, the 2 billion ids before it are "older" and the 2 billion after it are "newer," arranged in a circle. That works only as long as no live tuple has an `xmin` more than 2 billion transactions old. If the counter laps a tuple whose `xmin` is more than ~2 billion behind the current id, that ancient-but-live row would suddenly appear to be **in the future** — invisible — and your committed data would vanish. As the malisper.me explainer describes it, "some transactions in the past would appear to be in the future, causing massive data corruption."

The defense is **freezing**. VACUUM, as it sweeps, replaces the `t_xmin` of sufficiently old, definitely-committed-and-visible-to-everyone tuples with a special **frozen** marker (historically `FrozenTransactionId = 2`; modern versions set a flag) that is treated as *older than every possible transaction id*. A frozen tuple can never be lapped. Each table tracks `relfrozenxid` — the oldest unfrozen xid it still contains — and the database tracks `datfrozenxid` per database. The age of `relfrozenxid` is how close that table is to disaster.

The figure shows the escalation zones as the oldest unfrozen xid ages:

- **Normal (age 0–50M):** lazy autovacuum and HOT pruning keep up; all green.
- **Anti-wraparound (age 200M, the `autovacuum_freeze_max_age` default):** Postgres triggers a *forced* "aggressive" autovacuum specifically to freeze old tuples, even on tables that wouldn't otherwise be vacuumed and even if autovacuum is disabled. This is the safety net.
- **Warn (age ~2.1B − 40M):** Postgres logs `WARNING: database "X" must be vacuumed within N transactions` on every transaction, where N counts down toward zero.
- **Halt (age 2.1B − 1M):** with about a million transactions left, Postgres **stops accepting new commands** with `ERROR: database is not accepting commands to avoid wraparound data loss in database "X"` and demands an offline, single-user-mode `VACUUM`. Your database is down.

Watch your distance from the cliff with this query, which every Postgres operator should have wired into alerting:

```sql
-- Per-table age of relfrozenxid; alert well before 1.5 billion.
SELECT relname,
       age(relfrozenxid) AS xid_age,
       round(100.0 * age(relfrozenxid) / 2.1e9, 1) AS pct_to_wraparound,
       pg_size_pretty(pg_total_relation_size(oid)) AS size
FROM   pg_class
WHERE  relkind IN ('r','m','t')
ORDER  BY age(relfrozenxid) DESC
LIMIT  20;

-- Database-level horizon:
SELECT datname, age(datfrozenxid) AS xid_age FROM pg_database ORDER BY xid_age DESC;
```

If you are anywhere near the halt zone, the recovery from a shutdown is exactly what Mailchimp and Sentry had to do: bring Postgres up in single-user mode and run `VACUUM` (specifically a freezing vacuum) on the offending database before it will accept connections again:

```bash
# Last-resort recovery after a wraparound shutdown (database offline).
postgres --single -D /var/lib/postgresql/data mydb
backend> VACUUM FREEZE;
backend> \q
```

The reason this section exists is that it has happened to companies you have heard of:

- **Sentry (July 2015)** was down for most of a US working day. As a "very write heavy application with some very large relational tables," it consumed XIDs fast enough that autovacuum couldn't freeze in time, and Postgres halted to protect the data. ([Sentry postmortem](https://blog.sentry.io/transaction-id-wraparound-in-postgres/))
- **Mailchimp's Mandrill (February 2019)** ran roughly 41 hours degraded. Their sharded Postgres setup had a hashing algorithm that "favored shard4, causing it to be hotter than the others"; autovacuum "likely fell behind due to the higher load," wraparound protection kicked in on that shard, and a normal offline VACUUM was estimated at *40+ days* — they ultimately recovered only by truncating the largest tables. ([Mailchimp postmortem](https://mailchimp.com/what-we-learned-from-the-recent-mandrill-outage/))
- **Notion (2021)** never had the outage, but it was the gun to their head. As their sharding writeup explains, "the Postgres `VACUUM` process began to stall consistently, preventing the database from reclaiming disk space from dead tuples," and "more worrying was the prospect of transaction ID (TXID) wraparound, a safety mechanism in which Postgres would stop processing all writes." They called it "an existential threat to the product" and sharded their monolith into 480 logical shards across 32 physical databases to escape it. ([Notion: Herding Elephants](https://www.notion.com/blog/sharding-postgres-at-notion))

InnoDB's transaction id is 48 bits, not 32 — about 281 trillion values — so wraparound is not a practical concern; you would need to run for centuries. InnoDB has a different ticking clock (the undo log and history list, section 5), but it does not have the wraparound-shutdown failure mode, and this is one of the genuine operational advantages of InnoDB over Postgres. (Postgres has a long-discussed 64-bit XID project that would eliminate the concern entirely; until it ships, freezing is mandatory hygiene.)

### Second-order optimization: tuning the freeze knobs

The defaults are conservative for small databases and dangerous for huge, write-hot ones. The levers:

```sql
-- Trigger forced freezing earlier on very busy clusters (lower = more aggressive).
ALTER SYSTEM SET autovacuum_freeze_max_age = '1000000000';  -- default 200M
-- Make autovacuum more aggressive overall so it never falls behind:
ALTER SYSTEM SET autovacuum_max_workers = 6;          -- default 3
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 3000; -- default 200 (throttle off)
-- Per-table override for a hot table:
ALTER TABLE events SET (autovacuum_freeze_max_age = 100000000,
                        autovacuum_vacuum_scale_factor = 0.02);
SELECT pg_reload_conf();
```

The single most common mistake is leaving `autovacuum_vacuum_cost_limit` at its throttled default of 200 on a fast SSD-backed server, which caps vacuum's I/O so low that on a multi-terabyte table it literally cannot finish freezing before the next wraparound deadline. Figma hit precisely this wall — at terabyte/billion-row table scale they "began to see reliability impact during Postgres vacuums, which are essential background operations that keep Postgres from running out of transaction IDs and breaking down" ([Figma databases postmortem](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/)).

## 9. HOT updates and the visibility map: how Postgres fights its own bloat

**Rule of thumb: a HOT update is the difference between an UPDATE that costs one heap write and an UPDATE that costs one heap write plus an entry in every index — and you control which one you get.**

![Non-HOT update vs HOT update: HOT keeps the new version on the same page and writes zero index entries](/imgs/blogs/mvcc-deep-dive-postgres-vs-innodb-9.webp)

The append-in-heap design has a brutal corollary for indexes: because a normal update writes the new tuple to a (possibly) different page with a different `ctid`, *every* index on the table needs a new entry pointing at the new location — even indexes on columns that didn't change. Update one unindexed column on a table with eight indexes and you've written nine things: the heap tuple plus eight index entries. That is how index bloat outpaces heap bloat on update-heavy tables.

**Heap-Only Tuple (HOT) updates**, introduced in Postgres 8.3, are the escape hatch. A HOT update happens when two conditions hold: (1) **no indexed column is being changed**, and (2) **the new version fits on the same page** as the old one. When both hold, Postgres does something clever: it writes the new tuple to the same page, sets the old tuple's `t_ctid` to point at the new one (forming a **HOT chain**), marks the old tuple `HEAP_HOT_UPDATED` and the new one `HEAP_ONLY_TUPLE`, and writes **zero** new index entries. The existing index entry still points at the old tuple's line pointer, and an index scan that lands there simply follows the `t_ctid` chain to the live version. Per the Postgres source's `README.HOT`, the old tuple is marked `HEAP_HOT_UPDATED` and the new one `HEAP_ONLY_TUPLE`, and the chain is traversed via `t_ctid` — no index maintenance required.

The figure contrasts the two paths. On the left, a non-HOT update (indexed column changed, or page full): new tuple on a different page, one new entry in *every* index, index bloat per update, and VACUUM must later clean both heap and indexes. On the right, a HOT update: new tuple on the same page, zero new index entries, the old tuple's `t_ctid` chains to the new one, and — the underrated bonus — **HOT pruning** can reclaim dead HOT tuples on the fly during ordinary page access, without waiting for VACUUM at all.

You can measure your HOT ratio directly, and it is one of the best health metrics for a write-hot table:

```sql
SELECT relname,
       n_tup_upd                       AS total_updates,
       n_tup_hot_upd                   AS hot_updates,
       round(100.0 * n_tup_hot_upd / nullif(n_tup_upd, 0), 1) AS hot_pct
FROM   pg_stat_user_tables
WHERE  n_tup_upd > 0
ORDER  BY n_tup_upd DESC
LIMIT  20;
```

A low `hot_pct` on a high-`total_updates` table is a flashing red light. The two fixes follow directly from the two conditions: **lower `fillfactor`** (section 4) so updates have room to stay on-page, and **drop or rethink indexes on frequently-updated columns** so updates can qualify as HOT in the first place. An `updated_at` timestamp column with an index on it is a classic HOT-killer: every row touch changes the indexed column, so no update is ever HOT.

To make the stakes concrete, consider a `sessions` table with a primary key, an index on `user_id`, an index on `expires_at`, and an index on `last_seen_at`, updated on every request to bump `last_seen_at`. With the index on `last_seen_at`, *every* update changes an indexed column, so zero updates are HOT: each request writes a new heap tuple plus a new entry in all four indexes — five physical writes and four index entries to later vacuum, per request. Drop the `last_seen_at` index (you almost certainly don't range-query on it) and set `fillfactor = 85`, and suddenly the vast majority of updates are HOT: one same-page heap write, zero index entries, and dead versions reclaimed inline by HOT pruning without VACUUM ever touching the indexes. On a busy session store this is frequently a 4–5× reduction in write I/O and a near-elimination of index bloat — from one schema change. The lesson generalizes: on any update-hot Postgres table, the index list is not free; every index you add is a tax on every update of any indexed column, paid in lost HOT updates and the bloat that follows.

### The visibility map and index-only scans

There's one more structure that ties bloat, vacuum, and read performance together: the **visibility map** (VM), a compact bitmap with two bits per heap page. The `all-visible` bit means every tuple on that page is visible to *all* transactions (no dead tuples, nothing in-flight); the `all-frozen` bit means every tuple is frozen. VACUUM sets these bits as it cleans.

The VM enables two big wins. First, **freezing vacuum can skip all-frozen pages entirely** — this is the optimization (Postgres 9.6+) that turned anti-wraparound vacuums from full-table scans into incremental sweeps, and it's why keeping the VM healthy is essential on huge tables. Second, **index-only scans**: if a query can be answered entirely from an index and the heap pages it would visit are marked all-visible in the VM, Postgres skips the heap lookup entirely. A bloated table with a stale VM defeats index-only scans, because the planner can't trust that the heap doesn't hold a conflicting version, so it falls back to heap fetches. So bloat doesn't just slow sequential scans — it silently downgrades your fast index-only scans into slow index+heap scans. Keeping VACUUM current keeps the VM current keeps your index-only scans fast. It is all one system.

## Case studies from production

These are the incidents that put faces on the mechanisms above. Every one is the same story — the cleaner fell behind because a transaction stayed open, or the XID clock ran out — told in a different accent.

### 1. Sentry's day-long wraparound shutdown

On Monday, July 20, 2015, Sentry was down for most of the US working day. The root cause was textbook transaction-ID wraparound. Sentry described itself as "a very write heavy application with some very large relational tables (both in number of rows, as well as physical size on disk)" — exactly the profile that burns XIDs fastest. Autovacuum could not freeze old tuples quickly enough on the largest tables, the oldest `relfrozenxid` aged past the safety threshold, and Postgres did what it is designed to do: it stopped accepting commands to avoid wrapping the 32-bit counter and corrupting data. The fix was an offline freezing VACUUM, and the lesson Sentry drew — and published so the rest of us could learn it cheaply — was to monitor `age(relfrozenxid)` proactively and tune autovacuum to be far more aggressive on write-hot tables, rather than discovering the cliff by falling off it. This single postmortem is probably responsible for more "set up wraparound alerting" tickets across the industry than any other document. ([Source](https://blog.sentry.io/transaction-id-wraparound-in-postgres/))

### 2. Mailchimp's Mandrill: 41 hours and a truncate

In February 2019, Mailchimp's transactional-email product Mandrill ran roughly 41 hours in a degraded state — sending only about 80% of queued email. Mandrill used a sharded Postgres setup, and the hashing algorithm "favored shard4, causing it to be hotter than the others." That hot shard burned XIDs faster than its autovacuum could freeze, hit wraparound protection, and went read-only. The wrong first hypothesis was that this was a transient load spike; the actual root cause was structural skew in shard assignment that had been quietly overloading one node for a long time. The recovery is the part that haunts every DBA: a straightforward offline VACUUM was estimated at *over 40 days*, because the table was so large and so bloated. They escaped only by **truncating** the largest offending tables (Search and Url) — accepting data loss in those tables — after which the VACUUM finished in about an hour. The lessons they published were as much organizational as technical: detection was slowed because "unrelated exceptions coming in from other systems drowned it out in our centralized reporting," and recovery was slowed because "knowledge of and access to Mandrill systems and monitoring is concentrated in a small number of individuals." Wraparound is a database failure; the *duration* was an observability and bus-factor failure. ([Source](https://mailchimp.com/what-we-learned-from-the-recent-mandrill-outage/))

### 3. Notion: sharding to outrun VACUUM

By mid-2020, Notion's Postgres monolith — which had served the company through "five years and four orders of magnitude of growth," accumulating billions of blocks — was failing in the most MVCC way possible. As they wrote, "the Postgres `VACUUM` process began to stall consistently, preventing the database from reclaiming disk space from dead tuples," and "more worrying was the prospect of transaction ID (TXID) wraparound." They named it "an existential threat to the product." The symptom was the canonical one: a table so large and so churned that a single vacuum couldn't finish reclaiming dead tuples between wraparound deadlines, so dead tuples and XID age both crept up together. The wrong easy answer would have been "buy a bigger box"; they'd already done that. The actual fix was architectural — shard the monolith into 480 logical shards across 32 physical databases, so that each table on each shard was small enough for VACUUM to keep up and XID consumption per shard dropped by the shard count. The lesson is that vacuum cost scales with table *size* and XID burn scales with write *rate*; past a certain scale, the only durable fix is to make each table smaller, which means sharding or partitioning. ([Source](https://www.notion.com/blog/sharding-postgres-at-notion))

### 4. Figma: vacuum reliability at the terabyte frontier

Figma's database stack grew almost 100x from 2020. Along the way they hit the same wall as Notion from a different angle: at terabyte-scale tables with billions of rows, they "began to see reliability impact during Postgres vacuums, which are essential background operations that keep Postgres from running out of transaction IDs and breaking down." Their highest-write tables were also "growing so quickly that we would soon exceed the maximum IO operations per second (IOPS) supported by Amazon's RDS." Crucially, vertical partitioning — moving big tables to their own databases — wasn't enough, because "the smallest unit of partitioning is a single table" and their single hottest tables were themselves too big for one machine's vacuum and IOPS budget. The fix was horizontal sharding via their DBProxy layer. The lesson here is subtle and important: vacuum isn't just a CPU job, it's an *I/O* job, and on a table big enough to saturate your storage's IOPS, vacuum competes with your live traffic for the same scarce I/O — so the bloat problem and the throughput problem become the same problem. ([Source](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/))

### 5. Percona's hung transaction: history list length 6,000,000

A Percona customer experienced steadily degrading SELECT performance that only a server restart could fix — the classic signature of an InnoDB undo problem. `SHOW ENGINE INNODB STATUS` revealed the smoking gun: the history list length had climbed to **6 million** and was still growing, and `information_schema.innodb_trx` showed transactions in state `ACTIVE 766132 sec` — over eight days old. These were abandoned connections holding open transactions under the default `REPEATABLE READ` isolation level. Because each held a read view that purge could not advance past, "InnoDB can't purge the undo records" — every old version since those transactions began had to be retained, and every SELECT had to wade through that ever-growing undo to reconstruct visible versions. The fix was immediate once diagnosed: killing the 940 dormant connections caused `trx_rseg_history_len` to drop from 6 million to **793**. The wrong first hypothesis (a slow query, a missing index) wasted time; the actual root cause was a transaction-lifecycle bug in the application that left connections open. This is section 7's long-running-transaction trap, in InnoDB clothing, with a real number attached. ([Source](https://www.percona.com/blog/chasing-a-hung-transaction-in-mysql-innodb-history-length-strikes-back/))

### 6. The idle-in-transaction microservice

A widely-reported pattern (and one I have personally debugged more than once): a microservice opens a database transaction to log a unit of work, then — inside the still-open transaction — makes a synchronous call to a slow external API that takes tens of seconds. For the duration of every such call, the session sits `idle in transaction`, holding any row locks it took *and* pinning the global `xmin` horizon so VACUUM can reclaim nothing newer than that snapshot. In one documented case this "prevented cleanup of old data, leading to massive table bloat which slowed overall performance by 30%," on top of blocking hundreds of other transactions on held locks. The fix is two lines: never hold a transaction open across an external call (commit the local work first, do the API call, then start a new transaction for the result), and set `idle_in_transaction_session_timeout` as a backstop so a leaked transaction is killed automatically. The deeper lesson is that *transaction scope is a latency budget*: every millisecond a transaction is open is a millisecond it pins the vacuum horizon, so the cardinal rule is to keep transactions as short as physically possible and to never, ever wait on the network inside one. ([Source](https://stormatics.tech/blogs/idle-transactions-cause-table-bloat-wait-what))

### 7. GitLab and the subtransaction cliff

GitLab.com spent a month eliminating PostgreSQL subtransactions after a baffling, intermittent performance collapse. Rails' `find_or_create_by` with `requires_new: true` issues a `SAVEPOINT`, which creates a *subtransaction* with its own subtransaction id. Postgres maps subtransaction ids to parent XIDs in a small SLRU cache (`pg_subtrans`) of only 32 pages (~65,000 entries). On their replicas, "a single SAVEPOINT during a long transaction" could trip `suboverflow`, forcing every subsequent XID visibility lookup to consult the SLRU cache on disk — and under that condition, "transaction rates plummeted to 50,000 TPS on the replicas" from a baseline near 360,000, with "database queries used to retrieve CI/CD builds data timing out." The incident "would show up for 15 minutes and disappear for days," which is exactly what makes SLRU-contention bugs so hard to catch. The connection to MVCC is direct: subtransactions are part of how Postgres tracks *which transaction created which tuple version*, and the visibility check has to resolve subtransaction ids to their parents to decide commit status. Scale that lookup past its cache and the whole visibility machinery grinds. GitLab's fix was to eliminate SAVEPOINT usage entirely — rewriting to `INSERT ... ON CONFLICT` — and to alert on any single SAVEPOINT in the application. ([Source](https://about.gitlab.com/blog/why-we-spent-the-last-month-eliminating-postgresql-subtransactions/))

### 8. The "small table, huge file" support ticket

This one doesn't have a famous blog post because it happens to *everyone*, constantly. A queue or counter or session table holds a handful of live rows but is updated millions of times a day. Because each update leaves a dead tuple and autovacuum is throttled (or disabled, or starved by a long transaction), the table's physical file grows to gigabytes while `SELECT count(*)` returns a few thousand. Queries that should be instant take seconds, because a sequential scan reads millions of dead tuples to find the live ones (section 3), and even index scans suffer from a bloated index full of dead pointers. The first wrong hypothesis is always "we need an index" or "we need a bigger instance"; the actual diagnosis is `pgstattuple` showing 99% dead tuples. The fix is `VACUUM` (to stop the bleeding) plus `pg_repack` or `VACUUM FULL` (to reclaim the space) plus a much more aggressive per-table autovacuum setting and a lower `fillfactor`. The lesson: on an MVCC engine, *physical size is driven by churn and vacuum lag, not by row count* — and the two diverge spectacularly on hot small tables. The same workload on InnoDB stays compact in the table (the churn goes to undo instead), which is why "small hot table" workloads sometimes behave better on MySQL out of the box.

### 9. The mysqldump that bloated undo

A nightly `mysqldump --single-transaction` is the recommended way to take a consistent logical backup of InnoDB without locking, because `--single-transaction` opens one long `REPEATABLE READ` transaction and dumps everything from its consistent read view. The trap: that transaction can run for *hours* on a large database, and for its entire duration it pins the oldest read view, so purge cannot advance and the history list length climbs with every concurrent UPDATE/DELETE on the live system. Teams discover this when their nightly backup window coincides with a write-heavy batch job and the history list length spikes into the millions, undo tablespaces balloon, and morning queries are slow from all the read-time reconstruction. The fix is to run consistent backups against a *dedicated replica* (so the long read view lives on a node with no write traffic to retain undo for), or to use a physical backup tool like Percona XtraBackup that doesn't hold a long logical read view. The lesson mirrors the Postgres replica trap: a long-running read transaction is a long-running read transaction regardless of how virtuous its purpose, and "it's just a backup" doesn't exempt it from the purge horizon. ([Source](https://www.percona.com/blog/chasing-a-hung-transaction-in-mysql-innodb-history-length-strikes-back/))

### 10. The autovacuum that couldn't get the lock

A more insidious Postgres failure: autovacuum starts on a busy table but repeatedly yields because it can't acquire the lock it needs for the final truncation phase, or because it's continually canceled by conflicting DDL/lock requests, or because a long transaction keeps the xmin horizon pinned so each run reclaims nothing. The table's dead-tuple count and `relfrozenxid` age both creep up despite autovacuum appearing to "run" in the logs. GitLab's runbooks document exactly this — needing to release "PostgreSQL table locks held by autovacuum using manual vacuum" and dealing with vacuum being "ineffective at cleaning up due to a long-lived transaction (possibly on a replica due to a replication slot with a stale xmin)." The diagnosis requires correlating `pg_stat_progress_vacuum` (is it actually making progress?), `pg_stat_activity` (is something holding xmin?), and `pg_replication_slots` (is a stale slot pinning xmin globally?). The fix depends on the culprit: kill the long transaction, drop the stale replication slot, or schedule a manual `VACUUM (FREEZE, VERBOSE)` during a maintenance window. The meta-lesson: "autovacuum is running" in the logs does not mean "autovacuum is reclaiming anything" — you must check that it's making *progress against the horizon*, not just executing. ([Source](https://docs.gitlab.com/administration/troubleshooting/postgresql/))

### 11. READ COMMITTED vs REPEATABLE READ and the purge horizon

A subtle one that bites teams who tune isolation levels for "performance" without understanding the MVCC cost. A reporting workload on InnoDB ran under the default `REPEATABLE READ`, where the read view is taken once at the first read and held for the whole transaction. Long report transactions therefore pinned the purge horizon for their full duration, and the history list length grew during reporting windows. Switching the *reporting* sessions to `READ COMMITTED` — where InnoDB takes a *fresh* read view at the start of each statement — meant each statement only pinned undo for its own (short) duration, letting purge advance between statements. The history list stopped climbing. The same logic applies in Postgres: a `READ COMMITTED` transaction re-snapshots per statement, so it holds the xmin horizon back only as far as its current statement, while a `REPEATABLE READ` transaction holds it for the whole transaction. This is a real, load-bearing reason `READ COMMITTED` is the default in many shops despite its weaker guarantees: it is *gentler on the cleaner*. The tradeoff is that you give up cross-statement repeatability — which is exactly the kind of decision the [isolation levels](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) discussion exists to make deliberately rather than accidentally. (Note the Percona case found `READ COMMITTED` resolved the issue on stock MySQL 5.6 but *not* on Amazon Aurora, whose storage layer changes the calculus — a reminder that "InnoDB" behavior is not identical across distributions.)

## When MVCC's costs are acceptable — and when to fight them

MVCC is not optional on these engines; it's how they work. The real decision is how hard you fight its second-order costs, and that depends on your workload.

**Lean into MVCC as-is when:**

- Your transactions are short. Short transactions keep the xmin/purge horizon moving, which makes vacuum and purge effortless. This is the single biggest lever, and it's free.
- Your write churn is moderate relative to your hardware's vacuum/purge throughput. If autovacuum comfortably keeps `n_dead_tup` low and `age(relfrozenxid)` flat, the defaults are fine.
- You value the latency guarantee — readers never block writers — more than you mind the background work. For most OLTP apps this trade is exactly right, which is why MVCC won.
- You're read-heavy with a stable working set; index-only scans on an all-visible heap are extremely fast, and bloat stays bounded.

**Fight the costs deliberately when:**

- You have hot small tables updated millions of times a day. Set aggressive per-table autovacuum (`autovacuum_vacuum_scale_factor = 0.01`), lower `fillfactor` to enable HOT, drop indexes on churning columns, and consider whether a queue belongs in Postgres at all versus Redis/Kafka.
- You run anything long — reports, backups, migrations, analytics. Route them to a replica, use `READ COMMITTED` where repeatability isn't needed, chunk big DELETEs/UPDATEs into batches, and set `statement_timeout`/`idle_in_transaction_session_timeout` so a leak can't pin the horizon.
- You're approaching terabyte tables or billions of rows on Postgres. Partition or shard *before* vacuum and wraparound force your hand — the Notion and Figma postmortems are the warning shots. Per-table size, not total data size, is what makes vacuum tractable.
- You're on a write-amplifying, many-index schema and bloat is winning. This is where InnoDB's undo-log design (and MyRocks' LSM design) genuinely shine over naive Postgres; if you're on Postgres, maximize HOT and minimize indexes-on-updated-columns.

**Wire these into monitoring on day one, not after the outage:**

- Postgres: `age(relfrozenxid)` per table (alert at 1.5B), `n_dead_tup` per table, oldest `backend_xmin` age in `pg_stat_activity`, stale `pg_replication_slots`, autovacuum progress.
- InnoDB: history list length / `trx_rseg_history_len` (alert in the low millions), oldest `innodb_trx.trx_started`, undo tablespace size.

The throughline of every case study above is the same sentence, and it's the one thing to take away if you take away nothing else: **MVCC trades lock-wait latency for background cleanup work, and that cleanup can only proceed past your oldest open transaction.** Keep transactions short, watch the cleaner's backlog, and respect the XID clock on Postgres. Do those three things and MVCC is the invisible miracle that lets a thousand readers and writers share a database without ever waiting on each other. Ignore them and it's the mechanism by which one forgotten `BEGIN;` takes your company offline for 41 hours.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 7 (Transactions) — the canonical treatment of snapshot isolation and the anomalies MVCC does and doesn't prevent.
- PostgreSQL documentation: "Concurrency Control" (Chapter 13) and "Routine Vacuuming" (Chapter 25), plus the `README.HOT` in the source tree.
- MySQL Reference Manual: "InnoDB Multi-Versioning" and "Purge Configuration."
- Sibling posts on this blog: [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), [B-trees: how database indexes really work](/blog/software-development/database/b-trees-how-database-indexes-work), and [LSM trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines).

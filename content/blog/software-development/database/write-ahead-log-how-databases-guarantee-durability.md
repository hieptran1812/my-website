---
title: "Write-Ahead Logging: How Databases Guarantee Durability"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "How databases keep the D in ACID: why you can't fsync data pages on every commit, how write-ahead logging turns durability into one sequential sync, and how Postgres and InnoDB recover, protect torn pages, replicate, and tune it."
tags:
  [
    "write-ahead-log",
    "wal",
    "durability",
    "acid",
    "postgres",
    "innodb",
    "crash-recovery",
    "checkpoint",
    "redo-log",
    "database-internals",
    "fsync",
    "replication",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-1.webp"
---

There is a question every backend engineer eventually has to answer for real, usually at 3 a.m. with a pager going off: when your application got back "OK, committed" from the database, and then the power died one millisecond later, is that row still there when the box comes back up? Most engineers assume the answer is yes and never think about why. The "why" is one of the most elegant pieces of engineering in all of systems software, and it is almost never what people guess.

The naive mental model is that "commit" means "the database wrote my row to disk." If that were true, every commit would have to find the data page that holds the row, write it back to its home location on disk, and force that write all the way through the operating system and the drive's volatile cache to the physical platters or flash cells. That is a random-location write followed by an `fsync`. On a spinning disk it is a seek plus a rotation; on flash it is a program-erase cycle. Either way, it is the single most expensive thing the storage stack can do, and a busy OLTP database does tens of thousands of commits per second. If each one paid for a random page write plus a sync, no production database would ever hit the throughput numbers we take for granted.

The trick that makes it work — the trick that is, in some sense, the entire reason durable databases are fast — is the **write-ahead log** (WAL). Instead of writing the changed data page on commit, the database writes a tiny description of the change to a separate, append-only log file, and forces *that* to disk. The data page itself is left dirty in memory and written out later, lazily, in big batches. Durability is preserved because the log is enough to reconstruct any change that had not yet reached its data page. The diagram below is the mental model for everything that follows: a commit is one sequential append-and-`fsync` to the log, and the data pages drift to disk on their own schedule.

## The mental model: commit is a log append, not a page write

![The WAL write path: a transaction commits when its WAL record is fsynced, not when its data page reaches disk](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-1.webp)

Read the figure left to right, top to bottom. A transaction modifies a row. The database does **not** go to disk to do that — it pulls the relevant page into the in-memory buffer pool (if it is not already there), mutates it in place, and marks the page **dirty** (changed since last written to disk). Before that dirty page is allowed to go anywhere, the database builds a **WAL record** describing exactly what changed and appends it to the in-memory WAL buffer. At commit, the database does one thing that touches stable storage: it `fsync`s the WAL up through this transaction's record. The instant that sync returns, the transaction is durable, and the client gets its acknowledgment. The dirty data page is still sitting in memory, and it will stay there until a background process — the checkpointer or page cleaner — gets around to writing it out.

This inversion is the whole game. The expensive, durability-defining operation has moved off the data files and onto a single sequential log. A sequential append to one file is the friendliest possible I/O pattern: no seeks, the OS and drive can coalesce it, and — crucially — many concurrent commits can ride a single `fsync`. PostgreSQL's own documentation states the principle plainly: "only the WAL file needs to be flushed to disk to guarantee that a transaction is committed, rather than every data file changed by the transaction," and "when the server is processing many small concurrent transactions, one `fsync` of the WAL file may suffice to commit many transactions" ([PostgreSQL: Reliability and the Write-Ahead Log](https://www.postgresql.org/docs/current/wal-intro.html)).

> A write-ahead log is the bet that you can always reconstruct a data page from its log records, so you never have to pay for a durable data-page write on the commit hot path.

The rest of this article is a tour of that bet: what "committed" really means, why forcing data pages is a non-starter, the exact ordering rule that makes the log sufficient, how checkpoints bound recovery, how crash recovery replays the log, how torn pages are defeated, how `fsync` can lie to you (the famous fsyncgate), how the same log powers replication and point-in-time recovery, how MySQL juggles *two* logs, and how to tune all of it. The treatment leans on Postgres and InnoDB because they are the two engines most people run, but the ideas trace straight back to ARIES and to the durability chapter of Martin Kleppmann's *Designing Data-Intensive Applications*.

## Why this is harder than it looks

Most explanations of durability stop at "the database writes a log." That is true and useless. The interesting part is the gap between what engineers assume the storage stack guarantees and what it actually guarantees. Almost every durability bug in real systems lives in that gap.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "Commit means my data is on disk." | The row's page is written and synced at commit. | At commit only the WAL is synced; the data page may sit dirty in memory for minutes, written lazily by a checkpoint. Durability comes from the log, not the page. |
| "`write()` puts data on disk." | After `write()` returns, it's safe. | `write()` only copies into the OS page cache. Without `fsync`/`fdatasync` the data lives in volatile RAM and a power cut loses it. |
| "`fsync` either works or returns an error I can act on." | A failed sync is reported; I retry it. | On Linux a writeback error can be reported once and then the dirty pages are marked clean — a retry returns success while the data is gone. This is fsyncgate. |
| "A page write is atomic." | The 8 KB page lands all-or-nothing. | A page is many 512 B / 4 KB sectors; a crash mid-write leaves a *torn page* — new and old bytes mixed. Redo alone can't fix a corrupt base image. |
| "The disk writes data when I tell it to." | `fsync` flushes to the platters/cells. | Drives have volatile write caches. Unless the cache is honored by `fsync` (or has power-loss protection), an acknowledged sync can still be lost. |
| "Replication and durability are the same thing." | If it replicated, it's safe. | Local durability (survives a process/OS crash on this node) and replicated durability (survives losing this node) are different guarantees, with different `synchronous_commit` settings and different failure modes. |

Hold these in mind. The WAL is the mechanism that lets a database give a *precise* answer to "what survives what," instead of the hand-wave most application code relies on. Every tuning knob we cover later is really a choice about which row of this table you are willing to accept.

## 1. The D in ACID, and what "committed" really means

Durability is the D in ACID: once a transaction commits, its effects survive subsequent failures. But "failures" is not one thing, and a serious system is explicit about the failure model it tolerates. Kleppmann, in the durability discussion of Chapter 7 of *Designing Data-Intensive Applications*, makes the point that durability is not absolute — it is durability *against a stated class of faults*. A single-node database with a WAL on local disk promises: if the database process crashes, or the operating system panics, or the machine loses power, any transaction that returned a successful commit will be present after restart. It does **not** promise survival of the disk physically dying, or the machine catching fire, or a cosmic-ray bit flip that the storage layer does not catch. Those require replication, backups, and checksums — different mechanisms layered on top.

So the honest definition of "committed" is procedural, not magical. A transaction is committed when **the information required to redo it has reached stable storage** — storage that survives a power cut. In a WAL system, that information is the set of WAL records for the transaction, ending in a commit record, flushed to durable media. It is emphatically *not* the data pages. This is the single most counterintuitive fact about how databases work: at the moment your transaction is durable, the actual table data on disk may still hold the *old* values. The new values exist only in two places: the volatile buffer pool, and the durable log. If the machine crashes right then, the buffer pool evaporates, the data files still show old values, and recovery reconstructs the new values by replaying the log. That is why it works.

We can make the timing concrete. Consider a single `UPDATE accounts SET balance = balance - 100 WHERE id = 42` followed by `COMMIT`. Here is the sequence of events that actually happens inside Postgres, with the durable boundary marked:

```
1. parse + plan UPDATE
2. read page for id=42 into buffer pool (if not cached)
3. mutate the tuple in the buffered page  -> page is now DIRTY (in RAM)
4. build a WAL record (heap update) in the WAL buffer (in RAM)
5. build a WAL commit record in the WAL buffer (in RAM)
6. fsync the WAL up to the commit record's LSN   <-- DURABLE BOUNDARY
7. return COMMIT to the client
   ... minutes later ...
8. checkpointer writes the dirty page to its data file
9. (eventually) fsync the data file during the next checkpoint
```

Steps 1 through 5 are all in volatile memory. The transaction becomes durable at step 6, the WAL sync — not at step 8, the data page write, which might be five minutes later. Steps 8 and 9 are not on the commit path at all; they are background housekeeping. If you internalize one thing from this article, make it the location of that durable boundary. It is the reason a database can commit 50,000 transactions per second on hardware that can only do a few hundred random synced page writes per second.

You can watch the WAL advance in real time. In Postgres, `pg_current_wal_lsn()` returns the current write position as a **log sequence number** (LSN) — we will dissect LSNs shortly. Run a write and watch the number move:

```sql
SELECT pg_current_wal_lsn();          -- e.g. 0/16B3F20
UPDATE accounts SET balance = balance - 100 WHERE id = 42;
COMMIT;
SELECT pg_current_wal_lsn();          -- e.g. 0/16B40A8 -- advanced by the bytes of WAL we wrote
```

The difference between the two LSNs is the exact number of bytes of WAL that commit generated. That number, multiplied by your commit rate, is your WAL write bandwidth — and it is the first thing to look at when WAL volume becomes a problem.

## 2. Why you can't just fsync the data pages

The obvious objection to all of this is: why not skip the log? Just write the data page and `fsync` it at commit. The data is then on disk, durability is trivially satisfied, and there is no second file to manage. The reason no high-throughput database does this is a combination of three concrete costs: random I/O, write amplification, and torn pages.

![Force vs no-force: forcing data pages on commit means many random syncs; the WAL turns it into one sequential sync](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-3.webp)

The figure contrasts the two buffer-management policies. The naive policy on the left is called **force**: force all of a transaction's dirty pages to disk before the commit returns. The WAL policy on the right is **no-force**: do not force the data pages at commit; instead force only the log. (The companion policy is **steal**: allow a dirty page to be written to disk *before* the transaction that dirtied it commits, which the WAL also makes safe because the log carries undo information. We will come back to steal in the recovery section.)

**Random I/O.** A transaction that touches three rows in three different tables, plus their indexes, might dirty a dozen pages scattered all over the data files. Forcing them means a dozen writes to a dozen different disk locations, each potentially a seek. On a 7200-RPM spinning disk a seek is roughly 8–10 ms; you are limited to ~100 such operations per second. Flash removes the seek but not the problem: random 4 KB/8 KB writes trigger the SSD's flash-translation-layer garbage collection, and sustained random write throughput on consumer and even enterprise SSDs is commonly 5–10× lower than sequential. Contrast a WAL append: it is one write to the end of one file, the most sequential pattern there is. The OS merges it with adjacent appends; the drive's cache absorbs it; one `fsync` covers it all.

**Write amplification.** Pages are big — 8 KB in Postgres, 16 KB by default in InnoDB. To make a 100-byte change durable under a force policy, you write the whole 8 KB page. That is an 80× amplification before you even count the index pages. The WAL record for the same change is on the order of a hundred bytes plus a small header. The log is dramatically *smaller* per change than the page it describes, which is the second reason the sequential append wins: you are syncing far fewer bytes.

**Group commit.** The killer advantage of logging only is that one sync can commit many transactions. When 200 transactions all want to commit within the same few hundred microseconds, their commit records are all sitting contiguously at the end of the WAL buffer. A single `fsync` of the WAL flushes all 200 commit records at once, and all 200 transactions become durable together. This is **group commit**, and it is why WAL throughput scales with concurrency instead of collapsing under it. Under a force policy there is no equivalent — each transaction's pages are at different disk locations and cannot be coalesced into one sync.

Here is a back-of-the-envelope that shows the gap. Say each commit dirties 8 pages and you have a drive that can do 500 synced random 8 KB writes per second.

```
Force policy:
  500 synced writes/s ÷ 8 pages/commit = ~62 commits/s   (durability-bound)

No-force + group commit (WAL):
  one sync flushes a batch of, say, 100 commit records
  drive does ~500 syncs/s sequentially
  500 syncs/s × 100 commits/sync = ~50,000 commits/s     (concurrency-bound)
```

The two regimes are three orders of magnitude apart, and the only difference is *what* you sync. That is the entire economic argument for the write-ahead log. The cost you pay is complexity: you now have a log to write, a recovery procedure to replay it, and a checkpoint mechanism to keep the log bounded. The next sections build each of those.

## 3. The write-ahead rule and the LSN

The WAL is only safe if one invariant holds absolutely: **the log record describing a change must be on durable storage before the data page holding that change is written to disk.** This is the *write-ahead rule*, and it is the source of the whole technique's name. The PostgreSQL manual states it as: "changes to data files ... must be written only after those changes have been logged, that is, after WAL records describing the changes have been flushed to permanent storage." ARIES, the canonical recovery algorithm we cover in section 5, states the same thing more formally: "all log records for an updated page are written to non-volatile storage before the page itself is allowed to be over-written," precisely so that the undo information required by a steal policy is guaranteed present after a crash ([ARIES: A Transaction Recovery Method...](https://blog.acolyer.org/2016/01/08/aries/)).

![The write-ahead rule: a modification's WAL record reaches stable storage before its dirty data page may be flushed](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-2.webp)

The timeline shows why the rule is non-negotiable. Suppose you violated it: you wrote dirty page P to disk at t1, but the WAL record for P had not been synced. Now the machine crashes. On restart, the data file holds the *new* version of P, but the log has no record of the change. If the change was part of an uncommitted transaction, recovery has no way to undo it — the log it would use to roll back does not exist on disk. The data file is now corrupt in a way nothing can repair. By forcing the log first, recovery always has at least as much information as the data files, never less. The log is the ground truth; the data files are a lazily-updated cache of it.

Enforcing this rule requires bookkeeping, and the bookkeeping unit is the **LSN** — the log sequence number.

![Anatomy of a WAL record and the LSN: a monotonic byte offset that orders the whole log and stamps each page](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-8.webp)

An LSN is, concretely, a monotonically increasing 64-bit number that is the byte offset of a record within the logical log stream. In Postgres it is printed as two 32-bit halves, `XLogRecPtr` style, like `0/16B3FC8`. In InnoDB it is a plain 8-byte counter. Because it is a byte offset, an LSN both *identifies* a record and *orders* every record: if record A has a smaller LSN than record B, A was written first. The figure shows the append-only stream of records, each tagged with its LSN, and the structure of one record: its own LSN, a back-pointer to the previous record (used to walk a transaction's changes in reverse during undo), the transaction ID, the redo payload (how to reapply the change), and undo/CRC information.

The mechanism that enforces the write-ahead rule is dead simple given LSNs: **every data page header stores the LSN of the last WAL record that modified it** (`pd_lsn` in Postgres, the FIL page LSN in InnoDB). Before the buffer manager writes a dirty page to disk, it checks: has the WAL been flushed up to at least this page's `pd_lsn`? If not, it first flushes the WAL to that point, then writes the page. One comparison, and the invariant is guaranteed. This is why the page stamps its LSN — so the buffer manager knows exactly how far the log must be durable before this particular page may go out.

You can dump the actual records to see all of this. Postgres ships `pg_waldump`:

```bash
# find the current WAL file and dump recent records
psql -Atc "SELECT pg_walfile_name(pg_current_wal_lsn());"
# -> 000000010000000000000016

pg_waldump 000000010000000000000016 | tail -5
# rmgr: Heap   len (rec/tot): 54/  54, tx: 7421, lsn: 0/16B3FC8, prev 0/16B3F90,
#       desc: UPDATE off 12 ... blkref #0: rel 1663/5/16385 blk 8
# rmgr: Transaction len ...,            tx: 7421, lsn: 0/16B4030, prev 0/16B3FC8,
#       desc: COMMIT 2026-06-14 12:00:01 UTC
```

Each line is one record: its resource manager (`Heap`, `Btree`, `Transaction`...), length, transaction id, its LSN, the `prev` back-pointer, and a human-readable description including which relation file and block it touched. The `COMMIT` record at LSN `0/16B4030` is the durable boundary for transaction 7421 — once the WAL is synced to that LSN, 7421 survives any crash.

## 4. Checkpoints: bounding recovery and spreading the I/O

The no-force policy has an obvious tension. If data pages are written out lazily, and you never force the issue, then after running for an hour you might have a million dirty pages in memory and an hour's worth of WAL on disk. A crash at that point would require replaying an hour of log — recovery could take longer than the downtime you are trying to avoid. Worse, the WAL would grow without bound, because you can never delete a WAL record while a dirty page that depends on it still hasn't been written. The mechanism that resolves both problems is the **checkpoint**.

![Checkpoints advance the redo start point so recovery only replays WAL written after the last checkpoint](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-4.webp)

A checkpoint does two things. First, it writes a special checkpoint record to the WAL recording the current LSN — call it the **redo point**. Second, it flushes all (or enough) currently-dirty pages to disk and `fsync`s the data files. After the checkpoint completes, every change with an LSN *before* the redo point is guaranteed to be in the data files on disk. That is the payoff: **crash recovery only ever needs to replay WAL from the last completed checkpoint forward.** Anything older is already durable in the data files. The checkpoint bounds the recovery window, and it lets the database recycle (delete or rename) WAL segments older than the redo point, bounding the log's disk footprint.

The cost is an I/O burst. Flushing every dirty page at once is a thundering herd of writes that can starve foreground queries. The fix is to *spread* the checkpoint over time. In Postgres this is `checkpoint_completion_target`: the fraction of the inter-checkpoint interval over which the dirty-page flush is smeared. With the modern default of `0.9`, the checkpointer aims to finish writing dirty pages when 90% of the time-to-next-checkpoint has elapsed, turning a spike into a gentle ramp. The relevant knobs:

```sql
SHOW checkpoint_timeout;             -- 5min default: max time between checkpoints
SHOW max_wal_size;                   -- 1GB default: soft cap; checkpoint if WAL would exceed
SHOW checkpoint_completion_target;   -- 0.9: spread the flush over 90% of the interval
SHOW min_wal_size;                   -- 80MB: floor for recycled WAL segments
```

A checkpoint is triggered by whichever comes first: `checkpoint_timeout` elapsing, or WAL accumulation approaching `max_wal_size`. The interplay is the heart of write-load tuning, and we devote section 11 to it. For now the key idea is the tradeoff the checkpoint sits on: **frequent checkpoints mean short recovery and small WAL, but more total I/O (more full-page writes, more repeated page flushes); infrequent checkpoints mean less I/O but longer recovery and larger WAL.** There is no universally correct setting — only a position on that curve that matches your tolerance for recovery time versus your steady-state I/O budget.

InnoDB has the same machinery under different names. The redo log has a fixed total capacity (`innodb_redo_log_capacity`, default 100 MB historically, often raised to several GB); as the log fills, InnoDB performs **fuzzy checkpointing**, flushing dirty pages to advance the checkpoint LSN and free redo space. The MySQL manual describes the relationship directly: when `innodb_redo_log_capacity` is reduced, "dirty pages are flushed more aggressively"; when it is increased, "flushing becomes less aggressive" ([MySQL 8.4 Reference Manual: Redo Log](https://dev.mysql.com/doc/refman/8.4/en/innodb-redo-log.html)). The sizing of the redo log is therefore the primary checkpoint-aggressiveness control in InnoDB, exactly as `max_wal_size` is in Postgres.

## 5. Crash recovery: redo from the checkpoint, undo the losers

This is where the WAL pays off. The machine crashed; the buffer pool is gone; the data files contain a confusing mix of pages — some checkpointed and current, some stale because their dirty versions never made it out, possibly some with changes from transactions that never committed (because of the steal policy). On restart, the database must turn this into a consistent state that reflects exactly the committed transactions and nothing else. The canonical algorithm is **ARIES** (Algorithm for Recovery and Isolation Exploiting Semantics), and essentially every serious engine implements a variant of it.

![ARIES recovery: an analysis pass, a redo pass that repeats all history, then an undo pass that rolls back the losers](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-5.webp)

ARIES runs three passes over the log.

**Analysis.** Starting from the last checkpoint, scan the log forward to reconstruct two things: the set of transactions that were active at the time of the crash (the **losers** — those without a commit record), and the **dirty page table**, which records the earliest LSN whose change might not yet be on disk for each dirty page. Analysis tells redo where to start and tells undo who to roll back.

**Redo.** Scan forward from the redo point and **repeat history**: reapply *every* logged change — committed or not — to bring the database to the exact state it was in at the moment of the crash. This is ARIES's famous and counterintuitive principle. Why redo changes from transactions that never committed? Because doing so is simpler and uniform: redo blindly reconstructs the crash-time state, and the subsequent undo pass cleanly removes the uncommitted parts. Redo is made cheap and idempotent by the page LSN: for each log record, compare its LSN to the LSN stamped on the target page. If the page's LSN is already greater than or equal to the record's LSN, the change is already present (the page was flushed after this change), so skip it. This is the same `pd_lsn` from the write-ahead rule, now doing double duty as a redo idempotency check — replay the log twice and you get the same result, which matters because recovery itself can crash and restart.

**Undo.** Walk backward through the log undoing every change made by a loser transaction, using the undo information (or, in physiological logging, the inverse of the redo) and the per-transaction back-pointer chain to find each loser's records in reverse. To make undo itself crash-safe, ARIES writes **compensation log records** (CLRs) as it undoes — so if recovery crashes during undo, a re-run does not undo the same change twice and never gets stuck in a loop.

When all three passes finish, the database holds exactly the committed state and opens for traffic. The whole thing rests on two properties the WAL guarantees: redo is possible because every change was logged before its page could be written (write-ahead rule), and undo is possible because the log carries enough information to reverse uncommitted changes (which is what makes the steal policy safe).

InnoDB's recovery is recognizably ARIES-shaped. On startup it finds the checkpoint LSN in the redo log header, scans forward applying redo records to bring pages up to date, then uses the rollback segments (undo logs) to roll back transactions that were not committed. The MySQL manual notes that "redo log application is performed during initialization, before accepting any connections," and is skipped entirely if a clean shutdown flushed everything ([MySQL 8.4 Reference Manual: Redo Log](https://dev.mysql.com/doc/refman/8.4/en/innodb-redo-log.html)). Postgres's recovery is the redo half of ARIES applied to physical/physiological WAL, with no separate undo pass because Postgres keeps old row versions in the heap (its MVCC model) rather than overwriting in place — uncommitted changes are simply never made visible and are reclaimed by `VACUUM`, so there is nothing to physically undo at recovery. This is a deep design fork worth understanding, and it connects directly to the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb): InnoDB overwrites in place and needs undo logs both for MVCC reads and for recovery rollback, while Postgres appends new tuple versions and needs neither for recovery.

You rarely run recovery by hand, but you can observe it. After a crash, the Postgres log shows the redo trajectory:

```
LOG:  database system was interrupted; last known up at 2026-06-14 11:58:03 UTC
LOG:  database system was not properly shut down; automatic recovery in progress
LOG:  redo starts at 0/16A0028
LOG:  invalid record length at 0/16B40A8: wanted 24, got 0
LOG:  redo done at 0/16B4030 system usage: CPU ...
LOG:  database system is ready to accept connections
```

"redo starts at 0/16A0028" is the redo point from the last checkpoint; "redo done at 0/16B4030" is where the WAL ran out (the last valid record before the torn tail). The distance between them is exactly how much log recovery had to replay — and it is bounded by your checkpoint settings, which is the whole reason checkpoints exist.

## 6. Torn pages: full-page writes and the doublewrite buffer

There is a gap in the recovery story that, left unaddressed, makes the WAL insufficient. Recovery's redo pass reapplies a change *to an existing page* — a heap update record says "at offset 12 on this page, change these bytes." That only works if the page on disk is a coherent base image to apply the change onto. But a page is not written atomically. An 8 KB Postgres page is sixteen 512-byte sectors (or two 4 KB sectors); a 16 KB InnoDB page is more. If the OS or drive crashes after writing some of those sectors but not all, you get a **torn page**: half new bytes, half old bytes, a page that corresponds to no consistent state. The PostgreSQL docs put it precisely: "a page write that is in process during an operating system crash might be only partially completed, leading to an on-disk page that contains a mix of old and new data." Redo cannot repair this — applying a delta to a corrupt base image yields more corruption.

![Torn pages defeated: Postgres logs a full-page image after each checkpoint; InnoDB stages writes in a doublewrite buffer](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-6.webp)

The two engines solve this in two characteristically different ways, but both come down to *keeping a known-good copy of the whole page somewhere durable, so recovery can overwrite the tear instead of trying to patch it.*

**PostgreSQL: `full_page_writes`.** The first time a page is modified after each checkpoint, Postgres writes the *entire* page image into the WAL, not just the delta. After that, until the next checkpoint, modifications to that page log only the delta. On recovery, if a page might be torn, redo finds the full-page image in the WAL and restores the whole page wholesale, then applies subsequent deltas on top. The cost is WAL volume: right after a checkpoint there is a burst of full-page images as hot pages get touched for the first time, and these dominate WAL size on many workloads. This is also why making checkpoints *less* frequent reduces WAL volume — fewer "first writes after a checkpoint" means fewer full-page images. EDB's analysis of full-page writes shows this burst-then-taper pattern clearly and is the reason `full_page_writes` interacts so strongly with checkpoint tuning ([On the impact of full-page writes — EDB](https://www.enterprisedb.com/blog/impact-full-page-writes)).

```sql
SHOW full_page_writes;   -- on (default). Turn off ONLY on storage with atomic page writes.
SHOW wal_compression;    -- off by default; on => compress full-page images to cut the burst
```

**InnoDB: the doublewrite buffer.** Rather than bloat the redo log with page images, InnoDB writes every page *twice*. Before flushing dirty pages to their real locations in the data files, it writes them as a large sequential chunk to a dedicated doublewrite area and issues a single `fsync`; only then does it write them to their final, scattered positions. If a crash tears a page in its final position, recovery finds the intact copy in the doublewrite buffer and restores it. The manual is careful to note this is not 2× the I/O cost: "data is written to the doublewrite buffer in a large sequential chunk, with a single `fsync()` call" ([MySQL 8.4 Reference Manual: Doublewrite Buffer](https://dev.mysql.com/doc/refman/8.4/en/innodb-doublewrite-buffer.html)).

```sql
SHOW VARIABLES LIKE 'innodb_doublewrite';   -- ON (default) / DETECT_AND_RECOVER / DETECT_ONLY / OFF
SHOW VARIABLES LIKE 'innodb_flush_method';  -- O_DIRECT avoids double-caching page writes in the OS
```

The tradeoff between the two approaches is real and measurable. Postgres pays in WAL volume (and therefore in WAL bandwidth, archiving cost, and replication traffic) but keeps the data-file write path single-write. InnoDB keeps the redo log small but pays a second write for every page flush (mitigated by the sequential-chunk batching). Percona's side-by-side analysis frames it well: full-page writes bloat the log, doublewrite bloats the page-flush path, and both can be disabled *only* on storage that guarantees atomic page writes — for example certain NVMe devices, or InnoDB's automatic disabling of doublewrite on Fusion-io hardware with atomic-write support ([A Tale of Two Databases: torn pages — Percona](https://www.percona.com/blog/a-tale-of-two-databases-how-postgresql-and-mysql-handle-torn-pages/)). Turn either off on ordinary storage and you are betting your data on page-write atomicity the hardware does not actually promise.

## 7. fsync, O_DIRECT, and the buffered-I/O minefield

Everything above assumes that when the database calls `fsync` and it returns success, the data is durable. That assumption is the foundation of the entire edifice, and on Linux it has a crack in it large enough to lose committed data through. Understanding this crack — and how databases now defend against it — is the difference between "we have a WAL, we're safe" and actually being safe.

Start with the layers. When the database does `write()` on a WAL or data file opened normally (buffered I/O), the bytes are copied into the kernel's **page cache** — volatile RAM — and `write()` returns. Nothing is on disk yet. Later, either the kernel's background writeback or an explicit `fsync()` pushes those dirty page-cache pages down to the drive. So durability requires `fsync` (or `fdatasync`, which skips syncing inode metadata you do not need), and a database that forgot to `fsync` its WAL would acknowledge commits that a power cut erases. This is the first and most basic rule, and modern engines get it right: Postgres `fsync`s the WAL according to `synchronous_commit` and `wal_sync_method`; InnoDB `fsync`s the redo log according to `innodb_flush_log_at_trx_commit`.

The subtler problem is what happens when `fsync` *fails*. You would assume a failed `fsync` reports an error you can handle, and that the dirty data is still in the page cache so you can retry. On Linux, historically, neither was reliably true — and this is the famous **fsyncgate** of 2018.

In March 2018, Craig Ringer reported to the PostgreSQL hackers list that Postgres's handling of `fsync` errors was unsafe and risked data loss ([fsyncgate thread — postgresql.org](https://www.postgresql.org/message-id/CAMsr+YHh+5Oq4xziwwoEfhoTZgr07vdGG+hu=1adXx59aTeaoQ@mail.gmail.com)). The mechanism, dissected at the Linux Filesystem, Storage, and Memory-Management summit and written up by LWN, is brutal in its simplicity. When a buffered write fails at writeback time (a transient I/O error, a thin-provisioned volume hitting its limit, a USB drive yanked), several Linux filesystems would mark the failed dirty pages **clean** and clear the error. The next `fsync` then finds no dirty pages and no error to report — and returns **success**. As the LWN write-up summarizes, "errors could easily be lost so that no application would ever see them," and after a writeback failure the kernel "clears error status and marks dirty pages as clean, enabling silent data corruption" ([PostgreSQL's fsync() surprise — LWN](https://lwn.net/Articles/752063/)).

The consequence for a WAL database is severe. Postgres's checkpointer issues the `fsync` that makes data-file writes durable, but the *backends* do the actual `write()` calls. If a backend's writeback failed and the error was consumed (or attributed to a different file descriptor), the checkpointer's later `fsync` could return success, the checkpoint would complete and advance the redo point — discarding the WAL that could have recovered the lost write — and the database would believe durable data that was, in fact, gone. The data was already discarded from the cache; **retrying the `fsync` cannot bring it back**, because there is nothing left to write.

Two fixes followed, one in each layer.

In the kernel, Jeff Layton's `errseq_t` work (Linux 4.13+) attached a 32-bit error-and-sequence value to each inode's `address_space`, so a writeback error becomes visible to *every* file descriptor that `fsync`s afterward, closing the "the wrong fd ate the error" hole. But — and this is the part people miss — `errseq_t` only makes the error *visible if you were watching*; it does not bring the discarded data back. As the LWN coverage notes, a process that opens the file *after* the error occurred (exactly the checkpointer's situation) can still miss a pre-existing failure.

In PostgreSQL, the fix was to stop trusting `fsync` retries entirely. The `data_sync_retry` parameter (default `off`), committed by Craig Ringer in November 2018, makes Postgres treat an `fsync` failure on a data file as a `PANIC`: the database crashes hard, deliberately, rather than continuing in a state where it might advance the redo point past data the kernel has silently dropped. After the PANIC-and-restart, crash recovery re-derives the data from the WAL — which is the only copy that is still trustworthy. It is a violent fix, but it is the correct one: when `fsync` fails, the safe assumption is that you have lost data from the cache, so you must rebuild from the log rather than reason about what survived.

The deepest fix is to bypass the buffered-I/O layer entirely with **direct I/O** (`O_DIRECT`), where writes go straight to the device without sitting in the page cache, giving the database explicit control over what is in flight and when it is durable. InnoDB has long supported `innodb_flush_method=O_DIRECT` to avoid double-buffering page writes in the OS cache. Postgres historically used buffered I/O for everything and has been moving toward direct I/O (the `io_method`/`debug_io_direct` work), which the hackers acknowledged is "a metric ton of work." The practical takeaways for an operator are concrete:

- Keep `full_page_writes` and `innodb_doublewrite` on unless your storage genuinely guarantees atomic page writes.
- Make sure your drives' volatile write cache is either disabled or backed by power-loss protection; an `fsync` that the drive cache silently swallows is fsyncgate one layer lower.
- Treat an `fsync` error in the database log as an emergency — it means the durability contract was violated, not a transient blip to ignore.
- Leave `data_sync_retry = off` (the PANIC behavior). It feels scary; it is the safe default.

## 8. Group commit and the synchronous_commit spectrum

Section 2 introduced group commit as the reason WAL throughput scales with concurrency. It is worth making the mechanism and its tuning explicit, because the single most impactful durability/performance knob in both engines lives here: *how durable does a commit have to be before we acknowledge it?*

Group commit works because commits are batched at the sync. When transaction T1 calls `fsync` on the WAL, transactions T2..Tn whose commit records were appended in the same window do not each issue their own sync — they wait for T1's sync to complete and then all return. The wider you make that window, the more commits ride each sync, the higher the throughput, and the higher the per-commit latency. Postgres exposes the window directly:

```sql
SHOW commit_delay;    -- microseconds to wait before syncing, to accumulate a bigger batch
SHOW commit_siblings; -- only delay if at least this many other txns are active
```

The far more consequential knob is `synchronous_commit`, which chooses *what* the commit waits for. Its values trade durability for latency along a precise spectrum ([PostgreSQL 18 docs: WAL configuration](https://www.postgresql.org/docs/current/runtime-config-wal.html)):

| `synchronous_commit` | Commit waits for | Survives DB crash? | Survives this node's OS crash? | Survives loss of this node? |
| --- | --- | --- | --- | --- |
| `off` | nothing — async; WAL flushed within a small window (≤ ~`wal_writer_delay`) | maybe (window of loss) | no (lose ≤ that window) | no |
| `local` | local WAL write + `fsync` | yes | yes | no |
| `on` (default) | local flush, and if a sync standby exists, its flush too | yes | yes | yes (standby has it) |
| `remote_write` | standby received the WAL into its OS (not yet `fsync`ed) | yes | yes | survives standby DB crash, not standby OS crash |
| `remote_apply` | standby flushed *and applied*, visible to reads there | yes | yes | yes, and read-your-writes on standby |

The most important and most misused setting is `off`. With `synchronous_commit = off`, a commit returns *before* its WAL record is flushed; a background WAL writer flushes shortly after. As the docs state, this makes it "possible to lose some recent (< 1s by default) allegedly-committed transactions in case of a server crash." Crucially, `off` does **not** risk *corruption* or *torn transactions* — the WAL still goes out in order, recovery still produces a consistent state — it only risks losing the most recent sliver of committed work. For a workload of analytics ingestion or click events where losing the last few hundred milliseconds on a crash is acceptable, `synchronous_commit = off` can multiply throughput by removing the sync from the hot path. For anything touching money, it is a footgun. The right move is often per-transaction: keep the cluster default at `on`, and set `SET LOCAL synchronous_commit = off` inside the specific high-volume, loss-tolerant transactions.

InnoDB's equivalent is `innodb_flush_log_at_trx_commit`, with three values whose semantics are the mirror image of the Postgres spectrum:

| Value | Behavior at commit | Data loss on crash | ACID? |
| --- | --- | --- | --- |
| `1` (default) | write redo to log + `fsync` to disk every commit | none | full ACID |
| `2` | write redo to the OS cache every commit; `fsync` once per second | survives `mysqld` crash; OS/power crash loses ≤ ~1s | not strictly |
| `0` | write + `fsync` once per second; commits don't touch the log | even a `mysqld` crash can lose ≤ ~1s | not strictly |

The key distinction is between `0` and `2`: with `2` the redo *is* handed to the OS at every commit, so if only the `mysqld` process crashes (not the OS), the OS still flushes it and nothing is lost; only an OS/power crash exposes the up-to-1-second gap. With `0`, even a clean `mysqld` crash can lose up to a second because commits don't even reach the OS cache. The default `1` is the only fully-ACID value, and it is the right default for the same reason `synchronous_commit = on` is: it makes the durability boundary coincide with the commit acknowledgment.

> Every durability knob is the same question wearing different clothes: how much recently-acknowledged work are you willing to lose, and to which failures? Answer that for each table, not once for the cluster.

## 9. One log, many uses: replication and point-in-time recovery

The most beautiful property of the write-ahead log is that *the exact same byte stream that guarantees local durability is also a complete, ordered, replayable description of every change the database has ever made.* That is an extraordinarily useful artifact, and databases exploit it for two more capabilities that, on the surface, look unrelated to crash safety: replication and point-in-time recovery.

![One log, many consumers: the same WAL stream drives crash recovery, streaming replication, and point-in-time recovery](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-9.webp)

**Streaming replication.** A standby server is just a database that, instead of generating its own WAL, receives the primary's WAL stream over the network and replays it continuously — which is precisely the redo pass of recovery, run forever. Postgres's `walsender` process ships WAL records to a `walreceiver` on the standby, which applies them; the standby is always "recovering" toward the primary's current LSN. Because the standby applies the same physical changes in the same order, it is a byte-for-byte copy. This is **physical replication**, and it is why the `synchronous_commit` levels `remote_write` / `on` / `remote_apply` exist — they let the primary wait until the standby has received/flushed/applied the WAL before acknowledging the commit, extending durability across machines. (Logical replication, which decodes the WAL into row-level change events for selective or cross-version replication, is a different consumer of the same log; the [replication deep dive](/blog/software-development/database/database-replication-sync-async-logical-physical) covers the sync/async and physical/logical axes in full.)

**Point-in-time recovery (PITR).** If you keep a base backup of the data files plus *every* WAL segment generated since that backup, you can reconstruct the database state at *any* instant in that window: restore the base backup, then replay WAL up to a chosen LSN or timestamp, and stop. The PostgreSQL docs make a subtle but powerful observation: the base backup "doesn't have to be an instantaneous snapshot — if it is made over some period of time, then replaying the WAL for that period will fix any internal inconsistencies." The backup can be smeared across a busy hour, taken with the database fully live, and the WAL replay heals it into a consistent state. The setup is archiving WAL as it's filled:

```ini
# postgresql.conf
wal_level = replica            # enough WAL detail for archiving + physical replication
archive_mode = on
archive_command = 'test ! -f /archive/%f && cp %p /archive/%f'   # copy each filled segment
```

```bash
# take a base backup while the DB serves traffic
pg_basebackup -D /backups/base -Fp -Xs -P

# later: restore to a specific instant
# restore the base backup, then in postgresql.conf:
#   restore_command = 'cp /archive/%f %p'
#   recovery_target_time = '2026-06-14 11:59:30 UTC'
# Postgres replays archived WAL up to that timestamp and stops.
```

InnoDB's redo log is not designed for archiving and replay the way Postgres WAL is (it is fixed-size and recycled), so MySQL's PITR is built on the **binlog** instead — which is the cue for the next section, the most operationally tricky part of MySQL's design. The unifying insight stands: a write-ahead log is not only a durability mechanism. It is a *change capture* mechanism, and durability, replication, and PITR are three different consumers of the one ordered stream. Build the log once; get all three.

## 10. MySQL's two logs and the internal two-phase commit

Here is where MySQL diverges sharply from Postgres and where a lot of subtle bugs and confusion live. MySQL has not one durability log but **two**, owned by two different layers, and keeping them consistent requires an internal two-phase commit.

The **InnoDB redo log** is the storage engine's WAL: physical/physiological records used for crash recovery, exactly as described throughout this article. The **binary log (binlog)** is the *server layer's* logical record of changes — statement-based or row-based events — used for replication and point-in-time recovery. They serve different masters: the redo log answers "is this change durable in the storage engine?" and the binlog answers "what change should replicas apply, in what order?"

The problem is that a transaction must appear in *both* or *neither*. If a transaction is in the redo log (so it survives crash recovery on the primary) but not in the binlog, replicas never receive it and silently diverge from the primary. If it is in the binlog but not the redo log, replicas apply a change the primary rolled back. Either way the cluster is corrupt. Keeping two independent logs atomically consistent across a crash is exactly the distributed-commit problem, solved here with an internal **XA two-phase commit** coordinated by the MySQL server across the InnoDB engine and the binlog.

![MySQL's internal two-phase commit ties the InnoDB redo log to the binlog so replication never diverges from storage](/imgs/blogs/write-ahead-log-how-databases-guarantee-durability-7.webp)

The protocol, shown in the figure:

1. **Prepare.** InnoDB writes the transaction's changes to the redo log and `fsync`s it, marking the transaction `PREPARED` and stamping it with a transaction id (XID). The transaction is now recoverable in the engine but not yet committed.
2. **Binlog write.** The server writes the transaction's events (tagged with the same XID) to the binlog and `fsync`s it. This is the **commit point** of the whole transaction — once the binlog is durable, the transaction *will* commit.
3. **Commit.** InnoDB writes a commit record to the redo log, marking the transaction `COMMITTED`. This step does not even need its own `fsync` for correctness.

The magic is the crash-recovery rule that ties them together: **on recovery, a transaction found `PREPARED` in the redo log is committed if and only if its XID is present in the binlog; otherwise it is rolled back.** The binlog is the tiebreaker — the source of truth. Trace the two crash windows in the figure. If the crash hits *before* the binlog write (left branch), the XID is not in the binlog, so recovery rolls the prepared transaction back — and no replica ever saw it, so the cluster stays consistent. If the crash hits *after* the binlog write but before the InnoDB commit record (right branch), the XID *is* in the binlog, so recovery promotes the prepared transaction to committed by replaying it — matching what replicas will receive. Either way, redo log and binlog agree.

This correctness costs `fsync`s — two per transaction (redo prepare, binlog) plus the unsynced commit — which is brutal under high concurrency. The mitigation is **binlog group commit**, which batches the work into three pipelined stages, each protected by a mutex, with one leader thread doing the `fsync` for a whole batch:

- **Flush stage:** collect a batch of transactions and write their binlog events.
- **Sync stage:** the leader `fsync`s the binlog once for the entire batch.
- **Commit stage:** InnoDB commits the batch in binlog order.

This is the same group-commit idea as section 8, applied to the two-log dance: it preserves the property that *the order transactions commit in InnoDB matches the order they appear in the binlog* (so replicas replay in the right order) while amortizing the `fsync` cost across the batch. The relevant knobs:

```sql
SHOW VARIABLES LIKE 'sync_binlog';                  -- 1 = fsync binlog every commit (durable, default)
SHOW VARIABLES LIKE 'innodb_flush_log_at_trx_commit'; -- 1 = fsync redo every commit (durable, default)
SHOW VARIABLES LIKE 'binlog_group_commit_sync_delay'; -- microseconds to widen the batch window
```

For full durability *and* a crash-consistent binlog, you need both `sync_binlog = 1` and `innodb_flush_log_at_trx_commit = 1`. Relaxing either (`sync_binlog = 0`, or `innodb_flush_log_at_trx_commit = 2`) buys throughput by risking that, after a crash, the two logs disagree about the last sliver of transactions — which on a replicated cluster means a replica that has to be rebuilt. The choice is the same shape as the Postgres `synchronous_commit` choice, just with two logs to keep in step instead of one.

## 11. The durability-tuning playbook

Tuning a WAL system is not about finding magic numbers; it is about deliberately choosing a position on a small number of tradeoff curves. Here is a practical, ordered playbook for the two engines, grounded in the mechanisms above.

**Step 1 — Right-size the checkpoint interval (the single biggest lever).** In Postgres, raise `max_wal_size` and `checkpoint_timeout` so that checkpoints are driven by time, not by WAL filling up, and so they happen infrequently enough to keep full-page-write volume down. Cybertec's guidance is blunt: increasing the checkpoint distance is the primary way to reduce WAL, and `max_wal_size` should be kept "as high as possible," trading disk for far less write amplification ([Reduce WAL by increasing checkpoint distance — Cybertec](https://www.cybertec-postgresql.com/en/reduce-wal-by-increasing-checkpoint-distance/)). A heavily-loaded system commonly runs `max_wal_size` in the tens of GB.

```sql
ALTER SYSTEM SET max_wal_size = '16GB';            -- soft cap; bigger => fewer checkpoints => less WAL
ALTER SYSTEM SET checkpoint_timeout = '30min';     -- time-driven, infrequent checkpoints
ALTER SYSTEM SET checkpoint_completion_target = 0.9; -- spread the flush, no I/O spike
ALTER SYSTEM SET min_wal_size = '2GB';             -- keep recycled segments to avoid churn
SELECT pg_reload_conf();
```

The watch-out: bigger `max_wal_size` means longer crash recovery (more WAL to replay) and more disk consumed by WAL. You are explicitly buying steady-state I/O efficiency with recovery time and disk. Measure recovery time on a representative crash and make sure it fits your RTO. In InnoDB the equivalent is sizing the redo log generously:

```sql
SET GLOBAL innodb_redo_log_capacity = 8589934592;  -- 8 GB: larger => less aggressive checkpointing
```

**Step 2 — Place the WAL on fast, dedicated storage.** The WAL is the durability hot path; its `fsync` latency is your commit latency. Put `pg_wal` (and InnoDB's redo log / doublewrite files via `innodb_log_group_home_dir` and `innodb_doublewrite_dir`) on the fastest, lowest-latency device you have, ideally separate from the data files so sequential WAL writes don't contend with random data-file I/O. On the WAL device, low `fsync` latency matters more than raw bandwidth.

**Step 3 — Size the in-memory WAL buffer.** `wal_buffers` is the in-memory staging area for WAL before it's flushed. The default (auto, ~3% of shared_buffers up to 16 MB) is often too small for write-heavy, high-concurrency workloads; a larger buffer lets more commits accumulate before a sync, improving group-commit batching.

```sql
SHOW wal_buffers;
ALTER SYSTEM SET wal_buffers = '64MB';
```

**Step 4 — Cut full-page-write volume.** Beyond fewer checkpoints (step 1), enable WAL compression so the post-checkpoint burst of full-page images is compressed, and keep an eye on how much of your WAL is full-page images versus deltas:

```sql
ALTER SYSTEM SET wal_compression = 'lz4';   -- compress full-page images; small CPU cost, big WAL win
```

**Step 5 — Choose your durability level per workload, not globally.** This is section 8 turned into policy. Keep the cluster at the safe default (`synchronous_commit = on`, `innodb_flush_log_at_trx_commit = 1`, `sync_binlog = 1`). For specific high-volume, loss-tolerant transactions, relax *locally*:

```sql
-- Postgres: this transaction may lose <1s on a crash, but commits ~10x faster
BEGIN;
SET LOCAL synchronous_commit = off;
INSERT INTO clickstream_events (...) VALUES (...);   -- bulk, loss-tolerant
COMMIT;
```

**Step 6 — Keep the torn-page defenses on.** Leave `full_page_writes = on` and `innodb_doublewrite = ON` unless you have *verified* atomic page writes on your storage. Verify drive write caches are protected. These are not performance knobs to toy with; they are correctness floors.

**Step 7 — Monitor WAL generation and checkpoint behavior.** Watch WAL bytes per second (diff `pg_current_wal_lsn()` over time), the ratio of timed-vs-requested checkpoints (`pg_stat_checkpointer` / `pg_stat_bgwriter`), and recovery-time estimates. A high fraction of *requested* (WAL-size-driven) checkpoints means `max_wal_size` is too small and you are checkpointing too often.

```sql
SELECT pg_size_pretty(
  pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0')) AS total_wal_written;
SELECT num_timed, num_requested FROM pg_stat_checkpointer;  -- want num_requested low
```

The summarizing principle: **the WAL gives you a small set of dials that all trade the same three currencies — write I/O, recovery time, and recently-acknowledged data at risk.** Tuning is deciding the exchange rate that fits your business, then setting the dials to match.

## Case studies from production

These are composite incidents drawn from common WAL-related failure modes. Each follows the same arc: the symptom, the wrong first hypothesis, the actual root cause, the fix, and the lesson.

### 1. The checkpoint I/O storm that looked like a query problem

A payments service saw p99 latency spike to 2–3 seconds every five minutes, like clockwork. The on-call engineer's first hypothesis was a slow query running on a cron schedule, and they spent an hour grepping `pg_stat_statements` for a periodic offender. There was none. The real cause was the default `checkpoint_timeout = 5min` combined with an old `checkpoint_completion_target` of `0.5`: every five minutes Postgres dumped all accumulated dirty pages to disk in a tight burst, saturating the data device and starving foreground commits whose WAL `fsync` now queued behind the flood. The fix was to lengthen `checkpoint_timeout` to 15 minutes, raise `max_wal_size` to 8 GB so checkpoints stayed time-driven, and set `checkpoint_completion_target = 0.9` to smear the flush. p99 dropped to 40 ms and the periodic spike vanished. The lesson: periodic latency spikes that don't correlate with any query are almost always checkpoint I/O, and the fix is fewer, more-spread checkpoints — not query tuning.

### 2. The WAL partition that filled and froze the database

A reporting database stopped accepting writes with `PANIC: could not write to file "pg_wal/...": No space left on device`. The first hypothesis was that the data had grown and the volume needed expanding. But the *data* volume was fine — it was the **WAL** volume that filled. The root cause was a forgotten replication slot: a standby had been decommissioned without dropping its slot, so Postgres dutifully retained every WAL segment the dead standby "might still need," and WAL grew without bound until the partition filled. The fix was `SELECT pg_drop_replication_slot('dead_standby')`, after which the retained segments were recycled within minutes. The lesson: WAL retention is held hostage by the slowest consumer — an inactive replication slot, a stalled `archive_command`, or a long-running transaction can all pin WAL forever. Monitor `pg_replication_slots` and the gap between current LSN and each slot's `restart_lsn`.

### 3. The "fast" setting that lost an hour of orders

An e-commerce team, chasing throughput, set `synchronous_commit = off` cluster-wide after reading a blog post that promised a big speedup. Throughput did improve. Three weeks later a kernel panic took the primary down hard, and on restart the last ~900 ms of committed orders were simply gone — acknowledged to customers, charged to cards, and not in the database. The first hypothesis was data corruption or a replication gap. The actual cause was exactly what `synchronous_commit = off` documents: commits acknowledged before the WAL was flushed, and the sub-second window of unflushed commits evaporated in the crash. The fix was to restore `synchronous_commit = on` as the default and apply `SET LOCAL synchronous_commit = off` only to the genuinely loss-tolerant analytics-ingest path. The lesson: `off` does not corrupt anything, but it *will* lose your most recent committed work on a crash — never make it the cluster default for transactions you cannot afford to lose.

### 4. fsyncgate in miniature: the thin-provisioned volume

A staging Postgres ran on a thin-provisioned LVM volume that quietly hit its backing limit. Writes started failing at writeback time, but the database kept running and even reported successful checkpoints for a while. When it finally `PANIC`ed, recovery found the data files inconsistent with the WAL. The first hypothesis was a Postgres bug. The actual cause was the fsyncgate failure mode: the filesystem dropped the failed dirty pages and cleared the error, so an `fsync` returned success over data that had been discarded, and a checkpoint advanced the redo point past WAL that could have recovered it. The fix was operational — never thin-provision a database volume without monitoring the backing pool, and keep `data_sync_retry = off` so an `fsync` failure crashes hard and forces WAL-based recovery rather than continuing on a lie. The lesson: `fsync` success is only as trustworthy as the storage beneath it; treat any database `fsync` error as a durability emergency.

### 5. The torn page that `full_page_writes = off` created

An engineer disabled `full_page_writes` to shrink WAL volume, reasoning that the SSD "surely" wrote pages atomically. Months later an unclean power loss produced a single corrupt page — a heap page with half its bytes from a new write and half from the old. Recovery's redo pass applied deltas onto the torn base and produced garbage rows; a `SELECT` on that table errored with an invalid page header. The first hypothesis was failing hardware. The actual cause was the torn page: the SSD did *not* guarantee 8 KB atomic writes, and with no full-page image in the WAL, recovery had no clean base to restore. The fix was to restore the affected page from a backup, re-enable `full_page_writes`, and (separately) verify that the storage's atomic-write claim was real before ever revisiting the setting. The lesson: torn-page protection defends against a rare-but-catastrophic event; disabling it to save WAL is trading a guaranteed small cost for an occasional total-loss risk.

### 6. The redo log too small for the write burst

A MySQL instance handling a nightly bulk import slowed to a crawl during the import window, with `InnoDB: page cleaner` warnings in the error log and write throughput collapsing. The first hypothesis was lock contention from the import. The actual cause was an undersized redo log: with the legacy default `innodb_redo_log_capacity` of a few hundred MB, the redo log filled during the burst, forcing InnoDB into aggressive synchronous flushing (advancing the checkpoint to free redo space) that throttled every write. The fix was to raise `innodb_redo_log_capacity` to 8 GB, giving the burst room to write redo without immediately triggering desperate checkpointing. Import time dropped by more than half. The lesson: in InnoDB the redo-log size *is* the checkpoint-aggressiveness control; too small a log turns every write burst into a flushing storm, exactly mirroring Postgres's `max_wal_size`.

### 7. The replica that diverged after a crash

A MySQL primary crashed and was restarted; days later, a replica was found returning rows the primary did not have. The first hypothesis was a replication filter misconfiguration. The actual cause was a relaxed two-log durability setting: the primary ran `sync_binlog = 0`, so on the crash the binlog lost its last few transactions while the InnoDB redo log retained them. Recovery committed those redo-only transactions on the primary (they were `PREPARED` with no binlog XID only because the binlog had lost them post-`fsync`-omission), and the binlog never carried them to the replica — divergence. The fix was `sync_binlog = 1` together with `innodb_flush_log_at_trx_commit = 1`, restoring the crash-consistent two-phase commit, and a full rebuild of the diverged replica from a fresh backup. The lesson: on a replicated MySQL cluster the redo log and binlog must *both* be synced per commit, or a crash can split them and silently diverge your replicas.

### 8. The "instant" backup that wasn't consistent

A team took backups with a naive `cp -r` of the live data directory and were baffled when restores occasionally came up corrupt. The first hypothesis was disk errors during the copy. The actual cause was that a raw file copy of a live database captures data files at different moments — a torn snapshot — with no accompanying WAL to heal it. The fix was to use `pg_basebackup -Xs` (which streams the WAL generated *during* the backup alongside the data files) or a snapshot plus archived WAL, so that recovery on restore replays the WAL and resolves the internal inconsistencies. The lesson, straight from the Postgres docs: a base backup need not be instantaneous *because the WAL fixes it up* — but only if you actually keep the WAL spanning the backup. A data-file copy without its WAL is not a backup.

### 9. The long transaction that pinned the WAL

A nightly analytics job opened a transaction, ran for six hours, and during that window the WAL volume on an otherwise-modest OLTP database grew alarmingly and would not recycle. The first hypothesis was a write spike from the analytics job itself. The actual cause was subtler: the long-open transaction held back the system's ability to recycle WAL (and, relatedly, prevented `VACUUM` from cleaning up old row versions), so segments accumulated. The fix was to break the analytics job into shorter transactions and to monitor `pg_stat_activity` for transactions open longer than a threshold. The lesson: the WAL (and MVCC cleanup) can only advance as far as the oldest still-needed point; a single marathon transaction can pin both, inflating WAL and bloating tables at once.

### 10. The commit latency cliff under fsync contention

A service migrated from a dedicated NVMe WAL device to a shared network volume to cut costs, and commit latency went from sub-millisecond to 15–20 ms under load. The first hypothesis was network bandwidth saturation. The actual cause was `fsync` *latency*, not bandwidth: each WAL flush now incurred a network round-trip and the shared volume's sync latency, and because commit latency *is* WAL-`fsync` latency, every commit paid it. Group commit helped amortize it under high concurrency but could not hide it for low-concurrency, latency-sensitive paths. The fix was to move the WAL back onto a local low-latency device and keep only cold data on the network volume. The lesson: for the WAL, sync *latency* dominates throughput; provision the WAL device for the lowest `fsync` latency you can get, and never share it with high-latency storage.

## When to reach for these techniques, and when not to

The write-ahead log is not optional in a durable database — every engine you would run in production has one. The real decisions are about *how you configure the dials it exposes*.

**Reach for aggressive checkpoint/WAL tuning when:**

- You run a write-heavy OLTP workload and see periodic latency spikes synchronized with checkpoints — lengthen and spread checkpoints (`max_wal_size`, `checkpoint_timeout`, `checkpoint_completion_target`).
- WAL volume is a real cost (archiving bandwidth, replication traffic, disk) — fewer checkpoints plus `wal_compression` cut full-page-write volume directly.
- You have a dedicated low-latency device available for the WAL — separating it from data files is almost always a win on busy systems.
- Crash-recovery time is comfortably within your RTO, so you can afford the larger WAL that bigger checkpoint distances imply.

**Reach for relaxed durability (`synchronous_commit = off`, `innodb_flush_log_at_trx_commit = 2`) when:**

- The specific transactions are genuinely loss-tolerant — metrics, click events, append-only logs where losing the last sub-second on a rare crash is acceptable — and you apply the relaxation *per transaction*, not cluster-wide.
- Throughput is bottlenecked on commit `fsync` and you have measured that group commit alone isn't enough.

**Skip these adjustments — keep the safe defaults — when:**

- The data is financial, legal, or otherwise unforgivable to lose. Keep `synchronous_commit = on` / `innodb_flush_log_at_trx_commit = 1` / `sync_binlog = 1`. The throughput you'd gain is not worth the orders you'd lose.
- You're tempted to disable torn-page protection (`full_page_writes`, `innodb_doublewrite`) to save I/O. Don't, unless you have *verified* atomic page writes on your hardware. The savings are small and the downside is total page corruption on a crash.
- You're considering disabling `fsync` entirely (`fsync = off` in Postgres). This is for throwaway/CI databases only; on real data it means a crash can leave the database unrecoverable, not merely lose recent commits.
- You haven't measured. Every knob here trades the same three currencies — write I/O, recovery time, data-at-risk. Don't move a dial until you know which currency you're short on. Diff `pg_current_wal_lsn()` over time, watch checkpoint counters, and time a real recovery before you tune.

The deeper point is that the write-ahead log is the place where the abstract promise of ACID durability meets the concrete physics of storage. Understanding it changes how you read every other part of the system: replication is WAL replay over a network ([replication](/blog/software-development/database/database-replication-sync-async-logical-physical)), point-in-time recovery is WAL replay against a backup, MVCC visibility ([MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb)) determines whether recovery needs a separate undo pass, and even the [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) behind RocksDB and Cassandra begin their write path with — what else — a write-ahead log, before the memtable, for exactly the durability reason laid out here. The buffer pool and heap-file layout that the WAL protects are covered in [pages, heap files & the buffer pool](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool). One sequential, ordered, durable log turns out to be the load-bearing primitive under almost everything a database does.

## Further reading

- [PostgreSQL: Reliability and the Write-Ahead Log](https://www.postgresql.org/docs/current/wal-intro.html) and [WAL Configuration](https://www.postgresql.org/docs/current/wal-configuration.html) — the canonical statements of the WAL principle and checkpoint tuning.
- [PostgreSQL's fsync() surprise (LWN)](https://lwn.net/Articles/752063/) and the [fsyncgate hackers thread](https://www.postgresql.org/message-id/CAMsr+YHh+5Oq4xziwwoEfhoTZgr07vdGG+hu=1adXx59aTeaoQ@mail.gmail.com) — how the OS can lose your `fsync` error.
- [MySQL 8.4 Reference Manual: Redo Log](https://dev.mysql.com/doc/refman/8.4/en/innodb-redo-log.html) and [Doublewrite Buffer](https://dev.mysql.com/doc/refman/8.4/en/innodb-doublewrite-buffer.html) — InnoDB's crash-recovery and torn-page mechanisms.
- [ARIES: A Transaction Recovery Method (the morning paper)](https://blog.acolyer.org/2016/01/08/aries/) — the analysis/redo/undo algorithm every engine descends from.
- [Reduce WAL by increasing checkpoint distance (Cybertec)](https://www.cybertec-postgresql.com/en/reduce-wal-by-increasing-checkpoint-distance/) and [On the impact of full-page writes (EDB)](https://www.enterprisedb.com/blog/impact-full-page-writes) — practical checkpoint and full-page-write tuning.
- [A Tale of Two Databases: torn pages (Percona)](https://www.percona.com/blog/a-tale-of-two-databases-how-postgresql-and-mysql-handle-torn-pages/) — Postgres full-page writes vs InnoDB doublewrite, side by side.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 3 (write-ahead logs in storage engines) and Chapter 7 (durability and the failure models it tolerates).

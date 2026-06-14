---
title: "Database Locks and Deadlocks in Production"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A production engineer's field guide to database locking — shared and exclusive locks, InnoDB gap and next-key locks, Postgres lock modes, how deadlocks form and get resolved, reading the lock logs, and the anti-deadlock playbook that actually works."
tags:
  [
    "locks",
    "deadlocks",
    "postgres",
    "innodb",
    "mysql",
    "concurrency",
    "gap-locks",
    "select-for-update",
    "skip-locked",
    "transactions",
    "database",
    "performance",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 57
image: "/imgs/blogs/database-locks-and-deadlocks-in-production-1.webp"
---

The first deadlock I ever debugged in anger was on a payments service at 2 a.m. The pager said `ERROR 1213: Deadlock found when trying to get lock; try restarting transaction`. The graph said our checkout error rate had jumped from 0.01% to 4%. The on-call playbook said, helpfully, "investigate." What made it maddening was that the offending query was a plain `UPDATE accounts SET balance = balance - ? WHERE id = ?` — a one-row update on a primary key. There was no obvious second table, no obvious second lock, nothing in the code that looked like it could possibly form a cycle. And yet two of those innocent updates, run by two unrelated requests on two unrelated accounts, were killing each other.

That is the thing about database locks. They are almost entirely invisible until they aren't, and when they become visible they do so as latency cliffs, mysterious timeouts, and rollbacks that the application code never asked for. A lock is the database keeping a promise it made to you — that your transaction sees a consistent slice of the world and that nobody else corrupts the rows you are touching — and the price of that promise is that *somebody, somewhere, is waiting*. Most of the time the waiting is microseconds and nobody notices. Occasionally the waiting is forever, the engine notices, and it shoots one of your transactions in the head to break the tie.

This article is the field guide I wish I had that night. We will build up from the two primitive locks every engine has — shared and exclusive — to the full machinery: row versus table locks, intention locks and the lock hierarchy, Postgres's eight table-level lock modes and four row-level modes, InnoDB's record / gap / next-key / insert-intention zoo, how a deadlock actually forms as a cycle in a graph, how the engine detects and resolves it, how to read the raw lock logs from both databases, and — most importantly — the concrete playbook for *not* getting paged at 2 a.m.: consistent lock ordering, short transactions, `SKIP LOCKED` job queues, sharded counters, and the optimistic-versus-pessimistic decision. Intuition first, then the math, then runnable two-session SQL you can paste into a pair of `psql` or `mysql` shells to watch a deadlock happen on your own machine.

## The mental model: it's all one compatibility table

![The lock-compatibility matrix: two lock requests on the same object are granted together only where the cell is green, otherwise one waits](/imgs/blogs/database-locks-and-deadlocks-in-production-1.webp)

The diagram above is the mental model for the entire post. Strip away every name — record lock, gap lock, `FOR UPDATE`, `ACCESS EXCLUSIVE` — and what is left is a single question the engine asks millions of times a second: *transaction A holds some lock on object X, transaction B wants some lock on object X; are those two compatible?* If the cell where A's lock-mode row meets B's lock-mode column is green, both proceed. If it is red, B waits in a queue behind A until A commits or rolls back. Every locking feature in every relational database is an elaboration of that table: more lock *modes* (so the table has more rows and columns), more granular *objects* (rows, gaps, pages, tables, the whole database), and more *rules* about which modes a statement implicitly acquires.

The two foundational modes are **shared (S)** and **exclusive (X)**. A shared lock says "I am reading this and I need it to not change under me." An exclusive lock says "I am changing this and nobody else may touch it." The rule is the one every reader-writer lock in computer science uses: many readers *or* one writer, never both. S is compatible with S — any number of transactions can hold a shared lock on the same row simultaneously, because reading does not interfere with reading. X is compatible with nothing — not with another X, not even with an S — because a writer must not have the row read or written out from under it. That single asymmetry is the seed of every contention problem in this article. Two reads never block. A read and a write block. Two writes block. The entire performance story of a write-heavy system is the story of how often you force two transactions to want incompatible modes on the same object at the same time.

> A lock is not a thing the database *has*; it is a promise the database *keeps*. The cost of every consistency guarantee is paid in someone else's wait time.

Before we go deeper, name the mismatch directly, because almost every wrong intuition about locks comes from imagining the database as a single-threaded program that happens to be fast.

| You assume | The naive mental model | The reality in production |
| --- | --- | --- |
| "An `UPDATE` on a primary key locks one row." | One row, one lock, done. | It locks an index record *and*, under `REPEATABLE READ`, possibly a gap before it; secondary-index entries get locked too. |
| "Two transactions touching different rows can't conflict." | Disjoint rows → disjoint locks. | Gap locks and next-key locks cover *ranges*, so different rows in the same range collide. |
| "A `SELECT` takes no locks." | Reads are free. | A plain `SELECT` takes a table-level `ACCESS SHARE` (Postgres) that conflicts with `DROP`/`TRUNCATE`/some `ALTER`; `SELECT ... FOR UPDATE` takes real row locks. |
| "Deadlocks come from complex multi-table transactions." | You need ≥ 2 tables to deadlock. | Two single-row updates on one table, in opposite order, deadlock perfectly well. |
| "If I lock, I'm safe; the lock prevents the bug." | Locking = correctness. | Locking trades a *correctness* bug for a *liveness* bug: now you can deadlock, time out, or serialize your throughput to one core. |
| "Optimistic locking has no locks, so no contention." | Lock-free = contention-free. | Under high write contention optimistic locking becomes a retry storm that burns more CPU than a lock would. |

The single most important row is the second one. The reason my 2 a.m. payments deadlock was so confusing is that I believed two updates on *different accounts* could not possibly contend. They could, because InnoDB under `REPEATABLE READ` was taking next-key locks that covered *gaps* in the primary-key index, and a batch job inserting new accounts in the same key range was colliding with those gaps. The rest of this article is, in large part, an effort to make that sentence obvious.

## 1. Shared, exclusive, and the granularity ladder

**Senior rule of thumb: the lock mode tells you *what you may do*; the lock granularity tells you *how much you block*. Optimize the second one harder than the first.**

A lock has two independent axes. The **mode** is S or X (we will add more modes shortly). The **granularity** is *which object* the lock is on: a single row, a range of rows, a page, an index, an entire table, the whole database. The two combine: you can have a shared lock on a row, an exclusive lock on a row, a shared lock on a table, an exclusive lock on a table, and so on. The mode determines compatibility; the granularity determines *blast radius*. An X lock on one row blocks writers to that row. An X lock on the whole table blocks every reader and writer of every row.

Engines default to **row-level locking** for data modification precisely because the blast radius is small: two transactions updating different rows of a million-row table do not block each other at all. This is why modern databases scale write concurrency far past the old table-locking engines (the original MyISAM table lock turned every write into a stop-the-world event). But row-level locking is not free. The engine must *track* every individual row lock, which costs memory and bookkeeping, and — crucially — it must be able to answer table-level questions ("can I `ALTER` this table?") without scanning a billion rows to see whether any of them is locked. That second problem is what intention locks solve, and we will get there in section 4.

The granularity ladder, coarsest to finest:

| Granularity | What it protects | Typical use | Blast radius |
| --- | --- | --- | --- |
| Database | the whole DB | `pg_dump` snapshot consistency, some DDL | catastrophic |
| Table | every row + structure | `LOCK TABLE`, `DROP`, `TRUNCATE`, `ALTER` | whole table stalls |
| Page / extent | one disk page of rows | legacy SQL Server escalation; rare in PG/InnoDB | a page of rows |
| Row / record | one index record | `UPDATE`, `DELETE`, `SELECT FOR UPDATE` | one row |
| Gap / range | a range *between* records | InnoDB phantom prevention | rows that don't exist yet |
| Advisory | an application-chosen integer | "only one cron at a time" | whatever you decide |

The most common production mistake is reaching for a coarser granularity than you need because it is *easier to reason about*. `LOCK TABLE accounts IN EXCLUSIVE MODE` does make your logic trivially correct — nobody else can touch `accounts` while you hold it — but you have just converted a row-level concurrency problem into a table-level serialization point, and your throughput is now one transaction at a time. The discipline of production locking is to use the *finest* granularity that still gives you correctness, and then to deal with the contention that finer locks expose, rather than papering over it with a sledgehammer.

Here is the absolute foundation in runnable form. Open two `psql` sessions side by side. (These also work in `mysql` with minor syntax changes noted later.)

```sql
-- Setup (run once, in either session)
CREATE TABLE accounts (
    id      bigint PRIMARY KEY,
    owner   text NOT NULL,
    balance numeric NOT NULL DEFAULT 0
);
INSERT INTO accounts (id, owner, balance) VALUES
    (1, 'alice', 100),
    (2, 'bob',   100);
```

```sql
-- Session A
BEGIN;
UPDATE accounts SET balance = balance - 10 WHERE id = 1;
-- A now holds an EXCLUSIVE row lock on row id=1. It does NOT commit yet.
```

```sql
-- Session B (run while A's transaction is still open)
BEGIN;
UPDATE accounts SET balance = balance + 10 WHERE id = 1;
-- B BLOCKS here. It is waiting for A to release its X-lock on row id=1.
-- The shell just hangs. Nothing is wrong; this is a lock wait.
```

```sql
-- Back in Session A
COMMIT;
-- The instant A commits, B's UPDATE unblocks and proceeds.
```

That hang in session B is the most important phenomenon in this article. It is not an error. It is the lock-compatibility table doing its job: B asked for an X lock on a row A already held X on, the cell is red, and B waits. Ninety-nine percent of the time, this wait is so short you never see it. The other one percent is what we spend the rest of the article on.

### Second-order: locks are held until *transaction* end, not statement end

The non-obvious gotcha that trips up every engineer once: a row lock acquired by an `UPDATE` is **not** released when the `UPDATE` statement finishes. It is held until the surrounding transaction `COMMIT`s or `ROLLBACK`s. This is what makes lock duration a *transaction-design* problem, not a query problem. If your transaction does an `UPDATE` and then makes an HTTP call to a payment provider before committing, you are holding that row lock for the entire duration of the network round-trip — hundreds of milliseconds during which nobody else can update that row. The single highest-leverage anti-deadlock and anti-contention technique in existence is therefore embarrassingly simple: **keep transactions short, and never do I/O while holding a lock.** We will return to this with numbers, but internalize it now: every line of code between your first lock acquisition and your `COMMIT` is a line of code that someone else might be blocked behind.

## 2. Postgres lock modes: eight table levels, four row levels

**Senior rule of thumb: in Postgres you rarely take a lock by name; you take it by *running a statement*. Know which statement implies which mode, and the conflict matrix tells you what you'll block.**

Postgres has two largely separate locking systems: **table-level locks** (eight modes) and **row-level locks** (four modes), plus advisory locks (section 9). You almost never write `LOCK TABLE ... IN ... MODE` explicitly; instead, ordinary statements acquire table locks implicitly, and the conflict rules decide whether two concurrent statements can run. The eight table-level modes, from weakest to strongest:

| Lock mode | Acquired by | Conflicts with |
| --- | --- | --- |
| `ACCESS SHARE` | `SELECT` (read-only) | only `ACCESS EXCLUSIVE` |
| `ROW SHARE` | `SELECT FOR UPDATE/SHARE/...` | `EXCLUSIVE`, `ACCESS EXCLUSIVE` |
| `ROW EXCLUSIVE` | `INSERT`, `UPDATE`, `DELETE`, `MERGE` | `SHARE` and stronger |
| `SHARE UPDATE EXCLUSIVE` | `VACUUM` (no FULL), `ANALYZE`, `CREATE INDEX CONCURRENTLY` | self and stronger |
| `SHARE` | `CREATE INDEX` (no CONCURRENTLY) | `ROW EXCLUSIVE` and most stronger |
| `SHARE ROW EXCLUSIVE` | `CREATE TRIGGER`, some `ALTER TABLE` | self, `SHARE`, and stronger |
| `EXCLUSIVE` | `REFRESH MATERIALIZED VIEW CONCURRENTLY` | everything except `ACCESS SHARE` |
| `ACCESS EXCLUSIVE` | `DROP`, `TRUNCATE`, `REINDEX`, `CLUSTER`, `VACUUM FULL`, most `ALTER TABLE`, `LOCK TABLE` default | *everything*, including plain `SELECT` |

The two rows that bite people are the first and the last. A plain `SELECT` takes only `ACCESS SHARE`, which conflicts with exactly one thing: `ACCESS EXCLUSIVE`. So reads never block reads, never block writes, and never block most schema changes — *except* the schema changes that take `ACCESS EXCLUSIVE`, like a naive `ALTER TABLE ... ADD COLUMN ... DEFAULT ...` on an old Postgres, or `DROP`/`TRUNCATE`. This is the entire reason zero-downtime migrations are hard: an `ALTER TABLE` that needs `ACCESS EXCLUSIVE` must wait for every open transaction holding even an `ACCESS SHARE` (i.e., every in-flight `SELECT`) to finish, and meanwhile it queues *ahead of* all new statements, so a single long-running read can stall your migration, which then stalls every subsequent query on the table. The fix is `SET lock_timeout = '2s'` before the `ALTER` so it backs off instead of blocking the world, plus splitting destructive DDL into concurrent-safe steps.

### Row-level modes: FOR UPDATE vs FOR NO KEY UPDATE vs FOR SHARE vs FOR KEY SHARE

For row locks, Postgres has four modes, deliberately ordered from strongest to weakest so that ordinary foreign-key checks don't block ordinary updates:

| Requested ↓ / Held → | `FOR KEY SHARE` | `FOR SHARE` | `FOR NO KEY UPDATE` | `FOR UPDATE` |
| --- | --- | --- | --- | --- |
| `FOR KEY SHARE` | ✓ | ✓ | ✓ | ✗ |
| `FOR SHARE` | ✓ | ✓ | ✗ | ✗ |
| `FOR NO KEY UPDATE` | ✓ | ✗ | ✗ | ✗ |
| `FOR UPDATE` | ✗ | ✗ | ✗ | ✗ |

Read it as: `FOR UPDATE` (the strongest) blocks everything; `FOR KEY SHARE` (the weakest) blocks only `FOR UPDATE`. The reason this four-way split exists is subtle and worth understanding because it eliminated a whole class of foreign-key deadlocks. When you `UPDATE` a row but do not change any column that a foreign key or unique index references, Postgres takes `FOR NO KEY UPDATE` instead of full `FOR UPDATE`. Simultaneously, when Postgres validates a foreign key — "does the parent row I'm pointing at still exist?" — it takes only `FOR KEY SHARE` on the parent. Because `FOR NO KEY UPDATE` and `FOR KEY SHARE` are *compatible* (the top-right of that matrix is a ✓), a child-row insert that needs to check the parent does not block an unrelated non-key update of that same parent. Before this split (Postgres < 9.3), every parent update and every child insert fought over the same lock and deadlocked constantly under load.

```sql
-- Explicit row locking in Postgres.
BEGIN;
-- Strongest: I intend to UPDATE or DELETE this row.
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;

-- Weaker: I will UPDATE a non-key column, so don't block FK checks.
SELECT * FROM accounts WHERE id = 1 FOR NO KEY UPDATE;

-- Shared: I'm reading and must prevent concurrent modification.
SELECT * FROM accounts WHERE id = 1 FOR SHARE;

-- Weakest: I just need the key to stay valid (FK check).
SELECT * FROM accounts WHERE id = 1 FOR KEY SHARE;
COMMIT;
```

### Second-order: SKIP LOCKED and NOWAIT change the *waiting* behavior, not the lock

`FOR UPDATE` has two modifiers that are among the most useful tools in the entire locking toolbox, and they are about what happens *when the lock is unavailable*:

```sql
-- Wait (default): block until the row is free.
SELECT * FROM jobs WHERE state='ready' ORDER BY id LIMIT 1 FOR UPDATE;

-- NOWAIT: if the row is locked, fail immediately with an error
-- (SQLSTATE 55P03), don't wait. Good for "try, else give up fast".
SELECT * FROM jobs WHERE id = 42 FOR UPDATE NOWAIT;

-- SKIP LOCKED: silently skip rows another transaction has locked,
-- returning only currently-unlocked rows. The job-queue superpower.
SELECT * FROM jobs WHERE state='ready' ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED;
```

`NOWAIT` turns a potential indefinite wait into an immediate, catchable error — ideal when "I couldn't get the lock right now" is a normal, recoverable outcome rather than something to wait out. `SKIP LOCKED` is even more powerful: it makes a `SELECT` ignore rows that are currently locked, as if they weren't there. We will build an entire lock-free job queue on it in section 7.

## 3. InnoDB locks: record, gap, next-key, and insert-intention

**Senior rule of thumb: in InnoDB, a lock is on an *index record*, not on a row. Once you internalize that, gap locks and phantom prevention stop being mysterious.**

MySQL's InnoDB engine has the same S/X foundation, the same intention locks (next section), but its row-level locking is richer and weirder than Postgres's, and it is the source of more production deadlocks than any other single mechanism I know of. The MySQL manual ([InnoDB Locking](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html)) is the canonical reference; here is the working engineer's version.

There are three flavors of row-level lock in InnoDB, and the trick is that they are all locks *on index records*:

- **Record lock**: a lock on a single index record. `SELECT c1 FROM t WHERE c1 = 10 FOR UPDATE` takes a record lock on the index record for `c1=10`, preventing anyone else from updating or deleting that row. In the lock log it appears as `lock_mode X locks rec but not gap`.
- **Gap lock**: a lock on the *gap between* index records — or before the first record, or after the last. `SELECT c1 FROM t WHERE c1 BETWEEN 10 AND 20 FOR UPDATE` locks the gaps so that nobody can `INSERT` a `c1=15` into the range, *even though no such row exists yet*. Gap locks are "purely inhibitive": their only job is to prevent inserts. Critically, two transactions can hold *conflicting* (S and X) gap locks on the *same* gap simultaneously — a gap lock does not block another gap lock. This surprising rule is exactly what makes some insert deadlocks possible.
- **Next-key lock**: the combination of a record lock on an index record *and* a gap lock on the gap *before* it. This is InnoDB's default locking unit under `REPEATABLE READ`, and it is how InnoDB prevents phantom reads without taking a table lock.

![A next-key lock is a record lock plus the gap before it, so consecutive next-key locks fence the whole index range against phantom inserts](/imgs/blogs/database-locks-and-deadlocks-in-production-4.webp)

The figure above is worth staring at. Suppose an index on `age` contains the values 10, 11, 13, 20. InnoDB conceptually divides the index into intervals: `(-∞, 10]`, `(10, 11]`, `(11, 13]`, `(13, 20]`, `(20, +∞)`. Each interval, *including the gap before the record and the record itself*, is a next-key lock unit. When a `REPEATABLE READ` transaction runs `SELECT * FROM t WHERE age BETWEEN 11 AND 13 FOR UPDATE`, InnoDB locks the next-key intervals covering that range — which includes the gap `(11, 13)`. Now if another transaction tries `INSERT INTO t (age) VALUES (12)`, that 12 lands inside the locked gap `(11, 13)`, and the insert *blocks*. That is phantom prevention: the first transaction can re-run its range scan and be guaranteed to see the same rows, because no phantom row can be inserted into the locked range. Without next-key locks, the second transaction could slip a 12 in, and the first transaction's repeated read would suddenly see a new row — a phantom.

This is also why isolation level changes everything in InnoDB. Under `READ COMMITTED`, InnoDB *disables gap locking* almost entirely — it takes only record locks, not next-key locks. The `INSERT age=12` in the example above succeeds freely, because the gap is no longer locked. You give up phantom protection and get dramatically fewer gap-lock deadlocks. This is the single most effective InnoDB deadlock mitigation that most teams never try: a large fraction of InnoDB deadlocks in the wild are gap-lock deadlocks that *simply do not occur* under `READ COMMITTED`. (You also need `binlog_format=ROW` for this to be replication-safe, which is the modern default.)

### Insert intention locks: the deadlock you didn't see coming

The fourth InnoDB lock type is the **insert intention lock**, a special gap lock that `INSERT` sets *before* inserting a row. The design intent is concurrency: two transactions inserting different values into the same gap — say 5 and 6 into the gap `(4, 7)` — should not block each other, because they aren't inserting the *same* value. The insert intention lock signals "I intend to insert at this position in the gap" without blocking other inserts at *different* positions. In the lock log it reads `lock_mode X locks gap before rec insert intention waiting`.

The deadlock comes from the interaction between gap locks and insert intention locks, and it is exactly the pattern that produced the famous Shopify postmortem we'll dissect in the case studies. Two transactions both acquire shared next-key/gap locks over the same range (because gap locks coexist), and then both try to `INSERT` into that range. Each `INSERT`'s insert intention lock must wait for the *other* transaction's gap lock to be released — but neither will release until it commits, and neither can commit until its insert proceeds. Cycle. Deadlock. This is the mechanism behind the maddening "two inserts into the same table, on different keys, deadlocked each other" reports that fill MySQL forums. The canonical write-up of this exact failure is [Understand deadlock by gap locking in InnoDB](https://tanishiking.hashnode.dev/avoid-deadlock-caused-by-a-conflict-of-transactions-that-accidentally-acquire-gap-lock-in-innodb-a114e975fd72), and the MySQL manual's [Locks Set by Different SQL Statements](https://dev.mysql.com/doc/refman/8.0/en/innodb-locks-set.html) is the precise spec for which statement takes which lock.

### Second-order: the lock follows the index used, not the column in your WHERE

A non-obvious InnoDB gotcha that causes "why is this unrelated row locked?" confusion: InnoDB locks the *index records it scans*, not the rows that match your `WHERE`. If your `DELETE FROM t WHERE non_indexed_col = 5` cannot use an index, InnoDB scans the entire clustered index and takes next-key locks on *every row it examines* — effectively locking the whole table for the duration. Add the right index and the same `DELETE` locks only the matching records. A huge fraction of "InnoDB locked way more than I expected" incidents are missing-index incidents in disguise. Run `EXPLAIN` on your locking statements; if it says `type: ALL`, you are about to lock far more than you think.

## 4. Intention locks and the lock hierarchy

**Senior rule of thumb: intention locks exist so the engine can answer "can I lock this whole table?" in O(1) instead of scanning a billion rows.**

![Intention locks at the table level let the engine detect conflicts with a row write without scanning every row in the table](/imgs/blogs/database-locks-and-deadlocks-in-production-6.webp)

Here is the problem intention locks solve. Suppose transaction A is updating one row deep inside a billion-row table, so A holds an X lock on that one row. Now transaction B wants to `LOCK TABLE ... IN EXCLUSIVE MODE` — it wants an X lock on the *whole table*. B's table-level X lock must conflict with A's row-level X lock (otherwise B could lock the table while A is mid-write, which is the whole point of a table lock). But how does B *find out* that some row is locked, without iterating over all billion rows checking each one for a lock? Scanning is O(n) and would make every table lock catastrophically slow.

The answer is the **intention lock**, a table-level marker that announces "I hold, or am about to hold, row-level locks inside this table." Two flavors:

- **Intention Shared (IS)**: "I intend to take S locks on some rows." Acquired implicitly before `SELECT ... FOR SHARE`.
- **Intention Exclusive (IX)**: "I intend to take X locks on some rows." Acquired implicitly before `SELECT ... FOR UPDATE`, `UPDATE`, `DELETE`.

The protocol is a hierarchy: before a transaction can take an S lock on a *row*, it must first take at least an IS lock on the *table*; before an X lock on a row, at least an IX lock on the table. Now B's question is trivial. B wants a table-level X lock; it checks the table's intention locks; A is holding an IX (because A took a row X lock); IX conflicts with table-level X; B waits. No row scan. The intention lock is the index that makes hierarchical locking O(1).

The table-level compatibility matrix — the one in Figure 1, but now with the names attached — is exactly this:

| Held ↓ / Requested → | IS | IX | S | X |
| --- | --- | --- | --- | --- |
| **IS** | ✓ | ✓ | ✓ | ✗ |
| **IX** | ✓ | ✓ | ✗ | ✗ |
| **S** | ✓ | ✗ | ✓ | ✗ |
| **X** | ✗ | ✗ | ✗ | ✗ |

The key cells: **IX is compatible with IX**. Two transactions can both hold IX on the table simultaneously, because "I intend to lock *some* rows" and "I also intend to lock *some* rows" don't conflict at the table level — the actual conflict, if any, is resolved at the row level when they happen to want the same row. This is exactly what lets a million concurrent writers hammer different rows of one table without table-level contention: they all hold IX, IX coexists with IX, and only the row locks decide who waits. The intention locks are pure bookkeeping; they almost never cause a wait themselves. They exist so that the *rare* table-level operation (an `ALTER`, a `LOCK TABLE`, a `DROP`) can conflict correctly without paying an O(n) cost.

In an `SHOW ENGINE INNODB STATUS` dump, intention locks show up as table-level lines:

```TABLE LOCK table `shop`.`orders` trx id 10080 lock mode IX
RECORD LOCKS space id 58 page no 3 n bits 72 index PRIMARY of table `shop`.`orders`
trx id 10080 lock_mode X locks rec but not gap
```

The first line is the intention lock (table-level IX); the second is the actual record lock that the IX was announcing. Postgres expresses the same idea differently — its `pg_locks` view shows `RowExclusiveLock` on the *relation* (table) for any transaction doing DML, which plays the role of InnoDB's IX.

## 5. How a deadlock actually forms

**Senior rule of thumb: a deadlock is never about one transaction. It is a *cycle* in the wait-for graph, and it takes at least two transactions acquiring the same locks in opposite order.**

![A deadlock is a cycle in the wait-for graph: T1 holds row A and waits for row B while T2 holds row B and waits for row A](/imgs/blogs/database-locks-and-deadlocks-in-production-2.webp)

Now we can build the deadlock from first principles. The **wait-for graph** is a directed graph where each node is a transaction and an edge T1 → T2 means "T1 is waiting for a lock that T2 holds." A deadlock is exactly a *cycle* in this graph. The simplest cycle has two transactions, shown above: T1 holds an X lock on row A and is waiting for row B; T2 holds an X lock on row B and is waiting for row A. T1 → T2 (T1 waits for what T2 holds) and T2 → T1 (T2 waits for what T1 holds). Neither can proceed, because each is waiting for a lock the other will release only after it proceeds. There is no amount of waiting that resolves this; it is logically unbreakable without intervention.

The recipe for a two-transaction deadlock is always the same: **two transactions acquire the same two locks in opposite order.** T1 locks A then B; T2 locks B then A. If their executions interleave so that T1 gets A and T2 gets B before either asks for the second, the cycle closes. This is why my payments deadlock happened: two transfer transactions, one moving money from account 1 to account 2 and the other from account 2 to account 1, locked the rows in opposite order. The fix, as we'll see, is to *always lock rows in a consistent order* (e.g., ascending by id), which makes the cycle impossible to form.

![A two-transaction deadlock timeline: each step succeeds in isolation, but the interleaving at t4 closes a cycle that the engine breaks by rolling back one transaction](/imgs/blogs/database-locks-and-deadlocks-in-production-3.webp)

The timeline above makes the interleaving concrete and is the deadlock you can reproduce on your own machine in two minutes. The key insight is that *every individual step succeeds*. At t1, T1's update of row 1 succeeds. At t2, T2's update of row 2 succeeds. At t3, T1 asks for row 2 and *blocks* (T2 holds it). At t4, T2 asks for row 1 and *blocks* (T1 holds it) — and now the cycle is closed. The engine notices, and at t5/t6 it rolls one transaction back. Here is the exact reproduction:

```sql
-- Setup: a table with two rows, id 1 and id 2 (use the accounts table above).

-- Session A
BEGIN;
UPDATE accounts SET balance = balance - 10 WHERE id = 1;  -- t1: locks row 1, OK

-- Session B
BEGIN;
UPDATE accounts SET balance = balance - 10 WHERE id = 2;  -- t2: locks row 2, OK

-- Session A
UPDATE accounts SET balance = balance + 10 WHERE id = 2;  -- t3: wants row 2, BLOCKS on B

-- Session B
UPDATE accounts SET balance = balance + 10 WHERE id = 1;  -- t4: wants row 1 -> DEADLOCK
-- One session immediately gets:
--   Postgres: ERROR:  deadlock detected
--   MySQL:    ERROR 1213 (40001): Deadlock found when trying to get lock; try restarting transaction
```

Run that and one of your two shells will instantly error out with a deadlock. Notice the speed: in MySQL the victim is killed almost instantly (InnoDB detects cycles eagerly); in Postgres the victim is killed after `deadlock_timeout` (default 1 second) elapses, because Postgres only *checks* for a cycle when a transaction has been waiting that long. That timing difference is itself a tuning knob, which we'll cover next.

> A deadlock is not a bug in the database. It is the database *correctly* refusing to wait forever. The bug is in your lock ordering, and the database is telling you about it the only way it can — by killing one of the transactions.

## 6. Detection, victim selection, and reading the logs

**Senior rule of thumb: you cannot prevent every deadlock, so the deadlock-retry loop is not optional — it is part of correct application code.**

![Without detection both transactions wait forever; the detector finds the cycle and rolls back the transaction with the least undo work so the survivor can commit](/imgs/blogs/database-locks-and-deadlocks-in-production-7.webp)

Two strategies exist for handling deadlocks: prevention (impose an order so cycles can't form — what two-phase locking with ordered acquisition gives you) and detection (let cycles form, then break them). Real databases use **detection** for general workloads because prevention requires knowing the full lock set in advance, which application code never does.

**InnoDB** maintains the wait-for graph incrementally and checks for a cycle *every time a lock request would block*. Detection is therefore near-instant — InnoDB rarely lets a deadlock persist for more than the moment it forms. When it finds a cycle, it picks as victim the transaction that has done the *least work* (fewest rows modified, smallest undo log), on the theory that rolling it back is cheapest, and aborts it with `ERROR 1213`. (For very large transaction counts InnoDB can fall back to `innodb_deadlock_detect=OFF` plus a `innodb_lock_wait_timeout`, trading prompt detection for lower CPU overhead under extreme concurrency.)

**Postgres** does it lazily. It does *not* maintain the wait-for graph continuously. Instead, when a transaction blocks on a lock, it waits for `deadlock_timeout` (default `1s`); only then does Postgres build the wait-for graph and check for a cycle. If it finds one, it rolls back one transaction — typically the one whose lock request *closed* the cycle (the last waiter) — with `ERROR: deadlock detected` (SQLSTATE `40P01`). The reason for the delay is efficiency: most lock waits are short and resolve on their own, so it would be wasteful to run cycle-detection on every brief wait. The cost is that a real deadlock in Postgres lingers for up to `deadlock_timeout` before resolution. Lowering it to, say, `200ms` makes deadlocks resolve faster but runs the detector more often; the default 1s is a reasonable balance for most systems.

The application consequence is the same for both engines: **any transaction that takes row locks can be chosen as a deadlock victim, and your code must retry it.** This is non-negotiable. Here is the canonical retry wrapper:

```python
import os
import time
import psycopg
from psycopg.errors import DeadlockDetected, SerializationFailure

def run_with_retry(conn_factory, work, max_attempts=5, base_delay=0.02):
    """Run `work(cur)` inside a transaction, retrying on deadlock / serialization
    failure with exponential backoff + jitter. After max_attempts, re-raise."""
    for attempt in range(1, max_attempts + 1):
        try:
            with conn_factory() as conn:
                with conn.cursor() as cur:
                    result = work(cur)
                conn.commit()
                return result
        except (DeadlockDetected, SerializationFailure) as exc:
            if attempt == max_attempts:
                raise
            # Exponential backoff with jitter avoids two retriers
            # immediately re-deadlocking in lockstep.
            delay = base_delay * (2 ** (attempt - 1))
            delay *= (0.5 + os.urandom(1)[0] / 255.0)  # 0.5x..1.5x jitter
            time.sleep(delay)
    raise RuntimeError("unreachable")
```

The MySQL equivalent catches the `1213` error code (`pymysql.err.OperationalError` with `e.args[0] == 1213`). The jitter matters: without it, two transactions that deadlocked once will retry at the same instant and deadlock again, and again, in a synchronized loop. The exponential backoff plus randomization breaks the lockstep.

### Reading the Postgres deadlock log

When `log_lock_waits = on` (turn this on in production), Postgres logs the gory details. A real deadlock log looks like:

```ERROR:  deadlock detected
DETAIL:  Process 70725 waits for ShareLock on transaction 891718;
         blocked by process 70713.
         Process 70713 waits for ShareLock on transaction 891717;
         blocked by process 70725.
HINT:    See server log for query details.
CONTEXT: while updating tuple (0,7) in relation "accounts"
STATEMENT:  UPDATE accounts SET balance = balance + 10 WHERE id = 1
```

Read it as a wait-for graph dumped to text: process 70725 waits for the transaction held by process 70713, and process 70713 waits for the transaction held by process 70725 — a two-node cycle. The `CONTEXT` line tells you *which tuple* (`(0,7)` is `(block, offset)`) and `STATEMENT` gives you the query that closed the cycle. To find the *other* statement (the one already holding the lock), you cross-reference the server log around the same timestamp for the other PID's statements, which is why you also want `log_statement` or at least `log_min_duration_statement` set. The combination of `log_lock_waits`, the deadlock `DETAIL`, and statement logging is enough to reconstruct exactly which two queries acquired which two locks in which order.

### Reading the InnoDB deadlock log

InnoDB keeps the *most recent* deadlock in `SHOW ENGINE INNODB STATUS` under the `LATEST DETECTED DEADLOCK` section (and writes it to the error log if `innodb_print_all_deadlocks=ON`):

```------------------------
LATEST DETECTED DEADLOCK
------------------------
*** (1) TRANSACTION:
TRANSACTION 10078, ACTIVE 4 sec starting index read
mysql tables in use 1, locked 1
LOCK WAIT 2 lock struct(s), heap size 1136, 1 row lock(s)
UPDATE accounts SET balance = balance + 10 WHERE id = 2
*** (1) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 58 page no 4 n bits 72 index PRIMARY of table `shop`.`accounts`
trx id 10078 lock_mode X locks rec but not gap waiting

*** (2) TRANSACTION:
TRANSACTION 10079, ACTIVE 6 sec starting index read
UPDATE accounts SET balance = balance + 10 WHERE id = 1
*** (2) HOLDS THE LOCK(S):
RECORD LOCKS space id 58 page no 4 n bits 72 index PRIMARY of table `shop`.`accounts`
trx id 10079 lock_mode X locks rec but not gap
*** (2) WAITING FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 58 page no 3 n bits 72 index PRIMARY of table `shop`.`accounts`
trx id 10079 lock_mode X locks rec but not gap waiting
*** WE ROLL BACK TRANSACTION (1)
```

This is the single most useful diagnostic artifact in MySQL, and learning to read it pays for itself a hundred times over. Transaction (1) is waiting for an X record lock on a row (the `WAITING FOR` block). Transaction (2) *holds* that lock (its `HOLDS THE LOCK(S)` block) and is *itself* waiting for another lock that (1) holds — there's your cycle. The decisive lines:

- `lock_mode X locks rec but not gap` → a pure record lock (no gap). If you instead see `lock_mode X` with no "rec but not gap" qualifier, it's a next-key lock (record + gap). If you see `locks gap before rec`, it's a pure gap lock. `insert intention waiting` means an `INSERT` is blocked on a gap lock.
- `index PRIMARY` tells you it's the clustered index; `index idx_foo` would tell you a secondary index is involved (often the surprising part of a deadlock).
- `WE ROLL BACK TRANSACTION (1)` is the victim — transaction (1) got the `ERROR 1213`.

If you only learn one diagnostic skill from this article, make it reading these two log formats. They turn deadlock debugging from guesswork into a five-minute exercise.

## 7. Lock-wait monitoring with pg_locks and innodb_lock_waits

**Senior rule of thumb: a deadlock is loud (it errors); a *lock wait* is silent (it just hangs). The silent one is what's actually killing your p99.**

Most lock pain in production is not deadlocks — it's lock *waits*: transactions blocked behind a long-held lock, piling up, blowing your latency budget without a single error in the logs. You find these by querying the live lock state. In Postgres, the killer function is `pg_blocking_pids()`:

```sql
-- Who is blocking whom, right now? The one-query lock-wait dashboard.
SELECT
    blocked.pid              AS blocked_pid,
    blocked.query           AS blocked_query,
    blocking.pid            AS blocking_pid,
    blocking.query          AS blocking_query,
    now() - blocked.query_start AS blocked_for
FROM pg_stat_activity AS blocked
JOIN LATERAL unnest(pg_blocking_pids(blocked.pid)) AS bpid ON true
JOIN pg_stat_activity AS blocking ON blocking.pid = bpid
WHERE blocked.wait_event_type = 'Lock'
ORDER BY blocked_for DESC;
```

`pg_blocking_pids(pid)` returns the array of PIDs that are blocking the given PID — it does the wait-for graph traversal for you, and it is far more reliable than trying to self-join `pg_locks` by hand (which is fiddly because lock-grant semantics are subtle). For the raw lock inventory, `pg_locks` joined to `pg_stat_activity` shows every lock held and awaited:

```sql
SELECT l.locktype, l.mode, l.granted,
       a.pid, a.state, left(a.query, 60) AS query
FROM pg_locks l
JOIN pg_stat_activity a ON a.pid = l.pid
WHERE l.relation = 'accounts'::regclass
ORDER BY l.granted, a.query_start;
```

`granted = false` rows are transactions *waiting* for a lock; `granted = true` rows are holders. The pattern you're hunting for: one long-running holder (often `state = 'idle in transaction'` — a transaction someone forgot to commit) with a queue of waiters behind it. The fix for `idle in transaction` is `SET idle_in_transaction_session_timeout = '30s'`, which automatically kills transactions that open, lock, and then sit there (a bug-ridden application or a developer's forgotten `psql` session).

In MySQL 8.0, the equivalent lives in `performance_schema` (the older `information_schema.innodb_lock_waits` was deprecated):

```sql
-- Who is blocking whom in InnoDB (MySQL 8.0+)
SELECT
    r.trx_id              AS waiting_trx,
    r.trx_mysql_thread_id AS waiting_thread,
    r.trx_query           AS waiting_query,
    b.trx_id              AS blocking_trx,
    b.trx_mysql_thread_id AS blocking_thread,
    b.trx_query           AS blocking_query
FROM performance_schema.data_lock_waits w
JOIN information_schema.innodb_trx r ON r.trx_id = w.requesting_engine_transaction_id
JOIN information_schema.innodb_trx b ON b.trx_id = w.blocking_engine_transaction_id;
```

And `performance_schema.data_locks` shows the full lock inventory, with `LOCK_TYPE` (RECORD/TABLE), `LOCK_MODE` (X/S, REC_NOT_GAP, GAP, etc.), and `LOCK_STATUS` (GRANTED/WAITING). Build a Grafana panel on top of either of these — count of waiters, max wait age, top blocking query — and you will catch contention before it pages you.

### The job-queue pattern: SELECT FOR UPDATE SKIP LOCKED

The single most valuable *positive* use of locks (as opposed to "things to avoid") is the lock-free job queue built on `FOR UPDATE SKIP LOCKED`. This pattern is the engine room of [Que](https://github.com/que-rb/que) and [GoodJob](https://github.com/bensheldon/good_job) (Ruby), [River](https://github.com/riverqueue/river) (Go), Oban (Elixir), and dozens of in-house queue systems, and it lets you use your existing relational database as a high-throughput, exactly-once, transactionally-consistent task queue without a separate broker.

![SELECT FOR UPDATE SKIP LOCKED turns a queue table into a lock-free dispatcher where each worker claims the first un-locked pending row and skips the ones peers hold](/imgs/blogs/database-locks-and-deadlocks-in-production-5.webp)

The figure shows why it works. Each worker runs the same query: "give me the oldest pending job, lock it, and skip any job another worker has already locked." Because of `SKIP LOCKED`, worker 2 doesn't *wait* for worker 1's locked row — it treats that row as invisible and grabs the next one. N workers therefore drain the queue in parallel with essentially zero contention, and the row lock guarantees no two workers ever process the same job. Here is the production-grade version:

```sql
-- Claim and lock one job atomically. The CTE locks the row with SKIP LOCKED,
-- then the outer UPDATE marks it running and returns the payload. All in one
-- statement, so the claim and the state transition are atomic.
WITH claimed AS (
    SELECT id
    FROM jobs
    WHERE state = 'ready'
      AND run_at <= now()
    ORDER BY priority DESC, run_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE jobs j
SET state = 'running',
    started_at = now(),
    attempts = attempts + 1
FROM claimed
WHERE j.id = claimed.id
RETURNING j.id, j.payload;
```

Why this beats the naive `SELECT ... WHERE state='ready' LIMIT 1` then `UPDATE`: without `SKIP LOCKED`, every worker selects the *same* oldest row, all but one block on the `FOR UPDATE`, and you've serialized your entire worker pool behind one row — a thundering herd on a single lock. With `SKIP LOCKED`, contention vanishes. There are caveats, well documented in [Potential Consequences of Using Postgres as a Job Queue](https://richyen.com/postgres/2026/05/04/postgres_job_queue.html): the constant churn of `state` updates generates dead tuples that `VACUUM` must clean up, so a busy queue table needs aggressive autovacuum tuning, and very high job rates eventually justify a purpose-built broker. But for the overwhelming majority of systems — anything under a few thousand jobs/second — `SKIP LOCKED` on your existing Postgres is simpler, transactional, and one fewer piece of infrastructure to operate. The pattern works identically in MySQL 8.0+, which added `SKIP LOCKED` and `NOWAIT`.

## 8. Hot-row contention and how to defeat it

**Senior rule of thumb: contention is not about *how many* writes you have; it's about how many writes want the *same row* at the same time. Spread the writes and the contention evaporates.**

The flip side of deadlocks is plain old contention: a single row that everyone wants to write. The textbook example is a counter — a `likes` count, a global sequence, an inventory quantity, a daily-total row. Every increment is `UPDATE counters SET value = value + 1 WHERE id = X`, every increment takes an X lock on row X, and X locks don't coexist, so your increments *serialize*. Your maximum throughput on that counter is `1 / (lock hold time)` — if each transaction holds the lock for 1 ms, you cap at ~1000 increments/second *no matter how many cores or connections you throw at it*. This is a per-row Amdahl's law, and it is the cause of a huge fraction of "we scaled the hardware but throughput didn't move" mysteries.

![A single hot counter row serializes every writer; splitting it into N sharded rows and summing on read multiplies write concurrency N-fold](/imgs/blogs/database-locks-and-deadlocks-in-production-8.webp)

The cure, shown above, is **counter sharding**: instead of one row holding the count, keep N rows (say 16), each holding a partial count. Every increment picks a *random* shard and updates that one. The true value is `SELECT SUM(value) FROM counters WHERE counter_id = X`. Now you have N independent X locks instead of one, so up to N writers proceed in parallel, and your throughput multiplies by ~N. This is exactly what Shopify's [multirow_counter](https://github.com/Shopify/multirow_counter) library does: it creates multiple rows per logical counter and "sends each UPDATE to a random one of the related rows, lowering the chance of lock contention." Reads pay a small `SUM` cost; writes get N-fold concurrency. The tradeoff is paid in read complexity and a tiny window where the read isn't perfectly atomic across shards — acceptable for almost any counter.

```sql
-- Sharded counter. 16 shards per logical counter.
CREATE TABLE counter_shards (
    counter_id text   NOT NULL,
    shard      int    NOT NULL,
    value      bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (counter_id, shard)
);

-- Increment: pick a random shard, so writers spread across 16 rows.
UPDATE counter_shards
SET value = value + 1
WHERE counter_id = 'page:42:views'
  AND shard = floor(random() * 16)::int;

-- Read the true total.
SELECT COALESCE(SUM(value), 0) AS total
FROM counter_shards
WHERE counter_id = 'page:42:views';
```

Counter sharding is one of three tools for hot-row contention; here is the full toolkit:

| Technique | How it works | When to use | Cost |
| --- | --- | --- | --- |
| **Shard the row** | N rows, write a random one, `SUM` on read | high-frequency counters, totals | read does a `SUM`; non-atomic across shards |
| **Shorten the transaction** | release the lock sooner: commit before the slow part | any lock held across I/O or computation | requires restructuring transaction boundaries |
| **Reorder access (consistent ordering)** | always lock rows in a fixed order (e.g. by id) | multi-row transactions that deadlock | requires sorting the lock set up front |
| **Batch / coalesce** | accumulate increments in app memory, flush periodically | metrics, analytics counters | loses real-time accuracy; risk of loss on crash |
| **Move to an append-only model** | `INSERT` an event row instead of `UPDATE` a count; aggregate later | event/audit/metrics workloads | reads aggregate; storage grows |
| **Optimistic locking** | version column, retry on conflict (section 10) | low-contention rows | retry storm under high contention |

The append-only swap deserves emphasis because it's the cleanest. Instead of `UPDATE accounts SET balance = balance + 10`, which contends on the balance row, you `INSERT INTO ledger (account_id, delta) VALUES (1, 10)` — and inserts of *different* rows don't contend at all (modulo gap locks). The balance becomes `SELECT SUM(delta) FROM ledger WHERE account_id = 1`, optionally materialized into a periodically-recomputed snapshot. This is the double-entry ledger pattern, and it's why serious financial systems are append-only: it sidesteps hot-row contention *and* gives you a perfect audit trail. The tradeoff is reads get more expensive and you need a snapshotting strategy, but for a balance that's written far more than it's read, it's the right call. (For the storage-engine reasons append-only writes are also cheaper at the disk level, see the [LSM trees deep dive](/blog/software-development/database/lsm-trees-write-optimized-storage-engines).)

## 9. Advisory locks: when you need a mutex, not a row lock

**Senior rule of thumb: when you need to serialize *application logic* — "only one cron job at a time" — don't invent a lock table; use advisory locks.**

Sometimes you don't want to lock a *row*; you want to lock a *concept*. "Only one instance of this nightly job may run." "Only one process may rebuild this cache at a time." "This user's onboarding workflow must not run twice concurrently." The wrong way to do this is to create a `locks` table and `INSERT`/`DELETE` rows, which has its own contention and cleanup problems. The right way in Postgres is **advisory locks**: locks keyed by an application-chosen 64-bit integer (or two 32-bit ints), with no associated database object, that mean whatever your application decides they mean.

```sql
-- Transaction-level advisory lock: auto-released at COMMIT/ROLLBACK.
-- Blocks if another transaction holds the same key.
SELECT pg_advisory_xact_lock(hashtext('nightly-report-job'));
-- ... do the exclusive work ...
COMMIT;  -- lock released automatically

-- Try-variant: returns true if acquired, false immediately if not.
-- The canonical "is the cron already running?" check.
SELECT pg_try_advisory_xact_lock(hashtext('nightly-report-job'));
-- => false means another instance holds it; exit quietly.

-- Session-level: survives across transactions, must be released manually.
-- Survives ROLLBACK (does NOT honor transaction semantics).
SELECT pg_advisory_lock(42);
-- ... work across multiple transactions ...
SELECT pg_advisory_unlock(42);
```

There are two flavors, and choosing the wrong one is the classic advisory-lock bug. **Transaction-level** locks (`pg_advisory_xact_lock`) are released automatically when the transaction ends — you cannot leak them, and they're what you want 95% of the time. **Session-level** locks (`pg_advisory_lock`) persist until you explicitly unlock or the connection drops; they survive `ROLLBACK`, which makes them powerful but dangerous — a code path that acquires one and forgets to release it leaks the lock for the life of the connection (and connection pools make this worse, because the next user of that pooled connection inherits the leaked lock). Default to the transaction-level variant; reach for session-level only when you genuinely need a lock that outlives a single transaction, and pair every acquire with a guaranteed release.

The `pg_try_advisory_xact_lock` non-blocking variant is the idiomatic distributed-mutex primitive for "single-flight" patterns: every cron instance tries the lock, exactly one wins, the rest see `false` and exit. No race, no lock table, no external coordination service for what is fundamentally a single-database concern. (For genuinely *distributed* coordination across multiple databases you'd reach for something like a consensus system, but for "single Postgres, only one of these at a time," advisory locks are perfect.) MySQL has an analogous facility in `GET_LOCK('name', timeout)` / `RELEASE_LOCK('name')`, which are session-scoped named locks serving the same purpose.

## 10. Optimistic vs pessimistic locking

**Senior rule of thumb: pessimistic locking prevents conflicts by waiting; optimistic locking detects conflicts by retrying. Pick based on whether conflicts are rare or common — not on which sounds cleverer.**

Everything so far has been *pessimistic* locking: assume conflicts will happen, take a lock up front (`SELECT ... FOR UPDATE`), and hold it until commit. The alternative is *optimistic* locking: assume conflicts are rare, take no lock, read the data with a version number, and at write time check that the version hasn't changed. If it has, someone else modified the row first; you abort and retry.

```sql
-- Optimistic locking with a version column.
-- 1. Read the row and its current version.
SELECT id, quantity, version FROM inventory WHERE id = 99;
-- (suppose version = 7)

-- 2. Do your business logic in the application, no lock held.

-- 3. Conditional update: only succeeds if version is still 7.
UPDATE inventory
SET quantity = quantity - 1,
    version  = version + 1
WHERE id = 99 AND version = 7;
-- If this affects 0 rows, someone else updated first: retry from step 1.
```

The contrast is sharp, and the right choice depends entirely on contention:

| Dimension | Pessimistic (`FOR UPDATE`) | Optimistic (version column) |
| --- | --- | --- |
| Conflict model | prevent: lock before touching | detect: check at write time |
| Lock held during think time | yes (whole transaction) | no |
| Behavior under low contention | small overhead, lock rarely contended | excellent — near-zero overhead |
| Behavior under high contention | writers queue, serialized but make progress | retry storm — threads 2..N fail and retry, burning CPU |
| Long user think-time (UI edit) | terrible — lock held across human latency | ideal — no lock during the human's thinking |
| Risk | deadlocks, lock waits, blocked connections | wasted work on retries, possible starvation |
| Best for | inventory decrement, money transfer, hot rows | content editing, wikis, low-collision records |

The decisive insight, captured well in the production write-ups ([Optimistic vs Pessimistic Locking: What Nobody Tells You Until You've Burnt in Production](https://medium.com/@liberatoreanita/optimistic-vs-pessimistic-locking-what-nobody-tells-you-until-youve-burnt-in-production-c12f972ec90d) and [Vlad Mihalcea's analysis](https://vladmihalcea.com/optimistic-vs-pessimistic-locking/)): **optimistic locking is not "lock-free, therefore contention-free."** Under high contention it is *worse* than pessimistic. Imagine 100 threads incrementing the same inventory row. With pessimistic locking they queue and each makes progress in turn. With optimistic locking, all 100 read version 7, one wins the conditional update, the other 99 fail (0 rows affected), retry, read version 8, one wins, 98 fail — a quadratic retry storm that burns enormous CPU for very little forward progress. The rule of thumb: optimistic for low-collision, read-heavy, long-think-time records (editing a wiki page, updating a user profile); pessimistic for high-collision hot rows (decrementing inventory, moving money). Most real systems use both, chosen per access pattern.

There's a subtle correctness benefit to optimistic locking worth noting: because it never holds locks during the read-modify-write think time, it cannot cause lock-wait pileups or deadlocks at all. The cost is moved entirely to the retry path. That makes it a genuinely good default for the *long-tail* of your tables — the ones updated occasionally by humans — even as you keep pessimistic locking for the hot path. For the deeper transaction-isolation context behind both — how `REPEATABLE READ` and `SERIALIZABLE` change what counts as a conflict — see [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) and the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), since under MVCC reads don't take locks at all and the entire locking story above applies only to writes and explicit `SELECT ... FOR ...`.

## 11. Two-phase locking: the theory behind serializability

**Senior rule of thumb: every lock-based isolation guarantee is some flavor of two-phase locking; knowing the theory tells you exactly which anomalies a lock can and cannot prevent.**

It's worth grounding all of this in the theory, because it explains *why* the engines behave as they do. In Martin Kleppmann's *Designing Data-Intensive Applications* (Chapter 7, "Transactions"), the canonical lock-based mechanism for serializability is **two-phase locking (2PL)**. The "two phases" are not begin/commit; they are the *growing* phase, during which a transaction may acquire locks but not release any, and the *shrinking* phase, during which it may release locks but not acquire any. In practice databases use *strict* 2PL: all locks are held until the transaction commits or aborts (so the shrinking phase is instantaneous, at commit). This is exactly the "locks held until transaction end" behavior we observed in section 1.

2PL gives serializability — the strongest isolation, where concurrent transactions behave as if run one at a time — but it does so at a cost that Kleppmann is blunt about: 2PL has dramatically worse performance and concurrency than optimistic or snapshot-based approaches, precisely because of all the waiting and the deadlocks. Readers block writers and writers block readers under 2PL, which is why most databases default to MVCC-based snapshot isolation (where readers never block writers) and reserve true 2PL-style locking for the rows you explicitly lock with `FOR UPDATE` or for `SERIALIZABLE` mode.

The other crucial idea from that chapter is the distinction between locking a *thing that exists* and locking a *thing that doesn't yet exist*. A record lock protects an existing row. But to prevent a **phantom** — a row that *would* match your query if someone inserted it — you cannot lock the row, because it doesn't exist. You must lock the *condition*. Kleppmann describes two mechanisms: **predicate locks**, which lock all rows matching a search predicate (including future ones), and **index-range locks** (a.k.a. next-key locks), which approximate predicate locks by locking a range of an index. True predicate locks are expensive, so real engines use index-range locking — which is *exactly* InnoDB's next-key lock from section 3. The gap lock is the practical, index-based approximation of the theoretical predicate lock. When you understand that a next-key lock is "an index-range lock standing in for a predicate lock to prevent phantoms," the entire gap-lock mechanism stops being a MySQL quirk and becomes an instance of a fundamental concept. Postgres takes a different route to the same guarantee: instead of gap locks it uses **Serializable Snapshot Isolation (SSI)**, which detects the *read-write dependencies* that would constitute a phantom and aborts a transaction at commit — optimistic detection rather than pessimistic prevention, but achieving the same `SERIALIZABLE` correctness.

> Two-phase locking buys you serializability with a credit card called "throughput," and deadlocks are the interest. MVCC and SSI are the database industry's long campaign to pay less of that interest.

## Case studies from production

### 1. The Shopify batch-upsert gap-lock deadlock

The cleanest real-world gap-lock deadlock is Shopify's, documented in [Mitigating Deadlocks in High Concurrency Environments](https://shopify.engineering/mitigating-deadlocks-in-high-concurrency-environments). Multiple ETL processes ran concurrent batch *upserts* into one MySQL table. The symptom was `ERROR 1213` deadlocks even though, the engineers swore, each process operated on a *different batch of records*. The wrong first hypothesis was the obvious one: "two processes must be touching the same row." They weren't. The actual root cause was gap locks. When an upsert updates an existing record, InnoDB takes a gap lock "that targets the actual record being updated and the one right before it in the index structure." With a sequential auto-increment primary key, records from different batches were stored *physically adjacent* on disk, so the gap locks from different processes *overlapped* even though the rows didn't — and overlapping gap locks acquired in different orders by the insert-intention machinery closed a cycle.

The fix was pure schema, no application change: they added the `account` column to the primary key, making it a **composite primary key** `(account, id)`. This physically clustered each account's rows together, so "the gap locks required stay within each process instance, reducing the chances of overlapping gaps dramatically." They preserved backward compatibility by adding a separate unique index on `id` alone. The lesson is the deepest one in this article: *physical data layout determines lock behavior.* The same logical operation deadlocks or doesn't depending entirely on how rows are clustered on disk, which is controlled by the primary key — and that's a [B-tree clustering](/blog/software-development/database/b-trees-how-database-indexes-work) decision. "The overall throughput of our import jobs increased significantly by reducing the wait time of those jobs."

### 2. The 2 a.m. money-transfer deadlock (mine)

My own opening story. A payments service did transfers as `BEGIN; UPDATE accounts SET balance = balance - ? WHERE id = ?; UPDATE accounts SET balance = balance + ? WHERE id = ?; COMMIT;`. The two updates locked the source then the destination, in *whatever order the transfer happened to go*. A transfer from account 1→2 locked row 1 then row 2; a simultaneous transfer 2→1 locked row 2 then row 1. Opposite order, classic cycle. Under normal load the window was tiny and we saw a deadlock a week. During a promotion that drove a burst of mutual transfers (refunds crossing with charges on the same account pairs), the window widened and the deadlock rate hit 4% of checkouts. The fix was three lines: sort the two account IDs and always lock the lower id first — `SELECT ... FROM accounts WHERE id IN (?, ?) ORDER BY id FOR UPDATE`. Consistent lock ordering makes the cycle impossible: every transfer now locks `min(a,b)` then `max(a,b)`, so two transfers on the same pair always acquire in the same order and simply queue instead of deadlocking. Deadlock rate went to zero. We also added the retry wrapper from section 6 as defense in depth.

### 3. GitHub's read-replica deadlock cascade

GitHub experienced a deadlock-triggered availability incident where read replicas that hit a deadlock entered a crash-recovery state, increasing load on the remaining healthy replicas. "Due to the cascading nature of this scenario, there were not enough active read replicas to handle production requests, which impacted the availability of core GitHub services." The lesson here isn't about the deadlock mechanism — it's about *blast radius*. A deadlock on a primary is a local event (one transaction rolls back). But a deadlock interacting with replication and crash-recovery can cascade: the recovery load shifts to healthy nodes, which then struggle, which shifts more load, a classic metastable failure. The takeaway: deadlock *handling* must be cheap and bounded, because deadlocks under load come in bursts, and a slow or amplifying recovery path turns a survivable burst into an outage. Cap your retry attempts, add jitter, and make sure a deadlock storm degrades throughput linearly rather than triggering recovery cascades.

### 4. The job queue that serialized to one worker

A team built a job queue as `SELECT id FROM jobs WHERE state='ready' ORDER BY created_at LIMIT 1 FOR UPDATE`, then updated the row to `running`. They scaled to 20 workers and throughput *did not improve at all*. The wrong hypothesis was "the database is the bottleneck." It wasn't — the database was 90% idle. The real cause: all 20 workers selected the *same* oldest row, one acquired the `FOR UPDATE` lock, and the other 19 *blocked* on that exact row. They had a 20-worker pool with a concurrency of one. Adding `SKIP LOCKED` (section 7) was a one-word fix: now each worker that finds the oldest row locked simply skips to the next available one, and all 20 drain in parallel. Throughput went up ~18× (not a clean 20× because of the `UPDATE` churn and commit overhead). The lesson: `FOR UPDATE` without `SKIP LOCKED` on a queue is a thundering-herd-on-one-row anti-pattern, and it presents as "scaling does nothing," not as an error.

### 5. The idle-in-transaction connection that froze a table

An incident where a migration `ALTER TABLE` hung indefinitely and then every query on that table started timing out. The `ALTER` needed `ACCESS EXCLUSIVE`, which conflicts with everything — but it was stuck behind a single connection in `state = 'idle in transaction'`. A developer had run `BEGIN; SELECT * FROM big_table WHERE id = 1;` in a `psql` session and walked away to lunch. That open transaction held an `ACCESS SHARE` lock on `big_table`. The `ALTER`'s `ACCESS EXCLUSIVE` request couldn't be granted while `ACCESS SHARE` was held, so it queued — and crucially, in Postgres, a *pending* `ACCESS EXCLUSIVE` request blocks all *new* lock requests behind it (to prevent writer starvation). So every new `SELECT` (wanting `ACCESS SHARE`) queued behind the `ALTER`, and the whole table froze. The diagnosis was the `pg_blocking_pids()` query from section 7, which immediately showed the idle `psql` PID at the root of the wait chain. The fixes: `SET idle_in_transaction_session_timeout = '60s'` to auto-kill abandoned transactions, and `SET lock_timeout = '3s'` on migrations so a blocked `ALTER` backs off instead of freezing the table.

### 6. The missing index that locked the whole table

A `DELETE FROM events WHERE session_id = ?` started causing massive lock waits under load. `session_id` had no index. As covered in section 3's second-order note, InnoDB locks the index records it *scans*, and with no usable index the `DELETE` scanned the entire clustered index, taking next-key locks on *every row it examined* — effectively an exclusive table lock for the duration of the scan. Concurrent inserts and updates all piled up behind it. `EXPLAIN` showed `type: ALL` (full scan). Adding `CREATE INDEX idx_events_session ON events(session_id)` changed the access path so the `DELETE` locked only the matching records, and the lock waits vanished. The lesson worth tattooing: *the lock footprint equals the scan footprint.* If you don't know which index a locking statement uses, run `EXPLAIN` — an unindexed `UPDATE`/`DELETE` doesn't just run slowly, it locks enormously.

### 7. The counter that wouldn't scale (and the multirow fix)

A view-counter feature: `UPDATE pages SET views = views + 1 WHERE id = ?` on hot pages. A handful of viral pages got thousands of concurrent views, all serializing on one row's X lock. The p99 on those updates climbed to seconds and `lock_wait` timeouts started firing. The team first tried throwing more database connections at it — no effect, because the bottleneck was a single row's lock, not connection count (per-row Amdahl's law from section 8). The real fix was counter sharding à la Shopify's `multirow_counter`: 16 shard rows per page, increment a random shard, `SUM` on read. The 16-way split gave ~16× the write concurrency on the hot rows, p99 dropped back under 10 ms, and the timeouts stopped. For the very hottest pages they went further and *coalesced* increments in the application (buffer counts in memory, flush every few seconds), trading sub-second accuracy for another order of magnitude of headroom.

### 8. The foreign-key deadlock that the four-mode split fixed

On an old Postgres (pre-9.3) application, inserting child rows (order line items) while concurrently updating parent rows (orders) deadlocked constantly. The reason: every child insert took a full `FOR SHARE` lock on the parent to validate the foreign key, and every parent update took `FOR UPDATE`, and these two fought. When the team upgraded to a version with the four-way row-lock split (section 2), the deadlocks largely disappeared *without any code change*: child inserts now take only `FOR KEY SHARE` on the parent (they only need the parent's *key* to stay valid), parent updates that don't touch key columns take only `FOR NO KEY UPDATE`, and those two modes are *compatible*. The lesson is partly "keep your database current," but more deeply: the existence of `FOR NO KEY UPDATE` and `FOR KEY SHARE` is not academic — it was added specifically to kill this extremely common foreign-key deadlock, and understanding it tells you that the *kind* of update you do (key column vs not) changes your lock footprint.

### 9. The advisory-lock leak through a connection pool

A team used session-level advisory locks (`pg_advisory_lock`) to ensure single-instance cron execution. It worked in testing. In production, behind PgBouncer in transaction-pooling mode, it broke bizarrely: crons would occasionally refuse to run at all, claiming the lock was held, even when no cron was running. The root cause: session-level advisory locks live on the *connection*, but transaction-mode connection pooling hands a different physical connection to each transaction, so the `pg_advisory_lock` was acquired on one pooled connection and the `pg_advisory_unlock` ran on a *different* one — leaking the lock on the first connection, which then sat in the pool holding a phantom lock. The fix was to switch to *transaction-level* advisory locks (`pg_advisory_xact_lock`), which release automatically at commit regardless of which connection the next transaction lands on. Lesson: session-scoped database state and transaction-mode pooling are fundamentally incompatible; always prefer transaction-level advisory locks unless you control the connection lifecycle end to end.

### 10. The Stripe-style retry-storm self-DDoS

A general pattern Stripe-class systems guard against, and which I've seen take down a service: a deadlock retry loop *without* backoff or jitter. Two hot transactions deadlocked; both retried *immediately*; they re-deadlocked at the same instant; retried again; forever. The retry loop turned a transient deadlock into a sustained 100%-CPU spin that starved every other transaction — a self-inflicted denial of service driven by the very mechanism meant to recover from deadlocks. The retry loop *worked* in the sense that it kept retrying; it just never made progress because both parties retried in lockstep. The fix is the jittered exponential backoff in section 6's retry wrapper: randomizing the delay desynchronizes the retriers so one wins on the next attempt. Lesson: a retry without backoff and jitter is not a recovery mechanism, it's an amplifier. Always cap attempts, always back off, always add jitter.

### 11. The SERIALIZABLE migration that traded deadlocks for serialization failures

A team moved a correctness-critical workflow from `READ COMMITTED` with explicit `FOR UPDATE` to `SERIALIZABLE` isolation, hoping to "stop thinking about locks." Deadlocks dropped to near zero — but a new error appeared: `ERROR: could not serialize access due to read/write dependencies among transactions` (SQLSTATE `40001`), Postgres's SSI abort. They'd traded pessimistic lock-deadlocks for optimistic serialization-failures. The good news: the *handling* is identical — `40001` retries with the exact same backoff wrapper as `40P01`. The lesson, straight from Kleppmann's chapter: `SERIALIZABLE` via SSI doesn't eliminate the "some transaction must restart" reality; it just moves the abort from "deadlock detected mid-flight" to "dependency cycle detected at commit." Either way, *your application must have a retry loop.* There is no isolation level that frees you from handling transaction restarts.

### 12. The NOWAIT that saved a checkout flow

A high-traffic flash-sale checkout used `SELECT ... FOR UPDATE` on inventory rows. During the sale, thousands of users hit the same popular SKU; the `FOR UPDATE` requests queued, each waiting up to `lock_wait_timeout` (50 s by default in MySQL) behind the holder. Users sat on a spinning checkout button for tens of seconds, then got a timeout error anyway. The fix was `SELECT ... FOR UPDATE NOWAIT`: if the inventory row is currently locked, fail *instantly* with SQLSTATE `55P03` (Postgres) instead of queuing, and the application immediately returns "this item is busy, try again" — a fast, honest failure instead of a 50-second hang. Combined with a short client-side retry, the perceived latency went from "30-second hang then error" to "instant retry, usually succeeds within a second." Lesson: when a lock wait is *expected* to be common and a fast failure is better UX than a long wait, `NOWAIT` converts an invisible hang into a catchable, actionable error.

## When to reach for locking — and when not to

Reach for **pessimistic locking** (`SELECT ... FOR UPDATE`) when:

- You have a genuine hot row that *will* be contended (inventory decrement, money transfer, a single aggregate row) and you need writers to make guaranteed forward progress rather than fight.
- The transaction is short and does no I/O while holding the lock — you can acquire, mutate, and commit in milliseconds.
- Correctness depends on no concurrent modification between your read and your write (the read-modify-write must be atomic).
- You can guarantee a consistent lock-acquisition order across all code paths, eliminating the deadlock cycle by construction.

Reach for **`SKIP LOCKED` / `NOWAIT`** when:

- You're building a job/work queue and want N workers to drain it in parallel without contending on the same row (`SKIP LOCKED`).
- A fast failure is better UX than a long wait, and "couldn't get the lock right now" is a normal, recoverable outcome (`NOWAIT`).

Reach for **optimistic locking** (version column) when:

- Conflicts are rare (low-collision records) and retries are cheap.
- A human is in the loop with long think-time (editing a profile, a wiki page, a draft) — you must not hold a lock across human latency.
- The workload is read-heavy and you want reads to never block.

Reach for **advisory locks** when:

- You need to serialize *application logic*, not table rows ("only one cron at a time," "single-flight cache rebuild").
- Prefer the transaction-level variant (`pg_advisory_xact_lock`) so the lock can't leak.

**Skip explicit locking entirely** when:

- MVCC already gives you what you need. Plain reads under snapshot isolation don't block writers and don't need locks; reaching for `FOR UPDATE` on a read you're not going to write back is pure downside.
- You're tempted to `LOCK TABLE` to "make it simple." That trades a row-level concurrency problem for a table-level serialization point — almost always the wrong trade. Find the finest granularity that's correct.
- The contention is on a counter you could shard or make append-only. Don't lock harder; restructure the data so the writes don't collide in the first place.
- You think `SERIALIZABLE` will let you "stop thinking about locks." It won't — it trades deadlocks for serialization failures, and you still need the retry loop. Use it when you genuinely need serializable correctness, not as a way to avoid understanding concurrency.

The throughline of every one of these decisions and every case study above is a single principle: **a lock is a serialization point, and the art is putting it on the smallest possible thing for the shortest possible time, in a consistent order, with a retry loop for when the engine breaks a tie you couldn't avoid.** Locks aren't something to fear — they're the mechanism by which a database keeps its promises under concurrency. But every promise has a cost denominated in someone's wait time, and a production engineer's job is to make sure that cost is microseconds, spread across many rows, and never, ever a 2 a.m. page.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 7 ("Transactions") — two-phase locking, predicate and index-range locks, serializability, and SSI. The canonical theory under everything above.
- MySQL Reference Manual: [InnoDB Locking](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html) and [Locks Set by Different SQL Statements in InnoDB](https://dev.mysql.com/doc/refman/8.0/en/innodb-locks-set.html) — the precise spec for record/gap/next-key/insert-intention locks.
- PostgreSQL docs: [Explicit Locking](https://www.postgresql.org/docs/current/explicit-locking.html) (the full table- and row-level conflict matrices) and [pg_locks](https://www.postgresql.org/docs/current/view-pg-locks.html).
- Shopify Engineering: [Mitigating Deadlocks in High Concurrency Environments](https://shopify.engineering/mitigating-deadlocks-in-high-concurrency-environments) and the [multirow_counter](https://github.com/Shopify/multirow_counter) library.
- [Understand deadlock by gap locking in InnoDB](https://tanishiking.hashnode.dev/avoid-deadlock-caused-by-a-conflict-of-transactions-that-accidentally-acquire-gap-lock-in-innodb-a114e975fd72) — the clearest walkthrough of the gap-lock + insert-intention deadlock.
- [Potential Consequences of Using Postgres as a Job Queue](https://richyen.com/postgres/2026/05/04/postgres_job_queue.html) — the vacuum/churn caveats of the `SKIP LOCKED` pattern.
- Sibling posts on this blog: the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work), and [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines).

---
title: "Transaction Isolation Levels and the Anomalies They Prevent"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles, reproduce-it-yourself tour of dirty reads, write skew, phantoms, and the gap between what ANSI isolation levels promise and what Postgres and MySQL actually deliver."
tags:
  [
    "transactions",
    "isolation-levels",
    "mvcc",
    "serializability",
    "write-skew",
    "snapshot-isolation",
    "postgres",
    "mysql",
    "concurrency",
    "database",
    "two-phase-locking",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-1.webp"
---

There is a specific kind of production incident that ruins a senior engineer's week, and it always starts the same way. The code is correct. You read it ten times. The test suite is green. The `SELECT` returns the right number, the `UPDATE` writes the right number, and the transaction commits cleanly. And yet the ledger is off by a few cents, or two doctors both went off-call, or you sold 101 of a product you had 100 of. There is no exception in the logs. There is no failed query. The database did exactly what you told it to — and the result is still wrong.

What you have hit is a **concurrency anomaly**: a behavior that is impossible if your transactions had run one at a time, but which becomes possible the instant two of them overlap. The database's defense against these is the **isolation level**, the "I" in ACID, and it is simultaneously the most important knob in your database and the one almost nobody sets deliberately. Most teams ship on whatever default their database picked — `READ COMMITTED` in PostgreSQL, `REPEATABLE READ` in MySQL/InnoDB — without knowing that those two defaults are *different levels with different guarantees*, that the same level name means different things in different engines, and that the ANSI SQL standard which named all of them is, by the admission of the researchers who later formalized it, broken.

This article is a tour through the whole mess, and the spine of it is the matrix below. Every isolation level is a promise about which anomalies it forbids. The diagram above is the mental model: read it as a staircase, where each step up the rows outlaws one more column of bad behavior, until the top step — `SERIALIZABLE` — is the only one that forbids the two most dangerous anomalies, **phantoms** and **write skew**, at the same time. The rest of the article is a tour through every cell of that matrix: what each anomaly is, how to reproduce it on your own laptop with two `psql` sessions, why the levels stop where they stop, and which level you actually need.

## The mental model: isolation is a promise about anomalies

![A matrix of five isolation levels against six anomalies, showing which each one blocks or allows](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-1.webp)

Read the figure top to bottom. `READ UNCOMMITTED` blocks essentially nothing except dirty *writes*. Each level below it adds one more guarantee: `READ COMMITTED` stops you from reading uncommitted data; `REPEATABLE READ` (here, MySQL/InnoDB's gap-locking flavor) and `SNAPSHOT ISOLATION` stop a row from changing value mid-transaction; and only `SERIALIZABLE` closes the last two columns — phantoms and write skew. The single most important thing this matrix teaches is the gap in the second-to-last row: **snapshot isolation gives you a stable, consistent view of the whole database and still permits write skew.** That one fact is responsible for a disproportionate share of the "the code is correct but the data is wrong" incidents in this industry, and we will spend a long section on it.

> An isolation level is not a performance setting. It is a *correctness contract* that happens to have a performance cost. Choosing it by benchmark instead of by the invariants your application must hold is how you end up debugging a ledger at 2 a.m.

Before we go further, it is worth being precise about the mismatch between what engineers assume an isolation level does and what it actually does, because almost every isolation bug is a gap between those two columns.

| You assume | The naive mental model | The reality |
| --- | --- | --- |
| "The default level is safe enough." | The vendor picked a sensible safe default. | Postgres defaults to `READ COMMITTED`, which permits read skew, lost updates, phantoms, and write skew. The default protects you from almost nothing. |
| "`REPEATABLE READ` means the same thing everywhere." | It's a standard; it's standardized. | Postgres `REPEATABLE READ` is *snapshot isolation*; MySQL `REPEATABLE READ` uses gap locks and has different (and, per Jepsen, sometimes broken) semantics. |
| "`SERIALIZABLE` is just slow `REPEATABLE READ`." | One more notch of the same dial. | `SERIALIZABLE` is a categorically different guarantee — it forbids write skew, which no weaker level does — and it can *abort your transaction* with a serialization error you must be coded to retry. |
| "If my query is correct, my transaction is correct." | Correctness is a property of statements. | Correctness under concurrency is a property of the *schedule*. A correct statement run on a stale snapshot produces a wrong commit. |
| "My ORM handles this." | The framework abstracts it away. | Your ORM issues `read-modify-write` cycles that are textbook lost-update generators unless you opt into atomic updates, `SELECT ... FOR UPDATE`, or `SERIALIZABLE` + retry. |

Hold that table; we will hit every row of it with a concrete, runnable reproduction. If you have Postgres and MySQL handy, open two terminal sessions for each anomaly and run the transcripts as you read — feeling an anomaly happen on your own laptop is worth more than any prose.

## 1. What a transaction actually promises

**Senior rule of thumb: a transaction promises you a serial *illusion*, not serial *execution*. The whole game of a database is to run transactions concurrently while making each one believe it ran alone.**

The four letters of ACID — Atomicity, Consistency, Isolation, Durability — are not four independent features; they are four facets of one promise. Atomicity says a transaction is all-or-nothing: if it fails partway, the database reverses everything, as though it never started. Durability says that once you get a commit acknowledgment, the data survives a power cut. Consistency is the application's job as much as the database's: it says the transaction moves the database from one valid state to another, where "valid" is defined by your invariants (foreign keys, `CHECK` constraints, and the business rules the database cannot see, like "at least one doctor is always on call").

Isolation is the one this article is about, and it is the slipperiest. The textbook definition is **serializability**: a concurrent execution of transactions is correct if its outcome is identical to *some* serial execution of those same transactions, run one after another with no overlap. Note the "some": the database is free to pretend the transactions ran in whatever order is convenient, as long as it picks *one* order and the result is consistent with it. If transaction A and transaction B overlap in wall-clock time, a serializable database guarantees the final state is either "as if A ran fully, then B" or "as if B ran fully, then A" — never some impossible blend that neither serial order could produce.

Here is the catch, and it is the catch that the entire rest of the article elaborates: **serializability is expensive.** To guarantee it perfectly, the database must either lock aggressively (blocking transactions that *might* conflict, even when they don't) or track every read-write dependency and abort transactions when it detects a dangerous pattern (forcing you to retry). Both cost throughput. So in practice databases offer a menu of *weaker* isolation levels that allow some concurrency anomalies in exchange for going faster. Choosing an isolation level is choosing which anomalies you are willing to tolerate.

Let us set up a running example we will reuse for the rest of the article. We will use a simple bank-account schema in PostgreSQL, because money is the domain where everyone's intuition for "this must be correct" is strongest.

```sql
-- Run once in any session to set up.
CREATE TABLE accounts (
    id      int PRIMARY KEY,
    owner   text NOT NULL,
    balance int  NOT NULL CHECK (balance >= 0)
);

INSERT INTO accounts (id, owner, balance) VALUES
    (1, 'alice',   500),
    (2, 'alice',   300),   -- alice's savings
    (3, 'bob',     500);

-- Check the current isolation level of your session at any time:
SHOW transaction_isolation;     -- Postgres
-- SELECT @@transaction_isolation;  -- MySQL
```

Everything that follows assumes you can open two independent client sessions — call them **Session A** and **Session B** — that each control their own transaction. In `psql` that is two terminals. In MySQL it is two `mysql` clients. The anomalies only appear when the two sessions *interleave*, so the order in which you type the statements matters enormously; I will mark the intended ordering with `t1`, `t2`, `t3`, … comments so you can reproduce the exact schedule.

## 2. Dirty write and dirty read: the floor of the basement

**Senior rule of thumb: even the weakest usable isolation level forbids dirty writes. If your database lets one uncommitted transaction overwrite another uncommitted transaction's write, throw the database away.**

### Dirty write

A **dirty write** happens when transaction A overwrites a value that transaction B has written but not yet committed. It is the most catastrophic anomaly because it can violate atomicity itself: imagine A and B both transfer money, and their writes to two rows interleave so that one row reflects A's transfer and the other reflects B's. When one of them rolls back, the database cannot cleanly undo it, because the other transaction has already built on top of the dirty value. Every real database — at *every* isolation level, including `READ UNCOMMITTED` — prevents dirty writes by holding a row-level write lock from the moment a row is modified until the transaction ends. The second writer simply blocks until the first commits or aborts. This is the one guarantee you get for free, which is why the matrix shows "blocked" in the dirty-write column for every level.

### Dirty read

A **dirty read** is the read-side twin: transaction B reads a value that transaction A has written but not yet committed — and A then rolls back, so B has now acted on a number that, in the database's official history, never existed.

![A two-session timeline where session A writes then rolls back, and session B reads the uncommitted value in between](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-2.webp)

The figure shows the schedule. Session A debits the account but has not committed. In the gap, Session B reads the debited balance. Then A rolls back — perhaps the transfer was declined downstream — and the balance returns to its original value. Session B is now holding `400`, a number that was never a committed state of the database. If B used it to decide whether to approve a withdrawal, it just approved against phantom money.

PostgreSQL's MVCC architecture makes dirty reads *impossible to enable* — even if you ask for `READ UNCOMMITTED`, Postgres silently gives you `READ COMMITTED`, because its storage model simply never exposes uncommitted row versions to other transactions. MySQL/InnoDB does honor `READ UNCOMMITTED` and will show you dirty reads. Here is the reproduction in MySQL:

```sql
-- ===== MySQL: reproducing a dirty read at READ UNCOMMITTED =====

-- Session A (the writer)
-- t1:
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;   -- 500 -> 400, NOT committed
-- ...pause here, do not commit yet...

-- Session B (the reader), in a second client:
-- t2:
SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
START TRANSACTION;
SELECT balance FROM accounts WHERE id = 1;   -- t3: returns 400 (!!) — A has not committed
COMMIT;

-- Session A, back in the first client:
-- t4:
ROLLBACK;   -- balance is restored to 500; B already read the phantom 400
```

Run the same schedule with Session B at `READ COMMITTED` (`SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;`) and the `SELECT` at `t3` returns `500` — B sees only A's *committed* state, and A has not committed, so B sees the value as it was before A started. That is the entire point of `READ COMMITTED`: a read never returns data that is not committed. The cost is essentially nothing, which is why nobody should ever run at `READ UNCOMMITTED` in production. There is no workload for which dirty reads are an acceptable price. The level exists in the standard mostly as a cautionary tale.

> `READ UNCOMMITTED` is a trap door labeled "faster." Postgres won't even let you open it. MySQL will, and the only thing on the other side is data corruption you can't reproduce on demand.

## 3. The ANSI standard and why its definitions are flawed

**Senior rule of thumb: the ANSI SQL-92 isolation levels are defined by *which of three anomalies they prevent*, and that list of three anomalies is incomplete. The level names survived; the definitions did not.**

The 1992 ANSI SQL standard defined four isolation levels — `READ UNCOMMITTED`, `READ COMMITTED`, `REPEATABLE READ`, `SERIALIZABLE` — by enumerating three phenomena and saying which each level must prevent:

- **P1 (Dirty Read):** reading uncommitted data.
- **P2 (Non-Repeatable Read):** reading a row twice and getting two different values because another transaction committed an update in between.
- **P3 (Phantom):** running a predicate query twice and getting a different set of rows because another transaction committed an insert/delete in between.

The standard then defined the levels as a ladder: `READ UNCOMMITTED` prevents nothing; `READ COMMITTED` prevents P1; `REPEATABLE READ` prevents P1 and P2; `SERIALIZABLE` prevents P1, P2, and P3. Clean, tidy, and — as a 1995 paper by Berenson, Bernstein, Gray, Melton, O'Neil, and O'Neil ([*A Critique of ANSI SQL Isolation Levels*](https://arxiv.org/abs/cs/0701157)) demonstrated — fundamentally broken in two ways.

The first flaw is **ambiguity**. The phenomena are written in loose English ("transaction T2 modifies or deletes a row that T1 has read"), and depending on whether you read them *strictly* (the phenomenon is forbidden as a *possibility*) or *loosely* (forbidden only as an *actual occurrence*), you get wildly different guarantees. Real database vendors interpreted them differently, which is the root cause of the headline problem: `REPEATABLE READ` means different things in Oracle, PostgreSQL, MySQL, and SQL Server, even though they all cite the same standard.

The second flaw is **incompleteness**. The three phenomena do not cover all the anomalies that can occur. The most important omission is **write skew**, which we will see is allowed by any level short of `SERIALIZABLE` and is invisible to the P1/P2/P3 framework entirely. The standard also fails to cleanly name **lost update**. The Berenson critique fixed this by adding more phenomena (P4 lost update, A5A read skew, A5B write skew) and, crucially, by introducing **snapshot isolation** as a distinct level that the ANSI ladder cannot place — SI prevents P1, P2, and P3, so by the ANSI definition it "is" serializable, yet it permits write skew, so it provably is *not*.

This is why the modern way to reason about isolation is the one Adya, and later Kleppmann in *Designing Data-Intensive Applications*, popularized: **forget the level names; reason about which concrete anomalies your application can tolerate, then pick the level that forbids exactly those.** The matrix at the top of this article is that reasoning made visual. The level names are a lossy compression of the real question. Here is the standard's ladder against the anomalies it actually covers and the ones it misses:

| ANSI level | Prevents (per ANSI) | Silently allows | The real story |
| --- | --- | --- | --- |
| `READ UNCOMMITTED` | nothing (allows P1, P2, P3) | dirty read, read skew, lost update, phantom, write skew | Never use it. |
| `READ COMMITTED` | P1 (dirty read) | read skew, lost update, phantom, write skew | The Postgres default. Far weaker than most teams assume. |
| `REPEATABLE READ` | P1, P2 | phantom (per ANSI), write skew, sometimes lost update | The name is the same everywhere; the meaning is not. |
| `SERIALIZABLE` | P1, P2, P3 | (nothing, *if implemented correctly*) | The only level that forbids write skew. Also the only one that aborts you. |

Notice that `SERIALIZABLE`'s "(nothing)" carries a giant asterisk: *if implemented correctly.* As we will see in the case studies, "we implement serializable" has been a falsifiable claim that several major databases have failed.

## 4. Read skew (non-repeatable read): the same row, two values

**Senior rule of thumb: if a single transaction reads the same data twice and needs the two reads to agree, `READ COMMITTED` is not enough — you need a snapshot.**

A **non-repeatable read**, also called **read skew**, is anomaly P2: within one transaction, you read a row, someone else commits an update to it, you read it again, and you get a different value. It sounds benign — "so what, the data changed" — but it is poison for any computation that reads multiple rows and assumes they are mutually consistent. The classic failure is a backup or a report: you sum across a thousand rows, and halfway through your scan, a transfer commits that moves $100 from account #1 (which you already counted at its old value) to account #999 (which you count at its new value). Your report now shows $100 that exists in neither account, or $100 counted twice. The data on disk is perfectly consistent; your *read of it* is not.

![A two-session timeline where session B reads a balance, session A commits a change, and B re-reads and sees a different value](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-3.webp)

Here is the reproduction in PostgreSQL at the default `READ COMMITTED` level:

```sql
-- ===== Postgres: non-repeatable read at READ COMMITTED =====

-- Session B (the reader)
-- t1:
BEGIN;  -- default READ COMMITTED
SELECT balance FROM accounts WHERE id = 1;   -- returns 500

-- Session A (the writer), in a second terminal:
-- t2:
BEGIN;
UPDATE accounts SET balance = 600 WHERE id = 1;
COMMIT;                                        -- t3: now committed and visible

-- Session B again:
-- t4:
SELECT balance FROM accounts WHERE id = 1;   -- returns 600 (!!) — same txn, different value
COMMIT;
```

Within Session B's single transaction, the same `SELECT` returned `500` and then `600`. That is the non-repeatable read. Under `READ COMMITTED`, *each statement* sees a fresh snapshot of the latest committed data, so a long-lived transaction sees the world shift under its feet.

The fix is to take the snapshot once, at the start of the transaction, and hold it. That is exactly what `REPEATABLE READ` does in PostgreSQL (which, recall, is implemented as snapshot isolation). Re-run the schedule with Session B opening `BEGIN ISOLATION LEVEL REPEATABLE READ;`:

```sql
-- ===== Postgres: the same schedule at REPEATABLE READ =====

-- Session B
-- t1:
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM accounts WHERE id = 1;   -- returns 500, and PINS the snapshot

-- Session A: BEGIN; UPDATE...600; COMMIT;   -- t2, t3 (commits fine)

-- Session B
-- t4:
SELECT balance FROM accounts WHERE id = 1;   -- returns 500 again — repeatable!
COMMIT;
```

Now both reads return `500`, because Session B is reading the database *as it existed at the moment `t1` ran*. Session A's commit is real and durable; B simply cannot see it, because it happened after B's snapshot was taken. The cost of this guarantee in an MVCC system is close to zero on the read path — no extra locks, no blocking — which is why snapshot-based `REPEATABLE READ` is such a good default for read-heavy reporting workloads. We will see its limits soon.

## 5. Lost update: the increment that vanishes

**Senior rule of thumb: any `read-modify-write` cycle — read a value into the application, compute a new value, write it back — is a lost-update bug waiting for a second concurrent caller. Either make the write atomic, lock the row on read, or run serializable and retry.**

A **lost update** is the anomaly your ORM generates by default. Two transactions each read the same value, each compute a new value based on what they read, and each write it back. The second write clobbers the first, and one of the two updates is silently gone. The canonical example is a counter: a "likes" count, an inventory quantity, a wallet balance.

![A before-after comparison of a counter increment: the naive read-modify-write path loses an update, the atomic path counts both](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-5.webp)

The left side of the figure is the bug. Both transactions read `counter = 10`. Both compute `10 + 1 = 11`. Both write `11`. The final value is `11`, but two increments happened — it should be `12`. One `+1` evaporated. The right side is the fix: an atomic `UPDATE ... SET counter = counter + 1`, or a `SELECT ... FOR UPDATE` that locks the row so the second transaction blocks until the first commits.

Here is the lost update reproduced at `READ COMMITTED` in Postgres, using the application-style read-modify-write that an ORM emits:

```sql
-- ===== Postgres: lost update at READ COMMITTED =====

-- Session A
-- t1:
BEGIN;
SELECT balance FROM accounts WHERE id = 1;   -- reads 500; app computes 500 + 100 = 600

-- Session B
-- t2:
BEGIN;
SELECT balance FROM accounts WHERE id = 1;   -- ALSO reads 500; app computes 500 + 50 = 550

-- Session A
-- t3:
UPDATE accounts SET balance = 600 WHERE id = 1;
COMMIT;                                       -- balance is now 600

-- Session B
-- t4:
UPDATE accounts SET balance = 550 WHERE id = 1;
COMMIT;                                       -- balance is now 550 — A's +100 is LOST
```

The account should hold `500 + 100 + 50 = 650`. It holds `550`. Alice's $100 deposit is gone, and nothing errored. This is *the* most common concurrency bug in application code, and it survives `READ COMMITTED` and even snapshot-isolation `REPEATABLE READ` (in Postgres, the second `COMMIT` would actually error — more on that below). There are four standard fixes, in rough order of preference:

```sql
-- Fix 1: atomic write — never read into the app at all. ALWAYS prefer this.
UPDATE accounts SET balance = balance + 100 WHERE id = 1;

-- Fix 2: pessimistic lock — read FOR UPDATE so the second reader blocks.
BEGIN;
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;  -- exclusive row lock
-- compute in app, then:
UPDATE accounts SET balance = 600 WHERE id = 1;
COMMIT;

-- Fix 3: optimistic concurrency — a version column, write only if unchanged.
UPDATE accounts SET balance = 600, version = version + 1
WHERE id = 1 AND version = 7;        -- if rowcount = 0, someone beat you: retry

-- Fix 4: run at REPEATABLE READ / SERIALIZABLE and retry on serialization failure.
```

The interesting one is Fix 4. In PostgreSQL, snapshot-isolation `REPEATABLE READ` *detects* the lost update and refuses it. Re-run the schedule above with both sessions at `REPEATABLE READ`, and Session B's `UPDATE` at `t4` fails with:

```ERROR:  could not serialize access due to concurrent update
```

This is the SQLSTATE `40001` (`serialization_failure`) error, and it is the central fact of life when you run above `READ COMMITTED`. Postgres uses a rule called **first-committer-wins**: when an update at `REPEATABLE READ` discovers that the row it is trying to modify was changed by a transaction that committed *after* its snapshot, it aborts rather than silently lose the write. Your application must catch `40001` and retry the entire transaction from the top — re-reading the now-updated value and recomputing. MySQL/InnoDB behaves differently here: at its `REPEATABLE READ`, a bare `UPDATE` takes a current read (not a snapshot read), so the second `UPDATE` blocks on the row lock and then applies on top of the committed value — which prevents *this* lost update but, as Jepsen found, does not make InnoDB's `REPEATABLE READ` anomaly-free in general.

> The cheapest correct fix for lost update is almost always to stop reading the value into your application at all. `SET x = x + 1` is one round trip, atomic, and immune. `SELECT x; ...; UPDATE x = (x+1)` is two round trips and a race. Reach for the database's arithmetic before you reach for a lock.

## 6. Phantom reads: the set changes underneath you

**Senior rule of thumb: row-versioning gives you a stable *row*, but a stable *predicate* — "all rows where status = open" — needs predicate locking, gap locks, or serializable. A snapshot freezes the rows you've seen, not the rows that don't exist yet.**

A **phantom** is the predicate-level cousin of the non-repeatable read. You run a query with a `WHERE` clause — a search over a *range* or *predicate* rather than a single key — and get a set of rows. Another transaction inserts (or deletes) a row that matches your predicate and commits. You re-run the same query and get a different set. The new row is the "phantom": it appeared in your transaction's view of a set you thought was frozen.

![A two-session timeline where session B counts matching rows, session A inserts a matching row and commits, and B's recount disagrees](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-4.webp)

Phantoms matter whenever a transaction enforces a constraint over a *set* of rows rather than a single row. "There must be at most one row with `is_primary = true` per user." "No two bookings may overlap for the same room." "There must be at least N rows matching some condition." You check the constraint by querying the set, find it satisfied, and write — but a concurrent transaction inserts a row that breaks the constraint in the window between your check and your write. Here is the phantom reproduced in Postgres at `READ COMMITTED`:

```sql
-- ===== Postgres: phantom read at READ COMMITTED =====

-- Session B (enforcing "at most 3 open tickets")
-- t1:
BEGIN;
SELECT count(*) FROM tickets WHERE status = 'open';   -- returns 3

-- Session A
-- t2:
BEGIN;
INSERT INTO tickets (status) VALUES ('open');
COMMIT;                                                 -- t3: a 4th open ticket commits

-- Session B
-- t4:
SELECT count(*) FROM tickets WHERE status = 'open';   -- returns 4 — a phantom appeared
COMMIT;
```

Snapshot isolation (Postgres `REPEATABLE READ`) actually *does* prevent the phantom from being *visible*: because B reads as of its snapshot, the second `count(*)` still returns `3`, since the inserted row is invisible to B's snapshot. This is a subtle and important point that the ANSI framework muddles: **snapshot isolation prevents phantom reads as a read phenomenon, but it does not prevent the dangerous *consequence* of phantoms — write skew over a predicate — which we get to next.** MySQL/InnoDB takes a completely different route: its `REPEATABLE READ` uses **next-key locks** (a combination of a row lock and a **gap lock** on the index range) to lock not just the rows that exist but the *gaps between them*, so a concurrent `INSERT` into the locked range *blocks*. That is how InnoDB's `REPEATABLE READ` prevents phantoms where Postgres's snapshot only hides them. The two engines reach "no visible phantom" by opposite mechanisms with opposite costs: Postgres pays in nothing on the read side but allows write skew; MySQL pays in lock contention and blocked inserts.

| Mechanism | Postgres `REPEATABLE READ` (SI) | MySQL/InnoDB `REPEATABLE READ` |
| --- | --- | --- |
| How reads stay stable | MVCC snapshot pinned at first read | MVCC consistent read (snapshot) for plain `SELECT` |
| How phantoms are handled | Hidden by the snapshot; not blocked | **Blocked** by next-key / gap locks on locking reads |
| Concurrent insert into scanned range | Allowed (invisible to reader) | **Blocked** until the locking reader commits |
| Lost update | Aborts with `40001` (first-committer-wins) | Blocks on row lock, then applies on current value |
| Write skew | **Allowed** | **Allowed** (gap locks only cover the locked predicate) |
| Read-side cost | Near zero | Lock acquisition; deadlock risk on inserts |

## 7. Snapshot isolation: a consistent view that isn't serializable

**Senior rule of thumb: snapshot isolation is the best price-to-performance isolation level ever invented, and it has exactly one fatal blind spot — write skew. Know that blind spot or it will find you.**

We have now mentioned snapshot isolation a dozen times; it deserves its own anatomy, because it is what both Postgres `REPEATABLE READ` and (effectively) Oracle's `SERIALIZABLE` actually give you, and because its blind spot is the most important concept in this whole article.

Under **snapshot isolation (SI)**, every transaction, the moment it begins (more precisely, at its first read), is assigned a consistent snapshot of the *entire database* as of that instant. For the rest of the transaction, every read sees that frozen world — no other transaction's commits leak in, ever. This is implemented with **multi-version concurrency control (MVCC)**: each row update creates a *new version* of the row tagged with the transaction that created it, and old versions are kept around so that older snapshots can still read them. A transaction reads the newest version of each row that was committed before its snapshot. Readers never block writers and writers never block readers, because they are looking at different versions of the same row. This is the heart of why MVCC databases scale read concurrency so well — and it is worth reading the [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) alongside this section, because the version-chain mechanics are what make snapshot isolation cheap.

![A timeline showing a row's version chain and a transaction reading its pinned snapshot version while ignoring later commits](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-7.webp)

The figure is the mechanism. A single row has accumulated three versions: `v1` (balance 500, created by transaction T0), `v2` (600, by T1), `v3` (700, by T3). New transactions starting now would see `v3`. But transaction T2, whose snapshot was taken back when only `v1` was committed, reads `v1 = 500` every single time it touches that row, for its entire lifetime, regardless of how many newer versions commit. T2 lives in a frozen world. The reads are stable, repeatable, and never block — and they reflect a *genuinely consistent* state of the whole database, because the snapshot was atomic.

SI is wonderful. It prevents dirty reads (you only see committed versions), non-repeatable reads (your snapshot is frozen), read skew (the whole database view is consistent), and even visible phantoms (rows inserted after your snapshot are invisible). It does all of this with near-zero read-side overhead. For write-write conflicts on the *same* row, it uses first-committer-wins to prevent lost updates. By the ANSI P1/P2/P3 definitions, SI looks serializable.

It is not. And the reason it is not is the single most important sentence in this article:

> Snapshot isolation guarantees that each transaction reads a consistent snapshot, but it does **not** guarantee that the transactions' *writes* are consistent with each other. Two transactions can each read a consistent snapshot, each make a locally-valid decision, write to *different* rows, both commit — and jointly violate an invariant that neither could have violated alone. That is **write skew**, and no amount of snapshotting prevents it, because the conflict is between a *read* in one transaction and a *write* in another, on rows that don't overlap.

## 8. Write skew: the anomaly that demands serializable

**Senior rule of thumb: any time two transactions read an overlapping set, each makes a decision based on what it read, and each writes a *different* row, you have a write-skew hazard. The classic tells are "check then act on a shared constraint" and "decrement a shared budget from separate rows."**

Write skew is the anomaly the ANSI standard forgot, the anomaly snapshot isolation cannot stop, and the reason `SERIALIZABLE` exists as a distinct level at all. The canonical illustration is the **doctors-on-call** scenario, and it is worth walking through carefully because once you see the shape, you will recognize it everywhere.

![A branching graph where two transactions read the same on-call count, each takes a doctor off-call, and the invariant breaks](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-6.webp)

The invariant is: **at least one doctor must always be on call.** The schema has a `doctors` table with an `on_call` boolean. Alice and Bob are both on call. Both, simultaneously, want to take themselves off call to go home. The application logic each runs is identical and looks completely safe: *count how many doctors are on call; if it's at least two, it's safe to take myself off.*

Now run the two transactions under snapshot isolation:

```sql
-- ===== Postgres: write skew at REPEATABLE READ (snapshot isolation) =====
-- invariant: at least one of alice/bob must stay on_call

-- Session A (Alice wants to leave)
-- t1:
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT count(*) FROM doctors WHERE on_call = true;   -- returns 2 -> safe to leave

-- Session B (Bob wants to leave), second terminal:
-- t2:
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT count(*) FROM doctors WHERE on_call = true;   -- ALSO returns 2 -> safe to leave

-- Session A
-- t3:
UPDATE doctors SET on_call = false WHERE name = 'alice';
COMMIT;                                               -- alice off; B can't see this

-- Session B
-- t4:
UPDATE doctors SET on_call = false WHERE name = 'bob';
COMMIT;                                               -- bob off too — ZERO on call (!!)
```

Both transactions read a snapshot showing two doctors on call. Both conclude, correctly *for their snapshot*, that it is safe to leave. A writes Alice's row; B writes Bob's row. They write to **different rows**, so there is no write-write conflict for SI to catch — first-committer-wins never fires, because no one is updating the same row anyone else updated. Both commit. The invariant — at least one doctor on call — is now violated, and *neither transaction did anything wrong on its own snapshot*. This is write skew. It is exactly the shape in the figure: a fan-out from a shared read, two locally-valid decisions, two disjoint writes, converging on a broken invariant.

The shape generalizes far beyond doctors:

| Domain | The shared read | The disjoint writes | The broken invariant |
| --- | --- | --- | --- |
| On-call scheduling | count of on-call staff | each removes a different person | at least one stays on call |
| Meeting-room / hotel booking | "no overlapping booking exists" | each inserts a non-overlapping-with-itself booking | no two bookings overlap |
| Bank overdraft across accounts | sum of checking + savings ≥ 0 | each withdraws from a different account | combined balance never negative |
| Username / unique handle | "no row with this handle exists" | each inserts the same handle | handles are unique |
| Inventory across warehouses | total stock across regions ≥ order | each ships from a different warehouse | never ship more than total stock |
| Allocation against a quota | "budget remaining ≥ request" | each spends against a different line item | total spend ≤ budget |

Every one of these is a write-skew bug under snapshot isolation. The username case is interesting because it is also a *phantom* — the conflicting write is an `INSERT`, so the "shared read" is over a predicate whose result both transactions see as empty. SI hides the phantom from each reader, so both proceed to insert, and you get two users with the same handle (unless a `UNIQUE` constraint saves you — which is exactly the lesson: when the database can express the invariant as a constraint, let it, because a `UNIQUE` index is enforced *outside* the snapshot mechanism).

There are exactly three robust ways to defeat write skew:

1. **Materialize the conflict.** Force the disjoint writes to touch a shared row so SI's write-write detection fires. In the doctors example, both transactions could `UPDATE` a single `shift_summary` row, or `SELECT ... FOR UPDATE` the rows they read so the reads themselves take locks. `SELECT ... FOR UPDATE` is the pragmatic fix: it converts the "read" into a locking read, so the two transactions serialize on the row lock.
2. **Express the invariant as a database constraint.** `UNIQUE`, `EXCLUDE` (Postgres exclusion constraints are purpose-built for the no-overlap-booking case), foreign keys, and `CHECK` constraints are all enforced outside the snapshot, so they catch write skew the snapshot can't see.
3. **Run at `SERIALIZABLE`.** Let the database detect the dangerous read-write structure and abort one transaction. This is the only fully general fix, and it is what the next two sections are about.

## 9. Serializable Snapshot Isolation: detecting the dangerous structure

**Senior rule of thumb: Postgres `SERIALIZABLE` is not two-phase locking. It is snapshot isolation plus a conflict detector that watches for one specific graph pattern — a "dangerous structure" — and aborts a transaction to break it. You pay for it in occasional retries, not in blocking.**

How does a database make snapshot isolation *actually* serializable without throwing away the cheap, non-blocking reads that make SI great? PostgreSQL's answer, shipped in 9.1 (2011), is **Serializable Snapshot Isolation (SSI)**, based on the work of Cahill, Röhm, and Fekete ([*Serializable Isolation for Snapshot Databases*](https://courses.cs.washington.edu/courses/cse444/08au/544M/READING-LIST/fekete-sigmod2008.pdf), SIGMOD 2008). The insight is beautiful: you do not need to *prevent* all conflicts to be serializable; you only need to detect when a set of concurrent transactions has formed a pattern that *could* produce a non-serializable outcome, and abort one of them.

The pattern SSI watches for is built on **read-write antidependencies** (also called `rw`-conflicts). There is an `rw`-antidependency from transaction T1 to T2 when T1 reads a version of some data, and T2 writes a *newer* version of that data — meaning T1 "should have come before" T2 in any serial order, because T1 didn't see T2's write. Fekete et al. proved a theorem that is the entire basis of SSI: **every non-serializable execution under snapshot isolation contains a "dangerous structure" — two consecutive `rw`-antidependency edges, T1 → T2 → T3, where T2 is the "pivot" with both an incoming and an outgoing `rw`-edge** (and, in the full theorem, T3 commits first, or T1 = T3 forming a cycle).

![A graph showing three transactions forming a dangerous structure, with the pivot transaction having two read-write antidependency edges, which SSI aborts](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-8.webp)

The figure shows it. T2 is the pivot: it has an inbound `rw`-edge from T1 (T1 read something T2 then overwrote) and an outbound `rw`-edge to T3 (T2 read something T3 then overwrote). That doubled-up pivot is the fingerprint of a possible serialization-order cycle. Postgres tracks these edges at runtime using lightweight predicate locks called **SIREAD locks** — they do not *block* anything (that's the magic; they are not real locks, just bookkeeping), they merely record "this transaction read this page/tuple/range." When the conflict detector sees a transaction become a dangerous pivot, it aborts one of the participants with our friend `40001`, and the client retries.

In the doctors example, here is what happens at `SERIALIZABLE`:

```sql
-- ===== Postgres: the same doctors schedule at SERIALIZABLE =====

-- Session A
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT count(*) FROM doctors WHERE on_call = true;   -- 2; SSI records the SIREAD
-- Session B
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT count(*) FROM doctors WHERE on_call = true;   -- 2; SSI records its SIREAD
-- Session A
UPDATE doctors SET on_call = false WHERE name = 'alice';
COMMIT;                                               -- commits; creates rw-edge B -> A
-- Session B
UPDATE doctors SET on_call = false WHERE name = 'bob';
COMMIT;
-- ERROR:  could not serialize access due to read/write dependencies among transactions
-- DETAIL:  Reason code: Canceled on identification as a pivot, during commit attempt.
-- HINT:  The transaction might succeed if retried.
```

Session B is aborted as the pivot. The application catches `40001`, retries B from `BEGIN`, and this time B's re-read sees only *one* doctor on call (Alice already left), so B correctly refuses to take Bob off call. The invariant holds. **This is the whole reason `SERIALIZABLE` exists: it is the only level that catches the doctors / booking / quota class of write-skew bugs without you having to manually find every shared read and slap a `FOR UPDATE` on it.**

The properties of SSI are worth stating precisely, because they shape how you design for it:

- **Reads never block.** SIREAD "locks" are bookkeeping, not mutexes. Read concurrency is identical to plain snapshot isolation.
- **You will get false positives.** SSI is *conservative*: it aborts on the dangerous *structure*, which can occur even in some schedules that would actually have been serializable. The abort rate is the price; on low-contention workloads it is typically well under 1%, but on hot rows it can spike.
- **Predicate tracking is granular but imperfect.** SSI tracks reads at the tuple, page, and relation level, escalating to coarser granularity (and thus more false aborts) under memory pressure on the predicate-lock structures (`max_pred_locks_per_transaction`).
- **Every transaction must be retryable.** This is non-negotiable. If your application cannot safely re-run a transaction from the top — because it has side effects outside the database, or because it isn't structured as a retriable unit — you cannot correctly use `SERIALIZABLE`.

## 10. Two-phase locking vs MVCC serializability

**Senior rule of thumb: there are exactly two ways to be serializable — block conflicting transactions (locking) or let them run and abort the losers (MVCC + SSI). The first trades latency for predictability; the second trades retries for throughput. Pick based on your contention, not your taste.**

SSI is the modern, optimistic way to reach serializability. The classical, pessimistic way is **two-phase locking (2PL)**, and it is still what MySQL/InnoDB's `SERIALIZABLE` and SQL Server's default lock-based serializable use. Understanding the contrast is understanding the two fundamentally different philosophies of concurrency control.

![A before-after comparison contrasting two-phase locking's blocking behavior with MVCC and SSI's optimistic versioning and aborts](/imgs/blogs/isolation-levels-and-the-anomalies-they-prevent-9.webp)

**Two-phase locking** has two phases per transaction: a *growing* phase where it acquires locks (shared locks for reads, exclusive locks for writes) and never releases any, and a *shrinking* phase — in *strict* 2PL, this is just the instant of commit/abort — where it releases all of them at once. The "two-phase" rule (acquire-all-before-release-any) is what makes it serializable: it guarantees that conflicting operations from two transactions can never interleave. To make it serializable against *phantoms*, plain row locks are not enough — you need **predicate locks** or, in practice, **gap locks** (InnoDB) / **range locks** that lock the gaps where matching rows could be inserted. This is exactly the next-key locking we saw in InnoDB's `REPEATABLE READ`, turned up to also serialize predicate reads.

The cost of 2PL is **blocking and deadlock**. Readers block writers and writers block readers (shared and exclusive locks conflict). A transaction that holds locks while waiting on a slow client or a network round-trip blocks everyone who needs those rows. And two transactions that acquire locks in opposite orders **deadlock** — the database detects the cycle and aborts a victim. Under high contention, 2PL throughput collapses because everyone is queued behind everyone else; latency becomes a function of the slowest transaction holding a hot lock.

**MVCC + SSI** is the opposite bet. Writers create new versions instead of locking; readers read old versions instead of waiting. Nothing blocks on the read path. Conflicts are *detected*, not prevented, and resolved by aborting the loser at commit time. Under low contention this is dramatically faster than 2PL, because the common case — transactions that don't actually conflict — pays nothing. Under *high* contention it degrades differently than 2PL: instead of latency rising as transactions queue, the *abort rate* rises as more transactions form dangerous structures, and you spend CPU on retries. The same hot row that makes 2PL slow (everyone queued) makes SSI churny (everyone aborting and retrying).

| Dimension | Two-phase locking (InnoDB / SQL Server) | MVCC + SSI (Postgres) |
| --- | --- | --- |
| Philosophy | Pessimistic: prevent conflicts | Optimistic: detect conflicts |
| Reads | Take shared locks; block writers | Read versions; never block |
| Writes | Take exclusive locks; block everyone | Create new versions |
| Phantom defense | Gap / next-key / predicate locks | Predicate (SIREAD) tracking + abort |
| Failure mode | Deadlock; latency under contention | `40001` aborts; retry churn under contention |
| Best when | Long transactions, predictable order, high conflict | Short transactions, low conflict, retryable |
| What you must build | Lock-ordering discipline; deadlock retry | Retry-on-`40001` wrapper around every txn |

Notice the last row: *both* require retry logic. With 2PL you retry on deadlock; with SSI you retry on serialization failure. There is no serializable world without a retry loop. If you are going to run serializable, build the retry wrapper once and wrap every transaction in it:

```python
# A retry wrapper for serialization failures — required for SERIALIZABLE on any engine.
import psycopg2
from psycopg2 import errorcodes
import time, random

def run_serializable(conn, fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            with conn:                       # commits on success, rolls back on exception
                conn.set_session(isolation_level='SERIALIZABLE')
                with conn.cursor() as cur:
                    return fn(cur)           # all the transaction's logic lives in fn
        except psycopg2.errors.SerializationFailure:   # SQLSTATE 40001
            if attempt == max_retries - 1:
                raise
            # exponential backoff with jitter so retriers don't synchronize
            time.sleep((2 ** attempt) * 0.01 + random.random() * 0.01)
        except psycopg2.errors.DeadlockDetected:        # SQLSTATE 40P01 (if using locks)
            if attempt == max_retries - 1:
                raise
            time.sleep((2 ** attempt) * 0.01 + random.random() * 0.01)
    raise RuntimeError("transaction did not converge after retries")
```

The two non-negotiable details: (1) the **entire** transaction body, including the application logic that decides which SQL to run, must be inside `fn` so the retry re-reads fresh data and recomputes — retrying just the failing statement re-uses the stale decision and loops forever; and (2) **backoff with jitter**, or a thundering herd of retriers will keep colliding on the same hot row and never converge.

### Second-order optimization: what the retry tax actually costs

The objection I hear most to `SERIALIZABLE` is "retries will kill my throughput." This is almost always wrong, and it is worth putting numbers on it, because the cost is not a fixed tax — it is a sharp function of contention that is near-zero for most workloads and only explosive for a narrow pathological one.

The abort probability under SSI scales roughly with the probability that two concurrent transactions form a dangerous structure, which in turn scales with how many transactions touch the *same* contended rows in the same window. For a workload where transactions hit mostly-disjoint data — the common case for anything sharded by user, tenant, or entity — the abort rate sits well under 1%, and a transaction that aborts and retries once costs you one extra round trip, not a catastrophe. Concretely: a transaction that normally takes 2 ms, retried once at a 1% abort rate, has an *expected* latency of `2 ms × (1 + 0.01) ≈ 2.02 ms` and an expected `1.01` executions. That is a 1% throughput cost to never debug a write-skew incident. That trade is almost always worth taking.

The cost only explodes when many transactions contend on a *single hot row* — a global counter, a single inventory SKU during a flash sale, one popular event's seat map. There, the abort rate can climb past 50%, every retry re-collides, and throughput collapses. But notice: at that same contention, **2PL doesn't save you** — it just converts the cost from aborts into a serialized queue, and your latency climbs instead of your error rate. The hot row is the problem, not the isolation level. The right fix at that point is to *destroy the contention*, not tune the level:

| Symptom | Wrong fix | Right fix |
| --- | --- | --- |
| `40001` storm on a global counter | lower isolation to `READ COMMITTED` (reintroduces lost update) | sharded counters: N rows summed on read, each incremented at random |
| Flash-sale oversell on one SKU | distributed lock in Redis (second source of truth) | atomic `UPDATE ... SET qty = qty - 1 WHERE qty > 0`; check rowcount |
| Seat-map double-booking | application mutex around the booking service | `UNIQUE`/`EXCLUDE` constraint enforced outside the snapshot |
| Long-running report aborting writers | raise everyone to `SERIALIZABLE` | run the report at snapshot-isolation `REPEATABLE READ` (no aborts, no blocking) |

The general principle: **isolation level is the wrong knob for a contention problem.** It controls *correctness*, and reaching for it to fix *throughput* either reintroduces anomalies (by lowering it) or trades errors for latency (by raising it). When you see a retry storm, the question is not "what level should I use" but "why are so many transactions fighting over the same row, and how do I spread them out." Sharding, batching, queueing, and pushing the invariant into a constraint are all contention fixes; the isolation level is not.

There is also a memory cost specific to Postgres SSI worth budgeting: predicate-lock (SIREAD) bookkeeping lives in shared memory sized by `max_pred_locks_per_transaction × max_connections`. When a long-running serializable transaction reads many rows, its predicate locks can exhaust the per-transaction budget and *escalate* from tuple-granularity to page- and relation-granularity, which dramatically increases false-positive aborts (now any write to the whole page or table conflicts). The tell is a serializable workload whose abort rate climbs as transactions get longer or scan more rows; the fix is to either raise `max_pred_locks_per_transaction` or keep serializable transactions short and narrow. Long, wide reads are the enemy of low abort rates under SSI — another reason to run reporting at plain snapshot isolation and reserve `SERIALIZABLE` for short, surgical, invariant-enforcing writes.

## 11. The defaults you inherited, and what they actually mean

**Senior rule of thumb: write down, for your specific database and version, what your default isolation level forbids — not what its *name* suggests. The name is marketing; the semantics are what bites you.**

You almost certainly did not choose your isolation level. You inherited a default. Here is what the major engines actually give you by default, stripped of the standard's vocabulary:

| Engine | Default level (name) | What it actually is | Forbids | Allows |
| --- | --- | --- | --- | --- |
| PostgreSQL | `READ COMMITTED` | per-statement snapshot | dirty read | read skew, lost update, phantom, write skew |
| PostgreSQL `REPEATABLE READ` | (snapshot isolation) | txn-wide snapshot + first-committer-wins | dirty/read skew, visible phantom, lost update | write skew |
| PostgreSQL `SERIALIZABLE` | SSI | snapshot + dangerous-structure detection | everything (correctly, post-fix) | nothing (you retry instead) |
| MySQL / InnoDB | `REPEATABLE READ` | snapshot reads + next-key locks | dirty/read skew, phantom (on locking reads) | write skew; per Jepsen, more |
| MySQL / InnoDB `SERIALIZABLE` | 2PL (implicit `LOCK IN SHARE MODE`) | locks all reads | most, single-node | RDS replicas violate it |
| Oracle | `READ COMMITTED` | statement-level snapshot | dirty read | read skew, lost update, phantom, write skew |
| Oracle `SERIALIZABLE` | snapshot isolation (!) | txn-wide snapshot, first-updater-wins | dirty/read skew, phantom | **write skew** — Oracle "serializable" is SI, not serializable |
| SQL Server | `READ COMMITTED` (lock-based) | shared locks released after read | dirty read | read skew, lost update, phantom, write skew |
| SQL Server `SNAPSHOT` | MVCC snapshot isolation | txn-wide snapshot | dirty/read skew, phantom | write skew |
| CockroachDB | `SERIALIZABLE` | SSI-style, distributed | everything | nothing (retry) — serializable by default |

Three rows in that table are landmines worth saying out loud. **First:** the two most popular open-source databases ship *different* defaults — Postgres `READ COMMITTED` versus MySQL `REPEATABLE READ` — so code that is anomaly-prone on one is sometimes (accidentally) protected on the other, and a migration between them silently changes your concurrency semantics. **Second:** Oracle's `SERIALIZABLE` is *not serializable* — it is snapshot isolation, and it permits write skew. Decades of Oracle applications carry latent write-skew bugs because the level named `SERIALIZABLE` does not mean what the word means. **Third:** CockroachDB (and Google Spanner, and FoundationDB) default to genuine serializability, having decided the retry cost is worth never debugging a write-skew incident — a defensible and, in my opinion, increasingly correct default for new systems.

The Postgres-vs-MySQL split is worth one more paragraph because so many teams run both. Postgres `REPEATABLE READ` is pure snapshot isolation: cheap reads, write skew allowed, lost updates aborted with `40001`. MySQL `REPEATABLE READ` is snapshot reads *plus* gap locks on locking statements: it blocks phantom inserts (which Postgres SI only hides), but it does so with locks that can deadlock, and — critically — Jepsen's 2023 analysis found that InnoDB's `REPEATABLE READ` does not actually deliver repeatable reads in all cases, exhibiting lost updates, read skew (G-single), and even reads of values that appear "out of thin air" within a single transaction. The lesson is not "MySQL bad, Postgres good"; it is that **the same level name is two different engines' two different bets, and neither is the textbook definition.**

## Case studies from production

The anomalies above are not academic. Here are eight named incidents and analyses where the gap between an isolation level's name and its semantics did real damage, with primary sources.

### 1. PostgreSQL 12.3: "serializable" wasn't, for nine years

In June 2020, Kyle Kingsbury's [Jepsen analysis of PostgreSQL 12.3](https://jepsen.io/analyses/postgresql-12.3) found that Postgres's `SERIALIZABLE` isolation level allowed **G2-item** anomalies — read-write dependency cycles that are forbidden under true serializability — during normal operation. Two concurrent transactions involving inserts and updates could each fail to observe the other's effect, forming a cycle that should have been impossible. The bug was not new: it had been present since SSI was introduced in 2011, affecting at least versions 9.5.22, 10.13, 11.8, 12.3, and 13 beta. Peter Geoghegan traced the root cause to the conflict detector attributing a tuple's update to the *updating* transaction's XID rather than the transaction that *originally created* the tuple, so a real `rw`-antidependency was sometimes missed. The fix flagged the correct transaction ID and added a regression test, shipping in the next minor release. The lesson is humbling: a flagship database's flagship correctness guarantee carried a serializability hole for nearly a decade, and it took a dedicated adversarial test harness to find it. "We implement serializable" is a claim that needs continuous verification, not a one-time proof. Jepsen also confirmed, in the same report, that Postgres `REPEATABLE READ` is in fact snapshot isolation — which, with no SI-violating anomalies observed, was the *more* trustworthy level at the time.

### 2. MySQL 8.0.34: `REPEATABLE READ` that doesn't repeat reads

Jepsen's [December 2023 analysis of MySQL 8.0.34](https://jepsen.io/analyses/mysql-8.0.34) is the most damning isolation report in recent memory. At the default `REPEATABLE READ`, MySQL exhibited **G2-item** (214 cycles in 40 seconds), **G-single / read skew** (244 instances in 60 seconds), and **lost updates** (446 transactions across 198 instances out of ~9,000). Worse, it found **internal consistency violations**: 126 of 9,048 transactions observed values "out of thin air" — one concrete example showed a value changing from "pebble" to "moss" between two reads *within the same transaction*, directly contradicting MySQL's documented promise that "consistent reads within the same transaction read the snapshot established by the first read." MySQL's `REPEATABLE READ` satisfies neither Adya's formal definition nor even the ambiguous ANSI one. The practical takeaway for anyone on MySQL: do not trust the default level's name. If you need repeatable reads, test that you are actually getting them, and assume write skew and lost updates are on the table unless you add explicit locking.

### 3. AWS RDS MySQL: `SERIALIZABLE` violated on replicas

The same Jepsen MySQL report found that while *single-node* MySQL appeared to satisfy serializability, **AWS RDS MySQL clusters routinely violated it at the `SERIALIZABLE` level**, exhibiting G2-item and G-single anomalies including "fractured read"–like behavior where a transaction observes one writer's effect but misses an earlier transitive write. Jepsen attributed this partly to RDS defaulting `replica_preserve_commit_order=OFF`, so replicas could apply commits in an order inconsistent with the primary. The lesson is one every cloud-database user should internalize: **your managed database's isolation guarantees are the guarantees of its *configuration*, not of the engine in the abstract.** A read replica with relaxed commit ordering is a different consistency model than the primary, and "we run serializable" on the primary tells you nothing about what a query routed to a replica will see. Audit your replica flags before you trust a serializable claim.

### 4. The doctors-on-call write skew

The canonical write-skew example — used in the Cahill/Fekete SSI papers, in Kleppmann's *Designing Data-Intensive Applications* Ch. 7, and in countless vendor docs — is the [on-call scheduling scenario](https://www.cockroachlabs.com/docs/stable/demo-serializable): an invariant that at least one doctor stays on call, two doctors each checking "are there at least two of us on call?" and each taking themselves off, both reading a snapshot of "two on call," both proceeding, both committing disjoint writes, leaving zero on call. It is the cleanest possible demonstration that snapshot isolation's consistent *reads* do not imply consistent *writes*. CockroachDB's documentation walks through running this exact workload and shows that under `SERIALIZABLE`, one transaction is allowed to proceed and the other is forced to abort and retry, preserving the invariant. The reason this example is repeated everywhere is that the *shape* — check a shared aggregate, then mutate one of several rows that feed it — is one of the most common shapes in business logic, and almost nobody recognizes it as a serializability hazard until they have shipped the bug.

### 5. Inventory oversell and double-booking

The most expensive write-skew/lost-update bugs in commerce are inventory oversell and double-booking. The pattern, documented across [reservation-system](https://medium.com/@oyebisijemil_41110/preventing-double-booking-in-databases-with-two-phase-locking-9a4538650496) and [omnichannel-retail](https://dev.to/millietechinsights/conquering-race-conditions-in-omnichannel-retail-a-developers-guide-to-inventory-sync-3ehc) post-mortems, is identical every time: multiple checkout transactions read the same stock level (say, 1 unit left), each independently concludes "there is enough," and each decrements and confirms the order. Under `READ COMMITTED` this is a textbook lost update; across warehouses it becomes write skew; with the "is this seat/room free?" check it becomes a phantom-driven double-booking. The fixes the literature converges on map exactly onto our three write-skew defenses: a pessimistic `SELECT ... FOR UPDATE` on the stock row (serialize on the row lock), an optimistic version column with retry, or — for the seat-uniqueness case — a `UNIQUE` constraint that the database enforces outside any snapshot. The recurring failure is teams reaching for application-level coordination (a mutex in the service, a distributed lock in Redis) when the database already had the right primitive; a row lock or a unique index is both simpler and correct, where an external lock is a second source of truth waiting to drift.

### 6. The transactional outbox `40001` storm

A pattern I have personally debugged: a service using the transactional-outbox pattern (write a domain change and an outbox row in one transaction, a separate poller publishes the outbox) ran at `REPEATABLE READ`/`SERIALIZABLE` and started throwing floods of `could not serialize access due to concurrent update` ([SQLSTATE 40001](https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html)) under load. The team's first hypothesis was a database bug; the actual cause was that *every* outbox writer and the poller were contending on a small set of hot rows, and there was no retry wrapper — so serialization failures bubbled up as 500s to users. The fix was two parts: wrap every transaction in a retry-on-`40001` loop with jittered backoff (the wrapper from §10), and reduce contention by sharding the hot rows. The deeper lesson is that **`40001` is not an error; it is the database doing its job.** At any level above `READ COMMITTED`, serialization failures are an expected, normal outcome that the application is *contractually required* to handle by retrying. A serializable database without a retry loop is a serializable database that randomly returns errors to users.

### 7. MariaDB's `--innodb-snapshot-isolation` flag

In response to the 2023 findings that InnoDB's `REPEATABLE READ` allowed lost updates and non-repeatable reads, MariaDB added an opt-in flag, [`--innodb-snapshot-isolation=true`](https://jepsen.io/blog/2024-11-07-mariadb-snapshot-isolation), intended to make `REPEATABLE READ` actually provide snapshot isolation by preventing lost updates, non-repeatable reads, and Monotonic-Atomic-View violations. As of Jepsen's November 2024 note, the flag had not yet been independently validated, but its very existence is the most concrete admission you will find that the default `REPEATABLE READ` was *not* snapshot isolation despite years of users assuming it was. The case study lesson is about defaults and inertia: a level that has shipped weaker than its name for so long cannot simply be strengthened (it would break applications that depend on the current behavior, including the lock semantics), so the fix has to be an opt-in flag — which means the unsafe behavior remains the default, and you have to *know* to turn the safe one on.

### 8. The ledger drift in payments

Engineers who have worked in payments describe the same recurring [isolation-driven ledger bug](https://materializedview.io/p/database-isolation-is-broken-you-should-care): a double-entry ledger where the invariant is "debits equal credits and a balance never goes negative," running at the default `READ COMMITTED`, where two concurrent operations each read a balance, each compute a new one, and a lost update or a read-skew across the debit and credit rows leaves the books out of balance by a small amount. Because the anomaly is intermittent and the per-statement SQL is correct, the database is the *last* thing anyone suspects — engineers chase application logic, retries, and idempotency keys for weeks before someone realizes the schedule, not the statements, is the bug. This is the most insidious property of isolation anomalies and the reason to take them seriously *before* you ship: they do not announce themselves. There is no stack trace. The query plan is fine. The only signal is a slow, intermittent drift in aggregate state that no individual transaction can be blamed for — which is exactly the signature of a non-serializable schedule masquerading as correct code.

### 9. The Postgres-to-MySQL migration that silently changed semantics

A migration I have watched go wrong: a team moved a service from PostgreSQL to MySQL (or the reverse — the direction does not matter) and carried their application code over unchanged, assuming "a transaction is a transaction." What they did not account for is that the two engines ship *different defaults* — Postgres `READ COMMITTED` versus MySQL/InnoDB `REPEATABLE READ` — and that those defaults defend against different anomalies. Code that had been quietly protected from a read-skew bug on MySQL (because InnoDB's transaction-wide snapshot held the read stable) started exhibiting it on Postgres `READ COMMITTED` (where each statement re-snapshots). The reverse migration produced the opposite surprise: code that relied on `READ COMMITTED`'s per-statement freshness — re-reading a value mid-transaction to see other commits — suddenly read stale data on MySQL's snapshot. Worse, the locking behavior differs: InnoDB's gap locks block concurrent inserts into a scanned range, so the migrated code that depended on that blocking lost it on Postgres, where snapshot isolation hides the insert instead of blocking it. The lesson is that **portability of SQL syntax is not portability of concurrency semantics.** Before any cross-engine migration, write down the exact isolation level you depend on, set it *explicitly* on both sides (never rely on the default matching), and re-run your concurrency tests — because the schedule, not the statements, is what changes.

### 10. CockroachDB's deliberate serializable-by-default bet

CockroachDB (and Google Spanner, and FoundationDB) made a decision the rest of the industry is slowly converging toward: [serializable as the default and only "real" isolation level](https://www.cockroachlabs.com/docs/stable/demo-serializable). The reasoning, which their docs and engineering blogs lay out, is that the entire class of write-skew and lost-update bugs — the most expensive and hardest-to-diagnose concurrency bugs — simply *cannot happen* if serializability is the floor, and that the retry cost, paid by a well-built retry wrapper with backoff, is a price worth paying to never debug one of those incidents in production. The catch they are honest about: applications *must* be written to retry serialization failures from day one, because at serializable isolation those failures are routine, not exceptional. This is the inverse of the inherited-default trap from case study 9 — instead of shipping the weakest level by accident, you ship the strongest by design and build the retry discipline into the framework so individual feature code never has to think about it. Having now debugged the alternative several times, I think this is the correct default for any *new* system that can afford the retry loop. The cost of the loop is paid once, in the framework; the cost of a write-skew incident is paid repeatedly, at 2 a.m., by whoever is on call.

## When to reach for serializable, and when not to

### Reach for `SERIALIZABLE` (or explicit locking) when:

- **You enforce a multi-row invariant in application code.** "At least one X," "at most one Y," "the sum of these rows stays ≥ 0," "no two of these overlap." These are the write-skew shapes, and `SERIALIZABLE` is the only general defense. If you find yourself writing `SELECT count(*) ... ; if (ok) UPDATE ...`, you are in this territory.
- **Correctness is worth more than tail latency, and transactions are short and retryable.** Financial ledgers, inventory, scheduling, allocation against a quota. The retry cost is real but bounded; the cost of a silent invariant violation is unbounded.
- **You are building a new system and can afford the retry discipline from day one.** Defaulting new services to `SERIALIZABLE` (the CockroachDB/Spanner philosophy) means you never debug a write-skew incident. Build the retry wrapper once; wrap everything.
- **You cannot enumerate every shared read by hand.** Manually finding every read that needs a `FOR UPDATE` is error-prone and breaks the next time someone adds a query. Letting the database detect dangerous structures is more robust than human auditing.

### Skip `SERIALIZABLE`; use a weaker level plus targeted tools when:

- **The invariant fits a database constraint.** A `UNIQUE` index, a Postgres `EXCLUDE` constraint (perfect for no-overlap bookings), a `CHECK`, or a foreign key is enforced outside the snapshot, costs less than serializable, and never needs a retry. Prefer a constraint to an isolation level every time the database can express the rule.
- **The conflict is a single-row read-modify-write.** Don't reach for serializable to increment a counter — use atomic `SET x = x + 1` or `SELECT ... FOR UPDATE`. It is cheaper and immune to lost update without any retry churn.
- **The workload is read-heavy reporting that just needs a consistent view.** Snapshot-isolation `REPEATABLE READ` gives you a stable, consistent read of the whole database with no blocking and no aborts. Don't pay for write-skew protection on a workload that doesn't write.
- **Contention on hot rows is extreme.** Under pathological contention, `SERIALIZABLE` degrades into a retry storm. Sometimes the right answer is to *remove the contention* — shard the hot row, batch the updates, use a queue — rather than to crank the isolation level and pay the abort tax.
- **Your transactions are not retryable.** If a transaction has external side effects (sends an email, calls a payment API) that cannot be safely re-executed, you cannot correctly run it at `SERIALIZABLE`, because you cannot retry it. Restructure it (move side effects to an outbox processed after commit) before you raise the level.

The honest summary is the one the matrix at the top encodes: **`READ COMMITTED` is too weak for any invariant that spans more than one statement; snapshot isolation closes most of the gap but leaves write skew open; only `SERIALIZABLE` closes everything, and it makes you pay in retries.** The skill is not memorizing which level is "best" — there is no best — it is naming the exact invariants your application must hold, finding their shape in the anomaly matrix, and choosing the cheapest mechanism that forbids exactly those anomalies. Sometimes that's a unique index. Sometimes it's a `FOR UPDATE`. Sometimes it's `SERIALIZABLE` with a retry loop. The wrong answer is the one almost everyone ships by accident: the default, chosen by no one, defending against almost nothing.

## Further reading

- [*A Critique of ANSI SQL Isolation Levels*](https://arxiv.org/abs/cs/0701157) (Berenson, Bernstein, Gray, Melton, O'Neil, O'Neil, 1995) — the paper that broke and rebuilt the standard, and introduced snapshot isolation as a distinct level.
- [*Serializable Isolation for Snapshot Databases*](https://courses.cs.washington.edu/courses/cse444/08au/544M/READING-LIST/fekete-sigmod2008.pdf) (Cahill, Röhm, Fekete, 2008) — the dangerous-structure theorem that Postgres SSI implements.
- *Designing Data-Intensive Applications*, Martin Kleppmann, Chapter 7 (Transactions) — the best plain-English treatment of every anomaly in this article.
- [Jepsen analyses](https://jepsen.io/analyses) of [PostgreSQL 12.3](https://jepsen.io/analyses/postgresql-12.3) and [MySQL 8.0.34](https://jepsen.io/analyses/mysql-8.0.34) — adversarial testing that turned isolation claims into falsifiable, and falsified, statements.
- [PostgreSQL: Serialization Failure Handling](https://www.postgresql.org/docs/current/mvcc-serialization-failure-handling.html) — the official guidance on retrying `40001`.
- Sibling posts on this blog: the [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), [B-trees: how database indexes really work](/blog/software-development/database/b-trees-how-database-indexes-work), and [LSM trees: the write-optimized engine](/blog/software-development/database/lsm-trees-write-optimized-storage-engines).

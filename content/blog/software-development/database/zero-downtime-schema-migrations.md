---
title: "Zero-Downtime Schema Migrations at Scale"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Why a naive ALTER TABLE freezes a busy table, which operations are actually safe in Postgres and MySQL, and how the expand/contract pattern, online DDL tools, and batched backfills let you change a schema under live traffic without a single dropped request."
tags:
  [
    "schema-migration",
    "zero-downtime",
    "postgres",
    "mysql",
    "gh-ost",
    "online-ddl",
    "expand-contract",
    "database",
    "devops",
    "backfill",
    "lock-timeout",
    "system-design",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/zero-downtime-schema-migrations-1.webp"
---

There is a particular flavor of outage that every engineer who has shipped a database-backed product eventually causes exactly once, and then spends the rest of their career making sure they never cause again. It goes like this. You write a migration. It is one line. `ALTER TABLE users ADD COLUMN ...`, or `ALTER TABLE orders ALTER COLUMN total TYPE numeric`, or `CREATE INDEX ON events (created_at)`. You run it in staging against a few thousand rows. It returns in 40 milliseconds. You approve the deploy. The migration runs in production against 180 million rows. Within two minutes, every API endpoint that touches that table is returning 503s, the primary database is pinned at 100% CPU, the connection pool is exhausted, and your on-call channel is on fire. The migration that took 40 ms in staging has been "running" for ten minutes and shows no sign of stopping, and worse, *every other query against that table is hung behind it*. You did not ask for a global lock. You did not know you were taking one. The database took it for you.

This exact incident — a Postgres `ALTER TABLE` on a 180-million-row table that ran fine in staging and "took production down in 2 minutes and kept it down for hours" — is documented over and over in postmortems ([oneuptime metadata-lock writeup](https://oneuptime.com/blog/post/2026-03-31-mysql-metadata-lock-issues-alter-table/view)), and it is so common that an entire category of tooling exists solely to prevent it. The mismatch at the heart of it is this: **the cost of a schema change has almost nothing to do with the length of the SQL statement, and almost everything to do with what lock the statement takes and for how long, and whether that lock forces the engine to rewrite every row in the table.** A one-line statement can be free, or it can rewrite a terabyte and hold an exclusive lock the entire time. The SQL does not tell you which. You have to know.

The figure below is the mental model for the whole problem, and it is worth burning into memory before we go anywhere else. The diagram above — read it left to right — shows the single mechanism behind nearly every schema-change outage: one slow query holds a lightweight lock, the `ALTER` queues behind it waiting for a *heavy* lock, and then **every subsequent query — even plain `SELECT`s that would normally never conflict with each other — queues behind the `ALTER`.** The table does not slowly degrade. It goes from fully healthy to fully frozen the instant the `ALTER` joins the queue, because Postgres grants locks roughly in order and a pending `ACCESS EXCLUSIVE` request blocks everything behind it.

![A single ALTER waiting for a lock queues all subsequent reads and writes behind it](/imgs/blogs/zero-downtime-schema-migrations-1.webp)

The rest of this article is a tour of how to never be the person in that diagram. We will cover, in order: exactly why naive `ALTER TABLE` blocks (the table rewrite, the `ACCESS EXCLUSIVE` lock in Postgres, the metadata lock in MySQL); precisely which operations are safe versus which rewrite or lock in both engines, broken down by version; the **expand/contract pattern** (also called parallel change) that lets you make *any* breaking change safely by never making it breaking; batched backfills that move billions of rows without bloating or locking anything; the online schema-change tools — gh-ost, pt-online-schema-change, pg_repack, Vitess/PlanetScale online DDL — that automate the dangerous parts, and exactly how each one works and where it betrays you; how to coordinate application and schema deploys so the running app is never looking at a schema it does not understand; and the rollout-safety toolkit (`lock_timeout`, `statement_timeout`, low-priority DDL, replica-lag awareness) that turns a potential outage into a harmless automatic retry. At the end there is a copy-paste checklist you can put in your migration review template.

> A schema migration is not a SQL statement. It is a distributed-systems change with a database, an old app, and a new app all running at once — and the database must remain compatible with every app version that is live during the rollout.

## Why a naive ALTER TABLE blocks

**Senior rule of thumb: before you run any DDL, answer two questions — does it rewrite the table, and what lock does it hold while it runs. If you cannot answer both from memory for the exact engine and version in production, do not run it.**

Let us be precise about the two distinct costs hiding inside a schema change, because they are independent and people conflate them constantly.

The first cost is the **table rewrite**. Some schema changes can be satisfied by changing only the catalog — the database's own metadata about what columns exist and what types they have. The data on disk is untouched. These are effectively instant regardless of table size: adding a nullable column with no default, in modern engines, is a metadata-only change. Other schema changes require the engine to physically rewrite every row: changing a column's type from `integer` to `text`, or adding a column with a value that has to be computed per-row, forces the engine to build a new copy of the table with the new physical layout and then swap it in. A rewrite of a 500 GB table moves 500 GB of data, generates 500 GB of WAL (write-ahead log), and takes however long your disk and CPU allow — minutes to hours.

The second cost is the **lock**. Independent of whether it rewrites, every DDL statement acquires some lock on the table while it runs. In Postgres the strongest and most dangerous is `ACCESS EXCLUSIVE`, which conflicts with *every other lock mode including `ACCESS SHARE`* — the lock that a plain `SELECT` takes. While an `ACCESS EXCLUSIVE` lock is held, nothing else can touch the table at all, not even a read. A metadata-only change still takes `ACCESS EXCLUSIVE`, but only for a few milliseconds. A full rewrite takes `ACCESS EXCLUSIVE` for the *entire duration of the rewrite*. That is the combination that kills you: a long rewrite holding the strongest lock.

### The Postgres lock queue, and why a brief lock is still dangerous

Here is the subtlety that catches even experienced engineers. You might reason: "My `ALTER TABLE ADD COLUMN` is metadata-only on Postgres 16, so it only holds `ACCESS EXCLUSIVE` for a millisecond. That is safe." It is *not* necessarily safe, because of how Postgres's lock queue works.

When your `ALTER` requests `ACCESS EXCLUSIVE`, Postgres first checks whether the lock is available. If any other transaction is holding *any* conflicting lock — including a long-running `SELECT` that holds `ACCESS SHARE` — your `ALTER` cannot acquire the lock immediately, so it *waits in a queue*. The poison is in the next step: **while your `ALTER` is waiting at the head of the queue, every new query that wants a conflicting lock also has to queue — behind your `ALTER`.** As [Xata's writeup of the Postgres lock queue](https://xata.io/blog/migrations-and-exclusive-locks) puts it: "Any other statements that require a lock on the users table are now queued behind this ALTER TABLE statement, including other SELECT statements that only require ACCESS SHARE locks." The result is that "the table is effectively blocked for reads and writes until the ALTER TABLE statement completes" — even though the `ALTER` itself would finish in a millisecond *if it could ever start*.

So the sequence that produces the outage in figure 1 is:

1. An analytics query, a slow report, a forgotten `psql` session, or an ORM doing a `SELECT ... FOR UPDATE` holds `ACCESS SHARE` (or stronger) on the table for 90 seconds.
2. Your migration's `ALTER TABLE` requests `ACCESS EXCLUSIVE`, cannot get it, and waits behind the slow query.
3. Every `/login` `SELECT`, every `/checkout` `UPDATE`, every `/signup` `INSERT` that arrives now queues behind your `ALTER`.
4. The connection pool fills with hung connections. New requests cannot get a connection. The whole service returns 503s. The database CPU spikes not from work but from thousands of waiting backends.

This is why "it's metadata-only, so it's fine" is wrong. The fix — which we will return to in depth — is `lock_timeout`: tell the `ALTER` to give up after a couple of seconds rather than wait (and pile everyone up) indefinitely.

Let us look at the actual lock modes. Here is a compact map of the Postgres locks that matter for migrations:

| Lock mode | Taken by | Conflicts with | Migration relevance |
| --- | --- | --- | --- |
| `ACCESS SHARE` | `SELECT` | only `ACCESS EXCLUSIVE` | the lock readers hold; a long one blocks your DDL |
| `ROW EXCLUSIVE` | `INSERT`/`UPDATE`/`DELETE` | `SHARE` and above | writers hold this; conflicts with index builds |
| `SHARE UPDATE EXCLUSIVE` | `CREATE INDEX CONCURRENTLY`, `VALIDATE CONSTRAINT`, `VACUUM` | itself and above | the "safe DDL" lock — lets reads and writes continue |
| `SHARE` | `CREATE INDEX` (non-concurrent) | `ROW EXCLUSIVE` | blocks writes but not reads |
| `ACCESS EXCLUSIVE` | `ALTER TABLE` (most forms), `DROP`, `TRUNCATE`, `VACUUM FULL` | **everything** | the dangerous one; blocks even `SELECT` |

The whole craft of zero-downtime DDL in Postgres reduces to: **stay out of the `ACCESS EXCLUSIVE`-for-a-long-time quadrant, and when you must take `ACCESS EXCLUSIVE` briefly, bound the wait with `lock_timeout`.**

### MySQL's metadata lock

MySQL has the same disease with different terminology. Every statement that touches a table — DML or DDL — acquires a **metadata lock** (MDL) on it. A `SELECT` or `INSERT` takes a *shared* MDL; an `ALTER TABLE` ultimately needs an *exclusive* MDL, at least briefly, even when it runs "online." The exclusive MDL conflicts with all shared MDLs.

The failure mode mirrors Postgres exactly. As the MySQL community documents it, "the operation requires an exclusive metadata lock, which blocks when any transaction holds an open read or write lock on the table." The cascade is: an application transaction stays open (an uncommitted transaction that did a `SELECT` and then went off to call a slow external service), the `ALTER TABLE` waits for the exclusive MDL, and then new queries "get blocked in the MDL pending queue," so "the table becomes completely inaccessible" ([oneuptime](https://oneuptime.com/blog/post/2026-03-31-mysql-metadata-lock-issues-alter-table/view)). Same shape, same outage, different lock name.

The crucial difference is *how* MySQL performs the alter once it has the lock, and that is where the INSTANT / INPLACE / COPY distinction lives — which we will dissect in its own section. For now, the headline is identical to Postgres: a brief lock can still pile up the queue, and a long-held lock plus a rewrite is catastrophic.

## A map of safe and unsafe operations

**Senior rule of thumb: the safety of an operation is a property of the (operation, engine, version) triple, not of the operation alone. `ADD COLUMN` with a default went from "rewrites the whole table" to "instant" in Postgres 11 — the SQL did not change, the safety did.**

The single most useful artifact you can internalize is the matrix below. It answers, for the operations you actually run, whether they rewrite or lock on each engine.

![Operation safety depends on engine and version, not on how the statement reads](/imgs/blogs/zero-downtime-schema-migrations-2.webp)

Read the matrix as the figure does: the same row often has wildly different costs across the two columns. Let us walk every row, because the nuances are exactly the things that bite.

### Adding a column

`ADD COLUMN` with no default is the safest operation in both engines. In Postgres it has always been metadata-only — it adds an entry to the catalog and existing rows simply return `NULL` for the new column. In MySQL 8.0.12+, it uses the `INSTANT` algorithm: a pure data-dictionary change, no data file touched.

`ADD COLUMN` *with a default* is where versions matter enormously. Before Postgres 11, adding a column with a default value rewrote the entire table — every existing row was physically updated to store the default. On a billion-row table that is a multi-hour `ACCESS EXCLUSIVE` lock and an outage. **Postgres 11 changed this**: a column with a *constant* (non-volatile) default no longer triggers a rewrite. As the Postgres 11 docs put it, "adding a column with a constant default value no longer means that each row of the table needs to be updated when the ALTER TABLE statement is executed. Instead, the default value will be returned the next time the row is accessed." The mechanism is two new catalog columns, `atthasmissing` and `attmissingval`, in `pg_attribute`: the default is stored once in the catalog and synthesized for old rows at read time ([EnterpriseDB on PG11 fast defaults](https://www.enterprisedb.com/blog/adding-new-table-columns-default-values-postgresql-11)).

The trap is the word *constant*. If the default is **volatile** — `clock_timestamp()`, `gen_random_uuid()`, `random()` — Postgres cannot store a single value that applies to all rows, so it falls back to a full rewrite: "if the default value is volatile (e.g. `clock_timestamp()`) each row will need to be updated with the value calculated at the time ALTER TABLE is executed." So:

```sql
-- Postgres 11+: SAFE. Constant default, no rewrite, ACCESS EXCLUSIVE held ~1ms.
ALTER TABLE users ADD COLUMN status text NOT NULL DEFAULT 'active';

-- Postgres any version: UNSAFE. Volatile default forces a full table rewrite
-- under ACCESS EXCLUSIVE for the entire rewrite. On a large table, an outage.
ALTER TABLE users ADD COLUMN external_id uuid NOT NULL DEFAULT gen_random_uuid();
```

The safe rewrite for the volatile case is the expand/contract approach: add the column nullable with no default, then backfill in batches, then add the default and constraint. We will build that out.

### Setting NOT NULL

This is the most surprisingly dangerous "small" operation in Postgres. The naive form scans the entire table under an `ACCESS EXCLUSIVE` lock to verify no existing row is `NULL`:

```sql
-- UNSAFE: full table scan under ACCESS EXCLUSIVE. Blocks everything for the scan.
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
```

The safe rewrite exploits a Postgres 12+ optimization. You first add a `CHECK` constraint marked `NOT VALID`, which records the constraint *without scanning the table* (it only enforces the constraint on new and modified rows, and takes a brief lock). Then you `VALIDATE` it in a separate statement, which scans the table under the much weaker `SHARE UPDATE EXCLUSIVE` lock — compatible with reads and writes. Finally you `SET NOT NULL`, and Postgres 12+ will *skip the scan entirely* because the validated check constraint already proves the column is non-null:

```sql
-- 1. Record the constraint without scanning. Brief ACCESS EXCLUSIVE only.
ALTER TABLE users ADD CONSTRAINT email_not_null
  CHECK (email IS NOT NULL) NOT VALID;

-- 2. Validate it. SHARE UPDATE EXCLUSIVE — reads and writes continue.
ALTER TABLE users VALIDATE CONSTRAINT email_not_null;

-- 3. Set NOT NULL. PG12+ sees the valid constraint and SKIPS the scan.
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- 4. (Optional, separate statement) drop the now-redundant check constraint.
ALTER TABLE users DROP CONSTRAINT email_not_null;
```

The reason `VALIDATE CONSTRAINT` is safe is precisely the lock level: it "requires a much less restrictive lock SHARE UPDATE EXCLUSIVE, which is compatible with ROW EXCLUSIVE locks required by insert/update/delete operations, so normal database operations are not blocked" ([the SET NOT NULL trap](https://dev.to/andrewpsy/the-set-not-null-downtime-trap-in-postgresql-1o71)). Engineering teams have built this pattern into their migration helpers specifically because the naive `SET NOT NULL` keeps causing incidents — see [Doctolib's writeup on adding NOT NULL with minimal locking](https://medium.com/doctolib-engineering/adding-a-not-null-constraint-on-pg-faster-with-minimal-locking-38b2c00c4d1c) and the [GitLab issue tracking it](https://gitlab.com/gitlab-org/gitlab/issues/38060). Do step 4 as a *separate* statement, never combined with step 3, or you reintroduce a longer lock.

### Creating an index

In Postgres, a plain `CREATE INDEX` takes a `SHARE` lock, which blocks all writes (but not reads) for the entire build — on a large table that is minutes of write downtime. The safe form is `CREATE INDEX CONCURRENTLY`, which we dedicate a full section to below. In MySQL 8.0, adding a secondary index uses `INPLACE` with `LOCK=NONE`, permitting concurrent reads and writes — it is genuinely online out of the box.

### Dropping a column, and changing a type

`DROP COLUMN` is metadata-only in both engines (Postgres marks the column dropped in the catalog and reclaims space lazily; MySQL 8.0.29+ supports `INSTANT` drop). But changing a column's *type* in a way that changes the on-disk representation — `integer` → `bigint`, `varchar(50)` → `text` when it forces a rewrite, `numeric` precision changes — forces a full rewrite under the strongest lock in both engines. The safe approach for a type change is *always* expand/contract: add a new column of the new type, dual-write, backfill, switch reads, drop the old column. There is no shortcut; a type change that rewrites is unsafe by definition on a large hot table.

Here is the same information as a prescriptive table you can paste into a review template:

| Operation | Postgres safe form | MySQL safe form | Never do this on a large hot table |
| --- | --- | --- | --- |
| Add nullable column | `ADD COLUMN` (instant) | `ADD COLUMN` (INSTANT) | — (already safe) |
| Add column w/ default | constant default, PG11+ | INSTANT, 8.0.12+ | volatile default (rewrites) |
| NOT NULL | `CHECK NOT VALID` → `VALIDATE` → `SET NOT NULL` | add check then validate | bare `SET NOT NULL` (scans) |
| Add index | `CREATE INDEX CONCURRENTLY` | `ALGORITHM=INPLACE, LOCK=NONE` | bare `CREATE INDEX` (locks writes) |
| Drop column | `DROP COLUMN` (instant) | `DROP COLUMN` INSTANT (8.0.29+) | — |
| Change type | new column + backfill | new column + backfill | in-place `ALTER ... TYPE` (rewrites) |
| Add FK constraint | `ADD ... NOT VALID` → `VALIDATE` | online with caveats | bare `ADD CONSTRAINT` (scans + locks) |
| Rename column | expand/contract (new col) | expand/contract (new col) | bare `RENAME` (breaks running app) |

The last row deserves emphasis: `RENAME COLUMN` is *catalog-instant* — it takes a trivial lock and finishes immediately. It is "unsafe" not because of locks but because it breaks the running application, which still references the old name. This is the bridge to the most important pattern in this entire article.

## The expand/contract pattern (parallel change)

**Senior rule of thumb: never make a breaking change. Make a sequence of backward-compatible changes that, composed, achieve the breaking change — and make sure the database is compatible with every app version that is simultaneously live during the rollout.**

The reason a column rename causes an outage has nothing to do with the database. `ALTER TABLE ... RENAME COLUMN` is instant. The outage happens because in any non-trivial deployment, the old app code and the new app code run *at the same time* — during a rolling deploy, during a canary, during a rollback. If you rename `email` to `email_address` in one migration, then for the window where old pods still reference `email`, every query they run throws "column email does not exist." You have a partial outage scoped to however long the rollout takes, plus the entire duration of any rollback.

The pattern that dissolves this is **expand/contract**, also called **parallel change**, formalized by Martin Fowler: break a backward-incompatible change "into three distinct phases: expand, migrate, and contract" — expand adds the new design without breaking the old, migrate switches over, contract removes the old design ([parallel change](https://blog.jakubholy.net/wiki/development/parallel-design-parallel-change/)). For database schema specifically, the canonical expansion is six reversible phases. The figure below is the timeline, and the invariant it encodes is the whole point: **at every phase boundary, the database schema is compatible with both the previous and the next app version.**

![Expand and contract splits one breaking change into six independently reversible phases](/imgs/blogs/zero-downtime-schema-migrations-3.webp)

Let us work the canonical example — renaming `users.email` to `users.email_address` — through all six phases, because once you have done a rename this way you can do any change this way.

### Phase 1 — Expand: add the new structure

Add the new column, nullable, with no default. This is metadata-only and instant. The old app does not know it exists; the new column simply sits empty.

```sql
-- Instant, ACCESS EXCLUSIVE held ~1ms. Old app unaffected.
ALTER TABLE users ADD COLUMN email_address text;
```

The schema is now compatible with the old app (which uses `email`) and with the upcoming app (which will use `email_address`). Reversible: just `DROP COLUMN email_address`.

### Phase 2 — Dual-write: keep both columns in sync

Deploy app code that writes to *both* columns on every insert and update. This is the most important phase to get right, because from here forward both columns must stay consistent for any row touched. There are two ways to implement dual-write: in application code, or with a database trigger. Application code is cleaner and easier to reason about; triggers guarantee no write path is missed but add hot-path overhead. For a rename, application dual-write looks like:

```python
# New app version: write both columns. Read still uses the OLD column.
def update_email(user_id: int, new_email: str) -> None:
    db.execute(
        """
        UPDATE users
        SET email = %(e)s, email_address = %(e)s
        WHERE id = %(id)s
        """,
        {"e": new_email, "id": user_id},
    )
```

If you prefer the trigger approach (useful when many code paths write the column and you cannot find them all), a Postgres trigger keeps the columns in lockstep without touching every query site:

```sql
CREATE OR REPLACE FUNCTION sync_email() RETURNS trigger AS $$
BEGIN
  -- Mirror whichever column the writer set into the other.
  IF NEW.email_address IS DISTINCT FROM OLD.email_address THEN
    NEW.email := NEW.email_address;
  ELSE
    NEW.email_address := NEW.email;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sync_email
  BEFORE INSERT OR UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION sync_email();
```

Reversible: roll back the app deploy; the trigger is harmless. New rows are now correct in both columns; old rows are still stale in `email_address`.

### Phase 3 — Backfill: copy the historical data

Existing rows still have `NULL` in `email_address`. Backfill them in batches (the next section is entirely about doing this safely). The key property: the backfill is idempotent and restartable, because dual-write guarantees that any row written *after* dual-write started is already correct, and the backfill only needs to fix rows untouched since then.

```sql
-- Run in a loop, small batches, with throttling (see next section).
UPDATE users
SET email_address = email
WHERE email_address IS NULL
  AND id BETWEEN :lo AND :hi;
```

Reversible: the backfill only writes the new column; nothing depends on it yet.

### Phase 4 — Switch reads: read the new column, verify first

Deploy app code that *reads* from `email_address` while still dual-writing. Critically, before flipping reads in production, run a **dark-read comparison**: read both columns and compare, alerting on any mismatch, without using the new value for anything user-facing. This is exactly the technique Stripe used for its subscriptions migration with GitHub's [Scientist library](https://stripe.com/blog/online-migrations) — "a Ruby library that allows you to run experiments and compare the results of two different code paths, alerting you if two expressions ever yield different results in production." Only after the comparison is clean for a soak period do you make `email_address` the source of truth.

```python
# Phase 4: read new, write both, compare in the background.
def get_email(user_id: int) -> str:
    row = db.fetch_one("SELECT email, email_address FROM users WHERE id=%s", user_id)
    if row["email"] != row["email_address"]:
        metrics.increment("email_mismatch")          # alert if this ever fires
    return row["email_address"]                       # new column is now the read source
```

Reversible: redeploy the previous version to read `email` again.

### Phase 5 — Stop writing the old column

Deploy app code that writes only `email_address`. The old `email` column is now dead weight but still present, so a rollback to phase 4 is still possible. Let this soak.

### Phase 6 — Contract: drop the old structure

After a soak period long enough that you are confident you will not roll back (often a day or a week, depending on your release cadence), drop the old column and the sync trigger. This is metadata-only and instant.

```sql
DROP TRIGGER IF EXISTS trg_sync_email ON users;
DROP FUNCTION IF EXISTS sync_email();
ALTER TABLE users DROP COLUMN email;     -- instant, metadata-only
```

The single most common failure of this pattern is not technical: **the contract phase never happens.** The expand shipped, the migrate happened, the team moved on, and a year later you have a half-migrated schema with dead columns and triggers nobody remembers. Track contract phases as real tickets with owners, or they rot into permanent technical debt.

### A harder example: splitting a table

The same six phases handle structurally larger changes. Suppose `subscriptions` started life as a JSON array column on `customers` and you want to extract it into a proper `subscriptions` table — exactly the migration Stripe describes, moving "hundreds of millions of Subscription objects" out of the customer record into a dedicated table. The phases map directly: **expand** = create the `subscriptions` table; **dual-write** = on every subscription change, write both the JSON array and the new table; **backfill** = a parallelized offline job (Stripe used Hadoop/MapReduce rather than "expensive queries on production databases") that explodes each customer's array into rows; **switch reads** = read from the new table with Scientist comparisons; **stop old writes**; **contract** = drop the JSON column. The discipline is identical; only the backfill is heavier.

> The expand/contract pattern is the database equivalent of feature flags: it converts a risky atomic switch into a sequence of independently deployable, independently reversible steps, each safe in isolation.

## Batched backfills that don't bloat or lock

**Senior rule of thumb: a backfill is not one big UPDATE. It is thousands of small, committed, throttled UPDATEs that each touch a bounded number of rows, watch replica lag, and can resume from where they stopped.**

The single most common way teams turn a safe migration into an outage is the backfill. They write `UPDATE users SET email_address = email WHERE email_address IS NULL;` and run it against 200 million rows in one transaction. Three things go wrong at once. First, that single `UPDATE` holds row locks on every row it touches for the *entire duration* — minutes — blocking concurrent writes to those rows. Second, in Postgres's MVCC model every updated row becomes a dead tuple (the old version) that `VACUUM` must later reclaim, so a 200-million-row update doubles the table's physical size and creates a massive bloat-and-vacuum burden ([MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) covers why). Third, the giant transaction generates a giant WAL burst that floods replication and spikes replica lag, which can cascade into read-replica unavailability.

The figure below is the loop that avoids all three. The essential properties: small committed batches release locks frequently, bounded batch size caps dead-tuple generation per transaction, and the throttle check between batches keeps replica lag and primary load within budget.

![Backfilling in keyset batches with sleeps avoids long locks bloat and replica lag](/imgs/blogs/zero-downtime-schema-migrations-6.webp)

Here is a production-grade backfill in Python. The two non-obvious correctness details are **keyset pagination** (never `OFFSET`, which re-scans) and the **throttle** that watches replica lag:

```python
import time
import psycopg2

BATCH = 1_000          # rows per transaction — small enough to commit fast
SLEEP = 0.05           # seconds between batches — gives replicas time to keep up
MAX_LAG_BYTES = 50 * 1024 * 1024   # pause if a replica falls > 50 MB behind

def replica_lag_bytes(cur) -> int:
    cur.execute("""
        SELECT COALESCE(MAX(
            pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)
        ), 0) FROM pg_stat_replication
    """)
    return cur.fetchone()[0] or 0

def backfill(conn) -> None:
    last_id = 0
    while True:
        with conn:                         # commits at the end of the block
            cur = conn.cursor()
            # Keyset pagination: WHERE id > last_id ORDER BY id LIMIT BATCH.
            # This uses the primary-key index every time — no OFFSET re-scan.
            cur.execute("""
                WITH batch AS (
                    SELECT id FROM users
                    WHERE id > %s AND email_address IS NULL
                    ORDER BY id
                    LIMIT %s
                )
                UPDATE users u
                SET email_address = u.email
                FROM batch
                WHERE u.id = batch.id
                RETURNING u.id
            """, (last_id, BATCH))
            rows = cur.fetchall()
            if not rows:
                break                       # done: no rows left to fix
            last_id = max(r[0] for r in rows)

        # Throttle: back off while any replica is behind.
        with conn.cursor() as c:
            while replica_lag_bytes(c) > MAX_LAG_BYTES:
                time.sleep(0.5)
        time.sleep(SLEEP)
```

A few things make this safe that are easy to get wrong:

- **Keyset, not OFFSET.** `LIMIT ... OFFSET n` re-scans the first `n` rows every batch, making the backfill O(n²). `WHERE id > last_id` uses the index and is O(n).
- **Commit per batch.** The `with conn:` block commits each batch, releasing row locks immediately and bounding dead-tuple generation. A `VACUUM` (or autovacuum) can then reclaim space *during* the backfill instead of after.
- **Throttle on lag, not just sleep.** A fixed sleep is a guess; checking actual replica lag adapts to real conditions. GitLab's [batched background migrations](https://docs.gitlab.com/development/migration_style_guide/) framework does exactly this, running batches as background jobs that "should not change the schema" and that pause under load.
- **Idempotent and resumable.** The `WHERE email_address IS NULL` predicate means re-running the backfill is harmless — already-filled rows are skipped. If the job crashes, restart it; it picks up where it left off.

For MySQL the structure is identical; you watch `Seconds_Behind_Master` (or the more precise `pt-heartbeat` lag) instead of WAL LSN diff, and you commit each chunk. The Percona toolkit and gh-ost both implement exactly this chunked-copy-with-throttle internally, which is the next topic.

### Second-order optimization: avoid the post-backfill VACUUM cliff

Even with per-batch commits, a large Postgres backfill leaves dead tuples behind that autovacuum must clean up. If autovacuum is tuned for steady-state, it can fall behind during a heavy backfill and never catch up, leaving the table permanently bloated. Two mitigations: temporarily raise `autovacuum_vacuum_cost_limit` (or lower `autovacuum_vacuum_cost_delay`) for the table during the backfill so autovacuum works harder, and run a manual `VACUUM (ANALYZE)` on the table after the backfill completes to reclaim space and refresh planner statistics. Do *not* run `VACUUM FULL` — it takes `ACCESS EXCLUSIVE` and rewrites the table, recreating the very outage you have been avoiding. Use `pg_repack` (below) if you genuinely need to reclaim the disk online.

## CREATE INDEX CONCURRENTLY, in depth

**Senior rule of thumb: in production Postgres, `CREATE INDEX` without `CONCURRENTLY` is an outage waiting to happen on any table that takes writes. Always concurrent, always with a plan for the INVALID-index failure case.**

Indexing a large table is the most common single DDL operation, and Postgres makes the safe path opt-in. A plain `CREATE INDEX` takes a `SHARE` lock that blocks all writes for the entire build. `CREATE INDEX CONCURRENTLY` instead takes only `SHARE UPDATE EXCLUSIVE`, which permits concurrent reads *and* writes — at the cost of being slower and doing two passes over the table. The figure below shows why it needs two passes.

![Concurrent index builds run two table scans while keeping the table open to writes](/imgs/blogs/zero-downtime-schema-migrations-5.webp)

The two-pass dance is the clever part. In **pass 1**, Postgres builds the index against a snapshot of the table taken at the start. But because writes continue during the build, rows inserted or updated after the snapshot are not in the index yet. So Postgres **waits out all transactions that were open when pass 1 finished** (they might still be using the old snapshot), then does **pass 2**: a second scan that catches every row written since the snapshot. Only after pass 2 succeeds does it mark the index valid and ready for query planning.

```sql
-- Safe: SHARE UPDATE EXCLUSIVE, reads and writes continue. Two passes.
CREATE INDEX CONCURRENTLY idx_events_created_at ON events (created_at);
```

The catch — and it is a real operational footgun — is the failure case. If a `CREATE INDEX CONCURRENTLY` fails (a deadlock with a long transaction, a unique-constraint violation discovered in pass 2, the connection dropping), it **leaves behind an INVALID index** that is not used for queries but *is* maintained on every write (slowing writes) and consumes disk. You must detect and clean these up:

```sql
-- Find leftover invalid indexes after a failed concurrent build.
SELECT c.relname AS index_name
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
WHERE i.indisvalid = false;

-- Drop the invalid index, also concurrently, then retry the build.
DROP INDEX CONCURRENTLY idx_events_created_at;
```

Two more gotchas: `CREATE INDEX CONCURRENTLY` *cannot run inside a transaction block*, which means most migration frameworks need a special "no transaction" mode for it (Rails: `disable_ddl_transaction!`; many frameworks have an equivalent). And on **partitioned tables**, you cannot create a concurrent index on the parent in one shot — you create it concurrently on each partition, then attach. Plan for both.

For MySQL, the equivalent is built in: `ALTER TABLE ... ADD INDEX, ALGORITHM=INPLACE, LOCK=NONE` builds a secondary index online with concurrent DML allowed. There is no two-pass INVALID-index hazard to manage; the engine handles the catch-up internally. The tradeoff lands in the MySQL algorithm picker, which is the next section.

## MySQL's INSTANT / INPLACE / COPY algorithms

**Senior rule of thumb: never run a MySQL `ALTER` in production without knowing which algorithm it will use. Pin it with `ALGORITHM=...` so MySQL *refuses* to run if it would silently fall back to a full COPY.**

MySQL 8.0 has three algorithms for `ALTER TABLE`, and the difference between them is the difference between a millisecond and an all-night outage. The figure below lays them side by side on the dimensions that matter operationally.

![MySQL picks INSTANT then INPLACE then COPY each with very different lock and cost](/imgs/blogs/zero-downtime-schema-migrations-8.webp)

- **INSTANT** (introduced in MySQL 8.0.12, contributed by the Tencent DBA team) "performs only metadata changes in the data dictionary. It doesn't acquire any metadata lock during schema changes and as it doesn't touch the data file of the table." This is the dream: adding a column, dropping a column (8.0.29+), renaming a column — milliseconds, no rewrite, no real lock. ([Mydbops on DDL algorithms](https://www.mydbops.com/blog/ddl-algorithms-in-mysql))
- **INPLACE** (5.6+) rebuilds the table or index *in place* where possible, avoiding a full copy. It permits concurrent DML with `LOCK=NONE`. It takes a brief exclusive metadata lock at the start and end of the operation but allows reads and writes in between. This is how online secondary-index creation works.
- **COPY** (the pre-5.6 default behavior, and the fallback) builds an entire new copy of the table, applies the change, and swaps. It blocks concurrent writes (`LOCK=SHARED` permits reads only) and holds locks for the full duration. On a 500-million-row table, this is hours of degraded service.

The dangerous part is the *automatic fallback*. As the docs describe, "if ALGORITHM is not specified, the server will first try the DEFAULT=INSTANT algorithm for all column addition. If it can not be done, then the server will try INPLACE algorithm; and if that can not be supported, at last server will finally try COPY algorithm." So an `ALTER` you *believed* was instant can silently degrade to a full COPY because of some detail of your table (a generated column, a full-text index, a specific type change), and you find out only when production seizes.

The fix is to pin the algorithm so MySQL errors out rather than silently falling back:

```sql
-- Refuses to run if it can't be done as a pure metadata change.
ALTER TABLE orders ADD COLUMN shipped_at DATETIME, ALGORITHM=INSTANT;

-- Refuses to run if it would require a full table copy.
ALTER TABLE orders ADD INDEX idx_status (status), ALGORITHM=INPLACE, LOCK=NONE;
```

If the first statement fails with "ALGORITHM=INSTANT is not supported for this operation," that is a *good* failure — MySQL just told you the operation is not free, before you took an outage. You then either find an online-compatible form or reach for gh-ost.

The MySQL online-DDL reference ([dev.mysql.com 8.0](https://dev.mysql.com/doc/refman/8.0/en/innodb-online-ddl-operations.html)) has a per-operation table of which algorithms each operation supports; treat it as authoritative and consult it for the *exact* version in production, because the support matrix shifts between point releases (instant `DROP COLUMN` arrived in 8.0.29, for example).

## Online schema-change tools

When the engine cannot do an operation online — a type change, a column reorder, an `ALTER` that falls back to COPY — you reach for an external tool that performs the change on a copy of the table while keeping the original live, then swaps. There are two architectural families, and the difference between them is the source of most of their tradeoffs.

### gh-ost: triggerless, binlog-based

GitHub built [gh-ost](https://github.blog/news-insights/company-news/gh-ost-github-s-online-migration-tool-for-mysql/) because the trigger-based tools had problems at GitHub's write volume. The figure below is gh-ost's architecture.

![gh-ost copies rows and replays binlog events into a ghost table before an atomic swap](/imgs/blogs/zero-downtime-schema-migrations-4.webp)

gh-ost creates a **ghost table** (`_users_gho`) with the desired new schema, empty. It then does two things concurrently. It **copies existing rows** from the original table in throttled chunks. And — this is the key innovation — instead of putting triggers on the original table to capture live writes, it **tails the binary log** (in Row-Based Replication format) and replays the captured INSERT/UPDATE/DELETE events into the ghost table asynchronously. When the ghost table has caught up and is in sync, gh-ost performs an **atomic cut-over**: a single `RENAME` that swaps the ghost table in for the original.

Why triggerless matters: trigger-based tools add "overhead of a parser and interpreter to each query," create "lock contention" between the original query and its triggers, and "cannot be paused without risking data loss." By reading the binlog instead, gh-ost decouples the migration's write load from production traffic entirely — "the master only observes a single connection that is sequentially writing to the ghost table." This gives gh-ost two superpowers: it can be **throttled or fully paused** at any time (via load thresholds like `Threads_running=30`, replication-lag heartbeats, or even an interactive `echo throttle | socat - /tmp/gh-ost.sock`), and it can run its copy load against a **replica**, testing the migration with `--test-on-replica` before ever touching the master. GitHub runs it "daily, as engineering requests come, sometimes multiple times a day," on tables "as small as empty and as large as many hundreds of GB."

```bash
# A real gh-ost invocation: migrate `users`, run the copy load against a
# replica, throttle when threads_running gets high, and require explicit cut-over.
gh-ost \
  --host=replica.db.internal \
  --database="app" --table="users" \
  --alter="ADD COLUMN last_seen_at DATETIME NULL" \
  --max-load="Threads_running=40" \
  --critical-load="Threads_running=120" \
  --chunk-size=1000 \
  --max-lag-millis=1500 \
  --allow-on-master \
  --postpone-cut-over-flag-file=/tmp/ghost.postpone \
  --execute
```

The tradeoffs: gh-ost requires RBR binlogs, it needs roughly **double the disk space** during the copy (the ghost table is a full second copy), and the atomic `RENAME` cut-over briefly takes a metadata lock — so even gh-ost's "zero downtime" has a sub-second blocking window at the very end, which you want to schedule and guard with a `lock_wait_timeout`.

### pt-online-schema-change: trigger-based

Percona's [pt-online-schema-change](https://docs.percona.com/percona-toolkit/pt-online-schema-change.html) (pt-osc) is the older, trigger-based approach. It creates a new table with the desired schema, **adds AFTER INSERT / UPDATE / DELETE triggers** on the original table that replicate every change into the new table, copies existing data in chunks, then does an atomic `RENAME` swap. It is battle-tested and ubiquitous.

Its tradeoffs come directly from the triggers. It "will not work if any triggers are already defined on the table" (MySQL 5.7 and earlier allowed only one trigger per event). The triggers run *synchronously on the application's write path*, so they add latency to every write during the migration — and crucially, "the use of triggers means" the migration write load *is* coupled to production write load, so it cannot be cleanly paused. **Foreign keys** are a serious complication: "the technique of atomically renaming the original and new tables does not work when foreign keys refer to the table," forcing awkward workarounds. And it should never be run directly on a replica, because "the statements coming via binlog will not be processed by triggers, so whatever new data is coming in via replication will be missing in the new table." Like gh-ost, it needs roughly double the disk space.

The decision between them is usually: gh-ost if you have RBR binlogs and want pausability and replica-testing (the GitHub-scale answer); pt-osc if you are on statement-based replication, have no binlog access, or are already deep in the Percona toolkit.

### Postgres: pg_repack

Postgres has no exact gh-ost equivalent for arbitrary `ALTER`s (its DDL is more often online natively), but for the specific job of **rebuilding a bloated table or index online**, [pg_repack](https://github.com/reorg/pg_repack) is the standard. It works much like pt-osc: it creates a log table and a trigger to capture concurrent changes, bulk-copies the live data into a fresh compact table, applies the logged changes to catch up, and then "briefly acquires an exclusive lock to swap table names." Throughout most of the process it "only uses ACCESS SHARE lock on the original table," so reads and writes continue; the `ACCESS EXCLUSIVE` is held only for the final instantaneous swap. Use it instead of `VACUUM FULL` whenever you need to reclaim space without an outage.

```bash
# Rebuild a bloated table online; ACCESS EXCLUSIVE only for the final swap.
pg_repack --table=public.events --no-order -d app_production
```

### Vitess / PlanetScale online DDL, and the revert superpower

[PlanetScale](https://planetscale.com/blog/safely-making-database-schema-changes) (built on Vitess) productizes online DDL into a workflow: a schema change is a **deploy request**, reviewed like a pull request, executed as a Vitess online migration that "creates an empty shadow table, implements the change, copies data, tracks incoming changes, and replaces the original." The genuinely novel capability is **instant, lossless revert**. Vitess's VReplication "has a transactionally accurate journal of the state of migration. At any time, it knows exactly which rows have been copied and which changelog events have been processed," which lets you "revert a deployment for up to 30 minutes after deploying" while *keeping the data written to the original schema during that window* ([behind the scenes: schema reverts](https://planetscale.com/blog/behind-the-scenes-how-schema-reverts-work)). Most online-DDL tools throw the old table away after cut-over; Vitess keeps the reverse path live, which is why a fat-fingered migration can be undone in seconds rather than re-migrated. (The catch: instant-deployed changes cannot be reverted, and the window is 30 minutes.)

Here is the whole tool landscape as a decision table:

| Tool | Engine | Mechanism | Pausable | Replica test | Revert | Big caveat |
| --- | --- | --- | --- | --- | --- | --- |
| Native online DDL | MySQL 8.0 | INSTANT / INPLACE | n/a | n/a | no | silent COPY fallback |
| `... CONCURRENTLY` / `NOT VALID` | Postgres | engine-native | n/a | n/a | no | INVALID-index cleanup |
| gh-ost | MySQL | binlog tail + ghost table | **yes** | **yes** | no | needs RBR, 2× disk |
| pt-online-schema-change | MySQL | triggers + shadow table | no | no | no | FK pain, trigger overhead |
| pg_repack | Postgres | trigger + log table | no | no | no | bloat-rebuild only, not arbitrary ALTER |
| Vitess / PlanetScale | MySQL/Vitess | VReplication shadow | yes | yes | **yes (30 min)** | platform lock-in |

## Coordinating app and schema deploys

**Senior rule of thumb: the schema change must be deployed such that the currently-running application is compatible with it. Expand before you deploy code that needs the new structure; contract only after you deploy code that no longer needs the old structure.**

The dual-write timeline below makes the ordering constraint concrete. The reason Stripe's four-step pattern (dual write → switch reads → switch writes → remove old) works is that at no point is any live app version asked to use a schema it does not understand.

![Dual writing then a verified read switch lets the cutover happen with no flag day](/imgs/blogs/zero-downtime-schema-migrations-7.webp)

The general ordering rules, derived from the "schema must be compatible with the running app" invariant:

- **Additive schema changes go *before* the code that uses them.** Add the column, then deploy code that reads/writes it. If you deploy the code first, it references a column that does not exist yet → errors during the window.
- **Destructive schema changes go *after* the code that stops using them.** Deploy code that no longer references the column, let it fully roll out and soak, *then* drop the column. If you drop first, old pods still mid-rollout reference a gone column → errors.
- **Renames are never single-step** — they are an additive change (expand) followed much later by a destructive change (contract), with dual-write in between. There is no safe atomic rename in a rolling-deploy world.
- **A migration must be backward-compatible with the version of the app it is deployed alongside, AND with the previous version** (for rollback). This is the rule that makes people uncomfortable, because it means you frequently cannot do the "obvious" thing in one PR. Backward compatibility with the previous version is what makes a rollback safe; skip it and a rollback becomes a second outage.

This is exactly why tooling like the [strong_migrations](https://github.com/ankane/strong_migrations) gem exists: it "scans your migrations for operations that can block or lock your database," refuses the dangerous ones in development, and prints the safe rewrite. It encodes precisely the rules in this article — "adding a column with a non-null default causes the entire table to be rewritten," "in Postgres, adding an index non-concurrently locks the table," "backfilling in the same transaction that alters a table locks the table for the duration of the backfill" — and forces you to do the multi-step safe version instead. Even if you are not on Rails, the *list* of operations strong_migrations flags is a ready-made review checklist. GitLab's [migration helpers](https://docs.gitlab.com/development/migration_style_guide/) (`rename_column_concurrently`, batched background migrations) and PlanetScale's deploy-request workflow are the same idea at the platform level.

### Mechanical sympathy: the deploy pipeline

In practice the ordering is enforced by how you stage migrations relative to deploys. A common, robust setup runs two migration phases per release:

```release N:
  1. pre-deploy migration   (expand: additive, backward-compatible only)
  2. deploy app code        (uses new structure, still tolerates old)
  3. post-deploy migration  (data backfill, runs as a background job)
release N+1 (or later):
  4. deploy app code        (stops using old structure)
  5. post-deploy migration  (contract: destructive, drops old structure)
```

The hard rule that makes this safe: **pre-deploy migrations contain only operations that the *old* (currently-running) app code tolerates** — additive, non-breaking. Destructive operations live in a *later* release's post-deploy phase, after the code that needed them is fully gone. Backfills never share a transaction (or even a release) with the schema change that created the column they fill.

## Rollout safety: timeouts, priorities, and lag

**Senior rule of thumb: assume every DDL statement will, sometimes, collide with a long-running query. Bound the collision with `lock_timeout` so it fails fast and retries, rather than waiting forever and dragging the whole table down with it.**

We opened with the lock-queue pile-up. Here is how to defang it. The figure below contrasts a migration session with and without a lock timeout — the same brief-lock operation that froze the table in figure 1 becomes a harmless, auto-retrying no-op.

![Setting a short lock_timeout converts a cascading lock pile-up into a fast safe retry](/imgs/blogs/zero-downtime-schema-migrations-9.webp)

The single most important setting for Postgres DDL is `lock_timeout`. It bounds how long a statement will *wait to acquire a lock* before giving up. The guidance from teams who have learned this the hard way is unambiguous: "DDL statements in migration sessions should always set lock_timeout to an appropriate value for the application; values of less than 2 seconds are common" ([Xata](https://xata.io/blog/migrations-and-exclusive-locks); see also [avoiding downtime with lock_timeout](https://www.denhox.com/posts/avoiding-downtime-in-postgres-with-lock-timeout/)). The pattern is: set a short `lock_timeout`, attempt the DDL, and if it fails with "canceling statement due to lock timeout," retry with backoff. Because the `ALTER` never sits at the head of the queue for more than 2 seconds, it never piles up a queue behind it.

```sql
-- Postgres: bound the lock wait. If the table is busy, fail fast and retry.
SET lock_timeout = '2s';
SET statement_timeout = '0';   -- the work itself may legitimately take time
ALTER TABLE users ADD COLUMN status text NOT NULL DEFAULT 'active';
-- On "canceling statement due to lock timeout", sleep with backoff and re-run.
```

The two timeouts do different jobs and you usually want both, set differently:

- **`lock_timeout`** — how long to wait to *acquire* the lock. Keep this short (1–2s) for DDL so you never pile up the queue.
- **`statement_timeout`** — how long the statement may *run once started*. For a fast metadata change set this short too; for a legitimate long operation (a concurrent index build) you may need it long or zero, but then you rely on `lock_timeout` alone to protect the queue.

For MySQL the equivalent waiting bound is `lock_wait_timeout`, which limits how long an `ALTER` waits for its metadata lock:

```sql
-- MySQL: bound the metadata-lock wait so a busy table fails the ALTER fast.
SET SESSION lock_wait_timeout = 2;
ALTER TABLE users ADD COLUMN status VARCHAR(16) NOT NULL DEFAULT 'active',
  ALGORITHM=INSTANT;
```

Three more rollout-safety practices that separate teams who never have a migration incident from teams who have them monthly:

- **Kill long-running transactions before migrating, or run migrations in a low-traffic window.** A migration's `lock_timeout` protects *you*, but the thing actually blocking you is some other session's long query. Identify and (if appropriate) cancel them: query `pg_stat_activity` for transactions older than a threshold before starting.
- **Watch replica lag as a first-class migration metric.** Both gh-ost (`--max-lag-millis`) and good backfill jobs throttle on lag. A migration that runs faster than your replicas can keep up will create read-replica unavailability even if the primary is fine.
- **Use the engine's "low-priority DDL" affordances where they exist.** gh-ost's throttling, MySQL's `LOCK=NONE`, Postgres's `CONCURRENTLY` and `NOT VALID` are all ways of telling the engine "do this work without prioritizing it over live traffic." Reach for them by default, not as an optimization.

> Set `lock_timeout` on every migration session, no exceptions. It is the cheapest possible insurance: one line that converts your worst migration outage into a log line and an automatic retry.

## Schema evolution and the compatibility model

It is worth grounding all of this in the formal model, because the operational rules above are really one idea applied to storage. Martin Kleppmann's *Designing Data-Intensive Applications* (Chapter 4, "Encoding and Evolution") frames every long-lived system around two directions of compatibility. **Backward compatibility** means new code can read data written by old code. **Forward compatibility** means old code can read data written by new code. In a rolling deploy — and in any system where you cannot atomically swap every component at once — *both* must hold simultaneously, because old and new code are both running and both reading and writing the shared store.

A schema is just an encoding of your data on disk, and a migration is a schema evolution. The expand/contract pattern is precisely the discipline of maintaining backward *and* forward compatibility through the change: in the expand phase you add the new field without removing the old, so old code (which ignores the new field) and new code (which can use it) both work — that is forward compatibility for old readers and backward compatibility for new readers, held at once. Kleppmann's point that you should add fields in a way that old readers can ignore, and remove fields only after no writer produces them, is *exactly* the additive-then-destructive ordering we derived operationally. The reason a bare column drop is dangerous is that it breaks forward compatibility (old code expects a field new data no longer has); the reason a bare add-with-the-app-deployed-first is dangerous is that it breaks backward compatibility (new code expects a field old data does not have). Every safe-migration rule in this article is a special case of "preserve both directions of compatibility for the entire window in which mixed versions are live."

This is also why the database is the *hardest* place to evolve a schema: unlike a message format, where old messages eventually flush out of the queue, **data on disk lives forever**. A field you stop writing today still exists in rows written years ago. There is no "all the old data has expired" moment. That permanence is what forces the backfill phase — you cannot wait for old-format data to age out; you must actively rewrite it — and it is why the contract phase must be deferred until you are certain no reader and no rollback will ever need the old structure again.

## Case studies from production

### 1. The 180-million-row ALTER that took staging by surprise

A team adds a column to a 180-million-row Postgres table. Staging has 5,000 rows; the migration returns in 40 ms and is approved. In production the statement is a *volatile-default* add — `DEFAULT gen_random_uuid()` — which forces a full table rewrite under `ACCESS EXCLUSIVE`. Within two minutes the API returns 503s across all services, the primary is at 100% CPU, and the connection pool is exhausted ([source incident](https://oneuptime.com/blog/post/2026-03-31-mysql-metadata-lock-issues-alter-table/view)). The wrong first hypothesis was "the database is overloaded," and they tried to add read replicas — which did nothing, because the table was *locked*, not slow. The actual root cause: a one-statement add with a volatile default rewrites every row. The fix: cancel the migration, add the column nullable with no default, backfill UUIDs in batches, then add the default. The lesson: staging row count, not staging schema, is what made this invisible — *always test migrations against a production-scale dataset, or at least reason about the rewrite explicitly.*

### 2. The metadata lock behind an idle transaction

A MySQL `ALTER TABLE` hangs indefinitely. `SHOW PROCESSLIST` shows it "waiting for metadata lock." The team's first hypothesis is a bug in the `ALTER`. The real cause: an application connection had run `SELECT ... ` inside a transaction and then *never committed* — it was blocked on a slow downstream HTTP call, holding a shared MDL the whole time. The `ALTER` could not get its exclusive MDL, and every subsequent query queued behind it (per the [oneuptime MDL writeup](https://oneuptime.com/blog/post/2026-03-31-mysql-metadata-lock-issues-alter-table/view)). The fix: find the offending transaction in `performance_schema.metadata_locks` and `information_schema.innodb_trx`, kill it, and set `lock_wait_timeout=2` on the migration session so the `ALTER` fails fast next time instead of becoming the head of a queue. The lesson: *the thing blocking your DDL is almost never the DDL — it is some other long or idle-in-transaction session.*

### 3. gh-ost cut-over during a replication-lag spike

A team runs gh-ost on a several-hundred-GB MySQL table during business hours. The copy phase is fine — gh-ost throttles on `--max-load`. But at cut-over, replicas are 8 seconds behind because of an unrelated batch job, and the atomic `RENAME` briefly blocks. Reads served from replicas see stale data; the cut-over's brief metadata lock collides with the lag. The fix: gh-ost's `--max-lag-millis` and `--postpone-cut-over-flag-file` — they postpone the cut-over until replicas are caught up, and let an operator trigger the final swap manually during a quiet moment. GitHub's whole operational model around gh-ost is built on this: the copy can run for hours harmlessly; only the cut-over needs a calm window. The lesson: *with gh-ost, the copy is free and the cut-over is the only risky instant — schedule it, gate it on lag, and make it manual for big tables.*

### 4. The pt-osc foreign-key surprise

A team uses pt-online-schema-change on a table that other tables reference via foreign keys. The migration appears to work, but afterward the foreign keys point at the *old* table name (now an orphaned `_table_old`), or the FK rebuild silently used the slow `rebuild_constraints` method and held locks far longer than expected. Per Percona's docs, "the technique of atomically renaming the original and new tables does not work when foreign keys refer to the table." The fix: understand pt-osc's `--alter-foreign-keys-method` *before* running, choose `rebuild_constraints` vs `drop_swap` deliberately, and on heavily-referenced tables prefer gh-ost or a hand-rolled expand/contract. The lesson: *foreign keys are the Achilles heel of trigger-based online-schema-change; check them first.*

### 5. The backfill that bloated the table to 2×

A team backfills a new column on a 300-million-row Postgres table with a single `UPDATE`. It runs for 40 minutes, holding row locks and generating 300 million dead tuples. The table's physical size doubles; autovacuum, tuned for steady state, cannot keep up and falls permanently behind. Disk usage climbs for days. Queries slow down because the table is full of dead tuples the planner must skip. The fix: rewrite the backfill as batched, committed, keyset-paginated updates (as in the backfill section), and run a manual `VACUUM (ANALYZE)` after — *not* `VACUUM FULL`, which would have taken `ACCESS EXCLUSIVE` and caused a second outage; `pg_repack` if disk reclaim was genuinely needed online. The lesson: *one big UPDATE is two problems — locks and bloat — and batching solves both.*

### 6. The CREATE INDEX that blocked every write

An engineer runs `CREATE INDEX idx ON orders (status)` — without `CONCURRENTLY` — on a write-heavy Postgres table at peak. The `SHARE` lock blocks all writes for the eleven minutes the build takes. Checkout, which writes `orders`, hangs entirely; the read path is fine, which confuses the initial diagnosis ("reads work, so the DB is up"). The fix: cancel, run `CREATE INDEX CONCURRENTLY` instead (and clean up the partial INVALID index the cancel left behind, via `DROP INDEX CONCURRENTLY`). The team then adds a CI check (strong_migrations-style) that fails any migration containing a non-concurrent `CREATE INDEX`. The lesson: *in Postgres, `CONCURRENTLY` is not an optimization, it is the only acceptable way to index a table that takes writes.*

### 7. The rename that broke the canary

A team renames `users.name` to `users.full_name` in a single migration and deploys it with the code change. During the canary rollout, old pods (still referencing `name`) throw "column name does not exist" for ~6 minutes until the rollout completes — and when the canary is rolled back because of the errors, the *new* pods' rollback can't help because the column is already gone. A small but real partial outage, made worse by the rollback. The fix: the six-phase expand/contract rename — add `full_name`, dual-write, backfill, switch reads, stop old writes, drop `name` weeks later. The lesson: *a rename is never atomic in a rolling-deploy world; it is an expand followed by a contract with dual-write in between, and the contract waits until rollback is off the table.*

### 8. The volatile-default add disguised as a constant

A team adds `created_at TIMESTAMPTZ NOT NULL DEFAULT now()` on Postgres 14, expecting the fast-default path. `now()` is *stable* within a transaction, so it does store a single value and avoids a rewrite — but they then "improve" it to `DEFAULT clock_timestamp()` to get a per-row timestamp, not realizing `clock_timestamp()` is *volatile*. The volatile default forces a full rewrite of a 400-million-row table under `ACCESS EXCLUSIVE`. The migration that was instant yesterday is an outage today, from a one-word change. The fix: add nullable, backfill `clock_timestamp()` per-row in batches, then add the default. The lesson: *constant vs volatile is the entire difference between instant and outage for column defaults — `now()` is safe, `clock_timestamp()` and `gen_random_uuid()` are not.*

### 9. The MySQL ALTER that silently fell back to COPY

A team runs an `ALTER TABLE ... MODIFY COLUMN` on MySQL 8.0 expecting INSTANT or at worst INPLACE. The specific change (a type change that alters storage) is not INPLACE-eligible, so MySQL silently picks COPY, rebuilds the whole 500-million-row table, and blocks writes for hours. The fix going forward: *always* pin `ALGORITHM=INPLACE, LOCK=NONE` (or `ALGORITHM=INSTANT`) so MySQL refuses to run rather than silently degrading — turning a multi-hour outage into an immediate, safe error message. For the change itself, use gh-ost. The lesson: *MySQL's automatic algorithm fallback is a footgun; pin the algorithm so a "free" operation that isn't free fails loudly instead of silently.*

### 10. The contract phase that never came

Eighteen months after an expand/contract column migration, an audit finds the database has a dozen dead columns, three sync triggers, and two "ghost" tables left over from migrations whose contract phase was never executed. The dead triggers add measurable write latency; the dead columns confuse new engineers and inflate every `SELECT *`. Nothing broke, but the schema rotted. The fix: a quarterly "schema debt" review and a rule that every expand/contract migration files its contract phase as a dated, owned ticket at the moment the expand ships. The lesson: *expand/contract's most common failure is organizational, not technical — the contract phase is invisible work that never gets prioritized unless you make it a tracked obligation.*

### 11. Stripe's subscriptions migration at scale

Stripe needed to move subscriptions from a JSON array nested in the customer record into a dedicated table — "hundreds of millions of Subscription objects" — with 99.999% uptime ([Stripe: online migrations at scale](https://stripe.com/blog/online-migrations)). They used the four-step dual-write pattern: write both stores, backfill offline (Hadoop/MapReduce, *not* expensive production queries), switch reads with [Scientist](https://stripe.com/blog/online-migrations) dark-read comparisons alerting "as soon as a single piece of data was inconsistent in production," then switch writes and remove the old data. The discipline that made it safe: "never attempting to change more than a few hundred lines of code at one time" and making every change "highly transparent and observable." The lesson: *at the largest scale the pattern does not change — it just adds an offline backfill engine and a rigorous dark-read verification step; the six phases are the same.*

### 12. The replica that fell over during a fast backfill

A team's backfill is correctly batched and committed, but with *no* throttle and no sleep — it runs as fast as the primary allows. The primary is fine, but the backfill generates WAL faster than the replicas can replay it. Replica lag climbs to minutes; the read-replica-served dashboards go stale and then time out; a read-heavy service that routes to replicas starts erroring even though the primary is healthy. The fix: add a lag check between batches (pause while `pg_wal_lsn_diff` to any replica exceeds a budget) and a small inter-batch sleep, exactly as gh-ost does with `--max-lag-millis`. The lesson: *a backfill's blast radius includes your replicas — throttle on actual replica lag, because "the primary is fine" is not the same as "the system is fine."*

## When to reach for each technique, and when not to

### Reach for native online DDL when

- The operation is genuinely metadata-only on your exact engine and version: nullable `ADD COLUMN`, constant-default `ADD COLUMN` (PG11+/MySQL 8.0.12+), `DROP COLUMN`, `RENAME` (with expand/contract for the app side).
- You can pin the algorithm (`ALGORITHM=INSTANT/INPLACE` in MySQL) so a non-free operation fails loudly instead of falling back to COPY.
- You bound the lock wait with `lock_timeout` / `lock_wait_timeout` so a busy table produces a fast retry, not a pile-up.
- In Postgres, you always use `CONCURRENTLY` for indexes and `NOT VALID` → `VALIDATE` for constraints and NOT NULL.

### Reach for expand/contract when

- The change is breaking by nature: a rename, a type change, a column split, a table extraction.
- Old and new app versions will run simultaneously during the rollout (i.e. always, in any rolling-deploy system).
- You need the change to be reversible at every step — each phase can be rolled back independently.
- The data already on disk must be transformed, not just the schema (forcing a backfill phase).

### Reach for an online-schema-change tool (gh-ost / pt-osc / pg_repack / Vitess) when

- The native engine cannot do the operation online (a MySQL type change that falls back to COPY).
- The table is large enough and hot enough that even the brief native locks are unacceptable, and you need pausable, throttleable, replica-testable migration (gh-ost).
- You need bloat reclamation without `VACUUM FULL` (pg_repack).
- You want migration-as-a-reviewed-deploy-request with instant revert (Vitess/PlanetScale).

### Skip the heavy machinery when

- The table is small (a few hundred thousand rows): a plain `ALTER` under a `lock_timeout` in a quiet window is fine, and an online-schema-change tool's 2× disk and binlog plumbing is pure overhead.
- The operation is already metadata-only on your engine: do not wheel out gh-ost to add a nullable column.
- You are in a true maintenance window with traffic stopped: if you can genuinely take the table offline, a direct `ALTER` is simpler and faster than expand/contract — but be honest about whether "maintenance window" really means zero traffic, because partial traffic plus a global lock is the classic trap.
- The change is to a brand-new table with no production data and no readers yet: there is nothing to be compatible with, so just `ALTER` it.

The throughline of all of it: a schema migration is a distributed-systems change, not a SQL statement. Treat the database as one of several components — old app, new app, replicas — that must all remain mutually compatible for the entire window in which they coexist. Make every change additive-then-destructive, bound every lock, batch every backfill, verify every cut-over, and the migration that used to be an outage becomes a non-event nobody notices. That is the entire goal: the best schema migration is the one your users never feel.

## A copy-paste migration checklist

Put this in your migration review template. Every box must be checked before a migration ships.

```SCHEMA MIGRATION SAFETY CHECKLIST
=================================
[ ] I know the table's production row count (not staging's).
[ ] I know whether this operation REWRITES the table on the
    exact engine + version in production.
[ ] I know what LOCK it takes and for how long.
[ ] No volatile DEFAULT (gen_random_uuid/clock_timestamp/random)
    on ADD COLUMN — use nullable + batched backfill instead.
[ ] NOT NULL via CHECK ... NOT VALID -> VALIDATE -> SET NOT NULL
    (Postgres), in SEPARATE statements.
[ ] Indexes use CREATE INDEX CONCURRENTLY (Postgres) /
    ALGORITHM=INPLACE, LOCK=NONE (MySQL); INVALID-index cleanup planned.
[ ] MySQL: ALGORITHM pinned so a non-free op fails loudly,
    not a silent COPY fallback.
[ ] lock_timeout (1-2s) / lock_wait_timeout (2s) SET on the
    migration session; retry-with-backoff on lock-timeout error.
[ ] statement_timeout considered separately from lock_timeout.
[ ] Breaking changes use expand/contract; NO bare RENAME or
    in-place TYPE change on a large hot table.
[ ] Migration is backward-compatible with the CURRENTLY-RUNNING
    app AND the previous version (rollback-safe).
[ ] Additive schema BEFORE the code that uses it; destructive
    schema AFTER the code that stops using it (+ soak).
[ ] Backfill is batched, committed per-batch, keyset-paginated,
    idempotent, resumable, and throttled on replica lag.
[ ] No backfill in the same transaction (or release) as the DDL.
[ ] Cut-over (gh-ost RENAME / tool swap) gated on replica lag
    and scheduled for a calm window for large tables.
[ ] Long-running / idle-in-transaction sessions checked and
    cleared before starting (pg_stat_activity / innodb_trx).
[ ] Contract phase filed as a dated, owned ticket the moment
    the expand ships.
[ ] Post-migration VACUUM (ANALYZE) planned (never VACUUM FULL);
    pg_repack if online disk reclaim needed.
```

For the lock mechanics underneath all of this, see the companion deep dives on [locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) and the [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), and for why index builds cost what they cost, [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work).

### Further reading

- [Stripe — Online migrations at scale](https://stripe.com/blog/online-migrations): the canonical four-step dual-write writeup, with the Scientist verification pattern.
- [GitHub — gh-ost: GitHub's online migration tool for MySQL](https://github.blog/news-insights/company-news/gh-ost-github-s-online-migration-tool-for-mysql/): why triggerless binlog-based migration beats triggers at scale.
- [Percona — pt-online-schema-change docs](https://docs.percona.com/percona-toolkit/pt-online-schema-change.html): the trigger-based approach and its foreign-key caveats.
- [PlanetScale — Safely making schema changes](https://planetscale.com/blog/safely-making-database-schema-changes) and [how schema reverts work](https://planetscale.com/blog/behind-the-scenes-how-schema-reverts-work): online DDL as a reviewable, revertible workflow.
- [Xata — Schema changes and the Postgres lock queue](https://xata.io/blog/migrations-and-exclusive-locks): the definitive explanation of the lock pile-up and `lock_timeout`.
- [strong_migrations gem](https://github.com/ankane/strong_migrations) and [GitLab migration style guide](https://docs.gitlab.com/development/migration_style_guide/): codified safe-vs-unsafe operation lists you can adopt as a review checklist.
- [pg_repack](https://github.com/reorg/pg_repack): online table/index rebuild for Postgres without `VACUUM FULL`.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 4 (Encoding and Evolution): the backward/forward-compatibility model that every safe migration is an instance of.

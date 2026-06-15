---
title: "Schema and API Evolution at Scale: Migrations Without Downtime"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn the expand-contract pattern as the one master technique for changing the hardest things in a live system — database schemas, API contracts, and event formats — with online migrations, billion-row backfills that never spike load, and a rollback at every step."
tags:
  [
    "system-design",
    "schema-migration",
    "api-versioning",
    "expand-contract",
    "zero-downtime",
    "backfill",
    "protobuf",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/schema-and-api-evolution-at-scale-1.webp"
---

There is a moment in the life of every system where someone proposes a change that looks like a one-line diff and is in fact a six-week project. "We just need to rename the `email` column to `email_address`." "Let's change `amount` from cents-as-integer to a decimal." "Marketing wants `status` to be an enum instead of a free-text string." On a laptop with a toy database, each of these is a single `ALTER TABLE` and a code edit. On a system serving live traffic, with a table that has two billion rows, with thirty service instances mid-rollout running two different versions of your code, and with four downstream teams consuming an event you publish — each of these is the kind of change that takes a service down if you do it the obvious way. Data and contracts are the hardest things to change in a running system, precisely because you can never change them *atomically*. There is no instant where all the data, all the code, and all the consumers flip at once. There is always a window, and in that window the old and the new must coexist.

I want to be precise about what this post is. The `database/` folder on this blog already deep-dives the mechanisms underneath all of this: how [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) stream row changes reliably, and how [you reshard and rebalance a live system without downtime](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime). This post is the *architect's decision layer*. Not how a trigger replays a write, but how a senior *sequences* a change so that at no single instant is the system in a state that breaks — and how that one discipline, the **expand-contract pattern** (also called parallel-change), is the same whether you are evolving a SQL column, a REST payload, a gRPC message, or a Kafka event that consumers you have never met depend on. Figure 1 is the entire post in one picture: every safe change is three independently deployable phases — expand, migrate, contract — never one switch.

![A timeline showing the expand-contract migration as three independently shippable phases moving from old shape only through expand and migrate to contract and new shape only](/imgs/blogs/schema-and-api-evolution-at-scale-1.webp)

By the end you should be able to do five concrete things. First, take any breaking change and decompose it into an expand-migrate-contract sequence where each step is independently deployable and independently reversible. Second, run an online schema migration on a large table — adding and dropping columns, changing types, renaming — without the long-running lock that takes the table down. Third, plan a backfill of a billion rows with a batch size, a throttle, and a defensible time estimate that does not melt your primary. Fourth, evolve a serialization format (protobuf, Avro) and the events that carry it so that consumers running old code and new code both decode every message. Fifth, reason about backward *and* forward compatibility well enough that you stop fearing the rollout window and start designing for it. Let us build to all five.

## 1. Why you can never flip the switch atomically

Start from the constraint that makes this whole topic hard, because every technique is a response to it. In a single-process program, you can change a data structure and the code that uses it in the same commit, and there is never an instant where one is updated and the other is not — the compiler and the process boundary give you atomicity for free. A distributed, continuously-deployed system has none of that. Your code runs on many machines. A deploy is a *rolling* deploy: you replace instances one at a time, or a fraction at a time, so that the service stays up. For minutes — sometimes much longer if a canary bakes — half your fleet runs the old code and half runs the new. Your database is one shared thing that both halves talk to. So the schema and the code can never be in lockstep, because the code itself is not in lockstep with itself.

This is the single insight the entire post hangs on, so let me state it as a rule. **During any rollout there are at least four states running simultaneously: old code on old schema, old code on new schema, new code on old schema, and new code on new schema.** A change is safe only if all four of those states work. The blocking ALTER, the in-place rename, the "just change the field type" — each of them creates at least one of those four cells that does not work, and that cell is the outage. The expand-contract pattern is nothing more than a recipe for guaranteeing that all four cells always work, by never removing the old shape until the new shape is fully live and verified.

The same logic, exactly, governs the boundary between your service and the world. When you change an API, your clients do not redeploy the instant you do. Mobile apps live on phones you do not control and update on the user's schedule, which is to say never for a meaningful tail of your install base. Partner integrations were written once and forgotten. Other internal teams are mid-sprint. So an API change has the same coexistence window as a schema change, just with a longer and less controllable tail. And an *event* schema is the worst of all: when you publish a Kafka or Pub/Sub event, you often do not even *know* every consumer, and you certainly do not control their deploy schedule, so the coexistence window for an event format is effectively *forever*. The discipline that makes a column rename safe is the same discipline that makes an event format safe — it just has to hold for longer.

There is a deep connection here to [evolutionary architecture and designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change), which is the sibling post on building systems you can change at all. This post is the operational specialization: given that you have to change the hardest, most-shared things — the data and the contracts — *how* do you do it without taking the system down. The answer is always the same shape, and that shape is expand-contract.

It is worth being honest about *why* the data and the contract are uniquely hard, because it explains why this single pattern keeps showing up. Most things in a system are *internal* and *fast to replace*: a function, a class, a module, even a whole service can be rewritten and redeployed in an afternoon, and the only thing that has to agree on its shape is the code that calls it — which you own and ship together. The data and the contract are different on both counts. They are *shared* across boundaries you do not control in lockstep (other instances of your own service, other teams' services, clients on the public internet), and they are *durable* — the data persists for years, an old event sits in a topic indefinitely, an API call is made by software you have never seen. Sharedness means you cannot change all the readers at once; durability means the old shape outlives the moment you stop writing it. The combination is what removes the atomic switch, and every technique in this post is a way to manage the resulting overlap window. If you remember nothing else, remember that the difficulty is not technical complexity — a `RENAME COLUMN` is trivial — it is that the change touches something shared and durable, so there is no instant where everyone agrees.

## 2. Expand-contract: the master technique

Let me define the pattern crisply, because everything downstream is an application of it. **Expand-contract** changes a shape in three phases, each a separate, deployable, reversible step:

1. **Expand.** Add the new shape *alongside* the old one. Add the new column, the new field, the new API parameter, the new event field. Crucially, the old shape still exists and still works. You deploy code that *writes both* the old and new shapes and *reads the old one*. After this phase the system is doing redundant work, but it is completely backward compatible: old code that knows nothing about the new shape keeps working, because the old shape is untouched.

2. **Migrate.** Backfill. The new shape exists for all *new* writes, but historical data only has the old shape. So you copy the old data into the new shape, in batches, in the background, throttled so you do not hurt the live workload. When the backfill finishes, every row has both shapes populated and consistent. You verify this — count rows, sample-compare values, run a shadow read — before you trust it.

3. **Contract.** Now, and only now, remove the old shape. Deploy code that *reads the new shape* and *stops writing the old one*. Bake that. Then, in a final deploy, *drop* the old column, the old field, the old API version. The system now runs on the new shape alone.

The power of the pattern is that **each phase is independently deployable and each phase up to the drop is reversible.** If expand goes wrong, you roll back the deploy and the old shape was never disturbed. If the backfill goes wrong, you stop it and delete the new column's data; nothing read it yet. If cutover goes wrong, you flip a flag back to reading the old shape, which you are still writing. Only the very last drop is a one-way door, and by the time you reach it you have proven, with data, that nothing depends on the old shape anymore. Compare that to the blocking ALTER, which is one giant irreversible step that either works or takes you down.

Figure 2 contrasts the two approaches for the canonical case — adding a column to a large table — and it is worth internalizing the difference in *mechanism*, not just outcome.

![A before and after diagram contrasting a blocking ALTER that locks the whole table for forty minutes against an online gh-ost migration that copies into a shadow table and swaps with a sub-second lock](/imgs/blogs/schema-and-api-evolution-at-scale-2.webp)

I want to nail down one piece of terminology because people muddle it constantly. **Backward compatible** means new code can read data written by old code — the new version understands the old shape. **Forward compatible** means old code can read data written by new code — the old version tolerates the new shape, typically by ignoring fields it does not recognize. A rolling deploy needs *both*: new instances must read what old instances wrote (backward), and old instances must not choke on what new instances wrote (forward). Expand-contract gives you both, because in the expand phase the new shape is purely additive (old code ignores it: forward compatible) and the old shape is still written (new code can read it: backward compatible). The whole pattern is a machine for staying in the doubly-compatible region long enough to migrate the data.

## 3. The compatibility matrix you must keep green

Because the four-state insight is load-bearing, let me make it concrete with the exact case it governs: a rolling deploy across a schema change. Figure 5 lays out the grid — old and new code on one axis, old and new schema on the other — and the goal of expand-contract is that *every cell is green*.

![A two by two matrix of old code and new code against old schema and new expanded schema showing all four cells working because the change is additive](/imgs/blogs/schema-and-api-evolution-at-scale-5.webp)

Walk the cells with a concrete change: you are adding a nullable `phone_verified BOOLEAN` column. Old code on old schema is your baseline — fine. New code on new schema is the destination — fine, it writes and reads `phone_verified`. The two off-diagonal cells are where migrations die. *New code on old schema*: this happens during expand if you deploy code that reads `phone_verified` before the column exists — every query errors with "column does not exist." The fix is ordering: the schema change (add column) must ship *before* the code that reads it. *Old code on new schema*: this happens because the moment you add the column, instances still running the old code are now talking to the new schema. If you added the column as `NOT NULL` with no default, the old code's `INSERT` (which does not mention `phone_verified`) fails the not-null constraint. The fix is to make the column nullable, or give it a default, so old code's inserts still succeed.

That second failure is the one that bites people who think "add a column" is always safe. It is safe *if and only if* the column is nullable or has a default, because old code that does not know about the column must still be able to insert rows. This is why the iron rule of the expand phase is: **additive changes only, and never break the existing write path.** A new column must accept the inserts that old code already issues. A new API field must be optional. A new event field must be ignorable. The instant your "add" forces old code to do something different, it is not additive and you have a broken cell.

The matrix also tells you the *order of operations* for a rename, which is the change people most want to do in place and most must not. You cannot rename `email` to `email_address` in one step, because there is no single instant where every running instance agrees on the name. New code reading `email_address` breaks against the old schema; old code reading `email` breaks against the renamed schema. There is no atomic flip. A rename is therefore *always* an expand-contract: add `email_address`, write both, backfill, cut reads over, stop writing `email`, drop `email`. We will walk that exact sequence end to end in section 9. For now, internalize the consequence: **a rename is never a rename. It is an add, a copy, and a drop, separated by deploys.**

## 4. Online schema migrations: the long-running ALTER problem

Now to the database mechanics, because "just add a column" hides a sharp edge that depends entirely on your engine and table size. The problem is the lock.

On MySQL with InnoDB, many `ALTER TABLE` operations historically required a full copy of the table while holding a metadata lock, during which writes (and sometimes reads) to that table block. Modern MySQL supports `ALGORITHM=INPLACE` and `ALGORITHM=INSTANT` for some changes — adding a nullable column at the end of the row is often instant — but plenty of changes (changing a column type, adding a column in the middle, adding certain indexes, changing character sets) still rewrite the whole table and hold a lock long enough to be an outage. On a 2-billion-row table, a full rewrite can take *tens of minutes to hours*, and for that entire window writes pile up behind the lock. Your connection pool fills with blocked queries, your app's request threads block waiting for connections, and within a couple of minutes the whole service is timing out even though the database is technically "up." That is the locking-ALTER outage, and it has taken down more systems than almost any other single self-inflicted cause.

On PostgreSQL the failure modes are different but just as real. `ADD COLUMN` with a constant default is fast in modern Postgres (it does not rewrite the table), but `ADD COLUMN ... DEFAULT <volatile>` or changing a type with `ALTER COLUMN ... TYPE` rewrites the table and holds an `ACCESS EXCLUSIVE` lock — which blocks *everything*, including reads. Worse, Postgres DDL takes that lock at the *start* and will *wait in line* behind any long-running query holding even a shared lock, and while it waits it blocks every query behind *it*. So a migration that would take 5 seconds can stall the whole table for minutes because one slow analytics query was holding a read lock when the ALTER tried to acquire its exclusive lock. The senior's discipline on Postgres is: set a short `lock_timeout` on the migration so it fails fast rather than queueing behind it and blocking the world, and retry; never let a DDL statement sit in the lock queue indefinitely.

There is a non-obvious second-order danger on Postgres that catches even experienced engineers, and it is worth dwelling on because it is the difference between "my migration is slow" and "my migration took down the whole table." Postgres locks are *queued and fair*: when your `ALTER` requests `ACCESS EXCLUSIVE`, it goes to the back of the lock queue for that table. If a long-running `SELECT` is holding an `ACCESS SHARE` lock (which every read takes), your `ALTER` *waits* — and here is the trap — *every query that arrives after your `ALTER`* also waits, behind the `ALTER`, because lock requests do not jump the queue. So a migration that would have taken 5 milliseconds can freeze the entire table for as long as that one slow analytics query runs, and the freeze affects reads too. You did not run a slow migration; you ran a fast migration that got stuck behind a slow reader and then blocked the world behind itself. The defenses are specific: set `lock_timeout` to a small value (say `5s`) on the migration session so it gives up and releases the queue rather than holding everyone hostage, then retry the migration in a loop; and never run schema changes while a long analytics query or a `pg_dump` is in flight against the same table. This is the kind of failure mode that does not show up in staging — where there are no concurrent long readers — and detonates in production, which is exactly why a senior reasons about the lock queue before running any DDL on a busy table.

A related Postgres-specific rule: **add indexes with `CREATE INDEX CONCURRENTLY`, never a plain `CREATE INDEX`.** A plain index build takes a lock that blocks writes for the entire build, which on a large table is minutes. The `CONCURRENTLY` variant builds the index in the background without blocking writes, at the cost of taking longer and not being usable inside a transaction (and it can leave an invalid index behind if it fails, which you then drop and retry). It is the online-migration pattern baked directly into the index DDL, and forgetting the keyword is a classic way to block writes you did not mean to block.

The general escape from the long-running-ALTER problem is to **not let the database do the rewrite under a lock at all.** Instead, do the rewrite in the background with an online schema change tool. The mechanism, sketched in figure 2, is the same across the major tools: create a new empty *shadow table* with the desired schema, install triggers (or tail the binlog) so every live write to the original is also applied to the shadow, copy existing rows into the shadow in throttled batches, and when the shadow has caught up, perform an atomic `RENAME` to swap the tables — a swap that holds a lock for well under a second instead of for the whole rewrite. The two dominant tools in the MySQL world are **pt-online-schema-change** (Percona, trigger-based) and **gh-ost** (GitHub, binlog-based, triggerless). [Vitess](/blog/software-development/database/vitess-sharding-mysql-at-scale) builds online DDL directly into the sharding layer so a single statement runs a managed, throttled, revertible migration across every shard. The point is the same: trade a long lock for a long *background copy*, paying with extra disk and time to buy zero downtime.

#### Worked example: adding a column to a 500-million-row table

You need to add a nullable `last_login_at TIMESTAMP` to a `users` table with 500 million rows on MySQL. The table sustains about 3,000 writes per second of live traffic. Naively, `ALTER TABLE users ADD COLUMN last_login_at TIMESTAMP NULL` on this engine version rewrites the table; at a copy rate of, say, 50,000 rows/second on your hardware, that is 500{,}000{,}000 / 50{,}000 = 10{,}000 seconds, or about 2 hours and 45 minutes. For that entire window writes to `users` block. That is a guaranteed multi-hour outage. Unacceptable.

Run it with gh-ost instead:

```bash
gh-ost \
  --host=primary.db.internal \
  --database=app \
  --table=users \
  --alter="ADD COLUMN last_login_at TIMESTAMP NULL" \
  --chunk-size=1000 \
  --max-load='Threads_running=25' \
  --critical-load='Threads_running=80' \
  --max-lag-millis=1500 \
  --throttle-control-replicas='replica-1.db.internal,replica-2.db.internal' \
  --initially-drop-ghost-table \
  --postpone-cut-over-flag-file=/tmp/ghost.postpone \
  --execute
```

The cut-over (the atomic swap) is *postponed* behind a flag file you delete only when you are ready, so the multi-hour background copy runs entirely off the hot path and you choose the exact low-traffic minute for the sub-second swap. `--max-load` and `--max-lag-millis` are the throttle: gh-ost pauses copying whenever the primary gets busy or a replica lags past 1.5 seconds, so the live 3,000 writes/second never compete with the backfill. The migration takes *longer* than the blocking ALTER in wall-clock terms — maybe 4 hours because it throttles — but it costs *zero* downtime and zero user-visible latency, and you can abort it at any second before the cut-over with no trace left behind. That is the trade the senior makes every time: duration and disk for safety and reversibility.

## 5. The dual-write and shadow-read mechanism

The expand phase has a subtlety that deserves its own section, because the obvious implementation has a race and the correct one is a known pattern. When you add a new shape and "write both," you have two stores (or two columns) that must stay consistent, and you have historical data that only has the old shape. Two questions follow: how do you keep the two in sync going forward, and how do you trust the new shape before you cut reads over to it? The answers are **dual-write** and **shadow-read**, shown in figure 3.

![A pipeline showing an app write request fanning into a dual write of old and new columns, then a shadow read that compares the two, reaching zero divergence so cutover is safe](/imgs/blogs/schema-and-api-evolution-at-scale-3.webp)

**Dual-write** means application code, for every write, updates both the old and the new shape in the same transaction. The "same transaction" part matters: if you write the old column and the new column in one SQL `UPDATE`, they cannot diverge, because the database commits them atomically. The danger appears when the two shapes live in *different* stores — say you are migrating from one database to another, or splitting a column into a new table — because then a dual-write is two separate operations and a crash between them leaves them inconsistent. That is exactly the dual-write problem the [outbox pattern and change data capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) exist to solve: instead of writing to two stores from the app (which is not atomic), you write to one store and let CDC propagate the change to the other, so there is a single source of truth and the propagation is exactly-once-ish and replayable. For same-table column changes, a single transaction is enough; for cross-store changes, reach for CDC rather than app-level dual-writes.

There is a subtle ordering question inside dual-write that trips people up: in the *same-store, same-transaction* case, which column is the source of truth during the expand phase? The answer is: the *old* one, still, all the way through cutover. During expand you write both but the new column is, in effect, a derived copy you are validating; the old column remains authoritative because it is the one every reader still trusts. You do not flip authority to the new column until cutover, after shadow-read has proven agreement. This matters because if you treat the new column as authoritative too early — say a bug in the dual-write writes the wrong value to the new column but the right value to the old — and some reader has already started preferring the new column, you have served wrong data. The rule is: **the column you read is the column that is authoritative, and you change which you read in exactly one controlled step (cutover), never implicitly.** Keeping authority and readership tied together is what makes the cutover a single reversible flag flip instead of a fuzzy transition where different readers disagree about which column is real.

**Shadow-read** is how you earn the confidence to cut over. After dual-writing for a while and backfilling history, you have a new column you *believe* is correct. Before you bet production reads on it, you read *both* the old and the new shape on the live read path, return the old value to the user (so users are unaffected), and *compare* the two in the background. You count divergences and log them. If after a day of real traffic the new column matches the old column on 100% of reads, you have proven the new shape is correct under the actual workload — including all the weird edge cases your backfill might have missed — and cutover is safe. If you see divergence, you have caught a bug *before* it touched a user, and you fix the dual-write or the backfill and keep shadowing. Shadow-read turns "I think the migration is correct" into "I measured zero divergence over 80 million live reads," which is the difference between a hopeful cutover and a boring one.

## 6. The trade-off matrix: choosing a migration approach

Now the decision layer. There is no single "right" way to run a schema migration; there is a right way *for this table, this change, this risk tolerance.* Figure 4 is the matrix a senior carries in their head, scoring each approach on the properties that actually matter in a design review: downtime, risk, duration, reversibility, and impact on the live load.

![A matrix scoring blocking ALTER, gh-ost or pt-osc, Vitess online DDL, and app-level expand-contract against downtime, risk, duration, reversibility, and load impact](/imgs/blogs/schema-and-api-evolution-at-scale-4.webp)

Read the matrix as a set of trade-offs, never as a ranking. The same data in table form, because a design review wants the reasoning written down:

| Approach | Downtime | Risk | Duration | Reversible | Best when |
| --- | --- | --- | --- | --- | --- |
| **Blocking `ALTER`** | High (lock for whole rewrite) | High | Shortest wall-clock | No | Tiny table, maintenance window, or an instant/in-place change the engine supports |
| **gh-ost / pt-online-schema-change** | None | Low | Long (background copy, throttled) | Yes (abort pre-cutover) | Large MySQL table, single-shape DDL, no app code change needed |
| **Vitess online DDL** | None | Low | Long | Yes (revert flag) | Sharded MySQL; one statement across all shards, managed throttle |
| **App-level expand-contract** | None | Low | Longest (multi-deploy) | Yes (at every phase) | Renames, type changes, cross-store moves, anything the DDL tools can't express |

The key senior insight hiding in this matrix: **the blocking ALTER is not always wrong.** On a 10{,}000-row config table, a blocking ALTER finishes in 50 milliseconds and the lock is invisible. The cost of an online tool — the operational complexity, the extra disk, the hours of background copy — is pure waste on a small table. The size of the table is the first thing a senior asks, because it flips the recommendation. Roughly: under a few million rows, a direct `ALTER` (with a short `lock_timeout` on Postgres so it fails fast rather than blocking the world) is usually fine; in the tens-of-millions-and-up range, reach for an online tool; for renames, type changes, and cross-store moves that DDL cannot express atomically, reach for app-level expand-contract. The matrix is not telling you the best approach in the abstract — it is telling you that the approach is a *function of the inputs*, and naming those inputs out loud is the whole job. This is the same way [articulating trade-offs with CAP and PACELC](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) frames every architectural choice: name the gain, name the cost, name the condition under which it wins.

## 7. The optimization lens: backfilling huge tables without melting the database

The expand phase is cheap; the **migrate** phase — the backfill — is where the real engineering, and the real risk to the live system, lives. A backfill is a giant batch job that touches every row of a huge table, and the naive version of it is a famous way to take down a production database. So the optimization angle here is not "make the backfill fast"; it is "make the backfill *invisible* to the live workload while still finishing in a reasonable time." Figure 8 is the safe shape: batch, throttle, verify.

![A pipeline showing a backfill that processes five thousand rows per batch by primary key cursor, throttles when replica lag exceeds one second, and verifies that old and new counts match](/imgs/blogs/schema-and-api-evolution-at-scale-8.webp)

Start with what *not* to do, because the naive backfill is so tempting:

```sql
-- DO NOT run this on a large table.
UPDATE users SET email_address = email WHERE email_address IS NULL;
```

On a 500-million-row table this single statement is a catastrophe for three reasons. First, it is one transaction, so it holds locks on every row it touches for the entire duration — minutes to hours — blocking concurrent writes and bloating the transaction log. Second, it generates an enormous amount of write-ahead log / binlog in one burst, which floods replication and pushes replica lag from milliseconds to minutes, breaking every read that expects a fresh replica. Third, if it fails or you kill it at row 400 million, the whole transaction rolls back and you have accomplished nothing while having paid for everything. The naive backfill is the backfill-overloads-the-database failure mode in one line of SQL.

The correct backfill is **batched, keyed, throttled, and resumable.** Walk the table in small chunks by primary key (not by `LIMIT/OFFSET`, which gets quadratically slower as the offset grows — page by the last key you saw, which stays fast because it uses the index). Commit each chunk as its own transaction so locks are held for milliseconds, not hours. After each chunk, check replica lag (or primary load) and *sleep* if it is too high, so the backfill yields to live traffic. And track your position so that if the job dies, you resume from where you stopped instead of starting over.

```python
import time
import psycopg2

BATCH = 5000          # rows per chunk; small enough to commit in well under a second
LAG_CEILING_S = 1.0   # pause the backfill when replica lag exceeds this
SLEEP_ON_LAG = 0.5    # back off this long when lagging

def replica_lag_seconds(cur):
    cur.execute("SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))")
    row = cur.fetchone()
    return row[0] or 0.0

def backfill(conn, replica_conn):
    last_id = 0
    total = 0
    while True:
        with conn.cursor() as cur:
            # Page by primary key cursor: index-backed, constant cost per batch.
            cur.execute(
                """
                WITH batch AS (
                    SELECT id FROM users
                    WHERE id > %s AND email_address IS NULL
                    ORDER BY id
                    LIMIT %s
                )
                UPDATE users u
                SET email_address = u.email
                FROM batch b
                WHERE u.id = b.id
                RETURNING u.id
                """,
                (last_id, BATCH),
            )
            ids = [r[0] for r in cur.fetchall()]
            conn.commit()  # one short transaction per batch

        if not ids:
            break  # no more rows to backfill

        last_id = max(ids)
        total += len(ids)

        # Throttle: protect the live workload by yielding when replicas fall behind.
        with replica_conn.cursor() as rcur:
            while replica_lag_seconds(rcur) > LAG_CEILING_S:
                time.sleep(SLEEP_ON_LAG)

    return total
```

Now the part a senior actually gets asked in a design review: **how long will it take, and how do we keep it from hurting anything?**

#### Worked example: planning a one-billion-row backfill

You must backfill a new `region_code` column across 1{,}000{,}000{,}000 rows. Decide the batch size, the throttle, and produce a time estimate you can defend.

Pick the batch size from the commit-latency budget, not from a round number. You want each batch's `UPDATE` to commit in well under the time that would cause noticeable lock contention — target ~100 ms per batch. Measure: on this table, updating 5{,}000 rows by primary key and committing takes about 80 ms. Good — `BATCH = 5000`. That gives 1{,}000{,}000{,}000 / 5{,}000 = 200{,}000 batches.

Now the rate. If each batch takes 80 ms of work and you add a small inter-batch sleep so you are not pinning the database — say you target a *duty cycle* where the backfill consumes at most ~30% of one connection's time, sleeping ~190 ms between batches — then each batch cycle is about 80 + 190 = 270 ms. 200{,}000 batches × 0.27 s = 54{,}000 seconds ≈ **15 hours** of wall-clock time. If that is too slow, run *4 parallel workers*, each owning a disjoint primary-key range (worker 1 does ids 0–250M, worker 2 does 250M–500M, and so on), and the wall-clock drops to about 4 hours — at the cost of 4× the write load, which is exactly why the throttle that watches replica lag is non-negotiable when you parallelize. The throttle is the safety valve: if four workers push replica lag past 1 second, they all sleep until it recovers, so the backfill *automatically* slows down to protect the live workload instead of overrunning it. You measure the win in the negative: replica lag stays under 1 second throughout, p99 of live queries does not move, and zero requests time out — while a billion rows get rewritten underneath the running system.

The senior move embedded here: **make the backfill a knob, not a commitment.** Batch size, sleep duration, and worker count are all runtime-tunable, so if you start the backfill and watch the dashboards and see lag creeping up, you turn the throttle tighter without restarting. And because the job is keyed and resumable, you can pause it entirely during a traffic peak and resume it overnight. A backfill that you can speed up, slow down, pause, and resume from a dashboard is a backfill that never becomes an incident.

## 8. Data format evolution: protobuf, Avro, and the schema registry

Everything so far assumed the data lives in a SQL table you control. But an enormous amount of the data in a modern system lives as *serialized bytes* — protobuf or Avro messages in Kafka, in caches, in RPC payloads, in blob storage — and those bytes are written by one version of your code and read by another, often much later, often by a service on a different team. Evolving a serialization format is expand-contract again, but the rules are sharper and the failure mode is *silent data corruption* rather than a loud error, which makes it more dangerous.

The foundational rule of protobuf (and the source of the most expensive evolution bug there is) is that **a field is identified by its tag number, not its name.** When protobuf serializes `user_id = 42`, it does not write the string "user_id"; it writes the tag number (say `1`) and the value. The name exists only in the `.proto` file for the programmer's benefit. This is what makes protobuf compact and fast, and it is what makes one specific mistake catastrophic. Figure 7 shows it.

![A before and after diagram contrasting reusing protobuf tag five so old integer bytes decode as a string and corrupt, against an additive-only schema that reserves tag five forever and adds the new field as tag nine](/imgs/blogs/schema-and-api-evolution-at-scale-7.webp)

The catastrophic mistake is **reusing a tag number.** Suppose field `5` was `int64 legacy_score`, you removed it, and six months later someone adds `string user_tier` and gives it tag `5` because it looks free. Now every old message still sitting in a Kafka topic, a cache, or a backup has an int64 at tag 5. When new code deserializes that old message, it reads the bytes at tag 5 *as a string* — because the tag is all protobuf has to go on — and you get garbage: a corrupted value, a decode error, or, worst of all, a *plausible-looking wrong value* that flows silently into your business logic. There was no error. The bytes decoded "successfully" into the wrong meaning. This is why protobuf has the `reserved` keyword and why using it is mandatory, not optional:

```protobuf
message User {
  int64  user_id    = 1;
  string email      = 2;
  string display_name = 3;
  bool   phone_verified = 4;

  // legacy_score was field 5; it is gone forever.
  // Reserve the tag AND the name so nobody can ever reuse them.
  reserved 5;
  reserved "legacy_score";

  string user_tier = 9;  // new field gets a NEW, never-used tag.
}
```

The complete set of safe-evolution rules for protobuf, which are really just expand-contract specialized to tag-numbered wire formats:

- **Adding a field is always safe** if you give it a new, never-before-used tag number. Old code ignores unknown tags (forward compatible); new code reading old messages sees the field as its default/unset (backward compatible). This is the expand step, and it is free.
- **Removing a field**: stop populating it (contract), and `reserve` its tag number and name *forever*. Never delete the line and reuse the slot.
- **Never change a field's tag number** — that is a remove-plus-add and breaks every existing message.
- **Never change a field's type** in a wire-incompatible way (e.g., `int64` to `string`). Some changes are compatible (`int32`/`int64`/`bool`/enum share a wire type and can be widened with care), but the safe pattern is: add a *new* field with the new type, dual-write both, migrate, then reserve the old one — expand-contract, again.
- **Renaming a field is free in proto3** because the name is not on the wire, but it can break code or JSON mappings, so treat it as a code change with the usual rollout care.

Avro takes a different but related approach: the schema travels (or is referenced) with the data, and compatibility is checked against named fields with explicit defaults. Avro's evolution rule is that **every field you might remove later needs a default value**, so that a reader using a newer schema (which lacks the field) or an older schema (which lacks a newly-added field) can fill the gap with the default. Whether protobuf or Avro, a production event platform puts a **schema registry** in front of the topic: the registry stores every version of every schema and *rejects* a producer trying to register an incompatible new version. You configure the compatibility mode — `BACKWARD` (new schema can read old data, the common default), `FORWARD` (old schema can read new data), or `FULL` (both) — and the registry enforces it at publish time, so an incompatible schema change fails in CI instead of corrupting a consumer in production. The registry is how you make the expand-contract discipline *mechanical* instead of relying on every engineer remembering the tag rules. This is the natural extension of how [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) treats the event as a long-lived contract: the schema registry is the enforcement layer for that contract.

There is a second-order point that separates seniors here. With an event, the consumers are decoupled *in time*: a message you publish today may be read tomorrow, or replayed from the start of the topic next month during a backfill of a new consumer. So an event schema's coexistence window is not "the length of a rolling deploy" — it is "the retention of the topic," which can be infinite for a compacted topic or an event-sourced log (see [event sourcing and CQRS](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log)). You can never assume every old message has been consumed and can stop being decodable. That is why the additive-only, reserve-forever discipline is *stricter* for events than for a database column: with a column you can eventually drop it once nothing reads it; with an event format, the old shape may need to remain decodable essentially forever.

## 9. API evolution: expand-contract over breaking versions

The same pattern governs the contract you expose to clients. The instinct when you need a breaking API change is to cut a new version — `/v1/` to `/v2/` — and that has its place, covered in depth in [API design across REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql). But a senior reaches for expand-contract *first*, because a new major version is the most expensive form of evolution there is: it forks your handler logic, doubles your test surface, and saddles you with maintaining `/v1/` for years because some client never migrates. Most "breaking" changes do not actually need a new version if you evolve the existing one additively.

The decision tree in figure 6 routes a change to the cheapest safe pattern, and the cheapness gradient is the whole point: a true addition is one deploy; everything destructive is multi-step.

![A decision tree routing a change by type, where additive nullable changes are one deploy and destructive renames or type changes become a three-deploy expand-contract](/imgs/blogs/schema-and-api-evolution-at-scale-6.webp)

For an API, expand-contract looks like this. Suppose you need to change a response field `name` (a single string) into structured `first_name` and `last_name`. The breaking-version instinct says cut `/v2/`. The expand-contract approach:

1. **Expand.** Add `first_name` and `last_name` to the response *alongside* `name`. Keep populating `name` (now derived from the two new fields, or dual-maintained). New clients can read the structured fields; old clients keep reading `name` and never notice. On the request side, accept *both* the old flat field and the new structured fields, preferring the new if both are present.
2. **Migrate.** Update your own clients, your SDKs, your documentation, and any internal callers to use the new fields. Watch your access logs / API analytics to see the request rate against the old `name` field decay toward zero. This is the API equivalent of a backfill: you are migrating the *callers*, and the metric you watch is "how many requests still use the old field."
3. **Contract.** Once the old field's usage has dropped below a threshold you are comfortable with (and you have given any stragglers explicit notice with a deprecation header and a sunset date), stop populating it and eventually remove it. For an external API you may *never* fully reach this step for years, and that is fine — the old field costs little to keep, and forcing a sunset on partners is a business decision, not an engineering one.

The senior framing: **API versioning and expand-contract are not alternatives; expand-contract is the default and a new major version is the escape hatch for when the change is so pervasive that additive evolution would create a confusing Frankenstein schema.** You cut a `/v2/` when the *shape* of the resource changes fundamentally — a different resource model, a different auth scheme, a different pagination contract — not when you are adding or splitting a few fields. The instrument that makes this manageable is *measuring usage of every field and endpoint*, because you cannot contract what you cannot prove is unused. Teams that log per-field usage can deprecate confidently; teams that do not are stuck supporting every field forever because removing one is a leap of faith.

The deprecation mechanics deserve a concrete recipe, because "stop using the old field" is a process, not a flag. The standard signaling is HTTP: set the `Deprecation` header (with the date the field became deprecated) and the `Sunset` header (with the date you intend to remove it) on every response that still includes the old field, and add a `Link` header pointing to migration docs. These are machine-readable, so sophisticated clients can alert on them, and they create a paper trail. Then you *watch the metric*: instrument the handler to count requests that read the deprecated field, broken down by client (API key, user-agent, or partner ID). The deprecation is done not when a calendar date passes but when that count reaches zero or drops below a threshold where you can reach out to the remaining callers individually. For an internal API across teams you control, this can take weeks; for a public API with partners, it can take a year or more, and forcing the sunset before usage drops is a *business* decision about which partners you are willing to break, not an engineering one. The engineering job is to make the cost of *keeping* the old field low (it is just a few extra bytes in the response and a derived value) so that the business is never pressured to break clients prematurely.

A gRPC-specific note, since the [API design post](/blog/software-development/system-design/api-design-rest-grpc-graphql) covers the protocol mechanics: because gRPC rides on protobuf, API evolution and data-format evolution collapse into the *same* rules. Adding an RPC method, adding a field to a request or response message, adding an enum value — all additive, all safe by the tag-number rules from section 8. The danger is the same too: never renumber a field, never reuse a tag, reserve removed tags. The pleasant consequence is that a gRPC API designed with additive evolution rarely needs a new major version at all, because the wire format already tolerates additive change by construction — which is one of the underrated reasons large systems standardize on it for internal service-to-service contracts.

#### Worked example: renaming a column end-to-end with deploy steps

Let us make the whole pattern concrete with the canonical hard case — renaming `email` to `email_address` on a live 200-million-row Postgres table that an API also exposes — and write down every deploy. There are *six* deploys (or migration steps), and the ordering is the entire skill.

**Step 1 — Expand (schema):** Add the new column, nullable, no rewrite.

```sql
ALTER TABLE users ADD COLUMN email_address TEXT NULL;
```

This is fast on modern Postgres (a nullable column with no default is a catalog-only change) and the column is nullable so old code's inserts still succeed. The compatibility matrix stays green: old code ignores the new column, new code does not exist yet.

**Step 2 — Expand (code):** Deploy application code that **writes both** columns on every insert and update, in the *same transaction*, but still **reads** from `email`.

```python
# In the same transaction — they cannot diverge.
cur.execute(
    "INSERT INTO users (id, email, email_address) VALUES (%s, %s, %s)",
    (uid, value, value),
)
```

Now all *new* writes populate both. Old rows still have `email_address = NULL`. This deploy is reversible: roll it back and you are writing only `email` again, exactly as before.

**Step 3 — Migrate (backfill):** Run the batched, throttled, resumable backfill from section 7 to copy `email` into `email_address` for the historical rows. After it completes, every row has both columns consistent. Verify with a count:

```sql
SELECT count(*) FROM users WHERE email_address IS DISTINCT FROM email;  -- expect 0
```

If this returns anything but zero, do not proceed — find the divergence (usually a write path you missed in step 2) and fix it.

**Step 4 — Cutover (code, shadow first):** Deploy code that **reads `email_address`** (with shadow-read comparison against `email` for a day to confirm zero divergence under live traffic), while still **writing both**. Once shadow-read shows 100% agreement over real traffic, the new column is proven. This is the cutover, and it is reversible by a flag flip back to reading `email`.

**Step 5 — Contract (code):** Deploy code that **stops writing `email`** and reads/writes only `email_address`. Now nothing populates the old column. Still reversible: redeploy step 4 and you resume writing both. Bake this for long enough that a rollback window has comfortably passed.

**Step 6 — Contract (schema):** Drop the old column. This is the only irreversible step, and you take it only after weeks of nothing reading or writing `email`.

```sql
ALTER TABLE users DROP COLUMN email;
```

Six deploys to do what looked like one `RENAME COLUMN`. But every one of the first five is reversible, none of them takes the system down, and at no instant is any of the four compatibility cells broken. *That* is what it means to change the hardest thing in a live system without breaking it. The "wasteful" redundancy of writing both columns for a few weeks is the price of never having a window where the system is unsafe.

## 10. What is reversible, and when (the rollback discipline)

The reason expand-contract is the senior's default is not elegance — it is that **it is reversible right up until the last possible moment.** A migration that must roll back mid-flight is not a hypothetical; it is Tuesday. The backfill turns out to have a bug. The new code has a latency regression nobody caught in staging. A dependency you did not know about reads the old column. When that happens at 2 a.m., the only question that matters is: *can I get back to a known-good state without losing data?* Expand-contract is engineered so the answer is yes for every phase except the final drop. Figure 9 makes the reversibility gradient explicit, because knowing *which* phase you are in tells you exactly how you roll back.

![A stack showing reversibility shrinking from a fully reversible expand phase through reversible backfill and cutover and stop-writing, down to an irreversible contract drop](/imgs/blogs/schema-and-api-evolution-at-scale-9.webp)

Read the stack top to bottom as a decreasing-reversibility gradient:

- **Expand (add column / field):** Fully reversible. The old shape is untouched; roll back the deploy and you are exactly where you started. The new column sitting empty harms nothing.
- **Migrate (backfill):** Reversible. Nothing reads the new column yet, so if the backfill is wrong, you stop it and either re-run it correctly or `UPDATE ... SET new_col = NULL` to clear it. No user ever saw the new data.
- **Cutover (read new):** Reversible by a flag. You are still writing both shapes, so flipping reads back to the old shape is instant and lossless. This is why you keep dual-writing through cutover — it is the rollback insurance.
- **Stop writing old:** Reversible by redeploy. The old column still exists with its data intact (just no longer updated), so redeploying the dual-write code resumes maintaining it. There is a small window where the old column is stale, which is why you bake this step before dropping.
- **Contract (drop):** **Irreversible.** Once you `DROP COLUMN`, the data is gone (short of a restore). This is the one-way door, taken last, after every reversible step has proven the new shape under real traffic.

The discipline this dictates: **keep the old shape alive — and keep writing it — for as long as it is your rollback path, and drop it only when you can prove nothing depends on it.** The most common self-inflicted migration disaster is dropping the old column too early, the day after cutover, while you are still inside the rollback window — and then the new code hits a bug, and you have nothing to roll back *to*. Patience between "stop writing old" and "drop old" is not timidity; it is the entire safety margin. A senior treats the drop the way a climber treats letting go of the last hold: only when fully established on the new one.

## 11. Stress-testing the migration plan

Every design deserves a stress test, so let me throw the three hard cases at the expand-contract plan and show how it holds — or where you have to add something.

**Stress 1: the migration must roll back mid-flight.** You are at step 4 (cutover, reading the new column) and discover the backfill missed a class of rows — users created during a specific buggy deploy last year have `email_address = NULL`. Reads for those users now return null instead of their email. Because you are still dual-writing and the old column is intact, you flip the read flag back to `email` instantly: outage averted in seconds, no data lost. Then you fix the backfill to cover the missed rows, re-verify with the `IS DISTINCT FROM` count, and re-attempt cutover. The plan held because reversibility was designed in, not bolted on. The lesson the failure teaches: your *verify* step (step 3's count) was incomplete — it should have caught the null rows — so you strengthen verification before the next migration. This is exactly the kind of GitHub-style online-migration recovery their engineering team has written about: gh-ost's postponed cut-over and abortability exist precisely so that "we found a problem, abort the migration" is a non-event rather than an incident.

**Stress 2: the backfill overloads the database.** You start the billion-row backfill with four parallel workers and within minutes replica lag climbs to 8 seconds — read replicas are now serving stale data and your read-after-write guarantees are broken. This is the classic backfill-overload outage. With the naive single-`UPDATE` backfill you would be deep in an incident with no clean way out. With the batched, throttled design, the throttle *already caught it*: the lag-watching loop paused the workers when lag crossed the 1-second ceiling, so lag never reached 8 seconds in the first place. But suppose your ceiling was set too loose. The recovery is graceful because the job is batched and resumable: you pause all workers, let replicas catch up, lower the batch size and worker count (knobs, not commitments), and resume from the saved cursor. No transaction to roll back, no progress lost. The design *degrades* under pressure instead of *failing* under it.

**Stress 3: a consumer didn't get the new schema.** You publish events to a Kafka topic and you add a required field to the event, assuming all consumers will pick up the new schema. But one downstream team — whose service you do not own and whose deploy schedule you do not control — is still running last month's code, and last month's code chokes on the field it does not expect (or, if you made it *required*, the schema registry should have rejected your publish in the first place under `BACKWARD` compatibility). This is why event evolution must be *additive and optional, enforced by the registry*: a new event field is always optional with a default, old consumers ignore what they do not recognize (forward compatibility), and the registry refuses any schema that would break an existing consumer. The stress test reveals the design rule: for events, you cannot rely on coordinating consumer deploys, so the *format itself* must guarantee that an un-upgraded consumer keeps working. The plan that survives is the one where the registry enforces compatibility mechanically, so the incompatible change fails in CI rather than in a consumer you cannot reach. This connects directly to [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design): consumers that are robust to redelivery and tolerant of unknown fields are the consumers that survive schema evolution.

The throughline of all three stress tests: the migration plan is not judged by how it behaves when everything goes right — the blocking ALTER is fine when everything goes right — but by how it behaves when something goes wrong *in the middle*. Expand-contract wins because every failure mode has a clean recovery that does not require time travel.

## 12. Making the discipline mechanical: CI, linting, and migration tooling

The hardest part of schema evolution is not knowing the rules — they fit on an index card — it is *applying them consistently* across a team of dozens of engineers, most of whom touch a migration a few times a year and do not carry the lock-queue and tag-number rules in their heads. A senior's real leverage is not running a perfect migration once; it is making the safe path the *default* path so that the unsafe path is hard to take by accident. That means moving the discipline out of people's memory and into tooling that fails the build.

The first piece is a **migration linter** in CI that statically inspects every schema-change PR and rejects the dangerous patterns. There are off-the-shelf tools for this — `squawk` for Postgres, `online-migrations` checks for Rails, and similar — and they catch exactly the traps from sections 3 and 4: adding a `NOT NULL` column without a default (breaks old code's inserts), `CREATE INDEX` without `CONCURRENTLY` (blocks writes), `ALTER COLUMN ... TYPE` that rewrites the table (long lock), dropping a column that code might still reference. The linter turns "remember not to do that" into "the PR is red until you fix it," which is the only thing that scales past a handful of people.

```yaml
# CI step: lint every migration for unsafe operations before it can merge.
- name: lint database migrations
  run: |
    squawk migrations/*.sql \
      --exclude=prefer-text-field \
      --assume-in-transaction
    # Fails the build on: NOT NULL without default, non-concurrent index,
    # table-rewriting type changes, dropping referenced columns.
```

The second piece is **protobuf/Avro schema compatibility enforcement** in CI, which is the wire-format analog. The `buf` tool runs a *breaking-change detector* that compares the `.proto` in your PR against the version on the main branch and fails if you reused a tag, changed a field's type, removed a field without reserving it, or renamed a field that maps to a JSON name. For Avro and Kafka, the schema registry's compatibility check can be run in CI against the registered schema, so an incompatible schema is rejected at PR time rather than at publish time in production.

```yaml
# CI step: reject any wire-incompatible proto change before merge.
- name: check proto compatibility
  run: |
    buf breaking --against '.git#branch=main'
    # Fails on: reused tag number, changed field type, deleted-not-reserved
    # field, incompatible default changes.
```

The third piece is a **deploy-ordering contract** enforced by your migration framework. Recall from section 3 that the schema change must ship *before* the code that reads the new column, and *after* nothing reads the old one. Frameworks that decouple "migrate" from "deploy" let you sequence this explicitly: run the expand migration, deploy the dual-write code, run the backfill as a separate job, deploy the cutover code, run the contract migration. The anti-pattern is a framework that runs migrations automatically *during* the same deploy that ships the code — because then your add-column migration and your read-the-column code go out together, and during the rolling deploy you hit the broken cell where new code meets old schema (or old code meets new schema) for the seconds it takes the migration to apply across the fleet. Seniors keep migrations and code deploys as *separate, ordered steps* precisely so the four-cell matrix never has a broken cell, even momentarily.

The point of all three is the same: the safe migration pattern should not depend on every engineer remembering the rules under deadline pressure at 4 p.m. on a Friday. Encode the rules in linters and compatibility checks that fail the build, encode the ordering in tooling that separates migrate from deploy, and the discipline becomes the floor instead of the ceiling. This is the operational complement to [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design): you instrument the *process* of changing the system the same way you instrument the running system, so the dangerous move is visible and blocked before it ships.

## 13. Case studies

**GitHub and gh-ost: making online migrations boring.** GitHub runs MySQL at large scale and historically used trigger-based tools (like pt-online-schema-change) for online migrations, but triggers add write overhead and are hard to pause and control under load. GitHub built and open-sourced **gh-ost** specifically to make migrations *controllable*: it is triggerless (it tails the binlog instead of installing triggers), it can be *paused and resumed* on the fly, it dynamically *throttles* based on replica lag and primary load, and — the feature that most changes the operational experience — its cut-over is *postponable behind a flag file*, so the hours-long copy runs whenever and the sub-second swap happens at a human-chosen low-traffic moment. The lesson for an architect is not "use gh-ost"; it is that the *controllability* of a migration — pause, resume, throttle, abort, postpone — is what turns a migration from a high-stakes event into a routine background job. Design your own backfills with the same knobs.

**Stripe and the API that never breaks.** Stripe is famous for an API that has stayed backward compatible for many years. Their mechanism is a form of expand-contract taken to its logical extreme: every breaking change is gated behind a dated *API version*, and each integration is pinned to the version it was written against. Internally, requests from an old-version client pass through a chain of *version-transformation* shims that translate the old request shape into the current internal shape and translate the current response back into the old shape — so the core code only ever deals with the latest schema, while old clients see exactly the contract they were built for, forever. The architectural lesson: you can support an unbounded coexistence window if you *centralize the compatibility logic* (the version transformers) rather than scattering `if version < X` checks through your handlers. The old shape lives at the boundary, transformed, while the inside stays clean and singular.

**The protobuf field-reuse corruption.** This is the war story that teaches the tag-number rule the hard way, and a version of it has happened at many companies. A field — say `int32 account_status = 7` — is deprecated and removed from the `.proto` without a `reserved` declaration. Months later, a different engineer adds `string promo_code = 7`, reusing the now-"free" tag. Everything compiles, tests pass on new data, and it ships. Then a batch job replays old events from a Kafka topic — events serialized when tag 7 was an int32 — and the new code deserializes the old int32 bytes *as a string*. Depending on the bytes, this either throws a decode error (the lucky case, because at least it is loud) or, worse, produces a garbage string that flows silently into downstream logic, corrupting reports or, in the genuinely bad cases, customer-visible data. The fix is always the same and always retroactive-pain: stop reusing tags, `reserve` every retired tag and name, and ideally enforce it with a CI lint (buf, or a custom check) that fails any PR reusing a reserved tag. The lesson: with tag-numbered formats, *the tag is the contract*, and reusing one is silent data corruption waiting for a replay.

**The locking-ALTER outage.** The most common production migration incident needs no specific company name because it has happened everywhere: an engineer runs a "quick" `ALTER TABLE` — add an index, change a column type — directly against a large production table, in business hours, without an online tool. On MySQL the ALTER takes a metadata lock and starts rewriting the table; on Postgres the `ALTER COLUMN ... TYPE` takes `ACCESS EXCLUSIVE` and either rewrites under the lock or queues behind a slow query and then blocks everything behind *it*. Within a minute or two, writes to that table are blocked, the connection pool fills with stuck queries, app threads block waiting for connections, and the whole service is throwing timeouts even though the database is "up." The migration cannot be cleanly cancelled because killing it rolls back hours of work. The lesson seniors carry from this: **never run an unbounded DDL statement against a large table on the hot path.** Set a short `lock_timeout` (Postgres) so it fails fast instead of blocking the world, use an online tool for anything that rewrites a large table, and treat "it's just an ALTER" as the dangerous phrase it is.

## When to reach for this (and when not to)

Expand-contract is the default for **any change to data or a contract that has a coexistence window** — which, in a continuously-deployed or multi-client system, is *every* change to a schema, an API, or an event format. If old and new will run simultaneously (and they always will), you decompose the change into expand, migrate, contract, and you make each step independently deployable and reversible. This is not the heavyweight option you reach for only on big changes; it is the *normal* way to change shared state, and adding a column the wrong way (`NOT NULL`, no default) is just as capable of causing an outage as a botched rename.

Reach for an **online migration tool** (gh-ost, pt-osc, Vitess online DDL) when the table is large enough that a direct DDL would lock it for more than a blink — tens of millions of rows and up, or any change that rewrites the table. Below a few million rows, a direct `ALTER` (with a short `lock_timeout` on Postgres) is simpler and correct; the online tool's complexity is pure overhead on a small table.

Reach for a **batched, throttled, resumable backfill** for any data migration touching more than a few hundred thousand rows. The single-statement `UPDATE` is fine on a small table and a database-melting outage on a large one; the line is roughly where the statement would run longer than a second or two.

Do **not** reach for expand-contract's full ceremony when there genuinely is no coexistence window — a true offline maintenance window, a brand-new table no code reads yet, a feature behind a flag that no one has enabled. And do not cut a **new major API version** when an additive evolution would do; reserve `/v2/` for changes so pervasive that additive evolution would produce an incoherent schema. The skill is matching the weight of the technique to the actual risk: the four-cell compatibility check tells you when you need the full pattern and when you can take a shortcut. When a change touches an event consumed by teams you do not control, take the *strictest* version of the discipline — additive and optional only, enforced by a schema registry — because the coexistence window there is effectively infinite.

## Key takeaways

1. **You can never flip the switch atomically.** In a rolling-deploy, multi-client world, old and new always coexist for a window. Design for the window; do not pretend it does not exist.
2. **Expand-contract is the one master technique.** Add the new shape (expand), backfill (migrate), remove the old shape (contract) — for columns, APIs, and events alike. Each phase is independently deployable and, until the drop, reversible.
3. **Keep all four compatibility cells green.** Old/new code × old/new schema must all work. The expand phase is additive-and-optional precisely so old code never breaks; if your "add" forces old code to change, it is not additive.
4. **A rename is never a rename.** It is an add, a copy, a cutover, and a drop, separated by deploys. There is no in-place rename that is safe on a live shared store.
5. **Never run an unbounded DDL on a large table on the hot path.** Use an online tool (gh-ost, pt-osc, Vitess) for big rewrites; set a short `lock_timeout` on Postgres so DDL fails fast instead of blocking the world.
6. **Backfills are batched, keyed, throttled, and resumable — or they are outages.** Page by primary key, commit small chunks, sleep on replica lag, and track your cursor so you can pause and resume. Make every parameter a runtime knob.
7. **The tag is the contract.** In protobuf, identity is the tag number, not the name. Never reuse a retired tag; `reserve` it forever. Additive-only field changes keep old and new messages decodable.
8. **Put a schema registry in front of your events** and enforce a compatibility mode. It turns the evolution discipline from "everyone remembers the rules" into a mechanical check that fails incompatible changes in CI, not in a consumer you cannot reach.
9. **Drop last, and only when proven unused.** Keep writing the old shape as long as it is your rollback path. The most common migration disaster is dropping the old column while still inside the rollback window.
10. **Controllability is the property that matters.** Pause, resume, throttle, abort, postpone — a migration you can steer from a dashboard is a routine job; one you cannot is a high-stakes event.

## Further reading

- [Evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) — the sibling post on building systems you can change at all; this post is its operational specialization for the hardest-to-change things.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — the dual-write, backfill, cutover playbook applied to moving rows between shards.
- [API design: REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) — where API versioning lives in depth; expand-contract is the default and a new major version is the escape hatch.
- [Queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) — the event-as-long-lived-contract view that makes the schema registry essential.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the reliable mechanism for cross-store dual-writes, so you do not lose consistency when the two shapes live in different stores.
- [Live resharding and rebalancing without downtime](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime) — the mechanism deep-dive underneath zero-downtime data movement.
- **gh-ost** (GitHub) documentation — the design of a controllable, triggerless, throttled online MySQL migration tool.
- **Confluent Schema Registry** documentation — compatibility modes (BACKWARD, FORWARD, FULL) and how to enforce them at publish time.
- The protobuf and Avro language guides on field/schema evolution — the authoritative rules for additive-only changes and reserved tags.

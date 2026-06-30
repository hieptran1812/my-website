---
title: "Online Schema Changes at Scale: gh-ost, pt-osc, and Expand-Contract"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "At scale you change schema with purpose-built tools and the expand-contract pattern, because a single naive ALTER TABLE can lock a large table and take down every write behind it."
tags: ["schema-migration", "gh-ost", "pt-online-schema-change", "expand-contract", "postgresql", "mysql", "zero-downtime", "database-scaling", "ddl", "online-ddl"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 36
---

There is a class of outage that does not start with a bad deploy, a traffic spike, or a hardware failure. It starts with a one-line database migration that passed code review, ran instantly in staging, and looked completely harmless. Someone runs `ALTER TABLE orders ADD COLUMN ...` against the production primary at 2am because that is the quiet window, the command does not return, and within ninety seconds the checkout API is returning 5xx for every customer on the site. Nobody changed any application code. The schema change *was* the incident.

The reason is brutally simple: many `ALTER` statements take a lock on the table and hold it for the entire duration of the operation, and on a table that is hundreds of gigabytes that duration is measured in minutes to hours, not milliseconds. While the lock is held, every query that touches the table waits. The connection pool fills with waiters, the application's request threads block on the pool, and what was supposed to be a schema tweak becomes a full-site outage. At small scale you never notice, because the table is small and the lock is brief. At scale, the same command is a loaded gun.

![The same column change runs two completely different ways: a naive ALTER takes an outage-length exclusive lock and rewrites the whole table, while an online change copies into a shadow table and cuts over in under a second.](/imgs/blogs/online-schema-changes-at-scale-1.webp)

The diagram above is the mental model for the entire post. On the left is the naive path: one statement, one giant exclusive lock, the whole table rewritten in place, and everything that wants the table waiting until it finishes. On the right is how you do the *same change* at scale: you never touch the live table's lock for more than a fraction of a second. Instead you build a copy with the new schema, fill it gradually in small throttled chunks, keep it current with live writes the whole time, and swap it in with a single atomic rename. The rest of this article is a tour of that right-hand path — the MySQL tools that automate it (`gh-ost` and `pt-online-schema-change`), the Postgres operations and patterns that achieve it natively, and the application-level discipline (expand-contract) that ties schema changes to deploys so the running app never sees a breaking change.

This is the operational sibling of two patterns you already know if you have read the rest of this series: it is the same shadow-copy-and-cutover shape as [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime), and it lives in the same world of lock contention and write pressure as [keeping Postgres healthy under write load](/blog/software-development/database-scaling/keeping-postgres-healthy-under-write-load). If you want the application-deploy choreography in more depth, the companion post on [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) goes deeper on the release side.

## Why a naive ALTER is dangerous

> A schema change is not a query. It is a privileged operation that can take a lock no query can, and at scale the lock is the whole story.

The first thing to internalize is that "it ran fine in staging" tells you almost nothing, because the danger scales with table size and write concurrency, both of which staging lacks. Here is the gap between the comfortable assumption and the production reality.

| Assumption | Naive mental view | Production reality at scale |
| --- | --- | --- |
| `ALTER` is just another statement | It runs, it returns, done | It can hold an exclusive lock for the entire rewrite — minutes to hours |
| "Online DDL" means no impact | The database says ALGORITHM=INPLACE, so it's free | INPLACE still takes brief exclusive metadata locks and can stall behind long transactions |
| The replica will be fine | Replication handles it | The `ALTER` replays serially on each replica and can lag the whole fleet for hours |
| A quick `ALTER` is low-risk | It only takes a second of lock | It can wait behind a long query, and then *everything* waits behind it |
| Adding a column is cheap | It's metadata | With a volatile default or on older engines, it rewrites every row |

Let us take MySQL and Postgres in turn, because their failure shapes differ.

### MySQL: online DDL is real but full of caveats

Since MySQL 5.6, InnoDB supports `ALGORITHM=INPLACE` and later `ALGORITHM=INSTANT` for many operations, which means the engine can perform some schema changes without copying the whole table. `ADD COLUMN` at the end of the row (in 8.0.12+) is often `INSTANT` — pure metadata, genuinely fast. That is the good news, and for those operations you may not need any external tooling at all.

The caveats are where teams get hurt. First, even an in-place `ALTER` takes a brief but real **metadata lock** at the start and end of the operation. If a long-running transaction is holding the table (a big analytics `SELECT`, an idle transaction someone forgot to commit), the `ALTER` blocks waiting for that metadata lock — and now it is holding the request queue open, just like the Postgres lock-queue problem below. Second, many operations are *not* in-place: changing a column type, adding a column with a `DEFAULT` in older versions, adding an index on some engines, or reordering columns will copy and rebuild the entire table under a lock that blocks writes. Third — and this is the one that quietly ruins your night — even a perfectly in-place `ALTER` on the primary **replays serially on every replica**. Replication is single-threaded per schema for DDL, so a two-hour `ALTER` on the primary becomes two hours of replication lag on every replica, during which your read replicas are serving increasingly stale data or falling out of rotation entirely.

So the rule for MySQL at scale is: for anything beyond a trivially `INSTANT` operation on a large, busy table, do not run a bare `ALTER`. Use `gh-ost` or `pt-online-schema-change`, which we cover next.

### Postgres: some ALTERs are free, some rewrite the world

Postgres is more transparent about its lock levels but has its own sharp edges. The good news is that modern Postgres makes many operations metadata-only:

- `ADD COLUMN` with no default, or with a **constant** default (Postgres 11+), is instant — it stores the default in the catalog rather than writing every row.
- `DROP COLUMN` is instant — it just marks the column dropped; the space is reclaimed lazily by vacuum.
- `RENAME COLUMN` and `RENAME TABLE` are instant — pure catalog edits.

The bad news is that several common operations take an `ACCESS EXCLUSIVE` lock — the strongest lock Postgres has, which conflicts with *everything*, including plain `SELECT` — and hold it while they rewrite the table:

- `ADD COLUMN ... DEFAULT <volatile_expr>` (for example `DEFAULT now()` or `DEFAULT gen_random_uuid()`) rewrites every row, because each row needs a distinct computed value.
- Changing a column's type (`ALTER COLUMN ... TYPE`) generally rewrites the table.
- `SET NOT NULL`, historically, scans the entire table to verify there are no nulls (we will see the modern workaround later).

A full-table rewrite under `ACCESS EXCLUSIVE` on a 300 GB table is exactly the left-hand side of our mental-model diagram: an outage with a `commit` at the end.

### The lock-queue pileup: the failure that surprises everyone

Even when your `ALTER` itself would be fast, Postgres has a second, subtler trap that catches experienced engineers. Lock requests are granted roughly **first-in, first-out**, and a request that cannot be granted yet **blocks every request behind it**, even requests that would otherwise be compatible with the current lock holder.

![One slow ALTER waiting for an exclusive lock blocks every query queued behind it, even shared-lock reads that are perfectly compatible with the current holder.](/imgs/blogs/online-schema-changes-at-scale-2.webp)

Walk through the figure. Transaction A is a slow reporting `SELECT` that has been running for nine minutes; it holds an `AccessShareLock` on `orders`, which is the weakest lock and is held by every reader. Transaction B is your "quick" `ALTER TABLE`, which needs `AccessExclusive`. `AccessExclusive` conflicts with `AccessShareLock`, so B cannot proceed — it **waits** for A to finish. So far this is annoying but bounded. The disaster is what happens next: transaction C is a tiny customer-facing `SELECT` that needs only `AccessShareLock`. C's lock is *perfectly compatible with A's* — under normal circumstances C and A would run concurrently without a second thought. But C arrives *after* B in the queue, and the lock manager will not let C jump ahead of B's exclusive request. So C waits. And D waits. And every query that touches `orders` from this moment piles up behind a single `ALTER` that is itself waiting on one slow report.

This is why a "quick `ALTER`" can take down a site that is under no unusual load at all. The `ALTER` did not need a long lock; it needed a lock it could not get *yet*, and FIFO ordering converted one stuck statement into a total stall. The fix, which we will detail in the safe-migration section, is to never let an `ALTER` sit in that queue: set a short `lock_timeout` so it bails out instead of blocking the world, and retry.

## The MySQL online-DDL tools: gh-ost vs pt-online-schema-change

For MySQL, two battle-tested tools automate the shadow-table-and-cutover dance: Percona's `pt-online-schema-change` (pt-osc), which has been around since the early 2010s, and GitHub's `gh-ost`, which GitHub built and open-sourced in 2016 specifically because pt-osc's trigger-based approach was hurting their busiest tables. They solve the same problem with fundamentally different mechanisms, and the difference matters.

### gh-ost: triggerless, binlog-based, observable

`gh-ost` ("GitHub's online schema transmogrifier") does not put triggers on your table. Instead it pretends to be a replica: it connects to a MySQL server, reads the **binary log** (which must be in `ROW` format), and reconstructs the stream of changes happening to your table. The flow has three concurrent activities, which the animation below traces.

<figure class="blog-anim">
<svg viewBox="0 0 760 340" role="img" aria-label="gh-ost copies rows in chunks from the original table to the ghost table while tailing the binary log for live writes, then swaps the ghost in with a sub-second atomic rename" style="width:100%;height:auto;max-width:820px">
<title>gh-ost: chunked copy, binlog catch-up, and atomic cutover</title>
<style>
.gho-chip{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.gho-tbl{fill:var(--surface,#f3f4f6);stroke:var(--text-secondary,#6b7280);stroke-width:2}
.gho-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.gho-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.gho-row{fill:var(--accent,#6366f1)}
.gho-binlog{fill:var(--text-secondary,#6b7280)}
.gho-sweep{fill:var(--accent,#6366f1);opacity:.16}
@keyframes gho-flow{0%{transform:translateX(0);opacity:0}8%{opacity:1}90%{opacity:1}100%{transform:translateX(280px);opacity:0}}
@keyframes gho-step{0%{transform:translateX(0)}33.33%{transform:translateX(0)}33.34%{transform:translateX(245px)}66.66%{transform:translateX(245px)}66.67%{transform:translateX(490px)}100%{transform:translateX(490px)}}
.gho-mv{animation:gho-flow 4.5s linear infinite}
.gho-d2{animation-delay:1.5s}
.gho-d3{animation-delay:3s}
.gho-bmv{animation:gho-flow 4.5s linear infinite}
.gho-bd2{animation-delay:2.2s}
.gho-anim{animation:gho-step 9s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.gho-mv,.gho-bmv,.gho-anim{animation:none}.gho-mv,.gho-bmv{opacity:1}}
</style>
<rect class="gho-sweep gho-anim" x="20" y="20" width="225" height="50" rx="8"/>
<rect class="gho-chip" x="20" y="20" width="225" height="50" rx="8"/>
<rect class="gho-chip" x="265" y="20" width="225" height="50" rx="8"/>
<rect class="gho-chip" x="510" y="20" width="230" height="50" rx="8"/>
<text class="gho-lbl" x="132" y="50">1 - copy rows in chunks</text>
<text class="gho-lbl" x="377" y="50">2 - tail the binlog</text>
<text class="gho-lbl" x="625" y="50">3 - atomic cutover</text>
<rect class="gho-tbl" x="40" y="150" width="180" height="120" rx="10"/>
<rect class="gho-tbl" x="540" y="150" width="180" height="120" rx="10"/>
<text class="gho-lbl" x="130" y="200">orders</text>
<text class="gho-sub" x="130" y="222">live reads + writes</text>
<text class="gho-lbl" x="630" y="200">_orders_gho</text>
<text class="gho-sub" x="630" y="222">new schema</text>
<circle class="gho-row gho-mv" cx="226" cy="178" r="9"/>
<circle class="gho-row gho-mv gho-d2" cx="226" cy="178" r="9"/>
<circle class="gho-row gho-mv gho-d3" cx="226" cy="178" r="9"/>
<circle class="gho-binlog gho-bmv" cx="226" cy="242" r="7"/>
<circle class="gho-binlog gho-bmv gho-bd2" cx="226" cy="242" r="7"/>
<rect class="gho-row" x="372" y="300" width="14" height="14" rx="3"/>
<text class="gho-sub" x="470" y="312">row chunks copied</text>
<circle class="gho-binlog" cx="565" cy="307" r="7"/>
<text class="gho-sub" x="650" y="312">binlog events</text>
</svg>
<figcaption>gh-ost builds a ghost table, copies existing rows in throttled chunks while replaying live writes from the binary log, then renames the ghost in with a sub-second atomic cutover.</figcaption>
</figure>

The three activities run concurrently:

1. **Chunked copy.** `gh-ost` creates an empty *ghost table* named `_orders_gho` with the new schema, then copies existing rows from `orders` into it in small chunks (default 1,000 rows), one chunk at a time, pausing between chunks if the server is under load.
2. **Binlog catch-up.** While the copy runs, real writes keep hitting `orders`. `gh-ost` reads those writes from the binary log and applies the equivalent change to `_orders_gho`, so the ghost stays current with live traffic.
3. **Atomic cutover.** When the copy has caught up and the binlog tail is drained, `gh-ost` performs an atomic swap: it renames `orders` out of the way and `_orders_gho` into its place. The cutover holds a brief table lock for a fraction of a second, using a clever sentry-table-and-blocked-rename trick so that no writes are lost and the swap is genuinely atomic.

Because it reads the binlog instead of installing triggers, `gh-ost` adds *no write-path overhead* to your live table — your `INSERT`s and `UPDATE`s do not pay a trigger tax. And because it is an external process polling the binlog, it is fully **observable and controllable**: you can watch its progress, throttle it on demand, pause it, and even postpone the cutover indefinitely until you are ready. Here is a realistic invocation, explained line by line:

```bash
gh-ost \
  --host="replica-1.db.internal" \      # connect to a REPLICA, not the primary
  --port=3306 \
  --user="gh-ost" --password="$GHOST_PW" \
  --database="shop" \
  --table="orders" \
  --alter="ADD COLUMN fulfilled_at TIMESTAMP NULL, ADD INDEX idx_fulfilled (fulfilled_at)" \
  --assume-rbr \                          # binary log is already ROW-based
  --chunk-size=1000 \                     # rows copied per iteration
  --max-load="Threads_running=25" \       # pause copying if the server is busy
  --critical-load="Threads_running=100" \ # abort entirely if it gets this bad
  --max-lag-millis=1500 \                 # throttle if replica lag exceeds 1.5s
  --throttle-control-replicas="replica-2.db.internal,replica-3.db.internal" \
  --postpone-cut-over-flag-file="/tmp/ghost.orders.postpone" \
  --serve-socket-file="/tmp/gh-ost.orders.sock" \
  --initially-drop-ghost-table \          # clean up any leftover ghost from a prior run
  --verbose \
  --execute                               # without this, it's a dry run
```

Two flags deserve emphasis. `--postpone-cut-over-flag-file` tells `gh-ost` to do all the slow work — copy and catch-up — but *stop short of the cutover* while that file exists. The migration can sit fully caught-up for hours; when you delete the flag file, it performs the sub-second swap. This lets you decouple the expensive part (which you can run anytime) from the risky part (which you do during a calm, watched window). The `--serve-socket-file` opens a Unix socket you can talk to interactively:

```bash
# pause the copy immediately
echo throttle | nc -U /tmp/gh-ost.orders.sock
# resume
echo no-throttle | nc -U /tmp/gh-ost.orders.sock
# change a setting mid-flight
echo "chunk-size=200" | nc -U /tmp/gh-ost.orders.sock
# trigger the cutover now (equivalent to deleting the postpone flag)
echo unpostpone | nc -U /tmp/gh-ost.orders.sock
# emergency stop
echo panic | nc -U /tmp/gh-ost.orders.sock
```

That interactive control is exactly what you want at 3am when the migration is fine but a traffic spike just arrived: throttle it, ride out the spike, resume. You cannot do that with a bare `ALTER`.

### pt-online-schema-change: triggers keep the shadow in sync

Percona's `pt-online-schema-change` predates `gh-ost` and uses a different, equally clever mechanism: **triggers**. It creates a shadow table `_orders_new` with the new schema, then installs three `AFTER` triggers on the original `orders` table — one each for `INSERT`, `UPDATE`, and `DELETE`. Every write to `orders` fires a trigger that replays the same change into `_orders_new`. Meanwhile a chunked copy job walks the existing rows of `orders` and inserts them into `_orders_new`. When the copy finishes, the two tables are identical, and pt-osc does the atomic rename swap.

![Triggers on the original table replay every live change into the shadow table while a chunked job copies the existing rows; an atomic rename then swaps the shadow in.](/imgs/blogs/online-schema-changes-at-scale-3.webp)

The figure shows the dataflow: application writes hit `orders`; the `AFTER` triggers fan each write out to `_orders_new`; the chunked copy job reads `orders` and bulk-loads `_orders_new`; both streams converge on the shadow, and the final `RENAME` swaps it in. A representative invocation:

```bash
pt-online-schema-change \
  --alter "ADD COLUMN fulfilled_at TIMESTAMP NULL" \
  D=shop,t=orders \
  --max-load "Threads_running=25" \
  --critical-load "Threads_running=100" \
  --chunk-time 0.5 \                       # auto-size chunks to ~0.5s each
  --max-lag 1.5 \                          # throttle if replica lag > 1.5s
  --alter-foreign-keys-method=auto \       # how to fix child FKs after the swap
  --no-drop-old-table \                    # keep _orders_old as a safety net
  --execute
```

The trigger approach has two consequences you must respect. First, **the triggers add synchronous overhead to every write** on the original table for the entire duration of the migration — each `INSERT`/`UPDATE`/`DELETE` now does extra work inside the same transaction. On a write-hot table this can be the difference between a healthy database and a saturated one, which is precisely the pain that pushed GitHub to build `gh-ost`. Second, **triggers and foreign keys interact badly**. If `orders` has child tables referencing it, the rename has to deal with those foreign keys, and pt-osc's `--alter-foreign-keys-method` options (`rebuild_constraints`, `drop_swap`, `auto`) each have failure modes — `drop_swap` is fast but briefly leaves no table under the name, and a poorly chosen method can break referential integrity or cause a longer lock. You also cannot run pt-osc on a table that already has triggers MySQL would conflict with (pre-8.0, only one trigger per event was allowed).

### Choosing between them

| Dimension | gh-ost | pt-online-schema-change |
| --- | --- | --- |
| Sync mechanism | Reads the binary log (triggerless) | `AFTER` triggers on the original table |
| Write-path overhead | None on the live table | Trigger cost on every write, for the whole run |
| Where it runs | Connects to a replica; near-zero primary load | Runs against the primary; triggers live there |
| Observability / control | Rich: live progress, throttle, pause, postpone cutover | More limited; runs to completion |
| Requirements | `ROW`-based binlog; a reachable replica is ideal | Triggers allowed; careful FK handling |
| Maturity | Newer, very widely used at large MySQL shops | Older, extremely battle-tested, broad edge-case coverage |
| Best when | Large, write-hot tables where trigger overhead hurts | Simpler setups, or when you cannot read the binlog |

The default at most large MySQL shops today is `gh-ost` for big, busy tables, precisely because moving the sync work off the write path and onto a binlog reader on a replica is the difference between a migration the table notices and one it does not. But pt-osc remains the right tool when you cannot give a tool binlog access, or when its longer track record of edge-case handling matters more than the trigger overhead. Both, critically, throttle on replica lag — which connects directly to [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas): a migration that ignores replica lag will quietly poison your read fleet.

## Postgres online changes: native tools and patterns

Postgres has no single dominant "gh-ost equivalent," partly because so many operations are already cheap and partly because the patterns are baked into SQL. The skill is knowing which cost class an operation falls into and reaching for the right native tool when an operation would otherwise rewrite or lock.

![Postgres operations split into metadata-only, concurrent-build, full-rewrite, and constraint-validate classes, each with very different lock costs.](/imgs/blogs/online-schema-changes-at-scale-4.webp)

The tree sorts the common operations into four cost classes. The left branch (metadata-only) is free and needs no ceremony. The other three branches are where the technique lives.

### Build indexes with CREATE INDEX CONCURRENTLY

A plain `CREATE INDEX` takes a lock that blocks **writes** to the table for the entire build — on a large table that is a long write outage. The fix is one keyword:

```sql
CREATE INDEX CONCURRENTLY idx_orders_customer ON orders (customer_id);
```

`CONCURRENTLY` builds the index using only a `SHARE UPDATE EXCLUSIVE` lock, which does not block reads or writes. It pays for this by doing **two passes** over the table and waiting for in-flight transactions, so it is slower in wall-clock time — but it does not take the table offline. The catch: it **cannot run inside a transaction block**, and if it fails partway (a deadlock, a conflicting transaction, a cancelled session) it leaves behind an *invalid* index that still costs write overhead but is not used for queries. So the operational pattern is: run it outside a transaction, then verify and clean up:

```sql
-- after a CREATE INDEX CONCURRENTLY, check for invalid leftovers
SELECT c.relname
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
WHERE i.indisvalid = false;

-- if found, drop it (also concurrently) and retry the build
DROP INDEX CONCURRENTLY idx_orders_customer;
```

The same applies to `REINDEX CONCURRENTLY` (Postgres 12+) when you need to rebuild a bloated index without locking out writes. This pairs directly with [index strategy at scale](/blog/software-development/database-scaling/index-strategy-at-scale): the index you want is only half the problem; building it without an outage is the other half.

### The lock_timeout + retry pattern

The lock-queue pileup we diagrammed earlier is preventable with one habit: **never let a DDL statement wait in the lock queue**. Set a short `lock_timeout` so the statement either gets its lock almost immediately or fails fast, and wrap it in a retry loop so a transient conflict just means "try again in a few seconds" instead of "stall the whole database."

```python
import time
import psycopg
from psycopg import errors

DDL = "ALTER TABLE orders ADD COLUMN fulfilled_at timestamptz"

def run_ddl_with_retries(dsn, ddl, attempts=25, lock_timeout="2s", base_backoff=5):
    for attempt in range(1, attempts + 1):
        try:
            with psycopg.connect(dsn, autocommit=False) as conn:
                with conn.cursor() as cur:
                    # SET LOCAL applies only to this transaction; if the lock
                    # cannot be taken within 2s, Postgres raises LockNotAvailable
                    # instead of queuing behind every other waiter.
                    cur.execute(f"SET LOCAL lock_timeout = '{lock_timeout}'")
                    cur.execute(ddl)
                conn.commit()
            print(f"DDL succeeded on attempt {attempt}")
            return
        except errors.LockNotAvailable:
            wait = min(base_backoff * attempt, 60)
            print(f"attempt {attempt}: lock not available, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not acquire lock for DDL after {attempts} attempts")
```

This single pattern is the most important operational safeguard in the entire post, because it converts the worst failure mode — a stuck `ALTER` taking down the site — into a benign one: the `ALTER` politely fails and retries until it finds a gap with no long-running transaction in front of it. The Rails ecosystem bakes this into the `strong_migrations` and `online_migrations` gems; Postgres-savvy teams set a low `lock_timeout` (and a `statement_timeout`) on the migration role by default. If you take one habit away from this article, take this one.

### pg_repack for rewrites you cannot avoid

Some operations genuinely require rewriting the table — reclaiming bloat, or changing storage in ways no metadata trick can finesse. `pg_repack` is an extension that does this online: it builds a fresh copy of the table, uses a log table and triggers to capture concurrent changes (conceptually similar to pt-osc), and swaps the rebuilt table in, taking only a brief `ACCESS EXCLUSIVE` lock at the very end for the swap rather than for the whole rebuild. It is the Postgres answer to "I need to rewrite a huge table without locking it for an hour."

![An in-place rewrite holds an ACCESS EXCLUSIVE lock for the whole operation, while pg_repack builds a packed copy alongside the live table, replays live changes through a log table, and takes the exclusive lock only at the brief final swap.](/imgs/blogs/online-schema-changes-at-scale-8.webp)

The contrast above is the whole reason `pg_repack` exists. A plain `VACUUM FULL` or an in-place `ALTER COLUMN ... TYPE` rebuilds the table while holding `ACCESS EXCLUSIVE` for the entire rebuild — minutes to hours of total blocking on a large table. `pg_repack` instead builds the packed copy off to the side with no exclusive lock, keeps it current through a log table fed by triggers, and only grabs the strong lock for the sub-second swap at the end. For a type change that requires a rewrite, the cleaner pattern is usually to avoid the in-place rewrite entirely and instead add a new column and migrate to it — which is exactly the expand-contract pattern we are about to reach.

### pgroll and Reshape: expand-contract as a tool

Two newer tools encode the expand-contract pattern directly into Postgres. `pgroll` (from Xata) and `Reshape` both let you declare a schema change and have the tool manage the safe, reversible rollout. `pgroll` is especially worth understanding: it keeps *both* the old and new schema versions live simultaneously by exposing them as **versioned views**, so old application instances read the old shape and new instances read the new shape during the transition. When a change requires a breaking column edit, `pgroll` adds the new physical column, backfills it in batches, and installs triggers that keep the old and new columns in sync via `up`/`down` expressions you provide:

```json
{
  "name": "01_rename_email_column",
  "operations": [
    {
      "alter_column": {
        "table": "users",
        "column": "email",
        "name": "email_address",
        "up": "email",
        "down": "email_address"
      }
    }
  ]
}
```

```bash
pgroll start  01_rename_email_column.json   # expand: add view version, triggers, backfill
# ... deploy app pointing at the new view version, verify ...
pgroll complete                              # contract: drop old column, triggers, old view
# or, if something is wrong:
pgroll rollback                              # tear the expand back out, no data lost
```

`pgroll start` performs the expand phase — it does the additive, non-breaking changes and sets up dual-write triggers — and `pgroll complete` performs the contract phase. In between, both schema versions work, so you can roll your application forward (or back) without the database and the app ever disagreeing. That coexistence is the whole point of expand-contract, which we now make explicit.

## Expand-contract: the pattern that ties it all together

Every tool above solves the *physical* problem — change the table without a long lock. But there is a second, equally important problem: the *application* expects a particular schema, and you cannot atomically update the database and every running instance of your code at the same moment. During a deploy there is always a window where old code and new code run simultaneously against one database. If the schema change is breaking — you renamed a column the old code still selects — that window is an outage even if the `ALTER` itself was instant.

Expand-contract (also called parallel change) solves this by never making a breaking change. Instead you decompose every schema change into a sequence of **individually backward-compatible, reversible steps**, and you sequence them with your deploys so that at no point does any running version of the code see a schema it cannot handle.

![Expand-contract changes schema in backward-compatible steps so the running application never sees a breaking change: add, dual-write, backfill, switch reads, verify, then drop.](/imgs/blogs/online-schema-changes-at-scale-5.webp)

The figure lays out the six steps. The key property is that you can stop, pause, or reverse at any step, and the system is always in a consistent state where both old and new code work. Let us make it concrete with the most common awkward change: renaming a column.

### Worked example: renaming `email` to `email_address`

A column rename feels trivial and is one of the most dangerous changes you can make naively, because `RENAME COLUMN` is atomic in the database but *not* atomic with your deploy: the instant you rename, every running instance of the old code that does `SELECT email` breaks. Here is the same rename done safely.

**Step 1 — Expand (add the new column).** This is metadata-only in Postgres and instant.

```sql
ALTER TABLE users ADD COLUMN email_address text;
```

**Step 2 — Deploy code that dual-writes.** Ship a release that writes *both* columns on every change but still reads the old one. Old instances (still writing only `email`) and new instances (writing both) coexist fine, because `email` is still the source of truth.

```python
# Release N: write both columns, read the OLD one
def set_email(user, value):
    user.email = value             # old column — still the source of truth
    user.email_address = value     # new column — kept in sync
    repo.save(user)

def get_email(user):
    return user.email              # reads still come from the old column
```

**Step 3 — Backfill existing rows in throttled batches.** New writes populate `email_address`, but old rows are still `NULL` there. Backfill them in small, committed batches so you never hold a long transaction or a big lock, and so replicas can keep up.

```python
last_id = 0
BATCH = 5000
while True:
    rows = repo.execute("""
        UPDATE users
        SET email_address = email
        WHERE id > %(lo)s AND id <= %(hi)s
          AND email_address IS DISTINCT FROM email
        RETURNING id
    """, {"lo": last_id, "hi": last_id + BATCH})
    if not rows:
        break
    last_id += BATCH
    time.sleep(0.05)   # throttle: give replicas and autovacuum room to breathe
```

The `IS DISTINCT FROM` guard makes the backfill **idempotent** — you can re-run it safely, and it only touches rows that actually need updating, which keeps [write load and dead-tuple churn](/blog/software-development/database-scaling/keeping-postgres-healthy-under-write-load) down.

**Step 4 — Migrate reads.** Once the backfill is complete and verified, ship a release that reads from `email_address`. Still keep dual-writing for now, so you can roll back this release without losing data.

```python
# Release N+1: still dual-write, but read the NEW column
def get_email(user):
    return user.email_address      # reads now come from the new column
```

**Step 5 — Verify.** Run a reconciliation query to confirm the two columns agree before you throw away the old one. This is cheap insurance.

```sql
SELECT count(*) AS mismatches
FROM users
WHERE email IS DISTINCT FROM email_address;
-- expect 0 before proceeding to contract
```

**Step 6 — Contract.** Ship a release that stops writing `email`, then drop it. `DROP COLUMN` is metadata-only in Postgres.

```python
# Release N+2: write and read only the new column
def set_email(user, value):
    user.email_address = value
    repo.save(user)
```

```sql
ALTER TABLE users DROP COLUMN email;
```

At no point in that sequence does any running version of the application see a schema it cannot handle, and at every step you can halt and reverse. That is the entire value proposition: you have traded one risky atomic change for a handful of boring, reversible ones.

### The same shape handles every awkward change

Once you see expand-contract, you start applying it to every change that would otherwise be a breaking one. The pattern does not change; only the per-phase SQL does.

![The same four-phase pattern handles a rename, a type change, a NOT NULL addition, and a column split without downtime — every awkward change reduces to add-new, backfill, switch-reads, drop-old.](/imgs/blogs/online-schema-changes-at-scale-6.webp)

The matrix shows four common changes flowing through the same phases:

- **Rename** `email` to `email_address`: add the new column, copy old to new, switch reads, drop old. (Our worked example.)
- **Change type** `id int` to `id bigint` (the classic integer-overflow scramble): add `id_v2 bigint`, backfill `id` to `id_v2`, dual-write and switch reads to `id_v2`, then drop `id` and rename `id_v2`. Renaming an int primary key to bigint in place would rewrite the whole table and every foreign key; expand-contract makes it a background backfill. This is the same fight described in [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) when teams migrate key types.
- **Add NOT NULL** to `email`: this one has a Postgres-specific trick we detail next.
- **Split** `full_name` into `first_name`, `last_name`: add the two new columns, parse `full_name` to populate them, switch reads, drop `full_name`.

### The NOT NULL trick worth memorizing

Adding a `NOT NULL` constraint to an existing column on a large table historically meant a full-table scan under `ACCESS EXCLUSIVE` to verify there are no nulls — an outage. Postgres 12+ gives you a clean three-step path that never takes a long lock, because `SET NOT NULL` will *trust* an already-validated `CHECK` constraint instead of re-scanning:

```sql
-- 1. add a CHECK constraint NOT VALID — instant, only checks future rows
ALTER TABLE users
  ADD CONSTRAINT users_email_not_null CHECK (email IS NOT NULL) NOT VALID;

-- 2. fix any existing NULL rows (backfill), then VALIDATE the constraint.
--    VALIDATE takes only SHARE UPDATE EXCLUSIVE — it scans, but does NOT block writes.
UPDATE users SET email = '' WHERE email IS NULL;          -- or a real backfill
ALTER TABLE users VALIDATE CONSTRAINT users_email_not_null;

-- 3. SET NOT NULL is now cheap: Postgres trusts the validated CHECK (PG 12+)
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- 4. drop the now-redundant CHECK constraint
ALTER TABLE users DROP CONSTRAINT users_email_not_null;
```

The same `NOT VALID` then `VALIDATE` two-step works for adding foreign keys to large tables: `ADD CONSTRAINT ... REFERENCES ... NOT VALID` is instant and enforces the FK on new rows, and a later `VALIDATE CONSTRAINT` checks the existing rows under a weak lock.

## Safe-migration discipline

The tools and patterns above only protect you if you wrap them in operational discipline. These are the habits that separate teams that migrate continuously without anyone noticing from teams that schedule maintenance windows and still cause outages.

**Set `lock_timeout` and `statement_timeout` on every migration.** As shown above, a short `lock_timeout` is the single highest-leverage safeguard — it makes a stuck DDL fail fast instead of stalling the world. A `statement_timeout` caps the damage of a backfill or rewrite that runs longer than expected. Make these defaults on your migration database role, not something each migration remembers to set.

**Throttle on replica lag, always.** Whether it is `gh-ost`'s `--max-lag-millis`, pt-osc's `--max-lag`, or a `time.sleep()` in your backfill loop, every bulk operation must back off when replicas fall behind. A migration that saturates replication does not just slow itself down — it makes your read replicas serve stale data and can drop them out of rotation, turning a schema change into a read-availability incident. This is the operational link to [the slow query at 1am](/blog/software-development/database-scaling/the-slow-query-at-1am): the migration backfill *is* a slow query, and it deserves the same lag awareness.

**Batch everything, commit often.** Never run an unbounded `UPDATE ... SET col = ...` across a huge table — it is one giant transaction that holds locks and dead tuples for its entire duration and can blow out your write-ahead log. Always batch by primary-key range, commit each batch, and sleep between them.

**Prefer off-peak, but design for any time.** Running migrations during low-traffic windows reduces contention, but the goal is migrations so safe you *could* run them at peak. If a migration is only safe at 3am, it is not actually safe — it is just less likely to be caught.

**Make it observable.** Watch lock waits (`pg_locks`, `SHOW ENGINE INNODB STATUS`), replica lag, error rates, and the tool's own progress output during every migration. The `--postpone-cut-over-flag-file` discipline — do the slow work anytime, do the cutover under active observation — exists precisely so a human is watching the risky moment.

**Lint your migrations in CI.** Tools like `strong_migrations` (Rails), `squawk` (a Postgres migration linter), and `online_migrations` reject dangerous patterns — a bare `CREATE INDEX`, an `ADD COLUMN` with a volatile default, a `SET NOT NULL` without the `CHECK` dance — before they ever reach production. Encoding this knowledge in a linter is how you scale safe migrations beyond the few engineers who have been burned.

### The comparison table

Here is the whole landscape in one view — the question is always "what does this approach do to locks, replicas, and reversibility, and when do I reach for it?"

| Approach | Holds a blocking lock? | Replica-safe? | Reversible? | Reach for it when |
| --- | --- | --- | --- | --- |
| Naive `ALTER` | Yes — for the whole operation | No — replays serially, lags the fleet | No | The table is small, or the op is genuinely `INSTANT`/metadata-only |
| MySQL online DDL (`INPLACE`/`INSTANT`) | Brief metadata lock only | Often no — still replays serially | No | A small, well-understood op on MySQL with no long transactions around |
| `gh-ost` | No — sub-second cutover only | Yes — reads binlog off a replica, throttles on lag | Yes — abort before cutover | Large, write-hot MySQL tables; you want control and observability |
| `pt-online-schema-change` | No — sub-second cutover only | Yes — throttles on lag | Yes — keep `_old` table | MySQL where you cannot read the binlog; simpler setups |
| Postgres `CONCURRENTLY` + `lock_timeout` retry | No — weak locks, fails fast | Yes | Mostly — drop the index/constraint | Native Postgres index builds and constraint validations |
| Expand-contract (parallel change) | No — each step is non-breaking | Yes — backfills are throttled | Yes — halt or reverse at any step | Any change that would break running code: rename, type, NOT NULL, split |

The bottom row is the meta-pattern: even when you use `gh-ost` or `pgroll` for the physical change, you still wrap it in expand-contract at the application layer, because the tools change the table safely and expand-contract changes the *contract between the table and your code* safely. You need both.

## War stories from production

### 1. The "quick ALTER" that queued behind a report

A team needs to add a nullable column to `orders` — genuinely a fast operation. They run it at 2am against Postgres. What they do not know is that a nightly analytics job is mid-flight, holding an `AccessShareLock` on `orders` for a long aggregation. Their `ALTER` needs `AccessExclusive`, so it waits. Behind it, the entire stream of customer traffic — every `SELECT` and `UPDATE` touching `orders` — queues up. Within two minutes the connection pool is exhausted, request threads are blocked on the pool, and checkout returns 5xx.

![A single ALTER that grabbed an exclusive lock on a large table cascaded into a twenty-minute customer-facing outage; the lock was instant to request and impossible to escape.](/imgs/blogs/online-schema-changes-at-scale-7.webp)

The timeline shows the cascade: the `ALTER` deploys at 02:14, the lock queue forms by 02:15, the pool saturates by 02:17, customers see errors by 02:21, and it only clears at 02:34 when the analytics job finally finishes and releases its lock — letting the `ALTER` run (in milliseconds) and the queue drain. The fix was not "don't add columns at 2am." The fix was `SET lock_timeout = '2s'` on the migration plus the retry wrapper: with that in place, the `ALTER` would have failed at 02:16, retried, and slipped in cleanly the moment the report finished — with zero customer impact. One `SET LOCAL` would have turned a twenty-minute outage into a non-event.

### 2. Why GitHub built gh-ost

GitHub ran enormous, write-heavy MySQL tables and relied on `pt-online-schema-change` for years. The problem was the triggers: on their busiest tables, the synchronous trigger overhead on every write — during migrations that could run for hours — added real latency and load to the primary at exactly the moment they could least afford it, and the lack of fine-grained control (pause, throttle, postpone) meant a migration that started misbehaving was hard to rein in. They built `gh-ost` to move the change-capture work off the write path entirely (read the binlog instead of firing triggers) and to make migrations interactively controllable. The lesson generalizes: the cost of a migration tool is not just the cutover — it is everything the tool does to your write path for the *hours* in between, and trigger overhead on a hot table is a tax you pay on every transaction.

### 3. The integer primary key that almost overflowed

A fast-growing service stored its primary key as a signed 32-bit `int`, which caps at roughly 2.1 billion. Growth projections showed they would hit that ceiling in months, at which point every `INSERT` would fail. Changing the column type from `int` to `bigint` in place would rewrite the entire table and every index and foreign key referencing it — an unthinkable lock on their largest table. They used expand-contract: add `id_v2 bigint`, backfill it from `id` in throttled batches over several days, dual-write both, switch all foreign keys and reads to `id_v2`, then drop `id` and rename. The migration was invisible to users and took weeks of calendar time at near-zero risk — exactly the right trade when the alternative is a multi-hour outage on your most important table.

### 4. The CREATE INDEX that locked writes

An engineer adds an index to speed up a slow query (the right instinct — see [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer)). They run `CREATE INDEX idx_orders_status ON orders(status)` in a production Postgres console. On a 200 GB table the build takes 25 minutes, and a plain `CREATE INDEX` takes a lock that blocks writes for the whole build. Orders cannot be created for 25 minutes. The fix is one word — `CREATE INDEX CONCURRENTLY` — which trades a longer build for no write lock. The deeper fix is a CI linter that rejects a non-concurrent `CREATE INDEX` in a migration so the mistake cannot be merged.

### 5. The migration that throttled itself for three days

A team kicks off a `gh-ost` migration on a hot table with aggressive lag protection (`--max-lag-millis=500`) and a small `--max-load`. The table is so write-busy that `gh-ost` spends most of its time throttled — every time it copies a few chunks, replica lag creeps up and it backs off. The migration that they expected to take hours takes three days, blocking the feature waiting on it. Nothing was *wrong* — the throttling did its job and protected the replicas — but the lesson is that on a write-hot table you must tune `chunk-size`, `max-lag`, and `max-load` against real throughput, run the slow copy with `--postpone-cut-over-flag-file` so the long runtime does not also pin the cutover window, and budget calendar time accordingly. Online migrations are safe, not instant.

### 6. The foreign-key swap that orphaned children

A team runs `pt-online-schema-change` on a parent table that has several child tables with foreign keys pointing at it. They use the default foreign-key handling without thinking it through. When pt-osc renames the shadow table into place, the child foreign keys must be pointed at the new table; the method that briefly drops and recreates them leaves a window where referential integrity is not enforced, and a concurrent write slips an orphaned child row through. The lesson: when a table has child foreign keys, the rename step is not a detail — choose `--alter-foreign-keys-method` deliberately, understand its lock and integrity trade-offs, and prefer designs (or `gh-ost`) that minimize the foreign-key juggling. Foreign keys make the cutover the hard part.

### 7. The Rails `add_column` with a default that rewrote the table

On an older Postgres version, a developer writes a migration that adds a column with a default value: `add_column :users, :status, :string, default: "active"`. On that version, a non-constant default (or any default at all, pre-11) rewrites every row under `ACCESS EXCLUSIVE`. On the `users` table — tens of millions of rows — this locks the table for minutes during a routine deploy. The modern fix is to split it: add the column with no default (instant), set the default separately (metadata-only on Postgres 11+ for constants), and backfill existing rows in batches. This exact footgun is why `strong_migrations` exists and why it refuses this pattern with a message telling you the safe three-step version.

### 8. The idle-in-transaction connection that blocked a migration

A migration to add a constraint hangs indefinitely. The `ALTER` is not slow — it is waiting for a lock it cannot get, because somewhere a connection is sitting `idle in transaction`: an application opened a transaction, ran one query, and never committed (a bug in connection handling). That open transaction holds a lock on the target table, the `ALTER` queues behind it, and — per the lock-queue rule — everything queues behind the `ALTER`. The immediate fix is to find and kill the idle transaction (`SELECT pid FROM pg_stat_activity WHERE state = 'idle in transaction'` and `pg_terminate_backend(pid)`); the systemic fix is an `idle_in_transaction_session_timeout` so leaked transactions cannot hold locks forever, plus the `lock_timeout` retry wrapper so the migration never becomes the head of a doom queue in the first place.

## When to reach for each tool, and when not to

**Reach for a bare `ALTER` when:**

- The table is small enough that a brief lock is genuinely harmless.
- The operation is metadata-only: `ADD COLUMN` with no/constant default, `DROP COLUMN`, `RENAME` on Postgres; an `INSTANT` operation on MySQL 8.0.12+.
- You have a short `lock_timeout` set so a stuck statement fails fast rather than stalling the world — make this true even for "safe" ALTERs.

**Reach for `gh-ost` or `pt-online-schema-change` when:**

- You are on MySQL and the table is large enough that a rewrite would lock it for too long.
- The operation is a real rewrite — type change, adding an index on a huge table, reordering columns.
- Prefer `gh-ost` for large, write-hot tables and when you want pause/throttle/postpone control; prefer `pt-osc` when you cannot read the binlog or want its long edge-case track record.

**Reach for Postgres-native patterns (`CONCURRENTLY`, `NOT VALID`/`VALIDATE`, `pg_repack`, `pgroll`) when:**

- You are on Postgres — most changes are metadata-only or have a weak-lock path, so you rarely need an external copy-and-cutover tool.
- You need an index without locking writes (`CONCURRENTLY`), a constraint without a full scan (`NOT VALID` then `VALIDATE`), or a managed expand-contract rollout (`pgroll`).

**Always wrap physical changes in expand-contract when:**

- The change would break running application code — any rename, type change, NOT NULL addition, or column split. There is no exception here at scale: even an instant `RENAME COLUMN` needs the parallel-change choreography, because the database change and the deploy are never atomic together.

**Skip the heavy machinery when:**

- You are early-stage with small tables and brief maintenance windows are genuinely acceptable — do not build a migration pipeline before you have a table big enough to need one.
- The change is purely additive and read by no old code yet (a brand-new column nothing reads) — though even then, a short `lock_timeout` costs nothing.
- You are tempted to "just run it during the maintenance window." At scale, the maintenance window is the anti-pattern: the goal is migrations so safe that no window is needed, because the window itself is downtime you scheduled.

The throughline is that schema change at scale is not a database operation you run, it is a process you choreograph. The naive `ALTER` treats the change as atomic and instantaneous, and at scale it is neither — it is a lock that outlives your patience and a contract change your running code did not agree to. `gh-ost`, pt-osc, and the Postgres-native patterns make the physical change safe; expand-contract makes the application change safe; and a short `lock_timeout` with a retry makes the whole thing fail gracefully when the database is busy. Put those three together and you can change the shape of a hundred-gigabyte table, in the middle of peak traffic, and have no one notice — which is exactly how it should feel.

## Further reading

- [Zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — the application-deploy choreography in depth.
- [Resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) — the same shadow-copy-and-cutover shape, applied to moving data across shards.
- [Keeping Postgres healthy under write load](/blog/software-development/database-scaling/keeping-postgres-healthy-under-write-load) — why throttled batches and dead-tuple awareness matter during backfills.
- [The slow query at 1am](/blog/software-development/database-scaling/the-slow-query-at-1am) — the replica-lag and contention mindset your backfills must share.
- The `gh-ost`, `pt-online-schema-change`, `pg_repack`, and `pgroll` project documentation, plus the `strong_migrations` and `squawk` linters, for the exact flags and edge cases.

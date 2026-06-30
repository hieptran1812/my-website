---
title: "Index Strategy at Scale: When More Indexes Start to Hurt"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "At write scale every index flips from a free read speedup into a real tax on writes, storage, and contention — the craft is finding the minimal set that covers your actual queries."
tags: ["database-scaling", "indexing", "postgresql", "b-tree", "query-optimization", "write-amplification", "covering-index", "database-performance", "performance-tuning", "sql"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 36
---

The most common piece of database advice in the world is also the most dangerous when you take it too far: "your query is slow, add an index." It is true so often, and so cheaply, that it becomes a reflex. A dashboard times out, someone adds an index, the dashboard is fast again, everyone moves on. Multiply that reflex by three years, four teams, two ORMs that auto-create indexes on every foreign key, and a culture where nobody is ever blamed for adding an index but plenty of people get paged for a slow query — and you arrive at the table I have been called in to look at more than once: a hot write table with fourteen indexes on it, where the indexes collectively take more disk than the table, where a single `UPDATE` of one row generates a dozen index writes plus the WAL records to replicate them, and where the write throughput has quietly dropped by half over eighteen months without any single change being the obvious culprit.

That is the failure mode this post is about. Indexes are not free, and at scale they are not even cheap. Every index you add is a standing tax on every write to that table, extra storage that has to be cached and backed up, additional lock and latch surface during maintenance, and one more option the query planner has to consider on every query. For a read-mostly table with a handful of indexes, the read speedup dwarfs all of that and you should not think twice. For a high-write table, the arithmetic inverts, and the discipline that separates a fast system from a slow one is not knowing how to add an index — everyone knows that — it is knowing which indexes to *not* have.

![The index ledger: a read becomes a log-time seek (the win) while every write must maintain all indexes (the tax)](/imgs/blogs/index-strategy-at-scale-1.webp)

The diagram above is the mental model for the whole post: an index is a ledger with an entry on both sides. On the read side, it turns an `O(n)` table scan into an `O(log n)` seek — three or four page hops down a B-tree, one buffer read, sub-millisecond. That is the entry everyone sees. On the write side, every `INSERT`, `UPDATE`, and `DELETE` that touches an indexed column must now also maintain that index: find the right leaf, write the new entry, possibly split a page, and log it all to the write-ahead log so replicas and crash recovery can replay it. With one index that is a rounding error. With fourteen, the write side of the ledger is where your throughput goes to die. The job is to keep the read-side entries you actually cash in and ruthlessly cancel the write-side charges nobody is collecting on.

This post assumes you have read [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) and have a system under enough load that write cost matters — if you are at a thousand writes a second across the whole database, none of this will move your needle and you should go add the index. It also leans on three companion pieces it will not re-derive: [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans) for the mechanics of multi-column indexes, [Postgres special indexes — GIN, GiST, BRIN, and partial](/blog/software-development/database/postgres-special-indexes-gin-gist-brin-partial) for the non-B-tree access methods, and [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) for interpreting the plans I will show. Examples are PostgreSQL because its statistics views make the costs visible, but the tradeoff is universal — MySQL, SQL Server, and every other engine pay the same write tax, sometimes worse.

## Why "just add an index" stops being free

The reflex is built on a set of assumptions that hold at small scale and quietly break at large scale. Naming them is the first step to escaping the reflex.

| Assumption (true at small scale) | Reality at write scale |
| --- | --- |
| "An index only helps; worst case it is unused and harmless." | Every index is maintained on every write, used or not. An unused index is pure cost. |
| "Indexes are tiny compared to the table." | Indexes routinely sum to 1–3× the table's heap size; on wide-key or many-index tables they dominate disk and cache. |
| "Adding one more index is negligible." | Write cost is roughly linear in index count. The fifth index costs as much per write as the first. |
| "The planner will always pick the best index." | More indexes mean a larger plan search space and more chances to pick a worse plan; some indexes are never chosen at all. |
| "CREATE INDEX is a quick admin task." | A plain `CREATE INDEX` takes a lock that blocks writes for the entire build — minutes to hours on a large hot table. |
| "If a query is slow, the fix is a new index." | Often the fix is a *better* index that lets you drop two existing ones, or a query rewrite — not a net addition. |

None of these assumptions are wrong in the small. That is exactly why the reflex is so durable: it is reinforced every single time you use it on a small table, and the cost only shows up in aggregate, slowly, on the busiest tables — the precise place where you can least afford it and where it is hardest to attribute. The rest of this post is a tour of where the costs hide and how to find the minimal index set that pays for itself.

## 1. The core tradeoff: what every index actually costs

> A senior rule of thumb: before you add an index, say out loud what every write to this table will now have to do that it did not have to do before. If you cannot answer, you do not understand the cost yet.

An index buys you read speed and charges you on four separate accounts. Most engineers track only the first one, and even that they track only as "it got faster," not as a quantity.

**Write amplification.** This is the big one, and it gets its own section later. Every mutation to a row must propagate to every index that references that row. In PostgreSQL's MVCC model the situation is sharper than people expect: an `UPDATE` does not edit a row in place, it writes a *new* version of the row (a new tuple at a new physical location), and unless the update qualifies for the heap-only-tuple optimization, every index must get a new entry pointing at the new tuple. So an `UPDATE` of a single column on a table with eight indexes can generate eight index insertions even though you only changed one field. The write you wrote is small; the write the database performs is eight times larger.

**Storage and cache pressure.** A B-tree index on a `bigint` column is not small — it stores the key plus a six-byte tuple pointer per row, plus internal pages, and it does not pack to 100%. A realistic secondary index runs 30–60% of the heap size, and a wide composite or covering index can exceed the heap. The number that actually bites is not disk (disk is cheap) but RAM: your working set has to fit in `shared_buffers` and the OS page cache, and every index page competing for cache is a heap page or a useful index page that got evicted. I have watched a buffer cache hit ratio fall from 99.5% to 96% purely from index bloat, and that 3.5% turned into a flood of random read I/O that took the disk subsystem from bored to saturated.

**Lock and latch contention.** Index maintenance is not free of concurrency cost. Inserting into a B-tree can require latching and, when a page fills, splitting it — and if many writers are inserting ascending keys, they all contend on the same rightmost leaf page (the classic "right-hand-side" hotspot). Building or rebuilding an index takes table-level locks. The more indexes, the more of this surface you have, and the more your write path is gated on internal contention rather than raw I/O.

**Planner overhead.** Every index on a table is a candidate the optimizer must consider when planning a query against that table. With a handful, the planning cost is invisible. With dozens — and I have seen tables with thirty-plus indexes accreted over years — planning time becomes measurable, and worse, the planner occasionally talks itself into a worse plan because a tempting-looking but unhelpful index changed its cost estimates. Indexes you never query still slow down the queries you do run.

The asymmetry to internalize: the read benefit accrues only to the specific queries whose shape the index matches, while the write cost is paid by *every* write to the table regardless of which queries run. On a read-heavy table that asymmetry favors indexes overwhelmingly. On a write-heavy table it is a tax with no offsetting service, and the only sane response is to audit which indexes are actually earning their charge.

## 2. A fast refresher on index types (and when each earns its keep)

> A senior rule of thumb: the index *type* is a smaller decision than people think — 90% of indexes should be B-trees — but the few times you reach for GIN, BRIN, or partial, it is because the B-tree would be catastrophically wrong, so know the four cases cold.

You cannot reason about index cost without knowing what the different access methods cost. Here is the working set, with the dimension that actually drives the minimal-set decision — write cost — called out alongside what each is good for.

![Index types by what they serve, what they cost on writes, and how much space they take](/imgs/blogs/index-strategy-at-scale-2.webp)

The figure is the quick lookup; the table below is the version you can copy into a design doc.

| Index type | Good for | Write cost | Size | Gotcha |
| --- | --- | --- | --- | --- |
| **B-tree** | equality, range, `ORDER BY`, prefix `LIKE 'abc%'` | ~1× per indexed write | moderate | the default; if you are not sure, this is the answer |
| **Composite B-tree** | multi-column filters, `WHERE a=? AND b=?` | ~1× but wider entries | larger (more bytes/row) | leftmost-prefix rule: `(a,b,c)` serves `(a)` and `(a,b)`, not `(b)` alone |
| **Covering / INCLUDE** | index-only scans, avoiding heap fetches | ~1× but much wider entries | large (carries payload columns) | bloats fast under updates; only the leaf carries the included columns |
| **Partial** | a hot subset, `WHERE status='active'` | proportional to subset, not table | tiny | the predicate must match the query's `WHERE` for the planner to use it |
| **GIN** | `jsonb`, arrays, full-text search | high, but deferred via pending list | large | `fastupdate` buffers writes; a flush can stall; sized by distinct keys, not rows |
| **GiST** | geometry, ranges, nearest-neighbor (`<->`) | moderate | moderate | lossy, may need a recheck; tuning depends on the operator class |
| **BRIN** | huge naturally-ordered tables (time-series) | near-zero | tiny (a few KB for a billion rows) | only helps when physical order correlates with the column; useless on random data |

The two rows that change architectures are BRIN and partial. **BRIN** (block range index) does not store an entry per row; it stores the min and max of each block range (by default 128 pages). On an append-only table where the column you filter on grows monotonically with insertion order — a `created_at` on an events table, say — a BRIN index on a billion rows can be a few kilobytes and is maintained at essentially zero write cost, because a new row at the tail only ever widens the last range's max. Try the same on a randomly-distributed column and it is worthless, because every range's min/max spans the whole domain and the index can prune nothing. **Partial** indexes are the single most underused tool in the box: if your queries only ever touch `WHERE status = 'pending'` and 98% of rows are `'done'`, a partial index on the 2% is fiftyfold smaller and fiftyfold cheaper to maintain than the full index, and the planner uses it transparently. We will come back to both in the targeted-patterns section. For the deeper mechanics of the non-B-tree methods, see [Postgres special indexes](/blog/software-development/database/postgres-special-indexes-gin-gist-brin-partial).

### Second-order gotcha: the leftmost-prefix rule is also a redundancy detector

The composite index rule that `(a, b, c)` can serve any query that filters on a leftmost prefix — `(a)`, `(a, b)`, or `(a, b, c)` — is usually taught as a coverage rule: make sure your composite index leads with the column you always filter on. But run it backwards and it becomes a *redundancy* rule, which is far more useful at scale. If you have a standalone index on `(a)` and you also have `(a, b)`, the standalone `(a)` is redundant: any query the planner could satisfy with `(a)` it can also satisfy with `(a, b)`, just by ignoring the trailing column. The standalone index is pure write cost with no read benefit you cannot already get. This single observation, applied across a schema, routinely identifies 20–40% of indexes as droppable, and it is the heart of the minimal-index discipline.

## 3. Covering indexes and index-only scans: the scale win that is actually worth it

> A senior rule of thumb: a covering index is the one case where making an index *bigger* makes the system *faster* — but only for the exact query it covers, and only if you keep `VACUUM` healthy.

Most index scans are not free, even when the index is perfect, because of a step people forget: after the index finds the matching entries, the database still has to go fetch the actual row from the heap to read the columns you `SELECT`ed. The index entry holds the key and a pointer; your `SELECT email, full_name` needs `email` and `full_name`, which live in the heap. So a query that matches a thousand index entries does a thousand random heap fetches — and random I/O into a large heap is exactly the access pattern that destroys cache hit ratios.

The fix is the *covering index*: an index that carries the queried columns along with the key, so the scan can answer the query entirely from the index without ever touching the heap. PostgreSQL spells this with `INCLUDE` (since version 11), which stores extra columns in the leaf pages only — they are not part of the search key, just payload riding along for the read.

![A narrow index forces a heap fetch per row; a covering index with INCLUDE answers the query as an index-only scan](/imgs/blogs/index-strategy-at-scale-3.webp)

The figure shows the two paths side by side. The narrow index seeks to the row pointer and then must visit the heap — random I/O per matching row — before it can return your columns. The covering index returns `email` and `full_name` directly from the leaf as an *index-only scan*, no heap visit at all. Here is what that looks like in practice:

```sql
-- The query: a couple of columns, looked up by a high-cardinality key.
EXPLAIN (ANALYZE, BUFFERS)
SELECT email, full_name FROM users WHERE user_id = 42;

-- BEFORE — plain index on (user_id):
--   Index Scan using idx_users_user_id on users
--     Index Cond: (user_id = 42)
--     Buffers: shared hit=4         -- 3 index pages + 1 heap fetch
--   (the heap fetch is the random I/O we want to remove)

-- Add the selected columns as INCLUDE payload:
CREATE INDEX CONCURRENTLY idx_users_uid_cov
    ON users (user_id) INCLUDE (email, full_name);

-- AFTER — index-only scan:
EXPLAIN (ANALYZE, BUFFERS)
SELECT email, full_name FROM users WHERE user_id = 42;
--   Index Only Scan using idx_users_uid_cov on users
--     Index Cond: (user_id = 42)
--     Heap Fetches: 0               -- the win: no heap visit
--     Buffers: shared hit=3
```

The win scales: a report that returns ten thousand rows goes from ten thousand random heap fetches to a tight sequential walk of index leaf pages. On a cold cache the difference can be two orders of magnitude in wall-clock. This is the technique behind most "we made the dashboard 50× faster without changing the query" stories.

There is a catch that the word `INCLUDE` hides, and it is the reason index-only scans are not automatic. PostgreSQL's MVCC means the index does not know whether a given row version is visible to your transaction — that visibility information lives in the heap. To skip the heap fetch, the planner consults the *visibility map*, a bitmap that marks heap pages where every tuple is visible to everyone. Only for pages flagged all-visible can the scan trust the index payload and skip the heap. The visibility map is maintained by `VACUUM`. So an index-only scan on a table that is being heavily updated and under-vacuumed quietly degrades into a regular index scan with heap fetches — you will see `Heap Fetches: 9000` in the plan and wonder why the covering index "stopped working." It did not; your `VACUUM` fell behind. Keep autovacuum aggressive on tables you rely on index-only scans for. The full treatment of how composite ordering interacts with covering payloads is in [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans).

### Second-order gotcha: covering indexes bloat the fastest

Because the included columns ride in the leaf, every update to *any* included column writes a new, fat index entry. A covering index on `(user_id) INCLUDE (email, full_name, last_login_at)` where `last_login_at` changes on every request is a disaster — you have turned a stable key into an index that churns on every login, carrying three columns of payload through the churn. The rule: only `INCLUDE` columns that are nearly static relative to the key, and only when a specific hot query needs exactly those columns. A covering index is a precision instrument, not a default.

## 4. The minimal-index discipline: index to your queries, then subtract

> A senior rule of thumb: you do not decide how many indexes a table should have. You measure which queries run, build the minimal set that covers them, and then subtract everything that is unused, redundant, or ignored. The right number is whatever survives that subtraction.

This is the central craft, and the figure below is the whole method in one image: you do not add your way to the right index set, you prune your way to it.

![The minimal-index pruning funnel: start from all candidates, then subtract unused, redundant, and low-selectivity indexes](/imgs/blogs/index-strategy-at-scale-4.webp)

Start by mining what your application actually does. `pg_stat_statements` is the source of truth — it aggregates every query shape with its call count and total time, which tells you both what to index for (the slow, frequent shapes) and, by omission, which existing indexes serve nothing.

```sql
-- The 20 queries that cost the most total time. These are your index targets.
SELECT
    substr(query, 1, 90)              AS query_shape,
    calls,
    round(total_exec_time::numeric, 0)  AS total_ms,
    round(mean_exec_time::numeric, 2)   AS mean_ms,
    rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

Index for `total_exec_time`, not `mean_exec_time`: a query that takes 5 ms but runs a million times a minute deserves an index far more than one that takes 2 seconds but runs twice a day. Once you have built the minimal set that covers those shapes, run the three subtractions.

**Subtraction 1: drop the unused.** `pg_stat_user_indexes` records `idx_scan`, the number of times the planner has chosen each index since stats were last reset. An index with `idx_scan = 0` on a table that has seen real traffic has never been used and is pure write cost.

```sql
-- Indexes never scanned since the last stats reset, biggest first.
-- Keep primary keys and unique constraints regardless (they enforce integrity).
SELECT
    s.relname                                       AS table_name,
    s.indexrelname                                  AS index_name,
    s.idx_scan                                      AS scans,
    pg_size_pretty(pg_relation_size(s.indexrelid))  AS index_size
FROM pg_stat_user_indexes s
JOIN pg_index i ON i.indexrelid = s.indexrelid
WHERE s.idx_scan = 0
  AND i.indisprimary = false
  AND i.indisunique  = false
ORDER BY pg_relation_size(s.indexrelid) DESC;
```

Two caveats before you `DROP`. First, `idx_scan` resets when you run `pg_stat_reset()` or restart with stats lost — make sure the counter has accumulated across a full business cycle, including the monthly report that runs at 3 a.m. on the first. Second, read replicas have *their own* statistics: an index unused on the primary may be the backbone of every analytics query on a replica, and the primary's `pg_stat_user_indexes` cannot see that. Check every node before dropping. With those caveats respected, dropping unused indexes is the single highest-leverage write-throughput improvement available, and it is reversible (you can rebuild) so the risk is low.

**Subtraction 2: kill the redundant.** Apply the leftmost-prefix rule in reverse. The query below flags any index whose key columns are a strict prefix of another index on the same table.

```sql
-- Heuristic: index A is redundant if its key column list is a strict
-- leftmost-prefix of index B on the same table. (indkey prints as a
-- space-separated list of column numbers; we test the prefix on that.)
SELECT
    a.indrelid::regclass                            AS table_name,
    ia.relname                                      AS redundant_index,
    ib.relname                                      AS covered_by,
    pg_size_pretty(pg_relation_size(a.indexrelid))  AS reclaimable
FROM pg_index a
JOIN pg_index b
       ON a.indrelid    = b.indrelid          -- same table
      AND a.indexrelid <> b.indexrelid         -- different index
      AND a.indkey     <> b.indkey             -- not identical
      -- a's columns are a leftmost prefix of b's columns:
      AND array_to_string(b.indkey, ' ') LIKE array_to_string(a.indkey, ' ') || ' %'
JOIN pg_class ia ON ia.oid = a.indexrelid
JOIN pg_class ib ON ib.oid = b.indexrelid
WHERE a.indisprimary = false
  AND a.indisunique  = false
ORDER BY pg_relation_size(a.indexrelid) DESC;
```

Most hits here are an ORM's doing. Rails' `add_index`, Django's `db_index=True`, and Prisma's `@@index` all happily create a standalone index on a foreign key that you also lead a composite with. The standalone is redundant the moment the composite exists. (There is a narrow exception: a much smaller standalone `(a)` can win when `(a, b, c)` is enormous and the extra bytes per entry slow the scan you do on `(a)` alone. Measure before you keep it; the default should be to drop.)

**Subtraction 3: drop what the planner ignores.** A B-tree index on a low-selectivity column — a `boolean` flag, a `status` with three values where one value is 95% of rows, a `type` enum with four members — is one the planner will usually refuse to use, because scanning the index and doing a random heap fetch for most of the table is slower than just scanning the table sequentially. You can confirm by checking `idx_scan` (it will be near zero) or by running the query and seeing a `Seq Scan` in the plan despite the index existing. The fix is not to keep the dead index; it is to *delete* it and, if a specific hot query needs the rare value, replace it with a partial index on exactly that value. A full index on `is_deleted` is dead weight; a partial index `WHERE is_deleted = false` that serves your "active rows" queries is alive and tiny.

### Second-order gotcha: the "missing index" hunt is mostly the "wrong index" hunt

When a query is slow, the instinct is to look for a missing index. More often the right index almost exists — you have `(customer_id)` and the query is `WHERE customer_id = ? AND status = ?`, so it index-scans on `customer_id` then filters `status` in the heap. The fix is to *replace* `(customer_id)` with `(customer_id, status)`, not to add a second index. Replacing keeps the index count flat while making the query faster; adding grows the write tax. Always ask "can I extend an existing index instead of adding one?" before you reach for `CREATE INDEX`.

## 5. The write-amplification math: N indexes, roughly N× the write work

> A senior rule of thumb: on a hot write table, your effective write throughput is inversely proportional to the number of indexes. Halving the indexes is the closest thing to a free doubling of write capacity you will find.

Here is the arithmetic that the four-account cost model collapses into. Consider one `UPDATE` of one row on a table with `N` secondary indexes. The database must:

1. Write a new heap tuple (the new row version, under MVCC).
2. Write a WAL record for the heap change.
3. For each of the `N` indexes, insert a new index entry pointing at the new tuple — and WAL-log each one.

![One row mutation fans out to a new heap tuple plus a maintenance write into every index](/imgs/blogs/index-strategy-at-scale-5.webp)

The fan-out in the figure is the cost: the one logical write you issued became `1 + N` physical writes plus `1 + N` WAL records. The WAL amplification matters as much as the index writes themselves, because WAL is what your replicas replay and what your backups capture — more WAL means more replication lag, more network, more disk on every node in the cluster, not just the primary. A table that goes from three to twelve indexes does not get 4× slower writes in a vacuum; it generates roughly 4× the WAL, which can push a replica that was keeping up into chronic lag, which then breaks read-after-write on anything that reads from replicas. (If you are reading from replicas, see [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) for how that lag propagates.)

There is one escape hatch, and it is worth understanding because it changes the math: PostgreSQL's **heap-only-tuple (HOT) update**. If an `UPDATE` does *not* change any indexed column, and there is room on the same heap page for the new tuple, PostgreSQL writes the new version on the same page and chains it to the old one — and skips *all* index maintenance, because every index entry already points at the page and the chain leads to the new version. A HOT update costs one heap write and zero index writes regardless of how many indexes you have. This is enormous for tables with a few hot, frequently-updated columns (a `last_seen_at`, a counter) and a stack of indexes on stable columns: as long as you never index the churning column and you leave free space on the page, those updates stay HOT and dodge the entire fan-out.

That "room on the same page" condition is tunable with `fillfactor`. The default is 100% (pack pages full), which is right for append-only tables but wrong for update-heavy ones, because a full page forces every update off-page and breaks HOT.

```sql
-- Leave 20% of each page free so updates can stay HOT (heap-only tuple)
-- and skip index maintenance entirely when no indexed column changed.
ALTER TABLE sessions SET (fillfactor = 80);

-- Apply it by rewriting the table. VACUUM FULL takes an ACCESS EXCLUSIVE
-- lock; pg_repack does the same rewrite online without blocking traffic.
-- VACUUM FULL sessions;
-- or:  pg_repack --table sessions  -d mydb
```

The combination — index only stable columns, leave headroom with `fillfactor`, never index the churn column — can take a hot table's effective write cost from `1 + N` writes per update down to a single heap write for the common case. It is the difference between "writes scale linearly with index count" being a death sentence and being something you have engineered around. But it only works if you are disciplined about *which* columns carry indexes, which loops back to the minimal-index discipline: every index on a frequently-updated column is not just its own write cost, it is also a tripwire that disqualifies the HOT optimization for the whole row.

## 6. Maintaining indexes at scale: building and rebuilding without an outage

> A senior rule of thumb: on a table that matters, the words `CREATE INDEX` without `CONCURRENTLY` are an outage waiting for a maintenance window you did not schedule. Train your fingers to type the long version.

Two operational facts about indexes bite hard at scale: building them blocks writes, and using them under churn makes them bloat.

A plain `CREATE INDEX` takes a `SHARE` lock on the table for the entire duration of the build. A `SHARE` lock permits concurrent reads but blocks every `INSERT`, `UPDATE`, and `DELETE` until the index is built. On a small table that is milliseconds. On a 500 GB hot table it is the better part of an hour during which every write blocks — which is to say, an outage. The fix is `CREATE INDEX CONCURRENTLY` (CIC), which builds the index without ever blocking writes.

![The phases of CREATE INDEX CONCURRENTLY: two table scans and a wait, in exchange for never blocking writes](/imgs/blogs/index-strategy-at-scale-6.webp)

The price of not blocking, shown in the figure, is that CIC does more work: it registers the index, does a first table scan to build it from a snapshot, does a *second* scan to catch any rows that changed during the first scan, and then waits for all transactions that started before it to finish (because they might not know about the new index's constraints). That means a longer wall-clock and two scans' worth of I/O, but at no point does it hold a write-blocking lock. Two operational rules come with it:

```sql
-- Build without blocking writes. Note: CIC CANNOT run inside a transaction
-- block, so no BEGIN/COMMIT around it, and not inside most migration tools'
-- default transactional mode (you must opt out per-migration).
CREATE INDEX CONCURRENTLY idx_orders_customer_status
    ON orders (customer_id, status);

-- If CIC fails partway (deadlock, cancellation, a violated unique constraint),
-- it leaves behind an INVALID index that still costs writes but serves no reads.
-- Always check after a failed build:
SELECT indexrelid::regclass AS invalid_index
FROM pg_index
WHERE indisvalid = false;

-- Clean up the failed build (also concurrently) before retrying:
DROP INDEX CONCURRENTLY idx_orders_customer_status;
```

The `INVALID` index trap catches people: a CIC that fails does not roll back to nothing, it leaves a half-built index that the planner will not use for reads but that PostgreSQL *will* maintain on every write. So a series of failed CIC attempts can leave you with several invisible, write-taxing zombie indexes. Always query `pg_index WHERE indisvalid = false` after any failed build and drop them.

### Index bloat and the random-UUID problem

Indexes do not stay tight. Under `UPDATE`s and `DELETE`s, B-tree leaf pages accumulate dead entries that `VACUUM` reclaims as free space but does not give back to the OS, and page splits leave pages half-full. Over time a busy index can be 2–3× the size it would be if rebuilt — that is bloat, and it inflates every one of the four cost accounts. You rebuild with `REINDEX`, and at scale you rebuild *concurrently*:

```sql
-- Rebuild a bloated index without blocking writes (PostgreSQL 12+).
REINDEX INDEX CONCURRENTLY idx_orders_customer_status;

-- Rebuild every index on a table, online:
REINDEX TABLE CONCURRENTLY orders;
```

The worst bloat generator, and the most avoidable, is a random primary key — the classic random UUID (UUIDv4). Because a B-tree stores entries in key order, the physical insert location is determined by the key's value. A sequential key (a `bigserial`, or a time-ordered UUIDv7/ULID) always inserts at the rightmost leaf: pages fill to ~90% and stay there, the working set of "pages being written" is one page, and the index stays dense. A random UUIDv4 inserts into a *random* leaf every time, scattered across the entire key space, which means inserts constantly split half-full pages all over the index, the write working set is the entire index rather than one page, and the index settles at a much lower fill factor with far more pages.

![Sequential keys append to one dense rightmost leaf; random UUIDs split and half-fill leaves across the whole index](/imgs/blogs/index-strategy-at-scale-7.webp)

The figure makes the contrast concrete: the sequential key keeps every leaf packed at 98% and only ever writes the one hot tail page, while the random key leaves a trail of split, 55–60%-full pages across the whole index. The consequences compound — a random-UUID primary index is larger, has a worse cache hit ratio (because the write working set is the whole index, nothing stays hot), generates more WAL from the constant splits, and bloats faster. The fix is not to abandon UUIDs but to use a *time-ordered* one (UUIDv7, ULID, or a `bigserial` if you do not need global uniqueness), which restores sequential insert locality. This problem and its fixes get the full treatment in [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance); for the scale-out flavor, [Instagram's sharded ID scheme](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres) is a sequential-by-design alternative built for exactly this reason.

### Second-order gotcha: bloat hides your unused-index signal

There is a nasty interaction between bloat and the minimal-index audit. A bloated index reports a large `pg_relation_size`, which makes it look expensive and tempting to drop — but the bloat is not inherent, a `REINDEX` would shrink it. Conversely, a small reported size can hide an index that is heavily used but well-maintained. Always reason about an index from `idx_scan` (is it used?) first and size second, and `REINDEX` before you decide an index is "too big to keep."

## 7. Targeted patterns that pay for themselves

Two index patterns return far more than they cost, and recognizing the query shapes that call for them is most of the skill.

**Top-N: index for `ORDER BY` + `LIMIT`.** The single most common expensive query in any application is "the latest N things for this entity" — the last 20 orders for a customer, the most recent 50 events for a device, the newest 10 comments on a post. Written naively, the database filters by the entity, sorts the whole matching set, and returns the top N. With the right composite index, it does none of the sort: it walks the index in already-sorted order and stops after N rows.

```sql
-- "Latest 20 orders for a customer." Lead with the equality column,
-- then the sort column in the direction you query.
CREATE INDEX CONCURRENTLY idx_orders_cust_created
    ON orders (customer_id, created_at DESC);

EXPLAIN (ANALYZE)
SELECT * FROM orders
WHERE customer_id = 99
ORDER BY created_at DESC
LIMIT 20;
--   Limit  (actual rows=20)
--     ->  Index Scan using idx_orders_cust_created on orders
--           Index Cond: (customer_id = 99)
--   (no Sort node, scan stops after 20 rows)
```

The absence of a `Sort` node in the plan is the win — and it scales perfectly, because the query touches 20 index entries whether the customer has 20 orders or 20 million. The order of the columns is load-bearing: the equality predicate column (`customer_id`) must come first, the `ORDER BY` column (`created_at`) second, and the index direction should match the query direction. Get the order wrong and the planner falls back to a scan-and-sort.

**Partial index for a hot subset.** When your queries only ever touch a small, well-defined slice of a large table, a partial index over just that slice is the highest-leverage index you can build. The canonical case is a status column where the interesting rows are rare:

```sql
-- 98% of jobs are 'done' and never queried again; we only ever scan
-- the live ones. Index just those rows.
CREATE INDEX CONCURRENTLY idx_jobs_pending
    ON jobs (created_at)
    WHERE status = 'pending';

-- The planner uses it when the query predicate matches (or implies) the
-- index predicate:
SELECT * FROM jobs
WHERE status = 'pending'
ORDER BY created_at
LIMIT 100;
```

A partial index on the 2% of rows that are `'pending'` is ~50× smaller than the full index, so it caches better, builds faster, and — the part that matters for this whole post — costs almost nothing on writes, because a write only touches the index when the row matches the predicate. Inserting a `'done'` job, or updating a job from `'pending'` to `'done'`, touches the partial index minimally or removes the entry; the 98% of write traffic that never involves a `'pending'` row pays nothing. This is the rare index that is nearly all read benefit and nearly no write tax — the platonic ideal the minimal-index discipline is trying to approximate.

### Second-order gotcha: partial-index predicates must match the query exactly enough

The planner will only use a partial index when it can prove the query's `WHERE` clause implies the index's predicate. `WHERE status = 'pending'` uses an index `WHERE status = 'pending'`. But `WHERE status IN ('pending', 'running')` will *not* use it, because the query can return rows the index does not contain. And a parameterized query `WHERE status = $1` cannot use it either, because the planner does not know at plan time that `$1` is `'pending'`. If you build partial indexes, make sure the application's actual query text matches the predicate — check the plan, do not assume.

## Case studies from production

### 1. Uber's write-amplification migration off PostgreSQL

The most-cited real story about index write cost is Uber's 2016 decision to move a large part of their data tier from PostgreSQL to MySQL. Their engineering write-up named write amplification as a primary driver, and the mechanism was exactly the one in this post: under PostgreSQL's MVCC, an `UPDATE` writes a new tuple at a new physical location, and *every* secondary index must be updated to point at the new location — even for indexes on columns that did not change. On tables with many secondary indexes and a high update rate, a single logical update fanned out into many physical index writes plus the WAL to replicate them, and that WAL volume strained their replication. MySQL's InnoDB, by contrast, uses a clustered primary index and secondary indexes that reference the primary key rather than a physical location, so an update that does not touch a secondary index's columns need not rewrite that index. Whatever you think of the conclusion, the lesson is durable: on a high-update table, the number and design of secondary indexes is not a tuning detail, it is an architectural force large enough to move a company between database engines.

### 2. The fourteen-index orders table

A payments team I worked with had an `orders` table that had accreted fourteen indexes over three years — each one added in response to a specific slow query, each one reasonable in isolation. Write latency had roughly doubled over that period with no single cause anyone could point to. We pulled `pg_stat_user_indexes` across a full month and found that five indexes had `idx_scan = 0`, three were leftmost-prefix-redundant with composites added later, and two were on low-selectivity status columns the planner never used. We dropped ten of the fourteen over two weeks, one at a time, watching read latency for regressions that never came. Write throughput improved by roughly 40%, WAL volume dropped by a third (which fixed a chronic replica-lag problem), and p99 write latency came back down to where it had been two years earlier. Not one read query got slower. The indexes had been a tax with almost no service.

### 3. The covering index that 50×'d a dashboard

An analytics dashboard ran a query returning roughly 8,000 rows of `(account_id, event_type, occurred_at)` filtered by `account_id` and a date range. The existing index on `(account_id, occurred_at)` found the rows fine, but the query also selected `event_type`, so every one of the 8,000 matches did a random heap fetch — and on a cold cache the query took 6–8 seconds. Adding `event_type` as an `INCLUDE` column turned it into an index-only scan: `Heap Fetches: 0`, and the query dropped to ~120 ms. The subtlety that delayed the fix by a day: it only worked after we tuned autovacuum to keep the visibility map current on that table, because under the original lazy vacuum settings the index-only scan kept degrading back into heap fetches whenever a batch insert ran.

### 4. The CREATE INDEX that froze checkout

A junior engineer, fixing a genuinely slow query, ran `CREATE INDEX idx_cart_user ON carts(user_id)` against the production primary during business hours. `carts` was a 200 GB hot table. The plain `CREATE INDEX` took a `SHARE` lock and held it for eleven minutes while it built — eleven minutes during which every `INSERT` and `UPDATE` to `carts` blocked, which meant nobody could add to a cart or check out. The fix was a one-word change to `CREATE INDEX CONCURRENTLY`, which would have built the same index over maybe twenty minutes without blocking a single write. The incident became the reason that team's migration linter now rejects any `CREATE INDEX` not marked `CONCURRENTLY`. The cheapest outages to prevent are the ones a linter can catch.

### 5. The random-UUID primary key that bloated relentlessly

A team that had standardized on UUIDv4 primary keys everywhere noticed that their largest table's primary-key index was nearly twice the size of an equivalent table keyed on `bigserial`, and that `REINDEX` shrank it dramatically only for it to bloat back within weeks. The cause was textbook: random UUIDs scatter inserts across the whole key space, splitting half-full pages constantly and keeping the entire index in the write working set. They could not change the primary key on the live table cheaply, so they did two things: switched *new* tables to UUIDv7 (time-ordered, restoring insert locality) and put the worst existing table on a scheduled `REINDEX CONCURRENTLY` to keep the bloat capped. The new UUIDv7 tables never developed the problem. The deeper analysis lives in [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance).

### 6. The ORM that quietly doubled the index count

A Rails application's migrations had, over years, created a standalone index on nearly every foreign key (Rails encourages this) *and* composite indexes leading with those same foreign keys for specific queries. The result was dozens of redundant standalone indexes — each a strict leftmost prefix of a composite that already existed. The redundancy query from section 4 found 31 redundant indexes across the schema totaling about 90 GB of reclaimable space and, more importantly, a proportional chunk of write cost. Dropping them was uneventful because the composites covered every query the standalones did. The lesson for ORM-heavy shops: your framework's "add an index, it's good practice" default is correct in isolation and wrong in aggregate. Audit for redundancy on a schedule, because the ORM will keep generating it.

### 7. The boolean flag index the planner never touched

A `users` table had an index on `is_active` (a boolean, ~97% `true`). It had been added "to speed up queries for active users," but `pg_stat_user_indexes` showed `idx_scan = 0` over six months — the planner correctly judged that scanning an index covering 97% of the table and then heap-fetching was slower than a sequential scan, so it never used it. Worse, the team had a genuinely useful query for the rare *inactive* users (`WHERE is_active = false`) that the full index served poorly. We dropped the full boolean index and replaced it with a partial index `WHERE is_active = false` — 3% of the rows, a tiny index, instantly used by the inactive-user query, and nearly free on writes. The low-selectivity full index had been the worst of both worlds: write cost with no read benefit, while the query that mattered went unserved.

## When to add an index, and when to resist

Reach for a new index when:

- A query shows up high in `pg_stat_statements` by *total* time (high `mean_exec_time × calls`), and no existing index can be *extended* to cover it. Extend before you add.
- You have a top-N query (`WHERE x = ? ORDER BY y LIMIT n`) doing a sort or a scan — a composite `(x, y)` removes the sort and scales to any table size.
- A hot query selects a few stable columns by a high-cardinality key and you can serve it with a covering `INCLUDE` index, *and* you keep `VACUUM` healthy enough for index-only scans.
- Your queries only ever touch a small, well-defined subset of a large table — a partial index on that subset is nearly all benefit and nearly no write cost.
- A huge, naturally-ordered, append-only table needs range filtering on its ordering column — BRIN gives you that for kilobytes and near-zero write cost.

Resist, or actively remove, when:

- The table is write-heavy and the index's `idx_scan` is zero (or will be) — an unused index is pure tax. Drop it.
- The new index's key columns are a leftmost prefix of an index you already have — it is redundant the moment the longer one exists.
- The column is low-selectivity (a boolean, a dominant status value) — the planner will ignore a full index; use a partial index on the rare value if any query needs it, otherwise build nothing.
- The index would sit on a frequently-updated column on an otherwise HOT-friendly table — you would be disqualifying the heap-only-tuple optimization for every update to the row, which can cost far more than the index saves.
- You are reaching for an index to fix a query that a rewrite would fix better — sometimes the answer is a more selective `WHERE`, a `LIMIT`, or denormalizing one column, not a new B-tree.

The throughline: indexes are a portfolio, not a pile. Every one you hold should be earning its keep on the read side and you should know what it costs you on the write side. At small scale you can be sloppy and the read benefit covers everything. At write scale, sloppiness is a slow, compounding tax that nobody gets paged for until the day the writes can no longer keep up — and by then you have fourteen indexes and no memory of which four you actually needed. Build the minimal set, measure it, and prune it on a schedule. The right number of indexes is not as many as you can justify; it is as few as your queries can tolerate.

## Further reading

- [Composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans) — the mechanics of multi-column indexes, the leftmost-prefix rule, and how `INCLUDE` payloads interact with ordering.
- [Postgres special indexes: GIN, GiST, BRIN, and partial](/blog/software-development/database/postgres-special-indexes-gin-gist-brin-partial) — when to leave B-tree behind, and how each access method is maintained.
- [Random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) — the insert-locality problem in depth and the UUIDv7/ULID fixes.
- [Reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) — how to read the plans this post relies on, including `Heap Fetches`, `Index Only Scan`, and the missing `Sort` node.
- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — where index strategy sits relative to replicas, caching, and sharding.

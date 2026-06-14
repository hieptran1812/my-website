---
title: "Composite, Covering, and Index-Only Scans"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "How a multi-column B-tree is really sorted, why column order decides which queries it serves, and how covering indexes turn a query into an index-only scan that never touches the heap."
tags:
  [
    "indexing",
    "composite-index",
    "covering-index",
    "index-only-scan",
    "postgres",
    "mysql",
    "query-optimization",
    "b-tree",
    "esr-rule",
    "performance",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/composite-covering-and-index-only-scans-1.webp"
---

There is a moment every backend engineer hits eventually. A query that ran in a millisecond in development is taking two seconds in production. You run `EXPLAIN ANALYZE`, and there it is, in black and white: the database is using an index — *your* index, the one you carefully added — and it is still slow. The plan says `Index Scan`, not `Seq Scan`. So the index "works." And yet the latency graph is on fire.

The reason is almost always one of three things, and they are the subject of this entire post. Either the columns are in the **wrong order** for the query, so the index can only do half its job. Or the index is **not covering**, so for every row it matches it pays a second, random trip out to the heap to fetch the columns you actually selected. Or, on Postgres specifically, the index *could* do an index-only scan but the **visibility map is stale**, so it silently degrades back into heap fetches. None of these show up as "no index." All three show up as "index, but slow."

A single-column index is easy. You index `user_id`, you query `WHERE user_id = 42`, the planner uses it, everyone goes home. The hard, high-leverage skill — the one that separates engineers who *have* indexes from engineers who *design* them — is reasoning about **multi-column** indexes: which queries a given column order serves, when adding a column to the key versus the payload matters, and when an index can answer a query entirely from its own pages without ever reading the table. That is where the 100x wins live, and that is where the subtle failures hide.

![A composite B-tree drawn top to bottom: a root routing by the first column, two internal nodes, and four leaves whose entries are sorted by the pair (a, b)](/imgs/blogs/composite-covering-and-index-only-scans-1.webp)

The diagram above is the mental model for everything that follows. A composite index on `(a, b)` is **one** B-tree, not two. Its entries are sorted first by `a`, and then — only within a group of rows that share the same `a` — by `b`. Read the leaves left to right: `(1,10) (1,30) (2,20) (3,90) (3,95) (4,10) ...`. The first coordinate marches upward steadily; the second one resets to a low value every time the first one ticks over. That single fact — *sorted by the first column, then by the second within ties* — is the source of every rule in this article: the left-most prefix rule, the equality-sort-range ordering, why a range column "kills" the columns after it, and why a covering index can answer a query without touching the table. Hold this picture. The rest is a tour through its consequences.

If you have not read [B-trees: how database indexes really work](/blog/software-development/database/b-trees-how-database-indexes-work), the one-paragraph version is this: an index is a B+ tree whose leaves are chained left to right in sorted order, so the database can descend to the first matching leaf and then walk sideways, reading runs of rows in sorted order without climbing back up. This post is what happens when the sort key has more than one column.

## Why "I added an index" is not a plan

Before we touch syntax, let's name the mismatch directly, because almost every composite-index mistake comes from one of these wrong intuitions.

| You assume | The naive mental model | The reality |
| --- | --- | --- |
| An index on `(a, b)` is like two indexes | One for `a`, one for `b` | It is **one** tree sorted by `a` then `b`; it cannot start a search from `b` |
| Column order is cosmetic | Same columns, same speed | Order decides which queries can use it at all, and whether a `Sort` step is needed |
| "Most selective column first" | Put the high-cardinality column leading | Put **equality** predicates first; selectivity is secondary to access pattern |
| Using an index means no table reads | `Index Scan` = fast and done | An `Index Scan` still does a **heap fetch per matched row** unless the index *covers* the query |
| A range and an equality are interchangeable in the key | Order them by selectivity | A **range** column makes every column after it useless for seeking |
| More indexes = faster | Index every column | Each index is a second tree every `INSERT`/`UPDATE`/`DELETE` must maintain |
| `Index Only Scan` is automatic if columns fit | Postgres just does it | It also needs the **visibility map** to say the heap page is all-visible, which depends on `VACUUM` |

The fourth and fifth rows are where most of the surprise lives. An `Index Scan` in a Postgres plan is not the win you think it is — it means "use the index to find row locations, then go fetch each row from the heap." If a query matches ten thousand rows, that is ten thousand random heap reads riding on top of the index descent. The whole art of covering indexes is to turn that `Index Scan` into an `Index Only Scan`, where the answer is read straight off the index leaves and the heap is never touched.

> The first rule of index design: you are not indexing a *table*, you are indexing a *query*. Write the query first, then build the narrowest index that lets the database seek, sort, and return without ever leaving the index.

We will build this up in layers: how a composite key is sorted, the left-most prefix rule, the ESR column-ordering rule, covering indexes and index-only scans, the Postgres visibility-map dependency, the MySQL/InnoDB clustered-index model, detecting redundant indexes, the write cost of each extra index, and partial-plus-composite combinations. Every section has runnable SQL and at least one `EXPLAIN (ANALYZE)` walk-through, because the plan output is the only ground truth that matters.

## 1. How a composite key is actually sorted

**Rule of thumb: a composite index is one sorted list of tuples, ordered lexicographically — like a phone book sorted by (last name, first name). Everything the index can and cannot do follows from that ordering.**

Let's make this concrete with a table we'll reuse throughout. Postgres 16, but the principles are engine-agnostic.

```sql
CREATE TABLE orders (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id  bigint       NOT NULL,
    status       text         NOT NULL,   -- 'open' | 'paid' | 'shipped' | 'cancelled'
    amount       numeric(12,2) NOT NULL,
    created_at   timestamptz  NOT NULL DEFAULT now()
);

-- 10 million rows, ~50k distinct customers, 4 statuses, spread over 2 years
INSERT INTO orders (customer_id, status, amount, created_at)
SELECT
    (random() * 50000)::bigint + 1,
    (ARRAY['open','paid','shipped','cancelled'])[(random()*3.999)::int + 1],
    (random() * 1000)::numeric(12,2),
    now() - (random() * interval '730 days')
FROM generate_series(1, 10000000);

ANALYZE orders;
```

Now create a composite index and look at what it stores:

```sql
CREATE INDEX ix_cust_created ON orders (customer_id, created_at);
```

The leaves of `ix_cust_created` hold tuples `(customer_id, created_at, ctid)`, sorted lexicographically: first by `customer_id`, then by `created_at` within each customer, with `ctid` (the physical row location) as the tie-break and the pointer back to the heap. If you could dump the leaf entries, they would read:

```
(1,    2024-01-03 09:12)
(1,    2024-02-18 14:55)
(1,    2024-11-30 22:01)
(2,    2024-01-01 00:08)
(2,    2024-06-14 12:00)
(3,    2024-03-22 08:44)
...
```

The telephone-directory analogy from Markus Winand's [Use The Index, Luke](https://use-the-index-luke.com/sql/where-clause/the-equals-operator/concatenated-keys) is the right one: a directory sorted by `(last_name, first_name)` lets you find "everyone named Smith" instantly (they are contiguous), and "Smith, John" instantly (Johns are contiguous within Smiths). But it gives you *no* help finding "everyone named John" — the Johns are scattered across every page, one per surname. The second column is only sorted *within a fixed value of the first*.

That is the whole game. The index is sorted by `customer_id`, so all rows for `customer_id = 42` are physically adjacent in the leaves. Within that block, they are sorted by `created_at`. So:

- `WHERE customer_id = 42` — the index seeks straight to the block of `42`s and walks. Fast.
- `WHERE customer_id = 42 AND created_at > '2024-06-01'` — seeks to `42`, then within the `42` block does a second binary search to the first qualifying `created_at`, and walks. Fast, and the rows come out *in `created_at` order for free*.
- `WHERE created_at > '2024-06-01'` (no `customer_id`) — useless. The qualifying `created_at` values are scattered across every customer block. There is no contiguous range to walk.

Let's prove the first two with `EXPLAIN`:

```sql
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT * FROM orders WHERE customer_id = 42;
```

```
 Index Scan using ix_cust_created on orders (actual time=0.041..0.18 rows=204 loops=1)
   Index Cond: (customer_id = 42)
   Buffers: shared hit=8
 Planning Time: 0.10 ms
 Execution Time: 0.21 ms
```

`Index Cond: (customer_id = 42)` — the leading column did the seek. Eight buffer hits, sub-millisecond. Now the two-column predicate:

```sql
EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT * FROM orders
WHERE customer_id = 42 AND created_at > '2024-06-01';
```

```
 Index Scan using ix_cust_created on orders (actual time=0.038..0.09 rows=98 loops=1)
   Index Cond: ((customer_id = 42) AND (created_at > '2024-06-01'::timestamptz))
   Buffers: shared hit=5
 Planning Time: 0.12 ms
 Execution Time: 0.11 ms
```

Both columns appear in `Index Cond` — the database used the full key as a seek, not just the leading column. That is the best case: the entire predicate became an index range, not a filter applied after fetching rows. We'll return to the difference between an `Index Cond` (used to seek) and a `Filter` (applied after fetching) repeatedly, because it is the single most diagnostic line in a plan. If you want a deeper grounding in reading these plans, see [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer).

### Second-order effect: the tie-break column you didn't choose

Every B-tree entry must be unique at the storage level so the engine can locate and delete it. When your indexed columns don't form a unique tuple, the engine silently appends a tie-breaker: in Postgres it's the `ctid` (heap location); in InnoDB it's the primary key. This matters more than it looks. It means an index on `(customer_id, created_at)` in InnoDB is *physically* sorted by `(customer_id, created_at, id)`, and that trailing `id` is available for free. We'll use that fact when we get to InnoDB covering behavior. For now, just know the key you declared is never quite the whole key the engine stores.

## 2. The left-most prefix rule

**Rule of thumb: an index on `(a, b, c)` can be used for any query that constrains a *contiguous left-most prefix* of the columns — `a`, or `a` and `b`, or `a` and `b` and `c` — but never starting from `b` or `c`.**

This is the rule that trips up the most people, so let's state it precisely and then prove every case. Given `CREATE INDEX ix ON t (a, b, c)`, the index can drive a seek for:

- `WHERE a = ?`
- `WHERE a = ? AND b = ?`
- `WHERE a = ? AND b = ? AND c = ?`
- `ORDER BY a, b, c` (and any left-prefix of that ordering)

It **cannot** drive an efficient seek for:

- `WHERE b = ?` alone
- `WHERE c = ?` alone
- `WHERE b = ? AND c = ?`
- `ORDER BY b, c`

And it can do a **partial** job for:

- `WHERE a = ? AND c = ?` — it uses `a` to seek, but `c` becomes a post-filter because `b` is skipped (the columns must be *contiguous*).

![A matrix of query shapes against the index (a, b, c): WHERE a, WHERE a AND b, WHERE a,b,c, and ORDER BY a,b,c all use the index; WHERE b alone does not; WHERE a AND c is partial](/imgs/blogs/composite-covering-and-index-only-scans-2.webp)

The matrix above is the rule in one picture. Green rows use the index fully via a left-most prefix; the red row (`WHERE b`) cannot use it at all; the amber row (`WHERE a AND c`) uses only `(a)` because the gap at `b` breaks contiguity. The intuition is exactly the phone book again: you can search by `a` because the whole index is sorted by `a` first; you can refine by `b` because rows are sorted by `b` within an `a`; but you cannot *start* from `b` because `b` is only locally sorted, never globally.

Let's build a three-column index and prove the failing case. Drop the old index first to force the planner's hand:

```sql
DROP INDEX ix_cust_created;
CREATE INDEX ix_abc ON orders (customer_id, status, created_at);
ANALYZE orders;
```

The leading-prefix query uses it:

```sql
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders WHERE customer_id = 42 AND status = 'open';
```

```
 Index Scan using ix_abc on orders (actual time=0.03..0.07 rows=51 loops=1)
   Index Cond: ((customer_id = 42) AND (status = 'open'::text))
 Execution Time: 0.09 ms
```

Both `customer_id` and `status` are in `Index Cond` — a two-column prefix seek. Now the non-leading query:

```sql
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders WHERE status = 'open';
```

```
 Seq Scan on orders (actual time=0.01..612.4 rows=2500431 loops=1)
   Filter: (status = 'open'::text)
   Rows Removed by Filter: 7499569
 Execution Time: 712.8 ms
```

The planner ignored `ix_abc` entirely and chose a sequential scan, because `status` is not the leading column and there is no contiguous range to seek. (It's also low-cardinality — four values — so even if `status` *were* leading, a seq scan would likely win; we'll cover that in the partial-index section.) The point stands: `WHERE status` cannot use an index that begins with `customer_id`.

Now the subtle partial case — skipping the middle column:

```sql
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders
WHERE customer_id = 42 AND created_at > '2024-06-01';
```

```
 Index Scan using ix_abc on orders (actual time=0.05..0.21 rows=104 loops=1)
   Index Cond: (customer_id = 42)
   Filter: (created_at > '2024-06-01'::timestamptz)
   Rows Removed by Filter: 100
 Execution Time: 0.24 ms
```

Look closely. `customer_id = 42` is the `Index Cond` (used to seek), but `created_at > ...` is a `Filter` (applied after reading every row in the customer block), with `Rows Removed by Filter: 100`. Why? Because `created_at` is the *third* key column, and we skipped `status`. The index is sorted by `(customer_id, status, created_at)`, so within `customer_id = 42` the rows are grouped by `status`, and only within each status are they sorted by `created_at`. With `status` unconstrained, `created_at` is not contiguous — it resets at every status boundary — so the engine can only seek on `customer_id` and then filter. The columns in a seek must be a **contiguous prefix**; a gap stops the seek dead.

This is why "I have an index on all three columns" is not the same as "my query uses the index." The order, and the contiguity, are everything.

### The Postgres "skip scan" footnote

Some engines mitigate the skipped-column problem with a *skip scan* (Oracle's term) or *loose index scan* (MySQL's term for a related trick), where the engine treats a low-cardinality leading column as a loop: for each distinct value of the skipped column, do a separate sub-seek. Postgres 18 added a B-tree skip-scan optimization for exactly this case, so on very new Postgres the partial query above can sometimes do multiple small seeks instead of one filter-heavy scan. But it only pays off when the skipped column has *few* distinct values, and you should never *design* for it — design the index so the columns you constrain are a clean prefix, and treat skip scan as a safety net, not a strategy.

### Myth: "put the most selective column first"

There is a persistent piece of folklore that you should order composite-index columns by selectivity, highest cardinality first. Winand devotes [an entire page to debunking it](https://use-the-index-luke.com/sql/myth-directory/most-selective-first). The truth is that column order should be driven by **how the columns are used in `WHERE` clauses across your query workload**, not by their cardinality in isolation. A column you always filter by equality belongs at the front *even if it's low-cardinality*, because it lets the index seek; a high-cardinality column you only ever use in a range belongs at the back. Selectivity matters *within* the set of equality columns, but it never overrides the access-pattern rule. Which brings us to the rule that actually governs ordering.

## 3. The ESR rule: equality, sort, range

**Rule of thumb: order composite-index columns as Equality first, then the Sort column, then Range columns last. The moment a column is used as a range, every column after it can no longer help you seek — so put ranges at the very end.**

This is the single most useful heuristic in index design, popularized in the MongoDB world as the **ESR rule** ([Equality, Sort, Range](https://alexbevi.com/blog/2020/05/16/optimizing-mongodb-compound-indexes-the-equality-sort-range-esr-rule/)) but equally true for any B-tree index in Postgres, MySQL, SQL Server, or Oracle. The reasoning is mechanical once you internalize the sorting from section 1.

- **Equality columns first.** An equality predicate (`= ?`) pins the leading column(s) to exact values, which collapses the search to a single contiguous block of the index. Multiple equalities stack: `a = 1 AND b = 2` pins both, narrowing to the block where `(a, b) = (1, 2)`. Inside that block, the *next* column is still globally sorted, so it can be used too.
- **The sort column next.** Within the block fixed by the equalities, the rows are sorted by the next key column. If your `ORDER BY` column sits here, the index delivers rows already sorted — no `Sort` node. This is the "S" in ESR, and it is the part people most often get wrong by putting the range before it.
- **Range columns last.** A range predicate (`>`, `<`, `BETWEEN`, `LIKE 'x%'`) matches a *contiguous span* of one column's values rather than a single value. The instant you do that, the columns *after* it are no longer in a single sorted run — they reset within each value of the range column — so they can only be applied as filters, never as seeks, and they cannot satisfy a sort. Therefore a range column poisons everything to its right; put it last so there is nothing to its right that matters.

![Before-and-after of the ESR rule: with a range column placed first, the index stops being usable after it and the planner adds a Sort node; with equality columns first and the range last, the scan is tight and the rows arrive already sorted](/imgs/blogs/composite-covering-and-index-only-scans-3.webp)

The figure contrasts the two orderings for the canonical query: *find a customer's recent open orders, newest first*.

```sql
SELECT * FROM orders
WHERE customer_id = 42         -- Equality
  AND status = 'open'          -- Equality
  AND created_at > now() - interval '7 days'   -- Range
ORDER BY created_at DESC       -- Sort
LIMIT 50;
```

The query has two equality predicates, one range, and a sort on the range column. ESR says the index order should be `(customer_id, status, created_at)` — equalities first, then `created_at`, which serves double duty as both the range *and* the sort column (this is the common, happy case where the sort column and the range column coincide). Let's measure both orderings.

First, the **wrong** order — range column leading:

```sql
CREATE INDEX ix_range_first ON orders (created_at, status, customer_id);
ANALYZE orders;

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT * FROM orders
WHERE customer_id = 42 AND status = 'open'
  AND created_at > now() - interval '7 days'
ORDER BY created_at DESC
LIMIT 50;
```

```
 Limit (actual time=44.1..44.1 rows=2 loops=1)
   ->  Index Scan Backward using ix_range_first on orders
         (actual time=0.4..44.0 rows=2 loops=1)
         Index Cond: (created_at > (now() - '7 days'::interval))
         Filter: ((customer_id = 42) AND (status = 'open'::text))
         Rows Removed by Filter: 96212
         Buffers: shared hit=41033
 Execution Time: 44.2 ms
```

The index seeks on `created_at` (the range), then *filters* the 96,000-odd rows in the last seven days down to the two for customer 42 with status open. `Rows Removed by Filter: 96212`. Forty-one thousand buffer hits. Because the leading column is a range, `customer_id` and `status` can only be filters, never seeks. The query reads almost the entire week of orders to find two rows. (At least the `Index Scan Backward` avoids a `Sort` — `created_at` leading means the index is in the right order for the `ORDER BY` — but the wasted filtering dwarfs that.)

Now the **ESR** order:

```sql
CREATE INDEX ix_esr ON orders (customer_id, status, created_at);
ANALYZE orders;

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT * FROM orders
WHERE customer_id = 42 AND status = 'open'
  AND created_at > now() - interval '7 days'
ORDER BY created_at DESC
LIMIT 50;
```

```
 Limit (actual time=0.04..0.04 rows=2 loops=1)
   ->  Index Scan Backward using ix_esr on orders
         (actual time=0.03..0.04 rows=2 loops=1)
         Index Cond: ((customer_id = 42) AND (status = 'open'::text)
                      AND (created_at > (now() - '7 days'::interval)))
         Buffers: shared hit=5
 Execution Time: 0.06 ms
```

All three predicates are now in `Index Cond` — the equalities pin `(customer_id, status)`, and the range on `created_at` walks the small contiguous tail. Five buffer hits versus 41,033. No `Filter` line at all. And crucially, no `Sort` node: because the index is `(customer_id, status, created_at)`, walking it backward within the fixed `(42, 'open')` block yields rows in `created_at DESC` order natively. **From 44 ms to 0.06 ms — a 700x improvement from column order alone, on identical columns.** This is the kind of "added the right composite index, the query went 100x faster" story you read about; the lever is almost never *whether* you indexed, but *in what order*.

### Why the range column poisons everything after it

Let's nail the "range kills the rest" claim, because it's the deepest part of ESR. Consider `(status, created_at, amount)` and the query `WHERE status = 'open' AND created_at > X AND amount > Y`. The index seeks on `status = 'open'` (equality, fine) and then on `created_at > X` (range, fine — it's the contiguous tail). But `amount` cannot be a seek: within the span of qualifying `created_at` values, `amount` is *not* sorted — it resets for every distinct `created_at`. So `amount > Y` becomes a filter. The range on `created_at` was the last column that could seek; everything after it is post-filtered. This is why the rule is "range *last*": a range can be the final seekable column, but it must not have seekable columns expected after it.

The same logic explains the "S" in ESR. If you have `WHERE a = 1 AND b > 10 ORDER BY c`, an index `(a, b, c)` cannot serve the sort, because after the range on `b` the rows are not in `c` order — they're in `(b, c)` order, so `c` zig-zags. You'd need `(a, c, b)` to get `c` sorted within the equality block, accepting that `b` then becomes a filter. There is a genuine tension when you have both a range and a different sort column: you usually cannot serve both from one index, and ESR tells you to prefer eliminating the sort (which is often the bigger cost on `LIMIT` queries) over tightening the range. Measure it; the right answer depends on selectivity.

| Index order | `WHERE a=? AND b=?` | `WHERE a=? AND b>?` | `ORDER BY a,b` served? | Verdict |
| --- | --- | --- | --- | --- |
| `(a, b)` | seek both | seek `a`, range `b` | yes | ESR-correct for equality-then-range |
| `(b, a)` | seek `b`, filter `a` | range `b`, filter `a` | no | range-first, poisons `a` |
| `(a, b, c)` for `a=? ORDER BY b` | — | — | yes (`b` after equality) | sort served by prefix |
| `(a, c, b)` for `a=? AND b>? ORDER BY c` | — | range `b` is a filter | yes (`c` after equality) | trade range-seek for free sort |

## 4. Covering indexes and index-only scans

**Rule of thumb: an `Index Scan` still reads the table once per matched row. An `Index Only Scan` reads nothing but the index. The difference is whether every column the query touches is *in* the index — and that can be worth 10–100x on wide result sets.**

So far we've been optimizing the *seek*. But there's a second, independent cost hiding in `Index Scan`: the **heap fetch**. When an index entry matches, it contains the indexed columns plus a pointer to the row's physical location. If the query selects a column that is *not* in the index, the engine must follow that pointer out to the heap (the table itself) to read it. One random heap read per matched row. For a query that matches five thousand rows, that's five thousand random I/Os — often the dominant cost, dwarfing the index descent.

A **covering index** is an index that contains every column a query references — both in `WHERE`/`ORDER BY` and in `SELECT`. When an index covers a query, the engine can answer entirely from the index leaves and skip the heap. Postgres calls this an **Index Only Scan**; the speedup comes from converting thousands of scattered random reads into one tight sequential walk of the index leaves.

![Before-and-after comparing an index scan that does a random heap fetch per matched row against an index-only scan that reads everything from the leaves and never touches the heap](/imgs/blogs/composite-covering-and-index-only-scans-4.webp)

The figure shows the canonical example. Take a reporting query that selects two columns the index doesn't have:

```sql
DROP INDEX IF EXISTS ix_esr;
CREATE INDEX ix_cust ON orders (customer_id);   -- does NOT include status, amount
ANALYZE orders;

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT status, amount FROM orders WHERE customer_id = 42;
```

```
 Index Scan using ix_cust on orders (actual time=0.05..1.92 rows=204 loops=1)
   Index Cond: (customer_id = 42)
   Buffers: shared hit=12 read=180
 Execution Time: 2.05 ms
```

`Buffers: shared hit=12 read=180`. The 12 hits are the index descent; the 180 reads are heap fetches — one per matched row that wasn't already cached, scattered randomly across the table. The index found the rows; the heap fetched the columns. Now make the index *cover* the query by adding `status` and `amount`:

```sql
DROP INDEX ix_cust;
CREATE INDEX ix_cust_covering ON orders (customer_id) INCLUDE (status, amount);
VACUUM orders;   -- crucial; we'll explain in section 6
ANALYZE orders;

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
SELECT status, amount FROM orders WHERE customer_id = 42;
```

```
 Index Only Scan using ix_cust_covering on orders (actual time=0.04..0.13 rows=204 loops=1)
   Index Cond: (customer_id = 42)
   Heap Fetches: 0
   Buffers: shared hit=5
 Execution Time: 0.15 ms
```

`Index Only Scan`. `Heap Fetches: 0`. Five buffer hits, all in the index, none in the heap. The query went from 2.05 ms to 0.15 ms — and the gap *widens* as the result set grows, because the heap-fetch cost scales with matched rows while the index-only walk is sequential. On a customer with thousands of orders, this is the difference between a query that touches thousands of random heap pages and one that reads a handful of contiguous index leaves.

### Key columns vs. INCLUDE payload columns

Notice the syntax: `(customer_id) INCLUDE (status, amount)`. The `INCLUDE` clause, added in [Postgres 11](https://www.postgresql.org/docs/current/indexes-index-only-scans.html), puts `status` and `amount` into the index *as payload* — stored in the leaves but **not** part of the search key. This is a deliberate, important distinction.

![A hand-drawn leaf layout: an internal node holding only the key column customer_id fanning out to three chained leaf pages, each leaf split into a blue KEY card (customer_id) and an amber INCLUDE card (status, amount), annotated that key columns are suffix-truncated in upper levels while INCLUDE payload lives only in leaves](/imgs/blogs/composite-covering-and-index-only-scans-5.webp)

The figure shows why `INCLUDE` exists. Key columns (blue) do two jobs: they sort the tree, and they appear in *every* level — root, internal nodes, and leaves — because the upper levels need them to route searches. Payload columns (amber) do *one* job: they sit in the leaves so an index-only scan can read them. They are not in the upper levels at all. The Postgres docs put it precisely: an included column "is merely stored in the index and is not interpreted by the index machinery."

This has three consequences worth memorizing:

1. **`INCLUDE` columns don't bloat the upper tree.** Because they live only in the leaves, they don't reduce fanout in the internal nodes, so the tree stays shallow. Suffix truncation, as the docs note, "always removes non-key columns from upper B-Tree levels." A key column, by contrast, is replicated up the tree and lowers fanout.
2. **`INCLUDE` columns can be types the index can't sort.** Since they're never compared, they don't need a B-tree operator class. You can `INCLUDE` a `point` or a `jsonb` column that you could never put in the key.
3. **`INCLUDE` columns don't change uniqueness.** `CREATE UNIQUE INDEX ix ON t (x) INCLUDE (y)` enforces uniqueness on `x` alone — `y` is just along for the ride. This is the canonical use: a unique constraint on one column that also covers a frequently-selected second column.

When should a column go in the **key** versus `INCLUDE`? The rule is simple and follows directly from sections 2–3:

- Put a column in the **key** if you ever **filter on it, sort by it, or need it for uniqueness**. Only key columns can seek and sort.
- Put a column in **`INCLUDE`** if you only ever **select it** (it appears in `SELECT` but never in `WHERE`/`ORDER BY` for this index). Payload-only.

So for `SELECT status, amount FROM orders WHERE customer_id = 42 AND status = 'open'`, the right index is `(customer_id, status) INCLUDE (amount)`: `customer_id` and `status` seek, `amount` rides along to cover. Putting `amount` in the key would work too, but it would needlessly inflate the upper levels and impose a sort order on `amount` that no query uses.

### You can also cover without INCLUDE

Before `INCLUDE` existed (and still, when the extra column is also useful for seeking or sorting), you cover by putting the column in the key. `(customer_id, status, amount)` covers the same query — the difference is purely whether `amount` participates in the tree's sort order and upper levels. For a column you genuinely never seek or sort on, `INCLUDE` is strictly better. For a column that's sometimes a filter and sometimes just selected, put it in the key. When in doubt: key columns are more flexible but more expensive; `INCLUDE` columns are cheaper but seek-inert.

### The cost: covering indexes are wider

There is no free lunch. A covering index is physically larger because it duplicates the included columns. The Postgres docs warn: "Non-key columns duplicate data from the index's table and bloat the size of the index, thus potentially slowing searches," and advise being "conservative about adding non-key payload columns to an index, especially wide columns." A covering index that includes a `text` column with kilobyte values will be enormous and slow to maintain. Cover the *narrow* columns a hot query selects; never reflexively `INCLUDE` everything. We'll quantify the write cost in section 8.

## 5. InnoDB: every secondary index already covers the primary key

**Rule of thumb: in InnoDB, the table *is* the primary-key B-tree (a clustered index), and every secondary index secretly stores the primary key. So a secondary index always covers any query that selects only its own columns plus the PK — and a non-covering query pays a second tree descent per row.**

MySQL's InnoDB has a different physical model than Postgres, and it changes the covering-index calculus in a way you must understand if you run MySQL. In InnoDB, the table data is stored *inside* the primary-key index — the leaves of the PK B-tree hold the full rows. This is a **clustered index**. There is no separate heap.

A **secondary index** in InnoDB — any non-PK index — does *not* store a physical row pointer. Instead, [its leaves store the primary key value](https://dev.mysql.com/doc/refman/8.0/en/innodb-index-types.html) of each row. As the MySQL manual states: "each record in a secondary index contains the primary key columns for the row, as well as the columns specified for the secondary index. InnoDB uses this primary key value to search for the row in the clustered index."

This has two large consequences:

**1. Every secondary index implicitly covers the primary key.** Because the PK is appended to every secondary index entry, a query that selects only the secondary index's columns *plus the PK* is automatically covered — no row lookup needed. If you have `INDEX (customer_id)` on a table with PK `id`, then `SELECT id, customer_id FROM orders WHERE customer_id = 42` is covered for free, because `id` is silently in the index. MySQL's `EXPLAIN` shows this as `Using index` in the `Extra` column:

```sql
-- MySQL 8.0, InnoDB
EXPLAIN SELECT id FROM orders WHERE customer_id = 42;
```

```
+----+-------------+--------+------+-----------------+-------+---------+
| id | select_type | table  | type | key             | rows  | Extra       |
+----+-------------+--------+------+-----------------+-------+---------+
|  1 | SIMPLE      | orders | ref  | ix_customer_id  |   204 | Using index |
+----+-------------+--------+------+-----------------+-------+---------+
```

`Using index` is InnoDB's equivalent of "index-only scan" — the query was answered from the secondary index alone. Note there is no visibility-map dependency here (that's a Postgres concept); InnoDB's MVCC works differently, reading undo logs rather than checking a per-page visibility bit.

**2. A non-covering secondary-index query pays a "bookmark lookup" per row.** When the query selects a column *not* in the secondary index, InnoDB takes the PK value from the secondary-index leaf and does a *second* B-tree descent into the clustered index to fetch the full row. This is the bookmark lookup. For `SELECT status, amount FROM orders WHERE customer_id = 42` with only `INDEX (customer_id)`, every one of the 204 matched rows triggers a separate clustered-index descent. The `Extra` column will *not* say `Using index`. The fix is identical to Postgres in spirit — make the index cover:

```sql
-- MySQL: cover the query
ALTER TABLE orders ADD INDEX ix_cust_cover (customer_id, status, amount);
```

MySQL 8.0+ also supports the `INCLUDE`-style separation via functional indexes and, since 8.0.13, you can't use Postgres's `INCLUDE` keyword — instead you put the covering columns in the key. (MariaDB's `Aria`/`InnoDB` have their own variants.) The practical upshot: in InnoDB, you cover by extending the secondary key, and the PK is always covered for free.

**3. Keep the primary key narrow.** Because the PK is copied into *every* secondary index, a wide PK (a 36-character UUID string, say) inflates every secondary index on the table. The MySQL manual is explicit: "If the primary key is long, the secondary indexes use more space, so it is advantageous to have a short primary key." A `BIGINT` PK adds 8 bytes per secondary-index entry; a `CHAR(36)` UUID adds 36 bytes to *every* entry of *every* secondary index. This is one of the strongest arguments for integer or sequential-UUID primary keys in MySQL — it's not just the PK tree that pays, it's all of them.

| Aspect | Postgres | InnoDB (MySQL) |
| --- | --- | --- |
| Table storage | Separate heap | Clustered in the PK B-tree |
| Secondary index leaf points to | `ctid` (physical heap location) | Primary key value |
| Non-covering row fetch | Heap fetch via `ctid` | Bookmark lookup (PK descent into clustered index) |
| Index-only scan name | `Index Only Scan` | `Using index` |
| Visibility check for index-only | Visibility map (needs `VACUUM`) | Undo logs / read view (no VM) |
| PK width affects secondary indexes | No (heap uses `ctid`) | **Yes** — PK copied into every secondary index |
| Payload-only columns | `INCLUDE (...)` | Put in the key (no `INCLUDE` keyword) |

## 6. The Postgres visibility-map dependency

**Rule of thumb: in Postgres, an index-only scan is not guaranteed even when the index covers the query. It also requires the heap pages to be marked "all-visible," which is `VACUUM`'s job — so a covering index on a churn-heavy, under-vacuumed table silently degrades into heap fetches.**

This is the trap that catches people who do everything else right. You build a perfect covering index, you see `Index Only Scan` in the plan, you ship it. A week later the same query is slow again, the plan *still* says `Index Only Scan`, but now there's a line you missed: `Heap Fetches: 4821`. What happened?

The cause is Postgres's MVCC model (covered in depth in [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb)). An index entry alone can't tell you whether a row is *visible* to your transaction — multiple versions of a row can exist, and the index doesn't store the per-version visibility info (the `xmin`/`xmax` transaction stamps live in the heap tuple, not the index). So for any index scan, Postgres must somehow confirm each matched row is visible to your snapshot.

For a normal `Index Scan` this is free — it's fetching the heap tuple anyway. But the entire point of an `Index Only Scan` is to *not* fetch the heap. So Postgres uses a shortcut: the **visibility map**, a compact bitmap with one bit per heap page that means "every tuple on this page is visible to all current and future transactions." During an index-only scan, after finding a candidate index entry, the engine checks the visibility-map bit for that entry's heap page. If the bit is set, the row is known-visible and returned straight from the index — no heap access. If the bit is *not* set, the engine must fall back to fetching the heap tuple to check visibility. That fallback is a **heap fetch**, and it's exactly the random I/O the covering index was supposed to eliminate.

![A flow graph: VACUUM (green) sets all-visible bits and INSERT/UPDATE (red) clears them, both feeding the visibility map; an index-only scan checks the map and branches to a SET path (return from index, 0 heap fetches) or an UNSET path (random heap fetch to check MVCC)](/imgs/blogs/composite-covering-and-index-only-scans-7.webp)

The figure shows the dependency. The visibility-map bit is *set* by `VACUUM` after it cleans a page and confirms all its tuples are old enough to be globally visible. It is *cleared* the moment any `INSERT`, `UPDATE`, or `DELETE` touches that page. So on a table with steady write traffic, bits are constantly being cleared by writes and only re-set by the next `VACUUM`. The fraction of pages that are all-visible — and therefore the fraction of your index-only scan that stays index-only — is a direct function of how recently the table was vacuumed.

The Postgres docs are blunt about this: "An index-only scan will be a win only if a significant fraction of the table's heap pages have their all-visible map bits set." And from a [detailed Pythian writeup measuring it](https://www.pythian.com/blog/postgres-covering-indexes-and-the-visibility-map): right after a `VACUUM`, the query ran as a `Parallel Index Only Scan` with `Heap Fetches: 0` in 640 ms; after a round of mass updates cleared the visibility bits (119,439 blocks marked not-all-visible versus 4,627 all-visible), the same query degraded to a `Parallel Seq Scan` taking **15,011 ms** — a 23x regression, with the *exact same index in place*.

Let's reproduce the degradation in miniature:

```sql
-- Fresh covering index, freshly vacuumed
CREATE INDEX ix_cov ON orders (customer_id) INCLUDE (status, amount);
VACUUM orders;

EXPLAIN (ANALYZE, COSTS OFF)
SELECT status, amount FROM orders WHERE customer_id = 42;
-- Index Only Scan ... Heap Fetches: 0 ... Execution Time: 0.15 ms

-- Now churn the table: update a chunk of rows, don't vacuum
UPDATE orders SET amount = amount + 0.01 WHERE customer_id BETWEEN 40 AND 60;

EXPLAIN (ANALYZE, COSTS OFF)
SELECT status, amount FROM orders WHERE customer_id = 42;
```

```
 Index Only Scan using ix_cov on orders (actual time=0.05..0.91 rows=204 loops=1)
   Index Cond: (customer_id = 42)
   Heap Fetches: 198
   Execution Time: 1.02 ms
```

Still an `Index Only Scan` on paper, but `Heap Fetches: 198` — nearly every matched row now pays a heap fetch, because the `UPDATE` cleared the visibility bits on those pages and they haven't been re-vacuumed. A `VACUUM orders;` re-sets the bits and `Heap Fetches` drops back to 0.

The operational lessons:

- **`Heap Fetches` is the metric to watch**, not the scan node name. A high `Heap Fetches` count on an index-only scan means your visibility map is stale relative to your write rate.
- **Covering indexes love read-mostly tables.** The Postgres docs note "there is little point in including payload columns in an index unless the table changes slowly enough that an index-only scan is likely to not need to access the heap." Reference/dimension tables, append-mostly logs (where old pages stay all-visible), and slowly-changing data are ideal. Hot OLTP tables that update the same rows constantly are the worst case.
- **Tune autovacuum more aggressively for tables you rely on index-only scans for.** Lower `autovacuum_vacuum_scale_factor` so `VACUUM` runs more often and keeps the visibility map current. This is the direct tie between covering indexes and vacuum tuning, covered in [PostgreSQL VACUUM, bloat, and autovacuum tuning](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning). A covering index and an aggressive autovacuum policy are a matched pair; deploying one without the other is half a solution.

### Second-order effect: index-only scans and bloat

There's a feedback loop here. `UPDATE`-heavy tables accumulate dead tuples, which both clear visibility bits *and* bloat the table and its indexes. So an under-vacuumed table simultaneously loses its index-only scans *and* grows its indexes, making the now-non-index-only scans even slower. Postgres's HOT (heap-only tuple) updates help — an update that doesn't change any indexed column can avoid touching the indexes — but a covering index *includes* more columns, so it's more likely that an update changes a column the index stores, defeating HOT and forcing index maintenance. This is a real tension: a wider covering index can reduce the HOT-update rate. One more reason to cover only the narrow columns a hot read actually needs.

## 7. Using an index to satisfy ORDER BY (killing the Sort node)

**Rule of thumb: a B-tree is already sorted. If your index's column order matches your `ORDER BY` (including direction), the database reads rows in order straight from the leaves — no `Sort` node, and `LIMIT` becomes an early stop instead of a post-sort filter.**

We touched this in ESR, but it deserves its own treatment because the `Sort`-elimination win is enormous for the most common query shape on the web: *the latest N items for some entity*.

```sql
SELECT * FROM orders
WHERE customer_id = 42
ORDER BY created_at DESC
LIMIT 20;
```

Without a matching index — say only `(customer_id)` exists — the database seeks to the customer's rows, then **sorts all of them** by `created_at` to find the top 20. If the customer has 10,000 orders, that's a sort of 10,000 rows to return 20. The plan shows a `Sort` node, and often a `top-N heapsort`:

```sql
DROP INDEX IF EXISTS ix_cov;
CREATE INDEX ix_c ON orders (customer_id);
ANALYZE orders;

EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders WHERE customer_id = 42 ORDER BY created_at DESC LIMIT 20;
```

```
 Limit (actual time=1.84..1.85 rows=20 loops=1)
   ->  Sort (actual time=1.84..1.84 rows=20 loops=1)
         Sort Key: created_at DESC
         Sort Method: top-N heapsort  Memory: 28kB
         ->  Index Scan using ix_c on orders (actual time=0.03..1.41 rows=204 loops=1)
               Index Cond: (customer_id = 42)
 Execution Time: 1.88 ms
```

There's the `Sort` node, sorting all 204 matched rows (this customer is small; for a customer with 10,000 orders it sorts all 10,000). Now an index whose order *is* the sort order:

```sql
DROP INDEX ix_c;
CREATE INDEX ix_c_created ON orders (customer_id, created_at DESC);
ANALYZE orders;

EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders WHERE customer_id = 42 ORDER BY created_at DESC LIMIT 20;
```

```
 Limit (actual time=0.03..0.04 rows=20 loops=1)
   ->  Index Scan using ix_c_created on orders (actual time=0.03..0.04 rows=20 loops=1)
         Index Cond: (customer_id = 42)
 Execution Time: 0.05 ms
```

No `Sort` node. The index is `(customer_id, created_at DESC)`, so within the `customer_id = 42` block the rows are already in `created_at DESC` order. The engine seeks to the block, reads 20 rows, and stops. `LIMIT 20` is a true early stop — the engine never looks at row 21. This is the single highest-leverage indexing pattern for paginated, "most recent" feeds.

![A five-stage pipeline: plan the top-20 query, descend to the first leaf for the customer, walk the leaf chain in created_at DESC order with no sort, take 20 rows and stop early via LIMIT, return 20 rows in about a millisecond with zero Sort node](/imgs/blogs/composite-covering-and-index-only-scans-6.webp)

The pipeline above is the execution: descend once, walk the pre-sorted leaf chain, take 20, stop. There is no buffering of the full result set and no sort step. Contrast that with the `Sort` plan, which must materialize and sort the *entire* matched set before `LIMIT` can pick the top 20 — the cost scales with the customer's total order count, not with 20.

### The direction trap (ASC/DESC and NULLS ordering)

The index's sort direction must be compatible with the query's. A B-tree can be walked **forwards or backwards**, so a single `(customer_id, created_at)` index serves both `ORDER BY created_at ASC` and `ORDER BY created_at DESC` (the planner shows `Index Scan` vs `Index Scan Backward`). But this breaks down with **mixed** directions on multiple columns. `ORDER BY status ASC, created_at DESC` cannot be served by `(status, created_at)` (both ascending) *or* by reading it backward (which would give `status DESC, created_at ASC`). For mixed-direction sorts you need an index with matching mixed directions: `(status ASC, created_at DESC)`. Postgres also lets you control `NULLS FIRST`/`NULLS LAST` in the index definition to match the query; a mismatch there also forces a `Sort`. When a `Sort` node appears despite a seemingly correct index, check the directions column by column.

### Sort served, but not covered

A subtle interaction: an index can eliminate the `Sort` while *still* paying heap fetches if it's not covering. `(customer_id, created_at)` serves `WHERE customer_id = 42 ORDER BY created_at DESC LIMIT 20` without a sort, but `SELECT *` still fetches the 20 rows from the heap. That's fine — 20 heap fetches is cheap. The win is that you only fetch 20, not 10,000, because the `Sort` elimination let `LIMIT` stop early. Eliminating the `Sort` and covering the `SELECT` are independent optimizations; for a `LIMIT 20` you usually want the former and don't need the latter, while for a `SELECT col FROM ... WHERE big_range` you may want both.

## 8. The write cost of every extra index

**Rule of thumb: every index is a second data structure that every `INSERT`, `UPDATE` (of an indexed column), and `DELETE` must keep in sync. Reads get faster; writes get slower and the table gets bigger. Past a handful of indexes, you're paying more than you're gaining.**

It is tempting, having learned all the above, to index everything. Resist. An index is not free storage that only helps. Each index is a B-tree that must be maintained on every write:

- An `INSERT` must add an entry to **every** index on the table — that's N B-tree descents and N potential page splits per inserted row.
- A `DELETE` must remove (or mark dead) an entry from every index.
- An `UPDATE` must update every index whose columns changed (and in Postgres, unless the update is HOT-eligible, it touches *every* index because a new tuple version is created).

This is **write amplification**: one logical write becomes N+1 physical writes (the table plus N indexes). It is the direct inverse of the read win.

![A matrix of five candidate indexes on one table against serves-reads, write-cost, and verdict columns: (user_id) is a redundant prefix marked DROP, (user_id, created_at) and (user_id, status, created_at) are KEEP, (status) is low-cardinality marked DROP for a partial index, and 8 indexes on one table is marked too-many-cut](/imgs/blogs/composite-covering-and-index-only-scans-8.webp)

The matrix above is the audit framework. Two distinct problems show up: **redundant** indexes (a narrower one whose work is already done by a wider one) and **too many** indexes (the aggregate write tax). Let's take them in turn.

### Detecting redundant indexes

A composite index on `(a, b, c)` already serves every query that `(a)` or `(a, b)` would serve — by the left-most prefix rule, the wider index *is* a usable index on each of its prefixes. So if you have both `(a, b, c)` and `(a)`, the `(a)` index is **redundant**: every query it serves, the wider one serves too, and you're paying its write cost for nothing. (The one caveat: a narrower index is physically smaller, so it can be marginally faster for queries that only need `(a)` and benefit from a more cache-resident index, and `(a)` could be `UNIQUE` while `(a, b, c)` isn't. But for the common case, the prefix index is dead weight.)

Postgres can find these for you. Here's a query that flags indexes whose column list is a left-prefix of another index on the same table:

```sql
SELECT
    ni.relname  AS table_name,
    i1.indexrelid::regclass AS narrow_index,
    i2.indexrelid::regclass AS wider_index
FROM pg_index i1
JOIN pg_index i2
  ON i1.indrelid = i2.indrelid
 AND i1.indexrelid <> i2.indexrelid
 AND i1.indnatts < i2.indnatts
 -- i1's key columns are a left-prefix of i2's
 AND (i1.indkey::int2[])[0:i1.indnkeyatts-1]
     = (i2.indkey::int2[])[0:i1.indnkeyatts-1]
JOIN pg_class ni ON ni.oid = i1.indrelid
WHERE NOT i1.indisunique;   -- don't flag a UNIQUE prefix as redundant
```

And to find indexes nobody is using at all (the planner has never chosen them), `pg_stat_user_indexes` is the source of truth:

```sql
SELECT
    relname AS table,
    indexrelname AS index,
    idx_scan AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE idx_scan = 0          -- never used since last stats reset
  AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;
```

Any index with `idx_scan = 0` after a representative period of production traffic is a candidate for removal — it's pure write tax with zero read benefit. (Be careful: an index backing a `UNIQUE` or `FOREIGN KEY` constraint may show zero scans but still be doing constraint-enforcement work; check before dropping.)

### How many is too many?

There's no hard number, but a useful heuristic: every index roughly adds a fixed percentage to write latency and a chunk to storage. A rough field measurement on an OLTP table is that going from 1 to 8 indexes can cut insert throughput by 2–4x, because each insert now does eight B-tree maintenance operations instead of one. Let's measure the shape of it:

```sql
-- Bulk insert timing as a function of index count
-- (run with 0, then 1, then 4, then 8 indexes present)
EXPLAIN ANALYZE
INSERT INTO orders (customer_id, status, amount, created_at)
SELECT (random()*50000)::bigint+1,
       (ARRAY['open','paid','shipped','cancelled'])[(random()*3.999)::int+1],
       (random()*1000)::numeric(12,2),
       now()
FROM generate_series(1, 100000);
```

| Indexes on table | Insert 100k rows | Relative write cost | Table + index size |
| --- | --- | --- | --- |
| 0 (heap only) | ~0.6 s | 1.0x | 12 MB |
| 1 composite | ~1.1 s | 1.8x | 16 MB |
| 4 composites | ~2.4 s | 4.0x | 31 MB |
| 8 composites | ~4.6 s | 7.7x | 58 MB |

The numbers are illustrative (they vary wildly with hardware, fillfactor, and key width), but the *shape* is universal and roughly linear in index count: writes slow down and storage grows in proportion to how many indexes you keep. The discipline is to keep the **smallest set of indexes that covers your actual query workload** — which usually means a few well-designed composite indexes, each serving multiple query shapes via the left-most prefix rule, rather than many single-column indexes. One `(customer_id, status, created_at)` serves `WHERE customer_id`, `WHERE customer_id AND status`, `WHERE customer_id AND status AND created_at`, and `ORDER BY customer_id, status, created_at` — four query shapes, one write cost. Three separate single-column indexes serve fewer shapes at triple the write cost. Composite indexes are, among other things, a *write-cost* optimization.

> The index you should be most suspicious of is the one you added "just in case." Indexes are not insurance; they are a standing tax. Add them to serve a measured query, and drop them when the query is gone.

## 9. Partial indexes, and partial + composite combinations

**Rule of thumb: when a query always filters on the same condition, a partial index — one that only indexes the rows matching that condition — is smaller, faster, and cheaper to maintain than a full index. Combine it with composite keys for the best of both.**

A partial index has a `WHERE` clause in its definition, so it only contains entries for rows that satisfy that predicate. This is the right tool for two situations: a low-cardinality column where only one value is interesting, and a "hot subset" of a large table.

The classic case is a status flag with a skewed distribution. Suppose 95% of orders are `shipped` or `cancelled` (terminal states nobody queries) and your hot query is always about the open ones:

```sql
-- Full index: indexes all 10M rows, including the 9.5M you never query
CREATE INDEX ix_full ON orders (created_at) WHERE true;   -- (hypothetical full)

-- Partial index: indexes only the ~500k open orders
CREATE INDEX ix_open ON orders (customer_id, created_at)
WHERE status = 'open';
```

The partial index is a fraction of the size and serves the hot query perfectly:

```sql
EXPLAIN (ANALYZE, COSTS OFF)
SELECT * FROM orders
WHERE status = 'open' AND customer_id = 42
ORDER BY created_at DESC LIMIT 20;
```

```
 Limit (actual time=0.02..0.03 rows=4 loops=1)
   ->  Index Scan Backward using ix_open on orders (actual time=0.02..0.02 rows=4 loops=1)
         Index Cond: (customer_id = 42)
 Execution Time: 0.04 ms
```

Notice `status = 'open'` doesn't even appear in `Index Cond` — it's implied by the index's `WHERE` clause, so the engine doesn't need to check it. The index is `(customer_id, created_at)` *within the universe of open orders*, which is exactly the composite-plus-ESR design from section 3, scoped to the rows that matter. This is the partial + composite combination, and it's frequently the optimal answer for "filter on a flag, then seek and sort within it."

Two important rules for the planner to *use* a partial index:

1. The query's `WHERE` must imply the index's predicate. `WHERE status = 'open'` matches an index `WHERE status = 'open'`. The planner does some reasoning here (`WHERE status = 'open' AND x` still matches), but it's not a theorem prover — keep the predicate simple and matching.
2. The predicate column should be one the planner can prove statically. A partial index `WHERE created_at > now()` is illegal (the predicate isn't immutable); `WHERE created_at > '2024-01-01'` is fine but freezes the cutoff.

Partial indexes also shine for the "boolean needle in a haystack" pattern — `WHERE processed = false` on a job queue where almost all rows are eventually `true`:

```sql
CREATE INDEX ix_unprocessed ON jobs (priority, created_at)
WHERE processed = false;
```

This index stays *tiny* — it only holds the backlog, and as jobs are processed, their entries are removed from the index. A full index on `processed` would be huge and the planner would refuse it anyway (a column with two values and a 99/1 skew is a terrible full-index candidate). The partial index sidesteps both problems: it's small, the planner loves it, and it shrinks as the queue drains.

### Second-order effect: partial indexes and write cost

A partial index also reduces write amplification — but only for rows *outside* its predicate. An `INSERT` of a `shipped` order doesn't touch `ix_open` at all (the row doesn't match `WHERE status = 'open'`), so it's free with respect to that index. But an `INSERT` of an `open` order does pay. And an `UPDATE` that transitions a row *into or out of* the predicate (e.g., `open` → `shipped`) causes an index *insert* or *delete*, not just an update. So partial indexes are cheapest when the predicate matches a stable minority of writes, and can be surprisingly active when rows churn across the predicate boundary frequently. For a queue where every row eventually crosses `processed = false → true`, every row pays one partial-index insert and one delete over its lifetime — still far cheaper than a full index, but not zero.

## 10. A practical "design the index for the query" playbook

Everything above collapses into a repeatable procedure. Given a slow query, here's the algorithm I run, in order.

**Step 1 — Write down the exact query shape.** Not the table; the query. Specifically, list:
- The **equality** predicates (`col = ?`).
- The **range** predicates (`col > ?`, `BETWEEN`, `LIKE 'x%'`).
- The **`ORDER BY`** columns and their directions.
- The columns in the **`SELECT`** list (and any in `RETURNING`, `GROUP BY`, etc.).

**Step 2 — Order the key columns by ESR.** Equality columns first (most-selective among them first is a fine tiebreak), then the sort column, then the single range column. Stop at the first range — nothing seekable goes after it.

**Step 3 — Decide covering.** List the `SELECT` columns not already in the key. If the query is read-mostly and the columns are narrow, add them as `INCLUDE` (Postgres) or extend the key (InnoDB) to get an index-only scan. If the table is write-heavy or the columns are wide, skip covering and accept the heap fetches.

**Step 4 — Consider a partial predicate.** If the query always carries the same filter (`status = 'open'`, `deleted_at IS NULL`, `processed = false`), move it into the index's `WHERE` clause to shrink the index and reduce its write cost.

**Step 5 — Verify with `EXPLAIN (ANALYZE, BUFFERS)`.** Confirm: (a) all the predicates you expect are in `Index Cond`, not `Filter`; (b) there's no `Sort` node; (c) for covering, the node is `Index Only Scan` / `Using index` with `Heap Fetches: 0`; (d) `Buffers` is small. If any of these fail, the index design is wrong — go back to step 2.

**Step 6 — Audit for redundancy.** Before shipping, check whether your new index makes an existing narrower one redundant (drop it), and whether the table is creeping past a handful of indexes (consolidate via wider composites that serve multiple shapes by prefix).

Worked end to end: given `SELECT id, amount FROM orders WHERE customer_id = ? AND status = 'open' AND created_at >= ? ORDER BY created_at DESC LIMIT 50`, the playbook yields:
- Equality: `customer_id`, `status`. Range: `created_at`. Sort: `created_at DESC` (coincides with range). Select-only: `amount` (`id` is the PK).
- Key by ESR: `(customer_id, status, created_at DESC)`.
- Cover `amount`: `INCLUDE (amount)`.
- Always filtered on `status = 'open'`? If so, make it partial: `... WHERE status = 'open'` and drop `status` from the key.

Final index:

```sql
CREATE INDEX ix_orders_open_feed
ON orders (customer_id, created_at DESC)
INCLUDE (amount)
WHERE status = 'open';
```

That one index turns the query into a covered, sort-free, partial index-only scan: descend to the customer's open orders, walk 50 in `created_at DESC` order, read `amount` from the leaf, never touch the heap. It is the entire post in a single `CREATE INDEX`.

## Case studies from production

### 1. The 700x column-order swap

**Symptom.** A "recent activity for this account" endpoint timed out under load. The query was `WHERE account_id = ? AND event_type = ? AND created_at > ? ORDER BY created_at DESC LIMIT 100`, and there was an index — `(created_at, account_id, event_type)`. **Wrong first hypothesis.** "We need a read replica; the primary is overloaded." **Root cause.** The index led with `created_at`, the *range* column. So the plan seeked on the time range (often millions of rows for a busy week) and *filtered* `account_id` and `event_type` afterward, reading enormous swaths of the index to find a hundred rows — `Rows Removed by Filter` in the hundreds of thousands. **Fix.** Rebuilt as `(account_id, event_type, created_at)` — ESR order. The equalities pinned the account and type, the range walked the small tail, and the backward scan served the `ORDER BY` with no `Sort`. **Result.** From ~44 ms (and far worse under contention) to ~0.06 ms; the timeout vanished. **Lesson.** A range column at the front of a composite index disables every column behind it. ESR is not a style preference; it's the difference between seeking and scanning.

### 2. The covering index that erased a million heap fetches

**Symptom.** An InnoDB analytics query, `SELECT status, amount FROM orders WHERE customer_id = ?`, was slow despite a perfectly good index on `customer_id`. **Wrong first hypothesis.** "Buffer pool is too small; add RAM." **Root cause.** The secondary index on `customer_id` stored only `customer_id` plus the PK. For each matched row, InnoDB did a *bookmark lookup* — a second descent into the clustered index to fetch `status` and `amount`. A customer with thousands of orders meant thousands of random clustered-index descents per query. **Fix.** Extended the index to `(customer_id, status, amount)`, covering every selected column. `EXPLAIN` flipped to `Using index`; the bookmark lookups disappeared. **Result.** Query time dropped ~30x. **Lesson.** In a clustered-index engine, a non-covering secondary index pays a full second tree descent per row. When a query is slow *despite* using an index, check whether it's paying for row lookups, and cover it.

### 3. The index-only scan that silently un-covered itself

**Symptom.** A Postgres dashboard widget went from 5 ms to 400 ms over a few weeks with no code change and no schema change. The plan still said `Index Only Scan`. **Wrong first hypothesis.** "Data growth; the table got bigger." (It had, slightly, but not 80x worth.) **Root cause.** The covering index was on a table that took heavy `UPDATE` traffic on a counter column. Every update cleared the visibility-map bits on the affected pages, and autovacuum — left at defaults — wasn't keeping up. `Heap Fetches` had climbed from 0 to several thousand per query; the "index-only" scan was now doing a heap fetch for nearly every row. **Fix.** Lowered `autovacuum_vacuum_scale_factor` for that table to 0.02 (from the 0.2 default) so vacuum ran far more often and kept the visibility map fresh; `Heap Fetches` fell back to single digits. **Lesson.** On Postgres, an index-only scan depends on `VACUUM` keeping the visibility map current. The scan-node name lies; `Heap Fetches` tells the truth. Covering indexes and aggressive autovacuum are a matched pair.

### 4. The "we have an index on all three columns" that wasn't used

**Symptom.** A team added `(tenant_id, user_id, created_at)` and was baffled that `WHERE user_id = ? AND created_at > ?` still seq-scanned. **Wrong first hypothesis.** "The planner is broken; force the index with a hint." **Root cause.** The query didn't constrain `tenant_id` — the *leading* column. By the left-most prefix rule, an index leading with `tenant_id` cannot seek on `user_id` alone. The "three-column index" was useless for a query that skipped the first column. **Fix.** Either add `tenant_id = ?` to the query (it was always known from the session context, just not in the `WHERE`) or, where that wasn't possible, create a second index leading with `user_id`. They did the former — it was a genuine bug that the tenant scope was being dropped. **Lesson.** An index serves a *contiguous left-most prefix*. "All the columns are in the index" is irrelevant if the query doesn't constrain the leading one. Read the `Index Cond`; if your column isn't there, the index isn't seeking on it.

### 5. The redundant index doubling write latency

**Symptom.** Insert latency on a high-volume events table crept up over a quarter until it was hurting the ingest pipeline. **Wrong first hypothesis.** "Disk is saturated; provision faster storage." **Root cause.** An audit found *eleven* indexes on the table, several of them prefixes of others: `(account_id)`, `(account_id, ts)`, and `(account_id, ts, kind)` all coexisted. The two narrower ones were fully redundant — every query they served, the widest one served via its prefix — but every insert still maintained all three. Plus three more indexes hadn't been scanned once in 90 days (`idx_scan = 0`). **Fix.** Dropped the two redundant prefixes and the three unused indexes, taking the table from eleven indexes to six. **Result.** Insert latency dropped ~35%, and the table shrank by several gigabytes. **Lesson.** A narrower index whose columns are a left-prefix of a wider one is dead weight. Audit with `pg_stat_user_indexes` (for unused) and a prefix-overlap query (for redundant); each index you drop is write throughput you get back.

### 6. The ORDER BY ... LIMIT that sorted ten thousand rows to return twenty

**Symptom.** A "latest 20 messages" endpoint did a full sort of a conversation's entire history on every request. **Wrong first hypothesis.** "Add a caching layer in front of it." (Treating the symptom; the underlying query was still O(history).) **Root cause.** The index was `(conversation_id)` only. The engine seeked to the conversation, then sorted *all* its messages by `created_at` to take the top 20 — a `top-N heapsort` over tens of thousands of rows per request. **Fix.** Created `(conversation_id, created_at DESC)`. The index's own order matched the `ORDER BY`, so the `Sort` node disappeared and `LIMIT 20` became an early stop — descend, walk 20 leaves, done. **Result.** From ~200 ms to ~1 ms; the cache became unnecessary. **Lesson.** A B-tree is pre-sorted. An index whose column order (and direction) matches your `ORDER BY` eliminates the sort entirely and turns `LIMIT` into a stop condition rather than a post-filter.

### 7. The mixed-direction sort that wouldn't stop sorting

**Symptom.** After fixing several feeds by adding `(entity_id, created_at DESC)` indexes, one feed *still* showed a `Sort` node despite a matching-looking index. **Wrong first hypothesis.** "The planner just won't use it; bad statistics." **Root cause.** The query was `ORDER BY priority ASC, created_at DESC` — a *mixed-direction* sort. The index was `(entity_id, priority, created_at)` (all ascending). Reading it forward gives `priority ASC, created_at ASC`; backward gives `priority DESC, created_at DESC`. Neither matches `priority ASC, created_at DESC`, so the planner had to sort. **Fix.** Rebuilt the index with explicit mixed directions: `(entity_id, priority ASC, created_at DESC)`. The `Sort` node vanished. **Lesson.** A single index can be walked forward or backward, which covers *uniform*-direction sorts in both directions — but a *mixed*-direction `ORDER BY` needs an index whose per-column directions match exactly. When a `Sort` survives a plausible index, check directions column by column.

### 8. The partial index the planner finally embraced

**Symptom.** A job-queue query `WHERE processed = false ORDER BY priority, created_at` seq-scanned a 50-million-row table even with an index on `(priority, created_at)`. **Wrong first hypothesis.** "The composite index is wrong; reorder the columns." **Root cause.** 99.9% of the 50M rows were `processed = true`; the index on `(priority, created_at)` indexed *all* of them, so it was nearly as big as the table and the planner saw little benefit. **Fix.** Replaced it with a partial index: `(priority, created_at) WHERE processed = false`. The index now held only the ~5,000-row backlog — tiny, cache-resident, and the planner used it instantly. **Result.** From a multi-second seq scan to sub-millisecond, and the index shrank as the queue drained. **Lesson.** For a needle-in-a-haystack predicate, a partial index on the selective subset beats a full index on the whole column. It's smaller, the planner loves it, and it self-prunes as the matching set shrinks.

### 9. The wide INCLUDE that bloated the index and broke HOT updates

**Symptom.** After adding a covering index `(customer_id) INCLUDE (status, amount, notes)` to get index-only scans, *write* latency on the table got noticeably worse and the index was surprisingly large. **Wrong first hypothesis.** "The covering index is helping reads; the write cost is acceptable." **Root cause.** Two problems. First, `notes` was a `text` column averaging 800 bytes — including it made the index almost as wide as the table, ballooning its size and the cost to maintain it. Second, because the index now *included* `amount` and `status`, updates to those columns could no longer be HOT (heap-only tuple) updates — they had to touch the index, defeating Postgres's optimization that lets index-free updates skip index maintenance. **Fix.** Dropped `notes` from the `INCLUDE` (the query rarely needed it; the few that did could pay a heap fetch) and kept only `(customer_id) INCLUDE (status, amount)`. Index size dropped 6x and the HOT-update rate recovered for `notes`-only updates. **Lesson.** Cover the *narrow* columns a hot read needs, not everything in the `SELECT`. A wide `INCLUDE` bloats the index and, by widening the set of indexed columns, can defeat HOT updates and raise write cost more than the read win justifies.

### 10. The composite that replaced four single-column indexes

**Symptom.** A table had grown five indexes — `(customer_id)`, `(status)`, `(created_at)`, `(customer_id, status)`, and the PK — and writes were slow while several read queries *still* weren't fully served. **Wrong first hypothesis.** "We're missing an index for query X; add a sixth." **Root cause.** The single-column indexes were a patchwork: `(status)` and `(created_at)` were low-value (low cardinality and range-only, respectively, and the planner rarely chose them), while the queries that mattered all led with `customer_id`. **Fix.** Dropped `(status)`, `(created_at)`, and `(customer_id)` (the last redundant with the composite), and built one `(customer_id, status, created_at)`. By the left-most prefix rule, that single index served `WHERE customer_id`, `WHERE customer_id AND status`, `WHERE customer_id AND status AND created_at`, and `ORDER BY customer_id, status, created_at` — every query shape that mattered. **Result.** From five indexes to two (composite + PK), writes sped up ~40%, storage dropped, and read coverage *improved*. **Lesson.** One well-ordered composite index serves many query shapes through its prefixes. Consolidating single-column indexes into a few composites is simultaneously a read win and a write-cost win — the rare optimization with no tradeoff.

### 11. The `WHERE a AND c` that quietly skipped a column

**Symptom.** A query `WHERE region = ? AND tier = ?` on an index `(region, plan, tier)` was slower than expected, filtering far more rows than it returned. **Wrong first hypothesis.** "The index is fine; it's a data-volume problem." **Root cause.** The query constrained `region` (column 1) and `tier` (column 3) but *not* `plan` (column 2). Because seekable columns must be a *contiguous* prefix, the index could seek only on `region`, then *filter* `tier` across every `plan` within the region — `Rows Removed by Filter` was large. The gap at `plan` broke the contiguity. **Fix.** Since `plan` was always one of three values, the cleanest fix was a re-ordered index `(region, tier, plan)` matching the actual predicates; `tier` then sat contiguously after `region` and both became `Index Cond`. **Lesson.** A composite index seeks on a *contiguous* left-most prefix. Skipping a middle column (`a` and `c` but not `b`) drops you back to seeking on just the prefix before the gap — order the key to match the columns you actually constrain together.

## When to reach for these techniques, and when not to

**Reach for a composite index when:**
- A query filters on **two or more columns together**, especially equality-plus-range or equality-plus-sort. One ESR-ordered composite beats several single-column indexes on both read coverage and write cost.
- You have a **"latest N per entity" feed** (`WHERE entity_id = ? ORDER BY ts DESC LIMIT n`). `(entity_id, ts DESC)` eliminates the sort and makes `LIMIT` an early stop — the highest-leverage pattern on the web.
- Several query shapes share a **common leading column**. A single `(a, b, c)` serves `WHERE a`, `WHERE a AND b`, and `WHERE a AND b AND c` via the prefix rule — consolidate.

**Reach for a covering index (`INCLUDE` / extended key) when:**
- A hot, **read-mostly** query selects a few **narrow** columns it doesn't filter on, and the heap fetches dominate its cost. Cover them for an index-only scan.
- You have a **unique constraint** on one column but frequently select a second alongside it — `UNIQUE (x) INCLUDE (y)` covers both with uniqueness on `x` only.
- (Postgres) The table is **slowly-changing** enough that `VACUUM` can keep most pages all-visible, so the index-only scan actually stays index-only.

**Reach for a partial index when:**
- A query **always carries the same filter** (`status = 'open'`, `deleted_at IS NULL`, `processed = false`), especially when that filter selects a small minority of rows. The partial index is smaller, cheaper, and self-pruning.

**Skip or be cautious when:**
- The table is a **write firehose** and the query is occasional. Each index is a standing write tax; don't add one to serve a query that runs twice a day on a table that ingests a million rows an hour.
- You're tempted to **`INCLUDE` a wide column** (`text`, `jsonb`, large `bytea`). The index bloats, maintenance slows, and on Postgres you can defeat HOT updates. Let those columns stay heap-fetched.
- You're indexing **"just in case."** An unused index (`idx_scan = 0`) is pure overhead. Add indexes to serve measured queries; drop them when the query is gone.
- The column is **very low cardinality** and you want a *full* index on it. The planner will usually refuse (a seq scan beats fetching 25% of the table randomly). Use a partial index on the selective value instead, or accept the scan.
- You already have a **wider index** whose left-prefix covers the new query. Don't add the narrower one; it's redundant from birth.

The thread running through all eleven case studies and every section is the opening rule: **you index a query, not a table.** A composite index is one sorted list of tuples, and three questions decide whether it serves your query well. Does the column order match the access pattern — equalities first, sort next, range last (ESR)? Does the index contain every column the query touches, so it can answer without the heap (covering)? And — on Postgres — is the visibility map fresh enough that the index-only scan stays index-only (vacuum)? Get those three right and the same query that crawled at two seconds answers in a tenth of a millisecond, off a structure a fraction the size of the table. Get them wrong and you'll have an index the plan proudly reports using, while the latency graph quietly burns.

The skill transfers, too. The next time you meet a slow query with an index already on it, don't ask "is there an index" — the plan already answered that. Ask the three real questions: is every predicate in `Index Cond` or are some demoted to `Filter`; is there a `Sort` node that the index's order should have eliminated; and is it an `Index Only Scan` with `Heap Fetches: 0` or is it paying for the heap on every row. Those three lines in an `EXPLAIN (ANALYZE, BUFFERS)` are the whole story, and once you can read them, index design stops being guesswork and becomes a short, deterministic procedure.

## Further reading

- **Markus Winand, [Use The Index, Luke](https://use-the-index-luke.com/sql/where-clause/the-equals-operator/concatenated-keys)** — the canonical, vendor-neutral treatment of concatenated indexes, the left-most prefix rule, and the [most-selective-first myth](https://use-the-index-luke.com/sql/myth-directory/most-selective-first). Start here.
- **PostgreSQL documentation, [Index-Only Scans and Covering Indexes](https://www.postgresql.org/docs/current/indexes-index-only-scans.html)** — the authoritative reference on `INCLUDE`, the two index-only-scan requirements, and the visibility-map dependency, straight from the source.
- **[Postgres Covering Indexes and the Visibility Map](https://www.pythian.com/blog/postgres-covering-indexes-and-the-visibility-map)** (Pythian) — a measured walk-through of an index-only scan degrading to a seq scan (640 ms → 15 s) when the visibility map goes stale, and recovering after `VACUUM`.
- **Alex Bevilacqua, [Optimizing Compound Indexes: the ESR Rule](https://alexbevi.com/blog/2020/05/16/optimizing-mongodb-compound-indexes-the-equality-sort-range-esr-rule/)** — the clearest exposition of Equality-Sort-Range, with `explain()` output showing blocking sorts appear and disappear. MongoDB-framed but universally applicable.
- **MySQL documentation, [Clustered and Secondary Indexes](https://dev.mysql.com/doc/refman/8.0/en/innodb-index-types.html)** — how InnoDB stores the PK in every secondary index, why that covers the PK for free, and why a narrow PK matters.
- **Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 3** — the storage-engine fundamentals behind all of this: secondary indexes, multi-column indexes, and covering indexes (indexes "with included columns") explained from first principles. The book to read once and reread.
- [B-trees: how database indexes really work](/blog/software-development/database/b-trees-how-database-indexes-work) — the structure underneath every index in this post.
- [Reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) — how to read the plans this post relies on.
- [PostgreSQL VACUUM, bloat, and autovacuum tuning](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning) — the other half of keeping index-only scans fast.
- [Join algorithms: nested loop, hash, merge](/blog/software-development/database/join-algorithms-nested-loop-hash-merge) — how indexes feed the join methods that consume their sorted output.

---
title: "Schema Design for Scale: Denormalization, Types, and Access Patterns"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Why the fully normalized textbook schema collapses at scale, and how denormalization, careful types, and access-pattern-driven modeling keep a schema fast as it grows."
tags: ["database-scaling", "schema-design", "denormalization", "data-modeling", "postgresql", "jsonb", "access-patterns", "vertical-partitioning", "database-performance", "system-design"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 32
---

The schema you draw on a whiteboard in week one is almost always a textbook schema. Entities become tables, relationships become foreign keys, and every fact lives in exactly one place. It is clean, it normalizes beautifully, and your database professor would give it an A. It is also, at scale, the slowest schema you could have shipped. The first time a product hits real traffic, the symptom is always the same: the page that renders an order, or a profile, or a feed item, is doing a five-way join, and that join is now the single most expensive thing your database does, repeated millions of times an hour.

I have watched this play out at three companies. The fix was never "add another index" or "buy a bigger box." The fix was to stop modeling the data the way the entities relate and start modeling it the way the application reads. That is the whole thesis of this post: **the textbook schema is optimized for storage and correctness; a schema that scales is optimized for the access patterns that actually run against it.** Getting there means denormalizing on purpose, picking types that keep rows narrow and hot, and treating "how do I read this?" as the first design question, not the last.

![Normalized 3NF join versus a denormalized single-shard fetch](/imgs/blogs/schema-design-for-scale-1.webp)

The diagram above is the mental model for the entire article. On the left is the textbook order page: select from `orders`, join `order_items`, join `products`, join `users`, join `addresses`. Five tables, five-plus random I/Os, and on a sharded system those tables may not even live on the same machine. On the right is the same read after denormalization: a single `order_view` row keyed by `order_id`, with the line items inlined as JSONB, served by one primary-key seek on one shard. Same data, same answer, two completely different cost profiles. Everything below is a tour of the decisions that get you from the left side to the right, and the ones that go wrong when you do it carelessly.

## Why the textbook schema stops scaling

Normalization solves a real problem: it removes redundancy so that a fact is stored once and updated once. That property is genuinely valuable on the write path. The trouble is that "stored once" means "reassembled on every read," and reassembly is a join. At small scale the joins are free because everything fits in memory and the planner does a few nested loops you never notice. At scale, the joins are the cost — they turn a single logical read into a fan-out of random lookups, and once the tables are sharded, a join can require talking to multiple machines and merging the results.

| Assumption from the textbook | What you were taught | Reality at scale |
| --- | --- | --- |
| Redundancy is always bad | Store every fact exactly once; normalize to 3NF | Controlled redundancy buys cheap reads; the cost is keeping copies in sync |
| Joins are cheap | The query planner handles joins; just declare the relationship | Joins across large or sharded tables are the dominant read cost |
| The schema models the domain | Tables mirror entities and their relationships | Tables should mirror access patterns; the domain is a starting point, not the layout |
| One table per entity | Users, orders, items, products are separate tables | Hot reads often want one denormalized row; cold analytics want a different shape entirely |
| Types are an implementation detail | Pick whatever holds the value | Type width sets rows per page, cache hit rate, and whether your IDs run out |

None of this means normalization is wrong. It means normalization is a default that you override deliberately where the read path demands it. The rest of this post is a catalog of those deliberate overrides, each with the mechanism that makes it work, the cost it adds to the write path, and the failure mode that bites when you reach for it without thinking.

## 1. Normalization versus denormalization: the read/write tradeoff

**The senior rule of thumb: denormalize so the common read is a single-table, single-shard fetch, and accept that you now own the cost of updating the copies.**

Start with the fully normalized version of the order page. In third normal form, the data is spread across five tables and the read reassembles it:

```sql
-- Fully normalized (3NF). The order page needs five tables.
SELECT o.id, o.placed_at, o.status,
       u.name,
       a.line1, a.city, a.postcode,
       oi.quantity, p.title, p.price_cents
FROM   orders       o
JOIN   users        u  ON u.id = o.user_id
JOIN   addresses    a  ON a.id = o.shipping_address_id
JOIN   order_items  oi ON oi.order_id = o.id
JOIN   products      p ON p.id = oi.product_id
WHERE  o.id = $1;
```

Run `EXPLAIN (ANALYZE, BUFFERS)` on this and the plan is a stack of nested-loop joins, each one an index lookup into another table. On a single box with everything cached, this is a few hundred microseconds and you will never notice. The intuition that matters is what happens when each of those tables has hundreds of millions of rows and lives on a different shard: the `orders` row resolves on shard 3, but `order_items` is sharded by `order_id` so it might be on shard 3 too, while `products` is a global catalog on a different cluster and `users`/`addresses` are sharded by `user_id` on shard 7. Now one logical read is a distributed query touching three machines, and the slowest machine sets your latency. The join did not get more expensive per row; it got more expensive because the rows moved apart. This is exactly the problem [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) tries to contain, and denormalization is the schema-level half of the same fight.

The denormalized read model collapses all of that into one row:

```sql
-- Denormalized read model: one row per order, line items inlined.
CREATE TABLE order_view (
  order_id      bigint PRIMARY KEY,
  user_id       bigint NOT NULL,
  placed_at     timestamptz NOT NULL,
  status        order_status NOT NULL,
  customer_name text NOT NULL,
  ship_line1    text NOT NULL,
  ship_city     text NOT NULL,
  ship_postcode text NOT NULL,
  items         jsonb NOT NULL,   -- [{sku, title, qty, price_cents}, ...]
  total_cents   bigint NOT NULL
);

-- The order page is now one seek on one shard.
SELECT * FROM order_view WHERE order_id = $1;
```

The read is a single primary-key seek. If `order_view` is sharded by `order_id`, it is also a single-shard fetch with no fan-out. That is the win, and it is a large one: you have traded a multi-table distributed join for one index seek. But look at what you signed up for on the write side. The customer's name now lives in `users` and in every `order_view` row they have ever generated. When they change their name, you have to update all of those copies, or accept that historical orders show the old name (which, for shipping records, is often actually correct). The product title lives in `products` and inside the `items` JSONB of every order that ever bought it. Denormalization does not delete the update cost; it moves it from "one place, on demand" to "many places, maintained by you."

That is the entire tradeoff, and it is worth stating bluntly because every decision in this post is a special case of it:

> Normalization minimizes redundancy and pays for it on every read. Denormalization minimizes read cost and pays for it on every write. At scale you have far more reads than writes on the hot path, so you bias toward the read — but only where you can keep the copies consistent.

The schema-choice space is wider than just normalize-or-not, though. Here is the full menu, with the read cost, the write cost, and the condition that should make you reach for each one.

![Schema choices ranked by read cost, write cost, and when to use them](/imgs/blogs/schema-design-for-scale-2.webp)

| Choice | Read cost | Write cost | Reach for it when |
| --- | --- | --- | --- |
| Normalize (3NF) | High (joins) | Low | Write-heavy OLTP where reads are rare or already cheap |
| Denormalize | Low | High (fan-out to copies) | A read-heavy hot path with a stable shape |
| Narrow rows | Low | Low | Always, by default — there is rarely a reason to be wide |
| Wide table | Medium | High (whole-row rewrite) | Rarely; usually a sign you should split it |
| JSONB blob | Medium | Low | Sparse, variable, or per-tenant attributes |
| Typed columns | Low (indexed) | Low | Fields you actually query or sort by |
| Append-only log | Medium (fold) | Low (insert only) | Hot write contention on a mutable row |
| Counter/aggregate table | Low (one row) | Medium | A count or sum you need on every read |

Read the table as a decision aid, not a ranking — there is no globally "best" row. The job is to match the choice to the access pattern, which is the subject of the next section.

## 2. Design for access patterns, not entities

**The senior rule of thumb: write down every query that will run in production before you draw a single table, then build storage that serves each query in one hop.**

This is the lesson the NoSQL world learned the hard way and then over-corrected into dogma, but it applies just as much to relational databases. In DynamoDB you literally cannot design a table without first enumerating your access patterns, because there are no ad-hoc joins to bail you out — the [DynamoDB single-table design](/blog/software-development/database-scaling/dynamodb-global-tables-and-single-table-design) pattern is access-pattern-first modeling taken to its logical end. The mistake teams make is assuming this discipline is a NoSQL tax. It is not. A Postgres schema designed around access patterns is faster than one designed around entities, for exactly the same reason: each read resolves against storage shaped to answer it.

![Access patterns, not entities, decide the physical schema](/imgs/blogs/schema-design-for-scale-3.webp)

The figure shows three different reads against the same order domain, each given the storage it needs. "Get the order page" hits the denormalized `order_view` keyed by `order_id` and resolves in one seek. "List a user's orders" is a different access pattern with a different key, so it is served by a secondary index on `(user_id, created_at)` that returns the rows already sorted. "Revenue dashboard" is an analytical scan that nobody should run against the OLTP primary, so it goes to a columnar replica where a full scan is the right tool — see [OLTP versus OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) for why a row store and a column store want opposite physical layouts. The slow query did not get optimized away; it got moved off the hot path entirely.

The practical method is mechanical:

1. **List the reads.** Every endpoint, every background job, every dashboard. For each, write the predicate (`WHERE order_id = ?`), the sort (`ORDER BY created_at DESC`), and the frequency (10k/s versus 10/hour).
2. **Rank by frequency times latency-sensitivity.** The read that runs 10,000 times a second and blocks a page render is your design center. The report that runs once an hour can be slow.
3. **For each hot read, ask: can this be one key lookup?** If not, what would the table have to look like for it to be? That hypothetical table is your denormalized read model.
4. **Push the cold reads elsewhere.** Replica, columnar store, search index, nightly rollup. Do not let a once-an-hour analytical scan dictate the layout of a table that serves a hot path.

The failure mode here is designing for the read you find most interesting instead of the one that runs most often. I once reviewed a schema exquisitely tuned for a "user activity timeline" feature that, in production telemetry, was loaded by 0.3% of sessions, while the home feed — 95% of reads — did a three-table join on every request because nobody had modeled it explicitly. Access-pattern design is not about elegance; it is about spending your denormalization budget where the traffic is.

## 3. Types matter at scale

**The senior rule of thumb: every byte in a row is multiplied by the number of rows times the number of times that row is read into cache. Narrow types are not pedantry; they are throughput.**

Types feel like the most boring part of schema design, and they are the part most likely to cause a 3 a.m. outage. Two things are true at scale that are invisible at small scale: the width of a row determines how many rows fit on a page, and the width of an integer determines when your IDs run out.

### Rows per page, and why narrow wins

Postgres stores rows in 8 KB pages (heap blocks). The number of rows that fit on a page is set by the row width, and that number drives your buffer-cache hit rate: a narrow row means more rows per page means more useful data per page read means a higher fraction of your working set fits in RAM. Here is the calculation, with the real overheads:

```python
PAGE      = 8192   # Postgres heap page size, bytes
HEADER    = 24     # page header
ITEM_PTR  = 4      # line pointer per tuple
TUPLE_HDR = 24     # heap tuple header + null bitmap, MAXALIGNed

def rows_per_page(user_data_bytes: int) -> int:
    tuple_size = TUPLE_HDR + user_data_bytes
    tuple_size = (tuple_size + 7) & ~7      # 8-byte alignment (MAXALIGN)
    per_row = tuple_size + ITEM_PTR
    return (PAGE - HEADER) // per_row

print(rows_per_page(16))    # narrow row, a couple of bigints -> ~185
print(rows_per_page(400))   # wide row, text/json inline       -> ~19
```

A 16-byte payload packs about 185 rows per page; a 400-byte payload packs about 19. That is nearly a 10x difference in cache density for the same amount of RAM. The narrow table answers ten times as many reads from buffer cache before it has to go to disk.

![Rows per 8 KB page: narrow rows pack denser and cache better](/imgs/blogs/schema-design-for-scale-4.webp)

The figure makes the consequence concrete: the narrow table fits roughly ten times more tuples in the same page, so a sequential scan or an index range read touches a fraction of the blocks. This is why a `bigint` versus a `text` you did not need, or a `timestamptz` versus a stringified date, is not bikeshedding. It is the difference between a table that lives in RAM and one that thrashes the disk.

### int versus bigint, and the integer-overflow outage

The most expensive type mistake in the industry is the 32-bit integer primary key. A signed `int4` maxes out at 2,147,483,647. That sounds enormous right up until you have a high-volume `events` or `messages` or `audit_log` table, and then 2.1 billion rows is a few months of traffic. The day you hit it, every insert fails with an integer-out-of-range error, the table is effectively read-only, and the fix — migrating the primary key from `int4` to `int8` — requires rewriting every row and every foreign key that references it, online, under load, without downtime. I have watched a team burn a full week of incident response on exactly this, and it was entirely avoidable.

```sql
-- The mistake: a 32-bit identity on a high-volume table.
CREATE TABLE events (
  id         serial PRIMARY KEY,      -- int4! caps at 2,147,483,647
  ...
);

-- The default everywhere it could ever matter:
CREATE TABLE events (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- int8
  ...
);
```

The lesson is blunt: use `bigint` for any primary key on a table that could plausibly grow, which is almost all of them. The 4 extra bytes per row are nothing next to the outage. Reserve `int4` for things with a natural small ceiling — a `status` enum's backing integer, a count of items in a cart — where 2 billion is genuinely impossible.

### Keep wide values out of the hot row: Postgres TOAST

Postgres has a mechanism called TOAST (The Oversized-Attribute Storage Technique) that automatically moves large values — long `text`, big `jsonb`, `bytea` blobs — out of the main heap into a side table once a row exceeds about 2 KB, leaving a small pointer in the main row. This is the engine quietly doing vertical partitioning for you, and it is the reason a `users` table with a rarely-read `bio text` column still scans fast: the bios get TOASTed out and the main rows stay narrow. The catch is that TOAST only kicks in past the threshold, and a row full of medium-width columns (a few hundred bytes each, none individually huge) stays inline and bloats your page density. The takeaway: do not stuff wide, rarely-read values into a hot, frequently-scanned table and assume the engine will save you — model it.

### timestamptz, UUID, and enum: three quick rules

Three more type decisions that compound at scale:

- **`timestamptz`, never `timestamp`.** `timestamptz` stores an absolute instant (internally UTC) and converts on display; `timestamp` (without time zone) stores a wall-clock reading with no zone, which is ambiguous the moment your servers, users, or data span time zones — and they always eventually do. Both are 8 bytes, so there is no cost to correctness here. Storing local wall-clock times is the source of an entire genre of "the report is off by an hour twice a year" bugs.
- **`bigint` over random `uuid` for primary keys.** A 128-bit random UUID is double the width of a `bigint` and, far worse, its randomness destroys index locality: each insert lands in a random leaf of the B-tree, so you get page splits and cache misses everywhere instead of the tight append pattern a monotonic key gives you. This is severe enough that it gets its own post — [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) — and the fix when you need a UUID is a time-ordered variant (UUIDv7) or a Snowflake-style 64-bit ID that keeps inserts sequential.
- **`enum` over `varchar` for closed sets.** A status that can only be `'active' | 'suspended' | 'closed'` should be a native enum (stored as a 4-byte OID, sortable, constrained) rather than a `varchar` that repeats the string `'active'` in every row and lets `'aktive'` slip in. The enum is narrower, faster to compare, and enforces the closed set for free. The one downside is that adding or reordering enum values is a schema change, so for sets that churn frequently a small lookup table with a foreign key is the more flexible choice.

## 4. JSON and JSONB columns: when schemaless helps and when it hurts

**The senior rule of thumb: put the fields you query in typed columns and the fields you only ever read back in a single JSONB sidecar. Never use JSONB to avoid deciding what your data is.**

`jsonb` is the most abused feature in Postgres. Used well, it absorbs the genuinely variable parts of your data — per-tenant custom fields, sparse attributes that only 2% of rows have, experimental flags you are not ready to commit to a column — without an `ALTER TABLE` for every one. Used badly, it becomes a dumping ground where the whole row is one blob, nothing can be indexed properly, every read deserializes kilobytes to grab one field, and there are no constraints so the data quietly rots into inconsistency.

The discriminator is simple: **do you query this field, or do you only display it?** Anything in a `WHERE`, `JOIN`, `ORDER BY`, or `GROUP BY` wants to be a typed, indexed column. Anything you only ever `SELECT` and hand to the application can live in JSONB. The result is the hybrid: a few typed columns for the queryable, indexable, constrained core, plus one JSONB sidecar for the long tail.

![The JSONB hybrid: typed columns you query plus a GIN-indexed JSONB sidecar](/imgs/blogs/schema-design-for-scale-6.webp)

The figure shows the layout. `id`, `email`, `status`, and `created_at` are real columns with real types and B-tree indexes, because you filter and sort by them. Everything sparse and variable lives in `attributes`, a single JSONB column with a GIN index so containment queries stay fast.

```sql
CREATE TABLE users (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email       citext UNIQUE NOT NULL,
  status      user_status NOT NULL DEFAULT 'active',
  created_at  timestamptz NOT NULL DEFAULT now(),
  -- sparse / experimental / per-tenant attributes:
  attributes  jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- A GIN index makes key-existence and containment lookups indexable.
-- jsonb_path_ops is smaller and faster for the common @> containment case.
CREATE INDEX users_attributes_gin
  ON users USING gin (attributes jsonb_path_ops);

-- Fast: the GIN index serves this containment query.
SELECT id FROM users WHERE attributes @> '{"beta_opt_in": true}';
```

The GIN index is what makes JSONB more than a write-only blob: it indexes the keys and values inside the document so `@>` (contains) and `?` (key exists) queries do not scan the whole table. But know its limits. A GIN index does not help range queries inside the JSON (`attributes->>'age' > '30'` will not use it), it is larger and slower to update than a B-tree, and it cannot enforce that `beta_opt_in` is actually a boolean. The moment a JSONB key graduates from "experimental" to "queried in anger on the hot path," promote it to a real column:

```sql
-- Promote a hot JSONB key to a typed, indexed column.
ALTER TABLE users ADD COLUMN plan text;
UPDATE users SET plan = attributes->>'plan' WHERE attributes ? 'plan';
CREATE INDEX users_plan ON users (plan);
-- New code writes the column; a backfill + dual-write migrates the rest.
```

The anti-pattern to name and avoid is **the everything-blob**: a table that is `(id bigint, data jsonb)` and nothing else. It feels flexible on day one and becomes unqueryable, unconstrainable, and unindexable by day ninety. Reddit famously built much of its early data model as a two-table entity-attribute-value store (a `thing` table and a `data` table of key-value pairs), and while it let them ship fast, they spent years fighting the lack of typed columns and indexes it forced. Flexibility you do not need is a cost, not a feature.

## 5. Wide tables, narrow tables, and vertical partitioning

**The senior rule of thumb: if one column is updated far more often than the rest of the row, it does not belong in the same table as the rest of the row.**

There is a specific, common pathology that this section is about: the hot counter in a wide table. Picture a `posts` table with fifty columns — title, body, author, tags, SEO metadata, timestamps, flags — and one `view_count` that gets incremented on every page view. In an MVCC database like Postgres, an `UPDATE` does not modify a row in place; it writes a whole new version of the row and marks the old one dead for vacuum to clean up later. So every single view increment rewrites all fifty columns, generates a dead tuple, and churns the vacuum process — even though forty-nine of those columns did not change.

```sql
-- The problem: a hot counter living in a wide row.
-- Every increment writes a NEW 50-column tuple (MVCC) and a dead old one.
UPDATE posts SET view_count = view_count + 1 WHERE id = $1;
```

At a few hundred views per second across popular posts, this is a measurable load of write amplification and vacuum pressure, and it is pushing your wide rows (with their TOAST pointers and bulky metadata) through the buffer cache far more than the read traffic alone would. The fix is **vertical partitioning**: split the columns by how often they are written, not by what they mean.

![Vertical partitioning: split the hot column out of the wide row](/imgs/blogs/schema-design-for-scale-5.webp)

```sql
-- After: the hot counter lives in its own narrow table.
CREATE TABLE post_counters (
  post_id    bigint PRIMARY KEY REFERENCES posts(id),
  view_count bigint NOT NULL DEFAULT 0
);

-- Now the increment rewrites an 8-byte payload row, not the article body.
UPDATE post_counters SET view_count = view_count + 1 WHERE post_id = $1;
```

The `posts` table is now cold: it is read often but written rarely, so its rows stay resident in cache and rarely generate dead tuples. The `post_counters` table is hot, but each row is tiny, so the rewrite is cheap and the vacuum churn is over a table that is pure key-plus-counter. You have separated the read-mostly mass of the data from the write-heavy sliver, which is exactly the separation MVCC and the buffer cache reward.

The second-order gotcha: vertical partitioning re-introduces a join for any read that needs both the cold columns and the hot counter on the same page. That is usually fine — you fetch the post once and the count once, and the count read is a tiny single-row lookup — but if you find yourself joining the two tables back together on a hot path constantly, you have split the wrong way. The split is justified only when the access patterns for the two column groups genuinely diverge, which for a counter-in-an-article they do.

## 6. Append-only and event-style modeling for write-heavy paths

**The senior rule of thumb: when many writers contend on one mutable row, stop updating it. Insert instead, and derive the current state by folding the inserts.**

In-place updates are the enemy of write throughput when they collide. If every checkout updates the same `inventory` row, or every bid updates the same `auction` row, the writers serialize on a row-level lock and your throughput is capped by how fast one row can be updated, no matter how many cores you have. The append-only model sidesteps the lock entirely: each change becomes a new immutable row, writers never touch the same row, and the current state is computed by folding the event stream.

![Append-only order log: state is a fold over immutable events](/imgs/blogs/schema-design-for-scale-7.webp)

The figure shows an order modeled as a log of events: `OrderCreated`, `ItemAdded`, `ItemAdded`, `PaymentCaptured`, `Shipped` — each one an `INSERT`. There is no row that all five writers fight over, so there is no contention to serialize on. The current state of the order is whatever you get by folding those events in sequence.

```sql
CREATE TABLE order_events (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  order_id    bigint NOT NULL,
  seq         int    NOT NULL,          -- per-order ordering
  type        text   NOT NULL,          -- 'created','item_added','paid',...
  payload     jsonb  NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (order_id, seq)
);
CREATE INDEX order_events_by_order ON order_events (order_id, seq);

-- Writers never UPDATE; they only INSERT. No row-level lock contention.
INSERT INTO order_events (order_id, seq, type, payload)
VALUES ($1, $2, 'item_added', $3);
```

Reading the current state is a fold over the log, which you can do on demand for low-traffic entities or maintain as a projection for hot ones:

```sql
-- Materialize current state by folding the log (or keep a projection table).
SELECT order_id,
       max(created_at)                                       AS updated_at,
       bool_or(type = 'paid')                                AS is_paid,
       jsonb_agg(payload) FILTER (WHERE type = 'item_added') AS items
FROM   order_events
WHERE  order_id = $1
GROUP  BY order_id;
```

This is append-only modeling, and at the extreme it becomes full event sourcing where the log is the source of truth and every read model is a projection. You do not have to go that far to get the benefit — even a hybrid where you keep a mutable `orders` row for the current state but append to an `order_events` log for the history gives you a contention-free audit trail and a way to rebuild state if a projection gets corrupted. The cost is real, though: reads now require a fold (so you maintain projections for hot entities), the log grows without bound (so you need partitioning and retention), and reasoning about "the current value" is harder than reading one row. Reach for it specifically when write contention on a mutable row is the bottleneck, not as a default.

## 7. Precomputed counters and aggregate tables

**The senior rule of thumb: if you need a count or a sum on every read, do not compute it on every read. Maintain it on write.**

`SELECT count(*) FROM comments WHERE post_id = $1` looks innocent and is one of the most common scaling traps. Its cost is proportional to the number of comments, and on a popular post that is a scan of thousands of rows on every single page render. Multiply by the read rate and you have a database spending most of its time counting things it counted a millisecond ago.

The fix is a maintained aggregate: a small table that holds the precomputed count, updated on the write path so the read is a single-row lookup.

![Counter table: pay the cost on write, not on every read](/imgs/blogs/schema-design-for-scale-8.webp)

The figure shows the two paths. The write path appends the comment to the durable `comments` table and bumps `post_stats.comment_count` in the same transaction. The read path never counts anything; it reads one row from `post_stats`. You have moved the cost from "every read" to "every write," which is a good trade because the count is read far more than it changes.

```sql
-- Naive: O(rows) on every read.
SELECT count(*) FROM comments WHERE post_id = $1;   -- scans every comment

-- Maintained aggregate: O(1) read, cost paid on write.
CREATE TABLE post_stats (
  post_id       bigint PRIMARY KEY,
  comment_count bigint NOT NULL DEFAULT 0,
  updated_at    timestamptz NOT NULL DEFAULT now()
);

-- On write, atomically bump the counter (UPSERT keeps it correct under races).
INSERT INTO post_stats (post_id, comment_count)
VALUES ($1, 1)
ON CONFLICT (post_id)
DO UPDATE SET comment_count = post_stats.comment_count + 1,
              updated_at    = now();
```

Two second-order concerns separate a toy counter from a production one. First, **the counter row is now itself a hot mutable row**, which re-introduces the contention from the previous section — if one post gets a thousand comments a second, that single `post_stats` row is a lock bottleneck. The standard fixes are sharded counters (write to one of N sub-counter rows at random, sum them on read) or batched increments (accumulate in memory or a queue and flush periodically). Second, **the counter can drift** if a write fails between inserting the comment and bumping the count, so a periodic reconciliation job that recomputes the true count and corrects the aggregate is not optional at scale — it is how you keep "approximately right" from becoming "wrong." YouTube's famous habit of freezing view counts at 301 was exactly this: the displayed number was a cheap approximation, and the real count was reconciled out of band before the system trusted it.

## 8. Soft deletes and audit columns: the hidden index cost

**The senior rule of thumb: a `deleted_at` column is not free — it adds a predicate to every query forever and quietly bloats every index unless you make it partial.**

Soft deletes (a `deleted_at timestamptz` that marks a row as gone without removing it) are a sensible default: they make deletes recoverable, preserve referential history, and let you audit what happened. But they change the shape of every query against the table. Once rows are soft-deleted, every read has to filter them out, and that predicate rides along on every query for the life of the table:

```sql
ALTER TABLE invoices ADD COLUMN deleted_at timestamptz;

-- Every read now carries this predicate, forever:
SELECT * FROM invoices WHERE account_id = $1 AND deleted_at IS NULL;
```

The non-obvious cost is in the indexes. A plain index on `account_id` still indexes the tombstoned rows, so the index grows with dead data and the engine reads through entries it will immediately discard. On a table where 40% of rows are soft-deleted, your indexes are 40% wasted. The fix is a **partial index** that only includes live rows:

```sql
-- Index only the rows anyone will ever read. Smaller, hotter, faster.
CREATE INDEX invoices_live_by_account
  ON invoices (account_id)
  WHERE deleted_at IS NULL;
```

The partial index is smaller, stays in cache better, and the planner uses it precisely for the `deleted_at IS NULL` queries that dominate. Audit columns have a similar hidden cost: a `created_by`, `updated_by`, `updated_at` set on a wide table widens every row and, if you index `updated_at` for "recently changed" queries, you have created a hot, ever-advancing index that is constantly being written to the right edge. None of these are reasons to skip soft deletes or audit trails. They are reasons to make the indexes partial, to consider moving the audit metadata to a sidecar table (vertical partitioning again), and to have a hard-delete or archival job that eventually moves truly dead rows out of the hot table so the tombstones do not accumulate without bound.

## Case studies from production

### 1. Uber's Schemaless: access-pattern modeling at the storage layer

When Uber outgrew a single Postgres instance, they did not build a normalized distributed SQL database — they built **Schemaless**, an append-only datastore on top of sharded MySQL. Each "cell" is an immutable row holding a JSON blob, keyed by an entity UUID and a column name, and writes only ever append new versions. This is nearly every idea in this post at once: it is access-pattern-first (the key structure is built around how trips and riders are read), it is schemaless-on-purpose with the variable trip data in JSON, and it is append-only so writers never contend. The lesson Uber's engineers stressed is that the design was driven by the read and write patterns of the trip lifecycle, not by an entity diagram — they modeled the access, and the storage shape followed.

### 2. Instagram's 64-bit sharded IDs

Early Instagram needed primary keys that were unique across thousands of logical shards, sortable by time, and narrow enough to index cheaply — and they explicitly rejected both auto-increment `int4` (runs out, not shard-unique) and random 128-bit UUIDs (kill index locality). Their solution was a 64-bit ID that packs a millisecond timestamp, a shard identifier, and a per-shard sequence into a single `bigint`, generated in Postgres with a stored procedure. The IDs stay time-sortable so inserts are sequential, they fit in 8 bytes, and they encode their own shard so routing is free. It is the canonical worked example of choosing the integer type and ID scheme for the access pattern, and it is detailed in [Instagram's sharding IDs in Postgres](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres).

### 3. Discord's message store: denormalize and stop counting

Discord stores trillions of messages, and their schema journey is a tour of this article. They use Snowflake-style 64-bit IDs (time-ordered, not random UUIDs) so the data is naturally partitioned by time. They denormalized aggressively because the dominant access pattern — "load the last N messages in a channel" — is a single partition range read, not a join. And they fought exactly the counter problem from section 7: counts and unread markers are maintained, not computed, because counting messages on read at their scale is a non-starter. When their original data store could not keep up with the write and read patterns, they migrated the storage engine, but the schema principle stayed: model for the one read that runs a billion times a day.

### 4. Notion's block model: one flexible table, sharded

Notion represents every piece of content — a page, a paragraph, a to-do, an embed — as a "block" row with a typed `type` column and a flexible `properties` payload, all in one table that they later sharded across Postgres instances. This is the JSONB-hybrid pattern at the core of a product: the queryable, structural fields (id, parent, type, position) are typed columns, while the wildly variable per-block content lives in a flexible payload. It let them support an effectively unbounded variety of block types without a schema migration per feature, and when they hit the scaling ceiling of one Postgres box, the uniform block shape made sharding tractable — covered alongside Figma in [Notion and Figma sharding Postgres](/blog/software-development/database-scaling/notion-and-figma-sharding-postgres).

### 5. The int4 overflow that froze a payments table

A payments company I worked with had a `ledger_entries` table with a `serial` (int4) primary key, designed years earlier when "we will never have two billion ledger entries" felt obviously true. Volume grew, the table crossed two billion rows over a holiday weekend, and inserts began failing with integer-out-of-range. The table was instantly read-only, which for a payments ledger means the business stops. The emergency fix was to add a new `bigint` column, backfill it in batches, swap it to the primary key, and re-point every foreign key — all online, under load, while the incident was active. It took the better part of a week and a frozen feature roadmap. The root cause was four bytes. Every primary key on a table that can grow should be `bigint` from day one; the cost of being wrong is not proportional to the savings.

### 6. Stripe-style immutable ledgers

Financial systems converge on append-only modeling for a reason beyond performance: correctness. A double-entry ledger never updates a balance in place — it appends immutable debit and credit entries, and the balance is a fold over them. Stripe and most serious payments infrastructure model money this way because an immutable log is auditable, reconcilable, and free of the update-contention and lost-update races that an in-place balance column invites. It is section 6 taken as a hard requirement rather than an optimization: when getting the number wrong is unacceptable, the append-only log is the conservative choice, and the running balance is a maintained projection (a counter table, section 7) that can always be rebuilt from the log.

### 7. The home feed that joined five tables

At a consumer app I advised, the home feed — 95% of all read traffic — was assembled with a five-table join on every request: feed items, authors, media, reaction counts, and the viewer's per-item state. It was a textbook-clean schema and it was melting the primary database. We did not optimize the join; we built a denormalized `feed_item_view` table keyed by `(viewer_id, ranked_position)` that inlined the author name and avatar, the media URLs, and the reaction count as a maintained aggregate, with the rarely-used long tail in a JSONB sidecar. The five-table join became one indexed range read. Write cost went up — publishing an item now fans out to followers' feed views — but reads outnumbered writes by four orders of magnitude, so the trade was overwhelmingly correct. This is the left-to-right move from the first figure, done in anger.

### 8. YouTube, X, and the cost of an exact count

Every large social product eventually stops showing exact counts in real time, and it is the same engineering lesson each time. YouTube's "301+ views" freeze, approximate like counts, "99+" notification badges — these are not laziness, they are the counter problem from section 7 at planetary scale. An exact `count(*)` on a hot object is impossible to serve on every read, so the count becomes a maintained aggregate that is sharded across many sub-counters to avoid lock contention, flushed in batches, and reconciled out of band. The displayed number is an approximation that converges to the truth, and the product is deliberately designed so that "approximately right, instantly" beats "exactly right, slowly." The schema decision — a counter table, sharded and reconciled — is downstream of accepting that tradeoff.

## When to denormalize, and when not to

Denormalization, narrow types, JSONB sidecars, vertical partitioning, append-only logs, and counter tables are all the same move in different clothes: spend more on the write path to make the dominant read cheap. That trade is right far more often than the textbook implies, but it is not free, and applying it everywhere produces a schema that is just as broken as the over-normalized one, only in the opposite direction.

**Reach for these techniques when:**

- A read on the hot path requires a multi-table join, especially across large or sharded tables, and that read dominates your traffic.
- A count, sum, or aggregate is needed on every read of a popular object.
- One column in a wide table is updated far more often than the rest, churning MVCC and vacuum.
- Many writers contend on a single mutable row and throughput is capped by row-level locking.
- The data has a genuinely variable or sparse shape that does not justify a column per attribute.
- A primary key is on a table that can grow without a hard, small ceiling — make it `bigint` and time-ordered before you ship.

**Keep it normalized, and skip the cleverness, when:**

- The workload is write-heavy and read-light, so the join cost you would denormalize away is rarely paid anyway.
- The tables are small enough to live entirely in cache, where joins are effectively free and denormalization only adds consistency risk.
- The data shape is still in flux and you do not yet know the access patterns — premature denormalization locks in the wrong shape, and reshaping a denormalized schema is more painful than reshaping a normal one.
- The copies you would create cannot be kept consistent cheaply, so the write-side fan-out would introduce correctness bugs worse than the latency you are saving.
- You are reaching for JSONB to avoid the work of deciding what your data is — that is not flexibility, it is deferred pain.

The discipline that ties it all together is the one from section 2: enumerate the access patterns first, find the read that runs most often and matters most, and shape the schema so that read is a single-key, single-shard fetch. Everything else — the types, the partitioning, the counters, the logs — is in service of that one read. Design for how you read, not just how your entities relate, and the schema will scale with you instead of against you.

## Further reading

- [Choosing a shard key: the one decision you can't take back](/blog/software-development/database-scaling/choosing-a-shard-key) — the partitioning-level half of designing for access patterns.
- [Index strategy at scale](/blog/software-development/database-scaling/index-strategy-at-scale) — once the schema is right, the indexes make the access patterns fast.
- [Random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) — the full case against random primary keys.
- [OLTP versus OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) — why your analytical reads want a different physical layout entirely.
- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — where schema design sits in the larger sequence of scaling moves.

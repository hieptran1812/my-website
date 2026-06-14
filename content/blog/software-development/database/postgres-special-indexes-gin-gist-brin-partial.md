---
title: "Beyond B-Trees: GIN, GiST, BRIN, Partial, and Expression Indexes"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles tour of the Postgres index types most engineers never reach for — GIN inverted indexes, GiST and SP-GiST trees, BRIN block-range summaries, plus partial and expression indexes — with runnable SQL, EXPLAIN output, and a decision playbook for picking the right one."
tags:
  [
    "postgres",
    "indexing",
    "gin-index",
    "gist-index",
    "brin-index",
    "partial-index",
    "expression-index",
    "jsonb",
    "full-text-search",
    "time-series",
    "database",
    "query-optimization",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-1.webp"
---

Most engineers know exactly one Postgres index type, and they call it "an index." They mean a B-tree. It is the default, it answers `=` and `<` and `BETWEEN` and `ORDER BY`, and for a huge fraction of workloads it is the correct and only answer. But the moment your `WHERE` clause stops looking like `column = value` and starts looking like `tags @> '{urgent}'`, `metadata @> '{"plan":"pro"}'::jsonb`, `to_tsvector(body) @@ to_tsquery('postgres & index')`, `location <-> point(...)`, or `created_at > now() - interval '1 hour'` on a two-billion-row append-only table, the B-tree either silently degrades into a full scan or refuses to participate at all. Postgres has a whole second tier of access methods built precisely for these shapes, and the cost of not knowing them is measured in sequential scans, 30-second dashboards, and `pg_indexes` rows that are larger than the table they index.

This post is a tour of that second tier: **GIN** (the inverted index behind arrays, JSONB, and full-text search), **GiST** and its space-partitioned cousin **SP-GiST** (the trees behind ranges, geometry, and nearest-neighbour search), **BRIN** (the tiny block-range summary that indexes a billion-row time-series table in a few hundred kilobytes), **hash** (equality-only, finally crash-safe since PG10), and the two orthogonal modifiers — **partial** indexes (a `WHERE` clause on the index itself) and **expression** indexes (index a computed value, not a raw column). The thread running through all of them is the same one that explains B-trees: an index is a deal you cut with the planner, trading write cost and disk for the ability to answer a specific *shape* of query without reading every row. Each access method cuts that deal differently, and the art is matching the method to the data shape and the operator you actually filter with.

![A five-by-four decision matrix mapping data shapes like scalar, array, tsvector, geometry, and append-only against query patterns, with each cell naming the winning index type](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-1.webp)

The matrix above is the mental model for the entire post, and it is worth staring at before we go deep on any single method. The rows are *data shapes* — a scalar column, an array or JSONB document, a full-text `tsvector`, a geometry or range type, and a naturally-ordered huge table. The columns are *query patterns* — point lookup, range scan, containment, and nearest-neighbour. Every cell names the access method that wins that pairing, and the punchline is that **no single index type fills the matrix**. B-tree owns scalar lookups and ranges. GIN owns containment over composite values. GiST owns overlap and nearest-neighbour. BRIN owns range scans over enormous ordered tables for a thousandth of the disk. The rest of this article is a tour of why each cell is the way it is, with SQL you can paste into `psql` and `EXPLAIN` output showing the planner actually using the index. If you have not read it, [B-Trees: How Database Indexes Really Work](/blog/software-development/database/b-trees-how-database-indexes-work) is the prerequisite — everything here assumes you understand why the unit of cost is the disk page, not the comparison.

## Why "just add an index" is the wrong instinct

Before we build anything, name the mismatch directly. The instinct that gets people into trouble is treating "add an index" as a single, type-free operation, as if the database has one kind of index and the only decision is *which column*. The truth is closer to the opposite: choosing the column is the easy part, and choosing the *access method* and *predicate* is where the leverage — and the footguns — live.

| You assume | The naive view | The reality |
| --- | --- | --- |
| An index is a B-tree | One structure, pick a column | Postgres ships six access methods, each tuned to a different operator class |
| Indexing a JSONB column speeds up JSONB queries | `CREATE INDEX ON t (data)` | A plain B-tree on `jsonb` indexes the *whole document as one opaque value*; `@>` containment never uses it |
| Bigger index = faster | More index is more speed | A partial index over 1% of rows can be 100× smaller *and* faster than the full one |
| Indexes are exact | Index match = answer | GIN and GiST can be **lossy** — the index returns *candidates*, and the heap must recheck each one |
| An index on `email` helps `WHERE lower(email) = ...` | Same column, same index | The planner needs the index to match the *expression*; a raw-column index is useless here |
| A range scan on a 2B-row table needs a 40 GB B-tree | Index is proportional to rows | A BRIN index on the same column can be 30 MB — three orders of magnitude smaller |

The single most important row is the fourth: **lossy indexes**. A B-tree is exact — if the index says a key is at TID 5, the row is there, no recheck needed. GIN and GiST routinely store *approximations*: a GiST internal node holds a bounding box that says "the real geometry is somewhere inside this rectangle," and a BRIN range says "some row in blocks 256–383 has a timestamp between 04:00 and 05:59." Those are *candidate* answers. The planner adds a **recheck** step that re-reads the actual heap tuple and applies the original predicate to throw out false positives. This is not a bug; it is the entire reason these indexes can be small and fast. But it means an `EXPLAIN ANALYZE` on a GiST or BRIN scan will show a `Rows Removed by Index Recheck` line, and if that number is large, your index is doing more harm than good. We will see this concretely in every section.

> The first rule of Postgres indexing past the B-tree: the access method is chosen by the operator, not the column. If you do not know which operator class your `WHERE` clause invokes, you cannot pick the right index.

This framing — operator-driven, not column-driven — is the multi-dimensional indexing problem Kleppmann describes in *Designing Data-Intensive Applications*, Chapter 3. A standard B-tree (or LSM-tree) gives you a one-dimensional sorted key, which is perfect for "all the rows between X and Y on a single axis." But a geospatial query — "all the restaurants within this latitude *and* longitude box" — is fundamentally two-dimensional, and concatenating lat and lon into one B-tree key does not help, because the sort order interleaves the two axes in a way that makes a box query touch enormous swaths of the tree. Kleppmann's point is that you need a *specialized* structure (he names R-trees, the GiST family, and inverted indexes for full-text) precisely because the one-dimensional sorted index is the wrong shape for multi-dimensional, containment, and fuzzy-match queries. Postgres is one of the few mainstream databases that ships all of those structures in the box. Let us go through them.

## 1. GIN: the inverted index for composite values

**Rule of thumb: reach for GIN whenever a single row contains *many* indexable values and your query asks "does this row contain X?" — arrays, JSONB, and full-text are the three canonical cases.**

A B-tree indexes one key per row: the row has a `created_at`, the index has an entry for it. GIN — **Generalized Inverted Index** — inverts that relationship. It takes a *composite* value (an array of tags, a JSONB document, a `tsvector` of lexemes), explodes it into its component keys, and for *each key* stores the list of rows that contain it. The name "inverted index" comes straight from search engines: instead of mapping document → words, you map word → documents. Ask "which rows have the tag `urgent`?" and GIN hands you the posting list for `urgent` directly, no scan required.

![A three-column diagram showing heap rows with tag arrays on the left, an entry tree of individual keys in the middle, and posting lists of row TIDs on the right, with arrows showing one row exploding into many keys and many rows sharing one key](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-2.webp)

The figure shows the two structures inside a GIN index, straight from the [Postgres GIN documentation](https://www.postgresql.org/docs/current/gin.html). On the left, three heap rows, each holding a small array of tags. In the middle, the **entry tree** — a B-tree whose keys are the *distinct elements* extracted from all the rows (`brin`, `gin`, `jsonb`, `sql`). On the right, each entry's **posting list**: the sorted list of TIDs (tuple identifiers — physical row addresses) for every row that contains that key. Row 12 has three tags, so it appears in three posting lists; the key `jsonb` is shared by rows 12 and 88, so its posting list is `[12, 88]`. A containment query like `tags @> '{gin,jsonb}'` becomes "fetch the posting list for `gin` (`[12,88]`), fetch the posting list for `jsonb` (`[12,88]`), intersect them" — and the intersection, `[12]`, is the answer. That intersection of sorted integer lists is blisteringly fast, which is why GIN is the right tool for "contains all of" and "contains any of" queries.

There is a subtlety worth internalizing: when a key is rare (few matching rows), its posting list is small and stored inline in the entry-tree leaf. When a key is *common* — appearing in millions of rows — the posting list would blow up the leaf page, so GIN promotes it to a separate **posting tree**, a B-tree of its own holding delta-encoded TIDs. The delta encoding matters: TIDs in a posting list are stored sorted, and GIN stores the *differences* between consecutive TIDs rather than the absolute values, which compresses runs of nearby rows. This is the same trick search engines use to keep posting lists compact.

### GIN over arrays

The simplest GIN case is an array column. Say a `tickets` table tags each row with labels:

```sql
CREATE TABLE tickets (
    id          bigserial PRIMARY KEY,
    subject     text,
    labels      text[] NOT NULL DEFAULT '{}'
);

INSERT INTO tickets (subject, labels)
SELECT
    'ticket ' || g,
    (ARRAY['urgent','billing','bug','feature','spam','vip'])[
        1 + (random()*5)::int : 1 + (random()*5)::int
    ]
FROM generate_series(1, 2_000_000) g;
```

Now the query we care about: "all tickets labeled both `urgent` and `vip`." Without an index, that is a sequential scan over two million rows. With a GIN index on the array, it is a posting-list intersection:

```sql
CREATE INDEX idx_tickets_labels ON tickets USING gin (labels);

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, subject
FROM tickets
WHERE labels @> ARRAY['urgent','vip'];
```

```
 Bitmap Heap Scan on tickets  (cost=44.05..1893.21 rows=520 width=29)
                              (actual time=2.118..8.904 rows=611 loops=1)
   Recheck Cond: (labels @> '{urgent,vip}'::text[])
   Heap Blocks: exact=602
   Buffers: shared hit=618
   ->  Bitmap Index Scan on idx_tickets_labels
                              (cost=0.00..43.92 rows=520 width=0)
                              (actual time=1.984..1.984 rows=611 loops=1)
         Index Cond: (labels @> '{urgent,vip}'::text[])
   Planning Time: 0.214 ms
   Execution Time: 8.971 ms
```

Three things to read here. First, the `Bitmap Index Scan` is the GIN access — it intersects the posting lists for `urgent` and `vip` and produces a bitmap of matching TIDs. Second, the `Bitmap Heap Scan` reads only the 602 heap blocks that bitmap points at, not the whole table. Third, the `Recheck Cond` line: GIN over arrays with the default `array_ops` operator class is *not* lossy for `@>`, but the bitmap-scan machinery still lists a recheck because a lossy bitmap (one that overflowed `work_mem` and degraded to page-granularity) would need it. For small result sets the bitmap stays exact and the recheck is a no-op. Nine milliseconds for a two-million-row containment query is the GIN payoff.

The `array_ops` operator class supports `@>` (contains), `<@` (is contained by), `&&` (overlaps), and `=`. The one it does *not* accelerate is "find arrays of a specific length" or "the third element equals X" — GIN sees a set of values, not an ordered sequence, so positional queries fall back to a scan.

### GIN over JSONB

JSONB is where GIN earns its keep in modern application schemas, because half the industry now stores semi-structured documents in a `jsonb` column and then wonders why `data @> '{"plan":"pro"}'` is slow. A plain B-tree on the column is useless: `CREATE INDEX ON events (data)` indexes the *entire document as one comparable blob*, good only for `data = '{...exact...}'`. The containment operator `@>` needs GIN.

```sql
CREATE TABLE events (
    id      bigserial PRIMARY KEY,
    data    jsonb NOT NULL
);

INSERT INTO events (data)
SELECT jsonb_build_object(
    'user_id', (random()*100000)::int,
    'plan',    (ARRAY['free','pro','enterprise'])[1+(random()*2)::int],
    'country', (ARRAY['US','DE','VN','BR','IN'])[1+(random()*4)::int],
    'flags',   jsonb_build_object('beta', random() < 0.1)
)
FROM generate_series(1, 3_000_000);

CREATE INDEX idx_events_data ON events USING gin (data);

EXPLAIN (ANALYZE)
SELECT count(*) FROM events
WHERE data @> '{"plan":"pro","country":"DE"}';
```

```
 Aggregate  (cost=812.40..812.41 rows=1 width=8)
            (actual time=14.602..14.603 rows=1 loops=1)
   ->  Bitmap Heap Scan on events  (cost=92.10..811.90 rows=200 width=0)
                                    (actual time=4.110..14.001 rows=199540 loops=1)
         Recheck Cond: (data @> '{"country":"DE","plan":"pro"}'::jsonb)
         Heap Blocks: exact=18221
         ->  Bitmap Index Scan on idx_events_data
                                  (actual time=3.980..3.980 rows=199540 loops=1)
               Index Cond: (data @> '{"country":"DE","plan":"pro"}'::jsonb)
 Execution Time: 14.78 ms
```

The default JSONB operator class is `jsonb_ops`, which indexes *every key and every value* in the document as separate keys, so it supports the full set of JSONB operators: `@>` (contains), `@?` and `@@` (jsonpath match), `?` (key exists), `?|` (any of these keys), `?&` (all of these keys). That breadth costs space — `jsonb_ops` builds a large index because it has an entry for every key string and every scalar value.

If you only ever do `@>` containment and jsonpath queries, switch to the leaner `jsonb_path_ops`:

```sql
CREATE INDEX idx_events_data_path ON events USING gin (data jsonb_path_ops);
```

`jsonb_path_ops` does not index keys and values separately; it hashes each *root-to-leaf path* into a single key. That makes the index substantially smaller and `@>` lookups faster, at the cost of dropping the `?`, `?|`, and `?&` existence operators. The decision is mechanical: if your queries are pure containment, use `jsonb_path_ops`; if you need key-existence checks, stay on `jsonb_ops`. On the three-million-row table above, `jsonb_path_ops` typically lands 30–40% smaller than `jsonb_ops`.

| Operator class | Indexes | Operators supported | Relative size |
| --- | --- | --- | --- |
| `jsonb_ops` (default) | every key and every value | `@>`, `@?`, `@@`, `?`, `?\|`, `?&` | larger |
| `jsonb_path_ops` | hash of each full path | `@>`, `@?`, `@@` | ~30–40% smaller, faster `@>` |

There is one more pattern worth knowing: if you query a *single specific path* constantly — say `data->>'user_id'` — you do not want a whole-document GIN index at all. You want an **expression index** on that one path (`CREATE INDEX ON events ((data->>'user_id'))`), which is a tiny B-tree that answers equality and range on exactly that field. GIN is for "I query arbitrary paths"; an expression index is for "I always query *this* path." We come back to expression indexes in section 7.

### The pending list and the fastupdate tradeoff

**Rule of thumb: GIN is slow to update because one row insert can mean dozens of index inserts — and the pending-list mechanism that hides that cost from your writes leaks it into your reads.**

Here is the dark side of inverted indexes. When you insert one row with a 20-key JSONB document, a B-tree does one insert. GIN does *twenty* — one posting-list update per key. Naively, that makes GIN catastrophically slow to write. Postgres softens this with **fastupdate**, on by default, and it is the single most important GIN operational detail to understand.

![A branching diagram where an INSERT flows into an unsorted pending list, which both flushes via VACUUM into the entry tree and posting lists and is scanned by every search alongside the main tree](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-8.webp)

The figure traces the mechanism. With `fastupdate = on`, new entries do not go into the main entry tree at all. They are appended, unsorted, to a **pending list** — a simple linked list of pages. Appending is O(1) and cheap, so your `INSERT` returns fast. The pending list is later flushed into the real entry tree in bulk — during `VACUUM`, during `ANALYZE`, when an explicit `gin_clean_pending_list()` runs, or whenever the list grows past `gin_pending_list_limit` (default 4 MB), at which point the *unlucky* INSERT that crosses the threshold pays the foreground cost of the flush. Bulk-flushing many entries at once amortizes the cost far better than inserting them one at a time, which is the whole point.

The leak is on the *read* side. Because the pending list is not part of the sorted entry tree, every search must scan the pending list *in addition to* the main index, applying the predicate to each unsorted entry by brute force. A small pending list is negligible. A large one — say you bulk-loaded a million rows and have not vacuumed — turns every query into "binary-search the tree, *then* linear-scan a million pending entries." This is the classic "my GIN index was fast yesterday and is slow today" incident, and the fix is almost always `VACUUM` or lowering `gin_pending_list_limit`.

The tuning levers, per the [Postgres docs](https://www.postgresql.org/docs/current/gin.html):

```sql
-- Per-index: turn fastupdate off when you need predictable read latency
ALTER INDEX idx_events_data SET (fastupdate = off);

-- Or cap the pending list small so flushes happen often and stay cheap
ALTER INDEX idx_events_data SET (gin_pending_list_limit = '256kB');

-- Force a flush right now (e.g. after a bulk load, before serving reads)
SELECT gin_clean_pending_list('idx_events_data'::regclass);
```

The senior move on bulk loads is to **drop the GIN index, load the data, recreate the index**. Building a GIN index from scratch uses a sort-based bulk algorithm that is dramatically faster than maintaining the index through millions of individual inserts — and the build is extremely sensitive to `maintenance_work_mem`, so bump it (`SET maintenance_work_mem = '2GB'`) before the `CREATE INDEX`. This is the same drop-load-recreate pattern that applies to any heavy index, but it matters most for GIN because per-row maintenance is its weakest point.

#### Second-order: GIN does not do range or sort

The non-obvious gotcha: GIN has no concept of order *between* keys in the way a B-tree does. It answers "does this contain X" beautifully and cannot answer "rows where the array's max element > 5" or "order by the JSONB field." If your query needs a range or a sort on a value, that value needs a B-tree (often an expression index on the extracted scalar), not GIN. A surprising number of slow JSONB dashboards are caused by someone GIN-indexing a document and then sorting on `data->>'created_at'` — GIN cannot help the sort, so the planner sorts the whole result set. Index the *extracted timestamp* with a B-tree expression index and the sort becomes free.

## 2. GiST: the generalized tree for ranges, geometry, and nearest-neighbour

**Rule of thumb: reach for GiST when your data is multi-dimensional or has a notion of *overlap*, *containment*, or *distance* — ranges that might intersect, geometries that might touch, points you want sorted by nearness.**

Where GIN is an inverted index, GiST — **Generalized Search Tree** — is a balanced tree, structurally a cousin of the B-tree and the R-tree, but parameterized over a handful of *support functions* so that one piece of tree-traversal machinery can index radically different data types. The brilliance of GiST, and the reason it has spawned so many extensions (PostGIS, `pg_trgm`, range types, `ltree`, `hstore`), is that the tree does not know or care what it is indexing. It just asks the operator class four questions — *can these two keys be unioned into a bounding predicate? how much does adding this entry expand a node? where should an overflowing page split? does this query predicate possibly match this subtree?* — and builds a balanced tree from the answers.

![A GiST tree on the left with a root MBR holding union of two child boxes, lossy bounding-box internal nodes, and exact-geometry leaves, beside a KNN visit-order panel showing the query point ordering two MBRs by minimum possible distance](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-3.webp)

The figure captures the two ideas that make GiST work, both drawn from the [Postgres GiST documentation](https://www.postgresql.org/docs/current/gist.html). On the left is the tree itself. The **leaf** nodes hold exact indexed values — the real geometry, the real range. The **internal** nodes hold a *lossy bounding predicate*: the smallest box (the "minimum bounding rectangle," MBR) that encloses everything in the subtree below. The root's MBR is the union of its children's MBRs. When you query "geometries that intersect this rectangle," the tree descends only into subtrees whose MBR overlaps your query rectangle, pruning the rest. Because the MBR is an approximation, a subtree whose box overlaps your query might contain no actually-matching geometry — so GiST is **lossy**, and the heap rechecks each candidate leaf against the exact predicate. That is the `recheck` flag in the `consistent` support function, and it is why GiST scans show `Rows Removed by Index Recheck`.

The right panel shows the second idea: **KNN, nearest-neighbour search**. This is GiST's superpower and the thing no B-tree can do. The optional `distance` support function lets GiST answer `ORDER BY column <-> query LIMIT k` — "the k things closest to this point/value" — *as an index scan*, returning rows already in distance order without sorting the whole table. The trick is that the `distance` function, evaluated on an internal node's MBR, returns the *minimum possible* distance from the query to anything in that subtree. So GiST runs a best-first search: it visits subtrees in order of their minimum possible distance, and the instant it has emitted k results closer than the next subtree's minimum, it stops. MBR A is at min-distance 2.1, MBR B at 5.4; once we have k points from A closer than 5.4, B is provably irrelevant and pruned. The distance must under-estimate (≤ the real distance) for this to be correct, which is exactly the lossy-bounding-box guarantee turned into an ordering.

### GiST for range types and exclusion constraints

The most broadly useful non-spatial GiST application is range types. Suppose you are building a booking system and need to guarantee no two reservations for the same room overlap in time:

```sql
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- lets GiST index the room_id scalar

CREATE TABLE reservations (
    id        bigserial PRIMARY KEY,
    room_id   int NOT NULL,
    during    tsrange NOT NULL,
    EXCLUDE USING gist (room_id WITH =, during WITH &&)
);
```

That `EXCLUDE USING gist` clause is a thing of beauty. It is an **exclusion constraint**: it tells Postgres "no two rows may have the same `room_id` (`WITH =`) *and* overlapping `during` ranges (`WITH &&`)." This is impossible to express with a `UNIQUE` constraint, because uniqueness only knows about equality, not overlap. Under the hood it is a GiST index on `(room_id, during)`, and the `&&` overlap operator is exactly what GiST's bounding-box machinery accelerates. Inserts that would overlap are rejected atomically:

```sql
INSERT INTO reservations (room_id, during)
VALUES (101, '[2026-06-14 09:00, 2026-06-14 11:00)');   -- ok

INSERT INTO reservations (room_id, during)
VALUES (101, '[2026-06-14 10:00, 2026-06-14 12:00)');
-- ERROR: conflicting key value violates exclusion constraint
--        "reservations_room_id_during_excl"
```

The `btree_gist` extension is what lets us mix a scalar equality column (`room_id WITH =`) into a GiST index — ordinarily GiST does not handle plain scalar equality well, but `btree_gist` provides operator classes that give GiST B-tree-like behaviour for scalars so they can ride along in a multi-column GiST index or exclusion constraint. Querying for overlap uses the same index:

```sql
EXPLAIN (ANALYZE)
SELECT * FROM reservations
WHERE room_id = 101
  AND during && '[2026-06-14 10:30, 2026-06-14 10:45)';
```

```
 Index Scan using reservations_room_id_during_excl on reservations
     (actual time=0.041..0.043 rows=1 loops=1)
   Index Cond: ((room_id = 101) AND (during && '[...)'::tsrange))
 Execution Time: 0.067 ms
```

### GiST for geometry and KNN with PostGIS

The reason GiST exists at industrial scale is geospatial. PostGIS stores geometries and indexes them with GiST over their bounding boxes — an R-tree in everything but name. The KNN ordering operator is where it shines:

```sql
-- "the 5 stores nearest to this point"
SELECT name, geom <-> ST_SetSRID(ST_MakePoint(-73.985, 40.748), 4326) AS dist
FROM stores
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(-73.985, 40.748), 4326)
LIMIT 5;
```

[Crunchy Data's deep dive on PostGIS nearest-neighbour search](https://www.crunchydata.com/blog/a-deep-dive-into-postgis-nearest-neighbor-search) explains the catch every PostGIS user eventually hits: because the index works on *bounding boxes*, the `<->` distance for non-point geometries (polygons, linestrings) is the distance between *bounding boxes*, which is inexact. For points it is exact; for polygons you often need a two-phase query — use the index to grab the nearest N candidates by bounding-box distance, then re-sort those candidates by true `ST_Distance`. The KNN system, as the [PostGIS workshop](https://postgis.net/workshops/postgis-intro/knn.html) documents, evaluates distances between bounding boxes in the R-tree index using a best-first traversal, which is the general GiST `distance`-function mechanism specialized to geometry.

GiST also backs trigram search (`pg_trgm`), which gives you index-accelerated `LIKE '%substring%'` and fuzzy similarity (`%` operator) — a different fuzzy-matching strategy than the full-text `tsvector` approach, and exactly the "fuzzy index" Kleppmann discusses in DDIA Chapter 3 as a complement to exact-match indexes.

#### Second-order: GiST builds and updates cost more than you expect

The non-obvious gotcha: GiST's `picksplit` (how an overflowing page divides its entries) and `penalty` (which subtree an insert goes into) are heuristics, and a *bad* split heuristic produces overlapping MBRs, which means queries descend into multiple subtrees and the index loses its pruning power. For PostGIS, the modern fix is to build with the sorted-build method (default since PG14 for many opclasses, using a space-filling curve to order entries), which produces tighter, less-overlapping boxes than the old insert-one-at-a-time build. If your GiST index is mysteriously slow, check whether it was built on an old version or with a non-sortsupport opclass, and `REINDEX` it. Also: GiST indexes are typically 1–2× the size of the equivalent B-tree and slower to update than a B-tree, because every insert may trigger MBR recalculation up the tree. They are not a free lunch; they are the only lunch available for overlap and distance.

## 3. SP-GiST: space-partitioned trees for points and prefixes

**Rule of thumb: reach for SP-GiST when your data partitions cleanly into *non-overlapping* regions — points in a plane (quadtree/k-d tree), or strings sharing prefixes (radix tree/trie).**

SP-GiST — **Space-Partitioned GiST** — is the less-famous sibling. Where GiST builds *balanced* trees with possibly-overlapping bounding boxes, SP-GiST builds *unbalanced* trees that recursively partition the space into *disjoint* regions. The classic examples, per the [Postgres SP-GiST docs](https://www.postgresql.org/docs/current/spgist.html) and [Egor Rogov's deep dive at Postgres Professional](https://postgrespro.com/blog/pgsql/4220639), are quadtrees and k-d trees for points, and radix trees (tries) for text.

The distinction that matters operationally is *overlap*. A GiST R-tree over points can have sibling MBRs that overlap, so a point query might have to descend into several subtrees. A quadtree (SP-GiST) splits the plane into four disjoint quadrants at each level, so a point belongs to exactly one quadrant — no overlap, and a point lookup follows a single root-to-leaf path. For data that partitions cleanly, that disjointness makes SP-GiST faster and smaller than GiST.

```sql
-- Points: SP-GiST quadtree, great for "what's at this exact location"
CREATE INDEX idx_sensors_loc ON sensors USING spgist (location);

-- Text prefixes: SP-GiST radix tree (text_ops), great for "starts with"
CREATE INDEX idx_domains_name ON domains USING spgist (name text_ops);

EXPLAIN (ANALYZE)
SELECT * FROM domains WHERE name LIKE 'api.%';
```

```
 Index Only Scan using idx_domains_name on domains
     (actual time=0.028..0.512 rows=4120 loops=1)
   Index Cond: ((name ~>=~ 'api.') AND (name ~<~ 'api/'))
   Filter: (name ~~ 'api.%'::text)
 Execution Time: 0.71 ms
```

The radix tree stores each distinct *prefix* once and branches on the next character, so `api.github`, `api.stripe`, and `api.twilio` share the `api.` node — which is why prefix queries (`LIKE 'api.%'`) are so cheap and why the index is compact on data with shared prefixes (URLs, file paths, reverse-DNS names, IP ranges with `inet_ops`). The honest tradeoff: SP-GiST's unbalanced trees can degrade on pathological inputs, it does not support multi-column indexes as flexibly as GiST, and for many workloads a plain B-tree handles prefix matching (`text_pattern_ops`) just as well. SP-GiST is the right call specifically for *point* data with disjoint partitioning and for genuinely prefix-heavy text where the trie's prefix-sharing is a real space win. It is a precision tool, not a default.

## 4. BRIN: a billion rows indexed in kilobytes

**Rule of thumb: reach for BRIN when the table is *huge*, *append-mostly*, and the column you filter on is *physically correlated* with its values — the canonical case is a timestamp on a time-series table that only ever grows.**

BRIN — **Block Range Index** — is the most counter-intuitive and the most spectacular when it fits. Every other index stores something per *row*. BRIN stores something per *block range* — by default, one tiny summary tuple for every 128 consecutive 8 KB heap pages (1 MB of table). The summary is just the **min and max** of the indexed column across all rows in that block range. That is the entire index: a list of `(block range, min, max)` triples.

![A BRIN diagram with ordered heap block ranges on the left, each mapped to a tiny min/max summary tuple, and a range query on the right matching exactly one summary and pruning the other four block ranges](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-4.webp)

The figure shows why this is both tiny and powerful, and why it lives or dies on one property. The heap is a time-series table appended in timestamp order, so block range 0–127 holds the earliest timestamps, 128–255 the next, and so on. BRIN stores, per range, just `min` and `max`. A query `WHERE ts BETWEEN '04:30' AND '05:30'` checks each summary: does `[04:30, 05:30]` overlap `[min, max]`? Only the range `256–383` (min 04:00, max 05:59) overlaps, so Postgres reads *that one block range from the heap* and prunes the other four entirely — without reading a single heap page from them, [as the docs describe](https://www.postgresql.org/docs/17/brin.html). The index itself is microscopic: [Crunchy Data benchmarked](https://www.crunchydata.com/blog/postgres-indexing-when-does-brin-win) a BRIN at **24 KB versus a 21 MB B-tree** on the same column of a 42 MB table — a thousand-fold reduction. On a multi-terabyte table the difference is between a 30 MB BRIN and a 40 GB B-tree.

```sql
CREATE TABLE metrics (
    ts      timestamptz NOT NULL,
    sensor  int NOT NULL,
    value   double precision NOT NULL
);

-- Append two billion rows in timestamp order (this is the load shape BRIN needs)
INSERT INTO metrics (ts, sensor, value)
SELECT
    '2024-01-01'::timestamptz + (g || ' seconds')::interval,
    (random()*1000)::int,
    random()*100
FROM generate_series(1, 200_000_000) g;

CREATE INDEX idx_metrics_ts_brin ON metrics USING brin (ts);

EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM metrics
WHERE ts BETWEEN '2024-03-01' AND '2024-03-02';
```

```
 Aggregate  (actual time=58.114..58.115 rows=1 loops=1)
   ->  Bitmap Heap Scan on metrics  (actual time=1.902..49.221 rows=86400 loops=1)
         Recheck Cond: ((ts >= '2024-03-01') AND (ts <= '2024-03-02'))
         Rows Removed by Index Recheck: 41600
         Heap Blocks: lossy=1152
         Buffers: shared hit=1184
         ->  Bitmap Index Scan on idx_metrics_ts_brin
                                   (actual time=0.214..0.214 rows=11520 loops=1)
               Index Cond: ((ts >= '2024-03-01') AND (ts <= '2024-03-02'))
 Execution Time: 58.31 ms
```

Read the `Heap Blocks: lossy=1152` and `Rows Removed by Index Recheck: 41600` lines carefully — they are BRIN's signature. BRIN is *always* lossy: it identifies block *ranges* that might contain matches, never individual rows, so the heap scan reads every row in the matching ranges and rechecks each against the predicate. The 41,600 removed rows are the cost of block-granularity: they live in the same block ranges as real matches but fall outside the exact predicate. As long as the *number of block ranges* you must scan is small, that recheck cost is dwarfed by the disk you saved. The index scan itself read 11,520 summary tuples in 0.2 ms.

### Correlation is the whole game

**The one property BRIN requires is physical-logical correlation: rows that are close in value must be close on disk.** This is what [Cybertec means by "correlation, correlation, correlation"](https://www.cybertec-postgresql.com/en/brin-indexes-correlation-correlation-correlation/) and what [pganalyze covers in detail](https://pganalyze.com/blog/5mins-postgres-BRIN-index). You can measure it directly:

```sql
SELECT attname, correlation
FROM pg_stats
WHERE tablename = 'metrics' AND attname = 'ts';
```

```
 attname | correlation
---------+-------------
 ts      |        0.999998
```

A correlation near `1.0` (or `-1.0`) means the column's physical order tracks its logical order — exactly what an append-in-timestamp-order table gives you. A correlation near `0.0` means the values are scattered randomly across the disk, and BRIN is *useless*: every block range's `[min, max]` spans nearly the whole value domain, so no range can ever be pruned, and the "index scan" degrades to a full heap scan plus the overhead of checking summaries. This is the BRIN failure mode, and it is silent — the index builds fine, takes no space, and simply never prunes anything. If you BRIN-index a `user_id` column on a table where users are interleaved randomly, you have built a useless index that the planner will (correctly) ignore.

The corollary: BRIN and `UPDATE`-heavy or out-of-order-insert workloads do not mix, because they destroy correlation. BRIN's natural habitat is **append-only**: logs, events, IoT/sensor streams, GPS tracks, audit tables, partitioned time-series. There the inserts arrive in order, correlation stays near 1.0 forever, and BRIN gives you range scans for a rounding error of disk.

### Tuning pages_per_range and the recheck cost

The one knob is `pages_per_range` (default 128). It trades index size against pruning granularity. Smaller ranges mean more summary tuples (bigger index) but tighter `[min, max]` bounds and fewer rows rechecked per query; larger ranges mean a smaller index but coarser pruning. Crunchy's benchmark found that for narrow queries (100 rows) `pages_per_range = 16` ran in 3 ms versus 11 ms at the default 128, because the tighter ranges rechecked far fewer rows:

```sql
CREATE INDEX idx_metrics_ts_brin16 ON metrics
USING brin (ts) WITH (pages_per_range = 16);
```

The mental model: set `pages_per_range` so that a typical query's predicate spans a handful of block ranges, not hundreds and not a fraction of one. For very wide range scans the default is fine; for point-ish lookups on a huge table, shrink it. And for the case where correlation is *good but imperfect* — say, mostly-ordered with occasional stragglers — Postgres 14+ offers `minmax_multi` opclasses that store several min/max intervals per range instead of one, tolerating outliers without collapsing the whole range's bounds, a problem [Haki Benita documents well](https://hakibenita.com/postgresql-correlation-brin-multi-minmax).

#### Second-order: BRIN summaries go stale on append

The non-obvious gotcha: BRIN summaries for *newly appended* block ranges are not created instantly. New rows land in heap pages whose block range has no summary yet (the index treats unsummarized ranges as "always scan," so correctness is never at risk, but those ranges are not pruned). Autovacuum creates the summaries, or you force it with `SELECT brin_summarize_new_values('idx_metrics_ts_brin'::regclass)`. On a high-ingest time-series table, schedule summarization or rely on aggressive autovacuum, or your newest data — usually the most-queried — silently falls back to scanning unsummarized ranges. This is the BRIN equivalent of GIN's stale pending list: a maintenance task hiding behind a structure that looks zero-maintenance.

## 5. Hash indexes: equality, and only equality

**Rule of thumb: reach for a hash index only when you do pure equality lookups on a column whose values are large, and you have measured that the hash is meaningfully smaller than the equivalent B-tree.**

Hash indexes have a checkered history in Postgres. Before version 10 they were *not* WAL-logged, meaning they did not survive a crash and could not be replicated — a footgun severe enough that the manual told you not to use them. [Since PG10 they are WAL-logged and crash-safe](https://www.postgresql.org/docs/current/hash-index.html), which finally makes them usable. A hash index stores a hash of the key and supports exactly one operator: `=`. No range, no sort, no prefix, no inequality — equality and nothing else.

```sql
CREATE INDEX idx_sessions_token_hash ON sessions USING hash (token);

EXPLAIN (ANALYZE)
SELECT * FROM sessions WHERE token = 'a3f9c1...';
```

```
 Index Scan using idx_sessions_token_hash on sessions
     (actual time=0.029..0.030 rows=1 loops=1)
   Index Cond: (token = 'a3f9c1...'::text)
 Execution Time: 0.048 ms
```

The honest assessment: for most equality lookups, a B-tree is just as fast and far more flexible, and a B-tree on the same column *also* serves range and sort queries you might want later. The narrow case where hash wins is when the indexed value is *large* (long text, big hashes) — a hash index stores a fixed-size 4-byte hash code regardless of key size, so it can be substantially smaller than a B-tree that stores the full keys, which can matter for cache-residency on a wide key. But the planner only uses a hash index for `=`, so the moment a query wants `ORDER BY token` or `token LIKE 'a3%'`, the hash index sits idle and you wish you had a B-tree. Default to B-tree; reach for hash only with a measured size win on equality-only access.

## 6. Partial indexes: only index the rows you query

**Rule of thumb: if your queries always filter on a condition that selects a small fraction of the table, put that condition in the index with a `WHERE` clause — you get a smaller, hotter, faster index for free.**

A partial index is not a new access method; it is a *modifier* you can attach to any of them. It adds a `WHERE` predicate to the index definition so the index contains entries for *only* the rows matching that predicate. The two reasons this is powerful: the index is smaller (less disk, more of it cached), and a smaller index is faster to scan and cheaper to maintain on every write.

![A before-after comparison: a full index on 50 million rows with 49.5 million soft-deleted, 1.4 GB on disk and cold pages evicting hot ones, versus a partial index with WHERE deleted_at IS NULL covering 500K active rows at 14 MB that fits in cache](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-5.webp)

The figure is the canonical use case: **soft deletes**. Most applications never hard-delete; they set `deleted_at` and filter `WHERE deleted_at IS NULL` on every read. If 99% of rows are soft-deleted, a full index on the table indexes 50 million rows when your queries only ever touch the 500,000 live ones. A partial index drops the dead weight:

```sql
-- Full index: 50M entries, ~1.4 GB, mostly dead rows you never query
CREATE INDEX idx_users_email_full ON users (email);

-- Partial index: 500K entries, ~14 MB, exactly the rows you query
CREATE INDEX idx_users_email_active ON users (email)
WHERE deleted_at IS NULL;

EXPLAIN (ANALYZE)
SELECT id FROM users
WHERE email = 'ada@example.com' AND deleted_at IS NULL;
```

```
 Index Scan using idx_users_email_active on users
     (actual time=0.022..0.023 rows=1 loops=1)
   Index Cond: (email = 'ada@example.com'::text)
 Execution Time: 0.041 ms
```

Two things to notice. The planner picked the partial index, and the `Index Cond` does *not* mention `deleted_at` — because every entry in the index already satisfies `deleted_at IS NULL`, the predicate is implied and free. For the planner to use a partial index, the query's `WHERE` must *imply* the index's predicate; here `deleted_at IS NULL` in the query matches the index's `WHERE deleted_at IS NULL` exactly. [Heap's classic post on partial indexes](https://www.heap.io/blog/speeding-up-postgresql-queries-with-partial-indexes) is the canonical reference for this pattern, and [Heroku's index guide](https://devcenter.heroku.com/articles/postgresql-indexes) covers it well too.

A partial **unique** index is the elegant solution to "email must be unique among active users, but a deleted user's email can be reused":

```sql
CREATE UNIQUE INDEX uq_users_email_active ON users (email)
WHERE deleted_at IS NULL;
```

A plain `UNIQUE (email)` would forbid reusing a deleted user's email; the partial unique index enforces uniqueness only over live rows, which is almost always what you actually want. [PHP Architect's piece on soft-delete unique patterns](https://www.phparch.com/2026/02/advanced-unique-index-patterns-for-soft-deletes-mysql-and-postgresql/) walks through the cross-database nuances.

Other high-value partial-index patterns:

```sql
-- Sparse boolean: index only the flagged rows (e.g. 0.1% of the table)
CREATE INDEX idx_orders_needs_review ON orders (created_at)
WHERE needs_review IS TRUE;

-- Hot status partition: index only the rows in active states
CREATE INDEX idx_jobs_pending ON jobs (priority, created_at)
WHERE status IN ('queued','running');

-- Exclude a dominant default value the query never selects
CREATE INDEX idx_events_errors ON events (occurred_at)
WHERE severity <> 'info';
```

The `WHERE` clause can use any expression you could put in a query: regexes, function results, `IN` lists, inequalities. The pattern is the same every time — *the query has a condition that is always present and highly selective; bake it into the index*.

#### Second-order: the planner must *prove* implication

The non-obvious gotcha: the planner uses a partial index only when it can *prove* the query predicate implies the index predicate, and its prover is good but not omniscient. `WHERE deleted_at IS NULL` matches `WHERE deleted_at IS NULL` trivially. But `WHERE status = 'queued'` does *not* automatically match an index with `WHERE status IN ('queued','running')` unless the planner can prove the implication — and for some expression shapes it cannot, so the index sits unused. The safe practice: make the query's predicate *syntactically* match or obviously imply the index's predicate, and verify with `EXPLAIN` that the partial index is actually chosen. A partial index the planner refuses to use is pure overhead. Also remember that a parameterized query (`WHERE deleted_at IS NULL AND email = $1`) where the constant is a literal works cleanly, but predicates depending on a *parameter value* the planner cannot see at plan time can defeat the implication proof.

## 7. Expression indexes: index the computed value

**Rule of thumb: if your `WHERE` clause wraps the column in a function or expression, the plain-column index cannot help — index the *expression itself*.**

The planner matches an index to a query *syntactically*. An index on `email` is an index on the value of `email`; a query `WHERE lower(email) = 'ada@example.com'` asks about the value of `lower(email)`, which is a *different* expression, so the index does not apply and Postgres computes `lower(email)` for every row in a sequential scan. The fix is an **expression index** (also called a functional index): index the output of the function, and the planner will match it to the same function in the query.

![A before-after comparison: a plain index on email forced to compute lower() per row in a sequential scan at 850 ms, versus an expression index on lower(email) that the planner matches directly for a 0.3 ms point lookup](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-6.webp)

The figure shows the transformation. On the left, the plain index on `email` is useless for the case-insensitive lookup; Postgres scans 5 million rows computing `lower()` on each, 850 ms. On the right, an index on the *expression* `lower(email)` stores the pre-computed lowercased value in sorted order, and the planner matches `WHERE lower(email) = $1` to it directly — a 0.3 ms point lookup:

```sql
CREATE INDEX idx_users_email_lower ON users (lower(email));

EXPLAIN (ANALYZE)
SELECT id FROM users WHERE lower(email) = 'ada@example.com';
```

```
 Index Scan using idx_users_email_lower on users
     (actual time=0.026..0.027 rows=1 loops=1)
   Index Cond: (lower(email) = 'ada@example.com'::text)
 Execution Time: 0.044 ms
```

The expression in the query must match the expression in the index *exactly* (modulo the planner's normalization). `WHERE lower(email) = $1` matches `(lower(email))`; `WHERE upper(email) = $1` does not. Expression indexes shine for a whole family of patterns:

```sql
-- Case-insensitive search (the classic)
CREATE INDEX ON users (lower(email));

-- Extract and index a single JSONB field as a typed scalar B-tree
CREATE INDEX ON events ((data->>'user_id'));
CREATE INDEX ON events (((data->>'amount')::numeric));

-- Index a date truncation for "group by day" / "rows on this day" queries
CREATE INDEX ON orders (date_trunc('day', created_at));

-- Index a concatenation or computed key
CREATE INDEX ON people ((first_name || ' ' || last_name));

-- Index the result of a normalization function
CREATE INDEX ON products (regexp_replace(sku, '[^0-9]', '', 'g'));
```

The JSONB-path expression index deserves emphasis, because it is the answer to the question section 1 left open. When you always query *one specific field* inside a JSONB document — `WHERE data->>'user_id' = '4242'` — you do not want a whole-document GIN index. You want a B-tree expression index on `(data->>'user_id')`, which is tiny, supports equality *and* range *and* sort on that field, and is exactly the right tool. GIN for arbitrary-path containment; expression B-tree for a known hot path. Knowing which one to reach for is the difference between a 14 MB targeted index and a 2 GB document index that still cannot sort.

Expression indexes compose with everything else. A *partial expression index* is completely legal and often ideal:

```sql
-- case-insensitive uniqueness, only over active users
CREATE UNIQUE INDEX uq_users_email_ci_active ON users (lower(email))
WHERE deleted_at IS NULL;
```

#### Second-order: the function must be IMMUTABLE, and stats get fuzzy

The non-obvious gotcha is twofold. First, the indexed expression must be `IMMUTABLE` — it must return the same output for the same input forever. You cannot index `now()` or a function that reads other tables, because the index would be wrong the moment the inputs change. `lower()`, `date_trunc()`, arithmetic, and JSONB extraction are immutable; a custom function defaults to `VOLATILE` and you must declare it `IMMUTABLE` (truthfully!) before you can index it. Second, the planner's row-count estimates for expressions can be worse than for raw columns, because Postgres gathers statistics on the *expression* only after the index exists and `ANALYZE` runs — so a freshly-created expression index may produce bad cardinality estimates until the next analyze. After creating an expression index on a hot path, run `ANALYZE` on the table so the planner has expression statistics, or you may see it under-use the very index you built.

## 8. Full-text search end to end

Full-text search is the showcase that ties GIN, expression indexing, and a purpose-built data type together, so it earns its own end-to-end walkthrough. The goal: given a `documents` table, answer "find documents matching these words, ranked by relevance" without an external search engine.

![A five-stage pipeline from raw document through to_tsvector lexemes, into a GIN index mapping lexemes to TIDs, through a to_tsquery match, ending at ts_rank ordering, with example SQL and data under each stage](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-7.webp)

The pipeline in the figure has five stages. **(1)** A raw document — title plus body text. **(2)** `to_tsvector` normalizes it into a `tsvector`: a sorted list of distinct **lexemes** (stemmed root words) with their positions, after removing stop words and applying the language's stemming rules, so "running," "ran," and "runs" all collapse to the lexeme `run`. **(3)** A **GIN index** over the `tsvector` inverts the lexemes — exactly the inverted index from section 1, mapping each lexeme to the TIDs of documents containing it. **(4)** A query is parsed by `to_tsquery` into a boolean combination of lexemes, and the `@@` match operator probes the GIN index for documents matching the boolean expression. **(5)** `ts_rank` scores the matches by relevance (term frequency, proximity, weights) so you can `ORDER BY` rank and return the top k.

The standard, durable way to set this up uses a stored generated `tsvector` column so the lexemes are computed once at write time, not on every query:

```sql
CREATE TABLE documents (
    id     bigserial PRIMARY KEY,
    title  text NOT NULL,
    body   text NOT NULL,
    -- generated column: tsvector recomputed automatically on write,
    -- with the title weighted higher (A) than the body (B)
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title,'')), 'A') ||
        setweight(to_tsvector('english', coalesce(body,'')),  'B')
    ) STORED
);

CREATE INDEX idx_documents_tsv ON documents USING gin (tsv);
```

Now the ranked query — find documents about Postgres indexing, best matches first:

```sql
EXPLAIN (ANALYZE)
SELECT id, title,
       ts_rank(tsv, query) AS rank
FROM documents, to_tsquery('english', 'postgres & index') AS query
WHERE tsv @@ query
ORDER BY rank DESC
LIMIT 10;
```

```
 Limit  (actual time=12.404..12.410 rows=10 loops=1)
   ->  Sort  (actual time=12.402..12.405 rows=10 loops=1)
         Sort Key: (ts_rank(documents.tsv, query.query)) DESC
         Sort Method: top-N heapsort  Memory: 27kB
         ->  Nested Loop  (actual time=0.402..11.881 rows=3184 loops=1)
               ->  Function Scan on query  (actual time=0.013..0.014 rows=1)
               ->  Bitmap Heap Scan on documents
                                  (actual time=0.380..10.901 rows=3184 loops=1)
                     Recheck Cond: (tsv @@ query.query)
                     Heap Blocks: exact=2933
                     ->  Bitmap Index Scan on idx_documents_tsv
                                  (actual time=0.341..0.341 rows=3184 loops=1)
                           Index Cond: (tsv @@ query.query)
 Execution Time: 12.51 ms
```

The `Bitmap Index Scan on idx_documents_tsv` is the GIN doing the inverted-index intersection: it finds the 3,184 documents containing *both* lexemes `postgres` and `index` (the `&` operator in `to_tsquery`). The heap then rechecks and `ts_rank` scores them, and the `top-N heapsort` keeps only the 10 best — note it sorts just the 3,184 candidates the index found, not the whole table. The `setweight` calls in the generated column are why a match in the title ranks above a match buried in the body. Multiple [tutorials](https://oneuptime.com/blog/post/2026-01-21-postgresql-full-text-search/view) and [practitioner guides](https://www.thegnar.com/blog/postgres-full-text-search) cover variations (`websearch_to_tsquery` for Google-style query strings, `ts_headline` for snippet highlighting), but the GIN-over-tsvector core is identical.

The tuning detail unique to full-text GIN: `gin_fuzzy_search_limit` puts a soft cap on how many rows a full-text search returns from the index (default 0 = unlimited), useful when a common lexeme would otherwise match millions of rows and you only need a sample. And the same fastupdate caveat from section 1 applies — a stale pending list slows full-text queries exactly as it slows JSONB ones.

#### When this is and is not enough

The honest boundary: Postgres full-text search with GIN is genuinely excellent for small-to-medium corpora — up to tens of millions of documents — and it spares you running and syncing a separate Elasticsearch cluster. Where it stops scaling: very large corpora with high update rates, sophisticated relevance tuning (BM25, learned ranking), faceting, and typo-tolerance beyond trigram fuzzy matching. At that point a dedicated search engine earns its operational cost. But the number of teams running Elasticsearch for a use case Postgres GIN would have handled in 12 ms is large, and the migration back is a recurring "we over-engineered the search" story.

## The comparison, and a decision playbook

We have toured five access methods and two modifiers. Now put them side by side. The figure compresses the tradeoffs into one matrix:

![A four-column comparison matrix of B-tree, GIN, GiST, and BRIN across relative size, insert cost, best query type, and whether the index is lossy and needs a recheck](/imgs/blogs/postgres-special-indexes-gin-gist-brin-partial-9.webp)

| | B-tree | GIN | GiST | SP-GiST | BRIN | Hash |
| --- | --- | --- | --- | --- | --- | --- |
| **Best for** | `=`, `<`, range, sort | containment over composite values | overlap, distance, KNN | disjoint partitions: points, prefixes | range on huge ordered tables | equality only |
| **Data shapes** | scalars | arrays, JSONB, tsvector | ranges, geometry, tsvector, trigram | points, text prefixes, inet | timestamps, sequences on big tables | scalars |
| **Key operators** | `=`, `<`, `>`, `BETWEEN`, `ORDER BY` | `@>`, `<@`, `&&`, `?`, `@@` | `&&`, `@>`, `<->` (KNN), `=` (btree_gist) | `=`, `<<`, `~>=~` (prefix), `<->` | `<`, `>`, `BETWEEN` (lossy) | `=` |
| **Relative size** | baseline 1× | 2–3× larger | ~1–2× | compact on prefix data | up to 1000× smaller | small (fixed hash) |
| **Insert cost** | cheap | expensive (many keys/row) | moderate (MBR updates) | moderate | near zero | cheap |
| **Lossy / recheck** | exact | exact (mostly) | lossy, rechecks | lossy, rechecks | always lossy, rechecks | exact |
| **Sort / range** | yes | no | partial (KNN order) | partial | range yes, no sort | no |
| **Crash-safe** | yes | yes | yes | yes | yes | yes (PG10+) |

The decision playbook, in the order you should actually ask the questions:

**Reach for a B-tree when** the column is a scalar and your queries are equality, range, or sort. This is 80% of all indexes; do not over-think it. If you also want covering / index-only scans, see [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans).

**Reach for GIN when** a single row holds many indexable values and you ask "does it contain X" — arrays with `@>`/`&&`, JSONB with `@>`, or full-text with `@@`. Use `jsonb_path_ops` if you only do containment. Budget for slow writes and watch the pending list.

**Reach for GiST when** your data has overlap or distance: range types and exclusion constraints, PostGIS geometry, and especially `ORDER BY col <-> point LIMIT k` nearest-neighbour queries that no other index can serve. Accept the lossy recheck and the moderate write cost.

**Reach for SP-GiST when** your data partitions into *disjoint* regions: points (quadtree/k-d tree) or prefix-heavy text and IP ranges (radix tree). It is a precision tool over GiST for these specific shapes.

**Reach for BRIN when** the table is huge, append-mostly, and the filtered column's `pg_stats` correlation is near ±1.0. You trade a tiny recheck cost for a thousand-fold smaller index. Verify correlation first; a BRIN on scattered data is a silent no-op.

**Reach for a hash index** essentially never as a first choice — only with a measured size win on a wide equality-only key.

**Layer a partial index** onto any of the above when your queries always carry a selective condition (soft-delete `IS NULL`, sparse boolean, hot status). Smaller, hotter, cheaper.

**Layer an expression index** onto any of the above when your `WHERE` wraps the column in an `IMMUTABLE` function — `lower()`, `date_trunc()`, a JSONB path extraction. Index the expression, then `ANALYZE`.

And the meta-rule that subsumes all of these, the same one from the intro: **the access method is chosen by the operator in your `WHERE` clause, not by the column's name or the table's size.** When a query is slow, the first move is not "add an index" — it is `EXPLAIN (ANALYZE, BUFFERS)`, read which operator the predicate invokes, and pick the access method whose operator class lists it. If you are not fluent reading that output, [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) is the companion skill that makes everything in this post actionable.

## Case studies from production

### 1. The JSONB index that doubled write latency

A team stored a 40-key event document in a `jsonb` column and added `CREATE INDEX ... USING gin (data)` with the default `jsonb_ops`. Read queries got fast; p99 write latency doubled and the index grew larger than the table. The wrong first hypothesis was "the table is too big." The actual root cause: `jsonb_ops` indexes every key *and* every value as a separate GIN key, so each 40-key insert triggered ~80 posting-list updates, and the index stored an entry for every distinct value across millions of documents. The fix had two parts: switch to `jsonb_path_ops` (containment-only, hashes whole paths, ~35% smaller and far fewer keys per row), and for the three fields the app actually queried by equality, replace whole-document indexing with targeted B-tree expression indexes on `(data->>'user_id')`, `(data->>'tenant')`, and `((data->>'amount')::numeric)`. Write latency returned to baseline and total index size dropped 4×. The lesson: a whole-document GIN index is the *fallback* for arbitrary-path queries, not the default for "I have a JSONB column."

### 2. The BRIN that indexed nothing

An analytics table of 1.8 billion rows had a 38 GB B-tree on `created_at` that nobody could afford to keep cached. An engineer read about BRIN, swapped in a `USING brin (created_at)` index — 22 MB, beautiful — and range queries got *slower*. The wrong hypothesis was "BRIN is just slow." The actual root cause: the table was loaded by a backfill job that inserted rows grouped by `tenant_id`, not by time, so `created_at` had a `pg_stats` correlation of `0.04` — physically scattered. Every block range's `[min, max]` spanned almost the entire date domain, so BRIN could never prune, and the planner scanned the whole heap. The fix was to `CLUSTER` the table on `created_at` (physically reordering it by time), after which correlation rose to `0.998` and BRIN pruned correctly, range queries dropping from 40 s to 600 ms. The lesson: BRIN's prerequisite is physical correlation, and you must *check* it (`SELECT correlation FROM pg_stats`) before trusting the tiny index.

### 3. The soft-delete index bloated with the dead

A `users` table used soft deletes; 96% of rows had `deleted_at` set. The unique index on `email` was 2.1 GB and every login query (`WHERE email = $1 AND deleted_at IS NULL`) scanned a tree dominated by deleted users. The wrong hypothesis was "we need more RAM to cache the index." The actual root cause: the index stored 50 million dead entries to serve 2 million live lookups. The fix was a partial unique index, `CREATE UNIQUE INDEX ... ON users (email) WHERE deleted_at IS NULL` — which dropped to 90 MB, fit comfortably in cache, and as a bonus correctly *allowed* a deleted user's email to be reused by a new signup, which the old full unique index had been wrongly forbidding. One index definition fixed a performance problem and a product bug simultaneously. The lesson: if a condition is always in your `WHERE`, it belongs in the index.

### 4. The case-insensitive login that scanned five million rows

A login endpoint ran `WHERE email = $1`, then someone made email case-insensitive by changing the query to `WHERE lower(email) = lower($1)`. Login p99 jumped from 2 ms to 900 ms overnight. The wrong hypothesis was "the auth service is overloaded." The actual root cause: the existing index was on `email`, but the query now asked about `lower(email)` — a different expression — so the planner fell back to a sequential scan computing `lower()` on five million rows per login. The one-line fix was `CREATE INDEX ON users (lower(email))`, an expression index matching the new predicate, restoring 0.3 ms lookups. The lesson, learned the hard way: wrapping a column in a function in your `WHERE` clause silently disables the plain-column index. Index the expression, or use `citext`.

### 5. The KNN query that a B-tree could never serve

A "stores near me" feature sorted by distance and limited to 20: `ORDER BY ST_Distance(geom, $point) LIMIT 20`. With a B-tree (impossible on geometry anyway) or even a GiST index but the *non-KNN* form, Postgres computed distance for every store and sorted — 1.2 s on 800k stores. The wrong hypothesis was "we need a separate geo service." The actual root cause: the query used `ST_Distance` (a plain function) in `ORDER BY`, which the index cannot drive, instead of the indexable KNN operator `<->`. Rewriting to `ORDER BY geom <-> $point LIMIT 20` let the GiST index run a best-first nearest-neighbour traversal, returning the 20 closest in index order — 4 ms. For non-point geometries they added a second pass re-sorting the 20 bounding-box candidates by exact `ST_Distance`. The lesson: KNN is GiST's unique capability, but only through the `<->` operator; the equivalent function call does not trigger it.

### 6. The full-text search that was an Elasticsearch cluster

A team ran a three-node Elasticsearch cluster solely to power site search over ~4 million help-center articles, with a fragile sync pipeline that drifted out of date weekly. The wrong assumption was "search needs a search engine." The reality: their query was keyword match with simple relevance ranking — exactly what `to_tsvector` + GIN + `ts_rank` does natively. They added a generated `tsvector` column with `setweight` (title A, body B), a GIN index, and a `websearch_to_tsquery` query. Median search latency was 11 ms, p99 under 40 ms, the sync pipeline disappeared (search was now transactionally consistent with the data), and three EC2 nodes were decommissioned. The lesson: for small-to-medium corpora with straightforward ranking, Postgres full-text search eliminates an entire moving part; reach for Elasticsearch when you genuinely need BM25, faceting, or hundreds of millions of frequently-updated documents.

### 7. The exclusion constraint that replaced application-level locking

A meeting-room booking service prevented double-bookings with application code: `SELECT` for overlapping reservations, then `INSERT` if none found. Under concurrency it double-booked anyway — a textbook check-then-act race. The wrong fix attempted was a `SELECT FOR UPDATE` advisory lock per room, which serialized all bookings and became a bottleneck. The actual fix was a database-level exclusion constraint: `EXCLUDE USING gist (room_id WITH =, during WITH &&)` with `btree_gist`. The GiST index made overlap detection atomic and concurrent — conflicting inserts fail with a constraint violation the app catches and retries, with no application locking at all. Throughput rose and the race was provably impossible. The lesson: overlap is a relationship `UNIQUE` cannot express but a GiST exclusion constraint can, and pushing the invariant into the database beats coordinating it in application code.

### 8. The GIN index that was fast in staging and slow in production

A JSONB containment query benchmarked at 8 ms in staging and ran at 400 ms in production on identical hardware and data volume. The wrong hypothesis was "production has noisy neighbours." The actual root cause: staging was freshly loaded and auto-vacuumed, so its GIN pending list was empty; production had a steady write stream and, due to a misconfigured autovacuum that rarely ran on that table, a multi-megabyte pending list that every query had to linear-scan. The fix was to lower `gin_pending_list_limit` to 256 kB on that index (so flushes happened often and stayed small) and fix autovacuum's thresholds so it actually ran. Query latency returned to 8 ms. The lesson: GIN's fastupdate hides write cost in a pending list that leaks into read latency; a "mysteriously slow GIN" is almost always a stale pending list, and `VACUUM` or a smaller limit is the cure.

### 9. The expression index the planner ignored

An engineer added `CREATE INDEX ON orders (date_trunc('day', created_at))` to speed up a daily-rollup query, confirmed it existed, and saw no improvement — the plan still showed a sequential scan. The wrong hypothesis was "expression indexes do not work." The actual root cause was twofold: first, the query used `created_at::date` while the index used `date_trunc('day', created_at)` — different expressions, no match. After aligning them, the planner *still* under-used the index because no `ANALYZE` had run since creation, so its cardinality estimate for the expression was wildly off and it judged a scan cheaper. After `ANALYZE orders`, the planner had real expression statistics and chose the index, cutting the rollup from 6 s to 200 ms. The lesson: an expression index requires the query's expression to match *exactly*, and it requires an `ANALYZE` afterward so the planner has statistics on the expression — without both, you have a correct index the planner refuses to use.

### 10. The BRIN summaries that never caught up

A high-ingest IoT table (50k rows/second, append-only by timestamp) used a BRIN index on `ts`, which worked beautifully for historical queries but was slow for the *most recent* hour — the data dashboards queried most. The wrong hypothesis was "BRIN is bad for recent data." The actual root cause: the newest heap block ranges had no summary tuples yet (autovacuum had not summarized them), so the index treated those ranges as un-prunable and the recent-hour query fell back to scanning unsummarized blocks. The fix was to schedule `SELECT brin_summarize_new_values('idx_ts_brin'::regclass)` every few minutes and tighten autovacuum on the table, so new ranges got summarized promptly. Recent-hour queries dropped from 8 s to 300 ms. The lesson: BRIN summaries for newly appended data are created lazily; on a high-ingest table you must summarize new ranges proactively or your freshest, hottest data silently loses index pruning.

### 11. The multicolumn GIN that beat two separate indexes

A product-catalog query filtered on both an array of `categories` and a JSONB `attributes` document: `WHERE categories @> '{electronics}' AND attributes @> '{"brand":"acme"}'`. With two separate GIN indexes the planner did a BitmapAnd of two bitmap scans — correct but it built two full bitmaps before intersecting. The wrong hypothesis was "two indexes is optimal because each is independently reusable." The actual improvement came from a single multicolumn GIN index `USING gin (categories, attributes)`, which (as the docs note) builds one entry tree over composite `(column-number, key)` values and can satisfy both predicates in one index scan, intersecting posting lists internally without materializing two separate bitmaps. For this always-paired query it cut latency ~40% and halved index maintenance overhead versus two indexes. The nuance: a multicolumn GIN only wins when the columns are *frequently queried together*; for independently-queried columns, separate indexes the planner can mix and match are more flexible. The lesson: match the index's *shape* to the query's shape — paired predicates favour one multicolumn GIN, independent predicates favour separate ones.

### 12. The partial index defeated by a parameter

A queue table indexed pending jobs with `CREATE INDEX ON jobs (priority) WHERE status = 'queued'`, and the worker query `WHERE status = $1 ORDER BY priority` was expected to use it. It did not — sequential scan every time. The wrong hypothesis was "the partial index is broken." The actual root cause: the query passed `status` as a *parameter* (`$1`), so at plan time the planner could not prove `$1 = 'queued'` implies the index's `WHERE status = 'queued'` predicate — the implication proof needs the literal, not a runtime parameter. The fix was to make the query use the literal `WHERE status = 'queued'` directly (the worker only ever pulls queued jobs anyway), after which the planner proved the implication and used the partial index, cutting queue-poll latency from 120 ms to under 1 ms. The lesson: the planner uses a partial index only when it can *prove* the query predicate implies the index predicate, and a hidden parameter value can defeat that proof — keep the selective predicate a literal so the implication is visible at plan time.

## When to reach past the B-tree, and when not to

**Reach past the B-tree when:**

- Your `WHERE` uses a *containment* operator (`@>`, `<@`, `&&`) over an array, JSONB document, or `tsvector` — that is GIN's exact job, and a B-tree literally cannot help.
- Your `WHERE` uses *overlap* (`&&`) on ranges or geometry, or you need an exclusion constraint forbidding overlapping rows — GiST.
- You need `ORDER BY col <-> target LIMIT k` nearest-neighbour ordering — only GiST's KNN can drive this from an index.
- The table is huge, append-mostly, and you filter on a physically-correlated column (timestamp on a time-series table) — BRIN turns a 40 GB index into a 30 MB one.
- Your query *always* carries a selective condition (soft-delete `IS NULL`, sparse boolean) — a partial index makes the index small and hot.
- Your `WHERE` wraps the column in an `IMMUTABLE` function — an expression index on that function is the only thing that restores index usage.

**Stay on the plain B-tree (or do not index at all) when:**

- The query is scalar equality, range, or sort — the B-tree is faster, smaller, and more flexible than any specialized method, and it *also* serves the next query you have not thought of yet.
- You would BRIN a column whose `pg_stats` correlation is not near ±1.0 — it will silently never prune. Check first.
- You are tempted to GIN a JSONB column you only ever query by one known field — use a targeted expression B-tree on that field instead; it is smaller and supports sort.
- The table is small (a few thousand rows) — a sequential scan is often faster than any index, and the planner knows it. Do not index a lookup table.
- You are adding an index "just in case" — every index is a tax on every write and a chunk of disk and cache. Index the queries you actually run, verified with `EXPLAIN`, not the ones you guess at.
- You would reach for a hash index without measuring — default to B-tree; hash wins only on a measured size advantage for equality-only access on wide keys.

The throughline, one last time: Postgres gives you a toolbox of access methods because no single structure can be simultaneously good at equality, range, containment, overlap, distance, and tiny-on-huge-tables. The skill is not memorizing six index types; it is reading your `WHERE` clause, identifying the operator and the data shape, and reaching for the method whose operator class was built for exactly that. Start from [the B-tree](/blog/software-development/database/b-trees-how-database-indexes-work), learn to [read the plan](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer), and the rest of the toolbox stops being exotic and starts being obvious.

## Further reading

- [Postgres GIN index documentation](https://www.postgresql.org/docs/current/gin.html) — the authoritative reference on the entry tree, posting lists, pending list, and operator classes.
- [Postgres GiST index documentation](https://www.postgresql.org/docs/current/gist.html) and [SP-GiST documentation](https://www.postgresql.org/docs/current/spgist.html) — support functions, lossy bounding boxes, and the KNN `distance` function.
- [Postgres BRIN documentation](https://www.postgresql.org/docs/17/brin.html), [Crunchy Data: when does BRIN win](https://www.crunchydata.com/blog/postgres-indexing-when-does-brin-win), [Cybertec: correlation, correlation, correlation](https://www.cybertec-postgresql.com/en/brin-indexes-correlation-correlation-correlation/), and [pganalyze on BRIN tuning](https://pganalyze.com/blog/5mins-postgres-BRIN-index).
- [Heap: speeding up Postgres with partial indexes](https://www.heap.io/blog/speeding-up-postgresql-queries-with-partial-indexes) and [Heroku's index efficiency guide](https://devcenter.heroku.com/articles/postgresql-indexes).
- [Crunchy Data: a deep dive into PostGIS nearest-neighbour search](https://www.crunchydata.com/blog/a-deep-dive-into-postgis-nearest-neighbor-search) and the [PostGIS KNN workshop](https://postgis.net/workshops/postgis-intro/knn.html).
- *Designing Data-Intensive Applications*, Martin Kleppmann, Chapter 3 — secondary indexes, multi-dimensional and full-text/fuzzy indexes, the reasoning behind specialized structures.

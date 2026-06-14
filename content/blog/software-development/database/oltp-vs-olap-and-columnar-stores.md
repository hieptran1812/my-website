---
title: "OLTP vs OLAP: Row Stores, Column Stores, and Why You Shouldn't Run Analytics on Your Primary Database"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Why transactional and analytical workloads have opposite shapes, how column stores turn a 50-column scan into a 3-column read with 40x compression, and why pointing a reporting query at your OLTP primary quietly kills production."
tags:
  [
    "oltp",
    "olap",
    "columnar-storage",
    "data-warehouse",
    "clickhouse",
    "analytics",
    "parquet",
    "compression",
    "database",
    "vectorized-execution",
    "snowflake",
    "system-design",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/oltp-vs-olap-and-columnar-stores-1.webp"
---

Every analytics outage I have seen started the same way. Someone needed a number — revenue by region for the last quarter, daily active users by cohort, the conversion funnel for a launch — and the fastest way to get it was to write a `SELECT ... GROUP BY` against the production database, because that is where the data lives. The query worked in staging. It ran fine the first few times in production at 2 a.m. Then one afternoon a product manager scheduled it to refresh a dashboard every five minutes, the table had grown 10x since the query was written, and at 14:03 the on-call engineer's phone lit up: p99 latency on the checkout path had jumped from 4 ms to 800 ms, requests were timing out, and the database CPU was pinned. Nobody deployed anything. The only thing that changed was that a single innocent reporting query started scanning half a billion rows on the same box that serves customer traffic.

This is not a tooling problem and it is not an incompetence problem. It is a workload-shape problem. The database your application writes to — Postgres, MySQL, whatever — is an **OLTP** (online transaction processing) engine. It is exquisitely tuned for many tiny operations that each touch one row by key. The query you just ran is an **OLAP** (online analytical processing) operation: one query that scans millions of rows but only cares about three columns. These two shapes are not just different in degree; they want opposite things from the storage engine, opposite memory behavior, opposite I/O patterns, and opposite physical layouts on disk. The reason the industry built an entire parallel universe of systems — Snowflake, BigQuery, Redshift, ClickHouse, DuckDB, Druid, Pinot, Parquet, Arrow — is that you genuinely cannot serve both shapes well from one engine, and the attempt to do so is what wakes up your on-call engineer.

The figure below is the mental model for the whole article: two workloads, two shapes, and everything downstream is a consequence of that mismatch. The rest of this piece is a tour of why the shapes differ, why **row storage** is wrong for analytics and **column storage** is right, how column stores get their absurd compression and speed, why running reports on your primary is a self-inflicted wound, and how the modern stack splits the two worlds cleanly.

## The two workload shapes

![OLTP does many tiny keyed row reads while OLAP scans millions of rows over few columns](/imgs/blogs/oltp-vs-olap-and-columnar-stores-1.webp)

Read the figure left to right. On the left is OLTP. A transactional workload is a firehose of small, independent operations: "fetch order 84213," "insert this payment," "decrement inventory for SKU 9921," "update user 55's email." Each one touches a handful of rows — often exactly one — located by primary key or a selective secondary index. The latency budget is single-digit milliseconds because a human or an upstream service is waiting. The request rate is high: tens of thousands to hundreds of thousands per second is normal for a busy backend. Crucially, when OLTP touches a row, it usually wants *the whole row* — name, email, status, address, all 50 columns — because it is rendering a profile page or executing business logic that reads and writes many fields together.

On the right is OLAP. An analytical workload is a small number of large queries. A dashboard might issue one query per panel; an analyst runs a handful of ad-hoc explorations; a nightly job recomputes a few hundred aggregates. But each query is enormous in the data it touches: "sum revenue by region and month for the last two years" scans every fact row in that window — hundreds of millions or billions of them. And here is the asymmetry that defines everything: that query reads `region`, `month`, and `amount`. It does not read `customer_email`, `shipping_address`, `user_agent`, or the other 47 columns. The latency budget is seconds, sometimes tens of seconds, because a human is exploring, not waiting on a checkout. The working set is the entire table, which is far larger than RAM.

Martin Kleppmann's *Designing Data-Intensive Applications* opens its data-warehousing discussion with exactly this distinction ("Transaction Processing or Analytics?"), and the table he draws — point reads vs bulk scans, latest state vs history of events, end-user-facing vs analyst-facing — is the canonical framing. Let me sharpen it into the contrast that actually drives storage design.

| Property | OLTP (transactional) | OLAP (analytical) |
| --- | --- | --- |
| Primary access pattern | Point lookups + small range scans by key | Sequential scans over large row sets |
| Rows touched per query | 1 to a few hundred | 10^6 to 10^11 |
| Columns touched per query | Most or all columns of a row | A handful out of dozens or hundreds |
| Query rate | 10^4 to 10^5 / sec | 10^0 to 10^2 / sec |
| Latency budget | < 1 to 5 ms (p99) | 100 ms to 60 s |
| Write pattern | Many small inserts/updates, in place | Bulk load / append, rarely updated |
| Data represents | Current state of the business | History of events / facts |
| Working set | Hot rows fit in buffer pool | Whole table, far exceeds RAM |
| Bottleneck | Random I/O, lock contention, latency | Bytes scanned, CPU per row, bandwidth |
| Right storage | Row-oriented (heap + B-tree) | Column-oriented |

> The single most useful sentence a backend engineer can internalize about databases: OLTP optimizes for *which rows*, OLAP optimizes for *which columns*. Get that backwards and you pay a 10-50x tax.

Notice that the bottleneck row is different in kind, not just degree. OLTP is latency-bound: the enemy is the random seek, the lock you wait on, the round trip. OLAP is throughput-bound: the enemy is the sheer number of bytes you have to drag off disk and the CPU cycles you burn per row. A storage engine that minimizes seeks (B-tree, careful row layout, big buffer pool of hot rows) is the *wrong* engine for minimizing bytes-scanned, and vice versa. That is the whole ballgame. Everything else in this article follows from it.

If you have read the companion piece on [pages, heap files & buffer pool](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool), you already know how an OLTP engine lays out rows: tuples packed into fixed-size pages, located through a B-tree or hash index, cached hot in a shared buffer pool. That design is superb for "give me row 84213." We are about to see why it is a disaster for "scan 500M rows and sum one column."

## Why row storage is the wrong shape for analytics

Here is a concrete schema. A retailer's `orders` table, simplified, with 50 columns; I will show four:

```sql
CREATE TABLE orders (
    order_id      BIGINT PRIMARY KEY,
    customer_id   BIGINT,
    region        TEXT,
    amount        NUMERIC(12,2),
    status        TEXT,
    shipping_addr TEXT,
    user_agent    TEXT,
    -- ... 43 more columns ...
    created_at    TIMESTAMPTZ
);
```

In a row store, the physical unit on disk is the *row*. All of `order_id`, `customer_id`, `region`, `amount`, `status`, and the other 45 fields for order 84213 are stored contiguously, one after another, inside a page. The next page holds the next batch of full rows. This is exactly what you want for `SELECT * FROM orders WHERE order_id = 84213`: one index lookup, one page read, and the whole row is right there.

Now run the analyst's query:

```sql
SELECT region, SUM(amount) AS revenue
FROM orders
WHERE created_at >= '2026-01-01'
GROUP BY region;
```

This query needs three columns: `region`, `amount`, and `created_at`. It needs them for every row in the date window — say 500 million rows. In a row store, the engine has no way to read just those three columns. To get `region` for a row, it must read the page that row lives on, and that page is full of `shipping_addr`, `user_agent`, and 45 other fields it does not want. If each row averages 400 bytes and you need 12 bytes of useful data per row, you are dragging roughly **400 bytes off disk to use 12** — a 33x read amplification. For 500M rows that is 200 GB of I/O to extract 6 GB of relevant data, and then you throw 97% of it away.

The figure makes the contrast physical.

![Row storage interleaves all fields of one row while column storage groups one column contiguously](/imgs/blogs/oltp-vs-olap-and-columnar-stores-2.webp)

On the upper right, the row-major blocks hold complete rows: `1 | US | 40 | PAID`, then `2 | US | 90 | OPEN`, then `3 | UK | 15 | PAID`. A `SUM(amount)` has to load every one of those blocks in full, even though it only consumes the third field of each. On the lower left, the column-major layout stores all of `id` together, all of `region` together, all of `amount` together, all of `status` together. Now `SUM(amount)` reads exactly one contiguous run — the green `amount` column — and the scan engine never touches `id`, `region`, or `status` at all. Three values, one tight stream of bytes, summed to 145.

This is the entire thesis of column-oriented storage, and Kleppmann states it cleanly in his column-storage section: *don't store all the values from one row together; store all the values from each column together instead.* If each column is in its own file (or its own contiguous region), a query reads and parses only the columns it references. The savings compound with the width of the table: a 200-column fact table where a query reads 3 columns gets a ~66x reduction in bytes scanned before any compression. Wide tables, which are exactly what analytics loves (denormalized facts with dozens of measures and dimensions), are precisely where row storage hurts most and column storage wins biggest.

There is a subtlety worth stating because it trips people up. Column storage does not change *what* data exists; the row "84213, customer 55, US, $40, PAID, ..." is still fully recoverable. It changes the *layout*: the value of column `region` for every row is the i-th entry of the region file, and the value of `amount` for the same row is the i-th entry of the amount file. To reconstruct a full row you read the i-th entry of every column file — which is exactly the operation column stores are *bad* at and OLTP needs constantly. That is why column stores are terrible OLTP engines: a single `SELECT *` for one order means seeking into 50 separate column files. The layout that makes scans cheap makes point-row reconstruction expensive. There is no free lunch; there is a workload-appropriate lunch.

| Operation | Row store | Column store |
| --- | --- | --- |
| `SELECT * WHERE id = ?` | 1 page read, whole row adjacent | 50 column seeks, reassemble row |
| `SUM(amount)` over all rows | Read every full row (33x waste) | Read one column run |
| `INSERT` one row | One page, in place | Append to 50 column files |
| `UPDATE one_field` of one row | Mutate one page | Rewrite/version a column segment |
| Add a new column | Cheap-ish (table rewrite or null) | Add one new column file |
| Compression ratio | Modest (mixed types per page) | Huge (homogeneous per column) |

## Why a column compresses so well

The bytes-scanned win is only half the story. The other half is compression, and it is where column storage stops being a 2-3x improvement and becomes a 10-50x one. The reason is almost embarrassingly simple: **a column is homogeneous.** Every value in the `region` column is a country code drawn from maybe 200 possibilities. Every value in `status` is one of `{PAID, OPEN, REFUNDED, CANCELLED}`. Every value in `created_at` is a timestamp very close to its neighbors. General-purpose compressors love low entropy, and a single column has dramatically lower entropy than a row, where a string sits next to a float sits next to a timestamp sits next to a UUID.

Kleppmann's column-compression section catalogs the techniques that exploit this, and they are exactly the encodings you find inside Parquet, ORC, and every column-store engine. The figure walks through the two workhorses.

![Sorted homogeneous columns compress with run-length and dictionary encoding into a fraction of raw size](/imgs/blogs/oltp-vs-olap-and-columnar-stores-3.webp)

**Run-length encoding (RLE).** If the column is sorted, or even just clustered, identical values appear in long runs. The raw column `US US US US US UK UK` is seven values; RLE stores it as `(US, 5) (UK, 2)` — two runs. On a real column where a single region might span millions of consecutive rows, RLE turns megabytes into a handful of run descriptors. This is why **sort order is a first-class tuning decision** in column stores, a point Kleppmann makes explicitly in his "sort order in column storage" discussion: the column you sort by gets spectacular RLE, and you can keep multiple differently-sorted copies of the data (replicas sorted on different keys) so that more queries hit a well-compressed, well-clustered layout.

**Dictionary encoding.** For low-cardinality columns, build a dictionary — `PAID = 0, OPEN = 1, REFUNDED = 2, CANCELLED = 3` — and store the column as a sequence of small integer codes instead of full strings. A 6-byte string `REFUNDED` becomes a 2-bit code. Then **bit-pack** those codes: if there are 4 distinct values you need 2 bits each, so four rows of status fit in a single byte instead of 24+ bytes of strings. Parquet implements exactly this as `RLE_DICTIONARY`: a dictionary page holds the unique values, and the data pages hold the codes using a hybrid RLE / bit-packing scheme. If the dictionary grows too large (high cardinality), Parquet falls back to plain encoding automatically.

**Bitmap encoding.** When cardinality is very low and you filter on equality a lot, you can go further and store one bitmap per distinct value: a bit per row that is 1 if the row has that value. `WHERE status = 'PAID'` becomes a single bitmap read; `WHERE status IN ('PAID','OPEN')` becomes a bitwise OR of two bitmaps. These bitmaps are themselves run-length compressed (most are long runs of zeros). This is precisely how Druid and Pinot accelerate multi-dimensional filtering — Druid uses Roaring or CONCISE compressed bitmaps and combines predicates with fast AND/OR on the compressed representation, which is why those systems can prune to the matching rows of a high-dimensional event stream in milliseconds.

**Delta and frame-of-reference (FOR).** For sorted numeric or timestamp columns, store the difference from the previous value (delta) or from a per-block base (frame-of-reference), then bit-pack the small differences. Monotonically increasing IDs and tightly clustered timestamps compress to a couple of bits per value. Parquet's `DELTA_BINARY_PACKED` is this encoding.

A worked example pins down the magnitude. Take 4 columns of a 100M-row order table:

```
region   (cardinality 200, clustered)  : dictionary 8-bit codes + RLE -> ~0.1 byte/row effective
status   (cardinality 4)               : 2-bit dictionary code        -> 0.25 byte/row
amount   (NUMERIC, high cardinality)   : FOR + bit-pack               -> ~3 bytes/row
created_at (sorted timestamps)         : delta + bit-pack             -> ~1.5 bytes/row
```

Stored row-major with all 50 columns, each row is a few hundred bytes and compresses modestly because the page mixes types. Stored column-major, each column compresses against its own homogeneous distribution, and the columns the query does not touch cost *nothing* to skip. The aggregate effect is not theoretical. When Cloudflare moved its HTTP analytics pipeline onto ClickHouse, the per-event storage dropped from **1630 bytes** (Cap'n Proto serialized) to **36.74 bytes** in ClickHouse's columnar `MergeTree` — a ~44x reduction — which collapsed a projected **273.93 PiB** of one-year raw storage to **18.52 PiB** with 3x replication, and the annual storage cost from a projected **$28M** to **$1.9M** ([Cloudflare engineering blog](https://blog.cloudflare.com/http-analytics-for-6m-requests-per-second-using-clickhouse/)). That is the column-store compression argument stated in dollars.

It is worth tabulating which encoding wins on which column shape, because choosing the right one (and choosing the right sort key to enable it) is the bulk of physical column-store tuning. The general-purpose block compressor (LZ4, Snappy, ZSTD) runs *on top* of these lightweight encodings, squeezing whatever structure remains; the lightweight encoding does the heavy lifting and the block compressor mops up.

| Encoding | Best for | Mechanism | Typical column |
| --- | --- | --- | --- |
| Run-length (RLE) | Sorted / clustered, long runs | Store (value, count) per run | `region`, `status` on a sorted table |
| Dictionary + bit-pack | Low cardinality | Map values to small integer codes | `country`, `device_type`, enums |
| Bitmap | Very low cardinality, equality filters | One compressed bit-vector per value | `is_active`, `gender`, small enums |
| Delta | Sorted numeric/time, small gaps | Store difference from previous | `created_at`, monotonic ids |
| Frame-of-reference (FOR) | Clustered numeric ranges | Store offset from per-block base | `price`, `amount` within a block |
| Plain | High cardinality, no structure | Store values verbatim | `url`, `uuid`, free text |

The last row is the warning. A high-cardinality column with no order — random UUIDs, raw URLs, free text — has nothing for these encodings to exploit; it stores roughly verbatim and is where your column-store bytes actually go. That is a schema-design signal: if a high-cardinality column dominates your storage, ask whether you can extract a low-cardinality projection of it (domain from URL, prefix from id) that most queries can use instead.

> Row storage forces a string, a float, and a timestamp to share a page, so a general compressor sees high-entropy noise. Column storage hands the compressor a million near-identical values in a row. The difference between those two inputs is the difference between a 2x and a 40x ratio.

## How a column store actually executes a query

Saving bytes off disk is necessary but not sufficient; if you then process those bytes one row at a time in an interpreter loop, you waste the win on CPU. The second pillar of column-store performance is the *execution model*, and Kleppmann covers it under "vectorized processing." The idea originates in the MonetDB/X100 line of research from CWI, which later became the Vectorwise (now Actian Vector) commercial engine, and it now underpins ClickHouse, DuckDB, and the modern execution layers of Snowflake and BigQuery.

The classic database executes a query with the **Volcano iterator model**: each operator has a `next()` method that returns one tuple, and operators pull tuples from each other one at a time. This is elegant and composable, and it is murder on a modern CPU. Every `next()` call is a virtual function dispatch, a branch the predictor can mispredict, and a fresh trip through cold cache lines. For a point query touching one row, the overhead is irrelevant. For a scan over 500M rows, the per-tuple overhead *is* the runtime.

Vectorized execution flips this. Instead of one tuple at a time, operators pass **vectors** — batches of typically 1024 to 4096 values from a single column — through the pipeline. The MonetDB/X100 paper describes choosing the vector size so the working set fits in the CPU's L1/L2 cache. A filter operator now runs a tight loop over 1024 contiguous integers with no per-row dispatch; the compiler can auto-vectorize it into SIMD instructions that test 8 or 16 values per cycle; the branch predictor sees one predictable loop instead of millions of unpredictable virtual calls; and the data stays hot in cache the whole time. ClickHouse's own description of vectorized execution makes the same point: process data in large batches to amortize interpreter overhead, emit SIMD, and keep tight working sets in cache.

Paired with vectorization is **late materialization**, shown in the figure.

![Vectorized execution filters compressed column batches before materializing wide columns for surviving rows](/imgs/blogs/oltp-vs-olap-and-columnar-stores-4.webp)

Read the pipeline left to right. Stage 1 reads the `price` column as a 1024-value vector — still compressed, still narrow. Stage 2 applies `price > 100` over the entire vector with branch-free SIMD, producing not rows but a **selection vector**: a list of the row positions that survived (in the figure, ids 1, 3, 5 — the values 180, 260, 410). At this point the engine has touched only the `price` column. Stage 4 is where late materialization earns its name: only *now*, for the rows that passed the predicate, does the engine fetch the wide columns `name` and `region`. If the predicate is selective and throws away 99% of rows, you have avoided reading and decoding 99% of the expensive wide columns. Stage 5 aggregates over the small surviving set. The naive alternative — "early materialization," reconstructing full rows up front and then filtering — would pay the wide-column I/O for every row before discovering most of them lose. Late materialization defers that cost until the predicate has done its pruning.

```python
# A toy vectorized filter + late materialization, NumPy standing in for SIMD.
import numpy as np

price  = np.fromfile("price.col",  dtype=np.int32)   # one column, contiguous
# name and region are wide/expensive; we do NOT load them yet.

# Stage 2: branch-free predicate over the whole vector -> selection vector.
sel = np.nonzero(price > 100)[0]        # row ids that survive, e.g. [1, 3, 5]

# Stage 4: late materialization -- fetch wide columns ONLY for survivors.
name   = load_column("name.col",   rows=sel)   # 3 reads, not 6
region = load_column("region.col", rows=sel)

# Stage 5: aggregate on the narrow result set.
import collections
acc = collections.defaultdict(int)
for r, p in zip(region, price[sel]):
    acc[r] += p
```

The combination is multiplicative. Column storage cuts the bytes you read; compression cuts them again; vectorization cuts the CPU per byte; late materialization cuts the bytes you decode for wide columns. A query that would be 200 GB of I/O and a billion `next()` calls in a row store becomes a few GB of compressed column reads processed in cache-friendly SIMD loops. That is how systems like DuckDB — an in-process, single-file OLAP engine built by the same CWI lineage — can run analytical queries over hundreds of millions of rows on a laptop in under a second, and even query Pandas or Parquet data in place with zero copies.

## Skipping data you never read: zone maps and clustering

The cheapest byte is the one you never read. On top of reading fewer columns, column stores read fewer *blocks* of the columns they do touch, using lightweight per-block statistics often called **zone maps** (Redshift's term), **min/max statistics** (Parquet/ORC), or **data skipping indexes** (ClickHouse). The mechanism is in the figure.

![Per-block min and max statistics let the scanner skip blocks whose range excludes the predicate](/imgs/blogs/oltp-vs-olap-and-columnar-stores-8.webp)

The column is divided into blocks (Parquet row-group column chunks, ClickHouse granules, Redshift 1 MB blocks). For each block, the writer records the min and max value of the column in that block, stored in metadata that is tiny relative to the data. When a query has a predicate like `WHERE amount > 200`, the scanner first checks each block's min/max. Block 1 has `max = 98`; since `98 < 200`, no row in that block can match, and the scanner skips it entirely — it never issues the I/O to read the block, never decompresses it, never touches it. In the figure, five of eight blocks have `max < 200` and are pruned; only blocks 3, 5, and 8 (whose ranges overlap 200) are read. The scanner did 62% less work for free, just from metadata it was going to store anyway.

The catch — and it is the catch that separates a good column-store schema from a useless one — is that **zone maps are only as good as your data ordering.** If the table is sorted (or clustered) on `amount`, the blocks have tight, non-overlapping ranges and `WHERE amount > 200` prunes almost everything. If the table is in random insertion order, every block's range spans nearly the full domain (`min ≈ 0, max ≈ huge`), nothing gets pruned, and the zone map is dead weight. This is the same idea as RLE liking sorted data, and it is why every serious column store gives you control over the sort/cluster order:

```sql
-- ClickHouse: ORDER BY is the on-disk clustering key, not just a query hint.
-- Choose columns you filter on most, low-cardinality first for good granules.
CREATE TABLE events (
    event_date  Date,
    region      LowCardinality(String),
    user_id     UInt64,
    amount      Decimal(12,2),
    status      LowCardinality(String)
) ENGINE = MergeTree
ORDER BY (event_date, region, amount)   -- clustering -> tight zone maps
PARTITION BY toYYYYMM(event_date);      -- partition pruning on top
```

```sql
-- Snowflake: micro-partitions are auto-built; clustering keys keep them tight.
ALTER TABLE events CLUSTER BY (event_date, region);
```

Snowflake's architecture leans on exactly this. Incoming data is broken into immutable **micro-partitions** of roughly 50-500 MB of columnar data, and Snowflake stores per-micro-partition metadata (min/max per column, distinct counts) that the optimizer uses to prune partitions before reading them. The 2016 Snowflake SIGMOD paper frames the whole system around decoupling compute from storage and pulling only the relevant micro-partitions from object storage into elastic virtual warehouses. ClickHouse's `MergeTree` granules and Parquet's row-group statistics are the same idea at different granularities. Get the clustering right and a "scan 2 years of data" query becomes "read the three months and two regions the predicate actually selects."

## Column-store file internals: Parquet, ORC, Arrow

So far "column store" has been an abstraction. Let me make it concrete with the file format that has become the lingua franca of analytics: **Apache Parquet**. If you have data in a lake — S3, GCS, HDFS — it is overwhelmingly likely to be Parquet, and Snowflake, BigQuery, Redshift Spectrum, DuckDB, ClickHouse, Spark, and Trino all read it. Understanding its layout is understanding column storage in the concrete.

![Parquet nests row groups into column chunks into pages so readers skip irrelevant columns and blocks](/imgs/blogs/oltp-vs-olap-and-columnar-stores-9.webp)

The hierarchy is file -> row group -> column chunk -> page, as the figure shows. A Parquet **file** ends with a footer that holds the schema and all the per-chunk statistics. Read the footer first and you know the structure of the whole file without scanning it. The file is divided into **row groups**, each a horizontal slice of roughly 128 MB of rows. Within a row group, the data for each column lives in a **column chunk**: all the `price` values for that row group are contiguous, all the `region` values are contiguous, and so on. Each column chunk carries its own metadata — encoding, compression codec, and min/max statistics (the zone map). Finally, each column chunk is a stream of **pages**, the smallest unit of compression and encoding: data pages hold the encoded values (RLE + bit-packing), and a dictionary page (for dictionary-encoded columns) holds the unique values.

This nesting is what makes the two skip mechanisms work together. To run `SELECT region, SUM(amount) WHERE amount > 200`:

1. Read the footer; learn the schema and per-chunk min/max.
2. For each row group, check the `amount` chunk's min/max; **skip row groups** that cannot contain `amount > 200` (block skipping).
3. For the surviving row groups, read **only** the `region` and `amount` column chunks; never touch the other 48 columns (column projection).
4. Decompress only the pages needed, decode dictionaries lazily, aggregate vectorized.

```python
# pyarrow reads only the columns and row groups it needs.
import pyarrow.parquet as pq
import pyarrow.compute as pc

pf = pq.ParquetFile("orders.parquet")
print(pf.metadata.num_row_groups)            # e.g. 64 row groups
print(pf.metadata.row_group(0).column(3).statistics)  # min/max for amount

# Column projection + predicate pushdown: Arrow skips chunks via stats.
table = pq.read_table(
    "orders.parquet",
    columns=["region", "amount"],            # only 2 of 50 columns read
    filters=[("amount", ">", 200)],          # row groups pruned by min/max
)
revenue = (
    table.group_by("region")
         .aggregate([("amount", "sum")])
)
```

ORC (the Hadoop-world cousin) is structurally similar: stripes instead of row groups, with stream-level encodings and built-in indexes including bloom filters for equality predicates. The design decisions — horizontal slicing for parallelism, per-column contiguity for projection, per-block statistics for skipping, per-page encoding for compression — are the same in both.

**Apache Arrow** is the in-memory counterpart. Where Parquet is the on-disk format optimized for compression and scan, Arrow is the in-memory columnar layout optimized for processing and zero-copy interchange. Columns are stored as contiguous, type-specific buffers (with a separate validity bitmap for nulls), which is exactly the layout vectorized engines and SIMD want. Because Arrow is a standard memory format, DuckDB can hand a result to Pandas, or Pandas can hand a DataFrame to DuckDB, *without serializing or copying* — the same buffers are reinterpreted. This is the plumbing that lets the modern Python data stack move millions of rows between tools for free.

The on-disk and in-memory split is itself a lesson: Parquet pays decode cost to win compression and I/O; Arrow pays memory footprint to win processing speed. You convert at the boundary. A scan reads Parquet, decodes into Arrow, and runs vectorized operators on Arrow buffers.

## Writing to a column store

Everything above optimizes reads, and that bias is not accidental — it follows from the workload. But data has to get *into* a column store somehow, and the write path exposes the central tension of the model: the layout that makes reads cheap makes writes awkward. Kleppmann's "writing to column-oriented storage" section names the problem directly. To insert one row into a sorted, compressed, column-major table you would, in the worst case, have to rewrite every column file to splice the value into the right sorted position and re-run its RLE/dictionary encoding. That is hopeless for anything but bulk load. So column stores almost universally refuse to do in-place inserts and instead borrow the trick from [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines): **buffer writes in memory, flush them as immutable sorted column segments, and merge segments in the background.**

ClickHouse's `MergeTree` is exactly this. An `INSERT` does not modify existing data; it writes a new immutable **part** — a self-contained directory of column files, sorted by the table's `ORDER BY` key, with its own zone-map marks. A background merge process periodically combines small parts into larger ones, re-sorting and re-compressing, the same way LSM compaction merges SSTables. Reads consult all live parts and merge their results. The benefits are the LSM benefits: writes are sequential appends (fast, no random I/O), each part is internally sorted (so RLE and zone maps work), and the merge amortizes the sorting cost across millions of rows. The costs are the LSM costs: a freshly-inserted row lives in a small, less-compressed part until merged, and a query must touch every part until compaction reduces their count. This is why column stores are happiest with **bulk, append-mostly ingestion** — load a million rows at once, not one row a thousand times — and why single-row updates and deletes are second-class citizens, often implemented as tombstone parts or expensive `ALTER TABLE ... UPDATE` mutations that rewrite whole parts.

```sql
-- ClickHouse: batch inserts become immutable parts; merges run in background.
-- Insert in large batches (>= ~10k-100k rows) -- one row at a time is an anti-pattern.
INSERT INTO events (event_date, region, user_id, amount, status)
SELECT * FROM input_batch;          -- writes ONE new part, sorted by ORDER BY key

-- Updates/deletes are mutations that rewrite affected parts -- use sparingly.
ALTER TABLE events DELETE WHERE event_date < '2025-01-01';   -- async, rewrites parts
OPTIMIZE TABLE events FINAL;        -- force-merge parts (manual compaction)
```

The same shape appears everywhere. Druid and Pinot ingest streaming events into in-memory segments and periodically hand off sealed, immutable, indexed segments to deep storage (S3/HDFS), from which historical nodes serve queries. Snowflake writes new immutable micro-partitions and never mutates an existing one — an `UPDATE` creates new micro-partitions and marks the old ones for removal, with time-travel relying on the old versions still existing. Parquet files are write-once by construction; you "update" a dataset by writing new files and compacting, which is precisely the job that table formats like Apache Iceberg, Delta Lake, and Apache Hudi layer on top of raw Parquet — they add a transaction log over a directory of immutable column files so you get atomic appends, deletes-via-tombstone, schema evolution, and time travel without ever mutating a Parquet file in place. The pattern is universal because it is forced: immutability is the price of keeping columns sorted and compressed, and append-plus-background-merge is how you pay it without giving up write throughput.

| Concern | OLTP row store | Column store (LSM-style) |
| --- | --- | --- |
| Single-row insert | Cheap, in place | Discouraged; buffer + flush as a part |
| Bulk load | Per-row overhead | Ideal — one sorted, compressed segment |
| Single-row update | Mutate one page | Rewrite a segment / tombstone + merge |
| Delete | Mark + reuse slot | Tombstone part, reclaimed on merge |
| Immutability | Pages mutated in place | Segments immutable; merged, never edited |
| Background work | Vacuum / checkpoint | Compaction / merge of segments |

> A column store is a read-optimized LSM tree with the bytes turned ninety degrees: same append-only-and-merge discipline, but laid out column-major so a scan reads one homogeneous, compressed run instead of whole rows.

## Why you should NOT run reports on the OLTP primary

Now we can be precise about the opening incident. It is tempting to think "the reporting query was just slow." It was worse than slow: it actively degraded the unrelated OLTP traffic sharing the box, through several independent mechanisms that all fire at once. The figure enumerates them.

![A long analytical query on the primary damages OLTP latency through four distinct contention paths](/imgs/blogs/oltp-vs-olap-and-columnar-stores-5.webp)

**1. Buffer-pool pollution.** The OLTP workload depends on its hot rows living in the shared buffer pool (the in-memory page cache; see [pages, heap files & buffer pool](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool)). Point reads are fast because the index pages and hot row pages are already in RAM. A sequential scan of 500M rows reads hundreds of gigabytes of pages that will be used *once* and pulls them through the buffer pool, evicting the carefully-warmed OLTP working set under LRU pressure. After the scan, your point reads that used to hit RAM now miss to disk. The OLTP latency degrades not while the scan runs but for minutes *after*, as the cache slowly re-warms. (Mature engines have partial defenses — Postgres uses a ring buffer for large sequential scans to limit pollution, and InnoDB has a midpoint-insertion LRU — but a wide analytical join with sorts and hash tables defeats these.)

**2. Pinned xmin horizon, blocked vacuum.** This is the subtle, dangerous one, and it is specific to MVCC engines like Postgres (see [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb)). Under MVCC, an `UPDATE` or `DELETE` does not remove the old row version; it leaves a dead tuple that VACUUM must reclaim later. But VACUUM can only remove a dead tuple if *no active transaction's snapshot can still see it*. A long-running analytical query holds a snapshot for its entire duration — your eight-minute report pins the **xmin horizon** (the oldest transaction id whose snapshot is still live) at the value it had when the query started. For those eight minutes, VACUUM across the *entire database* cannot reclaim any dead tuple newer than that horizon, even on tables your report never touched. The Postgres documentation is explicit: such transactions show up in `pg_stat_activity` with a large `age(backend_xmin)`, and the dead row versions "must not be deleted while still potentially visible." On a write-heavy OLTP database, eight minutes of blocked vacuum can mean millions of un-reclaimable dead tuples, table and index **bloat** that does not shrink when the query ends, and degraded scan and index performance that persists. Run that report every five minutes and you have effectively disabled vacuum, and in the extreme you flirt with transaction-id wraparound. The Figma and Clerk postmortems both trace production outages to query plans gone wrong on the primary; the xmin-horizon failure mode is the quieter cousin that bloats you to death without a single dramatic error.

**3. I/O and lock contention.** A sequential scan saturates the disk's read bandwidth. The OLTP point reads that need a few random IOPS now queue behind a scan that is asking for everything. Even with separate IOPS, a big analytical query takes shared locks, holds them longer, and competes for the same lock-manager partitions, connection slots, and work-memory budget. A hash join or large sort that exceeds `work_mem` **spills to temporary files on disk**, adding write I/O to the read I/O and further starving the latency-sensitive traffic.

**4. Plan-cache and optimizer thrash.** Analytical queries are often ad-hoc and varied, which fills the plan cache with one-off entries and can evict the stable, hot OLTP plans. Worse, the statistics that the planner relies on are tuned for OLTP cardinalities; a big `GROUP BY` can get a bad plan (a nested loop where it wanted a hash, a sequential scan where it wanted an index) that runs far longer than expected — exactly the class of failure behind the Figma incident, where an automatic `ANALYZE` shifted statistics mid-incident and PostgreSQL mis-planned an expensive query into table scans and temp-buffer writes.

| Damage mechanism | Who it hurts | When it bites | Lingers after query ends? |
| --- | --- | --- | --- |
| Buffer-pool pollution | All OLTP point reads | During + after the scan | Yes — until cache re-warms |
| Pinned xmin / blocked vacuum | Entire database (bloat) | The full query duration | Yes — bloat is permanent until vacuum |
| I/O + lock contention | Latency-sensitive traffic | During the scan | No (mostly) |
| Temp spill (sort/hash) | Disk write bandwidth | During the scan | No |
| Plan-cache / bad plan | Queries sharing the planner | Until plan re-stabilizes | Sometimes |

> A reporting query on your OLTP primary is not a read. It is a denial-of-service attack that you scheduled on a cron, against yourself, with read-only credentials.

A read replica solves the I/O and lock contention and the buffer-pool pollution (the replica has its own cache), and it is the right first step. But on Postgres a long query *on a replica* can still pin the horizon and either cause replication lag or get cancelled by `max_standby_streaming_delay` (the "canceling statement due to conflict with recovery" error), and `hot_standby_feedback` pushes the bloat problem back onto the primary. A replica is a real improvement, not a cure. The cure is to stop running OLAP on an OLTP engine at all.

## The modern split: replicate into a purpose-built analytics tier

The architecture the industry converged on is a clean separation: keep the OLTP primary lean and transactional, and continuously copy its data into a separate, column-oriented analytics tier where heavy scans live. The figure shows the two main paths.

![Change data capture and batch ELT replicate the OLTP primary into a dedicated analytical column store](/imgs/blogs/oltp-vs-olap-and-columnar-stores-6.webp)

**The CDC / streaming path (top).** Tail the primary's write-ahead log or binlog with a change-data-capture connector — Debezium reading a Postgres logical replication slot or MySQL binlog is the canonical setup — and stream the row changes through Kafka into a column store like ClickHouse or Druid. This gives near-real-time analytics: the warehouse is seconds-to-minutes behind production. Because CDC reads the WAL rather than running queries against the primary, it imposes almost no analytical load on the OLTP engine. (CDC has its own correctness subtleties around ordering, exactly-once, and schema changes; the [change data capture & the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) piece covers them.)

**The batch ELT path (bottom).** On a schedule, snapshot the primary (or its replica) and dump the data — typically as Parquet — into object storage, forming a data lake. From there a warehouse (Snowflake, BigQuery, Redshift) loads or queries it directly. This is higher latency (hourly or daily) but simpler and cheaper for large historical reprocessing.

Both paths converge on the same consumers: BI dashboards and ad-hoc SQL that scan and `GROUP BY` to their heart's content, on an engine built for exactly that, never touching the box that serves checkout.

A word on **ETL vs ELT**, because the letters matter. ETL (extract, transform, load) transforms data *before* loading it into the warehouse, on a separate transform tier. ELT (extract, load, transform) loads raw data into the warehouse first and transforms it *with the warehouse's own compute* — the pattern that won once warehouses got cheap, elastic, separated compute. With Snowflake or BigQuery you load the raw events and then run SQL (often orchestrated by dbt) to build cleaned and aggregated tables. ELT is preferred today because the warehouse's columnar, vectorized engine is faster at the transform than a bespoke pipeline, and keeping the raw data lets you re-derive everything when requirements change.

```sql
-- ELT in the warehouse: raw events land first, then SQL shapes them.
-- (Snowflake / BigQuery syntax; dbt would manage this as a model.)
CREATE OR REPLACE TABLE analytics.daily_revenue AS
SELECT
    DATE_TRUNC('day', created_at) AS day,
    region,
    SUM(amount)                   AS revenue,
    COUNT(*)                      AS orders
FROM raw.orders            -- loaded straight from CDC / Parquet, untransformed
GROUP BY 1, 2;
```

This is also where **star-schema denormalization** comes back. OLTP schemas are normalized — facts and lookups split across many tables — to avoid update anomalies and keep writes cheap. Analytics wants the opposite: a wide, denormalized **fact table** surrounded by small **dimension tables**, the classic star schema.

![A star schema centers one large fact table surrounded by small denormalized dimension tables](/imgs/blogs/oltp-vs-olap-and-columnar-stores-7.webp)

The center is `fact_sales` — one row per event, hundreds of millions of rows, holding the foreign keys to the dimensions plus the numeric **measures** (`quantity`, `net_amount`). Around it sit small **dimension tables**: `dim_date`, `dim_product`, `dim_store`, `dim_customer`, `dim_promotion`, each holding the descriptive attributes (`category`, `brand`, `city`, `region`, `segment`). The layout is called a star because the fact table is the hub and the dimensions are the points. Kleppmann describes exactly this in his data-warehousing section: the fact table is enormous and narrow-per-row but wide-in-columns, and analytics queries join it to a few small dimensions and aggregate. A snowflake schema normalizes the dimensions further (dimensions reference sub-dimensions), trading some join cost for less redundancy; the star is usually preferred for analytics because the joins are simpler and the dimensions are small enough to broadcast.

```sql
-- A canonical star-schema query: scan the big fact, join small dims, aggregate.
SELECT
    d.year,
    d.quarter,
    p.category,
    s.region,
    SUM(f.net_amount)  AS revenue,
    SUM(f.quantity)    AS units
FROM fact_sales        f                       -- 500M rows, columnar
JOIN dim_date          d ON f.date_id    = d.date_id      -- ~3650 rows
JOIN dim_product       p ON f.product_id = p.product_id   -- ~50k rows
JOIN dim_store         s ON f.store_id   = s.store_id     -- ~2k rows
WHERE d.year IN (2025, 2026)
GROUP BY 1, 2, 3, 4
ORDER BY revenue DESC;
```

In a column store this query reads only the columns it names from the fact table (`date_id`, `product_id`, `store_id`, `net_amount`, `quantity`), prunes row groups outside 2025-2026 with zone maps, broadcasts the tiny dimensions into hash tables, and aggregates vectorized. The same query against the normalized OLTP schema on the primary would be the outage from the opening paragraph.

**Materialized views and data cubes** close the loop for the queries you run constantly. Kleppmann describes the materialized view as a cached, precomputed result that the database keeps up to date, and the **data cube** as the aggressive special case: precompute the aggregate (say `SUM(amount)`) across every combination of a fixed set of dimensions, so a dashboard reads a cell instead of scanning the fact table. ClickHouse's `AggregatingMergeTree` materialized views, Druid's ingestion-time roll-ups, and BigQuery's materialized views are all this idea: spend write-time and storage to make the hot read-time query a lookup. The tradeoff is the usual one — a materialized aggregate is fast and inflexible; you can slice it only along the dimensions you precomputed.

## HTAP and its limits

If OLTP and OLAP are so different, can one system do both well? That is the promise of **HTAP** (hybrid transactional/analytical processing): a single database that serves transactions and analytics without an ETL pipeline. The honest answer is that HTAP systems make a deliberate engineering compromise, and the compromise is to run *two* engines under one roof rather than one engine that magically serves both shapes.

The successful designs keep a row store for the transactional workload and maintain a *separate column-store copy* of the same data for analytics, kept in sync internally. TiDB pairs its row-based TiKV with a columnar TiFlash replica; SAP HANA and SingleStore keep in-memory row and column representations; Oracle's In-Memory option maintains a columnar copy of hot tables alongside the on-disk row store; Snowflake's Unistore and the cloud vendors' "zero-ETL" offerings replicate the OLTP store into the warehouse continuously. In effect this is the same split as the previous section — two physical layouts for two workload shapes — just packaged so you do not run the pipeline yourself.

The limits follow from that structure. You pay for two copies of the data and the machinery that keeps them consistent. The analytical side is typically a near-real-time replica, not strictly transaction-consistent with the latest write, so HTAP analytics see data a beat behind. Resource isolation is still a real concern: a giant query on the columnar side can contend with the transactional side for CPU, memory, and bandwidth unless the system enforces hard separation, which the best ones do and the marketing-driven ones do not. HTAP is genuinely useful when you need analytics on *current* data with operational simplicity and you can afford the doubled storage. It does not repeal the fact that the two workloads want opposite layouts; it just hides the seam. When someone sells you "one database for everything with no tradeoffs," ask where the second copy of the data lives, because it is there.

| Approach | Latency to fresh data | Isolation from OLTP | Storage cost | Operational complexity |
| --- | --- | --- | --- | --- |
| Reports on primary | 0 (live) | None (DoS risk) | 1x | Low to write, ruinous to operate |
| Read replica | Seconds (lag) | Partial (cache yes, horizon no) | 2x | Low |
| CDC -> column store | Seconds to minutes | Full | 2x + pipeline | Medium |
| Batch ELT -> warehouse | Hours to a day | Full | 2x + lake | Medium |
| HTAP (dual engine) | Sub-second to seconds | Depends on the product | 2x (built in) | Low to operate, vendor lock-in |

## Where should analytics run? A playbook

A practical decision guide, in the order you should consider the options as your needs grow:

1. **A handful of small reports, data fits in RAM, freshness matters.** Run them on a **read replica**, not the primary. Cap statement timeout and `max_standby_streaming_delay`. This buys you most of the isolation for almost no engineering. Watch for replication conflicts on long queries.
2. **Ad-hoc analysis on files or a single big table, on one machine.** Reach for **DuckDB**. It queries Parquet, CSV, and Arrow in place, runs vectorized on all your cores, and needs no server. An analyst with a laptop and a few hundred million rows is its sweet spot. It is also the right embedded engine inside a larger application that needs local OLAP.
3. **Real-time dashboards over high-volume event streams (clicks, logs, metrics, ad tech).** Use a real-time column store: **ClickHouse**, **Druid**, or **Pinot**. Ingest from Kafka via CDC, cluster on your common filter columns, lean on materialized views / roll-ups for the hot aggregates. This is the Cloudflare pattern: trillions of rows, sub-second dashboards, ingest measured in millions of rows per second.
4. **Heterogeneous data from many sources, BI for the whole org, elastic scale, you do not want to run infrastructure.** Use a cloud **warehouse**: **Snowflake**, **BigQuery**, or **Redshift**. Load via ELT (Fivetran/Airbyte/dbt or native zero-ETL), let storage and compute scale independently, give each team its own virtual warehouse so they do not contend.
5. **You need analytics on current transactional data with minimal pipeline ops, and you can pay for it.** Consider **HTAP** (TiDB/TiFlash, SingleStore) or a managed zero-ETL replica — but verify the isolation guarantees and the freshness SLA before betting production on them.

When to *not* build any of this: if your "analytics" is one query an analyst runs by hand once a week against a million-row table, a replica (or even a carefully-timed query on the primary at 3 a.m. with a hard timeout) is fine, and a Snowflake account is overkill. Right-size to the workload. The machinery in this article earns its complexity at scale; below that scale it is yak-shaving.

## Case studies from production

### 1. The five-minute dashboard that disabled vacuum

A growth team built a revenue dashboard that ran `SELECT region, date, SUM(amount) ... GROUP BY` directly against the Postgres primary, refreshing every five minutes. For weeks it was invisible. Then the `orders` table crossed 300M rows and each refresh took six to nine minutes — longer than the refresh interval, so a new query started before the old one finished, and the database always had at least one long analytical snapshot open. The symptom that finally got attention was not the dashboard; it was that autovacuum had not completed on the busiest table in two days, table bloat had doubled the on-disk size, and point-read latency had crept up as index scans waded through dead tuples. The root cause was the pinned xmin horizon: the always-open report snapshot meant VACUUM could never advance past it. The fix was a two-line change to point the dashboard at a CDC-fed ClickHouse table, after which vacuum caught up overnight and the bloat slowly reclaimed. The lesson: a long-running read on an MVCC primary is a write-availability problem, because it blocks the maintenance that writes depend on.

### 2. Cloudflare's escape from Postgres + Citus

Cloudflare's HTTP analytics pipeline, built in 2014 for under a million requests per second, was a Postgres "RollupDB" plus a 12-node Citus cluster fed by 100+ Kafka consumers running tens of thousands of lines of aggregation code. By the time it served 6M requests/second it had become a liability: the Postgres node had no replica and was a single point of failure ("if we were to lose this node, the whole analytics pipeline could be paralyzed"), and the API could serve only ~15 queries/second. They migrated to ClickHouse on 36 nodes and the numbers tell the column-store story: ingestion of 11M rows/second at 47 Gbps, per-event storage from 1630 bytes down to 36.74 bytes, projected one-year storage from 273.93 PiB to 18.52 PiB, and annual cost from $28M to $1.9M ([Cloudflare](https://blog.cloudflare.com/http-analytics-for-6m-requests-per-second-using-clickhouse/)). Tuning `index_granularity` (16,384 for the big non-aggregated table, 32 for small aggregate tables) cut query latency in half and tripled throughput. The lesson: the right storage model is not a 20% optimization at this scale; it is the difference between a $28M line item and a $1.9M one, and between a SPOF and a replicated cluster.

### 3. Dremel and the trillion-row-in-seconds bet

Google's Dremel (the engine under BigQuery) was the public proof that columnar storage plus a multi-level serving tree could run aggregation over trillion-row tables in seconds. Its specific contribution was a columnar encoding for *nested* data — protobuf-shaped records with repeated and optional fields — using repetition and definition levels so that even semi-structured logs could be stored column-by-column and scanned without materializing whole records ([Dremel paper](https://research.google/pubs/dremel-interactive-analysis-of-web-scale-datasets-2/), and the [decade-later retrospective](https://www.vldb.org/pvldb/vol13/p3461-melnik.pdf)). The retrospective is candid that the early wins came from combining columnar layout, in-situ analysis on shared storage, and disaggregated compute — the same three ideas Snowflake later commercialized. The lesson: columnar is not only for flat relational tables; the encoding generalizes to nested data, which is why your JSON-ish event logs can still be a column store.

### 4. The replica that fell behind during the board meeting

A finance team moved its month-end reports off the primary to a Postgres read replica — the right instinct. But the heaviest report ran 40 minutes, and during it the replica's queries conflicted with WAL replay. With `max_standby_streaming_delay` set low, the report got cancelled mid-run with "canceling statement due to conflict with recovery"; raising the delay let the report finish but made the replica lag 40 minutes behind, so other dashboards reading the same replica showed stale data right when executives were looking. With `hot_standby_feedback = on` to avoid cancellations, the long replica query started pinning the *primary's* xmin horizon and bloating the primary. The team learned that a read replica is not an analytics engine: it is an OLTP engine with a copy of the data, and long scans fight replication the same way they fight vacuum. They moved the month-end reports to a nightly Parquet export queried by DuckDB and the conflicts vanished.

### 5. DuckDB replacing a Spark cluster for a single analyst

A data scientist was running daily feature aggregations over ~400M rows of Parquet on a managed Spark cluster that cost four figures a month and took 20 minutes per run including cluster spin-up. The job was a few `GROUP BY` and window functions — embarrassingly within one machine's reach. Rewriting it as a DuckDB script that read the Parquet files directly (column projection and row-group pruning meant it touched maybe 8 of 60 columns) ran in under two minutes on a single 32-core box, with zero cluster to manage and the result handed to Pandas via Arrow with no copy. The lesson, straight from DuckDB's design thesis: distributed compute is for data that genuinely exceeds one machine; an enormous fraction of "big data" analytics is single-node analytics wearing a cluster it does not need, and a vectorized in-process column engine wins on both speed and operational simplicity.

### 6. The wide-table scan that the zone map saved (and then didn't)

A team stored clickstream events in ClickHouse with `ORDER BY (event_date, user_id)` and were thrilled with date-range query performance — zone maps pruned everything outside the queried days. Then product asked for "all events for these 500 specific URLs," filtering on a `url` column that was *not* in the sort key and was scattered randomly across granules. Every granule's min/max for `url` spanned the whole alphabet, nothing got pruned, and the query did a full scan of two years of data. The fix was a ClickHouse data-skipping index (`INDEX url_idx url TYPE bloom_filter GRANULARITY 4`) and, for the heaviest URL queries, a second materialized view sorted by `url`. The lesson: zone maps and clustering accelerate exactly the predicates that align with your sort order and do nothing for the ones that do not. Choose the sort key for your real query mix, and keep alternately-sorted projections for the second-most-common access pattern.

### 7. The Parquet file with one giant row group

An ingestion job wrote Parquet files by buffering each day's data and flushing once — producing files with a single 4 GB row group. Queries that filtered to a few hours of the day still had to read the entire row group, because row-group min/max is the finest skipping granularity and a one-row-group file has no granularity to skip. Worse, readers could not parallelize across row groups (there was only one), so a 16-core machine used one core. Re-tuning the writer to target ~128 MB row groups restored both row-group pruning and intra-file parallelism, and query times on selective filters dropped roughly 10x. The lesson: the column-store skip mechanisms are real but they live at the block boundary; a file with too-coarse blocks (one huge row group) or too-fine blocks (millions of tiny files, the "small files problem") both defeat them. Aim for a row-group/segment size that balances skip granularity against per-block overhead.

### 8. The "just add a read replica" that doubled the bill and half-fixed it

After an incident where analytics on the primary caused a checkout slowdown, a team added a read replica and routed all reporting to it. Contention on the primary disappeared and everyone declared victory. Two quarters later the replica itself was the bottleneck: it ran the same single-threaded-per-query Postgres engine, so concurrent dashboards queued behind each other, and the queries were still scanning whole tables row-major with no compression. They were paying for a second full-size database to get an OLTP engine doing OLAP work badly. Moving the actual analytical workload to BigQuery (ELT via scheduled Parquet exports) let the dashboards run concurrently on elastic compute, and the replica went back to being a small failover standby. The lesson: a replica fixes *isolation* but not *workload fit*. You still have a row store scanning whole rows. Isolation and the right storage model are two separate problems; solve both.

### 9. Druid roll-ups that made a billion rows into ten million

An ad-tech team ingested ~1B impression events per day into Druid for dashboards that only ever queried at minute granularity grouped by a few dimensions. Storing every raw event was wasteful for that query pattern. They enabled ingestion-time roll-up, pre-aggregating events into one row per (minute, campaign, creative, country) tuple with summed metrics. The 1B daily raw events collapsed to ~10M rolled-up rows, a 100x reduction, and dashboard queries that previously scanned the raw firehose now read pre-aggregated rows and returned in tens of milliseconds. The cost was loss of per-event detail — they kept a short raw retention in a separate table for the rare drill-down. The lesson, which is just Kleppmann's data-cube argument in production: if you know the granularity your queries need, pre-aggregate to it at ingest; storing raw events you never query at raw granularity is paying to scan data you will always immediately collapse.

### 10. Snowflake micro-partitions and the accidental full-table clustering churn

A team on Snowflake set `CLUSTER BY (event_timestamp, user_id, session_id, url)` on a high-ingest table, reasoning that more clustering keys meant more pruning. Their bill spiked: automatic clustering was constantly re-sorting micro-partitions because the high-cardinality `session_id` and `url` made nearly every new partition overlap existing ones, triggering endless reclustering credits. Most of their actual queries filtered on `event_timestamp` and `user_id` only. Dropping the cluster key to `(event_timestamp, user_id)` cut reclustering cost dramatically while keeping the pruning that mattered, because micro-partition min/max on those two columns was already tight. The lesson: clustering (like a sort key, like an index) is not free and not "more is better." It costs write-time work to maintain, and clustering on high-cardinality columns your queries do not filter on burns money to prune nothing. Cluster on the columns in your `WHERE` clauses, lowest-churn first.

### 11. The HTAP system that wasn't isolated

A team adopted an HTAP database expecting "transactions and analytics, one system, no tradeoffs." For a while it delivered. Then an analyst ran an unbounded join across the two largest tables on the columnar replica, and because the product shared a CPU and memory pool between the row and column engines, the transactional latency spiked and a few writes timed out — the exact contention HTAP was supposed to prevent. The vendor's answer was a resource-group configuration that hard-partitioned compute between the two engines, which fixed the isolation but meant they were now manually sizing two pools, i.e., operating two engines after all. The lesson: HTAP hides the OLTP/OLAP seam but does not delete it. The two workloads still compete for physical resources; a good HTAP product gives you hard isolation knobs, and you must actually set them. "No tradeoffs" is marketing; "managed tradeoffs" is the real value.

### 12. The nightly job that read the whole table to update one day

An ELT job rebuilt a `daily_revenue` aggregate by scanning the entire `fact_sales` table every night, even though only the last day's data had changed. At 50M rows it was fine; at 2B rows the nightly full scan took hours and the warehouse compute cost dominated the bill. The fix was incremental: partition the fact table by day, and have the job recompute only the new and late-arriving partitions (an incremental dbt model with a `WHERE created_at >= ...` predicate that the warehouse's partition pruning turned into reading one partition). Runtime went from hours to minutes and cost fell proportionally. The lesson: column stores make scans cheap, but "cheap per byte" times "all the bytes, every night" is still a large bill. Partition on time, prune to the changed partitions, and process incrementally. The cheapest scan is the one you scope to the data that actually changed.

### 13. The one-row-at-a-time insert that buried a MergeTree in parts

A service streamed events into ClickHouse with a synchronous single-row `INSERT` per event, treating it like the Postgres table it had replaced. Each insert created a new tiny `MergeTree` part. Within hours the table had tens of thousands of parts, the background merge scheduler could not keep up, queries slowed because each one had to merge results across thousands of parts, and eventually inserts started failing with "too many parts." The team had imported an OLTP write pattern into an OLAP engine whose entire write model is "buffer and flush in batches." The fix was to batch inserts — buffer events for a second or up to 50k rows in the application (or front them with ClickHouse's async insert / a Kafka engine table) so each `INSERT` became one substantial, well-compressed part. Part count dropped to the dozens, merges caught up, and "too many parts" disappeared. The lesson: the column store's append-and-merge write path wants large batches; row-at-a-time writes are the single most common way teams misuse it, because the API looks identical to the OLTP one they know.

### 14. The dictionary that fell back to plain and tripled the file

An ingestion pipeline wrote a `url` column to Parquet expecting dictionary encoding to shrink it the way it shrank the `region` column. But `url` had millions of distinct values per row group; Parquet's dictionary grew past its size threshold and **fell back to plain encoding** mid-chunk, so the column stored full strings and the files were roughly 3x larger than the team's capacity model predicted. The same data, when they extracted just the `domain` portion into a separate low-cardinality column (and kept the full `url` only where drill-down needed it), dictionary-encoded beautifully and most queries hit the small `domain` column. The lesson: encodings are data-dependent, not magic. Dictionary and RLE reward *low cardinality and clustering*; a high-cardinality column gets no dictionary benefit and can silently fall back to plain. Model your columns' cardinality, split or transform the genuinely high-cardinality ones, and verify the encoding the writer actually chose (`parquet-tools meta`, `system.parts_columns` in ClickHouse) rather than assuming the ratio you hoped for.

## When to reach for a column store, and when not to

Reach for column-oriented analytics infrastructure when:

- Your queries **scan many rows but project few columns** — aggregations, `GROUP BY`, dashboards, funnels, cohort analysis over wide fact tables.
- Your analytical workload is **degrading OLTP latency** or, more insidiously, **blocking vacuum / bloating** your primary. The xmin-horizon symptom is the strongest signal you have crossed the line.
- Data **volume exceeds what fits comfortably in your OLTP buffer pool**, so analytical queries are guaranteed to pollute the cache and hit disk.
- You need **real-time analytics over event streams** (logs, clicks, metrics) at high ingest rates — the ClickHouse/Druid/Pinot zone.
- You want **org-wide BI with elastic, isolated compute** and minimal infrastructure ops — the Snowflake/BigQuery zone.
- You are doing **single-machine analysis over files** (Parquet/CSV) and want speed without a cluster — DuckDB.

Skip the column store (or defer it) when:

- Your workload is **genuinely transactional**: point reads and writes, whole-row access, sub-millisecond latency. A column store is the wrong engine; it will make your point reads slow and your single-row updates painful. Stay on the row store.
- Your "analytics" is **a few small queries on small data** an analyst runs occasionally — a read replica (or a carefully time-boxed query) is enough; a warehouse is overkill.
- You need **strict transactional consistency with the latest write** for the analytical read. Replicated/CDC tiers are a beat behind; if you cannot tolerate that, you are back to the primary or a tightly-isolated HTAP system.
- You would be operating a **complex pipeline for a problem a read replica solves**. Don't build a CDC-to-Kafka-to-ClickHouse pipeline for a dashboard that one replica serves fine. Match the machinery to the scale.
- Your data has **high update/delete churn on individual rows** that must be reflected immediately. Column stores prefer append and bulk load; heavy in-place mutation fights their grain (and is exactly what the row store on your primary is for).

The deeper principle survives every specific tool: OLTP and OLAP are different physical problems wearing the same SQL syntax. The row store answers "which rows" by keeping a row's fields together and finding rows by key; the column store answers "which columns" by keeping a column's values together, compressing them because they are homogeneous, skipping blocks the predicate excludes, and crunching them in vectorized SIMD over only the columns you named. You do not need to choose a side — you need to run each workload on the engine built for its shape, and to keep the analytical one off the box that takes your customers' money. The companion pieces on [pages, heap files & buffer pool](/blog/software-development/database/how-databases-store-data-pages-heap-files-buffer-pool), [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), [change data capture & the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) each cover one organ of this body; this one is about not asking a single organ to do two jobs.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 3 — "Transaction Processing or Analytics?", data warehousing, star/snowflake schemas, column-oriented storage, column compression, sort order, and materialized views / data cubes. The canonical treatment of everything above.
- [Dremel: Interactive Analysis of Web-Scale Datasets](https://research.google/pubs/dremel-interactive-analysis-of-web-scale-datasets-2/) and the [decade retrospective](https://www.vldb.org/pvldb/vol13/p3461-melnik.pdf) — columnar storage for nested data, the foundation of BigQuery.
- [The Snowflake Elastic Data Warehouse](https://www.cs.cmu.edu/~15721-f24/papers/Snowflake.pdf) (SIGMOD 2016) — separation of storage and compute, immutable micro-partitions, per-partition pruning.
- [MonetDB/X100: Hyper-Pipelining Query Execution](https://www.cidrdb.org/cidr2005/papers/P19.pdf) — the origin of vectorized execution; vectors sized to the CPU cache.
- [HTTP Analytics for 6M requests/second using ClickHouse](https://blog.cloudflare.com/http-analytics-for-6m-requests-per-second-using-clickhouse/) — Cloudflare's migration off Postgres+Citus, with the compression and cost numbers cited throughout.
- [Apache Parquet file format](https://parquet.apache.org/docs/file-format/) and [encodings](https://parquet.apache.org/docs/file-format/data-pages/encodings/) — row groups, column chunks, pages, RLE/dictionary/delta encoding.
- [Why DuckDB](https://duckdb.org/why_duckdb) — the in-process, vectorized, single-node OLAP design thesis.
- [PostgreSQL routine vacuuming](https://www.postgresql.org/docs/current/routine-vacuuming.html) — the xmin horizon, why long transactions block dead-tuple removal, and wraparound risk.

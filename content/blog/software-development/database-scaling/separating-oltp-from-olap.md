---
title: "Separating OLTP from OLAP: Keep Analytics Off Your Production Database"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Running analytics on your transactional database is the fastest way to take down production; here is how to separate the two workloads with change data capture and a columnar store so each engine does what it was built for."
tags: ["oltp", "olap", "change-data-capture", "data-warehouse", "columnar-storage", "clickhouse", "dbt", "kafka", "database-scaling", "analytics"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 30
---

Almost every production outage I have been paged for that traced back to "the database is slow" had nothing to do with the database being undersized. It was someone running a report. A growth analyst, a finance team member, or a well-meaning backend engineer opens a SQL console pointed at the production primary, writes `SELECT date_trunc('day', created_at), count(*), sum(amount) FROM orders GROUP BY 1 ORDER BY 1`, and hits run. The query is correct. The business question is reasonable. And for the next ninety seconds, every real user's checkout p99 climbs from 8 ms to 1.4 seconds, the connection pool saturates, and the on-call engineer gets paged for a latency SLO breach with no deploy, no traffic spike, and no obvious cause.

This is not a freak event. It is the single most common way I have seen teams take down their own production database, and it is entirely structural: transactional and analytical workloads have *opposite physical shapes*, and forcing them to share one engine means the heavier one wins every fight. The fix is not "make the query faster" or "add an index." The fix is architectural — separate the two workloads onto engines built for each, and stream the data from one to the other.

![Separating OLTP from OLAP with change data capture: one transactional source of record fans out, through CDC, into analytical stores that never touch production.](/imgs/blogs/separating-oltp-from-olap-1.webp)

The diagram above is the mental model for this entire post. There is exactly one **system of record** — the OLTP primary your application writes to. Everything analytical hangs off a **change data capture** (CDC) stream that tails that database's write-ahead log and fans the changes out to purpose-built analytical stores: a warehouse or lakehouse for historical batch analytics, and a real-time columnar engine like ClickHouse for fresh dashboards. Analysts, BI tools, and machine-learning feature pipelines query those derived stores. They never touch the primary. The rest of this article is a tour of why each arrow in that picture exists and what breaks when you skip it.

## Why this is different from "just tune the query"

The instinct of most engineers, when the reporting query takes down production, is to optimize the query. That instinct is wrong, and understanding *why* is the whole game.

| Common assumption | The naive view | The reality |
| --- | --- | --- |
| "It's a slow query; add an index." | An index makes any query fast. | A `GROUP BY` over all history is a *full scan by design*. No index helps; the planner correctly chooses a sequential scan. |
| "The replica will absorb analytics." | Read replicas isolate read load. | Replicas scale OLTP *point reads*. A heavy scan on a replica still thrashes a row-store buffer cache and can stall replication. |
| "We'll run reports at night." | Off-peak means no impact. | Long analytical transactions hold back the `xmin` horizon and block vacuum, bloating the primary for everyone. |
| "One database is simpler." | Fewer moving parts is better. | One engine optimized for 8 ms point writes is *structurally* bad at 50-GB scans. You are paying tail-latency tax to avoid an ETL pipeline. |
| "Our data is small." | Small data fits in one box. | Transactional tables grow forever; the first 100-GB `GROUP BY` arrives without warning and the buffer pool is suddenly too small to protect. |

The thread running through every row is the same: OLTP and OLAP are not two settings of the same dial. They are two different machines. Let us look at why.

## 1. The conflict: two workloads with opposite shapes

> A transactional engine is a sports car tuned for a million tight corners a second; an analytical engine is a freight train built to haul one enormous load in a straight line. Asking either to do the other's job is how you end up in a ditch.

**Online transaction processing (OLTP)** is the workload your application runs: many small, low-latency operations. Insert one order. Read one user by primary key. Update one inventory row. The defining property is **selectivity** — each query touches a tiny fraction of the data, found through an index, and returns in single-digit milliseconds. Postgres, MySQL, and most operational databases store data **row-oriented**: all the columns of one row sit physically together on a page, because a transaction almost always wants the whole row at once. The schema is normalized to avoid update anomalies, and B-tree indexes on selective columns let you jump straight to the rows you need without scanning.

**Online analytical processing (OLAP)** is the opposite in every dimension. An analytical query touches *few rows of logic but enormous volumes of data*: "revenue by region by month for the last three years." There is no selective predicate — you want all of history. You read two or three columns out of fifty, aggregate across hundreds of millions of rows, and you do not care if it takes four seconds because a human is reading a chart. The bottleneck is not random-access latency; it is raw **scan bandwidth** and CPU throughput on the aggregation.

Now put them in the same engine and watch what happens when the analytical query runs.

<figure class="blog-anim">
<svg viewBox="0 0 760 330" role="img" aria-label="A heavy analytics scan sweeps across the OLTP buffer pool, evicting hot pages one column at a time while the p99 latency bar for real users grows from 2 milliseconds toward 800 milliseconds" style="width:100%;height:auto;max-width:820px">
<style>
.a2-pool{fill:#86efac;stroke:var(--text-primary,#1f2937);stroke-width:1.5}
.a2-ev{fill:#ef4444;stroke:var(--text-primary,#1f2937);stroke-width:1.5}
.a2-sweep{fill:var(--accent,#6366f1);opacity:.22}
.a2-bar{fill:#ef4444;transform-box:fill-box;transform-origin:bottom}
.a2-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.a2-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.a2-s{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a2-r{font:600 12px ui-sans-serif,system-ui;fill:#ef4444}
@keyframes a2-c0{0%{opacity:0}10%{opacity:1}90%{opacity:1}100%{opacity:0}}
@keyframes a2-c1{0%,16%{opacity:0}28%{opacity:1}90%{opacity:1}100%{opacity:0}}
@keyframes a2-c2{0%,34%{opacity:0}46%{opacity:1}90%{opacity:1}100%{opacity:0}}
@keyframes a2-c3{0%,52%{opacity:0}64%{opacity:1}90%{opacity:1}100%{opacity:0}}
@keyframes a2-mv{0%{transform:translateX(0);opacity:.22}78%{transform:translateX(360px);opacity:.22}90%{transform:translateX(360px);opacity:0}100%{transform:translateX(0);opacity:0}}
@keyframes a2-grow{0%{transform:scaleY(.07)}16%{transform:scaleY(.22)}34%{transform:scaleY(.45)}52%{transform:scaleY(.72)}70%,90%{transform:scaleY(1)}100%{transform:scaleY(.07)}}
.a2-ev0{animation:a2-c0 9s ease-in-out infinite}
.a2-ev1{animation:a2-c1 9s ease-in-out infinite}
.a2-ev2{animation:a2-c2 9s ease-in-out infinite}
.a2-ev3{animation:a2-c3 9s ease-in-out infinite}
.a2-mv{animation:a2-mv 9s ease-in-out infinite}
.a2-grow{animation:a2-grow 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a2-ev0,.a2-ev1,.a2-ev2,.a2-ev3{animation:none;opacity:1}.a2-mv{animation:none;opacity:0}.a2-grow{animation:none;transform:scaleY(1)}}
</style>
<text class="a2-t" x="40" y="34">OLTP buffer pool (hot working set)</text>
<rect class="a2-pool" x="40"  y="58" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="156" y="58" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="272" y="58" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="388" y="58" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="40"  y="142" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="156" y="142" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="272" y="142" width="100" height="72" rx="6"/>
<rect class="a2-pool" x="388" y="142" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev0" x="40"  y="58" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev0" x="40"  y="142" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev1" x="156" y="58" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev1" x="156" y="142" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev2" x="272" y="58" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev2" x="272" y="142" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev3" x="388" y="58" width="100" height="72" rx="6"/>
<rect class="a2-ev a2-ev3" x="388" y="142" width="100" height="72" rx="6"/>
<rect class="a2-sweep a2-mv" x="36" y="54" width="30" height="164" rx="6"/>
<text class="a2-s" x="40" y="248">analytics scan sweeps left to right, evicting the working set</text>
<text class="a2-s" x="40" y="270">hot OLTP pages (green) replaced by cold analytics blocks (red)</text>
<line class="a2-axis" x1="600" y1="58" x2="600" y2="238"/>
<rect class="a2-bar a2-grow" x="608" y="58" width="64" height="180" rx="4"/>
<text class="a2-t" x="640" y="34" text-anchor="middle">p99 latency</text>
<text class="a2-r" x="680" y="66">800 ms</text>
<text class="a2-s" x="680" y="238">2 ms</text>
</svg>
<figcaption>One heavy analytical scan walks the table, evicting the hot OLTP working set from the shared buffer pool; as cached pages turn cold, p99 latency for real user requests climbs from ~2 ms toward ~800 ms.</figcaption>
</figure>

Three distinct things go wrong at once, and they compound.

**The buffer pool gets evicted.** Your OLTP database keeps a hot working set in memory — the recently-touched index and table pages that serve 99% of traffic with zero disk I/O. A sequential scan over a 200-GB table reads pages your real queries do not need and, under a naive LRU policy, pushes the hot set out to make room. (Postgres mitigates this specific case with a ring buffer for large sequential scans, but shared buffers are still finite, and a sufficiently large or repeated scan still cools the cache.) The moment the hot pages are gone, every point read that used to hit memory now hits disk, and your p99 detonates — exactly the latency growth the animation above shows.

**I/O and CPU saturate.** The scan competes for the same disk bandwidth, the same page cache, and the same CPU cores as transactional traffic. On a cloud instance with provisioned IOPS, the scan burns your entire I/O budget and the transactional workload queues behind it.

**Vacuum stalls.** This is the subtle, slow-motion killer. Postgres uses MVCC: every row version carries transaction-ID visibility bounds, and `VACUUM` reclaims dead versions once no transaction can still see them. A long-running analytical query holds an old snapshot, which holds back the `xmin` horizon — the oldest transaction ID that anything in the system still needs. Vacuum cannot clean up any row version newer than that horizon, so dead tuples accumulate, tables bloat, and the same point queries get slower for reasons no `EXPLAIN` will show you. A single analytics query left open in a transaction for an hour can bloat a write-heavy table by gigabytes. This is the same mechanism covered in depth in [keeping Postgres healthy under write load](/blog/software-development/database-scaling/keeping-postgres-healthy-under-write-load) — analytics-on-primary is one of the most reliable ways to trigger it.

Here is the kind of query that does it, and why no index saves you:

```sql
-- The "innocent" reporting query, run against the production primary.
EXPLAIN (ANALYZE, BUFFERS)
SELECT date_trunc('month', o.created_at) AS month,
       o.region,
       count(*)        AS orders,
       sum(o.amount)   AS revenue,
       avg(o.amount)   AS aov
FROM   orders o
WHERE  o.created_at >= now() - interval '3 years'
GROUP  BY 1, 2
ORDER  BY 1, 2;

-- Plan (abbreviated):
--   Finalize GroupAggregate
--     -> Gather Merge  (workers planned: 4)
--        -> Partial GroupAggregate
--           -> Sort
--              -> Parallel Seq Scan on orders  (cost=0..3_812_445)
--                 rows=148_200_000  Buffers: shared read=2_140_551
-- There is NO index that turns this into a point lookup. The query, by
-- definition, must visit ~150M rows. The planner is right to seq-scan.
-- "shared read=2_140_551" pages = ~16 GB pulled through the buffer pool.
```

The planner is not being dumb. The query genuinely needs to look at every order in three years. An index on `created_at` would only help if the date range were narrow and selective; `GROUP BY` over all history is the antithesis of selective. **You cannot index your way out of an OLAP query on an OLTP engine.** The only real fix is to stop running it there.

## 2. Why "just add a read replica" only half-works

The first thing every team reaches for is a read replica. It is the right instinct for the wrong problem. A streaming replica replays the primary's WAL and serves read-only queries, which genuinely scales out OLTP *point reads*: route your `SELECT ... WHERE id = ?` traffic to replicas and the primary breathes easier. For light reporting — a daily summary, a handful of dashboards that scan a few million rows — a dedicated replica is a perfectly good answer, and you should use it.

But a read replica does not change the *physics* of the workload. It is the same row-store engine with the same row-oriented pages and the same buffer cache. Point the 150-million-row `GROUP BY` at the replica and you have simply moved the buffer-pool eviction and I/O saturation to a different box. That is fine if the replica is dedicated to analytics and nothing else depends on it. It is not fine for two reasons that bite in production.

![A read replica scales OLTP point reads, not OLAP scans: a replica multiplies point-read capacity, yet a heavy scan still thrashes the row-store buffer cache and stalls recovery.](/imgs/blogs/separating-oltp-from-olap-4.webp)

**Replication lag and recovery conflicts.** A standby is constantly replaying WAL. When a long analytics query is running on the standby and the primary vacuums away a row version the query still needs, you get a *recovery conflict*: Postgres must either cancel your query (`ERROR: canceling statement due to conflict with recovery`) or pause WAL replay and let the standby fall behind. The usual workaround, `hot_standby_feedback = on`, tells the primary to *not* vacuum rows the standby still needs — which means your long analytics query is now holding back vacuum *on the primary*, reintroducing the exact bloat problem you were trying to escape.

```ini
# postgresql.conf on the analytics standby — the tradeoff in two settings.
hot_standby_feedback = on        # avoid query cancellations on the standby ...
                                 # ... but now the standby's xmin pins vacuum
                                 # on the PRIMARY, so a long scan bloats prod.
max_standby_streaming_delay = 30s  # how long replay may pause for a query;
                                   # exceed it and the query is cancelled,
                                   # OR the standby's lag grows unbounded.
```

There is no setting that makes this conflict disappear, because it is fundamental: the standby is a faithful row-store copy that must stay consistent with the primary, and a long scan is at odds with both staying current *and* not affecting the primary. **Read replicas scale OLTP reads; they do not give you OLAP.** They are a row-store engine wearing a "read-only" hat.

The deeper issue is that the replica is still *row-oriented*. Even with infinite isolation, scanning 50 columns to aggregate 3 of them wastes ~94% of every byte you read off disk and through the cache. To actually make analytical queries fast — not just isolated, but *fast* — you need a fundamentally different storage layout.

## 3. Columnar storage: why a column store is ~100x faster

**The senior rule of thumb: row stores optimize for "give me this whole row"; column stores optimize for "give me this one column across all rows." Aggregations are the second question, so OLAP wants columns.**

Consider the order table with 50 columns and the query `SELECT region, sum(amount) GROUP BY region`. A row store lays each order's 50 columns out contiguously, so to read the `amount` and `region` columns it must stride across every row, pulling all 50 columns through cache and discarding 48 of them. A column store stores each column as its own contiguous run on disk, so the same query reads exactly two columns and touches nothing else.

![Row store vs column store on a GROUP BY scan: a wide aggregation reads every column of every row in a row store, but only the referenced columns in a column store.](/imgs/blogs/separating-oltp-from-olap-3.webp)

That layout difference alone is roughly a 10-20x reduction in bytes read for a typical wide table. But columnar storage compounds two more wins on top of it:

**Compression.** A single column holds values of one type with low cardinality and often high locality — all the `region` values are a handful of distinct strings, all the `status` values are an enum, timestamps are monotonic. Run-length encoding, dictionary encoding, and delta encoding routinely shrink a column 5-20x. Less data on disk means less I/O, and modern column stores decompress in vectorized blocks so the CPU cost is small. A row store cannot compress nearly as well because adjacent bytes belong to different types and have no shared structure.

**Vectorized execution.** Because a column is a dense array of one type in memory, the aggregation kernel can process it in SIMD batches — sum 8 doubles per instruction, branch-free — instead of the row-at-a-time interpreter loop a typical OLTP engine runs. This is another 5-10x on the CPU side of the aggregation.

Multiply the byte-reduction, the compression, and the vectorization, and a wide aggregation that takes 90 seconds on a row store runs in under a second on a column store on the same hardware. That is the ~100x in the figure, and it is the entire reason warehouses exist. You can feel the layout difference even in a few lines of NumPy, which is just a column store with one column:

```python
import numpy as np, time

N = 50_000_000
# Row-major: a struct-of-50-fields per row (an "AoS" / array-of-structs).
# We only need column 7 ("amount"), but it is interleaved with 49 others.
rows = np.zeros((N, 50), dtype=np.float64)      # ~20 GB row-oriented
rows[:, 7] = np.random.rand(N)

# Column-major: each column is its own contiguous array ("SoA").
amount_col = np.ascontiguousarray(rows[:, 7])   # the only column we read

t = time.perf_counter()
_ = rows[:, 7].sum()          # strided read across the wide row layout
row_t = time.perf_counter() - t

t = time.perf_counter()
_ = amount_col.sum()          # dense, cache-friendly, SIMD-vectorized
col_t = time.perf_counter() - t

print(f"row-store stride sum: {row_t*1000:6.1f} ms")
print(f"column-store sum:     {col_t*1000:6.1f} ms")
print(f"speedup: {row_t/col_t:4.1f}x")
# On a laptop this prints roughly:
#   row-store stride sum:  410.2 ms
#   column-store sum:       28.7 ms
#   speedup: 14.3x
# ... and a real column store adds compression + multi-core on top.
```

The contrast extends across every dimension of the two workloads. There is no single engine that sits at the optimum for both columns of this table — which is precisely why the industry runs two.

![OLTP vs OLAP want the opposite thing at every layer: read across any row and the two workloads disagree, which is why no single engine is optimal for both.](/imgs/blogs/separating-oltp-from-olap-5.webp)

| Dimension | OLTP | OLAP |
| --- | --- | --- |
| Storage layout | Row-oriented pages | Columnar segments |
| Query shape | Point read/write, 1 to few rows | Full scan + `GROUP BY` over all rows |
| Indexing | B-tree on selective keys | Zone maps / min-max / sort keys |
| Schema | Normalized (3NF) | Denormalized star/wide tables |
| Hardware bottleneck | Random IOPS, latency | Sequential scan bandwidth, CPU |
| Concurrency | 10k+ tiny transactions/sec | A few long-running queries |
| Freshness need | Immediate, read-your-writes | Seconds to hours is usually fine |
| Example engines | Postgres, MySQL, DynamoDB | Snowflake, BigQuery, ClickHouse |

If you want the deeper treatment of row-vs-column internals, compression schemes, and vectorized execution, I wrote a dedicated piece on [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores). For this post, the takeaway is structural: the analytical store must be a *different engine* with a *different layout*, fed from the transactional one. How you feed it is the next question.

## 4. The architecture: stream changes with CDC, don't query the source

You have two ways to get transactional data into an analytical store: pull it on a schedule, or push it as it changes.

**Batch ETL** runs a periodic job — every hour, every night — that queries the source for "everything that changed since last time" and loads it into the warehouse. It is simple and it is still the right answer for plenty of use cases. But it has two failure modes. First, the extract query itself is load on the source — `SELECT * FROM orders WHERE updated_at > :watermark` is a scan, and on a large table it is exactly the kind of query we are trying to keep off the primary. Second, watermark-based extraction silently misses hard deletes and any update that does not bump `updated_at`, so the warehouse slowly drifts out of sync with reality.

**Change data capture (CDC)** flips the direction. Instead of asking the database "what changed?", you read the database's own write-ahead log — the ordered, durable record of every insert, update, and delete it already writes for crash recovery and replication. A tool like **Debezium** acts as a logical replication client, decodes the WAL, and emits one message per row change onto Kafka. This is nearly free on the source (logical decoding is cheap and runs off the WAL the database already produces), it captures deletes, and it preserves exact order. Every downstream consumer becomes a replayable materialized view of one ordered stream of changes — which is exactly the mental model in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log).

A minimal Debezium-on-Kafka-Connect setup for a Postgres source looks like this:

```json
{
  "name": "orders-cdc-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "orders-primary.internal",
    "database.dbname": "orders",
    "plugin.name": "pgoutput",
    "slot.name": "debezium_orders",
    "publication.autocreate.mode": "filtered",
    "table.include.list": "public.orders,public.order_items",
    "topic.prefix": "cdc.orders",
    "snapshot.mode": "initial",
    "heartbeat.interval.ms": "10000",
    "decimal.handling.mode": "string"
  }
}
```

Two production gotchas hide in that config. `slot.name` creates a replication slot on the primary, and **a replication slot that no consumer is draining will pin the WAL forever** — if Debezium dies and you do not notice, the primary's `pg_wal` directory fills the disk and the database stops accepting writes. Monitor `pg_replication_slots` lag like you monitor disk. And `heartbeat.interval.ms` matters precisely because of the slot: on a low-traffic table the slot's confirmed LSN would otherwise never advance, so the heartbeat forces periodic progress.

The "right" way to emit business events through CDC — and to avoid the dual-write problem where you write to the database and to Kafka separately and they diverge on failure — is the transactional outbox pattern, where you write the event to an `outbox` table in the same transaction as your state change and let CDC ship it. That whole design space is covered in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern); for our purposes, CDC is the pipe that moves committed changes off the primary without ever running a query against it.

The stream lands in Kafka; a sink connector or a stream processor writes it into the analytical store. A bare-bones consumer that upserts into a warehouse staging table shows the shape:

```python
from confluent_kafka import Consumer
import json, warehouse  # your warehouse client

c = Consumer({"bootstrap.servers": "kafka:9092",
              "group.id": "orders-warehouse-sink",
              "enable.auto.commit": False,
              "auto.offset.reset": "earliest"})
c.subscribe(["cdc.orders.public.orders"])

batch = []
while True:
    msg = c.poll(1.0)
    if msg is None:
        continue
    change = json.loads(msg.value())          # Debezium envelope
    after, op = change["after"], change["op"]  # op: c=create u=update d=delete
    batch.append({"op": op, "row": after, "lsn": change["source"]["lsn"]})
    if len(batch) >= 5_000:
        # Idempotent MERGE keyed on (id) so replay is safe; the LSN gives
        # us a total order to resolve out-of-order updates to the same row.
        warehouse.merge_into("staging.orders", batch, key="id", order_by="lsn")
        c.commit(asynchronous=False)           # commit offsets AFTER the write
        batch.clear()
```

The critical detail is **idempotency**: you commit Kafka offsets only after the warehouse write succeeds, and the warehouse write is a `MERGE` keyed on the primary key. If the consumer crashes mid-batch, it reprocesses from the last committed offset and the `MERGE` makes the replay a no-op. This is what lets the analytical store converge on the truth no matter how many times messages are redelivered — at-least-once delivery plus idempotent writes equals effectively-exactly-once state.

## 5. The modern data stack: ingestion, storage, transform, query, consumption

Once the changes are flowing, the analytical side organizes into five loosely-coupled layers, each independently scalable and swappable. This decomposition is what people mean by "the modern data stack," and its whole point is that no layer needs to know how the others are implemented.

![The modern analytics stack as independent layers: bronze-silver-gold modeling lives in the transform layer, decoupled from the ingestion path below and the BI tools above.](/imgs/blogs/separating-oltp-from-olap-6.webp)

**Ingestion** is the CDC/Kafka pipe we just built, plus managed connectors (Fivetran, Airbyte) for SaaS sources you do not own. **Storage** is where the data lands: a cloud warehouse's managed storage (Snowflake, BigQuery, Redshift), or — increasingly — an open table format (Apache Iceberg, Delta Lake, Hudi) sitting on object storage like S3, which decouples storage from any single query engine and lets Spark, Trino, and Snowflake all read the same files. **Transform** is where raw change events become clean, modeled tables. **Query** is the engine that answers analytical questions (Trino, Snowflake, BigQuery, DuckDB for smaller scales). **Consumption** is BI tools, notebooks, and reverse-ETL jobs that push derived data back into operational systems.

The transform layer is where most of the engineering discipline lives, and the dominant pattern is **medallion architecture**: bronze (raw, append-only landing of the CDC stream), silver (cleaned, deduplicated, type-cast, one row per entity), gold (business-level aggregates and marts the BI tools actually query). **dbt** has become the standard way to express these as version-controlled SQL models with tests and lineage:

```sql
-- models/silver/stg_orders.sql
-- Silver: one clean, current row per order from the raw CDC stream.
with source as (
    select * from {{ source('bronze', 'orders_cdc') }}
),
deduped as (
    select *,
           row_number() over (
               partition by id order by _lsn desc   -- latest change wins
           ) as rn
    from source
    where _op != 'd'                                 -- drop tombstones
)
select
    id::bigint              as order_id,
    user_id::bigint         as user_id,
    region::text            as region,
    amount::numeric(12,2)   as amount,
    created_at::timestamptz as created_at
from deduped
where rn = 1
```

```sql
-- models/gold/monthly_revenue.sql
-- Gold: the mart the BI dashboard reads. This is the query that used to
-- take down production; here it runs on a columnar engine, off the primary.
select
    date_trunc('month', created_at) as month,
    region,
    count(*)      as orders,
    sum(amount)   as revenue,
    avg(amount)   as aov
from {{ ref('stg_orders') }}
group by 1, 2
```

Notice that the `monthly_revenue` model is *byte-for-byte the analytical query that caused the outage in the intro* — but now it runs on a columnar warehouse, against a deduplicated silver table, with zero impact on the database serving real users. That is the entire payoff of the architecture in one diff: the same business question, asked of a different engine.

One more arrow worth naming: **reverse-ETL**. Sometimes a derived value computed in the warehouse — a customer's lifetime value, a churn-risk score, a recommended-products list — needs to be read back by the application at OLTP latencies. Reverse-ETL tools (Census, Hightouch, or a homegrown job) push gold-layer tables back into the operational database or a key-value store, closing the loop without the application ever querying the warehouse synchronously.

## 6. Real-time, user-facing analytics: ClickHouse, Pinot, Druid

Batch and even streaming-into-a-warehouse architectures usually land data with seconds-to-minutes of latency and answer queries in seconds. That is fine for an internal dashboard a human refreshes a few times a day. It is not fine when the *analytics themselves are a product feature*: a restaurant owner watching live order volume, a creator watching impressions tick up, an ad buyer watching spend in real time. These **user-facing analytics** demand two things at once that warehouses do not give you: sub-second query latency at high concurrency, *and* data that is fresh within seconds of the event happening.

That is the niche of the real-time OLAP engines — **ClickHouse**, **Apache Pinot**, and **Apache Druid**. They are columnar (so aggregations are fast), they ingest streaming data continuously (so it is fresh), and they are built for high query concurrency with tight latency SLOs (so thousands of users can hit a dashboard at once).

![Real-time user-facing analytics path: when analytics must be both fresh and fast, events stream into a columnar real-time store that answers dashboards in milliseconds.](/imgs/blogs/separating-oltp-from-olap-7.webp)

The flow is the CDC/event pipeline pointed at a real-time store instead of (or in addition to) the warehouse: events land in Kafka, a streaming ingestion path appends them to the columnar engine within a second or two, and a thin query API serves dashboard requests with a p99 under 100 ms. A ClickHouse table for this looks deliberately denormalized and sort-key-optimized:

```sql
CREATE TABLE events_rt
(
    event_time   DateTime,
    merchant_id  UInt64,
    region       LowCardinality(String),   -- dictionary-encoded automatically
    event_type   LowCardinality(String),
    amount       Decimal(12, 2)
)
ENGINE = MergeTree
ORDER BY (merchant_id, event_time)          -- the sort key IS the index
PARTITION BY toYYYYMM(event_time)
TTL event_time + INTERVAL 90 DAY;           -- drop old partitions cheaply

-- A merchant's live dashboard query: scans one merchant's recent partition,
-- aggregates on a dense columnar layout, returns in single-digit ms.
SELECT toStartOfMinute(event_time) AS minute,
       countIf(event_type = 'order')  AS orders,
       sumIf(amount, event_type = 'order') AS revenue
FROM   events_rt
WHERE  merchant_id = 4815162342
  AND  event_time >= now() - INTERVAL 1 HOUR
GROUP  BY minute
ORDER  BY minute;
```

There is no replication slot pinning anything, no buffer pool to evict, no vacuum to stall — because this engine never sees a transaction. It only sees an append-only stream of immutable events, which is exactly what a columnar engine is happiest with. The cost is that it is a *second* system to operate and that it is not your source of truth; if it loses data you rebuild it by replaying Kafka. That tradeoff — a derived, rebuildable store fed by a log — is the same one running through every box in the original diagram.

## 7. The freshness-versus-isolation tradeoff (and a word on HTAP)

Step back and every option we have discussed sits somewhere on a two-axis tradeoff: how *fresh* is the data (lag from write to queryable), and how *isolated* is the analytical load from production (can a heavy query hurt real users)?

![Freshness vs isolation: pick the right analytics path. Coupling analytics to the OLTP store buys freshness at the cost of isolation; streaming to a columnar store buys both.](/imgs/blogs/separating-oltp-from-olap-8.webp)

Querying the primary directly is maximally fresh and minimally isolated — it is the intro outage. A read replica is fresh and *somewhat* isolated, until a heavy scan reintroduces contention and recovery conflicts. Nightly batch ETL is maximally isolated but stale by up to a day. The sweet spot — fresh *and* isolated — is streaming CDC into a separate columnar store, whether a warehouse for historical analytics or a real-time engine for user-facing dashboards. You get changes within seconds and the analytical load never touches the engine serving your users.

You will hear that **HTAP** (hybrid transactional/analytical processing) promises to dissolve this tradeoff: one engine that serves both workloads, no ETL, always fresh. Systems like TiDB (with its TiFlash columnar replica), SingleStore, and CockroachDB's analytical features genuinely deliver a version of this. But read the architecture and you will find the isolation is usually achieved by maintaining a *separate columnar replica inside the same product* — which is the same separation we have been describing, just packaged as one vendor's system rather than two. HTAP can be a real simplification when it fits, especially for mid-scale workloads that do not want to operate Kafka and a warehouse. It is not magic: the moment your analytical queries get large enough to matter, you are still paying for a columnar copy and still reasoning about resource isolation between the two. The tradeoff does not vanish; it moves inside the box.

## Case studies from production

### 1. The reporting query that paged the on-call team

A mid-size commerce company let analysts query the production Postgres primary directly because "the data is freshest there." For two years it was fine. Then the `orders` table crossed ~120 GB, and a new analyst wrote a year-over-year revenue query with a `GROUP BY` over all history. The query ran for 70 seconds. During those 70 seconds the buffer pool's hot working set was evicted by the scan, checkout p99 went from 9 ms to 1.6 seconds, the connection pool maxed out, and the latency-SLO alert paged the on-call backend engineer. The first hypothesis was a bad deploy (there had been none) and the second was a traffic spike (there was none). It took twenty minutes and a look at `pg_stat_activity` to find the analyst's query. The fix was immediate (kill the query, the cache rewarmed in minutes) and structural (a CDC pipeline into a warehouse, and revoking the analyst role's access to the primary). The lesson is that *direct human access to the production primary is a latent incident*, not a convenience.

### 2. Cloudflare: replacing PostgreSQL/Citus with ClickHouse for HTTP analytics

Cloudflare's customer-facing HTTP analytics — the dashboards showing requests, bandwidth, and threats — originally aggregated data in a Postgres/Citus pipeline that could not keep up as traffic grew into the millions of requests per second. They re-architected the analytics path onto ClickHouse, ingesting an enormous event stream and serving customer dashboard queries from a columnar store with materialized aggregations. The move took a system that was straining at the seams to one comfortably handling trillions of rows, because the workload — append-only events, aggregated by time and zone, read by many customers concurrently — is exactly what a columnar real-time engine is built for, and exactly what a row-store was not. The architectural lesson is that *user-facing analytics is a different product than your transactional database* and deserves its own engine.

### 3. Uber: real-time analytics with Apache Pinot

Uber runs a large number of internal and external real-time analytics use cases — the restaurant manager dashboard in UberEats, surge and demand monitoring, fraud signals — on Apache Pinot fed by Kafka. The defining requirement is freshness combined with concurrency: a restaurant owner expects their order dashboard to reflect orders from seconds ago, and tens of thousands of merchants hit those dashboards at once. None of that traffic touches the transactional databases that record the orders; it is served from Pinot tables continuously ingesting the same event streams via CDC and Kafka. The takeaway is that at scale, freshness and isolation are not in tension *if you decouple the analytical store* — you get both by streaming, not by querying the source.

### 4. The vacuum that quietly bloated the primary

A team did the responsible thing and built a dedicated analytics read replica, then ran their heavy queries there, confident the primary was protected. Over several weeks the primary's tables bloated, disk usage climbed faster than data growth explained, and the same point queries got slower. The cause was `hot_standby_feedback = on`, which they had enabled to stop their long analytics queries from being cancelled by recovery conflicts. That setting made the standby's oldest snapshot hold back the *primary's* `xmin` horizon, so vacuum on the primary could not reclaim dead tuples while a long analytics query ran on the standby. The replica had not isolated the workload at all — it had just relocated the symptom and hidden the cause. The real fix was a columnar store fed by CDC, which holds no snapshot against the primary. This is the failure mode that convinces teams a replica is not an analytics strategy.

### 5. The dual-write that drifted, then CDC that converged

A payments company emitted analytics events by writing to the database and then publishing to Kafka in application code — a classic dual write. Under normal operation the two agreed. Under partial failure (the DB commit succeeded but the Kafka publish timed out, or vice versa) they diverged, and over months the warehouse's view of transactions drifted measurably from the ledger, which is catastrophic for a payments business. They replaced the dual write with Debezium CDC reading the WAL: now the warehouse is, by construction, a replay of exactly the committed database changes, with no second write to fail independently. Reconciliation discrepancies dropped to zero. The lesson — covered in depth in the [outbox pattern post](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — is that *the database's own log is the only source of truth that cannot drift from the database*.

### 6. The nightly batch that could not keep up

A marketplace ran nightly batch ETL into a warehouse for years, and it worked until the business started needing intraday numbers — operations wanting to see today's metrics by noon, not tomorrow morning. Each attempt to run the batch more frequently (hourly, then every fifteen minutes) put more scan load on the source and still left a frustrating lag. The resolution was to move from "pull on a schedule" to "push as it changes": CDC into Kafka, stream processing into the warehouse's bronze layer, dbt transforms on a tight cadence. Data freshness went from up-to-24-hours to under a minute, *and* source load went down because logical decoding off the WAL is cheaper than repeated extract scans. The counterintuitive lesson: streaming CDC is often both fresher and gentler on the source than frequent batch.

### 7. The HTAP engine that still needed isolation

A team adopted an HTAP database specifically to avoid running a separate analytics stack, and for a while the single-system simplicity was a real win. As analytical queries grew, they discovered that getting acceptable isolation required pinning analytics to the columnar replica nodes and the OLTP traffic to the row-store nodes, configuring resource groups, and reasoning carefully about which queries hit which copy. In other words, they had rebuilt the OLTP/OLAP separation *inside* the HTAP product. The system was still a good choice — one vendor, one operational surface, automatic replication between the row and column copies — but it taught them that HTAP relocates the separation rather than eliminating it, and that the columnar copy and resource isolation are not optional extras but the load-bearing parts.

## When to separate OLTP from OLAP — and when not to

**Reach for a separate analytical store (CDC → columnar) when:**

- Analysts or BI tools run queries that scan large fractions of a table, and those queries share an engine with user-facing traffic.
- You have seen — or can predict — buffer-pool eviction, I/O saturation, or vacuum stalls caused by reporting load.
- Analytics is a product feature with its own latency and freshness SLOs (user-facing dashboards), not just an internal report.
- Your transactional tables are large enough (tens of GB and growing) that the first full-history aggregation will be expensive.
- You need to join data across multiple services or databases, which the warehouse can do and the operational stores cannot.

**Skip it, or stay simpler, when:**

- Your whole dataset is small (a few GB) and your "analytics" is a handful of light queries — a dedicated read replica is plenty, and a warehouse is operational overhead you do not need yet.
- You are pre-product-market-fit and the cost of a Kafka + warehouse + dbt stack outweighs the risk of the occasional slow report on a low-traffic database.
- Your reporting is genuinely off-hours, low-frequency, and bounded, and you have measured that it does not evict the working set or pin vacuum — though be honest that "we'll only run it at night" rarely survives contact with a growing team.
- An HTAP engine fits your scale and you would rather operate one system than four — just go in knowing you are buying a columnar replica and resource isolation, not escaping them.

The mistake is almost never "separated too early." It is running the year-over-year revenue query against the database that takes your customers' money, discovering the buffer pool the hard way, and explaining to the on-call engineer why checkout was down for a report nobody urgently needed. Build the pipe before you need it; the WAL is already there waiting to be tailed.

## Further reading

- [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) — the deep dive on row-vs-column internals, compression, and vectorized execution.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — how to ship changes reliably without the dual-write problem.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — why treating Kafka as a replayable log lets every derived store be a materialized view.
- [Keeping Postgres healthy under write load](/blog/software-development/database-scaling/keeping-postgres-healthy-under-write-load) — vacuum, the `xmin` horizon, and the bloat that long analytics transactions cause.

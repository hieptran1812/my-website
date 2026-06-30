---
title: "Time-Series Data at Scale: Partitioning, Downsampling, and Retention"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Metrics, events, and IoT data are a write-mostly, append-heavy workload that quietly destroys a general-purpose database but is the natural home turf of a time-series engine — here is how partitioning, downsampling, retention tiers, and compression make it scale."
tags: ["time-series", "database-scaling", "timescaledb", "prometheus", "clickhouse", "downsampling", "data-retention", "compression", "cardinality", "observability"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 33
---

The first time I watched a Postgres box fall over from "just metrics," the root cause was almost insulting in its simplicity. A team had wired their application's request counters straight into a regular `events` table — one row per request, a `created_at` timestamp, a B-tree index on it for "fast time-range queries," and a nightly cron that ran `DELETE FROM events WHERE created_at < now() - interval '30 days'`. It worked beautifully in staging. In production it ingested forty thousand rows a second, the index grew faster than the heap, autovacuum fell permanently behind the delete-driven bloat, and within three weeks a dashboard query that scanned "only the last hour" was taking nine seconds and pinning a CPU. The database was not slow. It was being asked to do a job it was architecturally wrong for.

That mismatch is the entire subject of this post. Metrics, events, IoT telemetry, and structured logs all share a workload shape — huge append-only write rates, strictly time-ordered, recent data hot and old data cold, queries that are almost always a time range plus an aggregation — and that shape is exactly what a general-purpose row store is built to *not* be good at. A time-series engine inverts every default: it partitions by time so old data is dropped in O(1), it pre-aggregates into rollups so a year-long dashboard never touches raw points, it tiers storage so old data costs cents, and it compresses ten-to-twelve-fold by exploiting the fact that consecutive timestamps and values barely change. None of these are exotic tricks. They are the obvious moves *once you accept that time is the primary key*.

![The life of a time-series point: born hot at the write head, aging through downsampled tiers, finally dropped — all keyed by time](/imgs/blogs/time-series-data-at-scale-1.webp)

Imagine the diagram above as the mental model for everything that follows. A point is *born* at the write head — append-only, no update, no random insert — lands in the hot tier at full one-second resolution on SSD, ages rightward into warm tiers where it has been rolled up to one-minute and then one-hour summaries, and is eventually dropped wholesale when its chunk exceeds the retention window. Queries cut across the bottom: a range query over `[t0, t1]` asking for an average, a p99, or a rate. The four techniques this article is about — time-partitioning, downsampling, retention tiers, and compression — are just the four arrows in that picture. The rest is detail.

## 1. Why time-series is a different animal

**The senior rule of thumb: if your access pattern is "append at the head, read recent ranges, aggregate, and expire the tail," you do not have a database problem — you have a time-series problem, and they are not the same.**

A transactional store is tuned for a workload of small random reads and writes against current state: fetch this user, update that balance, delete this order. Rows are mutable, the working set is the *whole* table, and the index exists to find any row by key in logarithmic time. A time-series workload violates almost every one of those assumptions. Writes are overwhelmingly appends near "now." Existing rows are essentially never updated — a temperature reading from last Tuesday at 14:32 is immutable history. Reads are dominated by ranges (`last 6 hours`, `this quarter`) and almost never by primary key. And the answer the reader wants is usually not the raw points at all but an aggregate over them: the average CPU, the 99th-percentile latency, the per-minute request rate.

| Dimension | General-purpose assumption | Time-series reality |
| --- | --- | --- |
| Write pattern | Random inserts/updates anywhere | Append-only at the head, ordered by time |
| Mutability | Rows updated and deleted in place | Points immutable; UPDATE is vanishingly rare |
| Working set | Entire table is "warm" | Last few hours hot; everything older is cold |
| Query shape | Point lookups, joins on keys | Time range + aggregation (avg, sum, rate, p99) |
| Deletion | Occasional, row-at-a-time | Bulk expiry of the *oldest* data, continuously |
| Cardinality driver | Row count | Number of distinct *series* (metric × labels) |
| Dominant cost | Random I/O, lock contention | Ingest throughput, index size, scan volume |

The consequences cascade. Because writes are append-only, you never need an index to *find a place to put* a new point — the place is always "the end." Because old points are immutable and cold, you can compress and even evict them on a schedule without coordinating with writers. Because queries are range-plus-aggregate, you can answer most of them from precomputed summaries instead of raw data. And because the data is fundamentally a function of time, you can use *time itself* as the partition key, which turns the single most expensive operation in the general-purpose world — bulk deletion of old data — into a metadata operation.

There is one more property that does not fit in the table because it is the silent killer: **cardinality**. In a relational table, scale is "how many rows." In a time-series system, scale is "how many distinct series," where a series is the unique combination of a metric name and all its label values. A single metric with a few well-chosen labels is a few hundred series; the same metric with one badly-chosen label can be a billion. We will spend a whole section on it because it is the failure mode that takes down more time-series deployments than every other cause combined.

## 2. Why a general-purpose database buckles

**The senior rule of thumb: a row store can *store* time-series data fine — it just can't *expire* it, can't *aggregate* it, and can't *compress* it, and those are the three things the workload demands constantly.**

Let us be precise about the three places a vanilla relational database breaks under sustained metric load, because "just use Postgres" is a real and sometimes correct answer, and you need to know exactly where its ceiling is.

![Why a row store buckles under metrics: constant inserts, DELETE-based retention, and full-window scans are exactly the costs a time-series engine designs away](/imgs/blogs/time-series-data-at-scale-2.webp)

**Index bloat from constant inserts.** A B-tree index on `created_at` has to find the right leaf and possibly split a page on *every* insert. With strictly increasing timestamps the inserts all land on the rightmost leaf, which sounds cheap but creates a single hot page that every writer contends on, and the index keeps growing in lockstep with the heap. We cover the mechanics of why this hurts in [B-trees and how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work); the short version is that an index that must be maintained synchronously on every one of forty thousand appends per second is pure overhead for a workload that never needs to *look up a point by key*. The append-optimized engines instead use an [LSM-tree](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) or a time-ordered column file, where appends are sequential writes with no in-place page splits.

**Retention by DELETE is a catastrophe.** This is the one that surprises people. In Postgres, `DELETE` does not free space — it marks tuples dead and leaves them for `VACUUM` to reclaim, and a continuous stream of deletes against the oldest data generates a continuous stream of bloat that autovacuum must chase forever. (The full pathology is in [Postgres VACUUM, bloat, and autovacuum tuning](/blog/software-development/database/postgres-vacuum-bloat-and-autovacuum-tuning).) Worse, the deleted rows are scattered across the same pages as live rows if your physical ordering does not match time, so reclaiming them means rewriting pages full of data you want to keep. A time-series engine sidesteps this entirely: old data lives in its own physical partition (chunk), and "retention" means `DROP TABLE` on that chunk — an O(1) metadata operation that frees the storage instantly with zero vacuum pressure.

**Full scans to aggregate a window.** Ask a row store for `avg(value) WHERE time > now() - interval '7 days'` and, even with a perfect index on `time`, it must read every row in that range to compute the average. Seven days at one-second resolution is roughly 600,000 rows *per series*; across a thousand series that is 600 million rows scanned to draw one line on a dashboard that refreshes every fifteen seconds. The row store has no concept of "this average was already computed." A time-series engine answers the same query from a rollup that stored one pre-aggregated value per minute — a 60× reduction before you even reach for the columnar scan.

**Storage is fat and uncompressed.** A naive row is a timestamp (8 bytes), a value (8 bytes), plus per-row tuple overhead (in Postgres, 23+ bytes of header *per row* before your columns). Time-series points, by contrast, compress to roughly one to two bytes each because consecutive timestamps and values are nearly identical — but only if the engine stores them column-wise and applies delta encoding, which a row store does not. We will return to this in the compression section; for now, note that the same dataset can differ by 20–50× in on-disk size depending on whether the engine understands that it is time-series.

The honest caveat: none of this means "Postgres can't do time-series." It means *unmodified* Postgres can't. The most popular time-series engine in the world, TimescaleDB, is literally a Postgres extension that adds exactly the four capabilities above — automatic time-partitioning, continuous aggregates, native columnar compression, and chunk-drop retention — while keeping full SQL and the Postgres ecosystem. The lesson is not "abandon your relational database." It is "the workload needs partitioning, rollups, tiering, and compression, and you must get them from *somewhere*."

## 3. The cardinality bomb

**The senior rule of thumb: in a time-series system you are not paying for data points, you are paying for distinct series — and the number of distinct series is a product, not a sum.**

This is the single most important concept in operating time-series infrastructure, and it is the one most teams learn the expensive way. A *series* is the unique combination of a metric name and the full set of label (tag) key-value pairs attached to it. `http_requests_total{region="us-east", status="200", method="GET"}` is one series. Change any label value and it is a different series, with its own index entry, its own in-memory chunk, its own compression state. The total number of active series is the cartesian product of the cardinalities of every label.

![The cardinality bomb: series count is the product of label cardinalities, and one unbounded label detonates the whole store](/imgs/blogs/time-series-data-at-scale-3.webp)

The arithmetic is unforgiving. Take `http_requests_total` with three sensible labels: `region` (5 values), `status` (8 values), `method` (4 values). That is 5 × 8 × 4 = 160 series — trivial, a rounding error in memory. Now a well-meaning engineer adds a `user_id` label "so we can break down requests by user." If you have ten million users, you have just multiplied 160 by ten million to get 1.6 *billion* active series. Every one needs an index entry and a live chunk in RAM. Prometheus will OOM. TimescaleDB's index will balloon. InfluxDB's TSM engine will spend all its time managing series files. The metric did not get more *voluminous* — the raw write rate is unchanged — it got more *cardinal*, and cardinality is what these systems are made of.

The labels that detonate cardinality are always the same shapes, and you can recognize them on sight:

- **Unbounded identifiers**: `user_id`, `request_id`, `trace_id`, `session_id`, `order_id`. These have effectively infinite distinct values. They belong in a logs/traces system or an OLAP store, never as a metric label.
- **High-cardinality free text**: full URL paths with IDs embedded (`/users/8f3a/orders/91c2`), raw error messages, email addresses. Normalize the path to a route template (`/users/:id/orders/:id`) *before* it becomes a label.
- **Timestamps or sequence numbers as labels**: someone always tries this; it makes every scrape a brand-new series.

Bounding cardinality is a discipline, not a feature you turn on. The practical defenses, in order of leverage:

```promql
# Find your worst offenders before they find you. Top 10 metrics by series count:
topk(10, count by (__name__)({__name__=~".+"}))

# How many series does one metric explode into?
count(http_requests_total)

# Which label is the culprit? Count distinct values per label on a metric:
count(count by (user_id) (http_requests_total))
```

Run these as standing dashboards and alerts. The first time `count(some_metric)` crosses, say, 100,000, you want a page, not a postmortem. Beyond monitoring, the structural fixes are: **drop the offending label at scrape time** (Prometheus `metric_relabel_configs` with `action: labeldrop`), **enforce a per-tenant series limit** (Mimir, Cortex, and VictoriaMetrics all support hard caps that reject new series past a threshold), and **move genuinely high-cardinality breakdowns to a different tool** — an exemplar, a trace, or a wide-event store like ClickHouse where each row is cheap and there is no per-series memory cost. The mental shift is that a metric label is a *dimension you will always want to group by*, not an *attribute you might occasionally filter on*. If you only need it occasionally and at high cardinality, it is not a label.

## 4. Technique one: time-partitioning

**The senior rule of thumb: partition by time and your two worst operations — expiring old data and scanning a window — both collapse from O(rows) to O(chunks).**

Time-partitioning is the foundational trick, the one that makes the other three possible. Instead of one giant table, the engine maintains many smaller physical partitions, each holding a contiguous slice of time — a day, an hour, six hours, whatever the *chunk interval* is set to. TimescaleDB calls these chunks and the logical table over them a *hypertable*; ClickHouse calls them parts within a partition; InfluxDB calls them shards. The names differ; the payoff is identical.

![Chunk pruning along the time axis: a query for the last two days touches only the chunks that overlap it, and the oldest chunk is dropped whole](/imgs/blogs/time-series-data-at-scale-4.webp)

Two things become cheap. First, **constraint exclusion / chunk pruning**: when you query `WHERE time > now() - interval '2 days'`, the planner inspects each chunk's time range and reads *only* the two or three chunks that overlap the window, skipping the other hundreds entirely without touching their data or indexes. The query cost is proportional to the data in range, not the data in the table. Second, **retention as a drop**: expiring data older than thirty days means finding the chunks whose time range is entirely older than the cutoff and issuing the equivalent of `DROP TABLE` on each. No row-by-row delete, no vacuum, no bloat — the storage is reclaimed atomically.

Here is the whole setup in TimescaleDB, which is the clearest way to see partitioning, rollups, compression, and retention as one coherent policy stack:

```sql
-- A hypertable is an ordinary Postgres table that auto-partitions into chunks by time.
CREATE TABLE metrics (
    time       TIMESTAMPTZ      NOT NULL,
    series_id  BIGINT           NOT NULL,
    value      DOUBLE PRECISION NOT NULL
);

-- One chunk per day. Pick the interval so a chunk's *indexes* fit comfortably in RAM
-- (a common rule: aim for the most recent chunk's data + indexes ≈ 25% of memory).
SELECT create_hypertable('metrics', 'time',
                         chunk_time_interval => INTERVAL '1 day');

-- The natural time-series index: per series, newest first.
CREATE INDEX ON metrics (series_id, time DESC);
```

The choice of `chunk_time_interval` is the one tuning knob that matters most. Too large and a chunk's index no longer fits in memory, so inserts start hitting disk and you have reinvented the problem partitioning was meant to solve. Too small and you drown in thousands of tiny chunks, each with planning and metadata overhead, and pruning has to consider a huge chunk list. The standard guidance is to size the interval so that the *most recent* chunk — the one taking all the writes — has its data plus indexes sitting at roughly a quarter of available RAM. For most deployments that lands somewhere between a few hours and a few days per chunk.

### Second-order optimization: align chunk interval to your query and retention windows

A subtle win: make the chunk interval a clean divisor of both your dominant query window and your retention boundary. If you almost always query "last 24 hours" and retain "90 days," a one-day chunk means a typical query touches exactly one or two chunks and retention drops exactly one chunk per day on a clean schedule. If you instead picked a 7-day chunk, your 24-hour query still has to scan a full week's chunk (you lose most of the pruning benefit), and retention can only drop data in 7-day granular steps, so you keep up to six extra days of data you meant to expire. Partitioning only pays off when the partition boundaries line up with how you actually read and expire — a chunk interval chosen in a vacuum is a chunk interval chosen wrong.

## 5. Technique two: downsampling and rollups

**The senior rule of thumb: store raw data at the resolution you *capture*, but query it at the resolution you can *see* — nobody can read 2.6 million points on a 1,200-pixel chart.**

Downsampling is the move that decouples how finely you record from how coarsely you usually read. You collect at one-second resolution because when something breaks at 3 a.m. you want the fine grain. But a dashboard showing the last quarter has, at most, a couple thousand horizontal pixels — feeding it ninety days of one-second points is asking it to compute an average over 7.7 million values to render a line that can only show a few thousand distinct positions. The fix is to precompute aggregates at coarser intervals — *rollups* — and serve each query from the coarsest rollup that still has enough resolution for the question.

<figure class="blog-anim">
<svg viewBox="0 0 760 380" role="img" aria-label="Twelve raw one-second points are aggregated into three one-minute rollup bars and then into a single one-hour bar, shrinking the point count at each tier" style="width:100%;height:auto;max-width:820px">
<style>
.ts5-raw{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ts5-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:bottom}
.ts5-sweep{fill:var(--accent,#6366f1);opacity:.16}
.ts5-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ts5-sub{font:500 13px ui-monospace,monospace;fill:var(--text-secondary,#6b7280)}
.ts5-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
@keyframes ts5-sweep{0%{transform:translateX(0);opacity:0}6%{opacity:.16}70%{transform:translateX(540px);opacity:.16}80%,100%{transform:translateX(540px);opacity:0}}
@keyframes ts5-b1{0%,16%{opacity:0;transform:scaleY(0)}26%,94%{opacity:1;transform:scaleY(1)}100%{opacity:0;transform:scaleY(0)}}
@keyframes ts5-b2{0%,34%{opacity:0;transform:scaleY(0)}44%,94%{opacity:1;transform:scaleY(1)}100%{opacity:0;transform:scaleY(0)}}
@keyframes ts5-b3{0%,52%{opacity:0;transform:scaleY(0)}62%,94%{opacity:1;transform:scaleY(1)}100%{opacity:0;transform:scaleY(0)}}
@keyframes ts5-h{0%,72%{opacity:0;transform:scaleY(0)}82%,94%{opacity:1;transform:scaleY(1)}100%{opacity:0;transform:scaleY(0)}}
.ts5-mv{animation:ts5-sweep 11s linear infinite}
.ts5-g1{animation:ts5-b1 11s ease-out infinite}
.ts5-g2{animation:ts5-b2 11s ease-out infinite}
.ts5-g3{animation:ts5-b3 11s ease-out infinite}
.ts5-gh{animation:ts5-h 11s ease-out infinite}
@media (prefers-reduced-motion:reduce){.ts5-mv{animation:none;opacity:0}.ts5-g1,.ts5-g2,.ts5-g3,.ts5-gh{animation:none;opacity:1;transform:scaleY(1)}}
</style>
<text class="ts5-lbl" x="12" y="72">raw · 1s</text>
<text class="ts5-sub" x="12" y="92">3600 pts/hr</text>
<text class="ts5-lbl" x="12" y="218">1-min roll</text>
<text class="ts5-sub" x="12" y="238">60 pts/hr</text>
<text class="ts5-lbl" x="12" y="326">1-hour roll</text>
<text class="ts5-sub" x="12" y="346">1 pt/hr</text>
<line class="ts5-axis" x1="120" y1="120" x2="690" y2="120"/>
<rect class="ts5-raw" x="120" y="80"  width="34" height="40"/>
<rect class="ts5-raw" x="168" y="65"  width="34" height="55"/>
<rect class="ts5-raw" x="216" y="72"  width="34" height="48"/>
<rect class="ts5-raw" x="264" y="58"  width="34" height="62"/>
<rect class="ts5-raw" x="312" y="70"  width="34" height="50"/>
<rect class="ts5-raw" x="360" y="62"  width="34" height="58"/>
<rect class="ts5-raw" x="408" y="76"  width="34" height="44"/>
<rect class="ts5-raw" x="456" y="60"  width="34" height="60"/>
<rect class="ts5-raw" x="504" y="68"  width="34" height="52"/>
<rect class="ts5-raw" x="552" y="74"  width="34" height="46"/>
<rect class="ts5-raw" x="600" y="64"  width="34" height="56"/>
<rect class="ts5-raw" x="648" y="70"  width="34" height="50"/>
<rect class="ts5-sweep ts5-mv" x="116" y="52" width="46" height="74"/>
<rect class="ts5-bar ts5-g1" x="124" y="180" width="168" height="70"/>
<rect class="ts5-bar ts5-g2" x="316" y="180" width="168" height="70"/>
<rect class="ts5-bar ts5-g3" x="508" y="180" width="168" height="70"/>
<rect class="ts5-bar ts5-gh" x="124" y="294" width="552" height="56"/>
</svg>
<figcaption>Downsampling collapses 3600 raw one-second points into 60 one-minute rollups, then into a single one-hour rollup — each retention tier stores far fewer points at coarser resolution.</figcaption>
</figure>

The animation shows the cascade: many raw one-second points fold into a single one-minute summary, then many of those fold into one hour. Each level is roughly a 60× reduction in point count. The critical detail that distinguishes a *good* rollup system from a naive one is that rollups are **maintained incrementally**, not recomputed from scratch. TimescaleDB's continuous aggregates store the rollup as a materialized view over the hypertable and update it as new data arrives, touching only the buckets that changed:

```sql
-- A 1-minute rollup, incrementally maintained. avg/max/count cover most dashboards.
CREATE MATERIALIZED VIEW metrics_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    series_id,
    avg(value) AS avg_value,
    max(value) AS max_value,
    min(value) AS min_value,
    count(*)   AS n
FROM metrics
GROUP BY bucket, series_id;

-- Refresh the rollup as data lands, but leave the most recent minute alone (it's
-- still filling). start_offset bounds how far back each refresh looks.
SELECT add_continuous_aggregate_policy('metrics_1m',
    start_offset      => INTERVAL '3 hours',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');
```

Now the read side. A 30-day dashboard query reads the one-minute rollup, which holds about 43,000 rows per series instead of 2.6 million, and can roll *that* up to hourly on the fly cheaply:

```sql
-- Query the resolution you need. This reads metrics_1m (43k rows/series for 30 days),
-- not the raw table (2.6M rows/series), then buckets to hourly for the chart.
SELECT
    time_bucket('1 hour', bucket) AS hour,
    avg(avg_value)                AS avg_value,
    max(max_value)                AS peak
FROM metrics_1m
WHERE bucket >= now() - INTERVAL '30 days'
  AND series_id = 42
GROUP BY hour
ORDER BY hour;
```

In the Prometheus world the same idea is a *recording rule*: a query evaluated on a schedule whose result is written back as a new, cheaper series.

```yaml
groups:
  - name: http_rollups
    interval: 1m
    rules:
      # Precompute the per-job request rate so dashboards read one cheap series
      # instead of re-deriving rate() over millions of raw samples every refresh.
      - record: job:http_requests:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))
```

### Second-order optimization: not every aggregate composes

The trap that bites people is assuming all aggregates roll up cleanly. `sum`, `min`, `max`, and `count` are *composable* — the sum of per-minute sums is the correct hourly sum, the max of per-minute maxes is the correct hourly max. But `avg` of per-minute `avg`s is **wrong** whenever the buckets have different counts; you must store `sum` and `count` separately and divide at read time. Percentiles are worse: you cannot average a p99 of p99s and get anything meaningful — to downsample percentiles correctly you must store a *sketch* (a t-digest or HDR histogram) per bucket and merge the sketches, not the percentile values. TimescaleDB ships `percentile_agg` and t-digest types precisely for this; Prometheus histograms work because they store per-bucket counts that *are* summable. The rule is: roll up the *summable building blocks*, derive the human-facing number last.

## 6. Technique three: retention tiers

**The senior rule of thumb: data does not have one lifetime, it has several — full resolution for a few days, downsampled for a few months, and a cheap archive (or nothing) after that.**

Retention is not a single switch; it is a tiered policy where each tier trades resolution and query latency for storage cost. The hot tier holds raw, full-resolution data on fast local storage for the window where you debug incidents minute-by-minute. The warm tier holds downsampled rollups on cheaper disk for the window where you do week-over-week and month-over-month analysis. The cold tier holds heavily-downsampled summaries — or nothing at all — in object storage for the window where you only ever ask coarse capacity-planning questions, if you keep it at all.

![Retention tiers trade resolution and latency for storage cost, so old data costs orders of magnitude less per gigabyte than fresh data](/imgs/blogs/time-series-data-at-scale-6.webp)

The economics are the whole point. Raw one-second data on NVMe answers in single-digit milliseconds but costs the most per gigabyte and dominates your storage bill if kept forever. The same signal as a one-hour rollup in object storage costs orders of magnitude less per gigabyte, answers in seconds rather than milliseconds, and is *all you need* for "show me CPU utilization across the fleet over the last year." The tiers exist because the *value* of a data point decays with age much faster than its *storage cost* does — so you continuously trade away the resolution you no longer need for money you would rather not spend.

Expressed as policy, retention is the other half of the continuous-aggregate story: keep the raw hot data short, keep the rollup long.

```sql
-- Drop RAW chunks older than 7 days — the rollups already captured what matters.
SELECT add_retention_policy('metrics', INTERVAL '7 days');

-- Keep the 1-minute rollup for 90 days; it's ~60x smaller than raw.
SELECT add_retention_policy('metrics_1m', INTERVAL '90 days');
```

The same pattern appears in every mature system. Thanos and Grafana Mimir run a compactor that produces 5-minute and 1-hour downsampled blocks in object storage and apply separate retention windows to raw, 5m, and 1h resolutions. ClickHouse expresses it as a `TTL` clause on the table that can both *delete* expired rows and *roll up* aging rows into an aggregated form in one declaration:

```sql
-- ClickHouse: keep raw rows 30 days, then aggregate to hourly, drop entirely at 1 year.
CREATE TABLE metrics (
    ts          DateTime,
    series_id   UInt64,
    value       Float64
) ENGINE = MergeTree
ORDER BY (series_id, ts)
TTL ts + INTERVAL 30 DAY GROUP BY series_id, toStartOfHour(ts) SET value = avg(value),
    ts + INTERVAL 1 YEAR DELETE;
```

### Second-order optimization: tier *resolution*, not just *deletion*

The mistake is treating retention as a binary "keep or delete." The teams who get their bill under control treat it as a *resolution schedule*: every signal has a curve of resolution versus age, and the policy enforces that curve. The reason this matters beyond cost is query performance — if your cold tier still stored raw points, a year-long query would scan billions of rows even from cheap storage and time out. By guaranteeing that anything older than ninety days exists *only* as an hourly rollup, you bound the worst-case scan for any query, no matter how wide its time range. Retention tiers are as much a *latency* guarantee as a *cost* control.

## 7. Technique four: compression

**The senior rule of thumb: time-series data is the most compressible data you will ever store, because the next point is almost always nearly identical to the last — exploit that and 16 bytes becomes 1.3.**

General-purpose compression (gzip, zstd) treats your data as an opaque byte stream. Time-series-aware compression treats it as what it is: a stream of timestamps that increase by nearly-constant intervals, and a stream of values that change slowly. Facebook's Gorilla paper (VLDB 2015) is the canonical description, and its two ideas are now in essentially every serious engine.

![Why time-series compresses about 12x: successive timestamps and values barely change, so delta-of-delta and XOR shrink a 16-byte point to roughly 1.3 bytes](/imgs/blogs/time-series-data-at-scale-7.webp)

**Delta-of-delta for timestamps.** Points usually arrive on a fixed interval — every 60 seconds, say. The deltas between successive timestamps are therefore nearly constant (60, 60, 60, ...), so the *delta of the deltas* is almost always exactly zero. Gorilla stores a single `0` bit for each such point and only spends real bits when the interval actually changes. A timestamp that took 8 bytes raw costs a fraction of a bit in steady state.

**XOR for floating-point values.** Consecutive values are close, so their IEEE-754 bit patterns share most of their leading and trailing bits. XOR two near-equal floats and you get a result that is mostly zeros, with a small window of "meaningful" bits in the middle. Gorilla stores the count of leading zeros and the meaningful window, skipping everything that did not change. A value that did not change at all (very common — a gauge that is flat) XORs to all zeros and costs one bit.

The Python below makes the mechanism concrete — it is the intuition, not the production bit-packing, but it shows exactly why the bits collapse:

```python
import struct

def to_bits(x: float) -> int:
    """IEEE-754 double as a 64-bit integer."""
    return struct.unpack(">Q", struct.pack(">d", x))[0]

# --- Timestamps: delta-of-delta ---
timestamps = [1623456000, 1623456060, 1623456120, 1623456179, 1623456239]
deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]          # [60, 60, 59, 60]
dod    = [deltas[0]] + [b - a for a, b in zip(deltas, deltas[1:])]    # [60, 0, -1, 1]
print("delta-of-delta:", dod)
# Most entries are 0 -> Gorilla writes a single '0' bit for each; only the jitter costs bits.

# --- Values: XOR of successive floats ---
values = [21.4, 21.4, 21.5, 21.4]
prev = to_bits(values[0])
for v in values[1:]:
    cur = to_bits(v)
    x = prev ^ cur
    if x == 0:
        print(f"value={v}: xor=0  -> 1 bit (unchanged)")
    else:
        lead  = 64 - x.bit_length()
        trail = (x & -x).bit_length() - 1
        print(f"value={v}: xor=0x{x:016x}  lead={lead} trail={trail} "
              f"meaningful={64 - lead - trail} bits")
    prev = cur

# Gorilla compressed Facebook's points from 16 bytes to ~1.37 bytes on average — a ~12x win.
```

Columnar storage is the structural enabler. Delta and XOR encoding only work when consecutive *same-column* values sit next to each other in memory, which is exactly what a column store gives you and a row store does not — another reason the engines converge on columnar layouts. (For the general row-vs-column tradeoff, see [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores).) In TimescaleDB you enable it per hypertable and let a policy compress chunks once they age past the hot window:

```sql
-- Convert chunks to compressed columnar form, grouped by series, ordered by time.
ALTER TABLE metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'series_id',   -- group rows of one series together
    timescaledb.compress_orderby   = 'time DESC');   -- so delta/XOR see ordered runs

-- Compress anything older than 7 days; the hot tier stays row-form for fast ingest.
SELECT add_compression_policy('metrics', INTERVAL '7 days');
```

### Second-order optimization: compression and mutability are a tradeoff

Compressed columnar chunks are wonderful to scan and terrible to modify. Once a chunk is compressed, an `UPDATE` or out-of-order `INSERT` into it is expensive — the engine must decompress, modify, and recompress a whole segment. This is fine *because the workload is append-only and immutable*, which is the entire premise. But it means you must keep the hot/recent window in uncompressed row form long enough to absorb late-arriving and out-of-order data (network buffering, agent retries, clock skew can all deliver points minutes or hours late), and only compress once a chunk is safely "closed." Set your compression delay longer than your worst realistic out-of-order lag, or you will pay the decompress-recompress tax on every straggler.

## 8. The engine landscape

**The senior rule of thumb: there is no "best" time-series database — there is the monitoring axis (pull vs push, low-cardinality metrics) and the events axis (push, high-cardinality wide rows), and almost every engine is optimized for one end.**

The market looks crowded, but it sorts cleanly along two axes: *what you are storing* (low-cardinality numeric metrics for monitoring, versus high-cardinality wide events and logs) and *how far you need to scale* (single-node-plus-remote versus natively horizontal).

![The time-series engine landscape: no engine wins everywhere, and the axes that decide are monitoring-vs-events and single-node-vs-horizontal](/imgs/blogs/time-series-data-at-scale-8.webp)

| Engine | Model | Storage | Downsampling | Compression | Scale story |
| --- | --- | --- | --- | --- | --- |
| **Prometheus** | Pull (scrape) | Local TSDB on disk | Recording rules | Gorilla-style delta-of-delta + XOR | Single node; offload long-term via remote-write |
| **InfluxDB** | Push | TSM (LSM-like), v3 → columnar Arrow/Parquet | Tasks / continuous queries | Columnar + delta encoding | Single node (OSS) → clustered / cloud |
| **TimescaleDB** | Push (SQL) | Postgres rows → native columnar compression | Continuous aggregates | Native columnar (delta, Gorilla, dictionary) | Single node → multi-node; full SQL |
| **ClickHouse** | Push | Columnar MergeTree | Materialized views + TTL rollup | DoubleDelta, Gorilla codec, ZSTD | Shards + replicas, petabyte scale |
| **M3 / Cortex / Thanos / Mimir** | Prometheus remote-write | Local + object storage | Block downsampling (5m, 1h) | M3TSZ / Gorilla-derived | Horizontally scalable, billions of series |
| **VictoriaMetrics** | Push + Prometheus-compatible | Custom columnar | Recording rules / downsampling | Aggressive delta + gorilla-like | Single binary or clustered, very high ingest |

A few opinions to navigate it:

- **Prometheus** is the default for infrastructure and application monitoring, and its pull model is a feature, not a bug — the scraper knows what is up, service discovery drives targets, and there is no agent fan-in to overwhelm. Its weakness is exactly what you would predict: a single node, bounded local retention (commonly ~15 days), and a hard cardinality ceiling. You do not "scale Prometheus"; you pair it with a long-term backend.
- **Cortex, Thanos, Mimir, and M3** are that backend. They take Prometheus remote-write, store blocks in object storage, run a compactor that downsamples to 5-minute and 1-hour resolutions, and serve queries horizontally across many machines. This is how you get global, multi-year, multi-tenant Prometheus.
- **TimescaleDB** wins when you want time-series *and* relational data in the same place with real SQL, joins, and the Postgres ecosystem — IoT platforms, financial tick data, anything where the time-series is one table among many. It is the lowest-friction choice for a team that already runs Postgres.
- **ClickHouse** is the events-and-logs answer, not the metrics answer. When each record is a wide row with high-cardinality fields (a request log with user, route, status, latency, region) and you want arbitrary ad-hoc aggregation across billions of rows, ClickHouse's columnar scan speed is in a different class. It does not have a per-series memory model, so high cardinality that would kill Prometheus is just more rows.
- **InfluxDB and VictoriaMetrics** sit in between — purpose-built TSDBs with their own query languages and strong single-node ingest, good defaults if you want a dedicated metrics store without the Postgres or Prometheus ecosystems.

This is the natural endpoint of [polyglot persistence](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store): you will likely run more than one. Prometheus for live monitoring, a remote-write backend for long-term metrics, and ClickHouse for logs and wide events is an extremely common — and correct — three-store topology.

## 9. Case studies from production

### 1. Facebook Gorilla: the in-memory TSDB that defined the genre

Facebook's monitoring data had outgrown its HBase-backed store — query latency was too high to be useful during an incident, which is the one time monitoring data must be fast. The Gorilla team's insight (VLDB 2015) was that you only need the *recent* window in fast storage; they built an in-memory TSDB holding roughly the last 26 hours, fronting the durable store. The enabling trick was compression: delta-of-delta on timestamps and XOR on values brought the average point from 16 bytes down to about 1.37 bytes — a roughly 12× reduction that made it economical to keep the working set in RAM across the fleet. The result was query latencies dropping by orders of magnitude and a design that got open-sourced as Beringei and copied, in spirit, into Prometheus, M3, InfluxDB, and ClickHouse's Gorilla codec. The lesson that outlived the system: *compression is not a storage optimization, it is a latency optimization* — it changes what fits in the fast tier.

### 2. Uber M3: Prometheus semantics at planetary scale

Uber ran Prometheus and hit its ceiling hard — a single Prometheus could not hold their metric volume, and federation was fragile. They built M3: M3DB as the horizontally-scalable storage engine, an M3 Coordinator presenting a Prometheus-compatible remote-write and query interface, and an M3 Aggregator for rollups. The scale it was built for is the headline — billions of unique time series and tens of millions of metrics ingested per second across the fleet. The architectural lesson is that they did *not* invent a new query language or monitoring model; they kept Prometheus and PromQL semantics and replaced only the storage and aggregation layers underneath. Compatibility at the interface, radical change underneath, is how you scale an entrenched system without retraining every engineer.

### 3. The `DELETE`-driven Postgres meltdown

This is the incident from the opening, and it is worth the anatomy because it is so common. A team stored request events in a plain Postgres table with a nightly retention `DELETE`. Symptoms: steadily climbing disk usage *despite* deletes, autovacuum running continuously and never catching up, and time-range queries that slowed week over week. The wrong first hypothesis was "we need a bigger box." The actual root cause was that `DELETE` only marks tuples dead, the dead tuples bloated both heap and index faster than autovacuum could reclaim them, and the physical disorder meant scans read mostly-dead pages. The fix was not vertical scaling — it was converting the table to a TimescaleDB hypertable so retention became `drop_chunks` (instant, no vacuum) and recent-window queries hit one chunk's index. Disk usage flattened the day the cron `DELETE` was deleted. The lesson: in time-series, *how you delete* dominates *how you write*.

### 4. The `user_id` label that OOM-ed Prometheus

A platform team added a `customer_id` label to a high-traffic request metric to enable per-customer dashboards. Within hours Prometheus memory climbed from a few gigabytes to the box limit and it OOM-killed, taking down alerting with it. The wrong hypothesis was "Prometheus is leaking." The truth was a cardinality explosion: the metric went from a few hundred series to a few million because `customer_id` was effectively unbounded, and each series carries fixed per-series memory overhead. The fix had three parts — drop the label at scrape time with `metric_relabel_configs`, add a `topk(10, count by (__name__)(...))` cardinality dashboard with an alert, and move per-customer breakdowns to wide events in ClickHouse where cardinality is free. The lesson, again: in a time-series store you do not pay for data, you pay for *distinct series*, and an unbounded label is an unbounded bill.

### 5. Cloudflare's analytics pipeline on ClickHouse

Cloudflare processes an enormous volume of HTTP request events for customer analytics — far too many to keep as raw rows indefinitely and far too high-cardinality (per-zone, per-status, per-country) for a metrics TSDB. They moved the analytics pipeline onto ClickHouse, exploiting columnar compression and materialized-view rollups so that customer-facing dashboards read pre-aggregated tables rather than scanning raw events, while ad-hoc internal analysis can still drop to the raw rows when needed. The design point worth stealing: they did not try to make one store serve both the live-metric and the wide-event workload. The wide, high-cardinality events went to the engine built for them, and the rollups bridged raw scale to dashboard latency.

### 6. Thanos / Mimir downsampling for multi-year retention

A growing infrastructure org needed years of metric history for capacity planning but could not afford to keep raw resolution that long, nor scan it if they did. They adopted Thanos (and later Mimir), whose compactor writes 5-minute and 1-hour downsampled blocks alongside the raw blocks in object storage, each with its own retention. Long-range dashboards automatically read the coarsest block that satisfies the query's step, so a two-year chart scans hourly blocks — kilobytes per series — instead of raw. The operational lesson: downsampling and tiering are not optional add-ons at this scale; they are the only reason a multi-year query returns before the dashboard times out. Without downsampled blocks, "show me last year" is a denial-of-service against your own query layer.

### 7. The IoT fleet that drowned in out-of-order writes

A connected-device company ingested sensor readings from hundreds of thousands of devices, many on flaky cellular links that buffered and replayed data hours late. Their first TimescaleDB setup compressed chunks aggressively at one hour old — and then every late batch from a reconnecting device triggered an expensive decompress-modify-recompress cycle, spiking write latency. The wrong hypothesis was "compression is too slow." The real issue was a compression delay shorter than the real-world out-of-order lag. The fix was to push the compression policy out to 48 hours, comfortably beyond the worst observed device delay, so chunks only compressed once they were truly closed. The lesson: compression assumes immutability, and you must let recent data stay mutable as long as stragglers can realistically arrive.

### 8. The dashboard that recomputed `rate()` over millions of samples

An SRE team's main dashboard had thirty panels, each running `rate(...)` and `histogram_quantile(...)` over high-resolution raw series, refreshing every fifteen seconds across dozens of viewers. The query layer was constantly pegged and dashboards loaded in tens of seconds. The wrong fix attempted was "add more query replicas." The right fix was recording rules: precompute the per-job rates and the histogram quantiles once a minute into cheap derived series, and point the panels at those. Query CPU dropped by more than an order of magnitude and load times fell to sub-second. The lesson mirrors caching the read path generally — *the cheapest query is the one whose answer you already computed* — applied to the time dimension via rollups.

## 10. When to reach for a time-series engine — and when not to

Reach for a dedicated time-series engine when:

- Your writes are **append-only and time-ordered** at high volume — metrics, IoT telemetry, financial ticks, clickstream, structured logs — and updates are rare to nonexistent.
- Your reads are **time-range plus aggregation** (avg, sum, rate, percentile over a window), not point lookups by primary key.
- You need **automatic expiry of old data** continuously, and `DELETE`-plus-vacuum is already hurting or obviously will.
- Your storage bill is dominated by **immutable historical data** that would compress 10–50× and that you would happily downsample with age.
- You are fighting **cardinality** — many distinct series — and need a system whose memory and index model is built for it (or, for genuinely unbounded cardinality, a wide-event store like ClickHouse).

Skip it — or stay on your relational database — when:

- Your data volume is **modest** (a partitioned Postgres table with a `BRIN` index and `pg_partman`, or just TimescaleDB-as-an-extension, is plenty; do not stand up Prometheus-plus-Mimir for ten thousand points a day).
- Your access pattern is **transactional**, not analytical — you update rows in place, join across entities, and read by key. That is OLTP; a TSDB is the wrong tool.
- You need **strong cross-series transactional guarantees** — time-series engines are built for high-throughput appends, not multi-row ACID transactions.
- Your "time-series" is really **a few hundred rows you graph occasionally** — the operational cost of a dedicated engine dwarfs the benefit.
- You can answer everything from **rollups computed in your existing warehouse** — if a nightly batch job into your OLAP store already serves the dashboards, you may not need a real-time TSDB at all.

> The deepest lesson across every one of these systems is the same one the opening incident taught: time-series is not "data that happens to have a timestamp." It is a workload whose primary key *is* time, and every winning design — partitioning, rollups, tiering, compression — falls out of taking that seriously. Treat time as the organizing principle, not as just another column, and the right architecture stops being a choice and starts being obvious.

## Further reading

- [LSM-trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) — the append-friendly storage layer most TSDBs are built on.
- [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) — why column layout is what makes time-series compression and scans fast.
- [Polyglot persistence: choosing the right store](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store) — why you will run a TSDB *alongside* your other databases, not instead of them.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the append-only log that usually sits in front of a TSDB as the ingestion buffer.
- *Gorilla: A Fast, Scalable, In-Memory Time Series Database* (Pelkonen et al., VLDB 2015) — the source of delta-of-delta and XOR compression.
- The TimescaleDB docs on hypertables, continuous aggregates, compression, and retention policies — the clearest single reference for the four techniques in one system.

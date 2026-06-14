---
title: "Join Algorithms: Nested Loop, Hash Join, and Merge Join"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer deep dive into the three physical join operators every relational database uses, their cost models, how the planner chooses between them, and how to force the right one in production."
tags:
  [
    "joins",
    "query-optimization",
    "hash-join",
    "merge-join",
    "nested-loop",
    "postgres",
    "mysql",
    "query-planner",
    "sql",
    "performance",
    "database",
    "explain",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/join-algorithms-nested-loop-hash-merge-1.webp"
---

There is a specific kind of pager alert that every database engineer eventually meets. A report that ran in 200 milliseconds yesterday is now taking four minutes. Nobody deployed schema changes. The data grew a little, but not 1000x. The query text is byte-for-byte identical. You run it in staging and it is instant. You run it in production and it hangs. And when you finally pull the `EXPLAIN ANALYZE`, the answer is sitting right there in one line: `Nested Loop`, where last week it said `Hash Join`.

The query did not change. The *plan* changed. And the plan changed because the optimizer's estimate of how many rows would flow out of one branch drifted from "about 12" to "about 12" while reality drifted from 12 to 1.4 million. A nested loop over 12 rows is the fastest thing in the world. A nested loop over 1.4 million rows, each one re-probing a table, is a small space heater that happens to also return your result set, eventually.

Almost everything interesting about relational database performance lives in the join. A `SELECT` from one table is a scan, and scans are easy to reason about. The moment you write `JOIN` you have handed the database a combinatorial problem — match every qualifying row on the left with every qualifying row on the right — and it has exactly three physical strategies to solve it: **nested loop**, **hash join**, and **merge join**. That is the entire menu. Every join you have ever run, in Postgres, MySQL, SQL Server, Oracle, SQLite, DuckDB, Spark, or a distributed system like Citus, was executed by one of those three algorithms or a minor variation on them. The optimizer's whole job, for the join, is to pick which one — and to pick the order in which to apply them when there are many tables.

This article is a tour of those three algorithms: how each one works, what it costs, when it is the right answer, how it fails, and — the part that actually saves you at 2am — how to read which one you got and how to force a better one. The diagram below is the mental model for the entire piece.

## The mental model: one logical join, three physical operators

![How the planner picks a physical join: selectivity, indexes, sortedness, and work_mem route the same SQL join to one of three operators](/imgs/blogs/join-algorithms-nested-loop-hash-merge-1.webp)

Read the figure left to right. On the far left is a *logical* join — `A JOIN B ON a.k = b.k`. That is what you wrote. It says nothing about how the match happens; it is a declarative statement of intent. The optimizer takes it and runs cost estimates: how many rows will each side produce, are there usable indexes, does either side already arrive sorted on the join key, will a hash table fit in the memory budget. Based on those estimates it routes the logical join to one of three physical operators.

If the outer side is tiny and the inner side has an index on the join key, it picks a **nested loop** — for each of the few outer rows, probe the inner index. If both sides are large and unsorted, it builds a hash table on the smaller side and probes it with the larger — a **hash join**. If both sides happen to arrive already sorted on the join key (because an index supplies that order for free, or because a sort higher in the plan needs it anyway), it sweeps two cursors down the inputs in lockstep — a **merge join**. Three operators, three different shapes of work, three different cost curves.

The thing to internalize before anything else: **the operator is chosen, not written.** Standard SQL gives you no syntax to say "use a hash join." You influence the choice indirectly — through indexes, statistics, query structure, and memory settings — and the optimizer decides. When the choice is wrong, it is almost always because the *estimate* feeding the choice was wrong. That is the through-line of this entire article: the algorithms are simple and well-understood; the failures come from the cost model being fed bad numbers. We will spend the back half of the piece on exactly that.

> A join algorithm is never "slow" in the abstract. It is slow because the planner picked it for a workload shape it is wrong for — and it picked it because someone, somewhere, lied to the cost model about how many rows there would be.

Before we dissect each operator, it helps to be honest about a common assumption mismatch. Most engineers carry a mental model of joins that is subtly, expensively wrong.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "A join is a join; the database just does it." | One operation, roughly constant behavior. | Three completely different algorithms with cost curves that cross; the wrong one is orders of magnitude slower. |
| "Adding an index makes joins fast." | Indexes universally help. | An index makes *nested loop* fast and is irrelevant — or actively misleading the planner — for hash and merge joins on large inputs. |
| "If it's fast in dev it's fast in prod." | Same query, same speed. | Dev has 10k rows so everything is a cheap nested loop; prod has 100M and needs a hash join. The plan flips on data size. |
| "The optimizer knows my data." | It has perfect information. | It has *sampled statistics* that go stale, and correlated columns and skewed values routinely break its row estimates by 100–1000x. |
| "More memory always helps joins." | Bigger `work_mem`, faster. | It helps until the hash table no longer fits and spills to disk; and over-provisioning it per-connection can OOM the box under concurrency. |

Every row in that table is a section of this article. Let us build each operator from first principles, starting with the one that is both the simplest and the most dangerous.

## 1. Nested loop join: the honest brute force

**The senior rule of thumb: a nested loop is the right join exactly when the outer side is tiny and the inner side is indexed on the join key — and a catastrophe otherwise.**

The nested loop is the join you would write yourself if you had never heard of a database. Two loops, one inside the other:

```python
def nested_loop_join(outer, inner, match):
    for r in outer:            # outer ("driving") relation
        for s in inner:        # inner relation, re-scanned for every r
            if match(r, s):    # join predicate, e.g. r.k == s.k
                yield (r, s)
```

That is the whole algorithm. It is correct for *any* join predicate — equality, range, inequality, a function of both rows, anything — which is why it is the only one of the three that can handle non-equi joins like `a.ts BETWEEN b.start AND b.end`. Hash and merge joins both require an equality on the join key; the nested loop does not care.

Its cost is also the easiest to reason about, and the reason it terrifies people who have been burned. CMU's Andy Pavlo frames the cost analysis of every join in terms of pages of I/O, which is the right currency because for disk-based systems I/O dominates. Let table $R$ (the outer) have $M$ pages and $m$ tuples; let table $S$ (the inner) have $N$ pages and $n$ tuples. The naive nested loop scans $R$ once ($M$ pages) and then, for every one of its $m$ tuples, scans all of $S$ ($N$ pages):

$$\text{Cost}_\text{naive} = M + (m \times N)$$

That $m \times N$ term is the monster. Pavlo's [15-445 join algorithms lecture](https://15445.courses.cs.cmu.edu/fall2024/notes/12-joins.pdf) uses a concrete example that is worth memorizing: $M = 1000$, $m = 100{,}000$, $N = 500$, $n = 40{,}000$, at $0.1$ ms per I/O. The naive nested loop costs $1000 + (100{,}000 \times 500) = 50{,}000{,}000$ page reads — about **1.4 hours**. The same join with the other algorithms runs in well under a second. That is the gap between getting the join right and getting it wrong.

![Naive nested loop: each of three outer rows triggers a full sequential scan of the entire inner table](/imgs/blogs/join-algorithms-nested-loop-hash-merge-2.webp)

The figure above is the failure mode made visual. Three outer rows, and each one drags a full sweep across the inner table. The inner table never changes between outer rows — we are reading the exact same 500 pages 100,000 times — but with no index there is no way to know which inner row matches without checking all of them. This is the literal $O(n \times m)$ trap, and it is what you saw in the opening pager story when the outer estimate of "12 rows" turned out to be 1.4 million.

### Block nested loop: the cheapest possible improvement

The first thing any real database does is stop re-reading the inner table one outer *tuple* at a time and instead buffer a chunk of the outer table in memory, comparing each inner row against the whole buffer. This is the **block nested loop**. Instead of $m \times N$, the inner is re-scanned once per *block* of the outer:

$$\text{Cost}_\text{block} = M + \left(\lceil M / (B - 2) \rceil \times N\right)$$

where $B$ is the number of memory buffers available (one reserved for scanning the inner, one for output, the rest for buffering the outer). On Pavlo's example numbers with $B = 100$, the block nested loop drops to about **50 seconds** — a 100x improvement over naive, purely from buffering, with no index at all. MySQL implemented exactly this for years. Per the [MySQL nested-loop join documentation](https://dev.mysql.com/doc/refman/8.0/en/nested-loop-joins.html), the Block Nested-Loop (BNL) algorithm "uses buffering of rows read in outer loops to reduce the number of times tables in inner loops must be read" — if 10 rows are buffered and passed to the inner loop, each inner row is compared against all 10 at once. The buffer size was the `join_buffer_size` knob. It is still a quadratic algorithm; it just shrinks the constant.

### Index nested loop: where the nested loop earns its keep

The real transformation comes from putting an index on the inner table's join key. Now, instead of scanning all of $S$ for each outer row, we probe a B-tree — a handful of page reads regardless of how big $S$ is. The cost becomes:

$$\text{Cost}_\text{index} = M + (m \times C)$$

where $C$ is the constant cost of one index probe (a B-tree traversal, typically 3–5 page reads for a tree of any realistic depth — see [how B-tree indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) for why the depth stays small even for billions of rows). The $N$ term — the size of the inner table — has vanished from the cost entirely. The inner table can be a billion rows and each probe still costs the same logarithmic handful of reads.

![Naive nested loop versus index nested loop: an inner-side index collapses the cost from quadratic to near-linear](/imgs/blogs/join-algorithms-nested-loop-hash-merge-3.webp)

This is the figure to show anyone who thinks "the nested loop is the bad join." On the left, every outer row triggers a full inner scan: 100,000 outer rows times 500 inner pages is 50 million I/Os, ~50 seconds. On the right, every outer row does a ~4-page index probe: 100,000 times 4 is 400,000 I/Os, ~40 milliseconds. Same algorithm. Same data. **A thousand times faster, from one index.** As [Use The Index, Luke!](https://use-the-index-luke.com/sql/join/nested-loops-join-n1-problem) puts it, the index nested loop is "particularly effective if the outer input is small and the inner input is preindexed and large."

Here is the index nested loop in pseudocode, this time showing the index probe explicitly:

```python
def index_nested_loop_join(outer, inner_index, match):
    for r in outer:                       # small outer relation
        for s in inner_index.lookup(r.k): # B-tree probe: O(log n), not O(n)
            if match(r, s):               # verify (index may be non-covering)
                yield (r, s)
```

The crucial property: this is cheap **only when the outer side is small.** Each outer row costs an index probe. If the outer side has 12 rows, that is 12 probes — instant. If the planner *thinks* the outer side has 12 rows but it actually has 1.4 million, that is 1.4 million index probes, and even at a few microseconds each you are now spending minutes on random I/O, thrashing the buffer cache, with the disk pegged. This is the single most common cause of the "fast in dev, slow in prod" plan flip, covered in depth in [why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod).

Let us make it concrete in Postgres. Set up two tables — a small filtered customer set joined to a large orders table.

```sql
CREATE TABLE customers (id bigint PRIMARY KEY, region text, signup_date date);
CREATE TABLE orders   (id bigint PRIMARY KEY, customer_id bigint, total numeric, created_at timestamptz);

INSERT INTO customers
SELECT g, (ARRAY['us','eu','apac'])[1 + g % 3], '2020-01-01'::date + (g % 1500)
FROM generate_series(1, 100000) g;

INSERT INTO orders
SELECT g, 1 + (g % 100000), (random()*500)::numeric(10,2), now() - (g % 1000) * interval '1 hour'
FROM generate_series(1, 10000000) g;

CREATE INDEX orders_customer_id_idx ON orders(customer_id);
ANALYZE customers; ANALYZE orders;
```

Now a query that selects a tiny slice of customers and joins to their orders. Because the filtered outer side is small and the inner has an index on `customer_id`, the planner should pick an index nested loop:

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.id, o.id, o.total
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE c.signup_date = DATE '2020-01-01';   -- ~67 customers
```

```
Nested Loop  (cost=0.86..2847.21 rows=6700 width=22) (actual time=0.043..4.911 rows=6700 loops=1)
  Buffers: shared hit=20371
  ->  Seq Scan on customers c  (cost=0.00..2041.00 rows=67 width=8) (actual time=0.018..18.4 rows=67 loops=1)
        Filter: (signup_date = '2020-01-01'::date)
        Rows Removed by Filter: 99933
  ->  Index Scan using orders_customer_id_idx on orders o  (cost=0.43..11.10 rows=100 width=22) (actual rows=100 loops=67)
        Index Cond: (customer_id = c.id)
Planning Time: 0.31 ms
Execution Time: 6.2 ms
```

Read the shape: the outer `Seq Scan` returns 67 customers, and the inner `Index Scan` runs `loops=67` — once per outer row — each probe pulling ~100 matching orders out of 10 million via the index. Six milliseconds. The `loops=N` annotation is the single most important number to learn in `EXPLAIN ANALYZE` output, because the per-loop cost is multiplied by it; a full treatment lives in [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer). When `loops` is small, the nested loop is a precision instrument. When `loops` is large, it is the space heater.

### Second-order optimization: the N+1 problem is a nested loop in disguise

The non-obvious gotcha here is that the nested loop's failure mode escapes the database entirely and shows up in application code. When an ORM loads 100 parent rows and then issues one query per parent to load children, that is a hand-rolled, *uncached* nested loop join executed across the network — 1 query for the parents plus N for the children, the infamous N+1 problem. [Use The Index, Luke! explicitly frames N+1 as a nested loop](https://use-the-index-luke.com/sql/join/nested-loops-join-n1-problem): the application is the outer loop and each round-trip to the database is the inner probe, except now each "probe" carries milliseconds of network latency instead of microseconds of index traversal. The fix — a single `JOIN` or an `IN (...)` batch — lets the database execute the loop in-process with a hash or merge join when N is large. The lesson generalizes: a nested loop is only as good as the cost of one inner probe, and crossing a network boundary makes that cost catastrophic.

## 2. Hash join: build small, probe big, scan once

**The senior rule of thumb: when both sides are large, unsorted, and joined by equality, hash join is almost always the right answer — provided the build side fits in `work_mem`.**

The nested loop's weakness is that without an index, finding matches for an inner row means scanning. The hash join attacks exactly that weakness by building an index *on the fly* — a hash table — over the smaller input, then streaming the larger input through it. As Pavlo's lecture notes put it, hash joins "use a hash table to split the tuples into smaller chunks based on their join attribute(s), which reduces the number of comparisons the DBMS needs to perform per tuple."

The algorithm has two phases:

```python
def hash_join(build_side, probe_side, key):
    # Phase 1 — BUILD: hash the smaller input into memory
    ht = defaultdict(list)
    for b in build_side:        # smaller input, measured in bytes
        ht[key(b)].append(b)

    # Phase 2 — PROBE: stream the larger input, look up each row
    for p in probe_side:        # larger input
        for b in ht.get(key(p), ()):   # expected O(1) lookup
            if key(b) == key(p):       # verify: hash collisions are possible
                yield (b, p)
```

![Hash join: the build side is hashed once into a memory table, then each probe row is matched in expected constant time](/imgs/blogs/join-algorithms-nested-loop-hash-merge-4.webp)

The figure shows the dataflow. The build input is hashed with function $h_1$ into a hash table living in memory; the probe input streams through, each row hashed with the same $h_1$ to jump straight to the matching bucket. Because hashing can collide — two different keys landing in the same bucket — the operator must re-check the actual key values before emitting a match. The payoff is the property at the bottom of the figure: **each input is scanned exactly once.** No re-scanning, no index required on the base tables. The whole point, as the [MySQL 8 hash join announcement](https://dev.mysql.com/blog-archive/hash-join-in-mysql-8/) phrases it, is that "the server has scanned each input only once, using constant time lookup to find matching rows between the two inputs."

The in-memory cost is therefore linear in both inputs: $O(M + N)$ I/O to read both sides once, plus CPU to hash and probe. Compare that to the nested loop's $M + (m \times N)$ and you see why hash join dominates as the inputs grow — it has no quadratic term at all. On Pavlo's running example the in-memory hash join runs in roughly **0.45 seconds** versus the naive nested loop's 1.4 hours.

Two details decide everything about hash join performance in practice: which side gets built, and whether it fits in memory.

### Which side to build

You build the *smaller* side, and "smaller" means smaller in bytes, not rows. MySQL's documentation is precise here: "Ideally, the server will choose the smaller of the two inputs as the build input (measured in bytes, not number of rows)." The reason is memory: the build side has to fit in the hash table, so you want the side that takes the least space. A table with 1 million narrow rows might be smaller in bytes than a table with 100,000 wide rows, so row count alone can mislead. Postgres makes this decision from its size estimates, which is one more place a bad estimate hurts: build the wrong (larger) side and you are far more likely to overflow memory and spill.

### When the hash table does not fit: grace hash join and spilling

The build-side-in-memory assumption is the whole ballgame. If the build side is bigger than the memory budget — `work_mem` in Postgres, `join_buffer_size` in MySQL — the operator cannot hold the hash table and must fall back to a partitioned strategy. This is the **grace hash join** (named after the GRACE database machine), also called the partitioned hash join.

![Grace hash join: when the build side exceeds work_mem, both inputs are partitioned to disk and joined one batch pair at a time](/imgs/blogs/join-algorithms-nested-loop-hash-merge-5.webp)

The idea, shown in the figure: instead of one giant hash table, partition *both* inputs by a hash of the join key into $B$ batches, written out to temporary files on disk. The partitioning guarantees that any two rows that could possibly match land in the *same* batch number on both sides — because they have the same key, so the same hash, so the same partition. Then you process one batch pair at a time: load build-batch[i] into memory, probe it with probe-batch[i], emit matches, move to the next batch. Each batch is sized so its build side fits in memory, so each batch is a fast in-memory join.

There is a subtle and important trick here that MySQL's documentation spells out: the partitioning hash function must be *different* from the in-memory hash function. As the MySQL blog explains, "if we were to use the same hash function for both operations, we would get an extremely bad hash table when loading a build chunk file into the hash table, since all rows in the same chunk file would hash to the same value." So $h_1$ partitions into batches and a different $h_2$ builds the in-memory table within each batch. Pavlo's lecture describes the same recursive-partitioning idea: if a single partition still does not fit, partition it again with yet another hash function until it does.

The cost of grace hash join is roughly $3 \times (M + N)$: read both inputs to partition them ($M + N$), write the partitions out ($M + N$ — this is the spill), then read them back to join ($M + N$). That is three passes instead of one — still linear, but 3x the I/O of an in-memory hash join, and now that I/O hits temporary files on disk. This is the single most important operational fact about hash joins: **a hash join that fits in memory is fast; a hash join that spills is 3x the I/O and that I/O is to disk.**

You can see the spill directly in Postgres `EXPLAIN ANALYZE` via the `Batches` count. Let us force it. Take a query that joins two large unindexed result sets — say, all orders joined to all customers, aggregating revenue per region:

```sql
SET work_mem = '4MB';   -- deliberately small to force a spill

EXPLAIN (ANALYZE, BUFFERS)
SELECT c.region, count(*), sum(o.total)
FROM orders o
JOIN customers c ON c.id = o.customer_id
GROUP BY c.region;
```

```
HashAggregate  (cost=... rows=3 width=44) (actual time=4120 rows=3 loops=1)
  Group Key: c.region
  ->  Hash Join  (cost=... rows=10000000 width=14) (actual time=180..3600 rows=10000000 loops=1)
        Hash Cond: (o.customer_id = c.id)
        ->  Seq Scan on orders o  (actual rows=10000000 loops=1)
        ->  Hash  (actual time=170 rows=100000 loops=1)
              Buckets: 32768  Batches: 8  Memory Usage: 3072kB
              ->  Seq Scan on customers c  (actual rows=100000 loops=1)
Planning Time: 0.2 ms
Execution Time: 4250 ms
```

The line that matters is `Batches: 8`. The build side (customers) did not fit in 4 MB of `work_mem`, so Postgres partitioned it into 8 batches and spilled 7 of them to temp files. As [pgMustard's hash join guide](https://www.pgmustard.com/docs/explain/hash-join) and the [Stormatics walkthrough of hash joins](https://stormatics.tech/blogs/understanding-hash-aggregates-and-hash-joins-in-postgresql) both note: `Batches: 1` means the hash table fit in memory entirely with no spill; `Batches > 1` means it spilled — you should either raise `work_mem` or find a way to make the build side smaller. Bump `work_mem` to `64MB` and re-run, and you will see `Batches: 1` and the execution time drop sharply, because now the whole hash table lives in memory and you pay one pass instead of three.

### The adaptive-spill failure: when the estimate is too small to recover

There is a darker corner of hash join spilling that has caused real production outages. The number of batches, $B$, is chosen *at query startup* from the planner's estimate of the build side's size. If that estimate is wildly too small, $B$ is too small, and an individual batch ends up far larger than `work_mem` — but the classic algorithm cannot increase $B$ after the fact. A [long-standing Postgres discussion on hash join memory](https://www.postgresql.org/message-id/15661.1109887540%40sss.pgh.pa.us) describes exactly this: "the hash join code is supposed to spill tuples to disk when the hashtable exceeds work_mem... [but] the code divides the hash key space into N batches chosen at query startup based on the planner's estimate. If that estimate is way too small, then an individual batch can be way too large, but the code can't recover by adjusting N after the fact." The result is a hash join that blows through its memory budget and can take down the server with an out-of-memory kill. Modern Postgres has improved on this — it can add batches dynamically in many cases — but the lesson stands: **hash join's memory safety depends on the row estimate, and a bad estimate can turn a bounded-memory operator into an OOM.** [Microsoft's writeup on hash join OOM risk](https://techcommunity.microsoft.com/blog/adforpostgresql/understanding-hash-join-memory-usage-and-oom-risks-in-postgresql/4500308) covers the modern version of this hazard in detail.

### Parallel hash join

Modern Postgres (since version 11) can build a *shared* hash table that multiple worker processes populate in parallel, then have all workers probe it concurrently — a parallel hash join. [EnterpriseDB's parallel hash join writeup](https://www.enterprisedb.com/postgres-tutorials/parallel-hash-joins-postgresql-explained) explains the mechanics: the build side is scanned in parallel by all workers cooperating to fill one shared hash table in shared memory, then the probe side is partitioned across workers. You see it in plans as `Parallel Hash Join` with a `Parallel Hash` node and `Workers Launched: N`. The win is real on large analytical joins — near-linear speedup in the number of workers for the probe phase — but it consumes `work_mem` *per worker* for the non-shared bits, so the memory math gets more dangerous under parallelism, not less.

### Second-order optimization: the Bloom filter sidecar

A non-obvious refinement that many engines apply: during the build phase, also construct a Bloom filter over the build keys. Then during the probe phase, before doing the (relatively expensive) hash table lookup, check the Bloom filter — if it says "definitely not present," skip the probe entirely. Pavlo's lecture calls this "sideways information passing," and it is a real win when most probe rows have no match, because a Bloom filter check is a couple of cache-friendly bit tests versus a full hash lookup and key comparison. It also reduces disk I/O in the spilled case by avoiding reading probe rows that cannot match anything. (Bloom filters are the same probabilistic structure that makes [LSM-tree](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) reads fast; the trick recurs all over storage engines.)

## 3. Merge join: two cursors, one linear sweep, sorted output

**The senior rule of thumb: merge join wins when both inputs already arrive sorted on the join key — typically from index order — or when the query needs sorted output anyway.**

The merge join is the most elegant of the three and the most situational. It exploits a property the other two ignore: if both inputs are *sorted* on the join key, you can find all matches in a single linear pass, like merging two sorted lists. Two cursors, one on each input, advance in lockstep: whenever one cursor points at a smaller key, advance it; when both point at equal keys, emit the match and advance.

```python
def merge_join(R, S, key):     # R and S MUST be sorted on key
    i, j = 0, 0
    while i < len(R) and j < len(S):
        if key(R[i]) < key(S[j]):
            i += 1                          # advance the smaller side
        elif key(R[i]) > key(S[j]):
            j += 1
        else:                               # keys equal: emit all matching pairs
            # handle duplicates: cross-product the equal-key runs on both sides
            kr = key(R[i])
            r_end = i
            while r_end < len(R) and key(R[r_end]) == kr: r_end += 1
            s_end = j
            while s_end < len(S) and key(S[s_end]) == kr: s_end += 1
            for a in range(i, r_end):
                for b in range(j, s_end):
                    yield (R[a], S[b])
            i, j = r_end, s_end
```

![Merge join: two cursors sweep two sorted inputs once, advancing the smaller key and emitting on equality](/imgs/blogs/join-algorithms-nested-loop-hash-merge-6.webp)

The figure shows the two sorted tapes and the merge rule. Cursor R sits on a key, cursor S sits on a key, and the rule is mechanical: smaller key advances, equal keys emit and both advance. Both cursors only ever move forward, so each input is read exactly once. The merge cost is $O(M + N)$ — one linear pass — the same as an in-memory hash join, and *cheaper* than a spilling hash join's $3 \times (M + N)$.

The catch is the precondition: **both inputs must be sorted on the join key.** Merge join comes in two economic flavors depending on where that sort comes from:

1. **The sort is free.** Both sides arrive sorted because the join key is the leading column of an index on each table, and the planner reads them in index order. In this case merge join is spectacular: a linear sweep over data you were going to read anyway, with no sort cost and no hash table. This is its sweet spot.

2. **The sort is paid.** One or both inputs are unsorted, so the planner must add explicit `Sort` nodes before the merge. Now the cost includes the external merge-sort of each side. Pavlo's lecture gives the sort costs as $2M \times (1 + \lceil \log_{B-1} \lceil M/B \rceil \rceil)$ for $R$ and similarly for $S$, plus the $M + N$ merge. On the running example this lands around **0.75 seconds** — slower than the in-memory hash join's 0.45s, because of the sort overhead, but still vastly better than the nested loop.

The decision between merge and hash, then, often comes down to: *is the sort already available?* If yes, merge join is the cheapest option on the board. If no, hash join usually wins because it skips sorting entirely. Pavlo's lecture concludes exactly this way: "Hash joins are almost always better than sort-based join algorithms, but there are cases in which sorting-based joins would be preferred. This includes queries on non-uniform data, when the data is already sorted on the join key, or when the result needs to be sorted."

That last point — "when the result needs to be sorted" — is the underappreciated advantage of merge join, shown in the green box of the figure: **merge join produces sorted output.** If your query is `... ORDER BY a.k` or feeds another merge join above it, the merge join gives you the sort ordering for free as a byproduct, whereas a hash join produces output in hash-bucket order (effectively random) and would need a separate sort on top. A good optimizer prices that downstream benefit into the choice.

Let us see a free merge join in Postgres. Both `customers.id` and `orders.id` are primary keys (hence B-tree indexes in sorted order), so a join on those keys can merge without sorting:

```sql
EXPLAIN (ANALYZE)
SELECT c.id, o.id
FROM customers c
JOIN orders o ON o.id = c.id      -- both sides indexed and sorted on the key
ORDER BY c.id;
```

```
Merge Join  (cost=... rows=100000 width=16) (actual time=0.05..210 rows=100000 loops=1)
  Merge Cond: (c.id = o.id)
  ->  Index Only Scan using customers_pkey on customers c  (actual rows=100000 loops=1)
  ->  Index Scan using orders_pkey on orders o  (actual rows=100001 loops=1)
Planning Time: 0.2 ms
Execution Time: 240 ms
```

No `Sort` nodes anywhere. Both sides are read in index order — already sorted on the join key — and the merge sweeps them once. The `ORDER BY c.id` is satisfied for free because the merge output is already in `id` order. This is merge join at its best: zero sort cost, sorted result, linear sweep.

### The worst case: all-equal keys

Merge join has one degenerate failure mode, noted in the cost box of the figure. If the join key has very low cardinality — for instance a boolean column where half the rows are `true` — then the equal-key runs on both sides are enormous, and the inner double loop that cross-products them turns the merge into something close to a nested loop. Pavlo's lecture notes the formal worst case is $M \times N$ "if the join attribute for all the tuples in both tables contains the same value," though "most of the time the keys are mostly unique, so the merge cost can be assumed to be approximately $M + N$." The practical takeaway: merge join is for high-cardinality join keys. Low-cardinality keys with many duplicates are hash join's territory.

### Second-order optimization: clustering for free merge joins

The non-obvious win: if you have two large tables you frequently join on a high-cardinality key, physically clustering both on that key (in Postgres, `CLUSTER table USING index`, or designing the primary key to be the join key) makes index-order scans cheap and sequential, which makes the free merge join the planner's natural choice. This is the relational analog of co-partitioning in distributed systems — and it is exactly how columnar and analytical engines lay out data to make merge joins the default for star-schema fact-to-dimension joins on sorted keys.

## 4. The connection to batch processing: it is the same three algorithms at scale

It is worth stepping back to notice that these are not database-specific tricks. They are *the* general-purpose join algorithms, and they reappear identically the moment you process data at scale outside a database. Martin Kleppmann's *Designing Data-Intensive Applications*, Chapter 3, develops exactly the same taxonomy for batch processing over MapReduce-style systems, and seeing the correspondence makes the OLTP versions click.

In Kleppmann's framing, a MapReduce job that joins two datasets has the same three choices, just renamed for the distributed setting:

- **Sort-merge join (reduce-side join).** The mappers tag each record with its join key, the framework's *shuffle* sorts all records by key and routes equal keys to the same reducer, and the reducer then sees the records for each key adjacent and sorted — at which point it does exactly the two-cursor merge we just described. The expensive part is the shuffle (sorting and copying data across the network), which is the distributed equivalent of the explicit `Sort` nodes a single-node merge join pays for when its inputs are not pre-sorted. The merge itself is the same linear sweep.

- **Broadcast hash join.** When one dataset is small enough, every mapper loads it into an in-memory hash table (or a read-only local index) and probes it while streaming the large dataset. This is the *index/hash nested loop* idea at cluster scale: broadcast the small "build" side to every node so each node can do a local, no-shuffle join. It is precisely the "build the small side, probe the big side" strategy of a single-node hash join, distributed by replicating the build side instead of partitioning it.

- **Partitioned hash join (map-side join).** When both datasets are already partitioned by the same key with the same hash function, the join can happen partition-by-partition with no shuffle at all — each node hash-joins its own slice. This is the grace hash join's partitioning insight applied across machines: co-partition both inputs so matching keys are guaranteed co-located, then join each partition independently. Kleppmann notes that if the inputs are not only co-partitioned but also co-sorted, you can even do a *map-side merge join* — the fully optimized case where neither shuffle nor sort is needed, the distributed twin of the "free" merge join over two clustered indexes.

The `GROUP BY` / reduce-side aggregation in MapReduce is the same shuffle-then-process pattern: partition by the grouping key, sort, and let the reducer fold each group — structurally identical to how a single node does a `HashAggregate` (hash the group key) or a `GroupAggregate` (sort then fold, the merge-join analog for aggregation). The point is that **physical joins are not an implementation detail of one database; they are the small handful of ways anyone, anywhere, can match two sets of records by key.** Whether the "memory budget" is `work_mem` on one box or the RAM of a worker node in a cluster, whether the "spill" is a temp file or a shuffle to disk across the network, the algorithms — and their failure modes — are the same. Once you can read a Postgres join plan, you can read a Spark physical plan, because they are drawing from the same three ideas.

## 5. How the planner actually chooses

**The senior rule of thumb: the planner is a cost minimizer fed estimates; it will always pick the plan with the lowest *estimated* cost, which is only as good as the estimates.**

We now have three operators with three cost curves. The optimizer's job is to estimate, for each candidate plan, the total cost, and pick the cheapest. The estimate depends on a handful of inputs, and understanding them is the difference between fighting the planner and steering it. As [Cybertec's overview of join strategies](https://www.cybertec-postgresql.com/en/join-strategies-and-performance-in-postgresql/) summarizes, Postgres weighs "the estimated sizes of the two sides, whether either side arrives already sorted on the join key, the type of join (inner versus semi, anti, or outer), which operators are mergejoinable or hashjoinable, and whether a hash table will fit inside `work_mem`."

The inputs to the decision:

| Factor | Pushes toward nested loop | Pushes toward hash join | Pushes toward merge join |
| --- | --- | --- | --- |
| **Outer-side row count** | Small (few probes) | Large | Large |
| **Inner-side index on join key** | Present (cheap probe) | Irrelevant | Helps (gives sort order) |
| **Input sort order** | Irrelevant | Irrelevant | Already sorted = free |
| **`work_mem` budget** | Irrelevant | Must hold build side | Must hold sort runs |
| **Join key cardinality** | Any | High or low | High (duplicates hurt) |
| **Output ordering needed** | No help | No help | Sorted output for free |
| **Join type** | Any predicate, incl. ranges | Equi-join only | Equi-join only |

The crucial variable, threaded through every row, is the **row estimate** — how many rows the planner thinks each side will produce after filters. That estimate moves the *crossover point* where one algorithm becomes cheaper than another.

![Cost crossover as inputs grow: nested loop is cheapest at a handful of rows but explodes by the millions while hash and merge stay flat](/imgs/blogs/join-algorithms-nested-loop-hash-merge-7.webp)

The figure is the cost model made physical, using Pavlo's example numbers. Read across each row. At ~10 rows the nested loop is best — sub-millisecond, and building a hash table would be pure overhead. By ~1k rows the algorithms are within a small factor of each other. By ~100k rows the nested loop is at ~50 seconds while hash and merge sit under a second. By ~10M the nested loop is at ~1.4 hours and the others are a couple of seconds. The colors flip from green (nested loop wins) to red (nested loop catastrophic) as you read left to right. **The planner's entire job is to figure out which column of this table it is in — and it does that from the row estimate.**

This is why one wrong estimate becomes a multi-minute query. If the planner estimates a side at 12 rows, it reads off the leftmost column and picks the nested loop — correctly, *for 12 rows*. If the side actually produces 1.4 million rows, the query is now executing the leftmost-column choice (nested loop) against rightmost-column data (millions). You are running the 1.4-hour algorithm. The cost model was not wrong; it was lied to. As the survey of [PostgreSQL join optimization by Philip McClarence](https://medium.com/@philmcc/postgresql-join-optimization-nested-loop-hash-and-merge-c87b86373908) puts it bluntly: "The single biggest cause of wrong-strategy joins is bad row estimates. If the planner thinks a side will produce 15 rows and it actually produces 150,000, it might pick a Nested Loop (optimal for 15) when a Hash Join (optimal for 150,000) would be better."

### Why estimates go wrong

Row estimates fail for a few recurring reasons, and knowing them tells you where to look:

- **Stale statistics.** Postgres estimates selectivity from sampled statistics gathered by `ANALYZE` (and autovacuum's auto-analyze). After a bulk load or a big delete, the stats describe the old data. A table that just grew 100x but has not been re-analyzed will be estimated at its old size. Fix: `ANALYZE` after bulk operations; do not rely on autovacuum to catch up immediately.

- **Correlated columns.** The planner assumes columns are independent by default. `WHERE city = 'Seattle' AND state = 'WA'` is estimated as $P(\text{city}) \times P(\text{state})$, but those columns are perfectly correlated, so the real selectivity is just $P(\text{city})$. The independence assumption massively *under*estimates the count. Fix in Postgres: `CREATE STATISTICS` extended statistics objects to teach the planner the correlation.

- **Skewed values.** If 90% of `orders.status` is `'completed'`, the planner's histogram knows this for a constant, but for a join or a parameter it may assume uniform distribution and badly misjudge.

- **Estimates through multiple joins.** Each join multiplies estimation error. A small error on a base table compounds as it flows up through three or four joins, so deep plans are where estimates rot worst. The [Postgres mailing list](https://www.postgresql.org/message-id/557A2555.7090807@gmail.com) is full of reports of exactly this: "Postgres chooses nested loop over hash join... wrong number of rows estimated."

### Diagnosing the choice with enable_* GUCs

Postgres exposes a beautiful diagnostic trick: you can *disable* a join algorithm for one session and see what the planner does instead. These are the `enable_*` GUCs:

```sql
-- Force the planner away from each strategy, one at a time, to compare.
SET enable_nestloop = off;    -- ban nested loop
SET enable_hashjoin = off;    -- ban hash join
SET enable_mergejoin = off;   -- ban merge join
```

They do not literally forbid the algorithm — they add a huge cost penalty (`1e10`) so the planner avoids it if any alternative exists. The workflow is: run `EXPLAIN ANALYZE` to see the chosen plan and its *actual* time; then `SET enable_nestloop = off` and run it again. If the alternative plan is dramatically faster, you have proven the planner made the wrong choice and you know *which* choice — now you can fix the underlying estimate. This is exactly how the production stories get diagnosed; one [Cybertec writeup on exploding nested loops](https://www.cybertec-postgresql.com/en/exploding-runtime-how-nested-loops-can-destroy-speed/) documents queries dropping from 37 seconds to 1.5 seconds, and from 605 seconds to 42 seconds, simply by setting `enable_nestloop = off` and forcing a hash join — proof that the nested loop, not the data, was the problem.

Always set these back (`RESET enable_nestloop;`) — they are diagnostic tools, not a fix. The real fix is to correct the estimate so the planner makes the right call on its own.

## 6. Join ordering: the other half of the problem

**The senior rule of thumb: with N tables, the planner is not just choosing algorithms, it is searching the space of join *orders* — and that search is exponential, so it has hard cutoffs you can hit.**

So far we have considered joining two tables. Real queries join five, ten, twenty. Now there are two intertwined decisions: which algorithm for each join, *and in what order to join the tables*. The order matters enormously because the size of an intermediate result depends on which tables you have joined so far. Join the two tables that produce a tiny intermediate result first, and every subsequent join is cheap. Join the two that produce a huge cross-product first, and you carry that bulk through the rest of the plan.

The classic approach, from IBM's System R optimizer, is **dynamic programming over left-deep trees.** A left-deep tree is a join order where the result of each join is fed as the *outer* (left) input to the next join, and each *inner* (right) input is always a base table:

![Left-deep join order search: dynamic programming keeps the cheapest sub-plan per relation set, with the inner side always a base table](/imgs/blogs/join-algorithms-nested-loop-hash-merge-8.webp)

The figure shows a left-deep plan for `((A ⋈ B) ⋈ C) ⋈ D`. Read bottom-up: join the tiny driver table A with B (a merge join here), feed that result as the outer to a join with C (hash join), feed *that* as the outer to a join with D (an index nested loop probing D's index). Each inner side is a base table that can use its own index. The "bushy" alternative — joining `(A ⋈ B)` and `(C ⋈ D)` separately and then joining the two intermediate results — is shown rejected on the side; bushy trees enable more parallelism but explode the search space and prevent the inner side from being an indexable base table, so most optimizers restrict the search to left-deep plans by default.

The dynamic programming works by building up the cheapest plan for every *subset* of tables. First find the cheapest way to scan each single table; then the cheapest way to join each pair; then each triple (reusing the cached cheapest pairs); and so on up to the full set. By memoizing the cheapest sub-plan for each relation set, it avoids re-deriving them — but the number of subsets is still $2^N$, and within each subset there are multiple join orders to consider. The search is exponential in the number of tables.

### The collapse limits

Because the search is exponential, Postgres caps how hard it tries, with two knobs that confuse everyone:

- **`from_collapse_limit`** (default 8) controls how subqueries in the `FROM` clause are flattened into the parent query for joint optimization. Below the limit, the planner merges them and considers reorderings across the boundary; above it, it leaves the subquery as a separate planning unit.

- **`join_collapse_limit`** (default 8) controls explicit `JOIN` syntax similarly: how many `JOIN`-clause items the planner will flatten into one list it can freely reorder. As the [Google Cloud writeup on join_collapse_limit](https://medium.com/google-cloud/how-join-collapse-limit-and-geqo-threshold-can-help-to-optimize-complex-queries-involving-joins-fc7be9fec5c3) explains: "If a query joins many tables, the optimizer will only consider all possible combinations for the first eight tables, and joins the remaining tables as written in the statement."

That last clause is a sharp edge: **above the collapse limit, the planner stops reordering and uses the order you wrote.** This is occasionally a feature — you can set `join_collapse_limit = 1` to force the planner to honor your exact join order, an "optimization barrier" trick covered in [pganalyze's guide to forcing join order](https://pganalyze.com/blog/5mins-postgres-forcing-join-order). But it is more often a trap: a 12-table query that joins fine in one written order is suddenly terrible in another, and the difference is that the planner gave up reordering past table 8 and your hand-written order happened to be bad.

### GEQO: when there are too many tables to search exhaustively

When the number of tables to join reaches **`geqo_threshold`** (default 12), Postgres switches from exhaustive dynamic programming to the **Genetic Query Optimizer (GEQO)** — a heuristic, randomized search that treats join orders as "genomes" and evolves toward a low-cost plan. As the Postgres docs describe it, GEQO "does query planning using heuristic searching... [which] reduces planning time for complex queries, at the cost of producing plans that are sometimes inferior." The two consequences that bite in production:

1. **GEQO is non-deterministic.** The same query can get different plans on different runs, because the genetic search is randomized. A query that is usually fine can occasionally get a bad plan, producing maddening intermittent slowness.

2. **You can push the threshold up.** If your box has CPU to spare at plan time, raising `geqo_threshold` (and `join_collapse_limit` to match) makes the planner use exhaustive search for more tables, trading planning time for plan quality and determinism. The Google Cloud guidance reports good results with `geqo_threshold` between 20–24 and `join_collapse_limit` between 16–18 on capable hardware. The general principle, per [pganalyze's tuning writeup](https://pganalyze.com/blog/5mins-postgres-tuning-deterministic-query-planner-extended-statistics-join-collapse-limits): "If you can afford deterministic planning, it's probably better," so raising `geqo_threshold` to keep the genetic algorithm from triggering is usually the right call for OLTP.

### Second-order optimization: the order amplifies estimate errors

The non-obvious interaction: join ordering and row estimation are not independent failures — they multiply. A bad estimate on a base table not only picks the wrong algorithm for that join, it picks the wrong *order* for the whole plan, because the planner orders joins to keep intermediate results small, and it sizes those intermediate results from the same estimates. So one stale-statistics table can poison both the algorithm choice *and* the join order, and the two errors compound. This is why "just run `ANALYZE`" fixes so many mysterious slowdowns at once: it repairs the input that both decisions depend on.

## 7. Semi-joins and anti-joins: EXISTS, NOT EXISTS, IN

**The senior rule of thumb: `EXISTS`/`IN` become semi-joins and `NOT EXISTS` becomes an anti-join — and getting `NOT IN` wrong with nullable columns is a classic correctness-and-performance footgun.**

Not every join wants the matched rows concatenated. Sometimes you only want to know *whether* a match exists. These are **semi-joins** and **anti-joins**, and they are not separate algorithms — they are *modes* of the three operators we already have. Postgres can do a hash semi-join, a merge anti-join, a nested loop semi-join, and so on. As [Cybertec notes](https://www.cybertec-postgresql.com/en/join-strategies-and-performance-in-postgresql/), semi/anti are "sub join types to joining methods such as hash, merge, and nested loop."

A **semi-join** returns each outer row *at most once* if it has any match on the inner side, regardless of how many matches there are. This is what `EXISTS` and `IN` compile to:

```sql
-- Customers who have placed at least one order. Semi-join.
EXPLAIN (ANALYZE)
SELECT c.id
FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

```
Hash Semi Join  (cost=... rows=100000 width=8) (actual time=... rows=99998 loops=1)
  Hash Cond: (c.id = o.customer_id)
  ->  Seq Scan on customers c  (actual rows=100000 loops=1)
  ->  Hash  (actual rows=10000000 loops=1)
        ->  Seq Scan on orders o  (actual rows=10000000 loops=1)
Execution Time: ...
```

The key word is `Semi` in `Hash Semi Join`. The semantics matter for performance: a semi-join can **stop probing as soon as it finds the first match** for an outer row, because it does not need to count or emit duplicates. That early-exit is why a well-planned `EXISTS` is often faster than a `JOIN` followed by `DISTINCT` — the semi-join builds the dedup into the operator instead of materializing all matches and then deduplicating.

An **anti-join** returns each outer row that has *no* match on the inner side. This is what `NOT EXISTS` compiles to:

```sql
-- Customers who have never placed an order. Anti-join.
EXPLAIN (ANALYZE)
SELECT c.id
FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

```
Hash Anti Join  (cost=... rows=2 width=8) (actual time=... rows=2 loops=1)
  Hash Cond: (c.id = o.customer_id)
  ->  Seq Scan on customers c  (actual rows=100000 loops=1)
  ->  Hash  (actual rows=10000000 loops=1)
        ->  Seq Scan on orders o  (actual rows=10000000 loops=1)
```

`Hash Anti Join`. The anti-join emits an outer row precisely when the probe finds nothing — the logical complement of the semi-join. As [the SQL semi/anti-join primer](https://medium.com/@ritusantra/anti-join-semi-join-in-sql-077582f67ea8) frames it: a semi-join "returns a single value for all the matching records," while an anti-join "returns rows when no matching records are found in the second table."

### The NOT IN footgun

There is a notorious trap with `NOT IN` and nullable columns that is both a correctness bug and a performance cliff. If the subquery's column can contain `NULL`, then `x NOT IN (SELECT col FROM ...)` cannot be optimized into a clean anti-join, because SQL's three-valued logic means `x NOT IN (1, 2, NULL)` evaluates to `UNKNOWN` (not `TRUE`) for every `x` — the presence of a single `NULL` makes the whole predicate return no rows. The planner, forced to honor this, often falls back to a much slower plan and the query both runs slowly *and* returns wrong (empty) results. The fix is to use `NOT EXISTS` instead, which has clean `NULL` semantics and reliably compiles to an anti-join. **Default to `NOT EXISTS` over `NOT IN` for subqueries; it is correct and it plans better.** This is one of those rules that separates engineers who have been burned from those who are about to be.

## 8. Reading the plan and forcing a better join

Everything above converges on one practical skill: open the `EXPLAIN ANALYZE`, see which join you got and whether it was a good choice, and steer the planner toward a better one. Here is the playbook, in the order you should apply it.

**Step 1 — Read the operator and the `loops`.** The join node name tells you the algorithm: `Nested Loop`, `Hash Join`, or `Merge Join` (plus `Semi`/`Anti`/`Left` modifiers). For a nested loop, the inner node's `loops=N` is the multiplier — a large `loops` on an expensive inner node is your problem. For a hash join, the `Hash` node's `Batches` tells you if it spilled. For a merge join, look for `Sort` nodes feeding it (paid sort) versus index scans (free sort).

**Step 2 — Compare estimated rows to actual rows.** This is the single highest-value diagnostic. Every node shows `rows=E` (estimate) in the cost section and `rows=A` (actual) in the timing section. When `E` and `A` diverge by more than ~10x on any node, you have found the lie that is poisoning the plan. A nested loop chosen on `rows=5` that actually produced `rows=500000` is the canonical disaster.

```sql
-- The diagnostic that finds most bad plans: estimate vs actual.
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT ...;
-- Scan for any node where (actual rows) / (estimated rows) > 10 or < 0.1.
```

**Step 3 — Fix the estimate first.** Before reaching for any override, repair the input:

```sql
ANALYZE orders;                              -- refresh stale stats after a bulk load
ALTER TABLE orders ALTER COLUMN customer_id SET STATISTICS 1000;  -- finer histogram
CREATE STATISTICS orders_corr (dependencies) ON customer_id, status FROM orders;  -- correlation
ANALYZE orders;
```

Eighty percent of bad-join incidents are fixed at this step, because the algorithm and the order are both downstream of the estimate. As one [pganalyze guide](https://pganalyze.com/docs/query-advisor/insights/inefficient-nested-loops) on inefficient nested loops notes, stale or missing statistics are the root cause far more often than the planner being "dumb."

**Step 4 — Give the planner a better option.** Sometimes the estimate is fine and the planner genuinely lacks a good plan. Add the index that makes an index nested loop cheap, or that supplies sort order for a free merge join:

```sql
-- Make the inner-side index nested loop cheap:
CREATE INDEX ON orders (customer_id) INCLUDE (total);  -- covering index, index-only probe
```

**Step 5 — Raise `work_mem` if a hash join is spilling.** If `Batches > 1` and the spill is your cost, raise `work_mem` for that query (set it locally, not globally):

```sql
SET LOCAL work_mem = '256MB';   -- inside a transaction, for this query only
```

But respect the danger: `work_mem` is allocated *per sort/hash node per connection*. A [guide to Postgres work_mem](https://medium.com/@muhammadalikhan0003/unlock-postgresql-performance-mastering-work-mem-for-faster-sorting-joins-5cf3b2c59df3) gives the cautionary arithmetic: a query with 3 sorts and 2 hash joins can allocate up to 5 × `work_mem`, and 50 concurrent such queries at 64 MB each is 16 GB. Setting it globally to a large value will OOM the box under concurrency. Set it locally for the heavy analytical query that needs it.

**Step 6 — Use `enable_*` to diagnose, then convert the finding into a real fix.** As covered above, `SET enable_nestloop = off` proves whether the nested loop is the culprit. If forcing it off makes the query 25x faster, you have your answer — but the *fix* is to repair the estimate (Step 3) or add the index (Step 4) so the planner reaches the fast plan on its own, not to ship `enable_nestloop = off` in your application (which would distort every other join in the session).

**Step 7 — As a last resort, use planner hints or query restructuring.** Postgres core has no hint syntax (by design), but `pg_hint_plan` exists, and you can restructure: a CTE with `MATERIALIZE`, a `join_collapse_limit = 1` optimization barrier to pin join order, or rewriting `NOT IN` to `NOT EXISTS`. MySQL does have optimizer hints and `SET optimizer_switch='hash_join=off'` to fall back to block nested loop. These are sharp tools; reach for them only after Steps 1–6.

The decision matrix below compresses the whole article into one picture — when each operator is the right answer, by workload shape.

![Which join wins by workload shape: the best operator changes with input size, indexes, and sort order, not table size alone](/imgs/blogs/join-algorithms-nested-loop-hash-merge-9.webp)

Read down each column to see how a given operator fares across workload shapes. Nested loop is the clear winner for a tiny indexed-inner join and an outright trap for two large unindexed inputs. Hash join is the workhorse for large unsorted inputs and overkill for tiny ones. Merge join is unbeatable when both inputs arrive presorted and a waste when it has to pay for the sort. The diagonal of "BEST" cells is the planner's target — and getting it right comes down to feeding the cost model honest estimates.

## Case studies from production

### 1. The 12-row estimate that ran for four minutes

The opening story, fully diagnosed. A dashboard query joined a filtered `events` table to a large `users` table. The filter on `events` was `WHERE campaign_id = $1`, and most campaigns had a few dozen events — so the planner's *average* selectivity estimate said ~12 rows, picked an index nested loop, and probed `users` 12 times. Fast. Then a viral campaign generated 1.4 million events. The query text was identical; the bound parameter pointed at the big campaign. The planner, using the generic estimate baked into the prepared statement's plan, still expected ~12 rows and still chose the nested loop — now probing `users` 1.4 million times. Four minutes. The fix was twofold: `CREATE STATISTICS` to capture that `campaign_id` selectivity was wildly skewed, and switching the prepared statement to `plan_cache_mode = force_custom_plan` so each execution re-planned with the actual parameter value. After the fix, the big-campaign execution re-planned to a hash join and finished in 900 ms. The lesson: generic plans for prepared statements assume average selectivity, and a skewed parameter turns "average" into a lie.

### 2. The hash join that OOM-killed the primary

An analytics team raised `work_mem` to 512 MB globally to speed up their reporting queries — which worked beautifully in isolation. Then a Monday-morning traffic spike ran 40 of those reports concurrently, each with two hash joins and a sort. Forty connections times three memory-hungry nodes times 512 MB is well past 60 GB on a 64 GB box. The OOM killer took the Postgres primary, triggering a failover. The root cause was not the hash join algorithm — it was the per-node, per-connection nature of `work_mem` combined with a global setting. The fix: drop the global `work_mem` back to 16 MB and use `SET LOCAL work_mem = '512MB'` inside the specific reporting transactions, plus a separate connection pool with a low max-connection cap for the analytics workload. The hash joins still got their memory; the box stopped dying. The general rule: never set `work_mem` globally to a value you cannot afford multiplied by your max connections times your typical node count per query.

### 3. The merge join that secretly paid for two sorts

A query joining two 50-million-row tables on a non-indexed key showed `Merge Join` in the plan, and the team assumed merge join meant "fast linear sweep." It ran for 90 seconds. The `EXPLAIN ANALYZE` revealed two giant `Sort` nodes feeding the merge, each spilling to disk because the sort exceeded `work_mem` — the merge itself was cheap, but the team was paying for two full external sorts of 50M rows. The planner had picked merge over hash because its (wrong) estimate said the hash build side would not fit in `work_mem` and spill anyway, making the sort competitive. Correcting `work_mem` for the query let the hash join fit in memory, and the plan flipped to a single-pass `Hash Join` that finished in 11 seconds. The lesson: `Merge Join` in the plan is not automatically good — check whether it is fed by free index scans or by paid `Sort` nodes. The former is the sweet spot; the latter is often a sign the planner couldn't afford a hash join.

### 4. The N+1 that no database tuning could fix

A service endpoint loaded 200 "projects," then for each project issued a separate query for its "tasks" — 201 queries, each fast on its own, but the endpoint took 3 seconds because of 200 sequential round-trips at ~15 ms of network-plus-planning each. The team spent a day adding indexes and tuning the per-query plans; nothing moved the needle, because each query was already optimal — the cost was in the *count* of round-trips, an application-level nested loop. The fix was a single `WHERE project_id = ANY($1)` batch query that pulled all tasks in one shot, which the database executed as one hash join internally in 40 ms. The lesson from [Use The Index, Luke!](https://use-the-index-luke.com/sql/join/nested-loops-join-n1-problem): the N+1 problem *is* a nested loop, just executed across a network where each inner probe costs a network round-trip instead of an index traversal. The only fix is to let the database do the loop in-process.

### 5. The `NOT IN` that returned zero rows after a NULL crept in

A nightly job used `WHERE id NOT IN (SELECT parent_id FROM children)` to find orphaned records. It worked for a year. Then a schema change made `children.parent_id` nullable, and one NULL row appeared. Overnight the job started reporting "0 orphans" — silently, with no error. The cause was SQL's three-valued logic: `id NOT IN (..., NULL)` evaluates to `UNKNOWN` for every `id`, so the `WHERE` matched nothing. Worse, before that, the planner had been unable to use a clean anti-join for the `NOT IN` with a nullable column, so the query had also been quietly slow. Rewriting to `WHERE NOT EXISTS (SELECT 1 FROM children c WHERE c.parent_id = t.id)` fixed both: correct NULL semantics and a fast `Hash Anti Join`. The lesson: `NOT IN` with a nullable subquery column is a correctness bug waiting to happen *and* a worse plan; default to `NOT EXISTS`.

### 6. The 12-table report that GEQO planned differently every morning

A finance report joined 14 tables. Most mornings it ran in 8 seconds; roughly once a week it ran for 6 minutes. The query and data were unchanged. The cause was GEQO: at 14 tables, past the default `geqo_threshold` of 12, Postgres used its randomized genetic search, which occasionally landed on a bad join order. Because the search is non-deterministic, the bad plan appeared unpredictably. The fix was to raise `geqo_threshold` to 16 and `join_collapse_limit` to 16, forcing the deterministic dynamic-programming optimizer to handle all 14 tables. Planning time rose from ~5 ms to ~120 ms — irrelevant for a report — and the runtime became a consistent 8 seconds every day. The lesson, echoing the [Google Cloud and pganalyze guidance](https://pganalyze.com/blog/5mins-postgres-tuning-deterministic-query-planner-extended-statistics-join-collapse-limits): for important complex queries where you can afford the planning time, push the thresholds up to get deterministic, exhaustive planning.

### 7. The MySQL upgrade that made a query 30x faster for free

A team upgrading from MySQL 5.7 to 8.0 noticed a reporting query that had always been slow suddenly ran in a fraction of the time, with no changes. The reason: before MySQL 8.0.18, equi-joins with no usable index fell back to the block nested loop algorithm; in 8.0.18+ they use a hash join. The query joined two large tables on a column with no index, which under 5.7 was a block nested loop re-scanning the inner table per outer block. Under 8.0 the same join became a hash join scanning each input once. As the [MySQL hash join blog](https://dev.mysql.com/blog-archive/hash-join-in-mysql-8/) reports, hash join "clearly outperforms block-nested loop in all queries where it is used... due to the fact that hash join scans each input only once." The lesson: the *available* algorithms differ by engine and version. The same SQL can get a fundamentally different (and better) physical plan after an upgrade — and knowing which algorithms your engine supports tells you which queries will benefit.

### 8. The correlated columns that under-estimated a join by 1000x

A query filtered on `WHERE country = 'US' AND currency = 'USD'` then joined to a transactions table. The planner estimated the filtered set at a handful of rows — it multiplied the selectivity of `country = 'US'` by the selectivity of `currency = 'USD'` as if independent — and picked an index nested loop. But country and currency are near-perfectly correlated; essentially every US row has USD currency, so the real count was 1000x the estimate. The nested loop probed the transactions table tens of thousands of times instead of the dozen it expected. The fix was `CREATE STATISTICS (dependencies, ndistinct) ON country, currency`, which taught the planner the correlation; the estimate corrected, and the plan flipped to a hash join that ran in a tenth of the time. The lesson: the independence assumption is the planner's most dangerous default, and correlated filter columns feeding a join are a classic source of catastrophic under-estimates — exactly the "1000:1 mis-estimation" pattern reported across the Postgres mailing lists.

### 9. The spill nobody noticed until the disk filled

A batch ETL hash-joined two large tables nightly. It always finished, so nobody looked at the plan. Then one night it failed with "could not write to temporary file: No space left on device." The hash join had been spilling to temp files (`Batches: 64`) every single night — it just usually had enough scratch disk. As the data grew, the spill grew, until one night the partitioned batches exceeded the available temp space. The `EXPLAIN ANALYZE` had been screaming `Batches: 64` the whole time; nobody read it. The fix was to raise `work_mem` for that ETL job so the build side fit in memory (`Batches: 1`), eliminating the spill entirely and incidentally cutting runtime by 40% since it no longer paid the 3x grace-hash I/O. The lesson: `Batches > 1` is not just a performance signal, it is a *capacity* risk — a spilling hash join consumes temp disk proportional to the build side, and that can fill the disk as data grows.

### 10. The distributed join that shuffled a terabyte

On a Citus-sharded cluster, a join between a large fact table and a dimension table ran for 20 minutes because the two tables were sharded on *different* keys. To join them, Citus had to repartition (shuffle) one of them across the network so matching keys ended up on the same node — moving roughly a terabyte over the wire, the distributed equivalent of a spill. The fix was to co-locate the tables by sharding both on the join key, so each node could do a local hash join on its own shard with no shuffle — exactly Kleppmann's partitioned hash join / map-side join. Runtime dropped to 90 seconds. The lesson connects the whole article: the three join algorithms and their costs apply identically in a distributed system, except the "memory budget" is a node's RAM and the "spill" is a network shuffle. Co-partitioning is the distributed version of clustering both inputs for a free local join, and it is the single highest-leverage decision in a sharded schema.

## When to reach for each join — and when not to

The three algorithms are not competitors so much as specialists. The optimizer's whole job is to match the specialist to the workload, and your job — when it gets it wrong — is to recognize the mismatch and fix the input that drove the bad choice.

**Reach for a nested loop when:**

- The outer side is small (tens to low hundreds of rows after filtering) and the inner side has an index on the join key — the textbook index nested loop.
- The join predicate is not a simple equality — ranges, inequalities, function-of-both-rows — because nested loop is the only operator that handles arbitrary predicates.
- You are doing a highly selective OLTP lookup where you want the first rows back immediately; nested loop streams results without a blocking build or sort phase.

**Skip the nested loop when:**

- Both sides are large and unindexed — this is the $O(n \times m)$ trap that turns a query into a space heater.
- The outer-side estimate is uncertain and could be large; the downside of a wrong nested loop is unbounded, while a hash join degrades gracefully.

**Reach for a hash join when:**

- Both inputs are large, joined by equality, and at least the smaller one fits in `work_mem` — the analytical workhorse.
- Neither side is usefully sorted and you do not need sorted output.
- The join key has low cardinality with many duplicates, where merge join's equal-key cross-products would hurt.

**Skip the hash join when:**

- One side is tiny and the other is indexed — a nested loop is cheaper and streams results without building anything.
- The build side is far too large for memory and there is no way to raise `work_mem` safely; a spilling hash join's 3x I/O may lose to a free merge join over sorted inputs.

**Reach for a merge join when:**

- Both inputs already arrive sorted on the join key from index scans — the cheapest option on the board, a linear sweep over data you were reading anyway.
- The query needs the result sorted (an `ORDER BY` on the join key, or feeding another merge join), so the sort comes for free as output.
- The join key is high-cardinality (mostly unique), avoiding the duplicate-cross-product worst case.

**Skip the merge join when:**

- Neither input is sorted and you would pay for two full sorts — a hash join that fits in memory almost always wins.
- The join key has heavy duplicates, which degrades the merge toward a nested loop.

And above all: **before forcing any of this, look at estimated rows versus actual rows.** The algorithms are simple and the planner is good at choosing among them — when it chooses wrong, it is almost always because the cost model was fed a bad estimate. Run `ANALYZE`, teach it about correlated columns with `CREATE STATISTICS`, give it the index it needs, and the right join usually falls out on its own. The `enable_*` GUCs and the per-query `work_mem` overrides are for diagnosis and for the rare genuine planner miss — not for papering over a statistics problem that will resurface on the next query you have not pinned. The deepest skill in database performance is not memorizing which join is fastest; it is reading the plan, finding the lie in the estimate, and fixing the input so the optimizer can do its job.

## Further reading

- [CMU 15-445 Lecture 12: Join Algorithms](https://15445.courses.cs.cmu.edu/fall2024/notes/12-joins.pdf) — Andy Pavlo's canonical treatment with the cost models and worked example used throughout this article.
- *Designing Data-Intensive Applications*, Martin Kleppmann, Chapter 3 — the batch-processing view of sort-merge, broadcast hash, and partitioned hash joins that mirrors the OLTP operators.
- [MySQL 8 hash join](https://dev.mysql.com/blog-archive/hash-join-in-mysql-8/) — the build/probe and grace-hash spill mechanics, plus the block-nested-loop history.
- [PostgreSQL join strategies and performance](https://www.cybertec-postgresql.com/en/join-strategies-and-performance-in-postgresql/) (Cybertec) — how the Postgres planner chooses, including semi/anti joins.
- [Use The Index, Luke! — joins](https://use-the-index-luke.com/sql/join/nested-loops-join-n1-problem) — the indexing-for-joins perspective and the N+1 connection.
- [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) and [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) — the sibling posts this one builds on.

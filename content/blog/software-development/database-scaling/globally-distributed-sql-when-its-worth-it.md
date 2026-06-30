---
title: "Globally Distributed SQL: When It's Worth the Latency Tax"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Spanner, CockroachDB, and YugabyteDB give you global ACID transactions and SQL with no manual sharding — but every write pays a consensus round-trip; here is how to know when that tax beats tunable consistency, and when it doesn't."
tags: ["distributed-sql", "spanner", "cockroachdb", "yugabytedb", "consensus", "geo-partitioning", "database-scaling", "consistency", "multi-region", "latency", "newsql", "system-design"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 29
---

Every few months a team shows me an architecture diagram with five regions, a globally distributed SQL database in the middle, and a write path that, on paper, gives them serializable transactions everywhere. Then I ask the question that deflates the room: "What's your commit latency from Singapore?" The answer, when they've measured it, is usually somewhere north of 200 milliseconds — for a single `INSERT`. The database is doing exactly what it promised. It is just that the promise has a price, the price is the speed of light, and nobody read that line of the contract.

Globally distributed SQL — Spanner, CockroachDB, YugabyteDB, TiDB, and the rest of the family marketing calls "NewSQL" or "distributed SQL" — is one of the genuinely impressive achievements of the last fifteen years of systems engineering. It gives you horizontal scale, a real SQL surface with secondary indexes and multi-row transactions, and *strong* consistency across regions, all without you hand-rolling a sharding and routing layer. That combination used to be considered impossible. It is not impossible. It is just expensive in a specific, physical way: **every write that needs a quorum of replicas in other regions must pay a cross-region round-trip before it can commit.** There is no clever engineering that removes that round-trip, because it is not an engineering problem. It is geography.

![The distributed-SQL stack: SQL surface, distributed transaction and consensus layer, auto-sharded ranges, replicated storage](/imgs/blogs/globally-distributed-sql-when-its-worth-it-1.webp)

The diagram above is the mental model for this entire post. A distributed SQL database is a layered stack: a familiar SQL surface on top, a distributed-transaction-and-consensus layer beneath it, an auto-sharding layer that splits your tables into ranges and rebalances them, and replicated key-value storage at the bottom. The top three layers exist to hide the bottom one from you — and they do a remarkable job. You write `BEGIN; UPDATE accounts …; COMMIT;` and it just works across three continents. But the amber layer in the middle, the consensus layer, is where the bill comes due. This post is a tour of that bill: what you're paying for, how to read the invoice, how to make it smaller with geo-partitioning, and — the part most architecture diagrams skip — when you shouldn't be shopping in this store at all.

## Why distributed SQL is different from everything that came before

The reason this category exists is that, for decades, you had to choose two of three properties and give up the third. The choices were brutal and the tradeoffs were real.

| You want…                         | Classic relational (Postgres/MySQL) | Manual sharding | NoSQL (Cassandra/Dynamo) | Distributed SQL |
| --------------------------------- | ----------------------------------- | --------------- | ------------------------ | --------------- |
| Horizontal write scale            | No (one primary)                    | Yes (you build it) | Yes                   | Yes (automatic) |
| Real SQL + multi-row transactions | Yes                                 | Per-shard only  | No (or weak)             | Yes             |
| Strong consistency across nodes   | Yes (single node)                   | Per-shard only  | Tunable / eventual       | Yes (global)    |
| No manual shard key / routing     | N/A                                 | No (it's the whole job) | Yes              | Yes             |
| Cheap writes                      | Yes                                 | Yes (local)     | Yes (local)              | **No — consensus tax** |

Look at the last two columns. NoSQL stores like Cassandra and DynamoDB scale writes horizontally and keep them fast, but they buy that by relaxing consistency: by default a write is acknowledged before it has propagated everywhere, and you reconcile conflicts later. Manual sharding (Vitess on MySQL, or hand-rolled Postgres sharding like Instagram's or Notion's) keeps SQL and fast local writes, but the price is that *you* build and operate the routing layer, *you* lose cross-shard transactions and joins, and *you* own the shard key forever — and the shard key is a [one-way door](/blog/software-development/database-scaling/the-database-scaling-decision-tree).

Distributed SQL is the box in the bottom-right: it gives you almost everything, automatically — and the single thing it cannot give you for free is a cheap cross-region write. That is the whole story. Everything else in this post is detail on that one tradeoff.

> Distributed SQL does not break the CAP theorem or repeal physics. It makes a specific, defensible choice — strong consistency over low write latency — and then works extremely hard to shrink the cost of that choice. Your job is to decide whether that's the choice you want.

## How it works, briefly (and where to go deeper)

You don't need to understand the internals to use one of these databases, but you need to understand *one* thing to predict their performance: where the consensus round-trip lives. Here is the compressed version. For the full treatment, the two deep-dives this post leans on are [Spanner's TrueTime and external consistency](/blog/software-development/database/spanner-truetime-and-external-consistency) and [CockroachDB's distributed SQL design](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive).

**Auto-sharding.** Your tables are split into contiguous key-ranges — Spanner and CockroachDB call them *ranges*, Yugabyte calls them *tablets*. A range starts small and **splits** automatically when it grows past a threshold (CockroachDB defaults to splitting ranges around 512 MiB) or gets too hot. The system **rebalances** ranges across nodes to spread load. You never pick a shard key or run a resharding job. This is the part that feels like magic compared to manual sharding, and it genuinely is the big operational win.

**Replication via consensus.** Each range is replicated — typically three or five copies — and the replicas of a single range form a consensus group. Spanner uses **Paxos**; CockroachDB and Yugabyte use **Raft**. One replica is the leader (CockroachDB calls the lease-holding replica the *leaseholder*). A write goes to the leader, which proposes the change to the group; the write is durable and committed once a *quorum* (a majority — 2 of 3, or 3 of 5) has appended it to their logs. That quorum step is the round-trip. If the quorum members are in the same datacenter, it's sub-millisecond. If they're on different continents, it's the speed of light through fiber, plus switching — tens to hundreds of milliseconds.

**Clock coordination.** To get *serializable* (and in Spanner's case, *externally consistent* / linearizable) transactions across independent consensus groups, the system needs a global ordering of commit timestamps. Spanner solves this with **TrueTime**: GPS receivers and atomic clocks in every datacenter give every node a clock with a *bounded* uncertainty interval `[earliest, latest]`. After committing, Spanner deliberately waits out that uncertainty — **commit-wait** — so the commit timestamp is guaranteed to be in the past everywhere before locks release. CockroachDB and Yugabyte refuse to require special hardware, so they use **hybrid logical clocks (HLC)** — a physical clock combined with a logical counter — plus an uncertainty window and read-restarts to achieve serializability without GPS. The deep-dives explain why each choice works; for our purposes, the takeaway is that Spanner pays an *extra* small wait on top of consensus, while HLC-based systems pay it as occasional read retries instead.

## 1. The latency tax: why every write pays a round-trip

**Senior rule of thumb: a write in a distributed SQL database is not "fast" or "slow" — it is exactly as fast as the round-trip to the nearest quorum of its range's replicas.** Internalize that and you can predict every latency number these systems produce.

Here is the mechanism. A client issues a write. It reaches the leaseholder for the relevant range. The leaseholder proposes the change to the other replicas and waits for a majority to acknowledge that they've durably written it to their logs. Only then does the commit return to the client.

![A cross-region write is a consensus round-trip: client to leaseholder, leaseholder fans out to replicas in us-west and eu-west, quorum reached at two of three, then commit acks to the client](/imgs/blogs/globally-distributed-sql-when-its-worth-it-2.webp)

The figure traces one write. The client in `us-east` reaches the leaseholder, also in `us-east`. The leaseholder appends the entry locally (free) and ships it to the two other replicas — one in `us-west` (~60 ms round-trip), one in `eu-west` (~80 ms round-trip). With three replicas, quorum is two: the leaseholder itself plus *one* remote. So the commit blocks until the *faster of the two remote replicas* acknowledges — about 60 ms here. The commit ack to the client then follows. That ~60 ms is the latency tax. It is not CPU, it is not disk, it is not a slow query — it is the round-trip to the second-fastest member of the quorum, and it is paid on **every single write**.

Two consequences fall out of this immediately:

- **Five replicas are slower to write than three.** A quorum of five is three, so a write must reach the *second*-nearest remote replica, not the nearest. More replicas means more durability and more failure tolerance, but a worse write-latency floor. This is why teams sometimes run 5-replica ranges for the data they cannot lose and 3-replica ranges for everything else.
- **Where the leaseholder lives decides read latency, and where the *replicas* live decides write latency.** These are different placement decisions, and conflating them is the single most common source of "why is this so slow" tickets.

Let's make the contrast concrete. Watch the same round of work commit two writes — one whose replicas are co-located, one whose replicas span an ocean:

<figure class="blog-anim">
<svg viewBox="0 0 680 320" role="img" aria-label="Two commits start in the same round: the in-region commit acks after a short local round-trip while the cross-region commit is still travelling to a distant replica and back" style="width:100%;height:auto;max-width:820px">
<title>In-region commit vs cross-region commit paying the consensus round-trip</title>
<style>
.g3-track{fill:none;stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:6 6}
.g3-node{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.g3-lead{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2.5}
.g3-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.g3-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.g3-pkt{fill:var(--accent,#6366f1)}
.g3-ackfast{fill:#2f9e44}
.g3-ackslow{fill:#e8590c}
@keyframes g3-local{0%{transform:translateX(0);opacity:0}6%{opacity:1}22%{transform:translateX(150px);opacity:1}28%{transform:translateX(0);opacity:1}34%,100%{opacity:0}}
@keyframes g3-wan{0%{transform:translateX(0);opacity:0}6%{opacity:1}55%{transform:translateX(430px);opacity:1}95%{transform:translateX(0);opacity:1}100%{opacity:0}}
@keyframes g3-fastack{0%,26%{opacity:0;transform:scale(.6)}32%{opacity:1;transform:scale(1)}90%{opacity:1}100%{opacity:0}}
@keyframes g3-slowack{0%,93%{opacity:0;transform:scale(.6)}98%{opacity:1;transform:scale(1)}100%{opacity:1}}
.g3-mlocal{animation:g3-local 9s ease-in-out infinite}
.g3-mwan{animation:g3-wan 9s ease-in-out infinite}
.g3-fa{animation:g3-fastack 9s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
.g3-sa{animation:g3-slowack 9s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
@media (prefers-reduced-motion:reduce){.g3-mlocal,.g3-mwan{animation:none;opacity:0}.g3-fa,.g3-sa{animation:none;opacity:1;transform:none}}
</style>
<text class="g3-lbl" x="340" y="28">The same round of decoding: two commits, two very different waits</text>
<line class="g3-track" x1="120" y1="110" x2="280" y2="110"/>
<rect class="g3-lead" x="40" y="86" width="80" height="48" rx="8"/>
<text class="g3-lbl" x="80" y="108">leader</text>
<text class="g3-sub" x="80" y="124">us-east</text>
<rect class="g3-node" x="280" y="86" width="100" height="48" rx="8"/>
<text class="g3-lbl" x="330" y="108">replica</text>
<text class="g3-sub" x="330" y="124">us-east ~1ms</text>
<circle class="g3-pkt g3-mlocal" cx="124" cy="110" r="9"/>
<circle class="g3-ackfast g3-fa" cx="80" cy="110" r="11"/>
<text class="g3-sub" x="80" y="160">in-region commit: ack in ~2ms</text>
<line class="g3-track" x1="120" y1="240" x2="560" y2="240"/>
<rect class="g3-lead" x="40" y="216" width="80" height="48" rx="8"/>
<text class="g3-lbl" x="80" y="238">leader</text>
<text class="g3-sub" x="80" y="254">us-east</text>
<rect class="g3-node" x="560" y="216" width="100" height="48" rx="8"/>
<text class="g3-lbl" x="610" y="238">replica</text>
<text class="g3-sub" x="610" y="254">eu-west ~80ms</text>
<circle class="g3-pkt g3-mwan" cx="124" cy="240" r="9"/>
<circle class="g3-ackslow g3-sa" cx="80" cy="240" r="11"/>
<text class="g3-sub" x="80" y="290">cross-region commit: ack only after the WAN round-trip</text>
</svg>
<figcaption>Both writes leave the leader in the same round; the in-region commit acks after a ~2ms local round-trip while the cross-region commit blocks for the full ~160ms WAN round-trip to reach quorum.</figcaption>
</figure>

The top write's quorum is local: the packet hops to a replica in the same region and the ack flies back in milliseconds. The bottom write's quorum requires a replica in `eu-west`: the same logical operation, but now it cannot return until a packet has crossed the Atlantic and come back. Same SQL, same database, same consistency guarantee — two orders of magnitude difference in latency, decided entirely by where the replicas of that range physically sit. **This is the lever you actually control, and most of the rest of this post is about pulling it.**

### Second-order effect: contention amplifies the tax

There's a nastier version of this. If two transactions touch the same row, the second one waits for the first to *commit and release its locks* before it can proceed. In a single-node database that wait is microseconds. In a cross-region distributed SQL database, the lock is held for the *entire* consensus round-trip plus (in Spanner) commit-wait. So a hot row in a multi-region cluster doesn't just make one write slow — it serializes a queue of writers, each waiting a full WAN round-trip behind the previous one. A row updated by 50 concurrent requests can collapse to ~5 writes per second if each commit holds its locks for 200 ms. The fix is never "add hardware"; it's to stop concentrating writes on one key (sharded counters, batching, or a different data model). Contention is where distributed SQL latency goes from "annoying" to "outage."

## 2. Reads can stay fast — if you let them be slightly stale

**Senior rule of thumb: writes pay consensus; reads only pay it if you insist on the absolute latest value.** This is the asymmetry that makes distributed SQL livable. The latency tax is fundamentally a *write* problem, because durability requires a quorum. Reads have more options, because a replica can often answer from data it already has.

![Three read paths fan out from a range: leaseholder read returns the latest write but may cross regions; follower read returns local data about five seconds stale; bounded-staleness read returns local data within a tunable lag target](/imgs/blogs/globally-distributed-sql-when-its-worth-it-8.webp)

The figure lays out the three read paths every mature distributed SQL system offers, and what each one trades:

- **Leaseholder (consistent) reads.** Route to the leaseholder, which holds the lease and knows it has the latest committed value. This is linearizable — you always see the most recent write. But if the leaseholder is in another region, the read pays that cross-region hop. This is the default, and it's why naive multi-region reads are slow.
- **Follower reads.** Read from the *nearest* replica, even if it's not the leaseholder, accepting data that may be slightly stale (CockroachDB's follower reads serve data that's a few seconds old by default — historically tuned around the closed-timestamp interval). The read stays *local*: no cross-region hop. For data that doesn't need to be read-your-own-write fresh — product catalogs, user profiles, most dashboards — this is a massive latency win.
- **Bounded-staleness reads.** A refinement: read locally but with a *tunable* maximum staleness ("give me data no more than 500 ms old"), and the system picks the freshest local replica that satisfies the bound, restarting only if it can't. You dial the freshness/latency tradeoff per query.

The practical pattern in a global app: serve the vast majority of reads as follower or bounded-staleness reads from the user's local region (fast), and reserve consistent leaseholder reads for the handful of operations that genuinely need the latest value (read-after-write on the user's own data, financial balances at point of sale). Done right, a global app can have ~5 ms reads everywhere and only the writes pay the tax.

```sql
-- CockroachDB: a follower read serves from the nearest replica,
-- accepting bounded staleness, so it never crosses a region.
-- AS OF SYSTEM TIME with follower_read_timestamp() picks a timestamp
-- old enough that any local replica can serve it without a leader hop.
SELECT name, price
FROM products
AS OF SYSTEM TIME follower_read_timestamp()
WHERE category = 'books';

-- Bounded staleness: "as fresh as you can, but no older than 10s,
-- and don't bounce to the leaseholder." Great for read-heavy dashboards.
SELECT total
FROM order_summary
AS OF SYSTEM TIME with_max_staleness('10s')
WHERE region = 'eu-west';
```

## 3. Geo-partitioning: pinning data to where it's used

**Senior rule of thumb: the cure for the latency tax is not a faster network — it's making the quorum local.** If a row's three replicas all live in `eu-west`, then a write to that row from an EU user is an *in-region* consensus round-trip — single-digit milliseconds — even though the cluster spans the globe. This is **geo-partitioning** (CockroachDB's `REGIONAL BY ROW`, Yugabyte's tablespaces with placement, Spanner's placement via leader/replica configs), and it is the feature that turns "globally distributed SQL is too slow for our write path" into "globally distributed SQL is fine."

![Before and after geo-partitioning: on the left, replicas spread across three regions so every write crosses an ocean and pays about 80ms WAN round-trip; on the right, all three replicas pinned to the home region so an EU user's writes stay in eu-west and pay about 2ms](/imgs/blogs/globally-distributed-sql-when-its-worth-it-4.webp)

The before/after is the whole pitch. On the left, with no partitioning, a single table's replicas are spread across regions for balance, so *every* write crosses an ocean to reach quorum — the ~80 ms tax on everything. On the right, with `REGIONAL BY ROW`, each row carries a hidden region column, and the database pins that row's replicas (and its leaseholder) to the row's home region. An EU user's row lives in `eu-west`; their writes reach a local quorum in ~2 ms. A US user's row lives in `us-east`; their writes stay there. The cluster is still one logical database with one SQL surface and global transactions — but the *common case*, where a transaction touches only rows from one region, never pays the cross-region tax.

Here is what that looks like in CockroachDB. The key insight is that you declare the regions once at the database level, then mark the table `REGIONAL BY ROW` and let the system manage the hidden `crdb_region` column:

```sql
-- 1. Declare the cluster's regions (done once, at the database level).
ALTER DATABASE shop SET PRIMARY REGION "us-east";
ALTER DATABASE shop ADD REGION "us-west";
ALTER DATABASE shop ADD REGION "eu-west";

-- 2. A REGIONAL BY ROW table: each row's replicas are pinned to that
--    row's home region. The hidden crdb_region column carries the home.
CREATE TABLE customer_orders (
    order_id    UUID NOT NULL DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL,
    -- the region column; defaults to the gateway node's region on INSERT,
    -- so a write issued in eu-west homes the row in eu-west automatically.
    region      crdb_internal_region NOT NULL DEFAULT default_to_database_primary_region(gateway_region())::crdb_internal_region,
    total_cents INT NOT NULL,
    placed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (region, order_id)
) LOCALITY REGIONAL BY ROW AS region;

-- 3. From here on, an EU customer's order is written, replicated, and read
--    entirely within eu-west. No application code changes; the locality is
--    a property of the schema, not the query.
```

Notice what you did *not* write: there is no routing layer, no shard-key hash, no application logic that decides which database a request goes to. The locality is declared in the schema. The database does the placement. That is the operational gift of distributed SQL — geo-partitioning gives you data locality with the ergonomics of a single Postgres table.

### The in-region transaction vs the cross-region transaction

The payoff is easiest to see by putting two transactions side by side. Both are correct, both are serializable, both run on the same global cluster. One is fast because it stays in one region; one is slow because it spans two.

```python
import psycopg2  # CockroachDB speaks the Postgres wire protocol

# Connected to a gateway node in eu-west.
conn = psycopg2.connect("postgresql://app@eu-west.shop.example:26257/shop")
conn.autocommit = False

# ---- FAST: an in-region transaction ---------------------------------------
# Every row this touches is homed in eu-west, so the quorum is local.
# Commit latency ~= one intra-region consensus round-trip (~2-5 ms).
with conn.cursor() as cur:
    cur.execute("""
        INSERT INTO customer_orders (customer_id, region, total_cents)
        VALUES (%s, 'eu-west', %s)
    """, (eu_customer_id, 4999))
    cur.execute("""
        UPDATE inventory SET qty = qty - 1
        WHERE region = 'eu-west' AND sku = %s
    """, (sku,))
conn.commit()   # local quorum -> single-digit ms

# ---- SLOW: a cross-region transaction --------------------------------------
# This transaction reads a row homed in eu-west and writes a row homed in
# us-east (a global ledger). Now COMMIT must reach quorum in BOTH regions,
# so it pays a cross-region round-trip (~80-150 ms) no matter how fast the
# SQL itself is. Same code shape, ~30x the latency.
with conn.cursor() as cur:
    cur.execute("SELECT total_cents FROM customer_orders "
                "WHERE region = 'eu-west' AND order_id = %s", (order_id,))
    (total,) = cur.fetchone()
    cur.execute("""
        INSERT INTO global_ledger (region, entry_id, amount_cents)
        VALUES ('us-east', gen_random_uuid(), %s)
    """, (total,))
conn.commit()   # cross-region quorum -> the latency tax, in full
```

The lesson encoded in those two blocks is the single most important design rule for distributed SQL: **keep each transaction's rows in one region.** The fast transaction is fast not because the SQL is simpler — it's about the same — but because every row it touches is homed in `eu-west`, so the commit's quorum is local. The slow transaction is slow because it spans `eu-west` and `us-east`, forcing the commit to reach quorum in both. If you find yourself writing a lot of the second kind, either your data model doesn't match your access pattern (the global ledger should perhaps be append-only with regional sub-ledgers reconciled asynchronously), or you genuinely have a global-consistency requirement and the tax is the price of correctness. Both are valid; the mistake is paying the tax *without realizing you are*.

### Second-order effect: `GLOBAL` tables for read-mostly reference data

Geo-partitioning isn't only about pinning rows to one region. Some tables — currency codes, feature flags, product metadata — are read constantly from everywhere and written rarely. For these, CockroachDB offers `LOCALITY GLOBAL`: reads are fast (local, non-stale) in *every* region, at the cost of making the rare writes slower (they must propagate everywhere). It's the inverse tradeoff from `REGIONAL BY ROW`, and choosing the right locality per table is the real skill. A common production layout is `REGIONAL BY ROW` for user data, `GLOBAL` for reference data, and plain `REGIONAL` (homed in one region) for region-specific operational tables.

## 4. Spanner's extra wait: TrueTime and commit-wait

**Senior rule of thumb: Spanner trades a small, bounded, deterministic wait for the elimination of an entire class of clock-skew bugs.** Spanner deserves its own section because it adds a second cost on top of consensus — and understanding it tells you why Google was willing to put atomic clocks in datacenters.

The problem Spanner solves is *external consistency* (linearizability): if transaction T1 commits before T2 starts in real wall-clock time, then T1's timestamp must be less than T2's. To guarantee that across machines with imperfect clocks, you need to know how wrong your clock might be. TrueTime gives every node a clock that returns an *interval* `[earliest, latest]` rather than a point, with a guarantee that true time lies somewhere inside. The width of that interval — the uncertainty epsilon, often a few milliseconds — is kept small precisely by the GPS-and-atomic-clock hardware.

![Spanner's commit timeline: acquire locks at t0, pick a commit timestamp via TrueTime.now() at t1, Paxos commit reaching quorum at t2, commit-wait until the timestamp is safely past at t3, then release locks and return the ack at t4](/imgs/blogs/globally-distributed-sql-when-its-worth-it-6.webp)

The timeline shows the full commit. Spanner acquires locks, picks a commit timestamp from `TrueTime.now().latest`, runs Paxos to commit the write to a quorum (the round-trip we already know about), and then does the thing that looks wasteful until you understand it: **commit-wait.** It deliberately sleeps until TrueTime guarantees the chosen commit timestamp is in the *past* everywhere — i.e., until `TrueTime.now().earliest` exceeds the commit timestamp — before releasing locks and acknowledging. The wait is on the order of the clock uncertainty epsilon, a few milliseconds.

Why pay it? Because that wait is exactly what makes the timestamp ordering match real-world ordering without any node ever needing to trust another node's clock. The cost is a small, *bounded* extra latency on every commit; the benefit is that you never get the subtle anomalies that plague systems trying to order events across drifting clocks. CockroachDB and Yugabyte make the opposite bet: no special hardware, HLC instead of TrueTime, and they handle the uncertainty window by occasionally *restarting* a transaction that reads a value within its uncertainty interval rather than by waiting upfront. The result is serializable transactions on commodity cloud VMs — at the cost of those occasional restarts under contention, and a weaker guarantee (serializable, not external consistency) unless you opt into stricter modes.

The practical upshot, stated as a table:

| System      | Clock mechanism            | Extra cost beyond consensus            | Hardware needed         |
| ----------- | -------------------------- | -------------------------------------- | ----------------------- |
| Spanner     | TrueTime (GPS + atomic)    | commit-wait (~few ms, bounded)         | Yes (specialized)       |
| CockroachDB | Hybrid logical clock (HLC) | occasional read/txn restarts           | No (commodity VMs)      |
| YugabyteDB  | Hybrid logical clock (HLC) | occasional read/txn restarts           | No (commodity VMs)      |

If you run on Google Cloud and want the strongest possible guarantee with the most predictable latency, Spanner's bounded commit-wait is attractive. If you want to run anywhere — on-prem, multi-cloud, a laptop — and can tolerate the occasional restart, the HLC systems are the pragmatic choice. Neither is "better"; they're different points on the same curve.

## 5. The comparison that actually matters

We've now seen every axis. Here is the full landscape — distributed SQL against the alternatives a team actually weighs when a single Postgres primary stops being enough.

![A matrix comparing Spanner, CockroachDB, YugabyteDB, sharded Postgres, and Cassandra across consistency, write latency, ops burden, SQL surface, and special hardware](/imgs/blogs/globally-distributed-sql-when-its-worth-it-5.webp)

Read the matrix column by column and the shape of each choice emerges:

- **Consistency.** Spanner gives external consistency (linearizable); CockroachDB and Yugabyte give serializable; sharded Postgres gives strong consistency *but only within a shard*; Cassandra gives tunable consistency that defaults toward eventual. If your invariant spans rows that might live on different shards — "this transfer debits one account and credits another" — only the distributed SQL options enforce it without you building two-phase commit yourself.
- **Write latency.** This is where the distributed SQL column glows amber: quorum RTT (plus commit-wait for Spanner). Sharded Postgres and Cassandra keep writes local and fast — that's their advantage. The whole post is about whether the consistency you buy is worth that amber cell.
- **Ops burden.** Spanner is fully managed but locks you to Google Cloud. CockroachDB and Yugabyte can be self-run or used as managed services (Cockroach Cloud, Yugabyte Aeon). Sharded Postgres is the heaviest operationally — you own the routing and resharding, in red. Cassandra is self-run and notoriously demanding to operate well.
- **SQL surface.** Spanner, CockroachDB, and Yugabyte give you real SQL (Cockroach and Yugabyte speak the Postgres wire protocol, so most Postgres tooling just works). Sharded Postgres gives full SQL per shard but no cross-shard joins or transactions. Cassandra's CQL looks like SQL but has no joins and a deliberately restricted query model.
- **Special hardware.** Only Spanner needs it (TrueTime's clocks), and you don't see that — Google operates it. Everyone else runs on commodity machines.

There is no row that wins every column. That is the point. You pick the row whose *weaknesses* you can live with.

## 6. When it's worth the tax — and when it isn't

**Senior rule of thumb: distributed SQL is the right answer only when three conditions bind at once — you need strong consistency, AND you span multiple regions, AND you want a relational surface without operating a sharding layer.** Drop any one condition and a cheaper, faster tool wins.

![A decision graph: starting from a scaling decision, ask single region (yes leads to Postgres plus read replicas); if multi-region, ask whether you need strong consistency (eventual is OK leads to Cassandra or DynamoDB; must be strong leads to distributed SQL)](/imgs/blogs/globally-distributed-sql-when-its-worth-it-7.webp)

The decision graph is deliberately short, because the decision is short. Walk it:

- **Single region?** Then you don't have the cross-region problem at all. Run Postgres or MySQL with read replicas and call it done. You get fast strongly-consistent writes (no WAN quorum), full SQL, and an operational story every engineer already knows. Reaching for distributed SQL here is paying the consensus tax — and the operational unfamiliarity — to solve a problem you don't have. This is the most common over-engineering mistake I see.
- **Multi-region, but eventual consistency is acceptable?** Read-heavy workloads where a few seconds of staleness is fine — social feeds, product catalogs, analytics, telemetry — are *better* served by a tuned [Cassandra or DynamoDB](/blog/software-development/database-scaling/tunable-consistency-at-scale). They keep writes local and fast everywhere, scale enormously, and cost less. You give up cross-region transactions, but you didn't need them. Don't pay for serializability you won't use.
- **Multi-region AND must-be-strong AND relational?** *This* is the box distributed SQL was built for. Financial systems where balances must be correct everywhere; inventory where you cannot oversell; global SaaS where a user in one region must see a consistent account state; identity and entitlements where stale data is a security bug. Here the latency tax is the price of correctness, and geo-partitioning shrinks it to where you only pay it on the genuinely cross-region transactions.

### Reach for distributed SQL when

- You have a **multi-region** deployment and a **hard correctness requirement** that spans rows which might live in different places (money, inventory, entitlements).
- You want **automatic sharding and rebalancing** and do *not* want to build, operate, and reshard a manual sharding layer — the operational savings are real and large.
- You need a **relational surface** — joins, secondary indexes, multi-row transactions — not just a key-value API.
- Your access pattern is **mostly region-local**, so geo-partitioning keeps the common-case write in-region and you only pay the tax on the occasional truly-global transaction.
- You value **survivability**: with replicas across regions, you survive an entire region's loss with no data loss and automatic failover — a guarantee single-region Postgres cannot make.

### Skip distributed SQL when

- You are **single-region.** Postgres or MySQL with read replicas is simpler, faster on writes, and cheaper. Don't import a global-scale problem you don't have.
- Your workload is **read-heavy and eventual-consistency is fine.** A tuned Cassandra or DynamoDB is cheaper and faster; the serializable guarantee is dead weight.
- Your write hot path is **cross-region by nature** and latency-critical — e.g., a global high-frequency counter. No geo-partitioning saves you here; rethink the data model (sharded/CRDT counters) instead of paying a WAN round-trip per increment.
- You're choosing it for the **résumé or the conference talk.** The operational maturity, the cost, and the latency tax are all real; adopt it because the three conditions bind, not because it's impressive.

## Case studies from production

### 1. The global signup form that timed out

A SaaS company went multi-region with CockroachDB and, within a week, support tickets piled up about slow signups from Europe. The signup transaction inserted a user, a default workspace, and an audit-log entry. The team's first hypothesis was a slow query or a missing index. The actual root cause: the cluster had no geo-partitioning, so all three tables' ranges were spread across `us-east`, `us-west`, and `eu-west` for "balance." Every EU signup paid a transatlantic quorum round-trip three times — once per table — for a total commit latency north of 250 ms. The fix was to make the user and workspace tables `REGIONAL BY ROW` so an EU user's rows homed in `eu-west`; commit latency for EU signups dropped to single-digit milliseconds. The lesson: distributed SQL defaults to *balanced*, not *local*. Locality is something you declare, and if you don't, you pay the tax on everything.

### 2. The hot sequence that serialized the world

A fintech used a single `accounts` row to hold a global "next invoice number" and incremented it inside every invoice-creation transaction. Single-region, this was fine. Multi-region, it became a catastrophe: because every transaction updated that one row, the row's leaseholder serialized all writers, and each writer held its lock for a full cross-region consensus round-trip. Throughput collapsed to a few invoices per second globally, and p99 latency exceeded a second. The wrong hypothesis was "the database can't handle our write volume" — the volume was trivial. The root cause was contention on a single hot key, amplified by WAN-length lock hold times. The fix was to stop using a global sequence: switch invoice IDs to UUIDs (no coordination) and, where a human-readable sequence was truly needed, shard it per region with a region prefix. The lesson: a hot row that's harmless single-region becomes a global serialization point under cross-region consensus.

### 3. The read replica that wasn't (Spanner stale reads)

A team on Cloud Spanner complained that read latency from their Asian region was high even though "Spanner is globally distributed." Their reads were all *strong* reads, which route to the leader of each split — and their leaders were in the US. The fix was to use Spanner's **stale reads** (`read_timestamp` bounded staleness) for the 90% of reads that tolerated a few seconds of staleness, letting them serve from a local replica without the cross-region hop to the leader. Latency for those reads dropped from ~150 ms to ~10 ms. The lesson is identical to CockroachDB's follower reads: in a distributed SQL system, *strong reads pay for leader locality and stale reads don't* — and most reads can be stale if you let them.

### 4. The five-replica config nobody profiled

A team deployed a CockroachDB cluster with five replicas per range across five regions "for maximum durability," then was puzzled that writes were slower than a three-replica test cluster had been. The reason is arithmetic: quorum of five is three, so every write had to reach the *third*-nearest replica, not the second. Spreading those five replicas across five regions meant the third one was often a continent away. The fix was to reduce most tables to three replicas (quorum two — only the nearest remote needed) and reserve five-replica configs for the small set of data where surviving two simultaneous region failures was a real requirement. Write p50 improved markedly. The lesson: replica count is a write-latency knob, not just a durability knob, and more is not free.

### 5. The migration that assumed Postgres semantics

A team migrated a Postgres app to YugabyteDB (Postgres-compatible) expecting a drop-in replacement, and were surprised by intermittent transaction restarts under load that their Postgres app had never seen. The cause: Yugabyte uses HLC and optimistic concurrency, so transactions that conflict get *restarted* rather than blocking-then-proceeding the way Postgres's default isolation often did. The app didn't have retry logic because Postgres rarely needed it. The fix was to wrap transactions in a retry loop (the standard pattern for serializable distributed SQL) and to reduce contention on a few hot rows. The lesson: "Postgres wire-compatible" means the *protocol* and most SQL are the same; the *concurrency model* is serializable-with-restarts, and your application has to be written to retry.

### 6. Spanner for inventory, where the tax was the point

A global retailer moved their inventory-reservation system to Spanner specifically *because* of the latency tax, not despite it. Overselling — two customers in different regions both buying the last unit — was a real revenue-and-trust problem on their previous eventually-consistent store. Spanner's external consistency made the reservation transaction globally serializable: the second buyer's reservation sees the first's commit, full stop. Yes, each reservation paid consensus plus commit-wait, but a reservation taking 50 ms instead of 5 ms was completely invisible to a human clicking "buy," while a double-sell was a customer-service incident every time. The lesson: when correctness *is* the product requirement, the latency tax is not a cost to minimize — it's the thing you're paying for, and it's cheap relative to the alternative.

### 7. The analytics workload that didn't belong here

A company ran heavy analytical aggregations — month-long scans over an events table — against their CockroachDB cluster and concluded distributed SQL was "slow for analytics." It is. Distributed SQL is an OLTP engine: optimized for many small, consistent transactions, not for scanning billions of rows. The fix wasn't to tune Cockroach; it was to stream events into a columnar analytics store (ClickHouse) built for exactly those scans, and leave the distributed SQL cluster to do transactional work. The lesson: distributed SQL solves the *consistency-across-regions* problem, not the *scan-a-lot-of-data* problem. Using it for analytics is the same category error as using Postgres as a message queue — it'll work until it spectacularly doesn't, and the right tool was always one store over.

## Further reading

- [Spanner: TrueTime and external consistency](/blog/software-development/database/spanner-truetime-and-external-consistency) — the full mechanics of GPS-and-atomic-clock timekeeping and commit-wait.
- [CockroachDB: distributed SQL and serializable transactions](/blog/software-development/database/cockroachdb-distributed-sql-deep-dive) — how Raft, HLC, and read-restarts deliver serializability on commodity hardware.
- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — where distributed SQL sits on the ladder of scaling moves, and why it's a top-rung, one-way-door choice.
- [Tunable consistency at scale](/blog/software-development/database-scaling/tunable-consistency-at-scale) — the eventual-consistency alternative for the read-heavy, staleness-tolerant workloads where distributed SQL is overkill.

The latency tax is real, it's physical, and no amount of marketing makes it go away. But it is also *predictable* and *controllable*: predictable because it equals the round-trip to your nearest quorum, controllable because geo-partitioning lets you put that quorum next door for the common case. Distributed SQL is not a magic global database with no downsides; it's a precise trade — strong consistency for cross-region write latency — wrapped in an operational experience so good it hides the trade until your first slow commit from Singapore. Know what you're buying, keep your transactions in one region, let your reads be a little stale, and the tax is one most global systems should be happy to pay. Just don't pay it when you're single-region, read-heavy, or doing analytics — because then you're buying a guarantee you'll never use, at a price the speed of light sets and refuses to negotiate.

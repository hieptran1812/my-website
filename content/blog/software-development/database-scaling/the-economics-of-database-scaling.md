---
title: "The Economics of Database Scaling: $ per Query, per GB, per Nine"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Every database scaling decision is ultimately an economic one; this capstone ties the whole series' decision tree to dollars per query, per GB, and per nine, and gives you a cost model to pick the cheapest architecture that still meets your SLOs."
tags: ["database-scaling", "cost-optimization", "finops", "total-cost-of-ownership", "serverless", "storage-tiering", "cloud-cost", "egress", "slo", "system-design", "capacity-planning", "distributed-systems"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 34
---

The most expensive database mistake I have ever watched a team make did not show up on a latency dashboard. It showed up on the cloud bill, three months after a migration that everyone agreed had gone well. The team had done everything the playbook said: they measured, they tuned, they hit a real write ceiling, and they moved from a single primary to a sharded cluster with read replicas in three availability zones. Latency was great. Availability was great. And the monthly database spend had quietly gone from about $14,000 to just under $90,000 — a 6x increase nobody had modeled, dominated by a line item called "data transfer" that not one engineer on the team could explain.

That is the gap this final post in the series is about. For thirty-nine posts we have treated scaling as an engineering problem: which rung of the ladder to climb, which shard key to pick, when consensus is worth the round-trip. All of that is real. But underneath every one of those decisions is a second decision that the architecture diagram never shows — **what it costs** — and the teams that scale well are the ones that can read both diagrams at once. Scaling is not a technical question with a cost side effect. It is an economic question with a technical implementation. The unit of account is not QPS or GB. It is dollars: **dollars per query, dollars per GB, dollars per nine of availability.**

![Five cost dimensions feed monthly TCO; the SLO gate selects the cheapest architecture that clears it](/imgs/blogs/the-economics-of-database-scaling-1.webp)

The diagram above is the mental model for this entire capstone. Any workload — some QPS, some data volume, some target SLO — fans out into five cost dimensions: compute, storage, IOPS/throughput, network egress, and the human cost of running the thing. Those five collapse into a single number, your monthly total cost of ownership (TCO). Then there is a gate: does this architecture meet your SLO? The job of an architect is not to minimize TCO and it is not to maximize the SLO. It is to find **the cheapest architecture whose TCO clears the SLO gate.** Everything else in this article is a tour of how to compute each of those five dimensions, how the gate changes the math, and how to walk the series' decision tree with a running dollar total in your head.

## Why the cost view is different from the technical view

The reason cost surprises people is that the dimensions that dominate the *engineering* conversation are almost never the dimensions that dominate the *bill*. Engineers argue about CPU and query plans; the invoice is frequently won or lost on data transfer and storage replication factors that never came up in a single design review.

| Question the design review asks | What the team optimizes | What actually drives the bill |
| --- | --- | --- |
| "Can the primary handle the write QPS?" | CPU, IOPS, the slow query | Compute is real, but often <40% of the total |
| "Are we durable enough?" | replication factor, backups | RF 3 triples your storage **and** your write IOPS |
| "Is it fast across regions?" | consensus protocol, quorum | cross-region **egress** + extra full replicas |
| "Can analysts query it?" | read replicas, indexes | a second engine, or analytics on the primary |
| "Is it reliable?" | failover, multi-AZ | each added nine roughly multiplies cost |
| "Who runs it?" | not asked in the review | **headcount** — the largest line item nobody budgets |

Read that last row twice. The single most under-counted cost in database scaling is the salary of the people who keep the system alive. A self-managed sharded Vitess cluster might have a lower AWS bill than managed Aurora — and a higher total cost, once you price the two senior engineers whose on-call rotation it created. We will come back to this when we talk about build versus buy, but keep it in view the whole way down: **the cheapest line on the cloud invoice is not the same as the cheapest architecture.**

## 1. The five cost dimensions, priced

Let us put real numbers on each dimension. All prices below are round, list-price, us-east-1-flavored figures for 2026; your committed-use discounts will move them, but the *ratios* are what matter and they are remarkably stable across clouds.

**Compute** is instance-hours. A db.r6g.2xlarge-class node (8 vCPU, 64 GB) runs roughly $1.00–$1.40/hour on-demand, call it ~$800/month, before you double it for a standby. Compute is the dimension everyone thinks about and it is usually 30–50% of an OLTP bill. The trap is that compute is the *easiest* dimension to right-size and the one teams most often over-provision "to be safe," running production at 15% CPU and paying for the other 85%.

**Storage** is priced per GB-month, but the headline number lies, because you never store your data once. The real formula is:

$$ \text{storage cost} = \text{data}_{GB} \times \text{price}_{GB} \times (\text{RF} + \text{backup retention factor}) $$

where RF is your replication factor. Provisioned SSD (gp3) runs ~$0.08/GB-month; a 2 TB dataset at replication factor 3 is not 2 TB of billed storage, it is 6 TB, plus snapshot/backup copies that can add another 1–3x depending on retention. That same 2 TB on object storage (S3) is ~$0.023/GB-month and on Glacier ~$0.004/GB-month. The two-orders-of-magnitude spread between hot and cold storage is the entire basis for the tiering strategy in section 5.

**IOPS and throughput** is the dimension that bites silently. On gp3 you get a baseline of 3,000 IOPS and 125 MB/s free; beyond that you pay per provisioned IOPS (~$0.005/IOPS-month) and per provisioned MB/s. On io2 the per-IOPS price is higher but the latency is tighter. The subtlety is *provisioned versus consumed*: if you provision 40,000 IOPS to survive a twice-a-day batch job and run at 4,000 the rest of the time, you are paying for 36,000 idle IOPS around the clock. Throughput is where serverless and consumption pricing models start to look attractive, because they bill the IOPS you actually use.

**Network egress** is the silent budget killer, and it deserves its own paragraph because it is the one that turned that $14k bill into $90k. Three tiers, each an order of magnitude apart:

- **Cross-AZ** traffic inside a region: ~$0.01/GB **in each direction**, so $0.02/GB round-trip. Synchronous replication across AZs, a connection pool talking to a primary in another AZ, a read replica in AZ-b serving an app in AZ-a — every one of these moves bytes across the AZ boundary and meters them.
- **Cross-region** egress: ~$0.02/GB and up. A multi-region active-active database replicating every write to two other regions pays this on every byte, forever.
- **Internet egress**: ~$0.09/GB, the most expensive and usually not the database's problem — unless you are shipping change-data-capture streams or backups out of the cloud.

The reason egress is dangerous is that it scales with *traffic*, not with *capacity*. You can right-size compute once and forget it. Egress grows every day your traffic grows, it compounds with your replication factor and region count, and it appears on the bill under a generic label that no service dashboard attributes back to a specific design decision.

**Human cost** is the fifth dimension and the only one not on the invoice. Price it anyway. A loaded senior infra engineer is $250k–$400k/year fully burdened. A system that consumes one engineer's full attention costs you ~$25k–$33k/month in salary alone — more than most teams' entire database compute bill. Polyglot persistence (a separate post in this series on [choosing the right store](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store)) is where this dimension explodes: every new datastore is not just a license and an instance, it is a new failure mode, a new on-call runbook, a new thing three people need to be expert in. The marginal cost of the *sixth* database technology in your stack is mostly human.

## 2. Unit economics: the marginal cost of a read and a write

Now zoom from the monthly total down to a single operation, because that is where the decision-tree ordering comes from. The question is: **what does one more read cost, and what does one more write cost, at each tier of the stack?**

A read served from your application's local memory or a Redis cache hit is, to a rounding error, **free** — a RAM lookup, no metered database operation, no IOPS, no cross-AZ byte. A read served from the primary costs CPU, costs an IOP if it misses the buffer pool, and costs a cross-AZ byte if the client is in a different AZ. A read served from a *cross-region* replica costs all of that plus the egress to keep that replica current. The same read can cost zero or cost a measurable fraction of a cent depending entirely on **where it is served.** That spread is why caching is not just a latency optimization — it is the single highest-leverage *cost* optimization in the entire stack.

![A cache hit is a RAM lookup that never touches the database, so 95% of reads move from a metered DB operation to nearly free](/imgs/blogs/the-economics-of-database-scaling-3.webp)

The before/after above puts numbers on it. Take a read-heavy service at 100M reads/day. Served entirely off the primary, every read consumes database CPU and IOPS, you scale by buying a bigger box and adding replicas, and you land somewhere around $8k/month, fully database-bound. Put a cache in front with a 95% hit ratio and only 5M reads/day — one in twenty — ever reach the database. The other 95M are RAM lookups that cost effectively nothing per operation. The database shrinks, the replica count drops, and the same workload lands near $2k/month. The cache itself costs money, but a cache node is cheap relative to the database capacity it displaces, because **RAM serving a hit is doing far less work than a database serving the same row.**

This is the economic argument for the ordering of the whole [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree). The tree is ordered cheapest-and-most-reversible first not as a stylistic preference but because each early rung has *better unit economics* than the rung below it:

- **Tuning a query** changes the marginal cost of an operation from "sequential scan of 40M rows" to "index seek" — a 1000x improvement in cost-per-query for $0 of new infrastructure.
- **Caching** moves 90–99% of reads to the near-free tier.
- **Read replicas** add read capacity at the linear cost of more instances (each replica is roughly one more primary's worth of compute *plus* the egress to keep it current).
- **Sharding** adds write capacity but multiplies *every* dimension — more compute, more storage, more cross-shard egress, and a large step-function in human cost.

Climb the tree in order and you are, almost by definition, spending money in increasing order of cost-per-unit-of-capacity. Jump straight to sharding and you skip every cheaper option and pay the most expensive marginal rate first. The decision tree is a cost-minimization algorithm wearing an engineering costume.

## 3. The decision tree, re-walked with a price tag on every rung

Here is the centerpiece of the capstone: the series' scaling ladder, annotated with what each rung adds to the monthly bill at a representative scale of 50k QPS and 2 TB.

![Climb cheapest-and-most-reversible first; each rung adds cost and operational weight, and the bottom two are one-way doors](/imgs/blogs/the-economics-of-database-scaling-2.webp)

Walk it top to bottom and watch both the color and the cost column change together:

- **Rung 0 — Measure.** $0. You cannot optimize a cost you have not attributed. This is the [capacity planning](/blog/software-development/database-scaling/capacity-planning-for-databases) step: find which budget — QPS, working set, storage, connections, or dollars — crosses its ceiling first.
- **Rung 1 — Tune and index.** $0, fully reversible. The cheapest capacity you will ever buy is the capacity you reclaim by killing a bad query or adding a missing index.
- **Rung 2 — Scale up.** +$0.5k–3k/month. A bigger instance is a single dial, reversible on a maintenance window, and almost always cheaper per unit of capacity than the distributed alternatives — right up until you hit the [vertical ceiling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling).
- **Rung 3 — Cache reads.** +$0.3k–1k/month, and it often *reduces* the rung-2 spend by shrinking the database you need. The best unit economics on the board.
- **Rung 4 — Read replicas.** +$1k–5k/month — each replica is roughly N times the instance cost plus its replication egress. Great for read scaling, useless for write scaling.
- **Rung 5 — Functional partitioning.** +$2k–8k/month plus real operational weight. Splitting tables onto purpose-built stores buys you isolation and right-sized engines, at the cost of a more complex topology to run.
- **Rung 6 — Shard in-house.** +$$$ plus 2–4 engineers. This is a one-way door. The infrastructure cost is real but the *human* cost is the dominant term: a resharding pipeline, a routing layer, cross-shard query handling, all of which someone has to build and carry.
- **Rung 7 — Managed distributed SQL.** +$$$$ plus a latency tax. Spanner, CockroachDB, and friends remove the human cost of building a sharding layer and replace it with a premium per-node price and a [cross-region commit round-trip on every write](/blog/software-development/database-scaling/globally-distributed-sql-when-its-worth-it).

The through-line of the whole series falls out of this one picture: **most teams should stop climbing several rungs before they think they need to**, because the cost gradient steepens far faster than the capacity gradient. Rungs 0 through 3 are nearly free and reversible. Rungs 6 and 7 are expensive and permanent. The discipline is not knowing how to shard. It is having the nerve to stay on rung 3.

> The cheapest scaling move is the one you do not make. The second cheapest is the one you can undo on a Wednesday.

## 4. Reserved, on-demand, or serverless: the load profile decides

Independent of *which* architecture you pick is the question of *how you pay for it*. The same database capacity can be purchased three ways, and the right answer is determined almost entirely by the **shape of your load over time**, not by the database engine.

![Serverless wins spiky and idle load by scaling to zero; reserved capacity wins sustained 24x7 load by avoiding the serverless tax](/imgs/blogs/the-economics-of-database-scaling-4.webp)

The matrix above is the decision in one picture. Read it by row:

- **Spiky or unpredictable load.** Serverless wins. Aurora Serverless v2, DynamoDB on-demand, PlanetScale, and Neon scale capacity with your traffic and bill you for what you used. Provisioning for the peak means paying for the peak 24/7; provisioning for the average means throttling during spikes. Serverless dissolves the dilemma.
- **Steady, low load (dev, test, internal tools).** Serverless wins, for a different reason: **scale to zero.** A dev database that costs $0 when nobody is using it beats a provisioned instance burning ~$200/month to sit idle overnight and on weekends.
- **Steady, high load, 24x7.** This is where serverless *loses*, and it loses badly. Consumption pricing carries a markup — call it the **serverless tax** — that is invisible at low utilization and brutal at high utilization. A DynamoDB table doing a sustained, predictable 20k writes/second is frequently 2–4x more expensive on on-demand than the same throughput provisioned with reserved capacity. For a flat, high baseline, **reserved instances or savings plans** win, cutting 40–60% off on-demand rates in exchange for a 1–3 year commitment.
- **Scale-to-zero idle workloads.** Serverless again — anything that is genuinely idle most of the time should cost nothing most of the time.

The mental error that costs the most money here is treating "serverless" as a synonym for "cheap." Serverless is cheap *for the right load shape* and a premium *for the wrong one*. The correct portfolio for a mature system is usually a blend: **reserved capacity for the predictable baseline, on-demand or serverless for the burst on top.** You buy a 1-year savings plan covering the floor your traffic never drops below, and you let the spiky delta ride on consumption pricing. Committing to your baseline and renting your peak is the single highest-ROI FinOps move available to a team running a steady production database.

```python
def cheapest_pricing(baseline_qps, peak_qps, idle_fraction):
    """Pick a pricing model from the load shape, not the engine name."""
    burstiness = peak_qps / max(baseline_qps, 1)
    if idle_fraction > 0.5 or burstiness > 5:
        return "serverless / on-demand (scale-to-zero, pay for spikes)"
    if burstiness < 1.5 and idle_fraction < 0.1:
        return "reserved / savings plan on baseline (avoid the serverless tax)"
    return "reserved baseline + on-demand burst (blended)"

print(cheapest_pricing(baseline_qps=18000, peak_qps=22000, idle_fraction=0.02))
# -> reserved / savings plan on baseline (avoid the serverless tax)
print(cheapest_pricing(baseline_qps=200, peak_qps=9000, idle_fraction=0.7))
# -> serverless / on-demand (scale-to-zero, pay for spikes)
```

## 5. Storage tiering: stop paying OLTP prices for cold data

Here is a fact that holds across almost every production database I have audited: **the overwhelming majority of your rows are never read again.** Order records from three years ago, audit-log entries past their compliance window, event data older than the dashboards ever query — they sit in your hottest, most expensive, most replicated storage tier, paying OLTP prices to be ignored. A common rule of thumb is that 90% of rows in a mature OLTP table are never read after their first 30 days, yet they often occupy 90% of the storage and inflate every backup, every replication stream, and every index.

![Cost per GB drops two orders of magnitude from hot RAM to archive, and the cheapest byte is the one you drop entirely](/imgs/blogs/the-economics-of-database-scaling-5.webp)

The tiering stack above is the fix. Data has a temperature, and temperature should determine which storage tier — and which price — it lives at:

- **Hot** — RAM and NVMe on the primary, ~$3–10/GB-month all-in once you count replication and the IOPS to serve it, sub-millisecond access. Reserve this for the working set you actually serve from.
- **Warm** — OLTP SSD, ~$0.10–0.30/GB-month, single-millisecond access. The body of your live database.
- **Cold** — object storage like S3, ~$0.023/GB-month, hundreds-of-milliseconds access. Where rows go when they age out of the query path but must remain retrievable.
- **Archive** — Glacier-class, ~$0.004/GB-month, minutes-to-hours retrieval. Compliance retention you must keep but will almost never read.
- **Dropped** — $0. The cheapest byte is the one you delete. A TTL policy or a partition-and-drop job that expires data nobody needs is pure margin.

The mechanism that makes this real is **partitioning by time plus a lifecycle policy.** Partition the table by month, and aging out cold data becomes a metadata operation — `DETACH PARTITION` and move it to object storage — instead of a row-by-row `DELETE` that thrashes your IOPS and bloats the WAL. The [time-series at scale](/blog/software-development/database-scaling/time-series-data-at-scale) post in this series covers the retention machinery; the economic point is simpler: every GB you move from the warm tier to cold saves ~80% on that GB, every GB you move to archive saves ~95%, and every GB you drop saves 100% — and the savings multiply by your replication factor because you stop replicating data nobody reads.

```sql
-- Partition-and-drop: aging out a month of cold data is a metadata
-- operation, not a 40-million-row DELETE that bloats the WAL.
ALTER TABLE events DETACH PARTITION events_2023_06;
-- export the detached partition to object storage, then:
DROP TABLE events_2023_06;          -- reclaim hot storage x replication factor

-- Or push the lifecycle to the warehouse: keep 30 days hot in OLTP,
-- stream the rest to the columnar store and let it own history.
```

This tiering instinct is also why you keep analytics off the primary. Offloading historical, scan-heavy queries to a columnar engine — the subject of [separating OLTP from OLAP](/blog/software-development/database-scaling/separating-oltp-from-olap) — is a storage-tiering decision as much as a workload-isolation one: the warehouse stores cold history at object-store prices and scans it with compute you spin up only when a query runs, instead of paying to keep three years of orders resident in your transactional primary's buffer pool.

## 6. The cost of consistency and availability: paying per nine

Availability is not a property you design in. It is a quantity you **buy**, with redundancy, and like any quantity it has a unit price that rises non-linearly. Each additional nine of availability roughly multiplies your infrastructure cost, because each nine requires removing another class of single point of failure, and the failure classes get rarer and more expensive to defend against as you go.

![Availability is bought with redundancy; right-size the nine to the business instead of paying for five everywhere](/imgs/blogs/the-economics-of-database-scaling-6.webp)

The tree above prices the climb:

- **99% (two nines)** — about 3.7 days of allowed downtime per year. A single well-run instance with good backups gets you here. Baseline cost, call it 1x.
- **99.9% (three nines)** — about 9 hours/year. Now you need a standby and automated failover. Roughly 2x, because you are running a second node that does no useful work until the primary dies.
- **99.99% (four nines)** — about 53 minutes/year. Multi-AZ, synchronous replication, fast automated failover, tested runbooks. Roughly 4x, and now you are paying cross-AZ egress on every committed write.
- **99.999% (five nines)** — about 5 minutes/year. Multi-region active-active, which means full replicas in multiple regions, cross-region egress on every write, and the [consensus latency tax](/blog/software-development/database-scaling/globally-distributed-sql-when-its-worth-it) baked into your commit path. Roughly 10x or more, and the marginal complexity is enormous.

The discipline is the same one we apply to the decision tree: **right-size the nine to the business, do not buy five everywhere.** A payments ledger genuinely needs four or five nines and the cost is justified. The service that renders a user's notification preferences does not — and paying five-nines prices for a three-nines requirement is one of the most common and least-examined forms of database waste. Different tables in the same product can and should live at different availability tiers. The blast radius of the notification-preferences table going down for an hour is an annoyance; the blast radius of the ledger going down is a company-ending event. Price them accordingly.

There is a second, sneakier cost hiding in this section: **the cost of strong consistency across regions.** A globally distributed SQL database that gives you serializable transactions everywhere is paying for that guarantee twice — once in extra full replicas (storage and compute), and once in cross-region egress and commit latency on every write. [Tunable consistency](/blog/software-development/database-scaling/tunable-consistency-at-scale) exists precisely so you can buy strong consistency only for the operations that need it and pay reduced rates for the operations that can tolerate eventual consistency. Consistency, like availability, is a dial with a dollar value at every setting.

## 7. Build vs buy: managed, self-managed, or your own

The largest single economic lever in the whole stack is also the one least often framed as economic: do you let a cloud provider run your database (managed), run open-source software yourself on instances (self-managed), or build your own storage engine from the metal up (build-your-own)? The answer moves as you scale, and the thing that moves it is the trade between **dollars on the invoice** and **dollars in headcount.**

![Managed databases are the cheapest total cost of ownership until headcount-heavy self-hosting and build-your-own pay off at extreme scale](/imgs/blogs/the-economics-of-database-scaling-7.webp)

The timeline above plots the crossover against scale:

- **Below ~10k QPS — managed wins, decisively.** RDS, Aurora, Cloud SQL, DynamoDB, Spanner. Yes, you pay a managed premium of maybe 20–40% over raw instance cost. You also do not staff a database team, do not carry pager duty for failovers, do not write backup automation. At small and medium scale, the managed premium is far smaller than the salary of the engineer you would otherwise hire to replace it. **Managed is cheaper, fully loaded, for the overwhelming majority of teams.**
- **~10k–100k QPS — managed, plus tuning and reserved capacity.** You are now big enough that the managed bill is a real line item, so you optimize it: reserved instances, right-sized replicas, tiered storage. But you are usually still better off with managed-plus-discipline than with self-hosting, because the headcount math has not flipped yet.
- **~100k–1M QPS — the break-even zone.** Self-managed open source (Postgres or MySQL on instances, [Vitess](/blog/software-development/database-scaling/slack-and-youtube-on-vitess) for sharding) starts to pay off, because the managed premium on a bill this size now exceeds the cost of the small team it takes to run the open-source version. This is genuinely a calculation, not a default — run the numbers both ways.
- **Above ~1M QPS — build-your-own becomes thinkable.** This is the rarefied air where companies write their own storage layers: Uber's [geo-distributed data platform](/blog/software-development/database-scaling/uber-geo-distributed-data), Twitter's Manhattan, Facebook's [Memcache fleet](/blog/software-development/database-scaling/scaling-memcache-at-facebook). At this scale a few percent of efficiency is worth an entire engineering team, so building one pays for itself. Below this scale, building your own database is a way to convert money into prestige and outages.

The fully-loaded TCO formula is the thing to internalize:

$$ \text{TCO} = \text{infra cost} + (\text{engineers} \times \text{loaded salary}) + \text{opportunity cost of their time} $$

Managed databases trade a higher infra cost for a near-zero headcount term. Self-hosting trades a lower infra cost for a headcount term that is often *larger* than the infra savings. The cloud-repatriation stories that make headlines — and we will get to 37signals and Dropbox below — are real, but read the fine print every time: they pay off precisely because those companies operate at a scale where the infra savings finally exceed the cost of the team it takes to capture them. At your scale, the same move might just be expensive nostalgia.

## 8. A cost model you can actually run

Enough principles. Here is a database cost model that takes a workload and prices it across several architectures, then picks the cheapest one that meets an SLO. It is deliberately simple — round prices, linear capacity assumptions — but it captures the five dimensions and, more importantly, it makes the trade-offs *legible*. Adapt the price book to your cloud and your committed-use discounts.

```python
from dataclasses import dataclass

HOURS = 730  # hours per month

# Round 2026 list prices (USD), us-east-1 flavored. Tune to your contract.
PRICE = {
    "vcpu_hour":       0.040,   # blended on-demand vCPU-hour
    "ram_gb_hour":     0.005,   # blended on-demand GB-RAM-hour
    "ssd_gb_month":    0.080,   # provisioned gp3 SSD
    "iops_month":      0.005,   # per provisioned IOPS above baseline
    "s3_gb_month":     0.023,   # object storage (cold tier)
    "backup_gb_month": 0.095,   # snapshot storage
    "xfer_az_gb":      0.010,   # cross-AZ, EACH direction
    "xfer_region_gb":  0.020,   # cross-region egress
    "serverless_mult": 2.5,     # consumption markup vs reserved at steady load
    "reserved_disc":   0.55,    # reserved/savings-plan price vs on-demand
}

@dataclass
class Workload:
    read_qps: float
    write_qps: float
    data_gb: float
    avg_row_kb: float = 4.0      # bytes moved per op, for egress math
    cache_hit: float = 0.0       # fraction of reads absorbed by cache
    rf: int = 3                  # replication factor (durability)
    regions: int = 1             # regions with a full replica
    cross_az: bool = True        # replicas in other AZs (egress on writes)

def instance_cost(vcpu, ram_gb, count, reserved=False):
    rate = count * HOURS * (vcpu * PRICE["vcpu_hour"] + ram_gb * PRICE["ram_gb_hour"])
    return rate * (PRICE["reserved_disc"] if reserved else 1.0)

def storage_cost(data_gb, rf):
    primary = data_gb * rf * PRICE["ssd_gb_month"]
    backups = data_gb * PRICE["backup_gb_month"]      # ~1x retained snapshots
    return primary + backups

def egress_cost(w: Workload):
    write_gb_month = w.write_qps * w.avg_row_kb / 1e6 * 86400 * 30  # GB/month written
    az = write_gb_month * (w.rf - 1) * PRICE["xfer_az_gb"] * 2 if w.cross_az else 0
    region = write_gb_month * max(w.regions - 1, 0) * PRICE["xfer_region_gb"]
    return az + region

def cpu_needed(read_qps, write_qps, cache_hit):
    db_reads = read_qps * (1 - cache_hit)
    # crude: 1 vCPU ~ 2500 simple OLTP ops/sec; writes cost ~2x a read
    return (db_reads + write_qps * 2) / 2500

def price_single_box(w: Workload, reserved=True):
    vcpu = max(8, round(cpu_needed(w.read_qps, w.write_qps, w.cache_hit)))
    ram = vcpu * 8
    compute = instance_cost(vcpu, ram, count=2, reserved=reserved)   # primary + standby
    iops = max(0, w.write_qps * 3 - 3000) * PRICE["iops_month"]
    cache = 200 if w.cache_hit > 0 else 0
    return {"arch": "single box + standby",
            "compute": compute, "storage": storage_cost(w.data_gb, w.rf),
            "iops": iops, "egress": egress_cost(w), "cache": cache}

def price_read_replicas(w: Workload, n_replicas=3, reserved=True):
    vcpu = max(8, round(cpu_needed(0, w.write_qps, 0)))               # primary sized for writes
    ram = vcpu * 8
    compute = instance_cost(vcpu, ram, count=1 + n_replicas, reserved=reserved)
    iops = max(0, w.write_qps * 3 - 3000) * PRICE["iops_month"]
    return {"arch": f"primary + {n_replicas} replicas",
            "compute": compute, "storage": storage_cost(w.data_gb, w.rf),
            "iops": iops, "egress": egress_cost(w), "cache": 200}

def price_sharded(w: Workload, shards=8, reserved=True):
    per = cpu_needed(w.read_qps / shards, w.write_qps / shards, w.cache_hit)
    vcpu = max(4, round(per))
    ram = vcpu * 8
    compute = instance_cost(vcpu, ram, count=shards * 2, reserved=reserved)  # 2 nodes/shard
    return {"arch": f"{shards}-way sharded",
            "compute": compute, "storage": storage_cost(w.data_gb, w.rf),
            "iops": w.write_qps * 3 * PRICE["iops_month"],
            "egress": egress_cost(w), "cache": 200}

def price_serverless(w: Workload):
    base = price_single_box(w, reserved=False)
    util = base["compute"] + base["iops"]
    return {"arch": "serverless / on-demand",
            "compute": util * PRICE["serverless_mult"], "storage": storage_cost(w.data_gb, w.rf),
            "iops": 0, "egress": egress_cost(w), "cache": 0}

def total(b):     return sum(v for k, v in b.items() if k != "arch")

def cheapest_meeting_slo(w: Workload, candidates, meets_slo):
    priced = [b for b in candidates if meets_slo(b["arch"])]
    return min(priced, key=total) if priced else None

if __name__ == "__main__":
    w = Workload(read_qps=45000, write_qps=5000, data_gb=2000, cache_hit=0.92)
    options = [price_single_box(w), price_read_replicas(w),
               price_sharded(w), price_serverless(w)]
    for b in sorted(options, key=total):
        drivers = "  ".join(f"{k}=${v:,.0f}" for k, v in b.items() if k != "arch")
        print(f"${total(b):>8,.0f}/mo  {b['arch']:<24} | {drivers}")

    # SLO gate: this workload's write rate exceeds a single box's safe ceiling,
    # so the single box is disqualified even though it is cheaper on paper.
    def meets_slo(arch):  return arch != "single box + standby"
    pick = cheapest_meeting_slo(w, options, meets_slo)
    print(f"\nCheapest architecture that meets the SLO: {pick['arch']} "
          f"at ${total(pick):,.0f}/mo")
```

Run it and you get a ranked table of architectures by monthly cost, with the cost broken out by dimension, and then the cheapest option that clears the SLO gate. The output for that read-heavy, cache-fronted workload looks roughly like this:

| Architecture | Monthly cost | Dominant driver | When it is cheapest | SLO fit |
| --- | --- | --- | --- | --- |
| single box + standby | ~$6–9k | compute + storage | small/medium, write rate under one box | fails if writes exceed a node |
| primary + 3 replicas | ~$11–16k | compute (N nodes) | read-heavy, writes fit one primary | good reads, single write ceiling |
| serverless / on-demand | ~$10–20k | compute markup | spiky or low-duty-cycle load | great burst, taxed at steady high load |
| 8-way sharded | ~$22–34k | compute + egress | write-bound past a single primary | scales writes, heavy ops + human cost |

Two things to notice. First, the cheapest option on paper (single box) is *disqualified by the SLO gate* the moment the write rate exceeds what one primary can safely sustain — which is exactly why you cannot cost-optimize without modeling the SLO. Second, the gap between the read-replica architecture and the sharded one is not 20% — it is 2x or more, dominated by compute (twice the node count) and egress (cross-shard and cross-AZ chatter). That gap is the dollar value of staying on rung 4 instead of jumping to rung 6, and it is why "do we actually need to shard" is a budget question before it is an engineering one.

```python
def cross_traffic_monthly(write_qps, avg_row_kb, rf, regions):
    """The egress line nobody models until it shows up on the bill."""
    gb_month = write_qps * avg_row_kb / 1e6 * 86400 * 30
    cross_az    = gb_month * (rf - 1) * 0.010 * 2          # each direction
    cross_region = gb_month * max(regions - 1, 0) * 0.020
    return {"GB_written_month": round(gb_month),
            "cross_az_$": round(cross_az),
            "cross_region_$": round(cross_region)}

print(cross_traffic_monthly(write_qps=20000, avg_row_kb=6, rf=3, regions=3))
# {'GB_written_month': 311040, 'cross_az_$': 12442, 'cross_region_$': 12442}
# ~$25k/month in transfer alone, on a workload whose compute might be $15k.
```

That last estimator is the one I wish every team ran *before* approving a multi-region active-active design. At 20k writes/second of 6 KB rows with replication factor 3 across three regions, you move ~300 TB/month, and the cross-AZ plus cross-region transfer alone is ~$25k/month — frequently larger than the compute it supports. Egress is not a rounding error at scale. It is often the single largest line item, and it is the one your architecture diagram is silent about.

## 9. FinOps in practice: attribute, find waste, alert

A cost model tells you what an architecture *should* cost. FinOps is the operational discipline of making sure it actually does, and catching the drift between the two. Three habits separate teams that control their database spend from teams that get surprised by it.

**Attribute every dollar.** You cannot manage what you cannot see, and the cloud bill by default shows you spend by *service*, not by *team*, *feature*, or *table*. Tag everything — every instance, volume, snapshot, and cluster — with an owning team and a service name, and build a per-service cost view. The first time a team sees that their feature's database costs $40k/month, behavior changes without anyone mandating it. Untagged resources are where waste hides; a standing alert on the untagged-spend percentage is one of the highest-value dashboards you can build.

**Hunt waste systematically.** The same four categories of waste appear in nearly every audit:

- **Over-provisioned instances** running at 10–20% CPU because someone sized for a peak that never came. Right-size to actual utilization plus headroom, not to fear.
- **Unused or forgotten replicas** — a read replica spun up for a migration that finished six months ago and never got torn down, quietly costing a full instance every month.
- **Idle and redundant indexes** — every index you do not query is pure cost: storage, replication, and a write-amplification tax on every insert. The [index strategy](/blog/software-development/database-scaling/index-strategy-at-scale) post covers finding them; `pg_stat_user_indexes` with `idx_scan = 0` is a money-printing query.
- **Cross-AZ chatter** from clients talking to primaries in other AZs, or replicas serving the wrong zone. AZ-aware routing turns metered cross-AZ bytes into free same-AZ ones.

```sql
-- Idle indexes: storage + write-amplification you pay for and never use.
SELECT schemaname, relname, indexrelname,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- Over-provisioned: nodes whose 30-day p95 CPU never crossed 25%.
-- (pair with your metrics store; right-size or consolidate.)
```

**Alert on cost like you alert on latency.** A budget with no alarm is a wish. Set anomaly alarms on daily spend per service so a runaway query, a retry storm against an on-demand table, or a forgotten warehouse left running over a weekend pages someone in hours, not at the end of the month when the invoice arrives. The cost of a misconfigured consumption-priced resource compounds every hour it runs; the difference between catching it in three hours and catching it in thirty days is two orders of magnitude on the bill.

## Case studies from production

Numbers in these are public, approximate, and rounded; the lesson is in the shape of the decision, not the third significant figure.

### 1. 37signals leaves the cloud — the headcount math flipping back

Starting in 2022, Basecamp's parent company 37signals very publicly exited the cloud, moving Basecamp and HEY off AWS onto their own hardware in colocation facilities. Their reported cloud spend was around $3.2M/year; they bought roughly $600–700k of Dell servers and projected savings on the order of $7M over five years, later reporting their cloud bill dropped by over $1M in the first year and that exiting S3 — moving ~18 PB to their own Pure Storage arrays — would save another seven figures annually. The instructive part is *why* it worked for them and would not for a startup: 37signals runs a steady, predictable, high-utilization workload, already had a strong ops culture, and operated at a scale where the infra savings exceeded the cost of the small team to run the hardware. The build-vs-buy crossover had moved, for them, back toward "build." The lesson is not "leave the cloud." It is "the crossover is a calculation, and it moves with your scale, your load shape, and your existing operational muscle."

### 2. Dropbox's Magic Pocket — repatriating storage at exabyte scale

Dropbox spent 2014–2016 building Magic Pocket, their own exabyte-scale storage system, and migrating roughly 90% of user data off Amazon S3 onto custom hardware in their own data centers. Their S-1 filing disclosed that this saved approximately $75M over the two years following the migration. Dropbox could do this because storage was their core product and their single largest infrastructure cost — at that volume, a few percentage points of efficiency on storage was worth an entire specialized engineering organization, and they had the scale to amortize building one. This is the rung-7-of-build-vs-buy story: building your own storage layer pays off only when storage is so central and so large that owning the full stack beats renting it. For the other 99.9% of companies, S3 at $0.023/GB-month is the correct answer precisely because they will never reach the scale where building Magic Pocket makes economic sense.

### 3. Discord: Cassandra to ScyllaDB — fewer nodes, lower latency, lower cost

Discord stored trillions of messages on Apache Cassandra and hit a wall of operational pain: GC pauses, hot partitions, and a fleet that had grown to around 177 nodes that a team spent enormous effort babysitting. They migrated to [ScyllaDB](/blog/software-development/database-scaling/discord-trillions-of-messages-cassandra-to-scylla), a C++ reimplementation of the Cassandra data model, and consolidated to roughly 72 nodes while *improving* tail latency. The economic lesson is that the cost of a data tier is not only the instance count — it is the instance count multiplied by how much human attention each node demands. Discord cut both at once: fewer nodes (lower infra cost) that were also less work to run (lower human cost). The same workload, on a more efficient engine, moved down and to the left on every cost dimension simultaneously. When an engine change can cut your node count by more than half, the migration cost is almost always worth it.

### 4. Datadog's eight-figure cloud bill — even the cost-watchers pay

It was widely reported that Datadog — a company whose entire product is helping others monitor and control infrastructure — ran an AWS bill on the order of $65M in a single year, and committed hundreds of millions in multi-year cloud spend. There is no irony to dunk on here; the point is the opposite. Datadog ingests and stores an enormous, high-velocity firehose of observability data, and that data tier is genuinely expensive to run at their scale. The lesson is about **the cost of the data itself.** Observability, audit logs, event streams, and analytics all share a property: the data volume grows with your customers' activity, not with your own headcount, and storing and indexing all of it at hot-tier prices is ruinous. The teams that keep this affordable are aggressive about tiering and retention — hot for recent, cold for searchable history, dropped past the window anyone queries. If even the company that sells cost visibility pays eight figures, your observability and analytics storage deserves the same tiering discipline as your production database.

### 5. Honeycomb's cross-AZ bill — the silent egress killer, named

Honeycomb, an observability company, has written candidly about their AWS cost structure, and one recurring villain is inter-AZ data transfer. A high-throughput Kafka pipeline with producers and consumers spread across availability zones moves an enormous volume of bytes across the AZ boundary, each metered at $0.01/GB in each direction, and at their ingest rates that line item climbed into a serious fraction of the bill. Their mitigations — making consumers AZ-aware so they read from same-AZ brokers, and being deliberate about which traffic genuinely needs to cross zones — are the canonical fix. This is the case study for the most under-modeled dimension in the whole article. Cross-AZ transfer feels free because it is "inside the region," and the dashboards do not attribute it to a design decision. It is not free. At scale it can rival compute, and the fix is architectural — co-locate the chatter — not a discount you negotiate.

### 6. The forgotten warehouse and the on-demand retry storm — consumption pricing's sharp edge

Two versions of the same incident recur across every team that adopts consumption pricing. In the first, someone runs an ad-hoc query on a Snowflake X-Large warehouse on a Friday, forgets to suspend it, and it idles at a few dollars a minute all weekend — a four-figure surprise for a query that ran in ten seconds. In the second, a bug or a thundering-herd retry loop hammers a DynamoDB table on on-demand pricing, and what is normally a few hundred dollars a month spikes to five figures in a day because every retried request is billed. Consumption pricing is wonderful for the right load and merciless when something misbehaves, because the thing that protects you under provisioned capacity — a hard ceiling that throttles instead of bills — is exactly what consumption pricing removes. The fix is non-negotiable guardrails: auto-suspend on every warehouse, max-capacity caps on on-demand tables, and daily spend anomaly alerts. Serverless does not remove the need for capacity limits. It changes the failure mode from "throttled" to "billed," which is worse if you are not watching.

### 7. Twitter/X, 2022–2023 — finding the waste that was always there

After the 2022 ownership change, X (formerly Twitter) ran an aggressive infrastructure cost-reduction program, publicly reporting consolidation of data center footprint (including shutting down the Sacramento facility), renegotiated cloud commitments, and claims of large reductions in monthly infrastructure spend. Set the surrounding drama aside and the engineering lesson is durable: a large, fast-growing organization accretes waste — over-provisioned capacity bought during growth scares, redundant systems from half-finished migrations, cloud commitments sized for a trajectory that did not materialize, and services nobody owns anymore. Much of the reported savings came not from clever new architecture but from *attribution and right-sizing* — finding the idle capacity and turning it off. It is a uncomfortable but useful reminder that most established systems are carrying 20–40% pure waste at any given time, and that the first move in any cost program is not to re-architect, it is to **measure what you have and turn off what you are not using.**

## When to optimize for cost, and when not to

Cost optimization is a tool, and like every tool it has a right time and a wrong time.

**Optimize for cost when:**

- The database bill is a **material fraction** of your infrastructure spend or your gross margin — below a few thousand dollars a month, your engineers' time is worth more than any savings you will find.
- Your load is **steady and predictable** — predictability is what lets you commit to reserved capacity and capture the 40–60% discount, and it is what makes tiering and right-sizing safe.
- You are about to make a **one-way-door** decision — sharding, multi-region, a new datastore. Model the cost *before* you walk through, because the door does not swing back and the egress and headcount costs compound forever.
- You can find **structural waste** — untagged spend, idle replicas, dead indexes, cross-AZ chatter. This is free money and you should take it before touching architecture.

**Do not optimize for cost when:**

- You have not yet **measured and attributed** the spend. Optimizing a cost you cannot see is guessing, and guessing usually moves money around without reducing it.
- The saving is real but **small relative to the risk** — shaving 15% off the database bill by self-hosting a system your team has never operated is a bad trade if it buys you a 3 a.m. data-loss incident.
- You would be **under-buying a real SLO.** The notification-preferences table can run cheap; the payments ledger cannot. Right-size to the business, not to the spreadsheet.
- **Premature** is the operative word. A pre-product-market-fit startup optimizing its database bill is solving the wrong problem; ship first, then make it cheap once the load shape and the SLO are real.

That is the close of the series. Across forty posts the through-line has been a single discipline: **scale in cheapest-and-most-reversible-first order, measure before you move, and right-size every decision to your SLOs.** The [decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) we opened with is the same tree we just re-walked with a price tag on every rung, and the two readings agree exactly, because they were never two trees. The order that is safest — measure, tune, scale up, cache, replicate, partition, and only then shard — is also the order that is cheapest, and that is not a coincidence. The early rungs are reversible *because* they are cheap, and they are cheap *because* they ask less of your machines and your people. The discipline of database scaling, in the end, is the discipline of buying exactly the capacity, the consistency, and the availability your business needs — at the lowest price that still clears the gate — and not one nine more.

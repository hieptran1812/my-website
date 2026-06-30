---
title: "DynamoDB at Scale: Single-Table Design and Global Tables"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "DynamoDB productizes the Dynamo paper: design the partition key to avoid hot partitions, collapse your schema into one table so a single query serves an access pattern, and accept last-writer-wins when you go multi-region active-active."
tags: ["dynamodb", "single-table-design", "global-tables", "nosql", "partition-key", "hot-partition", "database-scaling", "multi-region", "last-writer-wins", "aws", "data-modeling", "distributed-systems"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 31
---

The first time most engineers meet DynamoDB, they bring a relational brain to it and lose. They create one table per entity — `customers`, `orders`, `line_items` — because that is what you do everywhere else. They pick the primary key by uniqueness, because that is what a primary key is for. They write code that reads an order and then reads the customer and then reads the line items, because that is three foreign keys and three lookups. And then they ship, and at low traffic it all works, and at high traffic the bill is enormous, half the queries are `Scan` operations that read the whole table, and one partition is throttling at a fraction of the throughput they are paying for. None of this is DynamoDB being bad. It is DynamoDB being a different machine that punishes you for driving it like the old one.

DynamoDB is the productized descendant of Amazon's 2007 Dynamo paper, and the 2022 USENIX ATC paper ["Amazon DynamoDB: A Scalable, Predictably Performant, and Fully Managed NoSQL Database Service"](https://www.usenix.org/system/files/atc22-elhemali.pdf) is the document that explains what it actually became. The thesis of this post is two-part and blunt. First: **the partition key is your shard key, and at scale the single most important thing you do is choose it so that load spreads evenly and no single key gets hot.** Second: **because DynamoDB has no joins and cross-table queries fan out, you usually model your entire schema into one table** — overloading the partition and sort keys so that a single `Query` returns a parent and all its children in one round trip. On top of those two ideas, global tables make the whole thing multi-region active-active, and they do it at a specific, named cost: last-writer-wins conflict resolution, which can silently drop a concurrent write.

![The partition key is hashed to choose one physical partition; the sort key orders items inside that partition's item collection, and partitions auto-split as they grow](/imgs/blogs/dynamodb-global-tables-and-single-table-design-1.webp)

The diagram above is the mental model for this entire post. A partition key — here `CUST#42` — is hashed onto a 128-bit ring, and that hash chooses exactly one physical partition. Inside that partition, items are stored sorted by their sort key, so `PROFILE` comes before `ORDER#1001` comes before `ORDER#1002`. That sorted run of items sharing a partition key is an **item collection**, and it is the unit that makes DynamoDB fast: one `Query` on the partition key reads the whole collection sequentially. Partitions auto-split — by size when they cross roughly 10 GB, and by throughput when a portion runs hot — and that split behavior is exactly where the "choose your key well" lesson lives. Everything else in this post is a tour of that picture.

## Why a relational brain mis-models DynamoDB

Before the mechanics, the assumption table. These are the specific beliefs that a relational engineer carries into DynamoDB, and what each one costs once you are in production.

| What a relational brain assumes | What DynamoDB actually does |
| --- | --- |
| "One table per entity, joined at query time." | There are no joins. A query that needs two tables means two round trips, or a `Scan`, or a denormalized copy. The idiom is to collapse entities into one table. |
| "The primary key is for uniqueness." | The partition key is for *distribution*. Uniqueness is incidental; spreading load is the job. A monotonic key is uniquely terrible. |
| "I'll add a `WHERE status = 'PENDING'` later." | You cannot filter on a non-key attribute without scanning. Every query path must be designed up front as a key or a secondary index. |
| "Provision generously and throughput is fine." | Throughput is enforced *per partition*, not per table. One hot key throttles at a partition's limit while the table sits half-idle. |
| "I'll change the data model as requirements evolve." | Adding a new access pattern often means a new global secondary index and a full backfill. Evolving access patterns is the workload DynamoDB fights hardest. |

The throughline is that DynamoDB asks you to know your access patterns *before* you create the table, and to encode them into the keys. That feels backwards to anyone trained on SQL, where the schema models the data and the queries come later. In DynamoDB the queries come first and the schema is whatever shape makes those queries a single key lookup. If you cannot enumerate your access patterns, you are not ready to model the table — and that, not any technical limit, is the most common reason DynamoDB projects go wrong.

## 1. The partition key is your shard key, productized

A DynamoDB table is split across many physical partitions, each living on storage nodes, each replicated three ways across Availability Zones. When you write an item, DynamoDB takes the item's **partition key**, hashes it, and the hash deterministically selects one partition. That is identical in spirit to `hash(shard_key) % N` in a hand-rolled sharded system — DynamoDB has simply hidden the `N`, the rehashing, and the rebalancing behind a managed service. If you have read [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key), every property you learned there applies here, because the partition key *is* the shard key.

The two properties that matter most:

- **High cardinality.** The number of distinct partition-key values should be large — ideally proportional to the number of items. `user_id` across millions of users is excellent. `country_code` across 195 values is poor, because no matter how many partitions exist, all writes funnel into at most 195 buckets.
- **Uniform access.** Cardinality is necessary but not sufficient. A key can have a billion distinct values and still be hot if 90% of your traffic targets a handful of them — a celebrity's `user_id`, today's date, a single viral product. Distribution is about *how requests land*, not just how many distinct keys exist.

The optional **sort key** does not affect which partition an item lands in; it orders items *within* a partition. The partition key plus sort key together form the item's primary key. This composite design is what enables item collections and, as we will see, single-table design. A point lookup specifies both keys (`GetItem`); a range read specifies the partition key and a condition on the sort key (`Query` with `begins_with`, `>`, `between`).

It is worth being precise about what a capacity unit actually buys, because the per-partition limits only make sense once you can convert items into RCU and WCU. One **WCU** writes one item up to 1 KB per second (a 1.5 KB item costs 2 WCU; round up). One **RCU** does one *strongly consistent* read of an item up to 4 KB per second, or *two* eventually-consistent reads of that size — eventually-consistent reads are half-price, which is why defaulting reads to eventual consistency where you can tolerate it is one of the cheapest wins available. So a partition's 1,000 WCU ceiling means at most 1,000 one-KB writes per second to any single partition key, and its 3,000 RCU ceiling means at most 3,000 strongly-consistent four-KB reads (or 6,000 eventually-consistent ones). Concretely: if your hot key receives 4,000 writes per second of 1 KB items, that is 4,000 WCU against a 1,000 WCU partition ceiling — you are 4× over and *must* spread the key, no matter how much table-level capacity you provision. This arithmetic, not intuition, is how you decide whether a key needs write-sharding before you ever ship it.

### The hot-partition problem and adaptive capacity

Here is the mechanism that bites everyone, and it is the productized version of the hot-shard problem.

![A hot, low-cardinality key funnels all writes into one partition that throttles at 1,000 WCU while the table's other partitions sit idle](/imgs/blogs/dynamodb-global-tables-and-single-table-design-2.webp)

Throughput is enforced **per partition**, and a single partition has a hard ceiling: roughly **3,000 read capacity units (RCU) and 1,000 write capacity units (WCU) per second**. Those numbers are limits of the storage node, not of your provisioned capacity. Suppose you provision a table for 10,000 WCU and choose `PK = today's date` for an events table. Every write today lands on the one partition that holds today's date. That partition throttles at 1,000 WCU — one tenth of what you are paying for — while nine-tenths of your capacity sits idle on partitions holding past dates that nobody writes to. This is the canonical hot-partition failure: a `ProvisionedThroughputExceededException` (or, in on-demand mode, throttling) on a table that the dashboard says is barely utilized.

**Adaptive capacity** softens this but does not abolish it. The 2022 paper describes how DynamoDB tracks per-partition access and reacts: burst capacity lets a partition borrow unused throughput it accrued earlier, and adaptive capacity can even isolate a single hot key onto its own partition so that one item gets the full per-partition ceiling. But "the full per-partition ceiling" is still 3,000 RCU / 1,000 WCU. If your single logical key genuinely needs 5,000 WCU, no amount of adaptive capacity will give it to you — that key is bigger than a partition, and you must split it yourself.

> Adaptive capacity is a safety net for *temporary* skew, not a license to choose a bad partition key. If a key is structurally hot, the platform cannot save you; you have to spread it.

### Second-order optimization: write-sharding a hot key

When a single logical key must absorb more than 1,000 WCU, the standard technique is **write-sharding by suffix**: append a bounded random (or calculated) suffix to the partition key so that what was one key becomes several, spread across several partitions.

![Appending a bounded suffix turns one 1,000-WCU partition into ten, trading a point read for a 10-way scatter-gather read](/imgs/blogs/dynamodb-global-tables-and-single-table-design-3.webp)

Suppose you are recording live reactions to one event, `EVENT#superbowl`, and that single event needs 8,000 WCU during the game. You cannot get 8,000 WCU on one partition key. So on write, you append a suffix `0..9`:

```python
import random
import boto3

ddb = boto3.client("dynamodb")
SHARDS = 10  # fixed fan-out; 10 partitions x 1,000 WCU = 10,000 WCU headroom

def record_reaction(event_id: str, user_id: str, payload: dict) -> None:
    suffix = random.randrange(SHARDS)
    ddb.put_item(
        TableName="reactions",
        Item={
            "PK": {"S": f"EVENT#{event_id}#{suffix}"},  # sharded partition key
            "SK": {"S": f"USER#{user_id}"},
            "payload": {"S": str(payload)},
        },
    )
```

The cost is the read path. To read *all* reactions for the event you can no longer issue one `Query`; you must scatter-gather across all ten suffixes and merge:

```python
def read_all_reactions(event_id: str) -> list[dict]:
    items: list[dict] = []
    for suffix in range(SHARDS):  # 10-way scatter-gather
        resp = ddb.query(
            TableName="reactions",
            KeyConditionExpression="PK = :pk",
            ExpressionAttributeValues={
                ":pk": {"S": f"EVENT#{event_id}#{suffix}"},
            },
        )
        items.extend(resp["Items"])
    return items  # caller merges / sorts as needed
```

That is the whole trade: you bought write capacity (10× a single partition's ceiling) by paying it back on reads (a 10-way fan-out instead of a single query). The suffix count is a tuning knob — too few and you do not spread enough; too many and every read is an expensive fan-out. Use a *calculated* suffix (`hash(user_id) % SHARDS`) instead of a random one when you need to read a *specific* item back without fanning out, since you can recompute which shard it lives on. Random suffixes are simpler when you only ever read the whole collection.

### How DynamoDB stopped splits from starving hot data

There is a subtle history here that the 2022 paper tells well, and it is worth understanding because it explains why modern DynamoDB behaves better than its reputation.

![Splitting a partition by size used to halve a hot key's throughput; split-by-throughput and global admission control fixed it](/imgs/blogs/dynamodb-global-tables-and-single-table-design-4.webp)

Originally, throughput was allocated statically and a partition split was triggered by *size*. The assumption baked into that design was that keys are accessed uniformly, so splitting a partition in half by data size would split its traffic in half too. But when the workload is skewed — most traffic hitting a small hot range — splitting by size and dividing throughput proportionally could leave the hot half with *less* capacity than the whole partition had before the split. The split made throttling worse. Burst and adaptive capacity were the first mitigations: let partitions borrow idle capacity. The deeper fix was to **split on throughput, not just size** (cut the partition where the access concentration is, not where the bytes are), and then to build **Global Admission Control** — a stateful service that tracks access across the fleet and proactively moves or splits partitions before throttling occurs. This machinery is why on-demand mode can absorb sudden traffic spikes: it is the same admission control reacting in real time rather than waiting for you to re-provision.

## 2. Capacity modes: on-demand versus provisioned

DynamoDB bills throughput two ways, and the choice interacts directly with the hot-partition story.

| | Provisioned | On-demand |
| --- | --- | --- |
| You specify | RCU/WCU per second (with optional auto-scaling) | nothing; you pay per request |
| Best for | steady, predictable traffic | spiky, unpredictable, or new workloads |
| Cost shape | cheaper per unit if well-utilized | ~6–7× the per-request price of provisioned, but zero idle waste |
| Throttling | throttles when you exceed provisioned capacity (then adaptive/burst help) | scales automatically, but still bounded by the per-partition ceiling |
| Cold start | capacity is always there | a brand-new on-demand table starts with limited throughput and ramps |

The trap people fall into: on-demand is *not* a way around the per-partition limit. A hot key throttles in on-demand mode exactly as it does in provisioned mode, because the 1,000 WCU / 3,000 RCU ceiling is physical, not a billing artifact. On-demand removes the "I forgot to provision enough" failure and the "I'm paying for idle capacity" waste; it does nothing for a bad partition key. Choose on-demand for genuinely spiky or unknown workloads and for new tables where you cannot yet forecast capacity; switch to provisioned with auto-scaling once traffic is predictable enough that the per-unit savings matter. Either way, the partition key still has to be good.

## 3. Secondary indexes: serving a second access pattern

The partition key answers exactly one question: "give me the item(s) for this key." Real applications need to ask more than one. "Give me all orders with status PENDING" cannot be answered by a table keyed on customer, because status is not part of the key and DynamoDB will not filter on a non-key attribute without scanning every item. Secondary indexes are the answer, and there are two flavors with very different semantics.

| | Local Secondary Index (LSI) | Global Secondary Index (GSI) |
| --- | --- | --- |
| Partition key | same as base table | any attribute (re-keyed) |
| Sort key | different from base table | any attribute, optional |
| Created | only at table creation, never added later | added or removed any time |
| Throughput | shares the base table's capacity | its own RCU/WCU (provisioned) or on-demand |
| Consistency | supports strong consistency | eventually consistent only |
| Size limit | item collection per PK capped at 10 GB | no per-key size limit |

The GSI is the workhorse and the one you will reach for in practice. A GSI is best understood as **a differently-keyed, asynchronously-maintained projection of the base table**. You declare which attributes form the GSI's partition and sort keys; DynamoDB then maintains a second physical copy of the projected attributes, keyed that new way, and keeps it in sync by replaying base-table writes.

![A GSI re-keys each item on a different attribute and maintains it asynchronously, so a second access pattern becomes a single Query](/imgs/blogs/dynamodb-global-tables-and-single-table-design-6.webp)

Two things about that picture matter operationally. First, **a GSI is eventually consistent**: a write to the base table propagates to the GSI asynchronously, typically within milliseconds but not atomically. If you write an order and immediately query the GSI for it, you may not see it yet. Never use a GSI for a read that must reflect a just-completed write — read the base table by key for that. Second, in provisioned mode **a GSI has its own throughput, and under-provisioning it can throttle writes to the base table**, because every base write that touches indexed attributes must also be written to the GSI. A starved GSI back-pressures the table. This surprises people who think of an index as a passive read accelerator; a GSI is an active write multiplier.

```python
# Define a GSI at table creation (or add later via UpdateTable).
ddb.create_table(
    TableName="orders",
    AttributeDefinitions=[
        {"AttributeName": "PK", "AttributeType": "S"},
        {"AttributeName": "SK", "AttributeType": "S"},
        {"AttributeName": "GSI1PK", "AttributeType": "S"},  # status-keyed
        {"AttributeName": "GSI1SK", "AttributeType": "S"},  # created-at sort
    ],
    KeySchema=[
        {"AttributeName": "PK", "KeyType": "HASH"},
        {"AttributeName": "SK", "KeyType": "RANGE"},
    ],
    GlobalSecondaryIndexes=[{
        "IndexName": "GSI1",
        "KeySchema": [
            {"AttributeName": "GSI1PK", "KeyType": "HASH"},
            {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
        ],
        "Projection": {"ProjectionType": "ALL"},
    }],
    BillingMode="PAY_PER_REQUEST",
)
```

The `Projection` choice is its own optimization: `KEYS_ONLY` (just the keys, cheapest), `INCLUDE` (keys plus named attributes), or `ALL` (a full copy, most flexible and most expensive in storage and write cost). Project only what the query needs; a fat `ALL` projection doubles your storage and write amplification for attributes the index query never reads.

## 4. Single-table design: the idea people get wrong

This is the part that separates engineers who tolerate DynamoDB from engineers who get leverage out of it. The senior rule of thumb is stark:

> If your access pattern is "fetch a parent and all its children," they belong in one table, in one item collection, returned by one query.

DynamoDB has no joins. A relational model with `customers`, `orders`, and `line_items` tables would, in DynamoDB, require you to query three tables and stitch the results in application code — three network round trips for one logical read, with no transactional guarantee that you saw a consistent snapshot. The single-table technique collapses all of those entities into **one physical table** by *overloading* the partition and sort keys so that different entity types coexist, and arranging them so that everything you want to read together shares a partition key and therefore lives in one item collection.

![Overloading the partition and sort keys stores a customer's profile, orders, and line items in one item collection that a single Query returns in one round trip](/imgs/blogs/dynamodb-global-tables-and-single-table-design-5.webp)

The figure shows the canonical worked example. Every item for customer 42 shares `PK = CUST#42`. The sort key encodes both the entity type and its identity, designed so that a `begins_with` range read can slice the collection:

| PK | SK | attributes |
| --- | --- | --- |
| `CUST#42` | `PROFILE` | `name`, `email`, `tier=gold` |
| `CUST#42` | `ORDER#1001` | `total=120.00`, `status=SHIPPED` |
| `CUST#42` | `ORDER#1001#ITEM#1` | `sku=A7`, `qty=2`, `price=40.00` |
| `CUST#42` | `ORDER#1002` | `total=60.00`, `status=PENDING` |
| `CUST#99` | `PROFILE` | a different customer, a different item collection |

Now the access patterns become single queries:

- **Everything about customer 42** — `Query PK = CUST#42`. One round trip returns the profile, both orders, and the line item, sorted by SK.
- **Just the orders for customer 42** — `Query PK = CUST#42 AND begins_with(SK, "ORDER#")`. The sort-key prefix slices the collection without reading the profile.
- **One order and its line items** — `Query PK = CUST#42 AND begins_with(SK, "ORDER#1001")`. Because `ORDER#1001` sorts immediately before `ORDER#1001#ITEM#1`, the parent and its children come back together.

```python
# Fetch a customer's entire item collection in one round trip.
resp = ddb.query(
    TableName="app",  # ONE table for all entity types
    KeyConditionExpression="PK = :pk",
    ExpressionAttributeValues={":pk": {"S": "CUST#42"}},
)
items = resp["Items"]  # [PROFILE, ORDER#1001, ORDER#1001#ITEM#1, ORDER#1002]

# Or slice to just the orders, skipping the profile and line items.
orders = ddb.query(
    TableName="app",
    KeyConditionExpression="PK = :pk AND begins_with(SK, :sk)",
    ExpressionAttributeValues={
        ":pk": {"S": "CUST#42"},
        ":sk": {"S": "ORDER#"},
    },
)["Items"]
```

The second access pattern — "all PENDING orders across all customers" — cannot use `PK = CUST#42`, because it spans customers. That is what the GSI is for. You add `GSI1PK = STATUS#PENDING` (and a sort key like `GSI1SK = <created_at>`) as attributes on each order item, and the GSI re-keys them so `Query GSI1PK = STATUS#PENDING` returns every pending order, newest or oldest first, in one query. This is the **inverted-index** trick: the base table is keyed for the "by customer" pattern, and the GSI provides the orthogonal "by status" pattern. A single-table design typically ships with one to three GSIs, each serving one additional access pattern that the base key cannot.

### Writing the collection consistently: transactions and conditional writes

Reads in single-table design are a single `Query`; writes that span multiple items in the collection need more care, because a `PutItem` writes exactly one item. When you must create an order *and* its line items *and* decrement an inventory counter as a unit — all-or-nothing — DynamoDB offers `TransactWriteItems`, which applies up to 100 writes atomically across one or more tables, with optional condition checks on each.

![A DynamoDB transaction runs a two-phase prepare that checks every condition, then commits all writes or aborts the entire batch so the item collection never half-updates](/imgs/blogs/dynamodb-global-tables-and-single-table-design-10.webp)

The transaction is not free — it costs roughly 2× the WCU of the equivalent non-transactional writes, because it runs a two-phase prepare/commit internally, and it cannot span Regions — but it gives you the multi-item atomicity that single-table reads otherwise lack. The figure above is the contract: the prepare phase checks every item's condition, and only if all of them pass does the commit apply; one failed condition aborts the whole batch with nothing written, so the collection is never left half-updated.

```python
# Create an order and a line item atomically, only if the order does not exist.
ddb.transact_write_items(
    TransactItems=[
        {
            "Put": {
                "TableName": "app",
                "Item": {
                    "PK": {"S": "CUST#42"},
                    "SK": {"S": "ORDER#1003"},
                    "total": {"N": "75.00"},
                    "status": {"S": "PENDING"},
                    "GSI1PK": {"S": "STATUS#PENDING"},   # feed the status GSI
                    "GSI1SK": {"S": "2026-06-30T10:00Z"},
                },
                "ConditionExpression": "attribute_not_exists(SK)",  # no clobber
            }
        },
        {
            "Put": {
                "TableName": "app",
                "Item": {
                    "PK": {"S": "CUST#42"},
                    "SK": {"S": "ORDER#1003#ITEM#1"},
                    "sku": {"S": "B2"},
                    "qty": {"N": "3"},
                },
            }
        },
    ]
)
```

The `ConditionExpression` is doing real work even outside transactions: `attribute_not_exists(SK)` makes the write idempotent and prevents two concurrent callers from both "creating" the same order. Conditional writes are how you implement optimistic concurrency in DynamoDB — read an item with its version attribute, write back with `ConditionExpression: "version = :expected"`, and let the write fail if someone else changed it first. This is the single-table analogue of a relational `UPDATE ... WHERE version = ?`, and it is the correct way to guard a mutable item against lost updates *within* a Region. (Across Regions, recall, even a conditional write is subject to the last-writer-wins reconciliation we cover next.)

### Why people get single-table design wrong

Three failure modes, all common:

1. **They never enumerate access patterns.** Single-table design is *derived* from a complete list of "the application needs to read X by Y" statements. Skip that step and you end up retrofitting GSIs forever, each with a backfill, because every new screen needs a key shape you did not plan for. The design discipline is to write the access-pattern list first and let the keys fall out of it.
2. **They fight the model with normalization instincts.** They store an order without denormalizing the few customer fields the order view needs, then do a second `GetItem` for the profile on every order render. In DynamoDB, duplicating a handful of attributes to avoid a round trip is correct, not a sin. Storage is cheap; round trips and missing transactional snapshots are not.
3. **They over-rotate and put genuinely unrelated entities in one table for its own sake.** Single-table design is a tool for entities that are *queried together*. If two entity types are never read in the same query and never need to be transactionally consistent, forcing them into one table buys nothing and complicates the key scheme. The goal is fewer round trips for real access patterns, not table-count minimization as an aesthetic.

The honest counterpoint, which the DynamoDB community argues about endlessly: single-table design optimizes for *known, stable* access patterns at the cost of *flexibility*. If your access patterns are still churning weekly, a looser multi-table design (or a different database) may serve you better until they stabilize. Single-table design is the right answer when you know your queries and you need them to be cheap and fast at scale — not a mandate to apply on day one of an exploratory product.

## 5. Global tables: multi-region active-active

Global tables turn a DynamoDB table into a set of replicas across AWS Regions, all of which accept reads and writes. The current version, **2019.11.21**, is genuinely active-active: there is no primary Region. An application in `us-east-1` reads and writes its local replica at single-digit-millisecond latency; an application in `eu-west-1` does the same against the European replica; and DynamoDB asynchronously replicates every change to every other replica.

![Each Region's replica accepts local reads and writes, and every change replicates any-to-any to the other Regions in roughly a second](/imgs/blogs/dynamodb-global-tables-and-single-table-design-7.webp)

The replication is **asynchronous and any-to-any**, propagating through DynamoDB Streams typically in **about a second**. That sub-second figure is the win: a user in Singapore reads from the `ap-southeast-1` replica without a cross-Pacific round trip, and a write in `us-east-1` shows up in Tokyo a beat later. The two headline use cases follow directly:

- **Low-latency global reads (and writes).** Put a replica near each population of users; everyone gets local-Region latency. This is the most common reason teams adopt global tables.
- **Regional failover / disaster recovery.** If a Region degrades, you route traffic to another replica that already holds a near-current copy of the data, with no promotion step and no data-restore window. Recovery is a routing change, not a database operation.

### The cost: last-writer-wins can silently lose a write

Active-active multi-master replication has an unavoidable problem: two Regions can write the same item before replication reconciles them. DynamoDB resolves these conflicts with **last-writer-wins (LWW)**, and you need to understand exactly what that means before you trust it with anything financial.

![Two Regions edit the same item before replication catches up; the higher timestamp survives and the other edit is silently discarded](/imgs/blogs/dynamodb-global-tables-and-single-table-design-8.webp)

Each item carries a hidden system timestamp set at write time. When replicas exchange conflicting versions of the same item, the version with the **higher timestamp wins**, on a per-item basis, implemented as a conditional write that only applies an incoming version if its timestamp exceeds the stored one. All replicas eventually converge to that one winning version. The trap is in the word *silently*: the losing write returned success to its caller. There was no error, no conflict exception, no merge. The data is simply gone, overwritten by a write that happened to carry a later timestamp.

Walk the timeline in the figure. At `t=0`, `us-east-1` sets `qty=5` (timestamp 100). At `t=20ms`, before replication has propagated anything, `eu-west-1` sets `qty=8` (timestamp 120) on the same item. At `t=300ms` the replicas exchange both versions. LWW compares timestamps: 120 > 100, so `qty=8` wins everywhere. The `qty=5` write — which succeeded, which the US application believes it made — is discarded across all replicas. If those two writes were two ATMs decrementing the same balance, you just lost a debit.

The defensive patterns, in rough order of preference:

- **Partition writes by Region.** The cleanest fix is to ensure a given item is only ever written in one Region — for example, pin each user's writes to their home Region and let other Regions read. If two Regions never write the same item concurrently, LWW never fires destructively. This is the design most production global-table deployments actually use.
- **Use atomic counters and conditional writes for shared-mutable state.** `ADD` on a numeric attribute and `ConditionExpression` guards survive better than blind `PutItem` overwrites, though they are still subject to cross-Region races on the same item.
- **Do not put a strongly-consistent invariant on a globally-writable item.** If the data genuinely requires "no two writers ever lose an update" — bank balances, inventory you cannot oversell — then either keep that item single-Region-writable or use a system designed for it (Spanner-style external consistency, or a single-writer log). DynamoDB global tables give you availability and locality; they do not give you cross-Region linearizability. For the spectrum of guarantees here, see [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and the leaderless-replication mechanics in [Cassandra and DynamoDB: a leaderless deep-dive](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive).

> Last-writer-wins is not a bug; it is the price of writing in two places at once without a coordinator. The question is never "is LWW safe?" but "which of my items can two Regions write at the same instant?" Design so the answer is "none of the ones that matter."

This is the same tension that runs through every multi-region system; the broader patterns live in [multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture) and the tradeoff knobs in [tunable consistency at scale](/blog/software-development/database-scaling/tunable-consistency-at-scale).

## 6. When DynamoDB fits — and when it fights you

Put the whole decision on one grid.

![DynamoDB rewards known, key-based access at scale and punishes ad-hoc, relational, or analytic workloads](/imgs/blogs/dynamodb-global-tables-and-single-table-design-9.webp)

The matrix is the closing argument. DynamoDB is exceptional in the top row — known, key-based lookups at scale — where it delivers single-digit-millisecond latency, serverless operations, and effectively unbounded throughput as long as your keys spread. It is poor in the lower rows: ad-hoc filters and joins (there is no `WHERE` on non-key attributes and no `JOIN`), analytics and aggregation (no `GROUP BY`; you export to S3 and query with Athena), and evolving access patterns (each new pattern is a new GSI and a backfill).

**Reach for DynamoDB when:**

- Your access patterns are **known and enumerable** up front, and stable enough to encode into keys.
- You need **predictable single-digit-millisecond latency** at **very high scale** (millions of requests per second) with **no operational burden** — no servers, no failover scripts, no vacuum, no resharding projects.
- Your reads are **key-based** — fetch by ID, fetch a parent and its children, fetch a list by a known secondary key.
- You want **multi-region active-active** with minimal effort and can design so concurrent same-item writes do not matter (see the LWW discussion above).
- Your write volume is **spiky or unpredictable** and on-demand's auto-scaling is worth the per-request premium.

**Skip DynamoDB (or pair it with something else) when:**

- You need **ad-hoc queries** — arbitrary `WHERE` clauses, analyst-driven exploration, "just add a filter." That is a relational database's job, or a search engine's.
- You need **joins or aggregations** as a first-class query shape. Modeling those into single-table design is possible but painful, and any genuinely new aggregation is a re-modeling exercise.
- Your **access patterns are still churning**. The single-table model's strength — keys frozen to known patterns — becomes a liability when the patterns change weekly.
- You need **strong cross-region consistency** on globally-writable items. Global tables are eventually consistent with LWW; if you cannot tolerate a lost concurrent write, this is the wrong tool for that item.
- You are doing **analytics or reporting**. Stream the table to S3 and run Athena, Redshift, or Spark there; do not try to make DynamoDB an OLAP store.

## Case studies from production

### 1. The events table that throttled at 10% utilization

A team built an audit-log table with `PK = log_date` (one partition key per calendar day) provisioned at 12,000 WCU. The first wrong hypothesis was under-provisioning, so they doubled to 24,000 WCU and it got *no* better. The actual root cause: every write on a given day targets that day's single partition, which throttles at 1,000 WCU regardless of table-level provisioning. They were paying for 24,000 and could use 1,000. The fix was write-sharding the date key with a `0..29` suffix (`log_date#shard`), spreading the day's writes across 30 partitions, and scatter-gathering on the rare full-day read. Throttling vanished and they cut provisioned capacity back to 8,000. The lesson: a hot-partition throttle looks exactly like under-provisioning on the table-level dashboard, and throwing capacity at it does nothing.

### 2. The GSI that throttled the base table

An orders service added a GSI keyed on `status` to power an operations dashboard. In provisioned mode they generously sized the base table but left the GSI at default low capacity, reasoning that the dashboard was low-traffic. Within hours, *order writes* started throttling. The cause: every order write also writes to the GSI, and the under-provisioned GSI's throttling back-pressured the base table's writes. The fix was to provision the GSI for the *write* rate of the base table (not the read rate of the dashboard) or switch the table to on-demand so both scaled together. The lesson, restated: a GSI is a write multiplier, not a passive read accelerator, and its capacity is gated by base-table write volume.

### 3. The relational port that became a Scan farm

A migration lifted a PostgreSQL schema directly: one DynamoDB table per SQL table, primary keys copied verbatim. Every list view became a `Scan` with a `FilterExpression`, because there was no key to query on — `Scan` reads the entire table and *then* filters, charging RCU for every item read, not every item returned. The bill was an order of magnitude over forecast and latency grew linearly with table size. The fix was a six-week re-model into a single table with overloaded keys and two GSIs, derived from a written list of the application's twelve access patterns. After the re-model, every screen was a `Query` or `GetItem`, the bill dropped roughly 90%, and latency went flat. The lesson: you cannot port a relational schema to DynamoDB; you port the *access patterns*.

### 4. The double-debit across two Regions

A wallet service ran global tables across `us-east-1` and `eu-west-1` with both Regions accepting balance writes, because "active-active" sounded like the safe choice. A user with two open sessions triggered near-simultaneous debits from both Regions. Both succeeded locally; LWW kept the one with the higher timestamp; the other debit vanished and the balance was wrong by one transaction. There was no error to alert on — both writes returned success. The fix was to pin each account's writes to a home Region (reads still served locally everywhere) so the same balance item is only ever written in one place, with the other Region used purely for low-latency reads and failover. The lesson: active-active does not mean "write the same item anywhere safely"; it means "write *different* items in different Regions," and you must design which Region owns each mutable item.

### 5. The cold-start throttle on a new on-demand table

A team launched a feature behind a brand-new on-demand table and pushed a large backfill at it on day one. The backfill throttled immediately, which made no sense to them — on-demand is supposed to scale infinitely. The cause: a fresh on-demand table starts with a limited initial throughput and ramps up as DynamoDB observes sustained traffic and splits partitions; a sudden cold spike outruns the ramp. The fix was either to pre-warm the table by ramping the backfill rate gradually, or (in provisioned mode) to temporarily provision high capacity for the backfill window and then drop it. The lesson: on-demand is elastic but not instantaneous; a cold table cannot absorb an instant flood, and large initial loads need a warm-up ramp.

### 6. The item-collection size limit nobody planned for

A team used an LSI and modeled a parent with an unbounded number of children all under one partition key (one `tenant_id` accumulating millions of events). Writes started failing with an item-collection-size error. The cause: with an LSI present, the total size of all items sharing one partition-key value is capped at 10 GB, and one heavy tenant blew past it. The fix was to drop the LSI in favor of a GSI (which has no per-key size cap) and to shard the hot tenant's events by time bucket so no single partition key accumulated unboundedly. The lesson: the 10 GB item-collection limit only applies when an LSI exists, and unbounded item collections under one key are a modeling smell regardless — partition large collections by a time or hash bucket.

### 7. The eventually-consistent read-after-write on a GSI

A checkout flow wrote an order, then immediately queried a GSI to display "your recent orders," and intermittently the just-placed order was missing. The wrong hypothesis was a bug in the write. The actual cause: GSIs are eventually consistent, and the read raced the asynchronous index propagation. The fix was to render the just-placed order from the base-table write the application already had in hand (or to `GetItem` the base table by key, which *can* be strongly consistent), reserving the GSI for the broader historical list where staleness of a few milliseconds is invisible. The lesson: never gate a read-after-write correctness requirement on a GSI; the base table by key is your strongly-consistent path, the GSI never is.

### 8. The analytics query that should never have been DynamoDB

A growth team needed "revenue by product category by week," and tried to compute it by scanning the orders table nightly. The scan read every item, the RCU cost dwarfed the production read traffic, and the job took hours and grew with the data. The fix was to stop: enable a DynamoDB-to-S3 export (or stream via Kinesis), land the data in S3 as Parquet, and run the aggregation in Athena. DynamoDB went back to serving the operational read path it is good at, and analytics moved to a columnar store built for `GROUP BY`. The lesson maps directly to [OLTP versus OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores): DynamoDB is an OLTP key-value/document store, and forcing OLAP aggregations onto it is paying transactional prices for analytical work.

### 9. The lost update that a conditional write would have caught

A loyalty service read a user's points balance, added the earned points in application code, and wrote the new total back with a plain `PutItem`. Under concurrency — two events for the same user landing within milliseconds — both reads saw the same starting balance, both computed a new total from it, and the second write clobbered the first. Points silently went missing, and the bug was invisible in single-threaded testing. The wrong hypothesis was a replication or eventual-consistency issue; it was none of those, just a classic read-modify-write race entirely within one Region. The fix was twofold: use an atomic `UpdateItem` with `ADD points :earned` (which increments server-side without a read-modify-write window), or, where the new value depends on the old in a way `ADD` cannot express, carry a `version` attribute and write with `ConditionExpression: "version = :expected"` so the losing write fails loudly and retries. The lesson: DynamoDB will happily let two writers clobber each other unless you ask it not to; atomic updates and conditional writes are the guardrails, and they cost almost nothing compared to the bug.

## Closing: the two decisions that decide everything

If you remember two sentences from this post, make them these. **The partition key is your shard key, so choose it for distribution and design away hot keys** — high cardinality is not enough; uniform access is the real requirement, and write-sharding is your escape hatch when a single logical key is bigger than a partition. **Model your access patterns into one table** so that a single `Query` returns a parent and its children from one item collection, and add a GSI for each orthogonal pattern the base key cannot serve. Global tables extend that single table across Regions for locality and failover, at the named price of last-writer-wins — which is safe exactly when you ensure no item that matters is written concurrently in two Regions.

DynamoDB is not a relational database with a different API. It is a different machine with a different contract: tell it your access patterns up front, encode them into keys, and it will give you predictable single-digit-millisecond latency at any scale with no operational burden. Bring relational habits and it will hand you a `Scan` farm, a throttled hot partition, and a surprising bill. The database is the same in both cases. The only variable is whether you designed for the machine you actually have.

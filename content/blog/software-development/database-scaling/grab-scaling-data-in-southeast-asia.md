---
title: "Scaling Data at a Super-App: Grab Across Southeast Asia"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "How Grab scales a polyglot storage fleet across many countries with a high-write geospatial and transactional workload — and why a super-app is really N businesses held together by database-per-service, a CDC event backbone, and per-country deployment."
tags: ["grab", "super-app", "polyglot-persistence", "database-per-service", "change-data-capture", "kafka", "dynamodb", "data-residency", "data-lake", "database-scaling"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 29
---

The fastest way to misunderstand a super-app is to think of it as one product with a big database. Open the Grab app and you can hail a car, order lunch, pay a merchant, top up a wallet, and watch a driver's marker slide across a map in real time. Those are not features of one system. They are separate businesses — transport, deliveries, financial services, mapping — that happen to share a login screen and a brand. Each has its own read and write shape, its own consistency requirements, its own regulator, and in Grab's case its own country-by-country footprint across Singapore, Indonesia, Malaysia, Vietnam, the Philippines, Thailand, and more.

When you internalize that, the storage architecture stops looking like a puzzle and starts looking like the only thing that could possibly work. You cannot run a payments ledger and a stream of driver GPS pings out of the same engine with the same operational model. The ledger wants strong transactional guarantees and careful schema evolution; the location stream wants to absorb an enormous, bursty write volume and forget most of it within minutes. A single database tuned for one is actively hostile to the other. So Grab, like most super-apps that survived their growth curve, did not scale one database. They scaled a *fleet* of stores, gave each service its own, wired them together with an event backbone, and stamped the whole pattern out per country.

![Grab super-app service-to-store map showing each business owning its own polyglot store](/imgs/blogs/grab-scaling-data-in-southeast-asia-1.webp)

The diagram above is the mental model for the entire article: every business owns its own store, and the store *type* is chosen from that business's access pattern rather than a company-wide standard. Transport and maps push their high-write, geospatial load into DynamoDB; GrabPay's wallet and ledger and the deliveries order system sit on Aurora/MySQL where transactions are first-class; Redis fronts the hot session reads. There is no single "Grab database." There is a polyglot fleet, and the interesting engineering is in how those stores stay decoupled while still feeding a coherent analytics and machine-learning platform.

A note on sourcing before we go further. The shape of Grab's platform — the **Coban** streaming team and its self-served Kafka, **Debezium**-based change data capture on Kafka Connect, the **Trailblazer** Spark Structured Streaming application that lands MySQL binlogs in the data lake, the **Caspian** team synchronizing MySQL/Aurora/DynamoDB into Kafka, and a large multi-tenant **Presto** estate — comes from Grab's own engineering blog. I will treat those component names and the architecture as reported, and I will keep numbers qualitative or tied to the year Grab published them, because exact figures move and were specific to a moment in time. Where I show code, it is a faithful *model* of the pattern, not Grab's source.

## Why a super-app is a different storage problem

Most teams scale a database by sliding along a single axis: small instance, big instance, read replicas, then sharding. That axis assumes one workload. A super-app breaks the assumption on day one, because the thing you are scaling is not a database — it is an *organization* of many teams, each shipping a different workload into production every week. The table below is the contrast that drives every decision later in the post.

| Question | Single-product instinct | Super-app reality |
| --- | --- | --- |
| What are we scaling? | One database under one workload | N service-owned stores under N workloads |
| Who picks the engine? | A platform standard, once | Each service, from its own access pattern |
| What couples teams? | A shared schema everyone edits | An event contract on a shared log |
| What does an analyst query? | The production database | A data lake fed by change capture, never the source |
| Where does data live? | One region | Per country, with residency rules |
| What dominates the budget? | Compute for the big instance | Write volume, hot tiers, and cross-AZ traffic |

The right column is harder. You have to operate many stores, run an event backbone, and build a data platform that does not reach into anyone's production database. The payoff is that each piece is *legible and independent*: a delivery-team schema migration cannot block the payments team, a spike in driver pings cannot throttle the wallet, and a country's regulator can be satisfied without re-architecting the world. The rest of this article is a tour of how those pieces fit, in the order you would actually build them.

> A super-app is not one system that got big. It is many systems that agreed on a login screen, a brand, and an event bus. Scale the agreement, not the database.

## 1. N businesses, N stores: database-per-service

**Senior rule of thumb: the unit of database scaling in a super-app is the service, not the company. Give each service the store it would have chosen if it were a startup, then make them talk through events instead of through tables.**

In the earliest version of almost every company, there is one database and everyone writes to it. It is the right call at first — a shared schema is the cheapest way for a tiny team to move fast. It becomes the wrong call at exactly the moment two teams want to change the same tables for different reasons on the same afternoon. At a super-app's growth rate, that moment arrives early and never leaves.

![Before-and-after of a shared database versus database-per-service ownership](/imgs/blogs/grab-scaling-data-in-southeast-asia-2.webp)

The figure makes the trade explicit. On the left, a shared database couples every team to one schema: a migration blocks unrelated teams, one hot table throttles everyone behind the same connection pool and lock manager, and a bad query has no blast-radius containment — it degrades the whole app. On the right, database-per-service breaks those couplings. Each service owns its store, evolves its schema on its own cadence, and — crucially — picks the *engine* that fits its workload. The cost you pay is that data is now spread across many stores, so any cross-service read or analytics query needs a new mechanism. That mechanism is the event backbone in section 3; the rest of this section is about why the decomposition is worth it.

The decomposition is not free philosophy; it is a concrete failure-isolation property. When deliveries and payments share a database, a lock-heavy promotion campaign in deliveries can starve the payment path of connections, and a customer cannot pay for the food they just ordered. When they are separate, the deliveries store can be on fire and payments keeps clearing transactions. For a business where the payment *is* the revenue, that isolation is not a nicety — it is the reason the company keeps making money during an incident.

There is a second-order trap here worth naming. Database-per-service is often sold as "microservices," and teams over-rotate into dozens of tiny services each with a trivial store, paying enormous coordination cost for services that were never going to fail independently. The discipline is to draw service boundaries around *businesses that genuinely have different data shapes and failure domains* — transport, deliveries, payments, identity, maps — not around every noun in the domain model. Grab's polyglot fleet maps onto real business lines, which is why it scales the org as much as the data. If you want the decision framework for when a separate store is justified at all, this blog's [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) walks the same fork.

## 2. Polyglot by workload, not by fashion

**Senior rule of thumb: choose the store from the access pattern and the consistency requirement, in that order. The data's business importance tells you how careful to be operationally; it does not tell you which engine to use.**

"Polyglot persistence" gets a bad reputation because people read it as "use a trendy database for every service." That is not the idea. The idea is that a handful of workload *shapes* recur across a large system, and each shape has a store class that is simply better at it. The skill is recognizing the shape, not collecting databases. Grab's fleet — Aurora/MySQL, DynamoDB, Redis, Kafka, and a Parquet data lake queried by Presto — is a near-canonical mapping of the five shapes a super-app actually produces.

![Matrix mapping workload shape to store class and the reason it wins](/imgs/blogs/grab-scaling-data-in-southeast-asia-3.webp)

Read the matrix by row, by workload. A **transactional** workload — a wallet ledger, an order with line items, anything where two writes must be all-or-nothing and you need real joins — goes on Aurora/MySQL, because strong, ACID semantics and a mature relational engine are exactly what it needs. A **high-write key-value** workload — driver location, a feature store keyed by user id, anything that is "look up by key, write constantly, never join" — goes on DynamoDB, because it scales writes flatly with no node babysitting and tunable consistency where you can trade a little freshness for throughput. A **hot read** that would otherwise hammer a primary goes in Redis with a TTL. **Events** — the change stream that decouples everything — live on Kafka, an ordered, replayable log. And **analytics** lands in the data lake: columnar Parquet on object storage, cheap per gigabyte, eventually consistent, scanned by Presto.

The same logic, with the full decision tree and the failure modes of each choice, is the subject of [polyglot persistence: choosing the right store](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store). Here is the compact version Grab's fleet embodies:

| Workload | Store | Consistency | Why it wins |
| --- | --- | --- | --- |
| Transactional (ledger, orders) | Aurora / MySQL | Strong, ACID | Multi-row transactions, joins, mature ops |
| High-write KV (location, features) | DynamoDB | Tunable | Flat write scaling, no instances to manage |
| Hot read (sessions, config) | Redis | Cache + TTL | Sub-millisecond reads, offloads the primary |
| Events (the change stream) | Kafka | Ordered log | Decouples producers from consumers, replayable |
| Analytics (BI, ML features) | Data lake (Parquet) | Eventual | Cheap, columnar, scales scans independently |

The discipline that makes this work is resisting two temptations. The first is forcing a workload onto the "main" database because that is where the team is comfortable — the way a payments-shaped problem ends up jammed into DynamoDB, or a location-shaped firehose ends up melting a MySQL primary. The second is the opposite: adopting a sixth or seventh engine for a workload one of the existing five already handles, just because a new database is exciting. Five well-understood stores you can operate in your sleep beats nine you cannot. Grab kept the list short and the boundaries clear, and that is a feature, not a limitation.

## 3. The event backbone: capture once, fan out forever

**Senior rule of thumb: once you have database-per-service, the most important system you build is not any of the databases — it is the log that carries their changes. Every cross-service read and every analytics query should subscribe to that log, never query the source store.**

The instant you split one database into many, you create a problem: data that used to be one JOIN away is now in someone else's store, behind someone else's API, with its own scaling limits. The naive fix — have services call each other synchronously, or worse, let analytics reach directly into production databases — recreates all the coupling you just removed, plus adds latency and a load source the owning team did not plan for. The correct fix is a change-data-capture event backbone.

![Graph of the CDC event backbone capturing service-store changes into Kafka and fanning out to consumers and the data lake](/imgs/blogs/grab-scaling-data-in-southeast-asia-4.webp)

The figure is the heart of the architecture. Each service store publishes its *changes* — not its data on request, but every committed change — exactly once onto a shared log. For Grab's relational stores that capture is **Debezium** running on Kafka Connect, tailing the MySQL/Aurora binlog; for DynamoDB it is DynamoDB Streams. Both land on Grab's **Kafka** backbone, the self-served platform the **Coban** team operates. From there, the change stream fans out to two kinds of consumers. Other services subscribe to the events they care about (an order-created event, a payment-cleared event) and update their own stores. And **Trailblazer**, a Spark Structured Streaming application, sinks the same stream into the data lake, where hourly compaction turns the raw change log into queryable Presto tables. The **Caspian** team's work to synchronize MySQL, Aurora, and DynamoDB into Kafka is precisely this backbone generalized across the polyglot fleet.

Notice what this buys you. Analytics never touches a production database — it reads the lake, which is fed asynchronously by the log, so a heavy query cannot slow down a driver's payment. Services stay decoupled — a consumer that falls behind or crashes does not block the producer; it just resumes from its offset when it recovers. And because Kafka is a *replayable* log, you can rebuild a downstream store from scratch by re-reading history, which is the single most underrated operational superpower in this design.

The cleanest way to *produce* those events from a transactional service is the **transactional outbox** pattern: write your business change and an outbox record in the same database transaction, then let CDC ship the outbox. That guarantees the event exists if and only if the data change committed — no dual-write race where the database commits but the Kafka publish fails, or vice versa. Here is the producer side as a faithful model:

```python
# Service-owned write with the transactional outbox pattern.
# The business row and the event are committed atomically; CDC ships the event.
import json
import uuid
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://svc:***@orders-aurora:3306/orders")

def place_order(user_id: str, items: list[dict]) -> str:
    order_id = str(uuid.uuid4())
    event = {
        "type": "order.placed.v1",
        "order_id": order_id,
        "user_id": user_id,
        "items": items,
        "country": "SG",
    }
    # One transaction: the order AND the outbox row commit together, or neither does.
    with engine.begin() as tx:
        tx.execute(
            text("INSERT INTO orders (id, user_id, status) "
                 "VALUES (:id, :uid, 'PLACED')"),
            {"id": order_id, "uid": user_id},
        )
        tx.execute(
            text("INSERT INTO outbox (id, aggregate, payload, created_at) "
                 "VALUES (:id, 'order', :payload, NOW())"),
            {"id": str(uuid.uuid4()), "payload": json.dumps(event)},
        )
    return order_id
```

The service never publishes to Kafka itself. Debezium tails the binlog, sees the `outbox` insert, and emits it to a Kafka topic; consumers and Trailblazer take it from there. The full treatment of why this beats a naive dual write — including the exactly-once-versus-at-least-once subtleties and idempotent consumers — is in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern). The consumer side is deliberately boring:

```python
# A downstream consumer. Idempotent by event id; commits offsets only after the
# local store write succeeds, so a crash replays rather than loses the event.
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "orders.events",
    bootstrap_servers="coban-kafka:9092",
    group_id="delivery-assignment",
    enable_auto_commit=False,            # we commit only after a successful write
    auto_offset_reset="earliest",
)

for msg in consumer:
    event = json.loads(msg.value)
    if event["type"] != "order.placed.v1":
        continue
    if already_processed(event["order_id"]):  # idempotency guard
        consumer.commit()
        continue
    assign_driver(event["order_id"], event["country"])
    mark_processed(event["order_id"])
    consumer.commit()                    # advance the offset last
```

### The second-order win: a row change becomes a lake table on its own

The non-obvious benefit of CDC is that your analytics platform stops being a fleet of brittle batch ETL jobs that hammer production every night. Instead, the same change stream that decouples services *also* populates the warehouse, in near-real time, without a single query against the source database.

![Timeline of a single row change flowing through capture, Kafka, raw Parquet, and compaction into a queryable Presto table](/imgs/blogs/grab-scaling-data-in-southeast-asia-5.webp)

Follow one row down the timeline. At `t0` a row commits in Aurora. Within milliseconds Debezium reads the binlog event and publishes it to a Kafka topic. A streaming job — Trailblazer — writes the raw change into a Parquet "real-time" bucket near-real-time. Then, on a schedule (hourly is typical), a compaction job merges the accumulated change log, de-duplicates, and materializes a clean Presto table for analysts and ML pipelines. The source MySQL was never queried by a batch job; it only ever wrote its binlog, which it was going to write anyway. This is the move that let Grab's Presto estate grow to a very large multi-tenant platform — by Grab's 2020 account, on the order of hundreds of user groups running hundreds of thousands of queries a day over thousands of tables and up to a petabyte scanned daily — without that read load ever landing on the operational databases. If you are separating your reporting workload from your operational one, the same principle drives [separating OLTP from OLAP](/blog/software-development/database-scaling/separating-oltp-from-olap).

The cost to respect here is correctness under failure. CDC connectors do fail — a Debezium connector dies, the binlog rotates, a topic lags — and when they do, your lake silently falls behind or, worse, drops changes. Grab's published experience includes exactly this: connectors failing, fewer messages arriving than expected, and duplicate change records that must be resolved during compaction. The lesson is that a CDC pipeline needs the same monitoring rigor as a primary database: lag alerts, row-count reconciliation between source and lake, and a compaction step that is idempotent so a replayed or duplicated change converges to the same result.

## 4. The high-write path: location pings without hot partitions

**Senior rule of thumb: in a geospatial workload, the partition key is the whole ballgame. A natural-looking key like "city" concentrates a metro area's entire write firehose onto one physical partition, and no amount of provisioned capacity will save you.**

Driver and rider location is the workload that most clearly separates a super-app from a normal app. Tens of thousands of active drivers in a single city, each emitting a GPS ping every few seconds, is a sustained, bursty, write-dominated stream. It maps perfectly onto DynamoDB's strengths — key-value, high write, no joins — but only if you respect DynamoDB's one unforgiving rule: throughput is allocated *per partition*, and a single partition tops out around 1,000 write capacity units per second and 3,000 read capacity units per second. A hot partition is a design problem, not a capacity problem; provisioning more table-level capacity does nothing if all the writes hit one key.

![Before-and-after of a single hot location key versus a write-sharded key spreading load across N partitions](/imgs/blogs/grab-scaling-data-in-southeast-asia-6.webp)

The figure is the canonical fix. On the left, a partition key of `city#SG` sends every Singapore ping to one partition; once the city's write rate crosses the per-partition ceiling, DynamoDB throttles, and adaptive capacity (which can eventually "split for heat") reacts too slowly for a sudden surge. On the right, you append a shard suffix — `city#SG#7` — computed from the driver id, spreading the same writes across `N` partitions so each stays comfortably under the ceiling. You have multiplied effective write throughput by `N`. The tax you accept is on reads: a query for "all drivers in Singapore" must now scatter across all `N` shards and gather the results. For a write-heavy, point-lookup-light workload like location ingestion, that is exactly the right trade. The general theory of this failure and its mitigations is in [hot partitions and hot rows](/blog/software-development/database-scaling/hot-partitions-and-hot-rows), and the single-table modeling it builds on is in [DynamoDB global tables and single-table design](/blog/software-development/database-scaling/dynamodb-global-tables-and-single-table-design).

Here is the write path and the scatter-read as a working model:

```python
# Write-sharding a high-write location key across N partitions.
import boto3
import hashlib
import time

ddb = boto3.client("dynamodb", region_name="ap-southeast-1")
N_SHARDS = 16          # tune to peak per-city write rate / ~1000 WCU per partition

def _shard(driver_id: str) -> int:
    # Deterministic so a driver's pings land on one shard (cheap point reads),
    # uniform so the city's load spreads evenly across partitions.
    h = hashlib.blake2b(driver_id.encode(), digest_size=2).digest()
    return int.from_bytes(h, "big") % N_SHARDS

def put_ping(city: str, driver_id: str, lat: float, lng: float) -> None:
    pk = f"city#{city}#{_shard(driver_id)}"      # spreads writes across N partitions
    ddb.put_item(
        TableName="driver_location",
        Item={
            "pk":        {"S": pk},
            "sk":        {"S": f"driver#{driver_id}"},
            "lat":       {"N": str(lat)},
            "lng":       {"N": str(lng)},
            "geohash":   {"S": _geohash(lat, lng, precision=6)},  # ~1.2 km cell
            "updated_at":{"N": str(int(time.time()))},
            "ttl":       {"N": str(int(time.time()) + 120)},      # stale pings expire
        },
    )

def drivers_in_city(city: str) -> list[dict]:
    # Scatter across every shard, gather. The cost of write-sharding shows up here.
    out: list[dict] = []
    for shard in range(N_SHARDS):
        resp = ddb.query(
            TableName="driver_location",
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": {"S": f"city#{city}#{shard}"}},
        )
        out.extend(resp["Items"])
    return out
```

Two details carry the design. The TTL attribute lets DynamoDB expire stale pings automatically, so the table does not grow without bound and you are not paying to store a location from an hour ago that no one will ever read — a quiet but real cost win at this volume. And the `geohash` attribute encodes location into a string whose prefix length corresponds to a grid cell, so a proximity search ("drivers near this pickup") becomes a prefix query rather than a full scan. Grab's real-time matching is far more sophisticated than this sketch, but the bones — sharded write keys plus geohash buckets, with hot data expiring fast — are the standard shape for the problem. Matching itself, where the freshest data must be milliseconds old, leans on the in-memory tier; the durable store absorbs the firehose and the TTL keeps it lean.

## 5. Many countries, one platform: residency and deployment

**Senior rule of thumb: treat each country as a deployment target, not a row in a table. Regulated personal data is bound to its country; only de-identified aggregates are allowed to cross a border. Design the boundary explicitly or a regulator will design it for you.**

Operating across Singapore, Indonesia, Malaysia, Vietnam, the Philippines, and Thailand is not a localization problem — it is a data-architecture problem. Several of these jurisdictions have data-residency and data-protection rules that constrain where personal data may physically live and how it may move. The naive "one global database with a `country` column" model is a non-starter the moment a regulator asks you to prove that Indonesian users' personal data is stored in-region and is not silently replicated to a cluster in another country.

![Grid of per-country stacks with a data-residency boundary keeping PII in-region and only aggregates crossing to a shared lake](/imgs/blogs/grab-scaling-data-in-southeast-asia-7.webp)

The figure shows the boundary made concrete. Each country runs its own regional stack — its own services and its own stores — and regulated personal data stays resident in that country's region. What crosses the border is not raw rows but *de-identified aggregates*: data that has been stripped or anonymized before it leaves, feeding a regional or global lake used for analytics and modeling. The residency boundary is a first-class architectural object, not a `WHERE` clause you hope everyone remembers to add. This is the geo dimension of the same problem covered in [multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture); the residency twist is that the regions are not just for latency and failover — some data is legally *forbidden* from leaving one.

Per-country deployment also buys two things you want anyway. Latency: a driver in Jakarta hits an Indonesian region, not a transpacific round trip, which matters when live matching budgets are measured in tens of milliseconds. And blast-radius containment: a bad deploy or a regional outage is bounded to one country instead of taking down the whole of Southeast Asia at once. The price is operational multiplicity — you now run and observe N stacks — which is exactly why the deployment and routing layer has to be configuration-driven rather than hand-managed. A faithful model of that routing layer:

```python
# Per-country routing. The store a request touches is a function of its country,
# and the residency policy is data, not tribal knowledge in someone's head.
from dataclasses import dataclass

@dataclass(frozen=True)
class CountryConfig:
    region: str            # AWS region the regional stack runs in
    aurora_endpoint: str   # in-region transactional store
    ddb_region: str        # in-region DynamoDB
    pii_may_leave: bool     # residency policy, enforced at the egress boundary

REGISTRY: dict[str, CountryConfig] = {
    "SG": CountryConfig("ap-southeast-1", "orders.sg.internal", "ap-southeast-1", False),
    "ID": CountryConfig("ap-southeast-3", "orders.id.internal", "ap-southeast-3", False),
    "MY": CountryConfig("ap-southeast-1", "orders.my.internal", "ap-southeast-1", False),
    "VN": CountryConfig("ap-southeast-1", "orders.vn.internal", "ap-southeast-1", False),
}

def store_for(country: str) -> CountryConfig:
    try:
        return REGISTRY[country]
    except KeyError:
        raise ValueError(f"no regional stack configured for {country!r}")

def export_to_lake(country: str, record: dict) -> dict:
    cfg = store_for(country)
    if not cfg.pii_may_leave:
        record = anonymize(record)   # strip/scrub PII BEFORE it crosses the border
    return record                    # only the safe projection leaves the region
```

The point of writing residency as `pii_may_leave` data in a registry — rather than as scattered `if country == ...` checks — is that the boundary becomes auditable. You can show a regulator one table and one enforcement function, point at the `anonymize` call on the egress path, and demonstrate that raw personal data structurally cannot leave the region. Compliance stops being a promise and becomes a property of the code path. The second-order gotcha: the anonymization has to be genuine — k-anonymity or aggregation, not a reversible pseudonym — because "de-identified" data that can be re-joined to an individual is still personal data in the eyes of most of these regulators.

## 6. Cost discipline at Southeast Asia scale

**Senior rule of thumb: at super-app write volume, storage and traffic cost is a load-bearing design constraint, not a finance afterthought. The cheapest byte is the one you let expire; the second cheapest is the one you moved to cold storage before anyone asked.**

A company operating across emerging Southeast Asian markets cannot price like a Silicon Valley app with infinite-margin enterprise contracts. Margins are thin, volume is enormous, and a careless storage decision compounds into a real line item fast. This is why cost shows up repeatedly in Grab's engineering writing — DynamoDB cost-optimization work, and pointed efforts like driving Kafka consumer traffic cost toward zero by being deliberate about cross-availability-zone data transfer. Cross-AZ traffic in particular is a cost that is invisible on an architecture diagram and very visible on a bill: every byte a consumer pulls from a broker in another AZ is metered, and at Grab's message volume that is a number worth engineering against.

![Layered stack of hot, warm, cold, and archive data tiers matching storage price to access frequency](/imgs/blogs/grab-scaling-data-in-southeast-asia-8.webp)

The figure is the discipline in one picture: match storage price to access frequency. Hot data that must answer in milliseconds — live sessions, current driver positions — sits in Redis and DynamoDB and is the most expensive per gigabyte, so you keep as little of it as possible and expire it aggressively with TTLs. Warm data, recent history that is queried in seconds, lives in Aurora and DynamoDB at mid cost. Cold data — the vast majority, the analytical history — lives as Parquet in the data lake on object storage, an order of magnitude cheaper per gigabyte. And archival data ages out via lifecycle policies to cold object-storage classes, cents per gigabyte, restored on demand. Data flows *down* the tiers as it ages, and each step trades latency for cost.

The three concrete levers that make tiering pay off are worth stating plainly. First, **TTLs on high-write tables** so the hot tier never accumulates data nobody reads — the location table from section 4 is the textbook case. Second, **lifecycle policies** that move lake objects to colder storage classes automatically by age, so a five-year-old partition costs a fraction of a five-day-old one. Third, **right-sizing per market** — provisioning a small market's stores for a small market rather than copy-pasting the capacity plan from your largest one. The meta-lesson is that cost is a first-class design axis at this scale: you do not bolt it on after the architecture is "done," because the architecture *is* the cost. The same instinct underlies [capacity planning for databases](/blog/software-development/database-scaling/capacity-planning-for-databases) — plan for the workload you have, in the market you have, not the one you wish you had.

## Case studies from production

These vignettes are composites — the kind of incident this architecture produces, drawn from the patterns Grab and similar super-apps have written about. The names are illustrative; the failure shapes and fixes are real.

### 1. The city that throttled at rush hour

A location-ingestion service ran clean for months, then began throttling every weekday at 6 p.m. in the largest market. The first hypothesis was "we need more write capacity," and the team raised the table's provisioned WCUs — to no effect, because the table-level capacity was never the limit. The actual root cause was a partition key of `city#<id>`: the entire metro area's evening write surge landed on one physical partition that capped out around 1,000 WCU/s no matter what the table was provisioned for. The fix was write-sharding — appending a `#<shard>` suffix derived from the driver id, spreading the load across 16 partitions — plus a TTL so stale pings expired instead of accumulating. Throttling vanished, and the bill went *down*, because they stopped over-provisioning a table to fight a problem provisioning could never solve. The lesson: in DynamoDB, a throttle at a fraction of provisioned capacity is almost always a hot key, not a capacity shortfall.

### 2. The analytics query that took down checkout

An analyst, needing fresh data, pointed a heavy reporting query directly at a production replica of the orders database during a campaign. The replica's buffer pool got blown out by the scan, replication lag spiked, and the checkout service — reading from that replica — started timing out. Revenue stopped for the duration. The wrong first instinct was to throttle the analyst; the real fix was architectural. No analytics workload should ever touch a production store. The event backbone existed precisely so that the lake, fed by CDC, carries the analytical copy. The post-incident action was not a policy memo but a network boundary: production database endpoints were made unreachable from the analytics environment, so the lake was the *only* path. The lesson is that "please don't query production" is a control that fails; "you cannot query production" is one that holds.

### 3. The Debezium connector that quietly fell behind

A dashboard showed a metric flatlining. Not crashing — flatlining, which is worse, because nothing alerted. The cause was a Debezium connector that had died after a binlog rotation; the source database kept committing, but no changes were flowing to Kafka, so the lake silently stopped updating. Downstream tables looked fine, just frozen in time. The first fix was to restart the connector. The durable fix was to treat CDC as a monitored data path: lag alerts on connector offset versus binlog head, and an automated row-count reconciliation that compared source table counts to lake counts and paged when they diverged beyond a threshold. The lesson: a pipeline that fails *silently* is more dangerous than one that fails loudly, and CDC fails silently by default. You have to build the alarm.

### 4. The duplicate that double-counted revenue

After a connector restart and replay, a finance report showed a revenue spike that did not match reality. The replay had re-delivered a window of change events, and the compaction job had naively appended them, double-counting a batch of transactions. The wrong fix was to manually delete the duplicates from the report. The right fix was to make compaction *idempotent*: key every change by its source primary key plus log offset, and merge so that re-processing the same change converges to the same row rather than adding a second one. Once compaction was idempotent, replay — the operational superpower of a log-based backbone — became safe to use freely instead of feared. The lesson: in any at-least-once pipeline, idempotent consumers are not optional; they are the thing that makes the "at least once" tolerable.

### 5. The shared table that coupled two teams

Early on, deliveries and a promotions service shared a table because it was expedient. Months later, promotions needed a schema change that deliveries' queries depended on, and a routine migration turned into a multi-team negotiation that blocked both roadmaps for a sprint. The symptom looked like "slow migration"; the root cause was a shared schema that coupled two teams with no business reason to be coupled. The fix was to split the table along the service boundary, give each team its own store, and have promotions consume the `order.placed` event from the backbone instead of reading deliveries' table directly. The migration pain was real but one-time; the coupling pain would have recurred forever. The lesson: a shared table is a standing tax on every future schema change, and the interest compounds with team count.

### 6. The cross-AZ bill nobody saw coming

A Kafka consumer group was rebalanced across availability zones for resilience, and the next month's data-transfer bill jumped. Nothing was broken; the consumers were simply pulling messages from brokers in *other* AZs, and every cross-AZ byte is metered. The first reaction was to blame the volume of messages; the real lever was *locality*. By making consumers prefer to fetch from a replica in their own AZ — trading a little cross-AZ replication for a lot less cross-AZ consumer traffic — the transfer cost for those consumers dropped dramatically. The lesson is that at super-app message volume, network topology is a cost decision, and the cheapest byte to move is the one that never leaves its availability zone.

### 7. The cold partition that cost like a hot one

A data-lake table accumulated years of history in the same storage class as last week's data. The table worked fine; it just cost far more than it should, because five-year-old partitions almost never queried were priced like five-day-old partitions queried constantly. There was no incident — which is exactly why it festered. The fix was a lifecycle policy that transitioned objects to a colder storage class by age, plus partition pruning so queries skipped cold partitions entirely unless a date filter reached back that far. Storage cost for that table dropped by a large factor with no change to the queries analysts actually ran. The lesson: cost regressions do not page you, so you have to go looking for them — a periodic audit of "what are we storing in the expensive tier that nobody reads" pays for itself.

### 8. The market that was provisioned like the flagship

A newly launched small market had its stores provisioned by cloning the configuration of the largest market — generous capacity, multi-shard tables, the works. It ran flawlessly and cost a fortune relative to its tiny traffic. The instinct to "use the proven config" was reasonable and wrong: the proven config was proven *for a different load*. Right-sizing the small market's stores to its actual traffic — fewer shards, on-demand capacity instead of large provisioned reservations — cut its infrastructure cost to a fraction with no user-visible change. The lesson: capacity plans do not transfer between markets, and "right-size for the market" is a per-deployment decision, not a global default.

## When to reach for the super-app data pattern (and when not to)

Reach for the full pattern — database-per-service, polyglot stores, a CDC event backbone, and per-country deployment — when:

- You are genuinely **N businesses under one app**, each with a distinct read/write shape and failure domain (transport, deliveries, payments, identity), not one product with several screens.
- Your workloads **span the shapes**: at least one strongly-transactional store, at least one high-write key-value firehose, and a real analytics need that cannot be served off the operational database.
- You operate across **multiple jurisdictions** with data-residency or data-protection rules, so per-country deployment and an explicit residency boundary are requirements, not gold-plating.
- Your **write or message volume is large enough that cost is an engineering constraint** — cross-AZ traffic, hot-tier storage, and provisioned capacity show up as real money you can engineer down.
- You need **failure isolation between business lines** so that an incident in one cannot take revenue offline in another.

Skip or simplify the pattern when:

- You have **one product and one workload shape**. A single well-run relational database with read replicas will outscale your needs for a long time, and a polyglot fleet is pure overhead you will pay for in operational complexity. Start with [vertical scaling and its ceiling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling) before you shard anything.
- Your team is **small enough that the coordination cost of many services exceeds the coupling cost of one schema**. Database-per-service is an organizational scaling tool; below a certain headcount it is friction without payoff.
- You operate in **one jurisdiction**. Per-country deployment is a tax you should not pay until a second regulator forces it.
- Your analytics fits in the **operational database with a read replica**. You do not need a CDC backbone and a data lake to run a few dashboards; you need them when reporting load threatens production or when reporting freshness must be near-real-time across many stores.
- You are tempted to adopt the pattern **because it is what the big companies do**. The big companies adopted it because their constraints forced it. Adopt the constraint-driven subset you actually need, and no more.

The throughline of Grab's data architecture is not any single clever system. It is a refusal to pretend a super-app is one thing. It is N businesses, so it is N stores, chosen by workload; they are decoupled by an event backbone instead of coupled by a shared schema; they are deployed per country because the law and the latency budget demand it; and the whole fleet is operated with cost as a first-class constraint because the markets demand that too. Scale the decomposition, watch the bill, and let each store be the boring, well-understood thing it was always meant to be.

## Further reading

- [Polyglot persistence: choosing the right store](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store) — the decision tree behind the workload-to-store matrix.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — how to produce events reliably from a transactional service.
- [Hot partitions and hot rows](/blog/software-development/database-scaling/hot-partitions-and-hot-rows) and [DynamoDB global tables and single-table design](/blog/software-development/database-scaling/dynamodb-global-tables-and-single-table-design) — the high-write path in depth.
- [How Uber built geo-distributed storage on MySQL](/blog/software-development/database-scaling/uber-geo-distributed-data) and [multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture) — the same problems from a different company's lens.

---
title: "Manual MySQL Sharding Done Right: Pinterest and Shopify"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "How Pinterest sharded MySQL with self-describing 64-bit ids and Shopify sharded it by shop into isolated pods, and why disciplined manual sharding still beats a clustering layer for control and operability."
tags: ["database-sharding", "mysql", "pinterest", "shopify", "pods", "distributed-ids", "multi-tenancy", "scaling", "case-study", "system-design"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 37
---

There is a quiet assumption in most "how do I shard MySQL" conversations: that the answer is a piece of software. Vitess, or a proxy, or some clustering layer that makes many machines look like one. It is a reasonable instinct — sharding is the hardest thing most teams ever do to a database, and outsourcing the hardest thing to a mature tool is usually good engineering.

But two of the most-studied production MySQL fleets on the internet did not do that. Pinterest sharded MySQL by hand and ran its core data on that scheme for years. Shopify sharded MySQL by hand into units it calls **pods**, and ran Black Friday — the single most demanding retail traffic event on earth — on top of it. Neither reached for an automatic clustering layer first. Both shipped deliberately *boring* designs, understood every failure mode, and got further on hand-rolled sharding than most teams get with a framework. This post is about how they did it, where the two designs agree, where they diverge, and what that tells you about when manual sharding is not a compromise but the right call.

![Pinterest's 64-bit id is partitioned into reserved, shard, type, and local-id fields, so any id reveals its own shard with a bit-shift.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-1.webp)

The diagram above is the mental model for the Pinterest half of the story. A 64-bit integer is not just a unique number; it is a *packed record*. The top bits hold the shard, the next bits hold the object type, the bottom bits hold the row's local id. Given any id, you shift out the shard bits and you know — with zero network calls, zero lookups, zero shared state — which database that row lives on. Hold that picture. Half of disciplined manual sharding is making the routing key tell you where to go, and the other half is making sure you never need a query that spans two of those locations.

## Why manual sharding is different from how it is taught

The instinct, when you outgrow one MySQL box, is to install a sharding framework and let it route for you. That can be the right move at the right scale. But the framework hides a set of decisions that you are still on the hook for, and Pinterest and Shopify made every one of them explicitly. Here is the gap between the textbook mental model of "just shard it" and what disciplined hand-sharding actually requires.

| Question | Naive view | What Pinterest and Shopify actually did |
| --- | --- | --- |
| Who routes a query? | A proxy or cluster layer figures it out | The application computes the target from the key — a bit-shift, or a directory lookup |
| What is the unit of partitioning? | One database server per shard | A *logical* shard (Pinterest) or a *pod* of shops (Shopify), decoupled from physical capacity |
| How many shards? | As many as you have machines | A large fixed pool, over-provisioned on day one, never re-counted |
| How do cross-shard joins work? | The framework joins for you | They don't — you denormalize and join in the application, or you don't ask the question |
| How do you rebalance? | Auto-rebalancing magic | Move a shard or a shop deliberately, then flip one routing entry |
| What technology underneath? | The newest thing that promises scale | The most boring, best-understood thing available: plain MySQL |

The pattern under every row is the same discipline: **separate the logical unit of partitioning from the physical unit of capacity, make routing a property of the key, and refuse to let any single query depend on data that lives in two places.** A framework can enforce some of that for you. But if you understand the four decisions well enough to make them by hand, you often discover you do not need the framework — and you gain something it can never give you, which is a system whose every failure mode you can reason about at three in the morning.

We will build the Pinterest design first, because its ideas are the cleanest, then the Shopify design, because it solves a different problem — multi-tenant isolation — with the same philosophy. If you have read [how Instagram sharded Postgres with ids that know their own shard](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres), the first three sections will feel like a cousin of that design, because they are: embedded-shard ids are a small family of independent inventions, and Pinterest's is the MySQL member.

## 1. The id knows its own shard

> The cheapest lookup is the one you never make. If the routing key can carry its own destination, routing stops being a system and becomes arithmetic.

Pinterest's primary keys are 64-bit integers, and the bits are not random. They are carved into four fields. From the top: **2 reserved bits** left at zero, a **16-bit shard id**, a **10-bit type**, and a **36-bit local id**. The encoding is a single expression:

```python
# Pinterest's id packing. shard in the high bits, then type, then local id.
# 16 + 10 + 36 = 62 bits of payload; the top 2 bits stay 0 so the value is
# always a positive signed 64-bit integer (safe in MySQL BIGINT and in Java).
SHARD_BITS = 16
TYPE_BITS  = 10
LOCAL_BITS = 36

def make_id(shard_id: int, type_id: int, local_id: int) -> int:
    assert 0 <= shard_id < (1 << SHARD_BITS)   # up to 65,536 logical shards
    assert 0 <= type_id  < (1 << TYPE_BITS)    # up to 1,024 object types
    assert 0 <= local_id < (1 << LOCAL_BITS)   # ~68.7B rows per type per shard
    return (shard_id << (TYPE_BITS + LOCAL_BITS)) | (type_id << LOCAL_BITS) | local_id

def decode_id(pin_id: int) -> tuple[int, int, int]:
    shard_id = (pin_id >> (TYPE_BITS + LOCAL_BITS)) & ((1 << SHARD_BITS) - 1)
    type_id  = (pin_id >> LOCAL_BITS) & ((1 << TYPE_BITS) - 1)
    local_id =  pin_id & ((1 << LOCAL_BITS) - 1)
    return shard_id, type_id, local_id

PIN = 0  # type 0
# A pin on logical shard 1234, the 7329th pin created on that shard:
pid = make_id(shard_id=1234, type_id=PIN, local_id=7329)
assert decode_id(pid) == (1234, PIN, 7329)
```

Notice what is *not* in that code: no call to a service, no read from a table, no cache. The shift `pin_id >> 46` (because `TYPE_BITS + LOCAL_BITS = 46`) recovers the shard from any id, anywhere, on any machine, in nanoseconds. That single property is the load-bearing wall of the whole design. When a request arrives carrying a pin id — in a URL, in a foreign reference inside another object, in a list of "pins on this board" — the server already knows which database to ask without consulting anything.

The three fields each buy something specific. The **shard id** is the routing key. The **type** lets one numeric space hold pins, boards, users, comments, and everything else without collisions, and lets the application dispatch on type after decoding — a board id and a pin id are never confused even though both are 64-bit integers. The **local id** is just an `AUTO_INCREMENT` within that shard's table for that type, so generating a new id is a plain MySQL insert that hands back the next sequence value; the application then packs it together with the shard and type it already knows. There is no central id server, no Snowflake-style coordination, no clock dependency. The id generator is `INSERT`.

The 36-bit local space holds roughly 68.7 billion rows per type per shard, the 16-bit shard field allows up to 65,536 logical shards, and 10 type bits give 1,024 object types. Those ceilings were chosen once, in early 2012, and frozen. The reserved 2 bits are the kind of paranoia that ages well: they cost two bits of address space and they leave a clean expansion slot if the scheme ever needs another flag. Boring, deliberate, and exactly the sort of decision a framework would have made invisibly and possibly wrong.

## 2. Routing without a directory

> A directory in the hot path is a second database you now have to scale, cache, invalidate, and keep available. The art is in not needing one.

Decoding an id gives you a *logical* shard number, not a machine. The mapping from logical shard to physical MySQL host is the only lookup in the path, and Pinterest made it as small and as static as possible.

![From a 64-bit id, a bit-shift yields the shard, one cached shard map yields the host, and the query lands on a master-master MySQL pair.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-3.webp)

The resolution path above has exactly one indirection. The 64-bit id decodes to shard 1234. A small configuration — the shard-to-host map — says shard 1234 currently lives on host `db07`. The application opens (or reuses) a connection to `db07` and runs `SELECT ... WHERE local_id = 7329`. That configuration is a few kilobytes: thousands of shard entries pointing at a much smaller set of physical hosts. Pinterest kept the authoritative copy in ZooKeeper and cached it in every application server, so the steady-state cost of "which host?" is an in-memory dictionary read. ZooKeeper is consulted on change, not per query.

```python
# The entire routing layer. Decode the shard from the id, look it up in a
# cached config, return a connection. The config is the ONLY thing that
# changes when a shard moves machines.
SHARD_TO_HOST = {
    # thousands of logical shards -> a handful of physical MySQL hosts
    **{s: "db01" for s in range(0,    4096)},
    **{s: "db02" for s in range(4096, 8192)},
    # ... shard 1234 falls in the db01 range in this snapshot
}

CONNECTIONS = {
    "db01": "mysql://app@db01.internal:3306/pins",
    "db02": "mysql://app@db02.internal:3306/pins",
}

def connection_for_id(pin_id: int):
    shard_id, _type, _local = decode_id(pin_id)
    host = SHARD_TO_HOST[shard_id]      # in-memory dict, refreshed from ZooKeeper
    return connect(CONNECTIONS[host])

# Reading a pin is: decode -> resolve host -> one indexed primary-key lookup.
def get_pin(pin_id: int) -> dict:
    conn = connection_for_id(pin_id)
    _shard, _type, local_id = decode_id(pin_id)
    return conn.query_one("SELECT data FROM pins WHERE local_id = %s", local_id)
```

Underneath, each shard is not a single server but a **master-master MySQL pair** — two machines replicating to each other, with one taking writes at a time. The pair is for availability, not for write scaling: if the active master dies, the standby is already warm and consistent, and the shard-to-host config is repointed. This is the most conventional MySQL high-availability setup there is, which is precisely why Pinterest chose it. The team's stated philosophy was to use the most mature, dependable technology they could and to understand its failure modes completely, rather than adopt a newer clustering system whose behavior under partition or hardware failure they could not predict. Sharding gave them horizontal scale; plain replication gave them durability; nothing in the stack was novel enough to surprise them during an incident.

The second-order win is what the design *prevents*. Because the shard is encoded in the id and the host map is tiny and cached, there is no per-row routing table to grow, no central router to become a bottleneck, and no lookup service that can be the thing that takes the site down. The hardest part of routing — keeping a directory consistent and available at the same scale as the data — simply does not exist here. For a deeper taxonomy of how this choice compares to range, hash, and directory routing, see [sharding strategies compared](/blog/software-development/database-scaling/sharding-strategies-compared); Pinterest's scheme is the "encode the shard in the key" branch taken to its logical end.

### Second-order optimization: shards are small enough to carry

The non-obvious reason to provision thousands of logical shards on a handful of machines is that it makes rebalancing a *copy*, not a *re-key*. A logical shard is a small, self-contained slice of data. To relieve a hot machine, you copy a few logical shards' worth of tables to a new host and change their entries in the shard-to-host map. The ids never change, because the shard number is baked into them and the *logical* shard did not move — only its physical home did. Contrast this with the naive "one shard per machine" design, where adding a machine means changing the hash, which means re-keying live data. The cost of growth in Pinterest's design is bandwidth and a config edit; the cost of growth in the naive design is a migration. This is the same insight that makes [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) tractable: never tie the count of partitions to the count of machines.

## 3. Objects and mappings: the join lives in the app

> A relational join is a promise that two rows live close enough to touch. Sharding breaks that promise. Either you stop asking, or you keep the join and pay for it in the application.

Encoding the shard in the id solves *where a row lives*. It does nothing for the harder problem: relationships. A board has many pins. A user has many boards. A pin is liked by many users. In a single database you express those with foreign keys and `JOIN`. Across thousands of shards, a `JOIN` is meaningless — the two tables are not even on the same machine. Pinterest's answer is a deliberately small data model with exactly two kinds of table.

![Mapping tables return id lists and object tables return blobs; the application stitches them together by id, so the database never joins.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-2.webp)

Everything is either an **object** or a **mapping**. An object — a pin, a board, a user — lives in a table keyed by its local id, with a single `data` column holding a denormalized blob of its fields. A mapping — `board_has_pins`, `user_has_boards`, `pin_liked_by_user` — is a relationship table that stores the relationship as rows of ids, not as a foreign-key constraint. To render a board, you do not join; you walk. You read the mapping to get a list of pin ids, then you fetch each pin object — and because each pin id encodes its own shard, the fetches route themselves.

```sql
-- An OBJECT table. One row per pin, all its fields denormalized into `data`.
-- No foreign keys, no columns the database needs to understand or index for joins.
CREATE TABLE pins (
  local_id   BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  data       MEDIUMTEXT,           -- JSON blob: {title, link, user_id, board_id, ...}
  PRIMARY KEY (local_id)
);

-- A MAPPING table. The relationship "this board contains these pins" as ID rows.
-- Lives on the BOARD's shard so "pins on this board" is one local indexed scan.
CREATE TABLE board_has_pins (
  board_id   BIGINT UNSIGNED NOT NULL,   -- full 64-bit id of the board
  pin_id     BIGINT UNSIGNED NOT NULL,   -- full 64-bit id of a pin (knows its shard)
  sequence   INT NOT NULL,               -- ordering within the board
  PRIMARY KEY (board_id, pin_id),
  KEY ix_board_seq (board_id, sequence)
);
```

The application-side read is a fan-out, not a join:

```python
def render_board(board_id: int, limit: int = 50) -> list[dict]:
    # 1. One LOCAL indexed read on the board's own shard: get ordered pin ids.
    board_conn = connection_for_id(board_id)
    rows = board_conn.query(
        "SELECT pin_id FROM board_has_pins "
        "WHERE board_id = %s ORDER BY sequence DESC LIMIT %s",
        decode_id(board_id)[2], limit,
    )
    pin_ids = [r["pin_id"] for r in rows]

    # 2. Fetch each pin OBJECT. Each id self-routes to its shard; batch by shard
    #    so co-located pins share one round trip. No cross-shard join anywhere.
    pins = multiget_objects(pin_ids)            # groups ids by decode_id(id)[0]

    # 3. The "join" is this dict assembly, in the app, in memory.
    by_id = {p["local_id"]: p for p in pins}
    return [by_id[decode_id(pid)[2]] for pid in pin_ids if decode_id(pid)[2] in by_id]
```

This feels like more work than a `JOIN`, and it is — but it is *bounded, predictable* work that the database can always serve with primary-key reads, and it never asks two machines to cooperate inside one query. The `multiget_objects` helper groups the requested ids by their decoded shard, so fetching fifty pins that happen to live on six shards is six parallel batched reads, not fifty round trips and not one impossible distributed join. The cost you pay is denormalization discipline: the pin's `data` blob carries its author's `user_id`, and if you want the author's display name you fetch the user object too, or you copy the display name into the pin at write time and accept that renames need a fan-out update. That tradeoff — denormalize to make reads local, and own the write-time cost of keeping copies fresh — is the recurring tax of every hand-sharded system, and the full menu of techniques for paying it is in [cross-shard queries and distributed joins](/blog/software-development/database-scaling/cross-shard-queries-and-distributed-joins).

### Second-order optimization: model the access pattern, not the entities

The subtle skill in the objects-and-mappings model is that you do not create a mapping table for every relationship in your domain — you create one for every *read you must serve cheaply*. `board_has_pins` exists because "show me a board's pins, newest first" is a hot path. There is a separate `pin_liked_by_user` mapping and a `user_likes_pin` mapping precisely because both directions are queried, and each is the cheap path for its own question. You are, in effect, hand-building the indexes that a single-node database would have built from your foreign keys — except you get to choose exactly which ones exist, on which shard they live, and in what order they sort. It is more work up front and far less magic, but every query's cost is something you decided on purpose. For how this interacts with key choice, [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) is the companion read: Pinterest's key is the object id, and the mapping tables are what make a non-id-based access pattern still resolve to local reads.

## 4. Shopify: shard by shop, not by id

> When your data has a natural tenant, the tenant is your shard key, and the right unit of isolation is the whole tenant — not a row, not a table, but everything that tenant touches.

Pinterest's world is one giant social graph: pins and boards and users that all reference each other, with no natural top-level partition. Shopify's world is the opposite. It is millions of independent stores. Two shops almost never share data; a query for shop A's orders has no business touching shop B's products. That structure hands you a shard key for free — the `shop_id` — and it changes what good sharding looks like. Where Pinterest shards by object and routes by arithmetic, Shopify shards by *tenant* and routes by *directory*, and it isolates not a row but an entire shop's world inside a unit called a pod.

![A pod is an isolated MySQL cluster owning a disjoint set of shops; the stateless app tier is shared, so a pod is the blast-radius unit.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-4.webp)

A pod, in the diagram above, is a self-contained slice of stateful infrastructure: a MySQL cluster (with its replicas), and the stateful services that hang off it like per-pod Redis and background-job queues, serving one disjoint group of shops. Shop 1 through shop 10,000 live entirely in Pod A; shop 10,001 through 20,000 live entirely in Pod B; and crucially, *every* piece of a shop's data — its products, orders, customers, inventory, checkout sessions — lives within that one pod. A pod is wholly isolated from the rest of the database infrastructure. The stateless web and application servers are *not* podded: they are shared and autoscaled on Kubernetes according to traffic, because stateless tiers are the easy part to scale. The hard, stateful part — the databases — is what gets split into pods, because that is the component you cannot simply add more replicas of and walk away.

The reason to draw the boundary around the whole tenant, rather than around individual tables the way Pinterest does, is that a shop is a *consistency domain*. A checkout has to read inventory, write an order, and decrement stock, all transactionally, all for one shop. If those rows were spread across shards, every checkout would be a distributed transaction. By guaranteeing that a shop's entire dataset lives in one pod, Shopify keeps every per-shop transaction a plain local MySQL transaction — the same property Pinterest gets for a single object, extended to an entire tenant's relational graph. This is the deep reason tenant-sharding and object-sharding look so different: they are optimizing for the same thing (local reads and local transactions) against two very different data shapes.

```python
# Shopify-style routing: shop_id -> pod, via a directory. Unlike Pinterest's
# bit-shift, the mapping is data, not arithmetic, because a shop can MOVE pods
# without changing its id. The directory is the price of being able to rebalance
# tenants freely.
SHOP_TO_POD = {
    # in production this is a table in an unsharded "control" database,
    # cached at the edge; shown here as a dict for clarity
    9173: "pod-7",
    9174: "pod-2",
    # ... millions of entries, but only ~one per shop, and shops are coarse
}

POD_DSN = {
    "pod-2": "mysql://app@pod-2-primary.internal:3306/shopify",
    "pod-7": "mysql://app@pod-7-primary.internal:3306/shopify",
}

def pod_for_shop(shop_id: int) -> str:
    return SHOP_TO_POD[shop_id]              # directory lookup, cached

def connection_for_shop(shop_id: int):
    return connect(POD_DSN[pod_for_shop(shop_id)])

# Every request is scoped to one shop, so every request resolves to one pod.
def shop_orders(shop_id: int, since: str) -> list[dict]:
    conn = connection_for_shop(shop_id)
    return conn.query(
        "SELECT * FROM orders WHERE shop_id = %s AND created_at >= %s",
        shop_id, since,
    )
```

The directory is the structural difference from Pinterest. Pinterest can route by arithmetic because the shard is fixed in the id forever; an object can never change shards without changing its id. Shopify *wants* shops to change pods — that is how it rebalances load — so it cannot encode the pod in the shop id. The price of free tenant mobility is a lookup. Shopify pays it with a routing table that lives in a separate, unsharded control database and is cached aggressively, so the steady-state cost is again an in-memory read, not a query per request.

## 5. The pod as a failure domain

> Isolation is not a performance feature. It is the promise that your worst day affects a known, bounded fraction of your customers instead of all of them.

The single most valuable property of the pod is not throughput. It is **blast radius**. Because a pod is wholly isolated — its own MySQL, its own replicas, its own connection pools — an incident inside a pod is contained to the shops in that pod. A runaway query, a corrupted table, a failed primary, a bad migration: the worst it can do is take down its own pod's shops. In the figure above, Pod B is on fire, and shops 10,001 through 20,000 are having a bad day — but every other shop on the platform never notices. There is no shared database whose failure is everyone's failure.

This is the property that justifies the entire architecture, and it is worth being precise about why a shared cluster cannot offer it. In a single large cluster, every tenant shares the same buffer pool, the same replication stream, the same connection limit, the same lock manager. One tenant's pathological workload — a merchant running an unindexed report across millions of orders during a flash sale — degrades latency for every tenant on that cluster. The industry name for this is the noisy-neighbor problem, and it is unsolvable by tuning, because the resource being contended is physical. Podding solves it structurally: the noisy neighbor can only starve the dozen-thousand shops that share its pod, and the platform can move a chronically noisy shop — or a chronically *important* one — to a dedicated pod that guarantees it complete resource isolation. Large, traffic-heavy merchants frequently get exactly that: their own pod, so their spikes and everyone else's spikes can never collide. The same dynamic, seen from the database's side, is the [hot partition and hot row](/blog/software-development/database-scaling/hot-partitions-and-hot-rows) problem; pods are Shopify's containment strategy for it.

Pods are also a placement unit. Because a pod is self-contained, pods can sit in different data centers and regions. Where a shop's pod physically runs becomes a deployment decision rather than a data-model decision — you can place a pod near the merchants it serves, or spread pods across regions to limit the blast radius of a regional failure to the pods in that region. The pod boundary that contains software failures also contains physical ones. For the broader treatment of running databases across regions, [multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture) is the companion piece; the pod is what makes regional placement a per-tenant lever instead of an all-or-nothing migration.

## 6. Routing a request to its pod

> Locality is the whole game. If a request touches one pod, you can reason about it, isolate it, and move it. If it touches two, you have rebuilt the distributed monolith you were trying to escape.

The routing mechanism is simple by design, and its simplicity is the point.

![A request resolves its shop's pod at the load balancer and stays inside that one pod; it never fans out across pods.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-5.webp)

A web request for a shop arrives. A load balancer (in front of the stateless app tier) consults the routing table, finds the pod that owns this shop, and forwards the request there. From that point on, the request is *inside one pod*: it reads and writes that pod's MySQL, hits that pod's Redis, enqueues into that pod's jobs. It never fans out to other pods, because by construction the shop's entire dataset is local. The pods labeled 1 through N in the figure are untouched by this request; they are busy serving their own shops.

This shop-level locality is what makes everything else work. It is why a per-shop transaction is a local transaction. It is why a pod can fail without cascading. It is why you can move a shop by moving exactly the data the routing table points at. And it is why capacity planning is additive: each pod has a known capacity, each pod serves a known set of shops, and adding capacity means adding pods, not resizing a shared monster. The connection math alone is a relief — instead of millions of shops contending for one cluster's connection limit, each pod's connection pool serves only its own shops, a property explored in depth in [connection management at scale](/blog/software-development/database-scaling/connection-management-at-scale).

The discipline this demands of application code is that *every* unit of work must declare its shop. A background job that processes orders must be scoped to a shop so it can be routed to the right pod. An admin tool that touches many shops must iterate pod by pod, not issue one query against "all orders." Shopify enforces this in the framework: code that is not scoped to a shop is, in effect, code that does not know which database to talk to, and the platform treats that as a bug. The constraint feels heavy until you realize it is the same constraint Pinterest enforces with "no cross-shard joins" — both are the rule that *no single operation may depend on two shards* — and both companies decided that enforcing it in application discipline was cheaper than building a layer that hides it.

## 7. Pods on Black Friday

> Black Friday is not a traffic problem you solve once. It is a structural test of whether your worst tenant can hurt your best one, and whether you can add capacity faster than demand arrives.

Black Friday and Cyber Monday — BFCM — are where the pod model earns its complexity. The contrast with a shared cluster is stark.

![A shared cluster couples every shop's fate; podding makes capacity additive and contains every failure to one pod.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-7.webp)

On a single shared cluster, BFCM is a coupled-fate nightmare. All shops sit in one database, so one merchant's viral flash sale starves every other merchant's checkout. The cluster's capacity is a single number you must provision for the global peak. And an outage is total: if the cluster goes down, every store on the platform goes down at the worst possible moment of the retail year. None of these are bugs you can fix; they are properties of putting everyone in one place.

Podding turns each of those global properties into a per-pod, additive one. Capacity scales by *adding pods* — each new pod brings its own MySQL, its own headroom, its own connection budget — so the platform grows horizontally toward the peak instead of vertically into a wall. A noisy neighbor stays inside its pod, so the merchant having the best day of their life cannot ruin the day of a merchant three pods over. And an outage is contained: one pod down means its shops only, while the rest of the platform keeps taking orders. The architecture converts "can we survive the global peak on one machine" into "can we add enough independent pods, and balance shops across them well enough, that no single pod sees a peak it cannot handle" — a far more tractable question, and one you can answer by provisioning rather than praying. This is capacity planning as a fleet exercise rather than a single-box exercise, the mindset laid out in [capacity planning for databases](/blog/software-development/database-scaling/capacity-planning-for-databases).

## 8. Rebalancing: moving a shop with Ghostferry

> A sharding scheme is only as good as its ability to move data without taking it offline. The cleverness is never in the steady state; it is in the migration.

Shops grow at different rates. A pod that was balanced last quarter can drift until its busiest shops cluster together and its load is several times another pod's. Left alone, that imbalance becomes a reliability risk: the hot pod is the one that falls over first. So Shopify *moves shops between pods*, deliberately, to keep the fleet balanced — and it does so without taking the shop offline, using an open-source tool it built called Ghostferry.

![Batch copy and binlog tailing run while the shop stays online; only a brief writer-lock cutover pauses writes before the routing table flips.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-6.webp)

The move has three phases, and only the last one touches availability. First, a **live batch copy**: Ghostferry reads the shop's rows from the source pod with `SELECT ... FOR UPDATE` and writes them to the target pod, table by table, in parallel threads, while the shop keeps serving traffic on the source. Second, **live binlog tailing**: simultaneously, Ghostferry tails the source pod's binary log, filters out the changes that belong to *this shop*, and replays them on the target, so the target stays caught up with writes that land during the copy. The shop is fully online through both phases.

The only moment of downtime is the **cutover**, and it is engineered to be short. Shopify scopes every unit of work for a shop under a multi-reader-single-writer (MRSW) lock: ordinary requests hold a reader portion, so many can run at once. To cut over, the shop mover acquires the *writer* portion, which blocks new work for that one shop and waits for in-flight work to drain. With writes paused, the final binlog events are applied, the **routing table is updated** to point the shop at its new pod, and the writer lock is released — now every new request resolves to the target. Finally, after traffic has moved, a **verification** pass confirms the target's data matches, and only then is the stale copy pruned from the source. The shop experienced a pause measured in the time it takes to drain in-flight writes and flip a config row, not the hours it would take to copy its data offline. This is the same family of technique — copy, tail, cut over under a brief lock, verify — that underlies every credible [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) story; Ghostferry is Shopify's productionized version of it, scoped to the convenient unit of one tenant.

## The contrast: two entities, one discipline

We now have both designs in full, and the most useful thing is to lay them side by side. They look different because they shard different entities, but the shape of the discipline is identical.

![Pinterest and Shopify diverge on shard key, routing, isolation, and rebalancing, yet converge on denormalize and join in the app.](/imgs/blogs/pinterest-and-shopify-mysql-sharding-8.webp)

| Dimension | Pinterest | Shopify pods |
| --- | --- | --- |
| Shard key | Object id (the shard is encoded in it) | `shop_id` — the natural tenant |
| Routing mechanism | Bit-shift the id; resolve a tiny cached host map | Directory lookup: `shop_id` → pod, cached |
| Unit of isolation | One logical shard (a slice of tables) | One pod (a whole tenant's stateful world) |
| Why that unit | An object has no natural tenant; the graph is global | A shop is a consistency and blast-radius domain |
| Adding capacity | Move logical shards to new hosts, edit the map | Add pods; move shops to balance, via Ghostferry |
| Cross-shard stance | No joins; denormalize into object blobs, fan out by id | No cross-shard work; every request scoped to one shop |
| HA underneath | Master-master MySQL pair per shard | MySQL primary + replicas per pod |
| Technology choice | Plain MySQL, understood completely | Plain MySQL, understood completely |

Read down the table and the divergence is real: a bit-shift versus a directory, a logical shard versus a pod, an object versus a tenant. But read the last three rows and the convergence is the lesson. Both refuse cross-shard operations. Both denormalize so that the data a query needs is co-located. Both route in the application or its edge rather than in a magic layer. Both sit on the most boring MySQL configuration that does the job. **The shard key follows the natural entity of the domain — object for Pinterest, tenant for Shopify — and everything else is the same discipline applied to that choice.** If you internalize one thing from these two systems, it is that manual sharding is not a single technique; it is a posture, and the posture transfers across wildly different data shapes.

## When Shopify reached for Vitess

It would be dishonest to end the Shopify story at "manual sharding forever," because Shopify itself did not. For its core commerce platform the pod model still rules, but for a newer product — the consumer-facing Shop app — Shopify adopted [Vitess](https://vitess.io/), the MySQL clustering layer that grew out of YouTube. The reason is a clean illustration of when manual sharding stops being the right tool, so it is worth understanding rather than glossing.

The pod model shards by merchant, and it works because a merchant is a coarse, meaningful unit — there are millions of them, but one merchant's footprint is large and the blast-radius unit "a group of shops" makes sense. The Shop app has an order of magnitude more *users* than the platform has merchants, and a single user's data is tiny and their individual impact negligible. Sharding that by a coarse tenant unit does not fit; the natural key is `user_id`, with far more keys each carrying far less weight. And the team had hit the specific walls that push you off manual federation: the primary database had grown to many terabytes, schema migrations took weeks, and they were throttling background jobs because the database was constantly busy. Splitting the primary further by hand would have meant pushing more routing and more cross-database coordination into application code — exactly the complexity manual sharding is supposed to avoid.

Vitess earns its place precisely there. It shards a keyspace by a vindex on `user_id`, routes queries through a proxy (VTGate) that plans and forwards them to the right tablets, coordinates monotonic ids with Vitess sequences instead of a central server, and — the part that mattered most — makes resharding and cross-shard schema migrations a managed operation rather than a hand-built project. It is not free magic: the application still has to include the sharding key in its queries for them to route efficiently, and Shopify had to build query verifiers and patch its ORM so that relationships and updates carried the shard key down. But the bargain is different from manual sharding's. You trade "I control and understand every routing decision" for "the cluster handles routing, resharding, and migration coordination, and I enforce a few rules to keep it efficient." When you have thousands of small shards that must be resharded and migrated constantly, that trade is worth it — and a hand-rolled scheme would have you rebuilding a worse Vitess inside your application. The honest summary is that Shopify uses **both**: manual pods where the tenant is coarse and isolation is the goal, and Vitess where the keys are fine-grained and managed resharding is the goal. Choosing between them is the subject of [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree); these two products inside one company are a live example of the tree branching.

## Case studies from production

### 1. The id server Pinterest deliberately did not build

The conventional way to generate unique ids across a sharded fleet is a central ticket server — Flickr ran a pair of them, Twitter built Snowflake — a dedicated service that hands out ids so they never collide. Pinterest looked at that and declined. By encoding the shard in the id and using a per-shard `AUTO_INCREMENT` for the local id, the id generator became a plain MySQL insert: the database that owns the row also owns its sequence, and the shard and type are values the application already knows at write time. The first hypothesis a team usually reaches for here is "we need a Snowflake," and the lesson is that you often do not. A central id service is a new thing to scale, cache, and keep available in the write path; Pinterest's design has zero new moving parts because the id's structure does the coordinating. The cost was the discipline of choosing the shard at object-creation time and never being able to change it — a constraint that, like the embedded shard in [Instagram's Postgres ids](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres), turned out to be entirely livable.

### 2. The friends-of-friends query that was never written

A classic failure of teams new to sharding is to port a graph query directly: "show me pins liked by people who follow this board," which on one database is a three-way join and across shards is a distributed join that no amount of cleverness makes fast. The wrong first hypothesis is "we need a query engine that joins across shards." The actual fix in the objects-and-mappings model is to never let the query exist: you maintain the mapping tables that answer the *specific* reads you serve, and you fan out object fetches that self-route by id. If a new access pattern appears, you add a mapping table for it, populated at write time, rather than a cross-shard join at read time. The lesson is that denormalization is not a workaround you apply under duress; it is the data model. The system has no general join capability, on purpose, and that absence is what keeps every query's cost predictable. The price is paid by writers keeping mappings current, which is a cost you can see, budget, and optimize — unlike a distributed join, whose cost you discover in production.

### 3. The hot shard that moved in an afternoon

A single logical shard on Pinterest's fleet started running hot — one popular board and a cluster of heavy users had landed on the same shard, and its host's load climbed. Because logical shards are small and numerous, the fix was not a re-architecture; it was to copy a handful of logical shards off the hot host to a fresh machine and update their entries in the shard-to-host map. The ids of the moved rows did not change, because the *logical* shard number is what lives in the id and that did not move — only its physical home did. The team's instinct to over-provision logical shards on day one, which looked like premature complexity in 2012, is exactly what made this a copy-and-config-edit instead of a migration. The lesson generalizes: the time to make rebalancing cheap is before you need it, by decoupling partition count from machine count so that "move some shards" is always an option.

### 4. The flash-sale merchant who couldn't take down the platform

A Shopify merchant ran a heavily promoted product drop and their shop's traffic spiked by orders of magnitude in seconds. On a shared cluster, that spike would contend for the same buffer pool and connection limit as every other store, and checkout latency would climb platform-wide. Because the merchant lived in a pod, the contention was bounded to that pod's shops, and the platform's other millions of stores were unaffected. The deeper move is that Shopify can pre-empt this for known-large merchants by giving them a dedicated pod — complete resource isolation, so their spike has nothing to collide with. The wrong hypothesis here is "we need to autoscale the database for the spike"; the right one is "we need the spike to be unable to reach anyone else." Isolation, not elasticity, was the property that mattered, and the pod boundary delivered it structurally rather than reactively.

### 5. The pod primary that failed without an incident bridge

A MySQL primary in one pod suffered a hardware failure. In a single-cluster world this is a platform-wide, all-hands incident. In the podded world, it was a single pod failing over to a replica, affecting only that pod's shops for the duration of the failover, while every other pod kept serving. The blast radius was, by construction, one pod. The lesson the team drew was about *operational scale*, not just availability: because failures are contained, the operational load of running the fleet drops, since most database incidents touch a small, known group of shops rather than escalating to everyone. A bounded blast radius is not only kinder to customers; it is kinder to the on-call engineer, who can reason about a problem scoped to one pod instead of triaging a global outage. This is the same containment logic that makes per-pod regional placement safe — the boundary that stops a software failure also stops a hardware one.

### 6. The shop that outgrew its pod

A single merchant grew until their data and traffic were a disproportionate share of their pod, unbalancing it against its neighbors. The naive fix — take the shop offline, dump it, load it elsewhere — was unacceptable for a live store. Ghostferry made it a background operation: batch-copy the shop's rows to a target pod while tailing the binlog to keep up with live writes, then cut over under a brief writer lock and flip the routing table. The shop's downtime was the few seconds it took to drain in-flight writes and update one config row. The wrong hypothesis is "rebalancing requires a maintenance window"; the right one is that with per-tenant copy-and-tail tooling, rebalancing is a routine, online operation. The fact that the migration unit is one tenant — self-contained, with no cross-pod references to fix up — is what makes the cutover lock so short, and that property traces all the way back to the decision to put a whole shop in one pod.

### 7. The reserved bits that bought a decade of headroom

When Pinterest froze its id layout in 2012, it left two high bits reserved and chose field widths — 16 / 10 / 36 — with deliberate slack: room for tens of thousands of shards, a thousand object types, and tens of billions of rows per type per shard. None of those ceilings was close to binding at launch; all of them were chosen to never bind. The temptation in any bit-packing scheme is to maximize the field you are currently stressing and shave the others, and the lesson here is the opposite: in a format you can never change without rewriting every id in the system, generous fields and a couple of reserved bits are the cheapest insurance you will ever buy. The cost was a few bits of address space; the benefit was that the scheme survived years of growth without anyone ever having to confront "we are running out of shard numbers." A format frozen on day one rewards paranoia.

### 8. The team that adopted manual sharding and skipped the discipline

Not every manual-sharding story is a success, and the instructive failures look the same: a team copies the *mechanism* — shard by a key, route in the app — but skips the *discipline*. They allow a cross-shard query "just this once" for an admin report, and now there is a code path that fans out to every shard and falls over when shard count grows. They forget to scope a background job to a tenant, and it quietly reads from the wrong shard. They under-provision logical shards because more felt like premature optimization, and now rebalancing means re-keying. The lesson from Pinterest and Shopify is that the architecture is not the id format or the pod; it is the *refusal* — no cross-shard operations, every unit of work scoped to one shard, partition count decoupled from machine count, the most boring database underneath. Manual sharding done right is mostly a set of things you consistently decline to do. The teams that struggle are the ones that adopted the freedom of hand-rolling without adopting the constraints that make it safe.

## When to reach for manual sharding, and when not to

Manual sharding is a real, durable choice — both of these companies ran enormous fleets on it — but it is not a default. Here is how to tell which side of the line you are on.

**Reach for manual sharding when:**

- Your data has a **natural partition key** that almost every query already carries — an object id, a tenant id, a user id. If the shard key is in the request anyway, routing is nearly free and the cross-shard discipline is natural.
- You value **operational understanding** over convenience. A hand-rolled scheme on plain MySQL has failure modes you can fully reason about; that is worth a great deal at 3 a.m.
- **Isolation is a goal in itself** — multi-tenant blast-radius containment, noisy-neighbor control, per-tenant placement. The pod model exists because isolation was the requirement, and a transparent clustering layer would have hidden the very boundary you wanted.
- You can **enforce the discipline in your framework**: no cross-shard joins, every job scoped to a shard, ids or directories that route. If your platform can make un-scoped data access a build error, manual sharding stays safe as the codebase grows.
- You can **over-provision logical shards up front** and keep them small, so rebalancing is always a copy plus a config change, never a re-key.

**Skip manual sharding when:**

- You have **many small, fine-grained shards that must be resharded and migrated constantly.** That is Vitess's home turf; hand-rolling it means rebuilding a worse version of a managed resharding system inside your app.
- Your access patterns are **genuinely cross-cutting** and cannot be denormalized into per-shard reads — if most queries legitimately need to join across the partition key, sharding of any kind will hurt, and you should first exhaust [read replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) and [vertical scaling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling).
- You **cannot enforce the discipline** — if cross-shard queries will inevitably creep in because the team is large or the rules are unenforceable, a layer that handles routing centrally is safer than trusting every engineer to honor the constraints.
- You are **not yet at the scale where one well-tuned primary plus replicas falls over.** Sharding is the most expensive thing you can do to a database's developer experience; the [database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) exists to make sure you have spent every cheaper option first.

The enduring lesson of Pinterest and Shopify is not that you should always shard by hand. It is that manual sharding, done with discipline, is a first-class engineering choice and sometimes the *better* one — because it gives you control and operability that a framework abstracts away, and because the constraints it forces on you (no cross-shard work, denormalize, route by the natural key, isolate the blast radius) are exactly the constraints that make any sharded system fast and survivable. Shard by your natural entity, encode the route in the id or a directory, refuse the cross-shard query, and keep the database boring. Two of the largest MySQL fleets ever built did precisely that, by hand, and thrived.

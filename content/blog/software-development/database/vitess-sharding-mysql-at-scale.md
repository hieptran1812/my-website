---
title: "Vitess: Sharding MySQL at YouTube Scale"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "How Vitess turns a fleet of MySQL boxes into one logical database: vtgate's smart proxy, vindexes and the VSchema, scatter-gather routing, vttablet connection pooling, and online resharding via VReplication."
tags:
  [
    "vitess",
    "mysql",
    "sharding",
    "vtgate",
    "vreplication",
    "scalability",
    "distributed-systems",
    "planetscale",
    "resharding",
    "databases",
    "vindex",
    "system-design",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/vitess-sharding-mysql-at-scale-1.webp"
---

There is a particular kind of pain that every successful product eventually earns, and it always arrives the same way. MySQL has been excellent. It has been excellent for years — boring, correct, fast, the InnoDB buffer cache warm and the replicas caught up. Then one Tuesday the primary's write throughput plateaus, the disk is 80% full, the largest table has crossed two billion rows, and `ALTER TABLE` has become a multi-hour event that everyone schedules for 3 a.m. and prays through. The database is not slow because it is badly written. It is slow because it is *one machine*, and the product has outgrown what one machine can hold. YouTube hit this wall. Slack hit it. Square (now Block) hit it. The honest, terrible truth is that the next step — sharding — is where most teams convert a working database into a distributed-systems quagmire of their own making.

The naive response is to shard in the application: hash the user ID, pick a database from a list, and route every query yourself. This works for exactly as long as it takes to write the first cross-shard query, the first re-shard, or the first cross-shard transaction — at which point the routing logic metastasizes into every service, the "split this hot shard" project becomes a downtime-laden weekend, and a global unique constraint becomes impossible. Vitess is the alternative that YouTube built in 2010 and open-sourced in 2011 precisely because they had lived the hand-rolled version and refused to keep paying for it. Vitess sits in front of a fleet of ordinary MySQL servers and presents the application with a single logical database that *speaks the MySQL protocol* — same driver, same wire protocol, mostly the same SQL — while transparently sharding the data underneath. As of 2026 it underpins clusters with tens of thousands of MySQL nodes, and at [Slack it serves 2.3 million QPS at peak](https://slack.engineering/scaling-datastores-at-slack-with-vitess/) with a 2 ms median and an 11 ms p99.

![The app speaks the MySQL protocol to vtgate, which consults the topology service, routes to vttablets, and each vttablet fronts one MySQL](/imgs/blogs/vitess-sharding-mysql-at-scale-1.webp)

The diagram above is the mental model for the entire article: **vtgate** is a smart proxy that looks like one MySQL to your application; behind it, every MySQL server has a **vttablet** sidecar bolted to its front; a **topology service** (etcd or ZooKeeper) holds the cluster map but never sits on the query hot path; and **vtctld** is the control plane that mutates that map. Your application connects to vtgate as if it were a single MySQL. vtgate parses your SQL, figures out which shard (or shards) the query touches, talks to the right vttablets, and stitches the results back into one answer. The rest of this piece is a tour of that picture: how the logical schema (the VSchema) describes which tables are sharded and by what; how *vindexes* turn a column value into a shard; how query routing decides between hitting one shard and scattering to all of them; how vttablet's connection pooling tames MySQL's brutal per-connection cost; and — the killer feature — how VReplication splits or moves a shard *live*, with no downtime, via a snapshot-copy / binlog-catchup / verify / atomic-cutover dance. If you have read [partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding), this is the production-grade machine that implements the horizontal cut for MySQL.

## Why "just shard it yourself" is a trap

**Senior rule of thumb: the hard part of sharding is not splitting the data once — it is everything that comes after the split, and application-level sharding makes you re-pay for all of it by hand, forever.**

It is worth being precise about why hand-rolled sharding fails, because the failure is not obvious on day one. On day one it looks trivial: you have, say, 16 MySQL servers, you compute `shard = crc32(user_id) % 16`, you keep a config map from shard number to host, and every service that reads or writes user data imports a little routing library. The first query you write is `SELECT * FROM orders WHERE user_id = ?`, it carries the shard key, the library routes it to one box, life is good. Then reality arrives in five waves.

The first wave is the query that *doesn't* carry the shard key. A support tool wants to look up an order by its `order_id`, or a user by their `email`. Neither column is the shard key, so your routing library has no idea which of the 16 boxes holds the row. Now you are writing fan-out-and-merge code by hand: open 16 connections, run the query 16 times, collect 16 result sets, merge and sort them in application memory, apply `LIMIT` after the merge. You will write this code badly the first time (forgetting that `LIMIT 10 OFFSET 100` cannot be pushed down to each shard), and you will write a slightly different broken version in every service.

The second wave is **resharding**. Sixteen shards were plenty until one customer — a giant enterprise account whose data all hashes to shard 7 — turns shard 7 into a hot spot. [Slack lived exactly this](https://slack.engineering/scaling-datastores-at-slack-with-vitess/): their original workspace-sharded design meant "when a database shard experienced an outage, every single customer whose data was on that shard also experienced a full Slack outage," and large customers "ended up with a few hot spots in our database tier." Splitting shard 7 means choosing new boundaries, dual-writing to old and new during the transition, backfilling historical rows, verifying nothing was dropped, and cutting over — by hand, under load, usually with a maintenance window. Most teams do this maybe twice before they decide they would rather quit.

The third wave is the **cross-shard transaction**. A user transfers credit to another user whose row is on a different shard. Now you need two-phase commit, or a saga, or you give up and accept anomalies. The fourth wave is the **global unique constraint** — usernames must be unique across all shards, but MySQL's `UNIQUE` index only enforces uniqueness within one box. The fifth wave is **connection management**: each application instance now needs a connection pool *per shard*, so 100 app instances times 16 shards is 1,600 pools to size and monitor, and MySQL's per-connection thread cost punishes you for every one.

Vitess does not make these problems disappear — they are inherent to distributed data. What it does is solve each one *once*, in the data tier, so it never leaks into application code. The table below is the comparison I draw on a whiteboard whenever a team is deciding whether to adopt it.

![On every axis that matters, hand-rolled sharding leaks distributed complexity into app code while Vitess absorbs it in the data tier](/imgs/blogs/vitess-sharding-mysql-at-scale-8.webp)

| Concern | Hand-rolled sharding | Vitess |
| --- | --- | --- |
| Where rows live | `if/else` routing in every service | One vindex declared in the VSchema |
| Splitting a hot shard | Weekend project, dual-write, downtime | `Reshard`, live, sub-second cutover |
| Query without the shard key | Manual fan-out + merge code | Scatter-gather by the query planner |
| MySQL connection limits | App-side pool per shard, per instance | vttablet pools connections centrally |
| Cross-shard transaction | Build your own 2PC / saga | Built-in (best-effort or 2PC) |
| Global unique value | Impossible with raw `UNIQUE` | Lookup vindex enforces it |
| `ALTER` on a billion-row table | `pt-osc` per shard, run by hand | Managed online DDL |
| Wire protocol | Custom routing layer | Standard MySQL protocol — drivers unchanged |

The right-hand column is the promise. The rest of the article is how Vitess keeps it.

## 1. The architecture: four components, one of which is on the hot path

**Senior rule of thumb: there is exactly one component (vtgate) on the query hot path, exactly one (vttablet) on the storage hot path, and the topology service must never be touched while serving a query. If you internalize that, the rest of the operational model follows.**

Let me name the four boxes from the mental-model figure precisely, because every conversation about Vitess uses these terms and conflating them is the most common source of confusion.

**vtgate** is the smart proxy. It is a stateless, horizontally-scalable server that [speaks both the MySQL wire protocol and a gRPC protocol](https://andrewjdawson2016.medium.com/understanding-the-architecture-of-vitess-5f3c042c4cdd). Your application connects to a vtgate (usually a pool of them behind a load balancer) exactly as it would connect to a MySQL server — same `mysql://` URL shape, same driver. vtgate's job is to take the SQL you send, parse it, plan it (decide which shards it touches and how), execute it against the relevant vttablets, and return a single consolidated result. Because vtgate is stateless, you scale it by adding instances; because it speaks the MySQL protocol, your ORM and your `mysql` CLI do not know they are talking to a sharded fleet.

**vttablet** is the sidecar. Every MySQL instance in the cluster — every primary and every replica — has its own vttablet process sitting directly in front of it, almost always on the same host. vttablet is the only thing that talks to "its" MySQL. It owns the connection pool to that MySQL, it enforces query guardrails (rejecting queries that would scan the whole table, capping result sizes), it does query consolidation and hot-row protection, it manages backups and replication, and it is the agent that VReplication drives during resharding. The crucial insight: vttablet is *per MySQL*, vtgate is *per cluster*. vtgate is the brain that knows about all shards; vttablet is the local hands that know about one MySQL.

**The topology service** is the cluster's source of truth, stored in a strongly-consistent key-value store — [etcd, ZooKeeper, or Consul](https://vitess.io/docs/archive/12.0/reference/features/topology-service/). It holds small amounts of metadata: which keyspaces exist, which shards each keyspace has, which vttablet is the primary for each shard, and the VSchema. The single most important property to understand is this: **the topology service is never on the query hot path.** vtgate reads the topology at startup and then *watches* it for changes, caching the map in memory. A normal `SELECT` does not touch etcd at all. This is what lets a Vitess cluster keep serving 2 million reads per second even if the topology store has a hiccup — the map is already in vtgate's memory.

**vtctld** is the control plane: a server that accepts administrative commands (from the `vtctldclient` CLI or VTAdmin's web UI) and applies them to the topology and the tablets. Creating a keyspace, starting a reshard, switching a primary, applying a VSchema — these all go through vtctld. Note the dashed lines in the figure: vtctld → topo and vtctld's whole existence are *control plane*, deliberately drawn as metadata channels, not data channels. You can take vtctld down for an hour and queries keep serving; you just cannot make administrative changes while it is down.

A few quantities make the architecture concrete. A keyspace is a logical database; it maps to a set of shards. A shard is a range of the keyspace ID space (more on that below) served by one MySQL primary plus its replicas. In the figure, the keyspace has two shards, `-80` and `80-`, each backed by its own MySQL primary-plus-replicas. Reads can be routed to replicas (`@replica`) to offload the primary; writes always go to the primary. vtgate's `tablet_types` configuration controls this — Slack runs reads against replicas in six regions for their [International Data Residency product](https://slack.engineering/scaling-datastores-at-slack-with-vitess/), which is only possible because the routing layer, not the application, decides which tablet serves a read.

### Second-order consequence: the topology is a cache-invalidation problem

Because vtgate caches the topology in memory and watches for changes, there is a subtle window during operations like a planned primary failover (`PlannedReparentShard`): the topology is updated, and vtgates must notice the new primary and re-point writes. Vitess closes this window with topology watches and short TTLs, but it is the reason a Vitess failover is "seconds of elevated write errors" rather than truly zero — the data plane is eventually consistent with the control plane by design, which is the right trade because it keeps the control plane off the hot path. In Kleppmann's framing from *Designing Data-Intensive Applications* Chapter 6, this is the classic "request routing" problem: something has to map a key to a node, and Vitess chose the "routing tier (vtgate) consults a separate coordination service (the topo)" design rather than baking routing into clients or into the nodes themselves.

## 2. The VSchema: telling Vitess which tables are sharded and how

**Senior rule of thumb: MySQL's own schema (`CREATE TABLE`) describes columns and indexes on one box; the VSchema is a second, logical schema layered on top that describes how those tables are distributed across boxes. They are independent, and you will edit both.**

A Vitess cluster has two kinds of schema. The first is the ordinary MySQL schema — the `CREATE TABLE` statements that define columns, types, and local indexes inside each MySQL. The second is the **VSchema**, a JSON document stored in the topology that tells vtgate how to route. The VSchema is per-keyspace, and the most fundamental thing it declares is whether the keyspace is *sharded* or *unsharded*.

An unsharded keyspace is the simple case: one shard, all tables live on it, every query routes to that single shard. You use unsharded keyspaces for small reference data, for sequence tables (more below), and as the *starting point* before you shard — you almost always begin with an unsharded keyspace and move tables into a sharded one later. A sharded keyspace is where the interesting machinery lives. Here is a realistic VSchema for a sharded `customer` keyspace, lifted in shape from the [Vitess VSchema reference](https://vitess.io/docs/15.0/reference/features/vschema/):

```json
{
  "sharded": true,
  "vindexes": {
    "hash": {
      "type": "hash"
    },
    "email_lookup": {
      "type": "consistent_lookup_unique",
      "params": {
        "table": "lookup.email_user_idx",
        "from": "email",
        "to": "keyspace_id"
      },
      "owner": "user"
    }
  },
  "tables": {
    "user": {
      "column_vindexes": [
        { "column": "user_id", "name": "hash" },
        { "column": "email",   "name": "email_lookup" }
      ],
      "auto_increment": {
        "column": "user_id",
        "sequence": "lookup.user_seq"
      }
    },
    "order": {
      "column_vindexes": [
        { "column": "user_id", "name": "hash" }
      ]
    }
  }
}
```

Read it top to bottom. `"sharded": true` says this keyspace's data is spread across multiple shards. The `vindexes` block defines two routing functions: `hash` (a functional vindex that computes a keyspace ID from a column) and `email_lookup` (a lookup vindex backed by a physical table). The `tables` block then maps each table's columns to those vindexes. The `user` table is sharded by `user_id` via the `hash` vindex — that is its **primary vindex**, the one that decides which shard each row physically lives on. It *also* has `email` mapped to `email_lookup`, a secondary vindex that lets vtgate route an `email`-based query to the right shard without scattering. The `order` table is sharded by `user_id` too, using the same `hash` vindex, which is deliberate: it co-locates each user's orders on the same shard as the user, so `JOIN`s between `user` and `order` on `user_id` stay single-shard.

The mental hinge here is the **keyspace ID**. Every sharded table's primary vindex maps the sharding-key column to a keyspace ID, an opaque byte string. The keyspace ID, not the original key, is what determines the shard. Shards own contiguous ranges of the keyspace ID space, expressed in hex: `-80` means "from the beginning up to (but not including) `0x80`," and `80-` means "from `0x80` to the end." vtgate computes the keyspace ID, finds the shard whose range contains it, and routes there.

![A hash vindex maps the sharding key to a keyspace ID whose leading byte selects exactly one shard range](/imgs/blogs/vitess-sharding-mysql-at-scale-2.webp)

The figure walks one row through the machine. `user_id = 31415` goes into the `hash` vindex, which produces the 8-byte keyspace ID `0x3a7e11049c2f88d1`. To find the shard, Vitess compares the *leading* bytes against the shard ranges. The leading byte is `0x3a`. With four shards splitting the byte at quarter boundaries — `-40`, `40-80`, `80-c0`, `c0-` — `0x3a` is less than `0x40`, so it falls in the first range: shard `-40` owns this row. Notice what is *not* happening: there is no modulo by the number of shards. The keyspace ID is computed once and is stable; the *shard ranges* are what change when you reshard. That separation — stable keyspace IDs, mutable range-to-shard mapping — is the entire reason Vitess can reshard online, which we will get to.

### Sequences: auto-increment that works across shards

MySQL's `AUTO_INCREMENT` is per-table-per-server. On a sharded table that is useless — two shards would both hand out `id = 1`. Vitess solves this with a **sequence table** in an *unsharded* keyspace, referenced from the sharded table's `auto_increment` block (you can see `lookup.user_seq` in the VSchema above). The sequence table has a `next_id` and a `cache` size; vtgate reserves a block of IDs (say 1,000 at a time) from the unsharded sequence and hands them out locally, refilling when the block runs dry. This gives you globally-unique, monotonically-increasing-ish IDs without a round trip per insert. The cache is the tuning knob: a larger cache means fewer round trips but bigger gaps if a vtgate restarts.

### Second-order consequence: the VSchema is a contract you cannot casually break

Changing a primary vindex is effectively re-sharding — you are changing where every row lives. Adding a *secondary* (lookup) vindex is cheap and reversible. The discipline this imposes is identical to the [shard-key discipline from the partitioning post](/blog/software-development/database/database-partitioning-and-sharding): choose your primary vindex column as carefully as you would choose a shard key, because it is the least-reversible decision in the system. A good default is to shard by the dominant access dimension — `user_id` for a B2C app, `tenant_id`/`workspace_id` for B2B — so that the overwhelming majority of queries naturally carry the key and route to one shard.

## 3. Vindexes: the sharding function, and how to route without the key

**Senior rule of thumb: a primary vindex decides where a row lives; a secondary (lookup) vindex is a distributed index that lets you find a row by a non-sharding column without scanning every shard. The whole art of avoiding scatter-gather is having the right lookup vindexes.**

"Vindex" is short for "Vitess index," and it is the most important concept in the whole system. A vindex is a function (sometimes a function backed by a table) that maps a column value to one or more keyspace IDs. There are two orthogonal axes for classifying them, and you need both to reason about routing.

The first axis is **functional vs lookup**. A *functional* vindex computes the keyspace ID directly from the input value with pure math — `hash`, for instance, applies a hash function; `xxhash`, `numeric`, and `binary` are other functional vindexes. No table lookup, no extra round trip; the keyspace ID is computed in vtgate's memory. A *lookup* vindex is backed by an actual MySQL table that stores the mapping from value to keyspace ID. To route by a lookup vindex, vtgate first queries the lookup table ("what keyspace ID does this email map to?"), then routes the real query to that shard. Functional vindexes are free but only work for the column you sharded on; lookup vindexes cost a round trip but work for any column you choose to index.

The second axis is **unique vs non-unique**. A *unique* vindex maps each input value to exactly one keyspace ID — `user_id → one shard`. A *non-unique* vindex maps an input to potentially several keyspace IDs — `country = 'US' → many shards`, because many users with different keyspace IDs share a country. Unique vindexes route a query to a single shard; non-unique vindexes narrow a scatter to a subset of shards. The naming follows: `lookup_unique` and `consistent_lookup_unique` are the unique flavors; `lookup` and `consistent_lookup` are non-unique.

Here is why this matters in production, told through a real number. Square (now Block) found that [25% of their database traffic came from scatter queries](https://developer.squareup.com/blog/cross-shard-queries-lookup-tables/) — queries that lacked the sharding key and therefore "Vitess has no choice but to search *all* your shards... a classic scatter-gather pattern." Their fix was a lookup vindex: to find an entity by a non-sharding column, they created "a special VIndex that's backed by a database table and intended to solve this very problem," storing the mapping from the non-sharding key to the sharding key. vtgate consults that table first, then routes the query to the one correct shard. The trade they accepted is explicit: "one extra database round trip versus hitting every shard."

The `consistent_lookup` family deserves a special note. A naive lookup vindex has a consistency hazard: when you insert a row into the sharded table, you must *also* insert the (value → keyspace ID) mapping into the lookup table, and if those two writes are not coordinated you can end up with a lookup entry pointing at a row that does not exist, or a row with no lookup entry. The `consistent_lookup` vindexes solve this by participating in the *same transaction* as the main write, with careful ordering — Square's note that "lookups update *before* data changes to prevent orphaned references" is exactly this discipline. The owner of a lookup vindex (the `"owner": "user"` field in the VSchema) is the table whose inserts, updates, and deletes automatically maintain the lookup table. You write `INSERT INTO user ...`; Vitess writes the lookup row for you.

| Vindex flavor | Computes from | Round trip? | Routes to | Use for |
| --- | --- | --- | --- | --- |
| `hash` (functional, unique) | Pure math on the column | No | One shard | The primary vindex / shard key |
| `lookup_unique` | A backing table | Yes (one) | One shard | Find by a unique non-key column (email) |
| `lookup` (non-unique) | A backing table | Yes (one) | A subset of shards | Find by a non-unique column |
| `consistent_lookup_unique` | A backing table, txn-consistent | Yes (one) | One shard | Same, with transactional consistency |
| `numeric` / `binary` | Identity mapping | No | One shard | Keys that are already well-distributed |

### Second-order consequence: lookup vindexes have a write-amplification cost

Every lookup vindex you add is a second table that must be written on every insert/update/delete of the owner table. Two lookup vindexes on a hot table triple your write fan-out for that table (the row plus two lookup rows), and each lookup row is itself a sharded write that may land on a different shard, pulling in cross-shard transaction machinery. So lookup vindexes are not free indexes you sprinkle liberally — they are a deliberate trade of write cost for read routing. The senior move is to count your scatter queries (Vitess exposes per-query-pattern stats; this is how Square found their 25%) and add lookup vindexes only for the scatter patterns that actually hurt.

## 4. Query routing and scatter-gather: one shard when it can, all shards when it must

**Senior rule of thumb: vtgate is a query planner, not a dumb proxy. Its entire job is to push as much work down to a single shard as possible, and to scatter only when the query gives it no choice. Read your scatter stats; a creeping scatter rate is a creeping outage.**

When a query arrives at vtgate, it goes through a planner — parse the SQL, build a plan, and (because parsing is expensive) cache that plan keyed by the query's normalized signature. The plan's central decision is *fan-out*: how many shards must this query touch?

The happy path is a query whose `WHERE` clause pins the primary vindex to a single value: `SELECT * FROM user WHERE user_id = 31415`. vtgate computes the keyspace ID, finds the one shard, sends the query to that one vttablet, returns the result. One shard, one round trip, latency identical to unsharded MySQL. This is the case you want for ~95%+ of your traffic, and it is why choosing a sharding key that matches your dominant access pattern is the whole ballgame.

The next case is the lookup-vindex route: `SELECT * FROM user WHERE email = 'a@b.com'`. The planner sees that `email` has a lookup vindex, so it does a two-step plan — query the `email_user_idx` lookup table to get the keyspace ID, then route the real query to the resolved shard. Two round trips, still one *data* shard. Cheap enough.

Then there is the case the title of this section is about. `SELECT * FROM user WHERE country = 'US'` (no vindex on `country`), or `SELECT count(*) FROM user`, or `SELECT * FROM user ORDER BY created_at DESC LIMIT 20`. None of these carry a routable predicate, so vtgate has no choice: it scatters.

![A query lacking the sharding key scatters to every shard in parallel, and vtgate must merge and re-sort the partial results](/imgs/blogs/vitess-sharding-mysql-at-scale-3.webp)

The figure traces a scatter. vtgate's **ScatterConn** [executes the query across multiple shards in parallel](https://www.augmentcode.com/open-source/vitessio/vitess), manages a connection per shard, holds transaction state, and coordinates retries. Each shard runs the query locally and returns partial results. Then vtgate does the part that hand-rolled sharding always gets wrong: it *merges* the partials, and for an `ORDER BY ... LIMIT` it does a streaming k-way merge and applies the limit *after* the merge. Crucially it must ask each shard for enough rows to satisfy the limit globally — a `LIMIT 20` on the global result means each of N shards might contribute up to 20 rows, so vtgate fetches up to `N × 20` and trims. An `OFFSET` is worse: `LIMIT 20 OFFSET 1000` cannot push the offset down at all, so each shard must return its first 1,020 rows and vtgate discards 1,000 after merging. Deep pagination on a sharded scatter is a performance trap, full stop.

The performance characteristics of scatter are worth stating bluntly:

- **Tail latency, not mean latency, dominates.** A scatter is as slow as its *slowest* shard, because vtgate must wait for all partials before merging. With 4 shards a slow GC pause on one shard slows the whole query; with 256 shards the probability that *some* shard is having a bad moment approaches one. This is why scatter latency degrades as you add shards even when each shard is faster.
- **Aggregations partially push down.** `count(*)`, `sum()`, `min()`, `max()` can be computed per-shard and combined in vtgate (sum of counts, min of mins). `avg()` is decomposed into sum and count. But `count(DISTINCT ...)` cannot be exactly combined without shipping the distinct values up, and `GROUP BY` on a non-vindex column forces vtgate to re-group across shards in memory.
- **Cross-shard joins are the expensive frontier.** If you `JOIN user` and `order` and both are sharded by `user_id`, the join is co-located on each shard and pushes down cleanly. If you join two tables sharded by *different* keys, vtgate does the join itself — typically a nested-loop where it fetches rows from one side and probes the other side shard-by-shard. That is potentially `shards × rows` round trips. Vitess will do it, but it is the single most important thing to design your VSchema to avoid.

Square's data makes the abstract concrete: even after splitting one shard into two, they observed the "two smaller shards' QPS are almost identical, however they aren't even close to the 1/4 you would expect" — because scatter queries kept hitting all shards regardless of the split, so adding shards did not reduce per-shard scatter load. That is the central lesson of routing: **sharding only helps queries that route to one shard; scatter queries get no benefit and pay more tail latency as you add shards.**

### Reading a routing plan, concretely

It helps to see what the planner actually decides, because "scatter vs single-shard" is not a vague heuristic — it is a deterministic consequence of which predicates pin which vindexes. Walk three queries against the `customer` keyspace from the VSchema above, sharded by `user_id` via `hash`, with a lookup vindex on `email`:

```sql
-- (a) SINGLE SHARD: predicate pins the primary vindex.
SELECT * FROM `order` WHERE user_id = 31415;
-- plan: hash(31415) -> keyspace_id -> shard -40. One vttablet, one round trip.

-- (b) LOOKUP ROUTE: predicate pins a lookup vindex.
SELECT * FROM `user` WHERE email = 'a@b.com';
-- plan: SELECT keyspace_id FROM email_user_idx WHERE email='a@b.com'  (lookup keyspace)
--    -> resolves to keyspace_id 0x91...  -> shard 80-c0. Two round trips, one data shard.

-- (c) SCATTER: no routable predicate.
SELECT count(*) FROM `user` WHERE country = 'US';
-- plan: scatter count(*) to all shards; vtgate sums the partial counts.
```

The discipline this exposes is that you can *predict* a query's fan-out by inspecting its `WHERE` clause against the VSchema, before you ever run it. A predicate of the form `vindex_column = value` or `vindex_column IN (...)` routes to one shard (or the few shards the `IN` list spans); anything else scatters. This is why a code-review habit of "does this new query carry a vindex predicate?" catches scatter regressions before they reach production — the answer is mechanical, not a judgment call. Vitess also exposes the plan via `EXPLAIN FORMAT=vitess`-style introspection and per-pattern stats, so you can audit the live fleet for the patterns that scatter.

### Query guardrails: rejecting queries that shard badly

vttablet enforces guardrails that protect MySQL from queries that would behave pathologically on a sharded fleet. It can reject a `DELETE` or `UPDATE` with no `WHERE` clause, cap the number of rows a single query may return, kill queries that run longer than a deadline, and (with the right configuration) reject queries that would scatter when you have declared that table should never be scattered. These are not annoyances; they are the difference between "one bad query slows one shard" and "one bad ad-hoc query scatters a full-table scan to 256 shards and browns out the cluster." A senior team treats a new scatter query the way it treats a new full-table-scan in EXPLAIN — as something to be justified, not assumed.

### Second-order consequence: the planner's plan cache is a tuning surface

Because vtgate caches plans by normalized query signature, two things follow. First, queries that differ only in literal values share a plan (Vitess normalizes literals to bind variables), so the cache stays small and hit rates stay high — this is also how Vitess counts query patterns to surface scatter stats. Second, a query that is *not* parameterized (literals baked into the SQL) defeats normalization and bloats the plan cache; this is one more reason to use bind variables, beyond the SQL-injection reason you already know.

## 5. Connection pooling: how vttablet survives MySQL's per-connection cost

**Senior rule of thumb: MySQL spends roughly a quarter-megabyte of memory and a full OS thread on every open connection, and it falls off a cliff near `max_connections`. vttablet exists in part to make sure MySQL only ever sees a few hundred connections no matter how many thousands the application opens.**

MySQL uses a thread-per-connection model. Each client connection gets its own OS thread with its own stack (the per-thread buffers and stack add up to roughly 256 KB and up, depending on configuration), and the server context-switches among them. This is fine at hundreds of connections and catastrophic at tens of thousands: memory balloons, the scheduler thrashes, and once you cross `max_connections` new connections are simply refused — which, in an outage, is exactly when every app instance is frantically retrying and opening more. This is the same wall I covered in depth in [connection pooling](/blog/software-development/database/database-connection-pooling); Vitess's contribution is to put the pool *server-side*, in vttablet, where it can be shared across every application instance instead of fragmented across each one's local pool.

![vttablet multiplexes thousands of application connections onto a few hundred MySQL connections to dodge the thread cliff](/imgs/blogs/vitess-sharding-mysql-at-scale-5.webp)

The before-and-after is the whole argument. On the left, raw MySQL: 5,000 application connections become 5,000 MySQL threads, ~256 KB of stack each, a context-switch storm, and an eventual `Too many connections` error under load. On the right, vttablet: those same 5,000 application connections terminate at vttablet, which [multiplexes them onto a small pool of MySQL connections](https://planetscale.com/blog/connection-pooling) — a few hundred, sized to the box. The application never opens a MySQL connection directly; it only ever talks to vtgate, which talks to vttablet, which owns the real MySQL connections. The pool is deliberately **lock-free**, built on atomic operations and non-blocking data structures, because at this connection volume a contended mutex on the pool would itself become the bottleneck.

vttablet does not maintain *one* pool; it maintains several, segmented by workload so that one class of query cannot starve another. The headline ones:

- **OLTP pool** (`queryserver-config-pool-size`) — the regular short transactional/read queries that make up the bulk of traffic.
- **Transaction pool** (`queryserver-config-transaction-cap`) — connections currently inside a multi-statement transaction, which must stick to one MySQL connection for the transaction's lifetime.
- **Stream (OLAP) pool** (`queryserver-config-stream-pool-size`) — long-running streaming queries (large result sets, exports) that would hog an OLTP connection.
- **DBA / app pools** — administrative and internal connections.

The [Vitess sizing guidance](https://vitess.io/docs/21.0/reference/features/connection-pools/) is precise enough to budget against: the maximum MySQL connections a vttablet can drive is approximately `transaction-cap × 2 + pool-size + stream-pool-size + dba_pool_size + app_pool_size` plus a handful of reserved connections for transaction-read and online DDL. The documentation is emphatic that you should set MySQL's own `max_connections` 50–100% higher than that sum, because "Vitess may have to kill connections and open new ones" and MySQL's accounting of closed connections lags, so MySQL's view of the count can briefly exceed what vttablet actually holds open.

Two more vttablet features ride on top of the pool and are worth knowing by name:

- **Query consolidation.** If 200 application requests all issue the identical read at the same instant — a classic "thundering herd" on a hot cache miss — vttablet detects the in-flight duplicate and runs the query against MySQL *once*, then fans the single result out to all 200 waiters. This collapses a herd into one query, which is the difference between a cache stampede taking down a shard and it being a non-event.
- **Hot-row protection.** When many transactions contend for the same row (the canonical "everyone increments the same counter" pattern), vttablet serializes access to that row at the proxy layer rather than letting MySQL pile up a deadlock-prone queue of row-lock waiters. It turns a contention storm into an orderly line.

### Second-order consequence: pooling changes your transaction discipline

Because a transaction must hold one MySQL connection for its whole life (it lives in the transaction pool), a long-running transaction occupies a scarce pooled connection the entire time. With a transaction pool of, say, 300, you can have at most 300 concurrent open transactions per shard — and an application that holds transactions open across slow external calls (the classic "begin transaction, call a payment API, commit" anti-pattern) will exhaust the transaction pool and stall every other writer. The pool makes the cost of a long transaction *visible and bounded* rather than letting it silently bloat MySQL's thread count, which is a feature, but it means "keep transactions short" graduates from style advice to a hard capacity constraint.

## 6. Online resharding via VReplication: the killer feature

**Senior rule of thumb: the reason to adopt Vitess is not that it can shard — anyone can shard once. It is that it can *re-shard live*, splitting or merging or moving shards under production load with a cutover measured in sub-seconds. This is the capability that hand-rolled sharding never gets right.**

Everything so far has assumed a fixed set of shards. The hard part of real systems is that the right number of shards changes over time: you start with one shard, you split it into two when it gets hot, you split the hot half again, you eventually have dozens. Doing that without downtime is the whole game, and Vitess's answer is **VReplication** — a general-purpose, table-level replication engine that copies and continuously syncs data between sets of tablets. Resharding and table-moving are both built on it.

The mechanism is the same regardless of whether you are splitting shards (`Reshard`) or moving tables between keyspaces (`MoveTables`). It has four conceptual phases, and the magic is that the system keeps serving the entire time, with only a sub-second pause at the very end.

![Resharding stays online through copy and catch-up; only the final routing-rule cutover briefly pauses writes](/imgs/blogs/vitess-sharding-mysql-at-scale-4.webp)

The timeline above is the dance. Walk it left to right:

1. **Create the target shards (empty).** You declare the new shard topology — say, splitting one shard `0` into `-80` and `80-` — and Vitess creates the target shards with empty MySQL instances. Nothing serves from them yet.
2. **Copy phase.** VReplication takes a *consistent snapshot* of the source shard's relevant rows and streams them into the target shards, routing each row to the target shard its keyspace ID belongs to. This can take hours for a large shard. The source keeps serving all reads and writes the entire time — the copy is happening on replicas in the background.
3. **Catch-up phase.** The snapshot is, by the time it finishes, stale — writes happened during the copy. So VReplication switches to tailing the source's binary log (the binlog) and replays every change that occurred after the snapshot point, applying it to the targets. The replication lag between source and target shrinks toward zero. From here on the targets are a live, continuously-updated copy of the source data, just under a different shard layout.
4. **VDiff.** Before you trust the targets, you verify them. [VDiff captures consistent snapshots of source and target and compares them row by row](https://vitess.io/docs/22.0/reference/vreplication/internal/cutover/) (it computes and compares checksums of the data) to prove the target is an exact representation of the source. You do not cut over until VDiff is clean. This is the step hand-rolled migrations skip and then regret.
5. **Cutover (`SwitchTraffic`).** Now the atomic part. Reads are switched first (`SwitchTraffic` for read tablet types): vtgate's routing rules are flipped so that queries for these shards go to the targets. Reads have no consistency hazard, so this is zero-downtime. Then writes are switched: Vitess briefly stops writes to the source (typically well under a second), lets the last in-flight binlog events drain to the targets so they are fully caught up, flips the routing rules so writes now go to the targets, and resumes. The write stall is the *only* moment of unavailability in the entire operation, and it is sub-second. Vitess also arms a **reverse replication stream** (target → source) at this point, so if something looks wrong you can `ReverseTraffic` and roll straight back.
6. **Complete.** Once you are confident, `Complete` drops the now-unused source shards and cleans up the VReplication artifacts (the rows in the internal `_vt` bookkeeping tables and the routing rules).

The corresponding command sequence with `vtctldclient`, from the [Reshard reference](https://vitess.io/docs/22.0/reference/vreplication/reshard/), is short enough to read in full:

```bash
# 1. Create the workflow: split source shard '0' into targets '-80' and '80-'.
vtctldclient Reshard --workflow split_customer --target-keyspace customer create \
    --source-shards '0' \
    --target-shards '-80,80-'

# 2. Watch the copy + catch-up progress until the targets are caught up.
vtctldclient Reshard --workflow split_customer --target-keyspace customer show

# 3. Verify the target is an exact copy before trusting it.
vtctldclient VDiff --workflow split_customer --target-keyspace customer create
vtctldclient VDiff --workflow split_customer --target-keyspace customer show last

# 4. Cut over reads first (zero downtime), then writes (sub-second stall).
vtctldclient Reshard --workflow split_customer --target-keyspace customer \
    switchtraffic --tablet-types "rdonly,replica"
vtctldclient Reshard --workflow split_customer --target-keyspace customer \
    switchtraffic --tablet-types "primary"

# 5. If anything looks wrong before Complete, roll straight back.
# vtctldclient Reshard --workflow split_customer --target-keyspace customer reversetraffic

# 6. Drop the source shards and clean up.
vtctldclient Reshard --workflow split_customer --target-keyspace customer complete
```

There are a few flags worth knowing because they are the ones you reach for under pressure. `--max-replication-lag-allowed` gates `SwitchTraffic` so it refuses to cut over while the targets are lagging (you do not want to switch writes to a target that is 30 seconds behind). `--timeout` (default 30s) bounds how long the cutover will wait for the write stall to complete before aborting. `--on-ddl` controls what happens if a schema change shows up in the binlog mid-stream. And `--enable-reverse-replication` (default true) is what arms the rollback stream.

### MoveTables: the same engine, moving tables instead of splitting shards

`MoveTables` uses the identical VReplication lifecycle but for a different goal: relocating specific tables from one keyspace to another with no downtime. The canonical use is the very first sharding step — you have an unsharded `commerce` keyspace and you want to peel the `customer` and `order` tables out into a new (eventually sharded) `customer` keyspace.

![MoveTables copies, replicates, verifies, cuts over, and cleans up, so tables migrate between keyspaces with zero downtime](/imgs/blogs/vitess-sharding-mysql-at-scale-7.webp)

The five verbs are the same as Reshard — `Create`, then it runs through copy and replicating phases, then `VDiff` to verify, then `SwitchTraffic` to flip routing, then `Complete` to clean up — which is the point of building both on one engine. Here is the [MoveTables sequence](https://vitess.io/docs/archive/13.0/reference/vreplication/movetables/):

```bash
# Move 'customer' and 'order' out of the unsharded 'commerce' keyspace
# into a new 'customer' keyspace, live.
vtctldclient MoveTables --workflow move_cust --target-keyspace customer create \
    --source-keyspace commerce \
    --tables "customer,order"

# Confirm the copy is done and replication is caught up.
vtctldclient MoveTables --workflow move_cust --target-keyspace customer show

# Verify before cutover.
vtctldclient VDiff --workflow move_cust --target-keyspace customer create

# Cut over reads then writes.
vtctldclient MoveTables --workflow move_cust --target-keyspace customer \
    switchtraffic --tablet-types "rdonly,replica"
vtctldclient MoveTables --workflow move_cust --target-keyspace customer \
    switchtraffic --tablet-types "primary"

# Clean up: drop the source tables from 'commerce' and remove artifacts.
vtctldclient MoveTables --workflow move_cust --target-keyspace customer complete
```

So the canonical Vitess adoption path is now visible end to end: start unsharded; `MoveTables` to carve your data into its own keyspace; apply a VSchema with a `hash` primary vindex on that keyspace (initially still one shard); then `Reshard` from one shard to two, to four, to however many you need — each step live, each step verified, each step reversible. The split itself is just range arithmetic on the keyspace ID space.

![Shard 0 owns the whole keyspace-ID line; a split at 0x80 carves it into -80 and 80- with no overlap or gap](/imgs/blogs/vitess-sharding-mysql-at-scale-6.webp)

The split figure is the geometry behind step 4 of the lifecycle. Before, one shard `0` owns the entire `[0x00, 0xff]` range. The split at `0x80` carves it into shard `-80` owning `[0x00, 0x80)` and shard `80-` owning `[0x80, 0xff]`. The bounds are half-open exactly like Postgres range partitions — lower inclusive, upper exclusive — so `0x80` belongs to `80-` and never to `-80`, guaranteeing no overlap and no gap. Because every row's keyspace ID was computed once and never changes, "which shard owns this row" is purely a function of which range its keyspace ID lands in — and that is a routing-rule change, not a data rewrite. VReplication moves the bytes; the range arithmetic decides where they go. For the deeper treatment of how this generalizes to rebalancing without downtime, see [live resharding and rebalancing](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime).

### Second-order consequence: VReplication is also your migration and CDC engine

Because VReplication is a general "copy a consistent snapshot then tail the binlog" engine, it is not only for resharding. The same machinery powers `Materialize` (maintaining a transformed copy of a table — a rollup, a denormalized view), online schema migrations (apply the DDL to a shadow copy, VReplicate into it, cut over), and `VStream`, a change-data-capture API that lets external consumers subscribe to the binlog stream through vtgate. Adopting Vitess for sharding quietly hands you a [CDC](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) and online-migration platform too — which is a non-obvious part of the ROI.

## 7. Transactions across shards: available, discouraged, and why

**Senior rule of thumb: a single-shard transaction in Vitess is a normal, fast MySQL transaction. A cross-shard transaction is a distributed transaction with all the costs that implies — Vitess offers it, but the right design avoids needing it.**

Within one shard, a transaction is just a MySQL transaction — full ACID, normal isolation, normal performance. vtgate routes all statements of the transaction to the one shard, vttablet keeps them on one pooled connection, MySQL does what MySQL does. This is the case you want, and a well-designed VSchema (shard by the entity that transactions are scoped to — the user, the workspace, the account) keeps the vast majority of transactions single-shard.

Cross-shard is where it gets expensive. Vitess offers three transaction modes, set via `--transaction-mode`:

- **`SINGLE`** — only single-shard transactions are allowed; a transaction that tries to write to a second shard is rejected. This is the strictest and safest setting; it forces your application to keep transactions shard-local, which is usually what you want.
- **`MULTI`** (the default) — best-effort multi-shard transactions. Vitess will let a transaction span shards and commits each shard's part, but the commit is *not* atomic across shards: if shard A commits and shard B fails, you have a partial commit. This is acceptable for many workloads precisely because well-designed schemas rarely span shards, and the rare ones that do can tolerate the small inconsistency window or compensate for it.
- **`TWOPC`** — true two-phase commit across shards, giving atomicity. This is correct but slow: it requires a prepare phase, a coordinator, and durable transaction metadata, and it holds locks across the round trip. The [two-phase commit failure modes](/blog/software-development/database/two-phase-commit-and-how-it-fails) apply in full — a coordinator crash between prepare and commit blocks the participants. Vitess supports it for the cases that genuinely need cross-shard atomicity, but the official guidance and the field wisdom agree: design so you don't need it.

The reason 2PC is discouraged is not that Vitess's implementation is weak; it is that distributed atomic commit is fundamentally expensive and fragile, the way Kleppmann lays out in *DDIA*. The pragmatic stance, which Slack and Square both took, is to choose a sharding key such that your transactional units are shard-local. If your transactions are "everything for one user" and you shard by user, you never need cross-shard atomicity. The schema design *is* the transaction strategy.

| Transaction mode | Atomic across shards? | Cost | Use when |
| --- | --- | --- | --- |
| `SINGLE` | N/A (rejects multi-shard) | Cheapest | You can keep all transactions shard-local |
| `MULTI` (default) | No (best-effort) | Cheap | Multi-shard writes are rare and tolerant |
| `TWOPC` | Yes | Expensive, lock-holding | You genuinely need cross-shard atomicity |

### Second-order consequence: the sequence table is a hidden cross-shard write

Recall the sequence table for auto-increment lives in an unsharded keyspace. That means every insert into a sharded table that uses `auto_increment` does a small read against the sequence keyspace to reserve IDs. The `cache` size on the sequence is what keeps this from being a per-insert cross-keyspace round trip — with a cache of 1,000, you pay the cross-keyspace cost once per 1,000 inserts per vtgate. Setting the cache too low silently turns your high-volume insert path into a cross-keyspace chatterbox; it is one of the first things to check when insert latency on a sharded table looks worse than it should.

## 8. How Vitess hides sharding from the application

**Senior rule of thumb: the entire value proposition is that your application code does not change when you shard. If you find yourself writing Vitess-aware logic in the app, something in the VSchema is wrong.**

It is worth pulling together exactly *what* the application sees, because that interface is the product. Your application opens a connection to vtgate using a standard MySQL driver and a standard connection string. It issues standard SQL. It uses bind variables (which it should anyway). It does not know how many shards exist, which shard a row is on, or that a `Reshard` happened last night that doubled the shard count. The keyspace name appears in the connection (you connect to a keyspace, optionally with a tablet-type suffix like `customer@replica` to route reads to replicas), but that is the extent of the Vitess-awareness in a well-run setup.

The things that *would* leak sharding into a naive design, and how Vitess absorbs each:

- **Routing.** Absorbed by vindexes in the VSchema. The app says `WHERE user_id = ?`; vtgate routes. No routing code in the app.
- **Cross-column lookups.** Absorbed by lookup vindexes. The app says `WHERE email = ?`; vtgate consults the lookup table and routes. No fan-out code in the app.
- **Aggregation across shards.** Absorbed by the planner's scatter-gather + merge. The app says `SELECT count(*)`; vtgate sums per-shard counts. No merge code in the app.
- **Globally-unique IDs.** Absorbed by the sequence table. The app inserts without specifying the ID; Vitess assigns a globally-unique one. No ID-generation code in the app.
- **Global uniqueness constraints.** Absorbed by `consistent_lookup_unique` vindexes, which enforce that a value (a username, an email) maps to at most one row across all shards. No app-side uniqueness check.
- **Resharding.** Absorbed entirely by VReplication and routing-rule flips. The app sees, at worst, a sub-second blip on writes during cutover. No dual-write logic in the app.
- **Connection management.** Absorbed by vttablet's server-side pool. The app keeps a normal pool to vtgate; it never sizes per-shard pools. No per-shard connection code in the app.

This is the precise sense in which Slack could say Vitess "lets you flexibly shard and grow your database without adding logic to your application." The migration from their bespoke workspace-sharded system to Vitess took three years and moved 99% of query load — and the end state is an application that issues SQL to what looks like one MySQL. The complexity did not vanish; it moved into a layer (vtgate + vttablet + the VSchema) that is operated by a platform team and is invisible to product engineers. That relocation of complexity, from a thousand scattered `if/else` routing decisions into one declarative VSchema, is the whole point.

### Second-order consequence: the interface leaks exactly where SQL semantics force it to

Vitess is honest that the one-MySQL illusion is not perfectly transparent: a handful of SQL constructs behave differently on a sharded keyspace because they fundamentally cannot mean the same thing. `LAST_INSERT_ID()` interacts with the sequence machinery; some correlated subqueries and exotic joins are not pushable and either scatter or are rejected; `SELECT ... FOR UPDATE` across shards needs the cross-shard transaction machinery; and ordering/limit semantics on a scatter require the merge described earlier. A team adopting Vitess should run their query workload through vtgate in a staging cluster and look for the queries that scatter or get rejected — the leaks are predictable and finite, but you want to find them before production does, not after.

## Case studies from production

The principles above are easier to trust when you see them play out as incidents. These are composite case studies drawn from the public engineering writeups (Slack, Square/Block, YouTube, and the Vitess and PlanetScale teams) and the patterns I have seen repeated across teams running Vitess in anger. Each is the same shape: a symptom, a wrong first hypothesis, the actual root cause, the fix, and the lesson.

### 1. The creeping scatter rate

A team's p99 latency had been climbing for months, slowly, with no single deploy to blame. The first hypothesis was MySQL — surely the boxes needed bigger instances. They upsized; p99 barely moved. The actual cause surfaced when they finally looked at vtgate's per-query-pattern stats and found that scatter queries had crept from 8% to 31% of traffic, almost entirely from new analytics-flavored endpoints that filtered on non-vindex columns. Each new shard split (they had gone from 8 to 32 shards over the year) had made every scatter *slower*, because scatter latency is set by the slowest of N shards and N had quadrupled. This is exactly Square's "two shards aren't 1/4 each" observation: adding shards does nothing for scatter queries and worsens their tail. The fix was a pair of lookup vindexes for the two hottest scatter patterns and a guardrail that rejected new unindexed scatters in code review. The lesson: **the scatter rate is a vital sign; monitor it like you monitor error rate, because it degrades silently and shard splits accelerate the decay.**

### 2. The reshard that wouldn't cut over

A scheduled split of a hot shard reached the catch-up phase and then `SwitchTraffic` refused to proceed, failing with a replication-lag error. The on-call's first guess was a Vitess bug. It was not: `--max-replication-lag-allowed` was doing exactly its job — the target shards were 12 seconds behind because a batch job on the source was generating writes faster than VReplication could apply them to the targets. Switching writes to a target 12 seconds stale would have lost the last 12 seconds of data. The fix was to pause the batch job, let the targets catch up to sub-second lag, then `SwitchTraffic` cleanly. The lesson: **the lag gate is a feature that will save you from data loss, and a reshard cutover should be scheduled in a low-write window, not during a batch-heavy period.** A cutover that "won't switch" is usually the system protecting you, not failing you.

### 3. The transaction pool that quietly capped throughput

Write throughput on one shard plateaued at a number that looked suspiciously round, and adding application instances did nothing. The team suspected MySQL write contention. The real cause: the application held transactions open across a slow third-party HTTP call ("begin; charge card; commit"), so each in-flight payment occupied a vttablet transaction-pool slot for the ~400 ms the external call took. With a transaction pool of 300, the shard topped out at roughly `300 / 0.4s = 750` transactions per second regardless of how many app instances pushed at it. The fix was to move the external call outside the transaction (authorize first, then a short transaction to record the result). The lesson: **vttablet's transaction pool turns "long transactions are bad style" into a hard, measurable capacity ceiling — and that visibility is a gift, because it points straight at the offending code path.**

### 4. The lookup vindex that orphaned rows

A team rolled their own lookup vindex with the non-consistent `lookup` type and a hand-written trigger to maintain the mapping table. Under a burst of concurrent inserts, the mapping writes and the data writes interleaved badly, leaving lookup entries pointing at rows that had been rolled back and rows with no lookup entry — so some users became unfindable by email even though their row existed. The first hypothesis was a replication lag issue. The actual cause was the non-transactional coupling between the two writes. The fix was to switch to `consistent_lookup_unique`, which makes Vitess maintain the lookup table inside the same transaction with the correct ordering (lookup row written before the data row, exactly as Square documented). The lesson: **never hand-maintain a lookup mapping; use the `consistent_lookup` family so the mapping and the data commit atomically — the ordering subtleties are precisely what the built-in vindex exists to get right.**

### 5. The COVID traffic spike that Vitess shrugged off

In a single week in March 2020, [Slack's query rate jumped 50%](https://slack.engineering/scaling-datastores-at-slack-with-vitess/) as the world went remote overnight. On their old workspace-sharded system this would have meant frantic manual shard splits and hot-spot firefighting, because load could not be redistributed without moving whole workspaces by hand. On Vitess, the response was mechanical: spin up more vtgates (stateless, trivially horizontal) to absorb the connection and routing load, and `Reshard` the hottest keyspaces live to spread write load across more shards. No downtime, no application changes. The lesson: **the payoff of online resharding is not visible on a calm day; it is visible on the worst day, when the difference between "add shards live" and "schedule a maintenance window" is the difference between a non-event and an outage during your highest-ever traffic.**

### 6. The cross-shard join that melted a shard

An innocuous-looking report joined `orders` (sharded by `user_id`) with `products` (sharded by `product_id`) — two different sharding keys. On a small dataset in staging it was fine. In production, vtgate executed it as a nested-loop cross-shard join: for each batch of orders it probed the product shards, generating a storm of cross-shard round trips that pinned one product shard's CPU and dragged down every query on it. The first hypothesis was a missing index; the index existed. The real cause was the join topology — it could not push down because the two tables did not share a sharding dimension. The fix was twofold: a `Materialize` workflow to maintain a denormalized `orders_with_product` table sharded by `user_id` for the report, and a guardrail flag to reject unbounded cross-shard joins. The lesson: **co-locate the tables that join. A join across two different sharding keys is the most expensive thing you can ask vtgate to do, and the VSchema is where you prevent it — design the sharding dimensions so your hot joins stay shard-local.**

### 7. The deep pagination that fetched a million rows

A "load more" endpoint used `LIMIT 50 OFFSET 100000` for infinite scroll. On an unsharded MySQL it was merely slow. On a 64-shard keyspace it was a disaster: vtgate had to ask *each* of the 64 shards for their first 100,050 rows, merge ~6.4 million rows, discard 100,000, and return 50. Memory on vtgate spiked and the query timed out. The first hypothesis was a vtgate memory leak. The actual cause was offset-based pagination, which cannot push the offset down to shards. The fix was keyset (seek) pagination — `WHERE created_at < ? ORDER BY created_at DESC LIMIT 50` — which lets each shard return only 50 rows and lets vtgate merge a manageable `64 × 50`. The lesson: **`OFFSET` is poison on a scatter; use keyset pagination on any sharded table, because offset turns a cheap page fetch into a per-shard near-full-scan.**

### 8. The topology hiccup that didn't cause an outage

An etcd cluster upgrade went sideways and the topology service was unavailable for several minutes. The team braced for a full outage. It did not come: queries kept serving at full rate the entire time, because vtgate had the topology map cached in memory and a normal query never reads etcd. What *did* fail was a primary failover that happened to be attempted during the window (vtctld could not write the new primary to the topo) and a VSchema change that was queued. The lesson, and it is the most reassuring one in the list: **the topology service being off the hot path is not a documentation detail — it is the property that keeps a coordination-store outage from becoming a data-tier outage. Control-plane unavailability degrades your ability to *change* the cluster, not to *serve* from it.**

### 9. The unbounded result set that OOM'd a vttablet

A migration script ran `SELECT * FROM events` against a large sharded table with no `LIMIT`, intending to stream and process every row. Without streaming mode it tried to buffer the entire scatter result, and a vttablet on one shard ran out of memory trying to materialize its partial. The first hypothesis was a memory leak in the migration tool. The real cause was a non-streaming full-table scatter; the fix was to use the OLAP/streaming connection mode (`@replica` with streaming), which pulls rows in bounded chunks through the stream pool rather than buffering, plus a vttablet guardrail capping non-streaming result rows. The lesson: **large reads must be streamed; the stream pool exists precisely so that an export does not buffer a billion rows into one connection's memory, and the row-cap guardrail is the seatbelt for when someone forgets.**

### 10. The vindex change nobody could undo

A team realized two years in that they had sharded by the wrong key — they had sharded `events` by `event_id` (uniformly distributed but never queried) instead of by `user_id` (the dominant access dimension), so nearly every real query scattered. Changing the *primary* vindex is not an in-place edit; it means every row's keyspace ID changes, i.e. a full re-sharding into a new keyspace with the new vindex, via `MoveTables`. The good news is that Vitess *could* do it live — `MoveTables` from the old keyspace to a new one with the corrected VSchema, copy, catch up, VDiff, cut over — but it was a weeks-long project to move a large table. The lesson: **the primary vindex is the least-reversible decision in the system, exactly like a shard key. Choose it to match your dominant query, validate it against your real query workload before you commit, and treat "we picked the wrong shard key" as the multi-week project it is — Vitess makes it survivable, not free.**

### 11. The sequence cache that throttled inserts

A bulk-import job that loaded tens of millions of rows into a sharded table ran an order of magnitude slower than the same job against a single MySQL, and CPU on neither vtgate nor the shards was saturated. The first hypothesis was that scatter writes were the problem — but these were keyed inserts that routed to one shard each, so they should have been cheap. The actual cause was the sequence table: the table's auto-increment used a sequence with `cache` set to the default low value, so every small batch of inserts forced a fresh round trip to the unsharded sequence keyspace to reserve more IDs. At import volume, those reservation round trips serialized the whole job behind a single unsharded keyspace. Raising the sequence `cache` to a large block (so each vtgate reserved tens of thousands of IDs at once) removed the chatter and the import ran at full speed. The lesson: **the sequence `cache` is a throughput knob, not a cosmetic one — high-volume insert paths must reserve IDs in big blocks, or the unsharded sequence keyspace becomes the choke point for an otherwise perfectly-sharded write path.**

### 12. The replica read that returned stale rows

A user updated their profile and immediately reloaded the page, only to see the old values — intermittently, and never reproducible in staging. The first hypothesis was a caching bug in the app. The real cause was that the read path used `keyspace@replica` to offload reads to replicas, and MySQL replication is asynchronous: the read landed on a replica that had not yet applied the write the user had just made on the primary. This is read-after-write inconsistency, not a Vitess bug — it is the [async replication](/blog/software-development/database/database-replication-sync-async-logical-physical) trade-off surfacing through Vitess's tablet-type routing. The fix was to route the specific "read your own write" paths to `@primary` (or use Vitess's bounded-staleness/`@replica` with a freshness guarantee where supported) while keeping the bulk of read traffic on replicas. The lesson: **tablet-type routing makes the replication-lag trade-off a per-query decision — use `@replica` for the 95% of reads that tolerate staleness, and consciously route the read-after-write paths to `@primary`. The routing layer gives you the knob; you still have to know which reads need consistency.**

## When to reach for Vitess, and when not to

Vitess is one of the highest-leverage and highest-commitment pieces of infrastructure you can adopt. The decision deserves the same seriousness as choosing a shard key, because in effect it *is* that decision plus an operational platform around it.

**Reach for Vitess when:**

- **You are on MySQL and genuinely cannot fit on one primary anymore** — write throughput, data size, or `ALTER` pain on a single box has become the binding constraint, and you have already exhausted vertical scaling, read replicas, and single-node [partitioning](/blog/software-development/database/database-partitioning-and-sharding). Sharding is the right tool only after those are spent.
- **You will need to reshard more than once.** If you can confidently say you will only ever shard one time and never again, the value of online resharding is muted. If shard count will grow as the product grows — which is the normal case for anything successful — VReplication's live reshard is worth the entire adoption cost by itself.
- **You want to keep MySQL and the MySQL protocol.** Your team knows MySQL, your tools speak MySQL, your drivers are MySQL. Vitess lets you scale horizontally without abandoning that ecosystem or rewriting to a NewSQL/NoSQL store with a new query model and new failure modes.
- **You have, or can build, a platform team to operate it.** Vitess has real operational surface area: the topology store, vttablet tuning, VSchema management, reshard runbooks. The payoff is that this complexity is centralized and invisible to product teams — but someone owns it.
- **You can choose a sharding key that makes most queries single-shard.** A natural tenancy dimension (`user_id`, `workspace_id`, `account_id`) that the bulk of your queries already filter on is the strongest signal that Vitess will fit your workload cleanly.
- **You want online schema change and CDC for free.** Even teams whose sharding need is modest sometimes adopt Vitess for its managed online DDL and `VStream` change-data-capture, because building those well is itself a large project.

**Skip Vitess (or wait) when:**

- **One MySQL primary is still comfortable.** The most common mistake is adopting sharding infrastructure years before the data requires it. A single well-tuned primary with replicas and native partitioning handles an enormous amount of load; cross the chasm only when forced. Premature Vitess is premature distributed systems.
- **Your workload is overwhelmingly analytical.** Vitess optimizes OLTP — many small, key-routed queries. If your traffic is big aggregations and scans over all the data, you want a columnar/OLAP store ([OLTP vs OLAP](/blog/software-development/database/oltp-vs-olap-and-columnar-stores)), not a sharded OLTP layer that turns every analytical query into a scatter.
- **You have no clean sharding dimension.** If your queries spray across many access patterns with no dominant key — if you cannot find a column that the majority of queries filter on — then most traffic will scatter, and sharding will cost you tail latency without buying throughput. Fix the access patterns first, or pick a different storage model.
- **You are a small team without operational bandwidth.** If you cannot staff the operational ownership, consider [PlanetScale](https://planetscale.com/blog/connection-pooling) or another managed Vitess offering, which runs the control plane, the topology, and the reshard tooling for you and lets you consume Vitess as a service. Self-hosting Vitess with a two-person team is a way to spend your scarcest resource on plumbing.
- **You truly need cross-shard atomic transactions everywhere.** If your domain genuinely cannot be modeled with shard-local transactions and you would lean on `TWOPC` for a large fraction of writes, you are fighting the model; reconsider whether the data should be sharded along that dimension at all, or whether a different architecture fits better.

The throughline, from the first wall to the last decision, is that Vitess does not eliminate the hard problems of distributed data — request routing, scatter-gather, cross-shard transactions, online migration — it *relocates* them out of your application and into a declarative, operable layer that YouTube, Slack, and Square have collectively run at tens of thousands of nodes and millions of QPS. You still have to choose a good sharding key, keep your scatter rate low, keep your transactions shard-local, and respect the lag gate during cutover. But you do it once, in a VSchema, instead of a thousand times, in a thousand `if/else` branches — and when the day comes that one box can no longer hold YouTube, you split a shard live instead of taking the site down.

## Further reading

- [Scaling Datastores at Slack with Vitess](https://slack.engineering/scaling-datastores-at-slack-with-vitess/) — the canonical production writeup: 2.3M QPS, three-year migration, the COVID spike, and why workspace sharding failed.
- [Cross-Shard Queries & Lookup Tables](https://developer.squareup.com/blog/cross-shard-queries-lookup-tables/) — Square/Block on the 25% scatter problem and lookup vindexes as the fix.
- [Vitess VReplication / Reshard reference](https://vitess.io/docs/22.0/reference/vreplication/reshard/) and [How Traffic Is Switched](https://vitess.io/docs/22.0/reference/vreplication/internal/cutover/) — the authoritative cutover mechanics.
- [Vitess VSchema reference](https://vitess.io/docs/15.0/reference/features/vschema/) and [connection pools sizing](https://vitess.io/docs/21.0/reference/features/connection-pools/) — vindex types and the pool-sizing formula.
- [Connection pooling in Vitess (PlanetScale)](https://planetscale.com/blog/connection-pooling) — the server-side, lock-free pool and why MySQL's per-connection cost forces it.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 6 (Partitioning) — the vocabulary for partitioning, secondary-index partitioning, rebalancing, and request routing that underpins everything above.
- Sibling posts on this blog: [partitioning & sharding](/blog/software-development/database/database-partitioning-and-sharding), [connection pooling](/blog/software-development/database/database-connection-pooling), [replication](/blog/software-development/database/database-replication-sync-async-logical-physical), and [live resharding & rebalancing](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime).

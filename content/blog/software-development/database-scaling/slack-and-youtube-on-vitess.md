---
title: "Sharding MySQL with Vitess: How YouTube Built It and Slack Adopted It"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A case study of Vitess, the MySQL sharding system born at YouTube and adopted by Slack, and how both teams retrofitted horizontal sharding onto a grown MySQL fleet without rewriting the app."
tags:
  [
    "vitess",
    "mysql",
    "sharding",
    "database-scaling",
    "vtgate",
    "vreplication",
    "vindex",
    "slack",
    "youtube",
    "case-study",
    "distributed-systems",
    "system-design",
  ]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 31
---

Every team that outgrows a single MySQL server arrives at the same fork in the road, and they arrive at it angry. The database is the slowest, most fragile, most carefully-guarded thing they own. It is also full. Not full of data, exactly: full of writes the primary cannot absorb, full of connections the server cannot hold open, full of one table that has grown past the largest instance the cloud will rent you. You cannot scale up anymore. The only direction left is sideways, into shards. And the application has ten thousand SQL queries that all assume there is exactly one database to talk to.

This is the problem YouTube hit around 2010, and the problem Slack hit again around 2017 with a fleet that was already manually sharded and still buckling. Both teams reached for the same answer, because the second team adopted the thing the first team built: **Vitess**, a clustering and sharding system that sits in front of a fleet of plain MySQL servers and makes them look, to the application, like one enormous MySQL. You keep most of SQL. You keep MySQL's storage engine, its replication, its backups, its operational muscle memory. What you add is a proxy that knows how the data is split and routes every query to the right shard, plus the machinery to split shards online while traffic keeps flowing.

![Vitess architecture: the app speaks the MySQL wire protocol to a single vtgate proxy, which reads the shard map from the topology service and fans queries out to sharded vttablet plus MySQL backends](/imgs/blogs/slack-and-youtube-on-vitess-1.webp)

The diagram above is the mental model, and the rest of this article is a tour of it. The application opens an ordinary MySQL connection, but it connects to **vtgate**, a stateless smart proxy that speaks the MySQL wire protocol. vtgate reads the shard map out of a **topology service** (etcd, ZooKeeper, or Consul) and decides, per query, which shard or shards to touch. Each shard is a real MySQL instance fronted by a per-MySQL agent called **vttablet** that pools connections, rewrites queries, and guards the database from itself. When a query carries the sharding key, vtgate routes it to exactly one shard; when it does not, vtgate scatters it to all of them and merges the results. That single design choice, hide sharding behind a MySQL-speaking proxy, is what let two very different companies bolt horizontal scale onto an existing application without a rewrite. This is a companion to the [Vitess fundamentals deep-dive](/blog/software-development/database/vitess-sharding-mysql-at-scale); here we focus on the two migrations and what they teach.

## Why Vitess is different from rolling your own sharding

Most teams shard the obvious way first: they write the routing logic into the application. A function takes the user ID, hashes it, picks a database, opens a connection. It works, and then it slowly becomes the most expensive code in the building. Here is the assumption-versus-reality table that every hand-rolled sharding layer eventually writes in its own postmortems.

| Concern | Hand-rolled app-level sharding | Vitess |
| --- | --- | --- |
| Where routing lives | In every service that touches the DB, duplicated and drifting | In vtgate, one place, declared as a VSchema |
| Cross-shard queries | You write the scatter, merge, and re-sort by hand, per query | vtgate does scatter-gather and merge automatically |
| Resharding | A bespoke, terrifying, usually-downtime migration project | `Reshard` + VReplication, online, with a verification step |
| App changes to add a shard | Code change, deploy, pray | A topology change; the app sees nothing |
| Routing by a non-key column | A second lookup table you maintain by hand | A lookup vindex Vitess keeps consistent for you |
| Connection management | Every app process holds connections to every shard | vttablet pools and consolidates on the DB side |
| Operational cost | Low to start, brutal at scale | High fixed cost: you now operate Vitess too |

The last row is the honest one. Vitess is not free. You trade a pile of application complexity for a pile of infrastructure complexity, and you have to run vtgate, vttablet, the topology service, and the control plane yourself (or pay someone like PlanetScale to run it). The bet pays off precisely when the application complexity of sharding-by-hand would have been worse, which is to say at the scale where YouTube and Slack live. Let us start with why a single MySQL stops being an option at all.

## 1. The shared problem: one MySQL, and nowhere left to grow

A single MySQL server is a remarkable machine. On modern hardware it will happily serve tens of thousands of simple queries per second, hold terabytes, and survive years of abuse. The trouble is that all of its ceilings are vertical. You can buy a bigger box, add RAM, add IOPS, tune the buffer pool, and every one of those moves buys you a finite, shrinking amount of headroom. The classic dead-ends are the same across both case studies in this post: the **write ceiling** (a primary can only apply so many writes per second before replication lag and fsync latency eat you alive), the **connection ceiling** (each MySQL connection costs memory and a thread, so a few thousand app processes each holding connections will exhaust the server long before CPU does), and the **size ceiling** (one table grows past what the largest available instance can store or back up in a sane window). We cover the vertical limit in detail in [vertical scaling and its ceiling](/blog/software-development/database-scaling/vertical-scaling-and-its-ceiling); the short version is that you always hit the wall, the only question is when.

> Vertical scaling buys time, not a future. The day you order the biggest instance the vendor sells is the day you have run out of road, and you usually discover it during a traffic spike rather than in a planning meeting.

YouTube hit this first, and at a shape almost no one had seen before: video metadata, view counts, and comments growing at the rate of the entire internet discovering online video. The obvious fix, shard MySQL across many servers, ran straight into the connection ceiling. YouTube's frontend was a fleet of application servers, and the naive sharded design has every app server holding a connection to every shard. Multiply a few thousand app processes by a few hundred shards and MySQL falls over from connection overhead alone, never mind the queries. YouTube needed something between the app and MySQL that could pool connections, route intelligently, and protect the database from pathological queries. They built it, and they called it Vitess.

Slack arrived at the same wall from a different direction, and this is what makes their story instructive: **Slack was already sharded** when they adopted Vitess. They had not skipped sharding; they had done it the hand-rolled way and grown past it. Every Slack workspace (a "team") lived entirely on one database shard, and the application's monolith, "webapp," contained the logic to look up which shard a given workspace lived on and open a connection straight to that MySQL host. It was a clean, simple model, and for years it was the right one. Then it stopped being right, for reasons we will dig into, and the team spent three years migrating onto Vitess to fix it. Before we get to that migration, we need the parts of Vitess it depends on.

## 2. What Vitess actually is: sharded MySQL behind a proxy

Vitess has four components, and exactly one of them sits on the hot path of every query. Knowing which is which is most of understanding the system.

**vtgate** is the smart proxy and the only thing the application talks to. It speaks the MySQL wire protocol, so as far as your driver, your ORM, and your `mysql` CLI are concerned, vtgate *is* a MySQL server. Internally it is nothing like one: it holds no data, it is stateless, and you run a whole fleet of vtgates behind a load balancer. Its entire job is to take a SQL statement, parse it, consult the sharding metadata, and produce a *routing plan*: which shard or shards must be touched, what sub-query to send to each, and how to combine the answers. Because it is stateless, vtgate scales horizontally just by adding more of it, which matters because parsing and planning SQL is not free.

**vttablet** is an agent process that runs next to each MySQL instance, one per MySQL, and it is the unsung hero of the architecture. It owns the connection pool to its local MySQL, so the thousands of vtgate-to-vttablet connections collapse into a small, bounded pool of vttablet-to-MySQL connections. This is the move that solved YouTube's connection ceiling. vttablet also rewrites queries (adding `LIMIT`s, rejecting full-table scans), consolidates duplicate in-flight reads, manages transactions, and coordinates failovers and backups. The application never sees it; vtgate talks to it.

**The topology service** is a small, strongly-consistent metadata store (etcd is the common choice) that holds the shard map: which keyspaces exist, how each is sharded, which MySQL is the primary for each shard, and where every vttablet lives. It is the source of truth for "where is the data," and vtgate watches it so that a failover or a reshard propagates without anyone editing application config. It is deliberately tiny and off the data path; a query never reads user data from the topo.

**vtctld** and its CLI `vtctldclient` are the control plane, the admin surface you use to run reshards, schema changes, failovers, and backups. They write to the topology and command the vttablets. They are not involved in serving a query, which is why they sit off to the side in the mental-model diagram.

A **keyspace** is Vitess's word for a logical database. An *unsharded* keyspace is one MySQL (plus replicas); a *sharded* keyspace is split into shards, each owning a contiguous range of the keyspace's key-space. Shards are named by their range in hex: `-80` means "from the start up to 0x80," `80-` means "from 0x80 to the end," and a four-way split is `-40`, `40-80`, `80-c0`, `c0-`. The genius of naming shards by range rather than by number is that splitting a shard is a local operation: to split `-80` you create `-40` and `40-80`, and no other shard is affected. That is the foundation everything else is built on, and it is why the next question, "which range does a given row belong to," is the heart of the system.

## 3. The VSchema and vindexes: how routing is configured

vtgate cannot route a query unless it knows two things: which tables are sharded, and how to compute a shard from a row. That knowledge lives in the **VSchema**, a JSON document you give Vitess. The VSchema names each table's *vindex*, the function that maps a column value to a **keyspace id**, a fixed-width binary number. The keyspace id is the universal address: every row in a sharded keyspace has one, and the shard that owns the range containing it is the shard that stores the row.

![A hash vindex maps the shard key through a hash function to a keyspace id, whose byte range selects exactly one of the four shard ranges](/imgs/blogs/slack-and-youtube-on-vitess-2.webp)

The figure walks the whole path. Take `workspace_id = 1234`. The table's primary vindex is a hash function, `xxhash`, which maps that value to a keyspace id, say `0x9a3f8c...`. The shards partition the keyspace-id space into ranges; `0x9a` falls inside `0x80-0xc0`, so the row lives on shard `80-c0` and nowhere else. The hash matters: a hash vindex scatters consecutive IDs across the whole range, so a single hot customer's sequential keys do not all land on one shard. Here is what a minimal sharded VSchema looks like for a Slack-like schema keyed on workspace:

```json
{
  "sharded": true,
  "vindexes": {
    "xxhash": {
      "type": "xxhash"
    }
  },
  "tables": {
    "channels": {
      "column_vindexes": [
        { "column": "workspace_id", "name": "xxhash" }
      ]
    },
    "channel_members": {
      "column_vindexes": [
        { "column": "workspace_id", "name": "xxhash" }
      ]
    }
  }
}
```

Two design decisions are hiding in that small file. First, both tables use the *same* column and the *same* vindex, which means a `channels` row and a `channel_members` row for the same workspace compute the same keyspace id and therefore land on the same shard. That co-location is what makes a join between them a single-shard query instead of a cross-shard nightmare. Choosing a shared sharding key for tables you join is the single most important schema decision you will make; we treat it as its own subject in [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key). Second, the vindex here is a *primary* vindex, which is unique-ish and decides where the row is *written*. You get exactly one primary vindex per table.

The harder case is routing by a column that is *not* the sharding key. Suppose `messages` is sharded by `channel_id`, but a feature needs to fetch a message by its globally-unique `message_id`. Hashing `message_id` would point at the wrong shard. The answer is a **lookup vindex**: a second, Vitess-managed table that maps `message_id` to `keyspace_id`, so vtgate can do a quick lookup and then route to the single correct shard rather than scattering.

```json
{
  "vindexes": {
    "xxhash": { "type": "xxhash" },
    "message_lookup": {
      "type": "consistent_lookup_unique",
      "params": {
        "table": "lookup.message_lookup",
        "from": "message_id",
        "to": "keyspace_id"
      },
      "owner": "messages"
    }
  },
  "tables": {
    "messages": {
      "column_vindexes": [
        { "column": "channel_id",  "name": "xxhash" },
        { "column": "message_id",  "name": "message_lookup" }
      ]
    }
  }
}
```

The `owner` field tells Vitess to keep the lookup table in sync automatically: insert a message and Vitess writes the lookup row in the same transaction. The cost is real and worth stating plainly. A lookup vindex turns one write into two writes (the row plus its lookup entry), and the lookup table is itself a sharded table that can become a hot spot. Lookup vindexes are how you avoid scatter-gather on secondary access patterns, and they are not free; you are trading write amplification for read locality. The VSchema is also a contract you cannot casually break: change a table's primary vindex and you have changed where every row should live, which is not an edit, it is a reshard.

## 4. Single-shard versus scatter-gather: the cost model that shapes your schema

Here is the rule that governs every query in a sharded Vitess deployment, and it is worth memorizing: **a query that carries the sharding key in its `WHERE` clause hits one shard; a query that does not hits all of them.** Everything about performance, and most of what Slack spent three years fighting, follows from that one sentence.

![Scatter-gather fans a key-less query out to every shard and merges in vtgate, while a query carrying the shard key collapses to a single-shard round trip that scales horizontally](/imgs/blogs/slack-and-youtube-on-vitess-6.webp)

Consider these two queries against a `messages` table sharded by `channel_id`:

```sql
-- Single-shard: vtgate hashes channel_id, routes to ONE shard.
-- Latency and cost are independent of how many shards exist.
SELECT * FROM messages
WHERE channel_id = 'C0288'
  AND ts > '2026-06-01'
ORDER BY ts DESC
LIMIT 50;

-- Scatter-gather: no channel_id in the predicate, so vtgate fans
-- the query out to EVERY shard, collects up to 50 rows from each,
-- then merges and re-sorts the union in its own memory to find
-- the global top 50.
SELECT * FROM messages
WHERE author_id = 'U7781'
ORDER BY ts DESC
LIMIT 50;
```

The first query is what you want: vtgate computes the shard from `channel_id`, sends one sub-query to one vttablet, and returns. Add shards and it does not get slower. The second query is the trap. With no `channel_id`, vtgate cannot know which shard holds `author_id = 'U7781'`'s messages, so it asks all of them. Its tail latency becomes the latency of the *slowest* shard responding, its throughput cost grows with the shard count, and the `ORDER BY ... LIMIT` forces vtgate to gather more rows than it returns and re-sort them in memory. A scatter-gather that is fine at 4 shards melts a cluster at 256. We go deep on the merge, sort, and aggregation mechanics in [cross-shard queries and distributed joins](/blog/software-development/database-scaling/cross-shard-queries-and-distributed-joins); the practical takeaway is that scatter rate is the metric you watch.

Vitess gives you guardrails so a scatter does not become a self-inflicted outage. You can configure vtgate to reject unsharded scatter queries outright (`--no_scatter`), to require a `LIMIT` on scatters, and you can use `vtexplain` to see a query's routing plan before it ships:

```bash
# Show the routing plan for a query without running it.
vtexplain \
  --vschema-file vschema.json \
  --schema-file schema.sql \
  --shards 4 \
  --sql "SELECT * FROM messages WHERE author_id = 'U7781' LIMIT 50"
# => ... "ScatterGather" ... selects from all 4 shards
```

When `vtexplain` prints `ScatterGather` for a query you expected to be cheap, you have found a bug in your schema or your access pattern, and you fix it by adding the sharding key to the query or by adding a lookup vindex. The whole craft of running on Vitess is keeping the common queries single-shard and making the rare scatter queries cheap or asynchronous. This is the lever both YouTube and Slack pulled hardest, and it is why they chose their sharding keys the way they did.

## 5. YouTube: the origin story

Vitess was born inside YouTube in 2010 as an internal tool, open-sourced in 2011, and got its first public commit on GitHub in February 2012. It joined the Cloud Native Computing Foundation as an incubation project in 2018 and graduated in 2019, the same tier as containerd and Fluentd. The timeline below traces that arc next to Slack's adoption, because the two stories interlock: Slack started its migration in 2017, while Vitess was still maturing in CNCF, and bet on it precisely because the project had a decade of YouTube scale behind it.

![Timeline interleaving Vitess's YouTube origins (2010 internal, 2011 open-sourced, 2012 GitHub, 2018 CNCF incubation, 2019 graduation) with Slack's 2017-2020 migration to 99 percent of traffic](/imgs/blogs/slack-and-youtube-on-vitess-3.webp)

YouTube built Vitess to solve four problems that all show up the moment you shard a high-traffic MySQL fleet, and each one maps to a piece of the architecture we just toured.

**Connection limits.** This was the first fire. MySQL allocates per-connection memory and a thread per connection, so a frontend fleet of thousands of processes, each opening connections to hundreds of shards, exhausts a MySQL server through bookkeeping alone. vttablet's connection pool was the answer: the app's many connections terminate at vtgate, vtgate's connections terminate at vttablet, and vttablet multiplexes them onto a small, bounded pool of real MySQL connections. The fan-in is enormous, and it is invisible to the application.

**Query consolidation.** YouTube's traffic had a brutal property: a popular video meant thousands of near-simultaneous identical reads for the same row. vttablet consolidates these. If an identical read is already in flight, later callers wait for the first one's result instead of issuing their own query. One physical query serves a thundering herd of logical ones. This is the same instinct as request coalescing in a cache layer, applied at the database agent; it is closely related to the patterns in [cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd).

**Protecting MySQL from itself.** A single unbounded query, a missing `LIMIT`, a `SELECT *` on a huge table, can drag down a shard that thousands of users depend on. vttablet rewrites and guards: it caps result sets, enforces query timeouts, throttles transactions that hold rows too long, and rejects queries that would obviously hurt. YouTube learned that at scale you cannot trust every query the application emits, even your own, so you put a guard between the query and the disk.

**Resharding live as it grew.** YouTube could not take downtime to add capacity, and "add a shard" the naive way means rebalancing data, which means moving rows while they are being written. Vitess made resharding an online, routine operation rather than a heroic project, which is the feature that turned sharding from a one-time architecture decision into a continuous capacity-management practice. We come back to exactly how that works in section 8.

The throughline is that YouTube did not set out to build a database. They set out to keep MySQL, the thing they trusted, and wrap it in just enough machinery to scale it horizontally without rewriting the application or giving up SQL. That conservative instinct is exactly what made Vitess attractive to the next team.

## 6. Slack's migration: retrofitting sharding onto a grown fleet

Slack's pre-Vitess architecture was, on paper, a textbook sharded design, and walking through why it failed is more instructive than any greenfield example. Every workspace's data, channels, messages, members, files, lived together on a single shard. Each shard was at least two MySQL instances in different datacenters replicating to each other asynchronously, an active-active setup for datacenter resilience. The "webapp" monolith held the routing logic: given a workspace, look up its shard in a metadata table, then open a connection directly to that MySQL host. Simple, comprehensible, and for years, correct.

![Slack's sharding model before and after Vitess: app-routed per-workspace shards with hot spots and no replicas, versus vtgate-routed finer keys with load spread across the fleet and automatic failover](/imgs/blogs/slack-and-youtube-on-vitess-4.webp)

The before-and-after captures what broke and what replaced it. The workspace-as-shard-unit model has a fatal coupling: load is granular at the workspace level and nothing finer. As Slack onboarded ever-larger enterprise customers, a single huge workspace's traffic landed entirely on a single shard, and that shard reached the largest hardware available with nowhere to go. Meanwhile the long tail of small workspaces left the rest of the fleet massively underutilized. Slack was simultaneously out of capacity on its hottest shards and wasting most of its fleet, which is the worst of both worlds. Three more pains compounded it. Availability was coupled to a single shard: login, messaging, joining a channel, all required that one workspace's database to be up, so a single hot shard's trouble was a customer-visible outage. The topology had no read replicas in the serving path, because the application routed straight to hosts. And operating the whole thing required a mountain of bespoke internal tooling for something Vitess provided off the shelf: failovers, backups, and topology management.

The motivation for moving was not only the hot shards; it was flexibility. Slack wanted to shard by keys finer than a workspace so a giant customer's load could spread, and they wanted the freedom to shard *different tables by different keys*. Some data is naturally per-channel, some per-user, some genuinely per-workspace. With routing buried in the application, every such change was a code change in the monolith. With Vitess, the sharding key is declared in the VSchema and enforced by vtgate, so the application keeps issuing ordinary SQL while the routing strategy evolves underneath it. Concretely, instead of putting every message of every channel in a workspace onto the same shard, Slack chose to shard message data by `channel_id`, which spread a big customer's load across many shards and erased the single-customer hot spot. Other tables they sharded by `user_id`, others by `workspace_id`. The per-table choice is the whole point.

> The deepest lesson from Slack's migration is that the sharding *key* is a product decision disguised as an infrastructure one. Workspace was the right unit until the largest workspace outgrew a server, and no amount of operational heroics could fix a key that was too coarse. Vitess did not save them; choosing `channel_id` did. Vitess made choosing it survivable.

## 7. How Slack moved table by table, behind vtgate

The migration is the part that sounds impossible until you see the mechanism: Slack moved a huge, hot, live MySQL fleet onto Vitess over three years, table by table, query by query, while serving production traffic the whole time, and they did it *behind vtgate* so the application believed it was still talking to one MySQL. By the end of 2020, three years in, 99 percent of all Slack MySQL traffic ran through Vitess.

![Slack's table-by-table migration: app writes dual-write to both stores, a backfill seeds historical rows into the Vitess table, and a double-read diff compares every read until old and new agree, then traffic cuts over](/imgs/blogs/slack-and-youtube-on-vitess-7.webp)

The figure is the safe-migration recipe Slack used for each table, and every arrow is load-bearing. The hard problem in moving a live table is that it is *live*: rows are changing while you copy them. Slack's approach had three coordinated parts. A **generic backfill system** cloned the existing table's historical rows into the new Vitess-backed table. Because the source kept changing during the copy, the new table also had to receive ongoing writes, so writes were sent to both the legacy store and the Vitess store during the transition. And critically, a **parallel double-read diffing system** read from both the old and new tables on live traffic and compared the results, so the team had continuous, quantitative proof that the Vitess-powered table returned byte-identical answers before they trusted it. Only when the diff rate held at zero did they cut reads over to Vitess, then writes, then retire the legacy table.

They proved the whole approach on something small first, integrating an RSS feed into a Slack channel, to validate the end-to-end path on a low-stakes feature before betting the messages table on it. And the messages table was the real test: at one point a single table comprised roughly 20 percent of Slack's entire query load, and moving it was an engineering project unto itself. Doing this work made Slack one of the largest contributors to the Vitess project, because operating it at their scale surfaced bugs and missing features that they then fixed upstream.

The scale they reached is the argument for the whole exercise. At peak, Slack served around **2.3 million queries per second** through Vitess, roughly 2 million reads and 300 thousand writes, at a **median latency near 2 ms and a p99 near 11 ms**, across multiple clusters with dozens of keyspaces spread over six geographic regions. The resilience showed during the March 2020 work-from-home surge, when Slack's query rate jumped 50 percent in a single week and the Vitess fleet absorbed it, the kind of event the old per-workspace model would have met with hot-shard outages. The single most important property in all of this is the one the application never noticed: every one of those queries was an ordinary MySQL statement to a vtgate that looked exactly like a MySQL server, which is what made a three-year incremental migration possible at all.

## 8. Online resharding via VReplication

The feature that turns sharding from a one-time decision into a routine operation is **VReplication**, the engine underneath `Reshard` and `MoveTables`. It is what lets you split a shard, or move a table between keyspaces, with no downtime and a built-in correctness check. Slack's plan to "move whole shards instead of tables" in later phases, and YouTube's original need to add capacity live, both rest on it.

![Online resharding with VReplication: copy rows to the new shards, tail the binlog to stay in sync, run VDiff to verify, then SwitchTraffic for reads before writes, achieving the split with no downtime](/imgs/blogs/slack-and-youtube-on-vitess-5.webp)

The figure shows the five phases, and they are the same whether you are splitting one shard into two or sharding an unsharded keyspace for the first time. **Copy**: VReplication bulk-copies the rows that belong on each new shard, using the vindex to decide which rows go where. **Running**: because the source is still taking writes during the copy, VReplication tails the source's binary log and applies the changes to the new shards continuously, so the targets catch up and then stay caught up in near-real-time. **VDiff**: before you trust anything, you run a row-by-row checksum of source against target to prove they are identical, which is the step that makes resharding a verified operation rather than an act of faith. **SwitchTraffic (reads)**: you flip read traffic to the new shards first, because reads are reversible, watch your dashboards, and roll back instantly if anything looks wrong. **SwitchTraffic (writes)**: only then do you cut writes over and retire the old shard. The full operation runs from the control plane and reads, in modern Vitess, roughly like this:

```bash
# 1. Create the reshard: split shard -80 into -40 and 40-80.
vtctldclient Reshard create \
  --workflow split80 \
  --target-keyspace slack \
  --source-shards '-80' \
  --target-shards '-40,40-80'

# 2. Watch the copy + catch-up progress.
vtctldclient Workflow --keyspace slack show --workflow split80

# 3. Verify the new shards row-for-row against the source.
vtctldclient VDiff create --target-keyspace slack --workflow split80

# 4. Flip reads first (reversible), then writes.
vtctldclient Reshard SwitchTraffic \
  --workflow split80 --target-keyspace slack --tablet-types REPLICA
vtctldclient Reshard SwitchTraffic \
  --workflow split80 --target-keyspace slack --tablet-types PRIMARY

# 5. Tear down the old shard once you are confident.
vtctldclient Reshard complete --workflow split80 --target-keyspace slack
```

`MoveTables` is the same engine pointed at a different job: instead of splitting a shard within a keyspace, it streams a set of tables from one keyspace to another, which is exactly how you migrate a table out of a legacy unsharded database and into a sharded one without downtime.

```bash
# Move the messages tables from the legacy keyspace into the sharded one.
vtctldclient MoveTables create \
  --workflow move_messages \
  --source-keyspace legacy \
  --target-keyspace slack \
  --tables 'messages,channels'
```

The second-order insight is that VReplication is not only a resharding tool; it is a general-purpose change-data-capture and migration engine. The same binlog-tailing machinery powers materialized views that re-shard on a different key, online schema changes, and feeding downstream systems. Once you have a reliable, verified way to stream MySQL changes from anywhere to anywhere, a surprising amount of your data-movement tooling collapses into it. We treat the broader online-migration discipline, including the dual-write and verify pattern, in [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime).

## The comparison, with the operational cost made explicit

Pulling the two approaches side by side, with everything we have covered, makes the tradeoff concrete. This is the table to put in front of anyone arguing for either choice.

| Dimension | Hand-rolled app-level sharding | Vitess |
| --- | --- | --- |
| Routing logic | In application code, per service | Declared in VSchema, executed by vtgate |
| Adding a query pattern | New routing code, redeploy | New query; add a lookup vindex if it scatters |
| Cross-shard read | You scatter, merge, re-sort by hand | vtgate scatter-gathers and merges automatically |
| Resharding | Bespoke, risky, often with downtime | `Reshard` + VReplication + VDiff, online, verified |
| Routing by non-key column | A lookup table you maintain | A Vitess-managed lookup vindex |
| Connections to MySQL | App holds connections to every shard | vttablet pools; app talks only to vtgate |
| Failover and backups | Your own tooling | Built into vttablet and the control plane |
| Cross-shard transactions | You build 2PC or avoid them | Available (best-effort or 2PC), discouraged |
| Fixed operational cost | Low | High: you operate vtgate, vttablet, topo, vtctld |
| Best fit | Modest shard counts, simple access | Large fleets, evolving keys, online resharding |

The pattern is clear. Hand-rolled sharding wins on day one and loses on year three; Vitess loses on day one and wins at the scale where the alternative's complexity would have buried you. Both YouTube and Slack were unambiguously on the Vitess side of that line. Most teams are not, and pretending otherwise is how you end up operating a Vitess cluster to serve a workload that fit on one Postgres box.

## Lessons from production

These are the recurring failure modes and wins that show up once you actually run on Vitess, drawn from the patterns the YouTube and Slack stories illustrate and from the common shape of Vitess incidents.

### 1. The creeping scatter rate

A team ships on Vitess with 95 percent of queries single-shard and a healthy cluster. Six months later p99 is creeping up and no single query looks slow. The cause is almost always a slow rise in the *scatter rate*: new features added queries that omit the sharding key, each cheap in isolation, collectively turning a fraction of traffic into all-shards fan-outs. The fix is not a bigger cluster; it is finding the scatter queries with `vtexplain` and the vtgate query logs, then adding the sharding key to the predicate or a lookup vindex to route them. The lesson: scatter rate is a first-class SLO, and you alert on it, because it degrades silently.

### 2. The big customer that outgrew the workspace

This is Slack's story in miniature, and it recurs everywhere a coarse key is chosen. A sharding key that is "one tenant per shard" works beautifully until one tenant is bigger than a server. The symptom is a single hot shard at 100 percent while the fleet idles. There is no operational fix for a key that is too coarse; you must reshard onto a finer key, which is why Slack moved messages from `workspace_id` to `channel_id`. The lesson: choose the sharding key for the *largest* tenant you will ever have, not the median one.

### 3. The reshard that would not cut over

A reshard runs for days in the Running phase, perfectly caught up, but the team is afraid to flip writes because they cannot prove the new shards are correct. The mistake is skipping or rushing VDiff. The whole point of the verify phase is to convert "we think it copied right" into "we checksummed every row and they match," and a reshard without a clean VDiff is a reshard you should not complete. The lesson: VDiff is not optional ceremony; it is the difference between a routine operation and a data-loss incident.

### 4. The lookup vindex that doubled the write cost

A team adds a lookup vindex to kill a scatter query, the read gets fast, and a week later write latency on the owning table has quietly risen. A lookup vindex turns one write into two, the row and its lookup entry, in the same transaction, and the lookup table can itself become a hot shard. The fix is to use lookup vindexes deliberately, only where read locality is worth the write amplification, and to shard the lookup table sensibly. The lesson: every routing convenience has a write-side bill, and you should know its size before you add it.

### 5. The connection pool that capped throughput

Throughput plateaus well below CPU saturation, and the culprit is vttablet's transaction pool: long-running transactions hold pooled connections, and once the pool is exhausted new transactions queue. This is usually an application problem (a transaction held open across a slow network call, or a forgotten `COMMIT`) surfacing as a database limit. The fix is to shorten transactions, not just to enlarge the pool, because a bigger pool only delays the wall. The lesson: pooling changes your transaction discipline; transactions are now a shared, finite resource and holding one is anti-social.

### 6. The unbounded result set that OOM'd a vttablet

A query without a `LIMIT` against a large table tries to stream millions of rows back, and either vttablet's memory or vtgate's merge buffer blows up. This is exactly the failure vttablet's query guards exist to prevent, and it bites teams that disabled or never configured them. The fix is to enforce result-set caps and require `LIMIT` on scatters, treating an unbounded query as a bug. The lesson: at scale you cannot trust every query the application sends, including your own, so the guard between the query and the disk is mandatory, not optional.

### 7. The COVID spike the cluster shrugged off

Not every story is a failure. When Slack's traffic jumped 50 percent in a single week in March 2020, the Vitess fleet absorbed it where the old per-workspace model would have produced hot-shard outages, because load was spread across many shards by a fine key and vtgate could route around any single slow shard. The lesson: the payoff of a good sharding key and a proxy that load-balances is invisible until the day it saves you, and that day is a traffic spike you did not schedule.

### 8. The cross-shard join that melted a shard

An innocent-looking join across two tables sharded by different keys forces vtgate to pull a large intermediate result from one set of shards to join against another, hammering a shard and ballooning vtgate memory. The fix is to co-locate joined tables on the same sharding key (so the join is single-shard) or to denormalize so the join is unnecessary. The lesson, and it echoes the whole post: design your schema so the queries you run most are single-shard, and treat any common cross-shard join as a schema bug to be fixed, not a query to be tuned.

## When to reach for Vitess, and when not to

Reach for Vitess when:

- You are already on MySQL at a scale where a single primary cannot hold the write throughput, the connection count, or the largest table, and vertical scaling is out of road.
- You need to shard but cannot afford to rewrite thousands of queries, and you want the application to keep speaking ordinary SQL to one logical database.
- Your access patterns have a natural high-cardinality sharding key (channel, user, tenant) that keeps the hot queries single-shard.
- You expect to reshard repeatedly as you grow, and you want that to be a routine, verified, online operation rather than a recurring crisis.
- You have the operational maturity, or a managed provider, to run vtgate, vttablet, a topology service, and the control plane as first-class infrastructure.

Skip Vitess when:

- Your workload fits on one well-tuned MySQL or Postgres, with read replicas, for the foreseeable future. Most workloads do, and adding Vitess buys you complexity you will not use.
- Your access patterns have no good sharding key, so most queries would scatter; Vitess makes scatter survivable, not cheap, and a workload that is all scatter is a workload Vitess cannot save.
- You need rich cross-shard transactions and joins as the common case; Vitess supports cross-shard transactions but discourages them, and a schema that depends on them is fighting the system.
- You are early enough that you do not yet know your access patterns, in which case sharding now, by any method, is premature, and you should buy a bigger box and keep learning.
- You do not have the people to operate it. Vitess is a distributed system you now own, and an under-staffed Vitess cluster is a worse outage than the single MySQL it replaced.

The unifying lesson of both migrations is the one in the title: Vitess is "sharded MySQL as a service." You pick a sharding key, you declare it in a VSchema, you route through a proxy that hides the shards, and you reshard online when you grow. YouTube built that pattern because it was the only way to scale the thing they trusted without giving it up, and Slack adopted it because they had grown past hand-rolled sharding and needed the same escape hatch. The cost, always, is the operational weight of running Vitess itself. If your scale justifies that weight, there is no better-proven way to make a fleet of MySQL boxes behave like one database. If it does not, the most senior move is to keep your single database and come back when the workload, not the architecture diagram, forces your hand.

## Further reading

- [Vitess: sharding MySQL at YouTube scale](/blog/software-development/database/vitess-sharding-mysql-at-scale) — the mechanics of vtgate, vttablet, vindexes, and VReplication in depth.
- [Cross-shard queries and distributed joins](/blog/software-development/database-scaling/cross-shard-queries-and-distributed-joins) — what scatter-gather, merge, and distributed joins actually cost.
- [Resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) — the dual-write, backfill, and verify discipline behind online migrations.
- [Choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) — why the key is a product decision and how to choose one you will not have to undo.
- [Vitess documentation](https://vitess.io/docs/) — the official reference for VSchema, vindexes, and the `Reshard`/`MoveTables` workflows.
- [Scaling Datastores at Slack with Vitess](https://slack.engineering/scaling-datastores-at-slack-with-vitess/) — Slack Engineering's account of the three-year migration.

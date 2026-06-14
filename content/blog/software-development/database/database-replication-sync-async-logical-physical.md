---
title: "Database Replication: Synchronous, Asynchronous, Logical, and Physical"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A principal-engineer tour of single-leader replication — physical vs logical, synchronous vs asynchronous, the lag anomalies that break read-your-writes, and the failover playbook that keeps split-brain from eating your data."
tags:
  [
    "replication",
    "postgres",
    "mysql",
    "high-availability",
    "streaming-replication",
    "logical-replication",
    "failover",
    "replication-lag",
    "distributed-systems",
    "database",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/database-replication-sync-async-logical-physical-1.webp"
---

There is a specific kind of 3 a.m. page that teaches you everything you need to know about replication, and it always sounds the same. The primary database is fine. CPU is fine, disk is fine, the application is healthy. But your support queue is filling up with users swearing they saved a change that vanished — they edited a profile, hit reload, and the old value came back. Or worse: a teammate ran a "routine" network maintenance, a 43-second blip happened, and now two database servers in two regions each believe they are the one true leader, both accepting writes, and you are about to spend the next twenty-four hours reconciling them by hand.

Every one of those incidents is a replication story. Replication is the single most load-bearing piece of database infrastructure that most engineers treat as a checkbox — "yeah, we have a read replica" — without understanding the three or four orthogonal decisions hiding inside that one sentence. Is the replica kept up to date *synchronously* or *asynchronously*? Does it ship raw disk bytes (*physical*) or decoded row changes (*logical*)? What exactly does "committed" mean when a standby is involved? And when the leader dies, who decides who takes over, and what stops two servers from both deciding it's them?

This article is the long version of the answer. The diagram below is the mental model for the entire piece: a single leader accepts all writes, turns them into a change stream, and fans that stream out to a fleet of followers that serve reads. Almost everything else — sync vs async, physical vs logical, lag, failover — is a detail of how that stream is shipped and what guarantees you attach to it.

## The mental model: one writer, many readers

![Single-leader replication: all writes funnel through one leader and the WAL change stream fans out to read-only followers](/imgs/blogs/database-replication-sync-async-logical-physical-1.webp)

Read the figure left to right. The application sends every write — every `INSERT`, `UPDATE`, `DELETE`, `DDL` — to exactly one node, the **leader** (Postgres and the literature call it the *primary*; MySQL historically called it the *master*, now the *source*). The leader is the only node permitted to mutate data. Every other node is a **follower** (Postgres *standby*, MySQL *replica*) that holds a copy and serves read-only queries.

The magic is in the middle box: the **change stream**. The leader does not re-execute your SQL on each follower. Instead it writes a durable, ordered log of every change it made — Postgres calls this the **Write-Ahead Log (WAL)**, MySQL calls it the **binary log (binlog)** — and ships that log to the followers, which apply it in order. Because the log is ordered and append-only, a follower that applies it faithfully ends up byte-identical (physical) or row-identical (logical) to the leader, modulo however far behind it currently is.

That single architectural choice — one writer, an ordered change log, many appliers — is what Martin Kleppmann's *Designing Data-Intensive Applications* (Chapter 5, the canonical text on this topic) calls **single-leader** or **leader-based** replication. It is by far the most common topology in production: it is what Postgres streaming replication, MySQL replication, MongoDB replica sets, and the relational layer of nearly every "cloud-native" database give you out of the box. The reason it dominates is that it sidesteps the hardest problem in distributed data — write conflicts — by construction. If there is only ever one writer, two conflicting writes to the same row cannot happen, because they are serialized through one node's commit order. Multi-leader and leaderless systems give that up to gain write availability, and they pay for it with conflict resolution; we will draw that boundary explicitly at the end and forward-reference where that story continues.

> Single-leader replication is the decision to make write *conflicts* impossible by accepting a single point of write *availability*. Every other replication design is a different answer to "what do we do when the leader is gone?"

The rest of this article is a tour of the change stream and its guarantees. We will start with *why* you would replicate at all, then dissect the four axes — physical vs logical, sync vs async — that define how the stream behaves, then spend a long time on **replication lag**, the asynchronous tax that causes the most user-visible bugs, and finally on **failover**, the operation that causes the most catastrophic ones. Along the way: runnable Postgres and MySQL config, app-level patterns in code, comparison tables, and a stack of production case studies from GitHub, GitLab, Notion, and others who learned these lessons the expensive way.

## 1. Why replicate at all: four problems, one mechanism

**Senior rule of thumb: never add a replica without naming which of the four problems it solves, because the right replication settings are different for each one.** A replica configured for high availability and a replica configured for analytics offload want opposite things, and conflating them is how you end up with an HA standby that falls hours behind because someone pointed a reporting dashboard at it.

![Four reasons to run a replica: high availability, read scaling, geo locality, and offload — each with its own lag budget](/imgs/blogs/database-replication-sync-async-logical-physical-9.webp)

The four motivations, as the figure lays out:

**High availability (HA).** A single database server is a single point of failure. Disks die, kernels panic, EBS volumes detach, availability zones go dark. If your only copy of the data is on that one machine, its death is your outage and possibly your data loss. A replica that is kept reasonably current can be *promoted* to leader in seconds to minutes, bounding your **Recovery Time Objective (RTO)** — how long you are down — and, if the replica is synchronous, your **Recovery Point Objective (RPO)** — how much data you lose — to zero. HA wants the freshest possible replica and a robust automatic-or-manual promotion path. This is the motivation that, done wrong, causes the GitHub-style split-brain disaster we will dissect at length.

**Read scaling.** Most OLTP workloads are read-heavy: 10:1, 100:1, sometimes 1000:1 reads to writes. A single leader can only do so much, and reads compete with writes for buffer cache, CPU, and I/O. The classic move is to route `SELECT`s to a pool of read-only followers and keep the leader for writes. Crucially, this scales reads but **not writes** — every follower still has to apply the full write stream, so adding followers does nothing for write throughput and in fact adds replication load to the leader. Read scaling tolerates some lag (the follower is allowed to be a few hundred milliseconds behind) but the lag is exactly what breaks read-your-writes, as we will see.

**Geo locality.** If your users are in Frankfurt and your database is in Virginia, every query pays ~90 ms of round-trip latency, and a page that does ten sequential queries pays ~900 ms before it does any work. A read replica physically near the user collapses read latency to single-digit milliseconds. Writes still cross the ocean to the leader, but reads — the bulk of traffic — go local. The lag budget here is generous in absolute terms but the cross-region link is exactly where lag spikes and partitions live.

**Offload.** Backups, `pg_dump`, full-table analytical scans, BI dashboards, search-index rebuilds, and CDC pipelines are all heavy, long-running, cache-polluting operations that you do not want competing with latency-sensitive OLTP on the leader. Point them at a dedicated follower. This follower can be allowed to lag substantially — an analytics query that reads data from thirty seconds ago is almost always fine — which means you can also tune it differently (e.g. larger `work_mem`, longer `max_standby_streaming_delay`).

| Motivation | Lag tolerance | What it wants from replication | Failure if misconfigured |
| --- | --- | --- | --- |
| High availability | Near zero (sync candidate) | Fresh replica + safe promotion | RPO loss, split-brain |
| Read scaling | Low (sub-second) | Many cheap async followers | Stale reads, broken read-your-writes |
| Geo locality | Low locally, high cross-region | Replica near user, write to leader | Latency spikes, partition-induced lag |
| Offload / analytics | High (seconds to minutes) | One dedicated, isolated follower | Long queries cancel under `max_standby_*` |

### Second-order optimization: don't make one replica do four jobs

The cheap mistake is to provision two database nodes — "primary and replica" — and then route HA failover, read scaling, *and* analytics at the same single replica. When the analytics job runs a `SELECT * FROM events` table scan, it floods the replica's buffer cache and its apply process falls behind, so your HA RPO silently degrades and your read-scaling followers serve staler data exactly when load is highest. The fix is the cascading topology we cover in §9: separate the synchronous HA candidate from the read-scaling pool from the analytics replica, even if it costs another node. Replication is cheap; reconciling a split-brain is not.

## 2. The two questions that define every replica

Before the mechanics, internalize the two orthogonal axes. They are independent — you can pick any combination — and they are the source of essentially all the vocabulary confusion in this space.

**Axis 1 — What is shipped: physical or logical?** Physical replication ships the WAL/binlog at the *storage* level: the exact byte changes to specific disk blocks. The follower is a byte-for-byte clone. Logical replication ships *decoded row changes*: "row with primary key 42 in table `users` changed column `bio` to this value." The follower applies these as normal SQL-ish operations and can differ structurally from the leader.

**Axis 2 — When the leader acks: synchronous or asynchronous?** Asynchronous: the leader commits locally and tells the client "done" immediately, shipping the change to followers afterward. If the leader dies before the change reaches a follower, that committed write is lost. Synchronous: the leader waits for at least one follower to confirm receipt (or apply) *before* telling the client "done." No acknowledged write can be lost, but every commit now pays a network round trip.

These two axes are independent. You can have asynchronous physical (the Postgres default streaming replica), synchronous physical (a quorum HA standby), asynchronous logical (most CDC pipelines), and so on. The rest of the article is essentially a deep dive into each axis and then the consequences — lag and failover — that fall out of choosing asynchronous, which almost everyone does for most replicas.

| | Physical (block-level) | Logical (row-level) |
| --- | --- | --- |
| **Async** | Postgres streaming standby (default); MySQL async replication | Postgres logical subscriber; MySQL binlog → Debezium CDC |
| **Sync** | Postgres `synchronous_standby_names`; MySQL semi-sync (loss-less) | Rare; logical sync exists but is unusual |

## 3. Physical / streaming replication: ship the bytes

**Senior rule of thumb: physical replication is the right default for HA and read scaling because it is the cheapest, lowest-lag way to make an exact copy — but it locks you to one major version and gives you no selectivity.**

Physical replication treats the database as what it physically is on disk: a set of files made of pages. The leader already maintains a Write-Ahead Log for crash recovery — every change is written to the WAL *before* the data pages are modified, so that after a crash the database can replay the WAL to reach a consistent state. (If you want the durability story in depth, the WAL is also the spine of [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer) when you start reasoning about why writes touch disk the way they do.) The insight of physical replication is: that same WAL stream, shipped to another machine and replayed there, reconstructs the database exactly. Replication is just crash recovery that never stops.

In Postgres this is **streaming replication**. The leader runs a process called `walsender`; the standby runs a `walreceiver` that opens a TCP connection, requests the WAL from a given position, and streams it continuously. The standby replays each WAL record into its own page cache and disk. Because it is replaying byte-level page changes, the standby is an exact physical clone — same page layout, same internal row positions, same everything. That is also why both nodes **must run the same major version**: WAL format is internal and version-specific, and a different major version reads it differently.

Let me show the actual config, because the mechanics are clearer in real files than in prose. Here is a minimal Postgres 16 streaming setup.

On the **leader** (`postgresql.conf`):

```ini
# Generate enough WAL detail for replicas (default 'replica' is fine for physical).
wal_level = replica
# How many concurrent walsender processes (one per streaming standby + tools).
max_wal_senders = 10
# Keep WAL around for standbys; with slots this becomes unnecessary, see below.
wal_keep_size = 1GB
# Allow read queries on standbys (set on the standby, but pairs with this).
hot_standby = on
```

And `pg_hba.conf` to let the standby authenticate for replication:

```
# TYPE  DATABASE        USER         ADDRESS            METHOD
host    replication     replicator   10.0.0.0/24        scram-sha-256
```

Create the replication role on the leader:

```sql
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'a-real-secret';
```

On the **standby**, you bootstrap from a base backup and then point it at the leader. The clone:

```bash
pg_basebackup \
  --host=leader.db.internal \
  --username=replicator \
  --pgdata=/var/lib/postgresql/16/main \
  --wal-method=stream \
  --write-recovery-conf \
  --create-slot --slot=standby_1 \
  --checkpoint=fast --progress
```

The `--write-recovery-conf` flag writes the standby's connection settings into `postgresql.auto.conf` and creates a `standby.signal` file (which is what tells Postgres 12+ to boot in standby mode). The key line it writes:

```ini
# postgresql.auto.conf on the standby
primary_conninfo = 'host=leader.db.internal user=replicator passhost=... application_name=standby_1'
primary_slot_name = 'standby_1'
```

Start the standby and it connects, catches up, and then streams continuously. You can verify from the leader:

```sql
SELECT application_name, state, sync_state,
       sent_lsn, write_lsn, flush_lsn, replay_lsn,
       pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replay_lag_bytes
FROM pg_stat_replication;
```

That `pg_stat_replication` view is the single most important query in your operational toolkit; we will return to it constantly when measuring lag.

### Replication slots: never lose your place

![WAL streaming through a replication slot: the slot pins the leader's WAL at the standby's confirmed LSN](/imgs/blogs/database-replication-sync-async-logical-physical-5.webp)

The figure traces the whole physical pipeline, and the box that matters most is the **replication slot**. Here is the problem it solves. The leader's WAL is finite — old WAL segments (16 MB files) get recycled once the leader no longer needs them for its own recovery. But a standby that has been disconnected (network blip, reboot, slow apply) might still need WAL the leader has already deleted. When that happens the standby cannot catch up by streaming; it has to be re-cloned from scratch. On a multi-terabyte database, that is hours of pain.

A **replication slot** fixes this. Every position in the WAL is identified by a **Log Sequence Number (LSN)** — a monotonically increasing 64-bit offset into the logical WAL stream, printed as `16/B374D848`. A slot is a named bookmark on the leader that records the LSN each standby has confirmed receiving. The leader then refuses to recycle WAL past the oldest slot's confirmed LSN. The standby can disconnect for an hour and reconnect; the WAL it needs is still there.

```sql
-- On the leader: create a physical slot the standby can use.
SELECT pg_create_physical_replication_slot('standby_1');

-- Inspect slots and how much WAL they are pinning.
SELECT slot_name, slot_type, active,
       restart_lsn,
       pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained_wal
FROM pg_replication_slots;
```

There is a sharp edge here, and it is the most common slot-related outage: a slot for a standby that is **gone and never coming back** will pin WAL forever, and the leader's disk fills up until it crashes. A slot is a promise to retain WAL; an abandoned slot is a memory leak that fills your disk. Postgres 13+ added `max_slot_wal_keep_size` to cap this — past the cap, the slot is invalidated rather than letting the disk fill. Set it. The senior move:

```ini
# On the leader: invalidate a slot rather than fill the disk, capping retention.
max_slot_wal_keep_size = 100GB
```

This is the exact tradeoff: with a generous cap you preserve the ability of a long-disconnected standby to catch up; with a tight cap you protect the leader's disk at the cost of forcing a re-clone for any standby that lags past it. There is no universally correct value — it depends on your WAL generation rate (check `pg_stat_wal`) and how long a standby might realistically be offline.

### MySQL: binlog positions and GTIDs

MySQL's physical-ish equivalent ships the **binary log**. (I say "physical-ish" because MySQL's binlog is closer to logical than Postgres WAL — more on that in §4 — but operationally it is the standard async replication everyone uses.) The modern setup uses **GTIDs (Global Transaction Identifiers)**, which tag every transaction with a unique ID so a replica knows precisely which transactions it has and has not applied, making failover far less error-prone than the old "file + position" scheme.

On the source (`my.cnf`):

```ini
[mysqld]
server_id              = 1
log_bin                = mysql-bin
binlog_format          = ROW          # see §4 — ROW is the safe default
gtid_mode              = ON
enforce_gtid_consistency = ON
binlog_expire_logs_seconds = 604800   # keep 7 days of binlog
```

On the replica, the modern command is `CHANGE REPLICATION SOURCE TO` (it replaced the deprecated `CHANGE MASTER TO` in MySQL 8.0.23):

```sql
CHANGE REPLICATION SOURCE TO
  SOURCE_HOST     = 'source.db.internal',
  SOURCE_USER     = 'repl',
  SOURCE_PASSWORD = 'a-real-secret',
  SOURCE_AUTO_POSITION = 1;   -- use GTIDs, not file+position

START REPLICA;
SHOW REPLICA STATUS\G   -- check Replica_IO_Running, Replica_SQL_Running, Seconds_Behind_Source
```

`SOURCE_AUTO_POSITION = 1` is the GTID magic: the replica tells the source "here is the set of GTIDs I already have," and the source sends exactly the transactions it is missing. No fragile byte offsets, and a failover can re-point a replica at a newly promoted source without a human computing log positions.

## 4. Logical replication: ship the changes, not the bytes

**Senior rule of thumb: reach for logical replication when you need selectivity, cross-version, cross-engine, or a writable target — accept that it is slower, fussier about conflicts, and does not replicate DDL automatically.**

![Physical replication ships byte-identical WAL blocks while logical replication ships decoded INSERT/UPDATE/DELETE row operations](/imgs/blogs/database-replication-sync-async-logical-physical-2.webp)

The before/after figure is the whole concept. Physical replication, on the left, ships raw WAL bytes and produces a byte-for-byte clone — fast, exact, but rigid: same major version, whole cluster, read-only standby. Logical replication, on the right, runs a **logical decoding** step that reads the WAL and turns it back into a stream of row-level operations — `INSERT this row`, `UPDATE that row's columns`, `DELETE this key` — keyed by the table's replica identity (usually the primary key). Those operations are applied on the subscriber as ordinary writes.

That decoding step unlocks four things physical replication cannot do:

1. **Selectivity.** You publish specific tables (or even filtered rows / specific columns in PG 15+), not the whole cluster. A reporting database can subscribe to just the five tables it needs.
2. **Cross-version.** Because you ship logical row changes, not version-specific WAL format, the publisher and subscriber can be different major Postgres versions. This is the foundation of **near-zero-downtime major upgrades**: stand up the new version as a logical subscriber, let it catch up, then flip traffic.
3. **Cross-engine and CDC.** The decoded stream can feed anything — Kafka, a data warehouse, a search index, another database engine entirely. This is **Change Data Capture (CDC)**, and tools like Debezium are built on exactly this mechanism (Postgres logical decoding, MySQL binlog reading).
4. **Writable target.** The subscriber is a normal, fully writable database. You can add indexes it doesn't have on the publisher, run different extensions, or merge multiple publishers into one subscriber.

Here is a runnable Postgres pub/sub example. On the **publisher**:

```sql
-- wal_level must be 'logical' for logical decoding (a restart-required change).
-- ALTER SYSTEM SET wal_level = 'logical';  then restart.

-- Publish two specific tables (PG 15+ supports row filters and column lists).
CREATE PUBLICATION analytics_pub
  FOR TABLE public.orders, public.order_items;

-- Or publish only paid orders to the analytics subscriber (row filter, PG 15+):
CREATE PUBLICATION paid_orders_pub
  FOR TABLE public.orders WHERE (status = 'paid');
```

On the **subscriber** (a different database, possibly a newer major version):

```sql
CREATE SUBSCRIPTION analytics_sub
  CONNECTION 'host=publisher.db.internal dbname=app user=replicator password=...'
  PUBLICATION analytics_pub
  WITH (copy_data = true, create_slot = true, slot_name = 'analytics_sub');

-- Watch progress:
SELECT subname, received_lsn, latest_end_lsn,
       latest_end_time
FROM pg_stat_subscription;
```

The `copy_data = true` does an initial snapshot of existing rows, then switches to streaming new changes. Under the hood, logical replication still uses a replication slot on the publisher (this time a *logical* slot tied to an output plugin, `pgoutput`), so all the slot caveats from §3 apply — an abandoned logical slot fills the publisher's disk just the same.

### The sharp edges of logical replication

Logical replication is powerful but it is genuinely fussier, and these are the gotchas that bite:

- **DDL is not replicated.** If you `ALTER TABLE orders ADD COLUMN ...` on the publisher, the subscriber does *not* get the column, and replication breaks the moment a new-column write arrives. You must apply schema changes to the subscriber first, in a careful order. This is the single most common way logical replication breaks in production, and it is exactly why [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) become a coordinated dance once logical replication is in the picture.
- **Conflicts cause it to halt.** If the subscriber is writable and someone (or another stream) inserts a row with a primary key the publisher then also sends, you get a unique-violation conflict and the apply worker stops until you resolve it.
- **No sequences by default.** Sequence state isn't replicated logically, which matters at cutover for a major upgrade — you must advance sequences on the new node before taking writes.
- **Replica identity.** `UPDATE` and `DELETE` need to identify the row. With a primary key this is automatic; without one you must set `REPLICA IDENTITY FULL`, which logs the entire old row and is much heavier.

### MySQL binlog formats: row vs statement

MySQL's binlog has long had this physical/logical tension built in, exposed as `binlog_format`. This is one of the highest-leverage correctness knobs in MySQL, so it is worth being precise.

- **`STATEMENT`** logs the actual SQL statement (`UPDATE accounts SET balance = balance * 1.05`). Compact, but **dangerous**: any non-deterministic statement (`NOW()`, `RAND()`, `UUID()`, `LIMIT` without `ORDER BY`, certain trigger interactions) can produce *different results* when re-executed on the replica, silently diverging the data. Per the MySQL manual, statement-based logging cannot safely replicate these.
- **`ROW`** logs the actual changed row images — the before and after of each row touched. Larger binlog, but it is the **safe and now-default** format because the replica isn't re-executing logic, just applying concrete row changes. This is the logical end of the spectrum.
- **`MIXED`** uses statement when it is provably safe and automatically switches to row for non-deterministic statements.

```ini
# my.cnf — ROW is the right default for correctness.
binlog_format = ROW
# Log full before/after images so downstream CDC has complete rows.
binlog_row_image = FULL
```

There is a real compatibility trap the MySQL docs call out explicitly: you cannot mix formats across the replication chain carelessly. A replica cannot convert binlog entries received in `ROW` format back to `STATEMENT` format for its own binlog, so a source-on-ROW / replica-on-STATEMENT configuration breaks replication outright. Standardize on `ROW` everywhere. The storage cost is real (row images are bigger), but the correctness guarantee is worth it, and `ROW` is what every modern CDC tool expects.

## 5. Synchronous vs asynchronous: what "committed" means

**Senior rule of thumb: the default is asynchronous, and you should keep it asynchronous for almost everything — turn on synchronous replication only for the specific data where losing one acknowledged transaction is unacceptable, and only after you understand what it does to your availability.**

This is the axis that confuses the most people, because the word "committed" quietly changes meaning depending on the setting. Let's nail it down with the acknowledgement timeline.

![Asynchronous commit acks after the local WAL flush; synchronous commit waits for a standby to confirm, adding a network round trip](/imgs/blogs/database-replication-sync-async-logical-physical-3.webp)

Trace the figure. At `t0` the client sends `COMMIT`. At `t1` the leader flushes the transaction's WAL to its own local disk. Now the paths diverge:

- **Asynchronous** (`t2`): the leader acknowledges the commit to the client *right now*, the instant its local flush is durable. The change will be shipped to standbys *afterward*. The transaction is committed and durable **on the leader**, but if the leader's disk and the leader itself are destroyed before the WAL reaches a standby, that acknowledged write is **lost**. Your RPO is greater than zero — it equals however much WAL was in flight, i.e. the replication lag at the moment of death.
- **Synchronous** (`t3`–`t5`): the leader ships the WAL to a standby, the standby flushes it (and acks), and only *then*, at `t5`, does the leader acknowledge the commit to the client. Now the write exists durably on two machines before the client is told "done." If the leader dies, the standby has it. RPO = 0 for acknowledged transactions. The cost: every commit pays a network round trip to the standby, so commit latency rises and — crucially — if the standby is unreachable, commits can *block*.

That last point is the part people miss and the reason synchronous replication is dangerous if you misunderstand it. **Naive synchronous replication trades the availability of one node for the durability of the cluster.** If you require a synchronous standby to ack and that standby goes down, your leader will refuse to commit writes — your "high availability" setup just took down your write path because a *replica* failed. This is why production synchronous setups use a *quorum* of multiple candidate standbys, so any single standby failure doesn't block writes.

### Postgres: synchronous_commit levels and quorum standbys

Postgres exposes this with surprising granularity through `synchronous_commit`, which controls *how far* a commit must propagate before the leader acks:

| `synchronous_commit` | Leader acks after… | Durability | Use for |
| --- | --- | --- | --- |
| `off` | nothing — WAL flush is deferred | crash can lose recent local commits (but no corruption) | high-volume metrics/events where loss is OK |
| `local` | local WAL flush only | survives leader crash, not leader loss | default-equivalent when no sync standby named |
| `remote_write` | standby's OS received it (not yet fsync'd) | survives leader loss unless standby also crashes simultaneously | balanced sync |
| `on` | standby flushed WAL to disk | survives leader loss | the strong default for sync |
| `remote_apply` | standby *replayed* it (visible to reads there) | strongest; standby reads see the write | causal read-your-writes across nodes |

A subtle and important note from the Postgres docs: `synchronous_commit = off` does **not** risk *corruption*. A crash might lose some recent allegedly-committed transactions, but the database state is exactly as if those transactions had cleanly aborted — it's a durability tradeoff, not an integrity one. That makes `off` a legitimate, if scary-sounding, choice for clickstream and metrics tables where availability and write throughput matter far more than never losing the last second of events.

To actually require synchronous standbys you set `synchronous_standby_names` on the leader. The most robust form is **quorum-based**:

```ini
# On the leader: commit when ANY 2 of these 3 standbys confirm.
# Survives one standby failure with zero write-path impact.
synchronous_standby_names = 'ANY 2 (standby_1, standby_2, standby_3)'
synchronous_commit = on
```

The `ANY 2 (s1, s2, s3)` syntax means each commit proceeds as soon as *any two* of the three named standbys reply — so one standby can be down or slow and writes keep flowing at full durability. This is exactly the configuration the Postgres documentation recommends for surviving a single standby failure without sacrificing the RPO=0 guarantee. There is also `FIRST 2 (s1, s2, s3)` which prefers the standbys in priority order, useful when one standby is in the same rack and you want it as the primary sync candidate.

Verify which standby is actually synchronous right now:

```sql
SELECT application_name, sync_state, sync_priority,
       replay_lsn, flush_lsn
FROM pg_stat_replication
ORDER BY sync_priority;
-- sync_state: 'sync' (counts toward quorum), 'potential' (standby), 'async'
```

### MySQL: semi-synchronous replication

MySQL's answer is **semi-synchronous replication**, and the "semi" is the honest part of the name. Standard MySQL replication is fully asynchronous. Semi-sync adds a rule: the source waits for **at least one replica to acknowledge that it has received and logged the transaction's events to its relay log** before the source returns success to the client. It is "semi" because the replica acks *receipt*, not *apply* — the change is durable on a second machine's disk but may not yet be visible to queries there.

```sql
-- On the source: install and enable the semi-sync source plugin.
INSTALL PLUGIN rpl_semi_sync_source SONAME 'semisync_source.so';
SET GLOBAL rpl_semi_sync_source_enabled = 1;
-- Wait for at least 1 replica ack:
SET GLOBAL rpl_semi_sync_source_wait_for_replica_count = 1;
-- How long to wait before degrading to async if no replica acks (ms):
SET GLOBAL rpl_semi_sync_source_timeout = 1000;

-- On each replica:
INSTALL PLUGIN rpl_semi_sync_replica SONAME 'semisync_replica.so';
SET GLOBAL rpl_semi_sync_replica_enabled = 1;
```

The `rpl_semi_sync_source_timeout` is the critical knob and the source of the most important semantic: **MySQL semi-sync degrades to asynchronous if no replica acks within the timeout.** This is a deliberate availability choice — the source will not block writes forever waiting for a dead replica — but it means semi-sync gives you "RPO=0 *unless* something has gone wrong long enough to trip the timeout, in which case you're silently back to async and can lose data." For true loss-less behavior you set `rpl_semi_sync_source_wait_point = AFTER_SYNC` (the default in 8.0), which makes the source wait for the replica ack *before* committing locally, so a source crash can't expose a transaction the replica never received.

> Synchronous replication doesn't make your data safe; it makes a *specific, named* failure (loss of the leader's local storage) survivable, in exchange for a *different* failure mode (write unavailability when standbys are gone). Quorum and timeouts are how you tune which of those two pains you'd rather have.

### The honest cost table

| Setting | RPO on leader loss | Commit latency | Write availability when a standby dies |
| --- | --- | --- | --- |
| Async (PG default, MySQL default) | > 0 (= lag at death) | local flush only | unaffected |
| MySQL semi-sync | 0 until timeout, then > 0 | + 1 RTT | degrades to async (stays up) |
| PG `synchronous_standby_names` single | 0 | + 1 RTT | **blocks** (single point of failure!) |
| PG `ANY n (... n+k ...)` quorum | 0 | + 1 RTT to fastest n | survives k failures |

The row that ends careers is the third one: naming a *single* synchronous standby. It feels like the safe, strong choice, and it is — right up until that one standby reboots for a kernel patch and your entire write path hangs. Always use a quorum with at least one node of headroom.

## 6. Replication lag: the asynchronous tax

We have established that almost every replica is asynchronous, which means almost every replica is, at any instant, *behind* the leader. The gap is **replication lag**, and it is the source of the most common and most baffling category of bug: data that is correct but appears, to a specific user at a specific moment, to be wrong or to have time-traveled. Kleppmann's Chapter 5 catalogs three distinct anomalies that lag produces, and they have three distinct fixes. Getting these straight is the difference between a system that feels solid and one that generates a steady trickle of "I swear I saved that" tickets.

Lag is not constant. Under light load a Postgres async standby is typically milliseconds behind. Under a write spike, a long-running transaction on the standby blocking apply, a network hiccup, or — the classic — a big batch job, lag can balloon to seconds or minutes. And critically, **lag recovery is not linear**: when a standby falls far behind and then the load that caused it subsides, it catches up faster and faster as the backlog shrinks. GitHub's 2018 postmortem noted exactly this, that recovery "had adhered to a power decay function instead of a linear trajectory," which is why their linear ETAs for catch-up were badly wrong. Measure lag in *bytes* (`pg_wal_lsn_diff`) and in *time* (`now() - pg_last_xact_replay_timestamp()`), not by eyeballing a counter.

```sql
-- On a Postgres standby: time lag (how old is the data I'm serving?)
SELECT now() - pg_last_xact_replay_timestamp() AS time_lag,
       pg_last_wal_replay_lsn() AS applied_lsn;

-- On the leader: byte lag per standby (how much WAL is each one behind?)
SELECT application_name,
       pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) AS byte_lag
FROM pg_stat_replication;
```

### Anomaly 1: read-your-writes (read-after-write)

![Replication lag breaks read-your-writes: a user writes, reloads, and a lagging follower returns the stale pre-write value](/imgs/blogs/database-replication-sync-async-logical-physical-4.webp)

The timeline shows the canonical bug. A user updates their profile bio (`t0`); the leader commits and acks the user (`t1`); the page reloads and — because reads are load-balanced — the reload happens to hit a follower (`t2`) that is 200 ms behind (`t3`); the follower hasn't received the change yet, so it returns the **old** bio (`t4`); the user sees their edit vanish. A moment later (`t5`) the WAL applies and the follower is current, but the damage to user trust is done.

**Read-your-writes consistency** (a.k.a. read-after-write) is the guarantee that *a user always sees their own updates*. It says nothing about seeing other users' updates promptly — only that your own writes are visible to you. The fixes, in increasing sophistication:

```python
# Fix 1: After a write, route this user's subsequent reads to the LEADER
# for a short window. The bluntest, most reliable fix.

class ReadAfterWriteRouter:
    def __init__(self, leader, followers, window_s=1.0):
        self.leader = leader
        self.followers = followers          # connection pool
        self.last_write = {}                # user_id -> monotonic timestamp
        self.window_s = window_s

    def on_write(self, user_id):
        self.last_write[user_id] = time.monotonic()

    def read_conn(self, user_id):
        ts = self.last_write.get(user_id)
        if ts is not None and (time.monotonic() - ts) < self.window_s:
            return self.leader        # recently wrote: read your own writes
        return random.choice(self.followers)
```

This "read from leader for N seconds after a write" pattern is the workhorse. It works because the user's own writes are rare relative to reads, so only a small fraction of reads get routed to the leader. The window must exceed your worst-case lag — set it from your monitored p99 lag, not a guess.

A more precise version doesn't read from the leader at all; it reads from a follower but **waits for that follower to have caught up to the LSN of the user's write** (causal / bounded-staleness read):

```python
# Fix 2: LSN-based causal read. Remember the leader LSN at write time,
# then read from any follower that has replayed past it.

def write_and_record_lsn(leader_conn, user_id, new_bio):
    with leader_conn.cursor() as cur:
        cur.execute("UPDATE users SET bio = %s WHERE id = %s", (new_bio, user_id))
        cur.execute("SELECT pg_current_wal_lsn()")
        write_lsn = cur.fetchone()[0]
    leader_conn.commit()
    session_set(user_id, "min_lsn", write_lsn)   # stash in session/cookie
    return write_lsn

def causal_read(followers, user_id, sql, params):
    min_lsn = session_get(user_id, "min_lsn")
    for f in followers:
        with f.cursor() as cur:
            cur.execute("SELECT pg_last_wal_replay_lsn()")
            replayed = cur.fetchone()[0]
            if min_lsn is None or replayed >= min_lsn:
                cur.execute(sql, params)        # this follower is caught up enough
                return cur.fetchall()
    # No follower caught up yet -> fall back to leader.
    return read_from_leader(sql, params)
```

This is exactly how a serious read-scaling tier handles read-your-writes at scale: track the write LSN in the user's session, and only serve their reads from a replica that has demonstrably applied it. It preserves read scaling (most reads still hit followers) while closing the anomaly precisely.

### Anomaly 2: monotonic reads

Monotonic reads is the guarantee that *time does not go backward for a single user across a sequence of reads*. The bug: a user reads from follower A (caught up, sees a comment) then refreshes and reads from follower B (lagging, comment gone). The data appears to flicker in and out of existence. This is *weaker* than read-your-writes but it has a clean, cheap fix that Kleppmann states directly: **make each user always read from the same replica.** Different users can use different replicas (so you still scale), but one user is sticky to one replica so they never see it regress.

```python
# Fix: sticky replica routing. Hash the user to a stable replica so
# a single user never time-travels backward across reads.

def sticky_replica(followers, user_id):
    idx = hash(user_id) % len(followers)
    return followers[idx]
```

The catch: if that replica falls far behind or dies, the user is stuck on stale or no data, so you need health checks and a deterministic fallback (e.g. next replica in a hash ring). But for the common case it is a one-line fix that eliminates the flicker.

### Anomaly 3: consistent-prefix reads

Consistent-prefix reads is the guarantee that *if writes happened in a certain order, anyone reading them sees them in that order.* The classic violation, from Kleppmann: an observer sees Mrs. Cake's *answer* to a question before they see the *question* — causality is violated because the two writes (to possibly different partitions) replicated at different speeds. In a single-leader, single-partition system this anomaly doesn't arise, because the WAL is a single totally-ordered stream and a follower applies it in order. It becomes a problem the moment your data is **partitioned/sharded**, where each shard replicates independently and there is no global order across shards. The fix is to keep causally-related writes in the same partition (so they share one ordered log) or to use a system that tracks causal dependencies explicitly.

![Which read-routing strategy buys which consistency guarantee, and what it costs in read scaling](/imgs/blogs/database-replication-sync-async-logical-physical-8.webp)

The matrix sums up the tradeoffs. Reading any follower gives you full read scaling but none of the three guarantees. Reading the leader after a write fixes read-your-writes but partially serves the others and costs you scaling. Sticky-to-one-replica nails monotonic reads cheaply. LSN-based causal reads ("wait for LSN") satisfy all three guarantees while preserving most of your read scaling — at the cost of complexity. **There is no single setting that gives everything; you compose these routing rules to cover the specific anomalies your app actually exhibits.** Most teams start with "read from leader after write" (kills the most user-visible bug) and add LSN-based causal reads only on the hottest read paths once they outgrow it.

### Second-order optimization: cap the lag, don't just measure it

Measuring lag is table stakes; the senior move is to *bound* it. On the routing layer, eject any replica from the read pool whose lag exceeds a threshold — better to overload the leader briefly than to serve users data from a replica that's five minutes stale. Postgres exposes the apply position; wire it into your load balancer's health check:

```python
# Health check: a replica is "in service" only if its lag is bounded.
def replica_healthy(conn, max_lag_s=2.0):
    with conn.cursor() as cur:
        cur.execute("SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))")
        lag = cur.fetchone()[0]
    return lag is not None and lag <= max_lag_s
```

This converts a silent correctness problem (stale reads) into a visible capacity problem (leader load), which is a far better failure to have because it pages you instead of confusing your users.

## 7. Failover and promotion: where the real disasters live

**Senior rule of thumb: lag bugs annoy users; failover bugs lose data and make the news. Treat automatic failover as a loaded gun — useful, occasionally necessary, and capable of taking your foot off if you point it wrong.**

Failover is the act of promoting a follower to be the new leader when the old leader is gone. It sounds simple and it is anything but, because it forces you to answer the single hardest question in distributed systems under time pressure and incomplete information: **is the leader actually dead, or do I just can't reach it?** Those two are indistinguishable from the outside, and getting the answer wrong in either direction is catastrophic. Declare a live leader dead and promote a second one, and now you have **two leaders both accepting writes** — split-brain, the worst outcome in this entire field, because both copies diverge and reconciling them after the fact may be impossible.

### The anatomy of a safe failover

![Automatic failover with fencing: the cluster manager fences the dead leader before promoting a standby so two leaders never accept writes](/imgs/blogs/database-replication-sync-async-logical-physical-6.webp)

The timeline shows the steps a *correct* automatic failover takes, and the order is everything:

1. **Detect** (`t1`): the leader misses heartbeats. A single missed heartbeat means nothing — networks blip. You need a threshold of consecutive misses over a window to avoid flapping.
2. **Agree** (`t2`): a *consensus* of cluster members (not one observer) declares the leader dead. This is why serious failover tools are built on a consensus protocol — Raft (Orchestrator), or an external consensus store like etcd/Consul/ZooKeeper (Patroni). A single node's opinion is not enough, because that node might be the one that's partitioned.
3. **Fence** (`t3`): *before* promoting anyone, **fence the old leader** — make absolutely sure it cannot accept writes. This is **STONITH** ("Shoot The Other Node In The Head"): power it off, revoke its network access, demote it via a VIP/proxy that now refuses it, or trip a `touch /tmp/promote-blocked` style guard. Fencing is the step that prevents split-brain. **Skipping fencing is the single most common cause of split-brain.**
4. **Select** (`t4`): choose the standby with the **highest applied LSN** (most up-to-date) to minimize data loss. With GTIDs (MySQL) or LSN comparison (Postgres) this is mechanical.
5. **Promote** (`t5`): promote the chosen standby to leader (`pg_promote()` / `STOP REPLICA; RESET REPLICA ALL`).
6. **Repoint** (`t6`): redirect clients to the new leader — flip a Virtual IP, update the proxy (PgBouncer/HAProxy/ProxySQL), or update service discovery. Clients should connect to a *stable endpoint*, never a hard-coded leader IP, precisely so this step is one atomic flip.

The order **fence-then-promote** is non-negotiable. If you promote before fencing, there is a window where the old leader (which may just be slow, not dead) and the new leader both think they're in charge.

### The tools: Patroni, Orchestrator, repmgr

You do not write this logic yourself. The production-grade options:

- **Patroni** (Postgres) is the de facto standard. It stores cluster state and performs leader election through an external **Distributed Configuration Store** — etcd, Consul, or ZooKeeper — so the consensus is delegated to a battle-tested system. GitLab runs exactly this: PostgreSQL + Patroni + Consul, where Patroni handles failover and Consul holds the cluster state and notifies PgBouncer to repoint at the new leader. Patroni's use of an external quorum store means a partitioned node *loses* its leader lock and demotes itself, which is the fencing mechanism.
- **Orchestrator** (MySQL) discovers and manages MySQL topologies and automates failover; modern deployments run it on Raft for its own consensus so the orchestrator cluster itself can't split-brain. This is what GitHub used — and, as we'll see, the source of one of the most instructive outages in the field.
- **repmgr** (Postgres) is the older, simpler tool — witness-based rather than full consensus. GitLab's own docs note that migrating *from* repmgr *to* Patroni is straightforward while the reverse is painful, which tells you which direction the industry moved.
- **pg_auto_failover** (Postgres) is a Microsoft-originated alternative using a separate monitor node, simpler than Patroni for small fleets.

### The case for and against automatic failover

This is genuinely contested, and the honest answer is "it depends on your failure rate and your blast radius."

**Automatic failover** minimizes RTO — promotion happens in seconds without a human — which matters enormously if leader failures are frequent or downtime is extremely expensive. But it removes human judgment from a decision that is irreversibly destructive if wrong, and it is exactly an automatic system that, given an ambiguous signal (a network partition), can make the wrong call confidently and instantly.

**Manual (or semi-automatic) failover** keeps a human in the loop. The system detects and *recommends*, but a person confirms the promotion. RTO is worse (minutes, paging delay), but a human can distinguish "the leader is on fire" from "there's a weird network thing happening, let's wait 60 seconds." Many large shops landed on a hybrid: automatic failover **within** a region/AZ (where latency is low and partitions are rare and short), manual confirmation for anything **across** regions (where a partition is plausible and a cross-region promotion is destructive). This precise boundary is the lesson GitHub paid for.

| | Automatic | Manual | Hybrid (auto in-AZ, manual cross-region) |
| --- | --- | --- | --- |
| RTO | Seconds | Minutes (paging) | Seconds locally, minutes for big moves |
| Split-brain risk | Higher (acts on ambiguity) | Lower (human sanity check) | Bounded to safe scope |
| Best for | Frequent failures, low blast radius | Rare failures, high blast radius | Most large multi-AZ deployments |

## 8. The GitHub 2018 outage: a failover masterclass

I want to walk through GitHub's October 21, 2018 incident in detail, because it is the most instructive failover postmortem in public and it touches every concept in this article. (The full post-incident analysis is on the GitHub blog; what follows is my reading of it.)

The trigger was trivial: routine maintenance to replace failing 100 Gb optical equipment severed connectivity between GitHub's US East Coast network hub and its primary US East Coast data center for **43 seconds**. Forty-three seconds. The recovery took **over 24 hours**.

Here is the chain. GitHub ran MySQL with **Orchestrator** for automated topology management and failover, built on Raft for consensus. During the 43-second partition, the Orchestrator nodes in the US West Coast data center and the US East Coast public cloud were able to **establish a Raft quorum** (they could see each other; they couldn't see the partitioned East Coast primary) and, doing exactly what they were designed to do, **failed clusters over to direct writes to the US West Coast**.

Two things then went wrong, and they're the two failure modes we've been building toward.

**Split-brain / divergent writes.** During those 43 seconds, the East Coast primary had *accepted writes that hadn't replicated west*. When the West Coast was promoted, it became the source of truth — but it didn't have those East Coast writes. Then, after connectivity restored, the West Coast accumulated ~30–40 minutes more writes before engineers fully intervened. Now **both data centers contained writes the other didn't have.** One busy cluster had 954 affected writes. There was no safe automatic failback, because failing back to the East Coast would lose the West Coast writes, and staying West lost the East Coast writes. This is the split-brain reconciliation nightmare: the data physically diverged and a human had to merge it.

**Cross-region promotion was the original sin.** GitHub's applications were co-located in the East Coast facility. When Orchestrator promoted the West Coast database, every query from the East Coast app now had to cross the entire country — adding tens of milliseconds per query to applications doing many sequential queries — making the platform effectively unusable even though a database was "up." As the postmortem put it, leader election *within* a region is generally safe, but the sudden introduction of cross-country latency was a major contributing factor. The automation optimized for "a leader exists" without modeling "a leader the application can actually use at acceptable latency."

**Then lag turned a database problem into a day-long problem.** After restoring primaries to the East Coast, dozens of read replicas were still *hours* behind, and (as noted earlier) the catch-up followed a power-decay curve, not a line, so the ETAs were repeatedly wrong as European and US morning traffic piled on.

The remediation reads like a checklist for this whole article. GitHub configured Orchestrator to **prevent promotion of database primaries across regional boundaries** — exactly the "automatic in-region, never automatic cross-region" hybrid boundary. They accelerated a move to active/active multi-region. And they adopted chaos engineering to *practice* these failures before production forced the rehearsal.

The lessons, distilled:

1. **A short network partition can trigger a long outage** when automation reacts to it. The 43-second trigger isn't the story; the automated cross-region failover and the days of reconciliation are.
2. **Fence and scope your automation.** Cross-region automatic promotion was the foot-gun. Don't let automation make a destructive choice in a scope where a partition is plausible.
3. **Failover that an application can't use is not a successful failover.** Model latency and locality, not just liveness.
4. **Async replication means failover can lose data.** Those East Coast writes were async; promotion west lost them. If RPO=0 is required, you pay for it with synchronous replication and accept its availability cost — there is no free lunch.

## 9. Topology: cascading and delayed replicas

**Senior rule of thumb: the leader should fan WAL out to as few direct followers as possible — push the fan-out down to relay replicas, and always keep one replica deliberately behind.**

![Replication topology: cascading relays offload WAL fan-out from the leader and a delayed replica is a time machine for human error](/imgs/blogs/database-replication-sync-async-logical-physical-7.webp)

The tree shows two refinements that separate a toy setup from a production one.

**Cascading replication.** Every direct standby costs the leader a `walsender` process and outbound bandwidth — the same WAL shipped N times for N standbys. With a large read-scaling fleet, that fan-out becomes a real load on the leader and competes with its actual job. The fix is **cascading**: the leader streams to a small number of *relay* standbys, and those relays re-stream to the leaf replicas. In Postgres a standby can itself have downstream standbys (`hot_standby = on` plus the leaf pointing `primary_conninfo` at the relay). Now the leader fans out to 2–3 relays; each relay fans out to many leaves; the leader's replication load is bounded regardless of fleet size. The cost is one extra hop of lag for the leaves.

```ini
# On the relay standby's postgresql.auto.conf:
primary_conninfo = 'host=leader.db.internal user=replicator ...'
# It still serves WAL onward, so on the LEAF replica:
primary_conninfo = 'host=relay-1.db.internal user=replicator ...'
```

**Delayed replica.** This one is a sleeper and a lifesaver. A **delayed replica** is configured to apply WAL on a deliberate time delay — say, one hour behind. Why would you want stale data on purpose? Because it is a *time machine for human error*. The number-one cause of catastrophic data loss is not hardware; it's a human running `DELETE FROM orders` without a `WHERE`, or a bad migration, or an `UPDATE` that touches every row. A normal replica faithfully and instantly replicates that disaster. A delayed replica gives you a one-hour window: notice the mistake, stop the delayed replica's apply *before* it replays the bad transaction, and recover the pre-disaster state from it — far faster than restoring a multi-terabyte backup.

```ini
# Postgres standby: apply WAL one hour behind the leader.
recovery_min_apply_delay = '1h'
```

```sql
-- MySQL replica: same idea, delay applying by 3600 seconds.
CHANGE REPLICATION SOURCE TO SOURCE_DELAY = 3600;
```

The delayed replica is the cheapest insurance in databases. It costs one node and you hope to never need it, but the day someone fat-fingers a destructive query against production, it turns a multi-hour restore-from-backup into a minutes-long recovery. Pair it with the failover playbook below.

## 10. The HA / failover playbook

Pulling it together into something you could hand a new SRE. This is the practical operational layer on top of all the theory.

**Architecture baseline.** Three or more nodes minimum: a leader, at least one synchronous candidate (for RPO=0 on the data that needs it), and async read replicas. Use a quorum sync setting (`ANY n` in Postgres, semi-sync with a sane timeout in MySQL) so a single replica failure never blocks writes. Put a connection proxy layer (PgBouncer/Consul/HAProxy/ProxySQL or a Patroni-managed VIP) between clients and the database so repointing is one atomic flip and clients never hard-code a leader IP. Add a cascading relay tier once your read fleet grows, and always run one delayed replica.

**Detection thresholds.** Don't fail over on a single missed heartbeat. Require N consecutive misses over a window (e.g. 3 misses in 10s). Derive the window from your real network jitter, not a default.

**Fence before you promote. Always.** The promotion path must include an explicit step that makes the old leader incapable of accepting writes before the new leader takes over. With Patroni this is the DCS lock (a partitioned old leader loses the lock and self-demotes); make sure that mechanism is actually wired up and tested, not assumed.

**Scope automatic failover to where partitions are rare.** Automatic within an AZ; manual confirmation across AZs or regions. This is the single highest-value lesson from GitHub — encode the boundary in the tool's config, don't rely on it never happening.

**Pick the most-caught-up standby.** Promotion should select by highest applied LSN / GTID set to minimize data loss. Verify your tool does this; some naive scripts promote a fixed standby regardless of lag.

**Rehearse.** Run game days. Kill the leader in staging (and, when you're brave and well-instrumented, in production during low traffic). Time the RTO. Verify clients reconnect. Verify the delayed replica can be used for recovery. An untested failover is a failover that doesn't work — you find out which during the real incident otherwise.

**Measure the right things.** Alert on byte lag *and* time lag per replica. Alert on slot retention size (an abandoned slot filling the disk). Alert on `sync_state` flapping (a synchronous standby that keeps falling out of quorum). Alert on a replica that has fallen out of the read pool's health check.

```bash
# A minimal failover smoke test you can run in a game day (Patroni):
patronictl -c /etc/patroni.yml list                # see roles & lag
patronictl -c /etc/patroni.yml switchover --master pg-1 --candidate pg-2  # planned
# then verify the app reconnected and writes succeed on pg-2,
# and that pg-1 came back as a healthy replica (not a second leader).
```

## Case studies from production

### 1. GitHub's 43-second partition, 24-hour outage (2018)

Covered in depth in §8: a 43-second cross-region network partition let Orchestrator's Raft quorum (West + cloud, minus the partitioned East primary) promote the West Coast to leader. During the partition the East primary had taken async writes that never replicated west; after restoration both regions held divergent writes (one cluster: 954 affected), making automatic failback impossible. Cross-region promotion also broke application latency because apps were East-co-located. **Root cause:** automatic *cross-region* failover acting on an ambiguous partition signal, plus async replication's nonzero RPO. **Fix:** Orchestrator configured to never promote across regional boundaries; move toward active/active; chaos engineering. **Lesson:** scope automatic promotion to where partitions are rare and short; a leader the app can't reach at acceptable latency isn't a recovery.

### 2. GitLab's Patroni + Consul HA architecture

GitLab's reference HA Postgres architecture is a clean instance of the playbook: a minimum of three PostgreSQL nodes, three Consul server nodes, three PgBouncer nodes, and an internal TCP load balancer. **Patroni** handles failover — electing a new leader, promoting it, instructing the rest to follow — while a **Consul** agent on each node stores the Patroni cluster state and monitors node health. When the leader fails, Patroni and Consul coordinate: Consul runs a script that rewrites PgBouncer's config to point at the new leader and reloads it, so clients are repointed without touching application config. GitLab's docs also note that moving *from* repmgr *to* Patroni is straightforward but the reverse is painful — a strong signal about which tool the ecosystem standardized on. **Lesson:** delegate consensus and fencing to an external, battle-tested store (Consul/etcd); put a pooler in front so repointing is automatic.

### 3. Notion's audit-log "replication" for sharding

When Notion sharded their Postgres monolith, they needed to copy data from one big database into many shards with zero downtime. They first tried **logical replication** and found it couldn't keep up with their block-table write volume during the initial snapshot step. So they built their own: an **audit-log table** capturing every write to the migrating tables, plus a catch-up process replaying the audit log into the new shards. To validate the cutover they ran **dark reads** — every read executed against both old and new databases and compared results, logging discrepancies. They saw near-100% equivalence (the gap attributable to nondeterministic queries and replication lag) before flipping over. To preserve RDS replication guarantees they capped tables at 500 GB and physical databases at 10 TB. **Lesson:** built-in logical replication has throughput limits at extreme scale; a custom CDC-via-audit-log plus dark-read verification is a battle-tested pattern for risky cutovers.

### 4. The single synchronous standby that blocked all writes

A team enabled synchronous replication for durability by setting Postgres `synchronous_standby_names = 'standby_1'` — a *single* named standby — with `synchronous_commit = on`. It worked beautifully and gave RPO=0. Then `standby_1` was rebooted for a routine kernel patch. Every commit on the leader **hung**, because the leader was waiting for an ack from a standby that was gone, and the application's write path froze across the fleet. The on-call's first hypothesis was a leader problem; the leader was perfectly healthy and idle, *waiting*. **Root cause:** a single synchronous standby is a single point of *write* failure. **Fix:** switch to quorum, `ANY 1 (standby_1, standby_2)`, so either standby can satisfy the sync requirement and one can be down for maintenance. **Lesson:** synchronous replication trades a *replica's* availability for the cluster's durability; never let one replica gate every write.

### 5. The abandoned replication slot that filled the disk

A standby was decommissioned by terminating its VM, but its **replication slot** on the leader was never dropped. The slot dutifully kept its promise: it pinned WAL at the dead standby's last confirmed LSN and refused to let the leader recycle it. Over the next several days the leader's WAL directory grew without bound until `pg_wal` filled the disk and the leader **crashed** — taking down the write path for a problem that had nothing to do with writes. **Root cause:** an orphaned slot is a WAL retention leak. **Fix:** drop the slot (`SELECT pg_drop_replication_slot('standby_old')`), and set `max_slot_wal_keep_size` so a future orphan invalidates the slot instead of filling the disk. **Lesson:** every slot is a standing promise to retain WAL; decommissioning a replica must include dropping its slot, and you should cap retention defensively.

### 6. Statement-based binlog and the nondeterministic UPDATE

A MySQL deployment ran with the legacy `binlog_format = STATEMENT` for its smaller binlog footprint. An application query did `UPDATE sessions SET token = UUID() WHERE expired = 1`. On the source, each row got a fresh UUID; on the replica, the statement re-executed and generated *different* UUIDs. The data silently diverged — replica session tokens didn't match the source's — and the divergence only surfaced weeks later when a read replica served a token that didn't exist on the primary. **Root cause:** statement-based replication re-executes non-deterministic functions, which the MySQL manual explicitly warns against. **Fix:** `binlog_format = ROW` (the modern default), which logs the concrete row images so the replica applies exact values rather than re-running the function. **Lesson:** `ROW` is the safe default; the binlog-size savings of `STATEMENT` are not worth silent data divergence.

### 7. The analytics query that starved the HA standby

A team ran HA with one synchronous standby and, to save a node, also pointed their nightly BI export at it. The export ran a multi-hour full scan that flooded the standby's buffer cache and held a long-running transaction, which (combined with `hot_standby_feedback` and apply conflicts) slowed WAL apply on the standby until it fell minutes behind. Because that standby was the synchronous candidate, commit latency on the *leader* spiked every night during the export, and the team's RPO guarantee silently degraded — the "synchronous" standby was so far behind that a leader failure during the window would have lost more than expected. **Root cause:** one replica doing two incompatible jobs (HA and heavy analytics). **Fix:** a dedicated analytics replica (cascading off a relay), isolating the sync HA candidate. **Lesson:** don't make one replica serve HA and analytics; their lag budgets are opposite.

### 8. The delayed replica that saved a fat-fingered DELETE

An engineer ran a data-cleanup script against production that was supposed to delete a few thousand stale rows but, due to a bug in the `WHERE` clause, issued an effectively unbounded `DELETE` that removed millions of rows from `orders`. The normal replicas faithfully replicated the deletion within milliseconds. But the team ran a **delayed replica** at `recovery_min_apply_delay = '1h'`. Within minutes of noticing, they stopped the delayed replica's apply *before* it reached the bad transaction's LSN, then used it to recover the deleted rows — restoring the pre-disaster state in under an hour, versus an estimated 6+ hours to restore the multi-terabyte database from a base backup plus WAL. **Root cause:** human error, the most common cause of data loss. **Fix:** the delayed replica was the fix, prepared in advance. **Lesson:** a delayed replica is the cheapest catastrophe insurance you can buy; run one, and rehearse recovering from it.

### 9. The logical replication that broke on a missing column

A team used Postgres logical replication to feed a reporting database. A developer shipped a migration that added a column to a published table — on the *publisher only*, as part of a normal deploy. The first `INSERT` carrying the new column arrived at the subscriber, which didn't have the column, and the apply worker errored and **halted all replication** for that subscription. Reporting silently froze at the last applied LSN; nobody noticed for a day because the dashboards still rendered (stale) data. **Root cause:** logical replication does not replicate DDL, so schema changes must be applied to the subscriber first, in a coordinated order. **Fix:** add the column to the subscriber *before* the publisher, treat publisher/subscriber schema changes as a coordinated [zero-downtime migration](/blog/software-development/database/zero-downtime-schema-migrations), and alert on `pg_stat_subscription` apply lag. **Lesson:** logical replication makes schema changes a two-sided dance; automate the ordering or it will break in production.

### 10. The geo-replica that lagged during the trans-Atlantic blip

A company served EU users from a read replica in Frankfurt streaming async from a leader in Virginia. Normal lag was ~80 ms. During a brief trans-Atlantic network degradation, the cross-region link's throughput dropped below the leader's WAL generation rate, and the Frankfurt replica's lag climbed to several *minutes* — EU users started seeing data minutes stale, including their own recent writes (read-your-writes broken). The replica's health check only checked liveness, not lag, so it stayed in the read pool serving increasingly stale data. **Root cause:** unbounded cross-region lag plus a liveness-only health check. **Fix:** add a *lag-bounded* health check (§6) that ejects a replica from the read pool past a threshold, and route recent-writers to the leader during the window. **Lesson:** cross-region replication's lag is bursty and tied to the WAN; health-check on lag, not just liveness, and fall back to the leader when a region falls behind.

### 11. Major-version upgrade via logical replication, zero downtime

A team needed to move from Postgres 12 to 16 on a database that could not take meaningful downtime. Physical replication was no help — it requires identical major versions. They used **logical replication** instead: stood up a PG16 instance, created a subscription to the PG12 publisher, let the initial copy and streaming catch-up complete (monitoring `pg_stat_subscription`), advanced sequences on the new node, then flipped the application's connection to PG16 during a brief, planned maintenance window measured in seconds. **Root cause / motivation:** physical replication's same-version constraint blocks online major upgrades. **Fix:** logical replication's cross-version capability. The gotchas they hit: sequences needed manual advancement, large tables needed `REPLICA IDENTITY` set, and they had to freeze DDL during the cutover. **Lesson:** logical replication is the standard tool for near-zero-downtime major upgrades — exactly the cross-version capability physical replication lacks.

### 12. Flapping failover from an aggressive heartbeat threshold

A new HA deployment used a 1-second heartbeat with failover after a *single* miss. The cluster ran in a cloud with occasional sub-second network jitter, and roughly weekly, a transient blip would cause one missed heartbeat, trigger a failover, promote a standby — and then the original leader, perfectly healthy, would come back and fight for leadership. The result was periodic, gratuitous failovers (each a brief write interruption and a connection storm) for a leader that was never actually down. **Root cause:** detection threshold too aggressive for the environment's jitter. **Fix:** require multiple consecutive missed heartbeats over a window (e.g. 3 misses / 10s) and add fencing so a returning old leader can't fight. **Lesson:** tune detection to your real network's jitter; failover should react to *outages*, not *blips*, and the GitHub lesson applies in miniature — a short network event should not trigger a destructive automated action.

## When to reach for each tool, and when not to

### Reach for physical/streaming replication when…

- You want HA standbys or read-scaling replicas with the lowest lag and least operational fuss — it's the right default.
- You need an exact byte-for-byte clone and the publisher/subscriber run the same major version.
- You want simple, robust failover candidates (a promoted physical standby just becomes the leader).
- You're offloading backups or `pg_dump` to a node and don't need selectivity.

### Reach for logical replication when…

- You need **selective** replication — specific tables, filtered rows, specific columns.
- You're doing a **near-zero-downtime major version upgrade** (physical can't cross major versions).
- You're feeding **CDC** pipelines, search indexes, data warehouses, or a different engine.
- You need the target to be **writable** or structurally different (extra indexes, different extensions).

### Reach for synchronous replication when…

- A specific, bounded set of data cannot tolerate losing even one acknowledged transaction: payments, account balances, inventory reservations, order state, audit trails.
- And you can afford the extra commit latency (one RTT) and have a **quorum** of candidates so a single replica failure doesn't block writes.

### Skip synchronous replication when…

- The workload is high-volume and loss-tolerant — clickstream, metrics, event logs — where availability and throughput beat perfect cross-node durability. `synchronous_commit = local` or even `off` is the right call.
- You only have one possible synchronous standby (a single sync standby is a single point of write failure — fix the topology first).
- The latency budget can't absorb a network round trip on every commit.

### Reach for automatic failover when…

- Leader failures are frequent enough, or downtime expensive enough, that seconds of RTO justify removing the human — **and** the promotion scope is one where partitions are rare and short (within an AZ), **and** fencing is wired up and tested.

### Skip (or gate) automatic failover when…

- The promotion would cross a boundary where a network partition is plausible (cross-region) — make those manual or strictly scoped, per GitHub's hard-won lesson.
- You can't guarantee fencing — without it, automatic failover manufactures split-brain.
- Failures are rare and the blast radius of a wrong promotion is enormous — a human sanity check is cheap insurance.

---

Everything in this article lives inside one assumption: **a single leader.** That assumption is what made conflicts impossible and let us focus entirely on *how the change stream is shipped* (physical vs logical), *when commits are acknowledged* (sync vs async), and *what happens when the one writer dies* (failover). It is the right model for the overwhelming majority of systems, and getting it right — quorum sync for the data that needs it, async with proper read-routing for everything else, fenced and scoped failover, a delayed replica, and lag-aware health checks — will carry you a very long way.

But the single-leader model has a hard ceiling: write throughput is bounded by one node, and write availability is bounded by that node's liveness. The moment you need to accept writes in multiple regions simultaneously, or survive the leader's failure with *zero* write downtime, you have to give up the single writer — and the instant you do, write conflicts become possible and you inherit a whole new universe of problems: conflict detection, last-write-wins vs CRDTs, quorum reads and writes, read repair, and the consistency models that govern them. That is **multi-leader and leaderless replication**, and it is a separate, larger story that picks up exactly where this one ends — in the distributed-systems track, where we'll trade the comfort of one writer for the resilience of many. For now: master the single leader. It's where the real systems are, and where the real incidents live.

### Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 5 (Replication) — the canonical treatment of single-leader, sync vs async, lag anomalies, and failover.
- [PostgreSQL: Log-Shipping Standby Servers](https://www.postgresql.org/docs/current/warm-standby.html) and [Replication configuration](https://www.postgresql.org/docs/current/runtime-config-replication.html) — streaming, slots, `synchronous_standby_names`, quorum syntax.
- [PostgreSQL: Logical Replication](https://www.postgresql.org/docs/current/logical-replication.html) and [Asynchronous Commit](https://www.postgresql.org/docs/current/wal-async-commit.html).
- [MySQL: Advantages and Disadvantages of Statement-Based and Row-Based Replication](https://dev.mysql.com/doc/refman/8.4/en/replication-sbr-rbr.html) and [Replication Formats](https://dev.mysql.com/doc/refman/8.0/en/replication-formats.html).
- [GitHub October 21 post-incident analysis](https://github.blog/2018-10-30-oct21-post-incident-analysis/) — the failover masterclass dissected in §8.
- [GitLab PostgreSQL replication and failover (Patroni + Consul)](https://docs.gitlab.com/administration/postgresql/replication_and_failover/).
- [Patroni](https://github.com/patroni/patroni) — Postgres HA with etcd/Consul/ZooKeeper.
- Notion engineering: [Sharding Postgres at Notion](https://www.notion.com/blog/sharding-postgres-at-notion) and [The Great Re-shard](https://www.notion.com/blog/the-great-re-shard).
- Sibling posts: [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), [Isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), [Zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations).

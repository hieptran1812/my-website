---
title: "Multi-Datacenter and Geo-Replication: MirrorMaker, Cluster Linking, and Active-Active"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Run a message system across regions without lying to yourself about the data you will lose — the offset-translation trap, the RPO window baked into every async link, and how active-active quietly creates duplicates and loops."
tags:
  [
    "message-queue",
    "geo-replication",
    "multi-datacenter",
    "kafka",
    "mirrormaker",
    "cluster-linking",
    "rabbitmq",
    "disaster-recovery",
    "distributed-systems",
    "active-active",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/multi-datacenter-geo-replication-1.webp"
---

Someday an entire region of your cloud provider is going to go dark. Not a broker, not a rack — the whole region: the control plane stops answering, the load balancers return nothing, your dashboards go gray because the dashboards were *in that region too*. When that day comes, the only question that matters is whether the messages flowing through your system also live somewhere else, and how many of the most recent ones you just lost. If your answer is "all of our Kafka is in us-east-1," then the answer to the second question is "all of them, indefinitely," and you are going to have a very long night.

This post is about the engineering you do *before* that night so it becomes a forty-five-minute failover instead of a resume-updating event. Running a message system across datacenters sounds like it should be a checkbox — turn on replication, point a second cluster at the first, done. It is not. Cross-region replication is fundamentally **asynchronous**, which means there is always a gap between what your source region has acknowledged and what your destination region actually holds. That gap, measured in milliseconds of lag, is the exact set of messages you lose on a hard failover. It has a name — your **Recovery Point Objective**, your RPO — and every architectural choice in this post is really a negotiation over how big that gap is and what nasty surprises come bundled with shrinking it. Figure 1 lays out the four topologies we will work through and what each one actually costs you.

![A decision matrix comparing active-passive, active-active, hub-and-spoke, and fan-out topologies across their use case, write model, and operational complexity](/imgs/blogs/multi-datacenter-geo-replication-1.webp)

By the end of this post you will be able to do four things most engineers who *deploy* multi-region messaging cannot. You will be able to explain why a consumer that was reading at offset 5,000,000 in one region cannot just resume at offset 5,000,000 in another — the **offset-translation problem** — and what MirrorMaker 2 does to paper over it. You will be able to compute your real RPO from your replication lag and your message rate, in actual messages-at-risk, and defend that number to a VP. You will know why active-active is the architecture everyone wants and almost nobody should build, and exactly which loop-prevention and conflict-handling machinery you must have before you turn it on. And you will be able to pick a topology — active-passive, active-active, hub-and-spoke, or fan-out — on purpose, for stated reasons, instead of cargo-culting whatever your last employer ran. This builds directly on [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability), which covers replication *inside* one cluster; here we cross the WAN, where every guarantee that post earned gets weaker.

## Why replicate across datacenters

Let me kill one assumption up front: you do not replicate across datacenters because it is good engineering hygiene. You do it because there are exactly four business pressures that demand it, and if none of them apply to you, multi-region messaging is an expensive, complexity-multiplying mistake that you should not make. Know *which* of the four you are solving for, because the topology you should choose is a direct function of that answer, not of your throughput or your broker brand.

The first pressure is **disaster recovery**. A single region is a single failure domain, no matter how many availability zones you spread across inside it. Cloud providers lose entire regions — control-plane outages, fiber cuts, a bad config push that cascades, a power event at the regional level. When that happens, an in-region replication factor of three buys you nothing, because all three replicas were in the region that died. DR means keeping a copy of your data, and ideally a warm cluster ready to serve it, in a *different* region, so a regional loss is survivable. This is the most common and most defensible reason to go multi-region, and it maps to the active-passive topology.

The second pressure is **data locality and latency**. If your users are in Frankfurt and your only Kafka cluster is in Virginia, every produce and every consume pays an 80-to-100-millisecond round trip across the Atlantic. For a fire-and-forget event stream that may be fine; for anything in a user-facing request path it is a disaster. Putting a cluster near each user population gives you single-digit-millisecond local latency, and then replication stitches the regional clusters together so each region still sees a global view of events. This pressure pushes you toward active-active or fan-out, the topologies where multiple regions are live.

The third pressure is **regulatory data residency**. The GDPR, China's PIPL, India's data-localization rules, financial-services regulations in dozens of jurisdictions — a growing list of laws say that certain data about certain citizens must be stored and processed within specific borders. If you have EU customer data that legally cannot leave the EU, you physically cannot have it land first in a US cluster and replicate out. Residency inverts the usual flow: instead of "replicate everything everywhere," you partition data by region of origin and replicate only what is legally allowed to cross. This is a topology *constraint* more than a topology choice, and it usually forces a hub-and-spoke or carefully filtered active-active design.

The fourth pressure is **global aggregation**. You have twelve regional clusters, each capturing local clickstream or telemetry or transactions, and a central analytics team that needs a single unified stream of *all* of it to run fraud models, fill a data warehouse, or compute global metrics. You do not want twelve consumers each reaching across the WAN into a different region; you want one place that holds everything. That is the hub-and-spoke topology — many spokes feeding one aggregation hub — and it is the cleanest, lowest-conflict multi-region pattern there is, because data flows in exactly one direction.

These four pressures are not mutually exclusive, and the hardest multi-region designs are the ones that have to satisfy two or three at once. A global payments platform might need DR *and* low-latency local writes *and* residency all at the same time, which is exactly why those platforms end up with the most elaborate topologies — a regionally-sharded active-active for the write path, plus a hub-and-spoke for the analytics path, plus per-region backups for DR. But you do not earn that complexity by default; you earn it one pressure at a time, and the discipline is to add a topology only when a specific pressure forces it. The most expensive multi-region mistakes I have seen all came from teams that built for a pressure they did not actually have — global active-active to serve users who were all in one country, or zero-RPO synchronous replication for an event stream where losing a few seconds of data would have cost nothing. Match the architecture to the real, named pressure, and no more.

### The honest caveat that colors everything

Here is the truth that every vendor slide deck buries: **cross-region replication is asynchronous, period.** It is not a CAP-theorem footnote you can configure away; it is physics plus economics. A round trip between Virginia and Frankfurt is roughly 80–90 milliseconds at the speed of light through fiber, before you add the broker's own latency. If you tried to make your producer wait for a synchronous acknowledgment from the remote region on every write — to drive your RPO to zero — you would add 80-plus milliseconds to every single produce, collapsing your throughput and coupling your local availability to a transcontinental link that drops packets. Nobody runs that. So instead, the local cluster acknowledges fast, and a background replicator ships the data across the WAN whenever it can. The consequence is unavoidable: at any instant, the remote region is *behind* the source region by some amount of lag, and the messages inside that lag window have been acknowledged to your producers but do not yet exist anywhere outside the source region. If the source region dies in that instant, those messages are gone. That gap is your RPO, and the entire rest of this post is about understanding it, measuring it, and deciding what you are willing to pay to shrink it. For a deeper grounding in why asynchronous replication forces eventual consistency, the broader treatment in [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) is worth a read; here we live with the eventual side and price it.

## The replication tools

Three tools dominate cross-region messaging in practice, and they are not interchangeable. Two are Kafka-world tools that differ in a subtle but operationally enormous way — how they handle offsets — and one is the RabbitMQ approach, which has a completely different model because RabbitMQ has no durable log to replicate in the first place. Figure 9 summarizes the comparison; let me walk through each tool's actual mechanics so you understand *why* the matrix cells read the way they do.

![A matrix comparing MirrorMaker 2, Confluent Cluster Linking, and RabbitMQ Federation across synchronization model, offset handling, broker, and failover seek behavior](/imgs/blogs/multi-datacenter-geo-replication-9.webp)

### MirrorMaker 2: the Kafka Connect-based replicator

MirrorMaker 2 (MM2) is the open-source, ships-with-Kafka answer. It is built on Kafka Connect and runs as a set of connectors: `MirrorSourceConnector` copies topic records from a source cluster to a target cluster, `MirrorCheckpointConnector` copies and *translates* consumer-group offsets, and `MirrorHeartbeatConnector` emits heartbeats so you can measure end-to-end replication lag. The fundamental thing to internalize about MM2 is that it is a **consumer plus a producer**: it consumes from the source cluster and re-produces into the target cluster. That means the records get *new* offsets in the target — and that single fact is the root of the offset-translation problem we will spend a whole section on.

MM2 names replicated topics with a prefix by default, the source cluster's alias. A topic called `orders` in a cluster aliased `us-east` becomes `us-east.orders` in the target. This `DefaultReplicationPolicy` prefixing is what makes loop prevention work in active-active — more on that later — but it also means consumers and tooling have to know about the renaming. You can configure an `IdentityReplicationPolicy` to keep names identical, which is cleaner for one-way DR but removes the built-in loop protection.

```properties
# mm2.properties — a one-way replication from us-east to eu-west
clusters = us-east, eu-west
us-east.bootstrap.servers = kafka-use-1:9092,kafka-use-2:9092
eu-west.bootstrap.servers = kafka-euw-1:9092,kafka-euw-2:9092

# Replicate us-east -> eu-west only (DR direction)
us-east->eu-west.enabled = true
us-east->eu-west.topics = orders, payments, inventory

# Translate consumer offsets so a failed-over consumer can resume
us-east->eu-west.emit.checkpoints.enabled = true
us-east->eu-west.sync.group.offsets.enabled = true

# How many MM2 tasks (parallelism) and how aggressively to mirror
tasks.max = 8
replication.factor = 3
refresh.topics.interval.seconds = 30
```

The honest tradeoff with MM2: it is free, it is open, it runs anywhere, and it is *fiddly*. You are operating a Kafka Connect cluster, tuning `tasks.max` against partition counts, watching for rebalance storms, and living with the fact that offsets are translated approximately rather than preserved exactly. For many teams that is a perfectly good deal. For teams that need clean, automatic consumer failover, the offset translation is a sharp edge.

A few MM2 operational realities worth knowing before you commit to it. First, MM2 *is* a consumer group on the source cluster and a producer to the target, so it has all the operational characteristics of both — it can lag, it can rebalance, and it competes for the same broker resources as your application traffic. On a busy source cluster, a fleet of MM2 tasks fetching everything can add meaningful read load, so you size the Connect cluster and the `tasks.max` against your real partition count and throughput, not by guessing. Second, MM2 does not preserve the source's partition-to-record assignment unless you keep the partition count identical and use the default partitioner; if the target topic has a different partition count, records can land in different partitions, which quietly breaks any per-key ordering you depended on. Keep partition counts aligned across source and target. Third, MM2 replicates topic configurations and ACLs optionally — and if you forget to enable that, your target topics come up with default retention and no ACLs, which is its own incident waiting to happen during a failover. The rule of thumb: replicate the topics, the configs, the ACLs, *and* the offsets, or you have only replicated part of your system.

### Confluent Cluster Linking: byte-for-byte, offset-preserving

Cluster Linking is Confluent's answer, and it solves the problem MM2 papers over by attacking it at a lower level. Instead of running a consumer-plus-producer that re-writes records and assigns new offsets, a cluster link makes the *target* brokers fetch directly from the source brokers using the same replication protocol Kafka uses internally between leader and follower. The target topic is a true mirror: the records arrive with the **exact same offsets** they had in the source. Offset 5,000,000 in the source is offset 5,000,000 in the target. No translation, no approximation, no checkpoint connector.

This changes the failover story completely. With Cluster Linking, a consumer that was at offset 5,000,000 in the source can fail over and resume at offset 5,000,000 in the target *with no seek and no translation math* — because it is literally the same offset pointing at the same record. The mirror topics are read-only on the target until you promote them, which prevents accidental dual writes. The cost is that it is a commercial Confluent feature, not open-source Kafka, so you are paying Confluent and you are inside their ecosystem. The architectural simplification is real, though: offset-preserving replication removes an entire class of failover bugs. The matrix in Figure 9 puts this difference front and center, because it is the single most important axis on which these two Kafka tools diverge.

There is a deeper reason Cluster Linking can preserve offsets where MM2 cannot, and it is worth understanding because it explains the tradeoff. MM2 sits *outside* the brokers — it is an external Connect application that reads with a normal consumer and writes with a normal producer, and a normal producer has no way to say "write this record at exactly offset 5,000,000." Offsets are assigned by the partition leader as records are appended; no external client controls them. Cluster Linking, by contrast, runs *inside* the broker and uses the inter-broker replication protocol, the same one a follower uses to copy from a leader within a cluster. A follower fetches records *with their offsets* and writes them at those exact offsets, because byte-for-byte log replication is what Kafka's internal replication already does. Cluster Linking essentially makes a broker in region B a follower of a partition leader in region A. That is why it preserves offsets: it is not re-producing records, it is *replicating the log* the way Kafka replicates internally — just across a WAN instead of a LAN. The price you pay is tighter coupling to Confluent's broker and the commercial licence, but the payoff is that the offset-translation problem simply does not exist.

### RabbitMQ Federation and the Shovel plugin

RabbitMQ has no append-only log, so it cannot do offset-based replication at all — there are no offsets. Its two cross-cluster tools work by *moving messages* between brokers, which is a different model with different guarantees.

**Federation** links exchanges or queues across brokers. A *federated exchange* makes messages published to an upstream exchange flow downstream to a federated exchange on another broker, as if the downstream broker had a subscriber on the upstream. A *federated queue* lets a downstream queue pull messages from an upstream queue on demand, balancing load across brokers. Federation is loosely coupled, tolerant of links going up and down, and speaks plain AMQP internally, so it works fine over a WAN with intermittent connectivity. It is the natural choice for connecting RabbitMQ clusters across regions.

The **Shovel** plugin is the lower-level, more explicit tool: it is essentially a built-in, reliable consumer-plus-publisher that drains messages from a queue on a source broker and republishes them to an exchange on a destination broker. Where Federation is declarative ("these exchanges are linked"), Shovel is a concrete moving job ("take from queue X here, publish to exchange Y there"). You reach for Shovel when you need precise control over exactly which messages move where, or when you are draining a queue from one cluster into another during a migration.

```bash
# Define a dynamic shovel: drain orders from a DR-source broker
# into the local exchange on the DR-target broker.
rabbitmqctl set_parameter shovel orders-dr-shovel \
'{
  "src-protocol": "amqp091",
  "src-uri": "amqp://user:pass@primary.eu-west.internal:5672",
  "src-queue": "orders",
  "dest-protocol": "amqp091",
  "dest-uri": "amqp://user:pass@localhost:5672",
  "dest-exchange": "orders-ingest",
  "ack-mode": "on-confirm"
}'
```

The critical thing about both RabbitMQ tools: there is **no offset concept**, so failover does not face the offset-translation problem — but it faces a *different* problem. Because messages are republished, a consumer reading from the downstream broker may see messages that were already consumed upstream, and the at-least-once republishing means duplicates are normal. You handle this the same way you handle any at-least-once system: idempotent consumers, which the post on [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) covers in depth. For the broader question of when RabbitMQ's model fits at all versus Kafka's, see [choosing a message broker](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs).

Figure 2 shows the shape that all three tools share at the architecture level: producers write to a source cluster, a replicator carries records across a high-latency WAN link, and a target cluster serves consumers in the second region.

![A grid architecture diagram showing region A producers writing to a source cluster, an asynchronous replicator with about two hundred milliseconds of lag, a target cluster in region B, consumers in region B, and a disaster-recovery failover target](/imgs/blogs/multi-datacenter-geo-replication-2.webp)

## The offset-translation problem

This is the section that separates people who have *operated* multi-region Kafka from people who have only read about it. Get this wrong and your DR failover, the one you practiced in a runbook, reprocesses millions of messages or — worse — silently skips them. The way this works is unintuitive until you see it, and Figure 4 is the picture to hold in your head while you read.

![A branching graph of the MirrorMaker 2 data flow showing a source cluster at offset five million, the MirrorSourceConnector, the target cluster, a checkpoint offset map, the translated offset around three point two million, and a failed-over consumer that seeks to the translated offset](/imgs/blogs/multi-datacenter-geo-replication-4.webp)

### Why offsets are not the same across clusters

A Kafka offset is not a property of a message. It is a *position in a particular partition of a particular topic on a particular cluster*. It is the index of where that record landed in that specific log. When MirrorMaker 2 replicates topic `orders` partition 0 from the source cluster to the target, it consumes the records and re-produces them — and the target partition assigns its *own* offsets as records arrive. There is no law of nature that says record number 5,000,000 in the source becomes record number 5,000,000 in the target, and in practice it almost never does. Here is why.

The source partition has been alive longer, or shorter, than the target partition. The source partition may have had records that were already aged out by retention before replication started, so its offsets begin at, say, 1,200,000 while the target's mirror begins at 0. The source partition may have had transaction control records and aborted-transaction markers that occupy offsets but never get replicated. Compaction may have removed records in one and not yet the other. Replication may have started mid-stream. Every one of these effects shifts the mapping. The result: a record at source offset 5,000,000 might sit at target offset 3,200,000, or 5,800,000, or anything — and the offset *delta* is not even constant across partitions of the same topic, because each partition has its own history.

So if your consumer group `fraud-detector` had committed offset 5,000,000 on the source `orders` partition 0, and you fail it over to the target cluster and tell it to resume at offset 5,000,000, you are pointing it at a *completely different record* — one that is 1.8 million records away from where it should be. Resume too early and you reprocess 1.8 million messages (at-least-once, painful but survivable if you are idempotent). Resume too late and you *skip* messages permanently (data loss, often silent). This is the offset-translation problem, and it is the reason naive Kafka DR runbooks fail in the moment.

### How MirrorMaker 2 translates offsets

MM2 solves this with the `MirrorCheckpointConnector`. As it replicates, it maintains a mapping between source offsets and the corresponding target offsets, derived from the record provenance it tracks during mirroring. It periodically writes **checkpoints** — records in a topic named `<source-alias>.checkpoints.internal` on the target cluster — that say, in effect, "consumer group `fraud-detector` was at source offset 5,000,000, which corresponds to target offset 3,200,000." With `sync.group.offsets.enabled = true`, MM2 goes further and proactively writes those translated offsets into the target cluster's `__consumer_offsets` for the group, so a failed-over consumer that just starts up on the target finds an already-correct committed position and resumes from the right place.

The translation is *approximate and slightly conservative*. MM2 generally errs toward an earlier target offset rather than a later one, because reprocessing a few duplicate messages is recoverable in an at-least-once system but skipping messages is not. This is exactly the right bias, and it is also exactly why your failed-over consumers must be idempotent — you should *expect* to reprocess a small tail of messages after every failover. Cluster Linking sidesteps all of this by preserving offsets exactly, so its "translation" is the identity function; that is the entire value proposition of the offset-preserving approach.

#### Worked example: mapping a consumer offset across clusters

Let me make the numbers concrete. Your `orders` topic, partition 0, has been live in the source cluster `us-east` for eighteen months. Retention has aged off the oldest records, so the partition's *log-start offset* is currently 1,200,000 and its *log-end offset* is 9,400,000 — about 8.2 million live records. You set up MM2 last month. When MM2 started mirroring, it began from the source partition's then-current earliest available offset, and the target topic `us-east.orders` partition 0 started its own offsets at 0.

Now: your `fraud-detector` consumer group has committed source offset **5,000,000** on partition 0. Where does that land in the target?

The target partition started at 0 when MM2 began, at which point the source partition's earliest available record was at, say, source offset 1,800,000 (retention had already moved the floor up from 1,200,000 by the time MM2 attached). So the very first record MM2 copied — source offset 1,800,000 — became target offset 0. The mapping is therefore approximately:

```
target_offset ≈ source_offset − replication_start_source_offset
target_offset ≈ 5,000,000 − 1,800,000
target_offset ≈ 3,200,000
```

So source offset 5,000,000 corresponds to roughly target offset **3,200,000**, a delta of 1.8 million. If you had blindly told the failed-over consumer to resume at 5,000,000 on the target, you would have jumped 1.8 million records *into the future* relative to where you actually were, silently skipping 1.8 million orders' worth of fraud checks. MM2's checkpoint translation is what writes 3,200,000 into the target's offset store instead, so the consumer resumes correctly. The delta is partition-specific: partition 1, with a different history, might have a delta of 2.1 million. There is no single number you can hardcode — which is precisely why you need the checkpoint machinery and cannot do this by hand. And note the asymmetry once more: the translation rounds *backward*, so the consumer will likely reprocess a few thousand records near the boundary. That is fine if and only if it is idempotent.

## Active-passive: the warm DR standby

Active-passive is the topology you should reach for first, because it solves the most common reason to go multi-region — disaster recovery — with the least complexity. One region is active: producers write to it, consumers read from it, it is the source of truth. The second region is passive: a warm standby that continuously receives replicated data but serves no live traffic. It sits there, costing you money, doing nothing visible, until the day the active region fails — and then it is the only reason you still have a company. Figure 3 contrasts the single-region world, where a region outage is a total outage, against the multi-region world, where it is a bounded failover.

![A before-and-after comparison showing a single-region deployment where a region outage causes a full outage with unbounded recovery, versus a multi-region deployment with a second-region replica that fails over in minutes with an RPO measured in seconds](/imgs/blogs/multi-datacenter-geo-replication-3.webp)

### What "warm" actually means

There is a spectrum from cold to hot, and "warm" is a deliberate point on it. A **cold** standby is just the data — replicated topics in a remote cluster, but no running consumers, no pre-provisioned application capacity. To fail over you have to spin everything up, which is cheap to run but slow to recover (high RTO). A **hot** standby runs full application capacity in the second region at all times, ready to take traffic instantly — fast recovery but you are paying for double the compute around the clock, most of it idle. A **warm** standby is the pragmatic middle: the data is continuously replicated and a minimal cluster plus a skeleton of services runs in the standby region, enough to come up in minutes when scaled out, without paying full double capacity continuously.

The thing people forget is that DR is not just your message data — it is your *consumer positions*. If you replicate the `orders` topic but not the consumer offsets, then on failover your consumers start either from the beginning of retention (reprocessing days of data) or from the end (skipping everything in flight). Neither is acceptable. This is why `sync.group.offsets.enabled` is not optional for a real DR setup; the translated offsets are as much a part of your DR data as the messages themselves. This is the single most common mistake in homegrown Kafka DR: people replicate the topics, run a failover drill, and discover their consumers have no idea where they were.

There is also the matter of the *producer* side of failover, which gets less attention than the consumer side but is just as important. When you fail over, your producers must repoint to the new region's bootstrap servers — and that repointing is itself a source of data loss and duplication if you do it carelessly. A producer with in-flight batches and `acks=all` that loses its connection to the dying region may or may not have had those batches acknowledged; on retry against the new region, you can get duplicates if the original write actually landed and replicated. The cleanest pattern is to drive producer endpoint selection through DNS or a service-discovery layer with a short TTL, so a failover is a config flip rather than a code deploy, and to make producers idempotent so retries against the new region do not double-write. The point is that failover is a whole-system event — producers, consumers, offsets, ACLs, configs — and any one of those that you forgot to plan for is the one that bites you at 3 a.m.

### Running a failover drill

A DR setup you have never failed over to is not a DR setup; it is a hope. You must run drills, and they must be real enough to expose the offset and DNS and connection-string problems that only surface under fire. A real Kafka active-passive failover looks like this:

```bash
# 1. Stop producing to the (now dead, or being-drained) source region.
#    In a real outage the source is gone; in a drill you fence it.

# 2. Verify replication has caught up as far as it can — check MM2
#    heartbeat lag and the checkpoint topic for the latest translated offsets.
kafka-console-consumer --bootstrap-server kafka-euw-1:9092 \
  --topic us-east.heartbeats --from-beginning --max-messages 5

# 3. Confirm consumer-group offsets were synced into the target.
kafka-consumer-groups --bootstrap-server kafka-euw-1:9092 \
  --describe --group fraud-detector

# 4. Repoint producers and consumers at the target cluster's bootstrap
#    servers (via DNS / service discovery, not hardcoded IPs).

# 5. Start consumers on the target. With synced offsets they resume
#    from the translated position; expect a small tail of reprocessing.
```

Steps 2 and 3 are where homegrown setups die. If you skipped the checkpoint connector, step 3 returns nothing, and you are now guessing where five hundred consumers should resume. Practice this quarterly. The first time you do it, something will be wrong — a topic that was not on the replication list, an ACL that was not replicated, a consumer group whose offsets were not synced. Better to find it in a drill. The companion post [durability and disaster recovery for message queues](/blog/software-development/message-queue/durability-and-disaster-recovery-for-message-queues) goes deeper on the full DR runbook, RTO targets, and the broader resilience picture beyond replication alone.

### What active-passive does not buy you

Active-passive does nothing for latency. Users still talk to the single active region; the standby serves no reads. It does nothing for data residency on its own, because by default it replicates everything one direction. And the standby capacity is *idle spend* — you are paying for a cluster that produces zero business value until disaster strikes. Those are real costs, and they are the price of the simplest, safest multi-region pattern. If those costs bother you, the temptation is to "use" the standby region by letting it take writes too — and that temptation is exactly how you end up in active-active, which is a far deeper and more dangerous pool than it looks.

## Active-active: writes anywhere, and its problems

Active-active is the architecture every product manager draws on a whiteboard: every region is live, users write to their nearest region, all regions converge to the same global state, and a region failure just means traffic shifts to the others with zero data loss. It is beautiful. It is also the source of more multi-region production incidents than every other topology combined, because the moment you allow writes in more than one place, you have signed up for the two hardest problems in distributed systems: **duplicate detection** and **conflict resolution**. Figure 8 makes the tradeoff explicit against the active-passive baseline.

![A before-and-after comparison contrasting active-passive, where writes happen in one region with one-way conflict-free replication and an idle standby, against active-active, where writes happen in any region with roughly two-millisecond local latency but duplicates and write conflicts](/imgs/blogs/multi-datacenter-geo-replication-8.webp)

### Why writes-anywhere is so hard

When region A and region B both accept writes and replicate to each other, several uncomfortable things become true at once.

First, **every record now exists in two places with two different offsets**, and replication has to carry A's writes to B *and* B's writes to A. If you are not careful, A replicates a record to B, B's replicator sees it as a new local record and replicates it back to A, A sees *that* as new and replicates it to B again — an infinite **replication loop** that doubles your traffic on every round and never terminates. Loop prevention is not optional in active-active; it is the thing that keeps the system from melting. We will dedicate a full section to it.

Second, **the same logical entity can be modified in two regions concurrently**, and because replication is asynchronous, neither region knows about the other's change for tens to hundreds of milliseconds. If a user updates their shipping address in region A at the same instant a customer-service rep updates it in region B, you have two conflicting versions of the truth and no synchronous coordinator to pick a winner. This is the classic multi-leader write-conflict problem, and the deep treatment lives in [distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless). Multi-region active-active *is* multi-leader replication, with all of its conflict baggage, applied to your message log.

Third, **ordering across regions is not preserved**. A message produced in A at time T and a message produced in B at time T+5ms may arrive at a third aggregating region in either order, because the two WAN paths have different latencies. Any consumer that depends on cross-region ordering is going to have a bad time. Within a single partition in a single region, Kafka gives you total order; *across* regions, you get no such guarantee, and you must design as if you have none. The post on [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) explains why ordering is a per-partition local property — and crossing regions breaks exactly that locality.

### The patterns that make active-active survivable

Active-active is survivable, and many large companies run it, but only with discipline. The patterns that work all share one idea: avoid concurrent conflicting writes to the same key in different regions, rather than trying to resolve them after the fact.

The strongest pattern is **partition ownership by region**, sometimes called regional sharding or write-locality. You assign each entity — each user, each account, each tenant — a *home region*, and all writes for that entity go to its home region only; other regions get a read-only replicated copy. There is exactly one writer per key, so there are no write conflicts by construction. Region B can serve reads of an entity homed in A from its local replica (fast, possibly slightly stale), but writes route to A. This turns "active-active" into "active-active at the cluster level, active-passive per key," which is the only version of active-active most teams should ever run. It also dovetails neatly with data residency: home an EU user in the EU region and their writes never leave.

When you genuinely cannot avoid concurrent writes, you fall back to **conflict resolution policies**: last-writer-wins by timestamp (simple, lossy — it silently discards the losing write), application-level merge functions (correct but you have to write them per entity type), or CRDTs (conflict-free replicated data types, which converge automatically but constrain what operations you can express). Every one of these is more complex than partition ownership, and every one occasionally surprises you. The honest senior-engineer position is: do partition-by-region if you possibly can, and only accept true write conflicts where the business genuinely requires globally-writable shared state — which is rarer than people think.

#### Worked example: the duplicate-event blast radius in active-active

Let me quantify why active-active is so much riskier operationally, not just conceptually. Suppose you run bidirectional MM2 between two regions, A and B, each producing 30,000 messages per second to a shared `events` topic, and your inter-region replication lag is 250 milliseconds. In steady state this is fine — A's records flow to B as `A.events`, B's flow to A as `B.events`, and the prefix prevents loops. Now suppose a deployment bug temporarily strips the replication-policy prefix on region B's replicator for *just 10 seconds* before you catch it and roll back. During those 10 seconds, B replicates `A.events` records back to A under a bare `events` name. A's replicator sees them as new local records and sends them back to B. Count the damage: in 10 seconds, region A produced 300,000 records that were legitimately replicated to B; the moment the prefix is stripped, those 300,000 records start echoing. Even if the loop only runs for those 10 seconds before rollback, you have injected on the order of 300,000 duplicate records into the cross-region stream, plus a second-order echo of those, and your consumers — which read both `events` and the remote-prefixed topics — now see roughly double the volume for the affected window. If those consumers are not idempotent, every one of those 300,000 duplicates becomes a duplicated side effect: a double-counted metric, a re-sent notification, a re-charged transaction. Compare this to active-passive, where the same 10-second deployment bug would have caused, at worst, a brief replication stall and zero duplicates because the flow is one-directional and there is no return path to echo into. *That* is the active-active tax measured in blast radius: a single misconfiguration that is a non-event in active-passive becomes a six-figure duplicate storm in active-active. This is why the recommendation is so emphatic — every additional writable region multiplies the number of ways a small mistake becomes a large incident.

## Hub-and-spoke and fan-out topologies

Between the simple active-passive and the dangerous active-active sit two directional topologies that solve specific problems cleanly *because* data flows one direction. Figure 6 places all four topologies in a taxonomy so you can see how they relate — single-writer models on one branch, multi-writer on the other.

![A taxonomy tree of geo-replication topologies branching from single-writer models into active-passive and fan-out, and from multi-writer models into active-active and hub-and-spoke](/imgs/blogs/multi-datacenter-geo-replication-6.webp)

### Hub-and-spoke: many regions into one

Hub-and-spoke is the global-aggregation pattern. You have N spoke clusters — one per region, each capturing local events — and one hub cluster that aggregates all of them. Every spoke replicates *into* the hub; the hub replicates back to *no one* (or replicates a curated subset back out, but the primary flow is inward). The result is a single cluster, the hub, that holds the union of all regional event streams, ready for global analytics, fraud detection across regions, a unified data warehouse feed, or any workload that needs to see everything.

The reason hub-and-spoke is so much safer than active-active is that it is *write-disjoint*. Each spoke is the sole writer of its own data; the hub never writes data that flows back to a spoke. There are no conflicts because no two writers ever touch the same records. There are no loops because the flow is acyclic — spokes to hub, full stop. The hub does need enough capacity to hold the *sum* of all spoke throughput, which is the main capacity-planning gotcha: if each of twelve spokes does 20k messages per second, your hub must ingest and store 240k per second, which is a serious cluster. But the topology itself is conflict-free and loop-free, which is why it is the go-to for global aggregation.

With MM2, hub-and-spoke is just N one-way replication flows all targeting the same cluster, with the source-alias prefixing keeping each spoke's topics namespaced in the hub:

```properties
# Each spoke replicates its 'events' topic into the hub.
# In the hub, they appear as us-east.events, eu-west.events, ap-south.events ...
clusters = us-east, eu-west, ap-south, hub

us-east->hub.enabled = true
us-east->hub.topics = events, transactions

eu-west->hub.enabled = true
eu-west->hub.topics = events, transactions

ap-south->hub.enabled = true
ap-south->hub.topics = events, transactions

# The hub replicates to no one in pure hub-and-spoke.
```

A consumer on the hub that wants the global stream subscribes to a pattern like `.*\.events` and gets all regions' events in one consumer group. Clean.

### Fan-out: one region to many

Fan-out is the mirror image: one source region produces, and the data is broadcast to many regions that consume it locally. This is the pattern for distributing a global reference dataset — a product catalog, a pricing table, a feature-flag stream, a config feed — from a central authority out to every region so that each region's services read it locally at single-digit-millisecond latency instead of reaching across the WAN. There is one writer (the source) and N readers (the regions), so like hub-and-spoke it is write-disjoint and conflict-free. The flow is acyclic — source to each region, no return path — so there are no loops.

Fan-out's capacity profile is the inverse of hub-and-spoke: the source must push its full throughput out *N times*, once per destination region, so your WAN egress scales with the number of regions. For a slow-changing reference dataset that is trivial; for a high-volume stream fanned out to twenty regions it is a real bandwidth bill. But it is a bandwidth problem, not a correctness problem, and bandwidth problems are the kind you want.

### Why directional topologies are the sweet spot

Notice what hub-and-spoke and fan-out have in common: data flows in exactly one direction along every link, so the global graph is a DAG. No cycles means no loop-prevention machinery to maintain. One writer per record means no conflict resolution to debug. You get genuine multi-region benefits — global aggregation, low-latency local reads of shared data — without paying the active-active complexity tax. Whenever you can express your problem as a directional flow, do it; reach for active-active only when the business truly requires writes to the *same* data in multiple regions, which, again, is rarer than the whiteboard suggests.

A common and powerful refinement is to combine these directional topologies into a single design. Many large systems run a hub-and-spoke for their analytics and aggregation path *and* a fan-out for their reference-data path *at the same time*, because those two flows are independent and both directional. Regional events flow inward to the analytics hub; the global product catalog flows outward to every region. Neither path has conflicts or loops, and the two together give you a system that is globally observable and locally fast without ever needing a writable-everywhere design. When someone proposes active-active, the first question to ask is whether the real requirement can be decomposed into an inward aggregation flow plus an outward distribution flow — because if it can, you have just turned the hardest topology into two of the easiest, and your future on-call self will thank you.

One more nuance on hub-and-spoke capacity that catches teams off guard: the hub does not just need ingest bandwidth equal to the sum of the spokes — it needs *storage* and *consumer* capacity for that sum too. If you retain the aggregated stream for seven days at 240k msg/s with an average record of 1KB, that is roughly 240 MB/s of writes, which over seven days is on the order of 145 TB of retained log on the hub. The hub is, by construction, the largest cluster in your fleet, and it is the one most likely to hit storage and partition-count limits first. Plan it as a first-class capacity problem, not as an afterthought bolted onto your regional clusters. The partitioning math for a cluster of that size is its own discipline — [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) covers the principles that apply directly to how you partition a high-volume aggregation hub.

## Asynchronous replication and the RPO window

We have circled this several times; now we pin it down with numbers, because your RPO is a number you must be able to state, defend, and design to. Figure 5 shows the anatomy of a failover and exactly which messages fall into the RPO window — the ones produced and acknowledged in the source region but not yet replicated when the region died.

![A timeline of a regional failover showing steady-state replication lag of about two hundred milliseconds, the hard failure of region A, roughly ten thousand un-replicated records lost, promotion of cluster B, producers repointing, and the booked RPO once serving resumes from region B](/imgs/blogs/multi-datacenter-geo-replication-5.webp)

### Replication lag *is* your RPO

Recovery Point Objective is the answer to "how much data are we willing to lose in a disaster?" — measured in time, or equivalently in messages. For a synchronous system the RPO can be zero, because nothing is acknowledged until it is safely in a second place. But we established that nobody runs synchronous cross-region replication for messaging, because the transcontinental round trip would wreck local latency and throughput. So your RPO is not zero; it is exactly your **replication lag** at the moment of failure. Whatever the source region had acknowledged but not yet shipped across the WAN is gone, because it never existed anywhere but the region that just disappeared.

This is the single most important sentence in the post, so let me say it plainly: **with asynchronous cross-region replication, your RPO equals your replication lag, and your replication lag is never zero.** You do not get to configure it away. You only get to make it smaller — by provisioning more replicator throughput, fatter WAN pipes, and more replication parallelism — and you get to *measure* it, so the number is honest, and you get to *price* it, so the business decides consciously how many messages it is willing to lose.

### Measuring lag honestly

MM2's heartbeat connector emits timestamped heartbeats into the source that get replicated to the target; the difference between the heartbeat's original timestamp and when it lands on the target is your end-to-end replication lag. You should alert on it. A lag that normally sits at 200ms and spikes to 30 seconds during a traffic surge means your RPO just got 150x worse precisely when you are most likely to need it. Replicator lag is not a steady number — it grows under load, during network congestion, during target-cluster slowness — and it grows *most* during the kind of stress that precedes outages. Treat a lag spike as a leading indicator of risk, not just a metric.

There are two different lag numbers and you must not confuse them. **Throughput lag** is how far behind the replicator is in record count — the difference between the source's log-end offset and the offset the replicator has successfully copied. **Time lag** is how old the most recently replicated record is — the wall-clock gap the heartbeat measures. Time lag is the one that maps to RPO, because RPO is "how many seconds of data would I lose," but throughput lag is what tells you *why* time lag is growing (the replicator cannot keep up with the produce rate). Watch both: rising throughput lag is the early warning, and time lag is the number you put in your RPO dashboard. A subtle trap: if produce volume drops to zero, time lag keeps growing (the last record gets older) even though the replicator is perfectly caught up — which is why the heartbeat topic, with its constant trickle of timestamped records, is the honest lag signal rather than inferring lag from your application topics alone.

A second operational point: lag is per-partition, and your RPO is governed by the *worst* partition, not the average. If 99 of your 100 partitions replicate with 150ms lag but one partition — perhaps a hot one with a skewed key — runs 5 seconds behind, your effective RPO for that partition's data is 5 seconds. Averages lie here. Monitor the maximum lag across all replicated partitions, and investigate any partition that consistently lags the pack, because it is usually a sign of key skew or a single overloaded replicator task that needs more parallelism.

#### Worked example: how many messages are at risk on failover

Now the number that matters to your VP. Your source region produces to the `orders` topic at a steady **50,000 messages per second**. Your measured steady-state cross-region replication lag is **200 milliseconds**. The source region suffers a hard failure with no warning. How many acknowledged messages do you lose?

The messages at risk are exactly those produced during the lag window — acknowledged locally but not yet replicated:

```
messages_at_risk = produce_rate × replication_lag
messages_at_risk = 50,000 msg/s × 0.200 s
messages_at_risk = 10,000 messages
```

So a "healthy" 200-millisecond lag means a hard failover loses about **10,000 orders**. That is your RPO, stated in the only unit a business understands: ten thousand lost orders. Now watch what happens under stress. Suppose the failure is preceded by a traffic spike that pushes produce to 80,000 msg/s *and* congests the WAN so replication lag climbs to 2 seconds:

```
messages_at_risk = 80,000 msg/s × 2.0 s
messages_at_risk = 160,000 messages
```

Same architecture, sixteen times the loss, because lag and rate both spiked together — which is exactly the correlated-failure scenario you must plan for. This is why you do not quote your *average* lag as your RPO; you quote your *worst-case-under-stress* lag, because that is the lag you will actually have when the region dies. And it is why driving down replicator latency is not premature optimization: every 100ms you shave off lag is 5,000 fewer orders lost in this example. If 10,000 lost orders is unacceptable, your options are: shrink the lag (more replicator capacity, fatter pipes), reduce the blast radius (regional sharding so only one region's orders are ever at risk), or accept synchronous replication's latency penalty for the specific high-value topics where zero RPO is worth it. There is no free lunch — only an honest, numbers-backed choice.

## Loop prevention and conflict handling

If you run active-active or any bidirectional topology, loop prevention is the machinery that stops your replicators from feeding each other in an infinite cycle, and conflict handling is what you do when the same data is written in two places. Get loop prevention wrong and your cluster does not lose data — it *drowns* in duplicated traffic until it falls over. This is the section to read twice before you turn on bidirectional replication.

### How replication loops happen

Picture two regions, A and B, replicating to each other for active-active. A producer writes record `R` to region A. A's replicator copies `R` to region B. Now B has `R`. But B's replicator is configured to copy B's records to A — and it sees `R` sitting in B's log as just another record. It does not inherently know `R` *came from* A. So it copies `R` back to A. A's replicator sees the returned `R` as a new local record and copies it to B again. The record ping-pongs forever, and every other record does the same, so your inter-region traffic explodes geometrically. Within minutes the WAN link and both clusters are saturated with nothing but echoes. This is the replication loop, and it is the first thing that breaks in a naive bidirectional setup.

### The prefix / provenance solution

MirrorMaker 2 prevents loops with its naming policy and provenance tracking. When MM2 replicates topic `orders` from cluster `A` to cluster `B`, it names the target topic `A.orders` — the source alias is prefixed. Now the key rule: **MM2 only replicates topics that do not already carry a remote-cluster prefix matching the destination**, and its `DefaultReplicationPolicy` recognizes its own prefixing. When B's replicator looks at its topics to mirror back to A, it sees `A.orders` and recognizes that prefix means "this came from A" — so it does not replicate `A.orders` back to A. Records originating in B live in plain `orders` (no prefix) and *do* get replicated to A as `B.orders`. The prefix is the provenance marker that breaks the cycle: a record is never sent back to the cluster named in its prefix.

So in steady active-active, region A holds `orders` (its own writes) and `B.orders` (replicated from B); region B holds `orders` (its own writes) and `A.orders` (replicated from A). A consumer that wants the *global* order stream subscribes to both `orders` and the remote-prefixed topics. The model is clean once you see it: local topics are unprefixed, remote topics carry their origin's name, and nothing flows back to its origin. Cluster Linking and other tools have their own provenance mechanisms, but the principle is universal — **every record must carry where it came from, and no replicator forwards a record back toward its origin.**

This is also why the temptation to "simplify" by using identical topic names everywhere is so dangerous, and why the case study below about a doubled-traffic loop happened. The prefix is not for humans; it is the loop-breaker. If your application teams find the prefixed names awkward, the right fix is a consumer-side wrapper — a topic-pattern subscription or a thin client library that hides the naming — not stripping the provenance that keeps the system from eating itself. There is a general principle here that applies to any bidirectional replication scheme in any broker: **the provenance metadata is load-bearing infrastructure, and removing it to make things prettier is removing a load-bearing wall.** Treat the naming convention with the same respect you treat the replication factor, because in a bidirectional topology it is doing exactly as critical a job: the replication factor keeps you from losing data when a broker dies, and the provenance prefix keeps you from drowning in duplicates when both regions are healthy. Both are non-negotiable, and both are the kind of thing that looks optional right up until the moment it is the root cause on your incident timeline.

### Conflict handling when writes do collide

Loop prevention stops infinite echoes; it does *not* resolve the case where the same logical entity was genuinely written in two regions concurrently. That is conflict handling, and your options, from best to worst:

| Strategy | How it works | When to use | The catch |
| --- | --- | --- | --- |
| Partition by region | Each key has a home region; only that region writes it | Almost always, if you can | Cross-region writes must route to home (a little latency) |
| Last-writer-wins | Highest timestamp wins | Tolerable data loss, simple needs | Silently discards the losing write; clock skew bites |
| Application merge | Custom merge function per entity | Correctness matters, conflicts are structured | You must write and test merge logic per type |
| CRDTs | Data types that converge by construction | Counters, sets, collaborative state | Constrains the operations you can express |

The overwhelming recommendation: **partition by region** so conflicts cannot happen, and treat the other three as fallbacks for the narrow cases where shared global writes are genuinely required. Last-writer-wins in particular is a trap — it looks simple, it works in the demo, and then a clock skew of a few hundred milliseconds between regions silently discards a customer's real change in favor of a stale one, and you find out from a support ticket three weeks later. If you must use it, use a logical clock or a hybrid logical clock, not wall time, and accept that you are trading correctness for simplicity with eyes open. The consistency tradeoffs here are the same ones the [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) post frames formally: in the multi-region case you are almost always choosing availability and low latency over strong consistency, and conflict handling is where that choice gets paid for.

## Choosing a topology

You now have all four topologies and the tools to implement them. Here is how to actually choose, framed as a decision flow rather than a menu. Start from the *business pressure*, not the technology, because the pressure determines the topology and the topology determines the tool.

**If your driver is disaster recovery and nothing else, choose active-passive.** It is the simplest, safest, conflict-free topology. One write region, one warm standby, one-way replication with synced offsets. Use MM2 if you are on open-source Kafka and can tolerate offset translation plus idempotent consumers; use Cluster Linking if clean offset-preserving failover is worth the Confluent cost. Do not over-build — most teams that "need multi-region" need exactly this and nothing more.

**If your driver is global aggregation — one place that sees everything — choose hub-and-spoke.** Each region writes its own data; everything flows into a hub; no conflicts, no loops, just a fat hub you must capacity-plan for the sum of all spokes. This is the analytics and global-fraud pattern, and it is delightfully boring to operate.

**If your driver is distributing shared reference data to every region for fast local reads, choose fan-out.** One writer, N readers, acyclic, conflict-free; your only concern is WAN egress scaling with region count.

**If your driver is genuinely low-latency local writes to the *same* data in multiple regions, and you have exhausted the option of partitioning by region, only then choose active-active** — and go in knowing you have signed up for loop prevention, conflict resolution, no cross-region ordering, and a class of incidents the other three topologies simply do not have. Most teams that think they need active-active actually need active-active *at the cluster level with per-key regional ownership*, which gives them the latency and availability wins without the true write-conflict pain.

Figure 7 shows the layered view that underlies every one of these choices: a record passes through the producer, the local cluster, the cross-DC replicator, and the remote cluster — and the replicator is always the slow, lag-introducing, RPO-defining layer, no matter which topology you wrap around it.

![A layered stack diagram showing a record passing through the producer, the local in-region cluster, the asynchronous cross-datacenter replicator over the WAN, and the remote cluster holding the durable copy](/imgs/blogs/multi-datacenter-geo-replication-7.webp)

### A decision table you can defend

| Business driver | Topology | Write model | Primary tool | Key risk |
| --- | --- | --- | --- | --- |
| Disaster recovery | Active-passive | Single write region | MM2 / Cluster Linking | Offset translation; idle spend |
| Global aggregation | Hub-and-spoke | N writers, 1 sink | MM2 (N one-way flows) | Hub capacity = sum of spokes |
| Distribute reference data | Fan-out | 1 writer, N readers | MM2 / Cluster Linking | WAN egress × region count |
| Low-latency local writes | Active-active | Any region writes | MM2 (bidirectional) | Loops, conflicts, no global order |
| Data residency | Regional sharding | Per-key home region | Filtered replication | Routing writes to home region |

## Case studies and war stories

Theory survives contact with production differently than it survives a blog post. Here are four patterns drawn from how real organizations run multi-region messaging, and the specific lesson each teaches.

### The DR failover that reprocessed three days of data

A payments company ran Kafka active-passive across two regions with MirrorMaker — but they replicated only the topics, not the consumer-group offsets, because the checkpoint connector "seemed optional" and the docs were confusing. They ran a DR drill once, it "worked" (the data was there), and they signed off. A year later the primary region had a genuine multi-hour outage. They failed over. The consumers started on the standby cluster with no committed offsets, so per the default `auto.offset.reset=earliest`, every consumer group started from the *beginning of retention* — three days back. Suddenly every downstream system was reprocessing three days of payment events at once: duplicate fraud alerts, duplicate ledger entries (the ledger was, thankfully, idempotent and absorbed them), and a thundering-herd load spike that took an hour to drain. **Lesson:** replicating topics without replicating and translating offsets is not DR; it is a data-reprocessing bomb with a delay fuse. The offset state is part of your recovery data, full stop.

### The active-active loop that doubled traffic every minute

A media company wanted active-active for low-latency writes in two regions and set up bidirectional MirrorMaker — but they used the `IdentityReplicationPolicy` to keep topic names identical across regions, because the prefixed names "looked ugly" to their application teams. The identity policy strips the provenance prefix that prevents loops. So region A's records replicated to B under the same name, B's replicator saw them and replicated them back to A, and the echo began. Inter-region traffic doubled every replication cycle. Their WAN link saturated within minutes, lag exploded, and both clusters started shedding load. They had to kill replication entirely, purge the duplicated records, and rebuild with the default prefixing policy. **Lesson:** the topic-name prefix is not cosmetic — in bidirectional replication it *is* the loop-prevention mechanism. Strip it and you strip the only thing stopping an infinite cycle. Provenance must survive replication.

### The residency violation hiding in a replication config

A SaaS company subject to GDPR ran a hub-and-spoke topology to aggregate all regional events into a central US analytics hub. It worked great — until a compliance audit discovered that EU personal data was flowing into the US hub through the `eu-west.events` mirror, which violated their data-residency commitments. The replication config said "replicate `events` from every spoke to the hub," and `events` happened to contain personally identifiable EU customer data. Nobody had filtered it. **Lesson:** in any multi-region topology, your replication config is a *data-export policy*, and it must be reviewed as one. Filter at the replication boundary — replicate only the topics, and ideally only the fields, that are legally allowed to cross. "Replicate everything" is a compliance incident waiting for an auditor. Data residency forces you to think about *what* crosses the WAN, not just whether replication works.

### The RPO nobody had ever calculated

A logistics company had a multi-region Kafka setup and a stated RPO of "near zero" in their DR documentation — a number that had been copied from a template and never computed. During an actual regional failover, they lost about 90 seconds' worth of shipment-scan events, because their replicator had been quietly running 90 seconds behind for weeks under their growing load, and nobody alerted on replication lag. The "near zero" RPO was fiction; the real RPO was 90 seconds, which at their event rate was roughly 400,000 lost scans. The post-incident review found that the lag metric existed but had no alert, and the gap between the documented RPO and the real one had never been measured because no one had multiplied lag by rate. **Lesson:** your RPO is not what the document says; it is your *measured worst-case replication lag times your produce rate*, and if you are not alerting on lag, you do not know your RPO — you are guessing, and the guess is optimistic.

## When to reach for multi-region (and when not to)

**Reach for multi-region messaging when:** a regional outage is an existential business risk (payments, healthcare, anything where hours of downtime is catastrophic); when you have a genuine regulatory residency requirement; when you have user populations far enough apart that cross-region latency materially hurts the product; or when you need global aggregation of regionally-produced data. These are real, common, and worth the complexity.

**Do not reach for it when:** you are a single-region product with single-region users and a tolerance for a few hours of downtime in a rare regional event — the complexity, the idle standby spend, and the new failure modes (replication lag, loops, conflicts, offset translation) cost more than the rare outage they prevent. Multi-region is not a maturity badge; it is a specific solution to specific pressures. Plenty of excellent, large systems run in one region with strong in-region replication and good backups, and they are right to. Adding a second region to a system that does not need one *reduces* your reliability by adding moving parts, not increases it.

**And when you do reach for it, reach for the simplest topology that solves your actual pressure.** Active-passive for DR. Hub-and-spoke for aggregation. Fan-out for reference data. Save active-active for the genuine, business-mandated case of writes to the same data in multiple regions — and even then, partition by region first. The ranking from simplest-and-safest to hardest is exactly the order in which you should consider them.

A final piece of hard-won advice: whatever topology you pick, write down your RPO as a *number you have actually computed*, put a lag alert in place that fires well before that number is breached, and schedule a real failover drill on the calendar — recurring, not "someday." These three habits separate teams whose multi-region setup works in a crisis from teams who discover during the crisis that their setup was theater. The architecture diagrams are the easy part. The discipline of measuring your real RPO, alerting on the lag that defines it, and rehearsing the failover until it is boring — that is the part that actually saves you the night the region goes dark.

## Key takeaways

- **Cross-region replication is always asynchronous, so your RPO equals your replication lag and is never zero.** Measure it, alert on it, price it. The messages in the lag window are gone on a hard failover.
- **Compute your RPO in messages, not vibes:** produce rate times worst-case-under-stress lag. At 50k msg/s and 200ms lag that is 10,000 lost messages; under a correlated spike it can be 16x worse.
- **Kafka offsets are cluster-local positions, not message identities.** A consumer at source offset 5,000,000 maps to a *different* target offset (e.g. ~3,200,000), per-partition, because each partition has its own history. You cannot hardcode the delta.
- **MirrorMaker 2 translates offsets approximately via checkpoints and biases backward** (reprocess, never skip) — so failed-over consumers must be idempotent. **Cluster Linking preserves offsets byte-for-byte**, eliminating the translation problem at a commercial cost.
- **Active-passive is the topology to reach for first** — simplest, conflict-free, the right answer for pure DR. Replicate offsets, not just topics, or your failover reprocesses or skips everything.
- **Active-active means multi-leader replication** with all its conflict baggage. Prefer **partition-by-region** (one writer per key) so conflicts cannot occur; treat last-writer-wins, merges, and CRDTs as narrow fallbacks.
- **Loop prevention is mandatory in bidirectional topologies.** Provenance — MM2's source-alias prefix — is what stops records from echoing forever. Strip it and your cluster drowns in duplicated traffic.
- **Directional topologies (hub-and-spoke, fan-out) are the sweet spot** when they fit: one-way flow means no loops and no conflicts, with genuine multi-region benefits.
- **Your replication config is a data-export policy.** Review it for residency; "replicate everything" is a compliance incident in waiting.
- **A DR setup you have never failed over to is a hope, not a plan.** Drill quarterly; the first drill always finds a missing topic, ACL, or offset sync.

## Further reading

- [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — how replication works *inside* one cluster, the foundation this post extends across the WAN.
- [Durability and disaster recovery for message queues](/blog/software-development/message-queue/durability-and-disaster-recovery-for-message-queues) — the full DR runbook, RTO targets, backups, and resilience beyond replication alone.
- [Choosing a message broker: Kafka, RabbitMQ, Pulsar, NATS, SQS](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs) — when each broker's replication model fits your problem.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the consumer-side discipline every failover and republishing scheme requires.
- [Message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — why ordering is a per-partition local property that cross-region replication breaks.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the formal model behind active-active write conflicts.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the consistency spectrum you are choosing on with async replication.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the availability-versus-consistency tradeoff that every multi-region decision pays for.
- Apache Kafka documentation: Geo-Replication (MirrorMaker 2) — the official connector configuration reference.
- Confluent documentation: Cluster Linking — the offset-preserving replication design and operational guide.
- RabbitMQ documentation: Federation and Shovel plugins — cross-broker message movement without offsets.

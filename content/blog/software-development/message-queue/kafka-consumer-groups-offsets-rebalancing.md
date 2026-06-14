---
title: "Kafka Deep Dive, Part 2: Consumer Groups, Offsets, and Rebalancing"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "How a Kafka consumer group splits a topic across its members, tracks progress in the __consumer_offsets topic, and survives the infamous rebalance — heartbeats, timeouts, the join/sync handshake, eager vs cooperative rebalancing, the four partition assignors, and static membership for rolling restarts."
tags:
  [
    "message-queue",
    "kafka",
    "consumer-groups",
    "offsets",
    "rebalancing",
    "distributed-systems",
    "event-driven",
    "partitioning",
    "consumers",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-1.webp"
---

A Kafka topic does not know how many consumers it has, and it does not care. It is a partitioned, append-only log sitting on a set of brokers, and it will happily accept writes whether anybody is reading or not. All of the interesting machinery — the part that decides who reads what, the part that remembers how far each reader got, the part that reshuffles work when a reader dies or a new one shows up — lives entirely on the consumer side, in an abstraction called the **consumer group**. If the storage layer is the part of Kafka that earns its reputation for durability and throughput (covered in [Part 1 on log segments, the page cache, and storage](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage)), the consumer group is the part that earns its reputation for being subtle, occasionally infuriating, and absolutely central to running Kafka at scale.

Here is the one sentence that holds the whole thing together: **within a consumer group, every partition is owned by exactly one consumer at a time.** That is the invariant Kafka enforces, and almost everything else in this post is a consequence of it. It is why you can scale a group up to the partition count and not one consumer further. It is why a slow or crashed consumer triggers a redistribution of partitions — a **rebalance** — so that some other member picks up its work. It is why offsets are tracked per partition and not per consumer. And it is why the rebalance, the process of moving partition ownership around, is the single most operationally consequential event in the Kafka consumer lifecycle: get it wrong and you get duplicate processing, stalled consumption, or in the worst case a group that thrashes itself into uselessness.

![A consumer group where a broker acting as group coordinator assigns three partitions across two consumers so that each partition has exactly one owner](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-1.webp)

By the end of this post you will be able to size a consumer group against a topic's partition count and know exactly when adding consumers stops helping. You will know where Kafka stores your consumer's progress, why it is a compacted topic and not a database, and the precise difference between auto-commit and manual commit including the duplicate-and-loss windows each one opens. You will be able to read a rebalance from the logs: which timeout fired, which member dropped, who became the group leader, and how long the group stalled. You will understand the difference between eager stop-the-world rebalancing and cooperative incremental rebalancing well enough to choose an assignor on purpose instead of inheriting a default. And you will know the one configuration — static membership — that turns a rolling restart from a sequence of disruptive rebalances into a no-op. This post goes much deeper than the rebalance sketch in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log); think of that post as the map and this one as the territory.

## 1. Consumer groups: one owner per partition

Start with the smallest possible mental model and refuse to complicate it until you have to. A **topic** is a named log split into some fixed number of **partitions**, say twelve. Each partition is an independent ordered sequence of records, addressed by a monotonically increasing integer called an **offset**. A **consumer** is a process running the Kafka consumer client. A **consumer group** is a set of consumers that share a single string identifier, the `group.id`, and that cooperate to consume a topic exactly once *as a group*.

The cooperation rule is the invariant from the intro, and it is worth restating with full precision because the precision is the whole point. Within one group, **each partition is assigned to exactly one consumer instance.** Not at most one. Not usually one. Exactly one, at all times when the group is in a stable state. The figure above shows the minimal version: three partitions, two consumers, a broker acting as **group coordinator** that hands P0 and P1 to consumer C1 and P2 to consumer C2. No partition is read by two members; no record is delivered twice within the group during stable operation. That is the guarantee a consumer group buys you, and it is what makes a group a unit of *parallel-but-non-overlapping* consumption.

### Why "exactly one owner" is the right design

Picture a different design where every consumer reads every partition and they coordinate on which records to actually process — a shared work queue with locking. Kafka deliberately does not do this, and the reason is the same reason Kafka is fast in the first place: it wants each consumer to read its partitions sequentially, from a single position, with no per-record coordination. When a consumer owns a partition outright, it can fetch a large batch starting at its current offset, process the batch in order, and advance a single number. There is no lock to take, no contention with another consumer over the same record, no need to ask a central authority "may I process offset 4,210?" The ownership *is* the lock, granted once per rebalance and held until the next one.

This is also what gives Kafka its **per-partition ordering guarantee** at the consumer level. Because exactly one consumer reads a partition, that consumer sees the partition's records in offset order, full stop. If you need two events to be processed in order relative to each other, you put them in the same partition (usually via a partition key — covered in [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees)), and the single-owner rule guarantees one consumer processes them in sequence. Spread them across partitions and you have given up cross-partition ordering on purpose, in exchange for parallelism. The consumer group is the mechanism that turns "twelve independent partitions" into "up to twelve parallel workers, each strictly ordered."

### Groups are independent of each other

One more foundational point that trips up beginners: **different groups are completely independent.** The single-owner rule applies *within* a group, not across groups. If you have an analytics group and a billing group both subscribed to the same `payments` topic, each group gets its own full copy of every record, and each group tracks its own offsets independently. Partition P0 is owned by exactly one consumer *in the analytics group* and, separately, by exactly one consumer *in the billing group*. The two groups never interfere; they do not even know about each other.

This is the feature that lets Kafka serve as both a queue and a publish-subscribe system at once, the duality explored in [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models). One group = a queue (work split among members). Many groups on the same topic = pub/sub (every group gets everything). The `group.id` string is the entire switch. Reuse it and you scale a queue; pick a new one and you fan out a new subscriber. There is no other knob.

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "payments",
    bootstrap_servers="broker1:9092,broker2:9092",
    group_id="billing-service",        # the group identity — this is the whole thing
    enable_auto_commit=False,          # we will commit manually (more on this later)
    auto_offset_reset="earliest",      # where to start if we have no committed offset
    max_poll_records=500,
)

for record in consumer:
    handle_payment(record.value)       # exactly one consumer in billing-service sees each record
```

That `group_id` line is doing an enormous amount of work. It decides which partitions this process is eligible to own, which offsets it reads and writes, which other processes it will rebalance with, and whether this process is "the billing queue" or "a brand-new independent subscriber." Everything in the rest of this post hangs off it.

### subscribe versus assign: the two ways to read

There is a fork in the road the moment you create a consumer, and it is worth naming because it determines whether you are in the consumer-group machinery at all. The `subscribe()` call — what the snippet above uses — opts you into the group: you declare *which topics* you want, and Kafka, through the coordinator and the assignor, decides *which partitions* you get and can change that decision at any time via a rebalance. You give up control of partition assignment in exchange for automatic load distribution and failover. This is what almost everyone wants and what this entire post is about.

The other path is `assign()`, where you name *exact* partitions yourself — `consumer.assign([TopicPartition("payments", 0)])` — and Kafka never rebalances you, never reassigns, never coordinates. You are entirely on your own for distributing partitions across processes and handling failures. There is no group, no coordinator-driven membership, no rebalance. People reach for `assign()` for tightly-controlled use cases: a process that must own one specific partition for stateful reasons, or a system that does its own partition-to-worker mapping. The two modes are mutually exclusive on a single consumer — you cannot mix `subscribe()` and `assign()` — and 95% of the time you want `subscribe()` and the group it puts you in. The rest of this post assumes `subscribe()`, because that is where offsets, coordinators, and rebalances all live.

## 2. Scaling consumers up to the partition count (and the idle-consumer ceiling)

Now the question every capacity-planning meeting asks: I have a topic falling behind, how many consumers do I add? The answer is governed by one hard ceiling and it is not subtle once you see it.

Because each partition is owned by exactly one consumer, **the maximum useful parallelism in a group equals the partition count.** A twelve-partition topic can keep at most twelve consumers busy. Add a thirteenth consumer and it will join the group, participate in rebalances, send heartbeats, and consume exactly nothing, because there is no partition left to assign to it. It is not an error. It is not a warning. It is a perfectly healthy idle consumer, burning a connection and a thread for no throughput, sitting there as a hot standby in case one of the twelve active members dies.

### The three-regime picture

Think of consumer count `C` relative to partition count `P` as three regimes:

- **`C < P` (fewer consumers than partitions):** every consumer is busy and at least one consumer owns multiple partitions. With 12 partitions and 4 consumers, each consumer owns 3 partitions. This is the common steady state. Your parallelism is `C`; your headroom for scaling is `P − C`.
- **`C = P` (one consumer per partition):** maximum parallelism, every consumer owns exactly one partition, perfectly balanced if the assignor cooperates. This is the sweet spot for throughput. 12 partitions, 12 consumers, one each.
- **`C > P` (more consumers than partitions):** `P` consumers are active and `C − P` consumers are idle. With 12 partitions and 16 consumers, 12 work and 4 idle. The idle four contribute zero throughput but do provide fast failover — if an active member dies, an idle standby can take its partition on the next rebalance without you having to spin up a new process.

The practical consequence is that **partition count is a throughput cap you bake in at topic-creation time and pay for forever.** If you might one day need 30 parallel consumers, you need at least 30 partitions, and you generally want to over-provision partitions at creation because increasing partition count later is disruptive (it changes key-to-partition mapping and, as section 7 will note, itself triggers a rebalance). The usual advice — pick a partition count well above your current consumer count, sized to your peak future parallelism plus headroom — is entirely a consequence of this ceiling.

#### Worked example: a 12-partition topic with 4, 12, and 16 consumers

Take a concrete topic, `orders`, with **12 partitions** numbered P0 through P11. Walk through three group sizes and write out the actual assignment a range-style assignor would produce.

**Case A — 4 consumers (C < P).** The assignor divides 12 partitions among 4 members, 3 each:

| Consumer | Partitions owned | Count |
| --- | --- | --- |
| C1 | P0, P1, P2 | 3 |
| C2 | P3, P4, P5 | 3 |
| C3 | P6, P7, P8 | 3 |
| C4 | P9, P10, P11 | 3 |

All four busy, perfectly balanced, parallelism = 4. If `orders` produces 48,000 messages per second and each message takes 1 ms to process, each consumer must handle 12,000 msg/s, which is 12,000 ms of work per wall-clock second per consumer — already over budget. So 4 consumers cannot keep up; you have headroom to add 8 more. Good, you have 12 partitions.

**Case B — 12 consumers (C = P).** Now each consumer owns exactly one partition:

| Consumers | Partitions owned each | Idle |
| --- | --- | --- |
| C1–C12 | one partition each (P0…P11) | 0 |

Parallelism = 12, the maximum. Each consumer handles 4,000 msg/s = 4,000 ms of work per second — still over the 1,000 ms-per-second budget of a single thread doing 1 ms of work per message. So even at the partition ceiling, a single thread per consumer cannot keep up at 1 ms/msg; you would need to make processing faster, batch it, or use more partitions. This is the moment the ceiling bites: you are out of consumers to add.

**Case C — 16 consumers (C > P).** You panic and add 4 more consumers, for 16 total against 12 partitions:

| Consumers | Partitions owned | Status |
| --- | --- | --- |
| C1–C12 | one partition each | active |
| C13–C16 | none | idle standby |

Parallelism is still 12. The four extra consumers consume *nothing*. Your lag does not improve by a single message. The only thing you bought is failover: if C7 crashes, C13 can take P6 on the next rebalance instead of you scrambling to launch a replacement. The lesson is blunt — **once C reaches P, the answer to "add more consumers" is "you cannot, repartition or speed up processing instead."** Many a 3 a.m. incident has been prolonged by an engineer adding consumers to a maxed-out group and watching lag refuse to move.

## 3. The group coordinator and membership

Someone has to decide who owns what, detect when a member dies, and drive the reassignment. That someone is the **group coordinator**, and it is a specific broker — not a separate service, not ZooKeeper, just one of your existing brokers wearing an extra hat for a particular group.

### How the coordinator is chosen

The coordinator for a group is determined by the group's identity, deterministically. Kafka hashes the `group.id`, mods it by the number of partitions in the internal `__consumer_offsets` topic (50 by default), and the leader of *that* partition of `__consumer_offsets` is the coordinator for the group. This is elegant: the same broker that stores a group's committed offsets also manages its membership. Offsets and coordination live together. When a consumer starts, it sends a `FindCoordinator` request to any broker, learns which broker is the coordinator for its `group.id`, and from then on talks to that broker for everything group-related: joining, heartbeating, committing offsets, leaving.

Because the coordinator is just the leader of a `__consumer_offsets` partition, it inherits Kafka's normal failover. If the broker hosting that partition dies, a follower is elected leader, and the new leader becomes the coordinator. Consumers detect the coordinator move (their requests start failing with a "not coordinator" error), re-run `FindCoordinator`, and reconnect. The committed offsets are already there because they were replicated — that is the payoff of co-locating coordination and offset storage on the same partition.

### Membership: the group's state machine

The coordinator maintains a state machine for the group. The states you will see in logs are roughly: **Empty** (no members), **PreparingRebalance** (members are being asked to rejoin), **CompletingRebalance** (waiting for the leader to deliver an assignment), and **Stable** (everyone has an assignment and is consuming). Each member is identified by a `member.id` the coordinator assigns on first join, and the coordinator tracks each member's subscription (which topics it wants), its session deadline (when it must heartbeat by), and, once stable, its assignment.

Crucially, **the coordinator does not compute the assignment itself.** That is a deliberate design choice that keeps brokers simple and lets clients innovate on assignment strategy. The coordinator runs the *protocol* — it collects members, picks one member as the **group leader**, forwards all members' subscriptions to that leader, receives the leader's computed assignment, and distributes each member's slice back out. The actual algorithm that decides "C1 gets P0–P2" runs on a client, the elected group leader, using a pluggable assignor (section 8). The broker is a referee, not a planner.

![A grid relating the consumer members to the group coordinator broker and the internal offsets partition it hosts, which replicates to followers and reloads on a coordinator move](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-9.webp)

The figure above shows the coordinator's two hats together: on the top row, the members talk to the coordinator broker for membership; on the bottom row, that same broker hosts the `__consumer_offsets` partition the group commits to, which replicates to followers and is reloaded if the coordinator ever moves to another broker. Keep that dual role in mind — it is why "the coordinator" appears in both the rebalance story and the offset story, and why a coordinator failover never loses your committed progress: the offsets were already replicated to the followers that the new coordinator is elected from.

### What the coordinator stores per member

For each member the coordinator holds a small but important bundle of state: the `member.id` it assigned, the `group.instance.id` if the member is static (section 9), the member's subscribed topics, its supported assignors, its current assignment once stable, and its session deadline — the wall-clock time by which the next heartbeat must arrive or the member is presumed dead. None of this is large; a group of a few hundred members is a few kilobytes of coordinator state. The coordinator also holds the group's generation number, an integer it increments on every rebalance. The generation is a fencing token: a member that was slow and missed a rebalance will try to commit or heartbeat with an *old* generation, the coordinator rejects it with an "illegal generation" error, and the stale member knows it must rejoin. That generation number is how Kafka prevents a zombie consumer — one that thinks it still owns partitions after a rebalance has moved them — from committing offsets over the top of the rightful new owner.

## 4. Offsets and the __consumer_offsets topic

A consumer's entire notion of progress is one number per owned partition: the offset of the next record it intends to read. Lose that number and the consumer either reprocesses everything from the start of retention or skips ahead and never sees the gap. So where Kafka stores that number, and how durably, is not a footnote — it is the difference between a clean restart and a data incident.

![A layered stack showing a polled batch flowing through processing into an offset commit that lands in the replicated and compacted internal offsets topic](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-6.webp)

### The current offset, the committed offset, and the gap between them

There are two offsets to keep straight, and conflating them is the source of most offset confusion. The **current position** (sometimes called the consumer position) is in-memory: it is the offset the consumer will fetch next, and it advances every time `poll()` returns records. The **committed offset** is durable: it is the position the consumer has told Kafka to remember, the place a *replacement* consumer will resume from after a rebalance or restart. The whole game of offset management is controlling the gap between these two — how far ahead your in-memory position runs before you persist it — because that gap is precisely the set of records that get reprocessed on a crash.

### __consumer_offsets is a compacted topic, not a table

Here is the part that surprises people: Kafka stores committed offsets *in Kafka*. There is an internal topic named `__consumer_offsets`, created automatically, with 50 partitions by default and a replication factor matching your cluster's `offsets.topic.replication.factor`. When your consumer commits, it does not write to a database; it produces a tiny record to `__consumer_offsets`. The record's key is `(group.id, topic, partition)` and its value is the committed offset plus some metadata. The coordinator for the group is the leader of the relevant `__consumer_offsets` partition, so the commit goes straight to the broker the consumer is already talking to.

The topic is **log-compacted**, which is the perfect storage model for this data. Compaction means Kafka retains only the *latest* value for each key and garbage-collects older values in the background. You do not care that partition P5 of group `billing` was once at offset 1,000 and then 2,000; you only care that it is now at 4,210. Compaction keeps exactly that — the most recent offset per `(group, topic, partition)` key — so the topic stays small and bounded no matter how many billions of commits you have made over the years. It is, in effect, a durable, replicated, infinitely-updatable key-value store built out of a Kafka topic, which is a beautiful piece of dogfooding.

The figure above traces the path top to bottom: `poll()` returns a batch up to some offset, your code processes it and does its real work (writes to a database, calls an API), then you commit the new offset, the commit lands as a record in the compacted `__consumer_offsets` topic, and that record is replicated to follower brokers. Only after that last replication step is your progress durable — a crash before it is a crash that loses the commit. That layering is why "commit" and "durable" are not the same instant, and why the order in which you process versus commit (section 5) decides whether you risk duplicates or loss.

### Inspecting offsets in practice

You can read this topic and the lag it implies with standard tooling. Lag — the gap between the latest produced offset (the log-end offset) and the group's committed offset — is *the* health metric for a consumer group.

```bash
# Show every member, its partitions, current offset, log-end offset, and LAG
kafka-consumer-groups.sh --bootstrap-server broker1:9092 \
  --describe --group billing-service

# Example output (trimmed):
# TOPIC     PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG   CONSUMER-ID
# payments  0          4210            4210            0     consumer-1-a1b2
# payments  1          9980            10240           260   consumer-1-a1b2
# payments  2          3001            3001            0     consumer-2-c3d4
```

A non-zero, growing `LAG` means consumers are falling behind producers. A flat lag means they are keeping pace. A lag that suddenly jumps to a huge number, then drops to zero, often means a consumer reset to `earliest` and re-read everything — usually a symptom of a lost commit, which is exactly the failure mode section 5 is about to dissect.

### Lag is a rate problem, not a depth problem

The instinct on seeing lag is to read the absolute number — "we are 260 messages behind" — and panic or relax based on its size. That is the wrong reading. **What matters about lag is its derivative, not its value.** A consumer steadily 50,000 messages behind but holding that number is keeping pace with production; it is processing exactly as fast as messages arrive, just with a fixed buffer of delay. A consumer only 2,000 messages behind but where that number climbs by 500 every minute is in trouble — it is falling behind at 500 msg/min and will be 30,000 behind in an hour and at the retention cliff by tomorrow. Alert on the *trend* (lag increasing for N consecutive minutes) far more than on the absolute level, and translate lag into time-to-drain (`lag ÷ consumer_throughput`) because "we are 12 minutes behind" is a decision-grade number in a way that "we are 50,000 messages behind" is not. Tools like Burrow exist precisely to evaluate lag as a sliding-window trend rather than a single threshold, and that is the right model.

### Reading and seeking offsets programmatically

Beyond the CLI, the consumer client lets you inspect and manipulate position directly, which is essential for replays and for tooling. You can ask for the committed offset of a partition, the current position, and the beginning and end offsets, and you can `seek()` to an arbitrary offset to replay or skip:

```python
from kafka import TopicPartition

tp = TopicPartition("payments", 0)
consumer.assign([tp])

committed = consumer.committed(tp)          # last durably committed offset
position  = consumer.position(tp)           # in-memory next-fetch position
end       = consumer.end_offsets([tp])[tp]  # log-end offset (latest produced)

# Replay the last hour: seek back, then poll from there.
consumer.seek(tp, committed - 60_000)       # rewind 60k records (rough "last hour")
```

This is how a controlled replay works: stop the consumer, `seek()` to an earlier offset (or look one up by timestamp with `offsetsForTimes`), and restart processing from there. It is also how you *skip* a poison record that crashes your processor — seek one past it and commit, deliberately abandoning that one message. Both operations are just moving the single number that is your position, which is the whole elegance of an offset-addressed log.

### auto.offset.reset: where you start with no committed offset

One config decides what happens when a consumer has *no* committed offset for a partition — a brand-new group, or an old offset that has aged out of retention. That is `auto.offset.reset`:

- `earliest`: start from the oldest retained record. The consumer reads the entire backlog. Choose this when you must not miss data and reprocessing is safe (idempotent consumers, analytics replays).
- `latest`: start from the next record produced after the consumer joins, skipping all history. Choose this when you only care about live data and old records are irrelevant.
- `none`: throw an exception and refuse to start. Choose this when "no committed offset" is itself a bug you want to be loudly told about.

This setting only fires when there is no valid committed offset. Once a group has committed, the committed offset always wins; `auto.offset.reset` is the cold-start fallback, not the everyday resume logic. A classic incident is a consumer group that was offline long enough for its committed offset to fall off the retention cliff (Kafka deleted the segment that offset pointed into); on restart the offset is invalid, `auto.offset.reset` kicks in, and a `latest` setting silently skips the gap while an `earliest` setting reprocesses everything from the retention floor. Neither is what you wanted; both look like a data bug downstream.

## 5. Auto-commit vs manual commit (and their failure modes)

This is the section that separates people who *use* Kafka from people who *understand* it, because the difference between auto-commit and manual commit is the difference between "I get duplicates I don't understand" and "I chose exactly which records I am willing to reprocess." There is no commit strategy that is free of both duplicates and loss; you are choosing which one you can tolerate.

### Auto-commit: convenient, and a duplicate factory

With `enable.auto.commit=true` (the default), the consumer commits offsets *for you* on a timer. The timer is `auto.commit.interval.ms`, default **5000 ms**. The commit is not actually a separate thread; it happens inside `poll()`. Each time you call `poll()`, the client checks whether 5 seconds have elapsed since the last auto-commit and, if so, commits the offsets of everything the *previous* `poll()` returned. That last clause is the crucial one: **auto-commit commits the offsets of records that were returned by `poll()`, on the assumption that returning them means you finished processing them.** It commits on a clock, not on completion of your work.

This is convenient and, for many workloads, fine. But it builds in two failure windows, and which one you hit depends on the order in which the crash, the processing, and the 5-second tick interleave.

**Duplicate window (the common one).** Suppose you `poll()` and get records up to offset 1,000. You start processing. At the next `poll()` the auto-committer might commit offset 1,000. But if you crash *after processing but before the next poll commits*, those processed records were never committed, so the replacement consumer re-reads from the last committed offset and reprocesses them. Up to a full 5-second batch can be reprocessed. With at-least-once semantics this is expected, but engineers are routinely surprised by *how many* duplicates a single crash produces, because they did not internalize the 5-second window.

**Loss window (the dangerous one).** Now suppose you do your processing *asynchronously* or in a way where `poll()` returns control before your work finishes. `poll()` returns records up to 1,000, auto-commit later commits 1,000, but your background processing of records 900–1,000 has not finished when the process crashes. Those records are now committed (Kafka thinks you are done with them) but were never actually processed. The replacement consumer starts at 1,000 and **the work for 900–1,000 is silently lost.** This is the auto-commit loss trap, and it is why auto-commit plus async processing is a footgun. Auto-commit is only safe when "returned by poll()" reliably implies "fully processed by the time the next poll() runs," which means: do all your processing synchronously, inside the poll loop, between successive `poll()` calls.

#### Worked example: a crash 3 seconds after a 5-second auto-commit

Let me nail down the duplicate window with a clock. Assume `enable.auto.commit=true`, `auto.commit.interval.ms=5000`, and a consumer processing a steady 200 messages per second, processed synchronously inside the poll loop.

- **T = 0.0 s.** An auto-commit fires inside `poll()`. The committed offset is, say, **10,000**. Everything up to 10,000 is durably recorded.
- **T = 0.0 → 5.0 s.** The consumer keeps polling and processing. At 200 msg/s it processes 1,000 more messages, advancing its *in-memory* position to **11,000**. But no auto-commit fires in this window — the next tick is due at T = 5.0 s. So the committed offset is still **10,000** the whole time.
- **T = 3.0 s.** The process crashes. At 200 msg/s, it had processed 600 messages since the last commit, reaching in-memory offset **10,600**. Those 600 messages (offsets 10,000–10,599) were *processed* but their offsets were *never committed*.
- **Recovery.** A rebalance assigns the partition to a surviving consumer (or the restarted one). It reads the committed offset: **10,000**. It resumes there. It reprocesses **600 messages** — exactly the records processed in the 3 seconds since the last commit.

So a crash 3 seconds into a 5-second interval reprocesses 3 seconds' worth of work: 600 messages here. A crash at 4.9 seconds would reprocess ~980. The reprocessed count is `throughput × (time since last commit)`, bounded by `throughput × auto.commit.interval.ms`. Shrinking the interval shrinks the duplicate window but increases commit traffic to `__consumer_offsets`. There is no free lunch; you are trading duplicate volume against commit overhead. And note the critical assumption that made this *only* a duplicate problem and not a loss problem: processing was synchronous inside the loop, so every reprocessed message is a *re-do*, never a *skip*. Break that assumption (async processing) and the same clock produces silent loss instead.

### Manual commit: commitSync and commitAsync

To control the window precisely, set `enable.auto.commit=false` and commit yourself, *after* your processing succeeds. Now the order is **process, then commit**, which guarantees at-least-once: a crash between processing and committing causes reprocessing (a duplicate), never loss, because you never commit work you have not finished.

```java
props.put("enable.auto.commit", "false");
KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(List.of("payments"));

while (running) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(200));
    for (ConsumerRecord<String, String> record : records) {
        handlePayment(record);          // do the real work FIRST
    }
    // commit only after the whole batch is processed
    consumer.commitSync();              // blocks until the broker acks the commit
}
```

`commitSync()` blocks until the coordinator acknowledges the commit (with retries on retriable errors). It is the safe, simple choice: when it returns, your offset is durable. The cost is latency — you pause the poll loop on every commit waiting for a broker round trip. For a batch processed in tens of milliseconds, a few-millisecond commit round trip per batch is real overhead.

`commitAsync()` fires the commit and returns immediately, taking a callback for the result. It does not block the loop, so throughput is higher, but it has two sharp edges. First, **a failed async commit is not retried** by default (retrying a stale async commit could overwrite a newer successful one), so a transient failure can leave you with a slightly older committed offset than you think — which on a crash means a few extra duplicates. Second, on shutdown you can lose the last async commit if the process exits before it completes. The standard production pattern uses both: `commitAsync()` in the hot loop for speed, and a final `commitSync()` in the `finally` block on shutdown to guarantee the last offset is durable.

```java
try {
    while (running) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(200));
        for (ConsumerRecord<String, String> record : records) {
            handlePayment(record);
        }
        consumer.commitAsync();         // fast path: non-blocking, best-effort
    }
} finally {
    try {
        consumer.commitSync();          // shutdown path: block to guarantee the last commit
    } finally {
        consumer.close();
    }
}
```

### Committing per record, per batch, and exactly-once

You can commit at finer or coarser granularity by passing explicit offsets to `commitSync(Map<TopicPartition, OffsetAndMetadata>)`. Commit after every record and you minimize the duplicate window to a single record, at the cost of a commit round trip per record — usually too slow. Commit after every batch (the loop above) is the standard balance. And if you genuinely need *no* duplicates and *no* loss — exactly-once — you stop relying on offset commits alone and use Kafka transactions, which atomically commit your output records *and* your input offsets together so that consumption and production are one all-or-nothing unit. That is its own deep topic; the [delivery semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) covers the exactly-once machinery in full. For this post, the takeaway is: auto-commit is at-least-once with a 5-second-wide duplicate window and a loss trap if you process asynchronously; manual process-then-commit is at-least-once with a window you control; transactions are exactly-once at the cost of throughput and complexity.

| Strategy | Window on crash | Risk | When to use |
| --- | --- | --- | --- |
| Auto-commit, sync processing | up to 5 s of records | duplicates | low-stakes, idempotent work |
| Auto-commit, async processing | up to 5 s of records | **silent loss** | avoid — footgun |
| `commitSync` after batch | one batch | duplicates | default for at-least-once |
| `commitSync` after each record | one record | duplicates (rare) | strict, low-throughput |
| Transactions (EOS) | none | none (slower) | financial, exactly-once |

The single most important rule in this entire section: **if you use auto-commit, do all your processing synchronously inside the poll loop, or you risk silent loss.** If you cannot guarantee synchronous processing, turn auto-commit off and commit after your work. This decision deserves its own treatment, which the forthcoming sibling post on [offset commit strategies](/blog/software-development/message-queue/consumer-offset-commit-strategies-failure-modes) will give it; here the goal is that you never again get surprised by a duplicate count or a missing record and not know which window produced it.

## 6. The rebalance protocol: heartbeats, timeouts, join/sync

Now the famous part. A **rebalance** is the process of reassigning partitions among the members of a group. It is how the group adapts to change: a consumer joins, a consumer leaves, a consumer dies, or the topic gains partitions. The rebalance is also where most consumer-group pain lives, because while it runs, an eagerly-rebalancing group stops consuming. Understanding the protocol — what detects the need for a rebalance and what the handshake does — is what lets you tune a group instead of cargo-culting configs.

![A pipeline of the rebalance handshake moving from heartbeat to JoinGroup to the leader planning the assignment to SyncGroup to consuming records](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-2.webp)

### Heartbeats and the two timeouts that detect failure

A consumer proves it is alive by sending **heartbeats** to the coordinator on a background thread, every `heartbeat.interval.ms` (default 3 seconds). The coordinator expects to hear from each member within `session.timeout.ms` (default 45 seconds in modern Kafka, historically 10 seconds). If `session.timeout.ms` passes with no heartbeat, the coordinator declares that member dead, removes it from the group, and triggers a rebalance to redistribute its partitions. The rule of thumb is `heartbeat.interval.ms` should be roughly one-third of `session.timeout.ms`, so a member gets a few missed heartbeats of slack before it is evicted. A short session timeout detects real failures fast but evicts members over transient network blips; a long one tolerates blips but leaves dead members' partitions unconsumed for longer.

There is a second, sneakier timeout: `max.poll.interval.ms` (default **5 minutes**). Heartbeats run on a background thread, so a consumer can keep its session alive even while its *main* thread is stuck processing a slow batch and not calling `poll()`. That is a problem — a consumer that heartbeats but never polls is consuming nothing while holding partitions hostage. So Kafka adds a separate liveness check: if the time between two `poll()` calls exceeds `max.poll.interval.ms`, the consumer voluntarily *leaves* the group and triggers a rebalance, on the theory that a consumer not polling for five minutes is effectively stuck. This is the timeout that bites batch processors and anyone doing heavy per-record work: process a batch slower than `max.poll.interval.ms` and you get kicked out mid-batch, the partitions move to someone else, and when your slow batch finally finishes and you try to commit, you discover you no longer own the partition. The fixes are to lower `max.poll.records` (fetch fewer records per poll so a batch processes faster) or raise `max.poll.interval.ms` (give yourself more time), and which one is right depends on whether your per-record work or your batch size is the real cause.

| Config | Default | What it bounds | Symptom if too low |
| --- | --- | --- | --- |
| `heartbeat.interval.ms` | 3 s | gap between heartbeats | extra coordinator traffic |
| `session.timeout.ms` | 45 s | silence before "member dead" | eviction on network blips |
| `max.poll.interval.ms` | 5 min | gap between poll() calls | eviction on slow batches |
| `max.poll.records` | 500 | records returned per poll() | n/a (raise to fetch more) |

### The JoinGroup → SyncGroup handshake

When a rebalance is triggered, the protocol runs in two coordinator round trips, shown in the pipeline figure above. The way this works is a careful separation between the broker running the protocol and a client computing the assignment.

**Phase 1 — JoinGroup.** Every member sends a `JoinGroup` request to the coordinator, declaring its subscription (the topics it wants and which assignors it supports). The coordinator waits until it has collected JoinGroup requests from all known members (or the rebalance timeout, which equals `max.poll.interval.ms`, expires). It then picks one member as the **group leader** — typically the first to join — and responds to every member. The leader's response is special: it contains the full membership list and every member's subscription. The other members get a response that just says "you are a follower, wait."

**Phase 2 — SyncGroup.** The leader now runs the assignor locally and computes the complete assignment: which member owns which partitions. It packages this and sends it to the coordinator in a `SyncGroup` request. Every other member also sends a `SyncGroup`, but empty — they are just asking "what did I get?" The coordinator takes the leader's plan, splits it per member, and responds to each `SyncGroup` with *that member's* slice of the assignment. Each member now knows its partitions, fetches the committed offsets for them, seeks to those offsets, and starts consuming. The group is `Stable` again.

The figure traces exactly this: heartbeat establishes the coordinator, JoinGroup elects the leader and gathers subscriptions, the leader plans the assignment, SyncGroup distributes each member's share, and finally `poll()` returns records. The beauty of this design is that the broker never needs to understand your assignment strategy — it shuttles opaque bytes between members and lets a client do the thinking, which is why you can drop in a custom assignor without touching the broker.

### What "stop the world" really means

During an *eager* rebalance (the historical default), every member **revokes all of its partitions** at the start of the rebalance, before JoinGroup. The group consumes nothing from the moment revocation begins until SyncGroup completes and members re-acquire partitions. For a small group with a fast assignor this might be tens of milliseconds; for a large group, a slow assignor, or a member that is slow to rejoin, it can be seconds. Multiply that by how often your group rebalances and you have the operational cost. Section 7 is the fix.

![A timeline of a rebalance triggered by a fourth consumer joining, showing the group going from stable to a global pause to a new assignment and back to consuming](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-4.webp)

The timeline above walks one concrete eager rebalance second by second. At T+0 the group is stable with three consumers, each consuming its partitions. At T+1 a fourth consumer joins and sends its JoinGroup. Because this is an eager rebalance, *all* members immediately revoke every partition and the group goes idle — no consumption anywhere, even on partitions that will not move. At T+3 the group leader has run the assignor and computed the new four-way split; at T+4 SyncGroup distributes each member's new partitions; and by T+5 all four consumers are consuming again. The shaded middle — roughly T+1 to T+5 — is the stop-the-world window, four seconds during which the group processed nothing. Notice what the timeline makes painfully clear: the three *original* consumers, who were doing fine, were paused for the entire window even though most of their partitions stayed with them. That waste is precisely what the cooperative protocol in the next section eliminates.

### The rebalance listener: your hook into the handshake

You are not a passive bystander in a rebalance. The consumer API gives you a `ConsumerRebalanceListener` with two callbacks: `onPartitionsRevoked` (called just before partitions are taken away) and `onPartitionsAssigned` (called just after new partitions arrive). The revoked callback is your last chance to **commit offsets for partitions you are about to lose** — if you do not, the new owner resumes from a stale committed offset and reprocesses more than necessary. This is the single most important thing the listener is for, and forgetting it is a classic cause of "why did we reprocess so much after a deploy."

```java
consumer.subscribe(List.of("payments"), new ConsumerRebalanceListener() {
    public void onPartitionsRevoked(Collection<TopicPartition> revoked) {
        // last chance: persist progress for partitions we are losing
        consumer.commitSync();
    }
    public void onPartitionsAssigned(Collection<TopicPartition> assigned) {
        // new partitions arrived; (re)build any per-partition local state here
    }
});
```

Under cooperative rebalancing there is also `onPartitionsLost`, called when partitions were taken away *without* a clean revoke (for example after the consumer was fenced for being too slow). You must not commit in `onPartitionsLost` — you no longer own those partitions and a commit would be rejected or, worse, clobber the new owner's progress. The distinction between *revoked* (graceful, commit now) and *lost* (fenced, do not commit) is subtle and exactly the kind of thing that separates a correct consumer from one that quietly corrupts offsets.

## 7. Eager vs cooperative incremental rebalancing

The single biggest improvement to the consumer group protocol in Kafka's history is **cooperative incremental rebalancing**, and understanding the before-and-after is understanding why your rebalances may or may not hurt.

![A two-panel comparison contrasting eager rebalancing where every member revokes every partition against cooperative rebalancing where only partitions that change owner are revoked](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-3.webp)

### Eager: revoke everything, then reassign

The eager protocol is simple and brutal. When a rebalance starts, **every member revokes every partition it owns**, sends an empty subscription's worth of "I have nothing now," and then JoinGroup/SyncGroup runs and hands everyone a fresh assignment. The problem is in the word *every*. Suppose you have 12 partitions across 4 consumers and a 5th consumer joins. The correct outcome is that a few partitions move to the new consumer and most stay put. But eager rebalancing does not know that until the assignment is computed — and to compute it safely, it first makes *everyone* give up *everything*, because if it let members keep partitions during the rebalance, two members might briefly believe they own the same partition, violating the one-owner invariant. So eager trades correctness for a stop-the-world pause: during the rebalance, the entire group consumes nothing, even the partitions that were never going to move.

The left panel of the figure makes the cost vivid: all members revoke, the group goes idle with no consumption, and then everything is reassigned from scratch. For a group that rebalances rarely this is acceptable. For a group that rebalances often — autoscaling, frequent deploys, flaky network — the cumulative stop-the-world time is a real availability problem, and it gets worse as the group grows because more partitions stop at once.

### Cooperative: revoke only what moves

Cooperative incremental rebalancing (the `cooperative-sticky` assignor, default in modern clients) fixes this by splitting the rebalance into two passes and revoking *only the partitions that change owner*. The way this works: in the first rebalance pass, the assignor computes the new desired assignment and compares it to the current one. Members keep every partition they currently own and are *still* supposed to own — those never get revoked, so they never stop consuming. Only partitions that need to move to a different member are revoked, in a first pass. Then a *second* rebalance assigns those now-free partitions to their new owners. The result, shown in the right panel: members keep their current assignment, only the genuinely-moving partitions are revoked, and the large majority of the group keeps consuming throughout.

Concretely: 12 partitions, 4 consumers (3 each), a 5th joins. The target is roughly 2–3 partitions per consumer. Cooperative rebalancing revokes only ~3 partitions (the ones that will move to the new member), leaves the other ~9 consuming uninterrupted, and assigns the 3 freed partitions to the newcomer in a second pass. Compared to eager — which would have stopped all 12 — you stopped 3. That is a 75% reduction in disrupted partitions for this case, and the advantage grows with group size.

The cost is that cooperative rebalancing takes *two* rebalance passes instead of one, so the total protocol runs a bit longer in wall-clock terms, but it does so without a global pause. You trade a longer-but-non-blocking process for a shorter-but-blocking one, and for almost every production group the non-blocking version wins. The migration from eager to cooperative is itself careful (you cannot flip both sides at once on a running group), but a fresh group on a modern client gets `cooperative-sticky` by default, and that is the right default. The forthcoming sibling post on [rebalance storms](/blog/software-development/message-queue/kafka-rebalance-storms-and-how-to-tame-them) goes deep on what happens when rebalances trigger *each other* into a thrash loop; this post's job is to make sure you understand the single-rebalance mechanics first.

### The taxonomy of triggers

It helps to have a clean mental catalog of what can *start* a rebalance, because every one of these is a partition-ownership change in disguise.

![A tree classifying rebalance triggers into membership changes like join leave and timeout and metadata changes like adding partitions to a topic](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-7.webp)

The figure splits triggers into two roots. **Membership changes** are the common ones: a consumer **joins** (you scaled up or deployed a new instance), or a consumer **leaves** — either gracefully (it called `close()` and sent a `LeaveGroup`) or ungracefully (it crashed and tripped `session.timeout.ms`, or it stalled and tripped `max.poll.interval.ms`). **Metadata changes** are rarer: the subscribed topic gains partitions (someone ran a partition-increase), so the group must assign the new partitions to someone. Every rebalance you ever debug is one of these. When you read "rebalance" in a log, your first question is always *which trigger* — a deploy (join/leave, expected), a crash (timeout, investigate), or a partition change (rare, check who ran it). Knowing the trigger taxonomy turns a scary log line into a quick diagnosis.

## 8. Partition assignors (range / round-robin / sticky / cooperative-sticky)

The assignor is the algorithm the group leader runs to map partitions to members. It is configured with `partition.assignment.strategy`, and the choice affects balance, how many partitions move during a rebalance, and whether the rebalance is eager or cooperative. There are four you need to know.

![A matrix scoring range round-robin sticky and cooperative-sticky assignors on even balance stickiness low movement and incremental support](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-5.webp)

### Range

`RangeAssignor` assigns *per topic*. For each topic, it lays out the partitions in order, lays out the consumers in order, and gives each consumer a contiguous *range* of partitions. With 12 partitions and 5 consumers, the first two consumers get 3 partitions each and the last three get 2 each (12 = 3+3+2+2+2). The problem with range is **imbalance when partitions do not divide evenly, and it gets worse with multiple topics**: because it assigns per topic, the *same* low-numbered consumers tend to get the extra partition on *every* topic, so consumer 1 ends up with the leftover from topic A *and* topic B *and* topic C, piling load onto the first few members. Range's one virtue is co-location: a consumer that owns partition P0 of topic A and partition P0 of topic B can join them locally, which is occasionally useful. Otherwise its skew makes it a poor default for multi-topic groups.

### Round-robin

`RoundRobinAssignor` assigns *across all topics and partitions at once*. It lays out every partition of every subscribed topic in one list and deals them to consumers round-robin, like dealing cards. This produces a **near-perfectly balanced** assignment — partition counts differ by at most one across members — which fixes range's skew. Its weakness is **stickiness, or the lack of it**: round-robin recomputes from scratch every rebalance, so a single member joining can reshuffle *almost every* partition to a new owner, even partitions that did not need to move. Balanced but disruptive.

### Sticky

`StickyAssignor` keeps round-robin's balance but adds **stickiness**: when it must rebalance, it tries to preserve the existing assignment as much as possible, moving the *minimum* number of partitions needed to rebalance and keep things balanced. A member joining takes a few partitions from the most-loaded members rather than triggering a full reshuffle. This dramatically reduces partition movement, which matters because every moved partition means a consumer seeks to a committed offset, possibly rebuilds local state, and reprocesses anything between its in-memory position and the committed offset. Sticky is balanced *and* stable. Its one limitation: classic sticky is still an *eager* protocol — it revokes everything first, then reassigns, even though it reassigns most partitions back to their original owners. The stickiness saves state-rebuild cost but not the stop-the-world pause.

### Cooperative-sticky

`CooperativeStickyAssignor` is sticky's balance and stability *plus* the cooperative incremental protocol from section 7. It moves the minimum number of partitions **and** does so without a global revoke — only the partitions that actually change owner are revoked, in a first pass, while everything else keeps consuming. It is the modern default in the Java client, and for almost every group it is the right choice: best balance, best stickiness, least movement, no stop-the-world pause. The figure's bottom rows make the comparison concrete: only sticky and cooperative-sticky achieve low movement, and only cooperative-sticky is incremental. If you take one config recommendation from this post, it is: use `cooperative-sticky` unless you have a specific reason not to.

| Assignor | Balance | Stickiness | Protocol | Pick when |
| --- | --- | --- | --- | --- |
| Range | uneven (multi-topic skew) | none | eager | co-locating same-partition across topics |
| Round-robin | even | none | eager | single topic, rebalances are rare |
| Sticky | even | high | eager | need minimal movement, stuck on old protocol |
| Cooperative-sticky | even | high | **cooperative** | default — almost always |

A subtle but important operational note: you cannot mix incompatible assignors in one group, and migrating a running group from an eager assignor to cooperative-sticky is a two-step rolling change (first deploy with *both* old and cooperative-sticky listed so the group can negotiate, then deploy with only cooperative-sticky). Flip it in one step on a live group and you can break the rebalance. For a new group, just set `cooperative-sticky` and move on.

### How the leader actually negotiates the assignor

A detail that confuses people: each consumer advertises a *list* of assignors it supports, in priority order, and the group must agree on one. During JoinGroup every member sends its supported assignors; the elected group leader picks the highest-priority assignor that *every* member supports. This is why the two-step migration works — in the intermediate deploy, some members support only the old eager assignor and some support both, so the only common choice is the old one, and the group stays eager and healthy. Once *every* member has been rolled to a build that lists cooperative-sticky, the leader can finally select it, and a single rebalance flips the whole group to the cooperative protocol. Trying to skip the intermediate step means a member that supports only cooperative-sticky and a member that supports only eager have *no* common assignor, and the group cannot form. The negotiation is also why a custom assignor must be deployed to every consumer before any consumer can rely on it.

### The cost of partition movement, quantified

Why obsess over minimizing partition movement? Because every partition that changes owner pays a tax. The losing consumer must commit (or lose) its progress; the gaining consumer must fetch the committed offset, `seek()` to it, possibly tear down and rebuild any per-partition local state (a windowed aggregation, a join buffer, a cache keyed by partition), and refill its fetch buffers from cold. For a stateless consumer this is milliseconds. For a stateful stream processor holding gigabytes of per-partition RocksDB state, a moved partition can mean *minutes* of state restoration before that partition produces output again. So an assignor that moves 3 partitions instead of 30 is not a 10x improvement in rebalance speed in the abstract — it can be the difference between a 2-second blip and a 20-minute recovery on a stateful application. This is the real-world reason sticky and cooperative-sticky matter so much for Kafka Streams and similar frameworks, and why their defaults lean that way.

## 9. Static membership and avoiding restart rebalances

There is one more configuration that eliminates a whole category of needless rebalances, and it is criminally underused: **static membership**, via `group.instance.id`.

![A two-panel comparison showing a restart without static membership causing two full rebalances versus a restart with a stable instance id causing none](/imgs/blogs/kafka-consumer-groups-offsets-rebalancing-8.webp)

### The rolling-restart problem

Consider a routine deploy: you have a 10-member consumer group and you do a rolling restart, replacing instances one at a time. Without static membership, each restart is *two* rebalances. When you stop instance 1, it sends a `LeaveGroup` (or trips the session timeout), the coordinator removes it, and the group rebalances to redistribute its partitions across the remaining 9 members. Then instance 1 comes back up as a *brand-new member* with a *fresh* `member.id`, joins the group, and the group rebalances *again* to give it partitions back. Two rebalances per instance, 20 rebalances for a 10-member rolling deploy, each one shuffling partitions and (under eager) pausing consumption. The left panel of the figure shows it: restart drops the member, a full rebalance fires on leave, and a second full rebalance fires on rejoin. For a group that deploys several times a day, this is a steady drumbeat of self-inflicted disruption.

### How static membership fixes it

Static membership assigns each consumer instance a **stable, operator-chosen identity** via `group.instance.id` — typically derived from the pod name, host, or ordinal index, something that stays the same across restarts of *that* instance. When a static member disconnects (a restart), the coordinator does **not** immediately remove it or trigger a rebalance. Instead it remembers the member's identity and its assignment for the duration of `session.timeout.ms`, treating the disconnect as a temporary blip. If the same `group.instance.id` rejoins within the session timeout, the coordinator simply hands it back the *exact same partitions* it had before — no rebalance, no reassignment, no pause. The right panel: the restart keeps the stable id, the instance rejoins under the session timeout, and it reclaims the same partitions with zero rebalances.

```properties
# Each instance gets a stable identity that survives restarts.
# In Kubernetes, this is naturally the StatefulSet pod ordinal.
group.instance.id=billing-consumer-3
group.id=billing-service
# Make the session timeout comfortably longer than your restart time,
# so a rolling restart of one pod finishes well inside the window.
session.timeout.ms=120000
```

The key tuning interaction is that **`session.timeout.ms` must be longer than the time it takes an instance to restart and rejoin.** If your pod takes 30 seconds to restart, a 45-second session timeout works but is tight; many shops raise it (to a minute or two) for static-membership groups specifically so a slow restart never slips past the window and falls back into a full rebalance. The trade is the usual one: a longer session timeout means a genuinely-dead member's partitions stay unconsumed longer before the group gives up on it. But for a planned restart that completes in seconds, the window is comfortable and you eliminate the rebalances entirely.

#### Worked example: a 10-member rolling restart, with and without static membership

Put numbers on it. A 10-member group, eager rebalancing, each rebalance pausing consumption for ~2 seconds, deploying via a rolling restart of all 10 pods one at a time, each pod taking 20 seconds to restart.

- **Without static membership:** each pod restart = 2 rebalances (leave + rejoin) = ~4 seconds of paused consumption. Ten pods = **20 rebalances**, roughly **40 seconds** of cumulative consumption pause across the deploy, plus a flurry of partition movement that means consumers reseek and reprocess the duplicate windows from section 5. A daily deploy means 20 self-inflicted rebalances a day.
- **With static membership and `session.timeout.ms=120000`:** each pod restarts in 20 seconds, well inside the 120-second window, so the coordinator holds its partitions and waits. The same instance rejoins, reclaims its exact partitions, and **no rebalance fires.** Ten pods = **0 rebalances**, **0 seconds** of paused consumption, no partition movement, no extra duplicates. The deploy is invisible to the group.

That is the whole pitch: static membership turns a deploy from a 20-rebalance event into a non-event, for the cost of one config line and a longer session timeout. There is no reason a stable, statefully-deployed consumer group should not use it. The one caveat: static membership delays detection of a *genuinely* dead member by the full session timeout, so if a pod truly dies (not a restart), its partitions sit unconsumed for up to `session.timeout.ms` before the group reassigns them. For most groups that delay on real failures is a fair price for eliminating the rebalances on routine restarts.

## Case studies and war stories

Patterns are easier to remember when they are attached to a scar. Here are four, each illustrating one of this post's concepts in production.

### The maxed-out group that would not catch up

A payments team woke to a lag alert: their `transactions` consumer group was 4 million messages behind and climbing. The on-call engineer did the obvious thing — scaled the consumer deployment from 8 pods to 24 — and watched lag refuse to budge. The topic had **8 partitions**. With `C > P`, 16 of the 24 pods were idle standbys consuming nothing; the original 8 were already maxed and the 16 new ones added zero throughput. The real fix was not more consumers but more partitions (a disruptive online repartition) plus faster per-record processing. **Lesson:** the partition count is the throughput ceiling. When lag will not move after scaling consumers, check `C` against `P` *first* — you may have hit the wall described in section 2.

### The 5-second duplicate mystery

An analytics service reported that roughly every database row it wrote from Kafka appeared two or three times after any pod restart, but only sometimes. The team had `enable.auto.commit=true` with the default 5-second interval and processed records synchronously. The duplicates were exactly the section-5 window: any crash reprocessed up to 5 seconds of records since the last auto-commit, and a deploy that bounced several pods produced several such windows. The "fix" was to make the database writes idempotent (upsert on a natural key) so reprocessing was harmless, and to switch to manual `commitSync` after each batch to shrink and control the window. **Lesson:** auto-commit is at-least-once with a 5-second-wide duplicate window. Either make consumers idempotent or commit manually after processing; do not be surprised by duplicates you architected in.

### The slow batch that kept getting evicted

A machine-learning feature pipeline pulled 500 records per poll, ran each through a model taking ~700 ms, and kept getting kicked out of its group mid-batch with a "member left the group" log. 500 records × 700 ms = 350 seconds per batch, far past the default `max.poll.interval.ms` of 5 minutes (300 s). The consumer heartbeated fine on its background thread but never returned to `poll()` in time, so Kafka evicted it, rebalanced its partitions away, and the consumer's eventual commit failed because it no longer owned the partition — so the batch reprocessed, slowly, forever. The fix was to drop `max.poll.records` to 100 (100 × 700 ms = 70 s, comfortably under the limit) and raise `max.poll.interval.ms` to 600 s for headroom. **Lesson:** `max.poll.interval.ms` bounds the gap between polls, not the heartbeat. Slow per-record work plus a large batch will evict you even while heartbeats are healthy; shrink the batch or raise the interval.

### The deploy that rebalanced twenty times a day

A streaming team's group of 12 consumers deployed 3–4 times a day, and each deploy produced a burst of rebalances that paused consumption and spiked tail latency on a latency-sensitive downstream. Their group used the eager `RangeAssignor` and no static membership, so every rolling restart was the 2-rebalances-per-pod pattern from section 9, times 12 pods, times 4 deploys — well over 80 disruptive rebalances a day. They made two changes: switched to `cooperative-sticky` (so the rebalances that *did* happen no longer stopped the world) and added `group.instance.id` static membership with a 2-minute session timeout (so routine restarts stopped triggering rebalances at all). Rebalances on deploy dropped to essentially zero and tail latency flattened. **Lesson:** cooperative-sticky plus static membership is the one-two punch for a group that deploys often. The assignor change makes unavoidable rebalances cheap; static membership makes restart rebalances disappear.

## When to reach for this (and when not to)

Consumer groups are not optional in Kafka — if you consume a topic, you are in a group, even a group of one. So "when to use them" is really a set of decisions *within* the consumer-group model.

**Use multiple consumers in a group** whenever a single consumer cannot keep up with a topic's produce rate and reprocessing/ordering constraints allow parallelism. Size the group up toward the partition count, and remember the ceiling: never expect throughput past `C = P`. **Use a single consumer in a group** when strict total ordering across the whole topic matters more than throughput, or the volume is low — one consumer over one partition is the simplest correct thing and it is fine to ship.

**Use auto-commit** only for low-stakes, idempotent, synchronously-processed workloads where a few seconds of duplicates on a crash is harmless. **Use manual `commitSync` after the batch** as your default for anything that matters — it is the at-least-once workhorse with a window you control. **Reach for transactions (exactly-once)** only when duplicates are genuinely unacceptable (money movement, non-idempotent external effects) and you can pay the throughput and complexity cost; do not reach for them reflexively, because most problems are better solved by making consumers idempotent.

**Use `cooperative-sticky`** as your assignor default, full stop, for any modern client. **Stay on an eager assignor** only if you are pinned to an old client or have a specific co-location need that `RangeAssignor` uniquely serves. **Use static membership** for any group with a stable deployment topology that restarts on deploys (essentially every Kubernetes StatefulSet-backed consumer), tuning `session.timeout.ms` above your restart duration; **skip it** only for ephemeral, autoscaling consumers whose identities legitimately churn, where a stable instance id would be meaningless.

And the meta-rule: **tune timeouts to your reality, do not inherit defaults blindly.** `session.timeout.ms`, `max.poll.interval.ms`, `max.poll.records`, and `heartbeat.interval.ms` are a coupled system. Set the session timeout to your tolerance for undetected death, the poll interval above your worst-case batch time, the batch size so a batch finishes inside the poll interval, and the heartbeat to roughly a third of the session timeout. Defaults are a starting point, not a decision.

## Key takeaways

- **Within a group, every partition has exactly one owner.** That single invariant explains the scaling ceiling, the rebalance, per-partition ordering, and offset tracking. Internalize it and the rest follows.
- **Parallelism caps at the partition count.** `C < P` leaves headroom, `C = P` is maximum throughput, `C > P` just adds idle standbys. When lag will not move after scaling, check `C` against `P` before anything else.
- **The group coordinator is a broker, chosen by hashing `group.id` onto a `__consumer_offsets` partition.** It runs the membership protocol but does *not* compute the assignment — a client-side group leader does, using a pluggable assignor.
- **Committed offsets live in the compacted `__consumer_offsets` topic, not a database.** Compaction keeps only the latest offset per `(group, topic, partition)`, so the topic stays small forever. A commit is durable only after it replicates.
- **Auto-commit is at-least-once with a 5-second duplicate window — and a silent-loss trap if you process asynchronously.** Manual process-then-commit gives a window you control. Pick based on idempotency and stakes, never by accident.
- **Heartbeats and `session.timeout.ms` detect death; `max.poll.interval.ms` detects a stuck-but-heartbeating consumer.** Slow batches trip the poll interval even while heartbeats are healthy — shrink `max.poll.records` or raise the interval.
- **Eager rebalancing stops the world; cooperative incremental revokes only the partitions that move.** Use `cooperative-sticky` as the default assignor and most of the group keeps consuming through a rebalance.
- **Static membership (`group.instance.id`) turns a rolling-restart's many rebalances into zero,** as long as restarts finish inside `session.timeout.ms`. It is one config line and almost every stable group should use it.

## Further reading

- [Kafka Deep Dive, Part 1: Log segments, the page cache, and storage](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage) — the storage layer underneath the consumer-group machinery in this post.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the broader Kafka model; this post zooms into the consumer-group corner of its rebalance sketch.
- [Push vs pull, acknowledgements, and how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — why Kafka's pull model gives consumers free backpressure, the delivery side of this story.
- [Delivery semantics: at-most-once, at-least-once, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — where offset commit order turns into a delivery guarantee, including Kafka transactions.
- [Message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — why one-owner-per-partition is what gives you per-partition ordering.
- [Kafka offset commit strategies](/blog/software-development/message-queue/consumer-offset-commit-strategies-failure-modes) — the forthcoming sibling that drills into commit patterns, sync vs async, and exactly-once offset handling.
- [Kafka rebalance storms and how to tame them](/blog/software-development/message-queue/kafka-rebalance-storms-and-how-to-tame-them) — the forthcoming sibling on rebalances that trigger each other into a thrash loop, and how to stop them.
- [Apache Kafka documentation: Consumer Configs and the Group Membership protocol](https://kafka.apache.org/documentation/#consumerconfigs) — the authoritative reference for every config named in this post.

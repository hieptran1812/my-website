---
title: "Partition Count, Sizing, and Capacity Planning for Throughput"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The partition count is the single most consequential and hardest-to-change capacity decision in a streaming system. Learn why the partition is the unit of parallelism, how to size partitions from per-producer and per-consumer throughput, the concrete costs of too many and too few, why repartitioning silently breaks per-key ordering, and a worksheet you can run before you create a topic."
tags:
  [
    "message-queue",
    "partitioning",
    "capacity-planning",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "throughput",
    "scalability",
    "performance",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/partitioning-capacity-planning-1.webp"
---

There is one number you type into a `CreateTopics` request that will haunt you for the entire life of the topic, and almost nobody thinks about it for more than ten seconds. It is the partition count. People agonize for days over the broker, the replication factor, the retention, the serialization format — and then they reach the partition count, shrug, type `6` because it felt round, and move on. Six months later that `6` is the reason a service cannot scale past a few thousand messages a second on a Black Friday, the reason a backlog took nine hours to drain when it should have taken twenty minutes, and the reason a "simple" migration to fix it broke per-customer ordering across the entire platform and produced a week of mystifying data-corruption tickets.

The partition count is the most consequential capacity decision you make about a topic, and it is also the hardest to change after the fact. It is consequential because **the partition is the unit of parallelism**: a Kafka consumer group can never run more useful consumers than there are partitions, full stop. If you have six partitions, the seventh consumer you add does literally nothing — it sits idle, holds no partition, processes no messages, and burns money. Your maximum consumer parallelism is frozen at topic-creation time. And it is hard to change because growing the partition count remaps `hash(key) % N` for every key, which means a given user's events suddenly land in a different partition than their older events, and the per-key ordering you were relying on shatters precisely at the moment of the change.

This post is the capacity-planning companion to the ordering work in the series. Where [message ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) taught you *which* key to partition by, this post teaches you *how many* partitions to create, and why getting that number wrong in either direction is expensive. The figure below is the whole tension in one picture: a topic with too few partitions, where adding consumers buys you nothing and lag piles up, beside a right-sized topic where every consumer is busy and there is headroom to grow.

![A two-panel comparison showing a topic with two partitions starving six consumers so four sit idle and lag climbs on the left, and a topic with twelve partitions keeping six consumers busy with headroom to double on the right](/imgs/blogs/partitioning-capacity-planning-1.webp)

By the end of this post you will be able to do four concrete things. You will be able to estimate the per-partition throughput your producers and consumers can actually sustain, instead of guessing. You will be able to plug those numbers into a sizing formula that gives you a defensible partition count with headroom baked in. You will be able to articulate, in numbers your platform team will respect, the real cost of over-partitioning — slower rebalances, more memory, slower failover — so the answer to "let's just make it 10,000 to be safe" is a confident no. And you will know exactly why repartitioning is the thing you plan to avoid rather than the thing you plan to do, which is what makes the up-front number matter so much. The sibling post on [consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) picks up where this one leaves off: once the partition count is fixed, how you wring throughput out of the consumers that read those partitions.

## 1. The partition is the unit of parallelism

Start with the single most important sentence in this entire post, because every other rule is a corollary of it: **in a log-based system, the partition is the unit of parallelism, and consumer parallelism is capped at the partition count.** Internalize that and the rest of capacity planning becomes arithmetic.

Here is the mechanism, stripped to its bones. A Kafka topic is split into partitions. Each partition is an independent, ordered, append-only log. When a consumer group subscribes to a topic, the group's coordinator assigns each partition to exactly one consumer in the group. A partition is never split across two consumers — that would break the per-partition ordering guarantee that is the entire reason partitions exist. So if a topic has `P` partitions and a consumer group has `C` consumers, the assignment gives each consumer roughly `P/C` partitions. The instant `C` exceeds `P`, the extra `C - P` consumers get zero partitions and do nothing. They are not a backup, they are not a warm standby that shares load — they are idle processes consuming heartbeat bandwidth and nothing else.

This is the hard ceiling that surprises people. You can throw hardware at a slow consumer group all day, but if the topic has 8 partitions, the 9th consumer is decorative. Your *only* lever for more consumer parallelism is more partitions, and you cannot add partitions without consequences (sections 6 and 9). That is why the partition count is the capacity decision: it is the dial that sets the ceiling on how fast you can ever drain this topic.

### Producers are not capped the same way

It is worth being precise, because the producer and consumer sides are asymmetric. On the producer side there is no hard cap analogous to the consumer cap. Any number of producer threads, processes, or machines can write to the same topic concurrently, and the broker spreads their writes across partitions by the partitioner. A thousand producer instances can hammer a 6-partition topic; the writes just funnel into 6 logs. So producers are limited by *per-partition write throughput*, not by a one-consumer-per-partition rule. This asymmetry matters for the sizing formula later: the consumer side usually binds the partition count, because it has both a per-partition rate limit *and* the hard parallelism cap, while the producer side has only the rate limit.

### Why one consumer per partition, and not finer

People reasonably ask: why can't two consumers split a single partition's load? Because the consumer commits a single monotonic offset per partition — "I have processed up to offset 4,901,233." That offset is a single scalar. If two consumers were reading the same partition, there would be two notions of "how far we've gotten," they would process overlapping or interleaved ranges, and the per-partition ordering and the offset bookkeeping would both collapse. The one-consumer-per-partition rule is not an arbitrary limitation; it falls directly out of wanting ordered, offset-tracked, exactly-assigned reads. The [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) post goes deep on exactly how that assignment and offset machinery works; here the only thing you need is the consequence: partitions are the quanta of parallelism, and you buy parallelism by the partition.

### Partitions in other brokers, and what travels

It is worth a moment to note that "the partition is the unit of parallelism" is a Kafka-shaped statement, and other brokers express the same idea differently — which tells you whether this whole post applies to your system. Kafka, Pulsar, and Kinesis are all *log-partitioned*: a topic (or stream) is split into partitions (Pulsar calls them partitions too; Kinesis calls them shards), each an ordered log, each consumed by at most one reader in a group. For all three, the partition/shard count is the parallelism unit and most of this post applies almost verbatim. Kinesis even prices and rate-limits *per shard* (1 MB/s in, 2 MB/s out, 1,000 records/s per shard), which makes the per-partition throughput accounting in this post literally the billing model — you size shards by dividing your target by the per-shard limit, exactly the formula in section 3.

RabbitMQ is different, and the difference is instructive. A classic RabbitMQ queue is a single ordered structure with a single active consumer for ordered delivery; you parallelize by adding *queues* (sharded queues, consistent-hash exchange) or by accepting unordered competing consumers on one queue. So RabbitMQ's "partition count" analog is "how many queues you shard across," and the [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) discussion of sharded queues is the RabbitMQ flavor of this exact capacity decision. The headline lesson — *parallelism is quantized, and the number of quanta is a deliberate up-front choice* — is broker-independent. Only the mechanism and the cost of changing the number differ. With that caveat noted, the rest of this post speaks in Kafka terms because Kafka makes the tradeoffs sharpest and the repartition pain most acute.

### What a single partition can and cannot do

One more foundation before the numbers: be clear-eyed about what a *single* partition can sustain, because the formula divides your target by exactly that. A single partition is one ordered log with one leader broker accepting writes and one consumer reading. Its write ceiling is the leader's sequential-append rate minus replication overhead; its read ceiling is the broker's stream-out rate, which is very high; and its *processing* ceiling is whatever your consumer does per message. The partition does not parallelize internally — there is exactly one writer-accepting leader and exactly one consumer in the group. So when people say "Kafka does millions of messages a second," that is across *many* partitions and *many* brokers; one partition is the much smaller building block you actually plan with. Treating the aggregate cluster benchmark as if it were a per-partition number is a classic over-confidence error that produces wildly under-partitioned topics.

### The mental shift

The way this works in practice changes how you think about scaling. In a stateless web tier, you scale by adding pods behind a load balancer, and the balancer evenly fans requests across however many pods you have — parallelism is continuous and elastic. A partitioned log is *not* like that. Parallelism is quantized: it comes in units of one partition, and the number of units is fixed when you create the topic. You do not scale a consumer group by "adding more pods until it's fast"; you scale it up to the partition count and then you are done, and the only way past that wall is to change the most expensive number in the system. Hold that distinction — continuous web scaling versus quantized partition scaling — because mis-applying the web-tier intuition to a log is the root cause of most under-partitioned topics.

## 2. Estimating per-partition throughput

You cannot size partitions without two numbers: how fast a single partition can be *written* and how fast a single partition can be *read and processed*. These are not broker-spec-sheet numbers; they are properties of your messages, your serialization, your consumer logic, and your hardware. The discipline is to measure them, not to trust a vendor benchmark run on 100-byte messages and a no-op consumer.

### Per-partition producer throughput

A single partition is an append-only log on one broker (the leader for that partition). Writing to it is sequential disk append plus replication to followers. On modern hardware a single partition can absorb a *lot* — tens to low hundreds of MB/s — but the realistic number depends on message size, batching, compression, and `acks`. Tiny messages with no batching are dominated by per-request overhead and you get far fewer MB/s of useful payload; large batched, compressed messages with `acks=all` push much higher. A reasonable planning figure for a well-batched producer is somewhere in the range of 10 to 50 MB/s per partition of *post-compression* bytes, but you should measure your own. Here is a producer config tuned for throughput, the kind that gets you into the upper end of that range:

```java
// Throughput-oriented Kafka producer config
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092,broker2:9092");
props.put("acks", "all");                    // durability; costs latency, not much throughput
props.put("batch.size", 131072);             // 128 KB batches, not the 16 KB default
props.put("linger.ms", 10);                  // wait up to 10ms to fill a batch
props.put("compression.type", "lz4");        // cheap CPU, big wire savings
props.put("max.in.flight.requests.per.connection", 5);
props.put("enable.idempotence", true);       // safe with in-flight > 1 on modern brokers
props.put("buffer.memory", 67108864);        // 64 MB accumulator
```

The single biggest lever here is batching. A producer that sends one record per request is making a network round-trip and a broker append per message; throughput is bounded by request rate, not bandwidth. Bump `batch.size` and `linger.ms` so each request carries hundreds of records, turn on `lz4` compression, and the *same partition* on the *same hardware* can suddenly absorb an order of magnitude more payload. This is why "per-partition producer throughput" is not a constant — it is a function of how well you batch.

The other factor people forget is `acks` and replication. With `acks=all` and replication factor 3, a write is not acknowledged until the leader has appended it *and* the in-sync followers have replicated it. That adds latency but, importantly, it usually does not slash *throughput* much, because replication is itself batched and pipelined — the followers fetch in bulk. What it does is make the per-partition write rate sensitive to the *slowest* in-sync replica. If one follower is on a degraded disk, the whole partition's `acks=all` write rate drops to that follower's pace. So your measured per-partition producer rate is partly a property of your worst replica, not just your leader. The ISR and `acks` replication mechanics determine this; for sizing, just know that you should benchmark with your *real* `acks` and replication setting, not `acks=1` on a single broker, or your per-partition number will be optimistic.

The factors that move per-partition producer throughput, roughly in order of impact:

| Factor | Effect on per-partition rate | Planning note |
| --- | --- | --- |
| Batching (`batch.size`, `linger.ms`) | Largest lever; 10x swings | Benchmark with production batch settings |
| Compression (`lz4`, `zstd`) | More logical bytes per wire byte | Record both logical and post-compression MB/s |
| Message size | Larger = more bandwidth-bound, fewer req/s overhead | Use your real average size |
| `acks` and replication factor | Sets latency; throughput sensitive to slowest ISR | Benchmark with real `acks=all`, RF=3 |
| Hardware (disk, NIC) | Caps the sequential append and replication rate | Measure on production-class instances |

#### Worked example: measuring per-partition producer throughput

Suppose your messages average 1 KB before compression and compress to about 0.4 KB with `lz4`. You run a single-producer, single-partition benchmark and observe it sustains 50,000 messages/s before the broker append rate becomes the bottleneck. That is `50,000 x 0.4 KB = 20 MB/s` of post-compression bytes, or `50,000 x 1 KB = 50 MB/s` of logical payload, on one partition. If you record per-partition throughput in *logical* MB/s (the number your capacity target is usually expressed in), this partition gives you 50 MB/s. Round it down for safety to a planning figure of 40 MB/s per partition for producers. That conservative 40 is the number you will feed into the formula in section 3.

### Per-partition consumer throughput

The consumer side is where reality bites, because consumer throughput is almost never limited by the broker's read speed — it is limited by *what your consumer does with each message*. Reading from a partition is sequential and fast; the broker can stream a partition to a consumer at hundreds of MB/s. But your consumer deserializes, maybe enriches with a database lookup, maybe calls a downstream service, maybe writes to a sink, and *that* work is what caps the per-partition consume rate. A consumer that does a 5 ms synchronous database call per message processes at most 200 messages/s per partition no matter how fast Kafka is. That is the number that matters.

```python
# A consumer whose per-partition throughput is set by its own work, not by Kafka
from confluent_kafka import Consumer

c = Consumer({
    "bootstrap.servers": "broker1:9092",
    "group.id": "enrichment-service",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
    "max.poll.records": 500,          # fetch many; process in a tight loop
    "fetch.min.bytes": 65536,         # let the broker accumulate before replying
    "max.partition.fetch.bytes": 1048576,
})
c.subscribe(["events"])

while True:
    records = c.consume(num_messages=500, timeout=1.0)
    for r in records:
        enrich_and_write(r)           # <-- THIS is your per-partition throughput ceiling
    c.commit(asynchronous=False)
```

The lesson encoded in that comment is the whole game on the consumer side. Kafka will happily feed you faster than you can process. Your per-partition consume rate is `1 / (per-message processing time)`. If processing takes 1 ms, one partition can sustain ~1,000 msg/s. If it takes 10 ms, ~100 msg/s. To get the per-partition *MB/s* number for the formula, multiply by your average message size. The [consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) post is entirely about pushing that per-message time down — batching downstream calls, async processing, parallelizing within a partition while preserving order — but for sizing, you take your *current* measured per-message time and work from it.

A subtlety that trips people up: the per-partition consumer rate must include *all* the steady-state overhead, not just the happy-path processing. Offset commits cost something (a synchronous commit per batch blocks the poll loop briefly). Deserialization costs something. Rebalances, even brief ones, periodically pause processing. Retries on transient failures stall a partition while the consumer backs off. If you benchmark a no-op consumer that just counts messages, you will measure a per-partition rate ten or a hundred times higher than what your real consumer sustains in production, and you will under-partition badly. Always benchmark the *real* handler, with commits, against a realistic message stream.

### Benchmark discipline: measure, don't trust

The single most common sizing failure is using a number from a blog post or a vendor benchmark instead of measuring your own system. Those benchmarks use tiny messages and no-op consumers precisely to show big numbers. Your messages are bigger, your consumer does real work, and your per-partition rate is a fraction of theirs. The discipline:

- **Benchmark one producer to one partition, and one consumer to one partition**, in isolation, on production-class hardware.
- **Use your real message sizes, serialization, compression, `acks`, and replication factor** on the producer side.
- **Use your real handler** — including the database call, the downstream HTTP request, the commit — on the consumer side.
- **Measure sustained throughput, not burst.** Run for minutes; let GC, page-cache warmup, and commit cadence settle. The number you want is the steady-state floor, not the first ten seconds.
- **Round down for safety.** A measured 50 MB/s becomes a planning figure of 40; a measured 480 msg/s becomes 400. You want the formula to over-estimate partitions slightly, not under-estimate.

These two measured numbers — per-partition producer rate and per-partition consumer rate — are the only inputs the formula needs that are specific to *you*. Everything else (target, headroom) is a business decision. Get these two right and the rest is arithmetic.

#### Worked example: per-partition consumer throughput

Your enrichment consumer does one database upsert per message, measured at 2 ms p50 and a synchronous commit every batch. Effective steady-state is about 400 messages/s per partition (the 2 ms work plus poll and commit overhead). At 1 KB logical message size that is `400 x 1 KB = 0.4 MB/s`... which sounds tiny, and it is, because this consumer is heavy. But suppose instead you batch the upserts 100 at a time so the per-message amortized cost drops to 0.2 ms; now you sustain ~5,000 msg/s = 5 MB/s per partition. For the sizing example in section 3 we will use a deliberately middling consumer that sustains **20 MB/s per partition** of logical throughput — heavier than a no-op consumer, lighter than the database-per-message extreme. The key takeaway: **the consumer's per-partition rate is almost always lower than the producer's**, because production is a cheap append and consumption is your business logic. That asymmetry is why the consumer side usually decides the partition count.

## 3. The sizing formula

Now we assemble the arithmetic. You have a target aggregate throughput `T` for the topic — the peak rate the topic must sustain, in MB/s (or msg/s; keep units consistent). You have a measured per-partition producer rate `R_p` and a measured per-partition consumer rate `R_c`. The number of partitions you need is governed by whichever side is slower, because the partition count has to satisfy *both* the producers and the consumers:

```
partitions_needed = max( ceil(T / R_p), ceil(T / R_c) )
partitions_final  = ceil( partitions_needed * headroom )
```

Two divisions, a `max`, and a headroom multiplier. The `max` is the crucial part: you cannot pick a partition count that works for producers but starves consumers, or vice versa. You take the larger of the two requirements. And because `R_c` (consumer per-partition rate) is almost always smaller than `R_p` (producer per-partition rate), the consumer division usually wins the `max` — which is the formal version of "the consumer side binds the partition count." The figure below is the formula as a flow: the target and the two per-partition rates feed into the binding-rate decision, which produces a partition count, which gets a headroom multiplier.

![A flow diagram showing target throughput of 600 megabytes per second and two per-partition rates feeding into a binding-rate decision that produces thirty partitions and then forty-five after a one-and-a-half-times headroom multiplier](/imgs/blogs/partitioning-capacity-planning-3.webp)

### Why headroom, and how much

The headroom multiplier exists because the formula gives you the number that *exactly* meets the target with zero slack, and a system running at exactly 100% of capacity is a system with no margin for traffic spikes, consumer slowdowns, a node failure that concentrates load, or growth between now and the next time anyone revisits this decision. And critically — because adding partitions later is painful (section 6) — the cheap insurance is to over-provision *now*. A common rule is 1.5x to 2x headroom over the current peak requirement, plus an explicit allowance for projected growth over the planning horizon (say, the next 12 to 18 months). If you expect to triple traffic in a year, that goes into the number too. Partitions are cheap up to a point (section 4 tells you where that point is); under-provisioning is expensive; so the asymmetry of costs justifies leaning generous.

#### Worked example: sizing for a 600 MB/s target

Let us run the headline example end to end. Requirements:

- Target peak throughput: **T = 600 MB/s** (logical bytes).
- Measured per-partition producer rate: **R_p = 40 MB/s** (from the section 2 producer worked example).
- Measured per-partition consumer rate: **R_c = 20 MB/s** (the middling consumer).

Apply the formula:

```
producer requirement = ceil(600 / 40) = ceil(15)  = 15 partitions
consumer requirement = ceil(600 / 20) = ceil(30)  = 30 partitions
partitions_needed    = max(15, 30)               = 30 partitions
partitions_final     = ceil(30 * 1.5)            = 45 partitions
```

The consumer side binds: it needs 30 partitions where the producer side only needed 15, so 30 is the floor, and 1.5x headroom takes you to **45**. Notice what just happened — if you had sized off the producer rate alone (a common mistake, because producers are easy to benchmark and consumers are annoying to benchmark), you would have created a 15-partition topic, capped your consumer group at 15 consumers, and been unable to keep up the moment consumer load matured. The `max` over both sides is what saves you. Round 45 up to a convenient number if you like — say 48, which divides evenly by 2, 3, 4, 6, 8, 12, 16, and 24, giving you many even consumer-group sizes that produce balanced assignments. Even divisibility is a small but real nicety: with 48 partitions a group of 12 consumers gets exactly 4 partitions each, while 45 partitions gives an uneven 4-or-5 split.

### Sizing in messages instead of bytes

If your bottleneck is per-message work rather than bandwidth (the common case for heavy consumers), run the exact same formula in messages per second. Target 200,000 msg/s, producer per-partition rate 50,000 msg/s, consumer per-partition rate 400 msg/s (the heavy database-per-message consumer):

```
producer requirement = ceil(200000 / 50000) = 4 partitions
consumer requirement = ceil(200000 / 400)   = 500 partitions
partitions_needed    = max(4, 500)           = 500 partitions
```

Five hundred partitions — driven entirely by a slow consumer. This is the case where you should stop and ask whether the right fix is 500 partitions or a faster consumer, because (as the next sections show) 500 partitions is not free, and a consumer that processes at 400 msg/s is begging to be optimized. Often the cheaper engineering move is to make the consumer 10x faster (batch its database writes) and need 50 partitions instead of 500. The formula does not tell you the *only* answer; it tells you the cost of your current consumer, and sometimes the right response is to fix the consumer.

## 4. The costs of too many partitions

If partitions buy parallelism, why not create 10,000 of them and never think about it again? Because every partition imposes real, continuous, cluster-wide costs, and they compound. "Just make it big to be safe" is the second most common partition mistake (after under-provisioning), and it is the one that quietly degrades the *whole cluster*, not just one topic. The figure summarizes how the symptoms shift as you move from too few to too many.

![A decision matrix with rows for too low at two partitions, right-sized at twelve to thirty, and too high at five thousand, scored across throughput, rebalance time, broker memory, and p99 latency](/imgs/blogs/partitioning-capacity-planning-2.webp)

Let me make each cost concrete, because vague hand-waving about "overhead" never wins an argument with a team that wants to over-provision wildly.

### Longer rebalances

A consumer-group rebalance has to reassign every partition. The work of a rebalance scales with the number of partitions being reassigned. With 50 partitions a rebalance completes in well under a second. With 5,000 partitions in a group, the coordinator is computing and distributing an assignment over 5,000 entries, every consumer is revoking and re-acquiring its share, and — with the classic eager rebalance protocol — the entire group stops consuming for the duration. Rebalances that took milliseconds now take tens of seconds, and during that window your consumers are processing *nothing*. Every deploy, every scale event, every consumer crash triggers a rebalance, so this cost is paid constantly, not once.

### More broker memory and open file handles

Each partition replica on a broker is backed by log segment files plus index files (offset index, time index). Each of those is an open file descriptor. A broker hosting 4,000 partition replicas, each with several segments and two index files per segment, can easily hold tens of thousands of open file handles — and you will hit the OS `nofile` limit and get cryptic "too many open files" crashes if you have not raised it. Beyond file handles, each partition carries in-memory state: producer state for idempotence, replication fetch state, buffers. None of it is huge per partition, but multiply by tens of thousands of partitions per broker and it becomes gigabytes of memory that is doing bookkeeping rather than buffering your actual data. The figure below stacks these per-partition cost layers so you can see what each partition is actually charging you.

![A layered stack showing the per-partition cost layers of controller metadata, leader election work, open file handles for index and log segments, broker memory for buffers and replica state, and a replication fetch stream per partition](/imgs/blogs/partitioning-capacity-planning-5.webp)

### Slower leader election and controller failover

Here is the cost that scares operators the most, because it is a cluster-wide availability risk rather than a per-topic performance nuisance. Every partition has a leader replica, and the cluster controller tracks the leadership and ISR (in-sync replica) state for *every partition in the entire cluster*. When a broker fails, the controller must elect new leaders for every partition that broker led. When the controller itself fails, a new controller has to load the metadata for every partition in the cluster before the cluster is fully operational again. In the older ZooKeeper-based architecture this metadata load was notoriously slow — controller failover with hundreds of thousands of partitions could take many seconds to minutes, during which leadership moves were stalled and producers/consumers saw errors. KRaft (the ZooKeeper-free controller) dramatically improved this, raising the practical ceiling, but it did not make it free: metadata still scales with partition count, and failover time still grows with it. The figure contrasts the over-partitioned and balanced regimes on exactly this axis.

![A two-panel comparison showing a cluster with two hundred thousand partitions suffering controller failover over thirty seconds and stop-the-world rebalances on the left versus a balanced cluster around ten thousand partitions with sub-three-second failover on the right](/imgs/blogs/partitioning-capacity-planning-4.webp)

### Higher end-to-end latency

This one is counterintuitive: more partitions can *raise* latency. The reason is batching. Producers and the replication layer batch records per partition. If your throughput is spread across 5,000 partitions, each partition receives records more slowly, so batches take longer to fill (or `linger.ms` expires with a half-empty batch), and more, smaller requests cross the network. The replication fetchers similarly do more, smaller fetches. The fixed per-request overhead is now amortized over fewer bytes, so per-message latency rises and effective throughput can actually *fall* past the sweet spot. Spreading the same workload over too many partitions dilutes the batching that makes the system fast in the first place.

### More replication overhead

Each partition is replicated independently. With replication factor 3, every partition has two follower fetch streams continuously pulling from the leader. Ten thousand partitions means twenty thousand replication streams across the cluster, each with its own request cadence and bookkeeping. Replication traffic and the threads/connections that carry it scale with partition count, so over-partitioning multiplies the background replication work the cluster does even when application throughput is unchanged. This stacks on top of the per-partition memory and file-handle costs to make a heavily over-partitioned cluster slower and more fragile across the board.

#### Worked example: rebalance and memory from 50 to 5,000 partitions

Make the over-partitioning cost concrete by scaling one number. Take a consumer group and grow the topic's partition count while holding everything else fixed.

| Partition count | Eager rebalance time | Open file handles (RF=3, ~4 segments/part) | Replication streams (RF=3) |
| --- | --- | --- | --- |
| 50 | ~0.3 s | ~1,800 | 100 |
| 500 | ~2 s | ~18,000 | 1,000 |
| 2,000 | ~10 s | ~72,000 | 4,000 |
| 5,000 | ~30 s | ~180,000 | 10,000 |

The numbers are illustrative, not a benchmark, but the *shape* is real and it is what matters: every column grows roughly linearly with partition count. Going from 50 to 5,000 partitions (a 100x increase) turns a 0.3-second rebalance into a 30-second one — and remember, you pay that 30-second stall on *every* deploy and *every* crash. The open file handles cross the default Linux `nofile` ceiling (often 1,024, raised to maybe 100,000 in production) and the replication stream count goes from trivial to a meaningful chunk of the cluster's network and CPU budget. None of this bought you anything if your actual throughput did not need 5,000 partitions. That is the trap: the costs are very real and the benefit is zero past the point where partition count exceeds what your throughput and consumer parallelism require. The timeline figure shows the rebalance-time growth curve directly.

### Cooperative rebalancing softens but does not erase the cost

A fair objection: the table above uses the *eager* (stop-the-world) rebalance protocol, and modern Kafka offers *cooperative* (incremental) rebalancing that only revokes the partitions that actually need to move, so the group keeps processing the unaffected partitions during a rebalance. That is a genuine improvement and you should use it (`partition.assignment.strategy` set to the cooperative sticky assignor). But it does not erase the partition-count cost — it changes its character. With cooperative rebalancing the group does not fully stop, but the rebalance still has to *compute* an assignment over all partitions, and the larger that number, the longer the coordinator works and the more revocation/re-acquisition churn happens. The metadata, file-handle, leader-election, and replication costs are entirely unchanged by the rebalance protocol — those are properties of the partitions existing, not of how they get assigned. So cooperative rebalancing trims one column of the cost table; the rest of the costs in section 4 stand regardless.

```properties
# Use cooperative (incremental) rebalancing to avoid stop-the-world pauses
partition.assignment.strategy=org.apache.kafka.clients.consumer.CooperativeStickyAssignor
# Static membership avoids a rebalance entirely on a quick restart
group.instance.id=consumer-pod-7
session.timeout.ms=45000
```

### Topic consolidation: the cure for sprawl

If you inherit an over-partitioned cluster (the section-on-meltdown scenario), the remedy is *topic consolidation*: many teams create one topic per tiny event type, each with a generous partition count "for headroom," and the sum is hundreds of thousands of partitions doing almost nothing. Consolidating related low-volume event types into a single topic (with an event-type field in the payload or header) collapses dozens of barely-used topics into one right-sized topic, reclaiming the per-partition costs across the board. The general principle: partition count should track *throughput*, and a topic that does 50 msg/s does not need 50 partitions no matter how much "headroom" felt prudent at creation. Headroom is for topics that will *grow*, not a default sprinkled on everything.

![A timeline showing rebalance time growing from under one second at fifty partitions to several seconds at two thousand partitions to around thirty seconds at five thousand partitions and minutes at twenty thousand](/imgs/blogs/partitioning-capacity-planning-6.webp)

## 5. The costs of too few partitions

If over-partitioning degrades the cluster slowly, under-partitioning fails you suddenly and visibly, usually at the worst possible time. The costs are the mirror image of the previous section.

### Limited parallelism, hard ceiling

We have hammered this, but it is the dominant cost so it bears restating in operational terms. With `P` partitions your consumer group tops out at `P` consumers. If `P = 6` and one consumer can process 1,000 msg/s, your group's maximum throughput is 6,000 msg/s — and there is *no way to go faster* without changing `P`. When traffic exceeds that ceiling, lag grows without bound, and your only emergency lever (add consumers) does nothing because consumers 7 and beyond are idle. You are stuck watching the backlog grow with the one tool that would normally fix it rendered useless by the partition count. This is the on-call nightmare that the intro promised: a backlog that should drain in twenty minutes takes nine hours because you cannot parallelize the drain.

### No room to scale consumers

Even if you are keeping up today, too few partitions means no headroom. The day a downstream dependency gets slower, or traffic doubles, or you need to reprocess history, you want to throw consumers at the problem — and you cannot, because the partition count caps you. Right-sizing with headroom (section 3) exists precisely to leave room for these scale events. A topic sized at exactly today's load with no slack is a topic that will fail the first time anything changes.

### Hot partitions become catastrophic

With few partitions, any key-distribution skew (section 8) is amplified. If one key is hot and you have 6 partitions, that hot key dominates one of only 6 partitions, so it is one-sixth of your total capacity bottlenecked on one consumer. With 60 partitions the same hot key is one-sixtieth of capacity and the skew is diluted across more partitions. Fewer partitions concentrate skew; the hot partition's consumer falls behind while the other five idle, and your effective throughput collapses to roughly the rate of the single hot partition. We will return to this, but note it here as a *cost of under-partitioning*: too few partitions make every skew problem worse.

### Painful to fix

And the final cost, the one that ties the whole post together: fixing under-partitioning is the painful repartition operation, because the only way to get more parallelism is to add partitions, and adding partitions breaks `hash(key) % N`. So under-provisioning does not just hurt now; it backs you into the one operation you most want to avoid. The asymmetry could not be sharper: over-provisioning is a gradual, recoverable tax you can tune down later by consolidating topics; under-provisioning is a wall you hit at speed, and climbing over it means breaking ordering. That asymmetry is the entire argument for leaning toward more partitions when in doubt — within the limits section 4 established.

## 6. Why repartitioning breaks ordering

This is the section that explains *why* the partition count is "hard to change" and not merely "annoying to change." The reason is not operational fiddliness — it is a correctness violation baked into how keys map to partitions.

### The default partitioner is hash modulo N

When you produce a record with a key, the default partitioner computes `partition = hash(key) % numPartitions`. This is what gives you per-key ordering: every record with key `user-42` hashes to the same value, mods to the same partition, and therefore lands in the same ordered log. All of `user-42`'s events are in one partition, in production order, read by one consumer in that order. The guarantee depends entirely on the mapping being *stable*: the same key must always go to the same partition.

Now add partitions. The topic goes from `N` partitions to `M`. The partitioner now computes `hash(key) % M` instead of `hash(key) % N`. For almost every key, `hash(key) % M` is a *different number* than `hash(key) % N` — that is just how modular arithmetic works when you change the modulus. So `user-42`, which used to map to partition 3, now maps to partition 11. The figure makes the break visible: on the left, repartitioning remaps the key and shatters per-key order; on the right, over-provisioning up front keeps the mapping fixed for the life of the topic.

![A two-panel comparison showing repartitioning from ten to sixteen partitions remapping hash modulo so a key jumps partitions and per-key order breaks on the left, versus over-provisioning thirty partitions up front keeping a fixed mapping and intact order on the right](/imgs/blogs/partitioning-capacity-planning-9.webp)

### Why the jump breaks order

Here is the corruption, step by step. Before the change, `user-42`'s events `e1, e2, e3` are all in partition 3, in order. You add partitions. Now `user-42`'s *new* events `e4, e5` go to partition 11, while `e1, e2, e3` are still sitting in partition 3 waiting to be consumed or already consumed. Two different consumers own partition 3 and partition 11. There is no longer any ordering relationship between `user-42`'s old events and new events — they are in different partitions, read by different consumers, with no cross-partition order guarantee whatsoever. If `e3` (in partition 3) is processed *after* `e4` (in partition 11) because partition 11's consumer happened to be faster, you have applied a newer event before an older one for the same user. For a state machine — account balance, profile state, inventory count — that is data corruption, exactly the email-overwrite bug from the [message ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) post, now caused by an innocent-looking `kafka-topics --alter --partitions 16`.

### It is not just an ordering blip at the boundary

You might hope this is a one-time hiccup that heals itself once the old events drain. It is worse than that in two ways. First, any in-flight per-key state across the boundary is at risk — a consumer that crashes and reprocesses, a key that is mid-sequence, a windowed aggregation keyed by the partition, all see the discontinuity. Second, and more insidiously, **stateful stream processors that key their local state store by partition are now broken**. A Kafka Streams or Flink job keeps per-key state co-located with the partition. When keys move partitions, the state for those keys is in the *old* partition's state store but the new events arrive at the *new* partition, whose state store has never seen that key. The aggregation starts from scratch, or worse, double-counts. Repartitioning a topic feeding a stateful processor is not a blip; it can silently corrupt every aggregate.

### Kafka cannot shrink partitions at all

And to underline that this is a one-way ratchet: Kafka does not even *allow* reducing the partition count. You can only ever add partitions, never remove them, because removing a partition would orphan its data and there is no safe automatic way to redistribute it. So the partition count is monotonic for the life of a topic — it can only go up, and every increase is an ordering-breaking event. The only truly clean way to "repartition" is to create a *new* topic with the desired partition count and migrate producers and consumers to it (often with a careful dual-write and replay), which is a project, not a config change. That is why the up-front number matters so much: you are choosing a value you can grow with pain and never shrink.

### The clean repartition: a new topic and a migration

When you genuinely must change the partition count of a keyed topic — you under-provisioned and you are now lagging — the safe path is not `--alter`, it is a topic migration. The shape of it:

1. **Create a new topic** `events-v2` with the desired (higher) partition count and the same key. From day one its `hash(key) % M` mapping is fixed; you will never repartition *it*.
2. **Dual-write** from producers to both `events` (old) and `events-v2` (new) for a transition window, so new data lands in both.
3. **Backfill** historical data from old to new if consumers need it, replaying through a job that re-keys correctly into the new partition layout — this rebuilds any per-partition stateful processing against the new mapping.
4. **Cut consumers over** to `events-v2` once it is caught up, verify lag and correctness, then stop the dual-write and retire `events`.

```bash
# Create the new, correctly-sized topic — do this once, off-peak
kafka-topics --bootstrap-server broker1:9092 --create \
  --topic events-v2 --partitions 96 --replication-factor 3 \
  --config retention.ms=604800000 --config min.insync.replicas=2

# Producers dual-write to events and events-v2 during the migration window,
# then flip the read side once events-v2 is verified caught up.
```

The reason this is "a project, not a config change" is steps 2 through 4: dual-writing, backfilling, and verifying correctness across the cutover is real engineering with real risk, and it wants to happen far from peak traffic. The entire reason to over-provision up front is to *never run this migration*. If you find yourself planning one, the lesson for next time is to size more generously at creation. Treat the migration as the expensive escape hatch it is, not a routine operation.

## 7. Over-provisioning vs scaling later

Given everything above, the standard professional advice crystallizes into a clear default: **over-provision partitions up front, then scale consumers (not partitions) later.** Let me defend that and then bound it, because "over-provision" taken to an extreme is the section-4 disaster.

### Why over-provision is the default

The logic is a cost asymmetry. Adding partitions later breaks ordering and stateful processing (section 6) — it is a *correctness* hazard requiring a migration project. Adding *consumers* later (up to the partition count) is trivial, safe, and reversible — it is just starting more processes. So if you provision enough partitions up front, all future scaling is the easy, safe kind: you grow the consumer group toward the partition ceiling as load grows, and you never touch the partition count. You have decoupled "scale the topic" (the painful thing, done once at creation) from "scale the consumers" (the easy thing, done continuously). That is the entire value of over-provisioning: it converts a future correctness hazard into a present, cheap, one-time decision.

### How far to over-provision

The bound is set by section 4. Over-provision enough that you will not need to add partitions within the planning horizon, but not so far that the per-partition costs degrade the cluster. Concretely: take the sizing-formula number, apply your headroom and growth factor, and land on a count that gives you comfortable consumer-parallelism room — but keep the *cluster-wide* partition total within healthy operating ranges. As a rough sanity bound, modern KRaft-based Kafka clusters comfortably handle low hundreds of thousands of partitions cluster-wide, and an individual broker is happy with a few thousand partition replicas; older ZooKeeper clusters wanted to stay well under ~4,000 partitions per broker and ~200,000 per cluster. So "over-provision" means "give a topic 30 or 50 or 100 partitions when 12 would do today," not "give it 10,000." The figure earlier (the cost matrix) is the guide: stay in the right-sized band, lean toward the generous end of it, and do not cross into the too-high band.

### The pre-splitting analogy

If you have done database sharding, this will feel familiar. The same logic appears in [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding): you pre-split into more shards than you currently need so that future growth is "move a shard to a new node" (easy) rather than "re-shard the whole keyspace" (a migration). Streaming partitions are the same pattern — pre-split the parallelism so future scaling is reassignment, not remapping. The difference is that some databases use consistent hashing or range partitioning that makes adding shards less disruptive, whereas Kafka's plain `hash % N` makes adding partitions maximally disruptive to ordering — which is exactly why the over-provision-up-front discipline is *more* important for Kafka than for many databases.

### When you genuinely cannot predict

Sometimes you truly cannot estimate future load — a brand-new product, an unknown adoption curve. In that case, two pragmatic moves. First, if per-key ordering is *not* a hard requirement for this topic, then repartitioning is far less scary: with no keyed ordering to break, adding partitions is just a rebalance of where new data lands, and you can grow partitions more freely. So the cost of getting the number wrong is much lower for unkeyed topics — size those leaner and grow them as needed. Second, if ordering *is* required, lean generous within the healthy band (say, provision for 5-10x current load) and accept a modest over-provisioning tax as insurance against the much larger cost of an ordering-breaking repartition. Match your conservatism to whether ordering is on the line.

### A decision you revisit, not a knob you tune

It helps to frame the whole over-provision question as: *what is the cheapest way to be wrong?* If you over-provision and it turns out you did not need the partitions, you paid a small continuous tax (a bit more memory, slightly longer rebalances) that you can later reclaim by consolidating topics — recoverable, gradual, low-stakes. If you under-provision and it turns out you needed more, you hit a wall at peak and the only fix breaks ordering and requires a migration — sudden, high-stakes, correctness-threatening. When the two ways of being wrong are that asymmetric, you bias toward the cheap-to-be-wrong direction, which is generous-up-front. This is the same risk logic that governs database pre-sharding and capacity reservations everywhere: when one mistake is recoverable and the other is catastrophic, lean toward the recoverable one. The partition count is a *decision you revisit annually with fresh numbers*, not a *knob you tune weekly* — and you revisit it by measuring whether you are approaching the consumer-parallelism ceiling, not by reflexively bumping it.

## 8. Hot partitions and key skew

The sizing formula assumes load spreads evenly across partitions. Reality often disagrees, because keys are not uniformly active, and an uneven key distribution produces *hot partitions* — partitions carrying far more than their fair share, whose consumers fall behind while other consumers idle. A perfectly sized partition count is useless if all the traffic piles into three of the forty partitions.

### Where skew comes from

Skew has two flavors. **Key skew**: some keys are intrinsically hotter than others. Partition by `customer_id` and your one whale customer who is 40% of your traffic puts 40% of the load into the single partition their `hash(customer_id)` lands in. **Key-cardinality skew**: too few distinct keys relative to partitions. Partition by `country_code` across 50 partitions and you have at most ~200 distinct keys, most traffic in a handful of countries, so most partitions are empty or cold and a few are scalding. The partition count can be perfect and the *key choice* still wrecks you, because the partitioner can only spread load as well as the key distribution allows.

### Detecting hot partitions

You detect skew by looking at per-partition metrics, not topic-level aggregates. A topic-level "1 GB/s, all good" hides that partition 7 is doing 300 MB/s and the rest are doing 15 MB/s each. The signals: per-partition byte and message rates that vary wildly, per-partition consumer lag concentrated in a few partitions, and per-partition log-size growth that is lopsided. Watch the partition with the *most* lag, not the average lag, because the hot partition's consumer is your real throughput ceiling.

```bash
# Per-partition lag — find the hot partition, don't trust the topic average
kafka-consumer-groups --bootstrap-server broker1:9092 \
  --describe --group enrichment-service

# TOPIC   PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# events  0          4500120         4500140         20
# events  7          2100000         9800000         7700000   <-- hot partition
# events  12         4500050         4500071         21
# ...
```

That `7700000` lag on partition 7 while every other partition has lag in the tens is the unmistakable signature of a hot partition. No amount of *total* partition count fixes it; the load is concentrated on one key's partition, and one consumer owns that partition.

### Fixing skew

There are a few real fixes, in rough order of preference. **Choose a higher-cardinality key**: if `customer_id` skews because of whales, but per-`order_id` ordering is sufficient for correctness, partition by `order_id` instead — many more keys, far smoother spread. The key choice is upstream of everything; fix it there if you can. **Composite or salted keys**: when one key is unavoidably hot but you can relax its ordering, append a small salt — `customer_id + ":" + (hash % 4)` — to spread that one customer across 4 partitions, trading strict per-customer order for 4x parallelism on the whale. Use this only when the hot key's events do not need strict mutual ordering. **A custom partitioner**: route known-hot keys deliberately to dedicated partitions, or implement weighted assignment. And as a blunt last resort, **more partitions**: more partitions dilute skew (the hot key is a smaller fraction of total capacity), which is part of why under-partitioning makes skew catastrophic (section 5) — but more partitions never *eliminate* skew, they only spread its blast radius, so the key-choice fix is always better when available.

The salting approach in code, applied selectively to only the known-hot keys so the vast majority of keys keep strict per-key ordering:

```java
public class SaltedPartitioner implements Partitioner {
    // Only these keys are hot enough to need spreading.
    private static final Set<String> HOT_KEYS = Set.of("merchant-9981", "merchant-4420");
    private static final int SALT_BUCKETS = 8;
    private final Random rnd = new Random();

    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                         Object value, byte[] valueBytes, Cluster cluster) {
        int n = cluster.partitionCountForTopic(topic);
        String k = (String) key;
        if (HOT_KEYS.contains(k)) {
            // Spread this one hot key across SALT_BUCKETS partitions.
            // Trades strict per-key order for parallelism on the whale only.
            int salt = rnd.nextInt(SALT_BUCKETS);
            return Math.floorMod(Objects.hash(k, salt), n);
        }
        // Every other key keeps the normal stable hash -> strict per-key order.
        return Math.floorMod(Objects.hashCode(k), n);
    }

    @Override public void close() {}
    @Override public void configure(Map<String, ?> configs) {}
}
```

The important property of this partitioner is that it is *surgical*: only the handful of genuinely hot keys lose strict ordering (and only because you have decided their events are independent enough to tolerate it), while every other key retains the normal stable mapping and its per-key ordering guarantee. This is far better than salting *every* key, which would destroy per-key ordering globally to solve a problem caused by two keys. Skew is usually a few-keys problem; treat it surgically.

### Cardinality is a partition-count constraint too

One more skew subtlety that loops back to sizing: your key cardinality sets an *upper* useful bound on partition count. If your partition key has only 200 distinct values, creating 1,000 partitions is pointless — at most 200 of them can ever be non-empty, and the load piles into however many of those 200 keys are active. The partition count should not exceed (and ideally is comfortably below) the effective key cardinality, or you are paying for partitions that can never carry data. When you reach for a high partition count, double-check that your key has the cardinality to fill it. A high partition count with a low-cardinality key is the worst of both worlds: all the per-partition cost of section 4, none of the parallelism, because the keys cannot spread across the partitions you created.

#### Worked example: a whale customer and a salted key

A payments topic is partitioned by `merchant_id` across 24 partitions, sized for 240 MB/s (10 MB/s per partition consumer rate). One merchant — a giant marketplace — is 30% of all volume: 72 MB/s, all hashing to one partition. That partition needs to absorb 72 MB/s but its consumer sustains only 10 MB/s, so it falls 62 MB/s behind continuously while the other 23 consumers handle their ~7 MB/s each with ease. The topic-level metric says you are at `240` of `240` MB/s capacity, "fine," but partition lag on the whale's partition grows without bound. The fix: the marketplace's per-transaction events do not need strict mutual ordering (different sub-merchants), so you salt — `merchant_id + ":" + (txn_id.hashCode() % 8)` for that one merchant — spreading its 72 MB/s across 8 partitions at 9 MB/s each, now within the 10 MB/s per-partition consumer rate. The whale is tamed, per-other-merchant ordering is untouched, and you did not have to repartition the topic. That is the skew playbook: fix the key, not the partition count, whenever ordering permits.

## 9. A capacity-planning worksheet

Pull it all together into a repeatable worksheet you run *before* you type a partition count into a topic-creation request. The figure shows the end-to-end model the worksheet sizes: producers feeding partitions, partitions read by consumers, partition count as the shared cap that gates both sides.

![A grid showing three producers each at two hundred megabytes per second feeding three groups of ten partitions each at twenty megabytes per second which are read by three consumers each owning ten partitions](/imgs/blogs/partitioning-capacity-planning-8.webp)

Run these steps in order:

1. **State the target throughput at peak, with units.** Not average — peak, because the partition count must survive the spike. In MB/s if bandwidth-bound, msg/s if work-bound. Add a projected-growth multiplier for your planning horizon. Example: 600 MB/s peak today, expecting 2x in 18 months → plan for 1,200 MB/s.
2. **Measure per-partition producer throughput** with your real message size, batching, compression, and `acks`. Benchmark one producer to one partition; record the sustained MB/s (or msg/s). Round down for safety. Example: 40 MB/s/partition.
3. **Measure per-partition consumer throughput** with your real consumer logic, including downstream calls and commits. This is usually the binding number and the one people skip. Record the sustained rate. Example: 20 MB/s/partition.
4. **Apply the formula:** `partitions_needed = max(ceil(T / R_p), ceil(T / R_c))`. The `max` ensures both sides are satisfied. Example: `max(ceil(1200/40), ceil(1200/20)) = max(30, 60) = 60`.
5. **Apply headroom and round to a convenient, highly-divisible number.** 1.5x to 2x for spikes and node-failure load concentration. Example: `60 * 1.5 = 90` → round to 96 (divisible by 2,3,4,6,8,12,16,24,32,48).
6. **Sanity-check against cluster limits.** Add this topic's partitions (times replication factor) to the cluster's existing partition total. Confirm you are within healthy per-broker (a few thousand replicas) and per-cluster (low hundreds of thousands on KRaft) bounds. If this one topic would blow the budget, your per-partition consumer rate is too low — go optimize the consumer instead.
7. **Check the key distribution for skew.** Estimate the cardinality and hotness of your partition key. If a few keys dominate, plan a salting or higher-cardinality-key strategy *now*, because skew defeats even a perfectly sized count.
8. **Record the decision and the assumptions.** Write down the target, the measured per-partition rates, the formula output, and the headroom, in a comment on the topic or in a runbook. When someone revisits this in a year, they need to know *why* it is 96, so they can re-derive it against new numbers instead of guessing again.

The whole decision is a balance of three forces, which the taxonomy figure lays out: the parallelism you want, the cluster cost you pay, and the ordering you must preserve. Every step of the worksheet is serving one of those three.

![A taxonomy tree of partition-sizing considerations branching into parallelism with consumer-cap and hot-partition leaves, cluster cost with rebalance and failover time, and ordering with the repartition-breaks-keys hazard](/imgs/blogs/partitioning-capacity-planning-7.webp)

### A worked worksheet, top to bottom

#### Worked example: sizing a clickstream topic with the full worksheet

A new clickstream topic. Step 1: peak is 200,000 events/s today, product expects 4x growth in a year → plan target `T = 800,000 events/s`. Step 2: events are small (300 bytes), producers batch well, one producer to one partition sustains 80,000 events/s → `R_p = 80,000`. Step 3: the consumer writes each event to a columnar sink in batches of 1,000, amortized 0.1 ms/event → ~10,000 events/s per partition → `R_c = 10,000`. Step 4: `max(ceil(800000/80000), ceil(800000/10000)) = max(10, 80) = 80` partitions, consumer-bound as always. Step 5: 1.5x headroom → 120 → round to 128 (a power of two, divides cleanly for many group sizes). Step 6: at RF=3 that is 384 partition replicas for this one topic; on a 6-broker cluster that is 64 replicas/broker added — well within bounds. Step 7: partition key is `session_id`, very high cardinality, naturally smooth — low skew risk, no salting needed. Step 8: record "128 partitions: target 800k/s, R_p 80k, R_c 10k, 1.5x headroom, keyed by session_id." Done. You have a defensible number, headroom for 4x growth, balanced assignment options, and a written rationale — and you will not be repartitioning this topic.

## Case studies and war stories

### The 6-partition Black Friday

A retail team created their `orders` topic with 6 partitions during early development, when traffic was a trickle and 6 felt generous. The number was never revisited. On Black Friday, order volume spiked 20x. The fraud-check consumer group, which did a ~15 ms model inference per order, could sustain about 65 orders/s per partition — 390 orders/s across 6 partitions. Peak order rate hit 3,000/s. Lag exploded into the millions; orders were being fraud-checked 40 minutes after they were placed; customers got "payment processing" spinners for half an hour. The on-call engineer did the natural thing and scaled the consumer deployment from 6 to 30 pods — and *nothing changed*, because 24 of those pods were idle, holding no partitions. The lesson burned into that team: **the partition count is the throughput ceiling, and you cannot raise it in an incident.** The permanent fix was a new `orders-v2` topic with 64 partitions and a careful dual-write migration done in January, well away from peak. The lesson is the entire reason this post exists: size for peak plus growth up front, because the day you need more parallelism is the day you cannot get it.

### The 200,000-partition controller meltdown

A platform team running a large multi-tenant Kafka cluster (still on ZooKeeper at the time) let every team create topics freely with generous partition counts "for headroom." Nobody tracked the cluster-wide total. It crept past 200,000 partitions. Then a broker hosting the controller had a hardware fault and the controller failover kicked in. Loading metadata for 200,000+ partitions from ZooKeeper took the new controller several minutes, during which leadership changes stalled cluster-wide and producers across every team saw `NOT_LEADER_FOR_PARTITION` errors and timeouts. A single broker fault became a multi-minute, all-tenants partial outage — caused not by any one topic but by the *aggregate* of everyone's "headroom." The fixes were structural: a partition budget per team, an alert on cluster-wide partition count, consolidating many tiny over-partitioned topics, and eventually migrating to KRaft, which made controller failover far faster. The lesson: **over-provisioning is a cluster-wide cost, not a per-topic one**, and "everybody add a little headroom" sums to a meltdown.

### The repartition that corrupted account balances

A fintech team had a `ledger-events` topic keyed by `account_id`, feeding a Kafka Streams job that maintained running account balances in a per-partition state store. They were under-partitioned and lagging, so an engineer ran `kafka-topics --alter --partitions` to double the partition count, expecting a simple throughput win. Within hours, balance discrepancies appeared. The cause: doubling partitions remapped `hash(account_id) % N`, so accounts moved to new partitions whose state stores had never seen them. The Streams job started computing those accounts' balances from a *blank* state, ignoring all prior history, while the old balances sat stranded in the old partitions' state stores. Some accounts double-counted, some reset. It took days to reconcile from the source of truth. The lesson, expensive and unambiguous: **never repartition a keyed topic feeding stateful processing** — the `hash % N` remap is a correctness bomb. The right move would have been a new topic and a full replay to rebuild state, or over-provisioning at creation so the alter was never needed.

### The whale that hid behind a green dashboard

A B2B SaaS company's event pipeline was partitioned by `tenant_id` across 48 partitions, comfortably sized for aggregate load, and the topic-level throughput dashboard was reassuringly green. Yet one specific large tenant kept reporting their data was "minutes behind." The aggregate metrics showed plenty of spare capacity. Only when an engineer pulled *per-partition* lag did the picture resolve: that tenant was 35% of all events, all hashing to one partition, whose single consumer was permanently saturated while the other 47 idled at a fraction of capacity. The aggregate dashboard averaged the hot partition's huge lag against 47 near-zero lags into a comfortable-looking mean. The fix was a salted key for the few largest tenants. The lesson: **monitor per-partition, not per-topic**, because skew is invisible in aggregates, and a green topic dashboard can hide a permanently saturated partition.

## When to reach for this (and when not to)

Run the full capacity-planning worksheet when the topic is **keyed and ordering matters**, when it carries **meaningful or growing throughput**, and especially when it **feeds stateful stream processing** — these are the topics where getting the partition count wrong is expensive to fix and the up-front analysis pays for itself many times over. Any topic on the critical path of a product, any topic you expect to grow, any topic where a repartition would break correctness: size it deliberately.

Lean toward **more partitions** (generous within the healthy band) when ordering is required and future load is uncertain, because the cost of under-provisioning a keyed topic (an ordering-breaking repartition) dwarfs the modest tax of a few extra partitions. Lean toward **fewer partitions** when the topic is small, low-throughput, short-lived, or — crucially — *unkeyed*, because an unkeyed topic has no per-key ordering to break, so you can grow its partition count freely later with little consequence. Do not reflexively apply heavy headroom to every topic; a low-volume config-change topic does not need 96 partitions, and giving it that many just adds to the cluster's partition budget for no benefit.

Do **not** solve a slow-consumer problem with partitions alone. If the worksheet tells you to create 500 partitions because your consumer does 400 msg/s, stop and ask whether the right fix is a 10x faster consumer and 50 partitions. Partitions are cheap up to a point but not free, and a 500-partition topic driven by a needlessly slow consumer is a smell — fix the consumer (see [consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling)) and re-run the formula. And do **not** treat the partition count as a thing you will tune iteratively in production the way you tune a thread pool. It is not — it only goes up, every increase risks ordering, so it is a decision you make carefully once, not a knob you fiddle.

## Key takeaways

- **The partition is the unit of parallelism.** A consumer group can never run more useful consumers than there are partitions; the extra consumers sit idle. Partition count is the hard ceiling on consumer parallelism.
- **Size from measured per-partition throughput, both sides.** `partitions = max(ceil(target / per_producer_rate), ceil(target / per_consumer_rate))`. The consumer side almost always binds, because consuming is your business logic and producing is a cheap append.
- **Add headroom and round to a divisible number.** 1.5x to 2x over peak plus a growth allowance, rounded to a value with many even divisors so consumer groups assign evenly.
- **Too many partitions is a cluster-wide tax.** Longer rebalances, more memory and file handles, slower leader election and controller failover, higher latency from diluted batching, more replication streams — all scale roughly linearly with partition count, and none of it buys throughput past what you need.
- **Too few partitions fails suddenly.** A hard parallelism ceiling you hit at peak, no scaling headroom, amplified skew, and the only fix is the painful repartition. The failure mode is a backlog you cannot drain.
- **Repartitioning breaks per-key ordering.** Growing partitions remaps `hash(key) % N`, so keys jump partitions, per-key order shatters across the change, and stateful processors lose their per-key state. Kafka cannot even shrink partitions — the count is monotonic.
- **Over-provision up front, scale consumers later.** Because adding partitions is a correctness hazard and adding consumers is trivial, provision enough partitions at creation that all future scaling is the safe kind. Lean generous for keyed topics, lean lean for unkeyed ones.
- **Watch per-partition metrics, not topic aggregates.** Hot partitions from key skew hide behind green topic-level dashboards. Fix skew at the key (higher cardinality, salting), not by piling on partitions.
- **Run the worksheet before you create the topic.** Target, per-partition rates, formula, headroom, cluster-budget check, skew check, written rationale. The partition count is the one number you most want to get right the first time.

## Further reading

- [Message ordering and partitioning: the guarantees you actually get](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — which key to partition by, the companion to this post's "how many."
- [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — the assignment and rebalance machinery whose cost scales with partition count.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — pushing per-partition consumer throughput up so you need fewer partitions.
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — the same pre-split-for-growth logic in the database world.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the append-only log model underneath partitions.
- [Apache Kafka documentation: topic and partition configuration](https://kafka.apache.org/documentation/) — official guidance on partition counts, limits, and KRaft.
- Confluent, "How to choose the number of topics/partitions in a Kafka cluster" — the classic reference on the throughput-driven sizing approach.

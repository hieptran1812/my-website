---
title: "Anatomy of a Message System: Producers, Brokers, Consumers, Topics, and Offsets"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn the precise vocabulary of messaging from the ground up: what a producer, broker, topic, partition, record, offset, consumer group, and retention each actually are, and the subtleties — partition as the unit of both ordering and parallelism, the offset as a bookmark, retention versus acknowledgement — that separate someone who can spell Kafka from someone who can operate it."
tags:
  [
    "message-queue",
    "producers",
    "consumers",
    "partitions",
    "offsets",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "consumer-groups",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-1.webp"
---

If you have ever sat in a design review where someone confidently said "we'll just put it on a topic and the consumers will pick it up," and you nodded along while quietly not being sure whether a topic was a queue, a stream, a channel, or a folder, this post is for you. The words people throw around when they talk about messaging — producer, broker, topic, partition, offset, consumer group, retention, ack — are not interchangeable, and the differences between them are exactly the differences that decide whether your system scales, whether your messages stay in order, and whether you lose data the next time a broker dies. Most messaging confusion is not conceptual. It is vocabulary. People reason correctly once they have the right nouns.

So this is the vocabulary post. It is the second installment of a forty-part series, and it sits deliberately right after the opener on [why a message queue exists at all](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling). That post made the case for asynchronous messaging — decoupling in time, space, and rate. This one defines the parts. We are going to name every component of a message system precisely, draw it, and then go deep on each one until you understand not just what it is but what it is *for* and where it bites you. By the end you will be able to read any messaging system's documentation — Kafka, RabbitMQ, Pulsar, Redpanda, Amazon SQS, Google Pub/Sub — and map its idiosyncratic terminology onto a single mental model, because they are all variations on the same anatomy.

The figure below is that anatomy in one image. On the left, producers serialize and route records. In the middle sits the broker: a topic, split into partitions, each partition a replicated append-only log on disk. On the right, a consumer group reads, each member owning a slice of the partitions and tracking its own position. Hold this picture in your head; everything that follows is a zoom into one box of it.

![A full anatomy grid showing producers on the left feeding a partitioned topic in the broker which is replicated and stored on disk and read by a consumer group on the right](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-1.webp)

A word on framing before we start. I am going to define each term for a junior engineer — clear, no assumed knowledge — and then immediately push past the definition into the subtlety that a senior engineer needs. The definition of an offset is "the position of a record in a partition," which a beginner can memorize in five seconds. The *interesting* part — that there are three different offsets you must never confuse, that lag is the arithmetic difference between two of them, and that the committed offset is the only one that survives a crash — is what actually matters in production. We will do both, for every noun.

## 1. The cast of characters

Before we go deep, here is the whole cast in one paragraph, so the names are loaded into your head before we examine each. A **producer** is any program that writes messages. A **message** (or **record**) is the unit of data: a key, a value, some headers, and a timestamp. A **broker** is a server process that receives records, stores them, and serves them to readers; a set of brokers cooperating is a **cluster**. A **topic** is a named logical channel — the address you publish to and subscribe from. Each topic is divided into one or more **partitions**, and a partition is the real workhorse: an ordered, append-only log of records, and the atomic unit of both parallelism and ordering. Each record in a partition has an **offset**, a monotonically increasing integer that is its permanent address. A **consumer** is a program that reads records; consumers cooperate in a **consumer group** so that the partitions of a topic are divided among the group's members. As a consumer makes progress, it **commits** an offset — saves a bookmark — so that after a restart it resumes where it left off rather than from the beginning. And **retention** is the policy that decides how long the broker keeps a record on disk, which, crucially in a log-based system, is *independent of whether anyone has read it*.

That is the entire vocabulary. Nine nouns and two verbs. The rest of this post is each of them, slowly, with the edges that matter. Notice already how the components pair off into responsibilities: producers and consumers are the *clients* (your code), the broker is the *server* (someone's infrastructure); topics and partitions are the *addressing and layout*, offsets and retention are the *bookkeeping and lifetime*. When you debug a messaging problem, the first diagnostic move is always to ask which of these four pairs the symptom lives in — is this a client problem (a misconfigured producer or a slow consumer), a server problem (a broker or replica down), an addressing problem (wrong partition, hot key), or a bookkeeping problem (uncommitted offsets, expired retention)? Almost every messaging incident resolves cleanly into one of those four, and the rest of this post equips you to tell them apart. The taxonomy figure later in the post arranges these into the three families they fall into — the produce side, the broker, and the consume side — but for now just notice that the data flows strictly left to right through the anatomy: producers create records, brokers hold them, consumers read them, and offsets are how the consume side remembers its place across all of it.

One clarification that prevents a lot of grief: I will use Kafka-flavored terminology as the default because it is the most widely understood and because the log-based model exposes every concept explicitly. But the anatomy generalizes. RabbitMQ calls the broker a broker, the channel an exchange-plus-queue, and it has no offsets because it deletes on ack — yet it still has producers, a broker, logical channels, messages, and consumers. Where a system genuinely differs, I will call it out. The point of learning the anatomy is precisely that it lets you translate.

## 2. Producers: serialize, partition, batch, ack

A producer is the easy one to under-respect. "It's the thing that sends messages." True, and useless. A production-grade producer does four jobs, and each of the four is a place where you make a decision that has consequences hours later when something breaks. Let us take them in the order a record actually flows through the producer, which happens to be exactly the order of the first three stages in this lifecycle figure.

![A pipeline figure showing a record being serialized then routed to a partition then appended to the log then replicated then fetched then processed then having its offset committed](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-2.webp)

### Serialization: bytes are the only thing a broker understands

A broker does not store your `Order` object. It stores bytes. The first job of a producer is **serialization**: turning your in-memory object into a byte array (and a separate byte array for the key). This sounds like plumbing, and it is, but it is the plumbing that causes the most cross-team pain in messaging systems, because the consumer has to *deserialize* those same bytes back into an object, and the producer and consumer are usually different teams deploying on different schedules.

If the producer serializes with JSON and the consumer expects JSON, life is simple and slow. If you want compact, fast, schema-checked messages, you reach for Avro, Protobuf, or similar, and now you need a **schema registry** so that the meaning of the bytes is agreed upon and can evolve. The whole discipline of *schema evolution* — can I add a field without breaking old consumers? — lives here, in the serializer. We have a dedicated post in this series on schemas and serialization; for now, internalize that serialization is a producer responsibility and that the format you pick is a contract with every consumer, present and future.

```python
from confluent_kafka import Producer
import json

p = Producer({
    "bootstrap.servers": "broker1:9092,broker2:9092",
    "acks": "all",            # durability: wait for all in-sync replicas
    "enable.idempotence": True,  # no duplicate appends on retry
    "linger.ms": 10,          # batching: wait up to 10ms to fill a batch
    "batch.size": 65536,      # 64 KB batches
    "compression.type": "lz4",
})

def deliver_report(err, msg):
    if err is not None:
        print(f"delivery failed: {err}")
    else:
        print(f"ok -> {msg.topic()}[{msg.partition()}]@{msg.offset()}")

order = {"order_id": "A-4711", "customer_id": "C-99", "total_cents": 1299}
p.produce(
    topic="orders",
    key="C-99".encode(),                 # key = customer id -> ordering per customer
    value=json.dumps(order).encode(),    # value = the payload bytes
    callback=deliver_report,
)
p.flush()
```

### Partitioning: choosing which log to append to

A topic has multiple partitions. The producer must decide, for each record, *which* partition it goes to. This is **partitioning**, and it is one of the two or three most consequential decisions in the entire system, because the partition is the unit of ordering. The default rule in Kafka is: if the record has a key, the partition is `hash(key) % partitionCount`; if it has no key, the producer round-robins (or, in newer clients, uses a sticky partitioner that fills one batch before moving on, for efficiency).

Read that hash rule again, because it is the single most important sentence about ordering you will ever learn. **All records with the same key go to the same partition.** Since a partition is strictly ordered, this means all records with the same key are strictly ordered relative to each other. If you key by `customer_id`, every event for a given customer lands in one partition and is consumed in the order you produced it. If you key by `null`, your records scatter across partitions and you have no ordering guarantee between them at all. The key is not for lookup. The key is for **co-location and ordering**. That is its job. We go very deep on this in the [ordering and partitioning post](/blog/software-development/message-queue/ordering-and-partitioning); here, just bind the rule: same key, same partition, ordered.

### Batching: amortizing the network

A producer that sends one record per network round trip is a producer that wastes most of its time waiting. Real producers **batch**: they accumulate records destined for the same partition into a buffer and send them as one request. The two knobs are `linger.ms` (how long to wait for more records before sending) and `batch.size` (the maximum batch size). Setting `linger.ms=0` minimizes latency at the cost of throughput; setting it to 10–50 ms can multiply throughput several-fold because each network round trip and each broker append now carries hundreds of records instead of one. Batching is why Kafka producers can push hundreds of thousands of records per second per client: the per-record overhead is amortized to near zero. Compression piggybacks on batching — you compress the whole batch, so larger batches compress better.

### Acks: how durable is "sent"?

When `p.produce()` returns, has the record been saved? Not necessarily. The producer's **acknowledgement** setting (`acks`) controls when the broker tells the producer "I have it":

- `acks=0`: fire and forget. The producer does not wait for any acknowledgement. Fastest, and you can lose records silently if the broker drops them.
- `acks=1`: the leader replica has written the record to its log. The producer is told "got it" as soon as the leader has it — but if the leader crashes before a follower copies the record, it is gone.
- `acks=all` (or `acks=-1`): every in-sync replica has the record. Slowest, and the only setting that survives a single broker failure without data loss.

This is your first encounter with the durability knob, and it is a true trade-off: `acks=all` costs you latency (you wait for replication) in exchange for not losing data. Combine it with `enable.idempotence=true` and the producer also guarantees that a retry does not produce a duplicate append — the broker dedups by a producer-id-and-sequence-number scheme. The full menu of guarantees this enables — at-most-once, at-least-once, exactly-once — is the subject of the [delivery semantics post](/blog/software-development/message-queue/delivery-semantics-at-least-most-exactly-once); for the anatomy, just know that `acks` is the producer's lever on durability, and the table below summarizes the trade.

There is a fifth job hiding inside the producer that does not get its own headline but bites people constantly: the **in-flight buffer and its failure handling**. When you call `produce()`, the record does not go straight to the network. It goes into an in-memory accumulator (default 32 MB in the Java client, governed by `buffer.memory`). A background I/O thread drains that buffer to the brokers. This decoupling is what makes `produce()` non-blocking and fast — but it means a record can be sitting in your process's RAM, unacknowledged, when the process dies, and that record is simply gone. It was never durable; it never left your heap. The corollary is that a producer that does not call `flush()` (or await its delivery futures) before shutdown can drop the tail of its buffer on a graceful exit. And when the broker is unreachable and the buffer fills, the default behavior is to *block* the calling thread (up to `max.block.ms`) — so a broker outage can back-pressure all the way into your request-handling threads if you are producing synchronously in a request path. The buffer is the producer's shock absorber, and like any shock absorber it has a limit; know where it is.

Two more knobs round out the durable producer. `retries` (effectively infinite by default in modern clients) and `delivery.timeout.ms` (default 2 minutes) together govern how hard and how long the producer tries before it gives up and surfaces an error to your callback — a transient leader election or network blip should be retried transparently, a two-minute total outage should fail loudly so you can react. And `max.in.flight.requests.per.connection` interacts subtly with ordering: with idempotence off and this value above 1, a retried earlier batch can land *after* a later batch and reorder records within a partition; with idempotence on, the broker's sequence numbers preserve order even with up to five in-flight requests, which is why you should essentially always run idempotent producers. The producer is not "the thing that sends messages." It is a small, stateful, batching, retrying, ordering-preserving I/O engine, and every default it ships with is a position on a trade-off you are now responsible for.

| `acks` setting | Waits for | Latency | Data loss on broker crash | Use when |
|---|---|---|---|---|
| `0` | nothing | lowest | likely | metrics, lossy telemetry |
| `1` | leader write | low | possible (leader dies pre-replication) | logs, best-effort events |
| `all` | all in-sync replicas | higher | none (with `min.insync.replicas≥2`) | payments, orders, anything you cannot lose |

## 3. Brokers and the cluster: where messages live

A **broker** is a server process — a single running instance of the messaging software — that accepts records from producers, stores them on its local disk, and serves them to consumers. A **cluster** is a group of brokers that coordinate so that the system as a whole is bigger and more available than any single machine. When you connect a client, you give it a `bootstrap.servers` list; the client contacts one broker, learns the full cluster topology (which broker leads which partition), and then talks directly to the right broker for each partition. The bootstrap list is just a door in; it is not where all the data lives.

The broker's defining trick — the thing that makes log-based messaging fast — is that it does not maintain a fancy index or a per-message data structure. It appends records to the end of a file and serves reads as sequential scans from an offset. Appending to the end of a file and reading sequentially are the two things spinning disks and SSDs do fastest, and because the broker leans on the operating system's page cache, recently written records are usually served straight from RAM without a disk read at all. This is covered in depth in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log); the anatomy point is that a broker is fundamentally a manager of append-only log files, not a database with a query planner.

### Leaders, followers, and replicas

A partition does not live on one broker. For durability, each partition is **replicated** across several brokers. One replica is the **leader**; the others are **followers**. Every read and write for a partition goes through its leader. Followers do exactly one thing: they continuously copy the leader's log so that if the leader's broker dies, a follower can be promoted to leader and no data is lost. The set of replicas that are fully caught up with the leader is called the **in-sync replica set**, or **ISR**, and it is the linchpin of the whole durability story.

The figure below shows the write path for one partition with a replication factor of three. The producer writes to the leader; the leader appends and forwards to two followers; once the in-sync replicas have the record, the write is acknowledged. The `acks=all` setting from the producer section is exactly the instruction "do not acknowledge until the ISR has it." Conceptually, durability is a property of how many replicas have appended the record, not of the API call returning.

![A pipeline showing a producer write reaching the partition leader on broker one which replicates to followers on brokers two and three and is acknowledged once the in-sync replica set has the record](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-9.webp)

Two configuration values govern this. `replication.factor=3` says each partition has three copies. `min.insync.replicas=2` says a write with `acks=all` is acknowledged only if at least two replicas (including the leader) have it. Together they give you the classic safe setup: you can lose one broker and still accept writes; you can lose two brokers and the partition goes read-only (refusing writes) rather than risking data loss. If you set `replication.factor=3` but forget `min.insync.replicas`, it defaults to 1, and you have quietly built a system that acknowledges writes that only the leader has — which is `acks=1` durability wearing an `acks=all` costume. This exact misconfiguration has caused real data loss at real companies; we will name one in the case studies.

### Where the bytes physically sit

Inside a broker, a partition is stored as a directory of **segment** files. The active segment is the one currently being appended to; when it reaches a size or age threshold, it is rolled closed and a new segment is started. Alongside each segment is an offset index (mapping offset → byte position) and a time index (mapping timestamp → offset), so the broker can answer "give me everything from offset 50,000" or "give me everything since 9 a.m." with a fast lookup instead of a full scan. Retention, which we cover later, operates by deleting whole closed segments — which is why it is cheap. You do not delete records one at a time; you delete a file. This stacked physical structure is worth picturing as layers, which is exactly how the next figure presents the record itself.

Two performance facts about the broker are worth carrying around because they explain almost everything surprising about messaging performance. The first is **zero-copy**. When a consumer fetches records, the broker does not read the bytes into application memory, copy them around, and write them to the socket. It uses the operating system's `sendfile` system call to ship bytes directly from the page cache to the network socket, skipping the application's user-space buffers entirely. That is why a single broker can saturate a 10-gigabit network card serving fetches: the CPU barely touches the data. The second is the **page cache dependency**. The broker deliberately does *not* maintain its own large in-process cache of records; it relies on the OS page cache, which is also where freshly written records still live before they are flushed to disk. So a consumer that reads near the head of the log — the common case — is served from RAM, and a consumer that reads cold historical data forces actual disk reads. This is why a single lagging consumer replaying old data can degrade everyone: it evicts hot pages from the cache and turns warm reads into cold ones for the other consumers sharing the broker. The broker's speed is borrowed from the kernel, and its failure modes are the kernel's failure modes.

There is also a coordination layer most newcomers never see but every operator must respect: something has to remember *which broker leads which partition*, *what the current ISR is*, and *who is the controller*. Historically Kafka stored this in ZooKeeper; modern Kafka uses an internal Raft quorum called **KRaft** and stores cluster metadata in an internal log of its own. RabbitMQ uses its own clustering and (in current versions) the Raft-based quorum-queue machinery. The detail differs per system; the principle does not. A messaging cluster is not just data brokers — it is data brokers plus a consensus-backed metadata store that elects leaders and tracks membership, and when *that* layer is unhealthy, producers and consumers can stall even though every data broker is up. Leader election, controller failover, and metadata propagation are the quiet third character behind the producer and consumer.

## 4. Topics, queues, and partitions: the unit of parallelism and ordering

A **topic** is a named logical channel. "orders," "user-signups," "payment-events" — these are topics. Producers publish to a topic; consumers subscribe to a topic. The topic is the address. In RabbitMQ-style systems the analogous named channel is a **queue** (often fronted by an **exchange** that routes to it), and there is a real semantic difference we will get to. But the *role* is the same: a topic or queue is the logical name you publish to without knowing or caring which machine actually holds the bytes.

Here is the distinction that trips people up. In a classic **queue**, a message is delivered to one consumer and then removed — the queue is a buffer that drains. In a **topic** in the publish-subscribe and log sense, a message is published once and can be read by many independent subscribers, and reading does not remove it. The Kafka "topic" is the log-based pub/sub kind: durable, replayable, multi-subscriber. The RabbitMQ "queue" is the drain-on-consume kind. This queue-versus-topic-versus-log distinction is important enough that it gets its own post in the series; for the anatomy, hold onto: *a topic is a logical channel, and whether reading destroys the message depends on the model.*

It is worth pinning down how the queue-model brokers split the one responsibility that the log model folds into the topic. In RabbitMQ, a producer does not publish to a queue directly; it publishes to an **exchange**, and the exchange routes the message to zero or more queues according to **bindings** — rules that match the message's routing key against a pattern. A *direct* exchange routes by exact key match, a *topic* exchange routes by wildcard pattern (the source of endless confusion, because RabbitMQ's "topic exchange" is a routing feature, not the same thing as a Kafka topic), and a *fanout* exchange copies to every bound queue. So in the queue model, the logical-channel role is split into two objects: the exchange decides *where a message goes*, and the queue decides *who competes to consume it and when it is deleted*. The log model collapses both into the topic-plus-partition: the partitioner decides where, and retention plus per-group offsets decide who reads and how long it lives. Same anatomy, factored differently — which is exactly the kind of translation the vocabulary lets you do once you hold the parts straight.

### The partition is the real protagonist

A topic is a logical wrapper. The thing that actually does the work is the **partition**. A topic is divided into one or more partitions, and a partition is an ordered, append-only, immutable sequence of records. Two properties of a partition are the two most important facts in this entire post, so I will state them as a pair:

1. **A partition is the unit of ordering.** Records within a single partition are strictly ordered by offset, and a consumer reads them in that order, always. There is *no* ordering guarantee *across* partitions. None. If event X is in partition 2 and event Y is in partition 5, the system makes no promise about which a consumer sees first.
2. **A partition is the unit of parallelism.** Within a consumer group, a partition is consumed by at most one member at a time. So the maximum read parallelism of a topic is its partition count. Ten partitions means up to ten consumers working in parallel; an eleventh consumer in the group sits idle.

These two facts are in tension, and the tension is the central design pressure of every log-based system. You want *more* partitions for parallelism (throughput) and *fewer* partitions to keep related records ordered together. The resolution is the key: you partition by a key that defines your ordering boundary (per-customer, per-account, per-device), which gives you ordering *within* each key and parallelism *across* keys. You get both, as long as your ordering requirement is per-key and not global.

### Why partition count is a decision you live with

Partition count is not a knob you turn lazily. It is hard to decrease (you generally cannot, without recreating the topic) and increasing it changes the `hash(key) % N` mapping, which means a key that used to land in partition 3 now lands in partition 7 — breaking per-key ordering across the resize boundary for any in-flight data. So you size it up front, with headroom, and you size it for two things at once: throughput (more partitions, more parallel consumers) and ordering granularity (enough distinct keys that no single partition becomes a hot spot). Too few partitions and you cap your throughput and create hot partitions; too many and you pay overhead in open file handles, replication chatter, leader-election time, and end-to-end latency, and you can starve consumers that now each own a sliver. Let us put real numbers on this.

#### Worked example: partition-count sizing for 600 MB/s

You are designing a topic that must absorb a peak of **600 MB/s** of produced data. Through load testing, you have measured that a single partition, on your hardware with your replication factor and your consumer logic, sustains **30 MB/s** end to end before lag starts to grow. How many partitions do you provision?

The naive answer is `600 / 30 = 20` partitions. That is the *floor*, not the answer. At exactly 20 partitions you have zero headroom: any one partition running hot, any consumer GC pause, any single-broker slowdown, and you are behind. The disciplined sizing adds headroom for three things:

- **Skew headroom.** Keys are never perfectly uniform. If your hottest key carries, say, 1.6x the average load, your busiest partition needs 1.6x capacity. Multiply the floor by your measured skew factor. Call it 1.5x to be safe: `20 × 1.5 = 30`.
- **Growth headroom.** You said 600 MB/s today. If you expect to double in eighteen months and you cannot easily repartition without breaking ordering, provision for the future now: another 2x. `30 × 2 = 60`.
- **Consumer-count flexibility.** Your maximum consumer parallelism equals your partition count. With 60 partitions you can run anywhere from 1 to 60 consumers and scale smoothly; with 20 you are capped at 20.

So you provision **60 partitions**, not 20. The throughput math gives you the floor; skew, growth, and the desire for consumer headroom give you the real number. Notice the ceiling check too: 60 partitions × 30 MB/s = 1.8 GB/s of *capacity*, comfortably above 600 MB/s of demand, with room for a 3x spike. And confirm you have enough distinct keys: if you only have 40 distinct customer ids producing traffic, 60 partitions is pointless because 20 of them will always be empty — partition count should not exceed your key cardinality by much. Sizing partitions is throughput-floor times skew times growth, bounded above by key cardinality and broker overhead.

The upper bound deserves a real number too, because "more partitions is always better" is a trap. Each partition costs the cluster: an open file handle (often several, for the segment and its indexes), a slice of replication traffic, a position in the metadata that must be propagated on every change, and — the expensive one — a unit of work during failover. When a broker dies, every partition it led must elect a new leader, and leader election is roughly linear in the number of partitions. A cluster with hundreds of thousands of partitions can take many seconds to fail over, during which producers and consumers for those partitions stall. As a rule of thumb on a healthy modern cluster, a few thousand partitions per broker is comfortable and tens of thousands per broker is where failover time starts to hurt. So the full sizing statement is: provision enough partitions to meet throughput with skew and growth headroom, but not so many that failover latency and per-partition overhead dominate, and never more than your key cardinality can fill. For our 600 MB/s topic, 60 partitions sits comfortably inside every one of those bounds.

One more partition subtlety that the throughput math hides: **partition count interacts with end-to-end latency through batching**. A producer batches per partition. If you spread the same record rate across more partitions, each partition's batch fills more slowly, so either you wait longer (higher `linger.ms` latency) for a batch to fill or you send smaller, less efficient batches. Past a point, adding partitions actively hurts the latency and efficiency of a fixed-rate producer. This is the counterintuitive twist that catches people who treat partition count as a free throughput dial: it is free for *consumer* parallelism and *not* free for *producer* batching efficiency, and the optimum balances both.

## 5. The message itself: key, value, headers, timestamp

We have been saying "record" and "message" interchangeably, and now we open the box. A record is not just a blob. It is a small structured envelope, and each field has a distinct job. The figure shows the structure as layers; let us walk them from the outside in.

![A stack figure showing a record as layers with headers on top then the key then the value payload then the timestamp and finally the broker-assigned offset at the bottom](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-3.webp)

### Value: the payload

The **value** is the actual content — the serialized bytes of your event or command. The order details, the sensor reading, the click event. This is what your consumer ultimately processes. The broker treats it as opaque bytes; it neither knows nor cares what is inside. Everything else in the record is metadata that helps the system route, order, and reason about that payload.

### Key: routing and ordering, not lookup

We met the **key** in the producer section, but it deserves its own paragraph because beginners consistently misunderstand it. The key is *not* a primary key for lookup — you cannot ask the broker "give me the record with key C-99." The key has exactly two jobs. First, **partitioning**: `hash(key)` decides the partition, so the key determines co-location and therefore ordering. Second, in log-compacted topics, the key is the **compaction identity**: the broker keeps only the latest value per key, turning the log into a changelog or a snapshot of current state (this is how Kafka's `__consumer_offsets` and Kafka-as-a-database patterns work, and it underpins the [change data capture and outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern)). A null key means "I don't care about ordering or compaction for this record" — it scatters round-robin. Choosing the key is choosing your ordering and locality boundary; it is the single most semantically loaded field in the record.

### Headers: out-of-band metadata

**Headers** are key-value pairs of metadata that travel with the record but are not part of the value payload. They are where you put things the *infrastructure* cares about without forcing the *business* payload to carry them: a trace id for distributed tracing, a schema version, a content-type, a tenant id for routing, a "this is a retry, attempt 3" counter. Headers let middleware (interceptors, dead-letter routers, tracing systems) read and act on metadata without deserializing the value, which is both faster and cleaner. Not every system has them — they were added to Kafka relatively late and RabbitMQ has long had them as properties — but where they exist, they are the right home for cross-cutting metadata.

### Timestamp: which clock?

Every record carries a **timestamp**, and here is the subtlety: *which* timestamp? There are two candidates. **Event time** is when the thing actually happened (the producer sets it, often from the event itself). **Append time** (or log-append time, or ingestion time) is when the broker wrote the record. Kafka lets you configure, per topic, which one the timestamp field holds. This matters enormously for stream processing: windowed aggregations ("count clicks per minute") give different answers depending on whether you bucket by when the click happened or when it arrived, and the gap between the two — caused by network delay, buffering, retries, and clock skew — is the source of the entire field of watermarks and late-data handling in stream processors. For the anatomy, internalize: the timestamp is real, it has two possible meanings, and you must know which one your topic is configured for before you build time-based logic on it.

### Offset: assigned last, by the broker

The **offset** is in the figure as the bottom layer because, unlike the others, it is *not set by the producer*. The producer sends headers, key, value, and (optionally) timestamp. The broker, on append, assigns the offset: the next integer in that partition's sequence. The offset is the record's permanent, immutable address within its partition — record number 4,711 in partition 2 of topic "orders." It is dense (no gaps in normal operation), monotonically increasing, and per-partition (offset 4,711 in partition 2 is a completely different record from offset 4,711 in partition 5). Because the offset is assigned by the broker and is stable forever, it is the perfect bookmark — which is the entire subject of the next section.

## 6. Offsets and the consumer's bookmark (committed vs latest vs lag)

This is the section where careful people pull ahead of careless people, because "offset" is one word for three different things, and conflating them is the source of more messaging bugs than any other single confusion. Let me name the three offsets precisely.

- The **latest offset** (also called the **log-end offset** or **high water mark**) is the offset of the *next* record that will be written to the partition. It is the position of the head of the log — how far the producers have gotten. It climbs as producers write.
- The **current position** (or **read position**) is where *this consumer* will read next. It is the consumer's in-memory pointer, advancing as it polls records. It is *not* durable — it lives in the consumer's memory.
- The **committed offset** is the last position this consumer *saved* — wrote back to the broker (or to whatever offset store it uses). It is durable. After a crash and restart, the consumer resumes from the committed offset, *not* from its in-memory current position, which is gone.

The relationship is `committed ≤ current ≤ latest`, almost always. The consumer has read up to `current` but has only promised (committed) up to `committed`, and the producers have written up to `latest`. The gap that everyone cares about is between the last two.

### Lag: the single most important consumer metric

**Consumer lag** is defined as `latest_offset − committed_offset`. It is the number of records that have been produced but not yet committed as consumed — the backlog. Lag is the heartbeat of a streaming system. Lag of zero means your consumer is fully caught up. Lag that is stable and small means your consumer is keeping pace. Lag that *grows* means your consumer is falling behind, and if it grows long enough it can hit the retention cliff (records get deleted before you read them — irreversible loss). If you monitor exactly one thing about your consumers, monitor lag, and alert on its *trend*, not just its absolute value. The timeline figure makes the relationship visceral: the produced offset climbs steadily, the committed offset trails behind, and the vertical gap between the two lines is lag.

![A timeline figure showing the latest produced offset climbing over time while the committed offset trails behind it with the widening gap between them labeled as growing lag before the consumer catches up](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-7.webp)

#### Worked example: computing lag and time-to-drain

Your monitoring shows a partition with **latest offset = 9,400,000** and your consumer group's **committed offset = 9,100,000**. Your consumers are processing at a steady **5,000 records/second**, and producers are writing at **3,000 records/second**.

First, the lag right now:

```
lag = latest − committed = 9,400,000 − 9,100,000 = 300,000 records
```

You are 300,000 records behind. Is that bad? It depends on how fast you can close the gap. The mistake people make is computing `300,000 / 5,000 = 60 seconds` and declaring victory. That is wrong, because *while you are draining, producers keep writing*. Your net drain rate is consume rate minus produce rate:

```
net drain rate = consume − produce = 5,000 − 3,000 = 2,000 records/second
time to drain = lag / net drain rate = 300,000 / 2,000 = 150 seconds
```

It will take **150 seconds** (2.5 minutes), not 60, to catch up, because the backlog is also being fed. Now the critical sanity check: what if producers were writing *faster* than you consume — say 6,000/second against your 5,000/second? Then your net drain rate is `5,000 − 6,000 = −1,000` records/second. The lag never drains. It grows by 1,000 records every second, forever, until you either add consumers (up to the partition count) or speed up per-record processing. Time-to-drain is lag divided by the *net* rate, and when produce exceeds consume there is no finite drain time — that is the precise condition under which you must scale out. This is also why lag alerting must be on the *derivative*: a flat lag of 300,000 is fine; a lag of 50,000 that is climbing by 1,000/second is an incident in progress.

### Auto-commit is a foot-gun worth understanding

Where does the committed offset actually live? In modern Kafka it is stored *in the broker*, in a special compacted internal topic called `__consumer_offsets`, keyed by (group, topic, partition). That is an elegant piece of self-reference: the consumer's bookmark is itself a record in a log, compacted so only the latest position per key survives — the consumer offset machinery is built out of the same anatomy this whole post describes. Older systems and some clients let you store offsets externally (in your own database, alongside the side effects of processing) precisely so that the offset commit and the business write can share a transaction and become atomic — which is one route to exactly-once processing. Whether the offset store is the broker or your database, the role is identical: it is the durable home of "how far has this group gotten."

The fact that the offset is just data you control also means you can **seek**. A consumer can call `seek(partition, offset)` to jump to any position, `seekToBeginning` to replay from the start, or `seekToEnd` to skip to the head and ignore the backlog. Operationally, the `auto.offset.reset` config decides what happens when a group has *no* committed offset (a brand-new group, or one whose offsets aged out): `earliest` starts from the oldest retained record (replay everything), `latest` starts from the head (only new records), and `none` throws an error so a human decides. Choosing `earliest` versus `latest` for a new consumer group is a surprisingly consequential one-line decision — pick `earliest` by accident on a topic with 30 days of retention and your "new" consumer will dutifully reprocess a month of history on first boot, which is sometimes exactly what you want and sometimes a self-inflicted thundering herd.

Most clients offer **auto-commit**: every few seconds the consumer automatically commits its current position. It is convenient and it is a classic source of subtle data loss or duplication. If auto-commit fires *after* you have polled records but *before* you have finished processing them, and you then crash, the offset is committed past records you never actually processed — they are lost (at-most-once). If you commit *after* processing and crash mid-batch, you reprocess (at-least-once). The choice between committing before or after processing, and whether to commit synchronously or asynchronously, is exactly the choice of delivery semantics, which is why this anatomy post keeps pointing at the [delivery semantics deep-dive](/blog/software-development/message-queue/delivery-semantics-at-least-most-exactly-once). For now: the committed offset is a promise, committing is the act of making the promise durable, and *when* you commit relative to *when* you process is the whole ballgame.

## 7. Consumers and consumer groups: competing consumers and assignment

A **consumer** is a program that reads records from one or more partitions, processes them, and commits offsets. A single consumer reading a multi-partition topic works fine until it cannot keep up — one process, one set of CPUs, one throughput ceiling. The fix is the **consumer group**: several consumer instances that share a group id and cooperatively divide the partitions among themselves. The before-and-after figure shows the leap. A single consumer must read every partition itself and becomes the bottleneck; a group of consumers each takes a slice and reads in parallel, and the backlog drains many times faster.

![A before and after figure contrasting one consumer reading all three partitions as a bottleneck against a consumer group of three where each member owns one partition and reads in parallel](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-5.webp)

### The competing-consumers pattern

The mechanism a consumer group implements is the classic **competing consumers** pattern: a pool of workers draws from a shared source so that work is spread across them, and adding workers adds throughput. But log-based competing consumers have a twist that queue-based competing consumers do not. In a plain queue (RabbitMQ), any free worker can grab the next message, so the unit of competition is the *message*. In a log (Kafka), the unit of competition is the *partition*: each partition is assigned to exactly one member of the group, and that member reads it exclusively. This is what preserves per-partition ordering even with many consumers — because only one consumer ever touches a given partition at a time, the order in that partition is never scrambled by concurrent readers.

The consequence is the rule we hit in the partition section, now from the consumer's side: **the number of useful consumers in a group is capped at the partition count.** Three partitions, three consumers, perfect one-to-one assignment. Add a fourth consumer to a three-partition topic and it gets *zero* partitions — it idles, a hot spare. This is why partition count is a throughput ceiling: it is literally the maximum number of group members that can do work. The assignment figure below shows the clean three-into-three case.

![A graph figure showing a topic with three partitions where each partition is assigned to exactly one consumer in the group giving a clean one to one mapping of partition to consumer](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-4.webp)

### Partition assignment and rebalancing

When a consumer joins or leaves the group — a deploy, a crash, an autoscale event — the group must **rebalance**: redistribute partitions among the current members. A coordinator (one of the brokers) runs the assignment, using a strategy (range, round-robin, sticky, cooperative-sticky) to decide who gets what. Rebalancing is necessary and it is also a performance hazard. In the older "eager" protocol, every member *stops consuming*, gives up all its partitions, and waits for a fresh assignment — a stop-the-world pause that, on a large group with slow members, can last seconds and shows up as a lag spike every time you deploy. Modern **cooperative rebalancing** moves only the partitions that need to move, so most members keep working. Tuning `session.timeout.ms` and `max.poll.interval.ms` so that a member doing legitimate slow work is not falsely declared dead — which triggers a rebalance — is one of the everyday craft skills of operating consumers.

```java
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092");
props.put("group.id", "order-processors");        // the consumer group
props.put("enable.auto.commit", "false");          // commit manually for control
props.put("max.poll.records", "500");              // batch size per poll()
props.put("partition.assignment.strategy",
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");

KafkaConsumer<String, byte[]> consumer = new KafkaConsumer<>(props);
consumer.subscribe(List.of("orders"));

while (true) {
    ConsumerRecords<String, byte[]> records = consumer.poll(Duration.ofMillis(200));
    for (ConsumerRecord<String, byte[]> r : records) {
        process(r.value());                        // do the work first
    }
    consumer.commitSync();                          // then commit -> at-least-once
}
```

### Many groups, one topic

### Inside the poll loop

The consumer's heartbeat is `poll()`, and it does more than fetch records. A single `poll()` call does four things on your behalf: it sends and receives fetch requests for the partitions you own, it sends heartbeats to the group coordinator (in the consumer's background thread in modern clients, but historically piggybacked on poll) so the coordinator knows you are alive, it participates in any pending rebalance, and it returns a batch of records bounded by `max.poll.records`. This is why the cardinal rule of consumer programming is **call poll often and do not block the poll thread for too long**. If your per-record processing of a `max.poll.records=500` batch takes longer than `max.poll.interval.ms` (default 5 minutes), the coordinator concludes your consumer has hung, evicts it from the group, and triggers a rebalance — even though your consumer is busy doing real work. The classic incident is a consumer that does a slow synchronous call per record (a database write, an external API), processes 500 of them, blows past the interval, gets kicked, rejoins, gets the same batch reassigned, and slows down again — a feedback loop that looks like the consumer is "stuck" when it is actually being repeatedly evicted for being slow. The fixes are all about respecting the poll cadence: shrink `max.poll.records`, raise `max.poll.interval.ms`, or move slow work off the poll thread.

The fetch itself is also tunable, and the tuning is the consumer-side mirror of producer batching. `fetch.min.bytes` tells the broker "do not answer my fetch until you have at least this many bytes, or until `fetch.max.wait.ms` elapses." Raising it trades a little latency for far better throughput and far less CPU on both sides, because each fetch round trip carries more data. `max.partition.fetch.bytes` caps how much one fetch pulls per partition, which matters when records are large or unevenly sized. These knobs rarely make a demo faster but routinely decide whether a production consumer keeps up at 3 a.m., and they are the consumer-side answer to the same question the producer's `linger.ms` answers: how much do you batch to amortize the network?

The other half of the consumer-group story is independence *between* groups. Two different consumer groups subscribed to the same topic each get the *full* stream, independently, with their own offsets. The "billing" group and the "analytics" group both read every order, at their own pace, without affecting each other, because the broker does not delete on read — it just tracks each group's committed offsets separately. This is the log model's superpower over the queue model: one published record, fanned out to N independent subscribers, each replayable on its own. It is also why you cannot ask "has this message been consumed?" in a log — consumed by *whom*? Each group has its own answer, and that is exactly the property that lets you add a new consumer group next year that replays the last 30 days of history to backfill a new service.

## 8. Retention: why a message can outlive its consumption

Here is the idea that most cleanly separates a queue from a log, and it is the one beginners coming from queues find most surprising. In a log-based system, **reading a record does not delete it.** A consumer reading offset 4,711 does not remove offset 4,711; the record sits there, and the consumer simply advances its bookmark to 4,712. The record is deleted only when **retention** says so — and retention is a policy about *time or size*, completely independent of whether anyone has read it.

There are two retention modes, and they can be combined:

- **Time-based retention** (`retention.ms`): keep records for N hours/days, then delete. "Keep 7 days of orders." A record produced today is deleted in 7 days whether it was read zero times or a thousand times.
- **Size-based retention** (`retention.bytes`): keep at most N bytes per partition, deleting the oldest segments when the cap is exceeded. "Keep at most 50 GB per partition." This bounds disk usage regardless of message rate.

When both are set, whichever triggers first wins — the broker deletes a segment when it exceeds *either* the age or the size limit. Deletion happens at the granularity of whole closed segments, which is why it is cheap: the broker unlinks a file, it does not walk individual records.

### Retention versus acknowledgement: the distinction that matters

This is the conceptual crux, so let me state it as sharply as I can. In a **queue**, deletion is tied to **acknowledgement**: a consumer acks a message and the broker removes it. The message's lifetime is "until someone successfully processes it." In a **log**, deletion is tied to **retention**: a record's lifetime is "until the time or size policy expires it," and acknowledgement (the offset commit) only moves a *bookmark* — it never deletes anything. The matrix figure earlier in this post assigns durability to the broker and acking to the producer and consumer for exactly this reason: in a log, the broker's retention owns the data's lifetime, and the consumer's commit owns only its own position within that data.

The practical consequences of log retention are large and mostly good:

1. **Replay.** Because data persists after consumption, a consumer can rewind — reset its committed offset backward — and reprocess history. Fix a bug in your consumer, redeploy, seek to where the bug started, reprocess. You cannot do this in a delete-on-ack queue; the data is gone.
2. **Multiple independent consumers.** As we saw, many groups read the same retained data at their own pace. Retention, not consumption, governs availability.
3. **A retention cliff.** The flip side: if your consumer's lag grows until the oldest unread record ages out of retention, that record is *deleted before you read it* — permanent, silent data loss. This is the single scariest failure mode of log-based systems and the reason lag monitoring is non-negotiable. Your effective time budget to recover a stalled consumer is exactly `retention period − current lag in time units`.

### Log compaction: a third retention mode

There is a special third mode worth naming: **log compaction**. Instead of deleting by age or size, a compacted topic keeps the *latest record per key* indefinitely and garbage-collects older records *with the same key*. The log becomes a snapshot of "current value per key" — a changelog or a materialized table. This is how Kafka stores consumer offsets, how Kafka Connect stores connector state, and how the [outbox and change-data-capture patterns](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) turn a stream of changes into a queryable current state. Compaction is the bridge between "log of events" and "table of current state," and it is why people say a log and a table are two views of the same thing.

#### Worked example: retention window and disk sizing

You want a topic to retain **7 days** of data and you need to know how much disk the cluster must reserve. Your numbers: average produce rate **50,000 records/second**, average record size on disk **500 bytes** (after compression), **replication.factor = 3**.

Start with the raw data written per second for one copy:

```
bytes/sec (1 copy) = 50,000 rec/s × 500 bytes = 25 MB/s
```

Over a 7-day retention window:

```
seconds in 7 days = 7 × 24 × 3600 = 604,800 s
one copy = 25 MB/s × 604,800 s ≈ 15.1 TB
```

But you keep three replicas, so the cluster holds three copies:

```
total = 15.1 TB × 3 ≈ 45.4 TB
```

So the *cluster* must reserve roughly **45 TB** to retain 7 days at this rate. Spread across, say, 6 brokers that is about 7.5 TB per broker — before you add the operational headroom you always want (never run a broker's disk past ~70% full, or compaction and segment rolling start to misbehave), which pushes the practical reservation to about 11 TB per broker. Two lessons fall out of this arithmetic. First, **retention is a disk-budget decision, not a free dial** — doubling retention to 14 days doubles your storage bill, and the multiplier is the replication factor, so a `replication.factor=3` cluster pays for three days of disk for every one day of retention you grant. Second, this gives you the *time* dimension of the retention cliff precisely: at 50,000 records/second, 7 days of retention is about 30 billion records of runway, and your stalled-consumer recovery budget is `7 days − (current lag converted to time)`. If a consumer is 2 days behind in time terms, you have 5 days to fix it before the oldest unread record is deleted. Retention in records, retention in bytes, and retention in *time-to-recover* are three readings of the same policy, and the operator must be able to convert between them on the fly.

## 9. Putting it together: a record's full lifecycle

We have defined every noun. Now let us trace one single record from birth to retirement through all of them, in order, and you will see that each stage corresponds to exactly one component we defined. This is the lifecycle pipeline figure again, and this time you have the vocabulary to read every box.

![A pipeline figure tracing a record through serialize then partition then append then replicate then fetch then process then commit offset showing the seven stages of a record lifecycle](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-2.webp)

1. **Serialize.** Your application calls `produce()` with an `Order` object and the key `C-99`. The producer's serializer turns the object into value bytes and the customer id into key bytes. (Section 2.)
2. **Partition.** The producer computes `hash("C-99") % 60` and gets, say, partition 17. Every record for customer C-99 will go to partition 17, so customer C-99's events stay ordered. (Sections 2 and 4.)
3. **Batch and send.** The record joins a batch destined for partition 17. After `linger.ms` or when the batch fills, the producer sends the batch to the broker that leads partition 17. (Section 2.)
4. **Append.** The leader broker appends the record to the active segment of partition 17's log and assigns it offset 4,711. (Sections 3, 5.)
5. **Replicate.** The leader forwards the record to the two follower replicas. Once the in-sync replica set has it, and because the producer set `acks=all`, the broker acknowledges the write back to the producer. The record is now durable. (Sections 2, 3.)
6. **Fetch.** The consumer in the "order-processors" group that owns partition 17 calls `poll()`, fetches a batch starting at its current position, and receives record 4,711. (Sections 6, 7.)
7. **Process.** The consumer deserializes the value, processes the order — charges the card, updates the database. (Sections 5, 7.)
8. **Commit.** Having processed up to 4,711, the consumer commits offset 4,712 (the next position to read). This is its durable bookmark; after a crash it resumes here. (Section 6.)
9. **Retire.** Seven days later (if `retention.ms` is 7 days), the segment containing 4,711 ages out and the broker deletes the whole segment — long after the consumer read it, and only because retention, not consumption, ended the record's life. (Section 8.)

Every stage maps to one box of the anatomy grid we opened with, and every component we defined shows up exactly once. That is the payoff of learning the vocabulary precisely: the lifecycle stops being a mystery and becomes a checklist. The taxonomy below organizes the same nouns into the three families they belong to — produce side, broker, consume side — which is the mental filing cabinet you should keep.

![A tree figure showing the taxonomy of a message system with producer broker and consumer as the three branches and partitioner topics replicas and consumer groups as their children](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-8.webp)

And because the recurring confusion in design reviews is *who is responsible for which guarantee*, here is the responsibility matrix made explicit. Ordering is a shared concern (the producer picks the key, the broker keeps a partition ordered, the consumer reads in order); durability lives in the broker's replication; scaling comes from partition count and group size; acking is split between the producer's `acks` and the consumer's commit.

![A matrix figure mapping producer broker and consumer against the responsibilities of ordering durability scaling and acking showing each component owns a distinct guarantee](/imgs/blogs/anatomy-of-a-message-system-producers-brokers-consumers-6.webp)

## Case studies and war stories

Vocabulary is abstract until it costs someone an outage. Here are four times a confused noun caused real damage, each pinned to a specific term we defined.

### The `min.insync.replicas` default that ate the data

A team ran a three-broker cluster with `replication.factor=3` and producers configured `acks=all`. They believed they had bulletproof durability — three copies, wait for all of them. During a rolling broker upgrade, one broker was down (expected), and a second broker had a disk hiccup and fell out of the in-sync replica set. With two replicas now out, the partition had an ISR of one — just the leader. But `min.insync.replicas` had been left at its default of **1**. So `acks=all` happily acknowledged writes that only the *leader* had. When the leader's broker then restarted as part of the rolling upgrade before the followers caught up, the un-replicated writes vanished. The lesson is a vocabulary lesson: `acks=all` means "all *in-sync* replicas," and if you do not pin `min.insync.replicas` to at least 2, the in-sync set can shrink to one and `acks=all` silently degrades to `acks=1`. Durability is a property of the ISR size, not of the `acks` label.

### The single-key hot partition

An analytics pipeline keyed its events by `environment` so that all `prod` events stayed ordered. There were three environment values, so despite the topic having 48 partitions, all production traffic — 95% of the volume — hashed to a *single* partition. That one partition was pinned at 100% while 47 others sat near idle, the owning consumer fell behind, and lag on that partition climbed toward the retention cliff while the dashboard's *average* lag looked fine. The bug is a key-cardinality bug: the key controls partitioning, and a low-cardinality key collapses your parallelism to the number of distinct key values, no matter how many partitions you provisioned. They rekeyed by `event_id` (high cardinality) with `environment` moved to a header for filtering, and the load spread evenly. Partition count is a ceiling; key cardinality is what determines whether you actually reach it.

### The rebalance storm on every deploy

A service with a 40-member consumer group used the default eager rebalancing protocol and ran long-running per-record work that occasionally exceeded `max.poll.interval.ms`. Every deploy, and every time a slow record tripped the poll-interval timeout, the *entire* group stopped, surrendered all partitions, and rebalanced — a stop-the-world pause that spiked lag and occasionally cascaded as the rebalance itself took long enough to trip more timeouts. The fix was two vocabulary-driven changes: switch to the cooperative-sticky assignor so only moved partitions pause, and raise `max.poll.interval.ms` (and shrink `max.poll.records`) so legitimate slow work is not mistaken for a dead consumer. Rebalancing is the price of dynamic group membership; understanding what triggers it is how you stop paying it on every deploy.

### The team that thought the queue was empty

A team migrated from RabbitMQ to Kafka and a week later asked, alarmed, "why is our topic full of millions of old messages — the consumers already processed them, why didn't they get deleted?" Nothing was wrong. They had brought a *queue* mental model (delete on ack) to a *log* (delete on retention). In Kafka, processed records stay until retention expires; the committed offset moved forward, but the data remained, exactly as designed — which is what made replay and adding a second consumer group possible. The "bug" was a vocabulary mismatch: in a log, acknowledgement moves a bookmark, it does not delete a message. Once they internalized retention-versus-ack, they stopped seeing the retained data as a leak and started using it as a feature.

### The reset that replayed a million emails

An on-call engineer, trying to clear what looked like stuck consumers, ran an offset-reset command to "skip past the bad messages." They reached for `auto.offset.reset=earliest` thinking it meant "reset to a sane recent point," when it actually means "if there is no committed offset, start from the *oldest* retained record." Combined with a manual offset deletion they had done moments earlier, the group now had no committed offset and dutifully restarted from the beginning of a topic with 14 days of retention — replaying two weeks of `send-email` commands to a downstream consumer that was not idempotent. Roughly a million duplicate emails went out before someone paused the group. Three distinct vocabulary failures stacked up here: confusing *committed offset* (the durable bookmark) with *current position*; misreading what `auto.offset.reset` controls (the no-committed-offset case, not a live rewind); and consuming non-idempotently from an at-least-once log. The fix that mattered most long-term was not the offset surgery — it was making the email consumer idempotent so that replay, which is a *feature* of the log model, could never again become a weapon. Replay is only safe when your consumers treat reprocessing as a no-op; the offset is a loaded gun precisely because the data persists.

## When to reach for this model (and when not to)

The anatomy we just defined — partitioned, replicated, retained logs with consumer groups and offsets — is one of two dominant shapes in messaging. The other is the routed, delete-on-ack queue. Knowing the vocabulary is precisely what lets you choose between them, so here is the decisive guidance.

**Reach for the log model (Kafka, Pulsar, Redpanda, Kinesis) when:**

- You need **replay** — the ability to reprocess history after a bug fix or to backfill a new consumer. Retention-not-consumption is the whole reason this works.
- You have **multiple independent consumers** of the same stream, each at its own pace. Independent committed offsets per group is the enabling mechanism.
- You need **high throughput** with **per-key ordering** — keyed partitioning gives you both, scaling by partition count.
- You are building **event sourcing, CQRS, stream processing, or CDC**, all of which assume a durable replayable log.

**Reach for the queue model (RabbitMQ, SQS, classic brokers) when:**

- You want **per-message competing consumers** with the finest-grained work distribution — any free worker grabs the next message, no partition-level stickiness.
- You need **complex routing** — fan-out by pattern, priority queues, topic exchanges, dead-letter routing — which the [RabbitMQ production architecture post](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) covers in depth.
- Your messages are **commands to be done once and forgotten**, where delete-on-ack is exactly the semantics you want and replay would be meaningless or harmful.
- You want **lower operational ceremony** for moderate volumes and do not need a retained history.

The single best predictor: **do you need to replay or have multiple independent readers?** If yes, the log model's retention-and-offsets anatomy is what you want. If your messages are fire-once commands and you value routing flexibility over replayability, the queue model fits better. Most large systems end up running both, for different workloads, and the engineers who thrive are the ones who can name precisely which anatomy each workload needs.

## Key takeaways

- **The vocabulary is the model.** Producer, broker, topic, partition, record, offset, consumer, consumer group, retention, ack — get these nouns precise and most messaging confusion evaporates. Every system (Kafka, RabbitMQ, Pulsar, SQS) is a variation on this one anatomy.
- **The key is for partitioning and ordering, not lookup.** Same key, same partition, strictly ordered. A null key scatters and gives no cross-record ordering. Low-cardinality keys collapse your parallelism into hot partitions.
- **The partition is both the unit of ordering and the unit of parallelism.** There is no ordering across partitions, and your maximum consumer parallelism equals your partition count. Size partitions for throughput floor times skew times growth, bounded by key cardinality.
- **There are three offsets, not one.** Latest (head of log), current (in-memory read position), committed (durable bookmark). Lag = latest − committed, and you drain it at the *net* rate of consume minus produce — which is negative, and unrecoverable without scaling, when producers outpace consumers.
- **`acks=all` means all *in-sync* replicas.** Without `min.insync.replicas≥2`, the ISR can shrink to one and your `acks=all` silently becomes `acks=1`. Durability is a property of ISR size, not the `acks` label.
- **In a log, reading does not delete.** Retention (time or size), not acknowledgement, ends a record's life. That gives you replay and multiple independent consumers — and a retention cliff if lag ever exceeds the retention window.
- **A consumer group divides partitions; multiple groups each get the full stream.** Competing consumers within a group, independent replay across groups. Rebalancing is the cost of dynamic membership — use cooperative assignment to avoid stop-the-world pauses.
- **Acknowledgement is a producer-and-consumer concern; durability is a broker concern.** The producer's `acks` and the consumer's commit move promises and bookmarks; the broker's replication and retention own the data's actual lifetime.

## Further reading

- [What is a message queue? Async, decoupling, and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) — the series opener that motivates why any of this anatomy exists.
- [Delivery semantics: at-least-once, at-most-once, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-least-most-exactly-once) — the deep dive on acks, commits, and what "delivered" really guarantees.
- [Ordering and partitioning](/blog/software-development/message-queue/ordering-and-partitioning) — how the key and the partition combine to give you per-key order at scale.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the broker internals: segments, page cache, sequential I/O, and the append-only log model.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — how compacted logs turn a stream of changes into queryable current state.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — the queue-model counterpart: exchanges, routing, and delete-on-ack semantics.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the replication theory behind partition leaders, followers, and the ISR.
- The official Apache Kafka documentation, sections on Producer, Consumer, Topics, and Replication, for the authoritative configuration reference.

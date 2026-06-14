---
title: "Kafka as a Distributed Log: The Database Turned Inside Out"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Kafka is not a queue but a distributed, partitioned, replicated commit log, and once you see it as a log you can rebuild caches, search indexes, warehouses, and stream processors as replayable materialized views of one ordered stream."
tags:
  [
    "kafka",
    "distributed-log",
    "event-streaming",
    "stream-processing",
    "replication",
    "kraft",
    "exactly-once",
    "log-compaction",
    "distributed-systems",
    "databases",
    "event-sourcing",
    "cdc",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/kafka-as-a-distributed-log-1.webp"
---

The single most expensive misconception I have watched teams carry into a Kafka project is that Kafka is a message queue. They reach for it the way they would reach for RabbitMQ or SQS: a producer puts a message in, a consumer takes a message out, and once the message is taken it is gone. They size the cluster for that model, they reason about failures with that model, and then three months later they are paged because a consumer fell behind, a second team wanted to read the same data, someone needs to reprocess last Tuesday's events after a bug fix, and the queue mental model has nothing useful to say about any of it. A queue is a place messages pass through. Kafka is a place messages *stay*. That difference is not a feature; it is the entire design, and everything that makes Kafka strange and powerful falls out of it.

The thing Kafka actually is, the thing that explains its throughput, its replication model, its consumer semantics, and its role as the spine of modern data infrastructure, is a **distributed, partitioned, replicated commit log**. Not a log in the sense of `log.error("something broke")` — that is application logging, unstructured text for a human to read. A log here is the much older and more fundamental data structure that every database already runs internally: an append-only, totally-ordered sequence of records, each stamped with a position, where you only ever add to the end and you read forward from wherever you left off. Jay Kreps, who built Kafka at LinkedIn and then co-founded Confluent, wrote the canonical argument for this view in [The Log: What every software engineer should know about real-time data](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying), and Martin Kleppmann turned the same idea into an architectural manifesto in his talk [Turning the database inside out](https://martin.kleppmann.com/2015/11/05/database-inside-out-at-oredev.html) and in Chapter 11 of [Designing Data-Intensive Applications](https://dataintensive.net/). This article is a tour of that idea, built up from the log abstraction and pushed all the way to "the database turned inside out."

![A Kafka topic is split into ordered append-only partitions where every record lives at a monotonic offset, producers append at the tail, and consumers track their own read position](/imgs/blogs/kafka-as-a-distributed-log-1.webp)

The diagram above is the mental model for everything that follows. A topic named `orders` is not one log; it is several parallel logs called **partitions**. Each partition is an append-only sequence of records, and each record sits at a numeric **offset** — partition 0 has records at offsets 0, 1, 2, 3, 4; partition 1 has its own offsets 0, 1, 2; and so on. A producer appends to the tail of a partition, choosing which partition by hashing the record's key. Crucially, the green and lavender cards on the right are two independent **consumer groups** reading the same partition at different positions: billing has committed up to offset 2, the search index up to offset 4. Neither read deletes anything. The red card states the rule that breaks the queue model entirely: records age out by **time or size**, governed by `log.retention.hours` or `log.retention.bytes`, never by whether someone read them. A consumer reading a record does not consume it in the destructive sense. The log just sits there, and any number of readers can replay it from any offset, forever, until retention reclaims the old segments from the head. Hold onto that picture; the rest of this is a guided walk through each part of it.

## Why "Kafka is a queue" is wrong, precisely

Let me make the mismatch sharp, because the wrong model is not merely incomplete — it actively predicts the wrong behavior. The senior rule of thumb here is: **a queue is destructive and single-delivery; a log is non-destructive and replayable, and Kafka is a log.** If you internalize nothing else from this post, internalize that one sentence and the table below.

| Property | Traditional queue (RabbitMQ, SQS) | Kafka (a log) |
| --- | --- | --- |
| What a read does | Removes the message (ack deletes it) | Advances *your* offset; the record stays |
| Who can read a message | Usually one consumer, then it's gone | Any number of independent consumer groups |
| Replay old data | Impossible — it was deleted | Rewind your offset to 0 and reprocess |
| Ordering | Per-queue, lost under concurrency | Total order *within a partition* |
| Retention | Until consumed | By time/size, regardless of reads |
| Throughput model | Per-message broker bookkeeping | Sequential append + batch + zero-copy |
| Backpressure | Broker buffers, can fill up | Consumers lag; the log absorbs it on disk |

Walk down the rows and you can feel the design philosophy diverge. In a queue, the broker tracks *per-message* state — delivered, acknowledged, redelivered — because deletion-on-ack is the whole contract, and that bookkeeping is what limits a queue's throughput and what makes "let two services read the same message" awkward. In Kafka, the broker tracks essentially *nothing* per message. A record is bytes at an offset in a file; the broker does not know or care whether anyone has read it. The only read-position state that exists is the **consumer's own committed offset**, and that offset is itself just another record stored in a Kafka topic (`__consumer_offsets`). Kafka pushed the consumption bookkeeping out of the broker and into a number the consumer owns. That single inversion is why one Kafka cluster at LinkedIn, Uber, or Netflix can fan one event out to dozens of independent teams without the broker breaking a sweat — the broker is not doing per-consumer work, it is just letting many readers scan the same files.

This is the same move databases make internally, and it is not a coincidence. As Kreps puts it, the log is "the authoritative source in restoring all other persistent structures in the event of a crash" — the write-ahead log is written *first*, and the actual tables and indexes are just derived structures the database rebuilds from the log if it must. We covered exactly this mechanism in [write-ahead logging](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability): a database commits by appending to a log and considers the change durable the instant the log fsyncs, before the in-place data pages are even touched. Kafka takes that internal log, the thing a single database keeps private, and makes it the public interface. The log stops being an implementation detail of one system and becomes the integration backbone of *every* system.

> A queue asks "who still needs this message?" and deletes it when the answer is nobody. A log never asks. It keeps the bytes, hands every reader a bookmark, and lets the calendar — not the readers — decide when old data is reclaimed.

## 1. The log model: append-only, ordered, replayable

**Senior rule of thumb: the log is the simplest data structure that is also a complete record of what happened and when — treat it as the source of truth and everything else as a cache.**

Strip Kafka of its distribution, its replication, its clients, and what remains is almost insultingly simple. A partition is a file you only append to. Reads start at some offset and proceed forward. There is no update-in-place, no delete-by-key in the normal path, no random write. That simplicity is not a limitation that Kafka tolerates; it is the source of its power, exactly as it is for the [log-structured merge trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) that back Cassandra, RocksDB, and Kafka's own storage. Sequential append is the one I/O pattern that both spinning disks and SSDs do spectacularly well, and giving up random writes is what buys you the throughput.

Three properties define the log, and each one earns its keep.

**Append-only and immutable.** Once a record is written at offset 42 in partition 3, it is at offset 42 forever, with those exact bytes, until retention deletes the whole segment it lives in. You cannot rewrite offset 42. This immutability is what makes replay possible and what makes the log a trustworthy audit trail: nobody can quietly change history. It is also what makes the storage engine cheap — appending never has to find free space, never fragments, never needs to lock a page for update.

**Totally ordered within a partition.** Within a single partition, offset order *is* the order things happened, as far as that partition is concerned. Offset 41 strictly precedes 42 precedes 43. There is no ambiguity, no clock skew, no "well, it depends which replica you ask." Kreps calls the offset a "logical clock" — a notion of time "decoupled from any particular physical clock," which is precisely the kind of logical ordering we discussed in [time, clocks, and ordering in distributed systems](/blog/software-development/database/time-clocks-and-ordering-in-distributed-systems). The offset gives you a per-partition Lamport-style timestamp for free, simply by counting appends.

**Replayable because consumers own their position.** The reader holds a bookmark — the offset of the next record it wants — and Kafka will hand it records from there. If you want to reprocess everything, you set your offset to 0 and read forward again. If you want to skip ahead, you set it to the end. If a downstream system was buggy for two days and you fixed it, you rewind two days and replay. None of this is special machinery; it is the natural consequence of the broker not deleting on read.

Here is a producer appending to the log, written against the real Java client, with the durability and ordering settings that matter (which I will justify in detail later — for now notice that they exist):

```java
import org.apache.kafka.clients.producer.*;
import java.util.Properties;

Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092,broker2:9092,broker3:9092");
props.put("key.serializer",   "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

// Durability + ordering. As of Kafka 3.0 these are the defaults, but be explicit.
props.put("acks", "all");                 // wait for the full in-sync replica set
props.put("enable.idempotence", "true");  // no duplicates from retries (see §6)
props.put("max.in.flight.requests.per.connection", "5"); // safe with idempotence

try (Producer<String, String> producer = new KafkaProducer<>(props)) {
    // The KEY decides the partition: all events for order 7f3c land in one partition,
    // so they are totally ordered relative to each other.
    var record = new ProducerRecord<>("orders", "order-7f3c", "{\"status\":\"paid\"}");
    RecordMetadata md = producer.send(record).get();   // .get() blocks for the ack
    System.out.printf("appended to %s-%d at offset %d%n",
                      md.topic(), md.partition(), md.offset());
}
```

The line that captures the whole abstraction is the last one: the broker tells you the *offset* it assigned. That offset is the record's permanent address. Anyone who later wants this record asks for `orders` partition N at that offset, and gets exactly these bytes back — today, tomorrow, or after a replay next month. Notice too that the key `order-7f3c` is doing real work: it pins every event about that one order to a single partition, which is the only way Kafka gives you ordering, a point worth its own section.

### Second-order consequence: time travel is an offset, not a feature request

Because the log keeps everything within retention, "reprocess from a point in time" is not a project; it is a one-line offset reset. The CLI makes it concrete:

```bash
# Move consumer group "search-index" back to the start of every partition and replay.
kafka-consumer-groups.sh --bootstrap-server broker1:9092 \
  --group search-index --topic orders \
  --reset-offsets --to-earliest --execute

# Or to a wall-clock instant — Kafka maps the timestamp to the nearest offset per partition.
kafka-consumer-groups.sh --bootstrap-server broker1:9092 \
  --group search-index --topic orders \
  --reset-offsets --to-datetime 2026-06-10T00:00:00.000 --execute
```

In a queue, the equivalent of that second command does not exist — the data from June 10th was deleted the moment it was acked. The non-destructive log is what turns a category of "we'd have to rebuild that from a backup and pray" problems into routine operations. I have personally rebuilt a corrupted Elasticsearch index three times in one afternoon by resetting an offset to zero and letting it replay six hours of the `orders` topic, each time fixing a different mapping bug, and the source data was never at risk because the log was the source of truth and the index was a disposable cache.

## 2. One log, many readers: consumer groups and independent offsets

**Senior rule of thumb: the partition is the unit of parallelism for consumers, and within one consumer group each partition is owned by exactly one consumer — so your maximum useful parallelism equals your partition count.**

![A single partitioned log feeds many independent consumer groups that each track and replay their own offset](/imgs/blogs/kafka-as-a-distributed-log-2.webp)

The figure shows the property that the queue model cannot express. Producers append once at the tail. Then *separate* consumer groups — billing at offset 104, the search index lagging at offset 98, an audit job doing a full replay from offset 0 — each read the same records at their own pace, and a brand-new derived store can be spun up at any time, reset to the earliest offset, and rebuilt from the complete history. Each group's progress is its own committed offset; they do not interfere, they do not block each other, and a slow consumer does not back up the producers because the records are already durably on disk in the log.

The terminology is worth nailing down precisely, because people conflate three different things:

- A **consumer** is a single process/thread reading from Kafka.
- A **consumer group** is a set of consumers that cooperate to read a topic exactly once *as a group*, identified by a `group.id`. Kafka divides the partitions among the consumers in a group so that each partition goes to exactly one consumer in that group.
- An **offset** is a per-(group, partition) bookmark: "this group has processed up to here in this partition." It is stored in the internal `__consumer_offsets` topic — itself a log, compacted, which we will return to.

So "delivery to a consumer group" is at-most-once partition ownership: within one group, two consumers never read the same partition simultaneously. But *across* groups, every group sees every record. Billing and search-index are different `group.id`s, so both get the full stream. This is publish-subscribe and competing-consumers in one mechanism, selected purely by whether two consumers share a `group.id`.

Here is a consumer that joins a group and commits its offset deliberately after processing, which is the pattern that gives you at-least-once delivery:

```java
import org.apache.kafka.clients.consumer.*;
import java.time.Duration;
import java.util.*;

Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092");
props.put("group.id", "billing");                 // the group identity
props.put("enable.auto.commit", "false");         // we commit manually, after work
props.put("auto.offset.reset", "earliest");       // new group? start from the beginning
props.put("key.deserializer",   "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

try (Consumer<String, String> consumer = new KafkaConsumer<>(props)) {
    consumer.subscribe(List.of("orders"));
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));
        for (ConsumerRecord<String, String> r : records) {
            chargeCard(r.key(), r.value());        // the side effect we must not lose
        }
        // Commit AFTER the side effect. If we crash before this line, we reprocess
        // the batch on restart -> at-least-once. Charging must therefore be idempotent.
        consumer.commitSync();
    }
}
```

The ordering of those two operations — do the work, *then* commit the offset — is the entire correctness story for at-least-once processing. If you crash after `chargeCard` but before `commitSync`, you will reprocess that batch on restart, so `chargeCard` must be idempotent (keyed on the order id, say). If you committed *before* doing the work, you would get at-most-once and risk losing a charge on a crash. There is no third option in plain consumer code; exactly-once requires the transactional machinery in section 6. The senior habit is to assume at-least-once everywhere and make every consumer side effect idempotent, because at-least-once is what you actually get unless you go out of your way.

### Second-order consequence: partition count is a capacity decision you cannot easily undo

Because one partition goes to one consumer within a group, ten partitions cap a group at ten parallel consumers; an eleventh consumer in that group sits idle with no partition to own. So partition count is your consumer-side scalability ceiling. The trap is that you cannot freely *reduce* partitions (Kafka does not support it cleanly), and *increasing* them changes which partition a key hashes to, breaking per-key ordering for keys that move. The rule I give teams: over-provision partitions modestly at creation time (a topic that does 10 MB/s today but might do 200 MB/s in two years should start with enough partitions for 200 MB/s, since each partition realistically sustains tens of MB/s), but do not go wild — every partition is open files, memory, and replication overhead on every broker, and a cluster with hundreds of thousands of partitions pays for them in controller load and failover time. Uber, processing trillions of messages per day across tens of thousands of topics, manages partition counts as a first-class capacity-planning concern precisely because of this trade-off, as their [scaling writeups](https://www.uber.com/blog/kafka/) describe.

## 3. Topics, partitions, and the ordering you actually get

**Senior rule of thumb: Kafka gives you total order within a partition and no order across partitions, so the partition key is the single most consequential modeling decision you make.**

This is the part that bites teams hardest, because it is silent. Everything works in dev with one partition, where there is a single global order, and then it breaks in prod with twelve partitions, where there is not. Let me be exact about what Kafka guarantees:

- **Within a single partition:** records are totally ordered by offset, and a consumer reads them in that order. Always.
- **Across partitions of the same topic:** there is no ordering guarantee whatsoever. Partition 0's offset 5 and partition 1's offset 5 have no defined relative order. They might be processed in either order, by different consumers, at different times.

The partition is therefore the unit of *both* parallelism and ordering, and those two goals are in direct tension. More partitions means more parallelism and *less* global ordering. The lever you control is the **partition key**: Kafka assigns a record to a partition by `hash(key) % partitionCount` (for the default partitioner with a key set). All records with the same key land in the same partition and are thus totally ordered relative to each other. Records with different keys may land in different partitions and have no guaranteed order between them.

This means you do not get to choose "ordered" or "fast" globally. You choose *what unit you need ordered*, and you make that the key. Concretely:

| You need ordering by… | Use as the partition key | What you get |
| --- | --- | --- |
| Per-customer events | `customer_id` | All of one customer's events in order; customers parallelized |
| Per-order lifecycle | `order_id` | created → paid → shipped never reorder; orders parallelized |
| Per-account balance | `account_id` | debits/credits applied in order; accounts parallelized |
| Nothing in particular | `null` (round-robin) | Maximum spread, no ordering at all |
| Global total order | a single constant key | Total order, but throughput of one partition (anti-pattern) |

That last row is the trap people fall into when they panic about ordering: "I'll just use one partition / one key so everything is ordered." Now your entire topic's throughput is bounded by a single partition on a single broker, and you have thrown away the whole point of Kafka. The right move is almost always to find the *real* ordering requirement — which is virtually never "global," it is "per entity" — and key on the entity. Payments need debits and credits for *one account* in order; they do not need account A's transactions ordered against account B's. Key on account, and you get correctness and parallelism at the same time.

Here is how to be explicit about it when the default partitioner is not what you want, e.g. routing by a field inside the value:

```java
// Explicit partition selection: route by region extracted from the payload,
// so all events for a region stay ordered and colocated.
int partitionForRegion(String region, int numPartitions) {
    return Math.floorMod(region.hashCode(), numPartitions);
}

var record = new ProducerRecord<>(
    "orders",
    partitionForRegion("us-east", numPartitions), // explicit partition
    "order-7f3c",                                  // key (also stored)
    "{\"status\":\"paid\",\"region\":\"us-east\"}");
producer.send(record);
```

### Second-order consequence: adding partitions silently breaks key ordering

Because placement is `hash(key) % partitionCount`, changing `partitionCount` changes the partition for most keys. If `order-7f3c` was in partition 3 with 12 partitions, it might be in partition 7 with 16. Now its old events sit in partition 3 and its new events in partition 7, and a consumer can read partition 7's "shipped" before partition 3's "paid." There is no error; ordering just quietly breaks for the window of keys in flight during the change. The mitigations are real but constrained: plan partition count up front, or use a custom partitioner with stable key-to-partition mapping (consistent hashing, which we explored in [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning)), or accept a controlled cutover where you drain the topic before resizing. The honest answer most of the time is: get the partition count roughly right at creation and avoid resizing keyed topics.

## 4. Storage and performance: segments, the page cache, and zero-copy

**Senior rule of thumb: Kafka is fast not because it is clever in user space but because it gets out of the way and lets the OS do sequential I/O, page-cache reads, and `sendfile` — the broker is mostly a thin coordinator over the filesystem.**

People are often startled that Kafka, which writes everything to disk, routinely outperforms in-memory message systems. The resolution is that "writes to disk" and "slow" are only synonymous for *random* I/O. Kafka's access pattern is almost purely sequential, and on that pattern a commodity disk array delivers hundreds of MB/s, often saturating the network before the disk. Three design choices, all described in the [Kafka design docs](https://kafka.apache.org/documentation/#design) and Kreps's writeups, make this work.

**Segment files and sequential append.** Each partition is stored as a series of **segment** files on disk, e.g. `00000000000000000000.log`, `00000000000000368120.log`, where the filename is the base offset of the first record in that segment. Writes always append to the active (newest) segment, so the write head moves forward and never seeks. When the active segment hits a size or time threshold (`log.segment.bytes`, default 1 GB; `log.roll.hours`), Kafka rolls a new segment. Retention and compaction operate at segment granularity — deleting old data is just unlinking a whole file, which is cheap. Alongside each `.log` segment sit `.index` (offset → file position) and `.timeindex` (timestamp → offset) files so a consumer asking for "offset 368500" or "the record nearest this timestamp" can binary-search to the right byte without scanning.

```
# A partition directory on a broker. Sequential append to the active segment;
# old segments are whole files that retention can unlink atomically.
$ ls -la /var/lib/kafka/data/orders-3/
00000000000000000000.log        # records 0..368119      (sealed)
00000000000000000000.index      # offset -> byte position
00000000000000000000.timeindex  # timestamp -> offset
00000000000000368120.log        # records 368120..now    (active, being appended)
00000000000000368120.index
00000000000000368120.timeindex
leader-epoch-checkpoint         # which leader wrote which offset ranges (see §5)
```

**The OS page cache instead of an application heap.** Kafka does not maintain a large in-process cache of recent records. It writes records to the filesystem and lets the **OS page cache** hold hot data in RAM. This is a deliberate inversion of the usual "cache in your app" instinct, and it is the right call for a JVM service: data lives in the kernel's page cache once, shared, not duplicated into a multi-gigabyte Java heap where it would wreck GC. The Kafka docs put the consequence bluntly — "Data is copied into pagecache exactly once and reused on each consumption instead of being stored in memory and copied out to user-space every time it is read." The operational tell is striking: on a healthy cluster where consumers are caught up, you will see essentially *zero* read I/O on the disks, because every read is served from page cache. The disks are doing writes only; reads never touch them.

**Zero-copy with `sendfile`.** When a consumer fetches records, the naive path is: kernel reads the file into a kernel buffer, copies it into the application's user-space buffer, the application copies it into a socket buffer, the kernel sends it. That is four copies and two context switches per chunk. Kafka instead uses `FileChannel.transferTo()`, which maps to the `sendfile(2)` system call, sending bytes straight from the page cache to the network socket **without ever copying them into the JVM**. As the design docs note, this lets Kafka "transmit data at the rate supported by the disk or network." Combine zero-copy with the page cache and a caught-up consumer is, in effect, the OS streaming a file region directly onto a socket — the broker's CPU barely participates.

The throughput these three choices buy is not theoretical. Confluent documented a single Kafka cluster crossing [1.1 trillion messages per day](https://www.confluent.io/blog/apache-kafka-hits-1-1-trillion-messages-per-day-joins-the-4-comma-club/). Netflix's [Keystone pipeline](https://netflixtechblog.com/keystone-real-time-stream-processing-platform-a3ee651812a) moves on the order of trillions of events per day through roughly 100 Kafka clusters, ingesting about 3 PB and emitting about 7 PB daily. Uber processes [trillions of messages per day](https://www.uber.com/blog/kafka/) across tens of thousands of topics. None of that is possible if the broker is doing per-message work; it is possible because the broker is, at its core, sequential append plus `sendfile`.

### Second-order consequence: batching and compression compound the win

Because the unit of I/O is a record *batch*, not a record, producers accumulate records for a few milliseconds (`linger.ms`) up to a size (`batch.size`) and ship them together, and the broker stores and replicates the batch as a unit. Compression (`compression.type=zstd` or `lz4`) is applied to the whole batch, so larger batches compress better, and — critically — the broker stores the compressed batch *as-is* and `sendfile`s it compressed straight to the consumer, which decompresses. The CPU cost of compression is paid once by the producer and once by the consumer, never by the broker. Tuning `linger.ms` from 0 to even 5–10 ms often doubles throughput on a busy topic at the cost of a few milliseconds of latency, because it turns a flood of tiny appends into a stream of fat sequential ones. The lever is small; the effect is large.

## 5. Replication: leaders, the in-sync replica set, and the high-watermark

**Senior rule of thumb: a record is only safe once it is on the full in-sync replica set, and the high-watermark is the line that marks "safe" — never confuse "the leader has it" with "it is committed."**

A single log on a single disk is not a distributed system; it is a single point of failure. Kafka makes each partition durable by replicating it. One broker is the partition's **leader**; the others holding copies are **followers**. All reads and writes for a partition go to its leader; followers exist to take over if the leader dies and to provide the durability quorum. This is leader-based replication, the same family we surveyed in [leader, multi-leader, and leaderless replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless), specialized to an append-only log.

![A partition has one leader and a set of in-sync replicas, and the high-watermark is the highest offset replicated to all of them, with leader epochs preventing log divergence on failover](/imgs/blogs/kafka-as-a-distributed-log-3.webp)

The figure shows the machinery. The leader (broker 1) has records at offsets 0 through 5. Followers continuously **fetch** from the leader, pulling new records as they arrive — replication in Kafka is pull-based, followers asking the leader "what's after offset N for me?", which reuses the exact same fetch path consumers use. Brokers 2 and 3 have caught up to offset 3 and are **in-sync**; they are members of the **in-sync replica set (ISR)**. Broker 4 has fallen behind (only offsets 0–1) and has been **dropped from the ISR** because it lagged beyond `replica.lag.time.max.ms` (default 30 seconds in current versions; historically 10s).

The **high-watermark (HW)** is the crucial concept: it is the highest offset that has been replicated to *every* member of the ISR. In the figure HW = 4, meaning offsets 0–3 are on all in-sync replicas. Those records are **committed**: they are durable, they are visible to consumers, and — this is the guarantee — a committed record cannot be lost as long as at least one ISR member survives. Offsets 4 and 5 exist only on the leader (the leader's **log-end-offset** is 6). They are **uncommitted**: not yet replicated, *invisible to consumers* (consumers can only read up to the high-watermark), and they may be **truncated** if this leader fails right now, because a new leader elected from the ISR would not have them. This is why consumers never see uncommitted data: the high-watermark gates consumer reads precisely so that a consumer never reads a record that could later vanish in a failover.

Now the durability dial. The producer's `acks` setting chooses how many replicas must persist a record before the producer considers the write acknowledged, trading latency for safety.

![The producer acks setting trades write latency against the number of replicas that must persist before acknowledgement, with acks=all plus min.insync.replicas=2 the only no-data-loss configuration](/imgs/blogs/kafka-as-a-distributed-log-4.webp)

The matrix lays out the three settings, and the right column is the one that matters:

- **`acks=0`** — fire-and-forget. The producer does not wait for any acknowledgement. Lowest latency, but if the broker is down or the batch is lost in flight, the data is simply gone and the producer never knows. Use only for genuinely lossy data: metrics samples, debug telemetry.
- **`acks=1`** — the leader writes to its own log and acknowledges, without waiting for followers. Low latency, but if the leader crashes after acknowledging and before any follower replicated the record, that record is lost on failover. This is the dangerous default-feeling setting that loses data in exactly the failure mode that matters.
- **`acks=all`** (a.k.a. `acks=-1`) — the leader waits until the record is replicated to the full ISR before acknowledging. Highest latency, but combined with the next setting, it is the *only* configuration that guarantees no data loss.

The catch that trips everyone up: `acks=all` alone is not enough. If the ISR has shrunk to just the leader (all followers fell behind or died), then "the full ISR" is one broker, and `acks=all` degrades to `acks=1` — the leader acknowledges alone and a subsequent leader failure loses data. The fix is the topic-level `min.insync.replicas` setting. With `min.insync.replicas=2`, a partition with fewer than 2 in-sync replicas *refuses writes* (the producer gets a `NotEnoughReplicasException`) rather than accepting a write that only one broker holds. The canonical durable configuration is therefore:

```bash
# Topic created with replication.factor=3 and the durability floor.
kafka-topics.sh --create --topic payments \
  --bootstrap-server broker1:9092 \
  --partitions 12 --replication-factor 3 \
  --config min.insync.replicas=2

# Producer must use acks=all for the floor to apply.
#   acks=all + replication.factor=3 + min.insync.replicas=2
#   => tolerates 1 broker failure with zero data loss and continued availability;
#      tolerates 2 broker failures by refusing writes (preserving durability over availability).
```

That `3 / 2` configuration — replication factor 3, min in-sync 2, `acks=all` — is the load-bearing durability recipe for any topic that is a source of truth. It tolerates one broker failure with no loss and no downtime (still 2 in ISR ≥ min 2). It tolerates a second failure by refusing writes, choosing durability over availability — a deliberate CP-leaning choice, in the language of the [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc).

**Leader epochs prevent silent log divergence.** There is a subtle failure where naive high-watermark-based truncation can lose or fork data: a follower that becomes leader, then the old leader returns with extra uncommitted records and has to figure out what to discard. The original protocol used the high-watermark for this and had edge cases that could diverge the logs. [KIP-101](https://cwiki.apache.org/confluence/display/KAFKA/KIP-101+-+Alter+Replication+Protocol+to+use+Leader+Epoch+rather+than+High+Watermark+for+Truncation) fixed it by stamping every record range with a **leader epoch** — a monotonically increasing number bumped on every leader election (epoch 7 in the figure). On failover, a returning stale follower asks the new leader "what was the end offset of epoch 6?" and truncates exactly to that point, deterministically, with no guesswork. This is what guarantees that the replicas converge on one true history rather than silently forking — the kind of correctness property that distinguishes a real consensus-grounded system from a hopeful one.

### Second-order consequence: replication is the reason throughput has a durability tax

`acks=all` means a write's latency includes a round trip to the slowest in-sync follower. If one follower is on a degraded disk or a congested network link, it drags up p99 producer latency for the whole partition, and if it falls far enough behind it gets evicted from the ISR (which then *speeds up* acks but shrinks your durability margin). Monitoring `UnderReplicatedPartitions` and ISR shrink/expand rates is therefore not optional for any durable topic — a chronically under-replicated partition is a data-loss incident waiting for one more failure. The durable configuration is correct, but it makes your tail latency hostage to your slowest replica, and that is a property you must operate, not just configure.

## 6. KRaft: when the metadata itself becomes a log

**Senior rule of thumb: Kafka's own cluster metadata is now stored the same way Kafka stores everything else — as a replicated log — which is why KRaft is both simpler to operate and, in hindsight, inevitable.**

For most of Kafka's life, the brokers stored their data as logs but kept their *metadata* — which topics exist, how many partitions, who leads each partition, ACLs, configs — in [Apache ZooKeeper](https://zookeeper.apache.org/), a separate distributed coordination system. That worked, but it meant running and operating two distributed systems with two failure models, and it created a scaling ceiling: ZooKeeper's data model and the way the controller synced metadata out of it limited how many partitions a cluster could practically manage, and failover involved reading large amounts of state from ZooKeeper.

[KIP-500](https://cwiki.apache.org/confluence/display/KAFKA/KIP-500%3A+Replace+ZooKeeper+with+a+Self-Managed+Metadata+Quorum) removed ZooKeeper by doing the obvious-in-hindsight thing: store the metadata as a Kafka log. The result is **KRaft** (Kafka Raft), production-ready since Kafka 3.3 and the only supported mode as of Kafka 4.0 — ZooKeeper is gone entirely. If you have read [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch), the design will feel like home, because it *is* Raft, adapted to Kafka's log.

![KRaft stores cluster metadata in a Raft-replicated log written by the active controller and replayed by every broker, replacing ZooKeeper with a self-managed metadata quorum](/imgs/blogs/kafka-as-a-distributed-log-5.webp)

The figure traces a metadata change. An admin operation (create a topic, reassign partitions) goes to the **active controller**, which is the Raft leader of a small quorum of controller nodes — the one with the highest epoch, elected by Raft. The active controller appends the change to a special internal topic, `__cluster_metadata`, and replicates it to the **controller followers** via Raft. Once a quorum (majority) of controllers has the record, it is **committed** — exactly the quorum-commit semantics Raft gives us. Then every **broker** in the cluster tails the `__cluster_metadata` log, applying a snapshot plus the stream of deltas to build its local view of cluster state.

This is a genuinely elegant collapse of two systems into one idea. The metadata is a log. Consensus over the metadata is Raft over that log. Brokers learn the cluster state by being log consumers, the same way every other Kafka consumer learns anything. The benefits Kreps and the KIP authors cite are concrete:

| Dimension | ZooKeeper era | KRaft |
| --- | --- | --- |
| Systems to operate | Kafka + ZooKeeper (two) | Kafka only (one) |
| Metadata storage | External ZooKeeper znodes | Internal `__cluster_metadata` log |
| Metadata consensus | ZooKeeper's ZAB | Raft over the metadata log |
| Controller failover | Read state from ZooKeeper (slow) | New leader already has the log (fast) |
| Partition scalability | Bounded by ZK + controller sync | Millions of partitions per cluster |
| Metadata divergence risk | Two sources can disagree | One source of truth |

The failover improvement is the operationally visible one. In the ZooKeeper era, a controller failover meant the new controller reloading metadata from ZooKeeper, which on a large cluster could take many seconds to minutes, during which leadership changes stalled. In KRaft, a controller follower already has the full metadata log replicated locally, so when it becomes leader it is ready almost immediately — failover drops from minutes to well under a second on large clusters. That is the difference between a metadata system that is a separate thing you bolt on and one that is built from the same log primitive as everything else.

### Second-order consequence: the metadata log is compacted, which previews the next idea

The `__cluster_metadata` log cannot grow forever, and you do not need its full history — you need the *current* state of the cluster. So KRaft periodically snapshots the metadata and compacts the log, keeping only what is needed to reconstruct the present plus recent deltas. That is **log compaction**, the same mechanism Kafka offers to users, and it is the bridge to the deepest idea in this whole article: a compacted log of keyed changes *is* a table. The metadata log compacted down to "current topic configs, current partition leaders" is exactly a materialized table of cluster state, reconstructed by replaying a log. KRaft is, in miniature, the database turned inside out.

## 7. Consumer groups and the rebalance: who owns which partition

**Senior rule of thumb: a rebalance is the most disruptive routine event in a Kafka consumer, so design to make rebalances rare and cheap — cooperative protocol plus static membership — rather than tuning around frequent ones.**

We established that within a group each partition has exactly one owning consumer. The mechanism that assigns partitions to consumers, and reassigns them when group membership changes, is the **rebalance**. A rebalance triggers when a consumer joins (a new instance starts, a deploy scales up), leaves (crash, shutdown, or a heartbeat timeout), or the partition set changes. The group's broker-side **coordinator** runs the protocol, and a designated consumer (the group leader) computes the new partition assignment.

The problem is that the original rebalance protocol was **eager**, and eager rebalancing is brutal.

![Cooperative incremental rebalancing reassigns only the partitions that change owner instead of stopping the whole consumer group like eager rebalancing does](/imgs/blogs/kafka-as-a-distributed-log-6.webp)

The before/after captures it. Under **eager** rebalancing (left), when one new consumer C3 joins, *every* consumer in the group revokes *all* of its partitions, the entire group stops processing, the leader computes a fresh assignment from scratch, and everyone re-fetches their newly assigned partitions and resumes. For the duration — which can be seconds on a large group — the whole group is dead in the water, consumer lag spikes, and if consumers are flapping (say, an under-provisioned deployment where instances keep timing out) you get a **rebalance storm**: continuous stop-the-world rebalances where the group spends more time rebalancing than processing. I have watched a 40-instance consumer group melt down into a rebalance storm because a memory leak made instances slow to heartbeat; each timeout triggered a full rebalance, which slowed everyone further, which caused more timeouts. The group's lag climbed for twenty minutes until we rolled it.

[KIP-429 cooperative incremental rebalancing](https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A+Kafka+Consumer+Incremental+Rebalance+Protocol) fixed the stop-the-world problem. Under **cooperative** rebalancing (right), when C3 joins, the protocol computes which partitions actually need to move — just enough to give C3 a fair share — and revokes *only those*. C1 and C2 keep processing all their other partitions throughout; only the single partition that is migrating to C3 pauses, briefly. The disruption is proportional to the change, not to the group size. This is the default assignor (`CooperativeStickyAssignor`) in modern clients and you should use it.

The second lever is **static membership** ([KIP-345](https://cwiki.apache.org/confluence/display/KAFKA/KIP-345%3A+Introduce+static+membership+protocol+to+reduce+consumer+rebalances)). The default behavior treats every consumer connection as ephemeral, so a rolling restart — where each instance leaves and rejoins — triggers a rebalance per instance. Static membership gives each consumer a stable `group.instance.id` that persists across restarts. When a statically-identified consumer disconnects briefly (a deploy bounce), the coordinator holds its partitions for `session.timeout.ms` instead of immediately reassigning them, so a quick restart causes *no rebalance at all*. In a Kubernetes world where pods restart constantly, this is the difference between a rebalance on every deploy and none.

```java
// Modern consumer: cooperative rebalancing + static membership.
props.put("partition.assignment.strategy",
          "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
props.put("group.instance.id", "billing-consumer-" + podOrdinal); // stable across restarts
props.put("session.timeout.ms", "45000");        // tolerate a 45s pod bounce, no rebalance
props.put("heartbeat.interval.ms", "3000");
props.put("max.poll.interval.ms", "300000");     // ample time to process a batch

// And: handle revocation cleanly so cooperative rebalances commit before giving up partitions.
consumer.subscribe(List.of("orders"), new ConsumerRebalanceListener() {
    public void onPartitionsRevoked(Collection<TopicPartition> revoked) {
        consumer.commitSync(); // flush offsets for partitions we're about to lose
    }
    public void onPartitionsAssigned(Collection<TopicPartition> assigned) { /* warm state */ }
    public void onPartitionsLost(Collection<TopicPartition> lost) { /* drop without commit */ }
});
```

The newest evolution, [KIP-848](https://cwiki.apache.org/confluence/display/KAFKA/KIP-848%3A+The+Next+Generation+of+the+Consumer+Rebalance+Protocol), moves assignment computation to the broker-side coordinator entirely, removing the "group leader computes assignment" step and the synchronization barrier it implies, making rebalances even less disruptive. The trajectory across all three KIPs is one consistent goal: make the rebalance — the inherent cost of dynamic partition ownership — as small and infrequent as possible.

### Second-order consequence: `max.poll.interval.ms` is a correctness setting, not just a timeout

A consumer that takes too long *between* `poll()` calls — because one batch triggered a slow downstream call, or a GC pause, or a sync to a database — is assumed dead and kicked from the group, triggering a rebalance and reprocessing of its partitions by whoever takes over. The fix is not to crank `max.poll.interval.ms` blindly (that delays detecting genuinely dead consumers) but to bound the work per poll: lower `max.poll.records` so each batch is processable within the interval, and move genuinely long work off the poll thread. The most common "Kafka is unstable" complaint I get turns out, nine times in ten, to be a consumer doing too much per poll and getting evicted, which presents as mysterious rebalances and lag rather than as the timeout it actually is.

## 8. Exactly-once semantics: idempotent producers and transactions

**Senior rule of thumb: exactly-once in Kafka is a property of a whole read-process-write application, achieved by making the output and the input-offset commit one atomic transaction — it is not a magic per-message guarantee you sprinkle on.**

"Exactly-once" is the most misunderstood phrase in streaming, so let me frame it correctly before the mechanism. You cannot, in general, guarantee a message is *delivered* exactly once over an unreliable network — the FLP and Two Generals constraints are real, as the [two-phase commit and how it fails](/blog/software-development/database/two-phase-commit-and-how-it-fails) discussion makes clear. What Kafka guarantees is **exactly-once processing**: the *effect* of processing each input record is reflected exactly once in the output and in the committed input position, even across retries and crashes. Confluent's own framing, from [Exactly-once Semantics is Possible: Here's How Apache Kafka Does it](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/), is that the output of a read-process-write loop is "the same as if the stream processor saw each message exactly one time." It is an application-level property. Two mechanisms build it.

**The idempotent producer** eliminates duplicates from producer retries within a partition. When `enable.idempotence=true` (the default since Kafka 3.0), the broker assigns the producer a **producer ID (PID)** and the producer tags each record batch with a monotonic **sequence number** per partition. The broker remembers the last sequence number it accepted per (PID, partition) and *rejects duplicates*: if a producer's ack is lost and it retries the same batch, the broker sees the already-seen sequence number and silently dedups instead of appending a second copy. Crucially, this sequence-number state is **persisted in the replicated log**, so — as the Confluent post says — "even if the leader fails, any broker that takes over will also know if a resend is a duplicate." Idempotence turns at-least-once producing into exactly-once *producing* into a partition, for free, which is why it is on by default.

**Transactions** extend that to atomic, all-or-nothing writes across multiple partitions, and — this is the key for read-process-write — across the output records *and* the consumed input offsets together.

![A transactional read-process-write loop commits output records and consumed offsets atomically as a single unit, so on failure either both land or neither does](/imgs/blogs/kafka-as-a-distributed-log-7.webp)

The pipeline shows the loop. The application reads an input batch (with `isolation.level=read_committed`, so it never sees uncommitted transactional data), begins a transaction tied to its stable `transactional.id`, processes the records and updates local state, produces outputs to a downstream topic (idempotently, PID + sequence), and then — the load-bearing step — calls `sendOffsetsToTransaction()` to include the *input* offsets in the same transaction, before committing. The commit is coordinated by a broker-side **transaction coordinator** that records transaction state in the internal `__transaction_state` topic and writes **control markers** (commit/abort markers) into each affected partition so consumers know whether the transactional records are visible. The effect: either the output records and the advanced input offset both become visible atomically, or neither does. There is no state where you produced output but did not record that you consumed the input (which would duplicate on restart), nor where you recorded consumption without producing output (which would lose data).

```java
// Read-process-write with exactly-once. The transactional.id is stable per logical
// processor instance, so the coordinator can fence a zombie old instance.
Properties pProps = new Properties();
pProps.put("bootstrap.servers", "broker1:9092");
pProps.put("transactional.id", "order-enricher-1");  // pins this logical producer
pProps.put("enable.idempotence", "true");            // implied by transactional.id
pProps.put("acks", "all");
// ... serializers ...

Producer<String, String> producer = new KafkaProducer<>(pProps);
producer.initTransactions();                          // register with the coordinator

// Consumer reads committed data only; offsets are committed via the producer txn.
Properties cProps = new Properties();
cProps.put("group.id", "order-enricher");
cProps.put("isolation.level", "read_committed");      // do not see uncommitted txns
cProps.put("enable.auto.commit", "false");
// ... bootstrap, deserializers ...
Consumer<String, String> consumer = new KafkaConsumer<>(cProps);
consumer.subscribe(List.of("orders"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));
    if (records.isEmpty()) continue;

    producer.beginTransaction();
    try {
        Map<TopicPartition, OffsetAndMetadata> offsets = new HashMap<>();
        for (ConsumerRecord<String, String> r : records) {
            String enriched = enrich(r.value());               // the "process" step
            producer.send(new ProducerRecord<>("orders-enriched", r.key(), enriched));
            offsets.put(new TopicPartition(r.topic(), r.partition()),
                        new OffsetAndMetadata(r.offset() + 1));
        }
        // The two writes that MUST be atomic: the outputs above, and these input offsets.
        producer.sendOffsetsToTransaction(offsets, consumer.groupMetadata());
        producer.commitTransaction();                          // both land, or neither
    } catch (KafkaException e) {
        producer.abortTransaction();                           // nothing becomes visible
    }
}
```

The `transactional.id` does one more vital thing: **zombie fencing**. If a stuck instance of `order-enricher-1` revives after a new instance with the same `transactional.id` has started, the coordinator has bumped the epoch for that id, and the zombie's transactions are rejected — it cannot corrupt the output by writing stale data. This is how Kafka makes the "process exactly once" guarantee survive the nastiest failure mode, a slow process that comes back from the dead. [Kafka Streams](https://kafka.apache.org/documentation/streams/) wraps this entire loop behind a single config, `processing.guarantee=exactly_once_v2`, which is how most teams actually get exactly-once: you set one property and the framework runs the transactional read-process-write loop for you.

### Second-order consequence: exactly-once has a real cost and a real boundary

Transactions add latency (the commit is a multi-step coordination with the transaction coordinator and the control-marker writes) and they only cover effects *inside Kafka* — outputs to Kafka topics and offset commits. The moment your "process" step writes to an external system (a database, an HTTP call, an email), that write is outside the Kafka transaction and you are back to the [dual-write problem](/blog/software-development/database/change-data-capture-and-the-outbox-pattern): Kafka can commit while the external write fails, or vice versa. Exactly-once is genuinely end-to-end only when the entire pipeline is Kafka-to-Kafka (or the external sink is itself idempotent on a key, or you use the transactional outbox to bring the external write into a database transaction and CDC it back into Kafka). Sell exactly-once internally as "exactly-once within the Kafka boundary," never as "exactly-once into your billing provider," or you will mislead someone into a real incident.

## 9. Log compaction and the stream–table duality

**Senior rule of thumb: a compacted topic keyed by entity is not a stream of changes you happen to keep — it is a table, materialized as a log, and that equivalence is the deepest idea in the whole system.**

So far retention has meant time/size: old records age out from the head. **Log compaction** is a different retention policy (`cleanup.policy=compact`) that says: keep the *latest* value for every key, and garbage-collect older values for keys that have been superseded. The log stops being a complete history of changes and becomes a current-state snapshot — while still being a log you can replay.

![Log compaction retains the latest record per key, turning an append-only changelog into a materialized table where the latest value per key is the table row](/imgs/blogs/kafka-as-a-distributed-log-8.webp)

The before/after makes the transformation concrete. Before compaction, the changelog has every update: `u1 → Ana`, `u2 → Bo`, `u1 → Ana2` (an update), `u2 → null` (a **tombstone**, the marker that deletes a key). After compaction, the older `u1 → Ana` is gone because `u1 → Ana2` superseded it, and `u2` is gone entirely because the tombstone deleted it. What remains is exactly the *table*: the current value for each live key. The compacted topic *is* the table `{u1: Ana2}`, expressed as a log, and you can rebuild any cache, any in-memory map, any derived store by replaying it from the start.

This is the **stream–table duality** that Kreps and Kleppmann both center their arguments on, and it is worth stating both directions because it is genuinely symmetric:

- **A table is a snapshot of a stream.** Apply every change in the changelog in order and you get the current table. The table is the *integral* of the stream of changes.
- **A stream is a changelog of a table.** Watch a table over time and the sequence of changes it emits is a stream. The stream is the *derivative* of the table.

Kreps's analogy is the bank ledger: the stream is "all credits and debits," the table is "all the current account balances." The log of transactions is the more fundamental object — from it you can compute the balances, and not only today's balances but the balances at any past instant. The table is just a convenient cache of one particular query (the current balance) over the log.

Compaction is what makes this practical at scale, because a pure changelog grows without bound, but a compacted log is bounded by the number of distinct keys (plus in-flight uncompacted tail). This is exactly why Kafka stores its own `__consumer_offsets` and `__transaction_state` and KRaft's `__cluster_metadata` as compacted topics: those are *tables* (current offset per group-partition, current transaction state, current cluster metadata) that happen to be implemented as compacted logs. Kafka eats its own dog food at the most fundamental level.

```bash
# A compacted topic that IS a table of user profiles. Replaying it from offset 0
# reconstructs the full current state; new readers bootstrap by replaying.
kafka-topics.sh --create --topic user-profiles \
  --bootstrap-server broker1:9092 \
  --partitions 12 --replication-factor 3 \
  --config cleanup.policy=compact \
  --config min.cleanable.dirty.ratio=0.1 \
  --config segment.ms=86400000

# Producing an update: same key, new value -> compaction keeps only this.
#   key=u1  value={"name":"Ana2"}
# Producing a delete: same key, null value (a tombstone) -> compaction drops the key.
#   key=u2  value=null
```

In [Kafka Streams](https://kafka.apache.org/documentation/streams/) this duality is the API surface: a `KTable` is a view over a compacted changelog topic, a `KStream` is a view over a regular topic, and you convert between them freely (`stream.toTable()`, `table.toStream()`). A stateful stream operation (a join, an aggregation) keeps its state in a local store backed by a compacted changelog topic, so if the processor crashes, it rebuilds its state by replaying that changelog — the same replay-from-log durability the whole system is built on, applied to processor state. Kleppmann's "turning the database inside out" is precisely this: take the materialized views and triggers that live *inside* a database and externalize them as stream processors over compacted Kafka logs, refreshed continuously instead of on a refresh interval.

### Second-order consequence: tombstones and the delete-retention window

A tombstone (a `null` value for a key) tells compaction to delete the key, but compaction cannot remove the tombstone *immediately* — consumers that are behind need to see it to learn the key was deleted. So a tombstone is retained for `delete.retention.ms` (default 24 hours) after compaction before it too is removed. The failure mode: a consumer that is offline for longer than `delete.retention.ms` can come back, replay the compacted log, and *never see the tombstone*, so it keeps a key that should have been deleted. For a compacted topic acting as a table, your slowest acceptable consumer downtime is bounded by `delete.retention.ms`; size it for your real recovery windows, not the default, if you have consumers that can be offline for days.

## 10. The database turned inside out: Kafka as the data backbone

**Senior rule of thumb: stop thinking of Kafka as the thing between your services and start thinking of it as the source of truth, with every other store — search, cache, warehouse, OLAP — a disposable materialized view of one log.**

Now we can assemble the whole argument. A traditional database bundles together several things: a durable log (the WAL), a query/serving engine over current state (the tables and indexes), and the machinery to keep the second consistent with the first. Kreps's and Kleppmann's insight is that you can *unbundle* these. Put the durable log in the center, make it the source of truth, and let each serving need — full-text search, key-value lookup, analytical scan, machine-learning features — be a separate, specialized, *derived* store that subscribes to the log and materializes exactly the view it needs.

![A central Kafka log ingests source changes via CDC and app events, then feeds every derived store as a replayable materialized view, turning the database inside out](/imgs/blogs/kafka-as-a-distributed-log-9.webp)

The figure is the architecture this whole post has been building toward. On the left, the sources of truth-of-record: an OLTP database whose committed changes are captured by [Debezium](https://debezium.io/) reading the WAL ([change data capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern)), and application event producers. Both write into the central Kafka log. On the right, the derived systems, each a materialized view rebuilt by replaying the log: a search index (Elasticsearch), a read cache (Redis), an analytics warehouse or lake, and a stream processor (Flink or Kafka Streams) that itself emits derived streams feeding, say, a feature store for ML serving. The log is the integral; every store on the right is one query materialized.

This inversion solves a problem that quietly destroys data architectures: the **O(N²) integration explosion**. Without a central log, connecting N data systems means up to N×N point-to-point pipelines, each with its own consistency, ordering, and failure semantics, and adding the (N+1)th system means wiring it to all N existing ones. With a central log, every source writes *once* to the log and every consumer reads *once* from it: N producers plus M consumers, linear, and adding a system means one new consumer subscribing to the log it already has. Kreps makes exactly this scaling argument in The Log, and it is why LinkedIn built Kafka in the first place — to replace a tangle of bespoke pipelines with one log everyone agreed on.

The properties that make this work are the ones we have built up:

- **Replayable** (non-destructive log) means a new derived store rebuilds itself by replaying from offset 0 — no migration, no backfill job, just a consumer reset to earliest.
- **Ordered per partition** means each derived store applies changes in a consistent order, so it converges to the same state the source had.
- **Replicated** (ISR, `acks=all`, `min.insync.replicas=2`) means the source of truth is durable enough to *be* the source of truth.
- **Compacted** (where keyed) means a topic can be a table, so a cache or KV store can bootstrap from current state without replaying all of history.

This is also the right frame for two architectural patterns people often discuss as if they were separate from Kafka. **Event sourcing** — storing an entity's state as the sequence of events that produced it, rather than as a mutable row — is just "the log is the source of truth for this entity, the current state is a fold over it." A compacted Kafka topic keyed by entity is an event-sourced store with snapshotting built in. **CQRS** (command-query responsibility segregation) — separating the write model from one or more read models — is just "write to the log, materialize as many read models as you have query shapes." Kafka does not implement these patterns; it *is* the substrate they are descriptions of. When the log is the source of truth, event sourcing and CQRS stop being patterns you adopt and become the natural shape of the system.

### Second-order consequence: the source-of-truth boundary must be explicit and defended

The power of "everything is a derived view" depends entirely on a clear answer to "derived from *what*?" If two systems both believe they are the source of truth — the OLTP database *and* a Kafka topic that some service also writes to directly — you have recreated the dual-write divergence at the architecture level, and the derived views will disagree depending on which truth they followed. The discipline that makes the inside-out architecture safe is: name exactly one source of truth for each kind of data (usually the OLTP database, with CDC into Kafka; or, for born-in-the-stream data, the Kafka topic itself), and forbid any path that mutates a derived store except by consuming the log. The moment someone "just quickly updates the cache directly" to fix something, the cache is no longer a pure function of the log, and your ability to rebuild it by replay — the entire payoff — is gone. Defend the boundary like the invariant it is.

## Case studies from production

### 1. The single-partition global-ordering deadlock

A payments team wanted strict global ordering of all transactions, so they created the `transactions` topic with one partition. It worked in staging. In production it topped out at roughly 30 MB/s — one partition on one broker — while the business needed five times that during peak. The wrong first hypothesis was broker undersizing; they added brokers and nothing improved, because one partition lives on exactly one leader broker no matter how many brokers exist. The actual root cause was the conflation of "ordered" with "globally ordered." The real requirement was per-account ordering: account A's debits and credits must not reorder, but A's transactions need no order relative to B's. The fix was to repartition to 48 partitions keyed on `account_id`. Throughput scaled linearly to the needed rate, and per-account ordering held because every account's events hashed to a single partition. The lesson: "ordered" almost never means "globally ordered," and finding the true ordering unit (the entity) is how you get correctness and throughput together.

### 2. The `acks=1` silent data loss

A team ran a critical orders topic with `replication.factor=3` and felt safe — three copies, surely durable. Then a broker that happened to be the leader for several partitions crashed during a deploy, and a handful of orders vanished, present in no consumer and no replica. The wrong hypothesis was a consumer bug. The actual cause: the producer used `acks=1`, so the leader acknowledged each write the instant it hit *its own* log, before any follower replicated it. The crashed leader had acknowledged orders that no follower had yet, and the new leader, elected from the followers, simply did not have them — the high-watermark had never advanced to include them, so they were silently dropped on failover. Replication factor 3 is necessary but not sufficient; without `acks=all` and `min.insync.replicas=2`, the producer does not *wait* for replication, so the copies do not protect the most recent writes. The fix was the canonical `3 / 2 / all` recipe. The lesson: durability is a property of the producer's acks setting interacting with the topic's replication, not of the replication factor alone.

### 3. The rebalance storm

A 40-instance consumer group for an enrichment service began spending more time rebalancing than processing; lag climbed for twenty minutes. The first hypothesis was Kafka instability. The real cause was a memory leak in the consumer that made instances slow, so they missed `max.poll.interval.ms`, got evicted (triggering an eager, stop-the-world rebalance), came back, and got evicted again — a self-reinforcing storm where each rebalance slowed the survivors enough to cause the next eviction. Three changes fixed it: switching to the `CooperativeStickyAssignor` so a single eviction only moved that instance's partitions instead of stopping all 40; adding `group.instance.id` static membership so the frequent restarts during the firefight stopped triggering rebalances; and lowering `max.poll.records` so each batch processed well within the poll interval, eliminating the evictions at the source. The lesson: mysterious rebalances are almost always a consumer doing too much per poll and getting evicted, and the cure is bounding per-poll work plus cooperative/static rebalancing, not raising timeouts.

### 4. The page-cache eviction cliff

A cluster's consumer p99 fetch latency suddenly spiked from sub-millisecond to tens of milliseconds, and disk read I/O — normally near zero on a healthy cluster — jumped to saturation. The wrong hypothesis was failing disks. The actual cause was a newly added analytical consumer doing full replays from offset 0 on several large topics, reading data far older than what fit in the page cache. Those reads missed the cache and hit disk, and worse, they evicted the hot recent data that the real-time consumers depended on, so *everyone's* reads started missing cache. Kafka's performance rests on caught-up consumers being served from page cache; a far-behind consumer reading cold data both pays disk latency itself and poisons the cache for others. The fix was to isolate the heavy replay consumer onto a separate set of brokers (a follower-fetch / tiered-storage arrangement) and to schedule full replays off-peak. The lesson: the page cache is a shared, finite resource, and a single cold-reading consumer can turn a memory-speed cluster into a disk-bound one for everybody.

### 5. The exactly-once that wasn't (external write)

A stream processor used `processing.guarantee=exactly_once_v2` and the team confidently told finance that charges could not be double-applied. Then a deploy mid-transaction produced a small number of double charges. The wrong hypothesis was a bug in Kafka's transaction implementation. The actual cause: the "process" step called an external payment API, and that HTTP call is *outside* the Kafka transaction. Kafka's exactly-once covers Kafka outputs and offset commits atomically, but when the processor crashed after the external charge succeeded but before the Kafka transaction committed, the input offset was not advanced, so on restart the same record was reprocessed and the external charge fired again. The fix was to make the external call idempotent with an idempotency key derived from the record (so the payment provider deduped the retry) and, longer term, to move to a transactional-outbox pattern where the charge intent is written to a database in a transaction and CDC'd back to Kafka. The lesson: Kafka exactly-once is exactly-once *within the Kafka boundary*; any external side effect needs its own idempotency, and claiming end-to-end exactly-once across a non-idempotent external system is simply false.

### 6. The compaction tombstone that never arrived

A service kept a local cache bootstrapped from a compacted `user-profiles` topic. A batch of users was deleted (tombstones produced), and most caches dropped them correctly. But one read-replica service that had been down for a four-day maintenance window came back still serving the deleted users. The wrong hypothesis was a producer that forgot to send tombstones. The actual cause: `delete.retention.ms` was at its 24-hour default, so by the time the four-day-offline consumer replayed the compacted log, the tombstones had already been compacted away — it saw the absence of the keys' old values but never the explicit delete markers, and since compaction had removed the prior values too, it simply had no record that those keys ever existed to remove from its pre-existing local state. The fix was raising `delete.retention.ms` to comfortably exceed the maximum realistic consumer downtime, and adding alerting on consumers lagging beyond that window. The lesson: on a compacted topic acting as a table, tombstone retention sets a hard ceiling on how long a consumer can be offline and still converge correctly.

### 7. The ZooKeeper-to-KRaft failover speedup nobody planned for

A team running a large cluster (tens of thousands of partitions) on the old ZooKeeper-backed mode suffered multi-minute leadership stalls whenever the controller failed over, because the new controller had to reload all metadata from ZooKeeper before it could act. They had built elaborate operational runbooks around tolerating these stalls. Migrating to KRaft, they expected a wash and got a surprise: controller failover dropped from minutes to sub-second, because a KRaft controller follower already holds the full `__cluster_metadata` log replicated locally and is ready to lead the instant it wins the Raft election. An entire class of "the cluster is briefly unmanageable during controller failover" incidents disappeared, and the runbooks for them became obsolete. The lesson: storing metadata as a replicated log rather than in an external store does not just simplify operations, it changes the failover performance characteristics, because the new leader has nothing to reload.

### 8. The under-replicated partition that became a data-loss incident

A monitoring gap let a single partition sit under-replicated (ISR shrunk from 3 to 1, just the leader) for several days because a follower's disk was failing slowly and nobody watched `UnderReplicatedPartitions`. With `min.insync.replicas=2`, that partition should have refused writes once ISR dropped below 2 — but the topic had been created with `min.insync.replicas=1` (the cluster default nobody had overridden), so the lone leader kept accepting `acks=all` writes that were really `acks=1` writes in disguise. When the leader's own disk then failed, every write since the ISR shrank was lost. The wrong hypothesis was that `acks=all` had protected them. The actual cause was `min.insync.replicas=1` quietly turning `acks=all` into `acks=1` once the ISR collapsed to the leader. The fix was setting `min.insync.replicas=2` explicitly on every durable topic (and as the cluster default) plus alerting on ISR shrink and under-replicated partitions. The lesson: `acks=all` is only as strong as `min.insync.replicas`; with the floor at 1, your strongest acks setting degrades to your weakest the moment replicas fall behind.

## When to reach for Kafka, and when not to

Kafka is a remarkable piece of engineering, and like every remarkable piece of engineering it is the wrong choice for a great many problems. The log model earns its complexity only when you actually need a log.

**Reach for Kafka when:**

- **Multiple independent consumers need the same data stream.** The moment two or more teams or systems want to read the same events at their own pace, the non-destructive log is exactly right and a queue is exactly wrong. This is the single strongest signal.
- **You need replay.** If "reprocess the last N hours/days after a bug fix" or "bootstrap a new derived store from history" is a real requirement, you need a retained, replayable log, which is Kafka's whole premise.
- **You are building a data backbone.** CDC from databases, fan-out to search/cache/warehouse, event sourcing, stream processing — the inside-out architecture needs a durable central log, and Kafka is the mature default for it.
- **Throughput is genuinely high and sustained.** Tens of MB/s to GB/s, millions of messages per second, where sequential-append + page-cache + zero-copy economics matter. Below a few thousand messages per second, that engine is overkill.
- **Per-entity ordering plus parallelism is the requirement.** Keyed partitioning gives you ordered-per-key and parallel-across-keys simultaneously, which queues struggle to express.
- **Durability must survive broker failures with a tunable guarantee.** The ISR/`acks`/`min.insync.replicas` model gives you a precise, operable durability dial.

**Skip Kafka when:**

- **You need request/response or RPC.** Kafka is one-way, async, fire-into-a-log. If a caller needs a synchronous reply, use gRPC/HTTP. Bolting request/response on top of Kafka (correlation IDs, reply topics) is a well-known anti-pattern that reinvents RPC badly.
- **Volume is low and will stay low.** A few hundred or few thousand messages a day with one producer and one consumer does not justify a replicated log cluster. A database table polled as a queue, SQS, or a simple broker is less to operate and reason about.
- **The team is small and has no streaming operational muscle.** Kafka in production is real work: partition planning, ISR and lag monitoring, rebalance tuning, retention and compaction config, broker capacity. A two-person team shipping a CRUD app should not run Kafka to move a trickle of events; the operational tax will dwarf the benefit. (A managed offering like Confluent Cloud or MSK changes this calculus, but the *conceptual* surface area remains.)
- **You need true cross-system transactions including external stores.** Kafka transactions are Kafka-internal. If your invariant spans Kafka and a database and a third-party API atomically, Kafka exactly-once will not give it to you; you need the outbox/CDC pattern or a saga, as covered in [the saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions).
- **You only need a simple task queue with per-message ack, retry, and dead-lettering.** If the access pattern is genuinely "pull one job, do it, ack it, retry on failure," a purpose-built job queue (SQS, RabbitMQ, a database-backed queue) models that directly; Kafka's offset-based, partition-owned consumption fits it awkwardly and you will fight the model.

The throughline is the one we started with. Kafka is a log, not a queue, and the question to ask before adopting it is not "do I have messages to move" but "do I want those messages to *stay* — ordered, replicated, replayable, readable by many, the source of truth that everything else is derived from." When the answer is yes, Kafka is the inside-out database you are reaching for, and every piece of its design — partitions, offsets, the ISR and high-watermark, KRaft, consumer groups, transactions, compaction — is in service of that one idea. When the answer is no, reach for something simpler and save yourself the cluster.

## Further reading

- Jay Kreps, [The Log: What every software engineer should know about real-time data](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) — the foundational essay this entire article expands on.
- Martin Kleppmann, [Designing Data-Intensive Applications](https://dataintensive.net/), Chapter 11 (Stream Processing) and Chapter 3 (log-structured storage) — partitioned logs, consumer offsets, log-based brokers, exactly-once, and the stream-table duality, in depth.
- Martin Kleppmann, [Turning the database inside out](https://martin.kleppmann.com/2015/11/05/database-inside-out-at-oredev.html) — the architectural manifesto for the inside-out view.
- Confluent, [Exactly-once Semantics is Possible: Here's How Apache Kafka Does it](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/) and the [data replication course](https://developer.confluent.io/courses/architecture/data-replication/) — the mechanics of idempotent producers, transactions, ISR, and the high-watermark.
- Apache Kafka, [Design documentation](https://kafka.apache.org/documentation/#design) and [KIP-500](https://cwiki.apache.org/confluence/display/KAFKA/KIP-500%3A+Replace+ZooKeeper+with+a+Self-Managed+Metadata+Quorum) — segment storage, page cache, zero-copy, and the KRaft metadata quorum.
- Sibling posts on this blog: [change data capture & the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), [Raft consensus from scratch](/blog/software-development/database/raft-consensus-from-scratch), [write-ahead logging](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability), and [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines).

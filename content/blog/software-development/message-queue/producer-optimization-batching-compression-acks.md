---
title: "Producer Optimization: Batching, Linger, Compression, and acks"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Squeeze ten times the throughput out of a Kafka producer without losing the durability or ordering you need — batch.size, linger.ms, compression, acks, in-flight limits, and the sticky partitioner, knob by knob, with the tuning math worked out."
tags:
  [
    "message-queue",
    "kafka",
    "producer",
    "throughput",
    "batching",
    "compression",
    "performance-tuning",
    "distributed-systems",
    "event-driven",
    "reliability",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/producer-optimization-batching-compression-acks-1.webp"
---

A junior engineer once handed me a Kafka producer that was supposed to push two million events a minute into a topic and was managing about three hundred thousand. The brokers were bored — single-digit CPU, disks asleep, network barely warm. The consumers were idle. Everyone assumed Kafka was the bottleneck and started a thread about adding partitions. The bottleneck was four lines of producer config, all left at defaults that are tuned for a low-traffic correctness demo and not for a firehose. We changed three numbers, the throughput jumped past two million, the brokers still looked bored, and the partition thread quietly died. That is the recurring story of producer tuning: the broker is almost never the first thing that runs out of road, and the producer is shipping one record per network round trip when it could be shipping ten thousand.

The Kafka producer is a small, surprisingly sophisticated piece of machinery sitting between your `send()` call and the wire. It serializes your record, picks a partition, drops the record into an in-memory accumulator, lets records pile up into a batch, optionally compresses the whole batch, and ships batches to brokers over a bounded number of in-flight connections — all while a background thread does the actual network I/O so your application thread is not blocked on every send. Every one of those stages has a knob, and the knobs interact. Turn one and you change the meaning of another. The art is not memorizing the knobs; it is understanding *where the producer spends time* so you know which knob attacks which cost, and which knobs you must never touch because they hold up the durability or ordering guarantees your system depends on.

![A before-and-after diagram contrasting a producer that sends one record per network request against a producer whose accumulator lets records pile into a full batch before a single request goes out](/imgs/blogs/producer-optimization-batching-compression-acks-1.webp)

By the end of this post you will be able to take a producer that is leaving an order of magnitude of throughput on the floor and fix it deliberately, not by superstition. You will know exactly what `batch.size` and `linger.ms` do to the record accumulator, why bigger batches compress better and which codec to pick, how `buffer.memory` and `max.block.ms` apply backpressure when you outrun the network, what `acks` actually costs and why it is a durability knob and not a throughput knob, why `max.in.flight.requests.per.connection` greater than one can silently reorder your data on a retry unless the idempotent producer is on, and how the sticky partitioner quietly makes all of this better by filling one partition's batch before rotating. You will leave with a concrete tuning recipe for a high-throughput pipeline and the reasoning for every line. This post assumes you have met Kafka before; if you want the durability machinery in depth, read [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) alongside it, and if you want the broader framing of the throughput-versus-latency dial that every knob here sits on, read [Throughput vs latency](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff).

## 1. The producer's job and where time goes

Before touching a single knob, get a clear picture of the path a record takes from your `send()` call to a durable position on the broker's log, because every optimization in this post attacks a specific stage of that path. If you do not know where the time goes, tuning is cargo-culting — you copy someone's config from a blog post and hope. Let us instead build the map.

When your application calls `producer.send(record)`, the calling thread does a small amount of synchronous work and then returns almost immediately. That synchronous work is: run the key and value through their serializers to turn objects into bytes, run the partitioner to decide which partition this record belongs to, and then append the serialized record into an in-memory structure called the *record accumulator*. The accumulator is organized as a map from partition to a deque of batches; your record joins the current open batch for its partition, or starts a new batch if there is no room. At that point `send()` returns a `Future` and your thread is free. The record is not on the network yet. It is not on the broker yet. It is sitting in a buffer in your own process.

The actual network work happens on a separate thread — the *sender* thread, sometimes called the I/O thread — that the producer spins up internally. This thread loops: it looks at the accumulator, finds batches that are *ready* to send (full, or aged past `linger.ms`, which we will get to), groups ready batches by the broker that leads their partitions, and issues one produce request per broker containing all the ready batches for that broker. It then waits for responses, fires your callbacks, and loops again. This two-thread design is the single most important thing to internalize: your application thread and the network are decoupled by a buffer, and almost all of producer tuning is about feeding that buffer efficiently and draining it efficiently.

![A pipeline showing the producer send path as serialize then partition then accumulate then compress then send, each stage labeled with its cost](/imgs/blogs/producer-optimization-batching-compression-acks-3.webp)

### The five stages and their costs

Figure 3 lays out the five stages. Walk them with me, because each has a different cost profile and a different knob.

**Serialize.** Converting your object to bytes. This is CPU on the application thread, usually cheap for primitives and small JSON, more expensive for large nested structures or schema-registry-backed Avro with a wire-format lookup. You rarely tune this directly; you pick an efficient serializer (binary Avro or Protobuf over verbose JSON) and move on. It matters for *size* — a smaller serialized payload means more records per batch and less to compress — which is why schema choice quietly affects throughput. See [Schema management and evolution](/blog/software-development/message-queue/schema-management-evolution-avro-protobuf-registry) for that whole dimension.

**Partition.** Deciding which partition the record goes to. If you set a key, the default partitioner hashes it (murmur2 modulo partition count) so the same key always lands on the same partition — that is how you get [per-key ordering](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees). If you set no key, the *sticky partitioner* (section 7) chooses a partition cleverly to maximize batch sizes. Cheap CPU, but the choice has enormous downstream consequences for batching.

**Accumulate.** Appending to the in-memory batch for the partition. This is where `batch.size`, `linger.ms`, and `buffer.memory` live. The record sits here, in your process, waiting for company. This stage is where latency is *deliberately added* in exchange for throughput, and it is the heart of the post.

**Compress.** When a batch is closed and shipped, the producer compresses the *entire batch* as one unit (not record by record). This is CPU on the sender thread in exchange for fewer bytes on the wire and fewer bytes on the broker's disk. `compression.type` lives here. Bigger batches compress better, which is why this stage is coupled to the accumulate stage.

**Send.** The sender thread writes the request to the socket and waits for the broker's acknowledgement. The cost here is the network round trip plus, crucially, however long the broker takes to satisfy `acks` — with `acks=all` that includes the replication round trip to the in-sync followers. `acks` and `max.in.flight.requests.per.connection` live here. This is where durability is paid for in latency.

The single most important insight is this: **the send stage has a roughly fixed per-request cost** — a TCP round trip, request framing, the broker's per-request bookkeeping — that does not depend much on how many records the request carries. A request with one record and a request with ten thousand records cost almost the same in round trips. So the entire game of producer throughput is *amortizing that fixed per-request cost over as many records as possible*. That is what batching is. Everything else — linger, compression, the sticky partitioner — exists to make batches bigger so the fixed cost is paid less often. Hold that thought; it is the thesis of the whole post.

### A back-of-the-envelope for the fixed cost

Put numbers on "fixed per-request cost" so the thesis is not just rhetoric. A produce request to a broker incurs, at minimum, one network round trip. Within a single availability zone that round trip is on the order of 0.2–0.5 ms; across availability zones it can be 1–2 ms; across regions, tens of milliseconds. On top of the round trip there is request framing (Kafka's binary protocol header, the request and response serialization), and broker-side per-request work: parsing the request, validating it, locating the partition leaders, and writing the request through to the page cache. None of that scales with the number of records inside the request — it is paid once per request whether the request carries one record or ten thousand.

So consider a single-AZ producer with a 0.3 ms round trip and, say, another 0.2 ms of framing and broker bookkeeping per request, for roughly 0.5 ms of fixed cost per request. If you ship one record per request, that record costs 0.5 ms of overhead — and a single thread can issue at most about 2,000 such serial requests per second per connection. Pipelining (multiple in-flight requests) raises that ceiling, but the *overhead per record* stays at 0.5 ms. Now pack 1,000 records into each request: the same 0.5 ms of fixed cost is shared across 1,000 records, so the overhead drops to 0.0005 ms per record — a thousandfold reduction in per-record overhead, paid for with nothing but a few milliseconds of accumulation delay. That ratio is the entire economic case for batching, and it is why a bored-looking broker can still be the throughput ceiling: the broker is not CPU-bound, it is *request-bound*, drowning in the bookkeeping of too many tiny requests.

This also explains why the producer batches *per partition* but ships *per broker*. The accumulator groups records into batches by partition (because a batch belongs to exactly one partition's log), but when the sender thread builds a request it gathers *all* the ready batches whose partitions are led by the same broker into one request. So a single produce request to broker 1 might carry the ready batches for partitions 0, 3, and 6 if broker 1 leads all three. This is a second layer of amortization: not only does each batch amortize the per-request cost across its records, but each request amortizes it across multiple partitions' batches. The more partitions a broker leads, the more batches ride in one request, the better the amortization — which is one quiet argument *against* spreading a topic across an excessive number of brokers.

## 2. batch.size and linger.ms: the accumulator

The record accumulator is the producer's beating heart, and `batch.size` plus `linger.ms` are the two knobs that govern it. Get these two right and you have done eighty percent of producer throughput tuning. Get them wrong — leave them at defaults under a firehose — and you ship tiny batches forever.

`batch.size` is a *per-partition* byte ceiling on a single batch. Its default is 16384 bytes — sixteen kilobytes. When the producer accumulates records for a partition and the current batch reaches 16 KB, that batch is closed and becomes eligible to send; new records start a fresh batch. This is the maximum a batch can grow to. A common and costly misreading is to think `batch.size` is how big batches *will* be. It is not. It is the cap. How big batches *actually* get depends on whether records arrive fast enough to fill that cap before the batch is sent for some other reason — and the "other reason" is `linger.ms`.

`linger.ms` is the knob that makes batching actually happen under realistic traffic. Its default is 0. With `linger.ms=0`, the sender thread sends a batch *as soon as it is ready*, and a batch is "ready" the instant there is any data in it and the sender thread is free to send — it does not wait for the batch to fill. So at `linger.ms=0`, under moderate traffic, you frequently ship batches containing a handful of records, or even one, because the sender thread keeps draining the accumulator the moment anything lands in it. The fixed per-request cost is paid constantly. This is the default, and it is tuned for *latency* (send immediately) at the direct expense of *throughput* (tiny batches).

![A timeline showing records accumulating into a batch until either the batch fills or the linger timer expires, then the batch flushes](/imgs/blogs/producer-optimization-batching-compression-acks-5.webp)

Setting `linger.ms` to a small positive number — 5, 10, 20 milliseconds — tells the sender thread: *when a batch is not yet full, wait up to this long for more records before sending it.* Now the trade is explicit. You pay up to `linger.ms` of added latency on each record, and in exchange records get the chance to pile up into much bigger batches, the fixed per-request cost is amortized over far more records, and throughput climbs. The batch flushes when *either* condition fires first: the batch hits `batch.size`, or the batch's age hits `linger.ms`. Figure 5 shows exactly this — records dribble in, the batch grows, and it ships at whichever boundary it reaches first.

### Why linger.ms is nearly free under load

Here is the counterintuitive part that makes `linger.ms` the easiest win in all of producer tuning: **under high load, a non-zero `linger.ms` adds almost no real latency.** When records are arriving fast, batches fill to `batch.size` long before the linger timer expires, so the timer never actually delays anything — the batch was going to ship on the size boundary anyway. The linger only kicks in during *lulls*, when traffic is light, and during a lull you do not have a throughput problem so a few extra milliseconds of latency costs nothing operationally. So `linger.ms=10` is, for a busy producer, a free lunch: it dramatically improves batching when traffic is bursty or moderate, and it is invisible when traffic is heavy. This is why nearly every high-throughput producer config sets `linger.ms` to something like 5–20 ms and leaves it there. The fear that linger "adds latency to every message" is misplaced; it adds latency only to messages that would have been sent in a wastefully small batch.

The interaction with `batch.size` is the other half. If you raise `linger.ms` but leave `batch.size` at 16 KB, your batches cap out at 16 KB no matter how long you wait — you bought time to fill a small bucket. For a real high-throughput pipeline you raise *both*: `batch.size` to 64 KB, 128 KB, or 256 KB to give batches room to grow, and `linger.ms` to 10–20 ms to give them time to. The two work together: `batch.size` sets the ceiling, `linger.ms` buys the time to approach it.

```java
// High-throughput producer batching config
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092,broker2:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer");

// The two batching knobs that matter most:
props.put("batch.size", 131072);   // 128 KB ceiling per batch (default 16 KB)
props.put("linger.ms", 20);        // wait up to 20 ms to fill a batch (default 0)

KafkaProducer<String, byte[]> producer = new KafkaProducer<>(props);
```

#### Worked example: linger 0 vs 10 ms at 100k msg/s of 200-byte records

Let us make the batching math concrete. You are producing 100,000 messages per second, each record 200 bytes serialized, to a topic with 10 partitions, evenly keyed so each partition gets 10,000 records per second. Assume no compression for now so we can see the raw batching effect.

**With `linger.ms=0`:** the sender thread drains each partition the moment records land. Suppose the sender wakes roughly every 1 ms (a reasonable cadence under load). In 1 ms, a partition receiving 10,000 records/s accumulates about 10 records, which is 2,000 bytes — well under the 16 KB cap. So you ship batches of about 10 records. Across 10 partitions that is roughly 10,000 produce-request payloads per second worth of batches. The per-request overhead — round trips, framing, broker bookkeeping — is paid on tiny 2 KB batches. Your request rate is high, your bytes-per-request is low, and the network and broker are doing a lot of bookkeeping for not much data.

**With `linger.ms=10`:** now each partition's batch is allowed to accumulate for up to 10 ms. In 10 ms a partition receiving 10,000 records/s accumulates about 100 records, which is 20,000 bytes — and that *exceeds* the 16 KB default cap, so with default `batch.size` the batch ships on the size boundary at ~16 KB (about 80 records) before the timer even fires. Raise `batch.size` to 64 KB and the batch ships on the 10 ms timer at ~100 records, ~20 KB. Either way you have gone from ~10 records per batch to ~80–100 records per batch — roughly a **10x reduction in the number of produce requests**, from ~10,000/s of tiny batches down to ~1,000/s of fat ones. Same data, one tenth the requests, one tenth the fixed per-request overhead. The added latency is at most 10 ms per record, and under this load most records wait far less because the batch fills early. That 10x is the difference between the producer that did 300k and the one that did 2M in the story that opened this post.

The lesson: at `linger.ms=0` your batch size is an accident of how fast the sender thread happens to wake; at `linger.ms=10` your batch size is a deliberate function of your traffic rate, and bigger batches are strictly cheaper per record.

## 3. Compression: codecs and the ratio-vs-CPU tradeoff

Compression is the second great lever, and it stacks multiplicatively with batching because **Kafka compresses whole batches, not individual records.** This single fact is why everything in section 2 pays off twice: a bigger batch not only amortizes the fixed per-request cost over more records, it also compresses *better*, because compression algorithms find more redundancy across many similar records than within one. Two hundred JSON records that all share the same field names and structure compress far better as one blob than as two hundred tiny independent blobs. Batching feeds compression; compression rewards batching.

`compression.type` is set on the producer (and can be overridden per topic on the broker, but the producer-side setting is what does the work on the send path). Its values are `none`, `gzip`, `snappy`, `lz4`, and `zstd`. They sit on a spectrum from "compresses a little, costs almost no CPU" to "compresses a lot, costs more CPU." The right pick depends on whether your bottleneck is network/disk (favor higher ratio) or CPU (favor faster codecs).

![A matrix comparing compression codecs none, gzip, snappy, lz4, and zstd across ratio, CPU cost, and speed](/imgs/blogs/producer-optimization-batching-compression-acks-2.webp)

Figure 2 maps the five codecs across the three dimensions that matter — compression ratio, CPU cost, and speed. Read it as a decision table, not a leaderboard, because the "best" codec depends entirely on which resource is scarce.

- **none** — zero CPU, zero ratio. Correct only when records are already compressed (you are shipping JPEGs or pre-gzipped blobs) or when your payloads are tiny and incompressible. Compressing already-compressed data wastes CPU for no gain.
- **gzip** — the highest ratio of the classic codecs, but the slowest and most CPU-hungry. Great when bandwidth or storage is your hard constraint and you have CPU to spare, painful when the producer is CPU-bound. Historically common, increasingly displaced by zstd which beats it on both axes.
- **snappy** — Google's codec built for speed over ratio. Low CPU, modest compression (often 2–4x on JSON). A long-time safe default because it almost never becomes the bottleneck. Fine, but zstd and lz4 now generally dominate it.
- **lz4** — very fast, low CPU, ratio similar to or slightly better than snappy. An excellent default for latency-sensitive high-throughput pipelines where you want compression's bandwidth savings without paying much CPU. If someone asks "what should I just set it to," lz4 is a defensible answer.
- **zstd** — the modern winner on the Pareto frontier. At its default level it achieves ratios close to or better than gzip while running far faster, and it is tunable across levels to trade ratio for speed. For most high-throughput pipelines where bandwidth and broker storage cost real money, zstd is the recommendation. It needs Kafka 2.1+ on both producer and broker.

### Why compressed batches save you twice

The bytes you save with compression are saved in two expensive places at once: on the network between producer and broker (and again on broker-to-follower replication, since the broker stores and replicates the already-compressed batch without recompressing it), and on the broker's disk where the log is persisted. A 4x compression ratio means a quarter of the network bytes *and* a quarter of the disk footprint *and* a quarter of the replication bandwidth — the cross-AZ replication tax from [Kafka replication and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) shrinks by the same factor. The consumer fetches the compressed batch and decompresses it on its own side, so compression also shrinks consumer fetch bandwidth. The producer pays the CPU once; the savings ripple through the entire pipeline. That asymmetry — pay CPU once at the producer, save bytes at four downstream stages — is why compression is almost always worth turning on for high-volume topics.

#### Worked example: 1 KB JSON at 70% reduction with zstd vs snappy

You are shipping 1 KB JSON records, 200,000 per second, so 200 MB/s of raw serialized payload. JSON is highly compressible — repeated field names, predictable structure — so the ratios are good.

**With snappy:** suppose snappy gets you 2.5x on this JSON (a typical figure), reducing 200 MB/s to about 80 MB/s on the wire. Snappy's CPU cost is low — call it a few percent of a core per 100 MB/s of throughput on modern hardware, so a handful of percent of CPU on the producer. You saved 120 MB/s of bandwidth almost for free.

**With zstd:** suppose zstd at its default level gets you 3.3x — a 70% reduction — taking 200 MB/s down to about 60 MB/s. That is another 20 MB/s saved over snappy, and across producer-to-broker plus broker-to-two-followers replication that 20 MB/s is multiplied by the replication fan-out, so the real network and storage savings are larger than the headline number. The cost is more CPU than snappy — perhaps two to three times snappy's CPU for this workload, still a modest single-digit-to-low-double-digit percentage of a core at this rate. If your producers have CPU headroom (they usually do — they are I/O bound waiting on the network), zstd's better ratio is nearly pure profit: 25% less bandwidth and storage than snappy for CPU you were not using anyway.

The decision rule falls out cleanly: **if the producer has spare CPU and bandwidth or storage costs money, pick zstd.** If the producer is genuinely CPU-bound (rare, but it happens with huge record volumes on small instances), step down to lz4 or snappy to trade ratio for cycles. Only pick `none` when the data is already compressed. And always remember the multiplier from section 2: whatever codec you choose, bigger batches make it compress better, so `batch.size`, `linger.ms`, and `compression.type` should be tuned together, never in isolation.

### Where compression actually happens, and the recompression trap

It is worth being precise about *which* component does the compression work, because a subtle misconfiguration can make the broker silently recompress every batch and burn CPU you did not budget for. By default, the producer compresses each batch with the codec you set, and the broker stores the batch *as it arrived* — compressed, untouched — and replicates that same compressed batch to followers and serves it to consumers, who decompress on their own side. The producer pays CPU once; nobody in the middle recompresses. This is the efficient path and it is what you get when the broker's per-topic `compression.type` is left at its default value of `producer`, which means "keep whatever the producer sent."

The trap is setting a *different* codec on the broker than on the producer. If a topic's `compression.type` is set to, say, `gzip` while producers send `zstd`, the broker cannot store the batch as-is — it must decompress each incoming batch and recompress it to gzip before writing. Now the broker is doing heavy compression work on the hot write path for every batch from every producer, which is exactly the CPU you were trying to keep off the brokers. The fix is almost always to leave broker-side `compression.type=producer` and let the producer's choice flow through untouched. Only override it on the broker when you have a deliberate reason — for instance, forcing a uniform codec across a fleet of producers you do not control — and then budget the broker CPU for it explicitly.

There is a related subtlety with the message format and the magic byte. Modern Kafka (message format v2) compresses the batch as a single compressed blob with the batch header left readable, so the broker can inspect offsets and timestamps without decompressing the payload — that is what lets it store and replicate the compressed batch cheaply. Very old clients or very old message formats lacked this and forced more broker-side work; on any current cluster you have the efficient v2 path, but it is one more reason to keep clients reasonably up to date. The headline rule stands: compress at the producer, set the broker to `producer`, and the bytes you save are saved everywhere downstream with the CPU paid exactly once.

```java
// Compression that stacks with batching
props.put("compression.type", "zstd");   // best ratio-per-CPU for most JSON/Avro
props.put("batch.size", 131072);          // bigger batches => better compression ratio
props.put("linger.ms", 20);               // time to fill those bigger batches
```

## 4. buffer.memory, max.block.ms, and producer backpressure

So far we have made batches bigger and compressed them. But there is a failure mode lurking: what happens when your application produces faster than the sender thread can drain to the brokers? The records pile up in the accumulator, and the accumulator is not infinite. The knobs that govern this are `buffer.memory` and `max.block.ms`, and they implement the producer's backpressure — the mechanism that stops a runaway producer from exhausting its own heap.

`buffer.memory` is the total bytes the producer may use for the record accumulator across all partitions — the size of the buffer that decouples your application thread from the network. Its default is 33554432 bytes, 32 MB. Every un-sent record lives here: records in open batches, closed batches waiting for the sender, batches in flight. When records are produced faster than they drain, this buffer fills.

![A stack diagram of the producer buffer showing the accumulator holding per-partition batches with a slice reserved for in-flight requests](/imgs/blogs/producer-optimization-batching-compression-acks-4.webp)

Figure 4 shows the buffer's internal structure: the `buffer.memory` pool, carved into per-partition batches inside the accumulator, with the in-flight requests representing batches already handed to the network. The total of all of it is bounded by `buffer.memory`. When that bound is hit, the next `send()` call cannot immediately allocate space for its record — and here is where `max.block.ms` enters.

`max.block.ms` controls how long a `send()` call will *block* waiting for buffer space (or for metadata, like the partition count of a topic it has not seen before) before giving up. Its default is 60000 ms — one minute. When the buffer is full, `send()` does not fail instantly and it does not silently drop the record; it blocks the calling thread, applying backpressure up into your application, for up to `max.block.ms`. If space frees up within that window (the sender drains a batch), `send()` proceeds. If the window expires with the buffer still full, `send()` throws a `TimeoutException`.

### Backpressure is a feature, not a bug

This blocking behavior is the whole point, and people who do not understand it write panicked posts about "Kafka producer hangs." It is not hanging; it is applying backpressure exactly as designed. When your application produces faster than the network can absorb, *something* has to give: either you buffer unboundedly (and OOM), drop data silently (and lose records), or slow the producer down (backpressure). Kafka chose backpressure. A full buffer that blocks `send()` is the producer telling your application "I cannot keep up; stop handing me records so fast." The right response in your application is to let that backpressure propagate — slow down whatever is generating the records — not to crank `max.block.ms` to infinity or `buffer.memory` to the moon and pretend the imbalance does not exist.

The two knobs give you two distinct levers. Raising `buffer.memory` gives the producer a deeper buffer to absorb *bursts* — if your traffic is spiky but the average rate is within the network's capacity, a bigger buffer smooths the spikes so `send()` rarely blocks. This is the right fix for bursty workloads. But raising `buffer.memory` does *not* fix a *sustained* overload: if your average produce rate exceeds what the brokers can absorb, a bigger buffer just delays the inevitable — it fills more slowly but it still fills, and then you block anyway, having burned more heap. For a sustained overload the only real fixes are to speed up the drain (more partitions, more brokers, bigger batches, compression) or slow the source.

`max.block.ms` is the patience knob: how long to wait at a full buffer before declaring failure. Setting it too low turns transient buffer pressure into spurious `send()` failures during normal bursts; setting it to a huge value means a genuinely stuck producer (brokers unreachable) blocks your threads for a long time before you find out something is wrong. A common production choice is to keep `max.block.ms` modest — tens of seconds — and treat a `TimeoutException` from `send()` as a real signal that you are overloaded or the brokers are unreachable, surfaced to monitoring rather than swallowed.

```java
// Backpressure tuning for a bursty high-throughput producer
props.put("buffer.memory", 134217728);  // 128 MB pool to absorb bursts (default 32 MB)
props.put("max.block.ms", 30000);        // block up to 30s on a full buffer, then fail loudly
// If send() throws TimeoutException here, you are overloaded — alert, don't ignore.
```

There is one more interaction worth flagging: `buffer.memory` must comfortably exceed `batch.size` times the number of partitions you actively write to, or you cannot even hold one batch per partition in flight and the producer thrashes. If you write to 1,000 partitions with `batch.size=128KB`, that is 128 MB just to hold one batch each — your `buffer.memory` needs to be larger than that. High partition counts and big batches both push `buffer.memory` up; size it deliberately.

### A worked sizing for the buffer

Make the sizing concrete so it stops being a vibe. Suppose you write to 200 partitions with `batch.size=128KB` and you want the producer to tolerate a brief broker hiccup — say up to 2 seconds where the sender thread cannot drain — without blocking `send()`. First, the floor: holding one open batch per partition is 200 × 128 KB = 25.6 MB just for the open batches, before any closed batches queue up behind them. The default 32 MB barely clears that floor and leaves almost nothing for burst absorption, which is exactly why high-partition-count producers on default `buffer.memory` block far more than people expect.

Now the burst headroom. If your steady produce rate is 100 MB/s and you want to ride out a 2-second drain stall, you need 200 MB of slack on top of the open-batch floor to buffer the records that arrive while the sender is stalled. So `buffer.memory` of roughly 256 MB (the 25.6 MB floor plus ~200 MB of burst plus margin) is the right order of magnitude for this workload — eight times the default. The general formula is: `buffer.memory ≈ (active_partitions × batch.size) + (peak_produce_rate × max_tolerable_stall)`. The first term is the structural floor; the second is the burst budget. Size to the sum, not to a number you copied from a blog.

The corollary is a sizing *sanity check* that catches a lot of mistakes: if `active_partitions × batch.size` alone exceeds your `buffer.memory`, the producer cannot even hold one open batch per partition and will block or thrash constantly regardless of load — a pure misconfiguration. People hit this by cranking `batch.size` to 256 KB and writing to thousands of partitions while leaving `buffer.memory` at the 32 MB default, then blaming Kafka for "hanging." Run the arithmetic before you blame the broker; the buffer math almost always explains a producer that blocks under light load.

## 5. acks and the throughput cost of durability

Now we cross from the throughput knobs into the knob that is *not* a throughput knob even though it lives on the throughput-critical send path: `acks`. This is the producer's durability dial, and the single most important thing to understand is that **you tune `acks` for the durability your data requires, and then you make throughput work around that choice — never the other way around.** If you treat `acks` as a throughput knob and turn it down to go faster, you are trading data safety for speed, often without realizing it, and that is how acknowledged records get lost.

`acks` controls how many broker-side acknowledgements the producer waits for before it considers a produce request successful and fires your callback. It has three values:

- **`acks=0`** — fire and forget. The producer does not wait for any acknowledgement; it writes the request to the socket and immediately considers the record sent. Highest throughput, lowest latency, *zero durability guarantee*. If the broker is down, the leader changes, or the network drops the request, the record is gone and your producer never knows. Acceptable only for genuinely lossy-tolerant data — high-volume metrics, telemetry, logs where losing a sliver under failure is fine.
- **`acks=1`** — wait for the *leader* to write the record to its log, then acknowledge. The producer knows the leader received it. But the leader has *not* waited for followers to replicate it. If the leader acknowledges and then dies before any follower copies the record, the record is lost even though your producer got a success. This is the dangerous default-looking middle ground: it feels safe and is faster than `acks=all`, but it loses data on leader failure.
- **`acks=all`** (also written `acks=-1`) — wait until the record is replicated to all in-sync replicas, then acknowledge. Combined with `min.insync.replicas=2` on the broker, this guarantees the record is on at least two brokers before your producer hears success, so a single broker failure loses nothing. Highest durability, and the highest latency because it waits for the replication round trip.

![A before-and-after comparison of acks=all as the safe slower choice versus acks=1 as the faster weaker choice with the durability note](/imgs/blogs/producer-optimization-batching-compression-acks-7.webp)

Figure 7 puts the trade side by side: `acks=all` is safe and slower; `acks=1` is faster and weaker, losing data on leader failure. The durability mechanics — why `min.insync.replicas` and not `acks` is the real durability knob, what the in-sync replica set is, how the high watermark advances — are the entire subject of [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability), and I will not re-derive them here. What matters for *producer tuning* is the cost.

### The throughput cost is smaller than people fear

Here is the part that surprises engineers who avoid `acks=all` for performance reasons: **with batching and pipelining on, the throughput cost of `acks=all` is much smaller than its latency cost.** The latency cost is real and unavoidable — each request now waits for the replication round trip, so per-request latency goes up by roughly the leader-to-follower round trip time (a couple of milliseconds same-AZ, more cross-AZ). But *throughput* is about how much data you move per second, not how long one request takes, and throughput is preserved by having *multiple requests in flight at once*. While one batch is waiting for `acks=all` replication, the next batches are already on the wire. The pipeline (controlled by `max.in.flight.requests.per.connection`, the next section) hides the per-request latency behind concurrency. So `acks=all` raises your latency by a few milliseconds but, properly configured, costs you only a modest slice of throughput, often 10–25% rather than the 2x penalty people imagine.

This is why the right framing is: pick `acks=all` for any data you cannot afford to lose, accept the few-millisecond latency tax, and then recover throughput with batching, compression, and enough in-flight requests to keep the pipeline full. The mistake is dropping to `acks=1` "for speed" and quietly accepting silent data loss on every leader election — of which there are more than you think, because every broker restart, every deploy, every reassignment triggers leader changes. The throughput you buy with `acks=1` is small; the durability you sell is large. For the durability-versus-availability framing of all this, [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) is the deeper read, and [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) sits underneath it.

```java
// Durability first, then recover throughput around it
props.put("acks", "all");                // wait for in-sync replicas; lose nothing on 1 broker failure
props.put("enable.idempotence", true);   // (next section) makes acks=all + retries safe and ordered
// Broker side: min.insync.replicas=2 with replication.factor=3 is the canonical durable config.
```

## 6. max.in.flight, retries, and reordering

We just leaned on "keep multiple requests in flight" to recover throughput under `acks=all`. The knob that controls that is `max.in.flight.requests.per.connection`, and it carries a subtle, infamous trap: turned up without the idempotent producer, it can silently *reorder* your records on a retry. This is the knob that most often bites teams who care about ordering, so let us take it apart precisely.

`max.in.flight.requests.per.connection` is how many produce requests the producer will send to a *single broker connection* without waiting for the responses. Its default in modern Kafka is 5. With a value of 5, the producer can have five batches in flight to a given broker at once — request 1 sent, request 2 sent, ... request 5 sent, and only then does it wait. This pipelining is exactly what hides the per-request `acks=all` latency: while request 1 is waiting for replication, requests 2 through 5 are already on the wire. More in-flight requests means more of the latency is overlapped, which is why you want this above 1 for throughput.

The trap is *retries*. Suppose `max.in.flight=5` and request 1 and request 2 are both in flight to the same partition. Request 1 fails transiently (a momentary leader hiccup, a timeout) while request 2 succeeds. The producer retries request 1. Now request 2's records were written to the log *before* request 1's retried records — even though request 1's records were produced first. The batches landed out of order. If you depend on per-partition ordering (and most people who set keys do), this is a silent correctness bug: the log no longer reflects produce order.

![A taxonomy tree of producer tuning knobs grouped into batching, compression, durability, and ordering families using parent fields](/imgs/blogs/producer-optimization-batching-compression-acks-8.webp)

Figure 8 organizes all the knobs we are covering into families — batching, compression, durability, ordering — and `max.in.flight` sits squarely in the ordering family precisely because of this retry hazard. Historically, the only safe way to *guarantee* no reordering was `max.in.flight.requests.per.connection=1`: with only one request in flight at a time, a retry of request 1 completes before request 2 is ever sent, so order is preserved — at the cost of throughput, because you lost all the pipelining.

### The idempotent producer makes high in-flight safe

The modern answer, and the reason you almost never have to set `max.in.flight=1` anymore, is the **idempotent producer**: `enable.idempotence=true`. When idempotence is on, the producer attaches a producer ID and a per-partition monotonic *sequence number* to every batch. The broker tracks the last sequence number it accepted per partition per producer and *rejects out-of-order or duplicate batches*. If request 1 is retried after request 2 already landed, the broker sees request 1's sequence number is now out of order relative to what it expects and refuses it, forcing the producer to re-establish correct order. The net effect: with idempotence on, the producer guarantees both *no duplicates* (a retried batch that already committed is recognized and dropped) and *no reordering*, even with up to 5 requests in flight. You get pipelining throughput *and* ordering, together.

This is why modern Kafka (since 3.0) turns `enable.idempotence=true` on by default and caps `max.in.flight` at 5 when it is on. The old advice "set max.in.flight=1 for ordering" is obsolete; the correct modern advice is "turn on idempotence and leave max.in.flight at 5." You lose nothing and gain both ordering and exactly-once-into-the-log semantics on the produce side. The deeper exactly-once story — transactions, the transactional coordinator, exactly-once across read-process-write — is its own subject in [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions); for plain producer tuning, just know that idempotence is free safety and you should leave it on.

| Config combination | In-flight pipelining | Ordering on retry | Duplicates on retry | Verdict |
| --- | --- | --- | --- | --- |
| `max.in.flight=1`, no idempotence | None (serial) | Preserved | Possible | Old, slow, ordered |
| `max.in.flight=5`, no idempotence | Full | **Can reorder** | Possible | Fast but unsafe |
| `max.in.flight=5`, idempotence on | Full | Preserved | Prevented | Modern default — use this |
| `acks=0` | Full | N/A (no retries) | N/A | Lossy, never for ordered data |

One subtlety the table hides: idempotence *requires* `acks=all` and `retries > 0` to function (the producer enforces this — turning on idempotence forces `acks=all`). That is not a coincidence; the whole idempotence guarantee is built on top of waiting for the in-sync replicas and retrying transient failures safely. So "turn on idempotence" and "use acks=all" are really one decision, which is convenient: the durable, ordered, no-duplicate configuration is a single coherent setting, not a fragile combination you have to assemble by hand.

### retries, delivery.timeout.ms, and the real failure boundary

The word "retries" deserves a closer look, because the knob named `retries` is not the one that actually bounds how long the producer keeps trying. In older clients, `retries` was a literal count — retry a failed batch up to N times, then give up — and people set it to small numbers and were surprised when a transient broker blip caused a hard failure after three quick retries. Modern Kafka changed the model: `retries` defaults to a very large number (effectively unbounded), and the *real* boundary is `delivery.timeout.ms`, which defaults to 120000 ms — two minutes. `delivery.timeout.ms` is the total wall-clock budget from the moment `send()` returns until the producer either succeeds or gives up, *including* all the time spent batching in the accumulator, all the retries, and all the backoff between retries. When that budget is exhausted, the record fails permanently and your callback sees the exception.

This is the boundary that matters for end-to-end behavior, and it composes cleanly: `delivery.timeout.ms` must be at least `linger.ms` plus `request.timeout.ms`, because a record can spend up to `linger.ms` waiting in the accumulator and then needs at least one `request.timeout.ms` window to be sent. The producer enforces this lower bound. The practical implication is that you tune the *time budget*, not the retry count: set `delivery.timeout.ms` to how long you are willing to keep trying to deliver a record before declaring it failed and routing it to a fallback. For most pipelines two minutes is generous; for a latency-bounded system you might shorten it so a stuck producer fails fast and your application can react. Between retries the producer waits `retry.backoff.ms` (default 100 ms) so a struggling broker is not hammered, and that backoff time counts against the delivery budget too.

The interaction with idempotence is what makes this safe rather than dangerous. Without idempotence, aggressive retries are a double-edged sword — they recover from transient failures but they are exactly what can produce duplicates and reordering, because a retried batch might land after a later batch already committed. With idempotence on, the producer can retry as much as `delivery.timeout.ms` allows and the broker's sequence-number check guarantees the retries neither duplicate nor reorder anything. So the modern, correct posture is: idempotence on, `retries` left at its large default, `delivery.timeout.ms` set to your real delivery deadline, and the callback handling only the genuine permanent failures that survive that whole budget. You get aggressive, safe recovery and a clean failure signal at a deadline you chose.

## 7. The sticky partitioner and bigger batches

There is one more lever that makes every batch bigger, and it operates entirely on the partition-selection stage we glossed over in section 1: the **sticky partitioner**. It applies to records *without* keys, and it is the quiet hero that fixed a long-standing inefficiency in how keyless records were spread across partitions. Understanding it ties the whole batching story together.

Records *with* keys are partitioned by hashing the key, which is non-negotiable because that hash is what gives you per-key ordering — all records for user 42 must go to the same partition or their order is lost. So the sticky partitioner does not touch keyed records. But records *without* keys have no such constraint; they may go to any partition, and the only goal is to spread them evenly for load balance. The old default partitioner achieved even spread by round-robining: record 1 to partition 0, record 2 to partition 1, record 3 to partition 2, and so on, cycling through partitions one record at a time.

Round-robin spreads load beautifully but it *destroys batching*. Think about it with the accumulator in mind: if consecutive records go to different partitions, then each partition's batch accumulates only every Nth record (for N partitions). With 10 partitions, a given partition's batch gets one record, then waits while nine other records go elsewhere, then gets another. Batches fill ten times slower, so under `linger.ms` they ship ten times emptier — you have spread the records so thin across partitions that no single partition's batch ever fills up. Round-robin's even spread is the direct enemy of the big batches section 2 worked so hard to build.

![A branching flow of the sticky partitioner filling one partition's batch to capacity before rotating to the next partition](/imgs/blogs/producer-optimization-batching-compression-acks-9.webp)

The sticky partitioner, introduced in Kafka 2.4 and the default ever since, fixes this by inverting the strategy: instead of spreading consecutive records across partitions, it *sticks* to one partition and sends all keyless records there *until that partition's batch is full (or lingers out and ships)*, then rotates to a new partition for the next batch. Figure 9 shows the rotation: fill partition 2's batch, ship it, switch to partition 5, fill its batch, ship it, switch to partition 0, and so on. Each batch is now full because all the records that built it went to the *same* partition consecutively. You get the big batches back.

### Why sticky still balances load over time

The natural worry is: doesn't sticking to one partition unbalance the load? In the short term, momentarily, yes — one partition gets a burst while others wait. But the producer rotates partitions every time a batch closes, and over any meaningful window (seconds), every partition receives roughly the same number of full batches, so the load is balanced *over time* even though it is bursty *instant to instant*. Consumers do not care about millisecond-scale burstiness; they care about the average, and the average is even. So the sticky partitioner gets you the best of both: full, efficient batches *and* balanced partition load over time. The benchmark numbers Kafka published when it shipped were dramatic — substantially higher throughput and lower latency for keyless workloads, because batches went from one-tenth full to full, with no downside to balance.

The practical takeaways are short. First, if your records have no natural ordering key, *leave them keyless* and let the sticky partitioner do its job — do not invent a fake key just to "control" partitioning, because a key forces hash-based partitioning and gives up the sticky batching win. Second, if your records *do* need a key for ordering, you keep hash partitioning and the sticky partitioner does not apply — your batching efficiency for keyed records then depends on having enough records per key-hashed partition to fill batches, which is one more reason not to over-partition a topic (too many partitions thins every batch). Third, this is yet another knob you do not have to touch: the sticky partitioner has been the default since 2.4, so on any modern cluster you already have it. The lesson is to understand it so you do not accidentally defeat it by keying records that did not need keys.

## 8. Async send, callbacks, and error handling

Everything above assumes you are using the producer the way it was designed to be used: *asynchronously*. The shape of your `send()` calls determines whether you actually get the batching and pipelining the knobs enable, or whether you accidentally serialize everything and throw the throughput away. This section is about the API discipline that makes the tuning real.

`producer.send(record)` returns a `Future<RecordMetadata>` and does *not* block waiting for the broker (it blocks only briefly if the buffer is full, per section 4). The record goes into the accumulator and your thread moves on to the next `send()`. This is what lets records pile into batches: your application keeps producing while the sender thread drains in the background. The throughput knobs only work because send is async — if you blocked on every record, batches would contain exactly one record because you would not call `send()` again until the previous one was fully acknowledged.

The catastrophic anti-pattern is calling `.get()` on the returned future after every send:

```java
// ANTI-PATTERN: synchronous send destroys batching and throughput
for (Record r : records) {
    producer.send(toProducerRecord(r)).get();  // blocks until this record is acked!
}
```

That `.get()` blocks your thread until the record is fully acknowledged by the broker before you call `send()` again. Now you have one record per request, no batching, no pipelining, and the per-request latency (including the full `acks=all` replication round trip) is paid *serially* on every single record. Throughput collapses to roughly one-over-the-round-trip-time — a few thousand records per second at best, where async would do millions. People do this because it is the obvious way to "make sure the send succeeded," and it is the single most common reason a producer is mysteriously slow. If you see `.send(...).get()` in a hot loop, you have found the bottleneck.

### Callbacks: handle errors without blocking

The right way to know whether a send succeeded — without giving up async — is the *callback*. `send()` takes an optional `Callback` that the producer invokes on the sender thread when the record is acknowledged or finally fails:

```java
// Correct async send with a callback for error handling
producer.send(record, (metadata, exception) -> {
    if (exception != null) {
        // Final failure after retries are exhausted. Log, alert, route to a
        // fallback, or stash the record for replay — do NOT silently drop it.
        log.error("send failed for key {}: {}", record.key(), exception.toString());
        deadLetterLocally(record);
    } else {
        // Success: record is on the broker at this partition/offset.
        metrics.recordSent(metadata.partition(), metadata.offset());
    }
});
// Thread does NOT block here — it loops to the next send and keeps batches filling.
```

The callback fires asynchronously, so your producing thread never blocks; it keeps feeding the accumulator. The exception argument is non-null only after the producer has exhausted its retries on a *retriable* error, or immediately on a *non-retriable* one. Two classes of error matter here. **Retriable errors** — leader-not-available, network timeouts, not-enough-replicas — the producer retries automatically (controlled by `retries`, effectively infinite by default in modern versions, bounded by `delivery.timeout.ms`); your callback only sees them if retries are exhausted. **Non-retriable errors** — a record larger than `max.request.size`, a serialization failure, an authorization error — fail immediately and your callback must handle them, because retrying will never help.

The one ordering caveat with callbacks: callbacks for records sent to the *same partition* are guaranteed to execute in the order the records were sent, which is useful if your callback logic depends on order. But callbacks across different partitions have no ordering guarantee. And callbacks run on the sender thread, so they must be *fast and non-blocking* — never do slow I/O or, worse, call `producer.flush()` or `send().get()` inside a callback, because you will stall the very thread that does all the producer's network work and deadlock or cripple throughput.

Finally, `flush()` and `close()`. `producer.flush()` blocks until all buffered records are sent and acknowledged — use it at well-defined checkpoints (end of a batch job, before a graceful shutdown) to ensure nothing is left in the accumulator, *not* in your hot loop. `producer.close()` flushes and releases resources; always call it (or use try-with-resources) so you do not lose records that were still buffered when the process exits. The discipline is: send async with callbacks in the hot path, flush at checkpoints, close on shutdown.

### Throughput from concurrency, not from blocking

There is a tempting middle ground people reach for when they want both confirmation and throughput: blocking on each future but from *many threads at once*, so that while one thread blocks on `.get()` another is sending. This works, in the sense that you do get pipelining back, but it is the wrong tool and it teaches the wrong instinct. The producer is already thread-safe and already does the concurrency for you on its single sender thread; throwing dozens of application threads at it to recover the parallelism you destroyed with `.get()` is fighting the design. You burn application threads, you complicate ordering reasoning (now callback and `.get()` ordering interleave across threads), and you gain nothing over the simple async-with-callback pattern that one thread can already saturate. The clean mental model is: one producing thread (or a handful, partitioned by some natural key so each owns its own keys) calling `send()` with callbacks, and the producer's internal sender thread doing all the network concurrency. Reach for more application threads only when *serialization* (turning your objects into bytes) is genuinely CPU-heavy and you need parallelism on that specific stage — and even then, keep the `send()` itself async.

The other subtle async hazard is *partial failure visibility*. Because `send()` returns before the broker has acked, a record can fail seconds after the `send()` call returned successfully, and that failure surfaces only in the callback. Code that does not register a callback (or registers one that swallows the exception) will *silently lose records* on permanent failures and never know. This is the async cousin of the `.send().get()` anti-pattern: one over-blocks and kills throughput, the other under-checks and loses data invisibly. The correct pattern threads the needle — async send for throughput, a real callback for the error you must not ignore, and a fallback (dead-letter to local disk, a retry queue, an alert) for the records that exhaust their delivery budget. Treat a non-null exception in the callback as a first-class event in your design, not an afterthought, and your pipeline is both fast and honest about loss.

## 9. A high-throughput tuning recipe

Now assemble everything into one coherent configuration for a real high-throughput pipeline, with the reasoning for every line, so you are not copying a magic incantation. The scenario: a clickstream ingestion pipeline, hundreds of thousands of events per second, records around 500 bytes of JSON, where you can tolerate a few extra milliseconds of latency but you *cannot* tolerate losing acknowledged events, and per-key ordering matters for events that carry a session key.

![A decision matrix mapping producer knobs to their effect on throughput, latency, and durability](/imgs/blogs/producer-optimization-batching-compression-acks-6.webp)

Figure 6 is the matrix to keep next to you while you tune: each knob against its effect on throughput, latency, and durability. Read down a column to see which knobs move the metric you care about; read across a row to see what each knob costs on the other axes. The recipe below is one consistent point in that space — the high-throughput, durable, ordered corner.

```java
// High-throughput, durable, ordered producer recipe
Properties p = new Properties();
p.put("bootstrap.servers", "b1:9092,b2:9092,b3:9092");
p.put("key.serializer",   "org.apache.kafka.common.serialization.StringSerializer");
p.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer");

// --- Batching: amortize the fixed per-request cost ---
p.put("batch.size", 131072);   // 128 KB ceiling — room for big batches
p.put("linger.ms", 20);        // wait up to 20 ms to fill them (free under load)

// --- Compression: stacks with batching, saves bytes 4 ways downstream ---
p.put("compression.type", "zstd");  // best ratio-per-CPU for JSON; lz4 if CPU-bound

// --- Buffer / backpressure: absorb bursts, fail loudly on sustained overload ---
p.put("buffer.memory", 134217728);  // 128 MB pool
p.put("max.block.ms", 30000);       // 30s patience, then TimeoutException -> alert

// --- Durability + ordering: one coherent decision ---
p.put("acks", "all");                              // lose nothing on 1 broker failure
p.put("enable.idempotence", true);                 // no dupes, no reorder, forces acks=all
p.put("max.in.flight.requests.per.connection", 5); // pipelining hides acks=all latency

KafkaProducer<String, byte[]> producer = new KafkaProducer<>(p);
```

Walk the reasoning one block at a time. **Batching** (`batch.size=128KB`, `linger.ms=20`) gives batches both room and time to grow, taking you from tiny accidental batches to fat deliberate ones — the 10x request-rate reduction from the section 2 worked example. **Compression** (`zstd`) stacks on top: bigger batches compress better, and zstd's ratio saves bandwidth and storage four ways downstream while costing CPU the producer was not using. **Buffer and backpressure** (`buffer.memory=128MB`, `max.block.ms=30000`) absorb bursts and make sustained overload visible as a loud `TimeoutException` rather than a silent stall or OOM. **Durability and ordering** (`acks=all` + `enable.idempotence=true` + `max.in.flight=5`) is the single coherent decision from sections 5 and 6: you lose no acknowledged record on a single broker failure, you get no duplicates and no reordering on retries, and the five in-flight requests pipeline enough concurrency to hide the `acks=all` replication latency so throughput stays high.

### What to measure and adjust

A recipe is a starting point, not a final answer. Instrument the producer's own JMX metrics and let them tell you what to turn next:

- **`batch-size-avg`** — if this is far below your `batch.size`, batches are not filling; raise `linger.ms` or check whether you are over-partitioned or accidentally keying records that thin every batch.
- **`record-queue-time-avg`** — how long records wait in the accumulator. Rising sharply means you are buffer-bound; the sender cannot drain fast enough, so look at broker capacity or partition count.
- **`request-latency-avg`** — per-request round trip. Climbing means broker or replication pressure; this is where `acks=all` shows up.
- **`buffer-available-bytes`** approaching zero, or **`record-error-rate`** above zero — backpressure is firing. Either bursts exceed the buffer (raise `buffer.memory`) or you are in sustained overload (scale the drain, not the buffer).
- **`compression-rate-avg`** — your actual compression ratio. If it is poor, batches may be too small to compress well (raise `batch.size`/`linger.ms`) or the data is already compressed (consider `none`).

The tuning loop is: set the recipe, push representative load, watch `batch-size-avg` and `record-queue-time-avg`, and adjust `linger.ms`, `batch.size`, and partition count until batches fill and queue time is stable. Throughput follows batch fullness almost mechanically.

## Case studies and war stories

**The synchronous-send firehose.** The opening story is the most common producer incident there is, so let me name its anatomy. A team builds an ingestion service, sends each event with `producer.send(record).get()` because that is the obvious way to confirm success, and ships it. It works in testing at low volume. In production at scale it tops out at a few thousand events per second per instance and they scale horizontally — dozens of instances — to brute-force the throughput, burning money on machines that are each 99% idle waiting on round trips. The fix was deleting `.get()` and adding a callback: one instance then did what twenty had been doing, because async send finally let batches form. The lesson: the *shape* of your send calls caps your throughput before any config knob does. No amount of `batch.size` tuning helps if `.get()` serializes every record.

**The acks=1 data-loss audit.** A payments-adjacent team ran `acks=1` for two years "for performance." During a routine broker upgrade — a rolling restart that triggers a leader election on every partition as leadership migrates off the restarting broker — they lost a small number of acknowledged records on each partition whose leader changed mid-write, because those records had been acked by the old leader but not yet replicated when leadership moved. It surfaced as a reconciliation discrepancy weeks later, hard to trace because the producer had reported success. They moved to `acks=all` with `min.insync.replicas=2` and measured the cost: latency up a few milliseconds, throughput down about 15% after re-tuning in-flight requests, and zero further loss across the next year of rolling restarts. The lesson: `acks=1` does not lose data in steady state — it loses data during *leader elections*, which happen on *every* deploy and restart, which is exactly when you are not watching. The performance you bought was small; the loss you accepted was invisible until it was a reconciliation nightmare.

**The reorder-on-retry mystery.** A team that needed strict per-key ordering set `max.in.flight.requests.per.connection=5` for throughput, on a Kafka version old enough that idempotence was off by default, and did not connect the two. Under normal conditions order was fine. During a network blip that caused a few request timeouts and retries, a handful of keys had their events written out of order — event B before event A for the same key — corrupting a downstream state machine that assumed ordering. It was maddening to reproduce because it only happened under retries. Turning on `enable.idempotence=true` fixed it permanently: the producer's sequence numbers let the broker reject the out-of-order retry and preserve order even at `max.in.flight=5`. The lesson: `max.in.flight>1` without idempotence is a latent ordering bug that only fires under retries — exactly when you are already having a bad day. Modern defaults (idempotence on) close this, but plenty of older configs and pinned client versions still carry the hazard.

**The compression-saved-the-bandwidth-bill story.** A high-volume analytics pipeline was shipping verbose JSON uncompressed across availability zones, and the cross-AZ data transfer charge was a meaningful line item — the kind that shows up in a FinOps review. Turning on `zstd` with bigger batches cut the JSON by about 70%, and because the broker stores and replicates the compressed batch as-is, the savings hit producer-to-broker traffic, cross-AZ replication traffic, and broker disk *simultaneously*. The cross-AZ transfer bill dropped by roughly the compression ratio. Producer CPU rose by a few percent on instances that had been I/O-bound and idle on CPU anyway. The lesson: compression is not just a throughput knob, it is a *cost* knob, and on cross-AZ or cross-region topics the dollars saved on data transfer often dwarf the CPU spent — pay CPU once at the producer, save bytes at every downstream hop.

## When to reach for these knobs (and when not to)

**Reach for batching (`linger.ms` + `batch.size`) almost always.** It is the highest-leverage, lowest-risk producer tuning there is. Any producer doing more than a trickle of traffic benefits from `linger.ms` in the 5–20 ms range and a `batch.size` raised to 64–256 KB. The only reason not to is an ultra-low-latency path where even 5 ms of linger is unacceptable — and even then, set `linger.ms=0` rather than fighting the accumulator, and accept that you are choosing latency over throughput deliberately. Read [Throughput vs latency](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff) before you decide you are that latency-sensitive; most people are not.

**Reach for compression on any high-volume topic** where bandwidth or storage costs money, which is most of them at scale. Default to `zstd` if your producers have CPU headroom (they usually do), `lz4` if they are genuinely CPU-bound, `none` only for already-compressed payloads. The savings ripple through network, replication, and disk; the cost is CPU you are probably not using.

**Set `acks=all` and `enable.idempotence=true` for any data you cannot afford to lose or reorder** — which is most business data. Treat this as the default and only step down to `acks=1` or `acks=0` for explicitly lossy-tolerant streams like metrics or debug logs where you have consciously decided a sliver of loss under failure is acceptable. Do not trade durability for the small throughput gain; recover throughput with batching and pipelining instead.

**Do not over-partition to chase producer throughput.** More partitions thin every batch (each partition's accumulator gets fewer records), hurting batching and compression, and they raise the `buffer.memory` you need. Partition for consumer parallelism and ordering requirements, per [message ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees), not as a producer-throughput lever — the producer-throughput lever is bigger batches, not more partitions.

**Do not raise `buffer.memory` to paper over sustained overload.** A bigger buffer absorbs bursts; it does not fix an average produce rate that exceeds broker capacity. If `send()` blocks chronically, the fix is to speed the drain (more brokers, bigger batches, compression) or slow the source — not to keep enlarging the buffer until you OOM.

**Never put `.send(...).get()` in a hot loop.** If you take one thing from this post, it is that synchronous send caps your throughput below any config knob. Send async with callbacks; flush at checkpoints; close on shutdown.

## Key takeaways

- **The send stage has a fixed per-request cost; batching amortizes it.** Every throughput knob exists to make batches bigger so that cost is paid less often. That is the whole game.
- **`linger.ms` is nearly free under load.** Batches fill to `batch.size` before the timer fires when traffic is heavy, so a non-zero linger adds latency only during lulls, when you do not have a throughput problem. Set it to 5–20 ms and stop worrying.
- **`batch.size` is a ceiling, not a target.** Raise it *and* `linger.ms` together; `batch.size` gives batches room, `linger.ms` gives them time. Either alone underperforms.
- **Kafka compresses whole batches, so bigger batches compress better.** Compression stacks multiplicatively with batching. Default to `zstd`, drop to `lz4` if CPU-bound, `none` only for pre-compressed data — and the savings hit network, replication, and disk at once.
- **`acks` is a durability knob, not a throughput knob.** Pick `acks=all` for data you cannot lose, accept the few-millisecond latency tax, and recover throughput with batching and pipelining. `acks=1` loses data on every leader election — meaning every deploy.
- **`max.in.flight>1` without idempotence reorders on retry.** Turn on `enable.idempotence=true` and you get pipelining throughput *and* ordering *and* no duplicates, all at `max.in.flight=5`. The old "set it to 1 for ordering" advice is obsolete.
- **The sticky partitioner makes keyless batches full.** It fills one partition's batch before rotating, balancing load over time while keeping batches fat. Do not invent fake keys that defeat it.
- **Async send with callbacks is mandatory for throughput.** `.send().get()` in a loop serializes every record and collapses throughput by orders of magnitude. Callbacks handle errors without blocking; flush at checkpoints; close on shutdown.
- **`buffer.memory` + `max.block.ms` are backpressure, not a bug.** A blocking `send()` is the producer telling you it cannot keep up. Bigger buffer absorbs bursts; it does not fix sustained overload.
- **Tune by metric, not superstition.** Watch `batch-size-avg`, `record-queue-time-avg`, `request-latency-avg`, and `compression-rate-avg`. Throughput follows batch fullness almost mechanically.

## Further reading

- [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — the durability machinery behind `acks=all`, `min.insync.replicas`, and the in-sync replica set.
- [Message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — why keys force hash partitioning, how ordering is per-partition, and why over-partitioning hurts.
- [Throughput vs latency: the tuning tradeoff](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff) — the general dial that every knob in this post sits on.
- [Exactly-once in Kafka: idempotent producer and transactions](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) — where the idempotent producer leads next, and transactional exactly-once.
- [Schema management and evolution](/blog/software-development/message-queue/schema-management-evolution-avro-protobuf-registry) — serializer and schema choice, which quietly drives payload size and thus batching and compression.
- [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the consistency-versus-availability framing under the `acks` decision.
- [Apache Kafka producer configuration reference](https://kafka.apache.org/documentation/#producerconfigs) — the authoritative list of every producer knob and its default.

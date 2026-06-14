---
title: "Kafka Deep Dive, Part 1: The Log, Partitions, Segments, and the Page Cache"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Kafka sustains millions of messages a second on ordinary disks because of a few brutally simple storage decisions: an append-only log, fixed-size segments with sparse indexes, the OS page cache instead of the JVM heap, and zero-copy reads. This is the storage and performance deep dive that explains exactly why."
tags:
  [
    "message-queue",
    "kafka",
    "storage-engine",
    "page-cache",
    "zero-copy",
    "log-compaction",
    "distributed-systems",
    "event-driven",
    "performance",
    "segments",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-1.webp"
---

The first time I watched a single three-broker Kafka cluster swallow a sustained 900 MB per second of ingest without breaking a sweat, I did not believe the dashboards. The brokers were ordinary cloud instances with spinning-rust-era throughput expectations baked into my intuition, the JVM heaps were tiny — eight gigabytes each on machines holding terabytes of data — and yet the produce latency p99 sat under ten milliseconds while the consumers, dozens of them, replayed days of history at line rate. My mental model said this was impossible. Disks are slow. Random I/O is the enemy. The heap should be exploding. Every instinct I had carried over from building databases told me the numbers on the screen were a lie.

They were not a lie. They were the direct, almost boring consequence of a handful of storage decisions that Kafka makes and refuses to compromise on. Kafka is fast not because it is clever in some exotic way, but because it is relentlessly, aggressively simple about how bytes hit the disk and how they leave it. A partition is an append-only file. Appends are sequential, so the disk head never seeks. The cache is the operating system's page cache, not a JVM object graph, so a broker can hold terabytes of hot data with a tiny heap and survive a restart with the cache still warm. And reads to consumers skip user space entirely through the `sendfile` system call, so a fetch goes disk to page cache to network card without the data ever being copied into the broker process at all. None of these are tricks. They are the whole product.

This post is the storage and performance deep dive. It is a companion to the conceptual overview in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log), which explains *what* a log is and why treating Kafka as a queue is a category error — but that post spends exactly one figure on storage. Here we go all the way down to the bytes. We will open a partition directory and look at the actual files. We will trace a single produce all the way to a dirty page and a single fetch all the way through `sendfile`. We will do the throughput arithmetic that lets you size a cluster on the back of a napkin, and we will pull apart log compaction, the most misunderstood retention mode in the system. If you have read the [anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers), you already know the producer-broker-consumer shape; now we are going to make the broker's storage layer concrete. The figure below is where we start: a partition is not one file, it is a chain of segments.

![A partition shown as a left-to-right chain of closed segments followed by one active segment where new records are appended and offsets increase along the chain](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-1.webp)

By the end you will be able to do four concrete things. You will be able to look at a partition directory on a broker and read it like a map — which files are the records, which are the indexes, what their names mean, and which one is being written right now. You will be able to trace any consumer fetch from an offset to a byte position on disk and explain why it costs a binary search and a tiny forward scan rather than a full scan. You will be able to write a throughput and disk-sizing budget that a capacity planning review will not laugh at. And you will understand log compaction well enough to use Kafka as a durable key-value changelog, not just an event firehose. This is Part 1 of a three-part Kafka deep dive; Part 2 covers [consumer groups and the rebalance protocol](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing), and Part 3 covers [replication, the ISR, and the high-watermark](/blog/software-development/message-queue/kafka-replication-isr-acks-durability).

## 1. The partition is a log on disk

Start with the smallest true statement we can make about Kafka storage: **a partition is an append-only sequence of records, stored as files in a directory on one broker's disk.** That is the whole foundation. Everything else — segments, indexes, retention, compaction, replication — is machinery built around that one fact. If you internalize nothing else, internalize that a partition is a log, the log is on disk, and the only write operation the log ever performs on the hot path is *append to the end*.

Let us make it tangible. A topic named `payments` with twelve partitions is, on disk, twelve directories spread across the brokers that lead those partitions. On a broker leading partition 3, you will find a directory called `payments-3`. Inside it are files. If you list that directory you see something like this:

```bash
$ ls -la /var/lib/kafka/data/payments-3/
total 2098304
-rw-r--r-- 1 kafka kafka      10485760 00000000000000000000.index
-rw-r--r-- 1 kafka kafka    1073741821 00000000000000000000.log
-rw-r--r-- 1 kafka kafka      10485756 00000000000000000000.timeindex
-rw-r--r-- 1 kafka kafka      10485760 00000000000001048576.index
-rw-r--r-- 1 kafka kafka     734183122 00000000000001048576.log
-rw-r--r-- 1 kafka kafka      10485756 00000000000001048576.timeindex
-rw-r--r-- 1 kafka kafka            12 leader-epoch-checkpoint
-rw-r--r-- 1 kafka kafka            43 partition.metadata
```

There is the whole storage engine, sitting in plain sight. The big `.log` files are the records. The `.index` and `.timeindex` files are lookup structures we will dissect shortly. The long zero-padded numbers in the filenames are *base offsets* — the offset of the first record in each file. We will come back to all of it. For now, notice the shape: a partition is many files, the records live in `.log` files, and those files are named by where they start in the offset space.

### An offset is a position, not an ID

The single most important concept in Kafka storage is the **offset**. An offset is a 64-bit integer that identifies a record's position within a partition. It is monotonically increasing and dense: the first record in a partition is offset 0, the next is 1, the next is 2, and so on, with no gaps in a normal (non-compacted) log. The offset is assigned by the partition leader at the moment the record is appended, and it never changes. Offset 5,242,901 in `payments-3` means *the 5,242,902nd record ever appended to that partition* (zero-indexed), forever, on every replica.

This is profoundly different from a message ID in a traditional queue. In RabbitMQ or SQS, a message has an opaque identifier, the broker tracks which messages have been delivered and acknowledged, and once acknowledged a message is gone. There is no notion of "the 5-millionth message" because the broker is constantly deleting the middle of its dataset as consumers ack. Kafka does the opposite. It never deletes the middle. Records are appended at the tail and expired in bulk from the head, and the offset is a stable coordinate into that immutable sequence. A consumer's position is just an offset it remembers. Replaying history is just resetting an offset backward. Two consumer groups reading the same partition independently is just two different remembered offsets reading the same files. The offset being a position rather than an ID is the source of nearly every superpower Kafka has over a classic queue.

### Why "on one broker" matters

Notice I said a partition lives on *one* broker's disk — specifically, the leader's. (Replicas hold copies on other brokers; we cover that in Part 3.) This is not an accident or a limitation to engineer around; it is the load-balancing unit of the entire system. A topic's traffic is sharded across its partitions, and each partition's I/O lands on exactly one disk on one machine at a time. This is why **partition count is a capacity decision**: it sets the maximum write parallelism for a topic, because two partitions can be appended to in parallel on two brokers, but one partition is always one append stream on one disk. If you want more write throughput than one disk can sustain sequentially, you add partitions, not replicas. The [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) post develops this idea in the general case; Kafka is one concrete, opinionated instance of it.

Because a partition is a single sequential append stream on one disk, it gets to make an assumption that random-access databases cannot: it can always write to the end and never seek. That assumption is the foundation everything in section 3 is built on. But first we have to look at the fact that a partition is not literally one giant file — it is split into segments, and the split is what makes retention, indexing, and recovery tractable.

## 2. Segments: active and closed, and the file trio

A partition is conceptually an infinite log, but you cannot store an infinite log as one file. A single file that grows forever is a disaster: you cannot delete the old part without rewriting the whole thing, you cannot index it efficiently, and recovery after a crash means re-reading terabytes. So Kafka splits each partition's log into **segments** — fixed-size chunks, by default capped at 1 GB or one week of data, whichever comes first. Figure 2 shows what one segment actually is on disk.

![A grid showing a single segment as three sibling files, the dot-log holding records, the dot-index mapping offsets to byte positions, and the dot-timeindex mapping timestamps to offsets, all named by the same base offset](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-2.webp)

At any moment, exactly one segment per partition is the **active segment**: the one currently being appended to. All the others are **closed segments**: immutable, read-only, finished. When the active segment fills up — hits its size cap (`segment.bytes`, default 1 GB) or its time cap (`segment.ms`, default 7 days) — Kafka *rolls*: it closes the current active segment, makes it read-only, and opens a brand-new active segment whose base offset is the next offset after the last record of the one just closed. Appends now go to the new active segment. This is the chain you saw in figure 1: a sequence of closed segments, each frozen, ending in one active segment that is still growing.

### The file trio: .log, .index, .timeindex

Each segment is not one file but three siblings that share a base-offset filename prefix:

- **`.log`** — the actual records. This is the data. Each entry is a record batch (Kafka batches records together for efficiency) containing the records' keys, values, headers, timestamps, and the offsets. The `.log` file is the only one of the three that holds your message contents. Everything else is a lookup aid.
- **`.index`** — the *sparse offset index*. It maps a relative offset to a byte position within the `.log` file. "Sparse" is the key word: it does not have an entry for every record. By default it adds an entry roughly every 4 KB of `.log` written (`index.interval.bytes`). So if a segment holds a million records, the `.index` might hold only a few thousand entries. It is a coarse map: "offset around here is at byte position around there."
- **`.timeindex`** — the *timestamp index*. It maps a timestamp to a relative offset, also sparsely. This is what makes "give me all records since 9:00 AM" or `offsetsForTimes()` work without scanning. It lets a consumer seek by time instead of by offset.

The base offset in the filename is the offset of the first record in that segment. `00000000000001048576.log` holds records starting at offset 1,048,576. The `.index` and `.timeindex` store *relative* offsets (relative to the base) so they fit in four bytes instead of eight, halving the index size. This is a tiny detail with a real payoff: index entries are 8 bytes each (4-byte relative offset, 4-byte position) instead of 12, and over billions of records that adds up.

### Why split into segments at all

The segment boundary is where three operations become cheap that would otherwise be ruinously expensive:

1. **Retention deletion.** To expire old data, Kafka deletes whole closed segments — it unlinks the `.log`, `.index`, and `.timeindex` files. A file delete is O(1). If the log were one giant file, expiring the oldest hour would mean rewriting the entire file to chop off the front. With segments, you just `unlink()` the oldest segment's files. We will see this in section 7.
2. **Indexing.** A sparse index covers exactly one segment. The index file is memory-mapped, and because the segment has a known base offset, an offset lookup first picks the right segment by base offset (a binary search over a handful of segments), then binary-searches within that segment's small index. Bounded segments keep the index files small and the search shallow.
3. **Recovery.** After an unclean shutdown, Kafka must validate the log. It only needs to re-scan and rebuild indexes for the *active* segment (and any segment that was being written), not the entire partition history. Closed segments are immutable and already validated. This turns crash recovery from "re-read terabytes" into "re-read the last gigabyte."

The active-versus-closed distinction is doing a lot of work. Closed segments are immutable, which means they can be safely memory-mapped, served via zero-copy, deleted whole, and never re-validated. The active segment is the only mutable thing in the whole storage layer, and it is mutable only in the most restricted way possible: append at the end. That restriction is what the next section is about.

## 3. Sequential writes and why append-only is fast

Here is the claim that confuses people who learned that "disks are slow": **Kafka writes to disk on every produce, yet it is faster than systems that try to avoid disk.** The resolution to the paradox is that "disks are slow" is only true for *random* I/O. For *sequential* I/O, disks — even spinning disks — are shockingly fast, and Kafka's append-only design means every write is sequential.

### The seek is the enemy, not the disk

A spinning hard drive has a physical head that must move to the right track and wait for the platter to rotate the right sector under it. That mechanical seek-and-rotate costs on the order of 5 to 10 milliseconds. If you do random 4 KB reads or writes scattered across the platter, you pay that latency on every single one, and your effective throughput collapses to maybe a few hundred KB per second — a couple hundred IOPS times 4 KB. This is the number that makes people say disks are slow, and for random access it is true.

But if you write *sequentially* — keep appending to the same place, letting the head stay put while the platter streams under it — a single 7200 RPM drive sustains 100 to 200 MB per second. That is a thousandfold difference between random and sequential on the same physical device. The classic measurement, from Jeff Dean's numbers and reproduced in the original Kafka design notes, is that sequential disk access can be *faster than random memory access*, because random memory access blows the CPU cache and sequential disk access is one long streaming read that the drive and OS prefetch aggressively. SSDs narrow the gap — their random I/O is far better than a spinning drive's — but even on NVMe, sequential writes are still meaningfully faster and, crucially, they avoid write amplification and keep the flash translation layer happy.

Kafka's entire write path is engineered so that the *only* write that ever happens is an append to the end of the active segment's `.log` file. There is no in-place update. There is no reorganizing. There is no B-tree rebalancing, no compaction-into-the-middle, no random page writes. The head, or the SSD's write pointer, marches straight forward. This is why a Kafka broker can sustain write throughput that a random-access database on identical hardware cannot touch: the database is paying seek costs that Kafka structurally cannot incur.

### Batching compounds the win

Append-only is the foundation, but Kafka stacks two more multipliers on top. The first is **batching**. Producers do not send one record at a time; they accumulate records into batches (controlled by `batch.size`, default 16 KB, and `linger.ms`, the time to wait for a batch to fill). A batch is appended to the log as a single unit — one record batch header amortized over many records. This turns a thousand tiny appends into one larger sequential write, which is both fewer system calls and a longer sequential run. The second multiplier is **compression**. Set `compression.type` to `lz4`, `zstd`, `snappy`, or `gzip`, and the producer compresses the whole batch before sending. Because Kafka stores and transmits the batch *still compressed* — it does not decompress on the broker — compression reduces disk bytes written, disk bytes stored, network bytes replicated, and network bytes read by consumers, all at once. A 4x compression ratio is a 4x multiplier on effective throughput end to end.

Here is a producer configured for high sequential throughput:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092,broker2:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer");

// Batch aggressively: bigger batches = longer sequential writes on the broker
props.put("batch.size", 65536);        // 64 KB batches, up from the 16 KB default
props.put("linger.ms", 10);            // wait up to 10ms to fill a batch
props.put("compression.type", "lz4");  // compress the batch; broker stores it compressed
props.put("acks", "all");              // durability: wait for the in-sync replicas

// Allow more in-flight to keep the pipe full (ordering is preserved with idempotence on)
props.put("enable.idempotence", true);
props.put("max.in.flight.requests.per.connection", 5);

Producer<String, byte[]> producer = new KafkaProducer<>(props);
```

Notice `linger.ms=10`. That ten-millisecond wait feels like added latency, and it is, but it lets batches fill so that each append to the broker's log is a fat sequential write rather than a trickle of tiny ones. On a busy topic the batches fill before the linger expires anyway, so you pay nothing; on a quiet topic you pay ten milliseconds to get dramatically better efficiency. This is the kind of tradeoff that looks wrong until you understand that the disk wants long sequential runs.

### What "the write returned" actually means

There is a subtlety here that trips up everyone reasoning about durability, and it leads directly into the next section. When a producer's `send()` future completes successfully with `acks=all`, what has actually happened on disk? The honest answer is: **maybe nothing has hit the physical platter yet.** The record has been appended to the active segment's `.log` file — but "appended to the file" in Linux means "written into the page cache." The data is in RAM, marked dirty, and the OS will flush it to the physical disk on its own schedule. Kafka deliberately does *not* call `fsync()` on every write. Its durability comes from *replication*, not from forcing each write to the platter. That decision is the entire point of the next section, and it is the single most important performance idea in Kafka.

## 4. The page cache is the cache (not the JVM heap)

If you remember one thing from this entire post, remember this: **Kafka's cache is the operating system's page cache, and Kafka does almost nothing to manage it.** A Kafka broker is a JVM process, and JVM processes are famous for enormous heaps and garbage-collection pauses. Kafka sidesteps that entire class of problems by refusing to hold message data in the heap at all. The records live in the page cache — kernel-managed RAM that caches file contents — and the broker's job is mostly to shovel bytes between the network and the page cache, letting the kernel own the actual caching. Figure 5 traces the write path through it.

![A vertical stack showing the write path from a producer batch down through the broker append, into the OS page cache holding dirty pages, then an asynchronous flush, and finally a sequential write to the disk dot-log file](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-5.webp)

### What the page cache is

The page cache is a region of RAM the Linux kernel uses to cache the contents of files. When any process writes to a file, the bytes go into the page cache first, the page is marked *dirty*, and the kernel flushes dirty pages to the physical disk later, in the background, in efficient batches. When any process reads a file, the kernel checks the page cache first; if the data is there (a cache *hit*), it is served from RAM with no disk access at all. This is not Kafka behavior — it is how every file read and write on Linux works, for every program. Kafka's insight is to *lean on it completely* rather than building its own buffer pool on top of it.

When a producer appends a record, the broker does a plain file write. The bytes land in the page cache, dirty. The broker returns the produce response (subject to `acks`, which is about *replication*, not disk flush). The kernel flushes those dirty pages to disk asynchronously, governed by knobs like `vm.dirty_ratio` and `vm.dirty_background_ratio`. Kafka never blocks a produce on a disk flush. That is why produce latency is low even under heavy write load: the write completes when it hits RAM.

### Reads are served from the same RAM

The symmetry is the beautiful part. Because writes go *through* the page cache on their way to disk, the most-recently-written data is *already in the page cache*. And the most-recently-written data is exactly what consumers reading near the tail want. So a consumer caught up to the live edge of a partition reads entirely from RAM — its fetches hit pages that were populated milliseconds ago by the producer's write. The producer warms the cache for the consumer, for free, as a side effect of the write path. Kafka calls this its "single-cache" design: there is one copy of the hot data, in the page cache, shared by the write path and every read path. There is no separate read cache to keep coherent, no double-caching of the same bytes in both a JVM buffer pool and the OS cache.

This is why Kafka brokers run tiny heaps relative to their data. A broker holding 4 TB of partition data might run a 6 GB heap. The heap holds metadata, request buffers, the producer/consumer session state, the index memory-maps' bookkeeping — but *not the message data*. The message data is in the page cache, which is to say it is in the operating system's free RAM. Give the machine 64 GB of RAM, let Kafka take 6 GB of heap, and the kernel uses the remaining ~58 GB as page cache for your partition files. The more RAM, the more hot data stays in cache, the higher your cache hit rate, the less you touch disk.

```bash
# A broker holding terabytes of data with a small heap.
# Most of the machine's RAM is page cache, owned by the kernel, not the JVM.
$ free -g
              total        used        free      shared  buff/cache   available
Mem:             64           7           2           0          55           54
Swap:             0           0           0

# That 55 GB of buff/cache is your hot partition data. The JVM heap is the 7 GB "used".
$ jcmd $(pgrep -f kafka.Kafka) GC.heap_info | head -1
 garbage-first heap   total 6291456K, used 3145728K
```

### The warm-restart superpower

Here is the consequence that genuinely surprised me the first time I saw it. **The page cache survives a Kafka broker process restart.** The page cache belongs to the *kernel*, not to the Kafka JVM. If you stop the Kafka process and start it again — a rolling restart for an upgrade, say — the JVM heap is wiped and rebuilt from scratch, but the page cache is untouched, because the kernel never went anywhere. The new Kafka process opens the same partition files and finds their pages already hot in RAM. Compare this to a database with an in-process buffer pool: restart it and you start cold, every query hits disk, and you suffer a long, painful cache-warming period. Kafka restarts warm. The only thing that wipes the page cache is rebooting the *machine* (or memory pressure evicting the pages).

#### Worked example: heap vs page cache on a real broker

Take a broker with 64 GB of RAM holding 4 TB of partition data across, say, 200 partitions. Your working set — the data consumers actually read — is the recent tail of each partition. Suppose consumers stay within the last 30 minutes of data and you ingest 200 MB/s on this broker. Then the hot working set is 200 MB/s times 1,800 seconds, which is 360 GB. That does not fit in 58 GB of page cache, so you will *not* serve all reads from RAM — you will take some disk reads for the older-but-still-within-30-minutes data.

Now flip it: a broker with the same 64 GB of RAM but only ingesting 25 MB/s, with consumers staying within the last 30 minutes. The hot set is 25 MB/s times 1,800 seconds, which is 45 GB. That fits in 58 GB of page cache. This broker serves essentially all reads from RAM, hits the disk only for the occasional history-replay consumer, and runs cool. The lesson is sharp: your page-cache hit rate is a function of *ingest rate times how far behind your consumers fall*, divided by available RAM. The single worst thing for cache hit rate is a lagging consumer that drags the read working set deep into cold history. We will return to this as a case study, because it is the most common Kafka performance incident there is.

### The danger: a lagging consumer evicts the hot set

The flip side of the single-cache design is its failure mode. The page cache is finite and the kernel evicts least-recently-used pages under pressure. A consumer that falls far behind and starts replaying old history forces the kernel to read cold segments from disk into the page cache — and those cold pages evict the hot tail pages that all your *caught-up* consumers and your producers depend on. Now everyone is hitting disk, throughput craters, lag grows on the healthy consumers too, and you have a cascading slowdown that started with one slow consumer. This is the "page-cache eviction cliff," and it is why production Kafka operators watch consumer lag like hawks. We will quantify it later. For now, the headline stands: the page cache is the cache, it is shared, it is finite, and protecting its hit rate is the core of Kafka performance operations.

## 5. Zero-copy reads with sendfile

We have established that consumers reading near the tail are reading data that is already in the page cache. Now the question is: how does that data get from the page cache, through the broker, to the consumer over the network, as cheaply as possible? The answer is the second pillar of Kafka's read performance: **zero-copy via the `sendfile` system call.** Figure 4 contrasts the traditional path with the zero-copy path.

![A before-and-after comparison contrasting the traditional read path with four buffer copies through user space against the zero-copy sendfile path that moves data from page cache to the network card with no heap copy](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-4.webp)

### The traditional four-copy path

Consider, for a moment, how a naive server sends a file over a socket. The standard sequence is `read()` the file into a buffer, then `write()` that buffer to the socket. Internally, that path involves four copies and two context switches between user space and kernel space:

1. The kernel reads from disk into the **page cache** (one DMA copy from disk to RAM).
2. The kernel copies from the page cache into the **application's user-space buffer** (a CPU copy, kernel to user — context switch).
3. The application copies from its user-space buffer into the **kernel socket buffer** (a CPU copy, user to kernel — context switch).
4. The kernel copies from the socket buffer to the **network card** (one DMA copy).

Four copies, two of them done by the CPU shuffling bytes between kernel and user space, plus the context-switch overhead. For a server whose entire job is moving file bytes to sockets, those two redundant CPU copies and the context switches are pure waste — the data is identical at every step, it is just being shuffled around memory.

### The sendfile two-copy path

The `sendfile(2)` system call (and its cousin, the splice-based path) collapses this. You tell the kernel "send these bytes from this file descriptor directly to this socket descriptor," and the kernel does it without ever copying the data into user space at all:

1. The kernel reads from disk into the **page cache** (one DMA copy) — and if the data is already in the page cache, even this is skipped.
2. The kernel hands the page-cache pages directly to the **network card** (one DMA copy, often "gather" DMA that reads straight from the page cache).

Two copies, both DMA (done by hardware, not the CPU), zero copies into user space, and far fewer context switches. The data never enters the Kafka JVM. It goes page cache to NIC. The broker process is not a pipe the bytes flow through; it is a *coordinator* that tells the kernel which file range to splice to which socket.

In Java, Kafka uses `FileChannel.transferTo()`, which maps to `sendfile` on Linux:

```java
// Simplified shape of how Kafka serves a fetch.
// fileChannel is the segment's .log file; socketChannel is the consumer connection.
// transferTo() invokes sendfile under the hood: page cache -> socket, no user-space copy.
long position = indexLookup(targetOffset);   // find byte offset in the .log
long bytesToSend = fetchSize;                  // how much the consumer asked for
fileChannel.transferTo(position, bytesToSend, socketChannel);
// The record bytes never touch a Java byte[] on the heap.
```

### Why zero-copy and compression are friends

Zero-copy has a constraint that shapes the rest of the system: **the broker can only `sendfile` bytes it does not need to transform.** If the broker had to decompress, re-serialize, or rewrite each record before sending, it would have to pull the bytes into user space, defeating the whole mechanism. This is exactly why Kafka stores record batches in *the same format on disk, in the page cache, and on the wire*, and stores them *compressed*. A producer compresses a batch, the broker writes that compressed batch to the log verbatim, and when a consumer fetches, the broker `sendfile`s the compressed batch straight from the page cache to the socket — and the *consumer* decompresses. The broker never decompresses. The on-disk format and the wire format are deliberately identical so that the bytes can flow untouched. This is a design constraint masquerading as a feature, and it is why Kafka's read path is so cheap: the broker is barely involved.

The combined effect of the page cache and zero-copy is that a Kafka broker serving caught-up consumers is doing almost no work per byte. The bytes are in RAM (page cache) and they go straight to the NIC (sendfile). The CPU is not copying data, the heap is not allocating, the garbage collector is not running over message data. A single broker can saturate a 10 GbE link this way with CPU to spare. That is the performance story in one sentence: **caught-up reads are RAM-to-NIC DMA, and the broker barely touches the data.**

## 6. How a fetch resolves an offset (the sparse index)

We keep saying a consumer "fetches an offset" and the broker "finds the byte position." Now let us make that mechanism exact, because it is a small, elegant piece of engineering and understanding it dissolves a lot of confusion about why some reads are cheap and some are not. The question is: given a consumer asking for offset 5,242,901, how does the broker find the bytes on disk without scanning the whole partition? Figure 3 shows the resolution path.

![A pipeline showing an offset fetch flowing from the requested offset through picking the right segment, a binary search of the sparse dot-index, a resulting file position, and a short forward scan of under four kilobytes](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-3.webp)

### Step one: pick the segment

The broker holds, for each partition, the list of segments sorted by base offset: segment 0 starting at offset 0, segment 1 starting at 1,048,576, segment 2 at 2,097,152, and so on. To find offset 5,242,901, it binary-searches this list for the segment whose base offset is the largest value not greater than the target — the "floor." If segment base offsets are 0, 1.05M, 2.10M, 3.15M, 4.19M, 5.24M, 6.29M, then 5,242,901 falls in the segment with base offset 5,242,880 (5.24M). With a handful of segments this is a couple of comparisons. Even a partition with thousands of segments is a 10-or-12-comparison binary search. Picking the segment is essentially free.

### Step two: binary-search the sparse index

Now the broker opens that segment's `.index` file — which is memory-mapped, so "opening" it is just touching mapped memory, likely already in the page cache. The `.index` is an array of fixed-size 8-byte entries, each a (relative offset, byte position) pair, sorted by offset. Because it is sparse, it has an entry only roughly every 4 KB of `.log` data, not every record. The broker binary-searches this array for the largest relative offset *not greater than* the target's relative offset. The target relative offset is 5,242,901 − 5,242,880 = 21. The index might have entries at relative offsets 0, 18, 37, 51, ... mapping to byte positions 0, 4096, 8192, 12288, .... The floor entry for 21 is the one at relative offset 18, byte position 4096. So the broker now knows: the record for offset 5,242,901 is *at or after* byte position 4096 in this `.log` file. The binary search over a sparse index of a few thousand entries is, again, around a dozen comparisons.

### Step three: scan forward a little

The sparse index got us *close* but not exact — it pointed at byte position 4096, which is the position of the record at offset 5,242,898 (relative offset 18), not our target 5,242,901. So the broker seeks to byte 4096 in the `.log` and scans *forward*, reading record batches, until it reaches offset 5,242,901. Because the index granularity is 4 KB, this forward scan is bounded: it reads at most about 4 KB before hitting the target. And those 4 KB are almost certainly in the page cache. So the "scan" is a few-microsecond read of one page of RAM. The total cost of resolving an offset is: one binary search over segments, one binary search over a sparse index, and a sub-4-KB forward scan. No full file read. No table scan. Logarithmic plus a tiny constant.

#### Worked example: walk an offset lookup to a byte position

Let us do the whole thing concretely. A partition `payments-3` has these segments (base offsets): segment files named `...0000000000.log`, `...1048576.log`, `...2097152.log`, `...3145728.log`, `...4194304.log`, `...5242880.log`. A consumer requests offset **5,242,901**.

1. **Segment selection.** Binary-search base offsets [0, 1048576, 2097152, 3145728, 4194304, 5242880] for the floor of 5,242,901. The floor is 5,242,880 — the last segment in the list. Open `00000000000005242880.log` and its `.index`.
2. **Relative offset.** Target relative offset = 5,242,901 − 5,242,880 = **21**.
3. **Index search.** The `.index` (sparse, ~one entry per 4 KB) for this segment contains, say, entries: (relOff 0, pos 0), (relOff 18, pos 4096), (relOff 41, pos 8192), (relOff 67, pos 12288). Binary-search for the floor of relative offset 21. That is (relOff 18, pos 4096). So we know the record is at or after byte **4096**.
4. **Forward scan.** Seek to byte 4096 in the `.log`. Read the record batch there — it starts at offset 5,242,898. Read the next records: 5,242,899, 5,242,900, and **5,242,901**. Found. The scan read perhaps 200 bytes (three small records) before hitting the target, well under the 4 KB index granularity bound.
5. **Serve.** From offset 5,242,901's byte position, `sendfile` the requested fetch size straight from page cache to the consumer's socket.

The entire lookup touched two memory-mapped index reads and a few hundred bytes of `.log`, all from RAM. This is why Kafka can resolve any offset in any partition in microseconds, whether the partition holds a thousand records or a trillion. The sparse index trades a tiny forward scan (cheap, sequential, in cache) for an index that is small enough to keep fully memory-mapped (you do not want a dense index that is as big as the data). It is a textbook space-time tradeoff, tuned so that both space and time are small.

### Why the index is sparse on purpose

A natural objection: why not a *dense* index with an entry per record, so the lookup is exact and there is no forward scan? Because a dense index would be enormous — comparable in size to the data itself for small records — and it would have to be kept in memory to be fast. The sparse index is small enough to memory-map entirely and let the page cache hold, while the forward scan it forces is so cheap (sequential, in-cache, sub-4-KB) that making it exact would save microseconds at the cost of gigabytes of index. Kafka chose the right point on the curve. The `index.interval.bytes` knob lets you tune it: smaller interval means a denser index and shorter scans (good for tiny random reads), larger interval means a sparser index and slightly longer scans (good for streaming reads). The default 4 KB suits almost everyone.

## 7. Retention and segment rolling

Kafka keeps data for a while and then deletes it. The "while" is **retention**, and the mechanism by which it deletes is **segment rolling and deletion**. Because a partition is a chain of immutable closed segments plus one active segment, expiring data is gloriously simple: delete whole closed segments from the head of the chain. Figure 6 lays out the lifecycle over time.

![A timeline showing the active segment filling to one gigabyte, rolling into a new active segment, then days later the oldest segment aging past retention and being deleted whole, leaving a steady seven days of data on disk](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-6.webp)

### Rolling: closing the active segment

A new active segment is opened, and the current one closed, when any of these triggers fire:

- **Size:** the active segment reaches `segment.bytes` (default 1 GB). This is the common trigger on busy topics.
- **Time:** the active segment reaches `segment.ms` (default 7 days) since its first record. This is the common trigger on quiet topics — a low-traffic partition might take a week to fill 1 GB, so the time cap rolls it sooner.

When a roll happens, the active segment is closed (made immutable, indexes finalized) and a new active segment is created with base offset equal to the next offset. Rolling is cheap — it is closing two file handles and opening three new ones. The reason both triggers exist is retention granularity: retention deletes *whole segments*, so you cannot delete data finer-grained than a segment. If your segments are 7 days of data, you cannot retain "the last 3 days" precisely — you would have to keep the whole 7-day segment until all of it ages out. Smaller segments give finer retention granularity at the cost of more files. This is a real tuning lever: a topic that needs tight retention bounds wants smaller `segment.bytes` or `segment.ms`.

### Retention: deleting the oldest segments

Two settings govern deletion in the default `delete` retention policy:

- **`retention.ms`** (time-based, default 7 days / 168 hours): a segment is eligible for deletion when the *largest* timestamp in it is older than `retention.ms`. Kafka checks the *whole segment's* newest record, not the oldest, so a segment lingers until *all* its data is past the retention edge.
- **`retention.bytes`** (size-based, default −1 / unlimited): when a partition's total size exceeds `retention.bytes`, the oldest segments are deleted until it fits. This caps disk usage per partition regardless of time.

A background thread checks eligibility periodically (`log.retention.check.interval.ms`, default 5 minutes). When a segment is eligible, Kafka deletes its `.log`, `.index`, and `.timeindex` files. The delete is not instant on disk — there is a brief grace via `file.delete.delay.ms` (default 60 seconds) to let in-flight reads finish — but it is effectively a few `unlink()` calls. No data is rewritten. No compaction-into-the-middle. The head of the log just gets shorter.

This is the retention model that makes Kafka a *replayable* log rather than a queue: data is not deleted when a consumer reads it (Kafka does not even track per-message acknowledgement). Data is deleted when it *ages out* or the partition *exceeds a size cap*, independent of who has or has not read it. A consumer that wants to replay from three days ago can, as long as three days is within retention. A consumer that falls behind retention loses the data it never read — that is the dreaded "retention cliff," where lag exceeding retention means permanent data loss for that consumer, because the segments holding the unread data get deleted on schedule.

```bash
# Set a 3-day retention and a 256 MB segment size for tighter retention granularity
kafka-configs.sh --bootstrap-server broker1:9092 \
  --alter --entity-type topics --entity-name payments \
  --add-config retention.ms=259200000,segment.bytes=268435456

# Cap each partition at 50 GB regardless of time (whichever limit hits first wins)
kafka-configs.sh --bootstrap-server broker1:9092 \
  --alter --entity-type topics --entity-name payments \
  --add-config retention.bytes=53687091200
```

#### Worked example: a throughput and disk-sizing budget

This is the calculation you do before you provision a cluster, and getting it slightly wrong is how you wake up to "disk full" alerts at 3 AM. Suppose one partition sustains a steady **1 GB/s** of compressed-on-disk write throughput (an aggressive single-partition number, but it makes the arithmetic clean). Suppose `segment.bytes` is **1 GB**, so the active segment rolls roughly **every second**. Suppose `retention.ms` is **7 days**.

- **Segments accumulated:** at one roll per second, 86,400 segments per day, times 7 days = **604,800 closed segments** held at steady state for this one partition. (That many files in one directory is itself a reason to use larger segments at this rate — but we keep 1 GB for the clean math.)
- **Disk per partition:** 1 GB/s × 86,400 s/day × 7 days = **604,800 GB ≈ 591 TB** for one partition's `.log` data alone. The `.index` and `.timeindex` add a small percentage (sparse, ~0.5–1% of `.log` size each), so call it ~595 TB.
- **Replication multiplier:** with replication factor 3 (the production default), the *cluster* stores three copies: 591 TB × 3 = **~1.77 PB** of raw disk for this one partition's data, spread across the brokers holding its replicas. We cover replication in Part 3, but it must be in your disk budget from day one — replication factor multiplies your storage bill directly.
- **The lever:** if 7 days is more than you need, retention is your biggest disk knob. Dropping to 3 days takes the per-partition footprint from ~591 TB to ~253 TB (×3 for replicas: ~760 TB). Dropping to 24 hours takes it to ~84 TB per partition (~253 TB replicated). Retention time scales disk linearly, and it is usually the cheapest dimension to cut.

The shape to remember: **disk = write_rate × retention_seconds × replication_factor**, plus a couple percent for indexes. Everything else is rounding. A more realistic number — say a topic ingesting 200 MB/s across 24 partitions, retained 3 days, replication 3 — is 200 MB/s × 259,200 s × 3 ÷ 1,000,000 ≈ **155 TB** of raw cluster disk, which you then divide across your brokers and round up generously for headroom, because a Kafka cluster that runs out of disk does not degrade gracefully — it stops accepting writes for the affected partitions, and recovery from a full disk is genuinely unpleasant.

## 8. Log compaction: the cleaner, dirty ratio, tombstones

So far retention has meant "delete old data by age or size." There is a second, very different retention policy: **log compaction**. Compaction changes the contract of the log from "keep a time window of events" to "keep the latest value for every key, forever." This is what turns a Kafka topic into a durable, replayable key-value changelog — the substrate for [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), for Kafka Streams state stores, for KRaft's own metadata log, and for any "current state per entity" use case. Figure 8 shows what compaction does to a log. The matrix in figure 7 contrasts the two retention modes head to head.

![A matrix comparing delete and compact retention across what is kept, disk growth, CPU cost, and best use case, showing delete keeps a time window while compact keeps the latest value per key](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-7.webp)

### What compaction guarantees

Set `cleanup.policy=compact` on a topic and Kafka promises: **for any key, the log retains at least the most recent record with that key.** Older records for the same key are eligible to be removed. So if a key `user-42` is written with values `v1`, then `v2`, then `v3`, compaction will eventually collapse the log so only `v3` survives for `user-42`. The offsets of the surviving records do *not* change — `v3` keeps whatever offset it was written at — and the log remains an ordered sequence; it just gets *sparser* as superseded versions are removed. A consumer that reads a fully compacted topic from offset 0 sees exactly one record per key (the latest), which is precisely "the current state of every entity." A consumer reading a partially compacted topic might still see some old versions that have not been cleaned yet, but never *more* than the full history and always *at least* the latest.

![A before-and-after comparison showing a log with all versions of each key on the left being compacted into a log holding only the latest record per key plus tombstones on the right](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-8.webp)

### How the log cleaner works

Compaction is performed by background threads called the **log cleaner** (`log.cleaner.threads`, default 1). The cleaner does not touch the active segment — only closed segments are compacted, so the most recent data is always intact and readable. The mechanism is a two-pass clean per partition:

1. **Build the offset map.** The cleaner scans the "dirty" portion of the log (the segments written since the last clean) and builds an in-memory hash map from key to the *highest offset* seen for that key. This map is the answer to "what is the latest version of each key." Its size is bounded by `log.cleaner.dedupe.buffer.size` (default 128 MB), which limits how many distinct keys one clean pass can handle — a real constraint on very high-cardinality topics.
2. **Rewrite the segments.** The cleaner then reads the old segments and writes *new* compacted segments, copying a record only if its offset equals the highest offset for its key in the map (i.e., it is the latest version) — and dropping it otherwise. The new segments replace the old ones, the old files are deleted, and the offsets of surviving records are preserved. This is the one place Kafka *does* rewrite segments rather than just append or delete, and it is why compaction costs CPU and I/O that plain `delete` retention does not.

### The dirty ratio: when does cleaning kick in

The cleaner does not run constantly — it would waste I/O re-cleaning logs that barely changed. It triggers on the **dirty ratio**: the fraction of the log (by bytes) that is "dirty," meaning written since the last clean and not yet compacted. The controlling knob is `min.cleanable.dirty.ratio` (default **0.5**). When a partition's dirty ratio exceeds 0.5 — half the log is uncleaned — it becomes eligible for the cleaner. A lower ratio cleans more aggressively (more CPU/IO, fresher compaction, less wasted disk on stale versions); a higher ratio cleans lazily (less CPU/IO, but more stale versions linger). There is also `min.compaction.lag.ms` (keep a record uncompacted for at least this long, so consumers have a window to see every version) and `max.compaction.lag.ms` (force a clean within this window even if the dirty ratio is low, important for regulatory deletion and for KRaft).

```bash
# A compacted changelog topic: keep the latest per key, clean reasonably aggressively
kafka-topics.sh --bootstrap-server broker1:9092 --create \
  --topic user-profiles \
  --partitions 12 --replication-factor 3 \
  --config cleanup.policy=compact \
  --config min.cleanable.dirty.ratio=0.3 \
  --config min.compaction.lag.ms=3600000 \
  --config segment.bytes=536870912

# A hybrid: compact AND delete — keep latest per key, but also age out keys
# untouched for 30 days (so deleted entities eventually leave the log entirely)
kafka-topics.sh --bootstrap-server broker1:9092 --create \
  --topic session-state \
  --partitions 24 --replication-factor 3 \
  --config cleanup.policy=compact,delete \
  --config retention.ms=2592000000
```

### Tombstones: how you delete a key

Compaction keeps the latest record per key — so how do you delete a key entirely? You write a **tombstone**: a record with that key and a **null value**. A null-value record is Kafka's delete marker. When the cleaner processes it, the tombstone supersedes all prior records for that key (latest-wins, and the latest is null), and consumers reading the changelog see the null and interpret it as "this key is deleted." Then, after a grace period (`delete.retention.ms`, default 24 hours), the cleaner removes the tombstone *itself*, so the key vanishes from the log entirely.

That grace period is load-bearing and the source of a classic bug. The tombstone must live long enough for *every* consumer to see it. If a consumer is offline or lagging when the tombstone is written, and the tombstone gets cleaned away before that consumer catches up, the consumer will *never see the delete* — it will see the old value, miss the tombstone, and keep a deleted key alive forever in its local state. This is the "tombstone that never arrived" bug, and it is why `delete.retention.ms` must exceed your worst-case consumer downtime. If a consumer can be offline for 48 hours, a 24-hour tombstone retention will silently corrupt its state. The fix is to widen the window, not to hope your consumers never lag.

#### Worked example: compacting a changelog

Say `user-profiles` (compacted) receives these records on one partition, in order, with offsets:

```
offset 100: key=u1 value={email: a@x.com}
offset 101: key=u2 value={email: b@x.com}
offset 102: key=u1 value={email: a2@x.com}   # u1 updated
offset 103: key=u3 value={email: c@x.com}
offset 104: key=u2 value=null                 # u2 tombstone (deleted)
offset 105: key=u1 value={email: a3@x.com}   # u1 updated again
```

Before compaction, the log holds all six records. The cleaner builds the key-to-latest-offset map: u1 to 105, u2 to 104, u3 to 103. It then rewrites the closed segment keeping only the latest per key:

```
offset 103: key=u3 value={email: c@x.com}     # latest u3, kept
offset 104: key=u2 value=null                 # tombstone, kept for delete.retention.ms
offset 105: key=u1 value={email: a3@x.com}   # latest u1, kept
```

Offsets 100, 101, 102 are gone — they were superseded. Note the surviving offsets (103, 104, 105) are *unchanged* from their original values; compaction never renumbers. A new consumer reading from offset 0 now reconstructs the exact current state: u1 has email a3, u3 has email c, and u2 is deleted (it sees the null at 104, and after 24 hours even that tombstone is cleaned and u2 disappears entirely). This is how a compacted topic *is* a database table: read it from the start, apply each record, and you have the current value of every key. The log is the table, replayable from byte zero.

## 9. Putting the numbers together: a throughput budget

We have all the pieces; let us assemble them into the mental model that lets you reason about whether a Kafka cluster will hold up. The headline performance facts compose: appends are sequential (section 3), writes return at page-cache speed (section 4), caught-up reads are RAM-to-NIC via zero-copy (section 5), and lookups are logarithmic-plus-tiny-scan (section 6). Figure 9 organizes the on-disk storage hierarchy these facts live in.

![A tree showing the Kafka on-disk storage taxonomy from a partition directory down to closed and active segments, and from each segment down to its dot-log records and its index files](/imgs/blogs/kafka-deep-dive-log-segments-page-cache-storage-9.webp)

### The per-broker throughput ceiling

A single broker's write throughput is bounded by the slower of two things: its disk sequential write bandwidth and its network ingest bandwidth (including replication traffic). With NVMe SSDs sustaining 1–3 GB/s sequential writes and 10–25 GbE networking (1.25–3.1 GB/s), modern brokers are often *network*-bound, not disk-bound — which would have astonished an engineer from the spinning-disk era and is the whole point of the sequential-write design. Read throughput, for caught-up consumers, is bounded by network egress, because zero-copy means the CPU is barely involved and the data is already in the page cache.

But there is a tax you must always include: **replication multiplies your effective write load.** With replication factor 3 and `acks=all`, every byte a producer sends is written once by the leader and then replicated to two followers — so the *cluster's* internal write-and-network load is roughly 3x the producer's input rate (1 leader write + 2 follower writes/transfers). If producers send 1 GB/s of new data to a topic with RF=3, the cluster is doing ~3 GB/s of disk writes and ~2 GB/s of inter-broker replication network traffic on top of the 1 GB/s ingest. This is covered in depth in Part 3, but it belongs in every capacity number: your throughput budget is producer rate times replication factor on the disk side, and producer rate times (replication factor − 1) on the inter-broker network side.

### Partitions set the parallelism

A single partition is one sequential append stream on one disk on one leader broker. To exceed one disk's sequential bandwidth for a topic, you spread the load across more partitions on more brokers. If one partition tops out around, say, 100 MB/s in your setup (limited by per-partition handling, not raw disk), and you need 1 GB/s for the topic, you need at least 10 partitions, ideally spread so their leaders land on different brokers. Partition count is thus your throughput-scaling dial, with the caveats from [message ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees): more partitions means more parallelism but per-key ordering only within a partition, and you cannot easily reduce partition count later. The [Kafka-as-a-log](/blog/software-development/database/kafka-as-a-distributed-log) post calls this "a capacity decision you cannot easily undo," and it is right.

### The page-cache hit rate is the hidden variable

The throughput numbers above assume reads hit the page cache. The moment your read working set exceeds available RAM — because consumers lag, or because you have a fan of history-replay consumers — reads start hitting disk, and disk reads *compete with* the sequential writes for disk bandwidth and, worse, introduce *random* I/O (seeking back to cold segments) that wrecks the sequential-write performance you depend on. So the real throughput model has a cliff: comfortably above the cliff, the broker is network-bound and serene; below it (cache misses), the broker is disk-bound and struggling, and the two regimes are separated by your consumers' lag relative to your RAM. The single most important operational metric in Kafka is therefore not throughput but **page-cache hit rate**, which you infer from disk read I/O — a healthy caught-up broker does almost *no* disk reads, because everything is served from cache. When you see disk read bytes climbing, a consumer is dragging the working set into cold history, and the eviction cliff is near.

### A complete back-of-envelope

Put it together for a concrete topic. You ingest **300 MB/s** of compressed data, replication factor **3**, retention **3 days**, and consumers stay within ~5 minutes of the tail.

- **Disk per cluster:** 300 MB/s × 259,200 s × 3 = **~233 TB**, plus ~2% indexes ≈ 238 TB. Divide by brokers, add headroom.
- **Internal write load:** 300 MB/s × 3 = **900 MB/s** of disk writes cluster-wide; ~600 MB/s of replication network traffic. Spread across, say, 6 brokers, that is ~150 MB/s of disk write per broker — trivial for NVMe.
- **Read working set:** 300 MB/s × 300 s (5 min) = **90 GB** of hot data being read. If each broker has 128 GB RAM and ~110 GB usable as page cache, the per-broker share of that 90 GB (15 GB on 6 brokers) fits comfortably — reads are served from RAM, brokers are network-bound and happy.
- **The failure mode to watch:** if a consumer falls 2 hours behind, its read set jumps to 300 MB/s × 7,200 s = **2.1 TB**, which does not fit in any broker's RAM. That consumer's fetches hit cold disk, dragging the whole broker's cache hit rate down. *This* is the number that turns a healthy cluster sick, and it is why lag alerting is non-negotiable.

The whole budget reduces to three multiplications and one inequality: disk is rate × retention × RF; internal load is rate × RF; the read set is rate × consumer-lag-seconds, and as long as that read set fits in page cache you are network-bound and fine. Memorize that and you can size a Kafka cluster on a whiteboard.

## Case studies and war stories

These are real failure modes — some I have hit, some are well-documented in the community — and each one is a direct consequence of a storage decision in this post.

### 1. The page-cache eviction cliff (the slow consumer that took down a healthy cluster)

A team ran a Kafka cluster serving a dozen real-time consumers, all caught up, all reading from the page cache, all healthy. Then a batch analytics job spun up a consumer that started reading from offset 0 — *days* of cold history — to backfill a data warehouse. Within minutes, the real-time consumers' latency spiked and lag began climbing on consumers that had been perfectly healthy. Nothing about the real-time path had changed.

The cause: the backfill consumer forced the kernel to read terabytes of *cold* segments from disk into the page cache, and the kernel, under memory pressure, evicted the *hot tail* pages to make room. Now the real-time consumers' fetches missed the cache and hit disk too — and those disk reads, random across cold segments, competed with the sequential producer writes. The whole broker went from network-bound (fast) to disk-bound (slow). The lesson: **the page cache is shared and finite, and a single history-replaying consumer can evict the working set every other consumer depends on.** The fixes are operational — throttle backfill consumers (`fetch.max.bytes`, quotas), run them off replica/follower fetch or a tiered-storage path, schedule them off-peak, or isolate heavy replay workloads on dedicated brokers. The root cause is the single-cache design, which is also the source of all the performance you love; you do not get one without the other.

### 2. The retention cliff (lag exceeded retention, data was deleted unread)

A downstream consumer fell behind during a multi-hour outage. The topic had `retention.ms` of 24 hours. The consumer was down for 26 hours. When it came back, it tried to resume from its last committed offset — and got an `OffsetOutOfRangeException`, because the segments holding the records between its committed offset and the current log had been *deleted by retention while it was down.* The data it never read was simply gone. Per `auto.offset.reset`, the consumer either jumped to the latest offset (silently skipping all the lost records) or to the earliest (re-reading from the new head, also skipping the gap). Either way, **records were permanently lost for that consumer, and no error fired at write time** — the producer's writes all succeeded; the loss was on the read side, governed entirely by the relationship between lag and retention. The lesson: retention is a *deadline* for your consumers, not just a disk-saving knob. Retention must exceed your worst-case consumer downtime plus catch-up time, or you are one outage away from silent loss. Alert on `(retention − max consumer lag in time)` getting small, not just on lag in messages.

### 3. The tombstone that vanished before a consumer saw the delete

A team used a compacted topic as the source of truth for a search index. A document was deleted: a tombstone (null value) was written for its key. Most consumers saw it and removed the document. But one consumer — a secondary index in another region — was lagging by 30 hours due to a cross-region replication hiccup. The topic's `delete.retention.ms` was the default 24 hours. By the time the lagging consumer reached that offset range, the cleaner had already *removed the tombstone itself*. The consumer saw the key's prior (non-null) value, never saw the delete, and kept the deleted document searchable indefinitely. The lesson: **for compacted topics, `delete.retention.ms` must exceed your slowest consumer's worst-case lag.** A tombstone is only a delete if every consumer sees it; clean it too soon and the delete silently fails to propagate. This is the compaction analog of the retention cliff, and it is sneakier because it corrupts state instead of throwing an exception.

### 4. The "fsync on every write" misconfiguration that destroyed throughput

An operator, worried about durability, set `flush.messages=1` and `flush.ms=0` on a high-throughput topic — forcing an `fsync()` to physical disk on every single message. Throughput collapsed by more than an order of magnitude, and produce latency went from single-digit milliseconds to hundreds of milliseconds. They had defeated the page-cache design: instead of letting writes accumulate in the page cache and flush asynchronously, every append now blocked on a physical disk sync. The lesson: **Kafka's durability comes from replication, not from per-message fsync.** With `acks=all` and replication factor 3, a record is durable once it is in the page cache of multiple brokers, because for all of them to lose it simultaneously you would need multiple correlated machine failures within the flush window. Forcing fsync per message buys you marginally stronger single-node durability at a catastrophic throughput cost, and it is almost always the wrong trade. Trust replication; let the OS flush on its own schedule. (The narrow exception: a single-broker dev cluster with RF=1, where there is no replication to lean on — but you should not be running production durability on RF=1 anyway.)

### 5. The directory with a million files (segments too small)

A team set `segment.bytes` very small (16 MB) on a high-throughput topic, reasoning that smaller segments give tighter retention granularity. At their ingest rate, each partition rolled a new segment every few seconds, and within days each partition directory held tens of thousands of segment files, each with its `.log`, `.index`, and `.timeindex` — hundreds of thousands of files per broker. File-handle limits started getting hit, the broker's startup time (which scans all segments) ballooned, and metadata operations slowed. The lesson: **segment size is a tradeoff between retention granularity and file count.** Too large and you cannot retain finely or recover quickly; too small and you drown in files and file handles. The 1 GB default is a sane center; deviate deliberately, and if you go small, raise your file-descriptor limits and watch broker startup time. Match `segment.bytes` to your ingest rate so segments roll on the order of minutes-to-hours, not seconds.

## When to reach for this knowledge (and when not to)

The storage internals in this post matter intensely for some decisions and not at all for others. Knowing when you need to descend to this level is itself a skill.

**Reach for it when you are sizing or tuning a cluster.** Disk = rate × retention × RF, the page-cache hit-rate model, the replication tax — these are not academic. They are the difference between a cluster that runs serenely for years and one that fills its disks or falls off the eviction cliff under load. Anyone signing off on a Kafka capacity plan must do this arithmetic.

**Reach for it when you are debugging a performance incident.** Almost every Kafka performance problem is a storage-layer phenomenon: a lagging consumer evicting the page cache, a backfill job introducing random disk reads, an fsync misconfiguration, segments too small. If you do not understand the page cache and zero-copy, these incidents look like inexplicable magic. With this model, they are diagnosable in minutes from a single metric (disk read bytes climbing).

**Reach for it when you are deciding between delete and compact retention, or designing a changelog.** If you are using Kafka as a key-value source of truth — for CDC, for Streams state, for materialized views — you must understand compaction, tombstones, and `delete.retention.ms`, or you will ship a state-corruption bug.

**Do not reach for it for ordinary application development.** If you are writing a producer or consumer for a topic someone else operates, you do not need to think about segments and the page cache day to day. You need delivery semantics, partitioning, and ordering — the application-level guarantees covered in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) and [message ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees). The storage layer is the operator's and the architect's concern; the application developer relies on its guarantees without re-deriving them.

**Do not over-tune.** The Kafka defaults — 1 GB segments, 7-day retention, sparse index every 4 KB, dirty ratio 0.5, page cache owns the RAM — are well chosen for the common case. The failure mode I see more often than under-tuning is *over-tuning*: operators twisting knobs they do not fully understand, setting tiny segments or aggressive fsync or weird index intervals, and creating problems the defaults would never have had. Tune deliberately, change one thing, measure, and have a hypothesis grounded in the model above. If you cannot explain *why* a non-default value helps in terms of sequential writes, the page cache, or the index, leave the default alone.

## Key takeaways

- **A partition is an append-only log on one broker's disk**, split into segments. The offset is a stable *position* into that immutable sequence, not a deletable message ID — which is the root of replay, multiple independent readers, and time travel.
- **Each segment is three files**: `.log` (records), `.index` (sparse offset to byte position), `.timeindex` (timestamp to offset). Exactly one segment per partition is the mutable active segment; all others are immutable closed segments.
- **Append-only means sequential writes**, and sequential disk I/O is 100–1000x faster than random I/O. Batching and compression multiply the win, and compression carries through disk, replication, and network because the broker stores and ships batches still compressed.
- **The page cache is the cache, not the JVM heap.** Brokers run tiny heaps and let the kernel cache terabytes of partition data in free RAM. Writes return at page-cache speed; reads near the tail hit the same RAM the producer just warmed; the cache survives a process restart.
- **Zero-copy via `sendfile` sends bytes from page cache straight to the NIC** without copying through user space — RAM-to-NIC DMA — which is why a caught-up broker barely touches the data it serves and can saturate its network link with CPU to spare.
- **An offset resolves in logarithmic time plus a sub-4-KB scan**: binary-search the segment list, binary-search the sparse index, then scan forward a little. The index is sparse *on purpose* — small enough to stay memory-mapped, with a forward scan cheap enough not to matter.
- **Retention deletes whole closed segments** by time (`retention.ms`) or size (`retention.bytes`), independent of who has read them. Lag exceeding retention is permanent, silent data loss for that consumer — the retention cliff.
- **Compaction keeps the latest record per key forever**, turning a topic into a replayable changelog. The cleaner runs on dirty ratio, never touches the active segment, and tombstones (null values) delete keys — but `delete.retention.ms` must exceed your slowest consumer's lag or deletes silently fail to propagate.
- **The capacity model is three multiplications**: disk ≈ rate × retention × RF; internal load ≈ rate × RF; read working set ≈ rate × consumer-lag-seconds. Keep the read set inside page-cache RAM and your brokers are network-bound and serene; exceed it and you fall off the eviction cliff into random disk I/O.

## Further reading

- [Kafka as a distributed log: the database turned inside out](/blog/software-development/database/kafka-as-a-distributed-log) — the conceptual companion to this post: what a log is, consumer groups, KRaft, and exactly-once at the model level.
- [Anatomy of a message system: producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) — the producer-broker-consumer shape this post zooms into.
- [Kafka Deep Dive, Part 2: consumer groups and the rebalance protocol](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — how partitions get assigned to consumers, and how rebalances work and misbehave.
- [Kafka Deep Dive, Part 3: replication, the ISR, and the high-watermark](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — how the replication tax in this post's budgets actually works, and where durability really comes from.
- [Message ordering and partitioning: the guarantees you actually get](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — why partition count is a one-way capacity-and-ordering decision.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the canonical use case for compacted topics as a durable changelog.
- *Kafka: The Definitive Guide* (Shapira, Palino, Sivaram, Petty), the storage internals and broker configuration chapters, for the authoritative reference on every knob mentioned here.
- The Apache Kafka documentation on log compaction and broker configuration — `log.dirs`, `segment.bytes`, `retention.ms`, `cleanup.policy`, `min.cleanable.dirty.ratio`, and the page-cache-related OS tuning notes.

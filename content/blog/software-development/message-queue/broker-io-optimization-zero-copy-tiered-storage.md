---
title: "Broker I/O Optimization: Zero-Copy, the Page Cache, and Tiered Storage"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A broker's entire job is moving bytes between disk and network as cheaply as possible. This is the operational deep dive: page cache, zero-copy sendfile, JBOD disk layout, the OS knobs that actually matter, and tiered storage that pushes cold segments to S3 so retention becomes effectively infinite and nearly free."
tags:
  [
    "message-queue",
    "kafka",
    "zero-copy",
    "page-cache",
    "tiered-storage",
    "object-storage",
    "performance",
    "distributed-systems",
    "event-driven",
    "storage-engine",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-1.webp"
---

A message broker, stripped of all its protocol surface and all its replication ceremony, is a machine for moving bytes. Bytes come in from a producer's socket, they go onto a disk, they come back off the disk, and they go out to a consumer's socket. That is the whole job. Everything else — partitions, consumer groups, the ISR, exactly-once semantics — is bookkeeping wrapped around that one physical loop. And the difference between a broker that costs you a rack of machines and a broker that costs you three is almost entirely about how cheaply it executes that loop: how many times each byte gets copied, whether the disk seeks or streams, whether the cache lives in a place that survives a restart, and whether you are paying premium NVMe prices to store data nobody has read in three weeks.

I have watched the same cluster, on the same hardware, go from sweating at 200 MB per second to coasting at 900 MB per second with no code change at all — just the operating system tuned correctly and the read path going through `sendfile` instead of bouncing data through the heap. I have also watched a team burn a five-figure monthly bill storing ninety days of retention on local SSD when ninety-five percent of those bytes were never read after the first hour, and then watched that bill collapse by an order of magnitude when the cold segments moved to object storage. The byte path is where the money is. This post is about making it fast and making it cheap.

This is the operational, I/O-tuning companion to the storage foundation laid out in [Kafka Deep Dive Part 1](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage). That post explains *what* a segment is, why the log is append-only, and how the page cache and zero-copy reads make the storage engine fast in the first place. I will not re-derive segments from scratch here — go read that if "active segment" and "sparse index" are not already familiar. What this post does is take that foundation into the realm where you actually operate the thing: disk layout, kernel knobs, compression at rest, and the single biggest modern lever for both cost and elasticity, tiered storage. The figure below is the spine of the whole discussion — the path a single produced byte takes from a producer's socket to a dirty page in the page cache, and only later to the physical disk.

![A vertical stack showing the broker write path from producer through the socket receive buffer into the page cache as dirty pages then asynchronously flushed to a sequential disk append](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-1.webp)

By the end you will be able to do four concrete things. You will be able to reason about a fetch or a produce at the level of CPU copies and context switches, and explain exactly where `sendfile` saves you work. You will be able to lay out disks across a broker — JBOD versus RAID, how many log directories, NVMe versus HDD — with a cost model instead of a guess. You will be able to set the handful of Linux kernel parameters that actually move broker performance, and know why each one matters. And you will be able to design a tiered-storage retention policy that keeps a few days hot on local disk and a year cheap in S3, with an honest understanding of the latency and cost tradeoffs you are accepting when a consumer reads cold data. This connects directly to two sibling posts: [Choosing a message broker](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs), because tiered storage changes the economics enough to shift that decision, and [Throughput vs latency](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff), because every knob here trades one against the other.

## 1. The broker as a byte mover

Let us start by being ruthlessly literal about what happens when a producer sends a record. The producer has serialized a batch of records into a contiguous buffer — maybe 16 KB of compressed data — and it writes that buffer to a TCP socket connected to the broker. From there, the bytes take the path in figure 1: they land in the kernel's socket receive buffer, the broker reads them out into the partition's log, and the kernel places them into the page cache as *dirty pages* — pages that have been modified in memory but not yet written to the physical disk. At this point, depending on the durability configuration, the broker can already acknowledge the produce. The bytes are in memory, they are in the log's byte stream, and the offset has been assigned. The actual write to the spinning platter or the flash cells happens later, asynchronously, when the kernel decides it is time to flush.

That last sentence is the single most important thing to understand about broker write performance, and it is worth sitting with. **The broker does not wait for the disk on the hot path.** When a produce returns, the data is in the page cache, not necessarily on durable storage. Durability comes from *replication* — the data is on the page cache of multiple brokers — and from the eventual flush, but the acknowledgment latency is decoupled from disk latency. This is why a broker with `acks=all` can return in single-digit milliseconds even though the underlying disk's fsync latency might be tens of milliseconds: the ack is gated on the data reaching enough brokers' page caches, not on any single fsync. We will return to exactly when the flush happens when we get to dirty ratios in section 6, because that timing is one of the most consequential knobs you control.

### The four costs on the byte path

Every byte the broker handles incurs some subset of four physical costs, and almost all of broker tuning is about minimizing them:

1. **Disk seeks.** On a spinning disk, repositioning the head to a new track costs roughly 5 to 10 milliseconds. Do this on every write and you are limited to a couple hundred operations per second. Avoid it — write only to the end of a file — and you stream at hundreds of megabytes per second. Section 2 is entirely about this.
2. **Memory copies.** Every time a byte is copied from one buffer to another, the CPU spends cycles and memory bandwidth doing it. A naive read path copies each byte four times. Zero-copy gets it down to two DMA transfers that the CPU does not touch at all. Sections 3 and 4 are about this.
3. **Context switches.** Every system call that crosses from user space into the kernel and back costs a context switch — saving and restoring registers, polluting caches. A read-then-write does this multiple times per buffer. `sendfile` collapses it.
4. **Money per byte stored.** This is the cost people forget because it does not show up in a latency graph. Storing a terabyte on local NVMe in a cloud costs around eighty dollars a month; the same terabyte in object storage costs around two. Multiply by your retention window and your data volume and this dominates the bill. Sections 5, 7, and 8 are about this.

The genius of a well-designed broker is that it attacks all four of these at once, and it does so not with exotic algorithms but with disciplined use of facilities the operating system already provides: sequential file access, the page cache, the `sendfile` system call, and — increasingly — object storage as a backing tier. The rest of this post walks each one.

It is worth being precise about which of these costs dominates in which regime, because that determines where you spend your tuning effort. On a write-heavy broker — high produce volume, modest read fan-out — the binding constraints are sequential write bandwidth and the flush behavior, so disk layout and dirty ratios are where you win or lose. On a read-heavy broker — many consumer groups, big fan-out, replays — the binding constraint is the read path, so zero-copy and page-cache residency dominate, and TLS becomes a first-order capacity question because it defeats zero-copy. On a retention-heavy broker — long history, modest live throughput — the binding constraint is money per byte, and tiered storage is the lever that matters more than any latency knob. Most real brokers are some blend, but knowing which corner of that triangle you live in tells you which sections of this post to read twice. A broker that is CPU-bound while the network is half-idle is almost always paying for copies it should not be making; a broker whose produce latency spikes periodically is almost always flushing in bursts; a broker whose storage bill dwarfs its compute bill is almost always storing cold data on premium disk. Each symptom maps to one of the four costs, and each cost maps to a section below.

### Why this is the broker's whole personality

It is tempting to think of broker performance as a grab-bag of optimizations, but it is more coherent than that. A broker is fast for exactly one reason: it has arranged its access patterns so that the operating system's fastest paths are the *only* paths it uses. It never seeks because it only appends. It never manages its own cache because it lets the kernel's page cache do it. It never copies data into user space on a read because it hands the file descriptor and the socket to `sendfile` and lets DMA do the work. This is a philosophy, not a feature list: **do less, and let the kernel do the rest.** Once you internalize that, every tuning decision in this post becomes obvious rather than arbitrary, because each one is just another way of getting the broker out of the kernel's way.

## 2. Sequential I/O and why append-only wins

The foundational claim, the one that makes everything else possible, is this: **a broker writes to disk on every produce and is still faster than systems that try to avoid the disk, because its writes are sequential.** The myth that "disks are slow" is half-true. Disks are slow at *random* access and fast at *sequential* access, and the gap between the two is enormous — often two orders of magnitude. Figure 2 contrasts the two access patterns directly.

![A before and after comparison contrasting random I/O with scattered writes and a seek on every operation against sequential append that writes only to the tail with no seeks and streams at hundreds of megabytes per second](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-2.webp)

Consider a 7,200-RPM spinning disk, the workhorse of cheap bulk storage. Its average rotational latency is about 4 milliseconds, and its average seek time is around 8 or 9 milliseconds. So a random read or write that has to reposition the head costs you on the order of 10 milliseconds before a single useful byte moves. That caps you at roughly 100 random operations per second. If each operation is a 4 KB page, that is 400 KB per second of random I/O — pathetic. But that same disk, reading or writing sequentially with the head staying on track and just streaming as the platter spins under it, sustains 150 to 250 megabytes per second. The ratio between sequential and random throughput on that disk is several hundred to one. The disk is not slow. *Seeking* is slow.

A broker's append-only log is the design that lets it never seek. Because a partition is written by appending to the end of the active segment and nothing else — no in-place updates, no inserts in the middle, no random rewrites — the disk head for that partition's writes stays put and streams. Each partition is one sequential write stream. Even with many partitions on one disk, the kernel's I/O scheduler batches and orders the writes to minimize head movement, and on a disk dedicated to a few high-traffic partitions the access pattern is overwhelmingly sequential. This is the reason a broker can run terabytes of throughput on hardware that a random-access database would choke on.

### SSDs do not exempt you from this

You might think solid-state drives, with no moving head, make this irrelevant. They do not, for two reasons. First, even on NVMe SSDs, sequential throughput substantially exceeds random throughput — a modern NVMe drive might do 7 GB/s sequential reads but considerably less on small random reads, because of controller overhead, the flash translation layer, and queue depth effects. Second, and more importantly, **sequential writes are gentler on flash endurance.** SSDs erase in large blocks and suffer write amplification when forced to do small scattered writes; large sequential appends align with the erase-block structure and produce far less amplification, which extends the drive's life and keeps latency predictable. So the append-only discipline pays off on every storage medium. It is not a spinning-disk-era hack; it is the right access pattern for flash too.

#### Worked example: random versus sequential on one disk

Take a single 7,200-RPM HDD with a 9 ms average seek plus 4 ms average rotational latency, so 13 ms per random operation, and a sustained sequential rate of 200 MB/s. Suppose you want to write 1 GB of data, arriving as 4 KB records.

If you wrote each 4 KB record to a random location, you would perform 262,144 operations. At 13 ms each that is 3,407 seconds — about 57 minutes — to write 1 GB. Effective throughput: roughly 300 KB/s.

If you append all 262,144 records sequentially to the end of one log, the disk just streams at 200 MB/s, so 1 GB takes about 5 seconds. Effective throughput: 200 MB/s.

The sequential path is roughly 680 times faster on the *same disk* for the *same data*. That single number is why the append-only log exists. It is not a clever trick layered on top of a fast storage engine; it *is* the fast storage engine. And it is why you should be deeply suspicious of any broker configuration or plugin that introduces random writes into the hot path — a poorly designed index that rewrites in place, a compaction strategy that scatters writes, or a filesystem mounted with the wrong options can quietly turn your beautiful sequential stream back into seeks.

## 3. The page cache: let the OS do the caching

Here is a decision that looks strange until you understand it and then looks obviously correct: **a broker should keep almost no data cached in its own process heap, and instead rely entirely on the operating system's page cache.** A JVM-based broker like Kafka runs with a deliberately small heap — often 6 to 8 GB on a machine with 64 or 128 GB of RAM — and leaves the vast majority of memory to the kernel. That leftover memory is not wasted. It is the page cache, and it is doing the actual caching of your log data. Figure 9 later in this post shows the page cache as the top layer of the OS tuning stack; here we focus on why it lives in the kernel rather than the heap.

The page cache is the kernel's cache of file contents in memory. When the broker reads a byte from a `.log` file, the kernel checks whether the relevant page is already in the page cache; if it is, the read is a memory access with no disk I/O at all. When the broker writes, the bytes go into the page cache as dirty pages and are flushed to disk later. Crucially, *the broker does not manage any of this.* It issues ordinary file reads and writes, and the kernel transparently caches, evicts, and flushes. The broker gets a multi-gigabyte (often multi-hundred-gigabyte) cache for free, managed by code that has been optimized for decades, with eviction policies the broker authors never had to write.

### Why the heap is the wrong place for this cache

A junior engineer's instinct is that the application should manage its own cache, because the application knows its access patterns best. For a broker, that instinct is wrong, and understanding why is genuinely illuminating. There are three concrete reasons the page cache beats an in-heap cache here.

First, **garbage collection.** If you cache tens of gigabytes of message data as objects on the JVM heap, the garbage collector has to scan and manage all of it. Large heaps mean long, unpredictable GC pauses — exactly the tail-latency killer you are trying to avoid in a low-latency system. By keeping the heap small and the data in the page cache, the GC has almost nothing to do, and pauses stay short. The kernel manages the gigabytes of cached data with no GC involvement at all.

Second, **double caching.** If the broker cached data in the heap *and* the kernel cached the same file pages, you would store every hot byte twice in RAM — once as a Java object, once as a page-cache page. That halves your effective cache size for no benefit. By caching only in the page cache, every byte of RAM holds exactly one copy of the data.

Third, and most beautifully, **the cache survives a process restart.** The page cache belongs to the kernel, not the broker process. If you restart the broker — a deploy, a config change, a crash — the JVM heap is wiped, but the page cache is untouched. The broker comes back up and its hot data is still cached in RAM, so it serves reads at full speed immediately instead of cold-starting and hammering the disk to refill an in-heap cache. I have seen this in production: a rolling restart of a Kafka cluster causes barely a blip in read latency, because the page cache stays warm across the restart. An in-heap cache would mean every restart triggers a thundering herd of disk reads as the cache refills. This single property — a warm cache across restarts — is worth the entire design decision on its own.

### The operational consequence: do not steal the page cache's memory

The practical lesson is a negative one: **do not configure a giant broker heap, and do not run other memory-hungry processes on the broker.** Every gigabyte you give to the JVM heap, or to some sidecar, is a gigabyte the page cache cannot use to cache your log. The right posture is a modest heap sized to the broker's actual object needs — connection state, in-flight requests, metadata — and then leave all remaining RAM to the kernel. On a 128 GB broker, an 8 GB heap leaving roughly 115 GB of usable page cache is a healthy ratio. If you find yourself wanting a 60 GB heap "to cache more data," stop: you are fighting the design. The way the page cache works is that free memory automatically becomes cache, so the best thing you can do is give it as much free memory as possible and get out of its way.

## 4. Zero-copy reads with sendfile

Now the read path, where the most dramatic single optimization in broker I/O lives. When a consumer fetches records, the broker must take bytes from a `.log` file on disk and send them out over a network socket. The naive way to do that — the way you would write it in any introductory systems course — is slow in a specific, countable way, and the `sendfile` system call eliminates most of that cost. Figure 3 lays the two paths side by side.

![A before and after comparison showing a traditional read with four copies and a context switch on each system call versus zero-copy sendfile that keeps the data in the kernel and transfers it from the page cache straight to the network card with two DMA copies](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-3.webp)

The traditional path to send a file's bytes over a socket looks like this. The application calls `read()`, which copies data from the disk into the kernel's page cache via DMA (copy one), then copies it from the page cache into a user-space buffer in the application (copy two). The application then calls `write()` on the socket, which copies the data from the user-space buffer into the kernel's socket buffer (copy three), and finally the network card's DMA engine copies it from the socket buffer onto the wire (copy four). Four copies. And there are at least two context switches: user-to-kernel on `read()`, kernel-to-user on its return, user-to-kernel on `write()`, kernel-to-user on its return. The two copies in the middle — page cache to user buffer, user buffer to socket buffer — are pure waste. The data never needed to be in user space at all; the broker is not transforming it, just shoveling it from a file to a socket.

### What sendfile actually does

The `sendfile` system call (and its modern cousins using `splice` and pipe buffers) tells the kernel: take bytes from this file descriptor and send them to that socket descriptor, and never bring them into user space. The kernel copies the data from disk into the page cache via DMA (the one unavoidable copy to get it off the disk), and then, on hardware that supports it, the network card's DMA engine reads directly from the page cache and puts the bytes on the wire — the page cache contents are handed to the NIC by reference, not by another CPU copy. The CPU never touches the actual message bytes. There is one system call instead of two, so half the context switches. And critically, **the CPU does zero copies of the payload** — the two DMA transfers are done by hardware engines, not the CPU. This is the "zero-copy" in the name: zero CPU copies of the data.

The payoff is enormous for a broker because the broker's read workload is *almost entirely* this shape: read a chunk of a log file, send it to a consumer socket, unchanged. The broker does not deserialize the records, does not transform them, does not even look at them. It is a pure conduit. That is the ideal case for `sendfile`, and it is why a broker can serve many gigabytes per second of consumer reads while keeping CPU utilization low enough that the CPU is rarely the bottleneck — the bottleneck is the network or the disk, which is exactly where you want it.

#### Worked example: a 1 MB fetch, four copies versus zero-copy

Suppose a consumer fetch pulls 1 MB of records from a hot segment already in the page cache. Let us count the work both ways.

**Traditional read-then-write.** Four copies of 1 MB each. The two DMA copies (disk-to-cache, socket-to-NIC) are done by hardware. But the two middle copies — page cache to user buffer, and user buffer to socket buffer — are done by the CPU, touching 1 MB each, so 2 MB of CPU-driven memory copying. Plus four context switches across the two system calls. At a memory bandwidth of, say, 10 GB/s for a memcpy, 2 MB takes about 200 microseconds of pure CPU copy time, and the context switches add a few microseconds each plus cache pollution. Now multiply: a broker serving 5 GB/s of consumer reads this way would be doing 10 GB/s of CPU memory copies just for the redundant middle hops — saturating memory bandwidth that should be doing useful work.

**Zero-copy sendfile.** One system call. The CPU does *zero* copies of the 1 MB payload — both transfers (disk-to-cache and cache-to-NIC) are DMA. CPU copy time for the payload: zero. Context switches: two instead of four. That same broker serving 5 GB/s of reads now spends essentially no CPU on data movement, freeing the entire memory-bandwidth budget and most of the CPU for compression, request handling, and replication.

The headline numbers: per megabyte fetched, `sendfile` saves about 2 MB of CPU memory copying and halves the context switches. Across a busy broker that is the difference between the CPU being the bottleneck and the CPU being nearly idle while the network saturates. This is also why **TLS encryption disables zero-copy**: once the broker must encrypt the bytes, it has to bring them into user space to run them through the cipher, and you are back to copying. That is a real and often-overlooked cost of enabling in-flight encryption on a high-throughput broker — you lose `sendfile` and the CPU cost of serving reads jumps substantially. It is a tradeoff worth measuring before you turn TLS on for internal, already-private traffic.

## 5. Disk layout: JBOD, RAID, and multiple log dirs

So far the disk has been a single abstract thing. In practice a broker has several disks, and how you arrange them is one of the higher-leverage and most permanent decisions you make, because you cannot easily re-lay-out disks on a running fleet. There are two broad strategies: pool the disks behind a RAID controller and present one big volume, or expose each disk individually and let the broker spread partitions across them. The latter is called JBOD — "just a bunch of disks" — and for a modern broker it is usually the right answer.

### RAID versus JBOD

With **RAID** (specifically RAID 10 or RAID 5/6), several physical disks are combined by a controller into one logical volume with redundancy. The broker sees one big filesystem and writes all its log directories there. The appeal is operational simplicity and that the RAID layer handles a single disk failure transparently — the broker never even notices. The cost is real, though. RAID 10 halves your usable capacity (every byte is mirrored), which is expensive at scale. RAID 5/6 keeps more capacity but imposes a write penalty: every write requires reading and recomputing parity, which adds latency and, worse, turns some of your beautiful sequential writes into read-modify-write cycles. And the RAID controller's cache and rebuild behavior can introduce latency spikes that are hard to diagnose.

With **JBOD**, each physical disk is its own filesystem, and the broker is told about all of them via a list of log directories. The broker spreads its partitions across the disks itself, putting different partitions' segments on different disks. There is no mirroring at the disk layer, so you get the full raw capacity of every disk. There is no parity write penalty, so every write stays sequential. The tradeoff is that the broker, not the RAID controller, must handle a disk failure — and modern brokers do. When a disk dies in a JBOD setup, the broker takes the partitions on that disk offline, and replication (the copies on *other* brokers) keeps those partitions available. You have effectively moved redundancy up a level: instead of mirroring within a machine via RAID, you replicate across machines via the broker's own replication. Since you are already paying for cross-machine replication for availability, paying *again* for within-machine RAID mirroring is often redundant.

The consensus for high-throughput brokers is JBOD with replication handling failures, because it gives you full capacity, no parity write penalty, and redundancy you are already paying for at the right layer. RAID makes sense when operational simplicity matters more than capacity efficiency, or when your replication factor is low and you want a within-box safety net.

### Multiple log directories

Whether JBOD or RAID, you configure the broker with one or more **log directories** — the `log.dirs` setting in Kafka, a comma-separated list of paths. With JBOD you point each path at a different physical disk:

```properties
# Kafka broker: one log dir per physical disk (JBOD)
log.dirs=/data/disk1/kafka,/data/disk2/kafka,/data/disk3/kafka,/data/disk4/kafka

# The broker round-robins new partitions across these directories,
# balancing by partition count (and, in newer versions, by disk usage).
```

The broker places each new partition replica into one of these directories, balancing the count so that, on average, each disk holds a similar number of partitions and therefore a similar share of the I/O. This is how you scale a single broker's disk throughput: four disks each doing 200 MB/s sequential gives the broker an aggregate 800 MB/s of disk bandwidth, far more than any single disk. The kernel and the broker keep each disk's access pattern sequential within its own partitions, so you get near-linear scaling of disk throughput with disk count, which a single RAID volume's controller can sometimes bottleneck.

### NVMe versus HDD economics

The last disk-layout question is what kind of disk. The two ends of the spectrum are NVMe SSDs (fast, expensive, low capacity per dollar) and spinning HDDs (slow per operation, cheap, huge capacity per dollar). The right choice depends entirely on the access pattern, and a broker's access pattern is wonderfully favorable to cheap disks: writes are sequential, and reads of *hot* data come from the page cache in RAM, never touching the disk at all. So for a broker whose consumers mostly read the recent tail (the common case), even HDDs deliver excellent performance, because the disk only ever sees sequential writes and the occasional cold-read replay — and HDDs are great at sequential.

This is the dirty secret that makes brokers cheap to operate: **you can run enormous retention on slow, cheap spinning disks**, because the workload that hits the disk is sequential writes plus rare cold reads, and HDDs handle exactly that well. You pay NVMe prices only when you genuinely need low-latency cold reads — consumers that frequently replay old data and cannot tolerate HDD seek latency on a cache miss. For most workloads, HDD for bulk capacity with plenty of RAM for the page cache is the cost-optimal layout. But the modern answer, which collapses this whole question, is to not store the cold data on local disk at all — and that is tiered storage, which we get to in section 7. Before that, the kernel knobs.

There is one more disk-layout subtlety worth naming: the **filesystem and mount options** sitting between the broker and the block device. Most brokers run on `ext4` or `xfs`, and `xfs` is often preferred for very high partition counts because it handles large directories and parallel writes well. The mount option that matters most is `noatime` (or `relatime`), which disables updating each file's access-time metadata on every read. Without it, every consumer read triggers a tiny metadata *write* to update the access time — silently injecting random writes into a workload you worked hard to keep sequential, and inflating I/O for no benefit anyone uses. Mounting the log directories `noatime` removes that hidden write amplification entirely. It is a one-line change in `/etc/fstab` that costs nothing and quietly improves both read and write efficiency, and it is the kind of detail that separates a broker that performs to spec from one that mysteriously underperforms its hardware. Combined with the right filesystem and JBOD across enough spindles or flash devices, the disk layer disappears as a bottleneck and the broker streams at the aggregate sequential bandwidth of all its disks.

## 6. OS tuning: dirty ratios, file handles, socket buffers

A broker lives or dies by a small number of operating-system settings, and most performance problems I have debugged in the field came down to one of them being at a bad default. Figure 9, the OS tuning stack, names the four layers; this section is the detailed walk through each knob. None of these are exotic — they are standard Linux `sysctl` and `ulimit` settings — but the defaults shipped by most distributions are tuned for a laptop or a general-purpose server, not for a machine pushing gigabytes per second through the page cache.

### Dirty ratios: controlling when the flush happens

Recall from section 1 that writes go into the page cache as dirty pages and get flushed to disk later. *When* "later" is determines a lot about your latency profile, and two kernel parameters control it:

- **`vm.dirty_background_ratio`** — the percentage of total memory that can be filled with dirty pages before the kernel starts flushing them to disk *in the background*, without blocking the writing process. Default is often 10.
- **`vm.dirty_ratio`** — the percentage of total memory that can be filled with dirty pages before the kernel forces the writing process to *block* and flush synchronously. Default is often 20.

The danger with the defaults on a big-memory machine is that the percentages translate to enormous absolute amounts of dirty data. On a 128 GB machine, a `dirty_ratio` of 20 means up to roughly 25 GB of dirty pages can accumulate before a forced flush. When that flush finally fires, the kernel dumps 25 GB to disk in one violent burst, and *every* process writing during that burst stalls. The symptom is periodic, brutal latency spikes — produce latency p99 that looks fine for thirty seconds then spikes to seconds — that correlate with the flush. The fix is to lower these so the flushing is smooth and continuous rather than bursty:

```bash
# /etc/sysctl.d/99-broker.conf
# Flush in the background early and often; force-block much sooner.
# On large-memory machines, prefer the byte-valued variants for precision.
vm.dirty_background_bytes = 536870912   # start background flush at 512 MB dirty
vm.dirty_bytes            = 1073741824   # force synchronous flush at 1 GB dirty

# Apply with: sysctl -p /etc/sysctl.d/99-broker.conf
```

Using the `_bytes` variants instead of the `_ratio` percentages gives you precise, memory-size-independent control, which matters because a 10% ratio means something very different on a 16 GB box than on a 256 GB box. Setting a modest absolute cap means the kernel keeps a steady trickle of writes going to disk instead of saving them up for a catastrophic flush. You trade a slightly higher steady write rate to the disk for the elimination of latency spikes — almost always the right trade for a latency-sensitive broker. This is exactly the kind of throughput-versus-latency knob explored in depth in [Throughput vs latency](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff); the dirty ratio is one of the cleanest examples of the tradeoff at the kernel level.

### File handles: a broker opens thousands of files

A broker holds a lot of files open simultaneously. Every segment of every partition is potentially three open file descriptors (`.log`, `.index`, `.timeindex`), and every client connection is a socket, which is also a file descriptor. A broker with thousands of partitions and thousands of clients can easily need hundreds of thousands of open file descriptors. The default `ulimit` for open files on most systems is 1,024 — absurdly low for this workload. When a broker exhausts its file-descriptor limit, it cannot open new segments or accept new connections, and it fails in confusing, cascading ways: "Too many open files" errors, partitions going offline, clients unable to connect. Raise the limit aggressively:

```bash
# /etc/security/limits.d/broker.conf
kafka  soft  nofile  500000
kafka  hard  nofile  500000

# And for systemd-managed services, also set in the unit:
#   [Service]
#   LimitNOFILE=500000
```

Half a million open files sounds extravagant, but it costs almost nothing and removes an entire category of mysterious outages. The same applies to **memory-map limits** (`vm.max_map_count`): the broker memory-maps every index file, and a broker with hundreds of thousands of segments can exhaust the default map count (often 65,530), so raise it to several hundred thousand or more. These limits are not performance tuning in the throughput sense — they are correctness floors below which the broker simply breaks under load. Get them right once and forget them.

### Socket buffers: filling the network pipe

The last family of knobs is the TCP socket buffers, which govern how much data can be in flight on a connection before the sender must wait for an acknowledgment. This matters enormously for throughput over high-latency links — replication between data centers, or consumers reading across a region. The maximum throughput of a single TCP connection is bounded by the *bandwidth-delay product*: the window size divided by the round-trip time. If the socket buffer is too small, a single connection cannot keep a fat, long pipe full, and you leave bandwidth on the table no matter how fast your disks are.

```bash
# /etc/sysctl.d/99-broker.conf (continued)
# Allow large TCP windows so a single connection can fill a high-BDP link.
net.core.rmem_max = 16777216      # 16 MB max receive buffer
net.core.wmem_max = 16777216      # 16 MB max send buffer
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 250000   # deeper NIC queue for bursts
```

You also set the broker's own socket buffer sizes in its configuration (`socket.send.buffer.bytes` and `socket.receive.buffer.bytes` in Kafka), and for cross-datacenter replication you bump them up substantially. The rule of thumb: the buffer should be at least the bandwidth-delay product of the link. For a 1 Gbps link with a 50 ms round trip, that is about 6 MB; the default 64 KB or 128 KB would cap a single connection at a small fraction of the available bandwidth. Getting socket buffers right is the difference between cross-region replication that keeps up and replication that perpetually lags.

### Compression at rest

One more lever that lives partly at the I/O layer: compression. Producers can compress record batches (with `lz4`, `zstd`, `snappy`, or `gzip`), and the broker stores them compressed and serves them compressed — the consumer decompresses. This is doubly good for I/O. It reduces the bytes written to and read from disk, multiplying your effective disk throughput and capacity, and it reduces the bytes on the wire for both replication and consumer reads. Because the broker stores the already-compressed batches as-is and serves them via `sendfile` without decompressing, compression and zero-copy compose perfectly: the broker moves compressed bytes from disk to network without ever touching them. `zstd` typically gives the best ratio-to-CPU tradeoff for log data; `lz4` is the choice when you want the lowest CPU cost. A 3:1 compression ratio triples your effective retention and throughput for the cost of some producer and consumer CPU — almost always worth it.

## 7. Tiered storage: offloading cold segments to object storage

Now the big modern lever, the one that changes the economics of a broker more than any other single feature in the last decade. Everything so far has assumed your data lives on disks attached to your brokers. **Tiered storage breaks that assumption: it moves cold, closed segments off local disk and into object storage like S3 or GCS, leaving only the hot tail on the broker's fast local disk.** The local disk holds days; the object store holds months or years. Retention becomes effectively infinite and nearly free, and — this is the part that reshapes operations — storage decouples from compute. Figure 4 shows the layout: a hot local tier and a cold remote tier, with the broker fronting both behind one continuous offset space.

![A grid showing tiered storage with a producer writing to a hot local SSD tier holding seven days a broker fronting both tiers a remote segment manager indexing offsets and a cold object storage tier holding a full year that a consumer can fetch from by offset](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-4.webp)

The mechanism is clean and builds directly on the segment model from [Kafka Deep Dive Part 1](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage). Recall that a closed segment is immutable — once it rolls, it never changes. That immutability is exactly what makes it safe to copy elsewhere. With tiered storage enabled, a background process watches for segments that have been closed and have aged past a *local* retention threshold (say, the local copy is older than a few days). It uploads those segments — the `.log` plus its indexes — to object storage, records their location in a remote segment metadata store, and then deletes the local copies, reclaiming local disk. The data is not gone; it is in S3, indexed by offset, and the broker knows exactly where to find it. From a consumer's perspective the partition is unchanged: it is still one continuous offset space from offset zero to the latest. The consumer cannot tell, from the offsets, which records live locally and which live in S3. Figure 5 makes the tier comparison concrete.

![A matrix comparing local NVMe local HDD and object storage across cost per terabyte-month read latency and retention fit showing object storage costs far less but adds tens of milliseconds of read latency](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-5.webp)

### Two retention windows instead of one

Tiered storage splits the single retention knob into two. There is a **local retention** — how long a segment stays on the broker's fast disk before being offloaded — and a **total retention** — how long it stays in object storage before being deleted for good. You might set local retention to 3 days and total retention to 365 days. The hot tail that most consumers read sits on local NVMe and is served at full speed via the page cache and `sendfile`. The long cold tail — needed for occasional replays, backfills, compliance audits, training-data regeneration — sits in S3 at a fraction of the cost. The two windows let you pay premium prices only for the data that needs premium latency, and bulk-storage prices for everything else.

### Storage decouples from compute

The economic argument is huge, but the operational argument is just as important. In a classic broker, your data lives on the brokers, so the amount of data you retain dictates how many brokers you need and how big their disks are. Want a year of retention? You need enough local disk for a year of data, spread across enough brokers, and **adding a broker means copying terabytes of data to it during the rebalance** — a slow, I/O-heavy, risky operation. Your storage capacity and your compute capacity are welded together. You cannot scale one without the other.

Tiered storage cuts that weld. Since the bulk of the data lives in object storage — which is independently scalable and shared, not owned by any one broker — the brokers themselves hold only a few days of hot data. Now adding a broker means copying only a few days of data, not a year. Scaling out is fast because there is little local data to drag around. You can size your broker count for *throughput* (CPU, network, hot-disk bandwidth) and size your object storage for *retention* (total data volume), independently. This is the same decoupling that makes object-store-native systems attractive, and it is why tiered storage is the headline feature of modern broker releases. It turns "how much can I afford to retain?" from a hardware question into a billing-line question.

The elasticity payoff compounds in failure scenarios, not just planned scaling. When a broker dies in an all-local cluster, the partitions it led must be recovered from their replicas, and bringing a *replacement* broker into the cluster means re-replicating that broker's entire local dataset — potentially terabytes — before the cluster is back to full redundancy. The longer your retention, the longer that re-replication takes, and the longer you run degraded. With tiered storage, a replacement broker only needs to pull the small hot tail locally; the cold data it is responsible for already lives in shared object storage and does not need to be physically copied onto the new broker at all. Recovery time after a broker loss shrinks from hours to minutes, and it stops scaling with your retention window. This is a durability and availability argument, not just a cost argument: a cluster that recovers fast is a cluster that spends less time one failure away from data loss. The same property makes maintenance — kernel upgrades, instance-type migrations, rebalancing for skew — dramatically cheaper, because moving a broker's responsibilities around no longer means dragging its history with it.

#### Worked example: 30 days local NVMe versus 7 days local plus 1 year S3

Take a topic ingesting 100 MB/s of compressed data, sustained, all day. That is 100 MB/s times 86,400 seconds, about 8.64 TB per day. Assume replication factor 3, so every byte is stored three times on local disk.

**All-local, 30 days retention on NVMe.** Thirty days of data is 30 times 8.64, about 259 TB of unique data, times 3 replicas, about 778 TB of local NVMe. At a representative cloud price of \$80 per TB-month for NVMe-class block storage, that is 778 times \$80, about \$62,200 per month, just for storage. And every broker you add to handle throughput drags its share of that 778 TB during rebalances.

**Tiered, 7 days local NVMe plus 358 days in S3.** Seven days local is 7 times 8.64, about 60.5 TB unique, times 3 replicas, about 182 TB of NVMe at \$80, which is about \$14,500 per month. The remaining 358 days, 358 times 8.64, about 3,093 TB unique — and here is a second win: you store it in object storage at *one* replica's worth of cost because the object store handles its own redundancy internally, so you pay for roughly 3,093 TB, not three times that. At a representative \$23 per TB-month for standard object storage in some clouds (and far less for infrequent-access tiers), that is 3,093 times \$23, about \$71,100 per month. Hmm — that does not obviously win at full standard-tier pricing for a *year* of data. But move the cold tier to an infrequent-access class at around \$10 per TB-month and it drops to about \$30,900, and the total becomes \$14,500 plus \$30,900, about \$45,400 — already cheaper than \$62,200 for only 30 days, while retaining a *full year* instead of a month.

The real lesson of the arithmetic is subtler than "S3 is cheaper": it is that **tiered storage lets you buy far more retention for the same or less money, and it removes replication multiplication from the cold tier.** You stopped paying 3x for 358 days of data, and you stopped paying NVMe prices for data nobody reads. The exact crossover depends on your cloud's price sheet and which object-storage class you pick, so always run your own numbers — but the structural advantage (no replication multiplier on cold data, bulk-storage pricing, decoupled scaling) holds everywhere. For a year of retention, tiered storage is almost always dramatically cheaper than all-local, and the gap widens the longer you retain.

### The honest downsides

Tiered storage is not free of tradeoffs, and pretending otherwise sets teams up for surprises. First, **reads of cold data are slow** — a fetch that misses local disk and has to pull a segment from S3 adds tens of milliseconds of latency, sometimes more for a large segment, versus sub-millisecond for hot data. We trace exactly why in the next section. Second, **object storage has its own cost model with per-request charges**: every GET and PUT costs a fraction of a cent, and a workload that does many small remote reads can run up surprising request bills on top of the storage bills. Tiered storage implementations mitigate this by reading large chunks and caching them locally, but a badly behaved replay (many consumers seeking randomly into cold history) can be expensive. Third, **it adds operational complexity**: a remote segment metadata store to keep consistent, a background uploader to monitor, and a new failure mode where object storage being unavailable affects cold reads. None of these are dealbreakers, but they are real, and the right mental model is that tiered storage trades a little read latency and some operational surface for an enormous win in cost and elasticity.

## 8. The fetch path for a remote segment

When a consumer fetches data that is still on the broker's local disk — the common case, the hot tail — the path is the one we have already optimized: a binary search in the sparse index, a seek to a byte position, and a zero-copy `sendfile` from the page cache to the consumer socket. Sub-millisecond, CPU nearly idle. But when a consumer fetches an offset that has been tiered to object storage, a different and more expensive path runs. Figure 6 traces it.

![A pipeline showing a consumer fetch for an old offset that misses local disk because the segment was tiered then issues a GET to object storage fills a local read cache at about fifty milliseconds first byte and finally serves the records to the consumer](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-6.webp)

Here is the sequence. The consumer issues a fetch for some old offset. The broker looks up which segment contains that offset and discovers — from the remote segment metadata — that the segment is no longer local; it lives in object storage. This is the **local miss**. The broker (or a dedicated remote-storage-manager component) issues a `GET` request to the object store for the relevant portion of the segment. Object storage supports ranged reads, so the broker does not have to download the entire segment to serve a small fetch — it can request just the byte range covering the offsets the consumer wants, guided by the segment's index (which is also tiered and can be fetched first). The object store responds, the broker fills a **local read cache** with the fetched chunk, and then serves the records to the consumer. Subsequent fetches of nearby offsets hit that local cache and are fast again; the slow part is the first miss.

### Why the first byte is slow

The latency of that first remote read is dominated by the object store's time-to-first-byte, which is typically in the tens of milliseconds — call it 20 to 100 ms depending on the cloud, the region, and the object size. That is two to three orders of magnitude slower than a page-cache hit. For a consumer doing a bulk replay — reading a long contiguous stretch of cold history — this is fine, because the cost amortizes: one slow `GET` brings in a large chunk, and then the consumer streams through it at high speed from the local read cache. Bulk sequential replay from tiered storage can sustain excellent throughput precisely because the per-request latency is amortized over a large transfer. The pathological case is a consumer doing many small, scattered reads into cold data — each one a fresh remote miss with full latency. That pattern both runs up latency and racks up per-request charges, and it is the workload tiered storage is worst at.

### The design implication

This shapes how you should think about consumers in a tiered world. Consumers reading the hot tail — the overwhelming majority in most systems — never touch object storage and pay nothing extra. Consumers doing planned bulk backfills (reprocess the last six months to rebuild a derived dataset) read cold data sequentially and amortize the remote latency well. The combination to avoid is a latency-sensitive consumer that frequently and randomly accesses cold data — for that, you either keep more retention local (push the local retention window out) or rethink whether that data should be in a different store altogether. The broker gives you the knob (local retention) to tune exactly how much of the offset space is fast; tiered storage is not all-or-nothing, it is a slider between cost and read latency. Figure 7 places this whole optimization family in one taxonomy so you can see where tiering sits relative to the other levers.

![A tree taxonomy of broker I/O optimizations branching into sequential layout with append-only and JBOD page cache with warm restarts zero-copy via sendfile and tiering cold data to object storage](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-7.webp)

The taxonomy is worth pausing on because it organizes the entire post. Sequential layout (append-only writes, JBOD across many disks) removes seeks. The page cache (OS-managed, warm across restarts) removes redundant caching and cold-start penalties. Zero-copy via `sendfile` removes CPU copies on the read path. And tiering removes the cost and elasticity penalty of storing cold data on premium local disk. Each branch attacks a different one of the four costs from section 1 — seeks, copies, cache management, and money per byte. A well-tuned broker uses all four together. And the lifecycle of a single segment, shown in figure 8, threads through several of these: it is born hot and active, rolls closed, gets tiered to cheap remote storage, serves the occasional remote replay, and finally expires.

![A timeline of a segment lifecycle moving from active and appending to rolled closed at the one gigabyte cap to tiered into S3 with the local copy deleted to served from remote storage on replay to finally expired and the object deleted after a year](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-8.webp)

That lifecycle is the unifying picture. A segment spends a short, intense youth as the active segment, absorbing sequential appends into the page cache. It then spends a brief middle age closed but local, served hot via zero-copy to consumers reading the recent tail. Then, when it ages past local retention, it migrates to object storage and spends the vast majority of its existence there — cheap, durable, occasionally read for a replay or an audit, costing a fraction of what local NVMe would. Finally, at total retention, the object is deleted and the segment ceases to exist. Understanding that a segment's *time-weighted* life is mostly spent cold in object storage is the whole justification for tiered storage: you are paying premium prices only for the brief window when the data is actually hot.

## 9. A broker I/O tuning checklist

Let us consolidate everything into an operational checklist, the thing you would actually run down when commissioning a new broker fleet or diagnosing a slow one. Figure 9 shows the OS tuning layers as a stack; this section is the prose checklist that goes with it.

![A vertical stack of OS tuning layers from the page cache that should keep free RAM down through the dirty ratio flush threshold the file handle limit the socket buffer sizes and finally the NIC and disk hardware](/imgs/blogs/broker-io-optimization-zero-copy-tiered-storage-9.webp)

Work from the top of the stack down. At the page-cache layer, confirm the broker heap is small (single-digit gigabytes for a JVM broker) and the rest of RAM is free for the page cache; never run memory-hungry neighbors on a broker. At the dirty-ratio layer, set `vm.dirty_background_bytes` and `vm.dirty_bytes` to modest absolute values so flushing is a steady trickle rather than a periodic flood — this is the single most common fix for mysterious produce-latency spikes. At the file-handle layer, raise `nofile` to several hundred thousand and `vm.max_map_count` well above the segment count, or the broker will fail under load in confusing ways. At the socket-buffer layer, size `rmem`/`wmem` to at least the bandwidth-delay product of your fattest, longest link, especially for cross-region replication. And below all of that, make sure the hardware matches the workload: enough disks (JBOD) for aggregate sequential bandwidth, and enough network for your peak read fan-out.

### The verification step

Tuning without measurement is superstition, so verify each layer. To confirm the page cache is doing its job, watch the broker's disk read rate under a steady consumer load: if consumers are reading the recent tail and you see near-zero disk reads, the page cache is serving them — good. If you see heavy disk reads for tail consumers, something is evicting the cache (too-large heap, a noisy neighbor, or a cold-read consumer thrashing it). To confirm dirty-ratio tuning, watch produce-latency p99 for periodic spikes correlated with flush activity (visible in `/proc/vmstat` and `iostat`); smooth latency means the flush is well-behaved. To confirm zero-copy is active, watch broker CPU under heavy consumer reads: if CPU stays low while network saturates, `sendfile` is working; if CPU climbs with read volume, check whether TLS is forcing data through user space. To confirm socket buffers, check whether single-connection replication throughput matches the link's bandwidth-delay product; if a single replication stream caps well below the link capacity, the buffers are too small.

### A representative production config

Here is a consolidated set of broker and OS settings that I would consider a sane starting point for a high-throughput broker, to be measured and adjusted against your actual workload:

```properties
# --- Broker (Kafka-style) ---
log.dirs=/data/d1/kafka,/data/d2/kafka,/data/d3/kafka,/data/d4/kafka  # JBOD
num.io.threads=16              # match to disk count and load
num.network.threads=8
socket.send.buffer.bytes=1048576       # 1 MB; raise for cross-region
socket.receive.buffer.bytes=1048576
compression.type=producer       # store what producers send (often zstd)

# Tiered storage (where supported)
remote.log.storage.system.enable=true
log.local.retention.ms=259200000        # 3 days hot on local disk
log.retention.ms=31536000000            # 365 days total (mostly in S3)
```

```bash
# --- OS (/etc/sysctl.d/99-broker.conf) ---
vm.dirty_background_bytes = 536870912    # 512 MB
vm.dirty_bytes            = 1073741824    # 1 GB
vm.max_map_count          = 1048576
vm.swappiness             = 1            # avoid swapping out the page cache
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.netdev_max_backlog = 250000

# --- limits (/etc/security/limits.d/broker.conf) ---
# kafka soft nofile 500000
# kafka hard nofile 500000
```

Two of these deserve a final word. `vm.swappiness=1` is important and often missed: if the kernel swaps out page-cache or broker memory to disk, you get catastrophic latency, because now your "memory" cache is on disk. Setting swappiness near zero tells the kernel to essentially never swap, keeping the page cache and broker resident in RAM. And `compression.type=producer` means the broker stores exactly the compressed batches the producer sent, without recompressing — which preserves the zero-copy property end to end, because the broker never has to decompress and recompress on the path. Get these right and the broker does what it is supposed to do: move bytes between disk and network at the speed of the hardware, with the CPU mostly watching.

## Case studies and war stories

### The 25 GB flush stall

A team running a large-memory Kafka fleet — 256 GB per broker — kept hitting periodic produce-latency spikes: p99 would sit at 4 ms for nearly a minute, then jump to 1.5 *seconds* for a few seconds, then recover, on a roughly minute-long cycle. Replication looked healthy, network was fine, disks were not saturated on average. The culprit was the default `vm.dirty_ratio` of 20 percent on a 256 GB machine: up to ~51 GB of dirty pages could accumulate before the kernel forced a synchronous flush, and when it did, it dumped tens of gigabytes to disk in one burst that stalled every writer. The fix was a one-line change — set `vm.dirty_bytes` to 1 GB so the kernel flushes in a continuous trickle — and the spikes vanished entirely. The lesson: percentage-based dirty ratios are a trap on big-memory machines, and the symptom (periodic latency spikes uncorrelated with load) is a fingerprint for it. Convert to absolute byte limits.

### The TLS that ate the CPU

Another team enabled TLS on inter-broker and consumer traffic for a compliance requirement and watched broker CPU utilization roughly double overnight, with read-serving capacity dropping noticeably. The cause was exactly the `sendfile` tradeoff from section 4: TLS forces the broker to bring every byte it serves into user space to encrypt it, which defeats zero-copy and reintroduces the CPU copies that `sendfile` had eliminated. The bytes that used to flow disk-to-cache-to-NIC by DMA now had to be copied into user space, encrypted, and copied back. The team had not anticipated that turning on encryption would cost them their zero-copy read path. The resolution was nuanced: they kept TLS on the external client traffic (where it was genuinely required) but used a private, already-isolated network for inter-broker replication without TLS, recovering zero-copy on the highest-volume internal path. The lesson: encryption and zero-copy are mutually exclusive on the read path, and on a high-throughput broker that is a first-order capacity-planning fact, not a footnote.

### The retention bill that tiered storage fixed

A company retained 90 days of event history on local NVMe across a large Kafka cluster for a regulatory replay requirement, with replication factor 3. The storage bill was the dominant line item in the cluster cost — far larger than the compute — and 95 percent of those bytes had not been read since the first hour after they were produced. When tiered storage became available, they moved to 5 days local plus 90 days in object storage (infrequent-access class). The local NVMe footprint dropped by roughly 94 percent, the cold data stopped being multiplied by the replication factor (the object store handled its own redundancy), and the total storage bill fell by more than 80 percent — while *increasing* retention headroom, since extending the cold window to a year now cost almost nothing. The one adjustment: a nightly compliance-replay job that read old data was rewritten to read cold history *sequentially* in large chunks rather than seeking, so it amortized the remote read latency and avoided a per-request bill blow-up. The lesson: most retained data is cold, paying premium prices and replication multipliers to store cold data is pure waste, and tiered storage is the lever that fixes both at once — provided your cold-read access patterns are sequential.

### The page cache that did not survive the neighbor

A team co-located a metrics-collection agent on their broker hosts that, under load, would balloon to tens of gigabytes of resident memory. On a quiet day nobody noticed. During an incident, when both the broker and the metrics agent were under heavy load, the agent's memory growth squeezed the page cache, evicting hot log pages. Suddenly, tail consumers that had always been served from RAM started hitting the disk, disk read I/O spiked, and consumer fetch latency climbed at the worst possible moment — turning a minor incident into a major one. The lesson is the negative rule from section 3 stated as a war story: **the page cache is load-bearing, and anything that steals its memory will hurt you exactly when you can least afford it.** Brokers should run alone, or with strictly memory-capped neighbors, so that free RAM stays free for the cache.

## When to reach for this (and when not to)

Every technique in this post is worth applying, but they are not all equally urgent, and a few have real tradeoffs you should weigh deliberately.

The OS tuning — dirty ratios, file handles, socket buffers, swappiness — is **non-negotiable for any production broker.** These are not optimizations; they are correctness and stability floors. The defaults will hurt you. Set them once, correctly, in your provisioning automation, and you remove an entire class of outages. There is no "when not to" here; do it always.

Sequential I/O and the page-cache discipline are **inherent to the broker's design** rather than things you turn on. Your job is to not *defeat* them: keep the heap small, do not introduce random-write workloads, do not steal cache memory. The "when not to" is nonexistent — fighting these is always wrong.

Zero-copy via `sendfile` is **automatic and pure win, except when encryption forces it off.** The decision you actually make is about TLS: enable it where you must for compliance or untrusted networks, but recognize that on a high-throughput read path it costs you zero-copy and roughly doubles read-serving CPU. For internal, already-private replication traffic, consider whether TLS is buying you enough to justify losing zero-copy. That is a real, situation-dependent tradeoff.

JBOD versus RAID is a **commit-once decision** that should default to JBOD with cross-broker replication handling failures — you get full capacity and no parity penalty, and you are already paying for replication. Reach for RAID only when operational simplicity outweighs capacity efficiency, your replication factor is low, or a within-box safety net genuinely matters for your risk posture.

Tiered storage is the one with the most nuanced "when not to." **Reach for it when** your retention is long, most of your data is cold, your storage bill is a meaningful fraction of cluster cost, and your cold-read patterns are sequential (bulk replays, backfills, audits). **Be cautious when** you have latency-sensitive consumers that frequently and randomly read old data — for them the remote-read latency and per-request costs can bite, and you may be better off keeping more retention local or using a different store for that access pattern. Tiered storage is a slider, not a switch: tune local retention to keep the right amount of the offset space fast, and let the cold tail be cheap. For the common shape — hot tail read by everyone, cold history read rarely and in bulk — it is one of the highest-leverage features you can enable, and it changes the broker-selection calculus enough to revisit [Choosing a message broker](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs) if cost-at-scale is a deciding factor.

## Key takeaways

- **A broker is a byte mover.** Its performance is determined by four costs: disk seeks, memory copies, context switches, and money per byte stored. Every optimization targets one of these.
- **Append-only writes are sequential, and sequential I/O is hundreds of times faster than random** on the same disk — the worked example showed roughly 680x. Never let random writes creep into the hot path.
- **Let the OS page cache do the caching, not the heap.** A small heap plus a large page cache means short GC pauses, no double-caching, and — the killer feature — a warm cache that survives broker restarts.
- **Zero-copy `sendfile` eliminates the two user-space copies on a read**, saving about 2 MB of CPU memory copying per megabyte fetched and halving context switches. TLS defeats it; budget for that.
- **Prefer JBOD with cross-broker replication over RAID** for full capacity, no parity write penalty, and redundancy at the layer you already pay for. Spread partitions across multiple log directories to aggregate disk bandwidth.
- **Tune the kernel or the defaults will hurt you:** absolute `dirty_bytes` (not percentage ratios) to avoid flush stalls, huge `nofile` and `max_map_count` limits, BDP-sized socket buffers, and `swappiness=1` so the page cache never gets swapped to disk.
- **Compression at rest composes with zero-copy:** store what the producer compressed, serve it without recompressing, and triple your effective disk and network throughput.
- **Tiered storage decouples storage from compute.** Keep days hot on local disk, push months or years to object storage, get effectively infinite cheap retention, and scale brokers without dragging terabytes of cold data.
- **Cold reads from object storage cost tens of milliseconds and per-request fees.** They are excellent for bulk sequential replay and bad for frequent random access to old data. Tune local retention to keep the right slice fast.

## Further reading

- [Kafka Deep Dive Part 1: The Log, Partitions, Segments, and the Page Cache](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage) — the storage foundation this post builds on: segments, the page cache, and zero-copy explained from first principles.
- [Choosing a message broker: Kafka, RabbitMQ, Pulsar, NATS, SQS](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs) — how storage architecture and tiered-storage economics factor into broker selection.
- [Throughput vs latency: the tuning tradeoff](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff) — the broader framework for the dirty-ratio, batching, and buffer tradeoffs touched on here.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the conceptual model of the log that the storage engine implements.
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — why a partition is the unit of parallelism and how that shapes disk layout.
- "The Pathologies of Big Data" and the Linux kernel documentation on `vm.dirty_*` and `sendfile` — the primary sources for the page-cache and zero-copy mechanics.
- The Kafka Improvement Proposal for tiered storage (KIP-405) — the canonical design document for offloading segments to object storage, including the remote segment metadata model and the fetch path.

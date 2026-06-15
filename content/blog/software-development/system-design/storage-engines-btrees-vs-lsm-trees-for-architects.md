---
title: "Storage Engines for Architects: B-Trees vs LSM-Trees, and When Each Wins"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Stop re-deriving how B-trees and LSM-trees work and start choosing between them: a senior's guide to read, write, and space amplification, the decision matrix, compaction tuning, and the production incidents that teach the trade-offs."
tags:
  [
    "system-design",
    "storage-engine",
    "b-tree",
    "lsm-tree",
    "compaction",
    "write-amplification",
    "architecture",
    "distributed-systems",
    "scalability",
    "databases",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-1.webp"
---

Six months into a project, someone on your team picks a database. Maybe they typed `CREATE TABLE` in Postgres because that's what they know. Maybe they reached for Cassandra because a blog post said it "scales." Either way, a decision got made that you will live with for years — and almost nobody in the room understood that the *real* decision wasn't "Postgres or Cassandra." It was "B-tree or LSM-tree." That single choice, buried two layers below the API, governs how much disk you burn per write, how badly your tail latency degrades under load, whether your ingest pipeline survives a traffic spike, and how much money you spend on SSDs. Pick the wrong engine for your workload and no amount of indexes, caching, or vertical scaling buys it back. You will be fighting physics.

This post is not about *how* a B-tree or an LSM-tree works internally — that ground is already covered in depth in [B-Trees: How Database Indexes Really Work](/blog/software-development/database/b-trees-how-database-indexes-work) and [LSM Trees: The Write-Optimized Engine](/blog/software-development/database/lsm-trees-write-optimized-storage-engines). Read those if you want the page-split math and the from-scratch implementations. This post is the layer above: the *architect's choice*. Given a workload, which engine wins, and why? What does each one cost in real numbers — read amplification, write amplification, space amplification — and how do you move those costs around with tuning? When does an LSM fall over under sustained writes? When does a B-tree thrash under random-write load? And which of the databases on your shortlist is secretly which engine?

![A side-by-side of a B-tree scattering random in-place page rewrites versus an LSM-tree turning the same writes into one sequential append](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-1.webp)

The figure above is the entire debate in one image, and it's worth holding in your head for the rest of this article. On the left, a B-tree absorbs a write by seeking to a specific leaf page somewhere in a billion-row index, mutating it, possibly splitting it, and writing whole pages back to random disk locations. On the right, an LSM-tree absorbs the *same* logical write by appending sequentially to a log and updating an in-memory table, deferring all the sorting and merging to background compaction. Random writes versus sequential appends. That is the fork in the road, and everything else — the amplification trade-offs, the compaction strategies, the production incidents — is a consequence of it. By the end of this post you will be able to walk into a design review, name the workload, name the engine, compute its amplification cost on the back of an envelope, and defend the choice against the person who just wants to use what they used last time.

## The three numbers that decide everything: amplification

Forget Big-O for a moment. The unit of cost in a storage engine is not "comparisons" — it is **physical I/O**, and the language seniors use to reason about it is *amplification*. There are exactly three kinds, and every storage-engine decision is a negotiation between them.

**Read amplification** is how many physical reads you do to satisfy one logical read. If a point lookup of one key requires touching 6 files on disk plus 4 bloom-filter probes, your read amplification is high. A B-tree with a bounded depth of 4 has a read amplification of about 4 page reads, period. An LSM-tree with 7 levels might, in the worst case, probe all 7 before finding the key.

**Write amplification** is how many bytes the engine physically writes for each byte of logical data you hand it. You `INSERT` a 100-byte row; the engine writes 100 bytes to a write-ahead log, then later rewrites it during a page split or a compaction merge, perhaps several times as it moves down the levels. A write amplification of 30 means your 100 MB of application data turned into 3 GB of disk writes. This is the number that wears out SSDs and saturates disk bandwidth.

**Space amplification** is how many bytes on disk you consume per byte of live logical data. If you have 100 GB of unique current rows but the engine is holding 220 GB on disk because of un-compacted overwrites, dead tombstones, and stale versions, your space amplification is 2.2x. This is the number that fills disks and inflates your storage bill.

![A matrix showing read, write, and space amplification across a B-tree, size-tiered LSM, and leveled LSM, where each engine is strong on some rows and weak on others](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-2.webp)

Here is the iron law, and it is the single most important idea in this article: **you cannot minimize all three at once.** This is the storage-engineering equivalent of the project-management triangle. Tune for low write amplification and you pay in space and read amplification. Tune for low read amplification and you pay in write amplification. Tune for low space amplification and you pay in write amplification (more compaction work). The triangle in the figure above is not a diagram of three independent dials — it is a diagram of one budget split three ways. When a vendor tells you their engine is "the best at everything," they are either lying or they have silently picked a corner of this triangle that happens to match the benchmark they ran. A senior's job is to know *which corner the workload wants* and to tune toward it deliberately.

| Amplification | What it costs you | What it limits | Who suffers |
| --- | --- | --- | --- |
| Read | Extra disk reads per lookup | Point/range read latency, p99 | Read-heavy services, range scans |
| Write | Extra bytes written per insert | Disk bandwidth, SSD lifespan, ingest ceiling | Write-heavy pipelines, high churn |
| Space | Extra disk per live byte | Storage cost, disk-full risk | Cost-sensitive, large datasets |

Internalize this table and most storage decisions become mechanical. You ask: which of these three numbers is the one my workload cannot afford to let blow up? Then you pick the engine and the compaction strategy that protects that number, and you accept that you are paying on the other two axes. That is the entire discipline.

## Where the write bottleneck actually lives

Most engineers think "writes are slow" without being able to say *why*, which means they can't fix it. Let's be precise, because the location of the write bottleneck is exactly what separates the two engines.

A modern NVMe SSD can do something like 500,000–1,000,000 random 4 KB write IOPS in a burst — but *sustained* random write throughput is much lower, often 5–10x below the sequential number, because the SSD's flash translation layer has to garbage-collect and rewrite flash blocks behind the scenes. A spinning disk is far more brutal: a 7200 RPM drive does maybe 100–200 random IOPS but can stream 150–250 MB/s sequentially. The gap between random and sequential I/O is the single most exploitable fact in all of storage engineering. Sequential is cheap. Random is expensive. On a spinning disk the gap is roughly 1000x; on flash it is smaller but still 5–10x, and random writes additionally *wear the drive out* through write amplification inside the SSD itself.

A B-tree's write path is *fundamentally random*. Each logical insert has to land at the leaf page that owns that key range, and consecutive inserts in your application (a new user, then another new user) land at completely unrelated offsets in the file. The engine reads the target page (possibly a cache miss), mutates it in memory, and writes it back — and if the page is full, it splits, allocating a new page and rewriting the parent, which can cascade up the tree. Every one of those page writes is a random 8–16 KB I/O. The B-tree pays an `fsync` somewhere too: either it `fsync`s dirty pages on a checkpoint, or — more commonly — it writes to a sequential write-ahead log first (which *is* sequential and cheap) and lets a background process flush the random page writes later. The WAL hides some of the latency, but it does not change the physics: the data eventually lands at random offsets, and on a write-saturated workload that random write bandwidth is your ceiling.

An LSM-tree's write path is *fundamentally sequential*, by construction. The whole point of the design is to refuse to ever update in place. A write does exactly two things: append to the WAL (sequential), and insert into the in-memory memtable (RAM, no disk at all). The client's write is acknowledged the moment the WAL `fsync` returns. When the memtable fills, it is flushed to disk as one big sorted SSTable — a single sequential write of already-sorted data. No seeks, no in-place mutation, no page splits. The random-ish I/O is deferred entirely to background compaction, where it is *batched* and *amortized* across millions of keys, turning what would have been a million tiny random writes into a handful of large sequential merges. This is why an LSM can ingest at a rate a B-tree simply cannot match on the same hardware. The write bottleneck moved from the hot path to a background process you can throttle, schedule, and tune.

> The B-tree's bottleneck is random write bandwidth on the hot path. The LSM's bottleneck is whether background compaction can keep up with foreground ingest. Both are real ceilings — but the LSM's ceiling is *movable*, because compaction is a knob, and the B-tree's is closer to a hard wall set by your disk's random-write IOPS.

This distinction is the seed of every later trade-off. The LSM bought write throughput by deferring work — and the deferred work is *debt*, which you pay back in read amplification (now a key might be in any of several SSTables) and space amplification (the same key can exist in multiple files until compaction reconciles them). The B-tree never takes on that debt — every read touches a single bounded-depth path, and a key exists in exactly one place — but it pays cash up front, on every single write, in random I/O. Defer-and-pay-later versus pay-as-you-go. That framing will carry you through the rest of this post.

## Read amplification: the LSM's tax and the B-tree's bound

Let's flip to the read side, because this is where the B-tree's discipline pays off and the LSM's deferral comes due.

A B-tree read is gloriously bounded. To find a key, you start at the root, do a binary search within the page to pick a child, descend, repeat. With a fanout of a few hundred children per page, a tree over a billion rows is only 3–4 levels deep. So a point lookup is *at most* 3–4 page reads, and the upper levels are almost always cached in RAM, so in practice it is often 1 cold read at the leaf. The read amplification is small, bounded, and predictable. A range scan is even better: descend once to the start of the range, then walk the *linked leaf chain* sideways, reading sequential leaf pages, never touching the upper levels again. This is why a B-tree is magnificent at `WHERE created_at BETWEEN ... AND ...` and `ORDER BY` — the data is physically sorted and chained, so a range query is a sequential scan, the one thing disks love.

![A branching graph of the LSM read path showing a memtable check, bloom-filter probes per level, skipped SSTables, and a final merge where the newest value wins](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-6.webp)

An LSM read is a different animal, shown branching above. A key might live in the memtable, or in any SSTable on any level, and there's no single index that says where. So a read potentially has to check *everywhere*: the memtable first (RAM, cheap), then each level. Without help this would be catastrophic — probing 7 levels with multiple files each. Two mechanisms rescue it. First, **bloom filters**: each SSTable carries a small probabilistic filter that can say "this key is *definitely not* here" with certainty (and "probably here" with a tunable false-positive rate). A bloom probe is a few memory accesses, and it lets a read skip almost every SSTable that can't hold the key. Second, **leveled layout**: in a leveled LSM, each level below L0 has *non-overlapping* key ranges, so within a level there is exactly one SSTable that could hold the key — a binary search picks it directly. The combination means a well-tuned LSM point read does the memtable check plus a bloom probe per level, hits at most one or two actual SSTables, and merges to return the newest value (the most recent write wins; older versions and tombstones are filtered out).

So the LSM's read amplification is real but tameable. The worst case — a key that *doesn't* exist, so every bloom filter has to be probed and the false-positives force real reads — is the expensive one. And **range scans** are the LSM's genuine weakness: a range query can't use bloom filters at all (bloom filters only answer point membership), so it must open and merge the relevant SSTables across all levels, reconciling them into one sorted stream. That's K-way merge work proportional to the number of overlapping runs. A B-tree does a range scan in one descent plus a leaf walk; an LSM does it as a merge across levels. This is the cleanest single reason that read-heavy, range-scan-heavy OLTP workloads belong on a B-tree.

| Read pattern | B-tree | LSM-tree |
| --- | --- | --- |
| Point lookup (hit) | 1–4 page reads, bounded | memtable + bloom probes, ~1–2 SSTable reads |
| Point lookup (miss) | 1–4 page reads, bounded | every level's bloom probed; cheap if no false positives |
| Range scan | descend once + walk leaf chain | merge K overlapping runs across levels |
| Latest version of a hot key | one place, in-page | likely in memtable or L0, often cheap |
| p99 predictability | tight, depth-bounded | wider, depends on compaction state |

The deeper point for an architect: the B-tree's read cost is a *property of the structure* (bounded depth), so it's predictable and stable. The LSM's read cost is a *property of the current compaction state* — if compaction is keeping up, reads are fast; if compaction has fallen behind and L0 has a pile of overlapping files, every read suddenly has to check all of them, and your read p99 degrades exactly when your write load is highest. The LSM couples read latency to write pressure in a way the B-tree does not. Remember that — it is the heart of the "write stall" incident we'll stress-test later.

## Durability and fsync: the latency you cannot avoid

There's a fourth cost that hides underneath all three amplifications and bites both engines: **durability**. A write isn't durable until the bytes are physically on stable storage, and "physically on stable storage" means an `fsync` (or `fdatasync`) has returned, forcing the OS and the drive to flush their write buffers to the actual flash or platter. An `fsync` on a consumer SSD is a few hundred microseconds to a couple of milliseconds; on a drive without a power-loss-protected write cache it can be much worse. That `fsync` sits on the critical path of every durable write, and it is the same physics for a B-tree and an LSM — both have a WAL, both must sync it before acknowledging.

So why does this matter to the engine choice? Because of **how the sync amortizes**. An LSM's WAL is a single sequential append stream, which means many concurrent writes can be batched into one `fsync` — a technique called **group commit**. A hundred writes arrive in a 1 ms window, they all append to the WAL, and one `fsync` makes all hundred durable. The per-write sync cost is amortized across the batch, so effective write throughput scales far past what one-`fsync`-per-write would allow. B-tree engines do group commit on their WAL too (Postgres's `commit_delay`, MySQL's `innodb_flush_log_at_trx_commit`), but the B-tree *also* has to eventually flush its random dirty data pages, and those flushes are random I/O that the WAL's sequential group-commit trick can't help. The LSM defers the random work entirely; the B-tree only defers it to a background writer that still has to do it.

This produces a knob both engines expose and that architects routinely get wrong: **how often, and how hard, to fsync.** The choices form a durability-vs-latency spectrum:

| Setting | Durability guarantee | Latency cost | When to use |
| --- | --- | --- | --- |
| fsync every write | survives power loss, no data loss | full sync on hot path | financial ledgers, anything you cannot lose |
| group commit (batch fsync) | survives power loss, tiny batching window | sync amortized across batch | the sane default for most durable systems |
| fsync every N ms / periodic | lose last N ms on crash | near-zero per-write | high-throughput logs that tolerate a few ms loss |
| no fsync (OS cache only) | lose seconds on power loss | none | caches, regenerable data, dev only |

The senior framing: `fsync` is the one cost you genuinely *cannot* tune away if you require durability — you can only amortize it (group commit) or relax it (accept bounded data loss). Many "we made writes 5x faster" stories are quietly a downgrade from per-write `fsync` to periodic `fsync`, trading a few milliseconds of crash-window data loss for throughput. That can be a perfectly good trade — for a metrics store, losing the last 50 ms of samples on a crash is invisible — but it is a *trade*, and naming it is the difference between an informed decision and silent data loss. Whichever engine you pick, decide your durability point on this spectrum explicitly, because both engines will happily run anywhere on it and the default is not always the one you want.

#### Worked example: the cost of synchronous fsync at 50k writes/second

Take the event pipeline again: **50,000 durable writes/second**. Suppose you naively `fsync` once per write, and each `fsync` costs **1 ms** on your drive. One thread doing serial sync writes caps at 1000 writes/s — you'd need **50 threads** just to hit 50k writes/s, and that's if `fsync` calls don't contend on the same WAL file (they do). This is a textbook way to be 50x slower than your hardware allows while the CPU sits idle.

Now switch to **group commit** with a 1 ms batching window. In each 1 ms window, all writes that arrived — at 50k/s that's ~50 writes per millisecond — share a single `fsync`. So you do roughly **1000 fsyncs/second** (one per millisecond) regardless of write rate, each making ~50 writes durable. The per-write sync cost collapsed from 1 ms to ~20 microseconds amortized, and a single WAL writer now sustains the full 50k writes/s with one drive. Same durability guarantee (survives power loss; at most the in-flight 1 ms batch is at risk, and it isn't acknowledged yet), 50x the throughput. This is *the* reason every serious storage engine does group commit, and it's why "we need more write throughput" is often answered by fixing the commit batching, not by changing engines. The LSM gets this amortization most naturally because its entire write path is one sequential log; the B-tree gets it on the WAL but still owes the random page flushes behind the scenes.

## Write amplification: the B-tree's hidden cost and the LSM's compaction bill

Now the symmetric story on the write side, because the naive intuition — "B-trees write less because they don't compact" — is *wrong* in important regimes.

A B-tree's write amplification comes from three sources. First, the **WAL**: most B-tree engines log every change to a write-ahead log before applying it, so a write is logged once and applied once — that's already a 2x floor before anything else. Second, **full-page writes**: to change a few bytes in a row, the engine rewrites the entire 8–16 KB page, so a 100-byte update can become an 8 KB physical write — an 80x amplification on that single update. Postgres makes this worse with its `full_page_writes` setting, which (to survive partial-page-write torn pages on crash) logs the *entire* page to the WAL the first time it's touched after a checkpoint. Third, **page splits**: when a page fills, it splits into two, rewriting both plus the parent. None of this is compaction, but it is very real write amplification, and on a workload of small random updates to a large index, a B-tree's effective write amplification can quietly hit 10–30x.

An LSM-tree's write amplification comes almost entirely from **compaction**. Each key is written once on flush, then rewritten every time compaction merges its SSTable into a deeper level. How many times depends on the compaction strategy. The figure below previews the two dominant strategies, which we'll dissect in the optimization section — but the headline is that they sit at opposite ends of the write-amplification scale.

![A before-and-after comparison of size-tiered compaction with low write amplification but high space and read amplification versus leveled compaction with the opposite trade](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-8.webp)

With **leveled compaction** (RocksDB's default, Cassandra's LCS), each level is ~10x bigger than the one above and holds non-overlapping SSTables. To merge a file from L(n) into L(n+1), you have to rewrite the overlapping portion of L(n+1) — and since L(n+1) is 10x bigger, you rewrite roughly 10 bytes of L(n+1) for every byte coming from L(n). Stack that across ~6 levels and your write amplification is roughly 10 × number-of-levels, landing in the 20–40x range. That is *a lot* — a leveled LSM can write more bytes to disk per logical byte than a B-tree. With **size-tiered compaction** (Cassandra's STCS default historically, common in write-heavy setups), you instead wait until you have several similarly-sized SSTables and merge them all at once into one bigger SSTable. Each key gets rewritten only ~log(N) times as it climbs the size tiers, giving a much lower write amplification of roughly 10x — but at the cost of much higher *space* amplification (you keep multiple large overlapping SSTables around) and higher *read* amplification (a read may have to check all of them).

So the honest comparison is not "B-tree writes less, LSM writes more." It's:

- For **small random updates to a large dataset** (classic OLTP churn), a B-tree's full-page-write amplification can be *worse* than a well-tuned size-tiered LSM, because every tiny update rewrites a whole page.
- For **append-mostly, high-ingest** workloads (logs, events, time series, metrics), an LSM's sequential writes and amortized compaction crush a B-tree, even at 20–40x leveled write amplification, because sequential bandwidth is 5–10x cheaper than random and the B-tree would be bottlenecked on random IOPS long before bandwidth.

This is why the workload *shape* — not just its volume — decides the engine. Two systems doing the same total write bytes per second can want opposite engines depending on whether those writes are append-heavy or update-heavy, sequential-in-keyspace or scattered.

#### Worked example: write amplification for a B-tree vs an LSM under an event-ingest workload

Let's put numbers on it. Suppose you're building an event-ingestion pipeline: an analytics or audit-log system taking **50,000 writes/second**, each event **200 bytes**, keyed by `(tenant_id, event_time, uuid)` so writes are effectively append-ordered in time but spread across tenants. Logical write rate: 50,000 × 200 B = **10 MB/s** of application data, or about **864 GB/day**.

**B-tree path.** The index is large — say 2 TB, far bigger than RAM, so the target leaf pages are usually cold. Each 200-byte insert reads a leaf page (cache miss, random read), mutates it, and writes it back: a random 16 KB page write. Even ignoring splits, that's 16 KB physical per 200 B logical = **80x page-level write amplification**. Add the WAL (another 1x of the row plus periodic full-page-write logging) and the effective figure is comfortably in the **30–80x** range depending on how often a page is dirtied before flush. At 80x, 10 MB/s of logical writes becomes **800 MB/s** of physical random writes. A single NVMe drive sustaining maybe 200–400 MB/s of *random* writes is already saturated — you are out of write headroom at well under your target, and that's before secondary indexes, each of which multiplies the random-write count again. The B-tree hits the wall.

**LSM path (leveled, write amp ~25x).** The same 10 MB/s logical lands in the memtable and WAL sequentially. Compaction rewrites each key ~25 times as it descends levels: 10 MB/s × 25 = **250 MB/s** of *sequential* compaction writes. Sequential 250 MB/s is comfortable for a single NVMe drive (which streams 1–3 GB/s sequentially). So the LSM sustains the workload with headroom to spare — despite writing 25x — because its writes are sequential and the B-tree's are random. The lesson: **write amplification is meaningless without the sequential-vs-random qualifier.** 25x sequential beats 80x random by a wide margin. An architect who compares the raw amplification numbers without that qualifier will reach the wrong conclusion.

**LSM path (size-tiered, write amp ~10x).** If you tune the same LSM to size-tiered compaction, write amp drops to ~10x → 100 MB/s of compaction writes, even more headroom — but now you pay in space (let's say 1.7x, so your 2 TB of live data occupies ~3.4 TB on disk) and in read amplification (range scans must merge more overlapping runs). For a write-dominated audit log that's rarely range-scanned, that's an excellent trade. This is the optimization lever, and we'll come back to it.

## Space amplification: the cost nobody budgets for

Space amplification is the quiet one. It doesn't show up in latency dashboards; it shows up as a disk-full page at 3 a.m. and a storage bill that's 2x what you modeled.

A B-tree's space amplification is naturally low — typically around 1.1–1.3x. Data exists in exactly one place (the leaf that owns its key), and the only "waste" is partially-full pages (B-trees keep pages 50–70% full on average to leave room for inserts) and index overhead. There are no duplicate versions of a key sitting around. This predictability is a real architectural advantage: with a B-tree you can size your disks from `live_data × 1.3` and be done.

An LSM's space amplification is the price of never updating in place. Until compaction reconciles them, the same key can exist in the memtable, in L0, and in a deeper level, all with different versions — only the newest is live, but they all occupy disk. Deletes make it worse: an LSM can't actually remove a key on delete (the key might still exist in deeper SSTables it would have to rewrite), so it writes a **tombstone** — a marker that says "this key is dead" — and the tombstone *and* the dead data both sit on disk until a compaction finally drops them. A delete-heavy or update-heavy LSM workload can accumulate alarming amounts of dead weight. Worst case, during a compaction, you transiently need room for both the input SSTables *and* the output SSTable before the inputs are deleted, so peak space usage can spike well above steady state.

The numbers depend entirely on compaction strategy, which is exactly why it's a tuning knob:

- **Leveled compaction**: space amplification ~1.1x in steady state, because each level is kept tightly non-overlapping and dead data is reclaimed aggressively. You pay for that low space amp with high *write* amp (20–40x). This is the right default when disk is expensive and reads matter.
- **Size-tiered compaction**: space amplification 2x or worse, because large overlapping SSTables linger until enough same-size files accumulate to trigger a merge. The infamous failure mode: a size-tiered table needs roughly **2x its data size in free disk** to safely run its largest compaction, so a node that's 60% full can fail to compact and slowly bloat. You buy lower *write* amp (~10x) with that space cost.

This is the third corner of the triangle, and it completes the picture: leveled trades write-amp for space-amp; size-tiered trades space-amp for write-amp; the B-tree avoids both LSM problems but pays in random-write throughput on every insert. There is no free corner.

#### Worked example: sizing the compaction debt for an ingest pipeline

Back to the event pipeline: **864 GB/day** of logical writes, retained for **30 days**, so ~**26 TB of live data** at steady state on an LSM. Now we size the disk, and this is where teams get burned.

If you run **leveled** compaction (space amp ~1.1x), steady-state disk is 26 TB × 1.1 ≈ **29 TB**, plus headroom for the transient compaction spike — call it 35 TB provisioned. Comfortable.

If you run **size-tiered** compaction (space amp ~2x at peak), steady-state disk is 26 TB × 2 ≈ **52 TB**, and the largest compaction can transiently need to hold input + output simultaneously, pushing the requirement toward **2x the largest table on top of steady state**. Now you're provisioning **60+ TB** for 26 TB of live data — more than double. If your capacity planning assumed "26 TB of data, give it 30 TB of disk," size-tiered compaction will fill that disk and *then fail to compact*, because it can't find room to write the merge output, and the node enters a death spiral: SSTables pile up, reads slow as they probe more files, and eventually writes stall. This is one of the most common LSM production incidents, and it is a *space-amplification* failure masquerading as a write failure.

The senior move is to compute **compaction debt** explicitly. Compaction debt is the volume of data waiting to be compacted — the gap between "data on disk" and "data if fully compacted." You monitor it directly (RocksDB exposes `pending-compaction-bytes`; Cassandra exposes pending compaction tasks). The rule: **your compaction throughput must exceed your write ingest throughput, with margin, or debt grows without bound and the node dies.** If you ingest 10 MB/s logical, and leveled compaction needs to do 25x = 250 MB/s of compaction I/O to keep up, then your disk and CPU must sustain 250 MB/s of compaction *on top of* serving reads and flushes. The day a traffic spike pushes ingest above what compaction can drain, debt accumulates, and you have minutes-to-hours before it becomes an incident. Capacity planning for an LSM is not "size the data" — it's "size the compaction throughput." For the broader discipline of turning these into back-of-envelope numbers, see [Back-of-the-Envelope Estimation for System Design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design).

## How compaction actually moves the data (and the cost)

To tune compaction you have to picture where the data lives and how it migrates. The pipeline below is the canonical RocksDB/LevelDB shape.

![A pipeline of the LSM compaction flow moving a write from the memtable through overlapping L0 SSTables into the sorted non-overlapping L1 and deeper levels](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-3.webp)

A write enters the **memtable** (a sorted in-memory structure, typically 64 MB). When it fills, it's frozen and flushed to disk as an SSTable in **L0**. L0 is special: its SSTables can have *overlapping* key ranges, because they're just flushed memtables in arrival order — nobody sorted them relative to each other. That's why L0 is the read-amplification hotspot: a key could be in any L0 file, so a read has to check them all (bloom filters help, but the more L0 files, the more probes). Compaction's first job is to merge L0 files into **L1**, where SSTables are kept *non-overlapping* and globally sorted. From L1 down, each level is ~10x the size of the one above, and compaction continually merges files from level N into level N+1, dropping overwritten versions and expired tombstones as it goes. The deepest level holds the bulk of the data, rarely touched.

The level hierarchy is worth seeing as a stack, because it explains why an LSM read is logarithmically bounded despite all the files:

![A stack of LSM levels from a 64MB memtable down through levels each roughly ten times larger, so a terabyte fits in six or seven probeable levels](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-7.webp)

Because each level is 10x the last, the number of levels grows only *logarithmically* with the data: 64 MB → 256 MB → 2.5 GB → 25 GB → 250 GB → 2.5 TB is just six levels for several terabytes. A read therefore probes at most ~6 levels (one SSTable each below L0, thanks to non-overlap), plus the L0 pile and the memtable. This is the structural reason an LSM read stays sub-logarithmic even as data grows — *as long as compaction keeps the levels in shape.* The whole system's health reduces to one question: is compaction keeping up? When it is, the stack looks like the figure — clean, one file per level, fast reads. When it isn't, L0 bloats, levels overlap, and reads degrade. Hold that picture; the stress-test section is entirely about what happens when this stack falls out of shape.

A senior watches a small set of LSM signals that map directly onto this picture:

- **L0 file count**: how many overlapping files are piled up at the top. Rising L0 count = compaction falling behind = read amplification climbing. RocksDB triggers write *slowdowns* and then *stalls* at configurable L0 thresholds precisely to stop this.
- **Pending compaction bytes / pending tasks**: the compaction debt. Trending up = you are losing the race.
- **Write stall time**: the engine's emergency brake. If you see it, ingest has already outrun compaction.
- **Read p99 vs L0 count correlation**: when these move together, your reads are coupled to compaction state — the warning sign before an incident.

## The decision: mapping a workload to an engine

Now we can make the call. The whole analysis collapses into a short decision procedure, drawn as a tree below.

![A decision tree taking a workload from the dominant-load question through write-heavy versus read-heavy branches to an LSM or B-tree leaf](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-5.webp)

Walk the tree. The root question is *what dominates your load*. If you're **write-heavy / high-ingest** — logs, events, metrics, time series, a chat backend, an IoT firehose, anything where writes vastly outnumber reads and arrive faster than a B-tree's random-write ceiling — you want an **LSM-tree** (Cassandra, ScyllaDB, RocksDB-backed stores). Then a second question: do you also do *many in-place updates or deletes*? If so, watch your space amplification and lean toward leveled compaction, because update/delete churn is what bloats an LSM. If you're **read-heavy with range scans and you need strong transactions** — classic OLTP, an application database, anything doing `WHERE ... BETWEEN`, `ORDER BY`, joins, and ACID transactions — you want a **B-tree** (Postgres, MySQL/InnoDB). The B-tree's bounded read depth, native sorted-order scans, and mature transactional machinery are exactly what that workload needs, and its random-write cost is irrelevant because you aren't write-bound.

The full trade-off matrix makes the symmetry explicit:

![A matrix comparing a B-tree and an LSM-tree across point lookup, range scan, write throughput, and the three amplification dimensions](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-4.webp)

| Property | B-tree | LSM-tree | Winner |
| --- | --- | --- | --- |
| Point lookup | 1–4 page reads, bounded | bloom + 1–2 SSTable reads | roughly even; B-tree more predictable |
| Range scan / `ORDER BY` | descend + leaf-chain walk | merge K runs across levels | **B-tree** |
| Write throughput (high ingest) | random-write IOPS ceiling | sequential append, amortized | **LSM** |
| Read amplification | low, bounded by depth | higher, depends on compaction | **B-tree** |
| Write amplification (small updates) | high (full-page writes) | tunable, often lower | **LSM** |
| Space amplification | low (~1.1x) | tunable 1.1x–2x+ | **B-tree** |
| Strong transactions / MVCC maturity | decades of polish | improving, varies | **B-tree** |
| Predictable p99 under write load | stable | couples to compaction | **B-tree** |

Read the matrix as a senior reads it: there is no globally better engine. The B-tree wins the read/range/space/predictability columns; the LSM wins the write-throughput and (often) write-amp columns. So you don't ask "which is better" — you ask "which columns does my workload live or die on?" An audit-log pipeline lives or dies on write throughput → LSM. A user-facing transactional app lives or dies on range-scan latency and predictable p99 → B-tree. The matrix is the artifact you bring to the design review; it turns a religious argument into a column-by-column decision.

There's a subtlety worth flagging: **secondary indexes**. On a B-tree database, every secondary index is *another B-tree* that every write must also maintain at *its* random location, multiplying write amplification per index. This is a major reason write-heavy B-tree workloads degrade — five indexes can mean five random page writes per insert. LSM engines pay for secondary indexes too, but typically as additional column families or index SSTables that still write sequentially. If your write-heavy workload also needs several secondary indexes, the B-tree's write cost compounds and pushes you harder toward the LSM. This connects to the broader datastore decision — once you've chosen the engine *family*, picking the actual product is the subject of [Choosing a Datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql).

## Second-order consequences: how the engine ripples through the system

The engine choice doesn't stay confined to the storage layer. It leaks upward and outward into replication, the data model, and the access patterns your application can afford — and a senior anticipates those ripples before they become surprises.

**The data model bends to the engine.** An LSM rewards an *append-and-read-recent* access pattern and punishes *read-modify-write* on hot keys. A read-modify-write — read the current value, change it, write it back — is cheap on a B-tree (the row is in one place, in the buffer pool) but expensive and dangerous on an LSM, because the "read" half pays read amplification and the "write" half creates yet another version that compaction must later reconcile. Worse, on a distributed LSM like Cassandra, a naive read-modify-write across replicas is a lost-update race. The idiomatic LSM data model therefore avoids read-modify-write entirely: it uses *append-only* event records, *last-write-wins* on whole values, or purpose-built CRDT-like structures (counters, sets) that merge without reading first. If your domain is fundamentally about mutating shared state in place — bank balances, inventory counts with strict invariants — the LSM is fighting you, and the B-tree's in-place update with a row lock is the natural fit. The engine quietly dictates which data models are cheap, and choosing an engine that fights your data model is a slow, expensive mistake you discover six months in.

**Deletes and TTLs are a hazard on an LSM and a non-event on a B-tree.** On a B-tree, a delete frees the row's slot in its page; it's done. On an LSM, a delete writes a **tombstone** that lingers until compaction reclaims it, and *reads must scan past tombstones* until then. A workload that deletes heavily — a queue table, a session store, anything with high churn and TTL expiry — can accumulate so many tombstones that reads over a key range slow to a crawl. This is the exact trap Discord hit. The senior defenses are structural: use **time-windowed compaction** so whole expired windows drop without per-tombstone work; model queue-like data so you read forward past a moving cursor rather than range-scanning over deleted entries; and monitor tombstone counts as a first-class metric. None of this is a concern on a B-tree — which is a real, often-overlooked point in the B-tree's favor for delete-heavy and TTL-heavy workloads.

**Replication inherits the engine's write character.** When you replicate, you ship the write workload to every replica — so the engine's write cost is multiplied by the replication factor across the cluster. An LSM's sequential, compactable writes replicate cheaply and its log-structured nature pairs naturally with streaming SSTables or shipping the WAL. A B-tree's random writes replicate as either physical page changes (which can be large) or logical row changes (which each replica re-applies, re-paying the random-write cost locally). The interaction with replication strategy is real: an LSM-backed system tends to favor leaderless or multi-leader replication with last-write-wins conflict resolution (Cassandra's model), while a B-tree-backed system tends to favor single-leader replication with a strict log (Postgres/MySQL). The engine and the replication topology co-evolve, and the broader treatment of that space lives in the database series. The point for now: when you pick the engine, you are also nudging the entire replication and consistency story of the system, because the engine's write character is what gets replicated.

The meta-lesson across all three ripples: **the storage engine is not a leaf in your architecture — it's closer to the root.** It constrains your data model, your delete strategy, your replication topology, and your consistency options. That's exactly why the architect should make this choice consciously and early, rather than inheriting it from whoever typed the first `CREATE TABLE`.

## Optimization: tuning the amplification triangle

Choosing the engine is half the job. The other half is *moving the workload around the amplification triangle* with tuning, because the same LSM can be configured to sit in very different corners. This is where seniors earn their keep, and it's almost entirely about three knobs.

### Knob 1: compaction strategy (the big lever)

This is the dominant knob, and it directly sets which corner of the triangle you're in:

- **Leveled (LCS / RocksDB default)**: low read amp (one SSTable per level), low space amp (~1.1x), high write amp (20–40x). Choose when reads matter, disk is expensive, and your write rate leaves compaction headroom. The default for most general-purpose RocksDB and read-sensitive Cassandra tables.
- **Size-tiered (STCS)**: low write amp (~10x), high read amp (merge many runs), high space amp (2x+). Choose for pure write-heavy, rarely-range-scanned, append-mostly data where you have disk to spare and write throughput is sacred. The classic choice for write-dominated time-series and log tables — but watch the 2x disk requirement.
- **Time-windowed (TWCS, Cassandra)**: a specialization for time-series with TTL. Groups SSTables by time window and drops whole expired windows without compacting them, which is dramatically cheaper than tombstone-by-tombstone reclamation. If your data is time-series with a retention window, this is almost always the right answer and avoids the tombstone-accumulation trap entirely.

Switching strategy is the single highest-leverage tuning action. A team drowning in write amplification on leveled compaction can often cut their disk-write bandwidth 2–3x by moving write-dominated tables to size-tiered — accepting the space and read cost as a deliberate trade. Conversely, a team whose reads are timing out because size-tiered let read amplification explode can move hot-read tables to leveled. You are not eliminating cost; you are *relocating* it to the axis you can afford.

### Knob 2: bloom filter bits per key

Bloom filters are the lever on point-read amplification. More bits per key → lower false-positive rate → fewer wasted SSTable reads on lookups (especially lookups for keys that don't exist), at the cost of more RAM. The standard default is **10 bits/key** (~1% false-positive rate). Pushing to 15–20 bits/key drops false positives to ~0.1% or below, which matters enormously for workloads with many negative lookups (checking "does this key exist?" where it usually doesn't — deduplication, existence checks, cache-fill). Each extra bit is roughly proportional RAM: 10 bits/key over a billion keys is ~1.25 GB of bloom filters in RAM. The tuning question is whether your negative-lookup rate justifies the RAM, and it's measurable: watch the bloom false-positive metric and the ratio of bloom-probed-but-not-found reads.

### Knob 3: block cache and the read-path memory budget

The block cache holds recently-read SSTable blocks (and, for a B-tree, the buffer pool holds recently-read pages — same idea). This is where the B-tree's predictability shines and the LSM's hunger shows: a B-tree needs enough buffer pool to hold the upper levels of the tree (small) plus the hot working set; an LSM needs block cache *plus* the bloom filters *plus* the memtable, and it competes for that RAM against compaction's own buffers. A senior sizes the block cache against the *hot working set*, not the total data, and measures the cache hit rate. A 5% miss rate that becomes a 6% miss rate doubles the cold-read traffic that 6% represents if your hot set just barely fit — cache sizing has cliff behavior, and you find the cliff by watching hit-rate-vs-cache-size, not by guessing.

```yaml
# RocksDB-style options sketch (column-family level) for a write-heavy,
# rarely-range-scanned ingest table that must not stall under load.
column_family:
  compaction_style: kCompactionStyleUniversal   # size-tiered: low write amp
  write_buffer_size: 134217728                   # 128MB memtable, fewer flushes
  max_write_buffer_number: 4                      # absorb bursts before stalling
  level0_slowdown_writes_trigger: 20             # start throttling early, gently
  level0_stop_writes_trigger: 36                 # hard stall threshold (last resort)
  bloom_bits_per_key: 10                          # ~1% FP; bump to 15 if many misses
  target_file_size_base: 67108864                # 64MB SSTables
  max_background_compactions: 6                   # give compaction enough threads
  block_cache_size: 8589934592                    # 8GB, sized to hot working set
```

### Knob 4: the B-tree side — buffer pool, fill factor, and checkpoint pacing

The B-tree has fewer knobs than the LSM (which is part of its appeal — less to misconfigure), but the ones it has are decisive, and they map onto the same triangle. The **buffer pool** (`innodb_buffer_pool_size`, Postgres's `shared_buffers` plus the OS page cache) is the single most important: it must hold the hot working set or you thrash, as the page-cache-thrash failure mode below shows. The standard guidance of "60–75% of RAM for the buffer pool" is a starting point, not a law — the real target is "enough to hold the working set with the cache-miss rate below your latency budget." **Fill factor** (how full B-tree pages are kept) trades space amplification against page-split frequency: a lower fill factor leaves room for inserts so pages split less often (less write amplification) but wastes more disk (more space amplification) — the same triangle, on a different engine. **Checkpoint pacing** controls how aggressively dirty pages are flushed to disk: flush too eagerly and you do redundant random writes; flush too lazily and a crash means a long recovery and a checkpoint storm of random writes all at once. The B-tree's checkpoint storm is the structural cousin of the LSM's compaction storm — a burst of deferred random I/O that can starve foreground traffic — and you tame it the same way: spread the work out (`checkpoint_completion_target` in Postgres) so it's a steady trickle rather than a periodic flood.

```sql
-- Postgres-side knobs for a large, write-active OLTP table on a B-tree.
-- The goal: keep the hot working set in cache, spread checkpoint I/O,
-- and avoid full-page-write amplification storms after each checkpoint.
ALTER SYSTEM SET shared_buffers = '16GB';            -- hold the hot working set
ALTER SYSTEM SET effective_cache_size = '48GB';      -- planner hint: OS cache too
ALTER SYSTEM SET checkpoint_timeout = '15min';       -- fewer, larger checkpoints
ALTER SYSTEM SET checkpoint_completion_target = 0.9; -- spread flush over the window
ALTER SYSTEM SET max_wal_size = '8GB';               -- avoid forced early checkpoints
ALTER SYSTEM SET wal_compression = 'on';             -- shrink full-page-write WAL volume
ALTER SYSTEM SET commit_delay = 100;                 -- group commit window (microseconds)

-- Use a time-ordered key (uuidv7 / ULID) so inserts cluster into hot leaf
-- pages instead of scattering random writes across the whole keyspace.
CREATE TABLE events (
  id        uuid PRIMARY KEY,        -- prefer time-ordered uuidv7, NOT random uuidv4
  tenant_id bigint NOT NULL,
  payload   jsonb NOT NULL,
  ts        timestamptz NOT NULL DEFAULT now()
) WITH (fillfactor = 90);            -- leave 10% headroom to reduce page splits
```

The whole optimization mindset is captured in one sentence: **you cannot make a storage engine cheaper, only move where it's expensive.** The knobs above don't reduce total cost — they slide it between read, write, and space, and between RAM and disk. The senior decides *which axis the business can afford to pay on* and tunes toward it, then *measures* the move (write bandwidth, read p99, disk usage, cache hit rate) to confirm the cost landed where intended. The complement to engine tuning at the transaction layer — what isolation costs you and why — is covered in [Isolation Levels and the Anomalies They Prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent).

### Measuring the win: which numbers prove the tune worked

Tuning without measurement is superstition. After any engine or compaction change, a senior validates the move on four concrete metrics, because a change that helps one amplification axis almost always hurts another and you need to confirm the net was positive *for your workload*:

- **Write bandwidth to disk** (bytes/s the device is actually writing). This is your write-amplification proxy: divide it by your logical write rate to get effective write amp. If you switched leveled → size-tiered expecting lower write amp, this number should drop ~2x; if it didn't, the change didn't take.
- **Read p50 and p99** (separately — the mean lies). Read amplification shows up in p99 first, because the tail reads are the ones that had to probe extra SSTables or miss the cache. Watch p99, not the average.
- **Disk usage and space amplification** (bytes on disk ÷ live logical bytes). The number that turns into a disk-full incident if you ignore it. After a size-tiered switch, expect this to climb toward 2x — confirm you provisioned for it.
- **Throughput at fixed latency** (QPS the system sustains while keeping p99 under budget). The honest single-number summary: a tune that raises QPS-at-target-p99 is a real win; one that raises raw QPS but blows the p99 budget is not.

The discipline is to write down the *expected* direction of each metric before the change ("write bandwidth should halve, space should rise toward 2x, read p99 may rise 20%, throughput should hold") and then check reality against the prediction. When reality disagrees with your model, your model was wrong — and finding out *why* is where the real understanding of the engine comes from.

## Stress-testing the design: what breaks at 10×

A design isn't done until you've broken it on paper. Here are the two canonical failure modes, one per engine, and how a senior reasons about them.

### The LSM write stall: when compaction can't keep up

Picture the event pipeline humming along at 50k writes/s on a size-tiered LSM, compaction comfortably draining the debt. Now a product launch triples your ingest to 150k writes/s for an hour. Logical write rate jumps to 30 MB/s; at 10x size-tiered write amp, compaction now needs to do 300 MB/s of merge I/O *plus* serve reads and flush memtables. If your disk and CPU can't sustain that, compaction debt starts accumulating: L0 files pile up, then the engine does the only thing it can to protect itself from unbounded read amplification — it **throttles writes** (RocksDB's `level0_slowdown_writes_trigger`), and if debt keeps climbing, it **stalls writes entirely** (`level0_stop_writes_trigger`). Your write latency, which was 1 ms, suddenly becomes seconds or times out. Meanwhile every read is now probing a huge L0 pile, so read p99 blows up too. The system is in a death spiral: writes are stalled, reads are slow, and the only escape is for compaction to catch up — which it can't, because you're still ingesting.

This is the LSM's signature incident, and it's why the engine's write throughput is a *moving* ceiling, not a fixed one. The defenses a senior bakes in up front:

1. **Provision compaction headroom**, not just steady-state throughput — size for 2–3x peak ingest so a spike doesn't immediately outrun compaction.
2. **Throttle gently and early** (slowdown well before stop) so you trade a little write latency for staying out of the death spiral.
3. **Shard writes across more nodes** — an LSM's write ceiling is per-node, so horizontal partitioning multiplies it. This is exactly where [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) becomes the pressure-release valve.
4. **Cap or shed load at the application layer** with [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) so the firehose can't physically exceed what compaction can drain.

### The B-tree page-cache thrash: when random writes outrun RAM

Now the symmetric B-tree failure. Picture a Postgres or InnoDB instance with a 2 TB index and 64 GB of RAM, taking a sudden surge of *random* inserts — say a backfill job inserting rows whose keys are uniformly scattered across the keyspace (random UUIDs as primary keys is the classic offender). Every insert targets a leaf page that's almost certainly *not* in the buffer pool, because the working set (2 TB) dwarfs RAM (64 GB). So each insert triggers a cold page read, a mutation, and a dirty page that must eventually be written back to a random offset. The buffer pool fills with dirty pages faster than the background writer can flush them; flushing is random-write-bound and slow; eventually the engine has to evict-and-flush synchronously on the insert path, and write latency collapses. You've turned a fast in-memory operation into a random-disk-I/O-bound one, and the disk's random-write IOPS is your hard ceiling. This is **page-cache thrash**: the working set doesn't fit in RAM, so every operation pays a random disk read *and* a random disk write.

The senior defenses:

1. **Use sequential or near-sequential keys** (auto-increment, time-ordered, or `uuidv7`/ULID instead of random `uuidv4`) so inserts cluster into a few hot leaf pages that stay cached — this single change can be the difference between fitting in RAM and thrashing. Random-UUID primary keys are one of the most common self-inflicted B-tree performance wounds in production.
2. **Right-size the buffer pool** to hold the hot working set, and accept that if the working set genuinely exceeds RAM and writes are random, *this is the workload the B-tree is bad at* — which is your signal that you may have chosen the wrong engine.
3. **Batch and sort inserts** before applying them, so random writes become more sequential and pages are dirtied fewer times.
4. **If you're fundamentally write-bound on random keys, migrate to an LSM** — this is precisely the workload the LSM was built for, and no amount of B-tree tuning will match it.

The symmetry is the lesson: the LSM breaks when *background compaction* can't keep up with *foreground writes*; the B-tree breaks when the *random-write working set* can't fit in *RAM*. Both are capacity problems, but they have different shapes, different early-warning signals, and different fixes — and knowing which failure mode your engine is prone to is what lets you provision and monitor for it *before* the 3 a.m. page.

## Case studies: what production taught us

Theory is cheap. Here are three real-world stories where the engine choice and its tuning decided the outcome.

### Discord: Cassandra to ScyllaDB

Discord stores trillions of messages, and they ran them on Cassandra (an LSM-tree database) for years. The workload is brutally write-heavy — every message sent is a write — and reads are dominated by recent messages in a channel, which maps reasonably onto an LSM. But they hit two LSM-specific walls. First, **tombstones**: deleting messages (and the inherent churn of their data model) generated tombstones that accumulated faster than compaction reclaimed them, and reads over ranges full of tombstones got slow — a read that has to scan past thousands of dead markers to find live data is doing a lot of wasted I/O. Second, **compaction and GC pressure**: Cassandra runs on the JVM, and the combination of garbage-collection pauses and compaction competing for the same I/O produced unpredictable tail latencies — exactly the "read p99 coupling to compaction state" failure mode described earlier. Their fix was to migrate to **ScyllaDB**, which is a C++ reimplementation of Cassandra's LSM engine and protocol with no JVM GC pauses, a shard-per-core architecture, and more controllable compaction. The engine *family* stayed the same (LSM was the right choice for the write-heavy workload) — they changed the *implementation* to escape the JVM and compaction-scheduling pathologies. The architectural lesson: when an LSM's tail latency couples to compaction, the answer is often not "switch to a B-tree" (the workload still wants an LSM) but "get an LSM implementation with better compaction control and no GC stalls."

### RocksDB at scale: one engine, many corners of the triangle

RocksDB is the most instructive case study because it's the *same engine* tuned into wildly different shapes across the industry. It started at Facebook as a fork of LevelDB optimized for flash, and it now sits under MyRocks (MySQL with an LSM instead of InnoDB's B-tree), CockroachDB's Pebble (a RocksDB-inspired engine), TiKV, Kafka Streams state stores, and countless internal key-value services. The lesson from RocksDB's ubiquity is precisely the *triangle*: the same codebase is configured for low write amplification (universal/size-tiered compaction) in write-heavy ingest paths, for low read amplification (leveled compaction, fat bloom filters) in read-serving paths, and for low space amplification (leveled, aggressive compaction) where disk is the constraint. MyRocks at Facebook famously cut storage space ~2x and write amplification ~3–4x versus InnoDB for their write-heavy social-graph workload — a direct consequence of swapping a B-tree for a well-tuned LSM where the workload was write-and-space-bound. The architectural lesson: the engine is not a fixed-cost component; it's a *tunable* one, and the same LSM can be parked in any corner of the triangle by configuration. Knowing the knobs *is* the skill.

### The compaction-storm incident: when the background work becomes the foreground problem

A recurring production incident pattern across LSM deployments — Cassandra, HBase, and RocksDB-backed systems alike — is the **compaction storm**. The setup: a cluster running comfortably, then some trigger (a large bulk load, a schema change that rewrites data, a node coming back after downtime and catching up, or simply crossing a data-size threshold) causes a wave of large compactions to fire roughly simultaneously. Compaction is I/O-and-CPU-hungry, so the storm starves the foreground workload: read p99 spikes (compaction is stealing disk bandwidth from reads), write latency climbs (memtable flushes queue behind compaction), and in the worst case the storm causes enough debt that the engine throttles or stalls writes — the death spiral again, but triggered by a *burst of compaction* rather than a burst of ingest. The fixes are all about *governing* compaction: rate-limit compaction I/O (`compaction_throughput_mb_per_sec` in Cassandra, `rate_limiter` in RocksDB) so it can never starve foreground traffic; stagger compactions across nodes so they don't all fire at once; and provision enough I/O headroom that a compaction wave doesn't saturate the disk. The architectural lesson is the deepest one in this post: **in an LSM, the background work is not optional and not free — it is a first-class part of your capacity model, and if you don't budget I/O for it explicitly, it will take that I/O from your users at the worst possible moment.** A B-tree has no equivalent storm because it does its work synchronously on the write path — you pay continuously instead of in bursts, which is its own trade-off.

## When to reach for each (and when not to)

Here is the decisive recommendation, the part you came for.

**Reach for a B-tree (Postgres, MySQL/InnoDB, SQLite) when:**

- Your workload is **read-heavy or balanced**, not a write firehose.
- You do **range scans, `ORDER BY`, joins** — anything that benefits from data being physically sorted and chained.
- You need **mature ACID transactions and MVCC** with decades of polish.
- You want **predictable, bounded read latency** that doesn't fluctuate with background work.
- Your **working set fits in RAM** or your writes are not randomly scattered across a huge keyspace.
- **Do not** reach for it when you're ingesting writes faster than your disk's random-write IOPS, or when secondary-index write amplification is choking you.

**Reach for an LSM-tree (Cassandra, ScyllaDB, RocksDB-backed stores) when:**

- Your workload is **write-heavy / high-ingest**: logs, events, metrics, time series, chat, IoT, append-mostly data.
- Your writes **outrun a B-tree's random-write ceiling** and you need sequential-append throughput.
- You can **afford background compaction I/O** and you'll budget for it as a first-class capacity dimension.
- Your reads are **mostly point lookups or recent-data scans**, not heavy arbitrary range scans.
- You're willing to **own the compaction tuning** — strategy, throughput limits, monitoring debt — as ongoing operational work.
- **Do not** reach for it when your workload is read-and-range-scan-heavy, when you can't provision compaction headroom, or when you need rock-steady p99 that won't move under write pressure.

**The honest middle ground:** most application databases are read-heavy OLTP, and the boring answer — Postgres — is correct far more often than the "we need to scale, let's use Cassandra" instinct suggests. Reach for the LSM when the *write shape* genuinely demands it, not because it sounds more scalable. And remember that the engine choice is one input to the larger datastore decision (consistency model, distribution, operational maturity) covered in [Choosing a Datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — the engine sets the floor for what's physically cheap, but the rest of the system shapes the final choice.

## Where the real databases sit

It pays to recognize, on sight, which engine is under each database on your shortlist — because that engine is a hard architectural fact you inherit, not a setting you toggle at runtime.

![A matrix mapping real databases to their engine family and the workload sweet spot each engine serves](/imgs/blogs/storage-engines-btrees-vs-lsm-trees-for-architects-9.webp)

| Database | Engine | Sweet spot |
| --- | --- | --- |
| Postgres | B-tree (heap + B-tree indexes) | OLTP, range scans, rich queries, strong transactions |
| MySQL / InnoDB | B+tree (clustered index) | OLTP, point + range, mature replication |
| SQLite / LMDB | B-tree | Embedded, read-mostly, predictable |
| Cassandra | LSM-tree | High write ingest, wide-column, multi-DC |
| ScyllaDB | LSM-tree (C++ rewrite) | Same as Cassandra, lower tail latency, no GC |
| RocksDB / Pebble | LSM-tree | Embedded KV, write-heavy, building block under other DBs |
| HBase | LSM-tree | Massive write ingest on HDFS |
| MyRocks | LSM-tree (MySQL on RocksDB) | Write-heavy + space-constrained MySQL workloads |

Two notes for the architect. First, some databases let you *choose* the engine — MySQL can run InnoDB (B-tree) or MyRocks (LSM); this is a deliberate decision, not a default to accept blindly. Second, "NewSQL" distributed databases (CockroachDB, TiDB, YugabyteDB) almost universally build on an **LSM** at the storage layer (Pebble, RocksDB) and layer SQL, transactions, and Raft-based replication on top — so even when you're using SQL with strong transactions, the bytes underneath are often hitting an LSM, and its compaction and amplification characteristics still apply to your capacity model. The SQL surface hides the engine, but the engine's physics doesn't go away.

## Where the line blurs: the engines are converging

The B-tree-versus-LSM framing is the right *first cut*, and it will carry you correctly through almost every design review. But a senior should know that the research and the production engines have been quietly eroding the boundary for a decade, because the trade-offs are not laws of nature — they're consequences of specific design choices that cleverer designs can soften.

On the **B-tree side**, modern variants attack the random-write problem directly. **Fractal trees** and **B-ε trees** (used by TokuDB/PerconaFT and in some research systems) buffer inserts in the internal nodes and flush them down in batches, so a B-tree's writes become more sequential and its write amplification drops toward LSM territory — buying write throughput without giving up the bounded-read structure. **Bw-trees** (Microsoft's Hekaton, used in some SQL Server in-memory paths) use lock-free delta records appended to pages instead of in-place updates, dodging the in-place-write cost that defines the classic B-tree. These exist precisely because the B-tree's random-write weakness is its biggest liability, and you can engineer a lot of it away if you're willing to add complexity.

On the **LSM side**, the innovations attack read and space amplification. **WiscKey** (and RocksDB's `BlobDB` mode) separates keys from values: it keeps only keys in the LSM and stores large values in a separate log, so compaction only rewrites small keys instead of dragging big values down every level — slashing write amplification for large-value workloads. **Learned indexes** and better bloom-filter variants (ribbon filters, for instance) shrink the read-path memory cost. And tiered/leveled *hybrid* compaction strategies (RocksDB's universal compaction with size limits) let you sit *between* the size-tiered and leveled corners of the triangle rather than picking one extreme.

The architect's takeaway is not "learn all the variants" — it's a calibration. The clean B-tree-versus-LSM dichotomy is a *teaching model and a default*, not a ceiling. When you've correctly identified that your workload sits painfully between the two corners — write-heavy *and* range-scan-heavy, say, or LSM-shaped but with huge values — that discomfort is the signal to reach for one of these hybrid engines rather than forcing a pure B-tree or pure LSM to do a job it's bad at. Knowing the dichotomy tells you the default; knowing it's a spectrum tells you when to look past the default. That is the difference between an engineer who memorized a comparison table and an architect who understands the cost surface the table is sampling.

## Key takeaways

1. **The real decision is the engine, not the database.** "Postgres or Cassandra" is downstream of "B-tree or LSM-tree," and the engine governs your write ceiling, read predictability, and storage bill for years.
2. **Count physical I/O, not Big-O.** The currency is read amplification, write amplification, and space amplification — and you cannot minimize all three at once. Every knob relocates cost; none removes it.
3. **Random vs sequential is the whole game.** A B-tree's writes are random (its ceiling is random-write IOPS); an LSM's are sequential (its ceiling is whether compaction keeps up). Write amplification is meaningless without that qualifier — 25x sequential beats 80x random.
4. **The LSM's read cost and write stall couple to compaction state.** Reads get slow exactly when writes are heaviest. Budget compaction I/O as a first-class capacity dimension, monitor compaction debt and L0 file count, and throttle early.
5. **The B-tree's failure mode is page-cache thrash.** When the random-write working set exceeds RAM, every insert becomes random-disk-bound. Sequential keys (ULID, not random UUID) and a right-sized buffer pool are the defenses.
6. **Compaction strategy is the dominant tuning lever.** Leveled = low read/space amp, high write amp. Size-tiered = low write amp, high read/space amp. Time-windowed = the right answer for TTL time-series. Pick the corner your workload can afford.
7. **Size for compaction throughput, not just data volume.** A size-tiered LSM needs ~2x its data in disk and your compaction throughput must exceed ingest with margin, or the node bloats and stalls.
8. **Read-heavy OLTP almost always wants a B-tree.** Reach for an LSM when the *write shape* genuinely demands sequential-append throughput — not because it sounds more scalable.
9. **The engine sets the floor of what's physically cheap.** Recognize which engine is under each database on sight; even NewSQL's SQL surface usually sits on an LSM whose physics you still inherit.

## Further reading

- [B-Trees: How Database Indexes Really Work](/blog/software-development/database/b-trees-how-database-indexes-work) — the mechanism deep-dive: page math, splits, and a from-scratch implementation.
- [LSM Trees: The Write-Optimized Engine](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) — the mechanism deep-dive: memtables, SSTables, bloom filters, and compaction internals.
- [Isolation Levels and the Anomalies They Prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) — what the transaction layer above the engine costs you.
- [Choosing a Datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — turning the engine choice into a full datastore decision.
- [Back-of-the-Envelope Estimation for System Design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the capacity-math discipline behind sizing compaction debt and disk.
- *Designing Data-Intensive Applications*, Martin Kleppmann, Chapter 3 — the canonical treatment of storage engines and the B-tree/LSM trade-off.
- The RocksDB wiki (Tuning Guide, Compaction) and the Cassandra compaction documentation — the authoritative knob references for production tuning.
- *The Log-Structured Merge-Tree (LSM-Tree)*, O'Neil et al., 1996 — the original paper that started it all.

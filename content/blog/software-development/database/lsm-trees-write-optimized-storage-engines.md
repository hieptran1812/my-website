---
title: "LSM Trees: The Write-Optimized Engine Behind RocksDB, Cassandra, and Modern Databases"
date: "2026-06-11"
publishDate: "2026-06-11"
description: "How log-structured merge trees turn random writes into sequential appends, why every modern key-value store is built on them, and how to implement and tune one yourself."
tags:
  [
    "lsm-tree",
    "database",
    "storage-engine",
    "rocksdb",
    "cassandra",
    "compaction",
    "bloom-filter",
    "sstable",
    "write-amplification",
    "system-design",
    "performance",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 51
---

There is a moment in almost every backend engineer's career when they discover that the database is not a magic box. It happens the first time a write-heavy workload — an event pipeline, a metrics store, a chat backend — falls over at a fraction of the throughput the hardware should deliver. You profile it, and the CPU is idle. The network is idle. The disk is busy, but it is doing far more I/O than your data volume can explain. You wrote 100 MB; the disk wrote 4 GB. Where did the other 3.9 GB come from?

The answer is that the default storage engine you inherited — almost always a B-tree — was designed in 1970 for spinning disks and read-mostly workloads, and you are asking it to absorb a firehose of writes. B-trees update data **in place**. Every insert seeks to a specific page, mutates it, and writes the whole page back. On flash, that page write triggers an erase-and-rewrite cycle inside the SSD that can amplify a 100-byte write into a 4 KB (or larger) physical write. Scatter those writes randomly across a billion-row index and you have manufactured your own bottleneck.

Log-structured merge trees — LSM trees — are the answer the industry converged on. RocksDB, Cassandra, ScyllaDB, HBase, LevelDB, MyRocks, TiKV, CockroachDB's Pebble, BadgerDB, InfluxDB, QuestDB, and the storage layer under a dozen "cloud-native" databases are all LSM engines. The core idea is almost insultingly simple: **never update in place. Only ever append.** Buffer writes in memory, dump them to disk in sorted batches, and clean up the mess later in the background. The diagram below is the mental model for the entire article — everything else is a detail of this one picture.

## The mental model: log-structured, not tree-structured

![The LSM write path: every write is a sequential append, and the sorting work is deferred to background compaction](/imgs/blogs/lsm-trees-write-optimized-storage-engines-1.webp)

Read the figure left to right and top to bottom. A write enters and does exactly two cheap things: it is appended to a **write-ahead log** (WAL) for durability, and it is inserted into an in-memory sorted structure called the **memtable**. Both operations are O(log n) in memory or a sequential disk append — no seeks, no page reads, no random I/O. The client's write is acknowledged the instant the WAL `fsync` returns.

When the memtable fills (typically 64 MB), it is frozen — made immutable — and a new one takes its place so writes never block. The frozen memtable is flushed to disk as a **sorted string table** (SSTable): one big sequential write of already-sorted key-value pairs. SSTables are immutable forever. As they accumulate, a background process called **compaction** merges them, discarding overwritten and deleted keys, keeping the total file count bounded.

That is the whole architecture. Writes are fast because they are sequential. Reads are the price you pay — a key might live in the memtable or in any SSTable, so a read may have to look in several places. Space is the second price — until compaction runs, the same key can exist in multiple files. The genius of the LSM tree is that it takes the one operation hardware is best at — sequential writes — and builds everything else on top of it, then spends background CPU and I/O to keep reads and space under control. The rest of this article is a tour of how each of those mechanisms works, how to implement them, how they fail in production, and how to tune them.

> An LSM tree is a bet that you can always buy back read performance and disk space with background work, but you can never buy back the write throughput you lose to random in-place updates.

## Why B-trees hit the write wall

Before we go deeper, it is worth being precise about what problem the LSM tree solves, because if your workload is read-mostly, a B-tree is still the right answer and the LSM tree is pure downside. The mismatch most teams have is between what they assume about their storage engine and what it actually does under a write-heavy load.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "Writes are cheap; it's just an insert." | One row in, a little index maintenance. | A B-tree insert reads the leaf page, mutates it, may split it (cascading to parents), and writes whole pages back — random 4–16 KB writes per logical insert. |
| "My SSD does a million IOPS, I'm fine." | Throughput = IOPS × row size. | Random writes trigger flash garbage collection; sustained random write throughput is often 5–10× lower than sequential, and write amplification wears the drive out. |
| "Indexes make everything faster." | More indexes, more speed. | Every secondary B-tree index multiplies write amplification — each insert touches every index's random location. |
| "The database handles durability for free." | `fsync` is instant. | A B-tree may `fsync` per dirty page or rely on a separate WAL; either way durability costs a sync on the hot path. |
| "Compaction / vacuum is a tuning knob I can ignore." | Background stuff. | In LSM systems compaction *is* the write path's second half; misconfigure it and you get write stalls, disk-full, or read timeouts. |

The figure below makes the contrast physical. On the left is the B-tree: a logical insert lands at a random leaf, the page is rewritten in place, and an unlucky insert splits the page and rewrites its parents too. Every one of those is a random write, and on flash every random write is expensive. On the right is the LSM tree: the same insert is a sequential append to the log and an in-memory update; the only random-ish I/O happens later, in batched background compaction, where it is amortized across millions of keys.

![B-tree updates rewrite pages in place at random locations; LSM trees append sequentially and merge later](/imgs/blogs/lsm-trees-write-optimized-storage-engines-3.webp)

The numbers matter. Consider inserting 10 GB of 100-byte rows. A B-tree with a 16 KB page size and poor locality might rewrite a full page for nearly every insert, and with secondary indexes the **write amplification** — the ratio of bytes physically written to bytes logically inserted — routinely lands between 10× and 40×. That is 100–400 GB of physical writes for 10 GB of data, which both caps throughput and burns through the SSD's finite program-erase cycles. This is exactly the failure mode behind [random UUIDs killing database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance): a UUIDv4 primary key scatters inserts uniformly across the B-tree, maximizing page splits and cache misses.

An LSM tree's write amplification comes from a completely different place — compaction rewriting the same data as it moves down the levels — and as we will see, it is both lower for write-heavy workloads and, crucially, *tunable*. You get to choose how much write, read, and space amplification you are willing to pay. A B-tree gives you no such dial.

We can write the write-amplification intuition compactly. If $L$ is the number of levels and each level is a factor $T$ larger than the one above it, leveled compaction rewrites each byte roughly $O(T \cdot L)$ times over its lifetime, while size-tiered compaction rewrites it roughly $O(L)$ times. With $L \approx \log_T(N/M)$ for $N$ bytes of data and an $M$-byte memtable, both are logarithmic in the dataset size — not linear like a naive log, and not the random-write disaster of an in-place B-tree.

## 1. The write path: append now, sort later

Let us build the write path for real, in Python, because the mechanics are clearer in 40 lines of code than in three paragraphs. Everything in this article assembles into a working mini-LSM engine by the end; you can run it.

The first component is the **bloom filter**, because every SSTable carries one and we need it before we can write a table. We will return to *why* in its own section; for now treat it as a set membership test that can say "definitely not here" or "maybe here."

```python
import hashlib
import math


class BloomFilter:
    """Probabilistic set membership: no false negatives, tunable false positives."""

    def __init__(self, n_items: int, fp_rate: float = 0.01):
        n = max(1, n_items)
        self.m = max(8, int(-n * math.log(fp_rate) / (math.log(2) ** 2)))
        self.k = max(1, round((self.m / n) * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _positions(self, key: str):
        # Double hashing: derive k indices from one SHA-256 digest.
        digest = hashlib.sha256(key.encode()).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") | 1  # force odd
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, key: str) -> None:
        for pos in self._positions(key):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, key: str) -> bool:
        return all(self.bits[pos >> 3] & (1 << (pos & 7)) for pos in self._positions(key))
```

Next, the in-memory write buffer. The **memtable** must keep keys sorted so that flushing to an SSTable is a single sequential scan, and so that range scans are cheap. Production engines use a concurrent skiplist (RocksDB) or a lock-free B-tree; we use `sortedcontainers.SortedDict`, which is a sorted dictionary backed by a list of lists and is more than good enough to demonstrate the semantics.

```python
from sortedcontainers import SortedDict

TOMBSTONE = object()  # deleted marker: written as a tombstone (flag=1), shadows older values
_MISSING = object()   # "not present in this layer" — distinct from any stored value


class MemTable:
    def __init__(self):
        self.data = SortedDict()
        self.nbytes = 0  # approximate live size, drives the flush threshold

    def put(self, key: str, value) -> None:
        self.nbytes += len(key) + (len(value) if value is not TOMBSTONE else 0)
        self.data[key] = value

    def get(self, key: str, default=_MISSING):
        return self.data.get(key, default)
```

Now the durability half. The **write-ahead log** is an append-only file: every mutation is serialized and `fsync`-ed before the write is acknowledged, so a crash between the WAL append and the next flush loses nothing — on restart we replay the log back into a fresh memtable. The record format is deliberately trivial: a 9-byte header (`key length`, `op flag`, `value length`) followed by the raw key and value bytes.

```python
import os
import struct

_HEADER = struct.Struct(">IBI")  # key_len (u32), flag (u8), value_len (u32)


def encode_record(key: str, flag: int, value: bytes) -> bytes:
    kb = key.encode()
    return _HEADER.pack(len(kb), flag, len(value)) + kb + value
```

The thing to internalize from this section is that the entire hot path — `encode_record`, one buffered `write`, one `fsync`, one in-memory `SortedDict` insert — contains **zero seeks and zero read-modify-write cycles**. That is the whole point. The disk head (or the SSD's flash translation layer) only ever sees data arriving in the order it was produced. Compare that to the B-tree's "find the leaf, read it, mutate it, write it back, maybe split" dance and you understand the order-of-magnitude write-throughput gap.

The second-order consequence is **write stalls**, which we will meet again in the case studies. Because flushes and compactions are asynchronous, a sustained write burst can outrun them: memtables pile up waiting to flush, or L0 files pile up waiting to compact. Every LSM engine therefore has a back-pressure mechanism — RocksDB's `max_write_buffer_number` and `level0_slowdown_writes_trigger` — that deliberately throttles or pauses writers when the background work falls behind. The fast write path is fast precisely because it is allowed to write a check that compaction has to cash.

### Group commit: the durability dial

There is one more subtlety on the write path that separates a toy from a production engine: the `fsync`. A naive implementation calls `fsync` after every write, which is correct but slow — `fsync` on a datacenter SSD costs tens to hundreds of microseconds, and serializing every writer behind their own sync caps throughput at a few thousand writes per second regardless of how fast the CPU is. The fix is **group commit**: collect the WAL records of many concurrent writers into one buffer, do a single `fsync` for the whole batch, and acknowledge all of them together. One sync now amortizes across hundreds of writes.

```python
import threading


class GroupCommitWAL:
    def __init__(self, path: str):
        self.f = open(path, "ab")
        self.lock = threading.Lock()
        self.buffer = bytearray()
        self.pending = []  # threading.Event per waiting writer

    def append(self, record: bytes) -> None:
        done = threading.Event()
        with self.lock:
            self.buffer += record
            self.pending.append(done)
            is_leader = len(self.pending) == 1
        if is_leader:
            self._flush_group()       # one writer does the sync for the whole batch
        else:
            done.wait()               # followers ride along on the leader's fsync

    def _flush_group(self) -> None:
        with self.lock:
            data, waiters = bytes(self.buffer), self.pending
            self.buffer, self.pending = bytearray(), []
        self.f.write(data)
        self.f.flush()
        os.fsync(self.f.fileno())     # the single durability point for the whole group
        for w in waiters:
            w.set()
```

The durability *dial* sits right here. Synchronous `fsync` per group gives you the strongest guarantee — an acknowledged write survives a power loss — at the cost of latency bounded by sync time. Looser settings trade durability for speed: RocksDB's `WriteOptions.sync = false` buffers WAL writes in the OS page cache and survives a *process* crash but not a *power* loss; Cassandra's `commitlog_sync: periodic` syncs every 10 ms by default, accepting a 10 ms window of potential data loss for a large throughput gain. There is no universally right setting — a financial ledger wants synchronous group commit; a metrics firehose where losing the last 10 ms of points is irrelevant should absolutely use periodic sync. The mistake is not knowing which one you have configured.

## 2. The read path: why one lookup can touch many files

Here is the tax. In a B-tree, a point read is $O(\log n)$ and touches exactly one leaf. In an LSM tree, a key has no single home — the latest version might be in the memtable, or in a freshly frozen memtable, or in any SSTable that has not yet been compacted away. A naive implementation would check every one of them. That is **read amplification**, and taming it is what bloom filters, block caches, and leveled compaction are all secretly about.

![A single LSM lookup probes the memtable plus one SSTable per level, with bloom filters skipping most disk reads](/imgs/blogs/lsm-trees-write-optimized-storage-engines-2.webp)

Trace `get(key)` through the figure. The memtable is checked first because it holds the newest data — if the key is there, we are done with zero disk I/O. Otherwise the read fans out across the on-disk levels. For each candidate SSTable, the engine consults that table's **bloom filter** before touching the disk. If the bloom says "definitely not here," the file is skipped entirely — no disk read at all. If it says "maybe," the engine does a real read: binary-search the sparse index, seek to the right block, scan for the key. Whatever values are found across all the layers are reconciled by a single rule: **the newest version wins**, where "newest" is determined by sequence number or level position. A tombstone counts as a version, so a deleted-then-not-yet-compacted key correctly returns "not found."

Here is the read path in code, layering the memtable over a list of SSTables held newest-first:

```python
def lsm_get(memtable, sstables, key):
    # 1. Memtable: newest data, no disk I/O.
    v = memtable.get(key)
    if v is not _MISSING:
        return None if v is TOMBSTONE else v

    # 2. On-disk SSTables, newest first; stop at the first hit.
    for sst in sstables:            # sstables[0] is the most recent
        result = sst.get(key)       # consults the bloom filter internally
        if result is not _MISSING:
            return None if result is TOMBSTONE else result

    return None                     # truly absent everywhere
```

The critical detail is `return` on the **first** hit. Because we iterate newest-to-oldest, the first SSTable that contains the key holds its most recent version, so we never need to read the older copies. Without bloom filters this is still correct but slow: a point lookup for a non-existent key would probe every level. With well-tuned bloom filters, the expected number of *disk reads* for a point lookup drops to roughly one — the level that actually has the key — plus a small false-positive tax.

Quantitatively: read amplification for a point query is the number of SSTables actually opened. With leveled compaction and a 1% bloom false-positive rate across, say, 7 levels, a lookup for a missing key does on average $7 \times 0.01 = 0.07$ wasted disk probes — effectively free. A lookup for a present key does one real read at the level holding it. **Range scans** are the painful case: a bloom filter cannot help (it only answers point queries), so a scan must merge-iterate every overlapping SSTable across every level. This is exactly why leveled compaction, which guarantees non-overlapping key ranges within a level, exists — it bounds the number of files a range scan must touch.

The second-order consequence shows up with **deletes**. A delete is not a removal; it is a write of a tombstone. Until compaction reaches the bottom level and physically drops it, that tombstone sits there, and worse, a range scan over a region full of tombstones must read and skip every one of them. We will see in the case studies how a Cassandra cluster can be brought down by exactly this — millions of tombstones turning a cheap range scan into a multi-second, memory-exploding ordeal.

### Range scans and the merge iterator

A point lookup can stop at the first hit. A range scan cannot — it must produce *every* live key in `[start, end)` in sorted order, which means simultaneously walking the memtable and every overlapping SSTable, merging their sorted streams, and at each key emitting only the newest version. This is a k-way merge, classically implemented with a min-heap keyed on `(key, -source_age)` so that for equal keys the newest source is popped first.

```python
import heapq


def range_scan(memtable, sstables, start: str, end: str):
    """Yield (key, value) for live keys in [start, end), newest version wins."""
    # source_rank: lower = newer. memtable is newest (rank 0).
    streams = [(0, iter(sorted(memtable.data.items())))]
    for rank, sst in enumerate(sstables, start=1):
        streams.append((rank, ((k, TOMBSTONE if flag == DELETE else v)
                               for k, flag, v in sst.scan())))

    heap = []
    for rank, stream in streams:
        for key, value in stream:
            if key >= start:
                if key < end:
                    heapq.heappush(heap, (key, rank, value, stream))
                break

    last_key = None
    while heap:
        key, rank, value, stream = heapq.heappop(heap)
        if key != last_key:                       # first pop of a key = newest version
            last_key = key
            if value is not TOMBSTONE:
                yield key, value
        for nk, nflag_v in stream:                # advance this source
            nk_value = nflag_v
            if nk < end:
                heapq.heappush(heap, (nk, rank, nk_value, stream))
            break
```

The cost of a scan is therefore proportional to the number of overlapping sources it must merge, which is exactly the quantity leveled compaction minimizes by guaranteeing non-overlapping ranges within each level. On a leveled store a scan touches at most one file per level plus the L0 files; on a size-tiered store it may touch every overlapping table of every size. This is the single most important reason a range-scan-heavy workload — a queue, a feed, anything with `ORDER BY` and `LIMIT` — should prefer leveled compaction, and why bloom filters, which do nothing for scans, are not the answer there.

## 3. SSTables: the immutable file at the heart of it all

Everything on disk is an SSTable, so it pays to understand its anatomy precisely. An SSTable is a single immutable file containing sorted key-value pairs plus the metadata needed to search it without reading the whole thing. The figure shows the regions in the order they appear on disk.

![An SSTable is a self-describing immutable file: sorted data blocks, then a sparse index, a bloom filter, and a footer](/imgs/blogs/lsm-trees-write-optimized-storage-engines-4.webp)

The **data blocks** hold the actual key-value records, sorted by key, grouped into roughly 4 KB blocks so that each block is a natural unit of compression and of the block cache. The **sparse index** stores one entry per block — the first key and its byte offset — so the engine can binary-search the index in memory and then seek directly to the one block that might hold the target key. It is *sparse* (one entry per block, not per key) precisely to keep the index small enough to cache. The **bloom filter** block lets a reader rule the entire file out before any of this. The **footer** is a fixed-size trailer at the very end with the offsets of the index and bloom blocks plus a magic number, so a reader opens the file by seeking to the end first, reading the footer, then jumping straight to the metadata.

Immutability is the property that makes all of this safe and cheap. Because an SSTable never changes after it is written, it needs no locking for reads, it can be memory-mapped and shared across threads, its blocks can be cached without invalidation logic, and it can be copied or replicated byte-for-byte. The cost of immutability is that updates and deletes cannot modify it — they can only write *new* SSTables that shadow it — which is the source of both space amplification and the need for compaction.

Here is a writer that serializes a sorted list of records into exactly this layout, building the sparse index and bloom filter as it goes:

```python
import pickle

PUT, DELETE = 0, 1
_FOOTER = struct.Struct(">QQ")  # index_offset, bloom_offset
_MAGIC = b"LSMT"


class SSTableWriter:
    @staticmethod
    def write(path: str, items, index_interval: int = 16) -> None:
        """items: sorted iterable of (key:str, flag:int, value:bytes)."""
        items = list(items)
        bloom = BloomFilter(len(items))
        index = []  # sparse: (first_key_of_block, byte_offset)

        with open(path, "wb") as f:
            for i, (key, flag, value) in enumerate(items):
                if i % index_interval == 0:
                    index.append((key, f.tell()))
                bloom.add(key)
                f.write(encode_record(key, flag, value))

            index_offset = f.tell()
            f.write(pickle.dumps(index))
            bloom_offset = f.tell()
            f.write(pickle.dumps((bloom.m, bloom.k, bytes(bloom.bits))))
            f.write(_FOOTER.pack(index_offset, bloom_offset))
            f.write(_MAGIC)
```

And the reader, which opens from the footer, loads the metadata, and uses the bloom filter and sparse index to do the minimum possible I/O for a point lookup:

```python
import bisect


class SSTable:
    def __init__(self, path: str):
        self.path = path
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(size - _FOOTER.size - len(_MAGIC))
            index_offset, bloom_offset = _FOOTER.unpack(f.read(_FOOTER.size))
            assert f.read(len(_MAGIC)) == _MAGIC, "bad SSTable magic"
            f.seek(index_offset)
            self.index = pickle.loads(f.read(bloom_offset - index_offset))
            f.seek(bloom_offset)
            m, k, raw = pickle.loads(f.read(size - _FOOTER.size - len(_MAGIC) - bloom_offset))
        self.bloom = BloomFilter.__new__(BloomFilter)
        self.bloom.m, self.bloom.k, self.bloom.bits = m, k, bytearray(raw)
        self._index_offset = index_offset
        self._keys = [k for k, _ in self.index]

    def get(self, key: str):
        if key not in self.bloom:
            return _MISSING                      # bloom: definitely absent, no disk read
        pos = bisect.bisect_right(self._keys, key) - 1
        if pos < 0:
            return _MISSING
        with open(self.path, "rb") as f:
            f.seek(self.index[pos][1])           # jump to the candidate block
            while f.tell() < self._index_offset:
                header = f.read(_HEADER.size)
                klen, flag, vlen = _HEADER.unpack(header)
                k = f.read(klen).decode()
                v = f.read(vlen)
                if k == key:
                    return TOMBSTONE if flag == DELETE else v
                if k > key:
                    break                        # sorted: we've passed it
        return _MISSING

    def scan(self):
        """Yield every (key, flag, value) in sorted order — used by compaction."""
        with open(self.path, "rb") as f:
            while f.tell() < self._index_offset:
                klen, flag, vlen = _HEADER.unpack(f.read(_HEADER.size))
                yield f.read(klen).decode(), flag, f.read(vlen)
```

Notice the layered defense in `get`: the bloom filter avoids opening files that cannot contain the key; the sparse index avoids scanning blocks that cannot contain it; the in-block sorted scan stops the instant it passes the key. Each layer cuts the work by an order of magnitude.

### Compression and the block cache

Two production features turn this skeleton into something competitive: block compression and the block cache. Because data blocks are sorted, adjacent keys share long common prefixes, so block-level compression (LZ4 for speed, Zstandard for ratio) routinely shrinks SSTables by 2–4×. The unit of compression is the ~4 KB block precisely so that a read decompresses one block, not the whole file. The **block cache** then keeps *decompressed* hot blocks in RAM, so a repeated read of a hot key never touches disk or the decompressor at all.

```python
opts.set_compression(DBCompressionType.lz4())              # fast path for the small upper levels
opts.set_bottommost_compression(DBCompressionType.zstd())  # best ratio for the huge bottom level
```

This "LZ4 on top, Zstandard on the bottom" split is a recurring LSM tuning pattern, and it falls straight out of the level structure: the shallow levels are small, hot, and rewritten constantly, so they want cheap compression; the bottom level is enormous, cold per byte, and rewritten rarely, so it can afford expensive compression for the disk savings. A B-tree, whose pages are half-full and updated in place, gets neither the prefix-compression benefit nor this hot/cold split — another quiet reason the LSM tree packs the same data into roughly half the space.

## 4. Bloom filters: skipping the files you don't need

The bloom filter deserves its own section because it is the single component that makes LSM reads tolerable, and because most production read-latency incidents trace back to a misconfigured one. A bloom filter is a bit array plus $k$ hash functions. To insert a key, hash it $k$ ways and set those $k$ bits. To test a key, hash it $k$ ways and check those bits: if **any** is zero, the key was definitely never inserted; if **all** are one, the key is *probably* present — "probably" because other keys may have collectively set those same bits.

![A bloom filter sets k bits on insert; any zero bit on lookup proves the key is absent and the SSTable read is skipped](/imgs/blogs/lsm-trees-write-optimized-storage-engines-5.webp)

The figure shows the asymmetry that makes it useful. Inserting `user:42` sets three bits. A later lookup that finds **any** probed bit still zero — like the lookup for `log:88` hitting bit 11 — has *proven* the key is absent and can skip the SSTable read entirely. There are no false negatives, ever, which is exactly the guarantee an LSM read path needs: it can use the bloom to safely *skip* files, and the worst case for a false positive is one wasted disk read, never a wrong answer.

The math is worth knowing because it is the tuning knob. For a filter with $m$ bits, $k$ hash functions, and $n$ inserted keys, the false-positive probability after all insertions is approximately

$$p \approx \left(1 - e^{-kn/m}\right)^{k}.$$

The optimal number of hash functions for a target $m/n$ is $k = (m/n)\ln 2$, which gives a false-positive rate of roughly $p \approx 0.6185^{m/n}$. The practical takeaway: **bits per key is the dial.** Ten bits per key gives about a 1% false-positive rate; the cost is 10 bits of RAM per key, which for a billion keys is 1.25 GB. That is the trade you are making every time you size a bloom filter — memory for read amplification.

The non-obvious insight, formalized in the *Monkey* paper (Dayan, Athanassoulis, Idreos, 2017), is that **a uniform bits-per-key allocation across all levels is wrong.** Most of an LSM tree's data lives in the largest, deepest level, but most *false-positive cost* is paid there too, because that is where the most files are probed. Monkey showed that allocating *more* bits per key to the smaller, shallower levels and *fewer* to the huge bottom level minimizes total false positives for a fixed memory budget — often cutting point-lookup I/O by 50–80% at the same RAM. RocksDB later adopted exactly this idea. When a colleague tells you "we added more RAM but reads didn't get faster," the bloom filter allocation is the first place to look.

The second-order failure mode: bloom filters only help **point lookups**. A range scan (`WHERE k BETWEEN a AND b`) cannot use them, because the filter has no notion of ranges. If your workload is scan-heavy, bloom filters are wasted RAM, and your read performance depends entirely on compaction keeping the number of overlapping files small. This is the deepest reason the choice of compaction strategy matters so much — which is where we go next.

## 5. Compaction: the engine's heartbeat

Compaction is where LSM trees earn their reputation for being hard to operate. It is the background process that merges SSTables, discards shadowed and deleted keys, and keeps both the file count and the read amplification bounded. It is also, by far, the largest consumer of disk I/O and CPU in a healthy LSM database, and the source of nearly every operational surprise. The single most important configuration decision you will make about an LSM store is its **compaction strategy**, because that single choice sets where you land on the read/write/space trade-off.

### Leveled compaction

Leveled compaction (LCS), the default in RocksDB and the original LevelDB design, organizes SSTables into levels $L_0, L_1, \dots, L_n$ where each level is a fixed multiple (typically 10×) larger than the one above it. The defining invariant: **within any level except $L_0$, the SSTables have non-overlapping key ranges.** That means a given key appears in at most one file per level, so a point read examines at most one file per level and a range scan touches a bounded number of files.

![Leveled compaction keeps each level about ten times the previous, with non-overlapping key ranges below L0](/imgs/blogs/lsm-trees-write-optimized-storage-engines-6.webp)

$L_0$ is special: it receives whole memtable flushes, so its files *do* overlap (each is an independent snapshot of recent writes). When $L_0$ accumulates enough files, compaction picks one and merges it into the overlapping key range of $L_1$, rewriting that slice of $L_1$. When $L_1$ exceeds its size budget, one of its files is merged down into $L_2$, and so on. The "merge a file down into the overlapping range below" motion is the heartbeat in the figure.

The cost is write amplification. Because each level is ~10× larger, merging a file from $L_i$ into $L_{i+1}$ reads and rewrites roughly 10× the file's size (the overlapping range in the bigger level). Over a key's lifetime, as it migrates from $L_0$ down to $L_n$, it is rewritten on the order of $T \times L$ times — for $T = 10$ and 5 levels, that is potentially 50× write amplification in the worst case, though typical workloads see 10–30×. In exchange you get the tightest possible read and space amplification: at most one file per level holds any key, and obsolete versions are aggressively removed, so on-disk space is close to the live data size.

### Size-tiered compaction

Size-tiered compaction (STCS), the historical default in Cassandra, makes the opposite trade. Instead of strict levels, it groups SSTables into *tiers* by size and, when enough similarly-sized tables accumulate (default: four), merges them into a single larger table. There is no non-overlapping invariant; tables of similar size simply pile up and occasionally fuse.

![Size-tiered compaction minimizes write cost; leveled compaction minimizes read and space cost — you choose which to pay](/imgs/blogs/lsm-trees-write-optimized-storage-engines-7.webp)

The figure puts the two side by side. Size-tiered's write amplification is low — each byte is rewritten only $O(L)$ times, once per tier it passes through, because a merge produces a table roughly 4× larger and there are only logarithmically many such steps. But because overlapping tables of every size coexist, a point read may have to check many of them (high read amplification, mitigated by bloom filters), and — the killer — a major compaction that merges the largest tables needs **temporary free space equal to the size of the data being merged.** Size-tiered compaction is notorious for needing up to 50% of the disk free to safely compact, because at the moment of merging the biggest tier, both the inputs and the output exist on disk simultaneously.

### Time-window and universal compaction

Two specialized strategies round out the menu. **Time-window compaction** (TWCS) is built for time-series data with a TTL: it buckets SSTables by time window and only ever compacts within a window. The payoff is enormous for append-only, time-ordered data, because an entire expired window can be dropped as whole files — no merge, no tombstone scanning, just `unlink`. Using leveled or size-tiered compaction on time-series data is a classic, expensive mistake (see the case studies). **Universal compaction** (RocksDB's name for an STCS-like strategy) optimizes for write throughput and bounded space amplification, accepting higher read amplification; it is the right pick for write-dominated workloads like a Kafka-style log or an ingestion buffer.

Here is a simple full compaction — a k-way merge of every SSTable, newest-wins, dropping tombstones because at the bottom level there is nothing older left to shadow:

```python
def compact(sstables, output_path, drop_tombstones=True):
    """Merge all SSTables into one. sstables is newest-first."""
    merged = SortedDict()
    # Iterate OLDEST to NEWEST so newer versions overwrite older ones.
    for sst in reversed(sstables):
        for key, flag, value in sst.scan():
            merged[key] = (flag, value)

    items = [
        (key, flag, value)
        for key, (flag, value) in merged.items()
        if not (drop_tombstones and flag == DELETE)
    ]
    SSTableWriter.write(output_path, items)
    return SSTable(output_path)
```

Real engines never do a single global merge like this — they compact incrementally, file by file, to bound the I/O burst and keep the database writable throughout. But the semantics are identical: merge sorted inputs, keep the newest version of each key, drop tombstones once nothing older can be shadowed.

### Subcompactions: parallelizing the heartbeat

A single compaction thread can become the bottleneck on a fast NVMe drive — one CPU core merging and one I/O stream cannot saturate a device capable of gigabytes per second. Modern engines split a large compaction into **subcompactions**: the key range being compacted is partitioned into disjoint sub-ranges, and each is merged by a separate thread writing a separate output file. Because the input ranges are disjoint and the outputs are non-overlapping, the work parallelizes cleanly with no coordination beyond picking the split points.

```python
opts.set_max_background_compactions(8)   # how many compactions may run concurrently
opts.set_max_subcompactions(4)           # split one large compaction across 4 threads
opts.set_max_background_jobs(12)          # total background threads (flush + compaction)
```

The tuning tension is real: too few compaction threads and the database stalls under write pressure (case study 2); too many and compaction steals I/O bandwidth and CPU from foreground reads and writes, spiking tail latency. The right setting is the smallest amount of background parallelism that keeps the L0 file count and the pending-compaction-bytes metric flat under your peak write rate — which you find by watching those two metrics under load, not by guessing. On a shared-nothing engine like ScyllaDB, this is taken further: a dedicated I/O scheduler gives compaction a strict, lower priority than user queries, so compaction can never starve a foreground read no matter how far behind it falls.

### Choosing a strategy

There is no universally correct compaction strategy; there is only the one that matches your workload's shape. The matrix below is the decision table I keep in my head.

![The right compaction strategy is a function of the workload's read/write/TTL shape, not the data size](/imgs/blogs/lsm-trees-write-optimized-storage-engines-9.webp)

| Strategy | Write amp | Read amp | Space amp | Best for |
| --- | --- | --- | --- | --- |
| Size-tiered (STCS) | Low | High | High (≈2× transient) | Write-heavy, few reads, space to spare |
| Leveled (LCS) | High | Low | Low (≈1.1×) | Read-heavy, range scans, space-constrained |
| Time-window (TWCS) | Low | Medium | Low (drops whole windows) | Time-series with TTL, append-only |
| Universal | Low | Medium-High | Bounded by config | Write-dominated, ingestion buffers |

In CQL, choosing is a one-line table property, and getting it right for the workload is one of the highest-leverage decisions in operating Cassandra or ScyllaDB:

```sql
-- Read-heavy table: pay write amplification to keep reads and space tight.
ALTER TABLE app.user_profiles
  WITH compaction = {'class': 'LeveledCompactionStrategy', 'sstable_size_in_mb': 160};

-- Time-series with a 30-day TTL: bucket by day, drop expired windows whole.
ALTER TABLE app.sensor_readings
  WITH compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'DAYS',
    'compaction_window_size': 1
  }
  AND default_time_to_live = 2592000;
```

## 6. The RUM conjecture: you only get to pick two

Step back from the specifics and there is a single law governing all of this, named the **RUM conjecture** by Athanassoulis et al. (2016): for any access method, you can optimize for at most two of **R**ead overhead, **U**pdate (write) overhead, and **M**emory (space) overhead — improving one of the three forces you to give ground on another. Every storage engine, B-tree or LSM, is a point inside this triangle.

![Read, write, and space amplification form a trade-off triangle, and each compaction strategy is a point inside it](/imgs/blogs/lsm-trees-write-optimized-storage-engines-8.webp)

The figure plots the LSM compaction strategies inside the triangle. Leveled compaction sits near the read-and-space-optimized corner: it minimizes read amplification (one file per level) and space amplification (aggressive merging), and pays in write amplification. Size-tiered and universal sit near the write-optimized corner: low write amplification, at the cost of more read work and more transient space. Time-window compaction earns a privileged spot near the space-optimized corner for its specific workload, because dropping whole expired windows is nearly free on all three axes when the data is append-only with a TTL.

This is why the question "which is the best LSM compaction strategy?" has no answer — it is exactly as malformed as "which is the best point inside a triangle?" The right question is "where in the triangle does my workload need to be?" A metrics ingestion pipeline that writes a million points per second and queries recent windows wants the write-optimized corner. A user-facing profile store with strict latency SLAs and range scans wants the read-optimized corner. A B-tree, for comparison, lives at the read-optimized corner permanently with no dial at all — which is exactly why it is the wrong tool for the write-optimized corner and the right tool when you genuinely live in the read-optimized one.

The practical discipline the RUM conjecture imposes: **measure all three amplifications, not just the one you care about.** A team that tunes for write throughput and forgets to watch space amplification fills the disk; a team that tunes for read latency and forgets write amplification wears out the SSD. The triangle is not a metaphor — it is a budget, and you are spending on all three axes whether you watch them or not.

## 7. Putting it together: a working LSM engine

We now have every component. Here is the orchestrator that ties the memtable, WAL, SSTables, and compaction into a single durable, crash-recoverable key-value store. It is small enough to read in full and real enough to run.

```python
import glob


class LSMTree:
    def __init__(self, directory: str, memtable_limit: int = 4 * 1024 * 1024):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)
        self.memtable_limit = memtable_limit
        self.memtable = MemTable()
        self.sstables = []  # newest first
        self.wal_path = os.path.join(directory, "wal.log")
        self.seq = 0
        self._recover()
        self.wal = open(self.wal_path, "ab")

    def _recover(self) -> None:
        # Load existing SSTables; filenames sort by sequence so newest is last.
        for path in sorted(glob.glob(os.path.join(self.dir, "sst-*.sst"))):
            self.seq = max(self.seq, int(os.path.basename(path)[4:10]))
            self.sstables.insert(0, SSTable(path))  # prepend → newest first
        # Replay any WAL the last flush didn't clear.
        if os.path.exists(self.wal_path):
            with open(self.wal_path, "rb") as f:
                while header := f.read(_HEADER.size):
                    klen, flag, vlen = _HEADER.unpack(header)
                    key, value = f.read(klen).decode(), f.read(vlen)
                    self.memtable.put(key, TOMBSTONE if flag == DELETE else value)

    def _wal_append(self, key: str, flag: int, value: bytes) -> None:
        self.wal.write(encode_record(key, flag, value))
        self.wal.flush()
        os.fsync(self.wal.fileno())  # durability point — the write is now safe

    def put(self, key: str, value: bytes) -> None:
        self._wal_append(key, PUT, value)
        self.memtable.put(key, value)
        if self.memtable.nbytes >= self.memtable_limit:
            self.flush()

    def delete(self, key: str) -> None:
        self._wal_append(key, DELETE, b"")
        self.memtable.put(key, TOMBSTONE)
        if self.memtable.nbytes >= self.memtable_limit:
            self.flush()

    def get(self, key: str):
        v = self.memtable.get(key)
        if v is not _MISSING:
            return None if v is TOMBSTONE else v
        for sst in self.sstables:
            r = sst.get(key)
            if r is not _MISSING:
                return None if r is TOMBSTONE else r
        return None

    def flush(self) -> None:
        if not self.memtable.data:
            return
        items = [
            (k, (DELETE if v is TOMBSTONE else PUT), (b"" if v is TOMBSTONE else v))
            for k, v in self.memtable.data.items()
        ]
        self.seq += 1
        path = os.path.join(self.dir, f"sst-{self.seq:06d}.sst")
        SSTableWriter.write(path, items)
        self.sstables.insert(0, SSTable(path))
        self.memtable = MemTable()
        self.wal.close()
        open(self.wal_path, "w").close()  # truncate: data is now durable on disk
        self.wal = open(self.wal_path, "ab")

    def compact(self) -> None:
        if len(self.sstables) < 2:
            return
        self.seq += 1
        path = os.path.join(self.dir, f"sst-{self.seq:06d}.sst")
        merged = compact(self.sstables, path)
        old = self.sstables
        self.sstables = [merged]
        for sst in old:
            os.remove(sst.path)
```

And the demonstration that it all works — including the crucial property that a delete shadows an older flushed value, and that compaction physically removes it:

```python
db = LSMTree("/tmp/lsm-demo", memtable_limit=64)  # tiny limit to force flushes
db.put("user:1", b"alice")
db.put("user:2", b"bob")
db.flush()                      # both land in an SSTable
db.delete("user:1")            # tombstone in the memtable, shadows the SSTable value
db.put("user:3", b"carol")

assert db.get("user:1") is None     # deleted, even though the SSTable still has "alice"
assert db.get("user:2") == b"bob"   # served from the SSTable
assert db.get("user:3") == b"carol" # served from the memtable

db.flush()
db.compact()                    # merges everything, drops the tombstone + shadowed "alice"
assert db.get("user:1") is None
assert db.get("user:2") == b"bob"
print("LSM engine: all assertions passed")
```

This 200-line engine has, in miniature, the real architecture: durable appends, in-memory sorted buffering, immutable sorted on-disk files, bloom-filtered reads, newest-wins reconciliation, tombstone deletes, and merge compaction. What production engines add is concurrency (lock-free memtables, parallel compaction threads), leveling (the simple `sstables` list becomes $L_0 \dots L_n$ with size budgets), block caching and compression, manifest files for atomic metadata updates, and a hundred tuning knobs. But none of that changes the shape of the thing.

## 8. Tuning real engines

Knowing the theory, the operational reality of running RocksDB or Cassandra comes down to a handful of knobs that directly control where you sit in the RUM triangle and how the write path back-pressures. Here are the ones that actually move the needle.

For RocksDB (via the `rocksdict` Python binding, which mirrors the C++ options one-to-one):

```python
from rocksdict import Rdict, Options

opts = Options()
opts.create_if_missing(True)

opts.set_write_buffer_size(128 * 1024 * 1024)   # write path: bigger memtable, fewer/larger flushes
opts.set_max_write_buffer_number(4)             # memtables allowed to queue before a write stall

opts.set_max_bytes_for_level_base(512 * 1024 * 1024)  # compaction: L1 target size (the RUM dial)
opts.set_max_bytes_for_level_multiplier(10)           # each level 10x the previous
opts.set_level0_file_num_compaction_trigger(4)        # begin L0 -> L1 compaction at 4 files
opts.set_level0_slowdown_writes_trigger(20)           # throttle writers when L0 backs up
opts.set_level0_stop_writes_trigger(36)               # hard stop: full back-pressure

opts.set_bloom_locality(1)                      # read path: cluster bloom bits for cache locality

db = Rdict("/data/mydb", opts)
```

A 10-bits-per-key bloom filter (≈ 1% false positives), configured on the block-based table factory, is typically the single biggest read-latency lever — more impactful than block cache size for point-lookup-heavy workloads, because it is the only thing that prevents disk reads for keys that are not present.

And `db_bench`, RocksDB's built-in benchmark, is the fastest way to see a compaction strategy's amplification on your own hardware before you commit to it in production:

Fill 100M keys, then print the stats. The `--compaction_style` flag selects the strategy: `0` is leveled, `1` is universal, `2` is FIFO.

```bash
./db_bench \
  --benchmarks=fillrandom,stats \
  --num=100000000 \
  --compaction_style=0 \
  --write_buffer_size=134217728 \
  --max_bytes_for_level_base=536870912 \
  --statistics \
  --report_file=bench.csv
```

In the output, the "Cumulative compaction" section reports write amplification directly, and the "Stall" counters reveal whether compaction is keeping up with ingest — non-zero stall time means flushes are outrunning compaction and the back-pressure has engaged.

The discipline here is to **read the stats, not the docs.** RocksDB's `db.get_property("rocksdb.stats")` and Cassandra's `nodetool tablestats` both report the live amplifications and the per-level file counts. If write amplification is climbing, your levels are too aggressive or your writes too bursty. If L0 file count is pinned at the stop trigger, flushes are outrunning compaction and you are stalling. If space amplification is near 2×, a size-tiered major compaction is about to need half your disk. Every one of these is visible *before* it becomes an outage, if you are looking.

## 9. Crash recovery, the manifest, and atomic swaps

The write path is durable because of the WAL, but durability of a single write is not the same as consistency of the whole on-disk structure across a crash. Compaction is constantly creating new SSTables and deleting old ones; a crash in the middle must not leave the database referencing a half-written file or, worse, having deleted an input before its replacement was durable. The mechanism that makes this safe is the **manifest**: a separate, append-only log that records the authoritative set of live SSTables as a sequence of versioned edits. The actual data files on disk are never trusted directly — the manifest is the source of truth for which of them are part of the current database.

The recovery and compaction protocols follow from this. On startup, the engine reads the manifest to learn the live SSTable set, then replays any WAL records the last flush did not clear back into a fresh memtable. A compaction, in turn, is crash-safe because it never mutates anything in place:

```python
def install_compaction(manifest, inputs, output_path, items):
    # 1. Write the new SSTable to a fresh file and make it durable.
    SSTableWriter.write(output_path, items)
    os.fsync(os.open(output_path, os.O_RDONLY))

    # 2. Atomically record the swap: add the new file, drop the inputs.
    manifest.append_edit(add=[output_path], remove=[s.path for s in inputs])
    manifest.fsync()   # the commit point — after this the swap is official

    # 3. Only now delete the input files; a crash before here just re-runs compaction.
    for s in inputs:
        os.remove(s.path)
```

The ordering is the whole trick. The new file is written and synced *before* the manifest edit; the manifest edit is the single atomic commit point; the old files are deleted *only after* the manifest no longer references them. A crash at any point is recoverable: before step 2, recovery simply ignores the orphan output and re-runs the compaction; after step 2, recovery sees the new file as live and the inputs as gone, and the orphaned input files (if step 3 was interrupted) are garbage-collected on startup. This is the same write-ahead discipline as the WAL, applied to metadata instead of data — and it is why an LSM database can be killed with `kill -9` mid-compaction and come back clean. The immutability of SSTables is what makes it possible: because no file is ever modified, "atomically swap which files are live" is the only consistency problem there is.

## 10. Secondary indexes and the write-amplification multiplier

So far we have treated the database as a single key-value keyspace, but real tables have secondary indexes, and in an LSM store each secondary index is its own complete LSM keyspace — its own memtables, SSTables, bloom filters, and compaction. That has a direct and often-underestimated consequence: **a single row write fans out into one write per index plus one for the primary.** A table with four secondary indexes turns one logical insert into five LSM writes, each with its own compaction-driven write amplification downstream.

| Indexes on table | LSM writes per row insert | Relative compaction load |
| --- | --- | --- |
| 0 (primary only) | 1 | 1× |
| 2 | 3 | 3× |
| 4 | 5 | 5× |
| 8 | 9 | 9× |

This is not unique to LSM trees — a B-tree pays the same fan-out, and in fact pays it *worse*, because each index update is a random in-place page write rather than a sequential append. But the LSM multiplier interacts with compaction in a way that surprises people: the index keyspaces compact independently, so a write-heavy table with many indexes can have its compaction budget dominated by index maintenance, not primary-data maintenance, and a write stall can originate in an index nobody was thinking about. The practical disciplines are the ones you would expect once you see the multiplier: index only the columns you actually query, prefer a single composite index over several single-column ones where the query pattern allows, and use **covering indexes** (storing the queried columns in the index itself) to convert two lookups into one — trading a little more index write amplification for eliminating the primary-key read entirely. The meta-point is that in an LSM tree, every index is a tax paid on every write forever, collected by the compactor; add them deliberately, not reflexively.

## Case studies from production

The theory is clean. Production is where LSM trees teach you humility. These are the failure modes that recur across teams and companies — the wrong first hypothesis, the actual root cause, the fix, and the lesson.

### 1. The Cassandra node that filled its disk during compaction

A team running Cassandra with the default size-tiered strategy watched a node hit 72% disk usage and then, during a routine major compaction, abruptly run out of space and crash — taking a replica offline and triggering a cascading repair storm. The first hypothesis was a disk leak or a runaway snapshot. The root cause was structural: size-tiered compaction merging the largest tier needs temporary free space roughly equal to the size of the data being merged, because the input SSTables and the output SSTable coexist until the merge completes. At 72% full, there was not enough room for the output, and the compaction died mid-write. The fix was twofold: keep STCS nodes under ~50% disk utilization as a hard operational rule, and migrate the largest tables to leveled compaction, which compacts incrementally and never needs a giant transient buffer. The lesson: with size-tiered compaction, **your usable disk is half your disk**, and the other half is compaction headroom you do not get to use.

### 2. RocksDB write stalls under a sustained ingest burst

A streaming pipeline writing into RocksDB saw p99 write latency spike from 2 ms to 1,800 ms under load, with throughput collapsing to a fraction of nominal. CPU and disk had headroom; the writes were simply *pausing*. The first hypothesis was lock contention. The actual cause was the L0 back-pressure mechanism doing exactly its job: flushes were producing L0 files faster than compaction could merge them into L1, the L0 file count hit `level0_slowdown_writes_trigger`, and RocksDB deliberately throttled the writers — then hit `level0_stop_writes_trigger` and paused them entirely. The fix was to give compaction more resources (`max_background_compactions`), enlarge `max_bytes_for_level_base` so L0→L1 compactions did more useful work per run, and add more write-buffer slots so short bursts could be absorbed in memory. The lesson: in an LSM tree, a write stall is not a bug, it is the engine refusing to let the write path outrun the compaction it depends on. The fix is always to make compaction faster, never to disable the back-pressure.

### 3. Discord's migration from Cassandra to ScyllaDB

Discord ran one of the largest Cassandra deployments in the world for their message store, and hit a wall not in the LSM design but in its implementation. Cassandra runs on the JVM, and at their scale the combination of garbage-collection pauses and compaction I/O produced tail latencies — p99 read spikes into the hundreds of milliseconds — that no amount of tuning fully eliminated. "Hot partitions" of popular channels made it worse. They migrated to ScyllaDB, which implements the same Cassandra data model and the same LSM/compaction design but in C++ on the Seastar shared-nothing, thread-per-core framework with its own memory management and I/O scheduler. The same compaction strategies, with no JVM and a compaction-aware scheduler, flattened the tail. The lesson: the LSM *design* and its *implementation* are separable, and at extreme scale the implementation's resource scheduling — GC behavior, I/O prioritization, cache management — can matter more than the algorithm. The bottleneck was never the LSM tree; it was everything around it.

### 4. The tombstone read-timeout that looked like a slow query

A Cassandra-backed queue table started throwing `TombstoneOverwhelmingException` and read timeouts on queries that had been instant for months. The first hypothesis was a missing index. The real cause was the access pattern: the application enqueued and deleted rows in a narrow key range, and every delete wrote a tombstone that lived until compaction and `gc_grace_seconds` (default 10 days) allowed its removal. A range scan over the "head" of the queue had to read and skip *millions* of tombstones to find a handful of live rows, blowing through the tombstone warning and failure thresholds and ballooning query memory. The fix combined several moves: switch the table to leveled compaction so tombstones were removed faster, lower `gc_grace_seconds` for this table after confirming repair cadence, and — the real fix — redesign the access pattern to avoid scanning across deleted ranges (a time-bucketed key so consumed buckets could be dropped whole). The lesson: in an LSM tree a delete is a write, and a workload that deletes inside a scanned range is quietly accumulating a landmine that detonates when the tombstones outnumber the live data.

### 5. Time-series data on leveled compaction

An observability team stored high-cardinality metrics in a Cassandra table with the default compaction, and watched write amplification climb past 30× and disk I/O saturate, even though the data was append-only and aged out after two weeks. The first hypothesis was too many writes. The root cause was a compaction-strategy mismatch: leveled compaction kept rewriting append-only, time-ordered data down through the levels to maintain its non-overlapping invariant — work that bought nothing, because the data was never updated and would simply expire. Switching to `TimeWindowCompactionStrategy` with a one-day window collapsed write amplification to near 1×: each day's data compacts only within its window and, on expiry, the entire window's SSTables are dropped with an `unlink` — no merge, no tombstone scan, no rewrite. The lesson: for append-only data with a TTL, the right compaction strategy turns deletion from an expensive merge into a free file removal. Using a general-purpose strategy on time-series data is paying for work the data shape makes unnecessary.

### 6. Meta's MyRocks: trading a B-tree for an LSM tree at scale

Meta ran the user database for Facebook on MySQL with the InnoDB storage engine — a classic B-tree. As the data outgrew RAM and moved to flash, InnoDB's write amplification and space overhead became a hardware cost problem: B-tree pages are typically only ~½ to ⅔ full, and in-place updates plus the doublewrite buffer multiplied flash writes. They built MyRocks — MySQL on top of RocksDB — and migrated. The result was roughly **2× less space** (LSM SSTables are densely packed and compressed, versus half-full B-tree pages) and substantially **lower write amplification**, which directly extended SSD lifetime and cut the fleet's storage cost. Reads stayed competitive thanks to bloom filters and a large block cache. The lesson: the LSM-versus-B-tree choice is not academic. At fleet scale, the LSM tree's denser packing and lower write amplification translate into fewer servers and longer-lived flash — a balance-sheet difference, not a benchmark difference.

### 7. CockroachDB's rewrite from RocksDB to Pebble

CockroachDB was built on RocksDB, a mature and excellent C++ LSM engine. Over time, the team hit friction not with the LSM design but with the integration: every call from CockroachDB's Go code into RocksDB crossed the cgo boundary, which adds per-call overhead and complicates memory management, profiling, and garbage-collection interaction. They wrote **Pebble**, a from-scratch LSM engine in pure Go, bit-for-bit compatible with RocksDB's on-disk format so existing clusters could migrate in place. Pebble eliminated the cgo tax, gave them full control over compaction scheduling and Go-native memory behavior, and let them tailor the engine to CockroachDB's exact access patterns. The lesson: the LSM tree is an architecture, not a library — a sufficiently large user will eventually want to own the implementation to control its scheduling, its memory model, and its integration with the host runtime. The on-disk format is stable enough that a reimplementation can be a drop-in.

### 8. The under-provisioned bloom filter

A team complained that adding RAM to their RocksDB nodes did nothing for read latency, which made no sense to them — surely more cache is always better. Profiling showed point lookups doing two to three disk reads each, far more than the "roughly one" the theory promises. The cause was a bloom filter configured at 4 bits per key (about a 10% false-positive rate) to save memory, so one in ten point lookups probed a level the key was not in and ate a wasted disk read. The extra RAM had gone to the block cache, which does nothing for *negative* lookups. Raising the bloom filter to 10 bits per key (≈1% false positives) — and, following the Monkey insight, allocating more bits to the shallow levels — cut the false-positive disk reads by an order of magnitude. The lesson: for point-lookup-heavy workloads, bits-per-key on the bloom filter is often a higher-leverage use of RAM than the block cache, because the bloom filter is the only thing that prevents disk reads for keys that *are not there*.

### 9. Universal compaction and the 2× space surprise

A write-heavy service chose RocksDB universal compaction for its low write amplification and high ingest throughput, and it delivered — until the disk hit 95% on a node and the database went read-only. The first hypothesis was a metrics bug. The root cause was inherent to the strategy: universal (size-tiered-style) compaction allows space amplification up to a configured bound — by default it can let the database grow to roughly 2× the live data size before forcing a full compaction, because large overlapping SSTables of obsolete data coexist until a big merge reclaims them. The team had provisioned disk for the live data size, not for 2× it. The fix was to set `max_size_amplification_percent` to a value the disk could actually accommodate, accepting slightly higher write amplification in exchange for a hard space ceiling. The lesson: every compaction strategy has a worst-case space amplification, and you must provision disk for the worst case, not the steady state. Universal compaction trades space for write throughput — and you have to be able to pay the space.

### 10. WiscKey and BlobDB: when values dominate write amplification

A team storing large values — serialized blobs of tens of kilobytes each — under small keys saw brutal compaction write amplification: every time a key migrated down a level, its entire large value was rewritten along with it, even though the value never changed. The insight, from the *WiscKey* paper (Lu et al., 2016), is **key-value separation**: store the large values in a separate append-only log and keep only key→(value location) pointers in the LSM tree. Now compaction rewrites only the small keys and pointers; the big values are written exactly once. RocksDB implements this as **BlobDB**. The team enabled it and watched compaction write amplification fall dramatically, because the LSM tree was suddenly managing kilobytes of keys instead of gigabytes of values. The cost is an extra disk seek per read (follow the pointer to the value log) and a separate garbage-collection process for the value log. The lesson: the LSM tree's write amplification scales with the size of what it compacts, so if your values are large and immutable, taking them out of the compaction path is often the single biggest win available.

### 11. The consistent backup that took milliseconds

A team needed point-in-time backups of a multi-terabyte RocksDB instance without pausing writes, and their first design — copy the data directory with `rsync` while writes continued — produced corrupt, inconsistent backups, because compaction was deleting and creating files mid-copy. The first hypothesis was that they needed to stop the database during backup, which was operationally unacceptable. The actual solution fell straight out of SSTable immutability: RocksDB's `CreateCheckpoint` (and Cassandra's `nodetool snapshot`) creates a consistent snapshot by taking **hard links** to the current set of live SSTables into a new directory. Because the files are immutable, a hard link is a guaranteed-consistent, zero-copy reference that survives even after compaction "deletes" the original (the inode lives until the last link is gone). The snapshot completes in milliseconds regardless of data size, and the backup can then be copied off-box at leisure from the frozen link set. The lesson: immutability is not just a write-path optimization, it is what makes consistent, non-blocking, instantaneous snapshots possible — a property a mutable B-tree, whose pages are changing under you, has to work much harder to provide.

### 12. The bulk load that bypassed the write path entirely

A migration team needed to load two terabytes of existing data into a fresh RocksDB-backed service, and their first approach — insert it row by row through the normal API — pinned the write path for hours, drove compaction into permanent backlog, and triggered constant write stalls, because every row went through the memtable, the WAL, and the full leveled-compaction cascade. The realization was that for sorted bulk data, the entire write path is unnecessary overhead. They switched to **bulk ingestion**: generate SSTable files offline from the already-sorted source data, then atomically add them to the database with `IngestExternalFile` (Cassandra's equivalent is `sstableloader` / `nodetool import`). The ingested files are placed directly into an appropriate level — skipping the memtable, the WAL, and most compaction — and become live with a single manifest edit. The load time dropped from hours to minutes, with no write stalls, because no data ever traversed the hot path. The lesson: because the on-disk format is just immutable sorted files plus a manifest, you can manufacture those files out-of-band and install them atomically — the LSM tree's file-based design turns a brutal bulk load into a near-instant file move.

## The LSM family tree

By now it should be clear that LSM trees are not a niche technique — they are the dominant storage architecture for write-heavy and large-scale databases, and the lineage runs through three main families.

![Most modern key-value and wide-column databases are LSM engines descended from three lineages](/imgs/blogs/lsm-trees-write-optimized-storage-engines-10.webp)

The **Bigtable lineage** descends from Google's 2006 Bigtable paper, which introduced the memtable/SSTable/compaction design to the world: HBase, Cassandra, and ScyllaDB all implement this wide-column, LSM-backed model. The **RocksDB/LevelDB family** descends from Google's LevelDB (the embeddable distillation of Bigtable's storage layer) and its high-performance fork RocksDB; it powers MyRocks under MySQL, TiKV under TiDB, Kafka Streams' state stores, and countless embedded uses. The **Go-native engines** — Pebble under CockroachDB, BadgerDB — reimplement the same architecture for the Go ecosystem. The table below maps a few systems to their engine and default strategy.

| Database | Storage engine | Default compaction | Lineage |
| --- | --- | --- | --- |
| Cassandra | native (Java) | Size-tiered | Bigtable |
| ScyllaDB | native (C++/Seastar) | Incremental / size-tiered | Bigtable |
| HBase | native (Java) | Size-tiered-ish | Bigtable |
| RocksDB / MyRocks | RocksDB | Leveled | LevelDB |
| TiKV / TiDB | RocksDB | Leveled | LevelDB |
| CockroachDB | Pebble (Go) | Leveled | LevelDB-compatible |
| InfluxDB / QuestDB | native | Time-window-style | time-series |

That this many independent, well-funded teams converged on the same architecture is the strongest possible evidence that the core trade — sequential writes now, background cleanup later — is the right one for the workloads that define modern infrastructure. It is the same reason an in-memory store like [Redis](/blog/software-development/database/redis-applications-and-optimization) reaches for an append-only file (AOF) for its own durability: appending is what hardware is good at, and everything else is a derived structure you maintain in the background.

## When to reach for an LSM tree — and when not to

The LSM tree is a specialized tool that has become a default, which means it is now frequently used in situations where a B-tree would be simpler and faster. Choose deliberately.

**Reach for an LSM tree when:**

- Your workload is **write-heavy** — ingestion pipelines, event logs, metrics, time-series, chat/message stores, IoT telemetry. This is the home-field advantage and nothing else competes.
- You are **write-amplification- or flash-wear-bound**, and the densely-packed, sequential-write profile directly cuts hardware cost and extends SSD life (the MyRocks story).
- You need **high sustained ingest** with bounded, *tunable* read and space costs, and you are willing to operate compaction as a first-class concern.
- Your data is **append-only with a TTL** — time-window compaction turns expiry into free file deletion, which a B-tree cannot match.
- You are building a **distributed database** and want a storage engine whose immutable files replicate, snapshot, and back up byte-for-byte without locking.

**Skip the LSM tree when:**

- Your workload is **read-mostly with point lookups** and the dataset fits comfortably in or near RAM — a B-tree's single-leaf read beats the LSM tree's multi-level probe, and you avoid all of compaction's operational weight.
- You are **range-scan-dominated over frequently-updated data** — bloom filters do not help scans, and overlapping SSTables make scans touch many files unless leveled compaction works hard.
- You **cannot give compaction the I/O and CPU headroom** it needs — an LSM tree starved of background resources degrades into write stalls and unbounded read amplification. If your hardware is already saturated, the LSM tree will make it worse, not better.
- You need **predictable, low tail latency above all** and cannot tolerate the periodic I/O spikes compaction introduces, and you are not willing to invest in a compaction-aware scheduler.
- Your data is **small and static** — the entire LSM machinery is overhead with nothing to amortize it against.

The meta-lesson, true of every storage engine: the question is never "what is the best data structure," it is "where does my workload need to sit on the read/write/space triangle, and which engine lets me get there." An LSM tree gives you a dial across that triangle that a B-tree simply does not have. That flexibility is the whole reason it won — and the reason that using it without understanding the dial is how you end up writing 4 GB to store 100 MB.

## Further reading

- **The original LSM paper** — Patrick O'Neil, Edward Cheng, Dieter Gawlick, Elizabeth O'Neil, "The Log-Structured Merge-Tree (LSM-Tree)," *Acta Informatica*, 1996. The structure that named the field.
- **Bigtable** — Chang et al., "Bigtable: A Distributed Storage System for Structured Data," OSDI 2006. The memtable/SSTable/compaction design that everything descends from.
- **The RUM conjecture** — Athanassoulis et al., "Designing Access Methods: The RUM Conjecture," EDBT 2016. The triangle, formalized.
- **Monkey** — Dayan, Athanassoulis, Idreos, "Monkey: Optimal Navigable Key-Value Store," SIGMOD 2017. Why bloom-filter bits should not be allocated uniformly across levels.
- **WiscKey** — Lu et al., "WiscKey: Separating Keys from Values in SSD-conscious Storage," FAST 2016. Key-value separation for large values.
- **The RocksDB wiki** — the most thorough operational reference for compaction styles, tuning, and the options in this article.
- Sibling posts on this blog: [Random UUIDs Are Killing Your Database Performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance), [Database Connection Pooling: A Complete Guide](/blog/software-development/database/database-connection-pooling), and [Redis in Production](/blog/software-development/database/redis-applications-and-optimization).

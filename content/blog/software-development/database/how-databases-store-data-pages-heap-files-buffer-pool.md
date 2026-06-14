---
title: "How Databases Store Data: Pages, Heap Files, and the Buffer Pool"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles tour of how relational databases actually lay bytes on disk and in memory — the 8KB slotted page, heap files, tuple format and alignment padding, TOAST, the buffer pool and clock-sweep eviction, and how a single row read flows from logical query to physical block."
tags:
  [
    "database-internals",
    "storage-engine",
    "postgres",
    "innodb",
    "buffer-pool",
    "heap-file",
    "page-layout",
    "toast",
    "mvcc",
    "performance",
    "system-design",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-1.webp"
---

Ask a backend engineer where a row "lives" and you will usually get an answer that stops at the table name. The row is in `users`. Push one level deeper — *where in `users`, physically, on the disk?* — and the answers thin out fast. This is the layer almost everyone treats as a black box, and it is exactly the layer that decides whether your database does ten thousand transactions a second or two hundred. A row is not a row to the storage engine. It is a few dozen bytes wedged into a fixed-size block, addressed by an offset inside a slot array, cached in a frame of memory that some background process is constantly fighting to reclaim. The performance you observe at the SQL layer is the sum of a thousand small decisions made down here, in the format of a page and the policy of a cache.

The reason this matters is that the disk is slow and the page is the unit you pay for. You never read one row off an SSD. You read the 8 KB (Postgres) or 16 KB (InnoDB) block that contains it, the whole thing, every time, because that is the smallest chunk the storage engine and the operating system agree to move. If your row is 100 bytes and your page is 8192 bytes, a single uncached lookup drags 81× more bytes off the device than you asked for. Whether that is a catastrophe or a non-event depends entirely on how the engine packs rows into pages, how it decides which pages to keep in memory, and how cleverly it avoids touching the disk at all. Martin Kleppmann frames this in *Designing Data-Intensive Applications* (Chapter 3, "Storage and Retrieval"): the entire discipline of storage engineering is the management of the gap between the random-access illusion the query layer presents and the block-oriented reality the hardware enforces. Everything in this article is a tactic for managing that gap.

![The slotted 8KB page with a header, downward-growing line pointers, free space in the middle, and tuples packed upward from the end](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-1.webp)

The diagram above is the mental model for the entire post. A page is a small, self-contained filesystem: a header at the top with bookkeeping, an array of *line pointers* (Postgres calls them `ItemId`s) that grows downward, the actual *tuples* (rows) packed in from the bottom edge upward, and free space squeezed in the middle. An index never points at a row directly — it points at a line pointer, a `(block, slot)` pair, and the line pointer points at the tuple. That one layer of indirection is why a vacuum can shuffle rows within a page without rewriting every index, and it is the seed from which the rest of the design grows. We will spend the first half of the article touring this picture from the disk side, and the second half on the buffer pool — the memory cache that decides which of these pages you actually pay disk latency for.

## Why this layer breaks your intuition

Before we descend, let us name the mismatch directly, because almost every wrong performance intuition about databases comes from reasoning as if storage were row-oriented, byte-addressable, and free to read. It is none of those things.

| You assume | The naive mental model | The reality |
| --- | --- | --- |
| A row is stored as a row | One contiguous record on disk you can seek to | A row is a tuple wedged into a shared 8/16 KB page, reachable only by reading the whole page |
| Reads cost "one row's worth" of I/O | I/O scales with row size | I/O scales with *page* size; a 100-byte row read pulls 8192 bytes off disk |
| Column order is cosmetic | The optimizer sorts it out | Column order changes the on-disk row size via alignment padding — sometimes by 30%+ |
| Big values just make the row bigger | A 1 MB blob is a 1 MB row | Values over ~2 KB are compressed and moved out-of-line by TOAST; the main row keeps an 18-byte pointer |
| `shared_buffers` *is* the cache | Set it high, cache everything | Postgres double-buffers with the OS page cache; past ~25% of RAM you mostly waste memory |
| Caches are LRU | Least-recently-used eviction | Postgres uses clock-sweep; InnoDB uses a midpoint-insertion LRU that quarantines scans |
| A `DELETE` frees space | Row gone, bytes returned | The tuple is only marked dead; space returns at `VACUUM`/purge, and the page may never shrink |
| `SELECT` returns rows in order | Insertion order, or PK order | A heap scan returns rows in *physical page* order, which is arbitrary; only `ORDER BY` guarantees anything |

The single most expensive row to internalize is the second one. When you reason about a binary search tree you count comparisons and conclude that thirty of them is nothing. But the storage engine does not count comparisons — it counts **page reads**, because a comparison is a few nanoseconds and a page read from an SSD is tens of microseconds, from a spinning disk ten milliseconds. That is a five- to six-order-of-magnitude gap, and it reorders every design decision in this article. This is the same insight that makes [B-trees the right shape for an on-disk index](/blog/software-development/database/b-trees-how-database-indexes-work): pack hundreds of keys into one page so a billion-row lookup costs four page reads, not thirty.

> The first rule of storage engineering: the page is the atom of I/O. Every byte of design effort goes into reading fewer pages, packing more useful data per page, and keeping the hot pages in memory.

## 1. The page: why fixed-size blocks at all

**Rule of thumb: the storage engine never moves less than one page, so the page size is the granularity of every I/O, every lock, every checksum, and every cache slot. Pick your mental model accordingly.**

Why fixed-size pages? Because variable-size storage is a nightmare of bookkeeping and external fragmentation, and because the layers below the database — the filesystem, the block device, the SSD's flash translation layer — are themselves block-oriented. A modern SSD reads and writes in pages of its own (typically 4 KB or 16 KB) and erases in much larger blocks. Aligning the database page to a multiple of those hardware units means one logical page write maps cleanly onto the device's native operation instead of straddling two and triggering a read-modify-write. Postgres defaults to 8 KB (`BLCKSZ`, set at compile time; legal values are 1, 2, 4, 8, 16, 32 KB). InnoDB defaults to 16 KB (`innodb_page_size`, configurable to 4, 8, 16, 32, or 64 KB). Both are deliberate compromises: large enough to amortize per-page header overhead and to fit a fat B-tree node, small enough that reading one to fetch a single row is not absurdly wasteful, and small enough that the buffer pool can hold many of them.

The fixed size also gives the engine a trivially simple addressing scheme. A table's data file is just an array of pages, page 0 first, page 1 next, and so on. To find page `N`, you seek to byte `N × BLCKSZ`. No allocation table, no extent map for the basic case — the file *is* the array. Postgres calls a page a "block" and addresses it by a 32-bit block number, which caps a single relation file segment at `2^32 × 8 KB = 32 TB` before it must split into 1 GB segment files (`24576`, `24576.1`, `24576.2`, …). InnoDB addresses pages within a tablespace by a 32-bit page number too.

You can see the page structure directly. Postgres ships a contrib extension, `pageinspect`, that lets you crack open a raw block and read its bytes. This is the single best way to build intuition, so we will lean on it throughout.

```sql
-- Enable the low-level inspection extension (superuser).
CREATE EXTENSION IF NOT EXISTS pageinspect;

CREATE TABLE accounts (
    id       bigint PRIMARY KEY,
    balance  bigint NOT NULL,
    active   boolean NOT NULL,
    owner    text
);

INSERT INTO accounts
SELECT g, g * 100, (g % 2 = 0), 'owner_' || g
FROM generate_series(1, 500) AS g;

-- The page header of block 0 of the heap.
SELECT * FROM page_header(get_raw_page('accounts', 0));
```

That returns one row of header fields:

```
    lsn     | checksum | flags | lower | upper | special | pagesize | version | prune_xid
------------+----------+-------+-------+-------+---------+----------+---------+-----------
 0/1A2B3C4D |        0 |     0 |   292 |   848 |    8192 |     8192 |       4 |         0
```

Read those numbers against figure 1. `lower` (292) is the offset where the line-pointer array currently ends — everything below offset 292 is header plus line pointers. `upper` (848) is where the packed tuples begin. The free space on this page is exactly `upper - lower = 848 - 292 = 556` bytes. `special` is 8192, meaning the special space is empty (it sits at the very end of the page and is used only by index access methods; on a heap page it is zero-width). `pagesize` confirms 8192. This is the entire accounting model of a page in five integers: where the slots end, where the rows begin, and the gap between them is what you have left.

### Second-order: why 8 KB and not 4 KB or 64 KB

The page size is not free to choose after the fact, and the tradeoffs are real. A larger page (16, 32 KB) amortizes the per-page header (24 bytes in Postgres, ~120 bytes of fixed structure in InnoDB) across more rows, raises B-tree fanout so the tree is shallower, and turns more of your I/O into bigger sequential reads. But it also wastes more memory and bandwidth when you touch a single small row, it makes the buffer pool coarser (fewer, fatter cache slots, so a hot 200-byte row pins 16 KB of RAM), and it amplifies write cost: a one-byte change still rewrites the whole page to the WAL and to disk. Smaller pages cut that waste but raise tree height and per-page overhead. The 8 KB Postgres default and 16 KB InnoDB default landed where they did because for OLTP workloads — many small rows, point lookups, lots of small updates — that band minimizes total page reads across realistic access patterns. Analytics engines that scan huge ranges sequentially often prefer much larger blocks or abandon the page model entirely for columnar formats, precisely because their access pattern is different.

## 2. The slotted page: header, line pointers, tuples

**Rule of thumb: a page is a tiny append-structured arena. Slots grow down from the top, rows grow up from the bottom, and the engine only has to track two offsets to know exactly how much room is left.**

Now we zoom into one page and read it byte by byte. The Postgres page header (`PageHeaderData`) is exactly **24 bytes** and carries eight fields:

| Field | Bytes | Purpose |
| --- | --- | --- |
| `pd_lsn` | 8 | LSN of the last WAL record that changed this page (the WAL/checkpoint link) |
| `pd_checksum` | 2 | Page checksum, if `data_checksums` is on |
| `pd_flags` | 2 | Flag bits (has free line pointers, all-visible, etc.) |
| `pd_lower` | 2 | Offset to the *start* of free space (end of the line-pointer array) |
| `pd_upper` | 2 | Offset to the *end* of free space (start of the tuple area) |
| `pd_special` | 2 | Offset to special space (page end on heap pages) |
| `pd_pagesize_version` | 2 | Page size + layout version, packed |
| `pd_prune_xid` | 4 | Oldest un-pruned XMAX on the page, a hint for opportunistic pruning |

Immediately after the header comes the **line-pointer array**. Each entry, an `ItemIdData`, is exactly **4 bytes** and packs three things into a bitfield: a 15-bit offset to the tuple's first byte within the page, a 2-bit flag (`LP_NORMAL`, `LP_DEAD`, `LP_REDIRECT`, `LP_UNUSED`), and a 15-bit length. The array grows *downward* from the header as rows are added, which is why `pd_lower` advances. The tuples themselves are written *upward* from the end of the page (just before the special space), which is why `pd_upper` retreats. Free space is the shrinking gap between them. When `pd_lower` would cross `pd_upper`, the page is full and the row goes elsewhere.

This is called a **slotted page** layout, and it is nearly universal — Postgres, InnoDB, SQL Server, Oracle, and most embedded engines use a variant. The crucial property is the indirection: an index entry, and the `ctid` you see in queries, refers to the *line pointer slot* (`(block, slot)`), never to a raw byte offset. So the engine is free to physically rearrange tuples *within* a page — compacting away the holes left by dead rows during a "page prune" — by rewriting only the line pointers, while every index that points at those slots stays valid untouched. Move a tuple, repoint one 4-byte slot, done. That is the whole trick, and it is load-bearing for vacuum, for HOT updates, and for [how MVCC keeps old and new row versions side by side](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb).

Let us look at the line pointers and tuples directly with `heap_page_items`:

```sql
SELECT lp, lp_off, lp_flags, lp_len, t_ctid, t_xmin, t_xmax
FROM heap_page_items(get_raw_page('accounts', 0))
ORDER BY lp
LIMIT 5;
```

```
 lp | lp_off | lp_flags | lp_len | t_ctid | t_xmin | t_xmax
----+--------+----------+--------+--------+--------+--------
  1 |   8120 |        1 |     64 | (0,1)  |    742 |      0
  2 |   8056 |        1 |     64 | (0,2)  |    742 |      0
  3 |   7992 |        1 |     64 | (0,3)  |    742 |      0
  4 |   7928 |        1 |     64 | (0,4)  |    742 |      0
  5 |   7864 |        1 |     64 | (0,5)  |    742 |      0
```

Now figure 1 is concrete. `lp` is the slot index. `lp_off` is exactly where in the 8192-byte page that tuple's bytes start — and notice they descend (8120, 8056, 7992, …), 64 bytes apart, each tuple sitting just below the previous one as the tuple area grows upward from the end. `lp_flags = 1` is `LP_NORMAL` (a live pointer). `lp_len = 64` is the tuple length including its header. `t_ctid = (0, 1)` means "this version's current location is block 0, slot 1" — for an un-updated row, `t_ctid` points at itself. `t_xmin = 742` is the transaction that inserted it; `t_xmax = 0` means no transaction has deleted it. Those last two are the engine of MVCC: visibility is decided by comparing your snapshot against `t_xmin`/`t_xmax`, which is why every row carries them.

### Second-order: the line-pointer array never shrinks, and that bites

Here is a non-obvious gotcha that causes real production bloat. When a tuple dies and is vacuumed, its line pointer is not necessarily removed — it is marked `LP_DEAD` or `LP_UNUSED`, and the *slot* stays in the array so that `ctid`s referenced by indexes remain stable until a later, heavier cleanup. The line-pointer array (`pd_lower`) can therefore grow over a table's life and not come back down, even after the dead tuples are reclaimed. A table that churns through many short-lived rows can end up with pages that are mostly empty tuple space but still carry a long array of dead pointers, capping how many new rows fit per page. You can see this with `pageinspect` (count `lp_flags = 0` unused vs `lp_flags = 1` normal). The fix is `VACUUM FULL` or `pg_repack` (which physically rewrites the table and resets the arrays), not plain `VACUUM` — plain vacuum reclaims tuple space but is conservative about shrinking the pointer array.

## 3. The heap file: an unordered array of pages

**Rule of thumb: a Postgres table is a heap — an unordered pile of pages. Rows live wherever there was free space at insert time, and the only thing that imposes order on them is an index.**

Step back from the single page to the whole file. A Postgres table's main fork is a heap: literally an array of the pages we just dissected, with no ordering relationship between the rows in page 5 and the rows in page 6. When you `INSERT`, the engine asks the **Free Space Map** (a tiny side fork, `_fsm`, holding roughly one byte per page) for a page with enough room, drops the tuple there, and updates the map. There is no attempt to keep `id = 100` near `id = 101`. They could be in entirely different pages, in either order, depending purely on when each was inserted and where there happened to be space.

![A heap file drawn as a grid of 8KB page blocks with a free space map, visibility map, and a B-tree index pointing into one block](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-2.webp)

Figure 2 is the file-level view. Four pages, each independently packed, rows scattered by insertion timing. Two side forks ride alongside the heap: the Free Space Map answers "which page has room?" in O(1) so inserts do not scan, and the **Visibility Map** (`_vm`, two bits per page) marks pages where every tuple is visible to all transactions — which lets `VACUUM` skip them and lets the planner serve **index-only scans** without visiting the heap at all. And off to the right, the B-tree index: *the only structure that imposes key order*. Its leaves hold sorted keys, each pointing at a heap `ctid`. Without the index, finding `id = 512` means scanning every page; with it, you descend ~4 index pages to a leaf, read `(1, 2)`, and jump straight to block 1, slot 2.

This unordered-heap design has three consequences worth burning in:

1. **`SELECT … LIMIT 5` with no `ORDER BY` returns physical-order rows.** Whatever order the pages happen to be scanned in. New developers are forever surprised that the "first five rows" change after a vacuum or an update; there was never a first five.
2. **A sequential scan reads the file front to back** at the OS's sequential-read bandwidth, which is the *one* thing spinning disks and SSDs are both good at. This is why a full scan of a 1 GB table can be faster than an index scan that touches 30% of it via random page reads — the planner's `seq_page_cost` vs `random_page_cost` knobs encode exactly this tradeoff.
3. **Insert location is unpredictable, which interacts viciously with random keys.** This is the entire mechanism behind why [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance): a UUIDv4 primary key scatters the *index* inserts across every leaf page uniformly, so each insert dirties a different, probably-cold index page, maximizing the number of distinct pages touched per batch and shredding cache locality. The heap itself appends happily; it is the index over the random key that suffers.

You can watch a heap grow. The `ctid` pseudo-column is the live `(block, slot)`:

```sql
SELECT ctid, id FROM accounts WHERE id IN (1, 2, 250, 499, 500) ORDER BY id;
```

```
 ctid    |  id
---------+-----
 (0,1)   |   1
 (0,2)   |   2
 (1,108) | 250
 (3,103) | 499
 (3,104) | 500
```

`id = 1` is in block 0 slot 1; `id = 250` is in block 1; `id = 499` and `id = 500` are in block 3. The physical layout tracks insertion order here because we did one bulk insert, but the moment you start updating and vacuuming, that correspondence dissolves. Crucially, the `ctid` is *not stable*: an `UPDATE` writes a new tuple version somewhere else and the old `ctid` may be reused after vacuum. Never store a `ctid` as if it were a primary key.

### Second-order: CLUSTER buys ordering once, then decays

Postgres lets you physically reorder a heap to match an index with `CLUSTER accounts USING accounts_pkey`. This rewrites the table so rows are stored in primary-key order, which makes range scans over that key read contiguous pages — a real win for `WHERE id BETWEEN …` patterns. The catch is that it is a one-time, fully-locking rewrite and it does *not* stay clustered: every subsequent insert and HOT-violating update lands wherever there is room, and the ordering decays. Postgres has no automatic re-clustering. This is precisely the structural difference from InnoDB, where the table is *always* stored in primary-key order because the table *is* the primary-key B-tree. We will get to that contrast in section 7; it is one of the deepest design forks in the relational world.

## 4. Tuple format: header, NULL bitmap, and the alignment tax

**Rule of thumb: every row pays a fixed ~24-byte header before a single byte of your data, and the order you declare your columns silently changes the row size through alignment padding.**

Now the most under-appreciated layer: the format of a single tuple, and why two tables with the *exact same columns* can have different on-disk sizes. Every Postgres heap tuple begins with a `HeapTupleHeaderData` of **23 bytes**, rounded up by `MAXALIGN` (8 on 64-bit platforms) to **24 bytes** before the data starts. The header fields:

| Field | Bytes | Purpose |
| --- | --- | --- |
| `t_xmin` | 4 | Inserting transaction ID (MVCC) |
| `t_xmax` | 4 | Deleting/locking transaction ID (MVCC) |
| `t_cid` / `t_xvac` | 4 | Command ID within the transaction, or vacuum XID (union) |
| `t_ctid` | 6 | `(block, slot)` of this version or the next version in an update chain |
| `t_infomask2` | 2 | Number of attributes + flag bits (e.g. HOT-updated) |
| `t_infomask` | 2 | Flag bits (has nulls, has varwidth, has external/TOAST, …) |
| `t_hoff` | 1 | Offset from tuple start to user data (always a MAXALIGN multiple) |

![A tuple drawn as a byte strip: 23-byte header fields, a NULL bitmap, MAXALIGN padding, then user-data columns each padded to its alignment](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-3.webp)

Figure 3 lays the strip out. After the header, if *any* column in the row is nullable and actually NULL, the tuple carries a **NULL bitmap**: one bit per column, set to 1 for present and 0 for NULL. The bitmap costs `ceil(natts / 8)` bytes and exists only when the `HEAP_HASNULL` flag is set in `t_infomask`. A clever detail: a NULL column stores *nothing at all* in the data area — the bitmap bit is its entire representation. This is why a wide table full of NULLs can be dramatically smaller than you expect, and why "add a nullable column" is a cheap, instant DDL in modern Postgres (it just flips defaults; existing rows are read as NULL via the bitmap and `natts`).

Then comes the data, starting at `t_hoff`, and here is the tax. Each fixed-width type has an **alignment requirement**: an `int4` must start at a 4-byte boundary, an `int8`/`bigint`/`timestamptz`/`double precision` at an 8-byte boundary, a `bool` or `char` anywhere. The engine inserts **padding bytes** between columns to satisfy these boundaries, and those padding bytes store nothing — they are pure overhead. The order you declare columns therefore determines how much padding you pay.

Consider a row with `active bool` (1 byte, align 1), `id int4` (4 bytes, align 4), `balance int8` (8 bytes, align 8). Declared in that order:

- `active` at offset 0, 1 byte. Next column `id` needs a 4-byte boundary, so **3 padding bytes** (offsets 1–3).
- `id` at offset 4, 4 bytes. Next column `balance` needs an 8-byte boundary; offset is now 8, already aligned, **0 padding**.
- `balance` at offset 8, 8 bytes. Row data ends at 16. Useful data: 13 bytes. Padding: 3 bytes.

Now declare them `balance int8, id int4, active bool`:

- `balance` at offset 0, 8 bytes. `id` needs a 4-byte boundary; offset 8 is aligned, **0 padding**.
- `id` at offset 8, 4 bytes. `active` needs no alignment; offset 12, **0 padding**.
- `active` at offset 12, 1 byte. Data ends at 13. **0 inter-column padding.**

Same three columns, same data, but the first order wastes 3 bytes per row to inter-column padding and the second wastes none. Across a billion rows that is 3 GB of pure padding, plus the cache and I/O cost of reading it. The rule that falls out is mechanical: **declare fixed-width columns from widest alignment to narrowest** (8-byte types first, then 4-byte, then 2-byte, then 1-byte and variable-length), and you minimize padding automatically.

You can measure any value's stored size directly:

```sql
-- Compare the on-disk size of a column value.
SELECT pg_column_size(ROW(true, 42, 1000000::bigint))   AS bad_order,
       pg_column_size(ROW(1000000::bigint, 42, true))   AS good_order;
```

```
 bad_order | good_order
-----------+------------
        40 |         37
```

`pg_column_size` includes the row overhead, so the absolute numbers carry the header; the *difference* (3 bytes) is the padding the bad order cost. For a real table, `SELECT pg_column_size(t) FROM tbl t LIMIT 1` shows the per-row footprint, and `SELECT pg_size_pretty(pg_relation_size('tbl'))` shows the total.

![A 4x4 matrix showing the same four columns in a bad order with padding bytes versus a good widest-first order with zero inter-column padding](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-8.webp)

Figure 8 makes the column-order effect concrete across four columns. The red cells in the "bad order" columns are padding bytes that store nothing; the green/zero cells in the "good order" columns show the padding eliminated by reordering widest-to-narrowest. Same data, smaller tuple, more rows per page, fewer page reads on every scan that touches the table.

### Second-order: the header dominates skinny tables

Run the math for a two-column table of `(id bigint, n bigint)`: 24 bytes of header + 16 bytes of data = 40 bytes per tuple, plus a 4-byte line pointer = 44 bytes of storage for 16 bytes of payload. The header is **60% overhead**. A billion such rows spend 24 GB on tuple headers alone before storing a byte of your data. This is the case *against* the EAV (entity-attribute-value) anti-pattern and against splitting a wide entity into many skinny tables: every split multiplies the per-row header tax. Conversely, a wide table with thirty columns amortizes the same 24-byte header across far more payload, so the *relative* overhead shrinks. There is a genuine sweet spot, and "normalize until it hurts, then stop" is partly a statement about this header tax.

## 5. TOAST: when a value is too big for a page

**Rule of thumb: a value can never make a tuple exceed roughly a quarter of a page on its own. The moment a row threatens the 2 KB threshold, TOAST compresses the wide columns and moves the overflow out of the main page entirely.**

A page is 8192 bytes and a tuple cannot span pages, so what happens when you store a 50 KB JSON document or a 2 MB image? Postgres solves this with **TOAST** — The Oversized-Attribute Storage Technique — and it runs automatically and invisibly. The trigger is `TOAST_TUPLE_THRESHOLD`, normally **2032 bytes** on a standard 8 KB build (chosen so at least four tuples fit per page). When a row to be stored exceeds it, the toaster activates and works the row down until it is under `TOAST_TUPLE_TARGET` (also ~2 KB, tunable per table via `toast_tuple_target`) or no more gains are possible.

![A before-and-after showing a 50KB value that cannot fit a page, then TOAST compressing, chunking, and leaving a small pointer in the main row](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-6.webp)

Figure 6 shows the two-stage strategy. First, **compression**: each `TOASTable` column has a storage strategy (`EXTENDED`, the default for `text`/`jsonb`/`bytea`, tries compression then out-of-line; `EXTERNAL` skips compression but goes out-of-line; `MAIN` prefers inline-compressed; `PLAIN` forbids both). Postgres compresses with `pglz` by default, or LZ4 if the column is set to it (LZ4 is faster to compress and decompress and usually compresses better on real data — EnterpriseDB's measurements show it materially cheaper on CPU). Second, if compression alone does not get the row under target, **out-of-line storage**: the value is split into ~2 KB chunks, each chunk stored as a row in a hidden per-table TOAST table (`pg_toast.pg_toast_<oid>`), and the main tuple keeps only an **18-byte TOAST pointer** (a `varatt_external` struct: the value OID, the toasted length, the raw length, the compression method). The wide column has effectively left the building; the main heap page stays slim.

This is enormous for performance, and the win is subtle. A sequential scan that does not reference the big column never reads the TOAST table at all — the main heap is dense with small tuples, so the scan touches far fewer pages. Conversely, a query that *does* dereference the TOASTed value pays an extra lookup into the TOAST table (an index scan over its chunk index) plus decompression. The lesson: **a wide, rarely-read column is nearly free; a wide, hot column is expensive on every read.** If you `SELECT *` from a table with a big TOASTed column in a tight loop, you are paying TOAST detoast cost you may not need.

You can see TOAST in action:

```sql
CREATE TABLE docs (id int PRIMARY KEY, body text);
INSERT INTO docs VALUES (1, repeat('lorem ipsum dolor ', 5000));  -- ~90 KB

-- pg_column_size shows the *compressed, stored* size of the value.
SELECT pg_column_size(body) AS stored, length(body) AS logical FROM docs;
```

```
 stored | logical
--------+---------
   1432 |   90000
```

90 KB of repetitive text compresses to 1432 bytes — under threshold, so it may even stay inline. Try incompressible data and watch it go out of line:

```sql
INSERT INTO docs VALUES (2, encode(gen_random_bytes(50000), 'base64'));  -- ~67 KB random

SELECT relname, pg_size_pretty(pg_total_relation_size(relname::regclass))
FROM (SELECT 'docs' AS relname
      UNION SELECT relname FROM pg_class
            WHERE oid = (SELECT reltoastrelid FROM pg_class WHERE relname='docs')) s;
```

You will find the bulk of the storage in the TOAST relation, not the main table. InnoDB has its own version of this: rows whose variable-length columns overflow the page store a 20-byte pointer in the record and push the overflow to separate **overflow pages**, with the `DYNAMIC`/`COMPRESSED` row formats storing only the pointer in the clustered-index record and `COMPACT`/`REDUNDANT` storing a 768-byte prefix inline.

### Second-order: TOAST OID exhaustion and update churn

Two production hazards live here. First, every TOAST table addresses its chunks by a 32-bit OID, and a table that inserts billions of large values can theoretically exhaust the TOAST OID space, causing insert stalls while Postgres searches for a free OID — AWS has documented this "TOAST OID contention" on high-churn Aurora/RDS workloads. Second, TOAST interacts badly with updates: updating *any* column of a row with an out-of-line value rewrites the main tuple, and if the TOASTed column changed, it writes entirely new TOAST chunks (the old ones become dead and wait for vacuum). A table of large, frequently-updated documents generates enormous TOAST churn and bloat. The mitigation is to split rarely-changing large blobs into their own table keyed by the parent, so an update to the parent's hot columns never touches the blob's TOAST chunks.

## 6. The buffer pool: caching hot pages in RAM

**Rule of thumb: the buffer pool is the difference between a microsecond and a millisecond per row. Every page the engine touches must be in a buffer frame first; the entire game is keeping the pages you will touch next already resident.**

We have spent five sections on disk format. Now the half that decides your actual latency: almost no read touches the disk, because the engine keeps a pool of fixed-size **frames** in shared memory, each holding one page, and serves repeat accesses from RAM. In Postgres this is `shared_buffers`; in InnoDB it is the `innodb_buffer_pool`. A frame is exactly one page wide, so a default Postgres pool of 4 GB holds `4 GB / 8 KB = 524288` frames. Every page request first hashes `(relation, fork, block)` into a hash table to see if the page is already resident. A **hit** returns a pinned pointer to the frame in nanoseconds. A **miss** must find a free frame, possibly evict a victim, issue the disk read, and only then return.

![A branching graph of the buffer pool: a page request hits the hash table, splits into cache hit versus miss, and the clock hand sweeps frames to find a victim](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-4.webp)

Figure 4 traces both paths. The hot path is the green one: hit, bump the page's usage count, return. The cold path is where the policy lives. On a miss the engine needs a frame, and if the pool is full it must evict something — which raises the central question of every cache: *which page do you throw away?*

### Postgres clock-sweep

Postgres does not use a true LRU, because maintaining an exact LRU ordering requires moving an entry to the head of a list on *every* access, and that list is a global contention point under concurrency. Instead it uses **clock-sweep**, a cheap LRU approximation. Each buffer carries a small `usage_count` (0–5). When a page is accessed, its count is incremented (capped at 5). A "clock hand" (`nextVictimBuffer`) sweeps circularly through the frames looking for a victim: at each frame, if `usage_count > 0` it *decrements* the count and moves on; if `usage_count == 0` and the page is unpinned, that frame is the victim. Hot pages, accessed often, keep getting re-incremented and survive many sweeps; cold pages decay to zero and get evicted. It approximates LRU at O(1) amortized cost with no per-access list surgery.

You can watch the pool's contents with the `pg_buffercache` extension:

```sql
CREATE EXTENSION IF NOT EXISTS pg_buffercache;

-- Which relations occupy the most buffers, and how hot are they?
SELECT c.relname,
       count(*)                              AS buffers,
       pg_size_pretty(count(*) * 8192)       AS cached,
       round(avg(b.usagecount), 2)           AS avg_usage,
       sum(CASE WHEN b.isdirty THEN 1 ELSE 0 END) AS dirty
FROM pg_buffercache b
JOIN pg_class c ON b.relfilenode = pg_relation_filenode(c.oid)
GROUP BY c.relname
ORDER BY buffers DESC
LIMIT 10;
```

```
   relname    | buffers | cached  | avg_usage | dirty
--------------+---------+---------+-----------+-------
 accounts     |   18342 | 143 MB  |      3.81 |   204
 accounts_pkey|    9120 | 71 MB   |      4.55 |    18
 orders       |    4201 | 33 MB   |      2.10 |   512
 ...
```

`avg_usage` near 5 means the relation's pages are red-hot and will survive any sweep; near 0–1 means they are cooling and eviction candidates. `dirty` is the count of modified pages not yet written back — directly relevant to checkpoint cost, which we will get to. This single query is one of the most useful diagnostics in Postgres: it tells you, empirically, what your working set actually is rather than what you assume it is.

### InnoDB midpoint-insertion LRU

InnoDB takes a different tack on the same problem. Its buffer pool is a genuine LRU list, but with a twist designed to defeat the classic LRU failure mode: a big sequential scan (a backup, a one-off analytics query) that reads millions of pages once and would, under naive LRU, flush your entire hot working set to make room for data you will never touch again. InnoDB splits the LRU list into a **young (new) sublist** at the head and an **old sublist** at the tail, with the boundary at the **midpoint**. By default `innodb_old_blocks_pct = 37` — the old sublist is 3/8 of the pool. A freshly read page is inserted *at the midpoint* (the head of the old sublist), not at the very head. It only gets promoted into the young sublist if it is accessed *again* after sitting in the old sublist for at least `innodb_old_blocks_time` milliseconds (default 1000). So a sequential scan, which touches each page exactly once and moves on, populates only the old sublist and is evicted from there without ever displacing the genuinely hot pages in the young sublist. The old sublist acts as a quarantine for one-shot reads. This is called making the buffer pool "scan-resistant," and it is a more deliberate design than clock-sweep — it bakes in the assumption that scans are the enemy of cache locality.

| Aspect | Postgres clock-sweep | InnoDB midpoint LRU |
| --- | --- | --- |
| Data structure | Array of frames + circular hand | Doubly-linked LRU list, split in two |
| Eviction policy | Approximate LRU via usage_count decay | True LRU within young/old sublists |
| New page lands | In a free/victim frame, usage_count = 1 | At the midpoint (head of old sublist) |
| Scan resistance | Weak (one pass bumps usage to 1) | Strong (stays in old sublist, evicted fast) |
| Key knob | `shared_buffers` | `innodb_buffer_pool_size`, `innodb_old_blocks_pct/time` |
| Promotion | usage_count increments on access (cap 5) | Re-access after `old_blocks_time` ms → young |
| Inspect | `pg_buffercache` | `SHOW ENGINE INNODB STATUS`, `information_schema.INNODB_BUFFER_PAGE` |

### Second-order: the hit ratio is a trap

Everyone wants a 99% buffer cache hit ratio, and most tuning guides chase it. But the hit ratio is a deeply misleading single number, for two reasons. First, in Postgres it only measures the `shared_buffers` layer — a "miss" there often hits the **OS page cache** below and never touches the disk, so a 90% Postgres hit ratio can be a 99.9% true-RAM ratio. Second, a high hit ratio over a long window hides the brief, brutal misses that actually cause your tail latency: the cold-start after a failover, the first query after a deploy, the page that just got evicted by a scan. The number to watch is not the average hit ratio but the *rate* of `buffers_backend` reads under load and the p99 latency of queries that miss. Chase the misses that hurt, not the average that flatters.

## 7. Heap-plus-ctid versus the clustered index

**Rule of thumb: Postgres stores the table and its indexes as separate structures linked by physical location; InnoDB stores the table *inside* its primary-key index. This one fork explains a decade of write-amplification debates.**

This is the deepest design difference in the relational world, and it has real, measurable consequences. In Postgres, the heap is one structure and every index — primary key included — is a *separate* B-tree whose leaves point at heap `ctid`s (physical `(block, slot)` locations). In InnoDB, the table itself *is* the primary-key B-tree: the leaf pages of the clustered index hold the full rows, sorted by primary key, and there is no separate heap at all. Secondary indexes in InnoDB do not store a physical location — they store the **primary-key value**, and a lookup via a secondary index does *two* B-tree descents (secondary index → PK value → clustered index → row).

![A before-and-after comparing the Postgres heap-plus-ctid model with the InnoDB clustered primary-key index model](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-7.webp)

Figure 7 draws the contrast. The Postgres side (left) shows every index entry pointing at a physical `ctid`. The consequence, which Uber famously documented in "[Why Uber Engineering Switched from Postgres to MySQL](https://www.uber.com/en-US/blog/postgres-to-mysql-migration/)," is **write amplification**. Because Postgres MVCC makes an `UPDATE` write a brand-new tuple version at a new `ctid` (the old one stays until vacuum), and because every index entry points at a physical location, an update must insert *new entries into every single index* — even indexes on columns that did not change — so they point at the new `ctid`. Uber's example: updating only a person's birth year still forces a rewrite of the primary-key index, the `(first, last)` name index, *and* the birth-year index, because all three must now reference the new tuple location. "If we have a table with a dozen indexes defined on it, an update to a field that is only covered by a single index must be propagated into all 12 indexes." That amplification then flows into replication, because Postgres ships physical WAL records — so the write amplification becomes a replication-bandwidth amplification across data centers.

The InnoDB side (right) shows the alternative. Because secondary indexes store the *primary key value* (a logical pointer) rather than a physical location, and because InnoDB updates rows in place when it can, an update only has to touch the indexes on columns that actually changed. Update the birth year, and only the birth-year index is touched; the PK index and the name index do not move because the row's PK value did not change and the row stayed put. This is the architectural reason InnoDB's update path is lighter on multi-index tables.

The tradeoff is not free, and it cuts both ways:

| Dimension | Postgres (heap + ctid) | InnoDB (clustered PK) |
| --- | --- | --- |
| Primary-key range scan | Random heap I/O (heap not key-ordered) | Sequential (rows physically PK-ordered) |
| Secondary index lookup | One descent + heap fetch | Two descents (index → PK → clustered) |
| Update touching one indexed col | Rewrites *all* indexes (new ctid) | Touches only changed indexes |
| Big/random primary key | Heap unaffected; index suffers | PK *is* the table; random PK fragments the whole table |
| Secondary index size | Stores 6-byte ctid | Stores full PK value (fat PK = fat indexes) |
| Reclaiming dead versions | `VACUUM` (separate process) | Purge thread (background) |

The InnoDB clustered design is exactly why [random UUID primary keys are so much more destructive in MySQL](/blog/software-development/database/random-uuids-are-killing-your-database-performance) than in Postgres: a random PK in InnoDB scatters inserts across the *entire table* (because the table is ordered by that PK), fragmenting the clustered index and forcing page splits everywhere; in Postgres the heap just appends and only the index over the UUID suffers. It is also why InnoDB advises keeping the primary key small — every secondary index embeds a full copy of it.

### Second-order: HOT updates are Postgres clawing back the amplification

Postgres did not accept the write-amplification penalty quietly. The **HOT** (Heap-Only Tuple) optimization is its answer. When an `UPDATE` does not change any indexed column *and* there is room on the same page for the new tuple version, Postgres writes the new version on the same page, chains it to the old via `t_ctid`, and crucially **does not touch any index at all** — the index entries keep pointing at the original line pointer, which redirects (`LP_REDIRECT`) to the live version. A HOT update is the cheap path: one page write, zero index writes. The two preconditions are exactly the lever you control: *no indexed column changed* (a schema/query property) and *there is free space on the page* (a storage property), which brings us to fillfactor.

## 8. Fillfactor: leaving room to update in place

**Rule of thumb: a page packed to 100% has no room for a HOT update, so the next update to any row on it must go to a different page and rewrite every index. Leaving 10–20% free per page buys in-place updates.**

`fillfactor` is the percentage of a page the engine will fill before declaring it full and moving on, on *initial inserts*. The default for a Postgres heap is **100** (fill it completely); for a B-tree it is **90** (leave room for in-order inserts). Lowering a table's fillfactor reserves free space on every page specifically so that future updates to rows on that page can be HOT — written in place, same page, no index churn.

```sql
-- Reserve 15% of each page for in-place (HOT) updates.
ALTER TABLE accounts SET (fillfactor = 85);
-- fillfactor only affects pages written *after* this, so rewrite to apply:
VACUUM FULL accounts;   -- or pg_repack for online rewrite
```

Crunchy Data's measurements on this are striking: on an update-heavy table, dropping fillfactor from 100 to a value in the 70–90 range so that updates stay HOT can cut index bloat dramatically and improve update throughput, because the expensive part of an update on a multi-index table is the index maintenance, and HOT skips it entirely. The cost is space and read efficiency: a fillfactor of 80 means every page is 20% empty, so a sequential scan reads 25% more pages to cover the same rows, and your table is physically 25% larger. The right value is a workload decision: read-mostly tables want 100 (dense, cache-efficient); update-heavy tables on rows with stable indexed columns want 80–90 (room for HOT).

You can confirm HOT is actually happening:

```sql
SELECT relname, n_tup_upd, n_tup_hot_upd,
       round(100.0 * n_tup_hot_upd / NULLIF(n_tup_upd, 0), 1) AS hot_pct
FROM pg_stat_user_tables
WHERE relname = 'accounts';
```

```
 relname  | n_tup_upd | n_tup_hot_upd | hot_pct
----------+-----------+---------------+---------
 accounts |    482910 |        451203 |    93.4
```

A `hot_pct` of 93 means 93% of updates avoided all index writes. If that number is low on an update-heavy table, you are either updating indexed columns (a query/schema problem) or running out of page room (a fillfactor problem). InnoDB has the analogous `innodb_fill_factor`, defaulting to 100 but reserving 1/16 (≈6%) of clustered-index leaf pages for growth even at 100, and merging pages when they drop below `MERGE_THRESHOLD` (default 50%).

### Second-order: fillfactor is not retroactive and decays

The trap is that `fillfactor` only governs pages written *after* you set it. Existing pages keep whatever fill they had until rewritten. And as a table churns, HOT-update chains and dead tuples consume the reserved space, so a table that was at 85 can drift back toward fully-packed pages, at which point updates stop being HOT and the bloat-and-amplification cycle resumes. Healthy HOT-heavy tables need aggressive autovacuum so that dead-tuple space (and the room HOT updates need) is reclaimed promptly. Fillfactor and autovacuum are a pair; tuning one without the other gets you nowhere.

## 9. The write path: WAL, dirty pages, and checkpoints

**Rule of thumb: a committed write is durable the moment its WAL record is fsync'd, long before the data page itself reaches disk. The data page is written lazily, in bulk, at a checkpoint — which is where your I/O spikes come from.**

We have talked about reads. Writes have a subtlety that ties the whole system together: when you `UPDATE` a row, the engine modifies the page *in the buffer pool*, marks the frame **dirty**, and writes a small record describing the change to the **Write-Ahead Log**. It does *not* write the data page to disk yet. Durability comes entirely from the WAL: on `COMMIT`, the WAL record is flushed (`fsync`) to disk, and at that instant the transaction is durable even though the modified data page is still only in memory. If the server crashes, recovery replays the WAL from the last checkpoint forward, re-applying changes to data pages.

This is the **write-ahead** principle, and it is the reason databases can be both fast and durable. Sequential WAL appends are cheap (one fsync per commit, batchable across concurrent commits via group commit); random data-page writes are expensive and are deferred. The dirty pages accumulate in the buffer pool and are written back lazily — by the background writer trickling them out, or in bulk at a **checkpoint**.

A checkpoint is the moment the engine guarantees that all dirty pages up to a certain WAL position have been flushed to their data files, so that WAL before that position can be recycled and recovery never has to replay further back. Checkpoints flush *every* dirty page in the buffer pool. This is the source of the periodic I/O spike every Postgres operator knows: `pd_lsn` on each page (the very first field of the page header, from section 2) records which WAL record last touched it, and the checkpoint must ensure all those pages are on disk before advancing the redo point. The relevant knobs:

```ini
# postgresql.conf — checkpoint and WAL tuning
checkpoint_timeout = 15min          # max time between checkpoints
max_wal_size = 8GB                  # checkpoint triggered if WAL grows past this
checkpoint_completion_target = 0.9  # spread the flush over 90% of the interval
wal_compression = on               # compress full-page images in WAL
```

`checkpoint_completion_target = 0.9` is the single most important one: it tells Postgres to *spread* the dirty-page flush across 90% of the checkpoint interval instead of dumping it all at once, smoothing the I/O spike from a wall into a ramp. A larger `shared_buffers` means more dirty pages can accumulate, which means longer, heavier checkpoints — one of the real reasons not to set `shared_buffers` arbitrarily high. The relationship to the buffer pool is direct: the buffer pool's job is to *absorb* writes so they coalesce (a row updated ten times before checkpoint is written to disk once), and the checkpoint's job is to drain that absorption periodically so recovery stays bounded.

### Second-order: full-page writes and torn pages

There is a deep hazard here. If the OS writes an 8 KB page and the power fails after only 4 KB hit the platter, you have a **torn page** — half old, half new, and the page checksum will not even tell you which half. The WAL record, being a delta, cannot fix a torn page because it assumes the base page was intact. Postgres defends against this with **full-page writes** (`full_page_writes = on`): the *first* time a page is modified after a checkpoint, the *entire* page image is written to the WAL, not just the delta. Recovery can then restore the whole page from the WAL regardless of tearing. This is correct but expensive — it bloats the WAL, especially right after a checkpoint, and it is a major reason WAL volume can dwarf your logical write volume. (InnoDB solves the same problem differently, with the **doublewrite buffer**: it writes every page to a contiguous scratch area first, then to its real location, so a torn real-location write can be recovered from the intact doublewrite copy.) Either way, the torn-page defense is a hidden multiplier on your write volume that explains why "I wrote 100 MB but the disk wrote 1 GB" is normal.

## 10. Double buffering and tuning shared_buffers

**Rule of thumb: Postgres caches pages twice — once in `shared_buffers`, once in the OS page cache. That redundancy is why the conventional ceiling is 25% of RAM, not 80%.**

Here is the counterintuitive finale of the memory story. When Postgres reads a page from disk, the *operating system* also caches it in the kernel page cache. So a hot page can exist in RAM **twice**: once in `shared_buffers` and once in the OS cache. This is **double buffering**, and it is wasteful — the same bytes occupy memory in two places. Postgres does not bypass the OS cache (it uses buffered I/O, not direct I/O, by default), so this redundancy is structural.

The practical consequence governs the most-tuned setting in Postgres. The conventional recommendation, repeated by Crunchy, EDB, Cybertec, and the pganalyze team, is **`shared_buffers = 25% of RAM`**. Not because 25% is magic, but because of the double-buffering balance: you want enough dedicated buffer to hold the genuinely hot working set and to coalesce writes, but you want to leave the majority of RAM to the OS page cache, which catches the `shared_buffers` misses, handles read-ahead on sequential scans, and buffers writes. Setting `shared_buffers` to 80% of RAM does not give you an 80% cache — it gives you a 55%-wasted double-cache, starves the OS cache, makes checkpoints longer (more dirty pages), and on many workloads measurably *hurts*. The community guidance is that beyond ~40% it almost always regresses.

```sql
SHOW shared_buffers;             -- current setting
SELECT pg_size_pretty(pg_size_bytes(current_setting('shared_buffers')));

-- A rough working-set estimate: how much of each table is actually hot?
SELECT c.relname,
       pg_size_pretty(pg_relation_size(c.oid))            AS on_disk,
       pg_size_pretty(count(*) FILTER (WHERE b.usagecount >= 3) * 8192) AS hot_cached
FROM pg_buffercache b
JOIN pg_class c ON b.relfilenode = pg_relation_filenode(c.oid)
WHERE c.relkind = 'r'
GROUP BY c.relname, c.oid
ORDER BY count(*) DESC LIMIT 10;
```

The real tuning move is not to crank `shared_buffers` blindly but to measure: use `pg_buffercache` to see whether your hot tables actually fit, and `effective_cache_size` (a *hint* to the planner, not an allocation — set it to ~50–75% of RAM to tell the planner how much total cache, including the OS, it can assume) so the planner correctly favors index scans when data is likely cached. InnoDB is the opposite philosophy: because it uses `O_DIRECT` (`innodb_flush_method = O_DIRECT`) to bypass the OS cache and avoid exactly this double-buffering, the InnoDB buffer pool is meant to be *large* — 50–75% of RAM on a dedicated server — since it is the *only* cache. That single difference (buffered I/O + small pool for Postgres, direct I/O + huge pool for InnoDB) is one of the most common ways people mis-size a database after migrating between the two.

### Second-order: the cold cache after restart

A subtle operational cost of any large in-memory cache is that it is *empty after a restart*. A Postgres or MySQL instance that was serving sub-millisecond queries from a warm pool will, immediately after a restart or failover, serve every query from disk until the pool re-warms — and that cold period can mean minutes of 10–100× latency. InnoDB mitigates this with **buffer pool dump and restore** (`innodb_buffer_pool_dump_at_shutdown` / `_load_at_startup`), which persists the *list of page IDs* that were cached and reloads them on startup. Postgres has `pg_prewarm` for the same purpose — you can prewarm specific hot relations into `shared_buffers` (or the OS cache) right after startup. Forgetting this is why "the database is slow right after every deploy" is such a common, baffling complaint: the deploy restarted the instance and threw away the cache.

## 11. Putting it together: one row read, end to end

**Rule of thumb: trace any slow query down to the count of distinct pages it must touch and whether each is cached. That number, not the row count, is your latency.**

Let us assemble the whole stack into a single trip. You run `SELECT * FROM accounts WHERE id = 42`. Here is every layer the engine descends, with the format and the cache decision at each step.

![A pipeline tracing a single row read: SQL to B-tree descent to ctid to buffer pool lookup to disk miss to line pointer to row](/imgs/blogs/how-databases-store-data-pages-heap-files-buffer-pool-5.webp)

Figure 5 is the trace. Step by step:

1. **Plan.** The planner sees an equality on the primary key and chooses an index scan over `accounts_pkey`.
2. **Descend the index.** It reads the B-tree root, an internal node, and a leaf — typically ~3–4 page reads for a table of this size. *Each of those page reads is itself a buffer-pool lookup:* the root and upper internals are almost always cached (`usage_count` near 5, from section 6); the leaf may or may not be.
3. **Read the pointer.** The leaf entry for `id = 42` contains a `ctid`, say `(0, 1)` — block 0, slot 1. This is the `(block, slot)` indirection from section 2.
4. **Resolve through the buffer pool.** The engine hashes `(accounts, main, block 0)` into the buffer hash table. **Hit:** it gets a pinned pointer to the frame in nanoseconds and bumps `usage_count`. **Miss:** it runs the clock-sweep to find a victim frame, possibly fsyncs a dirty victim, issues a ~100 µs NVMe read for the 8 KB block, and installs it.
5. **Read the line pointer.** Inside block 0, slot 1's `ItemId` gives the byte offset and length of the tuple. The engine jumps to that offset.
6. **Check visibility.** It reads `t_xmin`/`t_xmax` from the 23-byte tuple header and tests them against the transaction's MVCC snapshot. If the version is not visible, it follows the `t_ctid` update chain to the visible version.
7. **Deform the tuple.** It walks the column data using the type catalog and alignment rules from section 4, skipping padding, consulting the NULL bitmap, and dereferencing any TOAST pointer (section 5) for wide columns the query actually selected.
8. **Return the row.**

The entire performance story is in steps 2 and 4: **how many distinct pages did this query have to fetch, and how many were already in the buffer pool?** A fully cached point lookup is a handful of hash-table probes and pointer chases — single-digit microseconds. The same query on a cold cache is 4–5 disk reads — hundreds of microseconds to milliseconds. The format work (steps 5–7) is real but tiny by comparison. This is why, when you profile a slow query, the first question is never "how many rows" — it is "how many *pages*, and were they hot?" `EXPLAIN (ANALYZE, BUFFERS)` answers exactly that:

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM accounts WHERE id = 42;
```

```
Index Scan using accounts_pkey on accounts  (cost=0.28..8.29 rows=1 width=45)
                                            (actual time=0.031..0.033 rows=1 loops=1)
  Index Cond: (id = 42)
  Buffers: shared hit=4
Planning Time: 0.084 ms
Execution Time: 0.052 ms
```

`Buffers: shared hit=4` is the whole story: four pages touched (three index, one heap), all cache hits, 52 microseconds. Change `hit=4` to `read=4` (a cold cache) and that same query is 100× slower. Read the `Buffers` line on every slow query; it is the ground truth the rest of this article was building toward.

## Case studies from production

### 1. The table that was 90% padding

A team stored time-series telemetry in a table declared `(ts timestamptz, sensor_id smallint, ok boolean, reading double precision, flags smallint)`. Storage was 40% larger than their back-of-envelope estimate and sequential scans were sluggish. The cause was alignment padding: `smallint` (2-byte align) and `boolean` (1-byte) interleaved with `timestamptz` and `double precision` (both 8-byte align) forced padding bytes between nearly every column. `pg_column_size` on a sample row showed 64 bytes for ~25 bytes of actual data. Reordering the columns widest-first — `(ts timestamptz, reading double precision, sensor_id smallint, flags smallint, ok boolean)` — dropped the row to 48 bytes, shrank the table 25%, and sped up the scans proportionally because each 8 KB page now held 25% more rows. The lesson: column order is a free 20–30% on wide tables of mixed-width types, and `CREATE TABLE` is the only convenient time to get it right.

### 2. The UUID primary key that fragmented InnoDB

A service migrated from Postgres to MySQL and kept its UUIDv4 primary keys. On Postgres, inserts had been fine — the heap appended, only the UUID index suffered. On InnoDB, throughput collapsed to a fifth of expected. The reason is the clustered index from section 7: in InnoDB the *table itself* is ordered by primary key, so a random UUID PK scatters every insert across the entire clustered B-tree, causing page splits everywhere and turning sequential inserts into random 16 KB page rewrites. The buffer pool thrashed because each insert dirtied a different, cold leaf page. The fix was a sequential surrogate `BIGINT AUTO_INCREMENT` primary key with the UUID demoted to a unique secondary index. Inserts went back to appending to the rightmost clustered-index page (hot, cached) and throughput recovered. This is the [random-UUID problem](/blog/software-development/database/random-uuids-are-killing-your-database-performance) amplified by the clustered design.

### 3. The checkpoint that stalled every 5 minutes

A write-heavy Postgres instance showed clockwork latency spikes: every five minutes, p99 jumped 20× for about 30 seconds. The metric that cracked it was `pg_stat_bgwriter`, showing `checkpoints_timed` firing every 5 minutes (the default `checkpoint_timeout`) and `buffers_checkpoint` flushing a huge dirty-page count in a tight burst. `shared_buffers` had been set aggressively at 60% of RAM, so an enormous number of dirty pages accumulated between checkpoints and then slammed the disk all at once. Two changes fixed it: lowering `shared_buffers` to 25% (fewer dirty pages to flush, and the freed RAM went to the OS cache, *improving* read performance) and setting `checkpoint_completion_target = 0.9` to spread the flush across the interval. The spikes flattened into a gentle, continuous trickle. The lesson: a big buffer pool is not free; it makes checkpoints heavier, and the default 5-minute timeout plus a default completion target is a recipe for I/O walls.

### 4. The TOAST table that dwarfed the heap

A product stored user-uploaded JSON documents in a `jsonb` column on a hot, frequently-updated table. The main table was 2 GB but `pg_total_relation_size` reported 180 GB. The hidden bulk was the TOAST relation: every update to *any* column of a row rewrote new TOAST chunks for the document (section 5's second-order hazard), and because the documents were large and updates frequent, dead TOAST chunks piled up faster than autovacuum could reclaim them. `SELECT *` queries were also slow because every row read detoasted and decompressed the full document even when the caller only wanted a status field. The fix was twofold: move the `jsonb` blob to a separate `document_bodies` table keyed by the parent id (so updates to the parent's hot columns never touch TOAST), and switch the column's compression to LZ4 for cheaper detoast. The 180 GB collapsed to 30 GB and the `SELECT` latency dropped because the hot table's pages were now dense with small rows.

### 5. The cold cache after every deploy

An e-commerce backend was "slow for the first two minutes after every deploy," then fine. Engineers chased application warmup, connection-pool ramp, JIT — none explained it. The real cause was that the deploy restarted the database container, emptying `shared_buffers` and the OS page cache. Every query for those two minutes hit cold disk: the hot product and inventory pages had to be re-read one by one. The fix was `pg_prewarm`: a startup hook ran `SELECT pg_prewarm('products'); SELECT pg_prewarm('inventory'); SELECT pg_prewarm('products_pkey');` to pull the known-hot relations into the buffer pool before traffic arrived. The cold period vanished. (On the MySQL side, `innodb_buffer_pool_dump_at_shutdown = ON` does this automatically by persisting the cached page list.) The lesson: an in-memory cache is a liability at startup, and warm-up must be part of the deploy, not an accident of traffic.

### 6. The DELETE that did not free any space

A team ran a nightly `DELETE` of millions of expired rows and was baffled that `pg_relation_size` never shrank. The DELETEs marked tuples dead (set `t_xmax`), but plain `VACUUM` only reclaims that space *for reuse within the same file* — it returns dead-tuple space to the free space map so new inserts can use it, but it does not return pages to the operating system unless the empty pages happen to be at the *end* of the file. Because the table kept getting new inserts that filled the reclaimed space, the file stayed the same size forever (steady state), which was actually fine — but they wanted the disk back after a one-time purge. The tool for that is `VACUUM FULL` or `pg_repack`, which physically rewrites the table into a new, dense file and drops the old one, also resetting the bloated line-pointer arrays from section 2's second-order note. The lesson: `DELETE` plus `VACUUM` reclaims space for the *database*, not for the *filesystem*; only a rewrite returns bytes to the OS.

### 7. The sequential scan that evicted the world (and didn't, on MySQL)

A reporting job ran a nightly full-table scan over a 200 GB archive table on a MySQL replica. On a naive-LRU cache this would have flushed the entire hot OLTP working set to make room for archive pages touched exactly once. It did not, and the reason is InnoDB's midpoint-insertion LRU from section 6: the scan's pages all landed in the *old sublist* and were evicted from there without ever being promoted into the young sublist, because they were never re-accessed within `innodb_old_blocks_time`. The hot OLTP pages in the young sublist were untouched. When the same pattern was tested on a Postgres replica, clock-sweep was weaker — the scan bumped each page's `usage_count` to 1, and a large scan did measurably cool the cache (Postgres mitigates this with a small "ring buffer" for large seq scans, capping their buffer footprint, but it is less aggressive than InnoDB's quarantine). The lesson: scan-resistance is a real, design-level property of the buffer manager, and it differs between engines — know which one you are running before you schedule heavy analytical scans against a transactional pool.

### 8. The HOT update rate that quietly collapsed

A user-session table with `(id, user_id, last_seen, payload jsonb)` updated `last_seen` on every request. It started fast — updates were HOT, touching no index — then degraded over weeks. `pg_stat_user_tables` showed `n_tup_hot_upd / n_tup_upd` had fallen from 95% to 30%. Two causes compounded: the table's pages had filled to fillfactor 100 (the default), so there was no room for in-place new versions, forcing each update to a new page and rewriting the `user_id` index; and autovacuum was throttled, so dead-tuple space (the room HOT needs) was not being reclaimed. The fix paired `ALTER TABLE sessions SET (fillfactor = 85)` plus a `pg_repack` to apply it, with aggressive per-table autovacuum settings (`autovacuum_vacuum_scale_factor = 0.02`). HOT rate climbed back to 90%+, index bloat stopped growing, and update latency halved. The lesson from section 8: fillfactor and autovacuum are a pair; the HOT path needs both free page space *and* prompt reclamation to stay on.

### 9. The `width` blow-up from a fat primary key in InnoDB

A MySQL schema used a `VARCHAR(255)` natural key (an email) as the clustered primary key. Every one of the table's six secondary indexes silently embedded a *full copy* of that email in every entry (section 7: InnoDB secondary indexes store the PK value). The secondary indexes were collectively larger than the table. Worse, comparisons in those indexes were string comparisons on long values. Switching to a compact `BIGINT` surrogate PK and making the email a unique secondary index shrank the secondary indexes by an order of magnitude and sped up every secondary lookup, because the embedded PK was now 8 bytes instead of up to 255. The lesson: in a clustered-index engine, the primary key is not just an identifier — it is replicated into every secondary index, so its *width* is a tax paid on every index in the table.

### 10. The page checksum that caught silent corruption

An operator enabled `data_checksums` on a Postgres cluster (it can be turned on at initdb or, in recent versions, online). Months later, a query failed with a checksum-mismatch error on a specific block. Rather than serving corrupt data silently, Postgres refused to return the page — the `pd_checksum` field in the header (section 2) did not match the page's computed checksum, meaning the bytes had been corrupted at rest, almost certainly by a failing disk. The corruption was isolated to a few blocks; the operator restored those from a backup and replaced the drive before the damage spread. Without checksums, the corrupt rows would have flowed silently into query results and replication. The lesson: the 2-byte `pd_checksum` in every page header is cheap insurance, and the page is the unit at which corruption is detected exactly because it is the unit of I/O — the same theme as the whole article.

### 11. The index-only scan that wasn't

A query that selected only indexed columns (`SELECT id, created_at FROM events WHERE created_at > now() - interval '1 day'`, with an index on `(created_at, id)`) was expected to be an index-only scan — never touching the heap. `EXPLAIN (ANALYZE, BUFFERS)` showed it hitting the heap on most rows anyway (`Heap Fetches: 84210`). The reason is the Visibility Map from section 3: an index-only scan can skip the heap *only* for pages marked all-visible in the `_vm`, and the table had high write churn, so most pages were not all-visible and the scan had to visit the heap to check `t_xmin`/`t_xmax` visibility. The fix was tuning autovacuum to run more often (vacuum is what sets the all-visible bits), after which `Heap Fetches` dropped to near zero and the query sped up several-fold. The lesson: index-only scans are a *cooperation* between the index, the visibility map, and vacuum — the format and the maintenance process are inseparable.

### 12. The `shared_buffers` that was too small to coalesce writes

The opposite of case study 3: a write-heavy ingestion service had `shared_buffers` set to a frugal 1 GB on a 64 GB box, on the theory that "the OS cache will handle it." Reads were fine (the OS cache was huge), but writes amplified badly — the same hot index pages were being read, dirtied, written, evicted, and re-read repeatedly because the small pool could not hold the write working set long enough to coalesce multiple updates into one disk write. Raising `shared_buffers` to 16 GB (25% of RAM) let the hot pages stay dirty in the pool across many updates, so a page updated dozens of times between checkpoints was written once instead of dozens of times. Write throughput tripled. The lesson balances case study 3: too-large hurts checkpoints, but too-small hurts write coalescing — the buffer pool must be big enough to hold the *write* working set, which is why 25% (not near-zero, not near-100%) is the durable default.

## When to reach for this knowledge — and when to skip it

You should descend to the page and buffer-pool layer when:

- **You are sizing or migrating a database.** `shared_buffers` vs `innodb_buffer_pool_size` are tuned in opposite directions (buffered + 25% vs direct + 70%); getting this wrong after a migration silently halves throughput.
- **Storage is bigger than your model predicts.** Padding, header overhead, TOAST churn, and bloat each explain a different 20–40% surprise, and `pg_column_size` / `pg_total_relation_size` / `pageinspect` localize which one.
- **Write throughput or update latency is the bottleneck.** HOT updates, fillfactor, write amplification, and the clustered-vs-heap fork are the levers, and they are invisible at the SQL layer.
- **You see periodic I/O spikes or post-restart slowness.** Checkpoints and cold caches are page-layer phenomena with page-layer fixes (`checkpoint_completion_target`, `pg_prewarm`, buffer-pool dump/restore).
- **You are debugging a "wrong" plan.** `EXPLAIN (ANALYZE, BUFFERS)` and the `Buffers:`/`Heap Fetches:` lines reason in pages; without the page model they are noise.

You should *not* spend time here when:

- **Your dataset fits comfortably in RAM and your load is light.** Below the point where page reads dominate, the engine's defaults are excellent and micro-tuning is wasted effort and a source of bugs.
- **You are reaching for `fillfactor` or column reordering on a read-mostly, append-only table.** Fillfactor below 100 just wastes space there, and there are no updates to make HOT.
- **You would trade clarity for a marginal byte.** Reordering columns for padding is free at `CREATE TABLE` time, but contorting a *schema's logical design* to save a few bytes per row is usually a bad trade against readability.
- **The real fix is a query or an index.** Most "slow database" problems are a missing index or an N+1 query, not a page-layout subtlety. Profile first; the page layer is the second or third stop, not the first.

The through-line of every section is one sentence: the page is the atom of I/O, and everything — the slotted layout, the tuple format, TOAST, the clustered-vs-heap fork, the buffer pool, fillfactor, checkpoints — is a strategy for touching fewer pages and keeping the ones you touch in memory. Once you can look at any operation and ask "how many distinct pages, and are they hot?", the database stops being a magic box and becomes a machine you can reason about.

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 3 ("Storage and Retrieval") — the canonical first-principles treatment of pages, page caches, B-trees, and LSM-trees.
- [PostgreSQL documentation: Database Page Layout](https://www.postgresql.org/docs/current/storage-page-layout.html) and [pageinspect](https://www.postgresql.org/docs/current/pageinspect.html) — the exact byte-level structures used throughout this article.
- [PostgreSQL documentation: TOAST](https://www.postgresql.org/docs/current/storage-toast.html) and [pg_buffercache](https://www.postgresql.org/docs/current/pgbuffercache.html).
- Jeremy Cole, ["The physical structure of InnoDB index pages"](https://blog.jcole.us/2013/01/07/the-physical-structure-of-innodb-index-pages/) — the definitive byte-level dissection of the 16 KB InnoDB page.
- [MySQL: Making the Buffer Pool Scan Resistant](https://dev.mysql.com/doc/refman/8.4/en/innodb-performance-midpoint_insertion.html) — the InnoDB midpoint-insertion LRU.
- Uber Engineering, ["Why Uber Engineering Switched from Postgres to MySQL"](https://www.uber.com/en-US/blog/postgres-to-mysql-migration/) — the write-amplification and ctid argument.
- [Crunchy Data: Postgres Performance Boost with HOT Updates and Fill Factor](https://www.crunchydata.com/blog/postgres-performance-boost-hot-updates-and-fill-factor).
- Sibling posts on this blog: [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work), [LSM trees](/blog/software-development/database/lsm-trees-write-optimized-storage-engines), [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance), and [MVCC deep dive: Postgres vs InnoDB](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb).

---
title: "Random UUIDs Are Killing Your Database Performance"
publishDate: "2026-03-15"
category: "software-development"
subcategory: "Database"
tags: ["database", "uuid", "performance", "indexing", "b-tree", "snowflake-id", "distributed-systems"]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
aiGenerated: true
image: "/imgs/blogs/random-uuids-are-killing-your-database-performance-20260315170553.png"
excerpt: "Switching from integer IDs to UUIDv4 can degrade write throughput by 20–90%. Understanding why — and how UUIDv7 and Twitter's Snowflake ID fix it — is critical for any system at scale."
---

## The Problem

You're building a system and decide to switch from integer IDs (`1, 2, 3…`) to UUIDs (`550e8400-e29b-41d4-a716-446655440000`) for security or distributed ID generation. Everything works fine at small scale, but as data grows, database writes start getting slower — sometimes **much slower**.

Why? To understand this, we need to go deep into how databases store and organize data at the lowest level.

## Foundations: Disk, Pages, and the Buffer Pool

Before talking about indexes, we need to understand how a database interacts with hardware.

### Data is stored on disk in "pages"

Databases don't read or write individual rows. Instead, data is organized into **pages** (typically 4KB, 8KB, or 16KB depending on the database engine). Each page holds multiple rows.

```
┌─────────────────────────────────────────────────┐
│                   Page (8KB)                     │
├─────────────────────────────────────────────────┤
│  Row 1: id=1, name="Alice", email="a@mail.com" │
│  Row 2: id=2, name="Bob",   email="b@mail.com" │
│  Row 3: id=3, name="Carol", email="c@mail.com" │
│  ...                                            │
│  Row N: id=N, name="...",   email="..."         │
│  [Free Space]                                    │
└─────────────────────────────────────────────────┘
```

When the database needs to read a single row, it must load the **entire page containing that row** into memory.

### Buffer Pool — The database's cache

The database maintains a region of RAM called the **Buffer Pool** to cache frequently accessed pages. When a page needs to be read:

1. Check if the page is in the Buffer Pool → **Cache hit** (fast, ~nanoseconds)
2. If not → **Cache miss** → Read from disk (slow, ~milliseconds)

```
┌──────────────────────────────────────────────────┐
│                  Buffer Pool (RAM)                │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │
│  │ Page 5 │ │ Page 12│ │ Page 99│ │ Page 3 │    │
│  └────────┘ └────────┘ └────────┘ └────────┘    │
│                                                   │
│  Cache hit → ~100ns    Cache miss → ~10ms         │
│  Difference: ~100,000x                            │
└──────────────────────────────────────────────────┘
```

The difference between a cache hit and a cache miss is roughly **100,000x**. This is why the **access pattern** matters more than almost anything else.

## B-Tree: The Data Structure Behind Indexes

Most database indexes use **B-Trees** (balanced trees). This is a balanced tree data structure where each node corresponds to a page on disk.

### B-Tree Structure

```
                     ┌─────────────────┐
                     │   Root Node     │
                     │  [50] [100]     │
                     └──┬──────┬──────┬┘
                        │      │      │
            ┌───────────┘      │      └───────────┐
            ▼                  ▼                   ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ Internal Node │  │ Internal Node │  │ Internal Node │
    │ [20] [35]     │  │ [70] [85]     │  │ [120] [150]   │
    └──┬────┬────┬──┘  └──┬────┬────┬──┘  └──┬─────┬───┬──┘
       │    │    │        │    │    │        │     │    │
       ▼    ▼    ▼        ▼    ▼    ▼        ▼     ▼    ▼
     ┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐   ┌───┐┌────┐┌───┐
     │1-19││20 ││36 │  │51 ││71 ││86 │   │101││121 ││151│
     │   ││-34││-49│  │-69││-84││-99│   │-119││-149││-N │
     └───┘└───┘└───┘  └───┘└───┘└───┘   └───┘└────┘└───┘
      Leaf Pages        Leaf Pages         Leaf Pages
```

Each **leaf page** contains the actual data (or pointers to data), sorted by key. To find a record, the database traverses from root to leaf, comparing keys at each level to choose the correct branch.

### Why B-Trees Matter

With 1 million records, a B-Tree needs only **~3-4 page reads** to locate any record (logarithmic lookup). But what matters even more than lookup is the **insert pattern** — how new data gets added to the tree.

## Sequential IDs: The Perfect Fit for B-Trees

When you use auto-increment integers (`1, 2, 3, 4…`), each new ID is always **larger than all previous IDs**. This means inserts always happen at the **rightmost leaf page**.

### Step-by-step example

Assume each leaf page holds a maximum of 4 records:

```
Step 1: Insert id=1, 2, 3, 4
┌──────────────────┐
│ Leaf Page 1      │
│ [1] [2] [3] [4]  │  ← Page full, next page will be created
└──────────────────┘

Step 2: Insert id=5, 6, 7, 8
┌──────────────────┐  ┌──────────────────┐
│ Leaf Page 1      │→ │ Leaf Page 2      │
│ [1] [2] [3] [4]  │  │ [5] [6] [7] [8]  │  ← Always appending to the right
└──────────────────┘  └──────────────────┘

Step 3: Insert id=9, 10, 11, 12
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Leaf Page 1      │→ │ Leaf Page 2      │→ │ Leaf Page 3      │
│ [1] [2] [3] [4]  │  │ [5] [6] [7] [8]  │  │ [9] [10] [11] [12] │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Why is this fast?**

- **Only 1 "hot" page**: The database only needs to keep the rightmost leaf page in the Buffer Pool. Every insert is a cache hit.
- **Sequential I/O**: When a page fills up and a new one is created, the new page is written adjacently on disk → sequential write (the fastest possible).
- **100% fill factor**: Each page is completely filled before moving to the next → zero wasted space.
- **Predictable**: The OS and disk controller can prefetch data because the access pattern is predictable.

## Random UUIDv4: Why It Destroys Performance

UUIDv4 values are **uniformly random**. For example:

```
Insert 1: 7c9e6679-7425-40de-944b-e07fc1f90ae7
Insert 2: 109156be-c4fb-41ea-b1b4-efe1671c5836
Insert 3: f47ac10b-58cc-4372-a567-0e02b2c3d479
Insert 4: 2c5ea4c0-4067-11e9-8bad-9b1deb4d3b15
Insert 5: 9a7b41cd-3e90-4c93-b7e1-2a7f8b9e5c1a
```

These UUIDs have no ordering. When sorted in a B-Tree, they scatter across the entire index.

### Problem 1: Random I/O — "Jumping around" on disk

Consider what happens when a database has 1 million records with a UUIDv4 index, and you insert 5 more records:

```
B-Tree Index (1 million records, thousands of leaf pages)

Insert "109156be..." → must write to Page 847
Insert "2c5ea4c0..." → must write to Page 2,103
Insert "7c9e6679..." → must write to Page 5,891
Insert "9a7b41cd..." → must write to Page 7,234
Insert "f47ac10b..." → must write to Page 11,502

Each insert → load a different page from disk → 5 random disk reads
```

Compare with sequential IDs: 5 inserts → all go to the same page → 0 disk reads (page already in cache).

When the index is larger than the Buffer Pool (e.g., 10GB index but only 4GB Buffer Pool), most pages will **not be in cache**. Each insert almost certainly causes a **cache miss → disk read**.

### Problem 2: Page Splitting — Splitting in half, wasting space

When a leaf page is already full and a new record needs to be inserted in the middle, the database must **split the page**:

```
BEFORE split:
┌──────────────────────────────────────┐
│ Leaf Page 847  (FULL)                │
│ [0fa3...] [10e2...] [1234...] [15ab...]│
└──────────────────────────────────────┘

Insert "109156be..." → needs to go between "10e2..." and "1234..."
But the page is full! → SPLIT

AFTER split:
┌──────────────────────────────────────┐
│ Leaf Page 847  (50% full)            │
│ [0fa3...] [10e2...]                  │
│ [     empty      ]                   │
└──────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────┐
│ Leaf Page 12,001 (50% full)          │ ← New page, possibly at a
│ [109156be...] [1234...] [15ab...]    │   completely different disk location
│ [     empty      ]                   │
└──────────────────────────────────────┘
```

**Consequences of page splits:**

- **2 pages instead of 1**: Doubles the number of pages needed to store the same amount of data.
- **50% utilization**: Each page uses only ~50% of its capacity → wastes RAM and disk.
- **3 disk writes**: Rewrite the old page + write the new page + update the parent node.
- **New page may be far away on disk**: Creates physical fragmentation.

### Problem 3: The "Swiss Cheese" Effect — A bloated, hole-filled index

After millions of inserts with UUIDv4, the index becomes "Swiss cheese":

```
Sequential ID index (1M records):
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
│100%││100%││100%││100%││100%││100%││100%││100%│
│full││full││full││full││full││full││full││full│
└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘
→ Compact, fewer pages, fits in RAM

UUIDv4 index (1M records):
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
│ 52%││ 78%││ 45%││ 61%││ 50%││ 83%││ 47%││ 70%││ 55%││ 49%││ 66%││ 58%│
│full││full││full││full││full││full││full││full││full││full││full││full│
└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘
→ Bloated, more pages, overflows RAM
```

**Real-world numbers:**

Suppose you have 10 million records, each page is 8KB, and each page holds ~100 records:

| Metric                   | Sequential ID | UUIDv4               |
| ------------------------ | ------------- | -------------------- |
| Pages required           | 100,000       | ~160,000 (~62% fill) |
| Index size               | ~800MB        | ~1.28GB              |
| Fits in 1GB Buffer Pool? | Yes (100%)    | No (only ~78%)       |
| Cache miss rate          | ~0%           | ~20-40%              |

When the index doesn't fit in RAM, each insert has a high probability of causing disk I/O. At thousands of inserts per second, performance **collapses**.

### Real-world benchmark

A common PostgreSQL benchmark with 50 million rows shows:

```
Throughput (inserts/second):

Sequential INT:  ████████████████████████████████████████  45,000 ops/s
UUIDv7:          ███████████████████████████████████████   42,000 ops/s
UUIDv4:          ████████████                              12,000 ops/s
                 ↑
                 Index exceeds RAM at ~20M rows,
                 performance begins to degrade sharply
```

UUIDv4 is **~3.7x slower** than sequential integers in this case. At larger scale or with slower disks, degradation can reach **20-90%**.

## UUIDv7: The Best of Both Worlds

**UUIDv7** is standardized in [RFC 9562](https://www.rfc-editor.org/rfc/rfc9562) and solves the problem by embedding a **timestamp at the beginning** of the UUID.

### UUIDv7 Structure

```
UUIDv7 format (128 bits):
┌──────────────────────────────────────────────────────────┐
│ 48-bit timestamp │ 4-bit │ 12-bit  │ 2-bit  │ 62-bit   │
│  (milliseconds)  │ ver=7 │ rand_a  │ variant│ rand_b   │
└──────────────────────────────────────────────────────────┘

Example UUIDv7s generated consecutively:
  2026-03-15 10:00:00.001 → 019e1a2b-3c01-7d4e-8f12-3a4b5c6d7e8f
  2026-03-15 10:00:00.002 → 019e1a2b-3c02-7a1b-9c2d-4e5f6a7b8c9d
  2026-03-15 10:00:00.003 → 019e1a2b-3c03-7f8e-a1b2-c3d4e5f6a7b8
                               ▲▲▲▲▲▲▲▲▲▲▲▲
                               Timestamp increases!
```

Because the **first 48 bits are a timestamp** (millisecond precision), UUIDs generated over time automatically **sort in increasing order** when compared lexicographically. This means they behave identically to sequential integers in a B-Tree.

### Why UUIDv7 Is Fast

```
Insert pattern with UUIDv7:

Timeline: 10:00:00 ──────────────────────────► 10:00:01

UUID 019e1a2b-3c01... → Leaf Page N (rightmost)
UUID 019e1a2b-3c02... → Leaf Page N (rightmost) ← same page!
UUID 019e1a2b-3c03... → Leaf Page N (rightmost) ← same page!
UUID 019e1a2b-3c04... → Leaf Page N (rightmost) ← same page!
...page full...
UUID 019e1a2b-3cFF... → Leaf Page N+1 (new rightmost) ← new page, adjacent

→ Identical to sequential integers!
→ 1 hot page in cache
→ Sequential I/O
→ 100% fill factor
```

### Comparison overview

| Property                            | Sequential INT                 | UUIDv4         | UUIDv7        |
| ----------------------------------- | ------------------------------ | -------------- | ------------- |
| Distributed generation              | No (needs central counter)     | Yes            | **Yes**       |
| Monotonic inserts (B-Tree friendly) | Yes                            | No             | **Yes**       |
| Prevents ID enumeration             | No (`/user/101` → `/user/102`) | Yes            | **Partial**   |
| Reveals creation time               | No                             | No             | Yes           |
| Index fragmentation                 | None                           | Severe         | **Minimal**   |
| Size (bytes)                        | 4-8                            | 16             | 16            |
| Index size overhead                 | Baseline                       | ~1.6-2x larger | ~1x (compact) |

### Code examples

**PostgreSQL:**

```sql
-- PostgreSQL 17+ supports UUIDv7 via extensions
-- UUIDv7 can be generated at the application layer

CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- ❌ UUIDv4
    -- replace with:
    id UUID PRIMARY KEY,                             -- ✅ UUIDv7 generated by app
    customer_id UUID NOT NULL,
    total DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Python:**

```python
import uuid

# ❌ UUIDv4 — random, causes fragmentation
old_id = uuid.uuid4()
# → 7c9e6679-7425-40de-944b-e07fc1f90ae7

# ✅ UUIDv7 — time-sorted, B-Tree friendly (Python 3.14+)
# Or use the uuid7 library:
# pip install uuid7
import uuid7
new_id = uuid7.create()
# → 019e1a2b-3c01-7d4e-8f12-3a4b5c6d7e8f
```

**Node.js:**

```javascript
// ❌ UUIDv4
import { v4 as uuidv4 } from "uuid";
const oldId = uuidv4();

// ✅ UUIDv7
import { v7 as uuidv7 } from "uuid";
const newId = uuidv7();
// → "019e1a2b-3c01-7d4e-8f12-3a4b5c6d7e8f"
```

### Trade-offs to consider

UUIDv7 is not a perfect solution for every case:

- **Timestamp leakage**: The ID reveals when a record was created. If this is sensitive information (e.g., when a user signed up), consider encrypting or obfuscating the ID at the application layer before exposing it externally.
- **Not fully random**: Unlike UUIDv4, UUIDv7 is partially predictable. An attacker can't guess exact IDs, but can estimate the creation time window.
- **Same millisecond**: If multiple requests arrive within the same millisecond on the same node, the random portion ensures uniqueness but ordering between them is undefined — in practice, this is rarely a problem.

## Snowflake ID: The Alternative from Twitter for Massive Distributed Systems

While UUIDv7 solves the B-Tree problem elegantly, it still uses 128 bits (16 bytes) — twice the size of a `BIGINT`. In 2010, Twitter faced a different challenge: they needed to generate **unique IDs across hundreds of servers** handling tens of thousands of tweets per second, but they wanted something smaller and more efficient than UUID.

Their solution: **Snowflake ID** — a 64-bit (8-byte) ID that packs a timestamp, machine identifier, and sequence counter into a single `long` integer.

### The Structure of a Snowflake ID

Snowflake uses exactly **64 bits**, the same size as a `BIGINT`/`long` type on 64-bit systems. But it's not storing a plain integer — it encodes three pieces of information using **bit packing**:

```
64 bits total:
┌───┬──────────────────────────────────────────┬──────────┬──────────────┐
│ 0 │           Timestamp (41 bits)            │Machine ID│ Sequence ID  │
│   │         milliseconds since epoch         │ (10 bits)│  (12 bits)   │
└───┴──────────────────────────────────────────┴──────────┴──────────────┘
 1b                    41b                         10b          12b
```

- **1 bit (sign)**: Always 0 — ensures the ID is a positive number. This is important because many languages and databases treat signed/unsigned integers differently. Keeping it positive avoids subtle bugs.
- **41 bits (timestamp)**: Milliseconds since a custom epoch (not Unix epoch). 41 bits can represent $2^{41} - 1 = 2,199,023,255,551$ milliseconds ≈ **69.7 years**. Twitter chose a custom epoch of November 4, 2010, meaning Snowflake IDs will work until approximately 2080.
- **10 bits (machine ID)**: Identifies which server generated this ID. 10 bits = $2^{10} = 1024$ possible machines. This is further split into 5 bits for datacenter ID (32 datacenters) and 5 bits for worker ID (32 workers per datacenter).
- **12 bits (sequence)**: A counter that increments within the same millisecond on the same machine. 12 bits = $2^{12} = 4096$ unique IDs per millisecond per machine. When the next millisecond arrives, the counter resets to 0.

### How Bit Packing Works

To pack three values into a single 64-bit `long`, Snowflake uses **bit shifting** — the most efficient approach at the CPU level:

```java
long id = (timestamp << 22)   // shift left 22 bits (10 + 12) to make room
        | (machineId << 12)   // shift left 12 bits to make room for sequence
        | sequenceId;         // fill in the last 12 bits
```

**Step-by-step visualization:**

```
Suppose: timestamp = 1710000000000 (some moment in time)
         machineId = 5
         sequenceId = 42

Step 1: timestamp << 22
  Binary: ...0001100011101001100000010101000000000000 0000000000000000000000
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^
                                                      22 zero bits (room for machine + sequence)

Step 2: machineId << 12
  Binary: ...0000000000000000000000000000000000000000 0000000101 000000000000
                                                      ^^^^^^^^^^
                                                      machineId=5  ^^^^^^^^^^^^
                                                                   12 zero bits

Step 3: Bitwise OR (|) all together
  = timestamp_shifted | machineId_shifted | sequenceId
  → A single 64-bit number encoding all three values
```

**Why bit shifting instead of string concatenation or multiplication?**

| Method | Operation | CPU Cycles | Notes |
|--------|-----------|------------|-------|
| Bit shifting | `<<` and `\|` | 1-2 cycles | Direct register manipulation |
| Multiplication | `timestamp * 4194304 + machineId * 4096 + seq` | 3-6 cycles | Some modern compilers optimize `* 4096` to `<< 12` |
| String concatenation | `"" + timestamp + machineId + seq` | 100+ cycles | Memory allocation, parsing overhead |

Bit shifting is the clear winner — it's a single CPU instruction that completes in one clock cycle. In modern compilers, multiplication by powers of 2 may be automatically converted to bit shifts, but it's better to be explicit.

### Extracting Values Back from a Snowflake ID

You can decode a Snowflake ID back into its components just as efficiently:

```java
long timestamp  = (id >> 22) + CUSTOM_EPOCH;  // shift right 22 bits
long machineId  = (id >> 12) & 0x3FF;         // shift right 12, mask to 10 bits
long sequenceId = id & 0xFFF;                 // mask to 12 bits
```

This means you can **extract the creation timestamp directly from the ID** — no need to query a `created_at` column or maintain a secondary index. This is incredibly powerful for time-based queries.

### Why Snowflake ID Solves All the Problems

Let's compare how Snowflake ID addresses the issues with both auto-increment integers and UUIDs:

**Problem 1: ID collision in distributed systems (auto-increment's weakness)**

With auto-increment, if you have 3 database servers each generating IDs independently, they'll all produce `1, 2, 3, 4...` — instant collision. Common workarounds (even/odd split, range allocation) are brittle and hard to scale.

Snowflake eliminates this entirely: as long as each server has a unique `machineId`, they can generate IDs independently at full speed with **zero coordination**. Server A (`machineId=1`) and Server B (`machineId=2`) will never produce the same ID, even if they generate at the exact same millisecond:

```
Server A (machineId=1), timestamp=T, sequence=0:
  = (T << 22) | (1 << 12) | 0 = ...some unique number

Server B (machineId=2), timestamp=T, sequence=0:
  = (T << 22) | (2 << 12) | 0 = ...a different unique number
```

**Problem 2: B-Tree fragmentation (UUID's weakness)**

Because the timestamp occupies the **most significant bits** (leftmost), Snowflake IDs are naturally **time-ordered**. IDs generated later are always numerically larger than IDs generated earlier. This means inserts always go to the rightmost leaf page — identical to auto-increment behavior:

```
Snowflake IDs generated over time:

t=0ms:  Server1: 000...0001_00000_000000000001  →  small number
t=1ms:  Server2: 000...0010_00001_000000000001  →  slightly larger
t=2ms:  Server1: 000...0011_00000_000000000001  →  even larger

→ Always increasing → Always appending to B-Tree rightmost → No page splits!
```

**Problem 3: Information leakage (auto-increment's weakness)**

With `id=12345`, an attacker knows there are at most 12,345 records, can enumerate `/api/user/12344` and `/api/user/12346`, and can estimate growth rate. Snowflake IDs look like large, seemingly random numbers (e.g., `1778027252434944001`) that don't reveal the total count and can't be trivially enumerated — though the timestamp is extractable if you know the epoch.

**Problem 4: Storage overhead (UUID's weakness)**

| Type | Size | B-Tree Pages (10M rows) | Index fits in 1GB RAM? |
|------|------|------------------------|----------------------|
| INT (32-bit) | 4 bytes | ~50,000 pages | Yes |
| BIGINT / Snowflake | 8 bytes | ~80,000 pages | Yes |
| UUID (128-bit) | 16 bytes | ~160,000 pages | Barely |

Snowflake ID uses exactly half the storage of UUID while encoding more useful information. Over billions of records, this translates to terabytes of saved storage and significantly better cache utilization.

### Real-World Adoption: Discord's Case Study

Discord handles **billions of messages** across millions of servers. They chose Snowflake ID as the primary key for their message storage in Cassandra (later migrated to ScyllaDB). Here's why this was a brilliant choice:

**Time-range queries without secondary indexes:**

When a user scrolls up in a channel to load older messages, Discord needs to fetch messages in a time range. With Snowflake IDs, they can compute the approximate ID range directly:

```python
# To find messages after a specific date:
target_time_ms = 1710000000000  # some timestamp
min_snowflake_id = (target_time_ms - DISCORD_EPOCH) << 22

# Query: WHERE message_id > min_snowflake_id
# → Uses the primary key index directly
# → No secondary index on created_at needed
# → Extremely fast, even across billions of rows
```

This eliminates the need for a separate `created_at` index — saving storage, reducing write amplification, and making reads faster. For a system with billions of rows, this optimization alone saves significant infrastructure cost.

**Partition key design:**

Discord uses the Snowflake ID's embedded timestamp to derive the partition key (channel_id + time bucket), ensuring messages from the same channel and time period are stored together on disk — maximizing sequential read performance.

### Implementation Example

Here's a complete Snowflake ID generator:

```java
public class SnowflakeIdGenerator {
    // Custom epoch: 2020-01-01T00:00:00Z
    private static final long EPOCH = 1577836800000L;

    private static final int MACHINE_ID_BITS = 10;
    private static final int SEQUENCE_BITS = 12;

    private static final long MAX_MACHINE_ID = (1L << MACHINE_ID_BITS) - 1; // 1023
    private static final long MAX_SEQUENCE = (1L << SEQUENCE_BITS) - 1;     // 4095

    private final long machineId;
    private long lastTimestamp = -1L;
    private long sequence = 0L;

    public SnowflakeIdGenerator(long machineId) {
        if (machineId < 0 || machineId > MAX_MACHINE_ID) {
            throw new IllegalArgumentException(
                "Machine ID must be between 0 and " + MAX_MACHINE_ID);
        }
        this.machineId = machineId;
    }

    public synchronized long nextId() {
        long timestamp = System.currentTimeMillis() - EPOCH;

        if (timestamp == lastTimestamp) {
            // Same millisecond: increment sequence
            sequence = (sequence + 1) & MAX_SEQUENCE;
            if (sequence == 0) {
                // Sequence overflow: wait for next millisecond
                timestamp = waitNextMillis(lastTimestamp);
            }
        } else {
            // New millisecond: reset sequence
            sequence = 0;
        }

        lastTimestamp = timestamp;

        return (timestamp << (MACHINE_ID_BITS + SEQUENCE_BITS))
             | (machineId << SEQUENCE_BITS)
             | sequence;
    }

    private long waitNextMillis(long lastTimestamp) {
        long timestamp = System.currentTimeMillis() - EPOCH;
        while (timestamp <= lastTimestamp) {
            timestamp = System.currentTimeMillis() - EPOCH;
        }
        return timestamp;
    }
}
```

```python
# Python implementation
import time
import threading


class SnowflakeIdGenerator:
    EPOCH = 1577836800000  # 2020-01-01T00:00:00Z in ms

    def __init__(self, machine_id: int):
        if not (0 <= machine_id <= 1023):
            raise ValueError("machine_id must be between 0 and 1023")
        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = -1
        self._lock = threading.Lock()

    def next_id(self) -> int:
        with self._lock:
            timestamp = int(time.time() * 1000) - self.EPOCH

            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF  # 4095 max
                if self.sequence == 0:
                    # Wait for next millisecond
                    while timestamp <= self.last_timestamp:
                        timestamp = int(time.time() * 1000) - self.EPOCH
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            return (
                (timestamp << 22)
                | (self.machine_id << 12)
                | self.sequence
            )

    @staticmethod
    def extract_timestamp(snowflake_id: int) -> int:
        """Extract the creation timestamp (ms since epoch) from a Snowflake ID."""
        return (snowflake_id >> 22) + SnowflakeIdGenerator.EPOCH

    @staticmethod
    def extract_machine_id(snowflake_id: int) -> int:
        return (snowflake_id >> 12) & 0x3FF

    @staticmethod
    def extract_sequence(snowflake_id: int) -> int:
        return snowflake_id & 0xFFF


# Usage
gen = SnowflakeIdGenerator(machine_id=1)
id1 = gen.next_id()  # e.g., 1778027252434944001
id2 = gen.next_id()  # e.g., 1778027252434948097

# Decode it back
print(SnowflakeIdGenerator.extract_timestamp(id1))   # → epoch ms
print(SnowflakeIdGenerator.extract_machine_id(id1))   # → 1
print(SnowflakeIdGenerator.extract_sequence(id1))      # → 0
```

### Snowflake ID Trade-offs

Despite its elegance, Snowflake ID has real limitations you should consider:

**1. No native database support — must generate at the application layer**

Unlike `AUTO_INCREMENT` or `gen_random_uuid()`, no major database provides built-in Snowflake ID generation. You must generate IDs in your application code before inserting. This means:
- Every application instance needs a Snowflake generator
- You need a mechanism to assign unique `machineId` to each instance (e.g., environment variable, service registry, or Zookeeper)
- More code to maintain, test, and debug

**2. BIGINT required instead of INT**

Snowflake IDs require `BIGINT` (8 bytes) instead of `INT` (4 bytes). For tables with billions of rows and multiple indexes, this doubles the primary key storage. In practice, this is rarely a problem — the overhead is small compared to the row data itself, and modern databases handle BIGINT efficiently.

**3. Client-side parsing challenges with 64-bit integers**

This is a real gotcha. JavaScript's `Number` type uses IEEE 754 double-precision floating point, which can only represent integers exactly up to $2^{53} - 1$ (9,007,199,254,740,991). Snowflake IDs can exceed this:

```javascript
// ❌ JavaScript loses precision with large Snowflake IDs
const id = 1778027252434944001;
console.log(id); // → 1778027252434944000 (WRONG! last digits lost)

// ✅ Solution: transmit as string in JSON
const response = { "message_id": "1778027252434944001" };
// Or use BigInt
const id = BigInt("1778027252434944001");
```

This is exactly why Twitter's API (and Discord's API) return Snowflake IDs as **strings** in JSON responses, with a separate `id_str` field. Any API using Snowflake IDs must do the same.

**4. Clock drift sensitivity**

Snowflake depends on the system clock being monotonically increasing. If the clock moves backward (due to NTP sync, leap seconds, or VM migration), the generator could produce duplicate IDs or must halt until the clock catches up. Production systems need clock monitoring and fallback strategies.

### Snowflake vs. UUIDv7: Head-to-Head

| Property | Snowflake ID | UUIDv7 |
|----------|-------------|--------|
| Size | **8 bytes** | 16 bytes |
| Time-ordered | Yes | Yes |
| B-Tree friendly | Yes | Yes |
| Distributed | Yes (via machineId) | Yes (via randomness) |
| Native DB support | No | Growing (some DBs support) |
| Encodes machine info | **Yes** | No |
| Max IDs/ms/machine | 4,096 | Practically unlimited |
| Standard | De facto (Twitter/Discord) | **RFC 9562** |
| JSON-safe in JavaScript | No (needs string) | **Yes** (already a string) |
| Lifespan | ~69 years from epoch | ~281 trillion years |

**Bottom line**: Use Snowflake ID when you need compact storage, machine traceability, and operate at massive scale (billions of records). Use UUIDv7 when you want standards compliance, broader ecosystem support, and simpler implementation.

## When to Use What

| Scenario                             | Recommendation                    | Reason                                            |
| ------------------------------------ | --------------------------------- | ------------------------------------------------- |
| Single DB, internal system           | Auto-increment integer            | Simplest, fastest, smallest                       |
| Distributed system, public API       | **UUIDv7**                        | Distributed generation + B-Tree friendly          |
| Massive scale (billions of rows)     | **Snowflake ID**                  | Compact (8 bytes), machine-traceable, time-sorted |
| Need randomness for security tokens  | UUIDv4 (but **not as PK**)        | Use UUIDv4 for tokens, UUIDv7/Snowflake for PK   |
| Legacy system using UUIDv4           | Migrate or add a sequential index | Add a `created_at` index to improve range queries |
| Multi-region, need global uniqueness | UUIDv7 or Snowflake ID            | Both provide timestamp + uniqueness guarantees    |
| Need to extract timestamp from ID    | **Snowflake ID**                  | Bit extraction is O(1), no index needed           |

## Key Takeaway

The core issue isn't UUID vs. integer — it's the **access pattern on the B-Tree**. Random key → random I/O → page split → index bloat → performance collapse.

You have two modern solutions:
- **UUIDv7**: Standards-compliant, 128-bit, drop-in UUID replacement with time-ordering. Best for most applications.
- **Snowflake ID**: Compact 64-bit, encodes machine identity, battle-tested at Twitter/Discord scale. Best for massive distributed systems where storage efficiency and machine traceability matter.

Both solve the fundamental problem by turning random inserts into sequential inserts. If you're using UUIDv4 as a primary key and your database is getting slow, switching to either of these is the highest-impact change you can make.

## References

1. [RFC 9562 - Universally Unique IDentifiers (UUIDs)](https://www.rfc-editor.org/rfc/rfc9562)
2. [UUID, serial or identity columns for PostgreSQL auto-generated primary keys? - CyberTec](https://www.cybertec-postgresql.com/en/uuid-serial-or-identity-columns-for-postgresql-auto-generated-primary-keys/)
3. [The effect of Random UUID on database performance - PlanetScale](https://planetscale.com/blog/the-problem-with-using-a-uuid-primary-key-in-mysql)
4. [Twitter Snowflake - GitHub (archived)](https://github.com/twitter-archive/snowflake)
5. [How Discord Stores Billions of Messages](https://discord.com/blog/how-discord-stores-billions-of-messages)
6. [Announcing Snowflake - Twitter Engineering Blog](https://blog.twitter.com/engineering/en_us/a/2010/announcing-snowflake)

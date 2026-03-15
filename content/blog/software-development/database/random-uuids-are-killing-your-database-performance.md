---
title: "Random UUIDs Are Killing Your Database Performance"
publishDate: "2026-03-15"
category: "software-development"
subcategory: "Database"
tags: ["database", "uuid", "performance", "indexing", "b-tree"]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
aiGenerated: true
image: "/imgs/blogs/random-uuids-are-killing-your-database-performance-20260315170553.png"
excerpt: "Switching from integer IDs to UUIDv4 can degrade write throughput by 20–90%. Understanding why — and how UUIDv7 fixes it — is critical for any system at scale."
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

## When to Use What

| Scenario                             | Recommendation                    | Reason                                            |
| ------------------------------------ | --------------------------------- | ------------------------------------------------- |
| Single DB, internal system           | Auto-increment integer            | Simplest, fastest, smallest                       |
| Distributed system, public API       | **UUIDv7**                        | Distributed generation + B-Tree friendly          |
| Need randomness for security tokens  | UUIDv4 (but **not as PK**)        | Use UUIDv4 for tokens, UUIDv7 for PK              |
| Legacy system using UUIDv4           | Migrate or add a sequential index | Add a `created_at` index to improve range queries |
| Multi-region, need global uniqueness | UUIDv7 + region prefix            | Timestamp + randomness ensures uniqueness         |

## Key Takeaway

The core issue isn't UUID vs. integer — it's the **access pattern on the B-Tree**. Random key → random I/O → page split → index bloat → performance collapse. UUIDv7 solves this by turning random inserts into sequential inserts, preserving the benefits of UUIDs (distributed, non-enumerable) without paying the performance penalty.

If you're using UUIDv4 as a primary key and your database is getting slow, **switching to UUIDv7** is the simplest change with the biggest impact.

## References

1. [RFC 9562 - Universally Unique IDentifiers (UUIDs)](https://www.rfc-editor.org/rfc/rfc9562)
2. [UUID, serial or identity columns for PostgreSQL auto-generated primary keys? - CyberTec](https://www.cybertec-postgresql.com/en/uuid-serial-or-identity-columns-for-postgresql-auto-generated-primary-keys/)
3. [The effect of Random UUID on database performance - PlanetScale](https://planetscale.com/blog/the-problem-with-using-a-uuid-primary-key-in-mysql)

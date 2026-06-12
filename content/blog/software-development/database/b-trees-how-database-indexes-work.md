---
title: "B-Trees: How Database Indexes Really Work, From First Principles to Code"
date: "2026-06-11"
publishDate: "2026-06-11"
description: "A first-principles tour of the B+ tree that powers almost every database index — the disk-page math, a working from-scratch implementation, how InnoDB and Postgres actually use it, and the query patterns it serves or sabotages."
tags:
  [
    "b-tree",
    "b-plus-tree",
    "database",
    "indexing",
    "storage-engine",
    "innodb",
    "postgres",
    "lsm-tree",
    "query-optimization",
    "data-structures",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 51
---

Ask ten engineers what a database index is and most will say "a sorted list" or "a hash map." Both answers are wrong in the way that matters. A sorted array gives you binary search but turns every insert into an O(n) memmove. A hash map gives you O(1) point lookups but cannot answer `WHERE created_at BETWEEN ... AND ...`, cannot serve `ORDER BY`, and degrades into chaos the moment your working set spills out of RAM. Real databases reach for neither. They reach for a **B+ tree** — a structure shaped less by Big-O than by a single brutal fact: the data lives on a disk that is read one fixed-size page at a time, and a page read costs roughly a hundred thousand times more than a comparison.

Everything strange about B-trees — the hundreds of children per node, the values hiding only in the leaves, the obsession with how full each page is — falls out of that one constraint. Once you internalize "one node is one page, and a page read is the expensive thing," the whole design stops looking like a textbook curiosity and starts looking inevitable.

![A B+ tree drawn top to bottom: a router root, two internal nodes, and four linked leaves holding the data](/imgs/blogs/b-trees-how-database-indexes-work-1.webp)

The diagram above is the mental model for this entire post. A B+ tree has three kinds of structure. The **root** and **internal** nodes hold nothing but *separator keys* — they are a routing table, a decision tree that tells you which child to descend into. The **leaf** nodes, at the bottom, hold the actual data (or pointers to it), and — this is the part the textbooks under-sell — they are **chained left to right** in a linked list. That chain is why a B+ tree can answer a range query by descending once and then walking sideways, never touching the upper levels again. Hold this picture; the rest of the article is a tour through it.

## Why an index isn't what you think

Before we build anything, let's name the mismatch directly. Almost every wrong intuition about indexes comes from reasoning as if memory were free and uniform. It isn't.

| You assume | The naive mental model | The reality |
| --- | --- | --- |
| An index is a sorted array | Binary search, O(log n) compares | Inserts would memmove millions of bytes; arrays are unusable for write workloads |
| An index is a hash map | O(1) lookups, done | No range scans, no `ORDER BY`, no prefix matches, terrible on disk |
| Lookups cost "log n comparisons" | Compares are the unit of cost | The unit of cost is the **disk page read**; compares are free by comparison |
| A deeper tree is fine, it's still log n | Height doesn't matter much | Every level is potentially one disk seek; height *is* latency |
| Indexes just make reads faster | Pure win | Every index is a second tree every write must maintain; they cost write throughput and disk |
| The structure is RAM-resident | Pointers chase freely | The tree is mostly on disk; only the hot upper levels stay cached |

The single most important row is the third. When you analyze a binary search tree you count comparisons, and you conclude that 30 comparisons (for a billion rows) is nothing. But on a database the relevant question is not "how many comparisons" — it is "how many *pages* did I have to pull off disk." A comparison is a few nanoseconds. A page read from an SSD is tens of microseconds; from a spinning disk, ten milliseconds. That is a five- to six-order-of-magnitude gap, and it reorders every design decision. The whole game of a B-tree is to make the number of page reads as small as possible, even if that means doing *more* comparisons inside each page.

> The first rule of storage engineering: count disk reads, not comparisons. Everything else is rounding error.

## Why not a binary tree or a hash map? {#why-not-a-binary-tree}

**Rule of thumb: a data structure tuned for RAM is almost always the wrong choice on disk, because it optimizes the cheap resource and ignores the expensive one.**

Take a balanced binary search tree — a red-black tree, say. It is a beautiful structure with O(log₂ n) operations. For a billion rows, log₂(10⁹) ≈ 30. Thirty comparisons. If the tree lived entirely in RAM, you would be done in well under a microsecond.

But put that tree on disk. Each node holds one key and two child pointers — maybe 40 bytes. Your storage engine reads in 8 KB pages, so each node read pulls an 8 KB page to retrieve 40 useful bytes, and worse, consecutive nodes on a root-to-leaf path are scattered all over the file. Following the tree from root to a leaf means up to 30 page reads, each one a potential disk seek, because there is no reason two nodes one hop apart in the tree are anywhere near each other on disk. Thirty seeks at 10 ms each is 300 ms to find one row. That is a catastrophe.

![A matrix showing binary-tree height growing to thirty levels while a fanout-256 B-tree stays at four, with one disk read per lookup](/imgs/blogs/b-trees-how-database-indexes-work-2.webp)

The matrix above is the entire argument for B-trees in one picture. The amber column is a binary tree's height: ~10 levels at a thousand rows, ~20 at a million, ~30 at a billion. The blue column is a B-tree's height at a fanout of 256: two, three, four. The green column is the punchline — the number of disk reads per point lookup stays at **one**, because (as we'll see) the upper levels are small enough to live permanently in the buffer pool, so only the leaf touch actually goes to disk.

The trick is **fanout**. Instead of two children per node, a B-tree node has *hundreds*. The height of the tree is `log_b(n)` where `b` is the fanout, and `log` shrinks fast as `b` grows. With `b = 256`, a billion rows is `log₂₅₆(10⁹) ≈ 3.7`, so four levels. Four page reads in the absolute worst case, and in practice one, versus thirty for the binary tree. We traded "fewer comparisons per node" (now we binary-search *within* a 256-key node) for "drastically fewer page reads," and on disk that trade is a landslide.

A hash map has the opposite problem. It gives genuine O(1) point lookups, and key-value stores like the hash index in older MySQL `MEMORY` tables or the in-memory structures we cover in [Redis in Production](/blog/software-development/database/redis-applications-and-optimization) use exactly this. But a hash destroys order. There is no "next key" in a hash table; the whole point of hashing is to scatter keys uniformly. So `WHERE price BETWEEN 10 AND 20`, `ORDER BY created_at`, `WHERE name LIKE 'sm%'`, and `MIN(value)` all become full scans. A B-tree keeps keys in sorted order at every level, which is what makes all of those queries cheap. That is why the *default* index type in Postgres, MySQL/InnoDB, SQLite, Oracle, and SQL Server is a B-tree, and hash indexes are a niche specialization.

It's worth noting the B-tree didn't win by being the only ordered structure — it won by being the best *fit for disk*. **Skip lists** give similar `O(log n)` ordered operations and are simpler to implement lock-free (Redis sorted sets and parts of LevelDB's memtable use them), but their pointer-chasing, node-per-element layout is a poor match for page-oriented storage — they shine in memory, not on disk. **AVL and red-black trees** are binary, so they suffer the exact height-equals-seeks problem we just described. **Radix trees and tries** avoid comparisons but waste space on sparse key spaces and don't naturally give range order across the whole key. The B-tree's specific combination — high fanout to crush height, sorted keys for range queries, fixed-size nodes matched to pages, and balance maintained by cheap local splits — is uniquely suited to the "data on disk, read a page at a time" regime that has defined databases for fifty years. As storage shifts to NVMe and persistent memory the constants change, and there's active research into cache- and flash-optimized variants (Bw-trees, fractal and B-ε trees), but the core idea has proven remarkably durable.

## Anatomy: nodes, order, and fanout {#anatomy}

**Rule of thumb: a node is not a logical abstraction you get to size freely — it is exactly one page, and that page size dictates everything else.**

Let's get concrete about what lives inside a node. The defining parameter of a B-tree is its **order** (sometimes called the branching factor or fanout): the maximum number of children an internal node may have. An order-`b` tree obeys a few invariants that together keep it balanced:

- Every node holds at most `b − 1` keys and at most `b` child pointers.
- Every node except the root holds at least `⌈b/2⌉ − 1` keys (the "half-full" rule).
- All leaves sit at the same depth — the tree is perfectly height-balanced, always.
- The keys within a node are sorted, and they partition the key space for the children.

That half-full rule is what guarantees the tree never degenerates: a node may not drop below ~50% occupancy, so the height stays `O(log_b n)` no matter the insert order. (We'll see in a later section that insert *order* still matters enormously for how full pages settle in practice, even though the worst-case height is bounded.)

![One internal-node page and one leaf-node page laid out slot by slot, showing keys, pointers, free space, and the next-leaf pointer](/imgs/blogs/b-trees-how-database-indexes-work-3.webp)

The figure shows the two page types byte-for-byte. The **internal node page** (top) is a page header followed by an alternating sequence of child pointers and separator keys: `ptr₀, key₁, ptr₁, key₂, ptr₂, …`. To find the child for a search key `k`, you binary-search the keys and follow the pointer between the two keys that bracket `k`. With ~16-byte entries in an 8 KB page, that's room for roughly 500 separators — hence fanout in the hundreds. The **leaf node page** (bottom) is different: it's a header, then `(key, row-or-rowpointer)` slots packed in sorted order, then free space, and critically a **next-leaf pointer** at the end that chains this leaf to its right sibling.

This is the heart of the "one node is one page" idea, and it's why fanout is not a tuning knob you set arbitrarily:

$$
\text{fanout} \approx \frac{\text{page size}}{\text{entry size}} = \frac{8\,\text{KB}}{\sim 16\,\text{B}} \approx 512
$$

You don't choose 512; it falls out of dividing the page size by the entry size. If your keys are bigger — say you index a 64-byte composite key — the entry size grows, fanout drops, and the tree gets taller. This is a real, measurable cost and the reason "index the narrowest key that answers your queries" is good advice. A `BIGINT` primary key (8 bytes) gives you a fatter, shallower tree than a `UUID` (16 bytes) or a `VARCHAR(255)` natural key, and the difference shows up as fewer levels and fewer cache misses on every single lookup.

A worked example makes the height concrete. Say each leaf holds 400 rows and each internal node fans out to 500:

- Level 0 (root): 1 node, routes to 500 children.
- Level 1: 500 nodes, route to 250,000 children.
- Level 2: 250,000 nodes, route to 125,000,000 leaves.
- Level 3 (leaves): 125,000,000 leaves × 400 rows = **50 billion rows**.

A four-level B-tree addresses fifty billion rows. That is the entire reason this structure won.

### Key compression: squeezing more fanout from the same page

Fanout is so valuable that engines fight for it with **key compression**. Two tricks dominate. **Prefix compression** notices that adjacent keys in a node often share a long common prefix — think `user:10001`, `user:10002`, `user:10003` — and stores the shared prefix once per page instead of on every key, so each entry shrinks and more fit per page. **Suffix truncation** (for the separator keys in internal nodes) keeps only as much of a key as is needed to route correctly: to separate `"smith"` from `"smithson"`, the internal node only needs to store `"smithso"`, not the full key, because a separator is a signpost, not data. Both tricks push effective fanout up without changing the page size, and both are why a real index on a long text key is often far shallower than the naive `page_size / full_key_size` estimate suggests. The lesson still stands, though: narrow keys give fatter, shallower trees, and compression only partially rescues a badly chosen wide key. When you're deciding between an 8-byte integer key and a 40-byte natural key, compression narrows the gap but never closes it.

## B-tree vs B+ tree {#b-tree-vs-b-plus-tree}

**Rule of thumb: when someone says "B-tree" in a database context, they almost always mean a B+ tree — the distinction is small on paper and enormous in practice.**

The classic B-tree (Bayer & McCreight, 1972) stores *values* in every node — internal and leaf alike. If a key lives in the root, its value lives in the root too, and a lookup for that key stops at the root. That sounds like a feature: hot keys near the top resolve in one hop. The B+ tree (a later refinement) makes a different choice: internal nodes hold **only keys**, no values, and **all** values live in the leaves. Every lookup, hot or cold, descends all the way to a leaf.

![Before-after comparison: a B-tree scatters values across all nodes with no leaf links; a B+ tree keeps keys-only internal nodes and linked leaves](/imgs/blogs/b-trees-how-database-indexes-work-4.webp)

Why does every production engine pick the B+ variant, even though it makes hot-key lookups slightly longer? Two reasons, both visible in the figure.

First, **fanout**. If internal nodes don't carry values, each internal entry is just `(key, child-pointer)` — small. More entries fit per page, so fanout is higher and the tree is shorter. In a plain B-tree, internal nodes carry values too, which bloats each entry and lowers fanout. Since the entire game is minimizing height, the B+ tree's keys-only internal nodes are a direct win on the metric that matters.

Second, **range scans**. Because all values live in linked leaves, a range query in a B+ tree descends once to the first matching leaf and then *walks the sibling pointers* — `leaf.next → leaf.next → …` — reading every qualifying row in sorted order without ever climbing back up the tree. In a classic B-tree, the values for a range are scattered across internal and leaf nodes at every level, so a range scan becomes an in-order traversal that bounces up and down the tree, touching far more pages. Range scans, `ORDER BY`, `GROUP BY`, and `LIMIT` are bread-and-butter database operations, and the B+ tree makes all of them a flat leaf-chain walk. That is why the leaf chain in our mental-model figure is drawn so prominently: it is the single most important structural feature of the index your database actually runs.

From here on, when I write "B-tree" I mean the B+ tree, exactly as your database's documentation does.

You'll occasionally meet the **B\* tree**, a variant that delays splits by first trying to redistribute keys to a sibling — splitting two full nodes into three rather than one full node into two — which pushes average occupancy from ~50% toward ~66% under random inserts. It's a real improvement on paper, but most production engines don't bother, because the lazy-merge, fill-factor, and bulk-load machinery we'll discuss addresses occupancy more pragmatically, and the extra split-time coordination costs concurrency. For working purposes there are exactly two trees that matter: the B+ tree your database actually runs, and everything else as historical context.

## The three operations: search, insert, delete {#the-three-operations}

**Rule of thumb: searches are trivial; the entire engineering of a B-tree is in keeping the tree balanced under inserts and deletes without ever letting a node overflow or starve.**

**Search** is the easy one. Start at the root, binary-search the keys to pick a child, descend, repeat until you hit a leaf, binary-search the leaf. The cost is the tree height in page reads plus a handful of in-page binary searches. Done.

**Insert** is where it gets interesting, because a leaf can fill up. When you insert into a leaf that is already at its maximum of `b − 1` keys, the leaf **splits**: it divides into two half-full leaves, and the median key is promoted (copied, for a leaf) up to the parent as a new separator. But the parent might now be full too — so it splits and promotes a key to *its* parent, and so on. In the worst case the split propagates all the way to the root, the root itself splits, and a brand-new root is created above it. That is the only way a B-tree ever grows taller: **from the bottom up, by a root split.**

![A five-step timeline of an overflowing insert: a full leaf splits, the median is pushed up, the parent overflows and splits, and the root splits to add a level](/imgs/blogs/b-trees-how-database-indexes-work-5.webp)

The timeline walks the whole cascade. We insert key 95 into a leaf that already holds four keys (step 1). The leaf splits into two half-full nodes (step 2) and the median key 72 is pushed up into the parent (step 3). If that push overflows the parent, the parent splits and recurses upward (step 4). If the recursion reaches the root and the root splits, the tree gains a whole new level (step 5). This bottom-up growth is what keeps every leaf at exactly the same depth — the tree can only get taller at the root, so it stays perfectly balanced for free. There is no rebalancing pass, no rotations like a red-black tree; balance is a structural consequence of how splits propagate.

**Delete** is the mirror image. When a deletion drops a node below the half-full threshold, the node must **rebalance**: either it *borrows* a key from a sibling that has spare capacity (rotating a key through the parent), or, if no sibling can spare one, it **merges** with a sibling into a single node, pulling a separator key down from the parent. A merge can leave the parent underfull, so the merge propagates upward exactly as splits do — and if the root ends up with a single child, the root is removed and the tree shrinks by a level. In practice, many production engines are lazy about merges: they tolerate underfull pages and reclaim them in background maintenance rather than rebalancing eagerly on every delete, because eager merging under concurrent writers is expensive and contentious. We'll see the consequences of that laziness — index bloat — in the case studies.

To make delete concrete: suppose an order-5 tree (each node holds 2–4 keys) has a leaf `[40, 50]` and we delete 50, leaving `[40]` — below the minimum of two keys. The engine first looks at a sibling. If the left sibling is `[10, 20, 30]` (three keys, one to spare), it **borrows**: 30 rotates up to replace the parent separator, the old separator drops into our leaf, and we end with `[30, 40]` — both nodes legal, no merge. If instead the sibling is also at minimum (`[10, 20]`), borrowing would starve it, so the two **merge** into `[10, 20, 40]`, the separator between them is pulled down from the parent, and the parent loses a key — which may cascade the merge upward exactly as a split cascades. The symmetry is exact: a split pushes a key up and can grow the tree by a level; a merge pulls a key down and can shrink it by a level.

#### Concurrency: many writers share one tree

Everything above assumed a single thread. Real databases run thousands of concurrent queries against the same B-tree, and the structure has to stay consistent while it is being split and merged under your feet. The mechanism is **latching** — short-term, physical locks on individual pages, distinct from the logical row **locks** that implement transactions. The vocabulary matters: in database internals, a "lock" is a transactional, logical lock held for the duration of a transaction; a "latch" is a brief mutex on an in-memory page held for microseconds. Confusing them is a classic interview tell.

The naive approach — lock the whole tree for every operation — serializes all writers and destroys throughput. The classic refinement is **latch coupling** (also called *crabbing*): descending the tree, you latch a child before releasing the parent, moving down hand-over-hand like a crab, so you never hold more than two latches at once. For reads this is cheap. For writes it is trickier, because a split can propagate upward, so a writer may need to hold latches on a chain of ancestors until it is sure the child won't split into them. The dominant modern solution is the **B-link tree** (Lehman & Yao, 1981), which adds a right-link pointer at *every* level (not just the leaves) plus a high-key per node, so a reader who arrives at a node mid-split can detect it and follow the right-link to the correct sibling *without* holding a latch on the parent. B-link trees are why production B-trees achieve high write concurrency — and they are exactly why case study 10's right-most-leaf hotspot is about *latch* contention, not *lock* contention: the page is in memory and uncontended for its data, but every inserter wants the same page latch at the same instant.

## Implementing a B+ tree from scratch

Theory is cheap. Let's build a working B+ tree in Python — search, insert with splitting, and a range scan that walks the leaf chain. This is real, runnable code, not pseudocode; paste it into a file and it works. I'm using `bisect` for the in-node binary search so the structure of the algorithm isn't buried under hand-rolled loops.

Start with the node and the search path:

```python
import bisect
from typing import Any, Optional


class Node:
    __slots__ = ("keys", "children", "leaf", "next")

    def __init__(self, leaf: bool) -> None:
        self.keys: list = []           # separators (internal) or sort keys (leaf)
        self.children: list = []       # child Nodes (internal) or values (leaf)
        self.leaf: bool = leaf
        self.next: Optional["Node"] = None   # leaf -> right sibling (the chain)


class BPlusTree:
    def __init__(self, order: int = 4) -> None:
        # order = max children per node; a node may hold at most order-1 keys.
        self.root = Node(leaf=True)
        self.order = order

    def _find_leaf(self, key: Any) -> Node:
        node = self.root
        while not node.leaf:
            # first child whose separator is > key
            i = bisect.bisect_right(node.keys, key)
            node = node.children[i]
        return node

    def search(self, key: Any) -> Optional[Any]:
        leaf = self._find_leaf(key)
        i = bisect.bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.children[i]
        return None
```

`_find_leaf` is the routing logic from the mental-model figure: at each internal node, `bisect_right` finds the child pointer that owns the key's range, and we descend until we hit a leaf. `search` then does one more binary search inside the leaf. The whole method is `O(height)` node visits, which on a real engine is `O(height)` page reads.

Now the part that does the work — inserting with splits that propagate upward:

```python
    def insert(self, key: Any, value: Any) -> None:
        self._insert(self.root, key, value)
        if len(self.root.keys) > self.order - 1:
            # root overflowed: grow a new level on top (the only way height grows)
            new_root = Node(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

    def _insert(self, node: Node, key: Any, value: Any) -> None:
        if node.leaf:
            i = bisect.bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                node.children[i] = value          # key exists -> update in place
                return
            node.keys.insert(i, key)
            node.children.insert(i, value)
            return

        i = bisect.bisect_right(node.keys, key)
        child = node.children[i]
        self._insert(child, key, value)
        if len(child.keys) > self.order - 1:       # child overflowed -> split it
            self._split_child(node, i)

    def _split_child(self, parent: Node, i: int) -> None:
        child = parent.children[i]
        mid = len(child.keys) // 2
        right = Node(leaf=child.leaf)

        if child.leaf:
            # leaf split: the median key is COPIED up and stays in the right leaf
            right.keys = child.keys[mid:]
            right.children = child.children[mid:]
            child.keys = child.keys[:mid]
            child.children = child.children[:mid]
            right.next = child.next                # splice into the leaf chain
            child.next = right
            separator = right.keys[0]
        else:
            # internal split: the median key MOVES up (it leaves both children)
            separator = child.keys[mid]
            right.keys = child.keys[mid + 1:]
            right.children = child.children[mid + 1:]
            child.keys = child.keys[:mid]
            child.children = child.children[:mid + 1]

        parent.keys.insert(i, separator)
        parent.children.insert(i + 1, right)
```

The one subtlety that trips people up is the difference between a **leaf split** and an **internal split**, and it is exactly the B+ tree property we discussed. On a leaf split the median is *copied* up — it stays in the right leaf because the leaf still has to hold the value. On an internal split the median *moves* up — it's a pure separator, so it leaves both children entirely. Get this wrong and you either lose keys or duplicate them. Notice also the two lines that splice `right` into the leaf chain (`right.next = child.next; child.next = right`); forget them and your range scans silently skip half the table.

Finally, the payoff for all that leaf-chaining — a range scan that descends exactly once:

```python
    def range_scan(self, lo: Any, hi: Any) -> list[tuple]:
        leaf = self._find_leaf(lo)
        out: list[tuple] = []
        while leaf is not None:
            for k, v in zip(leaf.keys, leaf.children):
                if k < lo:
                    continue
                if k > hi:
                    return out                     # past the range; stop
                out.append((k, v))
            leaf = leaf.next                        # walk the chain, no re-descent
        return out
```

This is the method that justifies the entire B+ design. We descend to the leaf containing `lo` — `O(height)` page reads — and then we walk `leaf.next` pointers, reading consecutive leaves in sorted order until we pass `hi`. We never climb back into the tree. A range that spans a thousand rows costs the descent plus however many leaves those rows occupy, read sequentially. Swap `range_scan` onto a hash index and it's impossible; swap it onto a classic B-tree and it bounces up and down the tree instead of walking flat.

A quick sanity check you can run:

```python
tree = BPlusTree(order=4)
for k in [10, 20, 5, 6, 12, 30, 7, 17, 95, 72, 50, 88, 1, 2, 3]:
    tree.insert(k, f"row-{k}")

print(tree.search(72))          # -> "row-72"
print(tree.search(999))         # -> None
result = [k for k, _ in tree.range_scan(6, 30)]
print(result)                   # [6, 7, 10, 12, 17, 20, 30] -- sorted, via the leaf chain
```

The range scan comes back perfectly sorted across multiple leaves, which only works because the leaf chain is intact. That is the whole structure, in about ninety lines. A production engine adds page serialization, a buffer pool, latching for concurrency, write-ahead logging, and variable-length keys — but the algorithm is exactly what you just read.

A few things the ninety lines above deliberately skip, so you know what production code adds. **Deletion with rebalancing** — the borrowing and merging we walked through — roughly doubles the code and is where most hand-rolled B-trees have bugs; many real engines (and our toy) get away with *lazy deletion*, tolerating underfull pages and reclaiming them later. **Serialization**: real nodes are packed byte layouts on 8 KB pages, not Python objects, with a slot directory mapping logical key order to physical offsets so an insert doesn't memmove the whole page. **The buffer pool**: nodes are referenced by page ID and fetched through a cache, not chased by object pointer. **Latching and write-ahead logging**: every structural change is logged before it's applied, so a crash in the middle of a split is recoverable on restart. And **bulk loading**: when you build an index on an existing table, engines don't insert keys one at a time — they sort all the keys first and build the tree bottom-up, packing leaves to the fill factor in a single pass, which is why `CREATE INDEX` on a big table is far faster than the equivalent row-by-row inserts *and* produces a denser tree.

## How real databases actually use it {#how-databases-use-it}

**Rule of thumb: the difference between a clustered index and a secondary index is whether the leaf holds the row or just a pointer to it — and that one difference drives half of all surprising query plans.**

A from-scratch B+ tree stores a value next to each key. Real databases are more interesting, and the details differ between engines in ways that bite you if you don't know them.

In **MySQL/InnoDB**, the table *is* a B+ tree. The primary key is the **clustered index**: its leaves hold the entire row, sorted by primary key. There is no separate heap. This is why InnoDB primary-key lookups are so fast — finding the key *is* finding the row, in one descent. But it has a consequence for **secondary indexes** (any non-primary index): a secondary index's leaves do not hold the row, they hold the **primary key value**. So a lookup by a secondary index finds the PK, then performs a *second* descent into the clustered index to fetch the row.

![A graph showing a primary-key lookup hitting the clustered tree directly while a secondary-index lookup lands on the PK and pays a second hop](/imgs/blogs/b-trees-how-database-indexes-work-6.webp)

The graph makes the asymmetry concrete. A `WHERE id = 42` query on the primary key descends the clustered B+ tree once and lands on the full row — one logical lookup. A `WHERE email = x` query on a secondary index descends the secondary tree, finds that the leaf holds PK 42 (not the row), and then does a **bookmark lookup**: a second descent into the clustered tree keyed by PK to actually fetch the row. That second hop is invisible in your SQL but very visible in your latency, and it's the reason a secondary-index point lookup in InnoDB costs roughly twice a primary-key one.

**Postgres** makes the opposite choice. Its tables are **heaps** — unordered piles of rows in pages — and *every* index, including the primary key, is a separate B+ tree whose leaves hold a `ctid`, a physical `(page, offset)` pointer into the heap. There is no clustered index by default; the primary key is just another B-tree pointing into the heap. This means Postgres has no "second hop asymmetry" between primary and secondary indexes — they all do one index descent plus one heap fetch — but it pays for that uniformity elsewhere: because the heap is unordered, a range scan that returns many rows can scatter heap fetches randomly across the table, and Postgres has a whole machinery (the visibility map, index-only scans, `CLUSTER`, bitmap heap scans) to mitigate it.

This clustered-vs-heap distinction explains a startling number of real behaviors:

- **Covering indexes** matter more in InnoDB, because avoiding the bookmark lookup (by putting every selected column in the secondary index) turns two descents into one.
- **Primary key choice** matters enormously in InnoDB, because the PK is duplicated into *every* secondary index's leaves. A 16-byte UUID primary key bloats every secondary index by 16 bytes per row; an 8-byte `BIGINT` halves that.
- **Postgres `HOT` updates and bloat** behave the way they do because indexes point at physical heap positions, so moving a row (an update) can require touching every index.

Two engine-specific optimizations are worth knowing because they show up constantly in query plans. In Postgres, an **index-only scan** can answer a query entirely from the index *if* the needed columns are all present — but Postgres still has to confirm each row is visible to your transaction, which it does via the **visibility map**, a bitmap marking heap pages where every row is visible to everyone. If the table is freshly vacuumed the visibility map is set and the index-only scan never touches the heap; if it's stale, the "index-only" scan secretly does heap fetches anyway. This is why `VACUUM` affects read performance, not just bloat. In InnoDB, the **change buffer** softens the random-write cost of secondary indexes: instead of immediately applying a secondary-index update that would require reading a random leaf page from disk, InnoDB buffers the change in memory and merges it later when the page is read for another reason, batching scattered random writes into sequential ones. It's a small LSM-like trick bolted onto a B-tree engine — a reminder that the B-tree-versus-LSM line is blurrier in practice than in the textbook.

It helps to see the three most common engines side by side, because the same word — "index" — means materially different things across them:

| | MySQL / InnoDB | PostgreSQL | SQLite |
| --- | --- | --- | --- |
| Table storage | Clustered B+ tree on the PK | Heap (unordered) | Clustered B+ tree on `rowid` |
| Secondary-index leaf holds | Primary-key value | Heap `ctid` | `rowid` |
| Extra hop for secondary lookup | Yes — bookmark lookup via PK | Heap fetch via `ctid` | Yes — `rowid` lookup |
| Range-scan locality | Excellent (rows sorted by PK) | Poor unless `CLUSTER`-ed | Excellent |
| Bloat mitigation | Change buffer, purge threads | `VACUUM`, visibility map | `VACUUM`, auto-vacuum |

The table is a cheat sheet for a surprising amount of behavior. "Why is my Postgres range query doing random heap I/O?" — heap storage; look at the locality row. "Why does my InnoDB secondary lookup cost double?" — bookmark lookup; look at the extra-hop row. "Why did `WITHOUT ROWID` change my SQLite performance?" — it turns a secondary-style `rowid` table into a true clustered index. Same structure underneath; three different sets of trade-offs layered on top.

The vector-search world, covered in [Vector Databases](/blog/machine-learning/ai-agent/vector-database), is the photo-negative of all this: there, the query is "find the nearest neighbors in a 1536-dimensional space," order is meaningless, and a B-tree is exactly the wrong tool. B-trees own *exact* and *range* queries on a totally-ordered key; similarity search needs an entirely different index family (HNSW, IVF). Knowing which problem you have is half of choosing the right index.

## Why four levels covers a billion rows {#why-four-levels}

**Rule of thumb: a point lookup on a billion-row table costs one disk read, not four — because the top three levels of the tree are small enough to live permanently in RAM.**

We keep saying a four-level tree handles a billion rows. Let's make the *cost* of that lookup precise, because the naive count ("four levels means four disk reads") is wrong, and the reason it's wrong is the single most important performance fact about B-trees.

![A layered stack: the root and internal levels stay pinned in the buffer pool while only the leaf level lives on disk and costs one read](/imgs/blogs/b-trees-how-database-indexes-work-7.webp)

The stack shows where each level physically lives. Walk it top to bottom:

- The **root** is a single page. It is read on the very first query after startup and then never leaves the buffer pool. One page, always cached.
- **Internal level 1** is ~128 pages (at fanout 128). That's about 1 MB. It is touched on essentially every query and stays resident. Cached.
- **Internal level 2** is ~16,000 pages — ~128 MB. On any database with a reasonably sized buffer pool, this is mostly cached too, because it's small relative to the data and constantly accessed.
- The **leaf level** is ~2,000,000 pages — gigabytes. This *cannot* all fit in cache, and it's where the cold data lives. A point lookup's final hop, to the specific leaf holding your row, is the one access likely to actually miss the buffer pool and hit disk.

So the real cost of a point lookup on a huge B-tree is: **zero disk reads for the upper levels** (they're cached) **plus one disk read for the leaf**. Not four. The tree's shallowness isn't just about minimizing total levels — it's about making the cacheable upper portion small enough that it always stays hot, so only the unavoidable leaf access pays for disk. This is also why the buffer-pool hit ratio is the metric DBAs watch obsessively, and why "is your working set bigger than RAM" is the question that predicts whether a database falls off a performance cliff. As long as the upper levels and the hot leaves fit in RAM, the index behaves like an in-memory structure. The moment the *hot leaf set* exceeds the buffer pool, every query starts paying for disk and latency collapses — the same cache-miss dynamic we dissect in [Random UUIDs Are Killing Your Database Performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance).

A quick numeric example. Suppose an 8 KB page, a buffer pool of 16 GB, and 200-byte rows:

- The whole upper structure (root + two internal levels) is ~130 MB — trivially cached.
- 16 GB of buffer pool, minus the ~130 MB of upper levels, leaves ~15.8 GB for leaves — about 2 million leaf pages, ~800 million rows' worth of *hot* leaves.
- As long as your queries concentrate on a working set smaller than that, you're effectively running an in-memory index with one occasional disk read. Spread your access uniformly over 5 billion rows and you'll miss the cache on nearly every leaf — same tree, 100× the latency.

What decides *which* leaves stay hot is the buffer pool's eviction policy, typically a variant of **LRU** (least-recently-used), often refined to resist scan pollution — InnoDB uses a midpoint-insertion LRU that protects the hot set so a single large table scan can't flush everything useful out of cache. The practical upshot is that a B-tree's performance is *bimodal*. While your hot set fits in the buffer pool, you get cache-speed lookups and the tree's height barely matters. The moment it doesn't, you fall off a cliff into disk-bound latency, and *then* every extra level — every byte of key width that cost you fanout — shows up as another potential miss on the way down. This is the real reason the seemingly academic obsession with fanout and key width matters: it doesn't change the best case much, but it sets *where the cliff is*.

## The cost of disorder {#the-cost-of-disorder}

**Rule of thumb: insert order is not cosmetic. Monotonic keys keep an index dense and cache-friendly; random keys shred it into half-empty pages and thrash the buffer pool.**

Here is the most expensive thing about B-trees that nobody warns you about. The worst-case *height* is bounded regardless of insert order — but the *density* of the pages, and therefore the real-world size and cache behavior of the index, depends enormously on the order in which you insert keys.

![Before-after: random keys split pages everywhere and settle half-full while sequential keys fill the right-most leaf and stay dense](/imgs/blogs/b-trees-how-database-indexes-work-8.webp)

The figure contrasts the two regimes. Consider inserting with a **monotonically increasing key** — an auto-increment integer, a Snowflake ID, a UUIDv7 (which is time-ordered). Every new key is larger than every existing key, so every insert lands in the **right-most leaf**. That leaf fills up, splits once cleanly (the old leaf stays full, a new empty leaf opens to the right), and the pattern repeats. Pages end up **~100% full**, the index is as small as it can be, and the working set is a tiny window at the right edge of the tree — perfectly cache-friendly.

Now insert with a **random key** — a UUIDv4, a hash, a random token. Each insert lands in a *random* leaf somewhere in the middle of the tree. That leaf is probably already reasonably full, so the insert triggers a **page split**: the leaf divides into two half-full leaves. Do this across the whole key space and your pages settle at roughly **50% occupancy** — meaning the index is nearly *twice* the size it needs to be, holds half as many rows per cached page, and the constant splitting writes far more pages than the data alone would require (write amplification). Worse, because inserts touch random leaves all over the tree, the *entire* leaf level is your working set, so the buffer pool thrashes and the cache-hit ratio collapses.

This is not a small effect. Switching a high-write table's primary key from an auto-increment integer to a UUIDv4 routinely degrades insert throughput by 20–90% and roughly doubles index size, and it's the entire subject of [Random UUIDs Are Killing Your Database Performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance). The fix is to use a *time-ordered* identifier (UUIDv7, ULID, Snowflake) so inserts stay monotonic, or to keep an integer clustered key and put the UUID in a secondary index where its randomness costs less.

The knob databases give you here is **fill factor** (Postgres) or **`innodb_fill_factor` / merge thresholds** (MySQL): the percentage a page is filled to during index build or bulk load. A fill factor of 100 packs pages completely — ideal for a read-only or append-only table, terrible for one that gets in-place updates, because the first update that grows a row has no room and forces a split. A fill factor of 70 leaves 30% slack so updates can happen in place without splitting. The right value depends entirely on your write pattern: high fill for append-mostly, lower fill for update-heavy.

| Insert pattern | Page occupancy | Index size | Cache behavior | Write amplification |
| --- | --- | --- | --- | --- |
| Monotonic (auto-inc, UUIDv7, Snowflake) | ~100% | Minimal | Hot window at right edge | Low — one clean split per fill |
| Random (UUIDv4, hash) | ~50% | ~2× larger | Whole leaf level is working set | High — constant mid-tree splits |
| Update-heavy with fill factor 70 | ~70% | ~1.4× larger | Moderate | Low — updates fit in slack |

There's a subtle second-order effect worth internalizing: random inserts don't just bloat the index, they bloat it *unpredictably*, because page splits happen at random moments as random leaves happen to fill. That makes write latency spiky — most inserts are fast, but the unlucky one that triggers a split (and maybe a cascade up several levels) is much slower, so your p99 write latency diverges from your p50 even though the *average* looks fine. Monotonic inserts split predictably — one clean split each time the right-most leaf fills — and keep the latency distribution tight. So when you're chasing a tail-latency regression on a write path, "are our keys random?" is one of the first questions to ask, because it explains a p99 problem that averages and medians will hide from you.

## B-tree vs LSM-tree {#b-tree-vs-lsm}

**Rule of thumb: B-trees update in place and optimize reads; LSM-trees append and compact and optimize writes. Choose by which one your workload does more of.**

The B-tree is not the only storage-engine architecture. Its main rival is the **log-structured merge tree (LSM)**, which powers RocksDB, LevelDB, Cassandra, ScyllaDB, and the write path of many time-series and analytics systems. Understanding the contrast is the fastest way to understand what a B-tree is *for*.

![A matrix contrasting B-tree and LSM-tree on write path, read amplification, write amplification, space amplification, and best-fit workload](/imgs/blogs/b-trees-how-database-indexes-work-9.webp)

A B-tree **updates in place**: to change a row, you find its leaf page and modify it, which means a random-write to that page (and the page must be read first if it's not cached — a read-modify-write). This is great for reads (one tree, one descent, the data is where the index says it is) but it imposes **write amplification**: a tiny logical change rewrites a whole 8 KB page, and the writes are scattered randomly across the file.

An LSM-tree never updates in place. Writes go to an in-memory buffer (the memtable) and an append-only log; when the buffer fills, it's flushed as an immutable **sorted file (SSTable)**. Reads must check the memtable and potentially several SSTables, merging results — that's **read amplification**, mitigated by Bloom filters. Periodically, background **compaction** merges SSTables to discard overwritten and deleted (tombstoned) keys. The result: writes are sequential and cheap (great for ingest-heavy and write-heavy workloads), but reads can touch multiple files, and space temporarily balloons with obsolete data until compaction runs.

The matrix lays out the trade in five dimensions. Neither is "better"; they trade the same costs in opposite directions:

- **Read-heavy OLTP** (lots of point and range reads, moderate writes) → B-tree. Your bank, your e-commerce orders table, anything transactional.
- **Write-heavy / ingest** (firehose of inserts, time-series, event logs, metrics) → LSM. The sequential-append write path is dramatically cheaper at high insert rates.
- **Mixed** → it depends, and modern systems blur the line (InnoDB has a change buffer to batch secondary-index writes; some engines offer both).

When you read that a database is "built on RocksDB" (CockroachDB's storage layer, TiKV, Kafka Streams' state stores), that's an LSM. When you read "InnoDB" or "Postgres heap + B-tree," that's the in-place B-tree world. The application-level symptom is usually latency shape: B-trees give predictable read latency and write latency that degrades with random keys; LSMs give cheap writes and occasional read/compaction latency spikes.

It helps to put numbers on *write amplification*, the metric that most cleanly separates the two. Suppose you update one 200-byte row. A B-tree must read the 8 KB leaf page (if it isn't cached), modify it, and eventually write the whole 8 KB page back — plus a write-ahead log record. The 200-byte logical change became kilobytes of physical I/O: write amplification of perhaps 20–40×. An LSM instead appends the 200-byte change to a log and a memtable immediately — near-zero amplification *on the write itself* — but pays later: compaction rewrites that key several times as it migrates from level to level, so the *total* write amplification over the key's life can also reach 10–30×, just spread out and sequential rather than immediate and random. The difference that matters operationally is exactly that — *random versus sequential*. An LSM's amplified writes are large sequential writes that SSDs and disks love; a B-tree's are scattered random writes. On a write-saturated system, sequential 30× beats random 20×, which is why ingest-heavy systems lean LSM even when the raw amplification numbers look similar. Meanwhile the LSM pays its tax on reads — a point lookup may probe the memtable plus several SSTables, rescued only by Bloom filters that answer "definitely not here" cheaply — whereas the B-tree's one-descent read path is hard to beat. The whole choice compresses to a single question: does your workload write more than it reads, or read more than it writes?

## Which queries actually use the index {#which-queries-use-the-index}

**Rule of thumb: an index helps only if the query's predicate matches the index's sort order from the left. Wrap the column in a function and you throw the index away.**

You can have the perfect index and still get a full table scan, because whether the planner *uses* an index is decided by the shape of your predicate, not by the existence of the index. This is the single most common "my query is slow even though I added an index" bug.

![A decision tree mapping predicate shapes to plans: leftmost match earns a seek; functions and leading wildcards force a full scan](/imgs/blogs/b-trees-how-database-indexes-work-10.webp)

The decision tree captures the four shapes that decide a B-tree index's fate:

- **`= or range on the leftmost column`** → the planner can do an **index seek** or **range scan**. This is the happy path. A B-tree is sorted, so equality and range predicates (`=`, `<`, `>`, `BETWEEN`, `LIKE 'prefix%'`) on the leading column descend straight to the right place.
- **`func(c)` or `LIKE '%x'`** → **full table scan**. The moment you wrap the indexed column in a function — `WHERE lower(email) = 'a@b.com'`, `WHERE date(created_at) = '2026-06-11'`, `WHERE col + 1 = 5` — the index is on `col`, not on `func(col)`, so the sorted order is useless and the planner scans everything. Same for a *leading* wildcard `LIKE '%smith'`: the index is sorted left-to-right, and you've left the left end unspecified. (A trailing wildcard `'smith%'` is fine — it's a range.)
- **`low-cardinality column`** → the **planner may choose a sequential scan anyway**. If a column has few distinct values (a boolean `is_active`, a `status` with three states), an index lookup that matches half the table is *slower* than just scanning, because of all the random heap fetches. The planner knows this from its statistics and rationally ignores the index.
- **`all needed columns are in the index`** → an **index-only / covering scan**, the best case: the query is answered entirely from the index without touching the table at all.

The most important practical corollary is the **leftmost-prefix rule** for composite indexes. An index on `(tenant_id, created_at, status)` is sorted first by `tenant_id`, then within each tenant by `created_at`, then by `status`. So it serves `WHERE tenant_id = 7`, and `WHERE tenant_id = 7 AND created_at > '...'`, and the full three-column match — but it does **not** efficiently serve `WHERE created_at > '...'` alone, because without fixing `tenant_id` first, the `created_at` values are scattered all through the index. It's a phone book sorted by last-name-then-first-name: great for "find all the Smiths," useless for "find everyone named John."

Here's the whole thing as runnable SQL you can paste into Postgres:

```sql
-- Setup: a composite index, leftmost column first.
CREATE INDEX idx_events ON events (tenant_id, created_at, status);

-- USES the index (leftmost prefix, range on the second column):
EXPLAIN ANALYZE
SELECT * FROM events
WHERE tenant_id = 7 AND created_at > now() - interval '1 day';
--  Index Scan using idx_events ...   (fast)

-- DOES NOT use it efficiently (skips the leftmost column):
EXPLAIN ANALYZE
SELECT * FROM events
WHERE created_at > now() - interval '1 day';
--  Seq Scan on events ...            (full scan)

-- DOES NOT use the email index (function wraps the column):
EXPLAIN ANALYZE
SELECT id FROM users WHERE lower(email) = 'a@example.com';
--  Seq Scan on users ...            (needs an expression index)

-- Fix the function case with an EXPRESSION index on the function itself:
CREATE INDEX idx_users_lower_email ON users (lower(email));
-- now the planner can seek lower(email) directly.

-- Covering index: include the selected columns so the query never hits the heap.
CREATE INDEX idx_events_covering
  ON events (tenant_id, created_at) INCLUDE (status, payload_size);
EXPLAIN ANALYZE
SELECT status, payload_size FROM events
WHERE tenant_id = 7 AND created_at > now() - interval '1 hour';
--  Index Only Scan using idx_events_covering ...   (no heap fetch)
```

`EXPLAIN ANALYZE` is the ground truth. It tells you whether you got `Index Scan`, `Index Only Scan`, `Bitmap Heap Scan`, or `Seq Scan`, and the row estimates that drove the choice. Read it before and after every index change; the planner is the authority, not your intuition.

One more concept ties the section together: **sargability** (from "Search ARGument ABLE"). A predicate is sargable if it can be resolved by seeking into an index's sorted order, and non-sargable if it forces the engine to evaluate every row. `WHERE created_at >= '2026-06-01'` is sargable; `WHERE date_trunc('month', created_at) = '2026-06-01'` is not, even though they can describe the same rows, because the function destroys the column's usable order. The fix is almost always to rewrite the predicate into a sargable range (`created_at >= '2026-06-01' AND created_at < '2026-07-01'`) rather than to bolt on an index for the function. A related trap is the **`OR` across different columns**: `WHERE a = 1 OR b = 2` generally can't use a single composite index, because the two arms want different sort orders; the planner's best move is often a **bitmap index scan** that probes an index on `a` and an index on `b` separately and ORs the resulting row bitmaps together — which is why two single-column indexes sometimes beat one composite index for `OR`-heavy queries. And `IN (1, 2, 3)` is just sugar for an OR of equalities, which a B-tree handles as three cheap seeks. The throughline: the planner can only exploit the index's *sorted order*, so any predicate you can phrase as "a contiguous range of the sorted key" is fast, and any predicate that scrambles or hides that order is not.

## Case studies from production

Theory becomes muscle memory through war stories. Here are ten incidents — the symptom, the wrong first hypothesis, the actual root cause, the fix, and the lesson. Every one is a B-tree behavior we built up above, hitting a real system.

### 1. The UUIDv4 write cliff

**Symptom.** A team migrated their `orders` table's primary key from `BIGINT AUTO_INCREMENT` to `UUIDv4` for "security" and to avoid exposing row counts. Insert throughput on the same hardware dropped from ~40k rows/sec to ~6k, and the table's on-disk size grew ~80%. **Wrong first hypothesis.** "UUID comparisons must be slow." (They're 16-byte memcmps — negligible.) **Root cause.** In InnoDB the primary key is the clustered index, so every insert places the row at a *random* position in the B-tree. Random inserts hit random leaves, each already near-full, so nearly every insert caused a page split, settling pages at ~50% occupancy and thrashing the buffer pool — exactly the random-insert regime from the cost-of-disorder section. **Fix.** Switched the clustered key back to an auto-increment `BIGINT` and moved the UUID to a secondary index (where its randomness costs far less because secondary leaves are smaller and the table itself stays dense). Where an external UUID was non-negotiable, they used UUIDv7, which is time-ordered, restoring monotonic inserts. **Lesson.** The clustered key's insert order is the single biggest write-throughput lever in InnoDB. Random primary keys are a self-inflicted wound.

### 2. The index that wouldn't stop bloating

**Symptom.** A Postgres `sessions` table with heavy churn (rows inserted and deleted constantly) had a primary-key index that grew to 3× the size of the live data and kept growing, even though the row count was stable. **Wrong first hypothesis.** "We need a bigger disk." **Root cause.** B-tree engines are lazy about merging underfull pages after deletes — they leave half-empty pages in place rather than rebalancing eagerly. Under constant insert/delete churn the index accumulated mostly-empty leaf pages that were never reclaimed, because the deleted-then-reused key space left pages perpetually fragmented. Compounding it, dead index entries from old row versions (MVCC) lingered. **Fix.** A `REINDEX CONCURRENTLY` rebuilt the index densely, dropping it back to ~1× the data size; longer term they scheduled periodic reindexing for the high-churn table and tuned autovacuum to keep up. **Lesson.** Index size is a function of *historical* churn, not current row count. High-delete tables need periodic `REINDEX`.

### 3. The composite index nobody's query matched

**Symptom.** An engineer added `CREATE INDEX ON events (created_at, tenant_id)` to speed up a per-tenant dashboard, and it did nothing — the dashboard query still did a full scan. **Wrong first hypothesis.** "The planner is broken / needs `ANALYZE`." **Root cause.** The dashboard query was `WHERE tenant_id = ? AND created_at > ?`, but the index was ordered `(created_at, tenant_id)` — `created_at` first. The query didn't constrain `created_at` to a tight enough range, so the leftmost-prefix rule meant the index couldn't seek to a tenant; the `tenant_id` values were scattered across every `created_at`. **Fix.** Recreated the index as `(tenant_id, created_at)` — tenant first, which matches how the query filters. The dashboard went from a 4-second scan to a 12 ms index range scan. **Lesson.** Column order in a composite index is not arbitrary. Put the columns used for equality first, then the range column. The index must match the query's predicate shape from the left.

### 4. The update-heavy table that split on every write

**Symptom.** A counters table (`UPDATE ... SET count = count + 1` millions of times an hour) had pathological write latency and constant page splits, despite the rows never changing size. **Wrong first hypothesis.** "Lock contention on the hot rows." (Real, but secondary.) **Root cause.** The table had been bulk-loaded with a fill factor of 100 — pages packed completely full. Some updates grew a variable-length column slightly, and with zero free space on the page, each such update forced a page split just to make room. **Fix.** Rebuilt the table with `fillfactor = 70`, leaving slack on every page so updates happen in place. Page splits dropped to near zero and update latency stabilized. **Lesson.** Fill factor is a write-pattern decision. Pack pages full for append/read-only data; leave slack for update-heavy tables so in-place updates don't trigger splits.

### 5. The covering index that erased a million heap fetches

**Symptom.** An InnoDB analytics query — `SELECT status, amount FROM orders WHERE customer_id = ?` — was slow despite a perfectly good index on `customer_id`. **Wrong first hypothesis.** "We need more RAM for the buffer pool." **Root cause.** The `customer_id` index leaves held only the primary key, so for each of the (often thousands of) matching rows, InnoDB did a **bookmark lookup** — a second descent into the clustered index to fetch `status` and `amount`. A few thousand random clustered-index lookups per query was the cost. **Fix.** Changed the index to `(customer_id, status, amount)` — a **covering index** that includes every selected column. Now the secondary index alone answers the query; no bookmark lookups. Query time dropped 30×. **Lesson.** In a clustered-index engine, a covering index turns two descents into one for every matched row. When a query is slow despite an index, check whether it's paying bookmark lookups, and cover it.

### 6. The ingest pipeline that needed an LSM, not a B-tree

**Symptom.** A telemetry pipeline ingesting ~500k events/sec into a Postgres table was saturating disk I/O on writes, with the B-tree index maintenance dominating. **Wrong first hypothesis.** "Throw more Postgres replicas at it." (Replicas don't help write throughput.) **Root cause.** Every insert maintained multiple B-tree indexes with random in-place writes; at half a million inserts a second, the write amplification of in-place page updates was the wall. This is precisely the workload B-trees are *worst* at and LSM-trees are *best* at. **Fix.** Moved the high-volume ingest path to an LSM-backed store (RocksDB-based) whose sequential-append write path absorbed the firehose, and kept Postgres for the lower-volume transactional data that benefits from B-tree reads. **Lesson.** Match the storage engine to the workload. A write-firehose belongs on an LSM; forcing it onto a B-tree means fighting write amplification forever.

### 7. The long transaction that froze VACUUM

**Symptom.** A Postgres database's indexes and tables bloated steadily over a weekend; autovacuum was running but reclaiming nothing. **Wrong first hypothesis.** "Autovacuum is misconfigured / too slow." **Root cause.** A forgotten analytics session held a transaction open for 40 hours. Postgres cannot vacuum (reclaim) any row version newer than the oldest running transaction's snapshot, because that transaction might still need to see it. So every dead tuple and dead index entry created during those 40 hours was un-reclaimable, and both heap and B-tree indexes grew unbounded. **Fix.** Killed the idle-in-transaction session; vacuum immediately reclaimed gigabytes. Added monitoring and `idle_in_transaction_session_timeout` to cap transaction age. **Lesson.** In MVCC databases, index bloat can be caused by something with no obvious connection to the index — a long-running transaction elsewhere. The oldest transaction gates all cleanup.

### 8. ORDER BY ... LIMIT served straight from the index

**Symptom.** A "latest 20 activity items" endpoint — `SELECT * FROM activity WHERE user_id = ? ORDER BY created_at DESC LIMIT 20` — was doing a full sort of tens of thousands of rows per request. **Wrong first hypothesis.** "Add caching in front of it." (Treating the symptom.) **Root cause.** The index was on `user_id` only. The database could seek to the user's rows, but then had to *sort* all of them by `created_at` to get the top 20. **Fix.** Created an index on `(user_id, created_at DESC)`. Now the B-tree's own sort order matches the query's `ORDER BY`: the engine seeks to the user, walks the leaf chain in `created_at DESC` order, takes 20 rows, and stops. No sort step at all — the index *is* the sorted order. Latency dropped from ~200 ms to ~1 ms. **Lesson.** A B-tree is pre-sorted. An index whose column order matches your `ORDER BY` eliminates the sort entirely and makes `LIMIT` an early stop, not a post-filter.

### 9. The boolean column index the planner refused

**Symptom.** A team indexed `WHERE is_processed = false` on a jobs table, saw the planner ignore the index and seq-scan, and concluded the index was "broken." **Wrong first hypothesis.** "Force the index with a hint." (They did, and it got *slower*.) **Root cause.** At the time, ~40% of rows had `is_processed = false`. An index that matches 40% of a table is worse than a scan, because using it means ~40% of the rows fetched via *random* heap access, versus a sequential scan that reads pages in order. The planner correctly estimated this from its statistics and chose the scan. **Fix.** Replaced the full-column index with a **partial index**: `CREATE INDEX ON jobs (id) WHERE is_processed = false`. This indexes *only* the unprocessed rows, so it's tiny and the planner uses it happily — and as the backlog shrinks, the index shrinks with it. **Lesson.** Low-cardinality columns are poor B-tree candidates; the planner's refusal is wisdom, not a bug. A partial index on the selective subset is usually the right answer.

### 10. The right-most leaf that became a latch hotspot

**Symptom.** A high-concurrency service with an auto-increment primary key — exactly the "good," monotonic insert pattern — hit a throughput ceiling on inserts under heavy concurrency, with threads stalling on internal locks even though disk and CPU had headroom. **Wrong first hypothesis.** "We need faster disks." **Root cause.** The flip side of monotonic inserts: *every* insert goes to the same right-most leaf page, so under high concurrency all inserting threads contend for the **latch** (the short-term lock) on that one hot page and its parent. The page itself is cache-resident and fast; the contention is on coordinating concurrent access to it. **Fix.** Options included hash-partitioning the table so inserts spread across several B-trees (several right-most leaves), or switching to a key scheme with a small leading shard component. They partitioned, and insert throughput scaled with cores again. **Lesson.** Monotonic keys are great for I/O and density but create a single write hotspot. At extreme insert concurrency, the right-most leaf is a contention point, and the answer is to spread inserts across multiple trees. The "best" insert pattern for one dimension can be the bottleneck in another.

## When to reach for a B-tree, and when not to

The B+ tree is the right default for so many workloads that the interesting question is when to *override* the default.

**Reach for a B-tree index when:**

- Your queries do **equality or range** lookups on a totally-ordered key (`=`, `<`, `>`, `BETWEEN`, `LIKE 'prefix%'`). This is the structure's home turf.
- You need results in **sorted order** (`ORDER BY`, `GROUP BY`, `MIN`/`MAX`, top-N with `LIMIT`). A B-tree is pre-sorted; it serves these without a sort step.
- The workload is **read-heavy or balanced** OLTP — the in-place update model gives predictable, low read latency.
- You want **one structure for both point and range** queries. A B-tree does both; you don't need separate indexes for "find this" and "find this range."
- You're building the **primary access path** of a transactional table. Almost every primary key and most secondary indexes should be B-trees, full stop.

**Skip a B-tree (or override the default) when:**

- The workload is a **write firehose** — high-rate ingest, time-series, event logs. An LSM-tree's sequential-append writes will crush a B-tree's random in-place writes. Reach for RocksDB-backed storage or a purpose-built time-series engine.
- You only ever do **exact-match point lookups** and never range or order, *and* the data fits in memory — a hash index is genuinely O(1) and beats the B-tree's O(log n). (This is rarer than people think; the moment you want "recent" or "sorted," you need the B-tree.)
- The query is **similarity search** in high-dimensional space (embeddings, nearest-neighbor). Order is meaningless there; you need HNSW/IVF vector indexes, as in [Vector Databases](/blog/machine-learning/ai-agent/vector-database).
- The predicate is **full-text search** (`WHERE document @@ 'query'`) — that's an inverted index (GIN in Postgres), not a B-tree.
- The column is **very low cardinality** and you're tempted to index it directly. Use a partial index on the selective subset, or accept the scan. A B-tree over three distinct values is mostly wasted.
- You're indexing a column you'll always query through a **function or transformation** — index the *expression* (`lower(email)`, `date_trunc('day', created_at)`) instead, or the plain B-tree won't be used.

The meta-lesson under all ten case studies and both lists is the one we opened with: a B-tree is a disk-shaped structure. Its design — high fanout, values in linked leaves, balance-by-splitting, in-place updates — is a set of answers to the question "how do I minimize page reads and keep pages dense on a device where a read costs a hundred thousand comparisons." Once you see every index decision through that lens — *is this keeping pages dense, is this minimizing disk reads, is this matching the sorted order I need* — the surprising behaviors stop being surprising. The index you thought was a sorted list turns out to be one of the most quietly brilliant pieces of engineering in the systems you use every day.

It's also why the skill transfers. The next time you meet a structure you haven't seen — an LSM's leveled SSTables, a vector index's proximity graph, a columnar store's zone maps — ask the same three questions you now ask of a B-tree: what's the expensive operation this is shaped to avoid, how does it keep its hot data dense and cache-resident, and what query shape does its physical order make cheap. Storage engines are not magic; each is a different answer to the one fact that "memory is a hierarchy and the lower levels are slow." The B-tree is the answer the industry has trusted longest, precisely because its answer is so well-matched to the machine it runs on.

## Further reading

- **Douglas Comer, "The Ubiquitous B-Tree" (ACM Computing Surveys, 1979)** — the classic survey; still the clearest exposition of the invariants.
- **Goetz Graefe, "Modern B-Tree Techniques" (2011)** — an encyclopedic tour of every production refinement: prefix compression, latching, concurrency, recovery. The reference.
- **PostgreSQL documentation, "Index Types" and the `btree` access method** — how a real engine implements all of the above, including index-only scans and `INCLUDE` columns.
- **MySQL/InnoDB documentation, "Clustered and Secondary Indexes"** — the clustered-index model and bookmark lookups in detail.
- [Random UUIDs Are Killing Your Database Performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) — the page-split story of this post, applied end to end to identifier design.
- [Database Connection Pooling](/blog/software-development/database/database-connection-pooling) — the layer above the storage engine: how connections, not just indexes, gate throughput.
- [Redis in Production](/blog/software-development/database/redis-applications-and-optimization) — the in-memory counterpoint, where the disk-page constraints that shape B-trees don't apply.

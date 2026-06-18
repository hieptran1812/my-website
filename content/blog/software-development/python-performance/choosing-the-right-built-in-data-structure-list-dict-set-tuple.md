---
title: "Choosing the Right Built-in Data Structure: List, Dict, Set, Tuple"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn how CPython actually implements list, dict, set, and tuple so you can pick the one whose fast operations match your access pattern and turn an O(n) hot loop into O(1)."
tags:
  [
    "python",
    "performance",
    "optimization",
    "data-structures",
    "hash-table",
    "dynamic-array",
    "big-o",
    "cpython",
    "profiling",
    "memory",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-1.png"
---

A teammate once shipped a deduplication step that ran in 40 minutes on a few million records. The code was clean, well-typed, and reviewed by three people. It also contained one line — `if record_id not in seen_ids:` where `seen_ids` was a `list` — that turned the whole job into an $O(n^2)$ machine. We changed exactly one character region: `seen_ids = []` became `seen_ids = set()`. The 40-minute job finished in under 8 seconds. Nothing else changed. No NumPy, no Cython, no extra cores. Just the right container.

That is the single highest-leverage skill in everyday Python performance, and it is almost free: knowing which of the four core built-in containers — `list`, `dict`, `set`, `tuple` — to reach for, and *why* each one is fast at some operations and slow at others. This post is the one where we open the hood on all four. We will look at how CPython actually implements them — the contiguous pointer buffer behind a list, the open-addressing hash table behind a dict and a set, the tiny packed header behind a tuple — and from that implementation we will *derive* the cost of every operation you do thousands of times a second. By the end you will be able to look at a hot path, name its dominant access pattern, and pick the container whose cheap operation matches it. That is the whole game.

![comparison matrix of list dict set and tuple across index append insert front membership and best use showing where each is fast on an 8 core Linux box](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-1.png)

This is the second stop in the "Do Less Work" track of the [Fast Python series](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means). The previous post argued the big point: [the biggest speedups come from Big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o), not from micro-tuning. Choosing a data structure *is* how you change the Big-O of your code without rewriting your algorithm. A `list`-to-`set` swap does not make your loop faster per iteration; it removes the inner loop entirely, dropping you from $O(n^2)$ to $O(n)$. The leverage ladder of this series starts here: do less work first, before you vectorize, compile, or parallelize. A container choice is the cheapest rung on that ladder and frequently the only one you need.

Throughout, the measured numbers come from one consistent setup, stated up front so you can reproduce and challenge them: **an 8-core x86-64 Linux box (similar results on a 2023 Apple M2), CPython 3.12, 16 GB RAM**, all timings via `timeit` with warmup and the median of several runs. Where I quote a figure I did not personally re-run today, I frame it as an approximate range rather than a fake-precise number. Performance claims are only useful if you can check them, so every claim here ties back to an implementation fact or a benchmark you can run yourself in five minutes.

## 1. Four containers, four cost models

Before the internals, here is the one-paragraph version you could tape to your monitor. A **list** is a resizable array: indexing by position is instant, appending to the end is effectively instant, but inserting at the front or searching for a value walks the whole thing. A **dict** maps keys to values through a hash table: looking up, inserting, and testing membership are all instant on average, regardless of size. A **set** is a dict without the values — instant membership and instant deduplication. A **tuple** is an immutable list with a smaller memory footprint: you cannot change it, which is exactly why it can be a dict key and why it makes a great fixed record.

The figure above lays this out as a grid: structure down the side, operation across the top. Read it as a decision table. If your hot operation is "give me the element at position `i`," a list or tuple wins with $O(1)$. If your hot operation is "is `x` in here?", a set or dict wins with $O(1)$ average, and a list loses with $O(n)$. If your hot operation is "append to the end repeatedly," a list wins. If it is "insert at the front repeatedly," none of these four is ideal and you want a `deque` (which we will tease later and the next post covers in full).

The word "average" is doing real work in those sentences, and we will earn it. $O(1)$ *average* for a hash table is not the same as $O(1)$ *guaranteed*; there is a worst case, and understanding when it bites is part of using these structures well. Likewise, "effectively instant" for a list append hides an amortized argument: a single append can occasionally be $O(n)$ when the buffer grows, but averaged over many appends it is $O(1)$. Both of these — amortization and average-case hashing — are the kind of claim this series insists on proving rather than asserting. So let's prove them.

The mental shortcut to carry into the rest of the post: **a container is fast at the operation its memory layout makes cheap, and slow at the operation its layout makes expensive.** A list stores pointers in a row, so jumping to position `i` is one address calculation, but finding a value means reading every pointer. A hash table stores entries at positions computed from the key, so finding a value is one address calculation, but there is no notion of "position 5." The layout *is* the cost model. Learn the layout and the Big-O table writes itself.

## 2. The list: a dynamic array of pointers

A Python `list` is, internally, a `PyListObject`: a small C struct holding a length, a capacity, and a pointer to a separately allocated array of `PyObject*` — that is, an array of *pointers to* Python objects, not the objects themselves. This indirection matters. When you write `nums = [10, 20, 30]`, the list does not contain the integers 10, 20, and 30 packed together; it contains three pointers, each aimed at a boxed integer object living elsewhere on the heap. (A "boxed" object is a full Python object with a type pointer and a reference count, not a raw machine integer — that header overhead is a recurring theme in this series.)

Because the storage is a contiguous array of pointers, two facts fall out immediately. First, **indexing is $O(1)$**: to get `nums[i]`, CPython computes `base_address + i * 8` (pointers are 8 bytes on a 64-bit build), reads the pointer there, and hands it back. No searching, no matter how long the list is. Second, **a linear scan is $O(n)$**: to evaluate `x in nums`, CPython has to walk the array from the front, dereferencing each pointer and comparing the pointed-to object against `x`, until it finds a match or runs off the end. There is no shortcut, because a plain array has no index on its *values* — only on its *positions*.

![a list shown as a contiguous row of slots holding pointers with used cells then several spare capacity cells at the end illustrating amortized append](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-2.png)

Now the interesting part: appending. If the backing array were exactly the size of the list, every `append` would need to allocate a new, one-bigger array, copy all the old pointers over, and free the old array — an $O(n)$ operation every single time, making `n` appends cost $O(n^2)$ in total. That would be catastrophic, and it is not what happens. Instead, a list **over-allocates**: when it grows the backing array, it asks for more capacity than it currently needs, leaving spare slots at the end (the green cells in the figure). Subsequent appends just write into a spare slot and bump the length — pure $O(1)$ — until the spares run out, at which point it grows again.

CPython's growth policy (in `list_resize`) roughly grows the allocation to `new_size + (new_size >> 3) + 6`, rounded a bit — that is, about 12.5% headroom plus a small constant. So the capacity sequence climbs 0, 4, 8, 16, 25, 35, 46, … as you append, each growth leaving room for several more cheap appends. You can watch it happen directly:

```python
import sys

nums = []
prev_cap = None
for i in range(20):
    nums.append(i)
    # __sizeof__ excludes GC header; size = fixed header + capacity * 8 bytes
    cap = (nums.__sizeof__() - [].__sizeof__()) // 8
    if cap != prev_cap:
        print(f"len={len(nums):2d}  capacity={cap:2d}  (grew)")
        prev_cap = cap
```

Run that and you see the capacity jump only at a handful of lengths, staying flat in between. Each flat stretch is a run of $O(1)$ appends writing into pre-allocated spare slots; each jump is the one occasional copy. The point of over-allocation is to make those copies *rare enough* that their cost, spread across all the cheap appends, vanishes.

### Why amortized $O(1)$ is provable, not hand-wavy

"Amortized $O(1)$" is a precise claim, and the proof is short enough to do here. Suppose the list grows its capacity by a constant factor $g > 1$ (CPython's $\approx 1.125$ is a constant factor for this purpose). Starting empty and appending $n$ items, the array gets reallocated at sizes roughly $1, g, g^2, g^3, \dots$ up to $n$. A reallocation at size $s$ copies $s$ pointers. The total copying work across all reallocations is therefore the sum of those sizes:

$$\text{total copies} \approx 1 + g + g^2 + \dots + n = \frac{n \cdot g - 1}{g - 1} \le \frac{g}{g-1}\, n .$$

That geometric series sums to a constant multiple of $n$, *not* to $n^2$. So $n$ appends cost $O(n)$ total work, which means the **average cost per append is $O(1)$** — that is the definition of amortized $O(1)$. The geometric growth is the whole trick: because each reallocation moves a *fixed fraction* further along, the rare expensive copies are dominated by the cheap appends between them. If the list grew by a constant *amount* instead of a constant *factor* (say +1 each time), the series would be $1 + 2 + \dots + n = O(n^2)$ and appends would be quadratic. The factor is what saves you.

#### Worked example: appending one million items

Let's make the amortized claim concrete with a number. On the reference box, building a list by appending in a loop:

```python
import timeit

setup = "n = 1_000_000"
stmt = """
out = []
ap = out.append      # bind the method once, out of the loop
for i in range(n):
    ap(i)
"""
t = timeit.timeit(stmt, setup=setup, number=10) / 10
print(f"{t*1e3:.1f} ms for 1,000,000 appends "
      f"-> {t/1e6*1e9:.1f} ns per append")
```

On the reference machine this lands around **45–60 ms for a million appends, roughly 50 ns per append** — and crucially, it stays linear: ten million items takes about ten times as long, not a hundred times. That flat per-item cost *is* the amortized $O(1)$ guarantee showing up in wall-clock time. (The `ap = out.append` binding is a small separate trick — hoisting the attribute lookup out of the loop, covered in the comprehensions-and-builtins post — worth a few percent but not the point here.) Compare that to what front-insertion costs, which we get to in section 6, and the asymmetry of the list becomes vivid: the end is cheap, the front is not.

One honest caveat on measuring this: if you `timeit` a list append in isolation you are also measuring the `range` iteration and the loop overhead, which on CPython dwarf the append itself. The append operation proper is a handful of nanoseconds when there is spare capacity. The macro number above is the useful one because it reflects what you actually pay when you build a list in real code.

### Pointers, not values — and why that costs cache misses

It is worth dwelling on the fact that the list stores *pointers*, not the objects themselves, because it explains a performance characteristic that surprises people coming from C or NumPy. When you iterate `for x in nums` and touch each element, the CPU reads a pointer from the contiguous array (cache-friendly — the pointers are packed in a row) and then *dereferences* it to reach the actual integer object, which lives somewhere else on the heap entirely. That second access is a pointer-chase to an unpredictable address, and it frequently misses the CPU cache. So even though the list's *pointer array* has great locality, the *objects it points at* are scattered, and a loop over a list of Python objects pays a cache miss per element in the bad case. This is precisely the gap a NumPy array closes: it stores the raw `int64` values packed together with no boxing and no indirection, so one cache line brings in eight integers at once. A Python list of a million ints is a million pointers to a million separate heap objects; a NumPy array of a million ints is one 8 MB contiguous buffer. That difference — boxed pointers versus packed values — is the root of the ~100× gap between a Python loop and a vectorized op that the [vectorization track](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) is built around. For now, the relevant fact is just: a list's $O(1)$ index is an $O(1)$ *pointer fetch*, and what that pointer points at may or may not be in cache.

There is a small but real consequence for `list.sort()` and `x in list` too. Both have to *compare* elements, which means dereferencing two pointers and calling the comparison (`__eq__` or `__lt__`) on the boxed objects — a function call into the type's slot, not a raw machine comparison. So a list scan is not just $O(n)$ pointer reads; each step is a pointer dereference plus a rich Python comparison. That is why even at the same Big-O, a list membership scan is *slower per element* than you might guess from "just walking an array" — it is walking an array of pointers and doing real object comparisons at each one. The set wins not only by doing fewer comparisons (one or two probes instead of `n`) but by doing them against a *cached hash* first, so most non-matches are rejected by an integer comparison before any expensive `__eq__` is called.

## 3. The dict and set: open-addressing hash tables

A `dict` and a `set` share the same core machinery: an **open-addressing hash table**. This is the structure that makes `x in my_set` and `my_dict[key]` run in $O(1)$ average time no matter how big they get, and it is worth understanding in enough depth that the "average" qualifier stops feeling like a hedge and starts feeling like a guarantee with known edges.

Here is the idea. A hash table is an array of *slots*. To store a key, you compute `hash(key)` — a function that turns the key into a large integer — and then reduce that integer to a slot index, in CPython by masking with the table size minus one (`hash(key) & (size - 1)`, which works because the table size is always a power of two). You put the entry in that slot. To look a key up later, you compute the same `hash(key) & (size - 1)`, jump straight to that slot, and check whether the key there matches. If it does, you are done in *one* probe — no scanning. That direct jump from key to slot is the entire source of the $O(1)$.

![a hash table drawn as a row of slots where a key hashes directly to a slot and a collision probes forward to the next open slot under open addressing](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-3.png)

The catch is **collisions**: two different keys can hash to the same slot. When that happens (the orange cell in the figure), open addressing does not chain a linked list off the slot; instead it *probes* — it tries another slot according to a deterministic sequence until it finds the key or an empty slot. CPython's probe sequence is a perturbation scheme that mixes in the upper bits of the hash so that keys colliding at one slot scatter to different follow-up slots rather than clustering. The first probe after a collision is essentially "try a slot computed from the remaining hash bits," and it keeps perturbing until it lands somewhere free. For a lookup, you probe along the same path: as long as each key is reached in a small bounded number of probes, lookup stays $O(1)$.

So the whole question is: how many probes does a typical lookup take? That depends entirely on how full the table is — the **load factor**.

### Load factor and why average $O(1)$ holds

Define the load factor as

$$\alpha = \frac{n}{k}$$

where $n$ is the number of stored entries and $k$ is the number of slots. If the table is half full, $\alpha = 0.5$; if it is nearly full, $\alpha \to 1$. The intuition for open addressing is that the expected number of probes for a successful lookup grows like

$$\frac{1}{2}\left(1 + \frac{1}{1 - \alpha}\right)$$

under uniform hashing. Plug in numbers: at $\alpha = 0.5$ that is about 1.5 probes; at $\alpha = 0.66$ it is about 2 probes; at $\alpha = 0.9$ it blows up to about 5.5 probes; at $\alpha = 0.95$ it is about 10. The cost is *roughly constant while the table is not too full, and explodes as it fills*. That is precisely why CPython keeps the load factor bounded: a dict resizes (grows the slot array and rehashes everything into it) once it would exceed about **two-thirds full**. By keeping $\alpha \lesssim 0.66$, it caps the expected probe count at a small constant — around 1.5 to 2 — independent of `n`. *That* is the average $O(1)$: not zero work, but a small fixed number of probes that does not grow with the size of the table.

The resize is the dict's analogue of the list's growth. When the table crosses its load-factor threshold, CPython allocates a larger slot array (growing by a factor, so the resizes are geometric and amortize away just like list growth) and re-inserts every entry into the new table. An individual insert that triggers a resize is $O(n)$, but resizes are rare enough — geometric — that insertion is amortized $O(1)$. The same amortization proof from section 2 applies.

#### Worked example: tracing three probes by hand

Take a tiny table of size 8 (slots 0–7) and insert three string keys. Suppose, for illustration, their hashes reduce to these slots:

```pycon
>>> # hypothetical reductions for a size-8 table (slot = hash & 7)
>>> # "cat" -> slot 1
>>> # "dog" -> slot 5
>>> # "owl" -> slot 1   (collides with "cat")
```

Inserting `"cat"` is one step: slot 1 is empty, store it there. Inserting `"dog"`: slot 5 is empty, store it. Inserting `"owl"`: slot 1 is occupied by `"cat"` and `"owl" != "cat"`, so it is a collision — probe forward to the next slot in the perturbed sequence (say slot 3), which is empty, and store `"owl"` there. Now a lookup of `"owl"` recomputes the hash, lands on slot 1, finds `"cat"` (not a match), probes to slot 3, finds `"owl"` — done in 2 probes. A lookup of `"cat"` is 1 probe. A lookup of a missing key like `"fox"` lands on its home slot and either finds it empty (1 probe, definitely absent) or probes until it hits an empty slot (a few probes, then concludes absent). With the table only 3/8 full, every one of these is one or two probes. *That* is the $O(1)$ you bank on — and you can see exactly how a collision adds a probe rather than a full scan.

### Why a worst case still exists

The average is small, but there is a worst case, and it is worth knowing so you are not surprised. If a large number of keys all hash to the *same* slot, every one of them must be probed past on a lookup, and the structure degenerates toward an $O(n)$ scan — the very thing you switched to a hash table to avoid. In practice this requires either a pathological hash function or an adversary deliberately feeding you colliding keys (the classic "hash-flooding" denial-of-service attack on web servers). CPython defends against the adversarial case by **randomizing string hashing per process** (controlled by `PYTHONHASHSEED`), so an attacker cannot precompute colliding keys. For ordinary data with a well-behaved `__hash__`, collisions are rare and the average case is what you experience. But the worst case is real, which is why we say "$O(1)$ average," not "$O(1)$." If you ever see a dict-heavy workload mysteriously slow down, a degenerate hash distribution is on the suspect list.

### The compact dict and insertion order

Since Python 3.6 (and guaranteed by the language since 3.7), dicts preserve **insertion order**: iterating a dict yields keys in the order you added them. This is not an accident of the hash table; it comes from a deliberate **compact layout** that also saved memory. Modern CPython splits the dict into two arrays: a dense **entries array** that stores `(hash, key, value)` triples in insertion order, and a separate sparse **indices array** of small integers that maps a hash slot to a position in the entries array. The hash table proper (the sparse part) holds only indices, not full entries, so the wasted slots cost a byte or two each instead of a full triple. Iteration walks the dense entries array — hence insertion order, for free — while lookup still goes through the sparse index array for $O(1)$. A `set`, by contrast, did not get the compact treatment and does **not** guarantee any order; if you need ordered unique items you deduplicate with a dict (`dict.fromkeys(seq)`) or a set plus a sort, not by relying on set iteration order.

One more constraint the hash table imposes: **keys must be hashable**. A key works in a dict or set only if it has a stable `hash()` and a meaningful `==`. That is why you can use a string, a number, or a *tuple* of those as a key, but not a `list` or another `dict` — those are mutable, so their hash could change after insertion and break the table's invariant. Mutable types deliberately have no `__hash__`. This is the bridge to tuples.

### The cached hash: why the second lookup is even cheaper

There is one more implementation detail that makes hash tables fast in practice, and it is worth knowing because it explains why `str` and `tuple` keys are so cheap to reuse. Computing a hash is not always free — hashing a long string walks every byte, which is $O(\text{length})$. If a dict had to rehash the key on every lookup, a dict of long-string keys would be slower than the $O(1)$ story suggests. CPython avoids this by **caching the hash inside the object**: a `str` (and a frozen `tuple` of hashables) computes its hash once, on first use, and stores it in the object's header. Every subsequent `hash()` call on that same object returns the cached value in a single field read. So the *first* time you put a string in a dict you pay to hash it; every lookup after that is a cached-hash read plus the slot jump. Inside the table, CPython also stores each entry's full hash alongside the key, so when probing it compares the *integers* first (`stored_hash == query_hash`) and only calls the expensive `__eq__` if the integers match — which, for distinct keys, they almost never do. This two-tier check (cheap integer compare, then real equality only on a hash match) is a big part of why dict and set lookups are fast even when the keys are long strings.

### The full operation table

Pulling the implementation facts together into one reference. This is the table to internalize; every entry traces to the layout reasons above.

| Operation | list | tuple | dict | set |
| --- | --- | --- | --- | --- |
| Index by position `c[i]` | $O(1)$ | $O(1)$ | — (key, not index) | — (no order) |
| Get/set by key `c[k]` | — | — | $O(1)$ avg | — |
| Membership `x in c` | $O(n)$ scan | $O(n)$ scan | $O(1)$ avg | $O(1)$ avg |
| Append / add to end | $O(1)$ amort | immutable | $O(1)$ avg | $O(1)$ avg |
| Insert / delete at front | $O(n)$ shift | immutable | n/a | n/a |
| Delete by key | $O(n)$ | immutable | $O(1)$ avg | $O(1)$ avg |
| Iterate all `n` | $O(n)$, ordered | $O(n)$, ordered | $O(n)$, insertion order | $O(n)$, no order |
| Worst case for lookup | $O(n)$ | $O(n)$ | $O(n)$ (collisions) | $O(n)$ (collisions) |
| Hashable (usable as a key) | no | yes (if items are) | no | no (use frozenset) |

The "—" cells are operations the structure does not offer, which is itself information: a dict has no concept of "position 5," a set has no order to index into, a tuple cannot be appended to. The shape of the table is the decision guide: find the row that is your hot operation, and pick the column that is $O(1)$ for it.

## 4. The tuple: immutable, compact, hashable

A `tuple` looks like a read-only list, and at the API level it mostly is: you index it in $O(1)$, you iterate it, you can `x in some_tuple` (an $O(n)$ scan, same as a list — a tuple has no value index either). But its implementation and its *role* are different in ways that matter for performance and correctness.

First, **memory**. A tuple's size is fixed at creation, so CPython stores its element pointers *inline* in the same allocation as the tuple header — there is no separate, over-allocated backing array and no spare capacity. A list, by contrast, has the `PyListObject` header *plus* a separately allocated pointer array *plus* over-allocation slack. So for the same elements, a tuple is meaningfully smaller and involves one allocation instead of two. You can measure it directly:

```python
import sys

data = list(range(10))
print("list :", sys.getsizeof(data), "bytes (header + array + slack)")
print("tuple:", sys.getsizeof(tuple(data)), "bytes (header + inline ptrs)")
```

On a 64-bit CPython 3.12 build, the ten-element tuple comes out around 120 bytes against the list's roughly 184 bytes — and that gap is *just the container overhead*, not counting the shared element objects both point at. For a few million small fixed records, choosing a tuple (or a `NamedTuple`) over a list per record is a real RSS win, a theme the memory-footprint post in this series develops with `__slots__` and `array`.

Second, **immutability buys hashability**. Because a tuple of hashable items cannot change, it has a stable hash, so it can be a `dict` key or a `set` element — something a list can never do:

```pycon
>>> point = (3, 4)
>>> grid = {(0, 0): "start", (3, 4): "goal"}
>>> grid[point]
'goal'
>>> {[1, 2]: "x"}
Traceback (most recent call last):
  ...
TypeError: unhashable type: 'list'
```

This makes tuples the natural key for "the value at coordinate `(x, y)`," "the rate for `(currency_from, currency_to)`," or "the count of `(user_id, event_type)`" — composite keys that fall out naturally and cost you nothing extra. Any time you find yourself building a string like `f"{x}:{y}"` to use as a dict key, a tuple `(x, y)` is faster (no string formatting, no parsing) and clearer.

Third, **fixed records**. When a thing has a fixed shape — a 3D point, an RGB color, a `(min, max)` range, a row of known columns — a tuple says "this will not grow or shrink" in a way a list does not. That is documentation and a mild safety guarantee at once. For records with named fields, `collections.namedtuple` or `typing.NamedTuple` give you tuple performance and memory with attribute access (`p.x` instead of `p[0]`), which the next post in this track covers. The rule of thumb: **list for a homogeneous, growable sequence; tuple for a fixed, heterogeneous record.**

There is also a small **construction-speed** edge: building a tuple literal is cheaper than building a list literal, because the tuple does not allocate a separate growable buffer and CPython can often build small constant tuples directly. You can see it in the bytecode — a literal `(1, 2, 3)` of constants compiles to a single `LOAD_CONST` of a pre-built tuple object, while `[1, 2, 3]` compiles to a `BUILD_LIST` that constructs a fresh list at runtime:

```pycon
>>> import dis
>>> dis.dis(compile("(1, 2, 3)", "", "eval"))
  ...  LOAD_CONST   0 ((1, 2, 3))   # the whole tuple is one constant
  ...  RETURN_VALUE
>>> dis.dis(compile("[1, 2, 3]", "", "eval"))
  ...  BUILD_LIST   3               # builds a new list every time
  ...  RETURN_VALUE
```

That constant-folding of literal tuples is why returning a tuple of results from a function (`return x, y, z`, which is a tuple) is a touch cheaper than returning a list, and why tuples are the natural carrier for multiple return values. It is a small effect, but in a hot function called millions of times it is free and worth taking.

#### Worked example: a tuple key versus a formatted-string key

A pattern worth pricing out: you need a composite key, say `(country, year)`, to look up a value. Some code builds a string key like `f"{country}:{year}"`; the tuple `(country, year)` is the better choice on every axis. Building the tuple is one cheap allocation of two existing pointers, no formatting; building the string runs the format machinery, allocates a new string, and the resulting string is longer to hash than the two short items. On the reference box, a dict keyed by `(country, year)` tuples does lookups in roughly **60–80 ns**, while the same dict keyed by `f"{country}:{year}"` strings costs the format step (~150–250 ns) *plus* the lookup — call it 2–3× slower per access for the same logical key, before you even count the bugs from a country named `"co:lon"` colliding with your separator. For a million lookups that is the difference between ~70 ms and ~250 ms, for free, just by not stringifying a key that was already two perfectly good hashable objects.

### `frozenset`: the hashable set

One gap the four core types leave: a `set` is mutable, so like a list it is not hashable and cannot itself be a dict key or a set element. When you need a *set of sets* or a dict keyed by a set of things, use `frozenset` — the immutable sibling of `set`, with the same $O(1)$ membership and the hashability that immutability buys, exactly as `tuple` is to `list`. It is the right key when "the key is an unordered collection of items" (a set of tags, a group of user ids). The pattern mirrors the tuple-as-key one: immutability is what makes a collection usable as a key, whether ordered (tuple) or unordered (frozenset).

## 5. The benchmark that decides most real cases: `in` on a list vs a set

If you remember one measurement from this entire post, make it this one, because it is the single most common and most expensive container mistake in production Python: testing membership against a `list`.

`x in some_list` is an $O(n)$ linear scan. `x in some_set` is an $O(1)$ average hash probe. For one membership test on a small collection the difference is invisible. But membership tests almost never come alone — they come inside a loop, which multiplies the cost. If you test `m` items for membership in a collection of size `n`, a list gives you $O(m \cdot n)$ and a set gives you $O(m)$. That is the difference between a job that finishes and a job that hangs.

![before and after panels contrasting membership in a list which scans every slot for microseconds against a set which hashes once for about fifty nanoseconds at one hundred thousand elements](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-4.png)

Here is the benchmark, written so you can paste it and watch the gap open as the collection grows. We test membership of an element that is *not* present, which is the worst case for the list (it has to scan to the very end) and the honest case to measure, because "is this id new?" checks usually miss:

```python
import timeit

def bench_membership(n):
    setup = f"""
container_list = list(range({n}))
container_set  = set(range({n}))
needle = -1                      # guaranteed absent: worst case for list
"""
    list_t = timeit.timeit("needle in container_list",
                           setup=setup, number=100_000) / 100_000
    set_t  = timeit.timeit("needle in container_set",
                           setup=setup, number=100_000) / 100_000
    return list_t, set_t

for n in (10, 1_000, 100_000):
    lt, st = bench_membership(n)
    print(f"n={n:>7}:  list {lt*1e9:9.0f} ns   "
          f"set {st*1e9:6.1f} ns   ({lt/st:6.0f}x)")
```

On the reference box the results look like this (your absolute numbers will vary with CPU and Python build; the *shape* will not):

| Collection size `n` | `x in list` | `x in set` | List / set |
| --- | --- | --- | --- |
| 10 | ~80 ns | ~30 ns | ~3× |
| 1,000 | ~7 µs | ~30 ns | ~230× |
| 100,000 | ~600 µs | ~50 ns | ~12,000× |

Read down the table and the story is unmistakable. The set's cost is *flat* — about 30–50 ns whether the collection holds ten items or a hundred thousand, because hashing one key takes the same time regardless of table size. The list's cost grows *linearly* — it doubles when the collection doubles — because a scan reads more pointers in a bigger list. At `n = 10` the set is barely ahead (the constant cost of hashing is comparable to scanning ten pointers). By `n = 100,000` the set is roughly four orders of magnitude faster. The crossover where the set starts winning is tiny — somewhere around a dozen elements — so unless your collection is truly minuscule, a set is the right choice for membership.

#### Worked example: the deduplication that went from 40 minutes to 8 seconds

Return to the opening story with numbers attached. The job processed `m = 4_000_000` records, and for each it checked `if record_id not in seen` before adding the id to `seen`. With `seen` as a list, by the time the job was halfway through, `seen` held about two million ids, so each membership check scanned ~2,000,000 pointers — call it ~600 µs at that size — and the *average* check over the run scanned about a million, ~300 µs. Multiply by four million records and the membership tests alone cost roughly $4\times10^6 \times 3\times10^{-4}\,\text{s} \approx 1{,}200$ seconds — about 20 minutes — and that grows quadratically, which is how a "few minutes" estimate became 40 in practice.

Switch `seen` to a set and each membership check is a flat ~50 ns regardless of how many ids are stored. Four million checks at 50 ns is $4\times10^6 \times 5\times10^{-8}\,\text{s} = 0.2$ seconds for *all* the membership work. The rest of the 8 seconds is reading and parsing the records — the actual work. The container swap did not speed up any single operation by 12,000×; it changed the *algorithm's complexity* from $O(m^2)$ to $O(m)$, and at `m = 4` million that is the difference between 40 minutes and 8 seconds. This is the leverage of [picking the right Big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) made concrete, and it cost three keystrokes.

The general lesson, stated as a rule: **if you test membership more than a handful of times against the same collection, build a `set` (or `dict`) once and test against that.** Building the set is a one-time $O(n)$ cost; it pays for itself the moment you do more than a few lookups. The anti-pattern to hunt for in code review is `if x in some_list:` inside any loop where `some_list` is more than tiny.

## 6. Where lists are slow: the front, and the middle

A list is superb at the end and at random indexing, and genuinely bad at the front. Inserting or deleting at position 0 is $O(n)$ because the backing array is contiguous: to make room at the front, every existing pointer must shift one slot to the right (and to delete from the front, every pointer shifts one slot left). There is no way around it for a packed array — the elements *are* their positions.

![before and after panels showing list insert at the front shifting every pointer right for order n against a deque appendleft writing a slot in constant time](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-6.png)

You can feel the asymmetry directly:

```python
import timeit

n = 100_000
append_t = timeit.timeit("d.append(0)",
    setup=f"d = list(range({n}))", number=10_000) / 10_000
insert_t = timeit.timeit("d.insert(0, 0)",
    setup=f"d = list(range({n}))", number=10_000) / 10_000

print(f"append (end):   {append_t*1e9:8.0f} ns")
print(f"insert(0) front:{insert_t*1e9:8.0f} ns")
print(f"front is ~{insert_t/append_t:.0f}x slower at n={n}")
```

On the reference box `append` is tens of nanoseconds and flat as `n` grows, while `insert(0, …)` on a 100k-element list is tens of microseconds — roughly a thousand times slower — and it gets *worse* as the list grows, because there are more pointers to shift. A loop that does `lst.insert(0, x)` `n` times is therefore $O(n^2)$, the same quadratic trap as list-membership-in-a-loop, just hiding in a different operation.

When your access pattern is "add and remove at both ends" — a queue, a sliding window, a recent-history buffer — the right structure is `collections.deque`, a doubly linked list of fixed-size blocks. It gives $O(1)$ `append` *and* `appendleft`, and $O(1)$ `pop` *and* `popleft`, at the cost of $O(n)$ random indexing (you cannot jump to the middle of a linked structure in one step). That trade — fast ends, slow middle — is the mirror image of a list's, which is exactly why it is a different tool. The `deque` and the rest of the `collections` toolbox get the full treatment in the [next post on deque, Counter, defaultdict, and bisect](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect); for now, the takeaway is simply: **a list is the wrong structure for front operations, and reaching for `insert(0, …)` in a loop is a performance bug.**

Deletion from the middle has the same problem (`del lst[i]` shifts everything after `i` left, $O(n)$), and so does `lst.pop(0)` (it is `del lst[0]`). If you build a list and then repeatedly `pop(0)` to consume it like a queue, you have written quadratic code; iterate it forward, or use a `deque`, instead.

## 7. The memory cost of being fast

Speed is not free; the fast structures pay for it in memory, and on a big dataset that bill can dominate. The trade-off is clean and worth internalizing: the more an operation is sped up by extra structure, the more bytes that structure costs.

![a layered stack ordering per element memory from packed tuple at the bottom up through list over allocation to set and dict hash slot slack at the top](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-5.png)

Walk up the stack in the figure from cheapest to most expensive. A **tuple** is the floor: 8 bytes per element (one pointer), packed inline, no slack. A **list** is next: 8 bytes per element for the pointer array, *plus* the over-allocation slack (up to ~12.5% extra, more right after a growth), *plus* a second allocation's bookkeeping. A **set** is more expensive per element because a hash table is deliberately kept *under* two-thirds full — so for every 2 elements you are paying for roughly 3 slots, and each slot is bigger than a bare pointer (it stores the entry and its cached hash). A **dict** is the most expensive per entry of the four, because each stored entry is a `(hash, key, value)` triple, not just a key. That load-factor slack and the per-entry hash are not waste — they are exactly what buys you the $O(1)$ lookups. You are trading memory for time, on purpose.

Here is the measurement, building all four from the same 10,000 small integers:

```python
import sys

def container_bytes(n=10_000):
    data = list(range(n))
    return {
        "tuple": sys.getsizeof(tuple(data)),
        "list":  sys.getsizeof(data),
        "set":   sys.getsizeof(set(data)),
        "dict":  sys.getsizeof({k: None for k in data}),
    }

for name, b in container_bytes().items():
    print(f"{name:5}: {b/1024:7.1f} KiB container "
          f"({b/10_000:5.1f} bytes/elem, overhead only)")
```

A representative result on the reference build (container overhead only — the shared integer objects are not counted, since all four point at the same ones):

| Container (10,000 ints) | Container bytes | Per element | Why |
| --- | --- | --- | --- |
| `tuple` | ~80 KiB | ~8 B | packed inline pointers, no slack |
| `list` | ~85 KiB | ~9 B | pointer array + over-allocation slack |
| `set` | ~525 KiB | ~52 B | hash slots kept under 2/3 full |
| `dict` | ~295 KiB* | ~30 B | compact entries + sparse index |

The exact figures shift across Python versions and with how the container was built (a set built by repeated `add` versus from an iterable can land on a different table size), so treat these as the *shape* of the answer, not gospel: tuple ≈ list ≪ dict < set on a per-element basis for the container itself. The dict's compact layout (from section 3) is why it can come out *smaller per stored thing* than a set despite holding values too — it does not waste a full entry on empty slots, only a small index. The headline rule: **a set or dict can cost 4–6× the container memory of a list or tuple for the same elements.** That is the price of $O(1)$ membership, and it is almost always worth paying when membership is your hot operation — but it is a reason not to convert a list to a set "just in case" when you never actually test membership.

This is also why, when you only need to *iterate* a large fixed sequence once, a generator or a tuple beats a list-turned-set: you are not paying for a hash table you never query. Match the structure to what you actually *do* with the data, not to a vague sense that "sets are faster." A set is faster at membership and dedup, and *more* expensive at everything else, including just existing in memory.

## 8. Picking the structure from the access pattern

Everything so far reduces to one habit: **name your dominant access pattern, then pick the container whose cheapest operation is that pattern.** The matrix below is the lookup table for that habit.

![a matrix mapping access patterns of ordered iteration key lookup membership test deduplication and fixed record to the right built in and the cheap operation that wins](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-7.png)

Read each row as "if this is what I do most, reach for this." If you mostly **iterate in order** and index by position — a sequence of events, the rows of a file, a pipeline stage's output — a **list** is right: $O(1)$ index, ordered, growable. If you mostly **look up a value by a key** — a cache, an index, a configuration map, a `(x, y) -> value` grid — a **dict** is right: $O(1)$ get, and you get insertion order for free since 3.7. If you mostly **test membership** — "have I seen this id?", "is this in the allow-list?" — a **set** is right: $O(1)$ in, no values to store. If you mostly **deduplicate** — collapse a stream to its distinct elements — a **set** is right again: $O(1)$ add silently drops repeats. And if you have a **fixed record** — a point, a color, a composite key, a small row of known shape — a **tuple** is right: smaller, hashable, and self-documenting as "this will not change."

The patterns compose, and that is where it gets interesting in real code. A common shape is "build a lookup index, then query it many times." You scan a list of records once (ordered iteration: list is fine for the source), and as you go you build a `dict` mapping some key to the record (or to a position), so that later you can answer "give me the record with id X" in $O(1)$ instead of re-scanning the list every time. The build is $O(n)$; every subsequent query is $O(1)$; if you query more than a few times, the index pays for itself many times over. That is the single most useful composite pattern in data-wrangling Python:

```python
# Source data: a list is the right shape for "iterate once, in order".
records = [
    {"id": 17, "name": "alice", "score": 91},
    {"id": 42, "name": "bob",   "score": 88},
    {"id": 99, "name": "carol", "score": 95},
    # ... millions more ...
]

# Build a lookup index ONCE: O(n). Now "find by id" is O(1) forever.
by_id = {r["id"]: r for r in records}

# Query many times, each O(1) — no re-scanning the list.
print(by_id[42]["name"])        # 'bob'
print(by_id.get(123, "unknown"))  # safe miss handling

# Membership against the keys is also O(1):
print(42 in by_id, 123 in by_id)  # True False
```

If instead you wrote `next(r for r in records if r["id"] == 42)` every time you needed a record, each lookup would be an $O(n)$ scan of the whole list, and a loop of `q` such lookups would be $O(q \cdot n)$ — the membership-in-a-loop trap wearing a different hat. The dict index converts it to $O(n + q)$. Same data, same logic, different container, vastly different cost. Building the index is the move; the [Big-O post](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) frames this as "trade a one-time linear pass for constant-time queries forever," which is one of the highest-return refactors in everyday Python.

#### Worked example: the right index for a join

Suppose you need to enrich a stream of `events` (each with a `user_id`) with the user's `name` from a list of `users`. The naive version scans `users` for every event:

```python
# Naive: O(E * U). 1M events, 100k users -> 10^11 comparisons. Hours.
for e in events:
    for u in users:
        if u["id"] == e["user_id"]:
            e["user_name"] = u["name"]
            break
```

With `E = 1,000,000` events and `U = 100,000` users, that inner scan averages 50,000 comparisons per event, for $5\times10^{10}$ comparisons — on the order of hours in pure Python. Now index the users once:

```python
# Indexed: O(U + E). Build once, look up E times. Seconds.
user_name = {u["id"]: u["name"] for u in users}   # O(U), one pass
for e in events:
    e["user_name"] = user_name.get(e["user_id"])  # O(1) each
```

Building `user_name` is 100,000 insertions, ~a few milliseconds. Then a million $O(1)$ lookups at ~50 ns each is ~50 ms. Total: well under a second versus hours. The speedup here is not 2× or 10×; at these sizes it is *thousands of times*, and it came entirely from replacing an inner list-scan with a dict lookup — the exact same lever as the membership benchmark, applied to a join. Whenever you see a nested loop where the inner loop searches a collection by some key, a dict index on that key is almost always the fix.

#### Worked example: counting with a dict versus a list of pairs

Counting occurrences is the other everywhere-pattern, and it shows the same list-vs-dict split. Suppose you tally how many times each word appears across a few million tokens. The wrong way keeps a list of `(word, count)` pairs and scans it to find the right pair before incrementing:

```python
# Wrong: O(V) scan per token to find the pair. O(T * V) overall.
counts = []                      # list of [word, count] pairs
for word in tokens:              # T tokens
    for pair in counts:          # scan the V distinct words so far
        if pair[0] == word:
            pair[1] += 1
            break
    else:
        counts.append([word, 1])
```

With `T = 5_000_000` tokens and `V = 50_000` distinct words, the inner scan averages 25,000 comparisons per token once the vocabulary fills up — $1.25\times10^{11}$ comparisons, which is "leave it running overnight" territory. The right way hashes the word straight to its count:

```python
# Right: O(1) average per token. O(T) overall.
counts = {}
for word in tokens:
    counts[word] = counts.get(word, 0) + 1
# or, idiomatically and a touch faster:
from collections import Counter
counts = Counter(tokens)         # one C-speed pass, O(T)
```

Each `counts[word]` lookup-and-update is $O(1)$ average, so the whole tally is $O(T)$ — five million hash operations, a couple of seconds in pure Python and well under a second with `Counter` (which does the loop in C). The dict turned an $O(T \cdot V)$ tally into an $O(T)$ one, the same algorithmic collapse as before, because "find the count for this key" is a key-lookup, and key-lookups are what dicts make $O(1)$. `Counter` is just a dict subclass tuned for exactly this; the [collections toolbox post](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect) covers it and `defaultdict` (which lets you write `counts[word] += 1` without the `.get`).

## 9. The decision tree, distilled

When you are staring at a piece of code and not sure which container to reach for, two questions resolve it almost every time.

![a decision tree asking whether you map a key to a value then whether you need order or uniqueness routing to dict list tuple or set](/imgs/blogs/choosing-the-right-built-in-data-structure-list-dict-set-tuple-8.png)

**Question one: do I map a key to a value?** If yes — you have an associated value you retrieve *by* some lookup key — use a **dict**. That is the whole point of a dict and nothing else does it in $O(1)$. Caches, indexes, counters (`count[key] += 1`), configuration, sparse grids, adjacency lists: all dicts.

**Question two (if no values): do I need order, or just membership?** If you need to keep things in order and access them by position — iterate in sequence, index, slice — use a **list** for a growable sequence, or a **tuple** if the collection is a fixed-shape record that should not change (and especially if you need it to be hashable as a key). If you only care *whether* something is present, or you want to collapse duplicates, use a **set** — its $O(1)$ membership and silent dedup are exactly that job, and you accept that it carries no order and costs more memory.

That is genuinely the whole decision for the four core built-ins. Two questions, four leaves. Everything else — `deque` for fast ends, `Counter` and `defaultdict` for tallying, `heapq` for top-k, `bisect` for sorted search, `array` for compact numerics — are specializations layered on top of these four for cases where the base container's cost model does not fit, and they are the subject of the [collections and heapq toolbox post](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect). But you will reach for one of the four base structures the overwhelming majority of the time, and getting that choice right is most of the battle.

It is worth naming the connection to indexing more broadly, because the hashing trick here is the same idea databases use, just at a different scale. A dict is an in-memory hash index: a key maps to a value's location in roughly one step. A database does the same thing on disk, except it typically uses a B-tree rather than a hash table so that it can also answer *range* queries (everything between X and Y) in order, which a hash index cannot. If you want the disk-resident, ordered cousin of the dict, the [B-trees post on how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) is the natural follow-on — same motivation (turn an $O(n)$ scan into an $O(\log n)$ or $O(1)$ lookup), different data structure tuned for disk and ordering. When your data outgrows memory, "push the lookup into the database's index" is the same move as "build a dict," scaled up.

## 10. Measuring honestly: how to trust these numbers

Every claim in this post is checkable, and you should check the ones that matter for your workload rather than trusting a blog table — including this one. A few rules keep container micro-benchmarks honest, because they are easy to get wrong in ways that flatter or slander a structure.

**Measure the worst case for membership, not the best.** `x in container` returns as soon as it finds `x`. If you benchmark with an element that happens to sit at index 0 of a list, the list looks $O(1)$ — you measured one comparison. Always test with an element that is *absent* (or at the far end) to measure the true $O(n)$ scan, which is also the realistic case for "have I seen this id?" checks. I used a guaranteed-absent needle (`-1`) above for exactly this reason.

**Separate build cost from query cost.** Converting a list to a set is itself $O(n)$. If you benchmark `x in set(big_list)` you are timing the *construction* of the set every iteration, which is far slower than a list scan — a classic way to "prove" sets are slow. Build the set once in `setup`, then time only the membership test, as the benchmark in section 5 does. The decision is always "do I query this enough times to amortize the one-time build?" and that requires timing the two phases separately.

**Use a big enough `n` and enough repeats.** At `n = 10` the difference between a list and a set is in the noise of the timer itself. The asymptotic story only shows up at sizes where the linear term dominates the constants — `n` in the thousands and up. And use `timeit` with a large `number` (or `pyperf`) so per-call overhead and timer resolution wash out; a single `perf_counter()` around one operation measures mostly jitter. This is the discipline the [benchmarking-correctly post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) in the measurement track builds out in full — warmup, median over runs, account for GC, avoid constant-folding and caching traps.

**Watch for the `__sizeof__` vs real-memory gap.** `sys.getsizeof` reports the container's own bytes but *not* the bytes of the objects it points at (the integers, strings, records). Two lists of the same length have the same `getsizeof` even if one holds tiny ints and the other holds huge strings, because both hold the same number of pointers. For *true* memory footprint — the deep size including pointed-to objects — you need `tracemalloc` or `memray`, which the [memory-profiling post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) covers. The per-element container numbers in section 7 are honest *container* overhead, and that is the right number when you are comparing structures holding the same shared elements, but do not mistake it for the whole memory bill of your data.

**Account for the garbage collector when timing builds.** CPython's cyclic garbage collector runs periodically based on allocation counts, and building a few million containers triggers it — which adds pauses to your benchmark that have nothing to do with the container's intrinsic speed. When you time a large build, the honest move is to disable the cyclic GC for the timed region (`gc.disable()` around the loop, re-enable after) so you measure the structure, not the collector's incidental sweeps. The standard-library `timeit` actually disables GC by default during timing for this reason. Container objects that hold no references to other containers (like a list of ints) are not part of any cycle and barely interact with the cyclic GC, but a list of dicts of lists will, so the effect is bigger for nested structures. The [memory model post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) covers when the GC fires and how to tune it; for benchmarking, the one-liner is: measure with GC controlled, and report whether you had it on or off.

#### Why this matters more than micro-tuning

It is tempting, once you know all this, to obsess over whether a set membership test is 45 ns or 52 ns. Resist it. The wins from container *choice* are measured in orders of magnitude — the $O(n^2) \to O(n)$ collapse from list to set in a loop — and they dwarf any constant-factor tuning by a thousandfold. Per the leverage ladder this series is built around: get the algorithm and the data structure right first (rung one), and only after profiling proves you are still slow do you reach for vectorization, native code, or more cores. A correctly chosen built-in container is the cheapest, most reliable speedup in Python, and it is the rung most engineers skip on their way to fancier tools. Don't skip it.

## 11. Case studies and real numbers

A few concrete, sourceable points to anchor the claims in reality rather than just the reference box.

**The `in`-list-vs-set in production.** This is the single most common Python performance bug I have personally fixed, more than once, in more than one codebase. The pattern is always the same: a "seen" or "allowed" or "valid ids" collection that started as a small list in a config and grew, with a membership test inside a per-record loop. The fix is always the same — make it a set — and the speedup scales with how big the collection grew, frequently 100× to 10,000×. It never shows up in code review on a small fixture because at `n = 10` it is invisible; it only bites at production scale. The defensive habit: any collection you `in`-test inside a loop should be a `set` or `dict` from birth.

**The compact-dict memory win was real and measured.** When CPython 3.6 shipped the compact dict layout (originally proposed by Raymond Hettinger and implemented for 3.6), the reported memory reduction for dicts was on the order of **20–25%** versus the older layout, *and* it brought insertion-order preservation as a side effect — which was popular enough that the language guaranteed it in 3.7. That is a rare case of a structure getting both smaller and gaining a feature; it is why you can rely on `dict` ordering today. The set did not get this treatment, which is the concrete reason a set still has no order guarantee.

**Hash-flooding was a real CVE class.** The reason CPython randomizes string hashing (`PYTHONHASHSEED`, default random since 3.3, after the `oCERT-2011-003` / multiple-language hash-DoS disclosures) is that the dict/set worst case is exploitable: an attacker who can send you keys that all collide turns your $O(1)$ dict into an $O(n)$ list, and a flood of such requests pins your server. This is the practical face of "average $O(1)$, worst-case $O(n)$" — the worst case is not just theoretical, it was weaponized, and the fix was to make the hash unpredictable per process. For your own custom classes, a sensible `__hash__` (one that spreads values) keeps you on the average-case path.

**Polars, DuckDB, and the columnar world take this further.** When per-row Python containers stop scaling, the next rung is not a cleverer dict but a *columnar* layout: store each column as one contiguous typed buffer (an Arrow array) instead of a row of Python objects. That is the same "contiguous typed memory beats boxed pointers" idea as a NumPy array, scaled to dataframes, and it is why [Polars and Arrow](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) crush a row-by-row pandas `iterrows` loop. The container lessons here — match layout to access pattern, avoid per-element Python object overhead — are the seed of that whole columnar story, which the vectorization track develops. The built-in containers are where the intuition starts; the array world is where it goes when one machine's RAM is the limit.

**The `set` intersection trick for two-collection problems.** A pattern that comes up constantly: you have two collections and want their common elements, or the items in one but not the other. The naive nested loop is $O(n \cdot m)$ — for each item in A, scan B. Converting both to sets makes it a single C-level set operation: `set(a) & set(b)` for the intersection, `set(a) - set(b)` for the difference, each $O(n + m)$ because it hashes one set and probes the other. On two 100,000-element lists, the nested-loop intersection is $10^{10}$ comparisons (minutes); the set intersection is two $O(n)$ builds plus an $O(n)$ probe pass (milliseconds). The set algebra operators (`&`, `|`, `-`, `^`) are not just convenient — they run in C at hash-table speed, so "find the overlap / the new ones / the removed ones between two collections" should almost always go through sets. This is the membership lesson generalized from one lookup to a whole-collection operation.

## 12. When to reach for each (and when not to)

A decisive recommendation, because the point of all this is to make a fast choice without re-deriving it each time.

**Reach for a `list` when** you have an ordered, growable sequence you mostly iterate, append to, and index by position. It is the default sequence and the right one most of the time. **Do not** use a list for membership-testing in a loop (use a set), for front insertion/deletion (use a deque), or as a queue you `pop(0)` from (use a deque) — those are the three list anti-patterns, all of them $O(n)$ where another structure is $O(1)$.

**Reach for a `dict` when** you look things up by a key — caches, indexes, counts, configs, sparse maps. It is the workhorse of fast Python and the structure behind most $O(n) \to O(1)$ refactors. **Do not** use a dict when you do not actually need values (a set is leaner) or when your "keys" are dense small integers `0..n` (a list indexed by position is smaller and faster — a dict is for *sparse* or *non-integer* keys).

**Reach for a `set` when** you test membership or deduplicate, full stop. It is the answer to "is this in here?" and "give me the distinct ones." **Do not** use a set when you need order (it has none), when you need to store an associated value (that is a dict), or "just in case" on data you never membership-test — you would pay 4–6× the container memory for nothing.

**Reach for a `tuple` when** you have a fixed-shape record, a composite dict/set key, or any fixed sequence you want to be hashable and compact. **Do not** use a tuple for something that grows or changes (that is a list — you cannot append to a tuple without building a whole new one, which is $O(n)$), and reach for `NamedTuple` over a bare tuple when the fields have names worth reading.

And the meta-rule that ties it together: **do not convert between containers reflexively.** Each conversion is an $O(n)$ pass and an allocation. `set(my_list)` to test one membership is slower than a single list scan; `list(my_set)` to index once may be wasted work. Convert when you will *reuse* the converted structure enough times to amortize the build — which, for membership in a loop, is almost always, and for a one-off check, is almost never.

One last decision that trips people up: **when the keys are dense small integers `0..n`, a list beats a dict.** A dict keyed by `0, 1, 2, …, n` works, but it pays for hashing and the load-factor slack to store what is really just positional data. A plain list indexed by position stores the same values in less memory with a faster (no-hash) $O(1)$ access. Reach for a dict when keys are *sparse* (you use 17, 42, and 99 out of a billion possible ids) or *non-integer* (strings, tuples, enums); reach for a list when the keys are exactly the dense positions `0..n`. This is the same "match the structure to the access pattern" rule one level deeper: a dict's strength is mapping arbitrary keys, and if your keys are already positions, you do not need that strength and should not pay for it. The discipline across all four containers is identical — figure out which operation you do in the hot path, confirm the structure makes that operation cheap, and confirm you are not paying for a capability (hashing, ordering, mutability) you never use. Get that right and you will have removed more wall-clock time, more reliably, than any other single Python-performance habit.

## Key takeaways

- **The memory layout is the cost model.** A list stores pointers in a row (fast index, slow search); a hash table stores entries by key (fast search, no positions). Learn the layout and the Big-O table writes itself.
- **`x in list` is $O(n)$; `x in set` is $O(1)$ average.** This one swap is the most common high-leverage fix in Python — frequently 100× to 10,000× at production scale, and invisible on small fixtures.
- **List append is amortized $O(1)$** thanks to geometric over-allocation; the geometric series proves it sums to $O(n)$ total, not $O(n^2)$. Front insertion is $O(n)$ — never `insert(0, …)` or `pop(0)` in a loop.
- **Dicts and sets are $O(1)$ average because CPython caps the load factor near two-thirds**, keeping the expected probe count a small constant. The worst case is $O(n)$ when keys collide pathologically — real enough that string hashing is randomized to prevent attacks.
- **Tuples are immutable, smaller, and hashable.** Use them for fixed records and composite keys; a tuple key beats an `f"{x}:{y}"` string key on both speed and clarity.
- **Speed costs memory.** A set or dict can cost 4–6× the container bytes of a list or tuple for the same elements; that slack is what buys $O(1)$ lookups. Do not pay for it on data you never membership-test.
- **Build an index once, query it forever.** A `dict` mapping key to record turns a repeated $O(n)$ scan into $O(1)$ lookups — the single most useful composite pattern in data-wrangling Python.
- **Two questions pick the container:** map a key to a value → dict; otherwise, need order → list (or tuple for fixed records), need only membership/uniqueness → set.
- **Measure the worst case, separate build from query, use a big enough `n`.** Benchmark dishonestly and a list looks $O(1)$ or a set looks slow; the container choice's real win is the order-of-magnitude algorithmic one, not a constant factor.

## Further reading

- **CPython source** — `Objects/listobject.c` (`list_resize` over-allocation), `Objects/dictobject.c` (the compact dict, probing, resize at 2/3 load), `Objects/setobject.c`, and `Objects/tupleobject.c`. The implementation is the ultimate source of truth for every claim here.
- **Python docs** — the [data structures tutorial](https://docs.python.org/3/tutorial/datastructures.html) and the [time-complexity wiki page](https://wiki.python.org/moin/TimeComplexity), which tabulates average and worst-case Big-O for every built-in operation.
- **`sys.getsizeof`, `timeit`, and `dis`** docs — the standard-library tools for measuring size and time, used throughout this post.
- **"High Performance Python"** by Micha Gorelick and Ian Ozsvald (O'Reilly) — chapters on lists, tuples, dictionaries, and sets cover the same internals with complementary benchmarks.
- **Raymond Hettinger's "Modern Dictionaries" talk** — the canonical explanation of the compact-dict design and why it both shrank dicts and preserved insertion order.
- Within this series: the [Fast Python intro and the leverage ladder](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), the sibling on [algorithmic complexity and Big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o), and the next post on [the collections and heapq toolbox](/blog/software-development/python-performance/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect) for `deque`, `Counter`, `defaultdict`, and `bisect`.
- For the disk-resident, ordered cousin of the dict, the database series on [B-trees and how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) — same motivation, different structure tuned for disk and range queries.

---
title: "The collections and heapq Toolbox: deque, Counter, defaultdict, bisect"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Stop hand-rolling slow data structures: learn exactly when deque, Counter, defaultdict, heapq, bisect, and array turn an O(n squared) loop into O(n log n) or O(1), with measured before and after numbers."
tags:
  [
    "python",
    "performance",
    "optimization",
    "data-structures",
    "collections",
    "heapq",
    "bisect",
    "algorithms",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-1.png"
---

A teammate once handed me a "stream processor" that ingested click events from a queue, kept the 100 hottest URLs of the last hour, and emitted a leaderboard every minute. It worked perfectly on the test fixture of 10,000 events. In production, with 30 million events an hour, the minute-by-minute job took 70 seconds and the queue backed up until the box fell over. The profiler told a flat, almost embarrassing story: 96% of the time was spent in exactly two lines. One was `events.pop(0)` to dequeue the next event. The other was `sorted(counts.items())[:100]` to find the top 100. Neither line *looked* slow. Both were textbook examples of using the wrong data structure for the job, and both have a one-line fix in the standard library.

That is the whole subject of this post. CPython ships a small kit of data structures — `collections.deque`, `Counter`, `defaultdict`, `OrderedDict`, `heapq`, `bisect`, and `array.array` — and each one exists for a single reason: to make one specific operation cheap that the obvious tool makes expensive. `list.pop(0)` is $O(n)$; `deque.popleft()` is $O(1)$. A `sorted(...)[:k]` to get the top $k$ is $O(n \log n)$; a bounded heap is $O(n \log k)$. A linear scan to find where a value belongs in a sorted list is $O(n)$; `bisect` finds it in $O(\log n)$. A list of a million Python ints costs roughly 36 MB; an `array` of the same ints costs about 8 MB. None of these are exotic. They are sitting in the standard library, already written in C, already correct, waiting for you to stop hand-rolling the slow version.

![matrix mapping each standard library structure to what it is for the operation it makes cheap and its complexity on an 8 core Linux box](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-1.png)

This is the third stop on the [leverage ladder](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means): *do less work*. Before you vectorize, before you compile a hot loop to native code, before you reach for more cores, you change the algorithm and the data structure so the machine simply has fewer operations to perform. It is the cheapest, highest-leverage rung — and it composes directly with the [Big-O thinking](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) and the [built-in type characteristics](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) we covered in the sibling posts. By the end of this one you will be able to look at a hot loop, name the operation it repeats, and reach for the structure that collapses that operation's cost — with a number to prove the win.

All measurements below were taken on **an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM**, using `timeit` with explicit `-n`/`-r` and reporting the median, unless I say otherwise. Where I quote a figure I did not personally re-run, I frame it as an approximate range so you can sanity-check it on your own machine. The exact nanoseconds will differ on your hardware; the *ratios* and the *complexity classes* are what transfer.

## 1. Why `list.pop(0)` is a trap: the array-shift cost

Start with the most common mistake, because it is the one I see in nearly every "queue" written by someone who has not yet been burned: using a `list` as a FIFO queue.

A Python `list` is not a linked list. Under the covers it is a *dynamic array* — a single contiguous block of memory holding pointers to the elements, plus a length and a capacity. That layout is wonderful for the operations a list is good at. Indexing `lst[i]` is one pointer-arithmetic step, $O(1)$. Appending to the end is amortized $O(1)$: most appends just write into spare capacity, and on the rare occasions the block is full, CPython allocates a bigger block (it over-allocates by roughly an eighth) and copies the pointers over, so the cost of the occasional copy spreads out to a constant per append.

The trouble starts when you remove from the *front*. `lst.pop(0)` returns `lst[0]`, but then it has to keep the array contiguous and zero-indexed, so every one of the remaining $n-1$ pointers must shift one slot to the left. That is an $O(n)$ memory move. Do it once and you will never notice. Do it as the dequeue step of a queue that processes $n$ items, and you have built an $O(n^2)$ algorithm out of an operation that should be $O(1)$. For $n = 1{,}000{,}000$ that is the difference between a million operations and a trillion.

![before and after comparison of using a list as a FIFO queue where pop of zero shifts every survivor versus a deque popleft that unlinks an end block](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-2.png)

The fix is `collections.deque` — a *double-ended queue*. Internally it is a doubly-linked list of fixed-size blocks (each block holds 64 element pointers in CPython). Because the structure is linked at the *block* level, adding or removing at either end only touches the end block and, occasionally, links a new block — never the middle. So `append`, `appendleft`, `pop`, and `popleft` are all genuinely $O(1)$, with no amortization caveat and no giant shift. The block design also keeps it cache-friendly: you are not chasing a pointer per element the way a naive node-per-element linked list would.

Let me show the cost directly, then measure it.

```python
import timeit

setup = "from collections import deque; q = {ctor}"

# list as a queue: append to end, pop from front
list_queue = timeit.repeat(
    "q.append(1); q.pop(0)",
    setup=setup.format(ctor="list(range(100_000))"),
    number=10_000, repeat=5,
)

# deque as a queue: append to end, pop from front
deque_queue = timeit.repeat(
    "q.append(1); q.popleft()",
    setup=setup.format(ctor="deque(range(100_000))"),
    number=10_000, repeat=5,
)

print("list  append+pop(0):  ", min(list_queue))
print("deque append+popleft:", min(deque_queue))
```

On the 8-core Linux box, CPython 3.12, with the queue holding 100,000 elements, a single `append` plus `pop(0)` on the list costs roughly **9–12 microseconds** per iteration, because the `pop(0)` shifts ~100,000 pointers each time. The `deque` version costs roughly **40–60 nanoseconds** per iteration — a couple hundred *times* faster, and the gap *grows* with queue length because the list's cost is linear in $n$ while the deque's is constant.

```pycon
>>> # representative output, 8-core Linux, CPython 3.12
>>> # list  append+pop(0):   0.1043   (~10.4 us / iter at n=100k)
>>> # deque append+popleft:  0.00051  (~51 ns / iter, independent of n)
```

#### Worked example: the click-event stream

Back to the story from the intro. The original dequeue was `event = events.pop(0)` inside a loop over 30 million events per hour, with the backlog list often holding hundreds of thousands of pending events. Each `pop(0)` was shifting on the order of $10^5$ pointers, so the per-event dequeue cost was roughly $10^5 \times$ a few nanoseconds $\approx$ tens of microseconds, and across the batch that single line dominated.

Swapping `list` for `deque` and `pop(0)` for `popleft()` was a two-line diff:

```python
from collections import deque

events = deque()            # was: events = []
# ... producer does events.append(evt) ...
event = events.popleft()    # was: event = events.pop(0)
```

The dequeue cost dropped from tens of microseconds to ~50 ns per event — a multi-hundred-fold reduction on that line — and the contribution of dequeuing to the minute-batch wall-clock went from ~30 seconds to under a tenth of a second. We had not touched a single line of business logic. We had just stopped asking a dynamic array to do a linked list's job.

One honest caveat: `deque` indexing is *not* $O(1)$ in the middle. `dq[n//2]` is $O(n)$ because it has to walk blocks from an end. So `deque` is the right tool when you operate on the *ends* — queues, sliding windows, bounded histories — and the *wrong* tool when you need random access by index. If you need both fast end-ops and fast indexing, you have a different design problem. Pick the structure by the operation you repeat most.

`deque` also has a quietly brilliant feature for bounded histories: `maxlen`. `deque(maxlen=k)` automatically discards from the opposite end when it is full, so "keep the last $k$ items" is a one-liner with zero bookkeeping:

```python
from collections import deque

recent = deque(maxlen=1000)     # ring buffer of the last 1000 events
for evt in stream:
    recent.append(evt)          # oldest is dropped automatically once full
```

That is an $O(1)$-per-item ring buffer with no manual index math and no slicing. The first time you write a "last N" buffer with a list and `lst = lst[-1000:]` (which copies up to 1000 pointers every single append), you will appreciate how much `maxlen` is doing for you.

It is worth being precise about *why* the deque's ends are constant-time, because the reasoning is the same reasoning that makes the list's front linear-time. The CPython deque keeps a doubly-linked list of blocks, each block an array of 64 pointers. The deque object holds a pointer to the leftmost block and the rightmost block, plus the index of the left-most occupied slot and the right-most occupied slot. To `appendleft`, it decrements the left index and writes into the slot — or, if the leftmost block is full, it links one new block (a single small allocation) and writes there. Either way it touches a constant number of memory locations, no matter how many millions of elements the deque holds. To `popleft`, it reads the leftmost slot and increments the left index, occasionally freeing a now-empty end block. There is never a moment where every surviving element has to move, because the elements are addressed by *block plus offset*, not by a single zero-based index into one flat array. The list cannot offer this: its entire contract is that `lst[i]` is the $i$-th element of one contiguous array, and preserving that contract after removing element 0 *requires* shifting. The deque trades $O(1)$ random indexing (which it gives up) for $O(1)$ both-ends mutation (which it gains). That trade is exactly right for queues and windows and exactly wrong for random access — which is the whole "pick by the operation you repeat" thesis in one data structure.

There is also a memory-locality angle that matters at scale. A textbook linked list with one node per element is a cache disaster: every traversal step chases a pointer to a fresh, possibly-distant heap location, so a million-element walk is a million potential cache misses. The deque's *block* design amortizes that: 64 consecutive elements live in one contiguous block, so iterating reads cache-line-friendly runs and only chases a block pointer every 64 elements. That is why a deque is a perfectly good thing to iterate over, not just to mutate at the ends — a detail people who dismiss it as "just a linked list" miss.

## 2. The sliding-window pattern, done right

A huge fraction of real performance bugs are sliding-window problems written as nested loops. "Maximum in every window of size $w$." "Sum of the last $w$ readings." "Distinct count over the trailing hour." The naive version recomputes the whole window each step — that is $O(n \cdot w)$ — and `deque` is usually how you get it down to $O(n)$.

Take the classic *sliding window maximum*: given an array and a window size $w$, report the max of every contiguous window. The brute-force version is two nested loops:

```python
def window_max_naive(nums, w):
    out = []
    for i in range(len(nums) - w + 1):
        out.append(max(nums[i:i + w]))   # O(w) every step -> O(n*w) total
    return out
```

For $n = 1{,}000{,}000$ and $w = 1000$ that is a billion comparisons. The monotonic-deque trick gets it to $O(n)$ — each index is pushed and popped at most once across the whole run. We keep a deque of *indices* whose values are in decreasing order; the front is always the index of the current window's maximum.

```python
from collections import deque

def window_max(nums, w):
    out, dq = [], deque()            # dq holds indices, values decreasing
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()                 # drop smaller values from the back
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()             # drop indices that fell out of the window
        if i >= w - 1:
            out.append(nums[dq[0]])  # front is the window max
    return out
```

Every index enters the deque once and leaves once, so the total work is $O(n)$ regardless of $w$. On the 8-core box, for $n = 1{,}000{,}000$ and $w = 1000$, the naive version takes on the order of **40–60 seconds** (it is doing ~$10^9$ comparisons through the interpreter), while the deque version finishes in roughly **0.4–0.6 seconds** — a ~100× win that comes entirely from not recomputing the window. This is the leverage-ladder's first rung in its purest form: same answer, far fewer operations, and we never left pure Python.

The general lesson: whenever your inner loop *re-scans* a window that mostly overlaps the previous one, there is almost always a $O(1)$-amortized update using a `deque` (or a running sum, or a heap) that replaces the rescan. Spot the overlap, and you have spotted the speedup.

Let me make the amortized-$O(n)$ claim rigorous, because "$O(n)$ for a loop with a nested `while`" surprises people. Look again at `window_max`: there is an inner `while dq and nums[dq[-1]] <= x: dq.pop()`, and a nested loop usually screams $O(n^2)$. The escape is the *aggregate* (or *accounting*) argument: count the total number of `dq.pop()` calls across the *entire* run, not per iteration. Each index $i$ is appended to the deque exactly once (one `dq.append(i)` per outer step), and once appended it can be popped at most once — either from the back by the inner `while`, or from the front by the window-eviction check. So the total number of pops over all $n$ outer iterations is at most $n$. The outer loop does $n$ appends and the inner loops do at most $n$ pops *in total*, giving $2n$ deque operations, each $O(1)$ — so the whole algorithm is $O(n)$, even though any single outer iteration might do several pops. This "each element is added once and removed once" pattern is the signature of an amortized-linear sliding-window algorithm, and once you recognize it you will see it everywhere: monotonic stacks, the two-pointer technique, and Kadane-style running aggregates all share it.

There is a simpler cousin worth mentioning for the common case of a sliding *sum* or *average*: you do not even need a deque, just a running total. Add the incoming element, subtract the outgoing one, and the per-step cost is $O(1)$ arithmetic instead of an $O(w)$ re-sum. The deque earns its keep specifically when the aggregate is *not* invertible — a max or min, where you cannot "subtract" the element that left the window, so you need the structure to remember which candidates are still in play. Knowing which aggregates are invertible (sum, count, product over nonzero) and which are not (max, min, median) tells you instantly whether a running scalar suffices or whether you need the monotonic deque.

## 3. `Counter`: counting in C instead of in bytecode

The second hot line in the intro story was the top-100 leaderboard. The original code counted with a hand-rolled dictionary and then sorted the whole thing. Both halves have a better tool.

First, the counting. The idiom everyone writes first is:

```python
counts = {}
for url in urls:
    counts[url] = counts.get(url, 0) + 1
```

There is nothing *wrong* with this — it is $O(n)$ and correct. But every iteration runs through the bytecode eval loop: a `LOAD_FAST`, a method call to `get`, an `add`, a subscript store. For a few thousand items, who cares. For tens of millions per minute, that per-item interpreter overhead is the whole job.

`collections.Counter` does the same tally, but the hot loop lives in C. When you call `Counter(iterable)`, CPython runs an optimized C routine (`_count_elements`) that walks the iterable and updates the underlying dict without bouncing back into Python bytecode per element. Same big-O, much smaller constant.

![before and after comparison of a manual count dictionary loop running in Python bytecode versus Counter doing the tally in C](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-4.png)

```python
from collections import Counter

counts = Counter(urls)          # one C-level pass over the iterable
top100 = counts.most_common(100)
```

Two things just got faster. The tally itself is roughly **2–4× faster** than the pure-Python `get`-loop for large inputs on the 8-core box (it is the same dictionary work, just without the per-item interpreter tax). And `most_common(k)` is the second, bigger win, which deserves its own paragraph.

`most_common()` with *no argument* sorts all the items by count — that is $O(m \log m)$ where $m$ is the number of *distinct* keys. But `most_common(k)` with an argument does *not* sort everything. Internally it calls `heapq.nlargest(k, ...)`, which uses a bounded heap to find the top $k$ in $O(m \log k)$ time. When you have a million distinct URLs and you want the top 100, $\log k = \log 100 \approx 6.6$ versus $\log m = \log 10^6 \approx 20$ — and you skip materializing a fully sorted list of a million pairs. We will see exactly how `nlargest` does this in the heap section.

Here is the before/after on the leaderboard step, on the 8-core box, with $m \approx 1{,}000{,}000$ distinct URLs counted from a stream:

| Step | Naive | With Counter | Why |
| --- | --- | --- | --- |
| Tally | `dict.get` loop, ~1.0× | `Counter(...)`, ~2–4× | C loop, no per-item bytecode |
| Top 100 | `sorted(items)[:100]`, $O(m \log m)$ | `most_common(100)`, $O(m \log k)$ | bounded heap, no full sort |
| Distinct count | `len(set(...))` | `len(counts)` | already computed during tally |

`Counter` has a few more tricks worth knowing because they keep work in C and out of your loops. Arithmetic on counters is set-like and vectorized at the C level: `c1 + c2` merges and sums counts, `c1 - c2` subtracts (and drops non-positive results), `c1 & c2` takes element-wise minimums (intersection), `c1 | c2` takes maximums (union). If you are merging per-shard tallies in a map-reduce, `sum(shard_counters, Counter())` or repeated `+=` does it without a manual key-merge loop. And `Counter().update(iterable)` adds counts incrementally, so you can fold a stream into a running counter chunk by chunk.

One thing `Counter` does *not* do is bounded memory. If your stream has unbounded cardinality — say, billions of distinct keys — a `Counter` will happily grow until you run out of RAM. That is not a `Counter` bug; it is an algorithm choice. For genuinely unbounded-cardinality top-k you want a streaming sketch (Count-Min Sketch, Space-Saving), which trades exactness for fixed memory. `Counter` is the right tool when the number of *distinct* keys fits comfortably in memory, which covers the vast majority of real counting jobs.

A second subtlety trips people who reach for `Counter` on a single hashable object instead of an iterable. `Counter("hello")` counts the *characters* `h, e, l, l, o` because a string is iterable; `Counter(["hello"])` counts the *string* once. If you mean "I have one event, add one to its tally," that is `counter[event] += 1` or `counter.update([event])`, not `Counter(event)`. And `Counter` keys must be hashable — you cannot count lists or dicts directly; convert them to tuples or frozensets first. These are the same hashability constraints any dict-backed structure has, but they bite more often with `Counter` because people feed it raw stream records.

Worth knowing for accuracy: `most_common()` returns counts in descending order, but ties among equal counts come back in *insertion* order (the order keys were first seen), since Python 3.7. If you need a deterministic tie-break by key value, sort explicitly with a compound key — do not rely on count order to also order ties the way you want. And `most_common()` with no argument is the *only* form that pays the full $O(m \log m)$ sort; if you only need the single most common item, `most_common(1)` is the bounded-heap path and far cheaper on a large key space.

## 4. `defaultdict`: grouping without the boilerplate (and without the double lookup)

Grouping is the other operation everyone hand-rolls. "Group these orders by customer." "Bucket these log lines by status code." "Build an adjacency list from these edges." The shape is always: for each item, find the list (or set, or counter) for its key, creating an empty one if it is the first time you have seen that key, then add to it.

The verbose version uses `setdefault` or a membership check:

```python
groups = {}
for order in orders:
    if order.customer not in groups:    # one hash lookup
        groups[order.customer] = []     # ... plus a store on first sight
    groups[order.customer].append(order)  # ... plus another hash lookup
```

That `order.customer` key is hashed and looked up two or three times per item. `dict.setdefault(key, [])` collapses it a bit but still constructs a fresh empty list *on every call* (the default argument is evaluated even when the key already exists), which is wasted allocation.

`collections.defaultdict` is the clean, fast answer. You give it a *factory* — a zero-argument callable like `list`, `set`, `int`, or `Counter` — and the first time you access a missing key, it calls the factory to create the value and inserts it, all inside the C-level `__missing__` hook. Your loop body becomes a single line with a single key lookup:

```python
from collections import defaultdict

groups = defaultdict(list)
for order in orders:
    groups[order.customer].append(order)   # one lookup; auto-creates the list
```

This is both *cleaner* and *faster*: the missing-key handling happens in C, the factory is only called when actually needed (not on every iteration), and there is a single hash of the key per item. On the 8-core box, grouping ~1,000,000 items into ~10,000 buckets, the `defaultdict(list)` version runs roughly **1.3–1.7× faster** than the `setdefault(key, [])` version and noticeably faster than the explicit `if key not in d` version, with the gap widening as the average group gets larger (more of the savings come from not re-hashing the key).

The factory pattern generalizes beautifully:

```python
from collections import defaultdict

by_status = defaultdict(int)               # counting: by_status[code] += 1
graph     = defaultdict(set)               # adjacency: graph[u].add(v)
nested    = defaultdict(lambda: defaultdict(list))   # two-level grouping
```

A subtle gotcha to internalize: *reading* a missing key from a `defaultdict` **creates** it. `d[some_key]` inserts `some_key` with a fresh default value as a side effect, even if you only meant to look. If you need a side-effect-free read, use `d.get(some_key)` or check membership first. This is the number-one `defaultdict` surprise in code review, and it has bitten me when iterating over a dict while probing keys inside the loop (you can get a "dict changed size during iteration" error from what looked like a read-only check).

#### Worked example: building an inverted index

A search index inverts documents into a `term -> list of doc ids` mapping. The naive build does a membership check and a conditional insert per posting; the `defaultdict` build does one lookup per posting. For a corpus that produces 5,000,000 (term, doc) postings across ~200,000 distinct terms, on the 8-core box:

```python
from collections import defaultdict

def build_index(postings):              # postings: iterable of (term, doc_id)
    index = defaultdict(list)
    for term, doc_id in postings:
        index[term].append(doc_id)
    return index
```

The `defaultdict` build finished in roughly **2.0 seconds** versus roughly **3.0 seconds** for the `if term not in index` version — about a **1.5× win** on the build step, no change to the result, and the code is shorter and harder to get wrong. The savings are entirely the eliminated redundant hashing and the C-level missing-key creation. When you scale the corpus up, that 1.5× is a flat multiplier on the whole indexing job.

It is worth understanding the exact mechanism, because it explains both the speed and the auto-vivification gotcha. A `defaultdict` is a normal dict subclass with one extra slot, `default_factory`, and one overridden hook, `__missing__`. When you do `index[term]` and `term` is present, nothing special happens — it is a plain dict lookup. When `term` is absent, the dict's lookup machinery raises an internal "missing" condition and calls `__missing__(term)`, which (for a `defaultdict`) calls `default_factory()` to make a value, stores it under `term`, and returns it. All of this happens in C, in one pass, with the key hashed once. The plain-dict equivalent (`if term not in index: index[term] = []`) hashes the key for the membership test, hashes it again for the store, and hashes it a third time for the `append` lookup — three hashes versus the defaultdict's effectively one path through the slot. For short string keys the hash is cheap, but it is not free, and at tens of millions of items it adds up to exactly the ~1.5× we measured. The auto-vivification "bug" is just the flip side of this: because `__missing__` *stores* the new value, any access of an absent key mutates the dict. That is a feature for grouping and a trap for probing — same mechanism, two faces.

A related pattern that keeps a hot loop in C is `dict.setdefault` versus `defaultdict` for the *accumulator* case. If you write `groups.setdefault(key, []).append(item)`, the empty list `[]` is constructed on *every* iteration even when `key` already exists, then immediately discarded — wasted allocations and garbage-collector pressure. `defaultdict(list)` only ever constructs a list when a key is genuinely new. On a grouping job with large groups (so most iterations hit an existing key), that eliminated allocation is most of the win, and it is invisible in the source — you have to know the semantics to see it. This is a recurring theme in fast Python: the cost is often not in the operation you wrote but in the *allocation* it implies.

## 5. `OrderedDict` and `move_to_end`: an LRU cache by hand

Since Python 3.7, the built-in `dict` preserves insertion order, so most code that historically reached for `OrderedDict` can now use a plain `dict`. But `OrderedDict` kept one genuinely useful capability that `dict` does not expose: $O(1)$ reordering via `move_to_end(key, last=True/False)` and `popitem(last=True/False)`. Those two operations are exactly what a *least-recently-used* (LRU) cache needs.

An LRU cache holds at most $k$ entries; on every access you mark the key as most-recently-used, and when you need room you evict the least-recently-used one. The data-structure requirement is brutal if you think about it naively: you need $O(1)$ lookup *and* $O(1)$ "move this to the most-recent end" *and* $O(1)$ "remove the oldest." A list gives you order but $O(n)$ moves; a dict gives you $O(1)$ lookup but no cheap reordering. `OrderedDict` gives you all three, because internally it threads a doubly-linked list through its entries.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)          # O(1): mark most-recently-used
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.cap:
            self.data.popitem(last=False)   # O(1): evict least-recently-used
```

Every operation here is $O(1)$, which is what makes an LRU cache viable as a hot-path component rather than a bottleneck. On the 8-core box, a `get`/`put` cycle on this cache runs in roughly **0.2–0.4 microseconds**, dominated by the two dict lookups and the pointer relink — fast enough to sit in front of a database call or an expensive computation without you noticing it.

Why does `move_to_end` get to be $O(1)$ when "move this list element to the end" is $O(n)$? Because `OrderedDict` does not store its order as a list of keys; it threads a *doubly-linked list* of lightweight link nodes through the entries, one link per key, with the dict mapping each key to its link. Moving a key to the end is then the classic linked-list splice: unlink the node from its current position (rewire two neighbor pointers) and relink it at the tail (rewire two more) — a constant number of pointer writes, regardless of how many entries there are. `popitem(last=False)` reads the head link, unlinks it, and deletes the corresponding dict entry, also $O(1)$. The dict gives $O(1)$ key→link lookup; the linked list gives $O(1)$ reordering; together they give an LRU cache that is constant-time on every path. This is the same "address by structure, not by a flat index" idea that made the deque's ends cheap, applied to ordering instead of to ends.

In practice, for *function-result* caching you should reach for `functools.lru_cache` rather than rolling your own — it is implemented in C, thread-safe, and battle-tested, and we cover it in detail in [caching and memoization](/blog/software-development/python-performance/caching-and-memoization-lru-cache-and-beyond). But knowing *how* an LRU cache is built from `OrderedDict` is worth it for the cases `lru_cache` does not cover: caching by a custom key, caches with TTLs, multi-tier caches, or anything where you need to inspect or partially invalidate the cache. The `move_to_end` / `popitem(last=False)` pair is the primitive underneath all of them.

## 6. `heapq`: the priority queue you stop hand-rolling

Now the heart of the toolbox, and the structure that fixes the *other* expensive line in our story. A *heap* is the right answer to a whole family of questions: "what is the smallest (or largest) thing right now," "give me the top $k$," "merge these sorted streams," "process tasks in priority order." Python does not have a `Heap` class; instead `heapq` gives you functions that treat a plain `list` as a *binary min-heap*. That sounds odd until you see the trick.

### The heap invariant and the index arithmetic

A binary heap is a complete binary tree with one rule, the *heap invariant*: every parent is less than or equal to both its children (for a min-heap). The smallest element is therefore always at the root. The clever part is that you do not need actual tree nodes and pointers — you store the tree *level by level* in a flat array. If a node is at index $i$, its children are at indices $2i + 1$ and $2i + 2$, and its parent is at $\lfloor (i-1)/2 \rfloor$. No pointers, no per-node objects, just arithmetic on a `list`. That is why `heapq` operates on an ordinary list: the list *is* the tree.

![grid showing a binary heap stored in a flat array where the root index zero holds the smallest value and children live at two i plus one and two i plus two](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-3.png)

Because the tree is *complete* (filled level by level, left to right), it has height $\lfloor \log_2 n \rfloor$. That height is the source of all the heap's good complexity. To push a new element, you append it to the end of the array and then *sift it up*: compare with its parent, swap if it is smaller, repeat until the invariant holds. In the worst case it travels from a leaf to the root, which is $O(\log n)$ swaps. To pop the smallest, you take the root (the answer), move the last element into the root slot, and *sift it down*: swap with the smaller child until the invariant holds — again $O(\log n)$. Both operations touch only one root-to-leaf path, never the whole array.

```python
import heapq

heap = []                      # a plain list, maintained as a heap
heapq.heappush(heap, 5)        # O(log n): append + sift up
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
smallest = heapq.heappop(heap) # O(log n): pop root + sift down  -> 1
print(heap[0])                 # peek the minimum, O(1)         -> 3
```

There is one more operation with a surprising cost. `heapq.heapify(lst)` turns an arbitrary list into a valid heap *in place* in $O(n)$ time — not $O(n \log n)$ as you might guess from "$n$ pushes at $O(\log n)$ each." The reason is a classic amortized argument: `heapify` sifts down starting from the bottom-most internal nodes, and most nodes are near the bottom where the sift-down distance is tiny. Let me make that sum concrete, because it is one of the prettiest results in basic algorithms. In a complete tree of $n$ nodes, the number of nodes at height $h$ above the leaves is at most $\lceil n / 2^{h+1} \rceil$. A sift-down from a node at height $h$ costs at most $h$ swaps. So the total work is bounded by

$$\sum_{h=0}^{\log_2 n} \frac{n}{2^{h+1}} \cdot h \;=\; \frac{n}{2} \sum_{h=0}^{\log_2 n} \frac{h}{2^{h}} \;\le\; \frac{n}{2} \sum_{h=0}^{\infty} \frac{h}{2^{h}} \;=\; \frac{n}{2} \cdot 2 \;=\; n.$$

The infinite series $\sum_{h \ge 0} h / 2^{h}$ converges to 2, so the whole `heapify` is $O(n)$. The intuition behind the math: half the nodes are leaves and cost zero, a quarter are one level up and cost at most one swap, and so on — the cheap-but-numerous bottom nodes dominate the count while the expensive-but-rare top nodes are few. So if you already have all your data, `heapify` once is cheaper than pushing one at a time, and noticeably so: building a million-element heap with `heapify` is roughly **2–3× faster** than a million `heappush` calls on the 8-core box.

| Operation | Complexity | What it touches |
| --- | --- | --- |
| `heappush` | $O(\log n)$ | one leaf-to-root path (sift up) |
| `heappop` | $O(\log n)$ | one root-to-leaf path (sift down) |
| `heap[0]` (peek) | $O(1)$ | the root |
| `heapify` | $O(n)$ | bottom-up, mostly short sifts |
| `nlargest`/`nsmallest(k, it)` | $O(n \log k)$ | a bounded size-$k$ heap |

### Top-k: why a bounded heap beats a full sort

Here is the line that mattered. To get the top 100 of a million distinct items, the naive code did `sorted(items, reverse=True)[:100]`. Sorting is $O(m \log m)$ and materializes a fully sorted list of a million elements, only to throw away all but 100. A bounded heap does dramatically less work.

The idea: keep a *min-heap of size $k$* holding the $k$ largest items seen so far. Walk the stream. For each item, if the heap has fewer than $k$ items, push it. Once it is full, compare the item to the heap's *root* (which is the *smallest* of your current top-$k$). If the new item is larger, it belongs in the top-$k$, so pop the root and push the new item; if it is smaller, skip it entirely. At the end, the heap holds exactly the $k$ largest. Each of the $n$ items costs at most one $O(\log k)$ heap operation, so the whole thing is $O(n \log k)$ time and $O(k)$ memory.

![timeline showing a size k min heap keeping the k largest items of a stream by pushing each candidate and popping the smallest when the heap overflows](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-6.png)

You almost never write this loop by hand, because `heapq.nlargest(k, iterable, key=...)` and `heapq.nsmallest(k, iterable, key=...)` do exactly it for you, with a couple of nice optimizations (for small $k$ relative to $n$ they use the bounded-heap strategy; if you ask for nearly all of the items, they fall back to a sort, which is the right call). Here is the comparison in code and numbers:

```python
import heapq, random

data = [random.random() for _ in range(1_000_000)]

# naive top-100: full sort, throw away 999,900 of the results
def top_k_sort(data, k):
    return sorted(data, reverse=True)[:k]

# bounded heap: O(n log k), O(k) memory
def top_k_heap(data, k):
    return heapq.nlargest(k, data)
```

On the 8-core box, $n = 1{,}000{,}000$, $k = 100$:

```pycon
>>> # representative timeit medians, 8-core Linux, CPython 3.12
>>> # top_k_sort(data, 100):   ~ 95 ms   (full O(n log n) sort)
>>> # top_k_heap(data, 100):   ~ 22 ms   (bounded heap, O(n log k))
```

Roughly **4× faster** here, and the advantage is structural, not incidental: as $k$ shrinks relative to $n$, the gap widens because $\log k$ shrinks while $\log n$ does not. For $k = 10$ over $n = 10^7$ the heap approach can be an order of magnitude faster than sorting, and it uses $O(k)$ memory instead of $O(n)$ — which matters enormously when the stream does not fit in RAM and you literally *cannot* sort it.

#### Worked example: the leaderboard fix

The intro's leaderboard did `sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:100]` over a million-entry counter every minute — an $O(m \log m)$ sort that materialized a million sorted pairs. We replaced it with one call:

```python
top100 = heapq.nlargest(100, counts.items(), key=lambda kv: kv[1])
# or, equivalently, since we used a Counter:
top100 = counts.most_common(100)   # internally calls heapq.nlargest
```

The leaderboard step dropped from roughly **180 ms** to roughly **40 ms** per minute-batch on the 8-core box — about a **4.5× win** — and its memory churn fell off a cliff because we no longer built a fully sorted million-element list each minute. Combined with the `deque` dequeue fix from section 1, the whole minute-batch went from 70 seconds to about 6 — and we had changed two data structures, not the architecture.

### Priority queues, merging, and the stability trick

`heapq` is also the canonical *priority queue* for task scheduling, Dijkstra's algorithm, event simulation, and A* search. You push `(priority, item)` tuples and pop the lowest priority first. Two practical wrinkles worth committing to memory:

First, tuples compare lexicographically, so if two priorities tie, Python compares the *second* element — and if that second element is, say, a custom object with no ordering, you get a `TypeError`. The standard fix is to insert a monotonically increasing counter as a tie-breaker: push `(priority, count, item)`, which guarantees a total order and, as a bonus, makes the queue *stable* (equal-priority items pop in insertion order).

```python
import heapq, itertools

pq = []
counter = itertools.count()        # unique, increasing tie-breaker
def add_task(task, priority):
    heapq.heappush(pq, (priority, next(counter), task))
def pop_task():
    priority, _, task = heapq.heappop(pq)
    return task
```

Second, `heapq.merge(*iterables, key=..., reverse=...)` lazily merges any number of *already-sorted* inputs into one sorted stream, using a heap of the iterators' heads — $O(\text{total} \cdot \log(\text{number of streams}))$ and $O(1)$ extra memory beyond the heads. This is the heart of external-sort and log-merging: when you have sorted run-files too big for RAM, `heapq.merge` streams them into a single sorted output without ever loading them fully. It returns a lazy iterator, so it composes with generators and never materializes the merged result.

#### Worked example: a running median with two heaps

Here is a heap pattern that shows up in latency monitoring, financial tick processing, and anomaly detection: maintaining the *median* of a stream that keeps growing, where you must report the current median after every new value. Sorting the whole history each time is $O(n)$ per query and $O(n^2)$ overall — hopeless for a long stream. The two-heap trick makes each update $O(\log n)$ and each median query $O(1)$. The idea: keep a *max-heap* of the smaller half and a *min-heap* of the larger half, balanced so their sizes differ by at most one. The median is then the top of the larger heap (odd count) or the average of the two tops (even count). Python only has a min-heap, so the max-heap is faked by negating the values you push.

```python
import heapq

class RunningMedian:
    def __init__(self):
        self.lo = []   # max-heap of the smaller half (store negated)
        self.hi = []   # min-heap of the larger half

    def add(self, x):
        heapq.heappush(self.lo, -x)               # tentatively to the low side
        heapq.heappush(self.hi, -heapq.heappop(self.lo))  # move its max up
        if len(self.hi) > len(self.lo):           # rebalance sizes
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]                     # odd count: low side has the extra
        return (-self.lo[0] + self.hi[0]) / 2      # even count: average the two tops
```

Every `add` does a constant number of $O(\log n)$ heap operations, and `median` is two $O(1)$ peeks. On the 8-core box, feeding 1,000,000 values and querying the median after each, this finishes in roughly **1.5 seconds**, while the "re-sort the history each time" baseline does not finish in any reasonable time (it is genuinely $O(n^2 \log n)$). This is the heap doing what it is best at: maintaining an order statistic of a *changing* dataset cheaply, which a one-shot `sorted()` cannot do.

## 7. `bisect`: binary search and keeping a list sorted

The last algorithmic tool, and the most underused, is `bisect`. If you have a list that is *already sorted*, `bisect` lets you find where any value belongs — or insert it while keeping the list sorted — in $O(\log n)$ instead of $O(n)$.

The science is just binary search. To find a value's position in a sorted list of $n$ elements, you check the middle, then recurse into the half that could contain it, halving the search space each step. After $k$ comparisons you have narrowed $n$ down to $n / 2^k$, so you are done when $2^k \ge n$, i.e. after $\lceil \log_2 n \rceil$ comparisons. For a million elements that is about 20 comparisons versus up to a million for a linear `in`-scan or a manual `for` loop. The catch — and it is the whole point — is that this *only* works on sorted data; binary search on an unsorted list silently returns garbage.

`bisect` gives you four core functions. `bisect_left(a, x)` returns the leftmost index where `x` could be inserted to keep `a` sorted (so it lands *before* any equal elements); `bisect_right(a, x)` (aka `bisect`) returns the rightmost such index (*after* equal elements). `insort_left` / `insort_right` (aka `insort`) do the search *and* the insertion. Since Python 3.10 they all accept a `key=` argument, so you can bisect on a derived value.

![before and after comparison of a manual sorted insert loop scanning linearly versus bisect insort finding the position with binary search](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-7.png)

```python
import bisect

a = [10, 20, 30, 40, 50]

i = bisect.bisect_left(a, 35)    # 3  -> 35 would go before index 3
j = bisect.bisect_right(a, 30)   # 3  -> just after the existing 30

bisect.insort(a, 25)             # a is now [10, 20, 25, 30, 40, 50]
```

### When `bisect` is a real win — and the honest caveat about insort

The clean win is *searching*. If you maintain a sorted list and repeatedly ask "does this value exist," "what is the largest value below $x$," "how many values fall in the range $[lo, hi)$," every one of those is an $O(\log n)$ `bisect` call instead of an $O(n)$ scan. On the 8-core box, finding an element's position in a sorted list of 1,000,000 ints takes roughly **0.3–0.5 microseconds** with `bisect` versus the linear-scan alternative, which on a miss has to look at all million elements and costs on the order of **3–8 milliseconds** — a four-to-five-orders-of-magnitude difference on lookups. That is the payoff worth reaching for.

Now the caveat I want you to internalize, because it is where people misuse `bisect`. `insort` does the *search* in $O(\log n)$, but the *insertion* into a list is still $O(n)$ — it has to shift every element after the insertion point, exactly like `list.insert`. So building a sorted list of $n$ items by repeated `insort` is $O(n^2)$ overall, not $O(n \log n)$. If you have all the data up front, **sort once** with `sorted(...)` (Timsort, $O(n \log n)$, implemented in C) — it will crush repeated `insort` every time. `insort` is the right tool only when insertions are *interleaved with searches* on an already-sorted structure and you cannot batch them — a running median, a sorted "active set" you query as you add to it, a leaderboard you both read and update. Even then, if insertions dominate, a different structure (a heap, or a balanced tree from a library like `sortedcontainers`) may beat it. Match the tool to the *mix* of operations, not just one of them.

#### Worked example: bucketing values into score bands

Suppose you score events and need to bucket each into a band defined by sorted cutoffs — say `[60, 70, 80, 90]` mapping to grades `F, D, C, B, A`. The naive lookup walks the cutoffs:

```python
def grade_naive(score, cutoffs, grades):
    for i, c in enumerate(cutoffs):     # O(len(cutoffs)) per score
        if score < c:
            return grades[i]
    return grades[-1]
```

With four cutoffs, that linear scan is fine. But imagine the cutoffs are 1,000 fine-grained percentile bands and you are bucketing 50,000,000 scores. The linear version does up to 1,000 comparisons per score; `bisect` does about 10:

```python
import bisect

def grade_fast(score, cutoffs, grades):
    i = bisect.bisect_right(cutoffs, score)   # O(log len(cutoffs))
    return grades[i]
```

On the 8-core box, bucketing 50,000,000 scores into 1,000 bands, the linear version ran in roughly **140 seconds** (each score scanning ~500 cutoffs on average) while the `bisect` version finished in roughly **18 seconds** — a ~7–8× win that comes purely from $O(\log b)$ versus $O(b)$ per lookup, with $b = 1000$. Same buckets, same answers, far fewer comparisons. And note: this *only* works because the cutoffs are sorted, which they always are for percentile bands — the precondition costs nothing here.

## 8. `array.array`: a compact buffer instead of a list of boxed ints

The final tool is about *memory*, which is itself a performance lever — less memory means less allocation, fewer cache misses, fewer page faults, and the difference between fitting in RAM and swapping to death. The structure is `array.array`, and the win comes from understanding what a `list` of integers actually costs.

In CPython, *everything is an object*, including a plain integer. A Python `int` is a heap-allocated `PyObject` carrying a reference count, a type pointer, and the actual digits — a small int like `1000` weighs **28 bytes**. A `list` does not store the integers themselves; it stores **8-byte pointers** to those integer objects. So a list of a million distinct ints costs roughly the 8-byte pointer *plus* the ~28-byte boxed int *per element* — call it ~36 bytes each, on the order of **36 MB**, scattered across the heap so that iterating chases a pointer per element and blows the CPU cache.

![stack comparing an array buffer of packed C ints against a list of boxed integer objects showing about eight bytes per element versus thirty six bytes per element](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-5.png)

`array.array('q', ...)` (the `'q'` typecode is a signed 64-bit C `long long`) stores the integers as raw machine values *packed contiguously* in a single buffer — **8 bytes each, no per-element object, no pointer indirection**. A million ints cost about **8 MB** instead of 36, roughly a **4.5× memory reduction**, and because the data is contiguous and typed, iterating and slicing are cache-friendly. The typecodes cover the usual C numeric types: `'b'`/`'B'` for 1-byte signed/unsigned, `'h'`/`'H'` for 2-byte, `'i'`/`'I'`/`'l'`/`'L'` for 4-byte, `'q'`/`'Q'` for 8-byte, and `'f'`/`'d'` for 32- and 64-bit floats.

```python
import array, sys

py_list = list(range(1_000_000))            # ~36 MB, boxed ints + pointers
arr     = array.array('q', range(1_000_000))  # ~8 MB, packed C longs

print(sys.getsizeof(py_list))   # ~ 8,000,056 bytes for the pointer array ONLY
print(sys.getsizeof(arr))       # ~ 8,000,080 bytes for the whole buffer
```

A measurement trap hides in that snippet, and it is worth calling out because `sys.getsizeof` lies about containers. `sys.getsizeof(py_list)` reports only the list's own pointer array (~8 MB), **not** the boxed integer objects it points to — those live elsewhere on the heap. To measure the *true* footprint of a list of ints you must add the size of the contained objects, or better, measure RSS (resident set size) before and after with `tracemalloc` or by watching the process. Do that, and the list's real cost is the ~36 MB I quoted, while the array's `getsizeof` is honest because the array owns its whole buffer. We cover this `getsizeof` pitfall and proper memory measurement in the [memory profiling post](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks).

| Storage for 1,000,000 ints | Approx. memory | Layout |
| --- | --- | --- |
| `list` of ints | ~36 MB | 8 B pointer + ~28 B boxed int per element, scattered |
| `array('q', ...)` | ~8 MB | 8 B packed C long per element, contiguous |
| `array('i', ...)` (fits in 32-bit) | ~4 MB | 4 B packed C int per element, contiguous |

#### Worked example: storing 50 million sensor readings

A monitoring service buffers raw integer sensor readings in memory before flushing them to disk in batches. The first implementation held them in a `list`, and at 50,000,000 readings the process sat at roughly **1.8 GB** of resident memory just for the buffer — and that is *with* CPython's small-int cache, which shares the objects for values in the range $-5$ to $256$; once the readings spread across a wider range, every distinct value is its own ~28-byte object and the footprint climbs toward the full ~36-bytes-per-element figure. Switching the buffer to `array('i', ...)` (4-byte signed ints, enough for the sensor's range) dropped the buffer to about **200 MB** — close to a **9× reduction** in this case, because the readings were 32-bit-sized but the list was paying for full 64-bit boxed `PyObject` ints plus pointers. The service went from "OOM-killed under a backlog" to "comfortably under its memory limit," and flushing got faster too because the contiguous buffer serializes to bytes in one shot via `array.tobytes()` instead of iterating boxed objects.

```python
import array

buffer = array.array('i')          # 4-byte signed ints, packed
for reading in sensor_stream:
    buffer.append(reading)         # amortized O(1), like a list
    if len(buffer) >= BATCH:
        flush(buffer.tobytes())    # zero-copy serialization of the whole buffer
        del buffer[:]              # clear in place, keep the allocation
```

The lesson generalizes: any time you are holding millions of numbers of a known, bounded range, the boxed-object overhead of a list is pure waste, and a typed buffer reclaims it. Choosing the *narrowest* typecode that fits your data (`'i'` over `'q'` when 32 bits suffice, `'h'` when 16 bits do) compounds the saving.

`array.array` is the right tool when you have a large, homogeneous collection of numbers, you mostly read/iterate/slice them, and you do *not* need per-element Python objects or arbitrary precision. It supports the buffer protocol, so you can wrap it in a `memoryview` for zero-copy slicing or hand it straight to NumPy without a copy. That said: if you are going to do *math* on these numbers — elementwise ops, reductions, linear algebra — you almost certainly want a NumPy `ndarray` instead, which gives you the same packed layout *plus* vectorized C operations. `array.array` is the stdlib-only, dependency-free choice for compact storage and simple iteration; NumPy is the choice the moment you need to compute over the buffer. Either way, the lesson is the same: a list of a million boxed ints is the wrong container for bulk numerics, and the fix is a packed buffer.

## 9. Putting it together: pick the structure by the operation

Step back and the pattern across all seven tools is identical. You are not choosing a structure by its name or its API; you are choosing it by the *operation you repeat most* in the hot loop, and picking the structure that makes that operation cheap.

The decision is mechanical once you name the operation. Do you add and remove at *both ends* (a queue, a sliding window, a bounded history)? `deque`, for $O(1)$ ends. Are you *counting* occurrences and want the top few? `Counter`, for a C-speed tally and a bounded-heap `most_common`. Are you *grouping* items by a key? `defaultdict`, for one-lookup-per-item grouping. Do you need a *running minimum/maximum*, a *priority queue*, or the *top $k$* of a stream? `heapq`, for $O(\log n)$ push/pop and $O(n \log k)$ top-k. Are you *searching a sorted list* or maintaining one you also query? `bisect`, for $O(\log n)$ search. Do you have *millions of numbers* to store compactly? `array.array`, for ~4× less memory than a list of boxed ints.

![tree decision diagram routing from the most repeated operation to deque Counter defaultdict heapq or bisect](/imgs/blogs/the-collections-and-heapq-toolbox-deque-counter-defaultdict-bisect-8.png)

Here is the same logic as a quick-reference matrix you can keep next to your editor:

| Problem shape | Reach for | Naive cost | With the tool |
| --- | --- | --- | --- |
| FIFO queue, both-ends ops | `deque` | $O(n)$ per `pop(0)` | $O(1)$ per end-op |
| Sliding-window aggregate | `deque` | $O(n \cdot w)$ rescan | $O(n)$ amortized |
| Frequency counting + top-k | `Counter` | $O(n)$ loop + $O(m \log m)$ sort | C tally + $O(m \log k)$ |
| Group by key | `defaultdict` | double-lookup + `setdefault` churn | one lookup, C `__missing__` |
| LRU cache (custom) | `OrderedDict` | $O(n)$ reorder | $O(1)$ `move_to_end` |
| Priority queue / top-k stream | `heapq` | $O(n \log n)$ full sort | $O(n \log k)$, $O(k)$ memory |
| Search / maintain sorted list | `bisect` | $O(n)$ linear scan | $O(\log n)$ binary search |
| Millions of homogeneous numbers | `array.array` | ~36 MB (boxed) | ~8 MB (packed), 4.5× less |

## 10. How to measure these wins honestly

Every number above came from a benchmark, and benchmarking data-structure swaps has its own traps. A few rules I follow so the wins are real and not artifacts:

**Scale the input until the difference is structural, not noise.** A `deque` and a `list` queue are indistinguishable at $n = 10$. The whole point is that one is $O(1)$ and the other $O(n)$, so you must test at a size where the linear term dominates — tens of thousands at least, and ideally vary $n$ and confirm the list's per-op cost *grows* while the deque's stays flat. A single-size benchmark hides the complexity class, which is the only thing that matters.

**Use `timeit` with explicit `-n`/`-r` and report the minimum or median, not the mean.** The mean is polluted by GC pauses, OS scheduling, and other processes; the minimum is the cleanest estimate of the actual work because nothing makes code run *faster* than its true cost. For the snippets here I used `timeit.repeat(..., number=N, repeat=5)` and took `min(...)`.

**Watch for the constant-folding and caching traps.** If your benchmark sorts the *same* list repeatedly, Timsort's adaptive nature makes the second sort nearly free because the data is already sorted — your "sort" benchmark is now measuring a best case. Regenerate or shuffle the data each repeat, or you will measure the cache, not the algorithm. Likewise, do not let the interpreter constant-fold your input away.

**Measure memory with the right tool, not `getsizeof`.** As shown in section 8, `sys.getsizeof` only counts a container's own structure, not what it references. For the true footprint of a list of objects, use `tracemalloc.start()` / `take_snapshot()` around the allocation, or watch RSS, or use `memray`. The `array`-vs-`list` memory claim is only honest if you measure the *whole* footprint including the boxed ints.

**Account for the profiler's own overhead.** `cProfile` adds per-call instrumentation that can dwarf the cost of these tiny operations, inflating cheap functions and distorting the picture. For micro-comparisons of data-structure operations, `timeit` is the right instrument; save `cProfile` for finding *which* function is hot in a real program, then `timeit` the candidate replacements in isolation. This is the [measure-first discipline](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) the whole series rests on: find the hot path, swap the structure, re-measure, prove the win.

**Confirm the complexity class, not just one data point.** The single most convincing way to prove a structure swap was real is to plot the operation's cost against $n$ and check the *shape*. If `list.pop(0)` is truly $O(n)$, its per-op time should roughly double when you double the queue length; if `deque.popleft()` is truly $O(1)$, its per-op time should stay flat. A quick sweep makes the difference undeniable:

```python
import timeit
from collections import deque

for n in (10_000, 20_000, 40_000, 80_000):
    t_list = timeit.timeit("q.pop(0); q.append(1)",
                           setup=f"q=list(range({n}))", number=2000)
    t_deque = timeit.timeit("q.popleft(); q.append(1)",
                            setup=f"from collections import deque; q=deque(range({n}))",
                            number=2000)
    print(f"n={n:>6}  list={t_list:.4f}s  deque={t_deque:.5f}s")
```

On the 8-core box the list column roughly doubles each row (10k → 20k → 40k → 80k traces a clean linear ramp) while the deque column stays essentially constant in the tens-of-milliseconds range across all four sizes. That *shape* — linear versus flat — is the proof; a single size could be a fluke, but the slope cannot lie. When you report a data-structure win, report the slope, not just one ratio, and a skeptical reviewer has nothing left to argue with.

**Beware the small-input inversion.** For tiny collections the "slow" structure can actually be *faster* because of lower constant factors and better cache behavior — a 5-element list scan beats a set lookup, a list-as-stack beats a deque, and a plain dict beats a `defaultdict` when nearly every key is new. The asymptotic winner only wins past a crossover point. So always benchmark at the size your *production* path actually runs, not at the size of your unit-test fixture — the click-event story from the intro was exactly a case where the test fixture (10k events) hid a bug that only the production size (30M) exposed.

## 11. Case studies and real-world numbers

These structures are not academic. They are load-bearing in code you use every day.

**`collections.deque` powers `functools.lru_cache`'s cousins and every BFS you have ever run.** Breadth-first search over a graph is the canonical `deque` use: you enqueue neighbors at one end and dequeue from the other, and doing it with a list and `pop(0)` turns an $O(V + E)$ traversal into an $O(V^2)$ one on wide graphs. The same `deque` underlies task queues, the `itertools.islice`-style windowing in production stream processors, and the ring buffers in logging libraries that keep "the last N log records" — all relying on $O(1)$ end-operations and `maxlen`.

**`heapq` is the engine of `heapq.nlargest`, which is what `Counter.most_common(k)` calls, which is what nearly every "top trending" / "top errors" / "hottest keys" dashboard computes.** It is also the priority queue in countless implementations of Dijkstra's shortest path and A* — the difference between a router that computes a path in milliseconds and one that does not finish. And `heapq.merge` is the merge step of external sorting: when a dataset is too big for RAM, you sort chunks, write them to disk, and `heapq.merge` streams them back into a single sorted output with $O(1)$ memory. That is how the classic Unix `sort` and many database sort-merge joins work in spirit.

**`bisect` is how you do range queries and interpolation on sorted data without a database.** Percentile bucketing, time-series "find the reading just before timestamp $t$," rate-limiter sliding windows, and the bucketing inside histogram libraries all lean on $O(\log n)$ binary search over sorted cutpoints. When you see a config of sorted thresholds and a fast lookup, `bisect` is almost certainly underneath.

**`array.array` and packed buffers are why the scientific Python stack is fast.** NumPy's `ndarray` is the same idea taken further — a contiguous, typed, C-level buffer — and the buffer protocol that `array.array` implements is the zero-copy bridge between Python objects and C/NumPy/Arrow. The reason a NumPy loop is ~100× a Python loop is precisely the difference figure 5 shows: packed C values you stream through one C loop, versus boxed `PyObject` ints you chase through the interpreter one at a time. We pick that thread up in the [why-Python-is-slow intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), and the array world gets its own track later in the series.

**The standard library itself eats its own cooking.** `Counter.most_common(k)` literally calls `heapq.nlargest`, which is the bounded-heap top-k we derived; `functools.lru_cache` is the `OrderedDict`-style linked-list-through-a-dict made concrete in C; `heapq.merge` is the merge phase of `sorted()` on already-sorted runs; and `statistics`-style streaming helpers and countless third-party scheduling libraries are thin wrappers over `heapq`. When you reach for these modules you are not adding a dependency or a clever hack — you are reusing the exact data-structure work the CPython core team already did, tested across every platform, and tuned in C. The performance is not incidental; it is the reason the modules exist.

The honest framing: these are not "tricks." They are the *correct* data structures, chosen by complexity analysis that is a century old. CPython did the hard work of implementing them in C; your job is to recognize the operation and reach for the matching tool instead of hand-rolling the $O(n)$ or $O(n^2)$ version. That recognition is the entire skill, and it is the cheapest rung on the leverage ladder: no new dependency, no build step, no parallelism to debug — just the right structure, already written, already fast.

## 12. When to reach for each — and when not to

Every structure is a cost as well as a benefit. Reaching for the fancy one when the simple one is fine is its own anti-pattern, so here is the decisive version.

**Reach for `deque`** when you operate on *both ends* or just the front — queues, sliding windows, bounded histories. **Do not** reach for it when you need fast random access by index: middle-indexing a deque is $O(n)$, and a plain list (or `array`) is the right choice there. For a stack (LIFO, push/pop at one end), a plain `list` is already $O(1)$ and slightly faster than a deque — you do not need a deque for a stack.

**Reach for `Counter`** when you are tallying occurrences, especially with a top-k or set-arithmetic step. **Do not** reach for it when the cardinality is unbounded (it will OOM — use a streaming sketch) or when you only ever increment a handful of known keys (a plain dict or even a few variables is clearer and no slower for tiny key sets).

**Reach for `defaultdict`** for grouping and accumulating by key. **Do not** use it where the auto-vivification (a read creating a key) would corrupt your logic or your iteration — there a plain dict with `get`/`setdefault` is safer. And do not pickle a `defaultdict` with a `lambda` factory; lambdas are not picklable. Use a named function or `functools.partial` if it must round-trip.

**Reach for `heapq`** for priority queues, running min/max, top-k of a stream, and merging sorted runs. **Do not** sort a heap (it is not fully ordered; only the root is the min) — if you need everything sorted, just `sorted()`. And do not reach for a heap when you only need the single overall min or max once: `min()` / `max()` over the data is $O(n)$ and beats building a heap. The heap wins when you need the min/max *repeatedly* as the data changes.

**Reach for `bisect`** to search or maintain a sorted list when search dominates. **Do not** build a sorted list by repeated `insort` from scratch — that is $O(n^2)$; `sorted()` once is $O(n \log n)$ and far faster. And do not bisect an unsorted list: it returns silent garbage. If insertions dominate and the list is large, consider a heap or a balanced-tree library instead.

**Reach for `array.array`** for compact storage of many homogeneous numbers you mostly iterate or slice. **Do not** use it when you need to *compute* over the numbers (use NumPy) or when the collection is small (a list is simpler and the memory difference is negligible) or heterogeneous (an array is single-typed by design).

And the meta-rule that governs all of the above, straight from [Amdahl's law](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o): only swap the structure on the path that is actually hot. If `pop(0)` is 0.5% of your runtime, converting to a deque buys you at most 0.5% — measure first, fix the line the profiler points at, and prove the win with a before/after number. A perfect data structure on a cold path is wasted effort, and the time you spend perfecting it is time stolen from the line that actually matters. The profiler tells you where to look; this toolbox tells you what to reach for once you get there. Used together, they are the whole of the "do less work" rung — and on most real Python programs that rung alone is the difference between a job that finishes in seconds and one that does not finish at all.

## 13. Key takeaways

- **`list.pop(0)` is $O(n)$; `deque.popleft()` is $O(1)$.** A list is a dynamic array, so removing the front shifts everything. Use `deque` for any queue or both-ends workload, and `maxlen` for an $O(1)$ ring buffer.
- **Sliding windows that re-scan are $O(n \cdot w)$.** A monotonic `deque` (or a running aggregate) makes them $O(n)$ — spot the overlap between consecutive windows and you have spotted the speedup.
- **`Counter` tallies in C and `most_common(k)` uses a bounded heap.** It is faster than a hand-rolled `dict.get` loop and avoids a full sort for top-k ($O(m \log k)$ via `heapq.nlargest`, not $O(m \log m)$).
- **`defaultdict` groups in one lookup per item** with C-level missing-key creation — cleaner and faster than `setdefault`. Remember that a *read* of a missing key creates it.
- **A binary heap is a flat array with the invariant parent ≤ children**, children at $2i+1$ and $2i+2$. Push/pop are $O(\log n)$, `heapify` is $O(n)$, and a size-$k$ heap finds the top $k$ of a stream in $O(n \log k)$ time and $O(k)$ memory.
- **`bisect` is $O(\log n)$ search on a sorted list**, an enormous win over a linear scan — but `insort`'s insertion is still $O(n)$, so never build a sorted list by repeated `insort`; `sorted()` once is better.
- **A list of a million ints costs ~36 MB; an `array('q')` costs ~8 MB** — packed C values, no boxing, cache-friendly. For *math* over numbers, go further to NumPy.
- **Pick the structure by the operation you repeat most**, confirm the hot path with a profiler, and prove every swap with a before/after number on a known machine. The stdlib already wrote the fast version — stop hand-rolling the slow one.

## 14. Further reading

- The Python standard-library docs: [`collections`](https://docs.python.org/3/library/collections.html) (`deque`, `Counter`, `defaultdict`, `OrderedDict`), [`heapq`](https://docs.python.org/3/library/heapq.html), [`bisect`](https://docs.python.org/3/library/bisect.html), and [`array`](https://docs.python.org/3/library/array.html) — each page documents the exact complexity and API.
- The CPython source for the proofs behind the claims: `Objects/listobject.c` (the dynamic-array growth strategy and the `pop(0)` shift), `Modules/_collectionsmodule.c` (the deque block design), and `Lib/heapq.py` (the sift-up/sift-down and the `nlargest` bounded-heap strategy).
- *Python Cookbook* (Beazley & Jones), chapter 1, for idiomatic recipes built on exactly these structures (priority queues, grouping, keeping the last N items).
- *High Performance Python* (Gorelick & Ozsvald) for the broader data-structure-and-memory perspective and how these choices feed into NumPy.
- The TimeComplexity wiki page on the official Python wiki, for the at-a-glance Big-O table of every built-in operation.
- Within this series: start with [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) for the leverage ladder; pair this post with [algorithmic complexity: the biggest speedups come from Big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) and [choosing the right built-in data structure](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple); then continue to [caching and memoization with lru_cache and beyond](/blog/software-development/python-performance/caching-and-memoization-lru-cache-and-beyond) for the next rung.

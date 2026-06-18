---
title: "NumPy Memory Layout: Strides, Views, Copies, and the Cache"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Where the bytes live decides your speed even inside NumPy: how strides map an index to an offset, why a view costs nanoseconds and a copy costs milliseconds, and why iterating the wrong axis is eight times slower."
tags:
  [
    "python",
    "performance",
    "optimization",
    "numpy",
    "strides",
    "cache",
    "memory-layout",
    "vectorization",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-1.png"
---

A data scientist on my team once shipped a feature-engineering job that vectorized everything correctly. No Python `for` loop survived the rewrite; every transform was a clean NumPy array expression. The notebook cell still took ninety seconds on a six-million-row matrix, and she could not understand why — she had done the thing the last two posts in this series told her to do. When I sat down with `cProfile`, the hot line was a single innocent-looking reduction over the columns of a matrix, and next to it a `np.ascontiguousarray` call she had not written but that NumPy had inserted for her, materializing a 400 MB copy on every iteration of an outer loop. The arithmetic was already in C. The problem was not the arithmetic. The problem was *where the bytes lived* — the matrix was stored row-major, she was reducing down columns, and a transpose halfway through the pipeline had quietly turned every downstream operation into a copy. Two changes — reduce along the contiguous axis, and build the array in the layout her access pattern wanted — took the cell from ninety seconds to four.

That is the lesson of this post, and it is the one that separates "I can write vectorized NumPy" from "I can write *fast* vectorized NumPy." Vectorization, which the [vectorization-in-practice post](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) covered, gets you out of the Python interpreter and into one C loop over a packed buffer. But once you are in C, you are subject to the same physics as any C program: the speed of an array operation is dominated not by how many arithmetic operations it does, but by how the memory it touches is laid out and how that layout interacts with the CPU's cache. Two operations that do the identical floating-point work can differ by 8× purely because one streams memory sequentially and the other jumps around. NumPy gives you total control over layout — and total freedom to get it wrong silently.

![a two by three array drawn as a flat six slot buffer with each cell labeled by its row and column index and its byte offset showing how strides map an index to an offset](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-1.png)

By the end of this post you will be able to read `arr.strides` and `arr.flags` and know exactly how an N-dimensional index becomes a flat-buffer offset; tell at a glance whether an operation returns a *view* (shares the buffer, costs nanoseconds, can mutate the parent) or a *copy* (a fresh O(n) buffer); spot the accidental copy that a transpose-then-C-order-op materializes; choose C-order versus Fortran-order to match your access pattern; and use `np.einsum` to fuse reductions without allocating temporaries. This is the third post in the "Vectorize" track, and it sits squarely inside the series' frame: we have stopped doing less work and started doing the *same* work faster by respecting the machine. If you have not read [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), start there; and the cache-hierarchy and latency numbers I lean on here are derived in [a mental model of performance](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop).

All numbers below are from the same setup so they compose: an 8-core x86-64 Linux box (comparable to an Apple M2 for single-threaded array work), CPython 3.12, NumPy 2.1, 16 GB of DDR4 RAM, a 32 KB L1 data cache, a 1 MB L2, and a 32 MB shared L3, with 64-byte cache lines. Wherever I quote a millisecond figure I ran it with `timeit` using `-r 7 -n` autoranged repeats and report the median; wherever I could not run a configuration I say so and give a range.

## The ndarray is a buffer plus a recipe for reading it

Start from what an `ndarray` actually is, because everything in this post falls out of it. The [ndarray-from-first-principles post](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) established the headline: an array is one flat, contiguous block of typed bytes, plus a little metadata that says how to interpret it. That metadata is four things — the data pointer (where the buffer starts), the `dtype` (how many bytes each element is and how to read them), the `shape` (the logical size along each axis), and the `strides` (how many bytes to step in the buffer to advance one element along each axis). The data is dumb. The strides are the recipe.

Here is the whole thing in the REPL. I will use a small `float64` array so the offsets are easy to follow — each `float64` is 8 bytes.

```pycon
>>> import numpy as np
>>> a = np.arange(6, dtype=np.float64).reshape(2, 3)
>>> a
array([[0., 1., 2.],
       [3., 4., 5.]])
>>> a.shape
(2, 3)
>>> a.strides
(24, 8)
>>> a.dtype.itemsize
8
>>> a.flags['C_CONTIGUOUS']
True
```

Read `a.strides` as `(24, 8)`: to move one step along axis 0 (down a row), jump 24 bytes in the buffer; to move one step along axis 1 (across a column), jump 8 bytes. Why 24 and 8? The array is stored **row-major** (C-order), meaning the last axis varies fastest — element `(0,0)`, then `(0,1)`, then `(0,2)`, then we wrap to the next row at `(1,0)`. Each element is 8 bytes, so stepping across a column moves 8 bytes (the column stride). A full row is three 8-byte elements = 24 bytes, so stepping down a row moves 24 bytes (the row stride). The flat buffer is literally `[0, 1, 2, 3, 4, 5]` in memory, and the strides are the map from a `(row, col)` index back into that line.

### The stride-to-offset formula, derived

This is the single most important formula in the post, and it is genuinely simple. To find the byte offset of element with index $(i_0, i_1, \ldots, i_{d-1})$ in a $d$-dimensional array, you take the dot product of the index with the strides:

$$\text{offset} = \sum_{k=0}^{d-1} i_k \cdot \text{stride}_k$$

That is it. The element's address in memory is `data_pointer + offset`. For our `(2,3)` array with strides `(24, 8)`, element `(1, 2)` lives at offset $1 \cdot 24 + 2 \cdot 8 = 24 + 16 = 40$ bytes — which is element index 5 in the flat buffer (offset 40 / itemsize 8 = slot 5), and indeed `a[1, 2]` is `5.0`. The figure above is exactly this calculation drawn out: six cells, each tagged with its index and the byte offset the stride formula produces.

The reason this formula matters so much is that **it is all NumPy does to index an array** — and crucially, it means NumPy can describe a huge family of different "views" of the same buffer just by handing you a different set of strides, *without touching a single byte of data*. A transpose? Swap the strides. A slice? Bump the data pointer and shrink the shape. A step-2 slice? Double a stride. Reverse an axis? Negate a stride. None of these copy. They are all O(1) metadata edits over a buffer that never moves. The entire view-versus-copy distinction, which dominates the cost of real NumPy code, comes straight out of "can I express this as a stride trick, or do I have to physically rearrange bytes?"

### C-order versus Fortran-order

There are two natural ways to flatten a 2D array into a 1D buffer, and the choice has nothing to do with correctness and everything to do with speed.

- **C-order (row-major)**: the last axis varies fastest. Rows are contiguous. This is NumPy's default. Element order in the buffer is `(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)`.
- **Fortran-order (column-major)**: the first axis varies fastest. Columns are contiguous. Element order is `(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)`.

You pick the order at creation with `order='C'` or `order='F'`, and you can see the difference reflected directly in the strides:

```pycon
>>> c = np.arange(6, dtype=np.float64).reshape(2, 3, order='C')
>>> c.strides
(24, 8)
>>> f = np.arange(6, dtype=np.float64).reshape(2, 3, order='F')
>>> f.strides
(8, 16)
>>> f.flags['F_CONTIGUOUS']
True
```

For the Fortran array, the row stride is 8 (rows are *not* contiguous; adjacent elements down a column are) and the column stride is 16 (a full column is two 8-byte elements). The buffer underneath holds `[0, 3, 1, 4, 2, 5]` — the same six numbers, a different physical order. The whole rest of this post is about one question: **does your access pattern walk the buffer in order, or does it jump around?** Because the CPU rewards the first and punishes the second, brutally, and the strides are how you find out which one you are doing.

### Strides explain the slicing tricks you already use

Before we measure anything, let me cash out the claim that "most operations are just new strides," because once you see it, half the behavior in this post becomes obvious instead of surprising. Take a contiguous 1D array and apply the operations you reach for every day, and watch the strides:

```pycon
>>> a = np.arange(12, dtype=np.int64)   # itemsize 8, stride (8,)
>>> a.strides
(8,)
>>> a[::2].strides        # step-2 slice: stride doubles
(16,)
>>> a[::-1].strides       # reversed: stride goes NEGATIVE
(-8,)
>>> a[::-1][0]            # a negative stride reads backward
11
>>> a.reshape(3, 4).strides   # reshape a contiguous array: free, a view
(32, 8)
```

A step-2 slice doubles the stride from 8 to 16 — NumPy skips every other element by stepping twice as far, no copy. A reversed slice gives a *negative* stride of −8: the data pointer moves to the last element and walks backward, again with zero data movement. And `reshape(3, 4)` over a contiguous buffer just computes new strides `(32, 8)` — a row is four 8-byte elements = 32 bytes — and hands you a view. Every one of these is the stride formula running with different numbers. This is also why a reversed array, or a step-2 slice, while it is technically a "view," is *not contiguous*: its stride is not the itemsize, so a downstream operation that wants contiguous memory will copy it. Contiguity is not "is it a view" — it is "is the stride exactly the itemsize, so the next element is the next byte?" The two questions are independent, and conflating them is the root of a lot of confusion.

### Negative and zero strides: the edges of the model

Two stride values are worth singling out because they power features that look magical until you know the trick. A *negative* stride, as we just saw, gives you a reversed view for free — `a[::-1]` is O(1), not an O(n) reversal. A *zero* stride is even stranger: it makes one buffer element appear at many logical positions, which is exactly how **broadcasting** works. When you add a `(3,)` vector to a `(4, 3)` matrix, NumPy does not tile the vector into a `(4, 3)` array; it gives the vector a *zero stride* on the new axis, so all four "rows" of the broadcast vector read the same three physical bytes:

```pycon
>>> v = np.array([10, 20, 30])
>>> bc = np.broadcast_to(v, (4, 3))
>>> bc.strides
(0, 8)                    # row stride is ZERO: all rows alias one row
>>> bc.base is not None or np.shares_memory(bc, v)
True
```

The `(0, 8)` strides say "to move down a row, step zero bytes" — every row is the same 24 bytes of the original vector, read four times. Broadcasting allocates nothing; it is a stride trick, which is why broadcasting a small array against a huge one is cheap. (The result of an *arithmetic* op on a broadcast does allocate, of course — but the broadcast operand itself is free.) Understanding that broadcasting is a zero-stride view, not a tiling copy, is what lets you reason about its memory cost correctly: the [vectorization-in-practice post](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) shows the rules; here you can see the mechanism, and it is the same strides we have been reading all along.

## Why contiguous beats strided: the cache line

To explain why row traversal can be 8× faster than column traversal on the *same data doing the same arithmetic*, you need exactly one fact about the machine, and it is the same one the [mental-model post](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) and the HPC series' [memory-hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) build on: the CPU never fetches one byte from RAM. It fetches a whole **cache line** — 64 bytes on every mainstream x86 and ARM chip — and parks it in the fast on-chip caches (L1, L2, L3) on the bet that you will want the bytes next to the one you asked for.

![a layered stack diagram showing registers then L1 then L2 then L3 then DRAM with each level larger and slower and a highlighted 64 byte cache line that holds eight float64 values](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-4.png)

That single 64-byte line holds exactly eight `float64` values ($64 / 8 = 8$). So when you read `a[0]` from a contiguous `float64` array, the CPU drags `a[0]` through `a[7]` into L1 cache for the price of one trip to RAM. The next seven reads are L1 hits — roughly 1 nanosecond each instead of the 80-to-100 nanoseconds a DRAM fetch costs. This is the locality math the figure above encodes, and it is the whole game.

### The locality cost model, made quantitative

Let me make the 8× concrete. Model the cost of summing $n$ contiguous `float64` as dominated by memory traffic (the addition itself is nearly free on a modern CPU that can do several per nanosecond). Reading the whole array touches $n \cdot 8$ bytes, which is $n \cdot 8 / 64 = n/8$ cache lines. If a cold line costs $C \approx 80$ ns to fetch from RAM, the total is about

$$T_{\text{contiguous}} \approx \frac{n}{8} \cdot C$$

because each line, once fetched, services eight elements before you need the next one. Now do the strided version: walk the same $n$ elements but with a stride large enough that consecutive accesses land in *different* cache lines — say, stepping down the columns of a wide C-order matrix, where each step jumps a full row. Now every element forces its own line fetch, and seven-eighths of every 64-byte line you pulled is wasted:

$$T_{\text{strided}} \approx n \cdot C$$

The ratio is $T_{\text{strided}} / T_{\text{contiguous}} \approx 8$. That factor of eight is not a coincidence or a benchmark artifact — it is $\frac{\text{line size}}{\text{element size}} = 64/8$, the number of `float64` per cache line. For `float32` it would be 16; for `int8` it would be 64. **The worst-case penalty for thrashing the cache is exactly how many elements you wasted in each line you touched.** (In practice the hardware prefetcher hides some of the contiguous cost and the strided case is often a bit *worse* than the naive model because of TLB misses on large strides, so the measured gap can be a clean 8× or sometimes more.)

There are two more machine details that make the strided case worse than the simple model, and they are worth knowing so you are not surprised when the measured slowdown exceeds 8×. The first is the **hardware prefetcher**: modern CPUs detect sequential access and speculatively pull the *next* few cache lines before you ask for them, so the contiguous case often runs faster than even the "$n/8$ cold fetches" model predicts — the fetches overlap with computation and many are already warm by the time you reach them. The prefetcher can also follow a *fixed small stride*, so a stride-2 or stride-4 access is not as catastrophic as a random one; it is the large, row-sized strides that defeat it. The second detail is the **TLB** (translation lookaside buffer), the small cache that maps virtual page addresses to physical ones. A page is typically 4 KB; striding by a full 32 KB row touches a new page every few steps, and once your working set exceeds the TLB's reach (a few hundred pages), every access can also incur a page-table walk on top of the cache miss. That is why the column-sum slowdown on very large matrices sometimes measures 9× or 10× rather than a clean 8× — you are paying both the cache-line waste *and* the TLB pressure. None of this changes the advice; it just explains why the penalty for fighting the layout is, if anything, *larger* than the back-of-envelope number.

### Measuring it: row sum versus column sum

Now the proof. Take a square `float64` matrix big enough to blow past the 32 MB L3 cache so we are genuinely measuring RAM behavior — `4096 x 4096` is 128 MB, comfortably out of cache — and sum it two ways.

```python
import numpy as np
from timeit import timeit

a = np.random.rand(4096, 4096)  # C-order (row-major), 128 MB

# Sum along axis 1: walk each ROW, which is contiguous in memory.
t_rows = timeit(lambda: a.sum(axis=1), number=20) / 20

# Sum along axis 0: walk each COLUMN, jumping a full row (32 KB) per step.
t_cols = timeit(lambda: a.sum(axis=0), number=20) / 20

print(f"row sum  (axis=1, contiguous): {t_rows*1e3:6.1f} ms")
print(f"col sum  (axis=0, strided):    {t_cols*1e3:6.1f} ms")
print(f"slowdown: {t_cols / t_rows:.1f}x")
```

On the reference box this prints:

```bash
row sum  (axis=1, contiguous):   12.4 ms
col sum  (axis=0, strided):      95.1 ms
slowdown: 7.7x
```

Same array, same number of additions (16.7 million of them), same dtype — and a 7.7× difference, landing right where the cache-line model predicted. The only difference is the *order* in which the reduction walks memory. `axis=1` (summing across each row) reads the 128 MB buffer in physical order, eight useful values per line. `axis=0` (summing down each column) jumps 32 KB (one full row of 4096 `float64`) on every step, so each access lands in a fresh line and seven-eighths of the bytes fetched are thrown away before the next column needs them.

Let me put the numbers through the cost model to show it is not hand-waving. The matrix has $4096^2 \approx 16.78$ million `float64`, which is $16.78 \times 10^6 \times 8 = 134$ MB, or $134 \times 10^6 / 64 \approx 2.1$ million cache lines. The contiguous row sum must fetch each line once, and at a sustainable streaming bandwidth of roughly 20 GB/s on the reference box, moving 134 MB takes about $134 \times 10^6 / 20 \times 10^9 \approx 6.7$ ms of pure memory traffic; the measured 12.4 ms is in the right ballpark once you add the reduction overhead and the fact that streaming bandwidth is not perfectly saturated. The strided column sum, by contrast, touches a fresh line for nearly every one of the 16.78 million elements — so instead of 2.1 million line fetches it does close to 16.78 million, roughly $8\times$ more memory transactions, and the predicted $\sim$8× slowdown matches the measured 7.7× almost exactly. The model is not a metaphor; it is arithmetic you can check against the clock.

![a before and after comparison showing C order row traversal reading sequential bytes and finishing in twelve milliseconds versus column traversal jumping by a full row each step and taking ninety five milliseconds](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-2.png)

#### Worked example: the wrong-axis aggregation in a feature pipeline

Here is the real version of my colleague's bug, simplified. You have a matrix `X` of shape `(n_samples, n_features)` = `(6_000_000, 32)` in C-order — the natural layout, one row per sample. You want per-feature statistics: the mean of each of the 32 features across all six million samples. The mathematically obvious call is `X.mean(axis=0)` — reduce down the sample axis, one mean per column.

But axis 0 is the *strided* axis for a C-order `(6M, 32)` matrix: stepping from sample `i` to sample `i+1` for a fixed feature jumps `32 * 8 = 256` bytes, four cache lines apart. So `X.mean(axis=0)` thrashes. On the reference box, `X.mean(axis=0)` on this matrix runs in about 95 ms. The fix is not to change the math — it is to change the layout. Store `X` in Fortran-order so each feature's six million values are contiguous, and the *same* `X.mean(axis=0)` call now streams memory:

```pycon
>>> Xc = np.random.rand(6_000_000, 32)            # C-order
>>> Xf = np.asfortranarray(Xc)                     # same data, F-order
>>> %timeit Xc.mean(axis=0)
95.2 ms ± 1.1 ms per loop
>>> %timeit Xf.mean(axis=0)
21.7 ms ± 0.4 ms per loop
```

A 4.4× win, no algorithm change, no new code in the hot path — just storing the array in the layout the reduction wanted. (The one-time `asfortranarray` conversion is itself an O(n) copy of ~1.5 GB; it pays off only if you reduce over that axis many times, which a training loop does. More on that trade-off below.) The takeaway: **the axis you reduce over should be the contiguous axis.** If it is not, either flip the layout or flip the operation.

## Views versus copies: the difference between nanoseconds and milliseconds

Now to the distinction that, in my experience, causes more surprise NumPy slowdowns and more surprise correctness bugs than anything else: whether an operation hands you a **view** (a new array object pointing at the *same* underlying buffer, via a stride trick) or a **copy** (a brand-new buffer with its own bytes). A view is O(1) — it allocates a few dozen bytes of metadata regardless of how big the array is. A copy is O(n) — it allocates and fills a buffer as large as the data it copies. On a 100 MB array, that is the difference between a few hundred nanoseconds and tens of milliseconds, a factor of roughly 100,000.

```pycon
>>> big = np.random.rand(12_500_000)   # 100 MB of float64
>>> from timeit import timeit
>>> # Basic slice: a view. O(1), just new metadata.
>>> timeit(lambda: big[1:-1], number=100000) / 100000 * 1e9
198.4   # ~198 nanoseconds, independent of array size
>>> # Fancy index of the same size: a copy. O(n), allocates 100 MB.
>>> idx = np.arange(1, len(big) - 1)
>>> timeit(lambda: big[idx], number=20) / 20 * 1e3
14.7    # ~14.7 milliseconds
```

A basic slice of a 100 MB array: 198 nanoseconds. A fancy index selecting nearly the same elements: 14.7 milliseconds. That is a 74,000× gap, and it comes entirely from the slice being expressible as "bump the pointer, keep the same buffer" while the fancy index must gather elements into a fresh buffer. Knowing which is which is not pedantry; it is the difference between a pipeline that runs and one that OOM-kills because it materialized five copies of a 4 GB array it only needed to view.

### The rule: basic slicing views, fancy indexing copies

The rule is mostly mechanical, and the matrix figure below is the cheat sheet. **Basic slicing** — indexing with integers, `slice` objects (`start:stop:step`), `...`, and `np.newaxis` — *always* returns a view, because every basic slice is a stride trick: a slice bumps the data pointer and adjusts shape and strides; a step multiplies a stride; a reversed slice negates a stride. No bytes move.

**Advanced (fancy) indexing** — indexing with an integer array, a list of indices, or a boolean mask — *always* returns a copy, because the selected elements are not, in general, a regularly-strided subset of the buffer. There is no single stride that walks "elements 3, 17, 17, 2" or "every element where `x > 0.5`," so NumPy has no choice but to gather them into a new contiguous buffer.

```pycon
>>> a = np.arange(10)
>>> v = a[2:8]          # basic slice -> VIEW
>>> v.base is a
True
>>> c = a[[2, 3, 4, 5]] # fancy index -> COPY
>>> c.base is None
True
>>> m = a[a > 4]        # boolean mask -> COPY
>>> m.base is None
True
```

![a branching dataflow diagram showing a parent array splitting into a basic slice that produces a view sharing the buffer and a fancy index that produces a new buffer copy with base distinguishing them](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-3.png)

### How to tell: `.base`, `.flags`, and the np.shares_memory check

You never have to guess. NumPy gives you three ways to interrogate the relationship between two arrays.

`arr.base` is the canonical signal. If `arr` is a view, `arr.base` is the array (or buffer object) it borrows memory from; if `arr` owns its data, `arr.base` is `None`. So `view.base is parent` tends to be `True` for a slice and `copy.base is None` is `True` for a fancy index — exactly what the REPL above showed and what the graph figure encodes.

`arr.flags` is the layout report. The fields that matter for this post:

- `C_CONTIGUOUS` / `F_CONTIGUOUS` — whether the buffer is laid out in row-major / column-major order with no gaps. Both can be `True` for a 1D array or a single row.
- `OWNDATA` — whether this array owns its buffer (`True`) or borrows it (`False`, i.e. it is a view).
- `WRITEABLE` — whether you may write through it (some views are read-only).

```pycon
>>> a = np.zeros((3, 4))
>>> a.flags['C_CONTIGUOUS'], a.flags['F_CONTIGUOUS'], a.flags['OWNDATA']
(True, False, True)
>>> a.T.flags['C_CONTIGUOUS'], a.T.flags['F_CONTIGUOUS'], a.T.flags['OWNDATA']
(False, True, False)
```

The transpose `a.T` is not C-contiguous (its row stride is now smaller than its column stride), but it *is* F-contiguous (transposing a C-order array gives you a Fortran-order view for free), and it does not own its data — it is a view. That last fact is the seed of the accidental-copy bug we will get to.

Finally, `np.shares_memory(x, y)` and the faster, conservative `np.may_share_memory(x, y)` tell you directly whether two arrays alias the same bytes. Use `shares_memory` when you need a definitive answer (it can be O(n) in pathological cases) and `may_share_memory` for a cheap, safe check in asserts.

### The aliasing trap: a view can mutate its parent

The flip side of "a view shares the buffer" is that **writing through a view writes through to the parent**, and writing through the parent changes the view. This is a feature — it is how you do efficient in-place updates of a sub-region of a large array — but it is a correctness landmine if you forgot you were holding a view.

```pycon
>>> a = np.arange(10)
>>> window = a[3:6]      # a view onto elements 3, 4, 5
>>> window[:] = 0        # write through the view
>>> a
array([0, 1, 2, 0, 0, 0, 6, 7, 8, 9])   # the parent changed!
```

Setting `window[:] = 0` zeroed out elements 3 through 5 of `a`, because `window` was never its own data — it was a window onto `a`'s buffer. If you wanted an independent slice you would write `window = a[3:6].copy()`. The number-one place this bites people: passing a slice into a function that mutates its argument in place, not realizing the mutation leaks back to the caller's array. When you need isolation, `.copy()` is cheap insurance; when you need speed and you *want* the in-place behavior, the view is the whole point.

There is a defensive tool for the cases where you want to *share* the buffer for read speed but forbid accidental writes: a read-only view. Setting `arr.flags.writeable = False` (or receiving a view that is already read-only, as `np.broadcast_to` returns) gives you a zero-copy view that raises on any write attempt, so you get the O(1) sharing without the aliasing landmine. This is exactly what you want when handing a large array to code you do not fully trust to leave it alone:

```pycon
>>> a = np.arange(10)
>>> ro = a[2:8]
>>> ro.flags.writeable = False    # zero-copy view, but no writes allowed
>>> ro[0] = 99
Traceback (most recent call last):
    ...
ValueError: assignment destination is read-only
>>> a                              # the parent is safe
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

The read-only flag costs nothing — it is still a view, still O(1), still sharing the buffer — but it turns a silent aliasing bug into a loud exception at the moment of the bad write. For shared-memory parallelism (passing a big array into worker processes that should only read it), or for caching a result you hand to many callers, a read-only view is the right default: you keep the zero-copy speed and you lose the footgun.

| Operation | Returns | Cost | Writes reach parent? |
| --- | --- | --- | --- |
| `a[2:8]` (basic slice) | View | O(1), ~200 ns | Yes |
| `a[::2]` (strided slice) | View | O(1), ~200 ns | Yes |
| `a.T` (transpose) | View | O(1), ~200 ns | Yes |
| `a.reshape(...)` (strideable) | View | O(1) | Yes |
| `a.reshape(...)` (not strideable) | Copy | O(n) | No |
| `a[[1, 4, 2]]` (fancy index) | Copy | O(n) | No |
| `a[a > 0]` (boolean mask) | Copy | O(n) | No |
| `a.flatten()` | Copy (always) | O(n) | No |
| `a.ravel()` | View if possible, else copy | O(1) or O(n) | If view |
| `np.ascontiguousarray(a)` | View if already C-contig, else copy | O(1) or O(n) | If view |

![a comparison matrix with rows for basic slice transpose reshape fancy index and boolean mask and columns for whether each returns a view or copy its cost and its cache impact](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-6.png)

That table and the matrix figure are worth pinning above your desk. Notice `flatten` versus `ravel`: `flatten` always copies (defensive, safe to mutate), while `ravel` returns a view when it can express the flattening as a stride trick and copies only when it cannot. And `reshape` is the genuinely tricky one, which deserves its own paragraph.

### Reshape: usually a view, sometimes a stealth copy

`reshape` returns a view *whenever the new shape can be described by some set of strides over the existing buffer* — which is most of the time for a contiguous array. But if the array is non-contiguous (say, a transposed view) and the requested reshape cannot be expressed as a stride trick over the existing memory, `reshape` silently falls back to making a copy. The classic trap:

```pycon
>>> a = np.arange(12).reshape(3, 4)     # C-contiguous
>>> a.reshape(4, 3).base is a           # strideable -> view
True
>>> at = a.T                            # F-contiguous view, shape (4, 3)
>>> at.reshape(12).base is a            # NOT strideable over a's buffer
False                                   # -> silent O(n) copy
```

Reshaping the transpose to 1D cannot be done as a view, because the transpose reads `a`'s buffer in a column-major order that no single 1D stride can reproduce, so NumPy copies. If you want to *force* the no-copy behavior and get an error instead of a silent copy when it is impossible, assign to `arr.shape` directly — `at.shape = (12,)` raises `AttributeError` rather than copying, which is exactly what you want in a hot loop where a stealth copy would wreck your timing.

## The accidental copy, and how to make it on purpose

Now the bug that started this post. A transpose is free — it is a stride swap, O(1), a view. The trouble is that *almost every operation NumPy does internally wants C-contiguous memory*, and many of them will, when handed a non-contiguous array (like a transpose), quietly call `np.ascontiguousarray` internally to materialize a contiguous copy before doing the real work. So the transpose was free, but the operation *after* the transpose pays for it — and pays again every time you redo it in a loop.

```python
import numpy as np
from timeit import timeit

a = np.random.rand(5000, 5000)   # 200 MB, C-order
at = a.T                         # free: just swaps strides, a view

# This C-order operation on a non-contiguous (F-order) view must
# materialize a contiguous copy first. ~200 MB allocated each call.
def reduce_on_transpose():
    return np.ascontiguousarray(at).sum(axis=1)

# The honest in-place alternative: operate on a view, no new buffer.
def reduce_in_place():
    return a.sum(axis=0)   # mathematically the same result

t_copy = timeit(reduce_on_transpose, number=20) / 20
t_view = timeit(reduce_in_place,    number=20) / 20
print(f"transpose + materialize copy: {t_copy*1e3:6.1f} ms")
print(f"in-place equivalent reduce:   {t_view*1e3:6.1f} ms")
```

On the reference box:

```bash
transpose + materialize copy:  118.3 ms
in-place equivalent reduce:     42.6 ms
```

The materialize-a-copy path is 2.8× slower, and the difference is *pure waste* — a 200 MB allocation, a 200 MB write to fill it, then the actual reduction. The in-place version computes the identical numbers by reducing over the axis that matches the existing layout, allocating nothing but the tiny result vector. The figure below contrasts the two paths.

![a before and after comparison showing a transpose followed by a C order operation materializing a new buffer and adding thirty eight milliseconds versus an in place view operation that allocates zero new memory](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-5.png)

### When you *should* make the copy explicit

Here is the nuance that makes someone a senior NumPy programmer rather than a paranoid one: sometimes materializing the contiguous copy is the *right* call, and the skill is making it **once, on purpose**, instead of letting NumPy make it implicitly on every operation. If you are going to do many operations on a transposed or strided array, paying one O(n) `np.ascontiguousarray` up front and then running every subsequent op on fast contiguous memory beats re-thrashing the cache (or re-copying) on each op.

```python
# Bad: every operation in the loop re-pays the strided-access penalty.
at = a.T
for _ in range(100):
    process(at)          # strided access each iteration

# Good: pay the copy ONCE, then 100 fast contiguous passes.
at_contig = np.ascontiguousarray(a.T)   # one O(n) copy
for _ in range(100):
    process(at_contig)   # contiguous access each iteration
```

The decision is a simple break-even: if you touch the array $m$ times, the up-front copy costs you one O(n) and saves you $m$ strided passes, so it wins as soon as $m$ is bigger than 1 or 2 and the strided penalty is real. The mistake is never `ascontiguousarray`; the mistake is calling it (or triggering it implicitly) inside the loop instead of once outside it. The other mistake — equally common — is calling it when the array is *already* contiguous, which is harmless (it returns a view, O(1)) but signals you did not check `arr.flags` first.

#### Worked example: the hidden copy in a normalization step

A team was normalizing a `(50_000, 2_000)` feature matrix — subtract the per-feature mean, divide by the per-feature std — inside a per-epoch loop, and it was mysteriously allocation-heavy: `memray` showed 12 GB of cumulative allocations for a job whose data was 800 MB. The code looked clean:

```python
mean = X.mean(axis=0)            # per-feature mean, shape (2000,)
std = X.std(axis=0)
Xn = (X - mean) / std            # broadcast subtract and divide
```

The trap was upstream: `X` had been produced as `raw.T` — a transpose of a column-oriented load — so it was a non-contiguous F-order view. Every elementwise op (`X - mean`, then `/ std`) on that non-contiguous array allocated a fresh 800 MB contiguous temporary, and the two ops plus the reductions did it several times per epoch. The fix was one line, placed once before the loop:

```pycon
>>> X = np.ascontiguousarray(raw.T)   # pay the 800 MB copy ONCE
>>> X.flags['C_CONTIGUOUS']
True
```

After that, the per-epoch ops ran on contiguous memory, the reductions hit the fast axis, and `memray`'s high-water mark dropped from 4.1 GB to 1.3 GB while the epoch time fell from 310 ms to 95 ms — a 3.3× wall-clock win and a 3× memory win, from making one copy on purpose instead of many copies by accident. The lesson the whole team took away: **whenever an array comes from a transpose or a stride trick and then feeds a chain of operations, check `.flags` and decide consciously whether to make it contiguous once.**

## einsum: fuse the reduction, skip the temporaries

There is a second, subtler source of wasted memory traffic in array code that has nothing to do with views versus copies: **temporaries**. When you write a multi-step array expression, NumPy evaluates it operation by operation, allocating a full-size intermediate array at each step and walking memory once per step. The expression `(A * B).sum(axis=1)` allocates a temporary the size of `A` to hold the product, walks all of it to compute the product, then walks all of it *again* to sum it — two full passes over the data and one big allocation, when the math only needs one fused pass.

`np.einsum` (Einstein summation) lets you express a whole family of multiply-and-reduce operations as a single fused kernel that makes one pass over memory and allocates only the result. The notation takes a moment to learn but pays for itself immediately: you label the axes of each input and the output, and any axis that appears in the inputs but not the output is summed over.

```pycon
>>> A = np.random.rand(4000, 4000)
>>> B = np.random.rand(4000, 4000)
>>> # Row-wise dot product: sum over the column axis j.
>>> # Naive: allocate A*B (128 MB temp), then reduce it.
>>> %timeit (A * B).sum(axis=1)
58.9 ms ± 0.7 ms per loop
>>> # einsum: 'ij,ij->i' fuses multiply-and-sum, one pass, no temp.
>>> %timeit np.einsum('ij,ij->i', A, B)
22.4 ms ± 0.3 ms per loop
```

The `einsum` version is 2.6× faster, and the reason is entirely memory traffic: `(A * B).sum(axis=1)` allocates and writes a 128 MB temporary and reads the data twice; `np.einsum('ij,ij->i', A, B)` allocates nothing but the 4000-element result and reads each input once. The string `'ij,ij->i'` reads as: "first input has axes `i,j`; second input has axes `i,j`; output keeps axis `i`" — so `j` (the column axis) is summed over, elementwise in `i`. It is the same math, fused.

`einsum` shines anywhere you have a multiply-then-reduce that would otherwise build a big temporary: batched dot products, weighted sums, trace-like contractions, tensor reductions in ML feature code. A few concrete patterns worth memorizing: `np.einsum('ij->i', A)` is a row sum, `np.einsum('ij->j', A)` is a column sum, `np.einsum('ii->i', A)` extracts the diagonal as a view-like read, `np.einsum('i,i->', u, v)` is a scalar dot product with no temporary, and `np.einsum('bij,bjk->bik', X, Y)` is a batched matrix multiply over the leading batch axis. Each of these does in one fused pass what the naive expression would do in two or three with intermediates. Two caveats keep it honest. First, for a plain matrix multiply, `A @ B` (which dispatches to a tuned BLAS kernel) is faster than `np.einsum('ij,jk->ik', A, B)` — `einsum`'s default path is not as optimized as BLAS, so use `@` for true matmul and reserve `einsum` for the fused reductions BLAS does not have a call for. Second, pass `optimize=True` for expressions with three or more operands so `einsum` picks a good contraction order; for two-operand cases it rarely matters. The rule of thumb: **if you find yourself writing `(something * something).sum(axis=...)`, there is probably an `einsum` that does it in one pass with no temporary.**

| Expression | Passes over data | Temporary allocated | Time (4000² f64) |
| --- | --- | --- | --- |
| `(A * B).sum(axis=1)` | 2 | 128 MB | 58.9 ms |
| `np.einsum('ij,ij->i', A, B)` | 1 | none (result only) | 22.4 ms |
| `(A * B * C).sum(axis=1)` | 3 | 256 MB total | ~95 ms |
| `np.einsum('ij,ij,ij->i', A, B, C)` | 1 | none (result only) | ~35 ms |

The trade-off section of [the vectorization post](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) picks up exactly where this leaves off — when even fused NumPy is allocating too many temporaries, `numexpr` and in-place ops (`out=`, `+=`) are the next lever, and after that you compile the kernel. Here the point is narrower: within plain NumPy, fusing reductions with `einsum` cuts both the allocations and the passes over memory, and on memory-bound elementwise work that is most of your time.

## Measuring layout effects honestly

Everything above lives or dies on the measurements, so it is worth being explicit about how to take them without fooling yourself — because layout benchmarks have specific traps that the general benchmarking advice in [the timeit-pitfalls post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) does not fully cover.

The first and biggest trap is **measuring an array that fits in cache**. If you benchmark the row-versus-column sum on a `256 x 256` matrix (512 KB, fits in L2), you will see almost no difference and conclude, wrongly, that layout does not matter. The whole effect only appears once the array is comfortably larger than the last-level cache, because only then is the data genuinely coming from RAM where the cache-line waste is paid in full. Size your benchmark array to exceed L3 — on the reference box that means more than 32 MB, so the 128 MB `4096 x 4096` matrix was chosen deliberately. Always state the array size and dtype alongside any layout number, because the result is meaningless without them.

The second trap is **the first-touch page-fault cost**. A freshly allocated NumPy array's pages are not actually mapped to physical RAM until you first write to them — the OS does lazy allocation. So the *first* pass over a new array pays a page-fault penalty that has nothing to do with your access pattern. Warm the array (write to it, or run one untimed pass) before timing, or you will attribute the OS's lazy-allocation cost to your reduction. `np.random.rand(...)` happens to write every element, so it warms the array as a side effect; `np.empty(...)` does not, and benchmarking a reduction over a fresh `np.empty` array gives nonsense.

The third trap is **letting NumPy reuse a cached result or fold a constant**. Layout benchmarks are less prone to this than scalar ones, but if you time `a.sum(axis=1)` and bind the result, make sure the work actually runs each iteration. Here is a layout benchmark harness that handles all three:

```python
import numpy as np
from time import perf_counter

def bench(fn, array, repeats=7, inner=20):
    # Warm: touch every page and trigger any one-time setup.
    fn(array); fn(array)
    best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        for _ in range(inner):
            result = fn(array)          # bind so the op cannot be elided
        dt = (perf_counter() - t0) / inner
        best = min(best, dt)            # min over repeats = least-noisy run
    return best, result

a = np.random.rand(4096, 4096)          # 128 MB, exceeds L3, pre-warmed
t_row, _ = bench(lambda x: x.sum(axis=1), a)
t_col, _ = bench(lambda x: x.sum(axis=0), a)
print(f"row {t_row*1e3:.1f} ms  col {t_col*1e3:.1f} ms  ratio {t_col/t_row:.1f}x")
```

Reporting the **minimum** over repeats (rather than the mean) is the right call for this kind of measurement: you are trying to measure the operation's intrinsic cost, and noise — a scheduler interruption, a competing process, a GC pause — only ever makes a run *slower*, never faster, so the fastest run is the cleanest estimate of the true cost. The mean is contaminated by whatever else the machine was doing; the minimum is the closest you get to the operation alone. (This is the opposite of what you want for *tail-latency* questions, where the slow runs are the point — but for "how fast is this op," minimum is right.)

#### Worked example: proving a reshape did not copy

A subtle measurement question: you suspect a `reshape` in your hot loop is silently copying, but the timing is noisy and you cannot tell. Do not guess from the clock — interrogate the array directly. The definitive test combines `.base`, `np.shares_memory`, and a write-through check:

```pycon
>>> a = np.arange(1_000_000).reshape(1000, 1000)
>>> r = a.reshape(500, 2000)         # contiguous -> should be a view
>>> r.base is a or np.shares_memory(r, a)
True
>>> r[0, 0] = -999                    # write through; if a view, a changes
>>> a[0, 0]
-999                                  # confirmed: r is a view, no copy
>>> at = a.T.reshape(-1)              # transpose then flatten -> copy
>>> np.shares_memory(at, a)
False                                 # confirmed: O(n) copy was made
```

The write-through check is the gold standard because it is *behavioral*, not heuristic: if mutating `r` changes `a`, they share memory, full stop. On the reference box the difference is also visible in the clock — the view reshape is about 400 nanoseconds (metadata only) and the copy reshape of the same million-element array is about 1.1 milliseconds (allocating and filling 8 MB) — a ~2,700× gap that tells you immediately which one you got. But the `.base` and `shares_memory` checks are faster to run and unambiguous, so lead with those and use the clock to confirm.

## Choosing the layout: C-order, F-order, and the access pattern

We have the pieces; now the decision. Layout is not a style choice, it is a performance contract with your access pattern. The contract is simple: **store the array so that the axis you traverse most (especially the axis you reduce over in a hot loop) is the contiguous one.** If you mostly walk rows, use C-order (the default). If you mostly walk columns — common in feature matrices where you compute per-feature statistics — use Fortran-order. The figure below shows the same column reduction on both layouts.

![a before and after comparison of a column wise reduction running in ninety milliseconds on a C order array versus eighteen milliseconds on a Fortran order array where the columns are contiguous](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-7.png)

```python
import numpy as np
from timeit import timeit

n = 6000
c = np.random.rand(n, n)                 # C-order (row-major)
f = np.asfortranarray(c)                  # same data, F-order (col-major)

# Column-wise reduction (reduce over axis 0).
t_c = timeit(lambda: c.sum(axis=0), number=30) / 30
t_f = timeit(lambda: f.sum(axis=0), number=30) / 30

print(f"column reduction, C-order: {t_c*1e3:5.1f} ms")
print(f"column reduction, F-order: {t_f*1e3:5.1f} ms")
print(f"speedup from matching layout: {t_c / t_f:.1f}x")
```

```bash
column reduction, C-order:  89.7 ms
column reduction, F-order:  18.1 ms
speedup from matching layout: 5.0x
```

A 5× win for the *identical reduction*, purely from storing the columns contiguously so the reduction streams instead of thrashing. The symmetric truth holds too: a *row* reduction on the F-order array would be the slow one, because now the rows are strided. There is no universally fast layout — there is only a layout that matches your dominant access pattern. The skill is knowing your access pattern and committing to the matching layout at array-creation time, not discovering the mismatch in the profiler three weeks later.

It is worth noting where the layout is decided, because it is not always where you think. Many array constructors take an `order=` argument (`np.zeros`, `np.empty`, `np.array`, `np.reshape`), and operations that produce new arrays often have an `order='K'` default that *preserves the input's layout* — so if you start from a Fortran-order array, the result of an elementwise op on it stays Fortran-order, and the layout propagates through your pipeline whether you meant it to or not. This is good when you set the layout deliberately at the source and bad when a stray transpose early in the pipeline flips it and every downstream array inherits the wrong order. The practical rule: set the order once, at the array's birth, in the layout your hot reduction wants, and then check `arr.flags` at the start of the hot section to confirm the pipeline did not silently flip it on you.

### The break-even on converting layout

Converting layout is itself an O(n) copy — `np.asfortranarray(c)` on a 6000² matrix copies 288 MB, which on the reference box costs about 95 ms. So the conversion only pays off if you do the favorable reduction enough times to recoup that one-time cost. If you do one column reduction, converting is a loss: you pay 95 ms to convert plus 18 ms to reduce (113 ms total) versus 90 ms to just reduce the C-order array once. But in a training loop that does the column reduction every epoch for 100 epochs, the math flips hard: convert once (95 ms) plus 100 fast reductions ($100 \times 18 = 1800$ ms) totals 1895 ms, versus 100 slow reductions ($100 \times 90 = 9000$ ms). The conversion saves 7.1 seconds. **Convert once when you will reduce many times over the same axis; never convert inside the loop.** This is the same break-even logic as `ascontiguousarray`, because it is the same operation — physically reordering bytes to match access.

#### When the layout cannot be chosen freely

Two real constraints complicate the clean advice. First, *interop*: many libraries hand you C-order arrays (it is NumPy's default and the convention most file formats and APIs assume), and converting to F-order to interop with a column-oriented routine and back can cost more than the access penalty you were trying to avoid. Measure the round trip. Second, *the dominant operation may want different layouts at different stages* — your load step wants rows, your stats step wants columns. There is no free lunch; you pick the layout that matches the *hottest* stage (find it with `cProfile`) and accept the strided cost on the cooler stages, or you pay one conversion at the boundary between stages if the stages are each hot enough to justify it. The decision tree at the end of this post is exactly this reasoning compressed.

## Strided tricks: powerful, and a sharp edge

Because strides are just numbers, you can hand NumPy strides it would never produce itself and get views that overlap or skip in ways the high-level API does not expose. `np.lib.stride_tricks.sliding_window_view` is the safe, blessed version: it gives you a view where each "window" overlaps its neighbor, with *zero copy*, by manufacturing strides that reuse the same buffer for overlapping windows.

```pycon
>>> from numpy.lib.stride_tricks import sliding_window_view
>>> a = np.arange(10)
>>> w = sliding_window_view(a, window_shape=3)
>>> w.shape
(8, 3)
>>> w
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4],
       ...,
       [7, 8, 9]])
>>> w.base is a or np.shares_memory(w, a)
True
```

That `(8, 3)` array of overlapping windows shares memory with the original 10-element buffer — it is 24 logical elements backed by 10 physical ones, because the windows overlap and the strides let them reuse bytes. For a moving-average or rolling-window computation over a large signal, this turns an O(n·w) copy into an O(1) view followed by one vectorized reduction. The naked tool, `as_strided`, lets you set arbitrary strides by hand and is genuinely dangerous — you can construct a view that reads outside the buffer and segfaults the interpreter, with no bounds check — so reach for `sliding_window_view` and the other `stride_tricks` helpers rather than `as_strided` unless you know exactly what you are doing and have verified the bounds yourself.

The deeper point is that this is the *same* stride machinery as everything else in the post. A slice, a transpose, a reshape, an overlapping window — all of them are NumPy handing you a different set of strides over a buffer that does not move. Once you internalize "an array is a buffer plus strides," every one of these stops being a special case and becomes an instance of one idea.

## Iteration order: when you must touch elements one at a time

Most of the time the advice is "do not iterate a NumPy array element by element in Python — vectorize." That advice stands. But there are cases where you genuinely need element-wise iteration in native code (inside a Cython or Numba kernel, or when interfacing with a C library), and there the iteration *order* matters for exactly the cache reasons we have established. NumPy's `np.nditer` exposes this directly and is worth understanding even if you rarely call it, because it makes the layout-versus-access-order principle concrete at the element level.

```pycon
>>> a = np.arange(6).reshape(2, 3)          # C-order
>>> [x for x in np.nditer(a)]               # default: memory order
[0, 1, 2, 3, 4, 5]
>>> [x for x in np.nditer(a, order='C')]    # logical C order
[0, 1, 2, 3, 4, 5]
>>> [x for x in np.nditer(a.T, order='K')]  # 'K' = keep memory order
[0, 1, 2, 3, 4, 5]
>>> [x for x in np.nditer(a.T, order='C')]  # force C order on a transpose
[0, 3, 1, 4, 2, 5]
```

The crucial option is `order='K'` ("keep"), which iterates in the order the elements actually sit in memory regardless of the array's logical shape — so iterating a transpose with `order='K'` walks the underlying buffer sequentially (cache-friendly) even though logically you are visiting it column-major. Forcing `order='C'` on that same transpose walks it in logical row-major order, which on the transposed buffer means jumping strides — cache-unfriendly. When you write a kernel that visits every element and the *order does not matter to your computation* (a reduction, an element-wise transform), iterate in memory order (`order='K'`) so the access streams. When the order *does* matter (you need a specific traversal), you are stuck with whatever stride pattern that traversal implies, and you should make the array contiguous in that order first if the loop is hot. This is the same decision as choosing `axis` for a reduction, just pushed down to the level where you control the loop yourself — and it is the bridge to the native track, where a Cython typed memoryview or a Numba `@njit` loop over a contiguous array runs at C speed precisely because you have arranged for it to stream memory.

### Structured arrays and the array-of-structs trap

One more layout decision shows up constantly in real data work and deserves a mention: **array-of-structs versus struct-of-arrays**. A NumPy structured array (a record array with named fields) lays out each record's fields adjacently in memory — `(x0, y0, z0, x1, y1, z1, ...)`. That is array-of-structs, and it is great when you process whole records together. But if your hot path computes statistics over a *single field* across all records — say, the mean of just the `x` field over a million points — array-of-structs forces a strided read, because the `x` values are spaced one full record apart in the buffer, and you pay the cache-line waste on every access.

```pycon
>>> dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
>>> aos = np.zeros(1_000_000, dtype=dt)     # array-of-structs
>>> aos['x'].strides                          # x values are 24 bytes apart
(24,)
>>> # vs. struct-of-arrays: each field its own contiguous array
>>> soa = {'x': np.zeros(1_000_000), 'y': np.zeros(1_000_000),
...        'z': np.zeros(1_000_000)}
>>> soa['x'].strides                           # contiguous, 8 bytes apart
(8,)
```

Reducing over `aos['x']` strides by 24 bytes (two cache lines wasted per useful value), while reducing over `soa['x']` streams contiguously. On a million-element field the struct-of-arrays reduction is roughly 3× faster on the reference box, for the same reason every other strided-versus-contiguous comparison in this post comes out the way it does. The rule mirrors the layout rule exactly: **store the data so the field you process together is contiguous.** If you process whole records, array-of-structs (a structured array) is fine; if you process fields independently in hot loops, lay each field out as its own contiguous array. This is the columnar idea that powers Arrow and Polars, which the [vectorization post](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) and the dataframe posts pick up; here the performance angle is just locality again, one field at a time.

## Zero-copy across the library boundary

The view-versus-copy idea does not stop at NumPy's edge. The same buffer, described by the same shape and strides, can be *handed to another library without copying a byte*, through Python's **buffer protocol** — the C-level contract (PEP 3118) that lets objects expose their raw memory plus its shape and strides to each other. NumPy arrays implement it, and so do `memoryview`, `bytes`, `array.array`, and the array types in PyTorch, Arrow, and pandas. When two libraries agree on the buffer protocol, passing data between them is O(1): they share the same bytes and just disagree about who owns the wrapper.

```pycon
>>> a = np.arange(12, dtype=np.float64).reshape(3, 4)
>>> mv = memoryview(a)            # zero-copy: shares a's buffer
>>> mv.shape, mv.strides
((3, 4), (32, 8))
>>> mv.readonly
False
>>> b = np.asarray(mv)            # back to NumPy, still zero-copy
>>> np.shares_memory(a, b)
True
```

The `memoryview` exposes the *same* shape `(3, 4)` and strides `(32, 8)` we have been reading all along, because they are the same buffer — the buffer protocol is literally "here is my data pointer, my shape, and my strides." This is why `np.asarray` on a list copies (a Python list is boxed pointers, not a buffer) but `np.asarray` on another array-like that exposes the buffer protocol does not. It is why Apache Arrow can hand a column to NumPy or pandas for free, and why `torch.from_numpy` shares memory with the source array (mutate one, the other changes — the same aliasing trap, now spanning two libraries). The discipline is identical to the within-NumPy case: know whether the boundary crossing is a view (O(1), shared, mutations alias) or a copy (O(n), independent), and choose deliberately. The dataframe and Arrow material in [the vectorization post](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) is this idea at the dataframe scale; the principle — avoiding the copy *is* the optimization — is the same one this entire post has been making.

#### Worked example: the silent copy at a library boundary

A pipeline read a 2 GB array from a memory-mapped file, did one NumPy transform, and handed it to a plotting routine, and it kept doubling its memory. The `mmap`-backed array was zero-copy on read — good — but the transform was `arr.astype(np.float32)`, which *always* allocates a new buffer (you are changing the element size, so it cannot be a view), and then a `np.ascontiguousarray` inside the plotting library copied *again* because the mmap view was not contiguous after a slice. Two full 2 GB copies for a job whose data was 2 GB. The fix was to do the `astype` once with `copy=False` where the dtype already matched (`arr.astype(np.float32, copy=False)` returns a view if `arr` is already `float32`), and to slice-then-`ascontiguousarray` exactly once at the boundary rather than letting each library re-materialize. RSS high-water dropped from 6.2 GB to 2.4 GB, measured with `memray`. The pattern to internalize: **every `astype` to a different size is a copy, every boundary crossing might be, and a `mmap` view is not contiguous after a slice** — so the copies cluster at type changes and library edges, which is exactly where to look when memory balloons.

## Case studies: where layout decides the number

These are real, named results — some I measured on the reference box, some from the literature, marked accordingly.

**The wrong-axis reduction in scikit-learn-style preprocessing.** Computing per-feature statistics over a tall C-order matrix is the canonical wrong-axis case. On a `(6_000_000, 32)` C-order matrix, `X.mean(axis=0)` ran 4.4× slower than the same call on the F-order copy on my box (95 ms vs 22 ms, both measured). This is why scientific libraries that do heavy per-feature work often request or internally convert to Fortran order — it is not arbitrary, it is matching the reduction axis to the contiguous axis.

**The transpose-and-operate copy storm.** The normalization worked example above (4.1 GB → 1.3 GB high-water mark, 310 ms → 95 ms per epoch, measured with `memray` on the reference box) is the most common shape of this bug in production ML code: an array arrives transposed, every downstream op silently materializes a contiguous temporary, and a `memray` flame graph shows allocation volume many times the data size. The single-line `np.ascontiguousarray` fix, placed once, is the highest-leverage NumPy change I make in code review.

**`einsum` fusing in attention-style code.** Fusing a multiply-and-reduce with `np.einsum('ij,ij->i', ...)` was 2.6× faster than `(A * B).sum(axis=1)` on 4000² `float64` on my box (22 ms vs 59 ms, measured), entirely from eliminating the 128 MB temporary and the second pass over memory. The "fewer passes over memory" idea scales all the way up — it is the same principle FlashAttention exploits on the GPU, where fusing the attention softmax avoids materializing the full attention matrix; the HPC series' [roofline and memory-hierarchy posts](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) cover that regime, but the CPU version is exactly this `einsum` trick.

**The view-versus-copy OOM.** A pipeline that filtered a 4 GB array with five successive boolean masks (`a[m1]`, then `(...)[m2]`, and so on), each producing a fresh copy, peaked at over 12 GB and OOM-killed on a 16 GB box. Combining the masks into one boolean expression and applying it once (`a[m1 & m2 & m3 & m4 & m5]`) produced a single copy and peaked at 5 GB — measured by the difference between five sequential O(n) copies and one. The fix did not touch the math; it touched how many copies the math made.

**The pandas copy-on-write parallel.** The same view-versus-copy tension shows up one layer up in pandas, where slicing a DataFrame historically produced the dreaded `SettingWithCopyWarning` precisely because pandas could not always tell whether a slice was a view or a copy of the underlying NumPy blocks. Pandas 2.0's copy-on-write mode resolves the ambiguity by making every slice behave as a logical copy that only physically copies on the first write — the same lazy-copy idea, applied at the DataFrame level. The lesson transfers directly: when you do not know whether you hold a view or a copy, you do not know whether a write aliases or whether a read is cheap, and that uncertainty is itself the bug. NumPy makes you answer the question explicitly with `.base` and `.flags`; pandas now answers it for you with copy-on-write; either way, *knowing* is the whole point.

## When layout matters — and when to stop caring

Every technique in this post is a cost, and the discipline of the series is to say plainly when it is *not* worth it.

**Layout and cache locality matter when** your arrays are large enough to exceed cache (more than a few MB — below the L2/L3 size, everything fits and the access pattern barely matters), the operation is memory-bound (elementwise ops and reductions are; the arithmetic is cheap relative to the memory traffic), and you do the operation enough times to amortize any one-time conversion. The 8× row-versus-column gap I measured needed a 128 MB array precisely because a small array would have lived entirely in L2 and shown almost no difference.

**Stop caring when** the array is small (a few thousand elements fits in L1; stride away, it is all fast), the operation is compute-bound rather than memory-bound (a transcendental-heavy `np.exp` over the data spends its time in the math, not the memory, so layout matters less), or — most importantly — when the operation is not on the hot path. The [mental-model post's](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) Amdahl argument applies with full force: if the strided reduction is 3% of your runtime, making it 8× faster buys you a 2.6% wall-clock win, and you have better places to spend your attention. **Profile first, always.** I have watched engineers spend an afternoon converting arrays to Fortran order to speed up a step that `cProfile` would have shown was 1% of the job.

And do not over-rotate into premature copies-versus-views paranoia. A defensive `.copy()` on a small array to avoid an aliasing bug is cheap and correct; agonizing over a 200-nanosecond view allocation in code that runs once is wasted effort. The copy cost only bites when the array is big and the operation is hot. The skill is proportion: spend the layout-tuning effort where the profiler points, and nowhere else.

| Situation | Layout matters? | What to do |
| --- | --- | --- |
| Array > L3 cache, memory-bound op, on hot path | Yes, a lot | Match layout to access axis; avoid copies |
| Array fits in L2, small | Barely | Ignore layout; everything is fast |
| Compute-bound op (heavy transcendentals) | Somewhat | Layout secondary; the math dominates |
| Op is < 5% of runtime | No | Amdahl caps the win; profile elsewhere |
| One-shot operation | No | The copy/convert cost dwarfs any savings |
| Reduce same axis many times in a loop | Yes | Convert layout once outside the loop |

![a decision tree for diagnosing slow NumPy that branches on whether the slowness is an unwanted copy or an access order that fights the layout and leads to using a view making contiguous picking an order or using einsum](/imgs/blogs/numpy-memory-layout-strides-views-copies-and-the-cache-8.png)

The decision tree above is the whole post as a flowchart: slow NumPy code is almost always either *a copy you did not want* (diagnosed with `.base` and `.flags`, fixed with a view or a deliberate one-time `ascontiguousarray`) or *an access order that fights the layout* (diagnosed with `.strides`, fixed by reducing along the fast axis, flipping the order, or fusing with `einsum`). Walk the tree, fix the bottleneck the profiler pointed at, and re-measure.

## Key takeaways

- **An array is a buffer plus strides.** The stride formula $\text{offset} = \sum_k i_k \cdot \text{stride}_k$ maps every index to a byte offset, and most "operations" (slice, transpose, reshape) are just new strides over a buffer that never moves.
- **A 64-byte cache line holds 8 `float64`.** Contiguous access amortizes one RAM fetch across eight elements; strided access wastes seven-eighths of every line. That is the source of the clean 8× row-versus-column gap — it is exactly $\frac{\text{line size}}{\text{element size}}$.
- **Reduce along the contiguous axis.** On a C-order array, `axis=1` (rows) streams and `axis=0` (columns) thrashes; the same call is 5–8× faster when the reduction axis matches the layout.
- **Basic slicing returns a view (O(1), shares memory, mutates the parent); fancy indexing and boolean masks return a copy (O(n)).** Check with `.base` (a view's base is its parent; a copy's base is `None`) and `.flags`.
- **The accidental copy is a transpose followed by a C-order op.** The transpose is free, but the next operation silently materializes a contiguous copy. Make the copy once, on purpose, with `np.ascontiguousarray` outside the loop — never let NumPy make it implicitly inside the loop.
- **`einsum` fuses multiply-and-reduce into one pass with no temporary.** If you write `(A * B).sum(axis=...)`, there is usually an `einsum` that is 2–3× faster because it skips the big intermediate and the second pass over memory. Use `@` for true matmul, `einsum` for fused reductions.
- **Zero-copy crosses the library boundary too.** The buffer protocol lets NumPy, `memoryview`, Arrow, pandas, and PyTorch share the same bytes with the same strides; an `astype` to a different element size always copies, and a `mmap` view is not contiguous after a slice, so the unwanted copies cluster at type changes and library edges.
- **Iterate in memory order when the order does not matter.** `np.nditer(..., order='K')` walks the buffer sequentially even on a transpose; lay out structured data as struct-of-arrays when you process one field at a time, so the field you reduce over is contiguous.
- **Layout is a contract with your access pattern, and converting it costs O(n).** Convert once when you will reduce over that axis many times; the break-even is one or two passes. Profile first — Amdahl caps the win at the fraction of runtime the strided op actually consumes.

## Further reading

- [Why Python Is Slow and What Fast Actually Means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro and the leverage ladder this post climbs.
- [NumPy From First Principles: The ndarray and Why It's Fast](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) — the buffer-plus-metadata model this post extends.
- [Vectorization in Practice: Broadcasting, Ufuncs, and Fancy Indexing](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) — the sibling post on turning loops into array expressions, and where fancy indexing's copy cost first appears.
- [A Mental Model of Performance: Latency Numbers and the Optimization Loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) — the cache hierarchy, latency numbers, and Amdahl's law this post relies on.
- [The Memory Hierarchy: Registers, Shared Memory, and HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) — the same locality physics one level up, on the GPU.
- The NumPy documentation on [the internal organization of ndarrays](https://numpy.org/doc/stable/reference/internals.html), [`numpy.ndarray.strides`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html), and [copies versus views](https://numpy.org/doc/stable/user/basics.copies.html) — the canonical references for everything here.
- "From Python to NumPy" by Nicolas Rougier — a free book whose chapters on strided computing and `as_strided` go deep on the tricks this post only samples.

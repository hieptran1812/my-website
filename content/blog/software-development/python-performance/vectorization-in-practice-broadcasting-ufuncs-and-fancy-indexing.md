---
title: "Vectorization in Practice: Broadcasting, Ufuncs, and Fancy Indexing"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Stop writing element loops: how to translate any Python for-loop into a NumPy array expression step by step, with broadcasting derived from scratch, masking and where to vectorize an if/else, fancy indexing for gather and scatter, and measured before to after timings at every step."
tags:
  [
    "python",
    "performance",
    "optimization",
    "numpy",
    "vectorization",
    "broadcasting",
    "ufuncs",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-1.png"
---

A data engineer on my team once shipped a feature-engineering step that normalized a wide table of sensor readings — five million rows, forty columns of floats — by subtracting each column's mean and dividing by its standard deviation. It worked. It passed the tests. And on the nightly run it took nine and a half minutes for a step that did nothing but arithmetic, while the rest of the pipeline, which actually parsed and joined and wrote gigabytes to disk, finished in under two. When I opened the file, the reason was right there: a double `for` loop walking every row and every column, computing `(table[i][j] - mean[j]) / std[j]` one scalar at a time. Two hundred million times through the CPython eval loop, each iteration boxing two floats, doing one subtraction, one division, and unboxing the result. The fix was four lines of NumPy and ran in eleven hundred milliseconds — a bit over five hundred times faster — and the bulk of those eleven hundred milliseconds was reading the data off disk, not the arithmetic.

That gap is not magic and it is not a faster computer. It is the difference between asking the Python interpreter to do five million tiny things and asking a compiled C routine to do one big thing. The previous post in this track, [NumPy from first principles](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), established *why* that gap exists: a NumPy array is a single contiguous, typed buffer, and a vectorized operation is one C loop walking that buffer, with no per-element `PyObject` boxing and no bytecode dispatch. This post is the *how*. It is the skill post — the one where we sit down with an actual Python loop and convert it, line by line, into an array expression, and measure the win at each step.

We will cover the four primitives that, between them, vectorize the overwhelming majority of numeric loops you will ever write: **ufuncs** (elementwise C functions like `+`, `np.exp`, `np.sqrt`), **broadcasting** (the shape-alignment rules that let a column and a row combine into a grid without copying anything), **reductions with `axis`** (collapsing a dimension with `sum`, `mean`, `max`), and the indexing family — **boolean masking** (`arr[arr > 0]`), **`np.where`/`np.select`** (vectorizing an if/else), and **fancy indexing** (gather and scatter with an index array). We will derive the broadcasting rule rather than memorize it, because once you can derive it you stop guessing about shapes. And — this matters — we will be honest about the loops that *cannot* be cleanly vectorized, where a data dependency or genuinely irregular control flow means the right answer is to stay in a loop and compile it instead.

![before and after comparison of an explicit per element Python loop running in hundreds of milliseconds versus a one line NumPy array expression running in a few milliseconds on a one million element input](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-1.png)

This whole exercise sits on the second rung of the leverage ladder from the [series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means): once you have done less work algorithmically, the next biggest lever for numeric code is to do the remaining work *in bulk*, in C, over packed memory. Vectorization is that lever, and it is the one that pays off most often for data and ML code, because so much of that code is just arithmetic over arrays that someone wrote as a loop because loops are how we learned to program.

## The machine and the measurement rules

Every number in this post comes from the same setup, stated once so you can calibrate. The machine is an **8-core x86-64 Linux box, CPython 3.12, NumPy 2.1, 16 GB of RAM**, with a comparable run on an **Apple M2 (CPython 3.12, NumPy 2.1)** where it differs enough to matter. Arrays are `float64` (8 bytes per element) unless I say otherwise. The default working size is **5 million elements** — big enough that per-call overhead is negligible and we are measuring the actual loop, small enough (40 MB) to stay resident in RAM.

I measure with `timeit`, taking the **median of repeated runs**, not the mean, because the mean is dragged around by the occasional GC pause or context switch while the median is not. For a quick interactive number I use IPython's `%timeit`, which auto-tunes the number of inner loops and reports the best estimate. For a recorded number I use `timeit.timeit` with explicit `number` and `repeat` and take the minimum-of-repeats divided by `number`, which gives a stable per-call figure. The same discipline from [the benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) applies here: warm up once, repeat, report the median or the min, and make the input big enough that the thing you care about dominates.

One trap specific to NumPy timing: do **not** let the result get optimized away or the inputs get cached cleverly. NumPy does not constant-fold, so that is not the risk; the risk is timing a tiny array where the answer is dominated by Python-side call overhead (a few microseconds to enter the ufunc machinery) rather than the C loop. If your array is 1,000 elements, you are mostly measuring the fixed cost of *calling* NumPy, not the per-element cost, and the speedup over a pure-Python loop will look unimpressive. The crossover where vectorization clearly wins is usually a few thousand elements; by a few million it is overwhelming. I will flag this again when we hit small-array territory.

```python
import numpy as np
import timeit

def bench(stmt, setup, number=100, repeat=7):
    """Return median seconds per call, the honest way."""
    times = timeit.repeat(stmt, setup, number=number, repeat=repeat, globals=globals())
    per_call = sorted(t / number for t in times)
    return per_call[len(per_call) // 2]  # median of the repeats
```

## The cost model: counting what a loop actually pays

Before the primitives, let me make the speedup *quantitative* rather than a vibe, because once you can put a number on the per-element overhead you can predict the win before you measure it, and you can tell which loops are worth converting.

Take the simplest possible body, `out[i] = a[i] + b[i]`, and account for every machine operation in each version. In pure Python, the interpreter executes roughly this sequence of bytecodes per iteration: `LOAD_FAST` the loop index, `LOAD_FAST a`, `BINARY_SUBSCR` (call `__getitem__`, bounds-check, return a `PyObject*` to a boxed float), the same for `b`, `BINARY_ADD` (type-check both operands, dispatch to `float.__add__`, unbox two `double`s, add, allocate a fresh boxed float for the result, set its refcount), then `STORE_SUBSCR` to write the pointer back, plus the loop machinery (`FOR_ITER`, the index increment, the jump). Each of those bytecodes is a switch-case in the CPython eval loop with its own argument decoding, and several of them allocate or touch reference counts. Empirically this lands around **80 to 250 nanoseconds per element** depending on the operation — the [hidden-cost-of-objects post](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) measures these individual costs in detail. Call it ~100 ns for a cheap arithmetic body.

The ufunc version does, per element, approximately: load two `double`s from contiguous memory into registers, add, store one `double`. That is **1 to 3 nanoseconds per element** — and with SIMD, where one instruction adds 4 or 8 `double`s at once, the *amortized* cost drops below 1 ns. No boxing (the values are raw `double`s in the buffer, never `PyObject`s), no refcounts (raw memory has no refcount), no bytecode (it is a compiled C loop, dispatched *once* for the whole array, not once per element), no per-element allocation (the output buffer is allocated once up front).

So the speedup is roughly the ratio of those constants: $\frac{100\text{ ns}}{2\text{ ns}} \approx 50\times$ for a cheap op, and more when SIMD kicks in or when the Python body did extra work. This is why the headline figure for vectorization is "one to two orders of magnitude" — it is the ratio of the interpreter's per-element constant to C's per-element constant, and that ratio is structurally 30–100×. It is *not* a property of NumPy being a clever library; it is the cost of asking the interpreter to drive a loop versus asking it once to call a C loop. Express the win as

$$\text{speedup} \approx \frac{c_{\text{py}} \cdot n + k_{\text{py}}}{c_{\text{c}} \cdot n + k_{\text{c}}}$$

where $c$ is the per-element cost and $k$ is the fixed per-call cost. For large $n$ the $k$ terms vanish and the speedup tends to $c_{\text{py}}/c_{\text{c}}$ — the constant ratio above. For *small* $n$ the fixed cost $k_{\text{c}}$ of entering NumPy (a few microseconds) dominates the numerator's $k_{\text{py}}$ (nearly zero for a Python loop), and the loop can actually win. That single formula explains both the 50–100× at scale and the crossover at small sizes, and it tells you the lever for memory-bound ops: when arithmetic is trivial, $c_{\text{c}}$ is dominated by *bytes moved*, so halving the dtype roughly halves $c_{\text{c}}$ — which is exactly what we saw with `float32`.

## Ufuncs: the elementwise C functions you already use

Start with the simplest and most important primitive. A **ufunc** — universal function — is a function that operates elementwise on arrays, implemented as a compiled C loop. You already use them without naming them: when you write `a + b` for two arrays, the `+` dispatches to `np.add`, a ufunc. When you write `np.exp(a)`, `np.sqrt(a)`, `np.sin(a)`, `a * 2`, `a > 0` — every one of those is a ufunc. The defining property is that the operation is applied independently to each element, so NumPy can run a single tight C loop over the buffer with zero Python involved per element.

Here is the entire reason ufuncs are fast, made concrete. Consider squaring a million numbers. In pure Python:

```python
def square_loop(xs):
    out = [0.0] * len(xs)
    for i in range(len(xs)):
        out[i] = xs[i] * xs[i]
    return out
```

Each trip through that loop does an enormous amount of bookkeeping that has nothing to do with multiplication. The interpreter executes bytecode to evaluate `xs[i]`, which means a `BINARY_SUBSCR` that calls `list.__getitem__`, returning a `PyObject*` pointer to a boxed float — a 24-byte heap object whose payload is the actual `double`. The `*` is a `BINARY_MULTIPLY` that does a type check, looks up `float.__mul__`, unboxes both operands to C `double`s, multiplies, allocates a *new* boxed float for the result, and increments and decrements a flurry of reference counts along the way. Then `STORE_SUBSCR` writes the pointer back. Dozens of C operations, two or three heap allocations, several refcount touches — to do one multiply.

The ufunc version is `xs * xs` on a NumPy array. NumPy sees two `float64` buffers, allocates one output buffer, and runs a C loop that is, at its core, `for (i=0; i<n; i++) out[i] = a[i] * b[i];` over raw `double*` pointers. No boxing, no refcounts, no bytecode, no per-element allocation. On modern CPUs the compiler and NumPy's own loop selection will even emit SIMD instructions so several multiplies happen per cycle. The arithmetic is identical; everything *around* the arithmetic is gone.

```pycon
>>> xs = list(range(1_000_000))
>>> arr = np.arange(1_000_000, dtype=np.float64)
>>> bench("square_loop(xs)", "from __main__ import square_loop, xs", number=10)
0.0381          # 38.1 ms per call, pure Python
>>> bench("arr * arr", "from __main__ import arr", number=1000)
0.000412        # 0.412 ms per call, ufunc
```

That is roughly **92×** on a million elements, on the Linux box. The M2 lands around 75–110× depending on cache behavior. Compose ufuncs freely — `np.sqrt(a**2 + b**2)` is three ufunc calls (square, add, sqrt), each a C loop — and you have vectorized a Euclidean-distance computation that would otherwise be a loop with a `math.sqrt` call per element.

#### Worked example: vectorizing a sigmoid over a batch

A neural-network feature pipeline applies the logistic sigmoid $\sigma(x) = 1 / (1 + e^{-x})$ to a batch of 5 million activations. The naive version maps `math.exp` over a list:

```python
import math

def sigmoid_loop(xs):
    return [1.0 / (1.0 + math.exp(-x)) for x in xs]
```

The vectorized version is a literal transcription using ufuncs:

```python
def sigmoid_vec(arr):
    return 1.0 / (1.0 + np.exp(-arr))
```

Measured on the Linux box, `float64`, 5M elements:

| Version | Per call | ns per element | Speedup |
|---|---:|---:|---:|
| `sigmoid_loop` (list, `math.exp`) | 690 ms | 138 ns | 1× |
| `sigmoid_vec` (ufuncs) | 31 ms | 6.2 ns | 22× |
| `sigmoid_vec`, `float32` | 17 ms | 3.4 ns | 41× |

The list version pays 138 ns per element — almost all of it interpreter overhead and the per-call cost of `math.exp`, which is a Python-level function call. The vectorized version pays 6.2 ns per element, and *that* is now genuinely dominated by the transcendental `exp` itself (computing $e^{-x}$ is intrinsically expensive even in C) plus the memory traffic of three temporaries. Switching to `float32` halves the bytes moved and the per-element time drops again to 3.4 ns, because for this memory-bandwidth-sensitive op, moving half the bytes is most of the win — a preview of the bandwidth story in [the strides and cache post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache).

The lesson: vectorizing did not change the math, it changed *who runs it*. And note the honest caveat — 22× here, not 100×, because `exp` is expensive enough that even in C it costs real cycles. The speedup of vectorization depends on how much work the interpreter overhead was hiding. For cheap ops (add, multiply) the overhead dominates and you see 50–100×; for expensive ops (`exp`, `sin`) the real arithmetic shows through and you see 15–30×.

![a stacked diagram of the vectorization toolkit showing five layers from elementwise ufuncs up through reductions with axis then broadcasting then masking and finally fancy indexing](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-6.png)

## Broadcasting: deriving the shape-alignment rule

Ufuncs are elementwise, which naively means both operands must have the *same* shape. Broadcasting is the set of rules that relaxes that, letting NumPy combine arrays of *different but compatible* shapes by virtually stretching the smaller one — without ever materializing the stretched copy. This is the primitive people find most mysterious and it is the one most worth understanding deeply, because it is what lets you eliminate the *outer* loops, not just the inner one.

Let me build the rule from the requirement rather than stating it. A ufunc walks two buffers in lockstep, reading `a[i]` and `b[i]` and writing `out[i]`. For that to make sense across multiple dimensions, NumPy needs, for every output position, a defined input element from each operand. Broadcasting answers the question: given two shapes, what is the output shape, and how do we map each output position back to an input element of each array?

**The rule, in two parts.** Align the two shapes by their *trailing* dimensions (right-justify them). Then for each dimension, the sizes are compatible if they are **equal**, or if **one of them is 1**. The output size in that dimension is the larger of the two. If a dimension is size 1 in one operand, that operand is *stretched* along that dimension — meaning NumPy reads the same single element repeatedly instead of stepping forward. Shapes with different numbers of dimensions are aligned by prepending size-1 dimensions to the shorter shape until they match in rank.

Why "one of them is 1" and not some other rule? Because a size-1 axis has exactly one element along it, and stretching it means "reuse that one element for every output position along this axis." NumPy implements that by setting the **stride** for that axis to 0 — the C loop's pointer simply does not advance when it walks that dimension, so it re-reads the same memory. *That* is the deep reason broadcasting does not copy: a stretched axis is a stride-0 axis, a pure indexing trick, no new bytes. There is no expanded array; there is a buffer plus a clever set of strides that makes one element look like many. (Strides are the subject of [the memory-layout post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache); here you only need to know a stride is how many bytes to step to reach the next element along an axis, and a stride of 0 means "stay put.")

Let me work the canonical example by hand. Take a column vector of shape `(3, 1)` and a row vector of shape `(1, 4)`:

```pycon
>>> col = np.array([[10], [20], [30]])        # shape (3, 1)
>>> row = np.array([[1, 2, 3, 4]])            # shape (1, 4)
>>> (col + row).shape
(3, 4)
>>> col + row
array([[11, 12, 13, 14],
       [21, 22, 23, 24],
       [31, 32, 33, 34]])
```

Walk the rule. Align trailing dims: dimension 1 is `(1)` vs `(4)` — one is 1, so output is 4 and `col` stretches along columns. Dimension 0 is `(3)` vs `(1)` — one is 1, so output is 3 and `row` stretches along rows. Output shape `(3, 4)`. For output position `(i, j)`, NumPy reads `col` at row `i`, column 0 (because `col`'s column axis is stretched, stride 0 — it never advances past column 0), and reads `row` at row 0, column `j` (because `row`'s row axis is stretched). So position `(1, 2)` is `col[1, 0] + row[0, 2] = 20 + 3 = 23`. The whole `(3, 4)` grid is filled by reading from a 3-element buffer and a 4-element buffer — never from a 12-element expanded copy.

![a grid showing a column of shape three by one and a row of shape one by four aligning to fill a three by four result grid where each cell is the sum of its column value and its row offset](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-2.png)

This is exactly the normalization that opened the post. To subtract a per-column mean from a `(5_000_000, 40)` table, you compute a `(40,)` vector of column means and subtract it. Aligning trailing dims: `(5_000_000, 40)` vs `(40,)` — the `(40,)` is treated as `(1, 40)` by the rank-prepending rule, then dimension 0 is `(5_000_000)` vs `(1)`, so the mean vector stretches down all five million rows with stride 0. One C loop, no copy of the mean across rows.

```python
table = np.random.rand(5_000_000, 40)
col_mean = table.mean(axis=0)          # shape (40,)
col_std  = table.std(axis=0)           # shape (40,)
normalized = (table - col_mean) / col_std   # broadcasts (40,) over 5M rows
```

#### Worked example: why broadcasting saves a giant temporary

Suppose you want a pairwise difference: for `x` of shape `(n,)`, build the `(n, n)` matrix `D[i, j] = x[i] - x[j]`. The broadcasting expression is `x[:, None] - x[None, :]` — reshape `x` to a column `(n, 1)` and a row `(1, n)`, subtract, and broadcasting produces `(n, n)`.

Now consider the memory. The *result* is genuinely `(n, n)` and must be materialized — for `n = 20_000` that is 400 million `float64`s, **3.2 GB**, which is a real cost you should think twice about. But the *inputs* are not expanded. A naive person might fear that `x[:, None]` and `x[None, :]` each get tiled into a `(20000, 20000)` array before subtracting — that would be **6.4 GB of temporaries on top of the 3.2 GB result**, and the program would OOM. Broadcasting avoids exactly that. `x[:, None]` is a `(20000, 1)` *view* (8 bytes × 20000 = 160 KB) with a stride-0 column axis; `x[None, :]` is a `(1, 20000)` view (also 160 KB) with a stride-0 row axis. The subtract loop reads through those stride-0 axes and writes the `(n, n)` result directly. Peak memory is `result + two tiny views`, not `result + two huge tiles`.

| Quantity | Materialized-tile approach | Broadcasting |
|---|---:|---:|
| Left operand storage | 3.2 GB (tiled) | 160 KB (view) |
| Right operand storage | 3.2 GB (tiled) | 160 KB (view) |
| Result storage | 3.2 GB | 3.2 GB |
| Peak RAM | ~9.6 GB → likely OOM | ~3.2 GB |

That is the practical payoff of the stride-0 mechanism: broadcasting lets you write the clean `(n, n)` expression without paying for the expanded inputs. (If even the 3.2 GB result is too much, that is a signal to chunk the computation or rethink whether you need the full matrix — but that is a separate problem from broadcasting's temporaries.)

It is worth seeing the rank-mismatch case once, because it is where people get lost. Suppose you have a batch of images shaped `(batch, height, width, channels) = (64, 32, 32, 3)` and a per-channel mean shaped `(3,)`, and you want to subtract the mean from each channel. Align trailing dims: `(64, 32, 32, 3)` versus `(3,)`. The shorter shape is *right-justified* and padded on the left with size-1 axes until the ranks match, so `(3,)` is treated as `(1, 1, 1, 3)`. Now compare axis by axis: channels `3` vs `3` match; width `32` vs `1` stretches the mean; height `32` vs `1` stretches; batch `64` vs `1` stretches. The `(3,)` mean is broadcast across all 64 images, all 32 rows, all 32 columns — each of those three padded axes has stride 0 — and only the channel axis actually advances. One subtract, the 3-element mean read through stride-0 axes, no `(64, 32, 32, 3)` copy of the mean. This is *exactly* how per-channel normalization in every image pipeline works, and now you can derive why `mean.reshape(1, 1, 1, 3)` (or `mean[None, None, None, :]`) is sometimes written explicitly: to make the alignment unambiguous when you are not subtracting against the trailing axis.

The single most useful debugging habit follows directly: when a broadcast does not do what you expect, it is almost always because the axis you *meant* to align is not the trailing one. The fix is to insert size-1 axes with `[:, None]`, `[None, :]`, or `np.newaxis` so the axes you intend to combine line up under the right-justification rule. If you want a `(40,)` mean to broadcast *down rows* of a `(5_000_000, 40)` table, it already aligns (the 40 is trailing). If you wanted a `(5_000_000,)` per-*row* statistic to broadcast *across columns*, you must write it as `(5_000_000, 1)` with `stat[:, None]`, or the trailing-axis rule will try to align 5,000,000 against 40 and raise. Print `.shape` of both operands, right-justify them in your head, and the bug is always visible.

A caution worth internalizing: broadcasting failures are loud and that is good. If you try `np.ones((3, 4)) + np.ones((3, 5))`, you get `ValueError: operands could not be broadcast together with shapes (3,4) (3,5)`. When you hit that, read it as "my trailing dimensions are not compatible," check your shapes with `.shape`, and reshape with `[:, None]`, `np.newaxis`, or `reshape` to line up the axes you intend to combine. Ninety percent of NumPy bugs are shape bugs, and the fix is always to print shapes and re-derive the alignment.

![a graph showing two input buffers where one is a scalar that broadcasts feeding into a single C elementwise loop that writes one output buffer with no temporaries](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-5.png)

## Reductions and the axis argument

The third primitive collapses a dimension. A **reduction** combines many elements into fewer — `sum`, `mean`, `max`, `min`, `prod`, `std`, `any`, `all`, `argmax`. By itself `arr.sum()` reduces the entire array to a scalar, running a single C accumulation loop. The lever that makes reductions powerful for real data is the **`axis`** argument, which says *which* dimension to collapse, leaving the others intact.

This is where people coming from loops get the most leverage, because a reduction with `axis` replaces a *nested* loop. Consider summing each row of a matrix:

```python
def rowsum_loop(mat):
    n, m = mat.shape
    out = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += mat[i, j]
        out[i] = s
    return out
```

Two nested Python loops, `n × m` iterations, each one a boxed indexing-and-add. The vectorized form is a single call:

```python
out = mat.sum(axis=1)    # sum along columns, one value per row
```

The rule for `axis`: the axis you name is the one that *disappears*. `mat` is `(n, m)`; `sum(axis=1)` collapses axis 1 (the columns) and leaves `(n,)` — one sum per row. `sum(axis=0)` collapses axis 0 (the rows) and leaves `(m,)` — one sum per column. `sum()` with no axis collapses everything to a scalar. This is consistent across all reductions, so `mat.mean(axis=0)` is the per-column mean, `mat.max(axis=1)` is the per-row maximum, and so on. The column-mean and column-std in the normalization example were exactly `axis=0` reductions.

```pycon
>>> mat = np.random.rand(50_000, 200)
>>> bench("rowsum_loop(mat)", "from __main__ import rowsum_loop, mat", number=3)
2.41            # 2.41 s, nested Python loops
>>> bench("mat.sum(axis=1)", "from __main__ import mat", number=200)
0.00187         # 1.87 ms, one reduction
```

That is roughly **1,290×** for `50_000 × 200`, the kind of number that turns a minutes-long aggregation step into milliseconds. The nested loop is paying the interpreter tax `n × m = 10` million times; the reduction pays it zero times.

A reduction subtlety that bites people: **`keepdims`**. Often you reduce and then want to broadcast the result back against the original. `mat.sum(axis=1)` gives shape `(n,)`, but `mat - mat.mean(axis=1)` fails because `(n, m) - (n,)` aligns the `(n,)` against the trailing axis `m`, not the rows you intended. The fix is `keepdims=True`, which keeps the collapsed axis as size 1 — `mat.mean(axis=1, keepdims=True)` gives `(n, 1)`, which then broadcasts cleanly over the columns. This is the row-normalize idiom:

```python
row_mean = mat.mean(axis=1, keepdims=True)   # (n, 1), not (n,)
centered = mat - row_mean                     # broadcasts over columns, no copy
```

Reductions also have a numerical-stability note worth a sentence: NumPy's `sum` uses **pairwise summation**, which keeps floating-point error far lower than a naive left-to-right accumulation, so the vectorized sum is not only faster than your loop, it is usually *more accurate* than a naive `for`-loop accumulation. You get speed and correctness in the same call.

There is a memory-layout subtlety to reductions that connects forward to the cache. `mat.sum(axis=1)` (per-row sums) walks each row contiguously in a C-ordered array, which is cache-friendly and fast; `mat.sum(axis=0)` (per-column sums) strides across rows, touching one element from each cache line, which is slower for the same element count because of cache misses. The element count is identical; the *access pattern* differs, and on a large array `axis=0` can be several times slower than `axis=1` purely from locality. You usually do not need to think about this, but when a reduction is unexpectedly slow, the axis-versus-layout interaction is the first thing to check — and the reason [the strides and cache post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) exists. Most reductions take an `out=` parameter and a `dtype=` parameter too: `out=` lets you reduce into a preallocated buffer (no allocation), and `dtype=np.float64` lets you accumulate `float32` data in higher precision to avoid catastrophic cancellation on long sums — a one-keyword fix for a class of silent numeric bugs.

## Boolean masking: vectorizing a filter and an if

Now the indexing family, which is how you vectorize *conditional* logic — the loops with an `if` inside. A **boolean mask** is an array of `True`/`False` the same shape as your data, produced by a comparison ufunc. `arr > 0` does not loop in Python; it runs a C loop emitting one boolean per element. You then use that mask to select, count, or assign.

```pycon
>>> arr = np.array([-2.0, 3.0, -1.0, 5.0, 0.0, -4.0])
>>> mask = arr > 0
>>> mask
array([False,  True, False,  True, False, False])
>>> arr[mask]                 # select: pull out the elements where True
array([3., 5.])
>>> mask.sum()                # count: True is 1, so this counts the positives
2
>>> arr[arr > 0] = 0.0        # assign: clamp positives to zero in place
>>> arr
array([-2.,  0., -1.,  0.,  0., -4.])
```

Three operations, all in C. `arr[mask]` is boolean *indexing*: it returns a new 1-D array of just the selected elements (a copy, because the result is a different length than the input). `mask.sum()` counts how many passed, since a boolean reduces as 0/1 — `(arr > 0).mean()` gives the *fraction* positive, a one-liner for "what share of my data is above threshold." And `arr[arr > 0] = 0.0` is masked *assignment*, which writes only into the selected positions, in place, no copy.

Compare to the loop you are replacing:

```python
def clamp_loop(arr):
    out = arr.copy()
    for i in range(len(out)):
        if out[i] > 0:
            out[i] = 0.0
    return out
```

The branch — `if out[i] > 0` — is the expensive part in Python, because every iteration the interpreter evaluates the comparison, jumps on the result, and either does or skips the assignment, all in bytecode. The masked version evaluates the entire condition as one C compare into a boolean buffer and does the entire assignment as one C masked-store. The branch never reaches the interpreter.

Here is the science of *why the branch stays in C*. A Python `if` per element forces a round trip to the interpreter for the comparison and the conditional jump on every single element — there is no way to "batch" a Python-level branch. NumPy reframes the branch as **data**: the condition becomes a boolean array (computed in one pass), and the conditional action becomes an indexing or `where` operation driven by that array. Once the branch is data, the whole thing is one or two C loops with no per-element Python. You have traded *control flow* (which the interpreter must drive) for *data flow* (which C handles in bulk).

Masks **combine** with the bitwise operators `&` (and), `|` (or), and `~` (not), which is how you vectorize a compound condition — but mind two traps. First, you must use `&`/`|`/`~`, *not* Python's `and`/`or`/`not`, because the keyword operators try to evaluate the whole array's truth value and raise `ValueError: The truth value of an array with more than one element is ambiguous`. Second, `&`/`|` bind *tighter* than the comparison operators, so you must parenthesize each comparison: write `(arr > 0) & (arr < 10)`, never `arr > 0 & arr < 10` (which parses as `arr > (0 & arr) < 10` and is wrong). With that in hand, a multi-condition filter is one expression:

```pycon
>>> arr = np.array([-5.0, 3.0, 12.0, 7.0, -1.0, 9.5])
>>> band = (arr > 0) & (arr < 10)        # compound mask, both conditions in C
>>> arr[band]
array([3. , 7. , 9.5])
>>> arr[~band] = np.nan                   # invert the mask and assign
>>> arr
array([nan,  3., nan,  7., nan,  9.5])
```

Each comparison is a C pass producing a boolean buffer; the `&` is a C pass combining two boolean buffers; the final index or assignment is a C pass. Three or four C passes, zero Python branches, replacing a loop with a compound `if (x > 0 and x < 10)` inside. This is the workhorse for data cleaning — "keep rows where amount is positive and date is in range and status is active" becomes a chain of `&`-ed masks applied once.

#### Worked example: filtering and clamping 5 million rows

Take a 5M-element `float64` array of signed values. We want to clamp every positive value to a ceiling of 1.0 and leave the rest, then separately count how many were clamped. Loop version versus mask version, Linux box:

```python
ceiling = 1.0

def clamp_count_loop(arr):
    out = arr.copy()
    n = 0
    for i in range(len(out)):
        if out[i] > ceiling:
            out[i] = ceiling
            n += 1
    return out, n

def clamp_count_vec(arr):
    out = arr.copy()
    mask = out > ceiling           # one C compare → boolean buffer
    out[mask] = ceiling            # one C masked store
    return out, int(mask.sum())    # one C reduction for the count
```

| Version | Per call | ns per element | Speedup |
|---|---:|---:|---:|
| `clamp_count_loop` | 1,140 ms | 228 ns | 1× |
| `clamp_count_vec` | 14.8 ms | 3.0 ns | 77× |

Three C passes (compare, store, count) beat one Python loop by 77×. And notice we did *three* passes over the array in the vectorized version versus one pass in the loop — yet it is still 77× faster. That is the headline counterintuition of vectorization: **three C passes over packed memory routinely beat one Python pass over boxed objects**, because the per-element constant is something like 1–3 ns in C versus 100–250 ns in the interpreter. You can afford to be "wasteful" with passes as long as each pass is in C. (When you have so many passes that *memory bandwidth* becomes the wall, that is when you reach for fusion — `numexpr` or Numba — covered in [the next post on avoiding temporaries](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means); for two or three passes you are nowhere near that wall.)

## np.where and np.select: the vectorized if/else

Masking handles "do something to the elements that match." But the most common conditional is an if/**else** — produce one value when the condition holds and a *different* value when it does not. That is `np.where(condition, value_if_true, value_if_false)`, the single most useful function for translating branchy loops.

```pycon
>>> x = np.array([-2.0, 3.0, -1.0, 5.0])
>>> np.where(x > 0, x, 0.0)        # ReLU: x if positive, else 0
array([0., 3., 0., 5.])
>>> np.where(x > 0, 1, -1)         # sign-ish: +1 where positive, -1 elsewhere
array([-1,  1, -1,  1])
```

`np.where` evaluates the condition into a boolean array, then for each position picks from the second argument where `True` and the third where `False`. All three arguments broadcast against each other, so the "values" can be scalars, full arrays, or anything broadcast-compatible — `np.where(x > 0, x * 2, x / 2)` doubles positives and halves the rest. This directly transcribes the classic if/else loop:

```python
def signed_scale_loop(x):
    out = np.empty_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            out[i] = x[i] * 2.0
        else:
            out[i] = x[i] / 2.0
    return out

def signed_scale_vec(x):
    return np.where(x > 0, x * 2.0, x / 2.0)
```

![before and after comparison of an if else loop over five million rows versus a single np.where call showing the loop at almost two seconds and the where call at twenty two milliseconds](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-4.png)

One thing to know about `np.where`: it evaluates *both* branches fully — `x * 2.0` and `x / 2.0` are each computed over the whole array, then selected between. This is almost always fine and still vastly faster than the loop. But if one branch is invalid where the condition is false (a classic is `np.where(x > 0, np.log(x), 0)`, which computes `log` of negatives and warns about invalid values before discarding them), you either suppress the warning, or guard the input, or use masked assignment instead. Measured on the 5M-element if/else above, Linux box:

| Version | Per call | Speedup |
|---|---:|---:|
| `signed_scale_loop` | 1,920 ms | 1× |
| `signed_scale_vec` (`np.where`) | 22 ms | 87× |

For **more than two** branches — a chain of if/elif/elif/else — reach for `np.select`, which takes a list of conditions and a list of choices and applies the first matching condition per element, with an optional default. This vectorizes a multi-way bucketing:

```python
conditions = [x < 0, x < 10, x < 100]
choices    = ["neg", "small", "medium"]
labels = np.select(conditions, choices, default="large")
```

That replaces a four-arm if/elif/elif/else loop with one call. The conditions are evaluated as boolean arrays and `np.select` walks them in order, which is the vectorized equivalent of the short-circuiting elif chain. For numeric bucketing specifically, `np.digitize` or `np.searchsorted` against bin edges is even faster, but `np.select` is the readable general tool.

![a matrix mapping five common loop patterns of accumulate conditional filter gather and scale to their vectorized equivalents of sum where boolean mask fancy index and broadcast](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-3.png)

## Fancy indexing: gather and scatter with index arrays

The last primitive is **fancy indexing** — indexing with an *array of integers* instead of a single integer or a slice. This is how you vectorize a *gather* (read elements at a list of positions) or a *scatter* (write elements to a list of positions), which shows up constantly in lookup tables, reordering, sampling, and sparse updates.

```pycon
>>> src = np.array([10, 20, 30, 40, 50])
>>> idx = np.array([4, 0, 2, 2, 1])
>>> src[idx]                      # gather: read src at each index in idx
array([50, 10, 30, 30, 20])
```

`src[idx]` returns a new array the same shape as `idx`, where each element is `src` at that index. It is the vectorized form of `out = [src[j] for j in idx]` — a gather. The index array can repeat values (note `2` appears twice above, giving `30` twice) and can be any shape; the result takes the shape of the index array. This is the engine behind one-hot lookups, embedding lookups, applying a permutation, and table-driven transforms.

```python
def gather_loop(src, idx):
    out = np.empty(len(idx), dtype=src.dtype)
    for k in range(len(idx)):
        out[k] = src[idx[k]]
    return out

def gather_vec(src, idx):
    return src[idx]               # the entire loop, in C
```

The reverse direction is **scatter** — writing to positions named by an index array: `dst[idx] = values`. Most of the time this does what you want. The sharp edge is **duplicate indices**: `dst[idx] = values` with a repeated index leaves the *last* write winning, which is fine for assignment but wrong if you meant to *accumulate*. For accumulation into duplicate positions — the classic "bin these events into these buckets" — you must use `np.add.at(dst, idx, values)`, the unbuffered scatter-add, or better, `np.bincount`, which is purpose-built and far faster:

```pycon
>>> counts = np.zeros(5, dtype=int)
>>> idx = np.array([1, 1, 3, 1])
>>> counts[idx] += 1            # WRONG for counting: result is [0,1,0,1,0]
>>> counts[:] = 0
>>> np.add.at(counts, idx, 1)  # correct scatter-add: [0,3,0,1,0]
>>> np.bincount(idx, minlength=5)  # faster purpose-built counter
array([0, 3, 0, 1, 0])
```

Why does `counts[idx] += 1` undercount? Because it is sugar for `counts[idx] = counts[idx] + 1`: it *gathers* `counts[idx]` (reading the old value `0` three times for index `1`, since the gather happens before any write), adds 1 to each, and *scatters* back — so all three writes to index `1` store `0 + 1 = 1` and the last wins. There is no read-modify-write per duplicate; the gather and scatter are separate bulk steps. `np.add.at` performs the read-modify-write *per index occurrence*, which is the unbuffered behavior you need. This is one of the genuinely surprising NumPy gotchas and worth committing to memory: **fancy-indexed in-place accumulation needs `np.add.at` or `np.bincount`, never `+=`.**

#### Worked example: building a histogram by gather-scatter

Bin 5 million event values into 256 buckets and count. The loop walks events and increments a bucket; the vectorized version computes bucket indices with a ufunc and counts with `np.bincount`. Linux box:

```python
n_bins = 256

def histogram_loop(events):
    counts = np.zeros(n_bins, dtype=np.int64)
    for v in events:
        b = int(v * n_bins)        # which bucket
        counts[b] += 1
    return counts

def histogram_vec(events):
    bins = (events * n_bins).astype(np.int64)   # ufunc: all bucket indices at once
    return np.bincount(bins, minlength=n_bins)  # C scatter-add
```

| Version | Per call | Speedup |
|---|---:|---:|
| `histogram_loop` | 1,560 ms | 1× |
| `histogram_vec` (`bincount`) | 18 ms | 87× |

The loop's per-event work — multiply, `int()`, index, increment — is all interpreter overhead. The vectorized version does the index computation as one ufunc pass and the counting as one C scatter-add. (`np.histogram` exists for exactly this with float bin edges; I used `bincount` to show the gather-scatter mechanism explicitly.)

![a matrix listing four operations of sum normalize threshold and pairwise with their naive loop time and vectorized time and the resulting speedup ranging from sixty to one hundred thirty times](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-7.png)

## The core skill: converting a loop step by step

Everything above is a primitive. The actual skill is *converting your loop* — looking at a `for`-loop you wrote and mechanically rewriting it as array expressions, measuring at each step so you know the rewrite is correct and you can see the win accumulate. Let me walk one all the way through, because the procedure generalizes.

The task: a risk-scoring step over a table of transactions. For each transaction we have an `amount`, a `risk_weight` per category (looked up from a category index), and a flag for whether the account is new. The score is `amount * risk_weight`, doubled if the account is new, and then clamped to a maximum of 10,000. Here is the original loop, exactly as someone would write it first:

```python
def score_loop(amounts, category_idx, risk_weights, is_new):
    n = len(amounts)
    scores = np.empty(n)
    for i in range(n):
        w = risk_weights[category_idx[i]]   # gather the weight for this row's category
        s = amounts[i] * w                  # base score
        if is_new[i]:                       # conditional doubling
            s = s * 2.0
        if s > 10_000.0:                    # clamp
            s = 10_000.0
        scores[i] = s
    return scores
```

This loop has all four patterns in it: a **gather** (`risk_weights[category_idx[i]]`), an **elementwise** multiply, a **conditional** (the doubling), and a **clamp** (another conditional). We convert one pattern at a time, re-timing after each so a mistake shows up immediately as a wrong answer or a non-improvement.

**Step 0 — baseline.** Generate 5M rows and time the loop. On the Linux box: **2,310 ms**. Save the result array so we can assert correctness after every rewrite.

**Step 1 — vectorize the gather.** `risk_weights[category_idx]` is fancy indexing: it gathers the weight for *every* row in one C pass. Replace the per-row lookup. The rest stays a loop for now over the gathered weights:

```python
def score_step1(amounts, category_idx, risk_weights, is_new):
    weights = risk_weights[category_idx]    # vectorized gather, shape (n,)
    n = len(amounts)
    scores = np.empty(n)
    for i in range(n):
        s = amounts[i] * weights[i]
        if is_new[i]:
            s *= 2.0
        if s > 10_000.0:
            s = 10_000.0
        scores[i] = s
    return scores
```

Time: **1,980 ms**. A modest win — we removed the in-loop indexing but the loop still dominates. The point of doing it incrementally is that `score_step1` produces *exactly* the same array as `score_loop` (assert it), so we know the gather rewrite is correct before we touch anything else.

**Step 2 — vectorize the elementwise multiply and the conditional doubling.** The base score `amounts * weights` is a ufunc. The doubling-if-new is an if/else over a multiplier: the multiplier is `2.0` where `is_new` and `1.0` otherwise, which is `np.where(is_new, 2.0, 1.0)`. So:

```python
def score_step2(amounts, category_idx, risk_weights, is_new):
    weights = risk_weights[category_idx]
    base = amounts * weights                       # ufunc
    factor = np.where(is_new, 2.0, 1.0)            # vectorized if/else → multiplier
    scores = base * factor                          # ufunc
    # clamp still a loop
    for i in range(len(scores)):
        if scores[i] > 10_000.0:
            scores[i] = 10_000.0
    return scores
```

Time: **240 ms** — a 9.6× drop from baseline, because the bulk of the per-row arithmetic is now in C and only the clamp loop remains. Same result array (assert).

**Step 3 — vectorize the clamp.** Clamping to a maximum is `np.minimum(scores, 10_000.0)` — an elementwise min against a scalar, which broadcasts. (Equivalently `np.where(scores > 10_000, 10_000, scores)`, or `np.clip(scores, None, 10_000)`.) The fully vectorized function:

```python
def score_vec(amounts, category_idx, risk_weights, is_new):
    weights = risk_weights[category_idx]            # gather
    base = amounts * weights                        # elementwise
    base *= np.where(is_new, 2.0, 1.0)              # conditional doubling, in place
    return np.minimum(base, 10_000.0)              # clamp
```

Time: **41 ms** — **56× faster** than the original loop, and four readable lines. The full ladder:

| Step | What we vectorized | Per call | Cumulative speedup |
|---|---|---:|---:|
| 0 — baseline loop | nothing | 2,310 ms | 1× |
| 1 — gather | `risk_weights[category_idx]` | 1,980 ms | 1.2× |
| 2 — elementwise + if/else | `amounts*weights`, `np.where` | 240 ms | 9.6× |
| 3 — clamp | `np.minimum` | 41 ms | 56× |

#### Worked example: converting a distance-and-bucket loop

A second conversion, in a different shape, to show the procedure is mechanical. The task: given `points` of shape `(n, 2)` (x, y coordinates) and a single `center`, compute each point's Euclidean distance to the center, then bucket each distance into one of three rings — "near" (< 1.0), "mid" (< 3.0), "far" (otherwise) — and count how many points land in each ring. The original loop:

```python
import math

def rings_loop(points, center):
    counts = {"near": 0, "mid": 0, "far": 0}
    for i in range(len(points)):
        dx = points[i, 0] - center[0]
        dy = points[i, 1] - center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1.0:
            counts["near"] += 1
        elif dist < 3.0:
            counts["mid"] += 1
        else:
            counts["far"] += 1
    return counts
```

Classify the body: a **broadcast** subtract (`points - center`), an **elementwise** square-and-add-and-sqrt (ufuncs), and a **multi-way conditional** count (a job for `np.select` or, more directly, summing boolean masks). Convert in two moves. First the distance, which is pure broadcasting and ufuncs:

```python
def distances(points, center):
    diff = points - center               # (n, 2) - (2,) broadcasts → (n, 2)
    return np.sqrt((diff ** 2).sum(axis=1))   # reduce the coordinate axis → (n,)
```

The `points - center` broadcasts the 2-element center across all `n` rows (trailing axis `2` matches), `diff ** 2` is a ufunc, `.sum(axis=1)` reduces the coordinate axis to one distance-squared per point, and `np.sqrt` is a ufunc. Then the bucketing — three boolean masks summed gives the three counts, no `np.select` needed because we only want counts:

```python
def rings_vec(points, center):
    dist = distances(points, center)
    near = dist < 1.0                     # boolean mask
    mid  = (dist >= 1.0) & (dist < 3.0)   # compound mask
    return {
        "near": int(near.sum()),
        "mid":  int(mid.sum()),
        "far":  int((dist >= 3.0).sum()),
    }
```

Measured on 5M points, Linux box:

| Version | Per call | Speedup |
|---|---:|---:|
| `rings_loop` (Python, `math.sqrt`, dict) | 4,100 ms | 1× |
| `rings_vec` (broadcast + ufunc + masks) | 49 ms | 84× |

Same recipe as before: classify each piece of the body, map it to a primitive, chain them, and assert the counts match the loop. The distance computation is broadcasting plus a reduction; the bucketing is masks. Eighty-four times faster, and the vectorized version is arguably *clearer* about what it computes — a distance, then three ring counts — than the loop with its running dict.

Two procedural lessons from both conversions. First, **the biggest jump came from removing the loop body, not from any single clever function** — in the scoring ladder, steps 2 and 3 together eliminated the per-row interpreter trip, and that is where the 56× lives, while the gather rewrite alone (step 1) barely moved the needle because the loop was still there. Second, **assert equality after every step.** Vectorized code is easy to get subtly wrong (a broadcast you did not intend, a `where` with swapped branches, an off-by-one in an index array), and the only cheap defense is to keep the original loop around and `np.testing.assert_allclose(new, baseline)` after each rewrite. Once the fully vectorized version matches, delete the loop.

![a decision tree for vectorizing a loop branching on whether the loop body is elementwise conditional a reduction or has a data dependency and pointing each to ufunc where select sum axis or compile](/imgs/blogs/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing-8.png)

## When a loop cannot be cleanly vectorized

Now the honest part, because the worst thing this post could do is convince you that *every* loop becomes an array expression. Some do not, and recognizing them saves you hours of fighting NumPy into knots that are slower and less readable than the loop you started with.

**Data dependencies.** The clearest blocker is when element `i` depends on the *result* at element `i-1`. A running maximum, an exponential moving average, a recursive filter, a state machine that carries state across rows — these have a sequential dependency that an elementwise ufunc cannot express, because a ufunc computes every output independently. Some specific dependencies have dedicated cumulative ufuncs (`np.cumsum`, `np.cumprod`, `np.maximum.accumulate`, `np.cumulative_sum`) that NumPy implements as a single C scan, and you should reach for those when they fit. But a *general* recurrence like an EWMA `y[i] = alpha * x[i] + (1 - alpha) * y[i-1]` has no built-in scan, and a pure-NumPy "trick" for it (building a matrix of decay weights) is $O(n^2)$ and worse than the loop. The right move there is to keep the loop and **compile it** with Numba's `@njit`, which turns the sequential loop into machine code at near-C speed without leaving Python — that is the next rung of the ladder and the subject of the native-acceleration track.

**Irregular control flow.** Loops whose body does genuinely different amounts of work per element — early-exit search, ragged nested structures, "process until a condition then break," variable-length per-row work — do not map to the fixed-shape, do-the-same-thing-to-every-element model of ufuncs. You can sometimes vectorize a bounded version (compute all branches with `np.where` and select), but if the branches are expensive and rarely taken, computing them all is wasteful and the loop (compiled) can win.

**Operations that aren't elementwise or reducible.** Parsing strings, calling out to an external service per row, building heterogeneous objects, anything that touches Python objects rather than numbers — NumPy's array world is for *numbers in packed buffers*. The moment your per-element work is "call this Python function" (the `np.vectorize` trap — it is a convenience wrapper that loops in Python and gives you *none* of the speed), you are not vectorizing, you are looping with extra steps. For per-row Python logic over tabular data, the right tool is often a dataframe engine — pandas vectorized ops or Polars expressions, covered in [the dataframes post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — or, again, a compiled loop.

The decision is captured in the tree above: classify the loop body. **Elementwise** → ufunc. **Conditional** → `where`/`select`/masking. **Reduction** → `sum`/`mean`/`max` with `axis`. **Gather or scatter** → fancy indexing (`np.add.at`/`bincount` for duplicate-index accumulation). **Carries a dependency across iterations** → a cumulative ufunc if one exists, otherwise *stay in the loop and compile it*. Vectorization is the right tool for the first four; for the fifth, knowing it is the fifth is the skill.

#### Worked example: an EWMA that should NOT be vectorized

Make the data-dependency case concrete, because the temptation to vectorize it leads people somewhere expensive. An exponential moving average is `y[i] = alpha * x[i] + (1 - alpha) * y[i-1]` — each output depends on the *previous output*, a textbook sequential recurrence. The honest loop:

```python
def ewma_loop(x, alpha):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y
```

There *is* a closed form: $y_i = \alpha \sum_{j=0}^{i} (1-\alpha)^{\,i-j} x_j$, which you could build as a lower-triangular matrix of decay weights and a matrix-vector product. But that matrix is $(n, n)$ — for $n = 100{,}000$ it is 80 GB of `float64`, instantly out of the question, and even at small $n$ it is $O(n^2)$ work to avoid an $O(n)$ loop. That is the trap: the "clever vectorized trick" is asymptotically *worse* than the loop it replaces. Here are the three honest options, on 1M `float64` elements, Linux box:

| Approach | Per call | Notes |
|---|---:|---|
| `ewma_loop` (pure Python) | 240 ms | $O(n)$, but interpreter-bound |
| matrix decay-weights trick | OOM at 100k | $O(n^2)$ time and memory — do not |
| `@njit` compiled loop (Numba) | 1.6 ms | same $O(n)$ loop, in machine code |

The right answer is the *third*: keep the sequential loop exactly as written and put `@numba.njit` on it, which compiles it to machine code and runs the recurrence at C speed — about **150×** over the Python loop, with none of the $O(n^2)$ blowup. This is the entire reason the leverage ladder has a *compile* rung above the *vectorize* rung: some loops genuinely cannot be expressed as independent elementwise work, and for those you compile the loop rather than torture it into an array expression. (Numba and the compile rung are covered in the next track of this series; the point here is recognizing *which* loops belong there.) The cumulative ufuncs — `np.cumsum`, `np.cumprod`, `np.maximum.accumulate` — handle the *specific* recurrences that have a built-in scan; a general recurrence like EWMA does not, and that is the signal to compile.

## Putting the toolkit together: a real aggregation

Let me show one realistic composition that uses four of the five primitives at once, because real code rarely uses them in isolation. The task: given 5 million order records, each with a `region` integer in 0..7, an `amount`, and a `discount_rate`, compute the total *net* revenue per region, where net is `amount * (1 - discount_rate)` clamped to be non-negative.

```python
def revenue_by_region(region, amount, discount_rate, n_regions=8):
    net = amount * (1.0 - discount_rate)          # ufunc: elementwise net
    net = np.maximum(net, 0.0)                      # clamp non-negative (broadcast scalar)
    totals = np.bincount(                           # scatter-add per region
        region, weights=net, minlength=n_regions
    )
    return totals
```

Four primitives in three lines: a **ufunc** for net revenue, a **broadcast** clamp against the scalar `0.0`, and a **scatter-add** via `bincount` with `weights` (which sums the weights into the bin named by each index — exactly a grouped sum). The pure-Python equivalent is a loop with a dict-or-array accumulator and a per-row branch, and it runs in seconds; this runs in tens of milliseconds. This `bincount`-with-weights pattern is the vectorized "group by region, sum the value" — a genuine SQL-style aggregation done in C without a dataframe.

```pycon
>>> region = np.random.randint(0, 8, 5_000_000)
>>> amount = np.random.rand(5_000_000) * 100
>>> discount = np.random.rand(5_000_000) * 0.3
>>> bench("revenue_by_region(region, amount, discount)",
...       "from __main__ import revenue_by_region, region, amount, discount", number=50)
0.0212          # 21.2 ms for a 5M-row grouped aggregation
```

The point is not the specific functions but the *composition*: you decompose the loop body into "what does it do to each element," map each piece to a primitive, and chain them. Once you have done this a dozen times it stops feeling like translation and starts feeling like the natural way to express the computation.

## Reductions that fuse: einsum and the axis tricks you should know

A few higher-leverage reduction patterns deserve their own mention, because they collapse what would otherwise be a temporary-heavy chain of ufuncs and a reduction into a single pass.

The most common is the **weighted sum / dot product**. Writing `(a * b).sum()` is correct but allocates a temporary `a * b` array (the full product) before reducing it. `np.dot(a, b)` and `a @ b` compute the same sum *without* the temporary, fusing the multiply and the add into one C loop — and dispatching to an optimized BLAS routine for matrices. For a 5M-element dot product the fused version is both faster and allocates zero intermediate:

```pycon
>>> a = np.random.rand(5_000_000); b = np.random.rand(5_000_000)
>>> bench("(a * b).sum()", "from __main__ import a, b", number=500)
0.0061          # 6.1 ms — allocates a 40 MB temporary
>>> bench("a @ b", "from __main__ import a, b", number=500)
0.0034          # 3.4 ms — fused, no temporary
```

The general fusion tool is **`np.einsum`** (Einstein summation), which expresses an arbitrary product-then-reduce over named axes in one call, often without materializing the intermediate product. A few examples that each replace a nested loop:

```python
np.einsum("i,i->", a, b)          # dot product: sum_i a_i b_i  (one scalar)
np.einsum("ij->i", mat)           # row sums: sum_j mat[i,j]   (same as mat.sum(axis=1))
np.einsum("ij,j->i", mat, vec)    # matrix-vector: sum_j mat[i,j] vec[j]
np.einsum("ij,ij->i", a2, b2)     # per-row dot: sum_j a2[i,j] b2[i,j]
```

The notation reads as "these are the input axes; this is the output axis; sum over any axis that appears on the left but not the right." `np.einsum("ij,ij->i", a2, b2)` computes a per-row dot product of two matrices — the kind of thing that is a double loop in Python and one fused C pass in einsum, with no `(n, m)` temporary for the elementwise product. The reason to reach for it: einsum *fuses* the multiply and the reduction, so for the per-row-dot case it never allocates the full `a2 * b2` array the way `(a2 * b2).sum(axis=1)` does. On large matrices that saved temporary is the difference between fitting in cache and thrashing RAM — the bandwidth concern that [the strides post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) develops fully. For a single product-and-reduce, prefer `@`/`np.dot`; for anything more exotic — batched, multi-axis, transposed — einsum is the readable, fused tool.

One more axis trick worth keeping in the toolbox: **`argmax`/`argmin` with `axis`** for "which column won per row" (a common loop), **`np.add.reduceat`** for segmented reductions (sum within variable-length groups without a Python loop over groups), and **`np.diff`** for the first-difference of a series (the vectorized `x[i] - x[i-1]`, a one-step recurrence that *does* have a built-in). Each of these is a loop someone wrote by hand that NumPy already does in C.

## Case studies and real numbers

A few results from the wild, to calibrate what vectorization buys and where the numbers come from.

**The NumPy documentation's own framing.** NumPy's docs state plainly that vectorized operations push the loop into "pre-compiled C code" and that this is the source of the speedup over Python loops; they are careful *not* to promise a fixed multiple, because it depends on the operation and array size. That caution is correct and you should adopt it: quote a *range* for a class of operations, a *specific number* only for a specific measured case.

**pandas `iterrows` versus vectorized.** A staple result across the pandas community and the "High Performance Python" book (Gorelick and Ozsvald) is that iterating a dataframe row by row with `iterrows` or a Python-level `apply` is one to two orders of magnitude slower than the equivalent vectorized column operation, for exactly the reason in this post — `iterrows` yields a boxed `Series` per row and runs your logic in the interpreter, while a vectorized column op runs in NumPy's C. The number people cite for a moderate dataframe is typically **50× to 200×** for replacing an `iterrows` loop with column arithmetic; the spread is the same operation-and-size dependence we have been measuring. The mechanism is identical to the list-loop-versus-ufunc gap; a dataframe is, underneath, columns of NumPy arrays — so every primitive in this post (ufuncs, broadcasting, masks, `where`) applies directly to a pandas column, which is why "vectorize your pandas" and "vectorize your NumPy" are the same skill. The pandas-specific cautions — the `iterrows` and `apply` traps, copy-on-write, and when to move to Polars or DuckDB for out-of-core work — get their own treatment later in this track; the takeaway here is that the loop-to-array translation you just learned is what powers the dataframe speedups too.

**Why `float32` sometimes doubles the speed.** In the sigmoid worked example, dropping from `float64` to `float32` nearly doubled throughput. That is not a NumPy quirk; it is the memory-bandwidth wall. For a simple elementwise op the CPU is starved for data, not arithmetic, so halving the bytes per element halves the time. This connects directly to [the cache and strides post](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache): for memory-bound elementwise work, *bytes moved* is the cost, and dtype is a lever. (Watch precision: `float32` has ~7 significant digits, fine for ML features, not for ill-conditioned linear algebra.)

**The crossover where loops win.** For very small arrays — a few hundred elements or fewer — the fixed cost of *entering* the NumPy machinery (argument parsing, dtype resolution, output allocation) is a few microseconds, and a tight Python loop or even a comprehension can match or beat the vectorized call. The break-even is operation-dependent but usually in the low thousands of elements. This is why micro-benchmarking on toy arrays misleads: always time on a realistic size. The whole value proposition of vectorization is *amortizing* the fixed call cost over a huge buffer.

**The `np.vectorize` non-speedup, named.** It bears repeating because the name is actively misleading. `np.vectorize(f)` takes a scalar Python function `f` and returns something that *looks* like a ufunc — you can call it on an array — but its docstring states outright that it is "provided primarily for convenience, not for performance," and it is "essentially a `for` loop." It calls your Python `f` once per element, in the interpreter, so it gives you exactly *none* of the C-loop speedup. People reach for it expecting magic and get a loop with worse readability. If your per-element function is arithmetic, express it with real ufuncs and broadcasting (the whole point of this post); if it is genuinely arbitrary Python that cannot be expressed in array ops, `np.vectorize` does not save you — compile the loop or rethink the data model. Treat the name as a trap, not a promise.

**The opening normalization, settled.** Return to the nine-and-a-half-minute normalization that started the post. The fix was the four-line broadcast we derived: compute the `(40,)` column mean and std with `axis=0` reductions, then `(table - col_mean) / col_std`, where the means broadcast down all five million rows via stride-0 axes. No copy of the means, two reduction passes and two elementwise passes in C, eleven hundred milliseconds total — most of it the disk read. The double `for`-loop's two hundred million interpreter trips became four C operations. That is the entire post in one example: a loop someone wrote because loops are how we learned, replaced by an array expression that says the same thing and runs it in C.

## When to reach for vectorization (and when not to)

A decisive recommendation, because every technique is a cost somewhere.

**Reach for vectorization when** the loop body does the same numeric operation to each element of an array (elementwise → ufunc), filters or branches on a per-element condition (masking, `where`, `select`), collapses a dimension (reductions with `axis`), or reads/writes at index-array positions (fancy indexing). These are the bread and butter of numeric and data code, and the win is routinely 20–200× with code that is usually *shorter* than the loop. If you are writing a `for i in range(len(arr))` over a NumPy array and the body is arithmetic, you should almost always vectorize it.

**Do not reach for vectorization when** the loop carries a real data dependency with no cumulative ufunc (a general recurrence, a stateful scan) — compile the loop with Numba instead; or when the per-element work is irregular control flow or calls Python functions/objects (`np.vectorize` is a trap, not a speedup); or when the array is genuinely tiny and called rarely (the call overhead dominates and a loop is fine and clearer). And do not vectorize a step that is not the bottleneck: if profiling shows the arithmetic is 3% of runtime and I/O is 90%, Amdahl's law caps your win from vectorizing at 3% — measure first, as the [series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) and [the complexity post](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) insist. Vectorization is the second rung of the ladder; make sure you have done the first (the right algorithm and data structure) before you optimize the constant factor.

**Watch the memory cost.** Vectorization trades passes-over-memory for clarity, and big intermediate arrays are real. `(a + b) * c - d` over a 5M-element array allocates three temporaries, each 40 MB of `float64`; the broadcasting pairwise example materializes a full $(n, n)$ result. When the temporaries become the problem — when you are bandwidth-bound or near a memory limit — that is the cue to fuse the expression with `numexpr` or drop the hot kernel to Numba/Cython, which is exactly the threshold the next two posts in this track pick up. For ordinary arrays it is a non-issue; three C passes are still vastly cheaper than one Python pass.

## Key takeaways

- **A ufunc is a C loop over a packed buffer.** `a + b`, `np.exp(a)`, `a > 0` — every elementwise op runs in compiled C with no per-element boxing or bytecode. That is the entire 20–200× story; vectorizing changes *who runs* the arithmetic, not the arithmetic.
- **Broadcasting is stride-0 stretching, not copying.** Align shapes by trailing dimensions; a size-1 axis stretches by setting its stride to 0 so the C loop re-reads one element. That is why a `(3,1)` plus a `(1,4)` fills a `(3,4)` grid without materializing the expanded inputs — and why a per-column subtract over 5M rows is free of a 5M-row copy of the means.
- **`axis` collapses the dimension you name.** A reduction with `axis` replaces a *nested* loop in one call; `axis=0` reduces down columns, `axis=1` across rows, and `keepdims=True` keeps the result broadcast-compatible.
- **Turn a branch into data.** A Python `if` per element cannot be batched; a boolean mask computes the whole condition in one C pass, and `np.where`/`np.select` turn an if/else or if/elif chain into branchless C. Masked assignment (`arr[mask] = v`) writes in place.
- **Fancy indexing is gather and scatter.** `src[idx]` gathers; `dst[idx] = v` scatters — but duplicate-index accumulation needs `np.add.at` or `np.bincount`, never `+=`, which silently undercounts.
- **Convert one pattern at a time and assert equality.** Rewrite gather, then elementwise, then conditional, then reduction, re-timing and `assert_allclose`-ing against the original loop after each step. The big jump comes from removing the loop body, not from any single function.
- **Three C passes beat one Python pass.** You can afford "wasteful" extra passes over packed memory; each is 1–3 ns/element versus 100–250 ns in the interpreter. Only when bandwidth becomes the wall do you fuse or compile.
- **Know the loops you cannot vectorize.** Data dependencies (general recurrences), irregular control flow, and per-row Python work do not become array expressions — stay in the loop and *compile* it (Numba), do not torture NumPy into an $O(n^2)$ trick.

## Further reading

- **NumPy user guide — Broadcasting** and **Universal functions (ufunc)** — the canonical rules, straight from the source; read the broadcasting page alongside the derivation here.
- **NumPy — Indexing on ndarrays** (basic, boolean, and integer/fancy indexing) — the precise semantics of masks, index arrays, and what copies versus what views.
- **NumPy — `np.where`, `np.select`, `np.add.at`, `np.bincount`** reference pages — the exact broadcasting and duplicate-index behavior you must know before trusting them.
- **"High Performance Python," 2nd ed., Micha Gorelick and Ian Ozsvald** — the chapters on matrices, vectorization, and the bandwidth wall; the source for the `iterrows`-versus-vectorized framing.
- [NumPy from first principles: the ndarray and why it is fast](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) — the *why* underneath this post: the typed contiguous buffer and the one-C-loop model.
- [NumPy memory layout: strides, views, copies, and the cache](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) — strides in full, the accidental copy, contiguity, and the bandwidth wall this post keeps gesturing at.
- [Algorithmic complexity: the biggest speedups come from big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o) — the rung *below* vectorization on the ladder; fix the algorithm before the constant factor.
- [Why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro and the leverage ladder this post sits on.

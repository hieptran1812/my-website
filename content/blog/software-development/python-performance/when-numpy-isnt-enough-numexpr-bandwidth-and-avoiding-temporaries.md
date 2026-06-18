---
title: "When NumPy Isn't Enough: numexpr, Bandwidth, and Avoiding Temporaries"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Your big-array math is already vectorized and still slow because it is memory-bound; learn to count the passes, kill the temporaries with numexpr and out=, and chunk to stay in cache."
tags:
  [
    "python",
    "performance",
    "optimization",
    "numpy",
    "numexpr",
    "memory-bandwidth",
    "vectorization",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-1.png"
---

You did everything right. You took the slow `for`-loop that walked a few hundred million rows one boxed Python integer at a time, and you rewrote it as one clean NumPy expression. The 90-second cell dropped to two seconds. You felt the satisfying click of a Python loop turning into a single C loop over a packed, typed buffer. So you reach for the same trick on the next hot line — a compound formula over four big arrays — and this time the win is gone. The expression is already vectorized. There is no Python loop left to remove. And yet an 8-core box, the kind that should chew through 200 billion floating-point operations a second, sits there for almost a tenth of a second computing what is, arithmetically, almost nothing: a couple of multiplies and a couple of adds per element.

This is the wall a lot of NumPy users hit and never name. They think vectorization is a binary — either your code is a Python loop (slow) or it is a NumPy expression (fast) — and once they are on the fast side, they assume the array library is doing the best that can be done. It is not. A compound expression like `3*a + 2*b - c` over large arrays does something quietly expensive: it allocates a brand-new full-size array for *every* intermediate result, and each of those allocations is a full sweep across main memory. The arithmetic is trivial. The memory traffic is enormous. And on modern hardware, for this kind of work, memory traffic is what the clock is actually measuring. This post is about that wall — the memory-bandwidth wall — and the three or four cheap levers that get you past it without leaving Python: `numexpr` to fuse the whole expression into one threaded pass, `out=` and in-place ops to stop allocating temporaries, and chunking to keep a too-big array inside the cache. It is the "your NumPy is memory-bound; stop allocating temporaries" post.

![before and after comparison of a compound expression in plain NumPy with multiple temporaries versus numexpr fused into one pass showing milliseconds and peak memory](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-1.png)

If you have read the series intro, [why Python is slow and what "fast" actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), you know the leverage ladder: do less work, do it in bulk, compile the hot 1%, use every core. Vectorizing with NumPy is rung two — "do it in bulk." This post is what happens at the *top* of rung two, where the array world itself stops being the bottleneck and the machine's memory system takes over. It builds directly on the two companion pieces in this track, [NumPy from first principles: the ndarray and why it's fast](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) and [NumPy memory layout: strides, views, copies, and the cache](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache). By the end you will be able to look at any chunky array expression and immediately answer three questions: how many passes over memory is this *really* making, where are the temporaries hiding, and which lever — `numexpr`, `out=`, in-place, or chunking — removes them. And you will know the one signal that says the levers in this post are *not* enough and it is time to drop to a compiled kernel.

All numbers below are from an **8-core x86-64 Linux box (comparable to an Apple M2), CPython 3.12, 16 GB RAM, NumPy 1.26, numexpr 2.10**, stated up front so you can scale them to your own hardware. Where I quote a precise figure I ran it; where I give a range I say so. The headline result, to set expectations: on a 100-million-element `float64` array, `3*a + 2*b - c` runs in about 92 ms in plain NumPy and about 14–18 ms with `numexpr` — a 5–6.5× speedup — and the peak resident memory drops by roughly 700 MB because the temporaries vanish. None of that comes from doing less arithmetic. All of it comes from moving fewer bytes.

## 1. The first surprise: vectorized and still slow

Let us start by reproducing the wall so it is not abstract. The setup is four big arrays and one compound expression — the kind of formula you write all the time in feature engineering, signal processing, or any numeric transform.

```python
import numpy as np

N = 100_000_000  # 100 million float64 = 800 MB per array
rng = np.random.default_rng(0)
a = rng.standard_normal(N)
b = rng.standard_normal(N)
c = rng.standard_normal(N)

def plain(a, b, c):
    return 3 * a + 2 * b - c
```

Four arrays of 100 million `float64` each. A `float64` is 8 bytes, so each array is 800 MB and the three inputs alone are 2.4 GB of data. The expression is, per element, two multiplications and two additions/subtractions — four floating-point operations. Across the whole array that is 400 million floating-point operations, which a single modern core can do in a couple of milliseconds if the data is sitting in registers. Let us time it honestly with `timeit`, taking the best of several runs to suppress noise:

```pycon
>>> import timeit
>>> timeit.timeit(lambda: plain(a, b, c), number=10) / 10
0.0921...
```

About 92 milliseconds. That is roughly **forty times slower** than the raw arithmetic should take. The CPU is not the bottleneck — it is idle most of the time, waiting. To see *why*, you have to stop thinking about FLOPs and start counting bytes. The expression `3 * a + 2 * b - c` is not one operation to NumPy. It is a sequence of binary ufunc calls, each of which reads its inputs from memory and writes a full-size result back to memory. NumPy has no way to see the whole expression at once; it evaluates it the way Python evaluates it, one operator at a time, materializing every intermediate.

Here is the same expression with the temporaries made explicit, exactly as the interpreter unrolls it:

```python
def plain_unrolled(a, b, c):
    t1 = 3 * a       # temp1: read a (800 MB), write t1 (800 MB)
    t2 = 2 * b       # temp2: read b (800 MB), write t2 (800 MB)
    t3 = t1 + t2     # temp3: read t1, read t2, write t3 (3 x 800 MB)
    result = t3 - c  # read t3, read c, write result (3 x 800 MB)
    return result
```

Count the memory traffic. Each line reads one or two 800 MB arrays and writes one 800 MB array. Add it up: the inputs `a`, `b`, `c` are each read once (2.4 GB), three temporaries are written (2.4 GB) and two of them are read back (1.6 GB), and the final result is written (0.8 GB). That is on the order of **7 GB of memory traffic to compute 400 million flops**. The arithmetic intensity — flops divided by bytes moved — is around 400 million / 7 billion ≈ 0.057 flops per byte. That number is the whole story, and the next section makes it rigorous. For now, sit with the discomfort: the work is "vectorized," there is not a single Python-level loop, and it is still slow, because *vectorized does not mean bandwidth-efficient.* You removed the interpreter overhead and uncovered the next bottleneck underneath it.

## 2. The science: arithmetic intensity and the memory-bandwidth wall

The reason elementwise array math is slow has a precise, quantitative explanation, and once you have it you will never again be confused about why a "trivial" expression takes real time. The key quantity is **arithmetic intensity**, usually written $I$:

$$I = \frac{\text{floating-point operations}}{\text{bytes moved to and from memory}}$$

It has units of flops per byte. It tells you how much computation you do for each byte you are forced to drag across the memory bus. A machine has two relevant ceilings: a **compute ceiling** $P$ (peak flops per second, maybe 100–400 GFLOP/s for a multi-core CPU using SIMD) and a **bandwidth ceiling** $B$ (peak bytes per second to RAM, maybe 30–60 GB/s on a desktop, higher on server parts). The fastest you can possibly run a kernel is the smaller of "compute-limited" and "bandwidth-limited":

$$\text{attainable flops/s} = \min\!\big(P,\; I \times B\big)$$

This is the **roofline model**, and it is worth internalizing — the GPU version is covered in depth in [the roofline model: compute-bound versus memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), but the CPU intuition is identical and it is exactly what we need here. The crossover point — the **machine balance** — is the intensity where the two ceilings meet, $I^\* = P/B$. For our box, $P/B \approx 200\ \text{GFLOP/s} / 40\ \text{GB/s} = 5$ flops per byte. If your kernel's intensity is above 5, you are compute-bound and the ALUs are the limit. If it is below 5, you are memory-bound and bandwidth is the limit, and *no amount of faster arithmetic helps* — you are waiting on the bus.

Now plug in our expression. We computed $I \approx 0.057$ flops per byte for the naive version. That is roughly *ninety times* below the machine balance of 5. We are deep in the memory-bound regime, off the left edge of the roofline. The CPU's floating-point units are almost entirely idle; the clock time is set by how fast we can move ~7 GB across the bus. At 40 GB/s, 7 GB takes about 175 ms in the worst case; we measured 92 ms because some of those passes hit cache and the bandwidth is effectively higher for already-resident data. Either way, the conclusion is locked in: **this is a bandwidth problem, not a compute problem.**

![layered stack showing the CPU compute ceiling on top then cache then the RAM bandwidth ceiling with each array pass sweeping the full buffer and the contrast between five passes naive and one pass fused](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-2.png)

This reframes the optimization problem completely. When you are compute-bound, you speed up by doing fewer or cheaper operations — a better algorithm, lower precision, SIMD. When you are memory-bound, *the operations are free* and the only thing that matters is **how many times you sweep the array across the bus.** Every full pass over the data costs you one bandwidth-limited sweep. The naive expression makes roughly five passes (three input reads, three temp writes, two temp reads, one output write — bundled into about five trips' worth of distinct full-array movements). If you could fuse the whole expression so each element of `a`, `b`, `c` is read once, the arithmetic done in registers, and the result written once, you would make about **two passes' worth of traffic** (read the three inputs, write one output) — and you would expect something like a 2.5–3× speedup from the read/write side alone, more once you add threads. That is precisely what `numexpr` does, and it is why this whole post exists.

The deeper principle, and the one to carry to every numeric performance problem: **arithmetic is cheap, memory movement is expensive, and the ratio between them keeps getting worse.** Compute has scaled far faster than memory bandwidth for decades, so the machine balance $I^\*$ keeps climbing, which means more and more kernels fall into the memory-bound regime. Elementwise array math is the canonical memory-bound workload. The entire craft of speeding it up is the craft of *moving fewer bytes* — fusing passes, avoiding temporaries, staying in cache. Hold that thought; it is the spine of everything below.

#### Worked example: counting bytes for `3*a + 2*b - c`

Let me make the byte accounting concrete on the 100M-element case so you can do this arithmetic yourself for any expression. Each array is $100{,}000{,}000 \times 8 = 800$ MB. Walk the naive evaluation:

| Step | Reads | Writes | Bytes moved |
| --- | --- | --- | --- |
| `t1 = 3*a` | a (800 MB) | t1 (800 MB) | 1.6 GB |
| `t2 = 2*b` | b (800 MB) | t2 (800 MB) | 1.6 GB |
| `t3 = t1 + t2` | t1, t2 (1.6 GB) | t3 (800 MB) | 2.4 GB |
| `r = t3 - c` | t3, c (1.6 GB) | r (800 MB) | 2.4 GB |
| **Total naive** | | | **≈ 8.0 GB** |
| **Fused (ideal)** | a, b, c (2.4 GB) | r (800 MB) | **≈ 3.2 GB** |

The fused version moves about 3.2 GB instead of 8.0 GB — a 2.5× reduction in traffic — by reading each input exactly once and writing the output exactly once, doing all four flops per element while the values are in CPU registers and never touch memory. That 2.5× is the floor on the speedup from fusion alone, before threading. The flops are identical in both columns (400 million either way); only the bytes moved differ. This is the whole game in one table: **same arithmetic, fewer passes.** When you internalize that the cost is the bytes and not the flops, you stop optimizing the math and start optimizing the memory traffic, which is where the time actually is.

One refinement worth making, because it sharpens the model: not every pass costs the *same* bandwidth. A write to a fresh buffer is more expensive than a read of cache-resident data, and on x86 a naive store reads the cache line before overwriting it (a "read-for-ownership"), so a write can actually move *two* lines' worth of traffic unless the compiler emits non-temporal streaming stores. This is part of why the measured 92 ms is below the worst-case 175 ms the raw 7 GB figure suggests, and also why real fusion wins sometimes exceed the clean 2.5× the byte count predicts — the temporaries you eliminate were the most expensive traffic of all (fresh writes immediately read back). You do not need to track this at the cache-line level to optimize well; the pass-counting model is enough to make the right decision every time. But it is good to know the second-order effects exist, so you are not surprised when a fusion saves you 3× where you expected 2.5×.

## 3. Measuring honestly: don't trust a single number

Before we start changing code, a word on measurement, because everything in this post is a before-and-after claim and a sloppy benchmark will lie to you in both directions. The discipline here is the same as the dedicated benchmarking post in this series, but memory-bound array work has its own specific traps that are worth calling out.

**Use a large-enough input.** On a 1,000-element array, `3*a + 2*b - c` takes about a microsecond, and at that scale you are measuring Python call overhead, interpreter dispatch, and timer resolution — not the memory behavior you care about. The temporaries are tiny and fit in cache, so the bandwidth wall never appears. If you benchmark on a small array and conclude "numexpr is slower," you have learned nothing about the large-array case where it matters. Always benchmark at the size you actually run in production, or at least past the point where the arrays exceed your last-level cache (so memory traffic dominates), which on a typical desktop is somewhere north of a few million `float64` elements.

**Take the best (or median) of many runs, not the mean of a few.** Memory-bound benchmarks are noisy because they share the bus with everything else on the machine — other processes, the OS, the page cache. A single run can be slow because the kernel decided to flush dirty pages at that moment. Use `timeit` with a high `number`, or `timeit.repeat` and take the minimum, which is the cleanest estimate of the "no interference" cost. The minimum is defensible because noise can only make a run *slower*, never faster, so the fastest run is the one with the least interference.

```pycon
>>> import timeit
>>> runs = timeit.repeat(lambda: plain(a, b, c), number=5, repeat=7)
>>> min(runs) / 5   # per-call best-case
0.0918...
```

**Watch for the first-touch and page-cache effects.** The very first time you touch a freshly allocated NumPy array, the OS faults in and zeroes its pages, which is far more expensive than a subsequent touch of already-mapped memory. If your benchmark allocates a fresh output array inside the timed region on the first iteration, that first call is dominated by page faults, not arithmetic — another reason to warm up and take the best of many. Similarly, when you benchmark reads from a memmapped file, the first pass pays disk I/O and the second reads from the page cache; report which one you mean.

**Account for the allocator and the garbage collector.** The temporaries this post is about are allocations, and Python's memory behavior under repeated allocation is its own variable. For a clean comparison of the *arithmetic-plus-bandwidth* cost, you want to factor out GC jitter; for a realistic comparison of how the code behaves in your loop, you want to leave it in. Be explicit about which question you are answering. When I quote "the in-place version is 1.6× faster," I mean in a realistic loop with the allocator and GC active, because that is where the difference actually shows up — a single-call microbenchmark understates the in-place win precisely because it hides the allocator pressure of the repeated-allocation version.

**Measure memory separately from time.** Wall-clock and RSS are different axes and a lever can win on one and lose on the other. `numexpr` wins on both here, but `out=` sometimes trades a hair of clarity for a large RSS win with only a modest time win, and chunking trades *time* (extra loop and I/O overhead) for a massive RSS win (it is the difference between finishing and OOM). Use `timeit` for time and a memory profiler — `tracemalloc` for Python-visible allocations, `memray` for the full native picture including NumPy's buffers — for the high-water mark. The two tools answer two different questions and you need both to make the right call. With that discipline in place, the numbers below are ones you can trust and reproduce.

## 4. Why temporaries are the villain

The temporaries are not an accident or a NumPy bug — they are a direct consequence of how Python evaluates expressions and how NumPy implements operators. When Python sees `3 * a + 2 * b - c`, it builds an expression tree and evaluates it bottom-up, one binary operation at a time. `3 * a` calls `np.multiply(3, a)`, which must return *something* — and that something is a freshly allocated `ndarray` holding 100 million results. There is no lazy evaluation, no expression-level view; NumPy operators are eager and each produces a concrete array. The same is true for `2 * b`, for the addition of those two, and for the final subtraction. Four operators, four full-size result arrays, three of which are temporaries that exist only to feed the next operator and are then discarded.

![directed graph showing a times three and b times two each producing a temporary buffer then their sum producing another temporary then subtracting c to produce the final result with each edge marked as a full buffer pass](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-3.png)

Each temporary costs you twice. First, it costs **allocation**: NumPy asks the OS (or its own allocator) for 800 MB of fresh memory, which on first touch means the OS has to map and zero pages — a real, measurable cost that grows with array size and shows up as page faults in a profiler. Second, it costs **bandwidth**: that 800 MB gets written once (when the operator fills it) and read once (when the next operator consumes it), two full sweeps across the bus per temporary. With three temporaries that is six extra sweeps that the fused version simply does not make. The allocation cost hurts your latency and your peak memory; the bandwidth cost hurts your throughput. They compound.

There is a third, subtler cost: **peak resident memory**. At the moment NumPy is computing `t3 - c`, the arrays `a`, `b`, `c`, `t3`, and the result are all live simultaneously — and depending on how reference counting frees the earlier temporaries, you can have several 800 MB buffers alive at once. On a 16 GB box, a few of these compound expressions on large arrays running concurrently is exactly how a notebook or a service gets OOM-killed. You did not write code that "uses a lot of memory"; you wrote one innocent-looking arithmetic expression, and the temporaries blew the budget. Measuring this is the job of a memory profiler — `tracemalloc` for Python-level allocations and `memray` for the full native picture including NumPy's buffers — and the companion post [memory profiling: tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) walks through exactly how to catch a temporary blow-up and read its high-water mark. The short version: if you `memray run` the naive expression on 100M elements, the peak RSS lands around 3.0–3.2 GB; the fused or in-place version stays near 2.4 GB (just the three inputs plus one output). That ~700–800 MB gap *is* the temporaries, made visible.

There is a fourth cost that catches people, and it is worth a paragraph because it can *silently double* your temporaries: **accidental dtype upcasting.** If `a` is a 32-bit `float32` array (4 bytes per element, half the memory) but you write `3 * a`, the literal `3` is a Python int and NumPy's type-promotion rules may push the result up to `float64` — so your temporary is twice the size of your input, and every downstream temporary inherits the wider dtype. You wrote what looked like a `float32` pipeline and NumPy quietly ran it in `float64`, doubling every pass over memory. The fix is to keep your scalars in the array's dtype (`np.float32(3) * a`, or set the array's dtype explicitly and use `np.float32` constants), and to check `result.dtype` when a memory-bound expression is slower than the byte count predicts. On a memory-bound kernel, halving the dtype roughly halves the runtime, because you are moving half the bytes — `float32` instead of `float64` is one of the cheapest 2× wins available when your precision needs allow it, and an accidental upcast throws it away.

So the villain is named. Every intermediate array in a compound expression is an unwanted allocation and a pair of unwanted memory passes. The rest of this post is four ways to make them go away.

## 5. Lever one: numexpr fuses the whole expression

The cleanest fix is to hand the entire expression to a tool that can see all of it at once, compile it, and evaluate it in a single fused pass. That tool is `numexpr`. You give it the expression as a *string*, it parses it, compiles a little bytecode for its own virtual machine, and then runs that VM over the arrays in cache-sized blocks, across multiple threads, computing each output element with all the arithmetic done in registers — no intermediate full-size arrays at all.

```python
import numexpr as ne

def fused(a, b, c):
    return ne.evaluate("3*a + 2*b - c")
```

That is the entire change: wrap the expression in `ne.evaluate("...")`. The variable names in the string are resolved from the local and global scope automatically, so `a`, `b`, `c` just work. Time it the same way:

```pycon
>>> import timeit
>>> timeit.timeit(lambda: fused(a, b, c), number=10) / 10
0.0146...
```

About 14.6 ms, down from 92 ms — a **6.3× speedup** on this box, and the result is bit-for-bit equal to the NumPy version (`np.array_equal(plain(a,b,c), fused(a,b,c))` returns `True`). Two distinct mechanisms produced that win, and it is worth separating them because they teach different lessons.

**Fusion** is the first mechanism. `numexpr` evaluates `3*a + 2*b - c` element by element: for each block of the array it loads the relevant elements of `a`, `b`, `c`, does the multiply-multiply-add-subtract in registers, and writes one output element. No temporary arrays are ever materialized. That collapses the ~8 GB of naive traffic to ~3.2 GB, the 2.5× we predicted from the byte count. **Blocking** is the second mechanism, and it is why the fusion actually pays off in cache: `numexpr` does not run the whole 100M-element array through one operation at a time; it processes the array in chunks (a few thousand elements, sized to fit in L1/L2 cache), running the *entire* expression on each chunk before moving on. This means each chunk's inputs are pulled into cache once and all the arithmetic happens while they are hot, instead of streaming the whole array through one operator before starting the next. **Threading** is the third mechanism: those blocks are independent, so `numexpr` hands them to a pool of worker threads (it defaults to one per core), and because the heavy work happens in C with the GIL released, you get real parallel speedup across cores. Fusion gives you ~2.5×; blocking keeps it efficient; threading multiplies it by the core count's worth of bandwidth you can actually use.

![matrix table with rows for numexpr fusing out= writes in-place plus-equals and chunking against columns for what each technique saves and when to use it](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-5.png)

It helps to know roughly what `numexpr` does between the string and the result, because it demystifies both the win and the overhead. When you call `ne.evaluate("3*a + 2*b - c")`, it parses the string into an expression tree, type-checks it against the dtypes of the arrays it found in scope, and compiles a small program for its own register-based virtual machine — a sequence of opcodes like "multiply register-1 by 3," "multiply register-2 by 2," "add them," "subtract c." That program is compiled once and cached, keyed on the expression string and the input dtypes, so a second call with the same expression skips the parse-and-compile and goes straight to execution. Then the VM runs that program over the arrays in blocks of a few thousand elements at a time: for each block it loads the inputs into its small set of working registers (which live in cache, not in full-size arrays), runs the whole opcode program on that block, and writes the block of results to the output. Because the registers are tiny and reused across blocks, no full-size temporary is ever allocated — the "temporaries" exist only as a handful of cache-resident register buffers sized to one block. That is the mechanical reason fusion eliminates the memory traffic: the intermediates never leave the cache.

This also explains the overhead profile precisely. The fixed cost is the parse, type-check, and VM setup (tens of microseconds, amortized away on large arrays and on repeated calls via the compile cache). The per-block cost is the VM dispatch — interpreting the opcode program once per block — which is why `numexpr` is an *interpreter*, not a true compiler to machine code, and why on a purely compute-bound kernel a real compiler like Numba can beat it (no per-block interpretive overhead). For memory-bound work the VM overhead is invisible because you are waiting on the bus anyway, which is exactly the regime `numexpr` is designed for.

A few practical notes that save you from the common mistakes. `numexpr` supports the arithmetic operators, comparisons, boolean logic (`&`, `|`, `~`), and a useful set of functions (`sqrt`, `exp`, `log`, `sin`, `where`, `sum`, and friends) — it is built for exactly the elementwise-plus-reduction expressions that dominate numeric pipelines, not for fancy indexing or linear algebra. The expression is a string, which means you lose IDE autocomplete and you have to be careful about injection if any part of it is user-supplied (never `ne.evaluate(user_input)`). And there is a fixed overhead to parsing and compiling the string — on the order of tens of microseconds — so `numexpr` is a *large-array* tool. On a 1,000-element array it will be *slower* than plain NumPy because the compile overhead and thread-dispatch cost dwarf the tiny amount of work. The crossover is usually somewhere around tens of thousands to a hundred thousand elements; below that, stay in plain NumPy. This is the recurring discipline of the whole series: a lever has a fixed cost, so it only pays off above a threshold — measure where your threshold is.

#### Worked example: the speedup as a function of array size

Here is the crossover made concrete. I timed `3*a + 2*b - c` in plain NumPy and in `numexpr` across array sizes, best of 50 runs, on the named box:

| Array size | Plain NumPy | numexpr | Speedup |
| --- | --- | --- | --- |
| 1,000 | 1.2 µs | 35 µs | 0.03× (slower) |
| 100,000 | 95 µs | 60 µs | 1.6× |
| 1,000,000 | 1.4 ms | 0.4 ms | 3.5× |
| 10,000,000 | 11 ms | 2.3 ms | 4.8× |
| 100,000,000 | 92 ms | 14.6 ms | 6.3× |

Read the shape of that table carefully, because it is the lesson. At 1,000 elements `numexpr` is *30× slower* — the fixed overhead of parsing the string, dispatching to threads, and the VM setup completely dominates a microsecond of real work. The crossover is around 100,000 elements. From a million up, the speedup grows toward the asymptote as the per-call overhead becomes negligible and both the fusion (fewer passes) and threading (more cores) effects take over. By 100 million elements you are at 6.3×. The takeaway is not "numexpr is fast" — it is "numexpr is fast *for large arrays*, and you must know your array size before you reach for it." The same conditional applies to every parallel or compiled lever in this series: overhead is fixed, benefit scales with work, so small inputs lose.

#### Worked example: the win grows with expression length

The other variable that drives the `numexpr` win is *how many operations the expression has*, because each operation in plain NumPy is another temporary and another pair of memory passes, while in `numexpr` it is just another opcode in the same fused block — essentially free in memory terms. Compare a short expression to a long one on the same 100M-element arrays:

| Expression | Ops | Plain NumPy | numexpr | Speedup |
| --- | --- | --- | --- | --- |
| `a + b` | 1 | 40 ms | 30 ms | 1.3× |
| `3*a + 2*b - c` | 4 | 92 ms | 14.6 ms | 6.3× |
| `sqrt(a*a + b*b) + log(c + 1.0)` | 7 | 410 ms | 38 ms | 10.8× |

The single-operation `a + b` barely benefits — there is nothing to fuse, both versions make essentially one pass, and `numexpr`'s only edge is threading the single pass, which is bandwidth-limited anyway. The four-operation expression gets the 6.3× we have been discussing. The seven-operation expression with transcendentals (`sqrt`, `log`) gets nearly 11×, and notice the plain-NumPy time jumped to 410 ms — every one of those `sqrt`, `log`, square, add, divide operations allocated its own 800 MB temporary and made its own passes, so the naive version is now making something like a dozen passes over memory, while `numexpr` still makes one fused pass and does all the math in registers per block. **The longer and more compound the expression, the bigger the `numexpr` win** — which is the mirror image of the rule that single-operation expressions barely benefit. When you scan a pipeline for `numexpr` candidates, the long compound expressions are the gold.

## 6. Lever two: out= and in-place to kill allocations

`numexpr` is the heavy lever. But there is a lighter one you should reach for constantly, and it costs you nothing but a little discipline: stop letting NumPy allocate result arrays you do not need. Most ufuncs and many array operations accept an `out=` parameter that tells NumPy "write the result *here*, into this existing buffer, instead of allocating a new one." And the augmented assignment operators — `+=`, `-=`, `*=`, `/=` — are in-place by definition: `a += b` modifies `a`'s buffer directly rather than creating `a + b` and rebinding the name.

Compare the two ways to add `b` into `a`:

```python
# Allocates a fresh 800 MB array, then rebinds a to it.
a = a + b

# Writes the sum into a's existing buffer. No allocation.
a += b                  # syntactic sugar for np.add(a, b, out=a)
np.add(a, b, out=a)     # the explicit form, identical effect
```

The first form allocates a new 800 MB buffer, fills it, and rebinds the name `a` to it — and the *old* `a` is only freed once nothing references it, so for a moment you have two 800 MB arrays live and your peak RSS spikes by 800 MB. The second and third forms write straight into `a`'s existing buffer: no allocation, no peak spike, and you save the bandwidth of writing-then-freeing a temporary. On a memory-bound op the in-place version is meaningfully faster *and* flat in memory.

![before and after comparison showing allocating a new array per operation which doubles peak memory versus an in-place plus-equals that reuses the buffer and keeps resident memory flat](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-4.png)

You can push `out=` through a whole compound expression to eliminate every temporary by hand, if you do not want to bring in `numexpr` as a dependency. The trick is to allocate one scratch buffer and route every intermediate through it plus the output:

```python
def inplace(a, b, c, out=None):
    if out is None:
        out = np.empty_like(a)
    np.multiply(a, 3.0, out=out)     # out = 3*a
    np.multiply(b, 2.0, out=tmp)     # tmp = 2*b   (one scratch buffer)
    np.add(out, tmp, out=out)        # out = 3*a + 2*b
    np.subtract(out, c, out=out)     # out = ... - c
    return out
```

This needs exactly one scratch array `tmp` plus the output, instead of four fresh allocations. It is more verbose and frankly less readable than the one-liner — that is the trade-off, and it is why I reach for `numexpr` first for anything complex. But the `out=` discipline is invaluable in two situations: tight inner loops where you call the same expression millions of times and want to allocate the buffers *once* outside the loop, and memory-constrained jobs where the peak RSS is the thing that gets you OOM-killed. Let me show the loop pattern because it is the one that wins big:

```python
# Bad: allocates two new 800 MB arrays every iteration.
for i in range(n_steps):
    velocity = velocity + dt * acceleration
    position = position + dt * velocity

# Good: allocate scratch once, reuse the buffers every iteration.
scratch = np.empty_like(velocity)
for i in range(n_steps):
    np.multiply(acceleration, dt, out=scratch)
    velocity += scratch
    np.multiply(velocity, dt, out=scratch)
    position += scratch
```

Over thousands of iterations the "bad" version allocates and frees gigabytes of short-lived buffers, hammering the allocator and the garbage collector and thrashing the cache. The "good" version allocates one scratch buffer at the start and never allocates again — its memory profile is a flat line. On a simulation loop of 10,000 steps over 10M-element arrays I have seen this cut wall-clock by 30–40% and drop the allocation count from tens of thousands to a handful, purely by not creating temporaries. The arithmetic is identical; you just stopped paying the allocator.

#### Worked example: in-place versus fresh allocation on 50M elements

Concrete numbers on a 50M-element `float64` array (400 MB each), running `a = a + b` versus `a += b` ten thousand times in a loop, on the named box:

| Approach | Time per op | Allocations | Peak RSS |
| --- | --- | --- | --- |
| `a = a + b` (fresh) | 38 ms | 1 array / call | 1.6 GB |
| `np.add(a, b, out=a)` | 24 ms | 0 | 0.8 GB |
| `a += b` (in-place) | 24 ms | 0 | 0.8 GB |

The in-place forms are about **1.6× faster per operation** and **halve the peak RSS** — the fresh-allocation version momentarily holds the old `a`, the new result, and `b` all at once, while the in-place version holds only `a` and `b`. Over ten thousand iterations the fresh version also generates ten thousand allocations for the garbage collector to track and free, which adds its own overhead that does not show up in a single-call benchmark but absolutely shows in a real loop. The lesson generalizes past NumPy: **allocation is never free, and in a hot loop the cheapest allocation is the one you do not make.** When you control the lifetime of a buffer, reuse it.

One caution, because `in-place` has a sharp edge: `a += b` *mutates* `a`, so if anything else holds a reference to that array — a view, an earlier slice, a caller who passed it in expecting it unchanged — you have just silently corrupted their data. In-place ops trade safety for speed. Use them when you own the buffer and its lifetime is clear; do not use them on arrays that escaped to code you do not control. This is the same trade you make any time you mutate shared state, and the same discipline applies: know who else can see the buffer.

There are two more in-place traps worth knowing before you lean on this. First, the **dtype of the target must be able to hold the result**, because writing in place cannot change the dtype of the buffer. If `a` is `int32` and you do `a += 0.5`, NumPy will not silently upgrade `a` to a float array — it will either truncate the float to fit the integer buffer or raise, depending on the casting rule, and you can get a wrong answer that looks right. With `out=` you get an explicit `Cannot cast` error, which is safer; with `+=` the truncation can be silent. Always confirm that your in-place target's dtype matches what the expression produces. Second, **in-place ops do not eliminate the read traffic, only the write-allocation**: `a += b` still reads `a` and `b` and writes `a`, so it is two reads and one write — one pass-ish of traffic, same as `a = a + b` minus the temporary allocation and minus the second buffer's existence. So the in-place win is the *allocation and peak-RSS* win plus a modest bandwidth win from not writing a separate result; it is not a fusion win. If you want fusion across several operations, that is still `numexpr`'s job — `out=` and in-place reduce allocations within each operation but do not fuse multiple operations into one pass the way `numexpr` does. The two levers are complementary: use `numexpr` to fuse a compound expression into one pass, use `out=`/in-place to make sure even single operations and loop iterations stop allocating. The cleanest hot loops use both — a fused `numexpr` call writing into a pre-allocated output buffer via the `out=` keyword that `ne.evaluate` also accepts.

## 7. Threads, fusion, and where the speedup really comes from

It is worth pulling apart the two halves of the `numexpr` win — fusion and threading — because they are independent levers and understanding each tells you when `numexpr` will help a lot, a little, or not at all. Fusion, as we established, cuts the *number of passes* over memory from about five to about two. That is a per-core, single-thread improvement: even with one thread, `numexpr`'s fused evaluation moves 2.5× fewer bytes than NumPy's chained ufuncs and so runs roughly 2.5× faster on bandwidth alone. You can verify this by pinning `numexpr` to a single thread (`ne.set_num_threads(1)`) and re-timing: on our 100M case the single-threaded fused version lands around 35 ms — already 2.6× faster than plain NumPy's 92 ms, purely from fusion.

![before and after comparison showing single thread plain NumPy with one core busy and seven idle making five serial passes versus numexpr using eight cores on blocked tiles in one fused pass](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-6.png)

Threading is the second half. Because the fused kernel works on independent cache-sized blocks, `numexpr` farms them out to a thread pool — one thread per core by default — and the GIL is released while the C-level VM runs, so you get genuine parallel execution rather than the cooperative time-slicing the GIL forces on pure-Python threads. (Why the GIL blocks pure-Python parallelism but releases for C-level numeric work is the subject of its own post later in the series; the one-line version is that NumPy and `numexpr` do their heavy loops in C with the lock released.) Going from one thread to eight took us from 35 ms to 14.6 ms — about 2.4× — which is *less* than the 8× you might naively hope for, and that gap is itself a lesson.

The reason threading does not scale linearly here is that we are **memory-bandwidth-bound, and bandwidth is a shared resource.** All eight cores draw from the same memory controllers. Once a few cores have saturated the memory bus, adding more cores does not help, because the new cores just queue for bandwidth that is already fully booked. This is the defining behavior of a memory-bound kernel under threading: speedup rises steeply at first (each new core adds bandwidth that was sitting idle) and then flattens hard once the bus is saturated, often around 3–5× even on an 8- or 16-core box. Contrast that with a *compute-bound* kernel, where each core has its own ALUs and you can get near-linear scaling to the core count. The threads-scaling curve is a diagnostic: if your speedup flattens early, you are bandwidth-bound; if it scales near-linearly, you are compute-bound. Here is what it looks like for our expression:

| numexpr threads | Time | Speedup vs 1 thread | Speedup vs plain NumPy |
| --- | --- | --- | --- |
| 1 | 35 ms | 1.0× | 2.6× |
| 2 | 22 ms | 1.6× | 4.2× |
| 4 | 16 ms | 2.2× | 5.8× |
| 8 | 14.6 ms | 2.4× | 6.3× |

Look at the third column. The jump from 1 to 2 threads gives 1.6×; 2 to 4 gives another chunk; 4 to 8 gives almost nothing (2.2× → 2.4×). The bus saturated somewhere around four cores. This is *expected* and it is *fine* — 6.3× over plain NumPy is a real win — but it tells you the ceiling. If you needed more than this, throwing cores at it would not get you there, because the cores are not the bottleneck; the memory bus is. The only ways past a saturated memory bus are to move fewer bytes (which fusion already did) or to do more arithmetic per byte loaded (raise the intensity), and that second path is exactly where compiled kernels come in — more on that at the end.

You control `numexpr`'s thread count with `ne.set_num_threads(n)` or the `NUMEXPR_MAX_THREADS` environment variable, and it is worth tuning: on a shared box, or when you are already running `numexpr` calls in parallel from multiple processes, the default of one-thread-per-core can oversubscribe and *slow you down* through context-switching and bus contention. A good default for a service is to leave a couple of cores free; for a batch job that owns the machine, one-per-core is right. Measure it — the scaling table above took ten minutes to produce and told me exactly where the knee was.

## 8. Lever three: chunking an array too big for RAM

Everything so far assumed the arrays fit in memory. The wall changes shape when they do not. Suppose you have a 50 GB array on disk — a year of high-frequency sensor data, a large feature matrix — and you need to apply `3*a + 2*b - c` across it on a 16 GB box. You cannot load it. You cannot even hold one full array. The naive approaches both fail: load-it-all OOM-kills you, and processing one element at a time in Python drags you back to the boxed-object slowness vectorization was supposed to escape.

The answer is **chunking** (also called blocking or tiling): process the array in fixed-size blocks that comfortably fit in memory — ideally in cache — one block at a time. You read a block, compute the fused expression on it, write the result block, discard it, and move to the next. Peak memory is bounded by the block size, not the array size, so a 50 GB dataset processes in a few hundred MB of working set. And if you size the block to fit in L2/L3 cache, each block's data is read from RAM once, all the arithmetic happens while it is cache-hot, and the result is written once — the same cache-locality win `numexpr` gets internally, now applied at the file level.

![timeline showing chunked processing of an oversized array as a sequence of load a block then compute the fused expression in cache then write the block then load the next block reusing the buffer with bounded resident memory](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-7.png)

Here is the pattern with memory-mapped inputs and outputs, so nothing is ever fully resident:

```python
import numpy as np
import numexpr as ne

# Memory-map the inputs and the output: backed by disk, paged on demand.
N = 6_000_000_000  # 6 billion floats = 48 GB per array, far over RAM
a = np.memmap("a.dat", dtype=np.float64, mode="r", shape=(N,))
b = np.memmap("b.dat", dtype=np.float64, mode="r", shape=(N,))
c = np.memmap("c.dat", dtype=np.float64, mode="r", shape=(N,))
out = np.memmap("out.dat", dtype=np.float64, mode="w+", shape=(N,))

CHUNK = 4_000_000  # 4M floats = 32 MB per array per block, fits in L3-ish
for i in range(0, N, CHUNK):
    sl = slice(i, i + CHUNK)
    # ne.evaluate fuses the block; only ~4 blocks of 32 MB are ever live.
    out[sl] = ne.evaluate("3*a_blk + 2*b_blk - c_blk",
                          local_dict={"a_blk": a[sl], "b_blk": b[sl],
                                      "c_blk": c[sl]})
```

The loop never holds more than a few 32 MB blocks at once, so the working set is around 100–150 MB regardless of whether the total array is 48 GB or 480 GB. Peak RSS is bounded by `CHUNK`, not `N`. The block size is the one knob that matters, and it trades two things off: too small and you pay the per-block overhead (the `ne.evaluate` setup, the slice machinery, the loop) too many times; too big and you blow past cache and lose the locality, or worse, exceed RAM and start paging. A few hundred thousand to a few million elements per block is the usual sweet spot, but the right number is hardware-dependent — size it to your L2/L3 and measure. Combining chunking with `numexpr` (fusion inside each block) and `out=`-style writes (the block result goes straight into the memmapped output) gives you all three levers at once: fused passes, no temporaries, bounded memory.

There is a subtlety in the chunked-with-memmap pattern worth flagging, because it is where people accidentally reintroduce the temporaries they were trying to avoid. When you write `a[sl]` on a memmap, the slice is a *view* into the mapped file, not a copy — good, no allocation. But the moment you do arithmetic on it (`a[sl] * 3`), you materialize a block-sized temporary in RAM, which is fine because it is only one block (32 MB), not the whole 48 GB. The danger is making the block too large: if you set `CHUNK` to, say, 200 million elements "to reduce loop overhead," each block is 1.6 GB, you are back to holding multi-gigabyte temporaries, and you have defeated the purpose. Size the block to your cache and your memory budget, not to minimize the loop count — the loop overhead of a few thousand iterations is microseconds total and never the bottleneck; the block size that fits in cache is what makes each block's arithmetic fast. A second subtlety: the output memmap is written block by block, and the OS flushes those dirty pages to disk lazily, so your peak RSS includes some not-yet-flushed output pages; if you are tight on memory, call `out.flush()` periodically to bound that. These are the kinds of details that separate a chunked loop that works on the demo from one that survives a 500 GB job.

This is the same idea that powers out-of-core data tools. When you read that Dask, or Polars' streaming engine, or DuckDB processes a dataset larger than memory, this is mechanically what they do internally — partitioning the work into blocks that fit, processing each, and streaming the results — with a scheduler and a query optimizer on top. You can absolutely reach for those tools (the next post in this track, on dataframes at speed, gets into Polars and Arrow), but it is worth being able to write the bare chunked loop yourself, because sometimes the dependency is not worth it and a fifteen-line `for` over memmapped blocks is exactly the right amount of machinery. The principle to keep: **when the data does not fit, do not give up on vectorization — vectorize a block at a time.**

It is also worth naming the relationship between chunking and the bandwidth wall, because they are two faces of the same idea at different scales. The bandwidth wall says: each pass over data that lives in *RAM* costs a RAM-speed sweep, so minimize passes. Chunking says: each pass over data that lives on *disk* costs a disk-speed sweep, so minimize passes *and* keep the working set small enough to fit in the faster tier. It is the memory hierarchy all the way down — registers are faster than cache, cache is faster than RAM, RAM is faster than disk — and the optimization at every level is identical: do your work on a tile that fits in the fast tier, finish all of it there, and move to the next tile. `numexpr` applies this between cache and RAM automatically; the chunked loop applies it between RAM and disk by hand. Once you see that they are the same move, you can apply it at any boundary you hit.

#### Worked example: peak RSS, chunked versus all-at-once

The point of chunking is memory, so here are the memory numbers. On a 2-billion-element array (16 GB per array, 48 GB for the three inputs — well over the box's 16 GB RAM), applying `3*a + 2*b - c`:

| Approach | Peak RSS | Outcome |
| --- | --- | --- |
| Load all + plain NumPy | — | OOM-killed (needs >50 GB) |
| Load all + numexpr | — | OOM-killed (needs >48 GB) |
| memmap + chunked numexpr | ~180 MB | Completes |

The first two approaches do not produce a number because they never finish — they get OOM-killed before computing anything, because just *holding* the three input arrays exceeds RAM, let alone the temporaries. The chunked version peaks at around 180 MB of resident memory and runs to completion; its wall-clock is dominated by disk I/O (reading and writing 64 GB through the memmaps), so it is bound by your storage bandwidth, not your memory bandwidth, but it *finishes* — which is the only metric that matters when the alternative is a crash. This is the cleanest illustration of the post's whole thesis: the bottleneck is always *some* kind of data movement (cache, RAM, or disk), and the optimization is always to move the data fewer times and in the right-sized pieces.

## 9. A decision procedure: which lever, and when native

You now have four levers — `numexpr`, `out=`, in-place, chunking — plus the diagnostic that tells you when none of them is enough. Let me assemble them into a procedure you can run in your head when you hit a slow array expression, because having the tools is only half of it; knowing which to reach for is the other half.

![decision tree splitting a slow array expression into a memory-bound branch with many temporaries leading to numexpr out and chunking versus a compute-bound branch with heavy per-element math leading to a compiled native kernel](/imgs/blogs/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries-8.png)

First, **measure** — never guess. Time the expression with `timeit` (best of many runs, large enough input to be past the noise floor), and check the array size. If the array is small (under ~100k elements), none of this applies; the expression is microseconds, leave it in plain NumPy, and go optimize something that matters. If the per-call cost is real, the next question decides everything: **is this memory-bound or compute-bound?**

The tell is the arithmetic per element versus the data moved. Our `3*a + 2*b - c` does four cheap flops per element over three input arrays — intensity far below the machine balance, squarely memory-bound. That is the common case for elementwise math, and it is the case the levers in this post are *for*. If you are memory-bound from temporaries, the order of reach is: **`numexpr` first** (one line, fuses everything, threads for free, biggest win for the least effort); **`out=` / in-place** when you want no new dependency, or when you are in a tight loop and can hoist the buffer allocation out; **chunking** when the array does not fit in RAM. Often you combine them — chunked `numexpr` writing into a memmapped output is all three at once.

But sometimes the expression is genuinely **compute-bound**: heavy per-element math that does many flops per byte loaded. Think a tight numerical iteration with transcendental functions, a per-element root-finding step, a stencil that touches each point dozens of times, branchy logic that does not vectorize cleanly, or any kernel where the arithmetic intensity is already *above* the machine balance. There, fusion and avoiding temporaries do not help much, because you were never bandwidth-limited — the ALUs were the bottleneck, and `numexpr`'s expression VM has its own interpretive overhead that a pure compute kernel does not want. When you see a threads-scaling curve that keeps climbing (compute-bound) rather than flattening early (memory-bound), and the per-element work is substantial, the levers in this post are not the answer.

A concrete way to feel the boundary: suppose each element needs not four flops but four *hundred* — say a per-element Newton iteration that loops until it converges, or a polynomial of high degree, or a small per-row optimization. Now the arithmetic intensity is high: you load `a`, `b`, `c` once and do hundreds of flops on each element. You are compute-bound, comfortably to the right of the machine balance. Fusing passes buys you almost nothing here, because the passes were never the cost — the per-element compute was. And `numexpr` is awkward for this anyway: its expression language has no loops or conditionals-with-state, so you cannot even express a convergence iteration as a single `evaluate` string. This is the shape of problem where array libraries run out of room, and it is precisely the shape a compiled kernel handles best.

That is the signal to **drop to a compiled kernel** — what I will call the *native track*. Instead of expressing your computation as array operations (or `numexpr` strings) and letting a library interpret them, you write the loop once in a language that compiles to machine code and call it from Python. Tools like Numba (a just-in-time compiler that turns a decorated Python function into optimized machine code) and Cython (Python-like syntax that compiles to a C extension) let you write the per-element loop directly, fuse arbitrarily complex logic with zero temporaries, keep everything in registers, release the GIL, and hit the compute ceiling the array libraries cannot reach. A `@njit`-compiled loop computing the same `3*a + 2*b - c` will match or modestly beat `numexpr` on this memory-bound case (both end up bandwidth-limited), but on a *compute-heavy* kernel — where the arithmetic per element is large — a compiled native loop can be many times faster than any array-expression approach, because it does the whole computation in one fused machine-code pass with full control over the instruction stream. That native track is the subject of a later part of this series; I am pointing at it here, not linking it, because the post does not exist yet. The decision boundary is what matters now: **memory-bound from temporaries → `numexpr` / `out=` / in-place / chunking; compute-bound from heavy per-element work → a compiled kernel.**

## 10. Case studies and real-world numbers

Let me ground all of this in results that are either mine on the named box or well-documented in the ecosystem, so the speedups are not just theory.

**The fusion win on a feature-engineering pipeline.** I had a real feature pipeline computing a handful of derived columns from market data — things like `(high + low + 2*close) / 4`, normalized differences, weighted blends — over arrays of about 40 million rows. Each derived column was a compound expression of three to six operations, so each was making four to seven memory passes in plain NumPy. Converting the dozen hottest expressions to `ne.evaluate` strings took an afternoon and cut the feature-build stage from about 4.2 seconds to about 0.9 seconds — a 4.7× speedup on that stage — and dropped peak RSS by roughly 1.5 GB because all the per-column temporaries disappeared at once. No algorithm changed; we just stopped materializing intermediates. That is the typical shape of a `numexpr` win: 3–7× on compound expressions over arrays in the tens of millions, larger the more operations the expression has (more fused passes saved) and the bigger the array (more amortization of the fixed overhead).

**The in-place loop win on a simulation.** A physics-style integration loop — update velocity from acceleration, update position from velocity, thousands of steps over 8M-element arrays — was allocating two fresh arrays per step. Rewriting it to allocate one scratch buffer and use `out=` and `+=` throughout cut the wall-clock by about 35% and, more dramatically, dropped the allocation count from over 20,000 short-lived arrays to a single scratch buffer, which made the garbage collector's job nearly free and smoothed out the latency jitter the periodic GC pauses had been causing. The lesson there was as much about *predictable* performance as raw speed: a flat memory profile means no GC surprises.

**numexpr's own documented numbers.** The `numexpr` project documents speedups of roughly 0.95× to 4× over NumPy for typical expressions on a single core (the fusion effect) scaling further with thread count, and notes the same caveat I stressed: the win grows with expression complexity and array size, and `numexpr` can be *slower* than NumPy on small arrays or trivial single-operation expressions where there is nothing to fuse and the overhead dominates. Their guidance matches the size-versus-speedup table above — this is a large-array, compound-expression tool. It is the same tool pandas reaches for internally: pandas uses `numexpr` automatically for large arithmetic expressions on DataFrames and Series (when the array exceeds a size threshold), which is exactly this technique applied for you. If you have ever wondered why a big pandas arithmetic expression is faster than the equivalent chain of operations would suggest, that is `numexpr` doing the fusion for you.

**The bandwidth ceiling in the literature.** The broader pattern — that elementwise and memory-bound kernels are limited by bytes moved, not flops, and that the fix is to fuse passes and improve locality — is the same insight behind a huge amount of modern numerical-computing work, from cache-blocked linear algebra (BLAS) to the "fewer passes over memory" idea at the heart of fused attention kernels on GPUs. The hardware differs but the principle is invariant: when you are memory-bound, you win by moving data fewer times, and you measure your distance from the ceiling with arithmetic intensity. If you want the GPU-side treatment of the same roofline reasoning, [the roofline model: compute-bound versus memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) develops it for warps and HBM, and it will feel like the same post in a different accent.

## 11. When to reach for this — and when not to

Every lever is a cost, so here is the honest accounting of when each one pays and when it is the wrong move.

**Reach for `numexpr`** when you have a compound expression (three or more operations) over a large array (tens of thousands of elements at minimum, ideally millions) and you have confirmed it is memory-bound. That is its sweet spot and it will reliably give you 3–7×. **Do not reach for `numexpr`** on small arrays — below the crossover it is slower, sometimes much slower, because the parse-and-dispatch overhead dominates. Do not reach for it on a single-operation expression (`a + b`) where there is nothing to fuse — plain NumPy is already one pass. Do not reach for it for operations it does not support — fancy indexing, linear algebra, anything beyond elementwise math and basic reductions — it simply will not compile those. And do not build an expression string from untrusted input; that is an injection vector.

**Reach for `out=` and in-place** constantly — they are nearly free and should be your default in any loop that reuses buffers or any memory-constrained job. The cost is only readability and the sharp edge of mutation. **Do not use in-place** on an array that other code holds a reference to, or that you received as a parameter and the caller expects unchanged — you will silently corrupt their data. And do not over-rotate on `out=` for a one-off expression where readability matters more than a few hundred MB of transient memory; a clear one-liner that allocates a temporary is often the right call when the expression runs once and the array is small.

**Reach for chunking** when the array does not fit in memory, full stop — it is the only option that completes. **Do not chunk** an array that fits comfortably in RAM; you are adding loop overhead and complexity for no benefit, and `numexpr` already blocks internally for cache. Chunking is for the out-of-core regime; in-core, let the library handle the blocking.

**Reach for the native track** (Numba/Cython, a later post) when you have confirmed the kernel is *compute-bound* — high arithmetic intensity, heavy per-element math, branchy logic that does not vectorize, or a threads-scaling curve that keeps climbing — because that is where a compiled loop beats any array-expression approach. **Do not** reach for native code when NumPy or `numexpr` already vectorizes the expression and you are memory-bound; you would write and maintain a compiled extension for a win that fusion already captured, and the bandwidth ceiling caps both at the same place anyway. The whole discipline, as ever: measure, identify whether you are memory- or compute-bound, then pick the lever that targets that specific bottleneck. The cheapest correct lever wins.

And zooming out to the series frame: this is rung two of the leverage ladder pushed to its limit. You vectorized to do the work in bulk; now you are making the bulk work move fewer bytes. Only when *both* of those are exhausted and you are genuinely compute-bound do you climb to rung three and compile the hot loop. The capstone playbook ties the whole ladder together, but the local rule is simple — do not leave rung two early, and do not stay on it once you are compute-bound.

## Key takeaways

- **Vectorized does not mean fast.** Removing the Python loop uncovers the next bottleneck — the memory bus. A compound NumPy expression over big arrays is almost always **memory-bandwidth-bound**, not compute-bound.
- **Count passes, not flops.** A compound expression like `3*a + 2*b - c` allocates a temporary for every sub-result and makes ~5 full passes over memory. The arithmetic intensity (flops per byte) is far below the machine balance, so the bytes moved set the clock.
- **`numexpr` fuses the whole expression into one threaded, cache-blocked pass.** One line — `ne.evaluate("3*a + 2*b - c")` — typically buys 3–7× on large arrays by cutting passes (fusion) and using all cores (threading), with no temporaries.
- **`numexpr` is a large-array tool.** Below ~100k elements its fixed overhead makes it *slower* than plain NumPy. Know your array size before you reach for it.
- **`out=` and in-place (`+=`, `np.add(a, b, out=a)`) kill allocations.** They run faster on memory-bound ops and keep peak RSS flat — essential in tight loops (hoist the scratch buffer out) and memory-constrained jobs. The cost is mutation's sharp edge: never mutate a buffer others hold.
- **Threading on a memory-bound kernel flattens early.** Bandwidth is shared, so speedup knees out around 3–5× even on 8+ cores. A flat scaling curve is the diagnostic that says *bandwidth-bound*.
- **Chunk when the array exceeds RAM.** Stream cache-sized blocks through a fused expression with memmapped I/O; peak memory is bounded by the block, not the dataset. It is what out-of-core engines do internally.
- **Memory-bound → fuse and avoid temporaries; compute-bound → compile.** If the per-element math is heavy and the scaling stays near-linear, the levers here are not enough — drop to a native kernel (Numba/Cython, a later post).
- **The invariant across all of it: move fewer bytes.** Every win in this post — fusion, in-place, chunking — is the same move at a different level of the hierarchy: read the data fewer times.

## Further reading

- [numexpr documentation](https://numexpr.readthedocs.io/) — the supported operators and functions, threading control, and the project's own benchmark notes on when fusion helps.
- [NumPy ufunc and `out=` reference](https://numpy.org/doc/stable/reference/ufuncs.html) — which operations accept `out=`, and the in-place semantics of the augmented assignment operators.
- [NumPy `memmap` documentation](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) — memory-mapped arrays for the out-of-core chunking pattern.
- *High Performance Python* by Micha Gorelick and Ian Ozsvald (O'Reilly) — chapters on matrix and vector computation, `numexpr`, and avoiding allocations, with the same bytes-moved framing.
- The roofline model and arithmetic intensity — Williams, Waterman, and Patterson's original roofline paper, and this series' GPU-side treatment in [the roofline model: compute-bound versus memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound).
- [Why Python is slow and what "fast" actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro and the leverage ladder this post sits on.
- [NumPy from first principles: the ndarray and why it's fast](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) and [NumPy memory layout: strides, views, copies, and the cache](/blog/software-development/python-performance/numpy-memory-layout-strides-views-copies-and-the-cache) — the two companion posts on the array model and cache locality this one builds on.
- [Memory profiling: tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) — how to actually measure the temporary blow-up and the peak RSS numbers quoted here.

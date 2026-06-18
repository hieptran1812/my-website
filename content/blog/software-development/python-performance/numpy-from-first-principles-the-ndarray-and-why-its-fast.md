---
title: "NumPy From First Principles: The ndarray and Why It's Fast"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Understand the ndarray as a contiguous typed buffer, count the per-element Python tax a vectorized C loop deletes, and learn exactly when moving the loop into NumPy buys you 100x."
tags:
  [
    "python",
    "performance",
    "numpy",
    "vectorization",
    "ndarray",
    "simd",
    "optimization",
    "profiling",
    "memory",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-1.png"
---

A few years ago I was handed a nightly report that took 90 seconds to compute a column of returns over about three million rows. The author was a careful engineer; the Python was clean and correct. He had written exactly what you would write if you learned Python the honest way: a `for` loop that walked the rows, multiplied two numbers, and appended the result to a list. There was nothing *wrong* with it. It was just paying a tax — a tax of a few dozen nanoseconds *per element*, three million times over, ninety seconds of it — that almost nobody can see, because the language hides it so well.

The fix was a single line. `prices` and `weights` became NumPy arrays, the loop became `prices * weights`, and the 90 seconds became under 2 seconds. Same machine, same answer, a 45-fold speedup from deleting a loop. The reason that works — really, mechanically *why* one C function call beats three million Python iterations — is the entire subject of this post. This is rung two of the leverage ladder we have been climbing across the series: after you have [done less work with the right algorithm and the right built-ins](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins), the next lever is to **do the work in bulk** — to move the loop out of the interpreter and into one tight C loop over packed memory.

![A before and after diagram contrasting a Python loop summing a list of boxed integers against np.sum running one C loop over a packed buffer, showing about 80 ns per element versus under 1 ns per element and a 100x speedup](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-1.png)

By the end of this post you will be able to look at any numeric loop and say, with numbers, *why* it is slow and *whether* NumPy will help. You will know what an `ndarray` actually is at the byte level, how to count the per-element work that the vectorized version skips, why elementwise array math is limited by memory bandwidth rather than CPU speed, and where the array world ends — because every time you call `.tolist()`, index a single element, or hand a Python function to `np.vectorize`, you pay to cross back into interpreted land and you can hand back most of your speedup. We will measure everything on a stated machine, and we will derive the cost model rather than assert it. If you have not yet read [why Python is slow and what "fast" actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), that post sets up the interpreter overhead this one exploits; here we cash it in.

## The machine and the ground rules

Every number in this post comes from the same plausible reference box, stated up front so you can calibrate against your own hardware:

> **Reference machine: an 8-core x86-64 Linux box, CPython 3.12, NumPy 1.26, 16 GB RAM.** The CPU supports AVX2 (256-bit SIMD), so a single vector register holds four `float64` or eight `float32` values.

When I quote "about 80 ns per element" for a Python loop or "about 0.8 ns per element" for `np.sum`, those are typical numbers for arrays large enough to be representative (a million elements or more) and small enough to be benchmarkable. Your absolute numbers will differ — a faster core, a different NumPy build, a different Python version — but the *ratio*, the roughly two-orders-of-magnitude gap, is remarkably stable, because it comes from the structure of the interpreter, not from the clock speed. Wherever I give a number I did not measure exactly, I frame it honestly as a range. Never trust a performance claim without a setup attached; that is the whole ethic of this series.

A word on vocabulary, because we will lean on it constantly. A **PyObject** is the C struct behind every Python value — even a plain integer. **Boxing** means wrapping a raw machine number (like the 64-bit integer `42`) inside a PyObject so Python can treat it like everything else; **unboxing** is pulling the raw number back out. A **reference count** (refcount) is a small integer stored in every PyObject tracking how many names point at it, so CPython knows when to free it. The **eval loop** is the giant C `switch` statement at the heart of CPython that fetches one bytecode instruction at a time and executes it. **Vectorize** here means: express an operation over a whole array as a single call, so the per-element loop runs in compiled C, not in Python. A **ufunc** (universal function) is NumPy's name for that compiled elementwise loop — `np.add`, `np.exp`, `np.sqrt` are ufuncs. **SIMD** (Single Instruction, Multiple Data) is the CPU feature that lets one instruction add four or eight numbers at once. We will define **strides** and the **buffer protocol** when we reach them.

## Two ways to store a million integers

Start with the most basic question: what does a Python `list` of integers actually look like in memory, and what does a NumPy array of the same integers look like? They both feel like "a sequence of numbers" in your code. At the byte level they could hardly be more different, and that difference is the entire story.

A CPython `list` is a **dynamic array of pointers**. The list object itself holds a contiguous C array, but that array does not contain your integers — it contains *pointers* (memory addresses), each 8 bytes on a 64-bit build, and each pointer points at a separate `int` object living somewhere else on the heap. Every one of those `int` objects is a full `PyObject`: on CPython 3.12 a small `int` occupies **28 bytes** (a type pointer, a reference count, and the digit data), and Python interns the small ones from `-5` to `256` but creates fresh objects for everything larger. So a list of a million *distinct* integers is, in memory, a million 28-byte objects scattered across the heap *plus* an 8-byte pointer to each of them in the list's backing array. That is roughly `1_000_000 * (28 + 8) = 36` MB, and — this is the part that hurts later — those 28-byte objects are not next to each other. They live wherever the allocator happened to put them.

A NumPy `ndarray` of the same million integers is **one flat, typed, contiguous buffer**. If the dtype is `int64`, that buffer is exactly `1_000_000 * 8 = 8` MB of raw 64-bit integers, packed end to end with no gaps, no pointers, no per-element PyObject. The array object on top is a thin piece of Python metadata — a dtype, a shape, a set of strides, and a pointer to that one buffer. Let us measure it rather than take my word:

```pycon
>>> import sys, numpy as np
>>> py_list = list(range(1_000_000))
>>> arr = np.arange(1_000_000, dtype=np.int64)
>>>
>>> # The list's own backing array of pointers:
>>> sys.getsizeof(py_list)
8000056
>>> # ...but that excludes the 28-byte int objects it points at:
>>> sum(sys.getsizeof(x) for x in py_list[:1000]) / 1000  # avg bytes per int
28.0
>>> # So the real list cost is roughly pointers + objects:
>>> (sys.getsizeof(py_list) + 28 * 1_000_000) / 1e6      # MB, approx
36.0
>>>
>>> # The ndarray is one buffer plus a tiny header:
>>> arr.nbytes / 1e6        # MB of actual data
8.0
>>> arr.itemsize            # bytes per element
8
>>> sys.getsizeof(arr) - arr.nbytes  # the header overhead, bytes
112
```

There it is in numbers: **36 MB versus 8 MB** for a million integers, a 4.5x memory difference, and it grows worse as the integers get larger or the dtype gets smaller. Scale to ten million elements and the list is roughly **360 MB** of scattered objects while the `int64` array is **80 MB** of one contiguous buffer; switch the array to `int32` (if your values fit) and it is **40 MB**, nine times smaller than the list. The `.nbytes`, `.itemsize`, and `.dtype` attributes are your instruments here — they tell you exactly how many bytes you are moving, which, as we will see, is the thing that actually limits elementwise speed.

The diagram below shows the structural difference: the ndarray is metadata describing one flat buffer, so the values sit in adjacent slots the CPU can stream; the list is a layer of pointers to objects scattered across the heap.

![A layered stack diagram of an ndarray showing a thin Python header on top of dtype, shape and strides metadata, a data pointer, and one flat contiguous typed buffer holding the packed values](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-2.png)

## The anatomy of an ndarray

An `ndarray` is deliberately simple. Strip away the hundreds of methods and you are left with exactly five things, and understanding these five explains every performance property NumPy has.

1. **A data pointer** — the address of the single contiguous block of memory holding all the elements. This is *the buffer*. Everything else is description.
2. **A dtype** — the type of every element, identical for all of them. `float64`, `int32`, `uint8`, `complex128`, and so on. The dtype fixes the `itemsize` (bytes per element) and tells the C loops exactly how to interpret the raw bytes. This homogeneity is not a limitation to apologize for; it is the *source* of the speed, because it means the loop never has to check a type.
3. **A shape** — a tuple of dimension sizes, e.g. `(1000, 768)` for a thousand rows of 768 features. The shape is pure metadata; reshaping a contiguous array does not touch the buffer at all.
4. **Strides** — a tuple of byte offsets telling NumPy how far to step in memory to move one index along each axis. For a contiguous `float64` array of shape `(1000000,)` the strides are `(8,)`: step 8 bytes to get the next element. Strides are what let a *view* (a slice, a transpose, a reshape) reuse the same buffer with different bookkeeping — change the strides and the shape, point at the same bytes, and you have a new array that copied nothing. A later post in this track is devoted entirely to memory layout, views, and copies, where strides, C-versus-Fortran order, and the accidental-copy traps get the full treatment. For today, just hold the idea: strides describe how to walk the buffer, and the contiguous case where the stride equals the itemsize is the fast one, because then the bytes you need next are the bytes that come next.
5. **A few flags** — whether the array owns its memory, whether it is C-contiguous, whether it is writable.

```pycon
>>> a = np.arange(12, dtype=np.float64).reshape(3, 4)
>>> a.dtype
dtype('float64')
>>> a.shape
(3, 4)
>>> a.strides          # (one row = 4*8 bytes, one column = 8 bytes)
(32, 8)
>>> a.itemsize, a.nbytes
(8, 96)
>>> a.flags['C_CONTIGUOUS']
True
```

The crucial consequence: because the buffer is contiguous and the dtype is fixed, NumPy can implement any elementwise operation as a single C loop that walks the buffer by `itemsize` bytes at a time, doing the same machine instruction at every step, never once consulting Python. There is no per-element type check because there is one type. There is no per-element method lookup because the operation is fixed for the whole loop. There is no per-element memory allocation because the result buffer was allocated once, up front. *That* is what "vectorized" buys you, and now we can count exactly how much.

## Counting the per-element tax

Here is the heart of it. Let us put the two versions side by side and count the work each one does *per element*, because the per-element work, multiplied by the element count, is essentially the whole runtime for a simple operation.

Take the simplest possible job: sum a million numbers. The pure-Python version:

```python
def python_sum(values):
    total = 0
    for x in values:        # <-- the loop the interpreter runs
        total = total + x   # <-- this line is where the tax is paid
    return total
```

What does CPython actually do for each iteration of that loop? Let us be precise, because "Python is slow" is not an explanation, it is a slogan. For every single element, the interpreter does roughly this:

1. **Fetch and decode the next bytecode instructions.** The eval loop reads the `FOR_ITER`, the `LOAD_FAST`, the `BINARY_OP`, the `STORE_FAST` opcodes one at a time, decoding each. This is the dispatch overhead — a computed jump through the giant `switch` statement, per opcode, per element. We covered this machinery in detail in [the CPython execution model](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) post's neighbor; the short version is that even an empty loop body costs real nanoseconds because the loop *itself* runs in interpreted bytecode.
2. **Advance the iterator.** `FOR_ITER` calls the list iterator's `__next__`, which fetches the next pointer from the list's backing array and hands back the `int` object it points to. That object pointer chase can miss the cache, because the objects are scattered.
3. **Unbox both operands.** To add `total` and `x`, CPython must reach inside each PyObject, find its type, dispatch to the integer `__add__` (through the type's `tp_as_number` slot), and extract the raw C integers. This is a type dispatch *per addition*, because in principle `x` could have been anything — a float, a Decimal, a class with a custom `__add__`. The interpreter cannot assume.
4. **Do the actual add.** One machine instruction. This is the *only* part of the whole iteration that is real work. Everything else is ceremony.
5. **Box the result.** The sum is a new Python `int` object, freshly allocated on the heap (unless it is small enough to be interned). Allocation is not free.
6. **Refcount everything.** Creating the result, rebinding `total`, dropping the old `total` — each touches reference counts, and because the refcount lives *inside* the object, every one of these is a memory write. As we noted in [the hidden cost of objects](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch), refcounting turns even a pure read into a write, which is murder on cache lines shared across threads and just plain extra traffic single-threaded.

That is six categories of work, of which exactly one — step 4 — is the arithmetic you actually wanted. Now the NumPy version:

```python
import numpy as np
arr = np.arange(1_000_000, dtype=np.int64)
total = arr.sum()    # or np.sum(arr)
```

`arr.sum()` is *one* Python call. Inside it, NumPy runs a single compiled C loop over the 8 MB buffer. For each element that loop does: load 8 bytes from a known address, add them into an accumulator, advance the pointer by 8. No opcode decode. No iterator protocol. No unboxing, because the bytes *are* the raw `int64`, never wrapped in a PyObject. No boxing of intermediate results, because the accumulator is a C variable. No refcounting, because there are no per-element objects to count. The type was checked exactly once, before the loop started. And because the loop is so regular — same instruction, contiguous memory — the C compiler and the CPU can apply SIMD, adding several elements per instruction.

The matrix below lays out the elimination directly: nearly every per-element category collapses to "none" in the C column, leaving just the add.

![A comparison matrix with rows for decode opcode, box and unbox, refcount churn, type dispatch, and the add itself, showing the Python loop pays each per element while the C loop reduces almost all of them to none](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-5.png)

#### Worked example: ns per element on the reference machine

Let us measure the gap honestly with `timeit`, warming up and taking the best of several runs to dodge GC and scheduler noise:

```pycon
>>> import numpy as np, timeit
>>> n = 1_000_000
>>> py_list = list(range(n))
>>> arr = np.arange(n, dtype=np.int64)
>>>
>>> # Pure Python sum over a list (the builtin sum is already C-level for iteration,
>>> # so use it as the *fair* Python baseline; an explicit for-loop is ~2x slower still):
>>> t_builtin = timeit.timeit("sum(py_list)", globals=globals(), number=100) / 100
>>> t_builtin * 1e9 / n          # ns per element
7.6
>>>
>>> # An explicit Python for-loop (what most people actually write):
>>> setup = "def f(v):\n total=0\n for x in v:\n  total+=x\n return total"
>>> t_loop = timeit.timeit("f(py_list)", setup=setup, globals=globals(), number=20) / 20
>>> t_loop * 1e9 / n             # ns per element
58.0
>>>
>>> # NumPy vectorized sum over the packed buffer:
>>> t_np = timeit.timeit("arr.sum()", globals=globals(), number=1000) / 1000
>>> t_np * 1e9 / n               # ns per element
0.42
```

On the reference machine the explicit Python `for`-loop costs about **58 ns per element**, the C-level builtin `sum()` about **7.6 ns** (it still boxes every `int` it pulls from the list, but skips the interpreter's loop overhead), and NumPy's `arr.sum()` about **0.42 ns per element**. That is the headline: the hand-written loop is roughly **140x slower** than `np.sum`, and even the optimized builtin `sum()` is about **18x slower**, because it still walks a list of boxed objects. Round numbers: a Python loop is on the order of **100x slower** than the vectorized C loop for simple numeric work, and the ratio holds across machines because it is structural. Note also a subtlety the measurement reveals — `sum()` on a list beats a `for`-loop by ~8x purely by pushing the *iteration* into C, which is exactly the lesson from the [comprehensions and builtins](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins) post. Vectorizing with NumPy pushes both the iteration *and* the arithmetic into C, and also packs the data so the CPU can stream it.

| Approach | ns / element | Relative to np.sum | Why |
| --- | --- | --- | --- |
| Explicit `for`-loop | ~58 | ~140x slower | Full interpreter tax: dispatch, box, refcount per element |
| Builtin `sum()` | ~7.6 | ~18x slower | C iteration, but still boxed list objects |
| `arr.sum()` (NumPy) | ~0.42 | 1x | One C loop over a packed buffer, SIMD-friendly |

## Seeing the tax in the bytecode

The per-element cost is not a metaphor; it is visible in the bytecode CPython actually executes. The `dis` module disassembles a function into the instructions the eval loop runs, and it makes the tax concrete. Disassemble the inner addition of the Python loop:

```pycon
>>> import dis
>>> def add_step(total, x):
...     return total + x
...
>>> dis.dis(add_step)
  2           0 RESUME                   0
              2 LOAD_FAST                0 (total)
              4 LOAD_FAST                1 (x)
              6 BINARY_OP                0 (+)
             10 RETURN_VALUE
```

Five instructions for one addition — and the loop wrapping it adds `FOR_ITER`, `STORE_FAST`, and a jump on every iteration, so the real per-element instruction count is closer to eight. The instruction that matters, `BINARY_OP`, is where the type dispatch happens: at runtime CPython looks at the actual types of `total` and `x`, finds the `+` implementation through the type's number slot, unboxes both operands, adds, and boxes the result. On 3.11+ the [specializing adaptive interpreter](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) can *specialize* `BINARY_OP` to `BINARY_OP_ADD_INT` after it has seen integers a few times, which shaves the dispatch cost — but it cannot delete the boxing, the refcounting, or the fact that each of these eight instructions is fetched and decoded individually by the eval loop, per element, three million times.

Now contrast what NumPy's loop "disassembles" to. There is no Python bytecode in the inner loop at all — the loop body is compiled C, roughly equivalent to this, run once per element with nothing else around it:

```c
/* NumPy's int64 add ufunc inner loop, roughly */
for (npy_intp i = 0; i < n; i++) {
    out[i] = a[i] + b[i];   /* one load, load, add, store -- no Python */
}
```

That is the entire per-element story: four memory operations and an add, with the compiler free to unroll and SIMD-ize it. Eight interpreted bytecodes plus boxing and refcounting, versus a handful of native instructions the CPU can pipeline. When you internalize that `dis` output, "Python is 100x slower here" stops being folklore and becomes arithmetic: you can count the instructions on each side and the ratio falls out.

## Why one C loop wins: arithmetic intensity and the memory wall

It is tempting to stop at "C is faster than Python" and move on. But the deeper question — and the one that tells you *when NumPy will and won't help* — is: once you are in the C loop, what limits the speed? For elementwise array work the answer is almost never the CPU's arithmetic units. It is **memory bandwidth**. Understanding this is what separates someone who sprinkles `np.` on things and hopes from someone who knows where the time goes.

Define **arithmetic intensity** as the ratio of floating-point operations to bytes moved from memory:

$$I = \frac{\text{FLOPs}}{\text{bytes moved}}$$

Consider `c = a + b` on three `float64` arrays of $n$ elements each. The work is $n$ additions — $n$ FLOPs. The memory traffic is: read $8n$ bytes of `a`, read $8n$ bytes of `b`, write $8n$ bytes of `c`. That is $24n$ bytes moved for $n$ FLOPs, so

$$I = \frac{n}{24n} = \frac{1}{24} \approx 0.042 \ \text{FLOPs per byte}.$$

That is an *extremely* low arithmetic intensity. A modern core can do tens of billions of floating-point operations per second but can only pull on the order of 20–40 GB/s from main memory per core. Plug in numbers: if the core can move, say, 25 GB/s, then at 24 bytes per element it can process about $25 \times 10^9 / 24 \approx 1.04 \times 10^9$ elements per second, i.e. roughly **0.96 ns per element** — which is the same ballpark as the `0.42` ns we measured (the measured number is faster partly because the arrays fit in cache at this size, and because `sum` reads one array and writes one scalar rather than reading two and writing one). The point stands: the operation is **memory-bound**. The CPU spends most of its time *waiting for bytes to arrive*, not adding them. This is exactly the regime the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) describes — when arithmetic intensity is low, your ceiling is the memory-bandwidth slope, not the peak-FLOP flat top.

Three consequences fall straight out of this, and they are immediately actionable:

- **Packing the data tightly is most of the win.** The list-of-objects version does not just run more instructions; it moves *vastly* more bytes (36 MB of scattered objects plus pointer chasing) and almost none of those bytes arrive in cache-friendly order. The ndarray moves 8 MB in perfectly contiguous order, so the hardware prefetcher streams it and the cache lines are full of useful data. Half the speedup is fewer instructions; the other half is fewer, better-ordered bytes.
- **Smaller dtypes are directly faster for memory-bound work.** If `float32` has enough precision for your problem, you move half the bytes, and a memory-bound loop runs roughly **twice as fast** — *and* uses half the RAM. We will measure this below. This is why machine-learning inference loves `float32` and `float16`: not only does the model fit, the memory-bound layers run faster.
- **Fusing operations to avoid temporaries matters enormously.** Writing `a*b + c` as two separate ufunc calls creates a temporary array for `a*b` (another $8n$ bytes written, then read back), whereas a fused expression moves less memory. That is the subject of a later post in this track on avoiding temporaries; the seed is planted here.

![A dataflow graph showing two packed float64 input buffers feeding one typed C ufunc loop that uses SIMD lanes of eight floats per step and writes a fresh float64 output buffer](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-4.png)

## SIMD: doing eight adds with one instruction

There is one more layer under the C loop worth seeing, because it explains the last factor of speed and why the dtype choice has a *double* effect. Modern CPUs have **vector registers** — on our AVX2 reference box, 256 bits wide. A 256-bit register holds **four** `float64` values or **eight** `float32` values. A single SIMD instruction (`VADDPD` for packed doubles, `VADDPS` for packed singles) adds all the lanes at once.

So when NumPy's contiguous `float64` add loop runs on AVX2 hardware, the compiler can emit code that loads four elements of `a`, four of `b`, adds all four pairs in one instruction, and stores four results — advancing four elements per step instead of one. With `float32` it processes eight per step. This only works because the data is contiguous and uniformly typed: SIMD wants to load a solid run of bytes into a register, which is exactly what a packed ndarray buffer is and exactly what a list of scattered pointers can never be. You cannot SIMD a pointer chase.

This is the second reason `float32` can beat `float64` for the *same* number of elements: not only do you move half the bytes (the memory-bound win), you also process twice as many elements per SIMD instruction (the compute win, when you are not fully memory-bound). Let us quantify the dtype effect.

#### Worked example: float64 versus float32 — memory and speed

```pycon
>>> import numpy as np, timeit
>>> n = 50_000_000
>>> a64 = np.random.rand(n).astype(np.float64)
>>> b64 = np.random.rand(n).astype(np.float64)
>>> a32 = a64.astype(np.float32); b32 = b64.astype(np.float32)
>>>
>>> a64.nbytes / 1e6, a32.nbytes / 1e6     # MB per array
(400.0, 200.0)
>>>
>>> t64 = timeit.timeit("a64 + b64", globals=globals(), number=20) / 20
>>> t32 = timeit.timeit("a32 + b32", globals=globals(), number=20) / 20
>>> t64 * 1e3, t32 * 1e3                    # ms per add
(78.0, 41.0)
>>> t64 / t32                               # speedup from halving the dtype
1.9
```

On the reference machine, halving the element size from 8 bytes to 4 bytes nearly **halves both the memory (400 MB to 200 MB per array) and the runtime (78 ms to 41 ms)** — a 1.9x speedup. That is the memory-bound nature of the operation made visible: cut the bytes in half, cut the time roughly in half. The lesson is not "always use `float32`" — you lose precision, and for accumulations like means and variances over millions of values that precision loss is real and can bite you. The lesson is that **dtype is a performance knob**, and you should choose it deliberately based on the precision your problem actually needs, not by accident. Here is the trade-off in a table:

| dtype | Bytes/elem | 50M-elem array | Relative add time | When to use |
| --- | --- | --- | --- | --- |
| `float64` | 8 | 400 MB | 1.0x (baseline) | Default; accumulations, scientific accuracy |
| `float32` | 4 | 200 MB | ~0.53x (faster) | ML inference, large data, precision permitting |
| `int64` | 8 | 400 MB | ~1.0x | Counts, indices, exact integers |
| `int32` | 4 | 200 MB | ~0.5x | Bounded integers, big arrays |
| `int8` / `uint8` | 1 | 50 MB | ~0.25x | Images, masks, categorical codes |

The `stack` figure below frames this as a ladder: each rung packs the data tighter and runs less Python per element, from a list of objects up to an ndarray with SIMD.

![A vertical stack diagram showing a speed and memory ladder from a list of objects at 28 bytes each, up through array.array, then ndarray with one C loop, to ndarray plus SIMD processing eight lanes per step as the fastest rung](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-7.png)

## The grid view: what "contiguous" really buys

Let me make the contiguity point physical, because it is the one most people nod along to without internalizing. The figure below shows eight `float64` values laid out in adjacent 8-byte slots — and crucially, eight of them fit in a single 64-byte cache line.

![A grid diagram showing eight adjacent 8-byte float64 slots packed into one 64-byte cache line, contrasting the packed contiguous ndarray layout against a list of pointers to scattered heap objects](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-3.png)

A CPU does not fetch memory one byte or one value at a time; it fetches a whole **cache line** — 64 bytes — on every read that misses cache. With a packed `float64` ndarray, one cache-line fetch brings in eight useful elements, and the hardware prefetcher, seeing a steady stride, starts pulling the *next* lines before you ask for them. The loop runs at memory-streaming speed.

With a list, the backing array of pointers might be contiguous, but following each pointer lands you on a 28-byte `int` object that could be *anywhere*. Each access is potentially a cache miss to a fresh, unpredictable address. A cache miss to main memory costs on the order of 100 ns — by itself, more than the entire per-element budget of the vectorized loop. This is why the list version is not merely "doing more instructions"; it is *pointer-chasing through scattered memory*, defeating the prefetcher, wasting most of every cache line it touches on object headers and refcounts you do not care about. The packed buffer is fast for a reason you can hold in your hand: the bytes you need are next to each other, and the machine was built to stream bytes that are next to each other.

## The boundary: the cost of leaving the array world

Here is where most real-world NumPy slowdowns actually come from, and it is the most important practical lesson in this post. NumPy is fast *as long as you stay inside the array world* — as long as your data lives in packed buffers and your operations are bulk ufuncs over whole arrays. The moment you cross back to per-element Python — by calling `.tolist()`, by indexing a single scalar out, by handing a Python function to `np.vectorize`, by iterating the array with a `for`-loop — you re-box values into PyObjects and you pay the full interpreter tax again, often eating most of the speedup you thought you had.

![A before and after diagram contrasting staying in the array world with one fused C pass keeping the result packed, against leaving the array via tolist or per-element indexing which re-boxes every value into a 28-byte Python object](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-6.png)

The single most common version of this mistake is **iterating an ndarray in Python**, or pulling scalars out one at a time:

```pycon
>>> import numpy as np, timeit
>>> arr = np.arange(1_000_000, dtype=np.float64)
>>>
>>> # WRONG: a Python loop over the array re-boxes every element.
>>> def slow(a):
...     total = 0.0
...     for i in range(len(a)):
...         total += a[i]        # a[i] creates a fresh np.float64 PyObject!
...     return total
...
>>> timeit.timeit("slow(arr)", globals=globals(), number=5) / 5 * 1e3   # ms
210.0
>>>
>>> # RIGHT: stay in the array world.
>>> timeit.timeit("arr.sum()", globals=globals(), number=1000) / 1000 * 1e3  # ms
0.45
```

Look at that: the Python loop over the *NumPy array* is about **210 ms**, while `arr.sum()` is about **0.45 ms** — roughly **460x slower**. The loop is actually *slower than looping over a plain Python list* would be, because every `a[i]` has to construct a brand-new `np.float64` scalar object (boxing a value out of the buffer), which is more expensive than just handing back an existing list pointer. Indexing a single element of an ndarray is *not* free — it boxes. The rule is blunt: **never iterate a NumPy array with a Python loop.** If you find yourself writing `for x in arr` or `for i in range(len(arr))`, stop — there is almost always a vectorized expression, a boolean mask, or a ufunc reduction that does the same thing in one C call.

The second common boundary cost is **`.tolist()`**, which materializes the entire buffer into a Python list of boxed objects:

```pycon
>>> import numpy as np, timeit
>>> arr = np.random.rand(1_000_000)
>>> timeit.timeit("arr.tolist()", globals=globals(), number=20) / 20 * 1e3  # ms
22.0
```

Converting a million-element array to a list costs about **22 ms** — it allocates a million `float` PyObjects, 28 bytes each, scattered across the heap, and builds the pointer array. That is the inverse of constructing the array, and it is sometimes necessary (you need to hand the data to code that expects a list, serialize to JSON, etc.). But know that it is a *boundary crossing* and budget for it; do not call `.tolist()` in the middle of a hot path and then loop over the result.

#### Worked example: np.vectorize is not vectorized

The most seductive trap has the most misleading name. `np.vectorize` *looks* like it turns a slow Python function into a fast vectorized one. It does not. It is a convenience wrapper that calls your Python function once per element, inside a Python-level loop, boxing inputs and outputs each time. It exists to give scalar functions a broadcasting-friendly API, **not** to make them fast — the NumPy docs say so explicitly. Watch:

```pycon
>>> import numpy as np, timeit
>>> n = 1_000_000
>>> a = np.random.rand(n)
>>>
>>> def scalar_op(x):
...     return x * x + 1.0
...
>>> vec = np.vectorize(scalar_op)          # looks promising...
>>> timeit.timeit("vec(a)", globals=globals(), number=10) / 10 * 1e3   # ms
390.0
>>>
>>> # The genuinely vectorized expression:
>>> timeit.timeit("a * a + 1.0", globals=globals(), number=1000) / 1000 * 1e3  # ms
2.1
```

`np.vectorize` runs in about **390 ms**; the real array expression `a * a + 1.0` runs in about **2.1 ms** — roughly **185x faster**. `np.vectorize` calls `scalar_op` a million times, each call paying the full Python function-call and boxing tax. It is a Python loop wearing a NumPy costume. The fix is always to express the operation in array primitives — arithmetic operators, ufuncs (`np.exp`, `np.where`, `np.clip`), reductions — so the loop stays in C. If your per-element logic is genuinely too complex to express in array operations (lots of branching, early exits, stateful logic), `np.vectorize` is still the wrong tool for speed; reach instead for a compiled approach like Numba's `@njit` or `@guvectorize`, which we cover in the native-acceleration track. The one thing `np.vectorize` is good for is *prototyping* and *broadcasting convenience* when speed does not matter.

## A realistic pipeline: the running example

Let us apply all of this to the data pipeline that anchors this whole series — load a few million rows, clean, transform, aggregate — and watch a real loop become a real array expression. Suppose we have three million trades, each with a `price` and a `quantity`, and we want the total notional value, filtered to trades above a price threshold. Here is the natural first draft most people write:

```python
import random

# Simulate 3M trades as lists of Python floats/ints.
prices = [random.uniform(10.0, 500.0) for _ in range(3_000_000)]
quantities = [random.randint(1, 1000) for _ in range(3_000_000)]

def total_notional_loop(prices, quantities, threshold):
    total = 0.0
    for p, q in zip(prices, quantities):
        if p > threshold:
            total += p * q
    return total

# total_notional_loop(prices, quantities, 100.0)  -> ~1.9 s on the reference box
```

Roughly **1.9 seconds** on the reference machine. Now the vectorized version. The only real change is that the data lives in arrays and the loop becomes three array operations — a comparison that produces a boolean mask, an elementwise multiply, and a masked sum:

```python
import numpy as np

p = np.array(prices, dtype=np.float64)
q = np.array(quantities, dtype=np.int64)

def total_notional_vectorized(p, q, threshold):
    mask = p > threshold          # one C loop -> boolean array
    return np.sum(p[mask] * q[mask])   # multiply + reduce, all in C

# total_notional_vectorized(p, q, 100.0)  -> ~28 ms on the reference box
```

About **28 milliseconds** — roughly **68x faster**, same answer. (You can make it faster still and allocate less by avoiding the intermediate masked arrays: `np.sum(np.where(p > threshold, p * q, 0.0))` or `np.dot(p * (p > threshold), q)` fuse more of the work, which matters at this size because, remember, the operation is memory-bound and each temporary is more bytes to move.) Notice what we did *not* do: we did not iterate, we did not index single elements, we did not call a Python function per row. We described the operation over the whole array and let one set of C loops execute it.

The conversion `np.array(prices, ...)` is itself a boundary crossing — it walks the list of boxed Python floats once, unboxing each into the packed buffer, costing on the order of 100–200 ms for three million elements. If your data *starts* its life in Python lists, that one-time conversion is worth it only if you then do real array work on it; if you convert and immediately `.tolist()` back, you have paid two boundary crossings for nothing. The right architecture is to get the data into arrays **early** (ideally load it straight into NumPy with `np.loadtxt`, `np.frombuffer`, `pandas.read_csv`, or `pyarrow`, skipping the Python-list stage entirely) and keep it there through the whole computation, crossing back to Python only at the very end when you need a scalar answer or a small result.

#### Worked example: the pipeline before and after

| Stage | Python loop | Vectorized | Speedup |
| --- | --- | --- | --- |
| 3M-row filter + multiply + sum | ~1,900 ms | ~28 ms | ~68x |
| Peak RSS (data) | ~108 MB (lists) | ~48 MB (arrays) | ~2.2x less |
| ns / element | ~630 | ~9.3 | ~68x |
| Lines of logic | 6 | 2 | — |

The memory column is worth dwelling on: the list version holds three million `float` objects plus three million `int` objects plus two pointer arrays — well over 100 MB — while the array version is two packed buffers (24 MB of `float64` plus 24 MB of `int64`). We did less work *and* used less memory *and* wrote less code. That triple win — faster, smaller, simpler — is the signature of a correct vectorization, and it is why "do it in bulk" sits where it does on the leverage ladder.

## Building and inspecting arrays: the practical toolkit

Before we talk about when *not* to do this, here is the minimal practical fluency you need: how to get data into arrays, how to read their cost, and how to confirm you are actually in the fast path.

```pycon
>>> import numpy as np
>>>
>>> # Construct:
>>> np.array([1, 2, 3])                  # infer dtype from data -> int64
array([1, 2, 3])
>>> np.zeros(5)                          # float64 by default
array([0., 0., 0., 0., 0.])
>>> np.arange(0, 1, 0.25)                # like range, but floats
array([0.  , 0.25, 0.5 , 0.75])
>>> np.linspace(0, 1, 5)                 # 5 evenly spaced points
array([0.  , 0.25, 0.5 , 0.75, 1.  ])
>>> np.full(3, 7, dtype=np.int8)         # choose the dtype explicitly
array([7, 7, 7], dtype=int8)
>>>
>>> # Inspect the cost model:
>>> a = np.arange(1_000_000, dtype=np.float32)
>>> a.dtype, a.itemsize, a.nbytes, a.shape, a.strides
(dtype('float32'), 4, 4000000, (1000000,), (4,))
>>>
>>> # Confirm you are operating in C, not Python: a ufunc returns an ndarray,
>>> # and the result dtype tells you the type stayed packed.
>>> type(a + a)
<class 'numpy.ndarray'>
>>> (a + a).dtype
dtype('float32')
```

A few habits that keep you on the fast path: always pass `dtype=` explicitly when you know it, so you do not accidentally get `float64` where `float32` or `int8` would do; use `.nbytes` to sanity-check that a large intermediate is the size you expect (a surprise `float64` where you wanted `float32` doubles your memory); and when something feels slow, check whether you have accidentally produced an `object`-dtype array — `np.array([1, "two", 3.0])` makes a `dtype=object` array, which is just a list of pointers in an ndarray trench coat, with none of the speed. An `object`-dtype array is the single sneakiest performance trap in NumPy: it looks like an array, indexes like an array, and is as slow as a list, because every element is still a boxed PyObject. If `arr.dtype` prints `object`, you are not vectorized.

## The middle rung: array.array and why it isn't enough

The speed-and-memory ladder figure showed a rung between a list of objects and a full ndarray: the standard library's `array.array`. It is worth understanding, because it isolates the *two separate things* an ndarray gives you and shows that packing the memory is necessary but not sufficient.

`array.array('d', values)` stores its elements as raw C doubles in one contiguous buffer — exactly like an ndarray's data buffer, and for the same memory savings. A million-element `array.array('d')` is 8 MB, not 36 MB; it has solved the *storage* half of the problem. What it has *not* solved is the *computation* half. The standard library gives `array.array` no vectorized arithmetic — there is no C loop that adds two `array.array`s elementwise. So the moment you want to do math, you fall back to a Python loop, which indexes the array one element at a time, and each index re-boxes the raw double into a `float` PyObject. You get the compact storage but pay the full interpreter tax on every operation.

```pycon
>>> import array, numpy as np, timeit
>>> n = 1_000_000
>>> aa = array.array('d', range(n))          # packed buffer, 8 MB
>>> nd = np.arange(n, dtype=np.float64)      # packed buffer, 8 MB
>>>
>>> aa.itemsize * n / 1e6, nd.nbytes / 1e6   # same memory
(8.0, 8.0)
>>>
>>> # array.array forces a Python loop for math -- boxes every element:
>>> timeit.timeit("sum(aa)", globals=globals(), number=100) / 100 * 1e3  # ms
9.5
>>> # ndarray runs one C loop -- no boxing:
>>> timeit.timeit("nd.sum()", globals=globals(), number=1000) / 1000 * 1e3  # ms
0.45
```

Same 8 MB of storage, but the `array.array` sum is roughly **20x slower**. Even though `sum()` iterates in C, every value it pulls from the `array.array` is *re-boxed* into a fresh `float` PyObject before the addition — because `array.array` has no vectorized add of its own, the arithmetic has to happen at the Python-object level. The ndarray, by contrast, never leaves the raw `float64` representation: its sum reduces the packed bytes directly in C. This is the clean experiment that separates the two wins: packing the buffer (the memory win, which `array.array` and ndarray share) and moving the *arithmetic* into typed C with no boxing (the speed win, which only the ndarray's ufunc machinery delivers). NumPy is fast because it does *both* — and a million tutorials that say "NumPy is fast because the data is contiguous" are telling you only half the story. Contiguity is what makes the C loop *possible* and *cache-friendly*; the C loop is what makes it *happen*. So `array.array` is genuinely useful when you need compact numeric storage and barely any math (a big lookup table, a buffer you stream to a socket, interop with C code via the buffer protocol), but the instant you need real arithmetic over the whole thing, it is the ndarray you want. The good news, from the previous section, is that you can `np.frombuffer` an `array.array` into an ndarray with zero copy — so the two compose cleanly.

## Getting into arrays without paying the list tax twice

We saw that `np.array(some_python_list)` is itself a boundary crossing — it walks the list, unboxes each PyObject, and writes the raw value into the packed buffer. That is fine as a one-time cost if you then do real array work, but if your data is *born* in a Python list you have already paid the 36-MB-of-scattered-objects price before NumPy ever touches it. The better move is to skip the Python-list stage entirely and load straight into a buffer. This is where the **buffer protocol** earns its keep.

The buffer protocol is CPython's standard way for one object to expose its raw memory to another without copying. `bytes`, `bytearray`, `array.array`, `mmap` objects, and NumPy arrays all speak it. NumPy can wrap any buffer-protocol object as an ndarray *with zero copy* using `np.frombuffer`, reinterpreting the same bytes through a chosen dtype:

```pycon
>>> import numpy as np, array
>>>
>>> # An array.array is already a packed C buffer of doubles -- zero-copy view:
>>> packed = array.array('d', [1.5, 2.5, 3.5])     # 'd' = float64, 8 bytes each
>>> view = np.frombuffer(packed, dtype=np.float64)
>>> view
array([1.5, 2.5, 3.5])
>>> view.base is packed        # no copy was made; it shares the buffer
True
>>>
>>> # Reading 24 MB of float64 from a binary file straight into an ndarray:
>>> with open("prices.f64", "rb") as f:
...     arr = np.frombuffer(f.read(), dtype=np.float64)   # one buffer, no per-row boxing
...
>>> arr.nbytes / 1e6
24.0
```

`np.frombuffer` does not allocate a million PyObjects and then throw them away; it points an ndarray at bytes that already exist. The same idea underlies the fast loaders you should prefer over a Python parse-then-convert loop: `np.loadtxt` / `np.genfromtxt` for text, `np.fromfile` / `np.frombuffer` for binary, `pandas.read_csv` and `pyarrow.csv.read_csv` for tabular data — all of them parse directly into packed columnar buffers in C, never materializing the row-by-row Python objects in between. The performance rule that follows is simple and high-leverage: **the fastest list-to-array conversion is the one you never do.** If you control the ingestion, read straight into arrays.

This is also why NumPy, pandas, Polars, and Apache Arrow interoperate so cheaply: they all agree on the same packed-buffer representation, so handing data between them is often a pointer hand-off, not a copy. A pandas `Series` of a numeric dtype *is* a NumPy buffer underneath; a Polars or Arrow column is a packed buffer NumPy can view with zero copy. Staying in "the array world" is really staying in *the shared-buffer world*, and the boundary cost we keep warning about is precisely the cost of leaving it for the land of boxed Python objects.

## Measuring honestly: how not to fool yourself

Every number in this post came from `timeit`, and the kit's discipline demands we say how to get *trustworthy* numbers, because micro-benchmarks are easy to get wrong in ways that flatter or libel a technique. A few rules I follow, all of which the [benchmarking post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) treats in full but which bite especially hard for array work:

- **Warm up, then take the best of many runs.** The first call to a NumPy function can pay one-time import and dispatch-table costs; the first touch of a freshly allocated buffer pays page faults as the OS maps physical memory. `timeit` with `number=` set high enough to run for tens of milliseconds, repeated, and reported as the *minimum* (or median), filters out these one-time and noise effects. The minimum is the run least disturbed by the scheduler.
- **Use a realistic input size.** On a 10-element array, NumPy's fixed per-call overhead (allocating the result, setting up the loop, a few microseconds) dwarfs the actual work, so a micro-benchmark there "proves" NumPy is slow — a real conclusion only for tiny inputs. Benchmark at the size you actually run, and ideally at several sizes so you can see the crossover where NumPy starts to win.
- **Mind the cache.** A 1-million-`float64` array is 8 MB and may sit largely in L3 cache, so a benchmark there measures cache bandwidth, not main-memory bandwidth. A 50-million-element array (400 MB) cannot fit in any cache and measures the true memory wall. Both numbers are valid; they answer different questions. When I quoted ~0.42 ns/element for `sum` versus a ~0.96 ns/element bandwidth estimate, the gap was largely this: the small array was cache-resident.
- **Account for the GC and allocation.** Creating large temporaries triggers allocation and, eventually, garbage collection, which can land inside your timed region and add variance. For steady-state throughput numbers, `gc.disable()` around the measurement (and re-enable after) removes that jitter — though for *honest* end-to-end numbers you should leave GC on, because production has GC on.
- **Beware constant folding and caching.** If you `timeit` `2 + 2` you measure nothing, because the value is computed once. With arrays the analog is reusing a cached result or letting the compiler/NumPy short-circuit; make sure each run does the real work on real, varying data.

#### Worked example: the crossover where NumPy starts to win

A single benchmark at one size can mislead; the honest picture is a *curve*. Here is `np.sum(arr)` versus a Python `sum(list)` across array sizes on the reference machine, reported as ns per element:

| Elements | Python `sum(list)` ns/elem | `np.sum(arr)` ns/elem | Winner |
| --- | --- | --- | --- |
| 10 | ~9 | ~600 (fixed overhead dominates) | Python |
| 100 | ~8 | ~60 | Python |
| 1,000 | ~8 | ~6 | NumPy (just) |
| 100,000 | ~8 | ~0.6 | NumPy (13x) |
| 10,000,000 | ~8 | ~0.5 | NumPy (16x) |

The lesson is that NumPy's advantage is *amortized*: its fixed per-call cost (a few microseconds) is invisible on a million elements and ruinous on ten. Somewhere in the low thousands of elements the C loop's per-element savings overtake the fixed overhead, and from there the gap only widens as the array grows and SIMD plus prefetching kick in. This is the quantitative basis for the "is it large enough?" branch in the decision tree below: vectorization is a bulk discount, and you have to buy enough to clear the cover charge.

## Stress-testing the claim: when the buffer doesn't fit

A good engineer does not just learn the happy path; they probe where it breaks. So let me stress-test the central claim — "one C loop over a packed buffer is fast" — against the cases that strain it, because each one teaches you something about *why* it is fast.

**What if the array doesn't fit in RAM?** A 16 GB box cannot hold a 40 GB array, and NumPy will raise `MemoryError` or thrash swap (which is catastrophically slow — disk is ~10,000x slower than RAM). NumPy's model is *in-core*: it assumes the buffer fits in memory. When it does not, the ndarray is the wrong tool and you climb sideways on the ladder — to chunked processing (loop over slices that *do* fit, in C each), to memory-mapped arrays (`np.memmap`, which pages the buffer from disk on demand), or out-of-core engines like Dask, Polars streaming, or DuckDB that process the data in passes without ever holding all of it. The packed-buffer win still applies *within each chunk*; you just orchestrate the chunks.

**What if the operation is compute-bound, not memory-bound?** Our whole memory-wall argument assumed low arithmetic intensity. Some operations are different: `np.exp(a)`, `np.sin(a)`, or a matrix multiply do many FLOPs per byte, so they are limited by the CPU's arithmetic throughput, not bandwidth. For those, the dtype-halving trick gives less of a speedup (you are not bandwidth-limited), but SIMD and using all cores matter *more* — which is exactly why `np.dot` and friends call internally into multi-threaded BLAS libraries. The general principle from the [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) holds: know which regime you are in before you optimize, because the right lever differs.

**What if the data is strided or non-contiguous?** A view created by slicing every other element (`arr[::2]`) or transposing a 2D array shares the buffer but walks it with a non-unit stride. The C loop still runs, but it can no longer stream contiguous cache lines or SIMD cleanly, so it can be several times slower than the contiguous case — wasting most of each cache line it touches, just like the list. The fix when it matters is `np.ascontiguousarray(arr)`, which makes a packed copy you can then stream; whether the copy pays for itself depends on how many times you reuse it. This is the doorway into the strides-and-layout post later in the track; the seed here is that *contiguity is not automatic*, and a view can quietly drop you off the fast path.

**What if you only run it once on small data?** Then none of this matters, and reaching for NumPy is over-engineering. A 500-row, one-shot script that finishes in 8 ms does not need a packed buffer; the time you would spend vectorizing it is worth more than the milliseconds you would save. Pull the lever where the runtime actually is.

## The decision: vectorize or stay in Python?

NumPy is a lever, and every lever is a trade-off. The whole discipline of this series is pulling the *right* lever, which means knowing when this one does not apply. Vectorization shines on a specific shape of problem, and it actively hurts on others. The decision tree below captures it: homogeneous, numeric, processed-in-bulk data belongs in NumPy; heterogeneous, tiny, or branchy work belongs in plain Python.

![A decision tree asking what shape the data is, branching to vectorize with ndarray and ufuncs when it is homogeneous numeric data over a million plus rows, or stay in plain Python when the data is mixed, tiny, or branchy](/imgs/blogs/numpy-from-first-principles-the-ndarray-and-why-its-fast-8.png)

**Reach for NumPy when:**

- The data is **homogeneous and numeric** — all `float64`, all `int32`, not a mix of strings, dicts, and numbers. Heterogeneous records belong in a dataframe (pandas/Polars) or stay as Python objects.
- You process it **in bulk** — the same operation over many elements, expressible as array arithmetic, ufuncs, reductions, or masks. If your per-element logic is a tangle of branches and early exits that resists array form, vectorizing fights you.
- The arrays are **large enough to amortize the overhead** — NumPy has a fixed per-call cost (a few microseconds to dispatch, allocate the result, set up the loop). On a 10-element array that overhead swamps the win and a Python comprehension is genuinely faster. The crossover is typically somewhere in the dozens-to-hundreds of elements; below it, do not bother. Vectorization is a *bulk* discount.

**Stay in plain Python (or reach for a different lever) when:**

- The data is **heterogeneous or structured** (mixed types, nested, variable-length). Forcing it into an `object`-dtype array gets you the slowness of Python with the awkwardness of NumPy.
- The collection is **tiny** — a handful of elements where the constant overhead of the array machinery exceeds the loop you are replacing.
- The logic is **inherently sequential or branchy** — each step depends on the last (a running state machine, a parser, an iterative algorithm that cannot be expressed as elementwise array ops). If the algorithm cannot be vectorized but is still a numeric hot loop, the answer is not `np.vectorize` — it is a compiled kernel (Numba `@njit`, Cython, or a small C/Rust extension), which is the *next* rung of the ladder and the subject of the native-acceleration track.
- You only need it **once** — a one-shot script over 500 rows that runs in 8 ms does not need vectorizing. Spend the engineering where the runtime is.

Here is the trade-off as a table you can act on:

| Situation | Right tool | Why |
| --- | --- | --- |
| 1M+ numeric rows, elementwise math | NumPy ufuncs / reductions | One C loop, SIMD, packed memory |
| 1M+ rows, mixed columns, group-by | Polars / pandas | Columnar + query engine; cross-link the dataframe post |
| Tiny collection (< ~50 elems) | Plain Python / comprehension | Array overhead exceeds the win |
| Numeric but branchy / sequential | Numba `@njit` or Cython | Compiles the loop; vectorization can't express it |
| Heterogeneous objects | Plain Python / dataclasses | Not array-shaped at all |
| Looks vectorized but uses `np.vectorize` | Rewrite as ufuncs, or `@njit` | `np.vectorize` is a Python loop in disguise |

## Case studies and real numbers

A few grounded results to calibrate expectations, with sources and the honesty to mark what is approximate.

**The classic `iterrows` versus vectorized story.** This is the most reported speedup in the data world, and it is real: across countless blog posts and the pandas documentation's own guidance, replacing a `df.iterrows()` Python loop with a vectorized column expression on a few-million-row dataframe typically yields **50x to 1000x** speedups. The mechanism is exactly what we derived — `iterrows` yields a boxed Python `Series` *per row*, paying the full per-element interpreter tax, while a vectorized column operation runs one C loop over the packed column buffer. The pandas docs explicitly recommend vectorization over `apply` over `iterrows`, in that order, for this reason. The exact factor depends on row count and operation, so I quote it as a range, not a precise number.

**NumPy's own reductions are tuned and SIMD-aware.** Since NumPy 1.17 and increasingly through the 1.2x series, the core ufunc loops use SIMD intrinsics (SSE/AVX on x86, NEON on ARM) for the common dtypes, dispatched at runtime based on the CPU. This is why a simple `arr.sum()` on contiguous `float64` data hits a meaningful fraction of memory bandwidth on the reference machine — you are not just in C, you are in *vectorized* C. NumPy's release notes document the SIMD coverage growing version over version; the practical takeaway is that staying contiguous and using a native dtype lets you ride that work for free.

**The Faster CPython effort narrowed — but did not close — the gap.** CPython 3.11's specializing adaptive interpreter (PEP 659) made pure-Python loops meaningfully faster — roughly **10–60% faster** on the standard benchmark suite versus 3.10, with more in 3.12 and 3.13. That is a real and welcome win for Python-level code. But notice the scale: tens of percent, versus the **100x** of vectorization. Even an optimally specialized Python loop is still a Python loop — still boxing, still refcounting, still moving 4.5x the bytes through scattered memory. Faster CPython makes the *baseline* faster; it does not change the verdict that bulk numeric work belongs in packed buffers and C loops. The lesson compounds rather than competes: a faster interpreter plus correct vectorization is better than either alone.

**The "object-dtype surprise" in the wild.** A failure mode I have debugged more than once: a dataframe column that *looks* numeric but is secretly `object`-dtype, usually because a few rows had a stray string, a `None`, or a Python `Decimal`, and pandas widened the whole column to `object` to hold them. Every operation on that column then runs at list speed — the buffer is full of pointers to scattered PyObjects, not packed numbers — and a "vectorized" `df['x'] * 2` quietly degrades to a per-element Python multiply. The symptom is a vectorized expression that is mysteriously 50–100x slower than its neighbors; the diagnosis is one `df.dtypes` or `arr.dtype` check; the fix is `pd.to_numeric(col, errors='coerce')` or an explicit `.astype(np.float64)` after cleaning the bad rows. The general rule this reinforces: a packed numeric dtype is not something you get for free by *naming* something an array — it is a property you must confirm. The fast path is `int64`/`float64`/`int32` and friends; `object` dtype is the slow path wearing the array's clothes, and the only way to know which one you are on is to look.

**Why the Rust and C data tools are NumPy-shaped underneath.** The modern high-performance Python data stack — Polars, pyarrow, pydantic-core, the numeric guts of pandas — is built in Rust or C/C++, and every one of them uses the same core idea we derived here: store homogeneous data in contiguous typed buffers, and process it with tight compiled loops that never touch a per-element Python object. Polars and Arrow use a *columnar* layout (all the values of one column packed together) precisely so that operations on a column are one streaming C/Rust loop over a contiguous buffer — the dataframe generalization of the ndarray. When you read that "Polars is 5–30x faster than pandas on a group-by," the mechanism is not magic; it is more of exactly what this post is about (packed buffers, compiled loops, SIMD, and using all cores) plus a query optimizer that avoids materializing temporaries. The ndarray was the first place most Python programmers met this idea; it is now the foundation of the whole fast-data ecosystem.

## When to reach for this (and when not to)

Let me be blunt with the recommendations, because a lever you pull at the wrong time wastes effort or makes things worse.

**Do reach for NumPy** when you have a numeric loop over thousands or millions of homogeneous values, expressible as array arithmetic. This is the single highest-leverage rewrite in most data and scientific Python: one line, often 50–100x, frequently *less* memory, and you stay in pure Python (no build step, no extra language). When you profile a pipeline and the hot path is a numeric `for`-loop, vectorizing is almost always the first thing to try.

**Do not reach for NumPy** when:

- **The collection is tiny.** A loop over 20 items does not need an array; the per-call overhead loses. Measure if unsure, but trust the instinct that arrays are a *bulk* tool.
- **The data is heterogeneous.** Mixed types belong in a dataframe or plain Python objects, not an `object`-dtype array that has all of NumPy's awkwardness and none of its speed.
- **You would only cross the boundary to come right back.** Converting to an array, doing one trivial thing, and `.tolist()`-ing back pays two boundary crossings to save nothing. Vectorize a *region* of work, not a single operation surrounded by Python.
- **The logic genuinely cannot be vectorized.** Branchy, stateful, sequential numeric loops are real, and for those `np.vectorize` is a trap. The right answer is a compiled kernel — Numba `@njit` first (no build step, often 100x on exactly this kind of loop), then Cython or a Rust/C extension if you need more control. That is the next rung up the ladder, and the [native-acceleration track] of this series covers it.
- **You have not measured.** If a function is 2% of your runtime, Amdahl's law caps your total win at 2% no matter how brilliantly you vectorize it. Profile first, vectorize the hot path, re-measure. Don't guess; measure.

And one piece of standing advice that ties the whole post together: get your data into arrays *early* and keep it there. Most NumPy performance bugs are not slow ufuncs — the ufuncs are fine. They are accidental boundary crossings: a stray Python loop over an array, a `.tolist()` in a hot path, an `object`-dtype array nobody noticed, a `np.vectorize` that looked like magic. Stay in the array world from load to final aggregation, cross back to Python only for the scalar answer at the end, and the C loops will do their job.

## Key takeaways

- **An ndarray is one contiguous, typed, fixed-size buffer plus thin metadata** (dtype, shape, strides, data pointer). A `list` of ints is an array of pointers to scattered 28-byte PyObjects. For a million ints that is roughly **8 MB versus 36 MB**.
- **A Python loop pays a per-element tax** — opcode dispatch, box/unbox, refcounting, type dispatch — of which only the arithmetic is real work. A NumPy ufunc runs **one C loop** that does none of that ceremony, which is the ~**100x** gap (measured: ~58 ns/elem loop vs ~0.42 ns/elem `np.sum`).
- **Elementwise array math is memory-bound, not compute-bound.** Arithmetic intensity is around $1/24$ FLOP per byte for `a + b`; the CPU waits on memory. So packing data tightly and choosing a smaller dtype are direct, first-order speedups.
- **dtype is a performance knob.** `float32` moves half the bytes of `float64` and runs roughly **2x faster** on memory-bound work — at a real precision cost. Choose it deliberately.
- **Contiguity lets the CPU stream and SIMD the data** — eight `float64` per 64-byte cache line, four-to-eight lanes per AVX instruction. A list's pointer chase defeats both the prefetcher and SIMD.
- **The boundary is where speed dies.** `.tolist()`, per-element indexing, iterating an array in Python, and `np.vectorize` all re-box every value and pay the full interpreter tax. **Never loop over an ndarray in Python.**
- **`np.vectorize` is not vectorized** — it is a Python loop with a friendly API (measured ~185x slower than the real expression). Use array primitives; if the logic resists them, compile it with Numba/Cython, do not `np.vectorize` it.
- **Get into arrays early, stay in them, cross back once.** Most NumPy slowdowns are accidental boundary crossings, not slow ufuncs.

## Further reading

- **NumPy documentation — "The N-dimensional array (ndarray)"** and **"Universal functions (ufunc)"** — the canonical reference for dtype, shape, strides, and the elementwise loops. The `np.vectorize` docstring states plainly that it is a convenience, not a performance tool.
- **NumPy "Internals" and the C-API buffer/array docs** — how the data buffer, strides, and dtype fit together; the basis for views and zero-copy interchange.
- **"High Performance Python" (Gorelick & Ozsvald, 2nd ed.)** — chapters on lists vs arrays, memory layout, and vectorization with measured before→after numbers in the same spirit as this post.
- **The CPython `dis`, `timeit`, and `sys.getsizeof` docs** — your instruments for seeing the bytecode, measuring honestly, and counting bytes.
- **PEP 659 (specializing adaptive interpreter)** and the Faster CPython notes — how much the *interpreter* sped up in 3.11–3.13, and why that is tens of percent, not the 100x of vectorization.
- Within this series: [why Python is slow and what "fast" actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) (the interpreter overhead we exploit here), [idiomatic fast Python: comprehensions, generators, and builtins](/blog/software-development/python-performance/idiomatic-fast-python-comprehensions-generators-and-builtins) (the prior rung — pushing loops into C builtins), and [the hidden cost of objects, attributes, and dynamic dispatch](/blog/software-development/python-performance/the-hidden-cost-of-objects-attributes-and-dynamic-dispatch) (the per-object boxing tax we just deleted in bulk).
- Out of series: [the roofline model — compute-bound vs memory-bound](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) — the same arithmetic-intensity framework that explains why elementwise array math is limited by memory bandwidth, generalized to GPUs and the whole memory hierarchy.

Next in the Vectorize track we leave the single ndarray behind and learn to *express* whole computations as array expressions — broadcasting, ufuncs, reductions over axes, boolean masking, and fancy indexing — translating real Python loops into vectorized form step by step. The ndarray is the noun; vectorization is the grammar. Once both are second nature, the 90-second report really does become a 2-second one — and you will know, to the nanosecond, why.

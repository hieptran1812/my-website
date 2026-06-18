---
title: "The Native Acceleration Landscape: When to Leave Pure Python"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "A map of every way to make Python run at native speed, and the discipline to pick the right rung, profile first, rewrite the hot 1 percent, and pay the FFI toll once over big work, not a million times over tiny work."
tags:
  [
    "python",
    "performance",
    "optimization",
    "native-code",
    "numba",
    "cython",
    "rust",
    "ffi",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-1.png"
---

There is a particular kind of meeting that happens after a service has been slow for a quarter. Someone has finally profiled the thing, the numbers are on a slide, and a senior engineer leans back and says the words that have launched a thousand doomed rewrites: "We should just rewrite it in C." Or Rust. Or "drop the hot part into a native extension." The room nods. It feels decisive. It feels like the grown-up answer. And about half the time it is exactly the wrong move, because the person saying it has not yet asked the only three questions that matter: how hot is the hot path *really*, can it be vectorized without writing a single line of native code, and how much native code is the team actually willing to own and maintain for the next five years.

This post is the map for that decision. It is the opening post of the "go native" track in this series, and its job is not to teach you Numba or Cython or Rust in depth — the next four posts do that, one tool each. Its job is to give you the landscape so that by the time you reach for a specific tool you already know *why* that rung and not another, and you already know the one trap that sinks most native-acceleration efforts: paying the cost of crossing from Python into native code so many times that the crossing costs more than the work. We are going to be precise about that. We are going to put a number on it, in nanoseconds, and derive the rule that falls out of the number.

Here is the frame, and it is the same frame that runs through this entire series. You do not guess where the time goes; you measure. You find the hot path. You pick the lever with the most leverage for the least effort, and the levers come in a fixed order: do less work (better algorithm and data structure), then do it in bulk (vectorize with NumPy or Polars), *then* compile the hot 1 percent in native code, and only then reach for more cores. Native code is rung three on that ladder, and you do not get to rung three by skipping rungs one and two. If you have not read [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), start there; it sets up the whole cost model. And the question of *when* something is worth optimizing at all — the Amdahl's-law arithmetic we lean on heavily below — lives in [the mental model of performance, latency numbers, and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop).

![decision tree showing the native acceleration ladder branching from a profiled hot path into a vectorizable path leading to NumPy and a scalar loop path leading to Numba, Cython, ctypes, pybind11, and Rust](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-1.png)

By the end of this post you will be able to look at a profiler output and route the hot function to the right rung of the native ladder in about thirty seconds, justify that choice with an Amdahl bound and an FFI cost estimate, and explain to the engineer who wants to rewrite everything in C exactly why rewriting 1 percent gets the same speedup for a hundredth of the effort. Let me set the running numbers up front so every benchmark below is anchored to something real.

## The machine, and what "native" actually buys you

Every measured number in this post is on one named machine so you can reason about it consistently: **an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM**, a fairly ordinary cloud or workstation configuration as of this writing. Where I quote a speedup I will give the input size and dtype, because a speedup with no setup is a marketing number, not an engineering one. Where I have not literally run a microbenchmark I will say so and frame it as a typical range. The point of naming the machine is that "native code is faster" is not a useful sentence on its own — faster by how much, on what work, measured how. We are going to insist on the number every time.

So start with the foundational question: what does "go native" even mean, and what does it buy you that pure Python cannot? When you run a Python `for` loop that adds two lists of a million numbers, CPython does an astonishing amount of work *per element*. It fetches the next bytecode instruction, dispatches on it through the giant evaluation loop, looks up the `+` operation by walking the object's type to find its numeric protocol slot, unboxes two `PyObject` integers into machine integers, adds them, allocates a *new* boxed `PyObject` for the result, adjusts reference counts on everything it touched, and loops back to do it all again. That per-element overhead — the interpreter dispatch, the boxing and unboxing, the reference-count bookkeeping, the dynamic type check — is the tax that "native" removes. A native loop over a typed C array does the add and nothing else: no dispatch, no boxing, no refcount, just the arithmetic instruction the CPU was built for. That is the whole game.

Concretely, going native buys you five things, and it is worth naming them because different tools deliver different subsets:

- **No evaluation loop.** The bytecode interpreter is gone; the hot loop is real machine code, not instructions interpreted one at a time.
- **No boxing.** Numbers live as raw `int64` or `float64` in a packed buffer, not as individual heap-allocated `PyObject`s each with a header, a type pointer, and a refcount.
- **Static types.** The compiler knows at compile time that this is a `double`, so there is no per-operation type dispatch — the add is a single instruction, decided once.
- **SIMD.** With static types and a contiguous buffer the compiler (or the JIT) can emit vector instructions that process 4, 8, or 16 elements per clock instead of one.
- **A released GIL.** Inside a native loop that touches no Python objects, you can release the Global Interpreter Lock and let other threads run truly in parallel — something pure Python CPU-bound code can never do.

Notice that the first four are about *single-thread* speed and the fifth is about *parallelism*. That distinction matters when we talk about which rung gives you what. NumPy gives you the first two and some of the third and fourth, but it does not run *your* logic — only its built-in operations. Numba and Cython and Rust give you all five, including the ability to run your own scalar logic at machine speed with the GIL released. That is the spectrum we are mapping.

It is worth pausing on the magnitude of the per-element tax, because the size of it is what makes native acceleration worthwhile in the first place. On the named machine, a single iteration of a trivial Python `for` loop that does `total += x` measures around 30–50 ns of pure interpreter overhead — the bytecode fetch, the dispatch, the boxed-int add, the refcount churn — *before* you count the actual addition, which the CPU does in well under a nanosecond. So the interpreter is spending something like 50 to 100 times the cost of the arithmetic just to *arrive* at the arithmetic. A native loop deletes essentially all of that overhead: the same add in compiled C or Rust is a single machine instruction with no surrounding ceremony, often fused with neighbors by SIMD so that the effective cost is a fraction of a nanosecond per element. That two-orders-of-magnitude gap between "the work" and "the ceremony around the work" is the entire economic case for going native, and it is also why the gap *collapses* the moment your per-element work gets expensive: if each element already does a millisecond of real computation, the 50 ns of interpreter overhead is noise and native code buys you almost nothing. Native acceleration pays off precisely when the per-element work is *small* and the loop count is *large* — when the ceremony dominates the work. Keep that condition in mind; it reappears in every decision below.

## The ladder, rung by rung

Picture the choices as a ladder where each step down trades more build complexity and more native code to maintain for more raw control and, often, more speed. The figure at the top of this post lays it out as a decision tree; here is the ladder in words, from the rung you should try first to the rung you reach only when you genuinely need it.

**Rung 0 — vectorize with NumPy or Polars.** This is barely "native" in the sense most people mean, but it belongs at the bottom of the ladder because it gets you native-speed loops *without writing any native code*. NumPy's operations are already compiled C running over packed, typed buffers. If your hot work is array math — elementwise arithmetic, reductions, matrix operations, boolean masking — you stop here. No build step, no new language, full portability, and you stay in pure Python. The series covers this in depth across the vectorization track; the punchline for *this* post is that vectorizing is the rung most "we should rewrite it in C" proposals should have tried first and didn't. We covered the wall you hit when even vectorized NumPy is not enough — when you are memory-bandwidth-bound and allocating temporaries — in [when NumPy isn't enough: numexpr, bandwidth, and avoiding temporaries](/blog/software-development/python-performance/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries). That post is the natural rung just *before* this one: when `numexpr` and `out=` and chunking have run out, native is next.

**Rung 1 — Numba.** A just-in-time (JIT) compiler that turns a subset of Python and NumPy into machine code at runtime, triggered by a single `@njit` decorator. This is the least-effort native rung. You keep writing Python — actual Python, with loops and `if` statements and array indexing — and Numba compiles it to native code on the first call. No separate language, no build system, no `.c` files. It is the right answer for a scalar, branchy, numeric hot loop that does not vectorize cleanly. The cost is a first-call compilation delay and a real but bounded list of "things Numba cannot compile."

**Rung 2 — Cython.** A language that is a superset of Python: you write Python-looking code in a `.pyx` file, add C type declarations (`cdef int`, typed memoryviews like `double[:, ::1]`), and Cython translates it to C and compiles it to a real extension module. This is the rung when you need more control than Numba gives — when you want to call C libraries directly, manage memory deliberately, release the GIL around a specific block, or compile something Numba refuses. The cost is a build step (you ship and compile `.pyx`) and a genuine learning curve around the typed dialect.

**Rung 3 — a C or C++ extension via the FFI.** Sometimes the native code already exists — a battle-tested C library for image decoding, cryptography, or numerical solvers — and your job is not to *write* native code but to *call* it. That is the foreign function interface (FFI): the bridge between Python and a compiled library. You can call C with `ctypes` (no build step at all — you load a shared library at runtime) or `cffi`, wrap C++ with `pybind11`, or write directly against the CPython C-API. This rung is about *integration*, not authorship. The decision driver here is "does the C/C++ code I need already exist?"

**Rung 4 — Rust via PyO3 and maturin.** When you are going to write and *own* a substantial native module — not a 40-line kernel but a real component you will maintain for years — Rust is the modern default. PyO3 binds Rust to Python; maturin builds the wheel. You get C-class speed, you can release the GIL inside the Rust code for true parallelism, and crucially you get memory safety: no segfaults, no use-after-free, the class of bugs that make C extensions a maintenance nightmare. The whole modern Python toolchain has gone this way — Polars, pydantic-core, ruff, tokenizers, and uv are all Rust extensions. The cost is the steepest learning curve on the ladder and a real build pipeline.

One way to internalize the ladder is to notice that the rungs are ordered by a single hidden variable: **how much native code you have to write and own.** Rung 0 (NumPy) is zero lines of native code — you compose pre-built native operations. Rung 1 (Numba) is zero lines of *separate* native code — you write Python and a compiler produces the native version, which you never see or maintain. Rung 2 (Cython) is some native-flavored source you write and compile — a `.pyx` file you own. Rung 3 (FFI) is native code you *call* but, ideally, did not write — someone else maintains the C library; you maintain only the thin binding. Rung 4 (Rust) is native code you fully author and own — a real crate, a real build, a real component. As you descend, the line "native code I am responsible for" grows from nothing to a substantial module, and that responsibility — the building, the debugging, the onboarding, the security surface — is the true cost of each rung, far more than the raw lines of code. The speed at the bottom four rungs is roughly equal; the *ownership* is what separates them, and ownership is what you should be deciding about.

That is the landscape. Now the discipline: how to decide *which* rung, and that decision is governed by two pieces of arithmetic that almost everyone skips.

## The first law: Amdahl bounds what a native rewrite can buy

Before you rewrite anything, you have to know the most a rewrite *could possibly* help — because if the ceiling is low, the effort is wasted no matter how fast your native code is. The tool for this is Amdahl's law, and it is the single most important sentence in performance engineering: **your speedup is bounded by the fraction of time you do not touch.**

Suppose your program spends a fraction $p$ of its time in the hot path you are about to make native, and the rest of the time, $1 - p$, in code you leave alone. If you make the hot path $s$ times faster, the overall speedup is

$$ S = \frac{1}{(1 - p) + \dfrac{p}{s}} $$

Stare at that denominator. As your native code gets arbitrarily fast — as $s \to \infty$ — the term $p/s$ goes to zero and the speedup approaches a hard ceiling:

$$ S_{\max} = \frac{1}{1 - p} $$

This is the law that turns "we should rewrite it in C" from a feeling into a calculation. Suppose profiling shows the hot function is 50 percent of runtime. Even if you make it *infinitely* fast — free, zero time — the best you can do is $1/(1 - 0.5) = 2\times$. A 2x ceiling. Is a C extension, with its build pipeline and its segfault risk and its onboarding cost, worth a *maximum* of 2x? Almost never. Now suppose the hot function is 95 percent of runtime. The ceiling is $1/(1 - 0.95) = 20\times$, and if your native rewrite delivers a realistic 50x on that fraction, you get

$$ S = \frac{1}{0.05 + \frac{0.95}{50}} = \frac{1}{0.05 + 0.019} = \frac{1}{0.069} \approx 14.5\times $$

That is a real, life-changing speedup, and now the native effort is clearly justified. The difference between the two cases is not how fast your native code is — it is $p$, the fraction you are optimizing. **Native code is worth it when the hot path dominates runtime.** That is why you profile first. Not as a ritual, but because $p$ is the input to the only equation that tells you whether to proceed.

There is a corollary that names the most common waste in our field. If you optimize a function that is 2 percent of runtime, your ceiling is $1/(1 - 0.02) \approx 1.02\times$ — a 2 percent improvement, the best case, for whatever effort you spent. People do this constantly. They optimize the function they find interesting, or the function whose name they recognize, instead of the function the profiler points at. Amdahl is the antidote. Profile, read off $p$, compute the ceiling, and if the ceiling is under, say, 1.3x, walk away — the hot path is somewhere else.

A second corollary governs *where on the ladder the work is worth it*. The deeper a rung, the more fixed cost it carries — a build pipeline, a new language, a native module to maintain — so the ceiling has to be high enough to amortize that fixed cost. A 1.5x ceiling might justify an afternoon of NumPy vectorization (rung 0, near-zero fixed cost) but would never justify standing up a Rust build pipeline (rung 4, large fixed cost). This is why the ladder is ordered the way it is: the cheap rungs are worth pulling even for modest ceilings, while the expensive rungs need a big $p$ — a genuinely dominant hot path — to pay back their setup. So the two numbers you read off the profiler, the hot fraction $p$ and the *shape* of the hot path, jointly pick not just whether to go native but *how far down* to go. High $p$ plus a scalar branchy shape justifies descending toward Cython or Rust; a modest $p$ plus array-math shape says vectorize and stop. The arithmetic and the shape together route you to exactly one rung.

#### Worked example: should we go native on the nightly report?

A nightly reporting job runs in 600 seconds on the 8-core box. The team wants to rewrite the scoring function in Rust. Before anyone writes a line of Rust, we profile with `cProfile` (covered in detail in [CPU profiling: cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path)) and find:

- `score_rows` — 480 seconds, cumulative. So $p = 480 / 600 = 0.80$.
- Everything else (I/O, formatting, DB writes) — 120 seconds.

The Amdahl ceiling is $1/(1 - 0.80) = 5\times$. That is the absolute best case: 600 s down to 120 s, if the scoring function became literally free. A realistic native rewrite of `score_rows` might give 40x *on that function*, taking it from 480 s to 12 s. Plug in $s = 40$:

$$ S = \frac{1}{0.20 + \frac{0.80}{40}} = \frac{1}{0.20 + 0.02} = \frac{1}{0.22} \approx 4.5\times $$

So 600 s becomes about 132 s. That is a 4.5x win, near the ceiling, and it justifies the work. But notice the residual: after the rewrite, the *new* hot path is the 120 seconds you didn't touch — it is now 91 percent of the new 132-second runtime. The next optimization target moved. This is how the optimization loop actually goes: you climb a rung, you re-measure, and the bottleneck moves. The figure below contrasts the disciplined "rewrite the 1 percent" move against the heroic "rewrite 100 percent in C" move, which spends months to hit the *same* Amdahl ceiling.

![before and after comparison contrasting porting only the hot one percent kernel to native code in a day against rewriting the entire application in C over months for the same speedup ceiling](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-3.png)

The lesson the figure is making concrete: the speedup ceiling is set by $p$, not by how much code you rewrite. Rewriting 100 percent of the application in C does *not* raise the ceiling above what rewriting the hot 1 percent achieves, because the cold 99 percent was never the bottleneck. All it does is multiply your effort, your build complexity, and your bug surface, while throwing away the Python glue, the rich library ecosystem, and the readability that made the codebase maintainable. **Rewrite the 1 percent that is hot. Keep the 99 percent that is fine.** That is the golden rule of native acceleration, and Amdahl is its proof.

## The second law: the FFI boundary has a fixed per-call cost

Amdahl tells you *whether* to go native. The second piece of arithmetic tells you *how* to structure the native code once you decide to — and getting it wrong is the single most common way native-acceleration efforts end up *slower* than the pure Python they replaced. The culprit is the cost of crossing the boundary between Python and native code.

Every time Python calls into native code, work happens at the boundary that has nothing to do with your actual computation. The interpreter has to take your Python objects — boxed `PyObject`s — and marshal them into the C types the native function expects: unbox the integers, extract the pointer and length from an array, convert a Python string into a C `char*`. It may have to acquire or release the GIL. The native function runs. Then the result has to be marshaled *back*: a raw C `double` gets boxed into a fresh `PyObject`, a returned buffer gets wrapped, refcounts get set. None of that is your computation. It is pure overhead, and crucially it is a *fixed cost per crossing*, largely independent of how much work the native function does once it is inside.

![dataflow graph of a single FFI crossing where a Python call marshals arguments to C types and checks the GIL in parallel, runs native code with no eval loop, then marshals the result back to a Python object](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-4.png)

Let me put a number on it. Call the per-crossing overhead $c$ — the marshaling-in, GIL handling, and marshaling-out, measured in nanoseconds. The actual useful work per call we'll call $w$. If you make one native call that processes $N$ items, your total time is roughly $c + N \cdot t$, where $t$ is the per-item native cost. Amortized over $N$ items, the overhead per item is $c / N$ — it vanishes as $N$ grows. But if you make $N$ native calls, one per item, your total is $N \cdot (c + t)$, and now you pay the full overhead $c$ *every single time*. The ratio between the two strategies is

$$ \frac{N(c + t)}{c + N t} \approx \frac{c + t}{t + c/N} $$

When $N$ is large, $c/N \to 0$, and the per-element strategy is slower by a factor of roughly $(c + t)/t = 1 + c/t$. If the boundary overhead $c$ is large relative to the per-item work $t$ — which it almost always is for cheap per-item work — this ratio is enormous. You can easily make native code that is *slower than pure Python* by calling it a million tiny times.

How big is $c$ in practice? It depends on the FFI mechanism. As measured ranges on the 8-core box (these are typical, not exact, and depend heavily on argument types):

- A pure-Python function call: roughly **40–80 ns** of frame setup and teardown.
- A `ctypes` call into a simple C function with a couple of scalar arguments: roughly **150–300 ns** per call — `ctypes` does type conversion dynamically and it is not cheap.
- A `cffi` call (API mode): roughly **50–150 ns**, faster than `ctypes` because the binding is compiled.
- A Numba `@njit` function called from Python: roughly **150–300 ns** of dispatch the first time the types are seen, then cheaper on the warm path; called from *inside* another `@njit` function it is essentially free (it is just a native call).
- A pybind11 or Cython call with typed arguments: roughly **50–200 ns**, depending on how much conversion the signature requires.

These are the same order of magnitude — call it **a couple hundred nanoseconds per crossing** as a rule of thumb. Now do the arithmetic. If your per-item work $t$ is, say, 5 ns (a few arithmetic operations), and your crossing cost $c$ is 200 ns, then calling native once per item makes the *overhead* 40 times the work. Forty times. Your "native acceleration" runs at one-fortieth the speed it should, and quite possibly slower than the Python loop you were trying to beat, because at least the Python loop didn't marshal types across a boundary on every iteration.

![before and after comparison showing a million tiny native calls where the two hundred nanosecond per crossing overhead dominates versus one native call over the whole array where the crossing cost is paid once and amortized](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-6.png)

The rule that falls out is simple and it is the most important practical guidance in this whole post: **few crossings over big work, never many crossings over tiny work.** Push the *loop* into the native code. Hand the native function a whole array — a pointer and a length — and let it iterate inside, where iteration is free. Do not iterate in Python and call native per element. This single principle is why the right native interface always takes a buffer, never a scalar; why Numba's `@vectorize` and `@guvectorize` exist (to let you write per-element logic that Numba then wraps in a native loop); why every good C extension API processes batches.

There is a deeper reason the buffer interface wins beyond just amortizing the fixed overhead, and it is worth naming because it changes how you design the boundary. When you hand native code a whole contiguous buffer, you are also handing it the *ability to be fast inside* — the data is already in the packed, typed layout the CPU wants, contiguous in memory so the hardware prefetcher can stream it into cache, and uniform in type so the compiler can emit SIMD. When you instead pass one Python object at a time, every crossing not only pays the marshaling toll but also *destroys* that layout advantage: the native function receives one isolated scalar, can't vectorize across elements it never sees together, and can't benefit from cache locality because it's invoked fresh each time. So per-element crossing is doubly bad — it pays the fixed toll a million times *and* throws away the bulk-processing speedup that was the entire reason to go native. The buffer interface is not a micro-optimization; it is the difference between getting the native speedup and not getting it at all.

This also explains a subtle design rule for native APIs: prefer interfaces that take and return *arrays* (or other bulk containers) and that do as much work as possible per call. If you find yourself wanting a native function that takes a Python callback and invokes it per element — calling *back* into Python from inside the native loop — you have reintroduced the crossing cost on every iteration, in the worst possible place, and your native loop now runs at Python speed. The fix is always to move more of the logic *into* the native side so the boundary is crossed rarely. When you genuinely cannot — when the per-element logic truly needs Python — that is a strong signal the work doesn't belong in native code at all, and you should reconsider whether you're on the right rung.

#### Worked example: the FFI overhead that ate the speedup

A team has a scalar transformation — `transform(x)` does about 8 nanoseconds of arithmetic per call. They write it in C, expose it via `ctypes`, and call it in a Python loop over 10 million elements:

```python
import ctypes

lib = ctypes.CDLL("./libtransform.so")
lib.transform.argtypes = [ctypes.c_double]
lib.transform.restype = ctypes.c_double

def run_per_element(data):
    out = []
    for x in data:               # 10M Python iterations
        out.append(lib.transform(x))   # 10M FFI crossings
    return out
```

On the 8-core box, `lib.transform` measured at about **220 ns per call** through `ctypes` — almost entirely the marshaling, since the work is only 8 ns. Over 10 million elements:

- Per-element FFI: $10^7 \times 220\,\text{ns} = 2.2$ seconds, *plus* the Python loop overhead (another ~0.6 s for 10M iterations), so roughly **2.8 seconds**.
- The pure-Python version, doing the arithmetic inline with no FFI, ran in about **1.9 seconds**.

The native rewrite was *slower* — 2.8 s versus 1.9 s — a 1.5x regression, because the boundary cost dwarfed the 8 ns of real work. The team "went native" and went backward.

Now restructure so the C function takes the whole buffer and loops inside:

```python
import numpy as np
import ctypes

lib = ctypes.CDLL("./libtransform.so")
# C signature: void transform_array(double* data, double* out, long n)
lib.transform_array.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_long,
]

def run_whole_array(data):       # data is a contiguous float64 ndarray
    out = np.empty_like(data)
    n = data.size
    lib.transform_array(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
    )
    return out
```

Now there is exactly **one** crossing. The 220 ns of overhead is paid once and amortized over 10 million elements — $220\,\text{ns} / 10^7 = 0.000022$ ns per element, utterly negligible. The native loop does the 8 ns of work per element with no boxing and SIMD where the compiler can manage it, finishing in roughly **40 milliseconds**. That is the same C code, the same algorithm — restructured from 10 million crossings to one. The result went from a 1.5x *regression* to roughly a **47x speedup** (1.9 s to 0.04 s). Nothing changed about how fast the native arithmetic is. Everything changed about how many times you paid to cross the boundary. This is the whole ballgame, and it is why the suggested NumPy-style interface — operate on arrays, not scalars — is not a stylistic preference but a cost-model necessity.

## The same kernel across the whole ladder

Theory is cheap. Let me take one concrete kernel and show it climbing the ladder, so the spectrum from "no native code at all" to "Rust extension" is something you can feel rather than just read about. The kernel is deliberately simple and deliberately *scalar with a branch*, because that is the case where the choice of rung actually matters — pure elementwise math would just be NumPy and we'd stop.

The kernel: given an array of a million `float64` values, compute a clipped, scaled transform — for each element, if it is above a threshold, take a scaled square root; otherwise take a scaled square. It is the kind of branchy per-element logic that shows up in feature engineering, signal processing, and risk calculations all the time.

Here it is in **pure Python**, the baseline:

```python
import math

def kernel_python(data, threshold=0.5, scale=2.0):
    out = [0.0] * len(data)
    for i, x in enumerate(data):
        if x > threshold:
            out[i] = scale * math.sqrt(x)
        else:
            out[i] = scale * x * x
    return out
```

On the 8-core box, over 1 million `float64` values, this runs in roughly **310 ms**. Every iteration pays the eval-loop tax, boxes a fresh float, refcounts the list slot, and dispatches `math.sqrt` as a Python call. It is correct and readable and slow.

Now **vectorized NumPy** — rung 0, no native code authored. The branch becomes a `np.where`:

```python
import numpy as np

def kernel_numpy(data, threshold=0.5, scale=2.0):
    return np.where(
        data > threshold,
        scale * np.sqrt(data),
        scale * data * data,
    )
```

This runs in roughly **9 ms** — about 34x faster than pure Python. But notice the catch, the reason this kernel is a good teaching example: `np.where` evaluates *both* branches for *every* element and then selects. It computes `np.sqrt(data)` for the whole array even where the condition is false, and `data * data` for the whole array even where the condition is true, plus it allocates several full-size temporaries. For a cheap kernel that wasted work is fine; for an expensive branch or a memory-bound expression, it bites. We are doing roughly twice the arithmetic and several extra passes over memory. This is exactly the situation where dropping to a compiled scalar loop wins, because a compiled loop can evaluate *only the taken branch* per element.

Now **Numba** — rung 1. The beautiful thing is that the code is almost identical to the pure-Python version; you add a decorator and Numba compiles the actual scalar loop, branch and all, to machine code:

```python
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def kernel_numba(data, threshold=0.5, scale=2.0):
    out = np.empty_like(data)
    for i in range(data.size):
        x = data[i]
        if x > threshold:
            out[i] = scale * np.sqrt(x)
        else:
            out[i] = scale * x * x
    return out
```

After the first call pays the JIT compilation cost (a few hundred milliseconds, once), the warm runtime over 1 million elements is roughly **1.2 ms** — about 260x faster than pure Python and meaningfully faster than `np.where` *because it only computes the taken branch and allocates no temporaries*. The loop is real machine code, the floats are unboxed, and the branch is a single native comparison. Three lines of decorator and a loop you already knew how to write. This is why Numba is the first native rung you reach for on a scalar hot loop, and the next post, on Numba's JIT compiling Python to machine code, goes deep on exactly how the type specialization and nopython mode work.

Now **the FFI / ctypes path** — rung 3 — to show the *other* end of the spectrum, calling hand-written C. The C, in `kernel.c`:

```c
#include <math.h>

void kernel_c(const double* data, double* out, long n,
              double threshold, double scale) {
    for (long i = 0; i < n; i++) {
        double x = data[i];
        if (x > threshold) {
            out[i] = scale * sqrt(x);
        } else {
            out[i] = scale * x * x;
        }
    }
}
```

Compiled with `gcc -O3 -shared -fPIC -o kernel.so kernel.c`, and called from Python — note carefully that we pass the *whole buffer* in one crossing, applying the FFI rule we derived:

```python
import numpy as np
import ctypes

lib = ctypes.CDLL("./kernel.so")
lib.kernel_c.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_long, ctypes.c_double, ctypes.c_double,
]

def kernel_ctypes(data, threshold=0.5, scale=2.0):
    out = np.empty_like(data)
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.kernel_c(
        data.ctypes.data_as(ptr), out.ctypes.data_as(ptr),
        data.size, threshold, scale,
    )
    return out
```

This runs in roughly **1.0 ms** over 1 million elements — comparable to Numba, as you'd expect since both produce an optimized native loop, and slightly faster here because `gcc -O3` vectorizes the math. The single FFI crossing is invisible against a million elements of work. The trade-off versus Numba is stark in *effort*: you wrote and compiled a separate C file, you manage the build, you marshal pointers by hand, and a mistake in `argtypes` is a segfault, not a Python exception. You'd take this path when the C already exists or when you need a control Numba can't give — not to beat Numba on a kernel Numba handles fine.

I'll spare you the full Cython and Rust listings — they are the subjects of the dedicated posts — but the shape is the same: a typed loop compiled to native code, landing in the same **~1 ms** ballpark, differing mainly in build pipeline and what else you can do (Cython: call C libs, release the GIL deliberately; Rust: memory safety and easy true parallelism). Here is the whole ladder on one kernel, on the named machine:

| Implementation | Native code? | Time (1M float64) | Speedup vs Python | Author effort |
| --- | --- | --- | --- | --- |
| Pure Python | none | 310 ms | 1x | trivial |
| NumPy `np.where` | none (uses NumPy) | 9 ms | ~34x | very low |
| Numba `@njit` | JIT, no build | 1.2 ms | ~260x | low (a decorator) |
| Cython `.pyx` | compiled, build step | ~1.1 ms | ~280x | medium |
| C via `ctypes` | hand-written C | 1.0 ms | ~310x | medium-high |
| Rust via PyO3 | hand-written Rust | ~1.0 ms | ~310x | high upfront |

Read that table the right way. The *speed* difference between Numba, Cython, C, and Rust on this kernel is in the noise — they all compile the same loop to roughly the same machine code, and the differences come down to compiler flags and SIMD, not the language. What actually differs by a lot is the **effort** column and the **what else you get** that the column can't show: does it need a build step, do you own a separate source file in another language, can you release the GIL, can you call an existing library, will a bug be a Python exception or a segfault. *That* is the axis your decision turns on, not the milliseconds, because past the first compiled rung the milliseconds are basically equal.

![matrix comparing NumPy, Numba, Cython, ctypes, pybind11, and Rust across effort, typical speedup, whether you own native code, and the build step required](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-2.png)

## What you give up at each rung

The table above shows what you *gain*. Engineering decisions are made on what you *give up*, and every rung down the ladder costs you something real. Let me be honest about each, because the marketing for these tools never is.

**Vectorizing (NumPy)** costs you almost nothing in build or portability, but it costs you *expressiveness*: you can only do what array operations can express. Genuinely sequential logic — where element $i$ depends on the result at element $i-1$, like a stateful parser or a recurrence — does not vectorize, and contorting it into array form (with cumulative tricks and masks) often produces code slower and far less readable than a plain loop. It also costs *memory*: as we saw, `np.where` and compound expressions allocate temporaries, and a vectorized formulation can use several times the peak memory of a streaming loop.

**Numba** costs you a first-call compilation delay (a few hundred milliseconds, or seconds for complex functions) and a real boundary of supported features — it compiles a subset of Python and NumPy, and the moment you touch a Python object it doesn't understand (a dict of mixed types, an arbitrary class, most of the standard library) it either falls back to slow object mode or refuses. Debugging is harder: a compiled `@njit` function does not drop into `pdb` the way Python does, and a type-inference failure produces an error message that takes practice to read. And it adds a heavyweight dependency (Numba pulls in LLVM) that you may not want in a lean deployment.

**Cython** costs you a *build step* — you now ship `.pyx` files that must be compiled, which means a C compiler on the build machine, a `setup.py` or `pyproject.toml` build backend, and platform-specific wheels if you distribute. It costs you a dialect to learn: the typing syntax, the memoryview semantics, the rules about when you're in fast C mode versus slow Python mode (the `cython -a` annotation that colors Python-interacting lines yellow exists precisely because this is non-obvious). And debugging a crash in compiled Cython means C-level debugging.

**A C/C++ extension** costs you the most in *safety and maintainability*. You are writing manual memory management, manual reference counting if you touch the C-API, and the failure mode of a bug is no longer an exception — it is a segfault, a memory corruption, or a refcount leak that surfaces as a mysterious crash three functions later. It is the least portable (you compile per platform), the hardest to onboard new engineers onto, and the easiest to get subtly, dangerously wrong. The flip side is total control and the ability to wrap any existing C/C++ library.

**Rust** costs you the steepest learning curve on the ladder — the borrow checker is a real investment — and a full build pipeline (maturin, the Rust toolchain). What it buys back is the thing C gives up: *memory safety*. The borrow checker makes the segfault and use-after-free classes of bug compile errors, not production incidents. That is why, for a substantial, long-lived native component that a team will maintain, Rust has become the default over C++: you pay the learning cost once and stop paying the debugging cost forever.

Here is the same trade-off as a table you can scan when you're deciding, with the costs stated plainly rather than in marketing terms. The "biggest risk" column is the one most people skip and the one that bites hardest in production.

| Rung | What you give up | Build step | Debuggability | Biggest risk |
| --- | --- | --- | --- | --- |
| NumPy / Polars | sequential logic, peak memory | none | normal Python | accidental temporaries, OOM |
| Numba | first-call compile, supported subset | JIT at runtime | harder, no pdb in kernel | unsupported construct falls to slow mode |
| Cython | a dialect, portability | compile `.pyx` | C-level for crashes | yellow Python-interacting lines stay slow |
| C / C++ FFI | memory safety, exceptions | per-platform compile | hardest, segfaults | refcount leaks, memory corruption |
| Rust + PyO3 | learning curve, toolchain | maturin wheel | good (panics, not UB) | upfront cost before any payoff |

Read the "biggest risk" column as the thing that will actually page you. For NumPy it is a silent 4 GB temporary that OOM-kills the box. For Numba it is a function that quietly fell back to object mode and runs at Python speed while you think it's compiled — always check that `nopython` mode actually engaged. For Cython it is the line you forgot to type, which `cython -a` paints yellow and which runs at interpreter speed inside your "compiled" kernel. For a C extension it is the use-after-free that corrupts memory and crashes somewhere unrelated an hour later. For Rust it is simply that you spent two weeks fighting the borrow checker before you shipped anything. Every rung is a real cost; choose the shallowest one that solves the problem.

The figure below stacks the rungs against what each layer removes from the Python overhead and what flexibility it gives up to do so.

![layered stack showing pure Python as flexible with the eval loop tax, then vectorized NumPy as one C loop, then JIT typed Numba as machine code with the GIL, then compiled Cython with static types and SIMD, then a Rust kernel with no GIL inside and memory safety](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-5.png)

#### Worked example: choosing the rung for three real hot paths

Three different teams profile three different services and each finds a hot path. Watch how the *same* decision process routes them to three different rungs.

**Team A — the analytics aggregation.** Profile shows 88 percent of a batch job in a function that computes, per row, a weighted sum across columns and a normalization. It is pure elementwise and reduction math over a few million rows of `float64`. Amdahl ceiling: $1/(1-0.88) \approx 8.3\times$. Decision: this *vectorizes*. It is array math with no sequential dependency. Rung 0 — NumPy (or Polars for the dataframe parts). No native code, no build, done in an afternoon, measured 7x overall. They never left Python. The team that wanted to "rewrite it in Rust" would have spent two weeks for a worse result.

**Team B — the risk-scoring loop.** Profile shows 91 percent in a function with a genuinely scalar, branchy recurrence: each timestep's value depends on the previous timestep, with several `if` branches per step, over 50 million steps. This does *not* vectorize — the sequential dependency kills `np.where`, and the branchiness wastes work even where it half-does. Amdahl ceiling: $1/(1-0.91) \approx 11\times$. Decision: scalar hot loop, no existing C library, want minimal new code to own. Rung 1 — Numba. A `@njit` decorator on the existing loop, warm runtime 60x faster on the function, ~9x overall, shipped in a day with no new build system. If Numba had choked on some unsupported construct, they'd have dropped to Cython (rung 2) for the control.

**Team C — the image-decode path.** Profile shows 80 percent of a thumbnail service in decoding a proprietary image format, for which a mature, fast, well-tested **C library already exists**. Amdahl ceiling: $1/(1-0.80) = 5\times$. Decision: the native code *already exists* — the job is integration, not authorship. Writing this decoder from scratch in Numba or Rust would be reinventing a tested wheel. Rung 3 — wrap the existing library with `ctypes` (it has a clean C API) or `pybind11` (if it's C++), processing whole images per crossing. ~4.5x overall, a week of careful FFI work, zero new decoding logic to maintain.

Same process, three answers. The driver was never "which tool is fastest" — past the compiled rungs they're all similar — it was *the shape of the problem*: vectorizable math, scalar branchy loop, or existing native library. The matrix figure below is the lookup table for that routing.

![matrix mapping each profiler symptom such as vectorizable array math, a scalar branchy hot loop, an existing C library, already near peak, or I/O bound, to the best lever and whether native is the answer](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-7.png)

## The GIL, and the parallelism only native code can unlock

There is a dimension of the native decision that pure single-thread speedups miss entirely, and it is the reason the most ambitious native rewrites exist: the Global Interpreter Lock. The GIL is one lock per interpreter that protects CPython's internal state — most importantly the reference counts on every object. Because every Python operation touches refcounts, only one thread can execute Python bytecode at a time. This is why running a CPU-bound Python loop on eight threads does not make it eight times faster; the eight threads take turns holding the one lock, and you get roughly single-core throughput plus some contention overhead. Threads help Python only when they spend their time *waiting* — on I/O, on a socket, on a disk — because the GIL is released during the wait, letting another thread run. For pure CPU-bound work in pure Python, the GIL is a hard ceiling at one core's worth of compute, no matter how many cores the box has.

Native code is the escape hatch, and this is a genuinely different reason to go native than raw single-thread speed. When your code is running inside a native loop that touches no Python objects — just packed C arrays and machine numbers — there are no refcounts to protect, so it is *safe to release the GIL*. While the native loop runs with the GIL released, other Python threads can run, and crucially *other instances of the same native loop on other threads can run truly in parallel*, one per core. This is the only way to get CPU-bound work in a Python process to scale across all the cores. NumPy does this internally for some operations; Numba gives you `nogil=True` and `parallel=True` with `prange`; Cython lets you wrap a block in `with nogil:`; and in Rust with PyO3 you call `Python::allow_threads` to drop the GIL around the heavy computation. The Rust ecosystem leans on this hard — a library like Polars releases the GIL and runs its query engine across all cores, which is a large part of why it beats single-threaded pandas by so much on a multi-core box.

This reframes the bottom of the ladder. The reason "Rust kernel, no GIL inside" sits at the deepest rung is not only that Rust is fast on one core — Numba and C are too — but that a substantial native module is the natural place to release the GIL and *parallelize the hot kernel across cores within a single process*, no pickling, no inter-process serialization, shared memory by default. That is a capability NumPy alone cannot give you for *your own* logic and that pure-Python threading can never give you for CPU-bound work. If your hot path is CPU-bound and you want it to use all eight cores without the serialization tax of `multiprocessing`, the answer is a native kernel that releases the GIL. The detailed mechanics of the GIL, of process-based parallelism, and of where each concurrency tool fits live in the concurrency track later in this series; here the point is narrow and important: *true multi-core CPU parallelism inside one Python process requires native code that releases the GIL.* That capability is one of the strongest reasons to descend the ladder past Numba toward Cython or Rust, even when single-thread speed alone wouldn't justify it.

#### Worked example: scaling a native kernel across cores

Take the scoring loop from earlier — 50 million scalar steps, CPU-bound, 91 percent of runtime. A single-threaded Numba `@njit` version on the named 8-core box runs the kernel in, say, 800 ms, a big win over pure Python. But the box has eight cores and seven of them are idle the entire time, because even compiled Numba code holds the GIL by default when called normally and a single `@njit` call is single-threaded. Now rewrite the loop to release the GIL and parallelize across cores. In Numba that is changing the decorator to `@njit(parallel=True, nogil=True)` and the loop to `prange`:

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def score_parallel(data, out):
    for i in prange(data.size):    # prange splits across cores
        x = data[i]
        out[i] = x * x if x <= 0.5 else 2.0 * np.sqrt(x)
    return out
```

On eight cores, the kernel does not hit a perfect 8x — there is scheduling overhead, memory-bandwidth contention (all eight cores reading and writing the same arrays compete for the memory bus), and the sequential fraction of setup. A realistic result on this kind of memory-light kernel is roughly **5.5–6.5x** over the single-thread native version, so 800 ms drops to about 130 ms. Combine that with the single-thread native win and you've gone from a pure-Python baseline of tens of seconds to a fraction of a second, using all the hardware you paid for. The key enabler was *releasing the GIL inside native code* — without it, those seven cores stay dark. This is the parallelism story that pure Python cannot tell and that sits at the heart of why the deepest native rungs exist.

## Profiling to find the 1 percent

The entire decision rests on knowing $p$, the fraction of time in the hot path, and the only way to know it is to measure. Guessing is worse than useless because human intuition about where time goes is reliably wrong — we suspect the code we wrote recently, the function with the scary name, the part we don't understand. The profiler does not have those biases.

The fast first pass is `cProfile`, the deterministic profiler in the standard library. It records every call and gives you cumulative and total time per function:

```bash
python -m cProfile -o report.prof run_job.py
```

Then read it sorted by cumulative time, which is the right sort for finding the hot path because it tells you which function (and everything it calls) accounts for the most wall-clock:

```pycon
>>> import pstats
>>> p = pstats.Stats("report.prof")
>>> p.sort_stats("cumulative").print_stats(8)
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.004    0.004  600.12  600.12  run_job.py:1(main)
        1    1.230    1.230  480.55    480.55  scoring.py:14(score_rows)
  3000000  402.11    0.000  402.11    0.000  scoring.py:31(_score_one)
        1    0.880    0.880  119.40    119.40  io.py:8(write_report)
```

Read that and the decision makes itself. `score_rows` is 480 of 600 seconds — $p = 0.80$. Inside it, `_score_one` is called 3 million times and is itself 402 seconds of `tottime` (time in the function excluding sub-calls). That `tottime` plus the 3-million `ncalls` is the signature of a per-row scalar function — exactly the shape that routes to Numba or Cython, *not* to NumPy (it's branchy and called per row, not array math) and *not* to "stop" (the ceiling is a healthy 5x). The profiler didn't just tell us *whether* to go native; the *shape* of the hot path told us *which rung*.

For per-line detail inside the hot function — which line of `_score_one` is actually expensive — `line_profiler` is the tool, and `py-spy` is the one you reach for when the job is already running in production and you can't restart it. Those are covered in the profiling track; the point here is the discipline: **never go native on a function the profiler hasn't fingered.** Profile, read $p$, compute the Amdahl ceiling, look at the *shape* (array math vs scalar loop vs existing library), and only then pick the rung. The figure below is that decision as a tree you can walk in your head.

![decision tree asking first whether the hot path is over twenty percent of time, routing small fractions to stopping, vectorizable work to NumPy, and scalar branchy loops to Numba then Cython or Rust](/imgs/blogs/the-native-acceleration-landscape-when-to-leave-pure-python-8.png)

## How to measure a native speedup honestly

When you do go native and want to claim a number, measure it correctly or you will fool yourself. A few traps specific to native code:

**Warm up the JIT.** Numba compiles on the first call with the given argument types. If you `timeit` and include the first call, you are timing the compiler, not the code — you'll see a "speedup" of *less than one* and conclude Numba is slow. Call the function once to compile, *then* measure the warm path. The honest report separates the one-time compile cost from the steady-state runtime, because they matter to different decisions (compile cost matters for a short-lived CLI; steady-state matters for a long-running service).

**Use a realistic input size.** The FFI crossing cost is fixed, so the *measured* speedup of a native function depends enormously on $N$. Benchmark on 100 elements and the crossing overhead dominates and native looks terrible; benchmark on 10 million and it vanishes and native looks miraculous. Both are misleading if your production $N$ is different. Measure at the size you actually run.

**Beat the constant-folding and caching traps.** If your benchmark passes the same constant input every time, the OS page cache, the CPU branch predictor, and sometimes the compiler itself will make the second run unrealistically fast. Vary the input, or at least be aware the warm-cache number is a best case.

**Report the right unit and the spread.** For a microbenchmark, ns/op or µs/call; for a macro job, wall-clock seconds; always with `×speedup` relative to a named baseline. Run it several times and report the median (means are dragged around by the occasional GC pause or scheduler hiccup), and if you can, the spread. A speedup with no baseline, no input size, and no machine is not a measurement.

**Account for memory, not just time.** A native rewrite that is 50x faster but allocates a 4 GB temporary where the streaming Python version used 200 MB may be the wrong trade on a memory-constrained box. Measure RSS (`memray` or even `/usr/bin/time -v`) alongside the wall-clock.

#### Worked example: the benchmark that lied

A team benchmarks their new `@njit` kernel against the Python version with a naive `timeit` over a *small* array — 500 elements — including the first (compiling) call:

```pycon
>>> import timeit
>>> timeit.timeit("kernel_numba(small)", globals=globals(), number=1)
0.412      # 412 ms?! for 500 elements?
>>> timeit.timeit("kernel_python(small)", globals=globals(), number=1)
0.00018    # Python is 2000x FASTER?
```

They nearly abandoned Numba. The 412 ms was the JIT *compiling*, paid once, not the kernel running. And 500 elements is far too small — even warm, the per-call dispatch overhead is a big fraction of 500 elements of work. Measured correctly — warm the JIT first, use a production-sized 1-million-element array, repeat and take the median:

```pycon
>>> kernel_numba(big)        # warm-up call: compiles, result discarded
>>> from statistics import median
>>> warm = [timeit.timeit("kernel_numba(big)", globals=globals(), number=1)
...         for _ in range(7)]
>>> median(warm)
0.0012       # 1.2 ms warm, over 1M elements
>>> base = [timeit.timeit("kernel_python(big_list)", globals=globals(), number=1)
...         for _ in range(7)]
>>> median(base)
0.31         # 310 ms
```

The honest number is **~260x** on the warm path at production size — not a 2000x regression. The first benchmark wasn't wrong about the numbers it produced; it was measuring the wrong thing (the compiler) at the wrong size (too small). This is why "measure honestly" is its own skill and why a claimed native speedup with no setup attached deserves suspicion.

## Case studies: where native actually paid off

The ecosystem is full of proof that the "rewrite the hot 1 percent in native, keep the Python" strategy works at scale. A few real, named results — versions and sources noted because a number with no source is a rumor.

**Polars and pydantic-core (Rust).** Polars is a dataframe library whose engine is written in Rust (using Apache Arrow's columnar memory) and exposed to Python; on many group-by and join benchmarks it runs several times to an order of magnitude faster than pandas on multi-core machines, because the heavy lifting is a multi-threaded Rust engine and Python is only the thin orchestration layer. Pydantic v2 moved its validation core into a Rust crate (`pydantic-core`) and the maintainers reported validation roughly 5–50x faster than v1 depending on the model, again by putting the hot validation loop in native code while keeping the Python-facing API. Both are textbook "1 percent native, 99 percent Python" — you write ordinary Python and the hot path is native underneath.

**ruff and uv (Rust).** Ruff, a Python linter written in Rust, is commonly 10–100x faster than the pure-Python linters it replaces (flake8, pylint) on large codebases — the same linting logic, but the file-walking and AST-checking hot loops are native. uv, a Python package installer and resolver also in Rust, resolves and installs dependency trees dramatically faster than pip for the same reason. The lesson these projects teach is not "rewrite Python tooling in Rust" as a blanket rule — it is that the *hot path* of these tools (parsing, walking, resolving over many files) is exactly the kind of scalar, branchy, sequential work that doesn't vectorize and benefits enormously from a compiled language.

**tokenizers (Rust) and the NumPy/SciPy core (C/Fortran).** Hugging Face's `tokenizers` library puts the tokenization hot loop in Rust and is orders of magnitude faster than pure-Python tokenization, which matters when you're feeding billions of tokens to a model. And the granddaddy of the whole strategy: NumPy and SciPy have always been thin Python wrappers over compiled C and Fortran. Every time you call `np.dot` you are "going native" — the Python is just the glue. That is the model the entire scientific Python stack is built on, and it has worked for two decades.

**The Faster CPython project.** Worth noting as a baseline shift: CPython 3.11 was roughly 10–60 percent faster than 3.10 across the pyperformance benchmark suite, and 3.12 and 3.13 continued the trend, thanks to the specializing adaptive interpreter (PEP 659) and other interpreter-level work. This matters to the native decision because the pure-Python baseline is a moving target — a chunk of the win you'd have gotten from a native rewrite on 3.10 may already be in the interpreter for free on 3.12. Always re-measure on your actual Python version before deciding the interpreter is the bottleneck.

There is also an instructive *negative* case worth holding alongside the wins, because survivorship bias makes native rewrites look like sure things and they are not. Plenty of teams have ported a hot function to C or Rust and gotten a disappointing 1.3x — or a regression — for weeks of work, and almost always for one of three reasons we've now named: the hot fraction $p$ was smaller than they thought (low Amdahl ceiling), the function was actually memory-bandwidth-bound rather than compute-bound (so a faster instruction stream didn't help because the bottleneck was waiting on RAM), or they structured the boundary to cross per element and the marshaling ate the gain. When you read a "we rewrote it in Rust and it's barely faster" post-mortem, it is nearly always one of those three. The fix for each is upstream of the rewrite: measure $p$ before committing, check whether the work is compute-bound or memory-bound (a rough roofline estimate tells you), and design the boundary for few crossings over big buffers. Native code is a powerful lever, but it is the *third* rung for a reason, and skipping the measurement that tells you whether to pull it is how good engineers waste good quarters.

The common thread across all the *wins*: nobody rewrote 100 percent of anything. They found the hot path — tokenization, validation, dataframe ops, dependency resolution — wrote *that* in native code, and kept the entire rest of the program in Python. The native module is a small, hot, well-bounded kernel behind a clean Python API. That is the pattern. That is the whole post in one sentence.

## When to reach for native code, and when not to

Let me be decisive, because the most valuable thing a landscape post can do is tell you when *not* to climb the ladder at all.

**Reach for native when:** the profiler shows a single CPU-bound function is a large fraction of runtime (high $p$, so a high Amdahl ceiling); the work is genuinely scalar and branchy or sequential so it does *not* vectorize; you can structure the interface so the native code processes whole buffers in few crossings, not scalars in many; and the speedup you need is more than the ~2–4x that the cheaper levers (better algorithm, vectorization, the 3.12 interpreter) already give you. If all four hold, go native, and start at the lowest rung that does the job — Numba before Cython, Cython before C, an existing library before writing one.

**Do not reach for native when:**

- **The hot path vectorizes.** If it's array math, NumPy or Polars gets you native-speed C loops with zero native code, no build, and full portability. Writing Cython for something `np.where` and a reduction handle is pure waste. Vectorize first; this is the rung most native proposals should have tried.
- **The function is a small fraction of runtime.** Amdahl caps your win at $1/(1-p)$. Optimizing a 5 percent function caps you at ~1.05x. The bottleneck is elsewhere; go find it. Native code on a cold path is effort spent to make nothing faster.
- **You'd cross the FFI boundary per element.** If you cannot restructure to process whole arrays in one crossing — if the algorithm truly needs a Python callback per item — the marshaling cost will eat the speedup and may make it slower than pure Python. Fix the structure first or stay in Python.
- **The work is I/O-bound or network-bound, not CPU-bound.** Native code makes *computation* faster. If your function spends its time waiting on a database, a disk, or a socket, a Rust rewrite changes nothing — the CPU was idle the whole time. That is a job for async or threads, not native acceleration. Profiling that shows low CPU and high wait time is the tell.
- **You're already within ~2x of peak.** If a back-of-envelope roofline says the work is near the memory-bandwidth or compute limit, there's little left to extract and the native effort buys a marginal win at a large maintenance cost. Know when to stop.
- **The team can't maintain it.** A C or Rust extension that one person wrote and nobody else understands is a liability the day that person leaves. The maintainability cost is real and it is paid forever. Sometimes the right call is "slightly slower but in Python the whole team can debug."

And the meta-rule, the one that subsumes all of these: **rewrite the hot 1 percent in native, never 100 percent.** The win is bounded by the hot fraction, so rewriting the cold 99 percent buys nothing and costs everything. Keep the Python glue, the libraries, the readability. Put one small, hot, well-bounded native kernel behind a clean Python API, hand it whole buffers, measure the before and after, and prove the win with a number.

## Key takeaways

- **Profile before you go native.** The Amdahl ceiling is $1/(1-p)$ where $p$ is the hot path's fraction of runtime. If $p$ is small, no native rewrite can help — the bottleneck is elsewhere. Measure $p$ first; it is the only input that tells you whether to proceed.
- **The ladder has an order: vectorize, then Numba, then Cython, then C/C++, then Rust.** Start at the lowest rung that solves the problem. Each rung down trades more build complexity and more native code to maintain for more control. Past the first compiled rung, the *speed* is similar — the real difference is *effort* and *what else you can do*.
- **Vectorize first.** If the hot path is array math, NumPy or Polars gives you native-speed C loops with no native code, no build step, and full portability. Most "rewrite it in C" proposals should have vectorized instead.
- **The FFI boundary has a fixed per-crossing cost — roughly a couple hundred nanoseconds.** Pay it once over a whole array, never once per element. A million tiny native calls lose to one native call over a million items, and can be slower than pure Python. Few crossings over big work; never many crossings over tiny work.
- **Native code buys five things:** no eval loop, no boxing, static types, SIMD, and a released GIL inside the kernel. The first four are single-thread speed; the fifth is the only path to true CPU parallelism in Python.
- **Match the rung to the shape of the problem,** not to which tool is trendiest. Vectorizable math goes to NumPy; a scalar branchy loop goes to Numba or Cython; an existing C/C++ library goes to ctypes or pybind11; a substantial new owned native component goes to Rust.
- **Measure the win honestly:** warm up the JIT before timing, use a production-sized input, report the median over several runs, give the unit and the baseline and the machine, and watch RSS as well as wall-clock. A claimed speedup with no setup attached is not a measurement.
- **Rewrite the 1 percent that is hot, keep the 99 percent that is fine.** That is the strategy behind Polars, pydantic-core, ruff, tokenizers, and NumPy itself: a small, hot, native kernel behind a clean Python API.

## Further reading

- [Why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro and the cost model this whole post assumes.
- [A mental model of performance: latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) — Amdahl's law, the optimization loop, and when not to optimize at all.
- [When NumPy isn't enough: numexpr, bandwidth, and avoiding temporaries](/blog/software-development/python-performance/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries) — the rung directly before native: what to try when vectorized NumPy is memory-bound.
- [CPU profiling: cProfile and finding the hot path](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path) — how to read a profile and read off the fraction $p$ that drives the native decision.
- The Numba documentation (`@njit`, `@vectorize`, `parallel=True`) — the next post in this track goes deep on JIT-compiling Python to machine code.
- The Cython documentation (typed memoryviews, `nogil`, `cython -a`) and the PyO3 + maturin guides for Rust extensions — the dedicated posts later in this track.
- "High Performance Python" by Micha Gorelick and Ian Ozsvald (2nd edition, O'Reilly) — the canonical book-length treatment of this whole ladder.
- The Faster CPython project notes and PEP 659 (the specializing adaptive interpreter) — for why the pure-Python baseline keeps moving and you should re-measure on your actual version.

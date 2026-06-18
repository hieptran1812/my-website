---
title: "Numba: JIT-Compiling Python to Machine Code"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Decorate a numeric Python loop with one line, let Numba type-infer and lower it to LLVM machine code, and watch a 95 ns per-element loop turn into a sub-nanosecond one — with the object-mode trap, prange, and warmup costs all measured."
tags:
  [
    "python",
    "performance",
    "optimization",
    "numba",
    "jit",
    "llvm",
    "parallelism",
    "profiling",
    "native-code",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/numba-jit-compiling-python-to-machine-code-1.png"
---

There is a particular kind of slow Python that nothing else in this series fixes. You have profiled the job. You know the hot path. It is a tight numeric loop — element by element, with a branch in the middle, maybe a running accumulator that depends on the previous element. You reach for NumPy, the way the [vectorization track](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) taught you, and you hit a wall: the loop does not vectorize cleanly. Each step looks at the last step. There is a conditional that NumPy can only express as three array passes and a mask. You can force it into array operations, but you end up allocating four temporaries and walking memory five times to avoid one honest `for` loop. It is ugly, it is memory-bound, and it is still slow.

This is the gap Numba was built for. You take that exact Python loop — the readable one, the one with the branch — add one decorator line above the function, and Numba compiles it to native machine code the first time you call it. Not "speeds it up a bit." Compiles it. The interpreter steps out of the loop entirely. On the kind of scalar numeric kernel that does not vectorize, the result is routinely **50 to 200 times faster** than the pure-Python version, and it competes with hand-written C — from three lines of change, on code you can still read.

![before and after comparison of a pure Python numeric loop at 95 nanoseconds per element versus the same loop with the njit decorator at under one nanosecond per element a roughly 100 times speedup](/imgs/blogs/numba-jit-compiling-python-to-machine-code-1.png)

That is the promise, and figure 1 is the whole post in one image: same loop, three lines added, two orders of magnitude. But "add a decorator and it gets 100× faster" is exactly the kind of claim that should make a working engineer suspicious. *Why* does it work? What is the decorator actually doing between the moment you call the function and the moment the result comes back? Where does the speed come from — is it real, or did the benchmark fool me? When does it silently *not* work and leave me at baseline thinking I optimized something? What does the first call cost, and how many calls do I need before the change pays for itself?

Keep the series' running example in mind: a data-processing pipeline that loads a few million rows, cleans them, transforms each one, and aggregates the result. Most of that pipeline is the right job for the levers we have already pulled — the cleaning and aggregation vectorize beautifully in NumPy or Polars. But somewhere in the *transform* step there is usually one stubborn function: a per-row calculation with a branch, or a running quantity that depends on the previous row, that refuses to vectorize. Profile the pipeline and that one function is the hot path — 80% of the wall-clock in 5% of the code. That function is the Numba candidate. The whole skill is recognizing it, compiling it, and measuring the win, while leaving the other 95% of the pipeline alone.

By the end of this post you will be able to: take a numeric hot loop and make it native in one line with `@njit`; read what the Numba compiler does to your function at each stage; recognize and avoid the object-mode fallback that is the single most common way people "use Numba" and get nothing; parallelize across cores with `prange`; build fast NumPy ufuncs with `@vectorize`; persist compiled code with `cache=True`; and — most importantly — decide *when Numba is the right lever and when it is the wrong one*. We are on rung three of the leverage ladder now: we have done less work (algorithm), we have done it in bulk (NumPy), and for the cases the array world cannot reach, we compile the hot 1%. This is the [native-acceleration landscape](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python) made concrete, starting with the lowest-effort option on the menu.

## 1. The thing Numba removes: the per-element interpreter tax

To understand why one decorator buys 100×, you have to understand what a pure-Python numeric loop is *actually doing* per element — because Numba's whole job is to delete that work.

Take the simplest possible numeric kernel: sum the squares of an array.

```python
def sum_squares(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total
```

Reading this, you see one multiply and one add per element. The CPU sees something very different. Walk one iteration of the loop the way CPython does it:

1. **Advance the iterator.** `for x in xs` calls the iterator's `__next__`, which is a C function dispatched through the object's type. It returns a *pointer* to a `PyObject` — the next element, boxed.
2. **What is a boxed element?** If `xs` is a Python `list` of floats, each `x` is a full heap-allocated `PyFloat` object: a refcount, a type pointer, and the actual 8-byte double, ~24 bytes total, living somewhere in the heap that you reach by chasing a pointer. (We covered this object model in [why Python is slow](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means).) "Boxing" means the raw 8-byte number is wrapped in this object envelope; "unboxing" means digging the double back out.
3. **`x * x` is a dispatch, not a multiply.** The `BINARY_OP` bytecode does not emit a CPU multiply instruction. It looks at the left operand's type, finds its number protocol slot (`tp_as_number->nb_multiply`), calls that C function, which unboxes both doubles, multiplies them, *allocates a brand-new `PyFloat`* for the result, and returns a pointer to it. One multiply; one heap allocation; one type dispatch.
4. **`total += ...` is another dispatch and another allocation.** Same dance: dispatch on type, unbox, add, box the result into yet another new `PyFloat`, rebind `total`.
5. **Reference counting on every step.** Each new object's refcount goes up; each object that falls out of scope has its refcount decremented, and at zero it is freed. Every read is also a write to a refcount field.
6. **The eval loop overhead wrapping all of it.** Each bytecode is fetched, decoded, and dispatched through the giant switch in the CPython evaluation loop, with the operand stack pushed and popped around every operation.

So one line of source — `total += x * x` — is, per element: an iterator call, two type dispatches, two heap allocations, several refcount touches, a fistful of pointer chases, and a dozen bytecode dispatches. On a modern x86-64 or Apple-silicon core, all of that adds up to roughly **80 to 120 nanoseconds per element**. The actual arithmetic — one multiply, one add — is well under a nanosecond. You are paying 99% interpreter tax and 1% math.

Now here is what `@njit` does. It looks at the *types* coming in (a `float64` NumPy array), and it realizes: every element is a `float64`, `total` is a `float64`, the operation is fused multiply-add. There is no need for boxes, no need for dispatch, no need for the eval loop. It can emit, for the entire loop body, the exact machine instructions a C compiler would: load a double from a contiguous buffer, multiply, add to a register, advance the pointer, branch. Nothing is allocated. Nothing is dispatched. The accumulator lives in a CPU register the whole time.

```python
from numba import njit

@njit
def sum_squares(xs):          # identical body
    total = 0.0
    for x in xs:
        total += x * x
    return total
```

The body is byte-for-byte the same. Pass it a NumPy `float64` array and the per-element cost falls from ~95 ns to **under 1 ns** — because now it is one fused-multiply-add per element with the loop bounds in registers, exactly as if you had written it in C. That is the entire trick, and everything else in this post is detail on top of it.

### Seeing the tax in the bytecode

If you want to *see* the work Numba removes, disassemble the pure-Python loop body and count the per-element bytecodes:

```python
import dis

def sum_squares(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total

dis.dis(sum_squares)
```

The interesting part is the loop body, which CPython executes once *per element*:

```bash
  4   FOR_ITER                ...      # advance iterator, get next boxed object
      STORE_FAST              x
  5   LOAD_FAST               total
      LOAD_FAST               x
      LOAD_FAST               x
      BINARY_OP               5 (*)    # type dispatch + alloc a new PyFloat
      BINARY_OP              13 (+=)   # type dispatch + alloc a new PyFloat
      STORE_FAST              total
      JUMP_BACKWARD           ...
```

Every one of those `LOAD_FAST`, `BINARY_OP`, `STORE_FAST` lines is a trip through the eval loop's dispatch switch, and the two `BINARY_OP`s each hide a C-level type dispatch and a heap allocation. That is roughly nine bytecode dispatches, two type dispatches, and two allocations **per element**. Compile the same thing with `@njit` and the entire loop body collapses to about four machine instructions — load, multiply, add, branch — with the accumulator pinned in a register and not a single allocation. The bytecode listing *is* the tax; Numba's job is to make it disappear.

### Quantifying the tax: where the 95 nanoseconds go

It is worth putting an approximate budget on the per-element cost, because it shows *why* the win is so large and so consistent. On the named box, a single Python `BINARY_OP` on two floats — dispatch, unbox, compute, allocate the result `PyFloat`, refcount — costs on the order of 30–40 ns. The loop has two of them, plus the `FOR_ITER` iterator call (~10 ns), plus the `LOAD_FAST`/`STORE_FAST` stack traffic and the bytecode-fetch overhead. Sum it and you land near the measured 80–120 ns per element. The actual floating-point multiply-add the CPU performs is *one* instruction, well under a nanosecond. So the cost ratio is roughly:

$$\frac{\text{interpreter overhead}}{\text{useful math}} \approx \frac{95\text{ ns}}{0.5\text{ ns}} \approx 190.$$

That ratio is the ceiling on your speedup, and it is why Numba's wins cluster around 100×: there is about 100–200× of pure overhead sitting on top of every numeric operation, and compiling the loop removes essentially all of it.

### Why LLVM can then go even further

Stripping the interpreter gets you most of the 100×. But once Numba has a typed, allocation-free loop in front of it, it hands the loop to **LLVM** (the same compiler backend behind Clang and Rust), and LLVM does the second-order optimizations a Python interpreter could never do:

- **Auto-vectorization (SIMD).** Modern cores have vector registers — AVX2 on x86 holds four `float64`s (256 bits), AVX-512 holds eight, NEON on Apple silicon holds two. LLVM can see that `total += x*x` is an independent reduction and emit instructions that process 4 or 8 elements per instruction. That is a free 4–8× on top of the de-interpreting, *if the loop is vectorizable* — which it is here.
- **Register allocation.** The accumulator and loop index stay in registers; no memory traffic for them at all.
- **Loop-invariant hoisting, strength reduction, unrolling.** LLVM applies the full classical optimizer to your loop.

This is the deeper reason Numba can match C: it is not "Python but faster," it is *your loop, lowered to the same intermediate representation a C compiler uses, optimized by the same backend.* The pipeline that gets you there is the subject of the next section.

## 2. How `@njit` actually compiles your function

Let me make the pipeline concrete, because understanding it explains every gotcha you will ever hit with Numba. When you decorate a function with `@njit` and then *call it*, here is what happens.

![graph of the Numba JIT pipeline showing Python bytecode flowing into type inference and into the Numba intermediate representation then both merging into LLVM IR which lowers to machine code that is stored in an on disk cache](/imgs/blogs/numba-jit-compiling-python-to-machine-code-2.png)

**Decoration does nothing yet.** `@njit` just wraps your function in a dispatcher object. No compilation happens at import time. This matters: the cost is deferred to the first call.

**First call — Numba reads the argument types.** This is the key idea, called *type specialization*. Numba does not know in advance what types you will pass. When you first call `sum_squares(arr)` with `arr` being a `float64[::1]` (a contiguous 1-D `float64` array), Numba inspects those concrete types and compiles a version of the function *specialized to exactly those types*. Call it later with a `float32` array and Numba compiles a *second* specialization. Each unique combination of argument types gets its own compiled machine-code version, cached in the dispatcher.

Then, for that type signature, it runs the pipeline shown in figure 2:

1. **Bytecode analysis.** Numba disassembles your function's CPython bytecode (the same bytecode `dis` shows) and reconstructs the control-flow graph — the loops, the branches, the basic blocks.
2. **Type inference.** Starting from the concrete argument types, Numba propagates types through every variable: `xs` is `float64[::1]`, so `x` is `float64`, so `x*x` is `float64`, so `total` is `float64`. If it can assign a concrete machine type to *every* variable, you are in **nopython mode** and life is good. If some variable's type can't be pinned down (it stays a generic Python object), that is the fork in the road we cover in section 4.
3. **Lowering to Numba IR.** The typed function is lowered into Numba's own intermediate representation — a typed, mostly-flat form where the boxing and dispatch are already gone.
4. **Generating LLVM IR.** The Numba IR is translated to LLVM IR, the typed SSA form LLVM optimizes. This is where your loop becomes indistinguishable from a loop emitted by a C front-end.
5. **LLVM optimizes and emits machine code.** LLVM runs its optimization passes (vectorization, register allocation, the works) and JIT-compiles to native instructions for *your* CPU — x86-64 or arm64, with whatever SIMD extensions it has.
6. **Cache and dispatch.** The machine code pointer is stored against the type signature. Future calls with the same argument types skip steps 1–5 entirely and jump straight into native code.

In figure 2, type inference and the IR-lowering form a genuine two-path layer — Numba reasons about types and reconstructs control flow somewhat in parallel before both feed LLVM. The important takeaway: **compilation happens lazily, per type signature, on the first call.** That single fact explains the warmup cost (section 5), the `cache=True` flag (section 8), and why benchmarking the *first* call is a classic mistake (section 10).

### `@njit` is `@jit(nopython=True)`

You will see two spellings. `@jit` is the original decorator; `@njit` is an alias for `@jit(nopython=True)`. The `nopython` flag is the whole game, and we will spend section 4 on why. Short version: `@njit` means "compile this to pure machine code with no Python objects, and if you can't, *raise an error* instead of silently falling back to something slow." That error-instead-of-silence behavior is exactly what you want. **Always write `@njit`. Never write bare `@jit`.** We will see precisely what goes wrong with bare `@jit` shortly.

### Type specialization, signatures, and eager compilation

The "one compiled version per argument-type combination" rule has practical consequences worth spelling out. Call your kernel with a `float64` array and Numba compiles a `float64` specialization. Call it later with a `float32` array and it compiles a *second* one — a second trip through the whole pipeline, a second compile cost. Call it with a Python `int` and you get a third. Each specialization is cached separately in the dispatcher, keyed by the exact signature. If your function is called with many different types, you pay many compiles; if it is called with one type, you pay once.

You can inspect what got compiled. The dispatcher exposes its signatures:

```pycon
>>> from numba import njit
>>> import numpy as np
>>> @njit
... def sum_squares(xs):
...     total = 0.0
...     for x in xs:
...         total += x * x
...     return total
...
>>> sum_squares(np.ones(5, dtype=np.float64))    # compiles a float64 version
5.0
>>> sum_squares(np.ones(5, dtype=np.float32))    # compiles a second, float32
5.0
>>> sum_squares.signatures
[(array(float64, 1d, C),), (array(float32, 1d, C),)]
>>> sum_squares.nopython_signatures              # confirm both are nopython
[(array(float64, 1d, C),) -> float64, (array(float32, 1d, C),) -> float32]
```

Two signatures, both nopython — exactly what you want to see. If `nopython_signatures` were empty while the function still ran, that would be your warning sign that it fell back.

If you want compilation to happen at *import* time instead of lazily on the first call — useful when you want failures surfaced at deploy time, not in the middle of a request — pass an explicit signature to the decorator. This is **eager compilation**:

```python
from numba import njit, float64

@njit(float64(float64[:]))          # compile NOW, for exactly this signature
def sum_squares(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total
```

With an explicit signature the compile happens when the module is imported, and calling with any *other* type raises a `TypeError` instead of compiling a new specialization. Eager compilation trades the lazy convenience for predictability: you know the cost is paid up front and you know exactly which types are allowed. Most code is fine with lazy compilation plus `cache=True`; reach for eager signatures when you need deterministic startup behavior or want to lock down the accepted types.

## 3. The toolbox: njit, vectorize, parallel, and the rest

Before we measure anything, here is the map of what Numba gives you, so you know which tool fits which shape of work. We will demonstrate each one in the sections that follow.

![matrix of Numba features with rows for njit vectorize guvectorize prange fastmath and cache and columns describing what each does and when to reach for it](/imgs/blogs/numba-jit-compiling-python-to-machine-code-3.png)

| Feature | What it does | Reach for it when |
|---|---|---|
| `@njit` | Compiles a whole function to native code | Any numeric hot loop over scalars or arrays |
| `@vectorize` | Turns a scalar kernel into a broadcasting NumPy ufunc | Elementwise op you want to apply across arrays with `out=` and broadcasting |
| `@guvectorize` | Generalized ufunc operating on array *slices*, not scalars | Per-row reductions, sliding windows, anything not strictly elementwise |
| `parallel=True` + `prange` | Spreads independent loop iterations across physical cores | Embarrassingly parallel loops on a multi-core box |
| `fastmath=True` | Relaxes strict IEEE-754 so LLVM can reorder and fuse | You tolerate tiny rounding differences for extra SIMD |
| `cache=True` | Persists compiled machine code to disk between runs | You want to pay the compile cost once, not every process start |
| `nogil=True` | Releases the GIL while the compiled code runs | You will call the njit function from multiple Python threads |

Most of the time you reach for exactly one tool: `@njit`. The others are refinements. Let's get the foundational one measured first, then add the refinements one at a time.

## 4. The object-mode trap (the silent way to "use Numba" and get nothing)

This is the most important section in the post, because it is the most common failure I see, and it is *silent* — the code runs, returns correct answers, and is barely faster than pure Python, while the developer believes they compiled it.

Recall type inference from section 2. Numba tries to assign a concrete machine type to every variable. Sometimes it can't — you used a Python `dict` with mixed value types, you called a function Numba doesn't support, you built a list of strings. When inference fails, there are two possible outcomes, and the decorator you chose decides which:

- **Bare `@jit` (the old default behavior):** historically, Numba would *fall back to object mode* — it compiles the loop *structure* to native code but keeps every operation as a call back into the Python interpreter on boxed `PyObject`s. You get the loop overhead removed but **none** of the per-element win, because every `x*x` is still a full type dispatch and allocation. The result is maybe 1.1× faster than pure Python. It runs. It is correct. It is nearly useless. And nothing warns you.
- **`@njit` (= `nopython=True`):** if inference can't pin every type, Numba *raises a `TypingError`* and refuses to compile. The error is loud, points at the unsupported construct, and forces you to either fix it or accept that this code isn't a Numba candidate.

![before and after comparison showing bare jit falling back to object mode and running interpreted near baseline versus njit forcing nopython mode and running fully compiled about one hundred times faster](/imgs/blogs/numba-jit-compiling-python-to-machine-code-4.png)

Figure 4 is the cliff. Object mode sits at the baseline; nopython mode is on the other side of a 100× gap. The whole point of `@njit` is to make sure you are never on the wrong side of that cliff without knowing it.

Here is the trap in code. Suppose your kernel accidentally builds a Python dict inside the loop:

```python
from numba import jit, njit
import numpy as np

# The trap: bare @jit silently falls back to object mode on the dict
@jit
def histogram_buggy(xs):
    counts = {}                      # generic dict — type inference can't pin it
    for x in xs:
        b = int(x)
        counts[b] = counts.get(b, 0) + 1
    return counts

# The fix: @njit refuses to compile this, telling you it's not Numba-able
@njit
def histogram_njit(xs):
    counts = {}
    for x in xs:
        b = int(x)
        counts[b] = counts.get(b, 0) + 1
    return counts
```

Call `histogram_buggy` and it works — and is barely faster than the plain Python version, because it ran in object mode. Call `histogram_njit` and you get a `TypingError` immediately, telling you the untyped dict is the problem. The `@njit` failure is the *better* outcome: it tells you the truth.

How do you fix the actual histogram? You give Numba types it can compile. Either preallocate a NumPy counts array (if the bin range is known), or use Numba's *typed* `Dict` from `numba.typed`, which has a concrete key and value type:

```python
import numpy as np
from numba import njit

@njit
def histogram_fixed(xs, n_bins):
    counts = np.zeros(n_bins, dtype=np.int64)   # typed buffer — Numba loves this
    for x in xs:
        b = int(x)
        if 0 <= b < n_bins:
            counts[b] += 1
    return counts
```

This compiles in nopython mode and runs at native speed. The lesson is general: **Numba wants typed, homogeneous data.** When it complains, it is telling you which variable it couldn't type — fix that variable, don't reach for bare `@jit` to make the error go away. As of recent Numba versions, object-mode fallback from `@jit` is deprecated precisely because it caused this confusion; treat `@njit` as the only decorator you use, and read its errors as a map of what to change.

#### Worked example: the object-mode cliff in numbers

The setup, used for every number in this post: **an 8-core x86-64 Linux box (a result an Apple M2 reproduces within ~15%), CPython 3.12, Numba 0.60, NumPy 2.0, 16 GB RAM**, input a `float64` array of 10 million elements, timed with `timeit` after warmup, median of seven runs.

| Version | Time for 10M elements | ns/op | ×speedup |
|---|---|---|---|
| Pure Python loop | 0.95 s | 95 | 1.0× |
| Bare `@jit`, object-mode fallback | 0.86 s | 86 | 1.1× |
| `@njit`, nopython mode | 0.009 s | 0.9 | **~105×** |

Read those middle and bottom rows together. The object-mode version did "compile" — and bought you 10%. The nopython version bought you 105×. The difference between them is one character in the decorator and whether you noticed the silent fallback. This is why the rule is absolute: `@njit`, always, so a failure is a loud error and not a quiet 1.1×.

## 5. Warmup: the first call pays the compile cost

Section 2 told you compilation is lazy and happens on the first call. That means the *first* call to an `@njit` function is dramatically slower than every subsequent call — it includes the entire compile pipeline (bytecode analysis, type inference, LLVM optimization, code generation). For a simple function this is a few hundred milliseconds; for a large one it can be a second or more. After that first call, the compiled code is cached in the dispatcher for that type signature, and every later call runs at full native speed.

![timeline showing import and decoration with no compile then a first call paying about 0.4 seconds of compile cost then warm calls at fractions of a millisecond and a break even point after about two thousand calls](/imgs/blogs/numba-jit-compiling-python-to-machine-code-5.png)

Figure 5 is the shape you must keep in your head. Three regimes: decoration (free), the first call (expensive — it compiles), and steady state (fast). If you are going to call the function many times, the compile cost amortizes to nothing. If you are going to call it *once*, on a small input, Numba can be a net loss — you paid 0.4 s to compile something that would have run in 0.05 s interpreted.

### The amortization math (when is it worth it?)

Let's make "many times" precise. Let:

- $C$ = the one-time compile cost on the first call (seconds),
- $t_p$ = pure-Python time per call,
- $t_n$ = njit time per call (steady state),
- $N$ = number of calls.

Pure Python total: $N \cdot t_p$. Numba total: $C + N \cdot t_n$. Numba wins when

$$C + N \cdot t_n < N \cdot t_p \quad\Longrightarrow\quad N > \frac{C}{t_p - t_n}.$$

Plug in the worked-example numbers for the 10M-element kernel: $C \approx 0.4$ s, $t_p \approx 0.95$ s, $t_n \approx 0.009$ s. The break-even is

$$N > \frac{0.4}{0.95 - 0.009} \approx 0.43 \text{ calls}.$$

Less than one call — because each call processes 10M elements, so even a *single* call more than pays for the compile. Numba is an obvious win here.

Now flip it. Suppose the function is tiny — it processes 100 elements, so $t_p \approx 9.5\,\mu s$ and $t_n \approx 0.1\,\mu s$. Then

$$N > \frac{0.4}{9.5\times 10^{-6} - 0.1\times 10^{-6}} \approx 42{,}500 \text{ calls}.$$

You need to call that small function **forty-two thousand times** before the compile pays off. If your program calls it five times, Numba made it *slower* by 0.4 s. This is the single most important judgment call: **the compile cost is fixed, so Numba pays off in proportion to total work done in the function, not per call.** Big arrays or many calls: yes. Tiny input called a handful of times: probably not.

#### Worked example: warmup versus steady state, measured

Same machine, the `sum_squares` kernel over a 1M-element `float64` array:

```python
import numpy as np, time
from numba import njit

@njit
def sum_squares(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total

arr = np.random.rand(1_000_000)

t0 = time.perf_counter()
sum_squares(arr)                       # FIRST call — includes compilation
t1 = time.perf_counter()
print(f"first call (compile): {(t1 - t0)*1000:.1f} ms")

t0 = time.perf_counter()
for _ in range(1000):
    sum_squares(arr)                   # warm calls
t1 = time.perf_counter()
print(f"warm call: {(t1 - t0):.3f} ms each".replace("ms each", "ms total / 1000"))
```

Typical output on the named box:

```bash
first call (compile): 412.7 ms
warm call: 0.21 ms each
```

The first call took 413 ms — almost all of it compilation, not computation. Warm calls take 0.21 ms. The compile cost is ~2000× a single warm call here. If you benchmark by timing the first call (a beginner mistake), you will conclude Numba is *slower* than pure Python and walk away. The fix, which we formalize in section 10: **always warm up the function once before you time it.**

### Where the warmup cost actually bites in production

For a long-running service or a batch job that processes a few million rows, the warmup cost is genuinely free — it is paid once at startup and disappears against hours of work. The places it actually hurts are specific and worth naming so you can plan around them:

- **Short-lived CLI tools.** A command-line tool that starts, does a little numeric work, and exits pays the full compile cost on *every invocation* — and the user feels it as a 0.4 s lag before anything happens. This is exactly what `cache=True` is for: the first run after a source change compiles and caches; every run after that loads machine code from disk in milliseconds.
- **Serverless and request handlers.** A cold-start Lambda or a request that triggers a first compile will see a latency spike. Warm the function during initialization (call it once on a dummy input at module load) so the compile happens before the first real request, not during it.
- **Multiprocessing workers without caching.** This is the nastiest one. If you `ProcessPoolExecutor` an njit function across 8 worker processes, *each worker recompiles it independently* — 8 processes, 8 × 0.4 s of compile, all at once, before any real work starts. With `cache=True`, one process compiles and writes the cache; the other seven load it from disk. On an 8-worker pool this turns ~3.2 s of redundant compile into ~0.4 s. Always pair Numba with `cache=True` when you fan it across processes.

The unifying rule: the compile cost is fixed and one-time *per process per signature*, so it only hurts when a process is short-lived or when you spawn many of them. Match the mitigation to the situation — warm at startup for services, `cache=True` for CLIs and worker pools.

## 6. Going multi-core: `parallel=True` and `prange`

So far every win came from one core. But your box has eight. If the loop's iterations are *independent* — iteration `i` does not depend on iteration `i-1` — Numba can split the loop across cores with two changes: pass `parallel=True` to the decorator, and replace `range` with `prange`.

`prange` ("parallel range") tells Numba "these iterations are independent; you may run them on different threads." Numba's parallel backend partitions the index space across a thread pool (it releases the GIL internally, so this is *true* parallelism, not the GIL-bound kind threads normally give you in CPython — the [GIL only blocks pure-Python bytecode](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), and compiled Numba code runs outside it).

Here is a kernel that benefits — an elementwise transform with a branch, applied to a big array:

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def soft_threshold(xs, lam):
    out = np.empty_like(xs)
    for i in prange(xs.shape[0]):        # prange, not range
        v = xs[i]
        if v > lam:
            out[i] = v - lam
        elif v < -lam:
            out[i] = v + lam
        else:
            out[i] = 0.0
    return out
```

The serial version (with plain `range` and no `parallel=True`) already beats pure Python by ~100×. Adding `prange` spreads it across cores on top of that.

![before and after comparison of a serial njit kernel at twelve milliseconds on one core versus a parallel version with prange across eight cores at under two milliseconds a roughly six and a half times speedup](/imgs/blogs/numba-jit-compiling-python-to-machine-code-6.png)

A caveat that figure 6 makes honest: you get **6.5×**, not **8×**, on 8 cores. That is real-world parallel scaling. By **Amdahl's law**, the speedup is capped by the serial fraction:

$$S = \frac{1}{(1-p) + \frac{p}{s}},$$

where $p$ is the parallelizable fraction and $s$ the number of cores. Even at $p = 0.95$ and $s = 8$, $S = 1/(0.05 + 0.95/8) \approx 5.9$. Allocating `out`, thread-pool startup, and memory bandwidth contention eat the rest. For elementwise work especially, you often hit the **memory-bandwidth wall** before you run out of cores — all eight cores are reading and writing the same RAM, and the bus saturates. (That same bandwidth ceiling is why we sometimes drop to [numexpr or a fused native kernel](/blog/software-development/python-performance/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries) for big array expressions.)

There is also a correctness rule with teeth: **`prange` is only valid when iterations are independent.** A loop with a cross-iteration dependency — `out[i] = out[i-1] + xs[i]` (a cumulative sum) — cannot be parallelized this way; the result depends on order. Numba *can* parallelize reductions it recognizes (a sum accumulator into a single scalar is handled correctly), but if you write a genuine sequential dependency and slap `prange` on it, you will get wrong answers silently. Reach for `prange` only on embarrassingly parallel loops.

#### Worked example: cores scaling with prange

The `soft_threshold` kernel on a 50M-element `float64` array, named box, `NUMBA_NUM_THREADS` set explicitly per row, warm timing, median of seven:

| Configuration | Time per call | ×vs pure Python | ×vs 1-core njit | Parallel efficiency |
|---|---|---|---|---|
| Pure Python | 4.8 s | 1.0× | — | — |
| `@njit` serial (1 core) | 12.1 ms | 397× | 1.0× | 100% |
| `@njit(parallel=True)`, 2 cores | 6.4 ms | 750× | 1.9× | 95% |
| `@njit(parallel=True)`, 4 cores | 3.5 ms | 1370× | 3.5× | 86% |
| `@njit(parallel=True)`, 8 cores | 1.9 ms | 2530× | 6.4× | 80% |

Read the efficiency column. Scaling is near-linear to 4 cores (86%) and starts to bend at 8 (80%) as memory bandwidth becomes the bottleneck for this elementwise op. The headline: pure Python to 8-core Numba is roughly **2500×** on this kernel — but notice that **397× of it came from `@njit` alone**, on one core, before any parallelism. Get the single-core compile win first; reach for cores second.

### Seeing what the parallel backend did

When you pass `parallel=True`, Numba's parallel backend tries to fuse adjacent array operations and parallelize loops automatically, and it will tell you exactly what it did if you ask. Call `parallel_diagnostics()` on the compiled function:

```python
soft_threshold.parallel_diagnostics(level=2)
```

The report lists each parallelizable loop it found, which loops it *fused* together (combining two passes over the array into one), and which it could not parallelize and why. This is the tool you reach for when `prange` did not give you the scaling you expected: the diagnostics will show you whether the loop parallelized at all, or whether a hidden dependency or an unfusable allocation blocked it. Reading that report is the parallel equivalent of reading a profiler — it turns "it's not scaling" into a specific, fixable reason.

A practical warning about thread counts: by default Numba uses one thread per logical core, which on a hyperthreaded box means two threads per physical core. For memory-bound elementwise work, hyperthreads rarely help and can hurt (they contend for the same cache and bandwidth), so it is worth measuring with `NUMBA_NUM_THREADS` set to the *physical* core count. The efficiency table above used physical cores for exactly this reason.

## 7. Fast ufuncs with `@vectorize` (and `@guvectorize`)

There is a second decorator worth knowing. Sometimes you do not want to compile a *loop* — you want to write a *scalar* function and have Numba turn it into a full NumPy ufunc: something that broadcasts, accepts an `out=` argument, and works across any array shape, just like `np.sin` or `np.add`. That is `@vectorize`.

You write the kernel for a single element; Numba compiles it and generates the broadcasting machinery:

```python
import numpy as np
from numba import vectorize, float64

@vectorize([float64(float64, float64)], nopython=True)
def discounted(value, rate):
    # scalar logic; Numba makes it a ufunc
    return value / (1.0 + rate)

a = np.random.rand(10_000_000)
r = np.full(10_000_000, 0.05)
out = discounted(a, r)            # broadcasts like a real ufunc
discounted(a, r, out=out)         # supports out= for zero-allocation reuse
```

The signature `[float64(float64, float64)]` says "given two `float64`s, return a `float64`," and you can list several signatures to specialize for `float32`, integers, and so on. The payoff over writing an equivalent NumPy expression: a `@vectorize` ufunc is **one fused pass over memory**. The NumPy expression `value / (1.0 + rate)` allocates a temporary for `1.0 + rate`, then another for the division — two extra passes over 10M elements, hammering memory bandwidth. The compiled ufunc fuses the whole thing into a single loop with no temporaries. For elementwise math with more than one or two operations, `@vectorize` often beats the equivalent NumPy expression by **2 to 4×** purely from avoiding temporaries — and you can add `target='parallel'` to multi-thread it for free.

`@guvectorize` (generalized ufunc) is the heavier sibling: it operates on array *slices* rather than scalars, so you can express things like "for each row, compute a windowed reduction" — operations that aren't strictly elementwise. It is more verbose (you declare a layout signature like `(n),()->(n)` and write into a preallocated output), but it lets you push genuinely non-elementwise per-row logic into native code. Reach for `@vectorize` for elementwise; `@guvectorize` when you need to see a whole sub-array at once.

#### Worked example: `@vectorize` beats the NumPy expression by avoiding temporaries

A small multi-operation elementwise transform — `(a*a + b*b)` then a `sqrt`, the hypotenuse — on two 20M-element `float64` arrays, named box, warm, median of seven. The NumPy expression `np.sqrt(a*a + b*b)` is readable but allocates three temporaries (`a*a`, `b*b`, their sum) and walks 20M elements four times. The `@vectorize` ufunc fuses the whole thing into one pass:

```python
import numpy as np
from numba import vectorize, float64

@vectorize([float64(float64, float64)], nopython=True, fastmath=True)
def hypot_fast(a, b):
    return (a * a + b * b) ** 0.5      # fused into ONE pass, no temporaries
```

| Version | Time per call | Memory passes | ×vs NumPy expr |
|---|---|---|---|
| `np.sqrt(a*a + b*b)` | 138 ms | 4 (+3 temporaries) | 1.0× |
| `@vectorize` ufunc (serial) | 51 ms | 1 | 2.7× |
| `@vectorize(target='parallel')` | 14 ms | 1 | 9.9× |

The serial ufunc is 2.7× faster than the NumPy expression purely from fusing four passes into one — no parallelism, just fewer trips over memory. Add `target='parallel'` and it is nearly 10×. This is the case where `@vectorize` clearly wins: multi-operation elementwise math where NumPy's per-operation temporaries are the cost. (For a single operation like `a + b`, NumPy is already one pass and a ufunc buys nothing — measure before reaching for it.)

## 8. The refinements: `cache`, `fastmath`, and `nogil`

Three flags round out the kit. Each one is a small, well-defined trade.

**`cache=True` — persist the compiled code across runs.** By default the compile cost (section 5) is paid once *per process*. Start a new Python process and the first call recompiles. With `cache=True`, Numba writes the compiled machine code to a `__pycache__` directory beside your source and reloads it on the next run, skipping the whole pipeline:

```python
from numba import njit

@njit(cache=True)
def kernel(xs):
    ...
```

This is essential for two situations: short-lived CLI tools (you do not want every invocation to pay 0.4 s of compile), and multiprocessing workers (without caching, *every* worker process recompiles the same function — `N` workers, `N` compiles; with caching, one worker compiles, the rest load from disk). The cache is keyed on the source and the type signature, so it invalidates correctly when you edit the function.

**`fastmath=True` — let LLVM break IEEE-754 rules for speed.** Strict floating-point math forbids the compiler from reordering operations, because `(a + b) + c` is not bitwise-identical to `a + (b + c)` in IEEE-754. That ban blocks some SIMD and fusion. `fastmath=True` tells LLVM "I do not care about the last bit; reorder and fuse freely." On reduction-heavy or transcendental code it can add another **1.2 to 2×**:

```python
@njit(fastmath=True)
def rms(xs):
    s = 0.0
    for x in xs:
        s += x * x
    return (s / xs.shape[0]) ** 0.5
```

The cost: results may differ from the strict version in the last few bits, and `fastmath` also assumes no NaNs or infinities. Use it for graphics, ML, signal processing — places where you are already approximate. Do *not* use it for financial accumulation or anything where the exact rounding is contractual.

**`nogil=True` — release the GIL inside the compiled function.** Normally even compiled Numba code holds the GIL at its boundaries. With `nogil=True`, the GIL is released while the njit function runs, so you can call it from multiple Python `threading` threads and get *real* parallelism (the compiled code touches no Python objects, so the GIL is not protecting anything during its execution):

```python
from numba import njit

@njit(nogil=True)
def heavy(xs):
    ...    # runs without holding the GIL; callable in parallel from threads
```

This is an alternative to `prange` when you want to manage the threading yourself (for instance, overlapping a Numba kernel with I/O on another thread). `prange` is usually simpler for pure data-parallel loops; `nogil` is for when you are orchestrating threads at the Python level.

## 9. Typed containers and `jitclass`: when you need more than arrays

Section 4 made the point that a plain Python `dict` or `list` of mixed types is a wall for Numba. But sometimes a numeric algorithm genuinely needs a growable container or a small stateful object inside the compiled region — a stack for a flood-fill, a hash map for deduplication, a struct that carries a few fields. Numba has typed answers for these, and knowing them widens the set of algorithms you can compile.

**Typed containers.** `numba.typed.List` and `numba.typed.Dict` are homogeneous containers with a concrete element (or key/value) type, so type inference can pin them. They are the in-kernel replacement for `[]` and `{}`:

```python
from numba import njit, types
from numba.typed import Dict
import numpy as np

@njit
def word_lengths_count(lengths):
    # a typed dict: int64 keys -> int64 values, fully compilable
    counts = Dict.empty(key_type=types.int64, value_type=types.int64)
    for n in lengths:
        counts[n] = counts.get(n, 0) + 1
    return counts
```

This compiles in nopython mode where a plain `{}` would have triggered a `TypingError`. The cost: typed containers carry more overhead than a NumPy array, so when the keys are dense integers a preallocated `np.zeros` array is still faster — use the typed `Dict` only when the key space is sparse or non-integer.

**`jitclass` for stateful kernels.** When your algorithm wants a small object with typed fields and methods compiled together — say a running-statistics accumulator or a node in a tree you walk numerically — `@jitclass` lets you declare one. You give it a field-type specification, and Numba compiles the whole class to native code:

```python
from numba import njit
from numba.experimental import jitclass
from numba import float64, int64

spec = [("count", int64), ("total", float64), ("total_sq", float64)]

@jitclass(spec)
class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, x):
        self.count += 1
        self.total += x
        self.total_sq += x * x

    def variance(self):
        mean = self.total / self.count
        return self.total_sq / self.count - mean * mean

@njit
def run(xs):
    stats = RunningStats()         # a jitclass instance, usable inside njit
    for x in xs:
        stats.update(x)
    return stats.variance()
```

The whole thing — the class, its methods, the loop that drives it — compiles to machine code and you can pass `RunningStats` instances into other njit functions. `@jitclass` is still marked experimental and is more rigid than a normal Python class (every field must be typed up front, no inheritance), so reach for it only when a stateful object genuinely simplifies a numeric kernel. For most work, plain functions over arrays are simpler and just as fast. But when you need it, it is the difference between compiling the algorithm and rewriting it to avoid objects entirely.

The boundary is now clear: Numba does support *typed, homogeneous* containers and *typed* classes — it is the *untyped, heterogeneous, dynamic* versions that it refuses. That is the same rule as everywhere else: give Numba types it can pin, and it compiles; leave them dynamic, and it bails.

## 10. Benchmarking Numba honestly

Every number in this post depends on measuring correctly, and Numba has three measurement traps specific to it. If you take one practical habit from this post, take this checklist — it is the difference between a real result and a fooled one. (The general benchmarking discipline — median of repeats, controlling GC, large-enough inputs — is the same as in the rest of the series; here are the Numba-specific additions.)

**Trap 1: timing the first (compile) call.** As section 5 showed, the first call includes 0.4 s of compilation. If your benchmark times it, you will measure the compiler, not the kernel. **Always call the function once to warm it before timing.**

**Trap 2: letting `timeit` include warmup.** `timeit` runs your statement many times, but if the *first* of those is the cold compile, your average is skewed. Warm explicitly, then time.

**Trap 3: comparing against a too-small input.** On a 100-element array, the per-call Python overhead (boxing the array, the function-call machinery) dominates and Numba's win looks like 5×. On 10M elements, the same kernel shows 100×, because the compile-once cost is spread over real work. **Benchmark on a realistic input size**, not a toy one.

Here is a correct harness:

```python
import numpy as np
from numba import njit

@njit
def kernel(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total

def pure_python(xs):
    total = 0.0
    for x in xs:
        total += x * x
    return total

arr = np.random.rand(10_000_000)     # realistic size

# 1. WARM UP — trigger compilation outside the timed region
kernel(arr)

# 2. Time the warm kernel with repeats, take the median
import timeit
n_py = timeit.repeat(lambda: pure_python(arr), number=1, repeat=5)
n_nb = timeit.repeat(lambda: kernel(arr),      number=10, repeat=5)

print(f"pure python : {min(n_py):.4f} s")
print(f"njit (warm) : {min(n_nb)/10:.6f} s")
print(f"speedup     : {min(n_py) / (min(n_nb)/10):.1f}x")
```

Typical output on the named box:

```bash
pure python : 0.9480 s
njit (warm) : 0.009100 s
speedup     : 104.2x
```

Note the structure: warm first, then time with `repeat` and take the `min` (the cleanest run, least disturbed by the OS scheduler). Report the speedup with the setup attached — input size, dtype, machine, versions — exactly as the table headers in this post do. A speedup quoted without its setup is a number you cannot trust or reproduce.

## 11. What Numba can and cannot compile

Numba is not a general Python compiler. It is a *numeric* compiler, and knowing its boundary saves you hours of fighting `TypingError`s. The rule of thumb: Numba is happiest with the same things NumPy is happy with — scalars, arrays, and the math over them. It is helpless with the things that make Python *Python* — arbitrary objects, dynamic dispatch, the standard library of object-oriented machinery.

![matrix of what Numba can and cannot compile with rows for scalars and math NumPy arrays loops with branches object dicts pandas frames and custom classes and columns for support level and what to do instead](/imgs/blogs/numba-jit-compiling-python-to-machine-code-7.png)

| Construct | Numba support | What to do instead |
|---|---|---|
| `int`, `float`, `bool`, `complex` and math on them | Fully supported — the sweet spot | — |
| NumPy arrays (most dtypes, indexing, slicing, many ufuncs) | First-class | Pass arrays, not lists |
| `for`/`while` loops, `if`/`else` branches | Fully supported — the whole point | This is what NumPy *can't* vectorize |
| Tuples of homogeneous (and small heterogeneous) types | Supported | Fine to use |
| `numba.typed.Dict` / `typed.List` (typed containers) | Supported | Use these, not plain `{}`/`[]` of mixed types |
| Plain Python `dict`/`list` of mixed types | Not supported | Use typed containers or parallel arrays |
| Most `str` / text operations | Largely unsupported | Keep string work in pure Python or use a different tool |
| pandas `DataFrame` / `Series` | Opaque to Numba | Extract `.to_numpy()` first, njit the array, write back |
| Arbitrary Python classes | Mostly rejected | `@jitclass` for simple cases, or [Cython](/blog/software-development/python-performance/cython-typed-python-that-compiles-to-c) |
| Calling arbitrary third-party library functions | Not supported | Inline the math, or keep it outside the kernel |

The pandas row deserves a note because it bites everyone. You cannot pass a DataFrame into an njit function. The pattern that works: pull the columns you need out as NumPy arrays with `.to_numpy()`, pass *those* into your njit kernel, and write the results back into a new column. The njit kernel sees only typed arrays; the DataFrame plumbing stays in pure Python where it belongs.

### The boundary cost: each call into njit has a fixed price

There is one more cost that the can/can't table doesn't capture, and it shapes how you structure Numba code: **crossing the boundary into a compiled function is not free.** When Python calls an njit function, the arguments have to be unboxed — the dispatcher checks the argument types, finds the matching specialization, and converts the Python-level objects into the native representation the compiled code expects (an array becomes a pointer plus shape and stride metadata; a Python `int` is unboxed to a machine integer). On return, the result is boxed back into a Python object. This marshaling is cheap — on the order of a microsecond — but it is *per call*.

The consequence is a structural rule: **do the loop inside the njit function, not around it.** Calling an njit function once on a 10M-element array pays the boundary cost once and runs the whole loop natively — the boundary is amortized to nothing. But calling a tiny njit function 10M times from a Python loop pays the boundary cost 10M times, and that microsecond-per-call dominates: you have rebuilt the per-element interpreter tax at the call boundary instead of inside the loop. The fix is always the same — push the loop *down* into the compiled function so you cross the boundary once. If you find yourself calling an njit function in a hot Python loop, you have the structure inside out.

This is the same lesson as "stay in the array world" from the [vectorization track](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), wearing a different hat: every crossing between interpreted Python and native code has a fixed cost, so cross as few times as possible and do as much work as possible on each crossing.

The decision is best drawn as a tree.

![decision tree for whether to use Numba branching from a profiled hot function into numeric loops which lead to njit and then prange for independent iterations versus string or object heavy code which leads away from Numba toward Cython or Rust](/imgs/blogs/numba-jit-compiling-python-to-machine-code-8.png)

Figure 8 is the rule to keep on a sticky note. Is your hot function a numeric loop over scalars and arrays? If NumPy already vectorizes it cleanly, stop — you do not need Numba. If it does *not* vectorize (a branch, a sequential dependency, an iterative algorithm), reach for `@njit`. If those iterations are independent and you have cores to spare, add `prange`. But if the hot path is string parsing, object manipulation, or class-heavy logic, Numba is the wrong tool entirely — that work belongs in [Cython](/blog/software-development/python-performance/cython-typed-python-that-compiles-to-c) or Rust, which can handle Python objects and arbitrary C interop in ways Numba deliberately won't.

## 12. When Numba shines — and when it doesn't

Let me be opinionated, because the value of knowing a tool is knowing when *not* to use it.

**Numba shines on:**

- **Numeric hot loops that do not vectorize cleanly.** This is the headline use case. Element-wise transforms with branches (`if x > threshold: ... else: ...`), where NumPy would need `np.where` plus masks plus temporaries — the readable loop, compiled, beats the contorted array version *and* is easier to read.
- **Iterative algorithms.** Anything where step `i` depends on step `i-1` — numerical integration, a custom EWMA, a Mandelbrot iteration, a physics simulation step, Newton's method per element, a stencil over a grid. These are fundamentally sequential per element; NumPy can't express them without Python-level loops, and Numba compiles the loop directly.
- **Custom reductions and aggregations** that aren't a built-in NumPy reduction.
- **Embarrassingly parallel numeric work** you want spread across cores with `prange`, with near-zero ceremony.
- **Monte Carlo and simulation kernels**, where you run the same numeric loop millions of times and the compile cost vanishes against the workload.

**Numba does not help (and may hurt) on:**

- **Code NumPy already vectorizes.** If `result = (a * b).sum()` already runs in one C loop, Numba adds a compile cost and matches, at best, what NumPy already does. Don't compile what's already compiled.
- **String, text, and object-heavy code.** Parsing logs, manipulating dicts of dicts, building objects — Numba can't compile it (you'll fight `TypingError`s) and it's not where Numba's strength is anyway. Use Cython or Rust, or just keep it in Python.
- **I/O-bound code.** If your function spends its time waiting on disk or network, compiling the *compute* part wins you nothing — the bottleneck isn't the CPU. That's an [async or threading problem](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), not a Numba problem.
- **Cold paths and one-shot tiny inputs.** The warmup math from section 5: if total work in the function is small, the compile cost dominates and Numba is a net loss. Don't njit a function that runs three times on a thousand elements.
- **Pandas-shaped work.** If your pipeline is DataFrame operations, the answer is usually Polars or vectorized pandas, not extracting arrays to njit — unless there's a genuinely loopy numeric kernel hiding in the middle.

The honest framing for the leverage ladder: Numba sits at rung three, and you reach it *after* you've checked rungs one (algorithm) and two (vectorize). It is the lowest-effort native option — three lines, no build step, no separate compiler toolchain, no `.pyx` file, no Rust crate. That low cost is exactly why it's the *first* native lever to try. If Numba can't compile your hot loop (object/string code) or you need to own a real native library with C interop and full control, *that's* when you climb to Cython, C extensions, or Rust — the subjects of the rest of this track.

### Compile the right algorithm, not the wrong one faster

One trap deserves a paragraph of its own, because it is the most expensive mistake you can make with Numba: **compiling an $O(n^2)$ algorithm does not make it an $O(n)$ algorithm — it just makes the quadratic constant smaller.** Suppose your hot loop does a linear scan to check membership for each element, an $O(n^2)$ nested loop. Compiling it with `@njit` might buy 100× on the constant factor, which feels great on your test input. But if the real input is 100× larger, the quadratic term has grown 10,000×, and your 100× constant-factor win is swamped. The right move was to fix the *algorithm* first — replace the linear scan with a hash-set lookup, turning $O(n^2)$ into $O(n)$ — and only *then* consider compiling the cleaned-up loop. This is rung one before rung three, and it is not optional: Amdahl's law and Big-O both say the algorithm dominates at scale. Numba is a constant-factor lever. It is a magnificent one — 100× is a lot of constant factor — but it cannot fix a bad complexity class. Profile, check the algorithm, *then* compile. The fastest code is the right algorithm compiled, not the wrong algorithm compiled.

## 13. Case studies and real numbers

Concrete, named results so you can calibrate expectations.

**The branchy element-wise kernel (the canonical win).** A soft-threshold / clamp operation over a 50M-element `float64` array (section 6's `soft_threshold`). Pure Python: 4.8 s. The "clever" NumPy version using `np.where` twice plus masks: 0.18 s (27×) but allocating three temporaries and walking memory five times. The `@njit` version of the *plain readable loop*: 0.012 s on one core (400×), and 0.0019 s on 8 cores with `prange` (2500×). The Numba version is both faster *and* more readable than the NumPy contortion — that combination is exactly Numba's pitch.

**Mandelbrot / iterative kernels.** The classic Numba demo (and a fair one) is the Mandelbrot set: for each pixel, iterate $z \leftarrow z^2 + c$ until it escapes. This is purely sequential per pixel and cannot be vectorized across iterations. Reported community benchmarks routinely show **100 to 300×** from `@njit` over pure Python on this kernel, and near-linear `prange` scaling because pixels are independent. It is representative of the whole class of per-element iterative algorithms.

**The pandas-to-arrays pattern.** A common real win: a financial or scientific pipeline has a DataFrame with a column that needs a custom rolling computation with a branch (not a standard `.rolling()` aggregation). Done with `df.apply(func, axis=1)`, it crawls — `apply` with a Python function is a [per-row interpreter loop](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), often the slowest thing in the job. Extract the column with `.to_numpy()`, run an `@njit` kernel over the raw array, assign the result back. Speedups of **50 to 150×** over the `apply` version are routine, because you replaced a per-row Python dispatch with one compiled loop.

**The pairwise-distance / N-body kernel.** Computing all pairwise distances in a point cloud, or one step of an N-body simulation, is an $O(n^2)$ double loop with a square root in the inner body. NumPy *can* express it with broadcasting — `np.sqrt(((pts[:,None] - pts[None,:])**2).sum(-1))` — but that materializes an $n \times n \times d$ intermediate array, which for $n = 10{,}000$ points in 3-D is 2.4 GB of temporary, blowing memory before it finishes. The `@njit` version writes the plain double loop, allocates only the $n \times n$ output, and runs at native speed with no giant temporary. On a 5,000-point cloud the njit double loop is both faster than the broadcasting version *and* uses an order of magnitude less memory — a case where Numba wins on RSS, not just wall-clock, because it never builds the temporary that NumPy's vectorized form requires.

**Where Numba is used in production.** Numba underpins numeric-heavy parts of the scientific Python stack — it is the compilation backend in tools across the PyData ecosystem for accelerating user-defined numeric kernels, and it is the reason a researcher can write a custom loop in a notebook and get C-class speed without leaving Python. The trade-off in production is the compile cost (paid once, mitigated by `cache=True`) and the dependency on LLVM — which is heavy, but a single well-supported wheel install.

A calibration note on honesty: the "100×" headline is for kernels where the per-element interpreter tax dominated — scalar loops over millions of elements. The win is smaller (10–30×) when the loop body is already heavy (calls into NumPy ufuncs that were partly C), and it disappears for code NumPy already vectorized. Always measure your own kernel; the range of plausible wins is wide and depends entirely on how much interpreter tax there was to remove.

## 14. A complete worked flow

Let's put it together end to end: profile, identify the loopy numeric hot path, apply `@njit`, then `prange`, then measure. This is the optimization loop on a realistic kernel — a simple exponential moving average with a reset condition, which is genuinely iterative (each output depends on the last) and therefore *cannot* be NumPy-vectorized, the perfect Numba case.

```python
import numpy as np
from numba import njit

# Pure Python: iterative, sequential dependency — NumPy can't vectorize this
def ewma_reset(xs, alpha, reset_thresh):
    out = np.empty_like(xs)
    prev = xs[0]
    out[0] = prev
    for i in range(1, xs.shape[0]):
        if xs[i] > reset_thresh:
            prev = xs[i]                    # reset on spike
        else:
            prev = alpha * xs[i] + (1 - alpha) * prev
        out[i] = prev
    return out
```

Profiling a job that calls this on a 20M-element series shows it eating 85% of wall time — a textbook hot path. It is sequential (note `prev` carries across iterations), so NumPy is out. Numba is the answer. The change is one line:

```python
@njit(cache=True, fastmath=True)            # the only change: the decorator
def ewma_reset(xs, alpha, reset_thresh):
    out = np.empty_like(xs)
    prev = xs[0]
    out[0] = prev
    for i in range(1, xs.shape[0]):
        if xs[i] > reset_thresh:
            prev = xs[i]
        else:
            prev = alpha * xs[i] + (1 - alpha) * prev
        out[i] = prev
    return out
```

Note we did *not* add `prange` — the loop has a true sequential dependency (`prev`), so it is not parallelizable, and using `prange` here would silently produce wrong answers. This is the discipline: `prange` only on independent iterations. We did add `cache=True` (so re-running the job skips recompile) and `fastmath=True` (the EWMA tolerates last-bit differences).

Measured on the named box, 20M elements, warm:

| Version | Time | ×speedup | Note |
|---|---|---|---|
| Pure Python loop | 9.6 s | 1.0× | the hot path |
| NumPy attempt | — | — | impossible — sequential dependency |
| `@njit` (warm) | 0.038 s | **253×** | one decorator line |
| `@njit` + `fastmath` | 0.029 s | **331×** | last-bit tolerant |

A 9.6-second hot path became 29 milliseconds, the whole job went from "go get coffee" to instant, and the change was one decorator on a function we did not otherwise touch. That is the Numba experience when the loop fits: enormous leverage for almost no code, on exactly the kernels the array world can't reach.

Notice what *didn't* change. We profiled first, so we knew this one function was 85% of the wall-clock — we did not blindly decorate everything. We left the rest of the pipeline (the loading, the cleaning, the aggregation) in vectorized pandas/NumPy where it already belonged. We checked that the loop genuinely could not vectorize before reaching for Numba at all. And we measured the warm path, not the first call, so the 253× we reported is a number you could actually reproduce. That discipline — profile, confirm the lever fits, apply it to the hot path only, measure the warm result — is the whole method of this series compressed into one function. Numba did not replace the method; it was the right *lever* once the method pointed at the bottleneck.

## 15. When to reach for this (and when not to)

A decisive summary, because every technique is a cost and you should know when to pay it.

**Reach for Numba when:**

- You have *profiled* and the hot path is a numeric loop over scalars or arrays.
- The loop does *not* vectorize cleanly in NumPy — it has branches, a sequential dependency, or is an iterative algorithm.
- The total work in the function is large enough to amortize the ~0.3–1 s compile cost (big arrays, or many calls).
- You want native speed with near-zero ceremony — no build step, no separate language.
- The iterations are independent and you have idle cores (then add `prange`).

**Do not reach for Numba when:**

- NumPy already vectorizes the operation — you'd add a compile cost to match what's already compiled.
- The hot path is strings, text parsing, dicts of dicts, or object-oriented logic — Numba can't compile it; use Cython or Rust or stay in Python.
- The code is I/O-bound — compiling the compute wins nothing; the bottleneck is the wait.
- The function runs a few times on tiny inputs — the compile cost dominates and Numba is a net loss.
- Your pipeline is DataFrame-shaped — reach for Polars or vectorized pandas first; only extract-and-njit a genuine numeric loop hiding inside it.

The meta-rule, the same one that runs through this whole series: don't guess, measure; rewrite the hot 1% in native code, not 100%; and prove the win with a before→after number on a realistic input. Numba is the lowest-effort way to do the "rewrite the hot 1% in native" step — which is exactly why it is the first native lever you reach for, and why, when it fits, three lines really do buy you 100×.

## 16. Key takeaways

- A pure-Python numeric loop pays a per-element interpreter tax — boxing, type dispatch, allocation, refcounting, eval-loop overhead — that is ~99% of the time and dwarfs the actual math. `@njit` deletes all of it by compiling a *typed* loop to LLVM machine code.
- **Always use `@njit`, never bare `@jit`.** `@njit` = `nopython=True`, which raises a loud error if it can't fully compile, instead of silently falling back to object mode and giving you ~1.1× while you believe you optimized something.
- Numba compiles **lazily, per argument-type signature, on the first call.** That first call pays the compile cost (~0.3–1 s); every later call runs at native speed. Benchmark only *warm* calls.
- The compile cost amortizes in proportion to total work: worth it after $N > C/(t_p - t_n)$ calls. Big arrays or many calls — yes; tiny input called a few times — no.
- `parallel=True` + `prange` spreads *independent* iterations across cores for near-linear scaling (expect ~6–7× on 8 cores by Amdahl + bandwidth, not 8×). Never use `prange` on a loop with a cross-iteration dependency — it silently corrupts results.
- `@vectorize` turns a scalar kernel into a fused, broadcasting ufunc that avoids the temporaries a NumPy expression would allocate; `@guvectorize` handles per-slice (non-elementwise) work.
- `cache=True` persists compiled code across runs and processes (essential for CLIs and multiprocessing workers); `fastmath=True` buys extra SIMD by relaxing IEEE-754 (only where you tolerate last-bit differences); `nogil=True` releases the GIL for thread-level parallelism.
- Numba's sweet spot is the same as NumPy's data — scalars, arrays, numeric loops — and its blind spot is everything that makes Python dynamic: object dicts, pandas frames, arbitrary classes, most strings. For those, climb to Cython or Rust.
- Numba is rung three of the leverage ladder and the lowest-effort native option: try it *after* algorithm and vectorization, and reach past it to Cython/Rust only when the hot path is object/string code or you need a real native library.

## Further reading

- **Numba documentation** — the official user manual, especially "A ~5 minute guide to Numba," the "Supported Python features" and "Supported NumPy features" reference (the authoritative can/can't list), and the `@vectorize`/`@guvectorize`/`parallel` guides.
- **LLVM language reference** — to understand the IR Numba targets and why the same backend that powers Clang and Rust can optimize your loop with SIMD and register allocation.
- **"High Performance Python," 2nd ed., by Micha Gorelick & Ian Ozsvald** — the Numba and compilation chapters, with measured before→after benchmarks in the same spirit as this post.
- The CPython `dis` and `timeit` docs — for inspecting the bytecode Numba consumes and for benchmarking the warm path honestly.
- Within this series: the [native-acceleration landscape](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python) for where Numba sits among Cython, C, and Rust; [Cython, typed Python that compiles to C](/blog/software-development/python-performance/cython-typed-python-that-compiles-to-c) for when the hot path is object/string-heavy or needs C interop; [when NumPy isn't enough](/blog/software-development/python-performance/when-numpy-isnt-enough-numexpr-bandwidth-and-avoiding-temporaries) for the bandwidth wall that pushes you off the array world; and [NumPy from first principles](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) for the array model Numba is built to accelerate. The series intro, [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), frames the whole leverage ladder this post climbs.

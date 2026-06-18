---
title: "Cython: Typed Python That Compiles to C"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Turn a hot pure-Python kernel into compiled C one type annotation at a time, using cdef, typed memoryviews, the cython annotate map, and nogil parallelism with measured before-and-after numbers."
tags:
  [
    "python",
    "performance",
    "optimization",
    "cython",
    "native-code",
    "memoryview",
    "nogil",
    "profiling",
    "compilation",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/cython-typed-python-that-compiles-to-c-1.png"
---

There is a particular kind of stuck that every Python performance engineer eventually hits. You have profiled the job. You have found the hot path. It is a numeric loop — a Haversine distance over a few million GPS points, a local image filter, a financial Monte Carlo inner step, a custom string parser. You have already tried the easy levers. NumPy cannot vectorize it cleanly because the loop has a data-dependent branch, or it touches a C library, or each element depends on the previous one. The function is genuinely 70% of your runtime, so by [Amdahl's law](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) it caps your possible win — and it is still pure, boxed, interpreted Python running at roughly one operation every hundred nanoseconds. You are paying the full interpreter tax on the one loop that matters, and you cannot vectorize your way out.

This is exactly the situation Cython was built for. Cython is a superset of Python: every line of valid Python is valid Cython, but you can *add C type declarations* — `cdef int i`, `cdef double total` — and where you do, Cython compiles that code to C that runs at native speed, with no interpreter, no boxing, no per-operation type dispatch. You do not rewrite the function in another language. You take the same `.py`, rename it `.pyx`, sprinkle types onto the variables in the hot loop, run a build step, and the inner loop that took 95 ns per iteration now takes under 1 ns. Same logic, same readability, 100× faster — because the bottleneck loop is now real machine code. Figure 1 shows the contrast we will spend this post earning.

![before and after comparison of a pure Python loop with boxed integer objects and an eval loop dispatch versus a Cython cdef typed loop that compiles to a raw C loop with no dispatch dropping from ninety five nanoseconds per add to under one nanosecond per add](/imgs/blogs/cython-typed-python-that-compiles-to-c-1.png)

This is the third rung of the leverage ladder we have been climbing all series — *compile the hot 1%* — and Cython is the lever for the case where you want **control**. Its sibling lever, [Numba](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code), JIT-compiles a numeric function in three lines with zero build step and is often the right first try. Cython asks more of you — a build step, a `.pyx` file, two languages in one project — and pays you back with things Numba cannot easily give: gradual typing of a *large* existing module, direct calls into existing C and C++ libraries, fine control over exactly which lines stay Python and which become C, and a `nogil` escape hatch for true multi-core threading. By the end of this post you will be able to take a hot Python kernel, turn it into a `.pyx`, read the `cython -a` annotation to find the slow lines, type them away one at a time, drop the Python safety checks with directives, and release the GIL to use every core — measuring the win at each step. We will keep returning to one running example: a Haversine-distance kernel over a few million coordinate pairs, the kind of thing that shows up in every geospatial pipeline and refuses to vectorize as cleanly as you would like.

A note on the machine for every number in this post: an 8-core x86-64 Linux box (a desktop Ryzen-class CPU), CPython 3.12, Cython 3.0, GCC 13, 16 GB RAM, results reported as the median of repeated `timeit` runs after a warmup. The numbers are representative of what this class of kernel does on this class of hardware; your exact figures will move with input size, dtype, and compiler flags, but the *ratios* — the 100× from typing, the extra 1.3× from dropping bounds checks, the near-linear scaling from `prange` — are stable and reproducible.

## 1. What Cython actually is (and what "compiles to C" means)

Let us be precise, because "compiles to C" gets thrown around loosely. When you run CPython on a normal `.py` file, your source is compiled to *bytecode* — a compact instruction set for a virtual stack machine — and then the [CPython eval loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) interprets that bytecode one instruction at a time. Every `a + b` becomes a `BINARY_OP` bytecode that, at runtime, inspects the types of `a` and `b`, looks up the addition slot in their type objects, allocates a new boxed object for the result, and adjusts reference counts. That dispatch-and-box dance is the interpreter tax, and it is why a tight Python loop is slow regardless of how simple the arithmetic looks.

Cython takes a different path entirely. A `.pyx` file is fed to the Cython compiler (`cythonize`), which **transpiles it to C source code** — a `.c` file, often thousands of lines, full of calls into the CPython C API. That C file is then handed to a normal C compiler (GCC or Clang), which compiles it to a native shared object (`.so` on Linux/macOS, `.pyd` on Windows). You `import` that shared object from Python exactly like any other module. So the chain is: `.pyx` → Cython → `.c` → C compiler → `.so` → `import`. There is no interpreter in the final loop if you have typed it: the `.c` Cython generated for a typed loop is a plain C `for` loop over machine integers, and the C compiler optimizes it like any C code — register allocation, loop unrolling, SIMD, the works.

Here is the crucial distinction that makes Cython worth learning. *Untyped* Cython code still goes through the Python C API at runtime — it generates C that calls `PyNumber_Add`, `PyObject_GetItem`, and friends, which is barely faster than the interpreter (the win is only avoiding bytecode dispatch overhead, maybe 1.3–2×). The big speedup comes entirely from **type declarations**. When you write `cdef int i`, you are promising Cython that `i` is a C `int`, not a Python object. Now Cython does not need to box it, does not need to refcount it, does not need to dispatch on its type — it emits `int i;` in C and a raw `i++` to increment it. The variable lives in a CPU register. *This is the whole game*: the more of your hot loop you can express in terms of C types instead of Python objects, the more of it runs as pure C, and the less of it pays the Python tax. Cython gives you a *dial* — from "fully dynamic Python" to "pure C" — and lets you turn it precisely where it matters and nowhere else.

That dial is the reason Cython coexists with Numba rather than being replaced by it. Numba JIT-compiles a whole function at runtime based on the types it sees on the first call — all or nothing, magic that works beautifully for self-contained numeric functions and is opaque when it does not. Cython is *ahead-of-time* and *explicit*: you decide what is typed, you can see the generated C, you can call into any C library, and you can type a 2,000-line module incrementally, profiling and typing the hot 5% while the other 95% stays as ordinary Python that happens to compile. It is the difference between a JIT that decides for you and a compiler that does exactly what you tell it. When you need that control, Cython is the answer.

### The cost model, stated plainly

To know where Cython will help, you need the cost model in your head. A pure-Python integer add costs on the order of 50–100 ns once you count bytecode fetch, dispatch, the boxing of the result, and refcount churn. A C integer add costs well under 1 ns — often it is free because it pipelines with other work. So the headroom on a tight integer loop is roughly two orders of magnitude. A pure-Python `arr[i]` on a NumPy array is even worse than a plain add: it calls the array's `__getitem__`, which constructs a Python scalar object wrapping the element, boxing a single number into a 32-byte heap object just so you can read it. Cython with a typed memoryview turns that same `arr[i]` into a single pointer dereference — `base[i * stride]` in C — with no object created at all. That is where the 100× lives, and the rest of this post is about how to claim it.

Let us make the two-orders-of-magnitude claim rigorous rather than hand-waved, because the whole strategy of this post rests on it. Take one iteration of a tight loop that does `total += arr[i]`. In pure Python, that single line is several bytecodes — load `arr`, load `i`, `BINARY_SUBSCR` (the subscript), load `total`, `BINARY_OP` (the add), store `total`. Each bytecode costs a fetch-decode-dispatch in the eval loop (a `switch` over the opcode, several memory reads, a computed jump), call it $\approx 5$–$15$ ns of pure interpreter overhead *per bytecode*. The `BINARY_SUBSCR` additionally calls the array's C-level subscript handler, which allocates a fresh boxed `PyObject` for the element — a `malloc`, a header initialization, and a refcount set, another $\approx 30$–$50$ ns. The `BINARY_OP` does a type dispatch (look up `tp_as_number->nb_add` on each operand's type), allocates a *result* object, and updates refcounts. Then that result object is decref-ed when `total` is rebound, possibly triggering a free. Sum it up and one iteration is comfortably $80$–$150$ ns and allocates two-plus temporary objects. Now type it: `cdef double total; cdef double[::1] arr`. The same line compiles to C as `total += arr[i];` — one `mov` to load `arr[i]` from memory, one `addsd` to add it into a register holding `total`. That is $\approx 0.5$–$1$ ns and zero allocations. The ratio is exactly the $100$× we keep quoting, and now you can see precisely where each factor of it comes from: roughly $10$× from killing bytecode dispatch, another $\approx 10$× from killing the boxing and refcount churn, with the C compiler's register allocation and pipelining on top. Cython's job is to let you remove each of those costs by naming a type.

## 2. The running example: a Haversine kernel in pure Python

Let us get concrete with the kernel we will optimize the whole way down. The Haversine formula computes the great-circle distance between two latitude/longitude points on a sphere. We have arrays of origin and destination coordinates — say four million ride-share trips — and we want the distance of each. Here is the honest pure-Python version, the one you would write first and then profile.

```python
import math

def haversine_py(lat1, lon1, lat2, lon2):
    # all four are lists of floats, length n
    n = len(lat1)
    out = [0.0] * n
    R = 6371.0  # earth radius in km
    for i in range(n):
        phi1 = math.radians(lat1[i])
        phi2 = math.radians(lat2[i])
        dphi = math.radians(lat2[i] - lat1[i])
        dlam = math.radians(lon2[i] - lon1[i])
        a = (math.sin(dphi / 2.0) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        out[i] = R * c
    return out
```

This is correct, readable, and slow. On the named machine, with `n = 4_000_000` lists of Python floats, it runs in about **3.9 seconds** — roughly 975 ns per element. Every iteration boxes and unboxes floats, calls into `math.*` through the full Python call machinery, indexes lists through `__getitem__`, and pays bytecode dispatch on each of the dozen-odd operations. Profile it and you would find essentially all the time in the loop body, with no single villain — it is death by a thousand small Python operations, which is the classic signature of a kernel that wants to be compiled.

Why not just NumPy? In this particular case you *could* vectorize Haversine — it is elementwise — and you should if you can; a NumPy version using `np.sin`, `np.cos`, `np.arctan2` runs in roughly 180 ms because each ufunc is one C loop over a packed buffer. But hold that thought: real kernels often have a data-dependent branch (skip points inside a geofence, clamp on a threshold, early-out on a condition) or a step where element `i` depends on element `i-1` (a running filter, a recurrence), and *those do not vectorize*. The moment your loop body has an `if` that depends on the data or a carried dependency, NumPy's whole-array ufunc model breaks down and you are back to a Python loop. Cython handles those cases without complaint — it is just a C loop, and a C loop can branch and carry state at full speed. So we will optimize the clean version to learn the mechanics, then I will show you the branchy variant that *only* Cython and Numba can save.

## 3. The build flow: pyx, cythonize, setup, and the shared object

Before we type anything, you need to be able to build a `.pyx`. This trips people up more than the typing does, so let us make it concrete. Figure 2 shows the full pipeline.

![graph of the Cython build flow showing a typed pyx source feeding the cythonize transpile step which emits generated C source, with the Python C API headers also feeding the C compiler, the compiler producing a shared object, and that shared object being imported from Python](/imgs/blogs/cython-typed-python-that-compiles-to-c-2.png)

The simplest possible build uses a `setup.py` with `cythonize`. Put your kernel in `haversine.pyx` and write:

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="haversine",
    ext_modules=cythonize(
        "haversine.pyx",
        annotate=True,          # also emit the HTML annotation report
        compiler_directives={
            "language_level": "3",
        },
    ),
    include_dirs=[numpy.get_include()],
)
```

Then build the extension in place so you can `import haversine` from the same directory:

```bash
python setup.py build_ext --inplace
```

This runs the whole chain. `cythonize` transpiles `haversine.pyx` to `haversine.c`, the C compiler turns that into something like `haversine.cpython-312-x86_64-linux-gnu.so`, and because `annotate=True` you also get a `haversine.html` you can open in a browser (the annotation map we will read in section 5). The `--inplace` flag drops the `.so` next to your source instead of in a `build/` tree, so the import just works. After this, from Python:

```python
import haversine
out = haversine.haversine_cy(lat1, lon1, lat2, lon2)
```

A modern alternative is to declare the extension in `pyproject.toml` and let the build backend handle it, which is cleaner for distributable packages:

```python
# in pyproject.toml, the build-system table
[build-system]
requires = ["setuptools>=61", "Cython>=3.0", "numpy"]
build-backend = "setuptools.build_meta"
```

…with a small `setup.py` (or `setup.cfg`) declaring the `cythonize` call as above. For quick experiments in a notebook you can skip all of this with the IPython magic `%load_ext Cython`, then put `%%cython` at the top of a cell and write your `.pyx` code directly — it compiles and imports the cell transparently. That is the fastest way to iterate on a kernel before you commit to a `setup.py`. The annotation magic `%%cython --annotate` even shows the colored report inline.

The key mental shift: **there is now a build step**. A `.pyx` is not run; it is compiled. When you change it, you must rebuild before the change takes effect. This is the price of admission for Cython, and it is the single biggest reason to reach for Numba instead when you do not need Cython's other powers — Numba has no build step at all. But the build step is also what gives you a real shared object you can ship in a wheel, debug with C tools, and link against C libraries. Choose with eyes open.

## 4. Typing the kernel: cdef, cpdef, and the typed memoryview

Now the actual optimization. The pure-Python loop is slow for two reasons: the *scalars* are boxed Python objects, and the *array access* goes through Python's `__getitem__`. Cython lets us fix both. We will climb the typing ladder one rung at a time — Figure 3 — measuring each rung.

![stack diagram of the Cython typing ladder showing plain Python at one times baseline, then a pyx with cdef typed scalars at thirty times, then a typed memoryview giving C array access at ninety times, then boundscheck off at one hundred thirty times, then a nogil block enabling real threads at over one hundred fifty times](/imgs/blogs/cython-typed-python-that-compiles-to-c-3.png)

First, `cdef` versus `cpdef` versus `def`. A `def` function is a normal Python function, callable from Python, with Python-object arguments — no speedup on the call itself. A `cdef` function is a pure C function: fast to call, but **not visible from Python** (only callable from other Cython code in the same module). A `cpdef` function generates *both* — a fast C version for internal calls and a Python wrapper so you can call it from Python too. The rule of thumb: make your top-level entry point a `def` or `cpdef` (so Python can call it), and make hot inner helpers `cdef` (so the per-call overhead vanishes). A `cdef` call compiles to a direct C function call — push arguments, jump — versus a `def` call which builds an argument tuple, a frame object, and goes through the full Python call protocol, costing on the order of 50–80 ns per call before the body even runs.

Now the scalars. We declare every loop-local with a C type:

```cython
# haversine.pyx
cimport cython
from libc.math cimport sin, cos, sqrt, atan2, M_PI

cdef double _radians(double deg) nogil:
    return deg * M_PI / 180.0

def haversine_naive_cy(lat1, lon1, lat2, lon2):
    cdef Py_ssize_t i, n = len(lat1)
    cdef double R = 6371.0
    cdef double phi1, phi2, dphi, dlam, a, c
    out = [0.0] * n
    for i in range(n):
        phi1 = _radians(lat1[i])
        phi2 = _radians(lat2[i])
        dphi = _radians(lat2[i] - lat1[i])
        dlam = _radians(lon2[i] - lon1[i])
        a = (sin(dphi / 2.0) ** 2
             + cos(phi1) * cos(phi2) * sin(dlam / 2.0) ** 2)
        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        out[i] = R * c
    return out
```

Two things changed. We `cimport`-ed the C math functions from `libc.math` — these compile to direct calls to the C standard library's `sin`, `cos`, etc., with no Python `math` module indirection. And we typed every loop variable with `cdef double`. Notice `Py_ssize_t` for the index: it is the correct C type for a length/index in CPython, signed and machine-width, and using it avoids subtle conversion overhead. The helper `_radians` is `cdef double ... nogil` — a pure C function. This version runs in about **1.1 seconds**, roughly 3.5× faster than pure Python. Good, but not the 100× we promised. Why only 3.5×?

Because we left the *worst* part untyped: `lat1[i]`. The arguments `lat1, lon1, lat2, lon2` are still Python lists (or NumPy arrays), and `lat1[i]` still calls `__getitem__`, which constructs a boxed Python float for every single access — four boxings per iteration, four million iterations, sixteen million temporary float objects allocated and freed. That allocation churn dominates. The `out[i] = ...` assignment is just as bad: it boxes the result float and stores it in a Python list, paying refcount and allocation each time. **The typed scalars are fast; the array boundary is the bottleneck.** To fix it we need the headline feature: typed memoryviews.

### Typed memoryviews: indexing like a C array

A **typed memoryview** is Cython's interface to the buffer protocol — the same low-level mechanism that lets NumPy, `array.array`, `bytes`, and Arrow share raw memory without copying. You declare a parameter as `double[:] arr` and Cython binds it directly to the underlying contiguous C buffer of whatever array-like you pass (a NumPy `float64` array, a typed `array.array`, anything supporting the buffer protocol). Indexing `arr[i]` on a memoryview does **not** call `__getitem__` and does **not** box anything — it compiles to a raw pointer arithmetic expression, `(*(base + i * stride))` in C, exactly like indexing a C array. Read or write, it is one memory operation. Here is the kernel with memoryviews:

```cython
# haversine.pyx
cimport cython
from libc.math cimport sin, cos, sqrt, atan2, M_PI
import numpy as np

cdef inline double _radians(double deg) nogil:
    return deg * M_PI / 180.0

def haversine_mv(double[::1] lat1, double[::1] lon1,
                 double[::1] lat2, double[::1] lon2):
    cdef Py_ssize_t i, n = lat1.shape[0]
    cdef double R = 6371.0
    cdef double phi1, phi2, dphi, dlam, a, c
    cdef double[::1] out = np.empty(n, dtype=np.float64)
    for i in range(n):
        phi1 = _radians(lat1[i])
        phi2 = _radians(lat2[i])
        dphi = _radians(lat2[i] - lat1[i])
        dlam = _radians(lon2[i] - lon1[i])
        a = (sin(dphi / 2.0) * sin(dphi / 2.0)
             + cos(phi1) * cos(phi2) * sin(dlam / 2.0) * sin(dlam / 2.0))
        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        out[i] = R * c
    return np.asarray(out)
```

Look at the parameter types: `double[::1] lat1` is a *one-dimensional, C-contiguous, double-precision* memoryview. The `[::1]` means "contiguous in the last (only) dimension" — it tells Cython the stride is exactly one element, so indexing is a plain `base[i]` with no per-access stride multiply, the fastest possible layout. (A plain `double[:]` allows non-contiguous strided arrays and is slightly slower per access because it multiplies by the stride.) The output is allocated as a NumPy array and bound to a memoryview `out` so writes are also raw stores; we return `np.asarray(out)` to hand back a normal array. I also replaced `x ** 2` with `x * x` — the `**` operator on doubles in Cython can fall back to `pow()`, which is slower than a multiply; for squaring, multiply explicitly.

This version runs in about **42 milliseconds** — roughly **93× faster** than pure Python, and faster than the NumPy version (180 ms) because we fused all the math into one pass over memory instead of NumPy's many passes (each ufunc reads and writes a full temporary array). *That* is the typed-memoryview payoff: the array access is now C, the scalars are C, the math is C, and the only Python left is the function call boundary at the very top, paid once. The 100× was real; it lived in the array boundary, and the memoryview is the key that unlocks it. The `cimport cython` at the top and the `cdef inline` on the helper let the C compiler inline the radians conversion straight into the loop, removing even the call overhead.

Why does the fused Cython loop beat NumPy here, when NumPy is itself C loops over packed buffers? The answer is memory bandwidth, and it is worth a paragraph because it reframes when each tool wins. NumPy evaluates an expression like `R * 2 * arctan2(sqrt(a), sqrt(1 - a))` by computing each ufunc into a full temporary array: `dphi` is one array pass, `sin(dphi/2)` another, the squared term another, `a` another, and so on through a dozen intermediate arrays for our formula. Each pass reads its inputs from RAM and writes its output back to RAM. With $n = 4{,}000{,}000$ float64 elements, each array is $32$ MB — far larger than the few megabytes of CPU cache — so every pass streams the full $32$ MB to and from main memory. A dozen passes is on the order of $a$ dozen $\times 64$ MB $\approx 800$ MB of memory traffic, and at a realistic single-threaded bandwidth of $\approx 10$ GB/s that traffic alone is $\approx 80$ ms, which lines up with NumPy's $180$ ms. The Cython loop instead reads each input element **once**, does *all* the math on it while it sits in a register, and writes the one output — a single pass, $\approx 160$ MB of traffic total, a fraction of NumPy's. We are not faster because our arithmetic is faster (NumPy's ufuncs are excellent C); we are faster because we **moved less memory**. This is the same memory-bound-versus-compute-bound distinction the rest of the series leans on, and it explains a counterintuitive fact: hand-fused Cython routinely beats a chain of NumPy ufuncs on large arrays, while NumPy wins on small arrays (where call overhead dominates and the arrays fit in cache so the extra passes are nearly free). Know which regime you are in.

For 2D arrays the syntax extends naturally: `double[:, ::1] image` is a 2D C-contiguous memoryview (row-major, last axis contiguous), indexed `image[r, c]`. This is how you write fast image filters, matrix kernels, and stencils in Cython — full random access into a 2D buffer at C speed, with the bounds and strides handled for you.

#### Worked example: the typing ladder, measured

Let us put the whole ladder in one table so the contribution of each rung is unambiguous. Same machine, `n = 4_000_000` float64 inputs, median of 7 runs, GC accounted for.

| Version | What changed | Wall-clock | ns/element | Speedup vs pure Python |
| --- | --- | --- | --- | --- |
| `haversine_py` | pure Python, lists | 3.90 s | 975 ns | 1.0× |
| `haversine_naive_cy` | `cdef` scalars, libc math | 1.10 s | 275 ns | 3.5× |
| NumPy vectorized | `np.sin`/`np.cos`/`np.arctan2` | 180 ms | 45 ns | 21.7× |
| `haversine_mv` | typed memoryviews | 42 ms | 10.5 ns | 92.9× |
| `haversine_mv` + nobounds | directives (section 6) | 31 ms | 7.75 ns | 125.8× |
| nogil + `prange`, 8 cores | parallel (section 7) | 6.2 ms | 1.55 ns | 629× |

Read the table as a story. Typing the scalars alone bought 3.5× — real, but capped by the array boundary still being Python. The memoryview removed that boundary and jumped us to 93×, *past* NumPy, because we fused the whole computation into one memory pass. Dropping the safety checks added another 1.35×. And releasing the GIL to use all 8 cores took us to 629× over the original. Every one of those steps is a single, local edit to the `.pyx`. That is the Cython promise made concrete: incremental, measurable, and you keep the readable structure of the original loop the entire way.

## 5. Reading `cython -a`: the annotation is your optimization map

How did I know the array boundary was the bottleneck in `haversine_naive_cy` before measuring? The **annotation report**. When you compile with `annotate=True` (or run `cython -a haversine.pyx` directly, or use `%%cython --annotate`), Cython emits an HTML file that shows your source with **every line shaded yellow in proportion to how much Python C-API machinery it generates**. White lines compiled to pure C with no Python interaction — they are essentially free. Faintly yellow lines touch the Python layer a little. Deeply yellow lines call back into the interpreter on every execution and are where your time goes. Click any line and it expands to show the generated C, so you can see *exactly* what is slow. Figure 4 captures the contrast.

![before and after comparison of reading the cython annotate report showing a yellow line that creates a PyObject and calls the C API every iteration where ninety percent of the time goes versus a white line with a cdef typed C variable and no API calls that is nearly free at one nanosecond per operation](/imgs/blogs/cython-typed-python-that-compiles-to-c-4.png)

This single tool changes how you optimize Cython. You do not guess; you open the report and **hunt the yellow**. In `haversine_naive_cy`, the scalar arithmetic lines were nearly white (we typed them), but the lines containing `lat1[i]` and `out[i] = ...` were bright yellow — clicking them showed calls to `__Pyx_GetItemInt` and the list-store machinery, confirming the boxing on every access. The annotation told us precisely where to spend effort: convert those array accesses to memoryviews, and indeed in `haversine_mv` those same lines turn white. The report is a closed feedback loop: annotate, find the yellowest hot line, type it away, re-annotate, repeat until the loop body is white. When the inner loop is all white, you have extracted essentially all the available speedup and further typing buys nothing.

A few patterns that reliably produce yellow, so you learn to spot them:

- **Untyped variables** — any variable Cython treats as a Python object. Add a `cdef` declaration.
- **Calling Python functions** — calling `math.sin` (the Python module) instead of `cimport`-ing `sin` from `libc.math`. Use the C library version.
- **Indexing Python containers** — `mylist[i]` or untyped `arr[i]`. Use a typed memoryview or `array.array`.
- **Integer/float operations on untyped values** — if Cython does not know the type, it must dispatch.
- **Building Python objects** — creating tuples, lists, or boxed numbers inside the loop. Hoist them out or avoid them.
- **`**` power operator** — may call `pow()`; replace `x ** 2` with `x * x` for the common small-integer cases.

The discipline is simple and it is the entire methodology of optimizing a Cython kernel: get the *inner loop* completely white. Lines outside the loop can stay yellow — they run once, who cares. It is the per-iteration yellow that kills you. The annotation is your map; the rest is just following it. This is the same "measure, do not guess" loop that [opens this series](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), applied at the level of individual source lines.

## 6. Directives: dropping Python's safety nets for raw C

Even with a fully-white loop body, Cython by default keeps two Python safety behaviors on every array access, and they cost real cycles. **Directives** let you turn them off when you are confident your indices are valid. The three that matter most for numeric kernels are `boundscheck`, `wraparound`, and `cdivision`. Figure 5 lays out what each one does, what it buys, and exactly how it can hurt you.

![matrix of Cython directives showing cdef types giving static C types and thirty times speedup with silent overflow risk, boundscheck off skipping the index check for extra speed with a segfault risk on a bad index, wraparound off disabling negative indexing, cdivision using C division rules with no zero division error, and nogil releasing the GIL to scale on cores but forbidding Python objects](/imgs/blogs/cython-typed-python-that-compiles-to-c-5.png)

**`boundscheck(False)`** removes the "is this index in range?" check that Cython inserts before every memoryview access. By default, `arr[i]` on a memoryview verifies `0 <= i < n` and raises `IndexError` if not — Python's safety contract. That check is a compare-and-branch per access; with four accesses per iteration over four million iterations, it adds up. Turn it off and `arr[i]` becomes a bare pointer dereference. The cost: if `i` *is* out of range, you now read or write arbitrary memory — a segfault if you are lucky, silent corruption if you are not. Figure 6 makes the trade-off vivid.

![before and after comparison of boundscheck on which checks zero to n on every access and raises an IndexError on a bad index at two point one nanoseconds versus boundscheck off which does a direct pointer access with no check at one point six nanoseconds but risks a segfault on a stray index](/imgs/blogs/cython-typed-python-that-compiles-to-c-6.png)

**`wraparound(False)`** disables Python's negative-indexing semantics. In Python, `arr[-1]` means the last element; to support that, Cython must check the sign of every index and adjust. If your loop only ever uses non-negative indices (almost all numeric loops do), turn it off and the sign check disappears. The risk is symmetric to boundscheck: a negative index now reads memory before the buffer instead of wrapping to the end.

**`cdivision(True)`** switches integer and float division from Python's semantics to C's. Python guarantees that `a // b` floors toward negative infinity and that dividing by zero raises `ZeroDivisionError`; enforcing those guarantees requires extra branches around every division. C division truncates toward zero and gives undefined behavior on divide-by-zero. If your kernel divides in the hot loop and you have already guaranteed non-zero divisors, `cdivision(True)` removes those guard branches — often a 2× win on division-heavy code. The risk: a zero divisor now crashes or produces garbage instead of raising a clean exception.

You apply directives three ways: as decorators on a function, as a `with` block scoped to a region, or globally in the build. The decorator form is the most common and the most local:

```cython
cimport cython
from libc.math cimport sin, cos, sqrt, atan2, M_PI
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def haversine_fast(double[::1] lat1, double[::1] lon1,
                   double[::1] lat2, double[::1] lon2):
    cdef Py_ssize_t i, n = lat1.shape[0]
    cdef double R = 6371.0
    cdef double phi1, phi2, dphi, dlam, a, c
    cdef double[::1] out = np.empty(n, dtype=np.float64)
    for i in range(n):
        phi1 = lat1[i] * M_PI / 180.0
        phi2 = lat2[i] * M_PI / 180.0
        dphi = (lat2[i] - lat1[i]) * M_PI / 180.0
        dlam = (lon2[i] - lon1[i]) * M_PI / 180.0
        a = (sin(dphi / 2.0) * sin(dphi / 2.0)
             + cos(phi1) * cos(phi2) * sin(dlam / 2.0) * sin(dlam / 2.0))
        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        out[i] = R * c
    return np.asarray(out)
```

The two decorators turn this `haversine_mv` into the `haversine_fast` row of our table: 42 ms drops to 31 ms, the extra ~1.35×. On this kernel there is no division in the loop, so `cdivision` would not help here, but on a kernel with a `a / b` inside the loop it routinely doubles that step.

The honest engineering judgment: **only turn off a safety net when you have proven you do not need it.** A clean way to get both safety and speed is to keep the checks on during development and tests, then disable them for the production build once you trust the index logic — or use the scoped `with cython.boundscheck(False):` block to drop checks *only* in the validated hot loop while keeping them everywhere else. A directive that segfaults in production at 3 a.m. is a far worse outcome than a kernel that runs 1.3× slower, so spend these where the speedup is large and the index logic is simple and well-tested. This is the same risk calculus the [native-acceleration landscape](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python) post frames for choosing how far down toward the metal to go: every step down trades a guarantee for a cycle, and you only take the step where the cycles matter.

#### Worked example: directives on a division-heavy normalizer

Consider a different kernel where directives shine — normalizing a feature matrix, dividing each row by its sum, with a guard. Pure Python over a 2,000×500 matrix: about 410 ms. A typed memoryview Cython version with bounds checking on: 5.8 ms (71×). Add `@cython.boundscheck(False)` and `@cython.wraparound(False)`: 4.3 ms (95×). Add `@cython.cdivision(True)` because we have already verified no row sum is zero: 2.6 ms (158×). The `cdivision` step alone bought a clean 1.65× *here* because division was on the hot path and Cython was emitting a zero-check branch around each `value / row_sum`. Same source, three decorators, and a careful one-line guarantee outside the loop (assert no zero sums) that lets us safely drop the in-loop guards. The lesson: directives are not a uniform speedup — they pay in proportion to how often the guarded operation runs in your specific loop. Annotate, find the guarded operation, decide if you can prove the guarantee, then drop it.

## 7. nogil and prange: true multi-core parallelism

Here is the capability that, more than any other, justifies the build step. The [Global Interpreter Lock (GIL)](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) is a single lock that CPython holds while executing Python bytecode; it serializes all Python execution so that exactly one thread runs Python code at a time. This is why threading does not speed up CPU-bound *Python*: spin up eight threads, they take turns holding the GIL, and you get the throughput of one core plus contention overhead. But the GIL only protects *Python objects and refcounts*. Code that touches **no Python objects** — pure C operating on C types and raw buffers — does not need the GIL at all. Cython lets you mark such a region `nogil`, releasing the lock so other threads (or other `nogil` regions) run in genuine parallel.

A typed Cython loop over memoryviews touches no Python objects: the indices are `Py_ssize_t`, the values are `double`, the array access is raw pointer arithmetic, the math is `libc`. It is already pure C. So we can wrap it in `with nogil:` and release the GIL for its duration, and then — the payoff — use `prange` from `cython.parallel` to split the loop iterations across threads with OpenMP, running on every core simultaneously. Figure 7 shows the difference.

![before and after comparison of a GIL held loop running on a single core with seven cores idle taking eight hundred milliseconds versus a nogil block with prange that releases the GIL and splits the work across eight cores finishing in one hundred twenty milliseconds for a six point five times speedup](/imgs/blogs/cython-typed-python-that-compiles-to-c-7.png)

Here is the parallel Haversine. The changes from `haversine_fast` are tiny: import `prange`, build the extension with the OpenMP flag, and change `range` to `prange`.

```cython
cimport cython
from cython.parallel cimport prange
from libc.math cimport sin, cos, sqrt, atan2, M_PI
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def haversine_parallel(double[::1] lat1, double[::1] lon1,
                       double[::1] lat2, double[::1] lon2,
                       int num_threads=8):
    cdef Py_ssize_t i, n = lat1.shape[0]
    cdef double R = 6371.0
    cdef double phi1, phi2, dphi, dlam, a, c
    cdef double[::1] out = np.empty(n, dtype=np.float64)
    for i in prange(n, nogil=True, num_threads=num_threads,
                    schedule="static"):
        phi1 = lat1[i] * M_PI / 180.0
        phi2 = lat2[i] * M_PI / 180.0
        dphi = (lat2[i] - lat1[i]) * M_PI / 180.0
        dlam = (lon2[i] - lon1[i]) * M_PI / 180.0
        a = (sin(dphi / 2.0) * sin(dphi / 2.0)
             + cos(phi1) * cos(phi2) * sin(dlam / 2.0) * sin(dlam / 2.0))
        c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
        out[i] = R * c
    return np.asarray(out)
```

`prange(n, nogil=True, ...)` does three things: it releases the GIL for the loop, splits the `n` iterations across `num_threads` OpenMP threads, and (here) uses a `static` schedule that gives each thread a contiguous chunk — ideal when every iteration costs the same, as ours does. Each thread's loop variables (`phi1`, `a`, etc.) are automatically *thread-private* because they are `cdef`-typed locals, so there is no data race. The accesses into `lat1[i]` and `out[i]` are disjoint across iterations, so no synchronization is needed. To compile this you must pass OpenMP to both Cython and the C compiler — add `extra_compile_args=["-fopenmp"]` and `extra_link_args=["-fopenmp"]` to the `Extension` in `setup.py`:

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "haversine",
    sources=["haversine.pyx"],
    extra_compile_args=["-fopenmp", "-O3"],
    extra_link_args=["-fopenmp"],
    include_dirs=[numpy.get_include()],
)
setup(ext_modules=cythonize([ext], annotate=True))
```

On the 8-core box, `haversine_parallel` runs in about **6.2 ms** — down from 31 ms single-threaded, a 5× scaling on 8 cores (not a full 8× because memory bandwidth and thread-startup overhead eat into it, classic [Amdahl plus bandwidth limits](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop)), and **629× over the original pure-Python loop**. This is true parallelism on the CPython interpreter we all said could not do CPU-bound threads — because inside the `nogil` region there is no Python, so there is no GIL to fight. Numba can do this too with `parallel=True` and its own `prange`; the difference is that Cython makes the GIL boundary *explicit and visible* — you see exactly where the lock is released and you control exactly what runs without it.

The constraint to internalize: **inside a `nogil` block you may not touch any Python object.** No creating lists, no calling Python functions, no boxed numbers, no `print`. If you try, Cython refuses to compile and tells you which line touches a Python object — which is helpful, because it forces the loop to be genuinely pure C, the precondition for safe parallelism. A `cdef ... nogil` helper function is callable from inside the block; a normal Python helper is not. This restriction is a feature: it is Cython statically proving your parallel region is GIL-free and therefore safe to run on many threads at once.

#### Worked example: scaling a Mandelbrot kernel across cores

To show `prange` scaling cleanly, take a kernel with *no* memory-bandwidth ceiling — a Mandelbrot set, which is pure compute (an iteration count per pixel, almost no memory traffic). Single-threaded typed Cython on a 1024×1024 grid at 256 max iterations: 95 ms. With `prange` and `nogil`: 2 threads → 49 ms (1.94×), 4 threads → 25 ms (3.8×), 8 threads → 13 ms (7.3×). Near-linear scaling, because the work is compute-bound and embarrassingly parallel — each pixel is independent and the data footprint is tiny, so cores do not contend for memory bandwidth. Contrast our Haversine, which only hit 5× on 8 cores because it streams large arrays and saturates memory bandwidth around 5 effective cores. The lesson the two examples teach together: **`prange` scaling is bounded by whichever runs out first, cores or memory bandwidth.** Compute-bound kernels scale near-linearly; memory-streaming kernels plateau when bandwidth saturates. Measure your own kernel's scaling curve — do not assume 8 cores means 8×.

## 8. Calling into C libraries: cdef extern

The other capability that sets Cython apart from Numba is direct, zero-overhead interop with existing C and C++ libraries. If there is a battle-tested C library that already does what your hot loop needs — a fast hashing routine, a SIMD-optimized math kernel, a domain library like a geometry or compression library — you can call it directly from Cython with `cdef extern`, no wrapper layer, no marshaling tax. You declare the C function's signature so Cython knows how to call it, and Cython emits a direct C call.

Suppose you want to use the C standard library's `qsort`, or a custom C function from a header you ship. You write a declaration block telling Cython the signature:

```cython
# fast_ops.pyx
cdef extern from "math.h":
    double erf(double x) nogil       # the error function from libm

cdef extern from "my_kernel.h":
    # a hand-written C function in your own header
    double dot_product(const double* a, const double* b, int n) nogil

@cython.boundscheck(False)
def gaussian_cdf(double[::1] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double[::1] out = np.empty(n, dtype=np.float64)
    cdef double inv_sqrt2 = 0.7071067811865476
    for i in range(n):
        out[i] = 0.5 * (1.0 + erf(x[i] * inv_sqrt2))   # direct libm call
    return np.asarray(out)
```

The `cdef extern from "math.h"` block tells Cython that `erf` is a C function with the given signature, declared in `math.h`. Now `erf(x[i] * inv_sqrt2)` in the loop compiles to a *direct call to the C library's `erf`* — the same call a C program would make, with no Python layer at all. Because we also marked it `nogil`, you can call it inside a `prange` block for parallel evaluation. To call your *own* C code, you put the function in a header, declare it in a `cdef extern from "my_kernel.h"` block, pass `&buffer[0]` to hand C a raw pointer to a memoryview's data, and add the `.c` file to the extension's `sources`. The pointer-passing — `dot_product(&a[0], &b[0], n)` — hands C a raw `double*` into your NumPy buffer with **zero copying**: C reads the exact same bytes NumPy holds.

This is something Numba simply cannot do well. Numba lives in its own JIT world; reaching out to an arbitrary C library from a Numba function is awkward at best. Cython treats C as a first-class neighbor — declare the signature, call it, pass pointers into your buffers, done. If your performance problem is "I need to use this specific high-performance C library inside my hot loop," Cython is not just *a* good answer, it is essentially *the* answer in the Python ecosystem (alongside the lower-level [C-extension and FFI approaches](/blog/software-development/python-performance/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11), which trade Cython's convenience for even more control). The marshaling cost across the boundary is effectively zero because there is no marshaling — it is a direct C call on raw pointers, not a serialize-send-deserialize round trip.

## 9. A 2D kernel: image filtering with a contiguous memoryview

Numeric Cython is not only 1D loops. The same memoryview machinery extends to 2D, which is where image filters, stencils, and matrix kernels live — and these almost never vectorize cleanly because each output pixel reads a *neighborhood* of input pixels. Take a 3×3 box blur over a grayscale image: each output pixel is the average of its 3×3 neighborhood. In pure Python over a 2048×2048 image that is roughly 4 million pixels each doing 9 boxed reads and an add — about 1.7 seconds on the named machine. The Cython version uses a 2D contiguous memoryview:

```cython
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def box_blur(double[:, ::1] img):
    cdef Py_ssize_t h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t r, c, dr, dc
    cdef double acc
    cdef double[:, ::1] out = np.empty((h, w), dtype=np.float64)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            acc = 0.0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    acc += img[r + dr, c + dc]
            out[r, c] = acc / 9.0
    return np.asarray(out)
```

The parameter type `double[:, ::1]` is a *2D, C-contiguous (row-major)* memoryview: the last axis is contiguous, so `img[r, c]` compiles to `base[r * row_stride + c]` — one multiply and one load, no Python. The nested loops over `dr, dc` are tiny and the C compiler unrolls them. This runs in about **24 milliseconds**, roughly **71× faster** than pure Python. The crucial detail is the loop order: we iterate `r` (rows) in the outer loop and `c` (columns) in the inner loop, which walks memory in row-major order, hitting consecutive cache lines. Swap the loops — `c` outer, `r` inner — and on a large image you stride across rows, blowing the cache on every access; that single mistake can make the same kernel 3–5× slower despite identical arithmetic. Cython gives you C-level control over memory access order, which is exactly the control you need for cache-friendly 2D kernels and exactly what a high-level NumPy expression hides from you. When your kernel's speed depends on *the order you touch memory* — and for 2D and stencil work it always does — Cython lets you write the access pattern explicitly.

## 10. Common pitfalls: where the speedup hides from you

Cython has a few traps that silently leave performance on the table, and they are worth a checklist because the symptom is always the same — "I typed everything and it is still slow." When that happens, open `cython -a` and look for these.

**The forgotten parameter type.** The single most common mistake: you type all the loop *locals* with `cdef` but leave the function *parameters* untyped, so every `arr[i]` still goes through `__getitem__`. Typing the locals does nothing if the array access is still Python. The fix is the memoryview parameter type (`double[::1] arr`), and the annotation will show those access lines as bright yellow until you add it. Always type the things you index.

**The accidental object in a `nogil` block.** If you put any operation that touches a Python object inside a `with nogil:` or `prange(nogil=True)` block — a `print`, a list append, a call to a non-`cdef` helper, even constructing an error message — Cython refuses to compile and points at the line. This is good (it is proving your parallel region is GIL-free) but surprising the first time. The discipline: pre-allocate all buffers *before* the `nogil` block, do only C-typed arithmetic inside, and convert back to Python objects *after*. Error handling inside a `nogil` loop needs care — you typically set a C flag and raise the exception after re-acquiring the GIL.

**The non-contiguous memoryview.** Declaring `double[:] arr` (no `::1`) accepts strided arrays and is slightly slower per access because it multiplies by the stride every time. If you *know* your input is contiguous (most NumPy arrays you create are), declare `double[::1]` to promise contiguity and get the faster pointer arithmetic. But beware: if a caller passes a non-contiguous view (a slice like `arr[::2]` or a transposed array), binding it to a `[::1]` memoryview raises a `ValueError` at call time. Either guarantee contiguity upstream with `np.ascontiguousarray`, or accept the small strided-access cost with `double[:]`. Picking `[::1]` and then passing a transpose is a classic confusing crash.

**The `**` power operator.** As noted, `x ** 2` on a `double` may compile to a `pow()` library call instead of `x * x`. For small integer powers, multiply explicitly. The annotation will not flag this as yellow (it is pure C), but it is slower C than it needs to be — check the generated `.c` if a power-heavy loop is slower than expected.

**Forgetting to type the return path.** If your hot loop is fully typed but you accumulate results into a Python list and return it, the per-iteration `list.append` and the boxing of each result re-introduce the Python tax you worked to remove. Accumulate into a typed memoryview (a pre-allocated NumPy array) and convert once at the end, as our kernels do with `np.asarray(out)`. The output boundary is as important as the input boundary.

**Not compiling with optimization.** Make sure the C compiler runs with `-O3` (set via `extra_compile_args=["-O3"]`). Without it, the generated C is compiled without the loop optimizations, vectorization, and inlining that deliver a good chunk of the speedup. A Cython loop compiled at `-O0` can be 2–3× slower than the same loop at `-O3` — the C compiler's optimizer is doing real work and you must turn it on.

## 11. The branchy kernel that NumPy cannot save

I promised a case where vectorization fails and only a compiled loop helps. Here it is, and it is the real reason to keep Cython in your toolbox even after you love NumPy. Suppose the Haversine pipeline has a business rule: skip any trip whose origin is inside a circular geofence (a depot we do not bill for), and clamp distances above a cap. In Python:

```python
def haversine_filtered_py(lat1, lon1, lat2, lon2, fence_lat, fence_lon,
                          fence_radius, cap):
    out = []
    for i in range(len(lat1)):
        d_origin = haversine_one(lat1[i], lon1[i], fence_lat, fence_lon)
        if d_origin < fence_radius:
            continue                      # data-dependent skip
        d = haversine_one(lat1[i], lon1[i], lat2[i], lon2[i])
        if d > cap:
            d = cap                       # data-dependent clamp
        out.append(d)
    return out
```

This has a `continue` that depends on the data (the geofence test) and produces a *variable-length* output (we drop some rows). NumPy's whole-array model cannot express "skip some elements and shrink the output" in one ufunc — you would have to compute everything, build a boolean mask, then filter, which wastes work computing distances for trips you discard and still materializes intermediate arrays. With a few million rows and a fence that drops 20% of them, that wasted computation and the extra mask passes cost real time and memory. A Cython version, by contrast, is just the C loop with the branches inside it — the `continue` and the `if d > cap` compile to native branch instructions, the skip genuinely avoids the work, and there are no intermediate arrays. On the named machine: pure Python 5.1 s, the masked-NumPy approach 240 ms (it computes everything then filters), and typed Cython with the branches inline **38 ms** — faster than masked NumPy *and* using less memory, because it never computes the dropped rows and never builds a mask array.

This is the structural point worth holding onto. **NumPy is fastest when the computation is uniform across all elements** — every element does the same arithmetic, no branches, no carried state. The moment your kernel has a data-dependent branch that should skip work, a carried dependency (element `i` needs the result of `i-1`, like a running maximum or an IIR filter), or a variable-length output, the vectorized model either cannot express it or has to compute-then-discard. Those are exactly the kernels where a compiled loop — Cython or Numba — wins decisively, because a C loop can branch, carry state, and stop early at full native speed. Knowing which camp your kernel is in is the single most useful judgment call in numeric Python optimization, and it is the fork at the top of our decision tree.

## 12. When Cython beats Numba (and when it does not)

Cython and Numba are siblings on the same rung of the [leverage ladder](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python) — both compile a hot Python loop to native code, both can release the GIL and parallelize, both routinely deliver 50–200× on numeric kernels. They are not redundant; they have different shapes. Figure 8 is the decision I actually use.

![tree decision diagram showing a hot loop that NumPy cannot vectorize branching into a need for C control or gradual typing leading to Cython with pyx and memoryview or to calling a C library with cdef extern, versus a pure numeric quick one-off leading to Numba with the njit decorator or staying in NumPy if it is still vectorizable](/imgs/blogs/cython-typed-python-that-compiles-to-c-8.png)

**Reach for Numba when** the hot path is a self-contained numeric function over NumPy arrays, you want a result in three lines with no build step, and you do not need to touch C libraries or ship a complicated package. `@njit` is genuinely the fastest path from "slow Python loop" to "fast native loop" — decorate the function and it compiles on first call. For a one-off script, a notebook, a numeric kernel that is purely arithmetic over arrays, Numba is usually the right first try, and you should reach for it before Cython.

**Reach for Cython when** any of these is true:

- **You need to call C or C++ libraries.** `cdef extern` is clean, direct, and zero-overhead. Numba cannot do this comfortably. If your kernel must use a specific native library, Cython is the answer.
- **You want to gradually type a large existing module.** Cython lets you take a 2,000-line `.py`, rename it `.pyx`, and type only the hot 5% — profiling, annotating, and typing incrementally while the rest stays as ordinary compiled Python. Numba is function-at-a-time and all-or-nothing within a function; it does not gradually accelerate a big mixed module.
- **You want explicit control and to see the generated C.** The `cython -a` annotation and the visible `.c` output mean you can reason about *exactly* what compiles to what. When you need to guarantee a specific machine-code shape or debug at the C level, Cython's transparency wins over Numba's JIT magic.
- **You are shipping a library and want a normal compiled extension** in a wheel, with no runtime JIT-compile latency on the first call. Cython's ahead-of-time `.so` is a standard binary extension; Numba compiles at runtime (cacheable, but still a runtime dependency on LLVM).
- **Your kernel mixes numeric work with string handling, object manipulation, or arbitrary Python.** Cython compiles *all* Python, not just the numeric subset Numba supports, so it handles kernels Numba's `nopython` mode rejects.

**The cost of Cython, stated honestly:** a build step (you must compile, and rebuild on every change), and *two languages in one project* (Python plus C type annotations, plus a `setup.py` and a toolchain). That is real friction. A new contributor has to understand the build; CI has to compile the extension; a typo in a `cdef` is a C compile error, not a friendly Python traceback. For a small numeric kernel with no C-library needs, Numba's zero-build simplicity often wins on total engineering cost even if the raw speed is identical. The decision is not "which is faster" — they are usually within a factor of each other — it is "which friction do you want to pay." Choose Cython when its powers (C interop, gradual typing, explicit control, shippable binaries) are worth the build step. Otherwise Numba.

| Factor | Numba | Cython |
| --- | --- | --- |
| Build step | None (JIT at runtime) | Yes (compile `.pyx` ahead of time) |
| First-call cost | Compile-on-first-call latency | None (already compiled) |
| Call C/C++ libraries | Awkward | First-class via `cdef extern` |
| Gradual typing of big module | All-or-nothing per function | Incremental, line by line |
| Languages in project | Python only | Python plus C annotations |
| Visible generated code | No (LLVM IR, opaque) | Yes (`cython -a` plus the `.c`) |
| Best for | Self-contained numeric one-off | C interop, large modules, shippable libs |
| Parallelism | `parallel=True`, `prange` | `nogil` plus `prange` (explicit) |
| Typical speedup on a tight loop | 50–200× | 50–200× |

## 13. Measuring honestly: how to trust your Cython numbers

Every number in this post came from a measurement, and there are traps specific to benchmarking compiled extensions that will lie to you if you are not careful. The discipline is the same as [benchmarking any Python](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) but with extra compiled-code caveats.

First, **warm up and repeat, then take the median.** A single timing is noise. Use `timeit` with enough repeats (`-r 7`) and enough inner loops (`-n` auto) that each measurement is well above the timer resolution, and report the median, not the mean (the mean is dragged by OS scheduling spikes). For a kernel that runs in milliseconds, one call is plenty per inner loop; for a microsecond kernel, you need thousands of inner loops to rise above timer granularity.

Second, **account for the build, not just the run.** Cython has zero runtime compile cost (unlike Numba's first-call JIT), but it has a one-time build cost — seconds to minutes depending on the module — that you pay at install/CI time, not per run. When you compare to Numba, be fair: Numba pays its compile cost on the first call of each session (mitigated by `cache=True`), Cython pays it once at build time. For a long-running service both are negligible; for a short script that runs once, Numba's first-call latency can actually matter and Cython's pre-compiled binary wins on total wall-clock-from-cold.

Third, **make the input realistic and large enough.** A kernel timed on 1,000 elements is dominated by call overhead and tells you nothing about per-element cost; a kernel timed on data that fits entirely in L2 cache will look faster than the same kernel on data that streams from RAM. I used 4 million float64 elements (32 MB per array, well past cache) precisely so the per-element numbers reflect real memory-bound behavior. State your input size and dtype with every number, or the number is meaningless.

Fourth, **watch for the compiler optimizing your benchmark away.** If your timed kernel computes a result you never use, an aggressive `-O3` C compiler can prove the work is dead and delete it, giving you an impossibly fast "result." Always consume the output — return it, sum it, store it — so the compiler cannot elide the loop. This is a real failure mode unique to compiled code; in pure Python the interpreter never does this, but a C compiler will.

Fifth, **the GC and refcount note.** A pure-Python loop allocates objects and triggers the cyclic garbage collector; a typed Cython loop allocates nothing in the hot path. So part of Cython's win is *not creating garbage*. When you compare, you are comparing "allocate millions of temporary boxed objects and occasionally pause for GC" against "operate on a flat buffer with zero allocation." That difference is real and is part of the speedup, but be aware that the pure-Python baseline's variance is partly GC pauses — disable GC during the baseline measurement (`gc.disable()`) if you want a cleaner number, then re-enable it.

## 14. Case studies: Cython in the real world

Cython is not a toy; it is load-bearing infrastructure under a large fraction of the scientific Python stack. A few real examples, with honest framing.

**scikit-learn** is built on Cython. Its core algorithms — the k-d trees and ball trees for nearest neighbors, the tree-growing inner loops of random forests and gradient boosting, the coordinate-descent solvers, the pairwise-distance kernels — are written in `.pyx` files with typed memoryviews and `nogil` `prange` loops. The pure-Python prototypes of these algorithms would be hundreds of times too slow for the datasets sklearn handles; Cython is what makes a Python-API machine-learning library competitive with C++ implementations. If you `pip install scikit-learn`, you are installing pre-compiled Cython extensions.

**pandas** has a `_libs` directory full of Cython. The groupby aggregations, the rolling-window operations, the join/merge inner loops, the time-series resampling — the operations where pandas has to iterate over millions of rows with logic too branchy or stateful to express as a single NumPy ufunc — are Cython kernels. This is exactly the "NumPy cannot vectorize this cleanly, so compile the loop" pattern from section 9, applied at the scale of a library used by millions.

**SciPy, Gensim, spaCy** (its older numeric core), and countless domain libraries (astronomy, bioinformatics, finance) reach for Cython for the same reason: they have a Python API for ergonomics but a hot numeric or string-processing inner loop that must run at C speed, and often they need to call into existing C/Fortran libraries (LAPACK, FFTW, domain-specific C code) — which Cython does cleanly via `cdef extern`. spaCy's older pipeline famously used Cython memoryviews and `nogil` to process text at C speed.

The honest counterpoint, and the reason this series spends a whole post on [Rust via PyO3](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python): the *newest* generation of high-performance Python tools — Polars, pydantic-core, ruff, the tokenizers library, uv — chose **Rust** rather than Cython for their native cores. Rust gives memory safety without a GC, fearless concurrency without the GIL inside the native code, and modern tooling. So the trajectory is: for accelerating an *existing* Python/C codebase incrementally, or for tight integration with C libraries, Cython remains the pragmatic workhorse and the default for the scientific stack; for a *new* from-scratch native core where you control everything, Rust is increasingly the choice. Both beat pure Python by the same two orders of magnitude; the difference is in the engineering ergonomics of building and maintaining the native code, not the raw speed of the inner loop.

## 15. When to reach for Cython (and when not to)

The decisive recommendations, because every technique is a cost and you should know when *not* to pay it.

**Reach for Cython when:** you have profiled and found a genuine hot numeric or tight-loop kernel that is a large fraction of runtime (Amdahl says optimizing a 5%-of-runtime function caps your win at 5% — do not bother); the loop has branches, carried state, or a variable output that defeats clean vectorization; you need to call into existing C/C++ libraries from the hot path; you want to incrementally type a large existing module rather than rewrite it; or you are shipping a library and want a standard pre-compiled binary extension with no runtime JIT latency.

**Do not reach for Cython when:** NumPy already vectorizes the kernel cleanly — a vectorized NumPy expression is usually within 2× of Cython and is one line of pure Python with no build step, so vectorize first and only compile what vectorization cannot reach. Do not reach for it when Numba would do — if the kernel is a self-contained numeric function with no C-library needs, `@njit` gives you the same speed with zero build friction; prefer Numba's simplicity unless you need Cython's specific powers. Do not reach for it for a function that is 2% of your runtime — you will spend a day and gain nothing measurable; profile first, optimize the real bottleneck. Do not reach for it when the bottleneck is I/O-bound, not CPU-bound — Cython speeds up *computation*; if your job spends its time waiting on the network or disk, compile nothing and reach for [async or threads](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python) instead. And do not turn off `boundscheck`/`wraparound`/`cdivision` reflexively — only drop a safety net where the speedup is large *and* you have proven the indices and divisors are valid, because a directive that segfaults in production is a strictly worse outcome than a kernel that is 1.3× slower but correct.

The meta-rule, the one this whole series turns on: **rewrite 1% in native, not 100%.** Cython is at its best as a scalpel — profile, find the one hot loop, turn *that* into a `.pyx`, type it, measure the win, and leave the other 99% of your codebase as ordinary readable Python. The moment you find yourself Cythonizing code that is not on the hot path, you are spending engineering effort that Amdahl's law guarantees will not pay off. Measure, target the bottleneck, compile the 1%, prove the win with a number.

## 16. Key takeaways

- **Cython is typed Python that compiles to C.** A `.pyx` transpiles to `.c`, compiles to a `.so`, and imports like any module. The speedup comes entirely from *type declarations* — untyped Cython is barely faster than the interpreter.
- **The build chain is `.pyx` → `cythonize` → `.c` → C compiler → `.so` → import.** There is a build step; that is the price of admission and the main reason to prefer Numba when you do not need Cython's other powers.
- **`cdef int i` compiles to a raw C loop counter** — no boxing, no refcount, no dispatch — which is where the ~100× on a tight loop lives.
- **Typed memoryviews (`double[::1] arr`) index like C arrays.** They turn `arr[i]` from a boxing `__getitem__` call into a single pointer dereference, removing the array-boundary tax that caps untyped Cython.
- **`cython -a` is your optimization map.** Yellow lines touch the Python C API and are slow; white lines are pure C and free. Hunt the yellow in the inner loop and type it away until the loop body is white.
- **Directives drop Python safety nets for speed.** `boundscheck(False)`, `wraparound(False)`, and `cdivision(True)` each remove a per-operation guard — and each has a precise crash risk if your indices or divisors are invalid. Only drop a net you have proven you do not need.
- **`nogil` plus `prange` gives true multi-core parallelism.** A typed loop touches no Python objects, so you can release the GIL and split iterations across every core with OpenMP — real CPU-bound parallelism on CPython.
- **`cdef extern` calls C libraries with zero overhead.** Direct C calls on raw pointers into your buffers, no marshaling — the capability Numba cannot match and a top reason to choose Cython.
- **Cython beats Numba for C interop, gradual typing of large modules, explicit control, and shippable binaries.** Numba beats Cython for a quick self-contained numeric one-off with no build step. They are siblings, not rivals; pick by which friction you want to pay.
- **Compile the 1%, not the 100%.** Profile first, target the genuine hot path, turn that one loop into a typed `.pyx`, and prove every win with a before-and-after number on a stated machine and input size.

## 17. Further reading

- **The Cython documentation** — the language reference, the typed-memoryview guide, the `cdef`/`cpdef` semantics, the directives reference, and the parallelism (`nogil`/`prange`) chapter. The single best source, and the annotation guide explains reading the `cython -a` report in detail.
- **The CPython C-API documentation** — what `cdef extern`, the buffer protocol, and the GIL release machinery actually call into. Essential when you start passing pointers and releasing the GIL.
- **"Cython: A Guide for Python Programmers"** by Kurt Smith (O'Reilly) — the canonical book; the memoryview and `nogil` chapters are worth the price alone.
- **"High Performance Python"** by Gorelick and Ozsvald — the chapter comparing Cython, Numba, and pure-Python rewrites on the same kernels, with the measurement discipline this series shares.
- **The OpenMP specification** (for the `prange` scheduling options) and the GCC/Clang `-fopenmp` docs — when you need to tune the parallel schedule.
- **Within this series**: [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) (the boxing and GIL foundations), [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) (what compiling to C escapes), [Numba: JIT-compiling Python to machine code](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code) (the no-build-step sibling), [the native-acceleration landscape](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python) (where Cython sits among all the native options), and [C extensions and the FFI: ctypes, cffi, and pybind11](/blog/software-development/python-performance/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11) (the lower-level interop approaches when you want even more control).

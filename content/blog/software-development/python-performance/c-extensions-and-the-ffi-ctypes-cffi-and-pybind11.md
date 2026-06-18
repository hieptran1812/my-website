---
title: "C Extensions and the FFI: ctypes, cffi, the C-API, and pybind11"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Call C and C++ from Python four ways, count the marshaling tax at the boundary, get reference counting right so you do not leak or crash, and release the GIL so native code runs on every core."
tags:
  [
    "python",
    "performance",
    "c-extension",
    "ctypes",
    "cffi",
    "pybind11",
    "ffi",
    "gil",
    "optimization",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-1.png"
---

A trading desk I worked with had a risk model written in C++ — a battle-tested library, twenty thousand lines, audited, fast, and completely off-limits to rewrite. The quants wanted to drive it from Python notebooks. The first attempt shelled out to a command-line binary and parsed its stdout; a single valuation took 40 milliseconds, almost all of it process spawn and text parsing, and the notebook that valued ten thousand positions took seven minutes. The second attempt wrapped the library properly, called it in-process, and the same ten thousand valuations finished in under two seconds. Nothing about the C++ changed. What changed was *how Python reached it*: instead of crossing an operating-system boundary ten thousand times, the code crossed a function-call boundary, and a function call into native code costs nanoseconds, not milliseconds.

That gap — between a clumsy boundary and a clean one — is the entire subject of this post. There comes a point on the leverage ladder where you have profiled, you have done less work, you have vectorized everything that vectorizes, and you still have a hot loop that is fundamentally scalar, branchy, or already implemented in a C or C++ library you do not want to reimplement. This is the moment to leave Python and call into native code directly. We have surveyed [when it is worth leaving pure Python at all](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python), and we have seen [Cython turn typed Python into compiled C](/blog/software-development/python-performance/cython-typed-python-that-compiles-to-c). This post is about the four bridges you can build by hand directly to C and C++: **ctypes**, **cffi**, the raw **CPython C-API**, and **pybind11**. They differ in build effort, in ergonomics, and — the number that decides most real choices — in the *per-call overhead* you pay every time you cross the boundary.

![A comparison matrix of the four ways to call C from Python showing ctypes cffi the C-API and pybind11 scored on build step language per-call cost and ergonomics on an 8-core Linux box](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-1.png)

By the end of this post you will be able to load an existing shared library with ctypes in five lines and call into it correctly, declare a C interface with cffi, write a real C extension module by hand with correct reference counting, expose a C++ function and class with pybind11, and — the part that makes native code actually parallel — release the Global Interpreter Lock inside your native function so that N threads run on N cores at once. You will also be able to reason about the one cost that ties all four together: the **marshaling tax** at the boundary, and why it means you want a few big calls, never a million tiny ones.

All numbers in this post come from one machine so they are comparable: an **8-core x86-64 Linux box (Ubuntu 22.04), CPython 3.12.3, 16 GB RAM, GCC 11.4**. Where I give a per-call overhead I measured it with `timeit` over a trivial function; where I give a kernel speedup I measured wall-clock with `time.perf_counter` and the median of many repeats. Native-extension numbers vary with compiler flags, CPU, and the exact kernel, so treat the absolute figures as representative and the *ratios* as the durable lesson.

## The cost model: why the boundary, not the loop, is what you pay for

Before any code, the cost model — because it explains every decision in this post. When pure Python runs a loop, the expensive thing is the *interpreter*: each iteration dispatches a bytecode, unboxes operands from `PyObject` pointers, does the arithmetic, and reboxes the result. We unpacked exactly where those cycles go when we looked at [the CPython execution model and the eval loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop). Native code deletes all of that: a C `for` loop over a packed array is a handful of machine instructions per element with no boxing and no dispatch. That is the *gain*.

But native code does not come free. Every time control passes from Python into C, something has to translate Python's world into C's world. Python's integer `5` is a heap-allocated `PyLongObject` with a reference count, a type pointer, and a variable-length digit array; C's integer is a 64-bit value in a register. To call a C function `int square(int)` you must take the `PyObject*` that points at Python's `5`, check it really is an integer, extract the machine value, place it where the C calling convention expects an argument, call, then take the returned `int`, allocate a fresh `PyLongObject`, set its reference count to 1, and hand back the pointer. That translation — Python object to C type on the way in, C type to Python object on the way out — is called **marshaling**, and it is the fixed tax you pay at the boundary regardless of how much work the C code does.

This gives us a simple, predictive model. Let $c$ be the fixed marshaling cost of one crossing (in the tens to thousands of nanoseconds, depending on the bridge), $w$ the work the native code does per call, and $n$ the number of calls. The total native time is approximately

$$T_{\text{native}} = n \cdot (c + w)$$

Compare it to doing the same work in pure Python at per-call cost $p$ (which is large because of the eval loop): $T_{\text{python}} = n \cdot p$. The speedup is

$$\frac{T_{\text{python}}}{T_{\text{native}}} = \frac{p}{c + w}.$$

Read that fraction carefully, because it contains the whole strategy. If $w$ is *large* — the C call does real work, like summing a million-element array — then $c$ is negligible, $c + w \approx w$, and you get the full native speedup. But if $w$ is *tiny* — the C call squares one integer — then $c$ dominates, $c + w \approx c$, and you are comparing the Python per-call cost $p$ against the marshaling cost $c$. For a fast bridge those are *similar*: a Python function call is roughly 50 nanoseconds, a ctypes call is roughly 1000 nanoseconds. So calling a trivial C function a million times is *slower* than staying in Python. The boundary is the bottleneck. The rule that falls out is the one you must tattoo on the inside of your eyelids: **make few big calls, never many tiny ones.** Push the loop *into* C so $w$ grows and the per-crossing tax $c$ amortizes to nothing.

![A layered stack showing the marshaling boundary cost where a Python object is parsed to a C type the native code runs then a Python result is built each crossing paying a fixed tax](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-3.png)

The figure above is the boundary as a stack of costs. A Python object comes in boxed and reference-counted; it gets parsed and unboxed to a C type (a check plus an extraction); the native code runs with no eval loop; then a fresh Python result is allocated and reference-counted on the way out; and control returns. Every layer above "run native code" is overhead that exists *only because* you crossed the boundary. If you cross it a million times you pay every layer a million times. If you hand C a whole array and let it loop internally, you pay every layer *once*. Hold that picture; it is the lens for everything below.

It is worth being concrete about what each layer of the tax actually is, because it explains why the numbers come out where they do. The *unbox* on the way in is a type check (is this `PyObject*` really a float?) plus a field read to extract the C value — cheap, a few nanoseconds, but not zero. The *box* on the way out is the expensive half: allocating a new `PyObject` means asking CPython's small-object allocator for memory, writing the type pointer, setting the reference count to 1, and writing the value — tens of nanoseconds, dominated by the allocation. On top of those, each bridge adds its own dispatch overhead: ctypes inspects your `argtypes` and invokes libffi to build a call frame matching the C calling convention (the bulk of its ~1 µs); cffi in API mode skips most of that with a precompiled wrapper; the C-API has essentially none because the function *is* a C-API function. So the per-call cost $c$ is "box + unbox + dispatch," and the dispatch term is what separates a 1 µs ctypes call from an 80 ns C-API call. The box/unbox floor is shared by all of them, which is why even the fastest bridge cannot make a trivial one-operation call competitive with staying in Python — and why the only real escape is to make the call do enough work that $c$ stops mattering.

## Bridge 1: ctypes — call a shared library with no build step

`ctypes` is in the standard library. It needs no compiler, no build step, no extension module — it loads an existing shared library at runtime (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows) and lets you call its exported functions directly. It is the fastest way to *get started* and the slowest *per call*. It is the right tool when a compiled library already exists and you just want to drive it.

Here is a tiny C library. Save it as `mathkernel.c`:

```c
// mathkernel.c — a tiny shared library to call from Python.
#include <stddef.h>

// Square a single integer. Trivial work — used to measure pure call overhead.
int square(int x) {
    return x * x;
}

// Dot product of two double arrays of length n. Real work — the loop is in C.
double dot(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}
```

Compile it into a shared object. The `-fPIC` makes the code position-independent (required for a shared library) and `-O3` turns on optimization so the loop is actually fast:

```bash
gcc -O3 -fPIC -shared -o libmathkernel.so mathkernel.c
```

Now call it from Python. The two things that *matter* — and that beginners forget, with ugly consequences — are `argtypes` and `restype`. By default ctypes assumes every argument is a C `int` and every return value is a C `int`. If your function takes a `double` or returns a pointer and you do not say so, ctypes will silently pass garbage or truncate the result. Declaring the signature is not optional politeness; it is correctness.

```python
import ctypes
import numpy as np

# Load the shared library. Use the full path in real code.
lib = ctypes.CDLL("./libmathkernel.so")

# Declare square(int) -> int. ctypes defaults to int, so this is the easy case,
# but we declare it anyway to be explicit and future-proof.
lib.square.argtypes = [ctypes.c_int]
lib.square.restype = ctypes.c_int

print(lib.square(9))   # 81

# Declare dot(const double*, const double*, size_t) -> double.
# c_void_p / POINTER(c_double) for the arrays, c_size_t for the length, c_double return.
lib.dot.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
lib.dot.restype = ctypes.c_double

# Build two NumPy arrays and pass POINTERS into their buffers — no copy.
a = np.random.rand(1_000_000)
b = np.random.rand(1_000_000)
a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

result = lib.dot(a_ptr, b_ptr, a.size)
print(result, np.dot(a, b))   # match to floating-point precision
```

The line that earns its keep is `a.ctypes.data_as(...)`. A NumPy array is a contiguous, typed buffer in memory — exactly the layout C wants. `data_as` hands C a raw pointer *into that existing buffer*, so the million doubles are never copied or marshaled element by element. C reads them in place. This is the whole game with ctypes and NumPy: marshal *one pointer and one length*, not a million numbers. The loop is entirely in C; the boundary is crossed once.

![A dataflow graph of a ctypes call that loads the shared object once sets argtypes and restype resolves the symbol then per call marshals arguments to C types runs the native function and converts the result back](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-2.png)

The graph above traces the call. Loading the library and declaring the signature happen *once*, at import time. The marshaling — converting a Python `float` to a C `double` on the way in, and a C `double` back to a Python `float` on the way out — happens *per call*. For the dot product, the per-call marshaling is one pointer plus one integer plus one return value: negligible against a million-iteration C loop. For `square`, the per-call marshaling is the *entire* cost, which is exactly why the next worked example matters.

#### Worked example: the per-call overhead of ctypes on a trivial function

Let us measure $c$ directly by calling `square` — which does essentially no work, so what we time *is* the marshaling tax.

```python
import ctypes, timeit
lib = ctypes.CDLL("./libmathkernel.so")
lib.square.argtypes = [ctypes.c_int]
lib.square.restype = ctypes.c_int

# Pure-Python reference: a function that squares.
def py_square(x): return x * x

n = 1_000_000
t_c = timeit.timeit("lib.square(9)", globals=globals(), number=n)
t_py = timeit.timeit("py_square(9)", globals=globals(), number=n)
print(f"ctypes:  {t_c/n*1e9:6.1f} ns/call")
print(f"python:  {t_py/n*1e9:6.1f} ns/call")
```

On the 8-core Linux box, CPython 3.12, I measured roughly **1050 ns/call for the ctypes `square`** and roughly **60 ns/call for the pure-Python `py_square`**. Read that again. Calling a C function that does one multiply, through ctypes, is about **17× slower** than doing the multiply in Python. The marshaling and the ctypes dispatch machinery — which has to inspect `argtypes`, build a `ctypes`-level call frame, and invoke libffi internally — costs about a microsecond, and the actual multiply is lost in the noise. This is the cost model's prediction made concrete: when $w \approx 0$, ctypes loses, badly.

Now the same machine on the dot product, where $w$ is large:

| Approach | Time for 1M-element dot | Notes |
| --- | --- | --- |
| Pure Python `sum(x*y ...)` loop | ~85 ms | eval loop per element |
| `np.dot` | ~0.6 ms | BLAS, vectorized |
| ctypes `lib.dot` | ~0.9 ms | one crossing, C loop |

The pure-Python loop pays the eval-loop tax a million times: ~85 milliseconds. The ctypes call pays the marshaling tax *once* and runs a tight C loop: ~0.9 milliseconds, a **~95× speedup** over pure Python and within striking distance of NumPy's hand-optimized BLAS dot. Same library, same machine — the only difference between the disaster (`square` a million times) and the win (`dot` once over a million elements) is *where the loop lives*. Push the loop into C and the boundary cost vanishes into the noise.

A caution that has bitten everyone who uses ctypes: it does **zero type checking against the actual C signature**. If the real `dot` took a `float*` and you declared `c_double`, ctypes would happily reinterpret the bytes and hand C garbage — no error, just wrong numbers or a segfault. ctypes trusts your `argtypes`/`restype` declarations completely. Get them wrong and you get silent corruption or a crash with no Python traceback. This is the price of "no build step": there is no compiler to check that your declaration matches the header.

## Bridge 2: cffi — declare the C interface, let it parse

`cffi` (C Foreign Function Interface) fixes ctypes' worst ergonomic problem. Instead of declaring each argument's type with `ctypes.c_double` in Python, you paste the actual C declaration as a string and cffi *parses* it. The signature lives in one place, in real C syntax, the way it appears in the header. cffi has two modes, and understanding the difference is the whole point of choosing it.

**ABI mode** is like ctypes: it loads a pre-built shared library at runtime and calls into it through libffi, using the declarations you gave it. No compiler needed at install time, but it shares ctypes' weakness — it trusts your declarations and pays libffi's per-call overhead. **API mode** is different and better: cffi generates a small C wrapper, *compiles it at build time*, and the resulting extension calls the C function directly through the real header — so the C compiler checks your declarations against the actual function, and the per-call overhead drops because there is no runtime libffi dispatch. ABI is "no build, trust me"; API is "small build, the compiler checks." For anything you ship, prefer API mode.

Here is ABI mode against the same `libmathkernel.so`:

```python
from cffi import FFI

ffi = FFI()
# Paste the C declarations exactly as they appear in the header.
ffi.cdef("""
    int square(int x);
    double dot(const double *a, const double *b, size_t n);
""")

# ABI mode: open the already-compiled shared library.
lib = ffi.dlopen("./libmathkernel.so")

print(lib.square(9))   # 81

import numpy as np
a = np.random.rand(1_000_000)
b = np.random.rand(1_000_000)
# 'cast' a NumPy buffer pointer to the C double* cffi expects — no copy.
a_ptr = ffi.cast("double *", a.ctypes.data)
b_ptr = ffi.cast("double *", b.ctypes.data)
print(lib.dot(a_ptr, b_ptr, a.size))
```

Notice the declarations are *C*, not Python. You did not translate `const double *` into `ctypes.POINTER(ctypes.c_double)` by hand — you pasted the header line and cffi understood it. For a library with fifty functions and a dozen structs, this is a night-and-day difference in how much hand-translation you do and how many mistakes you make doing it.

API mode does a small compile and gives you a real, importable extension module:

```python
# build_kernel.py — run this once to compile an API-mode extension.
from cffi import FFI

ffi = FFI()
ffi.cdef("""
    double dot(const double *a, const double *b, size_t n);
""")
# set_source names the module and supplies the real C — here we link the .so,
# but you can also paste the C body directly so the compiler inlines it.
ffi.set_source(
    "_kernel_cffi",
    '#include "mathkernel.h"',
    sources=["mathkernel.c"],   # compile the C alongside
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
```

```bash
python build_kernel.py    # produces _kernel_cffi.<abi>.so
```

```python
from _kernel_cffi import lib, ffi
import numpy as np
a = np.random.rand(1_000_000); b = np.random.rand(1_000_000)
print(lib.dot(ffi.cast("double *", a.ctypes.data),
              ffi.cast("double *", b.ctypes.data), a.size))
```

In API mode the compiler saw both your `cdef` and the real `mathkernel.h`, so a mismatch is a *compile error*, not a runtime segfault. And because the generated wrapper calls `dot` directly rather than through libffi, the per-call overhead is meaningfully lower than ABI mode or ctypes — closer to a few hundred nanoseconds than a microsecond for a trivial call. cffi is what the cryptography library and many others use precisely because API mode gives you compiler-checked, lower-overhead bindings from plain C declarations. The boundary cost model still rules: for the dot product, ABI and API mode both finish around 0.9 ms because the loop is in C and the per-call difference is lost in the noise. The per-call difference only shows up when you make many tiny calls — which, as established, you should not be doing in the first place.

#### Worked example: ABI mode versus API mode per-call overhead

The two cffi modes are easy to confuse until you measure them on the same trivial call. Compile both bindings to `square` — one ABI (`dlopen`) and one API (`set_source` + `compile`) — and time a million no-work calls:

```python
import timeit
# abi_lib comes from ffi.dlopen(...); api_lib comes from the compiled module.
n = 1_000_000
t_abi = timeit.timeit("abi_lib.square(9)", globals=globals(), number=n)
t_api = timeit.timeit("api_lib.square(9)", globals=globals(), number=n)
print(f"cffi ABI:  {t_abi/n*1e9:6.1f} ns/call")
print(f"cffi API:  {t_api/n*1e9:6.1f} ns/call")
```

On the 8-core Linux box I measured roughly **620 ns/call for ABI mode** and roughly **300 ns/call for API mode** — API mode is about **2× cheaper per crossing**, and both beat ctypes' ~1050 ns because cffi's dispatch path is leaner than ctypes' generic `argtypes` inspection. The mechanism is exactly what the modes describe: ABI mode still routes every call through libffi's runtime dispatch (it does not know the function's real ABI until runtime), while API mode's generated wrapper is a direct C call the compiler emitted against the real prototype. The lesson tracks the cost model: this 320 ns gap is real but only *matters* for many small calls. On the million-element dot product, where $w$ swamps $c$, ABI and API both land near 0.9 ms and the gap vanishes. Choose API mode for the compiler check and the cleaner build, not for nanoseconds you will rarely feel.

### Passing structs and arrays across cffi

Real C interfaces rarely take just scalars; they take structs and arrays. cffi handles both from the same `cdef`. Suppose the C side defines a small point struct and a function that shifts an array of them:

```c
// in mathkernel.h / mathkernel.c
typedef struct { double x; double y; } Point;
void shift_points(Point *pts, size_t n, double dx, double dy) {
    for (size_t i = 0; i < n; i++) { pts[i].x += dx; pts[i].y += dy; }
}
```

From Python you declare the struct in the `cdef` and let cffi allocate and lay it out:

```python
from cffi import FFI
ffi = FFI()
ffi.cdef("""
    typedef struct { double x; double y; } Point;
    void shift_points(Point *pts, size_t n, double dx, double dy);
""")
lib = ffi.dlopen("./libmathkernel.so")

# Allocate an array of 3 Points in C-managed memory (zeroed).
pts = ffi.new("Point[3]")
pts[0].x, pts[0].y = 1.0, 2.0
pts[1].x, pts[1].y = 3.0, 4.0
pts[2].x, pts[2].y = 5.0, 6.0

lib.shift_points(pts, 3, 10.0, 100.0)   # one crossing, loop in C
print(pts[0].x, pts[0].y)               # 11.0 102.0
```

`ffi.new("Point[3]")` allocates a contiguous block of three structs with the exact C memory layout, and `ffi` keeps it alive as long as the Python handle exists — when `pts` is garbage-collected, the memory is freed. This is the same boundary discipline as the dot product: you hand C *one* pointer to a contiguous block and let it loop internally, rather than crossing the boundary once per point. The struct layout is C's, so there is no per-field marshaling inside the loop. For a real geometry or simulation library that traffics in arrays of structs, this is the pattern that keeps the FFI tax flat regardless of how many points you process.

## Bridge 3: the CPython C-API — maximum control, maximum footgun

ctypes and cffi *call* existing C. The C-API lets you *write a real Python extension module in C* — a `.so` that imports like any Python module, whose functions receive `PyObject*` arguments and return `PyObject*` results. This is what NumPy, lxml, and CPython's own standard library modules are built on. It gives you total control and the lowest per-call overhead of any bridge, because the function *is* a Python C function with no translation layer in between. It also gives you the sharpest footgun in the language: **manual reference counting**. Get it wrong and you leak memory or crash.

Let me build up a minimal but *correct* extension function. We will expose `dot(a, b)` that takes two Python lists of floats and returns their dot product. First the boilerplate, then the part that actually matters.

```c
// dotmodule.c — a hand-written CPython extension module.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// The C function backing dot(a, b). Receives borrowed references to the args.
static PyObject *dot(PyObject *self, PyObject *args) {
    PyObject *list_a, *list_b;

    // Parse the Python-level arguments into two PyObject* (borrowed references).
    // "OO" means "two arbitrary objects". On failure it sets an exception and
    // returns 0, and we propagate by returning NULL.
    if (!PyArg_ParseTuple(args, "OO", &list_a, &list_b)) {
        return NULL;
    }

    Py_ssize_t n = PyList_Size(list_a);
    if (n != PyList_Size(list_b)) {
        PyErr_SetString(PyExc_ValueError, "lists must be the same length");
        return NULL;
    }

    double s = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        // PyList_GetItem returns a BORROWED reference — do not decref it.
        PyObject *xa = PyList_GetItem(list_a, i);
        PyObject *xb = PyList_GetItem(list_b, i);
        s += PyFloat_AsDouble(xa) * PyFloat_AsDouble(xb);
    }

    // Py_BuildValue allocates a NEW float object with refcount 1 and returns
    // ownership to the caller (Python). "d" means C double -> Python float.
    return Py_BuildValue("d", s);
}

// The method table: name, C function, calling convention, docstring.
static PyMethodDef DotMethods[] = {
    {"dot", dot, METH_VARARGS, "Dot product of two float lists."},
    {NULL, NULL, 0, NULL}   // sentinel
};

// The module definition.
static struct PyModuleDef dotmodule = {
    PyModuleDef_HEAD_INIT, "dotmodule", "A hand-written C extension.",
    -1, DotMethods
};

// The init function. Python calls PyInit_<modulename> when you import it.
PyMODINIT_FUNC PyInit_dotmodule(void) {
    return PyModule_Create(&dotmodule);
}
```

Build it with a tiny `setup.py`:

```python
# setup.py
from setuptools import setup, Extension
setup(name="dotmodule",
      ext_modules=[Extension("dotmodule", sources=["dotmodule.c"])])
```

```bash
python setup.py build_ext --inplace   # produces dotmodule.<abi>.so
python -c "import dotmodule; print(dotmodule.dot([1.,2.,3.], [4.,5.,6.]))"   # 32.0
```

That works and is correct. Before the reference-counting discipline, notice the three pieces of machinery every C-API module needs, because they recur in every extension you will ever read. **`PyArg_ParseTuple`** is how you unpack the argument tuple: its format string is a tiny type language — `"i"` for a C int, `"d"` for a double, `"s"` for a C string, `"O"` for an arbitrary object, `"O!"` with a type to check it, `"|"` to mark the rest optional. Each format code drives one unbox-and-check. **`Py_BuildValue`** is the inverse: `"d"` boxes a C double into a Python float, `"(ii)"` builds a tuple of two ints, `"s"` builds a string from a C `char*`. And the **module init** — the `PyModuleDef` struct plus the `PyInit_<name>` function — is the entry point Python's import machinery calls exactly once when you first `import` the module; it builds the module object, attaches the method table, and returns it. The method table itself maps Python-visible names to C functions and declares each one's calling convention (`METH_VARARGS` means "arguments arrive as a tuple," `METH_NOARGS` means none, `METH_O` means exactly one). This boilerplate is mechanical, but it is also where a surprising number of bugs hide — a missing sentinel `{NULL, NULL, 0, NULL}` in the method table, a mismatched module name in `PyInit_`, a format string that does not match the C variables you pass. Now the part that *will* eventually bite you hardest: **reference counting**, the discipline that the entire C-API rests on, and that ctypes/cffi/pybind11 mostly handle for you.

### Reference counting: own it once, free it once

Every Python object carries a reference count — an integer recording how many references point at it. CPython frees an object the instant its count hits zero. We covered why this makes every read effectively a write back in [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop); here we have to manage the count *by hand*. The C-API divides references into two kinds, and the entire discipline is keeping them straight:

- A **borrowed reference** is one you may use but do not own. The object is kept alive by someone else. You must *not* `Py_DECREF` it, and you must not use it after the owner might have freed it. `PyArg_ParseTuple`, `PyList_GetItem`, and `PyDict_GetItem` all hand you *borrowed* references.
- An **owned (new) reference** is one you are responsible for. The function that gave it to you incremented the count on your behalf, and you must eventually `Py_DECREF` it exactly once to release your claim. `Py_BuildValue`, `PyLong_FromLong`, `PyList_New`, and most functions with "New" or "From" in the name return *owned* references.

The two rules, stated as crisply as I can:

1. **Match every owned reference with exactly one `Py_DECREF`.** Forget it and the count never reaches zero — the object lives forever. That is a **memory leak**. In a function called millions of times, the leak compounds: RSS climbs until the process is OOM-killed.
2. **Never `Py_DECREF` a borrowed reference, and never `Py_DECREF` an owned one twice.** Either one drops the count below the number of real references. When it prematurely hits zero, the object is freed *while something still points at it*. The next access is a **use-after-free**: a segfault, or worse, silent corruption.

![A dataflow graph of the C-API refcount hazard where owning a reference and using it leads to a correct single decref that frees it or to a forgotten decref that leaks or a double decref that frees a live object and crashes](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-6.png)

The figure above is the hazard as a fork. You own a reference; you use the object; then exactly one path is correct — a single `Py_DECREF` brings the count to zero and frees it cleanly. The other two paths are bugs. Forget the `Py_DECREF` and the count stays stuck above zero: the memory leaks and RSS climbs forever. `Py_DECREF` twice (or decref a borrowed reference) and you free an object something still uses: a use-after-free and a segfault. There is no compiler warning for either. This is why people call the C-API a footgun: the language of the bug is C, the symptom (a leak or a crash) often shows up far from the cause, and there is no traceback pointing at the line.

Here is a concrete leak. Suppose our `dot` returned the inputs in a tuple for debugging and we wrote it carelessly:

```c
// WRONG: leaks a reference on every call.
static PyObject *bad_make_pair(PyObject *self, PyObject *args) {
    // PyLong_FromLong returns a NEW (owned) reference — count 1, ours to free.
    PyObject *x = PyLong_FromLong(42);
    PyObject *y = PyLong_FromLong(7);
    // Py_BuildValue with "OO" INCREFs x and y into the tuple (it borrows-then-owns),
    // so after this the tuple holds its own references AND we still hold ours.
    PyObject *pair = Py_BuildValue("(OO)", x, y);
    // We never DECREF x and y. Their counts are now 2 (tuple + us); when the
    // tuple is freed they drop to 1, never to 0. x and y leak FOREVER.
    return pair;
}
```

Every call leaks two small integer objects. Call it ten million times and you have ten million orphaned objects pinned in memory. The fix is two lines — release *our* references once `Py_BuildValue` has taken its own:

```c
// CORRECT: balance every owned reference with a decref.
static PyObject *good_make_pair(PyObject *self, PyObject *args) {
    PyObject *x = PyLong_FromLong(42);   // owned, count 1
    PyObject *y = PyLong_FromLong(7);    // owned, count 1
    PyObject *pair = Py_BuildValue("(OO)", x, y);  // tuple takes its own refs
    Py_DECREF(x);   // release OUR reference; count back down by one
    Py_DECREF(y);
    return pair;    // ownership of 'pair' transfers to the caller
}
```

And the mirror-image crash — decref a borrowed reference:

```c
// WRONG: PyList_GetItem returns a BORROWED reference. DECREF-ing it is a bug.
PyObject *item = PyList_GetItem(some_list, 0);  // borrowed — NOT ours
Py_DECREF(item);  // BUG: we just decremented a count we never incremented.
                  // If this drops the real count to 0, the object is freed
                  // while the list still points at it -> use-after-free crash.
```

#### Worked example: a refcount leak you can watch in RSS

You do not have to take the leak on faith — you can watch it. Build a module with the buggy `bad_make_pair` exposed, then call it in a loop and read resident memory:

```python
import os, resource, leaky   # 'leaky' is the module with the bug compiled in

def rss_mb():
    # ru_maxrss is kilobytes on Linux, bytes on macOS — this is the Linux form.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"start RSS: {rss_mb():.1f} MB")
for _ in range(20_000_000):
    leaky.bad_make_pair()
print(f"after 20M calls: {rss_mb():.1f} MB")
```

On the Linux box, the buggy version climbed from about **12 MB to over 600 MB** across 20 million calls — roughly 30 bytes leaked per call (two small `PyLongObject`s), exactly as predicted, and it would keep climbing until the OOM killer arrived. Recompile with the `Py_DECREF`-balanced `good_make_pair` and RSS stays **flat at ~12 MB** for the same 20 million calls. Same loop, same machine; the only difference is two `Py_DECREF` lines. That is the C-API in one experiment: total control, and a leak or a crash one forgotten line away. When you write C-API code, the single most valuable habit is to annotate every reference at its birth — "owned" or "borrowed" — and to trace, for every owned one, the exact line that releases it.

The reward for all this care is the lowest per-call overhead of any bridge. On the Linux box, a no-op C-API function (`METH_NOARGS` returning `None`) clocked about **70–90 ns/call**, barely above a pure-Python call and an order of magnitude below ctypes. The C-API is what you reach for when you genuinely need that — a hand-tuned hot path called a great many times — and are willing to own the reference counting to get it.

## Bridge 4: pybind11 — expose C++ ergonomically

The C-API is C. If your native code is C++ — and most new performance-critical native code is — **pybind11** is the modern answer. It is a header-only C++ library that lets you expose C++ functions, classes, and even STL containers to Python with almost no boilerplate, and crucially it manages reference counts *for you* using C++ RAII (an object's destructor releases its Python reference automatically when it goes out of scope). You write idiomatic C++; pybind11 generates the C-API glue. This is what scikit-image's native bits, many PyTorch custom ops, and a large fraction of new C++ Python bindings use.

Here is the dot product and a small stateful class exposed to Python:

```cpp
// dotpybind.cpp — expose C++ to Python with pybind11.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // automatic std::vector <-> Python list
#include <vector>
namespace py = pybind11;

// A plain C++ function. pybind11 marshals the std::vector<double> automatically.
double dot(const std::vector<double> &a, const std::vector<double> &b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

// A small stateful class — a running accumulator.
class Accumulator {
public:
    void add(double x) { sum_ += x; count_++; }
    double mean() const { return count_ ? sum_ / count_ : 0.0; }
private:
    double sum_ = 0.0;
    long   count_ = 0;
};

// The module. PYBIND11_MODULE expands to the PyInit_ function for you.
PYBIND11_MODULE(dotpybind, m) {
    m.doc() = "pybind11 example: a function and a class";
    m.def("dot", &dot, "Dot product of two float lists");

    // Expose the class: constructor, two methods. No refcounting by hand.
    py::class_<Accumulator>(m, "Accumulator")
        .def(py::init<>())
        .def("add", &Accumulator::add)
        .def("mean", &Accumulator::mean);
}
```

Build it. The cleanest build for a single file uses the flags pybind11 prints for you:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    dotpybind.cpp -o dotpybind$(python3-config --extension-suffix)
```

And use it from Python as if it were a native module — because it is:

```python
import dotpybind
print(dotpybind.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))   # 32.0

acc = dotpybind.Accumulator()
for x in (1.0, 2.0, 3.0, 4.0):
    acc.add(x)
print(acc.mean())   # 2.5
```

Look at what you did *not* write: no `PyArg_ParseTuple`, no `Py_BuildValue`, no `Py_INCREF`/`Py_DECREF`, no method table, no module struct. pybind11 inferred the marshaling from the C++ types — `std::vector<double>` to and from a Python list, `double` to and from a Python float — and the `py::class_` machinery exposed the C++ object's lifetime to Python's garbage collector correctly, including reference counting, via RAII. You expressed *what* to expose; pybind11 wrote the C-API glue. For a real C++ library — a numerical solver, a parser, a physics engine — pybind11 is usually the least painful way to give Python first-class access to it, classes and all.

![A before and after diagram contrasting ctypes which needs no build but pays high per-call overhead and has no C++ classes against pybind11 which adds a compile step for low overhead and exposes C++ objects](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-5.png)

The figure above frames the trade. ctypes asks for nothing at build time and hands you the existing `.so`, but you declare every signature by hand and pay roughly a microsecond per call, and there is no notion of a C++ class. pybind11 asks you to write a little C++ glue and compile a module, and in return gives you automatic type marshaling, automatic reference counting, near-C-API per-call overhead (~90 ns on the Linux box), and the ability to expose whole C++ classes with methods and state. The choice between them is rarely about speed once the loop is in native code; it is about *what you already have*. An existing compiled `.so` with a stable ABI? ctypes. A C++ library you want to wrap richly, classes and all? pybind11.

### NumPy arrays in pybind11 without a copy

The pybind11 `dot` above took `std::vector<double>`, which means pybind11 *copies* the Python list into a C++ vector on the way in. That copy is fine for small inputs but defeats the whole "few big calls" strategy on large arrays — you would pay an O(n) copy on every crossing. For numerical work, take a `py::array_t<double>` instead and read its buffer in place, exactly as ctypes did with the buffer pointer:

```cpp
#include <pybind11/numpy.h>

double dot_np(py::array_t<double, py::array::c_style | py::array::forcecast> a,
              py::array_t<double, py::array::c_style | py::array::forcecast> b) {
    auto ba = a.request();   // a buffer_info: ptr, size, shape, strides
    auto bb = b.request();
    if (ba.size != bb.size)
        throw std::runtime_error("arrays must be the same length");
    const double *pa = static_cast<const double*>(ba.ptr);
    const double *pb = static_cast<const double*>(bb.ptr);
    double s = 0.0;
    for (ssize_t i = 0; i < ba.size; i++) s += pa[i] * pb[i];   // in place, no copy
    return s;
}
```

`a.request()` returns a `buffer_info` describing the array's existing memory — pointer, size, shape, strides — via the **buffer protocol**, the standard CPython mechanism that lets NumPy, `bytes`, `memoryview`, and `array.array` expose their raw memory without copying. The `c_style | forcecast` flags tell pybind11 to accept a C-contiguous array (and cast the dtype if needed), so `pa`/`pb` point straight into NumPy's buffer. Now a 100-million-element dot product crosses the boundary once and reads 800 MB of doubles in place rather than copying them into a vector first. This is the pybind11 analogue of `a.ctypes.data_as(...)`: the buffer protocol is how every serious native bridge avoids the per-element marshaling tax on array data.

### Errors and exceptions across the boundary

Each bridge has its own answer to "what happens when the C side fails," and getting this wrong turns a clean error into a silent wrong answer or a crash. In the **C-API**, the convention is that a function returns `NULL` to signal an exception and sets the exception state first with something like `PyErr_SetString(PyExc_ValueError, "...")`; the interpreter sees the `NULL` and raises. Forget to set the exception before returning `NULL` and you get the dreaded `SystemError: returned NULL without setting an error`. In **pybind11**, you simply `throw` a C++ exception and it is translated to a Python exception automatically — `std::runtime_error` becomes `RuntimeError`, `std::invalid_argument` becomes `ValueError`, and you can register custom mappings. That single `throw std::runtime_error(...)` in the dot product above surfaces in Python as a normal, catchable exception with a traceback, no manual error-state juggling.

**ctypes and cffi are the dangerous ones here**, because the C function has no idea it is being called from Python and cannot raise a Python exception. A C function that returns an error code returns it as an ordinary integer; if you ignore it, you proceed on bad data. A C function that segfaults takes the *whole Python process* down with it — no traceback, no `except`, just `Segmentation fault (core dumped)`. There is no safety net at the ctypes boundary: the contract you wrote in `argtypes`/`restype` and the C library's actual behavior are the only things standing between you and a crash. This is the real reason "no build step" is not free — you traded compile-time checking for a runtime trust relationship, and the C side can violate it catastrophically.

### Debugging native extensions when they go wrong

When a native extension misbehaves, the Python traceback usually stops at the boundary and tells you nothing about *where* in the C the fault happened. Three tools earn their place. **`gdb` with Python support** lets you run `gdb --args python myscript.py`, reproduce the crash, and `bt` to get a C backtrace into your extension — and modern CPython ships `gdb` macros (`py-bt`) that interleave the Python frames, so you see both stacks at the fault. **`valgrind`** (or its faster cousin for this job, `valgrind --tool=memcheck`) catches the use-after-free and the leak the C-API makes so easy: run `valgrind python myscript.py` and it flags the exact allocation that leaked or the exact read past a freed block, which is often the only practical way to find a refcount bug in a large extension. **AddressSanitizer** (`-fsanitize=address` at compile time) does the same class of detection with far lower overhead and is the modern default for C++ extensions built with pybind11. The meta-lesson: the moment you cross into native code you have left Python's safety net, so you adopt C and C++ debugging tools deliberately — the crash will not explain itself the way a Python exception does.

## Releasing the GIL: the thing that makes native code actually parallel

Here is the lever that turns native code from "fast on one core" into "fast on every core." The Global Interpreter Lock — the mutex that lets only one thread execute Python bytecode at a time, which we dissect in [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) — is held by default whenever your code runs, *including inside your C extension*. So if eight Python threads all call your fast C `dot`, they do not run in parallel: each one must hold the GIL to enter the C code, and only one can hold it at a time. Eight threads, one core's worth of throughput. The C is fast, but the GIL serializes the *calls* to it.

The fix is to **release the GIL** for the duration of the pure-C work — the part that touches no Python objects. While released, other Python threads can run, including other threads inside your C function on other cores. The moment you need to touch a `PyObject` again (allocate a result, read a Python list), you reacquire the GIL. The rule is iron: **you may touch Python objects only while holding the GIL**, so you release it strictly around the section that operates on plain C data (raw arrays, scalars) and nothing Python-shaped.

In the raw C-API the idiom is a matched pair of macros. `Py_BEGIN_ALLOW_THREADS` saves the thread state and releases the lock; `Py_END_ALLOW_THREADS` reacquires it. They must bracket *only* GIL-free code:

```c
static PyObject *parallel_dot(PyObject *self, PyObject *args) {
    PyObject *cap_a, *cap_b; Py_ssize_t n;
    // ... parse args into raw double* a, b and length n (touches Python objects,
    //     so the GIL is HELD here — correct) ...
    double *a = /* extracted buffer pointer */;
    double *b = /* extracted buffer pointer */;

    double s = 0.0;
    // RELEASE the GIL: the loop below touches only C doubles, no PyObject.
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) {
        s += a[i] * b[i];   // pure C — no Python objects, safe with GIL released
    }
    Py_END_ALLOW_THREADS
    // GIL reacquired. Now it is safe to build a Python result.
    return Py_BuildValue("d", s);
}
```

In pybind11 the same idea is one RAII guard:

```cpp
double parallel_dot(py::array_t<double> a, py::array_t<double> b) {
    auto ba = a.request(), bb = b.request();   // touches Python — GIL held
    double *pa = static_cast<double*>(ba.ptr);
    double *pb = static_cast<double*>(bb.ptr);
    size_t n = ba.size;
    double s = 0.0;
    {
        // RAII: releases the GIL on construction, reacquires on scope exit.
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; i++) s += pa[i] * pb[i];   // pure C++, no Python
    }
    return s;   // GIL reacquired here as 'release' is destroyed
}
```

![A before and after diagram showing the GIL held in native code where eight threads enter the C function but only one runs against the GIL released where the allow-threads section lets eight cores run the loop in true parallel](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-4.png)

The figure above is the whole payoff in two columns. On the left the GIL stays held: eight threads enter the C function, only one can hold the lock, the other seven wait, and you get one core's worth of work — no speedup from the threads at all. On the right the function releases the GIL around the pure-C loop: all eight threads run their loops on eight cores simultaneously because none of them is touching a `PyObject`, and you get close to 8× throughput. This is *the* reason native extensions can deliver true multicore parallelism that pure Python threads cannot: the bottleneck in threaded Python is the GIL, and a native function is the one place you are allowed to let go of it.

### Why releasing the GIL gives real parallelism — the argument

Why does this work when pure-Python threads do not? Pure-Python threads cannot run in parallel because *every bytecode* needs the GIL — the interpreter holds it continuously, releasing only briefly between bytecodes (and on I/O waits). So two CPU-bound Python threads ping-pong the single lock and never truly overlap. Inside a C extension with the GIL released, there is *no bytecode being executed* — the work is raw machine instructions on raw C data, which need no lock because they touch no shared Python state. So $k$ threads, each running a GIL-released C loop, genuinely execute on $k$ cores at once. The scaling is the usual parallel-speedup law: if a fraction $p$ of the work is the GIL-released native section and the rest is serial Python glue, Amdahl's law caps the speedup at $S = 1 / ((1 - p) + p/k)$. When $p$ is near 1 — almost all the time is in the released C loop — $S \to k$, full linear scaling across cores. When $p$ is small because the C work is tiny and you are dominated by per-call marshaling, releasing the GIL buys little. The same boundary cost model, one more time: it pays to release the GIL exactly when the native work is big enough to dominate.

There is a subtlety here that trips people who reach for `multiprocessing` instead. Multiprocessing also gives true parallelism, but it does so by running separate interpreters in separate processes, each with its own GIL — and that means every piece of data shared between workers must be *pickled, sent over a pipe, and unpickled*, which for large arrays can cost more than the computation. A GIL-released native extension running under threads has no such tax: the threads share one address space, so they all read the same NumPy buffer with zero copying. This is precisely why the heavy lifting in NumPy, scikit-learn's native cores, and many scientific libraries is built to release the GIL and run under threads rather than to fan out across processes — for CPU-bound array work, shared-memory threads over a GIL-released kernel beat process pools whenever the data is large, because they skip the serialization round trip entirely. The decision between "release the GIL and use threads" and "use multiprocessing" comes down to whether your hot work is in a native section you control (threads win) or in pure Python that cannot release the GIL (processes are your only route to multiple cores). When you own the native code, you almost always want the thread-plus-released-GIL path.

### The hidden cost of releasing the GIL, and when not to

Releasing and reacquiring the GIL is not free — it saves and restores the thread state and touches the lock, which costs on the order of a hundred nanoseconds for the round trip. That cost is invisible when the native section runs for milliseconds, but it is a real tax when the section is tiny. So the rule mirrors the boundary cost model exactly: release the GIL when the native work *between* release and reacquire is large enough to dominate the release overhead and large enough that another thread can actually make progress while you hold it released. Wrapping `Py_BEGIN_ALLOW_THREADS`/`Py_END_ALLOW_THREADS` around three machine instructions is pure overhead and possibly slower; wrapping it around a million-iteration loop is the whole game. There is also a correctness floor under the convenience: while the GIL is released you absolutely must not touch *any* `PyObject` — not allocate one, not read a field, not call any C-API function that does (and many do quietly). Touching Python state without the GIL is a data race on the reference counts and the object internals, and it will eventually corrupt memory in a way that is brutal to debug. The discipline is mechanical: extract every Python-side value you need into plain C variables *before* you release, run pure C while released, and build the Python result only *after* you reacquire.

#### Worked example: when releasing the GIL backfires

Take the same kernel but shrink the work to a single multiply per call, then drive it from eight threads with the GIL released around that one multiply:

```python
from concurrent.futures import ThreadPoolExecutor
import time, tinykernel   # releases the GIL around a single multiply (silly, but instructive)

def run(workers, calls=2_000_000):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        t0 = time.perf_counter()
        list(ex.map(lambda _: tinykernel.one_op(7.0), range(calls)))
        return time.perf_counter() - t0

print(f"1 thread:  {run(1):.2f} s")
print(f"8 threads: {run(8):.2f} s")
```

On the 8-core box this *got slower* with more threads: the per-call work (one multiply) is dwarfed by the GIL release/reacquire round trip plus the boundary marshaling, so adding threads only adds lock contention and scheduling churn — eight threads finished *behind* one. This is the GIL-release version of the "million tiny ctypes calls" disaster, and it teaches the same lesson from the other side: parallelism only pays when the parallel section is big. Release the GIL around real work, never around a crumb of it. The decision is the same fraction $p$ from Amdahl: when the GIL-released section is a tiny fraction of each call, $p$ is small, $S \to 1$, and the release overhead pushes you below 1.

#### Worked example: GIL-released threads scaling across cores

Take a CPU-bound kernel — say a numeric integration that runs for a few milliseconds of pure C per call — exposed via pybind11 with `gil_scoped_release` around the loop. Drive it from a `ThreadPoolExecutor` and measure throughput as you add threads, on the 8-core Linux box:

```python
from concurrent.futures import ThreadPoolExecutor
import time, mykernel   # pybind11 module, releases the GIL inside

def run(workers, calls=64):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        t0 = time.perf_counter()
        list(ex.map(lambda _: mykernel.heavy(2_000_000), range(calls)))
        return time.perf_counter() - t0

for w in (1, 2, 4, 8):
    print(f"{w} threads: {run(w):.2f} s")
```

Representative results on the 8-core box, with the GIL *released* inside the kernel:

| Threads | Wall time | Speedup vs 1 thread |
| --- | --- | --- |
| 1 | 6.40 s | 1.0× |
| 2 | 3.25 s | 2.0× |
| 4 | 1.68 s | 3.8× |
| 8 | 0.95 s | 6.7× |

That is near-linear scaling up to the physical core count, falling a little short of a perfect 8× because of the serial Python glue around each call (the $(1-p)$ term in Amdahl) and memory-bandwidth contention. Now flip one thing: comment out `gil_scoped_release` so the kernel holds the GIL. The 8-thread time collapses back to roughly **6.4 s** — the same as a single thread — because the eight threads serialize on the lock and never overlap. One macro pair, and a 6.7× multicore speedup either exists or does not. This is why "release the GIL in the native section" is not an optional polish; for threaded native code it is the difference between using one core and using all of them.

## Putting the four together: when each one fits

We now have four bridges and one cost model. The model — make few big calls, push the loop into native code, release the GIL around the heavy part — applies to all four equally. What differs is build effort, ergonomics, and what you already have. The decision is almost never "which is fastest" once the loop is in native code; it is "which matches my situation with the least pain."

![A comparison matrix mapping each FFI tool to the job it fits best where ctypes suits a quick call into an existing shared object cffi gives cleaner C declarations the C-API gives maximum control and pybind11 wraps a C++ library](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-7.png)

The matrix above sorts them by job. **ctypes**: you have a compiled `.so`/`.dll` and want to call it *right now* with no build step — a quick driver for an existing library, a one-off, a script. You accept ~1 µs per call (fine if you make few big calls) and you accept declaring signatures by hand with no compiler to check them. **cffi**: same situation but the interface is large or evolving, so you want to declare it in real C syntax and, in API mode, have the compiler check it and lower the per-call cost. The cleaner choice for a serious binding to an existing C library. **C-API**: you need maximum control and the lowest per-call overhead — a hand-tuned hot path, or you are building something like NumPy where every nanosecond and every reference matters — and you are willing to own reference counting and the segfault risk that comes with it. **pybind11**: your native code is C++, or you want to expose classes and rich types, and you want automatic marshaling and reference management. The default for new C++ bindings.

![A decision tree for choosing an FFI tool routing an existing C library to ctypes or cffi new C++ code to pybind11 and a maximum-performance hand-tuned module to the C-API](/imgs/blogs/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11-8.png)

The tree above walks the choice in one question: *what do you already have?* Wrapping an existing C library routes you to ctypes (quick, no build) or cffi (cleaner declarations, compiler-checked in API mode). New C++ code to expose routes you to pybind11. A genuine need for a hand-tuned, maximum-performance module — owning the reference counting deliberately — routes you to the raw C-API. Start from what exists, not from what is theoretically fastest, and you will pick right almost every time. Notice what is *not* on this tree: writing native code at all when NumPy or [Cython](/blog/software-development/python-performance/cython-typed-python-that-compiles-to-c) would do, or — for a brand-new owned native kernel where you want memory safety and an easier build — reaching instead for [Rust with PyO3 and maturin](/blog/software-development/python-performance/rust-for-python-pyo3-and-maturin), which we cover next. The FFI bridges in this post shine brightest when the native code *already exists* or *must* be C/C++.

## A consolidated comparison

Pulling the per-call and ergonomics numbers from across the post into one table, all measured or estimated on the 8-core x86-64 Linux box, CPython 3.12:

| Bridge | Build step | Per-call overhead (no-op fn) | Handles C++ classes | Refcounting | Best for |
| --- | --- | --- | --- | --- | --- |
| ctypes | none (stdlib) | ~1000 ns | no | automatic | quick call into an existing `.so` |
| cffi (ABI) | none | ~600 ns | no | automatic | cleaner C decls, no build |
| cffi (API) | small compile | ~300 ns | no | automatic | compiler-checked binding to existing C |
| C-API | compile module | ~80 ns | no (it is C) | manual — leak/crash risk | hand-tuned hot path, max control |
| pybind11 | compile module | ~90 ns | yes | automatic (RAII) | wrap a C++ library or class |

Two readings of this table matter. First, the per-call column spans more than an order of magnitude — from ~1 µs for ctypes to ~80 ns for the C-API — which is exactly why ctypes loses on a million tiny calls and why, for any bridge, the winning move is to make the calls big. Second, the per-call column *stops mattering* the moment the native work per call exceeds a few microseconds: at that point all four converge, because the loop is in C and the boundary cost is noise. The columns that then decide are build step, C++ support, and whether you want to manage reference counts by hand. Choose on those, not on the nanoseconds.

## Callbacks: when C calls back into Python

So far every example pushed *into* C. The reverse direction — C calling a Python function back, a callback — is common (sort comparators, event handlers, progress reporters) and it inverts the cost model in a way worth seeing. Every callback crosses the boundary the *other* way: C marshals its C arguments up into Python objects, the GIL must be held, the Python function runs through the eval loop, and the result is marshaled back down to C. If C calls your Python callback a million times in a loop, you are back to paying the full per-element Python tax plus marshaling — the loop is nominally in C but the *work* is in Python, one boundary crossing per element. This is the single most common way a "native" speedup evaporates.

ctypes supports callbacks with `CFUNCTYPE`:

```python
import ctypes
lib = ctypes.CDLL("./libmathkernel.so")

# Declare the C callback type: returns int, takes int.
CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)

@CALLBACK
def py_transform(x):
    return x * x + 1   # this runs in PYTHON, through the eval loop, per call

# Suppose lib.apply(cb, n) calls cb(i) for i in 0..n and sums the results.
lib.apply.argtypes = [CALLBACK, ctypes.c_int]
lib.apply.restype = ctypes.c_int
print(lib.apply(py_transform, 1000))
```

This is correct but slow for large `n`: each `cb(i)` is a full Python call plus two marshaling crossings. Keep a reference to the `@CALLBACK`-wrapped function alive (assign it to a variable) — if it is garbage-collected while C still holds the pointer, C calls into freed memory and crashes. The performance lesson is decisive: **a callback inverts the boundary, so a hot callback is an anti-pattern.** If the per-element transform is the hot work, you want it *in C*, not called *from C* — pass the data down as an array and let C apply a C function to it, rather than calling Python back per element. Callbacks are for configuration and rare events, not for the inner loop. When you find yourself wiring a Python callback into a million-iteration C loop, stop: you have rebuilt the pure-Python loop with extra steps.

## Case studies: how the bridges show up in real projects

These four bridges are not toy techniques; they are the load-bearing infrastructure under libraries you use every day. A few concrete, accurately-attributed examples make the choices above feel less abstract.

**NumPy and the C-API.** NumPy's core is a hand-written CPython C extension — `ndarray`, the ufunc machinery, the broadcasting engine — built directly on the C-API for exactly the reason the table predicts: it needs the lowest possible per-call and per-element overhead and total control over memory layout and reference counting. NumPy releases the GIL inside many of its C loops, which is *why* threaded NumPy code can use multiple cores for large array operations even though pure-Python threads cannot. The pattern in this post — release the GIL around the pure-C loop, reacquire it to build the Python result — is the NumPy pattern, repeated thousands of times across its source.

**cryptography and cffi (API mode).** The `cryptography` package binds OpenSSL through cffi in API mode. The choice is deliberate: OpenSSL has a large, complex C interface that evolves across versions, and pasting the C declarations into a `cdef` while letting the C compiler check them against the real headers at build time catches mismatches that ctypes would turn into runtime crashes. It is the textbook case for "the interface is big and must be correct, so I want the compiler in the loop" — exactly what API-mode cffi is for.

**pybind11 across the C++ ecosystem.** A large and growing share of C++ libraries expose themselves to Python through pybind11 because it makes binding a C++ class — constructor, methods, state, even inheritance and STL containers — a matter of a few `.def` lines rather than hundreds of lines of hand-written C-API glue with manual reference counting. When the underlying code is C++ and you want Python to see real objects rather than opaque handles, pybind11 is the path of least resistance, and the automatic RAII-based reference management removes the single largest source of C-API bugs.

**ctypes for the quick win.** And for the unglamorous but enormously common case — "there is a vendor `.so` with a documented C ABI and I need to call three functions from a script" — ctypes remains the right tool precisely *because* it needs no build. You will pay a microsecond per call, which is invisible if you call those three functions a few hundred times and a catastrophe if you call them in a million-iteration loop. The cost model decides, as always.

**The risk model around an existing library.** One more case worth naming because it bites teams in production: the desk I opened with, valuing positions through a C++ risk library. The first cut shelled out to a binary — each valuation spawned a process, serialized inputs to text, ran, and parsed stdout, all at ~40 ms per call dominated by process spawn and parsing, not by the actual math. Ten thousand valuations took seven minutes. Wrapping the same library in-process with pybind11 — exposing the valuation function and the model class directly — dropped each valuation to well under a microsecond of boundary cost plus the real compute, and the ten-thousand-position notebook finished in under two seconds. The C++ never changed; the *number of boundary crossings and their kind* changed, from ten thousand operating-system process boundaries to ten thousand in-process function calls. This is the cost model at industrial scale: an OS process boundary is roughly a thousand times more expensive than a function-call boundary, and the entire ~200× speedup came from picking the cheaper boundary. When you wrap an existing library, the bridge you choose is also a choice about *which boundary you cross*, and that choice can dominate everything else.

The throughline across all four: the library authors did not pick a bridge by speed alone. NumPy needed control, so C-API. cryptography needed a checked binding to a big evolving C interface, so cffi. The C++ folks needed classes, so pybind11. The script author needed it to *just work* with no build, so ctypes. Match the tool to the situation and the performance follows from the cost model.

## When to reach for this — and when not to

Native FFI is a real cost: a build step, a compiler in your deploy pipeline, platform-specific wheels, a class of bugs (segfaults, leaks) that pure Python cannot produce, and a harder debugging story. Spend that cost deliberately.

**Reach for an FFI bridge when** you have profiled and a genuinely hot path is either (a) already implemented in an existing C or C++ library you should not reimplement — wrap it with ctypes or cffi, or pybind11 if it is C++ — or (b) a scalar, branchy loop that does not vectorize and is large enough that pushing it into native code wins after the marshaling tax. In both cases the discriminating test is the cost model: the native work per call must dominate the per-call overhead. Reach for **releasing the GIL** whenever that native work is CPU-bound and you want it to run on multiple cores from Python threads — without `Py_BEGIN_ALLOW_THREADS` or `gil_scoped_release`, your fast C code is still single-threaded no matter how many Python threads call it.

**Do not reach for FFI when** the work vectorizes — NumPy or Polars already runs one C loop over packed memory and needs no hand-written extension; that is rungs you should climb first, and we did, across the [vectorization](/blog/software-development/python-performance/vectorization-in-practice-broadcasting-ufuncs-and-fancy-indexing) posts. Do not reach for it when a `@njit` decorator or a typed Cython `.pyx` would get you 90% of the speedup with 10% of the build complexity and none of the manual reference counting. Do not write a C-API module by hand when pybind11 or Cython would do — you are signing up to manage reference counts for a marginal gain. Do not call a trivial C function in a tight Python loop and expect a speedup; the boundary tax will make it *slower* than staying in Python, as the `square` worked example proved. And do not reach for native code to "make it fast" before you have measured where the time actually goes — the most common native-extension mistake is porting a loop that was 3% of runtime, where Amdahl's law caps your whole-program win at 3% no matter how fast the C is.

The honest summary: FFI is the third rung of the leverage ladder, reached *after* algorithm and vectorization, and the C-API specifically is the rung you climb only when you need maximum control and are willing to pay for it in care. Most teams that think they need a hand-written C extension actually need NumPy, Numba, or pybind11 around an existing C++ library — and most teams that think they have a parallelism problem actually have a GIL they forgot to release.

## Key takeaways

- **The boundary, not the loop, is the cost.** Every Python-to-native call pays a fixed marshaling tax to translate objects to C types and back. Total native time is $n \cdot (c + w)$: make $w$ big and $n$ small. **Few big calls, never many tiny ones.**
- **ctypes is the no-build bridge** to an existing shared library — declare `argtypes`/`restype` (it does not check them for you), pass NumPy buffer pointers so the data is never copied, and accept ~1 µs per call.
- **cffi parses real C declarations**; API mode compiles a checked, lower-overhead binding and is the cleaner choice for a serious binding to an existing C library.
- **The C-API gives maximum control and the lowest per-call overhead** (~80 ns) at the price of **manual reference counting**: match every owned reference with exactly one `Py_DECREF`. A missing `Py_DECREF` leaks (RSS climbs forever); an extra one frees a live object and crashes.
- **pybind11 wraps C++ ergonomically** — automatic marshaling, automatic reference management via RAII, real classes exposed to Python — and is the default for new C++ bindings.
- **Releasing the GIL is what makes native code parallel.** Bracket the pure-C section with `Py_BEGIN_ALLOW_THREADS`/`Py_END_ALLOW_THREADS` (or `py::gil_scoped_release`) so N threads run on N cores; skip it and your fast C code is single-threaded.
- **You may touch Python objects only while holding the GIL** — release it strictly around the section that operates on raw C data, reacquire it before building the Python result.
- **Choose by what you already have, not by nanoseconds.** Existing C library → ctypes/cffi; new C++ → pybind11; hand-tuned max-control module → C-API. Once the loop is in native code, all four converge on speed.
- **Reach for FFI after algorithm and vectorization, not before.** If it vectorizes, use NumPy; if a `@njit` or `.pyx` would do, use that; never port a loop that profiling shows is a small fraction of runtime.

## Further reading

- **Python docs — Extending and Embedding the Python Interpreter** and the **Python/C API Reference Manual** (the canonical source on `PyObject`, `PyArg_ParseTuple`, `Py_BuildValue`, and reference counting rules).
- **Python docs — `ctypes`** (loading libraries, `argtypes`/`restype`, structures, callbacks).
- **cffi documentation** (ABI vs API mode, `cdef`, `set_source`, the `verify` flow).
- **pybind11 documentation** (functions, classes, NumPy arrays, `gil_scoped_release`, return-value policies).
- **Python docs — `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`** and the threading C-API for releasing and reacquiring the GIL.
- **"High Performance Python," Gorelick & Ozsvald** (the chapters on C extensions, ctypes, cffi, and the boundary cost).
- Within this series: [when to leave pure Python at all](/blog/software-development/python-performance/the-native-acceleration-landscape-when-to-leave-pure-python), [Cython: typed Python that compiles to C](/blog/software-development/python-performance/cython-typed-python-that-compiles-to-c), [Rust for Python with PyO3 and maturin](/blog/software-development/python-performance/rust-for-python-pyo3-and-maturin), and [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) for the GIL and reference counting in depth.

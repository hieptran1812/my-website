---
title: "Rust for Python: PyO3 and Maturin, the Native Extension Language of 2026"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn why Rust replaced C for new native extensions, build a hot parse kernel with PyO3 and maturin, release the GIL with rayon for real multicore, and measure the speedup with numbers you can trust."
tags:
  [
    "python",
    "performance",
    "rust",
    "pyo3",
    "maturin",
    "native-extensions",
    "concurrency",
    "optimization",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/rust-for-python-pyo3-and-maturin-1.png"
---

A few months ago I inherited a log-processing job that read about five million semi-structured lines an hour, split each one into fields, parsed a couple of numbers out of the middle, and rolled them up into a per-customer summary. The code was clean Python. It was also the single slowest thing in the pipeline: eight and a half seconds of pure CPU per batch, every batch, all day. NumPy could not help, because the work was not arithmetic over a packed array of floats — it was string slicing, integer parsing, and dictionary updates, the kind of irregular per-row logic that lives outside the array world. I tried `multiprocessing` and the pickling tax ate most of the win. I looked at a C extension and remembered the last time I shipped one: a refcount bug that leaked memory for three weeks before anyone noticed, and a use-after-free that only segfaulted in production. So I rewrote the hot kernel in Rust, wired it to Python with PyO3, built it with one `maturin develop`, and the batch dropped from 8.4 seconds to 0.46 seconds. Then I released the GIL inside the Rust code, fanned the work across eight cores with `rayon`, and it hit 0.071 seconds. Same answer, a 118-fold speedup overall, and not one line of unsafe memory management.

That story is the whole post. We are on rung three of the leverage ladder this series keeps climbing: after you have [done less work with the right algorithm](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) and [done it in bulk with NumPy](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), the next lever is to **compile the hot one percent into native code** — and in 2026, for new native code that you will own and maintain, the language to reach for is Rust. Not because it is fashionable, but because it removes the exact class of bugs that made C extensions dangerous, while matching C's speed and giving you real multicore parallelism that pure Python cannot.

![A before and after diagram contrasting a pure Python parse and aggregate loop running in 8.4 seconds against the same kernel rewritten in Rust and called once through PyO3 running in 0.46 seconds for an 18x speedup](/imgs/blogs/rust-for-python-pyo3-and-maturin-1.png)

By the end of this post you will be able to take an irregular hot loop that NumPy cannot vectorize, rewrite the kernel in Rust, expose it to Python with a `#[pyfunction]`, build it into a wheel with maturin, call it from Python, release the GIL so it runs truly parallel, and measure the speedup honestly on a stated machine. You will understand *why* the borrow checker eliminates use-after-free, double-free, and data races at compile time — the specific bugs that make hand-written C extensions a liability — and *why* dropping the GIL plus `rayon` buys real core scaling that threads in pure Python never can. We will also be honest about when Rust is overkill: a one-off numeric loop belongs in [Numba](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), and a quick wrap of an existing C library belongs in `ctypes`. Rust earns its place when the code is new, hot, concurrent, and yours to maintain.

## The machine and the ground rules

Every number in this post comes from the same reference box, stated up front so you can calibrate against your own hardware:

> **Reference machine: an 8-core x86-64 Linux box (or an Apple M2), CPython 3.12, Rust 1.78 (stable), PyO3 0.21, maturin 1.5, 16 GB RAM.** All Rust builds are release builds (`--release`, equivalently what `maturin develop --release` produces); debug builds are five to fifty times slower and never represent production.

When I quote "8.4 seconds for the Python version" or "0.46 seconds for the Rust version," those are typical wall-clock numbers for a five-million-line batch of the synthetic log format we will define, measured with `time.perf_counter` around the call, warmed up, and reported as the median of seven runs with the garbage collector disabled during timing. Your absolute numbers will differ with core count, clock speed, and Rust version, but the *ratio* — roughly one to two orders of magnitude for this kind of string and parse work, plus near-linear core scaling once the GIL is released — is stable, because it comes from the structure of the work, not the clock.

A few terms we will lean on, defined the first time. **FFI** (Foreign Function Interface) is the mechanism by which code in one language calls code compiled from another; a Python C extension or a Rust extension is an FFI boundary. **Marshaling** is converting a value from one language's representation to another's as it crosses that boundary — a Python `str` becoming a Rust `&str`, a Python `list` becoming a `Vec`. The **GIL** (Global Interpreter Lock) is the single lock CPython holds so that only one thread runs Python bytecode at a time; it protects reference counts and object internals, and it is why pure-Python threads cannot use more than one core for CPU-bound work. **Boxing** is wrapping a raw machine value (a 64-bit integer) inside a heap `PyObject` so Python can treat it uniformly. A **borrow** in Rust is a temporary reference to a value that the compiler tracks; the **borrow checker** is the part of the Rust compiler that proves, at compile time, that no reference outlives the data it points at and that no two writers touch the same data at once. We will define `#[pyfunction]`, `#[pyclass]`, `allow_threads`, and `rayon` as we reach them.

## Why C extensions are dangerous, and why Rust is not

To understand why Rust is the right native language, you have to be precise about *what goes wrong* in C extensions. It is not that C is slow — a careful C extension is as fast as anything. It is that C makes four specific mistakes easy, and all four are the kind that pass code review, pass the test suite, run fine for weeks, and then corrupt memory or leak under a load you never tested.

**Use-after-free.** You free a buffer, then read or write through a pointer that still points at it. In C this compiles silently and usually "works" because the freed memory has not been reused yet — until, under load, the allocator hands that block to something else and your two pointers now alias unrelated data. The symptom is a crash or silent corruption far from the bug.

**Double-free.** You free the same pointer twice, often because two code paths both think they own it. This corrupts the allocator's bookkeeping and crashes somewhere unrelated later.

**Data races.** Two threads touch the same memory without synchronization and at least one writes. In a Python C extension this most often shows up around reference counts: if you manipulate a `PyObject`'s refcount after releasing the GIL, two threads can `Py_INCREF` and `Py_DECREF` the same object concurrently, the increments and decrements interleave wrong, and the object is freed while still in use — a use-after-free born from a data race. This is the single most dangerous footgun in hand-written threaded C extensions.

**Refcount leaks.** The flip side: you forget a `Py_DECREF` on an error path, the object never gets freed, and the process slowly grows. The C-API's reference-counting rules — which functions return a "new reference" you must release versus a "borrowed reference" you must not — are a constant, error-prone tax. We covered them in the FFI post on [calling C and C++ from Python](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), and the honest summary is that getting them right by hand, on every path, forever, is hard.

Now the central claim: **Rust's borrow checker makes all four of these a compile error, not a runtime surprise.** This is not a style guideline or a linter you can ignore — it is the type system, and code that violates it does not compile.

Here is the mechanism, made rigorous. Rust enforces three rules at compile time. First, **ownership**: every value has exactly one owner, and when the owner goes out of scope the value is freed exactly once. You cannot free it yourself, so you cannot double-free; the compiler inserts the single free. Second, **borrowing with lifetimes**: a reference (`&T` or `&mut T`) is tracked with a *lifetime*, and the compiler proves the reference never outlives the value it points at. A use-after-free requires a reference that outlives its data; the borrow checker rejects exactly that, so it cannot compile. Third, **aliasing XOR mutability**: at any moment you may have *either* any number of shared references `&T` *or* exactly one exclusive reference `&mut T`, never both. A data race requires two threads, one of them writing, touching the same data — which requires two mutable aliases, or a shared and a mutable alias, to the same memory. That is precisely the configuration the aliasing rule forbids. So **data races are a compile error in safe Rust.** The Rust community's phrase for this guarantee is "fearless concurrency," and it is not marketing: the compiler genuinely rejects the racy program.

And it does all of this **without a garbage collector.** That is the property that makes Rust suitable as a Python extension at all. A GC would mean unpredictable pauses — stop-the-world collections that spike your p99 latency at the worst possible moment, the exact problem you were trying to escape by leaving Python. Rust frees memory deterministically, at the close of the owning scope, with zero runtime bookkeeping beyond what you would write by hand in C. You get C's predictable, pause-free deallocation and C's speed, but with the memory bugs proven absent at compile time.

![A two column matrix mapping each Rust property to a concrete benefit, with rows for memory safety, parallelism, latency, and tooling, showing the borrow checker removing use after free and double free, rayon and no GIL giving real multicore, no garbage collector giving predictable latency, and cargo giving one build](/imgs/blogs/rust-for-python-pyo3-and-maturin-3.png)

The figure above lines up the properties against the costs they remove. Read it as the thesis of the post: every column on the right is a C-extension hazard, and every cell on the left is the Rust feature that deletes it at compile time, with no garbage collector and no runtime overhead. The fourth row — tooling — matters more than people expect, so it gets its own section later, but the short version is that `cargo` builds, tests, and dependency management replace the hand-rolled `setup.py`-and-Makefile archaeology that C extensions drag along.

There is one honest caveat. Rust has an `unsafe` keyword, and inside an `unsafe` block you *can* dereference raw pointers and break the rules — this is how Rust talks to the operating system and to C libraries. But `unsafe` is opt-in, grep-able, and rare; in a PyO3 extension you typically write zero `unsafe` yourself, because PyO3 wraps the unsafe C-API calls for you behind a safe interface. The guarantee is not "Rust cannot crash"; it is "the memory-safety bugs are confined to the small, audited `unsafe` regions, and your hot kernel is in safe Rust where they cannot occur."

## PyO3: the bridge from Rust to Python

PyO3 is the Rust crate that lets Rust functions, types, and modules be called from Python and lets Rust call back into Python. It is the equivalent of pybind11 for C++, but with the borrow checker behind it. The whole interface is three macros and a handful of conversion traits.

A `#[pyfunction]` attribute turns a plain Rust function into something Python can call. A `#[pyclass]` turns a Rust struct into a Python class. A `#[pymodule]` builds the module object that `import` finds. PyO3 handles the marshaling: when Python passes a `str`, PyO3 hands your function a `&str` (a borrowed view into the Python string's buffer, no copy); when you return a `Vec<i64>`, PyO3 builds a Python `list`. The conversions are typed, so a mismatch is a compile error in Rust, not a segfault at runtime.

Let us build the actual log-parsing kernel from the intro. The input is lines of the form `customer_id,timestamp,bytes_sent,status` — a CSV-ish record where we want to sum `bytes_sent` per `customer_id`, but only for rows with `status == 200`. This is the work NumPy cannot do directly: it is string splitting and parsing, branching on a field, and accumulating into a hash map. Here is the pure-Python baseline, the thing that takes 8.4 seconds for five million rows:

```python
def aggregate_python(lines: list[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for line in lines:
        parts = line.split(",")
        if len(parts) != 4:
            continue
        customer, _ts, bytes_sent, status = parts
        if status != "200":
            continue
        totals[customer] = totals.get(customer, 0) + int(bytes_sent)
    return totals
```

Every line of that loop is paying the interpreter tax this series keeps measuring: `line.split(",")` allocates a fresh `list` of fresh `str` objects each iteration; `int(bytes_sent)` boxes a Python integer; `totals.get(...)` is a dict lookup with hashing of a boxed string; the `+` dispatches through the integer type. Five million times. None of it is wrong; all of it is overhead.

Now the Rust version. Here is the kernel, the file you would put at `src/lib.rs`:

```rust
use pyo3::prelude::*;
use std::collections::HashMap;

/// Sum bytes_sent per customer_id for status==200 rows.
/// Input is the whole batch as one big string; we split on newlines
/// inside Rust so we cross the FFI boundary exactly once.
#[pyfunction]
fn aggregate_rust(data: &str) -> HashMap<String, u64> {
    let mut totals: HashMap<String, u64> = HashMap::new();
    for line in data.lines() {
        let mut fields = line.split(',');
        let customer = match fields.next() {
            Some(c) => c,
            None => continue,
        };
        // skip the timestamp field
        if fields.next().is_none() {
            continue;
        }
        let bytes_sent = match fields.next().and_then(|s| s.parse::<u64>().ok()) {
            Some(b) => b,
            None => continue,
        };
        let status = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        if status != "200" {
            continue;
        }
        *totals.entry(customer.to_string()).or_insert(0) += bytes_sent;
    }
    totals
}

/// The module Python imports. The name here MUST match the
/// library name in Cargo.toml (the `lib.name`).
#[pymodule]
fn fastlog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_rust, m)?)?;
    Ok(())
}
```

Notice what is *not* there. No manual memory management — `totals` is owned by the function and freed automatically when it returns (after PyO3 has converted it to a Python `dict`). No reference counting — PyO3 owns the boundary. No `unsafe`. The `&str` argument is a borrowed view into Python's string with a lifetime the compiler tracks, so you physically cannot return a reference into it that outlives the call. The `.parse::<u64>()` returns a `Result` you must handle; a malformed number is `Err`, which we skip, rather than a silent wrong answer. The compiler forced you to handle every case.

One design decision is doing a lot of work here: **the function takes the entire batch as one big `&str` and splits it inside Rust.** This is the single most important performance rule for FFI extensions, and it deserves its own treatment.

## The FFI boundary still costs: few big calls, not many small ones

Crossing the language boundary is not free. Each call from Python into Rust costs something — argument marshaling, the function-call machinery, and (if you return a Python object) building it. For a `#[pyfunction]` the per-call overhead is small, on the order of 100 to 300 nanoseconds, but it is real, and it is per call.

Here is the cost model, made concrete. Suppose your batch has $n$ rows and the boundary costs $c$ per crossing. If you call Rust *once per row* — passing one line at a time — your total boundary cost is $n \cdot c$. For $n = 5{,}000{,}000$ and $c = 200$ ns, that is one full second spent purely crossing the boundary, before any actual parsing. Worse, calling per row means you cannot keep the parsed state in Rust between calls, so you marshal a string in and a result out five million times, and you have re-introduced the per-row Python tax you were trying to escape — now with FFI overhead stacked on top. The per-row Rust function would likely be *slower* than the pure-Python loop.

If instead you call Rust *once for the whole batch*, your boundary cost is a single $c$ — 200 nanoseconds, negligible — and the five million iterations happen inside one tight Rust loop with no boundary crossings at all. That is why the kernel takes the whole batch as one `&str`. The rule generalizes to every FFI binding, C or Rust:

> **Make few big calls across the FFI boundary, never many small ones.** Move the loop *inside* the native code; hand it the largest unit of work that makes sense (the whole batch, the whole array, the whole file), and get back one aggregated result.

This is the same principle that makes NumPy fast — one C loop over a whole array beats a Python loop calling a C function per element — applied to your own native code. If you remember one engineering rule from this post, make it this one, because it is the difference between an 18x speedup and an actual slowdown.

## Building it: Cargo.toml, pyproject.toml, and maturin develop

Now the part that used to be miserable for C extensions and is genuinely easy for Rust: the build. You need two small config files and one command.

The `Cargo.toml` declares the crate, names the library, and pulls in PyO3:

```toml
[package]
name = "fastlog"
version = "0.1.0"
edition = "2021"

[lib]
# This name MUST match the #[pymodule] function name in lib.rs.
name = "fastlog"
# cdylib = a C-compatible dynamic library, which is what a
# Python extension module is.
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
rayon = "1.10"  # for the parallel version, added later
```

The `pyproject.toml` tells Python's packaging tools that maturin is the build backend:

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "fastlog"
version = "0.1.0"
requires-python = ">=3.9"
```

That is the entire configuration. Now the build. Maturin is the tool that compiles the Rust crate and packages it as a Python wheel — a `.whl` file, the standard installable Python artifact. In development you use one command:

```bash
# Inside a virtualenv, from the project root:
pip install maturin
maturin develop --release
```

`maturin develop` compiles the crate in release mode, builds the extension module, and installs it directly into your active virtualenv — so that the moment it finishes, `import fastlog` works in that interpreter. The `--release` flag is not optional for benchmarking: a debug build skips the optimizer and runs five to fifty times slower, which would make your "Rust" numbers a lie. When you are ready to ship, `maturin build --release` produces a wheel you can upload to a package index or install elsewhere:

```bash
# Produce a distributable wheel in target/wheels/:
maturin build --release
# For broad Linux compatibility, build a manylinux wheel:
maturin build --release --manylinux 2_28
```

![A directed graph showing the PyO3 and maturin build flow, with the Rust crate and the PyO3 bindings both feeding into cargo build, which feeds maturin develop, which produces a native wheel installed into site-packages and then imported and called from Python](/imgs/blogs/rust-for-python-pyo3-and-maturin-2.png)

The figure traces the whole flow: your Rust crate and the PyO3 bindings are the two inputs that `cargo build` compiles together; `maturin develop` links the result into a wheel and drops the native `.so` into `site-packages`; then it is just `import fastlog` from Python. Two inputs converge into one compile step, one build artifact, one import. Compare that to a C extension's `setup.py` with hand-specified include paths, compiler flags, and a separate step to handle every target platform, and you understand why the tooling row in the matrix above is not a throwaway. The build story is the thing that historically stopped people from writing native extensions, and Rust plus maturin removes it.

Calling the kernel from Python is now completely ordinary:

```python
import fastlog

with open("batch.log", "r") as f:
    data = f.read()          # one big string, the whole batch

totals = fastlog.aggregate_rust(data)   # one FFI call
print(totals["cust_00042"])             # a normal Python dict
```

`fastlog.aggregate_rust` looks and behaves exactly like a Python function. It takes a `str`, returns a `dict`, raises Python exceptions on error. The only difference is that its body runs as compiled native code at the speed of Rust.

#### Worked example: the single-threaded Rust speedup

Let us put real numbers on the rewrite, on the reference machine, for a five-million-line synthetic batch (about 180 MB of text), CPython 3.12, PyO3 0.21, release build. I time only the aggregation, not the file read, with `time.perf_counter` around the call, GC disabled, median of seven runs:

| Version | Wall time | Throughput | Speedup vs Python |
| --- | --- | --- | --- |
| `aggregate_python` (pure Python loop) | 8.40 s | 0.60 M rows/s | 1.0x |
| `aggregate_python` with `str.split` hoisting tricks | 6.10 s | 0.82 M rows/s | 1.4x |
| `aggregate_rust` (single thread, via PyO3) | 0.46 s | 10.9 M rows/s | 18.3x |

The middle row matters for honesty: you can buy a 1.4x with pure-Python micro-optimization (precomputing, avoiding `.get`, using `defaultdict`), and you should try that first because it is free. But it caps out fast — you are still running the interpreter loop five million times. The Rust version deletes the interpreter from the inner loop entirely: no boxing, no per-iteration allocation of `list` and `str` objects, no dict-lookup-of-boxed-string, just a native `HashMap` keyed by string slices and a compiled parse. That structural difference is the 18x, and it is stable across runs because it comes from the work, not from luck.

Where does the 18x come from, mechanically? In Python, each row costs roughly: one `list` allocation plus four `str` allocations for the split (call it 400 ns), one `int()` parse with a boxed result (about 60 ns), one dict `get` and one dict set with string hashing (about 200 ns combined), plus the eval-loop overhead of about a dozen bytecodes (roughly 600 ns). That is around 1.3 microseconds per row, times five million, which lands near 6.5 seconds of pure work plus GC pressure from the constant allocation — consistent with the measured 8.4. In Rust, each row is a few machine instructions over already-resident bytes: split is iterator advancement with no allocation, the parse is a handful of instructions, the `HashMap` entry is one hash and one slot write. That is tens of nanoseconds per row, and five million of them is well under a second. The gap is not magic; it is the boxing-and-eval-loop tax, deleted.

## Marshaling: how Python types become Rust types and back

The boundary does one job: convert values. Understanding exactly what PyO3 converts, and at what cost, is what lets you design a fast interface rather than an accidentally slow one, so it is worth being precise about the type mapping.

PyO3 defines two traits that do all the work. `FromPyObject` converts a Python object *into* a Rust value (used for function arguments); `IntoPy` and `IntoPyObject` convert a Rust value *into* a Python object (used for return values). The standard mappings are what you would hope:

| Python type | Rust argument type | Cost | Notes |
| --- | --- | --- | --- |
| `str` | `&str` | Zero-copy borrow | Borrows Python's UTF-8 buffer; no allocation |
| `str` | `String` | One copy | Owns a fresh copy; use only if you must keep it |
| `int` | `i64` / `u64` / `i32` | Unbox | Reads the C-level integer out of the `PyLong` |
| `float` | `f64` | Unbox | Reads the C double |
| `bytes` | `&[u8]` | Zero-copy borrow | Borrows the buffer directly |
| `list[int]` | `Vec<i64>` | One pass + alloc | Iterates and unboxes each element |
| `dict[str, int]` | `HashMap<String, i64>` | One pass + alloc | Iterates and converts each pair |

The crucial line is the first one. Taking `&str` rather than `String` means PyO3 hands your function a *borrowed view* into Python's existing string buffer — no allocation, no copy, just a pointer and a length, with a lifetime the borrow checker ties to the call so you cannot keep it past the function's return. For our 180 MB batch, taking `&str` means the giant string is *never copied* across the boundary; Rust reads it in place. Taking `String` instead would copy all 180 MB on every call, which on a hot path is a self-inflicted wound. **Default to borrowing (`&str`, `&[u8]`) for inputs you only read; reach for owned types only when you must store the value past the call.** This single choice is the difference between a zero-copy boundary and a 180 MB memcpy per call.

The list and dict conversions are where the per-element cost reappears, and they are why we return a `HashMap` from one big call rather than streaming results out one at a time. Converting a `Vec<i64>` of a million elements into a Python `list` means building a million `PyObject` integers — that is a million boxings, the very tax we left Python to avoid. So a fast interface keeps the *bulk* data in compact native form as long as possible and only converts the small aggregated result. Our kernel returns a per-customer `dict` with maybe a few thousand keys, not a five-million-element list, precisely so the return marshaling is cheap. If you find yourself returning a huge list from Rust, ask whether you can return a NumPy array instead (which shares one buffer, no per-element boxing) or aggregate further before returning.

## Errors: turning a Rust Result into a Python exception

A native extension that aborts the process on bad input is worse than the Python it replaced. PyO3's error story is one of the things that makes Rust pleasant here: a Rust function that can fail returns `PyResult<T>`, which is `Result<T, PyErr>`, and PyO3 turns an `Err(PyErr)` into a *raised Python exception* automatically. You never crash the interpreter; you raise the exception the caller expects.

Here is the kernel hardened to raise a real Python exception on a malformed line instead of silently skipping it — useful when you want strictness:

```rust
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;

#[pyfunction]
fn aggregate_strict(data: &str) -> PyResult<HashMap<String, u64>> {
    let mut totals: HashMap<String, u64> = HashMap::new();
    for (lineno, line) in data.lines().enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 4 {
            return Err(PyValueError::new_err(
                format!("line {}: expected 4 fields, got {}", lineno + 1, parts.len())
            ));
        }
        let bytes_sent: u64 = parts[2].parse().map_err(|_| {
            PyValueError::new_err(format!("line {}: bad integer '{}'", lineno + 1, parts[2]))
        })?;
        if parts[3] != "200" {
            continue;
        }
        *totals.entry(parts[0].to_string()).or_insert(0) += bytes_sent;
    }
    Ok(totals)
}
```

From Python, this behaves exactly like a function that raises `ValueError`:

```python
import fastlog

try:
    fastlog.aggregate_strict("cust_1,t,notanumber,200")
except ValueError as e:
    print("rejected:", e)   # rejected: line 1: bad integer 'notanumber'
```

The `?` operator is doing the heavy lifting: `parts[2].parse()` returns a `Result`, `.map_err(...)` converts a parse failure into a `PyValueError`, and `?` returns it early as an `Err`, which PyO3 raises as a Python `ValueError`. The compiler *forced* you to handle the failure — a `u64::parse` returns a `Result` you cannot ignore — so there is no path where a malformed integer silently becomes a wrong answer. This is the safety story extended from memory to logic: the type system makes the error path explicit, and PyO3 maps it to the Python exception the caller already knows how to catch. PyO3 ships the standard exception types (`PyValueError`, `PyKeyError`, `PyRuntimeError`, `PyTypeError`, and the rest), and you can define custom ones; the boundary is fully expressive, not a lossy `abort()`.

## Zero-copy NumPy: sharing one buffer instead of converting

There is one more boundary trick worth knowing, because it composes Rust with the array world from earlier in the series. When your data *is* numeric and already in a NumPy array, you do not want to convert it element by element — you want Rust to operate on the array's buffer directly, zero-copy. The `numpy` crate (the PyO3 companion `rust-numpy`) gives you exactly that: a `PyReadonlyArray1<f64>` argument borrows the NumPy array's contiguous buffer, and `.as_slice()` hands you a Rust `&[f64]` over the *same memory* the array uses — no copy, no boxing.

```rust
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

#[pyfunction]
fn weighted_sum(py: Python<'_>, xs: PyReadonlyArray1<f64>, ws: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let xs = xs.as_slice()?;   // &[f64] over NumPy's buffer, zero-copy
    let ws = ws.as_slice()?;
    if xs.len() != ws.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }
    // Release the GIL and let the native dot product run free.
    let total = py.allow_threads(|| {
        xs.iter().zip(ws).map(|(x, w)| x * w).sum::<f64>()
    });
    Ok(total)
}
```

From Python you pass arrays and get back a float:

```python
import numpy as np, fastlog
xs = np.random.rand(10_000_000)
ws = np.random.rand(10_000_000)
print(fastlog.weighted_sum(xs, ws))   # zero-copy, GIL released
```

This is the same zero-copy idea as the buffer protocol from [the NumPy first-principles post](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast): Rust and NumPy share the one contiguous buffer, so there is no per-element marshaling at all — the boundary cost is two pointers and two lengths, constant, regardless of array size. For pure elementwise float work like this, honestly, NumPy's own `(xs * ws).sum()` is already a C loop and will be about as fast; the win for Rust appears when the per-element logic is irregular (branches, lookups, string work) such that NumPy cannot express it as one ufunc, but the *data* still arrives as an array. Then you get array-speed input with custom native logic on top — the best of both worlds — and the borrow checker still guarantees you do not read past the buffer.

## The Rust extension stack, layer by layer

It helps to hold a clear picture of where the speed comes from, because it tells you exactly where the boundary cost lives and where it does not.

![A layered stack diagram showing the Python caller at the top making one call with a big batch, then the PyO3 bindings converting arguments once, then safe Rust code with no boxing and borrow checking, then py.allow_threads releasing the GIL, then OS threads with rayon keeping all cores busy at the bottom](/imgs/blogs/rust-for-python-pyo3-and-maturin-4.png)

Read the stack top to bottom. At the top, the **Python caller** makes one call with a big batch — this is the only place the interpreter is involved, and you cross it once. The **PyO3 bindings** layer is the boundary: it converts the Python `str` to a Rust `&str` once, and on the way back converts the Rust `HashMap` to a Python `dict` once. That conversion is the FFI cost, and because you do it once per batch rather than once per row, it is negligible. Below that is **safe Rust code**: no boxing (the integers are raw `u64`, not heap `PyObject`s), no eval loop (it is compiled machine code), and borrow-checked so the memory bugs cannot occur. The bottom two layers — `py.allow_threads` releasing the GIL and OS threads running `rayon` — are the parallelism story, which is the next section.

The key insight the stack makes visible: **the interpreter and its taxes live only in the top layer, and you visit it exactly once per batch.** Everything expensive about Python — boxing, the eval loop, the GIL, refcounting — is above the PyO3 line. Below it, you are in a world that looks like C but is proven memory-safe. That is the structural reason a Rust extension is fast, and it is the same structure as any good native extension; Rust just makes the safe version the easy version.

## Releasing the GIL: real parallelism with allow_threads and rayon

Here is where Rust does something C extensions *can* do but rarely do safely, and that pure Python fundamentally cannot do at all: use every core.

Recall the GIL. CPython holds one lock so only one thread executes Python bytecode at a time. This is why, as the [GIL and the eval loop post](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) explains, pure-Python threads give you zero speedup on CPU-bound work — they take turns holding the lock and run one at a time. But the GIL only protects *Python* objects and the interpreter's own state. While your code is inside a native extension, not touching any Python object, holding the GIL accomplishes nothing — it just needlessly blocks every other Python thread.

PyO3 exposes exactly the right tool: `py.allow_threads(|| { ... })`. You wrap the pure-Rust portion of your kernel — the part that touches no Python objects — in `allow_threads`, and PyO3 releases the GIL for the duration of that closure and re-acquires it before returning. During that window, your Rust code runs at full native speed *and* other Python threads can run concurrently, and — the part that matters here — your Rust code can spin up its own OS threads and use every core, because it is no longer constrained by the GIL.

The crate that makes multicore Rust trivial is `rayon`. Rayon gives you data-parallel iterators: change `.iter()` to `.par_iter()` and rayon splits the work across a thread pool sized to your cores, with a work-stealing scheduler, and — because of the borrow checker — *guarantees there are no data races at compile time*. You do not write locks. You do not reason about thread safety by hand. If your parallel code has a race, it does not compile.

Here is the parallel version of the kernel. The strategy: split the batch into chunks, parse each chunk's partial `HashMap` in parallel with rayon, then merge the partials. The merge is the only cross-thread step, and rayon's `reduce` handles it.

```rust
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

fn aggregate_chunk(chunk: &str) -> HashMap<String, u64> {
    let mut totals: HashMap<String, u64> = HashMap::new();
    for line in chunk.lines() {
        let mut fields = line.split(',');
        let (customer, _ts, bytes_sent, status) = match (
            fields.next(),
            fields.next(),
            fields.next().and_then(|s| s.parse::<u64>().ok()),
            fields.next(),
        ) {
            (Some(c), Some(_), Some(b), Some(s)) => (c, (), b, s),
            _ => continue,
        };
        if status != "200" {
            continue;
        }
        *totals.entry(customer.to_string()).or_insert(0) += bytes_sent;
    }
    totals
}

#[pyfunction]
fn aggregate_rust_parallel(py: Python<'_>, data: &str) -> HashMap<String, u64> {
    // Release the GIL for the whole native computation, then run
    // the parse+aggregate across all cores with rayon.
    py.allow_threads(|| {
        // Split into ~one chunk per core, on line boundaries.
        let chunks: Vec<&str> = split_on_lines(data, rayon::current_num_threads());
        chunks
            .par_iter()
            .map(|chunk| aggregate_chunk(chunk))
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            })
    })
}
```

Two things to see here. First, the `py.allow_threads(|| ...)` wrapper: everything inside the closure runs with the GIL released, so the rayon threads are genuinely parallel and other Python threads in the process keep running. Second, `par_iter()` is the *only* change from sequential to parallel — rayon handles the thread pool, the scheduling, and the work-stealing. The borrow checker has already verified that `aggregate_chunk` only borrows its chunk immutably and returns an owned `HashMap`, so there is no shared mutable state, so there is no possible data race. You did not write a single lock, and the compiler proved the parallelism safe.

![A before and after diagram contrasting the GIL held case where one Python thread runs the extension serially on one of eight cores in 0.46 seconds against the py.allow_threads plus rayon case where the GIL is released and rayon par_iter spreads work across all eight cores in 0.071 seconds for a 6.5x speedup](/imgs/blogs/rust-for-python-pyo3-and-maturin-5.png)

The figure contrasts the two regimes. On the left, with the GIL held, even though the extension is native, it runs on one core — the other seven sit idle. On the right, `py.allow_threads` releases the lock and rayon fans the work across all eight cores. The single-threaded Rust was already 18x over Python; the parallel version multiplies that by core scaling.

#### Worked example: GIL-released rayon core scaling

Same five-million-line batch, reference 8-core machine, release build, median of seven runs. I vary `RAYON_NUM_THREADS` to control how many cores rayon uses:

| Cores | Wall time | Speedup vs 1-core Rust | Speedup vs Python | Parallel efficiency |
| --- | --- | --- | --- | --- |
| 1 (sequential, GIL held) | 0.460 s | 1.0x | 18.3x | — |
| 2 | 0.245 s | 1.88x | 34.3x | 94% |
| 4 | 0.132 s | 3.48x | 63.6x | 87% |
| 8 | 0.071 s | 6.48x | 118x | 81% |

A few things are worth reading off this table honestly. The scaling is real but *sub-linear*: 8 cores give 6.48x, not 8x, an efficiency of about 81%. Why? Two reasons, both predictable from Amdahl's law, $S = 1/((1-p) + p/s)$, where $p$ is the parallel fraction and $s$ the number of cores. First, the final `reduce` that merges the per-chunk `HashMap`s is partly serial — combining eight partial maps is work that does not parallelize. Second, the parse is memory-bound: all eight cores are reading the 180 MB batch and writing hash maps, and they contend for memory bandwidth and the last-level cache. As you add cores, bandwidth, not compute, becomes the limit. If $p \approx 0.93$ here, Amdahl predicts $S = 1/(0.07 + 0.93/8) \approx 5.5$ to $6.5$ — right in line with the measured 6.48x. This is the honest shape of real parallel speedup, and it is why you measure rather than assume linear scaling.

The headline, though, is the bottom-right cell: **118x over pure Python**, from a rewrite that is maybe 40 lines of Rust, with the memory safety proven at compile time. Eighteen of that is leaving the interpreter; the remaining 6.5 is using the cores the GIL was hiding from you. No pure-Python approach — not threads, not even `multiprocessing` once you account for the pickling tax of shipping 180 MB of strings to workers — gets close to this on this workload.

## Why the parallel version cannot have a data race

It is worth slowing down on the claim that the parallel kernel is *guaranteed* race-free, because it is the property that makes parallel native code in Rust categorically safer than in C, and it is easy to take on faith without seeing the mechanism.

In a C extension, the obvious way to parallelize the aggregation would be to share one `HashMap` across threads and have each thread insert into it. That is a data race: two threads writing the same hash table without a lock corrupt it, and the corruption surfaces as a crash or wrong answer far from the bug, intermittently, usually only under production load. The C compiler says nothing. You would have to add a mutex around every insert by hand — and then the lock contention would erase much of your parallel speedup, and forgetting the lock on one path would reintroduce the race silently.

Rust's approach in the kernel above is structurally different: each rayon task builds its *own* `HashMap` from its *own* chunk, and the maps are merged only at the end by `reduce`. There is no shared mutable state in the parallel region at all, so there is nothing to race on. But suppose you tried the C-style shared-map approach in Rust anyway — shared one mutable `HashMap` and had every task write to it. That code **does not compile.** The borrow checker sees that you are trying to hand a `&mut HashMap` to multiple threads simultaneously, which violates the aliasing-XOR-mutability rule (you may have many readers *or* one writer, never many writers), and it rejects the program with a borrow-check error before a single instruction runs. To share mutable state across threads in Rust you are *forced* to wrap it in a synchronization type — a `Mutex` or an atomic — and the type system will not let you forget, because the bare `&mut` simply will not cross the thread boundary.

This is the concrete content of "fearless concurrency." The compiler does not make your parallel code correct in the sense of computing the right answer — that is still your job — but it makes a specific, catastrophic, hard-to-debug class of bug *impossible to express* in safe code. You cannot ship the data race, because it does not compile. In fifteen years of writing and reviewing threaded C, the data race around shared mutable state was the bug I most feared, because it passes review and the test suite and then corrupts memory at 3 a.m. under a load you never tested. Rust deletes it at compile time. For a parallel native extension that a team will maintain, that guarantee is worth more than the raw speed.

There is a corresponding rule you must still follow on the PyO3 side: **inside `allow_threads`, touch no Python objects.** The GIL is released, so reading or mutating a `PyObject` there would be the refcount data race we described at the top of the post. PyO3 helps you here too — the closure you pass to `allow_threads` cannot capture a GIL-bound token, so the type system pushes you toward doing only pure-Rust work inside it — but the discipline is yours to keep: convert Python values to Rust values *before* `allow_threads`, do the heavy native work *inside* it, and convert back to Python *after*. Our kernel does exactly that, which is why it is both fast and correct.

## A `#[pyclass]`: stateful Rust objects in Python

So far we have exposed a function. Sometimes you want a *stateful* object — a parser you configure once and feed many batches, or an aggregator that accumulates across calls. PyO3's `#[pyclass]` turns a Rust struct into a Python class, with `#[pymethods]` for its methods.

Here is an incremental aggregator: you create it once, feed it batches with `update`, and read the result with `result`. The state — the running `HashMap` — lives in Rust between calls, so you pay the FFI boundary once per batch, not once per row, and you never re-marshal the accumulated state.

```rust
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct Aggregator {
    totals: HashMap<String, u64>,
}

#[pymethods]
impl Aggregator {
    #[new]
    fn new() -> Self {
        Aggregator { totals: HashMap::new() }
    }

    /// Feed one batch; state accumulates in Rust across calls.
    fn update(&mut self, py: Python<'_>, data: &str) {
        py.allow_threads(|| {
            for line in data.lines() {
                let mut fields = line.split(',');
                let customer = match fields.next() { Some(c) => c, None => continue };
                let _ts = fields.next();
                let bytes_sent = match fields.next().and_then(|s| s.parse::<u64>().ok()) {
                    Some(b) => b, None => continue,
                };
                if fields.next() != Some("200") { continue; }
                *self.totals.entry(customer.to_string()).or_insert(0) += bytes_sent;
            }
        });
    }

    /// Return the accumulated totals as a Python dict.
    fn result(&self) -> HashMap<String, u64> {
        self.totals.clone()
    }
}
```

Add `m.add_class::<Aggregator>()?;` to the `#[pymodule]` and it is a normal Python class:

```python
import fastlog

agg = fastlog.Aggregator()
for batch in stream_of_batches():     # each batch is one big string
    agg.update(batch)                 # state stays in Rust, GIL released
totals = agg.result()                 # one final marshal to a Python dict
```

The pattern is worth internalizing because it is how the real Rust-backed libraries are built. Polars' `DataFrame`, pydantic-core's validators, the HF `Tokenizer` — these are all `#[pyclass]` objects whose heavy state lives in Rust, exposing a Python API that crosses the boundary coarsely. You hold the object in Python, but the data and the work are native.

## Rust extension vs hand-written C: same speed, very different safety

It is worth being precise about the comparison with a C extension, because both reach native speed and the speed is not the differentiator — the safety is.

![A before and after diagram comparing a hand written C extension that is fast at runtime but carries manual Py_INCREF refcount footguns and use after free segfault risk against a Rust plus PyO3 extension that is the same speed but has refcounts handled by PyO3 and is borrow checked safe at compile time](/imgs/blogs/rust-for-python-pyo3-and-maturin-7.png)

Both columns are fast — a careful C extension and a Rust extension produce comparable machine code for the same kernel, often within a few percent of each other, because both compile to native instructions with no interpreter in the loop. The difference is everything to the *left* of "fast." The C extension requires you to manage reference counts by hand (`Py_INCREF` / `Py_DECREF` on every path, including error paths), manage memory by hand (`malloc` / `free`, with use-after-free and double-free a `free` typo away), and reason about thread safety by hand (releasing the GIL with `Py_BEGIN_ALLOW_THREADS` is correct only if you touch no Python object inside, which the compiler does not check). Every one of those is a class of bug that ships, runs for weeks, and then corrupts memory under load.

The Rust extension hands all of that to the compiler. PyO3 manages the reference counts. The borrow checker rejects use-after-free, double-free, and data races before the code runs. `py.allow_threads` takes a closure, and the type system ensures you are not smuggling a Python object across the boundary in a way that would be unsafe. You write the kernel; the compiler writes the safety proof. For new code that a team will maintain for years, that is not a marginal preference — it is the difference between a 3 a.m. page and a clean deploy.

Here is the trade-off table, including the older C/C++ FFI options from the [previous post on ctypes, cffi, and pybind11](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast):

| Approach | Speed | Memory safety | Build story | Best for |
| --- | --- | --- | --- | --- |
| `ctypes` | Native (of the wrapped lib) | Unsafe (raw pointers) | None (no build) | Quick wrap of an existing C `.so` |
| `cffi` | Native (of the wrapped lib) | Unsafe | Light (a build step) | Cleaner wrapping of a C ABI |
| Raw C-API | Native | Unsafe, manual refcounts | Heavy (`setup.py`, flags) | Legacy, full control |
| pybind11 (C++) | Native | Unsafe (C++ rules) | Medium (CMake-ish) | Wrapping or writing C++ |
| **Rust + PyO3** | **Native** | **Safe at compile time** | **Easy (maturin)** | **New, owned, perf-critical, concurrent code** |

The thing the table makes plain: Rust is the only row that is simultaneously native-fast, compile-time-safe, and easy to build. That combination is why it has become the default for new native extensions.

## The ecosystem proof: Rust is already the native language of fast Python

If you want evidence that this is mainstream rather than a niche experiment, look at the tools that defined fast Python in the last few years. A striking number of them are Rust extensions wearing a Python API.

![A two column matrix showing six production tools with a Rust core and what each replaced, listing Polars as a multicore Arrow dataframe engine replacing single threaded pandas, pydantic-core powering pydantic v2 about five times faster than the pure Python v1, ruff a linter and formatter ten to one hundred times faster replacing flake8 isort and black, and HF tokenizers processing gigabytes per second replacing Python tokenizers](/imgs/blogs/rust-for-python-pyo3-and-maturin-6.png)

Walk through the matrix:

- **Polars** is a DataFrame library with a Rust core (built on Apache Arrow) that is multi-threaded by default and uses a lazy query optimizer. On many real analytical workloads it runs several times to an order of magnitude faster than single-threaded pandas, and it is the subject of [the dataframes-at-speed post](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) in this series. When you `import polars`, you are importing a Rust extension built with PyO3-style bindings.
- **pydantic-core** is the Rust validation engine underneath pydantic v2. Rewriting the core from pure Python (v1) into Rust made validation on the order of five to twenty times faster depending on the schema, which matters enormously for web frameworks like FastAPI that validate every request.
- **ruff** is a Python linter and formatter written in Rust that replaces the combined functionality of flake8, isort, pyupgrade, and (with its formatter) black — and runs commonly 10 to 100 times faster than the tools it replaces, fast enough to lint a large codebase in well under a second. Its speed changed how people use linters: you can run ruff on every keystroke.
- **HF tokenizers** is Hugging Face's tokenization library, a Rust core that tokenizes text at gigabytes per second, far faster than a pure-Python tokenizer, which matters when you are feeding a training run or a high-throughput inference server.
- **uv** is an extremely fast Python package installer and resolver, written in Rust, that resolves and installs dependencies 10 to 100 times faster than pip in many scenarios — fast enough that it changes the ergonomics of creating environments.
- **orjson** is a JSON library with a Rust core that serializes and deserializes several times faster than the standard library `json`, with correct handling of types `json` does not support natively.

The pattern is unmistakable: when a Python tool needs to be *fast* and the work is string processing, parsing, validation, or concurrent — exactly the irregular work NumPy cannot vectorize — the 2026 answer is a Rust core behind a Python API, built with PyO3 and maturin. These are not toys; they are load-bearing infrastructure for a large fraction of the Python ecosystem. That is the strongest possible proof that the path this post teaches is the mainstream one.

#### Worked example: estimating ruff's structural advantage

Let us make the ruff number concrete with a back-of-the-envelope that explains *why* it is so large, rather than just quoting it. Linting is fundamentally: parse each file into a syntax tree, then walk the tree applying hundreds of rules. The parse and the walk are exactly the per-node, branch-heavy, allocation-heavy work that Python is worst at and Rust is best at. Suppose a codebase has 10,000 files averaging 200 lines. A pure-Python linter might process, optimistically, a few thousand lines per second per rule pass after the interpreter tax, so a full multi-tool run (flake8 plus isort plus pyupgrade plus black) over two million lines can take tens of seconds. ruff parses with a Rust parser, walks a Rust syntax tree with no boxing and no eval loop, and parallelizes across cores with no GIL — so the same two million lines is processed in a fraction of a second. The 10-to-100x is not a tuned benchmark; it is the structural gap between an interpreted tree-walk and a compiled, parallel one, which is the same gap we measured at 18x single-threaded and 118x parallel on our own kernel. ruff's advantage and our kernel's advantage have the same source.

## When Rust is the right call, and when it is overkill

Rust is powerful, but it is not the answer to every performance problem, and reaching for it reflexively is its own mistake — you can sink a week into a Rust rewrite that a three-line Numba decorator would have matched. Here is the honest decision.

![A decision tree for choosing a native tool, branching from need native speed into three cases, new code you will own that is performance critical pointing to Rust with PyO3 for string parse and concurrent work, wrapping an existing C library pointing to ctypes or cffi with no build step, and a one off numeric loop pointing to Numba njit in three lines](/imgs/blogs/rust-for-python-pyo3-and-maturin-8.png)

The tree splits on the question that actually matters: *what kind of native code do you need?*

**Reach for Rust + PyO3 when** the code is **new, performance-critical, and yours to own and maintain for the long term**, especially when the work is **string processing, parsing, or concurrent**. This is the irregular work that NumPy cannot vectorize and that Numba is weak at — text munging, tree-walking, custom data structures, anything with branches and hash maps in the hot loop. It is also the right call when you need **real multicore parallelism** on CPU-bound work, because the GIL-release-plus-rayon path has no equal in the Python world. If you are building a library that thousands of people will depend on — a parser, a validator, a dataframe engine — Rust's compile-time safety pays for itself many times over.

**Reach for `ctypes` or `cffi` instead when** you just need to **wrap an existing C library** and call a few functions from it. You do not need to rewrite anything; you need a thin binding, and `ctypes` needs no build step at all. Writing a Rust crate to wrap a C library you are not changing is overkill — though if you are wrapping *and* adding substantial new logic, Rust (which has excellent C FFI itself) can be the better home.

**Reach for Numba instead when** you have a **one-off numeric loop** — a hot kernel of array math, a Monte Carlo inner loop, a numerical integration — that you can decorate with `@njit` and get a 50-to-200x speedup in three lines, with no separate build, no wheel, no language to learn. Numba is the [JIT-compile-the-loop lever](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast); for numeric work it is faster to reach for than Rust and often just as fast at runtime. Where Numba struggles — strings, complex data structures, code you want to ship as a maintained library — is exactly where Rust shines.

And the most important "when not": **do not reach for any native rewrite until you have measured that the kernel is actually the bottleneck.** Amdahl's law is unforgiving. If the parse loop is 70% of your runtime, a 100x rewrite of it takes the total from 100% to about 31% of original — a 3.2x overall win, excellent. But if the parse loop is only 10% of runtime, the same 100x rewrite takes you from 100% to about 90.1% — an 11% win, for a week of Rust work and a new toolchain in your build. Profile first. The native lever is the third rung of the ladder for a reason: you climb it only after you have done less work and vectorized what you can, and only on the slice the profiler proves is hot.

Two more honest costs of the Rust path. First, **build complexity for distribution**: while `maturin develop` is trivial locally, shipping wheels for every platform (Linux manylinux, macOS x86 and ARM, Windows) means a CI matrix — though maturin and tools like `maturin-action` make this far easier than it was for C. Second, **the learning curve**: Rust's borrow checker is genuinely harder to learn than Python, and the first week fighting the compiler is real. The payoff is that once it compiles, a whole class of bugs is simply gone — but budget for the ramp.

## How to measure a native extension honestly

Because this series' ethic is "always prove the win with a number," a word on measuring Rust extensions correctly, since they have their own traps.

**Always benchmark a `--release` build.** A debug build (`maturin develop` without `--release`) disables the optimizer and runs 5 to 50x slower. Benchmarking a debug build and reporting it as "Rust" is the single most common way people accidentally lie about native performance. Make `--release` a reflex.

**Account for the one-time compile cost separately.** The Rust compile happens at build time, not call time, so unlike Numba there is no first-call JIT warmup — but the *first import* of a freshly built extension may touch the filesystem. Warm up with one call, then time.

**Time the boundary explicitly when it might dominate.** For small inputs the FFI overhead can be a meaningful fraction. Benchmark across input sizes — 1k rows, 100k rows, 5M rows — and watch where the per-call boundary cost stops mattering. For tiny inputs, the pure-Python version may even win, because 200 ns of boundary crossing is more than the whole computation; the Rust win is for *large* batches, which is the point of the few-big-calls rule.

**Measure the GIL-released parallel scaling with the GIL actually contended.** The whole point of `allow_threads` is that other Python threads run during the Rust computation. To prove it, run Python threads doing other work alongside the Rust call and confirm they make progress; a single-threaded benchmark cannot show the GIL benefit. And report parallel efficiency, not just speedup, so the sub-linear scaling (memory bandwidth, the serial merge) is visible and honest.

Here is a minimal honest harness:

```python
import gc, time, statistics
import fastlog

def bench(fn, data, n=7):
    fn(data)                          # warm up (filesystem, caches)
    gc.disable()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - t0)
    gc.enable()
    return statistics.median(times), statistics.pstdev(times)

with open("batch.log") as f:
    data = f.read()

med, sd = bench(fastlog.aggregate_rust, data)
print(f"rust single: {med*1000:.1f} ms  (sd {sd*1000:.2f} ms)")
med, sd = bench(fastlog.aggregate_rust_parallel, data)
print(f"rust rayon:  {med*1000:.1f} ms  (sd {sd*1000:.2f} ms)")
```

Median of repeats, GC disabled during timing, a warmup call, the same input each run — the same discipline as [benchmarking any Python correctly](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple), applied to the native boundary.

## Case studies and real numbers

Beyond our own kernel, the public record has several well-documented Rust-rewrite results worth knowing, framed honestly with their sources.

**pydantic v1 to v2 (pydantic-core).** When pydantic moved its validation core from pure Python (v1) to Rust (v2, via the `pydantic-core` crate built with PyO3), the maintainers reported validation speedups on the order of 5x to 50x depending on the model and operation, with a commonly cited "around 17x on average" figure for typical models. The reason is structural: validation is per-field branching and type-checking, the per-object Python tax multiplied across every field of every request. Moving it to compiled Rust deletes that tax. For a FastAPI service validating thousands of requests per second, this is a direct latency and throughput win on the hot path.

**ruff.** The ruff project documents 10x-to-100x speedups over the Python linters it replaces, and the difference is large enough to be qualitative: linting a large monorepo that took tens of seconds with flake8 takes a fraction of a second with ruff, which means it can run on save, in pre-commit, and in CI without anyone noticing the time. The mechanism, as we estimated above, is a compiled parallel tree-walk replacing an interpreted serial one.

**Polars vs pandas.** On analytical benchmarks (the kind in the TPC-H-style query suites the Polars team publishes), Polars frequently runs several times to an order of magnitude faster than pandas on multi-core machines, because it is multi-threaded by default (no GIL inside the Rust core, the same trick as our `allow_threads` + rayon) and has a query optimizer that fuses operations and avoids materializing intermediate columns. The exact multiple depends heavily on the query, the data size, and the core count, so treat any single number with suspicion — but the direction, and the *reason* (a Rust columnar engine using all cores), is solid and is covered in [the dataframes post](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow).

**Our kernel, restated.** On the reference machine, the log-aggregation kernel went from 8.40 s (pure Python) to 0.46 s (single-threaded Rust, 18.3x) to 0.071 s (8-core rayon with the GIL released, 118x). Forty lines of safe Rust, one `maturin develop`, zero memory-management bugs possible by construction. That is the shape of the win this lever buys when the work fits it.

A note on trusting these numbers: every public speedup is workload-dependent, and the marketing figure is usually the best case. The discipline is the same as the rest of this series — when you read "17x faster" or "100x faster," ask *on what workload, what input size, what core count, what version*, and then measure your own case. The structural argument (no interpreter, no boxing, no GIL, all cores) tells you the *direction* and roughly the *magnitude*; only your own profiler tells you the number that matters for your code.

## A second worked example: choosing the lever for two real loops

To make the decision concrete, here are two loops I have actually had to speed up, and the lever each one wanted.

#### Worked example: two hot loops, two different answers

**Loop A — a Monte Carlo option-pricing inner loop.** Pure numeric work: draw random numbers, evolve a price path, average the payoffs, repeated over millions of paths. No strings, no branches to speak of, all `float64` arithmetic over arrays. On the reference machine the pure-Python version ran in about 12 seconds for one million paths. The right lever is **Numba**: a `@njit(parallel=True)` decorator with a `prange` over the paths took it to about 0.18 seconds — roughly 67x — in *three lines of changes and no new toolchain*. Rewriting this in Rust would have produced a comparable runtime but cost a day of work, a Cargo project, and a wheel to ship, for no benefit over the three-line Numba version. **Verdict: Numba.** Reaching for Rust here would have been overkill — the work is numeric, one-off, and Numba owns that case.

**Loop B — the log-parsing kernel of this post.** Irregular work: string splitting, integer parsing, branching on a status field, hash-map accumulation, and it is a maintained library function called all day in production. Numba is weak here — it does not handle arbitrary string work and `dict` of strings well, and even where it does, the result is fragile. NumPy cannot vectorize branching string parses at all. The pure-Python version ran in 8.4 seconds; the Rust version ran in 0.46 seconds single-threaded and 0.071 seconds across eight cores, with the parse logic memory-safe by construction and the code maintainable for years. **Verdict: Rust + PyO3.** This is exactly the case Rust is built for: new, hot, string-and-parse-heavy, concurrent, and owned.

The contrast is the whole lesson. The two loops are both "a hot loop I need to speed up," and they want opposite tools. The numeric one-off wants the three-line JIT; the irregular, owned, concurrent library function wants the compiled, memory-safe, parallel native rewrite. Diagnosing *which* you have — by looking at the kind of work and your ownership horizon, not just the speedup you want — is the skill. The decision tree from the previous section is the checklist; these two loops are why it has the branches it does.

## Where this sits in the leverage ladder

Step back to the series' spine. The leverage ladder is: (1) do less work with the right algorithm and data structure, (2) do it in bulk by vectorizing into NumPy or pushing it into Polars, (3) compile the hot one percent into native code, and (4) use every core and overlap I/O. Rust + PyO3 is the most powerful tool on rung three, and — because of `allow_threads` plus rayon — it reaches up into rung four, giving you real multicore parallelism that the GIL denies pure Python.

But notice the discipline the ladder enforces. You do not start here. You start by measuring, you climb the cheap rungs first (a better algorithm is free; vectorizing is a few lines; Numba is three lines), and you reach for Rust only when the work is irregular enough that vectorization and Numba do not fit, *and* hot enough that the rewrite pays, *and* yours enough to own that the maintenance cost is worth it. When all three are true — which, for string and parsing and concurrent work, they often are — Rust is the best tool in the box, and the existence of Polars, ruff, pydantic-core, tokenizers, uv, and orjson proves the whole industry agrees. It is, genuinely, the native extension language of 2026.

## Key takeaways

- **Rust's borrow checker turns the four dangerous C-extension bug classes — use-after-free, double-free, data races, and refcount leaks — into compile errors**, and it does so without a garbage collector, so you keep C's predictable, pause-free deallocation.
- **PyO3 is three macros**: `#[pyfunction]` exposes a function, `#[pyclass]` exposes a stateful object, `#[pymodule]` builds the importable module. PyO3 handles the marshaling and the reference counts so you do not.
- **maturin makes the build trivial**: `maturin develop --release` compiles and installs into your virtualenv so `import` just works; `maturin build --release` produces a distributable wheel. Always benchmark release builds.
- **The FFI boundary costs per call, so make few big calls, not many small ones.** Move the loop inside Rust and hand it the whole batch; calling Rust per row re-introduces the per-row tax plus FFI overhead and can be slower than pure Python.
- **`py.allow_threads` releases the GIL for the pure-Rust portion**, and `rayon`'s `par_iter` then uses every core with data races impossible by compile-time guarantee — real multicore parallelism that pure-Python threads cannot deliver.
- **Speedups are structural, not magical**: roughly 18x single-threaded (deleting boxing and the eval loop) times core-scaling efficiency around 80% (limited by the serial merge and memory bandwidth, exactly as Amdahl predicts) gave 118x on eight cores for our kernel.
- **Reach for Rust when the code is new, hot, string-or-parse-or-concurrent, and yours to maintain.** Reach for `ctypes` to wrap an existing C library, and for Numba for a one-off numeric loop. Profile first — Amdahl caps your win at the fraction you are actually speeding up.
- **This is mainstream, not exotic**: Polars, pydantic-core, ruff, HF tokenizers, uv, and orjson are all Rust cores behind Python APIs, which is why Rust is the default for new native extensions in 2026.

## Further reading

- The PyO3 user guide and API docs (`pyo3.rs`) — the canonical reference for `#[pyfunction]`, `#[pyclass]`, `#[pymodule]`, and `allow_threads`.
- The maturin documentation (`maturin.rs`) — `develop`, `build`, manylinux wheels, and CI packaging.
- The rayon crate docs — data-parallel iterators (`par_iter`, `reduce`) and the work-stealing scheduler.
- *The Rust Programming Language* ("the book") — ownership, borrowing, and lifetimes, the model behind the safety guarantees.
- The pydantic v2, ruff, and Polars project blogs and benchmark pages — primary sources for the production speedup numbers, each with the workload stated.
- The CPython C-API "Initialization, Finalization, and Threads" docs — what `Py_BEGIN_ALLOW_THREADS` does, the thing PyO3's `allow_threads` wraps safely.
- *High Performance Python* by Gorelick and Ozsvald — the broader native-extension landscape this post sits in.
- Within this series: the [series introduction on why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), the companion [post on dataframes at speed with Polars and Arrow](/blog/software-development/python-performance/dataframes-at-speed-pandas-pitfalls-polars-and-arrow) (Polars is itself a Rust extension), and [the CPython execution model and the eval loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) for the GIL mechanics this post releases. This post closes the Go-Native track, which also covers the native-acceleration landscape, Numba, Cython, and the C and C++ FFI options; the capstone playbook ties every lever together into one decision framework.

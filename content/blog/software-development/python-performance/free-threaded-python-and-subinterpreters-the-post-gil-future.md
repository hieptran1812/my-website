---
title: "Free-Threaded Python and Subinterpreters: The Post-GIL Future"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "See what changes when CPython drops the GIL, run a CPU-bound thread that finally scales on python3.13t, spin up isolated subinterpreters with channels, and learn what is actually usable today versus what is still an experiment."
tags:
  [
    "python",
    "performance",
    "free-threading",
    "subinterpreters",
    "gil",
    "concurrency",
    "parallelism",
    "optimization",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-1.png"
---

I spent most of a Tuesday last year staring at a graph that should not have existed. We had a batch scoring job — a few million records, a per-record feature computation that was pure CPU arithmetic in Python, no I/O in the hot loop at all — and someone had "parallelized" it with a `ThreadPoolExecutor` and eight workers. The box had eight cores. The expectation, reasonable on its face, was a roughly eight-fold speedup. What we got was a job that ran *slightly slower* than the single-threaded version, with all eight cores hovering politely around twelve percent utilization, none of them ever pinned. The CPU graph looked like eight people taking turns at one keyboard. Which is exactly what was happening. The Global Interpreter Lock — the one mutex that, on the standard CPython build, only ever lets a single thread execute Python bytecode at a time — was doing precisely its job, and its job was to make our eight threads run one at a time.

If you have read [the GIL post in this series](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs), none of that surprises you. The GIL is why CPU-bound Python threads do not scale, and the standard answer for the last twenty years has been: reach for [multiprocessing and eat the pickling tax](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling), because separate processes each get their own GIL and so run genuinely in parallel. That works, but it is a workaround. You pay to start the processes, you pay to serialize your data across the process boundary, and you give up the one thing threads were good for: cheap, shared, in-memory state.

This post is about the two changes that are finally dismantling that twenty-year compromise. The first is **free-threaded CPython** (PEP 703): a real build of the interpreter, shipping as `python3.13t`, with the GIL *removed*, so that CPU-bound `threading` threads run truly in parallel across cores, sharing memory, with no pickling and no separate processes. The second is **subinterpreters** (PEP 554 and PEP 734): multiple independent Python interpreters living inside one process, each with its own GIL and its own isolated state, giving you real parallelism with strong isolation but without the cost of spawning OS processes. Both are usable *today* on Python 3.13 and 3.14 — and both come with caveats sharp enough that you need to understand them before you bet a production system on them.

![A before and after diagram contrasting eight CPU-bound threads on the standard CPython build running in 8.1 seconds at one times speedup against the same threads on the free-threaded build running in 1.3 seconds at 6.2 times speedup](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-1.png)

This is the forward-looking post that closes the concurrency track of the series. We are on the top rung of the leverage ladder this whole series climbs — after you have done less work with the right algorithm, done it in bulk with NumPy, and compiled the hot one percent into native code, the last lever is to *use every core*. Until now, using every core from Python meant processes or native code. The post-GIL future is about using every core from threads, the way every other language already can. By the end you will know what free-threading actually changes inside the interpreter and why it was so hard, what it costs you today, how to install and try `python3.13t`, how to create subinterpreters and pass data through channels, and — the part that matters most for your next architecture decision — what is genuinely worth using in 2026 versus what is still an experiment you should not yet stake a deadline on.

## Why removing the GIL was hard in the first place

It helps to start with the question that took the CPython core team a literal decade to answer with a confident "yes": *why was the GIL there at all, and why couldn't you just delete it?*

The GIL exists to protect the interpreter's internal data structures — most importantly, the reference counts on every object. CPython manages memory with **reference counting**: every Python object carries a small integer, its refcount, that records how many references currently point at it. When you write `b = a`, you create a new reference to the same object, so its refcount goes up by one. When a name goes out of scope or you `del` it, the refcount goes down by one. When the count hits zero, the object is freed immediately. This is the dominant memory-management strategy in CPython; a separate generational garbage collector exists only to clean up *reference cycles* that pure refcounting cannot reclaim. If you want the full mechanics of how bytecode execution drives all of this, [the CPython execution model post](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) walks through the eval loop where these increments and decrements actually happen.

Here is the trap. Incrementing and decrementing a refcount is a *read-modify-write*: load the current value, add or subtract one, store it back. On a single thread that is three cheap instructions. On multiple threads running at once, it is a textbook data race. If two threads both hold a reference to the same object and both decrement its count at the same time, the classic interleaving — thread A reads 2, thread B reads 2, A writes 1, B writes 1 — leaves the count at 1 when it should be 0. The object never gets freed: a memory leak. Run the same race the other way and you can free an object while another thread still points at it: a use-after-free, which is a crash at best and a security hole at worst. Every single attribute access, every function call, every loop iteration in Python touches refcounts constantly. There is no way to make that safe across threads without *some* synchronization, and the GIL was the simplest possible synchronization: one lock, held while any thread runs bytecode, released only at well-defined points (between bytecode instructions periodically, and around blocking I/O and many C calls).

So the naïve fix — "just put a lock around each refcount operation" — does not work, and the reason is worth internalizing because it is the entire performance story of free-threading.

#### Worked example: why per-refcount locking is a disaster

Suppose you replace the GIL with a per-object atomic increment on every refcount touch. An uncontended atomic operation on a modern x86-64 core costs on the order of 15 to 25 nanoseconds because it forces the CPU to lock the cache line and order memory globally — perhaps ten to twenty times the cost of a plain non-atomic add, which is effectively free at well under a nanosecond once it is in a register.

Now count how often Python touches refcounts. A trivial loop like `for x in data: total += x` over ten million integers performs, very roughly, tens of millions of refcount operations: binding `x` each iteration, the arithmetic creating and discarding temporary integer objects, the name lookups. If every one of those costs an extra ~20 nanoseconds instead of being free, you have added on the order of *hundreds of milliseconds to seconds* of pure synchronization overhead to a loop that did real work measured in tens of milliseconds. You would have a thread-safe interpreter that is several times slower on *one* thread than the GIL build — which is worse than useless, because most Python programs and most parts of every Python program are single-threaded. A no-GIL build that tanks single-thread performance is dead on arrival, and that is exactly why every previous attempt failed.

The PEP 703 design that finally shipped solves this with two clever refcounting schemes that make the *common* case cheap and only pay the atomic cost in the *rare* case.

The first is **biased reference counting**. The insight is that most objects are only ever touched by the thread that created them. So each object's refcount is split internally into a "local" count, owned by the creating thread, and a "shared" count for everyone else. The owning thread mutates its local count with plain, non-atomic operations — fast, just like the GIL build — because no other thread is allowed to touch that local count. Only when a *different* thread references the object does it go through the slower atomic path on the shared count. For the overwhelmingly common single-owner object, you pay nothing extra; the atomic tax falls only on genuinely shared objects.

The second is **deferred reference counting**, used for objects that are referenced constantly from many places and whose refcount would otherwise thrash — things like the top-level functions, modules, type objects, and other long-lived globals that every thread reaches for. For these, CPython stops counting individual references from the interpreter's own evaluation stack and instead defers the accounting: the cyclic garbage collector becomes responsible for deciding when such an object is truly unreferenced. This removes a whole category of high-frequency contended refcount updates. Add to these a handful of *immortal* objects — `None`, `True`, `False`, small integers — whose refcount is simply never modified at all (they are pinned to a sentinel value and live forever), and the most heavily shared objects in any program stop generating refcount traffic entirely.

There is a third piece people forget: the **memory allocator**. CPython's default allocator, `pymalloc`, is a small-object allocator with internal free lists, and like the refcounts it was not thread-safe — it assumed the GIL serialized all access. A no-GIL build needs an allocator that many threads can call into at once without corrupting those free lists. PEP 703 adapts a thread-safe allocator (based on `mimalloc`) so that allocation and deallocation, which Python does constantly, can proceed in parallel. Without that, every `list.append` that grows a buffer, every temporary object, would funnel through one lock and you would have rebuilt the GIL under a different name.

Even with all of that, free-threading is not free. The single-thread overhead does not drop to zero — biased refcounting still has to check ownership, deferred counting adds GC pressure, and, importantly, the free-threaded build had to *disable* the PEP 659 specializing adaptive interpreter at first, because that specialization machinery (the part of the "Faster CPython" work that rewrites hot bytecodes in place) was itself not thread-safe. Losing specialization is a real performance hit on single-threaded code. That is the core of "the cost today," and it is why the honest framing of free-threading in 2026 is *experimental, improving, not yet the default.*

There is one more layer of the design worth understanding, because it explains why removing the GIL is not the same as removing *all* locking, and it shapes how you reason about correctness. Refcounting was only the most pervasive thread-safety problem; the interpreter's *other* mutable internals — the contents of a `dict`, the backing array of a `list`, the freelists inside the allocator, the type cache — also needed protecting once threads could touch them simultaneously. PEP 703 protects these not with one big lock but with **fine-grained, per-object locks**. Each container essentially carries its own small mutex, taken only for the brief operations that mutate it. So when two threads append to two *different* lists, they take two *different* locks and never contend; only two threads hammering the *same* list serialize, and only for the duration of the append. This is the entire philosophy of the design in one sentence: replace the single coarse lock that serialized *everything* with many tiny locks that serialize only *actual* conflicts. The cost moves from "always pay for the lock" to "pay only when you genuinely share," which is exactly the same principle as biased reference counting applied to data structures instead of counts.

A related subtlety is the **critical-section** mechanism the interpreter uses internally to make a sequence of operations on one or two objects atomic without risking deadlock. When CPython needs to do something like "look up a key and then mutate the dict based on it" atomically, it cannot naïvely hold a lock across an operation that might call back into arbitrary Python code (because that Python code might try to take the same lock and deadlock). The free-threaded build handles this with a scheme that can temporarily *release* an object's lock at well-defined suspension points and re-acquire it, preserving safety without the deadlock. You do not write this code yourself — it lives inside the C implementation of the built-in types — but knowing it exists explains why the built-ins remain safe to *use* concurrently even though *your* compound operations on them are not automatically atomic. The container will not corrupt; your "increment this shared counter" logic still needs your own lock.

And there is a garbage-collection consequence that is easy to miss. Under the GIL, the cyclic garbage collector ran while it alone held the lock, so it could walk the entire object graph knowing nothing else was mutating it. Without the GIL, the collector needs another way to get a consistent view, so the free-threaded build uses a brief **stop-the-world** pause: it signals all running threads to reach a safe point and pause, does its cycle-detection work over a stable graph, then lets them resume. These pauses are short, but they are a new source of latency that did not exist before in the same form, and they are the kind of thing you would want to measure if you were running a latency-sensitive service. It is one more line on the ledger of what free-threading costs, alongside the per-op refcount tax and the lost specialization — all of it the price of letting your threads finally use every core.

## Free-threaded CPython: what python3.13t actually gives you

With the *why* established, the *what* is gloriously simple to state: on the free-threaded build, the GIL is gone, so `threading.Thread` and `concurrent.futures.ThreadPoolExecutor` run Python bytecode on multiple cores at the same time. CPU-bound code that was flat under the GIL now scales. And because threads share one address space — the same heap, the same objects — you pass data between them by *reference*, with no pickling, no serialization, no copy. That last property is the whole reason this matters and not just "multiprocessing with nicer syntax."

The build is distributed as a separate interpreter named `python3.13t` (the trailing `t` is for "threaded"). It is a genuinely different build of CPython, compiled with the `--disable-gil` configure option, not a runtime flag you flip on the normal interpreter. You can have both `python3.13` and `python3.13t` installed side by side.

Here is the first thing you should actually run once you have it: ask the interpreter whether the GIL is disabled, because you never want to *think* you are running free-threaded and be wrong.

```python
import sys
import sysconfig

# True only on a free-threaded build like python3.13t.
# On a standard build this attribute may not exist at all on older 3.13.x,
# so guard it.
gil_disabled = getattr(sys, "_is_gil_enabled", None)
if gil_disabled is not None:
    print("GIL currently enabled:", sys._is_gil_enabled())

# The build-time configuration is the source of truth.
print("Free-threaded build:", bool(sysconfig.get_config_var("Py_GIL_DISABLED")))
print("Version:", sys.version)
```

On `python3.13t` you will see `Free-threaded build: True`, and `sys._is_gil_enabled()` returns `False`. On the ordinary `python3.13` you get `False` and `True` respectively. The reason there are *two* checks is subtle and worth knowing: even a free-threaded build can *re-enable* the GIL at runtime. If you import a C extension that has not declared itself compatible with free-threading, CPython will, by default, switch the GIL back on for the whole process to keep that extension safe. So `Py_GIL_DISABLED` tells you "this binary can run without the GIL," while `sys._is_gil_enabled()` tells you "is the GIL actually off *right now*." Both being the answer you expect is the only state you should trust.

This automatic re-enabling deserves a moment of respect, because it is both a safety feature and a silent trap. It is a safety feature because it means a free-threaded build will not crash when it loads an extension that was never audited for thread-safety: rather than run that extension unsafely, the interpreter conservatively turns the GIL back on and everything behaves like the standard build. It is a silent trap because nothing stops your program from running — it just runs *serially*, and unless you check, you might conclude that "free-threading does not help my workload" when the truth is that one stray import switched it off. There is an environment variable, `PYTHON_GIL=0`, that forces the GIL to stay disabled even when such an extension loads, but you should reach for it only deliberately, because it overrides the very safety mechanism just described: with it set, an extension that was *not* designed for free-threading runs without the GIL anyway, and any latent thread-safety bug in that extension becomes your crash. The correct default workflow is to *not* force it, to check `sys._is_gil_enabled()` at startup, and to fix the offending dependency (find a free-threaded wheel, or upstream the support) rather than silently overriding the guard.

The shared-memory property is the one that most distinguishes free-threading from every other CPU-parallel option, so it is worth making concrete. Suppose you have a two-gigabyte read-only lookup structure — a trained model, a big index, an interned vocabulary — that every worker needs to consult. With multiprocessing you either load it once per process (eight workers, sixteen gigabytes of duplicated memory, plus eight separate load times) or you fight with `shared_memory` and manual layout to share the bytes. With subinterpreters you face a similar duplication, because each interpreter has its own object graph. With free-threaded threads you load it *once*, in shared memory, and all eight threads read the same two gigabytes by reference — no duplication, no copy, no serialization, and the read is a plain attribute access at full speed. For read-mostly shared state, free-threading is not just faster, it is dramatically lighter on RAM, and that memory win is frequently the deciding factor on a box where eight copies simply would not fit.

Now the workload that motivates the whole thing. We want something purely CPU-bound, with no I/O and no NumPy (because NumPy already drops into C and releases the GIL — we want to stress *Python-level* parallelism). A classic choice is counting primes by trial division: it is all integer arithmetic in the interpreter, exactly the kind of work the GIL serializes.

```python
import time
from concurrent.futures import ThreadPoolExecutor

def count_primes_below(limit: int) -> int:
    """Pure-Python CPU work: no I/O, no C library that releases the GIL."""
    count = 0
    for n in range(2, limit):
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            count += 1
    return count

def run(n_workers: int, chunk: int = 200_000, tasks: int = 8) -> float:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(count_primes_below, [chunk] * tasks))
    elapsed = time.perf_counter() - start
    return elapsed

if __name__ == "__main__":
    single = run(n_workers=1)
    multi = run(n_workers=8)
    print(f"1 worker:  {single:6.2f} s")
    print(f"8 workers: {multi:6.2f} s   speedup {single / multi:.2f}x")
```

Run that exact file under both interpreters and the difference is the entire point of this post.

```bash
# Standard build: the GIL serializes the eight threads.
$ python3.13 primes_threads.py
1 worker:    8.10 s
8 workers:   8.40 s   speedup 0.96x

# Free-threaded build: the eight threads run on eight cores.
$ python3.13t primes_threads.py
1 worker:    8.60 s
8 workers:   1.39 s   speedup 6.19x
```

Two things in those numbers deserve your attention, and I want to be honest that they are *illustrative figures on a plausible machine* — an 8-core x86-64 Linux box, 16 GB RAM, CPython 3.13 — not a benchmark I am asking you to treat as a published result. First, on the standard build, eight workers are *slower* than one (0.96x is a slowdown), because the GIL forces serialization *and* you now pay for thread-switching overhead and lock contention on top of the same total work. This is precisely the eight-people-one-keyboard graph from my Tuesday. Second, on the free-threaded build, eight workers hit roughly 6.2x rather than a perfect 8x — and notice the single-worker time is a touch *slower* on the free-threaded build (8.60 s versus 8.10 s). That single-thread gap is the free-threading tax we derived earlier: biased refcounting overhead plus the lost specialization gains. The 6.2x rather than 8x reflects that tax, plus normal scaling losses (memory bandwidth, the chunks not being perfectly equal, the OS scheduler).

### Measuring free-threaded scaling without fooling yourself

If you are going to make a decision based on numbers like these, you have to gather them honestly, and free-threading adds a few traps on top of the usual benchmarking ones. The general discipline — warm up, repeat, report the median rather than the minimum, use a large-enough workload that fixed overheads do not dominate — applies exactly as it does everywhere in this series. But there are free-threading-specific things to watch.

The most important is to measure the *right kind* of work. If your "CPU-bound" loop secretly calls into a C library that already releases the GIL — NumPy, hashlib, zlib, regex on large inputs — then it already scaled across cores on the *standard* build, and comparing it on the two builds tells you nothing about free-threading. That is why the prime-counting kernel above is pure interpreter arithmetic with no library calls: it is the part of your program that the GIL actually serialized. When you benchmark your own code, isolate the genuinely Python-level CPU hot path first (a profiler will show you where it is), and measure *that*, not the parts that already parallelized.

The second trap is the single-thread baseline. To quantify the free-threading tax you need an apples-to-apples single-thread number on *both* builds, running the identical code, ideally with the same package versions. Subtract them and you have the per-op overhead the free-threaded build is charging you; that number is what your serial fraction will pay forever, so it belongs in the Amdahl calculation from earlier. Do not estimate it from a blog post (including this one) — the value moves between releases and depends on what your hot loop does. Measure it on your release, your machine, your workload.

Here is a small harness that produces a scaling curve you can actually act on — wall time and computed speedup at 1, 2, 4, and 8 workers, with a warmup pass and a median over repeats:

```python
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

def measure(fn, args_list, n_workers, repeats=5):
    # One warmup pass so caches and any JIT/compile costs are paid.
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(fn, args_list))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(fn, args_list))
        times.append(time.perf_counter() - start)
    return statistics.median(times)

def scaling_curve(fn, args_list):
    base = measure(fn, args_list, n_workers=1)
    for w in (1, 2, 4, 8):
        t = measure(fn, args_list, n_workers=w)
        print(f"{w} workers: {t:6.2f} s   speedup {base / t:.2f}x")
```

A clean scaling curve on the free-threaded build looks like 1.0x, ~1.9x, ~3.6x, ~6.2x — sub-linear, because of the single-thread tax and bandwidth, but clearly climbing. On the standard build the same curve is flat or worse: 1.0x, ~0.97x, ~0.95x, ~0.96x — the unmistakable signature of the GIL. If your free-threaded curve is *also* flat, that is your signal that something re-enabled the GIL (a non-compatible extension), and you should check `sys._is_gil_enabled()` before you conclude that free-threading "did not help." The flat curve is diagnostic; do not skip it.

## The cost today, stated honestly

I labor this point because the marketing version of free-threading — "Python finally has real threads!" — is true but dangerously incomplete. Let me lay out the costs the way I would in a design review, because they decide whether you should use it.

![A matrix figure showing the four things free-threaded CPython changes — parallel threads, shared memory, refcounting, and C extensions — each paired with what changes and the cost such as single-thread overhead and a mandatory rebuild](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-3.png)

**Single-thread overhead.** As shipped in 3.13, the free-threaded build ran a single thread meaningfully slower than the standard build — early reports put it in roughly the 30 to 40 percent range on some workloads, largely because specialization was disabled. That is a steep price if most of your program is single-threaded, which most programs are. The good news, and the reason 3.14 matters so much, is that this gap has been narrowing fast: with specialization re-enabled in a thread-safe form and further tuning, the 3.14 free-threaded build brings the single-thread overhead down substantially — into the rough neighborhood of five to ten percent on many workloads. I want to flag clearly that those percentages are *approximate and move release to release*; treat them as "double-digit in 3.13, single-digit-ish in 3.14, still nonzero," and measure on your own code before you decide.

**C-extension compatibility.** This is the operational blocker today, not the interpreter itself. Any C extension — and that includes NumPy, pandas, anything with a compiled wheel — was written assuming the GIL serialized access to Python objects. On a free-threaded build those assumptions can be violated. An extension must be *rebuilt* against the free-threaded ABI and must explicitly declare it supports running without the GIL (it does this by calling `PyUnstable_Module_SetGIL` / setting the appropriate slot, so the interpreter knows not to re-enable the GIL on its account). If an extension has *not* opted in, importing it flips the GIL back on for the whole process — silently defeating the entire point. So in practice, whether free-threading helps your program depends on whether *every CPU-relevant C extension in your stack* has shipped a free-threaded-compatible wheel. As of 2026 the major scientific packages have made real progress here, but coverage across the long tail of PyPI is far from complete.

**Thread-safety is now your problem.** Under the GIL, a lot of accidentally-not-thread-safe Python code was *accidentally safe*, because the GIL made many individual bytecode operations effectively atomic — two threads could not actually run at the same instant. Remove the GIL and that accidental safety evaporates. A `dict` or `list` shared across threads and mutated concurrently can now race in ways it could not before. CPython guarantees the *interpreter* will not crash (the built-in containers have their own internal locking, so you will not segfault), but it does *not* guarantee your logic is correct: a read-modify-write on a shared counter still needs your own `threading.Lock`, and a "check then act" sequence across two objects is still a race. Free-threading does not make concurrency easy; it makes concurrency *possible*, and concurrency was always hard.

Here is the same trade summarized as a table, because a design review wants the costs and the gains in one place.

| Property | Standard build (GIL) | Free-threaded build (3.13t / 3.14t) |
| --- | --- | --- |
| CPU-bound threads scale across cores | No (serialized) | Yes |
| Memory shared between threads | Yes | Yes |
| Pass data without pickling | Yes | Yes |
| Single-thread speed | Baseline | ~30–40% slower in 3.13, ~5–10% in 3.14 (approx.) |
| C extensions work out of the box | Yes | Only if rebuilt and opted in |
| Your shared-state bugs are caught by the GIL | Often (accidentally) | No — you must lock correctly |
| Status | Default, stable | Experimental (3.13), supported but not default (3.14) |

The roadmap for all of this is deliberate and slow, and that is a feature, not a defect. Removing the GIL touches the most fundamental assumption in twenty years of CPython and C-extension code; the core team is right to gate the default behind years of real-world hardening.

![A timeline showing the GIL removal roadmap from PEP 703 accepted in 2023 through the experimental 3.13t build in 2024, the supported and faster 3.14 build in 2025, the ecosystem rebuild in 2026, and becoming the default someday with no committed date](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-2.png)

PEP 703 was accepted in 2023 with an explicit phased plan. Phase one: 3.13 ships the free-threaded build as an *experimental, opt-in* binary in 2024 — you have to choose `python3.13t`, and the steering council was clear it could still be pulled if it caused too much pain. Phase two: 3.14 in 2025 moves it from "experimental" to "officially supported but not the default," with the single-thread overhead substantially reduced and specialization restored. Phase three, with no committed date, is making it the default — and that only happens once the ecosystem has broadly shipped compatible extensions and the single-thread cost is small enough that nobody loses by it. We are squarely in phase two as of 2026. The build works, it is supported, you can ship it if your dependencies cooperate — but the standard GIL build remains the default for good reason.

#### Worked example: when is free-threading worth the single-thread tax?

This is a clean Amdahl's-law decision, so let me make it quantitative. Suppose your job is 80 percent CPU-bound parallelizable work and 20 percent inherently serial setup, and you are choosing between the standard build and the free-threaded build on an 8-core box.

On the standard build, threads do not help the CPU part at all, so your wall time is essentially the whole thing serial: call it 1.0 (normalized). On the free-threaded build, the serial 20 percent runs at, say, a 7 percent single-thread penalty (3.14-era), so it costs $0.20 \times 1.07 = 0.214$. The parallel 80 percent also pays the per-op penalty *and* then divides across 8 cores at, say, 85 percent scaling efficiency, costing roughly $0.80 \times 1.07 / (8 \times 0.85) \approx 0.126$. Total free-threaded wall time $\approx 0.214 + 0.126 = 0.34$, versus 1.0 standard — about a 2.9x win.

Now flip it: a job that is only 20 percent parallelizable and 80 percent serial. Standard build: 1.0. Free-threaded: serial part $0.80 \times 1.07 = 0.856$, parallel part $0.20 \times 1.07 / (8 \times 0.85) \approx 0.031$, total $\approx 0.89$. You saved about 11 percent — and if the single-thread penalty were the 3.13-era 35 percent instead of 7 percent, the serial part alone ($0.80 \times 1.35 = 1.08$) would make you *slower* than the standard build. The lesson is exact: free-threading pays off in proportion to how parallel your workload is, and the single-thread tax is a fixed cost you pay on the serial fraction whether or not the parallel part materializes. Measure your parallel fraction $p$ before you switch; the break-even moves with the build's current single-thread overhead.

## Subinterpreters: parallelism with isolation, no processes

Free-threading is one of the two doors out of the GIL. The other is older, more conservative, and in some ways more immediately practical because it does *not* require a special build of CPython: **subinterpreters**.

A subinterpreter is a second (or third, or hundredth) complete Python interpreter running inside the *same OS process*. Each subinterpreter has its own set of imported modules, its own `sys.modules`, its own top-level namespace — and, crucially, **its own GIL**. That last detail is the whole trick. The GIL is not actually one global lock for the process; since Python 3.12 it is per-interpreter. So if you have four subinterpreters, you have four independent GILs, and four threads — one running in each subinterpreter — can execute Python bytecode genuinely in parallel, on four cores, on a completely standard, unmodified CPython build. You get real CPU parallelism without removing anything and without spawning a single extra process.

![A graph showing one process with a shared address space branching into subinterpreter A with its own GIL and subinterpreter B with its own GIL, both feeding a channel that does send and receive, which produces a merged result without pickling to the operating system](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-4.png)

The price you pay for keeping the standard build is **isolation**: subinterpreters do *not* share objects. Each one has its own memory view of the Python world. You cannot just hand a list from interpreter A to a function running in interpreter B the way you can between threads, because they do not share the object graph the way threads do. This is the mirror image of the free-threading trade. Free-threading gives you shared memory and asks you to handle the locking; subinterpreters give you isolation and ask you to handle the *communication* — you pass data explicitly, through **channels**.

If you have internalized the [multiprocessing model](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling), subinterpreters will feel familiar, because the isolation story is similar: separate state, communicate by passing messages. The difference is what is underneath. Multiprocessing creates separate OS processes — that means a `fork` or `spawn` (expensive), and every message crosses a real process boundary through a pipe or socket, which means pickling on the way out and unpickling on the way in. Subinterpreters live in one process, so there is no `fork`/`spawn` of an OS process and no kernel IPC pipe; passing data through a channel can move it within the one shared address space, which is fundamentally cheaper.

Here is the modern API, which is the `concurrent.interpreters` module (the stdlib home as of 3.14, building on the work of PEP 734; on 3.13 the same capability lives in the lower-level `_interpreters` / the `interpreters` PyPI backport). Let me start with the most basic thing: create a subinterpreter and run code in it.

```python
from concurrent import interpreters

# Create a brand-new, isolated interpreter inside this same process.
interp = interpreters.create()

# Run a string of source in it. This executes in the subinterpreter's
# own namespace, with its own GIL, completely isolated from ours.
interp.exec("""
import sys
# This print happens inside the subinterpreter.
print("hello from interpreter", id(sys.modules))
""")

# Clean up when done.
interp.close()
```

Running a string is the lowest-level form. The far more useful form is `call`, which runs a function in the subinterpreter and returns its result to you — this is how you actually offload work:

```python
from concurrent import interpreters

def heavy(limit: int) -> int:
    count = 0
    for n in range(2, limit):
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        count += is_prime
    return count

interp = interpreters.create()
# The function and its argument are sent into the subinterpreter,
# it runs there (own GIL), and the integer result comes back.
result = interp.call(heavy, 200_000)
print("primes:", result)
interp.close()
```

Two constraints follow directly from isolation, and they bite the first time you hit them. The function and its arguments have to be *transferable* into the subinterpreter, and the data you move across that boundary must be of a kind the runtime knows how to share or copy safely. Simple, immutable, well-understood values — `int`, `float`, `str`, `bytes`, `bool`, `None`, and tuples of those — move cleanly. Arbitrary live objects with their own mutable state and references do not just teleport; that is the deliberate isolation boundary, and it is what keeps one subinterpreter's bug from corrupting another's memory.

## Channels: how subinterpreters actually talk

`call` is request/response. For ongoing communication — a producer in one interpreter, a consumer in another — you want a **channel**: a thread-safe, cross-interpreter queue. You create a connected pair of ends, give the send end to one interpreter and the receive end to another, and they pass data without ever sharing an object.

```python
from concurrent import interpreters
import threading

# A channel is a connected (send, receive) pair.
send, recv = interpreters.create_channel()

def producer(send_end):
    for i in range(5):
        # Only safely-shareable values cross the channel.
        send_end.send(i * i)
    send_end.send(None)  # sentinel: we are done

# Run the producer inside a fresh subinterpreter, on its own thread,
# so it executes in parallel with the consumer below.
interp = interpreters.create()

def run_producer():
    # Hand the send end into the subinterpreter and drive it.
    interp.prepare_main(send_end=send)
    interp.exec("""
for i in range(5):
    send_end.send(i * i)
send_end.send(None)
""")

t = threading.Thread(target=run_producer)
t.start()

# Meanwhile, the main interpreter consumes.
while True:
    item = recv.recv()
    if item is None:
        break
    print("got", item)

t.join()
interp.close()
```

The rule to carry away is that the channel is the *only* sanctioned bridge between two otherwise-sealed worlds. Nothing else crosses. There is no shared global you can both poke at, no object reference you can both hold. That sounds restrictive, and it is, but it is the same restriction that makes the model *safe to reason about*: because the only interaction is explicit message passing, there are no surprise data races on shared mutable state. You traded the hard problem (locking shared memory correctly) for the merely tedious one (deciding what to put on the channel).

For the common "I have a pool of work, run it across interpreters" case, you do not hand-roll any of this. The standard library ships an executor that mirrors `ThreadPoolExecutor` and `ProcessPoolExecutor` but uses subinterpreters as its workers:

```python
from concurrent.futures import InterpreterPoolExecutor

def heavy(limit: int) -> int:
    count = 0
    for n in range(2, limit):
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1
        count += is_prime
    return count

if __name__ == "__main__":
    # Each worker is its own subinterpreter with its own GIL,
    # so this runs in parallel on a standard CPython build.
    with InterpreterPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(heavy, [200_000] * 8))
    print("total primes:", sum(results))
```

If you have used `concurrent.futures` at all — and the [threading post in this series](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) leans on exactly this API — then `InterpreterPoolExecutor` is a drop-in you already know how to drive. The difference from `ThreadPoolExecutor` is that the workers truly parallelize CPU work; the difference from `ProcessPoolExecutor` is that they are cheaper to start and the data does not have to cross an OS process boundary.

### What isolation actually buys you, and where it bites

It is worth being precise about *why* subinterpreters are isolated, because the reason is also the source of every gotcha you will hit. The reason is module state. A normal CPython process has, historically, a lot of *global* state shared by everything in the process — the imported module table, certain caches, and module-level globals in C extensions. Subinterpreters work by giving each interpreter its *own* copy of that state: its own `sys.modules`, its own set of imported module objects, its own top-level namespace. That per-interpreter copy is what lets two of them run in parallel without stepping on each other — and it is exactly why you cannot reach across from one to another, because there is no shared global through which to reach.

This has a sharp practical consequence for C extensions. An extension that keeps module-level global state in C and assumes there is only ever *one* interpreter per process — which was a safe assumption for thirty years — will misbehave under subinterpreters, because now there can be several interpreters sharing that one C global. Extensions have to be written to support multiple interpreters per process (the modern multi-phase initialization style, where per-interpreter state lives in the module object rather than in C globals) to be safe here. Many widely used extensions have done this work; many older ones have not. So just as free-threading is gated by "is this extension thread-safe?", subinterpreters are gated by "does this extension support multiple interpreters per process?" — a different question, sometimes answered differently by the same package. Always check before you fan a third-party-heavy workload across subinterpreters.

The startup cost has a structural reason too. Because each subinterpreter gets its own module table, it has to *import* the modules your worker code needs — they are not shared from the parent. If your worker imports a heavy stack (pandas, say), each new subinterpreter re-runs those imports into its own table, and that import time, not the bare interpreter creation, often dominates spin-up. The practical lesson is to keep the per-task worker imports lean and to *reuse* interpreters across many tasks (which the `InterpreterPoolExecutor` does for you) rather than creating a fresh one per task. This is the same "amortize the startup over many tasks" discipline that makes a `ProcessPoolExecutor` worth it; the constant is just smaller for subinterpreters.

#### Worked example: stress-testing a subinterpreter pool

Pose the real problem. You have a CPU-bound scoring function, pure Python, that takes about 40 milliseconds per record, and a million records to process on an 8-core box. Serial, that is roughly 40,000 seconds — over eleven hours, clearly unacceptable. You want to fan it out.

Walk the decision. Threads on the standard build are out immediately: the work is CPU-bound, the GIL serializes it, you would get no speedup. That leaves three real options. Multiprocessing would work but each record is small, so the per-task pickling is cheap *relative to* the 40 ms of work — multiprocessing is actually fine here, and its fault isolation is a bonus. Subinterpreters on a standard build also work, with cheaper startup; the win over multiprocessing is modest because the data is small, so pickling was never the bottleneck. Free-threading would be the cleanest *if* your scoring function and its dependencies are free-threaded-ready, because then there is zero transfer cost at all.

Now stress-test the decision against the variations, which is where the interesting reasoning lives. *What if each record were 50 megabytes instead of small?* Then multiprocessing's pickling tax explodes — serializing 50 MB per task across a pipe could rival or exceed the 40 ms of compute — and subinterpreters pull clearly ahead, with free-threading (zero copy) ahead of both. *What if the scoring function occasionally segfaults from a flaky native dependency?* Then you want process isolation: a crashed worker process is retried, a crashed subinterpreter can take the whole pool down with it, so multiprocessing's heavier isolation suddenly earns its cost. *What if you scale from 8 to 64 cores?* Then the serial fraction and any shared bottleneck (a lock, a single output file) starts to dominate by Amdahl's law no matter which tool you pick, and the question stops being "which parallelism API" and becomes "what is still serial." *What if the work turns out to be I/O-bound after all — each record fetches from a database?* Then *none* of these is right and you want asyncio, because the cores were never the bottleneck. The right answer is not a tool; it is the *match* between the tool and the precise shape of the bottleneck, which is why you measure first and choose second.

## Subinterpreters versus processes: where the savings come from

Let me make the "cheaper than multiprocessing" claim concrete, because "cheaper" is doing a lot of work in that sentence and you should know exactly *which* costs go away.

![A before and after diagram contrasting multiprocessing, which forks or spawns a new operating system process and pickles arguments through IPC at roughly 50 milliseconds spawn cost, against subinterpreters, which create a new interpreter in the same process and send through a channel at roughly 1 millisecond spin-up](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-5.png)

There are two distinct costs in multiprocessing, and subinterpreters attack both.

The first is **startup**. Creating an OS process is not cheap. With the `spawn` start method (the default on macOS and Windows, and increasingly recommended everywhere because `fork` interacts badly with threads), Python launches a fresh interpreter executable, re-imports your modules, and re-runs initialization — that is on the order of tens of milliseconds per process, sometimes more if you have a heavy import graph. A subinterpreter, by contrast, is created inside the already-running process; it still has to set up its own module state, but it does not pay for an OS process launch or a fresh executable load. Spin-up is on the order of a millisecond or low single-digit milliseconds — *roughly* an order of magnitude cheaper than `spawn`, though I will stress these are approximate and depend heavily on how much each interpreter imports at startup.

The second, and often larger, cost is **data transfer**. To send an argument to a worker process, multiprocessing must `pickle` it (serialize the whole object graph to bytes), push those bytes through an OS pipe or socket (a kernel boundary crossing and a copy), and `unpickle` them on the other side (allocate and rebuild the object graph). For a big argument — a multi-megabyte list or array — pickling dominates everything; I have seen jobs where the "parallel" workers spent more time pickling than computing. Subinterpreters live in one address space, so passing data through a channel does not require a kernel pipe, and for the shareable types it can avoid the full pickle-to-bytes-and-back dance. You still cannot share a mutable object by reference (that is the isolation contract), but the per-message cost is lower and there is no kernel round-trip.

Let me put rough numbers on it, clearly marked as approximate, on the same plausible 8-core Linux box.

| Cost | multiprocessing (spawn) | subinterpreters | threads |
| --- | --- | --- | --- |
| Worker startup | ~30–60 ms / process (approx.) | ~1–3 ms / interp (approx.) | ~50–100 µs / thread |
| Pass a small int/str | pickle + pipe, ~µs–tens of µs | channel send, lower, no pipe | by reference, ~0 |
| Pass a 10 MB object | pickle + copy + unpickle, ~tens of ms | cheaper, one address space | by reference, ~0 |
| CPU parallelism | Yes (own GIL per process) | Yes (own GIL per interp) | Only on free-threaded build |
| Memory isolation | Strong (separate process) | Strong (separate interp) | None (shared) |
| Crash blast radius | One process only | Can take the whole process | Whole process |

That last row is the honest counterweight. A worker *process* that segfaults or runs out of memory dies alone; the parent survives and can retry. A subinterpreter shares the process, so a hard crash — say, from a buggy C extension — can take down everything. Multiprocessing's heavyweight isolation buys you fault containment that subinterpreters do not provide. So "subinterpreters are just cheaper processes" is too glib: they are cheaper *and* they share a fate. For most pure-Python compute that is a fine trade; for running untrusted or crash-prone native code, processes still earn their cost.

#### Worked example: a 10 MB array across the boundary, three ways

Concretely: you have a 10-megabyte list of numbers and a worker that does, say, 50 milliseconds of CPU work on it, and you want to run sixteen such tasks across 8 cores.

With **multiprocessing** (`spawn`), each task pays roughly: ~40 ms to start a worker (amortized if you reuse a pool, so call it near-zero after warmup), plus the killer — pickling 10 MB out, copying through the pipe, and unpickling 10 MB back, which can easily be 20 to 40 ms *per task* depending on the object's structure. So your 50 ms of real work is wrapped in 20 to 40 ms of pure serialization overhead. Effective efficiency: maybe 55 to 70 percent of theoretical, and on tiny tasks the pickling would dominate entirely and you would go *slower* than serial — the classic multiprocessing overhead trap.

With **subinterpreters**, you skip the OS process launch and the kernel pipe. The data transfer is still real work (the bytes have to get into the other interpreter), but without the pickle-to-bytes-and-back and without the kernel round-trip, the per-task overhead is meaningfully lower — call it single-digit milliseconds in this rough sketch. Your 50 ms of work is wrapped in much less overhead, so efficiency climbs.

With **free-threaded threads**, the data does not move at all: the worker reads the same 10 MB list by reference, zero transfer cost. The only overhead is the ~6.2x-not-8x scaling tax we measured earlier. For "lots of shared read-only data, CPU-heavy work," free-threading is the clear winner *if your extensions support it*; if they do not, subinterpreters are the best parallelism you can get on a standard build, and they beat multiprocessing on exactly the transfer cost that so often dominates. These figures are deliberately rough — I want you to remember the *shape* of the comparison (transfer cost: threads ≪ subinterpreters < processes), not memorize numbers I did not benchmark for your machine.

## Installing and trying 3.13t and subinterpreters today

None of this is hypothetical — you can run all of it this afternoon. Here is the practical "how to try it" path.

The cleanest way to get a free-threaded build is through a tool that manages Python versions and knows about the `t` variants. With `uv`, the modern installer many of us have standardized on, you ask for the free-threaded build explicitly:

```bash
# Install the free-threaded CPython 3.13 build with uv.
uv python install 3.13t

# Or pin a project to it.
uv venv --python 3.13t
source .venv/bin/activate
python -VV   # should mention "free-threading build"
```

If you use `pyenv`, the variant is available there too, and on some platforms the official python.org installers offer free-threading as an optional component you tick during install. The key signal, whichever route you take, is the verification snippet from earlier: `sysconfig.get_config_var("Py_GIL_DISABLED")` must be truthy. If you build from source yourself, the magic words are `./configure --disable-gil` (optionally `--enable-experimental-jit` on 3.13/3.14 if you also want to play with the JIT).

When you `pip install` packages into a free-threaded environment, pay attention to whether they ship free-threaded wheels. A package that has done the work publishes wheels tagged for the free-threaded ABI (you will see `cp313t` in the wheel filename rather than plain `cp313`). If only the non-free-threaded wheel exists, pip may still install it, but importing it will flip the GIL back on. You can check at runtime whether that has happened with the `sys._is_gil_enabled()` call — if it returns `True` on a build where `Py_GIL_DISABLED` is set, some extension re-enabled the GIL, and you should hunt down which import did it (toggling on the `PYTHONWARNDEFAULTGIL` style diagnostics, or importing your extensions one at a time, will find the culprit).

Subinterpreters need *no* special build at all, which is the underrated practical advantage. On a perfectly normal CPython 3.13, you can use the lower-level interface today; on 3.14 the friendly `concurrent.interpreters` module and `InterpreterPoolExecutor` are right there in the standard library. So if you want to experiment with the post-GIL future *without* swapping your interpreter and rebuilding your dependency tree, subinterpreters are the lower-friction entry point — your existing pure-Python code, your existing standard build, just a different parallelism API.

A sensible way to actually trial free-threading on an existing codebase, rather than betting on it blind, is a staged audit. First, stand up a free-threaded virtual environment and `pip install` your dependency tree into it, watching for any package that fails to install or installs only a non-free-threaded wheel — that list *is* your blocker list. Second, run your existing test suite under `python3.13t` (or `3.14t`); most pure-Python tests pass unchanged, and the failures point you at either thread-safety bugs the GIL used to hide or extensions that re-enabled the GIL. Third, add an assertion early in your process startup — `assert not sys._is_gil_enabled()` in a free-threaded build — so that if some import silently flips the GIL back on, you find out loudly at boot instead of discovering a mysteriously flat scaling curve in production. Fourth, only *then* benchmark your real hot path with the scaling harness from earlier. This sequence turns "should we adopt free-threading?" from a gamble into a checklist with a measured answer, and it surfaces the two failure modes — missing wheels and hidden races — before they cost you an outage.

For subinterpreters the trial is gentler, precisely because no build swap is involved. You can introduce an `InterpreterPoolExecutor` behind the same interface as your existing `ProcessPoolExecutor` and A/B them on the same workload, comparing wall time and peak memory. Because the API surface mirrors `concurrent.futures`, the swap is often a few lines, and the comparison is honest: same machine, same data, same task granularity, only the worker mechanism changing. If the subinterpreter version is faster and your data marshals cleanly over channels, you have a low-risk win on a standard build; if it is not, you have learned something concrete and reverted cheaply.

```bash
# Subinterpreters: standard build is fine.
$ python3.14 -c "from concurrent import interpreters; print(interpreters.create())"

# Free-threading: you need the special build.
$ python3.13t -c "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED'))"
1
```

## The 2026 concurrency toolbox: five tools, one decision

Step back and look at the whole landscape, because the post-GIL future does not *replace* the existing tools — it *adds* two and resorts the priority list. You now have five distinct ways to do more than one thing at a time in Python, and the skill is matching the tool to the bottleneck.

![A layered stack figure showing the five 2026 parallelism tools from threads for I/O only at the bottom through asyncio for tens of thousands of sockets, multiprocessing for CPU with a fork and pickle tax, subinterpreters for CPU with isolated channels, and free-threaded for CPU with shared memory at the top](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-6.png)

**Threads on the standard build** remain exactly what they always were: great for I/O-bound concurrency, useless for CPU-bound work. When a thread is waiting on a socket, a file, or a database, it releases the GIL, so other threads run — that is why a `ThreadPoolExecutor` of web requests scales beautifully even under the GIL. For CPU work, threads on the standard build still serialize. Nothing about free-threading or subinterpreters changes this; it just adds new options for the CPU case.

**asyncio** owns massive I/O fan-out: tens of thousands of concurrent connections in a single thread, with no per-connection OS thread stack and no GIL contention because there is only one thread cooperatively yielding at `await` points. If your problem is "talk to 50,000 endpoints at once," asyncio is still the answer, and free-threading does not change that — although an interesting near-future pattern is *several* event loops, one per core, on a free-threaded build, to scale async work across cores too.

**multiprocessing** remains the battle-tested way to get CPU parallelism *with fault isolation* on a standard build. Its costs — startup and pickling — are real, but its isolation is the strongest of any in-language option: a worker that crashes or leaks does not take the parent down. For long-running, coarse-grained, crash-prone, or untrusted work, it is still the right call.

**Subinterpreters** are the new middle path: CPU parallelism with strong isolation, on a standard build, cheaper than processes on startup and data transfer, but sharing a process fate. They shine for fan-out CPU work where the data is awkward to share by reference and you do not need process-level crash containment.

**Free-threading** is the new top of the ladder for the specific case it was built for: CPU-bound work over *shared* in-memory state, where pickling or channel-passing the data would dominate. It is the only option that gives you both real multicore CPU execution *and* zero-copy shared memory — provided your C extensions cooperate and you can stomach the single-thread tax.

Here is the comparison in one matrix, which is the figure I would put on the whiteboard in an architecture review.

![A matrix comparing threads, processes, asyncio, free-threaded, and subinterpreters across whether each gives CPU parallelism, shared memory, isolation, and what overhead each carries, showing that only free-threaded and subinterpreters give CPU parallelism on the relevant builds](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-7.png)

And the same logic as a decision you can run top-down in your head: start from the bottleneck, then branch on shared memory versus isolation.

![A decision tree starting from what is the bottleneck, branching to CPU-bound which splits into wanting shared memory leading to free-threaded 3.13t or needing isolation leading to subinterpreters or processes, and I/O-bound leading to asyncio for tens of thousands of sockets](/imgs/blogs/free-threaded-python-and-subinterpreters-the-post-gil-future-8.png)

The tree reads in one breath. Is the bottleneck I/O? Use asyncio (or threads, for smaller fan-out). Is it CPU? Then: do you want to share a big in-memory structure across the workers? If yes and your extensions support free-threading, use the free-threaded build and plain threads. If you need isolation — because the work is crash-prone, or you want clean separation, or your extensions are not free-threaded-ready — use subinterpreters for in-process isolation, or full processes when you also need fault containment. That single fork in the road — *shared memory or isolation?* — is the question the whole post-GIL future asks you to answer consciously, where before the GIL answered it for you (badly, by serializing everything).

## Case studies and real numbers

Let me ground this in what is actually being reported, with the standing caveat that free-threading numbers in 2026 are early and move release to release.

**The single-thread overhead trajectory.** The most-cited real figure is the single-thread regression of the free-threaded build versus the standard build. The 3.13 free-threaded build, with specialization disabled, was reported in the rough neighborhood of 30 to 40 percent slower on single-threaded benchmarks — a number the CPython team itself flagged as the main blocker to making it the default. The 3.14 work to re-enable specialization in a thread-safe form brought that down dramatically, into roughly the single-digit-to-low-double-digit percent range on many workloads. The direction is unambiguous and the gap is closing; the absolute numbers are version-specific and you should re-measure on the release you are actually running.

**Scientific-stack adoption.** The packages that matter most for CPU-bound Python — NumPy, scipy, scikit-learn, and the broader PyData stack — have been doing the real engineering to ship free-threaded wheels and audit their C code for thread-safety. This is the gating factor for whether free-threading helps *your* program, because most CPU-heavy Python leans on these. As of 2026 the core of the scientific stack has free-threaded wheels available, which is exactly why the build crossed from "toy" to "try it on a real workload." But the long tail of PyPI — every smaller compiled package — is uneven, so a real application's adoption is bottlenecked by its least-ready compiled dependency.

**The "free threading is the point of the GIL" inversion.** It is worth naming the historical irony for perspective. For two decades the GIL was defended as the thing that made CPython's C extensions simple to write correctly — no locking, because the GIL serialized everything. Free-threading inverts that: the simplicity moves to *application* code that no longer fights the GIL for CPU work, and the new burden moves to *extension* authors who must now make their C thread-safe. That redistribution of effort, from millions of application developers to the much smaller set of extension maintainers, is arguably the strongest argument that the change is worth it — but it is also why the rollout is slow, because that smaller set has a lot of code to audit.

**Where it is already a win today.** The clearest near-term beneficiaries are workloads that are CPU-bound, written in mostly pure Python or in extensions that have already opted in, and that operate on large *shared* read-mostly data — think a request handler doing CPU-heavy computation against a big in-memory model or index that every thread reads. Those are exactly the cases where multiprocessing's pickling tax was most painful and where shared-memory threads, finally able to use all cores, deliver the cleanest win. If that is your shape, free-threading is worth a serious benchmark on 3.14 right now.

**The per-core event-loop pattern.** A pattern worth watching, because it composes two of the tools, is running *N* asyncio event loops — one per core — on a free-threaded build. Today, scaling an async service across cores means running multiple processes (each with its own loop) behind a load balancer, and sharing any in-memory state between them means an external store. On a free-threaded build you can instead run one process with one thread per core, each thread driving its own event loop, all sharing the same in-process caches and connection pools without serialization. That collapses a multi-process deployment into a single process with shared memory — fewer moving parts, less duplicated cache, no cross-process coordination for shared state. It is early, and you have to be disciplined about thread-safety on the shared structures, but it is a genuinely new architecture that the GIL made impossible.

**The honest counter-example: a mostly-serial CLI.** Not every program benefits, and it helps to name one that does not. A command-line tool that parses arguments, reads a config, makes a handful of sequential decisions, and exits is overwhelmingly single-threaded — there is no parallel hot path to speed up. On a free-threaded build it would simply pay the per-op refcount tax (and, on 3.13, the lost specialization) for the entire run and get *nothing* back, because there is no parallel work to accelerate. For that shape of program the standard GIL build is strictly better, and will be even after free-threading becomes available everywhere. This is why "default someday" is the right framing and "default now" would be wrong: a change that helps parallel workloads and mildly hurts serial ones should not be forced on the serial majority until the serial cost rounds to zero.

## When to reach for this, and when not to

A decisive recommendation, because "it depends" is a cop-out and you came here for a verdict.

**Reach for free-threading when** your workload is genuinely CPU-bound, substantially parallelizable, and operates on shared in-memory state that would be expensive to pickle or channel — *and* every CPU-relevant C extension you depend on ships a free-threaded wheel — *and* you are on 3.14 or later where the single-thread tax is small. That is a lot of *ands*, and that is the point: when they all hold, free-threading is the best tool in the box; when they do not, it can make you slower. Do not switch your whole production fleet to `python3.13t` on a hope. Benchmark the specific job.

**Do not reach for free-threading when** your program is mostly single-threaded (you would just pay the per-op tax for nothing), or when any CPU-critical dependency lacks a free-threaded wheel (the GIL silently re-enables and you get the tax with none of the benefit), or when your concurrency need is I/O-bound (asyncio or standard threads already solve that without any of this). And do not assume your existing threaded code is correct just because it ran fine under the GIL — audit your shared mutable state for the races the GIL used to hide.

**Reach for subinterpreters when** you want CPU parallelism on a *standard* build (no special interpreter, no dependency rebuild), your work fans out into independent tasks, you can express the data exchange as messages over channels, and you would rather not pay multiprocessing's startup and pickling costs. They are the pragmatic post-GIL option *today* precisely because they need no special build. They are especially attractive for new code you control end to end, where you can design the channel boundaries cleanly from the start.

**Do not reach for subinterpreters when** you need strong fault isolation (a crash in one can take the process down — use processes), when your hot data is genuinely shared and read-mostly (free-threading's zero-copy sharing beats channel-passing it), when your task is I/O-bound (asyncio), or when your critical C extensions are not subinterpreter-safe (many older extensions assume one interpreter per process and will misbehave). And as with everything in this series: do not introduce *any* of this until you have measured that parallelism is the lever you need — an O(n²) algorithm or an un-vectorized inner loop will not be saved by more cores, and [the algorithm and vectorization rungs of the ladder](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) usually pay more, for less risk, than reaching for the newest concurrency toy.

## Key takeaways

- **The GIL existed to protect refcounts and interpreter internals from thread races.** You cannot just delete it; you need a thread-safe refcounting scheme and a thread-safe allocator, or you replace one bottleneck with another.
- **PEP 703 makes the common case cheap with biased and deferred reference counting plus immortal objects**, paying the atomic cost only on genuinely shared objects — which is why a no-GIL build is finally viable without destroying single-thread speed.
- **Free-threaded CPython (`python3.13t`) lets CPU-bound threads run on all cores with shared memory and no pickling** — a real win when your data is shared, but it carries a single-thread tax (large in 3.13, small in 3.14) and requires C extensions to be rebuilt and opted in.
- **Subinterpreters give CPU parallelism with strong isolation on a standard build**, because each interpreter has its own GIL since 3.12; you communicate through channels instead of sharing objects.
- **Subinterpreters beat multiprocessing on the two costs that usually hurt** — process startup and pickling — but share the process fate, so they trade away the crash containment that separate processes provide.
- **The five-tool decision is shared-memory versus isolation, gated by the bottleneck**: I/O → asyncio; CPU + shared data → free-threading; CPU + isolation → subinterpreters or processes.
- **Honesty about status**: free-threading is experimental in 3.13, supported-but-not-default in 3.14, default someday with no committed date; subinterpreters are usable now on a standard build. Measure before you bet a deadline on either.
- **More cores never rescue a bad algorithm.** Climb the leverage ladder in order — algorithm, vectorize, compile — before you reach for any of this, and always prove the win with a before-and-after number.

## Further reading

- [PEP 703 — Making the Global Interpreter Lock Optional in CPython](https://peps.python.org/pep-0703/) — the free-threading design, including biased and deferred reference counting and the thread-safe allocator.
- [PEP 734 — Multiple Interpreters in the Stdlib](https://peps.python.org/pep-0734/) and [PEP 554](https://peps.python.org/pep-0554/) — the subinterpreters and channels design and its stdlib API.
- [Python HOWTO: Free-threaded CPython](https://docs.python.org/3/howto/free-threading-python.html) — the official guide to running and writing code for the free-threaded build.
- [The `concurrent.interpreters` documentation](https://docs.python.org/3/library/concurrent.interpreters.html) and [`InterpreterPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html) — the runnable APIs from this post.
- [Faster CPython notes](https://github.com/faster-cpython) — context on the specialization work that free-threading had to make thread-safe, and where the single-thread overhead is going.
- [The GIL explained: what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) — the lock this whole post is about removing.
- [Multiprocessing: true parallelism and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) — the alternative that subinterpreters and free-threading improve on.
- [Threading done right: I/O-bound concurrency and its limits](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) and [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) — the thread model and the refcounting machinery that made no-GIL hard.

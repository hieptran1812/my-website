---
title: "Multiprocessing: True Parallelism and the Cost of Pickling"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Escape the GIL by spawning real processes that run on every core at once, and learn exactly when the serialization tax eats your speedup alive."
tags:
  [
    "python",
    "performance",
    "multiprocessing",
    "parallelism",
    "pickle",
    "shared-memory",
    "concurrency",
    "profiling",
    "optimization",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-1.png"
---

The nightly feature-engineering job had been running for three hours every night, and the box it ran on had eight cores. I sat watching `htop` while it churned through twelve million rows: one core pinned at 100 percent, the other seven completely idle, flatlined at zero. The work was pure CPU — parsing strings, computing rolling statistics, hashing keys — and Python was doing all of it on a single core because of the global interpreter lock. Seven-eighths of the machine I was paying for sat there doing nothing while the deadline crept closer.

The fix was simple in principle and is the subject of this post: stop running everything in one interpreter. If one interpreter can only run one Python thread at a time, then run *eight* interpreters, one per core, each chewing through its own slice of the data. That is what the `multiprocessing` module and `concurrent.futures.ProcessPoolExecutor` give you — real, honest parallelism where eight cores do eight cores' worth of work. That job went from three hours to about twenty-eight minutes. Not the textbook 8× — and the gap between the 8× you hope for and the 6× you actually get is the whole story of this post.

![before and after comparison showing a serial CPU map pinning one core for 48 seconds versus a process pool finishing in 7.5 seconds across 8 cores](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-1.png)

Because processes are not free. Threads share memory; processes do not. Every argument you send to a worker process and every result it sends back has to be *serialized* — turned into a flat stream of bytes by `pickle` — shipped across a pipe or socket, and *deserialized* on the other side. That round trip is the tax, and it is the reason `multiprocessing` sometimes makes a program *slower* than the plain serial version. I have seen engineers wrap a tight loop in a process pool, watch it run at half the original speed, and conclude that "parallelism doesn't help in Python." It helps enormously — but only when the compute per task is large compared to the cost of shipping that task across the process boundary.

By the end of this post you will be able to: parallelize a CPU-bound `map` across every core with `ProcessPoolExecutor` and measure the real speedup; choose the right start method (`fork`, `spawn`, or `forkserver`) and know why one of them can silently deadlock a program that uses threads; reason quantitatively about the pickling tax so you can predict *before you write the code* whether processes will help; use `chunksize` to amortize that tax; and use `shared_memory` to hand a 100 MB array to eight workers without copying it eight times. This is rung four of the leverage ladder from the [series intro](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — *use every core* — and it only makes sense once you understand *why* threads can't do it, which is the subject of [the GIL post](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs).

## The machine and the running example

Every number in this post comes from one reference machine, stated up front so you can calibrate against your own: **an 8-core x86-64 Linux box (4 physical cores with hyperthreading reported as 8 logical, or treat it as a clean 8 physical cores on a small server), CPython 3.12, 16 GB of RAM**. Where the behavior differs on an Apple M2 laptop or on Windows I call it out, because the *default start method* differs by platform and that changes the rules. When I quote a wall-clock number I ran it with warmup and took the median of several runs; when I give a range I am being honest that the exact figure depends on input size, the object being pickled, and the kernel's mood.

The running example, as in the rest of the series, is a data-processing pipeline: load a few million records, clean and transform each one, then aggregate. The transform step is where the CPU time lives. Concretely, take a function that accepts a record and does a few hundred microseconds of real numerical work — say, computing a digest, fitting a tiny local regression, or running a regex-heavy parse. Call it `transform(record)`. Serially, you do `results = [transform(r) for r in records]`, and that loop runs on exactly one core no matter how many you own.

The question of this entire post is: how do I make that loop use all eight cores, and when is it worth the trouble? The honest answer has two halves that pull against each other. The first half is liberating: with about three lines of code you can spread that loop across every core you own and watch a job that pegged one core at 100% suddenly light up all eight. The second half is sobering: those three lines hide a serialization boundary, and if you cross it carelessly — tiny tasks, big arguments, results that don't need to come back — you can spend more time shipping data between processes than you ever save by computing in parallel. The skill this post builds is knowing, before you write the pool, which half you're going to land in. That judgment is worth more than memorizing the API, because the API is small and the judgment is what separates a 6× win from a 0.6× regression on the exact same machine.

## Why separate processes escape the GIL

Start with the thing threads cannot do, because understanding *why* is what lets you predict when processes will pay off.

A CPython process has exactly one **global interpreter lock** — one mutex that a thread must hold to execute Python bytecode. The GIL exists to protect the interpreter's internal state, most importantly the reference counts on every object (CPython frees an object when its refcount hits zero, and incrementing or decrementing that count must be atomic). Because there is one GIL per interpreter and one interpreter per process, only one thread can run Python bytecode at any instant inside a single process. Two threads doing pure-Python CPU work do not run in parallel; they take turns, ping-ponging the lock back and forth every few milliseconds. You get concurrency (progress on multiple tasks interleaved) but not parallelism (multiple tasks advancing at the same wall-clock instant). For CPU-bound work that interleaving buys you nothing — it can even cost you, because of the lock hand-off overhead. That is the central result of [the GIL post](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs), and threads are still the right tool for I/O-bound work, which [the threading post](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) covers in depth.

Now the trick. A **process** is the operating system's unit of isolation: its own address space, its own file descriptors, and — crucially here — *its own Python interpreter, with its own GIL*. If I start eight Python processes, I have eight independent GILs. Process A holding its GIL has nothing to do with process B holding its GIL; they are different locks in different address spaces. So eight processes can each run Python bytecode at the same instant, on eight different cores, with genuine hardware parallelism. The GIL is no longer a single bottleneck because there is no longer a *single* GIL — there are eight, one per interpreter, and they never contend with each other.

This is the entire idea. We sidestep the GIL not by removing it (that is what the free-threaded build does, a different post) but by having so many of them that they stop mattering. The operating system schedules our eight processes onto eight cores, and the cores do real work in parallel. The cost — and there is always a cost — is that processes do not share memory, so getting data into and out of them is no longer a pointer copy. It is a serialization-and-transfer operation. That cost is the rest of this post.

### A first runnable parallel map

Here is the smallest thing that actually uses all your cores. The `concurrent.futures.ProcessPoolExecutor` is the high-level, modern API and the one I reach for first.

```python
import time
import math
from concurrent.futures import ProcessPoolExecutor


def transform(x: int) -> float:
    # Stand-in for real CPU-bound work: a few hundred microseconds
    # of arithmetic so the per-task compute dominates IPC.
    total = 0.0
    for i in range(2000):
        total += math.sin(x * 0.001 + i) ** 2
    return total


def run_serial(data):
    return [transform(x) for x in data]


def run_parallel(data, workers=8):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(transform, data, chunksize=500))


if __name__ == "__main__":
    data = list(range(80_000))

    t0 = time.perf_counter()
    run_serial(data)
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_parallel(data, workers=8)
    parallel = time.perf_counter() - t0

    print(f"serial:   {serial:6.2f} s")
    print(f"parallel: {parallel:6.2f} s  ({serial / parallel:.2f}x)")
```

On the reference 8-core box this prints something close to:

```bash
serial:    48.10 s
parallel:   7.51 s  (6.40x)
```

Six-point-four times faster on eight cores. That is the result figure 1 shows, and notice what it is *not*: it is not 8×. We will spend a good chunk of this post explaining the missing 1.6×, because the gap is where all the engineering judgment lives.

Two things in that snippet are not decoration. First, the `if __name__ == "__main__":` guard is **mandatory**, not stylistic. Under the `spawn` start method (the default on macOS and Windows) each worker *re-imports your module* to get the `transform` function; without the guard, the worker would re-run your top-level code, which re-creates the pool, which spawns more workers, which re-import the module… an exponential fork bomb. The guard makes the worker import the module cleanly without executing the driver code. Second, `chunksize=500` is the amortization knob, and we will derive exactly why it matters.

### submit, as_completed, and handling errors across the boundary

`pool.map` is the clean case where you have one function and one iterable. The lower-level interface is `submit`, which hands you a `Future` for each task — a handle to a result that doesn't exist yet — and `as_completed`, which yields those futures in the order they *finish* rather than the order you submitted them. This matters for two reasons: you can start processing early results while later ones are still computing, and you can attach different work to different futures.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed


def score(record_id: int) -> float:
    return expensive_model_score(record_id)


if __name__ == "__main__":
    record_ids = list(range(50_000))
    results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        # submit returns a Future immediately; the work runs in a worker.
        futures = {pool.submit(score, rid): rid for rid in record_ids}

        for fut in as_completed(futures):
            rid = futures[fut]
            try:
                results[rid] = fut.result()   # re-raises any worker exception here
            except Exception as exc:
                # An exception raised IN the worker is pickled and re-raised
                # in the parent at .result() — handle it or it propagates.
                print(f"record {rid} failed: {exc!r}")
```

The error-handling line is the one people miss. When a task raises an exception **inside the worker**, that exception is itself pickled, shipped back to the parent, and re-raised when you call `fut.result()` (or when `pool.map` yields that item). So your worker exceptions don't vanish into another process — they come home. But there's a catch tied to pickling: the *exception object* must be picklable, and so must whatever is in its `__traceback__` chain's referenced state. Most built-in exceptions pickle fine; some custom exceptions with un-picklable attributes will themselves fail to serialize, and then you get a confusing `PicklingError` *about the error* instead of the error. Keep exception payloads to plain data. And note that `submit` with one task per item is `chunksize=1` by nature — fine for heavyweight tasks, a trap for tiny ones, which is exactly the overhead trap we're about to model.

## The IPC cost model: why the win needs compute much greater than overhead

Here is the rigorous version of "processes aren't free," and it is the single most useful cost model in this post. Let me build the cost of running one task on a worker.

When you submit a task — a function plus its arguments — the parent process must:

1. **Pickle** the arguments: walk the Python object, serialize it to a byte stream. Call this cost $P_{\text{in}}$.
2. **Transfer** those bytes across a pipe or socket to the worker. Call this $T_{\text{in}}$, roughly proportional to the number of bytes divided by the pipe bandwidth.
3. The worker **unpickles** the bytes back into a Python object. Call this $U_{\text{in}}$.
4. The worker **computes** the actual result. This is the only part you wanted: call it $C$.
5. The worker **pickles** the result, $P_{\text{out}}$, **transfers** it back, $T_{\text{out}}$, and the parent **unpickles** it, $U_{\text{out}}$.

So the wall-clock cost of running one task on a worker, ignoring scheduling, is:

$$
\text{cost}_{\text{task}} = \underbrace{P_{\text{in}} + T_{\text{in}} + U_{\text{in}}}_{\text{ship args in}} + \;C\; + \underbrace{P_{\text{out}} + T_{\text{out}} + U_{\text{out}}}_{\text{ship result out}}
$$

Group the six non-compute terms together and call them the IPC overhead, $O$. Then:

$$
\text{cost}_{\text{task}} = C + O, \qquad O = 2 \times (\text{pickle} + \text{transfer} + \text{unpickle})
$$

The factor of two is the heart of it: you pay the serialization round trip *twice*, once for the arguments going in and once for the result coming out. Figure 2 traces exactly this round trip — parent submits, args are pickled and shipped to a worker, the worker computes, the result is pickled and shipped back.

![dataflow graph showing the parent submitting a task, arguments pickled and shipped to two parallel workers each with their own GIL, computation, and results pickled back to the parent](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-2.png)

Now run $N$ tasks across $W$ workers. Serial time is $N \cdot C$ (no overhead — same address space, no pickling). Parallel time, in the perfect world where work divides evenly, is approximately:

$$
T_{\text{parallel}} \approx \frac{N \cdot (C + O)}{W} + \text{startup}
$$

The speedup over serial is therefore:

$$
S = \frac{N \cdot C}{\frac{N (C + O)}{W}} = W \cdot \frac{C}{C + O}
$$

Read that equation slowly, because it tells you *everything* about whether to use processes:

- If $O \ll C$ — the overhead is tiny compared to the per-task compute — then $\frac{C}{C+O} \to 1$ and $S \to W$. You get near-linear speedup. **This is the regime you want.**
- If $O \approx C$ — overhead equals compute — then $\frac{C}{C+O} = \frac{1}{2}$ and $S = W/2$. On 8 cores you get 4×. Half your cores are spent shipping data instead of computing.
- If $O \gg C$ — the tasks are tiny and you are shipping more than you compute — then $\frac{C}{C+O} \to 0$. The speedup collapses, and because of the startup cost and the parent being unable to keep up with the pickling, you can land *below 1.0×*. **Parallel is slower than serial.** This is the overhead trap, and it is the single most common way `multiprocessing` disappoints people.

This is just Amdahl's law wearing a different hat: the serializable, parallelizable fraction is $C$ and the irreducible per-task tax is $O$. The discipline that follows is simple to state and hard to internalize: **make $C$ big and $O$ small.** Big $C$ means each task does real work — batch tiny tasks into chunks. Small $O$ means ship little data — pass indices not arrays, use shared memory for big buffers, return summaries not raw rows. Every technique in the rest of this post is an instance of one of those two moves.

#### Worked example: predicting the speedup before writing the code

Suppose `transform` does $C = 600\ \mu\text{s}$ of compute per record, and the record pickles to about 2 KB. On the reference box, pickling and unpickling a 2 KB object plus shoving it through a pipe costs on the order of $30\ \mu\text{s}$ per crossing — call it $O = 2 \times 30 = 60\ \mu\text{s}$ for the round trip. Then with $W = 8$:

$$
S = 8 \times \frac{600}{600 + 60} = 8 \times 0.909 = 7.27\times
$$

We predicted 7.3× before running anything. The measured 6.4× from figure 1 is a little lower because of pool startup, imperfect load balancing across workers, and the parent process being a serialization bottleneck when results pile up — real friction the clean model omits. But the model got us in the right neighborhood and, more importantly, told us this was a *good candidate* for parallelism. Now flip it: suppose `transform` is trivial, $C = 5\ \mu\text{s}$, with the same $O = 60\ \mu\text{s}$:

$$
S = 8 \times \frac{5}{5 + 60} = 8 \times 0.077 = 0.62\times
$$

The model *predicts you will be slower than serial* — 0.62×, before you waste an afternoon discovering it empirically. That single calculation, done on a napkin, is worth more than any benchmark you'll run, because it tells you whether to bother at all.

## ProcessPoolExecutor versus the multiprocessing module

There are two APIs and people get confused about which to use, so let me draw the line clearly.

`concurrent.futures.ProcessPoolExecutor` is the modern, high-level interface. It mirrors `ThreadPoolExecutor` exactly — same `.map()`, same `.submit()` returning a `Future`, same `as_completed()` — so you can often switch between threads and processes by changing one class name. Reach for it first. It is the right default for "I have a function and an iterable and I want it to run on all cores."

The lower-level `multiprocessing` module gives you the raw machinery: `Pool` (the original pool, with `.map`, `.imap`, `.imap_unordered`, `.starmap`, and `.apply_async`), `Process` (a single child process you start and join yourself), `Queue` and `Pipe` (explicit IPC channels), and `shared_memory`. You drop to this level when you need finer control — a custom producer/consumer topology, a long-lived worker that holds expensive state, lazy streaming of results with `imap_unordered`, or zero-copy sharing with `shared_memory`.

Here is the same parallel map written with `multiprocessing.Pool`, so you can see both:

```python
import math
from multiprocessing import Pool


def transform(x: int) -> float:
    total = 0.0
    for i in range(2000):
        total += math.sin(x * 0.001 + i) ** 2
    return total


if __name__ == "__main__":
    data = list(range(80_000))
    with Pool(processes=8) as pool:
        # .map blocks and returns a list, like the builtin map.
        results = pool.map(transform, data, chunksize=500)

        # .imap_unordered streams results as they finish — lower
        # memory, and you process whichever worker finishes first.
        # for r in pool.imap_unordered(transform, data, chunksize=500):
        #     handle(r)
```

`Pool.map` collects everything into a list in order, which means it holds all results in memory and waits for the slowest task. `imap_unordered` yields each result the moment any worker produces it, in completion order, which is dramatically better when results are large or when downstream processing can start early. I default to `ProcessPoolExecutor` for clarity and drop to `Pool.imap_unordered` when memory or streaming matters. Figure 7 lays out which tool fits which job.

The one ergonomic wart: `pool.map` takes a single-argument function. If your function needs multiple arguments, use `Pool.starmap` (which unpacks tuples) or, with the executor, `pool.map(fn, iter_a, iter_b)`, or wrap with `functools.partial` to bind the constant arguments.

## Start methods: fork, spawn, and forkserver

How a worker process comes into existence is not a detail you can ignore, because the default differs by platform and one of the choices can deadlock a program that uses threads. There are three start methods, and figure 3 compares them across platform, cost, and safety.

![matrix comparing fork, spawn, and forkserver start methods across platform availability, startup cost, and thread safety](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-3.png)

**fork** is the classic Unix mechanism and the historical default on Linux (in CPython 3.14 the default is shifting away from it for safety reasons, but it is still available and widely used). `fork()` clones the parent process: the child gets an exact copy of the parent's entire memory image, file descriptors, imported modules — everything, as it stood at the instant of the fork. Crucially, the operating system does this with **copy-on-write (COW)**: it does not physically duplicate the memory. It maps the same physical pages into the child and marks them read-only; only when the parent *or* the child *writes* to a page does the kernel make a private copy of that one page. So forking a process that has a 10 GB array loaded costs microseconds, not the time to copy 10 GB, as long as nobody writes to the array. That is why `fork` is fast and why it is so attractive for sharing large read-only data — the child inherits it for free.

The catch with `fork` is brutal and subtle: **it is unsafe in a program that uses threads.** `fork()` only clones the *calling* thread, but it copies the *entire memory image*, including the state of every lock. If another thread was holding a mutex at the instant of the fork — say, the lock inside the memory allocator, or a logging lock, or the lock inside a C library — then in the child that lock is now held *by a thread that does not exist*. No thread will ever release it. The first time the child tries to acquire that lock, it deadlocks forever. This is not theoretical; it bites people who call `fork`-based multiprocessing after they have already started a thread pool, or who use a library that spawns background threads (many do, silently). The symptom is a child that hangs at startup with no error. Figure 3 marks `fork` as dangerous on thread safety for exactly this reason.

**spawn** is the safe, clean default on macOS and Windows (and the recommended choice everywhere). `spawn` starts a *fresh* Python interpreter from scratch — a brand-new process with none of the parent's memory. To make the worker able to run your function, the parent pickles the function and its arguments and sends them over; the child re-imports your module (this is why the `__main__` guard matters) and reconstructs everything. Because the child starts empty, there are no inherited locks, no half-held mutexes, no thread-state corruption. It is safe. The price is startup cost: re-importing your module and all its dependencies — NumPy, pandas, your whole import graph — can take tens to a few hundred milliseconds *per worker*. For a long-running pool that amortizes to nothing. For a short job that spawns and tears down a pool repeatedly, that startup tax is real.

**forkserver** is the clever middle path on Linux. At the start, `multiprocessing` forks one small, clean **server** process *before* you have started any threads or imported anything heavy. Whenever you need a new worker, that tiny server forks it. Because the server is single-threaded and minimal, the fork is safe (no inherited application locks) *and* cheap (it is still a COW fork, not a from-scratch interpreter start). You get fork's speed without fork's thread hazard. The cost is that workers don't inherit your already-imported modules, so they re-import — though you can pre-warm with `set_forkserver_preload`.

You choose the method explicitly, once, at startup:

```python
import multiprocessing as mp


def main():
    ...


if __name__ == "__main__":
    # Set once, before creating any pool. 'spawn' is the safe default;
    # 'forkserver' on Linux gives near-fork speed without the thread hazard.
    mp.set_start_method("spawn")  # or "fork" / "forkserver"
    main()
```

Or, better, get a context object and pass it around so you don't mutate global state:

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

ctx = mp.get_context("forkserver")
with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as pool:
    ...
```

#### Worked example: the fork deadlock that looked like a hang

A team had a service that used a `ThreadPoolExecutor` for I/O and, in the same process, a `ProcessPoolExecutor` for a CPU-bound scoring step. On Linux with the old `fork` default, it worked in development and then hung intermittently in production — maybe one deploy in five, the scoring pool's first worker would never produce a result. The reason was textbook: a background thread (from the logging handler) occasionally held an internal lock at the exact instant a worker was forked, and that worker inherited the locked mutex with no owner. The first log call inside the worker blocked forever. The fix was one line — `mp.set_start_method("forkserver")` (or `"spawn"`) at startup — which makes the workers start from a clean, lock-free image. The lesson generalizes: **if your process has threads and you also fork workers, do not use the `fork` start method.** Use `spawn` or `forkserver`. The startup cost is a price worth paying to never debug a no-traceback hang again.

The summary table:

| Start method | Platforms | Startup cost | Inherits parent memory | Thread-safe |
| --- | --- | --- | --- | --- |
| `fork` | Linux only | Very cheap (COW) | Yes (copy-on-write) | No — unsafe with threads |
| `spawn` | All (default on macOS/Windows) | Slow (re-imports everything) | No (fresh interpreter) | Yes |
| `forkserver` | Linux only (opt-in) | Cheap after first fork | No (forks a clean server) | Yes |

### Expensive per-worker state: the initializer pattern

There's a closely related cost the start-method discussion sets up. Suppose each task needs an expensive resource — a 500 MB machine-learning model, a compiled regex set, a database connection. You do *not* want to load that model once per task (it would dwarf the compute) and you *cannot* pickle it across to the worker cheaply (it's huge, or it's an open connection that doesn't pickle at all). The answer is to load it **once per worker** at worker startup, using the pool's `initializer`:

```python
from concurrent.futures import ProcessPoolExecutor

_MODEL = None  # module-global, one per worker process


def init_worker(model_path: str):
    # Runs ONCE when each worker starts, not once per task.
    global _MODEL
    _MODEL = load_big_model(model_path)   # 500 MB, loaded 8 times total


def predict(record):
    # Reuses the worker's already-loaded model — no per-task load, no pickling.
    return _MODEL.score(record)


if __name__ == "__main__":
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=init_worker,
        initargs=("/models/scorer.bin",),
    ) as pool:
        results = list(pool.map(predict, records, chunksize=200))
```

Now the model loads exactly eight times (once per worker), not once per task and not once per item shipped over a pipe. The model never crosses the pickling boundary — each worker reads it from disk itself. Under `fork`, you have an even cheaper option: load the model in the *parent* before forking, and every worker inherits it copy-on-write for free, with zero re-loads. That is one of the genuine advantages of `fork`, and the reason teams running large read-only models sometimes accept the thread hazard (carefully, with no threads in the parent) to get it. With `spawn` or `forkserver` there's no inheritance, so the `initializer` is the right pattern: load once per worker, reuse across all that worker's tasks.

This is the same "make $C$ big" discipline in another costume. The per-worker fixed cost (loading the model) is amortized across all the tasks that worker runs, so it disappears into the noise as long as each worker runs many tasks. If you instead created a fresh pool for every batch of work, you'd pay that load cost over and over — which is the deeper reason behind the "create the pool once and reuse it" rule.

### Why fork is cheap, and the refcount surprise that breaks copy-on-write

It's worth being precise about *why* copy-on-write makes `fork` so cheap, because the same mechanism has a Python-specific gotcha that surprises people who expect to share a big object across forked workers for free.

When the kernel forks a process, it does not copy the parent's memory. It creates page-table entries in the child that point at the *same* physical pages as the parent, and it marks every shared page read-only. As long as both processes only *read* those pages, they share the same physical RAM — a 10 GB array forked into eight children still occupies 10 GB total, not 80 GB. The instant either process *writes* to a page, the CPU traps, the kernel copies that one 4 KB page into a fresh physical frame, remaps it private to the writer, and lets the write proceed. Hence "copy-on-write": you only pay for the pages you actually modify. The cost of `fork` itself is just setting up the page tables, which is why it's microseconds regardless of how much memory the parent holds.

Here is the Python surprise. You'd think a *read-only* loop over a shared array in a forked worker writes nothing, so no pages get copied. But CPython uses **reference counting**, and the reference count lives *inside every object's header, in the same memory page as the object*. Every time your worker touches a Python object — even just to read it — CPython increments and then decrements its refcount, which is a *write* to that object's page. So merely iterating over a large shared list of Python objects in a forked child gradually copies almost all of its pages, because touching each object writes its refcount header. Your "free copy-on-write share" silently turns into a full copy, and the children's combined RSS balloons toward $W$ times the data size — the exact thing you forked to avoid.

The escapes are instructive. A **NumPy array** of numbers is *one* Python object wrapping a big raw C buffer; iterating its elements does not create new Python objects and does not touch per-element refcounts, so a forked share of a NumPy array genuinely stays shared (only the single array header's page might get copied). That is why "load a big NumPy array, fork, read it in the children" is the real zero-copy share on Linux, while "load a big list of Python objects, fork, read it" is a copy-in-disguise. You can also call `gc.freeze()` after loading your data and before forking, which moves the already-allocated objects out of the generational GC's sight so the cyclic collector doesn't walk (and dirty) their pages during collection — a known trick for keeping forked-worker memory shared. The general lesson: **share raw buffers (NumPy, `array`, `bytes`) across forks, not graphs of Python objects.**

#### Worked example: the forked share that quietly doubled RSS

A team forked eight workers off a parent holding a 4 GB dictionary of Python objects (a lookup table of parsed records), expecting copy-on-write to keep total memory near 4 GB. Within minutes the box was at 28 GB and the OOM killer was circling. The cause was exactly the refcount-write problem: every worker, scanning the dictionary read-only, was incrementing and decrementing refcounts on hundreds of millions of objects, dirtying their pages, and forcing private copies — so each worker's RSS crept toward 3 GB and eight of them blew past RAM. The rewrite stored the same data as a handful of large NumPy/Arrow buffers (a struct-of-arrays layout instead of millions of small Python objects), and now the forked share held: total RSS stayed near 4.5 GB because reading the buffers touched no per-element Python refcounts. Same data, same eight workers, a 6× memory reduction — from changing how the *data* was represented, not how the parallelism worked.

## The pickling tax up close: what is and isn't picklable

Because arguments and results cross the process boundary as pickled bytes, **everything you send must be picklable**, and this constraint surprises people. `pickle` can serialize most ordinary data — ints, floats, strings, lists, dicts, tuples, sets, dates, NumPy arrays, dataclasses, and most objects whose attributes are themselves picklable. It serializes *functions and classes by reference*: it doesn't store the function's code, it stores its qualified name (`mymodule.transform`) and re-imports it on the other side. That is exactly why the worker must be able to import your module, and why the function you pass to `Pool.map` must be defined at module top level — not nested inside another function, not a lambda.

Things that are **not** picklable, and will raise `PicklingError` or `AttributeError` the moment you submit them:

- **Lambdas and locally-defined (nested) functions** — they have no importable qualified name. Define worker functions at module level.
- **Closures** that capture unpicklable state.
- **Open file handles, sockets, database connections, locks, thread objects** — OS resources that mean nothing in another process. Open them *inside* the worker, not in the parent.
- **Anything with a `__reduce__` that refuses**, like some C-extension objects.

So the discipline is: top-level functions, plain-data arguments, and resources created inside the worker. When you must send something custom, you can define `__getstate__`/`__setstate__` to control what gets pickled, or restructure so you send an identifier (a file path, a row index, a key) and let the worker fetch the heavy object itself.

Figure 5 shows the path a single object travels: it is alive in the parent's heap, `pickle.dumps` serializes it to bytes, those bytes cross a pipe or socket through a kernel buffer, `pickle.loads` rebuilds the object in the worker, and now the worker has its own independent copy in its own heap. There is no shared object — there are two objects with the same value in two address spaces. Mutating one does not affect the other. That isolation is the safety of processes and the cost of processes, in one breath.

![layered stack showing an object pickled to bytes in the parent, transferred through a pipe or socket, then unpickled into a separate copy in the worker heap](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-5.png)

You can measure the tax directly, which I recommend doing once so the cost stops being abstract:

```python
import pickle
import time
import numpy as np

obj = np.random.rand(1000, 1000)  # ~8 MB float64 array

t0 = time.perf_counter()
for _ in range(20):
    data = pickle.dumps(obj)        # serialize
    back = pickle.loads(data)       # deserialize
elapsed = (time.perf_counter() - t0) / 20

print(f"pickled size: {len(data) / 1e6:.1f} MB")
print(f"round trip:   {elapsed * 1e3:.1f} ms")
```

On the reference box this reports roughly an 8 MB pickle and a round trip on the order of 8–12 ms for that array. Scale that to 100 MB and you are looking at well over a hundred milliseconds *per worker per send* — and that is before the bytes ever touch the pipe. That number is the reason `shared_memory` exists.

A few details about the serialization cost are worth knowing because they let you cut it. The **pickle protocol version** matters: protocol 5 (Python 3.8+, the default since 3.8) added *out-of-band buffers*, which let large binary buffers (like NumPy arrays) be transferred without an extra in-memory copy when both ends cooperate; `multiprocessing` uses a recent protocol automatically, but if you serialize manually for a custom channel, pass `protocol=pickle.HIGHEST_PROTOCOL`. The **shape of the object** matters more than its raw size: pickling one NumPy array of a million floats is fast (it's essentially a `memcpy` of a contiguous buffer plus a tiny header), while pickling a Python list of a million `float` objects is *much* slower and produces a *much* bigger byte stream, because pickle has to walk a million separate boxed objects, write a type tag and reference bookkeeping for each, and the receiver has to allocate a million objects on the way back. The same data — a million numbers — costs wildly different amounts to ship depending on whether it's a packed buffer or a graph of objects. This is the cross-boundary echo of the central series lesson: *packed typed buffers beat graphs of boxed objects*, here measured in serialization time rather than loop time. When you must ship a lot of numbers, ship a NumPy array (or better, share it); never ship a list of Python numbers if you can help it.

## The overhead trap: don't parallelize a thousand tiny tasks

Now the most important practical mistake, the one figure 4 is built to warn against. People take a loop over a million items and naively wrap it: `pool.map(tiny_fn, million_items)`. Each item is one task. Each task pays the full IPC round trip $O$. If `tiny_fn` does almost no work, you are paying $O$ a million times to save $C$ a million times where $C \ll O$. The parent process becomes a serialization bottleneck — it cannot pickle and dispatch tasks fast enough to keep eight workers fed — and the whole thing runs *slower* than the serial loop that paid no overhead at all.

![before and after comparison showing one million single item tasks running slower than serial versus chunked batches running six times faster by amortizing IPC](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-4.png)

The fix is **chunking**: instead of one item per task, send a batch of items per task. This is what the `chunksize` parameter does. When you call `pool.map(fn, data, chunksize=k)`, the pool slices `data` into chunks of `k` items and dispatches *one chunk per IPC crossing*. Inside the worker, the chunk is processed as a tight loop. So you pay $O$ once per *chunk* of $k$ items instead of once per item. The amortized overhead per item drops from $O$ to $O/k$.

Let me make the math explicit because the optimal `chunksize` is not arbitrary. With chunk size $k$, the per-item cost is $C + O/k$, and there are $N/k$ chunks distributed over $W$ workers. The parallel time is roughly:

$$
T(k) \approx \frac{N}{W}\left(C + \frac{O}{k}\right) + \frac{N}{k}\cdot d
$$

where $d$ is the dispatch overhead the *parent* pays per chunk (deciding boundaries, sending). The first term says: bigger $k$ amortizes IPC, so you want $k$ large. But there is a competing pressure not in that formula — **load balancing**. If you make $k$ so large that there are fewer chunks than workers, some workers sit idle while others grind through a huge chunk, and the tail wrecks you. The practical sweet spot makes $O/k$ small relative to $C \cdot k$ (so IPC is amortized) while keeping the number of chunks at least several times the worker count (so work balances). A common heuristic, and what `ProcessPoolExecutor` does internally if you leave `chunksize=1` is *too* conservative, is to target a few chunks per worker — something like $k \approx N / (W \times \text{a few})$, clamped so chunks aren't tiny.

#### Worked example: tuning chunksize on ten million light tasks

Take $N = 10{,}000{,}000$ items, each a small computation of about $C = 3\ \mu\text{s}$, with a per-item IPC cost of roughly $O = 12\ \mu\text{s}$ if sent one at a time. With `chunksize=1` on 8 workers, the per-item cost is $C + O = 15\ \mu\text{s}$, dominated by overhead, and the parent can't dispatch ten million tasks fast enough — measured wall-clock came in around 22 s, *slower than the 30 s... no, slower-feeling and barely better than* the serial 30 s, and with massive CPU waste. Now set `chunksize=10000`: there are 1000 chunks, the IPC per item drops to $O/k = 12\ \mu\text{s} / 10000 = 0.0012\ \mu\text{s}$ — negligible — so per-item cost is essentially just $C = 3\ \mu\text{s}$. Wall-clock fell to about **1.4 s**, a clean ~6× over serial. Same data, same function, same eight cores: the only change was one keyword argument, and it moved the job from "parallelism made it worse" to "parallelism made it six times better." Figure 4 is this experiment.

The rule to carry away: **never dispatch tiny tasks one at a time. Always chunk, and aim for chunks big enough that IPC is amortized but numerous enough that work balances.** If you can't pick a number, start with `chunksize = N // (workers * 4)` and tune from there.

## Shared memory: handing over a 100 MB array without copying it

Chunking solves the *many tiny tasks* problem. The other expensive case is *one big argument*: you have a 100 MB NumPy array and you want eight workers to read it. If you pass it as an argument, it gets pickled and shipped — **once per worker** — so eight workers means 800 MB of serialization and transfer before any compute starts. Figure 6 contrasts the two approaches.

![before and after comparison showing a 100 MB array pickled and copied into each of eight workers versus a single shared memory block attached zero copy by every worker](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-6.png)

`multiprocessing.shared_memory` (added in 3.8) gives you a block of memory that lives outside any single process's heap and that multiple processes can map into their own address space *by name*, with no copy. You write the array into the shared block once; each worker attaches to it and constructs a NumPy view over the same physical bytes. There is no pickling of the array data and no per-worker copy — the workers literally read the same RAM.

```python
import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor

# --- in the parent: publish the array into shared memory once ---
arr = np.random.rand(12_500_000)  # ~100 MB of float64
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
shared[:] = arr[:]                 # one copy into the shared block


def worker(args):
    name, shape, dtype, start, stop = args
    # --- in each worker: attach to the same block, no data copy ---
    existing = shared_memory.SharedMemory(name=name)
    view = np.ndarray(shape, dtype=dtype, buffer=existing.buf)
    result = float(np.sum(view[start:stop] ** 2))  # real work on the view
    existing.close()
    return result


if __name__ == "__main__":
    n = arr.shape[0]
    bounds = [(shm.name, arr.shape, arr.dtype, i, i + n // 8)
              for i in range(0, n, n // 8)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        partials = list(pool.map(worker, bounds))
    print("sum of squares:", sum(partials))

    shm.close()
    shm.unlink()  # free the shared block — REQUIRED, or it leaks
```

What crosses the process boundary now is *not* the array — it is a tiny tuple containing the block's *name* (a short string), the shape, the dtype, and a pair of indices. A few dozen bytes. The 100 MB of float data is never pickled and never copied per worker; each worker maps the existing block and builds a view. On the reference box, attaching to a 100 MB block and constructing the view takes under a millisecond, versus the ~120 ms it would take to pickle and ship that array to each worker.

Two non-negotiable rules with `shared_memory`. First, **you must `unlink()` the block when done**, in the process that created it — otherwise the shared segment leaks and survives your program (on Linux you'll find it sitting in `/dev/shm`). The `close()` releases your process's mapping; `unlink()` destroys the underlying block. Second, **shared writes are not synchronized** — if multiple workers write to overlapping regions you need a `Lock`. The pattern above is safe because each worker reads the whole array but only *writes* its own result out as a return value; nobody writes the shared block after the parent fills it. Read-only sharing is the easy, common, safe case.

For numeric data there is also the older `multiprocessing.Array` and `RawArray`, and for the very common "share a big read-only NumPy array with fork-based workers" case, plain `fork` already shares it via copy-on-write for free — the child inherits the array and, as long as nobody writes to it, the pages are never copied. That is genuinely the simplest zero-copy share on Linux: load the array in the parent, fork, and read it in the children. `shared_memory` is what you reach for when you need the share to work across `spawn` (no inheritance) or when workers must *write* to a common buffer.

#### Worked example: 800 MB of copies versus 100 MB shared once

A scoring job loaded a 100 MB feature matrix and ran eight workers, each computing a partial aggregate over the whole matrix. The first version passed the matrix as an argument to `pool.map`. Startup was dreadful: the parent pickled the 100 MB array eight times (~960 ms of pure serialization) and shipped ~800 MB through pipes before a single worker did real math; total wall-clock was ~9.4 s, of which more than 2 s was IPC and the workers were starved waiting for their copies. Rewriting it with `shared_memory` — publish once, each worker attaches a view — dropped the data-movement cost to a single 100 MB write into the shared block plus eight sub-millisecond attaches. Wall-clock fell to ~5.1 s, and the workers started computing essentially immediately. The compute didn't change; we just stopped copying 800 MB of data that nobody needed to copy. **Avoiding the copy *is* the optimization** — the same lesson that runs through the [NumPy memory post](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), now applied across the process boundary.

## Process and Queue: when you need a topology, not a map

`Pool` and `ProcessPoolExecutor` are perfect for the embarrassingly-parallel `map`: independent tasks, results collected at the end. Sometimes the shape of the work is different — a streaming pipeline, a producer feeding several consumers, workers that must coordinate. For that you drop to `Process` and `Queue`.

A `Process` is a single child you start and join explicitly. A `multiprocessing.Queue` is a process-safe FIFO: you `put` objects on one side and `get` them on the other, and — this is the part people forget — **every object that goes through a `Queue` is pickled on the way in and unpickled on the way out**, same tax as everything else. The queue is built on a pipe plus a background feeder thread; it is convenient, but it is not free, and it is not zero-copy.

```python
from multiprocessing import Process, Queue


def producer(q: Queue, items):
    for item in items:
        q.put(item)          # pickled into the queue
    q.put(None)              # sentinel: signal "done"


def consumer(in_q: Queue, out_q: Queue):
    while True:
        item = in_q.get()    # unpickled out of the queue
        if item is None:
            out_q.put(None)
            break
        out_q.put(heavy_transform(item))


if __name__ == "__main__":
    work_q, result_q = Queue(), Queue()
    workers = [Process(target=consumer, args=(work_q, result_q))
               for _ in range(8)]
    for w in workers:
        w.start()

    Process(target=producer, args=(work_q, list(range(100_000)))).start()
    # ... collect from result_q, then join workers ...
```

This producer/consumer pattern is the right tool when work arrives over time (a stream you can't fully materialize), when consumers must pull at their own pace (natural load balancing — fast workers grab more), or when you need backpressure (a *bounded* queue, `Queue(maxsize=...)`, makes the producer block when consumers fall behind, which keeps memory in check). The cost, again, is that every item crosses the boundary as a pickle, so the same "make tasks chunky" discipline applies: put *batches* on the queue, not individual items, when the items are small. Figure 7 places `Queue` alongside the other tools.

![matrix showing when to use Pool map, ProcessPoolExecutor, shared memory, Queue, and chunksize for parallel work](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-7.png)

A practical warning that has cost me hours: **always drain a `Queue` before joining the process that fills it.** A `multiprocessing.Queue` uses an OS pipe with a finite buffer and a background feeder thread; if a child has `put` more data than the pipe can hold and you `join()` that child before the parent has `get`-ten the data, the child blocks inside the feeder thread (it can't finish flushing to the pipe) while the parent blocks in `join()` (waiting for the child to exit). Classic deadlock. Drain first, then join.

There's a subtler reason to prefer `Pool`/`ProcessPoolExecutor` over hand-rolled `Process`/`Queue` topologies unless you genuinely need streaming: the pool manages worker lifecycle, restarts dead workers (in recent Python a worker that dies is replaced), propagates exceptions, and handles the chunking and dispatch for you. With raw `Process` and `Queue` you own all of that — sentinels to signal completion, draining order, handling a worker that crashes mid-stream, sizing the bounded queue for backpressure. It's the right tool for a true streaming pipeline (data that arrives over time, where you can't materialize the whole input as a list), but for "I have a list and want it mapped across cores," the pool is less code and fewer footguns. The decision in figure 7 reflects that: `Queue` is for streaming and flow control; `Pool.map` and `ProcessPoolExecutor` are for bounded, materializable maps.

One more pattern worth naming for the streaming case: when a producer is much faster than the consumers and you don't bound the queue, the queue grows without limit and you're back to an out-of-memory crash — the data piles up in the pipe and feeder buffers faster than workers drain it. A `Queue(maxsize=k)` turns that into backpressure: once `k` items are pending, the producer's `put` *blocks* until a consumer frees a slot, so the producer naturally slows to the consumers' pace and memory stays bounded. This is the same backpressure idea that runs through any producer/consumer system, applied across the process boundary.

## Results: how multiprocessing actually scales

Let me put the numbers together honestly, because "8 cores means 8×" is a lie people keep telling and it sets up disappointment. Here is a representative scaling table for the CPU-bound `transform` map on the reference 8-core box, $N = 80{,}000$ items, `chunksize=500`, `spawn` start method:

| Workers | Wall-clock | Speedup vs serial | Efficiency (speedup / workers) |
| --- | --- | --- | --- |
| 1 (serial) | 48.1 s | 1.00× | 100% |
| 2 | 24.9 s | 1.93× | 96% |
| 4 | 13.0 s | 3.70× | 92% |
| 6 | 9.1 s | 5.29× | 88% |
| 8 | 7.5 s | 6.41× | 80% |
| 16 (oversubscribed) | 7.9 s | 6.09× | 38% |

Three things to read off that table. First, **efficiency erodes as you add cores** — 96% at two workers, 80% at eight. That decline is exactly the $\frac{C}{C+O}$ factor plus the parent becoming a serialization bottleneck (it has to pickle dispatches and unpickle results for *all* workers, and there is only one parent). Second, **going past your physical core count hurts.** The jump from 8 to 16 workers on an 8-core box made it *slower* (7.9 s vs 7.5 s): the extra processes don't get extra cores, they just add context-switching, memory pressure, and more pickling contention on the single parent. The rule is **`max_workers = number of physical cores` for CPU-bound work** — `os.cpu_count()` or, better, `len(os.sched_getaffinity(0))` on Linux to respect cgroup limits in a container. Third, the *startup* cost is hidden in these numbers; for a job this size it's a small fraction, but for a sub-second job, spawning 8 workers (each re-importing NumPy and friends) can cost more than the work itself — which is the next warning.

This is the same scaling story that plays out at datacenter scale with GPUs, where the analog of pickling-and-piping is the all-reduce of gradients across devices — if you want to see how the IPC-versus-compute trade-off generalizes to multi-GPU training, the [collective-communication post](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) tells that story. The lesson rhymes: parallelism is bounded by the cost of moving data between the parallel workers, whether those workers are processes on a CPU or GPUs on a node.

### The parent is a single bottleneck — and the tail

Two effects explain almost all of the gap between the speedup you predict and the speedup you measure, and both are worth naming because they change how you design the code.

The first is the **single-parent serialization bottleneck**. There are $W$ workers but only *one* parent process, and that parent has to pickle every dispatched chunk and unpickle every returned result. All of that runs on one core, holding one GIL. If the workers finish faster than the parent can serialize their inputs and deserialize their outputs, the workers starve — they sit idle waiting for the parent to feed them or to drain their results. You can spot this in `htop`: the parent core is pinned at 100% while the worker cores dip below it. The cure is to *reduce what crosses the boundary*: return small summaries instead of large arrays (compute the aggregate in the worker, return the scalar), and chunk so the parent dispatches fewer, larger units of work. A worker that reads a million rows and returns one number puts almost no load on the parent; a worker that returns a million transformed rows makes the parent the bottleneck.

The second is the **long tail**. Total wall-clock for a parallel map is set by the *slowest* worker to finish, not the average. If one chunk happens to contain unusually expensive items — a few records that are 100× the typical cost — that one worker runs long after the other seven have gone idle, and your eight-core machine spends the tail running on one core again. The fix is finer-grained chunking (more, smaller chunks means the slow items spread out and idle workers can grab the next chunk) traded against the IPC amortization that wants *larger* chunks. That tension is exactly why `chunksize` tuning is empirical: you're balancing amortization (wants big chunks) against load balance (wants many chunks). When the cost per item is highly variable, `imap_unordered` with a moderate `chunksize` and dynamic dispatch beats a static even split, because fast workers keep pulling new chunks instead of waiting on the slow one.

### Measuring multiprocessing honestly

Benchmarking parallel code has traps the [benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) flags, plus a few specific to processes. **Warm the pool before you time it.** Pool creation — especially the first `spawn` of eight interpreters re-importing NumPy — can cost hundreds of milliseconds, and if you fold that into your timing you'll blame the work for the startup. Create the pool, run one throwaway batch, *then* start the clock. **Use `time.perf_counter()` for wall-clock**, never `time.process_time()`: `process_time` measures only the *calling* process's CPU time and is blind to what the workers do on other cores, so it will tell you parallel "took no time," which is nonsense. **Make the input large enough** that the per-run noise (scheduling jitter, the OS deciding to do something else on a core) is small relative to the signal; a job that runs for 50 ms is mostly noise on a busy machine. **Pin or at least know your core count** — `os.sched_getaffinity(0)` on Linux respects cgroup limits, so inside a container that reports 2 cores you should not spawn 8 workers expecting 8× ; you'll get 2× at best and pay overhead for the other six. And **run the comparison on a quiet machine**: if something else is using your cores, your "8 cores" are really 4, and the numbers lie. Take the median of several runs, not the minimum (the minimum hides the variance that real workloads will hit) and not the mean (one slow run from a GC pause or a background job skews it).

### Debugging across the process boundary

When a parallel job misbehaves, remember that you cannot simply drop a `breakpoint()` into a worker and expect it to work — the worker has no terminal attached, and `pdb` inside a forked or spawned child usually hangs or errors. A few habits make process debugging bearable. **Develop serially first**: write and debug `transform` as a plain function over a small input, get it correct, *then* wrap it in the pool. Most bugs are in the function, not the parallelism, and they're a hundred times easier to find in one process. **Set `max_workers=1` to reproduce**: a one-worker pool still goes through the full pickling-and-IPC path (so it catches picklability bugs and serialization issues) but runs one task at a time, which makes errors deterministic and easier to trace. **Log with the PID**: have each worker log `os.getpid()` so you can tell which worker did what when output interleaves. And **make sure exceptions surface** — if you use `submit`, you only see a worker's exception when you call `fut.result()`; a future whose exception you never retrieve fails silently, which is one of the more maddening ways a parallel job can "succeed" while producing wrong output.

## Case studies and real numbers

A few grounded data points, named and honest about their setup.

**The feature-engineering job from the intro.** Twelve million rows, a per-row transform of roughly 500–800 µs (regex parse plus rolling statistics), records that pickled small. On an 8-core production box with `ProcessPoolExecutor`, `chunksize` tuned to ~2000, and `forkserver` start method, the job went from ~3 hours single-core to ~28 minutes — about 6.4× — which is squarely in the range the cost model predicts for $O/C$ around 0.1–0.15. The remaining gap from 8× was the single-parent serialization bottleneck and some load imbalance on a long tail of unusually large records.

**The tiny-task disaster.** A colleague parallelized a `sum`-like aggregation over ten million floats by sending one float per task to `Pool.map`. It ran at roughly **0.6×** of the serial version — slower — and pegged the parent core at 100% doing nothing but pickling. The serial NumPy one-liner `arr.sum()` was, of course, ~100× faster than *either*, because for elementwise numeric work the right lever is vectorization, not parallelism. The fix was to delete the multiprocessing entirely. The lesson: **before parallelizing, make sure you've already pulled the cheaper levers** — algorithm and vectorization — because processes are rung four of the ladder and the cheapest 100× usually lives on rungs one and two.

**Polars and "free" parallelism.** It is worth naming the alternative explicitly. Polars (a DataFrame library written in Rust) runs its query engine multi-threaded *with the GIL released*, so a Polars `group_by().agg()` already uses all your cores with no pickling, no process startup, no `chunksize` to tune — the parallelism happens below Python in native code over a shared Arrow buffer. For tabular work, that is almost always better than hand-rolling `multiprocessing` over pandas, because there is no serialization boundary at all: the data never leaves the one process. The general principle — pushing the parallel loop down into native code that holds a shared buffer and releases the GIL — is exactly what [Numba's `prange`](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code) does inside one process, and it is frequently a better answer than processes precisely because it has *no pickling tax*. Reach for processes when your work is Python-level and not easily vectorized or JIT-compiled; reach for native multithreading (Polars, NumPy with a threaded BLAS, Numba `prange`) when it is.

**Apple M2 versus the Linux box.** On the M2 laptop the same `ProcessPoolExecutor` map gave a similar *shape* of result — efficiency around 75–85% on the performance cores — but the default start method is `spawn`, so the *startup* cost was visibly higher: a short job that spawned and tore down a pool repeatedly spent a meaningful slice of its time re-importing modules in fresh interpreters. Keeping one long-lived pool (rather than creating a new pool per batch) erased that difference. The portable lesson: **create the pool once and reuse it**; pool creation is the expensive part under `spawn`. The M2 also reminded me that "8 cores" is not uniform — its performance and efficiency cores run at different speeds, so the slowest-worker tail is a little worse than on a homogeneous server, and oversubscribing past the performance-core count helped less than the raw logical-core count suggested.

**The web-scraper that should have been async.** A batch job fetched and parsed forty thousand pages. Someone parallelized the *whole* fetch-and-parse with a process pool, reasoning "more cores, more speed." It barely beat serial, because fetching is I/O-bound — the workers spent almost all their time blocked on the network, not computing, and you don't need eight cores to wait on eight sockets; you need eight *threads* (or one event loop). Worse, every fetched page (hundreds of KB of HTML) was pickled back to the parent for aggregation, so the parent drowned in deserialization. The right design split the two phases by their nature: fetch concurrently with threads or `asyncio` (I/O-bound, GIL released on the wait — see the [threading post](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits)), then CPU-parse the downloaded bytes with a process pool *only if* parsing was actually the bottleneck. Profiling showed parsing was 15% of the time, so a process pool over just the parse step gave a small, real win, and the threads gave the big one. The lesson is the decision tree in figure 8: match the tool to whether the work is CPU-bound or I/O-bound *before* reaching for processes.

## When to reach for processes, and when not to

This is the decision section, and figure 8 is the flowchart. Be honest about the cost: a process pool is real complexity (picklability constraints, start-method hazards, shared-memory lifecycle, harder debugging) and it earns its keep only in a specific regime.

![decision tree showing CPU-bound and parallelizable work with compute much greater than IPC routing to multiprocessing, tiny tasks or big data movement routing to batching first, and I/O-bound work routing to threads or asyncio](/imgs/blogs/multiprocessing-true-parallelism-and-the-cost-of-pickling-8.png)

**Reach for `multiprocessing`/`ProcessPoolExecutor` when all three hold:**

- The work is **CPU-bound** — it pegs one core, the GIL is the wall, threads don't help. (If it's I/O-bound, use threads or `asyncio`; processes add overhead for no parallelism gain, because I/O already releases the GIL.)
- The work is **parallelizable** — tasks are independent, or nearly so.
- **Compute dwarfs IPC** — each task does enough real work ($C \gg O$) that the pickling tax is a small fraction. If not, *batch first* (chunk) to make $C$ big, or *don't bother*.

**Do NOT reach for processes when:**

- **The task is I/O-bound.** Threads release the GIL during the wait; `asyncio` scales to far more concurrent waits without per-worker overhead. Processes here just add pickling and startup cost. See the [threading post](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits).
- **The tasks are tiny.** A million single-item tasks lose to serial. Batch them (`chunksize`) or rethink.
- **You're moving lots of data per task.** If the win in compute is swamped by pickling a big argument or result, use `shared_memory`, send indices instead of data, or compute closer to where the data lives.
- **You haven't pulled the cheaper levers.** If the loop can be *vectorized* (NumPy) or *JIT-compiled* (Numba) or pushed into a *multithreaded native library* (Polars, threaded BLAS), do that first — it's often a bigger win with no serialization boundary at all. Parallelism is rung four; check rungs one through three first.
- **The function or its data isn't picklable** and can't easily be made so (open connections, lambdas, captured locks). Restructure or stay serial.
- **It's a short-lived process** where pool startup (especially under `spawn`) costs more than the work. Amortize by reusing a long-lived pool, or skip it.

The shortest version: *processes give you real parallelism, but you pay at the serialization boundary. Make each task big and the data crossing the boundary small, or the boundary eats your speedup.*

It helps to hold the three concurrency tools side by side, because the choice among them is the most common decision this part of the series asks you to make:

| Tool | True parallelism | Best for | Main cost | Memory sharing |
| --- | --- | --- | --- | --- |
| Threads | No (one GIL) | I/O-bound work (the GIL releases on the wait) | Lock hand-off, races, deadlocks | Shared — direct, zero-copy |
| Processes | Yes (one GIL each) | CPU-bound work that's parallelizable | Pickling + IPC + startup | None — must serialize or use shared memory |
| asyncio | No (one thread) | Massive I/O concurrency (10k connections) | Cooperative — one blocking call stalls all | Shared — single thread, no copy |

Read it as a decision: is the work I/O-bound? Then threads or `asyncio`, never processes — the GIL isn't your wall, the network is, and processes only add a pickling tax for parallelism you don't need. Is it CPU-bound and parallelizable with compute much greater than IPC? Then processes — they're the only one of the three that actually puts more than one core to work on Python bytecode. Anything else, look back down the leverage ladder first.

## Key takeaways

- **Separate processes escape the GIL because each has its own interpreter and its own GIL.** Eight processes run Python bytecode on eight cores at the same instant — true parallelism, not interleaving.
- **The cost model is `cost = C + O`, where `O = 2 × (pickle + transfer + unpickle)`.** You pay the serialization round trip twice, in and out. The speedup is $S = W \cdot \frac{C}{C+O}$ — near-linear only when compute dwarfs IPC.
- **Predict before you code.** A napkin calculation of $C$ versus $O$ tells you whether processes will help (near $W$×), help half as much ($W/2$), or make things *slower* (below 1×). Do that math first.
- **Pick the start method deliberately.** `fork` is cheap (copy-on-write) but **unsafe with threads** (inherited locked mutexes deadlock); `spawn` is safe and clean but slow (re-imports everything); `forkserver` gives near-fork speed safely on Linux. If you have threads, never use `fork`.
- **Never dispatch tiny tasks one at a time.** Chunk them. `chunksize` amortizes the per-item IPC from $O$ to $O/k$; aim for chunks big enough to amortize but numerous enough to balance load.
- **For big buffers, share, don't copy.** `shared_memory` (or copy-on-write `fork` for read-only data) hands eight workers one 100 MB array as a zero-copy view instead of pickling it eight times. Remember to `unlink()`.
- **Everything through a `Queue` is pickled too**, and you must drain before you join or you deadlock.
- **`max_workers = physical cores` for CPU-bound work.** Oversubscribing past your core count adds overhead with no extra parallelism. Create the pool once and reuse it.
- **Pull the cheaper levers first.** Vectorization (NumPy), JIT (Numba), and native multithreaded libraries (Polars) often beat processes with *no serialization boundary at all*. Processes are rung four of the ladder, not rung one.

## Further reading

- The CPython `multiprocessing` documentation — the authoritative reference for `Pool`, `Process`, `Queue`, start methods, and `shared_memory`, including the programming guidelines and platform notes.
- The `concurrent.futures` documentation — `ProcessPoolExecutor`, `Future`, `as_completed`, and the shared executor interface with threads.
- The `pickle` module documentation — what is and isn't picklable, `__reduce__`/`__getstate__`/`__setstate__`, and the protocol versions.
- The `multiprocessing.shared_memory` documentation and `SharedMemoryManager` — the lifecycle, the `unlink()` requirement, and the `/dev/shm` notes.
- *High Performance Python* by Gorelick and Ozsvald — the multiprocessing and IPC chapters cover the cost model and shared-memory patterns in depth.
- Series intro: [Why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the leverage ladder this post's rung four sits on.
- [The GIL explained: what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) — why threads can't parallelize CPU-bound Python and processes can.
- [Threading done right: I/O-bound concurrency and its limits](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) — the right tool when the work is I/O-bound, not CPU-bound.
- [Numba: JIT compiling Python to machine code](/blog/software-development/python-performance/numba-jit-compiling-python-to-machine-code) — native multithreading with `prange` that parallelizes inside one process with no pickling tax.
- [NumPy from first principles: the ndarray and why it's fast](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) — the packed buffer that shared memory lets you hand across the boundary for free.
- [Collective communication and NCCL all-reduce from scratch](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) — the multi-GPU analog of the IPC-versus-compute trade-off at datacenter scale.

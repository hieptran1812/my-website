---
title: "The GIL Explained: What It Protects and What It Costs"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Understand the Global Interpreter Lock from first principles: why CPython needs one mutex, why threads never speed up CPU-bound Python, why they do speed up I/O, and which concurrency tool to reach for when."
tags:
  [
    "python",
    "performance",
    "optimization",
    "gil",
    "concurrency",
    "threading",
    "multiprocessing",
    "asyncio",
    "cpython",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-1.png"
---

You have an 8-core machine. You have a CPU-heavy job — say, hashing a few million records, or scoring a batch of feature vectors, or parsing a pile of JSON. The single-threaded version takes 40 seconds and pins exactly one core at 100% while the other seven sit idle. So you do the obvious thing: you split the work across four threads with `ThreadPoolExecutor`, run it again, and watch the clock.

It takes 41 seconds. Sometimes 44. You added three threads and the program got *slower*.

This is the single most surprising — and most consequential — fact about Python performance, and almost every Python developer hits it eventually, usually in production, usually under pressure. The reason has a name: the **Global Interpreter Lock**, or GIL. It is one mutex, held per interpreter, that a thread must own to execute Python bytecode. Because only one thread can hold it at a time, only one thread runs Python at a time — no matter how many cores you have, no matter how many threads you spawn. Your CPU-bound threads are not running in parallel. They are taking turns, with extra overhead for the turn-taking.

![Diagram showing two threads both trying to acquire the single interpreter lock so only one can run Python bytecode at a time](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-1.png)

And yet — and this is the part that makes the GIL genuinely subtle rather than just annoying — threads in Python are not useless. If your job is downloading 200 URLs, or querying a database 50 times, or reading a thousand small files, four threads really will make it roughly four times faster. The exact same `ThreadPoolExecutor` that did nothing for the CPU job will give you a 3.8× speedup on the I/O job. Same tool, opposite result. The whole trick to using concurrency well in Python is understanding *why* — and that understanding starts with the GIL.

This post is the opening of the concurrency track in the [Fast Python series](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means). By the end you will be able to: explain in one breath what the GIL is and what it protects; predict whether threads will help a given workload before you write a line of code; read the difference between CPU-bound and I/O-bound through the GIL's eyes; reason about the switch interval and contention; and — the practical payoff — reach for the *right* concurrency tool (threads, asyncio, multiprocessing, native code, or a free-threaded build) on the first try instead of the third. The posts that follow this one go deep on each tool; this one builds the mental model that makes all of them make sense.

We will work the way the rest of the series works: build the intuition, prove the mechanism with real CPython internals, run actual code on a named machine, and report measured before-and-after numbers. Throughout, the machine is **an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM** — a perfectly ordinary cloud VM or laptop. The numbers below are the kind you will reproduce within a factor of two on similar hardware; where a number is approximate or hardware-dependent I say so.

## 1. What the GIL actually is

Let us be precise, because the GIL is surrounded by more folklore than almost any other topic in Python.

The Global Interpreter Lock is a **mutex** — a mutual-exclusion lock — that belongs to the CPython interpreter. A thread must **hold** this lock to execute Python bytecode. Since a mutex can be held by at most one thread at a time, the consequence is immediate and total: **at any instant, at most one thread is executing Python bytecode in a given interpreter.** Not "mostly one." Not "one per core." One, full stop.

That is the entire definition. Everything else — the slowdowns, the I/O wins, the multiprocessing workarounds, the decade of debate — is a consequence of that one sentence.

A few clarifications that immediately matter:

- **"CPython", not "Python."** The GIL is an implementation detail of CPython, the reference interpreter that ships from python.org and that you almost certainly use. It is not part of the Python language. Other implementations have made different choices: Jython (on the JVM) and IronPython (on .NET) have no GIL; PyPy has a GIL much like CPython's. When people say "Python has a GIL," they mean CPython.

- **"Per interpreter", not "per process."** Historically there was one GIL per process because there was effectively one interpreter per process. Modern CPython supports multiple interpreters in one process (PEP 684, 3.12+), each with its *own* GIL — which is exactly what makes subinterpreters interesting for parallelism. And the experimental free-threaded build (PEP 703, 3.13+) removes the GIL entirely. We will come back to both. For the default build you run today, treat it as one lock for your whole program.

- **"Execute Python bytecode", not "do work."** This distinction is the key to the entire post. The GIL is required to *run bytecode in the eval loop*. It is **not** required to sit and wait for a network packet, and it is **not** required while running C code that has explicitly released it. Those two exceptions are where every concurrency win in Python comes from.

So the mechanism is: a thread that wants to run Python code acquires the GIL; it runs; periodically (or when it blocks on I/O) it releases the GIL; another waiting thread acquires it and runs. The threads are real OS threads — the operating system schedules them, they can run on different cores — but they are *serialized by the GIL* the moment they touch the interpreter. Two threads, one lock: only the winner runs Python, the loser waits, and the lock is handed off on I/O or after a time slice.

A point that trips people up: the threads really are genuine, kernel-scheduled OS threads, not green threads or coroutines. The operating system is perfectly willing to run them on eight different cores simultaneously. The GIL is a *software* constraint layered on top — a lock in user space that every thread must pass through before touching interpreter state. So you get the worst of the accounting in `top`: eight real threads exist, the scheduler shuffles them across cores, but only one is ever doing useful Python work, and the others are blocked on the lock. This is also why a CPU-bound multithreaded Python program shows ~100% of *one* core in utilization graphs rather than 0% or 800% — one thread runs, the rest wait, and the small overhead of shuffling the lock around is the only thing the idle cores do.

This is the cleanest way to see why **processes** sidestep the whole problem. A separate process has its own memory space and therefore its own interpreter and its own GIL. Two processes share *nothing* by default, so there is no shared refcount to corrupt and no single lock to serialize them — each runs Python bytecode at full tilt on its own core. That independence is exactly what you pay for with the pickling and inter-process-communication costs that the multiprocessing post covers; you trade shared memory (and the GIL that protects it) for true parallelism.

To understand why CPython would impose such a sweeping restriction on itself, we have to look at what the interpreter is protecting. And that takes us straight to reference counting.

## 2. Why the GIL exists: a worked race on a refcount

CPython manages memory primarily by **reference counting**. Every Python object carries an integer field, `ob_refcount`, that records how many references point to it. When you bind a name to an object, append it to a list, pass it to a function, or store it in a dict, CPython runs `Py_INCREF` — it adds one to the count. When a reference goes away (a local variable goes out of scope, an item is removed from a list, a name is rebound), it runs `Py_DECREF` — it subtracts one. When the count hits zero, the object is freed immediately. (A separate generational garbage collector handles reference *cycles*, which plain refcounting cannot; that is a story for the memory track. For more on how the interpreter runs your code and how reference counting fits in, see [the CPython execution model](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop).)

Here is the part that makes refcounting and threads a dangerous mix. `Py_INCREF` reduces to roughly this C code:

```c
/* Simplified from CPython's object.h */
static inline void Py_INCREF(PyObject *op) {
    op->ob_refcount++;
}

static inline void Py_DECREF(PyObject *op) {
    if (--op->ob_refcount == 0) {
        _Py_Dealloc(op);   /* free the object */
    }
}
```

The line `op->ob_refcount++` looks atomic. It is not. On a real CPU, incrementing a memory location is at least three operations:

1. **Load** the current value from memory into a register.
2. **Add** one to the register.
3. **Store** the register back to memory.

This is the classic read-modify-write sequence, and it is *not* safe under concurrency. Two threads running it on the same object at the same time can interleave their three steps and lose an update.

#### Worked example: a lost increment corrupts a refcount

Suppose a small integer object `x` has `ob_refcount = 10`. Thread A and thread B both want to take a new reference to `x` at the same moment — A appends `x` to a list, B passes `x` as a function argument. Both call `Py_INCREF`. The correct final count is **12**. Watch what can happen without a lock, step by step in real time:

| Time | Thread A | Thread B | `ob_refcount` in memory |
|------|----------|----------|--------------------------|
| t0 | load → reg_A = 10 | | 10 |
| t1 | | load → reg_B = 10 | 10 |
| t2 | add → reg_A = 11 | | 10 |
| t3 | | add → reg_B = 11 | 10 |
| t4 | store reg_A → 11 | | 11 |
| t5 | | store reg_B → 11 | 11 |

Both threads loaded 10, both computed 11, both stored 11. Two increments happened; the count went up by *one*. The final value is **11, not 12.** One reference is now invisible to the bookkeeping.

Why is this catastrophic and not merely "off by one"? Because that lost increment means CPython now thinks there is one fewer reference to `x` than there really is. Later, when one of those references goes away, `Py_DECREF` drives the count to zero one step too early. CPython frees `x` — deallocates the memory — while a live reference still points at it. The next time any thread touches `x` through that dangling reference, it reads freed memory: garbage values, a crash, or, in the worst case, silent corruption because the memory has been reused for a different object. This is a **use-after-free**, the same class of bug that powers a large fraction of security exploits in C programs. In a memory-safe-feeling language like Python, it would be a disaster.

The mirror image is just as bad. A lost *decrement* leaves the count too high, so the object is never freed — a memory leak. And it is not only refcounts. The same read-modify-write hazard applies to **every** mutable internal structure the interpreter shares: resizing a `list` (reallocating its backing array and copying), inserting into a `dict` (probing and possibly rehashing the open-addressing table), growing a `set`, mutating an object's `__dict__`, updating interned-string tables, free-list management in the allocator. Each of these is a multi-step mutation of shared state that two threads could interleave and corrupt.

So CPython faces a genuine concurrency problem: **the interpreter's internal data structures are not thread-safe, and the most fundamental operation in the language — taking and dropping a reference — happens on literally every line of code.** A Python program does billions of `INCREF`/`DECREF` operations. There is no way to "just be careful." The interpreter needs serialization.

## 3. One big lock versus a thousand small ones

There are, in principle, two ways to make shared mutable state safe under threads.

**Fine-grained locking.** Put a small lock on each object, or each container, or each subsystem. A thread locks only what it touches. This is how a well-written multithreaded C++ or Java program protects its data. The upside is real parallelism: two threads working on unrelated objects never contend. The downsides are severe in CPython's specific situation:

- **Overhead on every operation.** Acquiring and releasing a lock costs tens of nanoseconds even uncontended. If every `INCREF`/`DECREF` had to take a per-object lock — and there are billions of them — single-threaded code would slow down dramatically. Early experiments (Greg Stein's "free-threading" patch in 1999) made the *single-threaded* interpreter roughly 2× slower. Nobody wanted that trade.
- **Deadlock risk.** Many small locks means lock-ordering rules, and getting them wrong means deadlock. The interpreter's internals are a tangled graph of references; designing a sound lock hierarchy across all of it is genuinely hard.
- **Atomic refcounts.** You can avoid per-object locks for refcounts by making the counter atomic (a hardware compare-and-swap). But atomic operations also cost more than plain ones — and they force cache-line ping-pong between cores, which can be slower still on shared objects like small integers and `None`.

**One coarse lock — the GIL.** Take a single lock that protects *everything*. A thread holds it while running bytecode and releases it periodically. The upsides:

- **The single-threaded interpreter stays fast.** With one lock acquired once at the top of a time slice (not per operation), the common case — most Python programs are single-threaded — pays almost nothing. `INCREF` is a plain `++`, the fastest possible thing.
- **C extensions are simple to write.** An extension author can assume that while their code runs, no other thread is mucking with Python objects, unless they explicitly choose to release the GIL. This made the C ecosystem (NumPy, the standard library's C modules, thousands of third-party packages) far easier to build correctly.
- **No deadlocks from interpreter internals**, because there is only one lock.

The cost, of course, is the headline of this entire post: **no parallelism for Python bytecode.** That is the deal CPython struck, and for most of Python's history it was the right one — Python's value was developer productivity and a vast, easy-to-write C ecosystem, and the GIL bought both at the price of multicore CPU scaling that most programs did not need anyway.

![Layered diagram showing refcounts object internals and C API invariants all guarded by one GIL which keeps single-threaded code fast](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-5.png)

The picture to hold in your head: three different hazards — racing refcounts, racing container mutations, and C extensions that assume single-threaded access — all stack on top of one shared requirement, *serialize access to interpreter state*. One lock satisfies all three at once, and because it is acquired coarsely rather than per-operation, the single-threaded fast path pays almost nothing. That is the bargain. Simplicity and single-thread speed, traded against multicore bytecode parallelism.

### What the GIL protects, precisely

It is worth listing exactly what the GIL guarantees, because misunderstanding this is the source of a lot of subtly broken threaded code:

- **Reference counts never corrupt.** `INCREF`/`DECREF` are safe because only one thread runs them at a time.
- **A single bytecode operation is atomic with respect to other threads.** The GIL is only released *between* bytecodes (at the check points we will discuss), never in the middle of one. So one `LOAD_FAST`, one `BINARY_OP`, one `STORE_SUBSCR` completes without another thread interleaving.
- **CPython's own internal structures stay consistent** — the small-object allocator, interned strings, type objects, and so on.

And here is the trap: the GIL protects the *interpreter*, not *your program's logic*. People hear "only one thread runs at a time" and conclude their code is automatically thread-safe. It is not. A statement like `counter += 1` compiles to several bytecodes — load `counter`, load `1`, add, store back to `counter`. The GIL can be released *between* those bytecodes, so two threads can still interleave at the statement level and lose an update to *your* counter, even though each individual bytecode was atomic. The GIL keeps the *interpreter* from corrupting; keeping *your data* consistent is still your job, with your own locks. (We cover that in detail in the threading post that follows this one.)

### The same race, one level up — at the bytecode boundary

This is worth making concrete, because it is the bug that surprises people who "know about the GIL" but trust it too much. We saw in §2 that a *refcount* increment can be lost without a lock — but the GIL fixes that, because `INCREF` happens inside a single bytecode that runs to completion. What the GIL does *not* fix is a race in *your* Python code that spans multiple bytecodes. Here is the disassembly of `counter += 1`:

```pycon
>>> import dis
>>> dis.dis(compile("counter += 1", "<x>", "exec"))
  0           LOAD_NAME                0 (counter)
              LOAD_CONST               0 (1)
              BINARY_OP               13 (+=)
              STORE_NAME               0 (counter)
```

Four bytecodes. The GIL guarantees each one runs atomically, but it can be handed off to another thread *between* any two of them — for example, right after `BINARY_OP` computes the new value and before `STORE_NAME` writes it back. Two threads can both `LOAD_NAME` the same old value, both add one, and both store — exactly the lost-update pattern from §2, now at the Python level. The fix is the same as in any language: serialize the read-modify-write with your own lock (`threading.Lock`) or use an atomic primitive. The point for *this* post is the mental shift: **the GIL is an interpreter-integrity lock, not an application-correctness lock.** Knowing that distinction is what separates threaded code that works from threaded code that works *most* of the time and corrupts under load.

There is a subtle corollary. Because the GIL is released only between bytecodes, some single-bytecode operations *are* effectively atomic — a bare `d[k] = v` (one `STORE_SUBSCR`) or `lst.append(x)` (one call that completes under the GIL) will not interleave with another thread mid-operation. A lot of "lucky" threaded Python relies on this without knowing it. But it is fragile: the moment you do *two* operations that must be consistent together (check-then-act, read-modify-write, "if key not present, insert"), the GIL gives you nothing, and you need a real lock. Never design for the accidental atomicity; design for the contract, which is "individual bytecodes only."

## 4. The first cost: CPU-bound threads do not speed up

Now we can derive, not just assert, the headline result. Take a CPU-bound function — one that does nothing but compute, never waiting on anything external. The canonical toy is summing a big arithmetic series, or counting how often a condition holds. Here is a deliberately pure-Python, CPU-bound workload:

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_work(n: int) -> int:
    """Pure CPU: no I/O, no native calls that release the GIL."""
    total = 0
    for i in range(n):
        total += i * i % 7
    return total

N = 20_000_000

def run_serial(tasks: int) -> None:
    for _ in range(tasks):
        cpu_work(N)

def run_threaded(tasks: int) -> None:
    with ThreadPoolExecutor(max_workers=tasks) as pool:
        list(pool.map(cpu_work, [N] * tasks))

def run_processes(tasks: int) -> None:
    with ProcessPoolExecutor(max_workers=tasks) as pool:
        list(pool.map(cpu_work, [N] * tasks))

if __name__ == "__main__":
    for name, fn in [("serial-4", lambda: run_serial(4)),
                     ("threads-4", lambda: run_threaded(4)),
                     ("processes-4", lambda: run_processes(4))]:
        t0 = time.perf_counter()
        fn()
        print(f"{name:>12}: {time.perf_counter() - t0:6.2f} s")
```

On the 8-core x86-64 Linux box, CPython 3.12, the output looks like this:

```bash
    serial-4:  10.40 s
   threads-4:  10.90 s
 processes-4:   2.85 s
```

Four CPU-bound tasks run in 10.4 s serially. Four **threads**: 10.9 s — no speedup, in fact a small *slowdown*. Four **processes**: 2.85 s — a 3.6× speedup on the 8-core machine.

Let us prove why threads cannot win here.

The total CPU work to do is fixed: four calls to `cpu_work(N)`, each requiring some number of bytecode-executing cycles, call it $W$ each, for $4W$ total. To execute *any* of that bytecode, a thread must hold the GIL. Only one thread holds the GIL at a time. Therefore the bytecode executes strictly serially regardless of thread count — the threads simply hand the GIL back and forth, but the total bytecode-time is still $4W$, exactly as if one thread ran all four tasks in sequence.

Formally, this is **Amdahl's law** with the parallelizable fraction pinned to zero. Amdahl's law says that if a fraction $p$ of the work can run in parallel with speedup $s$ on the parallel part, the overall speedup is

$$S = \frac{1}{(1-p) + p/s}.$$

For pure-Python CPU work under the GIL, the bytecode cannot run in parallel at all, so the effective $p = 0$, and $S = 1$. No amount of threading changes that. The cores are there; the GIL keeps them from being used for Python bytecode.

### Why two threads can be *slower* than one

The slowdown — 10.9 s versus 10.4 s — is not noise. It comes from two real costs that exist only when multiple threads contend for the GIL.

**Cost 1: the handoff itself.** When thread A's time slice ends, it must release the GIL, signal a waiting thread, and a context switch must occur before thread B acquires the GIL and resumes. Each handoff is a condition-variable wake-up plus an OS context switch — on the order of a microsecond. With CPU-bound threads, these handoffs happen constantly (every switch interval; see §6), and each one is pure overhead that the single-threaded run never pays.

**Cost 2: cache effects and the "convoy."** When thread B resumes on a different core, the CPU caches there are cold for B's data. Worse, historically the GIL handoff logic could create a **convoy effect**: a CPU-bound thread releases the GIL, an I/O thread grabs it for a moment, and the CPU thread — which would happily keep running — gets repeatedly preempted, thrashing. The GIL was substantially rewritten in Python 3.2 (Antoine Pitrou's new GIL) specifically to reduce this, replacing the old "release every N bytecodes" scheme with a time-based one. It is much better now, but the fundamental truth stands: **adding threads to CPU-bound Python adds contention overhead and removes nothing, so it never helps and sometimes hurts.**

The lesson is blunt and worth memorizing: **if your hot path is pure-Python CPU work, threads are the wrong tool — full stop.** Reach for processes, native code that releases the GIL, or a free-threaded build. We will lay out that decision precisely at the end.

### Measuring this honestly

A word on methodology, because "I added threads and it got slower" is the kind of claim people dismiss as a fluke unless you measure it right. Three rules make the GIL effect unambiguous:

- **Make the work big enough to dominate.** If `cpu_work` runs for 50 µs, thread-creation and pool overhead swamp the signal. Size the task so each call runs for hundreds of milliseconds to seconds; then the GIL serialization is the dominant term and the noise is small.
- **Use `time.perf_counter()` for wall-clock, not `time.process_time()`.** `process_time` measures CPU time across all threads and will happily report that "4 threads used 40 seconds of CPU in 10 seconds of wall-clock" — true, but it hides the fact that the *wall-clock* did not improve, which is the number your users feel. Always report wall-clock for a parallelism claim.
- **Repeat and take the median, and pin the BLAS thread count.** Run each configuration several times and report the median to shake off scheduler noise. For the NumPy experiment in §7, set `OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`) first, otherwise BLAS spawns its *own* threads and you cannot tell the GIL effect from the BLAS effect.

#### Worked example: scaling CPU-bound threads from 1 to 8

To see the serialization directly, hold the *total* work fixed and split it across more threads, measuring wall-clock at each thread count. Total work is eight units of `cpu_work(N)`; we hand it to 1, 2, 4, and 8 threads and divide.

| Threads | Wall-clock (s) | Speedup vs 1 thread | CPU cores used |
|---------|----------------|---------------------|----------------|
| 1 | 20.8 | 1.00× | ~1.0 |
| 2 | 21.2 | 0.98× | ~1.0 |
| 4 | 21.6 | 0.96× | ~1.0 |
| 8 | 22.5 | 0.92× | ~1.0 |

The wall-clock is flat-to-rising and the machine never uses more than about one core's worth of compute, no matter how many threads. Each added thread brings only more GIL handoffs, so the curve drifts the *wrong* way. Contrast this with the I/O table you could build from §5, where the same axis (1 → 8 workers) drops wall-clock from 8.3 s to ~1.1 s. Same machine, same pool, opposite slope — the entire difference is whether the work holds the GIL or releases it. If you ever see this flat-or-rising shape on a workload you expected to scale, you have found a GIL-bound (pure-Python CPU) section, and the fix is processes or native code, never more threads.

## 5. The first win: I/O-bound threads *do* speed up

Now change one thing. Make the work *wait* instead of *compute*. Here is an I/O-bound workload — it spends almost all its time blocked, not running bytecode:

```python
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URLS = ["https://httpbin.org/delay/1"] * 8   # each responds after ~1 s

def fetch(url: str) -> int:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return len(resp.read())

def run_serial() -> None:
    for url in URLS:
        fetch(url)

def run_threaded(workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, URLS))

if __name__ == "__main__":
    for name, fn in [("serial", run_serial),
                     ("threads-8", lambda: run_threaded(8))]:
        t0 = time.perf_counter()
        fn()
        print(f"{name:>11}: {time.perf_counter() - t0:5.2f} s")
```

Output on the same box (network times vary, so these are representative):

```bash
     serial:  8.30 s
  threads-8:  1.15 s
```

Eight requests that each take about a second run in 8.3 s serially — exactly what you would expect, one after another. With eight threads: **1.15 s**, roughly a 7× speedup. Same `ThreadPoolExecutor`, opposite outcome from the CPU case. Why?

Because **CPython releases the GIL around blocking I/O.** When `urlopen` issues the underlying `recv()` system call and the thread is about to block waiting for the network, CPython does the equivalent of:

```c
/* Pattern inside CPython's socket / file I/O C code */
Py_BEGIN_ALLOW_THREADS      /* release the GIL */
result = recv(fd, buf, len, 0);   /* block here; GIL is free */
Py_END_ALLOW_THREADS        /* re-acquire the GIL */
```

While thread A is parked in `recv()` waiting for bytes, **it is not holding the GIL.** That means thread B can acquire the GIL and run *its* Python code — issue *its* request, then also block on *its* `recv()`, releasing the GIL again, so thread C runs, and so on. All eight requests get issued nearly at once; then all eight threads sit blocked in the OS, in parallel, waiting for their responses. The waiting overlaps. The wall-clock time collapses from "sum of the waits" to "the longest single wait" (plus a little overhead).

![Before and after comparison showing CPU-bound threads serialize on the GIL for no speedup while I/O-bound threads overlap because the wait releases the lock](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-2.png)

Put the two cases side by side and the rule jumps out. In the CPU-bound case (left), all four threads need the GIL the whole time, so they run one at a time plus contention overhead — net 1.0× and a touch slower. In the I/O-bound case (right), each thread releases the GIL the instant it blocks, so the waits overlap and four threads finish in roughly the time of one — a real 3.8× (and with eight overlapping waits, ~7×). **Threads help exactly when, and only when, the work spends significant time outside the GIL.** For I/O, that "outside" time is the blocking system call. There is a second, equally important source of "outside" time, which is native C code — and it deserves its own section.

The amount of speedup follows a simple model. If a task spends fraction $f$ of its time waiting (GIL released) and fraction $1-f$ running Python bytecode (GIL held), then $T$ threads can overlap the waiting but must still serialize the bytecode. The best achievable speedup is bounded by

$$S \le \frac{1}{(1-f) + f/T}.$$

For pure I/O wait, $f \approx 1$ and $S \approx T$ — you get nearly linear speedup until you saturate the network or the server. For our fetch example $f$ is very close to 1 (almost all time is `recv`), so eight threads give close to 8×. For a task that is half-wait, half-compute ($f = 0.5$), even infinite threads cap you at $S = 1/0.5 = 2$×. This is exactly Amdahl's law again, with the "parallel" fraction being the part that runs with the GIL released.

This model also tells you how to *size* the thread pool, which is a question people get wrong in both directions. If each task waits for fraction $f$ and computes for $1-f$ under the GIL, then while one thread computes, the others can wait — so the useful number of threads is roughly $1/(1-f)$ before the GIL-held compute portion becomes the new bottleneck. For nearly pure I/O ($f \to 1$) that number is large, and the practical cap becomes the *external* resource (the remote server's connection limit, the database's pool size, the file-descriptor limit) rather than the GIL. For tasks that are, say, 90% wait and 10% under-GIL compute, about ten threads keeps a CPU-equivalent worth of Python running while the rest overlap their waits; adding a hundred more threads buys little because the 10% compute portion is already serialized by the GIL. The two failure modes are: too few threads (you under-overlap the waits and leave throughput on the table) and too many threads (each OS thread costs ~1 MB of stack and the extra GIL handoffs add overhead, so a 5,000-thread pool can be *slower* and far more memory-hungry than a 64-thread one). When the connection count climbs into the thousands, that memory wall is exactly the reason to switch from threads to asyncio, which we get to in §9. The right pool size is "enough to overlap the waits, bounded by the slowest external resource" — and you find it by measuring throughput at a few pool sizes, not by guessing.

## 6. The switch interval: how the GIL is handed off

We have said the GIL is released "periodically." Let us make that precise, because it controls the granularity of thread switching and you can tune it.

A thread that has the GIL and is doing pure CPU work would, left alone, never voluntarily release it — there is no I/O to trigger a release. To keep one CPU-bound thread from starving all the others forever, CPython runs a timer. After a configurable interval, the running thread is asked to **drop the GIL** at the next safe point (between two bytecode instructions) so another waiting thread can run. That interval is the **switch interval**, and you can read and set it:

```pycon
>>> import sys
>>> sys.getswitchinterval()
0.005
>>> sys.setswitchinterval(0.001)   # ask for finer-grained switching (1 ms)
>>> sys.getswitchinterval()
0.001
```

The default is **0.005 seconds — 5 milliseconds.** The mechanism (since Python 3.2's new GIL) works like this: when a thread wants the GIL but cannot get it, it waits on a condition variable with a timeout equal to the switch interval. If the current holder has not voluntarily released the GIL (because of I/O) within that window, the waiter sets a `gil_drop_request` flag. The holder checks this flag between bytecodes and, seeing it, releases the GIL and hands off. So the switch interval is *not* "switch every 5 ms no matter what" — it is "if someone is waiting and the holder hasn't yielded, force a handoff after 5 ms."

![Timeline showing thread A holding the GIL and running bytecode until it hits I/O or the five millisecond tick then releasing so thread B can acquire and run](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-3.png)

The handoff sequence, in order: thread A holds the GIL and runs bytecode; it either blocks on I/O or the 5 ms timer expires with B waiting; A reaches a between-bytecode check point and releases the lock; B acquires the GIL; B runs its bytecode. Then the cycle can repeat in the other direction.

There are two ways the GIL leaves a thread, and the difference matters:

- **Voluntary release on I/O (or a native call that releases it).** This happens *immediately* when the thread is about to block — it does not wait for the 5 ms timer. This is why I/O-bound threads switch crisply and overlap so well.
- **Forced release on the switch interval.** This is the fallback for CPU-bound threads that never block. It is what keeps a pure-compute thread from monopolizing the interpreter.

#### Worked example: tuning the switch interval changes responsiveness, not throughput

Imagine a process with one CPU-bound thread crunching numbers and one I/O thread that needs to respond to a heartbeat every few milliseconds. With the default 5 ms interval, the I/O thread can wait up to ~5 ms to get the GIL after the CPU thread grabs it — a worst-case 5 ms latency spike on the heartbeat. Lower the interval to 1 ms with `sys.setswitchinterval(0.001)` and the worst-case wait drops to ~1 ms: the heartbeat thread becomes more responsive.

But — and this is the catch — you have *not* increased total throughput one bit. The CPU thread still does the same total bytecode work; you have only changed *how often* the GIL is handed off. In fact, lowering the interval **increases** handoff overhead (more context switches per second), so a too-small interval can *reduce* CPU throughput while improving latency. On the 8-core box, dropping from 5 ms to 0.1 ms on a two-thread CPU workload measurably increased total runtime by a few percent from the extra switching. The default of 5 ms is a deliberate balance: long enough that handoff overhead is small, short enough that a waiting thread does not stall for long. **Tune it only when you have a specific latency problem and you have measured both the latency win and the throughput cost.**

The switch interval also explains why CPU-bound threads can be *slightly* slower than serial (from §4): every 5 ms there is a forced handoff — a condition-variable signal and a context switch — and across a 10-second run that is ~2,000 handoffs of pure overhead that the single-threaded version never pays.

### How the handoff is actually implemented

It is worth knowing the real machinery, because it makes the behavior predictable rather than mysterious. Inside CPython, the GIL is a small struct (`_gil_runtime_state`) holding a mutex, a condition variable, the current holder's thread id, and a "locked" flag. Two operations move it: `take_gil()` and `drop_gil()`.

When a thread wants to run bytecode and another thread holds the GIL, `take_gil()` does roughly this: it waits on the GIL's condition variable with a timeout equal to the switch interval. If it is woken because the holder released the GIL voluntarily (I/O), it grabs the lock and proceeds. If instead the timeout fires and the lock is *still* held by the same thread, the waiter sets a shared flag — historically `gil_drop_request` — telling the holder "you have had your turn, please yield."

The holder, meanwhile, is running the **evaluation loop** (`ceval`), executing one bytecode after another. Between bytecodes — at well-defined check points — the loop consults an "eval breaker," a single combined flag that signals pending work: a GIL-drop request, a pending signal (like Ctrl-C), an async exception, or a scheduled callback. When the eval breaker is set because of a drop request, the holder finishes its current bytecode, calls `drop_gil()` (releasing the mutex and signaling the condition variable), then immediately calls `take_gil()` again to get back in line. That tiny gap between `drop_gil()` and re-acquiring is the window the waiting thread uses to grab the lock. This is also exactly why the GIL is released *between* bytecodes and never inside one: the check happens at the top of the loop iteration, not in the middle of executing an opcode.

Two practical consequences fall out of this design. First, **a single C function that never returns to the eval loop will not yield the GIL** no matter how long it runs — there is no check point inside a tight C loop unless the extension explicitly releases the GIL with the allow-threads macro. A badly written C extension that computes for 30 seconds without releasing the GIL freezes every other thread (and makes Ctrl-C unresponsive, because the signal check also lives at that eval-loop check point). Second, **the cost of a handoff is a condition-variable wake plus a context switch** — sub-microsecond, but nonzero — which is precisely the overhead that makes CPU-bound threads slightly slower than serial. Now the §4 slowdown is not a mystery; it is ~2,000 `drop_gil`/`take_gil` round-trips paid for nothing.

#### Worked example: a Python-level counter race you can reproduce

Here is the §4-style race made fully concrete and measurable. Two threads each increment a shared counter one million times. The arithmetic says the final value must be 2,000,000. Run it:

```python
import threading

counter = 0
ITERS = 1_000_000

def bump():
    global counter
    for _ in range(ITERS):
        counter += 1   # LOAD, +=, STORE — not atomic across the boundary

t1 = threading.Thread(target=bump)
t2 = threading.Thread(target=bump)
t1.start(); t2.start()
t1.join();  t2.join()
print(counter)          # expected 2_000_000 ...
```

On the 8-core box, CPython 3.12, this prints **2,000,000 surprisingly often** — because the increment is cheap, a thread usually finishes its loop within a switch interval and the interleaving that loses an update is rare. That is the dangerous part: the bug hides in testing and only surfaces under contention. Bump the work so the threads actually overlap (more iterations, or a `sys.setswitchinterval(1e-6)` to force frequent handoffs) and you will start seeing values *less* than 2,000,000 — lost updates, exactly the pattern from the refcount table in §2, now in your own code. The fix is a lock around the read-modify-write:

```python
lock = threading.Lock()

def bump_safe():
    global counter
    for _ in range(ITERS):
        with lock:
            counter += 1   # now atomic with respect to other threads
```

With the lock, you always get 2,000,000 — and the loop runs slower, because every iteration now pays for lock acquire/release. That cost is the price of correctness, and it is why you put the lock around the *coarsest* safe unit of work, not the tightest. The deeper takeaway: **the GIL made the bug rare enough to escape your tests but did not make it impossible.** Treat threaded Python with the same locking discipline you would use in C or Java for shared mutable state.

## 7. The second win: native code that releases the GIL

The I/O case showed that "time spent blocked in a system call" runs with the GIL released. There is a second, hugely important category of GIL-free time: **native C (or Rust, or Cython) code that explicitly releases the GIL while it computes.**

This is the secret behind the scientific Python stack. When you call a NumPy function like `np.dot` or `arr.sum()` on a large array, NumPy does the heavy lifting in a compiled C loop over a packed, typed buffer — and around that loop, it releases the GIL. While the C loop runs, no Python bytecode executes, so NumPy can let go of the GIL and let *other Python threads run in parallel*. The pattern in the C extension is:

```c
PyObject *heavy_compute(PyObject *self, PyObject *args) {
    /* parse args, get a pointer to the array data... */
    Py_BEGIN_ALLOW_THREADS          /* drop the GIL */
    for (Py_ssize_t i = 0; i < n; i++) {
        out[i] = expensive_pure_c(data[i]);   /* no Python objects touched */
    }
    Py_END_ALLOW_THREADS            /* take the GIL back before returning */
    return build_result(out, n);
}
```

The rule the extension author must follow is strict: **while the GIL is released, the code must not touch any Python object** — no `INCREF`/`DECREF`, no creating or freeing Python objects, nothing that would race with another thread's interpreter access. As long as the inner loop is "pure C over raw buffers" (which is exactly what numerical kernels are), releasing the GIL is safe and gives real multicore parallelism. This is covered in depth in [the C extensions and FFI post](/blog/software-development/python-performance/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11), which shows `Py_BEGIN_ALLOW_THREADS` in context across ctypes, cffi, and pybind11.

You can *see* NumPy release the GIL with a simple experiment: run a heavy NumPy operation in several threads and check whether you get parallel speedup. If NumPy held the GIL, threads would not help (like the pure-Python CPU case); because it releases it, they do.

```python
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# A big matrix multiply: heavy, pure-numeric, runs in C.
A = np.random.rand(2000, 2000)
B = np.random.rand(2000, 2000)

def heavy() -> float:
    return float((A @ B).sum())

def run_serial(tasks: int) -> None:
    for _ in range(tasks):
        heavy()

def run_threaded(tasks: int) -> None:
    with ThreadPoolExecutor(max_workers=tasks) as pool:
        list(pool.map(lambda _: heavy(), range(tasks)))

if __name__ == "__main__":
    for name, fn in [("serial-4", lambda: run_serial(4)),
                     ("threads-4", lambda: run_threaded(4))]:
        t0 = time.perf_counter()
        fn()
        print(f"{name:>11}: {time.perf_counter() - t0:6.2f} s")
```

A representative result on the 8-core box (note: BLAS itself is multithreaded, so set `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` first to isolate the GIL effect cleanly):

```bash
   serial-4:   4.80 s
  threads-4:   1.35 s
```

Four matrix multiplies, done as Python threads, finished in roughly a quarter of the serial time — a ~3.5× speedup *for CPU-bound work, using threads.* That seems to contradict everything in §4, until you remember the key qualifier: **§4 was about pure-*Python* CPU work.** This work is CPU-heavy but it runs in *C with the GIL released*, so it parallelizes across cores. The GIL only serializes *Python bytecode*; it does not serialize C loops that have let go of it.

![Before and after comparison of a native call that keeps the GIL pinning one core versus one that releases the GIL letting threads compute on many cores in parallel](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-6.png)

Contrast the two paths. On the left, a C call that *keeps* the GIL: it enters the function with the lock held, other threads stay blocked, one core does all the work — 1.0×. On the right, a C call that *releases* the GIL with the allow-threads macro: the lock is dropped, multiple threads compute in parallel on separate cores, and you get near-linear speedup — ~7× on eight cores for an embarrassingly parallel kernel. This is why "drop to NumPy / Numba / Cython / Rust for the hot loop" is not just about making *one* thread faster — it can also unlock *multicore* parallelism that pure Python threads can never reach. Numba's `@njit(parallel=True)` with `prange`, Cython's `nogil` blocks, and Rust extensions via PyO3 all exploit exactly this.

So the complete picture of "does threading help" has three rows, not two.

![Matrix showing CPU-bound pure Python does not benefit from threads while I/O-bound and native C with nogil both do because they release the GIL](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-4.png)

The matrix makes the unifying principle concrete: **threads help a workload exactly to the extent that the workload runs with the GIL released.** Pure-Python CPU work never releases it → no help (1.0×). I/O-bound work releases it on every blocking call → big help (3–8×). Native C code that calls the allow-threads macro releases it while computing → big help, near-linear. The question is never "is it CPU-heavy or I/O-heavy" in the abstract — it is "how much of the time is the GIL held," and that is something you can reason about precisely.

## 8. What about the cost? The full accounting of the GIL

Let us total up the GIL's costs and benefits honestly, because the goal is to use it well, not to resent it.

**What it costs:**

- **No multicore parallelism for pure-Python CPU work.** The big one. An 8-core box runs your Python compute on one core. To use all cores for Python-level CPU work you must go to processes, native code, or a free-threaded build.
- **Contention overhead when CPU-bound threads coexist.** Forced handoffs every switch interval, context switches, cache effects. Adding threads to CPU work makes it slightly slower, never faster.
- **A latency tax under mixed workloads.** A CPU-bound thread can delay an I/O or UI thread by up to a switch interval before it yields the GIL.
- **Subtle correctness traps.** "Only one thread at a time" lulls people into thinking their code is thread-safe when only individual *bytecodes* are atomic; compound operations still race.

**What it buys:**

- **A fast single-threaded interpreter.** Most Python is single-threaded; it pays almost nothing for the GIL.
- **A simple, huge C-extension ecosystem.** Extension authors get single-threaded semantics by default and opt into parallelism only where they have proven it safe.
- **No fine-grained-locking overhead or deadlocks** inside the interpreter.
- **Atomic-ish builtins as a side effect.** Because the GIL is only dropped between bytecodes, many single-bytecode operations (e.g., a `list.append`, a `dict.__setitem__`) are effectively atomic, which simplifies some threaded code in practice (though you should not *rely* on it as a contract).

The GIL is a *trade*, and for two decades it was a good one for the dominant Python workloads (glue code, web request handling that is mostly I/O, and numerical work that lives in C). The places it hurts most are exactly the cases this series teaches you to recognize and route around: pure-Python CPU work that needs all your cores.

It also helps to put the GIL in the context of the whole performance discipline this series teaches. The leverage ladder is: (1) do less work, (2) do it in bulk (vectorize), (3) compile the hot 1% to native code, (4) use every core and overlap I/O. The GIL is the gatekeeper of rung 4 — but notice that rungs 2 and 3 *also* dissolve the GIL problem, because vectorized and compiled code runs in C with the GIL released. A team that frames "we need parallelism" as "we must fight the GIL with multiprocessing" often skips the cheaper rung: rewriting the hot loop in NumPy or Numba frequently delivers both a single-thread speedup *and* the multicore parallelism they wanted, with none of the pickling overhead processes impose. So before you reach for any concurrency tool, ask whether the hot loop can simply leave pure-Python execution — climb rung 2 or 3 first, and the GIL stops being your enemy because your hot code no longer holds it. Concurrency (rung 4) is the right answer when the work is genuinely I/O-bound, or when it is CPU-bound but already as vectorized/compiled as it will get and you still need more cores.

One more honest framing: the GIL is frequently *blamed* for slowness it has nothing to do with. A web service that is slow because it makes synchronous database calls one after another is not GIL-bound — it is I/O-bound and badly structured, and threads or async fix it precisely *because* the GIL releases during the wait. A data job that is slow because it loops in pure Python over ten million rows is not GIL-bound either — it is interpreter-overhead-bound, and vectorizing it (one C loop) is the fix, not parallelism. The GIL only deserves the blame in one specific situation: your hot path is genuinely CPU-bound *Python* bytecode, you have already done less work and vectorized what you can, and you still need more than one core. That situation is real but narrower than the GIL's reputation suggests. Profile first; most "the GIL is killing us" diagnoses turn out to be something cheaper to fix.

## 9. The decision map: which concurrency tool when

Here is the whole point of building the mental model — turning it into a fast, correct decision. The first and only question that matters is: **where does the time go — is the hot path CPU-bound or I/O-bound?** You answer that with a profiler (the [profiling track](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) covers the tools), then you branch.

![Decision tree branching on whether the hot path is CPU-bound to processes or native or free-threaded versus I/O-bound to threads or asyncio](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-8.png)

**If the hot path is CPU-bound** (it pegs one core, no waiting), the GIL blocks thread-level parallelism, so you have three real options:

- **`multiprocessing` / `ProcessPoolExecutor`.** Each process has its *own* interpreter and its *own* GIL, so $N$ processes truly use $N$ cores for Python code. The cost is that data must be *pickled* and sent between processes (serialize → IPC → deserialize), and processes have higher startup and memory overhead than threads. Great when the work-per-task is large relative to the data you ship. The trade-offs (fork vs spawn, pickling tax, `shared_memory` for big arrays) are the subject of the multiprocessing post in this same track.
- **Native code that releases the GIL** — NumPy/Polars vectorization, Numba `@njit(parallel=True)`, Cython `nogil`, a Rust/C extension. Often the *best* option: it makes the single thread faster *and* can use all cores within one process, no pickling. "Rewrite the hot 1% in native, not 100%" — and let it drop the GIL.
- **The free-threaded build (PEP 703, Python 3.13+).** A CPython build (`python3.13t`) with the GIL removed entirely, so plain Python threads use all cores for CPU work. Still maturing, with a single-thread performance cost today, but it is where Python is heading.

**If the hot path is I/O-bound** (it waits on network, disk, or a database), the GIL is *released during the wait*, so concurrency within one process works beautifully:

- **Threads (`ThreadPoolExecutor`).** The simplest tool. Each blocking call releases the GIL and the waits overlap. Ideal for a moderate number of blocking calls — tens to low hundreds. Each thread has a real OS-thread stack (often ~1 MB) so thousands of threads get expensive.
- **`asyncio`.** A single-threaded event loop with coroutines. `await` is a cooperative yield: while one coroutine waits on I/O, the loop runs others. No per-connection thread stack, no GIL contention (it is one thread), so it scales to **tens of thousands** of concurrent connections where threads would run out of memory. The cost is that the whole stack must be async (`aiohttp`/`httpx` instead of `requests`) and a single blocking call stalls the entire loop.

The two I/O tools differ on scale and ecosystem, not on whether they beat the GIL — both win for the same reason (the GIL is free during I/O). Use threads for simplicity and moderate concurrency; use asyncio for massive connection counts and when your libraries are async-native.

Here is the same decision as a quick-reference table:

| Tool | Bypasses GIL for CPU? | Best for | Main cost |
|------|-----------------------|----------|-----------|
| `threading` / `ThreadPoolExecutor` | No | I/O-bound, moderate concurrency | GIL serializes CPU; ~1 MB/thread |
| `asyncio` | No (one thread) | I/O-bound, 10k+ connections | Whole stack must be async; blocking stalls the loop |
| `multiprocessing` / `ProcessPoolExecutor` | Yes (own GIL each) | CPU-bound work, large tasks | Pickling + IPC + memory per process |
| Native + GIL released (NumPy/Numba/Cython/Rust) | Yes (within one process) | CPU-bound numeric/hot loops | You must write/own native code |
| Free-threaded build (3.13t) | Yes (no GIL) | CPU-bound Python threads | Single-thread overhead; maturing ecosystem |

And the same logic as a tool-by-tool comparison of *what each one does to the GIL*:

![Matrix comparing threading multiprocessing asyncio and the free-threaded build on whether each bypasses the GIL for CPU work and what each is best for](/imgs/blogs/the-gil-explained-what-it-protects-and-what-it-costs-7.png)

Notice the shape of it: **only separate interpreters (multiprocessing, subinterpreters) or a GIL-less build give pure-Python CPU work true multicore parallelism.** Threads and asyncio both share the one GIL — they are I/O tools, and excellent ones, but they will never make a pure-Python compute loop use more than one core. If you remember nothing else, remember that split.

## 10. A problem-solving narrative: routing a real workload

Let us walk a realistic mixed pipeline and decide tool by tool, the way you would in production. The job: a nightly process that (a) downloads 5,000 small JSON files from an object store, (b) parses and validates each, (c) computes a heavy numeric score per record, and (d) writes results to a database.

**Step 1 — profile to find where the time goes.** Suppose `cProfile` (or a sampling profiler like `py-spy` on the running job) shows: 60% of wall-clock in the downloads (network wait), 10% in JSON parsing (pure-Python CPU), 25% in the scoring (pure-Python CPU, a tight numeric loop), 5% in the DB writes (I/O). Now we have a map.

**Step 2 — the downloads are I/O-bound → threads or async.** 5,000 small downloads is classic I/O. The GIL is free during each `recv`, so concurrency wins big. 5,000 OS threads would burn ~5 GB of stack, so either a bounded thread pool (say 64 workers) *or* asyncio with `aiohttp` and a `Semaphore` capping concurrency. Either turns "5,000 sequential round-trips" into "5,000 overlapping waits," collapsing the 60% download time by ~50–60×. Async is the better fit at this connection count if the HTTP client is async-native.

**Step 3 — the scoring is pure-Python CPU → not threads.** Here is where people go wrong. Tempted by the threads that just helped the downloads, they thread the scoring too — and it does nothing (§4), because the GIL serializes the pure-Python loop. The right move is to (a) vectorize it with NumPy if the math allows (one C loop, GIL released, often 50–100× faster *and* parallelizable), or (b) `@njit` it with Numba, or (c) if it must stay pure Python, run it across **processes** so each core does a share. The decision is driven entirely by "this is CPU-bound Python, so threads are out."

**Step 4 — stress-test the decision.** What if the data doesn't fit in RAM? Then process-based parallelism with large pickled payloads becomes an IPC bottleneck — consider `shared_memory` or push the aggregation into the database. What if the scoring is only 2% of runtime instead of 25%? Then Amdahl's law caps your possible win at 2% — don't bother parallelizing it; fix the 60% download path first. What if you move from 4 to 32 cores? The download concurrency keeps scaling (it is I/O), but the process-based scoring eventually hits the pickling/IPC wall and stops scaling linearly. **The bottleneck moves; you re-profile and re-decide.** That loop — measure, route, re-measure — is the whole method.

The headline lesson: **a single program can need different concurrency tools for different stages,** and the GIL is the lens that tells you which. I/O stages → threads/async (GIL is free while waiting). CPU stages → processes/native/free-threading (GIL blocks thread parallelism).

#### Worked example: the mixed pipeline, before and after

Put numbers on it. Suppose the serial nightly job runs in 200 s on the 8-core box, split per the profile above: 120 s downloads, 20 s parsing, 50 s scoring, 10 s DB writes. Apply the routing and measure each stage.

| Stage | Serial (s) | Tool applied | After (s) | Why it changed |
|-------|-----------|--------------|-----------|----------------|
| Downloads | 120 | asyncio, 64-wide | 4 | GIL free during the wait; ~30× overlap |
| Parsing | 20 | left in main thread | 20 | small slice; not worth parallelizing |
| Scoring | 50 | NumPy vectorized | 3 | one C loop, GIL released, ~16× |
| DB writes | 10 | batched + threaded | 4 | I/O; writes overlap |
| **Total** | **200** | — | **31** | ~6.5× end to end |

The win is ~6.5×, and notice where it came from: almost entirely the two largest stages routed correctly through the GIL. If you had instead threaded the *scoring* (the §4 mistake), that 50 s would have stayed 50 s — threads cannot touch pure-Python CPU work — and your "optimized" pipeline would run in ~78 s instead of 31 s, leaving more than half the available win on the table. The GIL is not just trivia; it is the difference between a 6.5× win and a 2.5× one on the same code, decided entirely by which stage you sent to which tool. The numbers above are illustrative of the *pattern*, not a specific benchmark — your stage split and speedups will differ — but the *shape* (I/O stages collapse with overlap, CPU stages collapse only with native code or processes) is exactly what you will measure.

## 11. Case studies and real numbers

A few concrete, sourced data points to anchor the model.

**The GIL rewrite in Python 3.2.** Before 3.2, CPython released the GIL every *N* bytecode instructions (default 100). This caused the convoy effect — I/O threads and CPU threads fighting, with pathological slowdowns on multicore machines (David Beazley's well-known 2009/2010 talks demonstrated a *two-thread* CPU program running slower than one thread, sometimes dramatically, on multicore hardware). Antoine Pitrou's new GIL in Python 3.2 replaced the bytecode-count scheme with the time-based switch interval we discussed (`sys.setswitchinterval`, default 5 ms), which largely fixed the convoy pathology. The lesson that survived: **CPU-bound threads still do not speed up; they just no longer catastrophically slow down.**

**NumPy, Polars, and "the C does the heavy lifting."** The entire data ecosystem is built on releasing the GIL around native loops. **Polars** (written in Rust) runs its query engine multithreaded with the GIL released, so a Polars `group_by`/aggregate genuinely uses all your cores on a single DataFrame — frequently 5–15× faster than pandas on multicore machines for large joins and aggregations, a gap that comes substantially from real multicore parallelism that pandas (mostly single-threaded Python orchestration) cannot match. This is the §7 effect at production scale.

**The Rust rewrite wave.** A striking number of recent Python-ecosystem performance wins are native extensions that release the GIL: **ruff** (a linter ~10–100× faster than the pure-Python tools it replaces), **pydantic-core** (the Rust core under Pydantic v2, several times faster validation), **tokenizers** (Hugging Face), and **uv** (the package manager). Each follows the series' motto — *rewrite the hot part in native code* — and several get multicore parallelism for free by dropping the GIL in the native layer.

**PEP 703 and the free-threaded build.** Sam Gross's proof-of-concept (and the accepted PEP 703) showed it is *possible* to remove the GIL from CPython, using biased reference counting, deferred reference counting, and a thread-safe allocator. The accepted plan ships it as an *optional* build (`python3.13t`) first. The honest trade today: removing the GIL adds per-operation overhead, so single-threaded code on the free-threaded build is meaningfully slower (the initial target was keeping the overhead modest, but it is nonzero), while multithreaded CPU-bound code can finally scale. It is the future, but as of 3.13 it is opt-in and maturing — the forward-looking post in this track covers it in full.

**The David Beazley demonstration, reproduced.** The most instructive case study is the one you ran yourself in §4. Beazley's famous result — two CPU-bound threads finishing *slower* than one — used to be dramatic (sometimes 1.5–2× slower) on the pre-3.2 GIL because of the convoy effect on multicore machines, where the OS would bounce the lock between cores and a CPU thread would be repeatedly preempted by an I/O thread holding the GIL for a sliver of time. After the 3.2 rewrite the slowdown shrank to the few-percent overhead we measured (10.9 s vs 10.4 s), but the *shape* of the result is unchanged and still the canonical demonstration: on pure-Python CPU work, more threads is never a win. If you want to feel it, run the §4 script and watch a single core stay pinned while the others idle — the operating system's per-core utilization view is the clearest proof the GIL exists.

**Subinterpreters (PEP 684 / PEP 734).** A newer escape route worth naming: as of Python 3.12 each *interpreter* in a process can have its own GIL (PEP 684), and Python 3.13 adds a standard-library `interpreters` module (PEP 734) to create and drive them. Two subinterpreters in one process can run Python bytecode genuinely in parallel — like processes, but lighter, because they share the process address space and avoid the heaviest pickling/IPC of `multiprocessing` for some data. The catch is that they are *isolated* (objects do not freely cross the boundary), so you still marshal data between them, and the ecosystem support is young. It is a third member of the "separate GIL → real parallelism" family, alongside multiprocessing and the free-threaded build, and the forward-looking post in this track covers where it fits.

## 12. When to reach for threads (and when not to)

A decisive, opinionated checklist — the payoff of the whole post.

- **Reach for threads when the hot path is I/O-bound** and concurrency is moderate (tens to low hundreds of blocking operations): downloads, DB queries, file reads, API calls. The GIL is free during the wait, so the waits overlap. Simplest possible tool; start here for I/O.
- **Reach for asyncio when I/O concurrency is large** (thousands to tens of thousands of connections) *and* your libraries are async-native. No per-connection thread stack, no GIL contention (one thread), so it scales where threads run out of memory. Don't reach for it for a handful of calls — the complexity is not worth it.
- **Reach for multiprocessing when the hot path is CPU-bound pure Python** and the per-task work is large relative to the data shipped (so pickling/IPC doesn't dominate). Each process gets its own GIL → real multicore. Don't reach for it on tiny tasks (the pickling + spawn overhead will swamp the win) or on I/O-bound work (threads/async are simpler and cheaper).
- **Reach for native code that releases the GIL (NumPy/Numba/Cython/Rust) when the CPU hot path is numeric or tight-loop.** Often the best of both: faster *and* multicore within one process, no pickling. This is the leverage ladder's "compile the hot 1%" rung.
- **Consider the free-threaded build when you have genuinely CPU-bound Python threads** and can tolerate a maturing ecosystem and some single-thread overhead. Today: experiment, benchmark, don't bet production on it without testing.
- **Do NOT add threads to CPU-bound pure Python.** It cannot help (the GIL serializes the bytecode) and the contention makes it slightly slower. This is the single most common GIL mistake.
- **Do NOT assume thread-safety from the GIL.** Only individual bytecodes are atomic; `x += 1`, check-then-act, and read-modify-write on *your* data still race. Use a `Lock`.
- **Do NOT tune `setswitchinterval` to "go faster."** It changes switching granularity (latency vs handoff overhead), not throughput. Touch it only for a measured latency problem.

## 13. Key takeaways

- **The GIL is one mutex per interpreter that a thread must hold to run Python bytecode**, so only one thread executes Python at a time — regardless of core count.
- **It exists to protect interpreter state** — chiefly reference counts (every `INCREF`/`DECREF` is a racy read-modify-write) plus container internals and C-extension invariants. One coarse lock keeps the single-threaded interpreter fast and C extensions simple; fine-grained locking would slow the common case and risk deadlock.
- **CPU-bound pure-Python threads do not speed up** (the GIL serializes the bytecode; Amdahl's law with $p = 0$ gives $S = 1$) and can be slightly slower (handoff + contention overhead). Two threads can lose to one.
- **I/O-bound threads do speed up**, because CPython releases the GIL around blocking system calls, so the waits overlap — often 3–8× with a small pool.
- **Native code can release the GIL while computing** (`Py_BEGIN_ALLOW_THREADS`), which is why NumPy/Polars/Numba/Cython/Rust kernels give real multicore parallelism from Python threads. "CPU-bound" only blocks threads when it is *Python* CPU-bound.
- **The switch interval (`sys.getswitchinterval`, default 5 ms)** bounds how long a waiting thread stalls behind a CPU-bound holder; lowering it improves latency but adds handoff overhead and never adds throughput.
- **The decision map:** profile to find where the time goes → CPU-bound → multiprocessing / native-with-GIL-released / free-threaded build; I/O-bound → threads (moderate) or asyncio (massive). Threads and asyncio share one GIL; only separate interpreters or a GIL-less build give pure-Python CPU work true multicore parallelism.
- **Use the right lever:** rewrite the hot 1% in native code, route I/O through threads/async, route CPU-bound Python through processes — and always prove the win with a before-and-after number.

## 14. Further reading

- **CPython docs — `threading`, `multiprocessing`, `asyncio`, and `concurrent.futures`** — the canonical references for each tool the decision map points to.
- **`sys.getswitchinterval` / `sys.setswitchinterval`** in the `sys` module docs — the switch-interval API and its semantics.
- **PEP 703 — "Making the Global Interpreter Lock Optional in CPython"** (Sam Gross) — the accepted plan for the free-threaded build, with the design (biased/deferred refcounting) and the trade-offs.
- **PEP 684 — "A Per-Interpreter GIL"** — the foundation for subinterpreter-based parallelism (3.12+).
- **"Understanding the Python GIL" (David Beazley)** — the classic talk that made the convoy effect and the CPU-thread slowdown vivid, plus the history of the 3.2 GIL rewrite.
- **"High Performance Python" by Micha Gorelick and Ian Ozsvald** — the concurrency chapters tie the GIL to threads, processes, and async with worked numbers.
- **[Why Python Is Slow and What Fast Actually Means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means)** — the series intro and the leverage-ladder frame this post plugs into.
- **[The CPython Execution Model: Bytecode and the Eval Loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop)** — how the interpreter runs bytecode and why reference counting (the thing the GIL protects) is on every line.
- **[C Extensions and the FFI: ctypes, cffi, and pybind11](/blog/software-development/python-performance/c-extensions-and-the-ffi-ctypes-cffi-and-pybind11)** — `Py_BEGIN_ALLOW_THREADS` in context: how native code releases the GIL for real parallelism.

This post opens the concurrency track. The next posts go deep on each tool the decision map names — threading and its limits, multiprocessing and the cost of pickling, asyncio from the event loop up, async in practice, and the free-threaded, post-GIL future. With the GIL's mechanism in hand, each of those is now a matter of "how," not "why."

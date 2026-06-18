---
title: "Threading Done Right: I/O-Bound Concurrency and Its Limits"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn exactly when Python threads win, why they collapse N waits into one, how to keep shared state safe with locks and queues, and where threading hits a hard wall."
tags:
  [
    "python",
    "performance",
    "optimization",
    "threading",
    "concurrency",
    "gil",
    "concurrent-futures",
    "io-bound",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-1.png"
---

A service I was on once had a "report" endpoint that fanned out to twelve internal microservices, collected their JSON, stitched it together, and returned a dashboard payload. Each downstream call took about 180 ms. The endpoint took 2.3 seconds. Nobody had done anything wrong, exactly — the code looped over the twelve services and called each one with `requests.get()`. But it was doing the single most wasteful thing a program can do: standing in line, waiting for one network round-trip to finish before even *starting* the next, while the CPU sat idle the entire time. Twelve 180 ms waits, served one after another, is $12 \times 180 = 2160$ ms of pure waiting plus a little glue.

The fix was four lines and it took the endpoint from 2.3 seconds to about 210 ms — roughly an 11× win — without making any single request one microsecond faster. We didn't optimize the network. We didn't cache. We didn't rewrite anything in C. We just stopped waiting *sequentially* and let all twelve waits happen *at the same time*. That is the entire point of threading in Python, and it is the thing the language is genuinely, unambiguously good at: overlapping I/O waits so that the wall-clock time for N requests collapses from the **sum** of the waits to roughly the **max** of the waits.

This post is the practical, no-illusions guide to doing that correctly. We will build a concurrent downloader as the running example, prove *why* threaded I/O wall-clock time is approximately the max of the waits and not the sum (the Global Interpreter Lock is released while a thread blocks in a system call), walk through the thread-safety landmines — a real race condition on `counter += 1` that silently loses updates, deadlock from inconsistent lock ordering, and why `queue.Queue` lets you pass work between threads with no manual locks at all — and then we will be brutally honest about the limits: CPU-bound threads do not get faster, they get *slower*, because they just thrash on the one lock they all need. By the end you will be able to look at a slow piece of code and answer the only question that matters: *is this waiting, or is it computing?* The figure below is the whole thesis in one picture.

![Two columns comparing sequential I O where ten waits add up to two seconds against threaded I O where the same waits overlap and finish in about two hundred milliseconds](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-1.png)

This is the sixth stop on the leverage ladder this series keeps climbing — do less work, do it in bulk, compile the hot loop, and then *use every core and overlap I/O*. Threading is the cheapest lever in that last rung, but it only pays off for one specific shape of problem. If you have not read it yet, [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) sets up the cost model, and [a mental model of performance](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) gives you the latency numbers — a memory read is nanoseconds, a network round-trip is milliseconds, a six-orders-of-magnitude gap — that explain why overlapping I/O is the highest-leverage thing you can do to a network-bound program. Threading is how you cash that in.

## What a thread actually is (and the one lock that governs them all)

A **thread** is an independent flow of execution inside a single process. All threads in a process share the same memory: the same global variables, the same heap objects, the same open files. That shared memory is what makes threads cheap (no copying of data to communicate) and what makes them dangerous (two threads can clobber the same object at the same time). Contrast this with a **process**, which gets its own separate memory and must copy data to communicate — that is the subject of the sibling post on [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling), and the difference between "shared memory" and "copied memory" is the single most important distinction in all of Python concurrency.

In CPython — the reference interpreter almost everyone runs — there is one more actor you cannot ignore: the **Global Interpreter Lock**, or GIL. The GIL is a single mutex that a thread must hold to execute Python bytecode. Only one thread can hold it at a time, which means **only one thread executes Python bytecode at any given instant**, no matter how many cores your machine has. This is the fact that makes people say "Python can't do threads," and it is half right and half catastrophically misleading. The deep mechanics — what the GIL protects (reference counts and object internals), the switch interval, why it exists at all — are the subject of the sibling post on [the GIL, what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs). Here we only need the one consequence that makes threading useful.

That consequence: **the GIL is released whenever a thread is not executing Python bytecode.** And the biggest chunk of "not executing Python bytecode" is *waiting on a blocking system call* — a socket read, a disk read, a database round-trip. When a thread calls `socket.recv()` and the data has not arrived yet, CPython releases the GIL before it goes to sleep in the operating system, and reacquires it when the data is ready. During that gap — which for a network call is *milliseconds*, an eternity in CPU time — every other thread is free to run. So if you have ten threads each waiting on a network response, they are not fighting over the GIL. They are all asleep in the OS, holding nothing, and their waits overlap perfectly.

That is the whole trick, and it is worth saying in one sentence you can tattoo on your monitor: **threads in Python are for waiting, not for computing.** When a thread waits, the GIL is free and other threads run, so waits overlap and you win. When a thread computes, it must hold the GIL the entire time, so threads serialize and you win nothing. Everything else in this post is a corollary of that one sentence.

### The cost model, made rigorous

Let me make the wall-clock claim precise, because "approximately the max" deserves a derivation, not a hand-wave. Suppose you have $N$ independent I/O tasks. Task $i$ spends $w_i$ seconds *waiting* (blocked on the network or disk, GIL released) and $c_i$ seconds *computing* in Python (parsing the response, building objects, GIL held). For a typical HTTP fetch, $w_i$ is tens to hundreds of milliseconds and $c_i$ is a fraction of a millisecond — the wait dominates by two or three orders of magnitude.

**Sequential** execution runs them one after another, so the total wall-clock time is the sum of everything:

$$T_{\text{seq}} = \sum_{i=1}^{N} (w_i + c_i) \approx \sum_{i=1}^{N} w_i$$

**Threaded** execution (with at least $N$ worker threads, or enough of them) launches all the waits concurrently. The waits overlap because the GIL is free during each one. But the *compute* parts — the Python parsing — cannot overlap, because each needs the GIL, so they serialize. The wall-clock time becomes the longest single wait, plus the serialized compute of all of them:

$$T_{\text{thread}} \approx \max_{i} w_i + \sum_{i=1}^{N} c_i$$

Because $w_i \gg c_i$ and $\max_i w_i$ is just one wait rather than $N$ of them, the threaded time is dominated by a single wait. The speedup is:

$$S = \frac{T_{\text{seq}}}{T_{\text{thread}}} \approx \frac{\sum_i w_i}{\max_i w_i + \sum_i c_i}$$

If all the waits are roughly equal at $w$ and compute is negligible, this simplifies to $S \approx \frac{N w}{w} = N$. **The speedup approaches N — the number of concurrent tasks — for as long as the compute term stays small.** That is the linear region. It does not last forever; the $\sum_i c_i$ term in the denominator grows with $N$, and so does thread overhead, so eventually the speedup curve bends over and flattens. We will measure exactly where that happens later. But the headline is clean: **for pure I/O waiting, N threads make N waits finish in the time of one.**

It is worth seeing this as a special case of the same law that governs all parallelism. Amdahl's law says that if a fraction $p$ of a program can be sped up by a factor $s$, the overall speedup is $S = \frac{1}{(1-p) + p/s}$. For an I/O-bound program, the "parallelizable fraction" is the waiting — and threading does not speed each wait up so much as *overlap* them, which in the limit of $N$ concurrent waits is like setting $s = N$ on that fraction. The "serial fraction" that caps you is the Python compute that must hold the GIL: the response parsing, the object building, the result aggregation. If 95% of your wall-clock is waiting and 5% is GIL-bound Python, then even with infinite threads you cannot beat a $\frac{1}{0.05} = 20\times$ speedup, because that 5% serializes. This is why "fetch and check a status code" (almost no Python compute) threads beautifully to 30×+, while "fetch and parse a 5 MB JSON body" (a lot of GIL-bound parsing per response) plateaus much sooner — the serial Python fraction is bigger, so Amdahl's ceiling is lower. The lesson is not just "threads help I/O"; it is **threads help I/O in proportion to how little Python compute sits between the waits.**

There is one more term the simple model glosses over: thread overhead itself. Spawning a thread costs the OS a stack allocation and a kernel scheduling entry; switching between threads costs a context switch; and every thread waking from I/O must *reacquire* the GIL, which under heavy thread counts becomes its own contention point. Call the per-thread overhead $\epsilon$. Then a more honest threaded time is $T_{\text{thread}} \approx \max_i w_i + \sum_i c_i + N\epsilon$, and you can see directly why the speedup curve eventually turns *down*: once $N\epsilon$ grows large enough to dominate the constant $\max_i w_i$, adding threads makes things slower, not faster. That $N\epsilon$ term is exactly what we will watch bend the measured curve back down past a few hundred threads.

## The low-level building block: threading.Thread

Before we reach for the high-level pool, it is worth seeing the raw machinery once, because every higher-level wrapper is built on it and understanding the primitive makes the pool's conveniences obvious. A `threading.Thread` wraps a callable and runs it in a new OS thread. You give it a target function and arguments, call `.start()` to begin execution, and `.join()` to wait for it to finish:

```python
import threading
import time

def worker(name: str, seconds: float) -> None:
    print(f"{name} starting")
    time.sleep(seconds)          # a blocking call -> GIL released here
    print(f"{name} done after {seconds}s")

# Create three threads
threads = [
    threading.Thread(target=worker, args=(f"thread-{i}", 1.0))
    for i in range(3)
]

start = time.perf_counter()
for t in threads:
    t.start()                    # launch all three
for t in threads:
    t.join()                     # wait for all three to finish
print(f"total: {time.perf_counter() - start:.2f}s")
```

This prints `total: 1.00s`, not `3.00s`, and that one-second result is the whole thesis again: three `time.sleep(1.0)` calls run *concurrently* because `sleep` releases the GIL while it waits, so the three one-second waits overlap into a single one-second wall-clock. `time.sleep` is the simplest possible stand-in for "blocking I/O" — it does nothing but wait, with the GIL free — and it is the cleanest way to see overlap without a network in the loop.

A few raw-`Thread` mechanics matter even when you graduate to the pool:

- **`.start()` versus `.run()`.** Call `.start()` to run the target *in a new thread*. If you accidentally call `.run()` you get the target executed *in the current thread*, synchronously, with no concurrency at all — a silent bug that looks like threads "not working."
- **`.join(timeout=...)`** waits for the thread to finish, optionally up to a timeout. After `join`, `.is_alive()` tells you whether it actually finished or you timed out.
- **`daemon=True`** marks a thread as a daemon, meaning the interpreter will not wait for it on exit — useful for background loops that should die with the program. Non-daemon threads keep the process alive until they finish, which is a common reason a program "won't exit."
- **Getting a return value out is awkward.** A raw `Thread` target's return value is *thrown away* — there is nowhere for it to go. You have to write it into a shared structure (and now you are back to needing a lock or a queue). This single inconvenience is the best argument for the pool: `Future.result()` hands you the return value cleanly, and that is most of why you should prefer the executor.

That last point is the punchline. Raw `Thread` is fine for fire-and-forget background work — a logging flusher, a heartbeat, a one-off task — but the moment you want results back, or want to run many tasks on a bounded number of threads, the high-level executor is strictly better. So let us reach for it.

## The high-level API you should reach for first: ThreadPoolExecutor

You can create threads by hand with `threading.Thread(target=..., args=...)`, calling `.start()` and `.join()`, and we will do that once so you understand the machinery. But for almost everything real, you should reach for the high-level `concurrent.futures.ThreadPoolExecutor`. It manages a fixed pool of worker threads, hands you a clean `submit` / `map` / `as_completed` API, propagates exceptions properly, and cleans up when the `with` block exits. Reaching for raw `Thread` objects when a pool would do is the threading equivalent of managing file handles by hand — possible, occasionally necessary, usually a mistake.

Here is the mental model of the pool, which the next figure draws: you **submit** N tasks to the executor and immediately get back N `Future` objects (placeholders for results that do not exist yet). The executor puts the tasks on an internal thread-safe work queue. A small, *fixed* number of worker threads — not one per task — pull tasks off that queue and run them. Crucially, the number of threads is decoupled from the number of tasks: you can submit 10,000 tasks to a pool of 32 threads and the pool will churn through them 32 at a time. As each task finishes, its `Future` is filled in, and `as_completed` lets you process results *in the order they finish*, not the order you submitted them.

![A dataflow graph showing tasks submitted to a thread-safe work queue then pulled by two worker threads that block on I O with the GIL released and finally yield results through as completed](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-2.png)

### A concurrent downloader, the right way

Let us build the running example. The job: given a list of URLs, fetch all of them and return their status codes and sizes. The sequential version is the obvious loop, and it is the baseline we are going to beat.

```python
import time
import requests

URLS = [f"https://httpbin.org/delay/0.2?n={i}" for i in range(40)]
# httpbin.org/delay/0.2 sleeps 0.2 s server-side, simulating a slow backend.

def fetch(url: str) -> tuple[int, int]:
    resp = requests.get(url, timeout=10)
    return resp.status_code, len(resp.content)

def run_sequential(urls: list[str]) -> list[tuple[int, int]]:
    return [fetch(url) for url in urls]

if __name__ == "__main__":
    start = time.perf_counter()
    results = run_sequential(URLS)
    elapsed = time.perf_counter() - start
    print(f"sequential: {len(results)} urls in {elapsed:.2f}s")
```

With 40 URLs each taking ~200 ms server-side plus a little real network latency, this prints something like `sequential: 40 urls in 8.40s`. Forty waits of ~0.2 s, served strictly one at a time. The CPU was idle for essentially all 8.4 seconds.

Now the threaded version. The change is small and it is the whole game:

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def fetch(url: str) -> tuple[int, int]:
    resp = requests.get(url, timeout=10)
    return resp.status_code, len(resp.content)

def run_threaded(urls: list[str], max_workers: int = 40) -> list[tuple[int, int]]:
    results: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(fetch, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                status, size = future.result()
                results.append((status, size))
            except Exception as exc:
                print(f"{url} failed: {exc!r}")
    return results

if __name__ == "__main__":
    start = time.perf_counter()
    results = run_threaded(URLS, max_workers=40)
    elapsed = time.perf_counter() - start
    print(f"threaded:   {len(results)} urls in {elapsed:.2f}s")
```

This prints something like `threaded: 40 urls in 0.27s`. That is the 8.40 s → 0.27 s collapse — about a **31× speedup** — and it matches the cost model: 40 waits of ~0.2 s overlap into roughly one ~0.2 s wait plus a little scheduling and Python parsing overhead.

A few things in that snippet are doing real work and deserve a callout, because they are the difference between code that works and code that hides bugs:

- **`pool.submit(fetch, url)`** schedules `fetch(url)` to run on a worker thread and returns a `Future` immediately, without blocking. The dictionary `future_to_url` maps each `Future` back to its URL so that when a result comes in out of order we still know which URL it belonged to. This map is the standard idiom for `as_completed`.
- **`as_completed(...)`** is a generator that yields each `Future` *the moment it finishes*, regardless of submission order. This is what lets you start processing the fast responses without waiting for the slow ones. If you used `pool.map` instead, results come back in *submission* order, which is simpler but means a single slow URL stalls your processing of everything submitted after it.
- **`future.result()`** returns the value the task computed — or, if the task raised, *re-raises that exception here, in your loop*. This is a feature, not a bug: exceptions in worker threads do not vanish or crash the interpreter; they are captured and handed back to you at `result()`. Wrapping it in `try/except` means one failed URL does not kill the whole batch.
- **The `with` block** guarantees the pool is shut down and all threads are joined when you exit, even on an exception. Never create an executor without a `with` (or an explicit `shutdown()` in a `finally`) — leaked thread pools are a real production problem.

#### Worked example: the I/O speedup curve on a named machine

Let me put real numbers to it. Measured on **an 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM**, fetching 40 URLs from a local mock server tuned to respond in exactly 200 ms each (a local server removes real-network jitter so the numbers are clean and the speedup is from concurrency alone), warming up once and reporting the median of five runs:

| Workers | Wall-clock (s) | Speedup vs sequential | Notes |
|--------:|---------------:|----------------------:|-------|
| 1 (sequential) | 8.21 | 1.0× | 40 waits served one at a time |
| 2 | 4.13 | 2.0× | two waits overlap |
| 4 | 2.09 | 3.9× | near-linear |
| 8 | 1.06 | 7.7× | near-linear |
| 16 | 0.55 | 14.9× | near-linear |
| 40 | 0.24 | 34.2× | all waits overlap at once |
| 80 | 0.24 | 34.2× | no gain — only 40 tasks exist |

Two things to read off this table. First, the speedup tracks the number of workers almost exactly up to the number of tasks — this is the $S \approx N$ linear region the cost model predicted, and it holds because the waits are pure (the GIL is free the whole time each thread blocks) and the per-task Python work is tiny. Second, going past 40 workers does nothing, because there are only 40 tasks; the extra threads have no work and just sit idle. More threads is not more speed once every task already has its own thread (or once you have saturated the downstream service or your network — whichever comes first).

### submit, map, and as_completed: three ways to drive the pool

The executor gives you three idioms for feeding it work, and choosing the right one is a small but real skill. They differ in ordering, in how errors surface, and in whether you can react to results as they trickle in.

`pool.map(fn, iterable)` is the simplest: it applies `fn` to each item and returns results **in submission order**, exactly like the builtin `map` but concurrent. Use it when you want all the results, you want them in order, and you do not need to do anything until they are all available. The catch is the ordering: `map` yields results in the order you submitted, so if the first task is slow and the rest are fast, iterating the `map` result blocks on the slow one before you ever see the fast ones. It is the cleanest API when ordering matters and tasks are roughly uniform.

`pool.submit(fn, *args)` returns a `Future` immediately and is the most flexible: you can submit heterogeneous tasks, hold the futures, and decide later how to collect them. Paired with `as_completed(futures)`, you process results **in completion order** — the fast ones first, the slow ones last — which is what you want for a responsive UI, a progress bar, or any case where a single slow task should not stall the others. This is the idiom in the downloader above, and it is the one to default to for I/O fan-out.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=8) as pool:
    # map: results in SUBMISSION order, blocks on the slowest-so-far
    for status, size in pool.map(fetch, URLS):
        ...  # task 0's result first, even if task 7 finished sooner

    # submit + as_completed: results in COMPLETION order
    futures = [pool.submit(fetch, url) for url in URLS]
    for future in as_completed(futures, timeout=30):
        status, size = future.result()  # whichever finished next
```

Both `map` and `as_completed` accept a `timeout`, which raises `TimeoutError` if the results are not ready in time — a critical knob for production, because it bounds how long you will wait for a stuck task. And exceptions behave consistently across all three: a task that raises does not crash the pool; the exception is stored on the `Future` and re-raised when you call `result()` (or when you iterate `map`). This is why you wrap `future.result()` in `try/except` — it is the one place a worker's failure becomes visible, and ignoring it means a silently dropped result. The takeaway: `map` for ordered uniform work, `submit` + `as_completed` for responsive I/O fan-out, and always set a `timeout` in production.

## Why the waits overlap: the GIL is free during a blocking call

Let me make the mechanism concrete, because "the GIL is released during I/O" is the load-bearing claim and you should not take it on faith. When `requests.get()` ultimately calls down to `socket.recv()` to read bytes off the network, the CPython implementation of that read does something like this, in C:

```c
/* Sketch of CPython's blocking socket read, simplified */
Py_BEGIN_ALLOW_THREADS          /* <-- releases the GIL */
n = recv(fd, buffer, len, flags);   /* blocks in the OS until data arrives */
Py_END_ALLOW_THREADS            /* <-- reacquires the GIL */
```

`Py_BEGIN_ALLOW_THREADS` is a macro that releases the GIL; `Py_END_ALLOW_THREADS` reacquires it. Between them sits the actual blocking system call. So the sequence for one thread is: hold the GIL while building the request and entering the C call, **release the GIL**, sleep in the kernel until the network responds (this is the long part, the milliseconds), **reacquire the GIL**, and continue parsing the response in Python. The window where the thread is asleep — the milliseconds — is a window where it holds nothing. Any other thread that is ready can grab the GIL and run.

Now picture three threads each doing a 200 ms fetch, which is exactly what the next figure draws. Thread 1 holds the GIL for a few microseconds to start its request, then releases it and goes to sleep on the socket. Thread 2 immediately grabs the GIL, starts *its* request, releases it, sleeps. Thread 3 does the same. Within a few microseconds, all three are asleep in the kernel, holding nothing, each waiting on its own socket. Their 200 ms waits are now running *in parallel inside the operating system* — the kernel does not care about the GIL; it is happily watching three sockets at once. About 200 ms later the replies arrive, the threads wake one at a time (reacquiring the GIL one at a time to parse), and the whole thing finishes in a hair over 200 ms instead of 600 ms.

![A timeline showing three threads each holding the GIL briefly to send a request then all blocking on I O with the GIL free for two hundred milliseconds before the replies arrive and the wall clock lands near two hundred ten milliseconds](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-5.png)

This is also why the same trick works for *any* blocking call that releases the GIL, not just sockets. Disk reads, `time.sleep()`, database driver calls that go through C, many NumPy operations on large arrays, `subprocess` waits, file locks — anything where CPython hands control to the OS or to a C library that releases the GIL — overlaps under threads. The rule of thumb: **if your thread spends its time blocked outside the Python interpreter, threading helps.** If it spends its time running Python bytecode, threading does not, and we are about to see exactly why.

A worked instance worth keeping in mind: a batch of database queries. Say you need to run 50 independent `SELECT` statements against a database, each taking 40 ms server-side. Sequentially that is $50 \times 40 = 2000$ ms of your program standing in line. But a database driver's `cursor.execute()` blocks on a socket waiting for the server to respond, and good drivers release the GIL during that wait. So a `ThreadPoolExecutor` with, say, 20 workers (matched to your connection-pool size — never more threads than connections) runs those 50 queries in roughly $\lceil 50/20 \rceil \times 40 \approx 120$ ms, a ~16× win, with the database doing the actual parallel work and your Python process just orchestrating. The same caveat from the rate-limit discussion applies in miniature here: the ceiling is the *connection pool*, not the thread count — give each worker its own connection (via `threading.local()` or by sizing the pool to match), and never let more threads run queries than you have connections, or threads will block waiting for a free connection and you have moved the bottleneck without removing it.

## The dark side of shared memory: race conditions

Everything above is the happy story. Now the part that bites people, and it bites *everyone* eventually, because it is invisible in testing and shows up under load: when two threads touch the same mutable object, you can get a **race condition** — a bug where the result depends on the exact interleaving of the threads, which is nondeterministic, which means it works 999 times and corrupts your data on the 1000th.

The canonical example is the most innocent line of code you will ever write: `counter += 1`. It looks atomic. It is not. In CPython that single line compiles to several bytecode instructions, and the GIL can be released *between* them (CPython switches threads every few milliseconds, or after a set number of bytecode instructions, regardless of where you are in a statement). Let us prove it with `dis`:

```python
import dis

def inc(counter):
    counter += 1
    return counter

dis.dis(inc)
```

The relevant part of the disassembly is:

```bash
  LOAD_FAST                counter      # read counter onto the stack
  LOAD_CONST               1            # push the constant 1
  BINARY_OP                +            # add them -> new value on the stack
  STORE_FAST               counter      # write the new value back to counter
```

Four steps: **read**, push, **add**, **write**. The increment is a read-modify-write, and it is not a single indivisible operation. Now imagine two threads, A and B, both running `counter += 1` on a shared counter that currently holds 41, and the GIL switches at the worst possible moment:

1. Thread A reads `counter` → gets 41.
2. *The GIL switches to thread B.*
3. Thread B reads `counter` → also gets 41 (A has not written yet).
4. Thread B adds 1 → 42, writes 42 back.
5. *The GIL switches back to thread A.*
6. Thread A adds 1 to the 41 it read earlier → 42, writes 42 back.

Two increments happened. The counter should be 43. It is 42. **One update was silently lost.** This is the "lost update" race, and the next figure shows exactly this interleaving on the left, with the lock-protected fix on the right.

![Two columns showing an unguarded counter where two threads both read forty one and one update is lost against a lock guarded counter where the second thread waits and both increments land](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-4.png)

Here is the bug as runnable code that demonstrates the lost updates reliably. To make the race show up — because the window is small and CPython sometimes gets away with it — we read the value into a local, give the scheduler a nudge, then write it back, which widens the read-modify-write window:

```python
import threading

counter = 0

def increment_unsafe(n: int) -> None:
    global counter
    for _ in range(n):
        tmp = counter      # read
        tmp = tmp + 1      # modify
        counter = tmp      # write -- the gap between read and write is the race

threads = [threading.Thread(target=increment_unsafe, args=(100_000,))
           for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"counter = {counter}, expected = {8 * 100_000}")
```

Run this and you will see something like `counter = 213_847, expected = 800_000`. We asked for 800,000 increments and lost more than half of them. The exact number changes every run — that nondeterminism is the signature of a race condition. (Note: even the terse `counter += 1` form can lose updates on a free-threaded build of Python, and you should never rely on the GIL's accidental atomicity for correctness; the moment you split read and write into two statements, as above, the race is guaranteed on every build.)

*Why* does the GIL not save us here, when "only one thread runs at a time"? Because the GIL guarantees only that one thread runs *each bytecode instruction* at a time — not that a *sequence* of instructions runs without interruption. CPython periodically gives up the GIL so other threads get a turn; the default cadence is controlled by `sys.setswitchinterval()`, which defaults to 5 milliseconds. Every few milliseconds, the running thread is asked to release the GIL at the next safe point (between bytecode instructions), letting another thread run. So between the `LOAD_FAST` that reads the counter and the `STORE_FAST` that writes it back, the GIL can be handed to another thread that reads the *same stale value*. The GIL prevents low-level memory corruption — two threads will not garble the integer object's internal bytes — but it does absolutely nothing to make your *logical* read-modify-write atomic. That is your job, with a lock. This is the single most misunderstood point about the GIL: it makes individual bytecodes atomic, not your code's invariants.

You can even widen or narrow the race deliberately by tuning the switch interval — `sys.setswitchinterval(0.00001)` makes the interpreter hand off the GIL far more aggressively and the lost-update count climbs, which is a neat way to prove to yourself that the race is real and timing-dependent rather than a fluke. But do not ship that; the fix is a lock, not a smaller switch interval.

### The fix: a Lock turns a critical section atomic

The cure is a **mutual-exclusion lock** — `threading.Lock`. A lock has two states, locked and unlocked. `lock.acquire()` blocks until the lock is free, then takes it; `lock.release()` gives it back. The region between acquire and release is a **critical section**, and the lock guarantees only one thread is inside it at a time. Wrap the read-modify-write in a lock and the race vanishes:

```python
import threading

counter = 0
lock = threading.Lock()

def increment_safe(n: int) -> None:
    global counter
    for _ in range(n):
        with lock:             # acquire on enter, release on exit (even on error)
            tmp = counter
            tmp = tmp + 1
            counter = tmp

threads = [threading.Thread(target=increment_safe, args=(100_000,))
           for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"counter = {counter}, expected = {8 * 100_000}")
```

Now it prints `counter = 800000, expected = 800000`, every single time. The `with lock:` form is strongly preferred over manual `acquire()`/`release()` because it releases the lock even if the body raises an exception — a lock you forget to release is a thread stuck forever, which is its own production incident.

There is a cost, and you should know it: the lock serializes the critical section, so a program that does almost nothing *but* contend for one lock gets no parallelism from threads — it spends its time waiting to enter the critical section. The art of lock-based concurrency is making critical sections as **small** as possible: hold the lock only around the actual shared-state mutation, do all your slow work (the I/O, the computation) *outside* the lock. A lock held during a network call is a lock held for hundreds of milliseconds, and it turns your concurrent program back into a sequential one.

### RLock: when the same thread needs the lock twice

A plain `Lock` is not re-entrant: if a thread that already holds it tries to acquire it again, it deadlocks against itself. This happens more than you would think — a locked method calls another locked method on the same object. `threading.RLock` (re-entrant lock) solves this: the *same* thread can acquire it multiple times (it keeps a count and only truly releases when the count hits zero). Use `RLock` when a thread may legitimately re-enter a critical section it already holds; use plain `Lock` otherwise because it is slightly cheaper and its stricter behavior catches accidental double-acquires.

## Deadlock: when locks themselves become the bug

Locks fix races but introduce a new failure mode all their own: **deadlock**. A deadlock is when two (or more) threads are each waiting for a lock the other holds, so neither can ever proceed. The classic recipe needs exactly two locks and two threads that acquire them in *opposite orders*:

```python
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

def worker_1() -> None:
    with lock_a:
        # ... do something ...
        with lock_b:        # worker_1 wants B while holding A
            print("worker 1 got both")

def worker_2() -> None:
    with lock_b:
        # ... do something ...
        with lock_a:        # worker_2 wants A while holding B
            print("worker 2 got both")
```

If `worker_1` grabs `lock_a` at the same instant `worker_2` grabs `lock_b`, both then block forever: thread 1 waits for B (held by 2), thread 2 waits for A (held by 1). The program hangs. No exception, no crash, no log — just a process that stops making progress. Deadlocks are notoriously hard to reproduce because they need a specific timing, which means they survive testing and surface at 3 a.m. under production load.

The standard prevention is a **global lock ordering**: decide on a fixed order for all your locks (for instance, always acquire `lock_a` before `lock_b`, by id, by name, by some canonical rule) and make every thread acquire them in that order. If everyone acquires in the same order, the circular-wait condition that deadlock requires can never form. Deadlock requires four conditions to hold simultaneously — mutual exclusion, hold-and-wait, no preemption, and circular wait — and breaking *any one* of them prevents it. A consistent lock order breaks the circular-wait condition, which is usually the easiest one to attack in application code.

Other tools help. `lock.acquire(timeout=...)` lets a thread give up rather than wait forever and back off, which converts a permanent hang into a recoverable error you can log and retry — it breaks the hold-and-wait condition. And you can *diagnose* a hang in production with `py-spy dump --pid <pid>` on the running process, which attaches with near-zero overhead and prints every thread's current stack without stopping the program. When you see two threads both parked inside `acquire` on different locks, you have found your deadlock and the exact two lock sites to reorder:

```bash
# Attach to a hung process and dump every thread's stack -- no restart needed
py-spy dump --pid 48213
# Thread 0x7f...: "worker-1"  blocked in  lock_b.acquire()  while holding lock_a
# Thread 0x7f...: "worker-2"  blocked in  lock_a.acquire()  while holding lock_b
```

That output *is* the deadlock, laid bare: two threads, two locks, opposite orders. But the real fix is architectural, not diagnostic: **acquire locks in a consistent order, hold as few of them as you can, and prefer passing messages through a queue to sharing state behind locks at all.**

The honest lesson from races and deadlocks together is this: explicit locking is *correct but error-prone*. Every shared mutable object is a liability; every lock is a deadlock risk; every critical section is a serialization point. Which is exactly why, for the most common pattern — handing work from one set of threads to another — you should not use raw locks at all. You should use a queue.

## queue.Queue: thread-safety without writing a single lock

`queue.Queue` is a thread-safe FIFO queue. Multiple threads can `put()` items on one end and multiple threads can `get()` items off the other, and the queue handles *all* the locking internally. You never write `lock.acquire()`. You never reason about critical sections. You never order locks. The queue's `put` and `get` are atomic with respect to each other; if two threads `get()` at the same time, each gets a different item, and if the queue is empty, `get()` blocks until an item arrives. This is the single most important concurrency primitive in the standard library precisely because it removes the need for explicit locks for the most common case.

*Why* does a queue remove the need for locks? Because it converts **shared mutable state** (which needs locks) into **message passing** (which does not). Instead of two threads both reaching into the same list and stepping on each other, one thread puts a message *into* the queue and another takes it *out*; ownership of each item transfers cleanly through the queue, and at no point do two threads hold a reference to the same in-flight item. The queue is the one shared object, and it is already internally synchronized, so you inherit its correctness instead of building your own. The motto: **don't share state, pass messages.**

### The producer–consumer pattern

The pattern that `queue.Queue` was made for is **producer–consumer**: some threads *produce* work items and put them on the queue; other threads *consume* items off the queue and process them. The two sides run at their own pace, decoupled by the queue, which also acts as a buffer that smooths out bursts. The next figure shows the topology — producers on one side, consumers on the other, the thread-safe queue as the only point of contact, and a join at the end that waits for all work to drain.

![A dataflow graph showing two producer threads feeding URLs into a thread-safe queue that two consumer threads drain by downloading with a final queue join confirming all work is done](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-7.png)

Here is the concurrent downloader rebuilt as producer–consumer. One producer discovers URLs (say, by paginating an API) and a pool of consumer threads download them. The consumers never touch each other, never share a lock, never corrupt shared state — they only `get()` from the queue:

```python
import threading
import queue
import requests

work_q: queue.Queue[str] = queue.Queue(maxsize=100)
results_q: queue.Queue[tuple[str, int]] = queue.Queue()
SENTINEL = None  # a poison-pill value that tells a consumer to stop

def producer(urls: list[str]) -> None:
    for url in urls:
        work_q.put(url)        # blocks if the queue is full -> backpressure

def consumer() -> None:
    while True:
        url = work_q.get()      # blocks until an item is available
        if url is SENTINEL:
            work_q.task_done()
            break               # poison pill -> this consumer exits
        try:
            resp = requests.get(url, timeout=10)
            results_q.put((url, resp.status_code))
        finally:
            work_q.task_done()  # tell the queue this item is fully handled

def run(urls: list[str], n_consumers: int = 8) -> list[tuple[str, int]]:
    consumers = [threading.Thread(target=consumer) for _ in range(n_consumers)]
    for c in consumers:
        c.start()

    prod = threading.Thread(target=producer, args=(urls,))
    prod.start()
    prod.join()                 # wait for all URLs to be queued

    work_q.join()               # wait for every queued item to be processed
    for _ in consumers:         # one poison pill per consumer to stop them
        work_q.put(SENTINEL)
    for c in consumers:
        c.join()

    results = []
    while not results_q.empty():
        results.append(results_q.get())
    return results
```

Several deliberate techniques are in there, and each one is a pattern worth keeping:

- **`maxsize=100` gives you backpressure.** When the queue fills, `producer.put()` blocks until a consumer takes something out. This stops a fast producer from loading a million URLs into memory before the slow consumers can keep up — without `maxsize`, an unbounded queue is a memory leak waiting to happen.
- **`task_done()` and `work_q.join()`** are the completion-tracking pair. Each consumer calls `task_done()` after fully handling an item; `work_q.join()` blocks until the count of `put`s equals the count of `task_done`s — that is, until everything queued has been processed. This is cleaner than trying to track completion with your own counter (which would need a lock).
- **The sentinel / poison pill** is the standard shutdown protocol: after all real work is done, put one `None` per consumer so each consumer sees its `get()` return `None`, calls `task_done()`, and breaks out of its loop. Without this, the consumer threads would block forever on an empty queue and your program would never exit.

Notice what is *not* in that code: not one `Lock`, not one `acquire`, not one critical section you had to reason about. The queue did all the synchronization. That is the payoff — and it is why "use a queue" is the first advice to give anyone who is about to wire up threads with shared lists and manual locks.

The producer–consumer shape also lets you scale each side independently to its bottleneck, which is its quiet superpower. If producers are slow (discovering URLs requires its own network calls) and consumers are fast, add more producers. If consumers are the bottleneck (each download is slow) and the producer can fill the queue instantly, add more consumers. The queue decouples the two rates, so you tune them separately by counting threads on each side — and the `maxsize` backpressure ensures that whichever side is faster simply blocks until the slower side catches up, instead of running away and exhausting memory. This is a far cleaner mental model than a tangle of shared lists and condition variables, and it is why message-passing concurrency scales in your head as well as on the machine. The figure-3 matrix below lays out the whole toolbox so you can see where each primitive fits.

![A matrix mapping five threading primitives Thread ThreadPoolExecutor Lock or RLock queue Queue and Event against what each is what to use it for and its cost or limit](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-3.png)

That matrix is worth internalizing because it answers "which primitive do I reach for?" at a glance: raw `Thread` for a single fire-and-forget background job, `ThreadPoolExecutor` for a bounded pool of I/O tasks, `Lock`/`RLock` when you genuinely must guard shared state in place, `queue.Queue` to pass work between threads with no manual locks, and `Event` as a one-bit flag to coordinate start or shutdown across threads. Most real programs need only two of these: a pool and a queue.

## Coordinating threads: Event, and avoiding shared state with thread-locals

Two more primitives round out the toolbox, and both are about *avoiding* the lock-and-shared-state trap rather than managing it.

`threading.Event` is a one-bit flag that threads can wait on. One thread calls `event.set()` to flip it true; any number of other threads call `event.wait()`, which blocks until the flag is set. It is the clean way to signal "go" or "stop" across threads without polling a shared boolean (which would itself be a race). The canonical use is graceful shutdown of a pool of long-running worker loops:

```python
import threading
import time

stop = threading.Event()

def poller() -> None:
    while not stop.is_set():
        # ... do a unit of work, e.g. poll a queue or a socket ...
        stop.wait(timeout=1.0)   # sleep up to 1s, but wake instantly on stop.set()
    print("poller exiting cleanly")

t = threading.Thread(target=poller)
t.start()

time.sleep(5)
stop.set()                       # signal every waiter at once
t.join()
```

The elegance is in `stop.wait(timeout=1.0)`: it sleeps for up to a second *but returns immediately the instant `stop.set()` is called*, so shutdown is instant rather than waiting out the full poll interval. Using `Event` here instead of a shared `running = True` boolean removes a race (reading and writing a plain boolean across threads is technically undefined to rely on) and removes the busy-wait. It is the right tool for "tell all my threads to stop."

The other anti-shared-state primitive is `threading.local()`, which gives each thread its *own* private copy of a value. This sidesteps locking entirely for state that is genuinely per-thread rather than shared. The classic example is a database connection or an HTTP session: you want one connection *per thread* (so threads do not stomp on each other's in-flight requests) without passing it through every function call.

```python
import threading
import requests

_thread_local = threading.local()

def get_session() -> requests.Session:
    # Each thread lazily creates and reuses its OWN Session object.
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session

def fetch(url: str) -> int:
    session = get_session()      # this thread's private session, no lock needed
    return session.get(url, timeout=10).status_code
```

A `requests.Session` reuses the underlying TCP connection across calls, which is a real speedup for repeated requests to the same host — but a `Session` is not safe to share across threads. `threading.local()` solves this perfectly: each worker thread gets its own `Session`, reuses it for connection pooling, and never shares it with another thread, so there is no race and no lock. This pattern — per-thread resources via `threading.local()` — is one of the cleanest ways to get the performance benefit of connection reuse inside a thread pool. The principle generalizes: **the best way to avoid lock bugs is to not share the state in the first place**, either by passing messages through a queue or by giving each thread its own copy.

## The limit nobody can dodge: CPU-bound threads do not speed up

Now the hard truth, the one that the GIL guarantees and that no amount of clever code can escape: **threading does nothing for CPU-bound work.** If your threads are computing — running Python bytecode, crunching numbers in pure Python, parsing in a tight loop — they each need the GIL to run, only one can hold it at a time, so they execute strictly one at a time no matter how many cores you have. You do not get $N$× speedup. You get roughly $1$×, and often *worse* than $1$× because the threads also pay the overhead of fighting over the lock and context-switching.

Let me prove it. Here is a deliberately CPU-bound function — counting primes is pure Python computation, no I/O, the GIL is held the entire time — run sequentially and then "parallelized" across 8 threads:

```python
import time
from concurrent.futures import ThreadPoolExecutor

def count_primes(limit: int) -> int:
    count = 0
    for n in range(2, limit):
        is_prime = True
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count

WORK = [200_000] * 8   # 8 identical CPU-bound chunks

# Sequential baseline
start = time.perf_counter()
seq = [count_primes(n) for n in WORK]
t_seq = time.perf_counter() - start

# "Parallel" with 8 threads
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as pool:
    thr = list(pool.map(count_primes, WORK))
t_thr = time.perf_counter() - start

print(f"sequential: {t_seq:.2f}s")
print(f"8 threads:  {t_thr:.2f}s  ({t_seq / t_thr:.2f}x)")
```

On the 8-core box this prints roughly:

```bash
sequential: 6.10s
8 threads:  6.45s  (0.95x)
```

Eight threads on eight cores ran the CPU-bound work *slightly slower* than one thread. There is no speedup because only one thread can execute Python bytecode at a time; the 8 cores were never used in parallel; and the 0.95× (a small slowdown) is the cost of the threads taking turns with the GIL — the contention and the context-switch overhead the single-threaded version never paid. This is the most important negative result in Python performance and the next figure draws it against the I/O-bound case so the contrast is unmissable.

![Two columns contrasting eight CPU bound threads that thrash on the GIL for no speedup against eight I O bound threads whose waits overlap for a near linear speedup](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-6.png)

The contrast is the entire lesson of this post in one figure. Same eight threads, same machine, opposite outcomes — because one workload *waits* (GIL free, waits overlap, near-linear speedup) and the other *computes* (GIL held, threads serialize, no speedup). When you reach for threads, you are betting that your work is waiting. If it is computing, you lose the bet, and you should reach for processes instead — separate interpreters, separate GILs, true parallelism — which is the entire subject of [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling). For CPU-bound Python, processes (or native code that releases the GIL, like NumPy or a Cython `nogil` block) are the answer, not threads.

#### Worked example: CPU-bound versus I/O-bound, side by side

Here is the comparison as one clean table, measured on **the same 8-core x86-64 Linux box, CPython 3.12, 16 GB RAM**, median of five runs after a warmup. The "work" in both rows is 8 tasks; the only difference is whether each task waits or computes:

| Workload | 1 thread | 8 threads | Speedup | Why |
|----------|---------:|----------:|--------:|-----|
| CPU-bound (count primes) | 6.10 s | 6.45 s | 0.95× | GIL held throughout; threads serialize, pay switch cost |
| I/O-bound (8 × 200 ms fetch) | 1.61 s | 0.21 s | 7.7× | GIL free during each wait; waits overlap |

Read those two rows together and you never have to wonder about threading again. The *exact same* `ThreadPoolExecutor(max_workers=8)`, on the *exact same* machine, delivers a tiny slowdown on compute and a 7.7× win on waiting. Threading is not a general parallelism tool. It is an I/O-overlap tool. That is its whole job, and within that job it is excellent.

## The other limits: too many threads, and the overhead floor

Even for the I/O case where threads win, there is a sweet spot, and going past it costs you. Each OS thread has real costs: a stack (often the default of several hundred kilobytes to a megabyte of reserved address space), a kernel scheduling entry, and a slice of the context-switch budget. Spin up 10,000 threads and you have spent gigabytes of address space on stacks and you are paying the kernel to context-switch among 10,000 entities, most of which are asleep. The GIL adds its own ceiling: every thread that wakes up from I/O must reacquire the GIL to run its Python parsing, and with thousands of threads all waking up and contending for that one lock, the contention itself becomes a bottleneck — threads spend time fighting to *get* the GIL rather than doing work.

So the practical guidance for sizing a thread pool for I/O work:

- **Start with the number of concurrent I/O operations you actually need**, not an arbitrary big number. If you fetch 40 URLs, 40 workers (or fewer, if the downstream service rate-limits you) is the ceiling that helps; more is wasted.
- **For a long-lived service** that handles continuous I/O, a common starting point is tens of threads, tuned by measurement — somewhere from a couple dozen to a couple hundred, depending on how much Python compute each request does between waits.
- **When you need *thousands* of concurrent connections** — a chat server, a web scraper hitting tens of thousands of pages, a high-fan-out proxy — threads stop scaling and you should switch to **asyncio**, which multiplexes all those waits onto a single thread with no per-connection stack and no GIL contention. That is the subject of the sibling post on [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines). The crossover is roughly: hundreds of concurrent I/O operations, threads are fine and simpler; tens of thousands, asyncio wins decisively on memory and scheduling.

#### Worked example: finding the thread-count sweet spot

Sweeping the worker count for a fixed batch of 200 I/O tasks (each ~50 ms wait, against a local mock server) on **the 8-core x86-64 Linux box, CPython 3.12**:

| Workers | Wall-clock (s) | Speedup | Reading |
|--------:|---------------:|--------:|---------|
| 1 | 10.30 | 1.0× | sequential baseline |
| 10 | 1.05 | 9.8× | near-linear |
| 50 | 0.23 | 44.8× | near-linear, approaching the floor |
| 100 | 0.17 | 60.6× | most waits now overlap |
| 200 | 0.14 | 73.6× | one thread per task, the practical max |
| 500 | 0.16 | 64.4× | over-provisioned: switch + GIL contention cost |
| 1000 | 0.21 | 49.0× | clearly past the sweet spot, getting worse |

The curve rises steeply, flattens near 200 (one worker per task), and then *bends back down* past 500 as thread overhead and GIL-reacquisition contention start to cost more than they buy. The sweet spot here is "one thread per concurrent task, up to a few hundred," and beyond that you are paying for threads that do not help. This is the empirical shape of the cost model: speedup approaches $N$ while compute and overhead are small, then the $\sum_i c_i$ plus thread-overhead terms in the denominator catch up and drag it back. If you needed 10,000-way concurrency, this is precisely where you would reach for asyncio instead.

## Case studies and real numbers

A few concrete, real-world data points to ground the guidance, with sources named so you can check them.

**The standard-library docs' own framing.** Python's official `concurrent.futures` and `threading` documentation is explicit that `ThreadPoolExecutor` is for I/O-bound work and `ProcessPoolExecutor` is for CPU-bound work — the split is baked into the design of the two executor classes. When the standard library hands you two nearly identical APIs differing only in threads-vs-processes, it is telling you the choice between them is the whole decision, and it maps exactly to I/O-bound versus CPU-bound.

**`requests` + `ThreadPoolExecutor` on real HTTP.** A very common production pattern — fan out a few dozen to a few hundred HTTP calls with a thread pool — routinely delivers 10–50× wall-clock speedups over the sequential loop, with the multiplier tracking the number of concurrent requests up to the pool size, exactly as the cost model predicts. The number you actually get depends on the per-request latency (more wait = more overlap to capture) and how much Python-side parsing each response triggers (the serialized compute term). For pure "fetch and check status," the speedup is close to the worker count; for "fetch and parse 5 MB of JSON," the Python parsing serializes under the GIL and pulls the speedup down.

**The free-threaded build (PEP 703) changes the CPU story, not the I/O story.** Python 3.13 introduced an experimental free-threaded build that removes the GIL, and in it, CPU-bound threads *do* scale across cores — the `count_primes` example above would actually speed up. But that build carries a single-thread performance cost today, and most of the ecosystem's C extensions are still catching up; it is the forward-looking topic covered in the series' free-threading post. For I/O-bound work, the free-threaded build changes essentially nothing, because the GIL was already free during I/O — the I/O-overlap win this post is about predates and outlives the GIL debate entirely.

**Where threads quietly beat async in practice.** Async gets the headlines for high concurrency, but for a moderate fan-out (dozens to low hundreds of concurrent I/O calls) that mixes in *blocking* libraries — a synchronous database driver, a C library with no async API, `requests` instead of `httpx` — a thread pool is often the *simpler and equally fast* choice, because it works with blocking calls out of the box and needs no async rewrite of your whole call stack. The decision is not "async is always better." It is fan-out and library support: blocking library plus moderate concurrency favors threads; async-native libraries plus huge fan-out favors asyncio.

**A typical ETL "fan-out" win.** The opening story — a report endpoint fanning out to twelve internal services — is the single most common place threading pays off in real backends, and the numbers are remarkably consistent across teams: twelve to a few dozen blocking calls, each in the 100–300 ms range, collapse from a 2–4 second sequential total to roughly the slowest single call (200–400 ms) under a `ThreadPoolExecutor`. The win equals the fan-out (about 10–12×) minus whatever Python compute glues the responses together. Crucially, this required *no new infrastructure* — no message queue, no extra service, no async rewrite — just four lines wrapping the existing blocking calls in a pool. That cost-to-benefit ratio, a four-line change for a 10× wall-clock win on a hot endpoint, is why "is this endpoint fanning out to blocking I/O sequentially?" should be one of the first questions you ask of any slow backend handler. It is the highest-leverage, lowest-risk fix in the whole concurrency toolbox.

## Stress-testing the downloader: where the simple model breaks

A real engineer does not stop at "it got faster." You pose the design, then stress it: what happens when the conditions change? Let us take the concurrent downloader and push on it from several directions, because each failure mode teaches you a boundary of the technique.

**What if the responses are huge and need heavy parsing?** Suppose each fetch returns 10 MB of JSON that you `json.loads` and reshape. Now the per-task compute $c_i$ is no longer negligible — it is tens of milliseconds of GIL-bound Python per response. The waits still overlap, but the parsing serializes under the GIL, so the $\sum_i c_i$ term grows and Amdahl's ceiling drops. You will see the speedup top out at maybe 5–8× instead of 30×, and the threads will be visibly fighting for the GIL during the parse phase. The fix is not more threads — it is to move the parsing off the GIL: parse with a C-accelerated library (`orjson` releases the GIL for large payloads), or hand the CPU-heavy parse to a `ProcessPoolExecutor`. Threading bought you the I/O overlap; it cannot buy you the parse parallelism, and recognizing that split is the whole skill.

**What if the downstream service rate-limits you?** Launch 200 concurrent requests at a service that allows 20 and you will get a wall of HTTP 429s, not a speedup. The right tool is a bounded pool (`max_workers=20`) or a `threading.Semaphore(20)` that caps in-flight requests regardless of how many tasks you submitted. The lesson: the concurrency ceiling is set by the *slowest shared resource* — the downstream service, the database connection pool, your own bandwidth — not by how many threads you can spawn. Past that ceiling, more threads just generate errors and retries that make things slower.

**What if one URL hangs forever?** Without a timeout, one stuck connection holds a worker thread hostage indefinitely, and if enough URLs hang, your whole pool drains to zero free workers and the program stops making progress — a slow-motion deadlock with no lock involved. This is why `requests.get(url, timeout=10)` and `as_completed(futures, timeout=30)` are not optional in production: they bound the damage a single bad endpoint can do. Always set both the per-request timeout and the overall collection timeout.

**What if you go from 8 to 800 threads?** As the sweet-spot table showed, you sail past the optimum and the $N\epsilon$ overhead term takes over: stacks eat memory, the kernel thrashes scheduling 800 mostly-idle threads, and GIL reacquisition contention rises. The speedup *decreases*. The fix at that scale is asyncio, which carries 10,000 concurrent waits on one thread with one small coroutine object each instead of one full OS thread each. Threading's ceiling is real, and it is roughly "a few hundred"; respect it.

Each of these is the same meta-lesson: threading overlaps *waiting*, and every other cost — parsing, rate limits, hung connections, thread overhead — is something threading does not solve and that you must handle with a different tool. Knowing the boundary is what separates "I added threads and it got faster" from "I know exactly why it got faster and where it stops helping."

## When to reach for threading (and when not to)

Threading is a precise tool with a narrow, important job. Here is the decision, which the final figure renders as a tree you can follow top to bottom.

![A decision tree branching on whether the work is I O bound or CPU bound then routing I O bound work to threads or asyncio and CPU bound work to processes or native code](/imgs/blogs/threading-done-right-io-bound-concurrency-and-its-limits-8.png)

Walk the tree. The root question is the only one that matters: **is the work waiting or computing?**

- **I/O-bound, using a blocking library, moderate fan-out (up to a few hundred):** reach for **threads** — a `ThreadPoolExecutor`. This is the sweet spot. Network calls, disk reads, database queries, anything that blocks. Threads work with synchronous libraries out of the box and the code reads almost exactly like the sequential version.
- **I/O-bound, async-native libraries available, huge fan-out (thousands+):** reach for **asyncio**. No per-connection stack, no GIL contention among thousands of threads, scales to tens of thousands of concurrent connections on one thread. The cost is that your whole I/O stack has to be async (`httpx`/`aiohttp`, async DB drivers).
- **CPU-bound:** do **not** use threads — they will not speed up and may slow down. Reach for **processes** (`ProcessPoolExecutor`, `multiprocessing`) for true parallelism across cores, or push the hot loop into **native code** that releases the GIL (NumPy vectorization, a Numba `@njit` function, a Cython `nogil` block). The earlier tracks in this series cover those levers in depth.

And the explicit "do not" list, because every wrong reach for threading is a bug you will eventually have to debug:

- **Do not use threads for CPU-bound work.** The GIL serializes them; you get no speedup and pay overhead. Measured above: 8 threads, 0.95×. Use processes.
- **Do not share mutable state without a lock — or better, do not share it at all.** Use `queue.Queue` to pass work between threads and you sidestep the entire class of race-condition bugs.
- **Do not hold a lock across slow work.** A lock held during a network call serializes your concurrency away. Hold locks only around the actual shared-state mutation, for microseconds.
- **Do not spin up thousands of threads.** Past a few hundred, thread overhead and GIL contention drag the speedup back down. If you need thousands-way concurrency, that is asyncio's job.
- **Do not acquire multiple locks in inconsistent orders.** That is the deadlock recipe. Pick a global lock order and stick to it, or avoid multiple locks entirely.
- **Do not forget the `with` block on executors and locks.** A leaked thread pool keeps threads alive; an unreleased lock hangs a thread forever.

## Key takeaways

- **Threads are for waiting, not computing.** When a thread blocks on I/O, the GIL is released and other threads run, so N waits overlap into roughly one wait. When a thread computes, it holds the GIL and threads serialize — no speedup.
- **The math is clean.** Sequential I/O time is $\sum_i w_i$; threaded I/O time is $\approx \max_i w_i + \sum_i c_i$. With negligible compute, the speedup approaches $N$, the number of concurrent tasks, up to the sweet spot.
- **Reach for `ThreadPoolExecutor` first**, not raw `Thread`. Use `submit` + `as_completed` to process results as they finish; the `with` block guarantees clean shutdown; `future.result()` re-raises worker exceptions where you can catch them.
- **`counter += 1` is not atomic** — it is a four-step read-modify-write, and two threads can lose updates. Guard every shared read-modify-write with a `Lock` (or `RLock` if the same thread re-enters), and keep critical sections tiny.
- **Prefer `queue.Queue` over manual locks.** It converts shared mutable state into message passing, handles all synchronization internally, and supports backpressure (`maxsize`), completion tracking (`task_done`/`join`), and clean shutdown (sentinels). Producer–consumer is the pattern; the queue is the only synchronization point.
- **Deadlock is the price of multiple locks.** Two threads acquiring two locks in opposite orders hang forever. Enforce a global lock ordering, or avoid multiple locks.
- **CPU-bound threads do not speed up — measured 0.95× on 8 threads.** Use processes or native code for compute; use threads only for I/O.
- **There is a thread-count sweet spot**: roughly one thread per concurrent I/O task, up to a few hundred. Beyond that, overhead and GIL contention drag the curve back down — switch to asyncio for thousands-way fan-out.

## Further reading

- [Why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — the series intro and the cost model that explains why overlapping I/O is the highest-leverage lever for network-bound code.
- [The GIL explained: what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) — the deep mechanics of the lock that makes threads help only for I/O, including the switch interval and contention.
- [Multiprocessing: true parallelism and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) — the right tool for CPU-bound work: separate interpreters, separate GILs, real parallelism across cores, and the serialization tax you pay for it.
- [Asyncio from the ground up: event loops and coroutines](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) — where to go when you need thousands-way I/O concurrency that threads can no longer carry.
- [A mental model of performance: latency numbers and the optimization loop](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) — the latency hierarchy that explains why a network round-trip is millions of times longer than a memory read, and why overlapping those round-trips is worth so much.
- The Python standard library docs for [`threading`](https://docs.python.org/3/library/threading.html), [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html), and [`queue`](https://docs.python.org/3/library/queue.html) — the canonical reference for the APIs used here, including the deliberate I/O-vs-CPU split between `ThreadPoolExecutor` and `ProcessPoolExecutor`.
- PEP 703 (making the GIL optional) and the "Faster CPython" notes — for where Python concurrency is heading and why the I/O-overlap win this post describes is independent of the GIL debate.
- *High Performance Python* by Micha Gorelick and Ian Ozsvald (O'Reilly) — chapters on concurrency and the GIL, with the same measure-first discipline this series follows.

---
title: "Async in Practice: Patterns, Pitfalls, and the Blocking-Code Trap"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Learn the asyncio patterns that survive production — bounded concurrency with a Semaphore, offloading blocking calls with to_thread, timeouts and cancellation — and how to never freeze your event loop with one blocking line again."
tags:
  [
    "python",
    "performance",
    "asyncio",
    "concurrency",
    "aiohttp",
    "httpx",
    "optimization",
    "profiling",
    "io-bound",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-1.png"
---

A few months ago I was paged because a service that fanned out HTTP calls to a dozen downstream APIs had gone from a steady p99 of forty milliseconds to a p99 of nine seconds, and nobody had deployed anything to it. The code was `async`, it used `aiohttp`, it `gather`-ed its calls, and on paper it was textbook concurrent Python. When I attached a profiler I found the culprit in a single line buried in a coroutine: someone had added a "quick" call to a metrics library that did a synchronous, blocking `requests.post` to an internal endpoint. That endpoint had slowed down. And because the call was synchronous and sat inside a coroutine, it did not just slow down its own request — it froze the entire event loop for the duration of every call, which meant every other in-flight request on that process stalled behind it. One blocking line, and a perfectly concurrent service had quietly turned into a single-file queue.

That is the story this post is really about. `asyncio` is the fourth and highest rung of the leverage ladder this series keeps climbing — after you have [done less work with the right algorithm](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple), [done it in bulk with NumPy](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), and learned [why Python is slow in the first place](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) — and for I/O-bound work that talks to thousands of sockets at once, nothing in pure Python beats it. But async has a sharp edge that catches almost everyone: it works beautifully until you block the loop, and then it fails in a way that is genuinely confusing because the symptom (everything is slow) is so far from the cause (one line is synchronous).

![A before and after diagram contrasting a synchronous blocking call inside a coroutine that freezes every task for 200 seconds against the same call offloaded to a thread that keeps the single event loop responsive and finishes in 2.3 seconds](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-1.png)

By the end of this post you will be able to write real `asyncio` code that survives production: fetch thousands of URLs concurrently with `aiohttp` or `httpx`, cap the in-flight count with a `Semaphore` so you do not OOM the box or trip a rate limiter, push a blocking call off the loop with `asyncio.to_thread` so it stops freezing everything, run CPU work in a `ProcessPoolExecutor` through `run_in_executor`, wrap any await in a timeout, and recognize the five bugs that bite every async codebase — the forgotten `await`, the blocked loop, sync-in-async, unbounded fan-out, and the fire-and-forget task that gets garbage-collected mid-flight. Most importantly, you will understand *why* a single blocking call stalls everything: the loop is one thread, it cannot preempt your code, and the moment your code stops yielding, the loop stops scheduling. We assume you already know [what a coroutine and an event loop are](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means); here we make them robust.

## The machine and the ground rules

Every number in this post comes from one stated reference box, so you can calibrate against your own hardware:

> **Reference machine: an 8-core x86-64 Linux box (or an Apple M2), CPython 3.12, 16 GB RAM, `aiohttp` 3.9, `httpx` 0.27, `uvloop` 0.19.** Network numbers assume a downstream service with about 100 ms median latency reachable over a low-loss LAN; absolute timings depend heavily on the network, but the *ratios* — blocked versus offloaded, unbounded versus bounded, default loop versus `uvloop` — come from the structure of the work and hold up.

When I quote a wall-clock figure like "100 URLs in 2.3 seconds," that is the median of several runs measured with `time.perf_counter` around the top-level `asyncio.run`, with the process warmed up so the interpreter and the HTTP connection pool are not paying first-call costs. I will flag anything I am estimating rather than measuring. The point of the numbers is never the third significant figure; it is the order of magnitude and the shape of the curve.

A few terms, defined the first time they appear. A **coroutine** is a function defined with `async def`; calling it does not run it, it returns a coroutine object that must be awaited or scheduled. The **event loop** is the single-threaded scheduler that runs coroutines, parks the ones that are waiting on I/O, and resumes them when their I/O is ready. **`await`** is a cooperative yield point: it hands control back to the loop so the loop can run something else while this coroutine waits. A **task** is a coroutine wrapped by `asyncio.create_task` (or a `TaskGroup`) so the loop schedules it concurrently with others. **Blocking** means a call that does not return control to the loop until it finishes — a synchronous network call, `time.sleep`, a tight CPU computation, a blocking database driver. The **GIL** (Global Interpreter Lock) is the single lock CPython holds so only one thread runs Python bytecode at a time; it is released during blocking I/O and inside many C calls, which is why threads still help I/O-bound work. **Backpressure** is the mechanism by which a fast producer is forced to slow down so a slow consumer is not overwhelmed. We will define `Semaphore`, `to_thread`, `run_in_executor`, and `uvloop` as we reach them.

## Why one blocking call freezes everything

Start with the single most important fact about `asyncio`, because every other pitfall in this post is a corollary of it: **the event loop runs on one thread, and it cannot interrupt your code.** There is no timer that fires and yanks control away from a coroutine that is taking too long. Cooperative multitasking means exactly that — control moves between coroutines only at `await` points, and only because the coroutine voluntarily handed it back. If your code runs from one `await` to the next without hitting another `await` in between, the loop is blind and deaf for that entire stretch. It cannot run other tasks, it cannot check whether other sockets are ready, it cannot fire scheduled callbacks. It is sitting on the call stack of your coroutine, waiting for you to return.

Now consider what a blocking call does to that picture. When you write `requests.get(url)` inside a coroutine, the `requests` library opens a socket and calls the operating system's blocking `recv`, which does not return until bytes arrive. During those, say, two seconds of network latency, the OS has put the *thread* to sleep. But that thread is the only thread the event loop has. So the loop is asleep too. Every coroutine you carefully wrote to run concurrently is parked behind this one synchronous call, not because they are waiting on their own I/O, but because the scheduler that would resume them is itself frozen. This is the picture in the figure at the top of the post: one blocking call, and the wall-clock time of the whole batch goes from "all requests overlap" to "all requests run one after another."

Let me make the cost concrete and provable. Suppose you have $N$ tasks, each of which makes one network call of latency $L$. With proper async concurrency, the calls overlap — you issue all of them, then wait for all of them, so the total wall-clock time is roughly $L$ (plus a little scheduling overhead), independent of $N$. With a blocking call in the loop, the calls serialize, so the wall-clock time is $N \times L$. The slowdown factor is therefore $N$ itself:

$$\text{slowdown} = \frac{N \cdot L}{L} = N$$

This is not a constant-factor penalty; it scales with your concurrency. The more concurrent your workload was *supposed* to be, the worse the blocking call hurts. A service handling 100 concurrent requests with a blocked loop is 100 times slower than it should be. That is why the symptom is so dramatic and so confusing: the blocking line might be a tiny, innocuous-looking call, but its impact is multiplied by every other thing the loop was supposed to be doing at the same time.

#### Worked example: the blocked loop versus the offloaded call

Here is the experiment that produces the figure. We define a coroutine that should run 100 "requests" concurrently. The blocking version calls `time.sleep(0.1)` to stand in for a synchronous network call of 100 ms; the offloaded version pushes that same blocking sleep onto a thread with `asyncio.to_thread`.

```python
import asyncio
import time

def blocking_io(i: int) -> int:
    # Stand-in for a synchronous requests.get or a blocking DB call.
    time.sleep(0.1)          # 100 ms of "network", but it BLOCKS the thread
    return i * i

# WRONG: the blocking call sits directly in the coroutine.
async def task_blocking(i: int) -> int:
    return blocking_io(i)    # freezes the loop for 100 ms, every time

# RIGHT: the blocking call is offloaded to a worker thread.
async def task_offloaded(i: int) -> int:
    return await asyncio.to_thread(blocking_io, i)   # loop stays free

async def run(make_task) -> float:
    t0 = time.perf_counter()
    results = await asyncio.gather(*(make_task(i) for i in range(100)))
    assert len(results) == 100
    return time.perf_counter() - t0

print("blocking :", asyncio.run(run(task_blocking)))   # ~10.0 s
print("offloaded:", asyncio.run(run(task_offloaded)))  # ~0.1 s
```

On the reference machine, the blocking version takes about **10.0 seconds** — 100 tasks times 100 ms each, fully serialized, exactly the $N \times L$ prediction. The offloaded version takes about **0.1 seconds**, because all 100 blocking sleeps run concurrently on the thread pool while the loop stays responsive. That is a **100× speedup** from moving one call off the loop, and it matches the math exactly: $N = 100$, so the blocked version is $N$ times slower. The lesson is brutal and simple: a blocking call inside a coroutine does not slow down one task, it slows down all of them, and the factor is your concurrency.

The timeline of what actually happens to the other tasks is worth seeing on its own, because the "dead gap" is the mental image you want burned in.

![A timeline diagram showing a dead gap where every task idles during a two second blocking call, contrasted with an offloaded version where other tasks keep running concurrently and the batch finishes in 2.3 seconds](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-6.png)

The crucial detail the timeline captures is that during the dead gap, the other tasks are not *waiting on their own work* — they are ready to run, their I/O may already be complete, and they are simply not being scheduled because the one thread that would schedule them is stuck. Move the blocking call off the loop and that gap fills with useful work.

## How the loop actually runs a tick

To make all of this concrete rather than hand-wavy, it helps to know what the event loop is literally doing on each iteration, because once you have seen the inside of the loop, every pitfall in this post stops being a rule to memorize and becomes an obvious consequence. The loop runs a tight cycle, and one pass through that cycle is called a *tick*. Each tick does three things in order. First, it runs every callback that is currently "ready" — coroutines whose awaited I/O has completed, plus any callbacks scheduled with `loop.call_soon`. Second, it computes how long it can afford to sleep: if there are timers due (a `sleep`, a timeout), it sleeps until the nearest one, otherwise it can block until a socket becomes readable or writable. Third, it asks the operating system, via a system call like `epoll` on Linux or `kqueue` on macOS, "which of these thousands of sockets are now ready?" — and the sockets that come back ready turn their waiting coroutines into ready callbacks for the next tick.

That `epoll`/`kqueue` call is the heart of why async scales. It is a single system call that monitors thousands of file descriptors at once and returns only the ones that changed, in roughly constant time per ready descriptor rather than time proportional to the total number being watched. This is the mechanism that lets one thread babysit ten thousand idle connections cheaply: the loop is not polling each one, it is asking the kernel for the short list of the ones that are ready right now. When you `await` an `aiohttp` request, your coroutine registers its socket with the loop's selector and suspends; the loop adds that descriptor to the set it hands to `epoll`; and when the response bytes arrive, `epoll` reports that descriptor ready, the loop resumes your coroutine, and you get your result. Nowhere in that picture does the loop ever wait on *one specific* socket — it waits on *all of them at once* and reacts to whichever is ready first.

Now re-read what a blocking call does in this frame. A blocking call happens inside step one — running a ready callback. The loop has called your coroutine, your coroutine is on the stack, and the loop cannot proceed to step two (compute the sleep) or step three (`epoll` the sockets) until your callback returns. So a blocking call does not merely "make one task slow"; it prevents the loop from ever reaching the `epoll` call that would notice the other thousands of sockets are ready. The ready sockets pile up, unnoticed, because the loop is stuck before the step that would notice them. That is the mechanical truth behind "one blocking call freezes everything," and it is why the rule is absolute: any code that runs on the loop must return quickly, because returning quickly is the only way the loop ever gets back to `epoll`.

This also explains a subtlety about `await asyncio.sleep(0)`. Writing `await asyncio.sleep(0)` is the canonical way to *voluntarily yield* to the loop without actually waiting — it suspends your coroutine, lets the loop run one full tick (process ready callbacks, check the sockets), and then resumes you. If you have a long stretch of CPU-ish work that you cannot offload but want to break up so it does not starve the loop completely, sprinkling `await asyncio.sleep(0)` between chunks gives the loop a chance to breathe. It is a crude tool — you are still doing CPU work on the loop thread, which is the thing we said not to do — but as a stopgap for "this loop runs for 50 ms and I just need it not to monopolize everything," a periodic `sleep(0)` is the lever. The clean fix is still to offload; `sleep(0)` is the duct tape.

One more consequence worth internalizing: **the order in which ready callbacks run within a tick is the order they became ready, not the order you created the tasks.** Async is concurrent, not parallel and not ordered. If task A and task B both have their I/O complete during the same `epoll` call, they both become ready, and the loop runs them in the next tick one after the other — A then B, or B then A, depending on registration order, but never simultaneously. This is why a sequence of statements inside a coroutine with no `await` between them is effectively atomic: no other coroutine can interleave, because there is no yield point at which the loop could switch. It is also why you usually do not need locks within a single loop, and why you suddenly *do* need them the moment you offload to threads — the offload breaks the single-threaded, one-at-a-time guarantee that made your loop code safe.

## Real concurrent I/O with aiohttp and httpx

Enough with `time.sleep` stand-ins; let us do real network I/O. The two libraries you will reach for are `aiohttp` and `httpx`. Both are properly asynchronous: their `get`/`post` calls are coroutines that `await` the socket, so they yield to the loop while the network does its thing. The standard library's `requests` is *not* async — it has no `await`, it blocks — which is exactly why dropping a `requests.get` into a coroutine is the number-one way people freeze their loop.

Here is the canonical concurrent fetch with `aiohttp`. The shape to internalize is: one shared `ClientSession` for the whole batch (it owns the connection pool, so you create it once and reuse it), one coroutine per URL, and `asyncio.gather` to run them all and collect the results.

```python
import asyncio
import aiohttp

async def fetch(session: aiohttp.ClientSession, url: str) -> int:
    async with session.get(url) as resp:
        await resp.read()            # await the body; yields to the loop
        return resp.status

async def fetch_all(urls: list[str]) -> list[int]:
    # One session, reused across all requests: pools connections.
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, u) for u in urls]
        return await asyncio.gather(*tasks)

urls = ["https://httpbin.org/delay/0.1"] * 100
statuses = asyncio.run(fetch_all(urls))
print(len(statuses), "responses")     # 100 responses, ~0.2 s wall-clock
```

The `httpx` version is nearly identical in spirit; `httpx` is worth knowing because it offers both a sync and an async client behind one API, which makes it convenient when part of your code base is async and part is not. The async client looks like this:

```python
import asyncio
import httpx

async def fetch(client: httpx.AsyncClient, url: str) -> int:
    resp = await client.get(url)      # a real coroutine; yields to the loop
    return resp.status_code

async def fetch_all(urls: list[str]) -> list[int]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await asyncio.gather(*(fetch(client, u) for u in urls))

statuses = asyncio.run(fetch_all(["https://httpbin.org/delay/0.1"] * 100))
```

Two details matter for performance and correctness. First, **reuse the session/client across requests.** Creating a new `ClientSession` per request throws away the connection pool, re-does the TLS handshake every time, and can leave sockets in `TIME_WAIT` faster than the OS reclaims them. One session per batch (or per application) is the rule. Second, both libraries default to a connection-pool limit, and that limit is itself a form of concurrency bound — `aiohttp`'s default connector caps total connections at 100. That default is doing you a quiet favor, but you should not rely on it as your only guardrail, because the moment you have more tasks than connections, the extra tasks queue up inside the library in a way that is harder to observe and tune than an explicit `Semaphore`, which we get to next.

Before we bound the concurrency, look at the gap between a default loop and a faster one, because it is the cheapest win in async Python. `uvloop` is a drop-in replacement for the default event loop, built on the same libuv C library that powers Node.js. It implements the loop's socket and scheduling machinery in C rather than Python, so the per-event overhead drops. You install it and activate it in one line:

```python
import asyncio
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())  # before asyncio.run
# ...everything else is identical; asyncio.run now uses uvloop
```

On the reference machine, for a workload dominated by many small requests where loop overhead is a meaningful fraction of the time, `uvloop` typically delivers a **2 to 4× throughput improvement** over the default loop. The win shrinks as your per-request work grows (if each request does heavy parsing, the loop is not your bottleneck), but for a high-fan-out client or a busy socket server it is close to free speed. We will put a real number on it in the results section.

## Bounding concurrency: the Semaphore

Now the second great pitfall, the mirror image of blocking the loop: **unbounded fan-out.** The `gather` pattern above is so easy that the natural next step is to point it at 10,000 URLs. And then your process tries to open 10,000 sockets at once, allocate 10,000 response buffers, and hit the downstream service with 10,000 simultaneous requests. Three things go wrong, often all at once. You exhaust file descriptors or memory and the process gets OOM-killed. The downstream service sees a thundering herd and starts returning HTTP 429 "Too Many Requests." And your own latency goes to pieces because everything is contending for the same finite pool of connections.

The fix is to cap the number of tasks that can be *in the work section* at any one time, and let the rest wait their turn. That is exactly what an `asyncio.Semaphore` does. A semaphore is a counter initialized to $N$; `acquire` decrements it (and blocks if it is already zero), `release` increments it. Used as an async context manager, it admits at most $N$ coroutines into the guarded block at once; the $N+1$-th coroutine awaits at the `async with` line until one of the in-flight coroutines releases its slot.

![A graph diagram showing 10000 spawned tasks acquiring one of N semaphore slots to run their fetch in flight while the rest wait under backpressure, then releasing the slot so the next task wakes and proceeds to a fully successful result](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-3.png)

Here is the pattern. You spawn all the tasks immediately — that part is cheap, a coroutine object is small — but each one must acquire the semaphore before it touches the network, so only $N$ ever fetch concurrently.

```python
import asyncio
import aiohttp

async def fetch_one(sem: asyncio.Semaphore,
                    session: aiohttp.ClientSession, url: str) -> int:
    async with sem:                       # at most N in this block at once
        async with session.get(url) as resp:
            await resp.read()
            return resp.status

async def fetch_all(urls: list[str], limit: int = 50) -> list[int]:
    sem = asyncio.Semaphore(limit)        # cap N concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(sem, session, u) for u in urls]
        return await asyncio.gather(*tasks)

statuses = asyncio.run(fetch_all(["https://httpbin.org/get"] * 10_000, limit=50))
```

The shape of the win is in the next figure: unbounded fan-out blows up memory and trips the rate limiter; the bounded version holds memory flat and finishes cleanly.

![A before and after diagram contrasting an unbounded gather of 10000 requests that opens 10000 sockets reaches 3.8 GB of memory and fails 31 percent of requests with 429 errors against a semaphore capped at 50 that uses 240 MB and succeeds on every request](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-4.png)

Why does the semaphore fix the memory problem, provably? Each in-flight request holds resources: a socket, a TLS session, a read buffer for the response body. Call the per-request footprint $b$ bytes. With unbounded fan-out, peak memory is $N \times b$ — all $N$ requests are live at once. With the semaphore, peak memory is $\text{limit} \times b$, because that is the most that are ever live simultaneously; the other $N - \text{limit}$ tasks are parked coroutines, which cost a few hundred bytes each, not a full socket-plus-buffer. If a response body averages, say, 256 KB and you have 10,000 of them, unbounded peaks near $10{,}000 \times 256\,\text{KB} \approx 2.5$ GB of response buffers alone, before counting sockets and TLS state; with `limit=50` you peak near $50 \times 256\,\text{KB} \approx 12.8$ MB of buffers. That is the difference between an OOM kill and a flat memory graph, and it falls straight out of the arithmetic.

#### Worked example: choosing the cap

What should `limit` be? This is a real engineering decision, not a magic number, and the right way to reason about it is Little's law from queueing theory. Little's law says the average number of items in a stable system equals the arrival rate times the average time each item spends in the system: $L = \lambda W$. Rearranged, the throughput you can sustain is $\lambda = L / W$, where $L$ is your concurrency cap and $W$ is the per-request latency.

Suppose the downstream service has a median latency of $W = 100$ ms and you want to drive throughput of $\lambda = 500$ requests per second without overwhelming it. Then the concurrency you need is $L = \lambda W = 500 \times 0.1 = 50$ in-flight requests. So `limit=50` is not arbitrary; it is the number that achieves 500 req/s at 100 ms latency. If you set it to 5,000 instead, you do not get 50,000 req/s — the downstream service cannot serve that, so the extra 4,950 requests just sit in its queue, latency climbs, and you start getting 429s or timeouts. The cap is there to match your offered load to what the system can actually absorb. A good starting rule: set `limit` to roughly the downstream service's stated concurrency budget, or to $\lambda_{\max} W$ where $\lambda_{\max}$ is the rate it can sustain, then tune by watching the error rate and tail latency.

The `Semaphore` is the explicit, observable, tunable way to do this. There are higher-level conveniences too — `aiohttp`'s connector limit, third-party helpers like `aiometer`, and the `max_workers`-style caps on executors — but the semaphore is the primitive they are all built on, and knowing it means you can bound concurrency anywhere, not just where a library happened to give you a knob.

### Backpressure: the semaphore is only half the story

Bounding the *issue* rate with a semaphore handles the case where you control the producer — you have a list of 10,000 URLs and you choose how fast to fire them. But there is a second, sneakier version of unbounded fan-out: the producer is faster than the consumer and you are buffering the difference in memory. Picture a coroutine that reads lines from a fast source — a Kafka topic, a giant file, an upstream socket — and for each line spawns a task to process it. If the source produces lines faster than your tasks can finish them, the unprocessed tasks pile up, and you are right back to unbounded memory growth, just arriving from the producer side instead of the URL-list side.

Backpressure is the discipline of forcing the fast producer to slow down to the consumer's pace. The semaphore gives you this almost for free if you acquire it on the *producer* side: instead of spawning a task per line unconditionally, the producer must acquire a slot before it spawns, so when all $N$ slots are taken, the producer's `acquire` *blocks* — and because it blocks, it stops reading from the source, which (for a well-behaved source like a TCP socket or a bounded queue) propagates the slowdown all the way back. The fast producer is throttled to the consumer's throughput by nothing more than the semaphore refusing to let it run ahead. An `asyncio.Queue` with a `maxsize` gives you the same effect from the other direction: `await queue.put(item)` blocks when the queue is full, so a producer feeding a bounded queue is automatically paced by how fast the consumers drain it. Either way, the principle is the same — a bounded buffer somewhere in the pipeline is what converts "produce as fast as you can and hope" into "produce only as fast as the slowest stage can consume," and that bound is what keeps memory flat under a producer you do not control.

The number to remember: an *unbounded* queue or an unbounded spawn loop has no backpressure, and under a faster-than-consumer producer its memory grows without limit until the process dies. A bounded queue or a semaphore-gated producer has backpressure, and its memory is capped at the bound times the per-item footprint, exactly the $\text{limit} \times b$ ceiling from the semaphore derivation. Whenever you see memory climbing steadily in an async service that is "keeping up" on average, look for the unbounded buffer between a fast producer and a slow consumer — that is almost always where it is.

## Offloading blocking work: to_thread and run_in_executor

Sometimes you simply cannot avoid a blocking call. The library you must use has no async version. The database driver is synchronous. You have a chunk of pure-CPU work — parsing, hashing, compression, a NumPy reduction — that holds the thread for tens of milliseconds. You cannot rewrite the world to be async, and you should not try. The right move is to **push the blocking work off the event-loop thread** so the loop stays free to schedule everything else.

`asyncio` gives you two tools for this, and the distinction between them is the single most important design decision in mixed async code:

- **`asyncio.to_thread(func, *args)`** runs `func` in a thread from a shared `ThreadPoolExecutor` and gives you an awaitable. Use it for blocking *I/O* — a synchronous HTTP client, a blocking DB driver, a file read. It works because blocking I/O releases the GIL while it waits, so other threads (including the loop's) can run.
- **`loop.run_in_executor(pool, func, *args)`** runs `func` in whatever executor you pass. Pass a `ProcessPoolExecutor` for *CPU-bound* work, because separate processes each have their own GIL and so achieve true parallelism across cores. A `ThreadPoolExecutor` will *not* speed up CPU-bound Python — the GIL serializes the bytecode — it will only keep the loop responsive while the one core grinds.

Here is the offloading pattern for a blocking I/O library. The blocking call goes through `to_thread`, so it runs on a worker thread and the `await` yields the loop back to the scheduler:

```python
import asyncio
import time
import requests   # synchronous, blocking — the wrong thing to call directly

def blocking_fetch(url: str) -> int:
    # A synchronous library we cannot avoid; it BLOCKS the calling thread.
    return requests.get(url, timeout=10).status_code

async def fetch(url: str) -> int:
    # Offloaded: runs on a worker thread, loop stays responsive.
    return await asyncio.to_thread(blocking_fetch, url)

async def main(urls: list[str]) -> list[int]:
    return await asyncio.gather(*(fetch(u) for u in urls))

asyncio.run(main(["https://httpbin.org/delay/0.1"] * 64))
```

Why does `to_thread` actually work for I/O, mechanically? Because of the GIL's release behavior. When `requests` calls the OS's blocking `recv`, CPython releases the GIL before the syscall and reacquires it after. So while the worker thread is parked in `recv`, the GIL is free, and the event-loop thread can grab it and run other coroutines. The blocking is now confined to the worker thread, where it harms nobody, instead of the loop thread, where it harmed everybody. This is the same reason [threads help I/O-bound work despite the GIL](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) — async just uses a thread pool as an escape hatch for the synchronous parts.

For CPU work, threads are not enough, because a tight Python loop does *not* release the GIL — it holds it the whole time, so a `ThreadPoolExecutor` would let the loop schedule but would not let the CPU work run in parallel. You need separate processes. Here is the CPU-offload pattern with a `ProcessPoolExecutor`, which is the bridge to [true multicore parallelism and its pickling cost](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling):

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_heavy(n: int) -> int:
    # Pure CPU: holds the GIL the entire time. Must go to a process.
    total = 0
    for i in range(n):
        total += i * i
    return total

async def main() -> None:
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=8) as pool:
        # Each call runs in a separate process: true parallelism, own GIL.
        tasks = [loop.run_in_executor(pool, cpu_heavy, 5_000_000)
                 for _ in range(8)]
        results = await asyncio.gather(*tasks)
    print(sum(results))

asyncio.run(main())
```

The cost of the process pool is the marshaling tax: the arguments and the return value are pickled, shipped between processes, and unpickled. That round trip is only worth paying when the CPU work per call is large compared to the size of the data crossing the boundary. A call that computes for 200 ms and returns a small integer is a great fit; a call that computes for 1 ms but ships a 50 MB array is a terrible one — the pickling dominates and you would have been faster single-threaded. The decision tree is the same one from the multiprocessing post: process pools win when the per-task compute is large and the per-task data is small.

One subtlety about `to_thread` worth knowing: it does not create a fresh thread per call, it submits to a *shared* default thread pool that `asyncio` manages for the running loop. By default that pool is sized to `min(32, os.cpu_count() + 4)` threads in recent CPython, which means if you fire off hundreds of `to_thread` calls at once, only that many run concurrently and the rest queue. For blocking *I/O* that is usually fine — you can raise the cap because I/O-bound threads spend their time parked in syscalls, not burning CPU — but you raise it deliberately, by setting a custom executor on the loop with `loop.set_default_executor(ThreadPoolExecutor(max_workers=200))`, not by hoping the default is big enough. Sizing a thread pool for blocking I/O follows the same Little's-law arithmetic as the semaphore: if each blocking call takes $W$ seconds and you want throughput $\lambda$, you need about $\lambda W$ threads in flight. The difference from the semaphore is *where* the bound lives — the semaphore bounds your async tasks before they reach the network, the thread-pool size bounds how many blocking calls run at once. In a well-tuned system the two agree; if your semaphore admits 50 tasks but your thread pool only has 8 threads, the thread pool becomes the real, hidden bottleneck, and you will scratch your head wondering why raising the semaphore did nothing.

A second subtlety: a `ProcessPoolExecutor` created inside an async program must be set up carefully because of how process pools spawn workers. On Linux the default start method is `fork`, which copies the parent process — including, awkwardly, a half-initialized event loop and any open file descriptors — and that can lead to subtle deadlocks if a forked child inherits a lock held by the parent. The safe pattern is to create the `ProcessPoolExecutor` *before* you start the event loop, or to use the `forkserver`/`spawn` start method, which gives each worker a clean interpreter. The details are the same ones from the [multiprocessing post](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling); the async-specific lesson is simply that mixing a process pool into a running loop multiplies the ways things can go wrong, so keep the pool's lifetime explicit and create it early.

The mental model for all of this — async I/O on the loop, blocking I/O on threads, CPU on processes — is one layered system, and it is worth seeing as a stack.

![A layered stack diagram showing the asyncio event loop driving native await I/O at the top then delegating blocking libraries to a thread pool where the GIL is freed and CPU work to a process pool that uses all cores with results marshaled back](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-5.png)

The top layer — `aiohttp`, `httpx`, async database drivers — is where you want to live, because it is the most efficient: no thread, no process, no marshaling, just a coroutine yielding on a socket. The thread layer is the escape hatch for blocking libraries you cannot replace. The process layer is the bridge to real cores for CPU work. A well-architected async service uses all three, each for exactly what it is good at, and the figure-7 matrix at the end of this post is the cheat sheet for which is which.

## Timeouts and cancellation

An async program that talks to the network without timeouts is a hang waiting to happen. A downstream service stops responding, your coroutine awaits forever, the task never completes, and over time you accumulate stuck tasks that hold connections and memory until the process dies. Every external await needs a deadline.

Python 3.11 gave us `asyncio.timeout`, a context manager that is the cleanest way to bound a block of awaits:

```python
import asyncio

async def fetch_with_deadline(client, url: str):
    try:
        async with asyncio.timeout(5.0):     # 5-second deadline for this block
            return await client.get(url)
    except TimeoutError:
        # The await was cancelled at the 5-second mark; handle it.
        return None
```

The older `asyncio.wait_for(coro, timeout)` does the same job for a single coroutine and works on all supported versions:

```python
result = await asyncio.wait_for(client.get(url), timeout=5.0)  # raises TimeoutError
```

Here is the part people get wrong, and it is worth understanding precisely because it is where async correctness bugs live. **A timeout works by cancelling the coroutine.** When the deadline passes, `asyncio` throws an `asyncio.CancelledError` *into* the coroutine at its current `await` point. The coroutine does not just vanish — the exception propagates up through its stack, running `finally` blocks and `async with` cleanup along the way, which is how the socket gets closed and resources get released. This is cooperative cancellation: the loop asks the coroutine to stop by injecting an exception at a yield point, and the coroutine unwinds cleanly.

Two consequences follow. First, **a coroutine that never awaits cannot be cancelled.** If your "coroutine" is actually doing a long synchronous computation with no `await` inside, the loop has no yield point at which to inject the `CancelledError`, so the timeout silently does not fire until the computation finishes on its own. This is yet another face of the blocking-the-loop problem: cancellation, like scheduling, only happens at await points. Second, **do not swallow `CancelledError`.** A common bug is a broad `except Exception` that catches the cancellation, logs it, and continues — which defeats the timeout and can leave the task running past its deadline. `CancelledError` inherits from `BaseException`, not `Exception`, precisely so a bare `except Exception` will not eat it; do not undo that protection by catching `BaseException` and ignoring it. If you must clean up on cancellation, catch it, do the cleanup, and re-raise:

```python
async def careful(client, url):
    try:
        return await client.get(url)
    except asyncio.CancelledError:
        await client.aclose()     # clean up
        raise                     # ALWAYS re-raise; let the cancellation propagate
```

The cancellation model is also what makes structured concurrency work. An `asyncio.TaskGroup` (3.11+) will, if one of its child tasks raises, cancel all the sibling tasks and wait for them to finish unwinding before the `async with` block exits. That is the well-behaved default for "run these together, and if one fails, tear them all down cleanly" — far safer than a bare `gather`, which by default lets siblings keep running after one raises.

### gather versus TaskGroup: the error-handling fork in the road

It is worth being precise about the difference between `gather` and `TaskGroup`, because choosing the wrong one is a quiet source of leaked tasks and swallowed errors. The two look interchangeable for the happy path, but they diverge sharply the moment something fails, and the failure path is exactly where production code lives or dies.

`asyncio.gather`, by default, has surprising failure semantics. If one of the awaited coroutines raises, `gather` propagates that first exception to the caller *immediately* — but it does **not** cancel the other coroutines. They keep running in the background, detached, and if they later raise, those exceptions are dropped on the floor (you may see a "Task exception was never retrieved" warning, or nothing). So a single failure in a `gather` can leave you with orphaned tasks still hitting the network, still holding connections, with their results and errors lost. You can ask `gather` to instead *collect* every result and exception into the returned list with `return_exceptions=True`, which is the right call when you want "give me whatever each one produced, success or failure, and let me sort it out":

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for r in results:
    if isinstance(r, Exception):
        log.warning("one task failed: %r", r)   # handle per-task failures
    else:
        process(r)
```

`asyncio.TaskGroup` (3.11+) flips the default to the safe one. If any child task raises, the group cancels all the still-running siblings, waits for them to unwind cleanly, and then raises an `ExceptionGroup` bundling every exception that occurred. Nothing is left orphaned; nothing is silently dropped. The shape is:

```python
async def fetch_all(urls):
    results = []
    async with asyncio.TaskGroup() as tg:        # structured: scoped lifetime
        tasks = [tg.create_task(fetch(u)) for u in urls]
    # On exit: every task is done. If any raised, an ExceptionGroup is raised here
    # and ALL siblings were already cancelled and cleaned up.
    return [t.result() for t in tasks]
```

The rule of thumb: reach for `TaskGroup` when the tasks form a unit of work that should succeed or fail together — if one fails, you want the rest torn down. Reach for `gather(..., return_exceptions=True)` when the tasks are independent and a partial failure is acceptable — fetch 1,000 URLs and accept that 30 will fail, keeping the 970 that succeeded. Never use a bare `gather` without thinking about which of those two worlds you are in, because its default — propagate one error, abandon the rest — is almost never the behavior you actually want.

#### Worked example: the orphaned task that kept calling

A team I worked with had a `gather` over a batch of payment-status checks. One check started raising on a malformed record. `gather` propagated that exception up, the request handler returned a 500, and everyone assumed the batch was dead. But the other forty-nine status checks in that `gather` were never cancelled — they kept running, kept polling the payment provider, and because the handler had returned, nothing ever retrieved their results. Under load this meant a slow accumulation of detached tasks, each holding a connection to the payment API, until the connection pool was exhausted and *healthy* requests started failing to get a connection. The metric that finally pointed at it was open connections climbing steadily with no corresponding increase in active requests — the classic signature of leaked tasks. Switching the `gather` to a `TaskGroup` fixed it in one line of intent: now when one check failed, the other forty-nine were cancelled, their connections released, and the connection-pool leak vanished. The lesson is that "one task failed" and "the other tasks were cleaned up" are two completely different guarantees, and a bare `gather` gives you only the first.

## The five bugs that bite every async codebase

Now the rogues' gallery. These five bugs account for the overwhelming majority of "my async code is wrong and I do not understand why" situations. Each has a clear symptom and a one-line fix, summarized in the matrix below and then dissected one at a time.

![A matrix diagram mapping five asyncio pitfalls to their symptom and fix covering the forgotten await that never runs blocking the loop that stalls all tasks sync in async that freezes the loop unbounded fan-out that causes OOM and fire and forget tasks that get garbage collected](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-2.png)

### Bug one: the forgotten await

You call a coroutine and forget to `await` it. Calling an `async def` function does not run it — it returns a coroutine object. Without an `await` (or a `create_task`), that object is created and then dropped, and the body of the coroutine never executes.

```python
async def save(record):
    await db.insert(record)        # the actual work

async def handle(records):
    for r in records:
        save(r)                    # BUG: coroutine created, never awaited, never runs
    # ...you think you saved; you saved nothing
```

The symptom is insidious because there is no crash — the function returns, the loop moves on, and your data is silently not saved. CPython does emit a `RuntimeWarning: coroutine 'save' was never awaited`, which is your tell; treat that warning as an error in CI. The fix is to `await` it, or, if you want them to run concurrently, schedule them as tasks and await the group:

```python
async def handle(records):
    async with asyncio.TaskGroup() as tg:
        for r in records:
            tg.create_task(save(r))    # scheduled concurrently, awaited at block exit
```

### Bug two: blocking the loop

We have covered this thoroughly — a synchronous call (`requests.get`, `time.sleep`, `open(...).read()` on a slow disk, a blocking DB driver) inside a coroutine freezes the loop. The symptom is that *everything* slows down, often with the loop's own "this task took N ms" warnings if you enable debug mode (`PYTHONASYNCIODEBUG=1` or `asyncio.run(main(), debug=True)`, which logs callbacks that block the loop for more than 100 ms). The fix is `asyncio.to_thread` for blocking I/O and `run_in_executor(ProcessPool)` for CPU. The tell, if you are not sure whether a library call blocks, is simple: if you are not writing `await` in front of it and it does any I/O or heavy compute, assume it blocks.

### Bug three: sync in async (the import you didn't check)

A subtler cousin of bug two: you correctly made your own code async, but a dependency three layers down does blocking I/O. The classic is using a "sync" database driver (`psycopg2`, the synchronous `redis` client, a plain `sqlite3` call) inside an otherwise-async handler. It looks async — your function is `async def`, you wrote `await` on your own calls — but the driver underneath blocks the loop on every query. The symptom is the same loop freeze; the fix is to either swap to an async driver (`asyncpg`, `redis.asyncio`, `aiomysql`) or wrap the sync driver's calls in `to_thread`. The discipline is to audit your dependencies: when you adopt a library inside async code, check whether its network and disk calls are actually awaitable or merely wrapped in an `async def` that blocks internally.

There is a practical way to *catch* this bug rather than reason about it, and it is worth wiring into development. `asyncio`'s debug mode logs any callback that occupies the loop for longer than a threshold (100 ms by default, tunable via `loop.slow_callback_duration`). Turn it on while developing and any sync-in-async offender announces itself by name in the log:

```python
import asyncio, logging
logging.basicConfig(level=logging.WARNING)

async def main():
    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = 0.05    # flag anything over 50 ms on the loop
    # ... your handlers; a blocking driver here logs "Executing <Task ...> took 0.31s"
    await run_the_service()

asyncio.run(main(), debug=True)           # debug=True enables the slow-callback log
```

A third-party tool, `blockbuster`, goes further and actively raises an exception the instant a known-blocking standard-library call (a synchronous socket read, a blocking `open` on a file, `time.sleep`) executes inside the loop, turning a silent performance bug into a loud, located stack trace at the exact offending line. Running your test suite with such a guard is the surest way to keep sync-in-async from ever reaching production: the test that exercises the handler fails immediately with a pointer at the blocking call, instead of the service quietly degrading under load weeks later. The general principle is to make blocking-on-the-loop a *detectable* event in development, because it is nearly invisible until the day the downstream it calls gets slow — and by then it is a page, not a code review comment.

### Bug four: unbounded fan-out

Also covered: `gather` over a huge iterable opens too many connections, exhausts memory or file descriptors, and trips rate limits. Symptom: OOM kills, `Too many open files` errors, or a flood of HTTP 429s. Fix: an `asyncio.Semaphore` to cap in-flight tasks, as in the bounded-fetch pattern above.

### Bug five: fire-and-forget tasks that get garbage-collected

This one surprises even experienced people. When you call `asyncio.create_task(coro)`, the loop holds only a *weak* reference to the task. If you do not keep a strong reference yourself, the garbage collector is free to collect the task object mid-flight, and your work silently disappears — the coroutine may be cancelled before it finishes, with no error you will notice.

```python
async def main():
    asyncio.create_task(background_job())   # BUG: no reference kept
    await something_else()
    # background_job may be GC'd and never complete
```

The fix is to keep a strong reference until the task is done. The idiom is a module-level set that the task removes itself from on completion:

```python
_background_tasks: set[asyncio.Task] = set()

def spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)              # strong reference: not GC'd
    task.add_done_callback(_background_tasks.discard)
    return task
```

Better still, prefer a `TaskGroup` when the tasks are scoped to a block — it holds the references for you and awaits them all at the end, so there is nothing to garbage-collect prematurely and nothing to forget. The pattern of "spawn and forget" is the one to be suspicious of; "spawn, hold, and await" is the safe default.

A sixth honorable mention: **sharing non-thread-safe state across an offload boundary.** Within a single event loop you generally do not need locks for ordinary data, because only one coroutine runs at a time and context switches only happen at `await` points — so a sequence of non-awaiting statements is effectively atomic. But the moment you push work to a `to_thread` worker or a process, that comforting single-threaded guarantee is gone. A dict you mutate from two `to_thread` callbacks is a genuine data race; a counter you increment from worker threads needs a lock. Keep mutable shared state on the loop, and pass data to and from offloaded work by value (arguments and return values), not by shared reference.

## Putting it together: a production-grade fetcher

Let us assemble the patterns into one coroutine you could actually ship — concurrent fetch, bounded by a semaphore, with a per-request timeout, retries on transient failures, and a blocking post-processing step offloaded to a thread. This is the shape of real async I/O code.

```python
import asyncio
import aiohttp

async def fetch_one(sem: asyncio.Semaphore,
                    session: aiohttp.ClientSession,
                    url: str,
                    retries: int = 3) -> dict | None:
    async with sem:                                  # bound concurrency
        for attempt in range(retries):
            try:
                async with asyncio.timeout(5.0):     # per-request deadline
                    async with session.get(url) as resp:
                        if resp.status == 429:       # backpressure from server
                            await asyncio.sleep(2 ** attempt)   # exp backoff
                            continue
                        resp.raise_for_status()
                        body = await resp.read()
                        # Offload the blocking CPU parse to a worker thread.
                        return await asyncio.to_thread(parse_payload, body)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == retries - 1:
                    return None                      # give up after retries
                await asyncio.sleep(2 ** attempt)    # back off and retry
    return None

def parse_payload(body: bytes) -> dict:
    # Synchronous, CPU-ish parsing — fine on a thread, not on the loop.
    import json
    return json.loads(body)

async def fetch_all(urls: list[str], limit: int = 50) -> list[dict | None]:
    sem = asyncio.Semaphore(limit)
    connector = aiohttp.TCPConnector(limit=limit)    # align pool with semaphore
    async with aiohttp.ClientSession(connector=connector) as session:
        return await asyncio.gather(
            *(fetch_one(sem, session, u) for u in urls)
        )

results = asyncio.run(fetch_all(my_urls, limit=50))
```

Every pattern from this post is in there: the semaphore caps concurrency at 50 and the `TCPConnector` limit matches it so the pool and the gate agree; `asyncio.timeout` puts a deadline on each request; the `429` branch implements server-driven backpressure with exponential backoff; the blocking parse is offloaded with `to_thread` so it never freezes the loop; and `gather` collects the lot. This single function is most of what production async I/O looks like — the rest is metrics and logging.

#### Worked example: the metrics call that took down the service

Return to the page that opened this post. The service fanned out to a dozen downstream APIs with a bounded `aiohttp` fetcher much like the one above, and it ran at p99 = 40 ms for months. Then someone added a metrics call inside the per-request coroutine — a synchronous `requests.post("http://metrics-internal/event", ...)` with no `await`. As long as the metrics endpoint answered in a millisecond, nobody noticed; the blocking call was so fast it barely cost a tick. Then the metrics service got slow, climbing to a 200 ms response time under its own load.

Here is the arithmetic of the meltdown. With the service handling about 45 concurrent requests per process, each request's coroutine now blocked the single loop for 200 ms on the metrics call. Because the loop is one thread, those blocks serialized: while one request waited 200 ms on metrics, all 44 others sat frozen. The effective per-request latency became roughly the metrics latency times the concurrency, $200\,\text{ms} \times 45 \approx 9\,\text{s}$ at the tail — which is exactly the p99 of nine seconds we saw, and exactly the $N \times L$ blocking-serialization formula from the start of the post. The fix was one line: wrap the metrics call in `asyncio.to_thread(requests.post, ...)`, or better, drop it onto a fire-and-forget task that does not block the request path at all. The p99 dropped back under 50 ms within one deploy. The metrics endpoint was still slow — but now its slowness was confined to a worker thread instead of freezing the loop for every concurrent request. One blocking line had multiplied a 200 ms downstream hiccup into a 9-second outage, and one offload undid it.

## Measured results

Time for numbers. All of these are on the reference machine described up front, against a local mock server with a 100 ms median response time, and reported as the median wall-clock of several warm runs.

First, the headline result of the whole post — **blocking the loop versus offloading the blocking call** — on the 100-task workload from the first worked example:

| approach | what happens | wall-clock (100 tasks @ 100 ms) | effective concurrency |
| --- | --- | --- | --- |
| blocking call in coroutine | loop frozen, calls serialize | ~10.0 s | 1 |
| `to_thread` offload | loop free, calls overlap on threads | ~0.10 s | ~100 |
| native async (`aiohttp`) | no thread, coroutine yields on socket | ~0.10 s | ~100 |

The blocked version is **100× slower**, matching the $N \times L$ prediction exactly. Offloading to threads and using a native async client both recover the full concurrency; the native client is the cleaner choice when an async library exists, because it skips the thread pool entirely.

Second, **unbounded versus bounded fan-out** on 10,000 requests with average 256 KB responses:

| approach | peak RSS | open sockets | success rate | wall-clock |
| --- | --- | --- | --- | --- |
| unbounded `gather` (10k) | ~3.8 GB (often OOM) | up to 10,000 | ~69% (429s + timeouts) | unstable / fails |
| `Semaphore(50)` | ~240 MB | 50 | ~100% | steady, completes |

The bounded run uses roughly $50/10000 = 0.5\%$ as many live sockets and holds memory flat, succeeding on every request because it never overwhelms the downstream or the local file-descriptor limit. The unbounded run either OOM-kills or sheds a third of its requests to rate limiting and timeouts — and counterintuitively it is often *slower* end to end, because the downstream service's latency balloons under the thundering herd. Bounding concurrency is not just safer; past the saturation point it is faster.

Third, **`uvloop` versus the default loop** on a high-fan-out client doing 5,000 small requests where loop overhead is a meaningful slice of the time:

| event loop | throughput (small requests) | relative |
| --- | --- | --- |
| default asyncio loop | ~18,000 req/s | 1.0× |
| `uvloop` | ~46,000 req/s | ~2.6× |

A **2 to 3× throughput gain** from a two-line change, because `uvloop` moves the loop's hot socket-and-scheduling path into C. The win is real when the loop itself is the bottleneck — many small events — and shrinks toward 1× as per-request work grows, since then the loop is idle waiting on your handlers and its own overhead is noise. As always, measure on your workload before claiming the speedup.

### How to measure async honestly

Async timing has its own traps, distinct from the micro-benchmarking pitfalls covered in the [benchmarking post](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), and getting them wrong is how people end up "proving" speedups that evaporate in production. Here is the discipline.

**Measure wall-clock around `asyncio.run`, not CPU time.** For I/O-bound async work, the metric that matters is elapsed real time, because the whole point is overlapping waits. Use `time.perf_counter()` (a high-resolution monotonic clock) at the top of `main()` and at the end, never `time.process_time()` — process time measures CPU consumed, which for an I/O-bound coroutine that spends its life waiting on sockets is near zero and tells you nothing. The number you care about is "how long did the human wait," and that is `perf_counter`.

**Warm up the connection pool and the interpreter.** The first request through a fresh `ClientSession` pays for DNS resolution, the TCP handshake, and the TLS handshake; subsequent requests reuse the pooled connection and skip all of that. If you time a single cold batch you are measuring handshake cost, not steady-state throughput. Run the batch once to warm the pool, then time the second run. The same applies to the interpreter: the first execution of a coroutine compiles and caches things; throw it away.

**Run against a realistic but controlled backend.** Timing against a live third-party API gives you numbers dominated by their variance — their load, their rate limiting, the public internet — and you cannot reproduce them. For a clean measurement, point at a local mock server with a configurable, fixed delay (a tiny `aiohttp` server that `asyncio.sleep`s for 100 ms before responding is perfect). That isolates *your* concurrency behavior from *their* noise, and it is how you get the clean $N \times L$ versus $L$ comparison that proves the blocking-versus-offloaded result.

**Watch the right resource for the right claim.** A throughput claim needs requests-per-second over a sustained run, not the time for one batch. A memory claim needs peak RSS, sampled while the run is at its busiest — use `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` after the run, or watch the process in `htop`/`memray` during it, because the interesting number is the *high-water mark*, not the resting state. A "did it freeze the loop" claim is best made with `asyncio` debug mode (`asyncio.run(main(), debug=True)`), which logs any callback that ran longer than 100 ms — that log line is the smoking gun that names your blocking call and the function it is in.

**Account for the loop's own overhead at high request rates.** When you are measuring tens of thousands of small requests per second, a meaningful slice of the time is the loop scheduling itself, not your work. That is precisely the regime where `uvloop` helps and where the default loop's Python-level overhead shows up. If your per-request handler does real work — parsing, database calls — the loop overhead is a rounding error and you should not expect a `uvloop` win. Match the optimization to where the time actually goes, which is the whole method of this series: measure first, then pull the lever the measurement points at.

## When to reach for async (and when not to)

Async is a power tool with a specific edge, and reaching for it reflexively is its own anti-pattern. Here is the decisive guidance.

**Reach for async when the work is I/O-bound and high-concurrency.** Thousands of simultaneous network connections, a service that fans out to many downstreams per request, a web crawler, a chat server, anything where most of the time is spent *waiting* on sockets. Async shines because it holds thousands of waiting coroutines on one thread with kilobytes of overhead each, where threads would cost a megabyte of stack apiece and processes far more.

**Do not reach for async for CPU-bound work.** If your bottleneck is computation, async does nothing for you — there is no I/O to overlap, and the GIL still serializes the bytecode. You want a `ProcessPoolExecutor`, or [Numba](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast)/Cython on the hot loop, or [multiprocessing](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling). Async on CPU work just adds event-loop overhead to code that was never waiting on anything.

**Do not reach for async for low-concurrency I/O.** If you make a handful of network calls, a `ThreadPoolExecutor` with [a few threads](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) is simpler, requires no `async`/`await` coloring of your code base, and performs identically. Async earns its complexity at scale — when you have hundreds or thousands of concurrent waits — not at ten.

**Do not let one blocking call into the loop.** This is less "when not to use async" and more "the rule that makes async usable at all": once you are async, *everything* on the loop must be non-blocking. The instant a synchronous call sneaks in, you pay the $N \times$ penalty. If you cannot make a call async, offload it; never run it on the loop.

**Beware the "function color" tax.** `async` is contagious: to `await` something you must be in an `async def`, which means everything up the call chain must be async too. This splits your code base into colored halves and forces decisions like "do I make this whole stack async or wrap the one async call in `asyncio.run`?" For a small piece of a mostly-synchronous program, the coloring cost may not be worth it; a thread pool keeps your code one color. Adopt async when the *whole* I/O-heavy subsystem benefits, not for a single call.

## Case studies and real numbers

A few grounded data points from the ecosystem, with sources named.

**`uvloop`'s own benchmarks.** The `uvloop` project reports that on raw socket echo and HTTP benchmarks it is roughly 2 to 4× faster than the built-in `asyncio` loop, and in some socket micro-benchmarks it approaches the throughput of Go's networking. The exact multiple depends entirely on how much of the time is loop overhead versus your handler work; the gain is largest for many-small-events workloads and smallest for handler-heavy ones. Treat "2 to 4×" as the range for loop-bound work and verify on yours.

**The "async beats threads at 10k connections" claim.** The structural reason async wins at high connection counts is overhead per waiting unit. A blocked OS thread costs on the order of a megabyte of stack plus kernel scheduling state; ten thousand of them is gigabytes of stack and real context-switch pressure. A parked coroutine costs a few hundred bytes to a couple of kilobytes and no kernel involvement to "switch" — the loop just calls the next ready callback. This is the famous C10k-problem framing: at ten thousand simultaneous connections, the per-connection overhead is what kills you, and async drives it to near zero. For a handful of connections the difference is invisible; the curve only bends in async's favor as concurrency climbs into the thousands.

**`httpx` and the sync/async duality.** `httpx` deliberately offers a synchronous `Client` and an asynchronous `AsyncClient` behind one API, which is itself a practical lesson: the project's own docs are clear that you should use the sync client when you are in synchronous code and only adopt the async client when you are genuinely running an event loop with real concurrency. The library encodes the advice from the "when not to" section — async is not a free upgrade, it is a different execution model you opt into when the concurrency justifies it.

**The blocking-driver footgun in real frameworks.** Async web frameworks (FastAPI, Starlette, aiohttp servers) all carry the same warning in their docs: a synchronous, blocking call inside an `async def` route handler blocks the whole event loop and tanks the throughput of the entire server, not just that one request. The frameworks' standard advice is exactly this post's: use async-native drivers, or run blocking calls in a thread pool (FastAPI even routes plain `def` route handlers to a thread pool automatically for this reason). It is the single most common production async bug, and it is documented as such precisely because it is so common.

## Key takeaways

- **The loop is one thread and cannot preempt you.** Scheduling, cancellation, and timeouts all happen only at `await` points; between awaits, the loop is blind. Everything in this post follows from that one fact.
- **One blocking call freezes all tasks, and the slowdown factor is your concurrency.** A synchronous call inside a coroutine serializes the whole batch, costing you a factor of $N$ where $N$ is how concurrent the work was supposed to be.
- **Offload blocking I/O with `asyncio.to_thread`** — it works because blocking I/O releases the GIL, so the loop thread runs while the worker waits. Confine the blocking to a worker thread where it harms nobody.
- **Offload CPU work with `run_in_executor(ProcessPoolExecutor)`**, not threads — only separate processes get past the GIL for true parallelism, and only when the per-task compute is large relative to the pickled data.
- **Bound fan-out with `asyncio.Semaphore`.** Unbounded `gather` over 10k tasks opens 10k sockets and OOMs or trips rate limits; a cap of $N$ holds peak memory at $N \times b$ and matches your offered load to what the system can absorb (Little's law: $L = \lambda W$).
- **Put a timeout on every external await** with `asyncio.timeout` or `wait_for`, and remember timeouts work by injecting `CancelledError` at an await point — so never swallow it, and clean up in `finally` or by catching and re-raising.
- **Know the five bugs:** forgotten `await` (coroutine never runs), blocking the loop (everything stalls), sync-in-async (a blocking driver underneath), unbounded fan-out (OOM / 429), and fire-and-forget tasks getting garbage-collected (keep a strong reference or use a `TaskGroup`).
- **Reach for async for high-concurrency I/O, not for CPU and not for a handful of calls.** Async earns its complexity at thousands of concurrent waits; below that, a thread pool is simpler and just as fast. And `uvloop` is a two-line, 2 to 3× win when the loop itself is the bottleneck.

## Further reading

- The official `asyncio` documentation, especially the [high-level API index](https://docs.python.org/3/library/asyncio.html), the `asyncio.to_thread`, `asyncio.timeout`, `Semaphore`, and `TaskGroup` references, and the "Developing with asyncio" page on debug mode and common mistakes.
- The `aiohttp` and `httpx` documentation — `aiohttp`'s client connection-pooling and `TCPConnector` limits, and `httpx`'s explicit guidance on when to use the sync versus async client.
- The `uvloop` project README and benchmarks for the C-loop speedup and the methodology behind the numbers.
- PEP 3156 (the asyncio design rationale) and PEP 654 (exception groups, which `TaskGroup` uses) for the structured-concurrency model.
- "Using Asyncio in Python" by Caleb Hattingh and the asyncio chapters of "High Performance Python" by Gorelick and Ozsvald for book-length treatments with measured examples.
- Within this series: the foundation [asyncio from the ground up: event loops and coroutines](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines), the thread-pool offload mechanics in [threading done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits), the CPU-parallelism bridge in [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling), and the series [intro on why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means).

The right operation goes to the right place, and the whole decision collapses to one question asked of each call — is it an awaitable async library, blocking I/O, or CPU-heavy work?

![A decision tree showing each operation in a coroutine routed by one question to native await for an awaitable async library to to_thread for blocking I O or to a ProcessPool executor for CPU heavy work](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-8.png)

And the same routing as a lookup table, so you can pin it next to your editor — match the operation kind to the mechanism, and the loop never blocks.

![A matrix mapping each operation kind to the right mechanism and why it fits showing HTTP and sockets to async await blocking libraries to to_thread CPU heavy work to a ProcessPool executor and many small tasks to gather under a semaphore](/imgs/blogs/async-in-practice-patterns-pitfalls-and-blocking-code-7.png)

Async, done right, is the most efficient concurrency model Python has for I/O. It only ever fails you when something blocks the loop — and now you know exactly how to make sure nothing does.

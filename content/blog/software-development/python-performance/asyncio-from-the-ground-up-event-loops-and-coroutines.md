---
title: "Asyncio From the Ground Up: Event Loops and Coroutines"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Build asyncio from first principles: how one thread runs an event loop that cooperatively multiplexes thousands of waits, what await actually does when it suspends a coroutine, and why async beats threads for ten thousand connections."
tags:
  [
    "python",
    "performance",
    "asyncio",
    "concurrency",
    "event-loop",
    "coroutines",
    "io-bound",
    "optimization",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-1.png"
---

The first time asyncio earned its keep for me, the problem was a crawler. We needed to fetch the status of about twelve thousand URLs every few minutes — health checks against partner endpoints, most of which answered in 80 to 200 milliseconds and then sat there doing nothing while the network round-tripped. The first version was a plain loop with `requests.get`, and it took eleven minutes per sweep because it waited for each response before starting the next. The obvious fix was threads, so we threw a `ThreadPoolExecutor` at it with two hundred workers, and it dropped to about forty seconds — better, but the box was now running two hundred OS threads, each carrying a stack measured in megabytes, and the memory chart had a new staircase. When someone suggested bumping the pool to two thousand workers to go faster, the machine started swapping and the whole thing got *slower*. Two thousand threads is two thousand kernel-scheduled stacks, and the operating system spends real time shuffling between them.

So we rewrote the fetch loop with asyncio and `aiohttp`. One thread. Twelve thousand coroutines in flight, each one a tiny Python object holding a saved stack frame rather than an OS thread holding a megabyte of stack. The sweep finished in about four seconds, the process sat at a couple hundred megabytes of RSS instead of climbing toward swap, and the CPU was mostly idle because the work was almost entirely *waiting* — which is exactly the kind of work one thread can do for thousands of connections at once, as long as it never blocks on any single one. That is the whole idea of asyncio, and by the end of this post you will be able to build the mental model from scratch: what the event loop is, what a coroutine is, what `await` does at the moment it runs, why this crushes threads for massive I/O concurrency, and the one limit that bites everyone — it is still a single thread, so one blocking call or one CPU-heavy function freezes everything.

![A directed acyclic diagram showing one tick of the event loop where a ready queue and an io selector both feed a single loop thread that picks a task, runs it until it awaits, suspends it, marks io pending, and later resumes it when the file descriptor is ready](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-1.png)

This is rung four of the leverage ladder this series keeps climbing. After you have [done less work with the right algorithm and data structure](/blog/software-development/python-performance/choosing-the-right-built-in-data-structure-list-dict-set-tuple) and [done it in bulk with NumPy](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast), the last lever is to use every core and overlap I/O. For CPU-bound work that means processes; for I/O-bound work — waiting on networks, disks, databases, and other services — it usually means async. This post is the foundational half of the async story: the machinery and the mental model. The companion post, [async in practice](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code), covers the real-world patterns and the pitfalls — bounding concurrency, mixing in blocking calls, timeouts, and the bugs you will actually hit. Here we build the engine so that post makes sense.

## What problem is asyncio actually solving?

Before any code, get the problem precisely right, because asyncio is a sharp tool for one specific shape of work and a bad tool for everything else.

A huge amount of real software spends most of its wall-clock time *waiting*. A web service waits on a database query. A crawler waits on remote servers. A proxy waits on an upstream. A chat backend holds ten thousand open WebSocket connections, almost all of which are idle at any given instant, waiting for someone to type. In all of these, the CPU is barely working. The expensive thing is the wait — a network round trip that takes tens or hundreds of milliseconds, during which a modern CPU could have executed hundreds of millions of instructions. This is **I/O-bound** work: limited by waiting for input and output, not by computation.

When work is I/O-bound, the question is never "how do I compute faster" — it is "how do I wait on ten thousand things at once without paying ten thousand times the overhead." That is the entire game. And there are exactly two strategies the operating system offers.

The first strategy is **one thread per wait**. You spin up a thread (or process) for each connection, and each one blocks on its own socket. The kernel handles the rest: when data arrives on socket 4,217, the kernel wakes the thread parked on it. This is simple to reason about — each connection gets its own little linear program — but it is expensive, because a thread is a heavyweight kernel object. Every thread needs its own **stack**, a contiguous region of memory the function-call chain runs on, and on Linux that stack defaults to around 8 MB of virtual address space (the resident part is smaller, but it grows as the call chain deepens, and the kernel still tracks the whole reservation). Ten thousand threads is a lot of stacks. Worse, the kernel scheduler has to time-slice between all of them, and each switch flushes registers, reloads page-table state, and pollutes the CPU caches. Threads are great up to dozens or low hundreds; they fall apart at tens of thousands.

The second strategy is **one thread watching all the waits**. Instead of one thread per socket, you use a single thread that asks the operating system one question repeatedly: "of these ten thousand sockets I care about, which ones are ready for me to read or write *right now*?" The OS answers cheaply through a facility called a **readiness selector** — `epoll` on Linux, `kqueue` on macOS and BSD, IOCP-style mechanisms on Windows. The thread then services exactly the sockets that are ready, and goes back to waiting on the rest. No socket gets its own thread; one thread multiplexes them all. This is the model asyncio is built on, and it is why one thread can hold ten thousand connections in a couple hundred megabytes instead of tens of gigabytes.

Asyncio's job is to make strategy two *ergonomic*. Writing a raw `epoll` loop by hand — registering file descriptors, decoding readiness events, dispatching to the right callback, threading state through callbacks — is miserable and was historically how this was done (the "callback hell" era). Asyncio wraps the selector in an **event loop**, and lets you write each connection's logic as a normal-looking linear function — a **coroutine** — that simply `await`s whenever it needs to wait. The `await` is the magic word: it suspends your coroutine and hands the single thread back to the loop, so the loop can go service some other ready connection while yours waits. When your data finally arrives, the loop resumes your coroutine right where it left off. You get the readability of one-linear-program-per-connection with the efficiency of one-thread-for-everything.

That sentence is the whole post in one line: **async is cooperative single-threaded I/O multiplexing.** Cooperative, because each coroutine voluntarily yields control at every `await` (nothing preempts it). Single-threaded, because one thread runs the loop and all the coroutines. I/O multiplexing, because that one thread watches many I/O sources at once through the selector. Everything else is detail. Let us build up each word.

## The cost model: why 10,000 threads cost gigabytes and 10,000 coroutines cost megabytes

I want to make the central claim rigorous before we touch the API, because this number — the per-wait overhead — is the entire reason async exists. Throughout the post I will quote measurements from a fixed reference machine so the numbers are concrete and comparable: an 8-core x86-64 Linux box (or an Apple M2 of similar vintage), CPython 3.12, 16 GB of RAM. When I give a wall-clock figure, assume that machine unless I say otherwise.

Start with threads. A POSIX thread is a kernel-scheduled execution context, and its dominant cost is the **stack**. On 64-bit Linux the default thread stack size is 8 MB of *virtual* address space (`ulimit -s` shows 8192 KB). Virtual address space is not all resident in physical RAM — pages are faulted in lazily as the call chain touches them — so a thread that only ever nests a few frames deep might resident-occupy tens of kilobytes. But the reservation matters: ten thousand threads at 8 MB each reserve 80 GB of virtual address space, which on a 16 GB box you simply cannot back, and even modest resident growth as call chains deepen pushes you toward swap. People reduce the stack size (you can request, say, 512 KB per thread), but 512 KB times ten thousand is still 5 GB, and you have made every deep call chain a potential stack overflow. There is also a hard per-process thread-count ceiling and real kernel bookkeeping per thread. Beyond memory, there is the **context-switch tax**: the kernel scheduler must rotate through runnable threads, and each switch costs on the order of a microsecond of pure overhead (saving and restoring registers, swapping the kernel stack, often a TLB and cache disturbance). With thousands of runnable threads, the scheduler itself becomes a measurable fraction of your CPU.

Now coroutines. A Python coroutine is not a kernel object at all — it is a plain heap-allocated Python object that holds a **suspended stack frame**: the local variables, the instruction pointer (which bytecode to resume at), and a small amount of bookkeeping. There is no separate stack reservation, because the coroutine runs on the one thread's normal C stack while it is executing, and when it suspends, only its frame's state is retained on the heap. The cost is therefore on the order of the data a single function call already needs — roughly a kilobyte or two per coroutine for a typical handler, dominated by the frame and its locals. Wrap it in an `asyncio.Task` to schedule it and you add a modest fixed object on top, but you are still talking single-digit kilobytes. Ten thousand coroutines is on the order of ten to thirty megabytes of Python objects, not gigabytes of stacks. And there is no kernel context switch between them: switching from one coroutine to another is the event loop returning from one coroutine's `send` and calling another's — a function call and a dictionary-ish lookup inside one thread, no syscall, no scheduler, no cache flush.

That contrast is the whole argument, and the next figure makes the two columns concrete and parallel — same connection count, two completely different cost structures.

![A before and after comparison contrasting one OS thread per connection with megabyte stacks and gigabytes of total memory plus context switching against one event loop thread running thousands of coroutines that each cost about a kilobyte](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-2.png)

Put a back-of-the-envelope on it. Let $n$ be the number of concurrent connections, $s$ the per-thread stack reservation, and $c$ the per-coroutine object size. The thread model needs roughly $n \cdot s$ of address space; the coroutine model needs roughly $n \cdot c$ plus one fixed loop. With $s \approx 8\,\text{MB}$ and $c \approx 2\,\text{KB}$, the ratio is about $s/c \approx 4000$. So at the same connection count, the async approach uses on the order of *thousands of times less memory* for the per-connection state. That is not a tuning win; it is a different order of magnitude, and it is why "ten thousand connections" is the canonical async example. The C10k problem — handling ten thousand concurrent connections on one machine — is solved by the selector-plus-coroutine model precisely because the per-connection cost collapses.

There is a second, subtler cost the table hints at: **GIL contention**. CPython has a Global Interpreter Lock, [a single lock that lets only one thread execute Python bytecode at a time](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs). Threads that do I/O release the GIL while they wait, so they overlap their *waits* fine — that is why threads do help I/O-bound work. But the threads still have to re-acquire the GIL to run any Python between waits, and with thousands of threads waking up around the same time, they pile up on that one lock, and the interpreter spends time arbitrating who gets it next. Asyncio sidesteps this entirely: one thread means there is never any contention for the GIL, because there is only ever one thing wanting it. You traded "many threads fighting over one lock" for "one thread that never has to fight."

| Property | One thread per connection | One event loop, many coroutines |
| --- | --- | --- |
| Per-connection memory | a thread stack, megabytes | a coroutine object, a kilobyte or two |
| 10k connections, total | tens of GB of stack reservation | tens of MB of Python objects |
| Switching mechanism | kernel scheduler, a syscall | a function call inside one thread |
| Switching cost | about a microsecond, cache disturbance | tens of nanoseconds, no syscall |
| GIL | many threads contend on re-acquire | one thread, zero contention |
| Scales to | dozens to low hundreds comfortably | tens of thousands of connections |
| Failure mode | swap, scheduler thrash | one blocking call freezes everything |

The last row is the catch, and we will spend a whole section on it. But first, the engine.

It helps to quantify the context-switch tax directly, because "about a microsecond" sounds small until you multiply it. Suppose your server processes a burst where ten thousand connections each have a little Python to run between waits. Under threads, servicing them means the kernel rotating through ten thousand runnable threads, and if each switch costs roughly $1\,\mu\text{s}$ of pure scheduler-and-cache overhead, that burst spends about $10{,}000 \times 1\,\mu\text{s} = 10\,\text{ms}$ on switching *alone*, before a single byte of useful work — and that overhead recurs on every burst, scaling linearly with the thread count. Under one event loop, "switching" from one coroutine to the next is the loop returning from one `.send` and calling the next, on the order of tens of nanoseconds, so the same ten thousand resumes cost on the order of $10{,}000 \times 50\,\text{ns} = 0.5\,\text{ms}$ — roughly twentyfold cheaper, and crucially it does not involve the kernel scheduler at all. The gap widens as concurrency climbs, because the kernel scheduler's own bookkeeping grows with the number of runnable threads while the loop's per-resume cost stays flat. This is the quiet, second reason high-concurrency servers favor the event-loop model: not just the memory, but the switching itself is an order of magnitude cheaper when it never touches the kernel.

## The event loop: one thread that runs ready tasks until each one awaits

Strip away the syntax and asyncio is one loop running on one thread. The loop holds two things: a **ready queue** of callbacks and coroutine steps that can run *right now*, and a **selector** registered with file descriptors that the loop is waiting on. A single tick of the loop does this, and only this:

1. Run everything currently in the ready queue, one item at a time, to completion-or-suspension. Each item is a small step of work: a coroutine resumed until its next `await`, or a callback fired. Critically, each item runs *uninterrupted* until it voluntarily yields (hits an `await` that actually suspends) or finishes. Nothing preempts it.
2. When the ready queue empties, ask the selector: "which of my registered file descriptors are ready, and is there a timer due?" The loop *blocks here* — this is the only place a healthy loop blocks — sleeping in `epoll_wait`/`kqueue` until at least one fd is ready or the nearest timer fires.
3. For each ready fd, schedule the coroutine that was waiting on it back onto the ready queue (it becomes runnable again). For each due timer, schedule its callback.
4. Go back to step 1.

That is the entire loop. Look again at figure 1 with this in mind: the ready queue and the selector are the two parallel sources that feed the loop; the loop picks a ready task, runs it until it `await`s, suspends it, registers its fd as pending with the selector, and — in a *separate, later* tick — the selector reports that fd ready and the task is rescheduled. I drew it as an acyclic flow on purpose. The loop is a cycle in the sense that it goes round and round forever, but the *causal* structure of one tick is a clean directed flow: ready → run → suspend → pending, and independently, fd-ready → resume. Keeping those two halves separate in your head is the key to understanding why nothing ever spins.

The crucial invariant, the one thing to tattoo on your brain: **a coroutine runs until it `await`s, and then control returns to the loop.** Between two `await` points, your code has the thread entirely to itself — no other coroutine runs, no preemption, no surprise interleaving. This is *cooperative* scheduling, in contrast to the *preemptive* scheduling the OS does to threads (where the kernel can yank the CPU from a thread at any instruction). Cooperative scheduling is a double-edged sword we will sharpen later: it makes reasoning about shared state easy (you only yield where you write `await`), but it means a coroutine that *never* awaits — because it is grinding on a CPU loop or stuck in a blocking call — never gives the thread back, and the whole loop, every other connection included, freezes behind it.

Notice the one thing the loop *does not* do: it never busy-waits. A naive way to watch many sockets would be to spin in a tight loop asking each one "anything yet? anything yet?" — burning a full CPU core checking sockets that are mostly idle. The selector is precisely what avoids that. When the ready queue is empty, the loop calls `epoll_wait` (or the `kqueue` equivalent) and the thread *sleeps* — consuming no CPU — until the kernel itself wakes it because a socket became ready or a timer came due. So an asyncio server holding ten thousand idle connections sits at essentially zero percent CPU, parked in one `epoll_wait`, and springs to life only when there is genuine work. That property — many idle connections cost almost nothing — is what makes the model viable for long-lived connections like WebSockets and server-sent events, where the overwhelming majority of connections are idle at any instant. A thread-per-connection design pays for those idle connections in stacks and scheduler slots; the event loop pays for them in nearly nothing.

There is one subtlety worth naming so it does not surprise you. Because scheduling is cooperative and a resumed task runs to its next `await` before the loop regains control, a coroutine that runs *briefly* between awaits is being a good citizen, but the loop's responsiveness is bounded by the *longest* run any single task takes between yields. If every task always yields within a few microseconds, the loop services thousands of connections with microsecond-level fairness. If one task occasionally runs for fifty milliseconds straight without awaiting, then once per such occurrence *every other task* waits up to fifty milliseconds. The loop is fair only to the extent your coroutines are frequent yielders. Keep that in your peripheral vision; it is the seed of the blocking-call section later.

Let us see the smallest possible asyncio program, just to anchor the API before we go deeper.

```python
import asyncio


async def greet(name: str) -> str:
    # `async def` makes this a coroutine function: calling it returns a
    # coroutine object, it does NOT run the body yet.
    await asyncio.sleep(1)  # suspend here, yield to the loop for ~1 second
    return f"hello, {name}"


async def main() -> None:
    result = await greet("world")  # await runs the coroutine to completion
    print(result)


asyncio.run(main())  # creates an event loop, runs main() to completion, closes it
```

`asyncio.run(main())` is the front door. It creates a fresh event loop, schedules `main()` as the first task, runs the loop until `main()` finishes, then closes the loop. Inside, `await asyncio.sleep(1)` is the interesting part: it does *not* block the thread for a second. It registers a one-second timer with the loop and suspends `greet`, handing the thread back. If there were other coroutines queued, the loop would run them during that second. Because there is nothing else to do here, the loop simply sleeps in the selector until the timer is due, then resumes `greet`. The difference between `time.sleep(1)` (which blocks the whole thread, dead) and `await asyncio.sleep(1)` (which suspends one coroutine, thread stays free) is the difference between async working and async being pointless. We will hammer that distinction.

## Coroutines: `async def`, and what `await` actually compiles to

A **coroutine** is a function that can suspend itself partway through and be resumed later, picking up exactly where it left off, with all its local variables intact. Python spells coroutine functions with `async def`. The defining property is that calling one does *not* run it:

```python
async def fetch(url: str) -> str:
    ...

c = fetch("https://example.com")  # c is a coroutine object; nothing ran yet
print(type(c))                    # <class 'coroutine'>
```

`fetch("...")` returns a coroutine object — a paused recipe holding its arguments and a pointer to the first line of its body. To *drive* it, something has to repeatedly resume it. You usually never do this by hand; `await` and the event loop do it for you. But understanding the mechanism underneath demystifies everything, so let us look at what `await` actually is.

Under the covers, a coroutine is built on the same machinery as a generator. A generator suspends at `yield` and resumes when you call `.send()` on it; a coroutine suspends at `await` and is resumed by whoever is driving it (the loop) calling `.send()` on it. When you write `await something`, Python roughly does this: it asks `something` for an iterator of "waitable steps" (via `__await__`), and it pulls values out of that iterator, *yielding each one up to the driver*. Each value that bubbles all the way up to the event loop is, in effect, the coroutine saying "I am parked; here is what I am waiting for; wake me when it is ready." The loop registers that wait (a file descriptor with the selector, or a timer), stops resuming this coroutine, and goes off to run others. When the wait completes, the loop calls `.send()` again, and execution continues from immediately after the `await`.

So the answer to "what does `await` compile to" is: **a suspension point that yields control up to the event loop.** It is, mechanically, a `yield` that propagates out of your coroutine, through any coroutines that awaited it, up to the loop that is driving the whole chain. The loop catches that yield, sees what you are waiting on, parks you, and runs someone else. Your stack frame — locals, the line you were on — is held alive in the coroutine object on the heap the entire time. When you resume, nothing was lost; you are exactly where you stopped.

![A directed acyclic diagram of await as a yield point showing a running coroutine that hits await, saves its frame and hands control to the loop while registering its file descriptor, then resumes at the same line with the same locals once the descriptor is ready](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-7.png)

This is why an `await` can only appear inside an `async def` (or an async comprehension / async generator): the suspension has to have somewhere to yield *to*, and that machinery only exists inside a coroutine. It is also why `await` is a precise yield *point* — it is the exact set of places your coroutine can be suspended. Anywhere there is no `await`, your code runs straight through with the thread to itself. That precision is the gift of cooperative scheduling, and the trap: if you go a long time between `await`s — a tight numeric loop, a giant `json.loads`, a synchronous `requests.get` — you are holding the thread that long, and every other coroutine waits behind you.

The next figure shows the cooperative dance with two coroutines, A and B, sharing one thread. A runs, starts a request, and awaits it — which yields to the loop. The loop runs B while A's I/O is in flight. When A's data is ready, the selector wakes A, and the loop resumes A from its `await`. Notice that A and B are never running at literally the same instant — there is one thread — but their *waits* overlap completely. That is concurrency without parallelism, and for I/O it is exactly enough, because the expensive thing was the wait, and the waits ran in parallel even though the code did not.

![A timeline of cooperative scheduling on one thread where coroutine A runs and starts a request then awaits and yields to the loop, the loop runs coroutine B doing useful work, and when A's file descriptor is ready the selector wakes A and it resumes from where it left off](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-3.png)

#### Worked example: sequential awaits leave the thread idle

Let us prove the "concurrency without parallelism" claim with timing, because this is the single most common beginner mistake — writing async code that is no faster than synchronous code because the `await`s are serialized.

```python
import asyncio
import time


async def fake_request(label: str, delay: float) -> str:
    # Stand-in for a network call: it WAITS `delay` seconds, no CPU work.
    await asyncio.sleep(delay)
    return f"{label} done"


async def sequential() -> None:
    start = time.perf_counter()
    a = await fake_request("A", 1.0)   # wait 1s here, fully, before moving on
    b = await fake_request("B", 1.0)   # only now start B, wait another 1s
    elapsed = time.perf_counter() - start
    print(a, b, f"-- {elapsed:.2f}s")


asyncio.run(sequential())
```

```bash
A done B done -- 2.00s
```

Two one-second waits, run back to back, take two seconds. This async code is doing nothing useful: each `await` suspends the coroutine, but there is no *other* coroutine queued to run during the wait, so the loop just sleeps a full second, resumes, and does the second wait. We wrote `async` and got synchronous behavior. The lesson: `await` does not create concurrency; it creates a *suspension point*. Concurrency comes from having multiple coroutines *in flight at once* so their suspensions overlap. That is what tasks and `gather` are for, and they are next.

## Driving a coroutine by hand: the generator protocol underneath

To make the suspension mechanism completely concrete — not magic — let us drive a coroutine without any event loop at all, by hand, using the exact protocol the loop uses internally. A coroutine object exposes `.send(value)`, which resumes it (passing `value` in as the result of the `await` it was parked on) and runs it until its next suspension or its return. When the coroutine finishes, `.send` raises `StopIteration`, and the return value is attached to that exception. This is identical to how you drive a generator with `.send`, because coroutines are built on the same C-level frame machinery as generators.

```python
import types


@types.coroutine
def park():
    # `yield` here is the low-level suspension: it bubbles a value up to
    # whoever is driving this coroutine. A real loop yields a "wait for fd"
    # request; we just yield a marker so we can watch the suspension happen.
    received = yield "I am parked, wake me later"
    return f"resumed with {received!r}"


async def work():
    # `await park()` propagates park()'s yield all the way up to our driver.
    result = await park()
    return f"done: {result}"


# Drive it by hand, exactly like the event loop would.
coro = work()                  # nothing has run yet
parked_value = coro.send(None)  # start it; runs until the yield inside park()
print("coroutine yielded:", parked_value)
try:
    coro.send("data is ready")  # resume; runs to the return, raising StopIteration
except StopIteration as stop:
    print("coroutine returned:", stop.value)
```

```bash
coroutine yielded: I am parked, wake me later
coroutine returned: done: resumed with 'data is ready'
```

Read that transcript against the model. The first `coro.send(None)` *starts* the coroutine (you always prime it with `None`) and runs it until the `yield` inside `park()` — that `yield` is the suspension point, and the value it produces bubbles all the way up through `await` to our hand-driver. That is precisely what reaches the event loop in real asyncio: a request that says "park me, here is what I am waiting on." Our driver then does what the loop does — it decides the wait is satisfied and calls `coro.send("data is ready")`, which feeds that value back in as the result of the suspended `await` and runs the coroutine to its `return`, surfacing as `StopIteration.value`. There is no thread switch, no syscall, nothing hidden: resuming a coroutine is a plain method call that re-enters a saved frame. The event loop is *only* the part that decides *when* to call `.send` again — namely, when the selector reports the awaited file descriptor is ready. Everything you have read about "the loop resumes the coroutine" reduces to this `.send` call. That is the entire suspension-and-resume mechanism, and it is why a coroutine costs a frame on the heap rather than a stack: between sends, all that survives is the paused frame.

This also explains a rule that otherwise looks arbitrary: you may only `await` things that know how to participate in this protocol — coroutines, Tasks, Futures, and objects with `__await__`. Awaiting a plain value (`await 5`) is a `TypeError`, because `5` has no `__await__` to yield up a suspension. The `await`able protocol is the contract that lets a wait propagate from deep inside your call chain all the way to the loop.

## Tasks: scheduling coroutines to run concurrently

A bare coroutine is inert until something drives it, and `await`ing one drives it to completion *before your code continues* — that is the serialization we just saw. To get concurrency, you must hand multiple coroutines to the loop so it can interleave them. The primitive that does this is `asyncio.create_task`.

```python
task = asyncio.create_task(fetch(url))
```

`create_task` wraps a coroutine in a `Task` and *schedules it onto the loop's ready queue immediately*. The task starts running concurrently with whatever you do next; you have not awaited it, so your code keeps going. A `Task` is the unit the event loop actually schedules — it is the thing that gets resumed and suspended; the coroutine inside is just the recipe. Later, when you do `await task`, you wait for that already-running work to finish and collect its result (or re-raise its exception). The split is the whole trick: **create the task now (it starts), await it later (you collect).** Between those two moments, the task makes progress whenever your code yields control by awaiting something.

```python
import asyncio
import time


async def fake_request(label: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{label} done"


async def concurrent() -> None:
    start = time.perf_counter()
    # Schedule BOTH onto the loop now; both start their waits.
    ta = asyncio.create_task(fake_request("A", 1.0))
    tb = asyncio.create_task(fake_request("B", 1.0))
    # Now await them. While we wait on A, B's wait is already running.
    a = await ta
    b = await tb
    elapsed = time.perf_counter() - start
    print(a, b, f"-- {elapsed:.2f}s")


asyncio.run(concurrent())
```

```bash
A done B done -- 1.00s
```

One second, not two. The instant we `create_task`'d both, both started their `asyncio.sleep`. When we `await ta`, our coroutine suspends; the loop, with the thread free, lets B's sleep run too; both timers fire at the one-second mark; both tasks finish; we collect. The two waits *overlapped*, so the wall clock is the *max* of the delays, not the sum. That is the core async win, stated precisely: for $N$ independent waits of durations $d_1, \dots, d_N$, sequential awaits cost $\sum_i d_i$, while concurrent tasks cost $\max_i d_i$ (plus a tiny scheduling overhead). For our crawler with twelve thousand 100-millisecond fetches, sequential would be twenty minutes and concurrent is a fraction of a second — limited only by how fast the selector and the network let us push.

There is a sharp edge worth flagging early, because it bites everyone once: if you `create_task` and then your function returns without ever awaiting the task, the task may be garbage-collected and silently never finish — and you will not get its exception. You must keep a reference and eventually await it (or use a `TaskGroup`, below, which does the bookkeeping for you). "Fire and forget" in asyncio is really "fire and lose."

## `gather`: awaiting many at once

Creating tasks one by one and awaiting them in sequence works, but the common case — "run these N things concurrently and give me all their results" — has a dedicated primitive: `asyncio.gather`.

```python
import asyncio
import time


async def fake_request(label: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{label} done"


async def gathered() -> None:
    start = time.perf_counter()
    # gather schedules all coroutines concurrently and waits for all of them,
    # returning results in the SAME ORDER as the arguments (not completion order).
    results = await asyncio.gather(
        fake_request("A", 1.0),
        fake_request("B", 0.5),
        fake_request("C", 1.5),
    )
    elapsed = time.perf_counter() - start
    print(results, f"-- {elapsed:.2f}s")


asyncio.run(gathered())
```

```bash
['A done', 'B done', 'C done'] -- 1.50s
```

`gather` takes coroutines (or tasks), schedules them all to run concurrently, and returns when *all* have finished, with results in argument order. The wall clock here is 1.5 seconds — the slowest of the three — because all three waits overlapped. The before-and-after of this is the single most important picture in async, so here it is: sequential awaits sum the waits; `gather` overlaps them and the slowest one dominates.

![A before and after comparison contrasting awaiting one call after another for a total of two hundred milliseconds against gathering both calls so their waits overlap and the slowest at one hundred milliseconds dominates](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-6.png)

Two `gather` behaviors you must know, because they cause real bugs. First, **ordering**: results come back in the order you passed the arguments, *not* the order they finished. B finished first above (0.5s) but still lands second in the list. If you need completion order — say, to stream results as they arrive — use `asyncio.as_completed` instead, which yields each result the moment it is ready. Second, **errors**: by default, if any gathered coroutine raises, `gather` propagates that exception immediately, but the *other* coroutines keep running in the background (they are not cancelled), which can leak work and resources. Passing `return_exceptions=True` flips this to collect exceptions as results instead of raising, so one failure does not blow up the batch. The default error behavior of `gather` is the main reason the newer `TaskGroup` exists, which we turn to now.

When you want results *as they finish* rather than all at the end — to update a progress bar, to start processing the first response without waiting for the slowest — `as_completed` is the tool. It returns an iterator of awaitables that yield in completion order:

```python
import asyncio


async def fake_request(label: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return label


async def stream_results() -> None:
    coros = [fake_request("slow", 1.5), fake_request("fast", 0.3),
             fake_request("mid", 0.8)]
    # Yields each result the instant it is ready, not in argument order.
    for finished in asyncio.as_completed(coros):
        label = await finished
        print("arrived:", label)


asyncio.run(stream_results())
```

```bash
arrived: fast
arrived: mid
arrived: slow
```

The fast request lands first even though it was passed second, because `as_completed` reflects *when* each coroutine finished, not the order you listed them. Use `gather` when you need all results together in order; use `as_completed` when you want to react to each result the moment it arrives. Both run the underlying coroutines concurrently — the difference is purely how you collect.

One more primitive belongs in this neighborhood even though its real workout is in the [practice post](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code): the `asyncio.Semaphore`, for *bounding* concurrency. Launching twelve thousand fetches with `gather` will indeed start twelve thousand coroutines, but the server (and your local file-descriptor limit) may not tolerate twelve thousand simultaneous connections. A semaphore caps how many run at once:

```python
import asyncio

sem = asyncio.Semaphore(100)  # at most 100 concurrent at any instant


async def bounded_fetch(url: str) -> str:
    async with sem:               # await a slot; suspends if 100 are in flight
        await asyncio.sleep(0.1)  # stand-in for the real awaited I/O
        return url
```

The `async with sem` acquires a slot, suspending the coroutine if all hundred slots are taken until one frees up. This keeps thousands of *queued* coroutines cheap (they are just parked frames waiting on the semaphore) while only a hundred are actively connected at any moment. It is the standard way to fan out a huge batch without overwhelming the other end — and a clean illustration that asyncio's coordination primitives (`Semaphore`, `Lock`, `Event`, `Queue`) are themselves just things you `await`, parking on the loop like any other wait.

This is a good moment to put a name on each primitive, because people conflate them. The next figure lays the five out: `async def` defines a coroutine, `await` is the suspension point, `create_task` schedules concurrency, `gather` awaits a batch, and `TaskGroup` does the same with structured cleanup.

![A matrix of the five asyncio primitives showing for async def, await, create task, gather, and TaskGroup what each one does and what kind of concurrency it provides from none for a bare coroutine to overlapped waits with cancel on error for a task group](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-5.png)

## Structured concurrency: `asyncio.TaskGroup` (3.11+)

`gather`'s loose error handling — one task fails, the rest run on orphaned — is a footgun, and so is bare `create_task`, where forgetting to await a task loses its result and its exceptions. Python 3.11 introduced **structured concurrency** to fix both, through `asyncio.TaskGroup`. The idea borrowed from the structured-concurrency movement: tasks should have a clear scope, like a block, and when that block exits, *all* tasks it spawned are guaranteed to be done — completed, or cancelled and cleaned up. No orphans, no leaks, no swallowed errors.

```python
import asyncio


async def fetch(label: str, delay: float) -> str:
    await asyncio.sleep(delay)
    if label == "B":
        raise ValueError("B failed")  # simulate one task erroring
    return f"{label} done"


async def main() -> None:
    results = {}
    try:
        async with asyncio.TaskGroup() as tg:
            # Tasks created in the group are scheduled concurrently.
            t_a = tg.create_task(fetch("A", 1.0))
            t_b = tg.create_task(fetch("B", 0.5))  # this one will raise
            t_c = tg.create_task(fetch("C", 1.5))
        # We only reach here if ALL tasks succeeded.
        results = {"A": t_a.result(), "C": t_c.result()}
    except* ValueError as eg:  # except* unpacks an ExceptionGroup
        print("a task failed:", [str(e) for e in eg.exceptions])


asyncio.run(main())
```

```bash
a task failed: ['B failed']
```

Read what happened carefully, because it is exactly the behavior you want. All three tasks started concurrently. B raised at the half-second mark. The `TaskGroup` immediately **cancelled the still-running siblings** (A and C, which had not finished) and then, when the `async with` block exited, re-raised the failure as an `ExceptionGroup` — a 3.11 type that bundles multiple concurrent failures, because more than one task can fail at once. We catch it with `except*` (the new "except a group" syntax). The guarantee is airtight: when control leaves the `async with` block, *every* task in the group is finished — succeeded, failed, or cancelled. There are no orphaned tasks running in the background, no leaked connections, no exception silently dropped. This is what "structured" means: the lifetime of the spawned tasks is bounded by the syntactic block, the same way a function's locals are bounded by the function.

For new code, prefer `TaskGroup` over `gather`. It is more code by a couple of lines, but it eliminates two of the nastiest classes of async bug — leaked tasks and swallowed exceptions — by construction. Reach for `gather` when you genuinely want the "collect everything, including failures, and keep going" semantics (with `return_exceptions=True`), and for `as_completed` when you want results streamed in finish order.

## The selector and readiness: how the loop wakes the right coroutine

I have been hand-waving "the selector wakes the coroutine when its fd is ready." Let us make that concrete, because it is the bridge between the high-level coroutine story and the operating system, and it is where the "one thread watching everything" magic actually lives.

A **file descriptor** (fd) is a small integer the OS gives you for an open resource — a socket, a pipe, a file. When you make a network connection in async code, under all the layers there is a non-blocking socket with some fd, say 42. "Non-blocking" is essential: a normal (blocking) socket read parks the calling thread until data arrives — which is the *last* thing we want, because there is only one thread. A non-blocking socket instead returns immediately with "would block, nothing here yet" if no data is ready. So the async I/O routine does this: it tries to read; if the socket says "would block," it does not spin — it registers fd 42 with the loop's **selector**, asking to be told when 42 becomes readable, and suspends the coroutine (yields to the loop).

The selector is a thin Python wrapper (`selectors.DefaultSelector`) over the OS readiness facility — `epoll` on Linux, `kqueue` on macOS/BSD. You hand `epoll` a set of file descriptors and the events you care about (readable, writable), and then you call `epoll_wait`, which **blocks the one thread until at least one of those fds is ready**, then hands you back the list of ready ones. This is the only place a healthy event loop blocks, and it blocks *productively*: it is not burning CPU spinning, and it is not parked on one connection — it is asleep on *all* of them at once, waking the instant any becomes ready. The cost of watching ten thousand fds with `epoll` is roughly constant per ready event, not proportional to the total count (that is `epoll`'s whole reason for existing, versus the old `select`/`poll` that scanned every fd every call). This is why the model scales to C10k: the OS-level readiness check is cheap and does not grow with idle-connection count.

So the full path of one wait is: coroutine tries to read → socket says "would block" → loop registers fd 42 with the selector and remembers "this coroutine is waiting on 42" → loop suspends the coroutine and runs others → eventually the ready queue empties → loop calls `epoll_wait`, sleeping the thread → data arrives on the network, the kernel marks fd 42 readable → `epoll_wait` returns "42 is ready" → loop looks up "who was waiting on 42," finds our coroutine, schedules it back onto the ready queue → next tick, the loop resumes the coroutine, which retries the read and this time gets the data. The selector is the matchmaker between "raw kernel readiness events" and "which coroutine to resume." The stack of layers involved, from your coroutine down to the OS sockets, is worth seeing as one picture:

![A layered stack diagram showing the async stack from your coroutine with async def and await on top, down through the Task that schedules it, the single threaded event loop driver, the selector using epoll or kqueue, and the operating system sockets holding thousands of file descriptors at the bottom](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-4.png)

The practical upshot for you as a user: **you almost never touch the selector.** Libraries like `aiohttp` and `httpx`, and the asyncio stream APIs, do the non-blocking-socket-and-register dance for you under their `await`s. Your job is only to `await` the right things and never block the thread. But knowing the selector is down there explains the model's strengths (cheap, scalable I/O readiness) and its single hard limit (one thread, so anything that does not go through the selector — any synchronous blocking call or CPU loop — stalls the whole machine).

It also explains a question beginners ask constantly: "if it is all one thread, how is anything happening at the same time?" The honest answer is that nothing in *your Python code* runs at the same time as anything else — there is exactly one thread executing your coroutines, one statement at a time. What runs in parallel is the *waiting*, which happens outside your process entirely: the network hardware, the remote servers, the disk controllers are all making progress on your behalf while your one thread is parked in `epoll_wait`. Asyncio gives you concurrency (many things in progress) without parallelism (many things executing simultaneously). For I/O that is exactly the right trade, because the thing you needed to overlap was never CPU work — it was the dead time spent waiting for someone else. The instant the bottleneck shifts to your own CPU, this trade stops paying, which is the whole reason the single-thread limit exists and the next section is about.

## A real high-concurrency client: many requests on one thread

Let us put it together with the example async was built for: fetching many URLs concurrently. We will use `aiohttp`, the most common async HTTP client. The shape generalizes to any async-native library — database drivers like `asyncpg`, Redis clients, message-queue consumers — because they all expose `await`able operations that go through the loop's selector.

```python
import asyncio
import time

import aiohttp


async def fetch_status(session: aiohttp.ClientSession, url: str) -> tuple[str, int]:
    # Each call to session.get() awaits the network round trip, suspending
    # this coroutine and freeing the loop to service other fetches.
    async with session.get(url) as resp:
        await resp.read()  # drain the body so the connection can be reused
        return url, resp.status


async def fetch_all(urls: list[str]) -> list[tuple[str, int]]:
    # One shared session pools connections across all fetches.
    async with aiohttp.ClientSession() as session:
        # TaskGroup gives us structured concurrency: if one fetch errors,
        # the rest are cancelled and cleaned up, no orphaned requests.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch_status(session, u)) for u in urls]
        # On clean exit, every task is done; collect results.
        return [t.result() for t in tasks]


def main() -> None:
    urls = ["https://example.com"] * 500  # 500 concurrent fetches
    start = time.perf_counter()
    results = asyncio.run(fetch_all(urls))
    elapsed = time.perf_counter() - start
    ok = sum(1 for _, status in results if status == 200)
    print(f"{ok}/{len(results)} ok in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
```

Five hundred fetches, all in flight on one thread, finishing in roughly the time of the slowest few round trips rather than the sum of all five hundred. On the reference machine against a fast endpoint, this typically lands in the low single-digit seconds for five hundred requests, where the synchronous `requests` loop would take five hundred times one round trip — close to a minute. The exact number depends on the server, the network, and how many connections the client and server allow at once, which is exactly the kind of real-world detail the [async in practice](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code) post handles with connection limits and a `Semaphore`. The point here is structural: one `asyncio.run`, one thread, one event loop, five hundred coroutines, and the waits all overlapped.

#### Worked example: sequential vs gather vs threads for 200 requests

Time to put real numbers on the three approaches for the same workload: fetch 200 endpoints that each respond in about 100 ms. Reference machine, CPython 3.12, results are representative of what you would measure with warmup and a median over several runs (run each a few times and take the median, because network latency is noisy — a single run can be skewed by one slow response or a GC pause).

| Approach | What it does | Wall clock (200 reqs, ~100 ms each) | Peak RSS | Notes |
| --- | --- | --- | --- | --- |
| Sync loop (`requests`) | one request at a time | about 20–22 s | ~40 MB | waits are summed: $200 \times 100\,\text{ms}$ |
| Threads (200 workers) | one thread per request | about 0.3–0.6 s | ~300–500 MB | waits overlap, but 200 OS stacks |
| `asyncio.gather` / TaskGroup | 200 coroutines, one thread | about 0.15–0.3 s | ~60–90 MB | waits overlap, ~KB per coroutine |

Read the table as the whole argument in numbers. The sync loop is the baseline disaster: it sums the waits, so 200 requests at 100 ms is twenty-plus seconds of mostly-idle CPU. Threads fix the wall clock by overlapping the waits — they are roughly as fast as async here — but they pay for it in memory, because two hundred threads is two hundred stacks, and the RSS balloons. Async matches or beats the thread wall clock and does it in a fraction of the memory, because two hundred coroutines is a rounding error in RSS. Now mentally scale the request count from 200 to 20,000: the thread approach falls over (20,000 stacks will not fit, and the scheduler thrashes), while the async approach grows linearly in cheap coroutine objects and keeps going. *That* is the regime where async is not just nicer but the only option that survives.

## Why async beats threads for massive I/O concurrency — the full argument

We now have every piece to state the central claim completely. Async beats threads specifically when you have *many* concurrent I/O waits — thousands or tens of thousands — for three compounding reasons, all of which we have built up:

**Memory.** A thread carries an OS stack reservation measured in megabytes; a coroutine carries a saved frame measured in kilobytes. At ten thousand connections that is the difference between tens of gigabytes you cannot fit and tens of megabytes you barely notice — a factor of roughly a thousand, as the cost model showed. Memory is the first wall threads hit, and it is a hard wall: when the stacks do not fit, you swap, and swapping makes everything slower, which is exactly the failure I opened with.

**Context-switch cost.** Switching between threads is a kernel operation — save registers, swap kernel stacks, often disturb the TLB and CPU caches — costing on the order of a microsecond and growing worse as the scheduler juggles more runnable threads. Switching between coroutines is the loop returning from one and calling another inside a single thread: tens of nanoseconds, no syscall, no scheduler, no cache flush. At low concurrency this difference is noise; at high concurrency, with switches happening constantly, it becomes a real fraction of your CPU.

**No GIL contention.** Threads that wait on I/O release the GIL, so their waits overlap — but every time a thread wakes to run Python between waits, it must re-acquire the GIL, and with thousands of threads waking around the same time, they contend on that one lock and the interpreter spends time arbitrating. One event loop thread never contends, because nothing else wants the GIL. You replaced "many threads fighting over one lock" with "one thread that owns it."

The honest counterpoint, which keeps you out of trouble: for a *small* number of I/O waits — a few dozen — threads are completely fine, often simpler, and you should not reach for async just to fetch five URLs. [Threads done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) is the right tool at that scale, and that post covers exactly where threads win and where they stop scaling. Async earns its added complexity — the colored-function split between `async` and sync code, the discipline of never blocking the loop — at *high* concurrency, where threads run out of memory and the scheduler thrashes. The crossover is somewhere in the low hundreds of concurrent waits, depending on your stack size and box. Below it, use threads; above it, use async. The deeper question of *why* the per-wait latency dominates these decisions — how a single network round trip compares to everything else your program could be doing — is exactly what the [latency-numbers guide](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) in this series quantifies: a network round trip is tens of millions of CPU-cycles' worth of waiting, which is precisely why overlapping those waits is the whole win.

#### Worked example: memory at 1k, 10k, and 50k idle connections

Make the memory wall concrete by holding *idle* connections — the WebSocket-server case, where most connections just sit open. On the reference machine (8-core x86-64 Linux, CPython 3.12, 16 GB), here is how the two models scale, using representative figures (the resident set grows roughly linearly in each model, so the slope is the story, not any single point).

| Connections held idle | Thread per connection (stack ~1 MB, reduced) | One loop, one coroutine each (~2 KB) |
| --- | --- | --- |
| 1,000 | ~1.1 GB RSS, scheduler fine | ~25 MB RSS |
| 10,000 | ~10 GB RSS, near the box limit | ~70 MB RSS |
| 50,000 | does not fit, swapping or OOM | ~250 MB RSS, still comfortable |

Even with a *reduced* 1 MB thread stack (eight times smaller than the default, and risky for deep call chains), the thread model crosses the machine's memory ceiling somewhere around ten thousand connections and simply cannot reach fifty thousand on a 16 GB box. The coroutine model is still relaxed at fifty thousand, sitting in a few hundred megabytes, because each connection costs a small Python object rather than a stack reservation. The slope is the whole point: thread RSS climbs by roughly a megabyte per connection, coroutine RSS by roughly a kilobyte or two, so the lines diverge by three orders of magnitude. There is no thread tuning that closes a thousandfold gap — at this scale you change the model, not the parameters. And note this is the *idle* case, which is the kindest one for threads (idle threads are parked, not contending); add real traffic and the scheduler-thrash and GIL-contention costs pile on top of the memory wall.

## The hard limit: it is still one thread

Everything good about asyncio flows from "one thread, cooperative scheduling." Everything *dangerous* about it flows from the exact same fact. Because there is one thread and nothing preempts a coroutine, **any code that does not yield control freezes the entire loop** — every connection, every task, all of it — until that code finishes. There are two ways to break the rule, and both are common.

The first is a **blocking call**: any synchronous operation that parks the thread instead of going through the loop. The classic offenders are `time.sleep` (blocks the thread; use `asyncio.sleep`), `requests.get` (synchronous network I/O; use an async client like `aiohttp`/`httpx`), a synchronous database driver, plain file reads on a slow disk, or any third-party library that does its own blocking I/O. When a coroutine calls one of these, it does not yield to the loop — it just stops the one thread dead until the call returns. While it is blocked, the loop cannot run any other coroutine, cannot service the selector, cannot do anything. Your ten thousand connections all hang behind one `requests.get`.

```python
import asyncio
import time


async def good() -> None:
    await asyncio.sleep(1)  # suspends THIS coroutine, loop runs others


async def bad() -> None:
    time.sleep(1)  # blocks the WHOLE thread for 1s, freezing every coroutine
```

The two lines look almost identical and behave like night and day. `await asyncio.sleep(1)` frees the loop for a second; `time.sleep(1)` murders it for a second. Spotting these — a sync call where an async one belongs — is the single most common async bug, and the [async in practice](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code) post is largely about detecting and fixing them, including the escape hatch `asyncio.to_thread`, which pushes a blocking call onto a real thread pool so it does not stall the loop.

The second way to freeze the loop is **CPU-bound work**: a tight numeric loop, a big `json.loads` of a fifty-megabyte payload, an image resize, a regex over a huge string, a `sorted` on a million items. None of these `await`, so none of them yield, so they hold the one thread for their entire duration. Async does *nothing* for CPU-bound work — there is no wait to overlap, just computation, and one thread can only compute one thing at a time (the GIL would serialize Python-level CPU work even across threads). If your bottleneck is the CPU, asyncio is the wrong lever entirely; you want [multiple processes](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) to use every core, or you want to compile the hot loop into native code. Putting CPU-heavy work directly in a coroutine is a category error: it does not just fail to help, it actively harms, because now your CPU loop is starving all your I/O coroutines of the thread.

This is the whole reason the decision tree matters. The choice of concurrency tool is not a matter of taste — it follows mechanically from the *shape* of the work. The next figure encodes it: many concurrent I/O ops with async-native libraries go to asyncio; CPU-bound work goes to processes; a library that only offers blocking calls (no async client exists) goes to threads, where you can let the call block one thread without stalling a loop.

![A decision tree for which concurrency tool to reach for branching from the question of what the work is into many concurrent io leading to asyncio on one thread, cpu bound leading to processes that use every core, and a blocking only library leading to threads that let the call block](/imgs/blogs/asyncio-from-the-ground-up-event-loops-and-coroutines-8.png)

#### Worked example: one blocking call freezes ten thousand connections

Make the failure concrete, because reading about it does not land the way measuring it does. Imagine a server holding ten thousand idle async connections, ticking along happily, the loop spinning through the selector in microseconds. Now one request handler calls a synchronous, badly written third-party function that does a 500-millisecond CPU-bound parse — no `await` anywhere inside it.

For that full 500 milliseconds, the one thread is inside that parse. The event loop does not run. The selector is not polled. The other 9,999 connections — including ones with data sitting ready in their socket buffers — get *zero* service. Their latency, which should be microseconds of loop overhead, becomes 500 milliseconds plus whatever was already queued. If parses like this arrive a few times a second, your p99 latency does not degrade gracefully; it falls off a cliff, because every connection is periodically stuck behind a half-second freeze. Measured on the reference machine, a server that holds steady at sub-millisecond p99 with pure-async handlers will jump to a p99 in the hundreds of milliseconds the moment a single synchronous 500 ms call enters a handler — not because the server got busier, but because one coroutine stopped cooperating. The fix is never "add more async"; it is to get the blocking work *off the loop thread*, with `asyncio.to_thread` for blocking I/O or a `ProcessPoolExecutor` (via `loop.run_in_executor`) for CPU work — both covered in the practice post. The diagnosis, though, comes entirely from this one rule: **find the code between two `await`s that runs too long, and that is your stall.**

## Case studies and real numbers

Concrete, sourced results to ground the model, with versions named so you can check them.

**The C10k problem and `epoll`.** The "C10k problem" — handling ten thousand simultaneous connections on commodity hardware — was articulated by Dan Kegel around 1999 and drove the adoption of readiness-based I/O (`epoll`, `kqueue`) over thread-per-connection. The entire async-server ecosystem (nginx, Node.js, and asyncio in Python) exists because the selector-plus-event-loop model made ten thousand connections cost kilobytes-per-connection instead of megabytes-per-connection. This is not a Python-specific trick; it is the standard way every high-concurrency server is built, and Python's asyncio is its idiomatic expression of the pattern. The memory math is the same everywhere: per-connection cost drops from a thread stack to a small object.

**uvloop, a faster event loop.** Asyncio's default pure-Python event loop is fast, but the third-party `uvloop` drop-in (built on libuv, the same C event-loop library that powers Node.js) replaces it and is commonly benchmarked at roughly 2–4× the throughput of the default loop for network-heavy workloads, approaching the performance of Go and Node for raw I/O. You opt in with a single line — `uvloop.install()` (or `asyncio.run(main(), loop_factory=uvloop.new_event_loop)` on newer versions). The fact that you can swap the loop and double throughput without changing a line of your coroutine code is itself a demonstration of the architecture: your coroutines do not know or care which loop drives them. The exact speedup depends heavily on the workload; treat 2–4× as a typical range, not a guarantee, and measure your own.

**Faster CPython and asyncio.** The CPython 3.11 and 3.12 releases ("Faster CPython") improved task creation and coroutine-stepping overhead, and 3.11 shipped `TaskGroup` and `except*` for structured concurrency — the modern recommended way to write concurrent async code. The practical takeaway: if you are on 3.11 or later, prefer `TaskGroup` over `gather` for new code, both because it is faster on these versions and because it eliminates the orphaned-task and swallowed-exception bugs by construction.

**The aiohttp crawler, real numbers.** The opening story is representative of a common, reproducible result: a synchronous `requests` loop over thousands of endpoints that each respond in ~100 ms runs in tens of minutes (the waits are summed); the equivalent `aiohttp` version with bounded concurrency finishes in single-digit seconds (the waits overlap), on one thread, in a couple hundred megabytes of RSS. The exact numbers depend on the endpoints and your concurrency limit, but the *structure* — minutes to seconds, gigabytes-of-threads to megabytes-of-coroutines — holds across every I/O-bound fan-out workload I have measured. When the work is waiting, overlap the waiting; that is the whole game.

## When to reach for asyncio (and when not to)

A decisive recommendation, because every tool is a cost and async's cost is real: it splits your code into "async" and "sync" worlds that do not freely mix, it demands the discipline of never blocking the loop, and it makes stack traces and debugging a bit harder. Pay that cost only when it earns its keep.

**Reach for asyncio when** the work is I/O-bound *and* high-concurrency *and* you have async-native libraries for it. The canonical cases: a server or proxy holding thousands of concurrent connections (web, WebSocket, gRPC); a client fanning out thousands of concurrent requests (a crawler, a health-checker, a batch API caller); any workload where you are waiting on many networks/databases/services at once and the per-connection thread cost would crush you. If you have an async driver (`aiohttp`, `httpx`, `asyncpg`, async Redis/Kafka clients) and thousands of concurrent waits, async is the right and often the only scalable lever.

**Do not reach for asyncio when** any of these hold. If the work is **CPU-bound** — number crunching, parsing huge blobs, image processing — async does nothing for you (there is no wait to overlap) and putting it in a coroutine freezes the loop; use [multiple processes](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling) to spread across cores, or compile the hot loop. If the concurrency is **low** — a handful of I/O waits — threads are simpler and just as fast; do not adopt a whole new programming model to fetch five URLs. If your **library only offers blocking calls** and has no async equivalent — a legacy synchronous database driver, a vendor SDK that does its own I/O — then async cannot help directly, because you would just block the loop; use threads (let the blocking call park one thread of a pool) or, if it is the dominant cost, `asyncio.to_thread` to bridge a little blocking work into an otherwise-async program. And if your bottleneck is one slow downstream you cannot parallelize (a single sequential dependency), no concurrency model helps — fix the downstream.

A blunt rule of thumb: if you cannot name the *waits* you are trying to overlap, you do not need asyncio. Async overlaps waits. No waits, no win.

## Key takeaways

- **Async is cooperative single-threaded I/O multiplexing.** One thread runs an event loop that watches many I/O sources at once through the OS selector (`epoll`/`kqueue`), and each coroutine voluntarily yields at every `await`. That one sentence is the whole model.
- **`await` is a suspension point that yields to the loop.** It saves your coroutine's frame, hands the thread back to the loop, and resumes you at the same line with the same locals once your wait is ready. Between two `await`s, your code has the thread to itself.
- **A coroutine runs until it `await`s — that is the invariant.** Nothing preempts it. This makes shared-state reasoning easy (you only yield where you write `await`) and creates the one hard failure mode (code that never yields freezes everything).
- **`create_task` schedules concurrency; `await` collects it.** A bare coroutine is inert; sequential awaits serialize; `gather` and `TaskGroup` overlap N waits so the wall clock is the *max*, not the *sum*.
- **Prefer `TaskGroup` (3.11+) over `gather` for new code.** Structured concurrency cancels siblings on failure, never orphans a task, and never swallows an exception — it bounds task lifetime to a block.
- **Async beats threads at high I/O concurrency on three counts:** kilobyte coroutines vs megabyte thread stacks (≈1000× less memory at 10k connections), nanosecond cooperative switches vs microsecond kernel switches, and zero GIL contention vs many threads fighting one lock.
- **It is still one thread.** A blocking call (`time.sleep`, `requests.get`, a sync driver) or any CPU-bound work freezes the entire loop. Get blocking work off the loop (`to_thread`), and send CPU work to processes.
- **The tool follows the shape of the work.** Many concurrent I/O waits with async libraries → asyncio. CPU-bound → processes. Blocking-only library → threads. Low concurrency → threads (simpler). If you cannot name the waits, you do not need async.

## Further reading

- The CPython `asyncio` documentation — the event loop, coroutines and tasks, `gather`, and `TaskGroup`/structured concurrency (`docs.python.org/3/library/asyncio.html`). The "Developing with asyncio" and "High-level API index" pages are the practical reference.
- PEP 492 (coroutines with `async`/`await`) and PEP 525/530 (async generators and comprehensions) — the language-level definition of what `async def` and `await` mean.
- The `selectors` module docs and your OS's `epoll`/`kqueue` manual pages — the readiness layer under the loop.
- Dan Kegel's "The C10K problem" — the historical statement of the high-concurrency problem that the selector-plus-event-loop model solves.
- The `uvloop` project README and benchmarks — a faster drop-in event loop and a concrete demonstration that your coroutines are loop-agnostic.
- "High Performance Python" by Gorelick and Ozsvald — the asyncio and concurrency chapters, in the broader context of the leverage ladder.
- Within this series: the [GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) for why one thread sidesteps lock contention; [threads done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) for the lower-concurrency alternative and where it stops scaling; [async in practice](/blog/software-development/python-performance/async-in-practice-patterns-pitfalls-and-blocking-code) for the real-world patterns, bounded concurrency, and the blocking-call fixes; and the [latency-numbers guide](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop) for why a network round trip dominates everything and is the wait worth overlapping.

---
title: "Processes, Threads, and How the OS Scheduler Runs Them"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "What actually runs your concurrent code: processes versus threads, the M:N model behind cheap goroutines, and the scheduler that slices a core microsecond by microsecond."
tags:
  [
    "concurrency",
    "parallelism",
    "threads",
    "processes",
    "scheduler",
    "context-switch",
    "operating-systems",
    "goroutines",
    "systems-programming",
    "multithreading",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-1.png"
---

A team I worked with shipped a clean little HTTP server. Every incoming request got its own thread: accept the socket, spawn a thread, do the work, write the response, let the thread die. It was easy to read, easy to debug, and it handled the load tests fine — a few hundred concurrent connections, no problem. Then it went to production behind a real load balancer, traffic climbed past a few thousand simultaneous connections, and the box fell over. Not with a clean error. The process got slower and slower, latency went vertical, memory climbed until the kernel's out-of-memory killer reaped it, and `top` showed the CPU busy doing… something, but throughput had collapsed.

The whole failure is in one number the team never thought about: each thread reserved roughly a megabyte of stack. Ten thousand connections meant ten thousand threads meant about ten gigabytes of address space reserved for stacks alone, and a scheduler trying to time-slice ten thousand runnable things across sixteen cores — so the machine spent most of its cycles *switching between threads* instead of *running* them. The code was correct. The model was wrong. To understand why, and to know what to reach for instead, you have to understand the things that actually run concurrent code: the process, the thread, and the operating-system scheduler that decides who gets a core and for how long.

That is this whole post. We are going to open up the box under every `Thread`, every `go func()`, every `tokio::spawn`, every `std::thread`. We will see what a process is and what a thread is, what they share and what they keep private, why a goroutine costs a few kilobytes while an OS thread costs a megabyte, how the scheduler slices a single core between many threads, what a context switch physically does to the CPU and what it costs in microseconds, and why blocking a kernel thread is the expensive thing that the entire async / lightweight-thread world exists to avoid. This is the mechanistic foundation for the rest of the series: once you can see the thread and the scheduler clearly, races, locks, and async stop being magic. The figure below is the picture to hold in your head — two processes are walled off from each other, but threads inside one process share a single heap.

![Side by side comparison of two isolated processes each with its own address space versus one process running three threads that share the heap and code but keep private stacks](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-1.png)

If you have not read the series opener, [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) sets up the spine we keep returning to: shared mutable state plus nondeterministic scheduling is the root hazard. This post is where that "nondeterministic scheduling" stops being a phrase and becomes a concrete mechanism you can reason about — and where "shared mutable state" gets its physical home, the shared heap of a multi-threaded process.

## The running example: a shared counter that lives in shared memory

Throughout the series we keep one concrete example as the spine: a shared counter (which grows into a bank account, then a producer–consumer pipeline, then a connection server). It is worth grounding the thread-versus-process distinction in it right away, because the distinction is exactly *where that counter lives*.

If two **threads** in one process both increment a counter, they are touching the same memory cell. There is one `counter` variable on the shared heap, one address, and both threads can load it, add one, and store it back. That is what makes the classic lost-update race possible — and it is *only* possible because the memory is shared. We will not dwell on the race here; [the anatomy of a race condition](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) is its own subject. The point for now is structural: threads in one process can race precisely *because they share an address space*.

If instead two **processes** each have a counter, they have two different counters at two different addresses in two different address spaces. Incrementing one does nothing to the other. To share a number between processes you must go out of your way — a pipe, a socket, a file, or an explicitly mapped shared-memory segment. The isolation is the default; sharing is the exception you opt into.

That single difference — *sharing by default versus isolation by default* — is the root of almost everything else: why threads are cheaper to create, why threads can corrupt each other, why a crash in one thread can take down the whole process, and why a crash in one process leaves its siblings untouched. Hold the counter in mind. We will keep asking the same three questions the series always asks — what is shared, what orders the accesses, what does it cost — and the answers start here.

## Process versus thread: the address-space picture

A **process** is a program in execution *together with everything the operating system gives it to run*: a private virtual address space, a set of open file descriptors, the program's code and data, a heap, at least one thread of execution, and a pile of kernel bookkeeping (process id, owner, scheduling priority, signal handlers, working directory). The defining feature is the **private virtual address space**. When process A dereferences the pointer `0x40000000`, the CPU's memory-management unit (MMU) translates that virtual address through A's page tables to some physical frame. When process B dereferences the *same* virtual address `0x40000000`, the MMU uses *B's* page tables and lands on a completely different physical frame. The two processes can both use the address `0x40000000` and never collide, because each has its own translation. That is the wall between processes, and it is enforced by hardware on every single memory access.

A **thread** is a single flow of execution *within* a process. It is the thing the scheduler actually runs. A thread has a program counter (which instruction it is on), a set of CPU registers, and a stack (for its function calls and locals). What a thread does *not* have is its own address space. All threads in a process share one address space — one set of page tables, one heap, one copy of the code, one file-descriptor table. When you create a second thread, you are not duplicating memory; you are adding another program counter and another stack to the *same* memory.

So the crisp distinction:

- A process is a *container* for resources and isolation — its own memory, its own file descriptors, its own everything.
- A thread is a *unit of execution* — a program counter, registers, and a stack — that lives inside a process and shares that process's memory with its sibling threads.

A process always has at least one thread (the "main" thread that starts at `main`). A process with three threads has one address space and three independent flows of execution running through it. Here is the inventory of what is shared and what is private between threads in the same process.

![Matrix showing heap and globals and file descriptors and code are shared across all threads while stack and registers and thread-local storage are private to each thread](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-2.png)

Read the table carefully, because every concurrency bug and every concurrency cost lives in it.

**Shared across all threads in the process:**

- **The heap and global/static variables.** One copy. If thread A `malloc`s an object and hands the pointer to thread B, B can read and write the same bytes. This is the power and the peril: it is how threads collaborate, and it is the only thing that can race.
- **The file-descriptor table.** Open a file in one thread, and every thread can `read`/`write` that same descriptor. Close it in one thread and it is closed for all.
- **The code (text) segment.** All threads execute the same compiled instructions, mapped read-only.
- **Process-wide state:** the current working directory, signal-handler dispositions, the process id, resource limits.

**Private to each thread:**

- **The stack.** Each thread gets its own stack for its call frames and local variables. A local variable in one thread is invisible to another (unless the thread leaks a pointer to it — a classic use-after-scope bug). The default stack size is large precisely so deep recursion does not overflow: roughly 1 MB on Linux (often 8 MB by `ulimit` default for the main thread), and you can tune it with `pthread_attr_setstacksize`.
- **The registers and program counter.** Each thread has its own. When the scheduler switches threads, *this* is the bulk of what gets saved and restored.
- **Thread-local storage (TLS).** A per-thread slot for "global-looking" variables that are actually private — `__thread` in C/GCC, `thread_local` in C++/Rust, `ThreadLocal<T>` in Java, and `errno` itself in modern libc. Each thread sees its own copy.

The consequence is sharp. Because the heap is shared, two threads writing the same object without synchronization is undefined behavior — a **data race** — and that is the entire reason locks, atomics, and the memory model exist. Because stacks are private, two threads recursing independently never interfere. And because the file-descriptor table is shared, one thread closing a socket out from under another is a bug you will eventually hit.

There is one more consequence worth stating plainly: a process is also the unit of *failure isolation*. If one thread dereferences a null pointer or corrupts the heap, the whole process dies — all its threads with it, because they share the address space that just got corrupted. But if one *process* crashes, its siblings are untouched, because their memory is walled off. This is why a browser puts each tab in its own process, why a database may fork a worker per connection, and why "fault isolation" is a reason to choose processes over threads even when the performance argument points the other way.

### How a process is created: fork, copy-on-write, and why it is pricier than a thread

It is worth pinning down *why* creating a process costs more than creating a thread, because the mechanism explains a lot of design decisions you will meet. On Unix, a new process is born by `fork()`, which makes the calling process's child a near-exact duplicate — same code, same data, same open file descriptors, the child even resumes from the same line. The naive reading is that `fork` copies the entire address space, which for a multi-gigabyte process would be ruinously slow. It does not. Modern kernels use **copy-on-write (COW)**: `fork` copies only the page *tables*, marks every shared page read-only in both parent and child, and lets them share the physical pages. Only when one side *writes* to a page does the kernel trap the write, copy that single page, and give the writer its own private copy. So `fork` is cheap-ish up front (duplicate the page tables, flip permission bits) and pays the real copy cost lazily, page by page, as the two processes diverge.

Even with COW, creating a *process* is meaningfully more expensive than creating a *thread*, for a precise reason: a thread shares the parent's page tables outright (nothing to duplicate), while a process needs its *own* page tables and its own kernel bookkeeping (a fresh address space, file-descriptor table copy, the lot). Duplicating page tables for a large address space is real work, and the first writes after `fork` each take a page-fault-and-copy hit. This is why a thread is the cheaper unit when you want to *share* and a process is the unit when you want to *isolate* — and why server frameworks that `fork` a worker per request (the old CGI model) were displaced by thread pools and event loops: per-request `fork` is just too much page-table churn at high request rates. (The Python story here — why `multiprocessing` pays a fork-or-spawn-plus-pickle cost to get around the GIL — is its own deep dive; see [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling).)

#### Worked example: the megabyte that sank the server

Return to the opening server. It spawned one OS thread per connection. On Linux, a default thread stack is about 1 MB of *reserved* virtual address space (it is mapped lazily, so not all of it is resident, but the address space and a guard page are committed, and the scheduler must track every thread).

Do the arithmetic. At 1,000 concurrent connections: 1,000 threads × ~1 MB ≈ 1 GB of stack address space. Uncomfortable but survivable. At 10,000 connections: 10,000 × ~1 MB ≈ 10 GB of stack address space — and that is *before* the heap, the request buffers, the TLS buffers. On a 16 GB box you are now thrashing. And the resident set (the pages actually touched) climbs steadily as each thread runs real code on its stack. The process did not crash from a bug. It crashed because the *cost model of an OS thread* — about a megabyte each, plus a scheduler entry — does not survive ten thousand of them. We will return to the fix (a bounded pool, or lightweight threads) at the end. First, why does an OS thread cost a megabyte and a goroutine cost a few kilobytes? That is the next question.

## Kernel threads, user threads, and the M:N model

So far "thread" has meant one thing. It is actually two things layered, and conflating them is the source of most confusion about why goroutines and virtual threads are cheap.

A **kernel thread** is a thread the operating-system kernel knows about and schedules. It has a kernel-side data structure (on Linux, a `task_struct`), it can be placed on a CPU run queue, and it is what gets a time slice. Creating one is a system call into the kernel (`clone` on Linux). The kernel scheduler manages it. When people say "OS thread," they mean a kernel thread.

A **user thread** (or *green thread*, or *fiber*, or *coroutine* depending on the runtime) is a thread that a user-space runtime — a language's runtime library, not the kernel — manages entirely on its own. The kernel does not know it exists. The runtime keeps its own data structure for it, its own (often tiny, growable) stack, and its own scheduler that decides when to run it. Switching between user threads can happen *without entering the kernel at all* — it is a function call that saves a few registers and swaps a stack pointer.

How user threads map onto kernel threads is the whole story, and there are three classic models.

![Tree of threading models showing one to one kernel mapping and N to one green threads and M to N hybrid each with example runtimes and their stack costs](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-3.png)

**1:1 (kernel-level threading).** Every user thread is backed by exactly one kernel thread. When you call `pthread_create`, `std::thread::spawn` in Rust, `new Thread()` in classic Java, or `std::thread` in C++, you get a kernel thread one-to-one. This is what nearly every mainstream language defaults to, because it is simple and gives you real parallelism for free: the kernel can put your N threads on N cores, and if one thread blocks in a system call, the others keep running because the kernel can schedule around the blocked one. The cost is the cost: each thread is a kernel object with a big stack, creation is a syscall, and switching between them is a kernel-level context switch (the expensive kind).

**N:1 (pure user-level / classic green threads).** Many user threads multiplexed onto *one* kernel thread. The runtime's scheduler picks which user thread runs; the kernel only ever sees the one thread. Switching is dirt cheap (no kernel involved), and you can have millions of them. But there are two fatal flaws. First, **no parallelism** — one kernel thread runs on one core, so all your user threads share a single core no matter how many cores the machine has. Second, and worse, **one blocking system call blocks everyone**: if a user thread calls `read()` and the kernel thread blocks waiting for the disk, the runtime cannot switch to another user thread, because the only kernel thread it has is parked in the kernel. The whole program stalls. This is why early green-thread systems (old Java "green threads" before 1.3, early Ruby) were abandoned for real workloads.

**M:N (hybrid).** Many user threads multiplexed over a small pool of kernel threads — typically about one kernel thread per CPU core. This is the model behind **Go goroutines** and **Java's Project Loom virtual threads** (and Erlang's processes, and most modern async runtimes). It is the best of both: you get parallelism (M:N spreads user threads across the N kernel threads, hence across cores), and user-thread switches are cheap (most happen in user space). The hard part — and the reason this took decades to get right — is the blocking problem. The runtime must ensure that when a user thread does something blocking, it does not strand a whole kernel thread. The runtimes that win do this by intercepting blocking operations and turning them into cooperative yields, plus, in Go's case, spinning up an extra kernel thread when a goroutine genuinely blocks in a syscall so the others keep running.

### The mechanism: why M:N makes a goroutine cheap

The cost of a goroutine or virtual thread is low because of three deliberate design choices, and it is worth making each one mechanical.

**Tiny, growable stacks.** An OS thread reserves a fixed, large stack (~1 MB) up front because the kernel cannot easily move a stack once a thread is running — pointers into it would break. A goroutine starts with a small stack (historically 8 KB, now about 2 KB) and *grows it on demand*: the Go compiler inserts a tiny check at the top of each function ("is there enough stack left?"), and if not, the runtime allocates a bigger stack, copies the frames over, and fixes up the pointers. Because the runtime owns the goroutine entirely, it *can* move the stack. So a million goroutines might cost a couple of gigabytes total, versus a terabyte for a million OS threads. The math is the whole reason `go func()` scales and `pthread_create` does not.

**User-space scheduling.** Switching from one goroutine to another that are both on the same kernel thread does not enter the kernel. The runtime saves a handful of registers and the stack pointer and jumps. That is on the order of ~100–200 ns, versus ~1–5 µs for a kernel context switch. An order of magnitude cheaper, on every switch.

**Cooperative-plus-preemptive yields at known points.** Go's runtime turns blocking operations (channel sends, network I/O, mutex waits, `time.Sleep`) into cooperative yield points: instead of blocking the kernel thread, the goroutine parks itself in the runtime's scheduler and the kernel thread picks up another runnable goroutine. Network I/O is the killer feature — Go's runtime registers the file descriptor with the OS's I/O readiness mechanism (`epoll` on Linux, `kqueue` on BSD/macOS) and parks the goroutine; when the descriptor is ready, the goroutine becomes runnable again. To the programmer it looks like a plain blocking `conn.Read()`; under the hood it is non-blocking I/O plus a scheduler. (Go also added asynchronous *preemption* in 1.14, so a goroutine in a tight loop with no yield points can still be preempted by a signal — fixing a real fairness bug.)

That is the engine. A goroutine looks like a thread, blocks like a thread, but is scheduled by the language runtime over a small pool of kernel threads, with a stack that starts tiny and grows. Java's virtual threads (Loom, stable in Java 21) do the same thing: a virtual thread runs on a *carrier* (platform) thread; when it blocks on I/O or a lock, it *unmounts* from the carrier and the carrier runs another virtual thread.

| Model | Parallelism | Switch cost | One blocking call | Examples |
|---|---|---|---|---|
| 1:1 kernel | Yes, across cores | ~1–5 µs (kernel) | Only that thread blocks | pthreads, `std::thread`, classic Java/C# threads |
| N:1 green | No, one core only | ~100 ns (user) | Blocks **all** user threads | Old Java green threads, early Ruby fibers without an I/O loop |
| M:N hybrid | Yes, across cores | ~100–200 ns (mostly user) | Runtime yields; others keep running | Go goroutines, Java Loom virtual threads, Erlang processes |

The table makes the trade visible. The 1:1 model is simple and parallel but expensive per thread. The N:1 model is cheap per thread but neither parallel nor robust to blocking. The M:N model takes work to build but gives you cheap *and* parallel *and* block-tolerant — which is why the modern answer to "I have ten thousand connections" is M:N lightweight threads (or async, which we will forward-link), not ten thousand OS threads.

## The scheduler: run queues, quanta, preemption, and fairness

You have N threads in the runnable state and a machine with C cores. Something has to decide which thread runs on which core, and for how long. That something is the **OS scheduler** (for kernel threads) — and, in the M:N world, a second runtime scheduler (for user threads on top). Let us start with the kernel scheduler, because everything else is layered on it.

The core fact, the one that surprises people who think "I have 50 threads so 50 things run at once": **a single core runs exactly one thread at a time.** A 16-core machine runs at most 16 threads *simultaneously*. If you have 50 runnable threads on 16 cores, 16 are running and 34 are waiting their turn. Concurrency on a single core is an *illusion of simultaneity* created by switching fast enough that everything appears to make progress. The figure shows it at its simplest: two threads, one core, the scheduler slicing time between them.

![Timeline of one core running thread one for a quantum then a context switch then thread two for a quantum then switching back showing time-slicing](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-4.png)

Here is the machinery.

**Run queues.** The scheduler keeps a set of runnable threads — threads that *could* run if given a core. On Linux, each core has its own run queue (to avoid a global lock that every core would contend on), and a load-balancer migrates threads between queues to keep cores busy. A thread in a run queue is in the **runnable** state; a thread actually executing is **running**; a thread waiting for I/O or a lock is **blocked** and is *not* in any run queue (it is on a wait queue instead). Only runnable threads compete for a core.

**Quantum (time slice).** When the scheduler dispatches a thread, it does not let it run forever. It gives the thread a **quantum** — a bounded slice of time — after which a hardware timer interrupt fires, control returns to the kernel, and the scheduler gets to reconsider who should run. Typical quanta are on the order of a few milliseconds (Linux's CFS targets a "scheduling latency" — the period in which every runnable thread should get a turn — of roughly 6–24 ms, divided among the runnable threads). The quantum is the leash that keeps one thread from monopolizing a core.

**Preemptive versus cooperative scheduling.** This is the dividing line.

- **Preemptive** scheduling: the scheduler can *forcibly* take the core away from a thread at the end of its quantum (or when a higher-priority thread becomes runnable), whether or not the thread cooperates. The mechanism is the timer interrupt: the hardware fires it, the CPU jumps into the kernel, and the kernel can context-switch to another thread. Every mainstream OS kernel scheduler is preemptive. This is *good* — a buggy thread in an infinite loop cannot freeze the machine; it just gets preempted like everything else.
- **Cooperative** scheduling: a thread runs until it *voluntarily* yields — calls `yield()`, blocks on I/O, or hits an `await`. The scheduler cannot take the core back on its own. This is how classic coroutines, early cooperative multitasking (Windows 3.x, classic Mac OS), and most single-threaded async event loops work. The advantage is that switches only happen at known points, so you never get preempted mid-update — fewer surprises, simpler reasoning. The catastrophic disadvantage is that **one thread that never yields freezes everyone**. A CPU-bound function with no `await` inside an async event loop stalls the entire loop; this is the "don't block the event loop" rule, and it is why async runtimes warn so loudly about it. (We pick this thread up in the async track; see [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) for the Python event-loop story.)

**Priorities.** Not all threads are equal. A scheduler uses priorities so that, say, an audio thread that must never glitch beats a background indexer. Linux has two regimes: real-time scheduling classes (`SCHED_FIFO`, `SCHED_RR`) for hard-priority threads that run ahead of everything normal, and the normal class (`SCHED_OTHER`, implemented by CFS / now EEVDF) for everyone else, where a "nice" value biases the share of CPU a thread gets. Priorities are powerful and dangerous: get them wrong and you get **priority inversion**, where a high-priority thread waits on a lock held by a low-priority thread that never gets scheduled — the bug that famously hit the Mars Pathfinder rover (covered in the case studies).

**Fairness (CFS-style).** The Linux Completely Fair Scheduler (the default from 2007 until EEVDF replaced it in 2023) frames scheduling as *fairness*: every runnable thread should receive a proportional share of CPU time. CFS tracks each thread's **virtual runtime** (`vruntime`) — roughly, how much CPU time it has consumed, weighted by its nice value — and always runs the thread with the *smallest* vruntime, the one that has had the least so far. It stores runnable threads in a red-black tree keyed by vruntime, so "pick the most-deserving thread" is an `O(log n)` lookup of the leftmost node. The effect is that all threads converge toward equal CPU shares, with priorities skewing the weights. EEVDF (Earliest Eligible Virtual Deadline First, the new default) refines this with explicit deadlines so latency-sensitive threads get served promptly, but the fairness intuition is the same: track who is owed time, serve them in order.

### Two schedulers, stacked: the kernel and the runtime

In an M:N world there are *two* schedulers at work, and it pays to keep them distinct. The **kernel scheduler** decides which kernel thread (carrier) runs on which core — that is everything above. On top of it, the **runtime scheduler** (Go's, Loom's, tokio's) decides which user thread (goroutine, virtual thread, task) runs on which carrier. The runtime scheduler is *also* a fairness-and-run-queue machine, but it lives in user space and switches without entering the kernel. Go's runtime, for instance, gives each carrier thread (called a `P`, a logical processor) its own local run queue of goroutines, with a global queue as overflow, and uses **work-stealing**: when a `P` empties its local queue, it steals half the goroutines from a busier `P` rather than sitting idle. This keeps all carriers busy without a single global lock that every core would contend on — the same per-core-queue-plus-balancing idea the Linux kernel uses, one layer up. The lesson: when you write `go func()` or `Thread.ofVirtual().start()`, your task is queued in a user-space scheduler, which schedules it onto a carrier kernel thread, which the kernel scheduler schedules onto a core. Three layers, two schedulers, and your code never sees any of it — until you profile a latency spike and have to reason about which scheduler delayed you.

#### Worked example: how long until my thread runs?

Say a core has 8 runnable threads of equal priority, and the scheduler targets a 24 ms scheduling period (everyone should get a turn within 24 ms). Then each thread's quantum is about 24 ms / 8 = 3 ms. In the worst case, a thread that just exhausted its quantum waits for the other 7 to each run 3 ms before it runs again: about 7 × 3 ms = 21 ms of latency. Add a runnable thread (now 9) and quanta shrink to ~2.7 ms but the round-trip stays near the 24 ms target — that is what "bounded scheduling latency" buys you. Now imagine 10,000 runnable threads (the failing server). The scheduler clamps the minimum quantum so it does not switch every microsecond, so the period stretches *far* past 24 ms, every thread's turn is rare, and meanwhile each turn is preceded by a context switch. This is the "scheduler thrash" half of why the thread-per-request server died: not just memory, but a scheduler drowning in runnable threads, spending a growing fraction of every core on *switching* rather than *working*.

## The context switch, step by step

We keep saying "context switch costs microseconds." Now let us earn that number by walking through what physically happens when the CPU stops running thread A and starts running thread B. A **context switch** is the act of saving enough of thread A's state that it can be resumed exactly where it left off, and restoring thread B's state so it continues exactly where *it* left off. The "context" is that state. Here are the steps.

![Vertical stack of the context switch steps from saving registers to swapping the stack pointer to switching page tables to flushing the TLB to cold caches to restoring the next thread](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-5.png)

1. **Enter the kernel.** A context switch is a kernel operation. It is triggered by an interrupt (the timer firing at the end of a quantum), a system call that blocks (thread A calls `read()` and must wait), or a thread voluntarily yielding. The CPU traps into kernel mode, saving the bare minimum (instruction pointer, flags) to get there.
2. **Save thread A's registers.** The kernel saves A's general-purpose registers, its program counter, its stack pointer, and its CPU flags into A's kernel-side thread structure. On x86-64 that is the 16 general registers plus the instruction pointer and flags; if A used floating-point or vector (SSE/AVX) registers, those get saved too — and the vector state is large (the AVX-512 register file is over 2 KB), which is why kernels lazily save FPU/vector state only when a thread has actually used it.
3. **Switch the stack pointer.** A and B have separate stacks. The kernel loads B's saved stack pointer so that from now on, pushes and pops land in B's stack.
4. **Switch the address space — only if B is in a different process.** This is the expensive fork in the road. If A and B are two threads of the *same* process, they share page tables, so there is nothing to switch — the address space is identical. But if B belongs to a *different* process, the kernel must load B's page-table base (on x86-64, write B's `CR3` register). This is the step that makes a *process* switch pricier than a *thread* switch within one process.
5. **The TLB cost.** The **translation lookaside buffer (TLB)** is a small, fast cache inside the CPU that remembers recent virtual-to-physical address translations, so the MMU does not walk the multi-level page table on every memory access. When you switch to a different process's address space, those cached translations are stale — they belong to the old process. Classically the CPU flushes the TLB on a page-table switch, and then B starts cold: its first memory accesses miss in the TLB and pay the full page-walk cost (dozens to hundreds of cycles each) until the TLB warms back up. Modern CPUs soften this with *tagged* TLBs (Process-Context Identifiers, PCID, on x86), which tag each entry with a process id so entries from different processes can coexist and you avoid a full flush — but it is still extra pressure and the new process still starts with cold-for-it entries.
6. **Restore thread B's registers and resume.** The kernel loads B's saved registers, program counter, and flags, returns from kernel mode, and the CPU resumes executing B exactly where it stopped. B has no idea it was ever paused.

### Direct cost versus indirect cost

The **direct cost** is steps 1–6: the cycles spent saving and restoring registers, switching the stack and (maybe) page tables, and the mode transitions. On modern hardware this is roughly **1–5 microseconds** for a kernel context switch — a number you can and should measure on your own box (we do, below). A switch between two threads of the *same* process is at the cheaper end (no page-table swap, no TLB pressure); a switch between two *processes* is at the pricier end (page-table swap, TLB and address-space cache effects).

The **indirect cost** is sneakier and often larger: **cache pollution**. While thread A was running, the CPU's L1 and L2 caches filled up with A's working set — its hot data and instructions. When B takes over, B's accesses miss in those caches (they hold A's data, not B's), so B runs slowly until it re-warms the cache, evicting A's data in the process. Then when A comes back, *its* caches are cold too. The pure register-shuffling might be 1–5 µs, but the *recovery* — the extra cache misses both threads pay after the switch — can be many microseconds more, depending on working-set size. This is why a workload that context-switches constantly (like ten thousand threads time-slicing) can spend a shocking fraction of its cycles not on switching itself, but on the cache misses that *follow* every switch. The deeper hierarchy story — why a cache miss to DRAM costs ~100 ns and an L1 hit ~1 ns — lives in the HPC series' [memory hierarchy post](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm); here the point is just that context switches *invalidate* that hierarchy's hard-won warmth.

| Switch type | What gets switched | Direct cost | Indirect cost |
|---|---|---|---|
| Same-process thread switch | Registers, stack pointer | ~1–2 µs | Cache pollution (shared page tables, TLB mostly survives) |
| Cross-process switch | Above + page tables, TLB | ~2–5 µs | Cache **and** TLB pollution; cold restart for the new process |
| User-thread switch (goroutine) | A few registers, stack pointer, no kernel entry | ~0.1–0.2 µs | Minimal; same address space, often same core, caches mostly warm |

The third row is the punchline of the whole M:N argument: switching between two goroutines on the same carrier thread skips the kernel entry, the page-table swap, and the TLB flush entirely. It is just a function that saves a handful of registers and swaps the stack pointer in user space. That is why a Go program can switch between goroutines tens of millions of times a second, while a program switching kernel threads is limited to a few hundred thousand switches a second per core.

#### Worked example: the cost of a busy switch

Suppose your service handles 50,000 requests per second, and each request involves a blocking database call that puts the handling thread to sleep and wakes it later — two context switches per request (sleep, then wake), conservatively. That is 100,000 context switches per second. At a direct cost of 3 µs each, that is 300,000 µs = 0.3 seconds of CPU per second *just shuffling registers* — about 30% of one core gone to switching overhead, before counting the cache-pollution tax. Now spread those 50,000 requests over goroutines instead: the same 100,000 switches happen in user space at ~0.15 µs each, totaling 15,000 µs = 0.015 seconds of CPU per second — about 1.5% of a core. Same logical work, a 20× difference in switching overhead, entirely because the switch never entered the kernel. This is not a micro-optimization; at scale it is the difference between needing 30 boxes and needing 2.

## Thread lifecycle: the states a thread moves through

A thread is not always running. Most of the time, most threads are *not* running — they are waiting for a core, or waiting for something to happen. Modeling these states precisely is how you reason about where time goes and why "I have 100 threads" does not mean "100 things are happening." The states form a small machine.

![Thread lifecycle state machine drawn as an acyclic flow from new to runnable to running then either blocked which re-queues to runnable or terminated](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-6.png)

- **New.** The thread object exists but has not been started. No stack is running yet; the scheduler is not considering it. In Java this is a `Thread` you have constructed but not `start()`ed.
- **Runnable.** The thread is ready to run and sitting in a run queue, waiting for the scheduler to give it a core. It is not consuming CPU; it is *eligible* for CPU. With more runnable threads than cores, this is where threads queue up. (Some systems split this into "ready" and "running"; Java lumps both into `RUNNABLE`, which is why a Java thread that is actually executing and one that is merely waiting for a CPU look the same in a thread dump.)
- **Running.** The thread is executing on a core *right now*. Only as many threads can be running as there are cores. This is the only state that consumes CPU.
- **Blocked / waiting.** The thread cannot make progress until some event occurs — data arrives on a socket, a lock is released, a timer fires, a condition variable is signaled. A blocked thread is removed from the run queue and parked on a wait queue tied to the event. Crucially, **a blocked thread consumes no CPU** — the scheduler will not run it — but it *does* still consume memory (its whole stack stays reserved) and a scheduler slot. This is the central tension of OS threads: blocking is CPU-free but not memory-free.
- **Terminated.** The thread has finished (returned from its top function or was cancelled). Its stack is reclaimed; it will never run again.

The transitions are the interesting part:

- **new → runnable:** you start the thread.
- **runnable → running:** the scheduler dispatches it to a core.
- **running → runnable:** the thread is *preempted* — its quantum expired or a higher-priority thread arrived. It did nothing wrong; it just got its turn taken. (The figure above draws this as the thread re-entering the runnable pool rather than as a back-edge, to keep the state diagram acyclic, but in practice running and runnable trade places constantly.)
- **running → blocked:** the thread does something that has to wait — a blocking `read()`, a `lock()` on a held mutex, a `wait()` on a condition. The scheduler immediately switches the core to someone else.
- **blocked → runnable:** the event the thread was waiting for happened (data arrived, lock freed, condition signaled). The thread is *woken* — moved from the wait queue back to a run queue — but it does **not** start running immediately; it has to wait its turn for a core again. This two-step (woken, then later scheduled) is a frequent source of confusion: "I notified the thread, why isn't it running?" Because notifying only makes it runnable; the scheduler still has to pick it.
- **running → terminated:** the thread's function returns.

The state that matters most for our story is **blocked**, because the cost of being blocked is exactly what divides OS threads from async and lightweight threads. Let us make that cost explicit.

| State | On a core? | Consumes CPU? | Consumes memory? | In a run queue? |
|---|---|---|---|---|
| New | No | No | Minimal | No |
| Runnable | No | No | Yes (stack reserved) | Yes |
| Running | Yes | Yes | Yes | No (it is running) |
| Blocked | No | No | Yes (**full stack still reserved**) | No (on a wait queue) |
| Terminated | No | No | No (stack reclaimed) | No |

Stare at the "Blocked" row. A blocked OS thread costs **zero CPU** — that is the good news, and it is why "just use blocking threads" works fine when you have a few hundred of them. But it costs a **full stack** of memory (~1 MB) the entire time it is parked. Ten thousand connections, each handled by a thread blocked on a slow client, is ten thousand reserved stacks doing nothing but waiting. That is the precise, mechanical reason the thread-per-connection model hits a memory wall — not CPU, *memory* — and the precise reason the lightweight-thread and async models exist.

## Why blocking a kernel thread is expensive — and what to do instead

Here is the crux, stated as plainly as I can. A kernel thread is a heavyweight object: ~1 MB of stack, a kernel data structure, a scheduler slot. When that thread **blocks** — say it calls `read()` on a socket and the data is not there yet — the thread parks, costs no CPU, but *holds onto all of that memory* until the data arrives. If your concurrency strategy is "one kernel thread per concurrent task," then your maximum concurrency is bounded by *memory*, not by useful work: 10,000 mostly-idle connections need 10,000 parked threads need ~10 GB of stacks, and almost all of those threads are doing nothing but waiting for slow network I/O.

This is the **C10k problem**, named in a famous 1999 essay by Dan Kegel: how do you handle ten thousand concurrent connections on one machine? The thread-per-connection answer does not survive it, for exactly the reason above — the per-thread cost is too high when most threads are idle. Two families of solutions emerged, and both are about *not tying up a kernel thread while you wait*.

**Family 1 — async / event-driven (non-blocking I/O + an event loop).** Instead of blocking a thread on each connection, use *non-blocking* I/O: ask the kernel "is this socket readable?" via a readiness API (`epoll`, `kqueue`, `io_uring`, IOCP), and a single thread (or a few — one per core) drives an **event loop** that services whichever connections are ready. Ten thousand idle connections cost ten thousand cheap kernel data structures in the `epoll` set, *not* ten thousand stacks. The thread is never blocked on any one connection; it is always working on a ready one. This is the nginx / Node.js / Netty / Go-runtime model. The cost is a programming-model shift — your code becomes callbacks or `async`/`await` rather than straight-line blocking calls — and the danger we already named: block the event loop with a CPU-bound call and you stall *every* connection. The full async story is its own track; this post just forward-links it.

**Family 2 — lightweight threads (M:N user threads).** Keep the blocking *programming model* — write straight-line code that "blocks" — but make the threads cheap and have the runtime turn each block into a yield. A goroutine's `conn.Read()` *looks* blocking, but the runtime parks the goroutine, registers the fd with `epoll`, and runs other goroutines on the same kernel thread; when the fd is ready, the goroutine is rescheduled. Ten thousand connections become ten thousand goroutines costing a few KB each — megabytes total, not gigabytes — multiplexed over a handful of kernel threads. You get the readability of blocking code with the scalability of an event loop. This is Go's whole pitch, and Java's Project Loom virtual threads bring the same to the JVM. The figure contrasts the two strategies for handling a flood of requests.

![Before and after comparison showing thread per request needing ten thousand OS threads and ten gigabytes of stack versus a bounded pool or virtual threads staying cheap](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-7.png)

The unifying insight: the problem was never *blocking* per se — a blocked task costs no CPU. The problem is **blocking a heavyweight kernel thread**, because the thread's memory and scheduler footprint are wasted while it waits. Async avoids it by never blocking a thread (the loop services ready work). Lightweight threads avoid it by making the thing that blocks cheap (a goroutine, not a kernel thread) and unmounting it from the kernel thread while it waits. Same disease, two cures. And there is a third, blunter cure that is right more often than people admit: a **bounded thread pool**, which we turn to in the code.

## Code: spawning threads, pooling them, and measuring the switch

Enough theory. Let us spawn threads in a few languages, see the thread-per-task bug at scale, fix it with a pool, and then *measure* the context-switch cost so the microseconds stop being hand-waving. The series is language-agnostic, so we show the idiom where it is clearest, in more than one language where they diverge.

### Spawning a kernel thread (the 1:1 model)

In **Rust**, `std::thread::spawn` creates one OS thread (1:1). The closure runs on the new thread; `join` waits for it.

```rust
use std::thread;

fn main() {
    let mut handles = Vec::new();
    for id in 0..4 {
        let handle = thread::spawn(move || {
            // Each thread runs on its own OS thread, its own stack.
            println!("hello from thread {id}");
        });
        handles.push(handle);
    }
    for h in handles {
        h.join().unwrap(); // wait for each thread to finish
    }
}
```

In **C with pthreads**, the same thing, closer to the metal. `pthread_create` is the system-call-backed creation; the stack size is tunable via the attribute.

```c
#include <pthread.h>
#include <stdio.h>

void *work(void *arg) {
    long id = (long)arg;
    printf("hello from thread %ld\n", id);
    return NULL;
}

int main(void) {
    pthread_t threads[4];
    for (long i = 0; i < 4; i++) {
        // Default stack ~1 MB; pthread_attr_setstacksize would shrink it.
        pthread_create(&threads[i], NULL, work, (void *)i);
    }
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    return 0;
}
```

In **Java**, classic platform threads are 1:1 with kernel threads; virtual threads (Java 21+) are M:N. Note how similar the call site is — and how different the cost is.

```java
public class Spawn {
    public static void main(String[] args) throws InterruptedException {
        // Platform thread: 1:1 with an OS thread, ~1 MB stack.
        Thread platform = new Thread(() -> System.out.println("platform thread"));
        platform.start();
        platform.join();

        // Virtual thread (Loom): M:N, a few KB, scheduled over carrier threads.
        Thread virtual = Thread.ofVirtual().start(() -> System.out.println("virtual thread"));
        virtual.join();
    }
}
```

In **Go**, the goroutine is the only spawn primitive and it is M:N by construction. There is no "OS thread" knob in the language at all.

```go
package main

import (
    "fmt"
    "sync"
)

func main() {
    var wg sync.WaitGroup
    for id := 0; id < 4; id++ {
        wg.Add(1)
        go func(id int) { // goroutine: ~2 KB stack, runtime-scheduled
            defer wg.Done()
            fmt.Println("hello from goroutine", id)
        }(id)
    }
    wg.Wait()
}
```

Four languages, four spawn idioms, but two cost models: Rust/C/Java-platform give you a kernel thread (~1 MB, syscall to create, kernel context switch); Go/Java-virtual give you a runtime thread (~2–4 KB, user-space create, user-space switch).

### The bug: thread-per-task at scale

Now reproduce the opening disaster in miniature. This Java program creates one platform thread per task. At a few tasks, fine. Crank `N` to 50,000 and watch it fall over — either `OutOfMemoryError: unable to create new native thread`, or the OS refusing past a thread-count limit, or the machine thrashing.

```java
import java.util.ArrayList;
import java.util.List;

public class ThreadPerTaskBug {
    public static void main(String[] args) throws InterruptedException {
        int N = 50_000;
        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            // BUG: one OS thread per task. ~1 MB stack each => ~50 GB of stacks.
            // You will hit OutOfMemoryError or the OS thread-count limit first.
            Thread t = new Thread(() -> {
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
            });
            t.start();
            threads.add(t);
        }
        for (Thread t : threads) t.join();
        System.out.println("done");
    }
}
```

The failure mode is exactly the lifecycle table's "Blocked" row at scale: 50,000 threads each in `sleep` (blocked), each holding a full stack, costing no CPU but exhausting memory and the OS thread limit.

### Fix A: a bounded thread pool

The first and most common fix is a **thread pool**: create a small, fixed set of worker threads (typically around the number of cores, or a few times that for I/O-bound work), and feed them tasks through a queue. The pool *decouples the number of tasks from the number of threads* — a million tasks, but only, say, 200 threads.

```java
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadPoolFix {
    public static void main(String[] args) throws InterruptedException {
        int N = 50_000;
        // FIX: a bounded pool. 200 threads handle all 50k tasks via a queue.
        ExecutorService pool = Executors.newFixedThreadPool(200);
        for (int i = 0; i < N; i++) {
            pool.submit(() -> {
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
            });
        }
        pool.shutdown();
        pool.awaitTermination(1, java.util.concurrent.TimeUnit.HOURS);
        System.out.println("done");
    }
}
```

Now memory is bounded (~200 stacks, not 50,000), creation cost is paid once, and the queue absorbs the backlog. The trade-off: if every task *blocks* for a long time (like the `sleep` here), 200 threads means at most 200 tasks run concurrently — the rest wait in the queue. For CPU-bound tasks that is exactly right (you cannot beat core count anyway). For I/O-bound tasks where you genuinely want thousands in flight at once, a pool of 200 is a bottleneck, and you want fix B.

### Fix B: lightweight threads (virtual threads / goroutines)

If the work is I/O-bound and you genuinely want tens of thousands concurrent, switch the *kind* of thread, not the count. Java's virtual threads make the original thread-per-task code *correct* at scale, because each "thread" now costs a few KB and unmounts from its carrier while blocked.

```java
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VirtualThreadFix {
    public static void main(String[] args) throws InterruptedException {
        int N = 50_000;
        // FIX: virtual threads. 50k of them cost ~MBs total, not ~50 GB.
        // The blocking sleep unmounts each virtual thread from its carrier.
        try (ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < N; i++) {
                pool.submit(() -> {
                    try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
                });
            }
        } // close() waits for all tasks
        System.out.println("done");
    }
}
```

The Go equivalent is even more direct — goroutines were lightweight from day one, so the "naive" code is already the scalable code:

```go
package main

import (
    "sync"
    "time"
)

func main() {
    var wg sync.WaitGroup
    for i := 0; i < 50_000; i++ {
        wg.Add(1)
        go func() { // 50k goroutines: a few hundred MB, scheduled over ~GOMAXPROCS kernel threads
            defer wg.Done()
            time.Sleep(time.Second) // parks the goroutine, not a kernel thread
        }()
    }
    wg.Wait()
}
```

That last contrast is the heart of the post. The *same* logical program — 50,000 tasks that each sleep a second — is a crash with platform threads and a non-event with virtual threads or goroutines, purely because the cost model of the thread changed from ~1 MB / kernel-scheduled to ~few KB / runtime-scheduled.

### Measuring the context-switch cost honestly

Now the measurement. We claimed ~1–5 µs for a kernel context switch. Here is how to actually measure it, and how to measure it *honestly*. The classic trick: two threads ping-pong a token back and forth through a pair of pipes (or channels). Each round-trip forces two context switches (A wakes B, B wakes A). Time a million round-trips, divide.

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

int p1[2], p2[2]; // two pipes for the ping-pong
const long ROUNDS = 1000000;

void *pong(void *arg) {
    char b;
    for (long i = 0; i < ROUNDS; i++) {
        read(p1[0], &b, 1);   // block until ping writes -> context switch
        write(p2[1], &b, 1);  // wake ping
    }
    return NULL;
}

int main(void) {
    pipe(p1);
    pipe(p2);
    pthread_t t;
    pthread_create(&t, NULL, pong, NULL);

    // Warm up: let the threads and caches settle before timing.
    char b = 0;
    for (long i = 0; i < 10000; i++) { write(p1[1], &b, 1); read(p2[0], &b, 1); }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (long i = 0; i < ROUNDS; i++) {
        write(p1[1], &b, 1);  // wake pong -> context switch
        read(p2[0], &b, 1);   // block until pong replies -> context switch
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    // Each round = 2 switches (wake pong, wake main).
    printf("avg context switch: %.0f ns\n", ns / (ROUNDS * 2));
    pthread_join(t, NULL);
    return 0;
}
```

A few honesty rules are baked into that code, and they matter for *every* concurrency measurement in this series:

- **Warm up.** The first iterations are slow — caches are cold, branch predictors untrained, the threads not yet co-scheduled. Run thousands of throwaway iterations before you start the clock, or your average is dominated by startup, not steady state.
- **Run many iterations.** A single context switch is too fast and too noisy to time directly. A million round-trips averages out the noise and the occasional unrelated interrupt.
- **Pin to a core if you want the floor.** If you `taskset -c 0` both threads onto one core, you measure the pure switch (no cross-core migration, no cache-coherence traffic) — typically the *lowest* number, often well under a microsecond for a same-process switch because no page-table swap happens. Let them run on different cores and you measure something closer to real cross-core wakeup latency, which is larger and includes inter-processor interrupt cost. *Say which one you did*, because they answer different questions.
- **Acknowledge nondeterminism.** Your number will vary run to run, machine to machine, kernel to kernel. Report a range, not a single magic figure.

On a typical modern Linux x86-64 box, this ping-pong reports somewhere in the **1–5 µs** range per switch for cross-thread wakeups through pipes (the syscall overhead dominates), dropping toward the sub-microsecond floor for same-core, same-process switches with the warmed path. The exact number is less important than the *order of magnitude* and the fact that you measured it rather than trusting a blog post (including this one). For the Python-specific version of these measurements — and why the GIL changes the threading story entirely — see [threading done right](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) and [the GIL explained](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs); for why true parallelism in Python often means processes, see [multiprocessing and the cost of pickling](/blog/software-development/python-performance/multiprocessing-true-parallelism-and-the-cost-of-pickling).

## Measured: the cost table that should drive your decisions

Let us collect the numbers into one place, with the honesty caveats attached. These are order-of-magnitude figures on commodity modern hardware (Linux, x86-64, 2020s-era CPUs); your machine will differ, and you should run the measurements above to get *your* numbers. The point is the *ratios* — they are stable across hardware even when the absolute numbers move.

![Cost matrix comparing OS thread and virtual thread across create cost and context switch and memory per thread and blocking call](/imgs/blogs/processes-threads-and-how-the-os-scheduler-runs-them-8.png)

| Cost dimension | OS (kernel) thread | Goroutine / virtual thread |
|---|---|---|
| Create | ~10–50 µs (syscall into kernel) | ~1 µs (runtime allocation) |
| Context switch | ~1–5 µs (kernel, page tables, TLB) | ~0.1–0.2 µs (user space, same address space) |
| Memory per thread | ~1–8 MB stack (fixed, reserved) | ~2–4 KB stack (growable on demand) |
| Max practical count | thousands to low tens of thousands | millions |
| Blocking a thread | parks a whole kernel thread (memory wasted while idle) | unmounts from carrier; carrier runs others |

Read the ratios. Creation is ~10–50× cheaper for a goroutine. Switching is ~10–25× cheaper. Memory is ~250–1000× cheaper. The practical ceiling on count is ~100–1000× higher. None of these are small differences; together they are why "10,000 concurrent connections" is a non-event for one model and a crash for the other.

One more honest measurement worth doing yourself: **threads versus memory.** Write a loop that spawns OS threads, each parking on a condition, and watch resident memory climb in `top` or `/proc/self/status`. You will see roughly the stack-size-per-thread slope — create 1,000 threads and watch RSS climb by hundreds of MB to a GB or so as stacks get touched. Do the same with goroutines and the slope is two to three orders of magnitude flatter. That single experiment, run on your own hardware, teaches the lesson more durably than any table: *the OS thread is a megabyte-class object, and you cannot have ten thousand of them for free.*

#### Worked example: sizing a connection server

You are building a service expecting 20,000 concurrent long-lived connections (think WebSocket clients), each mostly idle, occasionally sending a small message. Three designs:

1. **Thread per connection (OS threads):** 20,000 × ~1 MB = ~20 GB of stacks. On a 32 GB box this is a non-starter once you add heap and buffers; the scheduler is also juggling 20,000 mostly-blocked threads. *Rejected by the memory math alone.*
2. **Bounded thread pool (say 64 threads) + non-blocking I/O:** memory is trivial (64 stacks), but now you are hand-writing an event loop or a reactor — your "blocking" reads become `epoll` registrations and callbacks. It works (this is the nginx/Netty model) but the code is harder. Right choice if you need maximum efficiency and your team is comfortable with async.
3. **Virtual threads / goroutines (20,000 of them):** 20,000 × ~4 KB ≈ 80 MB of stacks — trivial — scheduled over ~32 carrier kernel threads, each connection written as straight-line blocking code. You get the simplicity of design 1 with the scalability of design 2. *Right choice for most teams today.*

The decision is not aesthetic; it is the cost table. Twenty thousand of a megabyte-class object is impossible; twenty thousand of a kilobyte-class object is nothing. Choose the thread whose cost model fits your fan-out.

## Case studies / real-world

**The C10k problem (Dan Kegel, 1999).** Kegel's essay crystallized a generation's worth of server design by asking a simple question: why can't one machine handle ten thousand simultaneous connections? The answer was precisely the thread cost model — thread-per-connection servers of the late 1990s drowned in per-thread memory and scheduler overhead well before the network or CPU was saturated. The essay catalogued the non-blocking-I/O alternatives (`select`, `poll`, and the then-new `epoll`/`kqueue`) that let one thread service thousands of sockets. Modern servers — nginx, Node.js, Netty, and the runtimes under Go and Java — are all, in one way or another, answers to C10k. The lesson generalizes: when a per-task resource (a thread, a stack, a connection slot) is expensive, your scaling limit is that resource, not the work. (Source: Dan Kegel, "The C10K problem," 1999, kegel.com/c10k.html.)

**Go goroutines and the growable stack.** Go's runtime team made the deliberate bet that *cheap threads* would beat *async callbacks* for programmer productivity, and the enabling mechanism was the segmented-then-contiguous growable stack. Early Go (pre-1.3) used segmented stacks (link a new segment when you run out), which caused a notorious "hot split" performance cliff when a tight loop straddled a segment boundary; Go 1.3 switched to *copying* growable stacks (allocate a bigger stack, copy frames, fix pointers) which removed the cliff. Goroutines start at ~2 KB and grow as needed, which is why a Go server can hold hundreds of thousands of goroutines where the same machine could hold only thousands of OS threads. (Source: the Go blog and runtime design docs on contiguous stacks, 2014.)

**Java Project Loom virtual threads (Java 21, 2023).** Loom brought M:N lightweight threads to the JVM after years of development, with the explicit goal of letting servers written in the simple thread-per-request style scale to millions of concurrent tasks. A virtual thread mounts on a platform "carrier" thread to run, and *unmounts* when it blocks on most blocking operations, freeing the carrier to run another virtual thread. The headline demonstration — spawning a million virtual threads that each sleep, on a machine that could never spawn a million platform threads — made the cost-model point viscerally. The remaining sharp edge is "pinning": a virtual thread that blocks inside a `synchronized` block or a native call cannot unmount and *does* tie up its carrier (largely mitigated in later JDKs) — a reminder that the M:N model has a seam where it meets the kernel. (Source: JEP 444, "Virtual Threads," and the Loom design documents.)

**Mars Pathfinder and priority inversion (1997).** A scheduler case study with teeth: the Pathfinder lander on Mars began resetting itself repeatedly. The cause was *priority inversion* — a high-priority bus-management thread blocked waiting for a mutex held by a low-priority meteorological thread, while a medium-priority thread kept preempting the low-priority one so it never finished and never released the lock. A watchdog timer noticed the high-priority task wasn't completing and reset the system. The fix, uploaded to Mars, was to enable *priority inheritance* on the mutex (temporarily boost the lock-holder's priority to the waiter's). The lesson sits squarely in this post's scheduler section: priorities are not just a "go faster" knob; they interact with locks in ways that can deadlock progress entirely. (Source: Glenn Reeves / JPL, "What really happened on Mars," 1997.)

## When to reach for this (and when not to)

The thread/process/scheduler machinery is the substrate under *everything* concurrent, but the practical decision you keep making is **which kind of execution unit to spawn**. Here is the decisive guidance.

**Reach for OS threads (1:1) when:**

- You need genuine CPU parallelism for compute-bound work and the count of parallel tasks is small — on the order of the core count, not thousands. A thread pool sized to cores is the textbook right answer for a CPU-bound workload; more threads than cores only adds switching overhead without adding throughput.
- You need fault isolation at the *process* level (use processes, not threads) — a crash in one worker must not take down the others. Browsers, databases, and supervised systems do this deliberately.
- You are calling into libraries or syscalls that genuinely block at the kernel and have no async variant, and the concurrency is modest. A few hundred blocked OS threads is completely fine; it is *thousands* that hurt.

**Reach for lightweight threads (goroutines / virtual threads, M:N) when:**

- You have many concurrent tasks that are mostly *I/O-bound* — connection servers, request handlers, fan-out to many downstream services — and you want straight-line, blocking-style code without the memory wall. This is the sweet spot, and for most server workloads today it is simply the default.
- You want tens of thousands to millions of concurrent tasks. OS threads cannot; lightweight threads can.

**Reach for async / event loops when:**

- You are in a runtime where it is the idiomatic concurrency model (Node.js, Python with [asyncio](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines), Rust with `tokio`) and the work is I/O-bound. You get C10k-class scalability with one or a few threads — at the cost of the `async` programming model and the strict rule never to block the loop.

**Do NOT:**

- **Do not spawn one OS thread per task at scale.** This is the opening server's bug. Past a few hundred to a few thousand, you hit the memory and scheduler wall. Use a pool or lightweight threads.
- **Do not add OS threads to speed up CPU-bound work beyond core count.** Eight cores means at most eight threads doing useful compute simultaneously; the ninth thread just adds context-switch overhead. More threads ≠ more speed; this is the Amdahl-and-overhead reality covered in [concurrency vs parallelism and the scaling laws](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws).
- **Do not add threads to an I/O-bound task expecting throughput.** A thread blocked on I/O is not doing CPU work; you do not need more *threads*, you need more *outstanding I/O* — which async or lightweight threads give you without the per-thread memory cost.
- **Do not block the event loop / pin a carrier with a long CPU-bound call.** In a cooperative model, one non-yielding task starves everyone. Offload CPU-bound work to a separate pool.
- **Do not micro-optimize context switches before you have measured that switching is your bottleneck.** Most services are bound by I/O latency or lock contention, not raw switch cost. Measure first.

The meta-rule, the one the [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) builds on: match the *cost model of your execution unit* to the *shape and scale of your concurrency*. Few, compute-heavy tasks → OS threads. Many, I/O-heavy tasks → lightweight threads or async. Need isolation → processes. The thread is not free; pick the one whose price you can afford at your scale.

## Key takeaways

1. **A process is isolation; a thread is execution.** A process has its own address space (hardware-enforced via page tables); threads in a process share the heap, code, and file descriptors but keep private stacks, registers, and thread-local storage. Sharing is the default *within* a process and the exception *between* processes.
2. **What is shared can race; what is private cannot.** The shared heap is the only thing two threads can corrupt — which is why the entire apparatus of locks, atomics, and the memory model exists. Private stacks never interfere.
3. **One core runs one thread at a time.** Concurrency on a single core is time-slicing, not simultaneity. The scheduler gives each runnable thread a quantum (a few ms), then a context switch hands the core to the next.
4. **Kernel scheduling is preemptive; the timer interrupt is the leash.** A thread cannot monopolize a core; the hardware fires a timer, the kernel regains control, and the scheduler (CFS/EEVDF, fairness-by-virtual-runtime) picks the most-deserving thread. Cooperative scheduling is simpler but one non-yielding task freezes everyone.
5. **A context switch costs ~1–5 µs directly, plus cache and TLB pollution indirectly.** It saves registers, swaps the stack pointer, switches page tables (only across processes), and flushes/pressures the TLB. The indirect cache-cold tax often exceeds the direct cost.
6. **A blocked OS thread costs no CPU but a full stack of memory.** That memory tax — ~1 MB per parked thread — is precisely why thread-per-connection servers hit a memory wall, not a CPU wall, at ten thousand connections (the C10k problem).
7. **The M:N model makes lightweight threads cheap.** Goroutines and virtual threads multiplex many tiny, growable-stack user threads over a small pool of kernel threads, switch in user space (~0.1–0.2 µs), and unmount from their carrier when they block — giving blocking-style code event-loop scalability.
8. **Match the execution unit to the workload.** Few CPU-bound tasks → OS threads sized to cores. Many I/O-bound tasks → lightweight threads or async. Need fault isolation → processes. The thread is a cost; pick the one you can afford at your scale, and measure before you optimize.

## Further reading

- **Andrew Tanenbaum & Herbert Bos, *Modern Operating Systems*** — the canonical treatment of processes, threads, scheduling, and context switches; the chapters on processes/threads and scheduling are the deep version of this post.
- **Remzi & Andrea Arpaci-Dusseau, *Operating Systems: Three Easy Pieces* (free online)** — the clearest modern explanation of the process model, the scheduler, and virtual address spaces; the "Concurrency" pieces follow directly.
- **Dan Kegel, "The C10K problem" (1999, kegel.com/c10k.html)** — the essay that defined server scalability and motivated non-blocking I/O over thread-per-connection.
- **JEP 444, "Virtual Threads" (OpenJDK)** — the design and rationale for Java's M:N lightweight threads, with the carrier/mount/unmount mechanism spelled out.
- **The Go scheduler design docs and "Scalable Go Scheduler Design Doc" (Dmitry Vyukov)** — how Go's M:N runtime, work-stealing, and growable stacks actually work.
- **Robert Love, *Linux Kernel Development*** — the real CFS/run-queue/`task_struct` machinery if you want the kernel-level view of how threads get scheduled.
- **[Why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it)** — the series opener and the shared-state / happens-before frame this post sits inside.
- **[The concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model)** — the capstone that turns "which execution unit?" into a full decision framework across the whole series.
